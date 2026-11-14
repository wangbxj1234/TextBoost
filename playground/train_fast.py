"""Eight-GPU training until the loss plateaus.

Stage 1 trains the whole network. Stage 2 freezes the backbone and trains the
fusion block. The run stops when lowering the learning rate no longer improves
the loss, instead of running a fixed epoch count.
"""

import argparse
import logging
import os
import sys
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config.config import model_config
from textboost.data import CroppedImageFolder, TextOCRCrops
from textboost.loss import Stage1RateDistortion, Stage2GuidanceLoss
from textboost.model import TextBoostELIC, freeze_backbone


def parse_args():
    p = argparse.ArgumentParser(description="8-GPU TextBoost training until plateau")
    p.add_argument("--dataset", required=True)
    p.add_argument("--image-root", required=True)
    p.add_argument("--ann", required=True)
    p.add_argument("--init-checkpoint", required=True)
    p.add_argument("--save-dir", required=True)
    p.add_argument("--lambda", dest="lmbda", type=float, required=True)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--patch-size", type=int, default=256)
    p.add_argument("--learning-rate", type=float, default=8e-4)
    p.add_argument("--aux-learning-rate", type=float, default=1e-3)
    p.add_argument("--stage2-learning-rate", type=float, default=2e-3)
    p.add_argument("--alpha", type=float, default=10.0)
    p.add_argument("--area-threshold", type=float, default=150.0)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--clip-max-norm", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=192)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--min-delta", type=float, default=0.005)
    p.add_argument("--stage1-max-epochs", type=int, default=80)
    p.add_argument("--stage2-max-epochs", type=int, default=30)
    p.add_argument("--lr-floor", type=float, default=1e-4)
    p.add_argument("--stage2-lr-floor", type=float, default=2.5e-4)
    return p.parse_args()


def setup():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        device_id=torch.device("cuda", local_rank),
    )
    return local_rank, dist.get_rank(), dist.get_world_size()


def make_logger(path, rank):
    log = logging.getLogger("fast")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    log.propagate = False
    if rank == 0:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fmt = logging.Formatter("%(asctime)s %(message)s")
        file_handler = logging.FileHandler(path)
        stream_handler = logging.StreamHandler(sys.stdout)
        file_handler.setFormatter(fmt)
        stream_handler.setFormatter(fmt)
        log.addHandler(file_handler)
        log.addHandler(stream_handler)
    else:
        log.addHandler(logging.NullHandler())
    return log


def split_params(net, lr, aux_lr):
    main, aux = [], []
    for name, param in net.named_parameters():
        if not param.requires_grad:
            continue
        if name.endswith(".quantiles"):
            aux.append(param)
        else:
            main.append(param)
    optimizer = torch.optim.Adam(main, lr=lr) if main else None
    aux_optimizer = torch.optim.Adam(aux, lr=aux_lr) if aux else None
    return optimizer, aux_optimizer


def atomic_save(payload, path):
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_weights(net, path, device):
    ckpt = torch.load(path, map_location=device)
    net.load_state_dict(ckpt["state_dict"])
    return ckpt


def reduce_mean(total, count, device):
    stats = torch.tensor([total, count], device=device, dtype=torch.float64)
    dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    return (stats[0] / stats[1]).item()


def run_stage1(args, raw, device, rank, log):
    save_dir = os.path.join(args.save_dir, "stage1")
    if rank == 0:
        os.makedirs(save_dir, exist_ok=True)
    dist.barrier()
    last_path = os.path.join(save_dir, "checkpoint_last.pth.tar")
    best_path = os.path.join(save_dir, "checkpoint_best.pth.tar")
    start_epoch = 0
    lr = args.learning_rate
    best = float("inf")
    patience = 0
    resumed = False
    if os.path.isfile(last_path):
        ckpt = torch.load(last_path, map_location=device)
        if ckpt.get("fast"):
            raw.load_state_dict(ckpt["state_dict"])
            start_epoch = int(ckpt.get("epoch", 0))
            lr = float(ckpt.get("lr", lr))
            best = float(ckpt.get("best", best))
            patience = int(ckpt.get("patience", 0))
            resumed = True
    if not resumed:
        load_weights(raw, args.init_checkpoint, device)
    net = DDP(raw, device_ids=[device.index], output_device=device.index)
    optimizer, aux_optimizer = split_params(net, lr, args.aux_learning_rate)
    if resumed and optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
        for group in optimizer.param_groups:
            group["lr"] = lr
    if resumed and aux_optimizer is not None and ckpt.get("aux_optimizer"):
        aux_optimizer.load_state_dict(ckpt["aux_optimizer"])
    criterion = Stage1RateDistortion(args.lmbda)
    dataset = CroppedImageFolder(args.dataset, patch=args.patch_size)
    sampler = DistributedSampler(dataset, shuffle=True, drop_last=True)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    log.info(
        "stage1 images=%d world=%d per_gpu_batch=%d lr=%g lambda=%s start_epoch=%d",
        len(dataset),
        dist.get_world_size(),
        args.batch_size,
        lr,
        args.lmbda,
        start_epoch,
    )
    main_params = [p for n, p in net.named_parameters() if p.requires_grad and not n.endswith(".quantiles")]
    epoch = start_epoch
    while epoch < args.stage1_max_epochs:
        sampler.set_epoch(epoch)
        net.train()
        loss_sum = 0.0
        mse_sum = 0.0
        bpp_sum = 0.0
        aux_sum = 0.0
        steps = 0
        t0 = time.time()
        for step, batch in enumerate(loader):
            batch = batch.to(device, non_blocking=True)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            if aux_optimizer is not None:
                aux_optimizer.zero_grad(set_to_none=True)
            out = net(batch)
            losses = criterion(out, batch)
            aux_loss = net.module.aux_loss()
            (losses["loss"] + aux_loss).backward()
            if args.clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(main_params, args.clip_max_norm)
            if optimizer is not None:
                optimizer.step()
            if aux_optimizer is not None:
                aux_optimizer.step()
            loss_sum += float(losses["loss"].detach())
            mse_sum += float(losses["mse_loss"].detach())
            bpp_sum += float(losses["bpp_loss"].detach())
            aux_sum += float(aux_loss.detach())
            steps += 1
            if step % 20 == 0:
                log.info(
                    "epoch %d step %d/%d loss %.4f mse %.6f bpp %.4f aux %.4f lr %g",
                    epoch,
                    step,
                    len(loader),
                    float(losses["loss"].detach()),
                    float(losses["mse_loss"].detach()),
                    float(losses["bpp_loss"].detach()),
                    float(aux_loss.detach()),
                    lr,
                )
        mean_loss = reduce_mean(loss_sum, steps, device)
        mean_mse = reduce_mean(mse_sum, steps, device)
        mean_bpp = reduce_mean(bpp_sum, steps, device)
        mean_aux = reduce_mean(aux_sum, steps, device)
        improved = mean_loss < best * (1.0 - args.min_delta)
        if improved:
            best = mean_loss
            patience = 0
        else:
            patience += 1
        payload = {
            "fast": True,
            "epoch": epoch + 1,
            "state_dict": net.module.state_dict(),
            "optimizer": None if optimizer is None else optimizer.state_dict(),
            "aux_optimizer": None if aux_optimizer is None else aux_optimizer.state_dict(),
            "lr": lr,
            "best": best,
            "patience": patience,
            "lmbda": args.lmbda,
        }
        if rank == 0:
            atomic_save(payload, last_path)
            if improved:
                atomic_save(payload, best_path)
        stop = False
        if patience >= args.patience:
            if lr <= args.lr_floor * (1.0 + 1e-8):
                stop = True
            else:
                lr = max(lr * 0.5, args.lr_floor)
                patience = 0
                best = mean_loss
                if optimizer is not None:
                    for group in optimizer.param_groups:
                        group["lr"] = lr
        if rank == 0 and patience == 0 and not improved:
            payload["lr"] = lr
            payload["best"] = best
            payload["patience"] = patience
            atomic_save(payload, last_path)
        log.info(
            "epoch %d mean loss %.4f mse %.6f bpp %.4f aux %.4f best %.4f patience %d lr %g sec %.1f%s",
            epoch,
            mean_loss,
            mean_mse,
            mean_bpp,
            mean_aux,
            best,
            patience,
            lr,
            time.time() - t0,
            " improved" if improved else "",
        )
        epoch += 1
        if stop:
            log.info("stage1 converged at lr floor")
            break
    else:
        log.info("stage1 reached max epochs")
    del loader
    del optimizer
    del aux_optimizer
    del net
    torch.cuda.empty_cache()
    dist.barrier()
    chosen = best_path if os.path.isfile(best_path) else last_path
    return chosen


def run_stage2(args, device, rank, log, stage1_ckpt):
    save_dir = os.path.join(args.save_dir, "stage2")
    if rank == 0:
        os.makedirs(save_dir, exist_ok=True)
    dist.barrier()
    raw = TextBoostELIC(model_config()).to(device)
    last_path = os.path.join(save_dir, "checkpoint_last.pth.tar")
    best_path = os.path.join(save_dir, "checkpoint_best.pth.tar")
    start_epoch = 0
    lr = args.stage2_learning_rate
    best = float("inf")
    patience = 0
    resumed = False
    if os.path.isfile(last_path):
        ckpt = torch.load(last_path, map_location=device)
        if ckpt.get("fast"):
            raw.load_state_dict(ckpt["state_dict"])
            start_epoch = int(ckpt.get("epoch", 0))
            lr = float(ckpt.get("lr", lr))
            best = float(ckpt.get("best", best))
            patience = int(ckpt.get("patience", 0))
            resumed = True
    if not resumed:
        load_weights(raw, stage1_ckpt, device)
    freeze_backbone(raw)
    net = DDP(raw, device_ids=[device.index], output_device=device.index, find_unused_parameters=True)
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=lr)
    if resumed and "optimizer" in ckpt and ckpt["optimizer"] is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
        for group in optimizer.param_groups:
            group["lr"] = lr
    criterion = Stage2GuidanceLoss(args.lmbda, alpha=args.alpha)
    dataset = TextOCRCrops(
        args.image_root,
        args.ann,
        patch=args.patch_size,
        area_threshold=args.area_threshold,
    )
    sampler = DistributedSampler(dataset, shuffle=True, drop_last=True)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    log.info(
        "stage2 images=%d lr=%g alpha=%s trainable=%d start_epoch=%d",
        len(dataset),
        lr,
        args.alpha,
        sum(p.numel() for p in params),
        start_epoch,
    )
    epoch = start_epoch
    while epoch < args.stage2_max_epochs:
        sampler.set_epoch(epoch)
        net.train()
        loss_sum = 0.0
        gc_sum = 0.0
        steps = 0
        t0 = time.time()
        for step, batch in enumerate(loader):
            image = batch["image"].to(device, non_blocking=True)
            guidance = batch["guidance"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            out = net(image, guidance=guidance, detach_base=True)
            losses = criterion(out, image, mask)
            losses["loss"].backward()
            if args.clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(params, args.clip_max_norm)
            optimizer.step()
            loss_sum += float(losses["loss"].detach())
            gc_sum += float(losses["gc_loss"].detach())
            steps += 1
            if step % 20 == 0:
                log.info(
                    "stage2 epoch %d step %d/%d loss %.4f mse %.6f gc %.6f bpp %.4f lr %g",
                    epoch,
                    step,
                    len(loader),
                    float(losses["loss"].detach()),
                    float(losses["mse_loss"].detach()),
                    float(losses["gc_loss"].detach()),
                    float(losses["bpp_loss"].detach()),
                    lr,
                )
        mean_loss = reduce_mean(loss_sum, steps, device)
        mean_gc = reduce_mean(gc_sum, steps, device)
        improved = mean_loss < best * (1.0 - args.min_delta)
        if improved:
            best = mean_loss
            patience = 0
        else:
            patience += 1
        payload = {
            "fast": True,
            "epoch": epoch + 1,
            "state_dict": net.module.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr": lr,
            "best": best,
            "patience": patience,
            "lmbda": args.lmbda,
            "alpha": args.alpha,
        }
        if rank == 0:
            atomic_save(payload, last_path)
            if improved:
                atomic_save(payload, best_path)
        stop = False
        if patience >= args.patience:
            if lr <= args.stage2_lr_floor * (1.0 + 1e-8):
                stop = True
            else:
                lr = max(lr * 0.5, args.stage2_lr_floor)
                patience = 0
                best = mean_loss
                for group in optimizer.param_groups:
                    group["lr"] = lr
        log.info(
            "stage2 epoch %d mean loss %.4f gc %.6f best %.4f patience %d lr %g sec %.1f%s",
            epoch,
            mean_loss,
            mean_gc,
            best,
            patience,
            lr,
            time.time() - t0,
            " improved" if improved else "",
        )
        epoch += 1
        if stop:
            log.info("stage2 converged at lr floor")
            break
    else:
        log.info("stage2 reached max epochs")
    dist.barrier()


def main():
    args = parse_args()
    local_rank, rank, world = setup()
    torch.manual_seed(args.seed + rank)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda", local_rank)
    log = make_logger(os.path.join(args.save_dir, "train_fast.log"), rank)
    log.info("boot rank %d/%d gpu %d", rank, world, local_rank)
    raw = TextBoostELIC(model_config()).to(device)
    stage1_ckpt = run_stage1(args, raw, device, rank, log)
    del raw
    torch.cuda.empty_cache()
    run_stage2(args, device, rank, log, stage1_ckpt)
    log.info("done")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
