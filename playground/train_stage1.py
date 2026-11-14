"""Stage 1: ELIC rate-distortion training with a zero guidance map.

The fusion block is in the graph and is trained. Guidance is a zero tensor,
which is the ImageNet / no-OCR regime in the paper.
"""

import argparse
import logging
import os
import sys

import torch
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config.config import model_config
from textboost.data import CroppedImageFolder
from textboost.loss import Stage1RateDistortion
from textboost.model import TextBoostELIC


def parse_args():
    p = argparse.ArgumentParser(description="TextBoost stage-1 rate-distortion training")
    p.add_argument("--dataset", required=True, help="Directory of training images")
    p.add_argument("--experiment", required=True)
    p.add_argument("--save-dir", required=True)
    p.add_argument("--lambda", dest="lmbda", type=float, required=True)
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--aux-learning-rate", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--patch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--clip-max-norm", type=float, default=1.0)
    p.add_argument("--seed", type=float, default=192.1)
    p.add_argument("--checkpoint", default=None)
    return p.parse_args()


def configure_optimizers(net, lr, aux_lr):
    params, aux = [], []
    for name, param in net.named_parameters():
        if not param.requires_grad:
            continue
        if name.endswith(".quantiles"):
            aux.append(param)
        else:
            params.append(param)
    return torch.optim.Adam(params, lr=lr), torch.optim.Adam(aux, lr=aux_lr)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    os.makedirs(args.save_dir, exist_ok=True)
    log_path = os.path.join(args.save_dir, "train_stage1.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)],
    )
    log = logging.getLogger("stage1")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = CroppedImageFolder(args.dataset, patch=args.patch_size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        drop_last=True,
    )
    net = TextBoostELIC(model_config()).to(device)
    optimizer, aux_optimizer = configure_optimizers(net, args.learning_rate, args.aux_learning_rate)
    criterion = Stage1RateDistortion(args.lmbda)
    start_epoch = 0
    if args.checkpoint and os.path.isfile(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device)
        net.load_state_dict(ckpt["state_dict"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "aux_optimizer" in ckpt:
            aux_optimizer.load_state_dict(ckpt["aux_optimizer"])
        start_epoch = int(ckpt.get("epoch", 0))
        log.info("resumed %s at epoch %d", args.checkpoint, start_epoch)

    log.info(
        "stage1 experiment=%s images=%d lambda=%s batch=%d lr=%s epochs=%d",
        args.experiment,
        len(dataset),
        args.lmbda,
        args.batch_size,
        args.learning_rate,
        args.epochs,
    )
    for epoch in range(start_epoch, args.epochs):
        net.train()
        for step, batch in enumerate(loader):
            batch = batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            aux_optimizer.zero_grad(set_to_none=True)
            out = net(batch)
            losses = criterion(out, batch)
            losses["loss"].backward()
            if args.clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip_max_norm)
            optimizer.step()
            aux_loss = net.aux_loss()
            aux_loss.backward()
            aux_optimizer.step()
            if step % 50 == 0:
                log.info(
                    "epoch %d step %d/%d loss %.4f mse %.6f bpp %.4f aux %.4f",
                    epoch,
                    step,
                    len(loader),
                    losses["loss"].item(),
                    losses["mse_loss"].item(),
                    losses["bpp_loss"].item(),
                    float(aux_loss.detach()),
                )
        payload = {
            "epoch": epoch + 1,
            "state_dict": net.state_dict(),
            "optimizer": optimizer.state_dict(),
            "aux_optimizer": aux_optimizer.state_dict(),
            "lmbda": args.lmbda,
            "experiment": args.experiment,
        }
        last_path = os.path.join(args.save_dir, "checkpoint_last.pth.tar")
        torch.save(payload, last_path)
        if (epoch + 1) % 20 == 0 or epoch + 1 == args.epochs:
            torch.save(payload, os.path.join(args.save_dir, "checkpoint_%03d.pth.tar" % (epoch + 1)))
        log.info("saved epoch %d", epoch + 1)


if __name__ == "__main__":
    main()
