"""Stage 2: freeze ELIC and train the fusion block on TextOCR.

Learning rate 1e-3, 20 epochs, alpha 10, T_train 150, crop 256.
The bitrate term is detached. Gradients do not enter the decoder base.
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
from textboost.data import TextOCRCrops
from textboost.loss import Stage2GuidanceLoss
from textboost.model import TextBoostELIC, freeze_backbone


def parse_args():
    p = argparse.ArgumentParser(description="TextBoost stage-2 fusion fine-tuning")
    p.add_argument("--image-root", required=True)
    p.add_argument("--ann", required=True, help="TextOCR train json")
    p.add_argument("--checkpoint", required=True, help="Stage-1 checkpoint")
    p.add_argument("--experiment", required=True)
    p.add_argument("--save-dir", required=True)
    p.add_argument("--lambda", dest="lmbda", type=float, required=True)
    p.add_argument("--alpha", type=float, default=10.0)
    p.add_argument("--area-threshold", type=float, default=150.0)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--patch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--clip-max-norm", type=float, default=1.0)
    p.add_argument("--font", default=None)
    p.add_argument("--seed", type=float, default=192.1)
    p.add_argument("--resume", default=None, help="Stage-2 checkpoint to continue")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    os.makedirs(args.save_dir, exist_ok=True)
    log_path = os.path.join(args.save_dir, "train_stage2.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)],
    )
    log = logging.getLogger("stage2")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = TextOCRCrops(
        args.image_root,
        args.ann,
        patch=args.patch_size,
        area_threshold=args.area_threshold,
        font_path=args.font,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        drop_last=True,
    )
    net = TextBoostELIC(model_config()).to(device)
    start_epoch = 0
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        net.load_state_dict(ckpt["state_dict"])
        start_epoch = int(ckpt.get("epoch", 0))
        log.info("resumed stage2 %s at epoch %d", args.resume, start_epoch)
    else:
        ckpt = torch.load(args.checkpoint, map_location=device)
        net.load_state_dict(ckpt["state_dict"])
    freeze_backbone(net)
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)
    if args.resume and os.path.isfile(args.resume) and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    criterion = Stage2GuidanceLoss(args.lmbda, alpha=args.alpha)
    log.info(
        "stage2 images=%d lambda=%s alpha=%s lr=%s epochs=%d trainable=%d",
        len(dataset),
        args.lmbda,
        args.alpha,
        args.learning_rate,
        args.epochs,
        sum(p.numel() for p in params),
    )
    for epoch in range(start_epoch, args.epochs):
        net.train()
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
            if step % 50 == 0:
                log.info(
                    "epoch %d step %d/%d loss %.4f mse %.6f gc %.6f bpp %.4f",
                    epoch,
                    step,
                    len(loader),
                    losses["loss"].item(),
                    losses["mse_loss"].item(),
                    losses["gc_loss"].item(),
                    float(losses["bpp_loss"].detach()),
                )
        payload = {
            "epoch": epoch + 1,
            "state_dict": net.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lmbda": args.lmbda,
            "alpha": args.alpha,
            "experiment": args.experiment,
        }
        torch.save(payload, os.path.join(args.save_dir, "checkpoint_last.pth.tar"))
        log.info("saved stage2 epoch %d", epoch + 1)


if __name__ == "__main__":
    main()
