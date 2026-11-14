import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def bitrate(output, num_pixels):
    return sum(
        torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)
        for likelihoods in output["likelihoods"].values()
    )


class Stage1RateDistortion(nn.Module):
    """Same Lagrangian as JiangWeibeta/ELIC: bpp + lambda * 255^2 * MSE."""

    def __init__(self, lmbda):
        super().__init__()
        self.lmbda = float(lmbda)

    def forward(self, output, target):
        n, _, h, w = target.shape
        num_pixels = n * h * w
        bpp = bitrate(output, num_pixels)
        mse = F.mse_loss(output["x_hat"], target)
        loss = bpp + self.lmbda * (255.0 ** 2) * mse
        return {"loss": loss, "bpp_loss": bpp, "mse_loss": mse}


class Stage2GuidanceLoss(nn.Module):
    """Frozen-rate objective.

    L = stopgrad(bpp) + lambda * 255^2 * (MSE + alpha * L_gc)
    L_gc = MSE(m * x, m * x_hat)
    The 255^2 factor is the ELIC codebase's definition of lambda, so stage 2
    keeps the same gradient scale as stage 1.
    """

    def __init__(self, lmbda, alpha=10.0):
        super().__init__()
        self.lmbda = float(lmbda)
        self.alpha = float(alpha)

    def forward(self, output, target, mask):
        n, _, h, w = target.shape
        num_pixels = n * h * w
        bpp = bitrate(output, num_pixels)
        mse = F.mse_loss(output["x_hat"], target)
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        l_gc = F.mse_loss(mask * target, mask * output["x_hat"])
        loss = bpp.detach() + self.lmbda * (255.0 ** 2) * (mse + self.alpha * l_gc)
        return {
            "loss": loss,
            "bpp_loss": bpp.detach(),
            "mse_loss": mse,
            "gc_loss": l_gc,
        }
