import torch
import torch.nn as nn
import torch.nn.functional as F
from compressai.layers import AttentionBlock


class FusionBlock(nn.Module):
    """Attention-guided fusion from the TextBoost paper.

    Decoder RGB is expanded 3 -> 13 and concatenated with the 3-channel
    Hadamard product of the guidance map and the decoder RGB. A Cheng-2020
    attention block refines the 16-channel tensor, and a 1x1 convolution
    projects it back to RGB.
    """

    def __init__(self):
        super().__init__()
        self.expand = nn.Conv2d(3, 13, kernel_size=1)
        self.attn = AttentionBlock(16)
        self.proj = nn.Conv2d(16, 3, kernel_size=1)

    def forward(self, x_base, guidance):
        if guidance.shape[-2:] != x_base.shape[-2:]:
            guidance = F.interpolate(
                guidance, size=x_base.shape[-2:], mode="bilinear", align_corners=False
            )
        if guidance.shape[1] == 1:
            guidance = guidance.repeat(1, 3, 1, 1)
        modulated = guidance * x_base
        feat = torch.cat([self.expand(x_base), modulated], dim=1)
        return self.proj(self.attn(feat))
