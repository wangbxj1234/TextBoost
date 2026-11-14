import torch.nn as nn

from models.elic import ELIC
from textboost.fusion import FusionBlock
from utils.func import update_registered_buffers


class TextBoostELIC(ELIC):
    """ELIC with the TextBoost fusion block on the decoder RGB.

    The encoder, hyperprior, entropy model, and synthesis transform are the
    open-source ELIC modules. Fusion is the only new stage.
    """

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.fusion = FusionBlock()

    def forward(self, x, guidance=None, detach_base=False):
        out = super().forward(x)
        x_base = out["x_hat"]
        if detach_base:
            x_base = x_base.detach()
        if guidance is None:
            guidance = x_base.new_zeros(x_base.shape)
        out["x_base"] = x_base
        out["x_hat"] = self.fusion(x_base, guidance)
        return out

    def decompress(self, strings, shape, guidance=None):
        out = super().decompress(strings, shape)
        x_base = out["x_hat"]
        if guidance is None:
            guidance = x_base.new_zeros(x_base.shape)
        out["x_base"] = x_base
        out["x_hat"] = self.fusion(x_base, guidance)
        return out

    def load_state_dict(self, state_dict, strict=True):
        if any(k.startswith("module.") for k in state_dict):
            state_dict = {
                (k[7:] if k.startswith("module.") else k): v for k, v in state_dict.items()
            }
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        has_fusion = any(k.startswith("fusion.") for k in state_dict)
        return nn.Module.load_state_dict(self, state_dict, strict=strict and has_fusion)


def freeze_backbone(model):
    """Stage 2 updates the fusion block only."""
    net = model.module if isinstance(model, nn.Module) and hasattr(model, "module") else model
    for name, param in net.named_parameters():
        param.requires_grad = name.startswith("fusion.")
    return net
