"""The small Fairseq surface required by the pinned inference/training blocks.

No global module injection. Unsupported quantization/layer dropping fails closed.
"""

from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F


class FairseqDropout(nn.Dropout):
    def __init__(self, p, module_name=None):
        super().__init__(p)


def LayerNorm(shape, export=False):
    return nn.LayerNorm(shape)


class LayerDropModuleList(nn.ModuleList):
    def __init__(self, p):
        if p != 0:
            raise ValueError("LayerDrop is not enabled in the QM9 reproduction")
        super().__init__()


def quant_noise(module, p, block_size):
    if p != 0:
        raise ValueError("Quantization noise is not enabled in the QM9 reproduction")
    return module


def softmax(x, dim, onnx_trace=False):
    return F.softmax(x, dim=dim, dtype=torch.float32)


def get_activation_fn(name):
    return {"relu": F.relu, "gelu": F.gelu, "tanh": torch.tanh, "linear": lambda x: x}[name]


utils = SimpleNamespace(softmax=softmax, get_activation_fn=get_activation_fn)
