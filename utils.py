import os
import json
import numpy as np
from collections import OrderedDict
import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_, zeros_
from typing import Any, Dict

# ----------------------
# Struct class (gán thuộc tính động)
# ----------------------
class Struct:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

# ----------------------
# Khởi tạo linear
# ----------------------
@torch.no_grad()
def init_linear(m: nn.Module):
    """
    Khởi tạo Linear layer:
    - weight: xavier_normal_
    - bias: zeros_
    """
    if isinstance(m, nn.Linear):
        xavier_normal_(m.weight, gain=1.0)
        if m.bias is not None:
            zeros_(m.bias)

# ----------------------
# Chuyển y sang log10 (DeepPurpose style)
# ----------------------
def y_log10_transform(y: np.ndarray) -> np.ndarray:
    """
    Transform y (nM) -> p(nM) scale
    log10 transformation
    """
    print("log10 transformation for targets")
    y = y.copy()
    zero_idxs = np.where(y <= 0)[0]
    y[zero_idxs] = 1e-10
    y = -np.log10(y * 1e-9)
    return y.astype(float)

# ----------------------
# Chuyển y kiểu KIBA
# ----------------------
def y_kiba_transform(y: np.ndarray) -> np.ndarray:
    """
    KIBA transformation
    """
    y = -y
    return np.abs(np.min(y)) + y

# ----------------------
# In các tham số gọn gàng
# ----------------------
def print_args(**kwargs) -> Dict[str, Any]:
    print("Print parameters:")
    args_dict = OrderedDict()
    for k, v in sorted(kwargs.items(), key=lambda x: x[0]):
        if isinstance(v, (str, float, int, list, type(None))):
            args_dict[k] = v
        else:
            print(k, type(v))
    print(json.dumps(args_dict, indent=2))
    return kwargs

# ----------------------
# Path join tiện dụng
# ----------------------
pathjoin = os.path.join

# ----------------------
# Chuyển Tensor -> numpy
# ----------------------
t2np = lambda t: t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else t

# ----------------------
# Lấy scalar từ dict (dạng float/int)
# ----------------------
def keep_scalar_func(in_dict: Dict[str, Any], prefix: str = '') -> Dict[str, float]:
    if prefix != '':
        prefix += '_'
    out_dict = {}
    for k, v in in_dict.items():
        if isinstance(v, torch.Tensor):
            v = t2np(v)
        if isinstance(v, (float, int)):
            out_dict[prefix + k] = v
    return out_dict
