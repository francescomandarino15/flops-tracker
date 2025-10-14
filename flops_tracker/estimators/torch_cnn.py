from __future__ import annotations
from typing import Dict, Tuple, List, Optional

import torch
import torch.nn as nn

from .base import FlopsEstimator


class TorchCNNLayerwiseEstimator(FlopsEstimator):

    def __init__(
        self,
        model: nn.Module,
        input_size: Tuple[int, int, int, int],   
        training_factor: float = 3.0,
        device: Optional[str] = None,
        use_eval_mode: bool = True,
    ):
        self.model = model
        self.input_size = input_size
        self.training_factor = float(training_factor)
        self.device = device or (
            next(model.parameters()).device.type if any(p.requires_grad for p in model.parameters()) else "cpu"
        )
        self.use_eval_mode = use_eval_mode

        self._conv_layers: List[nn.Conv2d] = []
        self._linear_layers: List[nn.Linear] = []
        self._conv_out_shapes: Dict[nn.Module, Tuple[int, int, int]] = {}  
        self._per_sample_forward_flops: Optional[int] = None

    # ---------- public API ----------
    def prepare(self):
        self._conv_layers = [m for m in self.model.modules() if isinstance(m, nn.Conv2d)]
        self._linear_layers = [m for m in self.model.modules() if isinstance(m, nn.Linear)]
        hooks = []

        def save_conv_shape(module: nn.Module, inp, out):
            if isinstance(out, (tuple, list)):
                out = out[0]
            _, c_out, h_o, w_o = out.shape
            self._conv_out_shapes[module] = (int(c_out), int(h_o), int(w_o))

        for m in self._conv_layers:
            hooks.append(m.register_forward_hook(save_conv_shape))

        # dry-run
        orig_mode = self.model.training
        if self.use_eval_mode:
            self.model.eval()
        with torch.no_grad():
            dummy = torch.zeros(self.input_size, device=self.device)
            _ = self.model(dummy)

        # pulizia hook
        for h in hooks:
            h.remove()
        if self.use_eval_mode and orig_mode:
            self.model.train()

        # FLOPs per 1 sample (forward only)
        per_sample = 0

        # Conv2d
        for m in self._conv_layers:
            c_out, h_o, w_o = self._conv_out_shapes[m]
            c_in = int(m.in_channels)
            if isinstance(m.kernel_size, tuple):
                k_h, k_w = int(m.kernel_size[0]), int(m.kernel_size[1])
            else:
                k_h = k_w = int(m.kernel_size)
            groups = int(m.groups)
            # MAC per output element: k_h*k_w*c_in/groups
            macs = h_o * w_o * c_out * (k_h * k_w * c_in // groups)
            per_sample += 2 * macs

        # Linear
        for m in self._linear_layers:
            per_sample += 2 * int(m.in_features) * int(m.out_features)

        self._per_sample_forward_flops = int(per_sample)

    def flops_for_batch(self, batch_size: int) -> int:
        if self._per_sample_forward_flops is None:
            self.prepare()
        fwd = self._per_sample_forward_flops * int(batch_size)
        return int(self.training_factor * fwd)
