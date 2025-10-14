from __future__ import annotations
from typing import Dict, Tuple, List, Optional

import torch
import torch.nn as nn

from .base import FlopsEstimator


class TorchCNNLayerwiseEstimator(FlopsEstimator):
    """
    Stima FLOPs "hardware-agnostic" per modelli PyTorch con layer Conv2d e Linear.

    Idea:
      - Durante prepare() esegue un dry-run con input finto (B=1) per catturare
        le shape di output dei layer Conv2d via forward hook (C_out, H_o, W_o).
      - Calcola i FLOPs forward per 1 sample sommando:
          Conv2d: 2 * H_o * W_o * C_out * (K_h * K_w * C_in / groups)
          Linear: 2 * in_features * out_features
        (1 MAC = 2 FLOPs)
      - Per un batch qualsiasi: FLOPs_forward_batch = per_sample * batch_size
      - Per il training: FLOPs_batch = training_factor * FLOPs_forward_batch
        (euristica comune: ≈3× forward per includere backward+update)

    Note:
      - Pooling/Dropout/BN sono trascurati (costo minore); documentalo nel README.
      - Supporta conv "grouped" (depthwise = groups == C_in) tramite il fattore /groups.
    """

    def __init__(
        self,
        model: nn.Module,
        input_size: Tuple[int, int, int, int],   # (B, C, H, W) con B=1 consigliato per prepare()
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
        self._conv_out_shapes: Dict[nn.Module, Tuple[int, int, int]] = {}  # module -> (C_out, H_o, W_o)
        self._per_sample_forward_flops: Optional[int] = None

    # ---------- public API ----------
    def prepare(self):
        """Esegue un dry-run (B=1) per catturare H_o, W_o di ogni Conv2d e precalcolare i FLOPs per sample."""
        # raccogli layer interessanti
        self._conv_layers = [m for m in self.model.modules() if isinstance(m, nn.Conv2d)]
        self._linear_layers = [m for m in self.model.modules() if isinstance(m, nn.Linear)]
        hooks = []

        def save_conv_shape(module: nn.Module, inp, out):
            # out: (B, C_out, H_o, W_o)
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
