cat > flops_tracker/estimators/torch_auto.py <<'PY'
from __future__ import annotations
from typing import Dict, Tuple, Optional, Any, Callable, List
import math
import warnings
import torch
import torch.nn as nn

from .base import FlopsEstimator

def _numel(x: torch.Tensor) -> int:
    return int(x.numel()) if isinstance(x, torch.Tensor) else 0

def _prod(xs) -> int:
    p=1
    for v in xs: p*=int(v)
    return p

def _unwrap_parallel(model: nn.Module) -> nn.Module:
    if hasattr(model, "module") and isinstance(model.module, nn.Module):
        return _unwrap_parallel(model.module)
    return model

class TorchAutoEstimator(FlopsEstimator):
  
    def __init__(
        self,
        model: nn.Module,
        input_example: Tuple[Tuple[int, ...], ...] | Tuple[int, ...],
        training_factor: float = 3.0,
        include_pooling_cost: bool = False,
        include_norm_cost: bool = False,
        include_softmax_cost: bool = False,
        device: Optional[str] = None,
        use_eval_mode: bool = True,
    ):
        self.model = _unwrap_parallel(model)
        self.input_example = input_example
        self.training_factor = float(training_factor)
        self.include_pooling_cost = include_pooling_cost
        self.include_norm_cost = include_norm_cost
        self.include_softmax_cost = include_softmax_cost
        self.device = device or (next(self.model.parameters()).device if any(p.requires_grad for p in self.model.parameters()) else torch.device("cpu"))
        self.use_eval_mode = use_eval_mode

        self._per_sample_forward_flops: Optional[int] = None
        self._handlers: Dict[type, Callable[[nn.Module, Tuple, Tuple], int]] = {}
        self._register_default_handlers()

    # ---------- public API ----------
    def prepare(self):
        modules = [m for m in self.model.modules() if m is not self.model]
        ios: Dict[nn.Module, Tuple[Tuple, Tuple]] = {}
        handles = []
        def h(m, inp, out):
            ios[m] = (inp, out if isinstance(out, tuple) else (out,))
        for m in modules:
            handles.append(m.register_forward_hook(h))

        # dry-run
        orig = self.model.training
        if self.use_eval_mode: self.model.eval()
        with torch.no_grad():
            dummy_inputs = self._make_dummy(self.input_example, self.device)
            _ = self.model(*dummy_inputs) if isinstance(dummy_inputs, tuple) else self.model(dummy_inputs)
        if self.use_eval_mode and orig: self.model.train()
        for hh in handles: hh.remove()

        # somma FLOPs per-sample
        per_sample = 0
        for m, (inp, out) in ios.items():
            per_sample += self._flops_for_module(m, inp, out)
        self._per_sample_forward_flops = int(per_sample)

    def flops_for_batch(self, batch_size: int) -> int:
        if self._per_sample_forward_flops is None:
            self.prepare()
        fwd = self._per_sample_forward_flops * int(batch_size)
        return int(self.training_factor * fwd)

    # ---------- helpers ----------
    def _make_dummy(self, example, device):
        if isinstance(example, tuple) and len(example) > 0 and isinstance(example[0], (tuple, list)):
            return tuple(torch.zeros(s, device=device) for s in example)
        if isinstance(example, tuple):
            return torch.zeros(example, device=device)
        raise ValueError("input_example deve essere una shape (tuple) o una tupla di shape.")

    def _flops_for_module(self, m: nn.Module, inp: Tuple, out: Tuple) -> int:
        # handler diretto per classe; altrimenti somma dei figli (composizione) o 0
        for cls in self._handlers:
            if isinstance(m, cls):
                return int(self._handlers[cls](m, inp, out))
        # se è un container, prova a sommare i figli
        if any(True for _ in m.children()):
            s = 0
            for c in m.children():
                pass
            return s
        return 0

    def _register(self, cls: type, fn: Callable[[nn.Module, Tuple, Tuple], int]):
        self._handlers[cls] = fn

    def _register_default_handlers(self):
        # ---- Linear ----
        def linear(m: nn.Linear, inp, out):
            x = inp[0]; B = x.shape[0] if x.ndim>1 else 1
            return 2 * int(m.in_features) * int(m.out_features)
        self._register(nn.Linear, linear)

        # ---- Conv[d] + groups ----
        def convNd(m, inp, out, dim: int):
            x = inp[0]
            B = x.shape[0] if x.ndim >= (2+dim) else 1
            Cin = int(m.in_channels); Cout = int(m.out_channels)
            k = m.kernel_size if isinstance(m.kernel_size, tuple) else (m.kernel_size,)*dim
            groups = int(m.groups)
            out_t = out[0]
            spatial = out_t.shape[-dim:]
            HoWoDo = _prod(spatial)
            macs = HoWoDo * Cout * ( _prod(k) * Cin // groups )
            return 2 * macs
        for cls, dim in ((nn.Conv1d,1),(nn.Conv2d,2),(nn.Conv3d,3)):
            self._register(cls, lambda m, i, o, d=dim: convNd(m,i,o,d))

        # ---- ConvTranspose[d] ----
        def convTNd(m, inp, out, dim: int):
            x = inp[0]
            Cout = int(m.out_channels); Cin = int(m.in_channels)
            k = m.kernel_size if isinstance(m.kernel_size, tuple) else (m.kernel_size,)*dim
            groups = int(m.groups)
            out_t = out[0]
            spatial = out_t.shape[-dim:]
            HoWoDo = _prod(spatial)
            macs = HoWoDo * Cin * (_prod(k) * Cout // groups)  # swap Cin/Cout
            return 2 * macs
        for cls, dim in ((nn.ConvTranspose1d,1),(nn.ConvTranspose2d,2),(nn.ConvTranspose3d,3)):
            self._register(cls, lambda m, i, o, d=dim: convTNd(m,i,o,d))

        # ---- Pooling ----
        def avgpool(m, inp, out, dim: int):
            if not self.include_pooling_cost: return 0
            y = out[0]; 
            vol = 1
            if hasattr(m, "kernel_size"):
                ks = m.kernel_size if isinstance(m.kernel_size, tuple) else (m.kernel_size,)*dim
                vol = _prod(ks)
            return _numel(y) * vol
        for cls, dim in ((nn.AvgPool1d,1),(nn.AvgPool2d,2),(nn.AvgPool3d,3),
                         (nn.AdaptiveAvgPool1d,1),(nn.AdaptiveAvgPool2d,2),(nn.AdaptiveAvgPool3d,3)):
            self._register(cls, lambda m,i,o,d=dim: avgpool(m,i,o,d))
        for cls in (nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d,
                    nn.AdaptiveMaxPool1d, nn.AdaptiveMaxPool2d, nn.AdaptiveMaxPool3d):
            self._register(cls, lambda m,i,o: 0)

        # ---- Normalization ----
        def norm_cost(y: torch.Tensor):
            if not self.include_norm_cost: return 0
            N = _numel(y)
            return 5 * N
        for cls in (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm, nn.GroupNorm, nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d):
            self._register(cls, lambda m,i,o: norm_cost(o[0]))

        # ---- RNN / GRU / LSTM ----
        def _get_BT(inp0, out0):
            x = inp0
            if x.ndim == 3:
                o = out0[0] if isinstance(out0, (tuple,list)) else out0
                if o.ndim==3:
                    if x.shape[0] == o.shape[0]: 
                        return x.shape[0], x.shape[1]
                    else:
                        return x.shape[1], x.shape[0]
            return 1, 1

        def rnn_base_cost(B:int, T:int, I:int, H:int, gates:int):
            macs = gates * (I*H + H*H)
            return 2 * macs * B * T

        def rnn(m: nn.RNN, inp, out):
            x = inp[0]; B,T = _get_BT(x, out)
            I = int(m.input_size); H = int(m.hidden_size)
            return rnn_base_cost(B,T,I,H,gates=1)

        def gru(m: nn.GRU, inp, out):
            x = inp[0]; B,T = _get_BT(x, out)
            I = int(m.input_size); H = int(m.hidden_size)
            return rnn_base_cost(B,T,I,H,gates=3)

        def lstm(m: nn.LSTM, inp, out):
            x = inp[0]; B,T = _get_BT(x, out)
            I = int(m.input_size); H = int(m.hidden_size)
            return rnn_base_cost(B,T,I,H,gates=4)
        self._register(nn.RNN, rnn)
        self._register(nn.GRU, gru)
        self._register(nn.LSTM, lstm)

        # ---- MultiheadAttention ----
        def mha(m: nn.MultiheadAttention, inp, out):
            
            x = inp[0]
            if x.ndim != 3:
                return 0
            if x.shape[0] == out[0].shape[0]: 
                B, L, E = x.shape
            else:
                L, B, E = x.shape
            H = int(m.num_heads)
            d = E // H if H>0 else E

            proj_qkv = 3 * 2 * (B * L * E * E)
            qkT = 2 * B * H * L * L * d
            soft = (5 * B * H * L * L) if self.include_softmax_cost else 0
            attn_v = 2 * B * H * L * L * d
            out_proj = 2 * (B * L * E * E)
            return proj_qkv + qkT + soft + attn_v + out_proj
        self._register(nn.MultiheadAttention, mha)

        # ---- Upsample/Interpolate ----
        def upsample(m, inp, out):
            y = out[0]; return _numel(y) 
        for cls in (nn.Upsample,):
            self._register(cls, upsample)

        # ---- Layers con costo ~0 (reshape/lookup/stocastici) ----
        zero_cost = lambda m,i,o: 0
        for cls in (nn.Dropout, nn.Dropout2d, nn.Dropout3d,
                    nn.Identity, nn.Flatten, nn.Unflatten, nn.ReLU, nn.GELU, nn.SiLU, nn.Sigmoid, nn.Tanh,
                    nn.AdaptiveLogSoftmaxWithLoss, nn.Softmax, nn.LogSoftmax, nn.Softmin, nn.LogSigmoid):
            self._register(cls, zero_cost)
        for cls in (nn.Embedding, nn.EmbeddingBag):
            self._register(cls, zero_cost)
        for cls in (nn.ReflectionPad1d, nn.ReflectionPad2d, nn.ReflectionPad3d,
                    nn.ReplicationPad1d, nn.ReplicationPad2d, nn.ReplicationPad3d,
                    nn.ZeroPad2d, nn.ConstantPad1d, nn.ConstantPad2d, nn.ConstantPad3d):
            self._register(cls, zero_cost)
                      
