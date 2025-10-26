cat > flops_tracker/tracker.py <<'PY'
from dataclasses import dataclass
from typing import List, Optional, Iterable, Callable, Any, Dict
import csv, time

from .estimators.base import FlopsEstimator

# -------------------- Logs --------------------
@dataclass
class StepLog:
    epoch:int; step:int; batch_size:int
    flops_batch:int; flops_epoch_cum:int; flops_total_cum:int
    wall_time_s:float

@dataclass
class EpochLog:
    epoch:int; flops_epoch:int; flops_total_cum:int

# -------------------- Core Tracker --------------------
class FlopsTracker:
    """
    Core: mantiene contatori e CSV. Può essere esteso con bind specifici (es. torch_bind).
    """
    def __init__(self, estimator: FlopsEstimator, run_name="run", log_wall_time=True):
        self.estimator = estimator
        self.run_name = run_name
        self.log_wall_time = log_wall_time

        self.total_flops = 0
        self._epoch_flops = 0
        self._epoch = 0
        self._step = 0
        self.step_logs: List[StepLog] = []
        self.epoch_logs: List[EpochLog] = []
        self._t0: Optional[float] = None

        # ---- torch bind state (opzionale) ----
        self._torch_ctx: Optional[Dict[str, Any]] = None

    # Context manager
    def __enter__(self): self.start(); return self
    def __exit__(self, exc_type, exc, tb): self.stop()

    def start(self):
        if hasattr(self.estimator, "prepare") and callable(self.estimator.prepare):
            self.estimator.prepare()
        self._t0 = time.time()

    def stop(self): pass

    # Epoch boundaries
    def on_epoch_start(self):
        self._epoch += 1
        self._step = 0
        self._epoch_flops = 0

    def on_epoch_end(self):
        self.epoch_logs.append(EpochLog(self._epoch, self._epoch_flops, self.total_flops))

    # Batch update
    def update_batch(self, batch_size:int):
        self._step += 1
        t1 = time.time()
        fb = int(self.estimator.flops_for_batch(batch_size))
        self._epoch_flops += fb
        self.total_flops += fb
        wall = (time.time()-t1) if self.log_wall_time else 0.0
        self.step_logs.append(StepLog(self._epoch, self._step, batch_size, fb, self._epoch_flops, self.total_flops, wall))

    # Export
    def save_batch_csv(self, path:str):
        with open(path,"w",newline="") as f:
            w=csv.writer(f)
            w.writerow(["epoch","step","batch_size","flops_batch","flops_epoch_cum","flops_total_cum","wall_time_s"])
            for r in self.step_logs:
                w.writerow([r.epoch,r.step,r.batch_size,r.flops_batch,r.flops_epoch_cum,r.flops_total_cum,f"{r.wall_time_s:.6f}"])

    def save_epoch_csv(self, path:str):
        with open(path,"w",newline="") as f:
            w=csv.writer(f)
            w.writerow(["epoch","flops_epoch","flops_total_cum"])
            for r in self.epoch_logs:
                w.writerow([r.epoch,r.flops_epoch,r.flops_total_cum])

    # -------------------- PyTorch binding (stile tqdm) --------------------
    def torch_bind(
        self,
        model: Any,
        optimizer: Any,
        loss_fn: Optional[Any],
        train_loader: Any,
        device: Optional[str] = None,
        *,
        autosave_prefix: Optional[str] = None,
        on_epoch_end: Optional[Callable[['FlopsTracker', int], None]] = None,
        zero_grad_kwargs: Optional[dict] = None,
    ) -> 'FlopsTracker':
        """
        Configura il tracker per il training PyTorch e lo rende *iterabile*:
            for _ in ft(range(EPOCHS)): pass

        Parametri:
          - model, optimizer, loss_fn: oggetti PyTorch
          - train_loader: DataLoader che produce (data, target)
          - device: se None usa device del modello
          - autosave_prefix: se impostato, a fine run salva:
                {prefix}_batch.csv e {prefix}_epoch.csv
          - on_epoch_end(ft, epoch): callback opzionale a fine epoch
          - zero_grad_kwargs: es. {"set_to_none": True}
        """
        self._torch_ctx = {
            "model": model,
            "optimizer": optimizer,
            "loss_fn": loss_fn,
            "train_loader": train_loader,
            "device": device,
            "autosave_prefix": autosave_prefix,
            "on_epoch_end_cb": on_epoch_end,
            "zero_grad_kwargs": zero_grad_kwargs or {"set_to_none": True},
        }
        return self

    def __call__(self, epoch_iterable: Iterable[int]):
        """
        Rende l'istanza *iterabile*: itera sulle epoch e gestisce internamente
        il training loop PyTorch + logging FLOPs.
        Uso:
            for _ in ft(range(EPOCHS)): pass
        """
        if self._torch_ctx is None:
            raise RuntimeError("torch_bind non chiamato: configura prima il contesto PyTorch.")
        ctx = self._torch_ctx
        model = ctx["model"]
        optim = ctx["optimizer"]
        loss_fn = ctx["loss_fn"]
        loader = ctx["train_loader"]
        device = ctx["device"]
        if device is None:
            try:
                device = next(model.parameters()).device
            except StopIteration:
                device = "cpu"

        # assicura prepare() fatta
        self.start()

        for e in epoch_iterable:
            self.on_epoch_start()
            model.train()
            for step, batch in enumerate(loader, 1):
                if isinstance(batch, (tuple, list)) and len(batch) >= 2:
                    data, target = batch[0], batch[1]
                else:
                    raise ValueError("Il train_loader deve produrre (data, target).")

                data, target = data.to(device), target.to(device)
                optim.zero_grad(**ctx["zero_grad_kwargs"])
                out = model(data)
                if loss_fn is None:
                    # default nll_loss sullo 0/1 di log_softmax
                    import torch.nn.functional as F
                    loss = F.nll_loss(out, target)
                else:
                    loss = loss_fn(out, target)
                loss.backward()
                optim.step()

                # FLOPs
                self.update_batch(batch_size=data.size(0))

            self.on_epoch_end()

            if ctx["on_epoch_end_cb"] is not None:
                ctx["on_epoch_end_cb"](self, self._epoch)

            # opzionalmente, stampa una riga stile progress
            last = self.epoch_logs[-1]
            print(f"[Epoch {last.epoch:02d}] FLOPs_epoch={last.flops_epoch:,} | cum={last.flops_total_cum:,}")

            yield e  # consente "for _ in ft(range(EPOCHS))"

        # autosave CSV a fine run
        if ctx["autosave_prefix"]:
            self.save_batch_csv(f"{ctx['autosave_prefix']}_batch.csv")
            self.save_epoch_csv(f"{ctx['autosave_prefix']}_epoch.csv")
