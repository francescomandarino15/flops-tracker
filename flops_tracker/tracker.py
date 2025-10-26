from dataclasses import dataclass
from typing import List, Optional, Iterable, Callable, Any, Dict, Literal
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
    Tracker generico dei FLOPs (hardware-agnostici).
    - Può essere "bindato" a PyTorch con .torch_bind(...)
    - È *iterabile* (stile tqdm): for _ in ft(range(EPOCHS), ...): pass
    """
    def __init__(
        self,
        estimator: FlopsEstimator,
        run_name: str = "run",
        log_wall_time: bool = True,
        keep_batch_logs: bool = False,   # se False, non salva lo storico per-batch
        keep_epoch_logs: bool = True,    # se False, non salva lo storico per-epoch
    ):
        self.estimator = estimator
        self.run_name = run_name
        self.log_wall_time = log_wall_time
        self.keep_batch_logs = keep_batch_logs
        self.keep_epoch_logs = keep_epoch_logs

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
        if self.keep_epoch_logs:
            self.epoch_logs.append(EpochLog(self._epoch, self._epoch_flops, self.total_flops))

    # Batch update
    def update_batch(self, batch_size:int):
        self._step += 1
        t1 = time.time()
        fb = int(self.estimator.flops_for_batch(batch_size))
        self._epoch_flops += fb
        self.total_flops += fb
        wall = (time.time()-t1) if self.log_wall_time else 0.0
        if self.keep_batch_logs:
            self.step_logs.append(StepLog(self._epoch, self._step, batch_size, fb, self._epoch_flops, self.total_flops, wall))

    # Export CSV (solo se i log esistono)
    def save_batch_csv(self, path:str):
        if not self.keep_batch_logs:
            raise RuntimeError("keep_batch_logs=False: nessun log per-batch disponibile.")
        with open(path,"w",newline="") as f:
            w=csv.writer(f)
            w.writerow(["epoch","step","batch_size","flops_batch","flops_epoch_cum","flops_total_cum","wall_time_s"])
            for r in self.step_logs:
                w.writerow([r.epoch,r.step,r.batch_size,r.flops_batch,r.flops_epoch_cum,r.flops_total_cum,f"{r.wall_time_s:.6f}"])

    def save_epoch_csv(self, path:str):
        if not self.keep_epoch_logs:
            raise RuntimeError("keep_epoch_logs=False: nessun log per-epoch disponibile.")
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
        zero_grad_kwargs: Optional[dict] = None,
    ) -> 'FlopsTracker':
        """Configura il contesto PyTorch (training loop gestito internamente)."""
        self._torch_ctx = {
            "model": model,
            "optimizer": optimizer,
            "loss_fn": loss_fn,
            "train_loader": train_loader,
            "device": device,
            "zero_grad_kwargs": zero_grad_kwargs or {"set_to_none": True},
        }
        return self

    def __call__(
        self,
        epoch_iterable: Iterable[int],
        *,
        # stampa
        print_level: Literal["none","epoch","batch","both"] = "epoch",
        # export CSV
        export: Literal["none","batch","epoch","both"] = "none",
        export_prefix: Optional[str] = None,
        # callback opzionale
        on_epoch_end: Optional[Callable[['FlopsTracker', int], None]] = None,
    ):
        """
        Rende l'istanza *iterabile* e gestisce il training loop.
        Uso:
            for _ in ft(range(EPOCHS), print_level="epoch", export="epoch", export_prefix="cnn_flops"): pass
        """
        if self._torch_ctx is None:
            raise RuntimeError("torch_bind non chiamato: configura prima il contesto PyTorch.")

        # abilita/disabilita raccolta log in base a export richiesto
        self.keep_batch_logs = export in ("batch","both")
        self.keep_epoch_logs = True if export in ("epoch","both") else self.keep_epoch_logs

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

        import torch.nn.functional as F

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
                loss = F.nll_loss(out, target) if loss_fn is None else loss_fn(out, target)
                loss.backward(); optim.step()

                # FLOPs
                self.update_batch(batch_size=data.size(0))

                # stampa per-batch (se richiesto)
                if print_level in ("batch","both"):
                    print(f"[Epoch {self._epoch:02d} Step {step:04d}] batch_FLOPs={self._epoch_flops:,} (epoch cum) | total={self.total_flops:,}")

            self.on_epoch_end()

            # callback e stampa per-epoch
            if on_epoch_end is not None:
                on_epoch_end(self, self._epoch)

            if print_level in ("epoch","both"):
                if self.keep_epoch_logs and self.epoch_logs:
                    last = self.epoch_logs[-1]
                    print(f"[Epoch {last.epoch:02d}] FLOPs_epoch={last.flops_epoch:,} | cum={last.flops_total_cum:,}")
                else:
                    # se non mantieni i log per-epoch, stampa il cumulato calcolato
                    print(f"[Epoch {self._epoch:02d}] FLOPs_epoch={self._epoch_flops:,} | cum={self.total_flops:,}")

            yield e  # consente "for _ in ft(range(EPOCHS), ...)"

        # export CSV (se richiesto)
        if export != "none":
            prefix = export_prefix or self.run_name
            if export in ("batch","both") and self.keep_batch_logs:
                self.save_batch_csv(f"{prefix}_batch.csv")
            if export in ("epoch","both"):
                # se non abbiamo log per-epoch ma vogliamo esportare, creali al volo con ultimo valore
                if not self.keep_epoch_logs and self._epoch > 0:
                    self.epoch_logs.append(EpochLog(self._epoch, self._epoch_flops, self.total_flops))
                self.save_epoch_csv(f"{prefix}_epoch.csv")
