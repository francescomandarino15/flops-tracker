from dataclasses import dataclass
from typing import List, Optional, Iterable, Callable, Any, Dict, Literal, Union
import csv, time, os

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
    - .torch_bind(...) per il training PyTorch
    - Iterabile (stile tqdm): for _ in ft(range(EPOCHS), ...): pass
    - One-liner: total = FlopsTracker(...).torch_bind(...).run(EPOCHS, ...)
    - Opzionale: logging su Weights & Biases (W&B)
    """
    def __init__(
        self,
        estimator: FlopsEstimator,
        run_name: str = "run",
        log_wall_time: bool = True,
        keep_batch_logs: bool = False,
        keep_epoch_logs: bool = True,
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

        # --- gestione log con Wandb (opzionale) ---
        self._wb_enabled = False
        self._wb = {
            "project": None,
            "run_name": None,
            "config": None,
            "log_mode": "none",  # "none" | "batch" | "epoch" | "both"
            "only_rank0": True,  # in DDP logga solo rank0
            "is_rank0": True,    # determinato automaticamente
            "run": None,         # oggetto wandb.run
        }

    # ------------- Context manager -------------
    def __enter__(self): self.start(); return self
    def __exit__(self, exc_type, exc, tb): self.stop()

    def start(self):
        if hasattr(self.estimator, "prepare") and callable(self.estimator.prepare):
            self.estimator.prepare()
        self._t0 = time.time()

    def stop(self):
        # chiudi eventualmente wandb
        self._wandb_finish()

    # ------------- Epoch boundaries -------------
    def on_epoch_start(self):
        self._epoch += 1
        self._step = 0
        self._epoch_flops = 0

    def on_epoch_end(self):
        if self.keep_epoch_logs:
            self.epoch_logs.append(EpochLog(self._epoch, self._epoch_flops, self.total_flops))
        # W&B epoch-level
        if self._wb_enabled and self._wb["is_rank0"] and self._wb["log_mode"] in ("epoch","both"):
            payload = {
                "flops/epoch": self._epoch_flops,
                "flops/total_cum": self.total_flops,
                "epoch": self._epoch,
            }
            self._wandb_log(payload, commit=True)

    # ------------- Batch update -------------
    def update_batch(self, batch_size:int):
        self._step += 1
        t1 = time.time()
        fb = int(self.estimator.flops_for_batch(batch_size))
        self._epoch_flops += fb
        self.total_flops += fb
        wall = (time.time()-t1) if self.log_wall_time else 0.0
        if self.keep_batch_logs:
            self.step_logs.append(StepLog(self._epoch, self._step, batch_size, fb, self._epoch_flops, self.total_flops, wall))

        # W&B batch-level
        if self._wb_enabled and self._wb["is_rank0"] and self._wb["log_mode"] in ("batch","both"):
            payload = {
                "flops/batch": fb,
                "flops/epoch_cum": self._epoch_flops,
                "flops/total_cum": self.total_flops,
                "epoch": self._epoch,
                "step": self._step,
                "batch_size": batch_size,
            }
            self._wandb_log(payload, commit=False)

    # ------------- CSV export -------------
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

    # -------------------- PyTorch binding --------------------
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
        self._torch_ctx = {
            "model": model,
            "optimizer": optimizer,
            "loss_fn": loss_fn,
            "train_loader": train_loader,
            "device": device,
            "zero_grad_kwargs": zero_grad_kwargs or {"set_to_none": True},
        }
        return self

    # -------------------- ONE-LINER --------------------
    def run(
        self,
        epochs: int,
        *,
        print_level: Literal["none","epoch","batch","both"] = "none",
        export: Literal["none","batch","epoch","both"] = "none",
        export_prefix: Optional[str] = None,
        on_epoch_end: Optional[Callable[['FlopsTracker', int], None]] = None,
        # --- W&B ---
        wandb: bool = False,
        wandb_project: Optional[str] = None,
        wandb_run_name: Optional[str] = None,
        wandb_config: Optional[Dict[str, Any]] = None,
        wandb_log: Literal["none","batch","epoch","both"] = "none",
        wandb_only_rank0: bool = True,
    ) -> int:
        for _ in self(
            range(epochs),
            print_level=print_level,
            export=export,
            export_prefix=export_prefix,
            on_epoch_end=on_epoch_end,
            # W&B
            wandb=wandb,
            wandb_project=wandb_project,
            wandb_run_name=wandb_run_name,
            wandb_config=wandb_config,
            wandb_log=wandb_log,
            wandb_only_rank0=wandb_only_rank0,
        ):
            pass
        return int(self.total_flops)

    # -------------------- Iterabile / tqdm-like --------------------
    def __call__(
        self,
        epoch_iterable: Union[int, Iterable[int]],
        *,
        # stampa
        print_level: Literal["none","epoch","batch","both"] = "epoch",
        # export CSV
        export: Literal["none","batch","epoch","both"] = "none",
        export_prefix: Optional[str] = None,
        # callback
        on_epoch_end: Optional[Callable[['FlopsTracker', int], None]] = None,
        # --- W&B ---
        wandb: bool = False,
        wandb_project: Optional[str] = None,
        wandb_run_name: Optional[str] = None,
        wandb_config: Optional[Dict[str, Any]] = None,
        wandb_log: Literal["none","batch","epoch","both"] = "none",
        wandb_only_rank0: bool = True,
    ):
        # Se passi un intero, comportati come run()
        if isinstance(epoch_iterable, int):
            return self.run(
                epochs=epoch_iterable,
                print_level=print_level,
                export=export,
                export_prefix=export_prefix,
                on_epoch_end=on_epoch_end,
                wandb=wandb, wandb_project=wandb_project, wandb_run_name=wandb_run_name,
                wandb_config=wandb_config, wandb_log=wandb_log, wandb_only_rank0=wandb_only_rank0,
            )

        if self._torch_ctx is None:
            raise RuntimeError("torch_bind non chiamato: configura prima il contesto PyTorch.")

        # log collection toggles per CSV
        self.keep_batch_logs = export in ("batch","both")
        self.keep_epoch_logs = True if export in ("epoch","both") else self.keep_epoch_logs

        # --- W&B init (se richiesto) ---
        self._wandb_setup(
            enable=wandb,
            project=wandb_project,
            run_name=wandb_run_name or self.run_name,
            config=wandb_config or {},
            log_mode=wandb_log,
            only_rank0=wandb_only_rank0,
        )

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

                # FLOPs + W&B batch log
                self.update_batch(batch_size=data.size(0))

                if print_level in ("batch","both"):
                    print(f"[Epoch {self._epoch:02d} Step {step:04d}] batch_size={data.size(0)} | epoch_cum={self._epoch_flops:,} | total={self.total_flops:,}")

            self.on_epoch_end()

            if on_epoch_end is not None:
                on_epoch_end(self, self._epoch)

            if print_level in ("epoch","both"):
                if self.keep_epoch_logs and self.epoch_logs:
                    last = self.epoch_logs[-1]
                    print(f"[Epoch {last.epoch:02d}] FLOPs_epoch={last.flops_epoch:,} | cum={last.flops_total_cum:,}")
                else:
                    print(f"[Epoch {self._epoch:02d}] FLOPs_epoch={self._epoch_flops:,} | cum={self.total_flops:,}")

            yield e

        # CSV export
        if export != "none":
            prefix = export_prefix or self.run_name
            if export in ("batch","both") and self.keep_batch_logs:
                self.save_batch_csv(f"{prefix}_batch.csv")
            if export in ("epoch","both"):
                if not self.keep_epoch_logs and self._epoch > 0:
                    self.epoch_logs.append(EpochLog(self._epoch, self._epoch_flops, self.total_flops))
                self.save_epoch_csv(f"{prefix}_epoch.csv")

        # chiudi W&B
        self._wandb_finish()

    # -------------------- Wandb helpers --------------------
    def _wandb_setup(self, enable: bool, project: Optional[str], run_name: Optional[str], config: Dict[str, Any], log_mode: str, only_rank0: bool):
        self._wb_enabled = False
        if not enable:
            return
        # rank detection (DDP)
        rank_env = os.environ.get("RANK")
        is_rank0 = True
        try:
            if rank_env is not None:
                is_rank0 = (int(rank_env) == 0)
        except ValueError:
            is_rank0 = True
        self._wb["is_rank0"] = is_rank0 if only_rank0 else True
        self._wb["only_rank0"] = only_rank0

        # import lazy
        try:
            import wandb  # type: ignore
        except Exception:
            print("[flops-tracker] wandb non installato: `pip install wandb` per abilitare il logging.")
            return

        if self._wb["only_rank0"] and (not self._wb["is_rank0"]):
            self._wb_enabled = True 
            self._wb["log_mode"] = "none"  
            self._wb["project"] = project
            self._wb["run_name"] = run_name
            self._wb["config"] = config
            self._wb["run"] = None
            return

        # init effettivo
        try:
            self._wb["run"] = wandb.init(project=project or "flops-tracker",
                                         name=run_name or self.run_name,
                                         config=config or {},
                                         reinit=True)
            self._wb["project"] = project or "flops-tracker"
            self._wb["run_name"] = run_name or self.run_name
            self._wb["config"] = config or {}
            self._wb["log_mode"] = log_mode
            self._wb_enabled = True
        except Exception as e:
            print(f"[flops-tracker] impossibile inizializzare wandb: {e}")
            self._wb_enabled = False

    def _wandb_log(self, payload: Dict[str, Any], commit: bool):
        if not self._wb_enabled:
            return
        if not self._wb["is_rank0"] and self._wb["only_rank0"]:
            return
        run = self._wb.get("run")
        if run is None:
            return
        try:
            import wandb  
            wandb.log(payload, commit=commit)
        except Exception as e:
            print(f"[flops-tracker] wandb.log fallito: {e}")

    def _wandb_finish(self):
        if not self._wb_enabled:
            return
        if not self._wb["is_rank0"] and self._wb["only_rank0"]:
            return
        run = self._wb.get("run")
        if run is None:
            return
        try:
            import wandb  
            wandb.finish()
        except Exception:
            pass
        finally:
            self._wb["run"] = None
            self._wb_enabled = False
