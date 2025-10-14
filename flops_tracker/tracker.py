from dataclasses import dataclass
from typing import List, Optional
import csv, time
from .estimators.base import FlopsEstimator

@dataclass
class StepLog:
    epoch:int; step:int; batch_size:int
    flops_batch:int; flops_epoch_cum:int; flops_total_cum:int
    wall_time_s:float

@dataclass
class EpochLog:
    epoch:int; flops_epoch:int; flops_total_cum:int

class FlopsTracker:
    def __init__(self, estimator: FlopsEstimator, run_name="run", log_wall_time=True):
        self.estimator = estimator; self.run_name = run_name; self.log_wall_time = log_wall_time
        self.total_flops = 0; self._epoch_flops = 0; self._epoch = 0; self._step = 0
        self.step_logs: List[StepLog] = []; self.epoch_logs: List[EpochLog] = []
        self._t0: Optional[float] = None

    def __enter__(self): self.start(); return self
    def __exit__(self, exc_type, exc, tb): self.stop()

    def start(self):
        if hasattr(self.estimator, "prepare") and callable(self.estimator.prepare):
            self.estimator.prepare()
        self._t0 = time.time()

    def stop(self): pass

    def on_epoch_start(self): self._epoch += 1; self._step = 0; self._epoch_flops = 0
    def on_epoch_end(self):
        self.epoch_logs.append(EpochLog(self._epoch, self._epoch_flops, self.total_flops))

    def update_batch(self, batch_size:int):
        self._step += 1; t1 = time.time()
        fb = int(self.estimator.flops_for_batch(batch_size))
        self._epoch_flops += fb; self.total_flops += fb
        wall = (time.time()-t1) if self.log_wall_time else 0.0
        self.step_logs.append(StepLog(self._epoch, self._step, batch_size, fb, self._epoch_flops, self.total_flops, wall))

    def save_batch_csv(self, path:str):
        with open(path,"w",newline="") as f:
            w=csv.writer(f); w.writerow(["epoch","step","batch_size","flops_batch","flops_epoch_cum","flops_total_cum","wall_time_s"])
            for r in self.step_logs: w.writerow([r.epoch,r.step,r.batch_size,r.flops_batch,r.flops_epoch_cum,r.flops_total_cum,f"{r.wall_time_s:.6f}"])
    def save_epoch_csv(self, path:str):
        with open(path,"w",newline="") as f:
            w=csv.writer(f); w.writerow(["epoch","flops_epoch","flops_total_cum"])
            for r in self.epoch_logs: w.writerow([r.epoch,r.flops_epoch,r.flops_total_cum])
