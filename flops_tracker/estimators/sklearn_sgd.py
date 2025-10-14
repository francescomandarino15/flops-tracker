from .base import FlopsEstimator

class SklearnSGDEstimator(FlopsEstimator):
    """
    FLOPs approx per batch per classificazione binaria:
        ~ 4 * B * f
    Per one-vs-rest multiclasse moltiplica per C.
    """
    def __init__(self, n_features:int, n_classes:int=1):
        self.f = int(n_features); self.C = int(max(1, n_classes))

    def flops_for_batch(self, batch_size:int) -> int:
        return 4 * int(batch_size) * self.f * self.C
