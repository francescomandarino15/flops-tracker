from abc import ABC, abstractmethod

class FlopsEstimator(ABC):
    """Interfaccia base: ritorna i FLOPs per batch."""
    @abstractmethod
    def flops_for_batch(self, batch_size: int) -> int:
        ...
