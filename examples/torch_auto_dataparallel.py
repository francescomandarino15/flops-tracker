import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import SGD

from flops_tracker import FlopsTracker
from flops_tracker.estimators import TorchAutoEstimator

class Cnn(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3)   
        self.conv2 = nn.Conv2d(16, 32, 3)  
        self.fc1 = nn.Linear(32*12*12, 64) 
        self.fc2 = nn.Linear(64, 10)
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return F.log_softmax(self.fc2(x), dim=1)

def main():
    has_cuda = torch.cuda.is_available() and torch.cuda.device_count() > 0
    device = "cuda" if has_cuda else "cpu"

    # dataset 
    X = torch.randn(2048, 1, 28, 28)
    y = torch.randint(0, 10, (2048,))
    loader = DataLoader(TensorDataset(X, y), batch_size=128, shuffle=True, num_workers=0, pin_memory=False)

    model = Cnn()
    if has_cuda:
        # DataParallel 1 processo diviso su tutte le GPU disponibili
        model = nn.DataParallel(model).cuda()
    else:
        print("[WARN] CUDA non disponibile: eseguo senza DataParallel.")
        model = model.to(device)

    # Estimator automatico 
    est = TorchAutoEstimator(
        model=model,
        input_example=(1, 1, 28, 28),
        training_factor=3.0,
        include_pooling_cost=False,   # attivabili a piacere
        include_norm_cost=False,
    )

    opt = SGD(model.parameters(), lr=1e-2, momentum=0.9)

    # Tracker
    ft = FlopsTracker(est, run_name="dp_cnn").torch_bind(
        model=model, optimizer=opt, loss_fn=None, train_loader=loader, device=device
    )

    EPOCHS = 3
    for _ in ft(range(EPOCHS), print_level="epoch", export="epoch", export_prefix="dp_cnn_flops"):
        pass

    print("FLOPs totali (DP o single-GPU):", ft.total_flops)

if __name__ == "__main__":
    main()
