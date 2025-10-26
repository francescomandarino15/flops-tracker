import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

from flops_tracker import FlopsTracker, TorchCNNLayerwiseEstimator

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 128
EPOCHS = 2
NUM_WORKERS = 0
PIN_MEMORY = False

# Modello 
class Cnn(nn.Module):
    def __init__(self, device=DEVICE):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        self.to(device)
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))      
        x = F.relu(F.max_pool2d(self.conv2(x), 2))      
        x = x.view(-1, 320)                             
        x = F.relu(self.fc1(x))
        return F.log_softmax(self.fc2(x), dim=1)

# Dataset 
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,))
])
root = "./data"
train_full = datasets.FashionMNIST(root, train=True, download=True, transform=transform)
val_size = int(0.2 * len(train_full))
train_set, val_set = random_split(train_full, [len(train_full)-val_size, val_size])
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

model = Cnn(DEVICE)
optimzr = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

# Estimator + Tracker
est = TorchCNNLayerwiseEstimator(model, input_size=(1,1,28,28), training_factor=3.0)

ft = FlopsTracker(estimator=est, run_name="cnn").torch_bind(
    model=model,
    optimizer=optimzr,
    loss_fn=None,                 # usa F.nll_loss di default se None
    train_loader=train_loader,
    device=DEVICE,
    autosave_prefix="cnn_flops",  # salva csv a fine run
)

for _ in ft(range(EPOCHS)):
    pass

print("FLOPs totali training:", ft.total_flops)
