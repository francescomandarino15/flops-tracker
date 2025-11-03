# flops-tracker

Libreria per stimare i **FLOPs** (hardware-agnostic) durante il training di modelli MachineLearning/DeepLearning.  
Stile **CodeCarbon**, ma agnostico all'hardware, la metrica tracciata è il **compute** (FLOPs), non l'energia.

## Caratteristiche
- `FlopsTracker`: context manager per loggare FLOPs per **batch/epoch** e salvare CSV.
- `SklearnSGDEstimator`: stimatore per **SGDClassifier** (LogReg/lineare) → ~ `4 · B · f · C`.
- `TorchCNNLayerwiseEstimator`: stimatore layer-wise per **PyTorch CNN** (Conv2d/Linear) con **dry-run** per dedurre le shape; training ≈ `training_factor × forward` (default **3.0**).
- `TorchAutoEstimator(model, input_example=(1, C, H, W)`: stimatore per i principali layer di **PyTorch**

**Assunzioni**:
- 1 **MAC** = **2 FLOPs**.
- Layer contati: **Conv2d** e **Linear**; Pool/BN/Dropout trascurati (costo minore).
- Ultimo batch può essere più piccolo → si usa la dimensione **reale** del batch.

| Layer (`torch.nn`)                                                        | Formula (forward)                                                                                                              | Note                                                        |
| ------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------- |
| `Linear(n_in, n_out)`                                                     | `2 * n_in * n_out`                                                                                                             | 1 MAC = 2 FLOPs                                             |
| `Conv{1,2,3}d`                                                            | `2 * (∏ spatial_out) * C_out * ( (∏ K) * C_in / groups )`                                                                      | Vale anche per depthwise (`groups=C_in`)                    |
| `ConvTranspose{1,2,3}d`                                                   | `2 * (∏ spatial_out) * C_in * ( (∏ K) * C_out / groups )`                                                                      | Scambia (C_{in})/(C_{out})                                  |
| `AvgPool{d}`, `AdaptiveAvgPool{d}` *(opzionale)*                          | `#out_elements * (∏ K)`                                                                                                        | Attiva con `include_pooling_cost=True`                      |
| `MaxPool{d}`, `AdaptiveMaxPool{d}`                                        | ~0                                                                                                                             | Comparazioni trascurate                                     |
| `BatchNorm{d}`, `LayerNorm`, `GroupNorm`, `InstanceNorm{d}` *(opzionale)* | `≈ 5 * N`                                                                                                                      | `N = #elementi output`, attiva con `include_norm_cost=True` |
| `RNN` (tanh)                                                              | `2 * (I*H + H*H) * B * T`                                                                                                      | (I)=input, (H)=hidden, (B)=batch, (T)=seq len               |
| `GRU`                                                                     | `6 * (I*H + H*H) * B * T`                                                                                                      | 3 gate × 2 FLOPs/MAC                                        |
| `LSTM`                                                                    | `8 * (I*H + H*H) * B * T`                                                                                                      | 4 gate × 2 FLOPs/MAC                                        |
| `MultiheadAttention`                                                      | `Proj(Q,K,V): 3*2*(B L E·E)`  +  `QKᵀ: 2·B H L L d`  + *(softmax: `≈5·B H L L`)*  + `Attn·V: 2·B H L L d` + `Out: 2*(B L E·E)` | (E=H·d). Softmax opzionale con `include_softmax_cost=True`  |
| `Upsample/Interpolate`                                                    | `≈ #out_elements`                                                                                                              | Stima conservativa                                          |
| `Dropout`, `Embedding`, `Padding`, `Activation`, `Shuffle`                | ~0                                                                                                                             | Lookup/reshape/permute: costo trascurabile                  |


## API ad argomenti 
**Opzioni chiave**:
- print_Total_FLOPs ONE-LAYER
- print_level: "none" | "epoch" | "batch" | "both"
- export: "none" | "epoch" | "batch" | "both"
- export_prefix: prefisso file CSV (default = run_name)
- gestione dei log con wandb

```python
ft = FlopsTracker(est).torch_bind(model, optimizer, loss_fn=None, train_loader=train_loader, device=DEVICE)

# totale FLOPs ONE-LAYER (simulazione epoch dietro le quinte)
total = (FlopsTracker(est)
         .torch_bind(model, optimizer, loss_fn=None, train_loader=train_loader, device=DEVICE)
         .run(EPOCHS, print_level="none", export="none"))

print("FLOPs totali:", total)

# Solo totale FLOPs
for _ in ft(range(EPOCHS), print_level="none", export="none"):
    pass
print(ft.total_flops)

# Stampa per epoch + CSV epoch
for _ in ft(range(EPOCHS), print_level="epoch", export="epoch", export_prefix="run1"):
    pass  # crea run1_epoch.csv

# Stampa per batch+epoch + CSV batch+epoch
for _ in ft(range(EPOCHS), print_level="both", export="both", export_prefix="run1"):
    pass  # crea run1_batch.csv e run1_epoch.csv

# gestione dei log con wandb
total = (FlopsTracker(est)
         .torch_bind(model, optimizer, loss_fn=None, train_loader=train_loader, device=DEVICE)
         .run(EPOCHS,
              print_level="epoch",                    # stampa locale
              export="epoch", export_prefix="run1",   # CSV opzionali
              wandb=True,                             # <— abilita W&B
              wandb_project="flops-tracker",
              wandb_run_name="cnn-baseline",
              wandb_config={"model":"Cnn28x28","batch_size":128},
              wandb_log="epoch"))                     # "none" | "batch" | "epoch" | "both"
```

## Installazione
Ambiente virtuale consigliato.

```bash
pip install -e .[torch,sklearn]
# Oppure solo una delle due:
# pip install -e .[torch]
# pip install -e .[sklearn]
```

