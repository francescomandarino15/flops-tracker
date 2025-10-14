# flops-tracker

Libreria per stimare i **FLOPs** (hardware-agnostic) durante il training di modelli ML/DL.  
Stile **CodeCarbon**, ma la metrica tracciata è il **compute** (FLOPs), non l'energia.

## Caratteristiche
- `FlopsTracker`: context manager per loggare FLOPs per **batch/epoch** e salvare CSV.
- `SklearnSGDEstimator`: stimatore per **SGDClassifier** (LogReg/lineare) → ~ `4 · B · f · C`.
- `TorchCNNLayerwiseEstimator`: stimatore layer-wise per **PyTorch CNN** (Conv2d/Linear) con **dry-run** per dedurre le shape; training ≈ `training_factor × forward` (default **3.0**).

**Assunzioni**:
- 1 **MAC** = **2 FLOPs**.
- Layer contati: **Conv2d** e **Linear**; Pool/BN/Dropout trascurati (costo minore).
- Ultimo batch può essere più piccolo → si usa la dimensione **reale** del batch.

## Installazione
Ambiente virtuale consigliato.

```bash
pip install -e .[torch,sklearn]
# Oppure solo una delle due:
# pip install -e .[torch]
# pip install -e .[sklearn]


