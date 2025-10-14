import math, csv
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from flops_tracker import FlopsTracker, SklearnSGDEstimator

RANDOM_STATE = 42
BATCH_SIZE = 256
EPOCHS = 5
LR = 1e-2
ALPHA = 1e-4

# Dataset
X, y = load_breast_cancer(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
scaler = StandardScaler()
X_tr = scaler.fit_transform(X_tr); X_te = scaler.transform(X_te)
m, f = X_tr.shape

# Modello
clf = SGDClassifier(loss="log_loss", alpha=ALPHA, learning_rate="constant", eta0=LR,
                    max_iter=1, tol=None, random_state=RANDOM_STATE, warm_start=True)
classes = np.unique(y_tr)
clf.partial_fit(X_tr[:min(BATCH_SIZE, m)], y_tr[:min(BATCH_SIZE, m)], classes=classes)

# Tracker
est = SklearnSGDEstimator(n_features=f, n_classes=1)
with FlopsTracker(estimator=est, run_name="sgd_breast_cancer") as ft:
    for epoch in range(1, EPOCHS+1):
        ft.on_epoch_start()
        # shuffle per epoch
        idx = np.random.permutation(m)
        X_tr, y_tr = X_tr[idx], y_tr[idx]
        steps = math.ceil(m / BATCH_SIZE)
        for step in range(steps):
            s, e = step * BATCH_SIZE, min((step+1)*BATCH_SIZE, m)
            Xb, yb = X_tr[s:e], y_tr[s:e]
            clf.partial_fit(Xb, yb)
            ft.update_batch(batch_size=len(Xb))
        ft.on_epoch_end()
        acc = accuracy_score(y_te, clf.predict(X_te))
        print(f"[Epoch {epoch}] acc={acc:.4f} | FLOPs_epoch={ft.epoch_logs[-1].flops_epoch:,} | cum={ft.total_flops:,}")

ft.save_batch_csv("sgd_flops_batch.csv")
ft.save_epoch_csv("sgd_flops_epoch.csv")
print("FLOPs totali:", ft.total_flops)
