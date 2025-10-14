import math
import csv
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from flops_tracker import FlopsTracker, SklearnSGDEstimator

# parametri
batch_size = 256
epochs = 5
random_state = 42
loss = "log_loss"
alpha = 1e-4
learning_rate = 1e-2


# DataSet
X, y = load_breast_cancer(return_X_y=True)
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.33, random_state=random_state, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.3882, random_state=random_state, stratify=y_temp)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

m, f = X_train.shape

# Modello
clf = SGDClassifier(
    loss=loss,
    alpha=alpha,
    learning_rate="constant",
    eta0=learning_rate,
    random_state=random_state,
    warm_start=True,
    max_iter = 1, tol = None
)

classes = np.unique(y_train)
init_B = min(batch_size, X_train.shape[0])
clf.partial_fit(X_train[:init_B], y_train[:init_B], classes=classes)

# Tracker
est = SklearnSGDEstimator(n_features=f, n_classes=1)
with FlopsTracker(estimator=est, run_name="sgd_breast_cancer") as ft:
  for epoch in range(1, epochs+1):
    ft.on_epoch_start()
    idx = np.random.permutation(m)
    X_train = X_train[idx]
    y_train = y_train[idx]
    steps = math.ceil(m / batch_size)
    for step in range(steps):
      s, e = step * batch_size, min((step + 1) * batch_size, m)
      Xb, yb = X_train[s:e], y_train[s:e]
      clf.partial_fit(Xb, yb)
      ft.update_batch(batch_size=len(Xb))
    ft.on_epoch_end()
    acc = accuracy_score(y_val, clf.predict(X_val))
    print(f"[Epoch {epoch}] acc = {acc:.4f} | FLOPs_epoch = {ft.epoch_logs[-1].flops_epoch:} | cum = {ft.total_flops:}")

ft.save_batch_csv("sgd_flops_batch.csv")
ft.save_epoch_csv("sgd_flops_epoch.csv")
print("FLOPs totali:", ft.total_flops)
