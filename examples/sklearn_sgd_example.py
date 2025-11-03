import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score

from flops_tracker import FlopsTracker
from flops_tracker.estimators import SklearnSGDEstimator

# Config
EPOCHS = 5
BATCH_SIZE = 128
RUN_NAME = "sgd_breast_cancer"


# Dataset
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

n_samples, n_features = X_train.shape
classes = np.unique(y)
n_classes = classes.size


# Modello SKLearn (SGD)
clf = SGDClassifier(
    loss="log_loss",          
    penalty="l2",
    alpha=1e-4,
    learning_rate="constant",
    eta0=0.01,
    max_iter=1,
    random_state=42
)

# Stimatore FLOPs SKLearn
# Formula (per-batch): ≈ 4 * B * n_features * n_classes
est = SklearnSGDEstimator(
    model=clf,
    n_features=n_features,
    n_classes=n_classes
)

# Tracker (modalità manuale)
# Salviamo sia logs per-epoch che per-batch (CSV a fine training).
ft = FlopsTracker(
    estimator=est,
    run_name=RUN_NAME,
    keep_epoch_logs=True,
    keep_batch_logs=True
)

def batch_iter(X, y, batch_size):
    n = X.shape[0]
    for i in range(0, n, batch_size):
        j = min(i + batch_size, n)
        yield X[i:j], y[i:j]

# Training
with ft: 
    for epoch in range(1, EPOCHS + 1):
        ft.on_epoch_start()

        # shuffle per-epoch
        idx = np.random.permutation(X_train.shape[0])
        X_train_shuf = X_train[idx]
        y_train_shuf = y_train[idx]

        for Xb, yb in batch_iter(X_train_shuf, y_train_shuf, BATCH_SIZE):
            if not hasattr(clf, "classes_"):
                clf.partial_fit(Xb, yb, classes=classes)
            else:
                clf.partial_fit(Xb, yb)
            ft.update_batch(batch_size=Xb.shape[0])

        ft.on_epoch_end()

        # stampa per-epoch 
        last = ft.epoch_logs[-1]
        print(f"[Epoch {last.epoch}] FLOPs_epoch={last.flops_epoch:,} | cum={last.flops_total_cum:,}")

# Export CSV 
ft.save_epoch_csv(f"{RUN_NAME}_epoch.csv")
ft.save_batch_csv(f"{RUN_NAME}_batch.csv")

print("FLOPs totali training:", f"{ft.total_flops:,}")

# Valutazione
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("Accuracy test:", f"{acc:.4f}")
