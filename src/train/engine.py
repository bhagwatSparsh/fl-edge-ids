"""Shared train/eval loop (plain PyTorch, no Flower imports -- plan's
'vanilla first, framework second' principle). The exact same functions run
centralized here in Phase 1 and get wrapped by Flower's ClientApp in Phase 3
without changing a line of this file."""
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, recall_score, confusion_matrix


def make_loaders(X_train, y_train, X_val, y_val, batch_size=512):
    train_ds = TensorDataset(torch.as_tensor(X_train, dtype=torch.float32),
                              torch.as_tensor(y_train, dtype=torch.long))
    val_ds = TensorDataset(torch.as_tensor(X_val, dtype=torch.float32),
                            torch.as_tensor(y_val, dtype=torch.long))
    return (DataLoader(train_ds, batch_size=batch_size, shuffle=True),
            DataLoader(val_ds, batch_size=batch_size, shuffle=False))


def class_weights(y, n_classes):
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    counts = np.clip(counts, 1, None)
    w = counts.sum() / (n_classes * counts)
    return torch.as_tensor(w, dtype=torch.float32)


def train_one_epoch(model, loader, optimizer, criterion, device="cpu"):
    model.train()
    total_loss = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, n_classes, device="cpu"):
    model.eval()
    all_preds, all_true = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.append(preds)
        all_true.append(yb.numpy())
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_true)

    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    per_class_recall = recall_score(y_true, y_pred, average=None,
                                     labels=range(n_classes), zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=range(n_classes))
    # FPR per class: FP / (FP + TN) from the one-vs-rest confusion matrix
    fpr = np.zeros(n_classes)
    total = cm.sum()
    for c in range(n_classes):
        fp = cm[:, c].sum() - cm[c, c]
        tn = total - cm[c, :].sum() - cm[:, c].sum() + cm[c, c]
        fpr[c] = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {
        "macro_f1": float(macro_f1),
        "per_class_recall": per_class_recall.tolist(),
        "per_class_fpr": fpr.tolist(),
        "confusion_matrix": cm.tolist(),
    }


def fit(model, X_train, y_train, X_val, y_val, n_classes, epochs=15,
        patience=3, lr=1e-3, batch_size=512, device="cpu", verbose=True):
    """Adam + class-weighted CE + early stopping on val macro-F1, per plan
    Phase 1: 'Class-weighted cross-entropy, Adam, early stopping on a
    validation slice of train.'"""
    model.to(device)
    train_loader, val_loader = make_loaders(X_train, y_train, X_val, y_val, batch_size)
    weights = class_weights(y_train, n_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_f1 = -1.0
    best_state = None
    no_improve = 0
    history = []
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, n_classes, device)
        history.append({"epoch": epoch, "train_loss": train_loss,
                         "val_macro_f1": val_metrics["macro_f1"]})
        if verbose:
            print(f"  epoch {epoch}: train_loss={train_loss:.4f} val_macro_f1={val_metrics['macro_f1']:.4f}")

        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                if verbose:
                    print(f"  early stop at epoch {epoch} (no improvement for {patience} epochs)")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    elapsed = time.time() - t0
    return model, history, elapsed
