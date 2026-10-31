"""A quick, human-readable "does it actually work" test: load the real
Phase 3 federated model (N-BaIoT, FedProx -- the best config, macro-F1
0.7463 on the full held-out test set) and run it on a real held-out
batch of traffic the model has never seen during training, printing
what it guessed vs what it actually was.

This is NOT a new experiment -- it's just making Phase 3's already-
measured result visible and checkable by hand, sample by sample,
instead of as a single aggregate number in a JSON file.
"""
import sys
sys.path.insert(0, "src")
import numpy as np
import torch

from models.mlp import MLP

NBAIOT_CLASSES = ["benign", "gafgyt.combo", "gafgyt.junk", "gafgyt.scan", "gafgyt.tcp",
                   "gafgyt.udp", "mirai.ack", "mirai.scan", "mirai.syn", "mirai.udp",
                   "mirai.udpplain"]

X_test = np.load("data/partitions/federated/nbaiot_test_X.npy")
y_test = np.load("data/partitions/federated/nbaiot_test_y.npy")

model = MLP(n_features=115, n_classes=11)
model.load_state_dict(torch.load("results/phase3_nbaiot_FedProx.pt"))
model.eval()

rng = np.random.default_rng(42)
idx = rng.choice(len(X_test), size=20, replace=False)

Xb = torch.tensor(X_test[idx], dtype=torch.float32)
yb = y_test[idx]

with torch.no_grad():
    logits = model(Xb)
    preds = logits.argmax(dim=1).numpy()

print(f"{'#':<3} {'ACTUAL (real label)':<22} {'MODEL GUESSED':<22} {'RESULT'}")
print("-" * 70)
n_correct = 0
for i, (p, y) in enumerate(zip(preds, yb)):
    correct = (p == y)
    n_correct += correct
    mark = "correct" if correct else "WRONG"
    print(f"{i:<3} {NBAIOT_CLASSES[y]:<22} {NBAIOT_CLASSES[p]:<22} {mark}")

print("-" * 70)
print(f"{n_correct}/20 correct on this random sample of real, never-trained-on traffic")

# Full held-out test set accuracy, for context (this is the real number
# behind Phase 3's reported macro-F1, computed the plain way: overall %
# correct across all ~N rows, not just this 20-row printable sample).
with torch.no_grad():
    all_logits = model(torch.tensor(X_test, dtype=torch.float32))
    all_preds = all_logits.argmax(dim=1).numpy()
overall_acc = (all_preds == y_test).mean()
print(f"\nOverall accuracy across the FULL held-out test set ({len(y_test)} rows): {overall_acc:.1%}")
