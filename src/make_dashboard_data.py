"""Builds the real-data JSON files for the final results dashboard.
Nothing here is simulated: every row is a real traffic record from the
real dataset, run through the real trained Phase 3 FedProx checkpoint
(the actual federated global model).
"""
import sys
sys.path.insert(0, "src")
import json
import numpy as np
import pandas as pd
import torch

from models.mlp import MLP
from data.pipeline import apply_scaler
from data.nbaiot import DEVICE_NAMES

NBAIOT_CLASSES = ["benign", "gafgyt.combo", "gafgyt.junk", "gafgyt.scan", "gafgyt.tcp",
                   "gafgyt.udp", "mirai.ack", "mirai.scan", "mirai.syn", "mirai.udp",
                   "mirai.udpplain"]
MIRAI_SET = {"mirai.ack", "mirai.scan", "mirai.syn", "mirai.udp", "mirai.udpplain"}
LOCALLY_BLIND_DEVICES = {3, 7}  # Ennio_Doorbell, Samsung_Webcam -- zero Mirai locally (Phase 2)

mean = np.load("results/phase0_nbaiot_federated_mean.npy")
std = np.load("results/phase0_nbaiot_federated_std.npy")

model = MLP(n_features=115, n_classes=11)
model.load_state_dict(torch.load("results/phase3_nbaiot_FedProx.pt"))
model.eval()

rng = np.random.default_rng(7)
rows = []
seq = 0
for d in range(1, 10):
    df = pd.read_parquet(f"data/partitions/device_{d}.parquet")
    feature_cols = [c for c in df.columns if c not in ("label", "device_id")]
    # a mixed sample per device: 3 rows of each real class present, so the
    # animation shows a genuine mix of benign + every attack family per
    # device rather than whatever happened to be most common.
    idx_parts = []
    for lbl, grp in df.groupby("label"):
        idx_parts.append(grp.sample(min(len(grp), 3), random_state=7).index)
    sample = df.loc[np.concatenate(idx_parts)].reset_index(drop=True)
    X = sample[feature_cols].values.astype(np.float64)
    X = apply_scaler(X, mean, std).astype(np.float32)
    y = sample["label"].values

    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32))
        preds = logits.argmax(dim=1).numpy()

    for i in range(len(sample)):
        actual = NBAIOT_CLASSES[y[i]]
        predicted = NBAIOT_CLASSES[preds[i]]
        rows.append({
            "seq": seq,
            "device_id": int(d),
            "device_name": DEVICE_NAMES[d],
            "actual_label": actual,
            "predicted_label": predicted,
            "correct": bool(actual == predicted),
            "is_attack": bool(actual != "benign"),
            "attack_family": actual.split(".")[0] if actual != "benign" else "none",
            "device_locally_blind_to_mirai": d in LOCALLY_BLIND_DEVICES,
            "is_mirai": actual in MIRAI_SET,
        })
        seq += 1

rng.shuffle(rows)
for i, r in enumerate(rows):
    r["seq"] = i

print(f"Generated {len(rows)} real traffic rows across all 9 real devices.")
n_correct = sum(r["correct"] for r in rows)
print(f"Model correct on {n_correct}/{len(rows)} ({n_correct/len(rows):.1%})")
blind_mirai = [r for r in rows if r["device_locally_blind_to_mirai"] and r["is_mirai"]]
blind_mirai_caught = sum(r["correct"] or r["predicted_label"] != "benign" for r in blind_mirai)
print(f"Mirai rows from the 2 locally-Mirai-blind devices: {len(blind_mirai)}, "
      f"flagged as some attack by the federated model: {blind_mirai_caught}")

with open("results/dashboard_sample_traffic.json", "w") as f:
    json.dump(rows, f, indent=2)
print("Saved results/dashboard_sample_traffic.json")
