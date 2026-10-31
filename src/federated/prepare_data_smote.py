"""
Phase 5 precompute: client-side SMOTE, exactly as the literature-review
deck specifies ("Non-IID Dirichlet split -> SMOTE balancing -> 1D-CNN+
BiLSTM -> FedProx -> compression", slide 5; "client-side SMOTE / synthetic
balancing before local training", slide 8's named fix for class
imbalance).

Applied PER CLIENT, on each client's own already-scaled Phase 3 training
data (data/partitions/federated/*_client{i}_*.npy) -- SMOTE generates
synthetic minority-class rows by interpolating between a real minority
sample and its nearest same-class neighbors in feature space, so it needs
to run on data that's already on a comparable scale (hence after scaling,
not before), and strictly per client (never pooling clients' data, or this
would violate the whole point of federated learning).

Standard SMOTE assumes every class has at least k_neighbors+1 members; a
class with only 1-4 samples (very possible in TON_IoT's small Dirichlet
clients -- see below) breaks that assumption. safe_client_smote() handles
this explicitly: k_neighbors is chosen per client from its OWN smallest
minority class, and a class with a single sample (no possible neighbor)
is first duplicated once via plain oversampling so SMOTE has something to
interpolate from. This is a deliberate, documented adaptation, not a
silent workaround -- the per-client before/after class counts are saved
to smote_report.json so the effect (and how aggressive it had to be) is
auditable.

Finding expected (and confirmed below once run): N-BaIoT's clients are
already near-balanced per class (Phase 0's load_device() caps every class
at 20k rows per device, so min/max class ratio per client is already
close to 1:1) -- SMOTE should do almost nothing there. TON_IoT's Dirichlet
clients were never capped this way and have real skew (some clients have
a class with as few as 2-10 rows against a majority class of ~2000-7000)
-- this is where SMOTE is actually expected to matter.
"""
import sys
sys.path.insert(0, "src")
import json
from collections import Counter

import numpy as np
from imblearn.over_sampling import SMOTE, RandomOverSampler

IN_DIR = "data/partitions/federated"
OUT_DIR = "data/partitions/federated_smote"


def safe_client_smote(X, y, seed=42):
    """SMOTE balance-to-majority, robust to classes too small for the
    standard k_neighbors=5 default. Returns (X_res, y_res, report_dict)."""
    counts = Counter(y.tolist())
    max_count = max(counts.values())
    if len(set(counts.values())) == 1:
        return X, y, {"applied": False, "reason": "already perfectly balanced",
                        "before": {int(k): int(v) for k, v in counts.items()}}

    # A class with exactly 1 sample has no neighbor to interpolate with --
    # duplicate it once first so SMOTE has 2+ points to work from.
    singleton_classes = [c for c, n in counts.items() if n == 1]
    if singleton_classes:
        ros = RandomOverSampler(sampling_strategy={c: 2 for c in singleton_classes}, random_state=seed)
        X, y = ros.fit_resample(X, y)
        counts = Counter(y.tolist())

    minority_classes = [c for c, n in counts.items() if n < max_count]
    if not minority_classes:
        return X, y, {"applied": False, "reason": "balanced after singleton bump",
                        "before": {int(k): int(v) for k, v in counts.items()}}

    smallest_minority = min(counts[c] for c in minority_classes)
    k = max(1, min(5, smallest_minority - 1))
    before = {int(k_): int(v) for k_, v in counts.items()}
    try:
        sm = SMOTE(k_neighbors=k, random_state=seed)
        X_res, y_res = sm.fit_resample(X, y)
        after = {int(k_): int(v) for k_, v in Counter(y_res.tolist()).items()}
        return X_res, y_res, {"applied": True, "k_neighbors": k, "before": before, "after": after}
    except ValueError as e:
        ros = RandomOverSampler(random_state=seed)
        X_res, y_res = ros.fit_resample(X, y)
        after = {int(k_): int(v) for k_, v in Counter(y_res.tolist()).items()}
        return X_res, y_res, {"applied": True, "method": "RandomOverSampler_fallback",
                                "fallback_reason": str(e), "before": before, "after": after}


def process_dataset(dataset_tag, n_clients):
    report = {}
    for i in range(n_clients):
        X = np.load(f"{IN_DIR}/{dataset_tag}_client{i}_X.npy")
        y = np.load(f"{IN_DIR}/{dataset_tag}_client{i}_y.npy")
        X_res, y_res, info = safe_client_smote(X, y)
        np.save(f"{OUT_DIR}/{dataset_tag}_client{i}_X.npy", X_res.astype(np.float32))
        np.save(f"{OUT_DIR}/{dataset_tag}_client{i}_y.npy", y_res.astype(np.int64))
        report[f"client{i}"] = {"n_before": int(len(y)), "n_after": int(len(y_res)), **info}
        tag = info.get("method", "SMOTE") if info.get("applied") else "skipped"
        print(f"  {dataset_tag} client {i}: {len(y)} -> {len(y_res)} rows ({tag})")
    return report


if __name__ == "__main__":
    import os
    os.makedirs(OUT_DIR, exist_ok=True)

    with open(f"{IN_DIR}/meta.json") as f:
        meta = json.load(f)

    print("="*60, "\nSMOTE-balancing N-BaIoT clients\n", "="*60)
    nbaiot_report = process_dataset("nbaiot", meta["N-BaIoT"]["n_clients"])

    print("\n", "="*60, "\nSMOTE-balancing TON_IoT clients\n", "="*60)
    tonIoT_report = process_dataset("tonIoT", meta["TON_IoT"]["n_clients"])

    # Test sets and scaler are untouched by SMOTE (it only ever rebalances
    # TRAINING rows) -- just copy them over so run_phase5.py has everything
    # it needs in one directory, matching Phase 3's layout.
    import shutil
    for tag in ["nbaiot", "tonIoT"]:
        for suffix in ["test_X.npy", "test_y.npy", "scaler_mean.npy", "scaler_std.npy"]:
            shutil.copy(f"{IN_DIR}/{tag}_{suffix}", f"{OUT_DIR}/{tag}_{suffix}")
    shutil.copy(f"{IN_DIR}/meta.json", f"{OUT_DIR}/meta.json")

    with open(f"{OUT_DIR}/smote_report.json", "w") as f:
        json.dump({"N-BaIoT": nbaiot_report, "TON_IoT": tonIoT_report}, f, indent=2)
    print("\nSaved SMOTE-balanced client data + smote_report.json to", OUT_DIR)
