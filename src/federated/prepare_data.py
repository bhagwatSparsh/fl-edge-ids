"""
Phase 3 precompute step.

Builds the per-client federated training data for the Flower simulation,
reusing Phase 0's pipeline functions exactly as-is (no changes to
src/data/pipeline.py -- the "vanilla first, framework second" rule applies
to the data pipeline too, not just the model code).

Key design choice: all 9 clients, in both datasets, use the SAME global
federated scaler (mean/std combined centrally from each client's own
(count, sum, sum_of_squares) -- never raw rows, exactly Phase 0's
federated_scaler_stats()). This is deliberately different from Phase 2,
where each client used its OWN local scaler (the "no coordination
whatsoever" lower bound). A real federated system fits the scaler this way
too, as one extra aggregation round before training starts -- it's the
correct, fair setup for Phase 3, and it makes Phase 3 comparable to
Phase 1's centralized run (which also used one global scaler) rather than
to Phase 2's per-client scalers.

N-BaIoT clients = the 9 real devices (unchanged from Phase 2).
TON_IoT clients = the SAME Dirichlet partition as Phase 2
(n_clients=9, alpha=0.5, seed=42) so the floor (Phase 2) and the federated
result (Phase 3) are measured on identical client splits -- a fair
before/after comparison, not a different random partition.

Outputs (gitignored, like all of data/partitions/):
  data/partitions/federated/{dataset}_client{i}_X.npy / _y.npy
  data/partitions/federated/{dataset}_test_X.npy / _y.npy   (shared global test, scaled once)
  data/partitions/federated/{dataset}_scaler_mean.npy / _std.npy
"""
import sys
sys.path.insert(0, "src")
import os
import json
import numpy as np
import pandas as pd

from data.pipeline import stratified_split, federated_scaler_stats, apply_scaler
from data.tonIoT import dirichlet_partition

OUT = "data/partitions/federated"


def prepare_nbaiot():
    X_global_test = pd.read_parquet("data/partitions/pooled_test.parquet")
    y_global_test = np.load("results/phase0_nbaiot_ytest.npy")
    feature_cols = X_global_test.columns.tolist()

    client_train = []  # list of (X_tr_df, y_tr)
    for d in range(1, 10):
        df = pd.read_parquet(f"data/partitions/device_{d}.parquet")
        Xc = df[feature_cols]
        yc = df["label"].values
        X_tr, X_te, y_tr, y_te = stratified_split(Xc, yc, test_size=0.2, dedupe=True)
        client_train.append((X_tr, y_tr))
        print(f"  N-BaIoT client {d-1} (device {d}): n_train={len(y_tr)}")

    mean, std, n_total = federated_scaler_stats(
        [X.values.astype(np.float64) for X, _ in client_train]
    )
    print(f"N-BaIoT federated scaler fit on {int(n_total)} rows total across 9 clients")

    for i, (X_tr, y_tr) in enumerate(client_train):
        X_scaled = apply_scaler(X_tr.values.astype(np.float64), mean, std).astype(np.float32)
        np.save(f"{OUT}/nbaiot_client{i}_X.npy", X_scaled)
        np.save(f"{OUT}/nbaiot_client{i}_y.npy", y_tr.astype(np.int64))

    X_test_scaled = apply_scaler(X_global_test.values.astype(np.float64), mean, std).astype(np.float32)
    np.save(f"{OUT}/nbaiot_test_X.npy", X_test_scaled)
    np.save(f"{OUT}/nbaiot_test_y.npy", y_global_test.astype(np.int64))
    np.save(f"{OUT}/nbaiot_scaler_mean.npy", mean)
    np.save(f"{OUT}/nbaiot_scaler_std.npy", std)
    return {"n_clients": 9, "n_features": X_test_scaled.shape[1], "n_classes": 11,
            "scaler_fit_rows": int(n_total)}


def prepare_tonIoT(n_clients=9, alpha=0.5):
    X_train_full = pd.read_parquet("data/partitions/tonIoT_train.parquet")
    y_train_full = np.load("results/phase0_tonIoT_ytrain.npy")
    X_global_test = pd.read_parquet("data/partitions/tonIoT_test.parquet")
    y_global_test = np.load("results/phase0_tonIoT_ytest.npy")

    # Same partition as Phase 2 (local_only.run_tonIoT): identical n_clients,
    # alpha, seed -- so Phase 2 (floor) and Phase 3 (federated) are measured
    # on the exact same per-client data, not a different random draw.
    client_idx = dirichlet_partition(X_train_full, pd.Series(y_train_full),
                                      n_clients=n_clients, alpha=alpha, seed=42)

    client_train = []
    kept_client_ids = []
    for i, idx in enumerate(client_idx):
        if len(idx) < 50:
            print(f"  TON_IoT client {i}: only {len(idx)} rows, skipping (matches Phase 2's skip rule)")
            continue
        Xc = X_train_full.iloc[idx].reset_index(drop=True)
        yc = y_train_full[idx]
        X_tr, X_te, y_tr, y_te = stratified_split(Xc, yc, test_size=0.2, dedupe=True)
        client_train.append((X_tr, y_tr))
        kept_client_ids.append(i)
        print(f"  TON_IoT client {i}: n_train={len(y_tr)}")

    mean, std, n_total = federated_scaler_stats(
        [X.values.astype(np.float64) for X, _ in client_train]
    )
    print(f"TON_IoT federated scaler fit on {int(n_total)} rows total across {len(client_train)} clients")

    for new_i, (X_tr, y_tr) in enumerate(client_train):
        X_scaled = apply_scaler(X_tr.values.astype(np.float64), mean, std).astype(np.float32)
        np.save(f"{OUT}/tonIoT_client{new_i}_X.npy", X_scaled)
        np.save(f"{OUT}/tonIoT_client{new_i}_y.npy", y_tr.astype(np.int64))

    X_test_scaled = apply_scaler(X_global_test.values.astype(np.float64), mean, std).astype(np.float32)
    np.save(f"{OUT}/tonIoT_test_X.npy", X_test_scaled)
    np.save(f"{OUT}/tonIoT_test_y.npy", y_global_test.astype(np.int64))
    np.save(f"{OUT}/tonIoT_scaler_mean.npy", mean)
    np.save(f"{OUT}/tonIoT_scaler_std.npy", std)
    return {"n_clients": len(client_train), "original_client_ids": kept_client_ids,
            "n_features": X_test_scaled.shape[1], "n_classes": 10,
            "scaler_fit_rows": int(n_total)}


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    print("="*60, "\nPreparing N-BaIoT federated partitions\n", "="*60)
    nbaiot_meta = prepare_nbaiot()
    print("\n", "="*60, "\nPreparing TON_IoT federated partitions\n", "="*60)
    tonIoT_meta = prepare_tonIoT()

    with open(f"{OUT}/meta.json", "w") as f:
        json.dump({"N-BaIoT": nbaiot_meta, "TON_IoT": tonIoT_meta}, f, indent=2)
    print("\nSaved per-client federated partitions + scaler to", OUT)
