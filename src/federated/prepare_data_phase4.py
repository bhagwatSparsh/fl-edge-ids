"""
Phase 4 precompute: TRUE zero-day setup -- leave-one-family-out.

Phase 2/3 already proved the easier case: an attack that exists SOMEWHERE
in the federation (just not on a particular device) gets learned anyway
once that device joins federated training ("locally unseen"). Phase 4
tests the harder, more realistic case the plan calls "truly unseen": an
attack family that NO client, anywhere, ever saw during training.

For a family nobody has seen, a multiclass model literally cannot output
the right class name for it -- that label doesn't exist in its output
layer. The meaningful question (the plan's "binary head" framing) is
whether the model still recognizes the traffic as *something wrong*,
i.e. does it ever predict any non-benign class for it, rather than
confidently calling it benign. That's what run_phase4.py measures;
this file only builds the data.

Held-out families (one per dataset, chosen for size and for being a
clean, recognizable "family" rather than a rare edge case):
  - N-BaIoT: the entire Mirai family (5 classes: mirai.ack/scan/syn/udp/
    udpplain) -- removed from ALL 9 devices, not just devices 3 and 7
    (who never had it to begin with). Remaining problem: 6-class
    (benign + 5 gafgyt variants).
  - TON_IoT: "ransomware" (20,000 raw rows -- big enough for a reliable
    held-out test set, unlike "mitm" which only has 1,043). Remaining
    problem: 9-class.

Every remaining class is RE-ENCODED to a fresh 0..N-1 label space (benign
always index 0) -- this experiment doesn't need to match Phase 0-3's
original label ints, since those included the now-excluded family.
"""
import sys
sys.path.insert(0, "src")
import os
import json
import numpy as np
import pandas as pd

from data.pipeline import stratified_split, federated_scaler_stats, apply_scaler
from data.tonIoT import dirichlet_partition
from data.nbaiot import CLASS_TO_LABEL

OUT = "data/partitions/federated_phase4"

NBAIOT_HOLDOUT_FAMILY = "mirai"  # matches class name prefix in CLASS_TO_LABEL
TONIOT_HOLDOUT_CLASS = "ransomware"


def prepare_nbaiot_holdout():
    held_out_labels = {v for k, v in CLASS_TO_LABEL.items() if k.startswith(NBAIOT_HOLDOUT_FAMILY)}
    kept_names = sorted([k for k in CLASS_TO_LABEL if not k.startswith(NBAIOT_HOLDOUT_FAMILY)],
                         key=lambda k: CLASS_TO_LABEL[k])
    # benign must be index 0 for the binary-collapse metric in run_phase4.py
    assert kept_names[0] == "benign"
    new_label_map = {old_name: i for i, old_name in enumerate(kept_names)}
    old_to_new = {CLASS_TO_LABEL[name]: i for name, i in new_label_map.items()}

    X_global_test = pd.read_parquet("data/partitions/pooled_test.parquet")
    y_global_test = np.load("results/phase0_nbaiot_ytest.npy")
    feature_cols = X_global_test.columns.tolist()

    client_train = []
    for d in range(1, 10):
        df = pd.read_parquet(f"data/partitions/device_{d}.parquet")
        keep_mask = ~df["label"].isin(held_out_labels)
        Xc = df.loc[keep_mask, feature_cols]
        yc_old = df.loc[keep_mask, "label"].values
        yc = np.array([old_to_new[v] for v in yc_old], dtype=np.int64)
        X_tr, X_te, y_tr, y_te = stratified_split(Xc, yc, test_size=0.2, dedupe=True)
        client_train.append((X_tr, y_tr))
        print(f"  N-BaIoT client {d-1} (device {d}): n_train={len(y_tr)} "
              f"(mirai rows removed: {int((~keep_mask).sum())})")

    mean, std, n_total = federated_scaler_stats(
        [X.values.astype(np.float64) for X, _ in client_train]
    )
    for i, (X_tr, y_tr) in enumerate(client_train):
        X_scaled = apply_scaler(X_tr.values.astype(np.float64), mean, std).astype(np.float32)
        np.save(f"{OUT}/nbaiot_client{i}_X.npy", X_scaled)
        np.save(f"{OUT}/nbaiot_client{i}_y.npy", y_tr)

    # In-distribution test set: global test minus the held-out family, re-encoded.
    keep_mask_test = ~pd.Series(y_global_test).isin(held_out_labels)
    X_indist = X_global_test.loc[keep_mask_test.values]
    y_indist = np.array([old_to_new[v] for v in y_global_test[keep_mask_test.values]], dtype=np.int64)
    X_indist_scaled = apply_scaler(X_indist.values.astype(np.float64), mean, std).astype(np.float32)
    np.save(f"{OUT}/nbaiot_test_X.npy", X_indist_scaled)
    np.save(f"{OUT}/nbaiot_test_y.npy", y_indist)

    # The held-out "truly unseen" test set: pooled across ALL 9 devices'
    # mirai rows (not just the global test split's mirai rows -- more data,
    # and these rows were never used for ANY training, so using the full
    # pooled mirai set is fine).
    holdout_frames = []
    for d in range(1, 10):
        df = pd.read_parquet(f"data/partitions/device_{d}.parquet")
        hold_mask = df["label"].isin(held_out_labels)
        if hold_mask.any():
            holdout_frames.append(df.loc[hold_mask, feature_cols + ["label"]])
    X_holdout_df = pd.concat(holdout_frames, ignore_index=True)
    y_holdout_old = X_holdout_df.pop("label").values
    X_holdout_scaled = apply_scaler(X_holdout_df.values.astype(np.float64), mean, std).astype(np.float32)
    np.save(f"{OUT}/nbaiot_holdout_X.npy", X_holdout_scaled)
    np.save(f"{OUT}/nbaiot_holdout_y_original.npy", y_holdout_old)  # original mirai.* label, for a breakdown

    np.save(f"{OUT}/nbaiot_scaler_mean.npy", mean)
    np.save(f"{OUT}/nbaiot_scaler_std.npy", std)

    meta = {"n_clients": 9, "n_features": X_indist_scaled.shape[1], "n_classes": len(kept_names),
            "class_names": kept_names, "held_out_family": NBAIOT_HOLDOUT_FAMILY,
            "n_holdout_rows": len(y_holdout_old), "benign_index": 0,
            "scaler_fit_rows": int(n_total)}
    print(f"N-BaIoT: {len(kept_names)} remaining classes {kept_names}, "
          f"{len(y_holdout_old)} held-out Mirai rows for the zero-day test")
    return meta


def prepare_tonIoT_holdout(n_clients=9, alpha=0.5):
    from data.tonIoT import load_tonIoT
    from pathlib import Path

    raw_path = Path("/mnt/user-data/uploads/Computer_network_project/Dataset_2/train_test_network.csv")
    X_all, y_type_all, _ = load_tonIoT(raw_path)
    y_type_all = y_type_all.reset_index(drop=True)
    X_all = X_all.reset_index(drop=True)

    all_names = sorted(y_type_all.unique().tolist())
    kept_names = [c for c in all_names if c != TONIOT_HOLDOUT_CLASS]
    # benign/"normal" must be index 0
    kept_names = ["normal"] + [c for c in kept_names if c != "normal"]
    new_label_map = {name: i for i, name in enumerate(kept_names)}
    assert new_label_map["normal"] == 0

    keep_mask = y_type_all != TONIOT_HOLDOUT_CLASS
    X_train_full = X_all.loc[keep_mask.values].reset_index(drop=True)
    y_train_full = y_type_all.loc[keep_mask.values].reset_index(drop=True).map(new_label_map).values.astype(np.int64)
    print(f"  TON_IoT: {keep_mask.sum()} rows kept, {len(y_type_all) - keep_mask.sum()} "
          f"'{TONIOT_HOLDOUT_CLASS}' rows held out, {len(kept_names)} remaining classes {kept_names}")

    client_idx = dirichlet_partition(X_train_full, pd.Series(y_train_full),
                                      n_clients=n_clients, alpha=alpha, seed=42)

    client_train = []
    for i, idx in enumerate(client_idx):
        if len(idx) < 50:
            print(f"  TON_IoT client {i}: only {len(idx)} rows, skipping (matches Phase 2/3's skip rule)")
            continue
        Xc = X_train_full.iloc[idx].reset_index(drop=True)
        yc = y_train_full[idx]
        X_tr, X_te, y_tr, y_te = stratified_split(Xc, yc, test_size=0.2, dedupe=True)
        client_train.append((X_tr, y_tr))
        print(f"  TON_IoT client {i}: n_train={len(y_tr)}")

    mean, std, n_total = federated_scaler_stats(
        [X.values.astype(np.float64) for X, _ in client_train]
    )
    for new_i, (X_tr, y_tr) in enumerate(client_train):
        X_scaled = apply_scaler(X_tr.values.astype(np.float64), mean, std).astype(np.float32)
        np.save(f"{OUT}/tonIoT_client{new_i}_X.npy", X_scaled)
        np.save(f"{OUT}/tonIoT_client{new_i}_y.npy", y_tr)

    # In-distribution test set and the held-out-class test set: built fresh
    # from the full raw file rather than reused from Phase 0's saved test
    # set, because Phase 0's integer label encoding was never committed to
    # code (that one-off script's name<->int mapping isn't recoverable --
    # see PROGRESS.md), so rebuilding here with OUR OWN explicit
    # new_label_map keeps this phase self-contained and unambiguous.
    return _finish_tonIoT(X_all, y_type_all, mean, std, kept_names, new_label_map, n_total)


def _finish_tonIoT(X_all, y_type_all, mean, std, kept_names, new_label_map, n_total):
    """Build the in-distribution test set and the held-out-class test set
    from a fresh split of the full raw file (clean, unambiguous labels)."""
    keep_mask = (y_type_all != TONIOT_HOLDOUT_CLASS).values
    X_kept, y_kept = X_all.loc[keep_mask], y_type_all.loc[keep_mask].map(new_label_map).values.astype(np.int64)
    # A fresh held-out stratified split JUST for evaluation (separate from
    # the per-client train splits built above, which already consumed 80%
    # of each client's share -- this global in-distribution test set plays
    # the same role Phase 0-3's shared global test set did).
    _, X_indist_te, _, y_indist_te = stratified_split(X_kept, y_kept, test_size=0.2, dedupe=True)
    X_indist_scaled = apply_scaler(X_indist_te.values.astype(np.float64), mean, std).astype(np.float32)
    np.save(f"{OUT}/tonIoT_test_X.npy", X_indist_scaled)
    np.save(f"{OUT}/tonIoT_test_y.npy", y_indist_te)

    holdout_mask = (y_type_all == TONIOT_HOLDOUT_CLASS).values
    X_holdout = X_all.loc[holdout_mask]
    X_holdout_scaled = apply_scaler(X_holdout.values.astype(np.float64), mean, std).astype(np.float32)
    np.save(f"{OUT}/tonIoT_holdout_X.npy", X_holdout_scaled)
    np.save(f"{OUT}/tonIoT_scaler_mean.npy", mean)
    np.save(f"{OUT}/tonIoT_scaler_std.npy", std)

    meta = {"n_features": X_indist_scaled.shape[1], "n_classes": len(kept_names),
            "class_names": kept_names, "held_out_class": TONIOT_HOLDOUT_CLASS,
            "n_holdout_rows": int(holdout_mask.sum()), "benign_index": 0,
            "scaler_fit_rows": int(n_total)}
    print(f"TON_IoT: {len(kept_names)} remaining classes, {int(holdout_mask.sum())} "
          f"held-out '{TONIOT_HOLDOUT_CLASS}' rows for the zero-day test")
    return meta


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    print("="*60, "\nPreparing N-BaIoT leave-Mirai-out partitions\n", "="*60)
    nbaiot_meta = prepare_nbaiot_holdout()
    nbaiot_meta["n_clients"] = 9

    print("\n", "="*60, "\nPreparing TON_IoT leave-ransomware-out partitions\n", "="*60)
    tonIoT_meta = prepare_tonIoT_holdout()
    tonIoT_meta["n_clients"] = 9

    with open(f"{OUT}/meta.json", "w") as f:
        json.dump({"N-BaIoT": nbaiot_meta, "TON_IoT": tonIoT_meta}, f, indent=2)
    print("\nSaved Phase 4 federated partitions + scaler to", OUT)
