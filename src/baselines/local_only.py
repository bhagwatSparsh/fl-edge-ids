"""
Phase 2 gate: local-only baseline. Each client trains ONLY on its own
partition -- no cross-client coordination of any kind, not even the
federated scaler (that would itself be a form of collaboration). Each
client fits its own local scaler from its own local train split only.

Evaluated on:
  (a) the client's own local test split (how good is this device's model
      for the device itself?)
  (b) the shared GLOBAL test set, scaled through that SAME client's own
      local scaler (this is what the client would actually do if deployed
      as-is -- it has no way to know the global feature statistics)

This produces the LOWER BOUND (nothing in Phase 3's federated results
should do worse than this) and the "before" picture for Phase 4: devices
3 (Ennio_Doorbell) and 7 (Samsung_Webcam) have zero Mirai in their own
data, so their local-only models should fail hard on Mirai when evaluated
against the global test set, which does contain Mirai.

Model: MLP only (see PROGRESS.md for why -- CNN-BiLSTM is ~60x slower to
train and was not the clear winner in Phase 1, so running it per-client
x9 x2 datasets was not a good use of time for this gate check).
"""
import sys
sys.path.insert(0, "src")
import argparse
import json
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from models.mlp import MLP
from train.engine import fit, evaluate, make_loaders
from data.pipeline import stratified_split, federated_scaler_stats, apply_scaler
from data.tonIoT import dirichlet_partition
from data.nbaiot import DEVICE_NAMES


def run_one_client(client_id, X_client_df, y_client, X_global_test, y_global_test,
                    n_classes, epochs=15, batch_size=256, label="", seed=42):
    """Train MLP on this client's own data only, with this client's own
    local scaler. Evaluate on (a) its own local test slice, (b) the global
    test set through its OWN scaler."""
    X_tr_raw, X_te_raw, y_tr, y_te = stratified_split(
        X_client_df, y_client, test_size=0.2, dedupe=True, seed=seed
    )

    # Local-only scaler: computed from THIS client's train split alone.
    mean, std, n = federated_scaler_stats([X_tr_raw.values.astype(np.float64)])
    X_tr = apply_scaler(X_tr_raw.values.astype(np.float64), mean, std).astype(np.float32)
    X_te_local = apply_scaler(X_te_raw.values.astype(np.float64), mean, std).astype(np.float32)
    X_te_global = apply_scaler(X_global_test.values.astype(np.float64), mean, std).astype(np.float32)

    # carve a small validation slice for early stopping
    counts = pd.Series(y_tr).value_counts()
    can_stratify = (counts >= 2).all() and len(counts) > 1
    if can_stratify and len(y_tr) > 20:
        X_fit, X_val, y_fit, y_val = train_test_split(
            X_tr, y_tr, test_size=0.15, random_state=seed, stratify=y_tr
        )
    else:
        X_fit, X_val, y_fit, y_val = X_tr, X_tr, y_tr, y_tr

    torch.manual_seed(seed)
    model = MLP(n_features=X_tr.shape[1], n_classes=n_classes)
    model, history, elapsed = fit(model, X_fit, y_fit, X_val, y_val, n_classes,
                                   epochs=epochs, batch_size=batch_size, verbose=False)

    _, local_loader = make_loaders(X_tr[:1], y_tr[:1], X_te_local, y_te, batch_size=batch_size)
    local_metrics = evaluate(model, local_loader, n_classes)

    _, global_loader = make_loaders(X_tr[:1], y_tr[:1], X_te_global, y_global_test, batch_size=batch_size)
    global_metrics = evaluate(model, global_loader, n_classes)

    print(f"  client {client_id} {label}: n_train={len(y_tr)} | "
          f"local-test macro-F1={local_metrics['macro_f1']:.4f} | "
          f"global-test macro-F1={global_metrics['macro_f1']:.4f} | {elapsed:.1f}s")

    return {
        "client_id": client_id,
        "label": label,
        "n_train": len(y_tr),
        "n_local_test": len(y_te),
        "local_test_macro_f1": local_metrics["macro_f1"],
        "global_test_macro_f1": global_metrics["macro_f1"],
        "global_test_per_class_recall": global_metrics["per_class_recall"],
        "global_test_per_class_fpr": global_metrics["per_class_fpr"],
        "epochs_run": len(history),
    }


def run_nbaiot(seed=42):
    NBAIOT_CLASSES = ["benign", "gafgyt.combo", "gafgyt.junk", "gafgyt.scan", "gafgyt.tcp",
                       "gafgyt.udp", "mirai.ack", "mirai.scan", "mirai.syn", "mirai.udp",
                       "mirai.udpplain"]
    mirai_idx = [i for i, c in enumerate(NBAIOT_CLASSES) if c.startswith("mirai")]

    X_global_test = pd.read_parquet("data/partitions/pooled_test.parquet")
    y_global_test = np.load("results/phase0_nbaiot_ytest.npy")
    feature_cols = X_global_test.columns.tolist()

    print(f"\n{'='*60}\nN-BaIoT local-only baseline (9 natural clients)\n{'='*60}")
    results = []
    for d in range(1, 10):
        df = pd.read_parquet(f"data/partitions/device_{d}.parquet")
        Xc = df[feature_cols]
        yc = df["label"].values
        r = run_one_client(d, Xc, yc, X_global_test, y_global_test, n_classes=11,
                            label=DEVICE_NAMES[d], seed=seed)
        r["mirai_recall_on_global_test"] = [r["global_test_per_class_recall"][i] for i in mirai_idx]
        r["mirai_recall_mean"] = float(np.mean(r["mirai_recall_on_global_test"]))
        results.append(r)

    return results


def run_tonIoT(n_clients=9, alpha=0.5, seed=42):
    X_train_full = pd.read_parquet("data/partitions/tonIoT_train.parquet")
    y_train_full = np.load("results/phase0_tonIoT_ytrain.npy")
    X_global_test = pd.read_parquet("data/partitions/tonIoT_test.parquet")
    y_global_test = np.load("results/phase0_tonIoT_ytest.npy")

    client_idx = dirichlet_partition(X_train_full, pd.Series(y_train_full),
                                      n_clients=n_clients, alpha=alpha, seed=seed)

    print(f"\n{'='*60}\nTON_IoT local-only baseline ({n_clients} Dirichlet clients, alpha={alpha}, seed={seed})\n{'='*60}")
    results = []
    for i, idx in enumerate(client_idx):
        if len(idx) < 50:
            print(f"  client {i}: only {len(idx)} rows, skipping (too small to train/evaluate meaningfully)")
            continue
        Xc = X_train_full.iloc[idx].reset_index(drop=True)
        yc = y_train_full[idx]
        r = run_one_client(i, Xc, yc, X_global_test, y_global_test, n_classes=10,
                            label=f"dirichlet_client_{i}", seed=seed)
        results.append(r)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    out_suffix = "" if args.seed == 42 else f"_seed{args.seed}"

    nbaiot_results = run_nbaiot(seed=args.seed)
    tonIoT_results = run_tonIoT(seed=args.seed)

    with open(f"results/phase2_local_only{out_suffix}.json", "w") as f:
        json.dump({"N-BaIoT": nbaiot_results, "TON_IoT": tonIoT_results}, f, indent=2)

    # flat CSV for the lower-bound table
    rows = []
    for r in nbaiot_results:
        rows.append({"dataset": "N-BaIoT", "client": r["label"], "n_train": r["n_train"],
                     "local_test_macro_f1": round(r["local_test_macro_f1"], 4),
                     "global_test_macro_f1": round(r["global_test_macro_f1"], 4),
                     "mirai_recall_mean": round(r.get("mirai_recall_mean", float("nan")), 4)})
    for r in tonIoT_results:
        rows.append({"dataset": "TON_IoT", "client": r["label"], "n_train": r["n_train"],
                     "local_test_macro_f1": round(r["local_test_macro_f1"], 4),
                     "global_test_macro_f1": round(r["global_test_macro_f1"], 4),
                     "mirai_recall_mean": ""})
    pd.DataFrame(rows).to_csv(f"results/phase2_summary{out_suffix}.csv", index=False)

    print("\n\n=== N-BaIoT: zero-Mirai devices' recall on Mirai (global test) ===")
    for r in nbaiot_results:
        if "mirai_recall_mean" in r:
            flag = " <-- device has ZERO Mirai locally" if r["label"] in ("Ennio_Doorbell", "Samsung_SNH_1011_N_Webcam") else ""
            print(f"  {r['label']:40s} mirai_recall_mean={r['mirai_recall_mean']:.4f}{flag}")

    print(f"\nSaved results/phase2_local_only{out_suffix}.json and results/phase2_summary{out_suffix}.csv")
