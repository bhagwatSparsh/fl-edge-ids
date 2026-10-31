"""
Phase 4 gate: TRUE zero-day generalization ("leave-one-family-out").

Trains on a class set with an entire attack family removed (nobody, in any
client, ever saw it), then asks the one question that's actually
answerable for a family the model was never given a name for: does the
model still recognize it as *something wrong*, rather than confidently
calling it benign? This is the plan's "binary head" framing, implemented
as a binary COLLAPSE of the existing multiclass softmax (predicted class
!= benign) rather than training a second, separate binary architecture --
cheaper, and it directly tests what the deployed multiclass model would
actually do if this attack showed up in the wild.

Three models per dataset, for a fair before/after/ceiling comparison,
mirroring the project's own Phase 1/2/3 pattern:
  - Centralized (pooled, mirai/ransomware-free)  -- upper bound, unrealistic
  - FedAvg   (same 9-client federated setup as Phase 3, reduced class set)
  - FedProx  (same, with the proximal term)

All three are evaluated two ways on the SAME held-out family:
  1. in-distribution macro-F1 (does removing the family hurt normal
     performance on everything else? -- sanity check, should look like
     Phase 1/3's numbers minus one easier/harder class)
  2. zero-day binary catch rate = fraction of held-out-family rows the
     model assigns ANY non-benign class to (the real result)
"""
import sys
sys.path.insert(0, "src")
import argparse
import json
import time
import numpy as np
import torch
from sklearn.model_selection import train_test_split

from flwr.client import ClientApp
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg, FedProx
from flwr.simulation import run_simulation

from models.mlp import MLP
from train.engine import fit, evaluate, make_loaders
from federated.engine_fed import make_client_fn, make_evaluate_fn

DATA_DIR = "data/partitions/federated_phase4"
NUM_ROUNDS = 20
LOCAL_EPOCHS = 2
BATCH_SIZE = 256
LR = 1e-3
PROXIMAL_MU = 0.01


def zero_day_catch_rate(model, X_holdout, benign_index, batch_size=1024):
    """Fraction of held-out-family rows the model does NOT call benign --
    i.e. it flags them as some kind of attack, even though it was never
    taught this specific family's name. Also returns the full predicted-
    class distribution, to see which KNOWN class the unseen family gets
    mistaken for (a real, reportable finding, not just a pass/fail number)."""
    model.eval()
    with torch.no_grad():
        logits = model(torch.as_tensor(X_holdout, dtype=torch.float32))
        preds = logits.argmax(dim=1).numpy()
    catch_rate = float((preds != benign_index).mean())
    pred_dist = np.bincount(preds, minlength=logits.shape[1]).tolist()
    return catch_rate, pred_dist


def run_centralized(dataset_tag, n_features, n_classes, epochs=15, seed=42, out_suffix=""):
    """Quick centralized reference: pool every client's train data (which
    already has the family removed), train one MLP, report both metrics
    on the same held-out set the federated runs use."""
    Xs, ys = [], []
    i = 0
    while True:
        try:
            Xs.append(np.load(f"{DATA_DIR}/{dataset_tag}_client{i}_X.npy"))
            ys.append(np.load(f"{DATA_DIR}/{dataset_tag}_client{i}_y.npy"))
            i += 1
        except FileNotFoundError:
            break
    X_train = np.concatenate(Xs).astype(np.float32)
    y_train = np.concatenate(ys)

    X_test = np.load(f"{DATA_DIR}/{dataset_tag}_test_X.npy")
    y_test = np.load(f"{DATA_DIR}/{dataset_tag}_test_y.npy")
    X_holdout = np.load(f"{DATA_DIR}/{dataset_tag}_holdout_X.npy")

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=seed, stratify=y_train
    )
    torch.manual_seed(seed)
    model = MLP(n_features=n_features, n_classes=n_classes)
    t0 = time.time()
    model, history, elapsed = fit(model, X_tr, y_tr, X_val, y_val, n_classes,
                                   epochs=epochs, batch_size=512, verbose=False)
    _, test_loader = make_loaders(X_tr[:1], y_tr[:1], X_test, y_test, batch_size=512)
    metrics = evaluate(model, test_loader, n_classes)
    catch_rate, pred_dist = zero_day_catch_rate(model, X_holdout, benign_index=0)
    print(f"Centralized {dataset_tag}: in-dist macro_f1={metrics['macro_f1']:.4f} "
          f"| zero-day catch rate={catch_rate:.4f} | {elapsed:.1f}s")
    torch.save(model.state_dict(), f"results/phase4_{dataset_tag}_centralized{out_suffix}.pt")
    return {"strategy": "Centralized", "in_distribution_macro_f1": metrics["macro_f1"],
            "zero_day_catch_rate": catch_rate, "predicted_class_distribution": pred_dist,
            "train_seconds": elapsed}


def run_federated(dataset_tag, n_clients, n_features, n_classes, strategy_name, seed=42, out_suffix=""):
    X_test = np.load(f"{DATA_DIR}/{dataset_tag}_test_X.npy")
    y_test = np.load(f"{DATA_DIR}/{dataset_tag}_test_y.npy")
    X_holdout = np.load(f"{DATA_DIR}/{dataset_tag}_holdout_X.npy")

    history_sink = []
    checkpoint_path = f"results/phase4_{dataset_tag}_{strategy_name}{out_suffix}.pt"
    evaluate_fn = make_evaluate_fn(X_test, y_test, n_features, n_classes, history_sink,
                                     checkpoint_path=checkpoint_path)
    client_fn = make_client_fn(dataset_tag, n_features, n_classes, DATA_DIR, base_seed=seed)
    client_app = ClientApp(client_fn=client_fn)

    def fit_config(server_round):
        return {"local_epochs": LOCAL_EPOCHS, "batch_size": BATCH_SIZE, "lr": LR}

    def server_fn(context):
        common = dict(
            fraction_fit=1.0, fraction_evaluate=0.0,
            min_fit_clients=n_clients, min_evaluate_clients=0, min_available_clients=n_clients,
            evaluate_fn=evaluate_fn, on_fit_config_fn=fit_config,
        )
        strategy = FedAvg(**common) if strategy_name == "FedAvg" else FedProx(**common, proximal_mu=PROXIMAL_MU)
        return ServerAppComponents(strategy=strategy, config=ServerConfig(num_rounds=NUM_ROUNDS))

    server_app = ServerApp(server_fn=server_fn)

    print(f"\n{'='*60}\n{dataset_tag} / {strategy_name} (zero-day): {n_clients} clients, "
          f"{NUM_ROUNDS} rounds x {LOCAL_EPOCHS} local epochs\n{'='*60}")
    t0 = time.time()
    run_simulation(
        server_app=server_app, client_app=client_app, num_supernodes=n_clients,
        backend_config={"client_resources": {"num_cpus": 1, "num_gpus": 0.0}},
    )
    elapsed = time.time() - t0

    # Reload the just-saved checkpoint to compute the zero-day catch rate
    # (evaluate_fn only computed in-distribution metrics during training).
    model = MLP(n_features=n_features, n_classes=n_classes)
    model.load_state_dict(torch.load(checkpoint_path))
    catch_rate, pred_dist = zero_day_catch_rate(model, X_holdout, benign_index=0)

    final = history_sink[-1]
    print(f"{dataset_tag} / {strategy_name}: in-dist macro_f1={final['macro_f1']:.4f} "
          f"| zero-day catch rate={catch_rate:.4f} | {elapsed:.1f}s")
    return {"strategy": strategy_name, "in_distribution_macro_f1": final["macro_f1"],
            "zero_day_catch_rate": catch_rate, "predicted_class_distribution": pred_dist,
            "train_seconds": elapsed, "checkpoint_path": checkpoint_path}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    out_suffix = "" if args.seed == 42 else f"_seed{args.seed}"

    with open(f"{DATA_DIR}/meta.json") as f:
        meta = json.load(f)
    nbaiot_meta, tonIoT_meta = meta["N-BaIoT"], meta["TON_IoT"]

    all_results = {"N-BaIoT": {"meta": nbaiot_meta, "runs": {}},
                    "TON_IoT": {"meta": tonIoT_meta, "runs": {}}}

    print("="*60, "\nN-BaIoT (Mirai held out entirely)\n", "="*60)
    all_results["N-BaIoT"]["runs"]["Centralized"] = run_centralized(
        "nbaiot", nbaiot_meta["n_features"], nbaiot_meta["n_classes"], seed=args.seed, out_suffix=out_suffix)
    for strategy_name in ["FedAvg", "FedProx"]:
        all_results["N-BaIoT"]["runs"][strategy_name] = run_federated(
            "nbaiot", nbaiot_meta["n_clients"], nbaiot_meta["n_features"],
            nbaiot_meta["n_classes"], strategy_name, seed=args.seed, out_suffix=out_suffix)

    print("\n", "="*60, "\nTON_IoT (ransomware held out entirely)\n", "="*60)
    all_results["TON_IoT"]["runs"]["Centralized"] = run_centralized(
        "tonIoT", tonIoT_meta["n_features"], tonIoT_meta["n_classes"], seed=args.seed, out_suffix=out_suffix)
    for strategy_name in ["FedAvg", "FedProx"]:
        all_results["TON_IoT"]["runs"][strategy_name] = run_federated(
            "tonIoT", tonIoT_meta["n_clients"], tonIoT_meta["n_features"],
            tonIoT_meta["n_classes"], strategy_name, seed=args.seed, out_suffix=out_suffix)

    with open(f"results/phase4_zero_day{out_suffix}.json", "w") as f:
        json.dump(all_results, f, indent=2)

    rows = []
    for dataset, block in all_results.items():
        for strategy, r in block["runs"].items():
            rows.append({"dataset": dataset, "strategy": strategy,
                         "in_distribution_macro_f1": round(r["in_distribution_macro_f1"], 4),
                         "zero_day_catch_rate": round(r["zero_day_catch_rate"], 4),
                         "train_seconds": round(r["train_seconds"], 1)})
    import pandas as pd
    pd.DataFrame(rows).to_csv(f"results/phase4_summary{out_suffix}.csv", index=False)

    print(f"\n\n=== Phase 4 summary (seed={args.seed}) ===")
    for row in rows:
        print(row)
    print(f"\nSaved results/phase4_zero_day{out_suffix}.json and results/phase4_summary{out_suffix}.csv")
