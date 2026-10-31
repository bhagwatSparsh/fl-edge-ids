"""
Phase 3 gate: federated simulation with Flower -- FedAvg vs FedProx, on both
datasets, using the exact same 9-client splits as Phase 2 (so Phase 2's
floor and this phase's result are directly comparable) and the exact same
MLP architecture and metric code as Phase 1 and Phase 2.

Scope decision (flagged explicitly, same transparency pattern as Phase 2's
MLP-only call): MLP only, not CNN-BiLSTM. CNN-BiLSTM took ~30 minutes for
ONE centralized epoch run in Phase 1; running it as a federated client
model (9 clients x several rounds x 2 datasets x 2 strategies) is not a
good use of time for this gate, whose point is comparing AGGREGATION
STRATEGIES (FedAvg vs FedProx), not architectures. MLP was also the
marginally better model on N-BaIoT in Phase 1.

Hyperparameters (one deliberate choice each, documented rather than swept,
per the project's time budget):
  - num_rounds = 20            (communication rounds)
  - local_epochs_per_round = 2  (standard FL shape: many rounds, few local
    epochs each, so no client's updates drift too far before the next
    aggregation -- the opposite of Phase 1/2's "train to convergence once")
  - batch_size = 256, lr = 1e-3, Adam (unchanged from every other phase)
  - FedProx proximal_mu = 0.01  (a standard literature value, e.g. Li et
    al. 2018's own experiments sweep {0.001, 0.01, 0.1, 1}; not swept here
    for time -- noted as a possible follow-up, same as Phase 2's note on
    CNN-BiLSTM)
  - fraction_fit = 1.0, all 9 clients participate every round (only 9
    clients total, so client sampling isn't the point of this gate)
"""
import sys
sys.path.insert(0, "src")
import argparse
import json
import time
import numpy as np

from flwr.client import ClientApp
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg, FedProx
from flwr.simulation import run_simulation

from federated.engine_fed import make_client_fn, make_evaluate_fn

DATA_DIR = "data/partitions/federated"
NUM_ROUNDS = 20
LOCAL_EPOCHS = 2
BATCH_SIZE = 256
LR = 1e-3
PROXIMAL_MU = 0.01

NBAIOT_CLASSES = ["benign", "gafgyt.combo", "gafgyt.junk", "gafgyt.scan", "gafgyt.tcp",
                   "gafgyt.udp", "mirai.ack", "mirai.scan", "mirai.syn", "mirai.udp",
                   "mirai.udpplain"]
MIRAI_IDX = [i for i, c in enumerate(NBAIOT_CLASSES) if c.startswith("mirai")]


def run_one(dataset_tag, n_clients, n_features, n_classes, strategy_name, result_prefix="phase3",
            seed=42):
    X_test = np.load(f"{DATA_DIR}/{dataset_tag}_test_X.npy")
    y_test = np.load(f"{DATA_DIR}/{dataset_tag}_test_y.npy")

    history_sink = []
    checkpoint_path = f"results/{result_prefix}_{dataset_tag}_{strategy_name}.pt"
    evaluate_fn = make_evaluate_fn(X_test, y_test, n_features, n_classes, history_sink,
                                     checkpoint_path=checkpoint_path)
    client_fn = make_client_fn(dataset_tag, n_features, n_classes, DATA_DIR, base_seed=seed)
    client_app = ClientApp(client_fn=client_fn)

    def fit_config(server_round):
        return {"local_epochs": LOCAL_EPOCHS, "batch_size": BATCH_SIZE, "lr": LR}

    def server_fn(context):
        common = dict(
            fraction_fit=1.0,
            fraction_evaluate=0.0,
            min_fit_clients=n_clients,
            min_evaluate_clients=0,
            min_available_clients=n_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=fit_config,
        )
        if strategy_name == "FedAvg":
            strategy = FedAvg(**common)
        else:
            strategy = FedProx(**common, proximal_mu=PROXIMAL_MU)
        return ServerAppComponents(strategy=strategy, config=ServerConfig(num_rounds=NUM_ROUNDS))

    server_app = ServerApp(server_fn=server_fn)

    print(f"\n{'='*60}\n{dataset_tag} / {strategy_name}: {n_clients} clients, "
          f"{NUM_ROUNDS} rounds x {LOCAL_EPOCHS} local epochs\n{'='*60}")
    t0 = time.time()
    run_simulation(
        server_app=server_app,
        client_app=client_app,
        num_supernodes=n_clients,
        backend_config={"client_resources": {"num_cpus": 1, "num_gpus": 0.0}},
    )
    elapsed = time.time() - t0
    print(f"{dataset_tag} / {strategy_name} finished in {elapsed:.1f}s, "
          f"final round macro_f1={history_sink[-1]['macro_f1']:.4f}")

    # Rebuild the final global model from the last round's metrics isn't
    # enough on its own -- re-run evaluate_fn's own last call already saved
    # per_class_recall/fpr/confusion_matrix into history_sink[-1], so no
    # extra forward pass is needed; this IS the final global model's metrics.
    final = history_sink[-1]
    result = {
        "dataset": dataset_tag,
        "strategy": strategy_name,
        "n_clients": n_clients,
        "num_rounds": NUM_ROUNDS,
        "local_epochs": LOCAL_EPOCHS,
        "proximal_mu": PROXIMAL_MU if strategy_name == "FedProx" else None,
        "train_seconds": elapsed,
        "checkpoint_path": checkpoint_path,
        "round_history": history_sink,  # per-round macro_f1 (+ full metrics each round)
        "final_macro_f1": final["macro_f1"],
        "final_per_class_recall": final["per_class_recall"],
        "final_per_class_fpr": final["per_class_fpr"],
    }
    if dataset_tag == "nbaiot":
        result["final_mirai_recall_mean"] = float(np.mean([final["per_class_recall"][i] for i in MIRAI_IDX]))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    out_suffix = "" if args.seed == 42 else f"_seed{args.seed}"
    result_prefix = "phase3" if args.seed == 42 else f"phase3_seed{args.seed}"

    with open(f"{DATA_DIR}/meta.json") as f:
        meta = json.load(f)
    nbaiot_meta, tonIoT_meta = meta["N-BaIoT"], meta["TON_IoT"]

    all_results = {}

    for strategy_name in ["FedAvg", "FedProx"]:
        all_results[f"N-BaIoT_{strategy_name}"] = run_one(
            "nbaiot", n_clients=nbaiot_meta["n_clients"], n_features=nbaiot_meta["n_features"],
            n_classes=nbaiot_meta["n_classes"], strategy_name=strategy_name,
            result_prefix=result_prefix, seed=args.seed)

    for strategy_name in ["FedAvg", "FedProx"]:
        all_results[f"TON_IoT_{strategy_name}"] = run_one(
            "tonIoT", n_clients=tonIoT_meta["n_clients"], n_features=tonIoT_meta["n_features"],
            n_classes=tonIoT_meta["n_classes"], strategy_name=strategy_name,
            result_prefix=result_prefix, seed=args.seed)

    with open(f"results/phase3_federated{out_suffix}.json", "w") as f:
        json.dump(all_results, f, indent=2)

    rows = []
    for key, r in all_results.items():
        row = {"dataset": r["dataset"], "strategy": r["strategy"],
               "final_macro_f1": round(r["final_macro_f1"], 4),
               "train_seconds": round(r["train_seconds"], 1)}
        if "final_mirai_recall_mean" in r:
            row["final_mirai_recall_mean"] = round(r["final_mirai_recall_mean"], 4)
        rows.append(row)
    import pandas as pd
    pd.DataFrame(rows).to_csv(f"results/phase3_summary{out_suffix}.csv", index=False)

    print("\n\n=== Phase 3 summary (seed={}) ===".format(args.seed))
    for row in rows:
        print(row)
    print(f"\nSaved results/phase3_federated{out_suffix}.json and results/phase3_summary{out_suffix}.csv")
