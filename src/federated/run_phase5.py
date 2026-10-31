"""
Phase 5 gate: does client-side SMOTE actually help?

Re-runs the EXACT same FedAvg/FedProx recipe as Phase 3 (same architecture,
rounds, local epochs, learning rate -- nothing else changes) on the
SMOTE-balanced client data from prepare_data_smote.py, so any difference
in the results is attributable to SMOTE alone, not a confound from a
different setup. Reuses run_phase3.run_one() directly (just pointed at the
SMOTE'd data directory) rather than duplicating it.
"""
import sys
sys.path.insert(0, "src")
import json

import federated.run_phase3 as rp

rp.DATA_DIR = "data/partitions/federated_smote"

if __name__ == "__main__":
    with open(f"{rp.DATA_DIR}/meta.json") as f:
        meta = json.load(f)
    nbaiot_meta, tonIoT_meta = meta["N-BaIoT"], meta["TON_IoT"]

    all_results = {}
    for strategy_name in ["FedAvg", "FedProx"]:
        r = rp.run_one("nbaiot", n_clients=nbaiot_meta["n_clients"], n_features=nbaiot_meta["n_features"],
                        n_classes=nbaiot_meta["n_classes"], strategy_name=strategy_name,
                        result_prefix="phase5_SMOTE")
        all_results[f"N-BaIoT_SMOTE_{strategy_name}"] = r

    for strategy_name in ["FedAvg", "FedProx"]:
        r = rp.run_one("tonIoT", n_clients=tonIoT_meta["n_clients"], n_features=tonIoT_meta["n_features"],
                        n_classes=tonIoT_meta["n_classes"], strategy_name=strategy_name,
                        result_prefix="phase5_SMOTE")
        all_results[f"TON_IoT_SMOTE_{strategy_name}"] = r

    with open("results/phase5_federated_smote.json", "w") as f:
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
    pd.DataFrame(rows).to_csv("results/phase5_summary.csv", index=False)

    print("\n\n=== Phase 5 (SMOTE) summary ===")
    for row in rows:
        print(row)
    print("\nSaved results/phase5_federated_smote.json and results/phase5_summary.csv")
