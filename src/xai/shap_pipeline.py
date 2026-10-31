"""
Phase 5: client-side SHAP explainability.

Per the plan: each client runs shap.GradientExplainer on the FINAL GLOBAL
model (Phase 3's FedProx checkpoint -- the project's proposed method),
using a small sample of its OWN local data. Only a per-class mean-|SHAP|
vector ever "leaves" the client -- never raw data, never per-row SHAP
values, consistent with every other phase's privacy design. The "server"
side then combines these into one global importance vector, weighted by
each client's sample count (same weighting rule as FedAvg's own
aggregation).

A second SHAP run on the Phase 1 CENTRALIZED model (same architecture,
trained on pooled data) gives the "agreement check" the plan asks for:
if the federated model is paying attention to roughly the same features
as a model that saw everything at once, that's evidence federated
training learned something real, not an artifact of how it was trained.
Measured as a Spearman rank correlation over the two models' overall
feature-importance rankings.

GradientExplainer on this MLP returns shap_values of shape
(n_samples, n_features, n_classes) (checked directly against the
installed shap version rather than assumed).
"""
import sys
sys.path.insert(0, "src")
import json
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
import shap

from models.mlp import MLP

DATA_DIR = "data/partitions/federated"


def client_shap_importance(model, X_client, n_classes, background_n=50, sample_n=200, seed=42):
    """One client's local SHAP computation. Returns:
    class_importance: (n_classes, n_features) mean(|shap|) per class
    n_rows: this client's sample count (for weighted aggregation)
    This is the ONLY thing a real client would ever transmit."""
    rng = np.random.RandomState(seed)
    n = len(X_client)
    bg_n = min(background_n, n)
    smp_n = min(sample_n, n)
    bg_idx = rng.choice(n, bg_n, replace=False)
    smp_idx = rng.choice(n, smp_n, replace=False)
    background = torch.as_tensor(X_client[bg_idx], dtype=torch.float32)
    samples = torch.as_tensor(X_client[smp_idx], dtype=torch.float32)

    explainer = shap.GradientExplainer(model, background)
    sv = explainer.shap_values(samples)  # (smp_n, n_features, n_classes)
    class_importance = np.abs(sv).mean(axis=0).T  # -> (n_classes, n_features)
    return class_importance, n


def aggregate_weighted(class_importances, weights):
    """Server-side aggregation: weighted mean across clients, same
    sample-count weighting FedAvg itself uses -- so a client with more
    data has proportionally more say in what the EXPLANATION says
    matters, matching how much say it had in training the model."""
    weights = np.array(weights, dtype=np.float64)
    stacked = np.stack(class_importances)  # (n_clients, n_classes, n_features)
    weighted = (stacked * weights[:, None, None]).sum(axis=0) / weights.sum()
    return weighted  # (n_classes, n_features)


def overall_ranking(class_importance, class_counts=None):
    """Collapse the per-class importance matrix into one overall
    per-feature score, weighted by how common each class actually is
    (so a rare class's idiosyncratic top feature doesn't dominate the
    headline ranking) -- falls back to a plain mean if counts aren't
    given (centralized run, where we don't track per-class counts the
    same way)."""
    if class_counts is None:
        return class_importance.mean(axis=0)
    w = np.array(class_counts, dtype=np.float64)
    w = w / w.sum()
    return (class_importance * w[:, None]).sum(axis=0)


def run_dataset(dataset_tag, n_features, n_classes, n_clients, feature_names,
                 federated_ckpt, centralized_ckpt):
    print(f"\n{'='*60}\n{dataset_tag}: SHAP over {n_clients} clients\n{'='*60}")

    fed_model = MLP(n_features=n_features, n_classes=n_classes)
    fed_model.load_state_dict(torch.load(federated_ckpt))
    fed_model.eval()

    client_importances, client_weights, client_class_counts = [], [], []
    per_client_top5 = {}
    for i in range(n_clients):
        X = np.load(f"{DATA_DIR}/{dataset_tag}_client{i}_X.npy")
        y = np.load(f"{DATA_DIR}/{dataset_tag}_client{i}_y.npy")
        class_imp, n = client_shap_importance(fed_model, X, n_classes)
        client_importances.append(class_imp)
        client_weights.append(n)
        counts = np.bincount(y, minlength=n_classes)
        client_class_counts.append(counts)

        client_overall = overall_ranking(class_imp, counts)
        top5_idx = np.argsort(-client_overall)[:5]
        per_client_top5[f"client{i}"] = [feature_names[j] for j in top5_idx]
        print(f"  client {i} (n={n}): top features -> {per_client_top5[f'client{i}']}")

    fed_class_importance = aggregate_weighted(client_importances, client_weights)
    total_class_counts = np.sum(client_class_counts, axis=0)
    fed_overall = overall_ranking(fed_class_importance, total_class_counts)

    # Centralized comparison, same recipe, on the shared (already-scaled)
    # global test set, which uses the identical federated-scaler math as
    # Phase 1's own training scaler (mathematically the same regardless of
    # whether it's computed as "9 clients combined" or "1 pooled client").
    cen_model = MLP(n_features=n_features, n_classes=n_classes)
    cen_model.load_state_dict(torch.load(centralized_ckpt))
    cen_model.eval()
    X_test = np.load(f"{DATA_DIR}/{dataset_tag}_test_X.npy")
    y_test = np.load(f"{DATA_DIR}/{dataset_tag}_test_y.npy")
    cen_class_importance, _ = client_shap_importance(cen_model, X_test, n_classes,
                                                        background_n=100, sample_n=500)
    test_class_counts = np.bincount(y_test, minlength=n_classes)
    cen_overall = overall_ranking(cen_class_importance, test_class_counts)

    rho, pval = spearmanr(fed_overall, cen_overall)

    top15_idx = np.argsort(-fed_overall)[:15]
    top15 = [{"feature": feature_names[j], "federated_importance": float(fed_overall[j]),
              "centralized_importance": float(cen_overall[j])} for j in top15_idx]

    print(f"  Federated vs centralized feature-ranking agreement (Spearman rho): {rho:.4f} (p={pval:.2e})")
    print(f"  Top 5 overall (federated): {[t['feature'] for t in top15[:5]]}")

    return {
        "n_clients": n_clients,
        "spearman_rho_federated_vs_centralized": float(rho),
        "spearman_pvalue": float(pval),
        "top15_features": top15,
        "per_client_top5": per_client_top5,
        "federated_overall_importance": {feature_names[j]: float(fed_overall[j]) for j in range(n_features)},
        "centralized_overall_importance": {feature_names[j]: float(cen_overall[j]) for j in range(n_features)},
    }


if __name__ == "__main__":
    with open(f"{DATA_DIR}/meta.json") as f:
        meta = json.load(f)

    nb_features = pd.read_parquet("data/partitions/pooled_test.parquet").columns.tolist()
    ton_features = pd.read_parquet("data/partitions/tonIoT_test.parquet").columns.tolist()

    results = {}
    results["N-BaIoT"] = run_dataset(
        "nbaiot", meta["N-BaIoT"]["n_features"], meta["N-BaIoT"]["n_classes"], meta["N-BaIoT"]["n_clients"],
        nb_features, "results/phase3_nbaiot_FedProx.pt", "results/phase1_N-BaIoT_MLP.pt")

    results["TON_IoT"] = run_dataset(
        "tonIoT", meta["TON_IoT"]["n_features"], meta["TON_IoT"]["n_classes"], meta["TON_IoT"]["n_clients"],
        ton_features, "results/phase3_tonIoT_FedProx.pt", "results/phase1_TON_IoT_MLP.pt")

    with open("results/phase5_shap.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved results/phase5_shap.json")
