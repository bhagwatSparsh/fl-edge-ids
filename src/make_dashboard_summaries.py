"""Assembles the dashboard's comparison/summary JSON files from results
already produced by Phases 1-6. No new computation -- every number here
already exists in results/*.json; this just reshapes them into flat row
lists so the dashboard page can bind each figure to a real source.
"""
import json

# ---------- model_comparison.json ----------
# Centralized (Phase 1, pooled "sees everything"), Local-only mean (Phase 2,
# averaged across the 9/9 real clients -- the floor), FedAvg / FedProx
# (Phase 3, the actual federated system), per dataset.
phase1 = json.load(open("results/phase1_centralized_baselines.json"))
phase2 = json.load(open("results/phase2_local_only.json"))
phase3 = json.load(open("results/phase3_federated.json"))

def mean_local_only(dataset_key):
    rows = phase2[dataset_key]
    return sum(r["global_test_macro_f1"] for r in rows) / len(rows)

rows = []
for ds_key, ds_label in [("N-BaIoT", "N-BaIoT"), ("TON_IoT", "TON_IoT")]:
    rows.append({"dataset": ds_label, "approach": "Centralized (pools everything)",
                 "macro_f1": phase1[ds_key]["MLP"]["test_macro_f1"], "n_seeds": 1})
    rows.append({"dataset": ds_label, "approach": "Local-only (no sharing, avg of 9 clients)",
                 "macro_f1": mean_local_only(ds_key), "n_seeds": 1})
    rows.append({"dataset": ds_label, "approach": "Federated FedAvg",
                 "macro_f1": phase3[f"{ds_label}_FedAvg"]["final_macro_f1"], "n_seeds": 1})
    rows.append({"dataset": ds_label, "approach": "Federated FedProx",
                 "macro_f1": phase3[f"{ds_label}_FedProx"]["final_macro_f1"], "n_seeds": 1})

with open("results/dashboard_model_comparison.json", "w") as f:
    json.dump(rows, f, indent=2)
print(f"model_comparison: {len(rows)} rows")

# ---------- zero_day.json ----------
phase4 = json.load(open("results/phase4_zero_day.json"))
rows = []
for ds_key, held_out in [("N-BaIoT", "an entire Mirai family"), ("TON_IoT", "ransomware")]:
    for strategy, r in phase4[ds_key]["runs"].items():
        rows.append({
            "dataset": ds_key, "strategy": strategy, "held_out_family": held_out,
            "in_distribution_macro_f1": r["in_distribution_macro_f1"],
            "zero_day_catch_rate": r["zero_day_catch_rate"],
        })
with open("results/dashboard_zero_day.json", "w") as f:
    json.dump(rows, f, indent=2)
print(f"zero_day: {len(rows)} rows")

# ---------- bandwidth: three flat files (totals, round-time, float16 check) ----------
phase6 = json.load(open("results/phase6_analysis.json"))
bw = phase6["bandwidth_comparison_totals"]
bw_rows = [
    {"item": "Model updates (float32), measured", "bytes": bw["total_model_traffic_bytes_f32_measured"], "method": "measured"},
    {"item": "Model updates (float16), estimated", "bytes": bw["total_model_traffic_bytes_f16_estimated"], "method": "estimated"},
    {"item": "Raw feature-CSV equivalent, estimated", "bytes": bw["total_feature_csv_bytes_estimated"], "method": "estimated"},
    {"item": "Raw packet-capture equivalent, estimated", "bytes": bw["total_raw_pcap_bytes_estimated"], "method": "estimated"},
]
with open("results/dashboard_bandwidth_totals.json", "w") as f:
    json.dump(bw_rows, f, indent=2)

round_time_rows = []
for tag, r in phase6["round_time_vs_delay"].items():
    round_time_rows.append({"delay_ms": r["delay_ms"], "round_seconds": r["mean_round_seconds"]})
with open("results/dashboard_round_time.json", "w") as f:
    json.dump(round_time_rows, f, indent=2)

f16 = phase6["float16_check"]
f16_rows = [{
    "macro_f1_float32": f16["macro_f1_float32"],
    "macro_f1_float16_roundtrip": f16["macro_f1_float16_roundtrip"],
    "delta": f16["delta"],
    "size_ratio_f16_over_f32": f16["size_ratio_f16_over_f32"],
}]
with open("results/dashboard_float16_check.json", "w") as f:
    json.dump(f16_rows, f, indent=2)
print(f"bandwidth_totals: {len(bw_rows)} rows, round_time: {len(round_time_rows)} rows, float16_check: 1 row")

# ---------- explainability: two flat files (agreement, top features) ----------
phase5 = json.load(open("results/phase5_shap.json"))
agreement_rows = []
for ds_key in ["N-BaIoT", "TON_IoT"]:
    d = phase5[ds_key]
    agreement_rows.append({
        "dataset": ds_key,
        "spearman_rho": d["spearman_rho_federated_vs_centralized"],
        "p_value": d["spearman_pvalue"],
    })
with open("results/dashboard_explainability_agreement.json", "w") as f:
    json.dump(agreement_rows, f, indent=2)

top_feature_rows = []
for ds_key in ["N-BaIoT", "TON_IoT"]:
    for rank, feat in enumerate(phase5[ds_key]["top15_features"][:5], start=1):
        if isinstance(feat, dict):
            name = feat.get("feature")
        elif isinstance(feat, (list, tuple)):
            name = feat[0]
        else:
            name = feat
        top_feature_rows.append({"dataset": ds_key, "rank": rank, "feature": name})
with open("results/dashboard_top_features.json", "w") as f:
    json.dump(top_feature_rows, f, indent=2)
print(f"explainability_agreement: {len(agreement_rows)} rows, top_features: {len(top_feature_rows)} rows")
