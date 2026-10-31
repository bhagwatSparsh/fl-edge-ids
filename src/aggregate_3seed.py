"""Aggregates Phase 1/2/3/4 results across the 3 rigor-pass seeds
(42 = original run, 123, 2026) into mean +/- std, and writes:
  - results/threeseed_summary.json   (full mean/std/values for PROGRESS.md)
  - results/dashboard_model_comparison.json   (overwritten, now 3-seed)
  - results/dashboard_zero_day.json           (overwritten, now 3-seed)

No new training -- every number already exists in results/phase*_seed*.json
and results/phase*.json (seed 42); this just averages across the 3 runs.
Phase 1 is MLP-only across seeds per the documented scope decision (CNN-BiLSTM
was only ever run at seed 42, already shown non-superior and far slower).
"""
import json
import statistics as stats

SEEDS = [42, 123, 2026]

def suffix(seed):
    return "" if seed == 42 else f"_seed{seed}"

def load(name, seed):
    return json.load(open(f"results/{name}{suffix(seed)}.json"))

def mean_std(values):
    m = stats.mean(values)
    s = stats.stdev(values) if len(values) > 1 else 0.0
    return m, s

# ---------------- Phase 1: Centralized MLP, per dataset ----------------
phase1_agg = {}
for ds in ["N-BaIoT", "TON_IoT"]:
    vals = [load("phase1_centralized_baselines", s)[ds]["MLP"]["test_macro_f1"] for s in SEEDS]
    m, sd = mean_std(vals)
    phase1_agg[ds] = {"values": vals, "mean": m, "std": sd}

# ---------------- Phase 2: Local-only, mean-of-9-clients, per dataset ----------------
phase2_agg = {}
for ds in ["N-BaIoT", "TON_IoT"]:
    per_seed_means = []
    for s in SEEDS:
        rows = load("phase2_local_only", s)[ds]
        per_seed_means.append(sum(r["global_test_macro_f1"] for r in rows) / len(rows))
    m, sd = mean_std(per_seed_means)
    phase2_agg[ds] = {"values": per_seed_means, "mean": m, "std": sd}

# ---------------- Phase 3: Federated FedAvg/FedProx, per dataset ----------------
phase3_agg = {}
for ds in ["N-BaIoT", "TON_IoT"]:
    for strat in ["FedAvg", "FedProx"]:
        vals = [load("phase3_federated", s)[f"{ds}_{strat}"]["final_macro_f1"] for s in SEEDS]
        m, sd = mean_std(vals)
        phase3_agg[f"{ds}_{strat}"] = {"values": vals, "mean": m, "std": sd}

# ---------------- Phase 4: zero-day, per dataset x strategy ----------------
phase4_agg = {}
for ds in ["N-BaIoT", "TON_IoT"]:
    for strat in ["Centralized", "FedAvg", "FedProx"]:
        in_dist = [load("phase4_zero_day", s)[ds]["runs"][strat]["in_distribution_macro_f1"] for s in SEEDS]
        catch = [load("phase4_zero_day", s)[ds]["runs"][strat]["zero_day_catch_rate"] for s in SEEDS]
        m_id, sd_id = mean_std(in_dist)
        m_c, sd_c = mean_std(catch)
        phase4_agg[f"{ds}_{strat}"] = {
            "in_distribution_macro_f1": {"values": in_dist, "mean": m_id, "std": sd_id},
            "zero_day_catch_rate": {"values": catch, "mean": m_c, "std": sd_c},
        }

summary = {
    "seeds": SEEDS,
    "note": "Phase 1 is MLP-only across all 3 seeds (CNN-BiLSTM only run at seed 42; "
            "already shown non-superior and far slower, so not reseeded -- documented scope cut). "
            "Phase 5 (SHAP) and Phase 6 (Mininet systems) are diagnostic/systems results, not "
            "reseeded. Phase 7 (stretch goals) skipped entirely.",
    "phase1_centralized_macro_f1": phase1_agg,
    "phase2_local_only_mean_macro_f1": phase2_agg,
    "phase3_federated_macro_f1": phase3_agg,
    "phase4_zero_day": phase4_agg,
}
with open("results/threeseed_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("Wrote results/threeseed_summary.json")

# ---------------- Rebuild dashboard_model_comparison.json (3-seed) ----------------
rows = []
for ds_key, ds_label in [("N-BaIoT", "N-BaIoT"), ("TON_IoT", "TON_IoT")]:
    c = phase1_agg[ds_key]
    rows.append({"dataset": ds_label, "approach": "Centralized (pools everything)",
                 "macro_f1": c["mean"], "macro_f1_std": c["std"], "n_seeds": 3})
    l = phase2_agg[ds_key]
    rows.append({"dataset": ds_label, "approach": "Local-only (no sharing, avg of 9 clients)",
                 "macro_f1": l["mean"], "macro_f1_std": l["std"], "n_seeds": 3})
    fa = phase3_agg[f"{ds_label}_FedAvg"]
    rows.append({"dataset": ds_label, "approach": "Federated FedAvg",
                 "macro_f1": fa["mean"], "macro_f1_std": fa["std"], "n_seeds": 3})
    fp = phase3_agg[f"{ds_label}_FedProx"]
    rows.append({"dataset": ds_label, "approach": "Federated FedProx",
                 "macro_f1": fp["mean"], "macro_f1_std": fp["std"], "n_seeds": 3})
with open("results/dashboard_model_comparison.json", "w") as f:
    json.dump(rows, f, indent=2)
print(f"model_comparison (3-seed): {len(rows)} rows")

# ---------------- Rebuild dashboard_zero_day.json (3-seed) ----------------
rows = []
for ds_key, held_out in [("N-BaIoT", "an entire Mirai family"), ("TON_IoT", "ransomware")]:
    for strategy in ["Centralized", "FedAvg", "FedProx"]:
        a = phase4_agg[f"{ds_key}_{strategy}"]
        rows.append({
            "dataset": ds_key, "strategy": strategy, "held_out_family": held_out,
            "in_distribution_macro_f1": a["in_distribution_macro_f1"]["mean"],
            "in_distribution_macro_f1_std": a["in_distribution_macro_f1"]["std"],
            "zero_day_catch_rate": a["zero_day_catch_rate"]["mean"],
            "zero_day_catch_rate_std": a["zero_day_catch_rate"]["std"],
        })
with open("results/dashboard_zero_day.json", "w") as f:
    json.dump(rows, f, indent=2)
print(f"zero_day (3-seed): {len(rows)} rows")

# ---------------- Print a human-readable headline summary ----------------
print("\n=== HEADLINE (mean +/- std across seeds 42/123/2026) ===")
for ds in ["N-BaIoT", "TON_IoT"]:
    print(f"\n{ds}:")
    print(f"  Centralized:  {phase1_agg[ds]['mean']:.4f} +/- {phase1_agg[ds]['std']:.4f}")
    print(f"  Local-only:   {phase2_agg[ds]['mean']:.4f} +/- {phase2_agg[ds]['std']:.4f}")
    print(f"  FedAvg:       {phase3_agg[f'{ds}_FedAvg']['mean']:.4f} +/- {phase3_agg[f'{ds}_FedAvg']['std']:.4f}")
    print(f"  FedProx:      {phase3_agg[f'{ds}_FedProx']['mean']:.4f} +/- {phase3_agg[f'{ds}_FedProx']['std']:.4f}")
    print(f"  Zero-day catch (FedProx): {phase4_agg[f'{ds}_FedProx']['zero_day_catch_rate']['mean']:.4f} +/- {phase4_agg[f'{ds}_FedProx']['zero_day_catch_rate']['std']:.4f}")
