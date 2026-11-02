# Implementation Plan — Federated Learning-Based Distributed IDS for Edge-IoT Networks

Author: Sparsh Y Bhagwat
Companion to: *FL Edge-IoT IDS — Stack & Method Review*

**Build principle:** vanilla first, framework second. Each phase produces a result that is checked before the next phase starts. Data and model code are plain PyTorch, with no Flower imports, so the same code runs centralized, local-only, simulated-federated and in Mininet.

**Assumption to confirm:** the model takes **per-record input**, with an MLP baseline. Windowed sequences are a stretch ablation (Phase 7). If the team picks windowing instead, Phases 0 and 1 change; nothing else does.

---

## 0. Locked decisions

| Area | Decision |
|---|---|
| Language / runtime | Python 3.11 (one venv, pinned `requirements.txt`) |
| ML framework | PyTorch only (CPU is enough) |
| FL framework | Flower, one pinned version, Message API only (`flwr.serverapp`, `flwr.clientapp`, `flwr.app`) |
| Partitioning | N-BaIoT: natural, 1 device = 1 client (9 clients). TON_IoT: Dirichlet via `flwr-datasets`, α ∈ {0.1, 0.5, 1.0} |
| Models | MLP (baseline) and 1D-CNN-BiLSTM (proposed) |
| Aggregation | FedAvg (μ = 0) vs FedProx (μ sweep), proximal term implemented client-side |
| Imbalance | Class-weighted cross-entropy by default; SMOTE as an ablation |
| Headline metric | Macro-F1, plus per-class recall and FPR |
| Explainability | Client-side SHAP `GradientExplainer` (Captum IG as backup); only mean-\|SHAP\| vectors are uploaded |
| Emulation | Mininet on native Ubuntu 22.04/24.04 or a VM; default OVS controller; no Ryu in core |
| Experiment tiers | Flower simulation runtime for all ML results; Mininet for systems results only |

---

## 1. Environment and repo layout

```
fl-edge-ids/
├── requirements.txt          # pinned versions, frozen at setup
├── configs/                  # one YAML per experiment run
├── data/
│   ├── raw/                  # downloaded CSVs (git-ignored)
│   └── partitions/           # client_<id>_{train,test}.parquet
├── src/
│   ├── data/                 # load, clean, partition, scaler (no Flower)
│   ├── models/               # mlp.py, cnn_bilstm.py
│   ├── train/                # train_one_epoch(), evaluate(), prox_loss()
│   ├── baselines/            # centralized.py, local_only.py
│   ├── fl/                   # Flower ServerApp + ClientApp (thin wrappers over src/train)
│   ├── xai/                  # client-side SHAP, aggregation of importance vectors
│   └── mininet/              # topology.py, launch scripts
├── results/                  # one CSV per run: round, client, metric, value
├── notebooks/                # plots only, no training logic
└── tests/                    # leakage tests, prox-term test, μ=0 ≡ FedAvg test
```

Setup checklist:

1. Pin Flower to the stable release current at setup and freeze it. Read only that version's docs; older tutorials use the deprecated API.
2. Install Mininet with `apt` on the Linux machine or VM, not with pip.
3. Fix random seeds (Python, NumPy, PyTorch, partitioner). Every reported number is the mean ± std of 3 seeds.

---

## 2. Phases

### Phase 0 — Data pipeline (plain Python)

**Build**

- **N-BaIoT:** load per-device CSVs. Use 11 classes: benign + 5 Gafgyt + 5 Mirai. Cap rows per (device, class) with a config value, e.g. 20k, so it fits on laptops.
- **TON_IoT (train_test_network):** use the multi-class `type` column (normal + 9 attacks). Drop identifier and leakage columns: src/dst IP, ports, timestamp. Drop or hash high-cardinality strings (DNS query, HTTP URI, SSL subject). One-hot encode low-cardinality categoricals (proto, service).
- **Per-client stratified train/test split, done before any balancing or scaling.**
- **Federated scaler:** each client sends (count, sum, sum of squares) per feature. The server combines them into a global mean and std and broadcasts the result. This costs one extra round, leaks only aggregate statistics, and avoids each client scaling differently.

```python
# client
stats = (len(X), X.sum(0), (X**2).sum(0))
# server
n = sum(s[0] for s in all_stats)
mean = sum(s[1] for s in all_stats) / n
var = sum(s[2] for s in all_stats) / n - mean**2
std = np.sqrt(var).clip(min=1e-8)
```

**Gate (show before moving on)**

- Per-client class-count table for both datasets. Confirm the Ennio doorbell and Samsung webcam clients have zero Mirai rows.
- `tests/test_leakage.py`: no row appears in both a train and a test split, and no identifier column survives cleaning.

### Phase 1 — Centralized baselines (plain PyTorch)

**Build**

- `MLP`: F → 128 → 64 → C.
- `CNN-BiLSTM`: input reshaped to (F, 1) → Conv1d(1→32, k=3) → ReLU → Conv1d(32→64, k=3) → ReLU → MaxPool → BiLSTM(hidden 64) → Dense(64) → C.
- Class-weighted cross-entropy, Adam, early stopping on a validation slice of train.

**Gate**

- Macro-F1, per-class recall and FPR for both models on both datasets. This is the **upper bound**.
- Parameter count and serialized model size in bytes (float32 and float16). These numbers feed Phase 6.
- Decision recorded: does the CNN-BiLSTM beat the MLP? If not, say so in the report and keep both.

### Phase 2 — Local-only baseline

**Build:** each client trains only on its own partition and is evaluated on its own test set and on the global test set.

**Gate**

- The **lower bound** table.
- Evidence that the doorbell and webcam local models fail on Mirai. This is the "before" picture for Phase 4.

### Phase 3 — Federated simulation (Flower)

**Build**

- `ClientApp` wraps `src/train`. `ServerApp` uses Flower's FedAvg and FedProx strategies.
- Proximal term implemented client-side as the **squared** norm, matching the FedProx paper:

```python
global_params = [p.detach().clone() for p in model.parameters()]  # after loading global weights
...
prox = sum(((w - wg) ** 2).sum() for w, wg in zip(model.parameters(), global_params))
loss = criterion(model(x), y) + (mu / 2) * prox
```

- Log per-round global macro-F1, per-client metrics and payload bytes (`arrays_size_mod` / `message_size_mod`).

**Experiments:** see the matrix in Section 3.

**Gate**

- `tests/test_fedprox.py`: a run with μ = 0 matches the FedAvg run with the same seed.
- Comparison table: Centralized vs Local vs FedAvg vs FedProx (best μ), per dataset and per α.
- Convergence curves: macro-F1 vs round.

### Phase 4 — Zero-day experiments

**Build**

- **Locally unseen (natural):** after federated training, evaluate the doorbell and webcam clients on Mirai. Compare with Phase 2.
- **Truly unseen:** add a binary head (benign vs attack). Withhold one attack family entirely from all clients, train, then test on the withheld family. Rotate the withheld family.

**Gate:** recall on unseen attacks for local vs federated models, reported per withheld family.

### Phase 5 — Explainability (client-side SHAP)

**Build**

- On each client, run `shap.GradientExplainer` on the final global model with a small local background set. Fall back to Captum Integrated Gradients if it fails.
- Each client uploads only a per-class mean-|SHAP| vector. The server averages the vectors, weighted by sample count.

**Gate**

- Global top-15 feature plot per dataset.
- Agreement check: rank correlation between the federated importance vector and SHAP computed on the centralized model.
- Per-client top features, showing where devices differ.

### Phase 6 — Mininet systems experiments

**Build**

- `topology.py`: 1 server host and 9 client hosts on one switch, using `TCLink` with delay ∈ {5, 20, 50} ms and an optional bandwidth cap.
- Run Flower's deployment runtime: the server process on the server host and one client process per client host. Each client loads only its own partition file. Mininet hosts share a filesystem, so the partition-id config is what enforces this.
- Use the best Phase 3 configuration; do not rerun sweeps in Mininet.

**Gate**

- Round time vs link delay.
- Total bytes up and down per client, float32 vs float16.
- Bandwidth comparison: measured model traffic vs that client's feature-CSV size vs an estimated raw PCAP size. The estimation method is stated in the report.
- Macro-F1 under float16 matches float32 within noise.

### Phase 7 — Stretch goals (report under "Future Work / Roadmap")

These are reported only if built.

- Differential privacy using Flower's DP wrappers: the privacy–utility curve.
- SDN mitigation loop: the IDS flags a host and Ryu (Python 3.9 container) or OS-Ken installs a drop rule.
- Windowed-sequence variant of the CNN-BiLSTM.
- Straggler realism: clients drop out mid-round.

---

## 3. Experiment matrix

| ID | Dataset | Partition | Method | Varied | Purpose |
|---|---|---|---|---|---|
| E1 | Both | none (pooled) | Centralized MLP, CNN-BiLSTM | model | Upper bound |
| E2 | Both | natural / α = 0.5 | Local-only | model | Lower bound |
| E3 | N-BaIoT | natural | FedAvg, FedProx | μ ∈ {0, 0.001, 0.01, 0.1, 1} | Natural non-IID |
| E4 | TON_IoT | Dirichlet | FedAvg, FedProx | α ∈ {0.1, 0.5, 1.0} × μ | Controlled skew |
| E5 | Both | as E3/E4 | FedProx (best μ) | local epochs uniform vs random 1–5 | Systems heterogeneity |
| E6 | Both | as E3/E4 | FedProx (best μ) | weighted loss vs SMOTE | Imbalance ablation |
| E7 | N-BaIoT | natural | Federated vs local | — | Locally unseen attacks |
| E8 | Both | as E3/E4 | Binary head | withheld family | Truly unseen attacks |
| E9 | Both | best config | FedProx in Mininet | delay, float16 | Systems claims |

Fixed defaults unless varied: 50 rounds, all clients each round, 2 local epochs, batch size 256, learning rate 1e-3, 3 seeds.

---

## 4. Metrics and logging

- **Detection:** macro-F1 (headline), per-class precision and recall, FPR on benign, confusion matrix.
- **Federated:** rounds needed to reach 95% of the centralized macro-F1, and the spread of client-level F1 (fairness).
- **Systems:** bytes per round per client, round wall-time, model size.
- Every run writes `results/<run_id>.csv` plus its config YAML. Plots are generated only from these files.

---

## 5. Suggested work split

These are workstreams to assign, not fixed roles.

| Workstream | Phases | Owns |
|---|---|---|
| Data and baselines | 0, 1, 2 | Pipeline, leakage tests, model definitions |
| Federated core | 3, 4 | Flower apps, FedProx, sweeps, zero-day experiments |
| XAI and systems | 5, 6 | SHAP pipeline, Mininet topology, bandwidth measurement |

The Phase 0 gate blocks everyone, so all three people help finish it first.

Suggested review mapping: Phases 0–3 for the next review, Phases 4–6 for the final review, Phase 7 only if time allows.

---

## 6. Risks and fallbacks

| Risk | Fallback |
|---|---|
| N-BaIoT too large for laptops | Lower the per-(device, class) cap; document it |
| FedProx shows no gain over FedAvg | Report it honestly; E5 is where a gain is most likely |
| SHAP `GradientExplainer` errors on the BiLSTM | Captum Integrated Gradients, same aggregation |
| Flower API mismatch from online examples | Use only the pinned version's docs; add a CI import check |
| Mininet unavailable on the host OS | Use one shared Ubuntu VM; Mininet is needed only in Phase 6 |
| CNN-BiLSTM no better than MLP | Keep both in the report; argue on robustness and per-class recall, not headline F1 |
