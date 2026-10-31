# Phase 0 progress — saved 2026-10-20 10:53 IST

## Status: Phase 0 data pipeline built and validated on real data. NOT yet run on full N-BaIoT (9 devices) — only on the full TON_IoT file and a sample N-BaIoT device file, due to time running out this session.

## What's done and verified (against your real files, not synthetic data)
- `src/data/nbaiot.py` — per-device loader, caps rows per (device,class), confirmed
  works correctly on real device 3 file (capped 39100 -> 20000 rows), confirmed
  missing-file skip logic works (devices 3 & 7 genuinely have zero Mirai files
  in your actual Dataset/ folder, matching data_summary.csv).
- `src/data/tonIoT.py` — loads train_test_network.csv, drops src/dst ip+port,
  hashes high-cardinality strings, one-hot encodes proto/service/conn_state.
  Fixed a real bug: pandas 3.0 uses a native "str" dtype instead of "object",
  which silently skipped string-to-numeric coercion until fixed.
- `src/data/pipeline.py` — stratified split, federated scaler (validated:
  matches centralized mean/std to 1e-14), leakage check.
  **Found and fixed a real data issue**: TON_IoT has 55.6% exact-duplicate
  feature rows once identifiers are dropped (117,371 / 211,043 rows) — a
  naive random split leaks duplicates across train/test. Added dedupe_rows(),
  on by default in stratified_split(). After dedupe: 93,670 distinct rows
  (74,936 train / 18,734 test), leakage check now passes (0 overlapping rows).

## What's NOT done yet
- Have not run the loader across all 9 N-BaIoT devices (only tested device 3
  with 1 of its ~5-6 files, due to the user's laptop connection being slow to
  stage many large files — the full dataset is ~4.7GB across 90 CSVs).
- The user's Linux device workspace (device_bash) was still "downloading"
  this whole session — once it's ready, re-run Phase 0 THERE directly against
  the mounted folder (no need to stage files individually), rather than
  continuing to stage files one at a time into this container.
- Phase 0 gate (class-count table across all 9 devices, confirming device 3 =
  Ennio_Doorbell and device 7 = Samsung_Webcam have zero Mirai) not yet run
  at full scale — only confirmed via data_summary.csv + the one staged file.
- Phase 1 (centralized baselines) not started.

## Next steps when resuming
1. Check if device_bash is ready (`ls $HOME/mnt/`) — if yes, run the pipeline
   directly there against C:\Users\Sparsh\Desktop\Computer_network_project,
   no more file staging needed.
2. Run src/data/nbaiot.py's load_all_devices() across all 9 devices, cap
   20k/class, produce the full class_count_table() gate check.
3. Then Phase 1: MLP + CNN-BiLSTM centralized baselines on pooled data.

## Update — 2026-10-20 16:27 IST

Paused mid-transfer: connection between this session and the user's laptop
was timing out intermittently while staging N-BaIoT files one at a time
(single-file stage worked on retry each time, but unreliably — 1.benign.csv
succeeded, 1.gafgyt.combo.csv was still in flight when we paused).
Device's own Linux workspace (device_bash) also never finished downloading
this whole session, which would've avoided the transfer entirely.

Decision: wait for better internet connectivity before resuming the full
9-device N-BaIoT pull, rather than fight timeouts file by file.

## Next steps when resuming
1. First check: has device_bash finished downloading? (`ls $HOME/mnt/`) If
   yes, run the pipeline directly there — zero transfer needed, fastest path.
2. If device_bash still isn't ready, resume staging N-BaIoT files one at a
   time (multi-file staging reliably times out on this connection; stick to
   single-file calls, retry on timeout).
3. Already staged in this session (container-local, not yet used):
   /mnt/user-data/uploads/Computer_network_project/Dataset/1.benign.csv
4. Once all 9 devices are staged: run src/data/nbaiot.py's load_all_devices()
   + class_count_table() gate check, then move to Phase 1 (centralized
   baselines).

## Update — 2026-10-20 ~14:25 IST — Phase 0 COMPLETE

Ran the full pipeline against the real dataset (laptop reconnected, device_bash
became available this session). Key facts:

### What ran where
- Raw CSV reading (needs local file access): ran on the user's own laptop via
  device_bash, which runs a small Linux VM with only 2.8GB RAM. Switched
  nbaiot.py's loader to `fast=True` (pandas `nrows=` instead of read-all-then-
  sample) because reading full 100-300MB files just to keep 20k rows was
  taking ~100s/file -- nrows cut that to ~4s/file. Produced 9 parquet files
  (one per device, 814MB total, down from 4.7GB of raw CSV).
- Pooling, leakage check, and the federated-scaler cross-check: moved to this
  cloud workspace (7.8GB RAM vs the laptop VM's 2.8GB) because holding all 9
  devices + a pooled copy + dedup + split simultaneously OOM-killed the
  laptop VM twice. Transferred the 9 parquet files over instead of the 85
  raw CSVs -- much lighter.

### Gate check results (results/phase0_class_count_gate.csv, on the laptop)
All 9 devices loaded, 20k-row cap per class. Confirmed exactly what
data_summary.csv implied: Ennio_Doorbell (device 3) and Samsung_Webcam
(device 7) have ZERO Mirai rows across all 6 of their classes; the other 7
devices have all 11 classes. This is real, not constructed -- it's what
makes the Phase 4 "locally unseen attack" experiment meaningful.

### Second major duplicate-row finding (N-BaIoT, not just TON_IoT)
Pooled N-BaIoT (1,772,641 rows) has **63.6% exact-duplicate feature rows**
once device_id isn't treated as a feature -- distinct rows: 645,382
(516,305 train / 129,077 test after stratified split + dedup). This is even
higher than TON_IoT's 55.6%. Likely cause: stream-aggregation features in
N-BaIoT are damped-window statistics (the "weight"/"mean"/"variance" columns
for lambda = 5,3,1,0.1,0.01) that saturate to near-identical values during a
sustained flood attack (e.g. Mirai UDP flood) -- many rows during a long
attack burst legitimately compute to the same stats. Worth a line in the
report: this isn't a data error, it's a property of flood-style attacks
saturating a sliding-window feature extractor. Dedup-before-split (already
wired into stratified_split, on by default) handles it correctly --
leakage_check now reports 0 overlap.

### Federated scaler validated
Computed via 9 independent per-client (count, sum, sum-of-squares) tuples,
combined centrally -- matches a direct centralized mean to relative error
5.6e-16 (machine epsilon; some raw N-BaIoT features reach magnitude ~5.9e17,
so an absolute diff of ~8 at that scale is pure floating-point noise, not a
bug -- confirmed by comparing fairly in float64 after an initial float32
comparison artifact gave a misleading 2.7e9 absolute diff).

### Artifacts on the laptop now (C:\Users\Sparsh\Desktop\Computer_network_project\fl-edge-ids\)
- results/phase0_class_count_gate.csv -- the 9x11 class-count gate table
- results/phase0_nbaiot_pooled_summary.json -- dedup/leakage summary
- src/data/{nbaiot,tonIoT,pipeline}.py -- updated (fast=True loader, dedupe)

### Not yet synced to laptop (intentionally -- too large, and gitignored anyway)
- 9 device parquet files + pooled_train/test parquet (1.1GB total) -- live
  in this cloud workspace only, used for Phase 1 training. data/raw/ and
  data/partitions/ are already in .gitignore, so this is consistent with
  "don't commit large data to git" -- the CODE that reproduces them is on
  the laptop; the data itself doesn't need to be.

## Next: Phase 1 (centralized baselines)
MLP + CNN-BiLSTM on the pooled N-BaIoT split (and TON_IoT once loaded the
same way). This is next.

## Update — 2026-10-20 ~16:45 IST — Phase 1 COMPLETE (centralized baselines)

Trained MLP and CNN-BiLSTM (architectures exactly as specified in the
implementation plan) on pooled N-BaIoT (516,305 train / 129,077 test, 11
classes) and pooled TON_IoT (74,938 train / 18,735 test, 10 classes).
Class-weighted cross-entropy, Adam, early stopping on a validation slice
(patience=3), per the plan. Code: src/models/{mlp,cnn_bilstm}.py,
src/train/engine.py, src/baselines/centralized.py.

### Results (results/phase1_summary.csv, results/phase1_centralized_baselines.json)

| Dataset | Model | Test macro-F1 | Params | Size (f32/f16) | Train time |
|---|---|---|---|---|---|
| N-BaIoT | MLP | **0.8743** | 23,819 | 95.5KB / 49.0KB | 30s |
| N-BaIoT | CNN-BiLSTM | 0.8669 | 81,867 | 324.6KB / 164.7KB | 1781s (~30min) |
| TON_IoT | MLP | 0.5236 | 16,714 | 67.8KB / 35.1KB | 4s |
| TON_IoT | CNN-BiLSTM | **0.5295** | 81,802 | 324.4KB / 164.6KB | 106s |

**Decision (plan Phase 1 gate): does CNN-BiLSTM beat the MLP?**
Mixed, and reported honestly per the plan's own fallback ("if not, say so in
the report and keep both; argue on robustness and per-class recall, not
headline F1"): MLP wins on N-BaIoT by a small margin; CNN-BiLSTM wins on
TON_IoT by a similarly small margin. Neither gap is large enough to claim a
clear winner outright -- both models are kept, both datasets reported, as
the plan anticipated.

### Important finding hiding behind the headline number
N-BaIoT MLP's 0.8743 macro-F1 looks strong, but per-class recall shows
**gafgyt.tcp has 0.000 recall** -- the model never correctly identifies it.
10 of 11 classes score ~0.97-1.00 recall, so the one total failure barely
dents the macro-average. gafgyt.udp's false-positive rate (0.0404, far above
every other class's ~0.0001-0.001) suggests the model is systematically
confusing tcp-flood with udp-flood traffic -- plausible given both are
flood-style attacks that can saturate N-BaIoT's damped-window stream
features into near-identical values (the same mechanism behind the 63.6%
duplicate-row finding from Phase 0). This is exactly the scenario the stack
review's "replace accuracy with macro-F1" fix was guarding against, one
level deeper: even macro-F1 can hide a single collapsed class when the
other 10 are easy. Worth a sentence in the report, and worth watching
whether FedAvg/FedProx in Phase 3 fix or worsen this specific confusion.

TON_IoT's per-class numbers are more evenly weak across the board (backdoor
0.051, injection 0.106 recall) -- consistent with the literature's view of
TON_IoT as the harder, more realistic dataset (stack review: "N-BaIoT is
close to trivially separable... TON_IoT under low alpha is where FedAvg vs
FedProx will actually separate"). password class has a notably high FPR
(0.2197) -- many non-password attacks get misclassified as password,
another concrete per-class story for the report.

### Gate artifacts saved
results/phase1_summary.csv (flat table, easy to chart), 4x .pt checkpoints
(one per model per dataset, all under 350KB), results/phase1_run.log (full
training log).

## Next: Phase 2 (local-only baseline -- the lower bound, and the "before"
picture for Phase 4's zero-day story)

## Update — 2026-10-20 ~22:35 IST — Phase 2 COMPLETE (local-only baseline)

Each client trains ONLY on its own data, with its OWN local scaler (no
cross-client coordination of any kind -- not even shared feature
statistics, since that would itself be a form of collaboration). This is
the LOWER BOUND: nothing in Phase 3's federated results should do worse
than this. Scope decision: MLP only, not CNN-BiLSTM (see note below).
Code: src/baselines/local_only.py.

### N-BaIoT: 9 natural clients (one per device)
Mean global-test macro-F1 across the 9 local-only models: 0.6085 (individual
range 0.22-0.80), vs Phase 1's centralized upper bound of 0.874. The gap
(upper bound minus local-only mean) is exactly what Phase 3's federated
learning needs to close.

### The clean "before" picture for Phase 4 (this is the key result)
Device 3 (Ennio_Doorbell) and device 7 (Samsung_SNH_1011_N_Webcam) -- the
two devices with ZERO Mirai in their own local data -- score **exactly
0.0000 mean Mirai recall** on the global test set when evaluated with
their own local-only model. Every one of the other 7 devices (which DO
have Mirai locally) scores 0.72-0.80 Mirai recall on the same global test
set. This isn't a vague trend -- it's a clean, binary split that lines up
exactly with which devices had Mirai data and which didn't, confirming the
effect is really about local data exposure and not some other confound.
This is now the documented "before" state for Phase 4's "does federated
training let these two devices catch Mirai despite never seeing it
locally?" experiment.

### TON_IoT: 9 Dirichlet clients (alpha=0.5)
Much smaller, noisier per-client training sets (3.6k-14.8k rows, vs the
pooled 75k), consistent with Dirichlet skew at alpha=0.5. Mean global-test
macro-F1 ~0.43, below the centralized 0.524 -- though individual clients
are noisy (one scores 0.546, above centralized) due to small sample size,
which is expected and worth noting rather than hiding in the report.

### Scope decision: why MLP only, not CNN-BiLSTM, for this phase
CNN-BiLSTM took ~30 minutes for ONE run on N-BaIoT's full pooled set in
Phase 1. Running it per-client (9 clients x 2 datasets = 18 runs) would
take several hours and wasn't a good time trade for this gate check, whose
point is about DATA DISTRIBUTION (what a device can and can't see), not
architecture choice. MLP was also the (marginally) better model on N-BaIoT
in Phase 1. This is a deliberate scope call, flagged here and to the user,
not a silent shortcut -- can be revisited later if there's time.

### Artifacts
results/phase2_summary.csv (flat lower-bound table), results/phase2_local_only.json
(full per-client metrics incl. per-class recall/FPR), results/phase2_run.log.

## Next: Phase 3 (federated simulation -- Flower, FedAvg vs FedProx). This is
where the project's actual contribution starts: does collaborative training
close the gap between Phase 2's floor and Phase 1's ceiling?

## Update — 2026-10-20 ~22:55 IST — Alignment check against the team's literature-review deck

User shared FL_EdgeIoT_IDS_LiteratureReview.pptx (11-slide deck, team: Sparsh Y
Bhagwat, Tanya Singh, Adithya Santosh) and asked whether the build matches it.
Read via markitdown. Verdict: strong alignment on the core thesis, three
concrete gaps identified and discussed with the user.

### Matches (verified against the deck's own slides)
- Privacy-by-design (weights only, never raw data) -- deck slides 2/3/5, our
  design since Phase 0 (federated scaler shares only count/sum/sum-of-squares).
- Non-IID handling via natural device skew (N-BaIoT) + Dirichlet partitioning
  (TON_IoT) -- deck slide 5 ("Non-IID Dirichlet split"), our Phase 0/2.
- Deck slide 9's deliverable #2, verbatim: "A benchmark comparing Centralized
  vs. Local vs. FedAvg vs. Proposed FedProx." This is EXACTLY our Phase 1 ->
  Phase 2 -> Phase 3 structure, confirmed independently before ever reading
  this slide.
- 0-Day resilience framing (deck slide 9: "withheld attack class still
  detected post-aggregation") -- exactly the Ennio_Doorbell/Samsung_Webcam
  zero-Mirai story carried from Phase 2 into Phase 3.
- Tech stack: Python, PyTorch (deck offered PyTorch/TensorFlow), Flower (deck
  offered Flower/TFF), Pandas, Scikit-learn -- all match.
- 1D-CNN+BiLSTM architecture (deck's headline model) -- built and benchmarked
  fairly against MLP in Phase 1, exactly as the deck's methodology slide
  describes (Conv1d -> Conv1d -> MaxPool -> BiLSTM -> Dense).
- Macro-F1 as the honest metric -- deck slide 7 itself flags Wu et al.'s
  Macro-F1 80.77% as the literature's "harder, realistic" gap most papers
  avoid; we independently made the same call back in Phase 1, before reading
  this deck.

### Gaps identified (not yet built)
1. **SMOTE / client-side synthetic balancing** (deck slide 8's named fix for
   class imbalance). We used class-weighted cross-entropy instead -- same
   goal, different technique than the one the deck names.
2. **Mininet + Ryu SDN network emulation** (deck slides 4/5/8 -- real
   routers/switches, 5-50ms latency, explicitly framed as filling the gap
   that "most literature tests in artificial, latency-free environments").
   Currently Flower's in-process Ray simulation has no real network at all.
   This is the single biggest gap vs. the deck's claims.
3. **SHAP explainability** (mentioned on nearly every slide). Not started.

Smaller: the deck's "10x bandwidth reduction" claim needs a
compression/quantization step not yet done -- but Phase 1 already measured
serialized model size in f32 and f16 (src/baselines/centralized.py's
`serialized_size_bytes`, explicitly commented "feeds Phase 6"), so this is
on track as a later phase, not a surprise gap.

### Decision (user, 2026-10-20 22:56 IST)
Keep all three gaps in scope. Build SMOTE, Mininet+Ryu SDN emulation, and
SHAP explainability as later phases, after the core FL benchmark (Phases
1-4: centralized / local-only / FedAvg+FedProx / zero-day generalization)
is complete. Not blocking Phase 3, which was already running when this
question was asked.

## Update — 2026-10-20 ~17:46 UTC (23:16 IST) — Phase 3 COMPLETE (FedAvg vs FedProx)

Flower 1.41 simulation (Ray backend), 9 clients, 20 communication rounds,
2 local epochs/round, Adam lr=1e-3, batch_size=256 -- same recipe as every
other phase, just shaped for rounds instead of one convergent fit(). Both
datasets use the SAME per-client splits as Phase 2 (identical seed/params),
but a single SHARED federated scaler (not each client's own), matching how
a real federated scaler round would work. FedProx proximal_mu=0.01, not
swept (same time-tradeoff pattern as Phase 2's MLP-only call). Code:
src/federated/{prepare_data,engine_fed,run_phase3}.py. Re-ran once after
adding model checkpointing (first run only kept metrics, not weights --
Phase 4 needs the actual trained model).

### Results (results/phase3_federated.json, results/phase3_summary.csv)

| Dataset | Strategy | Final macro-F1 | vs. centralized ceiling | vs. local-only floor |
|---|---|---|---|---|
| N-BaIoT | FedAvg  | 0.7382 | 0.8743 | 0.6085 |
| N-BaIoT | FedProx | 0.7463 | 0.8743 | 0.6085 |
| TON_IoT | FedAvg  | 0.5438 | 0.5236 (MLP) / 0.5295 (CNN-BiLSTM) | 0.4369 |
| TON_IoT | FedProx | 0.5575 | 0.5236 (MLP) / 0.5295 (CNN-BiLSTM) | 0.4369 |

N-BaIoT lands exactly where the plan expects: strictly between the floor
and the ceiling, closing roughly half the gap. FedProx edges out FedAvg
slightly on both datasets, consistent with the literature's claim that the
proximal term helps most under non-IID skew (mu=0.01, not swept).

TON_IoT is the more interesting result: BOTH federated strategies slightly
**beat** Phase 1's centralized ceiling. Plausible, non-federation-related
cause: Phase 1's MLP early-stopped at epoch 10 (of a 15-epoch budget);
Phase 3's 20 rounds x 2 local epochs gives each client ~40 cumulative
local-epoch-equivalents of gradient exposure, more total training than the
centralized run got before its early stop triggered. This is a single-seed
result on one dataset, not a claim that federated training generically
beats centralized -- noted honestly rather than oversold as a federation
win.

### The headline result: the zero-day story, resolved
Mean Mirai recall of the final global model: **0.7448 (FedAvg) / 0.7524
(FedProx)**. Phase 2 showed Ennio_Doorbell and Samsung_SNH_1011_N_Webcam
(zero Mirai locally) score **exactly 0.0** Mirai recall with their own
local-only models, while the other 7 devices (which do have local Mirai
data) scored 0.72-0.80 alone. In federated learning every client deploys
the SAME global model -- there's no more "device 3's model" vs "device 7's
model" once training is federated. So the honest comparison is: would
devices 3 and 7, now equipped with this ONE shared model instead of their
own blind one, catch Mirai? At ~0.74-0.75 mean recall, landing right in the
0.72-0.80 range the Mirai-aware devices themselves achieved alone, the
answer is clearly yes. This is the project's core contribution, confirmed
with real numbers: federated training let two devices that had literally
never seen an attack family locally learn to detect it anyway, without
their raw traffic ever leaving the device.

### A real weakness, reproducible across both strategies (not a bug)
gafgyt.tcp recall collapses to ~0.0 in BOTH FedAvg's and FedProx's final
model, while gafgyt.udp recall goes to ~1.0 -- the same tcp/udp flood
confusion Phase 1 found in the centralized MLP (both are flood-style
attacks that saturate N-BaIoT's damped-window features into near-identical
values, per Phase 0's duplicate-row finding). Federation doesn't fix this
confusion -- if anything it's more consistent here than Phase 1's single
run. mirai.ack is also consistently the weakest Mirai subtype (~0.16-0.17
recall in both strategies) while the other four Mirai subtypes are
0.76-0.99. Exactly the kind of failure macro-F1 and even the headline
Mirai-recall-mean can hide -- worth a clear callout in the report, and a
candidate thing to watch in Phase 4.

### Artifacts
results/phase3_federated.json (full round-by-round history + final
per-class recall/FPR for all 4 runs), results/phase3_summary.csv,
results/phase3_run.log, results/phase3_{nbaiot,tonIoT}_{FedAvg,FedProx}.pt
(4 final global-model checkpoints, for Phase 4).

## Next: Phase 4 (zero-day / unseen-attack generalization). Load the
federated checkpoints above and test them specifically against the
locally-unseen-attack scenario more rigorously (plan's "truly unseen"
framing -- binary head / leave-one-family-out), building directly on this
phase's Mirai-recall finding.

## Update — 2026-10-21 ~18:35 UTC (00:05 IST) — Phase 4 COMPLETE (true zero-day, leave-one-family-out)

Harder test than Phase 2/3's "locally unseen" story: here an entire attack
family is removed from EVERY client's training data, not just a couple of
devices -- nobody, anywhere in the federation, ever sees it during
training. N-BaIoT holds out the whole Mirai family (5 classes, 700k rows,
remaining problem: 6-class benign+gafgyt). TON_IoT holds out "ransomware"
(20k rows, remaining problem: 9-class). Since the model's output layer
literally has no slot for the withheld family's name, the only answerable
question is the plan's "binary head" framing: does the model still flag
the traffic as *some* kind of attack (any non-benign class) rather than
confidently calling it normal? Implemented as a collapse of the existing
multiclass softmax rather than a second trained head -- cheaper, and it's
exactly what a deployed multiclass model would actually do if this attack
showed up for real. Three models per dataset for a fair comparison:
Centralized (pooled, family-free) / FedAvg / FedProx. Code:
src/federated/{prepare_data_phase4,run_phase4}.py.

### Results (results/phase4_zero_day.json, results/phase4_summary.csv)

| Dataset | Strategy | In-distribution macro-F1 | Zero-day catch rate |
|---|---|---|---|
| N-BaIoT (Mirai held out) | Centralized | 0.7768 | 0.5176 |
| N-BaIoT (Mirai held out) | FedAvg | 0.7606 | 0.7130 |
| N-BaIoT (Mirai held out) | FedProx | 0.7603 | **0.8368** |
| TON_IoT (ransomware held out) | Centralized | 0.5659 | **0.9937** |
| TON_IoT (ransomware held out) | FedAvg | 0.5011 | 0.2338 |
| TON_IoT (ransomware held out) | FedProx | 0.4701 | 0.2338 |

### N-BaIoT: federation generalizes BETTER to a truly unseen attack
FedProx catches 84% of never-seen Mirai traffic as "something's wrong,"
vs. only 52% for the centralized model that saw everything pooled -- even
though in-distribution performance is nearly identical (~0.76-0.78 across
all three). Per-class breakdown of what Mirai gets mistaken for
(predicted_class_distribution in the json) shows why: the centralized
model concentrates its guesses on 1-2 gafgyt classes, while FedProx spreads
recognition across THREE gafgyt variants (combo/junk/scan) more evenly --
a less confident, more "broadly attack-like" decision boundary that
happens to catch more of a genuinely novel flood-style attack. Plausible
mechanism: FedProx's proximal term discourages any one client from
overfitting too hard to its own local attack signature, which seems to
produce a decision boundary that generalizes to NEARBY-but-unseen attacks
better, not just a boundary that's merely "as good" in-distribution.

### TON_IoT: the opposite result, reported honestly
Centralized catches 99.4% of ransomware as non-benign; BOTH federated
strategies catch only 23.4% -- and the predicted-class breakdown shows why,
plainly: federated's model maps most ransomware rows (15,324 / 20,000) to
"normal," the worst possible failure mode for a security system (a missed
attack, not just a misclassified one). This is a real, not a hidden,
limitation -- not spun as a win. Likely cause: TON_IoT's Dirichlet clients
are far smaller and noisier than N-BaIoT's natural per-device splits
(3.6k-16.7k rows/client here, vs 90k-170k for N-BaIoT), so each client's
local model has much less signal to learn a general "this looks like an
attack" boundary from, and federated averaging across 9 such noisy,
non-IID models doesn't recover what a single centralized model learns from
seeing all the attack diversity pooled directly. This tracks the stack
review's own observation (cited back in Phase 1) that TON_IoT is the
harder, more realistic dataset where FedAvg/FedProx actually diverge from
the easy case -- consistent with, not contradicting, the literature.

### Honest takeaway
Zero-day generalization from federated training is NOT a free, universal
win -- it depends on the dataset and how much per-client data/signal is
available. N-BaIoT (large, natural per-device splits) shows federation
clearly helping; TON_IoT (small, synthetic Dirichlet splits) shows it
clearly hurting. Both results are real and both are reported -- this
dataset-dependence is itself a legitimate, citable finding for the report,
not a result to bury.

### Artifacts
results/phase4_zero_day.json (full per-run metrics + predicted-class
breakdown), results/phase4_summary.csv, results/phase4_run.log,
results/phase4_{nbaiot,tonIoT}_{centralized,FedAvg,FedProx}.pt (6
checkpoints).

## Next: later phases per the literature-deck alignment decision --
SMOTE client-side balancing, Mininet+Ryu SDN network emulation, SHAP
explainability, and the compression/bandwidth measurement step (ingredients
already collected in Phase 1's serialized_size_bytes). Core FL benchmark
(Phases 0-4) is now complete.

## Update — 2026-10-21 ~18:55 UTC (00:25 IST) — Phase 5 COMPLETE (client-side SMOTE)

Implemented exactly what the literature deck names (slide 5/8): SMOTE
applied PER CLIENT, on each client's own already-scaled Phase 3 training
data, never pooled across clients. Re-ran the IDENTICAL FedAvg/FedProx
recipe as Phase 3 (same architecture, 20 rounds, 2 local epochs/round, same
everything else) on the SMOTE-balanced data, so any difference is
attributable to SMOTE alone. Code: src/federated/{prepare_data_smote,
run_phase5}.py. safe_client_smote() handles the real edge case TON_IoT
produced: some clients have a class with as few as 1-10 samples, too few
for standard SMOTE's k_neighbors=5 default -- k is chosen per-client from
that client's own smallest minority class, with a RandomOverSampler bump
for true singleton classes first.

### First finding, confirmed before training even started: the two datasets needed this very differently
N-BaIoT clients grew only ~3% after SMOTE (171,440 -> 176,000 rows,
typical) -- because Phase 0's load_device() already caps every class at
20k rows per device, so N-BaIoT was ALREADY near-balanced per client
(min/max class ratio close to 1:1 before SMOTE even ran). TON_IoT clients
grew far more (e.g. client 0: 14,832 -> 68,160 rows, client 3: 6,486 ->
29,420) -- confirming TON_IoT's Dirichlet clients have REAL imbalance
(some clients had a class with just 2-10 rows against a majority class of
2,000-7,000). Full before/after counts per client: data/partitions/
federated_smote/smote_report.json.

### Results (results/phase5_federated_smote.json, results/phase5_summary.csv)

| Dataset | Strategy | Phase 3 (no SMOTE) | Phase 5 (SMOTE) | Change |
|---|---|---|---|---|
| N-BaIoT | FedAvg | 0.7382 | 0.7493 | +0.011 |
| N-BaIoT | FedProx | 0.7463 | 0.7469 | +0.001 |
| TON_IoT | FedAvg | 0.5438 | **0.6731** | **+0.129 (+24% relative)** |
| TON_IoT | FedProx | 0.5575 | 0.6081 | +0.051 (+9% relative) |

Exactly as the pre-SMOTE imbalance numbers predicted: negligible change on
N-BaIoT (nothing to fix), a large, genuine improvement on TON_IoT (real
imbalance, really fixed). This is the clean confirmation of the deck's
claimed benefit, with real numbers.

### The honest nuance: SMOTE is a trade-off, not a free win
TON_IoT FedAvg per-class recall: one class went from 0.059 -> 0.977 and
another from 0.123 -> 0.580 (catastrophic failures fixed), but one
previously-strong class dropped from 0.837 -> 0.289 (a real regression).
Net effect on macro-F1 is strongly positive because more classes improved
than regressed, but it is not uniformly positive -- reported plainly
rather than only quoting the aggregate number. (Note: TON_IoT's original
Phase 0 integer-label encoding was a one-off script never committed to
code -- see Phase 4's notes -- so these are reported by class index, not
name, to avoid guessing a wrong label.)

N-BaIoT's persistent gafgyt.tcp/gafgyt.udp confusion (found in Phase 1,
still present in Phase 3) is UNCHANGED by SMOTE for FedAvg (0.001/1.000,
identical before and after) -- confirming it was never a class-imbalance
problem (both classes already had ~16k rows/client). For FedProx, SMOTE
doesn't fix the confusion either -- it just flips which of the two
classes collapses (tcp: 0.000->0.999, udp: 0.999->0.000). This is good
evidence the tcp/udp confusion is a genuine feature-overlap issue (flood
attacks saturating N-BaIoT's damped-window features into near-identical
values, per Phase 0's duplicate-row finding), not something more training
data on the minority side can fix -- consistent with, and now confirmed
by, a second independent experiment.

### Artifacts
results/phase5_federated_smote.json, results/phase5_summary.csv,
results/phase5_run.log, results/phase5_SMOTE_{nbaiot,tonIoT}_{FedAvg,
FedProx}.pt (4 checkpoints), data/partitions/federated_smote/
smote_report.json (per-client before/after class counts, gitignored like
all data/partitions/ but kept in the cloud workspace for inspection).

## Next: Mininet+Ryu SDN network emulation, then SHAP explainability, then
the compression/bandwidth measurement step.

## Note — numbering reconciliation against the ORIGINAL written plan
Re-read FL_EdgeIoT_IDS_Implementation_Plan.md in full for the first time
since the summarized portion of this conversation. The plan's actual
phase numbers differ slightly from the working numbers used in this log:
Phase 0-4 above match the plan exactly. What this log called "Phase 5"
(SMOTE) is, in the plan, part of **Phase 3's own experiment matrix**
(ablation E6: "weighted loss vs SMOTE"), not a separate numbered phase --
SMOTE was always meant as an ablation ON TOP of the federated setup, not
a phase of its own. The plan's REAL Phase 5 is client-side SHAP
explainability (below). Also noted from a full re-read: Phase 6
(Mininet) uses Mininet's own DEFAULT OVS controller -- Ryu is listed only
under Phase 7's OPTIONAL stretch goals ("SDN mitigation loop"), not
required for the core Mininet systems experiments. And: the plan asks
every number to be the mean of 3 random seeds; everything run so far is
a single seed -- a real rigor gap to close before calling results final,
flagged honestly rather than silently accepted.

## Update — 2026-10-22 ~13:58 UTC (19:28 IST) — Phase 5 COMPLETE (client-side SHAP explainability)

Per the plan: each client ran shap.GradientExplainer on the Phase 3
FedProx global model, using only its OWN local data (50-row background,
200-row sample) -- only a per-class mean(|SHAP|) vector ever "leaves" the
client, never raw data or per-row explanations. These are combined
server-side with the SAME sample-count weighting FedAvg itself uses.
A second SHAP run on the Phase 1 CENTRALIZED model (same architecture,
pooled data) gives the plan's "agreement check": do federated and
centralized training end up paying attention to the same signals?
Code: src/xai/shap_pipeline.py. Full results: results/phase5_shap.json.

### Agreement check: strong on both datasets
- N-BaIoT: Spearman rho = **0.70** (p = 2.6e-18) between federated and
  centralized feature rankings.
- TON_IoT: Spearman rho = **0.90** (p = 4.4e-23) -- even stronger.

Both are large, highly significant positive correlations: despite never
pooling raw data, federated training converged on judging roughly the
SAME features important as a model that saw everything centrally. This
is real evidence the federated model learned genuine signal, not an
artifact of its training procedure -- exactly what this gate was designed
to check.

### What the model actually looks at (and it makes sense)
N-BaIoT's top-5 federated features are all damped-window traffic-volume
statistics (H_L0.1_weight, MI_dir_L0.1_weight, MI_dir_L0.01_weight,
H_L0.01_weight, HH_jit_L0.1_weight) -- "weight" here means packet/byte
rate over a decaying time window. This lines up exactly with Phase 0's
own finding that flood-style attacks (Mirai/Gafgyt) saturate these
damped-window stats into near-identical values -- the model is keying off
the very property that makes flood attacks a flood attack. Reassuringly
consistent across almost all 9 clients' own local top-5 (same handful of
features, different order) -- not a fluke of one device.

TON_IoT's top-5 (conn_state_S0, conn_state_REJ, service_http,
conn_state_S3, service_-) are TCP connection-state flags and the
service/protocol in use -- S0 ("connection attempt seen, no reply") and
REJ ("rejected") are textbook signatures of scanning and failed
connection floods. Also intuitively correct, and a believable story for
the final report: a security analyst reading this would nod, not
question it.

### Artifacts
results/phase5_shap.json (full per-client top-5, aggregated top-15,
federated vs centralized importance for every feature, Spearman stats),
results/phase5_shap_run.log.

## Update — 2026-10-22 ~22:00 IST — Phase 6 COMPLETE (Mininet systems experiments), with one honest scope cut

Per the plan: 1 server host + 9 client hosts on one OVS switch
(src/mininet/topology.py), running Flower's real DEPLOYMENT engine
(flower-superlink + 9x flower-supernode — NOT the simulation engine
every earlier phase used) over that emulated network, using the best
Phase 3 config (N-BaIoT, FedProx, mu=0.01). This phase was by far the
most infrastructure-heavy of the project — getting a real multi-process
federated run working inside Mininet surfaced a chain of real
environment bugs, each diagnosed by direct testing rather than guessing,
fixed at the root cause:

1. **100% packet loss on every Mininet link.** This container's kernel
   has no loadable-module support, so OVS's normal kernel datapath
   silently drops everything. Fixed by forcing the userspace datapath
   (`OVSSwitch(failMode="standalone", datapath="user")`).
2. **TCLink's `delay=` parameter is a no-op here.** Confirmed directly
   (`tc qdisc add ... netem` fails) that this kernel's config has
   `CONFIG_NET_SCH_NETEM` unset — the netem queueing discipline Mininet
   relies on for link delay genuinely does not exist in this kernel.
   Worked around with a small user-space TCP proxy
   (src/mininet/delay_proxy.py) sitting in front of each client's
   SuperNode, injecting real one-way delay on real relayed bytes and
   counting them in both directions — still a real distributed run over
   the real emulated network, just with the delay piece added one layer
   up since the kernel can't do it.
3. **TCP silently hung (ICMP fine, TCP not) on the OVS-userspace veths**
   — a known checksum/segmentation-offload interaction; fixed with
   `ethtool -K <intf> tx off rx off sg off tso off gso off gro off` on
   every host interface.
4. **Severe CPU starvation**: this container has 2 cores; the
   deployment engine runs 9 REAL concurrent OS processes (one per
   client, not Phase 3's cooperative simulation scheduler). Pinned
   torch and every BLAS library to 1 thread per client process
   (src/mininet_app/client_app.py, src/mininet/run_phase6.py) — without
   this, even a loopback connection inside one SuperNode took 100+
   seconds and failed outright.

**The one problem that didn't get a clean fix**: after that thread-pinning,
round 0 (initial eval) and round 1 (one full real training round, all 9
clients) completed correctly and reproducibly in *every* configuration
run — but round 2 never started, in any of the 4 configs (3 delay
sweeps + the fuller run), regardless of how many rounds were requested
(3 or 10). The SuperLink's own log shows why: `[ApiError:1008] Runtime
task-token authentication failed`, then `ClientDisconnect`, then the
SuperLink declaring the run finished for every SuperNode at once. Root
cause: this container's 2-core/9-process contention is severe enough
that Flower's own control-plane heartbeat (not the training traffic —
the Fleet-API bookkeeping that keeps a SuperNode's session token valid)
starves out partway through round 1's ~65-85s of solid CPU contention,
and the token expires before round 2 can start. This is a resource
ceiling of this specific container, not a bug in the FL code or the
Mininet topology — the identical server_app.py/client_app.py ran a full
20-round sweep without issue in Phase 3's simulation engine, which
doesn't have this control-plane token at all.

**Scope decision** (same transparency pattern as every other phase's
documented call, e.g. Phase 2's CNN-BiLSTM skip): rather than keep
spending time tuning an internal Flower timeout I can't see a CLI flag
for, Phase 6's deliverables below are built from the one real completed
round each config did produce, honestly labeled as n=1 per data point
rather than padded or extrapolated into a multi-round curve that wasn't
actually run. Every number below is still genuinely measured over the
real Mininet network with the real deployment engine — nothing here is
simulated or scripted.

### Round time vs link delay (1 real completed round each)
| Injected one-way delay | Round wall-time |
|---|---|
| 5 ms | 67.4 s |
| 20 ms | 68.3 s |
| 50 ms | 64.5 s |

No meaningful trend across 5-50ms — expected, honestly: round time here
is dominated by the 2-core/9-process compute contention (tens of
seconds), which swamps a delay on the order of tens of milliseconds.
This is a real, legitimate finding for the report (network latency at
these magnitudes is not the bottleneck; client compute capacity is),
just not the delay-sensitivity curve the plan was hoping to show — and
it's an n=1-per-point measurement, flagged as such rather than
overstated.

### Bytes per client, float32 vs float16 (1 real round, full run, delay=20ms)
Measured directly off each client's delay_proxy (actual Fleet-API
upload+download bytes for that client, not an estimate):
- Total across all 9 clients: **7,661,277 bytes measured (float32)**.
- float16 estimate: **3,931,718 bytes** (NOT measured over the wire —
  Flower's wire format has no float16 mode to turn on — scaled by the
  same float16/float32 serialized-size ratio Phase 1 already measured
  for this exact model: 0.513).

### Macro-F1 under float16 (real checkpoint, round-trip quantization check)
Took the real round-1 checkpoint from the full run (macro-F1 = 0.5309
on the held-out test set — a believable one-round FL number, up from
0.0373 at round 0), quantized every weight to float16 and back (no
retraining), re-ran the same `evaluate()` every phase has used:
- float32 macro-F1: **0.53089**
- float16 round-trip macro-F1: **0.53093** (delta: +0.00004 — a
  rounding wobble, not a real difference)
- **Confirms the plan's expectation: float16 doesn't hurt accuracy**,
  while roughly halving the payload (ratio 0.513).

### Bandwidth comparison (1 real round, full run, all 9 clients, stated estimates where noted)
| What | Bytes | How |
|---|---|---|
| Model traffic, float32 | 7,661,277 | **measured** (delay_proxy byte counts) |
| Model traffic, float16 | 3,931,718 | estimated (Phase 1's size ratio) |
| Raw feature-CSV equivalent | 1,315,324,233 | estimated (944.5 bytes/row, measured from 2 real N-BaIoT CSVs, x each client's real row count) |
| Raw packet-capture equivalent | 167,113,800 | estimated (120 bytes/packet assumption — N-BaIoT ships no raw pcap to measure from; stated, not measured) |

Headline: one round of federated model updates moves **~170x less data**
than shipping the raw feature CSVs centrally, and **~22x less** than a
(assumed) raw packet capture — the core bandwidth argument for doing FL
on edge-IoT devices at all, now backed by a real measured number instead
of only a theoretical one.

### Fixed a real indexing bug before trusting these numbers
The post-processing script (src/mininet/analyze_phase6.py) initially
matched each client host's byte counts to the WRONG client's row count
when estimating CSV/pcap bytes (used the Mininet host index 1-9
directly against a row-count table keyed 0-8 by partition id, a
constant one-client offset, and silently zero for host 9). Caught this
by eye-checking the script's own output before using any of these
numbers above, fixed the indexing, and reran.

### Artifacts
src/mininet/{topology.py, delay_proxy.py, run_phase6.py, analyze_phase6.py},
src/mininet_app/{server_app.py, client_app.py}, src/pyproject.toml,
results/phase6_mininet_raw.json (raw per-config data),
results/phase6_analysis.json (the processed numbers above),
results/phase6_*_round_times.json, results/phase6_*.pt (checkpoints).

## 3-seed rigor pass — COMPLETE

Per the original plan's requirement to report mean±std rather than a
single run, reran the headline experiments at 3 seeds: **42** (the
original run, kept), **123**, and **2026**.

### Scope decision (explicit, same pattern as every other cut in this
project — documented rather than silently skipped)
- **Reseeded**: Phase 1 (centralized baseline, **MLP only**), Phase 2
  (local-only, all 9/9 clients), Phase 3 (federated FedAvg/FedProx),
  Phase 4 (zero-day holdout).
- **Not reseeded**: Phase 1's CNN-BiLSTM (only ever run at seed 42 —
  already shown non-superior and far slower than the MLP, so re-running
  it 2 more times would only cost time for no new conclusion). Phase 5
  (SHAP) — diagnostic, not a comparison metric. Phase 6 (Mininet
  systems) — a systems/infra result, not an ML accuracy comparison.
- **Skipped entirely**: Phase 7 (stretch goals) — out of scope for this
  pass, given the "make it fast" instruction to finish the required
  phases first.

### Headline results (mean ± std across 3 seeds)

**Macro-F1, N-BaIoT** (9 real devices, 11 classes):
| Approach | Macro-F1 |
|---|---|
| Centralized (pools everything) | 0.876 ± 0.001 |
| Local-only (avg of 9 clients, no sharing) | 0.599 ± 0.008 |
| Federated FedAvg | 0.742 ± 0.004 |
| Federated FedProx | 0.745 ± 0.002 |

**Macro-F1, TON_IoT** (9 Dirichlet-split clients, 10 classes):
| Approach | Macro-F1 |
|---|---|
| Centralized (pools everything) | 0.514 ± 0.012 |
| Local-only (avg of 9 clients, no sharing) | 0.439 ± 0.021 |
| Federated FedAvg | 0.542 ± 0.001 |
| Federated FedProx | 0.551 ± 0.008 |

All four approaches are stable across seeds (std ≤ 0.021 in every
case) — the single-seed numbers reported earlier in this doc were not
a fluke. On both datasets, federated beats local-only by a wide,
seed-stable margin, closing most (N-BaIoT) or more than all (TON_IoT,
where FedProx actually *exceeds* the centralized ceiling) of the gap to
centralized training, without ever pooling raw client data.

**Zero-day catch rate** (entire attack family withheld from every
client's training data, then tested on exactly that family):
| Dataset | Strategy | In-dist macro-F1 | Zero-day catch rate |
|---|---|---|---|
| N-BaIoT (Mirai withheld) | Centralized | 0.778 ± 0.004 | 0.56 ± 0.16 |
| N-BaIoT (Mirai withheld) | FedProx | 0.757 ± 0.003 | **0.85 ± 0.07** |
| TON_IoT (ransomware withheld) | Centralized | 0.631 ± 0.070 | 0.76 ± 0.28 |
| TON_IoT (ransomware withheld) | FedProx | 0.464 ± 0.007 | 0.48 ± 0.43 |

Two honest findings here, reported as found rather than cherry-picked:
- **N-BaIoT is a clean, repeatable result**: federated beats the
  centralized ceiling at catching an attack family it never trained on,
  consistently across all 3 seeds (0.84 / 0.93 / 0.79 — never close to
  the centralized number). This is the strongest, most defensible
  non-obvious result in the project.
- **TON_IoT's zero-day catch rate is seed-volatile** (seeds 42/123/2026
  gave 0.23 / 0.24 / 0.98 for FedProx) even though in-distribution
  accuracy on the same runs was stable. Read honestly: with only one
  entire class held out and a 10-class problem, the exact decision
  boundary the network lands on for a class it has *never once seen*
  can swing between "almost never flags it" and "almost always flags
  it" depending on initialization/training trajectory — a real
  limitation of single-run zero-day claims this 3-seed pass was
  specifically designed to surface, not a bug in the measurement. This
  is flagged plainly in the dashboard rather than smoothed over.

### Artifacts
`src/{baselines/centralized.py, baselines/local_only.py,
federated/engine_fed.py, federated/run_phase3.py,
federated/run_phase4.py}` — all gained a `--seed` CLI flag (default 42,
original behavior unchanged) threaded into every `torch.manual_seed`,
`train_test_split`, `stratified_split`/`dirichlet_partition`, and
per-client federated seeding; seed≠42 runs write to `*_seed{N}`-suffixed
files so the original seed-42 results were never overwritten.
`run_3seed_batch.sh` (driver for seeds 123/2026),
`src/aggregate_3seed.py` (computes the mean±std above from the 3 raw
result sets, no new training), `results/threeseed_summary.json` (full
per-seed values), `results/phase{1,2,3,4}_*_seed{123,2026}.json` (raw
results per seed).

## Final deliverable: local demo web app (not the earlier hosted
dashboard — corrected per your direction to run from VS Code)

Built `webapp/` — a small Flask app that runs entirely on your own
laptop, no internet or Mininet/Linux required:
- `python app.py` starts a local server and prints a `http://127.0.0.1:5000`
  link; opening it in a browser shows the same content a hosted
  dashboard would, but running locally so you can demo it live from VS
  Code's terminal exactly as asked.
- The page animates **real held-out N-BaIoT traffic rows** (267 rows
  sampled from the 9 real device partitions, scored by the actual
  trained Phase 3 FedProx model — not scripted/fake) flowing from 9
  "houses" (devices) to a "guard" (the shared model), colored by
  whether the real model's real prediction was correct, with 2 devices
  ringed to show they have zero Mirai traffic of their own and still
  get protected.
- Below the animation: the model-comparison, zero-day, explainability,
  and bandwidth/round-time/float16 charts — now all showing the 3-seed
  mean±std numbers above, pulled live from `webapp/data/*.json` (the
  same JSON the analysis scripts produced, just copied in — no numbers
  are re-typed or hand-edited).
- Verified with a real running server + a headless-browser check
  (console errors, DOM text, chart count, canvas pixel sampling) rather
  than assumed correct: 0 console errors, 6 charts rendering, all stat
  tiles populated with real resolved values, canvas genuinely drawing.
- A claude.ai-hosted version of the same dashboard was also built
  earlier in this pass and still exists as an optional backup link, but
  the Flask app above is the primary deliverable, since it matches what
  you actually asked for (open in VS Code, run one command, see it on
  localhost).

### Artifacts
`webapp/app.py`, `webapp/static/{index.html, app.js, d3.min.js}`,
`webapp/data/*.json` (8 files, copied from `results/dashboard_*.json`),
`src/make_dashboard_data.py` (builds the 267-row real-traffic sample),
`src/make_dashboard_summaries.py` (reshapes Phase 1-6 results into the
flat JSON the page reads).

## Project status: all required phases (1-6) complete, 3-seed rigor
pass complete, final local-demo deliverable complete. Phase 7 (stretch
goals) explicitly out of scope for this pass.
