# FairRFL reproduction spec

Paper: A. Augello, A. Gupta, G. Lo Re, S. K. Das, "FairRFL: Fair and Robust Federated Learning in the Presence of Selfish Clients", IEEE Trans. Emerging Topics in Computing, vol. 14, no. 1, Jan-Mar 2026, pp. 316-331 (extension of their ECAI 2024 paper "Tackling selfish clients in federated learning", ref. [13]).
Sources: `docs/impl-papers/FairRFL ...pdf` (printed page numbers 316-331 used below; PDF page = printed - 315); official code github.com/ndslab-group/FairRFL (files: experiments.py, Network.py, create_datasets.py, selfishness.py, utils.py; no README, no run scripts, no config files). Symbols: phi = selfishness (code `--selfishness`), Gamma = aggregation normalizer (k for FedAvg, sum of h_i for (D)q-FFL), tau = detection threshold.

## 1. Datasets

**CIFAR-10** (torchvision). Normalize mean (0.4914, 0.4822, 0.4465), std (0.2023, 0.1994, 0.2010); no augmentation (create_datasets.py `get_dataset`). 10 classes.

**WISDM-W** (watch modality of WISDM, via the FedAIoT benchmark, p. 326):
- Download: FedAIoT `datasets/wisdm/download.py` fetches `https://archive.ics.uci.edu/ml/machine-learning-databases/00507/wisdm-dataset.zip` (UCI "WISDM Smartphone and Smartwatch Activity and Biometrics Dataset"). Extract so that `data/wisdm-w/watch/accel/data_<id>_accel_watch.txt` and `data/wisdm-w/watch/gyro/data_<id>_gyro_watch.txt` exist, and copy FedAIoT's `datasets/wisdm/activity_key_filtered.txt` to `data/wisdm-w/`.
- `activity_key_filtered.txt` keeps 12 activities and remaps codes: walking A->0, jogging B->1, stairs C->2, sitting D->3, standing E->4, typing F->5, teeth G->6, eating L->7, drinking K->8, writing Q->9, clapping R->10, folding S->11.
- Preprocessing (create_datasets.py `process_dataset`, `filter_merge_interval`, `normalize_data`, `create_dataset`, `load_dataset`): subjects 1600-1650 (51); per subject, crop accel and gyro to their common time range, inner-join on identical timestamp, drop NaN, keep the 12 activities; z-score each of the 6 channels (x/y/z acc, x/y/z gyro) over the whole dataset; windows of 200 samples with 50% overlap (stride 100) per subject and activity (the loop starts at index 200, so the first window of each activity segment is skipped); X shape (200, 6), label = activity. Cached to `data/wisdm-w/processed_watch.csv` and `wisdm_watch.pkl`.
- Train/test: first 80% of windows (ordered by subject id) = train, last 20% = test (no shuffle), i.e. the test pool comes mostly from the highest subject ids. Natural per-subject clients are NOT used; the same 2-class split as CIFAR-10 is applied.

## 2. FL setup and training hyperparameters

| Item | Paper (Sec. VI, p. 326) | Code (experiments.py, utils.py) |
|---|---|---|
| Clients | 50 | `--clients` (default 5; set 50) |
| Participation | all clients every round (partial participation only as a selfish ablation) | all clients every round |
| Partition | each client gets 2 randomly selected classes | `--distribution non_iid` (default) |
| Local samples per client | not specified | `--size` default 200 (train); 100 test samples per client (hard-coded) |
| Rounds | 30 | `--rounds` default 30 |
| Local epochs | 5 | `--epochs` default 5 |
| Optimizer | SGD | SGD, momentum 0.9, weight decay 5e-4 (utils.py `distributed_training`) |
| Batch size | 32 (WISDM-W) / 256 (CIFAR-10) | `--batch-size` default 256, but `BATCH_SIZE = min(batch_size, size)` -> 200 for CIFAR-10 with size 200 (one full-batch step per epoch) |
| LR | 0.01 (WISDM-W) / 0.1 (CIFAR-10) | `--lr` (no default; must be passed) |
| Seed | not specified | `--seed` default 1 (python, numpy, torch, cudnn deterministic) |
| Hardware | PyTorch 1.12.1, Windows 11, RTX A5000 | - |

Important code details:
- **The SGD optimizer is re-created for every mini-batch** (inside the batch loop), so momentum 0.9 never accumulates; it is effectively plain SGD + weight decay 5e-4. Reproduce this, not "SGD momentum 0.9".
- Only `model.parameters()` are aggregated (flattened with `get_ravel_weights`); no buffers (the CIFAR CNN has none).
- Aggregation is unweighted mean over all 50 clients (all have equal local size).
- The loss reported to (D)q-FFL is the **sum** of batch losses over all local epochs (`lossess[i] = running_loss`), not a mean.

**2-class assignment (create_datasets.py `non_iid_split`)**: `splits = max(10, 2*50) = 100`; `digits = randperm(100, generator seed 0) % 10`; client i gets consecutive pair `digits[2i:2i+2]`. Computed with torch 2.14 CPU:
```
[[4,9],[3,0],[1,9],[7,5],[3,1],[1,2],[0,5],[4,6],[6,3],[2,0],[6,2],[7,0],[6,6],[9,1],[7,8],[1,4],[5,0],[5,7],[8,3],[0,9],
 [8,1],[4,5],[4,2],[9,5],[4,6],[3,8],[2,0],[6,0],[9,8],[5,2],[9,7],[5,1],[2,4],[1,0],[2,2],[7,0],[5,7],[6,6],[9,3],[2,4],
 [8,7],[5,7],[3,8],[9,1],[4,8],[8,7],[4,3],[3,8],[6,3],[9,1]]
```
- Clients 12, 34, 37 have a single class (pair of identical digits).
- Data: one shuffled DataLoader batch of `50 * size` samples (10,000 for size 200) is drawn from the train set; client i takes the **first `size` samples** in that pool whose label is in its pair. Clients with the same pair (e.g. {1,9} x4, {5,7} x4, {3,8} x4) therefore get **identical** data, and clients sharing one class get overlapping samples. Per-client class balance is whatever falls out of the pool (roughly 50/50).
- Per-client test set: same procedure on the test set with 100 samples per client (same class pairs).
- Selfish clients are **client indices 0..S-1** (so client 0 = classes {4,9}, etc.).
- For WISDM-W the same code gives `randperm(100) % 12` pairs.
- Dirichlet option (`--distribution dirichlet_<alpha>`, Fig. 10 with alpha in {1, 0.1, 0.01}): per-client class proportions from Dir(alpha), train and test both carved from the train set (80/20 per client, capped at size / 100).

## 3. Models (Network.py `CNN(DATASET)`)

**CIFAR-10** (307,842 params; no normalization layers; Xavier-uniform init with ReLU gain on conv1, conv2, fc2, fc3):
```
conv1 = Conv2d(3, 64, 5) -> ReLU -> MaxPool2d(2,2)
conv2 = Conv2d(64, 64, 5) -> ReLU -> MaxPool2d(2,2)
flatten 64*5*5 -> Dropout(0.5) -> Linear(1600, 120) -> ReLU -> Linear(120, 64) -> ReLU -> Linear(64, 10)
```
Paper (p. 326) says "a CNN with 3 convolutional layers and 2 fully connected layers"; the code has 2 conv + 3 FC. Use the code.

**WISDM-W**: `ResLSTMClassifier(hidden_size=128, num_classes=12, num_layers=4, dropout=0.2)`, 622,300 params (paper: "ResLSTM ... 4 LSTM cells with residual connections"). **As published it does not run**: `ResLSTM` is built with `batch_first=False`, and `forward` then references `transposed_input_tensor` before assignment (verified: UnboundLocalError on a (2, 200, 6) input). The `else` branch it would reach only uses `final_lstm = nn.LSTM(6, 6, batch_first=True)`, followed by the classifier Flatten -> Dropout -> Linear(200*6, 128) -> ReLU -> Dropout -> Linear(128, 12). The architecture actually used for the paper's WISDM-W numbers cannot be determined from the repo.

## 4. Selfish client strategy

Selfish clients train honestly, then overwrite their model with `global + crafted_delta` (selfishness.py). They act only from round index `e >= 2` (need two past rounds), after any q-FFL rescaling of their update.

**Paper (Sec. III, p. 320-321, Eq. 4-5)**: estimate Gamma and the other clients' mean update from the two previous rounds (arg min over Gamma and mean-other of the reconstruction error of the past global updates), then send
`delta_hat_s = Gamma*phi*delta_s - (Gamma*phi - ... )*mean_other + (1-phi)*...` (Eq. 5: phi = 0 means send the others' mean, phi = 1 = full model replacement, phi = 1/k = honest under FedAvg).

**Code, two variants** (experiments.py switch `--estimate-k`):
1. `selfish_training` (default, no flag): known Gamma = `clients` (= 50). Estimate of others: `other = global_delta_(t-1) * clients - own_previous_delta` (sum of the other clients' updates last round, assumed constant). Sent: `delta = (legit * clients - other) * phi + (1 - phi) * other / (clients - 1)`.
2. `selfish_training2` (`--estimate-k`, **budget = 50 hard-coded**): fits a small module (`K_estimation`: learnable vector = mean other update, learnable scalar k, Gamma = exp(100k), k init 0.02) with 50 SGD steps (lr 0.05) on RMSE between predicted and actual global deltas of the two previous rounds (`x` = own deltas at t-1, t-2; `y` = global deltas). Then `clients = exp(100k)^2`, `other = weights * (clients - 1)`, and sent `delta = legit * (||other|| / ||global_delta||) * phi - phi * other + (1 - phi) * other * ||global_delta|| / ||other||`.
- The paper's Gamma-estimation experiment (Fig. 6, p. 326: estimation for both FedAvg and Dq-FFL) implies the paper used the estimating variant (`--estimate-k`). Not stated explicitly (looked in: Sec. III, VI-A).
- Selfish clients in the code do **not** inflate their reported loss for Dq-FFL (the paper, p. 323, says they upscale it); losses are computed before the selfish step.

**Selfishness and counts**: phi swept 0 to 1 in 0.1 steps (Figs. 8, 9, 12, 13); phi = 0.7 for Table III, Fig. 11, 14-16; Fig. 17 uses phi in {0.3, 0.5, 0.7}. Number of selfish clients: 1 or 15 (Figs. 8-13), 0 / 10% / 20% / 30% = 0 / 5 / 10 / 15 of 50 (Table III), 1 (Fig. 14, 17).

**Non-persistent selfish settings** (Sec. VI-E, p. 329-330):
- `--persistence p`: each selfish client acts in a round with probability p (it always acts in the final round, since the skip condition excludes `e == rounds-1`). Fig. 15: p from 0 to 1, 15 selfish (implied "same setup"), phi = 0.7. Fig. 17: one selfish client.
- `--partial-participation {1,2,3,4}`: act only at beginning (e <= R/3), middle (R/3 < e < 2R/3), end (e >= 2R/3), never. Fig. 16 (paper: Start = rounds 1-10, Middle 11-20, End 21-30; 50 clients, phi = 0.7; number of selfish clients not stated).

## 5. Aggregation variants (experiments.py `--aggregation`, `--fairness`, `--dq`)

| Name in paper | Code flags | Notes |
|---|---|---|
| FedAvg | `--aggregation fedavg` | plain mean of models |
| q-FFL | `--aggregation fedavg --fairness q` | q-FFL update: delta_i = L_i^q (w_i - w)/lr, h_i = q L_i^(q-1) ||grad||^2 + L_i^q / lr, new w = w + k * mean(delta_i) / sum(h_i) |
| Dq-FFL | `--aggregation fedavg --fairness q --dq` | q_i = q * (L_i / median(L)) using previous-round losses (Alg. 3, p. 324) |
| Median | `--aggregation median` | coordinate-wise median |
| FedMut | `--aggregation fedmut --fedmut-radius r` | per-element random sign in {-1,+1} times (w_t - w_(t-1)) times radius, all clients; **differs from official FedMut** (which uses balanced per-layer +1/(-1+beta) pairs). Default radius 0 (= FedAvg); value used in paper not specified |
| FedCDA | `--aggregation fedcda` | K = 3, B = 3, L = 1e-3 hard-coded; all 50 clients participate; final model = mean of all clients' 3 cached models **plus** the selected ones; **differs from the FedCDA paper** (L = 1, no warmup, mean of selected only) |
| RFL-Self | `--aggregation rotation2 --tau 2.5` | Alg. 1-2 (p. 322-323): flag i if (||d_i|| - median norm) / (1.4826 * MAD) > tau; replace flagged d_i with beta*d_i + (1-beta)*coordinate-median update, beta found by 10-step bisection so the norm equals the median norm |
| q-FFL + RFL-Self | `--aggregation rotation2 --fairness q` | |
| FairRFL | `--aggregation rotation2 --fairness q --dq` | RFL-Self + Dq-FFL |
| (not in paper) | `downscale`, `rotation` (flags every above-median norm), `rejection` (undefined function), `whitebox` (undefined), `geomed` (needs `geom_median` package) | not used in paper tables |

- tau = 2.5 (p. 322, chosen from a sweep over [0, 6]; Fig. 11 sweeps tau in [0, 4]).
- **q value: not specified** (looked in: Sec. IV-D, VI, Table III caption, Figs. 9/12/13 captions, code defaults: `--fairness 0.0`). Must be chosen; q-FFL papers commonly use q in {0.1, 1, 5}.
- FedMut radius for Table III: not specified (looked in: Sec. VI-D, code default 0).

## 6. Metrics (experiments.py end of script; utils.py `model_accuracy`, `fairness_index`)

- Accuracy of the **final global model** (after round 30) on each client's own local test set (100 samples of its 2 classes). No averaging over rounds.
- ACC_n = mean over normal clients (indices S..49); ACC_s = mean over selfish clients (0..S-1); STD = standard deviation over **all** clients' accuracies (Table III footnote, p. 329); code also prints Jain fairness index.
- Printed line: `Final accuracy: <mean all>, accuracy split selfish: <ACC_n> <ACC_s>, fairness: <Jain>, std: <STD>`.
- Number of runs/seeds per point: not specified (looked in: Sec. VI).
- Detection metrics for rotation2 (Fig. 11): per-round flag accuracy, precision, recall, F1.

## 7. Target numbers

### Table III (p. 329): 50 clients, phi = 0.7, final global model, per-client local test sets

WISDM-W:

| Approach | 0%: ACC_n | ACC_s | STD | 10%: ACC_n | ACC_s | STD | 20%: ACC_n | ACC_s | STD | 30%: ACC_n | ACC_s | STD |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| FedAvg | 62.10 | - | 16.95 | 46.22 | 70.20 | 18.73 | 44.93 | 71.17 | 18.36 | 43.00 | 73.40 | 20.57 |
| q-FFL | 62.30 | - | 16.74 | 45.02 | 57.20 | 17.81 | 18.93 | 19.00 | 18.36 | 18.92 | 19.13 | 16.36 |
| Dq-FFL | 62.30 | - | 16.68 | 46.62 | 69.00 | 17.56 | 45.93 | 68.30 | 18.36 | 48.20 | 75.27 | 17.42 |
| Median | 53.52 | - | 17.08 | 43.07 | 55.60 | 17.10 | 37.33 | 51.80 | 16.98 | 37.74 | 43.67 | 16.92 |
| FedMut | 60.92 | - | 16.55 | 37.93 | 48.00 | 16.36 | 32.76 | 31.00 | 18.36 | 22.36 | 27.13 | 16.36 |
| FedCDA | 63.56 | - | 17.40 | 47.79 | 71.60 | 18.38 | 46.02 | 63.90 | 18.47 | 38.60 | 57.87 | 20.43 |
| RFL-Self | 62.10 | - | 16.95 | 61.13 | 63.20 | 17.06 | 58.92 | 59.90 | 17.38 | 58.14 | 61.80 | 17.40 |
| q-FFL+RFL-Self | 62.30 | - | 16.74 | 61.07 | 63.00 | 16.88 | 59.40 | 60.40 | 17.52 | 58.17 | 60.33 | 17.19 |
| FairRFL | 61.33 | - | 16.64 | 61.31 | 63.20 | 16.22 | 59.60 | 60.80 | 15.71 | 60.91 | 68.27 | 15.57 |

CIFAR-10:

| Approach | 0%: ACC_n | ACC_s | STD | 10%: ACC_n | ACC_s | STD | 20%: ACC_n | ACC_s | STD | 30%: ACC_n | ACC_s | STD |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| FedAvg | 61.14 | - | 6.89 | 57.47 | 34.60 | 15.85 | 42.85 | 60.90 | 19.34 | 43.60 | 49.97 | 18.99 |
| q-FFL | 60.44 | - | 6.70 | 51.49 | 74.60 | 17.83 | 56.58 | 63.40 | 9.55 | 59.23 | 62.93 | 8.54 |
| Dq-FFL | 60.36 | - | 6.61 | 53.93 | 73.40 | 16.19 | 55.92 | 63.80 | 10.91 | 54.37 | 57.73 | 11.33 |
| Median | 56.18 | - | 7.07 | 53.91 | 52.40 | 6.91 | 52.00 | 53.80 | 7.38 | 50.29 | 52.13 | 6.96 |
| FedMut | 57.20 | - | 7.41 | 55.49 | 61.00 | 8.18 | 51.15 | 57.10 | 7.57 | 39.83 | 39.60 | 15.39 |
| FedCDA | 58.28 | - | 10.10 | 41.69 | 56.60 | 21.45 | 40.98 | 51.40 | 18.49 | 37.49 | 41.93 | 17.19 |
| RFL-Self | 59.64 | - | 7.42 | 59.19 | 54.40 | 7.24 | 59.12 | 60.00 | 6.70 | 58.11 | 59.67 | 6.86 |
| q-FFL+RFL-Self | 59.22 | - | 7.05 | 59.71 | 54.00 | 6.81 | 59.02 | 59.50 | 6.68 | 57.97 | 59.33 | 7.75 |
| FairRFL | 59.64 | - | 6.73 | 59.72 | 53.20 | 6.67 | 59.58 | 60.30 | 6.62 | 59.20 | 60.40 | 6.52 |

(Transcribed from a 220-dpi render of p. 329; the table is an image, absent from the text extract.)

### Headline claims
- "A selfish client can increase the model accuracy on its data by up to 39%" (abstract, p. 316): exact source cell not identified. Consistent with Fig. 8(a) (CIFAR-10, FedAvg, one selfish client): selfish-client accuracy rises from about 42% at phi = 0 to about 85-87% at phi >= 0.6, while normal clients stay near 60% (values read from the plot, approximate).
- "More than quadruple the accuracy variance among clients" (abstract): e.g. CIFAR-10 FedAvg STD 6.89 (0% selfish) -> 15.85 (10%) -> 19.34 (20%), i.e. variance 47 -> 251 -> 374 (5.3x-7.9x) (Table III).
- Recovered performance: with 30% selfish clients, CIFAR-10 FairRFL ACC_n 59.20 / STD 6.52 vs FedAvg 43.60 / 18.99; WISDM-W FairRFL 60.91 / 15.57 vs FedAvg 43.00 / 20.57 (Table III). Worst unmitigated accuracy 18.93 (WISDM-W q-FFL, 20%); FedAvg STD up to 20.57 (p. 329).
- Sec. VI-D (p. 328-329): with no mitigation, STD up to doubles (Fig. 14, one selfish client, phi = 0.7; bar chart, no printed numbers).
- Fig. 6-7 (p. 326): Gamma estimate converges to k = 50 under FedAvg within about 10-15 rounds; cosine similarity between true and estimated global update about 0.8.
- Fig. 11 (p. 327): with tau = 2.5, recall 100% for 1 and 15 selfish clients; detection accuracy about 90% (read from plot).
- Fig. 12-13 (p. 328): with 15 selfish clients, FairRFL/RFL-Self keep normal-client accuracy about 60% (CIFAR-10) and about 60-65% (WISDM-W) across all phi; FedAvg collapses for phi > 0.4.

### Checks to run first
1. CIFAR-10, 50 clients, 0% selfish, FedAvg: ACC 61.14, STD 6.89 (validates data split, model, lr 0.1, batch, size).
2. CIFAR-10, 5 selfish (10%), phi = 0.7, FedAvg: ACC_n 57.47, ACC_s 34.60, STD 15.85; then RFL-Self (`rotation2`, tau 2.5): 59.19 / 54.40 / 7.24.
3. CIFAR-10, 15 selfish (30%): FedAvg 43.60 / 49.97 / 18.99 vs FairRFL 59.20 / 60.40 / 6.52 (needs a q guess).

## 8. Compute feasibility

- CIFAR-10 CNN (0.3M params), 50 clients x 200 samples x 5 epochs x 30 rounds = 1.5M sample-passes per run: **CPU-feasible** (minutes per run). All Table III CIFAR-10 cells and the phi sweeps (11 values x 2-4 methods) fit on a CPU laptop. `median` and `rotation2` compute a coordinate-wise median over 50 x 0.3M, cheap.
- WISDM-W ResLSTM: a Python loop over 200 time steps x 4 cells per batch; CPU-feasible but slow; a T4 helps. Blocked anyway until the model bug in section 3 is resolved; the windowed dataset must also be built from the UCI zip (about 300 MB of raw text).
- Nothing in this paper needs ResNet/VGG-scale GPU runs.

## 9. Not specified

- q for q-FFL / Dq-FFL / FairRFL (looked in: Sec. IV-D, VI, Table III, fig captions, code defaults).
- Per-client local dataset size (looked in: Sec. VI p. 326; code default `--size 200`).
- Which selfish variant (`selfish_training` vs `selfish_training2`) produced each figure/table (looked in: Sec. III, VI; code has no run scripts). Fig. 6 suggests the Gamma-estimating one.
- FedMut radius and FedCDA settings for Table III (looked in: Sec. VI-D; code defaults radius 0, FedCDA K = B = 3, L = 1e-3).
- Number of runs/seeds; whether results are single-seed (looked in: Sec. VI).
- WISDM-W model as actually run (published code crashes; section 3).
- Number of selfish clients in Fig. 16 (looked in: Sec. VI-E).
- Paper says 3 conv + 2 FC for CIFAR-10; code is 2 conv + 3 FC.
