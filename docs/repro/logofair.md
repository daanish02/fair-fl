# LoGoFair reproduction spec (Adult first)

Paper: Zhang et al., "LoGoFair: Post-Processing for Local and Global Fairness in Federated Learning", AAAI 2025.

Sources used:
- P = arXiv v1 (2503.17231), 18 pages, local copy `SCR/logofair_arxiv.pdf`. Page numbers below are PDF pages of this file.
- C = AAAI camera-ready, `docs/impl-papers/Logofair.pdf` (9 pages, no appendix). Table 1 is on C p.6, Table 2 on C p.7. The numbers match P exactly.
- Code = the official repo https://github.com/liizhang/LoGofair (README: "code for the paper '[AAAI 2025] LoGoFair ...'"). Commits from 2024-12-17 to 2025-03-21, plus a revert on 2026-03-15.
  - `SCR/lg_client.py` is byte-identical to `fedlearn/models/FedFairPostClient.py`, and `SCR/lg_du.py` is byte-identical to `fedlearn/utils/data_utils.py` (checked with `cmp`). **Both are official.**
  - In the code, LoGoFair is called "FedFairPost" / FFP. `config.py` maps `'logofair' -> FedFairPost`.
  - Other files were fetched to `SCR/lgrepo/`: options.py, main.py, FedFairPost.py (server), FedBase.py, models.py, sampling.py, and the shipped Adult alpha=0.5 split and pre-trained model.
- The code is authoritative where the paper is silent. Where code and paper disagree, both are listed in section 10.

SCR = `C:\Users\Danish\AppData\Local\Temp\claude\c--Users-Danish-...\e2083521-...\scratchpad\`.

---

## 1. Datasets and preprocessing

The paper uses Adult, ENEM and CelebA (P p.6 §5.1; P p.16 §D.1).

### Adult (P p.16 §D.1; code `adult_process`, `process_adult_csv`, `adult_get_sensitive_feature` in data_utils.py)
- Raw files: UCI `adult.data` + `adult.test` (README). **Both are concatenated** (`pd.concat([train.csv, test.csv])`), which gives **48,842 rows**. This was checked against the shipped split: 8877+5650+9685+17836+6794 = 48842.
- Rows with "?" are **not** dropped: `na_values=[]`, so "?" stays as its own category level.
- Label `salary`: `' >50K'` (train file) or `' >50K.'` (test file) maps to 1, everything else to 0.
- Sensitive attribute: `sex`. The map is Male=0, Female=1, so A=1 is female. `sex` is **removed from X** (`np.delete`). The paper says "gender as the sensitive attribute" (P p.16).
- Kept features: age, workclass, fnlwgt, education, education-num, marital-status, occupation, relationship, race, sex, capital-gain, capital-loss, hours-per-week, native-country.
- One-hot encoding (`pd.get_dummies`) of workclass, education, marital-status, occupation, relationship, race and native-country. The test file gets an extra all-zero column `native-country_ Holand-Netherlands` so its columns line up with the train file.
- Continuous columns (age, fnlwgt, education-num, capital-gain, capital-loss, hours-per-week) are min-max scaled to [0,1]. **Quirk:** scaling runs separately on adult.data and adult.test (min and max are taken per file), and the two files are only concatenated afterwards.
- Final input dimension: **106** features (measured on the shipped split_data.json).
- The code is adapted from FedFB's `DP_load_dataset.py` (link in the code comment).

### ENEM and CelebA (for reference only)
- ENEM: 2020 microdata. The label is the exam score quantised into 2 classes, following Alghamdi et al. 2022, and the sensitive attribute is race. The paper trains an MLP (P p.16).
- CelebA: the label is `Smiling`, the sensitive attribute is race (code: `Pale_Skin`), `sample_num=100000` images, and the model is ResNet18 (P p.16; code `get_data`).

## 2. Clients and Dirichlet partition

- **5 clients** (P p.6 §5.1 "Evaluation Protocols (3)"; P p.16; code default `--num_users 5`). Figure 3a also sweeps 2 to 50 clients at alpha=0.5.
- **The partition is on the sensitive attribute, not the label.** The paper says it "control[s] the heterogeneity of the sensitive attribute distribution at each client by determining the proportion of local sensitive group data based on a Dirichlet distribution Dir(alpha) as proposed in (Ezzeldin et al. 2023)" (P p.6, p.16). The code default is `data_setting = {'sensitive_attr':'sex','dirichlet':True,'by sensitive':True}`, so it calls `dirichlet(X, A, num_users, alpha)`.
- alpha is one of **{0.5, 5, 100}** (Table 1). The CLI flag is `--alpha`, with default 0.5.
- The partition runs on the **whole pooled dataset** (48,842 rows), before any train/test split.
- Exact algorithm (`fedlearn/utils/sampling.py::dirichlet`, adapted from FL-bench):
  - For each group k in {0,1}: shuffle that group's indices, then draw `p ~ Dirichlet(np.repeat(alpha*2, num_users))`. **Note: the concentration is 2*alpha, not alpha.**
  - Set `p_j = 0` for any client that already holds at least N/num_users samples, then renormalise.
  - Split the group's indices by `cumsum(p)`.
  - Repeat the whole draw until every client has at least `least_samples=20` samples of **each** group.
- Seeds: `np.random.seed(1+seed)` and `torch.manual_seed(12+seed)`, with default `seed=111`.
- The official alpha=0.5 Adult split is **shipped in the repo**: `data/adult/split_data/num_users=5 sensitive_attr=sex dirichlet=0.5 by_sensitive=True/split_data.json` (29 MB). Its per-client counts are:

| client | n | male (A=0) | female (A=1) | P(y=1) |
|---|---|---|---|---|
| 0 | 8877 | 7510 | 1367 | 0.275 |
| 1 | 5650 | 4519 | 1131 | 0.268 |
| 2 | 9685 | 5971 | 3714 | 0.223 |
| 3 | 17836 | 9752 | 8084 | 0.216 |
| 4 | 6794 | 4898 | 1896 | 0.252 |

  Load this file to reproduce the alpha=0.5 split exactly. There are no shipped splits for alpha=5 or alpha=100.

## 3. Train/test/validation split

- Paper: "partition each dataset into a 70% training set and the remaining 30% for test set, while post-processing models use half of training set as validation set" (P p.6 §5.1; P p.16 §D.1).
- Code (`read_data` in data_utils.py): the split is done **per client** with a random permutation.
  ```python
  train_index = indices[:int(n*0.7)]
  val_index   = indices[int(n*0):int(n*0.7)]   # = the SAME 70% as train
  test_index  = indices[int(n*0.7):]
  ```
  **Conflict:** in the released code the validation set is the whole training set, not half of it. For a paper-faithful run, use half of each client's 70% as validation. For a code-faithful run, use val = train. The more likely match to the published numbers is unknown. The split happens at load time and is not saved, so it is not fixed by the shipped JSON. It depends on the numpy RNG state at seed 112.
- Pre-training uses the train 70%. Calibration and post-processing use the validation part. Test metrics come from the 30% test part.

## 4. The model that produces eta (FedAvg pre-training)

- The paper says Adult uses "logistic regression as the classification model" (P p.16 §D.1). The code default is `--model logistic`: `nn.Linear(106, 1)` followed by a sigmoid. That is 107 parameters, which matches the shipped `pre_model.pth` of 1574 bytes, a flat CUDA tensor.
- The paper gives no Adult-specific hyperparameters. It only gives search ranges (Table 3, P p.17): learning rate {0.001, 0.005, 0.01}; global rounds {20, 30, 50, 80}; local rounds {5, 10, 20, 30}; local batch size {128, 256, 512}; hidden layer {16, 32, 64}; optimizer {Adam, SGD}.
- Values from the code (options.py defaults, plus the README example `python main.py --algorithm fedavg --data adult --num_round 50 --local_lr 0.001`):

| item | value | source |
|---|---|---|
| loss | `nn.BCELoss` on sigmoid output | `--criterion bceloss` |
| optimizer | Adam, weight_decay 1e-4 | `--local_optimizer adam`, `--wd 1e-4` |
| local lr | **0.005** by default; the README FedAvg example uses **0.001** | options.py / README |
| batch size | 512, shuffled | `--batch_size 512` |
| local work per round | **40 mini-batch steps**, not epochs (`num_local_round=40`; `local_train` loops `get_next_train_batch` 40 times) | options.py / client |
| global rounds | 25 by default; the README example uses **50** | options.py / README |
| client participation | all 5 (`clients_per_round=10`, capped at 5) | FedBase.select_clients |
| aggregation | **simple unweighted mean** of client parameters (`simple_average=True`) | FedFairPost.aggregate |
| optimizer state | Adam state persists per client across rounds; the client optimizer is built once | client __init__ |
| decision rule for FedAvg | predict 1 if score >= 0.5 (`sign(pred-0.5)`) | model_eval |

- The LoGoFair run reuses `pre_model.pth` if the split exists (`data_exist`). Otherwise it pre-trains with the same FedAvg loop for `num_round` rounds, using the settings of the logofair invocation (default 25 rounds, lr 0.005). **Which of these produced Table 1 is not specified.**

## 5. Calibration

- Paper: "we adopt model calibration (Guo et al. 2017) to group-wisely calibrate learned FL classifier" (P p.5 §4.3). No method name is given.
- Code (`FFPClient.calibration`): **Beta calibration** (`netcal.scaling.BetaCalibration`, default args). One calibrator is fitted **per client per sensitive group** (A=0 and A=1) on that client's validation scores and labels.
  - The fitted maps are applied to the validation scores used in post-processing.
  - At evaluation time, each client's own group calibrators are applied to its test scores (`model_eval(..., calibation=True)`).
- Enabled by default (`--calibration True`).

## 6. Post-processing hyperparameters

Paper objects: eq. (4)-(7) and Algorithm 1 (P p.5, P p.15). The smooth surrogate is `r_beta(x) = (1/beta) log(1+exp(beta x))` (P p.5). Alg. 1 uses projected gradient with learning rates {gamma_g, gamma_l}, T communication rounds and S local steps.

| symbol | paper | code default (options.py) |
|---|---|---|
| delta_l (local) | Table 1: **0.01** (P p.6 "we set delta^{l,c} = delta^g = 0.01"). Table 2 grid {0, 0.02, 0.04}. EO Table 4: 0.02 | `fairness_constraints={'global':0.02,'local':0.02}` |
| delta_g (global) | Table 1: **0.01**. Table 2 grid {0, 0.02, 0.04} | 0.02 |
| beta (for r_beta) | not specified | `--FFP_beta 1000` |
| gamma_l, gamma_g | not specified | `--post_lr 0.005`. mu uses an exponentiated-gradient step `mu *= exp(-0.005*grad)` then clamp >= 0. lambda uses a plain gradient step `lambda -= 5*0.005*grad` (effective 0.025) |
| T (post rounds) | not specified; Fig. 3b shows convergence within about 10 rounds | `--post_round 30`; the README example uses **20** |
| S (inner steps) | not specified | `post_local_round_mu=20` steps for mu, then `post_local_round_lamb=20` steps for lambda (sequential, not joint) |
| lambda init | 0 (Alg. 1) | server 0.01. The client constructor sets 0.1, but the server overwrites it each round |
| mu init | 0 (Alg. 1) | DP: `[0.5, 0.5]`. EO: `[0.1, 0.1]` each |
| server update | lambda <- lambda + sum_c Delta lambda_c (Alg. 1) | lambda <- **mean** over clients of the locally updated lambda_c |

Other code details:
- **DP uses a scalar lambda**, penalised as `delta_g*|lambda|/|C|`, instead of the paper's (lambda1, lambda2) >= 0 pair. mu is a length-2 vector (mu1, mu2) and enters only through mu1 - mu2.
- Statistics in `obj_H` (per client c, group a):
  - pi_{a,c} = N^val_{a,c} / N^val_total
  - p_a = (count of group a in test + val + train over all clients) / N^val_total. Note the mismatched denominator.
  - The term is weighted by 1/N^val_{a,c}.
- Final classifier (`local_post_eval`, DP): first re-optimise mu with the non-smooth `true_H` (relu) for `post_local_round_mu*5 = 100` steps. Then apply **deterministic group thresholds**:
  - `t0 = 0.5 + 0.5*lambda/p_0 + 0.5*(mu1-mu2)/pi_{0,c}`
  - `t1 = 0.5 - 0.5*lambda/p_1 - 0.5*(mu1-mu2)/pi_{1,c}`

  These are applied to calibrated scores: predict 1 if score >= t_a. There are no randomised decisions.
- In the EO branch, the final mu is set by a threshold search (`obj_local_mu_EO`) that nudges mu until the validation EO gaps are within 2*delta_l.
- Variants:
  - LoGoFair_l&g: both constraints.
  - LoGoFair_l: drop the global constraint (lambda = 0 / delta_g inactive).
  - LoGoFair_g: drop the local constraints (mu = 0).

  (P p.5 "Remark"; P p.6 Baselines.) The released code has no explicit flag for the variants. Reproduce them by fixing lambda=0 or mu=0. **Not specified further** (looked in: options.py, FedFairPost.py, FedFairPostClient.py, README).

## 7. Metrics

Definitions are in P p.11-12, App. A.2. A in {0,1} in code ({-1,1} in the paper).

- Acc: global test accuracy = sum over clients of correct predictions / sum of test samples (`post_test`).
- **M_DP^global**: `|P(Yhat=1|A=0) - P(Yhat=1|A=1)|` on the **union** of all clients' test data. Positives are pooled per group across clients, then divided by pooled group counts (`fair_cal`).
- **M_DP^local (Table 1)**: "maximal local fairness metric among clients" (P p.6 protocol (4)), i.e. `max_c |P(Yhat=1|A=0,C=c) - P(Yhat=1|A=1,C=c)|` on each client's test set.
  - **Code conflict:** `post_test` prints the max (`=== Test local fair DP: max(...)`), but `parato_result.txt` saves `avg_local`, the mean over clients. The paper states max, so use max.
- **M_EO^local**: `max_c max_{y in {0,1}} |P(Yhat=1|A=0,Y=y,C=c) - P(Yhat=1|A=1,Y=y,C=c)|`.
- **M_EO^global**: the same on pooled test data (`max(|TPR gap|, |FPR gap|)`).
  - Note: the paper's App. A.2 formula for phi_2 contains a typo ("2 - d eta"). Use the direct definition above.
- Significance: "All outcomes pass the significance test, p < 0.05". The number of runs or seeds is **not specified** (looked in: P §5, §D, Table 1-5 footnotes, README).
- Reference point: the shipped `parato_result.txt` for Adult alpha=0.5 DP with default deltas 0.02/0.02 ends at acc 0.8238, global DP 0.0082, avg-local DP 0.0128.

## 8. Target numbers

### Table 1 (DP, delta_l = delta_g = 0.01), Adult columns. P p.6, C p.6.

| alpha | Method | Acc | M_DP^local | M_DP^global |
|---|---|---|---|---|
| 0.5 | FedAvg | 0.8381 | 0.1922 | 0.1759 |
| 0.5 | FedFB | 0.8158 | 0.1174 | 0.0767 |
| 0.5 | FairFed | 0.8079 | 0.1416 | 0.0956 |
| 0.5 | FCFL | 0.8167 | 0.0832 | 0.1479 |
| 0.5 | LoGoFair_g | 0.8218 | 0.0889 | 0.0183 |
| 0.5 | LoGoFair_l | 0.8252 | 0.0367 | 0.0468 |
| 0.5 | LoGoFair_l&g | 0.8214 | 0.0489 | 0.0204 |
| 5 | FedAvg | 0.8418 | 0.1820 | 0.1725 |
| 5 | FedFB | 0.8243 | 0.1134 | 0.0658 |
| 5 | FairFed | 0.8157 | 0.1264 | 0.0971 |
| 5 | FCFL | 0.8134 | 0.0673 | 0.1297 |
| 5 | LoGoFair_g | 0.8264 | 0.0399 | 0.0104 |
| 5 | LoGoFair_l | 0.8237 | 0.0252 | 0.0215 |
| 5 | LoGoFair_l&g | 0.8244 | 0.0259 | 0.0138 |
| 100 | FedAvg | 0.8466 | 0.1802 | 0.1759 |
| 100 | FedFB | 0.8197 | 0.0901 | 0.0790 |
| 100 | FairFed | 0.8267 | 0.0977 | 0.1086 |
| 100 | FCFL | 0.8194 | 0.0613 | 0.0897 |
| 100 | LoGoFair_g | 0.8297 | 0.0368 | 0.0282 |
| 100 | LoGoFair_l | 0.8288 | 0.0335 | 0.0379 |
| 100 | LoGoFair_l&g | 0.8283 | 0.0362 | 0.0297 |

### Table 2 (DP sensitivity, alpha = 0.5, LoGoFair_l&g), Adult columns. P p.7, C p.7.

| (delta_l, delta_g) | Acc | M_DP^local | M_DP^global |
|---|---|---|---|
| (0.00, 0.00) | 0.8186 | 0.0439 | 0.0035 |
| (0.02, 0.00) | 0.8209 | 0.0488 | 0.0040 |
| (0.04, 0.00) | 0.8209 | 0.0543 | 0.0062 |
| (0.00, 0.02) | 0.8235 | 0.0454 | 0.0339 |
| (0.02, 0.02) | 0.8255 | 0.0507 | 0.0366 |
| (0.04, 0.02) | 0.8243 | 0.0634 | 0.0400 |
| (0.00, 0.04) | 0.8238 | 0.0423 | 0.0352 |
| (0.02, 0.04) | 0.8252 | 0.0548 | 0.0448 |
| (0.04, 0.04) | 0.8265 | 0.0667 | 0.0516 |

(Oddity: the realised test-set local DP is above delta_l in every row. For example, (0,0) gives local DP 0.0439. Table 2 at (0.02, 0.02) (global 0.0366) is also looser than Table 1 at (0.01, 0.01) (global 0.0204), which is consistent. Treat delta as a validation-set target, not a test-set guarantee.)

### Table 4 (EO, delta_l = delta_g = 0.02), Adult columns, arXiv only (P p.18).

| alpha | Method | Acc | M_EO^local | M_EO^global |
|---|---|---|---|---|
| 0.5 | FedAvg | 0.8381 | 0.2249 | 0.1410 |
| 0.5 | FedFB | 0.8242 | 0.1750 | 0.0977 |
| 0.5 | FairFed | 0.8125 | 0.1416 | 0.1048 |
| 0.5 | FCFL | 0.8291 | 0.0848 | 0.1195 |
| 0.5 | LoGoFair_g | 0.8350 | 0.0961 | 0.0293 |
| 0.5 | LoGoFair_l | 0.8302 | 0.0465 | 0.0665 |
| 0.5 | LoGoFair_l&g | 0.8334 | 0.0529 | 0.0346 |
| 5 | FedAvg | 0.8429 | 0.1899 | 0.1346 |
| 5 | FedFB | 0.8318 | 0.1249 | 0.0721 |
| 5 | FairFed | 0.8145 | 0.1172 | 0.0893 |
| 5 | FCFL | 0.8280 | 0.0824 | 0.1015 |
| 5 | LoGoFair_g | 0.8364 | 0.0615 | 0.0270 |
| 5 | LoGoFair_l | 0.8312 | 0.0413 | 0.0576 |
| 5 | LoGoFair_l&g | 0.8340 | 0.0487 | 0.0296 |
| 100 | FedAvg | 0.8466 | 0.1807 | 0.1285 |
| 100 | FedFB | 0.8357 | 0.0824 | 0.0590 |
| 100 | FairFed | 0.8203 | 0.0977 | 0.1086 |
| 100 | FCFL | 0.8294 | 0.0714 | 0.0832 |
| 100 | LoGoFair_g | 0.8378 | 0.0409 | 0.0241 |
| 100 | LoGoFair_l | 0.8352 | 0.0322 | 0.0347 |
| 100 | LoGoFair_l&g | 0.8335 | 0.0397 | 0.0262 |

Table 5 (EO sensitivity, Adult, alpha=0.5, P p.18), as Acc / local / global:
- (0,0): 0.8039 / 0.0206 / 0.0058
- (0.02,0): 0.8163 / 0.0447 / 0.0086
- (0.04,0): 0.8282 / 0.0560 / 0.0118
- (0,0.02): 0.8205 / 0.0239 / 0.0313
- (0.02,0.02): 0.8334 / 0.0529 / 0.0346
- (0.04,0.02): 0.8352 / 0.0634 / 0.0402
- (0,0.04): 0.8320 / 0.0297 / 0.0458
- (0.02,0.04): 0.8349 / 0.0362 / 0.0501
- (0.04,0.04): 0.8367 / 0.0687 / 0.0543

**Check first:**
1. FedAvg Adult alpha=0.5: Acc 0.8381, local DP 0.1922, global DP 0.1759. This validates data, partition and pre-training.
2. LoGoFair_l&g alpha=0.5: 0.8214 / 0.0489 / 0.0204.

## 9. Baseline settings (FedFB, FairFed, FCFL)

- Only the search ranges are given (Table 3, P p.17):
  - FedFB step size alpha {0.005, 0.01, 0.05, 0.3}.
  - FairFed fairness budget beta {0.01, 0.05, 0.5, 1} and local debiasing alpha {0.005, 0.01, 0.05}.
  - FCFL fairness constraint epsilon {0.01, 0.03, 0.05, 0.07}.
- "For all other hyperparameters, we follow the codes provided by authors and retain their default parameter settings" (P p.16).
- Code defaults for the baselines: `FB_alpha=0.3`, `fairfed_beta=0.5`.
- The public repo only contains FedAvg.py and FedFairPost.py in `fedlearn/algorithm`. **The baseline implementations are not released**, although config.py maps names for them.
- The chosen value per baseline, dataset and alpha is **not specified** (looked in: P §5, §D.1, Table 3; README; options.py).

## 10. Paper vs code conflicts and unspecified items

Conflicts (paper / code):
1. Validation set: half of train / **the entire 70% train** (`val_index = indices[0:0.7n]`).
2. Dirichlet concentration: Dir(alpha) / **Dir(2*alpha)**, plus a rejection loop requiring at least 20 samples per group per client.
3. Reported local metric: max over clients / max is printed, **mean** is saved to parato_result.txt.
4. delta for Table 1: 0.01 / default 0.02.
5. DP lambda: (lambda1, lambda2) >= 0 / a single unconstrained scalar with |lambda| penalty (DP branch has no clamp).
6. Server update: lambda + sum of Deltas / **mean** of client lambdas.
7. Initial lambda and mu: 0 / 0.01 (server) and [0.5, 0.5] (DP mu).
8. Validation statistics bug: `val_Y1_A_info` and `val_Y0_A_info` use `=` inside the client loop, so the "global" P(A=a,Y=y) in the EO objective comes from **only the last client's** validation data (data_utils.get_data_info). This affects EO only.
9. `p_a` in the DP objective uses (all-split group count)/N_val, which is not a probability.
10. DP mu optimisation in `local_fair_post` never zeroes `.grad`, so gradients accumulate across the 20 steps. `local_post_eval` does zero them.

Unspecified (looked in: P main text, App. C/D, Table 3 caption; C; README; options.py):
- The exact pre-training lr and rounds used for the tables (use lr 0.001, 50 rounds from README, or 0.005, 25 from defaults).
- beta, gamma_g, gamma_l, T, S in the paper (code values above).
- Number of repeated runs and seeds.
- How LoGoFair_l and LoGoFair_g are switched on in code.
- Baseline hyperparameters actually selected.
- Whether the test split is also scaled or calibrated identically across alpha settings. The code does it identically.
