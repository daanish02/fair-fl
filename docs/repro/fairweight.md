# FairWeight reproduction spec

Paper: Kasyap et al., "FairWeight: Private Fairness-aware Aggregation using Model Interpretation based Weightage in Federated Learning", IEEE Transactions on Services Computing, 2026 (DOI 10.1109/TSC.2026.3686323). This is the accepted author version, 13 pages.

Sources used:
- P = `docs/impl-papers/FairWeight - ...pdf`. Page numbers are the printed journal page numbers, which equal the PDF page numbers.
- Code = the official release `SCR/fairweight.zip` (the paper footnote on p.8 points to https://tinyurl.com/fairweight). It is extracted to `SCR/fw/` and contains: fairweight.py, load_data_utilities.py, utilities.py, constraint.py, README.md, requirements.txt, datasets/{adult,bank-full,default,law,kdd}.csv, and datasets/source.txt.
- **For reproduction, the code wins.** Every paper-vs-code conflict is listed in section 9.

SCR = `C:\Users\Danish\AppData\Local\Temp\claude\c--Users-Danish-...\e2083521-...\scratchpad\`.

Run command (README): `python fairweight.py --fairness_notion 'stat_parity' --num_clients 3 --dataset_name 'law' --epochs 15 --communication_rounds 50 --distribution_type 'random'`.
- **Note:** `--fairness_notion` is not defined in the argparse parser, so this exact command fails with "unrecognized arguments". Drop that flag.
- Actual CLI flags:
  - `--num_clients` {3,5,10,15}, default 3
  - `--dataset_name` {adult, default, kdd, bank, law}, default adult
  - `--epochs`, default 15
  - `--communication_rounds`, default 50
  - `--distribution_type` {random, attribute-based}, default random
- Dependencies used but not listed in requirements.txt: `fairlearn`, `psmpy`, `scikit-learn`. The pinned versions are torch 2.0.1 and pandas 1.5.3.
- Everything runs on CPU (`device = torch.device('cpu')`).

---

## 1. Datasets, labels, sensitive attributes, preprocessing, client distribution

Counts below were measured on the shipped CSVs.

### Common preprocessing (all "random" loaders: `load_adult`, `load_bank`, `load_default`, `load_law`, `load_kdd`)
- Categorical columns are **label-encoded** (`LabelEncoder`, integer codes). There is no one-hot encoding.
- Numerical columns are standardised with `StandardScaler`, **fitted on the whole dataset before splitting**.
- **The sensitive attribute is kept as an input feature** (it is not dropped from X).
- Missing values ("?") stay as their own label-encoded category. Only the attribute-based loaders call `dropna`.

### "random" distribution (`load_dataset`, k==0 branch)
1. `train_test_split(X, y, test_size=0.2, random_state=42)` gives the global test set (20%) and a pool (80%).
2. Clients are carved from the pool one at a time: `train_test_split(X_temp, y_temp, test_size=1/(n-i), random_state=42)` for i = 0..n-2, and the last client gets the remainder.

This gives **n IID, (almost) equal-size, stratification-free shards**. n = `--num_clients` (3 by default; the paper also uses 5, 10 and 15). The split is deterministic (random_state=42), except for KDD, where `load_kdd` calls an unseeded `shuffle` first.

### "attribute-based" distribution (Attr3C)
- **Always 3 clients**, whatever `--num_clients` says (the loaders hard-code client_1..3).
- Adult, Bank, Default and KDD split by **age**: client_1 age 0-29, client_2 age 30-39, client_3 age >= 40.
  - The data is shuffled with an **unseeded** `shuffle` before splitting.
  - Age is excluded from the global scaler and standardised **per client**.
  - Each client then holds out 10% (`train_test_split(test_size=0.1, random_state=42)`), and the three 10% parts are concatenated into the global test set. The test set's age is re-standardised on the test set itself.
- Law splits by **family income** (`fam_inc`): first a global `train_test_split(test_size=0.2, random_state=42)`, then masks on the **label-encoded** fam_inc:
  - client_1 = codes 1-2 (raw fam_inc 2-3)
  - client_2 = code 3 (raw 4)
  - client_3 = codes 4-5 (raw 5; code 5 does not exist)

  Raw fam_inc = 1 (308 rows) ends up in **no client** (encoded as 0).

### Per dataset

| dataset | file (rows) | label (1 = ?) | sensitive attr and encoding | notes |
|---|---|---|---|---|
| Adult | adult.csv, 32,561 rows (UCI train file only), 14 features | `income`: `>50K`=1 (7,841), `<=50K`=0 (24,720) | `sex`: Female=0 (10,771), Male=1 (21,790) | categorical: workclass, education, marital.status, occupation, relationship, race, sex, native.country. Numerical: age, fnlwgt, education.num, capital.gain, capital.loss, hours.per.week |
| Bank | bank-full.csv, **40,004 rows** (only married and single; "divorced" rows are absent from the shipped file), 16 features | `y`: yes=1 (4,667), no=0 (35,337) | `marital`: married=0 (27,214), single=1 (12,790) | categorical: job, marital, education, default, housing, loan, contact, month, **previous** (numeric, but label-encoded), poutcome. Numerical: age, balance, day, duration, campaign, pdays |
| Default | default.csv, 30,000 rows, 23 features | `y`: 1 = default (6,636), 0 (23,364) | `SEX`: female=0 (18,112), male=1 (11,888) | categorical: LIMIT_BAL (label-encoded), SEX, EDUCATION, MARRIAGE, PAY_0, PAY_2..PAY_6. Numerical: AGE, BILL_AMT1-6, PAY_AMT1-6 |
| Law | law.csv, 18,692 rows, 11 features | `y` is **flipped** (`replace({0:1,1:0})`), so after the flip y=1 is the raw-0 class (1,836) and y=0 is the raw-1 class (16,856). Raw 1 is presumably "pass", so y=1 = fail | `sex` also **flipped**: after the flip, 0 = 10,550 rows and 1 = 8,142 rows. The code comment says 0=female, 1=male, but raw coding is undocumented | categorical: decile1b, decile3, fulltime, fam_inc, sex, race, tier. Numerical: lsat, ugpa, zfygpa, zgpa |
| KDD | kdd.csv (census-income KDD), 299,285 rows, 40 features | `class`: 1 = >50K (18,568), 0 (280,717) | `sex`: Female=0 (155,775), Male=1 (143,510) | 29 categorical (label-encoded), 11 numerical (standardised). Unseeded shuffle in the loader |

In the paper's terminology (P p.5), s = sensitive value 1 and s' = sensitive value 0. The code always treats value 0 as the protected group (`mi = b_indices = (s==0)`; `saValue = 0` in the metrics). So the protected group is female for Adult, Default and KDD, married for Bank, and law sex=0.

The paper's class ratios (P p.7: Adult 1:3.0, Bank 1:7.87, Default 1:3.52, Law 1:3.50, KDD 1:15.11) match the files except Law (file 1:9.18) and Bank (file 1:7.57). The paper's Bank description ("60% married, 28% single") refers to the full 45,211-row file, not the shipped one.

## 2. Model

- `fairweight.py` contains two `create_model` definitions. The first (logistic regression, `Linear(d,1)+Sigmoid`) is **inside a `'''` string literal, i.e. commented out**. The active one is the second, the DNN:
  ```
  Linear(d,64) -> ReLU -> Linear(64,32) -> ReLU -> Linear(32,1) -> Sigmoid
  ```
- This matches P p.8 ("three fully connected layers with hidden dimensions of 64 and 32 ... final sigmoid").
- Parameter counts: Adult (d=14) 3,073; Bank (16) 3,201; Default (23) 3,649; Law (11) 2,881; KDD (40) 4,737.
- Default PyTorch init, **unseeded**.

## 3. Local training

| item | code value |
|---|---|
| optimizer | `optim.Adam(lr=0.001)`, **re-created every round for every client** (Adam state is reset) |
| batch | **full batch**. Each "epoch" is one forward/backward/step on the client's entire data |
| local epochs | `--epochs 15`, i.e. 15 full-batch Adam steps per round |
| rounds | `--communication_rounds 50` |
| participation | all clients, every round |
| init per round | the client model is loaded from the global model |

- Prediction loss: `torch.nn.BCEWithLogitsLoss(pos_weight=weights)` with `weights = torch.where(y==1, 10.0, 1.0)` (`cost_false_negatives=10.0`, `cost_false_positives=1.0`).
  - Because `pos_weight` only multiplies the y=1 term, this is equivalent to a **constant pos_weight = 10** for positives:
    `L = -mean[10*y*log sigma(z) + (1-y)*log(1-sigma(z))]`
  - **Quirk:** z here is the model output, which is **already sigmoided**. So the loss applies a second sigmoid (z lies in (0,1), so sigma(z) lies in (0.5, 0.731)). Reproduce this exactly for a code-faithful match.
  - Class weighting is not mentioned in the paper. `utilities.find_class_weights` (neg/pos ratio) exists but is unused.
- Total local loss = weighted BCE + fairness loss (section 4).

## 4. Fairness loss actually used

```python
dp_loss = DemographicParityLoss(alpha=100)
dp_loss = AverageTreatmentEffectLoss(alpha=100)   # overrides the line above
fairness_loss = dp_loss(X1, y_pred, s1, y1_potential)
```

The loss used is **AverageTreatmentEffectLoss(alpha=100)**, i.e. the FACE-constrained variant, matching P Table III. The paper says Table II was run with **DP-constrained** local optimisation (P p.8: "The local fairness-constrained optimization is based on DP"). To reproduce Table II, swap in `DemographicParityLoss(alpha=100)`. The "regularisation constraint is 100" (P p.8) is alpha.

Both losses (`constraint.ConstraintLoss.forward`) apply `torch.sigmoid(out)` again. Since out is the model's sigmoid output, the moments are computed on sigma(sigma(logit)). The loss is `alpha * ||relu(M mu - c)||^2` with c = 0 and p_norm = 2.

- **ATE loss** (in effect):
  - With m_a = mean of sigma(out) over rows with y_pot=1 and s=a, and m = mean over rows with y_pot=1, the M matrix yields relu(+-(m_a - m)) for a in {0,1}.
  - So `L_ATE = 100 * [(m_0 - m)^2 + (m_1 - m)^2]`.
  - This is an equal-opportunity-style moment constraint conditioned on the **potential outcome** y_pot, not the observed y.
- **DP loss** (if used): the constraint-matrix construction has a bug. Rows use `j = i % 2`, so only the group-0 column is ever set, and the rows are duplicated. The result is `L_DP = 100 * 2 * (E[sigma(out)|s=0] - E[sigma(out)])^2`. Group 1 is only constrained implicitly.

### Where y_pot (the potential outcome) comes from
`load_data_utilities.find_potential_outcomes(X_client, y_client, sensitive_feature)` is computed once at load time, per client and for the global test set, on the preprocessed features:
1. `PsmPy(G, treatment=sensitive_feature, indx='index', exclude=[])`, where G = X with an 'index' column from `reset_index`. Covariates are all other features.
2. `logistic_ps(balance=True)`: propensity scores P(s=1|X) from a liblinear LogisticRegression fitted on balanced folds (majority group chunked to minority size; minority scores averaged across folds).
3. `knn_matched(matcher='propensity_score', replacement=True)` is called, but its result is not used.
4. `NearestNeighbors(n_neighbors=2)`, fitted on the opposite group's `propensity_logit`. The nearest neighbour ([:,0]) is found for each row.
5. y_pot(row) = `observed_output[idx]`, where idx is the neighbour's position.

- **Quirk (checked against psmpy 0.3.16 source):** `idx` is a position *within the opposite-group subset*, but it indexes the full client label array. In addition, `psm.predicted_data` has a fresh RangeIndex in psmpy's internal order (minority group first). So the returned list follows that order, not the row order of X.
- Net effect: y_pot is **not aligned** with the rows it is paired with, and is not the matched neighbour's label. A faithful reproduction must replicate this exactly. A "fixed" version (the label of the true matched opposite-group neighbour, aligned to rows) is a different method; if you use it, say so.
- Test set: `ytest_potential` is computed the same way on X_test using the true y_test, and is used only for the ATE metric.

## 5. Shapley / weight-importance settings (fairweight.py lines 271-408)

These run per client, every round, after local training, on the client's **training data**.

- Groups (s = sensitive value, y = label; the code comments swap "pos" and "neg", so only the actual values are given here):
  - `t_map` = s=1, y=0, predicted correctly
  - `f_mip` = s=0, y=0, predicted 1
  - `t_mip` = s=0, y=0, predicted correctly
  - `f_min` = s=0, y=1, predicted 0

  Predictions come from `model1(X).round()`. These correspond to the paper's (i) tp_s, (ii) fn_s', (iii) tp_s', (iv) fp_s' (P p.5), with "pos" meaning the majority label y=0.
- **b set** (bias): `nb = min(|t_map|, |f_mip|)`, **capped at 1000**. b1 = first nb of t_map, b2 = first nb of f_mip, in index order with no random sampling.
- **c set** (misclassification): `nc = min(|t_mip|, |f_min|)`, capped at 1000. c1 = t_mip[:nc], c2 = f_min[:nc].
- `step = 40`: importance is computed on `run = floor(n/40)` consecutive mini-batches of 40 samples. The per-batch values are summed, then divided by `run`. If run = 0 the set is skipped.
- **KDD skips the b set** (`dataset_name != 'kdd'`); only the c set is used.
- `calculate_shapley_values_fa(model, x, y, repeats=100)`: for each of **repeat=100** draws:
  - Zero a random **25%** of all parameters (`random.sample`, unseeded). Weights and biases are flattened together, and "neuron" in the code means an individual parameter.
  - Rebuild the model, compute `BCELoss(output, y)` and backpropagate.
  - Accumulate `|grad * original_weight|` for every parameter.

  Values are **summed** over the 100 repeats, not averaged.
- Selection:
  - `b_diff = WI(b1) - WI(b2)` (signed, not absolute). Take the top **imp_weights_for_bias = 750** indices (`argpartition`).
  - Do the same for `c_diff = WI(c1) - WI(c2)`.
  - The client's biased set is the **union** (at most 1,500 indices).
  - 750 is fixed regardless of model size: 24% of Adult's 3,073 parameters per set. The paper says 10% (see section 9).
- The `if True or ...` guards mean the "bias exists" checks are disabled.

## 6. Aggregation and beta1 / beta2

The code implements the paper's Alg. 1 steps 4-8 in plaintext. SecAgg and HE are not implemented.

1. `freq_j` = number of clients whose biased set contains j. U = the union over clients.
2. For j in U and client i: `score_ij = 1` if j is not in client i's set, else `normalised_scores[freq_j] = (n - freq_j)/n`. The lookup table is hard-coded for n = 3, 5, 10 and 15; for n=3 it is {1: 0.66, 2: 0.33, 3: 0}.
3. `w_ij = (beta1 * score_ij) ** beta2`, then normalised across clients: `w_ij / sum_k w_kj`, with NaN (0/0) replaced by 0.
4. For j in U: global_j = sum_i w_ij * theta_ij. For all other parameters: global_j = (1/n) sum_i theta_ij, an **unweighted** mean (not by sample size).
5. If U is empty in a round, use plain unweighted FedAvg.

Consequences:
- A parameter flagged by all n clients gets 0 for everyone, so it is **set to 0** (pruned), as in P p.6.
- With beta2 = 100 the weighting is almost winner-take-all. For n=3: (1.5*1)^100 vs (1.5*0.66)^100, a ratio of about 1.5^100 to 0.99^100.

| hyperparameter | paper | code |
|---|---|---|
| beta1 | range [1, n/(n-1)); "initialised as 1" (P p.8); adapted each round by Alg. 3 (P p.12, App. B) with step lambda1 in (0, 1/(n-1)) | **fixed 1.5** (`beta1 = 1.5#.2`). For n=3, 1.5 is the excluded upper bound n/(n-1) |
| beta2 | >= 1; initialised as 1; adapted by Alg. 3 with step lambda2 > 1 | **fixed 100** (`beta2 = 100#00`) |
| adaptation | Alg. 3: if BA dropped or fairness loss rose, increase beta1 by lambda1 (while < n/(n-1)) and beta2 by lambda2; otherwise decrease both (keeping them >= 1) | present only as a commented-out block (`lambda1 = 0.1`, `lambda2 = 50`), so **not executed** |

Fig. 3 (P p.9) sweeps beta1 in [1, 1.5] and beta2 in {1, 20, 40, 60, 80, 100} on Law R3C. It shows accuracy and BA falling as beta2 rises, and DP lowest at mid values.

## 7. Metrics (evaluated every round on the global test set)

`global_model(X_test)` gives probabilities; `y_pred_cls = round(p)` (threshold 0.5).

- `utilities.all_metrics(y_test, p)`, from the confusion matrix at threshold 0.5:
  - sensitivity = TP/(TP+FN)
  - specificity = TN/(TN+FP)
  - **balanced accuracy = (sensitivity + specificity)/2**
  - also G-mean, FNR, FPR, precision, F1, **accuracy** = (TP+TN)/N, and AUC (`roc_auc_score` on probabilities)
- **SPD** = `fairlearn.metrics.demographic_parity_difference(y_test, y_pred_cls, sensitive_features=sex_list)` = |P(yhat=1|s=0) - P(yhat=1|s=1)|, unsigned. This is what the paper's tables call "DP".
- **EOD** = `fairlearn.metrics.equalized_odds_difference(...)` = max(|TPR gap|, |FPR gap|). It is printed but not reported in the paper's tables.
- **ATE** (the paper's "FACE") = `find_ate_2(y_pred_cls, ytest_potential, sex_list)`. The signature is `find_ate_2(y_potential, predictions, protected_attr)`, so **the arguments are swapped**: the prediction plays the role of truth.
  - It returns `P(y_pot=1 | yhat=1, s!=0) - P(y_pot=1 | yhat=1, s=0)`, a signed gap in precision against the (misaligned) potential outcome.
  - This differs from the paper's eq. (4) (P p.3): FACE = mean(|Y_pot^A1 - Y_pred^A1| - |Y_pot^A0 - Y_pred^A0|). It is signed, which explains the negative FACE values in Tables III, V and VI.
- These are metrics of the **aggregated global model on the pooled global test set**. The paper's Definition 4 and BA definition (P p.3) weight per-client local values by w_k instead.
- The code prints metrics every round and does not select or save a final value. The paper reports the **average of 10 runs** (P p.8), presumably at round 50 (10 rounds for Table IV).

## 8. Target numbers

### Table II (P p.9): baselines, R3C (random, 3 clients), 50 rounds, DP-constrained local optimisation

| Dataset | FedAvg Acc | FedAvg BA | FedAvg DP | Agnostic-Fair Acc | Agnostic-Fair BA | Agnostic-Fair DP | FCFL Acc | FCFL BA | FCFL DP |
|---|---|---|---|---|---|---|---|---|---|
| Adult | 0.77 | 0.79 | 0.409 | 0.77 | 0.58 | 0.067 | 0.8 | 0.61 | 0.092 |
| Bank | 0.87 | 0.79 | 0.076 | 0.86 | 0.53 | 0.064 | 0.88 | 0.55 | 0.003 |
| Default | 0.78 | 0.69 | 0.053 | 0.78 | 0.52 | 0.006 | 0.8 | 0.57 | 0.089 |
| Law | 0.88 | 0.72 | 0.037 | 0.93 | 0.5 | 0.001 | 0.91 | 0.61 | 0.029 |
| KDD | 0.94 | 0.59 | 0.012 | 0.93 | 0.5 | 0.001 | 0.93 | 0.52 | 0.003 |

| Dataset | FedFB Acc | FedFB BA | FedFB DP | FairFed Acc | FairFed BA | FairFed DP | FairTrade Acc | FairTrade BA | FairTrade DP | FairWeight Acc | FairWeight BA | FairWeight DP |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Adult | 0.76 | 0.5 | 0.0005 | 0.76 | 0.5 | 0.0008 | 0.76 | 0.77 | 0.001 | 0.71 | 0.75 | 0.01 |
| Bank | 0.89 | 0.56 | 0.0006 | 0.88 | 0.56 | 0.03 | 0.88 | 0.78 | 0.019 | 0.85 | 0.83 | 0.011 |
| Default | 0.8 | 0.58 | 0.008 | 0.79 | 0.58 | 0.026 | 0.77 | 0.69 | 0.01 | 0.75 | 0.71 | 0.015 |
| Law | 0.91 | 0.57 | 0.004 | 0.91 | 0.59 | 0.021 | 0.88 | 0.68 | 0.008 | 0.83 | 0.78 | 0.005 |
| KDD | 0.93 | 0.5 | 0.001 | 0.93 | 0.5 | 0.001 | 0.87 | 0.82 | 0.002 | 0.93 | 0.8 | 0.001 |

(Cross-checked between the page image and the text extract. The text extract shifts DP cells down one row, so the image values are used.)

### Table III (P p.9): FACE-constrained optimisation, R3C. This is what the released code runs (ATE loss).

| Dataset | FairTrade BA | FairTrade DP | FairTrade FACE | FairWeight BA | FairWeight DP | FairWeight FACE |
|---|---|---|---|---|---|---|
| Adult | 0.7945 | 0.0913 | -0.0094 | 0.721 | 0.01 | -0.004 |
| Bank | 0.806 | 0.0314 | 0.009 | 0.814 | 0.015 | -0.019 |
| Default | 0.6981 | 0.0227 | 0.0117 | 0.7 | 0.017 | 0.008 |
| Law | 0.7467 | 0.0333 | 0.010 | 0.773 | 0.01 | 0.002 |
| KDD | 0.8196 | 0.0558 | 0.011 | 0.8 | 0.04 | -0.008 |

### Table IV (P p.9): Law dataset, data distribution and client count. **10 rounds only.**

| Split | FairTrade BA | FairTrade DP | FairTrade FACE | FairWeight BA | FairWeight DP | FairWeight FACE |
|---|---|---|---|---|---|---|
| Attr3C | 0.74 | 0.003 | 0.007 | 0.76 | 0.002 | 0.006 |
| R3C | 0.76 | 0.015 | 0.003 | 0.78 | 0.003 | -0.001 |
| R5C | 0.76 | 0.019 | 0.008 | 0.77 | 0.003 | 0.002 |
| R10C | 0.74 | 0.006 | 0.018 | 0.77 | 0.001 | 0.003 |
| R15C | 0.72 | 0.020 | 0.010 | 0.77 | 0.001 | 0.003 |

### Table V (P p.9): FairWeight, non-IID balance rate lambda (br)

| lambda | Bank BA | Bank DP | Bank FACE | Law BA | Law DP | Law FACE |
|---|---|---|---|---|---|---|
| 0.25 | 0.731 | 0.027 | -0.0012 | 0.742 | 0.0022 | 0.0013 |
| 0.50 | 0.745 | 0.019 | 0.0019 | 0.757 | 0.0021 | -0.0009 |
| 0.75 | 0.772 | 0.012 | 0.0005 | 0.762 | 0.0011 | -0.0019 |

The released code has **no** non-IID / balance-rate partitioner. How the br split was made is not specified (the paper cites [43]).

### Table VI (P p.9): FairWeight R3C, % of weights selected as responsible

| Weights % | Bank BA | Bank DP | Bank FACE | Law BA | Law DP | Law FACE |
|---|---|---|---|---|---|---|
| 5 | 0.797 | 0.030 | -0.013 | 0.78 | 0.0025 | 0.0042 |
| 10 | 0.771 | 0.012 | -0.006 | 0.768 | 0.0021 | -0.0029 |
| 15 | 0.774 | 0.014 | 0.0012 | 0.769 | 0.001 | -0.007 |
| 20 | 0.773 | 0.019 | -0.0069 | 0.774 | 0.003 | 0.003 |
| 25 | 0.779 | 0.027 | -0.0067 | 0.775 | 0.0029 | 0.0049 |

**Check first:**
1. FedAvg Adult R3C: Acc 0.77, BA 0.79, DP 0.409. This validates data, model and weighted BCE. Plain FedAvg means FairWeight's code with the fairness loss and FairWeight aggregation disabled.
2. FairWeight Law R3C: BA 0.78, DP 0.005 (Table II), or BA 0.773, DP 0.01, FACE 0.002 with the ATE loss (Table III).
3. FairWeight Bank R3C: BA 0.83, DP 0.011.

## 9. Paper vs code conflicts (the code wins for reproduction)

1. **Fairness loss:** Table II uses DP-constrained local optimisation; the code runs ATE/FACE (`AverageTreatmentEffectLoss` overrides `DemographicParityLoss`). Use DP for Table II and ATE for Table III.
2. **beta1 / beta2:** the paper initialises them at 1 and adapts them with Alg. 3; the code fixes beta1 = 1.5 and beta2 = 100 with no adaptation. beta1 = 1.5 also violates the paper's range beta1 < n/(n-1) = 1.5 for n=3.
3. **Number of responsible weights:** the paper says 10% of total weights (P p.8), and Table VI sweeps 5-25%. The code takes the top 750 per set with a union of up to 1,500, i.e. about 24-49% of Adult's 3,073 parameters.
4. **Shapley sampling:** the paper (Alg. 2, P p.12) prunes n-k weights per repeat and loops per sample. The code zeroes 25% (keeps 75%), uses batches of 40, sums 100 repeats, and averages over batches. It also uses the signed WI difference rather than an absolute "widest gap".
5. **KDD:** the code skips the bias (b) set; the paper does not mention this.
6. **Metrics:** the paper defines global DP and BA as sample-weighted sums of local values (Def. 4, P p.3); the code computes them on a pooled 20% global test set. The paper defines FACE by eq. (4); the code's ATE is `find_ate_2` with swapped arguments (a signed precision gap against a misaligned y_pot).
7. **Potential outcomes:** the paper describes causal matching; the code's PSM + kNN lookup indexes the wrong rows and returns them in psmpy's internal order (section 4).
8. **Loss:** not in the paper. The code uses BCEWithLogits on sigmoid outputs (a double sigmoid) with pos_weight 10, and the constraint losses apply a further sigmoid.
9. **Aggregation weights:** the paper's Alg. 1 and w_k use the sample proportion for the model and BA; the code uses an unweighted mean for non-flagged parameters.
10. **Law labels and ratio:** the code flips y and sex. The paper's ratio of 1:3.50 does not match the file's 1:9.18.
11. **Bank:** the paper describes 60% married / 28% single (full file); the shipped file has married and single only (40,004 rows).
12. **Attr3C:** the paper implies a configurable client count ("varying numbers"); the attribute-based loaders are hard-coded to 3 clients. The R5C/R10C/R15C rows of Table IV use the random loader.
13. **README:** the `--fairness_notion` flag does not exist.
14. **Reproducibility:** no seeds for torch, `random` or the KDD and attribute-based shuffles. The paper averages 10 runs, so expect run-to-run variance.

Not specified (looked in: P §V p.7-9, App. A-C p.12-13, Table captions; code; README):
- Which round's metrics are reported (last vs best).
- Baseline hyperparameters (the paper says "same experimental setup of FairTrade [12]"; no values given).
- The non-IID balance-rate partitioner used for Table V.
- The per-run seed values.
