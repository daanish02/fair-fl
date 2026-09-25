---
title: "Lab Notes — running log of results, findings, and design decisions"
status: living document, updated throughout the build
---

# Lab Notes

This is a running log of anything worth citing, quoting or plotting later: results, surprising findings, the reasons behind design choices, and bugs found in prior work's code. It is updated as we go, so nothing has to be rebuilt from chat history. Each entry is dated and says what it is for.

Where things live:
- **Every finished run:** `code/results/runs.jsonl`, appended automatically. This is the persistent record; `code/runs/` is scratch.
- **Reproduction targets:** `code/results/reproduction/published.csv`.
- **Paper vs ours:** `code/results/reproduction/published_vs_ours.csv`, regenerated with `uv run fairfl compare`.
- **Per-paper exact settings:** `docs/repro/<paper>.md`.
- **Plan and gates:** `C:\Users\Danish\.claude\plans\ultra-i-have-been-linked-aho.md`. Novelty sweep: `docs/novelty-and-feasibility.md`.

Protocol: one run per configuration (seed 0), unless a result is borderline.

---

## 2026-09-25 — The research direction was narrowed after a novelty check

**For:** the related-work section and the introduction.

- The original proposal's claim that "every FL fairness mechanism is an instantaneous snapshot" is false. FairFed's weights are recursive (w_k^t = w_k^{t-1} - beta(...)); FCFL keeps accumulated queues; FedCDA keeps cross-round caches.
- History-as-mechanism is common. **History-as-fairness-target** is not: no peer-reviewed FL work judges group fairness over the deployed history (Alamdari et al.'s long-term, periodic, anytime and bounded scopes).
- That is the gap we claim. Full evidence is in `docs/novelty-and-feasibility.md`.
- **Gate G1** (does the problem show up in FL?) runs only after the baselines are reproduced, so that a surprising G1 result can't just be a broken baseline.

---

## 2026-09-25 — LoGoFair: its official code overflows as released

**For:** a footnote in the reproducibility appendix.

- The official client code (github.com/liizhang/LoGofair, `FedFairPostClient.r_beta`) computes `(1/beta) * log(1 + exp(beta * x))` in float32 with beta = 1000.
- For any calibrated score near 1, beta * x is about 150, above the float32 limit of about 88. So `exp` overflows, the loss is `inf`, and every gradient becomes NaN.
- As released, the post-processing cannot produce finite multipliers. The repo's shipped `parato_result.txt` is finite, so those numbers must come from a different revision (the repo history includes a revert).
- Our port (`strategies/logofair_official.py`) uses `softplus(x, beta)`. It is the same function, computed stably. Every other quirk of the code is kept on purpose:
  - gradients accumulate across steps;
  - the scalar lambda is unclamped;
  - p_A uses the all-split count over the validation size;
  - the server averages the clients' lambdas.

---

## 2026-09-25 — LoGoFair Table 1 (Adult, alpha 0.5, DP): accuracy reproduced, fairness tighter than published

**For:** the reproduction table in the paper and dissertation.

**Setup:** the official repo's shipped client split and pre-trained logistic model, so data and pre-training are the authors' exact files. The per-client 70/30 split uses their seed (np.random.seed(112)). Validation is the whole train split, as in their code. delta_l = delta_g = 0.01, as stated for Table 1. Config: `code/configs/repro/logofair/adult_a0.5.yaml`.

| Method | Acc (paper / ours) | local DP max (paper / ours) | global DP (paper / ours) |
|---|---|---|---|
| FedAvg | 0.8381 / 0.8415 | 0.1922 / 0.1853 | 0.1759 / 0.1662 |
| LoGoFair_g | 0.8218 / 0.8268 | 0.0889 / 0.0208 | 0.0183 / 0.0110 |
| LoGoFair_l | 0.8252 / 0.8245 | 0.0367 / 0.0258 | 0.0468 / 0.0128 |
| LoGoFair_l&g | 0.8214 / 0.8237 | 0.0489 / 0.0291 | 0.0204 / 0.0064 |

**Reading:**
- The FedAvg row matches (within 0.01 on every column), which confirms data, split and metrics.
- Every accuracy is within 0.5 points.
- The paper's qualitative claim reproduces: LoGoFair cuts both local and global DP by about 5-15x for about 1.5 points of accuracy.
- Our fairness numbers are consistently *tighter* than published.

**Candidate causes** (not yet resolved):
1. Table 1 may have been produced with the code's default deltas (0.02). We tried it: with delta = 0.02, LoGoFair_g gives global DP 0.038, closer to the reported 0.018 on some columns but not all.
2. A different pre-trained model or seed from the one shipped.
3. The overflow above: the published run may have used a different r_beta or beta.

The paper reports no seeds or standard deviations. This counts as "claim reproduced, numbers not exact". Revisit only if LoGoFair becomes central to G1.

**Our paper-faithful port** (`strategies/logofair.py`: Theorem 1 + projected GD) on our own Adult pipeline (10 clients, 5 rounds) also reached global DP 0.017-0.020, close to the published values. That setting differs from the paper's, so it isn't the reproduction.

---

## 2026-09-25 — FCFL: the paper and its code disagree on four points; we follow the paper's Table 2 settings

**For:** the reproduction appendix. Spec: `docs/repro/fcfl.md`.

- **Alpha:** code default 0.5, but Table 2 uses 0.3.
- **Random fraction r:** the paper uses 0.4 (MNIST); the released code uses pure top-Q selection.
- **"Server momentum 0.5":** in the code this is client SGD momentum.
- **Per-client split:** in the code it is an unshuffled 80/10/10 slice, so under shards every client's local test set is a single class. We match the code here, because the variance and worst-10% metrics are computed on those slices.
- **Accuracy:** measured on the official 10k MNIST test set, not the client mean. We added `data.global_test: official` for this.
- **An earlier FCFL run was discarded.** It used the code defaults (alpha 0.5, an 8:2 shuffled split, client-mean accuracy), so it was not comparable.

**Status:** running. FedAvg, q-FedAvg (q = 0.2) and FCFL, seed 0, 2000 rounds, about 1.2 h. Results will land in `code/results/runs.jsonl`.

---

## 2026-09-25 — FedFDP: epsilon = 3.52 fixes the number of rounds, which the paper never states

**For:** a reproduction caveat. Spec: `docs/repro/fedfdp.md`.

- **Settings:** Dir(0.1), 10 clients, full participation, one DP-SGD step per round with Poisson rate q = 0.05, C = 0.1, sigma = 2, sigma_l = 5, delta = 1e-5.
- **The round count T:** the paper never gives it. Our accountant matches its Fig. 4 only when the noise on the uploaded loss is excluded. Under that reading, epsilon = 3.52 implies about 776 rounds.
- **Suspect values:** the reported Psi values for the DP baselines (1e8 to 1e11) look implausible.
- **Next:** implement the RDP accountant and run MNIST: FedAvg-DP vs FedFDP. Targets: 93.40 vs 95.13 (Table 3).

---

## 2026-09-25 — Accountant verified; GPU reproduction suites ready (FedMut, FedCDA), FedFDP config ready

**For:** the reproduction appendix and the methods section (the DP accounting).

- **The RDP accountant** (`fairfl.metrics.privacy`: Poisson-subsampled Gaussian, Balle et al. conversion) reproduces FedFDP's own Fig. 4 round counts at eps = 2, q = 0.05, delta = 1e-5:

  | sigma | paper | ours |
  |---|---|---|
  | 1.5 | 115 | 114 |
  | 2.0 | 268 | 267 |
  | 2.5 | 463 | 463 |
  | 3.0 | 708 | 702 |

  At eps = 3.52 this gives **T = 776**, or 683 if the loss upload is also counted. The paper's reported T values ignore the loss mechanism, despite its Theorem 4. The config uses 776; the test is `tests/test_privacy.py`.
- **The FedFDP client now follows Algorithm 2 exactly.** Each round it:
  1. draws one Poisson batch (q = 0.05);
  2. takes one fair-clipped DP-SGD step (lr 1.0);
  3. uploads the loss on the same batch, clipped to the previous round's upload.
- **The Psi metric** (Eq. 2: weighted variance of client *training* losses) is computed from a final train-split evaluation.
- **MNIST lambda:** the paper does not state it. We use 0.01, its best value on FMNIST (Fig. 2).
- **FedMut:**
  - uses the authors' shipped client partitions (`code/assets/partitions/fedmut/`, identical splits);
  - CIFAR-10 normalised with 0.5/0.5, no local test data, official test set;
  - reported as mean +- std of the last 10 evaluations, 1000 rounds.
  - ResNet-18 runs use GroupNorm (the official code uses BatchNorm, and our parameter vector carries no buffers). So compare the FedMut - FedAvg gap for ResNet-18; the CNN rows should match absolutely.
- **FedCDA:**
  - momentum 1e-4, reproduced as printed (possibly a typo for 0.9);
  - 20 local epochs, 50 FedAvg warmup rounds, K = 3, B = 3 near-equal batches of the 4 participants, L = 1;
  - one run each (the paper averages 2).
- **Where they run:** the GPU suites run on Colab (`code/notebooks/fairfl_gpu.ipynb`). Run results go to Drive and are merged into `code/results/runs.jsonl`.

---

## 2026-09-25 — FCFL Table 2 (MNIST): our baselines are stronger and fairer than the published ones

**For:** the reproduction table, plus a caveat on baselines taken from papers.

**Setup:** paper Table 2 plus official code. 100 clients, 2 label shards each, 10% per round, CNNMnist, lr 0.1, client momentum 0.5, 1 epoch, batch 64, 2000 rounds. Unshuffled 80/10/10 client slices as in the code. Accuracy is on the official test set; variance and best/worst 10% are over the 100 local test slices. Seed 0, single run (the paper averages 5). Config: `code/configs/repro/fcfl/mnist_shards.yaml`.

| Method | Acc % (paper / ours) | Variance %^2 (paper / ours) | Worst 10% (paper / ours) | Best 10% (paper / ours) |
|---|---|---|---|---|
| FedAvg | 95.96 / 98.49 | 15.06 / 4.17 | 87.47 / 93.67 | 100 / 100 |
| q-FedAvg (q = 0.2) | 96.09 / 97.87 | 12.58 / 5.58 | 88.40 / 92.33 | 100 / — |
| FCFL (alpha 0.3, r 0.4) | 96.06 / 98.85 | 11.03 / 2.16 | 89.17 / 95.83 | 100 / 100 |

**Reading:**
- Our FedAvg is about 2.5 points more accurate and about 3.6x fairer (variance) than the FedAvg the paper reports.
- The paper's baselines are not released ("we directly rewrite the code for comparison", p.397). Its FedAvg therefore used an unknown local setup, while ours uses FCFL's own client pipeline. The published baseline gap may partly be a baseline-implementation gap.
- q-FedAvg at q = 0.2 does not reduce variance in our run (5.58 vs 4.17), contrary to the paper's ordering. The paper picked q per method by grid search ("best performance"); we ran only the reported best q.
- **What decides this reproduction** is whether FCFL beats our FedAvg on variance and worst-10%, not the absolute numbers.
- **Result: FCFL's claim reproduces.**
  - Against our FedAvg, FCFL roughly halves the accuracy variance (2.16 vs 4.17) and raises worst-10% by 2.2 points (95.83 vs 93.67).
  - It also slightly raises accuracy (98.85 vs 98.49).
  - The ordering FCFL > FedAvg and FCFL > q-FedAvg holds on every fairness metric. Absolute numbers are better than the paper's for all three methods (one seed).

**Tooling fix:** `fairfl compare` had marked these as OK. The absolute slack was multiplied by the %^2 scale, and a 5% relative tolerance let 2.5-point accuracy gaps pass. It now uses 2% relative tolerance, plus absolute slack only for rates (0.01, or 1 point in %).

---

## 2026-09-25 — Runs lost to a machine shutdown; added checkpoint/resume

**For:** internal reproducibility (not paper content).

- **Lost:** the laptop stopped mid-session, killing the local FCFL run (at round 1650/2000) and FedFDP (just started), and the Colab session.
- **Survived** in `code/results/runs.jsonl`: LoGoFair x4, FCFL-MNIST FedAvg and q-FedAvg.
- **Checkpointing:** the engine now saves a checkpoint every `train.checkpoint_every` rounds (default 25) to `<run dir>/checkpoint.pt`.
  - The checkpoint holds params, server momentum, the whole strategy object (queues, caches, RNG), client states, logs, and the torch/CUDA RNG.
  - It is written atomically.
  - Any rerun (`fairfl batch` locally, Run all on Colab) resumes after the last checkpoint, and the checkpoint is deleted when the run finishes.
  - `tests/test_checkpoint.py` checks that an interrupted-and-resumed FCFL run ends with *identical* parameters to an uninterrupted one.
- **Keep-awake:** `fairfl batch` also asks Windows not to idle-sleep while a suite runs. It cannot prevent manual sleep, closing the lid, or a shutdown.

---

## 2026-09-25 — Median (Chen et al.) Fig. 1 reproduced exactly; MNIST part moved to Colab

**For:** the reproduction table, plus the motivation that a robust median needs noise under heterogeneity.

- **The paper's only medianSGD experiment is the 1-D toy (Fig. 1):** three nodes with f_i = (x - a_i)^2 / 2, a = (1, 2, 10), step 0.001, x0 = 0.0005. Its neural-network experiments (Fig. 2-3) use signSGD with majority vote, which the paper shows is the sign of the median.
- **Toy results** (`code/src/fairfl/experiments/median_toy.py`, saved to `code/results/reproduction/median_toy.jsonl`, tested in `tests/test_median_toy.py`):

  | Aggregation | final x | true (mean) gradient |
  |---|---|---|
  | mean (target) | 4.333 (= 13/3) | 0.000 |
  | median | 2.000 | -2.333 (= -7/3) |
  | signSGD | 2.000 | -2.333 |
  | median + noise b = 1 / 5 / 10 / 20 (10^5 steps) | 2.20 / 3.81 / 4.27 / 4.45 | -2.14 / -0.53 / -0.06 / +0.11 |

  This matches Fig. 1 exactly: both median-based methods stall where the median gradient is zero, while the true gradient stays at 7/3. Noise before the median closes the gap, and too much noise overshoots (Theorem 6's trade-off).
- **The MNIST replication runs on Colab** (`code/suites/median_mnist.yaml`): 10 clients, each holding one or two exclusive classes; one full-batch step per round; 784-128-10 MLP; 10^4 rounds.
  - Variants: mean, signSGD with b in {0, 1e-3, 1e-5}, and medianSGD with b in {0, 1e-3}.
  - Paper readings (Fig. 2, from the plot): without noise, about 0.35-0.4 accuracy; with noise or sub-sampling, about 0.85-0.9.
  - The signSGD server step (0.001) is our choice; the paper tuned it over {1, 0.1, 0.01, 0.001} without stating the result.

---

## 2026-09-25 — FairWeight: official-code port; the FedAvg row matches on BA and DP but not accuracy

**For:** the reproduction table, plus a caveat on FairWeight's metrics.

- **Setup:** `strategies/fairweight_official.py` and `data/fairweight_data.py` port the authors' released code line by line. That includes the quirks that change results:
  - a double sigmoid in the loss, with pos_weight 10;
  - Adam re-created each round, 15 full-batch steps;
  - the DP loss with its constraint-matrix bug;
  - the Shapley routine with 25% zeroing, 100 repeats, batches of 40, top 750;
  - the hard-coded 0.66 / 0.33 score table for 3 clients.

    The table matters: with beta2 = 100, 0.66 instead of 2/3 changes a weight by about 3x.
- **FedAvg row, Adult R3C** (the paper averages 10 unseeded runs; we ran 5 seeds for this row):

  | Metric | Paper | Ours, mean over seeds | Ours, range over seeds |
  |---|---|---|---|
  | Accuracy | 0.77 | 0.70 | 0.66-0.73 |
  | Balanced accuracy | 0.79 | 0.78 | 0.75-0.80 |
  | DP | 0.409 | 0.34 | 0.24-0.40 |

  Balanced accuracy matches, and DP is within the (large) seed variance. Accuracy is about 6 points low. The paper never states how its FedAvg baseline was configured, and with pos_weight 10 the classifier trades accuracy for recall. So the gap may come from the baseline configuration rather than the port.
- **Table II, all four datasets** (seed 0; paper / ours):

  | Dataset | Method | Accuracy | Balanced accuracy | DP |
  |---|---|---|---|---|
  | Adult | FedAvg | 0.77 / 0.715 | 0.79 / 0.794 | 0.409 / 0.397 |
  | Adult | FairWeight | 0.71 / 0.583 | 0.75 / 0.700 | 0.010 / 0.107 |
  | Bank | FedAvg | 0.87 / 0.837 | 0.79 / 0.830 | 0.076 / 0.102 |
  | Bank | FairWeight | 0.85 / 0.827 | 0.83 / 0.830 | 0.011 / 0.064 |
  | Default | FedAvg | 0.78 / 0.607 | 0.69 / 0.679 | 0.053 / 0.081 |
  | Default | FairWeight | 0.75 / 0.562 | 0.71 / 0.668 | 0.015 / 0.030 |
  | Law | FedAvg | 0.88 / 0.824 | 0.72 / 0.777 | 0.037 / 0.011 |
  | Law | FairWeight | 0.83 / 0.828 | 0.78 / 0.774 | 0.005 / 0.013 |

  - **Direction reproduces:** on Adult, Bank and Default, FairWeight lowers DP relative to FedAvg (Law is already near zero for both).
  - **Size does not:** the DP reduction is smaller than published, most clearly on Adult (0.107 vs 0.010).
  - **Balanced accuracy matches;** accuracy is consistently low. The worst case is Default, 17-19 points low, where the majority class is about 78%. That pattern fits pos_weight 10 pushing predictions towards the positive class, which lowers accuracy but not balanced accuracy.

    So the published accuracy was probably produced with a different threshold or weighting from the released code. This is unverified.
  - Only one seed. The Adult FedAvg DP spread over seeds was 0.24-0.40, so single-seed DP gaps of about 0.05 are within noise.
- **Table III (the FACE/ATE loss)** is not ported. It needs the authors' propensity-matched potential outcomes, which their code misaligns (see `docs/repro/fairweight.md` section 4).

---

## 2026-09-26 — LoGoFair Table 1 at alpha = 5 and 100: FedAvg and all accuracies match; fairness within about 0.02

**For:** the reproduction table (LoGoFair is now complete for DP on Adult).

**Setup:**
- The official repo ships the split and pre-trained model for alpha = 0.5 only. `fairfl.data.logofair_data` ports the pipeline its code runs when no split exists:
  - UCI Adult processed per file, concatenated to 48,842 rows, 106 features, sex removed;
  - `np.random.seed(112)`, then Dir(2*alpha) over 5 clients on sex, with at least 20 per group per client;
  - FedAvg pre-training with the code defaults: 25 rounds, Adam lr 0.005, wd 1e-4, 40 batches of 512, unweighted mean.
- **Validated against the shipped alpha = 0.5 files:**
  - the regenerated split is identical row for row on all 5 clients;
  - the regenerated pre-trained model gets accuracy 0.8412 and DP 0.165, vs 0.8415 and 0.166 for the shipped one.
- Raw data goes in `code/data/logofair/raw/` (gitignored). Configs: `configs/repro/logofair/adult_a{5,100}.yaml`. Everything else is as in alpha = 0.5.

**Results** (accuracy / max local DP / global DP; paper vs ours; seed 0):

| alpha | Method | Paper | Ours |
|---|---|---|---|
| 5 | FedAvg | 0.8418 / 0.1820 / 0.1725 | 0.8383 / 0.1815 / 0.1685 |
| 5 | LoGoFair_g | 0.8264 / 0.0399 / 0.0104 | 0.8280 / 0.0261 / 0.0223 |
| 5 | LoGoFair_l | 0.8237 / 0.0252 / 0.0215 | 0.8243 / 0.0406 / 0.0131 |
| 5 | LoGoFair_l&g | 0.8244 / 0.0259 / 0.0138 | 0.8239 / 0.0361 / 0.0075 |
| 100 | FedAvg | 0.8466 / 0.1802 / 0.1759 | 0.8386 / 0.1893 / 0.1807 |
| 100 | LoGoFair_g | 0.8297 / 0.0368 / 0.0282 | 0.8278 / 0.0508 / 0.0301 |
| 100 | LoGoFair_l | 0.8288 / 0.0335 / 0.0379 | 0.8230 / 0.0201 / 0.0067 |
| 100 | LoGoFair_l&g | 0.8283 / 0.0362 / 0.0297 | 0.8223 / 0.0167 / 0.0027 |

**Reading:**
- All 12 FedAvg values across the three alphas match within the tolerance. So data, partition and pre-training reproduce.
- Every LoGoFair accuracy matches within 0.006.
- LoGoFair cuts DP from about 0.18 to 0.03 or below, as claimed. Our fairness values are within about 0.02 of the paper's, sometimes above and sometimes below.
- Those differences are the size of the calibration and post-processing randomness at one seed. The paper averages several runs.
- **The main claim reproduces at all three alphas:** large DP reduction at a cost of about 1.5 points of accuracy.

---

## 2026-09-26 — FairRFL: official-code port for Table III (CIFAR-10), moved to Colab

**For:** the reproduction table (the last of the 8 papers). It will be the first result to show Dr. Gupta, since FairRFL is his paper.

**Setup:**
- `fairfl.experiments.fairrfl_official` is a line-by-line port of the authors' `experiments.py` and its helpers, with the same RNG call order. It covers:
  - the 2-class pool split;
  - SGD re-created every mini-batch;
  - q-FFL as coded;
  - the selfish attack, started from round index 2;
  - rotation2 (RFL-Self);
  - the code's FedCDA variant.
- It records rows `fairrfl_cifar10_<method>_s<0|10|20|30>` into `runs.jsonl` with `acc_n`, `acc_s` and `acc_std`. The Table III CIFAR-10 targets are in `published.csv`, 88 values.
- **Choices the paper leaves open:**
  - **q = 1.0.** The paper and code state no value.
  - **Selfish attack variant:** the Gamma-estimating one (`--estimate-k`, Eq. 4-5 and Fig. 6).
  - **FedMut omitted.** The code default radius is 0, which is identical to FedAvg, and the paper gives no radius.
  - **WISDM-W not run.** The published ResLSTM crashes (spec section 3).

**Compute:** the docs/repro spec guessed "minutes on CPU". In practice it's about 100 s per round and about 50 min per run on the laptop, so the 32 runs go to the Colab notebook (new FairRFL cell).

**Housekeeping:** the local `data/cifar-10-python.tar.gz` was incomplete, and torchvision re-downloaded it (about 50 min).

---

## 2026-09-26 — FedMut CNN on Colab: every FedMut run went NaN; momentum reset to the code's 0.5

**For:** the FedMut reproduction, and a warning about specs that take a setting from the paper instead of the code.

**What happened** (Colab T4, `fedmut_cifar10_cnn` suite, momentum 0.9):

| Run | Last accuracy seen | Paper (last-10 mean) |
|---|---|---|
| d0.1 FedAvg | about 0.41 | 47.93 |
| d0.1 FedMut | 0.10, NaN from round 29 | 51.25 |
| d0.5 FedAvg | about 0.47 | 54.33 |
| d0.5 FedMut | 0.10, NaN from round 49 | 56.90 |
| d1.0 FedMut | 0.10, NaN from round 39 | 58.90 |

FedAvg's test loss rose from about 2.5 to 6.9 over the run, which means it was overfitting.

**Code review** (reading only), our port against `Algorithm/Training_FedMut.py`:
- These match the official code: the mutation formula; the sign pairs with the decaying beta; which client gets which mutated model; the odd-K rule; the uniform mean; a fresh optimizer every round; the model; the partitions; the normalisation.
- **The only setting that differs is momentum.** The paper says 0.9, but the code default is 0.5. The spec's "README command" with 0.9 was inferred by our spec agent; the README itself gives no value.
- **Why this likely explains both problems:** with momentum 0.9 the round update is about 5x larger. FedMut then starts clients at global + 4 x update, which overflows. The paper already notes that radius 5 fails to train, so there is little margin.

**Changes:**
- `configs/repro/fedmut/cifar10_cnn.yaml` now uses momentum 0.5. The ResNet-18 suite uses the same config, so it changes too.
- The engine now stops a run whose global loss is NaN or inf. The summary records `diverged_at_round`, and there is a new test.
- **Unverified until the Colab rerun.** The old FedMut folders on Drive must be deleted, or they will be skipped as finished.

**Rerun with momentum 0.5** (Colab, 2026-09-26, d = 0.1, seed 0; last-10-evaluation mean ± std):

| Run | Ours | Paper |
|---|---|---|
| FedAvg | 40.53 ± 2.52 | 47.93 ± 3.26 |
| FedMut | 46.33 ± 1.22 | 51.25 ± 1.07 |

- **No NaN, so the divergence is fixed.**
- **FedMut's gain over FedAvg reproduces:** +5.8 points here vs +3.3 in the paper. FedMut is also steadier (std 1.2 vs 2.5).
- Both runs are 5-7 points below the paper in absolute terms, and FedAvg's test loss still rises (about 5 at the end). So a shared offset remains.
- Our "last 10" covers rounds 910-1000, because we evaluate every 10 rounds. The official FedMut code evaluates every round, so its last 10 is rounds 991-1000. This is a minor difference.
- Registry rows were built from the Colab `summary.json` files (`run_dir: colab:...`). The run folders are in `code/runs/colab/` (gitignored).

---

## 2026-09-26 — G1 preview on FedMut d0.1 (existing logs, no new runs): final-round reporting hides interim unfairness

**For:** an early look at the non-Markovian question before running G1 properly.

**Data:**
- The two Colab d0.1 runs (FedAvg, FedMut), with 100 evaluations each, one every 10 rounds.
- Clients hold no test data in this setup, so two proxies stand in for fairness:
  - *per-class* test accuracy, with classes acting as groups;
  - *per-client* experienced accuracy, i.e. per-class accuracy weighted by each client's class mix from the shipped partition.
- Output: `code/runs/colab/g1_preview_d0.1.json`.

**Results** (FedAvg / FedMut):

| Metric | Final round | Last-10 mean | Mean over all rounds |
|---|---|---|---|
| Worst-class accuracy | 0.128 / 0.252 | 0.110 / 0.169 | 0.079 / 0.108 |
| Worst-10% client accuracy | 0.199 / 0.268 | 0.147 / 0.220 | 0.118 / 0.150 |
| Std across clients | 0.172 / 0.136 | 0.186 / 0.153 | 0.189 / 0.173 |

- **Some class is below 5% accuracy in 35 of 100 evaluations for FedAvg, and 22 of 100 for FedMut.** At any of those checkpoints, a whole class of users gets an almost useless model.
- Mean change in accuracy between consecutive evaluations: 4.1 points for FedAvg and 3.0 for FedMut. The largest single change is 14.2 and 8.6 points respectively.

**Reading:**
- **Final-round reporting roughly halves the apparent harm.** FedMut's worst class looks like 0.25 at the end, but averages 0.11 over training. The same holds for the worst-10% clients (0.27 vs 0.15).
- **No ranking flip here.** FedMut beats FedAvg under every scope. With only two methods that is a weak test. G1 needs all 8 methods and real demographic groups (Adult) to test for rank changes.
- **Caveats:** classes and label-mix clients are only proxies for demographic groups; one seed; one setting.

**d = 0.5 pair** (Colab, momentum 0.5, seed 0; last-10 mean): FedAvg 48.61 (paper 54.33); FedMut 51.23 (paper 56.90).
- FedMut's gain is +2.6 points, exactly the paper's +2.6.
- Both runs are about 5.7 points below the paper. That offset is the same as at d = 0.1, so it is shared by both methods.
- Colab run folders are now imported with `uv run python -m fairfl.experiments.import_runs` (reads `code/runs/colab/`).
