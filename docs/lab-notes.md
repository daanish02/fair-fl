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
| FCFL (alpha 0.3, r 0.4) | 96.06 / pending | 11.03 / pending | 89.17 / pending | 100 / pending |

**Reading:**
- Our FedAvg is about 2.5 points more accurate and about 3.6x fairer (variance) than the FedAvg the paper reports.
- The paper's baselines are not released ("we directly rewrite the code for comparison", p.397). Its FedAvg therefore used an unknown local setup, while ours uses FCFL's own client pipeline. The published baseline gap may partly be a baseline-implementation gap.
- q-FedAvg at q = 0.2 does not reduce variance in our run (5.58 vs 4.17), contrary to the paper's ordering. The paper picked q per method by grid search ("best performance"); we ran only the reported best q.
- **What decides this reproduction** is whether FCFL beats our FedAvg on variance and worst-10%, not the absolute numbers.

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
- **Results for FairWeight itself** (Adult, Bank, Default, Law; seed 0) are pending.
- **Table III (the FACE/ATE loss)** is not ported. It needs the authors' propensity-matched potential outcomes, which their code misaligns (see `docs/repro/fairweight.md` section 4).
