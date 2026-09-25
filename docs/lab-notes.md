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
