# FedFDP reproduction spec

Paper: Ling et al., "FedFDP: Fairness-Aware Federated Learning with Differential Privacy", ACNS 2026, arXiv 2402.16028.

Source used: `docs/impl-papers/Fedfdp.pdf` (30 pp). It is byte-identical (same md5) to arXiv v6 (7 Jan 2026) in the scratchpad, so "published vs arXiv" differences do not arise here. Page numbers below are the printed page numbers of that PDF.

Table numbering in this version (the task brief's "Table 1/2/3" map to these):

| Brief | This PDF | Content |
|---|---|---|
| "Table 1" (main results) | Table 2 (FedFair, no DP) and Table 3 (FedFDP, DP) | p.15 |
| "Table 2" (heterogeneity/scalability) | Table 4 (no DP) and Table 5 (DP) | p.16 |
| "Table 3" (epsilon sweep) | Table 6 (FMNIST, eps 1 to 4) | p.18 |
| n/a | Table 1 = communication overhead; Table 7 = eps at target accuracy (MNIST Dir(0.05)); Table 8 = notation | p.14, p.19, p.24 |

## 1. Datasets, heterogeneity, clients

- Datasets: MNIST, FashionMNIST, CIFAR10 (Sec 7.1 "Tasks setting", p.14).
- Partition: Dirichlet label skew Dir(alpha), "widely used heterogeneous settings [21, 27]", **default alpha = 0.1** (p.14).
- Other alphas: Dir(0.5) on MNIST, Dir(1) on FashionMNIST (Tables 4/5, p.16; the header groups Dir(0.1) and Dir(0.5) under MNIST and Dir(1) under FashionMNIST). Dir(0.05) on MNIST for Table 7 (p.19).
- Clients: **default N = 10**, all clients participate every round (Algorithm 1 line 4 / Algorithm 2 line 3: "for i = 1..N parallel"; no client sampling). Scalability: 10 / 20 / 50 clients, **CIFAR10 only**, alpha presumably the default 0.1 (not stated) (Tables 4/5, p.16).
- Train/test split: not specified (looked in: Sec 7.1, Appendix A, Table 8). Accuracy is "test accuracy"; whether it is a global test set or per-client test splits is not stated. Footnote 4 (p.5) says "the test data from the server cannot be used for training", implying a server-side test set.
- Preprocessing / normalisation: not specified.

## 2. Model and training

- Model: "4-layer CNN [30] which consists of two convolutional layers and two fully connected layers" (p.14), i.e. the McMahan et al. FedAvg CNN, same for all three datasets. Channel sizes, kernel sizes, hidden width: not specified (the McMahan CNN is conv5x5-32, pool, conv5x5-64, pool, FC-512, FC-10).
- Optimiser: plain SGD (Eq. 9, Eq. 12). No momentum or weight decay mentioned.
- Learning rate: **eta = 0.1 for FedFair (non-DP) and its baselines; eta = 1.0 for FedFDP (DP) and its DP baselines** (p.14 "Hyperparameters").
- Local work per round:
  - FedFair (Alg. 1, p.5): iterate over all batches of D_i once per round, i.e. **1 local epoch**; batch size not specified.
  - FedFDP (Alg. 2, p.8): **one DP-SGD step per round** on a single batch B_i "sampled randomly with probability q" (Poisson sampling, line 10); expected batch size q*|D_i| (Table 8, p.24). So T rounds = T noisy steps per client.
- Sampling rate: **q = 0.05** (p.14). Poisson sampling is implied by "with probability q" and by the SGM accountant; the loss-noise normalisation divides by the realised |B_i| (Alg. 2 lines 15 and 20).
- Rounds T: **not specified** for any table (looked in: Sec 7.1, 7.2, 7.3, Table 8, Appendix A). For DP runs T is implicitly set by the budget: train until eps = 3.52 is spent (see section 4 below; about 776 rounds under the paper's stated accountant, gradient term only). For FedFair (non-DP) runs T is unknown.
- Aggregation: w_{t+1} = sum_i p_i w_i^{t+1}, p_i = |D_i| / sum_j |D_j| (Eq. 1, Alg. 2 line 5).
- Seeds / repetitions: Tables 2/3 report mean +- std, number of runs not specified. Tables 4-7 have no std.
- Framework: PyTorch 1.8, single RTX 4090 (p.14). No code link in the paper.

## 3. Method defaults

- Fairness parameter lambda: **no default given in Sec 7.1**. Sec 5.1 (pp.10-11) gives a closed form lambda* (largest real root of the cubic G(lambda), Eqs. 17-22), but it depends on the unknown constants L, mu, G, Gamma, Q0, Q1, so the paper itself tunes lambda empirically (p.18, "Impact of Different lambda"). Fig. 2 (p.17):
  - FashionMNIST sweep lambda in {1e-4, 5e-3, 1e-2, 5e-1, 1}: best at **lambda = 1e-2** (acc about 86.0, Psi about 0.3e9). This matches Table 3 FMNIST (85.99), so 1e-2 is the likely FMNIST default.
  - CIFAR10 sweep lambda in {1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1}: best Psi at about **1e-6 to 1e-8**, acc about 55.
  - MNIST lambda: not specified. FedFair (non-DP) lambda: not specified.
- Fair clipping (Eq. 13, p.7), per sample j in B_i:
  `C_ij = min(1 + lambda * delta_ij, C / ||grad F_i(w; j)||)`, `delta_ij = F_i(w; j) - F~(w_t)`,
  update `w <- w - eta/|B_i| * ( sum_j C_ij * grad_j + C * N(0, I) )`, noise std = sigma*C (Eq. 5, Eq. 12; the sigma inside Eq. 12 is dropped by the PDF extraction but Eq. 5 and Alg. 2 define noise std C*sigma).
  Note F~(w_t) is the server's DP-noised weighted mean loss from the previous round (Alg. 2 line 6).
- **Gradient clip C = 0.1**, **sigma = 2.0** (p.14). Sweeps: C in {0.01, 0.1, 1, 5, 10} (Fig. 3), sigma in {1, 1.5, 2, 2.5, 3} (Fig. 4).
- Loss upload (Alg. 2 lines 16-20, Eqs. 14-15, p.8): after the step, each client computes per-sample loss on the same B_i, clips to [0, C_l^{i,t}], uploads
  `F~_i = (1/|B_i|) * ( sum_j clip(F_i(w_{t+1}; j)) + sigma_l * C_l^{i,t} * N(0,1) )`.
  Adaptive bound: C_l^{i,t} = previous round's noised clipped mean loss (Eq. 14). **Initial C_l^0 = 2.5**, **sigma_l = 5.0** (p.14). Eq. 14 can produce a non-positive bound after noise; handling not specified (guard with max(small, .)).
- delta = 1e-5, target eps = 3.52 (p.14).
- Initial F(w_0): Alg. 1 line 1 "Initial()", not specified (use mean initial loss, DP-noised, or 0).

## 4. Privacy accounting (how eps = 3.52 arises)

Per-client, sample-level DP; no SecAgg (Sec 6, p.11).

Per round, two subsampled Gaussian mechanisms (SGM) with the same Poisson rate q act on client i's data:
- gradient step: sensitivity C, noise std sigma*C, so noise multiplier sigma (Theorem 2, Eq. 23);
- loss upload: sensitivity C_l, noise std sigma_l*C_l, noise multiplier sigma_l (Theorem 3, Eq. 27).

RDP of one SGM step at integer order alpha (Mironov et al. 2019; Eq. 26):

```
A(q, s, alpha) = sum_{k=0}^{alpha} C(alpha,k) (1-q)^(alpha-k) q^k exp((k^2 - k) / (2 s^2))
eps_RDP_step(alpha) = log A(q, s, alpha) / (alpha - 1)
```

(Eq. 23/27 as printed write T/(alpha-1) * sum(...) without the log; that is a typo, the proof via Definition 3 has the log.)

Composition over T rounds (Definition 4):
`R(alpha) = T * [ log A(q, sigma, alpha) + log A(q, sigma_l, alpha) ] / (alpha - 1)` (Theorem 4, Eq. 28).

Conversion to (eps, delta) (Lemma 1, Balle et al. 2020; same formula as Opacus `get_privacy_spent`):
`eps = min_alpha [ R(alpha) + ln((alpha-1)/alpha) - (ln delta + ln alpha)/(alpha - 1) ]`.

Orders alpha: not specified. The Opacus default grid {1.1, 1.2, ..., 10.9} U {12, ..., 63} (fractional orders need the Mironov fractional-alpha formula) reproduces the paper's numbers; integer orders 2..63 give the same results here.

Calibration I ran (scratchpad `acc3.py`, `acc4.py`, `acc5.py`; q = 0.05, delta = 1e-5):

| Check | Paper | Gradient term only | Gradient + loss term (sigma_l = 5, rate q) |
|---|---|---|---|
| Max T at eps = 2, sigma = 1 / 1.5 / 2 / 2.5 / 3 (p.18, Fig. 4 text) | 6 / 115 / 268 / 463 / 708 | 6 / 114 / 267 / 463 / 702 | 5 / 108 / 237 / 379 / 525 |
| T at eps = 3.52, sigma = 2 | not stated | **776** | 683 |
| eps at T = 600, sigma = 2 | | 3.05 | 3.27 |

Conclusions:
- The paper's own T values (Fig. 4) match the **gradient-only** accountant with the Lemma 1 conversion almost exactly (off by at most 1 to 6 rounds, likely an order-grid or rounding difference). The loss mechanism of Theorem 3 does not appear to be counted in the reported T, despite Theorem 4.
- So the default run is most likely **sigma = 2, q = 0.05, delta = 1e-5, about T = 776 rounds (one DP step per round) to reach eps = 3.52**. If you count the loss term as Theorem 4 says, stop at T = 683.
- Side note: with the classic Mironov conversion eps = R(alpha) + ln(1/delta)/(alpha-1), gradient-only, T = 600 gives eps = 3.518, which is a suspiciously round match to 3.52. That conversion does NOT reproduce the Fig. 4 T values, so it is the less likely reading, but T = 600 is a reasonable alternative if numbers do not line up.
- Table 6 (p.18): eps in {1, 2, 3} uses "larger sigma to support more iterations" (sigma values not given); eps = 4 uses sigma = 1.65 "with a similar number of iterations as eps = 3". Gradient-only accountant: sigma = 2 gives T = 65 / 267 / 580 / 988 for eps = 1 / 2 / 3 / 4; sigma = 1.65 at eps = 4 gives T = 606 (similar to 580, consistent).
- Baselines use the same DP-SGD (C, sigma, q, eta = 1.0) and the same eps = 3.52 (p.13 "for FedFDP, DP was applied to each baseline algorithm"). Whether q-FFL's uploaded loss is also noised is not stated.

## 5. Fairness metric Psi

Eq. 2 (p.4): `Psi(w) = sum_i p_i (F_i(w) - F(w))^2`, `F(w) = sum_i p_i F_i(w)`, `p_i = |D_i| / |D|`, where F_i is the loss (cross-entropy assumed) of the **final global model w** on client i. Weighted variance of client losses, lower is better.

Which data: not specified (looked in: Sec 2.1, 7.1, 7.2, footnote 4, Appendix A). The definition uses F_i, the local (training) loss function; I suggest computing on each client's local training data with the final global model, and also logging it on per-client test splits.

Warning on magnitude: Tables 3/5/6/7 (DP) report Psi of order 1e8 to 1e11, versus 1e-3 to 1e0 without DP (Tables 2/4). A weighted variance of cross-entropy losses of order 1e10 implies per-client losses of order 1e5, which is not plausible for a converged 94%-accuracy model. Possibly the DP runs compute Psi from something else (e.g. unnormalised or noised losses) or have a scaling bug. Compare Psi across methods by ratio, not absolute value.

Accuracy: "test accuracy (%)", global model; test set not specified.

## 6. Target numbers

### Table 2 (p.15): no DP (FedFair and baselines), eta = 0.1, Dir(0.1), 10 clients. Acc % / Psi, mean +- std

| Method | MNIST Acc | MNIST Psi | FMNIST Acc | FMNIST Psi | CIFAR10 Acc | CIFAR10 Psi |
|---|---|---|---|---|---|---|
| FedAvg | 98.67 +- 0.09 | 7.3e-3 +- 5.4e-3 | 87.05 +- 1.02 | 3.6e-1 +- 2.2e-1 | 62.03 +- 1.00 | 1.1e0 +- 8.9e-1 |
| SCAFFOLD | 98.45 +- 0.03 | 3.0e-2 +- 2.6e-2 | 86.77 +- 0.25 | 6.1e0 +- 4.0e0 | 62.21 +- 1.37 | 9.9e0 +- 1.1e1 |
| FedProx | 98.70 +- 0.13 | 7.5e-3 +- 5.1e-3 | 87.30 +- 0.51 | 3.5e-1 +- 2.1e-1 | 62.13 +- 0.20 | 1.2e0 +- 3.1e-1 |
| FedDyn | 98.40 +- 0.23 | 8.0e-3 +- 2.1e-3 | 87.48 +- 0.31 | 6.2e-1 +- 3.5e-1 | 62.33 +- 1.15 | 3.2e0 +- 8.8e-1 |
| q-FFL | 98.72 +- 0.03 | 4.7e-3 +- 1.0e-3 | 87.35 +- 0.13 | 5.1e-1 +- 5.2e-2 | 63.33 +- 0.21 | 9.4e-1 +- 3.3e-2 |
| FedFair | 98.75 +- 0.09 | 3.9e-3 +- 1.3e-3 | 87.70 +- 0.76 | 2.6e-1 +- 2.3e-1 | 62.38 +- 0.88 | 8.5e-1 +- 6.6e-1 |

### Table 3 (p.15): DP, eps = 3.52, all baselines with DP

| Method | MNIST Acc | MNIST Psi | FMNIST Acc | FMNIST Psi | CIFAR10 Acc | CIFAR10 Psi |
|---|---|---|---|---|---|---|
| FedAvg | 93.40 +- 0.47 | 1.1e11 +- 0.5e10 | 84.15 +- 1.97 | 3.0e8 +- 1.4e8 | 52.61 +- 0.17 | 6.1e9 +- 5.2e9 |
| SCAFFOLD | 93.95 +- 1.39 | 7.0e10 +- 4.9e9 | 83.41 +- 4.87 | 3.3e9 +- 4.7e9 | 53.85 +- 1.16 | 6.2e9 +- 1.3e9 |
| FedProx | 90.50 +- 3.95 | 5.3e10 +- 7.3e9 | 83.85 +- 0.98 | 7.9e8 +- 4.5e8 | 51.34 +- 0.91 | 4.4e9 +- 2.1e9 |
| FedDyn | 91.42 +- 2.25 | 5.6e10 +- 9.8e9 | 84.04 +- 1.21 | 8.8e8 +- 3.2e8 | 52.14 +- 1.39 | 5.5e9 +- 1.2e9 |
| ALI-DPFL | 90.89 +- 1.65 | 3.3e10 +- 8.3e9 | 83.65 +- 1.68 | 6.6e8 +- 3.5e8 | 52.05 +- 1.26 | 3.6e9 +- 1.6e9 |
| q-FFL | 93.74 +- 1.41 | 7.8e10 +- 1.1e11 | 83.13 +- 2.74 | 4.2e9 +- 2.6e9 | 48.46 +- 1.00 | 4.7e9 +- 1.9e9 |
| FedFDP | 95.13 +- 0.83 | 2.3e10 +- 1.0e10 | 85.99 +- 0.76 | 2.8e8 +- 0.8e8 | 54.21 +- 0.98 | 2.6e9 +- 1.6e9 |

Inconsistency: FedFDP MNIST Dir(0.1) is 95.13 in Table 3 but 94.13 in Table 5 (same setting).

### Table 4 (p.16): no DP. Heterogeneity (MNIST Dir 0.1 / 0.5, FMNIST Dir 1) and scalability (CIFAR10, 10/20/50 clients)

| Method | MNIST Dir0.1 Acc / Psi | MNIST Dir0.5 | FMNIST Dir1 | C10 10 cl | C10 20 cl | C10 50 cl |
|---|---|---|---|---|---|---|
| FedAvg | 98.67 / 7.3e-3 | 98.37 / 1.2e-3 | 89.41 / 3.6e-2 | 62.03 / 1.1e0 | 60.89 / 1.5e0 | 59.78 / 1.5e0 |
| SCAFFOLD | 98.45 / 5.0e-3 | 98.51 / 2.1e-3 | 86.39 / 3.4e-2 | 62.21 / 9.9e0 | 59.48 / 8.7e0 | 58.64 / 2.3e0 |
| FedProx | 98.70 / 7.5e-3 | 98.35 / 1.2e-3 | 88.33 / 3.5e-2 | 62.13 / 1.2e0 | 60.33 / 2.2e0 | 61.31 / 1.6e0 |
| FedDyn | 98.40 / 8.0e-3 | 98.36 / 1.8e-3 | 88.26 / 3.5e-2 | 62.33 / 9.4e-1 | 61.35 / 2.6e0 | 62.52 / 1.5e0 |
| q-FFL | 98.72 / 4.7e-3 | 98.61 / 1.1e-3 | 89.03 / 3.6e-2 | 63.33 / 9.4e-1 | 60.63 / 1.3e0 | 59.65 / 1.5e0 |
| FedFair | 98.75 / 3.9e-3 | 98.72 / 1.2e-3 | 89.60 / 1.7e-2 | 62.38 / 8.5e-1 | 61.58 / 1.1e0 | 62.90 / 1.5e0 |

(SCAFFOLD MNIST Dir0.1 Psi is 5.0e-3 here vs 3.0e-2 in Table 2; FedDyn C10 10-client Psi 9.4e-1 here vs 3.2e0 in Table 2.)

### Table 5 (p.16): DP eps = 3.52, same layout

| Method | MNIST Dir0.1 | MNIST Dir0.5 | FMNIST Dir1 | C10 10 cl | C10 20 cl | C10 50 cl |
|---|---|---|---|---|---|---|
| FedAvg | 93.40 / 1.1e11 | 94.17 / 4.1e9 | 85.60 / 3.7e9 | 52.61 / 6.1e9 | 51.58 / 2.0e9 | 55.90 / 1.5e9 |
| SCAFFOLD | 93.95 / 7.0e10 | 94.06 / 1.9e9 | 84.56 / 3.6e9 | 53.85 / 6.2e9 | 54.34 / 2.9e8 | 53.98 / 2.2e9 |
| FedProx | 90.50 / 5.3e10 | 94.23 / 3.1e9 | 84.38 / 5.6e9 | 51.34 / 4.4e9 | 53.67 / 4.5e8 | 54.14 / 2.2e9 |
| FedDyn | 91.42 / 5.6e10 | 93.08 / 3.8e9 | 85.21 / 9.8e10 | 52.14 / 5.5e9 | 50.21 / 1.2e10 | 51.86 / 2.0e10 |
| ALI-DPFL | 90.89 / 3.3e10 | 93.98 / 2.6e9 | 83.66 / 2.5e10 | 52.05 / 3.6e9 | 38.35 / 2.4e10 | 33.46 / 1.6e10 |
| q-FFL | 93.74 / 7.8e10 | 93.86 / 2.5e8 | 85.20 / 7.4e8 | 48.46 / 4.7e9 | 51.12 / 2.7e8 | 49.83 / 1.1e7 |
| FedFDP | 94.13 / 2.3e10 | 94.56 / 3.4e9 | 86.77 / 6.0e8 | 54.21 / 2.6e9 | 54.52 / 2.2e8 | 56.49 / 5.3e8 |

### Table 6 (p.18): eps sweep, FashionMNIST (Dir 0.1, 10 clients assumed). Acc % / Psi

| Method | eps 1.0 | eps 2.0 | eps 3.0 | eps 4.0 |
|---|---|---|---|---|
| FedAvg | 61.68 / 1.1e6 | 82.69 / 2.3e7 | 84.07 / 3.8e8 | 86.83 / 8.0e8 |
| SCAFFOLD | 62.12 / 1.2e6 | 82.36 / 2.3e7 | 83.39 / 2.6e9 | 85.41 / 5.6e8 |
| FedProx | 61.25 / 1.1e6 | 81.62 / 2.3e7 | 85.32 / 3.8e8 | 85.68 / 2.4e9 |
| FedDyn | 62.38 / 1.5e6 | 82.43 / 3.1e7 | 83.29 / 2.9e8 | 84.51 / 5.2e9 |
| ALI-DPFL | 61.78 / 1.1e6 | 83.01 / 2.3e7 | 84.12 / 3.9e8 | 85.58 / 4.5e9 |
| q-FFL | 58.99 / 3.6e6 | 80.24 / 9.6e7 | 82.92 / 5.5e8 | 85.01 / 3.8e9 |
| FedFDP | 63.36 / 1.0e6 | 83.15 / 2.1e7 | 85.59 / 9.5e7 | 86.97 / 3.8e8 |

sigma per eps: "larger sigma" for eps 1-3 (values not given), sigma = 1.65 for eps 4 (p.18).

### Table 7 (p.19): eps and Psi needed to reach a target accuracy, MNIST Dir(0.05). eps / Psi

| Method | 80% | 85% | 90% | 95% |
|---|---|---|---|---|
| FedAvg | 2.43 / 2.10e9 | 3.02 / 8.90e9 | 3.45 / 1.30e10 | 4.89 / 1.30e11 |
| SCAFFOLD | 2.55 / 3.30e9 | 3.15 / 1.10e10 | 3.67 / 3.70e10 | 5.32 / 1.10e11 |
| FedProx | 3.02 / 1.30e9 | 2.98 / 1.30e10 | 4.31 / 5.80e10 | 4.89 / 2.50e11 |
| FedDyn | 2.68 / 2.20e9 | 3.55 / 7.80e9 | 3.98 / 1.10e10 | 5.15 / 3.30e11 |
| ALI-DPFL | 2.85 / 3.10e9 | 2.98 / 1.30e10 | 3.67 / 2.30e10 | 5.32 / 9.80e10 |
| q-FFL | 2.68 / 9.80e8 | 3.15 / 9.70e9 | 4.12 / 8.80e10 | 4.04 / 1.02e11 |
| FedFDP | 2.25 / 6.60e8 | 2.70 / 1.20e9 | 3.05 / 5.80e9 | 3.77 / 2.80e10 |

Figures: Fig. 2 (lambda), Fig. 3 (C), Fig. 4 (sigma, eps = 2), Fig. 5 (Psi vs eps at 60% target accuracy, MNIST and FMNIST, Psi 1e6 falling to about 3e4 over eps 1 to 7), all p.17-19.

Baseline DP status: Table 2 / Table 4 baselines have no DP (eta = 0.1); Table 3 / 5 / 6 / 7 baselines all have DP-SGD with the same settings (p.13).

## 7. Unspecified or conflicting

- Rounds T for all runs (DP: inferred about 776 from the accountant; non-DP: unknown).
- lambda default per dataset (FMNIST likely 1e-2, CIFAR10 about 1e-6 from Fig. 2; MNIST unknown; FedFair lambda unknown).
- Batch size for FedFair / non-DP runs; local epochs for non-DP baselines (Alg. 1 implies 1 epoch).
- Local steps per round for DP baselines (assume 1 DP step on a Poisson batch, same as FedFDP, so eps matches).
- Exact CNN widths, weight init, normalisation.
- Train/test split per client; data used for Psi; number of seeds.
- RDP order grid; whether the loss mechanism is counted in eps (evidence says not).
- Handling of negative adaptive loss clip bound; initial F(w_0).
- Hyperparameters of baselines (FedProx mu, FedDyn alpha, q-FFL q, ALI-DPFL settings): not specified.
- Psi magnitudes under DP (1e8 to 1e11) look implausible for cross-entropy loss variance.
- Internal conflicts: Table 3 vs Table 5 FedFDP MNIST (95.13 vs 94.13); Table 2 vs Table 4 SCAFFOLD MNIST Psi and FedDyn CIFAR10 Psi; Table 7 narrative (p.19) cites eps 2.70 / Psi 1.20e9 as the 95% column but those are the 85% column.
- Heterogeneity alpha for the scalability (20/50 client) runs: not stated, presumably 0.1.
