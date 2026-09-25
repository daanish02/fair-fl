# FedCDA reproduction spec

Paper: H. Wang, H. Xu, Y. Li, Y. Xu, R. Li, T. Zhang, "FedCDA: Federated Learning with Cross-rounds Divergence-aware Aggregation", ICLR 2024.
Source used: `docs/impl-papers/FEDCDA - ...pdf` (12 pages, main text + refs only). The ICLR proceedings PDF (proceedings.iclr.cc/paper_files/paper/2024/file/a559a5a8aa5ae6682ced009ad97cdb16-Paper-Conference.pdf) was downloaded and is also 12 pages with no appendix. The appendix (A: example, B: proofs, C: aggregation time, D: "broader range of hyperparameters") lives only in the OpenReview version (openreview.net/forum?id=nbPGqeH3lt), which returned a bot-challenge page to curl, the OpenReview API, and a text proxy. **No official code found** (gh search repos/code "FedCDA", web search, ICLR poster page: none linked).

All page numbers below refer to the paper's printed page numbers.

## 1. Datasets and models (Sec. 6.1, p. 7)

| Dataset | Classes | Model |
|---|---|---|
| Fashion-MNIST | 10 | simple CNN |
| CIFAR-10 | 10 | ResNet-18 |
| CIFAR-100 | 100 | ResNet-18 |

Simple CNN (FMNIST), as stated: conv 5x5, 32 ch -> 2x2 max-pool -> conv 5x5, 64 ch -> 2x2 max-pool -> FC 512 + ReLU -> FC 10. (This is the McMahan et al. 2017 "CNN".)
- Not specified (looked in: Sec. 6.1, proceedings PDF): padding of the convs, ReLU after the convs (implied, not stated), input normalization, data augmentation.
- ResNet-18 variant: not specified (looked in: Sec. 6.1). Whether it is the CIFAR variant (3x3 stem, no max-pool) or torchvision ImageNet ResNet-18, and whether BatchNorm or GroupNorm is used, is not stated. Assume CIFAR-style ResNet-18 with BatchNorm (most common in this line of work); if the testbed uses GroupNorm, expect differences, especially under Dirichlet 0.1 / shards 2.

## 2. FL setup (Sec. 6.1, p. 7)

- N = 20 clients total; 20% sampled uniformly at random per round -> P = 4 clients/round.
- Partitions (one per cell of Table 1):
  - Shards (McMahan et al. 2017a): sort samples by label, split into N x S shards, assign S shards per client at random. S in {2, 4, 8} shards per client.
  - Dirichlet (Lin et al. 2020, label-Dirichlet): alpha in {0.1, 0.3, 0.5}.
- Test set: not specified explicitly (looked in: Sec. 6.1, 6.2). Assume the global model is evaluated on the standard central test split.
- Seeds: not specified. Each setting is run twice (2 runs).

## 3. Training hyperparameters (Sec. 6.1, p. 7) -- all methods, all datasets

| Item | Value |
|---|---|
| Optimizer | SGD |
| Learning rate | 1e-3 |
| Momentum | 1e-4 (as printed; unusual, possibly a typo for 0.9 but reproduce as printed) |
| Weight decay | 1e-5 |
| Batch size | 64 |
| Local epochs E | 20 |
| Rounds T | 200 |
| Warmup (FedCDA only) | first 50 rounds use FedAvg, then FedCDA |
| Reported metric | mean +/- std over {2 runs x final 10 rounds} test accuracy |

- LR schedule/decay: not specified (looked in: Sec. 6.1). Assume constant.
- Hardware: PyTorch 2.0, 8x RTX 3090.

Table 2 variant (Sec. 6.2, p. 8): N = 100 clients, 10% sampled per round, SGD without momentum, lr 0.1, weight decay 1e-3, local epochs 5, ResNet-18 on CIFAR-100, Dirichlet {0.1, 0.3, 0.5}. Batch size, rounds, warmup, K, B not restated (assume same as main: 64, 200, 50, 3, 3).

## 4. FedCDA hyperparameters (Sec. 4, 6.1; Alg. 1 p. 6)

| Symbol | Meaning | Value |
|---|---|---|
| K | cached local models per client (memory size) | 3 |
| B | number of batches in greedy batch-wise selection | 3 |
| L | smoothness constant in L_n(w) = F_n(w) + (L/2)||w||^2 | 1 (all clients) |
| warmup | FedAvg rounds before FedCDA | 50 |

Algorithm (Alg. 1, eq. 7-8, p. 4-6):
1. Each sampled client trains E epochs from global w_t and also reports its accumulated local loss F_n(w_n) over its last local epoch.
2. Server updates each sampled client's cache W_n (FIFO, keep last K received models).
3. Participating clients are randomly split into B equal-size batches. For batch b, choose, for each client in the batch, one of its K cached models, to minimise
   `(1/(N-P+bP/B)) * sum_{n in selected set} L_n(w_n) - (L/2) ||w_bar||^2`
   where the sum covers non-participating clients (fixed, their previously selected model), batches 1..b-1 (fixed), and batch b (being chosen); w_bar is the mean of those models. Exhaustive search over K^(P/B) combos per batch.
4. Global model = uniform mean (1/N) over **all N** clients' selected/fixed models (not only the participants; not data-size weighted).
- Note: with P = 4 and B = 3 batches are not equal-size; how the split is handled is not specified (looked in: Sec. 4, 6.1).
- Before a client has K cached models, or for never-sampled clients (their "fixed" model), handling is not specified (looked in: Alg. 1, Sec. 4.1). Reasonable choice: use whatever is cached; never-sampled clients excluded or set to the current global.
- Local loss F_n(w_n) for older cached models is the loss recorded when that model was trained (not recomputed).

Fig. 2(a) ablation (p. 8-9): CIFAR-10 ResNet-18, Dirichlet 0.1/0.3/0.5, "only sample 10 clients in each round, where the sample ratio is 0.3", K = 3; compares FedAvg / newest / random / approximate (eq. 8) / optimal (eq. 3). Total clients for this ablation not specified (10/0.3 is ~33).
Fig. 2(b): K in {1,2,3,4} on CIFAR-10 Dir 0.1. Fig. 2(c,d): local epochs {10,20,30,40} for FedCDA and FedAvg. Numeric values only readable from plots.

## 5. Target numbers

### Table 1 (p. 8): test accuracy (%), mean +/- std, 20 clients, 20% participation

Shards (S = shards per client):

| Method | FMNIST S=2 | S=4 | S=8 | CIFAR-10 S=2 | S=4 | S=8 | CIFAR-100 S=2 | S=4 | S=8 |
|---|---|---|---|---|---|---|---|---|---|
| FedAvg | 64.69+/-5.62 | 74.78+/-4.55 | 76.81+/-3.33 | 28.10+/-3.96 | 59.83+/-2.94 | 70.87+/-1.91 | 11.86+/-1.19 | 15.87+/-1.00 | 21.91+/-0.55 |
| FedProx | 64.21+/-4.11 | 70.76+/-3.89 | 72.19+/-4.16 | 26.39+/-4.16 | 53.03+/-2.29 | 70.91+/-1.87 | 10.87+/-0.58 | 15.37+/-0.46 | 24.16+/-0.33 |
| FedExP | 65.24+/-3.47 | 69.31+/-4.62 | 76.66+/-5.04 | 26.84+/-4.75 | 59.31+/-3.61 | 69.53+/-1.94 | 11.59+/-0.81 | 16.47+/-0.99 | 23.58+/-1.36 |
| FedSAM | 59.28+/-0.15 | 75.19+/-0.10 | 76.07+/-0.09 | 29.31+/-0.32 | 57.12+/-0.08 | 61.56+/-0.31 | 11.19+/-0.16 | 15.95+/-0.15 | 22.44+/-0.16 |
| FedDF | 64.72+/-2.11 | 74.16+/-1.52 | 85.51+/-0.95 | 32.37+/-2.39 | 60.08+/-5.67 | 71.52+/-2.67 | 11.63+/-0.67 | 17.13+/-1.12 | 25.84+/-1.02 |
| FedGEN | 63.50+/-3.27 | 69.42+/-4.09 | 80.17+/-4.71 | 27.21+/-3.12 | 57.16+/-2.71 | 68.93+/-1.75 | 10.07+/-0.19 | 15.26+/-0.29 | 21.49+/-0.17 |
| FedMA | 64.71+/-4.92 | 74.98+/-5.03 | 77.13+/-4.10 | 28.61+/-1.39 | 59.97+/-0.96 | 70.91+/-1.02 | 11.89+/-0.57 | 15.90+/-0.92 | 22.02+/-0.82 |
| GAMF | 64.97+/-3.93 | 75.21+/-4.05 | 77.34+/-3.78 | 28.92+/-1.52 | 60.23+/-1.93 | 71.44+/-1.75 | 11.98+/-0.99 | 16.76+/-0.77 | 24.15+/-0.49 |
| FedLAW | 60.34+/-4.39 | 73.93+/-4.91 | 77.53+/-3.52 | 26.32+/-2.80 | 46.81+/-3.61 | 61.08+/-2.61 | 11.57+/-1.61 | 15.99+/-0.49 | 22.37+/-0.68 |
| **FedCDA** | 66.30+/-0.07 | 76.59+/-0.25 | 78.99+/-0.13 | 34.97+/-0.31 | 62.81+/-0.28 | 72.04+/-0.23 | 12.20+/-0.13 | 19.98+/-0.25 | 28.16+/-0.30 |

Dirichlet:

| Method | FMNIST a=0.1 | a=0.3 | a=0.5 | CIFAR-10 a=0.1 | a=0.3 | a=0.5 | CIFAR-100 a=0.1 | a=0.3 | a=0.5 |
|---|---|---|---|---|---|---|---|---|---|
| FedAvg | 71.81+/-5.61 | 75.97+/-3.21 | 79.73+/-1.94 | 50.43+/-1.68 | 61.11+/-2.68 | 67.37+/-1.69 | 30.13+/-0.70 | 35.73+/-0.56 | 38.86+/-0.35 |
| FedProx | 70.44+/-3.87 | 72.17+/-4.10 | 75.24+/-2.19 | 38.98+/-5.91 | 61.64+/-1.92 | 70.16+/-2.03 | 32.96+/-1.18 | 40.81+/-0.41 | 42.53+/-0.48 |
| FedExP | 73.42+/-4.22 | 76.57+/-3.39 | 80.22+/-3.78 | 60.63+/-4.32 | 70.22+/-2.40 | 74.37+/-1.91 | 36.76+/-1.18 | 44.18+/-0.53 | 47.80+/-0.58 |
| FedSAM | 71.65+/-0.07 | 75.91+/-0.06 | 77.67+/-0.10 | 49.96+/-0.20 | 59.53+/-0.19 | 64.54+/-0.21 | 21.54+/-0.12 | 24.72+/-0.18 | 28.59+/-0.19 |
| FedDF | 80.03+/-1.04 | 84.42+/-0.62 | 86.84+/-1.93 | 54.28+/-2.39 | 69.85+/-5.67 | 73.76+/-2.67 | 34.76+/-0.67 | 39.42+/-1.12 | 42.31+/-1.02 |
| FedGEN | 73.02+/-1.87 | 77.48+/-3.50 | 81.76+/-4.21 | 47.09+/-3.12 | 64.90+/-2.71 | 68.74+/-1.75 | 29.02+/-0.19 | 38.54+/-0.29 | 40.81+/-0.17 |
| FedMA | 71.87+/-4.28 | 75.89+/-4.15 | 80.12+/-3.23 | 49.98+/-2.01 | 61.32+/-2.17 | 68.42+/-1.95 | 30.02+/-0.58 | 36.21+/-0.83 | 39.55+/-0.52 |
| GAMF | 72.11+/-5.16 | 76.24+/-3.67 | 80.55+/-2.06 | 51.21+/-1.37 | 63.45+/-1.03 | 70.14+/-1.81 | 31.12+/-0.69 | 37.26+/-0.78 | 41.25+/-0.74 |
| FedLAW | 71.93+/-8.23 | 76.88+/-2.80 | 79.98+/-1.09 | 48.91+/-3.59 | 61.50+/-2.29 | 67.08+/-1.75 | 32.01+/-2.61 | 38.80+/-2.20 | 40.11+/-1.17 |
| **FedCDA** | 78.63+/-0.14 | 84.67+/-0.12 | 87.01+/-0.08 | 62.46+/-0.22 | 70.27+/-0.29 | 74.96+/-0.17 | 39.38+/-0.25 | 45.86+/-0.22 | 49.31+/-0.22 |

Note: FedDF beats FedCDA on FMNIST S=8 (85.51 vs 78.99) and is close on FMNIST Dirichlet; the paper acknowledges FedCDA is not best on small data / simple CNN (p. 8).

### Table 2 (p. 8): CIFAR-100, ResNet-18, 100 clients, 10% participation, lr 0.1, wd 1e-3, E = 5, SGD no momentum

| Method | Dir 0.1 | Dir 0.3 | Dir 0.5 |
|---|---|---|---|
| FedAvg | 38.89+/-0.85 | 40.38+/-0.55 | 42.23+/-0.30 |
| FedProx | 39.86+/-0.45 | 39.48+/-0.37 | 40.18+/-0.46 |
| FedExP | 38.04+/-3.37 | 44.10+/-1.69 | 41.79+/-1.62 |
| FedSAM | 16.35+/-0.25 | 20.53+/-0.25 | 25.70+/-0.28 |
| FedDF | 41.04+/-0.57 | 47.06+/-0.74 | 47.63+/-0.53 |
| FedGEN | 39.91+/-1.72 | 41.65+/-1.35 | 43.39+/-1.08 |
| FedMA | 39.12+/-0.52 | 40.42+/-0.61 | 42.89+/-0.21 |
| GAMF | 39.89+/-0.67 | 40.98+/-0.32 | 43.25+/-0.34 |
| FedLAW | 40.88+/-0.66 | 41.77+/-0.78 | 41.89+/-0.33 |
| **FedCDA** | 47.38+/-0.23 | 49.96+/-0.21 | 50.04+/-0.19 |

Text claims: CIFAR-10 Dir 0.1: FedCDA 62.46 vs best baseline FedExP 60.63 (p. 8). Fig. 2(a): approximate/optimal strategies up to ~10% above FedAvg (p. 9).

### Checks to run first
- CPU: FMNIST CNN, Dirichlet 0.5 and 0.1, FedAvg vs FedCDA (targets 79.73 / 87.01 and 71.81 / 78.63). 200 rounds x 4 clients x 20 local epochs of ~3000 samples each: feasible on CPU but slow (~48k local epochs of ~50 batches).
- GPU: CIFAR-10 ResNet-18 Dir 0.1 (50.43 vs 62.46), Dir 0.5 (67.37 vs 74.96).

## 6. Not specified

- Appendix D hyperparameter sweep contents (looked in: repo PDF, ICLR proceedings PDF; OpenReview blocked).
- Code: none public (looked in: GitHub repo/code search, ICLR poster page, OpenReview forum page via fetch).
- ResNet-18 variant and normalization layer (looked in: Sec. 6.1).
- Data augmentation / normalization (looked in: Sec. 6.1).
- LR schedule (looked in: Sec. 6.1).
- Handling of unequal batch sizes (P = 4, B = 3), cold cache, never-sampled clients (looked in: Sec. 4, Alg. 1).
- Whether shards/Dirichlet partitions are reused across runs; seeds (looked in: Sec. 6.1).
- Per-baseline hyperparameters (FedProx mu, FedSAM rho, FedDF/FedGEN server data, etc.) (looked in: Sec. 6.1, 6.2).
- Whether momentum "1e-4" is a typo (looked in: Sec. 6.1 only; reproduce as printed, and optionally also with 0.9).
- Whether the eval accuracy is of the global model on the central test set (implied, looked in: Sec. 6.1).
