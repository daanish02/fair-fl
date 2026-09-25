# Median (Chen et al.) reproduction spec

Paper: Xiangyi Chen, Tiancong Chen, Haoran Sun, Zhiwei Steven Wu, Mingyi Hong, "Distributed Training with Heterogeneous Data: Bridging Median- and Mean-Based Algorithms", NeurIPS 2020.

Sources:
- NeurIPS main paper: `docs/impl-papers/Distributed Training with Heterogeneous Data - ....pdf` (11 pp). Page numbers below are its printed page numbers.
- NeurIPS supplement: scratchpad `median_supp.pdf`, Appendix I "Details of the Implementation" (p.27).
- Scratchpad `median_arxiv.pdf` is **arXiv 1906.01736 v2 (June 2019)**, an earlier version with different, MNIST-only experiments (Sec 5 pp.9-10, Appendix H p.29). Used only where it fills gaps, and flagged as such.

## 1. What the experiments test

- **All neural-network experiments use signSGD with majority vote, not medianSGD.** NeurIPS Sec 5 (p.8): "Since SIGNSGD is better studied empirically and MEDIANSGD is more of theoretical interest so far, we use SIGNSGD to demonstrate the benefit of injecting noise." The link is the paper's result that signSGD with majority vote = step along sign(median of local gradients) (Sec 1, p.2).
- Variants compared: signSGD full batch without noise; Noisy signSGD (Alg. 3, p.7): each node adds iid Gaussian noise b * xi (xi ~ N(0, I)) to its local gradient before taking the sign; signSGD with data sub-sampling (minibatch noise, no artificial noise); and on CIFAR-10, signSGD on homogeneous (iid) data as a reference.
- **The only medianSGD experiment is the 1-D toy in Fig. 1 (p.3)**: f(x) = (1/3) sum_{i=1..3} (x - a_i)^2 / 2, a = (1, 2, 10), step size 0.001, x_0 = 0.0005, three nodes each holding one function. Both signSGD (Fig. 1a) and medianSGD (Fig. 1b) drive the median gradient to 0 (x stuck near a_2 = 2) while the true (mean) gradient stays constant, |x - 13/3| = 7/3, about 2.33. No noisy medianSGD run is shown. The same construction is the counterexample in Sec 2 (supplement around line "algorithm will converge to x = a2 with any stepsize < 2/L").
- Noisy medianSGD (Alg. 4, p.7) exists only as theory: Theorem 6 (p.8), E||grad f||^2 averaged over T <= 2 D_f/(T delta) + O(d / b^2) + O(delta d b^2); best rate O(d^{2/3}/T^{1/3}) with b = T^{1/6} d^{1/6}, step delta = 1/(T^{2/3} d^{2/3}). Noise is added to each node's gradient before the coordinate-wise median; each coordinate of xi iid, symmetric, unimodal, mean 0, variance 1.

## 2. Configuration

### NeurIPS version (primary)

| Item | MNIST (Fig. 2, p.8) | CIFAR-10 (Fig. 3, p.9) | Source |
|---|---|---|---|
| Partition | "each node contains some exclusive data for one or two out of ten categories" | same | Sec 5 p.8 |
| Number of nodes | not specified (arXiv v2 Fig. 3, the same experiment with the same legend, says **10 machines**, full MNIST) | not specified ("up to 5 AWS p3.2xlarge machines" is hardware, not node count) | App. I p.27; arXiv v2 p.10 |
| Which classes per node | not specified beyond "one or two, exclusive". With 10 nodes and exclusive classes, one class per node is the natural reading | not specified | |
| Model | 2-layer fully connected net, 128 hidden then 10 outputs (activation not stated; assume ReLU, softmax output), truncated-normal fan-in init (Keras `he`/`lecun`-style `variance_scaling`) | ResNet-20 (Keras ResNet20 v1) with all BatchNorm layers removed, Keras default init | App. I.2 |
| Input | pixels scaled to [0,1], one-hot labels, categorical cross-entropy | same processing | App. I.1 |
| Gradient | **full batch** local gradient per node (text p.8 "full batch Noisy SIGNSGD"); sub-sampling variant uses BS = 128 | batch size not specified | Sec 5, Fig. 2 legend |
| Step size | constant, tuned from {1, 0.1, 0.01, 0.001} "based on training performance"; chosen value not stated (arXiv v2 used 0.1 for its full-MNIST figure) | initial lr from {1, 0.1, 0.01, 0.001}; divided by 2, 10, 20 after 1000, 3000, 5000 iterations (cumulative or relative to initial not stated; read as lr0/2, lr0/10, lr0/20) | App. I.3 |
| Noise b | b = 1e-3 and b = 1e-5 (Gaussian std) | b = 0.001 nominal, but per weight tensor W_i: **b_i = 0.1 * max_j Q_{i,j}**, Q_{i,j} = max absolute element of node j's stochastic gradient for W_i (one extra float per tensor per node) | Fig. 2 legend, Fig. 3 caption, App. I.3 |
| Iterations | 5 x 10^4 (x-axis) | about 7000 (x-axis) | Figs. 2, 3 |
| Seeds | not specified; single curves | not specified | |
| Update | x <- x - lr * sign(sum_i sign(g_i)) (majority vote), one step per iteration, no local steps | same | Alg. 1/3 p.2, p.7 |
| Framework | Python 3.6.4, MPI4Py 3.0.0, NumPy 1.14.2, TensorFlow 1.10.0 | same | App. I p.27 |

Note on the CIFAR-10 noise: the text (p.9 caption) says b = 0.001, while App. I.3 describes the adaptive per-tensor rule. Treat the adaptive rule as the implementation and 0.001 as nominal/unclear.

### arXiv v2 (2019) experiments, for reference

- Fig. 2 (arXiv p.10): small MNIST subset, **5 machines, 100 data points each**, exclusive 1-2 classes per node, lr 0.01; curves: no noise; b = 1e-5; b = 1e-9; sub-sampling BS = 10; BS = 1; b = 1e-5 + BS = 10; 5 x 10^4 iterations; plots gradient norm and cross-entropy loss (log scale, down to about 1e-5 for noisy variants).
- Fig. 3 (arXiv p.10): full MNIST, **10 machines**, lr 0.1, b in {1e-3, 1e-5}, sub-sampling BS = 128, 5 x 10^4 iterations. This is the same experiment as NeurIPS Fig. 2.

## 3. Claims and numbers

All claims are qualitative; the paper has no tables. Values below are read off the plots (approximate).

| Claim | Where | Readable values |
|---|---|---|
| medianSGD and signSGD stall on heterogeneous data: median gradient reaches 0, true gradient stays constant | Fig. 1 p.3 (toy) | mean gradient stays at about 2.33 (= 7/3) |
| Full-batch signSGD without noise gets stuck with a constant gradient norm and very poor accuracy | Fig. 2 p.8, text p.8 | no-noise: gradient norm about 10^1 flat; training loss about 18-20 barely falling; train and test accuracy rise slowly from about 0.1 to about 0.35-0.4 at 5 x 10^4 iterations |
| Noisy signSGD (b = 1e-3, 1e-5) and signSGD with sub-sampling (BS = 128) drive the gradient smaller and reach high accuracy | Fig. 2 p.8 | noisy / sub-sampled: train and test accuracy about 0.85-0.9 within the first 10^4 iterations |
| On CIFAR-10, noise speeds up convergence of signSGD on heterogeneous data; noisy hetero approaches homogeneous signSGD | Fig. 3 p.9 | all reach about 100% train accuracy by about 5000 iterations; hetero w/o noise reaches it last |
| Noise also improves generalisation | Fig. 3 p.9, text p.9 | test accuracy roughly 0.75 (hetero w/ noise, homo) vs about 0.70 (hetero w/o noise), read from plot |
| Expected median of noisy numbers approaches the mean at rate O(1/sigma); median distribution becomes symmetric at O(1/sigma^2) | Sec 3, Theorem 3-4 | theory only |

## 4. Recommendation for our testbed (model-update median with Gaussian noise sigma)

Goal: reproduce the medianSGD claim "coordinate-wise median stalls under label-exclusive heterogeneity; adding symmetric noise to each client's vector before the median closes the gap."

1. **Put the noise before the median, per client.** Alg. 4: g_i <- g_i + b * xi_i, then median over clients. Adding noise to the aggregated median (DP-on-output style) is a different mechanism and does not shrink the median-mean gap.
2. **Match the theory's regime: one local full-batch gradient step per round** (FedSGD-style). Then the client update is u_i = -lr * grad f_i(x), median(u_i + N(0, sigma^2)) = -lr * median(grad_i + N(0, (sigma/lr)^2)), so sigma_update = lr * b. With multiple local epochs the theory does not apply directly (report that as an extension, not a reproduction).
3. **Partition:** label-exclusive, e.g. MNIST 10 clients x 1 class (or 5 clients x 2 classes), all clients every round. Odd client count (2n+1) avoids median-of-two averaging; the theory assumes 2n+1 nodes, so prefer 5 x 2 classes, or 9 / 11 clients.
4. **Model / data:** 784-128-10 MLP, pixels in [0,1], cross-entropy, full-batch gradients, constant lr from {1, 0.1, 0.01, 0.001} (start with 0.1), 5 x 10^4 rounds is the paper's horizon; shorter is fine if curves have flattened.
5. **Sweep:** sigma_grad = b in {0, 1e-5, 1e-3, 1e-2, 1e-1} (gradient scale; convert to update scale by multiplying by lr), plus a minibatch-128 variant with b = 0 (sub-sampling noise), plus plain mean aggregation (FedSGD) as the target. Optionally the CIFAR-style adaptive b_i = 0.1 * max_j max|g_{i,j}| per tensor.
6. **Report:** norm of the true mean gradient (1/M) sum_i grad f_i(x) at each round (the quantity the theory bounds), median-mean gap ||median(g_i) - mean(g_i)||, training loss, train/test accuracy. Expected: b = 0 stalls at a non-zero mean-gradient norm and low accuracy; increasing b lowers the gap and raises accuracy until noise dominates (Theorem 6 trade-off d/b^2 vs delta d b^2).
7. **Sanity check first:** the Fig. 1 toy (a = 1, 2, 10; lr 0.001; x_0 = 0.0005; 3 clients). Median aggregation should stall at x about 2 with mean gradient about -2.33; with Gaussian noise b of order |a_3 - a_1| (e.g. b = 5 to 10) the expected median moves toward the mean and x should drift toward 13/3.
8. Since the paper's NN evidence is for signSGD, also run signSGD-majority-vote with the same noise as a direct replication, then present the medianSGD result as the analogue predicted by Theorem 6.

## 5. Unspecified

- Number of nodes for NeurIPS MNIST and CIFAR-10 runs (MNIST likely 10 per arXiv v2; CIFAR-10 unknown).
- Exact class assignment per node ("one or two" exclusive classes).
- Chosen lr for each figure in the NeurIPS version; CIFAR-10 batch size; whether lr decay factors are relative to the initial lr.
- Hidden activation of the MNIST MLP.
- Seeds / repeats; no error bars.
- Test-accuracy evaluation protocol (presumably the official test set with the global model).
- No medianSGD neural-network experiment; no numeric tables.
