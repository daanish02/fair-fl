# FCFL reproduction spec

Paper: Wang et al., "FCFL: A Fairness Compensation-Based Federated Learning Scheme with Accumulated Queues", ECML-PKDD 2024 (LNCS pp. 389-405). Source: `docs/impl-papers/FCFL - ....pdf` (17 pp); page numbers below are the printed LNCS page numbers.

Official code: github.com/wlffffff/FCFL (first commit 2024-06-12; later commits only touch dummy files). Files read: `main_fcfl.py`, `utils/options.py`, `utils/FCFL_options.py`, `utils/utils.py`, `models/Update.py`, `models/Nets.py`, `models/qFed.py`, `models/test.py`, `client_split_sample/sampling.py`, `client_split_sample/Dirichlet_split_datasets.py`, `client_split_sample/sampling_by_proportion.py`, `run.sh`, `save/result/*`. Only `main_fcfl.py` is released; the baseline mains (`main_fedavg.py`, `main_qfedavg.py`, `main_fedfa.py`, `main_gifair.py`) are referenced in `run.sh` but are not in the repo.

## 0. Verified base settings

| Item | Paper (Sec 4.1, pp.396-397) | Code | Verified |
|---|---|---|---|
| Clients | 100 (MNIST, CIFAR-10); 31 (Shakespeare) | `--num_users 100` | yes |
| Participation | 10% per round | `--frac 0.1`, m = 10 | yes |
| Local batch | 64 | `--local_bs 64` | yes |
| Local epochs | 1 | `--local_ep 1` | yes |
| Rounds | 2000 (MNIST, CIFAR-10), 500 (Shakespeare) | `--epochs` default 20, `run.sh` uses 5; must pass `--epochs 2000` | yes |
| Seeds | 5 runs, averaged | `--seed` default 1; results appended to `*_5_avg.txt` per run | yes |
| lr | not stated in paper | `--lr 0.1` | yes |
| Momentum | "the server's momentum factor is 0.5" | **client-side** `torch.optim.SGD(lr, momentum=0.5)` re-created every round; no server momentum anywhere | yes, conflict (see 6) |
| alpha (fairness weight in Eq. 2) | 0.3 for MNIST and CIFAR-10 FCFL rows, 0.1 for Shakespeare (Table 2) | `--alpha` default 0.5 | yes, must pass `--alpha 0.3` |
| Selection | top-m by Q_i(t) plus a random fraction r (p.394 Remark; r = 0.4 MNIST, 0.6 CIFAR-10 in Table 2) | `main_fcfl.py` line 154 calls `sample_by_Q_top` (pure top-m). `sample_by_Q_top_add_random(m, Q, ratio)` exists in `utils/FCFL_options.py` but is unused; its `ratio` is the **top-Q** fraction (`--ratio` default 0.7), so paper r maps to `ratio = 1 - r` | yes |
| uf queue input | Acc_i^t = accuracy of global model on client i's local data | line 115-118: global model's accuracy on each client's local **testloader** (fraction in [0,1]) | yes |
| Acc_global estimate | weighted mean of selected clients' local training accuracy | `acc_global_estimate(local_accs_trainloader, aggregate_p)`: accuracy of each **locally updated** model on its own trainloader, weighted by the aggregation weights | yes |

## 1. Per-client split in code

`LocalUpdate.train_val_test` (`models/Update.py` lines 35-52): **80 / 10 / 10 train / validation / test**, not the paper's 8:2 (p.396 "We randomly divide the data for each local client into 8 : 2 for training and testing").
- Split is a plain slice of `list(idxs)` with no shuffle: first 80% train, next 10% validation (never used anywhere), last 10% test.
- Order of `idxs`:
  - Shards (`mnist_noniid` / `cifar_noniid`, returns np.array): concatenation of the two shards in Python-set iteration order, each shard contiguous in label-sorted order. So the client's local test set (last 10%, 60 MNIST or 50 CIFAR images) comes entirely from the second shard, typically a single label. Per-client "test accuracy" is therefore accuracy on one class for most clients.
  - Dirichlet (`split_noniid`, returns a Python `set`): order is set-iteration order of integer indices (roughly ascending mod hash-table size), which is effectively a label-mixed order because MNIST/CIFAR training indices are not label-sorted.
- All per-client data (train, val, test) comes from the official **training** set; the official test set is only used for the single global "Accuracy" number.
- Test batch size for local loaders: len/10.
- Transforms: MNIST Normalize((0.1307,), (0.3081,)); CIFAR-10 Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)); no augmentation.

## 2. Partitions: shards vs Dirichlet

Paper p.396 lists three settings: (1) 200 label-sorted shards, 2 per client; (2) "Dirichlet function to set different levels of non-IID"; (3) Shakespeare, 31 roles.
- **Dirichlet alpha values: not specified** (looked in: Sec 3.1 p.391, Sec 4.1 p.396, Table 2 caption, Fig. 2/4/5 captions). No experiment in the paper reports results per alpha.
- Which setting each table uses: **not specified**. Table 2 / Table 4 captions do not say. The code default is `--dirichlet 0.0`, which selects the **shard** split (main_fcfl.py lines 49-57, 66-74), so shards are the most likely setting for Tables 2 and 4.
- The only stated Dirichlet use is the motivation figure Fig. 2 (p.392): MNIST, CNN, 20 clients, 2 per round, alpha not given.
- Code hints: `sampling_by_proportion.py` and `Dirichlet_split_datasets.py __main__` use alpha = 1.0 (10 clients); these are helper scripts, not the experiment entry point.
- Dirichlet implementation (`split_noniid`): per class, draw proportions ~ Dir(alpha * 1_N) over N clients, split that class's indices by cumulative proportions. No minimum-size guarantee.
- Shard sizes: MNIST 200 x 300 images, 2 shards per client (600 images, 480 train). CIFAR-10 200 x 250, 2 per client (500 images, 400 train).

## 3. Models

- MNIST: `CNNMnist` (paper p.396 "two convolutions and maximum pooling"): conv(1->10, 5x5) - maxpool2 - ReLU - conv(10->20, 5x5) - Dropout2d - maxpool2 - ReLU - FC(320->50) - ReLU - dropout(0.5) - FC(50->10). Run with `--model cnn --dataset mnist`.
- CIFAR-10: `MLP` with **one hidden layer of 64** (main_fcfl.py line 99 hard-codes `dim_hidden=64`): flatten 3072 - Linear(3072, 64) - Dropout(0.5) - ReLU - Linear(64, 10). Paper p.396 "an MLP which contains a hidden layer on CIFAR-10" (width not stated in paper). Run with `--model mlp --dataset cifar`.
- CIFAR-10 lr: same `--lr 0.1` default (run.sh gives no override; `run.sh` shows `--lr 0.8` only for FedFa on Shakespeare). Paper does not state lr for any dataset.
- Loss: CrossEntropy. Optimiser: SGD, momentum 0.5, no weight decay.
- Table 3 (running time) also mentions MNIST(MLP) and CIFAR-10(CNN) settings (`CNNCifar` = LeNet-5 style); those are timing only.

## 4. Metrics (code: `main_fcfl.py` lines 193-226, `utils/utils.py`)

After the last round, with the final global model in eval mode:
- **Accuracy** (Table 2 "Accuracy" column): `test_img(net_glob, dataset_test)` = accuracy in % on the **official global test set** (10,000 images). It is NOT the mean over clients.
- Per-client accuracy: for all 100 clients, accuracy of the final global model on that client's local testloader (the last 10% slice of its training-set indices, see 1), times 100.
- **Variance**: `np.var(all_user_acc)` = population variance (ddof = 0) of the 100 per-client accuracies in percent, units %^2.
- **Worst 10%**: mean of the lowest `int(100 * 0.1)` = 10 per-client accuracies (%).
- **Best 10%**: mean of the highest 10 per-client accuracies (%).
- Reported value: average over 5 seeds (paper p.397). The code appends one block per run to `Result_FCFL_5_avg.txt`; averaging is done by hand.
- Paper p.397: for each baseline, "we take the best performance of each method" over its hyperparameter grid.

## 5. Target numbers

### Table 2 (p.398): "Statistics of the test accuracy distribution". Accuracy, Best 10%, Worst 10% in %; Variance in %^2

| Dataset | Method | Accuracy | Best 10% | Worst 10% | Variance |
|---|---|---|---|---|---|
| MNIST | FedAvg | 95.96 | 100.00 | 87.47 | 15.06 |
| MNIST | q-FedAvg, q = 0.2 | 96.09 | 100.00 | 88.40 | 12.58 |
| MNIST | FedFa, alpha = 0.6, beta = 0.4 | 96.23 | 100.00 | 88.37 | 12.61 |
| MNIST | GIFAIR, lambda = 0.5 | 96.12 | 100.00 | 88.63 | 12.72 |
| MNIST | FCFL, alpha = 0.3, r = 0.4 | 96.06 | 100.00 | 89.17 | 11.03 |
| CIFAR-10 | FedAvg | 46.35 | 68.67 | 20.16 | 178.93 |
| CIFAR-10 | q-FedAvg, q = 2.0 | 47.14 | 66.81 | 24.98 | 149.50 |
| CIFAR-10 | FedFa, alpha = 0.6, beta = 0.4 | 46.64 | 68.10 | 23.19 | 164.68 |
| CIFAR-10 | GIFAIR, lambda = 0.5 | 46.61 | 67.02 | 23.18 | 158.36 |
| CIFAR-10 | FCFL, alpha = 0.3, r = 0.6 | 46.12 | 65.39 | 28.03 | 114.59 |
| Shakespeare | FedAvg | 49.16 | 70.65 | 35.34 | 89.87 |
| Shakespeare | q-FedAvg, q = 2.0 | 50.24 | 69.77 | 37.92 | 75.99 |
| Shakespeare | FedFa, alpha = 0.5, beta = 0.5 | 49.03 | 69.06 | 36.23 | 79.54 |
| Shakespeare | GIFAIR, lambda = 0.3 | 50.01 | 68.50 | 36.24 | 78.25 |
| Shakespeare | FCFL, alpha = 0.1, r = 0.6 | 50.55 | 68.74 | 38.55 | 67.48 |

Baseline grids (p.397): q-FedAvg q in {0.1, 0.2, 0.5, 1.0, 2.0, 5.0}; FedFa (accuracy weight, frequency weight) in {(0.4,0.6), (0.5,0.5), (0.6,0.4)}; GIFAIR lambda in {0.3, 0.5, 0.7}; best per method reported. The FedFa alpha/beta in Table 2 are FedFa's own weights, not FCFL's alpha.

### Table 4 (p.400): ablation on CIFAR-10 (same metrics)

| Method | Accuracy | Best 10% | Worst 10% | Variance |
|---|---|---|---|---|
| FCFL | 46.12 | 65.39 | 28.03 | 114.59 |
| FCFL-RS (random selection, unfairness reweighting) | 46.06 | 65.60 | 22.91 | 151.31 |
| FCFL-DAR (unfairness selection, data-amount reweighting) | 43.56 | 61.00 | 26.57 | 96.31 |

### Table 3 (p.398): seconds per communication round (hardware not stated)

| Method | MNIST(MLP) | MNIST(CNN) | CIFAR-10(MLP) | CIFAR-10(CNN) |
|---|---|---|---|---|
| FedAvg | 0.87 | 1.91 | 0.89 | 1.84 |
| q-FedAvg | 0.95 | 1.99 | 0.96 | 1.90 |
| FedFa | 1.03 | 2.01 | 0.96 | 1.91 |
| GIFAIR | 0.88 | 1.90 | 0.88 | 1.85 |
| FCFL | 0.94 | 2.01 | 0.96 | 1.95 |

Figures: Fig. 4 (p.399) training loss and test accuracy curves on MNIST / CIFAR-10 (FCFL converges after about 1000 rounds on MNIST); Fig. 5 (p.400) accuracy and variance vs r in [0, 1] (MNIST flat; CIFAR-10 both rise with r; r = 0.6 chosen by grid search). No numeric values in text.

Numbers to check first: MNIST FedAvg 95.96 / var 15.06 vs FCFL 96.06 / var 11.03 / worst 89.17; CIFAR-10 FedAvg var 178.93 vs FCFL 114.59, worst 20.16 vs 28.03.

## 6. Paper vs code conflicts and gotchas

1. Split: paper 8:2 train/test; code 80/10/10 with an unused validation slice, un-shuffled, so shard-split clients have single-class local test sets.
2. Momentum: paper says server momentum 0.5; code has client SGD momentum 0.5 (fresh optimiser each round, so momentum spans only one local epoch) and plain weighted averaging at the server.
3. Selection: paper uses a random fraction r (0.4 / 0.6); released `main_fcfl.py` uses pure top-m (r = 0). To match the paper, use `sample_by_Q_top_add_random(m, Q, ratio = 1 - r)`.
4. alpha: code default 0.5; paper Table 2 uses 0.3 (MNIST, CIFAR-10).
5. Round-0 and all-zero-Q aggregation weights (lines 129 and 160): `local_data_volume = [len(dict_users[cid]) for cid in range(len(idxs_users))]` uses clients 0..m-1, not the selected clients. Harmless under shards (all clients have equal size, so weights = 1/m), wrong under Dirichlet. Paper Eq. 3 intends n_i of the selected clients.
6. Accuracy column is global-test-set accuracy, not the mean of per-client accuracy; variance / worst / best use per-client local test slices drawn from the training set.
7. Eq. 2 uses the previous round's aggregation weight as the penalty (`aggregate_p_all_clients`, zero for unselected clients); uf uses Acc_global from the previous round's aggregation (a weighted train accuracy of local models, so it is optimistic and uf is often positive).
8. Units: code's Acc_global and local accuracies are fractions in [0,1] during training, so uf is in [0,1] and alpha * uf is compared against weights p in [0,1]. Metrics are reported in %.
9. Dirichlet alpha and which partition each table used: unspecified; code default is shards.
10. Baseline implementations are not released; hyperparameter grids only in text (p.397). Paper says baselines were rewritten "based on the code provided by their authors".
11. Seeds: `np.random.seed(args.seed)` only before partitioning; torch seed is never set, so runs are not bit-reproducible.
12. The committed `save/result/*_5_avg.txt` files are short debug runs (e.g. FCFL accuracy 69.4 / 18.8), not the paper's results.
