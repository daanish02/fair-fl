# FedMut reproduction spec

Paper: M. Hu, Y. Cao, A. Li, Z. Li, C. Liu, T. Li, M. Chen, Y. Liu, "FedMut: Generalized Federated Learning via Stochastic Mutation", AAAI 2024, pp. 12528-12537.
Sources: `docs/impl-papers/FedMut - ...pdf` (page numbers below are the printed AAAI page numbers, 12528-12537); official code github.com/HMHelloWorld/FedMut (files read: README.md, main_fed.py, utils/options.py, Algorithm/Training_FedMut.py, models/Nets.py, models/resnetcifar.py, models/Update.py, models/Fed.py, models/test.py, utils/get_dataset.py, utils/dataset_utils.py, utils/sampling.py, utils/utils.py, utils/mydata.py, utils/set_seed.py, plot_figure.py).

## 1. FL setup and training hyperparameters

| Item | Paper (Settings, p. 12533-12534) | Code default (utils/options.py) | Use |
|---|---|---|---|
| Clients | not stated in paper | `--num_users 100` | 100 |
| Participation | not stated ("K activated clients", Alg. 1) | `--frac 0.1` -> 10 per round | 10% |
| Optimizer | SGD | `sgd` | SGD |
| LR | 0.01 | 0.01 | 0.01, constant (no schedule in code) |
| Momentum | 0.9 | **0.5** | **0.5 (code)**; momentum buffer is re-created every round (new optimizer per local train). Changed from 0.9 on 2026-09-26: with 0.9 every FedMut run went NaN (see lab notes) |
| Weight decay | not stated | none (SGD default 0) | 0 |
| Local batch size | 50 | 50 | 50 |
| Local epochs | 5 | 5 | 5 |
| Rounds | not stated; Fig. 5/6 x-axis ends at 1000 | `--epochs 2000` | 1000 (matches figures) |
| Test batch | - | 128 | - |
| Seed | not stated | `--seed 1` (random, numpy, torch) | 1 |
| Loss | - | CrossEntropy | - |

- Datasets (p. 12533): CIFAR-10, CIFAR-100, Shakespeare (LEAF non-IID split, LSTM; Table 2).
- **CIFAR-100 in the code is the 20-class coarse-label version** (`utils/mydata.py: CIFAR100_coarse`, uses `coarse_labels`; README: "Set 20 for CIFAR-100"). The paper does not say this. Use 20 superclasses to match Table 1 numbers.
- Preprocessing (utils/get_dataset.py): no augmentation. CIFAR-10: ToTensor + Normalize(0.5,0.5,0.5 / 0.5,0.5,0.5). CIFAR-100 coarse: Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010)).
- Test set: the standard central test split (10k images), evaluated on the global model (models/test.py `test_img`).
- Heterogeneity: Dirichlet d in {0.1, 0.5, 1.0} plus IID (p. 12533).
  - Dirichlet code (utils/dataset_utils.py `separate_data`): for each class k, draw p ~ Dir(d * 1_N) over N clients; clients already holding >= N_train/N samples get p = 0 for that class (renormalize); split class-k indices by cumulative proportions; repeat the whole thing until every client has >= 10 samples. Final per-client index lists shuffled.
  - IID: `iid()` in utils/sampling.py (equal random split).
  - **The exact partitions used are shipped in the repo** as JSON: `data/cifar10_100_noniidCase5_beta{0.1,0.5,1.0}.json`, `data/cifar10_100_iid.json`, `data/cifar100_100_noniidCase5_beta{0.1,0.5,1.0}.json`, `data/cifar100_100_iid.json` (format: `{"train_data": {"<client>": [indices...]}}`). Loaded with `--generate_data 0 --iid 0 --noniid_case 5 --data_beta <d>`. Importing these gives identical client splits.

Paper-equivalent command (inferred):
`python main_fed.py --algorithm FedMut --dataset cifar10 --num_classes 10 --model resnet18 --iid 0 --noniid_case 5 --data_beta 0.1 --momentum 0.9 --epochs 1000 --radius 4.0 --mut_acc_rate 0.5 --mut_bound 50`

## 2. Models (exact code)

`main_fed.py` model switch: `cnn` -> `CNNCifar`, `resnet18` -> `ResNet18_cifar10`, `vgg` -> `VGG16`. All forward() return a dict; logits are `['output']`.

**CNN (`models/Nets.py: CNNCifar`)**, LeNet-5 style, no normalization layers:
```
conv1 = Conv2d(3, 6, 5); pool = MaxPool2d(2,2); conv2 = Conv2d(6, 16, 5)
fc1 = Linear(16*5*5, 120); fc2 = Linear(120, 84); fc3 = Linear(84, num_classes)
x = pool(relu(conv1(x))); x = pool(relu(conv2(x))); flatten; relu(fc1); relu(fc2); fc3
```
(Paper cites it as "CNN (McMahan et al. 2017)", p. 12533, but the code is this 6/16-channel LeNet.) CPU-feasible.

**ResNet-18 (`models/resnetcifar.py: ResNet18_cifar10` = `ResNetCifar10(BasicBlock, [2,2,2,2])`)**: CIFAR stem (conv 3x3, 64, stride 1, pad 1, no bias) + BatchNorm2d + ReLU, **no max-pool**; layers 64/128/256/512 with strides 1/2/2/2; BasicBlock conv3x3-BN-ReLU-conv3x3-BN + identity/1x1-conv-BN downsample; AdaptiveAvgPool(1) -> Linear(512, num_classes). Kaiming-normal (fan_out) conv init, BN weight 1 / bias 0, `zero_init_residual=False`. **Uses BatchNorm.**

**VGG-16 (`models/Nets.py: VGG16`)**: 13 conv3x3 (pad 1) layers each followed by **BatchNorm2d** + ReLU: [64,64,M,128,128,M,256,256,256,M,512,512,512,M,512,512,512,M], then AvgPool2d(1) (no-op); classifier Linear(512,4096)-ReLU-Dropout(0.5)-Linear(4096,4096)-ReLU-Dropout(0.5); fc Linear(4096, num_classes). About 33.6M params. **Uses BatchNorm.**

**BatchNorm -> GroupNorm impact (our testbed):**
- ResNet-18 and VGG-16 use BN; the CNN does not (no change for CNN rows, so CNN rows are the cleanest comparison).
- FedAvg / FedMut in the code aggregate the whole `state_dict`, including BN `running_mean`, `running_var`, `num_batches_tracked`. FedMut also **mutates** those buffers (`w_glob[k] + alpha * ctrl * (w_glob[k] - w_old[k])` for every key). With GN there are no running stats, so this part of the mutation disappears.
- Under strong non-IID (d = 0.1), BN running stats averaged across skewed clients are a known source of FedAvg degradation; GN usually helps FedAvg there, so the FedMut - FedAvg gap on ResNet/VGG may shrink. IID absolute accuracy with GN is usually a few points below BN. Compare trends and gaps, not absolute numbers, for ResNet/VGG rows; the CNN rows should match directly.

## 3. FedMut hyperparameters

Mutation (Algorithm/Training_FedMut.py `mutation_spread`, paper Eq. 3-5, p. 12532-12533):
- `w_delta = w_glob(t) - w_glob(t-1)` (global update of this round).
- For each layer (state_dict key) separately: build a list of m = K coefficients in pairs, each pair = {+1, -1 + beta_t} in random order, then shuffle the list; client j gets `w_glob + alpha * coef[layer][j] * w_delta` for every layer. If K is odd, the last client gets the unmutated global.
- `beta_t = mut_acc_rate * (1 - min(t / mut_bound, 1))` (paper Eq. 5: beta_t = max(beta_0 (1 - t/T_b), 0)).
- The mutated models are kept in `w_locals[i]` and the i-th sampled client of the **next** round starts from mutated model i (clients do not get the plain global model).
- Global aggregation for FedMut: **uniform mean** of the K local models (`Aggregation(w_locals, None)`); FedAvg/FedProx in main_fed.py use sample-count weights.
- `delta_rank` (norm of w_delta) is computed but only printed; the adaptive-alpha line is commented out, so alpha is constant.

| Param | Paper Table 1 (p. 12534) | Code default | README |
|---|---|---|---|
| alpha (`--radius`) | 4.0 for all cells | 4.0 | "range of mutation", no value |
| beta_0 (`--mut_acc_rate`) | 0.3 for CNN and VGG-16; 0.5 for ResNet-18 | 0.3 | "acceleration rate", no value |
| T_b (`--mut_bound`) | not stated | 50 | not mentioned |
| `--min_radius` | - | 0.1 (unused) | - |

Baselines: FedProx mu = 0.01 (paper p. 12534; code `--prox_alpha 0.01`); FedGen as in Zhu et al. 2021; CluSamp with gradient-similarity clustering (code `--sim_type L1`).

Ablations (Fig. 6, p. 12535): alpha in {1,...,5} (alpha = 5 fails to train), beta_0 in {0.0, 0.1, 0.3, 0.5, 0.7}; setting (model/dataset) of Fig. 6 not stated, y-axis to 70% suggests ResNet-18 CIFAR-10.

## 4. How test accuracy is reported

- Paper: "Test accuracy (%)" as mean +/- std (Table 1); **the statistic is not described** (looked in: Settings, Table 1 caption, Comparison Evaluation, p. 12533-12534). Number of runs not stated.
- Code: FedMut evaluates the global model on the full test set **every round** and saves the list; FedAvg in main_fed.py evaluates **every 10 rounds** (`iter % 10 == 9`); FedProx every round. No script in the repo computes mean/std (plot_figure.py only plots curves).
- Most plausible reading: mean +/- std over the final several evaluated rounds of a single run (std values like 0.05-0.3 on IID are consistent with a last-rounds window, not multiple seeds). Recommended: report mean +/- std over the last 10 evaluations of the global model (and also the last-round value).

## 5. Target numbers

### Table 1 (p. 12534): test accuracy (%), alpha = 4.0, beta_0 = 0.3 (CNN, VGG-16) / 0.5 (ResNet-18)

| Model | Dataset | Het. | FedAvg | FedProx | FedGen | CluSamp | FedMut |
|---|---|---|---|---|---|---|---|
| CNN | CIFAR-10 | d=0.1 | 47.93+/-3.26 | 48.21+/-3.35 | 47.57+/-2.64 | 47.69+/-1.30 | 51.25+/-1.07 |
| CNN | CIFAR-10 | d=0.5 | 54.33+/-0.50 | 54.43+/-0.94 | 53.86+/-1.03 | 55.14+/-0.98 | 56.90+/-0.45 |
| CNN | CIFAR-10 | d=1.0 | 57.00+/-0.58 | 57.00+/-0.45 | 55.85+/-0.49 | 55.91+/-0.82 | 58.90+/-0.59 |
| CNN | CIFAR-10 | IID | 57.32+/-0.28 | 57.58+/-0.21 | 57.33+/-0.21 | 58.06+/-0.19 | 59.70+/-0.20 |
| CNN | CIFAR-100 | d=0.1 | 28.98+/-0.91 | 29.18+/-0.85 | 28.17+/-1.00 | 28.81+/-0.66 | 31.39+/-0.40 |
| CNN | CIFAR-100 | d=0.5 | 32.62+/-0.73 | 32.54+/-0.78 | 32.21+/-0.39 | 32.20+/-0.45 | 34.52+/-0.79 |
| CNN | CIFAR-100 | d=1.0 | 32.77+/-0.53 | 32.98+/-0.57 | 31.71+/-0.31 | 32.32+/-0.36 | 34.80+/-0.39 |
| CNN | CIFAR-100 | IID | 32.37+/-0.30 | 32.46+/-0.26 | 32.20+/-0.28 | 32.05+/-0.21 | 34.51+/-0.25 |
| ResNet-18 | CIFAR-10 | d=0.1 | 44.41+/-3.13 | 43.52+/-3.13 | 44.37+/-1.88 | 43.17+/-2.82 | 55.43+/-1.95 |
| ResNet-18 | CIFAR-10 | d=0.5 | 61.75+/-0.35 | 61.23+/-0.62 | 60.69+/-0.45 | 61.45+/-0.40 | 67.89+/-0.95 |
| ResNet-18 | CIFAR-10 | d=1.0 | 65.88+/-0.20 | 65.89+/-0.38 | 65.09+/-0.34 | 65.93+/-0.26 | 70.01+/-0.18 |
| ResNet-18 | CIFAR-10 | IID | 64.08+/-0.13 | 64.10+/-0.10 | 63.84+/-0.41 | 64.29+/-0.18 | 70.63+/-0.13 |
| ResNet-18 | CIFAR-100 | d=0.1 | 34.14+/-0.76 | 34.21+/-0.67 | 34.27+/-0.48 | 33.11+/-0.65 | 37.91+/-0.35 |
| ResNet-18 | CIFAR-100 | d=0.5 | 40.72+/-0.54 | 41.52+/-0.37 | 41.33+/-0.45 | 41.63+/-0.38 | 46.66+/-0.23 |
| ResNet-18 | CIFAR-100 | d=1.0 | 42.85+/-0.29 | 43.60+/-0.34 | 42.21+/-0.16 | 43.06+/-0.33 | 47.64+/-0.20 |
| ResNet-18 | CIFAR-100 | IID | 42.58+/-0.31 | 42.41+/-0.20 | 42.13+/-0.26 | 42.30+/-0.34 | 48.18+/-0.10 |
| VGG-16 | CIFAR-10 | d=0.1 | 56.14+/-13.61 | 56.60+/-6.77 | 64.84+/-3.66 | 60.55+/-6.63 | 68.23+/-2.29 |
| VGG-16 | CIFAR-10 | d=0.5 | 77.91+/-0.35 | 77.80+/-0.31 | 78.00+/-0.17 | 77.68+/-0.17 | 80.03+/-0.51 |
| VGG-16 | CIFAR-10 | d=1.0 | 79.17+/-0.23 | 79.45+/-0.19 | 79.03+/-0.62 | 79.67+/-0.67 | 81.09+/-0.38 |
| VGG-16 | CIFAR-10 | IID | 80.20+/-0.05 | 79.98+/-0.10 | 80.00+/-0.06 | 80.29+/-0.07 | 81.70+/-0.06 |
| VGG-16 | CIFAR-100 | d=0.1 | 46.84+/-1.08 | 45.63+/-1.67 | 47.36+/-2.84 | 46.82+/-0.85 | 50.23+/-0.65 |
| VGG-16 | CIFAR-100 | d=0.5 | 54.09+/-0.83 | 54.80+/-0.66 | 54.59+/-0.73 | 54.36+/-0.74 | 57.28+/-0.53 |
| VGG-16 | CIFAR-100 | d=1.0 | 55.62+/-0.25 | 55.52+/-0.90 | 55.25+/-0.65 | 55.89+/-0.55 | 57.93+/-0.36 |
| VGG-16 | CIFAR-100 | IID | 57.42+/-0.09 | 56.74+/-0.16 | 55.98+/-0.32 | 56.64+/-0.25 | 58.09+/-0.21 |

(Values transcribed from the rendered PDF page image and cross-checked against the text extract, which lists the same rows in the same order.) CIFAR-100 = 20 coarse classes in the code.

Headline claim: up to 11.02% over FedAvg on CIFAR-10 ResNet-18 d = 0.1 (55.43 vs 44.41; p. 12534).

### Table 2 (p. 12534): Shakespeare, LSTM
FedAvg 52.08, FedProx 52.53, FedGen 52.69, CluSamp 49.74, FedMut 55.53.

### Figures
Fig. 5 (p. 12535): learning curves, CIFAR-10 ResNet-18, d = 0.1/0.5/1.0/IID, 1000 rounds. Fig. 6: ablations for alpha and beta_0 (curves only).

### Checks to run first
- CPU: CNN, CIFAR-10, d = 0.1 and IID, FedAvg vs FedMut (47.93 -> 51.25; 57.32 -> 59.70). 1000 rounds x 10 clients x 5 epochs x ~500 samples on the LeNet CNN is CPU-feasible.
- GPU: ResNet-18 CIFAR-10 d = 0.1 (44.41 -> 55.43) and IID (64.08 -> 70.63).

## 6. Not specified / discrepancies

- Number of clients and participation rate: not in paper (looked in: Settings p. 12533-12534, Alg. 1 p. 12532); taken from code defaults (100, 0.1).
- Number of rounds: not in paper (looked in: Settings, Table 1 caption); figures show 1000; code default 2000.
- Momentum: paper 0.9 vs code default 0.5.
- mut_bound T_b: not in paper (looked in: Eq. 5 text, Table 1 text); code default 50.
- Accuracy statistic (+/- over what), number of runs/seeds: not specified (looked in: p. 12533-12535, code: no aggregation script).
- CIFAR-100 uses 20 coarse labels in code; paper silent.
- Weight decay, augmentation: none in code; paper silent.
- The paper's "CNN (McMahan et al. 2017)" is the LeNet-style 6/16-channel CNNCifar in code, not the 32/64-channel McMahan CNN.
