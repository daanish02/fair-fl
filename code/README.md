# fairfl

`fairfl` is a small federated learning (FL) testbed for studying **non-Markovian group fairness**: fairness judged over the whole sequence of models deployed during training, not only the final one (after Alamdari et al., *Remembering to Be Fair*, ICML 2024). It re-implements a set of fair and robust FL baselines behind one interface. After every round it evaluates every client per sensitive group, so each run can be scored under the long-term, periodic, anytime and bounded fairness scopes afterwards, from its logs alone. It is built with PyTorch and NumPy and does not depend on Flower.

> **Research-grade code.** The simulation runs sequentially in one process (CPU or one GPU), with no multiprocessing, networking, fault tolerance or real secure aggregation. It is not meant for production.

## Setup

Requires [uv](https://docs.astral.sh/uv/). Python 3.12 is pinned in `.python-version`.

```bash
cd code
uv sync
```

### Datasets (`data.name`)

Everything is cached under `data.root` (default `data/`, gitignored). Train and test sets are merged; each client splits its own share into train/test.

| Name | Source | Label / sensitive attribute |
|---|---|---|
| `adult` | `sklearn.fetch_openml` | income >50K / sex (male = 1) |
| `mnist`, `fmnist` | torchvision (auto download) | digit, clothing class / none |
| `cifar10`, `cifar100` | torchvision (auto download), per-channel normalisation | class / none |
| `celeba` | `root/celeba/` as torchvision lays it out: `list_attr_celeba.txt` and `img_align_celeba/*.jpg`. torchvision's Google Drive download is tried but often fails, so download manually. Images are centre-cropped and resized to 64x64 once and cached as `celeba64_uint8.pt`. | `data.target_attr` (default `Smiling`) / `data.sensitive_attr` (default `Male`). `data.subsample` keeps N random images (the full set is ~10 GB as float32). |
| `bank`, `default`, `law`, `kdd` | `root/<name>.csv`, copied from `datasets/` in the FairWeight authors' archive (`bank-full.csv` becomes `bank.csv`). Not downloaded automatically. | Preprocessed as the authors' `load_data_utilities.py`: label-encoded categoricals, standardised numerics. bank: y / marital (single = 1); default: y / SEX (male = 1); law: y / sex (both flipped, as in the authors' code); kdd: class / sex (male = 1). |
| `synthetic` | generated | biased binary task / binary group |

`data.augment: true` applies a random crop (zero padding 4) and horizontal flip to each training batch on the device (image data only; off by default).

### Devices and long runs

`device` (top level of the config: `auto`, `cpu` or `cuda`) selects where the model lives; `auto` picks CUDA when available. Client data is moved to the GPU once at start-up if it fits in half the free GPU memory, otherwise it stays on the CPU and each batch is moved when used. Batch orders and noise come from per-client CPU generators, so a seed gives the same draws on any device. Training is fp32 (no AMP). Aggregation runs on the device. Post-processing (LoGoFair) gets CPU copies of the client data, and `predict_proba` always returns CPU probabilities.

## Running experiments

```bash
uv run fairfl run configs/fedavg_adult.yaml                  # one run from a YAML config
uv run fairfl run configs/fedavg_adult.yaml --set train.rounds=5 --set strategy.name=fcfl
uv run fairfl run configs/fedavg_adult.yaml --seed 3         # override the config seed
uv run fairfl run configs/fedavg_adult.yaml --out runs/tmp --quiet
uv run fairfl strategies                                     # list registered methods
```

- `--set key=value` takes a dotted path into the config. The value is parsed as YAML, so `--set strategy.params={eta: 2.0}` also works. You can repeat it.
- The pydantic model `ExperimentConfig` (`src/fairfl/core/config.py`) validates every config. It has these sections:
  - `device`: `auto` (default), `cpu` or `cuda`.
  - `data`: `name` (see Datasets); the Dirichlet `alpha`; `partition_on` = `label`, `sensitive`, `joint`, `iid` or `shards`; `shards_per_client` (default 2: sort by label, cut into N x s shards, give each client s random shards, as in McMahan et al.); `val_fraction`; `augment`; CelebA's `target_attr`, `sensitive_attr`, `subsample`.
  - `model`: `mlp`, `cnn` or `logreg`; image models `cnn_fmnist` (FedCDA: conv5x5 32/64 with pooling, fc 512), `cnn_cifar` (FedMut repo's LeNet-style CNNCifar), `resnet18` (CIFAR stem: 3x3 conv, no max-pool), `vgg16` (FedMut's layout), `cnn_celeba` (4 x conv3x3(32)+pool, for 64x64). ResNet-18 and VGG-16 use GroupNorm(2) instead of BatchNorm, because the flat parameter vector does not carry BatchNorm's running-statistic buffers.
  - `train`: `rounds`, `clients_per_round`, `local_epochs`, `batch_size`, `lr`; `momentum` and `weight_decay` for client SGD (default 0; the optimiser is rebuilt every round); `server_momentum` (default 0), FedAvgM momentum applied by the engine to the global update returned by `strategy.aggregate`; `eval_every` (default 1).
    - `eval_every = k` evaluates and logs only rounds where (round + 1) is a multiple of k, plus the final round and the post-processed model. Unlogged rounds get no `RoundLog` and no `on_round_end` call. The fairness scopes treat the logged rounds as the deployed history, so k > 1 coarsens every scope (like `deploy_every`).
  - `strategy`: `name` plus `params`.
  - `fairness`: the scheme, described below.
  - `scenarios`.

Example configs are in `configs/`:
- `fedavg_adult.yaml`: 10 clients, 50 rounds, split on the sensitive attribute;
- `fedavg_mnist.yaml`: CNN, label split, client-level fairness;
- `fedavg_synthetic.yaml`: a small biased tabular task for quick checks.

## Tests

```bash
uv run python -m pytest
```

On this Windows machine, application control blocks the `pytest.exe` shim, so run pytest as a module (`python -m pytest`) rather than `uv run pytest`.

## Architecture

```
src/fairfl/
  cli.py                typer CLI (run, strategies)
  core/                 types.py (pydantic messages), config.py, engine.py (Simulator), registry.py, params.py
  clients/              base.py (ClientAlgorithm, SGDClient, evaluate), reweigh.py, fedfdp.py, fairweight_local.py
  strategies/           one file per method, all subclass strategies/base.py:Strategy
  fairness/             scheme.py (<U, W_ex, B> as pydantic), status.py (U), scopes.py (scopes, W_ex)
  metrics/group.py      per-group benefits, DP / EO gaps from confusion counts
  data/                 datasets.py (loaders, augmentation), partition.py (Dirichlet, IID, shards), federated.py
  models/               MLP, small CNN, FedCDA/FedMut CNNs, ResNet-18, VGG-16, CelebA CNN
  scenarios/            hook interface for perturbations (no scenarios implemented yet)
  experiments/runner.py run, overrides, summary
  agent/                placeholder (empty)
```

**Strategy / ClientAlgorithm split.** The design is borrowed from Flower but does not import it.
- **Server side:** a `Strategy` (`strategies/base.py`) implements some of these hooks:
  - `configure_eval(rnd, ids)`: which clients score the current global model on their *train* split before selection (FairFed, FCFL);
  - `configure_round(rnd, gparams, ids, pre_eval)`: returns `{client_id: FitIns}`;
  - `aggregate(rnd, gparams, results)`: returns the new flat parameter vector;
  - `on_round_end(log)`;
  - `postprocess(...)`: an optional phase after training (LoGoFair);
  - `decision_rule()`: how probabilities become predictions; post-processing can replace argmax.
- **Client side:** `ClientAlgorithm` (`clients/base.py`) does local training. Subclasses override `batch_loss` to change the local objective, or `fit` for full control (DP-SGD, FairWeight masks).
- **Messages:** `FitIns` (params plus a `config` dict) and `FitRes` (client id, params, sample count, `metrics` dict) are pydantic models. Parameters travel as one flat `torch.Tensor`.
- **Stateful clients:** the engine gives each client a `state` dict that persists across rounds. It is used for local reweighing weights in FairFed and for adaptive loss-clipping state in FedFDP. Flower clients are stateless by default.

**Per-round, per-group evaluation.** After aggregation in every round, `Simulator` evaluates the global model on **every** client's local test split, whether or not the client was selected. It logs `GroupCounts` confusion counts indexed [sensitive group][true label]: `n`, `correct` and `pred_pos`. Every fairness metric (DP, TPR/FPR, EO, per-group and per-client accuracy) comes from these counts. So any status function, scope or deployment filter can be recomputed later from `rounds.jsonl`, without re-running training. If a strategy post-processes (LoGoFair), one extra log entry records the post-processed model as the final deployment.

## Fairness module

`fairness/` implements the scheme ⟨U, W_ex, B⟩ of Alamdari et al. for FL (`FairnessScheme` in `scheme.py`).

- **Stakeholders and benefit (`StatusSpec`)**:
  - `benefit`: `dp` (positive rate), `tpr`, `fpr`, `eo` (the worse of TPR and FPR), or `accuracy`.
  - `level`:
    - `global`: groups pooled over all clients;
    - `local_max` / `local_mean`: the gap within each client, then the max or mean over clients;
    - `clients`: each client is a stakeholder, benefit is its accuracy.
- **Status U (`status.py`)**: each stakeholder's benefit accumulated over the deployed history, with accumulation `instant` (Markovian), `cumulative`, `window` (last w deployments) or `discounted`. Benefits, not gaps, are accumulated, so an early advantage for one group is only cancelled by a later advantage for the other. Unfairness u_t is the spread of the accumulated statuses: max minus min across groups, or mean minus min across clients at the `clients` level.
- **Filter B**: `deploy_every = k` counts only every k-th round plus the final model.
- **Scopes (`scopes.py`)**, applied to the deployed rounds:
  - `long_term`: the last deployment only;
  - `periodic`: every `period`-th deployment;
  - `anytime`: every deployment;
  - `bounded`: an explicit list of `rounds`.
- **W_ex**: each scope reports the `final`, `max` and `mean` of u_t over its judgement points, and a `violation_rate`, the share of points with u_t > `epsilon`.

**Vaccine test** (`tests/test_fairness_scopes.py`). Four rounds in which group A gets all the benefit in rounds 1-2 and group B in rounds 3-4:
- the cumulative unfairness trajectory is [1, 1, 1/3, 0];
- long-term max unfairness is 0, while anytime max is 1 with a violation rate of 0.75;
- the periodic scope (p = 2) sees max 1 and final 0;
- the `instant` status stays at 1 every round;
- `deploy_every = 2` keeps only rounds 2 and 4, giving [1, 0].

## Implemented baselines

All methods are registered in `strategies/__init__.py` and listed by `uv run fairfl strategies`. "Paper" is the source used for the port. Reference code was consulted where noted.

| Name | Method (venue) | File | Key params | Fidelity notes |
|---|---|---|---|---|
| `fedavg` | FedAvg (AISTATS'17) | `strategies/fedavg.py` | none | Sample-size-weighted mean. |
| `median` | Coordinate-wise median, Chen et al. (NeurIPS'20) | `strategies/median.py` | `sigma` | Median of model updates. Optional Gaussian noise before the median (the paper's correction); `sigma = 0` gives plain median. |
| `qffl` | q-FFL / q-FedAvg (ICLR'20) | `strategies/qffl.py` | `q` | Server step with L = 1/lr, using each client's loss before local training. |
| `fairfed` | FairFed (AAAI'23), Eq 6 | `strategies/fairfed.py`, `clients/reweigh.py` | `beta`, `metric` (`eod`/`spd`), `local_reweight` | Recursive weights from \|F_global - F_k\|, with an accuracy gap where F_k is undefined locally. Metrics come from each client's train split under the current global model. Local debiasing is Kamiran-Calders reweighing, as in the paper's evaluation. Weights are clipped at 0. |
| `fairrfl` | FairRFL (IEEE TETC'26): RFL-Self + Dq-FFL | `strategies/fairrfl.py` | `tau` (2.5), `q`, `use_dqffl` | MAD detection (scale 1.4826) and recovery with the largest root β in [0, 1]. Dq-FFL scales q by l_med / l_i using the **previous** round's reported losses. `use_dqffl = false` gives RFL-Self only. |
| `fedfair` | FedFair (ACNS'26), Eq 5-7 | `strategies/fedfdp.py`, `clients/fedfdp.py` | `lam` | Local step scaled by (1 + λ(F_i - F)), with F_i the batch loss and F the broadcast global loss. The scale is floored at 0, a case the paper does not discuss. |
| `fedfdp` | FedFDP (ACNS'26), Eq 10-13 | same | `lam`, `clip`, `sigma`, `sigma_loss`, `loss_clip_init` | Fairness-aware per-sample clipping plus Gaussian noise (DP-SGD via `torch.func.vmap`), and adaptive clipping and noise on the uploaded loss. **Fixed-size shuffled batches instead of Poisson sampling, and no privacy accountant**: report σ, not ε. |
| `fedcda` | FedCDA (ICLR'24), Alg 1 / Eq 8 | `strategies/fedcda.py` | `K` (3), `batch_clients`, `smoothness`, `warmup` | FedAvg during warm-up. After that, batch-greedy exhaustive search over each client's K cached models. The global model is the plain average of every client's currently selected model, including clients not in this round. |
| `fedmut` | FedMut (AAAI'24) | `strategies/fedmut.py` | `radius` (α), `mut_acc_rate`, `mut_bound`, `weighted` | Port of `mutation_spread` from the official repo (`HMHelloWorld/FedMut`): per-layer sign pairs (1, -1 + rate), with the acceleration decaying to 0 over `mut_bound` rounds. Uniform averaging by default, as in the official code. |
| `fcfl` | FCFL (ECML-PKDD'24), Eq 1-3 | `strategies/fcfl.py` | `eta`, `random_fraction` (r) | Accumulated queues Q_i drive selection (top-Q) and aggregation weights (proportional to Q; sample sizes if all Q are 0). **Supports the paper's random fraction r; the official code uses pure top-Q, which is the default (r = 0).** Round 0 samples uniformly. |
| `logofair` | LoGoFair (AAAI'25), Thm 1 / Eq 3-7 | `strategies/logofair.py` | `mode` (`lg`/`l`/`g`), `metric` (`dp`/`eo`), `delta_l`, `delta_g`, `beta`, `post_rounds`, `lr`, `inner_steps`, `inner_lr`, `calibrate`, `min_group_count` | Post-processing after FedAvg training: λ by federated projected GD, μ_c solved locally, hinge smoothed by softplus. Per-client temperature scaling of η (optional). Uses the `val` split if `data.val_fraction > 0`, otherwise the train split. **Not in the paper:** clients with fewer than `min_group_count` samples of a group in that split get no local constraint. |
| `fairweight` | FairWeight (IEEE TSC'26) | `strategies/fairweight.py`, `clients/fairweight_local.py` | `gamma1`, `gamma2`, `top_fraction`, `fair_weight`, `repeats`, `max_samples` | Follows the authors' released code: WI = \|grad · θ\| with 25% random zeroing, and a top-k mask per group-pair contrast. **Secure aggregation is simulated as a plain sum. The local loss is cross-entropy plus a squared demographic-parity penalty rather than the official ATE loss.** Default `repeats = 5` (official code: 100). Flagged coordinates get fairness scores; all others are averaged uniformly. |

LoGoFair and FairWeight assume a binary sensitive attribute and a binary task, so use them on `adult`, `bank`, `default`, `law`, `kdd`, `celeba` or `synthetic`.

## Paper worked examples covered by tests

`tests/test_strategies.py` checks:
- **FairRFL, Examples 4-5:** N_med ≈ 1.1, the right update is flagged, the median is [-0.20, 0.55], β ≈ 0.45 and the recovered update is ≈ [0.52, 0.96].
- **FairWeight, Section IV-3 toy:** per-coordinate scores for masks {3}, {1,3}, {1,2,3} match the paper's weightage.
- **FedMut:** mutation signs cancel in pairs, and the acceleration gives signs (1, -0.7).
- **Reductions to FedAvg:** FCFL with η = 0, FairFed with β = 0 and no reweighing, FedFair with λ = 0, and FedFDP with no noise or clipping all produce the same parameters as FedAvg.
- **Median and q-FFL:** Median is coordinate-wise; q-FFL with q = 0 equals the mean of updates, and q-FFL upweights high-loss clients.
- **LoGoFair:** post-processing more than halves the global DP gap on synthetic data.
- **Smoke tests** for FairRFL, FedCDA and FairWeight.

`tests/test_fairness_scopes.py` covers the vaccine example (above). `tests/test_partition.py` covers Dirichlet and IID partitioning. `tests/test_extensions.py` covers the shards partition, the image models, augmentation, `eval_every`, zero-momentum equivalence, device placement and the CelebA and tabular loaders.

## Output format

Each run writes to `runs/<name>/seed<k>/`, or to `--out` if given:

| File | Contents |
|---|---|
| `config.json` | The fully resolved `ExperimentConfig`. |
| `rounds.jsonl` | One `RoundLog` per line: round, selected clients, train loss, global accuracy and loss, per-client `EvalRes` with `GroupCounts`, `strategy_info` (for example FCFL queues, FairFed weights, FairRFL flagged clients), and wall-clock seconds. |
| `summary.json` | Final accuracy, client-accuracy std and worst client, final DP and EO gaps, total seconds, and `scopes`: for each report status (`global_dp`, `global_eo`, `local_dp_max`, `client_acc`) and each scope (`long_term`, `periodic`, `anytime`, `bounded`), the points, final, max, mean and violation rate. |

Notes on `summary.json`:
- The report statuses use cumulative accumulation.
- The config's `deploy_every`, `epsilon` and bounded `rounds` are applied.
- The periodic scope in the summary uses period 5.
- To score other schemes, use `fairfl.core.engine.load_logs` and `fairfl.fairness.scopes.evaluate_scheme`.

## Reproducing the baseline papers

Per-paper exact settings (paper plus official code, with conflicts listed) are in `../docs/repro/<paper>.md`. Configs are in `configs/repro/<paper>/`, and suites (configs × seeds) in `suites/`.

```bash
uv run fairfl batch suites/logofair_adult.yaml --workers 2 --threads 2   # parallel CPU processes; finished runs are skipped
uv run fairfl compare                                                   # paper vs ours
```

- **Every finished run** is appended to `results/runs.jsonl`. This is the persistent record; `runs/` is scratch. To store it elsewhere, set `FAIRFL_RESULTS_DIR`.
- **Targets** live in `results/reproduction/published.csv`. `fairfl compare` joins them with the latest run of each name and writes `results/reproduction/published_vs_ours.csv`, marking each metric OK, GAP or NOT RUN.
- **GPU suites** (FedMut and FedCDA on CIFAR-10, FedCDA on FMNIST) run on Colab through `notebooks/fairfl_gpu.ipynb`:
  1. Set `SUITES` (and optionally `ONLY`) in the first cell, then run all cells.
  2. Results go to `MyDrive/fairfl/results/runs.jsonl`.
  3. To merge them, append those lines to `results/runs.jsonl`, then run `uv run fairfl compare`.
- **LoGoFair** uses the authors' shipped Adult split and pre-trained model. These go in `data/logofair/` (gitignored, about 29 MB), copied from github.com/liizhang/LoGofair:
  - `data/adult/split_data/num_users=5 sensitive_attr=sex dirichlet=0.5 by_sensitive=True/split_data.json`, saved as `adult_alpha0.5_split.json`;
  - its `pre_model.pth`, saved as `adult_alpha0.5_pre_model.pth`.
- **FedMut's** client partitions are committed in `assets/partitions/fedmut/`.
- Deviations from the papers are logged in `../docs/lab-notes.md`.

## Adding your own method

1. Create `src/fairfl/strategies/mymethod.py`:

   ```python
   from pydantic import BaseModel, Field

   from fairfl.core.registry import register_strategy
   from fairfl.strategies.base import Strategy


   class MyParams(BaseModel):
       strength: float = Field(1.0, ge=0.0, description="Shown in help and generated forms.")


   @register_strategy("mymethod")
   class MyMethod(Strategy):
       Params = MyParams
       # client_cls = MyClient     # optional ClientAlgorithm subclass

       def aggregate(self, rnd, gparams, results):
           ...                     # return the new flat parameter tensor
   ```

2. Import the module in `src/fairfl/strategies/__init__.py`.
3. It then appears in `uv run fairfl strategies` and can be selected with `strategy: {name: mymethod, params: {strength: 2.0}}`. Params are validated by the pydantic model. The planned dashboard will build its forms from the same `Params` schema; it is not implemented yet.
