"""Builds fairfl_train.ipynb. Edit the cells here, then run: python notebooks/_build.py"""
import json
from pathlib import Path

cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(keepends=True)})


def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": s.strip("\n").splitlines(keepends=True)})


md("""
# fairfl: 8 FL strategies on CIFAR10 (ResNet18)

Self-contained notebook for Colab / Kaggle. No local project install needed — this cell pulls the
`fairfl` source straight from the repo and pip-installs only what's missing.

**Colab tip:** set `USE_DRIVE = True` below so the CIFAR10 download (and your results) persist
across sessions — otherwise `/content` is wiped on disconnect and every session re-downloads it
(the multi-minute "loading data" wait at the start).

Runs every strategy in `STRATEGIES` below, one after another, each from its own YAML config under
`code/src/fairfl/configs/`. Edit `STRATEGIES` or `OVERRIDES` and Run All.
""")

code("""
# ---- setup: get fairfl source + deps (works on Colab, Kaggle, or a plain GPU box) ----
import os, sys, subprocess
from pathlib import Path

USE_DRIVE = True   # Colab only: keep CIFAR10 + results on Google Drive so they survive a disconnect

try:
    import google.colab  # noqa: F401
    ON_COLAB = True
except ImportError:
    ON_COLAB = False
ON_KAGGLE = not ON_COLAB and "KAGGLE_KERNEL_RUN_TYPE" in os.environ
print("platform:", "kaggle" if ON_KAGGLE else "colab" if ON_COLAB else "local")

REPO_URL = "https://github.com/daanish02/fair-fl.git"
BRANCH = "main"

if ON_COLAB or ON_KAGGLE:
    REPO = Path("/content/fair-fl") if ON_COLAB else Path("/kaggle/working/fair-fl")
    if (REPO / ".git").exists():
        subprocess.run(f"git -C {REPO} pull -q", shell=True, check=True)
    else:
        subprocess.run(f"git clone -q --depth 1 -b {BRANCH} {REPO_URL} {REPO}", shell=True, check=True)
    SRC = REPO / "code" / "src"
    CONFIGS_DIR = SRC / "fairfl" / "configs"

    if ON_COLAB and USE_DRIVE:
        from google.colab import drive
        drive.mount("/content/drive")
        WORK = Path("/content/drive/MyDrive/fairfl")
        WORK.mkdir(parents=True, exist_ok=True)
        DATA_ROOT = str(WORK / "data")       # CIFAR10 downloads here once, reused every session
        RESULTS_DIR_DEFAULT = str(WORK / "results")
    elif ON_KAGGLE:
        DATA_ROOT = "/kaggle/working/data"   # /kaggle/working is kept when you "Save Version"
        RESULTS_DIR_DEFAULT = "/kaggle/working/results"
    else:
        DATA_ROOT = str(REPO / "code" / "data")   # ephemeral: re-downloads every Colab session
        RESULTS_DIR_DEFAULT = "../results"
else:
    SRC = Path.cwd().parent / "src"   # running locally from code/notebooks/
    CONFIGS_DIR = SRC / "fairfl" / "configs"
    DATA_ROOT = str(Path.cwd().parent / "data")
    RESULTS_DIR_DEFAULT = "../results"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# torch/torchvision are preinstalled on Colab and Kaggle; pydantic/pyyaml usually are too.
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pydantic>=2.7", "pyyaml"], check=True)

import torch
torch.backends.cudnn.benchmark = True   # fixed input sizes -> faster convs after warmup
print(torch.__version__, "| cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
print("data:", DATA_ROOT)
print("configs:", CONFIGS_DIR)
""")

code("""
# ---- CONFIG: which strategies to run, and any overrides on top of their YAML configs ----
# Each name maps to code/src/fairfl/configs/<name>.yaml. fedcda is excluded by default: its
# selection mechanism only activates after a 50-round warmup, so it needs its own call (see
# docs/meeting-notes/2026-10-08.md) before it's folded into this list.
STRATEGIES = ["median", "signsgd", "fairrfl", "fcfl", "fedmut", "fairweight", "fedfdp", "logofair"]

# Dotted overrides applied to every strategy's YAML, same syntax as `fairfl train --set`.
# data_root/device are injected here regardless, so configs stay environment-agnostic.
OVERRIDES = []

SEED = 0
RESULTS_DIR = RESULTS_DIR_DEFAULT   # set above; Drive/Kaggle-persistent path when available
""")

code("""
# ---- logging: INFO to notebook output, with timestamps ----
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S", force=True)
""")

code("""
# ---- run: every strategy in STRATEGIES, one after another, each saved as its own json ----
import time, json
from pathlib import Path
from fairfl.config import RunConfig
from fairfl.train.engine import RunResult, run

device = "cuda" if torch.cuda.is_available() else "cpu"
out_dir = Path(RESULTS_DIR)
out_dir.mkdir(parents=True, exist_ok=True)
saved_paths = []

for name in STRATEGIES:
    cfg_path = CONFIGS_DIR / f"{name}.yaml"
    overrides = [f"dataset.data_root={DATA_ROOT}", f"train.device={device}", f"dataset.seed={SEED}", *OVERRIDES]
    cfg = RunConfig.from_yaml_and_overrides(cfg_path, overrides)

    mode = "centralised" if cfg.is_centralised else f"federated ({cfg.dataset.num_clients} clients)"
    print(f"\\n=== {name} ===")
    print(f"[fairfl] {mode} | strategy={cfg.strategy.name} model={cfg.model.name} "
          f"rounds={cfg.train.rounds} alpha={cfg.dataset.alpha} lr={cfg.train.lr} device={cfg.train.device}")

    t0 = time.time()
    result = None
    for item in run(cfg):
        if isinstance(item, RunResult):
            result = item
    elapsed_min = (time.time() - t0) / 60
    print(f"{name}: total time {elapsed_min:.1f} min")

    print("--- final per-client accuracy ---")
    for c, acc in sorted(result.client_acc.items()):
        print(f"client {c}: {acc:.4f}")

    out_path = out_dir / f"{name}_c{cfg.dataset.num_clients}_a{cfg.dataset.alpha}_seed{cfg.dataset.seed}.json"
    payload = {
        "config": cfg.model_dump(),
        "history": [h.__dict__ for h in result.history],
        "client_acc": {str(k): v for k, v in result.client_acc.items()},
        "elapsed_min": elapsed_min,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    saved_paths.append(out_path)
    print("saved ->", out_path)

print("\\nall done:", [str(p) for p in saved_paths])
""")

md("""
## Plot: accuracy across clients

x-axis = client id, y-axis = test accuracy, one colour per strategy. Re-run this cell (without
retraining) to combine any set of saved result files.
""")

code("""
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", palette="deep", font_scale=1.05)

RESULT_FILES = saved_paths   # defaults to everything just run; replace with specific Path(...) entries to compare a subset

rows = []
for p in RESULT_FILES:
    d = json.loads(Path(p).read_text())
    label = d["config"]["strategy"]["name"]
    for k, v in d["client_acc"].items():
        rows.append({"client": int(k), "accuracy": v, "strategy": label})
df_clients = pd.DataFrame(rows)

fig, ax = plt.subplots(figsize=(9, 4.5))
sns.barplot(data=df_clients, x="client", y="accuracy", hue="strategy", ax=ax)
ax.set_xlabel("client")
ax.set_ylabel("test accuracy")
ax.set_ylim(0, 1)
ax.set_title("Accuracy across clients, by strategy")
sns.despine()
plt.savefig(Path(RESULTS_DIR) / "client_accuracy.png", dpi=150, bbox_inches="tight")
plt.show()
""")

md("""
## Plot: accuracy over rounds

x-axis = communication round, y-axis = global test accuracy, one line per strategy.
""")

code("""
rows = []
for p in RESULT_FILES:
    d = json.loads(Path(p).read_text())
    label = d["config"]["strategy"]["name"]
    for h in d["history"]:
        rows.append({"round": h["round"], "accuracy": h["test_acc"], "strategy": label})
df_rounds = pd.DataFrame(rows)

fig, ax = plt.subplots(figsize=(9, 4.5))
sns.lineplot(data=df_rounds, x="round", y="accuracy", hue="strategy", marker="o", ax=ax)
ax.set_xlabel("round")
ax.set_ylabel("global test accuracy")
ax.set_ylim(0, 1)
ax.set_title("Accuracy over rounds, by strategy")
sns.despine()
plt.savefig(Path(RESULTS_DIR) / "accuracy_over_rounds.png", dpi=150, bbox_inches="tight")
plt.show()
""")

nb = {"cells": cells, "metadata": {"accelerator": "GPU",
                                    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                    "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
Path(__file__).with_name("fairfl_train.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("wrote fairfl_train.ipynb")
