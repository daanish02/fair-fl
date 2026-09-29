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
# fairfl: centralised vs federated CIFAR10 (ResNet18)

Self-contained notebook for Colab / Kaggle. No local project install needed — this cell pulls the
`fairfl` source straight from the repo and pip-installs only what's missing.

**Colab tip:** set `USE_DRIVE = True` below so the CIFAR10 download (and your results) persist
across sessions — otherwise `/content` is wiped on disconnect and every session re-downloads it
(the multi-minute "loading data" wait at the start).

Edit the ALL_CAPS cell below to change hyperparameters, then Run All.
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
""")

code("""
# ---- CONFIG: tune these, then Run All ----
NUM_CLIENTS = 10      # 0 = centralised (no clients, plain DL baseline)
ROUNDS = 30           # communication rounds (FL) or epochs (centralised)
LOCAL_EPOCHS = 5      # local epochs per client per round (FL only)
ALPHA = 0.5           # Dirichlet concentration; lower = more non-IID
LR = 0.1              # SGD initial lr; cosine-annealed to 0 over ROUNDS
WEIGHT_DECAY = 5e-4
BATCH_SIZE = 64
SEED = 0
DEVICE = "auto"       # "auto", "cpu", "cuda"
LOG_EVERY = 50        # log every N batches; 0 = off
RESULTS_DIR = RESULTS_DIR_DEFAULT   # set above; Drive/Kaggle-persistent path when available
""")

code("""
# ---- logging: INFO to notebook output, with timestamps ----
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S", force=True)
""")

code("""
# ---- run ----
import time, json
from pathlib import Path
from fairfl.config import RunConfig
from fairfl.train.engine import RunResult, run

cfg = RunConfig(
    num_clients=NUM_CLIENTS, rounds=ROUNDS, local_epochs=LOCAL_EPOCHS, alpha=ALPHA,
    lr=LR, weight_decay=WEIGHT_DECAY, batch_size=BATCH_SIZE, seed=SEED, data_root=DATA_ROOT, device=DEVICE,
    log_every=LOG_EVERY,
)
mode = "centralised" if cfg.is_centralised else f"federated ({cfg.num_clients} clients)"
print(f"[fairfl] {mode} | rounds={cfg.rounds} alpha={cfg.alpha} lr={cfg.lr} device={cfg.device}")

t0 = time.time()
result = None
for item in run(cfg):
    if isinstance(item, RunResult):
        result = item
print(f"total time: {(time.time() - t0) / 60:.1f} min")

print("--- final per-client accuracy ---")
for c, acc in sorted(result.client_acc.items()):
    print(f"client {c}: {acc:.4f}")
""")

code("""
# ---- save results (json) so the plot cell below can be re-run without retraining ----
out_dir = Path(RESULTS_DIR)
out_dir.mkdir(parents=True, exist_ok=True)
tag = "centralised" if cfg.is_centralised else f"federated_c{cfg.num_clients}_a{cfg.alpha}"
out_path = out_dir / f"{tag}_seed{cfg.seed}.json"
payload = {
    "config": cfg.model_dump(),
    "history": [h.__dict__ for h in result.history],
    "client_acc": {str(k): v for k, v in result.client_acc.items()},
}
out_path.write_text(json.dumps(payload, indent=2))
print("saved ->", out_path)
""")

md("""
## Plot: accuracy across clients

x-axis = client id (0 = centralised / server-only, i.e. plain DL baseline), y-axis = test accuracy.
Re-run this cell (without retraining) to combine multiple saved result files, e.g. a centralised run
plus a federated run, on the same chart.
""")

code("""
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", palette="deep", font_scale=1.05)

RESULT_FILES = [out_path]   # add more Path(...) entries here to compare multiple runs

rows = []
for p in RESULT_FILES:
    d = json.loads(Path(p).read_text())
    label = "centralised" if d["config"]["num_clients"] == 0 else f"federated (n={d['config']['num_clients']}, a={d['config']['alpha']})"
    for k, v in d["client_acc"].items():
        rows.append({"client": int(k), "accuracy": v, "run": label})
df_clients = pd.DataFrame(rows)

fig, ax = plt.subplots(figsize=(7.5, 4.5))
sns.barplot(data=df_clients, x="client", y="accuracy", hue="run", ax=ax)
ax.set_xlabel("client (0 = centralised/server-only)")
ax.set_ylabel("test accuracy")
ax.set_ylim(0, 1)
ax.set_title("Accuracy across clients")
sns.despine()
plt.savefig(Path(RESULTS_DIR) / "client_accuracy.png", dpi=150, bbox_inches="tight")
plt.show()
""")

md("""
## Plot: accuracy over rounds

x-axis = round (communication round for FL, epoch for centralised), y-axis = global test accuracy.
Same `RESULT_FILES` list as above, so both runs overlay on one chart.
""")

code("""
rows = []
for p in RESULT_FILES:
    d = json.loads(Path(p).read_text())
    label = "centralised" if d["config"]["num_clients"] == 0 else f"federated (n={d['config']['num_clients']}, a={d['config']['alpha']})"
    for h in d["history"]:
        rows.append({"round": h["round"], "accuracy": h["test_acc"], "run": label})
df_rounds = pd.DataFrame(rows)

fig, ax = plt.subplots(figsize=(7.5, 4.5))
sns.lineplot(data=df_rounds, x="round", y="accuracy", hue="run", marker="o", ax=ax)
ax.set_xlabel("round")
ax.set_ylabel("global test accuracy")
ax.set_ylim(0, 1)
ax.set_title("Accuracy over rounds")
sns.despine()
plt.savefig(Path(RESULTS_DIR) / "accuracy_over_rounds.png", dpi=150, bbox_inches="tight")
plt.show()
""")

nb = {"cells": cells, "metadata": {"accelerator": "GPU",
                                    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                    "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
Path(__file__).with_name("fairfl_train.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("wrote fairfl_train.ipynb")
