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

Edit the ALL_CAPS cell below to change hyperparameters, then Run All.
""")

code("""
# ---- setup: get fairfl source + deps (works on Colab, Kaggle, or a plain GPU box) ----
import os, sys, subprocess

try:
    import google.colab  # noqa: F401
    ON_COLAB = True
except ImportError:
    ON_COLAB = False
ON_KAGGLE = not ON_COLAB and "KAGGLE_KERNEL_RUN_TYPE" in os.environ
print("platform:", "kaggle" if ON_KAGGLE else "colab" if ON_COLAB else "local")

REPO_URL = "https://github.com/daanish02/fair-fl.git"
BRANCH = "main"

from pathlib import Path
if ON_COLAB or ON_KAGGLE:
    REPO = Path("/content/fair-fl") if ON_COLAB else Path("/kaggle/working/fair-fl")
    if (REPO / ".git").exists():
        subprocess.run(f"git -C {REPO} pull -q", shell=True, check=True)
    else:
        subprocess.run(f"git clone -q --depth 1 -b {BRANCH} {REPO_URL} {REPO}", shell=True, check=True)
    SRC = REPO / "code" / "src"
    DATA_ROOT = str(REPO / "code" / "data")
else:
    SRC = Path.cwd().parent / "src"   # running locally from code/notebooks/
    DATA_ROOT = str(Path.cwd().parent / "data")

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# torch/torchvision are preinstalled on Colab and Kaggle; pydantic/pyyaml usually are too.
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pydantic>=2.7", "pyyaml"], check=True)

import torch
print(torch.__version__, "| cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
""")

code("""
# ---- CONFIG: tune these, then Run All ----
NUM_CLIENTS = 10      # 0 = centralised (no clients, plain DL baseline)
ROUNDS = 30           # communication rounds (FL) or epochs (centralised)
LOCAL_EPOCHS = 5      # local epochs per client per round (FL only)
ALPHA = 0.5           # Dirichlet concentration; lower = more non-IID
LR = 0.01
BATCH_SIZE = 64
SEED = 0
DEVICE = "auto"       # "auto", "cpu", "cuda"
LOG_EVERY = 50        # log every N batches; 0 = off
RESULTS_DIR = "../results"
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
    lr=LR, batch_size=BATCH_SIZE, seed=SEED, data_root=DATA_ROOT, device=DEVICE, log_every=LOG_EVERY,
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

RESULT_FILES = [out_path]   # add more Path(...) entries here to compare multiple runs

fig, ax = plt.subplots(figsize=(7, 4))
for p in RESULT_FILES:
    d = json.loads(Path(p).read_text())
    clients = sorted(int(k) for k in d["client_acc"].keys())
    accs = [d["client_acc"][str(c)] for c in clients]
    label = "centralised" if d["config"]["num_clients"] == 0 else f"federated (n={d['config']['num_clients']}, a={d['config']['alpha']})"
    ax.plot(clients, accs, marker="o", label=label)

ax.set_xlabel("client (0 = centralised/server-only)")
ax.set_ylabel("test accuracy")
ax.set_ylim(0, 1)
ax.legend()
ax.set_title("Accuracy across clients")
plt.savefig(Path(RESULTS_DIR) / "client_accuracy.png", dpi=150, bbox_inches="tight")
plt.show()
""")

nb = {"cells": cells, "metadata": {"accelerator": "GPU",
                                    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                    "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
Path(__file__).with_name("fairfl_train.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("wrote fairfl_train.ipynb")
