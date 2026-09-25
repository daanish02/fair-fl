"""Builds fairfl_gpu.ipynb. Edit the cells here, then run: uv run python notebooks/_build.py"""
import json
from pathlib import Path

cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(keepends=True)})


def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": s.strip("\n").splitlines(keepends=True)})


md("""
# fairfl on a GPU (Colab or Kaggle)

Runs the `fairfl` reproduction configs on a free GPU. The same notebook works on both platforms.

**Before running**
- **Colab:** Runtime > Change runtime type > T4 GPU. Set `USE_DRIVE = True` below so results survive a disconnect.
- **Kaggle:** Settings > Accelerator > GPU T4 x2 (one is used), and turn **Internet on**. Results go to `/kaggle/working`, which is kept when you "Save Version".

**Resuming:** each run writes `summary.json` when it finishes. Re-running the notebook skips finished runs, so after a disconnect just run all cells again.
""")

code("""
# ---- settings ----
REPO_URL = "https://github.com/daanish02/fair-fl.git"
BRANCH = "main"
USE_DRIVE = True          # Colab only: keep data and results on Google Drive
DRIVE_DIR = "fairfl"      # folder inside MyDrive

# Suites to run (paths relative to code/). Each suite lists configs x seeds; finished runs are skipped, so
# after a disconnect just run all cells again. Rough T4 times per run, in the order below:
#   fedmut_cifar10_cnn       8 runs  ~ 30-40 min each (LeNet CNN, 1000 rounds)
#   fedcda_fmnist           12 runs  ~ 10-15 min each (small CNN, 200 rounds x 20 local epochs)
#   fedfdp_mnist             2 runs  ~ 30-60 min each (per-sample DP gradients)
#   fedmut_cifar10_resnet18  8 runs  ~ 2.5-3 h each   -> needs several sessions; ONLY lists d0.1 and IID first
#   fedcda_cifar10          12 runs  ~ 4 h each       -> ResNet-18, 20 local epochs; run a subset (ONLY)
SUITES = [
    "suites/fedmut_cifar10_cnn.yaml",
    "suites/fedcda_fmnist.yaml",
    "suites/fedfdp_mnist.yaml",        # 2 runs, 776 DP rounds; ~70 s/round on a laptop CPU, run it here
    "suites/median_mnist.yaml",        # 6 runs, 10^4 full-batch rounds each (Chen et al. Fig. 2)
]
# Optional: restrict to runs whose name contains one of these strings (e.g. ["d0.1", "iid"]); empty = all.
ONLY = []
# Experiments run at the same time on the one GPU. The small CNNs leave the GPU mostly idle, so 2 roughly doubles
# throughput (free Colab has 2 CPU cores; use 1 for ResNet-18 suites). Each run then logs to its own log.txt.
PARALLEL = 2
""")

code("""
import os, sys, subprocess, time
from pathlib import Path

try:
    import google.colab  # noqa: F401  (only importable on Colab)
    ON_COLAB = True
except ImportError:
    ON_COLAB = False
# Colab images can contain a /kaggle folder, so detect Kaggle by its own environment variable.
ON_KAGGLE = not ON_COLAB and "KAGGLE_KERNEL_RUN_TYPE" in os.environ
print("platform:", "kaggle" if ON_KAGGLE else "colab" if ON_COLAB else "local")

if ON_KAGGLE:
    WORK = Path("/kaggle/working")
elif ON_COLAB and USE_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    WORK = Path("/content/drive/MyDrive") / DRIVE_DIR
else:
    WORK = Path("/content") if ON_COLAB else Path.cwd()
WORK.mkdir(parents=True, exist_ok=True)
REPO = Path("/content/fair-fl") if ON_COLAB else WORK / "fair-fl"   # code on local disk (faster than Drive)
DATA = WORK / "data"
RESULTS = WORK / "runs"
# Every finished run is appended to WORK/results/runs.jsonl (persistent on Drive / Kaggle output).
# Merge it into the repo's code/results/runs.jsonl afterwards (append the lines).
os.environ["FAIRFL_RESULTS_DIR"] = str(WORK / "results")
print("work:", WORK, "| repo:", REPO)
if ON_COLAB and USE_DRIVE:
    assert str(WORK).startswith("/content/drive"), "Drive is not mounted: results would be lost at session end"
""")

code("""
def sh(cmd):
    print("$", cmd)
    subprocess.run(cmd, shell=True, check=True)

if (REPO / ".git").exists():
    sh(f"git -C {REPO} fetch -q && git -C {REPO} checkout -q {BRANCH} && git -C {REPO} pull -q")
else:
    sh(f"git clone -q --depth 1 -b {BRANCH} {REPO_URL} {REPO}")

# torch/torchvision/numpy/pandas/sklearn are preinstalled on both platforms; don't reinstall them.
sh(f"pip install -q --no-deps --ignore-requires-python -e {REPO / 'code'}")  # Kaggle may run Python 3.11
sh("pip install -q 'pydantic>=2.7' pyyaml typer")
# An editable install is only picked up by a fresh interpreter; make the package importable in this kernel now.
if str(REPO / "code" / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "code" / "src"))
""")

code("""
import torch
assert torch.cuda.is_available(), "No GPU: enable one in the runtime/accelerator settings."
print(torch.__version__, "|", torch.cuda.get_device_name(0),
      f"| {torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GB")
torch.backends.cudnn.benchmark = True   # fixed input sizes -> faster convolutions
os.chdir(REPO / "code")
import shutil
(WORK / "results" / "reproduction").mkdir(parents=True, exist_ok=True)
shutil.copy("results/reproduction/published.csv", WORK / "results" / "reproduction" / "published.csv")
""")

md("""
### CelebA (only needed for CelebA configs)
torchvision's CelebA download from Google Drive usually fails because of quota.
- **Kaggle:** add the dataset `jessicali9530/celeba-dataset` (Add input), then run the cell below to link it into the layout torchvision expects.
- **Colab:** upload the torchvision layout (`celeba/img_align_celeba/`, `list_attr_celeba.txt`, `list_eval_partition.txt`) to `DATA/celeba`.
""")

code("""
def link_kaggle_celeba():
    src = Path("/kaggle/input/celeba-dataset")
    if not src.exists():
        print("CelebA input not attached; skipping."); return
    dst = DATA / "celeba"; dst.mkdir(parents=True, exist_ok=True)
    img = src / "img_align_celeba" / "img_align_celeba"
    if not (dst / "img_align_celeba").exists():
        os.symlink(img, dst / "img_align_celeba")
    import pandas as pd
    # Kaggle ships CSVs; torchvision expects the original whitespace-separated txt files.
    attr = pd.read_csv(src / "list_attr_celeba.csv")
    with open(dst / "list_attr_celeba.txt", "w") as f:
        f.write(f"{len(attr)}\\n" + " ".join(attr.columns[1:]) + "\\n")
        attr.to_csv(f, sep=" ", header=False, index=False)
    part = pd.read_csv(src / "list_eval_partition.csv")
    part.to_csv(dst / "list_eval_partition.txt", sep=" ", header=False, index=False)
    print("linked CelebA into", dst)

if ON_KAGGLE:
    link_kaggle_celeba()
""")

code("""
import gc, json, yaml
from fairfl.core.config import ExperimentConfig
from fairfl.experiments.batch import Suite
from fairfl.experiments.runner import apply_overrides, run_experiment

def suite_experiments(paths, only=()):
    out = []
    for p in paths:
        suite = Suite.model_validate(yaml.safe_load(open(p)))
        for r in suite.runs:
            name = r.set.get("name", "")
            if not only or any(o in name for o in only):
                out.append((r.config, r.seeds, r.set))
    return out

def run_all(experiments):
    done = []
    for cfg_path, seeds, overrides in experiments:
        base = ExperimentConfig.from_yaml(cfg_path)
        sets = [f"{k}={json.dumps(v)}" for k, v in overrides.items()] + [f"data.root={json.dumps(str(DATA))}", 'device="cuda"']
        base = apply_overrides(base, sets)
        for seed in seeds:
            cfg = base.model_copy(update={"seed": seed})
            out = RESULTS / cfg.name / f"seed{seed}"
            if (out / "summary.json").exists():
                print(f"skip (done): {out}"); done.append(out); continue
            t0 = time.time()
            run_experiment(cfg, out, verbose=True)
            print(f"finished {cfg.name} seed {seed} in {(time.time() - t0) / 60:.1f} min -> {out}")
            done.append(out)
            gc.collect(); torch.cuda.empty_cache()
    return done

EXPERIMENTS = suite_experiments(SUITES, ONLY)
print(len(EXPERIMENTS), "experiments queued")
if PARALLEL > 1:
    # Parallel worker processes (checkpointed and resumable like the sequential path). Progress per run is in
    # RESULTS/<name>/seed<k>/log.txt; finished runs are printed here as they complete.
    import tempfile
    from fairfl.experiments.batch import run_suite
    for path in SUITES:
        suite = yaml.safe_load(open(path))
        suite["name"] = ""  # outputs go to RESULTS/<name>/seed<k>, same as the sequential path
        extra = {"data.root": str(DATA), "device": "cuda"}
        suite["runs"] = [dict(r, set={**r.get("set", {}), **extra}) for r in suite["runs"]
                         if not ONLY or any(o in r.get("set", {}).get("name", "") for o in ONLY)]
        tmp = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
        yaml.safe_dump(suite, tmp); tmp.close()
        # run the suite in a background thread and print each active run's latest log line every minute
        import threading
        th = threading.Thread(target=run_suite, args=(tmp.name,), kwargs=dict(workers=PARALLEL, threads=1, root=str(RESULTS)))
        th.start()
        while th.is_alive():
            th.join(timeout=60)
            now = time.time()
            for log in sorted(RESULTS.glob("*/seed*/log.txt")):
                if now - log.stat().st_mtime < 180 and not (log.parent / "summary.json").exists():
                    lines = [l for l in log.read_text(errors="ignore").splitlines() if "round" in l]
                    if lines:
                        print(time.strftime("%H:%M"), lines[-1].strip(), flush=True)
else:
    finished = run_all(EXPERIMENTS)
""")

code("""
# FairRFL Table III, CIFAR-10 (Augello et al., IEEE TETC 2026): official-code port, 8 methods x 0/10/20/30% selfish
# clients = 32 runs of 30 rounds. Too slow on a laptop CPU (~50 min per run). Finished runs are skipped
# (looked up by name in results/runs.jsonl), so after a disconnect just run this cell again.
RUN_FAIRRFL = True
# Per-client train size and client lr: the paper/code do not pin these down (defaults: code 200, paper 0.1; with
# those, FedAvg reached only ~11% vs the paper's 61%). Non-default values get their own run names
# (..._size500_lr0.1), so they never overwrite the Table III runs. FAIRRFL_ONLY = ["fedavg_s0"] runs just one.
FAIRRFL_SIZE = 200
FAIRRFL_LR = 0.1
FAIRRFL_ONLY = []
if RUN_FAIRRFL:
    import torchvision
    torchvision.datasets.CIFAR10(str(DATA), download=True)   # extract once before the workers start
    cmd = [sys.executable, "-u", "-m", "fairfl.experiments.fairrfl_official", "--root", str(DATA),
           "--workers", str(PARALLEL), "--threads", "1", "--size", str(FAIRRFL_SIZE), "--lr", str(FAIRRFL_LR)]
    if FAIRRFL_ONLY:
        cmd += ["--only", *FAIRRFL_ONLY]
    env = {**os.environ, "PYTHONPATH": str(REPO / "code" / "src")}
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    for line in proc.stdout:          # live: one line per worker every 10 rounds, one per finished run
        print(line, end="", flush=True)
    assert proc.wait() == 0, "FairRFL runner failed (see the traceback above)"
""")

code("""
import pandas as pd

rows = []
for p in RESULTS.glob("*/seed*/summary.json"):
    s = json.loads(p.read_text())
    row = {"run": p.parent.parent.name, "seed": p.parent.name,
           "acc": s["final_accuracy"], "worst_client_acc": s["final_worst_client_acc"],
           "client_acc_std": s["final_client_acc_std"], "dp_gap": s["final_dp_gap"], "eo_gap": s["final_eo_gap"],
           "minutes": s["seconds"] / 60}
    for status, scopes in s.get("scopes", {}).items():
        for scope, v in scopes.items():
            row[f"{status}/{scope}/max"] = v["max"]
    rows.append(row)
df = pd.DataFrame(rows)
if len(df):
    table = df.drop(columns="seed").groupby("run").agg(["mean", "std"]).round(4)
    table.to_csv(WORK / "results_table.csv")
    display(table[["acc", "worst_client_acc", "client_acc_std", "dp_gap", "eo_gap", "minutes"]])
""")

code("""
# Published vs ours for everything run so far (writes WORK/results/reproduction/published_vs_ours.csv).
from fairfl.experiments.compare import compare
for r in compare():
    if r["status"] != "NOT RUN":
        print(f"{r['status']:4s} {r['run_name']:40s} {r['metric']:14s} paper {r['published']:>8s} ours {r['ours']:>8s}")
""")

code("""
# Download everything (Colab). On Kaggle, /kaggle/working is kept when you save a version.
import shutil
archive = shutil.make_archive(str(WORK / "fairfl_runs"), "zip", RESULTS)
print("archive:", archive)
if ON_COLAB and not USE_DRIVE:
    from google.colab import files
    files.download(archive)
""")

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
Path(__file__).with_name("fairfl_gpu.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("wrote fairfl_gpu.ipynb")
