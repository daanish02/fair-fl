"""FairRFL (Augello, Gupta, Lo Re, Das, IEEE TETC 2026) Table III on CIFAR-10, ported line by line from the authors'
experiments.py / utils.py / selfishness.py / create_datasets.py / Network.py (github.com/ndslab-group/FairRFL) so the
published numbers can be reproduced. The RNG call order of the original is kept (seed, 51 models, train pool, test
pool, then per-client DataLoader shuffles during training), so a run follows the same random stream.

Kept on purpose (see docs/repro/fairrfl.md):
- SGD(momentum 0.9, wd 5e-4) re-created for every mini-batch, so momentum never accumulates;
- batch = min(256, size) = 200: one full-batch step per local epoch; the reported loss is the sum over epochs;
- 2-class split: randperm(100, seed 0) % 10 pairs; each client takes the first `size` samples of its classes from one
  shuffled pool of 50 * size samples (clients with the same pair get identical data);
- q-FFL as coded: models are set to g + L^q (w - g) / lr and the mean is rescaled by clients / sum(h);
- `prev_models` is captured right after the global model is sent, so the selfish clients' "previous delta" is the
  previous global delta;
- selfish clients 0..S-1 act from round index 2 on, after q-FFL rescaling; losses are not inflated;
- rotation2 (RFL-Self): MAD rule on update norms, 10-step bisection towards the coordinate median;
- FedCDA as coded (K = B = 3, L = 1e-3, all clients, mean of the caches plus the selected models).

Choices the paper and code leave open (logged in each result's config): q (not stated; default 1.0) and the selfish
variant (default `estimate_k=True`, the Gamma-estimating attack of Eq. 4-5 that Fig. 6 evaluates).

Usage: uv run python -m fairfl.experiments.fairrfl_official [--only NAME ...] [--workers 3]
CPU is too slow (about 100 s per round, 50 min per run); run it on a GPU (Colab notebook).
"""

from __future__ import annotations

import argparse
import copy
import datetime
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # as utils.py


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.activation = nn.ReLU()
        self.conv1 = nn.Conv2d(3, 64, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 64, 5)
        self.fc1 = nn.Sequential(nn.Dropout(0.5), nn.Linear(64 * 5 * 5, 120))
        self.fc2 = nn.Linear(120, 64)
        self.fc3 = nn.Linear(64, 10)
        gain = nn.init.calculate_gain("relu")
        for layer in (self.conv1, self.conv2, self.fc2, self.fc3):
            nn.init.xavier_uniform_(layer.weight, gain=gain)

    def forward(self, x):
        x = self.pool(self.activation(self.conv1(x)))
        x = self.pool(self.activation(self.conv2(x)))
        x = x.view(-1, 64 * 5 * 5)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        return self.fc3(x)


def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_w(model):
    return np.concatenate([p.detach().cpu().numpy().ravel() for p in model.parameters()])


def set_w(model, w):
    off = 0
    for p in model.parameters():
        n = int(np.prod(p.size()))
        p.data.copy_(torch.from_numpy(np.asarray(w[off : off + n]).reshape(p.size())))
        off += n


# ---------------------------------------------------------------- data (create_datasets.py)

def non_iid_split(dataset, nb_nodes, n_per_node, batch_size, shuffle=True):
    splits = max(10, 2 * nb_nodes)
    digits = torch.randperm(splits, generator=torch.Generator().manual_seed(0)) % 10
    digits_split, i = [], 0
    for n in range(nb_nodes, 0, -1):
        inc = int((splits - i) / n)
        digits_split.append(digits[i : i + inc])
        i += inc
    loader = torch.utils.data.DataLoader(dataset, batch_size=int(nb_nodes * n_per_node), shuffle=shuffle)
    images, labels = next(iter(loader))
    out = []
    for i in range(nb_nodes):
        idx = torch.stack([y_ == labels for y_ in digits_split[i]]).sum(0).bool()
        ds = torch.utils.data.TensorDataset(images[idx][:n_per_node], labels[idx][:n_per_node])
        out.append(torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle))
    return out


def get_dataset(root, n_train, n_test, n_clients, batch_size):
    from torchvision import datasets, transforms

    tf = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    train = datasets.CIFAR10(root=root, train=True, download=True, transform=tf)
    test = datasets.CIFAR10(root=root, train=False, download=True, transform=tf)
    return (non_iid_split(train, n_clients, n_train, batch_size),
            non_iid_split(test, n_clients, n_test, batch_size))


# ---------------------------------------------------------------- training (utils.py)

def distributed_training(models, dls, epochs, lr):
    crit = nn.CrossEntropyLoss()
    losses = [0.0] * len(models)
    for i, m in enumerate(models):
        m.train()
        running = 0.0
        for _ in range(epochs):
            for x, y in dls[i]:
                x, y = x.to(DEVICE), y.to(DEVICE)
                opt = optim.SGD(m.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
                opt.zero_grad()
                loss = crit(m(x), y)
                loss.backward()
                running += loss.item()
                opt.step()
        losses[i] = running
    return losses


def model_accuracy(model, dls):
    model.eval()
    accs = []
    with torch.no_grad():
        for dl in dls:
            correct = total = 0
            for x, y in dl:
                x, y = x.to(DEVICE), y.to(DEVICE)
                correct += int((model(x).argmax(1) == y).sum())
                total += len(y)
            if total:
                accs.append(correct / total * 100)
    return accs


# ---------------------------------------------------------------- selfish clients (selfishness.py)

class KEstimation(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.weights = nn.Parameter(torch.tensor(params))
        self.k = nn.Parameter(0.020 * torch.eye(1))

    def forward(self, selfish_delta):
        clients = torch.exp(100 * self.k)
        return (selfish_delta + (clients - 1) * self.weights) / clients


def selfish_training2(model, g_model, prev_model, prev_g, pprev_model, pprev_g, phi, budget=50):
    m, g = get_w(model), get_w(g_model)
    legit = m - g
    mp, gp, mpp, gpp = get_w(prev_model), get_w(prev_g), get_w(pprev_model), get_w(pprev_g)
    global_delta, prev_global_delta = g - gp, gp - gpp
    est = KEstimation(global_delta)
    opt = torch.optim.SGD(lr=0.05, params=est.parameters())
    x = torch.tensor(np.array([mp - gp, mpp - gpp]))
    y = torch.tensor(np.array([global_delta, prev_global_delta]))
    for _ in range(budget):
        x, y = torch.flip(x, [0]), torch.flip(y, [0])
        loss = F.mse_loss(est(x), y).sqrt()
        loss.backward()
        opt.step()
        opt.zero_grad()
    clients = torch.exp(100 * est.k).item() ** 2
    other = est.weights.detach().numpy() * (clients - 1)
    no, ng = np.linalg.norm(other), np.linalg.norm(global_delta)
    delta = legit * (no / ng) * phi - phi * other + (1 - phi) * other * ng / no
    set_w(model, g + delta)


def selfish_training(model, g_model, prev_model, prev_g, clients, phi):
    m, g, mp, gp = get_w(model), get_w(g_model), get_w(prev_model), get_w(prev_g)
    legit, global_delta = m - g, g - gp
    other = global_delta * clients - (mp - gp)
    delta = (legit * clients - other) * phi + (1 - phi) * other / (clients - 1)
    set_w(model, g + delta)


def rotation_aggregation2(models, g_model, tau):
    m = [get_w(x) for x in models]
    g = get_w(g_model)
    deltas = [w - g for w in m]
    norms = [np.linalg.norm(d) for d in deltas]
    median_norm = np.percentile(norms, 50)
    med = np.median(deltas, axis=0)
    mad = 1.4826 * np.median(np.abs(np.array(norms) - median_norm), axis=0)
    flagged = []
    for i in range(len(deltas)):
        if (norms[i] - median_norm) / mad > tau:
            flagged.append(i)
            lo, hi = 0.0, 1.0
            for _ in range(10):
                beta = (lo + hi) / 2
                rot = beta * deltas[i] + (1 - beta) * med
                if np.linalg.norm(rot) > median_norm:
                    hi = beta
                else:
                    lo = beta
            m[i] = g + rot
    return np.mean(m, axis=0), flagged


def fedcda_aggregate(state, models, global_w, losses, n):
    K, B, L = 3, 3, 1e-3
    if "cache" not in state:
        state["cache"] = [[global_w] * K for _ in range(n)]
        state["loss_cache"] = [[losses[i]] * K for i in range(n)]
    cache, lcache = state["cache"], state["loss_cache"]
    for i in range(n):
        cache[i].pop(0)
        cache[i].append(get_w(models[i]))
        lcache[i].pop(0)
        lcache[i].append(losses[i])
    idx = np.arange(n)
    np.random.shuffle(idx)
    batches = np.array_split(idx, B)
    sel = [cache[i][-1].copy() for i in range(n)]
    sel_l = [lcache[i][-1] for i in range(n)]
    for b, batch in enumerate(batches):
        fixed = np.concatenate(batches[:b]) if b > 0 else np.array([], dtype=int)
        for i in batch:
            best = (float("inf"), None, None)
            for c, cand in enumerate(cache[i]):
                ts, tl = sel.copy(), sel_l.copy()
                ts[i], tl[i] = cand, lcache[i][c]
                agg = np.concatenate((np.setdiff1d(np.arange(n), idx), fixed, batch))
                w = np.mean([ts[j] for j in agg], axis=0)
                obj = sum(tl[j] for j in agg) / len(agg) - (L / 2) * np.linalg.norm(w) ** 2
                if obj < best[0]:
                    best = (obj, cand, lcache[i][c])
            sel[i], sel_l[i] = best[1], best[2]
    allm = [w for i in range(n) for w in cache[i]] + sel
    return np.mean(allm, axis=0)


# ---------------------------------------------------------------- experiment (experiments.py)

def run(aggregation="fedavg", fairness=0.0, dq=False, selfish=0, selfishness=0.7, estimate_k=True, tau=2.5,
        clients=50, rounds=30, epochs=5, size=200, batch_size=256, lr=0.1, seed=1, persistence=1.0, root="data",
        tag=""):
    batch = min(batch_size, size)
    set_seed(seed)
    global_model = CNN().to(DEVICE)
    prev_g, pprev_g = copy.deepcopy(global_model), copy.deepcopy(global_model)
    models = [CNN().to(DEVICE) for _ in range(clients)]
    prev_models = [None] * clients
    train_dls, test_dls = get_dataset(root, size, 100, clients, batch)
    tp = fp = fn = tn = 0
    if fairness != 0:
        grads, deltas, hs = [None] * clients, [None] * clients, [None] * clients
        losses = [1] * clients
        dyn = [fairness] * clients
    cda_state: dict = {}
    for e in range(rounds):
        for m in models:
            m.load_state_dict(global_model.state_dict())
        pprev_models = [copy.deepcopy(m) for m in prev_models]
        prev_models = [copy.deepcopy(m) for m in models]
        if fairness != 0:
            for i in range(clients):
                dyn[i] = fairness * (losses[i] / np.median(losses)) ** (1 if dq else 0)
        losses = distributed_training(models, train_dls, epochs, lr)
        if fairness != 0:
            g = get_w(global_model)
            for i in range(clients):
                grads[i] = (get_w(models[i]) - g) / lr
                deltas[i] = np.float_power(losses[i] + 1e-10, dyn[i]) * grads[i]
                hs[i] = (dyn[i] * np.float_power(losses[i] + 1e-10, dyn[i] - 1) * np.linalg.norm(grads[i]) ** 2
                         + (1 / lr) * np.float_power(losses[i] + 1e-10, dyn[i]))
                set_w(models[i], g + deltas[i])
        round_selfish = []
        if e > 1:
            for s in range(selfish):
                if np.random.uniform() > persistence and e != rounds - 1:
                    continue
                round_selfish.append(s)
                if estimate_k:
                    selfish_training2(models[s], global_model, prev_models[s], prev_g, pprev_models[s], pprev_g,
                                      selfishness)
                else:
                    selfish_training(models[s], global_model, prev_models[s], prev_g, clients, selfishness)
        if aggregation == "fedavg":
            new = np.mean([get_w(m) for m in models], axis=0)
        elif aggregation == "median":
            new = np.median([get_w(m) for m in models], axis=0)
        elif aggregation == "fedcda":
            new = fedcda_aggregate(cda_state, models, get_w(global_model), losses, clients)
        elif aggregation == "rotation2":
            new, flagged = rotation_aggregation2(models, global_model, tau)
            if round_selfish:
                for s in round_selfish:
                    tp, fn = (tp + 1, fn) if s in flagged else (tp, fn + 1)
                fp += sum(1 for s in flagged if s not in round_selfish)
                tn += sum(1 for c in range(clients) if c not in round_selfish and c not in flagged)
        else:
            raise ValueError(aggregation)
        if fairness != 0:
            g = get_w(global_model)
            new = clients * (new - g) / np.sum(hs) + g
        pprev_g = copy.deepcopy(prev_g)
        prev_g = copy.deepcopy(global_model)
        set_w(global_model, new)
        if tag and (e + 1) % 10 == 0:
            print(f"[fairrfl] {tag} round {e + 1}/{rounds}", flush=True)
    acc = np.array(model_accuracy(global_model, test_dls))
    out = {"acc_mean": float(acc.mean()), "acc_n": float(acc[selfish:].mean()),
           "acc_s": float(acc[:selfish].mean()) if selfish else None, "acc_std": float(acc.std()),
           "jain": float(acc.sum() ** 2 / (len(acc) * (acc ** 2).sum())), "client_acc": acc.tolist()}
    if aggregation == "rotation2" and tp + fn:
        out |= {"det_precision": tp / (tp + fp) if tp + fp else None, "det_recall": tp / (tp + fn)}
    return out


# Table III rows (CIFAR-10): name -> (aggregation, q enabled, dq)
METHODS = {
    "fedavg": ("fedavg", False, False),
    "qffl": ("fedavg", True, False),
    "dqffl": ("fedavg", True, True),
    "median": ("median", False, False),
    "fedcda": ("fedcda", False, False),
    "rflself": ("rotation2", False, False),
    "qffl_rflself": ("rotation2", True, False),
    "fairrfl": ("rotation2", True, True),
}
SELFISH = {"0": 0, "10": 5, "20": 10, "30": 15}


def _job(args):
    name, kw = args
    torch.set_num_threads(kw.pop("threads", 2))
    t0 = time.time()
    res = run(**kw, tag=name)
    return name, kw, res, time.time() - t0


def main() -> None:
    import multiprocessing as mp

    from fairfl.experiments.batch import keep_awake
    from fairfl.experiments.runner import RESULTS_DIR

    ap = argparse.ArgumentParser()
    ap.add_argument("--q", type=float, default=1.0)
    ap.add_argument("--no-estimate-k", action="store_true")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--root", default="data")
    ap.add_argument("--rounds", type=int, default=30, help="Paper: 30. Lower only for a dry run (not recorded).")
    a = ap.parse_args()
    done = set()
    reg = RESULTS_DIR / "runs.jsonl"
    if reg.exists():
        done = {json.loads(line)["name"] for line in open(reg, encoding="utf-8")}
    jobs = []
    for method, (agg, use_q, dq) in METHODS.items():
        for pct, s in SELFISH.items():
            name = f"fairrfl_cifar10_{method}_s{pct}"
            if name in done or (a.only and not any(o in name for o in a.only)):
                continue
            kw = dict(aggregation=agg, fairness=a.q if use_q else 0.0, dq=dq, selfish=s, selfishness=0.7,
                      estimate_k=not a.no_estimate_k, root=a.root, threads=a.threads, rounds=a.rounds)
            jobs.append((name, kw))
    print(f"[fairrfl] {len(jobs)} runs to do; {a.workers} workers", flush=True)
    keep_awake(True)
    try:
        with mp.get_context("spawn").Pool(a.workers, maxtasksperchild=1) as pool:
            for name, kw, res, secs in pool.imap_unordered(_job, jobs):
                if a.rounds != 30:
                    print(f"[fairrfl] dry run {secs / 60:5.1f} min  {name}  acc_n={res['acc_n']:.2f}", flush=True)
                    continue
                row = {"time": datetime.datetime.now().isoformat(timespec="seconds"), "name": name, "seed": 1,
                       "strategy": "fairrfl_official", "dataset": "cifar10", "run_dir": None, "summary": res,
                       "config": kw}
                with open(reg, "a", encoding="utf-8") as f:
                    f.write(json.dumps(row) + "\n")
                print(f"[fairrfl] ok {secs / 60:5.1f} min  {name}  acc_n={res['acc_n']:.2f} "
                      f"acc_s={res['acc_s']} std={res['acc_std']:.2f}", flush=True)
    finally:
        keep_awake(False)


if __name__ == "__main__":
    main()
