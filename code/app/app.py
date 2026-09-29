from __future__ import annotations

import queue
import sys
import threading
from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fairfl.config import RunConfig
from fairfl.train.engine import RunResult, run

st.set_page_config(page_title="fairfl", layout="centered")
st.title("Centralised vs Federated CIFAR10")

if "worker" not in st.session_state:
    st.session_state.worker = None
    st.session_state.stop_event = None
    st.session_state.out_queue = None
    st.session_state.cfg = None
    st.session_state.rounds_seen = []
    st.session_state.accs = []
    st.session_state.result = None

with st.sidebar:
    st.header("Hyperparameters")
    num_clients = st.number_input("num_clients (0 = centralised)", min_value=0, max_value=100, value=10)
    rounds = st.number_input("rounds", min_value=1, max_value=500, value=30)
    local_epochs = st.number_input("local_epochs", min_value=1, max_value=50, value=5)
    alpha = st.number_input("alpha (Dirichlet)", min_value=0.01, max_value=100.0, value=0.5)
    lr = st.number_input("lr", min_value=1e-5, max_value=1.0, value=0.1, format="%.4f")
    weight_decay = st.number_input("weight_decay", min_value=0.0, max_value=1e-2, value=5e-4, format="%.5f")
    batch_size = st.number_input("batch_size", min_value=1, max_value=1024, value=64)
    seed = st.number_input("seed", min_value=0, value=0)
    device = st.selectbox("device", ["auto", "cpu", "cuda"])

    running = st.session_state.worker is not None and st.session_state.worker.is_alive()
    col1, col2 = st.columns(2)
    run_btn = col1.button("Run training", type="primary", disabled=running)
    stop_btn = col2.button("Stop", disabled=not running)


def _worker(cfg: RunConfig, stop_event: threading.Event, out_q: queue.Queue):
    for item in run(cfg, stop_event):
        out_q.put(item)
    out_q.put(None)  # sentinel: worker done


if run_btn and not running:
    cfg = RunConfig(
        num_clients=num_clients, rounds=rounds, local_epochs=local_epochs, alpha=alpha,
        lr=lr, weight_decay=weight_decay, batch_size=batch_size, seed=seed, device=device,
    )
    st.session_state.cfg = cfg
    st.session_state.stop_event = threading.Event()
    st.session_state.out_queue = queue.Queue()
    st.session_state.rounds_seen = []
    st.session_state.accs = []
    st.session_state.result = None
    st.session_state.worker = threading.Thread(
        target=_worker, args=(cfg, st.session_state.stop_event, st.session_state.out_queue), daemon=True,
    )
    st.session_state.worker.start()
    st.rerun()

if stop_btn and st.session_state.stop_event is not None:
    st.session_state.stop_event.set()
    st.info("stop requested, finishing current step...")

cfg = st.session_state.cfg
if cfg is not None:
    mode = "centralised" if cfg.is_centralised else f"federated ({cfg.num_clients} clients)"
    st.write(f"Mode: **{mode}** | rounds={cfg.rounds} alpha={cfg.alpha} lr={cfg.lr}")

    progress = st.progress(min(len(st.session_state.rounds_seen) / cfg.rounds, 1.0))
    status = st.empty()
    chart_area = st.empty()

    # drain whatever the worker has produced since the last rerun
    q = st.session_state.out_queue
    if q is not None:
        try:
            while True:
                item = q.get_nowait()
                if item is None:
                    continue
                if isinstance(item, RunResult):
                    st.session_state.result = item
                else:
                    st.session_state.rounds_seen.append(item.round)
                    st.session_state.accs.append(item.test_acc)
                    status.text(
                        f"round {item.round}/{cfg.rounds}  loss={item.train_loss:.4f}  test_acc={item.test_acc:.4f}  "
                        f"({item.elapsed_s:.1f}s/round, ETA {item.eta_s:.0f}s)"
                    )
        except queue.Empty:
            pass

    if st.session_state.rounds_seen:
        progress.progress(min(st.session_state.rounds_seen[-1] / cfg.rounds, 1.0))
        fig, ax = plt.subplots()
        ax.plot(st.session_state.rounds_seen, st.session_state.accs)
        ax.set_xlabel("round")
        ax.set_ylabel("global test accuracy")
        chart_area.pyplot(fig)
        plt.close(fig)

    still_running = st.session_state.worker is not None and st.session_state.worker.is_alive()
    if st.session_state.result is None and still_running:
        import time as _time
        _time.sleep(0.3)
        st.rerun()

    if st.session_state.result is not None:
        status.text("done.")
        result = st.session_state.result
        st.subheader("Per-client accuracy")
        clients = sorted(result.client_acc.keys())
        accuracies = [result.client_acc[c] for c in clients]
        fig, ax = plt.subplots()
        ax.bar([str(c) for c in clients], accuracies)
        ax.set_xlabel("client (0 = centralised/server-only)")
        ax.set_ylabel("test accuracy")
        ax.set_ylim(0, 1)
        st.pyplot(fig)
        plt.close(fig)
        st.dataframe({"client": clients, "accuracy": accuracies})
