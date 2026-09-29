"""FL fairness testbed dashboard: run any registered strategy, compare saved runs under a fairness scheme, and
see how the paper reproductions compare with the published numbers (Reproduction tab).

Launch from code/:  uv run python -m streamlit run app/dashboard.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import altair as alt
import pandas as pd
import streamlit as st
from pydantic import ValidationError

import dash_helpers as h
import repro_tab
from fairfl.core.engine import Simulator
from fairfl.core.registry import get_strategy, list_strategies
from fairfl.fairness.scheme import FairnessScheme

st.set_page_config(page_title="fairfl dashboard", layout="wide")


def acc_chart(df: pd.DataFrame) -> alt.Chart:
    return alt.Chart(df).mark_line(point=True).encode(
        x=alt.X("round:Q"), y=alt.Y("accuracy:Q", scale=alt.Scale(zero=False)), color="run:N",
        tooltip=["run", "round", alt.Tooltip("accuracy:Q", format=".4f")],
    ).properties(height=320)


def unfair_chart(df: pd.DataFrame, epsilon: float) -> alt.Chart:
    base = alt.Chart(df).encode(x="round:Q", color="run:N")
    line = base.mark_line().encode(y=alt.Y("u:Q", title="unfairness u"),
                                   tooltip=["run", "round", alt.Tooltip("u:Q", format=".4f")])
    evald = base.transform_filter("datum.evaluated").mark_point(size=40).encode(y="u:Q")
    viol = alt.Chart(df).transform_filter("datum.violation").mark_point(
        shape="cross", size=120, color="red", filled=True).encode(x="round:Q", y="u:Q",
                                                                  tooltip=["run", "round", "u"])
    eps = alt.Chart(pd.DataFrame({"eps": [epsilon]})).mark_rule(strokeDash=[4, 4], color="gray").encode(y="eps:Q")
    return (line + evald + viol + eps).properties(height=320)


def param_form(strategy: str, key: str) -> dict:
    values = {}
    for spec in h.field_specs(get_strategy(strategy).Params):
        k, d, hp = f"{key}_{spec['name']}", spec["default"], spec["help"]
        if spec["kind"] == "bool":
            values[spec["name"]] = st.checkbox(spec["name"], value=bool(d), help=hp, key=k)
        elif spec["kind"] == "int":
            values[spec["name"]] = st.number_input(spec["name"], value=int(d), step=1, help=hp, key=k)
        elif spec["kind"] == "float":
            values[spec["name"]] = st.number_input(spec["name"], value=float(d), format="%.4g", help=hp, key=k)
        elif spec["kind"] == "choice":
            ch = spec["choices"]
            values[spec["name"]] = st.selectbox(spec["name"], ch, index=ch.index(d) if d in ch else 0, help=hp, key=k)
        else:
            values[spec["name"]] = h.parse_text_value(st.text_input(spec["name"], value=str(d), help=hp, key=k))
    return values


def scheme_form(defaults: FairnessScheme) -> FairnessScheme | None:
    s = defaults.status
    benefit = st.selectbox("benefit", ["dp", "tpr", "fpr", "eo", "accuracy"],
                           index=["dp", "tpr", "fpr", "eo", "accuracy"].index(s.benefit))
    level = st.selectbox("level", ["global", "local_max", "local_mean", "clients"],
                         index=["global", "local_max", "local_mean", "clients"].index(s.level),
                         help="'clients' compares client accuracies and ignores the benefit.")
    accumulation = st.selectbox("accumulation", ["instant", "cumulative", "window", "discounted"],
                                index=["instant", "cumulative", "window", "discounted"].index(s.accumulation))
    window = st.number_input("window", min_value=1, value=s.window, disabled=accumulation != "window")
    discount = st.number_input("discount", min_value=0.01, max_value=1.0, value=s.discount,
                               disabled=accumulation != "discounted")
    scope = st.selectbox("scope", ["long_term", "periodic", "anytime", "bounded"],
                         index=["long_term", "periodic", "anytime", "bounded"].index(defaults.scope.kind))
    period = st.number_input("period", min_value=1, value=defaults.scope.period,
                             help="Used by the periodic scope (also in the compare table).")
    rounds_txt = st.text_input("bounded rounds", value=",".join(map(str, defaults.scope.rounds)),
                               help="Comma-separated round indices for the bounded scope.")
    deploy_every = st.number_input("deploy_every", min_value=1, value=defaults.deploy_every)
    epsilon = st.number_input("epsilon", min_value=0.0, value=defaults.epsilon, format="%.4f")
    try:
        rounds = [int(x) for x in rounds_txt.replace(" ", "").split(",") if x]
        return h.build_scheme(benefit, level, accumulation, int(window), float(discount), scope, int(period),
                              rounds, int(deploy_every), float(epsilon))
    except (ValueError, ValidationError) as e:
        st.error(f"Invalid scheme: {e}")
        return None


# ---------------- sidebar ----------------
bases = h.base_configs()
with st.sidebar:
    st.header("Experiment")
    base_path = st.selectbox("base config", bases, format_func=lambda p: p.name,
                             index=next((i for i, p in enumerate(bases) if "synthetic" in p.name), 0))
    base = h.load_base(base_path)
    bk = base_path.stem
    d, t = base["data"], base["train"]
    data = {
        "name": st.selectbox("dataset", ["adult", "mnist", "synthetic"],
                             index=["adult", "mnist", "synthetic"].index(d["name"]), key=f"{bk}_ds"),
        "num_clients": st.number_input("num_clients", min_value=2, value=d["num_clients"], key=f"{bk}_nc"),
        "alpha": st.number_input("Dirichlet alpha", min_value=0.001, value=float(d["alpha"]), format="%.3f",
                                 key=f"{bk}_alpha"),
        "partition_on": st.selectbox("partition_on", ["label", "sensitive", "joint", "iid"],
                                     index=["label", "sensitive", "joint", "iid"].index(d["partition_on"]),
                                     key=f"{bk}_part"),
        "val_fraction": st.number_input("val_fraction", min_value=0.0, max_value=0.9, value=float(d["val_fraction"]),
                                        help="LoGoFair needs > 0 (e.g. 0.2).", key=f"{bk}_val"),
    }
    train = {
        "rounds": st.number_input("rounds", min_value=1, value=t["rounds"], key=f"{bk}_rounds"),
        "clients_per_round": st.number_input("clients_per_round (fraction)", min_value=0.01, max_value=1.0,
                                             value=float(t["clients_per_round"]), key=f"{bk}_cpr"),
        "lr": st.number_input("lr", min_value=1e-5, value=float(t["lr"]), format="%.4g", key=f"{bk}_lr"),
        "local_epochs": st.number_input("local_epochs", min_value=1, value=t["local_epochs"], key=f"{bk}_le"),
    }
    seed = st.number_input("seed", value=base["seed"], step=1, key=f"{bk}_seed")
    st.header("Fairness scheme")
    scheme = scheme_form(FairnessScheme.model_validate(base["fairness"]))

run_tab, cmp_tab, repro = st.tabs(["Run", "Compare", "Reproduction"])

# ---------------- run ----------------
with run_tab:
    strategies = list_strategies()
    c1, c2 = st.columns([1, 2])
    with c1:
        strategy = st.selectbox("Method", strategies, index=strategies.index(base["strategy"]["name"])
                                if base["strategy"]["name"] in strategies else 0)
        doc = (get_strategy(strategy).__doc__ or "").strip()
        if doc:
            st.caption(doc.splitlines()[0])
        st.subheader("Hyperparameters")
        raw_params = param_form(strategy, key=f"p_{strategy}")
        if not raw_params:
            st.caption("No hyperparameters.")
        params = None
        try:
            params = get_strategy(strategy).Params.model_validate(raw_params).model_dump()
        except ValidationError as e:
            st.error(str(e))
        run_name = st.text_input("run name", value=f"{data['name']}_{strategy}")
        if strategy == "logofair" and data["val_fraction"] <= 0:
            st.warning("LoGoFair needs val_fraction > 0.")
        go = st.button("Run", type="primary", disabled=params is None or scheme is None)
    with c2:
        status = st.empty()
        acc_ph, unf_ph = st.empty(), st.empty()

    if go:
        try:
            cfg = h.build_config(base, data=data, train=train, seed=int(seed), strategy=strategy,
                                 params=params, scheme=scheme, name=run_name)
        except ValidationError as e:
            st.error(str(e))
            st.stop()
        with c2:
            with st.spinner("Building federated data..."):
                sim = Simulator(cfg)
            logs = []

            def on_round(log):
                logs.append(log)
                status.write(f"round {log.round + 1}/{cfg.train.rounds}  acc={log.global_accuracy:.4f}  "
                             f"({log.seconds:.2f}s)")
                acc_ph.altair_chart(acc_chart(h.accuracy_frame(logs, run_name)), width="stretch")
                unf_ph.altair_chart(unfair_chart(h.trajectory_frame(logs, scheme, run_name), scheme.epsilon),
                                    width="stretch")

            sim.run(on_round=on_round)
            path = h.save_run(sim, sim.logs, h.new_run_dir(cfg))
        st.success(f"Saved to {h.run_label(path, h.CODE_DIR)}")
        st.dataframe(h.scope_table({run_name: sim.logs}, scheme), hide_index=True)

# ---------------- compare ----------------
with cmp_tab:
    runs = h.find_runs()
    labels = [h.run_label(p) for p in runs]
    picked = st.multiselect("Runs (folders under runs/ with rounds.jsonl)", labels)
    if st.button("Refresh run list"):
        st.rerun()
    if len(picked) < 2:
        st.info("Pick at least two runs.")
    elif scheme is None:
        st.error("Fix the fairness scheme in the sidebar.")
    else:
        loaded = {lab: h.load_run(h.RUNS_DIR / lab)[1] for lab in picked}
        st.caption(f"Scheme: benefit={scheme.status.benefit}, level={scheme.status.level}, "
                   f"accumulation={scheme.status.accumulation}, scope={scheme.scope.kind}, "
                   f"deploy_every={scheme.deploy_every}, epsilon={scheme.epsilon}")
        a, b = st.columns(2)
        with a:
            st.subheader("Global accuracy")
            st.altair_chart(acc_chart(pd.concat([h.accuracy_frame(v, k) for k, v in loaded.items()])),
                            width="stretch")
        with b:
            st.subheader("Unfairness trajectory")
            st.altair_chart(unfair_chart(pd.concat([h.trajectory_frame(v, scheme, k) for k, v in loaded.items()]),
                                         scheme.epsilon), width="stretch")
        st.subheader("Scope comparison")
        table = h.scope_table(loaded, scheme)
        metric = st.selectbox("metric", ["max", "mean", "final", "violation_rate"])
        wide = table.pivot(index="run", columns="scope", values=metric)
        wide["final_accuracy"] = table.groupby("run")["final_accuracy"].first()
        st.dataframe(wide.style.format("{:.4f}"))
        st.caption(f"Rank per scope by {metric} (1 = fairest)")
        st.dataframe(h.rank_table(table, metric))
        with st.expander("Full table"):
            st.dataframe(table, hide_index=True)

# ---------------- reproduction ----------------
with repro:
    repro_tab.render()
