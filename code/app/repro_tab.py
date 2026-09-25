"""Reproduction tab: published numbers vs ours (results/reproduction/published.csv joined with results/runs.jsonl),
plus per-round training curves of any run folder that has a rounds.jsonl."""

from __future__ import annotations

import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from fairfl.experiments.compare import compare

COLOURS = {"OK": "#2e7d32", "GAP": "#c62828", "NOT RUN": "#9e9e9e"}
STATUS_COLOURS = alt.Scale(domain=list(COLOURS), range=list(COLOURS.values()))


def _table() -> pd.DataFrame:
    df = pd.DataFrame(compare())
    df["published"] = pd.to_numeric(df["published"])
    df["ours"] = pd.to_numeric(df["ours"], errors="coerce")
    df["method"] = df["run_name"]
    return df


def overview(df: pd.DataFrame) -> None:
    counts = df.groupby(["paper", "status"]).size().reset_index(name="values")
    chart = alt.Chart(counts).mark_bar().encode(
        y=alt.Y("paper:N", title=None, sort="ascending"),
        x=alt.X("values:Q", title="published values", stack="zero"),
        color=alt.Color("status:N", scale=STATUS_COLOURS, legend=alt.Legend(orient="top")),
        tooltip=["paper", "status", "values"],
    )
    st.altair_chart(chart, width="stretch")
    st.caption("OK = within 2% relative (or 0.01 / 1 point absolute for rates) of the published value; "
               "GAP = outside; NOT RUN = no matching run in results/runs.jsonl yet.")


def paper_view(df: pd.DataFrame, paper: str) -> None:
    sub = df[df["paper"] == paper]
    settings = sorted(sub["setting"].unique())
    setting = st.selectbox("Setting", ["all"] + settings, key="repro_setting")
    if setting != "all":
        sub = sub[sub["setting"] == setting]
    long = sub.melt(id_vars=["method", "metric", "status"], value_vars=["published", "ours"],
                    var_name="source", value_name="value").dropna(subset=["value"])
    if long.empty:
        st.info("Nothing run yet for this selection.")
    else:
        # one small panel per metric (metrics have different scales), paper vs ours side by side per run
        chart = alt.Chart(long).mark_bar().encode(
            x=alt.X("source:N", title=None, axis=alt.Axis(labels=False, ticks=False)),
            y=alt.Y("value:Q", title=None),
            color=alt.Color("source:N", scale=alt.Scale(domain=["published", "ours"], range=["#90a4ae", "#1565c0"]),
                            legend=alt.Legend(orient="top", title=None)),
            column=alt.Column("method:N", title=None, header=alt.Header(labelAngle=-30, labelAlign="right")),
            row=alt.Row("metric:N", title=None),
            tooltip=["method", "metric", "source", alt.Tooltip("value:Q", format=".4f"), "status"],
        ).properties(width=38, height=110).resolve_scale(y="independent")
        st.altair_chart(chart)
    show = sub[["method", "metric", "published", "ours", "diff", "status", "source"]]
    st.dataframe(show.style.map(lambda s: f"color: {COLOURS.get(s, '')}", subset=["status"]),
                 hide_index=True, width="stretch")


@st.cache_data(show_spinner=False)
def _curve(path: str, mtime: float) -> pd.DataFrame:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            gt = r.get("global_test") or {}
            rows.append({"round": r["round"] + 1, "accuracy (clients)": r.get("global_accuracy"),
                         "accuracy (global test)": gt.get("accuracy"), "loss": r.get("global_loss")})
    return pd.DataFrame(rows)


def curves(runs_root: Path) -> None:
    files = sorted(runs_root.glob("**/rounds.jsonl"))
    if not files:
        st.info(f"No rounds.jsonl under {runs_root}.")
        return
    labels = {str(p.parent.relative_to(runs_root)): p for p in files}
    picked = st.multiselect("Runs", list(labels), default=[k for k in labels if k.startswith("repro_")][:4])
    if not picked:
        return
    frames = []
    for k in picked:
        d = _curve(str(labels[k]), labels[k].stat().st_mtime)
        frames.append(d.melt(id_vars="round", var_name="metric", value_name="value").assign(run=k))
    long = pd.concat(frames).dropna(subset=["value"])
    metric = st.radio("Metric", sorted(long["metric"].unique()), horizontal=True)
    chart = alt.Chart(long[long["metric"] == metric]).mark_line().encode(
        x="round:Q", y=alt.Y("value:Q", title=metric), color=alt.Color("run:N", legend=alt.Legend(orient="bottom")),
        tooltip=["run", "round", alt.Tooltip("value:Q", format=".4f")])
    st.altair_chart(chart, width="stretch")


def render() -> None:
    df = _table()
    st.subheader("Reproduction status")
    overview(df)
    st.subheader("Paper vs ours")
    paper = st.selectbox("Paper", sorted(df["paper"].unique()), key="repro_paper")
    paper_view(df, paper)
    st.subheader("Training curves")
    root = st.text_input("Runs folder (local runs/, or an unzipped Colab runs/ folder)", "runs")
    curves(Path(root))
