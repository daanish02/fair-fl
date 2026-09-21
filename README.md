# Fairness in Federated Learning — An Agentic Runtime Governor for Trajectory-Aware Fairness Enforcement

This is a dissertation research project (MSc Software Systems) working toward a strong Master's dissertation and, from it, a paper submittable to a workshop or journal — with a secondary, downstream goal of supporting a PhD application (see `docs/meta/handoff.md` for how these two goals are kept separate).

## The problem

Federated Learning (FL) lets many devices or institutions train a shared model together without sharing raw data. Making that shared model *fair* — not systematically worse for some clients or demographic groups than others — is an active research area, but every fairness mechanism in the current literature (FairFed, FedAA, pFedFair, and others) shares the same weakness: they decide how much to correct for unfairness using a fixed rule set once before training starts, and they judge fairness fresh each round using only that round's numbers. None of them look at a client's fairness *trajectory* over time, and none of them ask *why* a client's fairness numbers are drifting before deciding what to do about it — a genuine, gradual data shift and a one-off blip get treated identically by a static formula.

## What this project is doing

- Reviewing the current literature on fairness in Federated Learning and on agentic/LLM approaches to FL (20 papers — 7 fairness-in-FL, 3 agentic/LLM-in-FL as the "landscape" set, plus 10 professor-specific papers read for PhD outreach purposes) to map out where the actual gap sits.
- Designing a **Server Governor Agent**: a runtime agent that tracks each client's fairness trajectory over a rolling window of training rounds, diagnoses the likely cause of any drift (genuine distribution shift vs. a transient anomaly vs. suspicious/adversarial behavior), and only then chooses a response — adjusting that client's aggregation weight, leaving it alone, or flagging it.
- Deliberately scoping this as a rule-based/reasoning agent for the Master's phase (not a trained reinforcement-learning policy) — feasible in one semester, and a genuine research contribution since no empirical implementation of this kind exists yet in the literature reviewed.
- Evaluating it against static baselines (FedAvg, FairFed, FedAA-style filtering) on standard fairness benchmarks (Adult, COMPAS) under injected non-stationary client drift.

## Project materials

- `docs/meta/handoff.md` — full background, priorities, and planning context for this dissertation and its parallel PhD-application workstream.
- `docs/research-proposal.md` — the research proposal: problem, research question, proposed contribution, and technical feasibility plan.
- `docs/notes/` — literature review notes on all 20 source papers (fairness-in-FL landscape, agentic/LLM-in-FL landscape, and professor-specific papers).
- `docs/papers/` — the source papers themselves.

For a sibling project following the same documentation approach — literature review notes, a research proposal, and a handoff file — see the `agentic ai security` project's own `README.md` (path: `../agentic ai security/README.md`), which is working toward an IEEE S&P 2027 submission on LLM agent tool-chaining attacks.

## Status

Literature review (20 papers) is complete and notes are written up. The research proposal (problem framing, proposed Server Governor Agent design, and feasibility plan) is drafted and pending a scoping conversation with the dissertation supervisor before implementation begins.
