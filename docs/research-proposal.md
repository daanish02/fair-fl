---
title: "Research Proposal: An Agentic Runtime Governor for Trajectory-Aware Group Fairness Enforcement in Federated Learning"
status: draft
target_venue: "Workshop-tier venue (e.g. ICLR 2027 workshop) or Q1 journal, TBD after Dr. Gupta scoping conversation"
author: Danish Ahmed
date: 2026-09-21
---

# Research Proposal: An Agentic Runtime Governor for Trajectory-Aware Group Fairness Enforcement in Federated Learning

## 1. Problem

Federated Learning (FL) systems that enforce group fairness (e.g., Equal Opportunity Difference, EOD, or Statistical Parity Difference, SPD) rely on mechanisms whose behavior is fixed before training starts and re-evaluated only against the current round's data:

- **FairFed** (AAAI'23) recomputes each client's fairness gap Δ_k^t fresh every round and reweights aggregation using a single **static scalar budget β**, set once, before training.
- **FedAA** (AAAI'25) filters client updates using a **static percentage threshold M%**, and the authors explicitly avoid making M% adaptive because a naive learned policy is trivially gamed by colluding malicious clients (Module 3 of our notes on that paper).
- **pFedFair** (2025) personalizes fairness constraints via Moreau envelopes, but the balance parameters λ, η, γ are likewise **fixed offline hyperparameters**.

All three—and every other fairness mechanism across the 20 papers reviewed for this dissertation—share the same structural limitation: they evaluate fairness as an **instantaneous snapshot** at round t, using only that round's state. *Remembering to Be Fair* (Alamdari et al., ICML'24) proves formally that fairness in any sequential decision process is properly a property of the **full historical trace** τ_t = (s_1, a_1, ..., s_t), not the current state alone — a policy can look fair at every checkpoint while systematically favoring one client early and "correcting" only late, and instantaneous metrics cannot detect this. That paper calls the correctly-posed problem **non-Markovian** (the fairness judgment cannot be computed from the current state alone) and solves it by folding history into a bounded memory variable so a standard learner can still act on it.

**No FL fairness paper in this literature applies that framework.** Every existing mechanism also responds identically to every kind of fairness drift — a formula cannot distinguish a client's EOD drifting because of genuine non-stationary data shift (which should be tolerated and reweighted) from drift caused by a poisoning attack or a spurious one-round blip (which should be flagged or ignored). Static formulas are diagnosis-blind by construction.

## 2. Research Question

Can a runtime agent, reasoning over each client's **fairness trajectory** (not just its current-round metric) and available client metadata, diagnose *why* a client's group-fairness gap is drifting and select an appropriate response — outperforming static, instantaneous-snapshot fairness mechanisms (FairFed, FedAA) under non-stationary client data drift, without requiring a trained/learned policy?

## 3. Proposed Contribution

**System: the Server Governor Agent.** A single agent, co-located with the FL server, that replaces the static reweighting step in a FairFed-style pipeline. Each round it:

1. **Maintains state**: a per-client trajectory buffer of EOD/SPD values over the last *k* rounds (a bounded memory variable, not the full unbounded history — this keeps the "non-Markovian" fairness signal tractable, following the memory-augmentation approach in *Remembering to Be Fair*).
2. **Diagnoses**: given a client's trajectory shape (steady drift vs. spike-and-revert vs. sudden step change) plus lightweight client metadata (round-over-round data volume change, participation history), an LLM-driven reasoning step classifies the likely cause — e.g. *sustained non-IID distribution shift*, *transient noise*, or *anomalous/suspicious change* — using an explicit decision procedure, not a black-box RL policy.
3. **Responds**: based on the diagnosis, selects one of a small, fixed action set — increase/decrease that client's aggregation weight (a **per-client, time-varying β_k(t)**, generalizing FairFed's single global static β), leave the client's weight unchanged, or flag the client for exclusion from the round.

This is deliberately **not** a learned/RL policy (that is explicitly out of scope for this MSc phase — see Section 5.1 of the dissertation handoff, and Section 5 below). The agent's contribution is the diagnose-then-respond reasoning loop itself: a static formula reacts identically to every drift regardless of cause, by construction; separating causes and choosing among discrete responses is the part no existing mechanism in this literature does.

### 3.1 Architecture: Tool-Based, Extensible by Design

The agent's action set (Section 3, step 3) is implemented as a small set of discrete, independently callable **tools** — e.g. `get_client_trajectory(client_id)`, `adjust_weight(client_id, delta)`, `flag_client(client_id, reason)` — rather than as inline branching logic inside one monolithic reasoning function. The diagnosis step (step 2) selects which tool(s) to invoke based on its classification, the same pattern used by the ToolAgent proof-of-concept in the Agentic-FL note.

This is an implementation choice, not an evaluated claim: **only the fairness-diagnosis tool set is built and evaluated in this dissertation.** But building the agent against a generic tool interface from the start — rather than hardcoding fairness-specific logic into the agent's core loop — means the same runtime (trajectory buffer, diagnose-then-respond loop, LLM reasoning step) could later host additional tools for other FL concerns named across this literature but out of scope here, e.g. Byzantine-robustness filtering (RoFL, FedAA's M% threshold) or dynamic privacy-budget tuning (the R-DPCFL note's ε_select). This extensibility is named explicitly as **future work** (Section 8 of the dissertation handoff's open-items list is the natural place to track it going forward) — it is not something this dissertation builds, tests, or claims as a contribution, precisely to keep the evaluated scope narrow and defensible per Section 5.1 below.

### 3.2 Baselines and Evaluation

- **Baselines**: FedAvg (no fairness mechanism), FairFed (static global β), q-FedAvg or FedAA-style static threshold filtering.
- **Fairness metrics**: per-client and global EOD/SPD, tracked round-over-round (not just final-round).
- **Drift scenarios to construct**: (a) sustained non-IID shift injected mid-training on a subset of clients, (b) a transient one-round anomaly on an otherwise well-behaved client, (c) a genuinely malicious client with adversarially manipulated local metrics, to test whether the diagnosis step avoids being fooled the way FairFed's honest-reporting assumption is (Module 5, FairFed note). Scenario (c) is evaluated only against the fairness-diagnosis tool set's behavior (does it correctly flag rather than reweight); it is not a claim about Byzantine-robustness as a solved problem — that remains future work per Section 3.1.
- **Utility cost**: global model accuracy under each scenario, to confirm fairness gains aren't bought by destroying top-line performance — matching the utility-tradeoff framing FairFed and FedAA both report.

## 4. Why This Fits the Dissertation's Positioning

Per the dissertation handoff, the top priority is a rigorous, standalone research contribution — the PhD-application angle must not leak into the framing. This proposal is scoped so the contribution stands on its own merit:

- It closes a gap explicitly left open by name in three papers already in the literature review (FairFed Module 5, FedAA Module 5, pFedFair Module 5), not a gap invented after the fact.
- It uses the exact "does this need an agent?" test the handoff already settled: a plain formula cannot diagnose *cause* before responding; this system's entire contribution is that diagnostic step.
- It matches the two-stage MSc→PhD research programme already agreed with Dr. Gupta framing: rule-based/reasoning agent now, learned/RL policy as explicit, named future work — not smuggled in as a promise about this phase.
- The tool-based architecture (Section 3.1) keeps a credible, honest growth story for the PhD-application conversations (Section 2 of the handoff) — "this runtime is designed to host more tools" is an architectural fact, not a claim the dissertation needs to prove — without that story ever entering the dissertation's own evaluated claims.

## 5. Feasibility

### 5.1 Scope constraints (deliberate, not limitations)

- **Rule-based agent, not RL.** The agent's diagnosis step is an LLM call (or a small decision-procedure the LLM output is checked against) over a compact per-client feature vector — trajectory shape summary statistics, participation/data-volume deltas — not a trained policy network. This avoids the RL sample-complexity and reward-design risk that FedAA's authors flag as a reason they kept M% static, and matches the one-semester timeline.
- **Server-only, single-agent.** No client-side "Guardian Agent" tier in this phase (that two-tier design recurs across the notes' dissertation-bridge sections but roughly doubles implementation and evaluation surface). Clients remain standard FL clients; only the server-side reweighting step is replaced.
- **Bounded memory, not unbounded history.** The trajectory buffer is a fixed sliding window (e.g., last 10 rounds), not the full training history — keeps the state space small and avoids the "infinite state expansion for non-regular status functions" failure mode *Remembering to Be Fair* itself flags as an open limitation (Module 6 of that note).

### 5.2 Implementation plan

- **FL substrate**: Flower (already the framework used in the Agentic-FL and Helmsman papers' proofs-of-concept, so tooling precedent exists in this exact literature) with a custom server-side `Strategy` that intercepts the standard weighted-aggregation step.
- **Datasets**: Adult and COMPAS under Dirichlet non-IID partitioning (α sweep), matching FairFed's and pFedFair's own evaluation setup — this lets results be compared directly against numbers already reported in those papers rather than requiring a from-scratch baseline reproduction.
- **Agent reasoning**: a small number of LLM calls per round (one per flagged client, not one per client per round, to control cost) — following the OpenFedLLM/ToolAgent cost-scaling lesson from the Agentic-FL note (raw-context approaches don't scale with client count; targeted, filtered queries do).
- **Diagnosis logic**: start with a transparent, inspectable decision procedure (e.g., trajectory slope + volatility + metadata deltas → classification), with the LLM used to interpret ambiguous/borderline cases rather than for every decision — keeps the system auditable and keeps a fallback path if LLM costs or latency become a problem.
- **Tool interface**: define the action set (Section 3.1) as a small, documented function-call schema (name, arguments, return type) from the start, even though only the fairness tools are implemented this semester — this is a few hours of upfront interface design, not a scope increase, and is what makes the extensibility claim in Section 3.1/4 honest rather than aspirational.

### 5.3 Main technical risk and de-risking plan

- **Risk**: the diagnosis step must reliably distinguish "legitimate drift" from "adversarial/anomalous" using only aggregate metrics and metadata the server can see (no raw client data) — this is the same information-scarcity problem FairFed's own limitations section names (Module 5: FairFed assumes honest client reporting).
- **De-risk plan**: before committing to the full multi-scenario evaluation, run a small spike — two clients, one with injected genuine non-IID drift and one with an injected single-round anomaly, over ~20 rounds — and confirm the diagnosis step can tell them apart using only the trajectory + metadata signal, before scaling to the full client population and adversarial scenario.
- **Fallback if diagnosis proves unreliable**: degrade gracefully to a non-Markovian-but-non-diagnostic version (dynamically adjust β_k(t) from trajectory shape alone, no cause classification) — still a genuine improvement over FairFed's static global β, and still defensible as a standalone contribution if the full diagnose-then-respond loop doesn't hold up empirically.

## 6. Tooling

- **FL framework**: Flower.
- **Models/datasets**: Adult, COMPAS (tabular, matches FairFed/pFedFair exactly); CelebA/UTKFace as a stretch goal if vision-domain comparison against pFedFair's frozen-embedding results is wanted.
- **Agent reasoning backend**: a single LLM provider to start (cost/latency predictability), multi-provider comparison only if time permits — this dissertation's core claim doesn't depend on cross-model comparison the way the security-track proposal in the sibling project does.

## 7. Timeline (indicative — to be finalized after Dr. Gupta scoping conversation)

| Phase                                                    | Window   |
| --------------------------------------------------------- | -------- |
| Literature review write-up (20 papers already read)        | Done     |
| Feasibility spike (2-client drift-vs-anomaly diagnosis)    | ~1 week  |
| Implement Server Governor Agent + trajectory buffer        | ~2 weeks |
| Baseline reproduction (FedAvg, FairFed, FedAA-style)       | ~1 week  |
| Full evaluation across drift scenarios (a/b/c in §3.2)     | ~2 weeks |
| Write dissertation chapters + prepare paper draft          | ~2 weeks |
| Buffer / revision                                          | ~1 week  |

## 8. Open Question for Discussion

Whether the "rule-based decision procedure + LLM only for ambiguous cases" split in Section 5.2 is the right balance for the dissertation's positioning — a purer "LLM reasons over everything" design is closer to the notes' original Agentic-FL framing and may read as more novel to a PhD supervisor audience, but the hybrid design is lower-risk and more auditable for the dissertation's own evaluation. Worth settling before implementation starts, and explicitly before the Dr. Gupta conversation, per the handoff's instruction to lock dissertation scope with him first.
