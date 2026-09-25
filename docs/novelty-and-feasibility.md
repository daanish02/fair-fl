---
title: "Non-Markovian Group Fairness in Federated Learning: Novelty Sweep and Feasibility"
date: 2026-09-25
status: pre-build check
---

# Non-Markovian Group Fairness in Federated Learning: Novelty Sweep and Feasibility

**Direction:** non-Markovian (history-dependent) group fairness in federated learning (FL), enforced by an agentic server governor.

This document records the novelty sweep, the gap we claim, the go/no-go gates that decide whether the direction goes ahead, and the practical feasibility of the work.

---

## 1. The idea in plain terms

FL training is itself a sequential decision process. Every round the server takes actions: it sets aggregation weights, picks which clients take part, and chooses how strong the fairness correction is. Each round's global model can also be deployed. So the people served by those models (demographic groups at each client, and globally) experience a **history** of disparity, not just the disparity of the final model.

*Remembering to Be Fair* (Alamdari, Klassen, Creager, McIlraith, ICML 2024) formalises fairness over a history with a fairness scheme ⟨U, W_ex, B⟩:

- **Status function U(τ_t)**: maps the history τ_t up to time t to a per-stakeholder vector, for example the cumulative or windowed group benefit (or gap) each stakeholder has experienced so far.
- **Aggregator W_ex**: turns the sequence of judged status vectors into one fairness score.
- **Filter B**: decides which time points count, for example only rounds whose model was actually deployed.

It defines four **scopes**, which say when fairness is judged:

| Scope | When fairness is judged |
|---|---|
| Long-term | Once, at the end of the trace |
| Periodic | Every p steps |
| Anytime | Every step (periodic with p = 1) |
| Bounded | At triggered events selected by B |

**The vaccine example.** 80,000 vaccines go to countries A and B. One policy gives 40,000 to A in months 1-2 and 40,000 to B in months 3-4. At month 4 the split is equal, so long-term fairness calls it fair. But A benefited early while B faced higher infection during months 1-2. Anytime or periodic fairness catches this; long-term fairness does not.

**The FL analogue.** Fair-FL papers report fairness only for the final model, which is the long-term scope. A method can end with a small group gap while the models deployed along the way were strongly unfair to one group. Final-round reporting cannot see that interim harm.

---

## 2. Verdict

| Question | Answer |
|---|---|
| Gap | No peer-reviewed work instantiates a non-Markovian fairness scheme (status U, scopes, filter B) for **group** fairness over FL training rounds, or enforces one within a single training run. |
| Novelty risk | Moderate-low. The closest peer-reviewed work is centralised (Alamdari et al.), monitoring-only (Henzinger et al.), per-decision rather than per-round (Fairness Shields), or long-term fairness of *participation* rather than of group outcomes (Huang et al.). The closest arXiv work (Cumulative Utility Parity) is client-level and has no scopes. |
| What would close it | A peer-reviewed paper that (a) defines group fairness over the sequence of deployed FL models with explicit scopes, or (b) enforces a history-dependent fairness target during FL training. |
| What does not close it | Using history as a mechanism while keeping a per-round or final-model fairness target (FairFed, FCFL, FedCDA); client-level or participation-level long-term fairness; runtime monitors without enforcement; RL-based server agents with per-round rewards. |

---

## 3. Evidence

### 3.1 Peer-reviewed prior work

| Area | Work (venue) | Why it doesn't close the gap |
|---|---|---|
| Non-Markovian fairness theory | Alamdari et al., *Remembering to Be Fair* (ICML'24) | Centralised. Solved with RL (FairQCM), which needs many episodes; in FL one episode is a whole training run. |
| Runtime fairness monitoring | Henzinger et al., *Monitoring Algorithmic Fairness* (CAV'23); *Runtime Monitoring of Dynamic Fairness Properties* (FAccT'23) | Monitors a stream of decisions. No enforcement, no FL, no training. |
| Runtime fairness enforcement | *Fairness Shields* (AAAI'25) | Enforces group fairness over sequences of individual classifier decisions, not over federated training rounds or aggregation. |
| Group fairness in FL | FairFed (AAAI'23), LoGoFair (AAAI'25), FairWeight (IEEE TSC'26), FedFDP (ACNS'26) | Evaluated on the final model only. LoGoFair fixes fairness after training, so intermediate models stay unfair. |
| History-carrying FL mechanisms | FairFed recursive weights; FCFL accumulated queues (ECML-PKDD'24); FedCDA cross-round cache (ICLR'24) | Use history as a mechanism; the fairness target is still per-round or final. FCFL targets client accuracy only, not demographic groups. |
| Long-term fairness in FL | Huang et al., client selection with Lyapunov long-term fairness (IEEE TPDS'21) | Fairness of participation (being selected), not of group outcomes. |
| Group drift in FL | Group-Specific Distributed Concept Drift (IEEE TNNLS'25) | Adapts to drift; fairness scope is not formalised over history. |
| Agents in FL | FL-MAESTRO (IEEE GLOBECOM'26; energy/resource aggregation agent); Agentic FL reconfiguration (JSS'26) | No fairness objective, let alone non-Markovian fairness. |
| RL server agents for fairness | FedAA (AAAI'25) | Per-round reward and a trained policy. |

### 3.2 arXiv-only, cite as background

- *Cumulative Utility Parity* (arXiv 2602.13651): client-level long-term benefit under intermittent participation. Closest in spirit, but client-level rather than group-level, and with no scopes.
- *Federated Fairness Analytics* (arXiv 2408.08214): observes fairness fluctuating over rounds; measurement only.
- FedTSV (European Control Conf. '26): trajectory weighting for contribution fairness.
- Agentic-FL survey (arXiv 2604.04895), GuardFed, PFAttack.

---

## 4. Clarification: history as a mechanism vs. fairness judged over history

These are different things, and the earlier draft of the research proposal (`docs/research-proposal.md`) mixed them up by describing every existing mechanism as a per-round snapshot.

- **History as a mechanism.** Several FL methods carry state across rounds:
  - FairFed updates aggregation weights recursively, w_k^t = w_k^{t-1} - β(Δ_k - mean Δ);
  - FCFL keeps an accumulated unfairness queue Q_i per client;
  - FedCDA caches each client's last K local models and selects across rounds.

  The state is used to decide the next action. But what counts as success is still per-round or final-model fairness.
- **Fairness judged over history (the gap).** The *target* itself is defined on the trajectory: the status U accumulates what each group experienced from the deployed models, and a scope decides when that accumulated status must be fair. A method can use no memory at all and still be scored this way. Conversely, having memory does not make a method's objective non-Markovian.

The claim is therefore **not** "existing FL fairness methods have no memory". It is "no FL method defines or enforces group fairness over the history of deployed models". The proposal should be rewritten to say this.

---

## 5. Expected reviewer objections

**"Intermediate models are not deployed, so interim unfairness does not matter."**
Production FL systems push checkpoints on a schedule, not only at the end. The periodic scope together with the filter B (`deploy_every` in the testbed) models exactly this: only the rounds whose model is released count. With `deploy_every = T` the scheme reduces to long-term fairness, so the framework includes the usual evaluation as a special case rather than replacing it.

**"The governor is just a controller."**
Gate G2 tests this directly. The governor is compared against (i) a Markovian per-round controller, (ii) a memory controller on U (Lyapunov/queue, FCFL-style), and (iii) a non-LLM planner with the same tools (greedy/MPC over `what_if`). If the agent does not win on scope violations at comparable accuracy, the agentic claim is dropped.

**"RL already solves non-Markovian fairness."**
Alamdari et al. solve it with FairQCM, which learns over many episodes. In FL one episode is an entire training run, so collecting enough episodes is impractical. G2 includes FairQCM trained on K pilot runs to show this episode cost, and a FedAA-style RL baseline.

---

## 6. Go/no-go gates

**G1: does the problem exist in FL?**
Run all baselines × Dirichlet α ∈ {0.1, 0.5, 5} × Adult and MNIST (MNIST for client-level only) × 3-5 seeds, logging per-round local and global DP/EO and per-client accuracy.
- **Pass:** clear interim unfairness, plus a ranking change (Kendall τ between final-round and anytime/periodic rankings well below 1).
- **Fail:** the non-Markovian framing adds little in FL. Pivot to client-level cumulative fairness, or to cause-attribution.

**G2: does the agent earn its place?**
Governor vs:
1. Markovian per-round controller;
2. memory controller on U (Lyapunov/queue, FCFL-style);
3. non-LLM planner with the same tools (greedy/MPC over `what_if`);
4. FairQCM trained on K pilot runs, to show its episode cost;
5. FedAA-style RL.

Win condition: fewer scope violations at comparable accuracy, with a bounded number of LLM calls.

**G3 (secondary):** misreporting clients and one-round glitches don't break it.

G1 alone is a publishable empirical finding if it passes. G2 decides whether the agentic method is part of the contribution.

---

## 7. LLM choice

Models are taken from what agentic-FL papers actually used, so results are comparable.

| Model | Used by | Tool calling | Reasoning | Role |
|---|---|---|---|---|
| gpt-4o-mini | Agentic FL (ReAct, 5 tools) | yes | yes | primary |
| GPT-4.1-mini | FL-MAESTRO ($0.28 per 50-round run, LLM called every round) | yes | yes | primary |
| Qwen3-8B | Agentic FL (K-Agent, LangGraph + Ollama) | yes | yes (thinking) | open-weight main |
| Llama-3.1-8B | Agentic FL | yes | moderate | open-weight second |
| Llama-3.2-3B | Agentic FL | weak | weak | small-model lower bound |

Rationale:
- **Excluded:** GPT-5 and Claude (used only by Helmsman, and only for code synthesis); Qwen3.5-35B (needs 128 GB RAM).
- **Memory is our code**, not the model's: the status vector plus a short decision log go into the prompt. This keeps the non-Markovian state explicit and auditable.
- **Access** is through OpenRouter, one model name per config; Ollama is optional for models of 8B or smaller.
- **The LLM is called only when the deterministic monitor predicts a scope violation**, not every round. This bounds cost and makes the number of calls a reported metric.
- **Prompting ablation:** description-only, few-shot, and chain-of-thought, as in Agentic FL.

---

## 8. Feasibility

| Resource | Plan |
|---|---|
| Hardware | CPU-only Intel i5 laptop, 16 GB RAM |
| Scale | 10-20 clients, 50-100 rounds, 3-5 seeds; sequential simulation loop |
| Datasets | Adult (group fairness, sensitive attribute sex) and MNIST (client-level fairness only) |
| Models | Small MLP (Adult) and small CNN (MNIST) |
| LLM cost | Estimated under $30-50 for the full grid, because the LLM is called only on predicted violations (FL-MAESTRO reports $0.28 per 50-round run even when calling every round). A pilot confirms the estimate before the full grid. |
| Framework | Custom PyTorch/NumPy testbed, no Flower dependency. Project managed with uv (lock file, Python 3.12); configs and messages validated with pydantic v2. |

The testbed (`code/`) already runs FedAvg, Median, FairRFL, FedFair/FedFDP, FedCDA, FedMut, FCFL, LoGoFair, FairWeight, FairFed and q-FFL, and scores every run under all four scopes from per-round logs. That is what G1 needs. The governor, planners and the G1/G2 study scripts are not built yet.

---

## 9. Limitations of this sweep

- About 25 targeted searches, not a systematic review. No PRISMA protocol and no forward/backward citation chase for every paper.
- The 2025-2026 FL literature moves fast; a relevant paper may be in press or only on arXiv.
- Before submission: repeat the search on DBLP and Google Scholar (queries combining "federated" with "long-term fairness", "temporal fairness", "cumulative fairness", "non-Markovian", "anytime fairness", "fairness trajectory"), and check papers citing Alamdari et al. (ICML'24) and Henzinger et al. (CAV'23, FAccT'23).

---

## 10. Sources

The planning notes contain no verified web links, so none are given here; add DOIs or proceedings links once they have been checked. Citations and identifiers:

- Alamdari, Klassen, Creager, McIlraith. *Remembering to Be Fair: Non-Markovian Fairness in Sequential Decision Making.* ICML 2024. (Notes: `docs/notes/remembering-to-be-fair-non-markovian-fairness-in-sequential-decision-making.md`)
- Henzinger et al. *Monitoring Algorithmic Fairness.* CAV 2023.
- Henzinger et al. *Runtime Monitoring of Dynamic Fairness Properties.* FAccT 2023.
- *Fairness Shields.* AAAI 2025.
- Ezzeldin et al. *FairFed: Enabling Group Fairness in Federated Learning.* AAAI 2023. (Notes: `docs/notes/fairfed-enabling-group-fairness-in-federated-learning.md`)
- Zhang et al. *LoGoFair.* AAAI 2025.
- Kasyap, Atmaca, Maple, Lane. *FairWeight: Private Fairness-aware Aggregation using Model Interpretation based Weightage in Federated Learning.* IEEE TSC 2026.
- Ling et al. *FedFDP.* ACNS 2026.
- Wang et al. *FCFL: A Fairness Compensation-Based Federated Learning Scheme with Accumulated Queues.* ECML-PKDD 2024.
- Wang et al. *FedCDA: Federated Learning with Cross-Round Divergence-Aware Aggregation.* ICLR 2024.
- Huang et al. Client selection with Lyapunov long-term fairness. IEEE TPDS 2021.
- *Group-Specific Distributed Concept Drift.* IEEE TNNLS 2025.
- *FL-MAESTRO.* IEEE GLOBECOM 2026.
- Agentic FL reconfiguration. JSS 2026.
- *FedAA: A Reinforcement Learning Perspective on Adaptive Aggregation for Fair and Robust Federated Learning.* AAAI 2025. (Notes: `docs/notes/fedaa-a-reinforcement-learning-perspective-on-adaptive-aggregation-for-fair-and-robust-federated-learning.md`)
- *Agentic Federated Learning: The Future of Distributed Training Orchestration.* (Notes: `docs/notes/agentic-federated-learning-the-future-of-distributed-training-orchestration.md`)
- *Helmsman: Autonomous Synthesis of Federated Learning Systems via Collaborative LLM Agents.* (Notes: `docs/notes/helmsman-autonomous-synthesis-of-federated-learning-systems-via-collaborative-llm-agents.md`)
- Li, Sanjabi, Beirami, Smith. *Fair Resource Allocation in Federated Learning* (q-FFL). ICLR 2020.
- arXiv-only: *Cumulative Utility Parity* (arXiv 2602.13651); *Federated Fairness Analytics* (arXiv 2408.08214); Agentic-FL survey (arXiv 2604.04895); FedTSV (European Control Conf. 2026); GuardFed; PFAttack.
- Reference code used only to cross-check ports (GitHub repositories): `HMHelloWorld/FedMut`, `wlffffff/FCFL`, `ndslab-group/FairRFL`.
