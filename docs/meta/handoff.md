# Project Handoff: Fair Federated Learning Dissertation → PhD Application Pipeline

**Handoff date:** September 2026
**Handed off from:** Claude (claude.ai, project "Multifaceted Perspectives on Fairness in Federated Learning")
**Handed off to:** Claude Code
**Owner:** [Your name] — MSc Software Systems, BITS Pilani Dubai (GPA 9.34/10)

---

## 0. Read this first — how to use this document

This file is the single source of truth for everything done so far. Claude Code does not have access to the claude.ai conversation history that produced this — everything relevant has been consolidated here. The project directory (once set up) will also contain:

- The 10 "landscape" research papers (fairness in FL + agentic/LLM in FL) — as PDFs
- The 10 "professor-specific" papers (one or two per target PhD supervisor) — as PDFs
- Exported NotebookLM notes/reports (mind map, data table, briefing) if you export them before switching tools
- `dissertation_proposal.html` and `researcher_profile.html` — two interactive HTML artifacts already built (see Section 6)
- Four university research reports (Waterloo, McGill/Mila, University of Alberta, UofT/Vector) — as markdown/text, already summarized in Section 4 below

**If you (Claude Code) are asked to continue this work**, treat this document as context, not as a rigid script — the person's dissertation direction, supervisor conversations, and application strategy will keep evolving. Update this file as things change, and treat it as a living project log, not a one-time brief.

---

## 1. The big picture — what this project actually is, and in what priority

Three layers, all connected — but **not equal priority**. This ordering is deliberate and matters for how Claude Code should allocate effort and, more importantly, how it should keep these layers separated in the actual written output:

1. **TOP PRIORITY — the dissertation itself, done rigorously, aimed at a publishable paper.** The primary goal of this entire project is to produce a strong Master's dissertation on dynamic, agent-based fairness enforcement in Federated Learning (FL), and from it, a paper submittable to a journal or a CORE-ranked conference/workshop. This is the actual work product. Everything else exists in service of this being excellent — real technical contribution, real experiments, real writing quality, judged on its own merits as research.
2. **SECOND PRIORITY — using the dissertation/paper as the substance behind a PhD application**, including outreach to Dr. Ashish Gupta at BITS Dubai (whose FairRFL, ECAI 2023, and ESORICS 2022 work sits close to this direction — see Section 6) and to supervisors at the four target Canadian universities (Section 4), plus potentially open-sourcing a repo of the implementation.
3. Both layers 2 and 3 above are downstream consequences of layer 1 being done well — not the other way around.

### ⚠️ Critical constraint — keep this in mind at all times

**The PhD-application motive is an ulterior/internal motive only. It must never leak into the dissertation itself, the paper, or the applicant's framing in front of Dr. Gupta or any other academic.** Concretely:

- The dissertation content, the paper draft, the literature review, the methodology, the writing — none of it should read as if it exists to impress a PhD admissions committee or a target supervisor. It should read as if it exists because it is good, honest research the applicant cares about.
- When Claude Code is asked to write or refine dissertation/paper content, it should evaluate and improve that content purely on research merit — clarity, rigor, novelty, correctness — with **zero reference to PhD strategy, target universities, or supervisor fit** inside that content.
- The PhD strategy (Section 4, email templates, professor papers) is a **separate, parallel workstream** that happens to be informed by the same research, but is written, discussed, and reasoned about independently. Don't cross-contaminate the two — e.g., never insert a line into the dissertation that was clearly reverse-engineered from "what would appeal to Professor X."
- If asked to help with both in the same session, treat them as two distinct hats: "dissertation/paper mode" vs. "PhD outreach mode," and don't blend the language or framing between them.
- The one legitimate place these layers meet is the Dr. Gupta conversation guide (Section 6) — because Dr. Gupta is the actual dissertation supervisor, discussing PhD plans with him at the appropriate moment is natural and expected. Even there, however, the technical substance of the dissertation pitch should stand on its own merit first; the PhD mention is a closing note, not the frame for the whole conversation.

The throughline for the _research itself_: read the target person's own work first (for the Dr. Gupta conversation and, separately, for professor outreach), then frame the research as a natural extension of theirs. But this rhetorical strategy applies to _conversations and outreach_, never to the dissertation/paper's actual content or internal logic.

---

## 2. Applicant profile (for quick reference in any generated content)

| Field                        | Value                                                                                                                                                                                                                      |
| ---------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Degree                       | MSc Software Systems, BITS Pilani Dubai                                                                                                                                                                                    |
| GPA                          | 9.34 / 10                                                                                                                                                                                                                  |
| IELTS                        | 8.0                                                                                                                                                                                                                        |
| Professional background      | Building agentic AI systems (production experience)                                                                                                                                                                        |
| Dissertation title (working) | Agentic Orchestration for Fair (and Robust) Federated Learning                                                                                                                                                             |
| Core research question       | Can an autonomous agent at the FL server dynamically enforce fairness constraints at runtime — adapting to distribution shift, client heterogeneity, and adversarial participation — where static fairness protocols fail? |
| PhD target entry             | Fall 2027                                                                                                                                                                                                                  |
| PhD target universities      | Waterloo, McGill/Mila, University of Alberta/Amii, UofT/Vector                                                                                                                                                             |

---

## 3. Dissertation research corpus — 20 papers total

Two sets. Both were vetted against the requirement: conferences must be **CORE A\*, A, or B**; journals must be **Q1 with impact factor > 3–4**.

### Set A — Landscape papers (10): fairness in FL + agentic/LLM in FL

Skewed 7:3 toward fairness (the underlying problem) vs. agentic/LLM (the proposed solution layer), per explicit instruction.

**Fairness in FL (7):**

1. Benarba & Bouchenak — _Bias in Federated Learning: A Comprehensive Survey_ — ACM Computing Surveys, 2025 (Q1, IF ~23.8)
2. Mukhtiar, Mahmood, Zhou, Yang, Teng, Sheng — _Federated Learning at the Forefront of Fairness: A Multifaceted Perspective_ — IJCAI 2025 (CORE A\*)
3. Ezzeldin, Yan, He, Ferrara, Avestimehr — _FairFed: Enabling Group Fairness in Federated Learning_ — AAAI 2023 (CORE A\*)
4. He, Chen, Zhang — _FedAA: A Reinforcement Learning Perspective on Adaptive Aggregation for Fair and Robust Federated Learning_ — AAAI 2025 (CORE A\*)
5. Lei et al. — _pFedFair: Towards Optimal Group Fairness-Accuracy Trade-off in Heterogeneous Federated Learning_ — NeurIPS 2025 (CORE A\*)
6. Huang et al. — _Federated Learning for Generalization, Robustness, Fairness: A Survey and Benchmark_ — IEEE TPAMI, 2024 (Q1, IF ~20.8)
7. Shi, Yu, Leung — _Towards Fairness-Aware Federated Learning_ — IEEE TNNLS, 2024 (Q1, IF ~10.4)

**Agentic / LLM in FL (3):** 8. Jarczewski, Talasso, Villas, de Souza — _Agentic Federated Learning: The Future of Distributed Training Orchestration_ — arXiv 2604.04895, 2026 (the explicit "gap paper" — proposes the paradigm, doesn't implement it) 9. Li, Wang, Saeed — _Helmsman: Autonomous Synthesis of Federated Learning Systems via Collaborative LLM Agents_ — arXiv 2510.14512, under review at ICLR 2026 10. Ye, Wang, Chai, Li et al. — _OpenFedLLM: Training Large Language Models on Decentralized Private Data via Federated Learning_ — KDD 2024 (CORE A\*)

**Suggested reading order:** 1 → 2 → 3 → 7 → 6 → 5 → 4 → 8 → 9 → 10 (surveys first, then methods, then the agentic frontier).

### Set B — Professor-specific papers (10): one per target supervisor, read before outreach

| #   | Paper                                                                                                                                   | Professor        | University   |
| --- | --------------------------------------------------------------------------------------------------------------------------------------- | ---------------- | ------------ |
| 1   | _Calibrated One Round Federated Learning with Bayesian Inference in the Predictive Space_ — AAAI 2024                                   | Pascal Poupart   | Waterloo     |
| 2   | _Confidence Aware Inverse Constrained Reinforcement Learning_ — ICML 2024                                                               | Pascal Poupart   | Waterloo     |
| 3   | _Procedural Fairness in Multi-Agent Bandits_ — arXiv 2601.10600, Jan 2026                                                               | Kate Larson      | Waterloo     |
| 4   | _Soft Condorcet Optimization for Ranking of General Agents_ — AAMAS 2025 (Best Paper)                                                   | Kate Larson      | Waterloo     |
| 5   | _RoFL: Robustness of Secure Federated Learning_ — IEEE S&P 2023                                                                         | Anwar Hithnawi   | UofT         |
| 6   | _Fairness in Federated Learning: Fairness for Whom?_ — AAAI/AIES 2025                                                                   | Golnoosh Farnadi | McGill/Mila  |
| 7   | _Differentially Private Clustered Federated Learning_ — TMLR 2025                                                                       | Golnoosh Farnadi | McGill/Mila  |
| 8   | _Near-Optimal Thompson Sampling-Based Algorithms for Differentially Private Stochastic Bandits_ — UAI 2022                              | Nidhi Hegde      | U of Alberta |
| 9   | _Remembering to Be Fair: Non-Markovian Fairness in Sequential Decision Making_ — ICML 2024                                              | Sheila McIlraith | UofT         |
| 10  | _On the Necessity of Auditable Algorithmic Definitions for Machine Unlearning_ — USENIX Security 2022 (Thudi, Jia, Shumailov, Papernot) | Nicolas Papernot | UofT         |

**Note on Papernot:** no single clean, public, FL-specific paper of his was found — his most relevant FL-adjacent thread is machine unlearning/auditing. Flagged as needing more digging if the Papernot email becomes a priority. The RoFL paper (#5) is shared context between Hithnawi and Papernot conversations since both work in ML security/privacy at UofT/Vector.

---

## 4. University targets — condensed supervisor map

Full detail lives in the four original research reports (Waterloo, McGill/Mila, U of Alberta, UofT/Vector) — summarizing here for quick reference. Priority order within each school:

### Waterloo (deadline: Dec 1, 2026 for Fall 2027)

1. **Pascal Poupart** — TOP PRIORITY. Bayesian FL + fresh (Aug 2026) $255K NSERC grant on "Agentic Workflow Adaptation and Validation." Exact topic match.
2. **N. Asokan** — security/robustness of FL, University Professor (2026), h-index 78.
3. **Kate Larson** — mechanism design, procedural fairness in multi-agent systems (2026 paper).
4. **Xi He** — differential privacy, actively hiring.
5. **Robin Cohen** — trust-augmented RL for FL client selection (2024 paper), moderate fit.
6. ⚠️ **Gautam Kamath** — do NOT target for Waterloo; relocating to NYU Courant Fall 2026.

### McGill / Mila (McGill deadline: Dec 15, 2026; Mila supervision request window: Oct 15–Dec 1, 2026 — SEPARATE, non-negotiable)

1. **Golnoosh Farnadi** — TOP PRIORITY. EQUAL Lab. Direct FL + fairness + privacy match. Actively recruiting.
2. **Joelle Pineau** — fairness in RL (zero-violation online fairness), just returned to McGill from Meta FAIR (2025), also CAO at Cohere — high demand, confirm capacity.
3. **Reihaneh Rabbany** — graph ML, social good, backup/co-supervisor.
4. Xue (Steve) Liu, Doina Precup, William Hamilton — secondary/co-supervision candidates.
5. Note: Farnadi, Gidel, Mitliagkas, Lacoste-Julien at UdeM/Mila are technically UdeM DIRO applications, not McGill CS — verify which applies before Farnadi outreach (she has McGill + UdeM appointments; confirm current primary).

### University of Alberta / Amii (deadline: Dec 15, 2026 – Jan 15, 2027 for Fall 2027)

1. **Nidhi Hegde** — TOP PRIORITY. Privacy + fairness by design, actively hiring, industry background (Borealis AI).
2. **Matthew Taylor** — MARL, multi-agent coordination, large well-funded lab (IRL/RLAI).
3. **A. Rupam Mahmood** — continual/streaming RL, explicit relevance to privacy-sensitive distributed applications (2024 Nature paper on plasticity loss).
4. **Osmar Zaïane** — data privacy, senior network, secondary supervisor.
5. **Lili Mou** — only relevant if LLM-based orchestration becomes central to the thesis.
6. Note: U of A / Amii just completed a $30M, 25-faculty recruitment campaign (May 2026) — new AI ethics/privacy hires may not be publicly listed yet; check amii.ca/about/our-people again closer to outreach time.

### University of Toronto / Vector Institute (deadline: expect ~late Nov/early Dec 2026 — confirm from Oct 2026)

1. **Nicolas Papernot** — TOP PRIORITY by eminence. Privacy/security/ML auditing, 2025 Steacie Prize, CAISI Co-Director.
2. **Sheila McIlraith** — TOP PRIORITY by conceptual fit. Sequential decision-making, non-Markovian fairness, agentic multi-agent verification.
3. **Anwar Hithnawi** — new to UofT (Jan 2025), secure/robust FL, high probability of taking students.
4. **Toniann Pitassi** — theoretical fairness foundations, verify Toronto (vs. Columbia/IAS) availability.
5. **Zhijing Jin** — new (July 2025), multi-agent LLM ethics/coordination failures — relevant if LLM-orchestration angle grows.
6. Xiaoxiao Li (UBC) and Gautam Kamath (Waterloo) are Vector-affiliated but NOT UofT primary supervisors — only relevant as potential co-supervisors.

---

## 5. Email templates — status

Full draft email templates for the top 3 supervisors at each of the 4 universities (12 emails total) were already written in the original research reports. They are NOT reproduced in full here to keep this handoff lean — pull them from the original report documents in the project directory when ready to send. Each follows the same structure:

1. Open with a specific, named paper of theirs
2. State the dissertation's core contribution in one or two sentences
3. Identify the precise gap/connection between their work and the applicant's direction
4. Close with a soft ask (chat, CV review, fit assessment) — never a hard ask for a position

**Before sending any of these emails**, re-verify (a) the professor is still at the stated institution, (b) they are still taking students for Fall 2027, and (c) application deadlines haven't shifted. All research reports were compiled August 2026 and explicitly flag this as needing reconfirmation closer to the September–October 2026 outreach window.

---

## 6. Existing artifacts already built (in claude.ai, need porting/reference)

Two interactive HTML documents already exist from earlier work in this project:

1. **`dissertation_proposal.html`** — A tabbed, interactive guide structured as:
   - Tab 0: Profile of Dr. Ashish Gupta (BITS Dubai supervisor) — his papers, why he's the right fit, specific quotes from his site
   - Tab 1: The FL fairness landscape (2026 state of the field)
   - Tab 2: The core problem statement (one-sentence version + why static fairness fails)
   - Tab 3: Three concrete research directions to propose, ranked by preference/feasibility
   - Tab 4: Comparison table of the three directions
   - Tab 5: One-semester execution plan with monthly milestones
   - Tab 6: Full conversation script for the Dr. Gupta meeting — opening lines, anticipated pushback (on scope, on robustness vs. fairness framing), and closing the PhD/co-authorship ask

2. **`researcher_profile.html`** — A structured one-page profile document meant to be pasted at the start of any new professor-outreach conversation. Contains: background, skill stack (strong/solid/growing), research spine, six detailed research areas with "why this matters" notes, specific research interests, what the applicant wants in a PhD, and six adjacent fallback research areas if fair FL doesn't work out with a given supervisor.

**Action for Claude Code:** these two files should be copied into the project directory if not already present. They are reusable — the `researcher_profile.html` in particular should be pasted verbatim at the start of any new professor-specific research task.

---

## 7. Key intellectual position established so far (don't relitigate unless asked)

A few substantive things were worked through in earlier conversation that shape how this dissertation should be described going forward — worth preserving so they aren't re-derived from scratch:

- **The "does this need an agent?" question was directly confronted.** The applicant correctly intuited that a simple round-by-round adaptive feedback loop (if fairness gap > threshold, adjust weights) is _not_ inherently agentic — it's just a control loop, and several existing papers (e.g. FairFedCS with Lyapunov optimization) already do this without calling it an agent. The distinction that justifies calling it an "agent" needs to rest on something a plain formula cannot do — e.g., reasoning about _why_ fairness is degrading (distribution shift vs. new client vs. adversarial behavior) and responding differently to each cause, not just reacting to the same metric the same way every time. **Any future framing of the dissertation should make this distinction explicit and not hand-wave "agentic" as a buzzword.**
- **Scoping strategy for the Masters vs. PhD:** rule-based agent for the Masters dissertation (feasible in one semester, still novel since no empirical implementation exists yet per arXiv:2604.04895), explicitly deferring a learned/RL policy to the PhD as future work. This is the load-bearing argument in the Dr. Gupta conversation script and should carry through into PhD supervisor conversations too — it shows a coherent 2-stage research programme (MSc → PhD), which supervisors respond well to.
- **Fairness type in play:** primarily group/demographic fairness (demographic parity gap as primary metric), with client-level fairness as a secondary lens. Individual fairness was explicitly deprioritized as harder to enforce in FL without visibility into individual client data.

---

## 8. NotebookLM setup (for reference / re-creation if needed)

Notebook name: **"Multifaceted Perspectives on Fairness in Federated Learning"** (kept as-is; mirrors the IJCAI 2025 survey title).

Custom chat configuration used:

> You are a research assistant helping a Masters student in Software Systems (GPA 9.34) who is writing a dissertation on fair federated learning with agent-based dynamic enforcement. The student has strong hands-on experience in agentic AI systems and federated learning, and is preparing for a PhD application. Your role is to help them deeply understand the loaded papers as a coherent research landscape — not as isolated papers. Always connect concepts across papers. When explaining, pitch at a confident graduate researcher level... [full text in earlier conversation; reconstruct if needed]

Response length: Longer.

Three structured prompts were used to generate a mind map, a data comparison table, and a full research briefing report from the 10 landscape papers — see Section 9 below for the prompts, reusable for the 10 professor papers too.

---

## 9. Reusable prompt templates (for NotebookLM, or for Claude Code to replicate the same analysis directly)

**Mind map prompt:**

> Generate a mind map of the research landscape across all papers. The central node should be "Fairness in Federated Learning." Branch by: (1) fairness definitions used, (2) approaches to fairness, (3) static vs. dynamic enforcement, (4) where LLMs/agents enter FL, (5) open gaps. Name specific papers at each node.

**Data table prompt:**

> Create a comparison table with columns: Paper shortname, Year, Venue, Fairness type addressed, Enforcement mechanism, Static or Dynamic, Handles adversarial clients, Uses agents/LLMs, Key dataset/benchmark, Primary gap left open. Sort from most foundational to most novel.

**Report prompt:**

> Write a structured research briefing titled "The FL Fairness Landscape: What Exists, What's Missing, and Where My Dissertation Fits." Cover: the core problem, static approaches and limitations, the adaptive/RL turn, the fairness-robustness-generalization intersection, where agents/LLMs enter FL, and the specific gap the dissertation fills.

**For the professor-specific papers**, an additional prompt is worth running once all 10 are loaded:

> For each professor's paper(s), write 3-4 sentences: (1) what problem it solves, (2) the specific technical/conceptual bridge to my dissertation's agentic fairness-enforcement approach, (3) one precise, technical question I could ask this professor that shows I've engaged deeply with their specific contribution (not a generic question).

---

## 10. Open items / what's not yet done

**Dissertation & paper (top priority — do these first):**

- [ ] Confirm Dr. Gupta conversation has actually happened and note the outcome — locks in the actual dissertation scope (Option 1/2/3 from the proposal doc) before anything else downstream can proceed.
- [ ] Build out the actual technical work — literature review write-up, system design, implementation (rule-based agent per the agreed one-semester scope), experiments against baselines (FedAvg, q-FedAvg, Ditto, FairFed).
- [ ] Identify and track the target venue for the resulting paper — likely a workshop (e.g. an ICLR 2027 workshop, per the original plan) or a suitable journal; confirm submission deadlines once experiments are further along.
- [ ] Re-verify all CORE rankings and journal impact factors closer to any formal submission of the literature review, since these can shift year to year (checked against CORE2023 and current SJR/IF as of August 2026).
- [ ] Keep dissertation/paper writing free of any PhD-strategy framing per the constraint in Section 1 — spot-check drafts for this before they're considered final.

**PhD application campaign (second priority — downstream of the above):**

- [ ] Confirm current faculty status (still at institution, still taking students) for all 20 named professors before sending any emails — reports are dated August 2026 and explicitly flag this as needing reconfirmation.
- [ ] Decide whether to apply to all 4 universities or narrow the list — no explicit decision has been made yet on this.
- [ ] Draft the actual Statement of Purpose / Research Statement for each university (not yet started — email templates exist, SoP does not).
- [ ] Secure 3 reference letters — process not yet started; Waterloo requires letters within 14 days of the December 1 deadline.
- [ ] The McGill Mila Supervision Request (Oct 15–Dec 1 window) is a hard, separate deadline from the McGill CS application — track this independently; it's easy to miss.
- [ ] Consider whether/when to open-source a repo of the dissertation implementation — a secondary, PhD-supporting artifact, not a substitute for the paper itself.

---

## 11. How to pick this up in Claude Code

Suggested first steps for Claude Code when this project is opened:

1. `ls` the project directory to confirm which of the 20 papers (Section 3) and which HTML artifacts (Section 6) are actually present as files.
2. If NotebookLM exports (mind map, table, briefing) were saved before switching tools, ingest those as additional context — they contain synthesis already done and shouldn't be redone from scratch.
3. Cross-reference Section 4 (university targets) against the four original full-length research reports if they are in the directory — this handoff is a condensed summary, not a replacement for the full reports' detail (funding context, recent news, full email drafts).
4. When generating new content (emails, SoPs, further paper analysis), maintain the voice and strategy already established: specific, evidence-based, framed as extending the target's own published work — not generic enthusiasm.
5. Keep this file updated as a running log — append a dated changelog entry at the bottom each time significant new decisions are made (e.g., "2026-09-25: decided to drop U of A from target list" or "2026-10-02: Dr. Gupta meeting happened, agreed on Option 1 scope").
