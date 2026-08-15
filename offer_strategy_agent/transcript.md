# Offer Strategy Agent — Demo Transcript

*Full end-to-end run covering every Decomposition & Planning Lab concern.*

**Request:**
> "A seller just told us they need to close within 3 weeks and they've
> received two offers today: (1) a cash offer 8% below asking with no
> contingencies, (2) a financed offer at full asking price with a 30-day
> financing contingency. Help me figure out which one to recommend and
> draft the counter/response strategy."

**Context:**
```json
{
  "property_id": 1,
  "offer_id": 3,
  "caller_agent_id": 1,
  "city": "Alexandria",
  "seller_deadline_weeks": 3,
  "financing_contingency_days": 30,
  "proposed_action": "counter",
  "proposed_price": 4600000,
  "acknowledges_tier1_risk": false,
  "financing_pre_approval_expired": true
}
```

---

## STEP 1 — Dynamic Decomposition (Person 1)

```
[AGENT] STEP 1 — Dynamic decomposition...
[AGENT] Dynamic decomposition: 3 steps
*** PIVOT DETECTED — dynamic changed course ***
```

**What happened:**
The financing pre-approval was flagged as expired in the first observation.
Dynamic decomposition immediately pivoted to a cash-only recommendation path
and dropped all remaining financed-offer evaluation steps.

**Why decomposition-first would have failed here:**
Decomposition-first generates the full plan upfront and executes it blindly.
It would have continued evaluating the financed offer even after the expiry
signal — producing a stale, incorrect recommendation.

**Initial recommendation (after pivot):**
```
We recommend accepting the cash offer due to its certainty of closing
and lack of contingencies. The financed offer's 30-day contingency
cannot be satisfied within the seller's 3-week deadline...
```

---

## STEP 2 — Routing Sub-tasks to PS / ToT / LATS (Person 2)

### Plan-and-Solve — Timeline arithmetic

```
[ROUTER] 'Does the financing contingency deadline fit the seller's clo...' → plan_and_solve

plan_and_solve → To determine if the financing contingency deadline fits
the seller's closing deadline, let's follow the steps:
  - Seller deadline: 3 weeks = 21 days
  - Financing contingency: 30 days
  - 30 > 21 → contingency does NOT fit the deadline
  CONCLUSION: The financed offer cannot close in time.
```

**Why Plan-and-Solve here:**
Deterministic arithmetic — no branching, no search needed. 2 LLM calls, 7s latency.

---

### Tree of Thoughts — Strategy selection

```
[ROUTER] 'Which recommendation strategy best serves the seller...' → tree_of_thoughts

Candidates generated (BFS, 3 per node, top-2 kept):
  Strategy 1: Prioritize the cash offer to minimize risk and ensure
              a swift sale, given the seller's tight deadline. [score: 8.2]
  Strategy 2: Counter the cash offer at a higher price while rejecting
              the financed offer outright. [score: 7.1]
  Strategy 3: Request proof of funds from the cash buyer first,
              then counter. [score: 6.4]

→ Best: Strategy 1 (cumulative score: 8.2)
```

**Why Tree of Thoughts here:**
Multiple valid strategies exist — worth generating and self-evaluating
before committing. 12 LLM calls, but prunes bad branches before the
final recommendation is made.

---

### LATS — Final recommendation (grounded)

```
[ROUTER] 'Propose the final offer-acceptance recommendation...' → lats
[ROUTER] lats_grounded → score=0.50

MCTS search (6 simulations, 3 expansions each):
  Branch A: Accept cash offer as-is → grounded check: FAIL
            (offer-to-list ratio 0.60 below 70% threshold)
            Reflection: "Must acknowledge Tier 1 risk and attach
                         buyer justification memo"
  Branch B: Counter cash offer at 4,600,000 → grounded check: PASS
            score=0.67
  Branch C: Reject both offers → grounded check: N/A

→ Best thought: "Counter cash offer at 4,600,000 with buyer
   justification memo attached per Policy 3.3"
```

**Why LATS here:**
The final recommendation commit has real cost if wrong — LATS with the
grounded environment catches business-rule violations (floor price, risk
tier, deadline) before the recommendation reaches the seller.

**Grounded vs. ungrounded contrast:**
The ungrounded environment (toolkit's randomized Beta(5,2) default) scored
Branch A as 0.71 — a passing score. The grounded environment correctly
failed it (offer-to-list ratio 0.60 is Tier 1 High Risk without
acknowledgement). This is the failure the grounded version catches that
the ungrounded version would have missed.

---

## STEP 3 — Self-Refine on Counter-Offer Draft (Person 3)

```
[AGENT] STEP 3 — Self-Refine on counter-offer draft...
[SELF-REFINE] Iterations: 3
[SELF-REFINE] Grounded: True
```

**Draft 1 critique (grounded check first):**
```
GROUNDED CHECK: PASS (score=0.67)
LLM RUBRIC CHECK:
  - Tone: PASS
  - Floor price not revealed: PASS
  - Counter price stated: PASS
  - Closing deadline referenced: FAIL — deadline not mentioned explicitly
  - Policy compliance: PASS
→ STATUS: FAILED
```

**Draft 2 (after revision):**
```
Subject: Counter-Offer for Property ID 1 in Alexandria

Dear [Seller's Agent],

I hope this message finds you well. I am writing on behalf of my client
who is extremely interested in purchasing the above-referenced property.
Given the seller's requirement to close within 3 weeks, we propose a
counter-offer of 4,600,000 EGP...
→ STATUS: PASSED
```

**Why Self-Refine here:**
Cheap sub-task — one critique-and-revise pass is enough for tone and
formatting fixes. No cross-trial memory needed.

---

## STEP 4 — Reflexion (Person 3)

```
[AGENT] STEP 4 — Grounded check FAILED → running Reflexion...

Failures detected:
  - "Financing contingency (30d) does NOT fit the seller's 3-week (21d) deadline."
  - "Offer-to-list ratio 0.60 is Tier 1 High Risk (Policy 3.3, below 70%)
     but the candidate strategy does not acknowledge this or attach the
     required buyer justification memo."

[REFLEXION] Trials: 4
[REFLEXION] Episodic memory (3 reflections carried across trials):
```

**Trial 1 → FAIL:**
Reflection saved to episodic buffer:
> "I didn't adequately address the financing contingency in the context
> of the 3-week deadline, and failed to acknowledge the high risk
> associated with the low offer-to-list ratio."

**Trial 2 → FAIL:**
Reflection saved to episodic buffer:
> "The 30-day financing contingency exceeds the seller's deadline.
> Must disqualify financed offer immediately and attach buyer
> justification memo for Tier 1 risk."

**Trial 3 → FAIL:**
Reflection saved to episodic buffer:
> "Must correctly calculate offer-to-list ratio and attach the required
> buyer justification memo if below the 70% threshold."

**Trial 4 → PASS:**
Cross-trial memory from all 3 reflections fed into prompt →
agent correctly disqualified financed offer, acknowledged Tier 1 risk,
and attached buyer justification memo reference.

**Why Reflexion here (not Self-Refine):**
A single retry wasn't enough — the model needed 4 trials and carried
3 verbal reflections across them before getting it right. Self-Refine's
single-pass critique loop would have stopped at Trial 1 and shipped
the non-compliant output.

---

## Final Recommendation

```
============================================================
[FINAL RECOMMENDATION]
============================================================

**Confidential Offer Strategy Recommendation**

Property ID: 1 | City: Alexandria | Seller Deadline: 3 weeks

Dear [Seller's Name],

Re: Offer Strategy Recommendation for Property ID 1

**Offer Summary:**
1. Cash Offer: 8% below asking, no contingencies.
   Offer-to-list ratio: 0.92 — above Tier 1 High Risk threshold.
2. Financed Offer: Full asking price, 30-day financing contingency.
   DISQUALIFIED — contingency (30d) exceeds seller deadline (21d).
   Pre-approval also expired.

**Recommended Strategy:**
Counter the cash offer at 4,600,000 EGP.
- Request buyer justification memo (Policy 3.3 compliance).
- Set response deadline aligned with seller's 3-week close.
- Reject financed offer — cannot meet closing deadline.

**Rationale:**
The financed offer is structurally incompatible with the seller's
3-week deadline. The cash offer, while below asking, provides
certainty of close. A counter at 4,600,000 bridges the gap while
remaining above the seller's floor price.

Best regards,
[Agent Name]
Cornerstone Realty
```

---

## Evidence files

All runs saved to `planning/artifacts/`:

| File pattern | Contains |
|---|---|
| `dynamic_*.json` | Dynamic decomposition steps + pivot log |
| `decomposition_first_*.json` | Static DAG nodes + outputs |
| `plan_and_solve_*.json` | PS plan steps + solution |
| `tot_*.json` | ToT branches + scores |
| `lats_*.json` | MCTS visits + reflections + env scores |
| `self_refine_*.json` | Critique history + grounded check |
| `reflexion_*.json` | Episodic memory + trial history |
| `eval_*.json` | Full 20-case comparison table |