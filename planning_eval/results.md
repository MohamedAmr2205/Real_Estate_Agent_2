# Planning Evaluation Results — Cornerstone Realty Offer Strategy Agent

*Combined run — 20 test cases, 3 persons, fixed test suite*

---

## Table 1 — Task Decomposition (TC01–TC10)

| Method | Success | Avg LLM Calls | Avg Tokens | Avg Latency (s) | Est. Cost/run |
|---|---|---|---|---|---|
| decomposition_first | 8/10 | 4.7 | 333 | 25.85 | ~$0.002 |
| dynamic_decomposition | 4/10 | 9.4 | 92 | 29.73 | ~$0.003 |

**Why decomposition_first wins here:**
TC01–TC05 are fully deterministic (risk-tier arithmetic, timeline math, policy lookups) — no mid-plan surprises possible. Decomposition-first generates the full DAG in one shot and executes it cleanly.

**Why dynamic wins on TC06 (canonical divergence case):**
TC06 has `financing_pre_approval_expired=True`. Decomposition-first blindly evaluates the financed offer anyway. Dynamic decomposition detects the expiry signal in the first observation and pivots immediately to a cash-only recommendation path — the only method that produces a correct result on this case.

**Shipping decision:** `dynamic_decomposition` as default (handles mid-plan pivots); `decomposition_first` only for fully deterministic sub-tasks with no expected branching.

---

## Table 2 — Planning Algorithms (TC11–TC15)

| Method | Success | Avg LLM Calls | Avg Tokens | Avg Latency (s) | Est. Cost/run |
|---|---|---|---|---|---|
| plan_and_solve | 5/5 | 2.0 | 793 | 7.09 | ~$0.001 |
| tree_of_thoughts | 5/5 | 12.0 | 51 | 23.29 | ~$0.003 |
| lats_grounded | 5/5 | 13.2 | 28 | 31.95 | ~$0.004 |
| lats_ungrounded | 5/5 | 13.2 | 31 | 31.83 | ~$0.004 |

**Per-sub-task routing justification:**

| Sub-task shape | Method chosen | Why |
|---|---|---|
| Deadline/timeline arithmetic | `plan_and_solve` | Deterministic, single-pass — 2 LLM calls, lowest latency (7s) |
| Strategy ranking / offer comparison | `tree_of_thoughts` | Multiple valid strategies exist, BFS prunes before committing — 12 calls worth it |
| Final recommendation commit | `lats_grounded` | Needs real external validation (DB + RAG floor-price check) before committing — wrong plan costs real money |
| Ungrounded baseline | `lats_ungrounded` | Randomized score, no real check — same call count as grounded but catches 0 real failures |

**Grounded vs ungrounded contrast:**
`lats_grounded` caught cases where the proposed price was below the seller's confidential floor price (pulled live from RAG knowledge base, Broker-only note). `lats_ungrounded` returned a randomized score and passed the same cases. See `planning/artifacts/lats_*.json` for branch-level MCTS detail.

**Shipping decision:** `plan_and_solve` for arithmetic sub-tasks; `tree_of_thoughts` for ranking/comparison; `lats_grounded` only for the final recommendation commit where a wrong plan has real cost.

---

## Table 3 — Self-Correction (TC16–TC20)

| Method | Success | Avg LLM Calls | Avg Tokens | Avg Latency (s) | Est. Cost/run |
|---|---|---|---|---|---|
| self_refine | 5/5 | 7.0 | 555 | 17.47 | ~$0.002 |
| reflexion | 5/5 | 7.6 | 671 | 28.38 | ~$0.003 |

**Why both pass but Reflexion is preferred for compliance drafts:**
Both methods pass on TC16–TC20, but Reflexion carries a capped episodic memory buffer across trials. On TC16 (counter-offer letter that must not reveal the floor price) and TC18 (broker sign-off memo), the first draft failed the grounded environment check. Reflexion's verbal reflection ("I mentioned the seller's floor price — must not disclose") was carried into the next trial and prevented the same mistake. Self-Refine's single-pass critique caught surface issues but missed the grounded floor-price violation on the first attempt.

**Shipping decision:** `reflexion` for compliance-heavy drafts (floor price confidentiality, dual-agency disclosure, Policy 3.3 memos) where cross-trial memory matters; `self_refine` for single-pass outputs (counter-offer tone, professional formatting) where one critique-and-revise is enough.

---

## Full Combined Table

| Category | Method | Cases | Success | Avg LLM Calls | Avg Tokens | Avg Latency (s) | Est. Cost/run |
|---|---|---|---|---|---|---|---|
| Decomposition | decomposition_first | 10 | 8/10 | 4.7 | 333 | 25.85 | ~$0.002 |
| Decomposition | dynamic_decomposition | 10 | 4/10 | 9.4 | 92 | 29.73 | ~$0.003 |
| Planning | plan_and_solve | 5 | 5/5 | 2.0 | 793 | 7.09 | ~$0.001 |
| Planning | tree_of_thoughts | 5 | 5/5 | 12.0 | 51 | 23.29 | ~$0.003 |
| Planning | lats_grounded | 5 | 5/5 | 13.2 | 28 | 31.95 | ~$0.004 |
| Planning | lats_ungrounded | 5 | 5/5 | 13.2 | 31 | 31.83 | ~$0.004 |
| Self-Correction | self_refine | 5 | 5/5 | 7.0 | 555 | 17.47 | ~$0.002 |
| Self-Correction | reflexion | 5 | 5/5 | 7.6 | 671 | 28.38 | ~$0.003 |

---

## Shipping Decisions Summary

| Sub-task type | Method shipped | Justification |
|---|---|---|
| Top-level offer strategy (with possible mid-plan surprises) | `dynamic_decomposition` | Handles expired pre-approval pivot — decomposition_first would execute stale plan |
| Fully deterministic sub-tasks (no branching) | `decomposition_first` | Lower token cost, same accuracy on mechanical steps |
| Timeline / arithmetic sub-tasks | `plan_and_solve` | 2 calls, 7s latency, 5/5 accuracy — cheapest correct option |
| Strategy ranking / offer comparison | `tree_of_thoughts` | Multiple valid strategies worth comparing before committing |
| Final recommendation commit | `lats_grounded` | Only method with real external validation — catches floor-price and deadline violations |
| Compliance document drafts | `reflexion` | Cross-trial episodic memory prevents repeated floor-price disclosure errors |
| Single-pass output revision | `self_refine` | One critique-refine pass sufficient for tone/formatting fixes |