# ICS — Decisions & Assumptions

Every non-obvious engineering decision, and every assumption made where the
governing document was ambiguous or unavailable. Newest section last.

> **Standing constraint:** `mode: paper_only`. Nothing recorded here relaxes it.
> ICS-DOC-003 (real-trading governance) is out of scope and untouched.

---

## ⚠️ OPEN GAP — ICS-DOC-004 is not in the repository

**Date:** 2026-08-22 · **Status:** open · **Blocks:** formal phase sign-off

The execution order references `ICS-DOC-004` as the governing roadmap, but the
document is **not present in the repository and was never supplied in full**.
`grep -r "ICS-DOC-004"` returns nothing, and there is no
`ICS-DOC-004-learning-intelligence-roadmap.md`.

Phase 0 was still implementable because the execution order itself specified the
work items exactly (DecisionOutcome fields, 30-trade gate, ±5% cap, sum = 100,
LearningEvent, `/learning`, scipy/Spearman). Two things could **not** be taken
from the document:

| Needed from ICS-DOC-004 | What was done instead |
|---|---|
| The exact Phase 0 success gate | Applied the generic acceptance gate from the execution order (all tests pass, `paper_only` unchanged, numeric comparison vs baseline, docs updated). |
| The documented baseline | Supplied later by the Phase 1 amendment: **Sharpe 1.01, return +21.88%, drawdown −5.42%**. Used as the reference. |

The roadmap file itself was **not authored**: writing it would mean inventing
requirements and presenting them as the governing spec. Supply the document and
it will be added, and the Phase 1 amendment section appended to it.

---

## Phase 0 — learning feedback loop

**Date:** 2026-08-22

### D-01 · New tables instead of altering `decisions` / `positions`
The order said "extend Decision **/** add DecisionOutcome". Chose to add
`decision_outcomes` and `learning_events` as **new tables** and to leave every
existing table untouched.
**Why:** `create_all()` creates missing *tables* but never missing *columns*.
Adding a column to `positions` would apply locally and silently not apply to the
live Render database, producing `no such column` at runtime. New tables upgrade a
live database with no migration step.

### D-02 · Linking a closed position to its entry decision
`Position` has no `decision_id` (see D-01). The entry decision is matched on
`ticker` + the closest BUY decision at or before `entry_at`.
**Why:** the decision and the fill are written with the same `ts` by
`DecisionEngine`, so this is exact in practice, and it avoids a schema change.
**Risk accepted:** ambiguous if the same ticker were opened twice within one
timestamp — impossible today because the risk manager blocks pyramiding.

### D-03 · DQS weights made injectable, scoring rules untouched
`calculate_dqs()` gained an optional `weights` argument. Each component is now
computed as *fraction of its own maximum × weight* rather than raw-value-capped.
**Why:** the loop must be able to re-balance weights; without this, changing a
weight would only change a cap, not the component's influence.
**Verified:** with `DEFAULT_WEIGHTS` the output is **bit-identical** to the
pre-Phase-0 implementation (same score, same component breakdown) — the natural
maximum of each raw sub-score equals its default weight.

### D-04 · ASSUMPTION — "±5% cap" read as *relative*, not percentage points
The order says `سقف تعديل ±5% لكل مكوّن DQS لكل دورة` without stating the base.
Interpreted as **±5% of that component's current weight** (e.g. 20.0 → 19.0‥21.0).
**Why:** percentage *points* would allow 25 → 30 in a single week — a 20% swing —
which contradicts the gradual, bounded intent of a weekly learning loop.
**Confirm against ICS-DOC-004 when it is supplied.** Configurable via
`MAX_SHIFT_PCT` if the other reading was intended.

### D-05 · Zero-sum re-balance instead of cap-then-normalise
Signals are centred on the **weight-weighted** mean correlation, so the deltas
cancel and the total stays at exactly 100 without re-normalising.
**Why:** the first implementation capped each move at ±5% and *then*
re-normalised to 100 — which pushed a component to 5.38%, past its own cap. Caught
by `test_shift_never_exceeds_the_five_percent_cap`. The floor/ceiling
(5.0 / 40.0) still takes precedence over gradualism in extremes, and only then is
re-normalisation applied.

### D-06 · Undefined correlation treated as neutral (0.0), not frozen
A component whose values are constant has no defined Spearman correlation.
Treated as neutral evidence rather than excluded.
**Why:** with exclusion, a single informative component produced a zero signal
(mean of one value equals itself) and nothing moved — caught by
`test_cycle_applies_and_moves_driver_up`.

### D-07 · MFE/MAE degrade to `None` rather than blocking
Excursions need the price path between entry and exit. A `price_provider` is
injectable; the default fetches daily bars and any failure leaves `mfe`/`mae`
`None`.
**Why:** the loop must never be blocked by a flaky free data source. Related:
yfinance was observed silently dropping a random symbol per bulk fetch (JPM, then
NVDA), so comparisons must reuse one fixed dataset.

### D-08 · `scipy` pinned exactly
`scipy==1.17.1` (latest stable on Python 3.11 at implementation time), per the
explicit-pin rule. No other new dependency in Phase 0 — the loop reuses the
existing SQLAlchemy models and the APScheduler instance already running inside
the bot process.

---

## Amendment — Phase 1 statistical tightening

**Date:** 2026-08-22 · **Reference:** supplementary execution order
"أمر تنفيذ مكمّل — تشديد إحصائي المرحلة 1" · **Applies to:** Phase 1 only

**Reason:** to prevent a statistical fluke produced by multiplicity — running many
hyper-parameter trials until one looks good, then reporting it as a real edge.

Recorded now, to be implemented **only** when Phase 1 begins (Phase 0 must pass
its gate first; `optuna`/DSR are deliberately absent from Phase 0):

1. **`optuna`** for bounded Bayesian tuning of `lightgbm` — a fixed, predeclared
   search space (no open search) and a hard ceiling of **100 trials** per monthly
   training cycle. Every trial must be logged: trial number, parameters,
   validation score. That log feeds item 2.
2. **Quantitative gate replacing "clear statistical superiority":**
   - **Deflated Sharpe Ratio (DSR)**, accounting for the total number of trials
     actually run (model training + optuna trials), not raw Sharpe.
   - Validation over **≥ 3 non-overlapping time windows**, not one test window.
   - The model passes only if it beats the documented baseline —
     **Sharpe 1.01, return +21.88%, drawdown −5.42%** — in the **majority** of
     windows.
   - `ml_confidence` stays in shadow mode, with **no** effect on live DQS, until
     all three numbers above are met. No visual/qualitative exception.
3. `optuna` to be pinned exactly in `requirements.txt` at implementation time; a
   "تعديل — تشديد إحصائي المرحلة 1" section to be added to
   `ICS-DOC-004-learning-intelligence-roadmap.md` once that document exists.
