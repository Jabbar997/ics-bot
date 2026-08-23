# ICS — Decisions & Assumptions

Every non-obvious engineering decision, and every assumption made where the
governing document was ambiguous or unavailable. Newest section last.

> **Standing constraint:** `mode: paper_only`. Nothing recorded here relaxes it.
> ICS-DOC-003 (real-trading governance) is out of scope and untouched.

---

## ✅ RESOLVED — ICS-DOC-004 supplied

**Raised:** 2026-08-22 · **Resolved:** 2026-08-22

Phase 0 was implemented before the governing document was available (it was not
in the repo and `grep -r "ICS-DOC-004"` returned nothing). The work items were
taken from the execution order itself; two things could not be: the exact Phase 0
success gate, and the documented baseline.

The document has since been supplied and is now committed as
`ICS-DOC-004-learning-intelligence-roadmap.md`, with the Phase 1 statistical
amendment folded into it. Reviewing the implementation against it surfaced one
real conflict — see **D-04 (CORRECTED)** below.

Baseline now confirmed: **+21.88% return, −5.42% max drawdown, Sharpe 1.01,
38.7% win rate, average DQS 85.15** (5 years, walk-forward).

Phase 0 gate now known: **at least two weekly cycles run, no test broken, and
average DQS must not fall below 80.**

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

### D-04 (CORRECTED) · The ±5 cap is ABSOLUTE POINTS, not relative percent
**Implemented first as:** ±5% of the component's own weight (25 → 23.75‥26.25).
**ICS-DOC-004 actually says:** ±5 **absolute percentage points**, and explicitly
rejects the relative reading using this very example — "`strategy_alignment` عند
25 يتحرك بحد أقصى بين 20 و30 في دورة واحدة، **لا 25±1.25**".

**Corrected.** `MAX_SHIFT_POINTS = 5.0`; the re-balance now adds absolute points
and is zero-sum by centring signals on the plain (unweighted) mean correlation.
Guarded by `test_cap_is_absolute_points_not_relative_percent`, which asserts
25 → 30.0 and explicitly fails the 26.25 reading. `MAX_SHIFT_PCT` remains as a
backwards-compatible alias only.

**Lesson:** this was an assumption made in the absence of the governing document
and it was wrong. Phase 1 will not start until ICS-DOC-004 has been re-read
against the implementation.

### D-05 · Zero-sum re-balance instead of cap-then-normalise
Signals are centred on the **plain** mean correlation, so the absolute-point
deltas cancel and the total stays at exactly 100 without re-normalising.
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

### D-09 · Bound projection instead of re-normalisation after clamping
Once the cap became absolute points, ten consecutive cycles drove one component
to the ceiling and the rest to the floor; re-normalising that set
(40 + 5·4 = 60, scaled by 100/60) threw the first component to 66.7 — straight
back through the ceiling. Caught by
`test_repeated_cycles_stay_bounded_and_normalised`.
**Fixed** with `_project_to_bounds()`: clamp, then hand the residual only to
components that still have headroom in the required direction, repeating until
absorbed. Keeps the sum at exactly 100 *and* every weight inside [5, 40].

### D-10 (APPROVED) · Outcomes recorded by the weekly loop, not inside `paper/broker.py`
**Status:** raised as a deviation, **explicitly approved** in ICS-DOC-004 on
2026-08-22 ("توضيح معتمد (D-10)"). The roadmap now states that MFE/MAE are
computed **inside the weekly task**, that `close_position()` stays deterministic
and offline-testable, and that the non-negotiable requirement is only *no
outcome field may be computed or estimated before the position actually closes* —
not where the code lives. The implementation already satisfies this; **no code
change required**.

Original rationale, retained:
ICS-DOC-004 §0.1 says the outcome fields are filled "عند إغلاق الصفقة فعلياً في
`paper/broker.py`". The functional requirement — *only on a real close, never an
advance estimate* — is met: `record_outcomes()` creates a `DecisionOutcome` only
for positions already closed. The **location** differs deliberately.
**Why:** MFE/MAE need the price path between entry and exit. Putting that fetch
inside `close_position()` would give the paper broker network I/O and make it
fail on a flaky data source, on the one code path that must stay deterministic
and offline-testable. **Flag for review** — say the word and the call moves into
`close_position()` with MFE/MAE back-filled by the loop.

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

---

## Phase 1 — machine-learning layer (shadow)

**Date:** 2026-08-22 · Approved parameters: horizon **10 sessions**, **expanding**
training window (2-year minimum), **Spearman IC** tuning objective,
**3 contiguous windows** for the gate.

### D-11 · Ratio features are normalisation, not new features
ICS-DOC-004 Phase 1 §1 says "لا فيتشرز جديدة — استخدم الموجود فقط". All 13 model
inputs are ratios of indicators `feature_engine.py` already computes
(`close/ma50`, `atr14/close`, `volume/volume_ma20`, …).
**Why:** the model trains across symbols, and a raw `close` of 700 (SPY) against
60 (KO) is not comparable — the model would learn the ticker, not the setup.
No new indicator, data source, or lookback was introduced.
**Flag for review** if the constraint was meant to forbid transformations too.

### D-12 · The out-of-sample period is ~2.2 years, not 5
The gate windows came out at ~9 months each rather than the ~20 months implied
by "3 windows over 5 years". The walk-forward warm-up consumes the front of the
data: ~200 bars for MA200 plus the 504-session (2-year) training minimum, so the
first legal prediction is 2024-06 and only ~2.2 years remain out of sample.
**Not hidden, and it does not change the verdict** (the model failed 0/3), but a
longer history or a shorter training minimum would be needed for windows of the
originally intended length.

### D-13 · The gate's portfolio construction is a screening metric
Per-window return/drawdown/Sharpe come from a top-3, equally weighted portfolio
of the model's daily ranking, with each 10-day label spread evenly across its
holding period. That smoothing flatters Sharpe (window 2 reports 6.82, which is
not a realistic figure).
**Why acceptable:** it is the *cheap* gate, run first. A model that cannot pass
it has no case for the expensive full "DQS + ML" counterfactual backtest. Since
the model failed 0/3 windows with a negative IC in two of them, that second
backtest would only have confirmed a negative.

### D-14 · One tuning run per walk-forward, not one per month
`optuna` runs once on the earliest legal training window and those
hyper-parameters are reused for all 27 monthly retrains (the models themselves
are retrained monthly).
**Why:** re-tuning monthly would multiply the trial count by 27, and every trial
counts against the DSR deflation — it would raise the statistical bar enormously
for no modelling benefit on a dataset this size.
