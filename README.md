# ICS — Investment Command System (MVP v1.0)

> **⚠️ PAPER TRADING ONLY — EDUCATIONAL SIMULATION.**
> This system never connects to a real broker, never moves real money, and never
> places real orders. It does **not** provide financial advice and does **not**
> guarantee profit. It exists to *simulate, score, and audit* investment
> decisions for learning and internal research only. Mode is **Analyst → Trainee**.
> Real-money trading is out of scope and forbidden in this build (see
> *Future Roadmap → ICS-DOC-003*).

ICS is a modular decision-and-audit engine, not a "trading bot". Every candidate
decision flows through a fixed pipeline, is scored, risk-checked, executed only
on a **virtual** portfolio, and written to an immutable audit log.

---

## 1. Architecture

```
Market Data Layer (yfinance, read-only)
      ↓
Data Cleaner          (dedupe, sort, fill, reject incomplete)
      ↓
Feature Engine        (MA/RSI/MACD/ATR/vol/beta — manual, transparent)
      ↓
Market Regime Analyzer (SPY → bull / weak_bull / sideways / bear / panic)
      ↓
Strategy Engine       (trend, momentum, pullback, defensive_cash)
      ↓
Decision Quality Score (0–100, 5 weighted components, min 70)
      ↓
Risk Manager          (size, stops, limits, watchlist, kill switch)
      ↓
Paper Trading Engine  (virtual broker — NO real orders ever)
      ↓
Audit Log             (every decision; a decision without one is invalid)
      ↓
Performance Evaluator (return, Sharpe, drawdown, win rate, expectancy)
      ↓
Telegram Reporting Bot (reports + control commands, auth-gated)
```

Starting capital: **$266.00**. Benchmark: **SPY**.

### Project layout

```
ics/
  app/
    main.py            # CLI + daily/weekly/backtest/bot/scheduler/demo workflows
    config.py          # pydantic config (YAML + .env)
    logging_config.py
    domain.py          # shared enums + dataclasses (Signal, DQSResult, ...)
    data/              # market_data.py (yfinance), cleaner.py
    features/          # indicators.py, feature_engine.py
    market/            # regime.py
    strategies/        # base, trend, momentum, pullback, defensive_cash, engine
    decision/          # dqs.py, decision_engine.py
    risk/              # risk_manager.py, kill_switch.py
    paper/             # broker.py, portfolio.py, orders.py  (virtual only)
    performance/       # evaluator.py, benchmarks.py
    telegram/          # bot.py, reports.py, commands.py
    db/                # database.py, models.py, repositories.py
    backtesting/       # backtester.py, walk_forward.py
    utils/             # time.py, money.py
  tests/               # pytest suite (50 tests)
  config.yaml  .env.example  requirements.txt  README.md
```

---

## 2. Setup

Requires **Python 3.11+**.

```bash
cd ics
python3.11 -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env                # then edit .env
python -m app.main init-db          # create tables + sync watchlist
```

### Environment variables (`.env`)

| Variable                    | Purpose                                                       |
|-----------------------------|---------------------------------------------------------------|
| `TELEGRAM_BOT_TOKEN`        | Bot token from @BotFather. Leave blank to run without Telegram |
| `TELEGRAM_ALLOWED_USER_IDS` | Comma-separated numeric Telegram user IDs allowed to use the bot |
| `DATABASE_URL`              | SQLAlchemy URL. Default `sqlite:///./ics.db` (Postgres-ready)  |

Secrets live only in `.env` — never in `config.yaml`, never logged, never sent
to users.

---

## 3. Commands

```bash
python -m app.main init-db      # create schema + sync watchlist
python -m app.main demo         # OFFLINE synthetic daily cycle (no network) — great smoke test
python -m app.main daily        # one live paper cycle (fetches yfinance data)
python -m app.main daily --send # ... and push the daily report to Telegram
python -m app.main weekly       # build the weekly report (add --send to push)
python -m app.main backtest     # 5y backtest + walk-forward split summary
python -m app.main backtest --period 5y
python -m app.main bot          # run the Telegram bot (long-polling)
python -m app.main scheduler    # APScheduler: daily + weekly jobs (KSA times)
```

### Run the Telegram bot

1. Set `TELEGRAM_BOT_TOKEN` and `TELEGRAM_ALLOWED_USER_IDS` in `.env`.
2. `python -m app.main bot`
3. In Telegram, message your bot `/start`.

Commands: `/start /status /portfolio /positions /today /weekly /rules
/watchlist /audit /rejected /performance /kill /stop /resume`. Sending plain
`STOP` (or `/stop`) **freezes the system immediately** (cancels new entries; no
real positions exist to close). `/resume` clears the freeze. Unknown users get
exactly `Unauthorized.`

### Run a backtest

```bash
python -m app.main backtest
```

Pulls ~5y of daily data for the watchlist + SPY, prints the Train (70%) /
Validation (20%) / Walk-forward (10%) date ranges, then replays the **exact live
pipeline** and prints returns, Sharpe, max drawdown, win rate, average DQS, and
the audit-log count.

### Run the daily paper workflow manually

```bash
python -m app.main daily        # uses live yfinance data
# or, with no network:
python -m app.main demo         # deterministic synthetic data
```

### Run the tests

```bash
python -m pytest                # 50 tests, no network required
```

---

## 4. Configuration guide (`config.yaml`)

All business rules are centralised here (and in dedicated rule modules), never
scattered in code.

- `capital.initial_capital_usd` — starting virtual capital (**266.0**).
- `risk.*` — max open positions (3), max position size (10%), weekly/monthly loss
  limits (−5% / −12%), max drawdown (−15%), `minimum_dqs` (70), stop-loss as
  `min(2×ATR, 7%)`.
- `benchmark.symbol` — SPY.
- `market.*` — provider (yfinance), timeframe (1d), history (5y), timezone.
- `paper.*` — commission per trade (0.0, configurable) and slippage % (0.0).
- `telegram.*` — enabled flag, allowed IDs, daily report time (KSA), weekly day.
- `watchlist` — the 20 allowed US stocks/ETFs. **Symbols outside the watchlist
  are rejected** by the risk manager unless you change this list.
- `forbidden_assets` — crypto, options, futures, forex, penny stocks, leverage,
  short selling (documented and enforced: the broker is long-only & cash-funded).

### Decision Quality Score (DQS)

| Component               | Max |
|-------------------------|-----|
| Strategy alignment      | 25  |
| Risk management         | 25  |
| Timing quality          | 20  |
| Market regime strength  | 15  |
| Reason clarity          | 15  |
| **Total**               | 100 |

DQS ≥ 70 → eligible (if the risk manager also approves). DQS < 70 → rejected and
logged as a **rejected opportunity**. Target average DQS ≥ 75.

### Execution convention (no look-ahead)

- **Backtest:** signal generated after the close of day *t*; the order fills at
  the **next trading day's open** (`t+1`), and only if that bar exists.
  Indicators are strictly backward-looking. This is verified by
  `tests/test_backtester.py::test_backtester_no_lookahead_fills_at_next_open`.
- **Live daily:** fills at the latest available close (the best paper proxy when
  acting on the most recent bar).

### Kill Switch

| Level | Trigger | Action |
|-------|---------|--------|
| L1 Warning | weekly ≤ −5% or 3 consecutive losses | stop new entries, report, 48h cooldown |
| L2 Freeze | monthly ≤ −8% or severe-event flag | close 50% of paper positions, stop entries, review |
| L3 Full Stop | monthly ≤ −12% or drawdown > 15% | close all paper positions, freeze, manual review |
| Manual STOP | `STOP` / `/stop` | immediate freeze, cancel entries (no real positions) |

---

## 4b. v1.1 — Stability & Persistence Upgrade

**PostgreSQL support (SQLite stays the local default).**
Set `DATABASE_URL` to a Postgres URL and ICS uses it; leave it as SQLite for
local dev. The DB layer auto-normalises `postgres://` / `postgresql://` URLs to
the psycopg driver, so a managed provider's URL works as-is. `init-db`
(`create_all`) is idempotent — safe on an empty DB and a no-op on an existing one
(it never wipes data on restart). **No trading/strategy/DQS logic changed.**

```bash
# Local (default): SQLite
DATABASE_URL=sqlite:///./ics.db

# Production: PostgreSQL (driver added automatically)
DATABASE_URL=postgres://user:pass@host:5432/dbname
```

**`/health` command** — reports, with **no secrets ever** (only the dialect name):
bot status, scheduler status, database connection, `paper_only` mode, kill-switch
status, last decision-cycle time, last daily-report time, and the
**decisions == audit_logs** invariant.

**Daily status/backup heartbeat** — a short scheduled report (default `13:00`
KSA, `telegram.status_report_time_ksa`) confirming health and persisting a
lightweight state checkpoint (`last_state_backup` in `system_config`).

### Setting up PostgreSQL on Render
1. Render → **New → PostgreSQL** → create the database.
2. Copy its **Internal Database URL** (same-account/region; no SSL needed).
3. In the `ics-bot` worker → **Environment** → set
   `DATABASE_URL` = that URL. (No other new variables.)
4. The SQLite disk on `/var/data` is no longer required once on Postgres
   (keeping it is harmless).
5. Redeploy. On first boot `init-db` creates the schema in the empty Postgres;
   subsequent restarts are no-ops. Managed Postgres also gives you provider-side
   backups.

---

## 4c. v1.2 — Reporting Correctness

Three defects surfaced by 56 days of live paper trading, plus a stale-data fix.
**No trading, strategy, or DQS logic was changed.**

| Fix | Before | After |
|---|---|---|
| **Weekly return** | aggregated *all* history and labelled it "weekly" | `period_return` covers the last 7 days; cumulative shown on its own line |
| **Benchmark in `/weekly`** | manual command always printed SPY `+0.00%` (only the scheduled job computed it) | computed on demand, with a safe `0.0` fallback if the fetch fails |
| **"Rule violations"** | counted risk-manager *blocks* — i.e. rules being **enforced** — under an alarming label | `قرارات أوقفها مدير المخاطر` (blocks) + `مخالفات فعلية: 0` |
| **Weekend/stale bars** | re-ran the cycle on repeated Friday data (~28% of runs), inflating rejects | cycle skips when the benchmark's latest bar was already processed; `--force` overrides |

Command handlers now run off the event loop, so a command that touches the
network can no longer stall polling.

```bash
python -m app.main daily           # skips if no new market bar
python -m app.main daily --force   # run anyway
```

---

## 4d. ICS-DOC-004 Phase 0 — DQS learning feedback loop

The system now learns which parts of its own scoring actually predicted returns,
and re-balances the DQS weights within hard bounds. **Paper-only: the loop can
only change how candidates are scored — never how they are sized, risked, or
exited.**

**How one weekly cycle works**
1. Record a `DecisionOutcome` for each newly closed position — `realized_return`,
   `holding_period_days`, `mfe`, `mae`, plus the DQS components scored at entry.
2. **Gate:** fewer than **30** closed trades → nothing changes (still recorded).
3. Spearman correlation (scipy) between each component's points and the realised return.
4. Weights nudged toward what predicted returns, capped at **±5% of each
   component's own weight per cycle**, floor 5 / ceiling 40.
5. Zero-sum re-balance → the weights still sum to **exactly 100**.
6. A `LearningEvent` is written **every** cycle — applied or skipped — with
   correlations, before/after weights, and trade count (AuditLog-grade rigour).

Scheduled weekly, 5 minutes after the weekly report (so that report reflects the
pre-update weights). Weights persist in `system_config` under `dqs_weights`;
a database that never ran a cycle behaves exactly as before Phase 0.

```
/learning     # current DQS weights, recorded outcomes, recent learning cycles
```

**Behaviour-neutral until it learns:** with default weights `calculate_dqs()`
returns bit-identical scores to the pre-Phase-0 implementation, and a 5-year
backtest on a fixed dataset reproduces the baseline exactly
(+24.75% / −5.43% / Sharpe 1.11 / 164 trades, decisions == audit logs).

New dependency: `scipy==1.17.1`. See `DECISIONS_AND_ASSUMPTIONS.md` for the
assumptions made (notably how the ±5% cap is interpreted).

---

## 5. Safety model

- **No real broker integration anywhere.** The only execution path is
  `app/paper/broker.py`, which mutates a local virtual portfolio. There is no
  Alpaca / IBKR / Robinhood code, no API keys for order routing, no
  `real_order()` / `live_trade()` / margin / short / options / crypto.
- `mode` is validated to be `paper_only`; any other value raises at startup.
- The broker is **long-only and cash-funded** — it will never spend cash it
  doesn't have, so the portfolio can't go negative (no leverage/margin).
- Telegram is **auth-gated**: only configured user IDs are served; everyone else
  gets `Unauthorized.` Errors are logged server-side; users never see stack
  traces or tokens.

---

## 6. Known limitations

- Indicator warm-up requires ≥ ~210 clean bars; symbols with less history are
  rejected (never decided on from incomplete data).
- Live data quality depends entirely on yfinance (free, occasionally flaky); the
  fetch layer cleans and validates but cannot fix upstream gaps.
- Strategy logic is intentionally simple/explainable, not optimised; backtests on
  short windows will not reach the 100-trade qualification bar.
- Beta uses a rolling 60-day window vs SPY and is informational only.
- The scheduler/bot run as separate long-lived processes; there is no built-in
  process supervisor.
- "Severe economic/news event" (kill-switch L2) is a manual flag — no news feed
  is wired in for the MVP.

---

## 7. Future roadmap

- **ICS-DOC-003 — Real Trading Governance** (separate document; **not** in this
  build): any real-money path must be designed, reviewed, and governed there.
- Postgres migration (the repository layer is already isolated for this).
- Richer strategies + parameter optimisation with proper walk-forward validation.
- News / macro event ingestion to drive the L2 severe-event flag automatically.
- Web dashboard alongside the Telegram reports.
- Per-strategy attribution and a learning-note feedback loop on the audit log.

---

*ICS is for educational and internal simulation only. Nothing here is financial
advice.*
