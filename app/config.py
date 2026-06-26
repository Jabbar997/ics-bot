"""Configuration loading for ICS.

Configuration has two layers:

* ``config.yaml`` — non-secret business rules (risk limits, watchlist, ...).
* ``.env`` / environment — secrets and deployment-specific values
  (Telegram token, allowed user IDs, database URL).

All business rules are centralised here as typed pydantic models so the rest of
the codebase never reads raw dictionaries.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import List

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Repository root (the directory containing config.yaml / .env).
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


# --------------------------------------------------------------------------- #
# Secrets / environment layer
# --------------------------------------------------------------------------- #
class EnvSettings(BaseSettings):
    """Secrets and deployment values, read from the environment / ``.env``."""

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    telegram_bot_token: str = Field(default="", alias="TELEGRAM_BOT_TOKEN")
    telegram_allowed_user_ids: str = Field(default="", alias="TELEGRAM_ALLOWED_USER_IDS")
    database_url: str = Field(default="sqlite:///./ics.db", alias="DATABASE_URL")

    @property
    def allowed_user_ids(self) -> List[int]:
        raw = (self.telegram_allowed_user_ids or "").strip()
        if not raw:
            return []
        ids: List[int] = []
        for part in raw.split(","):
            part = part.strip()
            if part:
                ids.append(int(part))
        return ids


# --------------------------------------------------------------------------- #
# Business-rule layer (mirrors config.yaml)
# --------------------------------------------------------------------------- #
class CapitalConfig(BaseModel):
    initial_capital_usd: float = 266.0
    base_currency: str = "USD"


class RiskConfig(BaseModel):
    max_open_positions: int = 3
    max_position_size_pct: float = 10.0
    weekly_loss_limit_pct: float = 5.0
    monthly_loss_limit_pct: float = 12.0
    max_drawdown_limit_pct: float = 15.0
    minimum_dqs: int = 70
    target_average_dqs: int = 75
    stop_loss_atr_multiplier: float = 2.0
    absolute_stop_loss_pct: float = 7.0


class BenchmarkConfig(BaseModel):
    symbol: str = "SPY"


class MarketConfig(BaseModel):
    data_provider: str = "yfinance"
    timeframe: str = "1d"
    history_years: int = 5
    timezone: str = "America/New_York"


class PaperConfig(BaseModel):
    commission_per_trade_usd: float = 0.0
    slippage_pct: float = 0.0
    allow_fractional_shares: bool = True


class TelegramConfig(BaseModel):
    enabled: bool = True
    allowed_user_ids: List[int] = Field(default_factory=list)
    daily_report_time_ksa: str = "01:15"
    weekly_report_day: str = "FRI"
    # v1.1: short daily status/backup heartbeat time (KSA).
    status_report_time_ksa: str = "13:00"


class Config(BaseModel):
    """Top-level, fully-typed system configuration."""

    mode: str = "paper_only"
    capital: CapitalConfig = Field(default_factory=CapitalConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    benchmark: BenchmarkConfig = Field(default_factory=BenchmarkConfig)
    market: MarketConfig = Field(default_factory=MarketConfig)
    paper: PaperConfig = Field(default_factory=PaperConfig)
    telegram: TelegramConfig = Field(default_factory=TelegramConfig)
    watchlist: List[str] = Field(default_factory=list)
    forbidden_assets: List[str] = Field(default_factory=list)

    # Secrets are attached after YAML load (not stored in config.yaml).
    env: EnvSettings = Field(default_factory=EnvSettings)

    @field_validator("mode")
    @classmethod
    def _mode_must_be_paper(cls, v: str) -> str:
        # Hard safety rail: this MVP is paper-only. Real trading is forbidden.
        if v != "paper_only":
            raise ValueError(
                f"Unsupported mode {v!r}. ICS MVP only supports 'paper_only'. "
                "Real-money trading is forbidden in this build."
            )
        return v

    @field_validator("watchlist")
    @classmethod
    def _watchlist_upper(cls, v: List[str]) -> List[str]:
        return [s.strip().upper() for s in v if s and s.strip()]

    def is_in_watchlist(self, symbol: str) -> bool:
        return symbol.strip().upper() in self.watchlist

    @property
    def initial_capital(self) -> float:
        return self.capital.initial_capital_usd


def load_config(path: str | os.PathLike[str] | None = None) -> Config:
    """Load and validate configuration from YAML + environment."""
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    data: dict = {}
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}

    env = EnvSettings()
    config = Config(**data)
    config.env = env

    # Env-provided allowed IDs take precedence over the (empty) YAML default.
    if env.allowed_user_ids:
        config.telegram.allowed_user_ids = env.allowed_user_ids

    return config


@lru_cache(maxsize=1)
def get_config() -> Config:
    """Cached accessor for the default configuration."""
    return load_config()
