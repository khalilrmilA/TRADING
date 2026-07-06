"""Central configuration for the Trading AI Platform.

PAPER TRADING ONLY — this platform performs research, backtesting and simulated
trading. It never places real orders and never uses private exchange API keys.

All values can be overridden via environment variables or a `.env` file at the
project root (see `.env.example`).
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    """Application settings loaded from environment / .env."""

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Ollama / AI ---------------------------------------------------------
    ollama_url: str = "http://localhost:11434"
    # qwen3:14b fits entirely in a 16 GB VRAM GPU (e.g. RTX 5060 Ti) and is the
    # fastest high-quality option on this class of hardware. The 32B models
    # below are fully supported but partially offload to CPU on 16 GB cards.
    ollama_model: str = "qwen3:14b"
    ollama_fallback_model: str = "deepseek-r1:32b"
    ollama_code_model: str = "qwen2.5-coder:32b"
    ollama_timeout: float = 180.0
    ollama_retries: int = 3

    # --- Storage --------------------------------------------------------------
    db_path: Path = PROJECT_ROOT / "data" / "trading.db"

    # --- API / dashboard -------------------------------------------------------
    api_host: str = "127.0.0.1"
    api_port: int = 8000

    # --- Paper account ---------------------------------------------------------
    initial_capital: float = 100_000.0
    commission_rate: float = 0.001  # 0.1% per fill
    slippage_rate: float = 0.0005  # 0.05% adverse per market/stop fill

    # --- Risk management -------------------------------------------------------
    risk_per_trade: float = 0.01  # max 1% of equity risked per trade
    max_open_positions: int = 5
    daily_loss_limit: float = 0.03  # halt new entries at -3% daily PnL
    circuit_breaker_drawdown: float = 0.10  # halt all trading at 10% drawdown
    atr_stop_multiplier: float = 2.0
    atr_take_profit_multiplier: float = 3.0

    # --- Data updates ----------------------------------------------------------
    data_update_interval_minutes: int = 15

    # --- Improvement pack v3 -----------------------------------------------------
    heat_cap_fraction: float = 0.06  # portfolio stop-distance risk cap vs equity
    max_same_direction: int = 5  # max open positions in one direction, platform-wide
    # Windows install path of the Ollama desktop app; %VARS% are expanded with
    # os.path.expandvars at USE time (the watchdog), never at load time.
    ollama_app_path: str = "%LOCALAPPDATA%/Programs/Ollama/ollama app.exe"
    ollama_watchdog_enabled: bool = True

    @property
    def api_url(self) -> str:
        """Base URL of the backend API."""
        return f"http://{self.api_host}:{self.api_port}"


settings = Settings()
settings.db_path.parent.mkdir(parents=True, exist_ok=True)
