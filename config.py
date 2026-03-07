"""
Trading-R1 Backtest Configuration
All parameters match the paper: "Trading-R1: Financial Trading with LLM Reasoning via RL"
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ── API Keys ──────────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "")

# ── Algorithm S1 Parameters (Section 3.5 & Appendix S2.1) ────────────────────
EMA_SPAN = 3                          # EMA span for smoothing prices
HORIZONS = [3, 7, 15]                 # Forward return horizons (days)
SIGNAL_WEIGHTS = [0.3, 0.5, 0.2]     # Weights for each horizon signal
VOLATILITY_WINDOW = 20                # Rolling window for volatility normalization
PERCENTILE_THRESHOLDS = [0.03, 0.15, 0.53, 0.85]  # Asymmetric quantile cutoffs

# ── Trading Action Space (Section 3.7, Table 2) ──────────────────────────────
ACTION_LABELS = ["STRONG_SELL", "SELL", "HOLD", "BUY", "STRONG_BUY"]
POSITION_WEIGHTS = {
    "STRONG_SELL": -1.0,
    "SELL":        -0.5,
    "HOLD":         0.0,
    "BUY":          0.5,
    "STRONG_BUY":   1.0,
}

# ── Portfolio & Backtest Settings ─────────────────────────────────────────────
INITIAL_CAPITAL = 100_000.0
TRANSACTION_COST_BPS = 10             # 0.1% = 10 bps
REBALANCE_FREQUENCY_DAYS = 5          # Weekly rebalancing (~1 week holding)
RISK_FREE_RATE_ANNUAL = 0.04          # 4% (10Y Treasury benchmark, per Appendix S2)
TRADING_DAYS_PER_YEAR = 252

# ── Paper Tickers (Table S3, Appendix S1.3) ───────────────────────────────────
PAPER_TICKERS = [
    "NVDA", "AAPL", "MSFT",          # Info Tech
    "META",                            # Communication Services
    "AMZN", "TSLA",                    # Consumer Discretionary
    "BRK-B", "JPM",                    # Financials
    "LLY", "JNJ",                      # Health Care
    "XOM", "CVX",                      # Energy
    "SPY", "QQQ",                      # ETFs
]

# ── Default Backtest Date Range ───────────────────────────────────────────────
DEFAULT_START_DATE = "2024-06-01"      # Paper held-out period start
DEFAULT_END_DATE = "2024-08-31"        # Paper held-out period end
DATA_LOOKBACK_DAYS = 60               # Extra history for indicator warm-up

# ── Technical Indicators (Table S2) ───────────────────────────────────────────
TECHNICAL_INDICATORS = {
    "moving_averages": {"sma": [50, 200], "ema": [10, 50]},
    "macd": {"fast": 12, "slow": 26, "signal": 9},
    "rsi": {"period": 14},
    "bollinger": {"period": 20, "std_dev": 2},
    "atr": {"period": 14},
    "adx": {"period": 14},
    "stochastic": {"k_period": 14, "d_period": 3},
}

# ── News Collection (Table S1) ────────────────────────────────────────────────
NEWS_BUCKETS = {
    "last_3_days":   {"days_back": 3,  "max_samples": 10},
    "last_4_10_days": {"days_back": 10, "max_samples": 20},
    "last_11_30_days": {"days_back": 30, "max_samples": 20},
}

# ── LLM Settings ──────────────────────────────────────────────────────────────
LLM_MODEL = "claude-sonnet-4-20250514"
LLM_MAX_TOKENS = 4096
LLM_TEMPERATURE = 0.3
LLM_MAX_RETRIES = 3
LLM_RETRY_DELAY = 2.0                # Base delay in seconds (exponential backoff)

# ── Output Settings ───────────────────────────────────────────────────────────
RESULTS_DIR = "results"
LOG_LEVEL = "INFO"
