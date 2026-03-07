"""
Volatility-Based Signal Generation — Algorithm S1

Exact implementation of the multi-horizon volatility-aware trading signal
generation procedure from Trading-R1 (Section 3.5, Appendix S2.1).

Algorithm:
  1. Compute EMA(close, span=3)
  2. For each horizon τ ∈ {3, 7, 15}:
     a. Forward return: R_τ = (EMA - EMA.shift(τ)) / EMA.shift(τ)
     b. Rolling volatility: V_τ = R_τ.rolling(20).std()
     c. Sharpe-like signal: S_τ = R_τ / V_τ
  3. Weighted composite: WeightedSignal = Σ(w_τ · S_τ)
  4. Compute percentile thresholds from valid signals
  5. Assign labels based on asymmetric quantile cutoffs:
     {0.03, 0.15, 0.53, 0.85} → {Strong Sell, Sell, Hold, Buy, Strong Buy}
"""
import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)


def compute_ema(prices: pd.Series, span: int = None) -> pd.Series:
    """Compute Exponential Moving Average on close prices."""
    if span is None:
        span = config.EMA_SPAN
    return prices.ewm(span=span, adjust=False).mean()


def compute_forward_returns(ema: pd.Series, horizons: list = None) -> dict:
    """
    Compute forward returns for each horizon.
    R_τ = (EMA - EMA.shift(τ)) / EMA.shift(τ)
    """
    if horizons is None:
        horizons = config.HORIZONS

    returns = {}
    for tau in horizons:
        shifted = ema.shift(tau)
        returns[tau] = (ema - shifted) / shifted
    return returns


def compute_volatility_normalized_signals(
    returns: dict,
    vol_window: int = None
) -> dict:
    """
    Normalize each return series by its rolling volatility.
    V_τ = R_τ.rolling(20).std()
    S_τ = R_τ / V_τ
    """
    if vol_window is None:
        vol_window = config.VOLATILITY_WINDOW

    signals = {}
    for tau, r_series in returns.items():
        volatility = r_series.rolling(window=vol_window).std()
        # Avoid division by zero
        volatility = volatility.replace(0, np.nan)
        signals[tau] = r_series / volatility
    return signals


def compute_weighted_signal(
    signals: dict,
    horizons: list = None,
    weights: list = None
) -> pd.Series:
    """
    Combine normalized signals with empirical weights.
    WeightedSignal = Σ(w_τ · S_τ) with weights {0.3, 0.5, 0.2}
    """
    if horizons is None:
        horizons = config.HORIZONS
    if weights is None:
        weights = config.SIGNAL_WEIGHTS

    weighted = None
    for tau, w in zip(horizons, weights):
        component = signals[tau] * w
        if weighted is None:
            weighted = component
        else:
            weighted = weighted + component

    return weighted


def assign_labels(
    weighted_signal: pd.Series,
    quantiles: list = None
) -> pd.Series:
    """
    Assign 5-class trading labels based on percentile thresholds.
    Thresholds q = {0.03, 0.15, 0.53, 0.85}

    Below q[0]       → STRONG_SELL
    q[0] to q[1]     → SELL
    q[1] to q[2]     → HOLD
    q[2] to q[3]     → BUY
    Above q[3]       → STRONG_BUY
    """
    if quantiles is None:
        quantiles = config.PERCENTILE_THRESHOLDS

    # Only use valid (non-NaN) signals for threshold computation
    valid = weighted_signal.dropna()
    if valid.empty:
        return pd.Series(np.nan, index=weighted_signal.index)

    thresholds = [valid.quantile(q) for q in quantiles]
    logger.info(f"  Signal thresholds: {[round(t, 4) for t in thresholds]}")
    logger.info(f"  Quantile cutoffs: {quantiles}")

    labels = pd.Series(np.nan, index=weighted_signal.index, dtype=object)

    for t in weighted_signal.index:
        x = weighted_signal[t]
        if pd.isna(x):
            labels[t] = np.nan
            continue

        if x >= thresholds[3]:
            labels[t] = "STRONG_BUY"
        elif x >= thresholds[2]:
            labels[t] = "BUY"
        elif x >= thresholds[1]:
            labels[t] = "HOLD"
        elif x >= thresholds[0]:
            labels[t] = "SELL"
        else:
            labels[t] = "STRONG_SELL"

    return labels


def generate_signals(price_data: pd.DataFrame, observer=None, ticker: str = "") -> Tuple[pd.Series, pd.Series, pd.DataFrame]:
    """
    Full Algorithm S1 pipeline.

    Args:
        price_data: DataFrame with 'close' column, DatetimeIndex.
        observer: Optional Observer for detailed logging.
        ticker: Ticker symbol for logging context.

    Returns:
        labels: Series of trading labels (STRONG_SELL..STRONG_BUY)
        weighted_signal: Series of raw composite signal values
        diagnostics: DataFrame with intermediate computations
    """
    if price_data.empty or "close" not in price_data.columns:
        logger.error("Price data is empty or missing 'close' column")
        empty = pd.Series(dtype=object)
        return empty, empty, pd.DataFrame()

    close = price_data["close"]
    logger.info(f"  Generating signals on {len(close)} trading days")

    # Step 1: EMA smoothing
    ema = compute_ema(close)
    if observer:
        observer.log_signal_computation(ticker, f"Step 1: EMA(span={config.EMA_SPAN}) computed on {len(close)} days", ema)

    # Step 2: Forward returns per horizon
    returns = compute_forward_returns(ema)
    if observer:
        for tau in config.HORIZONS:
            valid_count = returns[tau].dropna().shape[0]
            observer.log_signal_computation(ticker, f"Step 2: Forward return τ={tau}d — {valid_count} valid values", returns[tau])

    # Step 3: Volatility normalization
    signals = compute_volatility_normalized_signals(returns)
    if observer:
        for tau in config.HORIZONS:
            s = signals[tau].dropna()
            if not s.empty:
                observer.log_signal_computation(
                    ticker,
                    f"Step 3: Normalized signal τ={tau}d — mean={s.mean():.4f}, std={s.std():.4f}, range=[{s.min():.4f}, {s.max():.4f}]"
                )

    # Step 4: Weighted composite
    weighted_signal = compute_weighted_signal(signals)
    if observer:
        ws = weighted_signal.dropna()
        if not ws.empty:
            observer.log_signal_computation(
                ticker,
                f"Step 4: Weighted composite — mean={ws.mean():.4f}, std={ws.std():.4f}, weights={config.SIGNAL_WEIGHTS}"
            )

    # Step 5: Label assignment
    labels = assign_labels(weighted_signal)

    # Log thresholds via observer
    if observer:
        valid = weighted_signal.dropna()
        if not valid.empty:
            thresholds = [valid.quantile(q) for q in config.PERCENTILE_THRESHOLDS]
            observer.log_signal_thresholds(ticker, thresholds, config.PERCENTILE_THRESHOLDS)

    # Build diagnostics DataFrame
    diag = pd.DataFrame({"close": close, "ema": ema, "weighted_signal": weighted_signal})
    for tau in config.HORIZONS:
        diag[f"return_{tau}d"] = returns[tau]
        diag[f"signal_{tau}d"] = signals[tau]
    diag["label"] = labels

    # Log distribution via observer (or fallback to standard logger)
    if observer:
        observer.log_signal_distribution(ticker, labels)
    else:
        label_counts = labels.value_counts()
        total_valid = label_counts.sum()
        if total_valid > 0:
            logger.info("  Label distribution:")
            for lbl in config.ACTION_LABELS:
                count = label_counts.get(lbl, 0)
                pct = count / total_valid * 100
                logger.info(f"    {lbl:>12s}: {count:4d} ({pct:5.1f}%)")

    return labels, weighted_signal, diag


def get_signal_for_date(
    labels: pd.Series,
    target_date: str
) -> Optional[str]:
    """Get the trading signal for a specific date."""
    target_dt = pd.Timestamp(target_date)

    # Find exact match or most recent prior signal
    available = labels[labels.index <= target_dt].dropna()
    if available.empty:
        return None
    return available.iloc[-1]


class SignalGenerator:
    """
    Wraps the signal generation pipeline for a single ticker.
    Accepts an optional Observer for rich logging.
    """

    def __init__(self, price_data: pd.DataFrame, ticker: str = "", observer=None):
        self.ticker = ticker
        self.price_data = price_data
        self.observer = observer
        self.labels: Optional[pd.Series] = None
        self.weighted_signal: Optional[pd.Series] = None
        self.diagnostics: Optional[pd.DataFrame] = None

    def generate(self) -> pd.Series:
        """Run Algorithm S1 and return the label series."""
        logger.info(f"═══ Signal generation for {self.ticker} ═══")
        self.labels, self.weighted_signal, self.diagnostics = generate_signals(
            self.price_data, observer=self.observer, ticker=self.ticker
        )
        return self.labels

    def get_label(self, date: str) -> Optional[str]:
        """Get the label for a specific trading date."""
        if self.labels is None:
            self.generate()
        return get_signal_for_date(self.labels, date)

    def get_diagnostics(self) -> pd.DataFrame:
        """Return intermediate computation details."""
        if self.diagnostics is None:
            self.generate()
        return self.diagnostics
