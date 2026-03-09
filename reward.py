"""
Multi-Dimensional Reward Function — CIO-Grade Analyst Evaluation

Computes 8 reward dimensions that evaluate not just financial outcome,
but analytical quality, risk discipline, learning ability, and reasoning coherence.

Dimensions:
  1. sharpe          — Risk-adjusted P&L (Sharpe-like)
  2. direction       — Did the position direction match the price move?
  3. conviction      — Was position sizing proportional to move magnitude?
  4. improvement     — Is forecast error decreasing over time?
  5. override        — When disagreeing with Algorithm S1, was it right?
  6. risk_discipline — Appropriate sizing for the volatility regime?
  7. coherence       — Did reasoning align with what actually happened?
  8. regime          — Did the action match the market regime?

The composite scalar = weighted sum of all dimensions (for GRPO training).
Per-dimension breakdown is shown in ICRL prompts for richer learning.

This reward function is shared by both ICRL and GRPO modes.
"""
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import config
import rl_config


@dataclass
class RewardDimensions:
    """Per-dimension scores for a single trade evaluation."""
    sharpe: float = 0.0
    direction: float = 0.0
    conviction: float = 0.0
    improvement: float = 0.0
    override: float = 0.0
    risk_discipline: float = 0.0
    coherence: float = 0.0
    regime: float = 0.0
    composite: float = 0.0
    weights: Dict[str, float] = field(default_factory=lambda: dict(rl_config.REWARD_DIMENSION_WEIGHTS))

    def to_dict(self) -> Dict[str, float]:
        """Return all dimension scores as a dict."""
        return {
            "sharpe": self.sharpe,
            "direction": self.direction,
            "conviction": self.conviction,
            "improvement": self.improvement,
            "override": self.override,
            "risk_discipline": self.risk_discipline,
            "coherence": self.coherence,
            "regime": self.regime,
            "composite": self.composite,
        }

    def summary_str(self) -> str:
        """One-line summary of all dimension scores."""
        parts = []
        for name, score in self.to_dict().items():
            if name == "composite":
                continue
            parts.append(f"{name}={score:+.2f}")
        return f"composite={self.composite:+.3f} [{', '.join(parts)}]"


@dataclass
class RewardRecord:
    """Record of a computed reward for a completed trade."""
    ticker: str
    entry_date: str
    exit_date: str
    action: str
    position_weight: float
    entry_price: float
    exit_price: float
    holding_days: int
    cumulative_return: float
    holding_volatility: float
    transaction_cost: float
    reward: float          # Composite scalar (weighted sum of dimensions)
    raw_pnl: float
    dimensions: Optional[RewardDimensions] = None


# ─────────────────────────────────────────────────────────────────────────────
# Individual dimension computation functions
# Each returns a score normalized to approximately [-1, +1]
# ─────────────────────────────────────────────────────────────────────────────

def _compute_sharpe_score(
    position_weight: float,
    cumulative_return: float,
    tx_cost: float,
    rf_cost: float,
    holding_vol: float,
) -> float:
    """
    Dimension 1: Risk-adjusted P&L (existing Sharpe-like).
    Clipped to [-3, +3] then normalized to [-1, +1].
    """
    raw_pnl = position_weight * cumulative_return
    sharpe_raw = (raw_pnl - tx_cost - rf_cost) / holding_vol
    # Normalize: clip to [-3, 3] and divide by 3
    clipped = max(-3.0, min(3.0, sharpe_raw))
    return clipped / 3.0


def _compute_direction_score(
    position_weight: float,
    cumulative_return: float,
) -> float:
    """
    Dimension 2: Directional accuracy.
    +1.0 if position direction matches price move, -1.0 if opposite, 0.0 for HOLD.
    """
    if position_weight == 0.0:
        return 0.0

    price_direction = 1.0 if cumulative_return > 0 else (-1.0 if cumulative_return < 0 else 0.0)
    position_direction = 1.0 if position_weight > 0 else -1.0

    if price_direction == 0.0:
        return 0.0  # Flat market — no signal

    return 1.0 if position_direction == price_direction else -1.0


def _compute_conviction_score(
    position_weight: float,
    cumulative_return: float,
    return_cap: float = None,
) -> float:
    """
    Dimension 3: Conviction calibration.
    Measures whether position sizing was proportional to the actual move.
    Score = 1.0 - |abs(position_weight) - normalized_abs_return|, mapped to [-1, +1].
    """
    if position_weight == 0.0:
        # HOLD: slightly penalize if there was a big move
        cap = return_cap or rl_config.REWARD_RETURN_CAP
        normalized_move = min(abs(cumulative_return) / cap, 1.0)
        return 1.0 - normalized_move  # Penalized for missing big moves

    cap = return_cap or rl_config.REWARD_RETURN_CAP
    abs_weight = abs(position_weight)
    abs_return = abs(cumulative_return)

    # Normalize return to [0, 1] using cap
    normalized_return = min(abs_return / cap, 1.0)

    # Perfect calibration: abs_weight matches normalized_return
    calibration_error = abs(abs_weight - normalized_return)

    # Map to [-1, +1]: 0 error = +1, 1.0 error = -1
    return 1.0 - 2.0 * calibration_error


def _compute_improvement_score(
    current_forecast_error: float,
    past_forecast_errors: List[float],
) -> float:
    """
    Dimension 4: Forecast error improvement.
    +1.0 if current error is significantly better than rolling average,
    -1.0 if significantly worse, 0.0 if at average.
    """
    if not past_forecast_errors:
        return 0.0  # No history to compare against

    trailing_mean = np.mean(past_forecast_errors)
    if trailing_mean < 1e-10:
        return 0.0  # All errors are near zero — can't improve

    # Relative improvement: (trailing_mean - current) / trailing_mean
    # Positive = improved, negative = worse
    improvement_ratio = (trailing_mean - current_forecast_error) / trailing_mean

    # Clip to [-1, +1]
    return max(-1.0, min(1.0, improvement_ratio))


def _compute_override_score(
    action: str,
    signal_label: str,
    cumulative_return: float,
    position_weight: float,
) -> float:
    """
    Dimension 5: Signal override quality.
    When the model disagrees with Algorithm S1, was it right?
    +1.0 for profitable overrides, -1.0 for unprofitable, 0.0 for agreement.
    """
    if action == signal_label:
        return 0.0  # Following the signal — neutral

    # Model overrode the signal
    pnl = position_weight * cumulative_return
    if pnl > 0:
        return 1.0    # Profitable override — good alpha
    elif pnl < 0:
        return -1.0   # Unprofitable override — bad call
    else:
        return 0.0    # Break-even override


def _compute_risk_discipline_score(
    position_weight: float,
    trailing_vol: float,
    vol_history: np.ndarray,
) -> float:
    """
    Dimension 6: Risk discipline — appropriate sizing for volatility regime.
    In high-vol: reward small positions, penalize large ones.
    In low-vol: reward deploying capital, penalize excessive caution.
    """
    if len(vol_history) < 5:
        return 0.0  # Not enough history

    # Determine vol regime percentile
    vol_percentile = np.mean(vol_history <= trailing_vol)
    abs_weight = abs(position_weight)

    high_pct = rl_config.REWARD_VOL_HIGH_PERCENTILE
    low_pct = rl_config.REWARD_VOL_LOW_PERCENTILE

    if vol_percentile > high_pct:
        # High-vol regime: reward smaller positions
        # |weight| = 0 → +1.0, |weight| = 1.0 → -1.0
        return 1.0 - 2.0 * abs_weight
    elif vol_percentile < low_pct:
        # Low-vol regime: reward deploying capital
        # |weight| = 1.0 → +1.0, |weight| = 0 → -0.5
        return -0.5 + 1.5 * abs_weight
    else:
        # Normal regime: neutral, slight preference for moderate sizing
        ideal_weight = 0.5
        deviation = abs(abs_weight - ideal_weight)
        return 1.0 - 2.0 * deviation


def _compute_coherence_score(
    reasoning: str,
    cumulative_return: float,
    holding_vol: float,
    vol_floor: float = None,
) -> float:
    """
    Dimension 7: Analytical coherence.

    Three sub-components:
      1. Directional coherence: Did bullish/bearish language match the outcome?
      2. Volatility coherence: Did vol language match realized volatility?
      3. Counter-argument quality: Did reasoning address opposing views?

    The counter-argument component rewards debate-quality thinking:
      - Addressing counter-arguments on winning trades → bonus (thorough analysis)
      - Addressing counter-arguments on losing trades → smaller penalty (at least rigorous)
      - NO counter-arguments on losing trades → extra penalty (overconfident failure)
      - NO counter-arguments on winning trades → no bonus (lucky, not rigorous)
    """
    if not reasoning:
        return 0.0

    reasoning_lower = reasoning.lower()
    vol_floor = vol_floor or rl_config.REWARD_VOLATILITY_FLOOR

    # Count bullish/bearish keyword mentions
    bullish_count = sum(1 for kw in rl_config.REWARD_COHERENCE_BULLISH_KEYWORDS
                        if kw in reasoning_lower)
    bearish_count = sum(1 for kw in rl_config.REWARD_COHERENCE_BEARISH_KEYWORDS
                        if kw in reasoning_lower)
    vol_count = sum(1 for kw in rl_config.REWARD_COHERENCE_VOL_KEYWORDS
                    if kw in reasoning_lower)

    # Count counter-argument keywords
    counter_count = sum(1 for kw in rl_config.REWARD_COHERENCE_COUNTER_KEYWORDS
                        if kw in reasoning_lower)

    total_keywords = bullish_count + bearish_count + vol_count
    if total_keywords == 0 and counter_count == 0:
        return 0.0  # No relevant keywords found

    # ── Additive scoring: base + adjustments, clipped to [-1, +1] ──
    #
    # Base signal: directional coherence (did sentiment match outcome?)
    # Adjustments: vol coherence (small), counter-argument quality (moderate)
    #
    # This avoids the dilution problem of component-averaging.

    # Base: Directional coherence [-0.7, +0.7]
    # Scaled below 1.0 to leave headroom for bonus adjustments
    base_score = 0.0
    if bullish_count > 0 or bearish_count > 0:
        net_sentiment = bullish_count - bearish_count
        if cumulative_return > 0.001:
            base_score = 0.7 if net_sentiment > 0 else -0.7
        elif cumulative_return < -0.001:
            base_score = 0.7 if net_sentiment < 0 else -0.7

    # Adjustment 1: Volatility coherence [-0.15, +0.15]
    vol_adjustment = 0.0
    if vol_count > 0:
        is_volatile = holding_vol > 0.02
        vol_adjustment = 0.15 if is_volatile else -0.15

    # Adjustment 2: Counter-argument quality [-0.2, +0.3]
    has_counter_args = counter_count >= 2
    trade_profitable = cumulative_return > 0.001
    trade_lost = cumulative_return < -0.001

    bonus = rl_config.REWARD_COHERENCE_COUNTER_BONUS
    penalty = rl_config.REWARD_COHERENCE_COUNTER_PENALTY

    counter_adjustment = 0.0
    if has_counter_args:
        if trade_profitable:
            counter_adjustment = bonus       # Rigorous + correct
        elif trade_lost:
            counter_adjustment = bonus * 0.5 # Rigorous but wrong
        else:
            counter_adjustment = bonus * 0.3 # Rigorous, flat market
    elif trade_lost:
        counter_adjustment = -penalty        # Overconfident failure

    return max(-1.0, min(1.0, base_score + vol_adjustment + counter_adjustment))


def _compute_regime_score(
    position_weight: float,
    prices_during_holding: np.ndarray,
    trailing_prices: np.ndarray,
) -> float:
    """
    Dimension 8: Regime awareness.
    Did the action align with the current market regime?
    Regime detection: 20-day SMA slope direction.
    """
    if len(trailing_prices) < 20 or position_weight == 0.0:
        return 0.0

    # Simple regime detection: slope of 20-day SMA
    sma_20 = np.convolve(trailing_prices[-20:], np.ones(5)/5, mode='valid')
    if len(sma_20) < 2:
        return 0.0

    sma_slope = (sma_20[-1] - sma_20[0]) / sma_20[0]

    # Classify regime
    if sma_slope > 0.01:
        regime = "uptrend"
    elif sma_slope < -0.01:
        regime = "downtrend"
    else:
        regime = "sideways"

    position_direction = 1.0 if position_weight > 0 else -1.0

    if regime == "uptrend":
        return 1.0 if position_direction > 0 else -0.5
    elif regime == "downtrend":
        return 1.0 if position_direction < 0 else -0.5
    else:
        # Sideways: slight preference for smaller positions
        return 0.5 - abs(position_weight) * 0.5


# ─────────────────────────────────────────────────────────────────────────────
# Main reward computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_reward(
    ticker: str,
    entry_date: str,
    exit_date: str,
    action: str,
    position_weight: float,
    entry_price: float,
    exit_price: float,
    prices_during_holding: np.ndarray,
    transaction_cost_bps: float = None,
    signal_label: str = None,
    reasoning: str = None,
    past_forecast_errors: List[float] = None,
    trailing_prices: np.ndarray = None,
    vol_history: np.ndarray = None,
) -> RewardRecord:
    """
    Compute multi-dimensional reward for a completed trade.

    Args:
        ticker: Stock ticker
        entry_date: Trade entry date
        exit_date: Trade exit date
        action: Trading action taken (STRONG_SELL..STRONG_BUY)
        position_weight: Position weight (-1 to +1)
        entry_price: Price at entry
        exit_price: Price at exit
        prices_during_holding: Array of daily closing prices during holding period
        transaction_cost_bps: Override transaction cost (default: from config)
        signal_label: Algorithm S1 label (for override dimension)
        reasoning: Model's reasoning text (for coherence dimension)
        past_forecast_errors: List of past forecast errors (for improvement dimension)
        trailing_prices: Array of ~60 trading days of prices before entry (for regime)
        vol_history: Array of trailing daily volatilities (for risk discipline)

    Returns:
        RewardRecord with computed composite reward and per-dimension breakdown
    """
    # Handle HOLD positions (zero exposure)
    if action == "HOLD" or position_weight == 0.0:
        # Still compute some dimensions for HOLD
        cumulative_return = (exit_price - entry_price) / entry_price if entry_price > 0 else 0.0
        dims = RewardDimensions(
            sharpe=0.0,
            direction=0.0,
            conviction=_compute_conviction_score(0.0, cumulative_return),
            improvement=0.0,
            override=_compute_override_score(action, signal_label or action, cumulative_return, 0.0) if signal_label else 0.0,
            risk_discipline=0.0,
            coherence=0.0,
            regime=0.0,
        )
        dims.composite = rl_config.REWARD_HOLD_VALUE
        return RewardRecord(
            ticker=ticker,
            entry_date=entry_date,
            exit_date=exit_date,
            action=action,
            position_weight=0.0,
            entry_price=entry_price,
            exit_price=exit_price,
            holding_days=len(prices_during_holding),
            cumulative_return=cumulative_return,
            holding_volatility=0.0,
            transaction_cost=0.0,
            reward=rl_config.REWARD_HOLD_VALUE,
            raw_pnl=0.0,
            dimensions=dims,
        )

    tx_bps = transaction_cost_bps if transaction_cost_bps is not None else config.TRANSACTION_COST_BPS
    tx_cost = tx_bps / 10_000 * abs(position_weight) * 2  # Entry + exit

    # Cumulative price return over holding period
    cumulative_return = (exit_price - entry_price) / entry_price

    # Holding period volatility (daily returns std)
    if len(prices_during_holding) >= 2:
        daily_returns = np.diff(prices_during_holding) / prices_during_holding[:-1]
        holding_vol = np.std(daily_returns) if len(daily_returns) > 1 else abs(daily_returns[0]) if len(daily_returns) == 1 else rl_config.REWARD_VOLATILITY_FLOOR
    else:
        holding_vol = rl_config.REWARD_VOLATILITY_FLOOR

    holding_vol = max(holding_vol, rl_config.REWARD_VOLATILITY_FLOOR)

    # Risk-free rate for the holding period
    holding_days = len(prices_during_holding)
    rf_daily = config.RISK_FREE_RATE_ANNUAL / config.TRADING_DAYS_PER_YEAR
    rf_cost = rf_daily * holding_days

    # Raw P&L (position-adjusted)
    raw_pnl = position_weight * cumulative_return

    # ── Compute all 8 dimensions ──────────────────────────────────────────

    # 1. Sharpe
    sharpe_score = _compute_sharpe_score(
        position_weight, cumulative_return, tx_cost, rf_cost, holding_vol
    )

    # 2. Direction
    direction_score = _compute_direction_score(position_weight, cumulative_return)

    # 3. Conviction calibration
    conviction_score = _compute_conviction_score(position_weight, cumulative_return)

    # 4. Forecast error improvement
    current_forecast_error = abs(
        abs(position_weight) - min(abs(cumulative_return) / rl_config.REWARD_RETURN_CAP, 1.0)
    )
    improvement_score = _compute_improvement_score(
        current_forecast_error,
        past_forecast_errors or [],
    )

    # 5. Signal override quality
    override_score = _compute_override_score(
        action, signal_label or action, cumulative_return, position_weight
    )

    # 6. Risk discipline
    trailing_vol = holding_vol
    risk_score = _compute_risk_discipline_score(
        position_weight,
        trailing_vol,
        vol_history if vol_history is not None else np.array([]),
    )

    # 7. Analytical coherence
    coherence_score = _compute_coherence_score(
        reasoning or "", cumulative_return, holding_vol
    )

    # 8. Regime awareness
    regime_score = _compute_regime_score(
        position_weight,
        prices_during_holding,
        trailing_prices if trailing_prices is not None else np.array([]),
    )

    # ── Weighted composite ────────────────────────────────────────────────
    weights = rl_config.REWARD_DIMENSION_WEIGHTS
    composite = (
        weights["sharpe"] * sharpe_score +
        weights["direction"] * direction_score +
        weights["conviction"] * conviction_score +
        weights["improvement"] * improvement_score +
        weights["override"] * override_score +
        weights["risk_discipline"] * risk_score +
        weights["coherence"] * coherence_score +
        weights["regime"] * regime_score
    )

    dims = RewardDimensions(
        sharpe=sharpe_score,
        direction=direction_score,
        conviction=conviction_score,
        improvement=improvement_score,
        override=override_score,
        risk_discipline=risk_score,
        coherence=coherence_score,
        regime=regime_score,
        composite=composite,
    )

    return RewardRecord(
        ticker=ticker,
        entry_date=entry_date,
        exit_date=exit_date,
        action=action,
        position_weight=position_weight,
        entry_price=entry_price,
        exit_price=exit_price,
        holding_days=holding_days,
        cumulative_return=cumulative_return,
        holding_volatility=holding_vol,
        transaction_cost=tx_cost,
        reward=float(composite),
        raw_pnl=float(raw_pnl),
        dimensions=dims,
    )


def compute_forecast_error(position_weight: float, cumulative_return: float) -> float:
    """
    Compute forecast error for the improvement dimension.
    Error = |abs(position_weight) - normalized_abs_return|

    This is called externally by the experience buffer to store errors
    for future improvement score computation.
    """
    normalized_return = min(abs(cumulative_return) / rl_config.REWARD_RETURN_CAP, 1.0)
    return abs(abs(position_weight) - normalized_return)
