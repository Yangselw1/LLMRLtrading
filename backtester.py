"""
Backtesting Engine for Trading-R1

Simulates portfolio performance based on trading signals with:
- Position sizing mapped from 5-class actions (Section 3.7)
- Weekly rebalancing (~5 trading day holding period)
- Transaction cost modeling
- All four evaluation metrics from Appendix S2:
  CR (Cumulative Return), SR (Sharpe Ratio), HR (Hit Rate), MDD (Max Drawdown)
"""
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)


@dataclass
class TradeRecord:
    """Record of a single trade/rebalance."""
    date: str
    ticker: str
    signal: str
    position_weight: float
    entry_price: float
    portfolio_value: float


@dataclass
class BacktestResult:
    """Complete results for a single-ticker backtest."""
    ticker: str
    # Metrics
    cumulative_return: float = 0.0
    sharpe_ratio: float = 0.0
    hit_rate: float = 0.0
    max_drawdown: float = 0.0
    # Series
    equity_curve: pd.Series = field(default_factory=pd.Series)
    daily_returns: pd.Series = field(default_factory=pd.Series)
    drawdown_series: pd.Series = field(default_factory=pd.Series)
    signals: pd.Series = field(default_factory=pd.Series)
    trades: List[TradeRecord] = field(default_factory=list)
    # Counts
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluation Metrics (Appendix S2.2)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_cumulative_return(equity_curve: pd.Series) -> float:
    """CR = V_N / V_0 - 1 = Π(1 + r_t) - 1"""
    if equity_curve.empty or equity_curve.iloc[0] == 0:
        return 0.0
    return (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1.0


def compute_sharpe_ratio(daily_returns: pd.Series) -> float:
    """
    Annualized Sharpe Ratio using 4% risk-free rate (10Y Treasury).
    SR_ann = sqrt(K) * SR_per, where K=252 for daily.
    Excess returns: x_t = r_t - r_f (daily)
    """
    if daily_returns.empty or daily_returns.std() == 0:
        return 0.0

    daily_rf = config.RISK_FREE_RATE_ANNUAL / config.TRADING_DAYS_PER_YEAR
    excess = daily_returns - daily_rf
    sr_per = excess.mean() / excess.std()
    sr_ann = np.sqrt(config.TRADING_DAYS_PER_YEAR) * sr_per
    return float(sr_ann)


def compute_hit_rate(signals: pd.Series, returns: pd.Series) -> float:
    """
    HR = (1/N) Σ 1{sign(a_t) = sign(r_t)}
    Fraction of trades where predicted direction matched actual direction.
    Hold signals are excluded from this calculation.
    """
    # Map signals to directional sign: buy/strong_buy → +1, sell/strong_sell → -1
    signal_sign = signals.map({
        "STRONG_BUY": 1, "BUY": 1, "HOLD": 0, "SELL": -1, "STRONG_SELL": -1
    })

    # Only evaluate non-hold positions
    active = signal_sign[signal_sign != 0].dropna()
    aligned_returns = returns.reindex(active.index).dropna()
    active = active.reindex(aligned_returns.index)

    if active.empty:
        return 0.0

    return_sign = np.sign(aligned_returns)
    correct = (active == return_sign).sum()
    return float(correct / len(active))


def compute_max_drawdown(equity_curve: pd.Series) -> float:
    """
    MDD = max_t (1 - V_t / max_{u≤t} V_u)
    """
    if equity_curve.empty:
        return 0.0

    running_max = equity_curve.cummax()
    drawdown = 1.0 - (equity_curve / running_max)
    return float(drawdown.max())


# ═══════════════════════════════════════════════════════════════════════════════
# Core Backtesting Engine
# ═══════════════════════════════════════════════════════════════════════════════

class Backtester:
    """
    Single-asset backtesting engine implementing Trading-R1 methodology.

    Position sizing from 5-class action space (Section 3.7):
        STRONG_BUY  →  +1.0  (fully long)
        BUY         →  +0.5  (half long)
        HOLD        →   0.0  (flat/cash)
        SELL        →  -0.5  (half short)
        STRONG_SELL →  -1.0  (fully short)

    Rebalancing: every `rebalance_freq` trading days (~weekly).
    """

    def __init__(
        self,
        initial_capital: float = None,
        transaction_cost_bps: float = None,
        rebalance_freq: int = None,
        observer=None,
    ):
        self.initial_capital = initial_capital or config.INITIAL_CAPITAL
        self.tx_cost = (transaction_cost_bps or config.TRANSACTION_COST_BPS) / 10_000
        self.rebalance_freq = rebalance_freq or config.REBALANCE_FREQUENCY_DAYS
        self.obs = observer

    def run(
        self,
        ticker: str,
        price_data: pd.DataFrame,
        signals: pd.Series,
        start_date: str,
        end_date: str,
    ) -> BacktestResult:
        """
        Run a backtest for a single ticker.

        Args:
            ticker: Stock ticker symbol
            price_data: DataFrame with 'close' column
            signals: Series of trading labels (STRONG_SELL..STRONG_BUY)
            start_date: Backtest start date (inclusive)
            end_date: Backtest end date (inclusive)

        Returns:
            BacktestResult with all metrics and series
        """
        logger.info(f"═══ Backtesting {ticker}: {start_date} to {end_date} ═══")

        # Filter to backtest window
        start_dt = pd.Timestamp(start_date)
        end_dt = pd.Timestamp(end_date)
        mask = (price_data.index >= start_dt) & (price_data.index <= end_dt)
        bt_prices = price_data.loc[mask, "close"].copy()

        if bt_prices.empty:
            logger.warning(f"No price data in backtest window for {ticker}")
            return BacktestResult(ticker=ticker)

        trading_days = bt_prices.index.tolist()
        logger.info(f"  Trading days: {len(trading_days)}")

        # Initialize tracking
        portfolio_value = self.initial_capital
        position_weight = 0.0  # Current position weight
        last_rebalance_idx = -self.rebalance_freq  # Force rebalance on day 0

        equity_values = []
        daily_rets = []
        signal_series = {}
        trades = []

        prev_value = portfolio_value

        for i, date in enumerate(trading_days):
            current_price = bt_prices[date]
            date_str = date.strftime("%Y-%m-%d")

            # Check if rebalance is due
            if i - last_rebalance_idx >= self.rebalance_freq or i == 0:
                # Get signal for this date
                available_signals = signals[signals.index <= date].dropna()
                if not available_signals.empty:
                    new_signal = available_signals.iloc[-1]
                else:
                    new_signal = "HOLD"

                new_weight = config.POSITION_WEIGHTS.get(new_signal, 0.0)

                # Apply transaction cost on position change
                weight_change = abs(new_weight - position_weight)
                tx_cost_dollars = 0.0
                if weight_change > 0:
                    tx_cost_dollars = portfolio_value * weight_change * self.tx_cost
                    portfolio_value -= tx_cost_dollars

                old_signal = signal_series.get(trading_days[i-1], "HOLD") if i > 0 else "NONE"
                old_weight = position_weight

                position_weight = new_weight
                last_rebalance_idx = i
                signal_series[date] = new_signal

                trades.append(TradeRecord(
                    date=date_str,
                    ticker=ticker,
                    signal=new_signal,
                    position_weight=new_weight,
                    entry_price=current_price,
                    portfolio_value=portfolio_value,
                ))

                # Observer: log rebalance
                if self.obs:
                    self.obs.log_rebalance(
                        date_str, ticker, old_signal, new_signal,
                        old_weight, new_weight, current_price,
                        portfolio_value, tx_cost_dollars
                    )
            else:
                signal_series[date] = signal_series.get(trading_days[i-1], "HOLD")

            # Compute daily P&L
            if i > 0:
                prev_price = bt_prices[trading_days[i - 1]]
                price_return = (current_price - prev_price) / prev_price
                portfolio_return = position_weight * price_return
                portfolio_value = prev_value * (1 + portfolio_return)
                daily_rets.append(portfolio_return)
            else:
                daily_rets.append(0.0)

            equity_values.append(portfolio_value)

            # Observer: log daily P&L (DEBUG level)
            if self.obs:
                self.obs.log_daily_pnl(
                    date_str, ticker, current_price,
                    daily_rets[-1], portfolio_value, position_weight
                )

            prev_value = portfolio_value

        # Build result series
        equity_curve = pd.Series(equity_values, index=trading_days, name="equity")
        daily_returns = pd.Series(daily_rets, index=trading_days, name="daily_return")
        signals_out = pd.Series(signal_series, name="signal")

        # Compute forward returns for hit rate (next-day return)
        fwd_returns = bt_prices.pct_change().shift(-1)

        # Compute all metrics
        cr = compute_cumulative_return(equity_curve)
        sr = compute_sharpe_ratio(daily_returns)
        hr = compute_hit_rate(signals_out, fwd_returns)
        mdd = compute_max_drawdown(equity_curve)

        # Win/loss counting
        active_trades = [t for t in trades if t.signal != "HOLD"]
        winning = 0
        losing = 0
        for j in range(len(active_trades) - 1):
            entry = active_trades[j]
            exit_trade = active_trades[j + 1]
            pnl = (exit_trade.entry_price - entry.entry_price) / entry.entry_price
            if entry.position_weight < 0:
                pnl = -pnl
            if pnl > 0:
                winning += 1
            else:
                losing += 1

        result = BacktestResult(
            ticker=ticker,
            cumulative_return=cr,
            sharpe_ratio=sr,
            hit_rate=hr,
            max_drawdown=mdd,
            equity_curve=equity_curve,
            daily_returns=daily_returns,
            drawdown_series=1.0 - equity_curve / equity_curve.cummax(),
            signals=signals_out,
            trades=trades,
            total_trades=len(active_trades),
            winning_trades=winning,
            losing_trades=losing,
        )

        logger.info(f"  Results for {ticker}:")
        logger.info(f"    CR:     {cr*100:+.2f}%")
        logger.info(f"    SR:     {sr:.2f}")
        logger.info(f"    HR:     {hr*100:.1f}%")
        logger.info(f"    MDD:    {mdd*100:.2f}%")
        logger.info(f"    Trades: {len(active_trades)}")

        # Observer: log metrics
        if self.obs:
            self.obs.log_backtest_metrics(ticker, cr, sr, hr, mdd, len(active_trades))

        return result


# ═══════════════════════════════════════════════════════════════════════════════
# Multi-Asset Portfolio Backtester
# ═══════════════════════════════════════════════════════════════════════════════

class PortfolioBacktester:
    """
    Runs backtests across multiple tickers and aggregates results.
    Equal-weight allocation across tickers.
    """

    def __init__(self, observer=None, **kwargs):
        self.backtester = Backtester(observer=observer, **kwargs)
        self.results: Dict[str, BacktestResult] = {}

    def run_all(
        self,
        ticker_data: Dict[str, Dict],
        start_date: str,
        end_date: str,
    ) -> Dict[str, BacktestResult]:
        """
        Run backtests for all tickers.

        Args:
            ticker_data: {ticker: {"price_data": df, "signals": series}}
            start_date, end_date: Backtest period
        """
        for ticker, data in ticker_data.items():
            try:
                result = self.backtester.run(
                    ticker=ticker,
                    price_data=data["price_data"],
                    signals=data["signals"],
                    start_date=start_date,
                    end_date=end_date,
                )
                self.results[ticker] = result
            except Exception as e:
                logger.error(f"Backtest failed for {ticker}: {e}")

        return self.results

    def get_summary_table(self) -> pd.DataFrame:
        """Build a summary table of all backtest results."""
        rows = []
        for ticker, r in self.results.items():
            rows.append({
                "Ticker": ticker,
                "CR(%)": round(r.cumulative_return * 100, 2),
                "SR": round(r.sharpe_ratio, 2),
                "HR(%)": round(r.hit_rate * 100, 1),
                "MDD(%)": round(r.max_drawdown * 100, 2),
                "Trades": r.total_trades,
                "Win": r.winning_trades,
                "Loss": r.losing_trades,
            })
        return pd.DataFrame(rows).set_index("Ticker")

    def get_portfolio_equity(self) -> pd.Series:
        """
        Equal-weight portfolio equity curve across all tickers.
        """
        curves = {}
        for ticker, r in self.results.items():
            if not r.equity_curve.empty:
                # Normalize each to start at 1.0
                curves[ticker] = r.equity_curve / r.equity_curve.iloc[0]

        if not curves:
            return pd.Series(dtype=float)

        df = pd.DataFrame(curves)
        return df.mean(axis=1) * config.INITIAL_CAPITAL
