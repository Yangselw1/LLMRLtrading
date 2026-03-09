"""
RL Walk-Forward Backtester — Anti-Lookahead RL Backtest Engine

Implements the walk-forward RL backtest loop for both ICRL and GRPO modes:
  1. Backfill rewards for completed holding periods (anti-lookahead)
  2. Check if GRPO retraining is triggered (enough completed trades)
  3. Rebalance: call analyst, create Experience, update position
  4. Track daily P&L

Returns standard BacktestResult for compatibility with existing visualization.
"""
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import config
from backtester import BacktestResult, compute_cumulative_return, compute_sharpe_ratio, compute_hit_rate, compute_max_drawdown
from experience_buffer import ExperienceBuffer, Experience
from reward import compute_reward

logger = logging.getLogger(__name__)


class RLBacktester:
    """
    Walk-forward RL backtest engine.

    Supports both ICRL and GRPO modes with a shared experience buffer.
    Anti-lookahead: rewards are only computed after the holding period
    completes, so the model never sees future information.
    """

    def __init__(
        self,
        mode: str,                       # "icrl" or "grpo"
        analyst,                          # ICRLAnalyst or GRPOAnalyst
        experience_buffer: ExperienceBuffer,
        observer=None,
        grpo_trainer=None,                # GRPOTrainer (only for GRPO mode)
        grpo_retrain_interval: int = 20,  # Retrain after N completed trades
        initial_capital: float = None,
        transaction_cost_bps: float = None,
        rebalance_freq: int = None,
    ):
        self.mode = mode
        self.analyst = analyst
        self.experience_buffer = experience_buffer
        self.obs = observer
        self.grpo_trainer = grpo_trainer
        self.grpo_retrain_interval = grpo_retrain_interval
        self.initial_capital = initial_capital or config.INITIAL_CAPITAL
        self.tx_cost_bps = transaction_cost_bps or config.TRANSACTION_COST_BPS
        self.tx_cost = self.tx_cost_bps / 10_000
        self.rebalance_freq = rebalance_freq or config.REBALANCE_FREQUENCY_DAYS

        # Track pending experiences for reward backfill
        self._pending_indices: List[Tuple[int, int, str]] = []  # (buffer_idx, entry_day_idx, ticker)
        self._last_retrain_count = 0
        self._step_counter = 0

    def run(
        self,
        ticker: str,
        price_data: pd.DataFrame,
        signal_labels: pd.Series,
        data_collector,
        start_date: str,
        end_date: str,
    ) -> Tuple[BacktestResult, ExperienceBuffer]:
        """
        Run walk-forward RL backtest for a single ticker.

        Args:
            ticker: Stock ticker
            price_data: DataFrame with 'close' column
            signal_labels: Series of Algorithm S1 labels
            data_collector: DataCollector for snapshot/news retrieval
            start_date: Backtest start date
            end_date: Backtest end date

        Returns:
            (BacktestResult, updated ExperienceBuffer)
        """
        logger.info(f"RL Backtest ({self.mode.upper()}): {ticker} {start_date} → {end_date}")

        # Filter to backtest window
        start_dt = pd.Timestamp(start_date)
        end_dt = pd.Timestamp(end_date)
        mask = (price_data.index >= start_dt) & (price_data.index <= end_dt)
        bt_prices = price_data.loc[mask, "close"].copy()

        if bt_prices.empty:
            logger.warning(f"No price data in backtest window for {ticker}")
            return BacktestResult(ticker=ticker), self.experience_buffer

        trading_days = bt_prices.index.tolist()
        close_prices = bt_prices.values

        # Initialize tracking
        portfolio_value = self.initial_capital
        position_weight = 0.0
        last_rebalance_idx = -self.rebalance_freq

        equity_values = []
        daily_rets = []
        signal_series = {}
        trades = []
        prev_value = portfolio_value

        for i, date in enumerate(trading_days):
            current_price = bt_prices[date]
            date_str = date.strftime("%Y-%m-%d")

            # ── Step 1: Backfill rewards for completed holding periods ────
            self._backfill_rewards(i, trading_days, close_prices, ticker)

            # ── Step 2: GRPO retrain check ────────────────────────────────
            if self.mode == "grpo" and self.grpo_trainer:
                completed_count = len(self.experience_buffer.get_completed())
                if (completed_count - self._last_retrain_count >= self.grpo_retrain_interval
                        and completed_count >= self.grpo_retrain_interval):
                    if self.obs:
                        self.obs.log_grpo_retrain_trigger(
                            completed_count, self._last_retrain_count
                        )
                    self.grpo_trainer.train(self.experience_buffer, epochs=1)
                    self._last_retrain_count = completed_count

            # ── Step 3: Rebalance decision ────────────────────────────────
            if i - last_rebalance_idx >= self.rebalance_freq or i == 0:
                # Get Algorithm S1 label
                available_labels = signal_labels[signal_labels.index <= date].dropna()
                signal_label = available_labels.iloc[-1] if not available_labels.empty else "HOLD"

                # Get snapshot and news
                snapshot = data_collector.get_snapshot(date_str)
                news = data_collector.get_news_for_date(date_str)

                if snapshot:
                    # Call the analyst (ICRL or GRPO)
                    reasoning, decision = self.analyst.analyze(
                        snapshot, news, signal_label
                    )
                else:
                    reasoning, decision = "", signal_label

                new_weight = config.POSITION_WEIGHTS.get(decision, 0.0)

                # Transaction cost
                weight_change = abs(new_weight - position_weight)
                tx_cost_dollars = 0.0
                if weight_change > 0:
                    tx_cost_dollars = portfolio_value * weight_change * self.tx_cost
                    portfolio_value -= tx_cost_dollars

                old_weight = position_weight
                position_weight = new_weight
                last_rebalance_idx = i
                signal_series[date] = decision

                # Create experience (reward will be backfilled later)
                exp = Experience(
                    step_idx=self._step_counter,
                    date=date_str,
                    ticker=ticker,
                    state_snapshot=snapshot or {},
                    state_news=news,
                    signal_label=signal_label,
                    action=decision,
                    reasoning=reasoning,
                    mode=self.mode,
                    entry_price=float(current_price),
                    position_weight=new_weight,
                )
                buf_idx = self.experience_buffer.add(exp)
                self._pending_indices.append((buf_idx, i, ticker))
                self._step_counter += 1

                if self.obs:
                    self.obs.log_experience_added(
                        ticker, date_str, decision, signal_label, buf_idx
                    )

                # Log rebalance
                if self.obs:
                    old_signal = trades[-1].signal if trades else "NONE"
                    self.obs.log_rebalance(
                        date_str, ticker, old_signal, decision,
                        old_weight, new_weight, float(current_price),
                        portfolio_value, tx_cost_dollars
                    )

                from backtester import TradeRecord
                trades.append(TradeRecord(
                    date=date_str, ticker=ticker, signal=decision,
                    position_weight=new_weight, entry_price=float(current_price),
                    portfolio_value=portfolio_value,
                ))
            else:
                signal_series[date] = signal_series.get(trading_days[i - 1], "HOLD")

            # ── Step 4: Daily P&L ─────────────────────────────────────────
            if i > 0:
                prev_price = bt_prices[trading_days[i - 1]]
                price_return = (current_price - prev_price) / prev_price
                portfolio_return = position_weight * price_return
                portfolio_value = prev_value * (1 + portfolio_return)
                daily_rets.append(portfolio_return)
            else:
                daily_rets.append(0.0)

            equity_values.append(portfolio_value)
            prev_value = portfolio_value

        # Final backfill for any remaining pending experiences
        self._backfill_rewards(len(trading_days), trading_days, close_prices, ticker)

        # Log buffer stats
        if self.obs:
            self.obs.log_experience_buffer_stats(self.experience_buffer.summary_stats())

        # Build result
        equity_curve = pd.Series(equity_values, index=trading_days, name="equity")
        daily_returns = pd.Series(daily_rets, index=trading_days, name="daily_return")
        signals_out = pd.Series(signal_series, name="signal")

        fwd_returns = bt_prices.pct_change().shift(-1)

        cr = compute_cumulative_return(equity_curve)
        sr = compute_sharpe_ratio(daily_returns)
        hr = compute_hit_rate(signals_out, fwd_returns)
        mdd = compute_max_drawdown(equity_curve)

        # Win/loss counting
        active_trades = [t for t in trades if t.signal != "HOLD"]
        winning = losing = 0
        for j in range(len(active_trades) - 1):
            entry = active_trades[j]
            exit_t = active_trades[j + 1]
            pnl = (exit_t.entry_price - entry.entry_price) / entry.entry_price
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

        if self.obs:
            self.obs.log_backtest_metrics(ticker, cr, sr, hr, mdd, len(active_trades))

        return result, self.experience_buffer

    def _backfill_rewards(self, current_day_idx: int, trading_days: list,
                          close_prices: np.ndarray, ticker: str):
        """
        Backfill rewards for experiences whose holding period has completed.

        Anti-lookahead: we only compute the reward after rebalance_freq days
        have passed since the entry.

        Passes extra context for multi-dimensional reward computation:
        - signal_label and reasoning from the Experience
        - past_forecast_errors from the experience buffer
        - trailing_prices for regime detection
        - vol_history for risk discipline
        """
        still_pending = []

        for buf_idx, entry_day_idx, exp_ticker in self._pending_indices:
            if exp_ticker != ticker:
                still_pending.append((buf_idx, entry_day_idx, exp_ticker))
                continue

            exit_day_idx = entry_day_idx + self.rebalance_freq

            if exit_day_idx < current_day_idx and exit_day_idx < len(trading_days):
                # Holding period completed — compute reward
                exp = self.experience_buffer.experiences[buf_idx]
                entry_price = close_prices[entry_day_idx]
                exit_price = close_prices[min(exit_day_idx, len(close_prices) - 1)]
                holding_prices = close_prices[entry_day_idx:exit_day_idx + 1]

                entry_date = trading_days[entry_day_idx].strftime("%Y-%m-%d")
                exit_date = trading_days[min(exit_day_idx, len(trading_days) - 1)].strftime("%Y-%m-%d")

                # Trailing prices for regime detection (up to 60 days before entry)
                trail_start = max(0, entry_day_idx - 60)
                trailing_prices = close_prices[trail_start:entry_day_idx + 1]

                # Volatility history for risk discipline
                vol_history = np.array([])
                if entry_day_idx >= 20:
                    daily_rets = np.diff(close_prices[:entry_day_idx]) / close_prices[:entry_day_idx - 1]
                    if len(daily_rets) >= 20:
                        # Rolling 5-day volatility
                        vols = []
                        for vi in range(4, len(daily_rets)):
                            window = daily_rets[vi - 4:vi + 1]
                            vols.append(np.std(window))
                        vol_history = np.array(vols)

                # Past forecast errors for improvement dimension
                import rl_config
                past_errors = self.experience_buffer.get_forecast_errors(
                    rl_config.REWARD_IMPROVEMENT_WINDOW
                )

                reward_record = compute_reward(
                    ticker=ticker,
                    entry_date=entry_date,
                    exit_date=exit_date,
                    action=exp.action,
                    position_weight=exp.position_weight,
                    entry_price=float(entry_price),
                    exit_price=float(exit_price),
                    prices_during_holding=holding_prices,
                    transaction_cost_bps=self.tx_cost_bps,
                    signal_label=exp.signal_label,
                    reasoning=exp.reasoning,
                    past_forecast_errors=past_errors,
                    trailing_prices=trailing_prices,
                    vol_history=vol_history,
                )

                self.experience_buffer.backfill_reward(buf_idx, reward_record)

                if self.obs:
                    self.obs.log_reward_computed(
                        ticker, entry_date, exit_date,
                        exp.action, reward_record.reward, reward_record.raw_pnl,
                    )
            else:
                still_pending.append((buf_idx, entry_day_idx, exp_ticker))

        self._pending_indices = still_pending
