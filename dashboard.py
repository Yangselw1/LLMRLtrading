"""
Trading-R1 Backtest Dashboard — Streamlit Web UI

Interactive dashboard for running backtests, viewing results,
and exploring multi-dimensional reward breakdowns.

Usage:
    streamlit run dashboard.py
    # or via .claude/launch.json:
    python -m streamlit run dashboard.py --server.port 8501
"""
import os
import sys
import time
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import streamlit as st

# Project imports
import config
from data_collector import DataCollector
from signal_generator import SignalGenerator
from backtester import PortfolioBacktester, BacktestResult

# Suppress noisy loggers
logging.basicConfig(level=logging.WARNING)

# ═══════════════════════════════════════════════════════════════════════════════
# Page Config
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Trading-R1 Backtest",
    page_icon="📈",
    layout="wide",
)

# Chart colors (consistent with visualizer.py)
COLORS = {
    "STRONG_BUY": "#2ecc71",
    "BUY": "#82e0aa",
    "HOLD": "#f0b27a",
    "SELL": "#e74c3c",
    "STRONG_SELL": "#922b21",
    "equity": "#2c3e50",
    "benchmark": "#95a5a6",
    "drawdown": "#e74c3c",
}


# ═══════════════════════════════════════════════════════════════════════════════
# Chart Functions (return Figure objects for st.pyplot)
# ═══════════════════════════════════════════════════════════════════════════════

def make_equity_chart(result: BacktestResult, benchmark_prices=None):
    """Equity curve chart."""
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(result.equity_curve.index, result.equity_curve.values,
            color=COLORS["equity"], linewidth=1.5, label="Strategy")

    if benchmark_prices is not None and not benchmark_prices.empty:
        mask = (benchmark_prices.index >= result.equity_curve.index[0]) & \
               (benchmark_prices.index <= result.equity_curve.index[-1])
        bm = benchmark_prices[mask]
        if not bm.empty:
            bm_equity = (bm / bm.iloc[0]) * result.equity_curve.iloc[0]
            ax.plot(bm_equity.index, bm_equity.values,
                    color=COLORS["benchmark"], linewidth=1.2,
                    linestyle="--", label="Buy & Hold")

    ax.set_title(f"{result.ticker} — Equity Curve", fontsize=13, fontweight="bold")
    ax.set_ylabel("Portfolio Value ($)")
    ax.yaxis.set_major_formatter(mtick.StrMethodFormatter("${x:,.0f}"))
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def make_drawdown_chart(result: BacktestResult):
    """Drawdown chart."""
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.fill_between(result.drawdown_series.index,
                     0, -result.drawdown_series.values * 100,
                     color=COLORS["drawdown"], alpha=0.4)
    ax.plot(result.drawdown_series.index,
            -result.drawdown_series.values * 100,
            color=COLORS["drawdown"], linewidth=0.8)
    ax.set_title(f"{result.ticker} — Drawdown", fontsize=13, fontweight="bold")
    ax.set_ylabel("Drawdown (%)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def make_signal_distribution(result: BacktestResult):
    """Signal distribution pie chart."""
    if result.signals.empty:
        return None

    counts = result.signals.value_counts()
    labels, sizes, colors = [], [], []
    for label in config.ACTION_LABELS:
        if label in counts:
            labels.append(label.replace("_", " "))
            sizes.append(counts[label])
            colors.append(COLORS.get(label, "#cccccc"))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(sizes, labels=labels, colors=colors,
           autopct="%1.1f%%", startangle=90, textprops={"fontsize": 9})
    ax.set_title(f"{result.ticker} — Signal Distribution", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


def make_reward_evolution(experience_buffer):
    """Reward evolution bar chart + cumulative."""
    completed = experience_buffer.get_completed()
    if not completed:
        return None

    rewards = [e.reward for e in completed if e.reward is not None]
    if not rewards:
        return None

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    x = range(len(rewards))
    bar_colors = ["green" if r > 0 else "red" for r in rewards]

    ax1.bar(x, rewards, color=bar_colors, alpha=0.6, width=0.8)
    ax1.axhline(y=0, color="black", linewidth=0.5)
    ax1.set_ylabel("Reward")
    ax1.set_title("Trade Rewards", fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.3)

    window = min(10, len(rewards))
    if window > 1:
        rolling = pd.Series(rewards).rolling(window).mean()
        ax1.plot(x, rolling.values, color="blue", linewidth=2, label=f"Rolling avg ({window})")
        ax1.legend()

    cumulative = np.cumsum(rewards)
    ax2.plot(x, cumulative, color="purple", linewidth=2)
    ax2.fill_between(x, cumulative, alpha=0.2, color="purple")
    ax2.set_xlabel("Trade Number")
    ax2.set_ylabel("Cumulative Reward")
    ax2.axhline(y=0, color="black", linewidth=0.5)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def make_dimension_radar(experience_buffer):
    """Radar chart of average per-dimension scores."""
    completed = experience_buffer.get_completed()
    if not completed:
        return None

    dim_names = ["sharpe", "direction", "conviction", "improvement",
                 "override", "risk_discipline", "coherence", "regime"]
    dim_labels = ["Sharpe", "Direction", "Conviction", "Improvement",
                  "Override", "Risk Disc.", "Coherence", "Regime"]

    dim_scores = {name: [] for name in dim_names}
    for exp in completed:
        if exp.dimensions:
            d = exp.dimensions.to_dict()
            for name in dim_names:
                if name in d:
                    dim_scores[name].append(d[name])

    if not any(dim_scores.values()):
        return None

    means = [np.mean(dim_scores[n]) if dim_scores[n] else 0.0 for n in dim_names]

    N = len(dim_names)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    means_plot = means + means[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.plot(angles, means_plot, 'o-', linewidth=2, color='#2ca02c')
    ax.fill(angles, means_plot, alpha=0.2, color='#2ca02c')
    ax.plot(angles, [0.0] * (N + 1), '--', linewidth=0.5, color='gray', alpha=0.5)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dim_labels, fontsize=9)

    max_abs = max(abs(m) for m in means) if means else 1.0
    limit = max(max_abs * 1.3, 0.5)
    ax.set_ylim(-limit, limit)

    for angle, value in zip(angles[:-1], means):
        ax.annotate(f"{value:+.2f}", xy=(angle, value), xytext=(0, 10),
                    textcoords='offset points', ha='center', va='bottom',
                    fontsize=8, fontweight='bold',
                    color='green' if value >= 0 else 'red')

    ax.set_title("Reward Dimension Profile", fontsize=14, fontweight="bold", pad=20)
    fig.tight_layout()
    return fig


def make_dimension_evolution(experience_buffer):
    """Per-dimension score evolution over time."""
    completed = experience_buffer.get_completed()
    if not completed:
        return None

    dim_names = ["sharpe", "direction", "conviction", "improvement",
                 "override", "risk_discipline", "coherence", "regime"]
    dim_labels = ["Sharpe", "Direction", "Conviction", "Improvement",
                  "Override", "Risk Disc.", "Coherence", "Regime"]
    dim_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
                  "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]

    dim_series = {name: [] for name in dim_names}
    trade_nums = []
    idx = 0
    for exp in completed:
        if exp.dimensions:
            d = exp.dimensions.to_dict()
            for name in dim_names:
                dim_series[name].append(d.get(name, 0.0))
            trade_nums.append(idx)
            idx += 1

    if not trade_nums:
        return None

    fig, axes = plt.subplots(2, 4, figsize=(16, 8), sharex=True)
    axes = axes.flatten()

    for i, name in enumerate(dim_names):
        ax = axes[i]
        scores = dim_series[name]
        x = trade_nums[:len(scores)]
        color = dim_colors[i]

        ax.scatter(x, scores, color=color, alpha=0.3, s=15)
        window = min(10, max(2, len(scores) // 5))
        if len(scores) >= window:
            rolling = pd.Series(scores).rolling(window).mean()
            ax.plot(x, rolling.values, color=color, linewidth=2)

        ax.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")
        ax.set_ylim(-1.3, 1.3)
        ax.set_title(dim_labels[i], fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.2)

    fig.suptitle("Dimension Evolution Over Time", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline Functions
# ═══════════════════════════════════════════════════════════════════════════════

def run_backtest_pipeline(tickers, mode, start_date, end_date, initial_capital,
                          tx_cost_bps, rebalance_days, icrl_params=None, grpo_params=None):
    """Run the full backtest pipeline and return results."""
    status = st.empty()
    progress = st.progress(0)
    status.text("Collecting data...")

    # Step 1: Data Collection
    collectors = {}
    for i, ticker in enumerate(tickers):
        progress.progress((i + 1) / len(tickers) * 0.3)
        status.text(f"Collecting data for {ticker}...")
        dc = DataCollector(ticker, start_date, end_date)
        data = dc.collect_all()
        if not data["price_data"].empty:
            collectors[ticker] = dc

    if not collectors:
        st.error("No valid data for any ticker.")
        return None, None, None

    # Step 2: Signal Generation
    progress.progress(0.4)
    status.text("Generating signals (Algorithm S1)...")
    signal_generators = {}
    for ticker, dc in collectors.items():
        sg = SignalGenerator(dc.price_data, ticker=ticker)
        sg.generate()
        signal_generators[ticker] = sg

    # Step 3: Run Backtest
    experience_buffer = None

    if mode == "Signal-Only" or mode == "LLM (Claude)":
        progress.progress(0.5)
        status.text(f"Running {mode} backtest...")
        use_llm = mode == "LLM (Claude)"

        if use_llm:
            from llm_analyst import LLMAnalyst
            analyst = LLMAnalyst(use_llm=True)
            llm_decisions = {}

            for i, (ticker, dc) in enumerate(collectors.items()):
                sg = signal_generators[ticker]
                bt_dates = dc.price_data.loc[start_date:end_date].index
                rebalance_dates = bt_dates[::rebalance_days]
                llm_decisions[ticker] = {}

                for j, date in enumerate(rebalance_dates):
                    pct = 0.5 + 0.3 * ((i * len(rebalance_dates) + j) /
                                        (len(collectors) * len(rebalance_dates)))
                    progress.progress(min(pct, 1.0))
                    status.text(f"LLM analyzing {ticker} ({date.strftime('%Y-%m-%d')})...")
                    date_str = date.strftime("%Y-%m-%d")
                    snapshot = dc.get_snapshot(date_str)
                    if not snapshot:
                        continue
                    news = dc.get_news_for_date(date_str)
                    signal_label = sg.get_label(date_str)
                    thesis, decision = analyst.analyze(snapshot, news, signal_label)
                    llm_decisions[ticker][date] = decision
                    time.sleep(0.5)

            # Build signals with LLM overrides
            ticker_data = {}
            for ticker in collectors:
                sg = signal_generators[ticker]
                if ticker in llm_decisions and llm_decisions[ticker]:
                    signals = sg.labels.copy()
                    for date, decision in llm_decisions[ticker].items():
                        if date in signals.index:
                            signals[date] = decision
                else:
                    signals = sg.labels
                ticker_data[ticker] = {
                    "price_data": collectors[ticker].price_data,
                    "signals": signals,
                }
        else:
            ticker_data = {}
            for ticker in collectors:
                sg = signal_generators[ticker]
                ticker_data[ticker] = {
                    "price_data": collectors[ticker].price_data,
                    "signals": sg.labels,
                }

        bt = PortfolioBacktester(
            initial_capital=initial_capital,
            transaction_cost_bps=tx_cost_bps,
            rebalance_freq=rebalance_days,
        )
        results = bt.run_all(ticker_data, start_date, end_date)
        summary_df = bt.get_summary_table()

        progress.progress(1.0)
        status.text("Done!")
        return results, summary_df, None

    elif mode in ("ICRL", "GRPO"):
        progress.progress(0.5)
        status.text(f"Running {mode} walk-forward backtest...")

        from experience_buffer import ExperienceBuffer
        from rl_backtester import RLBacktester

        experience_buffer = ExperienceBuffer()

        if mode == "ICRL":
            from icrl_analyst import ICRLAnalyst
            params = icrl_params or {}
            analyst = ICRLAnalyst(
                experience_buffer=experience_buffer,
                top_k=params.get("top_k", 5),
                bottom_k=params.get("bottom_k", 5),
                recent_n=params.get("recent_n", 10),
            )
        else:  # GRPO
            try:
                from grpo_trainer import GRPOTrainer, GRPOConfig
                from grpo_analyst import GRPOAnalyst

                params = grpo_params or {}
                grpo_config = GRPOConfig(model_name=params.get("model", "Qwen/Qwen3-4B"))
                trainer = GRPOTrainer(grpo_config)
                trainer.load_model()

                if params.get("sft_first", False):
                    from experience_buffer import Experience
                    sft_exps = []
                    for ticker, sg in signal_generators.items():
                        dc = collectors[ticker]
                        bt_dates = dc.price_data.loc[start_date:end_date].index
                        for date in bt_dates[::rebalance_days][:20]:
                            date_str = date.strftime("%Y-%m-%d")
                            snapshot = dc.get_snapshot(date_str)
                            if snapshot:
                                label = sg.get_label(date_str) or "HOLD"
                                sft_exps.append(Experience(
                                    step_idx=0, date=date_str, ticker=ticker,
                                    state_snapshot=snapshot, state_news=None,
                                    signal_label=label, action=label,
                                    reasoning="", mode="sft",
                                ))
                    if sft_exps:
                        trainer.supervised_pretrain(sft_exps)

                analyst = GRPOAnalyst(trainer)
            except ImportError as e:
                st.error(f"GRPO requires additional dependencies: {e}")
                return None, None, None

        rl_bt = RLBacktester(
            mode=mode.lower(),
            analyst=analyst,
            experience_buffer=experience_buffer,
            grpo_trainer=trainer if mode == "GRPO" else None,
            grpo_retrain_interval=grpo_params.get("retrain_interval", 20) if grpo_params else 20,
            initial_capital=initial_capital,
            transaction_cost_bps=tx_cost_bps,
            rebalance_freq=rebalance_days,
        )

        results = {}
        for i, (ticker, dc) in enumerate(collectors.items()):
            progress.progress(0.5 + 0.4 * ((i + 1) / len(collectors)))
            status.text(f"RL backtest: {ticker}...")
            sg = signal_generators[ticker]
            result, experience_buffer = rl_bt.run(
                ticker=ticker,
                price_data=dc.price_data,
                signal_labels=sg.labels,
                data_collector=dc,
                start_date=start_date,
                end_date=end_date,
            )
            results[ticker] = result

        # Build summary table
        rows = []
        for ticker, r in results.items():
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
        summary_df = pd.DataFrame(rows).set_index("Ticker")

        progress.progress(1.0)
        status.text("Done!")
        return results, summary_df, experience_buffer

    return None, None, None


# ═══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ═══════════════════════════════════════════════════════════════════════════════

st.sidebar.title("Trading-R1 Backtest")
st.sidebar.markdown("---")

# Ticker selection
tickers = st.sidebar.multiselect(
    "Tickers",
    options=config.PAPER_TICKERS,
    default=["AAPL"],
)

# Mode
mode = st.sidebar.radio(
    "Mode",
    options=["Signal-Only", "LLM (Claude)", "ICRL", "GRPO"],
    index=0,
)

st.sidebar.markdown("---")

# Date range
col1, col2 = st.sidebar.columns(2)
start_date = col1.date_input("Start Date", value=datetime(2024, 6, 1))
end_date = col2.date_input("End Date", value=datetime(2024, 8, 31))

# Portfolio params
initial_capital = st.sidebar.number_input(
    "Initial Capital ($)", value=100_000, step=10_000, min_value=1_000
)
tx_cost = st.sidebar.slider("Transaction Cost (bps)", 0, 50, 10)
rebalance_days = st.sidebar.slider("Rebalance Frequency (days)", 1, 20, 5)

# ICRL params
icrl_params = None
if mode == "ICRL":
    st.sidebar.markdown("**ICRL Parameters**")
    icrl_params = {
        "top_k": st.sidebar.slider("Top-K Best Trades", 1, 20, 5),
        "bottom_k": st.sidebar.slider("Bottom-K Worst Trades", 1, 20, 5),
        "recent_n": st.sidebar.slider("Recent-N Trades", 1, 30, 10),
    }

# GRPO params
grpo_params = None
if mode == "GRPO":
    st.sidebar.markdown("**GRPO Parameters**")
    grpo_params = {
        "model": st.sidebar.text_input("Model", "Qwen/Qwen3-4B"),
        "sft_first": st.sidebar.checkbox("SFT Pre-training", True),
        "retrain_interval": st.sidebar.slider("Retrain Interval", 5, 50, 20),
    }

st.sidebar.markdown("---")
run_button = st.sidebar.button("Run Backtest") if tickers else False
if not tickers:
    st.sidebar.warning("Select at least one ticker.")


# ═══════════════════════════════════════════════════════════════════════════════
# Main Area
# ═══════════════════════════════════════════════════════════════════════════════

st.title("Trading-R1 Backtest Dashboard")

if not tickers and not st.session_state.get("results"):
    st.info("Select at least one ticker from the sidebar to begin.")
    st.stop()

# Initialize session state
if "results" not in st.session_state:
    st.session_state.results = None
    st.session_state.summary_df = None
    st.session_state.experience_buffer = None
    st.session_state.run_mode = None
    st.session_state.collectors = None

# Run backtest
if run_button:
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

    results, summary_df, experience_buffer = run_backtest_pipeline(
        tickers=tickers,
        mode=mode,
        start_date=start_str,
        end_date=end_str,
        initial_capital=initial_capital,
        tx_cost_bps=tx_cost,
        rebalance_days=rebalance_days,
        icrl_params=icrl_params,
        grpo_params=grpo_params,
    )

    if results:
        st.session_state.results = results
        st.session_state.summary_df = summary_df
        st.session_state.experience_buffer = experience_buffer
        st.session_state.run_mode = mode

# Display results
if st.session_state.results:
    results = st.session_state.results
    summary_df = st.session_state.summary_df
    experience_buffer = st.session_state.experience_buffer
    run_mode = st.session_state.run_mode

    # Tabs
    tab_names = ["Metrics", "Equity Curves", "Signals"]
    if experience_buffer and run_mode in ("ICRL", "GRPO"):
        tab_names.append("RL Rewards")

    tabs = st.tabs(tab_names)

    # ── Tab 1: Metrics ─────────────────────────────────────────────────────
    with tabs[0]:
        st.subheader("Performance Metrics")
        st.dataframe(summary_df, )

        # Key metrics cards
        if len(results) == 1:
            ticker = list(results.keys())[0]
            r = results[ticker]
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Cumulative Return", f"{r.cumulative_return*100:+.2f}%")
            col2.metric("Sharpe Ratio", f"{r.sharpe_ratio:.2f}")
            col3.metric("Hit Rate", f"{r.hit_rate*100:.1f}%")
            col4.metric("Max Drawdown", f"{r.max_drawdown*100:.2f}%")

        # RL buffer stats
        if experience_buffer:
            st.markdown("---")
            st.subheader("RL Experience Buffer")
            stats = experience_buffer.summary_stats()
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Completed Trades", stats["completed"])
            c2.metric("Mean Reward", f"{stats['mean_reward']:+.4f}")
            c3.metric("Positive %", f"{stats['positive_pct']:.1f}%")
            c4.metric("Pending", stats["pending"])

            if "dimension_means" in stats:
                st.markdown("**Per-Dimension Averages**")
                dim_df = pd.DataFrame([stats["dimension_means"]]).T
                dim_df.columns = ["Mean Score"]
                st.dataframe(dim_df.style.format("{:+.3f}"), )

    # ── Tab 2: Equity Curves ───────────────────────────────────────────────
    with tabs[1]:
        st.subheader("Equity Curves & Drawdown")
        for ticker, r in results.items():
            if r.equity_curve.empty:
                continue

            st.markdown(f"### {ticker}")
            fig_eq = make_equity_chart(r)
            st.pyplot(fig_eq)
            plt.close(fig_eq)

            fig_dd = make_drawdown_chart(r)
            st.pyplot(fig_dd)
            plt.close(fig_dd)

    # ── Tab 3: Signals ─────────────────────────────────────────────────────
    with tabs[2]:
        st.subheader("Signal Distributions")
        cols = st.columns(min(len(results), 3))
        for i, (ticker, r) in enumerate(results.items()):
            with cols[i % len(cols)]:
                fig_sig = make_signal_distribution(r)
                if fig_sig:
                    st.pyplot(fig_sig)
                    plt.close(fig_sig)

        # Signal series table
        st.markdown("---")
        st.subheader("Signal Series")
        for ticker, r in results.items():
            if not r.signals.empty:
                with st.expander(f"{ticker} — Signal History"):
                    sig_df = pd.DataFrame({
                        "Date": r.signals.index,
                        "Signal": r.signals.values,
                    })
                    st.dataframe(sig_df, )

    # ── Tab 4: RL Rewards ──────────────────────────────────────────────────
    if experience_buffer and run_mode in ("ICRL", "GRPO") and len(tabs) > 3:
        with tabs[3]:
            st.subheader("Multi-Dimensional Reward Analysis")

            # Reward evolution
            fig_rew = make_reward_evolution(experience_buffer)
            if fig_rew:
                st.pyplot(fig_rew)
                plt.close(fig_rew)

            st.markdown("---")

            # Dimension radar + evolution side by side
            col_radar, col_evo = st.columns([1, 2])

            with col_radar:
                st.markdown("### Dimension Profile")
                fig_radar = make_dimension_radar(experience_buffer)
                if fig_radar:
                    st.pyplot(fig_radar)
                    plt.close(fig_radar)

            with col_evo:
                st.markdown("### Dimension Evolution")
                fig_dim = make_dimension_evolution(experience_buffer)
                if fig_dim:
                    st.pyplot(fig_dim)
                    plt.close(fig_dim)

else:
    st.info("Configure parameters in the sidebar and click **Run Backtest** to begin.")
