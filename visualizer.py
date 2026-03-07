"""
Visualization Module for Trading-R1 Backtest Results

Generates charts matching the paper's presentation style:
- Equity curves (vs buy-and-hold benchmark)
- Drawdown charts
- Signal distribution
- Sharpe ratio heatmap (Figure 5)
- Performance summary table
"""
import logging
import os
from typing import Dict, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

import config
from backtester import BacktestResult

logger = logging.getLogger(__name__)

# Style
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 10,
})

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


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Individual Charts
# ═══════════════════════════════════════════════════════════════════════════════

def plot_equity_curve(
    result: BacktestResult,
    benchmark_prices: pd.Series = None,
    save_path: str = None,
):
    """Plot equity curve with optional buy-and-hold benchmark."""
    fig, ax = plt.subplots(figsize=(12, 5))

    # Strategy equity
    ax.plot(result.equity_curve.index, result.equity_curve.values,
            color=COLORS["equity"], linewidth=1.5, label="Trading-R1 Strategy")

    # Benchmark (buy-and-hold)
    if benchmark_prices is not None and not benchmark_prices.empty:
        mask = benchmark_prices.index.isin(result.equity_curve.index) | (
            (benchmark_prices.index >= result.equity_curve.index[0]) &
            (benchmark_prices.index <= result.equity_curve.index[-1])
        )
        bm = benchmark_prices[mask]
        if not bm.empty:
            bm_equity = (bm / bm.iloc[0]) * config.INITIAL_CAPITAL
            ax.plot(bm_equity.index, bm_equity.values,
                    color=COLORS["benchmark"], linewidth=1.2,
                    linestyle="--", label="Buy & Hold")

    ax.set_title(f"{result.ticker} — Equity Curve", fontsize=13, fontweight="bold")
    ax.set_ylabel("Portfolio Value ($)")
    ax.yaxis.set_major_formatter(mtick.StrMethodFormatter("${x:,.0f}"))
    ax.legend(loc="upper left")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"  Saved equity curve: {save_path}")
    plt.close(fig)


def plot_drawdown(result: BacktestResult, save_path: str = None):
    """Plot drawdown over time."""
    fig, ax = plt.subplots(figsize=(12, 3))

    ax.fill_between(result.drawdown_series.index,
                     0, -result.drawdown_series.values * 100,
                     color=COLORS["drawdown"], alpha=0.4)
    ax.plot(result.drawdown_series.index,
            -result.drawdown_series.values * 100,
            color=COLORS["drawdown"], linewidth=0.8)

    ax.set_title(f"{result.ticker} — Drawdown", fontsize=13, fontweight="bold")
    ax.set_ylabel("Drawdown (%)")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"  Saved drawdown chart: {save_path}")
    plt.close(fig)


def plot_signal_distribution(result: BacktestResult, save_path: str = None):
    """Pie chart of signal distribution."""
    if result.signals.empty:
        return

    counts = result.signals.value_counts()
    labels = []
    sizes = []
    colors = []
    for label in config.ACTION_LABELS:
        if label in counts:
            labels.append(label.replace("_", " "))
            sizes.append(counts[label])
            colors.append(COLORS.get(label, "#cccccc"))

    fig, ax = plt.subplots(figsize=(6, 6))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors,
        autopct="%1.1f%%", startangle=90,
        textprops={"fontsize": 9}
    )
    ax.set_title(f"{result.ticker} — Signal Distribution", fontsize=13, fontweight="bold")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"  Saved signal distribution: {save_path}")
    plt.close(fig)


def plot_returns_histogram(result: BacktestResult, save_path: str = None):
    """Histogram of daily returns."""
    if result.daily_returns.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(result.daily_returns.values * 100, bins=50,
            color=COLORS["equity"], alpha=0.7, edgecolor="white")
    ax.axvline(0, color="red", linestyle="--", linewidth=0.8)
    mean_ret = result.daily_returns.mean() * 100
    ax.axvline(mean_ret, color="green", linestyle="-", linewidth=1.2,
               label=f"Mean: {mean_ret:.3f}%")

    ax.set_title(f"{result.ticker} — Daily Returns Distribution", fontsize=13, fontweight="bold")
    ax.set_xlabel("Daily Return (%)")
    ax.set_ylabel("Frequency")
    ax.legend()
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"  Saved returns histogram: {save_path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# Multi-Ticker Charts
# ═══════════════════════════════════════════════════════════════════════════════

def plot_sharpe_heatmap(
    results: Dict[str, BacktestResult],
    save_path: str = None,
):
    """
    Sharpe ratio heatmap across tickers (inspired by Figure 5).
    Single row since we have one model, but shows relative performance.
    """
    tickers = sorted(results.keys())
    if not tickers:
        return

    sharpes = [results[t].sharpe_ratio for t in tickers]

    fig, ax = plt.subplots(figsize=(max(8, len(tickers) * 1.2), 3))
    data = np.array([sharpes])

    vmax = max(abs(min(sharpes)), abs(max(sharpes)), 0.1)
    im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(len(tickers)))
    ax.set_xticklabels(tickers, fontsize=9)
    ax.set_yticks([0])
    ax.set_yticklabels(["Trading-R1"])

    # Annotate cells
    for j, val in enumerate(sharpes):
        color = "white" if abs(val) > vmax * 0.6 else "black"
        ax.text(j, 0, f"{val:.2f}", ha="center", va="center",
                fontsize=10, fontweight="bold", color=color)

    plt.colorbar(im, ax=ax, label="Sharpe Ratio", shrink=0.8)
    ax.set_title("Sharpe Ratio Heatmap: Asset Tickers", fontsize=13, fontweight="bold")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"  Saved Sharpe heatmap: {save_path}")
    plt.close(fig)


def plot_combined_equity(
    results: Dict[str, BacktestResult],
    save_path: str = None,
):
    """Overlay equity curves for all tickers (normalized to $1)."""
    fig, ax = plt.subplots(figsize=(12, 6))

    for ticker in sorted(results.keys()):
        r = results[ticker]
        if not r.equity_curve.empty:
            normalized = r.equity_curve / r.equity_curve.iloc[0]
            ax.plot(normalized.index, normalized.values, linewidth=1.2, label=ticker)

    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.7, alpha=0.5)
    ax.set_title("Normalized Equity Curves (all tickers)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Growth of $1")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"  Saved combined equity: {save_path}")
    plt.close(fig)


def plot_metrics_table(
    summary_df: pd.DataFrame,
    save_path: str = None,
):
    """Render the summary metrics as a table figure."""
    fig, ax = plt.subplots(figsize=(10, max(2, 0.5 * len(summary_df) + 1)))
    ax.axis("off")

    table = ax.table(
        cellText=summary_df.values,
        colLabels=summary_df.columns,
        rowLabels=summary_df.index,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.4)

    # Color header
    for j in range(len(summary_df.columns)):
        table[0, j].set_facecolor("#2c3e50")
        table[0, j].set_text_props(color="white", fontweight="bold")

    ax.set_title("Trading-R1 Backtest Results", fontsize=14,
                 fontweight="bold", pad=20)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"  Saved metrics table: {save_path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# Unified Visualizer
# ═══════════════════════════════════════════════════════════════════════════════

class Visualizer:
    """Generate all charts for a backtest run."""

    def __init__(self, output_dir: str = None):
        self.output_dir = output_dir or os.path.join(config.RESULTS_DIR, "charts")
        ensure_dir(self.output_dir)

    def generate_all(
        self,
        results: Dict[str, BacktestResult],
        price_data: Dict[str, pd.DataFrame] = None,
        summary_df: pd.DataFrame = None,
    ):
        """Generate all charts for the backtest results."""
        logger.info("═══ Generating visualizations ═══")

        # Per-ticker charts
        for ticker, r in results.items():
            benchmark = None
            if price_data and ticker in price_data:
                benchmark = price_data[ticker]["close"]

            plot_equity_curve(
                r, benchmark,
                save_path=os.path.join(self.output_dir, f"equity_{ticker}.png")
            )
            plot_drawdown(
                r,
                save_path=os.path.join(self.output_dir, f"drawdown_{ticker}.png")
            )
            plot_signal_distribution(
                r,
                save_path=os.path.join(self.output_dir, f"signals_{ticker}.png")
            )
            plot_returns_histogram(
                r,
                save_path=os.path.join(self.output_dir, f"returns_{ticker}.png")
            )

        # Multi-ticker charts
        if len(results) > 1:
            plot_combined_equity(
                results,
                save_path=os.path.join(self.output_dir, "equity_combined.png")
            )

        plot_sharpe_heatmap(
            results,
            save_path=os.path.join(self.output_dir, "sharpe_heatmap.png")
        )

        if summary_df is not None:
            plot_metrics_table(
                summary_df,
                save_path=os.path.join(self.output_dir, "metrics_table.png")
            )

        logger.info(f"  All charts saved to {self.output_dir}")
