"""
RL Visualizer — Comparison Charts for RL vs Standard Modes

Generates:
  1. Mode comparison equity curves (overlay 4 modes per ticker)
  2. Mode comparison Sharpe ratio heatmap
  3. Reward evolution over time (for ICRL/GRPO)
  4. GRPO training curve (loss, KL, reward over steps)
  5. Action distribution comparison across modes
  6. Reward dimensions radar chart (per-dimension average scores)
  7. Dimension evolution over time (per-dimension score trajectories)
"""
import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    logger.warning("matplotlib not installed; RL visualization unavailable")


MODE_COLORS = {
    "signal_only": "#1f77b4",   # Blue
    "llm_no_rl":   "#ff7f0e",   # Orange
    "icrl":        "#2ca02c",   # Green
    "grpo":        "#d62728",   # Red
}

MODE_LABELS = {
    "signal_only": "Signal-Only",
    "llm_no_rl":   "LLM (no RL)",
    "icrl":        "ICRL (Claude)",
    "grpo":        "GRPO (Local)",
}


def plot_mode_comparison_equity(
    results_by_mode: Dict,
    ticker: str,
    save_path: str = None,
    figsize: tuple = (14, 7),
):
    """
    Overlay equity curves from multiple modes for a single ticker.

    Args:
        results_by_mode: {mode_name: {ticker: BacktestResult}}
        ticker: Which ticker to plot
        save_path: Path to save the chart
        figsize: Figure size
    """
    if not HAS_MPL:
        return

    fig, ax = plt.subplots(figsize=figsize)

    for mode_name, mode_results in results_by_mode.items():
        if ticker not in mode_results:
            continue
        result = mode_results[ticker]
        if result.equity_curve.empty:
            continue

        color = MODE_COLORS.get(mode_name, "#888888")
        label = MODE_LABELS.get(mode_name, mode_name)

        # Normalize to start at 100
        normalized = result.equity_curve / result.equity_curve.iloc[0] * 100
        ax.plot(normalized.index, normalized.values,
                color=color, linewidth=2, label=f"{label} (SR: {result.sharpe_ratio:.2f})")

    ax.set_title(f"Equity Curve Comparison — {ticker}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value (normalized to 100)")
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=100, color="gray", linestyle="--", alpha=0.5)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved: {save_path}")
    plt.close()


def plot_mode_comparison_heatmap(
    summary_by_mode: Dict,
    save_path: str = None,
    figsize: tuple = (12, 6),
):
    """
    Heatmap of Sharpe ratios: modes (rows) x tickers (columns).

    Args:
        summary_by_mode: {mode_label: DataFrame with SR column}
        save_path: Path to save the chart
    """
    if not HAS_MPL:
        return

    # Build matrix
    modes = list(summary_by_mode.keys())
    tickers = list(summary_by_mode[modes[0]].index) if modes else []

    if not modes or not tickers:
        return

    data = np.zeros((len(modes), len(tickers)))
    for i, mode in enumerate(modes):
        df = summary_by_mode[mode]
        for j, ticker in enumerate(tickers):
            if ticker in df.index and "SR" in df.columns:
                data[i, j] = df.loc[ticker, "SR"]

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(data, cmap="RdYlGn", aspect="auto")

    # Labels
    ax.set_xticks(range(len(tickers)))
    ax.set_xticklabels(tickers, rotation=45, ha="right")
    ax.set_yticks(range(len(modes)))
    ax.set_yticklabels(modes)

    # Annotate cells
    for i in range(len(modes)):
        for j in range(len(tickers)):
            val = data[i, j]
            color = "white" if abs(val) > 1.0 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    color=color, fontsize=10, fontweight="bold")

    ax.set_title("Sharpe Ratio Comparison by Mode and Ticker",
                 fontsize=14, fontweight="bold")
    plt.colorbar(im, ax=ax, label="Sharpe Ratio")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved: {save_path}")
    plt.close()


def plot_reward_evolution(
    experience_buffer,
    save_path: str = None,
    figsize: tuple = (14, 6),
):
    """
    Plot reward evolution over time: bar chart + cumulative reward + rolling average.

    Args:
        experience_buffer: ExperienceBuffer with completed experiences
        save_path: Path to save the chart
    """
    if not HAS_MPL:
        return

    completed = experience_buffer.get_completed()
    if not completed:
        return

    rewards = [e.reward for e in completed if e.reward is not None]
    dates = [e.date for e in completed if e.reward is not None]

    if not rewards:
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)

    x = range(len(rewards))
    colors = ["green" if r > 0 else "red" for r in rewards]

    # Bar chart of individual rewards
    ax1.bar(x, rewards, color=colors, alpha=0.6, width=0.8)
    ax1.axhline(y=0, color="black", linewidth=0.5)
    ax1.set_ylabel("Reward")
    ax1.set_title("Individual Trade Rewards", fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.3)

    # Rolling average
    window = min(10, len(rewards))
    if window > 1:
        rolling_avg = pd.Series(rewards).rolling(window).mean()
        ax1.plot(x, rolling_avg.values, color="blue", linewidth=2,
                 label=f"Rolling avg ({window})")
        ax1.legend()

    # Cumulative reward
    cumulative = np.cumsum(rewards)
    ax2.plot(x, cumulative, color="purple", linewidth=2)
    ax2.fill_between(x, cumulative, alpha=0.2, color="purple")
    ax2.set_xlabel("Trade Number")
    ax2.set_ylabel("Cumulative Reward")
    ax2.set_title("Cumulative Reward Over Time", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color="black", linewidth=0.5)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved: {save_path}")
    plt.close()


def plot_grpo_training_curve(
    training_log: List[Dict],
    save_path: str = None,
    figsize: tuple = (14, 8),
):
    """
    Plot GRPO training metrics: loss, KL divergence, mean reward over steps.

    Args:
        training_log: List of step dicts from GRPOTrainer.get_training_log()
        save_path: Path to save the chart
    """
    if not HAS_MPL or not training_log:
        return

    steps = [s["step"] for s in training_log]
    losses = [s["loss"] for s in training_log]
    kl_divs = [s["kl_div"] for s in training_log]
    mean_rewards = [s["mean_reward"] for s in training_log]

    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)

    # Loss
    axes[0].plot(steps, losses, color="blue", linewidth=1.5)
    axes[0].set_ylabel("Policy Loss")
    axes[0].set_title("GRPO Training Curve", fontsize=14, fontweight="bold")
    axes[0].grid(True, alpha=0.3)

    # KL divergence
    axes[1].plot(steps, kl_divs, color="orange", linewidth=1.5)
    axes[1].set_ylabel("KL Divergence")
    axes[1].grid(True, alpha=0.3)

    # Mean reward
    axes[2].plot(steps, mean_rewards, color="green", linewidth=1.5)
    axes[2].set_xlabel("Training Step")
    axes[2].set_ylabel("Mean Reward")
    axes[2].axhline(y=0, color="black", linewidth=0.5)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved: {save_path}")
    plt.close()


def plot_action_distribution_comparison(
    results_by_mode: Dict,
    ticker: str,
    save_path: str = None,
    figsize: tuple = (12, 6),
):
    """
    Compare action distributions across modes for a single ticker.

    Args:
        results_by_mode: {mode_name: {ticker: BacktestResult}}
        ticker: Which ticker to plot
        save_path: Path to save the chart
    """
    if not HAS_MPL:
        return

    from config import ACTION_LABELS

    modes = []
    distributions = []

    for mode_name, mode_results in results_by_mode.items():
        if ticker not in mode_results:
            continue
        result = mode_results[ticker]
        if result.signals.empty:
            continue

        counts = result.signals.value_counts()
        total = counts.sum()
        dist = {label: counts.get(label, 0) / total * 100 for label in ACTION_LABELS}
        modes.append(MODE_LABELS.get(mode_name, mode_name))
        distributions.append(dist)

    if not modes:
        return

    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(ACTION_LABELS))
    width = 0.8 / len(modes)

    action_colors = {
        "STRONG_SELL": "#d62728",
        "SELL": "#ff7f0e",
        "HOLD": "#ffbb78",
        "BUY": "#2ca02c",
        "STRONG_BUY": "#1f77b4",
    }

    for i, (mode, dist) in enumerate(zip(modes, distributions)):
        values = [dist[label] for label in ACTION_LABELS]
        offset = (i - len(modes) / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=mode, alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(ACTION_LABELS, rotation=45, ha="right")
    ax.set_ylabel("Frequency (%)")
    ax.set_title(f"Action Distribution Comparison — {ticker}",
                 fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved: {save_path}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Multi-Dimensional Reward Charts
# ─────────────────────────────────────────────────────────────────────────────

DIMENSION_NAMES = [
    "sharpe", "direction", "conviction", "improvement",
    "override", "risk_discipline", "coherence", "regime",
]

DIMENSION_LABELS = {
    "sharpe": "Sharpe\n(P&L)",
    "direction": "Direction\nAccuracy",
    "conviction": "Conviction\nCalibration",
    "improvement": "Forecast\nImprovement",
    "override": "Override\nQuality",
    "risk_discipline": "Risk\nDiscipline",
    "coherence": "Analytical\nCoherence",
    "regime": "Regime\nAwareness",
}


def plot_reward_dimensions_radar(
    experience_buffer,
    save_path: str = None,
    figsize: tuple = (8, 8),
):
    """
    Radar/spider chart showing average per-dimension scores.

    Args:
        experience_buffer: ExperienceBuffer with completed experiences
        save_path: Path to save the chart
    """
    if not HAS_MPL:
        return

    completed = experience_buffer.get_completed()
    if not completed:
        return

    # Collect per-dimension scores
    dim_scores = {name: [] for name in DIMENSION_NAMES}
    for exp in completed:
        if exp.dimensions:
            d = exp.dimensions.to_dict()
            for name in DIMENSION_NAMES:
                if name in d:
                    dim_scores[name].append(d[name])

    if not any(dim_scores.values()):
        return

    # Compute means
    means = [np.mean(dim_scores[name]) if dim_scores[name] else 0.0
             for name in DIMENSION_NAMES]

    # Radar chart
    N = len(DIMENSION_NAMES)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    means_plot = means + means[:1]

    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))

    # Plot the values
    ax.plot(angles, means_plot, 'o-', linewidth=2, color='#2ca02c')
    ax.fill(angles, means_plot, alpha=0.2, color='#2ca02c')

    # Plot zero circle
    zero_line = [0.0] * (N + 1)
    ax.plot(angles, zero_line, '--', linewidth=0.5, color='gray', alpha=0.5)

    # Labels
    labels = [DIMENSION_LABELS.get(name, name) for name in DIMENSION_NAMES]
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9)

    # Set y limits
    max_abs = max(abs(m) for m in means) if means else 1.0
    limit = max(max_abs * 1.3, 0.5)
    ax.set_ylim(-limit, limit)

    # Add value labels
    for angle, value, name in zip(angles[:-1], means, DIMENSION_NAMES):
        ax.annotate(f"{value:+.2f}",
                    xy=(angle, value),
                    xytext=(0, 10),
                    textcoords='offset points',
                    ha='center', va='bottom',
                    fontsize=8, fontweight='bold',
                    color='green' if value >= 0 else 'red')

    ax.set_title("Multi-Dimensional Reward Profile",
                 fontsize=14, fontweight="bold", pad=20)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved: {save_path}")
    plt.close()


def plot_dimension_evolution(
    experience_buffer,
    save_path: str = None,
    figsize: tuple = (16, 10),
):
    """
    Line chart of each dimension score over time with rolling averages.

    Args:
        experience_buffer: ExperienceBuffer with completed experiences
        save_path: Path to save the chart
    """
    if not HAS_MPL:
        return

    completed = experience_buffer.get_completed()
    if not completed:
        return

    # Collect per-dimension score series
    dim_series = {name: [] for name in DIMENSION_NAMES}
    trade_numbers = []
    trade_idx = 0

    for exp in completed:
        if exp.dimensions:
            d = exp.dimensions.to_dict()
            for name in DIMENSION_NAMES:
                dim_series[name].append(d.get(name, 0.0))
            trade_numbers.append(trade_idx)
            trade_idx += 1

    if not trade_numbers:
        return

    # Create 2x4 subplot grid for 8 dimensions
    fig, axes = plt.subplots(2, 4, figsize=figsize, sharex=True)
    axes = axes.flatten()

    dim_colors = {
        "sharpe": "#1f77b4",
        "direction": "#ff7f0e",
        "conviction": "#2ca02c",
        "improvement": "#d62728",
        "override": "#9467bd",
        "risk_discipline": "#8c564b",
        "coherence": "#e377c2",
        "regime": "#7f7f7f",
    }

    for idx, name in enumerate(DIMENSION_NAMES):
        ax = axes[idx]
        scores = dim_series[name]
        x = trade_numbers[:len(scores)]
        color = dim_colors.get(name, "#333333")

        # Individual scores as scatter
        ax.scatter(x, scores, color=color, alpha=0.3, s=15)

        # Rolling average
        window = min(10, max(2, len(scores) // 5))
        if len(scores) >= window:
            rolling = pd.Series(scores).rolling(window).mean()
            ax.plot(x, rolling.values, color=color, linewidth=2)

        ax.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")
        ax.set_ylim(-1.3, 1.3)
        ax.set_title(DIMENSION_LABELS.get(name, name).replace("\n", " "),
                     fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.2)

        # Mean annotation
        mean_val = np.mean(scores) if scores else 0
        ax.axhline(y=mean_val, color=color, linewidth=1, linestyle=":",
                   alpha=0.5)

    # Common labels
    fig.text(0.5, 0.02, 'Trade Number', ha='center', fontsize=12)
    fig.text(0.02, 0.5, 'Dimension Score', va='center', rotation='vertical',
             fontsize=12)

    fig.suptitle("Reward Dimension Evolution Over Time",
                 fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0.03, 0.03, 1, 0.97])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved: {save_path}")
    plt.close()
