"""
In-Context RL (ICRL) Analyst — Simulated RL via Claude's Context Window

ICRL exploits Claude's large context window to simulate reinforcement learning
without modifying any model weights. The "policy update" happens through
prompt augmentation: we feed back past (state, action, reward) outcomes
so that Claude can learn from its own trading history.

Architecture: Adversarial Dimension Debate (ABM-inspired)
  Instead of a single monolithic analysis, ICRL structures every decision as
  a debate between two expert councils in REWARD-DIMENSION SPACE:

    Alpha Council (thesis):   Sharpe, Direction, Conviction, Improvement
    Risk Council (antithesis): Risk Discipline, Regime, Coherence, Override
    CIO Synthesis:             Weighs both councils → final composite-optimal action

  This is NOT bull-vs-bear. Both councils may agree on direction but disagree on
  sizing, timing, or regime alignment. The debate happens in the space of the
  8-dimensional reward function, forcing the model to optimize the full composite
  reward rather than just returns.

Prompt structure:
  1. Base system prompt (same as standard LLM mode)
  2. <experience_reflection> section with dimension debate instructions
  3. <trading_history> block containing:
     - Best/worst/recent trades with per-dimension breakdown
     - Per-dimension performance summary showing strengths and weaknesses
  4. Current market data prompt (same as standard LLM mode)

The key insight: By debating across reward dimensions (not just direction),
Claude learns to balance return-seeking with risk discipline — producing
decisions that score well on the composite reward, not just Sharpe alone.
"""
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

import config
import rl_config
from llm_analyst import (
    SYSTEM_PROMPT,
    build_data_prompt,
    call_claude,
    rule_based_decision,
)
from experience_buffer import ExperienceBuffer, Experience

logger = logging.getLogger(__name__)


# Extended system prompt for ICRL mode — Adversarial Dimension Debate
ICRL_SYSTEM_PROMPT = SYSTEM_PROMPT + """

<experience_reflection>
You are operating in IN-CONTEXT REINFORCEMENT LEARNING mode with ADVERSARIAL
DIMENSION DEBATE. You are optimizing an 8-DIMENSION COMPOSITE REWARD — not raw P&L.
Every decision is scored across all dimensions, weighted and summed. Understanding
these dimensions is essential to maximizing your composite score.

THE 8 REWARD DIMENSIONS (each scored -1.0 to +1.0):

  ALPHA COUNCIL owns (return-seeking, 60% of composite weight):
    1. Sharpe (25%):      (position_return - costs) / holding_volatility.
                           Maximize risk-adjusted returns, not raw returns.
    2. Direction (15%):   +1.0 if your position direction matched the price move.
                           -1.0 if opposite. The foundation of all other alpha.
    3. Conviction (10%):  Was your position size proportional to the actual move?
                           STRONG BUY on +5% move = +1.0. STRONG BUY on flat = -1.0.
                           Match sizing to certainty.
    4. Improvement (10%): Are your forecast errors shrinking over time?
                           Measures learning trajectory across trades.

  RISK COUNCIL owns (discipline & coherence, 40% of composite weight):
    5. Risk Discipline (10%): HIGH-vol regime → smaller positions score better.
                               LOW-vol regime → deploying capital scores better.
    6. Regime (10%):          Trend-follow in trending markets. Mean-revert in ranges.
                               Fighting the regime is penalized.
    7. Coherence (10%):       Reasoning must match reality. Bullish thesis + stock
                               went up = coherent. Also rewards counter-argument
                               awareness ("what could go wrong"). Ignoring downside
                               risks on losing trades is heavily penalized.
    8. Override (10%):        Overriding Algorithm S1 is net negative on average.
                               Only override with strong evidence. Good overrides
                               earn +1.0. Bad overrides earn -1.0.

DEBATE PROTOCOL — Replace the <conclusion> section with the following:

<debate>

<alpha_council>
You are the ALPHA COUNCIL. Your mandate: maximize the return-seeking dimensions
(Sharpe, Direction, Conviction, Improvement).

For each dimension, make a QUANTIFIED argument:

  1. SHARPE: Estimate the expected 5-day return vs. holding-period volatility.
     Structure: "Expected return [±X%] from [evidence]. Holding vol ~[Y%]/day.
     Sharpe ≈ X/Y ≈ [Z] → supports [ACTION]."

  2. DIRECTION: State the weight of evidence for direction.
     Structure: "[N] of [M] signals align [direction] ([list them]).
     Counter-signals: [list]. Net: [direction] with [high/moderate/low] confidence."

  3. CONVICTION: Map your signal alignment to a position size.
     "Strong alignment ([N]/[M] signals) + fundamental support → high conviction → STRONG BUY."
     "Mixed signals ([N]/[M] agree) → low conviction → BUY at most, or HOLD."
     Reference past calibration errors if your Conviction dimension is weak.

  4. IMPROVEMENT: Reference SPECIFIC past errors from your trading history.
     "My recent losses were [pattern]. This setup differs because [reasoning],
     so I'm correcting that bias by [specific adjustment]."
     If no history is available, state your analytical framework and potential blind spots.

Propose: STRONG SELL / SELL / HOLD / BUY / STRONG BUY
</alpha_council>

<risk_council>
You are the RISK COUNCIL. Your mandate: maximize the discipline dimensions
(Risk Discipline, Regime, Coherence, Override).

For each dimension, challenge the Alpha Council with specific evidence:

  1. RISK DISCIPLINE: Assess the current volatility regime.
     "ATR = [X]. BB width = [Z] ([above/below] average).
     This is a [HIGH/NORMAL/LOW]-vol regime. Alpha proposes [ACTION] = [W%] position.
     In this vol regime, the reward function [rewards/penalizes] that sizing.
     Recommendation: [adjust sizing / maintain]."

  2. REGIME: Determine market regime and check alignment.
     "SMA50 vs SMA200: [aligned/crossed]. ADX = [X]: [trending/range-bound].
     Price structure: [higher highs / lower lows / sideways].
     Regime: [UPTREND / DOWNTREND / RANGE]. Alpha's thesis [aligns with / fights]
     this regime."

  3. COHERENCE: Stress-test Alpha's reasoning for internal consistency.
     Identify cherry-picked data, ignored contradictions, or logical gaps.
     Structure: "Alpha cites [indicator]=[value] to support [conclusion], but
     [other indicator]=[value] contradicts this because [reasoning]."
     IMPORTANT: If Alpha's reasoning is internally consistent and well-evidenced,
     state this explicitly. Manufactured disagreement is penalized by the Coherence
     dimension. Genuine concurrence ("no objections on coherence grounds") is
     valuable — do not invent objections to justify your role.

  4. OVERRIDE: Check the <signal_reference> section in the market data for the
     current S1 signal. If Alpha's proposed action disagrees with S1, demand
     justification. "S1 = [SIGNAL]. Alpha proposes [ACTION]. This is an override.
     Based on my Override dimension history, overrides have been [net positive /
     net negative]. [Justified / Not justified] because [specific evidence]."
     If no history exists, apply a strong prior that overrides are net negative.

If warranted, propose a different action or different sizing.
</risk_council>

<synthesis>
You are the CHIEF INVESTMENT OFFICER. Produce the final decision that maximizes
the COMPOSITE REWARD across all 8 dimensions — not just returns.

Structure your synthesis:
  1. VERDICT: Which council's core thesis prevailed and why.
  2. COMPROMISE: What adjustments balanced return-seeking with discipline.
     Example: "Alpha correct on direction → BUY. But Risk's vol-regime analysis
     shows elevated ATR (80th percentile) → size down: BUY not STRONG BUY.
     Override of S1 is justified because [specific evidence]."
  3. KEY RISK: The single factor most likely to make this trade fail.
  4. CATALYST: What must happen within 5 trading days for the thesis to pay off.

DECISION: [[[STRONG SELL / SELL / HOLD / BUY / STRONG BUY]]]
</synthesis>

</debate>

CALIBRATION FROM HISTORY — Study the <trading_history> section below:
  • Dimensions tagged << WEAK need active correction in your debate
  • Dimensions tagged << STRONG: maintain current approach
  • COUNCIL TRUST CALIBRATION tells you which council to weight more
  • If Override scores are negative → default to following S1 unless evidence is extreme
  • If Coherence scores are negative → include more counter-arguments ("what could go wrong")
  • If Conviction is weak → be more deliberate about mapping certainty to sizing
  • If no trading history is available yet, apply equal weight to both councils
    and follow S1 unless you have strong independent evidence for an override
</experience_reflection>
"""


def _format_experience_for_prompt(exp: Experience, max_reasoning: int = None) -> str:
    """Format a single experience for inclusion in the ICRL prompt with dimension breakdown."""
    max_reasoning = max_reasoning or rl_config.ICRL_REASONING_TRUNCATE

    lines = [
        f"  Date: {exp.date} | Ticker: {exp.ticker}",
        f"  Signal (S1): {exp.signal_label} -> Action Taken: {exp.action}",
    ]

    if exp.reward_record:
        rr = exp.reward_record
        lines.append(
            f"  Composite Reward: {rr.reward:+.4f} | "
            f"Raw P&L: {rr.raw_pnl*100:+.2f}% | "
            f"Volatility: {rr.holding_volatility:.4f} | "
            f"Holding: {rr.holding_days}d"
        )

        # Per-dimension breakdown
        if rr.dimensions:
            d = rr.dimensions
            dir_sym = "+" if d.direction > 0 else ("-" if d.direction < 0 else "~")
            risk_sym = "+" if d.risk_discipline > 0 else ("-" if d.risk_discipline < 0 else "~")
            regime_sym = "+" if d.regime > 0 else ("-" if d.regime < 0 else "~")

            lines.append(
                f"  Dimensions: "
                f"Sharpe={d.sharpe:+.2f} | "
                f"Direction={dir_sym} | "
                f"Conviction={d.conviction:+.2f} | "
                f"Override={d.override:+.2f}"
            )
            lines.append(
                f"             "
                f"Improvement={d.improvement:+.2f} | "
                f"Risk={risk_sym} | "
                f"Coherence={d.coherence:+.2f} | "
                f"Regime={regime_sym}"
            )

    if exp.reasoning:
        truncated = exp.reasoning[:max_reasoning]
        if len(exp.reasoning) > max_reasoning:
            truncated += "..."
        lines.append(f"  Reasoning: {truncated}")

    return "\n".join(lines)


def build_icrl_prompt(
    data_prompt: str,
    experience_buffer: ExperienceBuffer,
    top_k: int = None,
    bottom_k: int = None,
    recent_n: int = None,
) -> str:
    """
    Build the full ICRL prompt by combining market data with trading history.

    Args:
        data_prompt: Standard market data prompt from build_data_prompt()
        experience_buffer: Buffer of past experiences
        top_k: Number of best trades to include
        bottom_k: Number of worst trades to include
        recent_n: Number of recent trades to include

    Returns:
        Combined prompt with trading history appended
    """
    top_k = top_k or rl_config.ICRL_TOP_K_BEST
    bottom_k = bottom_k or rl_config.ICRL_BOTTOM_K_WORST
    recent_n = recent_n or rl_config.ICRL_RECENT_N

    completed = experience_buffer.get_completed()

    if not completed:
        # No history yet — just return the data prompt
        return data_prompt

    sections = [data_prompt, "\n<trading_history>"]

    # Best trades
    best = experience_buffer.get_top_k(top_k)
    if best:
        sections.append(f"\n[BEST TRADES (top {len(best)} by composite reward)]")
        for i, exp in enumerate(best, 1):
            sections.append(f"\n  --- Trade #{i} ---")
            sections.append(_format_experience_for_prompt(exp))

    # Worst trades
    worst = experience_buffer.get_bottom_k(bottom_k)
    if worst:
        sections.append(f"\n[WORST TRADES (bottom {len(worst)} by composite reward)]")
        for i, exp in enumerate(worst, 1):
            sections.append(f"\n  --- Trade #{i} ---")
            sections.append(_format_experience_for_prompt(exp))

    # Recent trades
    recent = experience_buffer.get_recent_n(recent_n)
    if recent:
        sections.append(f"\n[RECENT TRADES (last {len(recent)})]")
        for i, exp in enumerate(recent, 1):
            sections.append(f"\n  --- Trade #{i} ---")
            sections.append(_format_experience_for_prompt(exp))

    # Summary statistics
    stats = experience_buffer.summary_stats()
    sections.append(f"\n[PERFORMANCE SUMMARY]")
    sections.append(f"  Total completed trades: {stats['completed']}")
    sections.append(f"  Mean composite reward: {stats['mean_reward']:+.4f}")
    sections.append(f"  Reward std: {stats['std_reward']:.4f}")
    sections.append(f"  Best reward: {stats['max_reward']:+.4f}")
    sections.append(f"  Worst reward: {stats['min_reward']:+.4f}")
    sections.append(f"  Positive trade %: {stats['positive_pct']:.1f}%")

    # Per-dimension averages with council grouping
    if "dimension_means" in stats:
        dim_means = stats["dimension_means"]

        alpha_dims = ["sharpe", "direction", "conviction", "improvement"]
        risk_dims = ["risk_discipline", "regime", "coherence", "override"]

        sections.append(f"\n[ALPHA COUNCIL DIMENSIONS — Return-Seeking]")
        for dim_name in alpha_dims:
            if dim_name in dim_means:
                score = dim_means[dim_name]
                bar_len = int(abs(score) * 10)
                bar = ("+" * bar_len if score >= 0 else "-" * bar_len).ljust(10)
                tag = " << WEAK" if score < -0.1 else (" << STRONG" if score > 0.15 else "")
                sections.append(f"  {dim_name:>16s}: {score:+.3f} {bar}{tag}")

        sections.append(f"\n[RISK COUNCIL DIMENSIONS — Discipline & Coherence]")
        for dim_name in risk_dims:
            if dim_name in dim_means:
                score = dim_means[dim_name]
                bar_len = int(abs(score) * 10)
                bar = ("+" * bar_len if score >= 0 else "-" * bar_len).ljust(10)
                tag = " << WEAK" if score < -0.1 else (" << STRONG" if score > 0.15 else "")
                sections.append(f"  {dim_name:>16s}: {score:+.3f} {bar}{tag}")

        # Council-level summary
        alpha_mean = np.mean([dim_means.get(d, 0) for d in alpha_dims])
        risk_mean = np.mean([dim_means.get(d, 0) for d in risk_dims])
        sections.append(f"\n[COUNCIL TRUST CALIBRATION]")
        sections.append(f"  Alpha Council avg: {alpha_mean:+.3f}  |  Risk Council avg: {risk_mean:+.3f}")
        if alpha_mean < risk_mean - 0.05:
            sections.append(f"  >> Alpha Council is underperforming — give Risk Council more weight")
        elif risk_mean < alpha_mean - 0.05:
            sections.append(f"  >> Risk Council is underperforming — focus on discipline dimensions")
        else:
            sections.append(f"  >> Councils are balanced — maintain debate equilibrium")

    sections.append("\n</trading_history>")

    return "\n".join(sections)


class ICRLAnalyst:
    """
    Trading analyst using In-Context RL with Claude.

    Same analyze() interface as LLMAnalyst, but augments prompts with
    past trading experience (including per-dimension reward breakdowns)
    to simulate reinforcement learning.
    """

    def __init__(
        self,
        experience_buffer: ExperienceBuffer,
        observer=None,
        top_k: int = None,
        bottom_k: int = None,
        recent_n: int = None,
    ):
        self.experience_buffer = experience_buffer
        self.obs = observer
        self.top_k = top_k or rl_config.ICRL_TOP_K_BEST
        self.bottom_k = bottom_k or rl_config.ICRL_BOTTOM_K_WORST
        self.recent_n = recent_n or rl_config.ICRL_RECENT_N

    def analyze(
        self,
        snapshot: Dict,
        news: Dict = None,
        signal_label: str = None,
    ) -> Tuple[str, str]:
        """
        Generate a trading decision using ICRL (Claude + experience history).

        Args:
            snapshot: Market data snapshot
            news: News data
            signal_label: Algorithm S1 label

        Returns:
            (thesis_text, decision)
        """
        ticker = snapshot.get("ticker", "UNKNOWN")
        date = snapshot.get("date", "")

        # Build standard data prompt (include S1 signal so the model can evaluate overrides)
        data_prompt = build_data_prompt(snapshot, news, signal_label=signal_label)

        # Augment with trading history
        icrl_prompt = build_icrl_prompt(
            data_prompt,
            self.experience_buffer,
            self.top_k,
            self.bottom_k,
            self.recent_n,
        )

        # Log prompt stats
        if self.obs:
            completed = len(self.experience_buffer.get_completed())
            self.obs.log_icrl_prompt_stats(
                ticker, date, len(icrl_prompt), completed,
                self.top_k, self.bottom_k, self.recent_n,
            )

        # Call Claude with ICRL system prompt
        thesis, decision = call_claude(
            icrl_prompt,
            observer=self.obs,
            ticker=ticker,
            date=date,
            system_prompt=ICRL_SYSTEM_PROMPT,
        )

        if decision and decision in config.ACTION_LABELS:
            return thesis or "", decision

        # Fallback to rule-based
        logger.warning(f"ICRL: Claude failed for {ticker} {date}; using rule-based fallback")
        thesis, decision = rule_based_decision(snapshot, signal_label)
        return thesis, decision
