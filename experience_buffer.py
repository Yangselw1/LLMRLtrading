"""
Experience Buffer — Stores (state, action, reward) tuples for RL

Supports:
- Anti-lookahead: experiences are added with reward_record=None, then
  backfilled after the holding period completes
- Top-K / Bottom-K / Recent-N retrieval for ICRL prompt construction
- Serialization (save/load) for resuming across sessions
- Per-dimension reward tracking for multi-dimensional reward analysis
"""
import json
import logging
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

import numpy as np

from reward import RewardRecord, RewardDimensions, compute_forecast_error

logger = logging.getLogger(__name__)


@dataclass
class Experience:
    """A single (state, action, reward) experience."""
    step_idx: int
    date: str
    ticker: str
    state_snapshot: Dict          # Market data snapshot
    state_news: Optional[Dict]    # News data
    signal_label: str             # Algorithm S1 label
    action: str                   # Action taken (STRONG_SELL..STRONG_BUY)
    reasoning: str                # Model's reasoning text
    mode: str                     # "icrl", "grpo", or "sft"
    reward_record: Optional[RewardRecord] = None
    entry_price: float = 0.0
    position_weight: float = 0.0

    @property
    def is_completed(self) -> bool:
        """Whether the holding period has ended and reward has been computed."""
        return self.reward_record is not None

    @property
    def reward(self) -> Optional[float]:
        """Shorthand for composite reward value."""
        return self.reward_record.reward if self.reward_record else None

    @property
    def dimensions(self) -> Optional[RewardDimensions]:
        """Shorthand for reward dimensions."""
        if self.reward_record and self.reward_record.dimensions:
            return self.reward_record.dimensions
        return None


class ExperienceBuffer:
    """
    Buffer for storing and retrieving trading experiences.

    Provides methods for:
    - Adding new experiences (initially without rewards)
    - Backfilling rewards after holding period completes
    - Retrieving top/bottom/recent experiences for ICRL prompts
    - Tracking forecast errors for the improvement reward dimension
    - Save/load for persistence across sessions
    """

    def __init__(self, max_size: int = 10000):
        self.experiences: List[Experience] = []
        self.max_size = max_size
        self._forecast_errors: List[float] = []  # Trailing forecast errors

    def add(self, experience: Experience) -> int:
        """Add an experience and return its index."""
        if len(self.experiences) >= self.max_size:
            # Remove oldest completed experience
            for i, exp in enumerate(self.experiences):
                if exp.is_completed:
                    self.experiences.pop(i)
                    break
            else:
                self.experiences.pop(0)

        self.experiences.append(experience)
        return len(self.experiences) - 1

    def backfill_reward(self, idx: int, reward_record: RewardRecord):
        """Backfill reward for a completed holding period."""
        if 0 <= idx < len(self.experiences):
            self.experiences[idx].reward_record = reward_record

            # Track forecast error for improvement dimension
            if reward_record.cumulative_return is not None:
                error = compute_forecast_error(
                    reward_record.position_weight,
                    reward_record.cumulative_return,
                )
                self._forecast_errors.append(error)

    def get_completed(self) -> List[Experience]:
        """Return all experiences with computed rewards."""
        return [e for e in self.experiences if e.is_completed]

    def get_pending(self) -> List[Experience]:
        """Return all experiences still awaiting reward computation."""
        return [e for e in self.experiences if not e.is_completed]

    def get_top_k(self, k: int = 5) -> List[Experience]:
        """Return the K best-rewarded completed experiences."""
        completed = self.get_completed()
        completed.sort(key=lambda e: e.reward or 0.0, reverse=True)
        return completed[:k]

    def get_bottom_k(self, k: int = 5) -> List[Experience]:
        """Return the K worst-rewarded completed experiences."""
        completed = self.get_completed()
        completed.sort(key=lambda e: e.reward or 0.0)
        return completed[:k]

    def get_recent_n(self, n: int = 10) -> List[Experience]:
        """Return the N most recent completed experiences."""
        completed = self.get_completed()
        return completed[-n:]

    def get_forecast_errors(self, n: int = None) -> List[float]:
        """
        Return trailing forecast errors for the improvement dimension.

        Args:
            n: Number of most recent errors to return. Default: all.

        Returns:
            List of forecast error values (most recent last).
        """
        if n is None:
            return list(self._forecast_errors)
        return list(self._forecast_errors[-n:])

    def summary_stats(self) -> Dict:
        """Summary statistics for the buffer, including per-dimension breakdowns."""
        completed = self.get_completed()
        rewards = [e.reward for e in completed if e.reward is not None]

        stats = {
            "total_experiences": len(self.experiences),
            "completed": len(completed),
            "pending": len(self.get_pending()),
            "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
            "std_reward": float(np.std(rewards)) if rewards else 0.0,
            "min_reward": float(np.min(rewards)) if rewards else 0.0,
            "max_reward": float(np.max(rewards)) if rewards else 0.0,
            "positive_pct": sum(1 for r in rewards if r > 0) / max(len(rewards), 1) * 100,
        }

        # Per-dimension means
        dimension_scores = {}
        for exp in completed:
            if exp.dimensions:
                dim_dict = exp.dimensions.to_dict()
                for dim_name, score in dim_dict.items():
                    if dim_name == "composite":
                        continue
                    if dim_name not in dimension_scores:
                        dimension_scores[dim_name] = []
                    dimension_scores[dim_name].append(score)

        if dimension_scores:
            stats["dimension_means"] = {
                name: float(np.mean(scores))
                for name, scores in dimension_scores.items()
            }

        return stats

    def save(self, path: str):
        """Save buffer to JSON file."""
        data = []
        for exp in self.experiences:
            exp_dict = {
                "step_idx": exp.step_idx,
                "date": exp.date,
                "ticker": exp.ticker,
                "signal_label": exp.signal_label,
                "action": exp.action,
                "reasoning": exp.reasoning[:500],  # Truncate reasoning
                "mode": exp.mode,
                "entry_price": exp.entry_price,
                "position_weight": exp.position_weight,
            }
            if exp.reward_record:
                rr = exp.reward_record
                rr_dict = {
                    "ticker": rr.ticker,
                    "entry_date": rr.entry_date,
                    "exit_date": rr.exit_date,
                    "action": rr.action,
                    "position_weight": rr.position_weight,
                    "entry_price": rr.entry_price,
                    "exit_price": rr.exit_price,
                    "holding_days": rr.holding_days,
                    "cumulative_return": rr.cumulative_return,
                    "holding_volatility": rr.holding_volatility,
                    "transaction_cost": rr.transaction_cost,
                    "reward": rr.reward,
                    "raw_pnl": rr.raw_pnl,
                }
                # Serialize dimensions if present
                if rr.dimensions:
                    rr_dict["dimensions"] = rr.dimensions.to_dict()
                exp_dict["reward_record"] = rr_dict
            data.append(exp_dict)

        save_data = {
            "experiences": data,
            "forecast_errors": self._forecast_errors[-1000:],  # Keep last 1000
        }

        with open(path, "w") as f:
            json.dump(save_data, f, indent=2, default=str)
        logger.info(f"Experience buffer saved: {len(data)} experiences -> {path}")

    def load(self, path: str):
        """Load buffer from JSON file."""
        with open(path, "r") as f:
            raw = json.load(f)

        # Handle both old format (list) and new format (dict with forecast_errors)
        if isinstance(raw, list):
            data = raw
            self._forecast_errors = []
        else:
            data = raw.get("experiences", [])
            self._forecast_errors = raw.get("forecast_errors", [])

        for exp_dict in data:
            reward_record = None
            if "reward_record" in exp_dict:
                rr = exp_dict["reward_record"]
                # Reconstruct dimensions if present
                dimensions = None
                if "dimensions" in rr:
                    dim_data = rr.pop("dimensions")
                    dimensions = RewardDimensions(
                        sharpe=dim_data.get("sharpe", 0.0),
                        direction=dim_data.get("direction", 0.0),
                        conviction=dim_data.get("conviction", 0.0),
                        improvement=dim_data.get("improvement", 0.0),
                        override=dim_data.get("override", 0.0),
                        risk_discipline=dim_data.get("risk_discipline", 0.0),
                        coherence=dim_data.get("coherence", 0.0),
                        regime=dim_data.get("regime", 0.0),
                        composite=dim_data.get("composite", 0.0),
                    )
                reward_record = RewardRecord(
                    ticker=rr["ticker"],
                    entry_date=rr["entry_date"],
                    exit_date=rr["exit_date"],
                    action=rr["action"],
                    position_weight=rr["position_weight"],
                    entry_price=rr["entry_price"],
                    exit_price=rr["exit_price"],
                    holding_days=rr["holding_days"],
                    cumulative_return=rr["cumulative_return"],
                    holding_volatility=rr["holding_volatility"],
                    transaction_cost=rr["transaction_cost"],
                    reward=rr["reward"],
                    raw_pnl=rr["raw_pnl"],
                    dimensions=dimensions,
                )

            exp = Experience(
                step_idx=exp_dict["step_idx"],
                date=exp_dict["date"],
                ticker=exp_dict["ticker"],
                state_snapshot={},  # Not saved to keep file size small
                state_news=None,
                signal_label=exp_dict["signal_label"],
                action=exp_dict["action"],
                reasoning=exp_dict.get("reasoning", ""),
                mode=exp_dict.get("mode", "unknown"),
                reward_record=reward_record,
                entry_price=exp_dict.get("entry_price", 0.0),
                position_weight=exp_dict.get("position_weight", 0.0),
            )
            self.experiences.append(exp)

        logger.info(f"Experience buffer loaded: {len(data)} experiences from {path}")

    def __len__(self) -> int:
        return len(self.experiences)
