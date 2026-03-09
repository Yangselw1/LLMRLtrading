"""
GRPO Analyst — Thin Wrapper Around GRPOTrainer for Inference

Provides the same analyze() interface as LLMAnalyst and ICRLAnalyst,
making it interchangeable in the backtest pipeline.
"""
import logging
from typing import Dict, Optional, Tuple

import config
from llm_analyst import rule_based_decision

logger = logging.getLogger(__name__)


class GRPOAnalyst:
    """
    Trading analyst powered by a GRPO-trained local model.

    Uses GRPOTrainer.predict() to generate decisions, with a
    rule-based fallback if the model fails to produce a valid decision.
    """

    def __init__(self, trainer, observer=None):
        """
        Args:
            trainer: GRPOTrainer instance (already loaded)
            observer: Observer for logging
        """
        self.trainer = trainer
        self.obs = observer

    def analyze(
        self,
        snapshot: Dict,
        news: Dict = None,
        signal_label: str = None,
    ) -> Tuple[str, str]:
        """
        Generate a trading decision from the local GRPO model.

        Args:
            snapshot: Market data snapshot
            news: News data
            signal_label: Algorithm S1 label

        Returns:
            (reasoning_text, decision)
        """
        from grpo_trainer import format_state_for_local_model

        ticker = snapshot.get("ticker", "UNKNOWN")
        date = snapshot.get("date", "")

        try:
            state_text = format_state_for_local_model(
                snapshot, news, signal_label or ""
            )
            reasoning, decision = self.trainer.predict(state_text)

            if decision and decision in config.ACTION_LABELS:
                if self.obs:
                    self.obs._print(
                        f"    GRPO Decision: {decision} ({ticker} {date})",
                        2  # NORMAL
                    )
                return reasoning, decision

        except Exception as e:
            logger.warning(f"GRPO prediction failed for {ticker} {date}: {e}")

        # Fallback to rule-based
        logger.info(f"GRPO: Falling back to rule-based for {ticker} {date}")
        thesis, decision = rule_based_decision(snapshot, signal_label)
        return thesis, decision
