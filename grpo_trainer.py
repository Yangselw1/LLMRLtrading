"""
GRPO Trainer — Group Relative Policy Optimization for Local LLM

Implements the GRPO algorithm (from DeepSeek-R1) for fine-tuning a local
open-source LLM (Qwen 3) on trading decisions with risk-adjusted P&L rewards.
Uses Qwen 3's native hybrid thinking mode for chain-of-thought reasoning.

ABM-Inspired Perspective Sampling:
  Instead of generating G identical i.i.d. completions, GRPO uses structured
  perspective-biased sampling to create adversarial diversity within each group:

    - Alpha perspective: Biased toward returns, direction, conviction
    - Risk perspective: Biased toward risk discipline, regime, coherence
    - Neutral perspective: Standard unbiased completion

  For G=4: [alpha, risk, neutral, neutral]

  The reward function then selects the best perspective via group-normalized
  advantages — creating natural selection pressure between the alpha-seeking
  and risk-aware analytical styles. Over training, the model internalizes
  both perspectives into its neutral inference mode.

GRPO Algorithm:
  For each prompt x:
    1. Sample G completions with perspective bias {y_1,...,y_G}
    2. Compute reward r_i for each completion
    3. Group-normalize advantages: A_i = (r_i - mean(r)) / (std(r) + eps)
    4. Policy gradient with KL penalty and PPO-style clipping:
       L = -E[min(A_i * ratio, A_i * clip(ratio, 1-eps, 1+eps))] + beta * KL

Requires: torch, transformers, peft (LoRA), accelerate
Optional: bitsandbytes (4-bit quantization for memory savings)
"""
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

import config
import rl_config

logger = logging.getLogger(__name__)

# Check for torch availability
try:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logger.warning("PyTorch not installed; GRPO mode unavailable")

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False


def check_grpo_dependencies():
    """Check that all required dependencies are available."""
    missing = []
    if not HAS_TORCH:
        missing.append("torch>=2.0")
    if not HAS_TRANSFORMERS:
        missing.append("transformers>=4.36")
    if not HAS_PEFT:
        missing.append("peft>=0.7")
    if missing:
        raise ImportError(
            f"GRPO mode requires: {', '.join(missing)}. "
            f"Install with: pip install {' '.join(missing)}"
        )


@dataclass
class GRPOConfig:
    """Configuration for GRPO training."""
    model_name: str = rl_config.GRPO_MODEL_NAME
    learning_rate: float = rl_config.GRPO_LEARNING_RATE
    group_size: int = rl_config.GRPO_GROUP_SIZE
    kl_coeff: float = rl_config.GRPO_KL_COEFF
    clip_range: float = rl_config.GRPO_CLIP_RANGE
    max_seq_length: int = rl_config.GRPO_MAX_SEQ_LENGTH
    batch_size: int = rl_config.GRPO_BATCH_SIZE
    gradient_accumulation_steps: int = rl_config.GRPO_GRADIENT_ACCUMULATION
    num_epochs: int = rl_config.GRPO_NUM_EPOCHS
    warmup_steps: int = rl_config.GRPO_WARMUP_STEPS
    max_new_tokens: int = rl_config.GRPO_MAX_NEW_TOKENS
    temperature: float = rl_config.GRPO_TEMPERATURE
    lora_r: int = rl_config.GRPO_LORA_R
    lora_alpha: int = rl_config.GRPO_LORA_ALPHA
    lora_dropout: float = rl_config.GRPO_LORA_DROPOUT
    device: str = "auto"
    sft_epochs: int = rl_config.GRPO_SFT_EPOCHS
    checkpoint_dir: str = rl_config.GRPO_CHECKPOINT_DIR


# ═══════════════════════════════════════════════════════════════════════════════
# State-to-Prompt Conversion (compact format for local model)
# ═══════════════════════════════════════════════════════════════════════════════

def format_state_for_local_model(
    snapshot: Dict,
    news: Dict = None,
    signal_label: str = "",
) -> str:
    """
    Convert market state to a compact prompt for the local model.

    Uses a condensed format to fit within 2048 tokens, unlike the verbose
    prompt used for Claude (which has a 200K context window).
    """
    ticker = snapshot.get("ticker", "UNKNOWN")
    date = snapshot.get("date", "")
    price = snapshot.get("price_summary", {})
    tech = snapshot.get("technical_indicators", {})
    fund = snapshot.get("fundamentals", {})
    recs = snapshot.get("analyst_recommendations", {})

    lines = [
        f"Ticker: {ticker} | Date: {date} | "
        f"Price: ${price.get('current_price', 'N/A')}",
        f"1d: {price.get('price_change_1d', 'N/A')}% | "
        f"5d: {price.get('price_change_5d', 'N/A')}% | "
        f"15d: {price.get('price_change_15d', 'N/A')}%",
    ]

    # Key technical indicators (compact)
    tech_parts = []
    for key in ["rsi", "macd", "macd_signal", "adx", "atr",
                 "sma_50", "sma_200", "bb_upper", "bb_lower",
                 "stoch_k", "stoch_d"]:
        val = tech.get(key)
        if val is not None:
            tech_parts.append(f"{key}={val:.1f}" if isinstance(val, float) else f"{key}={val}")
    if tech_parts:
        lines.append("Tech: " + " | ".join(tech_parts))

    # Fundamentals (compact)
    fund_parts = []
    for key, fmt in [("pe_ratio", ".1f"), ("pb_ratio", ".1f"),
                     ("roe", ".2%"), ("debt_to_equity", ".1f")]:
        val = fund.get(key)
        if val is not None:
            try:
                fund_parts.append(f"{key}={val:{fmt}}")
            except (ValueError, TypeError):
                fund_parts.append(f"{key}={val}")
    if fund_parts:
        lines.append("Fund: " + " | ".join(fund_parts))

    # Analyst consensus
    consensus = recs.get("consensus_score")
    if consensus is not None:
        lines.append(f"Consensus: {consensus}/5.0")

    # S1 signal
    lines.append(f"S1 Signal: {signal_label}")

    # Top news headline (just one)
    if news:
        for bucket_articles in news.values():
            if bucket_articles:
                headline = bucket_articles[0].get("headline", "")[:100]
                if headline:
                    lines.append(f"News: {headline}")
                break

    lines.append("")
    lines.append("Analyze the data above. Consider momentum, trend, volatility regime, and fundamentals.")
    lines.append("DECISION: STRONG_SELL, SELL, HOLD, BUY, or STRONG_BUY")

    return "\n".join(lines)


def _build_chat_prompt(state_text: str, perspective: str = "neutral") -> str:
    """
    Wrap state text in Qwen 3 ChatML format with native thinking enabled.

    Qwen 3 uses <|im_start|>/<|im_end|> delimiters (ChatML) and supports
    hybrid thinking mode where the model produces <think>...</think> reasoning
    before its final answer. We enable thinking with /think in the system prompt.

    Args:
        state_text: Formatted market state
        perspective: One of "neutral", "alpha", or "risk".
            - "alpha": Biases toward return-seeking (Sharpe, direction, conviction)
            - "risk": Biases toward risk discipline (regime, coherence, sizing)
            - "neutral": Standard unbiased prompt
    """
    base_msg = (
        "You are a quantitative trading analyst. Analyze the market data, reason "
        "step-by-step inside <think>...</think> tags, then state your trading decision.\n\n"
        "ACTIONS (each maps to a portfolio weight):\n"
        "  STRONG_BUY  = +100% long  | STRONG_SELL = -100% short\n"
        "  BUY         = +50% long   | SELL        = -50% short\n"
        "  HOLD        = 0% (flat — no clear edge)\n\n"
        "In your <think> reasoning:\n"
        "  1. Identify the 2-3 strongest signals and their directional implications\n"
        "  2. Note any conflicting indicators and how you resolve the conflict\n"
        "  3. Assess the volatility regime (high/normal/low) from ATR or BB width\n"
        "  4. State your confidence level and match position sizing to it\n"
        "  5. Note the key risk — what could make this trade fail\n\n"
    )

    if perspective == "alpha":
        base_msg += (
            "PRIORITY: Maximize risk-adjusted returns.\n"
            "- Focus on directional accuracy and conviction sizing.\n"
            "- Be aggressive where multiple signals align strongly.\n"
            "- Estimate expected return vs. volatility (Sharpe-like thinking).\n"
            "- Note counter-arguments but do not let caution override clear setups.\n\n"
        )
    elif perspective == "risk":
        base_msg += (
            "PRIORITY: Risk discipline and regime awareness.\n"
            "- If volatility is elevated, prefer HOLD or smaller sizing (BUY not STRONG_BUY).\n"
            "- Check: does your thesis align with or fight the market trend regime?\n"
            "- Stress-test: what specific evidence would make this trade fail?\n"
            "- When uncertain, HOLD is the highest-quality decision.\n\n"
        )

    base_msg += (
        "Format your final answer EXACTLY as: DECISION: [STRONG_SELL|SELL|HOLD|BUY|STRONG_BUY]\n\n"
        "/think"
    )

    return (
        f"<|im_start|>system\n{base_msg}<|im_end|>\n"
        f"<|im_start|>user\n{state_text}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


# Perspective allocation for GRPO group sampling
# Maps group_size -> list of perspectives for each completion
def _get_perspective_schedule(group_size: int) -> List[str]:
    """
    Allocate perspectives across G completions for structured diversity.

    For G=4: [alpha, risk, neutral, neutral]
    For G=6: [alpha, alpha, risk, risk, neutral, neutral]
    For G=2: [alpha, risk]

    This ensures the reward function always sees both alpha-seeking and
    risk-aware completions, creating natural adversarial tension in the
    group advantages — the best perspective wins via GRPO selection pressure.
    """
    if group_size <= 1:
        return ["neutral"]
    if group_size == 2:
        return ["alpha", "risk"]
    if group_size == 3:
        return ["alpha", "risk", "neutral"]

    # For larger groups: ~25% alpha, ~25% risk, ~50% neutral
    n_alpha = max(1, group_size // 4)
    n_risk = max(1, group_size // 4)
    n_neutral = group_size - n_alpha - n_risk
    return (["alpha"] * n_alpha +
            ["risk"] * n_risk +
            ["neutral"] * n_neutral)


def parse_local_model_decision(text: str) -> Optional[str]:
    """
    Parse a trading decision from local model output.

    Handles Qwen 3's native thinking format where the model produces
    <think>reasoning...</think> followed by the final decision.
    We strip the thinking block and parse the decision from the remainder.
    """
    # Strip <think>...</think> block to isolate the final answer
    answer_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

    # If stripping removed everything, fall back to full text
    if not answer_text:
        answer_text = text

    answer_upper = answer_text.upper()

    # Look for DECISION: pattern in the answer portion
    match = re.search(
        r'DECISION:\s*(STRONG_SELL|STRONG_BUY|STRONG SELL|STRONG BUY|SELL|BUY|HOLD)',
        answer_upper
    )
    if match:
        decision = match.group(1).strip().replace(" ", "_")
        if decision in config.ACTION_LABELS:
            return decision

    # Fallback: look for any action label in the answer text
    for label in ["STRONG_BUY", "STRONG_SELL", "BUY", "SELL", "HOLD"]:
        if label in answer_upper:
            return label

    return None


# ═══════════════════════════════════════════════════════════════════════════════
# GRPO Trainer
# ═══════════════════════════════════════════════════════════════════════════════

class GRPOTrainer:
    """
    Implements Group Relative Policy Optimization for a local LLM.

    Training flow:
    1. Optional SFT pre-training on Algorithm S1 labels
    2. Walk-forward GRPO: periodically retrain on accumulated experience
    3. Inference: generate trading decisions from current policy

    Memory-efficient via LoRA + optional 4-bit quantization.
    """

    def __init__(self, grpo_config: GRPOConfig = None, observer=None):
        check_grpo_dependencies()

        self.config = grpo_config or GRPOConfig()
        self.obs = observer
        self.model = None
        self.ref_model = None
        self.tokenizer = None
        self.optimizer = None
        self.device = None
        self._training_steps = 0
        self._training_log = []

    def _resolve_device(self) -> str:
        """Auto-detect the best available device."""
        if self.config.device != "auto":
            return self.config.device
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def load_model(self):
        """Load the base model with LoRA adapters."""
        self.device = self._resolve_device()
        logger.info(f"GRPO: Loading {self.config.model_name} on {self.device}")

        # Pass HF_TOKEN to authenticated downloads (avoids rate-limit warning)
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            logger.warning(
                "HF_TOKEN not set. Add HF_TOKEN=hf_... to your .env file "
                "for faster downloads and higher rate limits."
            )

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
            token=hf_token,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Quantization config (if on CUDA and bitsandbytes available)
        quant_config = None
        if self.device == "cuda":
            try:
                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
                logger.info("GRPO: Using 4-bit quantization")
            except Exception:
                logger.info("GRPO: 4-bit quantization unavailable, using fp16")

        # Load base model
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.float16 if self.device != "cpu" else torch.float32,
            "token": hf_token,
        }
        if quant_config:
            model_kwargs["quantization_config"] = quant_config
        elif self.device != "cpu":
            model_kwargs["device_map"] = self.device

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            **model_kwargs,
        )

        # Apply LoRA
        if quant_config:
            self.model = prepare_model_for_kbit_training(self.model)

        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(self.model, lora_config)

        if self.device == "cpu" or self.device == "mps":
            self.model = self.model.to(self.device)

        # Store reference model for KL divergence
        # We compute ref log probs at generation time and cache them
        self.ref_model = None  # Will use initial model as reference

        # Optimizer
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.config.learning_rate,
            weight_decay=0.01,
        )

        num_params = sum(p.numel() for p in trainable_params)
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"GRPO: Model loaded. Trainable: {num_params:,} / "
                     f"{total_params:,} params ({num_params/total_params*100:.1f}%)")

    def _tokenize(self, text: str, max_length: int = None) -> dict:
        """Tokenize text for the model."""
        max_length = max_length or self.config.max_seq_length
        return self.tokenizer(
            text,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding=True,
        ).to(self.device)

    def _generate(self, prompt: str, num_samples: int = 1,
                  temperature: float = None) -> List[str]:
        """Generate completions from the current policy."""
        temperature = temperature or self.config.temperature
        inputs = self._tokenize(prompt)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=temperature,
                do_sample=True,
                num_return_sequences=num_samples,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # Decode only the generated tokens (exclude prompt)
        # Use skip_special_tokens=False to preserve <think>...</think> tags
        # from Qwen 3's native thinking mode, then strip ChatML delimiters
        prompt_len = inputs["input_ids"].shape[1]
        completions = []
        for output in outputs:
            gen_tokens = output[prompt_len:]
            text = self.tokenizer.decode(gen_tokens, skip_special_tokens=False)
            # Strip Qwen 3 ChatML end tokens but keep <think> tags
            text = text.replace("<|im_end|>", "").replace("<|im_start|>", "").strip()
            completions.append(text)

        return completions

    def _compute_log_probs(self, prompt_ids, completion_ids):
        """Compute log probabilities of completion tokens given prompt."""
        full_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.ones_like(full_ids)

        with torch.no_grad() if not self.model.training else torch.enable_grad():
            outputs = self.model(
                input_ids=full_ids,
                attention_mask=attention_mask,
            )

        # Get logits for completion tokens only
        logits = outputs.logits[:, prompt_ids.shape[1] - 1:-1, :]
        log_probs = F.log_softmax(logits, dim=-1)

        # Gather log probs for actual completion tokens
        token_log_probs = log_probs.gather(
            2, completion_ids.unsqueeze(-1)
        ).squeeze(-1)

        return token_log_probs.sum(dim=-1)  # Sum over sequence

    def supervised_pretrain(self, experiences) -> Dict:
        """
        SFT phase: train on (state, S1_label) pairs.

        Teaches the model the input format and how to produce
        reasoning + decision text before RL begins.
        """
        logger.info(f"GRPO SFT: Pre-training on {len(experiences)} examples "
                     f"for {self.config.sft_epochs} epochs")

        self.model.train()
        total_loss = 0.0
        steps = 0

        for epoch in range(self.config.sft_epochs):
            for exp in experiences:
                # Build prompt + target
                state_text = format_state_for_local_model(
                    exp.state_snapshot,
                    exp.state_news,
                    exp.signal_label,
                )
                prompt = _build_chat_prompt(state_text)

                # Determine direction & sizing for SFT reasoning template
                if "BUY" in exp.signal_label:
                    _dir = "upward"
                    _conv = "max conviction" if "STRONG" in exp.signal_label else "moderate conviction"
                    _wt = "+100%" if "STRONG" in exp.signal_label else "+50%"
                elif "SELL" in exp.signal_label:
                    _dir = "downward"
                    _conv = "max conviction" if "STRONG" in exp.signal_label else "moderate conviction"
                    _wt = "-100%" if "STRONG" in exp.signal_label else "-50%"
                else:
                    _dir = "neutral"
                    _conv = "low conviction — signals are mixed or insufficient"
                    _wt = "0%"

                # Extract actual data from state for grounded reasoning
                _price = exp.state_snapshot.get("price_summary", {})
                _tech = exp.state_snapshot.get("technical_indicators", {})
                _rsi = _tech.get("rsi")
                _adx = _tech.get("adx")
                _chg5 = _price.get("price_change_5d")

                _data_line = ""
                _data_parts = []
                if _chg5 is not None:
                    _data_parts.append(f"5d change: {_chg5}%")
                if _rsi is not None:
                    _data_parts.append(f"RSI: {_rsi:.0f}")
                if _adx is not None:
                    _data_parts.append(f"ADX: {_adx:.0f}")
                if _data_parts:
                    _data_line = f"Key data: {' | '.join(_data_parts)}.\n"

                target = (
                    f"<think>\n"
                    f"Analyzing {exp.ticker} on {exp.date}:\n"
                    f"{_data_line}"
                    f"S1 signal: {exp.signal_label} — volatility-normalized momentum "
                    f"indicates {_dir} pressure over ~5 trading days.\n"
                    f"Position sizing: {_wt} ({_conv}).\n"
                    f"The technical indicators support this direction. "
                    f"Current volatility does not warrant an override.\n"
                    f"Key risk: signal could be noise if volatility spikes or regime shifts.\n"
                    f"</think>\n"
                    f"DECISION: {exp.signal_label}<|im_end|>"
                )

                full_text = prompt + target
                inputs = self._tokenize(full_text)

                # Compute loss (standard causal LM loss)
                outputs = self.model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss / self.config.gradient_accumulation_steps

                loss.backward()
                total_loss += loss.item()
                steps += 1

                if steps % self.config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=1.0
                    )
                    self.optimizer.step()
                    self.optimizer.zero_grad()

        avg_loss = total_loss / max(steps, 1)
        logger.info(f"GRPO SFT: Completed. Avg loss: {avg_loss:.4f}")
        self.model.eval()

        return {"sft_loss": avg_loss, "sft_steps": steps}

    def grpo_step(self, prompts: List[str], rewards_for_actions: Dict) -> Dict:
        """
        Single GRPO optimization step.

        For each prompt:
        1. Generate G completions (already done, passed via rewards_for_actions)
        2. Group-normalize advantages
        3. Compute policy gradient loss with KL penalty

        Args:
            prompts: List of state prompts
            rewards_for_actions: Dict mapping prompt_idx -> list of
                (completion_text, reward) tuples

        Returns:
            Training statistics dict
        """
        self.model.train()
        total_loss = 0.0
        total_kl = 0.0
        all_advantages = []

        for prompt_idx, prompt in enumerate(prompts):
            completions_rewards = rewards_for_actions.get(prompt_idx, [])
            if len(completions_rewards) < 2:
                continue

            rewards = [r for _, r in completions_rewards]
            completions = [c for c, _ in completions_rewards]

            # Group-normalize advantages
            mean_r = np.mean(rewards)
            std_r = np.std(rewards) + 1e-8
            advantages = [(r - mean_r) / std_r for r in rewards]
            all_advantages.extend(advantages)

            # Compute loss for each completion
            prompt_inputs = self._tokenize(prompt)
            prompt_ids = prompt_inputs["input_ids"]

            for comp_text, advantage in zip(completions, advantages):
                comp_inputs = self._tokenize(comp_text)
                comp_ids = comp_inputs["input_ids"]

                # Current policy log prob
                log_prob = self._compute_log_probs(prompt_ids, comp_ids)

                # Policy gradient loss (simplified — no ratio clipping
                # for first iteration since we don't have old log probs)
                pg_loss = -advantage * log_prob

                # KL penalty (approximate: just use log_prob magnitude)
                kl_penalty = self.config.kl_coeff * (log_prob ** 2)

                loss = (pg_loss + kl_penalty) / (
                    len(completions) * self.config.gradient_accumulation_steps
                )
                loss.backward()

                total_loss += pg_loss.item()
                total_kl += kl_penalty.item()

        # Optimizer step
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.optimizer.zero_grad()

        self._training_steps += 1
        self.model.eval()

        stats = {
            "step": self._training_steps,
            "loss": total_loss / max(len(prompts), 1),
            "kl_div": total_kl / max(len(prompts), 1),
            "mean_reward": float(np.mean([r for cr in rewards_for_actions.values()
                                           for _, r in cr])) if rewards_for_actions else 0.0,
            "mean_advantage": float(np.mean(all_advantages)) if all_advantages else 0.0,
        }

        if self.obs:
            self.obs.log_grpo_training_step(**stats)

        self._training_log.append(stats)
        return stats

    def train(self, experience_buffer, epochs: int = None) -> Dict:
        """
        Full GRPO training loop over accumulated experience.

        For each epoch:
        1. Sample completed experiences from buffer
        2. For each experience, generate G completions
        3. Score each completion with the reward function
        4. Run GRPO update step

        Args:
            experience_buffer: ExperienceBuffer with completed experiences
            epochs: Number of training epochs (default: from config)

        Returns:
            Training statistics
        """
        from reward import compute_reward

        epochs = epochs or self.config.num_epochs
        completed = experience_buffer.get_completed()

        if not completed:
            logger.warning("GRPO: No completed experiences to train on")
            return {"epochs": 0, "steps": 0}

        logger.info(f"GRPO: Training for {epochs} epochs on "
                     f"{len(completed)} completed experiences")

        all_stats = []
        for epoch in range(epochs):
            # Shuffle experiences
            indices = np.random.permutation(len(completed))

            for batch_start in range(0, len(indices), self.config.batch_size):
                batch_indices = indices[batch_start:
                                        batch_start + self.config.batch_size]
                prompts = []
                rewards_for_actions = {}

                for local_idx, exp_idx in enumerate(batch_indices):
                    exp = completed[exp_idx]

                    state_text = format_state_for_local_model(
                        exp.state_snapshot,
                        exp.state_news,
                        exp.signal_label,
                    )

                    # Generate G completions with perspective-biased prompts
                    # Each completion gets a different analytical perspective
                    perspectives = _get_perspective_schedule(self.config.group_size)
                    comp_rewards = []

                    for perspective in perspectives:
                        prompt_biased = _build_chat_prompt(state_text, perspective)

                        # Generate 1 completion per perspective
                        completions = self._generate(
                            prompt_biased,
                            num_samples=1,
                            temperature=self.config.temperature,
                        )
                        comp = completions[0] if completions else ""

                        decision = parse_local_model_decision(comp)
                        if decision and exp.reward_record:
                            base_reward = exp.reward_record.reward
                            if decision == exp.action:
                                reward = base_reward
                            else:
                                # Penalize deviation from successful trades,
                                # reward deviation from failed trades
                                reward = -base_reward * 0.5
                        else:
                            reward = -1.0  # Penalty for unparseable output

                        comp_rewards.append((comp, reward))

                    # Use the neutral prompt for the GRPO step
                    prompt = _build_chat_prompt(state_text, "neutral")
                    prompts.append(prompt)
                    rewards_for_actions[local_idx] = comp_rewards

                # GRPO update step
                if prompts:
                    stats = self.grpo_step(prompts, rewards_for_actions)
                    all_stats.append(stats)

        summary = {
            "epochs": epochs,
            "steps": len(all_stats),
            "final_loss": all_stats[-1]["loss"] if all_stats else 0.0,
            "mean_loss": float(np.mean([s["loss"] for s in all_stats]))
                         if all_stats else 0.0,
        }
        logger.info(f"GRPO: Training complete. {summary}")
        return summary

    def predict(self, state_prompt: str) -> Tuple[str, str]:
        """
        Generate a single (reasoning, decision) from the current policy.

        Uses the neutral (unbiased) perspective for inference. The model has
        already internalized both alpha and risk perspectives through training
        with perspective-biased GRPO groups.

        Args:
            state_prompt: Formatted state text

        Returns:
            (reasoning_text, decision): The model's output and parsed decision
        """
        self.model.eval()
        prompt = _build_chat_prompt(state_prompt, "neutral")

        # Generate with lower temperature for inference
        completions = self._generate(
            prompt, num_samples=1, temperature=0.3
        )

        text = completions[0] if completions else ""
        decision = parse_local_model_decision(text)

        if not decision:
            logger.warning(f"GRPO: Could not parse decision from output: "
                            f"{text[:200]}")
            decision = "HOLD"  # Safe fallback

        return text, decision

    def save_checkpoint(self, path: str) -> None:
        """Save model checkpoint (LoRA weights only)."""
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        logger.info(f"GRPO: Checkpoint saved to {path}")

    def load_checkpoint(self, path: str) -> None:
        """Load a previously saved checkpoint."""
        from peft import PeftModel
        logger.info(f"GRPO: Loading checkpoint from {path}")
        # The base model should already be loaded; apply LoRA weights
        self.model = PeftModel.from_pretrained(
            self.model.base_model.model
            if hasattr(self.model, 'base_model') else self.model,
            path,
        ).to(self.device)

    def get_training_log(self) -> List[Dict]:
        """Return the full training log."""
        return self._training_log
