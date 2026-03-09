"""
RL Configuration — Hyperparameters for In-Context RL (ICRL) and GRPO

Separated from config.py to avoid requiring RL dependencies (torch, transformers)
for basic backtest usage. Import this module only when --rl-mode is specified.
"""

# ── In-Context RL (ICRL) Settings ────────────────────────────────────────────
ICRL_TOP_K_BEST = 5              # Number of best trades to include in prompt
ICRL_BOTTOM_K_WORST = 5          # Number of worst trades to include in prompt
ICRL_RECENT_N = 10               # Number of most recent completed trades
ICRL_REASONING_TRUNCATE = 200    # Max chars of reasoning per historical trade in prompt

# ── GRPO (Group Relative Policy Optimization) Settings ───────────────────────
GRPO_MODEL_NAME = "Qwen/Qwen3-4B"               # Qwen 3 with native thinking mode
GRPO_LEARNING_RATE = 1e-5
GRPO_GROUP_SIZE = 4              # G: number of completions per prompt for GRPO
GRPO_KL_COEFF = 0.01             # Beta: KL divergence penalty coefficient
GRPO_CLIP_RANGE = 0.2            # PPO-style ratio clipping range
GRPO_MAX_SEQ_LENGTH = 2048       # Maximum sequence length for local model
GRPO_BATCH_SIZE = 4              # Prompts per training batch
GRPO_GRADIENT_ACCUMULATION = 4   # Gradient accumulation steps
GRPO_NUM_EPOCHS = 3              # Training epochs over collected experience
GRPO_WARMUP_STEPS = 50           # LR warmup steps
GRPO_MAX_NEW_TOKENS = 512        # Max tokens for model generation
GRPO_TEMPERATURE = 0.7           # Sampling temperature during GRPO group generation
GRPO_LORA_R = 16                 # LoRA rank
GRPO_LORA_ALPHA = 32             # LoRA alpha scaling
GRPO_LORA_DROPOUT = 0.05         # LoRA dropout
GRPO_SFT_EPOCHS = 1              # Supervised fine-tuning epochs before RL
GRPO_RETRAIN_INTERVAL = 20       # Retrain after this many completed trades
GRPO_CHECKPOINT_DIR = "checkpoints"

# ── Reward Function Settings ─────────────────────────────────────────────────
REWARD_VOLATILITY_FLOOR = 1e-8   # Minimum volatility to avoid division by zero
REWARD_HOLD_VALUE = 0.0          # Reward for HOLD positions (zero exposure)

# ── Multi-Dimensional Reward Weights ────────────────────────────────────────
# Each dimension is normalized to [-1, +1] then weighted.
# Composite reward = Σ(weight_i * dimension_i) — a single scalar for GRPO.
REWARD_DIMENSION_WEIGHTS = {
    "sharpe":          0.25,   # Risk-adjusted P&L (existing Sharpe-like)
    "direction":       0.15,   # Directional accuracy (+1 correct, -1 wrong)
    "conviction":      0.10,   # Position sizing vs move magnitude calibration
    "improvement":     0.10,   # Forecast error improvement over trailing average
    "override":        0.10,   # Quality of disagreement with Algorithm S1
    "risk_discipline": 0.10,   # Appropriate sizing for volatility regime
    "coherence":       0.10,   # Reasoning text alignment with realized outcomes
    "regime":          0.10,   # Action alignment with market regime (trend/mean-revert)
}

# Conviction calibration: cap the absolute return used for normalization
REWARD_RETURN_CAP = 0.05           # 5% return maps to 1.0 conviction target

# Forecast error improvement: rolling window of past errors
REWARD_IMPROVEMENT_WINDOW = 20     # Compare current error to trailing mean of last N

# Risk discipline: volatility regime percentile thresholds
REWARD_VOL_HIGH_PERCENTILE = 0.75  # Above this = high-vol regime
REWARD_VOL_LOW_PERCENTILE = 0.25   # Below this = low-vol regime

# Coherence: keywords for reasoning alignment scoring
REWARD_COHERENCE_BULLISH_KEYWORDS = [
    "uptrend", "bullish", "breakout", "support", "momentum", "growth",
    "beat", "strong earnings", "upgrade", "buy signal", "oversold",
    "recovery", "accumulation", "higher highs", "positive",
]
REWARD_COHERENCE_BEARISH_KEYWORDS = [
    "downtrend", "bearish", "breakdown", "resistance", "selling pressure",
    "miss", "weak earnings", "downgrade", "sell signal", "overbought",
    "decline", "distribution", "lower lows", "negative",
]
REWARD_COHERENCE_VOL_KEYWORDS = [
    "volatile", "volatility", "uncertainty", "risk", "whipsaw",
    "choppy", "unstable", "turbulent",
]

# Counter-argument awareness: reasoning that addresses opposing views
# Presence of these keywords signals analytical rigor and debate-quality thinking
REWARD_COHERENCE_COUNTER_KEYWORDS = [
    "however", "on the other hand", "risk is", "concern", "despite",
    "counter", "but ", "challenge", "downside", "caveat", "although",
    "risk factor", "worst case", "bear case", "bull case",
    "alternatively", "if wrong", "could fail", "what if",
    "stress test", "devil's advocate", "contrarian",
]
# Bonus/penalty scaling for counter-argument presence
REWARD_COHERENCE_COUNTER_BONUS = 0.3   # Max bonus for addressing counter-arguments
REWARD_COHERENCE_COUNTER_PENALTY = 0.2 # Penalty for no counter-arguments on losing trades

# ── Experience Buffer Settings ───────────────────────────────────────────────
EXPERIENCE_BUFFER_MAX_SIZE = 10000
EXPERIENCE_SAVE_INTERVAL = 50    # Save buffer to disk every N experiences
