# Trading-R1 Backtest

A complete backtesting system implementing the methodology from **"Trading-R1: Financial Trading with LLM Reasoning via Reinforcement Learning"** (Xiao et al., 2025).

Includes two RL approaches for learning from trading outcomes:
- **ICRL** (In-Context RL) — simulated RL via Claude's context window
- **GRPO** (Group Relative Policy Optimization) — true RL with a local Qwen 3 model

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure API keys
cp .env.example .env
# Edit .env with your ANTHROPIC_API_KEY and/or FINNHUB_API_KEY
# For GRPO mode, also add HF_TOKEN for Hugging Face model downloads

# 3. Signal-only mode (no API cost)
python run_backtest.py --tickers AAPL MSFT NVDA --signal-only

# 4. LLM mode (Claude)
python run_backtest.py --tickers AAPL

# 5. In-Context RL (Claude learns from past trades)
python run_backtest.py --tickers AAPL --rl-mode icrl

# 6. GRPO (local Qwen 3 model with true RL)
python run_backtest.py --tickers AAPL --rl-mode grpo --grpo-sft-first

# 7. Compare all 4 modes side-by-side
python run_backtest.py --tickers AAPL MSFT --compare-all
```

## Web Dashboard

An interactive Streamlit dashboard is available as an alternative to the CLI.

```bash
# Install Streamlit (already in requirements.txt)
pip install -r requirements.txt

# Launch the dashboard
streamlit run dashboard.py
```

Then open **http://localhost:8501** in your browser.

**Dashboard features:**
- **Sidebar** — select tickers, mode (Signal-Only / LLM / ICRL / GRPO), date range, capital, costs, and RL-specific params
- **Metrics tab** — performance summary table + KPI cards (CR, SR, HR, MDD) + RL buffer stats
- **Equity Curves tab** — per-ticker equity curves and drawdown charts
- **Signals tab** — signal distribution pie charts + full signal history
- **RL Rewards tab** (ICRL/GRPO only) — reward evolution, dimension radar chart, and per-dimension score trajectories

## Architecture

| File | Purpose |
|------|---------|
| `config.py` | All parameters matching the paper (Algorithm S1, quantiles, weights) |
| `rl_config.py` | RL hyperparameters (ICRL, GRPO, reward, experience buffer) |
| `data_collector.py` | Yahoo Finance (OHLCV, fundamentals), Finnhub/RSS (news), 20+ technical indicators |
| `signal_generator.py` | Algorithm S1: multi-horizon volatility-normalized signals -> 5-class labels |
| `llm_analyst.py` | Claude API integration for investment thesis generation + rule-based fallback |
| `icrl_analyst.py` | In-Context RL analyst: augments Claude prompts with past (state, action, reward) history |
| `grpo_trainer.py` | GRPO training loop: LoRA fine-tuning of Qwen 3 with group-normalized advantages |
| `grpo_analyst.py` | Inference wrapper around the GRPO-trained local model |
| `reward.py` | Risk-adjusted P&L reward function (Sharpe-like: return / volatility) |
| `experience_buffer.py` | Anti-lookahead experience storage with top-K/bottom-K/recent-N retrieval |
| `rl_backtester.py` | Walk-forward RL backtest engine with reward backfilling |
| `backtester.py` | Standard portfolio simulation: position sizing, weekly rebalancing, transaction costs |
| `visualizer.py` | Equity curves, drawdowns, signal distributions, Sharpe heatmap |
| `rl_visualizer.py` | Mode comparison charts, reward evolution, GRPO training curves |
| `observer.py` | Configurable logging (5 verbosity levels), input storage, audit trail |
| `run_backtest.py` | CLI entry point orchestrating all modes |
| `dashboard.py` | Streamlit web dashboard (interactive alternative to CLI) |

## RL Modes

### In-Context RL (ICRL)

Uses Claude's large context window to simulate RL. Past trading outcomes (best, worst, and recent trades with their rewards) are fed back into the prompt, so Claude can learn from its own history within a single session.

```bash
python run_backtest.py --tickers AAPL --rl-mode icrl --save-experience
```

Key flags: `--icrl-top-k 5`, `--icrl-bottom-k 5`, `--icrl-recent-n 10`

### GRPO (Group Relative Policy Optimization)

True RL with weight updates on a local open-source model (Qwen 3). For each prompt, generates G completions, scores them with the reward function, group-normalizes advantages, and applies a policy gradient update with KL penalty.

```bash
# Install RL dependencies first
pip install torch transformers peft accelerate bitsandbytes

# Run with supervised pre-training then RL
python run_backtest.py --tickers AAPL --rl-mode grpo --grpo-sft-first --save-experience
```

Key flags: `--grpo-model Qwen/Qwen3-4B`, `--grpo-sft-first`, `--grpo-retrain-interval 20`, `--grpo-checkpoint path/`

### Compare All Modes

Runs all 4 modes (signal-only, LLM, ICRL, GRPO) and generates comparison charts:

```bash
python run_backtest.py --tickers AAPL MSFT --compare-all
```

## Key Parameters (from the paper)

- **EMA span**: 3 | **Horizons**: {3, 7, 15} days | **Weights**: {0.3, 0.5, 0.2}
- **Volatility window**: 20 periods | **Quantile thresholds**: {0.03, 0.15, 0.53, 0.85}
- **Action space**: Strong Sell (-1.0), Sell (-0.5), Hold (0.0), Buy (+0.5), Strong Buy (+1.0)
- **Rebalancing**: Every 5 trading days | **TX cost**: 10 bps | **Risk-free rate**: 4%

## Reward Function

Risk-adjusted P&L (Sharpe-like) computed per trade:

```
reward = (position_weight * cumulative_return - tx_cost - rf_daily * days) / holding_volatility
```

Anti-lookahead: rewards are only backfilled after the holding period completes.

## Evaluation Metrics (Appendix S2)

- **CR** -- Cumulative Return
- **SR** -- Sharpe Ratio (annualized)
- **HR** -- Hit Rate (directional accuracy)
- **MDD** -- Maximum Drawdown

## CLI Reference

```
Standard options:
  --tickers AAPL MSFT       Ticker symbols to backtest
  --all-paper-tickers       Use all 14 paper tickers
  --signal-only             Rule-based mode (no API cost)
  --start-date 2024-06-01   Backtest start date
  --end-date 2024-08-31     Backtest end date
  --initial-capital 100000  Starting capital
  --tx-cost 10              Transaction cost in basis points
  --rebalance-days 5        Rebalance frequency
  --no-charts               Skip chart generation
  --output-dir results/     Custom output directory

RL options:
  --rl-mode {icrl,grpo}     RL mode selection
  --compare-all             Run all 4 modes with comparison
  --icrl-top-k 5            Best trades in ICRL prompt
  --icrl-bottom-k 5         Worst trades in ICRL prompt
  --icrl-recent-n 10        Recent trades in ICRL prompt
  --grpo-model MODEL        HuggingFace model (default: Qwen/Qwen3-4B)
  --grpo-sft-first          Pre-train on S1 labels before RL
  --grpo-checkpoint PATH    Load pre-trained LoRA weights
  --grpo-retrain-interval N Retrain every N completed trades
  --load-experience PATH    Resume from saved experience buffer
  --save-experience         Save experience buffer after run

Logging:
  -v {0,1,2,3,4}            Verbosity (0=silent, 2=normal, 4=debug)
  --store-inputs            Save all prompts/snapshots/responses
  --no-color                Disable colored output
```

## Environment Variables

| Variable | Required For | Description |
|----------|-------------|-------------|
| `ANTHROPIC_API_KEY` | LLM, ICRL modes | Claude API key |
| `FINNHUB_API_KEY` | News data | Finnhub API key (optional) |
| `HF_TOKEN` | GRPO mode | Hugging Face token for model downloads |
