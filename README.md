# Trading-R1 Backtest

A complete backtesting system implementing the methodology from **"Trading-R1: Financial Trading with LLM Reasoning via Reinforcement Learning"** (Xiao et al., 2025).

## Quick Start

```bash
# 1. Install dependencies (works with Python 3.8+)
pip install -r requirements.txt

# 2. Configure API keys (optional)
cp .env.example .env
# Edit .env with your ANTHROPIC_API_KEY and/or FINNHUB_API_KEY

# 3. Run (signal-only mode — no API cost)
python run_backtest.py --tickers AAPL MSFT NVDA --signal-only

# 4. Run with Claude LLM analysis
python run_backtest.py --tickers AAPL --start-date 2024-06-01 --end-date 2024-08-31

# 5. All 14 paper tickers
python run_backtest.py --all-paper-tickers --signal-only
```

## Architecture

| File | Purpose |
|------|---------|
| `config.py` | All parameters matching the paper (Algorithm S1, quantiles, weights) |
| `data_collector.py` | Yahoo Finance (OHLCV, fundamentals), Finnhub/RSS (news), 20+ technical indicators |
| `signal_generator.py` | Algorithm S1: multi-horizon volatility-normalized signals → 5-class labels |
| `llm_analyst.py` | Claude API integration for investment thesis generation + rule-based fallback |
| `backtester.py` | Portfolio simulation: position sizing, weekly rebalancing, transaction costs |
| `visualizer.py` | Equity curves, drawdowns, signal distributions, Sharpe heatmap |
| `run_backtest.py` | CLI entry point orchestrating the full pipeline |

## Key Parameters (from the paper)

- **EMA span**: 3 | **Horizons**: {3, 7, 15} days | **Weights**: {0.3, 0.5, 0.2}
- **Volatility window**: 20 periods | **Quantile thresholds**: {0.03, 0.15, 0.53, 0.85}
- **Action space**: Strong Sell (-1.0), Sell (-0.5), Hold (0.0), Buy (+0.5), Strong Buy (+1.0)
- **Rebalancing**: Every 5 trading days | **TX cost**: 10 bps | **Risk-free rate**: 4%

## Evaluation Metrics (Appendix S2)

- **CR** — Cumulative Return: Π(1 + r_t) - 1
- **SR** — Sharpe Ratio: annualized, sqrt(252) × (mean excess return / std)
- **HR** — Hit Rate: fraction of correct directional predictions
- **MDD** — Maximum Drawdown: max(1 - V_t / peak_t)

## CLI Options

```
--tickers AAPL MSFT     Ticker symbols to backtest
--all-paper-tickers     Use all 14 paper tickers
--signal-only           Rule-based mode (no API cost)
--start-date 2024-06-01 Backtest start date
--end-date 2024-08-31   Backtest end date
--initial-capital 100000
--tx-cost 10            Transaction cost in basis points
--rebalance-days 5      Rebalance frequency
--no-charts             Skip chart generation
--output-dir results/   Custom output directory
```
# LLMRLtrading
