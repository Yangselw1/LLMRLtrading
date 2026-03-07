#!/usr/bin/env python3
"""
Trading-R1 Backtest — Main Entry Point

Orchestrates the full pipeline with configurable verbosity:
  1. Data collection (Yahoo Finance, Finnhub, technical indicators)
  2. Signal generation (Algorithm S1: volatility-based label generation)
  3. LLM analysis (optional: Claude API for investment thesis generation)
  4. Portfolio backtesting (position sizing, weekly rebalancing)
  5. Evaluation (CR, Sharpe, Hit Rate, MDD)
  6. Visualization (equity curves, drawdown, heatmaps)

Verbosity levels (--verbosity):
  0  SILENT   — No output (just final summary)
  1  MINIMAL  — Pipeline stages and final metrics only
  2  NORMAL   — Per-ticker progress, signal distributions, key decisions (default)
  3  DETAILED — Indicator values, prompt previews, individual trade logs
  4  DEBUG    — Full data snapshots, raw API responses, every intermediate value

Usage:
  # Default verbosity (NORMAL):
  python run_backtest.py --tickers AAPL MSFT --signal-only

  # Maximum detail — see everything:
  python run_backtest.py --tickers AAPL --signal-only -v 4

  # Minimal output:
  python run_backtest.py --tickers AAPL --signal-only -v 1

  # Store all inputs (prompts, snapshots, responses) for later inspection:
  python run_backtest.py --tickers AAPL --store-inputs

  # All 14 paper tickers with detailed logging:
  python run_backtest.py --all-paper-tickers --signal-only -v 3
"""
import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime

import pandas as pd

import config
from data_collector import DataCollector
from signal_generator import SignalGenerator
from llm_analyst import LLMAnalyst
from backtester import PortfolioBacktester
from visualizer import Visualizer
from observer import Observer, SILENT, MINIMAL, NORMAL, DETAILED, DEBUG


# ═══════════════════════════════════════════════════════════════════════════════
# CLI Argument Parsing
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="Trading-R1 Backtest: LLM-Powered Financial Trading via Volatility-Based Signals",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Verbosity Levels:
  0 = SILENT    Only final summary
  1 = MINIMAL   Pipeline stages + metrics
  2 = NORMAL    Per-ticker progress + signal distributions (default)
  3 = DETAILED  Indicator values, prompt previews, trade-by-trade logs
  4 = DEBUG     Everything: raw data, API responses, daily P&L

Examples:
  python run_backtest.py --tickers AAPL MSFT --signal-only
  python run_backtest.py --tickers AAPL -v 4 --store-inputs
  python run_backtest.py --all-paper-tickers --signal-only -v 3
        """
    )
    parser.add_argument(
        "--tickers", nargs="+", default=["AAPL"],
        help="Ticker symbols to backtest (default: AAPL)"
    )
    parser.add_argument(
        "--all-paper-tickers", action="store_true",
        help="Use all 14 tickers from the paper"
    )
    parser.add_argument(
        "--start-date", default=config.DEFAULT_START_DATE,
        help=f"Backtest start date (default: {config.DEFAULT_START_DATE})"
    )
    parser.add_argument(
        "--end-date", default=config.DEFAULT_END_DATE,
        help=f"Backtest end date (default: {config.DEFAULT_END_DATE})"
    )
    parser.add_argument(
        "--signal-only", action="store_true",
        help="Use rule-based signals only (no LLM API calls)"
    )
    parser.add_argument(
        "--initial-capital", type=float, default=config.INITIAL_CAPITAL,
        help=f"Initial portfolio capital (default: ${config.INITIAL_CAPITAL:,.0f})"
    )
    parser.add_argument(
        "--tx-cost", type=float, default=config.TRANSACTION_COST_BPS,
        help=f"Transaction cost in basis points (default: {config.TRANSACTION_COST_BPS})"
    )
    parser.add_argument(
        "--rebalance-days", type=int, default=config.REBALANCE_FREQUENCY_DAYS,
        help=f"Rebalance frequency in trading days (default: {config.REBALANCE_FREQUENCY_DAYS})"
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Output directory for results (default: results/YYYYMMDD_HHMMSS)"
    )
    parser.add_argument(
        "--no-charts", action="store_true",
        help="Skip chart generation"
    )

    # ── Verbosity & Observability ─────────────────────────────────────────
    parser.add_argument(
        "-v", "--verbosity", type=int, default=NORMAL, choices=[0,1,2,3,4],
        help="Verbosity level: 0=silent 1=minimal 2=normal 3=detailed 4=debug (default: 2)"
    )
    parser.add_argument(
        "--store-inputs", action="store_true",
        help="Store all inputs (snapshots, prompts, responses) for later inspection"
    )
    parser.add_argument(
        "--no-color", action="store_true",
        help="Disable colored terminal output"
    )

    return parser.parse_args()


# ═══════════════════════════════════════════════════════════════════════════════
# Main Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    start_time = time.time()

    # Resolve tickers
    tickers = config.PAPER_TICKERS if args.all_paper_tickers else args.tickers

    # Set up output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = args.output_dir or os.path.join(config.RESULTS_DIR, timestamp)
    os.makedirs(results_dir, exist_ok=True)

    # Suppress standard logging when using observer (avoid duplicate output)
    log_level = logging.WARNING if args.verbosity >= NORMAL else logging.INFO
    logging.basicConfig(level=log_level, format="%(message)s")

    # ── Create Observer ───────────────────────────────────────────────────
    obs = Observer(
        verbosity=args.verbosity,
        store_inputs=args.store_inputs,
        output_dir=results_dir,
        use_colors=not args.no_color,
        log_to_file=True,
    )

    mode_str = "Signal-Only (rule-based)" if args.signal_only else "LLM-Enhanced (Claude)"
    obs.pipeline_start(tickers, args.start_date, args.end_date, mode_str, args.initial_capital)

    # ── Step 1: Data Collection ───────────────────────────────────────────
    obs.stage("Data Collection", step_num=1, total_steps=6 if not args.no_charts else 5)

    collectors = {}
    for i, ticker in enumerate(tickers, 1):
        obs.ticker_start(ticker, idx=i, total=len(tickers))

        dc = DataCollector(ticker, args.start_date, args.end_date, observer=obs)
        data = dc.collect_all()

        if not data["price_data"].empty:
            collectors[ticker] = dc
        else:
            obs._print(f"    Skipping {ticker}: no price data available", MINIMAL)

        obs.ticker_end(ticker)

    if not collectors:
        obs._print("No valid data for any ticker. Exiting.", MINIMAL)
        sys.exit(1)

    # ── Step 2: Signal Generation (Algorithm S1) ──────────────────────────
    obs.stage("Signal Generation (Algorithm S1)", step_num=2, total_steps=6 if not args.no_charts else 5)

    signal_generators = {}
    for i, (ticker, dc) in enumerate(collectors.items(), 1):
        obs.ticker_start(ticker, idx=i, total=len(collectors))

        sg = SignalGenerator(dc.price_data, ticker=ticker, observer=obs)
        sg.generate()
        signal_generators[ticker] = sg

        # Store diagnostics if requested
        if args.store_inputs:
            diag = sg.get_diagnostics()
            if not diag.empty:
                diag_path = os.path.join(results_dir, "inputs", f"signal_diagnostics_{ticker}.csv")
                os.makedirs(os.path.dirname(diag_path), exist_ok=True)
                diag.to_csv(diag_path)

        obs.ticker_end(ticker)

    # ── Step 3: LLM / Rule-Based Analysis ─────────────────────────────────
    step_name = "LLM Analysis (Claude API)" if not args.signal_only else "Rule-Based Analysis"
    obs.stage(step_name, step_num=3, total_steps=6 if not args.no_charts else 5)

    analyst = LLMAnalyst(use_llm=not args.signal_only, observer=obs)
    llm_decisions = {}
    llm_theses = {}

    if not args.signal_only:
        for i, (ticker, dc) in enumerate(collectors.items(), 1):
            obs.ticker_start(ticker, idx=i, total=len(collectors))
            sg = signal_generators[ticker]
            bt_dates = dc.price_data.loc[args.start_date:args.end_date].index

            rebalance_dates = bt_dates[::args.rebalance_days]
            llm_decisions[ticker] = {}
            llm_theses[ticker] = {}

            for j, date in enumerate(rebalance_dates):
                date_str = date.strftime("%Y-%m-%d")
                snapshot = dc.get_snapshot(date_str)
                if not snapshot:
                    continue

                news = dc.get_news_for_date(date_str)
                signal_label = sg.get_label(date_str)

                thesis, decision = analyst.analyze(snapshot, news, signal_label)
                llm_decisions[ticker][date] = decision
                llm_theses[ticker][date_str] = thesis

                time.sleep(0.5)  # Rate limiting

            obs.ticker_end(ticker)

    # ── Step 4: Backtesting ───────────────────────────────────────────────
    obs.stage("Portfolio Backtesting", step_num=4, total_steps=6 if not args.no_charts else 5)

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

    portfolio_bt = PortfolioBacktester(
        observer=obs,
        initial_capital=args.initial_capital,
        transaction_cost_bps=args.tx_cost,
        rebalance_freq=args.rebalance_days,
    )

    for i, (ticker, data) in enumerate(ticker_data.items(), 1):
        obs.ticker_start(ticker, idx=i, total=len(ticker_data))

    results = portfolio_bt.run_all(ticker_data, args.start_date, args.end_date)
    summary_df = portfolio_bt.get_summary_table()

    # ── Step 5: Results Summary ───────────────────────────────────────────
    obs.stage("Results Summary", step_num=5, total_steps=6 if not args.no_charts else 5)
    obs.log_summary_table(summary_df)

    # Portfolio-level metrics
    if len(results) > 1:
        portfolio_equity = portfolio_bt.get_portfolio_equity()
        if not portfolio_equity.empty:
            from backtester import (compute_cumulative_return, compute_sharpe_ratio,
                                    compute_max_drawdown)
            port_returns = portfolio_equity.pct_change().dropna()
            obs._print(
                f"\n  Portfolio (equal-weight) metrics:"
                f"\n    CR:  {compute_cumulative_return(portfolio_equity)*100:+.2f}%"
                f"\n    SR:  {compute_sharpe_ratio(port_returns):.2f}"
                f"\n    MDD: {compute_max_drawdown(portfolio_equity)*100:.2f}%",
                MINIMAL
            )

    # ── Save Results ──────────────────────────────────────────────────────
    summary_path = os.path.join(results_dir, "summary_metrics.csv")
    summary_df.to_csv(summary_path)

    for ticker, r in results.items():
        if not r.equity_curve.empty:
            r.equity_curve.to_csv(os.path.join(results_dir, f"equity_{ticker}.csv"), header=True)
        if not r.signals.empty:
            r.signals.to_csv(os.path.join(results_dir, f"signals_{ticker}.csv"), header=True)

    if analyst.get_log():
        with open(os.path.join(results_dir, "llm_analysis_log.json"), "w") as f:
            json.dump(analyst.get_log(), f, indent=2, default=str)

    if llm_theses:
        for ticker, theses in llm_theses.items():
            if theses:
                with open(os.path.join(results_dir, f"theses_{ticker}.json"), "w") as f:
                    json.dump(theses, f, indent=2, default=str)

    # ── Step 6: Visualization ─────────────────────────────────────────────
    if not args.no_charts:
        obs.stage("Generating Charts", step_num=6, total_steps=6)

        charts_dir = os.path.join(results_dir, "charts")
        viz = Visualizer(output_dir=charts_dir)
        price_data_dict = {t: collectors[t].price_data for t in collectors}
        viz.generate_all(results, price_data_dict, summary_df)

    # ── Done ──────────────────────────────────────────────────────────────
    obs.pipeline_end()


if __name__ == "__main__":
    main()
