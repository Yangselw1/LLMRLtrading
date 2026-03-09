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

RL Modes (--rl-mode):
  icrl   — In-Context RL with Claude: walk-forward learning via prompt augmentation
  grpo   — GRPO with local model: true RL with weight updates

Comparison (--compare-all):
  Runs all 4 modes (signal-only, LLM, ICRL, GRPO) and generates comparison charts.

Verbosity levels (--verbosity):
  0  SILENT   — No output (just final summary)
  1  MINIMAL  — Pipeline stages and final metrics only
  2  NORMAL   — Per-ticker progress, signal distributions, key decisions (default)
  3  DETAILED — Indicator values, prompt previews, individual trade logs
  4  DEBUG    — Full data snapshots, raw API responses, every intermediate value

Usage:
  # Default verbosity (NORMAL):
  python run_backtest.py --tickers AAPL MSFT --signal-only

  # RL modes:
  python run_backtest.py --tickers AAPL --rl-mode icrl
  python run_backtest.py --tickers AAPL --rl-mode grpo --grpo-sft-first

  # Compare all modes:
  python run_backtest.py --tickers SPY AAPL --compare-all
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

RL Modes:
  icrl  = In-Context RL with Claude (prompt augmentation with past outcomes)
  grpo  = GRPO with local model (true RL with weight updates)

Examples:
  python run_backtest.py --tickers AAPL MSFT --signal-only
  python run_backtest.py --tickers AAPL --rl-mode icrl
  python run_backtest.py --tickers AAPL --rl-mode grpo --grpo-sft-first
  python run_backtest.py --tickers SPY AAPL --compare-all
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

    # ── RL Mode ───────────────────────────────────────────────────────────
    parser.add_argument(
        "--rl-mode", choices=["icrl", "grpo"], default=None,
        help="RL mode: 'icrl' for In-Context RL with Claude, 'grpo' for GRPO with local model"
    )
    parser.add_argument(
        "--compare-all", action="store_true",
        help="Run all 4 modes (signal-only, LLM, ICRL, GRPO) and generate comparison"
    )

    # ── ICRL-specific ─────────────────────────────────────────────────────
    parser.add_argument(
        "--icrl-top-k", type=int, default=5,
        help="Number of best trades in ICRL prompt (default: 5)"
    )
    parser.add_argument(
        "--icrl-bottom-k", type=int, default=5,
        help="Number of worst trades in ICRL prompt (default: 5)"
    )
    parser.add_argument(
        "--icrl-recent-n", type=int, default=10,
        help="Number of recent trades in ICRL prompt (default: 10)"
    )

    # ── GRPO-specific ─────────────────────────────────────────────────────
    parser.add_argument(
        "--grpo-model", type=str, default="Qwen/Qwen3-4B",
        help="HuggingFace model name for GRPO (default: Qwen/Qwen3-4B)"
    )
    parser.add_argument(
        "--grpo-sft-first", action="store_true",
        help="Run supervised fine-tuning on S1 labels before GRPO RL"
    )
    parser.add_argument(
        "--grpo-checkpoint", type=str, default=None,
        help="Path to load a pre-trained GRPO checkpoint"
    )
    parser.add_argument(
        "--grpo-retrain-interval", type=int, default=20,
        help="GRPO retraining interval in completed trades (default: 20)"
    )

    # ── Experience buffer ─────────────────────────────────────────────────
    parser.add_argument(
        "--load-experience", type=str, default=None,
        help="Path to load a saved experience buffer"
    )
    parser.add_argument(
        "--save-experience", action="store_true",
        help="Save the experience buffer after the run"
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
# RL Pipeline Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def run_rl_backtest(mode, args, collectors, signal_generators, obs, results_dir):
    """
    Run RL walk-forward backtest (ICRL or GRPO).

    Returns:
        Dict[str, BacktestResult]: Results per ticker
        ExperienceBuffer: Updated experience buffer
    """
    from experience_buffer import ExperienceBuffer
    from rl_backtester import RLBacktester

    # Initialize experience buffer
    experience_buffer = ExperienceBuffer()
    if args.load_experience:
        experience_buffer.load(args.load_experience)

    grpo_trainer = None

    if mode == "icrl":
        from icrl_analyst import ICRLAnalyst
        analyst = ICRLAnalyst(
            experience_buffer=experience_buffer,
            observer=obs,
            top_k=args.icrl_top_k,
            bottom_k=args.icrl_bottom_k,
            recent_n=args.icrl_recent_n,
        )
        obs.log_rl_mode("icrl", {
            "top_k": args.icrl_top_k,
            "bottom_k": args.icrl_bottom_k,
            "recent_n": args.icrl_recent_n,
        })

    elif mode == "grpo":
        from grpo_trainer import GRPOTrainer, GRPOConfig
        from grpo_analyst import GRPOAnalyst
        from experience_buffer import Experience

        grpo_config = GRPOConfig(model_name=args.grpo_model)
        grpo_trainer = GRPOTrainer(grpo_config, observer=obs)
        grpo_trainer.load_model()

        # Optional SFT pre-training
        if args.grpo_sft_first:
            obs._print("  GRPO: Running supervised pre-training on S1 labels...", MINIMAL)
            sft_experiences = []
            for ticker, sg in signal_generators.items():
                dc = collectors[ticker]
                bt_dates = dc.price_data.loc[args.start_date:args.end_date].index
                for date in bt_dates[::args.rebalance_days][:20]:  # First 20 dates
                    date_str = date.strftime("%Y-%m-%d")
                    snapshot = dc.get_snapshot(date_str)
                    if snapshot:
                        label = sg.get_label(date_str) or "HOLD"
                        news = dc.get_news_for_date(date_str)
                        sft_experiences.append(Experience(
                            step_idx=0, date=date_str, ticker=ticker,
                            state_snapshot=snapshot, state_news=news,
                            signal_label=label, action=label,
                            reasoning="", mode="sft",
                        ))
            if sft_experiences:
                grpo_trainer.supervised_pretrain(sft_experiences)

        if args.grpo_checkpoint:
            grpo_trainer.load_checkpoint(args.grpo_checkpoint)

        analyst = GRPOAnalyst(grpo_trainer, observer=obs)
        obs.log_rl_mode("grpo", {
            "model": args.grpo_model,
            "sft_first": args.grpo_sft_first,
            "retrain_interval": args.grpo_retrain_interval,
        })

    # Run walk-forward for each ticker
    rl_bt = RLBacktester(
        mode=mode,
        analyst=analyst,
        experience_buffer=experience_buffer,
        observer=obs,
        grpo_trainer=grpo_trainer,
        grpo_retrain_interval=args.grpo_retrain_interval,
        initial_capital=args.initial_capital,
        transaction_cost_bps=args.tx_cost,
        rebalance_freq=args.rebalance_days,
    )

    results = {}
    for i, (ticker, dc) in enumerate(collectors.items(), 1):
        obs.ticker_start(ticker, idx=i, total=len(collectors))

        sg = signal_generators[ticker]
        result, experience_buffer = rl_bt.run(
            ticker=ticker,
            price_data=dc.price_data,
            signal_labels=sg.labels,
            data_collector=dc,
            start_date=args.start_date,
            end_date=args.end_date,
        )
        results[ticker] = result
        obs.ticker_end(ticker)

    # Save experience buffer
    if args.save_experience:
        buf_path = os.path.join(results_dir, f"experience_buffer_{mode}.json")
        experience_buffer.save(buf_path)

    # Save GRPO checkpoint
    if mode == "grpo" and grpo_trainer:
        ckpt_path = os.path.join(results_dir, "grpo_checkpoint")
        grpo_trainer.save_checkpoint(ckpt_path)

    return results, experience_buffer


def run_standard_backtest(signal_only, args, collectors, signal_generators, obs):
    """
    Run standard backtest (signal-only or LLM mode).

    Returns:
        Dict[str, BacktestResult]: Results per ticker
    """
    analyst = LLMAnalyst(use_llm=not signal_only, observer=obs)
    llm_decisions = {}
    llm_theses = {}

    if not signal_only:
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

    # Build signal series with LLM overrides
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

    results = portfolio_bt.run_all(ticker_data, args.start_date, args.end_date)
    return results, portfolio_bt


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
        store_inputs=args.store_inputs or bool(args.rl_mode) or args.compare_all,
        output_dir=results_dir,
        use_colors=not args.no_color,
        log_to_file=True,
    )

    # Determine mode string
    if args.compare_all:
        mode_str = "Compare All (Signal-Only / LLM / ICRL / GRPO)"
    elif args.rl_mode:
        mode_str = f"RL-Enhanced ({args.rl_mode.upper()})"
    elif args.signal_only:
        mode_str = "Signal-Only (rule-based)"
    else:
        mode_str = "LLM-Enhanced (Claude)"

    obs.pipeline_start(tickers, args.start_date, args.end_date, mode_str, args.initial_capital)

    # ── Step 1: Data Collection ───────────────────────────────────────────
    total_steps = 6 if not args.no_charts else 5
    obs.stage("Data Collection", step_num=1, total_steps=total_steps)

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
    obs.stage("Signal Generation (Algorithm S1)", step_num=2, total_steps=total_steps)

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

    # ═══════════════════════════════════════════════════════════════════════
    # Branch: RL Mode, Compare-All, or Standard
    # ═══════════════════════════════════════════════════════════════════════

    if args.compare_all:
        # ── Compare All Modes ─────────────────────────────────────────────
        results_by_mode = {}
        summary_by_mode = {}

        # Mode 1: Signal-Only
        obs.stage("Mode 1/4: Signal-Only Backtest", step_num=3, total_steps=total_steps)
        so_results, so_bt = run_standard_backtest(
            signal_only=True, args=args, collectors=collectors,
            signal_generators=signal_generators, obs=obs)
        results_by_mode["signal_only"] = so_results
        summary_by_mode["Signal-Only"] = so_bt.get_summary_table()

        # Mode 2: LLM (no RL)
        obs.stage("Mode 2/4: LLM Backtest (Claude, no RL)", step_num=3, total_steps=total_steps)
        llm_results, llm_bt = run_standard_backtest(
            signal_only=False, args=args, collectors=collectors,
            signal_generators=signal_generators, obs=obs)
        results_by_mode["llm_no_rl"] = llm_results
        summary_by_mode["LLM (no RL)"] = llm_bt.get_summary_table()

        # Mode 3: ICRL
        obs.stage("Mode 3/4: ICRL Backtest (Claude + In-Context RL)", step_num=3, total_steps=total_steps)
        icrl_results, icrl_buffer = run_rl_backtest(
            mode="icrl", args=args, collectors=collectors,
            signal_generators=signal_generators, obs=obs, results_dir=results_dir)
        results_by_mode["icrl"] = icrl_results
        # Build summary table for ICRL
        icrl_rows = []
        for ticker, r in icrl_results.items():
            icrl_rows.append({
                "Ticker": ticker,
                "CR(%)": round(r.cumulative_return * 100, 2),
                "SR": round(r.sharpe_ratio, 2),
                "HR(%)": round(r.hit_rate * 100, 1),
                "MDD(%)": round(r.max_drawdown * 100, 2),
                "Trades": r.total_trades,
                "Win": r.winning_trades,
                "Loss": r.losing_trades,
            })
        summary_by_mode["ICRL"] = pd.DataFrame(icrl_rows).set_index("Ticker")

        # Mode 4: GRPO
        try:
            obs.stage("Mode 4/4: GRPO Backtest (Local Model + RL)", step_num=3, total_steps=total_steps)
            grpo_results, grpo_buffer = run_rl_backtest(
                mode="grpo", args=args, collectors=collectors,
                signal_generators=signal_generators, obs=obs, results_dir=results_dir)
            results_by_mode["grpo"] = grpo_results
            grpo_rows = []
            for ticker, r in grpo_results.items():
                grpo_rows.append({
                    "Ticker": ticker,
                    "CR(%)": round(r.cumulative_return * 100, 2),
                    "SR": round(r.sharpe_ratio, 2),
                    "HR(%)": round(r.hit_rate * 100, 1),
                    "MDD(%)": round(r.max_drawdown * 100, 2),
                    "Trades": r.total_trades,
                    "Win": r.winning_trades,
                    "Loss": r.losing_trades,
                })
            summary_by_mode["GRPO"] = pd.DataFrame(grpo_rows).set_index("Ticker")
        except ImportError as e:
            obs._print(f"\n  {e}", MINIMAL)
            obs._print("  Skipping GRPO mode (missing dependencies)", MINIMAL)

        # Comparison summary
        obs.stage("Results Comparison", step_num=4, total_steps=total_steps)
        obs.log_mode_comparison(summary_by_mode)

        # Save comparison CSV
        all_rows = []
        for mode_name, summary_df in summary_by_mode.items():
            for ticker in summary_df.index:
                row = summary_df.loc[ticker].to_dict()
                row["Mode"] = mode_name
                row["Ticker"] = ticker
                all_rows.append(row)
        comparison_df = pd.DataFrame(all_rows)
        comparison_df.to_csv(
            os.path.join(results_dir, "comparison_metrics.csv"), index=False
        )

        # Use signal-only as primary results for saving
        results = so_results
        summary_df = summary_by_mode["Signal-Only"]

        # Save per-mode results
        for mode_name, mode_results in results_by_mode.items():
            mode_dir = os.path.join(results_dir, mode_name)
            os.makedirs(mode_dir, exist_ok=True)
            for ticker, r in mode_results.items():
                if not r.equity_curve.empty:
                    r.equity_curve.to_csv(
                        os.path.join(mode_dir, f"equity_{ticker}.csv"),
                        header=True
                    )

        # Comparison charts
        if not args.no_charts:
            obs.stage("Generating Comparison Charts", step_num=5, total_steps=total_steps)
            try:
                from rl_visualizer import (
                    plot_mode_comparison_equity,
                    plot_mode_comparison_heatmap,
                    plot_reward_evolution,
                    plot_reward_dimensions_radar,
                    plot_dimension_evolution,
                )
                charts_dir = os.path.join(results_dir, "charts", "comparison")
                os.makedirs(charts_dir, exist_ok=True)

                for ticker in collectors:
                    plot_mode_comparison_equity(
                        results_by_mode, ticker,
                        save_path=os.path.join(charts_dir, f"comparison_equity_{ticker}.png")
                    )

                plot_mode_comparison_heatmap(
                    summary_by_mode,
                    save_path=os.path.join(charts_dir, "comparison_sharpe_heatmap.png")
                )

                if 'icrl_buffer' in dir():
                    plot_reward_evolution(
                        icrl_buffer,
                        save_path=os.path.join(charts_dir, "icrl_reward_evolution.png")
                    )
                    plot_reward_dimensions_radar(
                        icrl_buffer,
                        save_path=os.path.join(charts_dir, "icrl_reward_dimensions.png")
                    )
                    plot_dimension_evolution(
                        icrl_buffer,
                        save_path=os.path.join(charts_dir, "icrl_dimension_evolution.png")
                    )
                if 'grpo_buffer' in dir():
                    plot_reward_evolution(
                        grpo_buffer,
                        save_path=os.path.join(charts_dir, "grpo_reward_evolution.png")
                    )
                    plot_reward_dimensions_radar(
                        grpo_buffer,
                        save_path=os.path.join(charts_dir, "grpo_reward_dimensions.png")
                    )
                    plot_dimension_evolution(
                        grpo_buffer,
                        save_path=os.path.join(charts_dir, "grpo_dimension_evolution.png")
                    )
            except ImportError:
                obs._print("  RL visualization module not available", MINIMAL)

    elif args.rl_mode:
        # ── Single RL Mode ────────────────────────────────────────────────
        obs.stage(f"RL Walk-Forward Backtest ({args.rl_mode.upper()})",
                  step_num=3, total_steps=total_steps)

        results, experience_buffer = run_rl_backtest(
            mode=args.rl_mode, args=args, collectors=collectors,
            signal_generators=signal_generators, obs=obs, results_dir=results_dir
        )

        # Build summary table
        rows = []
        for ticker, r in results.items():
            rows.append({
                "Ticker": ticker,
                "CR(%)": round(r.cumulative_return * 100, 2),
                "SR": round(r.sharpe_ratio, 2),
                "HR(%)": round(r.hit_rate * 100, 1),
                "MDD(%)": round(r.max_drawdown * 100, 2),
                "Trades": r.total_trades,
                "Win": r.winning_trades,
                "Loss": r.losing_trades,
            })
        summary_df = pd.DataFrame(rows).set_index("Ticker")

        # Results summary
        obs.stage("Results Summary", step_num=4, total_steps=total_steps)
        obs.log_summary_table(summary_df)

        # Save results
        summary_path = os.path.join(results_dir, "summary_metrics.csv")
        summary_df.to_csv(summary_path)

        for ticker, r in results.items():
            if not r.equity_curve.empty:
                r.equity_curve.to_csv(
                    os.path.join(results_dir, f"equity_{ticker}.csv"), header=True)
            if not r.signals.empty:
                r.signals.to_csv(
                    os.path.join(results_dir, f"signals_{ticker}.csv"), header=True)

        # Save reward evolution and dimension stats
        buffer_stats = experience_buffer.summary_stats()
        with open(os.path.join(results_dir, "rl_stats.json"), "w") as f:
            json.dump(buffer_stats, f, indent=2, default=str)

        # Generate RL-specific charts
        if not args.no_charts:
            try:
                from rl_visualizer import (
                    plot_reward_evolution,
                    plot_reward_dimensions_radar,
                    plot_dimension_evolution,
                )
                charts_dir = os.path.join(results_dir, "charts")
                os.makedirs(charts_dir, exist_ok=True)

                plot_reward_evolution(
                    experience_buffer,
                    save_path=os.path.join(charts_dir, f"{args.rl_mode}_reward_evolution.png")
                )
                plot_reward_dimensions_radar(
                    experience_buffer,
                    save_path=os.path.join(charts_dir, f"{args.rl_mode}_reward_dimensions.png")
                )
                plot_dimension_evolution(
                    experience_buffer,
                    save_path=os.path.join(charts_dir, f"{args.rl_mode}_dimension_evolution.png")
                )
            except ImportError:
                obs._print("  RL visualization module not available", MINIMAL)

    else:
        # ── Standard Pipeline (signal-only or LLM) ───────────────────────
        step_name = ("LLM Analysis (Claude API)" if not args.signal_only
                     else "Rule-Based Analysis")
        obs.stage(step_name, step_num=3, total_steps=total_steps)

        results, portfolio_bt = run_standard_backtest(
            signal_only=args.signal_only, args=args, collectors=collectors,
            signal_generators=signal_generators, obs=obs
        )
        summary_df = portfolio_bt.get_summary_table()

        # ── Results Summary ───────────────────────────────────────────────
        obs.stage("Results Summary", step_num=4, total_steps=total_steps)
        obs.log_summary_table(summary_df)

        # Portfolio-level metrics
        if len(results) > 1:
            portfolio_equity = portfolio_bt.get_portfolio_equity()
            if not portfolio_equity.empty:
                from backtester import (compute_cumulative_return,
                                        compute_sharpe_ratio,
                                        compute_max_drawdown)
                port_returns = portfolio_equity.pct_change().dropna()
                obs._print(
                    f"\n  Portfolio (equal-weight) metrics:"
                    f"\n    CR:  {compute_cumulative_return(portfolio_equity)*100:+.2f}%"
                    f"\n    SR:  {compute_sharpe_ratio(port_returns):.2f}"
                    f"\n    MDD: {compute_max_drawdown(portfolio_equity)*100:.2f}%",
                    MINIMAL
                )

        # Save Results
        summary_path = os.path.join(results_dir, "summary_metrics.csv")
        summary_df.to_csv(summary_path)

        for ticker, r in results.items():
            if not r.equity_curve.empty:
                r.equity_curve.to_csv(
                    os.path.join(results_dir, f"equity_{ticker}.csv"), header=True)
            if not r.signals.empty:
                r.signals.to_csv(
                    os.path.join(results_dir, f"signals_{ticker}.csv"), header=True)

    # ── Visualization (standard charts for non-compare mode) ──────────────
    if not args.no_charts and not args.compare_all:
        obs.stage("Generating Charts", step_num=total_steps, total_steps=total_steps)

        charts_dir = os.path.join(results_dir, "charts")
        viz = Visualizer(output_dir=charts_dir)
        price_data_dict = {t: collectors[t].price_data for t in collectors}
        viz.generate_all(results, price_data_dict, summary_df)

    # ── Done ──────────────────────────────────────────────────────────────
    obs.pipeline_end()


if __name__ == "__main__":
    main()
