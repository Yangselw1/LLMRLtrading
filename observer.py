"""
Observer — Real-Time Logging, Input Storage & Progress Tracking

Provides granular, configurable observability across the entire pipeline:

Verbosity levels (cumulative — each level includes everything below it):
  0  SILENT   — No output (just final summary)
  1  MINIMAL  — Pipeline stages and final metrics only
  2  NORMAL   — Per-ticker progress, signal distributions, key decisions (default)
  3  DETAILED — Indicator values, prompt previews, individual trade logs
  4  DEBUG    — Full data snapshots, raw API responses, every intermediate computation

Usage:
  observer = Observer(verbosity=3, store_inputs=True, output_dir="results/run1")
  observer.stage("Data Collection")
  observer.ticker_start("AAPL")
  observer.log_price_data(df)
  observer.store_snapshot("AAPL", "2024-06-03", snapshot_dict)
"""
import json
import logging
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional
from io import StringIO

import numpy as np
import pandas as pd

# ═══════════════════════════════════════════════════════════════════════════════
# Verbosity Levels
# ═══════════════════════════════════════════════════════════════════════════════

SILENT   = 0
MINIMAL  = 1
NORMAL   = 2
DETAILED = 3
DEBUG    = 4

LEVEL_NAMES = {0: "SILENT", 1: "MINIMAL", 2: "NORMAL", 3: "DETAILED", 4: "DEBUG"}


# ═══════════════════════════════════════════════════════════════════════════════
# Color & Formatting Helpers (terminal output)
# ═══════════════════════════════════════════════════════════════════════════════

class Colors:
    HEADER  = "\033[95m"
    BLUE    = "\033[94m"
    CYAN    = "\033[96m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    RED     = "\033[91m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    RESET   = "\033[0m"

    @staticmethod
    def disable():
        for attr in ["HEADER","BLUE","CYAN","GREEN","YELLOW","RED","BOLD","DIM","RESET"]:
            setattr(Colors, attr, "")


def _ts():
    """Compact timestamp for log lines."""
    return datetime.now().strftime("%H:%M:%S")


def _bar(pct: float, width: int = 30) -> str:
    """ASCII progress bar."""
    filled = int(width * pct)
    return f"[{'█' * filled}{'░' * (width - filled)}] {pct*100:5.1f}%"


def _fmt_dollars(val):
    if val is None: return "N/A"
    if abs(val) >= 1e12: return f"${val/1e12:.2f}T"
    if abs(val) >= 1e9:  return f"${val/1e9:.2f}B"
    if abs(val) >= 1e6:  return f"${val/1e6:.1f}M"
    return f"${val:,.0f}"


def _fmt_pct(val):
    if val is None: return "N/A"
    return f"{val*100:.2f}%" if abs(val) < 1 else f"{val:.2f}%"


# ═══════════════════════════════════════════════════════════════════════════════
# Observer Class
# ═══════════════════════════════════════════════════════════════════════════════

class Observer:
    """
    Central observability hub for the Trading-R1 backtest pipeline.

    Handles:
    - Real-time console output with color-coded verbosity
    - File logging (always captures everything regardless of console verbosity)
    - Input/snapshot storage (prompts, data, API responses)
    - Progress tracking with timing
    - Structured JSON audit trail
    """

    def __init__(
        self,
        verbosity: int = NORMAL,
        store_inputs: bool = True,
        output_dir: str = "results",
        use_colors: bool = True,
        log_to_file: bool = True,
    ):
        self.verbosity = verbosity
        self.store_inputs = store_inputs
        self.output_dir = output_dir
        self.use_colors = use_colors

        if not use_colors:
            Colors.disable()

        # Storage directories
        self.inputs_dir = os.path.join(output_dir, "inputs")
        self.snapshots_dir = os.path.join(output_dir, "snapshots")
        self.prompts_dir = os.path.join(output_dir, "prompts")
        self.responses_dir = os.path.join(output_dir, "llm_responses")
        self.trades_dir = os.path.join(output_dir, "trades")

        if store_inputs:
            for d in [self.inputs_dir, self.snapshots_dir, self.prompts_dir,
                      self.responses_dir, self.trades_dir]:
                os.makedirs(d, exist_ok=True)

        # File logger (always DEBUG level — captures everything)
        self.file_logger = None
        if log_to_file:
            os.makedirs(output_dir, exist_ok=True)
            self.file_logger = logging.getLogger("trading_r1_observer")
            self.file_logger.setLevel(logging.DEBUG)
            self.file_logger.propagate = False  # Prevent duplicate output via root logger
            self.file_logger.handlers = []  # Clear existing handlers
            fh = logging.FileHandler(
                os.path.join(output_dir, "detailed_log.log"), mode="w"
            )
            fh.setFormatter(logging.Formatter(
                "%(asctime)s [%(levelname)-7s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
            ))
            self.file_logger.addHandler(fh)

        # Timing
        self._stage_start = None
        self._pipeline_start = time.time()
        self._ticker_start = None

        # Audit trail
        self.audit_trail: List[Dict] = []

    # ── Console Output ────────────────────────────────────────────────────

    def _print(self, msg: str, min_level: int = NORMAL):
        """Print to console if verbosity >= min_level."""
        if self.verbosity >= min_level:
            print(msg)
        if self.file_logger:
            # Strip ANSI colors for file
            clean = msg
            for attr in ["HEADER","BLUE","CYAN","GREEN","YELLOW","RED","BOLD","DIM","RESET"]:
                clean = clean.replace(getattr(Colors, attr), "")
            self.file_logger.info(clean)

    def _file_only(self, msg: str, level: str = "DEBUG"):
        """Write to file log only (never printed to console)."""
        if self.file_logger:
            getattr(self.file_logger, level.lower(), self.file_logger.debug)(msg)

    # ── Pipeline Lifecycle ────────────────────────────────────────────────

    def pipeline_start(self, tickers: list, start_date: str, end_date: str,
                       mode: str, capital: float):
        """Called once at the very beginning."""
        self._print(
            f"\n{Colors.BOLD}{Colors.CYAN}"
            f"╔══════════════════════════════════════════════════════════════════╗\n"
            f"║  Trading-R1 Backtest System                                     ║\n"
            f"╚══════════════════════════════════════════════════════════════════╝"
            f"{Colors.RESET}", MINIMAL
        )
        self._print(f"  {Colors.BOLD}Tickers:{Colors.RESET}    {', '.join(tickers)}", MINIMAL)
        self._print(f"  {Colors.BOLD}Period:{Colors.RESET}     {start_date} → {end_date}", MINIMAL)
        self._print(f"  {Colors.BOLD}Mode:{Colors.RESET}       {mode}", MINIMAL)
        self._print(f"  {Colors.BOLD}Capital:{Colors.RESET}    {_fmt_dollars(capital)}", MINIMAL)
        self._print(f"  {Colors.BOLD}Verbosity:{Colors.RESET}  {LEVEL_NAMES.get(self.verbosity, '?')} (level {self.verbosity})", MINIMAL)
        self._print(f"  {Colors.BOLD}Inputs:{Colors.RESET}     {'Stored' if self.store_inputs else 'Not stored'}", MINIMAL)
        self._print(f"  {Colors.BOLD}Output:{Colors.RESET}     {self.output_dir}", MINIMAL)
        self._print("", MINIMAL)

        self._audit("pipeline_start", {
            "tickers": tickers, "start_date": start_date, "end_date": end_date,
            "mode": mode, "capital": capital, "verbosity": self.verbosity,
        })

    def pipeline_end(self):
        """Called once at the very end."""
        elapsed = time.time() - self._pipeline_start
        self._print(
            f"\n{Colors.BOLD}{Colors.GREEN}"
            f"╔══════════════════════════════════════════════════════════════════╗\n"
            f"║  Pipeline complete in {elapsed:.1f}s"
            f"{' ' * (41 - len(f'{elapsed:.1f}'))}║\n"
            f"║  Results: {self.output_dir}"
            f"{' ' * max(1, 52 - len(self.output_dir))}║\n"
            f"╚══════════════════════════════════════════════════════════════════╝"
            f"{Colors.RESET}", MINIMAL
        )

        # Save audit trail
        if self.store_inputs:
            audit_path = os.path.join(self.output_dir, "audit_trail.json")
            with open(audit_path, "w") as f:
                json.dump(self.audit_trail, f, indent=2, default=str)

    # ── Stage Tracking ────────────────────────────────────────────────────

    def stage(self, name: str, step_num: int = None, total_steps: int = None):
        """Mark the beginning of a major pipeline stage."""
        if self._stage_start is not None:
            elapsed = time.time() - self._stage_start
            self._print(
                f"  {Colors.DIM}└─ Stage completed in {elapsed:.1f}s{Colors.RESET}",
                NORMAL
            )

        self._stage_start = time.time()
        prefix = f"STEP {step_num}/{total_steps}" if step_num else "STAGE"
        self._print(
            f"\n{Colors.BOLD}{Colors.BLUE}{'═'*60}\n"
            f"  {prefix}: {name}\n"
            f"{'═'*60}{Colors.RESET}",
            MINIMAL
        )
        self._audit("stage_start", {"name": name, "step": step_num})

    # ── Ticker-Level Progress ─────────────────────────────────────────────

    def ticker_start(self, ticker: str, idx: int = None, total: int = None):
        """Mark the beginning of processing for a ticker."""
        self._ticker_start = time.time()
        progress = f" ({idx}/{total})" if idx and total else ""
        bar = _bar(idx / total) if idx and total else ""
        self._print(
            f"\n  {Colors.BOLD}{Colors.CYAN}▶ {ticker}{progress}{Colors.RESET}  {bar}",
            NORMAL
        )
        self._audit("ticker_start", {"ticker": ticker, "idx": idx, "total": total})

    def ticker_end(self, ticker: str):
        """Mark the end of processing for a ticker."""
        if self._ticker_start:
            elapsed = time.time() - self._ticker_start
            self._print(
                f"  {Colors.DIM}  └─ {ticker} done in {elapsed:.1f}s{Colors.RESET}",
                NORMAL
            )

    # ── Data Collection Logging ───────────────────────────────────────────

    def log_price_data(self, ticker: str, df: pd.DataFrame):
        """Log details about fetched price data."""
        if df.empty:
            self._print(f"    {Colors.RED}✗ No price data returned{Colors.RESET}", NORMAL)
            return

        self._print(
            f"    {Colors.GREEN}✓{Colors.RESET} Price data: {len(df)} days "
            f"({df.index[0].strftime('%Y-%m-%d')} → {df.index[-1].strftime('%Y-%m-%d')})",
            NORMAL
        )
        self._print(
            f"      Latest: O={df.iloc[-1]['open']:.2f}  H={df.iloc[-1]['high']:.2f}  "
            f"L={df.iloc[-1]['low']:.2f}  C={df.iloc[-1]['close']:.2f}  "
            f"V={int(df.iloc[-1]['volume']):,}",
            DETAILED
        )

        # DEBUG: full price range stats
        self._print(
            f"      Price range: ${df['close'].min():.2f} – ${df['close'].max():.2f}  "
            f"(mean: ${df['close'].mean():.2f}, std: ${df['close'].std():.2f})",
            DEBUG
        )

        if self.store_inputs:
            path = os.path.join(self.inputs_dir, f"price_{ticker}.csv")
            df.to_csv(path)
            self._file_only(f"Price data saved: {path}")

    def log_technical_indicators(self, ticker: str, df: pd.DataFrame):
        """Log computed technical indicators."""
        indicator_cols = [c for c in df.columns if c not in
                         ["open", "high", "low", "close", "volume"]]
        self._print(
            f"    {Colors.GREEN}✓{Colors.RESET} Technical indicators: {len(indicator_cols)} computed",
            NORMAL
        )

        if indicator_cols:
            latest = df.iloc[-1]
            key_indicators = {}
            for col in ["rsi", "macd", "adx", "atr", "sma_50", "sma_200", "bb_upper", "bb_lower"]:
                if col in df.columns and pd.notna(latest.get(col)):
                    key_indicators[col] = round(float(latest[col]), 2)

            if key_indicators:
                self._print(
                    f"      Key values (latest): {key_indicators}",
                    DETAILED
                )

            # DEBUG: all indicator values
            if self.verbosity >= DEBUG:
                for col in indicator_cols:
                    val = latest.get(col)
                    if pd.notna(val):
                        self._print(f"        {col:>20s} = {float(val):.4f}", DEBUG)

        if self.store_inputs:
            path = os.path.join(self.inputs_dir, f"indicators_{ticker}.csv")
            df[indicator_cols].tail(30).to_csv(path)

    def log_fundamentals(self, ticker: str, fundamentals: Dict):
        """Log fetched fundamental data."""
        if not fundamentals:
            self._print(f"    {Colors.YELLOW}⚠{Colors.RESET} No fundamentals available", NORMAL)
            return

        name = fundamentals.get("short_name", ticker)
        sector = fundamentals.get("sector", "N/A")
        mcap = _fmt_dollars(fundamentals.get("market_cap"))

        self._print(
            f"    {Colors.GREEN}✓{Colors.RESET} Fundamentals: {name} | {sector} | MCap: {mcap}",
            NORMAL
        )

        # DETAILED: key ratios
        pe = fundamentals.get("pe_ratio")
        roe = fundamentals.get("roe")
        margin = fundamentals.get("profit_margin")
        self._print(
            f"      P/E: {pe or 'N/A'}  |  ROE: {_fmt_pct(roe)}  |  "
            f"Margin: {_fmt_pct(margin)}  |  D/E: {fundamentals.get('debt_to_equity', 'N/A')}",
            DETAILED
        )

        # DEBUG: everything
        if self.verbosity >= DEBUG:
            for k, v in fundamentals.items():
                self._print(f"        {k:>20s} = {v}", DEBUG)

        if self.store_inputs:
            path = os.path.join(self.inputs_dir, f"fundamentals_{ticker}.json")
            with open(path, "w") as f:
                json.dump(fundamentals, f, indent=2, default=str)

    def log_news(self, ticker: str, news_buckets: Dict, note: str = ""):
        """Log fetched news data."""
        total = sum(len(v) for v in news_buckets.values())
        if total == 0:
            msg = "No news articles found"
            if note:
                msg += f" ({note})"
            self._print(f"    {Colors.YELLOW}⚠{Colors.RESET} {msg}", NORMAL)
            return

        bucket_counts = {k: len(v) for k, v in news_buckets.items()}
        self._print(
            f"    {Colors.GREEN}✓{Colors.RESET} News: {total} articles "
            f"(3d:{bucket_counts.get('last_3_days',0)} | "
            f"4-10d:{bucket_counts.get('last_4_10_days',0)} | "
            f"11-30d:{bucket_counts.get('last_11_30_days',0)})",
            NORMAL
        )

        # DETAILED: show top headlines
        for bucket, articles in news_buckets.items():
            for art in articles[:2]:
                self._print(
                    f"      {Colors.DIM}[{bucket}] {art.get('headline', '')[:80]}{Colors.RESET}",
                    DETAILED
                )

        if self.store_inputs:
            path = os.path.join(self.inputs_dir, f"news_{ticker}.json")
            with open(path, "w") as f:
                json.dump(news_buckets, f, indent=2, default=str)

    def log_analyst_recs(self, ticker: str, recs: Dict):
        """Log analyst recommendations."""
        if not recs:
            self._print(f"    {Colors.YELLOW}⚠{Colors.RESET} No analyst recs available", NORMAL)
            return

        score = recs.get("consensus_score", "N/A")
        self._print(
            f"    {Colors.GREEN}✓{Colors.RESET} Analyst consensus: {score}/5.0 "
            f"(SB:{recs.get('strong_buy',0)} B:{recs.get('buy',0)} "
            f"H:{recs.get('hold',0)} S:{recs.get('sell',0)} SS:{recs.get('strong_sell',0)})",
            NORMAL
        )

    # ── Signal Generation Logging ─────────────────────────────────────────

    def log_signal_computation(self, ticker: str, step: str, data: Any = None):
        """Log intermediate signal computation steps."""
        self._print(f"    {Colors.DIM}↳ {step}{Colors.RESET}", DETAILED)
        if data is not None and self.verbosity >= DEBUG:
            if isinstance(data, pd.Series):
                self._print(f"        Tail: {data.tail(5).to_dict()}", DEBUG)
            elif isinstance(data, dict):
                self._print(f"        {data}", DEBUG)

    def log_signal_thresholds(self, ticker: str, thresholds: list, quantiles: list):
        """Log the computed percentile thresholds."""
        self._print(
            f"    {Colors.BOLD}Signal thresholds:{Colors.RESET}",
            DETAILED
        )
        labels = ["STRONG_SELL", "SELL", "HOLD", "BUY", "STRONG_BUY"]
        boundaries = ["-∞"] + [f"{t:.4f}" for t in thresholds] + ["+∞"]
        for i, label in enumerate(labels):
            self._print(
                f"      {label:>12s}: [{boundaries[i]:>8s}, {boundaries[i+1]:>8s})",
                DETAILED
            )

    def log_signal_distribution(self, ticker: str, labels: pd.Series):
        """Log the distribution of generated signals."""
        counts = labels.value_counts()
        total = counts.sum()
        if total == 0:
            return

        from config import ACTION_LABELS
        self._print(f"    {Colors.BOLD}Signal distribution for {ticker}:{Colors.RESET}", NORMAL)
        for lbl in ACTION_LABELS:
            count = counts.get(lbl, 0)
            pct = count / total * 100
            bar_len = int(pct / 2)
            color = {
                "STRONG_BUY": Colors.GREEN, "BUY": Colors.GREEN,
                "HOLD": Colors.YELLOW,
                "SELL": Colors.RED, "STRONG_SELL": Colors.RED,
            }.get(lbl, "")
            self._print(
                f"      {lbl:>12s}: {color}{'█' * bar_len}{Colors.RESET}"
                f" {count:4d} ({pct:5.1f}%)",
                NORMAL
            )

    # ── LLM Analysis Logging ─────────────────────────────────────────────

    def log_llm_prompt(self, ticker: str, date: str, prompt: str):
        """Log (and optionally store) the LLM prompt."""
        self._print(
            f"    {Colors.CYAN}📝 Prompt built for {ticker} {date}{Colors.RESET} "
            f"({len(prompt):,} chars)",
            NORMAL
        )

        # DETAILED: show prompt preview (first 500 chars)
        preview = prompt[:500].replace("\n", "\n      │ ")
        self._print(
            f"      ┌─ Prompt preview ──────────────────────────\n"
            f"      │ {preview}\n"
            f"      └─ ... ({len(prompt):,} chars total) ─────────",
            DETAILED
        )

        if self.store_inputs:
            path = os.path.join(self.prompts_dir, f"prompt_{ticker}_{date}.txt")
            with open(path, "w") as f:
                f.write(prompt)

    def log_llm_response(self, ticker: str, date: str, response: str,
                         decision: str, latency: float = None):
        """Log the LLM response and parsed decision."""
        latency_str = f" in {latency:.1f}s" if latency else ""
        decision_color = {
            "STRONG_BUY": Colors.GREEN, "BUY": Colors.GREEN,
            "HOLD": Colors.YELLOW,
            "SELL": Colors.RED, "STRONG_SELL": Colors.RED,
        }.get(decision, "")

        self._print(
            f"    {Colors.GREEN}🤖 LLM Decision:{Colors.RESET} "
            f"{decision_color}{Colors.BOLD}{decision}{Colors.RESET}{latency_str} "
            f"({len(response or ''):,} chars)",
            NORMAL
        )

        # DETAILED: show the conclusion section (the most useful part)
        if response:
            # Try to extract the <conclusion> section — that's where the reasoning is
            import re
            conclusion_match = re.search(
                r'<conclusion>(.*?)</conclusion>', response, re.DOTALL
            )
            if conclusion_match:
                preview_text = conclusion_match.group(1).strip()[:500]
                preview_label = "Conclusion"
            else:
                # Fall back to the last 400 chars (usually contains the decision rationale)
                preview_text = response[-400:].strip()
                preview_label = "Response (tail)"

            preview = preview_text.replace("\n", "\n      │ ")
            self._print(
                f"      ┌─ {preview_label} ─────────────────────────\n"
                f"      │ {preview}\n"
                f"      └─ ... ({len(response):,} chars total) ────",
                DETAILED
            )

        if self.store_inputs and response:
            path = os.path.join(self.responses_dir, f"response_{ticker}_{date}.txt")
            with open(path, "w") as f:
                f.write(f"DECISION: {decision}\n\n")
                f.write(response)

    def log_llm_error(self, ticker: str, date: str, error: str, attempt: int):
        """Log an LLM API error."""
        self._print(
            f"    {Colors.RED}✗ LLM Error (attempt {attempt}): {error}{Colors.RESET}",
            NORMAL
        )

    def log_rule_based_decision(self, ticker: str, date: str, decision: str,
                                 bullish: int, bearish: int):
        """Log a rule-based (non-LLM) decision."""
        decision_color = {
            "STRONG_BUY": Colors.GREEN, "BUY": Colors.GREEN,
            "HOLD": Colors.YELLOW,
            "SELL": Colors.RED, "STRONG_SELL": Colors.RED,
        }.get(decision, "")

        self._print(
            f"    {Colors.YELLOW}📊 Rule-based:{Colors.RESET} "
            f"{decision_color}{Colors.BOLD}{decision}{Colors.RESET} "
            f"(bullish:{bullish} bearish:{bearish})",
            NORMAL
        )

    # ── Snapshot Storage ──────────────────────────────────────────────────

    def store_snapshot(self, ticker: str, date: str, snapshot: Dict):
        """Store a complete data snapshot for later inspection."""
        if not self.store_inputs:
            return

        path = os.path.join(self.snapshots_dir, f"snapshot_{ticker}_{date}.json")
        # Convert non-serializable types
        clean = _make_serializable(snapshot)
        with open(path, "w") as f:
            json.dump(clean, f, indent=2, default=str)

        self._print(
            f"    {Colors.DIM}💾 Snapshot stored: {path}{Colors.RESET}",
            DETAILED
        )

    # ── Backtesting Logging ───────────────────────────────────────────────

    def log_rebalance(self, date: str, ticker: str, old_signal: str,
                      new_signal: str, old_weight: float, new_weight: float,
                      price: float, portfolio_value: float, tx_cost: float):
        """Log a portfolio rebalance event."""
        if old_signal == new_signal and self.verbosity < DEBUG:
            return  # Skip no-change rebalances unless DEBUG

        arrow = "→" if old_signal != new_signal else "="
        weight_change = abs(new_weight - old_weight)

        decision_color = {
            "STRONG_BUY": Colors.GREEN, "BUY": Colors.GREEN,
            "HOLD": Colors.YELLOW,
            "SELL": Colors.RED, "STRONG_SELL": Colors.RED,
        }.get(new_signal, "")

        self._print(
            f"    {Colors.DIM}{date}{Colors.RESET}  "
            f"{old_signal:>12s} {arrow} {decision_color}{new_signal:<12s}{Colors.RESET}  "
            f"wt: {old_weight:+.1f}→{new_weight:+.1f}  "
            f"price: ${price:.2f}  "
            f"portfolio: {_fmt_dollars(portfolio_value)}  "
            f"tx: ${tx_cost:.2f}",
            DETAILED
        )

        self._audit("rebalance", {
            "date": date, "ticker": ticker,
            "old_signal": old_signal, "new_signal": new_signal,
            "old_weight": old_weight, "new_weight": new_weight,
            "price": price, "portfolio_value": portfolio_value,
            "tx_cost": tx_cost,
        })

    def log_daily_pnl(self, date: str, ticker: str, price: float,
                      daily_return: float, portfolio_value: float,
                      position_weight: float):
        """Log daily P&L (DEBUG only — very verbose)."""
        color = Colors.GREEN if daily_return >= 0 else Colors.RED
        self._print(
            f"      {Colors.DIM}{date}{Colors.RESET}  "
            f"${price:.2f}  "
            f"ret: {color}{daily_return*100:+.3f}%{Colors.RESET}  "
            f"pos: {position_weight:+.1f}  "
            f"val: {_fmt_dollars(portfolio_value)}",
            DEBUG
        )

    def log_backtest_metrics(self, ticker: str, cr: float, sr: float,
                              hr: float, mdd: float, num_trades: int):
        """Log final backtest metrics for a ticker."""
        cr_color = Colors.GREEN if cr >= 0 else Colors.RED
        sr_color = Colors.GREEN if sr >= 0 else Colors.RED

        self._print(
            f"\n    {Colors.BOLD}Results for {ticker}:{Colors.RESET}",
            MINIMAL
        )
        self._print(
            f"      CR:  {cr_color}{cr*100:+.2f}%{Colors.RESET}  |  "
            f"SR:  {sr_color}{sr:.2f}{Colors.RESET}  |  "
            f"HR:  {hr*100:.1f}%  |  "
            f"MDD: {Colors.RED}{mdd*100:.2f}%{Colors.RESET}  |  "
            f"Trades: {num_trades}",
            MINIMAL
        )

    # ── Summary Table ─────────────────────────────────────────────────────

    def log_summary_table(self, summary_df: pd.DataFrame):
        """Print a formatted summary table."""
        self._print(
            f"\n{Colors.BOLD}{Colors.CYAN}"
            f"{'═'*60}\n"
            f"  FINAL RESULTS\n"
            f"{'═'*60}{Colors.RESET}",
            MINIMAL
        )
        self._print(summary_df.to_string(), MINIMAL)

    # ── Audit Trail ───────────────────────────────────────────────────────

    def _audit(self, event: str, data: Dict = None):
        """Add an event to the audit trail."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "event": event,
            "data": data or {},
        }
        self.audit_trail.append(entry)
        self._file_only(f"AUDIT: {event} | {json.dumps(data or {}, default=str)}")


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _make_serializable(obj):
    """Convert numpy/pandas types to JSON-serializable Python types."""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_make_serializable(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict()
    elif isinstance(obj, pd.Series):
        return obj.to_dict()
    return obj
