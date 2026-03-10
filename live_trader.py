#!/usr/bin/env python3
"""
Live Trading with Claude + Performance Memory

Maximizes Claude Opus/Sonnet for real-time trading decisions WITHOUT RL training.
Instead of gradient updates, Claude gets structured self-performance feedback —
a rolling window of past trades with realized P&L, win/loss stats by action type,
and calibration insights.

Architecture:
  1. TradeMemory: Persistent JSON store of past trades and outcomes
  2. LiveTrader: Orchestrates data → signals → Claude → decision → logging
  3. Performance Memory Prompt: Compact prompt section appended to standard prompt

Daily Workflow:
  Day 0 (first run):
    - Collect today's data (prices, technicals, fundamentals, news)
    - Generate S1 signal (mechanical baseline)
    - Call Claude with market data + empty performance memory
    - Log decision + reasoning → trade_memory.json

  Day 1+ (subsequent runs):
    - Load trade_memory.json
    - Settle completed trades: compute realized P&L for trades whose
      holding period (5 trading days) has elapsed
    - Build performance memory section from completed trades
    - Collect today's data → call Claude with data + memory → log decision

Usage:
  # First run — no history yet
  python live_trader.py --tickers AAPL MSFT NVDA

  # Use Opus for maximum reasoning
  python live_trader.py --tickers AAPL --model claude-opus-4-20250514

  # Dry run (don't save to memory, just print decisions)
  python live_trader.py --tickers AAPL --dry-run

  # Show current portfolio state
  python live_trader.py --status
"""
import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd

import config
from data_collector import DataCollector
from signal_generator import SignalGenerator, generate_signals, get_signal_for_date
from llm_analyst import (
    SYSTEM_PROMPT,
    build_data_prompt,
    call_claude,
    parse_decision,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Trade Memory — Persistent Storage of Past Trades
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class MarketContext:
    """
    Condensed snapshot of market conditions at decision time.
    Stored alongside each trade so Claude can learn PATTERNS —
    not just outcomes — across its trading history.

    Total size: ~150 tokens per trade. For 10 recent trades = ~1500 tokens.
    """
    # Price context
    price: float = 0.0
    change_1d_pct: float = 0.0
    change_5d_pct: float = 0.0
    change_15d_pct: float = 0.0

    # Key technicals (the ones that matter most for 5-day decisions)
    rsi: Optional[float] = None
    macd_histogram: Optional[float] = None        # Positive = bullish momentum
    adx: Optional[float] = None                   # Trend strength
    bb_position: Optional[str] = None             # "above_upper" / "near_upper" / "middle" / "near_lower" / "below_lower"
    sma_alignment: Optional[str] = None           # "bullish" (50>200, price>50) / "bearish" / "mixed"
    stoch_k: Optional[float] = None
    atr_pct: Optional[float] = None               # ATR as % of price (volatility regime)

    # Fundamentals snapshot
    sector: Optional[str] = None
    pe_ratio: Optional[float] = None
    market_cap_label: Optional[str] = None        # "$2.8T" compact format

    # Analyst sentiment
    analyst_consensus: Optional[float] = None     # 1-5 scale

    # Top news headlines (just headlines, no summaries — compact)
    headlines: List[str] = field(default_factory=list)   # Top 5 headlines


@dataclass
class MacroSnapshot:
    """
    Broad world-state vector at decision time — ORTHOGONAL to ticker-specific data.

    Captures macro regime, cross-asset signals, sector rotation, and global
    headlines. The model may discover butterfly-effect correlations that humans
    wouldn't think to look for: e.g., "when VIX>25 AND oil drops >3% AND
    10Y yield falls, tech bounces within a week."

    Budget: ~100 tokens per trade (coarse, compressed).
    Combined with MarketContext (~150 tokens), total = ~250 tokens per trade.
    Full history: 252 trades/year × 5 years × 250 tokens = ~315K tokens.
    Fits comfortably in Claude's 200K-1M context window.
    """
    # ── Market-wide pulse (from index/ETF prices) ────────────────────────
    spy_change_1d: Optional[float] = None         # S&P 500 daily % change
    qqq_change_1d: Optional[float] = None         # Nasdaq-100 daily % change
    vix: Optional[float] = None                   # VIX level (fear gauge)
    yield_10y: Optional[float] = None             # 10Y Treasury yield %
    yield_10y_change_1d: Optional[float] = None   # 10Y yield daily bps change
    dxy_change_1d: Optional[float] = None         # US Dollar Index daily % change

    # ── Cross-asset signals (commodities, risk-on/risk-off) ──────────────
    oil_change_1d: Optional[float] = None         # Crude oil (CL=F) daily %
    gold_change_1d: Optional[float] = None        # Gold (GC=F) daily %

    # ── Sector rotation pulse (sector ETF daily changes) ─────────────────
    sector_changes: Dict[str, float] = field(default_factory=dict)
    # e.g. {"Tech": +1.2, "Energy": -0.8, "Financials": +0.3, ...}

    # ── Macro regime labels (derived) ────────────────────────────────────
    risk_regime: Optional[str] = None             # "risk_on" / "risk_off" / "mixed"
    vol_regime: Optional[str] = None              # "low" (<15) / "normal" (15-25) / "high" (>25) / "extreme" (>35)

    # ── Global macro headlines (NOT ticker-specific) ─────────────────────
    # Economy, geopolitics, central bank, regulation, commodities, etc.
    global_headlines: List[str] = field(default_factory=list)  # Top 5


@dataclass
class TradeEntry:
    """A single trade decision with optional realized outcome."""
    date: str                        # Decision date (YYYY-MM-DD)
    ticker: str                      # Stock ticker
    signal_label: str                # Algorithm S1 signal
    decision: str                    # Claude's decision (STRONG_SELL..STRONG_BUY)
    position_weight: float           # Mapped position weight (-1.0 to +1.0)
    entry_price: float               # Price at decision time
    reasoning_summary: str           # 1-2 sentence summary of Claude's thesis
    model_used: str                  # Which Claude model was used

    # Market context at decision time (ticker-specific patterns)
    market_context: Optional[Dict] = None         # Serialized MarketContext
    # Macro context at decision time (broad world state — orthogonal)
    macro_context: Optional[Dict] = None          # Serialized MacroSnapshot

    # ── Execution tracking ─────────────────────────────────────────────────
    # Tracks whether the user ACTUALLY followed this recommendation.
    # New trades start as "pending" — user must confirm via --confirm.
    # Only confirmed-executed trades are used for settlement and learning.
    execution_status: str = "pending"             # "pending" | "executed" | "skipped" | "partial"
    actual_entry_price: Optional[float] = None    # User's real fill price (None = same as model's)
    actual_position_weight: Optional[float] = None  # User's real size (None = same as recommended)
    execution_note: Optional[str] = None          # Free-text note ("filled at market open", etc.)

    # Realized outcome fields (filled after holding period)
    exit_price: Optional[float] = None
    realized_pnl_pct: Optional[float] = None    # Model P&L: position-weighted using RECOMMENDED entry/weight (for learning)
    raw_pnl_pct: Optional[float] = None         # Raw stock return % (using recommended entry price)
    executed_pnl_pct: Optional[float] = None     # Executed P&L: position-weighted using ACTUAL fill (for bookkeeping; None if skipped)
    exit_date: Optional[str] = None
    is_settled: bool = False
    direction_correct: Optional[bool] = None     # Did direction match?
    was_override: bool = False                    # Did Claude override S1?
    override_correct: Optional[bool] = None      # Was the override better?


class TradeMemory:
    """
    Persistent store of all trading decisions and outcomes.
    Stored as JSON for simplicity and human-readability.
    """

    def __init__(self, memory_path: str = "trade_memory.json"):
        self.memory_path = memory_path
        self.trades: List[TradeEntry] = []
        self._load()

    def _load(self):
        """Load existing trade memory from disk."""
        if os.path.exists(self.memory_path):
            try:
                with open(self.memory_path, "r") as f:
                    data = json.load(f)
                self.trades = [TradeEntry(**t) for t in data.get("trades", [])]
                logger.info(f"Loaded {len(self.trades)} trades from {self.memory_path}")
            except Exception as e:
                logger.error(f"Error loading trade memory: {e}")
                self.trades = []
        else:
            self.trades = []

    def save(self):
        """Persist trade memory to disk."""
        data = {"trades": [asdict(t) for t in self.trades],
                "last_updated": datetime.now().isoformat()}
        os.makedirs(os.path.dirname(self.memory_path) if os.path.dirname(self.memory_path) else ".", exist_ok=True)
        with open(self.memory_path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        logger.info(f"Saved {len(self.trades)} trades to {self.memory_path}")

    def add_trade(self, trade: TradeEntry):
        """Add a new trade decision."""
        self.trades.append(trade)

    def get_unsettled(self) -> List[TradeEntry]:
        """Get trades that haven't been settled yet."""
        return [t for t in self.trades if not t.is_settled]

    def get_settled(self) -> List[TradeEntry]:
        """Get trades with realized outcomes."""
        return [t for t in self.trades if t.is_settled]

    def get_recent(self, n: int = None) -> List[TradeEntry]:
        """Get the N most recent settled trades (default: all)."""
        settled = self.get_settled()
        if n is not None:
            return settled[-n:] if settled else []
        return settled

    def get_trades_for_ticker(self, ticker: str) -> List[TradeEntry]:
        """Get all trades for a specific ticker."""
        return [t for t in self.trades if t.ticker == ticker]

    def get_pending_confirmation(self) -> List[TradeEntry]:
        """Get trades that haven't been confirmed by the user yet."""
        return [t for t in self.trades if t.execution_status == "pending"]

    def get_executed(self) -> List[TradeEntry]:
        """Get trades that were actually executed (confirmed or partial)."""
        return [t for t in self.trades if t.execution_status in ("executed", "partial")]

    def settle_trades(self, price_data: Dict[str, pd.DataFrame], holding_days: int = 5):
        """
        Settle unsettled trades whose holding period has elapsed.

        DUAL-TRACK APPROACH:
        ALL recommendations are settled against actual market prices for LEARNING,
        regardless of execution_status. This way Claude learns from every thesis
        it generates — even ones the user skipped.

        The execution_status field is preserved for bookkeeping (what the user
        actually traded), but it does NOT gate learning/settlement.

        For executed/partial trades: uses actual_entry_price/actual_position_weight
        when available (for bookkeeping P&L). Model P&L always uses recommended values.
        """
        today = pd.Timestamp.now().normalize()
        settled_count = 0

        for trade in self.get_unsettled():
            entry_date = pd.Timestamp(trade.date)

            # Check if we have enough trading days elapsed
            if trade.ticker not in price_data:
                continue

            prices = price_data[trade.ticker]
            if prices.empty:
                continue

            # Find trading days after entry
            future_dates = prices.index[prices.index > entry_date]
            if len(future_dates) < holding_days:
                continue  # Not enough trading days elapsed

            # Get exit price at holding_days after entry
            exit_date = future_dates[holding_days - 1]
            exit_price = float(prices.loc[exit_date, "close"])

            # ── Model P&L (always uses recommended entry/weight — for LEARNING) ──
            model_return = (exit_price - trade.entry_price) / trade.entry_price
            model_position_return = trade.position_weight * model_return

            trade.exit_price = exit_price
            trade.exit_date = exit_date.strftime("%Y-%m-%d")
            trade.raw_pnl_pct = round(model_return * 100, 3)
            trade.realized_pnl_pct = round(model_position_return * 100, 3)
            trade.is_settled = True

            # Direction check (based on model's recommendation)
            actual_direction = 1 if model_return > 0 else (-1 if model_return < 0 else 0)
            predicted_direction = 1 if trade.position_weight > 0 else (-1 if trade.position_weight < 0 else 0)
            trade.direction_correct = (predicted_direction == actual_direction) if predicted_direction != 0 else None

            # ── Executed P&L (uses actual fill if available — for BOOKKEEPING) ──
            # Stored separately so we can show both tracks
            if trade.execution_status in ("executed", "partial"):
                eff_entry = trade.actual_entry_price if trade.actual_entry_price is not None else trade.entry_price
                eff_weight = trade.actual_position_weight if trade.actual_position_weight is not None else trade.position_weight
                exec_return = (exit_price - eff_entry) / eff_entry
                trade.executed_pnl_pct = round(eff_weight * exec_return * 100, 3)
            else:
                # Not executed — no real money P&L
                trade.executed_pnl_pct = None

            # Override analysis
            trade.was_override = (trade.decision != trade.signal_label)
            if trade.was_override and trade.direction_correct is not None:
                # Would S1 have been correct?
                s1_direction = config.POSITION_WEIGHTS.get(trade.signal_label, 0)
                s1_direction_sign = 1 if s1_direction > 0 else (-1 if s1_direction < 0 else 0)
                s1_correct = (s1_direction_sign == actual_direction) if s1_direction_sign != 0 else None
                # Override was good if Claude was right and S1 was wrong, or Claude captured more
                if s1_correct is not None and trade.direction_correct is not None:
                    trade.override_correct = trade.direction_correct and not s1_correct

            settled_count += 1

        if settled_count > 0:
            logger.info(f"Settled {settled_count} trades")

        return settled_count


# ═══════════════════════════════════════════════════════════════════════════════
# Market Context Extraction — Captures Snapshot at Decision Time
# ═══════════════════════════════════════════════════════════════════════════════

def _fmt_market_cap(val) -> Optional[str]:
    """Format market cap to compact label."""
    if val is None:
        return None
    if abs(val) >= 1e12:
        return f"${val/1e12:.1f}T"
    if abs(val) >= 1e9:
        return f"${val/1e9:.1f}B"
    if abs(val) >= 1e6:
        return f"${val/1e6:.0f}M"
    return f"${val:,.0f}"


def extract_market_context(snapshot: Dict, news: Dict = None) -> Dict:
    """
    Extract a condensed market context from the full data snapshot.
    This is stored alongside each trade for pattern learning.

    Returns a dict (serializable) representing a MarketContext.
    """
    price = snapshot.get("price_summary", {})
    tech = snapshot.get("technical_indicators", {})
    fund = snapshot.get("fundamentals", {})
    recs = snapshot.get("analyst_recommendations", {})

    # ── Bollinger Band position ──────────────────────────────────────────
    bb_pos = None
    current_price = price.get("current_price")
    bb_upper = tech.get("bb_upper")
    bb_lower = tech.get("bb_lower")
    bb_middle = tech.get("bb_middle")
    if current_price and bb_upper and bb_lower:
        if current_price > bb_upper:
            bb_pos = "above_upper"
        elif current_price > bb_middle and bb_upper:
            pct_to_upper = (bb_upper - current_price) / (bb_upper - bb_middle) if bb_upper != bb_middle else 0.5
            bb_pos = "near_upper" if pct_to_upper < 0.3 else "upper_half"
        elif current_price < bb_lower:
            bb_pos = "below_lower"
        elif current_price < bb_middle and bb_lower:
            pct_to_lower = (current_price - bb_lower) / (bb_middle - bb_lower) if bb_middle != bb_lower else 0.5
            bb_pos = "near_lower" if pct_to_lower < 0.3 else "lower_half"
        else:
            bb_pos = "middle"

    # ── SMA alignment ────────────────────────────────────────────────────
    sma_align = None
    sma50 = tech.get("sma_50")
    sma200 = tech.get("sma_200")
    if sma50 is not None and sma200 is not None and current_price is not None:
        if sma50 > sma200 and current_price > sma50:
            sma_align = "bullish"
        elif sma50 < sma200 and current_price < sma50:
            sma_align = "bearish"
        else:
            sma_align = "mixed"

    # ── ATR as % of price (volatility regime) ────────────────────────────
    atr = tech.get("atr")
    atr_pct = round(atr / current_price * 100, 2) if atr and current_price else None

    # ── Top news headlines ───────────────────────────────────────────────
    headlines = []
    if news:
        for bucket_name in ["last_3_days", "last_4_10_days", "last_11_30_days"]:
            articles = news.get(bucket_name, [])
            for art in articles:
                hl = art.get("headline", "").strip()
                if hl and len(hl) > 10:
                    # Truncate individual headlines to save space
                    headlines.append(hl[:100])
                if len(headlines) >= 5:
                    break
            if len(headlines) >= 5:
                break

    ctx = MarketContext(
        price=current_price or 0.0,
        change_1d_pct=price.get("price_change_1d", 0.0),
        change_5d_pct=price.get("price_change_5d", 0.0),
        change_15d_pct=price.get("price_change_15d", 0.0),
        rsi=round(tech["rsi"], 1) if tech.get("rsi") is not None else None,
        macd_histogram=round(tech["macd_histogram"], 4) if tech.get("macd_histogram") is not None else None,
        adx=round(tech["adx"], 1) if tech.get("adx") is not None else None,
        bb_position=bb_pos,
        sma_alignment=sma_align,
        stoch_k=round(tech["stoch_k"], 1) if tech.get("stoch_k") is not None else None,
        atr_pct=atr_pct,
        sector=fund.get("sector"),
        pe_ratio=round(fund["pe_ratio"], 1) if fund.get("pe_ratio") is not None else None,
        market_cap_label=_fmt_market_cap(fund.get("market_cap")),
        analyst_consensus=recs.get("consensus_score"),
        headlines=headlines,
    )

    return asdict(ctx)


def format_context_for_prompt(ctx_dict: Optional[Dict], compact: bool = False) -> str:
    """
    Format a stored market context dict into a prompt-friendly string.

    Args:
        ctx_dict: Serialized MarketContext dict from trade_memory.json
        compact: If True, use single-line format for aggregate views
    """
    if not ctx_dict:
        return "    (no market context stored)"

    lines = []

    if compact:
        # Single-line format for aggregate/recent trade lists
        tech_parts = []
        if ctx_dict.get("rsi") is not None:
            tech_parts.append(f"RSI={ctx_dict['rsi']:.0f}")
        if ctx_dict.get("adx") is not None:
            tech_parts.append(f"ADX={ctx_dict['adx']:.0f}")
        if ctx_dict.get("macd_histogram") is not None:
            macd_dir = "+" if ctx_dict["macd_histogram"] > 0 else "-"
            tech_parts.append(f"MACD={macd_dir}")
        if ctx_dict.get("bb_position"):
            tech_parts.append(f"BB={ctx_dict['bb_position']}")
        if ctx_dict.get("sma_alignment"):
            tech_parts.append(f"SMA={ctx_dict['sma_alignment']}")
        if ctx_dict.get("atr_pct") is not None:
            tech_parts.append(f"ATR={ctx_dict['atr_pct']:.1f}%")

        chg = f"1d={ctx_dict.get('change_1d_pct', 0):+.1f}% 5d={ctx_dict.get('change_5d_pct', 0):+.1f}%"
        tech_str = " | ".join(tech_parts) if tech_parts else "N/A"
        lines.append(f"    Price: ${ctx_dict.get('price', 0):.2f} ({chg}) | {tech_str}")

        # Headlines on one line
        headlines = ctx_dict.get("headlines", [])
        if headlines:
            lines.append(f"    News: {headlines[0][:80]}")
            if len(headlines) > 1:
                lines.append(f"          {headlines[1][:80]}")

    else:
        # Full format for ticker-specific detailed view
        lines.append(f"    Price: ${ctx_dict.get('price', 0):.2f}")
        lines.append(
            f"    Changes: 1d={ctx_dict.get('change_1d_pct', 0):+.2f}% | "
            f"5d={ctx_dict.get('change_5d_pct', 0):+.2f}% | "
            f"15d={ctx_dict.get('change_15d_pct', 0):+.2f}%"
        )

        # Technicals
        tech_lines = []
        if ctx_dict.get("rsi") is not None:
            rsi = ctx_dict["rsi"]
            rsi_label = "oversold" if rsi < 30 else ("overbought" if rsi > 70 else "neutral")
            tech_lines.append(f"RSI={rsi:.0f} ({rsi_label})")
        if ctx_dict.get("adx") is not None:
            adx = ctx_dict["adx"]
            adx_label = "strong_trend" if adx > 25 else "weak/range"
            tech_lines.append(f"ADX={adx:.0f} ({adx_label})")
        if ctx_dict.get("macd_histogram") is not None:
            macd_dir = "bullish" if ctx_dict["macd_histogram"] > 0 else "bearish"
            tech_lines.append(f"MACD={macd_dir}")
        if ctx_dict.get("bb_position"):
            tech_lines.append(f"BB={ctx_dict['bb_position']}")
        if ctx_dict.get("sma_alignment"):
            tech_lines.append(f"SMA={ctx_dict['sma_alignment']}")
        if ctx_dict.get("stoch_k") is not None:
            tech_lines.append(f"Stoch_K={ctx_dict['stoch_k']:.0f}")
        if ctx_dict.get("atr_pct") is not None:
            vol_label = "high_vol" if ctx_dict["atr_pct"] > 3.0 else ("low_vol" if ctx_dict["atr_pct"] < 1.0 else "normal_vol")
            tech_lines.append(f"ATR={ctx_dict['atr_pct']:.1f}% ({vol_label})")

        if tech_lines:
            lines.append(f"    Technicals: {' | '.join(tech_lines)}")

        # Fundamentals
        fund_parts = []
        if ctx_dict.get("sector"):
            fund_parts.append(f"Sector={ctx_dict['sector']}")
        if ctx_dict.get("pe_ratio") is not None:
            fund_parts.append(f"P/E={ctx_dict['pe_ratio']:.1f}")
        if ctx_dict.get("market_cap_label"):
            fund_parts.append(f"MCap={ctx_dict['market_cap_label']}")
        if ctx_dict.get("analyst_consensus") is not None:
            fund_parts.append(f"Analyst={ctx_dict['analyst_consensus']:.1f}/5")
        if fund_parts:
            lines.append(f"    Fundamentals: {' | '.join(fund_parts)}")

        # Headlines
        headlines = ctx_dict.get("headlines", [])
        if headlines:
            lines.append(f"    Headlines at decision time:")
            for hl in headlines[:5]:
                lines.append(f"      - {hl[:90]}")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# Macro Snapshot — Broad World State Capture
# ═══════════════════════════════════════════════════════════════════════════════

# Tickers for macro data capture
MACRO_TICKERS = {
    "indices": {"SPY": "S&P 500", "QQQ": "Nasdaq-100"},
    "volatility": {"^VIX": "VIX"},
    "rates": {"^TNX": "10Y Yield"},
    "currencies": {"DX-Y.NYB": "US Dollar Index"},
    "commodities": {"CL=F": "Crude Oil", "GC=F": "Gold"},
    "sectors": {
        "XLK": "Tech", "XLF": "Financials", "XLE": "Energy",
        "XLV": "Healthcare", "XLY": "Consumer Disc", "XLP": "Consumer Staples",
        "XLU": "Utilities",
    },
}

# Search queries for capturing broad, orthogonal global headlines
GLOBAL_NEWS_QUERIES = [
    "economy recession inflation GDP",
    "Federal Reserve interest rate decision",
    "geopolitics trade war sanctions",
    "oil energy commodities supply",
    "China Japan Europe economy markets",
]


def _fetch_1d_change(ticker_symbol: str) -> Optional[float]:
    """Fetch the latest 1-day % change for a Yahoo Finance ticker."""
    try:
        import yfinance as yf
        data = yf.Ticker(ticker_symbol).history(period="5d")
        if data.empty or len(data) < 2:
            return None
        return round(
            (data["Close"].iloc[-1] - data["Close"].iloc[-2]) / data["Close"].iloc[-2] * 100, 2
        )
    except Exception:
        return None


def _fetch_latest_price(ticker_symbol: str) -> Optional[float]:
    """Fetch the latest closing price for a Yahoo Finance ticker."""
    try:
        import yfinance as yf
        data = yf.Ticker(ticker_symbol).history(period="5d")
        if data.empty:
            return None
        return round(float(data["Close"].iloc[-1]), 2)
    except Exception:
        return None


def _fetch_global_headlines() -> List[str]:
    """
    Fetch broad global macro headlines from Google News RSS.
    These are NOT about any specific ticker — they capture economy, geopolitics,
    central bank policy, commodities, and other orthogonal world events.
    """
    headlines = []
    seen = set()  # Deduplicate

    try:
        from data_collector import HAS_FEEDPARSER
        if not HAS_FEEDPARSER:
            return []

        import feedparser
        import requests
        from urllib.parse import quote_plus

        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/120.0.0.0 Safari/537.36"
        }

        for query in GLOBAL_NEWS_QUERIES:
            if len(headlines) >= 7:
                break
            try:
                url = f"https://news.google.com/rss/search?q={quote_plus(query)}&hl=en-US&gl=US&ceid=US:en"
                resp = requests.get(url, headers=headers, timeout=8)
                resp.raise_for_status()
                feed = feedparser.parse(resp.text)

                for entry in feed.entries[:3]:
                    hl = entry.get("title", "").strip()
                    # Deduplicate by first 40 chars
                    key = hl[:40].lower()
                    if hl and len(hl) > 15 and key not in seen:
                        headlines.append(hl[:120])
                        seen.add(key)
                    if len(headlines) >= 7:
                        break
            except Exception:
                continue

    except Exception:
        pass

    return headlines[:7]


def capture_macro_snapshot() -> Dict:
    """
    Capture a broad world-state snapshot: indices, volatility, rates,
    currencies, commodities, sector rotation, and global headlines.

    Returns a serialized MacroSnapshot dict.

    This runs ~15 lightweight Yahoo Finance calls + 5 RSS fetches.
    Total latency: ~5-10 seconds (parallelizable in future).
    """
    logger.info("Capturing macro snapshot...")

    # ── Market-wide indicators ───────────────────────────────────────────
    spy_chg = _fetch_1d_change("SPY")
    qqq_chg = _fetch_1d_change("QQQ")
    vix = _fetch_latest_price("^VIX")
    yield_10y = _fetch_latest_price("^TNX")
    yield_10y_chg = _fetch_1d_change("^TNX")
    dxy_chg = _fetch_1d_change("DX-Y.NYB")

    # ── Cross-asset signals ──────────────────────────────────────────────
    oil_chg = _fetch_1d_change("CL=F")
    gold_chg = _fetch_1d_change("GC=F")

    # ── Sector rotation ──────────────────────────────────────────────────
    sector_changes = {}
    for etf_ticker, sector_name in MACRO_TICKERS["sectors"].items():
        chg = _fetch_1d_change(etf_ticker)
        if chg is not None:
            sector_changes[sector_name] = chg

    # ── Derive regime labels ─────────────────────────────────────────────
    # Risk regime: based on VIX + SPY direction + gold direction
    risk_regime = "mixed"
    if vix is not None and spy_chg is not None:
        if vix < 18 and spy_chg > 0:
            risk_regime = "risk_on"
        elif vix > 25 or (spy_chg is not None and spy_chg < -1.0):
            risk_regime = "risk_off"

    # Vol regime from VIX level
    vol_regime = None
    if vix is not None:
        if vix < 15:
            vol_regime = "low"
        elif vix < 25:
            vol_regime = "normal"
        elif vix < 35:
            vol_regime = "high"
        else:
            vol_regime = "extreme"

    # ── Global macro headlines ───────────────────────────────────────────
    global_headlines = _fetch_global_headlines()

    snapshot = MacroSnapshot(
        spy_change_1d=spy_chg,
        qqq_change_1d=qqq_chg,
        vix=vix,
        yield_10y=yield_10y,
        yield_10y_change_1d=yield_10y_chg,
        dxy_change_1d=dxy_chg,
        oil_change_1d=oil_chg,
        gold_change_1d=gold_chg,
        sector_changes=sector_changes,
        risk_regime=risk_regime,
        vol_regime=vol_regime,
        global_headlines=global_headlines,
    )

    logger.info(
        f"  Macro: SPY={spy_chg}% VIX={vix} 10Y={yield_10y}% "
        f"Oil={oil_chg}% Gold={gold_chg}% Regime={risk_regime}/{vol_regime} "
        f"Headlines={len(global_headlines)}"
    )

    return asdict(snapshot)


def format_macro_for_prompt(macro_dict: Optional[Dict], compact: bool = False) -> str:
    """
    Format a macro snapshot dict into a prompt-friendly string.

    Args:
        macro_dict: Serialized MacroSnapshot dict
        compact: If True, single-line format for recent trade lists
    """
    if not macro_dict:
        return "    (no macro context stored)"

    lines = []

    if compact:
        # One-line macro summary for recent trade list
        parts = []
        if macro_dict.get("spy_change_1d") is not None:
            parts.append(f"SPY={macro_dict['spy_change_1d']:+.1f}%")
        if macro_dict.get("vix") is not None:
            parts.append(f"VIX={macro_dict['vix']:.0f}")
        if macro_dict.get("yield_10y") is not None:
            parts.append(f"10Y={macro_dict['yield_10y']:.2f}%")
        if macro_dict.get("oil_change_1d") is not None:
            parts.append(f"Oil={macro_dict['oil_change_1d']:+.1f}%")
        if macro_dict.get("gold_change_1d") is not None:
            parts.append(f"Gold={macro_dict['gold_change_1d']:+.1f}%")
        if macro_dict.get("risk_regime"):
            parts.append(f"Regime={macro_dict['risk_regime']}")

        macro_str = " | ".join(parts) if parts else "N/A"
        lines.append(f"    Macro: {macro_str}")

        # Top global headline
        g_headlines = macro_dict.get("global_headlines", [])
        if g_headlines:
            lines.append(f"    World: {g_headlines[0][:85]}")

    else:
        # Full format for ticker-specific detailed view
        lines.append(f"    --- Macro Environment ---")

        # Market-wide
        idx_parts = []
        if macro_dict.get("spy_change_1d") is not None:
            idx_parts.append(f"SPY={macro_dict['spy_change_1d']:+.2f}%")
        if macro_dict.get("qqq_change_1d") is not None:
            idx_parts.append(f"QQQ={macro_dict['qqq_change_1d']:+.2f}%")
        if macro_dict.get("vix") is not None:
            vix = macro_dict["vix"]
            vol_label = macro_dict.get("vol_regime", "")
            idx_parts.append(f"VIX={vix:.1f} ({vol_label})")
        if idx_parts:
            lines.append(f"    Indices: {' | '.join(idx_parts)}")

        # Rates & currencies
        rc_parts = []
        if macro_dict.get("yield_10y") is not None:
            chg = macro_dict.get("yield_10y_change_1d")
            chg_str = f" ({chg:+.1f}bps)" if chg is not None else ""
            rc_parts.append(f"10Y={macro_dict['yield_10y']:.2f}%{chg_str}")
        if macro_dict.get("dxy_change_1d") is not None:
            rc_parts.append(f"DXY={macro_dict['dxy_change_1d']:+.2f}%")
        if rc_parts:
            lines.append(f"    Rates/FX: {' | '.join(rc_parts)}")

        # Commodities
        cmd_parts = []
        if macro_dict.get("oil_change_1d") is not None:
            cmd_parts.append(f"Oil={macro_dict['oil_change_1d']:+.2f}%")
        if macro_dict.get("gold_change_1d") is not None:
            cmd_parts.append(f"Gold={macro_dict['gold_change_1d']:+.2f}%")
        if cmd_parts:
            lines.append(f"    Commodities: {' | '.join(cmd_parts)}")

        # Sector rotation
        sector_changes = macro_dict.get("sector_changes", {})
        if sector_changes:
            sorted_sectors = sorted(sector_changes.items(), key=lambda x: x[1], reverse=True)
            sector_strs = [f"{name}={chg:+.1f}%" for name, chg in sorted_sectors]
            lines.append(f"    Sectors: {' | '.join(sector_strs)}")

        # Risk regime
        if macro_dict.get("risk_regime"):
            lines.append(f"    Risk regime: {macro_dict['risk_regime']}")

        # Global headlines
        g_headlines = macro_dict.get("global_headlines", [])
        if g_headlines:
            lines.append(f"    Global headlines:")
            for hl in g_headlines[:5]:
                lines.append(f"      - {hl[:100]}")

    return "\n".join(lines)


def _build_current_macro_prompt(macro_dict: Dict) -> str:
    """
    Build a prompt section for TODAY's live macro environment.
    This is appended to the current analysis prompt (not the history).
    """
    lines = ["\n<current_macro_environment>"]
    lines.append("Today's broad market state (use for cross-asset context):")
    lines.append(format_macro_for_prompt(macro_dict, compact=False))
    lines.append("</current_macro_environment>")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# Market Diary — Continuous Daily Snapshots (Independent of Trades)
# ═══════════════════════════════════════════════════════════════════════════════

class MarketDiary:
    """
    Stores daily macro + ticker snapshots INDEPENDENTLY of trading decisions.
    This fills the blind spots when the user doesn't run the full analysis.

    Two data sources:
      - "capture": real-time daily snapshots (includes headlines)
      - "backfill": reconstructed from Yahoo Finance (no headlines)

    Storage: market_diary.json
    {
      "entries": {
        "2025-03-01": {
          "macro": { ... MacroSnapshot dict ... },
          "tickers": { "AAPL": { price, rsi, ... }, "MSFT": { ... } },
          "source": "capture"
        }
      }
    }
    """

    def __init__(self, diary_path: str = "market_diary.json"):
        self.diary_path = diary_path
        self.entries: Dict[str, Dict] = {}
        self._load()

    def _load(self):
        if os.path.exists(self.diary_path):
            try:
                with open(self.diary_path, "r") as f:
                    data = json.load(f)
                self.entries = data.get("entries", {})
                logger.info(f"Loaded {len(self.entries)} diary entries from {self.diary_path}")
            except Exception as e:
                logger.error(f"Error loading market diary: {e}")
                self.entries = {}

    def save(self):
        data = {
            "entries": self.entries,
            "last_updated": datetime.now().isoformat(),
        }
        os.makedirs(os.path.dirname(self.diary_path) if os.path.dirname(self.diary_path) else ".", exist_ok=True)
        with open(self.diary_path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        logger.info(f"Saved {len(self.entries)} diary entries to {self.diary_path}")

    def add_entry(self, date_str: str, macro: Dict, tickers: Dict = None, source: str = "capture"):
        self.entries[date_str] = {
            "macro": macro,
            "tickers": tickers or {},
            "source": source,
        }

    def get_entry(self, date_str: str) -> Optional[Dict]:
        return self.entries.get(date_str)

    def get_range(self, start_date: str, end_date: str) -> List[Tuple[str, Dict]]:
        """Get diary entries in a date range (inclusive), sorted by date."""
        results = []
        for date_str, entry in sorted(self.entries.items()):
            if start_date <= date_str <= end_date:
                results.append((date_str, entry))
        return results

    def get_missing_dates(self, start_date: str, end_date: str) -> List[str]:
        """Get trading days in range that have no diary entry."""
        trading_days = pd.bdate_range(start_date, end_date)
        return [d.strftime("%Y-%m-%d") for d in trading_days if d.strftime("%Y-%m-%d") not in self.entries]

    def get_last_entry_date(self) -> Optional[str]:
        """Get the most recent diary entry date."""
        if not self.entries:
            return None
        return max(self.entries.keys())


def capture_daily_snapshot(
    tickers: List[str],
    diary: MarketDiary,
    date_str: str = None,
) -> Dict:
    """
    Lightweight daily capture: macro + per-ticker snapshots.
    NO Claude calls — just data collection.

    Run this daily via cron or manually to maintain continuous coverage.
    Cost: ~15 Yahoo Finance calls + 5 RSS fetches ≈ 5-10 seconds.
    """
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")

    # Skip if already captured today
    existing = diary.get_entry(date_str)
    if existing and existing.get("source") == "capture":
        print(f"  Already captured {date_str} — skipping")
        return existing

    print(f"  Capturing daily snapshot for {date_str}...")

    # 1. Macro snapshot
    macro = capture_macro_snapshot()

    # 2. Per-ticker snapshots (lightweight: just price + key technicals)
    ticker_snapshots = {}
    for ticker in tickers:
        try:
            end = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
            start = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")

            dc = DataCollector(ticker, start, end)
            dc.collect_all()

            snapshot = dc.get_snapshot(date_str)
            if snapshot:
                # Extract condensed context (reuses existing function)
                news = dc.get_news_for_date(date_str)
                ctx = extract_market_context(snapshot, news)
                ticker_snapshots[ticker] = ctx
                print(f"    {ticker}: ${ctx.get('price', 0):.2f} RSI={ctx.get('rsi', '?')}")
            else:
                print(f"    {ticker}: no data for {date_str}")
        except Exception as e:
            logger.warning(f"    {ticker}: capture failed — {e}")

    diary.add_entry(date_str, macro, ticker_snapshots, source="capture")
    diary.save()

    print(f"  Captured: macro + {len(ticker_snapshots)} tickers")
    return diary.get_entry(date_str)


def backfill_diary(
    tickers: List[str],
    diary: MarketDiary,
    start_date: str = None,
    end_date: str = None,
):
    """
    Reconstruct daily snapshots for missed days from Yahoo Finance.

    Headlines are NOT available for past dates, but all numeric data
    (prices, VIX, yields, sectors, technicals) is fully reconstructed.

    This fills the blind spot when the user forgets to run --capture.
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    if start_date is None:
        # Default: backfill from last diary entry (or 30 days ago)
        last = diary.get_last_entry_date()
        if last:
            start_date = (pd.Timestamp(last) + timedelta(days=1)).strftime("%Y-%m-%d")
        else:
            start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

    missing = diary.get_missing_dates(start_date, end_date)
    if not missing:
        print(f"  No missing dates between {start_date} and {end_date}")
        return 0

    print(f"\n  Backfilling {len(missing)} missing days: {missing[0]} to {missing[-1]}")
    print(f"  Note: Headlines not available for past dates (prices/technicals only)")

    # Fetch bulk historical data for efficiency
    import yfinance as yf

    # Macro tickers: bulk download
    all_macro_symbols = ["SPY", "QQQ", "^VIX", "^TNX", "DX-Y.NYB", "CL=F", "GC=F"]
    sector_symbols = list(MACRO_TICKERS["sectors"].keys())
    all_symbols = all_macro_symbols + sector_symbols + list(tickers)

    print(f"  Downloading history for {len(all_symbols)} symbols...")
    history_start = (pd.Timestamp(missing[0]) - timedelta(days=5)).strftime("%Y-%m-%d")
    history_end = (pd.Timestamp(missing[-1]) + timedelta(days=1)).strftime("%Y-%m-%d")

    bulk_data = {}
    for sym in all_symbols:
        try:
            data = yf.Ticker(sym).history(start=history_start, end=history_end)
            if not data.empty:
                bulk_data[sym] = data
        except Exception:
            pass

    print(f"  Got data for {len(bulk_data)}/{len(all_symbols)} symbols")

    # Reconstruct each missing day
    filled = 0
    for date_str in missing:
        dt = pd.Timestamp(date_str)

        # ── Reconstruct macro ────────────────────────────────────────────
        def _get_change(sym):
            if sym not in bulk_data:
                return None
            d = bulk_data[sym]
            mask = d.index.normalize() <= dt
            recent = d[mask]
            if len(recent) < 2:
                return None
            return round((recent["Close"].iloc[-1] - recent["Close"].iloc[-2]) / recent["Close"].iloc[-2] * 100, 2)

        def _get_price(sym):
            if sym not in bulk_data:
                return None
            d = bulk_data[sym]
            mask = d.index.normalize() <= dt
            recent = d[mask]
            if recent.empty:
                return None
            return round(float(recent["Close"].iloc[-1]), 2)

        spy_chg = _get_change("SPY")
        qqq_chg = _get_change("QQQ")
        vix = _get_price("^VIX")
        yield_10y = _get_price("^TNX")
        yield_10y_chg = _get_change("^TNX")
        dxy_chg = _get_change("DX-Y.NYB")
        oil_chg = _get_change("CL=F")
        gold_chg = _get_change("GC=F")

        sector_changes = {}
        for etf, name in MACRO_TICKERS["sectors"].items():
            chg = _get_change(etf)
            if chg is not None:
                sector_changes[name] = chg

        # Derive regimes
        risk_regime = "mixed"
        if vix is not None and spy_chg is not None:
            if vix < 18 and spy_chg > 0:
                risk_regime = "risk_on"
            elif vix > 25 or (spy_chg is not None and spy_chg < -1.0):
                risk_regime = "risk_off"

        vol_regime = None
        if vix is not None:
            if vix < 15: vol_regime = "low"
            elif vix < 25: vol_regime = "normal"
            elif vix < 35: vol_regime = "high"
            else: vol_regime = "extreme"

        macro = asdict(MacroSnapshot(
            spy_change_1d=spy_chg, qqq_change_1d=qqq_chg,
            vix=vix, yield_10y=yield_10y, yield_10y_change_1d=yield_10y_chg,
            dxy_change_1d=dxy_chg, oil_change_1d=oil_chg, gold_change_1d=gold_chg,
            sector_changes=sector_changes,
            risk_regime=risk_regime, vol_regime=vol_regime,
            global_headlines=[],  # Can't reconstruct headlines for past dates
        ))

        # ── Reconstruct per-ticker snapshots ──────────────────────────────
        ticker_snapshots = {}
        for ticker in tickers:
            if ticker not in bulk_data:
                continue
            d = bulk_data[ticker]
            mask = d.index.normalize() <= dt
            recent = d[mask]
            if len(recent) < 2:
                continue

            price = float(recent["Close"].iloc[-1])
            chg_1d = round((recent["Close"].iloc[-1] - recent["Close"].iloc[-2]) / recent["Close"].iloc[-2] * 100, 2)

            # 5d and 15d changes
            chg_5d = round((recent["Close"].iloc[-1] - recent["Close"].iloc[-min(5, len(recent))]) / recent["Close"].iloc[-min(5, len(recent))] * 100, 2) if len(recent) >= 5 else None
            chg_15d = round((recent["Close"].iloc[-1] - recent["Close"].iloc[-min(15, len(recent))]) / recent["Close"].iloc[-min(15, len(recent))] * 100, 2) if len(recent) >= 15 else None

            # Simple RSI calculation (14-day)
            rsi = None
            if len(recent) >= 15:
                delta = recent["Close"].diff()
                gain = delta.where(delta > 0, 0).rolling(14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                if loss.iloc[-1] != 0:
                    rs = gain.iloc[-1] / loss.iloc[-1]
                    rsi = round(100 - (100 / (1 + rs)), 1)

            ticker_snapshots[ticker] = {
                "price": price,
                "change_1d_pct": chg_1d,
                "change_5d_pct": chg_5d or 0.0,
                "change_15d_pct": chg_15d or 0.0,
                "rsi": rsi,
                "headlines": [],  # Not available for backfill
            }

        diary.add_entry(date_str, macro, ticker_snapshots, source="backfill")
        filled += 1

    diary.save()
    print(f"  Backfilled {filled} days")
    return filled


def format_diary_for_prompt(diary: MarketDiary, start_date: str, end_date: str) -> str:
    """
    Format diary entries between two dates into a prompt section.
    Used to show Claude what happened during gaps between trades.
    """
    entries = diary.get_range(start_date, end_date)
    if not entries:
        return ""

    lines = [f"\n[MARKET DIARY — {len(entries)} days of coverage]"]

    for date_str, entry in entries:
        macro = entry.get("macro", {})
        source = entry.get("source", "?")
        source_marker = "" if source == "capture" else " [backfill]"

        # Compact one-line macro summary
        parts = []
        if macro.get("spy_change_1d") is not None:
            parts.append(f"SPY={macro['spy_change_1d']:+.1f}%")
        if macro.get("vix") is not None:
            parts.append(f"VIX={macro['vix']:.0f}")
        if macro.get("yield_10y") is not None:
            parts.append(f"10Y={macro['yield_10y']:.2f}%")
        if macro.get("risk_regime"):
            parts.append(f"{macro['risk_regime']}")

        macro_str = " | ".join(parts) if parts else "no data"

        # Ticker prices
        tickers_data = entry.get("tickers", {})
        ticker_parts = []
        for tk, ctx in sorted(tickers_data.items()):
            p = ctx.get("price", 0)
            chg = ctx.get("change_1d_pct", 0)
            ticker_parts.append(f"{tk}=${p:.0f}({chg:+.1f}%)")
        ticker_str = " ".join(ticker_parts) if ticker_parts else ""

        lines.append(f"  {date_str}: {macro_str} | {ticker_str}{source_marker}")

        # Include top global headline if available (capture only)
        headlines = macro.get("global_headlines", [])
        if headlines:
            lines.append(f"             {headlines[0][:80]}")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# Trading Journal — Structured Daily Decision Rationale
# ═══════════════════════════════════════════════════════════════════════════════

class TradingJournal:
    """
    Stores structured daily trading rationales as readable markdown files.

    Each trading day produces one journal file:
      journals/2025-03-09.md

    The journal captures Claude's full decision-making process in a top-down
    structure that the user can read to understand WHY decisions were made.
    """

    def __init__(self, journal_dir: str = "journals"):
        self.journal_dir = journal_dir
        os.makedirs(journal_dir, exist_ok=True)

    def get_journal_path(self, date_str: str) -> str:
        return os.path.join(self.journal_dir, f"{date_str}.md")

    def exists(self, date_str: str) -> bool:
        return os.path.exists(self.get_journal_path(date_str))

    def write_journal(
        self,
        date_str: str,
        macro_snapshot: Dict,
        ticker_entries: List[Dict],
        method: str = "single-agent",
        model: str = "",
    ):
        """
        Write the daily journal file.

        Args:
            date_str: Trading date (YYYY-MM-DD)
            macro_snapshot: Today's macro environment
            ticker_entries: List of per-ticker analysis dicts from analyze_ticker/analyze_ticker_abm
            method: "single-agent" or "ABM"
            model: Claude model used
        """
        lines = []

        # ── Header ──────────────────────────────────────────────────────────
        lines.append(f"# Trading Journal — {date_str}")
        lines.append(f"")
        lines.append(f"**Model:** {model}  ")
        lines.append(f"**Method:** {method}  ")
        lines.append(f"**Tickers analyzed:** {', '.join(r['ticker'] for r in ticker_entries if 'error' not in r)}  ")
        lines.append(f"")

        # ── Executive Summary ───────────────────────────────────────────────
        lines.append(f"## Executive Summary")
        lines.append(f"")

        # Quick decision table
        lines.append(f"| Ticker | S1 Signal | Decision | Weight | Price | Override |")
        lines.append(f"|--------|-----------|----------|--------|-------|----------|")
        for r in ticker_entries:
            if "error" in r:
                lines.append(f"| {r['ticker']} | ERROR | — | — | — | — |")
                continue
            override = "YES" if r.get("was_override") else "no"
            lines.append(
                f"| {r['ticker']} | {r['signal_label']} | **{r['decision']}** | "
                f"{r['position_weight']:+.1f} | ${r['entry_price']:.2f} | {override} |"
            )
        lines.append(f"")

        # ── Macro Environment ───────────────────────────────────────────────
        lines.append(f"## Macro Environment")
        lines.append(f"")
        if macro_snapshot:
            spy = macro_snapshot.get("spy_change_1d")
            qqq = macro_snapshot.get("qqq_change_1d")
            vix = macro_snapshot.get("vix")
            y10 = macro_snapshot.get("yield_10y")
            risk = macro_snapshot.get("risk_regime", "?")
            vol = macro_snapshot.get("vol_regime", "?")

            lines.append(f"- **Risk regime:** {risk} | **Volatility regime:** {vol}")
            parts = []
            if spy is not None: parts.append(f"SPY {spy:+.1f}%")
            if qqq is not None: parts.append(f"QQQ {qqq:+.1f}%")
            if vix is not None: parts.append(f"VIX {vix:.1f}")
            if y10 is not None: parts.append(f"10Y yield {y10:.2f}%")
            lines.append(f"- **Key levels:** {' | '.join(parts)}")

            # Sector performance
            sectors = macro_snapshot.get("sector_changes", {})
            if sectors:
                leaders = sorted(sectors.items(), key=lambda x: x[1], reverse=True)[:3]
                laggards = sorted(sectors.items(), key=lambda x: x[1])[:3]
                lines.append(f"- **Sector leaders:** {', '.join(f'{s} ({v:+.1f}%)' for s, v in leaders)}")
                lines.append(f"- **Sector laggards:** {', '.join(f'{s} ({v:+.1f}%)' for s, v in laggards)}")

            # Global headlines
            headlines = macro_snapshot.get("global_headlines", [])
            if headlines:
                lines.append(f"- **Key headlines:**")
                for hl in headlines[:5]:
                    lines.append(f"  - {hl[:120]}")
        lines.append(f"")

        # ── Per-Ticker Deep Dives ───────────────────────────────────────────
        for r in ticker_entries:
            if "error" in r:
                continue

            ticker = r["ticker"]
            lines.append(f"---")
            lines.append(f"")
            lines.append(f"## {ticker}: {r['decision']} (weight {r['position_weight']:+.1f})")
            lines.append(f"")

            # Decision vs S1
            if r.get("was_override"):
                lines.append(f"> **Override:** S1 signaled {r['signal_label']} but decided **{r['decision']}**")
            else:
                lines.append(f"> **Aligned with S1:** Both agree on {r['decision']}")
            lines.append(f"")

            # Full thesis
            thesis = r.get("full_thesis", "")
            if thesis:
                lines.append(f"### Full Analysis")
                lines.append(f"")
                # Clean up XML tags for readability
                cleaned = self._clean_thesis_for_journal(thesis)
                lines.append(cleaned)
                lines.append(f"")

            # ABM agent breakdown (if ABM mode)
            agents = r.get("abm_agents", [])
            if agents:
                lines.append(f"### Agent Debate")
                lines.append(f"")
                lines.append(f"| Agent | Role | Decision | Confidence | Key Thesis |")
                lines.append(f"|-------|------|----------|------------|------------|")
                for ag in agents:
                    p = ag.get("parsed", {})
                    d = p.get("decision", "?")
                    c = p.get("confidence", "?")
                    thesis_short = (p.get("thesis") or "")[:80].replace("\n", " ").replace("|", "/")
                    lines.append(f"| {ag['name']} | {ag['role']} | **{d}** | {c}/5 | {thesis_short} |")
                lines.append(f"")

                # Agent details
                for ag in agents:
                    p = ag.get("parsed", {})
                    lines.append(f"#### {ag['name']} ({ag['role']})")
                    lines.append(f"")
                    if p.get("thesis"):
                        lines.append(f"**Thesis:** {p['thesis']}")
                    if p.get("key_risk"):
                        lines.append(f"")
                        lines.append(f"**Key risk:** {p['key_risk']}")
                    lines.append(f"")

                # Moderator synthesis
                mod = r.get("abm_moderator", {})
                if mod:
                    lines.append(f"### Moderator Synthesis")
                    lines.append(f"")
                    lines.append(f"**Final decision:** {mod.get('decision', '?')} (confidence {mod.get('confidence', '?')}/5)  ")
                    if mod.get("reasoning"):
                        lines.append(f"**Reasoning:** {mod['reasoning']}")
                    cws = mod.get("confidence_weighted_score")
                    if cws is not None:
                        lines.append(f"")
                        lines.append(f"**Confidence-weighted directional score:** {cws:+.2f}")
                    lines.append(f"")

            # Reasoning summary (always present)
            if r.get("reasoning_summary"):
                lines.append(f"### Decision Summary")
                lines.append(f"")
                lines.append(f"{r['reasoning_summary']}")
                lines.append(f"")

        # ── Footer ──────────────────────────────────────────────────────────
        lines.append(f"---")
        lines.append(f"")
        lines.append(f"*Generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")

        # Write file
        path = self.get_journal_path(date_str)
        with open(path, "w") as f:
            f.write("\n".join(lines))

        print(f"\n  Journal saved: {path}")
        return path

    def _clean_thesis_for_journal(self, thesis: str) -> str:
        """
        Convert Claude's XML-tagged analysis into readable markdown.
        Extracts key sections and formats them as nested headers.
        """
        import re

        sections = []

        # Map XML tags to readable section headers
        tag_headers = [
            ("technical_analysis", "Technical Analysis"),
            ("fundamental_analysis", "Fundamental Analysis"),
            ("news_analysis", "News & Sentiment Analysis"),
            ("macro_analysis", "Macro Analysis"),
            ("risk_assessment", "Risk Assessment"),
            ("performance_review", "Performance Review"),
            ("synthesis", "Synthesis"),
            ("conclusion", "Conclusion"),
            ("alternatives_considered", "Alternatives Considered"),
            ("key_risks", "Key Risks"),
            ("lessons_applied", "Lessons Applied"),
        ]

        found_tags = False
        for tag, header in tag_headers:
            match = re.search(rf'<{tag}>(.*?)</{tag}>', thesis, re.DOTALL)
            if match:
                content = match.group(1).strip()
                if content:
                    sections.append(f"**{header}:**\n{content}")
                    found_tags = True

        if found_tags:
            return "\n\n".join(sections)

        # Fallback: if no XML tags, return cleaned thesis text
        # Remove known XML tags but keep content
        cleaned = re.sub(r'</?[a-z_]+>', '', thesis)
        # Trim excessive whitespace
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned).strip()
        return cleaned

    def list_journals(self, last_n: int = None) -> List[str]:
        """List available journal dates, most recent first."""
        files = sorted(
            [f.replace(".md", "") for f in os.listdir(self.journal_dir) if f.endswith(".md")],
            reverse=True,
        )
        return files[:last_n] if last_n else files


# ═══════════════════════════════════════════════════════════════════════════════
# Performance Memory Prompt — Structured Feedback to Claude
# ═══════════════════════════════════════════════════════════════════════════════

def build_performance_memory(memory: TradeMemory, current_ticker: str, diary: MarketDiary = None) -> str:
    """
    Build a compact performance memory section for Claude's prompt.

    DUAL-TRACK APPROACH:
    - MODEL PERFORMANCE: All recommendations settled against market prices.
      This is the LEARNING signal — Claude learns from every thesis regardless
      of whether the user executed it.
    - EXECUTED PERFORMANCE: Only trades the user actually took.
      This is the BOOKKEEPING signal — real portfolio P&L.

    The gap between model and executed performance is itself informative.
    """
    settled = memory.get_settled()
    if not settled:
        return "\n<performance_memory>\nNo prior trading history. This is your first decision.\nFollow Algorithm S1 unless you have strong independent evidence for an override.\n</performance_memory>"

    # Split into executed vs skipped for dual-track
    executed_trades = [t for t in settled if t.execution_status in ("executed", "partial")]
    skipped_trades = [t for t in settled if t.execution_status == "skipped"]
    pending_trades = [t for t in settled if t.execution_status == "pending"]  # legacy/unconfirmed

    sections = ["\n<performance_memory>"]

    # ── MODEL PERFORMANCE (ALL recommendations — the LEARNING track) ─────
    # Claude learns from EVERY thesis it generated, executed or not.
    total = len(settled)
    wins = sum(1 for t in settled if t.realized_pnl_pct and t.realized_pnl_pct > 0)
    losses = sum(1 for t in settled if t.realized_pnl_pct and t.realized_pnl_pct < 0)
    flat = total - wins - losses

    pnls = [t.realized_pnl_pct for t in settled if t.realized_pnl_pct is not None]
    mean_pnl = np.mean(pnls) if pnls else 0
    total_pnl = sum(pnls)

    direction_correct = sum(1 for t in settled if t.direction_correct)
    direction_total = sum(1 for t in settled if t.direction_correct is not None)

    sections.append("[MODEL PERFORMANCE — all recommendations (learning track)]")
    sections.append(f"  Total recommendations: {total} (executed: {len(executed_trades)}, skipped: {len(skipped_trades)}, unconfirmed: {len(pending_trades)})")
    if total > 0:
        sections.append(f"  Win/Loss/Flat: {wins}/{losses}/{flat} ({wins/total*100:.0f}% win rate)")
    sections.append(f"  Model P&L: cumulative={total_pnl:+.2f}% | mean={mean_pnl:+.3f}%")
    if direction_total > 0:
        sections.append(f"  Direction accuracy: {direction_correct}/{direction_total} ({direction_correct/direction_total*100:.0f}%)")

    # ── EXECUTED PERFORMANCE (user's actual portfolio — bookkeeping track) ─
    if executed_trades:
        exec_pnls = [t.executed_pnl_pct if t.executed_pnl_pct is not None else t.realized_pnl_pct
                      for t in executed_trades if (t.executed_pnl_pct is not None or t.realized_pnl_pct is not None)]
        exec_wins = sum(1 for p in exec_pnls if p > 0)
        exec_total_pnl = sum(exec_pnls)
        exec_mean_pnl = np.mean(exec_pnls) if exec_pnls else 0

        sections.append(f"\n[EXECUTED PERFORMANCE — trades user actually took (bookkeeping track)]")
        sections.append(f"  Executed trades: {len(executed_trades)}/{total}")
        if exec_pnls:
            sections.append(f"  Win/Loss: {exec_wins}/{len(exec_pnls) - exec_wins} ({exec_wins/len(exec_pnls)*100:.0f}% win rate)")
            sections.append(f"  Executed P&L: cumulative={exec_total_pnl:+.2f}% | mean={exec_mean_pnl:+.3f}%")

        # ── Gap analysis: model vs executed ──────────────────────────────
        if exec_pnls and pnls:
            gap = total_pnl - exec_total_pnl
            if abs(gap) > 0.5:
                if gap > 0:
                    sections.append(f"  >> Gap: Model outperforms executed by {gap:+.2f}% — you may be skipping good calls")
                else:
                    sections.append(f"  >> Gap: Executed outperforms model by {-gap:+.2f}% — your trade selection adds value")

    # ── Skipped trade analysis ───────────────────────────────────────────
    if skipped_trades:
        skip_pnls = [t.realized_pnl_pct for t in skipped_trades if t.realized_pnl_pct is not None]
        if skip_pnls:
            skip_wins = sum(1 for p in skip_pnls if p > 0)
            skip_mean = np.mean(skip_pnls)
            sections.append(f"\n[SKIPPED TRADE OUTCOMES — what would have happened]")
            sections.append(f"  Skipped: {len(skipped_trades)} trades | Would-have-been win rate: {skip_wins/len(skip_pnls)*100:.0f}% | Mean P&L: {skip_mean:+.3f}%")
            if skip_mean >= mean_pnl and skip_mean > 0:
                sections.append(f"  >> WARNING: Skipped trades would have outperformed (mean={skip_mean:+.2f}%). Reconsider skip criteria.")
            elif skip_mean < -0.5:
                sections.append(f"  >> Good filtering: skipped trades were net losers. Your skip instincts are valuable.")

    # ── Per-Action Stats (all recommendations — for learning) ────────────
    action_stats: Dict[str, List[float]] = {}
    for t in settled:
        if t.realized_pnl_pct is not None:
            action_stats.setdefault(t.decision, []).append(t.realized_pnl_pct)

    if action_stats:
        sections.append("\n[PERFORMANCE BY ACTION TYPE (all recommendations)]")
        for action in config.ACTION_LABELS:
            if action in action_stats:
                pnls_a = action_stats[action]
                wins_a = sum(1 for p in pnls_a if p > 0)
                sections.append(
                    f"  {action:>12s}: {len(pnls_a)} trades | "
                    f"mean={np.mean(pnls_a):+.3f}% | "
                    f"win_rate={wins_a/len(pnls_a)*100:.0f}% | "
                    f"best={max(pnls_a):+.3f}% worst={min(pnls_a):+.3f}%"
                )

    # ── Override Analysis ────────────────────────────────────────────────
    overrides = [t for t in settled if t.was_override]
    if overrides:
        override_pnls = [t.realized_pnl_pct for t in overrides if t.realized_pnl_pct is not None]
        override_wins = sum(1 for p in override_pnls if p > 0)
        good_overrides = sum(1 for t in overrides if t.override_correct)
        sections.append(f"\n[OVERRIDE ANALYSIS (you disagreed with S1)]")
        sections.append(
            f"  Overrides: {len(overrides)}/{total} trades | "
            f"Override win rate: {override_wins/len(overrides)*100:.0f}% | "
            f"Truly better than S1: {good_overrides}/{len(overrides)}"
        )
        if override_wins / len(overrides) < 0.5:
            sections.append(f"  >> WARNING: Your overrides are net negative. Default to S1 unless evidence is compelling.")
        elif good_overrides / len(overrides) > 0.6:
            sections.append(f"  >> Your overrides add value. Continue exercising independent judgment.")

    # ── Ticker-Specific History (ALL recommendations, with full context) ──
    ticker_trades = [t for t in settled if t.ticker == current_ticker]
    if ticker_trades:
        ticker_pnls = [t.realized_pnl_pct for t in ticker_trades if t.realized_pnl_pct is not None]
        ticker_wins = sum(1 for p in ticker_pnls if p > 0)
        sections.append(f"\n[YOUR HISTORY WITH {current_ticker} — all recommendations with market context]")
        sections.append(
            f"  {len(ticker_trades)} past recommendations | "
            f"model win rate: {ticker_wins/len(ticker_trades)*100:.0f}% | "
            f"mean model P&L: {np.mean(ticker_pnls):+.3f}%"
        )
        # Show ALL recommendations for this ticker WITH full context
        for t in ticker_trades:
            pnl_str = f"{t.realized_pnl_pct:+.2f}%" if t.realized_pnl_pct else "pending"
            dir_str = "correct" if t.direction_correct else "wrong"
            override_str = " [OVERRIDE]" if t.was_override else ""

            # Execution status marker
            if t.execution_status == "skipped":
                exec_str = " [SKIPPED by user]"
            elif t.execution_status == "partial":
                exec_str = " [PARTIAL execution]"
            elif t.execution_status == "executed":
                exec_str = " [EXECUTED]"
            else:
                exec_str = ""  # pending/legacy — no marker

            # Show actual fill info if different
            actual_str = ""
            if t.execution_status in ("executed", "partial"):
                if t.actual_entry_price is not None:
                    actual_str += f" (actual fill: ${t.actual_entry_price:.2f})"
                if t.actual_position_weight is not None:
                    actual_str += f" (actual size: {t.actual_position_weight:+.1f})"
                if t.executed_pnl_pct is not None and t.executed_pnl_pct != t.realized_pnl_pct:
                    actual_str += f" (exec P&L: {t.executed_pnl_pct:+.2f}%)"

            sections.append(
                f"\n  --- {t.date} ---"
            )
            sections.append(
                f"    S1={t.signal_label} -> You={t.decision} | Model P&L={pnl_str} ({dir_str}){override_str}{exec_str}{actual_str}"
            )
            # Market context at decision time (ticker-specific)
            sections.append(format_context_for_prompt(t.market_context, compact=False))
            # Macro context at decision time (broad world state)
            sections.append(format_macro_for_prompt(t.macro_context, compact=False))
            if t.reasoning_summary:
                sections.append(f"    Your reasoning: {t.reasoning_summary[:150]}")

    # ── Full Cross-Ticker Trade Log (ALL settled, compact context) ────────
    if settled:
        sections.append(f"\n[ALL RECOMMENDATIONS ({len(settled)} total) — with market & macro context]")
        for t in settled:
            pnl_str = f"{t.realized_pnl_pct:+.2f}%" if t.realized_pnl_pct else "pending"
            override_marker = " [OVERRIDE]" if t.was_override else ""

            # Execution status marker
            if t.execution_status == "skipped":
                exec_marker = " [SKIP]"
            elif t.execution_status == "partial":
                exec_marker = " [PARTIAL]"
            elif t.execution_status == "executed":
                exec_marker = " [EXEC]"
            else:
                exec_marker = ""

            dir_str = ""
            if t.direction_correct is not None:
                dir_str = " | dir=correct" if t.direction_correct else " | dir=WRONG"
            sections.append(
                f"\n  {t.date} {t.ticker:>5s}: S1={t.signal_label:>12s} -> You={t.decision:>12s} "
                f"| Model P&L={pnl_str}{dir_str}{override_marker}{exec_marker}"
            )
            # Compact market context (technicals + top headline)
            sections.append(format_context_for_prompt(t.market_context, compact=True))
            # Compact macro context (SPY/VIX/yields + top global headline)
            sections.append(format_macro_for_prompt(t.macro_context, compact=True))

    # ── Calibration Insights ─────────────────────────────────────────────
    sections.append(f"\n[CALIBRATION GUIDANCE]")

    # Check if strong actions are calibrated (using ALL recommendations for learning)
    strong_actions = [t for t in settled if t.decision in ("STRONG_BUY", "STRONG_SELL")
                      and t.realized_pnl_pct is not None]
    moderate_actions = [t for t in settled if t.decision in ("BUY", "SELL")
                        and t.realized_pnl_pct is not None]

    if strong_actions:
        strong_mean = np.mean([t.realized_pnl_pct for t in strong_actions])
        strong_wins = sum(1 for t in strong_actions if t.realized_pnl_pct > 0)
        if strong_wins / len(strong_actions) < 0.55:
            sections.append(
                f"  Your STRONG BUY/SELL trades win only {strong_wins/len(strong_actions)*100:.0f}% "
                f"of the time (mean={strong_mean:+.2f}%). Consider using moderate sizing (BUY/SELL) "
                f"more often until calibration improves."
            )
        elif strong_mean > 0.5:
            sections.append(
                f"  Your STRONG actions are well-calibrated (win rate={strong_wins/len(strong_actions)*100:.0f}%, "
                f"mean={strong_mean:+.2f}%). Continue using high conviction when warranted."
            )

    if moderate_actions and strong_actions:
        moderate_mean = np.mean([t.realized_pnl_pct for t in moderate_actions])
        if moderate_mean > np.mean([t.realized_pnl_pct for t in strong_actions]):
            sections.append(
                f"  Moderate positions (BUY/SELL) outperform strong positions. "
                f"You may be over-sizing. Increase conviction threshold for STRONG actions."
            )

    # Overall direction bias check
    if direction_total >= 5:
        if direction_correct / direction_total < 0.45:
            sections.append(
                f"  Direction accuracy is low ({direction_correct/direction_total*100:.0f}%). "
                f"Consider following S1 more closely until accuracy improves."
            )

    # ── Market Diary: gap coverage between trades ─────────────────────────
    # Shows Claude what happened on days without trades — macro regime shifts,
    # VIX spikes, sector rotations, global headlines during gaps.
    if diary and settled:
        last_trade_date = settled[-1].date
        today = datetime.now().strftime("%Y-%m-%d")
        # Show diary entries from last trade to today (fills the gap)
        diary_section = format_diary_for_prompt(diary, last_trade_date, today)
        if diary_section:
            sections.append(f"\n{diary_section}")

    sections.append("</performance_memory>")

    return "\n".join(sections)


# Enhanced system prompt for live trading with performance memory
LIVE_SYSTEM_PROMPT = SYSTEM_PROMPT + """

<live_trading_context>
You are operating in LIVE TRADING MODE. Your decisions will be acted upon.

PERFORMANCE MEMORY (DUAL-TRACK):
You have access to a <performance_memory> section showing your past trading
outcomes WITH the full market context that existed at each decision. This is your
substitute for reinforcement learning — you can see exactly WHAT conditions led
to WHICH outcomes.

IMPORTANT — DUAL-TRACK LEARNING:
Your performance memory shows TWO tracks:
  - MODEL PERFORMANCE: Every recommendation you made, settled against actual
    market prices. This includes trades the user SKIPPED. Learn from ALL of them.
    A skipped trade that would have been profitable is a missed opportunity.
    A skipped trade that would have lost money validates the user's judgment.
  - EXECUTED PERFORMANCE: Only trades the user actually took. This is the real
    portfolio. The gap between model and executed P&L reveals how well the user
    filters your recommendations.

Trades are tagged [EXEC], [SKIP], or [PARTIAL] so you can see which were taken.
Learn from ALL recommendations — skipped trades are experiments you ran for free.

PATTERN LEARNING — USE YOUR HISTORY TO:

  TICKER-SPECIFIC PATTERNS (from <market_context> in your history):
  1. Recognize winning setups: Each past trade stores the exact technicals (RSI,
     MACD, ADX, BB, SMA), price momentum, and news at decision time. Compare
     today's setup to past winners. When conditions match, raise conviction.

  2. Avoid losing setups: If today resembles a past losing trade (e.g., "last time
     RSI was oversold but ADX was weak, I went STRONG BUY and the stock went
     sideways"), learn from that mistake.

  3. Calibrate by volatility: Past trades include ATR% (vol regime) and SMA
     alignment (trend state). If your wins cluster in trending markets but you
     lose in range-bound ones, adjust sizing accordingly.

  4. News-outcome association: Headlines stored per trade let you learn which
     news types (earnings, regulatory, analyst upgrades) precede wins vs losses.

  CROSS-DOMAIN / MACRO PATTERNS (from <macro_context> in your history):
  5. Macro regime recognition: Each past trade stores the BROAD world state —
     SPY/QQQ direction, VIX level, 10Y yield, dollar strength, oil/gold moves,
     sector rotation, and global headlines. Look for CROSS-ASSET correlations:
       - "When VIX > 25 AND oil dropped AND I went BUY on tech, I won/lost"
       - "When yields rose sharply AND dollar strengthened, my equity longs lost"
       - "When multiple sectors diverged (tech up, energy down), my trades in
         the weaker sector underperformed"

  6. Butterfly effects: Global headlines capture geopolitics, central bank
     actions, and economic data that may seem unrelated to a specific stock but
     create macro tailwinds or headwinds. Notice if patterns like "Fed hawkish
     signal → my tech longs lost within a week" recur.

  7. Sector rotation awareness: Past macro snapshots show which sectors were
     leading and lagging. If the current ticker's sector was previously a laggard
     when it appeared in your winning trades, that context matters.

  8. Today's live macro: A <current_macro_environment> section shows you the
     CURRENT broad market state. Compare it to past macro contexts on your
     winning and losing trades. If today's macro environment matches past
     losing environments, reduce sizing. If it matches past winners, that
     supports your thesis.

  CALIBRATION:
  9. Calibrate conviction: If STRONG BUY/SELL trades are underperforming,
     prefer moderate sizing until calibration improves.

  10. Learn from overrides: If overriding Algorithm S1 has been net negative,
      default to following S1 unless evidence is extreme.

  11. Ticker-specific learning: Your history with the current ticker is shown
      with full detail. If you've been consistently wrong on this stock, adjust.

RESPONSE FORMAT:
Your analysis MUST include these XML-tagged sections for journaling:

  <synthesis>
  Your main analysis integrating technicals, fundamentals, news, and macro.
  </synthesis>

  <alternatives_considered>
  What other decisions did you seriously consider? For each alternative:
  - What was the alternative? (e.g., "HOLD instead of BUY")
  - What evidence supported it?
  - Why did you reject it?
  This section is critical — it shows your reasoning depth and helps the user
  understand close calls vs. high-conviction decisions.
  </alternatives_considered>

  <key_risks>
  What could make this trade go wrong? Be specific:
  - The single biggest risk to your thesis
  - Any upcoming catalysts (earnings, Fed meeting, etc.) that could invalidate it
  - What price action would tell you this trade is wrong?
  </key_risks>

  <lessons_applied>
  What did you learn from your past performance memory that influenced THIS decision?
  - Specific past trades (by date/ticker) that shaped your approach
  - Patterns you noticed and how they applied here
  - If this is your first trade, say so — note what you'll watch for next time
  </lessons_applied>

  <conclusion>
  1-2 sentence summary of your final decision and the single strongest reason.
  </conclusion>

  [[[YOUR DECISION]]]

IMPORTANT — WHAT NOT TO DO:
  - Do not anchor on sunk costs. Past losses don't make a ticker "due" for a win.
  - Do not revenge-trade. A bad outcome doesn't mean the opposite is now correct.
  - Do not overfit to small samples. 3 past trades is suggestive, not conclusive.
  - Treat each decision independently, informed by — but not determined by — history.
  - If your history shows poor override performance, be honest about it. The cost
    of a bad override is double: you miss the S1 return AND take the wrong position.
  - Macro correlations with <5 data points are speculative. Note them but don't
    bet heavily on them until you see them repeat.
</live_trading_context>
"""


# ═══════════════════════════════════════════════════════════════════════════════
# ABM — Agent-Based Modeling for Multi-Perspective Trading
# ═══════════════════════════════════════════════════════════════════════════════

# Each agent is a Claude call with a distinct analytical lens.
# They all receive the same market data + performance memory, but their system
# prompts bias them toward different aspects of the analysis.
#
# Cost: 6 agents + 1 moderator = 7 calls per ticker.
# At Sonnet pricing (~$3/$15 per 1M in/out tokens):
#   ~35K input + ~14K output per ticker ≈ $0.30/ticker/day
#   3 tickers/day ≈ $1/day. Trivial for daily use.

AGENT_PERSONAS = [
    {
        "name": "Momentum",
        "role": "Momentum & Technical Trader",
        "prompt": """You are a MOMENTUM TRADER. Your edge comes from reading price action and technicals.

YOUR ANALYTICAL LENS:
- RSI, MACD histogram, ADX (trend strength) are your primary signals
- Price momentum (1d, 5d, 15d changes) tells you where the trend is
- SMA alignment (bullish/bearish/mixed) defines the regime you trade in
- Bollinger Band position flags mean-reversion vs breakout setups
- Stochastic K identifies short-term overbought/oversold extremes
- ATR% tells you how much the stock ACTUALLY moves — size accordingly

YOUR PHILOSOPHY:
- "The trend is your friend until it bends." Ride momentum.
- Strong trends (ADX > 25) with aligned SMAs deserve full conviction
- Weak/range-bound markets (ADX < 20) call for HOLD or reduced sizing
- MACD histogram direction change is your earliest reversal signal
- RSI divergences (price makes new high, RSI doesn't) are bearish warnings

YOU IGNORE OR DEPRIORITIZE:
- P/E ratios and valuations (that's the Value agent's job)
- News headlines (unless they cause a breakout/breakdown)
- Macro regime (unless it directly impacts the trend)

OUTPUT FORMAT — you MUST include these XML tags:
<agent_analysis>
<decision>STRONG_BUY or BUY or HOLD or SELL or STRONG_SELL</decision>
<confidence>1-5 integer</confidence>
<thesis>2-3 bullet points explaining your technical read</thesis>
<key_risk>One sentence: what could invalidate your thesis</key_risk>
</agent_analysis>""",
    },
    {
        "name": "Value",
        "role": "Fundamental & Value Investor",
        "prompt": """You are a VALUE INVESTOR. Your edge comes from understanding what a business is worth vs what the market prices it at.

YOUR ANALYTICAL LENS:
- P/E ratio relative to sector and historical norms
- Market cap trajectory and earnings quality
- Analyst consensus score — are professionals bullish or cautious?
- Sector fundamentals — is this sector in structural growth or decline?
- Mean reversion: stocks that have fallen far from fair value may be buys
- Stocks trading at extreme multiples with slowing growth may be sells

YOUR PHILOSOPHY:
- "Price is what you pay, value is what you get."
- If the stock is cheap by fundamentals and analysts are upgrading, BUY
- If the stock is expensive and momentum is the only bull case, be cautious
- Earnings quality matters more than price momentum
- Sector tailwinds/headwinds create value opportunities

YOU IGNORE OR DEPRIORITIZE:
- Short-term technical oscillators (RSI, Stochastic)
- Day-to-day price momentum — you think in quarters, not days
- Bollinger Bands and similar mean-reversion technicals

OUTPUT FORMAT — you MUST include these XML tags:
<agent_analysis>
<decision>STRONG_BUY or BUY or HOLD or SELL or STRONG_SELL</decision>
<confidence>1-5 integer</confidence>
<thesis>2-3 bullet points explaining your valuation case</thesis>
<key_risk>One sentence: what could invalidate your thesis</key_risk>
</agent_analysis>""",
    },
    {
        "name": "Macro",
        "role": "Global Macro Strategist",
        "prompt": """You are a GLOBAL MACRO STRATEGIST. Your edge comes from understanding how the broad market environment affects individual stocks.

YOUR ANALYTICAL LENS:
- SPY/QQQ direction: is the rising tide lifting all boats, or is it receding?
- VIX level and vol regime: low vol = complacency, high vol = opportunity or danger
- 10Y Treasury yield: rising yields pressure growth stocks, falling yields support them
- US Dollar (DXY): strong dollar hurts multinationals and commodities
- Oil and Gold: cross-asset signals for inflation, risk appetite, geopolitics
- Sector rotation: money flowing into/out of sectors signals regime shifts
- Global headlines: central bank policy, trade tensions, geopolitics create tailwinds/headwinds

YOUR PHILOSOPHY:
- "Don't fight the Fed." Monetary policy trumps individual stock stories.
- Risk regime (risk_on/risk_off) should influence ALL position sizing
- In risk_off (VIX > 25, SPY falling), even great stocks can drop — reduce longs
- In risk_on (VIX < 18, SPY rising), even mediocre stocks can rally — lean in
- Cross-asset divergences often resolve violently — watch for these
- A stock may look great technically but if macro headwinds are strong, fade it

YOU IGNORE OR DEPRIORITIZE:
- Stock-specific fundamentals (P/E, earnings) — that's micro, not macro
- Individual RSI/MACD signals — you care about the ENVIRONMENT, not the stock

OUTPUT FORMAT — you MUST include these XML tags:
<agent_analysis>
<decision>STRONG_BUY or BUY or HOLD or SELL or STRONG_SELL</decision>
<confidence>1-5 integer</confidence>
<thesis>2-3 bullet points on how macro environment affects this stock</thesis>
<key_risk>One sentence: what macro risk could hurt this position</key_risk>
</agent_analysis>""",
    },
    {
        "name": "Sentiment",
        "role": "News & Sentiment Analyst",
        "prompt": """You are a SENTIMENT ANALYST. Your edge comes from reading the information landscape — news flow, analyst actions, and market narrative.

YOUR ANALYTICAL LENS:
- News headlines: what stories are driving attention on this stock?
- Analyst consensus: are professionals upgrading or downgrading?
- Headline sentiment: is the narrative shifting bullish or bearish?
- News recency and intensity: are headlines fresh catalysts or stale news?
- Narrative divergence: when price moves one way but news suggests another, pay attention
- "Buy the rumor, sell the news" dynamics — anticipated events may already be priced in

YOUR PHILOSOPHY:
- "Markets are a voting machine in the short run." Sentiment drives price.
- Fresh positive catalysts (earnings beat, product launch, regulatory approval) = bullish
- Deteriorating sentiment (downgrades, lawsuits, regulatory risk) = bearish
- No news is often neutral news — HOLD unless technicals or macro override
- The INTENSITY of news matters: one headline is noise, five is a trend
- Contrarian signal: when sentiment is unanimously one-sided, the trade may be crowded

YOU IGNORE OR DEPRIORITIZE:
- Pure technical indicators without news context
- Macro regime unless directly reflected in headlines
- Fundamental valuations — you trade information flow, not spreadsheets

OUTPUT FORMAT — you MUST include these XML tags:
<agent_analysis>
<decision>STRONG_BUY or BUY or HOLD or SELL or STRONG_SELL</decision>
<confidence>1-5 integer</confidence>
<thesis>2-3 bullet points on what the information landscape signals</thesis>
<key_risk>One sentence: what could invalidate your sentiment read</key_risk>
</agent_analysis>""",
    },
    {
        "name": "Risk",
        "role": "Risk Manager",
        "prompt": """You are a RISK MANAGER. Your job is NOT to find trades — it's to SIZE them correctly and PROTECT the portfolio from catastrophic loss.

YOUR ANALYTICAL LENS:
- ATR% (volatility regime): high-vol stocks need smaller positions
- VIX level: portfolio-wide risk rises when VIX is elevated
- Max drawdown from past performance memory: are we in a drawdown? If so, reduce risk
- Position weight calibration: are STRONG_BUY/SELL decisions outperforming moderate ones?
- Correlation risk: if multiple positions are in the same direction/sector, net exposure rises
- Override history: if past overrides of S1 have been net negative, that's a risk flag

YOUR PHILOSOPHY:
- "Rule #1: Don't lose money. Rule #2: Don't forget Rule #1."
- The DIRECTION may be right but the SIZE may be wrong — that's your domain
- In high-vol (ATR% > 3%), downgrade STRONG to moderate even if the thesis is good
- In low-vol (ATR% < 1%), you CAN accept larger positions
- If performance memory shows poor calibration on strong actions, flag it
- Default position: you AGREE with the directional view of other agents but adjust SIZE
- You rarely override direction entirely — you trim, hedge, or reduce

YOUR UNIQUE ROLE:
- You may output the SAME direction as what the data suggests, but with lower confidence
  to signal "yes, but smaller"
- A confidence of 1-2 from you means "the risk isn't worth it at full size"
- A confidence of 4-5 means "risk/reward is favorable, full size OK"

OUTPUT FORMAT — you MUST include these XML tags:
<agent_analysis>
<decision>STRONG_BUY or BUY or HOLD or SELL or STRONG_SELL</decision>
<confidence>1-5 integer (YOUR confidence reflects risk-adjusted sizing)</confidence>
<thesis>2-3 bullet points on risk assessment and position sizing rationale</thesis>
<key_risk>One sentence: the single biggest risk to the portfolio from this trade</key_risk>
</agent_analysis>""",
    },
    {
        "name": "Contrarian",
        "role": "Contrarian & Devil's Advocate",
        "prompt": """You are the CONTRARIAN. Your job is to challenge the obvious trade and find what others are missing.

YOUR ANALYTICAL LENS:
- When technicals, fundamentals, AND sentiment all agree → the trade may be CROWDED
- When everyone is bullish, ask: who is left to buy?
- When everyone is bearish, ask: who is left to sell?
- Look for asymmetric risk: is the downside if wrong much larger than the upside if right?
- Historical pattern recognition: when past similar setups occurred, did the obvious trade work?
- Mean reversion: extreme moves in any direction tend to revert

YOUR PHILOSOPHY:
- "Be fearful when others are greedy, and greedy when others are fearful."
- You are NOT automatically bearish — you are automatically SKEPTICAL of consensus
- If 4 out of 5 other agents say BUY, you should seriously consider HOLD or SELL
- If the obvious trade is bearish (bad news, falling price), consider if it's OVERDONE
- Your best trades come when sentiment is at extremes — maximum bearishness = potential buy
- You add value by PREVENTING groupthink, not by always disagreeing

CRITICAL NUANCE:
- Don't be contrarian just for the sake of it. If the evidence overwhelmingly supports a direction AND positioning isn't crowded, you can agree.
- Your confidence should reflect how crowded/extreme the consensus is, not just whether you agree.
- Confidence 1-2 = consensus seems overcrowded, fade it
- Confidence 3 = consensus is reasonable, no strong view
- Confidence 4-5 = consensus is wrong, strong contrarian signal

OUTPUT FORMAT — you MUST include these XML tags:
<agent_analysis>
<decision>STRONG_BUY or BUY or HOLD or SELL or STRONG_SELL</decision>
<confidence>1-5 integer</confidence>
<thesis>2-3 bullet points challenging or validating the obvious trade</thesis>
<key_risk>One sentence: the contrarian risk (what if the crowd is right?)</key_risk>
</agent_analysis>""",
    },
]


# Moderator prompt — synthesizes all agent views into a final decision
ABM_MODERATOR_PROMPT = SYSTEM_PROMPT + """

<abm_moderator_context>
You are the MODERATOR in an Agent-Based Modeling (ABM) trading system.

You have received independent analyses from 6 specialist agents, each with a
different analytical lens. Your job is to SYNTHESIZE their views into a single,
well-reasoned trading decision.

SYNTHESIS PROTOCOL:

1. TALLY THE VOTES: Count how many agents are bullish, bearish, or neutral.
   Strong consensus (5-6 agree) → higher confidence in that direction.
   Split opinion (3-3) → lean toward HOLD or reduce sizing.

2. WEIGHT BY CONFIDENCE: An agent with confidence 5 should count more than
   one with confidence 2. Compute a confidence-weighted directional score.

3. RESPECT THE RISK MANAGER: If the Risk Manager flags high risk (low confidence)
   even when direction is agreed upon, REDUCE sizing from STRONG to moderate.

4. LISTEN TO THE CONTRARIAN: If the Contrarian has high confidence in the
   opposite direction of consensus, that's a WARNING. Don't ignore it.
   Consider reducing position size even if you maintain direction.

5. RESOLVE CONFLICTS EXPLICITLY: When Momentum says BUY but Value says SELL,
   explain WHY you side with one over the other in this specific context.

6. MATCH CONFIDENCE TO CONSENSUS QUALITY:
   - 6/6 agree with high confidence → STRONG action, high confidence
   - 5/6 agree, 1 dissent → moderate to strong action
   - 4/6 agree → moderate action
   - 3/3 split → HOLD unless one side has much higher confidence
   - Risk Manager flags danger → reduce sizing regardless

OUTPUT FORMAT — you MUST include these XML tags:
<moderator_synthesis>
<votes>
  List each agent's vote: Name: DECISION (confidence/5) — one-line thesis
</votes>
<consensus_direction>bullish/bearish/neutral (N of 6 agree)</consensus_direction>
<confidence_weighted_score>float from -5.0 (max bearish) to +5.0 (max bullish)</confidence_weighted_score>
<dissenting_views>Which agents disagree with consensus and why</dissenting_views>
<risk_assessment>Risk Manager's key concern and how you address it</risk_assessment>
<final_decision>STRONG_BUY or BUY or HOLD or SELL or STRONG_SELL</final_decision>
<final_confidence>1-5 integer</final_confidence>
<reasoning>2-3 sentences explaining your synthesis logic</reasoning>
</moderator_synthesis>

ALSO include DECISION: [your_decision] at the very end for parsing.
</abm_moderator_context>
"""


def _parse_agent_response(response_text: str) -> Dict:
    """Parse structured output from an agent's response."""
    import re

    result = {
        "decision": None,
        "confidence": 3,
        "thesis": "",
        "key_risk": "",
        "raw": response_text or "",
    }

    if not response_text:
        return result

    # Parse <agent_analysis> XML
    decision_match = re.search(r'<decision>\s*(.*?)\s*</decision>', response_text, re.DOTALL)
    if decision_match:
        d = decision_match.group(1).strip().upper()
        if d in config.ACTION_LABELS:
            result["decision"] = d

    conf_match = re.search(r'<confidence>\s*(\d)\s*</confidence>', response_text, re.DOTALL)
    if conf_match:
        result["confidence"] = int(conf_match.group(1))

    thesis_match = re.search(r'<thesis>\s*(.*?)\s*</thesis>', response_text, re.DOTALL)
    if thesis_match:
        result["thesis"] = thesis_match.group(1).strip()

    risk_match = re.search(r'<key_risk>\s*(.*?)\s*</key_risk>', response_text, re.DOTALL)
    if risk_match:
        result["key_risk"] = risk_match.group(1).strip()

    return result


def _parse_moderator_response(response_text: str) -> Dict:
    """Parse the moderator's synthesis response."""
    import re

    result = {
        "decision": None,
        "confidence": 3,
        "reasoning": "",
        "confidence_weighted_score": 0.0,
        "raw": response_text or "",
    }

    if not response_text:
        return result

    # Parse <moderator_synthesis> XML
    decision_match = re.search(r'<final_decision>\s*(.*?)\s*</final_decision>', response_text, re.DOTALL)
    if decision_match:
        d = decision_match.group(1).strip().upper()
        if d in config.ACTION_LABELS:
            result["decision"] = d

    conf_match = re.search(r'<final_confidence>\s*(\d)\s*</final_confidence>', response_text, re.DOTALL)
    if conf_match:
        result["confidence"] = int(conf_match.group(1))

    reasoning_match = re.search(r'<reasoning>\s*(.*?)\s*</reasoning>', response_text, re.DOTALL)
    if reasoning_match:
        result["reasoning"] = reasoning_match.group(1).strip()

    score_match = re.search(r'<confidence_weighted_score>\s*([-\d.]+)\s*</confidence_weighted_score>', response_text, re.DOTALL)
    if score_match:
        try:
            result["confidence_weighted_score"] = float(score_match.group(1))
        except ValueError:
            pass

    # Fallback: parse DECISION: line
    if not result["decision"]:
        fallback = re.search(r'DECISION:\s*(STRONG_BUY|BUY|HOLD|SELL|STRONG_SELL)', response_text)
        if fallback:
            result["decision"] = fallback.group(1)

    return result


def _build_agent_system_prompt(persona: Dict) -> str:
    """Build the full system prompt for a specialist agent."""
    return LIVE_SYSTEM_PROMPT + f"\n\n<agent_role>\n{persona['prompt']}\n</agent_role>"


def _build_moderator_data_prompt(
    agent_results: List[Dict],
    data_prompt: str,
    perf_memory: str,
    ticker: str,
) -> str:
    """Build the moderator's input: all agent opinions + original data."""
    sections = [data_prompt, perf_memory]

    sections.append("\n<agent_opinions>")
    sections.append(f"The following {len(agent_results)} specialist agents have independently analyzed {ticker}:\n")

    for ar in agent_results:
        name = ar["name"]
        role = ar["role"]
        parsed = ar["parsed"]
        sections.append(f"--- {name} ({role}) ---")
        sections.append(f"Decision: {parsed['decision'] or 'UNCLEAR'}")
        sections.append(f"Confidence: {parsed['confidence']}/5")
        sections.append(f"Thesis: {parsed['thesis']}")
        sections.append(f"Key Risk: {parsed['key_risk']}")
        sections.append("")

    sections.append("</agent_opinions>")
    sections.append("\nSynthesize these views into a single trading decision.")

    return "\n".join(sections)


# ═══════════════════════════════════════════════════════════════════════════════
# Live Trader — Main Orchestrator
# ═══════════════════════════════════════════════════════════════════════════════

class LiveTrader:
    """
    Orchestrates live trading decisions using Claude + Performance Memory.

    This is designed for daily use:
      1. Run once per trading day (or on rebalance days)
      2. Collects real-time data
      3. Generates S1 signal
      4. Calls Claude with market data + performance memory
      5. Returns structured decision with reasoning
    """

    def __init__(
        self,
        tickers: List[str],
        model: str = None,
        memory_path: str = "trade_memory.json",
        diary_path: str = "market_diary.json",
        journal_dir: str = "journals",
        holding_days: int = 5,
        dry_run: bool = False,
        abm: bool = False,
    ):
        self.tickers = tickers
        self.model = model or config.LLM_MODEL
        self.holding_days = holding_days
        self.dry_run = dry_run
        self.abm = abm
        self.memory = TradeMemory(memory_path)
        self.diary = MarketDiary(diary_path)
        self.journal = TradingJournal(journal_dir)

    def _extract_reasoning_summary(self, thesis: str) -> str:
        """Extract a 1-2 sentence summary from Claude's full analysis."""
        if not thesis:
            return ""

        # Try to extract from <conclusion> section
        import re
        conclusion_match = re.search(r'<conclusion>(.*?)</conclusion>', thesis, re.DOTALL)
        if conclusion_match:
            conclusion = conclusion_match.group(1).strip()
            # Get the first 2 meaningful sentences
            sentences = [s.strip() for s in conclusion.split('.') if len(s.strip()) > 20]
            return '. '.join(sentences[:2]) + '.' if sentences else conclusion[:200]

        # Try to extract from <synthesis> section (ICRL-style)
        synth_match = re.search(r'<synthesis>(.*?)</synthesis>', thesis, re.DOTALL)
        if synth_match:
            synth = synth_match.group(1).strip()
            sentences = [s.strip() for s in synth.split('.') if len(s.strip()) > 20]
            return '. '.join(sentences[:2]) + '.' if sentences else synth[:200]

        # Fallback: last 200 chars before DECISION
        decision_idx = thesis.upper().rfind("DECISION")
        if decision_idx > 0:
            snippet = thesis[max(0, decision_idx-250):decision_idx].strip()
            sentences = [s.strip() for s in snippet.split('.') if len(s.strip()) > 15]
            return '. '.join(sentences[-2:]) + '.' if sentences else snippet[-200:]

        return thesis[:200]

    def settle_past_trades(self):
        """Settle trades whose holding period has elapsed."""
        unsettled = self.memory.get_unsettled()
        if not unsettled:
            return 0

        # Collect price data for all tickers with unsettled trades
        unsettled_tickers = set(t.ticker for t in unsettled)
        price_data = {}

        for ticker in unsettled_tickers:
            try:
                # Fetch enough history to cover the holding period
                end_date = datetime.now().strftime("%Y-%m-%d")
                start_date = (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d")

                dc = DataCollector(ticker, start_date, end_date)
                dc.price_data = pd.DataFrame()  # Don't fetch full pipeline
                from data_collector import fetch_price_data
                dc.price_data = fetch_price_data(ticker, start_date, end_date)

                if not dc.price_data.empty:
                    price_data[ticker] = dc.price_data
            except Exception as e:
                logger.warning(f"Could not fetch settlement prices for {ticker}: {e}")

        count = self.memory.settle_trades(price_data, self.holding_days)
        if count > 0 and not self.dry_run:
            self.memory.save()
        return count

    def analyze_ticker(
        self, ticker: str, date_str: str = None,
        macro_snapshot: Dict = None,
    ) -> Dict:
        """
        Generate a live trading decision for a single ticker.

        Args:
            ticker: Stock ticker symbol
            date_str: Analysis date (default: today)
            macro_snapshot: Pre-captured macro snapshot (shared across tickers in a run)

        Returns:
            Dict with: ticker, date, signal_label, decision, position_weight,
                       entry_price, reasoning_summary, full_thesis
        """
        if date_str is None:
            date_str = datetime.now().strftime("%Y-%m-%d")

        print(f"\n{'='*60}")
        print(f"  Analyzing {ticker} for {date_str}")
        print(f"{'='*60}")

        # Step 1: Collect current data
        print(f"  [1/4] Collecting market data...")
        end_date = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")

        dc = DataCollector(ticker, start_date, end_date)
        data = dc.collect_all()

        if data["price_data"].empty:
            print(f"  ERROR: No price data available for {ticker}")
            return {"ticker": ticker, "error": "No price data"}

        # Step 2: Generate S1 signal
        print(f"  [2/4] Generating Algorithm S1 signal...")
        sg = SignalGenerator(dc.price_data, ticker=ticker)
        sg.generate()
        signal_label = sg.get_label(date_str) or "HOLD"
        print(f"         S1 Signal: {signal_label}")

        # Step 3: Build prompt with performance memory
        print(f"  [3/4] Building prompt with performance memory...")
        snapshot = dc.get_snapshot(date_str)
        if not snapshot:
            # If today is not a trading day, use the most recent available date
            available_dates = dc.price_data.index[dc.price_data.index <= pd.Timestamp(date_str)]
            if not available_dates.empty:
                actual_date = available_dates[-1].strftime("%Y-%m-%d")
                snapshot = dc.get_snapshot(actual_date)
                print(f"         (Using most recent trading day: {actual_date})")
            else:
                print(f"  ERROR: No trading data available for {date_str}")
                return {"ticker": ticker, "error": "No snapshot available"}

        news = dc.get_news_for_date(date_str)
        data_prompt = build_data_prompt(snapshot, news, signal_label=signal_label)

        # Append performance memory (includes both ticker context + macro context from past)
        perf_memory = build_performance_memory(self.memory, ticker, diary=self.diary)
        full_prompt = data_prompt + perf_memory

        # Append TODAY's macro snapshot as live context
        if macro_snapshot:
            full_prompt += "\n" + _build_current_macro_prompt(macro_snapshot)

        settled_count = len(self.memory.get_settled())
        print(f"         Performance memory: {settled_count} settled trades")

        # Step 4: Call Claude
        print(f"  [4/4] Calling Claude ({self.model})...")

        # Temporarily override model if user specified a different one
        original_model = config.LLM_MODEL
        config.LLM_MODEL = self.model

        try:
            thesis, decision = call_claude(
                full_prompt,
                ticker=ticker,
                date=date_str,
                system_prompt=LIVE_SYSTEM_PROMPT,
            )
        finally:
            config.LLM_MODEL = original_model

        if not decision:
            print(f"  WARNING: Claude failed to return a decision. Using S1 signal.")
            decision = signal_label
            thesis = ""

        position_weight = config.POSITION_WEIGHTS.get(decision, 0.0)
        entry_price = float(snapshot["price_summary"]["current_price"])
        reasoning_summary = self._extract_reasoning_summary(thesis)

        # Log and save
        result = {
            "ticker": ticker,
            "date": date_str,
            "signal_label": signal_label,
            "decision": decision,
            "position_weight": position_weight,
            "entry_price": entry_price,
            "reasoning_summary": reasoning_summary,
            "full_thesis": thesis,
            "model": self.model,
            "was_override": decision != signal_label,
        }

        # Extract market context for pattern learning
        mkt_context = extract_market_context(snapshot, news)

        # Add to memory (unless dry run)
        if not self.dry_run:
            trade = TradeEntry(
                date=date_str,
                ticker=ticker,
                signal_label=signal_label,
                decision=decision,
                position_weight=position_weight,
                entry_price=entry_price,
                reasoning_summary=reasoning_summary,
                model_used=self.model,
                market_context=mkt_context,
                macro_context=macro_snapshot,
            )
            self.memory.add_trade(trade)
            self.memory.save()

        return result

    def analyze_ticker_abm(
        self, ticker: str, date_str: str = None,
        macro_snapshot: Dict = None,
    ) -> Dict:
        """
        ABM analysis: 6 specialist agents independently analyze, then a moderator
        synthesizes into a final decision.

        Cost: 7 Claude calls per ticker (~$0.30 at Sonnet pricing).
        """
        import re

        if date_str is None:
            date_str = datetime.now().strftime("%Y-%m-%d")

        print(f"\n{'='*60}")
        print(f"  ABM Analysis: {ticker} for {date_str}")
        print(f"  6 agents + 1 moderator = 7 Claude calls")
        print(f"{'='*60}")

        # Step 1: Collect data (same as standard analysis)
        print(f"  [1/4] Collecting market data...")
        end_date = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")

        dc = DataCollector(ticker, start_date, end_date)
        data = dc.collect_all()

        if data["price_data"].empty:
            print(f"  ERROR: No price data available for {ticker}")
            return {"ticker": ticker, "error": "No price data"}

        # Step 2: Generate S1 signal
        print(f"  [2/4] Generating Algorithm S1 signal...")
        sg = SignalGenerator(dc.price_data, ticker=ticker)
        sg.generate()
        signal_label = sg.get_label(date_str) or "HOLD"
        print(f"         S1 Signal: {signal_label}")

        # Build the base data prompt (shared by all agents)
        snapshot = dc.get_snapshot(date_str)
        if not snapshot:
            available_dates = dc.price_data.index[dc.price_data.index <= pd.Timestamp(date_str)]
            if not available_dates.empty:
                actual_date = available_dates[-1].strftime("%Y-%m-%d")
                snapshot = dc.get_snapshot(actual_date)
            else:
                return {"ticker": ticker, "error": "No snapshot available"}

        news = dc.get_news_for_date(date_str)
        data_prompt = build_data_prompt(snapshot, news, signal_label=signal_label)
        perf_memory = build_performance_memory(self.memory, ticker, diary=self.diary)

        # Add current macro context
        macro_section = ""
        if macro_snapshot:
            macro_section = "\n" + _build_current_macro_prompt(macro_snapshot)

        base_prompt = data_prompt + perf_memory + macro_section

        # Step 3: Run all 6 agents
        print(f"  [3/4] Running 6 specialist agents...")
        original_model = config.LLM_MODEL
        config.LLM_MODEL = self.model

        agent_results = []
        try:
            for i, persona in enumerate(AGENT_PERSONAS, 1):
                name = persona["name"]
                role = persona["role"]
                print(f"         [{i}/6] {name} ({role})...", end="", flush=True)

                agent_system = _build_agent_system_prompt(persona)

                try:
                    response_text, _ = call_claude(
                        base_prompt,
                        ticker=ticker,
                        date=date_str,
                        system_prompt=agent_system,
                    )
                except Exception as e:
                    logger.error(f"Agent {name} failed: {e}")
                    response_text = None

                parsed = _parse_agent_response(response_text)
                agent_results.append({
                    "name": name,
                    "role": role,
                    "parsed": parsed,
                })

                decision_str = parsed["decision"] or "UNCLEAR"
                conf = parsed["confidence"]
                print(f" {decision_str} (conf={conf}/5)")

                time.sleep(0.5)  # Brief rate limiting between agents

            # Step 4: Moderator synthesis
            print(f"  [4/4] Moderator synthesizing {len(agent_results)} opinions...")

            moderator_prompt = _build_moderator_data_prompt(
                agent_results, data_prompt, perf_memory, ticker
            )
            if macro_snapshot:
                moderator_prompt += "\n" + _build_current_macro_prompt(macro_snapshot)

            mod_response, _ = call_claude(
                moderator_prompt,
                ticker=ticker,
                date=date_str,
                system_prompt=ABM_MODERATOR_PROMPT,
            )

        finally:
            config.LLM_MODEL = original_model

        # Parse moderator result
        mod_parsed = _parse_moderator_response(mod_response)
        decision = mod_parsed["decision"] or signal_label
        position_weight = config.POSITION_WEIGHTS.get(decision, 0.0)
        entry_price = float(snapshot["price_summary"]["current_price"])

        # Build reasoning summary from moderator + agent votes
        agent_summary_parts = []
        for ar in agent_results:
            p = ar["parsed"]
            agent_summary_parts.append(f"{ar['name']}={p['decision'] or '?'}({p['confidence']})")
        agent_summary = " ".join(agent_summary_parts)
        reasoning = f"ABM [{agent_summary}] → {decision}. {mod_parsed['reasoning'][:200]}"

        # Print ABM summary
        print(f"\n  ┌─── ABM Result ───────────────────────────────┐")
        print(f"  │ Agent Votes:                                   │")
        for ar in agent_results:
            p = ar["parsed"]
            d = p["decision"] or "UNCLEAR"
            c = p["confidence"]
            thesis_short = p["thesis"][:50].replace("\n", " ") if p["thesis"] else ""
            print(f"  │  {ar['name']:>12s}: {d:>12s} ({c}/5) {thesis_short}")
        print(f"  │                                                │")
        print(f"  │ Moderator Decision: {decision:>12s}              │")
        print(f"  │ Confidence: {mod_parsed['confidence']}/5                            │")
        print(f"  │ S1 Signal:  {signal_label:>12s}              │")
        print(f"  └────────────────────────────────────────────────┘")

        result = {
            "ticker": ticker,
            "date": date_str,
            "signal_label": signal_label,
            "decision": decision,
            "position_weight": position_weight,
            "entry_price": entry_price,
            "reasoning_summary": reasoning,
            "full_thesis": mod_response,
            "model": self.model,
            "was_override": decision != signal_label,
            "abm_agents": agent_results,
            "abm_moderator": mod_parsed,
        }

        # Extract market context
        mkt_context = extract_market_context(snapshot, news)

        # Save to memory
        if not self.dry_run:
            trade = TradeEntry(
                date=date_str,
                ticker=ticker,
                signal_label=signal_label,
                decision=decision,
                position_weight=position_weight,
                entry_price=entry_price,
                reasoning_summary=reasoning,
                model_used=f"{self.model}+ABM",
                market_context=mkt_context,
                macro_context=macro_snapshot,
            )
            self.memory.add_trade(trade)
            self.memory.save()

        return result

    def run(self, date_str: str = None) -> List[Dict]:
        """
        Run live analysis for all tickers.
        Settles past trades first, captures macro snapshot once,
        then generates new decisions.
        """
        if date_str is None:
            date_str = datetime.now().strftime("%Y-%m-%d")

        mode_str = "DRY RUN" if self.dry_run else "LIVE (saving to memory)"
        method_str = "ABM (6 agents + moderator)" if self.abm else "Single-agent"

        print(f"\n{'#'*60}")
        print(f"  LIVE TRADING ANALYSIS — {date_str}")
        print(f"  Model: {self.model}")
        print(f"  Method: {method_str}")
        print(f"  Tickers: {', '.join(self.tickers)}")
        print(f"  Mode: {mode_str}")
        print(f"{'#'*60}")

        # Step 1: Settle past trades
        print(f"\n  Settling past trades...")
        settled = self.settle_past_trades()
        print(f"  Settled {settled} trades from previous sessions")

        # Step 2: Auto-backfill diary for missed days
        last_diary = self.diary.get_last_entry_date()
        if last_diary:
            missing = self.diary.get_missing_dates(last_diary, date_str)
            if len(missing) > 1:  # >1 because today might be missing (we're about to capture it)
                print(f"\n  Detected {len(missing)} days without diary coverage since {last_diary}")
                print(f"  Auto-backfilling from Yahoo Finance...")
                backfill_diary(self.tickers, self.diary, start_date=last_diary, end_date=date_str)

        # Step 3: Capture macro snapshot ONCE (shared across all tickers)
        print(f"\n  Capturing macro environment...")
        macro_snapshot = capture_macro_snapshot()
        regime = macro_snapshot.get("risk_regime", "?")
        vol = macro_snapshot.get("vol_regime", "?")
        spy = macro_snapshot.get("spy_change_1d")
        spy_str = f"SPY={spy:+.1f}%" if spy is not None else "SPY=N/A"
        n_headlines = len(macro_snapshot.get("global_headlines", []))
        print(f"  Macro: {spy_str} | Regime={regime} | Vol={vol} | {n_headlines} global headlines")

        # Save today's diary entry (with live headlines)
        if not self.dry_run:
            # Ticker snapshots are captured per-ticker in analyze_ticker, but
            # we record the macro snapshot now so the diary is always up to date
            self.diary.add_entry(date_str, macro_snapshot, {}, source="capture")
            self.diary.save()

        # Step 4: Analyze each ticker (macro snapshot passed to each)
        analyze_fn = self.analyze_ticker_abm if self.abm else self.analyze_ticker
        results = []
        for i, ticker in enumerate(self.tickers, 1):
            print(f"\n  [{i}/{len(self.tickers)}] Processing {ticker}...")
            try:
                result = analyze_fn(ticker, date_str, macro_snapshot=macro_snapshot)
                results.append(result)
                time.sleep(1)  # Rate limiting between tickers
            except Exception as e:
                logger.error(f"Error analyzing {ticker}: {e}")
                results.append({"ticker": ticker, "error": str(e)})

        # Step 5: Print summary
        self._print_summary(results)

        # Step 6: Write daily journal
        if not self.dry_run:
            method_str = "ABM (6 agents + moderator)" if self.abm else "single-agent"
            self.journal.write_journal(
                date_str=date_str,
                macro_snapshot=macro_snapshot,
                ticker_entries=results,
                method=method_str,
                model=self.model,
            )

        return results

    def _print_summary(self, results: List[Dict]):
        """Print a clean summary of all decisions."""
        print(f"\n{'='*60}")
        print(f"  TRADING DECISIONS SUMMARY")
        print(f"{'='*60}")
        print(f"  {'Ticker':>6s}  {'S1 Signal':>12s}  {'Decision':>12s}  {'Weight':>7s}  {'Price':>10s}  Override")
        print(f"  {'─'*6}  {'─'*12}  {'─'*12}  {'─'*7}  {'─'*10}  {'─'*8}")

        for r in results:
            if "error" in r:
                print(f"  {r['ticker']:>6s}  {'ERROR':>12s}  {r['error']}")
                continue

            override_str = " YES" if r["was_override"] else "  no"
            weight_str = f"{r['position_weight']:+.1f}"
            print(
                f"  {r['ticker']:>6s}  {r['signal_label']:>12s}  {r['decision']:>12s}  "
                f"{weight_str:>7s}  ${r['entry_price']:>9.2f}  {override_str}"
            )

        print()

        # Print key reasonings
        for r in results:
            if "error" not in r and r.get("reasoning_summary"):
                print(f"  {r['ticker']}: {r['reasoning_summary'][:120]}")

        print(f"\n{'='*60}")

    def show_status(self):
        """Print current portfolio status and trade history."""
        print(f"\n{'='*60}")
        print(f"  PORTFOLIO STATUS — Trade Memory: {self.memory.memory_path}")
        print(f"{'='*60}")

        all_trades = self.memory.trades
        settled = self.memory.get_settled()
        unsettled = self.memory.get_unsettled()

        print(f"\n  Total trades: {len(all_trades)}")
        print(f"  Settled: {len(settled)}  |  Unsettled (pending): {len(unsettled)}")

        if settled:
            pnls = [t.realized_pnl_pct for t in settled if t.realized_pnl_pct is not None]
            wins = sum(1 for p in pnls if p > 0)
            total_pnl = sum(pnls)
            print(f"\n  Win rate: {wins}/{len(pnls)} ({wins/len(pnls)*100:.0f}%)")
            print(f"  Total P&L: {total_pnl:+.2f}%")
            print(f"  Mean P&L per trade: {np.mean(pnls):+.3f}%")
            print(f"  Best trade: {max(pnls):+.2f}% | Worst: {min(pnls):+.2f}%")

            # Per-ticker breakdown
            tickers_seen = set(t.ticker for t in settled)
            if len(tickers_seen) > 1:
                print(f"\n  Per-Ticker Breakdown:")
                for ticker in sorted(tickers_seen):
                    t_pnls = [t.realized_pnl_pct for t in settled
                              if t.ticker == ticker and t.realized_pnl_pct is not None]
                    t_wins = sum(1 for p in t_pnls if p > 0)
                    print(
                        f"    {ticker:>6s}: {len(t_pnls)} trades | "
                        f"win={t_wins/len(t_pnls)*100:.0f}% | "
                        f"total={sum(t_pnls):+.2f}%"
                    )

        if unsettled:
            print(f"\n  Open Positions (pending settlement):")
            for t in unsettled:
                print(
                    f"    {t.date} {t.ticker:>6s}: {t.decision:>12s} @ ${t.entry_price:.2f} "
                    f"(S1={t.signal_label})"
                )

        print(f"\n{'='*60}")

    def confirm_trades(self):
        """
        Interactive confirmation of pending trades.

        Workflow:
          1. Show each pending trade
          2. User chooses: [E]xecute as-is / [A]djust price/size / [S]kip / [Q]uit
          3. Only executed/partial trades will be settled and used for learning

        This is the critical feedback loop that makes performance memory REAL —
        the system only learns from trades that were actually taken.
        """
        pending = self.memory.get_pending_confirmation()

        if not pending:
            print("\n  No pending trades to confirm. All trades have been reviewed.")
            # Also show counts
            executed = len(self.memory.get_executed())
            skipped = sum(1 for t in self.memory.trades if t.execution_status == "skipped")
            print(f"  Executed: {executed}  |  Skipped: {skipped}")
            return

        print(f"\n{'='*60}")
        print(f"  TRADE CONFIRMATION — {len(pending)} pending trades")
        print(f"  Only confirmed trades are used for settlement & learning.")
        print(f"{'='*60}")

        confirmed = 0
        skipped = 0

        for i, trade in enumerate(pending, 1):
            print(f"\n  [{i}/{len(pending)}] {trade.date} — {trade.ticker}")
            print(f"    Model recommendation: {trade.decision} (weight={trade.position_weight:+.1f})")
            print(f"    S1 signal:            {trade.signal_label}")
            print(f"    Model entry price:    ${trade.entry_price:.2f}")
            if trade.reasoning_summary:
                print(f"    Reasoning: {trade.reasoning_summary[:120]}")

            while True:
                choice = input(
                    "\n    [E]xecute as recommended  "
                    "[A]djust price/size  "
                    "[S]kip (didn't trade)  "
                    "[Q]uit: "
                ).strip().upper()

                if choice == "E":
                    trade.execution_status = "executed"
                    confirmed += 1
                    print(f"    → Confirmed as executed @ ${trade.entry_price:.2f}")
                    break

                elif choice == "A":
                    # Adjust entry price
                    price_input = input(f"    Actual fill price (Enter = ${trade.entry_price:.2f}): ").strip()
                    if price_input:
                        try:
                            trade.actual_entry_price = float(price_input)
                        except ValueError:
                            print("    Invalid price. Using model's price.")

                    # Adjust position size
                    weight_input = input(
                        f"    Actual position weight -1.0 to +1.0 (Enter = {trade.position_weight:+.1f}): "
                    ).strip()
                    if weight_input:
                        try:
                            w = float(weight_input)
                            if -1.0 <= w <= 1.0:
                                trade.actual_position_weight = w
                            else:
                                print("    Weight must be between -1.0 and +1.0. Using recommended.")
                        except ValueError:
                            print("    Invalid weight. Using recommended.")

                    # Optional note
                    note = input("    Note (optional, Enter to skip): ").strip()
                    if note:
                        trade.execution_note = note

                    # Determine status
                    if trade.actual_position_weight is not None and abs(trade.actual_position_weight) < abs(trade.position_weight):
                        trade.execution_status = "partial"
                    else:
                        trade.execution_status = "executed"

                    actual_p = trade.actual_entry_price or trade.entry_price
                    actual_w = trade.actual_position_weight or trade.position_weight
                    print(f"    → Confirmed @ ${actual_p:.2f}, weight={actual_w:+.1f} ({trade.execution_status})")
                    confirmed += 1
                    break

                elif choice == "S":
                    trade.execution_status = "skipped"
                    note = input("    Reason for skipping (optional): ").strip()
                    if note:
                        trade.execution_note = note
                    skipped += 1
                    print(f"    → Skipped (will not be used for learning)")
                    break

                elif choice == "Q":
                    print(f"\n  Quitting. Confirmed {confirmed}, skipped {skipped}, "
                          f"remaining {len(pending) - i} still pending.")
                    if not self.dry_run:
                        self.memory.save()
                    return

                else:
                    print("    Invalid choice. Enter E, A, S, or Q.")

        # Save all confirmations
        if not self.dry_run:
            self.memory.save()

        print(f"\n{'='*60}")
        print(f"  CONFIRMATION COMPLETE")
        print(f"  Executed: {confirmed}  |  Skipped: {skipped}")
        print(f"{'='*60}")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI Entry Point
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="Live Trading with Claude + Performance Memory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze AAPL, MSFT, NVDA for today
  python live_trader.py --tickers AAPL MSFT NVDA

  # Use Claude Opus for maximum reasoning
  python live_trader.py --tickers AAPL --model claude-opus-4-20250514

  # Dry run (print decisions without saving)
  python live_trader.py --tickers AAPL --dry-run

  # Show current portfolio status
  python live_trader.py --status

  # Use all paper tickers
  python live_trader.py --all-paper-tickers

  # Custom memory file
  python live_trader.py --tickers AAPL --memory-file my_trades.json

  # ABM mode: 6 specialist agents + moderator
  python live_trader.py --tickers AAPL MSFT --abm

  # ABM with Opus for maximum reasoning
  python live_trader.py --tickers AAPL --abm --model claude-opus-4-20250514
        """
    )

    parser.add_argument(
        "--tickers", nargs="+", default=["AAPL"],
        help="Ticker symbols to analyze (default: AAPL)"
    )
    parser.add_argument(
        "--all-paper-tickers", action="store_true",
        help="Use all 14 tickers from the Trading-R1 paper"
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help=f"Claude model to use (default: {config.LLM_MODEL}). "
             "Options: claude-sonnet-4-20250514, claude-opus-4-20250514"
    )
    parser.add_argument(
        "--date", type=str, default=None,
        help="Analysis date (default: today). Format: YYYY-MM-DD"
    )
    parser.add_argument(
        "--memory-file", type=str, default="trade_memory.json",
        help="Path to trade memory file (default: trade_memory.json)"
    )
    parser.add_argument(
        "--holding-days", type=int, default=5,
        help="Holding period in trading days for P&L settlement (default: 5)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print decisions without saving to memory"
    )
    parser.add_argument(
        "--status", action="store_true",
        help="Show current portfolio status and exit"
    )
    parser.add_argument(
        "--settle-only", action="store_true",
        help="Only settle past trades (no new analysis)"
    )
    parser.add_argument(
        "--abm", action="store_true",
        help="Use Agent-Based Modeling: 6 specialist agents + moderator per ticker "
             "(~7 Claude calls per ticker, ~$0.30/ticker at Sonnet pricing)"
    )
    parser.add_argument(
        "--confirm", action="store_true",
        help="Interactively confirm pending trades (mark as executed/skipped/adjusted). "
             "Only confirmed trades are used for settlement and learning."
    )
    parser.add_argument(
        "--capture", action="store_true",
        help="Lightweight daily snapshot capture (no Claude calls). "
             "Stores macro + ticker data in market_diary.json. "
             "Automate via cron for continuous coverage."
    )
    parser.add_argument(
        "--backfill", type=str, nargs="?", const="auto", metavar="START_DATE",
        help="Backfill diary for missed days from Yahoo Finance. "
             "Omit date to auto-detect from last diary entry. "
             "Example: --backfill 2025-01-01"
    )
    parser.add_argument(
        "--diary-file", type=str, default="market_diary.json",
        help="Path to market diary file (default: market_diary.json)"
    )
    parser.add_argument(
        "--journal-dir", type=str, default="journals",
        help="Directory for daily trading journal markdown files (default: journals/)"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable verbose logging"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(level=log_level, format="%(message)s")

    # Resolve tickers
    tickers = config.PAPER_TICKERS if args.all_paper_tickers else args.tickers

    # Create trader
    trader = LiveTrader(
        tickers=tickers,
        model=args.model,
        memory_path=args.memory_file,
        diary_path=args.diary_file,
        journal_dir=args.journal_dir,
        holding_days=args.holding_days,
        dry_run=args.dry_run,
        abm=args.abm,
    )

    # Handle capture mode (no Claude calls — just data)
    if args.capture:
        date_str = args.date or datetime.now().strftime("%Y-%m-%d")
        print(f"\n  Daily capture mode — {date_str}")
        capture_daily_snapshot(tickers, trader.diary, date_str)
        return

    # Handle backfill mode
    if args.backfill:
        end_date = args.date or datetime.now().strftime("%Y-%m-%d")
        start_date = None if args.backfill == "auto" else args.backfill
        backfill_diary(tickers, trader.diary, start_date=start_date, end_date=end_date)
        return

    # Handle status mode
    if args.status:
        trader.show_status()
        return

    # Handle confirm mode
    if args.confirm:
        trader.confirm_trades()
        return

    # Handle settle-only mode
    if args.settle_only:
        print("Settling past trades...")
        settled = trader.settle_past_trades()
        print(f"Settled {settled} trades")
        trader.show_status()
        return

    # Run live analysis
    results = trader.run(date_str=args.date)

    # Print full thesis for each ticker if verbose
    if args.verbose:
        for r in results:
            if "error" not in r and r.get("full_thesis"):
                print(f"\n{'='*60}")
                print(f"  FULL THESIS: {r['ticker']}")
                print(f"{'='*60}")
                print(r["full_thesis"])


if __name__ == "__main__":
    main()
