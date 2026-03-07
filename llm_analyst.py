"""
LLM-Powered Financial Analyst — Trading-R1 Methodology

Uses Claude API to generate structured investment theses following the
Trading-R1 prompt format (Section 4.2, Figure 3):
  - Structured financial context → Investment thesis → Trading decision
  - Output: 5-class action {STRONG SELL, SELL, HOLD, BUY, STRONG BUY}

Also includes a rule-based fallback mode (no API cost) that uses
the volatility-based signal labels directly.
"""
import logging
import re
import time
import json
from typing import Dict, List, Optional, Tuple

import config

logger = logging.getLogger(__name__)

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False
    logger.warning("anthropic SDK not installed; LLM mode unavailable")


# ═══════════════════════════════════════════════════════════════════════════════
# Prompt Construction (mirrors Tauric-TR1-DB format)
# ═══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are a senior financial analyst producing a structured investment thesis.

Your task: Analyze the provided market data for the given asset and produce:
1. A structured investment thesis covering multiple perspectives
2. A final trading recommendation

FORMAT YOUR RESPONSE EXACTLY AS FOLLOWS:

<analysis>

<fundamentals>
Analyze the company's financial health, valuation metrics, growth prospects.
Support claims with specific data points from the provided context.
</fundamentals>

<technicals>
Analyze price action, momentum, trend, and volatility indicators.
Reference specific indicator values and what they suggest.
</technicals>

<sentiment>
Analyze news sentiment, analyst recommendations, and market perception.
Cite specific headlines or recommendation changes where relevant.
</sentiment>

<macro>
Consider broader market conditions and sector trends.
</macro>

<conclusion>
Synthesize all perspectives into a cohesive investment view.
State the key risk factors and catalysts.

DECISION: [[[STRONG SELL / SELL / HOLD / BUY / STRONG BUY]]]
</conclusion>

</analysis>

IMPORTANT RULES:
- Your DECISION must be exactly one of: STRONG SELL, SELL, HOLD, BUY, STRONG BUY
- Place your decision in triple brackets: [[[DECISION]]]
- Ground every claim in the provided data — do not hallucinate facts
- Be concise but thorough (target 400-600 words total)
- Consider a ~1 week holding period for your recommendation
"""


def build_data_prompt(snapshot: Dict, news: Dict = None) -> str:
    """
    Assemble a structured prompt from collected data.
    Mirrors the categorical-sampled data format from Tauric-TR1-DB.
    """
    ticker = snapshot.get("ticker", "UNKNOWN")
    date = snapshot.get("date", "")
    price = snapshot.get("price_summary", {})
    tech = snapshot.get("technical_indicators", {})
    fund = snapshot.get("fundamentals", {})
    recs = snapshot.get("analyst_recommendations", {})

    sections = []

    # ── Header ────────────────────────────────────────────────────────────
    sections.append(f"=== INVESTMENT ANALYSIS: {ticker} | Date: {date} ===\n")

    # ── Price Data (15-day window) ────────────────────────────────────────
    sections.append("<market_data>")
    sections.append(f"Current Price: ${price.get('current_price', 'N/A')}")
    sections.append(f"1-Day Change:  {price.get('price_change_1d', 'N/A')}%")
    sections.append(f"5-Day Change:  {price.get('price_change_5d', 'N/A')}%")
    sections.append(f"15-Day Change: {price.get('price_change_15d', 'N/A')}%")
    sections.append(f"Volume: {price.get('volume', 'N/A'):,}" if isinstance(price.get('volume'), (int, float)) else "")

    # OHLCV table (last 5 days for brevity)
    ohlcv = snapshot.get("ohlcv_table", [])
    if ohlcv:
        sections.append("\nRecent OHLCV (last 5 trading days):")
        for row in ohlcv[-5:]:
            sections.append(
                f"  {row['date']}: O={row['open']:.2f} H={row['high']:.2f} "
                f"L={row['low']:.2f} C={row['close']:.2f} V={row['volume']:,}"
            )
    sections.append("</market_data>\n")

    # ── Technical Indicators ──────────────────────────────────────────────
    sections.append("<technicals>")
    if tech:
        indicator_groups = {
            "Moving Averages": ["sma_50", "sma_200", "ema_10", "ema_50"],
            "Momentum": ["rsi", "macd", "macd_signal", "macd_histogram", "roc_10", "stoch_k", "stoch_d"],
            "Volatility": ["atr", "bb_upper", "bb_middle", "bb_lower", "zscore_75"],
            "Trend": ["adx", "adx_pos", "adx_neg"],
            "Volume": ["mfi", "vwma_20"],
        }
        for group_name, keys in indicator_groups.items():
            values = {k: tech[k] for k in keys if k in tech}
            if values:
                sections.append(f"  {group_name}:")
                for k, v in values.items():
                    sections.append(f"    {k}: {v}")
    else:
        sections.append("  No technical indicator data available.")
    sections.append("</technicals>\n")

    # ── Fundamentals ──────────────────────────────────────────────────────
    sections.append("<fundamentals>")
    if fund:
        sections.append(f"  Company: {fund.get('short_name', ticker)}")
        sections.append(f"  Sector: {fund.get('sector', 'N/A')} | Industry: {fund.get('industry', 'N/A')}")

        def fmt_large(val):
            if val is None: return "N/A"
            if abs(val) >= 1e12: return f"${val/1e12:.2f}T"
            if abs(val) >= 1e9: return f"${val/1e9:.2f}B"
            if abs(val) >= 1e6: return f"${val/1e6:.2f}M"
            return f"${val:,.0f}"

        def fmt_pct(val):
            return f"{val*100:.1f}%" if val is not None else "N/A"

        sections.append(f"  Market Cap: {fmt_large(fund.get('market_cap'))}")
        sections.append(f"  P/E (TTM): {fund.get('pe_ratio', 'N/A')} | Forward P/E: {fund.get('forward_pe', 'N/A')}")
        sections.append(f"  P/B: {fund.get('pb_ratio', 'N/A')}")
        sections.append(f"  Revenue: {fmt_large(fund.get('revenue'))}")
        sections.append(f"  Net Income: {fmt_large(fund.get('net_income'))}")
        sections.append(f"  Profit Margin: {fmt_pct(fund.get('profit_margin'))}")
        sections.append(f"  Operating Margin: {fmt_pct(fund.get('operating_margin'))}")
        sections.append(f"  ROE: {fmt_pct(fund.get('roe'))}")
        sections.append(f"  Debt/Equity: {fund.get('debt_to_equity', 'N/A')}")
        sections.append(f"  Free Cash Flow: {fmt_large(fund.get('free_cash_flow'))}")
        sections.append(f"  Earnings Growth: {fmt_pct(fund.get('earnings_growth'))}")
        sections.append(f"  Revenue Growth: {fmt_pct(fund.get('revenue_growth'))}")
    else:
        sections.append("  No fundamental data available.")
    sections.append("</fundamentals>\n")

    # ── Analyst Sentiment ─────────────────────────────────────────────────
    sections.append("<sentiment>")
    if recs:
        sections.append(f"  Analyst Consensus Score: {recs.get('consensus_score', 'N/A')}/5.0")
        sections.append(f"  Strong Buy: {recs.get('strong_buy', 0)} | Buy: {recs.get('buy', 0)} | "
                        f"Hold: {recs.get('hold', 0)} | Sell: {recs.get('sell', 0)} | "
                        f"Strong Sell: {recs.get('strong_sell', 0)}")
    else:
        sections.append("  No analyst recommendation data available.")
    sections.append("</sentiment>\n")

    # ── News ──────────────────────────────────────────────────────────────
    sections.append("<news>")
    if news:
        for bucket_name, articles in news.items():
            if articles:
                label = bucket_name.replace("_", " ").title()
                sections.append(f"  [{label}]")
                for art in articles[:5]:  # Top 5 per bucket
                    headline = art.get("headline", "")[:120]
                    source = art.get("source", "")
                    sections.append(f"    - {headline} ({source})")
    else:
        sections.append("  No recent news available.")
    sections.append("</news>")

    return "\n".join(sections)


# ═══════════════════════════════════════════════════════════════════════════════
# LLM Call & Response Parsing
# ═══════════════════════════════════════════════════════════════════════════════

VALID_DECISIONS = {"STRONG SELL", "SELL", "HOLD", "BUY", "STRONG BUY"}
DECISION_MAP = {
    "STRONG SELL": "STRONG_SELL",
    "SELL": "SELL",
    "HOLD": "HOLD",
    "BUY": "BUY",
    "STRONG BUY": "STRONG_BUY",
}


def parse_decision(response_text: str) -> Optional[str]:
    """
    Extract the trading decision from the LLM response.
    Looks for [[[DECISION]]] pattern as instructed in the prompt.
    """
    # Primary: triple-bracket pattern
    match = re.search(r'\[\[\[(.*?)\]\]\]', response_text)
    if match:
        decision = match.group(1).strip().upper()
        if decision in VALID_DECISIONS:
            return DECISION_MAP[decision]

    # Fallback: look for DECISION: pattern
    match = re.search(r'DECISION:\s*(STRONG\s+SELL|STRONG\s+BUY|SELL|BUY|HOLD)',
                      response_text, re.IGNORECASE)
    if match:
        decision = match.group(1).strip().upper()
        if decision in VALID_DECISIONS:
            return DECISION_MAP[decision]

    # Last resort: look for any valid decision keyword near end of text
    last_500 = response_text[-500:].upper()
    for d in ["STRONG BUY", "STRONG SELL", "BUY", "SELL", "HOLD"]:
        if d in last_500:
            return DECISION_MAP[d]

    logger.warning("Could not parse trading decision from LLM response")
    return None


def call_claude(prompt: str, observer=None, ticker: str = "", date: str = "") -> Tuple[Optional[str], Optional[str]]:
    """
    Call Claude API with retry logic and exponential backoff.
    Returns (full_response_text, parsed_decision).
    """
    if not HAS_ANTHROPIC:
        logger.error("anthropic SDK not installed")
        return None, None

    if not config.ANTHROPIC_API_KEY:
        logger.error("ANTHROPIC_API_KEY not set")
        return None, None

    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

    for attempt in range(config.LLM_MAX_RETRIES):
        try:
            t0 = time.time()
            response = client.messages.create(
                model=config.LLM_MODEL,
                max_tokens=config.LLM_MAX_TOKENS,
                temperature=config.LLM_TEMPERATURE,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            latency = time.time() - t0

            text = response.content[0].text
            decision = parse_decision(text)

            if decision:
                if observer:
                    observer.log_llm_response(ticker, date, text, decision, latency)
                return text, decision
            else:
                logger.warning(f"Attempt {attempt+1}: could not parse decision, retrying...")
                if observer:
                    observer.log_llm_error(ticker, date, "Could not parse decision from response", attempt+1)

        except Exception as e:
            wait = config.LLM_RETRY_DELAY * (2 ** attempt)
            logger.warning(f"API call failed (attempt {attempt+1}): {e}. Retrying in {wait:.1f}s")
            if observer:
                observer.log_llm_error(ticker, date, str(e), attempt+1)
            time.sleep(wait)

    logger.error("All LLM retry attempts exhausted")
    return None, None


# ═══════════════════════════════════════════════════════════════════════════════
# Rule-Based Fallback (no API cost)
# ═══════════════════════════════════════════════════════════════════════════════

def rule_based_decision(snapshot: Dict, signal_label: str = None) -> Tuple[str, str]:
    """
    Generate a decision without LLM API calls.
    Uses the volatility-based signal labels from Algorithm S1 directly,
    plus a simple heuristic cross-check with technical indicators.
    Returns (thesis_text, decision).
    """
    ticker = snapshot.get("ticker", "UNKNOWN")
    price = snapshot.get("price_summary", {})
    tech = snapshot.get("technical_indicators", {})

    # Start with the Algorithm S1 signal if available
    if signal_label and signal_label in config.ACTION_LABELS:
        base_decision = signal_label
    else:
        base_decision = "HOLD"

    # Simple technical cross-check
    bullish_signals = 0
    bearish_signals = 0

    rsi = tech.get("rsi")
    if rsi is not None:
        if rsi < 30: bullish_signals += 1
        elif rsi > 70: bearish_signals += 1

    macd = tech.get("macd")
    macd_sig = tech.get("macd_signal")
    if macd is not None and macd_sig is not None:
        if macd > macd_sig: bullish_signals += 1
        else: bearish_signals += 1

    sma50 = tech.get("sma_50")
    sma200 = tech.get("sma_200")
    current = price.get("current_price")
    if sma50 is not None and sma200 is not None:
        if sma50 > sma200: bullish_signals += 1
        else: bearish_signals += 1

    if current is not None and sma50 is not None:
        if current > sma50: bullish_signals += 1
        else: bearish_signals += 1

    adx = tech.get("adx")
    trend_strength = "weak" if (adx is not None and adx < 20) else "moderate-to-strong"

    # Build a simple thesis text
    thesis = (
        f"=== Rule-Based Analysis: {ticker} ===\n"
        f"Signal (Algorithm S1): {base_decision}\n"
        f"Price: ${price.get('current_price', 'N/A')} | "
        f"1d: {price.get('price_change_1d', 'N/A')}% | "
        f"5d: {price.get('price_change_5d', 'N/A')}%\n"
        f"RSI: {rsi} | MACD vs Signal: {'bullish' if macd and macd_sig and macd > macd_sig else 'bearish'}\n"
        f"Trend Strength (ADX): {trend_strength}\n"
        f"Technical Score: {bullish_signals} bullish / {bearish_signals} bearish\n"
        f"Decision: {base_decision}\n"
    )

    return thesis, base_decision


# ═══════════════════════════════════════════════════════════════════════════════
# Analyst Interface
# ═══════════════════════════════════════════════════════════════════════════════

class LLMAnalyst:
    """
    Unified interface for generating trading decisions.
    Supports both LLM mode (Claude API) and signal-only mode (rule-based).
    Accepts an optional Observer for rich logging and input storage.
    """

    def __init__(self, use_llm: bool = True, observer=None):
        self.use_llm = use_llm and HAS_ANTHROPIC and bool(config.ANTHROPIC_API_KEY)
        if use_llm and not self.use_llm:
            logger.warning("LLM mode requested but unavailable; falling back to rule-based mode")
        self.analysis_log: List[Dict] = []
        self.obs = observer

    def analyze(
        self,
        snapshot: Dict,
        news: Dict = None,
        signal_label: str = None
    ) -> Tuple[str, str]:
        """
        Generate a trading decision for the given data snapshot.

        Args:
            snapshot: Market data snapshot from DataCollector.get_snapshot()
            news: News data bucketed by temporal horizon
            signal_label: Pre-computed Algorithm S1 label (used in rule-based mode)

        Returns:
            (thesis_text, decision): The analysis text and one of
                STRONG_SELL, SELL, HOLD, BUY, STRONG_BUY
        """
        ticker = snapshot.get("ticker", "UNKNOWN")
        date = snapshot.get("date", "")

        # Store the snapshot for inspection
        if self.obs:
            self.obs.store_snapshot(ticker, date, snapshot)

        if self.use_llm:
            prompt = build_data_prompt(snapshot, news)

            # Log & store the prompt
            if self.obs:
                self.obs.log_llm_prompt(ticker, date, prompt)

            logger.info(f"  Calling Claude API for {ticker} on {date}...")
            thesis, decision = call_claude(prompt, observer=self.obs, ticker=ticker, date=date)

            if decision:
                self.analysis_log.append({
                    "ticker": ticker,
                    "date": date,
                    "decision": decision,
                    "thesis_length": len(thesis) if thesis else 0,
                    "mode": "llm",
                })
                return thesis or "", decision

            # LLM failed — fall back to rule-based
            logger.warning(f"  LLM failed for {ticker} on {date}; using rule-based fallback")

        thesis, decision = rule_based_decision(snapshot, signal_label)

        if self.obs:
            tech = snapshot.get("technical_indicators", {})
            # Count bullish/bearish for logging
            bullish = sum(1 for x in [
                tech.get("rsi", 50) < 30,
                (tech.get("macd", 0) or 0) > (tech.get("macd_signal", 0) or 0),
                (tech.get("sma_50", 0) or 0) > (tech.get("sma_200", 0) or 0),
            ] if x)
            bearish = 3 - bullish
            self.obs.log_rule_based_decision(ticker, date, decision, bullish, bearish)

        self.analysis_log.append({
            "ticker": ticker,
            "date": date,
            "decision": decision,
            "thesis_length": len(thesis),
            "mode": "rule_based",
        })
        return thesis, decision

    def get_log(self) -> List[Dict]:
        """Return the full analysis log."""
        return self.analysis_log
