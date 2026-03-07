"""
Data Collection Module for Trading-R1 Backtest

Collects data from free sources matching the paper's methodology:
- Price/Volume: Yahoo Finance (OHLCV, 15-day rolling window)
- Technical Indicators: Computed via `ta` library (Table S2)
- Fundamentals: Yahoo Finance quarterly financials
- News: Finnhub API (free tier) + Google News RSS fallback
- Sentiment: Analyst recommendations from Yahoo Finance
"""
import logging
import datetime
import time
from typing import Dict, List, Optional
from urllib.parse import quote_plus

import numpy as np
import pandas as pd
import yfinance as yf
import requests

try:
    import feedparser
    HAS_FEEDPARSER = True
except ImportError:
    HAS_FEEDPARSER = False

try:
    import ta
    HAS_TA = True
except ImportError:
    HAS_TA = False

import config

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Price & Volume Data
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_price_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch OHLCV data from Yahoo Finance.
    Includes extra lookback for indicator warm-up.
    """
    start_dt = pd.Timestamp(start_date) - pd.Timedelta(days=config.DATA_LOOKBACK_DAYS + 250)
    logger.info(f"Fetching price data for {ticker} from {start_dt.date()} to {end_date}")

    try:
        stock = yf.Ticker(ticker)
        df = stock.history(start=str(start_dt.date()), end=end_date, auto_adjust=True)
        if df.empty:
            logger.warning(f"No price data returned for {ticker}")
            return pd.DataFrame()

        df.index = pd.to_datetime(df.index).tz_localize(None)
        df = df[["Open", "High", "Low", "Close", "Volume"]]
        df.columns = ["open", "high", "low", "close", "volume"]
        logger.info(f"  {ticker}: {len(df)} trading days loaded")
        return df

    except Exception as e:
        logger.error(f"Error fetching price data for {ticker}: {e}")
        return pd.DataFrame()


# ═══════════════════════════════════════════════════════════════════════════════
# Technical Indicators (Table S2 in paper)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute technical indicators matching Table S2 of the paper.
    Uses the `ta` library when available, otherwise falls back to manual calculation.
    """
    if df.empty:
        return df

    result = df.copy()
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    # ── Moving Averages ───────────────────────────────────────────────────
    for period in config.TECHNICAL_INDICATORS["moving_averages"]["sma"]:
        result[f"sma_{period}"] = close.rolling(window=period).mean()

    for period in config.TECHNICAL_INDICATORS["moving_averages"]["ema"]:
        result[f"ema_{period}"] = close.ewm(span=period, adjust=False).mean()

    if HAS_TA:
        # ── MACD Family ──────────────────────────────────────────────────
        macd_cfg = config.TECHNICAL_INDICATORS["macd"]
        macd_ind = ta.trend.MACD(close, window_slow=macd_cfg["slow"],
                                  window_fast=macd_cfg["fast"],
                                  window_sign=macd_cfg["signal"])
        result["macd"] = macd_ind.macd()
        result["macd_signal"] = macd_ind.macd_signal()
        result["macd_histogram"] = macd_ind.macd_diff()

        # ── RSI ───────────────────────────────────────────────────────────
        rsi_period = config.TECHNICAL_INDICATORS["rsi"]["period"]
        result["rsi"] = ta.momentum.RSIIndicator(close, window=rsi_period).rsi()

        # ── Bollinger Bands ───────────────────────────────────────────────
        bb_cfg = config.TECHNICAL_INDICATORS["bollinger"]
        bb = ta.volatility.BollingerBands(close, window=bb_cfg["period"],
                                           window_dev=bb_cfg["std_dev"])
        result["bb_upper"] = bb.bollinger_hband()
        result["bb_middle"] = bb.bollinger_mavg()
        result["bb_lower"] = bb.bollinger_lband()

        # ── ATR (Volatility) ─────────────────────────────────────────────
        atr_period = config.TECHNICAL_INDICATORS["atr"]["period"]
        result["atr"] = ta.volatility.AverageTrueRange(
            high, low, close, window=atr_period
        ).average_true_range()

        # ── ADX (Trend Strength) ──────────────────────────────────────────
        adx_period = config.TECHNICAL_INDICATORS["adx"]["period"]
        adx_ind = ta.trend.ADXIndicator(high, low, close, window=adx_period)
        result["adx"] = adx_ind.adx()
        result["adx_pos"] = adx_ind.adx_pos()
        result["adx_neg"] = adx_ind.adx_neg()

        # ── Stochastic Oscillator ─────────────────────────────────────────
        stoch_cfg = config.TECHNICAL_INDICATORS["stochastic"]
        stoch = ta.momentum.StochasticOscillator(
            high, low, close,
            window=stoch_cfg["k_period"],
            smooth_window=stoch_cfg["d_period"]
        )
        result["stoch_k"] = stoch.stoch()
        result["stoch_d"] = stoch.stoch_signal()

        # ── ROC (Rate of Change) ──────────────────────────────────────────
        result["roc_10"] = ta.momentum.ROCIndicator(close, window=10).roc()

        # ── MFI (Money Flow Index) ────────────────────────────────────────
        result["mfi"] = ta.volume.MFIIndicator(high, low, close, volume, window=14).money_flow_index()

        # ── VWMA approximation ────────────────────────────────────────────
        result["vwma_20"] = (close * volume).rolling(20).sum() / volume.rolling(20).sum()

    else:
        # ── Manual fallback for key indicators ────────────────────────────
        logger.warning("'ta' library not available; computing basic indicators manually")

        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        result["rsi"] = 100 - (100 / (1 + rs))

        # MACD
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        result["macd"] = ema12 - ema26
        result["macd_signal"] = result["macd"].ewm(span=9, adjust=False).mean()
        result["macd_histogram"] = result["macd"] - result["macd_signal"]

        # Bollinger Bands
        sma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        result["bb_upper"] = sma20 + 2 * std20
        result["bb_middle"] = sma20
        result["bb_lower"] = sma20 - 2 * std20

        # ATR
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ], axis=1).max(axis=1)
        result["atr"] = tr.rolling(14).mean()

    # ── Z-Score (Volatility - Table S2) ───────────────────────────────────
    result["zscore_75"] = (close - close.rolling(75).mean()) / close.rolling(75).std()

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Fundamentals
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_fundamentals(ticker: str) -> Dict:
    """
    Fetch quarterly fundamentals from Yahoo Finance.
    Returns the most recent available data for use in LLM prompts.
    """
    logger.info(f"Fetching fundamentals for {ticker}")
    try:
        stock = yf.Ticker(ticker)
        info = stock.info or {}

        fundamentals = {
            "market_cap": info.get("marketCap"),
            "pe_ratio": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "pb_ratio": info.get("priceToBook"),
            "dividend_yield": info.get("dividendYield"),
            "revenue": info.get("totalRevenue"),
            "net_income": info.get("netIncomeToCommon"),
            "profit_margin": info.get("profitMargins"),
            "operating_margin": info.get("operatingMargins"),
            "roe": info.get("returnOnEquity"),
            "debt_to_equity": info.get("debtToEquity"),
            "current_ratio": info.get("currentRatio"),
            "free_cash_flow": info.get("freeCashflow"),
            "earnings_growth": info.get("earningsGrowth"),
            "revenue_growth": info.get("revenueGrowth"),
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "short_name": info.get("shortName", ticker),
        }
        return fundamentals

    except Exception as e:
        logger.error(f"Error fetching fundamentals for {ticker}: {e}")
        return {}


# ═══════════════════════════════════════════════════════════════════════════════
# News Collection
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_news_finnhub(ticker: str, target_date: str, api_key: str) -> List[Dict]:
    """Fetch company news from Finnhub API (free tier: 60 calls/min)."""
    if not api_key:
        return []

    target_dt = pd.Timestamp(target_date)
    from_date = (target_dt - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
    to_date = target_dt.strftime("%Y-%m-%d")

    url = "https://finnhub.io/api/v1/company-news"
    params = {"symbol": ticker, "from": from_date, "to": to_date, "token": api_key}

    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        articles = resp.json()

        news_items = []
        for art in articles[:50]:  # Cap at 50 raw articles
            pub_date = datetime.datetime.fromtimestamp(art.get("datetime", 0))
            news_items.append({
                "headline": art.get("headline", ""),
                "summary": art.get("summary", "")[:300],
                "source": art.get("source", ""),
                "date": pub_date.strftime("%Y-%m-%d"),
                "days_ago": (target_dt - pd.Timestamp(pub_date)).days,
            })
        return news_items

    except Exception as e:
        logger.warning(f"Finnhub news fetch failed for {ticker}: {e}")
        return []


def fetch_news_google_rss(ticker: str, company_name: str = "",
                          target_date: str = "") -> List[Dict]:
    """
    Fallback: fetch news from Google News RSS feed.

    NOTE: Google News RSS only returns *current* articles. For historical
    backtests, this means news from the actual trading date is NOT available —
    only today's news. The `target_date` parameter is used to compute
    `days_ago` relative to the target, which will be very large for old dates.
    """
    if not HAS_FEEDPARSER:
        logger.warning("feedparser not installed; skipping Google News RSS")
        return []

    query = f"{ticker} stock" if not company_name else f"{company_name} {ticker}"
    url = f"https://news.google.com/rss/search?q={quote_plus(query)}&hl=en-US&gl=US&ceid=US:en"

    try:
        # Use requests with a real User-Agent header (Google blocks bare feedparser)
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/120.0.0.0 Safari/537.36"
        }
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        feed = feedparser.parse(resp.text)

        target_dt = pd.Timestamp(target_date) if target_date else pd.Timestamp.now()
        news_items = []
        for entry in feed.entries[:30]:
            pub_date_str = entry.get("published", "")
            # Parse the RSS date to compute days_ago relative to the target date
            try:
                if hasattr(entry, "published_parsed") and entry.published_parsed:
                    pub_dt = pd.Timestamp(datetime.datetime(*entry.published_parsed[:6]))
                    days_ago = max(0, (target_dt - pub_dt).days)
                else:
                    days_ago = 0
            except Exception:
                days_ago = 0

            news_items.append({
                "headline": entry.get("title", ""),
                "summary": entry.get("summary", "")[:300],
                "source": entry.get("source", {}).get("title", "Google News"),
                "date": pub_date_str,
                "days_ago": days_ago,
            })

        if not news_items:
            logger.info(f"Google News RSS returned 0 articles for {ticker} "
                        f"(feed had {len(feed.entries)} entries)")

        return news_items

    except Exception as e:
        logger.warning(f"Google News RSS fetch failed for {ticker}: {e}")
        return []


def fetch_news(ticker: str, target_date: str, company_name: str = "") -> Dict[str, List[Dict]]:
    """
    Collect news bucketed by temporal horizon (Table S1 in paper).
    Tries Finnhub first, falls back to Google News RSS.
    """
    all_news = fetch_news_finnhub(ticker, target_date, config.FINNHUB_API_KEY)
    source = "finnhub"
    if not all_news:
        all_news = fetch_news_google_rss(ticker, company_name, target_date)
        source = "google_rss"

    # Bucket by temporal horizon (Table S1)
    buckets = {"last_3_days": [], "last_4_10_days": [], "last_11_30_days": []}
    skipped_too_old = 0
    for item in all_news:
        days_ago = item.get("days_ago", 0)
        if days_ago <= 3:
            buckets["last_3_days"].append(item)
        elif days_ago <= 10:
            buckets["last_4_10_days"].append(item)
        elif days_ago <= 30:
            buckets["last_11_30_days"].append(item)
        else:
            skipped_too_old += 1

    # Cap per bucket (Table S1)
    for key, cfg in config.NEWS_BUCKETS.items():
        buckets[key] = buckets[key][:cfg["max_samples"]]

    total = sum(len(v) for v in buckets.values())

    if total == 0 and skipped_too_old > 0 and source == "google_rss":
        logger.info(
            f"  {ticker}: Google RSS returned {skipped_too_old} articles but all are from "
            f"today (>30 days from backtest date {target_date}). "
            f"Historical news requires a Finnhub API key."
        )
    elif total == 0:
        logger.info(f"  {ticker} news: 0 articles found for {target_date}")
    else:
        logger.info(f"  {ticker} news: {total} articles across 3 buckets")

    return buckets


# ═══════════════════════════════════════════════════════════════════════════════
# Analyst Sentiment (Section S1.1.4)
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_analyst_recommendations(ticker: str) -> Dict:
    """Fetch analyst recommendations from Yahoo Finance."""
    try:
        stock = yf.Ticker(ticker)
        recs = stock.recommendations
        if recs is not None and not recs.empty:
            latest = recs.iloc[-1] if len(recs) > 0 else {}
            rec_summary = {
                "strong_buy": int(latest.get("strongBuy", 0)),
                "buy": int(latest.get("buy", 0)),
                "hold": int(latest.get("hold", 0)),
                "sell": int(latest.get("sell", 0)),
                "strong_sell": int(latest.get("strongSell", 0)),
            }
            total = sum(rec_summary.values())
            if total > 0:
                rec_summary["consensus_score"] = round(
                    (rec_summary["strong_buy"] * 5 + rec_summary["buy"] * 4 +
                     rec_summary["hold"] * 3 + rec_summary["sell"] * 2 +
                     rec_summary["strong_sell"] * 1) / total, 2
                )
            else:
                rec_summary["consensus_score"] = 3.0
            return rec_summary
        return {}
    except Exception as e:
        logger.warning(f"Could not fetch recommendations for {ticker}: {e}")
        return {}


# ═══════════════════════════════════════════════════════════════════════════════
# Unified Data Collector
# ═══════════════════════════════════════════════════════════════════════════════

class DataCollector:
    """
    Orchestrates all data collection for a given ticker and date range.
    Accepts an optional Observer for rich logging and input storage.
    """

    def __init__(self, ticker: str, start_date: str, end_date: str, observer=None):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.price_data: pd.DataFrame = pd.DataFrame()
        self.fundamentals: Dict = {}
        self.analyst_recs: Dict = {}
        self.news_cache: Dict[str, Dict] = {}
        self.obs = observer  # Observer instance (optional)

    def collect_all(self) -> Dict:
        """Run full data collection pipeline. Returns a dict of all data."""
        logger.info(f"═══ Collecting data for {self.ticker} ═══")

        # 1. Price + Technical Indicators
        self.price_data = fetch_price_data(self.ticker, self.start_date, self.end_date)
        if self.obs:
            self.obs.log_price_data(self.ticker, self.price_data)

        if not self.price_data.empty:
            self.price_data = compute_technical_indicators(self.price_data)
            if self.obs:
                self.obs.log_technical_indicators(self.ticker, self.price_data)

        # 2. Fundamentals
        self.fundamentals = fetch_fundamentals(self.ticker)
        if self.obs:
            self.obs.log_fundamentals(self.ticker, self.fundamentals)

        # 3. Analyst Sentiment
        self.analyst_recs = fetch_analyst_recommendations(self.ticker)
        if self.obs:
            self.obs.log_analyst_recs(self.ticker, self.analyst_recs)

        return {
            "ticker": self.ticker,
            "price_data": self.price_data,
            "fundamentals": self.fundamentals,
            "analyst_recs": self.analyst_recs,
        }

    def get_news_for_date(self, target_date: str) -> Dict[str, List[Dict]]:
        """Fetch (cached) news for a specific trading date."""
        if target_date not in self.news_cache:
            company_name = self.fundamentals.get("short_name", "")
            self.news_cache[target_date] = fetch_news(
                self.ticker, target_date, company_name
            )
            if self.obs:
                total = sum(len(v) for v in self.news_cache[target_date].values())
                note = ""
                if total == 0 and not config.FINNHUB_API_KEY:
                    note = "historical news requires FINNHUB_API_KEY in .env"
                self.obs.log_news(self.ticker, self.news_cache[target_date], note=note)
        return self.news_cache[target_date]

    def get_snapshot(self, target_date: str) -> Dict:
        """
        Build a complete data snapshot for a single trading date.
        This is used to construct the LLM prompt.
        """
        if self.price_data.empty:
            return {}

        target_dt = pd.Timestamp(target_date)
        mask = self.price_data.index <= target_dt
        available = self.price_data[mask]

        if available.empty:
            return {}

        # Last 15 trading days of price data (paper: 15-day rolling window)
        recent_prices = available.tail(15)
        latest = available.iloc[-1]

        # Technical summary
        tech_summary = {}
        for col in available.columns:
            if col not in ["open", "high", "low", "close", "volume"]:
                val = latest.get(col)
                if pd.notna(val):
                    tech_summary[col] = round(float(val), 4)

        # Price summary
        price_summary = {
            "current_price": round(float(latest["close"]), 2),
            "open": round(float(latest["open"]), 2),
            "high": round(float(latest["high"]), 2),
            "low": round(float(latest["low"]), 2),
            "volume": int(latest["volume"]),
            "price_change_1d": round(float(
                (latest["close"] - available.iloc[-2]["close"]) / available.iloc[-2]["close"] * 100
            ), 2) if len(available) >= 2 else 0.0,
            "price_change_5d": round(float(
                (latest["close"] - available.iloc[-6]["close"]) / available.iloc[-6]["close"] * 100
            ), 2) if len(available) >= 6 else 0.0,
            "price_change_15d": round(float(
                (latest["close"] - available.iloc[-16]["close"]) / available.iloc[-16]["close"] * 100
            ), 2) if len(available) >= 16 else 0.0,
        }

        # Recent OHLCV table
        ohlcv_table = []
        for idx, row in recent_prices.iterrows():
            ohlcv_table.append({
                "date": idx.strftime("%Y-%m-%d"),
                "open": round(float(row["open"]), 2),
                "high": round(float(row["high"]), 2),
                "low": round(float(row["low"]), 2),
                "close": round(float(row["close"]), 2),
                "volume": int(row["volume"]),
            })

        return {
            "ticker": self.ticker,
            "date": target_date,
            "price_summary": price_summary,
            "ohlcv_table": ohlcv_table,
            "technical_indicators": tech_summary,
            "fundamentals": self.fundamentals,
            "analyst_recommendations": self.analyst_recs,
        }
