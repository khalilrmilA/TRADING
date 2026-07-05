"""Public news & community fetchers — keyless, defensive, cached.

PAPER TRADING ONLY. This module talks exclusively to *public* endpoints that
require no authentication of any kind:

* Google News RSS search (``news.google.com/rss/search``) — parsed with
  stdlib ``xml.etree`` and tolerant of malformed XML.
* reddit's public listing JSON (``reddit.com/r/<sub>/hot.json``) — sent with
  a descriptive User-Agent; 403/429 blocks are tolerated silently.
* The alternative.me crypto Fear & Greed index (``api.alternative.me/fng/``).

Contract guarantees (CONTRACTS.md, "backend/intel/"):

* Fetchers NEVER raise — any network/HTTP/parse failure is logged and an
  empty list (or ``None`` for the Fear & Greed index) is returned.
* Nothing runs at import time; the first network request happens on the
  first call.
* Results are cached in-module for 10 minutes per ``(function, args)`` key
  using ``time.monotonic`` — the SentimentAgent, the API and the schedulers
  can all call these freely without hammering the public endpoints.
"""

from __future__ import annotations

import logging
import threading
import time
import xml.etree.ElementTree as ET
from typing import Any, Callable
from urllib.parse import quote_plus

import requests

logger = logging.getLogger(__name__)

__all__ = [
    "COIN_NAMES",
    "clear_cache",
    "fetch_fear_greed",
    "fetch_google_news",
    "fetch_reddit_hot",
]

# --------------------------------------------------------------------------- #
# Constants                                                                    #
# --------------------------------------------------------------------------- #

#: Watchlist symbol → (full name, ticker) for news queries and prompts.
#: Mirrors the AutoTrader's 20-coin watchlist (BotConfig, BTCUSDT..FILUSDT).
COIN_NAMES: dict[str, tuple[str, str]] = {
    "BTCUSDT": ("Bitcoin", "BTC"),
    "ETHUSDT": ("Ethereum", "ETH"),
    "SOLUSDT": ("Solana", "SOL"),
    "BNBUSDT": ("BNB", "BNB"),
    "XRPUSDT": ("XRP", "XRP"),
    "DOGEUSDT": ("Dogecoin", "DOGE"),
    "ADAUSDT": ("Cardano", "ADA"),
    "AVAXUSDT": ("Avalanche", "AVAX"),
    "LINKUSDT": ("Chainlink", "LINK"),
    "DOTUSDT": ("Polkadot", "DOT"),
    "TRXUSDT": ("Tron", "TRX"),
    "LTCUSDT": ("Litecoin", "LTC"),
    "BCHUSDT": ("Bitcoin Cash", "BCH"),
    "UNIUSDT": ("Uniswap", "UNI"),
    "ATOMUSDT": ("Cosmos", "ATOM"),
    "NEARUSDT": ("NEAR Protocol", "NEAR"),
    "APTUSDT": ("Aptos", "APT"),
    "ARBUSDT": ("Arbitrum", "ARB"),
    "OPUSDT": ("Optimism", "OP"),
    "FILUSDT": ("Filecoin", "FIL"),
}

_GOOGLE_NEWS_RSS_URL = "https://news.google.com/rss/search"
_REDDIT_BASE_URL = "https://www.reddit.com"
_FEAR_GREED_URL = "https://api.alternative.me/fng/"

#: (connect, read) timeout — public feeds must answer fast or be skipped.
_TIMEOUT: tuple[float, float] = (10.0, 10.0)

#: Browser-ish UA for Google News (bare library UAs get served empty feeds).
_NEWS_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "trading-ai-platform/1.0 (paper-trading research; public data only)"
    )
}
#: Descriptive UA per reddit's API etiquette (still unauthenticated/public).
_REDDIT_HEADERS = {
    "User-Agent": (
        "windows:trading-ai-platform:1.0 "
        "(paper-trading research bot; public read-only; no account)"
    )
}

#: In-module TTL cache: {(func_name, *args): (expires_monotonic, value)}.
_CACHE_TTL_S = 600.0
_cache: dict[tuple[Any, ...], tuple[float, Any]] = {}
_cache_lock = threading.Lock()


# --------------------------------------------------------------------------- #
# Cache helpers                                                                #
# --------------------------------------------------------------------------- #


def clear_cache() -> None:
    """Drop every cached fetch result (used by tests and manual refreshes)."""
    with _cache_lock:
        _cache.clear()


def _cached(key: tuple[Any, ...], compute: Callable[[], Any]) -> Any:
    """Return the cached value for ``key`` or compute-and-cache it.

    Failure sentinels (``[]`` / ``None``) are cached too: a dead endpoint is
    retried at most once per TTL window instead of on every call.

    Args:
        key: Cache key — function name plus its normalised arguments.
        compute: Zero-argument callable producing the fresh value.

    Returns:
        The cached or freshly computed value.
    """
    now = time.monotonic()
    with _cache_lock:
        hit = _cache.get(key)
        if hit is not None and hit[0] > now:
            return hit[1]
    value = compute()
    with _cache_lock:
        _cache[key] = (now + _CACHE_TTL_S, value)
    return value


# --------------------------------------------------------------------------- #
# Google News RSS                                                              #
# --------------------------------------------------------------------------- #


def _first_text(item: ET.Element, tag: str) -> str:
    """Text of the first ``tag`` child of ``item`` (stripped, '' when absent)."""
    child = item.find(tag)
    if child is None or child.text is None:
        return ""
    return child.text.strip()


def fetch_google_news(query: str, limit: int = 8) -> list[dict[str, Any]]:
    """Fetch recent Google News headlines for a crypto-related query.

    Hits ``news.google.com/rss/search?q=<query>+crypto`` (query URL-encoded,
    "crypto" appended to keep results on-topic) with a User-Agent header and
    parses the RSS with ``xml.etree``. Malformed XML, HTTP errors, timeouts —
    anything at all — degrades to an empty list. Results are cached for 10
    minutes per ``(query, limit)``.

    Args:
        query: Free-text search, e.g. ``"Bitcoin BTC"``.
        limit: Maximum number of headlines to return.

    Returns:
        Up to ``limit`` dicts ``{"title", "source", "url", "published"}``
        (all strings), newest first as served by the feed; ``[]`` on any
        failure.
    """
    key = ("google_news", str(query).strip().lower(), int(limit))
    return _cached(key, lambda: _fetch_google_news_uncached(query, limit))


def _fetch_google_news_uncached(query: str, limit: int) -> list[dict[str, Any]]:
    """Uncached body of :func:`fetch_google_news` (never raises)."""
    url = (
        f"{_GOOGLE_NEWS_RSS_URL}?q={quote_plus(f'{query} crypto')}"
        "&hl=en-US&gl=US&ceid=US:en"
    )
    try:
        resp = requests.get(url, headers=_NEWS_HEADERS, timeout=_TIMEOUT)
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
    except ET.ParseError as exc:
        logger.warning("Google News RSS for %r is malformed XML: %s", query, exc)
        return []
    except Exception as exc:  # noqa: BLE001 — contract: never raises
        logger.warning("Google News fetch failed for %r: %s", query, exc)
        return []

    items: list[dict[str, Any]] = []
    try:
        for item in root.iter("item"):
            title = _first_text(item, "title")
            if not title:
                continue
            items.append(
                {
                    "title": title,
                    "source": _first_text(item, "source"),
                    "url": _first_text(item, "link"),
                    "published": _first_text(item, "pubDate"),
                }
            )
            if len(items) >= max(0, int(limit)):
                break
    except Exception as exc:  # noqa: BLE001 — half-parsed feed is still usable
        logger.warning("Google News RSS items for %r unparsable: %s", query, exc)
    logger.debug("Google News: %d headlines for %r", len(items), query)
    return items


# --------------------------------------------------------------------------- #
# Reddit                                                                       #
# --------------------------------------------------------------------------- #


def fetch_reddit_hot(
    subreddit: str = "CryptoCurrency", limit: int = 25
) -> list[dict[str, Any]]:
    """Fetch the hot post titles of a subreddit via the public ``.json`` API.

    Unauthenticated read-only listing with a descriptive User-Agent. Reddit
    sometimes answers 403/429 to datacenter/burst traffic — that is tolerated
    silently (logged at INFO, empty list returned). Cached for 10 minutes per
    ``(subreddit, limit)``.

    Args:
        subreddit: Subreddit name without the ``r/`` prefix.
        limit: Maximum number of posts to return (reddit caps at 100).

    Returns:
        Up to ``limit`` dicts ``{"title", "score", "num_comments", "url",
        "subreddit"}``; ``[]`` on any failure.
    """
    key = ("reddit_hot", str(subreddit).strip().lower(), int(limit))
    return _cached(key, lambda: _fetch_reddit_hot_uncached(subreddit, limit))


def _fetch_reddit_hot_uncached(subreddit: str, limit: int) -> list[dict[str, Any]]:
    """Uncached body of :func:`fetch_reddit_hot` (never raises)."""
    url = f"{_REDDIT_BASE_URL}/r/{quote_plus(str(subreddit))}/hot.json"
    try:
        resp = requests.get(
            url,
            headers=_REDDIT_HEADERS,
            params={"limit": max(1, int(limit)), "raw_json": 1},
            timeout=_TIMEOUT,
        )
        if resp.status_code in (403, 429):
            # Expected for unauthenticated bursts — not worth a warning.
            logger.info(
                "Reddit r/%s returned HTTP %d (rate-limited/blocked) — skipping",
                subreddit,
                resp.status_code,
            )
            return []
        resp.raise_for_status()
        payload = resp.json()
    except Exception as exc:  # noqa: BLE001 — contract: never raises
        logger.warning("Reddit fetch failed for r/%s: %s", subreddit, exc)
        return []

    posts: list[dict[str, Any]] = []
    try:
        children = (payload.get("data") or {}).get("children") or []
        for child in children:
            data = child.get("data") if isinstance(child, dict) else None
            if not isinstance(data, dict):
                continue
            title = str(data.get("title") or "").strip()
            if not title:
                continue
            permalink = str(data.get("permalink") or "")
            posts.append(
                {
                    "title": title,
                    "score": int(data.get("score") or 0),
                    "num_comments": int(data.get("num_comments") or 0),
                    "url": f"{_REDDIT_BASE_URL}{permalink}" if permalink else "",
                    "subreddit": str(data.get("subreddit") or subreddit),
                }
            )
            if len(posts) >= max(0, int(limit)):
                break
    except Exception as exc:  # noqa: BLE001 — half-parsed listing is fine
        logger.warning("Reddit listing for r/%s unparsable: %s", subreddit, exc)
    logger.debug("Reddit: %d hot posts from r/%s", len(posts), subreddit)
    return posts


# --------------------------------------------------------------------------- #
# Fear & Greed index                                                           #
# --------------------------------------------------------------------------- #


def fetch_fear_greed() -> dict[str, Any] | None:
    """Fetch the current crypto Fear & Greed index from alternative.me.

    Public, keyless endpoint. Cached for 10 minutes (the index itself only
    updates daily).

    Returns:
        ``{"value": int (0-100), "classification": str}`` — e.g.
        ``{"value": 54, "classification": "Neutral"}`` — or ``None`` on any
        failure.
    """
    return _cached(("fear_greed",), _fetch_fear_greed_uncached)


def _fetch_fear_greed_uncached() -> dict[str, Any] | None:
    """Uncached body of :func:`fetch_fear_greed` (never raises)."""
    try:
        resp = requests.get(
            _FEAR_GREED_URL,
            headers=_NEWS_HEADERS,
            params={"limit": 1, "format": "json"},
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        payload = resp.json()
        entry = (payload.get("data") or [None])[0]
        if not isinstance(entry, dict):
            raise ValueError("missing data[0] entry")
        result = {
            "value": int(entry["value"]),
            "classification": str(entry.get("value_classification") or ""),
        }
    except Exception as exc:  # noqa: BLE001 — contract: never raises
        logger.warning("Fear & Greed fetch failed: %s", exc)
        return None
    logger.debug(
        "Fear & Greed: %d (%s)", result["value"], result["classification"]
    )
    return result
