"""Market intelligence package — public news/community feeds + LLM sentiment.

PAPER TRADING ONLY. Everything in this package reads *public, keyless*
sources (Google News RSS, reddit's public ``.json`` listings, the
alternative.me Fear & Greed index) and scores them with the local Ollama
server. Nothing here can place an order or touch an API key.

Modules:
    news: defensive fetchers with a 10-minute in-module TTL cache.
    sentiment: :class:`SentimentAgent` (batch LLM scoring, persisted to the
        ``coin_sentiment`` table) plus the ``intel_config`` get/set helpers.
"""
