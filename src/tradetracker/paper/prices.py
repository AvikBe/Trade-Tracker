"""Daily bars for the paper book: Yahoo's chart API, the backtest's price source.

Yahoo is free and needs no key, and using the backtest's own source keeps the live
inputs (drawdown, drift, dollar volume) measured the way the model was fitted. Bars are
fetched once per ticker per process and kept in memory.
"""

from __future__ import annotations

import logging
from datetime import date

from ..backtest.prices import Bars
from ..backtest.yahoo import YahooClient

log = logging.getLogger(__name__)

HISTORY_DAYS = 600           # 250 sessions of drawdown plus the 20-session window, with room


class YahooPrices:
    def __init__(self, client: YahooClient | None = None, start: date | None = None,
                 end: date | None = None):
        self.client = client or YahooClient()
        self.start = start or date.fromordinal(date.today().toordinal() - HISTORY_DAYS)
        self.end = end
        self.cache: dict[str, Bars | None] = {}
        self.status: dict[str, str] = {}

    def get(self, ticker: str) -> Bars | None:
        if ticker not in self.cache:
            status, bars = self.client.daily(ticker, self.start, self.end)
            self.status[ticker] = status
            self.cache[ticker] = Bars(bars) if bars else None
            if status != "ok":
                log.info("yahoo %s: %s", ticker, status)
        return self.cache[ticker]
