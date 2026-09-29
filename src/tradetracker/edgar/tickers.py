"""Current CIK -> ticker mapping from SEC's company_tickers.json.

This is today's mapping only. The backtest relies on the ticker printed on each
filing (point in time); this file fills gaps and seeds ticker_history.
"""

from .client import EdgarClient

URL = "https://www.sec.gov/files/company_tickers.json"


def parse_company_tickers(data: dict) -> dict[str, str]:
    out: dict[str, str] = {}
    for row in data.values():
        # A CIK with several share classes appears once per ticker; the first is the primary.
        out.setdefault(str(row["cik_str"]), row["ticker"].upper())
    return out


def fetch_company_tickers(client: EdgarClient) -> dict[str, str]:
    return parse_company_tickers(client.get(URL).json())
