"""Our rule-based calendars against independent references, 1990 to 2030.

The fixture files were generated from exchange_calendars (NYSE sessions) and
python-holidays (US federal holidays); see the header line of each file.
"""

from datetime import date, timedelta
from pathlib import Path

from tradetracker.features.calendar import FEDERAL_EXTRA, NYSE, SEC

REF = Path(__file__).parent / "fixtures" / "calendars"
START, END = date(1990, 1, 1), date(2030, 12, 31)


def _dates(name):
    lines = (REF / name).read_text().splitlines()
    return {date.fromisoformat(l.split("\t")[0]) for l in lines if l and not l.startswith("#")}


def _weekdays():
    d = START
    while d <= END:
        if d.weekday() < 5:
            yield d
        d += timedelta(days=1)


def test_nyse_matches_exchange_calendars_for_forty_years():
    closed = _dates("xnys_closures.txt")
    assert [d for d in _weekdays() if NYSE.is_business_day(d) == (d in closed)] == []


def test_sec_matches_federal_holidays_plus_ad_hoc_closures():
    closed = _dates("us_federal_holidays.txt") | FEDERAL_EXTRA
    assert [d for d in _weekdays() if SEC.is_business_day(d) == (d in closed)] == []
