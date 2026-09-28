from datetime import date, timedelta

import pytest

from tradetracker.features.calendar import NYSE, SEC, easter, federal_holidays, nyse_holidays

# Published NYSE full-day closures.
NYSE_PUBLISHED = {
    2021: ["01-01", "01-18", "02-15", "04-02", "05-31", "07-05", "09-06", "11-25", "12-24"],
    2022: ["01-17", "02-21", "04-15", "05-30", "06-20", "07-04", "09-05", "11-24", "12-26"],
    2023: ["01-02", "01-16", "02-20", "04-07", "05-29", "06-19", "07-04", "09-04", "11-23",
           "12-25"],
    2024: ["01-01", "01-15", "02-19", "03-29", "05-27", "06-19", "07-04", "09-02", "11-28",
           "12-25"],
    2025: ["01-01", "01-09", "01-20", "02-17", "04-18", "05-26", "06-19", "07-04", "09-01",
           "11-27", "12-25"],
    2026: ["01-01", "01-19", "02-16", "04-03", "05-25", "06-19", "07-03", "09-07", "11-26",
           "12-25"],
}

# OPM federal holiday schedules (observed dates), plus executive-order closures.
FEDERAL_PUBLISHED = {
    2021: ["01-01", "01-18", "02-15", "05-31", "06-18", "07-05", "09-06", "10-11", "11-11",
           "11-25", "12-24", "12-31"],
    2023: ["01-02", "01-16", "02-20", "05-29", "06-19", "07-04", "09-04", "10-09", "11-10",
           "11-23", "12-25"],
    2024: ["01-01", "01-15", "02-19", "05-27", "06-19", "07-04", "09-02", "10-14", "11-11",
           "11-28", "12-24", "12-25"],
}


@pytest.mark.parametrize("year", sorted(NYSE_PUBLISHED))
def test_nyse_holidays_match_published_schedule(year):
    expected = {date.fromisoformat(f"{year}-{md}") for md in NYSE_PUBLISHED[year]}
    assert nyse_holidays(year) == expected


@pytest.mark.parametrize("year", sorted(FEDERAL_PUBLISHED))
def test_federal_holidays_match_published_schedule(year):
    expected = {date.fromisoformat(f"{year}-{md}") for md in FEDERAL_PUBLISHED[year]}
    assert federal_holidays(year) == expected


@pytest.mark.parametrize("year,expected", [
    (2000, date(2000, 4, 23)), (2019, date(2019, 4, 21)), (2024, date(2024, 3, 31)),
    (2025, date(2025, 4, 20)), (2038, date(2038, 4, 25)),
])
def test_easter(year, expected):
    assert easter(year) == expected


def test_no_juneteenth_before_it_existed():
    assert date(2020, 6, 19) not in federal_holidays(2020)
    assert date(2021, 6, 18) not in nyse_holidays(2021)  # NYSE started in 2022


def test_new_year_on_saturday_closes_sec_but_not_nyse_the_friday_before():
    # 1 Jan 2022 was a Saturday.
    assert date(2021, 12, 31) in federal_holidays(2021)
    assert date(2021, 12, 31) not in nyse_holidays(2021)
    assert date(2022, 1, 1) not in nyse_holidays(2022)


def test_calendars_differ_where_they_should():
    good_friday = date(2024, 3, 29)
    assert SEC.is_business_day(good_friday) and not NYSE.is_business_day(good_friday)
    columbus = date(2023, 10, 9)
    assert NYSE.is_business_day(columbus) and not SEC.is_business_day(columbus)


@pytest.mark.parametrize("start,end,expected", [
    (date(2024, 3, 4), date(2024, 3, 4), 0),     # same day
    (date(2024, 3, 4), date(2024, 3, 6), 2),     # Mon -> Wed
    (date(2024, 3, 1), date(2024, 3, 4), 1),     # Fri -> Mon
    (date(2024, 3, 2), date(2024, 3, 4), 1),     # Sat -> Mon
    (date(2024, 3, 2), date(2024, 3, 3), 0),     # Sat -> Sun
    (date(2024, 1, 12), date(2024, 1, 16), 1),   # Fri -> Tue over MLK Day
    (date(2023, 10, 6), date(2023, 10, 10), 1),  # Fri -> Tue over Columbus Day
    (date(2024, 3, 28), date(2024, 4, 1), 2),    # Good Friday is an SEC business day
    (date(2023, 12, 29), date(2024, 1, 2), 1),   # over New Year's Day
    (date(2024, 12, 23), date(2024, 12, 26), 1),  # 24th (executive order) and 25th closed
    (date(2024, 3, 4), date(2024, 3, 18), 10),
    (date(2023, 1, 3), date(2024, 1, 3), 250),   # a full year: 261 weekdays - 11 holidays
    (date(2024, 3, 6), date(2024, 3, 4), -2),    # reversed range is negative
])
def test_sec_business_days(start, end, expected):
    assert SEC.business_days_between(start, end) == expected


def test_nyse_trading_days_in_2024():
    assert NYSE.business_days_between(date(2023, 12, 31), date(2024, 12, 31)) == 252
    assert NYSE.business_days_between(date(2024, 3, 28), date(2024, 4, 1)) == 1  # Good Friday


def test_business_days_matches_brute_force():
    start = date(2022, 11, 20)
    for cal in (SEC, NYSE):
        for span in range(0, 60):
            end = start + timedelta(days=span)
            brute = sum(cal.is_business_day(start + timedelta(days=i)) for i in range(1, span + 1))
            assert cal.business_days_between(start, end) == brute, (cal.name, end)


def test_on_or_before_and_add():
    assert NYSE.on_or_before(date(2024, 3, 31)) == date(2024, 3, 28)  # Sun, after Good Friday
    assert NYSE.on_or_before(date(2024, 3, 27)) == date(2024, 3, 27)
    assert NYSE.add(date(2024, 3, 28), 1) == date(2024, 4, 1)
    assert SEC.add(date(2024, 3, 28), 1) == date(2024, 3, 29)
    assert SEC.add(date(2024, 3, 4), 0) == date(2024, 3, 4)
