from datetime import date

from tradetracker.report import business_days


def test_business_days_skips_weekends():
    assert business_days(date(2024, 3, 1), date(2024, 3, 1)) == 0  # same day
    assert business_days(date(2024, 3, 1), date(2024, 3, 4)) == 1  # Fri -> Mon
    assert business_days(date(2024, 3, 4), date(2024, 3, 6)) == 2
    assert business_days(date(2024, 3, 4), date(2024, 3, 18)) == 10
