from tradetracker.cli import _quarters


def test_quarter_ranges_cross_years():
    assert _quarters("2024q3") == [(2024, 3)]
    assert _quarters("2023Q3:2024q2") == [(2023, 3), (2023, 4), (2024, 1), (2024, 2)]
    assert len(_quarters("2015q1:2026q2")) == 46
