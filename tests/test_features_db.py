"""The feature job against Postgres, using the synthetic bulk quarter."""

import os
from datetime import date

import pytest
from bulkfixture import write
from tradetracker import cli, store
from tradetracker.edgar.bulk import parse_quarter
from tradetracker.features import job, summary
from tradetracker.prices.tiingo import PriceBar


def _load(conn, tmp_path):
    for f in parse_quarter(write(tmp_path / "q.zip")):
        store.save_filing(conn, f)
    store.save_prices(conn, [
        PriceBar("EXWD", date(2024, 3, 4), 12, 12.6, 11.9, 12.4, 12, 12.0, 1000),
        PriceBar("EXWD", date(2024, 3, 6), 13, 13.5, 12.9, 13.2, 13, 13.2, 1000),
        PriceBar("OTHR", date(2024, 3, 5), 40, 41, 39, 40, 40, 40.0, 1000),
        PriceBar("OTHR", date(2024, 3, 7), 36, 37, 35, 36, 36, 36.0, 1000),
    ])
    conn.commit()


def _latest(conn):
    rows = conn.execute(
        """
        SELECT f.source_filing_id, x.lag_days, x.lag_ratio, x.drift_pct, x.role_weight,
               x.cluster_count, x.trade_value, x.duplicate_of_trade_id IS NOT NULL, x.flags,
               x.filing_date, x.days_since_filing
        FROM latest_trade_features x JOIN trades USING (trade_id) JOIN filings f USING (filing_id)
        ORDER BY 1
        """
    ).fetchall()
    return {r[0]: r[1:] for r in rows}


def test_feature_job_writes_point_in_time_rows(conn, tmp_path):
    _load(conn, tmp_path)
    stats = job.run(conn, as_of=date(2024, 3, 8))
    assert stats["trades"] == 3 and stats["written"] == 3
    rows = _latest(conn)

    lag, ratio, drift, role_w, cluster, value, dup, flags, filed, stale = rows["0000000001-24-000001"]
    assert (lag, float(ratio), float(role_w)) == (2, 1.0, 1.0)  # CFO buy
    assert float(drift) == pytest.approx(10.0)
    assert (cluster, float(value), dup, flags) == (1, 123400.0, False, [])
    assert (filed, stale) == (date(2024, 3, 6), 2)

    # The 4/A repeats that buy: original filing date, flagged as a duplicate.
    lag, _, drift, *_ , dup, flags, filed, _ = rows["0000000001-24-000002"]
    assert (lag, dup, filed) == (2, True, date(2024, 3, 6))
    assert float(drift) == pytest.approx(10.0)
    assert flags == ["amendment_duplicate"]

    # Sale by a 10% owner under a 10b5-1 plan; the price fell 10% before filing.
    lag, _, drift, role_w, *_ , flags, _, _ = rows["0000000001-24-000003"]
    assert (lag, float(role_w), flags) == (2, 0.25, ["10b5_1"])
    assert float(drift) == pytest.approx(10.0)


def test_feature_job_only_fills_missing_unless_recompute(conn, tmp_path):
    _load(conn, tmp_path)
    assert job.run(conn)["written"] == 3
    assert job.run(conn)["written"] == 0
    assert job.run(conn, recompute=True)["written"] == 3
    assert conn.execute("SELECT count(*) FROM trade_features").fetchone()[0] == 6
    assert conn.execute("SELECT count(*) FROM latest_trade_features").fetchone()[0] == 3


def test_feature_job_handles_missing_prices_and_tickers(conn, tmp_path):
    _load(conn, tmp_path)
    conn.execute("DELETE FROM daily_prices WHERE ticker = 'OTHR'")
    conn.execute("UPDATE trades SET ticker = NULL WHERE ticker = 'EXWD'")
    conn.commit()
    job.run(conn)
    flags = {k: v[7] for k, v in _latest(conn).items()}
    assert flags["0000000001-24-000001"] == ["no_ticker"]
    assert flags["0000000001-24-000003"] == ["10b5_1", "no_prices"]


def test_deleting_a_trade_cascades_and_duplicate_link_nulls(conn, tmp_path):
    _load(conn, tmp_path)
    job.run(conn)
    conn.execute(
        "DELETE FROM trades WHERE filing_id = "
        "(SELECT filing_id FROM filings WHERE source_filing_id = '0000000001-24-000001')"
    )
    conn.commit()
    rows = _latest(conn)
    assert "0000000001-24-000001" not in rows
    assert rows["0000000001-24-000002"][6] is False


def test_feature_report_and_cli(conn, tmp_path, monkeypatch, capsys):
    _load(conn, tmp_path)
    monkeypatch.setenv("DATABASE_URL", os.environ["TT_TEST_DATABASE_URL"])
    assert cli.main(["features", "--as-of", "2024-03-08"]) == 0
    assert "'written': 3" in capsys.readouterr().out
    assert cli.main(["feature-report"]) == 0
    out = capsys.readouterr().out
    assert "Coverage" in out and "10b5_1" in out and "on time (<=1x)" in out
    assert summary.render(conn) + "\n" == out
