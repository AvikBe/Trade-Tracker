"""Features on real SEC filings and real closes (tests/fixtures/real/2024q1_features_*).

Expected numbers were worked out by hand from the fixture closes, and drift was checked
against Yahoo Finance adjusted closes as a second source.
"""

import csv
from datetime import date
from pathlib import Path

import pytest

from tradetracker import store
from tradetracker.edgar.bulk import parse_quarter
from tradetracker.features import job

REAL = Path(__file__).parent / "fixtures" / "real"


@pytest.fixture
def loaded(conn):
    for f in parse_quarter(REAL / "2024q1_features_form345.zip"):
        store.save_filing(conn, f)
    store.link_amendments(conn)
    with open(REAL / "2024q1_features_prices.csv") as fh, conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO daily_prices (ticker, date, adj_close) VALUES (%s, %s, %s)",
            [(r["ticker"], r["date"], r["adj_close"]) for r in csv.DictReader(fh)],
        )
    conn.commit()
    job.run(conn, as_of=date(2024, 3, 28))
    return conn


def _rows(conn, accession):
    return conn.execute(
        """
        SELECT t.side, t.trade_date, x.filing_date, x.lag_days, x.drift_pct, x.cluster_count,
               x.role_weight, x.flags, x.duplicate_of_trade_id
        FROM latest_trade_features x JOIN trades t USING (trade_id) JOIN filings f USING (filing_id)
        WHERE f.source_filing_id = %s ORDER BY t.line_no
        """,
        (accession,),
    ).fetchall()


def test_ctbi_board_buying_together_is_a_cluster_of_eleven(loaded):
    # Eleven directors and officers bought on 24 Jan 2024 and filed on 25 Jan.
    for acc in ["0000350852-24-0000%d" % n for n in (20, 21, 22, 23, 24, 25, 27, 28, 29, 31)]:
        side, traded, filed, lag, drift, cluster, *_ = _rows(loaded, acc)[0]
        assert (side, traded, filed, lag, cluster) == (
            "buy", date(2024, 1, 24), date(2024, 1, 25), 1, 11)
        assert float(drift) == pytest.approx((37.9508898938 / 37.5990751313 - 1) * 100)


def test_ctbi_late_line_in_the_same_filing(loaded):
    # Smith reported a 2 Jan purchase 16 SEC business days later (MLK Day skipped).
    early = [r for r in _rows(loaded, "0000350852-24-000026") if r[1] == date(2024, 1, 2)][0]
    assert early[3] == 16 and early[5] == 1
    assert float(early[4]) == pytest.approx((37.9508898938 / 39.6738801409 - 1) * 100)


def test_plce_run_up_before_disclosure(loaded):
    # Bought 13 Feb at an $11.29 close; by the 15 Feb filing the stock closed at $26.29.
    rows = _rows(loaded, "0001104659-24-024492")
    assert {r[3] for r in rows} == {2}
    assert all(float(r[4]) == pytest.approx(132.8609, abs=1e-3) for r in rows)
    # Its 4/A repeats every line and inherits the 15 Feb filing date.
    amended = _rows(loaded, "0001104659-24-025046")
    assert all(r[7] == ["amendment_duplicate"] and r[2] == date(2024, 2, 15) for r in amended)


def test_plce_amendments_filed_under_another_joint_filer_stay_unlinked(loaded):
    # Known gap: the 4/As list Mithaq Capital first, the originals Alrajhi, so milestone 1
    # sees two filers and cannot link them. Their lines are dated by the 4/A.
    orig = _rows(loaded, "0001104659-24-022552")[0]
    amended = _rows(loaded, "0001104659-24-025044")[0]
    assert float(orig[4]) == pytest.approx((11.29 / 12.505 - 1) * 100)
    assert amended[7] == ["amendment"] and amended[2] == date(2024, 2, 16)


def test_nbix_planned_sales_flip_drift_sign(loaded):
    side, _, _, lag, drift, cluster, role_w, flags, _ = _rows(loaded, "0000914475-24-000080")[0]
    assert (side, lag, flags, float(role_w)) == ("sell", 2, ["10b5_1"], 0.5)
    # Price fell from 136.03 to 130.40 after the sale: the move is already captured.
    assert float(drift) == pytest.approx(-(130.4 / 136.03 - 1) * 100)
    assert cluster == 4


def test_rmcf_joint_filers_duplicate_filing_and_amendment_correction(loaded):
    # Two joint filers filed the same three buys the same day: one copy of each counts.
    a = _rows(loaded, "0000950170-24-018834")
    b = _rows(loaded, "0000950170-24-018836")
    for x, y in zip(a, b):
        assert sorted([x[7], y[7]]) == [[], ["duplicate_filing"]]
    corrected = [r for r in _rows(loaded, "0000950170-24-020136") if "amendment_correction" in r[7]]
    assert corrected and all(r[2] == date(2024, 2, 22) for r in corrected)


def test_every_real_trade_gets_a_row_and_recompute_is_stable(loaded):
    before = loaded.execute(
        "SELECT trade_id, lag_days, drift_pct, cluster_count, size_score, filer_lag_zscore, flags "
        "FROM latest_trade_features ORDER BY 1"
    ).fetchall()
    assert len(before) == loaded.execute("SELECT count(*) FROM trades").fetchone()[0] == 99
    job.run(loaded, recompute=True, as_of=date(2024, 3, 28))
    after = loaded.execute(
        "SELECT trade_id, lag_days, drift_pct, cluster_count, size_score, filer_lag_zscore, flags "
        "FROM latest_trade_features ORDER BY 1"
    ).fetchall()
    assert before == after


def test_independent_sql_recomputation_agrees(loaded):
    import importlib.util

    path = Path(__file__).parents[1] / "scripts" / "crosscheck_features.py"
    spec = importlib.util.spec_from_file_location("crosscheck", path)
    crosscheck = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(crosscheck)
    results = crosscheck.check(loaded, "2024-03-28")
    assert {name: wrong for name, (_, wrong) in results.items() if wrong} == {}
    assert results["cluster_count"][0] > 50 and results["drift_pct"][0] > 50
