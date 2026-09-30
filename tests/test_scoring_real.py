"""Scoring signals on real SEC filings and real prices, checked by hand.

Same fixtures as test_backtest_real.py. Every expected number is typed from the rows of
tests/fixtures/real/2024q1_features_form345.zip (NONDERIV_TRANS.TSV: shares, price,
SHRS_OWND_FOLWNG_TRANS) or tests/fixtures/real/backtest_bars, not produced by the code.
"""

from datetime import date
from pathlib import Path

import pytest

from tradetracker import store
from tradetracker.backtest import events, prices, sectors, study
from tradetracker.edgar.bulk import parse_quarter
from tradetracker.features import job
from tradetracker.scoring import model as M
from tradetracker.scoring import rank as R
from tradetracker.scoring import signals

REAL = Path(__file__).parent / "fixtures" / "real"


@pytest.fixture
def scored(conn):
    for f in parse_quarter(REAL / "2024q1_features_form345.zip"):
        store.save_filing(conn, f)
    store.link_amendments(conn)
    conn.commit()
    job.run(conn, as_of=date(2024, 3, 28))
    evs, _ = events.load_events(conn)
    src = prices.CachePrices(REAL / "backtest_bars")
    res = study.run_study(evs, src, sectors.load_sic(REAL / "backtest_sic.csv"))
    return res, signals.build(res, src), src


def _one(scored, ticker, filing_date):
    res, sigs, _ = scored
    (out,) = [(x, s) for x, s in zip(res, sigs)
              if x.event.ticker == ticker and x.event.filing_date == filing_date
              and x.event.side == "buy"]
    return out


def test_plce_stake_from_lines_filed_out_of_order(scored):
    # Filed 15 Feb (4, 0001104659-24-024492), three buys on 13 Feb, listed out of trade
    # order: 547,172 sh -> 6,614,237 owned; 282,022 -> 6,067,065; 137,150 -> 6,751,387.
    # Held before: 6,067,065 - 282,022 = 5,785,043. Bought 966,344.
    x, s = _one(scored, "PLCE", date(2024, 2, 15))
    assert x.event.shares == 966_344
    assert x.event.shares_held_before == 5_785_043
    assert s.stake_change == pytest.approx(966_344 / 5_785_043)
    assert s.role == "10% owner"
    assert s.repeat_buys == 2          # filed 13 and 14 Feb
    assert M.bucket("stake", s) == "5-20%"


def test_plce_first_buy_more_than_doubled_the_stake(scored):
    # Filed 13 Feb: four buys on 9 Feb. Smallest owned-after minus shares:
    # 1,837,119 - 589,248 = 1,247,871 held before; bought 589,248 + 167,906 + 615,095
    # + 477,148 = 1,849,397.
    x, s = _one(scored, "PLCE", date(2024, 2, 13))
    assert s.stake_change == pytest.approx(1_849_397 / 1_247_871)
    assert s.repeat_buys == 0
    assert M.bucket("stake", s) == ">=100%"


def test_plce_drawdown_from_the_52_week_high(scored):
    # Filed 14 Feb, entry 15 Feb. PLCE adjClose peaked at 28.56 on 15 Nov 2023 (the file
    # starts 1 Nov, 72 sessions before entry); last close before entry 14.515 (14 Feb).
    _, s = _one(scored, "PLCE", date(2024, 2, 14))
    assert s.drawdown == pytest.approx(14.515 / 28.56 - 1)
    assert M.bucket("drawdown", s) == "<=-45%"


def test_rmcf_holdings_are_per_account(scored):
    # Filed 14 Feb by a director through an investment firm (indirect): bought 38,280
    # shares, and the account held 1,016,696 before the first line.
    x, s = _one(scored, "RMCF", date(2024, 2, 14))
    assert x.event.held_before == {"indirect": 1_016_696}
    assert s.stake_change == pytest.approx(38_280 / 1_016_696)


def test_ctbi_new_stake_and_too_little_price_history(scored):
    res, sigs, _ = scored
    ctbi = [(x, s) for x, s in zip(res, sigs)
            if x.event.ticker == "CTBI" and x.event.filing_date == date(2024, 1, 25)]
    assert len(ctbi) == 11
    new = [s for _, s in ctbi if s.stake_change == signals.NEW_STAKE]
    assert len(new) == 2               # two officers held no shares before
    # CTBI bars start 1 Nov 2023: 58 sessions before the 26 Jan entry, under 60.
    assert all(s.drawdown is None for _, s in ctbi)
    assert all(s.cluster_count == 11 and s.track_record is None for _, s in ctbi)


def test_rank_on_real_filings(scored):
    res, sigs, src = scored
    sigs_by = list(zip(res, sigs))
    m = M.Model(tables={"role": {"CEO": 0.01, "CFO": 0.01, "officer": 0.0, "director": 0.0,
                                 "10% owner": 0.005, "none": 0.0}},
                counts={}, weights={"role": 1.0}, beta=3.0, half_life=10)
    out, hidden = R.rank([x for x, _ in sigs_by], [s for _, s in sigs_by], src, m,
                         as_of=date(2024, 2, 16))
    tickers = [r.result.event.ticker for r in out]
    # PLCE ran up 133% after the 13 Feb buys: hidden. RMCF trades ~$40-70k a day: hidden.
    # CTBI's January buys are within 30 sessions and liquid.
    assert "PLCE" not in tickers and "RMCF" not in tickers
    assert tickers == ["CTBI"]
    assert hidden["drift"] >= 1 and hidden["illiquid"] >= 1
    assert out[0].result.event.role in ("CEO", "CFO")


def test_cli_score_backtest_and_rank(scored, tmp_path, monkeypatch, capsys):
    import os

    from tradetracker import cli

    monkeypatch.setenv("DATABASE_URL", os.environ["TT_TEST_DATABASE_URL"])
    out = tmp_path / "run"
    args = ["--prices", str(REAL / "backtest_bars"), "--sic", str(REAL / "backtest_sic.csv")]
    assert cli.main(["score-backtest", *args, "--out", str(out), "--no-ablation"]) == 0
    for name in ("report.md", "model.json", "summary.json", "scored.csv.gz", "pool.csv.gz",
                 "events.csv.gz"):
        assert (out / name).exists()
    # One quarter is too little history for a test year, but the final model still fits.
    m = M.Model.from_json((out / "model.json").read_text())
    assert m.n_train > 0 and set(m.weights) == set(M.COMPONENTS)
    capsys.readouterr()
    assert cli.main(["rank", *args, "--model", str(out / "model.json"),
                     "--as-of", "2024-02-16"]) == 0
    text = capsys.readouterr().out
    assert text.startswith("Top 1 buys as of 2024-02-16")
    assert " 1. CTBI" in text


def test_independent_recomputation_agrees(scored, tmp_path, monkeypatch):
    """scripts/crosscheck_scoring.py rebuilds stake, repeat buys, track record and
    drawdown with SQL and pandas, without tradetracker.scoring."""
    import importlib.util
    import os

    pytest.importorskip("pandas")
    pytest.importorskip("scipy")
    from tradetracker import cli

    monkeypatch.setenv("DATABASE_URL", os.environ["TT_TEST_DATABASE_URL"])
    out = tmp_path / "run"
    assert cli.main(["score-backtest", "--prices", str(REAL / "backtest_bars"), "--sic",
                     str(REAL / "backtest_sic.csv"), "--out", str(out), "--no-ablation"]) == 0
    path = Path(__file__).parents[1] / "scripts" / "crosscheck_scoring.py"
    spec = importlib.util.spec_from_file_location("crosscheck_scoring", path)
    xc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(xc)
    assert xc.run_checks(out, REAL / "backtest_bars", os.environ["TT_TEST_DATABASE_URL"]) == []


def test_cli_cap_floor_hedged_run_and_its_cross_check(scored, tmp_path, monkeypatch):
    """The new-approach flags on real filings: RMCF ($26.5M on 15 Feb 2024) falls under
    the $50M floor, and the cross-check rebuilds every market cap, beta and volatility."""
    import csv
    import json
    import gzip
    import importlib.util
    import os

    pytest.importorskip("pandas")
    pytest.importorskip("scipy")
    from tradetracker import cli

    monkeypatch.setenv("DATABASE_URL", os.environ["TT_TEST_DATABASE_URL"])
    out = tmp_path / "run"
    shares = REAL / "shares_outstanding.csv"
    assert cli.main(["score-backtest", "--prices", str(REAL / "backtest_bars"), "--sic",
                     str(REAL / "backtest_sic.csv"), "--out", str(out), "--no-ablation",
                     "--shares", str(shares), "--min-cap", "50", "--bench", "beta",
                     "--vol-target", "0.4", "--name", "cap floor, vol control"]) == 0
    rows = list(csv.DictReader(gzip.open(out / "pool.csv.gz", "rt")))
    tickers = {r["ticker"] for r in rows}
    assert "RMCF" not in tickers and {"PLCE", "CTBI"} <= tickers
    assert all(float(r["market_cap"]) >= 50e6 for r in rows)
    summary = json.loads((out / "summary.json").read_text())
    assert summary["config"] == {"name": "cap floor, vol control", "h": 20, "bench": "beta",
                                 "min_cap": 50e6, "vol_target": 0.4}
    assert "cap >= $50M" in (out / "report.md").read_text()
    path = Path(__file__).parents[1] / "scripts" / "crosscheck_scoring.py"
    spec = importlib.util.spec_from_file_location("crosscheck_scoring", path)
    xc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(xc)
    assert xc.run_checks(out, REAL / "backtest_bars", os.environ["TT_TEST_DATABASE_URL"],
                         shares=shares) == []
