"""Command line entry point: `tt <command>`."""

import argparse
import json
import logging
import os
import sys
from datetime import date
from pathlib import Path

from . import config, db, report, store, validate
from .edgar import bulk, feed, tickers
from .features import job as features_job
from .features import summary as features_summary
from .edgar.client import EdgarClient
from .edgar.form4 import parse_form4
from .prices import cache as price_cache
from .prices import loader
from .prices.tiingo import TiingoClient

log = logging.getLogger("tt")
DATA_DIR = Path("data")


def cmd_migrate(args, settings):
    with db.connect(settings.database_url) as conn:
        applied = db.migrate(conn)
    print("applied: " + (", ".join(applied) or "nothing new"))


def _quarters(spec: str) -> list[tuple[int, int]]:
    """'2015q1:2016q4' or '2024q3' -> [(2015, 1), ...]."""
    first, _, last = spec.lower().partition(":")
    last = last or first
    y, q = int(first[:4]), int(first[-1])
    ly, lq = int(last[:4]), int(last[-1])
    out = []
    while (y, q) <= (ly, lq):
        out.append((y, q))
        y, q = (y + 1, 1) if q == 4 else (y, q + 1)
    return out


def cmd_load_bulk(args, settings):
    paths: list[Path] = [Path(p) for p in args.file or []]
    if args.quarters:
        client = EdgarClient(settings.require_edgar())
        DATA_DIR.mkdir(exist_ok=True)
        links: dict[tuple[int, int], str] | None = None
        for y, q in _quarters(args.quarters):
            dest = DATA_DIR / f"{y}q{q}_form345.zip"
            if not dest.exists():
                if links is None:
                    links = bulk.index_links(client.get(bulk.INDEX_URL).text)
                url = links.get((y, q)) or bulk.quarter_url(y, q)
                log.info("downloading %s", url)
                dest.write_bytes(client.get(url).content)
            paths.append(dest)
        client.close()

    with db.connect(settings.database_url) as conn:
        for path in paths:
            created = seen = 0
            for filing in bulk.parse_quarter(path):
                seen += 1
                _, new = store.save_filing(conn, filing)
                created += new
            conn.commit()
            print(f"{path.name}: {seen} Form 4 filings, {created} new")
        linked = store.link_amendments(conn)
        conn.commit()
        print(f"linked {linked} amendments to their original filing")


def cmd_poll_edgar(args, settings):
    client = EdgarClient(settings.require_edgar())
    entries = feed.fetch_latest(client, pages=args.pages)
    with db.connect(settings.database_url) as conn:
        known = {
            r[0]
            for r in conn.execute(
                "SELECT source_filing_id FROM filings WHERE source = 'edgar' "
                "AND source_filing_id = ANY(%s)",
                ([e.accession for e in entries],),
            )
        }
        new = 0
        for e in entries:
            if e.accession in known:
                continue
            try:
                url = feed.ownership_xml_url(client, e)
                filing = parse_form4(
                    client.get(url).content,
                    accession=e.accession,
                    accepted_at=e.accepted_at,
                    filed_at=e.accepted_at,
                    source_url=e.index_url,
                )
            except Exception:
                log.exception("failed to ingest %s", e.accession)
                continue
            store.save_filing(conn, filing)
            conn.commit()
            new += 1
        store.link_amendments(conn)
        conn.commit()
    client.close()
    print(f"feed: {len(entries)} Form 4 entries, {new} new")


def cmd_map_tickers(args, settings):
    client = EdgarClient(settings.require_edgar())
    current = tickers.fetch_company_tickers(client)
    client.close()
    with db.connect(settings.database_url) as conn:
        filled = store.fill_missing_tickers(conn, current)
        for cik, ticker in current.items():
            store.record_ticker(conn, cik, ticker, date.today(), "company_tickers")
        conn.commit()
    print(f"{len(current)} CIKs in SEC mapping; filled {filled} unmapped trades")


def cmd_load_prices(args, settings):
    client = TiingoClient(settings.require_tiingo())
    with db.connect(settings.database_url) as conn:
        stats = loader.run(conn, client, limit=args.limit)
    client.close()
    print(stats)


# The quarters the phase 2 report used, most recent first, then the other loaded ones.
# Within each |-separated group, tickers with buys come before sell-only tickers.
PRICE_ORDER = "2024q1,2023q4,2023q3,2023q2,2023q1|2026q2,2022q4,2020q1,2018q3,2015q1"


def _price_cache(args) -> price_cache.PriceCache:
    return price_cache.PriceCache(args.cache or os.environ.get("TT_PRICE_CACHE", "data/price-cache"))


def cmd_price_universe(args, settings):
    cache = _price_cache(args)
    with db.connect(settings.database_url) as conn:
        rows = price_cache.build_universe(conn)
    cache.write_universe(rows)
    print(f"{len(rows)} quarter-ticker rows, {len({r['ticker'] for r in rows})} tickers "
          f"-> {cache.universe_path}")


def cmd_fetch_prices(args, settings):
    cache = _price_cache(args)
    client = TiingoClient(settings.require_tiingo())
    try:
        stats = price_cache.run(cache, client, price_cache.parse_order(args.order), limit=args.limit)
    finally:
        client.close()
    print(stats)
    print((cache.root / "coverage.txt").read_text() if (cache.root / "coverage.txt").exists() else "")


def cmd_import_prices(args, settings):
    cache = _price_cache(args)
    with db.connect(settings.database_url) as conn:
        print(price_cache.import_to_db(cache, conn))


def cmd_price_coverage(args, settings):
    cache = _price_cache(args)
    print(price_cache.coverage_report(cache, price_cache.parse_order(args.order), price_cache.utcnow()))


def cmd_validate(args, settings):
    with db.connect(settings.database_url) as conn:
        print(validate.run(conn))


def cmd_features(args, settings):
    as_of = date.fromisoformat(args.as_of) if args.as_of else None
    with db.connect(settings.database_url) as conn:
        print(features_job.run(conn, recompute=args.all, as_of=as_of))


def cmd_feature_report(args, settings):
    with db.connect(settings.database_url) as conn:
        print(features_summary.render(conn))


def _backtest_tickers(conn) -> list[str]:
    from .backtest.yahoo import EXTRA_SYMBOLS

    rows = conn.execute("SELECT DISTINCT ticker FROM trades WHERE ticker IS NOT NULL ORDER BY 1")
    return loader.BENCHMARKS + EXTRA_SYMBOLS + [r[0] for r in rows]


def cmd_fetch_yahoo(args, settings):
    from .backtest import yahoo

    with db.connect(settings.database_url) as conn:
        tickers = _backtest_tickers(conn)
    print(yahoo.fetch_all(tickers, Path(args.cache), workers=args.workers, refresh=args.refresh))


def cmd_fetch_sic(args, settings):
    from .backtest import sectors

    with db.connect(settings.database_url) as conn:
        ciks = [r[0] for r in conn.execute(
            "SELECT DISTINCT issuer_cik FROM filings WHERE issuer_cik IS NOT NULL")]
    print(sectors.fetch_sic(ciks, Path(args.out), settings.require_edgar()))


def cmd_fetch_shares(args, settings):
    from . import fundamentals

    client = EdgarClient(settings.require_edgar())
    try:
        start = date.fromisoformat(args.start)
        print(fundamentals.fetch(client, Path(args.out), start, date.today()))
    finally:
        client.close()


def cmd_backtest(args, settings):
    from .backtest import events, prices, sectors, study

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with db.connect(settings.database_url) as conn:
        source = prices.open_source(args.prices, conn)
        evs, exclusions = events.load_events(conn)
        sic = sectors.load_sic(Path(args.sic)) if args.sic else {}
        results = study.run_study(evs, source, sic)
        extra = []
        if args.compare:
            other = prices.open_source(args.compare, conn)
            extra = study.source_comparison(results, study.run_study(evs, other, sic),
                                            source.name, other.name)
    study.write_events(results, out / "events.csv.gz")
    (out / "report.md").write_text(study.render(results, exclusions, source.name, extra))
    ok = sum(r.ok for r in results)
    print(f"{len(results)} events, {ok} with returns; wrote {out / 'report.md'}")


def _scoring_inputs(args, conn):
    from .backtest import events, prices, sectors
    from .backtest import study as bt_study
    from .scoring import signals

    source = prices.open_source(args.prices, conn)
    evs, _ = events.load_events(conn)
    evs = [e for e in evs if e.side == "buy"]
    sic = sectors.load_sic(Path(args.sic)) if args.sic else {}
    results = bt_study.run_study(evs, source, sic)
    shares = None
    if getattr(args, "shares", None):
        from .fundamentals import SharesTable
        shares = SharesTable.load(Path(args.shares))
    return source, results, signals.build(results, source, shares=shares), shares


def cmd_score_backtest(args, settings):
    from .scoring import study

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with db.connect(settings.database_url) as conn:
        source, results, sigs, _ = _scoring_inputs(args, conn)
    study.add_beta_legs(results, sigs)
    cfg = study.Config(args.name, h=args.horizon, bench=args.bench,
                       min_cap=args.min_cap * 1e6 if args.min_cap else None,
                       vol_target=args.vol_target)
    variants = None
    if args.variants:
        variants, sums = study.variants_table(results, sigs, source)
        (out / "variants.json").write_text(json.dumps(sums, indent=2))
    write_score_run(out, source, results, sigs, cfg, not args.no_ablation, variants)
    print(f"{len(results)} buy events; wrote {out / 'report.md'} and {out / 'model.json'}")


def write_score_run(out: Path, source, results, sigs, cfg, ablation: bool = True,
                    variants: str | None = None) -> None:
    """Fit, test and write one setup's run: report, models, summary and CSVs."""
    from .backtest import study as bt_study
    from .scoring import study

    out.mkdir(parents=True, exist_ok=True)
    folds = study.walk_forward(results, sigs, cfg=cfg)
    top = [x for f in folds for x in f.scored if x.above]
    _, half = study.decay_table(top, source, cfg.h, cfg.bench == "beta")
    final = study.fit_final(results, sigs, half_life=half, cfg=cfg)
    text, folds, pf = study.render(results, sigs, source, final, source.name,
                                  run_ablation=ablation, cfg=cfg, variants=variants)
    (out / "report.md").write_text(text)
    (out / "model.json").write_text(final.to_json())
    (out / "models").mkdir(exist_ok=True)
    for f in folds:
        (out / "models" / f"{f.year}.json").write_text(f.model.to_json())
    (out / "summary.json").write_text(json.dumps(study.summary(folds, pf, cfg), indent=2))
    study.write_scored(folds, out / "scored.csv.gz", cfg)
    study.write_pool(results, sigs, out / "pool.csv.gz", cfg)
    bt_study.write_events(results, out / "events.csv.gz")


def cmd_rank(args, settings):
    from .scoring import model, rank

    m = model.Model.from_json(Path(args.model).read_text())
    as_of = date.fromisoformat(args.as_of) if args.as_of else date.today()
    with db.connect(settings.database_url) as conn:
        source, results, sigs, shares = _scoring_inputs(args, conn)
    rows, hidden = rank.rank(results, sigs, source, m, as_of, top=args.top, shares=shares,
                             min_cap=args.min_cap * 1e6 if shares and args.min_cap else None)
    print(f"Top {len(rows)} buys as of {as_of} (hidden: "
          + ", ".join(f"{k} {v}" for k, v in hidden.items()) + ")")
    for i, r in enumerate(rows, 1):
        e = r.result.event
        print(f"{i:2d}. {e.ticker:6s} score {r.score:.3f} (Q {r.quality:.2f}), filed "
              f"{e.filing_date}: {r.reason}; round trip {rank.cost_note(r)}")


def cmd_report(args, settings):
    with db.connect(settings.database_url) as conn:
        print(report.render(conn))


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(prog="tt")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("migrate", help="apply SQL migrations").set_defaults(fn=cmd_migrate)

    b = sub.add_parser("load-bulk", help="load SEC insider data sets")
    b.add_argument("--quarters", help="e.g. 2015q1:2026q2")
    b.add_argument("--file", action="append", help="a local *_form345.zip")
    b.set_defaults(fn=cmd_load_bulk)

    f = sub.add_parser("poll-edgar", help="ingest new Form 4s from the live feed")
    f.add_argument("--pages", type=int, default=1, help="100 entries per page")
    f.set_defaults(fn=cmd_poll_edgar)

    sub.add_parser("map-tickers", help="fill tickers from SEC company_tickers.json").set_defaults(
        fn=cmd_map_tickers
    )

    lp = sub.add_parser("load-prices", help="backfill/update prices within Tiingo quota")
    lp.add_argument("--limit", type=int, help="max requests this run")
    lp.set_defaults(fn=cmd_load_prices)

    for name, fn, text in [
        ("price-universe", cmd_price_universe, "write the tickers to price into the cache"),
        ("fetch-prices", cmd_fetch_prices, "fill the file price cache within Tiingo quota (hourly)"),
        ("import-prices", cmd_import_prices, "copy the file price cache into daily_prices"),
        ("price-coverage", cmd_price_coverage, "price cache coverage per quarter"),
    ]:
        pc = sub.add_parser(name, help=text)
        pc.add_argument("--cache", help="cache directory (default $TT_PRICE_CACHE or data/price-cache)")
        pc.add_argument("--order", default=PRICE_ORDER, help="quarter priority: comma separated, groups split by |")
        pc.add_argument("--limit", type=int, help="max requests this run")
        pc.set_defaults(fn=fn)

    sub.add_parser("validate", help="apply validation rules").set_defaults(fn=cmd_validate)
    sub.add_parser("report", help="milestone 1 coverage report").set_defaults(fn=cmd_report)

    fe = sub.add_parser("features", help="compute trade_features for trades that lack them")
    fe.add_argument("--all", action="store_true", help="recompute every trade")
    fe.add_argument("--as-of", help="date for days_since_filing (default today, Eastern)")
    fe.set_defaults(fn=cmd_features)

    sub.add_parser("feature-report", help="feature distributions and coverage").set_defaults(
        fn=cmd_feature_report
    )

    fy = sub.add_parser("fetch-yahoo", help="research prices from Yahoo into a cache dir")
    fy.add_argument("--cache", required=True, help="directory for bars/<T>.csv.gz")
    fy.add_argument("--workers", type=int, default=4)
    fy.add_argument("--refresh", action="store_true", help="refetch tickers already done")
    fy.set_defaults(fn=cmd_fetch_yahoo)

    fs = sub.add_parser("fetch-sic", help="issuer SIC codes from EDGAR, for sector benchmarks")
    fs.add_argument("--out", required=True, help="CSV cache, appended to")
    fs.set_defaults(fn=cmd_fetch_sic)

    fsh = sub.add_parser("fetch-shares", help="shares outstanding per issuer (SEC XBRL frames)")
    fsh.add_argument("--out", required=True, help="CSV cache, appended to")
    fsh.add_argument("--start", default="2014-01-01")
    fsh.set_defaults(fn=cmd_fetch_shares)

    bt = sub.add_parser("backtest", help="phase 3 study: returns by lag and drift")
    bt.add_argument("--prices", required=True,
                    help="'db', 'db:<source>', or a cache directory (Tiingo layout)")
    bt.add_argument("--sic", help="SIC CSV from fetch-sic (sector benchmarks)")
    bt.add_argument("--compare", help="a second price source to check the first against")
    bt.add_argument("--out", required=True, help="directory for report.md and events.csv.gz")
    bt.set_defaults(fn=cmd_backtest)

    sb = sub.add_parser("score-backtest", help="phase 4: fit the score and test it walk-forward")
    sb.add_argument("--prices", required=True, help="as for backtest")
    sb.add_argument("--sic", help="SIC CSV from fetch-sic (sector benchmarks)")
    sb.add_argument("--out", required=True, help="directory for report.md, model.json, scored.csv.gz")
    sb.add_argument("--no-ablation", action="store_true", help="skip the drop-one-component runs")
    sb.add_argument("--shares", help="share counts CSV from fetch-shares (market cap)")
    sb.add_argument("--name", default="phase 4", help="name of the setup, for the report")
    sb.add_argument("--horizon", type=int, default=20, choices=(5, 20, 60), help="hold, sessions")
    sb.add_argument("--bench", default="spy", choices=("spy", "sector", "iwm", "beta"),
                    help="benchmark; beta = stock minus beta x SPY")
    sb.add_argument("--min-cap", type=float, help="hide market caps below this, $M (needs --shares)")
    sb.add_argument("--vol-target", type=float, help="size positions to this annual volatility")
    sb.add_argument("--variants", action="store_true", help="also compare the five setups")
    sb.set_defaults(fn=cmd_score_backtest)

    rk = sub.add_parser("rank", help="top buys filed recently, scored with a fitted model")
    rk.add_argument("--model", required=True, help="model.json from score-backtest")
    rk.add_argument("--prices", required=True, help="as for backtest")
    rk.add_argument("--sic", help="SIC CSV from fetch-sic")
    rk.add_argument("--as-of", help="date to rank as of (default today)")
    rk.add_argument("--top", type=int, default=20)
    rk.add_argument("--shares", help="share counts CSV from fetch-shares (market-cap floor)")
    rk.add_argument("--min-cap", type=float, default=50, help="$M floor when --shares is given")
    rk.set_defaults(fn=cmd_rank)

    args = p.parse_args(argv)
    try:
        args.fn(args, config.load())
    except config.ConfigError as e:
        print(f"config error: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
