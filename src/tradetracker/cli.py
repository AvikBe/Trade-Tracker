"""Command line entry point: `tt <command>`."""

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

from . import config, db, report, store, validate
from .edgar import bulk, feed, tickers
from .features import job as features_job
from .features import summary as features_summary
from .edgar.client import EdgarClient
from .edgar.form4 import parse_form4
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

    bt = sub.add_parser("backtest", help="phase 3 study: returns by lag and drift")
    bt.add_argument("--prices", required=True,
                    help="'db', 'db:<source>', or a cache directory (Tiingo layout)")
    bt.add_argument("--sic", help="SIC CSV from fetch-sic (sector benchmarks)")
    bt.add_argument("--compare", help="a second price source to check the first against")
    bt.add_argument("--out", required=True, help="directory for report.md and events.csv.gz")
    bt.set_defaults(fn=cmd_backtest)

    args = p.parse_args(argv)
    try:
        args.fn(args, config.load())
    except config.ConfigError as e:
        print(f"config error: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
