"""Command line entry point: `tt <command>`."""

import argparse
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


# Most recent first for the quarters the phase 2 report used, then the other loaded ones.
PRICE_ORDER = (
    "2024q1,2023q4,2023q3,2023q2,2023q1,2026q2,2022q4,2020q1,2018q3,2015q1"
)


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
        stats = price_cache.run(cache, client, args.order.split(","), limit=args.limit)
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
    print(price_cache.coverage_report(cache, args.order.split(","), price_cache.utcnow()))


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
        pc.add_argument("--order", default=PRICE_ORDER, help="quarter priority, comma separated")
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

    args = p.parse_args(argv)
    try:
        args.fn(args, config.load())
    except config.ConfigError as e:
        print(f"config error: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
