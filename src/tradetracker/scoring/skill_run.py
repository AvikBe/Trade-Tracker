"""`tt skill-study`: build the insider records, run the skill study, write the outputs."""

from __future__ import annotations

import csv
import gzip
import io
import json
from dataclasses import replace
from pathlib import Path

from .. import skill as SK
from . import skill_study as SS
from . import study as S

POOL_COLUMNS = ["filer_id", "owner", "issuer_cik", "ticker", "filing_date", "entry", "year",
                "skill_20", "skill_n20", "skill_raw20", "skill_hit20",
                "skill_60", "skill_n60", "skill_raw60", "skill_hit60",
                "drawdown", "drift", "market_cap", "net_20", "net_60", "exit_20", "exit_60"]


def issuer_tickers(results) -> dict[str, str]:
    """issuer CIK -> the ticker of its earliest backtest event (closest to 2014)."""
    out: dict[str, tuple] = {}
    for x in results:
        e = x.event
        k = SK.owner_key(e.issuer_cik)
        if k and e.ticker and (k not in out or e.filing_date < out[k][0]):
            out[k] = (e.filing_date, e.ticker)
    return {k: v[1] for k, v in out.items()}


def attach(signals, results, books, owner_keys):
    """Copies of `signals` with the skill fields filled from `books`."""
    out = []
    for x, s in zip(results, signals):
        owner = (SK.owner_key(owner_keys.get(x.event.filer_id)) if owner_keys
                 else f"filer:{x.event.filer_id}")
        fields = {}
        for h, b in books.items():
            r = b.record(owner, x.event.filing_date)
            fields.update({f"skill_{h}": r.shrunk, f"skill_n{h}": r.n,
                           f"skill_raw{h}": r.raw, f"skill_hit{h}": r.hit})
        out.append(replace(s, **fields))
    return out


def _v(v) -> str:
    return "" if v is None else repr(v)


def write_pool(results, signals, owner_keys, path: Path, cfg: S.Config) -> int:
    """Every shown buy (cap floor) with its skill fields, for the cross-check."""
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(POOL_COLUMNS)
    n = 0
    for x, s in zip(results, signals):
        if not S.shown(x, s, cfg):
            continue
        e, o = x.event, x.outcome
        w.writerow([e.filer_id, (owner_keys or {}).get(e.filer_id, ""), e.issuer_cik or "",
                    e.ticker, e.filing_date, o.entry, o.entry.year,
                    _v(s.skill_20), s.skill_n20, _v(s.skill_raw20), _v(s.skill_hit20),
                    _v(s.skill_60), s.skill_n60, _v(s.skill_raw60), _v(s.skill_hit60),
                    _v(s.drawdown), _v(s.drift), _v(s.market_cap),
                    _v(x.r(20, "spy", True)), _v(x.r(60, "spy", True)),
                    (o.legs.get(20) or {}).get("exit", ""), (o.legs.get(60) or {}).get("exit", "")])
        n += 1
    path.write_bytes(gzip.compress(buf.getvalue().encode()))
    return n


def write_episodes(books, path: Path) -> None:
    """Every episode of every horizon (owner, issuer, filed, known, clipped excess)."""
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["horizon", "owner", "issuer", "filed", "known", "excess"])
    for h, b in sorted(books.items()):
        for e in b.episodes:
            w.writerow([h, e.owner, e.issuer, e.filed, e.known, repr(e.excess)])
    path.write_bytes(gzip.compress(buf.getvalue().encode()))


def run(results, signals, prices, owner_keys, out: Path, history: Path | None = None,
        placebo_draws: int = SS.PLACEBO_DRAWS, configs=None) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    books = SK.build_books(results, prices, owner_keys, history=history,
                           tickers=issuer_tickers(results))
    sigs = attach(signals, results, books, owner_keys)
    S.add_beta_legs(results, sigs)
    configs = configs or S.SKILL_VARIANTS
    summary: dict = {}

    base60 = S.Config("cap floor, 60 days", h=60, min_cap=50e6)
    base20 = replace(base60, name="cap floor, 20 days", h=20)
    cov_text, summary["coverage"] = SS.coverage(sigs, results, books, base60)
    persist, spreads = {}, {}
    for cfg in (base20, base60):
        persist[cfg.h], summary[f"persistence_{cfg.h}"] = SS.persistence(results, sigs, cfg)
        spreads[cfg.h], summary[f"spread_{cfg.h}"] = SS.spread_by_year(results, sigs, cfg)

    vtext, summary["variants"] = S.variants_table(results, sigs, prices, configs, extra_slots=40)
    placebos = {}
    for cfg in configs:
        if cfg.skill_filter:
            text, summary.setdefault("placebo", {})[cfg.name] = SS.gate_placebo(
                results, sigs, prices, cfg, draws=placebo_draws)
            placebos[cfg.name] = text
    main = next((c for c in configs if c.skill_filter == "proven" and c.h == 60), configs[0])
    folds = S.walk_forward(results, sigs, cfg=main)
    ytable, _ = S.year_table(folds, main)
    S.write_scored(folds, out / "scored-proven-60-days.csv.gz", main)

    (out / "report.md").write_text(SS.render(results, sigs, prices, books, vtext, placebos,
                                             persist, spreads, cov_text, ytable))
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    summary["pool_rows"] = write_pool(results, sigs, owner_keys, out / "skill-pool.csv.gz", base60)
    write_episodes(books, out / "episodes.csv.gz")
    from ..backtest import study as bt_study
    bt_study.write_events(results, out / "events.csv.gz")
    keys = io.StringIO()
    w = csv.writer(keys, lineterminator="\n")
    w.writerow(["filer_id", "owner_cik"])
    w.writerows(sorted((owner_keys or {}).items()))
    (out / "owner-keys.csv.gz").write_bytes(gzip.compress(keys.getvalue().encode()))
    return summary
