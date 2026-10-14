"""report.md for a paper book: what it holds, what it has closed, what the last run did."""

from __future__ import annotations

import statistics
from datetime import date

from .engine import live_positions, mark, spy_return

BACKTEST = ("Backtest, 2017-2026 out of sample (two inputs, 60 days, $50M cap floor): "
            "+5.06% mean 60-day excess over SPY after costs, 51.3% beat SPY, "
            "top-20 portfolio information ratio 0.09.")


def _pct(v) -> str:
    return "" if v is None else f"{100 * v:+.2f}%"


def _money(v) -> str:
    return "" if v is None else f"${v:,.2f}"


def render(state: dict, prices, as_of: date, equity: float | None = None) -> str:
    kind = state["broker"]
    lines = [f"# Paper book ({'Alpaca paper account' if kind == 'alpaca' else 'dry run: orders logged, fills simulated'})",
             "",
             f"As of the close before {as_of}. Started {state['created'][:10]}. "
             f"Buys open-market insider purchases scoring at or above the model's cutoff, "
             f"20 slots of 1/20 of equity each, bought at the open after the filing and "
             f"sold at the close 60 sessions later.",
             "", BACKTEST, ""]
    closed = [p for p in state["positions"] if p["status"] == "closed"]
    live = live_positions(state)
    ex = [p["excess_net"] for p in closed if p.get("excess_net") is not None]
    lines += ["## Summary", "", "| | |", "| --- | --- |"]
    if equity is not None:
        lines.append(f"| Equity | {_money(equity)} (started at {_money(state['start_cash'])}) |")
    lines.append(f"| Slots in use | {len(live)} of 20 |")
    lines.append(f"| Closed positions | {len(closed)} |")
    if ex:
        hit = sum(1 for v in ex if v > 0) / len(ex)
        lines.append(f"| Mean 60-day excess over SPY, closed{' (net of estimated costs)' if kind == 'dry-run' else ''} "
                     f"| {_pct(statistics.mean(ex))} |")
        lines.append(f"| Beat SPY | {100 * hit:.0f}% of {len(ex)} |")
    else:
        lines.append("| Mean excess, closed | none closed yet (the first exits come 60 sessions after the first buys) |")
    lines.append("")

    if live:
        lines += ["## Holding", "",
                  "| ticker | insider | filed | bought | shares | price | sell on | return | SPY | excess | below 52w high at signal |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for p in sorted(live, key=lambda p: (p.get("entry_session") or p["entry_order"]["session"], p["ticker"])):
            if p["status"] == "pending_entry":
                lines.append(f"| {p['ticker']} | {p['filer_name']} ({p['role'] or 'insider'}) | {p['filing_date']} | "
                             f"order for {p['entry_order']['session']} ({p['entry_order']['status']}) | "
                             f"{p['entry_order']['qty']} | | | | | | {_pct(p['drawdown'])} |")
                continue
            r = mark(prices, p, as_of)
            s = spy_return(prices, date.fromisoformat(p["entry_session"]), as_of)
            lines.append(f"| {p['ticker']} | {p['filer_name']} ({p['role'] or 'insider'}) | {p['filing_date']} | "
                         f"{p['entry_session']} | {p['entry_qty']:g} | {_money(p['entry_price'])} | "
                         f"{p['exit_due']}{' (order in)' if p['status'] == 'pending_exit' else ''} | "
                         f"{_pct(r)} | {_pct(s)} | {_pct(None if r is None or s is None else r - s)} | "
                         f"{_pct(p['drawdown'])} |")
        lines.append("")

    if closed:
        lines += ["## Closed", "",
                  "| ticker | insider | bought | sold | return | SPY | excess | excess, net |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- |"]
        for p in sorted(closed, key=lambda p: (p["exit_session"], p["ticker"]), reverse=True):
            lines.append(f"| {p['ticker']} | {p['filer_name']} | {p['entry_session']} | {p['exit_session']} | "
                         f"{_pct(p.get('ret'))} | {_pct(p.get('spy_ret'))} | {_pct(p.get('excess'))} | "
                         f"{_pct(p.get('excess_net'))} |")
        lines.append("")

    others = [p for p in state["positions"] if p["status"] == "not_filled"]
    if others:
        lines += ["## Orders that didn't fill", ""]
        lines += [f"- {p['ticker']} ({p['entry_order']['session']}): {p.get('close_reason', '')}" for p in others[-20:]]
        lines.append("")

    runs = state["runs"][-10:]
    if runs:
        lines += ["## Recent runs", "", "| run | session | filings read | candidates | orders | notes |",
                  "| --- | --- | --- | --- | --- | --- |"]
        for r in reversed(runs):
            lines.append(f"| {r['at'][:16].replace('T', ' ')} | {r['session'] or 'closed'} | "
                         f"{'; '.join(r['ingested']) or '-'} | {', '.join(r['candidates']) or '-'} | "
                         f"{'; '.join(r['orders']) or '-'} | {'; '.join(r['notes']) or ''} |")
        lines.append("")
    return "\n".join(lines)
