"""Charts for scripts/correlation_study.py (matplotlib, static PNG)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

SURFACE, INK, INK2, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
BUY, SELL = "#2a78d6", "#eb6834"
DIVERGING = LinearSegmentedColormap.from_list("div", ["#e34948", "#f0efec", "#2a78d6"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "text.color": INK, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "legend.frameon": False,
})


def pct(ax, axis="y"):
    lim = max(abs(x) for x in (ax.get_ylim() if axis == "y" else ax.get_xlim()))
    digits = 0 if lim >= 0.05 else (1 if lim >= 0.005 else 2)
    f = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v * 100:.{digits}f}%")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(f)


def event_study(paths, out: Path):
    fig, ax = plt.subplots(figsize=(9, 5))
    x = paths["session"]
    for side, color, label in (("buy", BUY, "Insider buys"), ("sell", SELL, "Insider sells")):
        ax.plot(x, paths[f"{side}_mean"], color=color, lw=2, label=label)
        ax.text(x.iloc[-1] + 1, paths[f"{side}_mean"].iloc[-1], label, color=INK,
                va="center", fontsize=9)
    ax.axvline(0, color=AXIS, lw=1)
    ax.axhline(0, color=AXIS, lw=1)
    ax.text(1, ax.get_ylim()[1], " filing public: first chance to trade", color=INK2,
            va="top", fontsize=9)
    ax.set_xlabel("Trading days from the filing (0 = last close before we can trade)")
    ax.set_ylabel("Return vs S&P 500, measured from day 0")
    pct(ax)
    ax.set_xlim(-60, 72)
    ax.set_title("Insiders buy after a ~13% slide vs the market; afterwards the stock gains "
                 "about 1%,\nmostly on the first day, then tracks the market")
    fig.text(0.01, 0.01, "Read left to right: a buy line at +13% on day -60 means the stock "
             "was 13% higher (vs S&P 500) 60 days before the filing.\nClose-to-close "
             "returns, winsorized at 1%/99% per day.", color=INK2, fontsize=8)
    ax.legend(loc="upper right")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out / "1-event-study.png", dpi=160)
    plt.close(fig)


def ic_heatmap(ic, out: Path):
    d = ic[ic["return"] == "adj"]
    feats = list(dict.fromkeys(d["feature"]))
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.4), sharey=True)
    for ax, side, title in zip(axes, ("buy", "sell"), ("Buys", "Sells")):
        s = d[d["side"] == side]
        M = np.full((len(feats), 3), np.nan)
        T = np.full_like(M, np.nan)
        for i, f in enumerate(feats):
            for j, h in enumerate((5, 20, 60)):
                r = s[(s["feature"] == f) & (s["h"] == h)]
                if len(r):
                    M[i, j], T[i, j] = r["monthly_ic"].iloc[0], r["t"].iloc[0]
        ax.imshow(M, cmap=DIVERGING, vmin=-0.08, vmax=0.08, aspect="auto")
        for i in range(len(feats)):
            for j in range(3):
                if np.isnan(M[i, j]):
                    ax.text(j, i, "n/a", ha="center", va="center", color=INK2, fontsize=8)
                    continue
                star = "*" if abs(T[i, j]) >= 2 else ""
                ax.text(j, i, f"{M[i, j]:+.3f}{star}", ha="center", va="center",
                        color=INK, fontsize=9)
        ax.set_xticks(range(3), ["5 days", "20 days", "60 days"])
        ax.set_yticks(range(len(feats)), feats)
        ax.grid(False)
        ax.set_title(title)
    fig.suptitle("Rank correlation with the later return minus S&P 500 "
                 "(monthly average; * = |t| ≥ 2)", x=0.01, ha="left", fontsize=11,
                 fontweight="bold")
    fig.text(0.01, 0.01, "Blue: higher value → higher later return. Red: higher value → "
             "lower later return. ±0.05 is a meaningful stock-signal size.",
             color=INK2, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(out / "2-correlation-heatmap.png", dpi=160)
    plt.close(fig)


def bucket_grid(b, out: Path, h: int):
    d = b[(b["return"] == "adj") & (b["h"] == h)]
    feats = list(dict.fromkeys(d["feature"]))
    n = len(feats)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3.0 * rows))
    axes = axes.ravel()
    for ax, f in zip(axes, feats):
        for k, (side, color) in enumerate((("buy", BUY), ("sell", SELL))):
            s = d[(d["feature"] == f) & (d["side"] == side)]
            if s.empty:
                continue
            x = np.arange(len(s)) + (k - 0.5) * 0.36
            ax.bar(x, s["wmean"], width=0.34, color=color,
                   label="buys" if side == "buy" else "sells")
            ax.errorbar(x, s["wmean"], yerr=[s["wmean"] - s["lo"], s["hi"] - s["wmean"]],
                        fmt="none", ecolor=INK2, elinewidth=1, capsize=2)
            labels = list(s["bucket"])
        ax.set_xticks(np.arange(len(labels)), labels, fontsize=8,
                      rotation=20 if max(map(len, labels)) > 6 else 0)
        ax.axhline(0, color=AXIS, lw=1)
        pct(ax)
        ax.set_title(f, fontsize=10)
    for ax in axes[n:]:
        ax.axis("off")
    axes[0].legend(loc="best", fontsize=8)
    fig.suptitle(f"Average {h}-day return minus S&P 500 by bucket "
                 "(winsorized, bars = 95% interval)", x=0.01, ha="left",
                 fontsize=11, fontweight="bold")
    fig.text(0.01, 0.005, ("Q1 = lowest fifth, Q5 = highest fifth (for % below 52-week high, Q5 = "
                           "furthest below). Insiders trading together = distinct insiders "
                           "on the same side within 14 days."), color=INK2,
             fontsize=8.5)
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    fig.savefig(out / f"3-buckets-{h}-days.png", dpi=160)
    plt.close(fig)


def yearly_lines(y, ic, out: Path):
    d = y[(y["side"] == "buy") & (y["h"] == 60)]
    a = ic[(ic["side"] == "buy") & (ic["h"] == 60) & (ic["return"] == "adj")]
    top = list(a.reindex(a["t"].abs().sort_values(ascending=False).index)["feature"][:6])
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharey=True)
    for ax, f in zip(axes.ravel(), top):
        s = d[d["feature"] == f].sort_values("year")
        ax.bar(s["year"], s["ic"], color=[BUY if v >= 0 else "#e34948" for v in s["ic"]],
               width=0.7)
        ax.axhline(0, color=AXIS, lw=1)
        pos = int((s["ic"] > 0).sum())
        ax.set_title(f"{f}\npositive in {pos} of {len(s)} years", fontsize=10)
        ax.tick_params(axis="x", labelsize=8)
    fig.suptitle("Buys: rank correlation with 60-day return minus S&P 500, year by year",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out / "4-stability-by-year.png", dpi=160)
    plt.close(fig)


def joint_bars(j, out: Path, names: dict):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    for ax, h in zip(axes, (20, 60)):
        for k, (side, color) in enumerate((("buy", BUY), ("sell", SELL))):
            s = j[(j["side"] == side) & (j["h"] == h)].set_index("feature")
            s = s.reindex(list(names))
            y = np.arange(len(s)) + (k - 0.5) * 0.38
            ax.barh(y, s["slope"], height=0.36, color=color,
                    label="buys" if side == "buy" else "sells")
            for yy, v, t in zip(y, s["slope"], s["t"]):
                if np.isfinite(v) and abs(t) >= 2:
                    ax.text(v, yy, " *" if v >= 0 else "* ", va="center",
                            ha="left" if v >= 0 else "right", color=INK, fontsize=9)
        ax.set_yticks(np.arange(len(names)), list(names.values()))
        ax.axvline(0, color=AXIS, lw=1)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(steps=[1, 2, 5, 10]))
        pct(ax, "x")
        ax.invert_yaxis()
        ax.set_title(f"{h} days")
    h_, l_ = axes[0].get_legend_handles_labels()
    fig.legend(h_, l_, loc="upper right", ncol=2, fontsize=9)
    fig.suptitle("All characteristics at once: return gap, lowest to highest rank, others "
                 "held equal (* = |t| ≥ 2)", x=0.01, ha="left", fontsize=11,
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out / "5-joint-regression.png", dpi=160)
    plt.close(fig)


def draw_all(res, out: Path):
    from tradetracker import correlation as C
    out = Path(out)
    event_study(res["event_paths"], out)
    ic_heatmap(res["ic"], out)
    for h in (20, 60):
        bucket_grid(res["buckets"], out, h)
    yearly_lines(res["yearly_ic"], res["ic"], out)
    names = {c: n for n, c, k in C.FEATURES if k != "cat" and c != "value_to_cap"}
    names.update({"role_CEO": "CEO (vs other officer)", "role_CFO": "CFO (vs officer)",
                  "role_director": "director (vs officer)",
                  "role_10pct_owner": "10% owner (vs officer)"})
    joint_bars(res["joint"], out, names)
