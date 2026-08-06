#!/usr/bin/env python3
"""
dashboard.py — plain-English status for the trading bot.

Questions this page answers:
  1. Is evolution running right now?
  2. Are strategies trading enough (N>=30)?
  3. Is search improving, or just printing big fitness numbers?
  4. Did anything survive the promotion funnel (deploy gate)?

Important: "best fitness" is a SEARCH score, not proof of an edge.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time
import webbrowser
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from success_criteria import (
    BOOK_USD,
    LAB_MIN_TRADES_FULL,
    PAPER_MAX_DRAWDOWN_HARD,
    PAPER_MIN_DAYS,
    PAPER_MIN_NET_PNL_USD,
    PAPER_MIN_TRADES,
    criteria_public_dict,
    plain_english_summary,
    sanitize_search_score,
)

SIM = Path(__file__).resolve().parent
LOGS = SIM / "logs"
EVO = SIM / "evolution"
PORT = 8765

# Health threshold for trade sample (matches LAB selection min)
MIN_TRADES = LAB_MIN_TRADES_FULL


def sanitize_fitness(v: float) -> float:
    """Clamp legacy explodey fitness so charts/story stay readable."""
    return sanitize_search_score(v)


def load_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def process_alive() -> dict:
    """Check if broad evolution / any evo runner is alive."""
    try:
        out = subprocess.check_output(
            ["pgrep", "-fl", "run_broad_evolution|EvolutionEngine|evolution_runner"],
            text=True,
            timeout=3,
        )
    except Exception:
        out = ""
    lines = [ln for ln in out.splitlines() if ln.strip()]
    return {
        "running": any("run_broad_evolution" in ln or "EvolutionEngine" in ln for ln in lines),
        "detail": lines[0][:180] if lines else "",
        "n_procs": len(lines),
    }


def live_activity() -> dict | None:
    a = load_json(LOGS / "live_activity.json")
    if not a or not a.get("ts"):
        return None
    age = time.time() - float(a["ts"])
    a["age_s"] = round(age, 1)
    # During pop=150 each genome can take >30s on funnel cycle boundaries
    a["live"] = age < 120
    return a


def load_all_runs() -> list[dict]:
    pop_dir = EVO / "population"
    if not pop_dir.exists():
        return []
    files = sorted(pop_dir.glob("evolution_*.json"), key=lambda f: f.stat().st_mtime)
    runs = []
    for f in files:
        try:
            d = json.loads(f.read_text())
            d["_file"] = f.name
            d["_mtime"] = f.stat().st_mtime
            runs.append(d)
        except Exception:
            pass
    return runs


def run_trades(r: dict) -> int:
    bt = r.get("backtest") or {}
    return int(bt.get("total_trades") or 0)


def run_logic(r: dict) -> str:
    g = r.get("best_genome") or {}
    return str(g.get("entry_logic") or "?")


def is_funnel_era(r: dict) -> bool:
    mode = str(r.get("mode") or "")
    return "funnel" in mode or "n_promoted" in r or "total_trials" in r


def is_broad_era(r: dict) -> bool:
    mode = str(r.get("mode") or "")
    return "broad" in mode or is_funnel_era(r)


def summarize_window(runs: list[dict]) -> dict:
    if not runs:
        return {
            "n": 0,
            "fit_med": 0.0,
            "fit_avg": 0.0,
            "trade_med": 0.0,
            "pct_n30": 0.0,
            "pct_n50": 0.0,
            "pct_pos_pnl": 0.0,
            "pct_random": 0.0,
            "promoted": 0,
            "cycles_with_funnel": 0,
        }
    fits = [sanitize_fitness(r.get("best_fitness") or 0) for r in runs]
    trades = [run_trades(r) for r in runs]
    pnls = [float((r.get("backtest") or {}).get("total_pnl") or 0) for r in runs]
    logics = [run_logic(r) for r in runs]
    promoted = sum(int(r.get("n_promoted") or 0) for r in runs if is_funnel_era(r))
    funnel_cycles = sum(1 for r in runs if is_funnel_era(r))
    return {
        "n": len(runs),
        "fit_med": statistics.median(fits),
        "fit_avg": statistics.mean(fits),
        "trade_med": statistics.median(trades),
        "pct_n30": 100.0 * sum(t >= MIN_TRADES for t in trades) / len(trades),
        "pct_n50": 100.0 * sum(t >= 50 for t in trades) / len(trades),
        "pct_pos_pnl": 100.0 * sum(p > 0 for p in pnls) / len(pnls),
        "pct_random": 100.0 * sum(l == "RANDOM" for l in logics) / len(logics),
        "promoted": promoted,
        "cycles_with_funnel": funnel_cycles,
    }


def compute_honest_trend(recent: list[dict], prev: list[dict]) -> dict:
    """Trend on health metrics, not raw max fitness (max is lottery-polluted)."""
    a = summarize_window(prev)
    b = summarize_window(recent)
    if a["n"] < 5 or b["n"] < 5:
        return {
            "direction": "TOO EARLY",
            "why": "Need more cycles in the current search regime.",
            "metric": "pct_n30",
            "prev": a,
            "recent": b,
        }

    # Primary health: share of chairs with enough trades
    d_n30 = b["pct_n30"] - a["pct_n30"]
    # Secondary: median OOS-selection fitness of N>=30 only
    def med_fit_hard(runs):
        vals = [
            sanitize_fitness(r.get("best_fitness") or 0)
            for r in runs
            if run_trades(r) >= MIN_TRADES
        ]
        return statistics.median(vals) if vals else None

    prev_med = med_fit_hard(prev)
    rec_med = med_fit_hard(recent)

    if d_n30 > 5 or (prev_med is not None and rec_med is not None and rec_med > prev_med * 1.05):
        direction = "HEALTHIER"
    elif d_n30 < -5 or (prev_med is not None and rec_med is not None and rec_med < prev_med * 0.95):
        direction = "WORSE"
    else:
        direction = "FLAT"

    plain = plain_english_trend(direction, a, b, prev_med, rec_med)

    return {
        "direction": direction,
        "why": plain["summary"],
        "detail": plain["detail"],
        "metric": f"share with {MIN_TRADES}+ trades, and median score of those",
        "prev": a,
        "recent": b,
        "prev_med_hard": prev_med,
        "recent_med_hard": rec_med,
    }


def plain_english_trend(direction: str, prev: dict, recent: dict, prev_med, rec_med) -> dict:
    """Human commentary for the health trend box."""
    n30_was, n30_now = prev["pct_n30"], recent["pct_n30"]
    tr_was, tr_now = prev["trade_med"], recent["trade_med"]
    prom_now = recent.get("promoted", 0)

    if direction == "TOO EARLY":
        return {
            "summary": "Too early to judge. Need more finished cycles under the current rules.",
            "detail": "Wait until both the earlier block and the latest block have at least 5 cycles.",
        }

    fit_txt = ""
    if prev_med is not None and rec_med is not None:
        if rec_med > prev_med * 1.05:
            fit_txt = f"Typical search score of strategies with enough trades moved up ({prev_med:.0f} to {rec_med:.0f})."
        elif rec_med < prev_med * 0.95:
            fit_txt = f"Typical search score of strategies with enough trades moved down ({prev_med:.0f} to {rec_med:.0f})."
        else:
            fit_txt = f"Typical search score is about unchanged ({prev_med:.0f} to {rec_med:.0f})."

    trade_txt = f"Typical trade count went from {tr_was:.0f} to {tr_now:.0f}."
    n30_txt = f"Share of winners with at least {MIN_TRADES} trades: {n30_was:.0f}% to {n30_now:.0f}%."

    if direction == "HEALTHIER":
        head = "Looking healthier: the search is finding more usable sample sizes or better typical scores."
    elif direction == "WORSE":
        head = "Looking weaker: fewer well-sampled winners or lower typical scores."
    else:
        head = "Mostly flat: not clearly better or worse over the last blocks of cycles."

    promo_txt = (
        f" Recent funnel promotions in this window: {prom_now}."
        if recent.get("cycles_with_funnel")
        else " Funnel stats only count cycles that ran the gauntlet."
    )
    return {
        "summary": head,
        "detail": f"{n30_txt} {trade_txt} {fit_txt}{promo_txt}",
    }


# ---------------------------------------------------------------------------
# Learning curve: how far ideas get through the strict exam over time.
# The funnel gates are ordered; "depth" = how many gates the best candidate
# of a search cleared before dying (all gates passed = shortlist).
# ---------------------------------------------------------------------------

GATE_ORDER = [
    "kill_archive", "feasibility", "oos", "benchmark", "walk_forward",
    "fee_stress", "perturbation", "mev", "dsr",
]
GATE_FRIENDLY = {
    "kill_archive": "blocked at the door (known-bad family)",
    "feasibility": "basic check (enough trades, profit, drawdown)",
    "oos": "unseen later data",
    "benchmark": "beating buy-and-hold (beta filter)",
    "walk_forward": "chapter-by-chapter consistency",
    "fee_stress": "higher fees",
    "perturbation": "small rule nudges",
    "mev": "front-running stress",
    "dsr": "statistical luck filter",
}
N_GATES = len(GATE_ORDER)


def funnel_depth(r: dict):
    """Deepest exam gate cleared by any candidate of one finished search.

    Returns 0..8 (8 = full pass) or None if the search ran no funnel.
    """
    rows = r.get("funnel")
    if not rows:
        return None
    best = 0
    for row in rows:
        if row.get("all_passed"):
            best = max(best, len(GATE_ORDER))
            continue
        failed = str(row.get("failed_at") or "")
        if failed in GATE_ORDER:
            best = max(best, GATE_ORDER.index(failed))
    return best


def learning_stats(runs: list[dict]) -> dict:
    """Windowed 'is it getting smarter?' verdict from funnel gate depth."""
    rows = []
    for r in runs:
        d = funnel_depth(r)
        if d is None:
            continue
        fails = [
            str(x.get("failed_at"))
            for x in (r.get("funnel") or [])
            if x.get("failed_at")
        ]
        rows.append({
            "depth": d,
            "promoted": int(r.get("n_promoted") or 0),
            "kill_n": int(r.get("kill_archive_n") or 0),
            "fails": fails,
            "ts": r.get("timestamp") or r.get("_mtime") or 0,
        })
    if len(rows) < 4:
        return {"ok": False, "n": len(rows), "depths": [r["depth"] for r in rows]}

    half = len(rows) // 2
    prev, rec = rows[:half], rows[half:]

    def med(vals):
        return statistics.median(vals) if vals else 0.0

    prev_med = med([r["depth"] for r in prev])
    rec_med = med([r["depth"] for r in rec])
    rec_max = max(r["depth"] for r in rec)
    promoted_prev = sum(r["promoted"] for r in prev)
    promoted_rec = sum(r["promoted"] for r in rec)
    kill_growth = rows[-1]["kill_n"] - rows[0]["kill_n"]

    # Which gate is the current wall (most common recent failure)?
    recent_fails = [f for r in rec for f in r["fails"] if f in GATE_ORDER]
    wall = max(set(recent_fails), key=recent_fails.count) if recent_fails else None

    if promoted_rec > promoted_prev and promoted_rec > 0:
        direction, head = "SMARTER", (
            "Yes — ideas are now passing the whole exam more often than before."
        )
    elif promoted_rec > 0:
        direction, head = "SMARTER", (
            "Yes — some ideas pass the whole exam (shortlist hits in this window)."
        )
    elif rec_med >= prev_med + 0.5:
        direction, head = "SMARTER", (
            f"Slowly — ideas now typically clear {rec_med:.1f} of {N_GATES} exam gates, "
            f"up from {prev_med:.1f}. Closer to a shortlist pass."
        )
    elif rec_med <= prev_med - 0.5:
        direction, head = "WEAKER", (
            f"No — ideas are dying earlier in the exam "
            f"({prev_med:.1f} → {rec_med:.1f} of {N_GATES} gates)."
        )
    else:
        direction, head = "FLAT", (
            f"Not yet — exam progress is flat (typically {rec_med:.1f} of {N_GATES} gates, "
            f"best recent run reached {rec_max})."
        )

    detail_parts = []
    if wall:
        detail_parts.append(
            f"The wall right now is <b>{GATE_FRIENDLY.get(wall, wall)}</b> — "
            f"that is where most recent ideas die."
        )
    if kill_growth > 0:
        detail_parts.append(
            f"Memory is growing: <b>{kill_growth:,}</b> new bad idea-families were "
            f"blocked in this window, so the search wastes less time on known dead ends."
        )
    return {
        "ok": True,
        "n": len(rows),
        "depths": [r["depth"] for r in rows],
        "direction": direction,
        "head": head,
        "detail": " ".join(detail_parts),
        "prev_med": prev_med,
        "rec_med": rec_med,
        "rec_max": rec_max,
        "promoted_prev": promoted_prev,
        "promoted_rec": promoted_rec,
        "kill_growth": kill_growth,
        "wall": wall,
    }


def gate_depth_chart(depths: list[int], w: int = 720, h: int = 250) -> str:
    """Exam depth across retained clean-era searches. Never silently drops a day."""
    if not depths:
        return "<div class='muted'>No funnel data yet.</div>"
    # Keep the full retained clean era. 400 points is the on-disk cap, so an
    # older green/red result remains visible until the underlying file is pruned.
    ml, mr, mt, mb = 64, 16, 36, 48
    pw, ph = w - ml - mr, h - mt - mb
    n = len(depths)
    bw = max(4.0, pw / n - 3.0)
    top = float(len(GATE_ORDER))

    bars = []
    for i, d in enumerate(depths):
        x = ml + (pw / n) * i + 1.5
        bh = (d / top) * ph
        y = mt + ph - bh
        if d >= N_GATES:
            color = "#3ddc97"
        elif d >= 4:
            color = "#58a6ff"
        elif d >= 2:
            color = "#f5a623"
        else:
            color = "#e85d5d"
        bars.append(
            f"<rect x='{x:.1f}' y='{y:.1f}' width='{bw:.1f}' "
            f"height='{max(bh, 2):.1f}' rx='1.5' fill='{color}'/>"
        )

    gridlines = []
    for lvl in range(0, N_GATES + 1, 3):
        y = mt + ph - (lvl / top) * ph
        gridlines.append(
            f"<line x1='{ml}' y1='{y:.1f}' x2='{w - mr}' y2='{y:.1f}' "
            f"stroke='#243041' stroke-width='1'/>"
            f"<text x='{ml - 6}' y='{y + 4:.1f}' fill='#8b9bb4' font-size='11' "
            f"text-anchor='end'>{lvl}</text>"
        )
    pass_y = mt + ph - ph
    return f"""
<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;background:#0f1419;border-radius:8px">
  {''.join(gridlines)}
  <line x1="{ml}" y1="{pass_y}" x2="{w - mr}" y2="{pass_y}" stroke="#3ddc97" stroke-width="1" stroke-dasharray="4 4"/>
  <text x="{w - mr}" y="{pass_y + 14}" fill="#3ddc97" font-size="11" text-anchor="end">{N_GATES} = passed the whole exam</text>
  {''.join(bars)}
  <text x="18" y="{mt + ph/2:.1f}" fill="#8b9bb4" font-size="11" text-anchor="middle" transform="rotate(-90 18 {mt + ph/2:.1f})">Y-axis: strict exam gates cleared</text>
  <text x="{ml}" y="{h - 24}" fill="#8b9bb4" font-size="11">older retained searches</text>
  <text x="{(ml+w-mr)/2:.0f}" y="{h - 8}" fill="#8b9bb4" font-size="11" text-anchor="middle">X-axis: finished search order (full retained history)</text>
  <text x="{w - mr}" y="{h - 24}" fill="#8b9bb4" font-size="11" text-anchor="end">newer searches</text>
</svg>"""


def vintage_chart(cohorts: list[dict], w: int = 720, h: int = 250) -> str:
    """Weekly champion skill percentile vs same-day random strategies."""
    if not cohorts:
        return ""
    ml, mr, mt, mb = 40, 10, 14, 34
    pw, ph = w - ml - mr, h - mt - mb
    n = len(cohorts)
    xs = [ml + (pw / max(n - 1, 1)) * i if n > 1 else ml + pw / 2 for i in range(n)]

    def y_of(pct: float) -> float:
        return mt + ph - (max(0.0, min(100.0, pct)) / 100.0) * ph

    pts = " ".join(f"{x:.1f},{y_of(c['med_pct']):.1f}" for x, c in zip(xs, cohorts))
    dots = "".join(
        f"<circle cx='{x:.1f}' cy='{y_of(c['med_pct']):.1f}' r='4' "
        f"fill='{'#3ddc97' if c['med_pct'] >= 60 else ('#f5a623' if c['med_pct'] >= 45 else '#e85d5d')}'>"
        f"<title>{c['week']}: beats {c['med_pct']:.0f}% of randoms "
        f"(champ ${c['med_champ_pnl30']:.2f}/30d vs random ${c['med_rand_pnl30']:.2f}/30d, "
        f"{c['n']} champions)</title></circle>"
        for x, c in zip(xs, cohorts)
    )
    grid = []
    for lvl in (0, 25, 50, 75, 100):
        y = y_of(lvl)
        dash = "stroke-dasharray='4 4' stroke='#8b9bb4'" if lvl == 50 else "stroke='#243041'"
        grid.append(
            f"<line x1='{ml}' y1='{y:.1f}' x2='{w - mr}' y2='{y:.1f}' {dash} stroke-width='1'/>"
            f"<text x='{ml - 6}' y='{y + 4:.1f}' fill='#8b9bb4' font-size='11' "
            f"text-anchor='end'>{lvl}</text>"
        )
    labels = ""
    if n >= 1:
        labels = (
            f"<text x='{ml}' y='{h - 8}' fill='#8b9bb4' font-size='11'>{cohorts[0]['week']}</text>"
            f"<text x='{w - mr}' y='{h - 8}' fill='#8b9bb4' font-size='11' "
            f"text-anchor='end'>{cohorts[-1]['week']}</text>"
        )
    return f"""
<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;background:#0f1419;border-radius:8px">
  {''.join(grid)}
  <text x="{w - mr}" y="{y_of(50) - 6:.1f}" fill="#8b9bb4" font-size="11" text-anchor="end">50 = no better than random</text>
  <polyline points="{pts}" fill="none" stroke="#58a6ff" stroke-width="2"/>
  {dots}
  <text x="18" y="{mt + ph/2:.1f}" fill="#8b9bb4" font-size="11" text-anchor="middle" transform="rotate(-90 18 {mt + ph/2:.1f})">Y-axis: champion percentile vs random controls</text>
  {labels}
  <text x="{(ml+w-mr)/2:.0f}" y="{h - 8}" fill="#8b9bb4" font-size="11" text-anchor="middle">X-axis: control-cohort week</text>
</svg>"""


def vintage_html_block(s: dict) -> str:
    vs = s.get("vintage") or {}
    if not vs.get("ok"):
        reason = vs.get("reason") or "not started yet"
        return f"""
    <div class="note" style="margin-top:12px">
      <b>Forward ledger (the real proof):</b> every cycle's champion is frozen and
      later re-tested only on candles that arrived AFTER it was created — overfitting
      cannot help there. Status: <b>{reason}</b>. First points appear ~3 days after
      freezing; a trustworthy trend needs ~4 weeks.
    </div>"""
    badge = {
        "SMARTER": "ok", "WEAKER": "bad", "FLAT": "warn",
        "NO EDGE YET": "bad", "TOO EARLY": "warn",
    }.get(vs.get("verdict"), "warn")
    chart = vintage_chart(vs.get("cohorts") or [])
    return f"""
    <div style="margin-top:14px">
      <div class="row between">
        <b>Forward ledger — the real proof</b>
        <span class="badge {badge}">{vs.get('verdict')}</span>
      </div>
      <p style="margin-top:6px">{vs.get('verdict_text')}</p>
      <div class="note" style="margin-bottom:8px">
        <b>Chart D — frozen champions on unseen future data:</b> each dot is a week of
        frozen champions, scored ONLY on candles that arrived after they were frozen,
        as a percentile vs 20 random strategies frozen the same day (regime-proof).
        Above the dashed 50-line = real skill. Rising = the engine is genuinely
        getting smarter. Flat near 50 for months = no persistent edge with the
        current features.
      </div>
      {chart}
    </div>"""


def learning_html(s: dict) -> str:
    ls = s.get("learning") or {}
    vblock = vintage_html_block(s)
    if not ls.get("ok"):
        return f"""
  <div class="card">
    <h2>4. Is the bot getting smarter? <span class="section-hint">exam progress + forward proof</span></h2>
    <p class="lead">Too early to tell — only {ls.get('n', 0)} finished searches ran the strict exam so far.</p>
    <p class="muted">This section fills in once enough search jobs have been through the funnel.</p>
    {vblock}
  </div>"""

    badge = {"SMARTER": "ok", "WEAKER": "bad", "FLAT": "warn"}.get(ls.get("direction"), "warn")
    chart = gate_depth_chart(ls.get("depths") or [])
    detail = ls.get("detail") or ""
    return f"""
  <div class="card">
    <div class="row between">
      <h2>4. Is the bot getting smarter? <span class="section-hint">exam progress + forward proof</span></h2>
      <span class="badge {badge}">{ls.get('direction')}</span>
    </div>
    <p class="lead">{ls.get('head')}</p>
    {f'<p>{detail}</p>' if detail else ''}
    {vblock}
    <div class="grid4" style="margin-top:12px">
      <div class="stat">
        <div class="k">Exam gates cleared (typical)</div>
        <div class="v">{ls.get('rec_med', 0):.1f} / {N_GATES}</div>
        <div class="hint">earlier block: {ls.get('prev_med', 0):.1f} / {N_GATES}</div>
      </div>
      <div class="stat">
        <div class="k">Best recent run</div>
        <div class="v">{ls.get('rec_max', 0)} / {N_GATES}</div>
        <div class="hint">deepest any idea got lately</div>
      </div>
      <div class="stat">
        <div class="k">Full passes (recent vs earlier)</div>
        <div class="v">{ls.get('promoted_rec', 0)} vs {ls.get('promoted_prev', 0)}</div>
        <div class="hint">shortlist hits per window</div>
      </div>
      <div class="stat">
        <div class="k">New dead ends memorised</div>
        <div class="v">{ls.get('kill_growth', 0):,}</div>
        <div class="hint">bad families blocked in this window</div>
      </div>
    </div>
    <div style="margin-top:16px">
      <div class="note" style="margin-bottom:8px">
        <b>Chart C — exam progress:</b> each bar is one finished search; height = how many of
        the {N_GATES} exam gates its best idea cleared before dying.
        The exam order is: known-bad check → basic feasibility → unseen later data →
        beating buy-and-hold → chapter consistency → fee stress → rule nudges →
        front-running stress → luck filter.
        <b>Rising bars = real learning</b>, even while the shortlist is still empty.
      </div>
      {chart}
    </div>
  </div>"""


def _y_of(v: float, mn: float, mx: float, mt: float, ph: float) -> float:
    return mt + ph - (v - mn) / (mx - mn) * ph


def fitness_chart(vals: list[float], w: int = 720, h: int = 250) -> str:
    """Search-score chart only. No dollar targets on this axis."""
    if not vals:
        return '<div class="muted">Not enough finished searches yet.</div>'
    if len(vals) < 2:
        return '<div class="muted">Need at least 2 finished searches for this chart.</div>'

    clean = [sanitize_fitness(v) for v in vals]
    if all(v == 0 for v in clean) and any(abs(float(v or 0)) > 1000 for v in vals):
        return (
            f'<svg viewBox="0 0 {w} {h}" width="100%" height="{h}">'
            f'<rect width="{w}" height="{h}" fill="#0b1220"/>'
            f'<text x="20" y="{h//2}" fill="#fbbf24" font-size="14">'
            f'Old scores were broken. New runs will show here.</text></svg>'
        )

    ordered = sorted(clean)
    p05 = ordered[int(0.05 * (len(ordered) - 1))]
    p95 = ordered[int(0.95 * (len(ordered) - 1))]
    if p95 <= p05:
        p95 = p05 + 1.0
    scale_lo = min(p05, 0.0) - 5.0
    scale_hi = max(p95, 0.0) + 5.0
    if scale_hi <= scale_lo:
        scale_hi = scale_lo + 1.0
    clipped = [min(max(v, scale_lo), scale_hi) for v in clean]
    mn, mx = scale_lo, scale_hi
    med = statistics.median(clipped)

    win = max(3, min(7, len(clipped) // 5 or 3))
    ma = []
    for i in range(len(clipped)):
        lo = max(0, i - win + 1)
        ma.append(sum(clipped[lo : i + 1]) / (i - lo + 1))

    ml, mr, mt, mb = 56, 16, 48, 52
    pw, ph = w - ml - mr, h - mt - mb

    def xy(i, v):
        x = ml + i * pw / max(len(clipped) - 1, 1)
        y = mt + (1 - (v - mn) / (mx - mn)) * ph
        return x, y

    pts = " ".join(f"{xy(i,v)[0]:.1f},{xy(i,v)[1]:.1f}" for i, v in enumerate(clipped))
    ma_pts = " ".join(f"{xy(i,v)[0]:.1f},{xy(i,v)[1]:.1f}" for i, v in enumerate(ma))

    grid = []
    for i in range(5):
        tv = mn + i * (mx - mn) / 4
        y = xy(0, tv)[1]
        grid.append(
            f'<line x1="{ml}" y1="{y:.1f}" x2="{w-mr}" y2="{y:.1f}" stroke="#1a2332" stroke-width="1"/>'
        )
        grid.append(
            f'<text x="{ml-8}" y="{y+4:.1f}" fill="#8b9bb4" font-size="11" text-anchor="end">{tv:.0f}</text>'
        )

    refs = []
    if mn < 0 < mx:
        y0 = xy(0, 0.0)[1]
        refs.append(
            f'<line x1="{ml}" y1="{y0:.1f}" x2="{w-mr}" y2="{y0:.1f}" '
            f'stroke="#64748b" stroke-width="1.2" stroke-dasharray="3,3"/>'
            f'<text x="{ml+4}" y="{y0-4:.1f}" fill="#94a3b8" font-size="10">0 = break-even ranking</text>'
        )
    ym = xy(0, med)[1]
    refs.append(
        f'<line x1="{ml}" y1="{ym:.1f}" x2="{w-mr}" y2="{ym:.1f}" '
        f'stroke="#f5a623" stroke-width="1.2" stroke-dasharray="4,3"/>'
        f'<text x="{ml+4}" y="{ym+12:.1f}" fill="#f5a623" font-size="10">typical level ({med:.0f})</text>'
    )

    x0, y0p = xy(0, clipped[0])
    x1, y1p = xy(len(clipped) - 1, clipped[-1])
    legend = (
        f'<rect x="{ml}" y="8" width="14" height="3" fill="#38bdf8"/>'
        f'<text x="{ml+18}" y="12" fill="#c9d7ea" font-size="11">best idea each search</text>'
        f'<rect x="{ml+180}" y="8" width="14" height="3" fill="#a78bfa"/>'
        f'<text x="{ml+198}" y="12" fill="#c9d7ea" font-size="11">smoother trend</text>'
        f'<line x1="{ml+330}" y1="10" x2="{ml+346}" y2="10" stroke="#f5a623" stroke-width="1.5" stroke-dasharray="3,2"/>'
        f'<text x="{ml+350}" y="12" fill="#c9d7ea" font-size="11">typical</text>'
    )
    return f"""
    <svg viewBox="0 0 {w} {h}" width="100%" height="{h}" style="background:#0f1419;border-radius:8px">
      <text x="18" y="{mt + ph/2:.1f}" fill="#8b9bb4" font-size="11" text-anchor="middle" transform="rotate(-90 18 {mt + ph/2:.1f})">Y-axis: search ranking points (not dollars)</text>
      <text x="{ml}" y="32" fill="#e7eef7" font-size="14" font-weight="600">How good ideas look while searching</text>
      <text x="{w-mr}" y="32" fill="#8b9bb4" font-size="11" text-anchor="end">not money · not capital</text>
      {legend}
      {''.join(grid)}
      {''.join(refs)}
      <polyline fill="none" stroke="#a78bfa" stroke-width="2" opacity="0.9" points="{ma_pts}"/>
      <polyline fill="none" stroke="#38bdf8" stroke-width="2.5" points="{pts}"/>
      <circle cx="{x0:.1f}" cy="{y0p:.1f}" r="3" fill="#8b9bb4"/>
      <circle cx="{x1:.1f}" cy="{y1p:.1f}" r="3.5" fill="#38bdf8"/>
      <text x="{x1:.1f}" y="{y1p-8:.1f}" fill="#38bdf8" font-size="11" text-anchor="end">now {clipped[-1]:.0f}</text>
      <text x="{ml}" y="{h-18}" fill="#8b9bb4" font-size="11">older</text>
      <text x="{(ml+w-mr)/2:.0f}" y="{h-18}" fill="#8b9bb4" font-size="11" text-anchor="middle">X-axis: finished search order</text>
      <text x="{w-mr}" y="{h-18}" fill="#8b9bb4" font-size="11" text-anchor="end">newer</text>
      <text x="{ml}" y="{h-4}" fill="#6b7c94" font-size="10">
        Blue = that search's best idea ranking. Purple = short average. Higher can mean better hunt, not “we made $X”.
      </text>
    </svg>"""


def trades_chart(vals: list[int], w: int = 720, h: int = 230) -> str:
    """Trade-count bars with LAB min line only (engine gate)."""
    if len(vals) < 2:
        return '<div class="muted">Not enough finished searches yet.</div>'

    ml, mr, mt, mb = 56, 16, 48, 50
    pw, ph = w - ml - mr, h - mt - mb
    mx = max(max(vals), MIN_TRADES, 1)
    mx = max(mx, int(statistics.median(vals) * 2.5) if vals else mx)
    n = len(vals)
    bw = max(2.0, pw / n * 0.7)
    good = sum(1 for v in vals if v >= MIN_TRADES)
    bad = n - good
    med = statistics.median(vals)

    parts = [
        f'<text x="{ml}" y="20" fill="#e7eef7" font-size="14" font-weight="600">How many pretend trades each best idea took</text>',
        f'<text x="{w-mr}" y="20" fill="#8b9bb4" font-size="11" text-anchor="end">need at least {MIN_TRADES}</text>',
        f'<rect x="{ml}" y="28" width="10" height="10" fill="#3ddc97" rx="1"/>'
        f'<text x="{ml+14}" y="37" fill="#c9d7ea" font-size="11">enough (≥{MIN_TRADES})</text>'
        f'<rect x="{ml+130}" y="28" width="10" height="10" fill="#e85d5d" rx="1"/>'
        f'<text x="{ml+144}" y="37" fill="#c9d7ea" font-size="11">too few</text>'
        f'<line x1="{ml+230}" y1="33" x2="{ml+246}" y2="33" stroke="#f5a623" stroke-width="2" stroke-dasharray="4,2"/>'
        f'<text x="{ml+250}" y="37" fill="#c9d7ea" font-size="11">minimum the bot accepts ({MIN_TRADES})</text>',
    ]
    for i in range(5):
        tv = mx * i / 4
        y = mt + ph - (tv / mx) * ph
        parts.append(
            f'<line x1="{ml}" y1="{y:.1f}" x2="{w-mr}" y2="{y:.1f}" stroke="#1a2332" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{ml-8}" y="{y+4:.1f}" fill="#8b9bb4" font-size="11" text-anchor="end">{tv:.0f}</text>'
        )
    y30 = mt + ph - (MIN_TRADES / mx) * ph
    parts.append(
        f'<line x1="{ml}" y1="{y30:.1f}" x2="{w-mr}" y2="{y30:.1f}" stroke="#f5a623" stroke-width="1.8" stroke-dasharray="5,3"/>'
    )
    parts.append(
        f'<text x="{w-mr}" y="{y30-5:.1f}" fill="#f5a623" font-size="11" text-anchor="end">'
        f'must clear this line ({MIN_TRADES})</text>'
    )
    for i, v in enumerate(vals):
        x = ml + i * pw / max(n - 1, 1) - bw / 2
        bh = min(ph, (min(v, mx) / mx) * ph)
        y = mt + ph - bh
        color = "#3ddc97" if v >= MIN_TRADES else "#e85d5d"
        parts.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{bh:.1f}" fill="{color}" opacity="0.9" rx="1"/>'
        )
    parts.append(
        f'<text x="{ml}" y="{h-16}" fill="#8b9bb4" font-size="11">older</text>'
        f'<text x="{(ml+w-mr)/2:.0f}" y="{h-16}" fill="#8b9bb4" font-size="11" text-anchor="middle">'
        f'each bar = one finished search</text>'
        f'<text x="{w-mr}" y="{h-16}" fill="#8b9bb4" font-size="11" text-anchor="end">newer</text>'
    )
    parts.append(
        f'<text x="{ml}" y="{h-2}" fill="#6b7c94" font-size="10">'
        f'{good}/{n} cleared the minimum · typical {med:.0f} trades · {bad} too thin to trust</text>'
    )
    return (
        f'<svg viewBox="0 0 {w} {h}" width="100%" height="{h}" '
        f'style="background:#0f1419;border-radius:8px">'
        + "".join(parts)
        + "</svg>"
    )


def pnl_chart(vals: list[float], w: int = 720, h: int = 250) -> str:
    """Full-history paper P&L on $100 book. LAB target is simply > $0."""
    if len(vals) < 2:
        return '<div class="muted">Not enough finished searches yet.</div>'

    clean = []
    for v in vals:
        try:
            x = float(v or 0)
        except Exception:
            x = 0.0
        if abs(x) > 1e6:
            x = 0.0
        clean.append(x)

    ordered = sorted(clean)
    p05 = ordered[int(0.05 * (len(ordered) - 1))]
    p95 = ordered[int(0.95 * (len(ordered) - 1))]
    # Engine LAB gate: P&L > 0. Do NOT put +$5 paper-forward target here.
    lab_floor = 0.0
    scale_lo = min(p05, lab_floor, 0.0) - max(5.0, abs(p05) * 0.1)
    scale_hi = max(p95, lab_floor, 0.0) + max(5.0, abs(p95) * 0.1)
    if scale_hi <= scale_lo:
        scale_hi = scale_lo + 1.0
    clipped = [min(max(v, scale_lo), scale_hi) for v in clean]
    mn, mx = scale_lo, scale_hi
    med = statistics.median(clipped)

    win = max(3, min(7, len(clipped) // 5 or 3))
    ma = []
    for i in range(len(clipped)):
        lo = max(0, i - win + 1)
        ma.append(sum(clipped[lo : i + 1]) / (i - lo + 1))

    ml, mr, mt, mb = 56, 16, 48, 56
    pw, ph = w - ml - mr, h - mt - mb

    def xy(i, v):
        x = ml + i * pw / max(len(clipped) - 1, 1)
        y = mt + (1 - (v - mn) / (mx - mn)) * ph
        return x, y

    pts = " ".join(f"{xy(i,v)[0]:.1f},{xy(i,v)[1]:.1f}" for i, v in enumerate(clipped))
    ma_pts = " ".join(f"{xy(i,v)[0]:.1f},{xy(i,v)[1]:.1f}" for i, v in enumerate(ma))

    grid = []
    for i in range(5):
        tv = mn + i * (mx - mn) / 4
        y = xy(0, tv)[1]
        grid.append(
            f'<line x1="{ml}" y1="{y:.1f}" x2="{w-mr}" y2="{y:.1f}" stroke="#1a2332" stroke-width="1"/>'
        )
        grid.append(
            f'<text x="{ml-8}" y="{y+4:.1f}" fill="#8b9bb4" font-size="11" text-anchor="end">${tv:.0f}</text>'
        )

    y0 = xy(0, 0.0)[1]
    refs = [
        f'<line x1="{ml}" y1="{y0:.1f}" x2="{w-mr}" y2="{y0:.1f}" '
        f'stroke="#3ddc97" stroke-width="2" stroke-dasharray="6,4"/>'
        f'<text x="{w-mr}" y="{y0-5:.1f}" fill="#3ddc97" font-size="11" text-anchor="end">'
        f'LAB gate: profit > $0 on ${BOOK_USD:.0f} history</text>',
    ]
    ym = xy(0, med)[1]
    refs.append(
        f'<line x1="{ml}" y1="{ym:.1f}" x2="{w-mr}" y2="{ym:.1f}" '
        f'stroke="#f5a623" stroke-width="1.2" stroke-dasharray="4,3"/>'
        f'<text x="{ml+4}" y="{ym+12:.1f}" fill="#f5a623" font-size="10">typical ${med:.0f}</text>'
    )

    x1, y1p = xy(len(clipped) - 1, clipped[-1])
    above = sum(1 for v in clean if v > 0)
    end_cap = BOOK_USD + clean[-1]
    legend = (
        f'<rect x="{ml}" y="8" width="14" height="3" fill="#38bdf8"/>'
        f'<text x="{ml+18}" y="12" fill="#c9d7ea" font-size="11">paper profit/loss on history</text>'
        f'<rect x="{ml+220}" y="8" width="14" height="3" fill="#a78bfa"/>'
        f'<text x="{ml+238}" y="12" fill="#c9d7ea" font-size="11">smoother trend</text>'
        f'<line x1="{ml+370}" y1="10" x2="{ml+386}" y2="10" stroke="#3ddc97" stroke-width="2" stroke-dasharray="4,2"/>'
        f'<text x="{ml+390}" y="12" fill="#c9d7ea" font-size="11">must stay above $0</text>'
    )
    return f"""
    <svg viewBox="0 0 {w} {h}" width="100%" height="{h}" style="background:#0f1419;border-radius:8px">
      <text x="18" y="{mt + ph/2:.1f}" fill="#8b9bb4" font-size="11" text-anchor="middle" transform="rotate(-90 18 {mt + ph/2:.1f})">Y-axis: history P&amp;L ($, fake ${BOOK_USD:.0f} book)</text>
      <text x="{ml}" y="32" fill="#e7eef7" font-size="14" font-weight="600">Paper profit on old price history (fake ${BOOK_USD:.0f})</text>
      <text x="{w-mr}" y="32" fill="#8b9bb4" font-size="11" text-anchor="end">not the 30-day forward win</text>
      {legend}
      {''.join(grid)}
      {''.join(refs)}
      <polyline fill="none" stroke="#a78bfa" stroke-width="2" opacity="0.9" points="{ma_pts}"/>
      <polyline fill="none" stroke="#38bdf8" stroke-width="2.5" points="{pts}"/>
      <circle cx="{x1:.1f}" cy="{y1p:.1f}" r="3.5" fill="#38bdf8"/>
      <text x="{x1:.1f}" y="{y1p-8:.1f}" fill="#38bdf8" font-size="11" text-anchor="end">
        now ${clean[-1]:+.0f} → book ~${end_cap:.0f}
      </text>
      <text x="{ml}" y="{h-18}" fill="#8b9bb4" font-size="11">older</text>
      <text x="{(ml+w-mr)/2:.0f}" y="{h-18}" fill="#8b9bb4" font-size="11" text-anchor="middle">X-axis: finished search order</text>
      <text x="{w-mr}" y="{h-18}" fill="#8b9bb4" font-size="11" text-anchor="end">newer</text>
      <text x="{ml}" y="{h-4}" fill="#6b7c94" font-size="10">
        Starts from ${BOOK_USD:.0f}. Above green = history profitable (LAB). Big history $ is common and still not the real 30-day paper win.
        {above}/{len(clean)} searches ended above $0.
      </text>
    </svg>"""


def chart_commentary(fit_vals: list[float], trade_vals: list[int], trend: dict) -> dict:
    """Plain English under the plots."""
    n = len(fit_vals)
    if n < 3:
        return {
            "fitness": "Not enough searches yet.",
            "trades": "Not enough searches yet.",
            "overall": trend.get("why") or "Still collecting data.",
        }

    last_med = statistics.median(fit_vals[-max(5, n // 3) :])
    first_med = statistics.median(fit_vals[: max(5, n // 3)])
    if last_med > first_med * 1.08:
        fit_c = (
            f"The ranking line is a bit higher lately ({first_med:.0f} → {last_med:.0f}). "
            f"That only means the hunt likes recent ideas more — not that you made cash."
        )
    elif last_med < first_med * 0.92:
        fit_c = (
            f"The ranking line is a bit lower lately ({first_med:.0f} → {last_med:.0f}). "
            f"The hunt may be stricter, or recent winners look weaker on later data."
        )
    else:
        fit_c = f"The ranking line is roughly steady (around {last_med:.0f})."

    good = sum(1 for t in trade_vals if t >= MIN_TRADES)
    pct = 100.0 * good / max(len(trade_vals), 1)
    tmed = statistics.median(trade_vals) if trade_vals else 0
    if pct >= 90 and tmed >= MIN_TRADES:
        trade_c = (
            f"Good: {good}/{len(trade_vals)} best ideas traded enough times "
            f"(typical {tmed:.0f}). We can at least talk about them."
        )
    elif pct >= 50:
        trade_c = (
            f"Mixed: {good}/{len(trade_vals)} clear the {MIN_TRADES}-trade line "
            f"(typical {tmed:.0f})."
        )
    else:
        trade_c = (
            f"Worrying: only {good}/{len(trade_vals)} have {MIN_TRADES}+ trades. "
            f"Thin samples are mostly noise."
        )

    overall = trend.get("why") or ""
    detail = trend.get("detail") or ""
    if detail:
        overall = f"{overall} {detail}".strip()

    if trade_vals:
        last_t = trade_vals[-1]
        last_f = sanitize_fitness(fit_vals[-1])
        if last_t < MIN_TRADES:
            overall += f" Latest search ranked {last_f:.0f} on only {last_t} trades: treat as noise."
        else:
            overall += f" Latest search: ranking {last_f:.0f} with {last_t} pretend trades."

    return {"fitness": fit_c, "trades": trade_c, "overall": overall}


def fmt_age(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s ago"
    if seconds < 3600:
        return f"{seconds/60:.0f}m ago"
    return f"{seconds/3600:.1f}h ago"


def collect_status() -> dict:
    runs = load_all_runs()
    broad = [r for r in runs if is_broad_era(r)] or runs[-50:]
    # Focus charts / trend on recent broad/funnel era only
    recent = broad[-30:] if len(broad) >= 10 else broad
    prev = broad[-60:-30] if len(broad) >= 60 else broad[: max(0, len(broad) - len(recent))]

    activity = live_activity()
    proc = process_alive()
    trials = load_json(EVO / "trial_count.json") or {}
    champions = load_json(EVO / "champions.json") or {}
    last_promoted = load_json(EVO / "last_promoted.json")
    funnel_latest = load_json(EVO / "funnel_results" / "latest.json")
    latest_genome = load_json(EVO / "best_genome_latest.json") or {}

    last_run = runs[-1] if runs else None
    trend = compute_honest_trend(recent, prev if prev else recent[:1])
    recent_sum = summarize_window(recent)
    all_sum = summarize_window(runs[-100:] if runs else [])
    # The intelligence charts deliberately retain the full current-engine era.
    # Do not use the 30-search status window here, or yesterday's pass/fail dots
    # disappear after a busy run.
    learning = learning_stats([r for r in broad if is_funnel_era(r)])
    try:
        from evolution.vintage_ledger import ledger_summary
        vintage = ledger_summary()
    except Exception as e:
        vintage = {"ok": False, "reason": f"ledger unavailable ({e})"}

    # Questionable metrics to surface
    questions = []
    if last_run and run_trades(last_run) < MIN_TRADES and sanitize_fitness(last_run.get("best_fitness") or 0) > 50:
        questions.append(
            "Last cycle has high fitness but under 30 trades — treat as noise, not a champion."
        )
    if recent_sum["pct_random"] > 10:
        questions.append(
            f"RANDOM still winning {recent_sum['pct_random']:.0f}% of recent cycles. Selection pool may still be polluted."
        )
    if recent_sum["cycles_with_funnel"] and recent_sum["promoted"] == 0:
        questions.append(
            "Funnel is active and has promoted 0 strategies recently. High search scores are not deployable yet."
        )
    if all_sum["fit_avg"] > all_sum["fit_med"] * 3 and all_sum["n"] > 10:
        questions.append(
            "Mean fitness >> median fitness. A few outliers are distorting averages. Prefer median / N≥30."
        )
    if not proc["running"] and (not activity or not activity.get("live")):
        questions.append("No evolution process detected. Dashboard may be looking at stale history.")
    if activity and activity.get("live") and activity.get("phase") == "evaluate":
        questions.append(
            "Engine is mid-evaluation. Fitness numbers on this page are from finished cycles only."
        )
    if not questions:
        questions.append(
            "No major dashboard lies detected right now — still require funnel PASS before treating anything as real."
        )

    return {
        "now": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "activity": activity,
        "process": proc,
        "runs_total": len(runs),
        "runs_broad": len(broad),
        "last_run": last_run,
        "recent": recent,
        "recent_sum": recent_sum,
        "all_sum": all_sum,
        "trend": trend,
        "learning": learning,
        "vintage": vintage,
        "trials_total": int(trials.get("total_trials") or 0),
        "champions": champions,
        "last_promoted": last_promoted,
        "funnel_latest": funnel_latest,
        "latest_genome": latest_genome,
        "questions": questions,
        "min_trades": MIN_TRADES,
    }


def story_summary(s: dict) -> dict:
    """One clear paragraph about current state."""
    proc = s.get("process") or {}
    a = s.get("activity") or {}
    last = s.get("last_run")
    rs = s.get("recent_sum") or {}
    kill = load_json(EVO / "killed_dna.json") or {}
    kill_n = int(kill.get("n") or 0)

    running = bool(proc.get("running"))
    if running and a.get("live"):
        phase = a.get("phase", "?")
        if phase == "evaluate":
            now = (
                f"Yes — the bot is testing strategy idea "
                f"<b>{a.get('genome_index')}/{a.get('population')}</b> "
                f"in breeding round <b>{a.get('generation', 0)}</b>."
            )
        elif phase == "funnel":
            now = "Yes — a search just finished and the stricter exam (funnel) is running."
        elif phase == "breeding":
            now = f"Yes — mixing good ideas into breeding round <b>{a.get('generation', 0)}</b>."
        elif phase == "done":
            now = "A search just finished. Another will start in a few seconds."
        elif phase == "cycle_start":
            now = f"Yes — starting search job <b>#{a.get('cycle', '?')}</b>."
        else:
            now = f"Yes — working (step: {phase})."
    elif running:
        now = "The search program is up, but the heartbeat looks a bit stale."
    else:
        now = "No — the search program is not running right now."

    if last:
        bt = last.get("backtest") or {}
        trades = int(bt.get("total_trades") or 0)
        fit = sanitize_fitness(last.get("best_fitness") or 0)
        promo = last.get("n_promoted")
        logic = (last.get("best_genome") or {}).get("entry_logic", "?")
        if abs(float(last.get("best_fitness") or 0)) > 1000:
            score_note = "score was broken (old math); ignore the huge number"
        else:
            score_note = f"score {fit:.0f}"
        if trades < MIN_TRADES:
            last_line = (
                f"Last finished search picked an idea that only made <b>{trades}</b> paper trades. "
                f"That is too few to trust ({score_note})."
            )
        elif promo == 0:
            last_line = (
                f"Last finished search's best idea did <b>{trades}</b> paper trades "
                f"(type <b>{logic}</b>, {score_note}), then the strict exam "
                f"<b>rejected it</b> / the shortlist stayed empty."
            )
        else:
            last_line = (
                f"Last finished search found <b>{promo}</b> idea(s) that passed the strict exam. "
                f"Best idea in that search still made only paper profits on history "
                f"({trades} trades, {score_note}, type {logic}). Not live money."
            )
    else:
        last_line = "No finished search is saved yet."

    # Trend in plain words
    tdir = (s.get("trend") or {}).get("direction", "?")
    n30 = rs.get("pct_n30", 0)
    tmed = rs.get("trade_med", 0)
    prom = rs.get("promoted", 0)
    if tdir == "HEALTHIER":
        trend_line = (
            f"Over the last searches, things look a bit healthier: "
            f"winners usually take enough paper trades ({n30:.0f}% clear {MIN_TRADES}+)."
        )
    elif tdir == "WORSE":
        trend_line = (
            f"Over the last searches, quality looks a bit weaker "
            f"(score slipped), but trade counts are still often OK "
            f"(typical {tmed:.0f} trades)."
        )
    elif tdir == "FLAT":
        trend_line = (
            f"Over the last searches, quality is mostly flat — "
            f"not clearly better or worse. Typical winner still does about "
            f"<b>{tmed:.0f}</b> paper trades; "
            f"<b>{n30:.0f}%</b> of winners had enough trades to discuss."
        )
    else:
        trend_line = "Too few finished searches yet to call a trend."

    promo_line = (
        f"Across recent searches, <b>{prom}</b> ideas passed the strict exam (shortlist hits). "
        f"That is still paper history, not a permission to trade live. "
        f"Bad idea-families blocked forever: <b>{kill_n}</b>."
    )

    return {
        "running_line": now,
        "last_line": last_line,
        "trend_line": trend_line,
        "promo_line": promo_line,
        "kill_n": kill_n,
    }


def activity_html(s: dict) -> str:
    a = s.get("activity") or {}
    proc = s.get("process") or {}
    story = story_summary(s)

    if proc.get("running"):
        badge = '<span class="badge ok">SEARCHING</span>'
    else:
        badge = '<span class="badge bad">STOPPED</span>'

    if a and a.get("live"):
        live_badge = '<span class="badge ok">HEARTBEAT OK</span>'
        age = fmt_age(float(a.get("age_s") or 0))
    elif a:
        live_badge = '<span class="badge warn">HEARTBEAT OLD</span>'
        age = fmt_age(float(a.get("age_s") or 0))
    else:
        live_badge = '<span class="badge bad">NO HEARTBEAT</span>'
        age = "—"

    # Concrete counters for beginners
    bits = []
    if a.get("generation") is not None and a.get("phase") in ("evaluate", "breeding", "generation_done"):
        bits.append(f"breeding round {a.get('generation')}")
    if a.get("genome_index") and a.get("population"):
        bits.append(f"idea {a.get('genome_index')} of {a.get('population')}")
    if a.get("cycle") is not None:
        bits.append(f"search job #{a.get('cycle')}")
    if a.get("kill_archive_n") is not None:
        bits.append(f"{a.get('kill_archive_n')} blocked bad families")
    detail = " · ".join(bits) if bits else "waiting"

    return f"""
  <div class="card">
    <div class="row between">
      <h2 style="margin:0">1. Is the bot searching right now?</h2>
      <div class="row gap">{badge} {live_badge}</div>
    </div>
    <p class="lead">{story['running_line']}</p>
    <div class="muted">Detail: {detail} · last ping {age}</div>
  </div>"""


def last_cycle_html(s: dict) -> str:
    r = s.get("last_run")
    story = story_summary(s)
    if not r:
        return f"""
  <div class="card">
    <h2>2. What just happened?</h2>
    <p class="lead">{story['last_line']}</p>
  </div>"""

    bt = r.get("backtest") or {}
    g = r.get("best_genome") or {}
    trades = int(bt.get("total_trades") or 0)
    fit = sanitize_fitness(r.get("best_fitness") or 0)
    raw_fit = float(r.get("best_fitness") or 0)
    pnl = float(bt.get("total_pnl") or 0)
    win = float(bt.get("win_rate") or 0)
    logic = g.get("entry_logic", "?")
    n_prom = r.get("n_promoted")
    gens = r.get("generations_run")
    trade_ok = trades >= MIN_TRADES
    trade_color = "#3ddc97" if trade_ok else "#e85d5d"
    fit_label = f"{fit:.0f}" if abs(raw_fit) <= 1000 else "broken (old)"
    fit_note = "ranking only (~-200..+200)" if abs(raw_fit) <= 1000 else "ignore 1e17 legacy scores"

    # Funnel outcomes in plain language
    funnel_rows = r.get("funnel") or []
    funnel_bits = []
    for fr in funnel_rows[:5]:
        ok = fr.get("all_passed")
        if ok:
            funnel_bits.append("passed the strict exam")
        else:
            why = fr.get("failed_at") or "unknown step"
            plain_why = {
                "feasibility": "not enough good trades / lost money on full history",
                "oos": "did worse on unseen later data",
                "walk_forward": "failed when history was sliced into chapters",
                "fee_stress": "broke when fees were raised",
                "perturbation": "fell apart when rules were nudged slightly",
                "mev": "died under frontrun stress",
                "dsr": "score not trustworthy after many tries",
                "kill_archive": "this idea-family was already known bad",
                "RANDOM banned": "random guessing is not allowed",
            }.get(why, why)
            funnel_bits.append(f"failed ({plain_why})")
    if funnel_bits:
        funnel_txt = "<ul class='plain'>" + "".join(f"<li>{b}</li>" for b in funnel_bits) + "</ul>"
    else:
        funnel_txt = "<p class='muted'>No strict-exam details saved for this search.</p>"

    conds = g.get("entry_conditions") or []
    if conds:
        rule = " AND ".join(
            f"{c.get('indicator')} {c.get('operator')} {float(c.get('threshold') or 0):.1f}"
            for c in conds[:3]
        )
        rule_html = f"<div class='note'><b>Rough rule of that best idea:</b> <code>{rule}</code></div>"
    else:
        rule_html = ""

    trust = (
        f"Enough paper trades to discuss (≥{MIN_TRADES})."
        if trade_ok
        else f"Too few paper trades (<{MIN_TRADES}). Ignore the score."
    )

    return f"""
  <div class="card">
    <h2>2. What just happened? <span class="section-hint">one finished search job</span></h2>
    <p class="lead">{story['last_line']}</p>
    <div class="grid4" style="margin-top:12px">
      <div class="stat">
        <div class="k">Pretend trades</div>
        <div class="v" style="color:{trade_color}">{trades}</div>
        <div class="hint">{trust}</div>
      </div>
      <div class="stat">
        <div class="k">Computer ranking</div>
        <div class="v">{fit_label}</div>
        <div class="hint">{fit_note}</div>
      </div>
      <div class="stat">
        <div class="k">History win rate / P&L</div>
        <div class="v">{win*100:.0f}% / ${pnl:.2f}</div>
        <div class="hint">fake ${BOOK_USD:.0f} on old prices → book ~${BOOK_USD + pnl:.0f}</div>
      </div>
      <div class="stat">
        <div class="k">Passed strict exam?</div>
        <div class="v">{n_prom if n_prom is not None else "—"}</div>
        <div class="hint">0 = none good enough this job</div>
      </div>
    </div>
    <p style="margin-top:12px">
      Best idea type: <b>{logic}</b>
      · breeding rounds in that search: <b>{gens if gens is not None else "—"}</b>
    </p>
    {rule_html}
    <div style="margin-top:10px"><b>Strict exam on the top ideas from that search:</b></div>
    {funnel_txt}
  </div>"""


def fmt_exam_score(v) -> str:
    """Old runs stored exploding Sharpe-era scores (1e17). Flag, don't print."""
    try:
        x = float(v or 0)
    except (TypeError, ValueError):
        return "?"
    if abs(x) > 1e6:
        return "legacy (broken old score)"
    return f"{x:.0f}"


def champions_html(s: dict) -> str:
    champs = (s.get("champions") or {}).get("champions") or []
    last_p = s.get("last_promoted")
    if not champs and not (last_p and last_p.get("all_passed")):
        return """
  <div class="card">
    <h2>5. Shortlist (ideas that passed the strict exam)</h2>
    <p class="lead">Empty for now.</p>
    <p class="muted">
      An idea only reaches this list after enough paper trades, a clean check on later data,
      chapter-by-chapter tests, fee stress, and a few other hardness checks.
      Empty is honest. It does not mean the search is broken.
    </p>
  </div>"""

    rows = []
    seen = set()
    if last_p and last_p.get("all_passed"):
        g = last_p.get("genome") or {}
        gid = last_p.get("genome_id") or ""
        seen.add(gid)
        rows.append(
            f"<tr><td>Latest pass</td><td><b>{g.get('entry_logic','?')}</b></td>"
            f"<td>{fmt_exam_score(last_p.get('score'))}</td>"
            f"<td class='muted'><code>{gid[:36]}</code></td></tr>"
        )
    for c in champs[:6]:
        g = c.get("genome") or {}
        gid = c.get("genome_id") or ""
        if gid in seen:
            continue
        rows.append(
            f"<tr><td>On shortlist</td><td><b>{g.get('entry_logic','?')}</b></td>"
            f"<td>{fmt_exam_score(c.get('score'))}</td>"
            f"<td class='muted'><code>{gid[:36]}</code></td></tr>"
        )

    return f"""
  <div class="card">
    <h2>5. Shortlist <span class="section-hint">still paper only — not live trading</span></h2>
    <p class="lead">
      These ideas survived the strict exam. That is the first serious filter, not a green light for real money.
    </p>
    <table>
      <tr><th>Status</th><th>Idea type</th><th>Exam score</th><th>Id</th></tr>
      {''.join(rows)}
    </table>
  </div>"""


def html_page(s: dict) -> str:
    recent = s.get("recent") or []
    fit_vals = [sanitize_fitness(r.get("best_fitness") or 0) for r in recent]
    trade_vals = [run_trades(r) for r in recent]
    pnl_vals = [float((r.get("backtest") or {}).get("total_pnl") or 0) for r in recent]
    fit_chart = fitness_chart(fit_vals)
    money_chart = pnl_chart(pnl_vals)
    story = story_summary(s)
    rs = s.get("recent_sum") or {}
    kill_n = story.get("kill_n", 0)
    n = len(trade_vals)
    good = sum(1 for t in trade_vals if t >= MIN_TRADES)
    thin = n - good
    pos_pnl = sum(1 for p in pnl_vals if p > 0)
    trade_med = statistics.median(trade_vals) if trade_vals else 0
    paper_end_min = BOOK_USD + PAPER_MIN_NET_PNL_USD
    paper_end_tgt = BOOK_USD + 10.0
    try:
        from success_criteria import PAPER_TARGET_NET_PNL_USD
        paper_end_tgt = BOOK_USD + PAPER_TARGET_NET_PNL_USD
    except Exception:
        pass

    # Only resurface a trade chart if thin samples reappear (gate regression)
    trade_alert = ""
    if n >= 5 and good < n:
        trade_alert = f"""
    <div class="warnbox">
      <b>Trade-count warning:</b> {thin}/{n} recent best ideas had fewer than {MIN_TRADES} pretend trades.
      The bot is supposed to reject thin samples — this chart used to watch that. Worth a look.
    </div>
    <div style="margin-top:12px">{trades_chart(trade_vals)}</div>
"""

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<meta http-equiv="refresh" content="10"/>
<title>Bot status (simple)</title>
<style>
  :root {{ --bg:#0b0f14; --card:#151b23; --text:#e7eef7; --muted:#8b9bb4; --ok:#3ddc97; --bad:#e85d5d; --warn:#f5a623; --line:#243041; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
         background:var(--bg); color:var(--text); padding:20px; max-width:920px; margin-left:auto; margin-right:auto; }}
  h1 {{ margin:0; font-size:26px; }}
  h2 {{ margin:0 0 10px; font-size:17px; font-weight:700; }}
  .sub {{ color:var(--muted); font-size:13px; margin:6px 0 16px; }}
  .card {{ background:var(--card); border:1px solid var(--line); border-radius:12px; padding:16px 18px; margin-bottom:14px; }}
  .badge {{ display:inline-block; padding:3px 9px; border-radius:999px; font-size:11px; font-weight:700; }}
  .badge.ok {{ background:#1a2a22; color:var(--ok); }}
  .badge.bad {{ background:#2a1a22; color:var(--bad); }}
  .badge.warn {{ background:#2a2418; color:var(--warn); }}
  .grid4 {{ display:grid; grid-template-columns:repeat(4,1fr); gap:10px; }}
  .grid3 {{ display:grid; grid-template-columns:repeat(3,1fr); gap:10px; }}
  @media (max-width:800px) {{ .grid4,.grid3 {{ grid-template-columns:1fr; }} }}
  .stat {{ background:#0f1419; border-radius:8px; padding:10px 12px; }}
  .k {{ color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:.04em; }}
  .v {{ font-size:22px; font-weight:700; margin-top:2px; }}
  .hint {{ color:var(--muted); font-size:12px; margin-top:4px; line-height:1.35; }}
  .section-hint {{ color:var(--muted); font-size:12px; font-weight:400; margin-left:6px; }}
  .row {{ display:flex; align-items:center; }}
  .between {{ justify-content:space-between; }}
  .gap {{ gap:8px; }}
  table {{ width:100%; border-collapse:collapse; font-size:13px; }}
  th, td {{ text-align:left; padding:8px 6px; border-bottom:1px solid var(--line); vertical-align:top; }}
  th {{ color:var(--muted); font-weight:600; }}
  code {{ font-size:12px; color:#c9d7ea; }}
  .muted {{ color:var(--muted); }}
  a {{ color:var(--ok); }}
  p {{ margin:0 0 8px; line-height:1.5; }}
  p.lead {{ font-size:16px; line-height:1.55; margin:8px 0 0; }}
  ul.plain {{ margin:8px 0 0 18px; padding:0; color:#c9d7ea; }}
  ul.plain li {{ margin:6px 0; }}
  .mapbox {{ background:#0f1419; border:1px solid var(--line); border-radius:10px; padding:12px 14px; font-size:14px; line-height:1.55; }}
  .mapbox b {{ color:var(--text); }}
  .note {{ margin-top:10px; padding:10px 12px; border-radius:8px; background:#0f1419; border:1px solid var(--line); font-size:13px; line-height:1.45; color:#c9d7ea; }}
  .note b {{ color:var(--text); }}
  .warnbox {{ margin-top:12px; padding:12px 14px; border-radius:8px; border:1px solid #f5a62355; background:rgba(245,166,35,0.08); font-size:14px; line-height:1.5; }}
  .okbox {{ margin-top:12px; padding:12px 14px; border-radius:8px; border:1px solid #3ddc9755; background:rgba(61,220,151,0.07); font-size:14px; line-height:1.5; }}
</style>
</head>
<body>
  <h1>Is the bot finding anything useful?</h1>
  <div class="sub">Refreshes every 10s · {s['now']} · <a href="/api">tech details</a></div>

  <div class="card">
    <h2>In one minute</h2>
    <div class="mapbox">
      The bot invents <b>strategy ideas</b>, tries them on <b>old prices with fake ${BOOK_USD:.0f}</b>,
      and only keeps ideas that pass a <b>strict exam</b>.<br/><br/>
      <b>Nothing here is live money.</b> A high line on a chart is not “we earned $X”.<br/><br/>
      Real success later means: run an idea forward for <b>{PAPER_MIN_DAYS} days</b> on paper and
      end with about <b>${paper_end_min:.0f}+</b> (started from ${BOOK_USD:.0f}), with enough trades and no big crash.
    </div>
  </div>

  {activity_html(s)}

  {last_cycle_html(s)}

  <div class="card">
    <h2>3. The last {rs.get('n',0)} finished searches</h2>
    <p class="lead">{story['trend_line']}</p>
    <p>{story['promo_line']}</p>
    <div class="grid4" style="margin-top:12px">
      <div class="stat">
        <div class="k">Searches finished</div>
        <div class="v">{rs.get('n',0)}</div>
        <div class="hint">shown in the charts</div>
      </div>
      <div class="stat">
        <div class="k">Enough trades?</div>
        <div class="v" style="color:{'#3ddc97' if (n and good==n) else '#f5a623'}">{(100.0*good/max(n,1)):.0f}%</div>
        <div class="hint">need ≥{MIN_TRADES} · typical {trade_med:.0f}</div>
      </div>
      <div class="stat">
        <div class="k">History profitable</div>
        <div class="v">{(100.0*pos_pnl/max(n,1)):.0f}%</div>
        <div class="hint">paper P&L > $0 on old data</div>
      </div>
      <div class="stat">
        <div class="k">Passed exam / blocked</div>
        <div class="v">{rs.get('promoted',0)} / {kill_n}</div>
        <div class="hint">shortlist hits vs dead idea-families</div>
      </div>
    </div>

    <div style="margin-top:16px">
      <div class="note" style="margin-bottom:8px">
        <b>Chart 1 — pretend profit on history.</b>
        Starts from fake ${BOOK_USD:.0f}. Green line = must finish above $0 (what the engine requires in the lab).
        Big history profits are common and still <b>not</b> the 30-day forward win.
      </div>
      {money_chart}
    </div>

    <div style="margin-top:16px">
      <div class="note" style="margin-bottom:8px">
        <b>Chart 2 — computer ranking while hunting.</b>
        Blue = best idea each search. Purple = smoother trend. Orange = typical level.
        This is <b>not dollars</b> and not “capital + ROI”.
      </div>
      {fit_chart}
    </div>

    {trade_alert}

    <div class="warnbox">
      <b>Easy rule of thumb</b><br/>
      1) Prefer Chart 1 above the green $0 line (history not a loser).<br/>
      2) “Enough trades” above should stay near 100% (engine gate ≥{MIN_TRADES}).<br/>
      3) Ranking (Chart 2) only matters if trades and history P&L look sane.<br/>
      4) The real win is still later: 30 days forward paper ending near
      <b>${paper_end_min:.0f}+</b> (stretch ${paper_end_tgt:.0f}), not a pretty history chart.
    </div>
  </div>

  {learning_html(s)}

  {champions_html(s)}

  <div class="card">
    <h2>What counts as a win?</h2>
    <div class="okbox">
      <b>The only “first real success”</b> is a <b>{PAPER_MIN_DAYS}-day forward paper run</b>
      (after an idea is already on the shortlist): start ${BOOK_USD:.0f}, finish about
      <b>≥ ${paper_end_min:.0f}</b> (net +${PAPER_MIN_NET_PNL_USD:.0f}), take ≥ <b>{PAPER_MIN_TRADES}</b> trades,
      and don’t drop more than about <b>${BOOK_USD * PAPER_MAX_DRAWDOWN_HARD:.0f}</b> from the peak.
    </div>
    <table style="margin-top:12px">
      <tr><th>Stage</th><th>What the bot requires</th><th>In plain English</th></tr>
      <tr>
        <td><b>Lab hunt</b></td>
        <td class="muted">≥{MIN_TRADES} trades, history profit > $0, drawdown ≤ 20%, pass strict exam</td>
        <td class="muted">“Interesting on old prices.” Not your money. Not go-live.</td>
      </tr>
      <tr>
        <td><b>Paper win</b></td>
        <td class="muted">≥{PAPER_MIN_DAYS} days forward, ≥{PAPER_MIN_TRADES} trades, end ≥ ${paper_end_min:.0f}, DD ≤ {int(PAPER_MAX_DRAWDOWN_HARD*100)}%</td>
        <td class="muted">First real success. Still not live SOL.</td>
      </tr>
      <tr>
        <td><b>Live micro</b></td>
        <td class="muted">Start tiny (~$25), stop if down 20% or 5 losing days in a row</td>
        <td class="muted">Only after paper win. Not unlocked yet.</td>
      </tr>
    </table>
    <div class="note" style="margin-top:10px">
      <b>Not a win:</b> a high ranking number, a big history profit on Chart 1, a high win-rate with tiny dollars,
      fewer than {MIN_TRADES} trades, one lucky exam pass with no forward paper, or six copies of the same idea.
    </div>
  </div>

  <div class="card">
    <h2>Words you’ll see</h2>
    <table>
      <tr><td><b>Searching</b></td><td class="muted">The Mac Mini is inventing and grading ideas.</td></tr>
      <tr><td><b>Pretend / paper trades</b></td><td class="muted">Fake buy/sell on old prices. No real SOL moves.</td></tr>
      <tr><td><b>Ranking / search score</b></td><td class="muted">How much the computer liked an idea while hunting. Sorting tool only.</td></tr>
      <tr><td><b>Strict exam</b></td><td class="muted">Extra hardness checks after the hunt. Fail = idea dies.</td></tr>
      <tr><td><b>Shortlist</b></td><td class="muted">Ideas that passed the lab exam. Candidates only.</td></tr>
      <tr><td><b>Blocked families</b></td><td class="muted">Known bad idea neighborhoods the bot tries not to retest forever.</td></tr>
    </table>
  </div>
</body>
</html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):  # noqa: A003
        return

    def do_GET(self):
        if self.path.startswith("/api"):
            s = collect_status()
            # compact API
            out = {
                "now": s["now"],
                "process": s["process"],
                "activity": s["activity"],
                "runs_total": s["runs_total"],
                "runs_broad": s["runs_broad"],
                "recent_sum": s["recent_sum"],
                "trend": {
                    "direction": s["trend"].get("direction"),
                    "why": s["trend"].get("why"),
                    "detail": s["trend"].get("detail"),
                },
                "trials_total": s["trials_total"],
                "n_champions": len((s.get("champions") or {}).get("champions") or []),
                "last_run": {
                    "fitness": (s.get("last_run") or {}).get("best_fitness"),
                    "trades": run_trades(s["last_run"]) if s.get("last_run") else None,
                    "logic": run_logic(s["last_run"]) if s.get("last_run") else None,
                    "n_promoted": (s.get("last_run") or {}).get("n_promoted"),
                    "mode": (s.get("last_run") or {}).get("mode"),
                },
                "questions": s["questions"],
                "success_criteria": criteria_public_dict(),
            }
            body = json.dumps(out, default=str).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path not in ("/", "/index.html", ""):
            self.send_response(404)
            self.end_headers()
            return

        body = html_page(collect_status()).encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    p = argparse.ArgumentParser(description="Trading bot dashboard")
    p.add_argument("--port", type=int, default=PORT)
    p.add_argument("--no-open", action="store_true")
    p.add_argument("--lan", action="store_true", help="Bind to 0.0.0.0 instead of 127.0.0.1 to allow local network access")
    args = p.parse_args()

    host = "0.0.0.0" if args.lan else "127.0.0.1"
    httpd = ThreadingHTTPServer((host, args.port), Handler)
    url = f"http://{'127.0.0.1' if not args.lan else 'localhost'}:{args.port}"
    print(f"Dashboard: {url}")
    if args.lan:
        print(f"LAN Access enabled. Look up your local IP (e.g., 192.168.x.x) and use http://<your-ip>:{args.port} from other devices on your WiFi.")
        
    if not args.no_open and not args.lan:
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped")


if __name__ == "__main__":
    main()
