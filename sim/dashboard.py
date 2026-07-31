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

SIM = Path(__file__).resolve().parent
LOGS = SIM / "logs"
EVO = SIM / "evolution"
PORT = 8765

# Health threshold for trade sample (matches selection min)
MIN_TRADES = 30


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
    fits = [float(r.get("best_fitness") or 0) for r in runs]
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
            float(r.get("best_fitness") or 0)
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


def _y_of(v: float, mn: float, mx: float, mt: float, ph: float) -> float:
    return mt + ph - (v - mn) / (mx - mn) * ph


def fitness_chart(vals: list[float], w: int = 720, h: int = 220) -> str:
    """Easy chart: one score per finished cycle, left=older, right=newer."""
    if len(vals) < 2:
        return '<div class="muted">Not enough finished cycles yet to draw a score chart.</div>'

    ordered = sorted(vals)
    p95 = ordered[int(0.95 * (len(ordered) - 1))]
    p05 = ordered[int(0.05 * (len(ordered) - 1))]
    clipped = [min(max(v, p05), p95) for v in vals]
    mn, mx = min(clipped), max(clipped)
    if mx == mn:
        mx = mn + 1.0

    med = statistics.median(clipped)
    # short moving average for trend eye-guide
    win = max(3, min(7, len(clipped) // 5 or 3))
    ma = []
    for i in range(len(clipped)):
        lo = max(0, i - win + 1)
        ma.append(sum(clipped[lo : i + 1]) / (i - lo + 1))

    ml, mr, mt, mb = 56, 18, 36, 46
    pw, ph = w - ml - mr, h - mt - mb

    def xy(i, v):
        x = ml + i * pw / (len(clipped) - 1)
        y = _y_of(v, mn, mx, mt, ph)
        return x, y

    pts = " ".join(f"{xy(i,v)[0]:.1f},{xy(i,v)[1]:.1f}" for i, v in enumerate(clipped))
    ma_pts = " ".join(f"{xy(i,v)[0]:.1f},{xy(i,v)[1]:.1f}" for i, v in enumerate(ma))

    # Grid + y ticks
    y_ticks = [mn + i * (mx - mn) / 4 for i in range(5)]
    grid = []
    for tv in y_ticks:
        y = _y_of(tv, mn, mx, mt, ph)
        grid.append(
            f'<line x1="{ml}" y1="{y:.1f}" x2="{w-mr}" y2="{y:.1f}" stroke="#1a2332" stroke-width="1"/>'
        )
        grid.append(
            f'<text x="{ml-8}" y="{y+4:.1f}" fill="#8b9bb4" font-size="11" text-anchor="end">{tv:.0f}</text>'
        )

    # Median line
    my = _y_of(med, mn, mx, mt, ph)
    med_line = (
        f'<line x1="{ml}" y1="{my:.1f}" x2="{w-mr}" y2="{my:.1f}" '
        f'stroke="#f5a623" stroke-width="1.5" stroke-dasharray="5,3"/>'
        f'<text x="{w-mr}" y="{my-5:.1f}" fill="#f5a623" font-size="11" text-anchor="end">'
        f'typical (median) {med:.0f}</text>'
    )

    # Zero if relevant
    zero = ""
    if mn < 0 < mx:
        zy = _y_of(0, mn, mx, mt, ph)
        zero = (
            f'<line x1="{ml}" y1="{zy:.1f}" x2="{w-mr}" y2="{zy:.1f}" '
            f'stroke="#e85d5d" stroke-width="1" stroke-dasharray="3,3"/>'
            f'<text x="{ml+4}" y="{zy-4:.1f}" fill="#e85d5d" font-size="10">break-even score</text>'
        )

    # End dots + labels
    x0, y0 = xy(0, clipped[0])
    x1, y1 = xy(len(clipped) - 1, clipped[-1])
    ends = (
        f'<circle cx="{x0:.1f}" cy="{y0:.1f}" r="3" fill="#8b9bb4"/>'
        f'<circle cx="{x1:.1f}" cy="{y1:.1f}" r="3.5" fill="#3ddc97"/>'
        f'<text x="{x1:.1f}" y="{y1-8:.1f}" fill="#3ddc97" font-size="11" text-anchor="end">'
        f'latest {clipped[-1]:.0f}</text>'
    )

    # X labels
    xlabels = (
        f'<text x="{ml}" y="{h-14}" fill="#8b9bb4" font-size="11">older cycles</text>'
        f'<text x="{(ml+w-mr)/2:.0f}" y="{h-14}" fill="#8b9bb4" font-size="11" text-anchor="middle">'
        f'each point = one finished search cycle</text>'
        f'<text x="{w-mr}" y="{h-14}" fill="#8b9bb4" font-size="11" text-anchor="end">newer</text>'
    )

    return f"""
    <svg viewBox="0 0 {w} {h}" width="100%" height="{h}" style="background:#0f1419;border-radius:8px">
      <text x="{ml}" y="22" fill="#e7eef7" font-size="14" font-weight="600">Search score over time</text>
      <text x="{w-mr}" y="22" fill="#8b9bb4" font-size="11" text-anchor="end">higher can mean better ranked, not proven profit</text>
      {''.join(grid)}
      {zero}
      {med_line}
      <polyline fill="none" stroke="#2a9d6a" stroke-width="2" opacity="0.55" points="{ma_pts}"/>
      <polyline fill="none" stroke="#3ddc97" stroke-width="2.5" points="{pts}"/>
      {ends}
      {xlabels}
      <text x="{ml}" y="{h-2}" fill="#6b7c94" font-size="10">Green line = each cycle winner score. Dimmer green = short average. Orange dashed = typical level.</text>
    </svg>"""


def trades_chart(vals: list[int], w: int = 720, h: int = 200) -> str:
    """Bar chart: how often the winner traded enough."""
    if len(vals) < 2:
        return '<div class="muted">Not enough finished cycles yet to draw a trade chart.</div>'

    ml, mr, mt, mb = 56, 18, 36, 46
    pw, ph = w - ml - mr, h - mt - mb
    mx = max(max(vals), MIN_TRADES, 1)
    # Cap display height a bit if one bar is huge
    mx = max(mx, int(statistics.median(vals) * 2.5) if vals else mx)
    n = len(vals)
    bw = max(2.0, pw / n * 0.7)

    good = sum(1 for v in vals if v >= MIN_TRADES)
    bad = n - good
    med = statistics.median(vals)

    parts = [
        f'<text x="{ml}" y="22" fill="#e7eef7" font-size="14" font-weight="600">How many trades each cycle winner took</text>',
        f'<text x="{w-mr}" y="22" fill="#8b9bb4" font-size="11" text-anchor="end">green = enough sample · red = too thin</text>',
    ]
    # Grid
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
        f'<line x1="{ml}" y1="{y30:.1f}" x2="{w-mr}" y2="{y30:.1f}" stroke="#f5a623" stroke-width="1.5" stroke-dasharray="5,3"/>'
    )
    parts.append(
        f'<text x="{w-mr}" y="{y30-5:.1f}" fill="#f5a623" font-size="11" text-anchor="end">'
        f'minimum useful sample ({MIN_TRADES})</text>'
    )

    for i, v in enumerate(vals):
        x = ml + i * pw / max(n - 1, 1) - bw / 2
        bh = min(ph, (min(v, mx) / mx) * ph)
        y = mt + ph - bh
        color = "#3ddc97" if v >= MIN_TRADES else "#e85d5d"
        parts.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{bh:.1f}" fill="{color}" opacity="0.88" rx="1"/>'
        )

    parts.append(
        f'<text x="{ml}" y="{h-14}" fill="#8b9bb4" font-size="11">older</text>'
        f'<text x="{(ml+w-mr)/2:.0f}" y="{h-14}" fill="#8b9bb4" font-size="11" text-anchor="middle">'
        f'each bar = trades by that cycle\'s best genome</text>'
        f'<text x="{w-mr}" y="{h-14}" fill="#8b9bb4" font-size="11" text-anchor="end">newer</text>'
    )
    parts.append(
        f'<text x="{ml}" y="{h-2}" fill="#6b7c94" font-size="10">'
        f'{good} of {n} bars clear the {MIN_TRADES} line · typical trades {med:.0f} · '
        f'{bad} still too thin to trust</text>'
    )
    return (
        f'<svg viewBox="0 0 {w} {h}" width="100%" height="{h}" '
        f'style="background:#0f1419;border-radius:8px">{"".join(parts)}</svg>'
    )


def chart_commentary(fit_vals: list[float], trade_vals: list[int], trend: dict) -> dict:
    """Plain English under the plots."""
    n = len(fit_vals)
    if n < 3:
        return {
            "fitness": "Not enough cycles yet. Come back after several finished search rounds.",
            "trades": "Trade chart needs a few finished cycles before it is meaningful.",
            "overall": trend.get("why") or "Still collecting data.",
        }

    last3 = fit_vals[-3:]
    first3 = fit_vals[:3]
    last_med = statistics.median(fit_vals[-max(5, n // 3) :])
    first_med = statistics.median(fit_vals[: max(5, n // 3)])
    if last_med > first_med * 1.08:
        fit_c = (
            f"Over the last {n} cycles, the middle of the score line is higher than earlier "
            f"({first_med:.0f} then {last_med:.0f}). That can mean better ranking, but it is still only a search score."
        )
    elif last_med < first_med * 0.92:
        fit_c = (
            f"The middle of the score line is lower than earlier ({first_med:.0f} then {last_med:.0f}). "
            f"Either the search got stricter, or winners look less attractive on holdout data."
        )
    else:
        fit_c = (
            f"The green line is bouncing around a similar middle ({last_med:.0f}). "
            f"That usually means the engine is exploring, not clearly inventing a stronger edge."
        )

    # Spike warning
    if max(fit_vals) > statistics.median(fit_vals) * 2.5:
        fit_c += " One or more tall spikes are outliers. Judge the orange median line, not the spikes."

    good = sum(1 for t in trade_vals if t >= MIN_TRADES)
    pct = 100.0 * good / max(len(trade_vals), 1)
    tmed = statistics.median(trade_vals) if trade_vals else 0
    if pct >= 90 and tmed >= MIN_TRADES:
        trade_c = (
            f"Good news on sample size: {good}/{len(trade_vals)} winners traded at least {MIN_TRADES} times "
            f"(typical {tmed:.0f}). These are thick enough to discuss, not automatically good enough to trade live."
        )
    elif pct >= 50:
        trade_c = (
            f"Mixed: {good}/{len(trade_vals)} clear the {MIN_TRADES} trade line (typical {tmed:.0f}). "
            f"Ignore red bars when reading fitness."
        )
    else:
        trade_c = (
            f"Worrying: only {good}/{len(trade_vals)} winners have {MIN_TRADES}+ trades. "
            f"High scores on red bars are usually luck."
        )

    overall = trend.get("why") or ""
    detail = trend.get("detail") or ""
    if detail:
        overall = f"{overall} {detail}".strip()

    # Last finished context
    if trade_vals:
        last_t = trade_vals[-1]
        last_f = fit_vals[-1]
        if last_t < MIN_TRADES:
            overall += f" Latest cycle scored {last_f:.0f} on only {last_t} trades: treat as noise."
        else:
            overall += f" Latest cycle: score {last_f:.0f} with {last_t} trades."

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

    # Questionable metrics to surface
    questions = []
    if last_run and run_trades(last_run) < MIN_TRADES and float(last_run.get("best_fitness") or 0) > 500:
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
        fit = float(last.get("best_fitness") or 0)
        promo = last.get("n_promoted")
        logic = (last.get("best_genome") or {}).get("entry_logic", "?")
        if trades < MIN_TRADES:
            last_line = (
                f"Last finished search picked an idea that only made <b>{trades}</b> paper trades. "
                f"That is too few to trust, even if the score ({fit:.0f}) looks big."
            )
        elif promo == 0:
            last_line = (
                f"Last finished search's best idea did <b>{trades}</b> paper trades "
                f"(type <b>{logic}</b>, score {fit:.0f}), then the strict exam "
                f"<b>rejected it</b> / the shortlist stayed empty."
            )
        else:
            last_line = (
                f"Last finished search found <b>{promo}</b> idea(s) that passed the strict exam. "
                f"Best idea in that search still made only paper profits on history "
                f"({trades} trades, score {fit:.0f}, type {logic}). Not live money."
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
    fit = float(r.get("best_fitness") or 0)
    pnl = float(bt.get("total_pnl") or 0)
    win = float(bt.get("win_rate") or 0)
    logic = g.get("entry_logic", "?")
    n_prom = r.get("n_promoted")
    gens = r.get("generations_run")
    trade_ok = trades >= MIN_TRADES
    trade_color = "#3ddc97" if trade_ok else "#e85d5d"

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
        <div class="k">Paper trades by best idea</div>
        <div class="v" style="color:{trade_color}">{trades}</div>
        <div class="hint">{trust}</div>
      </div>
      <div class="stat">
        <div class="k">Search score (not $)</div>
        <div class="v">{fit:.0f}</div>
        <div class="hint">only for ranking ideas while searching</div>
      </div>
      <div class="stat">
        <div class="k">Paper win rate / P&L</div>
        <div class="v">{win*100:.0f}% / ${pnl:.2f}</div>
        <div class="hint">on fake $100 history money</div>
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


def champions_html(s: dict) -> str:
    champs = (s.get("champions") or {}).get("champions") or []
    last_p = s.get("last_promoted")
    if not champs and not (last_p and last_p.get("all_passed")):
        return """
  <div class="card">
    <h2>4. Shortlist (ideas that passed the strict exam)</h2>
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
            f"<td>{float(last_p.get('score') or 0):.0f}</td>"
            f"<td class='muted'><code>{gid[:36]}</code></td></tr>"
        )
    for c in champs[:6]:
        g = c.get("genome") or {}
        gid = c.get("genome_id") or ""
        if gid in seen:
            continue
        rows.append(
            f"<tr><td>On shortlist</td><td><b>{g.get('entry_logic','?')}</b></td>"
            f"<td>{float(c.get('score') or 0):.0f}</td>"
            f"<td class='muted'><code>{gid[:36]}</code></td></tr>"
        )

    return f"""
  <div class="card">
    <h2>4. Shortlist <span class="section-hint">still paper only — not live trading</span></h2>
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
    fit_vals = [float(r.get("best_fitness") or 0) for r in recent]
    trade_vals = [run_trades(r) for r in recent]
    fit_chart = fitness_chart(fit_vals)
    trade_chart = trades_chart(trade_vals)
    story = story_summary(s)
    rs = s.get("recent_sum") or {}
    kill_n = story.get("kill_n", 0)

    # Even simpler chart captions
    n = len(trade_vals)
    good = sum(1 for t in trade_vals if t >= MIN_TRADES)
    if n >= 3:
        score_cap = (
            f"Each green dotted path position is one finished search job. "
            f"Higher = the computer liked that job's best idea more while searching. "
            f"It is <b>not</b> guaranteed profit."
        )
        trade_cap = (
            f"Each bar is: how many paper trades the best idea of that search took. "
            f"Green bars cleared {MIN_TRADES}+ trades ({good}/{n}). Red bars are too thin to trust."
        )
    else:
        score_cap = "Not enough finished searches yet for a useful chart."
        trade_cap = "Not enough finished searches yet for a useful chart."

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<meta http-equiv="refresh" content="10"/>
<title>Bot search status</title>
<style>
  :root {{ --bg:#0b0f14; --card:#151b23; --text:#e7eef7; --muted:#8b9bb4; --ok:#3ddc97; --bad:#e85d5d; --warn:#f5a623; --line:#243041; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
         background:var(--bg); color:var(--text); padding:20px; max-width:920px; margin-left:auto; margin-right:auto; }}
  h1 {{ margin:0; font-size:24px; }}
  h2 {{ margin:0 0 10px; font-size:15px; color:var(--text); letter-spacing:0; text-transform:none; font-weight:700; }}
  .sub {{ color:var(--muted); font-size:13px; margin:6px 0 16px; }}
  .card {{ background:var(--card); border:1px solid var(--line); border-radius:12px; padding:16px 18px; margin-bottom:14px; }}
  .badge {{ display:inline-block; padding:3px 9px; border-radius:999px; font-size:11px; font-weight:700; }}
  .badge.ok {{ background:#1a2a22; color:var(--ok); }}
  .badge.bad {{ background:#2a1a22; color:var(--bad); }}
  .badge.warn {{ background:#2a2418; color:var(--warn); }}
  .grid4 {{ display:grid; grid-template-columns:repeat(4,1fr); gap:10px; }}
  .grid2 {{ display:grid; grid-template-columns:1fr 1fr; gap:10px; }}
  @media (max-width:800px) {{ .grid4,.grid2 {{ grid-template-columns:1fr; }} }}
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
  .flow {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size:12px; color:#9fb2cc; white-space:pre-wrap; background:#0b1016; border-radius:8px; padding:10px 12px; margin-top:10px; }}
  .note {{ margin-top:10px; padding:10px 12px; border-radius:8px; background:#0f1419; border:1px solid var(--line); font-size:13px; line-height:1.45; color:#c9d7ea; }}
  .note b {{ color:var(--text); }}
  .warnbox {{ margin-top:12px; padding:12px 14px; border-radius:8px; border:1px solid #f5a62355; background:rgba(245,166,35,0.08); font-size:14px; line-height:1.5; }}
</style>
</head>
<body>
  <h1>Is the bot finding anything useful?</h1>
  <div class="sub">Auto-refresh every 10s · {s['now']} · <a href="/api">tech JSON</a></div>

  <div class="card">
    <h2>40-second map</h2>
    <div class="mapbox">
      <b>Genome</b> = one strategy idea (a rulebook).<br/>
      <b>Trade</b> = one simulated buy/sell while testing that idea on old price data.<br/>
      <b>Generation</b> = one breeding round (keep good ideas, mix them, try tweaks).<br/>
      <b>Cycle</b> = one full search job from scratch → pick winners → run a strict exam.<br/>
      <b>Score / fitness</b> = how much the computer liked an idea <i>while searching</i>. Not real profit.<br/>
      <b>Strict exam / funnel</b> = harder checks after the search. Only then can an idea join the shortlist.
      <div class="flow">cycle
  └─ many generations
       └─ many genomes (ideas)
            └─ each idea → many paper trades → a score
  └─ best few ideas → strict exam
       └─ pass → shortlist (still not live money)</div>
    </div>
  </div>

  {activity_html(s)}

  {last_cycle_html(s)}

  <div class="card">
    <h2>3. Across the last {rs.get('n',0)} finished searches</h2>
    <p class="lead">{story['trend_line']}</p>
    <p>{story['promo_line']}</p>
    <div class="grid4" style="margin-top:12px">
      <div class="stat">
        <div class="k">Search jobs finished</div>
        <div class="v">{rs.get('n',0)}</div>
        <div class="hint">shown in the charts below</div>
      </div>
      <div class="stat">
        <div class="k">Winners with enough trades</div>
        <div class="v">{rs.get('pct_n30',0):.0f}%</div>
        <div class="hint">need ≥{MIN_TRADES} paper trades</div>
      </div>
      <div class="stat">
        <div class="k">Typical paper trades</div>
        <div class="v">{rs.get('trade_med',0):.0f}</div>
        <div class="hint">middle value of search winners</div>
      </div>
      <div class="stat">
        <div class="k">Shortlist hits / blocked families</div>
        <div class="v">{rs.get('promoted',0)} / {kill_n}</div>
        <div class="hint">passes vs idea-families we refuse to retry</div>
      </div>
    </div>

    <div style="margin-top:16px">
      <div class="note" style="margin-bottom:8px"><b>Chart A — search scores:</b> {score_cap}</div>
      {fit_chart}
    </div>
    <div style="margin-top:16px">
      <div class="note" style="margin-bottom:8px"><b>Chart B — paper trades:</b> {trade_cap}</div>
      {trade_chart}
    </div>

    <div class="warnbox">
      <b>Read this pair of charts together.</b>
      A high spike on Chart A means nothing if Chart B is red for the same period.
      Prefer many green bars and a calm score line over huge lonely spikes.
    </div>
  </div>

  {champions_html(s)}

  <div class="card">
    <h2>Words you'll see</h2>
    <table>
      <tr><td><b>SEARCHING</b></td><td class="muted">The Mac Mini is still inventing and grading strategy ideas.</td></tr>
      <tr><td><b>Paper trades</b></td><td class="muted">Fake trades on historical candles. No real SOL moves.</td></tr>
      <tr><td><b>Search score</b></td><td class="muted">Computer ranking while hunting. Useful for sorting ideas, useless alone for deciding go-live.</td></tr>
      <tr><td><b>Strict exam</b></td><td class="muted">Extra tests (later data, fee stress, chapter tests…). Fail = idea dies, and that family can enter the blocked list.</td></tr>
      <tr><td><b>Shortlist</b></td><td class="muted">Ideas that passed the exam. Candidates for later paper trading — still not your capital.</td></tr>
      <tr><td><b>Blocked families</b></td><td class="muted">Known failed idea neighborhoods. The bot tries not to retest them forever.</td></tr>
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
    args = p.parse_args()

    httpd = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}"
    print(f"Dashboard: {url}")
    if not args.no_open:
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped")


if __name__ == "__main__":
    main()
