#!/usr/bin/env python3
"""
sentinel.py — independent watchdog for the whole autopilot.

Every prior failure (DB corruption, dead Hermes scheduler, dead data
refresher — 8 days unnoticed) was SILENT. The sentinel is a stdlib-only,
dependency-free check of every subsystem, run by launchd every 30 minutes.
On a NEW problem it posts a macOS notification (and logs); on recovery it
notifies once too. It depends on nothing it monitors.
"""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

SIM = Path(__file__).resolve().parent
DATA = SIM / "data"
HEALTH = DATA / "health.json"


def _age_h(ts_ms) -> float:
    return (time.time() - ts_ms / 1000.0) / 3600.0


def _db_max(db: str, q: str):
    conn = sqlite3.connect(str(DATA / db), timeout=5)
    try:
        return conn.execute(q).fetchone()[0]
    finally:
        conn.close()


def run_checks() -> dict:
    alerts, ok = [], []

    def check(name, fn, alert_msg):
        try:
            good, detail = fn()
        except Exception as e:
            good, detail = False, f"check error: {e}"
        (ok if good else alerts).append(f"{name}: {detail}" if not good
                                        else f"{name} ({detail})")
        if not good:
            alerts[-1] = f"{name}: {alert_msg} [{detail}]"

    check("runner", lambda: (
        bool(subprocess.run(["pgrep", "-f", "run_broad_evolution.py"],
                            capture_output=True, text=True).stdout.strip()),
        "pid ok"), "search engine not running")
    check("candles", lambda: (
        (a := _age_h(_db_max("historical_features_1h.db",
                             "SELECT MAX(ts) FROM features_1h"))) < 6,
        f"{a:.1f}h old"), "market data stale — refresher dead?")
    check("derivatives", lambda: (
        (a := _age_h(_db_max("derivatives.db",
                             "SELECT MAX(ts) FROM derivs"))) < 4,
        f"{a:.1f}h old"), "derivatives collection stalled")
    check("dashboard", lambda: (
        urllib.request.urlopen("http://127.0.0.1:8770/",
                               timeout=6).status == 200,
        "http 200"), "dashboard down on :8770")

    def _paper():
        p = json.loads((DATA / "paper_status.json").read_text())
        a = (time.time() - p.get("updated_ts", 0)) / 3600.0
        return a < 2, f"updated {a:.1f}h ago"
    check("paper", _paper, "paper trader not updating")

    check("probe", lambda: (
        (a := (time.time() - _db_max("execution_probe.db",
                                     "SELECT MAX(ts) FROM quotes")) / 3600.0) < 4,
        f"{a:.1f}h old"), "execution probe stalled")

    def _disk():
        st = os.statvfs(str(SIM))
        free_gb = st.f_bavail * st.f_frsize / 1e9
        return free_gb > 5, f"{free_gb:.0f}GB free"
    check("disk", _disk, "low disk space")

    return {"ts": time.time(),
            "healthy": not alerts, "alerts": alerts, "ok": ok}


def notify(title: str, msg: str) -> None:
    try:
        subprocess.run(["osascript", "-e",
                        f'display notification "{msg[:180]}" '
                        f'with title "{title}" sound name "Basso"'],
                       timeout=10, capture_output=True)
    except Exception:
        pass


def main() -> int:
    prev = {}
    try:
        prev = json.loads(HEALTH.read_text())
    except Exception:
        pass
    h = run_checks()
    try:
        tmp = HEALTH.with_suffix(".tmp")
        tmp.write_text(json.dumps(h, indent=2))
        tmp.replace(HEALTH)
    except Exception:
        pass

    prev_alerts = set(prev.get("alerts") or [])
    new_alerts = [a for a in h["alerts"] if a.split(":")[0] not in
                  {p.split(":")[0] for p in prev_alerts}]
    recovered = [p for p in prev_alerts if p.split(":")[0] not in
                 {a.split(":")[0] for a in h["alerts"]}]
    if new_alerts:
        notify("Trading bot ALERT", "; ".join(new_alerts)[:180])
    if recovered and not h["alerts"]:
        notify("Trading bot recovered", "all systems healthy again")
    line = "HEALTHY" if h["healthy"] else "ALERTS: " + " | ".join(h["alerts"])
    print(f"[{time.strftime('%Y-%m-%d %H:%M')}] sentinel: {line}")
    return 0 if h["healthy"] else 1


if __name__ == "__main__":
    sys.exit(main())
