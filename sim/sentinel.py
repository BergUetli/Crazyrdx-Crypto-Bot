#!/usr/bin/env python3
"""
sentinel.py — independent watchdog for the whole autopilot.

Every prior failure (DB corruption, dead Hermes scheduler, dead data
refresher — 8 days unnoticed) was SILENT. The sentinel is a stdlib-only,
dependency-free check of every subsystem, run by launchd every 30 minutes.
On a NEW problem it posts a macOS notification (and logs); on recovery it
notifies once too. It depends on nothing it monitors.

Dashboard auto-remediation (added 2026-09-04 after a 14h wedge the
notify-only design left unfixed): after DASH_FAILS_TO_ACT consecutive
failed dashboard checks the sentinel captures forensics (ps state, native
stacks via `sample`, open files, system snapshot), THEN kills the wedged
pid — launchd KeepAlive respawns it — and only then re-probes to verify
recovery. Every action writes an incident report under
sim/logs/incidents/ for root cause analysis. The dashboard is stateless
and read-only, so an automatic restart cannot lose trading state; no
other subsystem is auto-remediated.
"""

from __future__ import annotations

import json
import os
import signal
import sqlite3
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

SIM = Path(__file__).resolve().parent
DATA = SIM / "data"
HEALTH = DATA / "health.json"
STATE = DATA / "sentinel_state.json"
INCIDENTS = SIM / "logs" / "incidents"

DASH_PORT = 8770
DASH_URL = f"http://127.0.0.1:{DASH_PORT}/"
DASH_FAILS_TO_ACT = 3          # consecutive failed checks before the fix runs
DASH_RECOVERY_WAIT_S = 150     # respawn + ~60s warm-up before first replies


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
        urllib.request.urlopen(DASH_URL, timeout=6).status == 200,
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


# ---------------------------------------------------------------- state ----

def load_state() -> dict:
    try:
        return json.loads(STATE.read_text())
    except Exception:
        return {}


def save_state(state: dict) -> None:
    try:
        tmp = STATE.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=2))
        tmp.replace(STATE)
    except Exception:
        pass


def update_dash_state(state: dict, dash_down: bool, now: float) -> bool:
    """Advance the consecutive-failure counter; return True when the fix
    must run NOW (before any retry of the check). Pure: mutates only
    `state`, touches nothing else."""
    if not dash_down:
        state["dash_fails"] = 0
        state.pop("dash_first_fail_ts", None)
        return False
    state["dash_fails"] = state.get("dash_fails", 0) + 1
    state.setdefault("dash_first_fail_ts", now)
    return state["dash_fails"] >= DASH_FAILS_TO_ACT


# ------------------------------------------------------------ forensics ----

def _run(cmd: list[str], timeout: int = 30) -> str:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=timeout)
        return (r.stdout or "") + (r.stderr or "")
    except Exception as e:
        return f"<{' '.join(cmd)} failed: {e}>\n"


def dash_pids() -> list[int]:
    pids: set[int] = set()
    for out in (_run(["lsof", "-nP", "-t", f"-iTCP:{DASH_PORT}",
                      "-sTCP:LISTEN"], 15),
                _run(["pgrep", "-f", "sim/dashboard.py"], 10)):
        for tok in out.split():
            if tok.isdigit():
                pids.add(int(tok))
    pids.discard(os.getpid())
    return sorted(pids)


def capture_forensics(pids: list[int], inc_dir: Path,
                      sample_secs: int = 5) -> None:
    """Everything the RCA needs, captured BEFORE the kill destroys it.
    Each artifact is independent; one failing must not stop the rest."""
    inc_dir.mkdir(parents=True, exist_ok=True)

    def w(name: str, text: str) -> None:
        try:
            (inc_dir / name).write_text(text)
        except Exception:
            pass

    for pid in pids:
        p = str(pid)
        # STAT column: U = uninterruptible wait (the 2026-09-03 wedge mode)
        w(f"ps_{p}.txt", _run(["ps", "-o",
                               "pid,ppid,%cpu,%mem,state,wchan,etime,"
                               "lstart,command", "-p", p]))
        w(f"threads_{p}.txt", _run(["ps", "-M", "-p", p]))
        w(f"lsof_{p}.txt", _run(["lsof", "-nP", "-p", p], 60))
        if sample_secs > 0:
            # native thread stacks — shows the exact blocked syscall
            w(f"sample_{p}.txt", _run(["sample", p, str(sample_secs)],
                                      sample_secs + 40))
    w("system.txt", "\n".join([
        "== uptime ==", _run(["uptime"], 10),
        "== vm_stat ==", _run(["vm_stat"], 10),
        "== df -h ==", _run(["df", "-h"], 10),
        "== top cpu ==", _run(["ps", "-Ao", "%cpu,pid,state,command",
                               "-r"], 10)[:4000]]))
    for src, dst in ((SIM / "logs" / "dashboard.log",
                      "dashboard_log_tail.txt"),
                     (DATA / "dashboard_heartbeat.json",
                      "dashboard_heartbeat.json"),
                     (HEALTH, "health_at_incident.json")):
        try:
            text = src.read_text(errors="replace")
            w(dst, text[-30000:])
        except Exception:
            pass


def _kill(pid: int) -> str:
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return "already gone"
    except Exception as e:
        return f"SIGTERM failed: {e}"
    for _ in range(10):
        time.sleep(1)
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return "exited on SIGTERM"
    try:
        os.kill(pid, signal.SIGKILL)
        return "needed SIGKILL"
    except ProcessLookupError:
        return "exited on SIGTERM (late)"
    except Exception as e:
        return f"SIGKILL failed: {e}"


def _wait_recovery(wait_s: int = DASH_RECOVERY_WAIT_S):
    t0 = time.time()
    while time.time() - t0 < wait_s:
        try:
            if urllib.request.urlopen(DASH_URL, timeout=6).status == 200:
                return round(time.time() - t0, 1)
        except Exception:
            pass
        time.sleep(5)
    return None


def write_report(inc_dir: Path, first_fail_ts, kill_results: dict,
                 recovered_s, fails: int = DASH_FAILS_TO_ACT) -> None:
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    ff = (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(first_fail_ts))
          if first_fail_ts else "unknown")
    ps_state = ""
    for f in sorted(inc_dir.glob("ps_*.txt")):
        ps_state += f"```\n{f.read_text()}```\n"
    lines = [
        f"# Dashboard incident {inc_dir.name}",
        "",
        "## Timeline",
        f"- First failed sentinel check: {ff}",
        f"- Consecutive failed checks before action: {fails}"
        " (30 min apart)",
        f"- Auto-remediation ran: {now}",
        f"- Kill results: {json.dumps(kill_results)}",
        f"- Recovery: " + (f"HTTP 200 after {recovered_s}s"
                           if recovered_s is not None else
                           "NOT RECOVERED within "
                           f"{DASH_RECOVERY_WAIT_S}s — needs a human"),
        "",
        "## Process state at capture (before kill)",
        ps_state or "(no ps capture)",
        "## Root cause analysis — how to read the artifacts",
        "- `ps_<pid>.txt` STAT column: U = uninterruptible wait (blocked"
        " in the kernel, usually disk or network I/O); the wchan column"
        " names the kernel wait channel.",
        "- `sample_<pid>.txt`: native stack traces of every thread —"
        " the deepest frames show the exact call the process is stuck in.",
        "- `lsof_<pid>.txt`: open files and sockets — look for the SQLite"
        " db or socket matching the stuck stack frame.",
        "- `dashboard_heartbeat.json`: last moment the process was alive"
        " (written every 30s), pins the wedge onset far tighter than the"
        " 30-min sentinel grid.",
        "- `system.txt` + `dashboard_log_tail.txt`: rule out machine-wide"
        " causes (load, disk full) and application errors.",
        "",
        "## RCA conclusion",
        "_To be filled in by the analysis session._",
    ]
    try:
        (inc_dir / "report.md").write_text("\n".join(lines))
    except Exception:
        pass


def remediate_dashboard(first_fail_ts) -> str:
    """Fix first, verify after: forensics -> kill -> (launchd respawns)
    -> recovery probe. Never re-probes the dashboard before acting."""
    inc_dir = INCIDENTS / f"incident_{time.strftime('%Y%m%d_%H%M%S')}"
    pids = dash_pids()
    capture_forensics(pids, inc_dir)
    kill_results = {str(p): _kill(p) for p in pids} or \
        {"none": "no dashboard pid found — launchd may already be cycling"}
    recovered_s = _wait_recovery()
    write_report(inc_dir, first_fail_ts, kill_results, recovered_s)
    if recovered_s is not None:
        notify("Trading bot: dashboard restarted",
               f"wedged {DASH_FAILS_TO_ACT} checks; auto-restarted, "
               f"back in {recovered_s}s. Forensics: {inc_dir.name}")
        verdict = f"recovered in {recovered_s}s"
    else:
        notify("Trading bot: dashboard NOT recovered",
               f"auto-restart failed to bring :{DASH_PORT} back — "
               f"see {inc_dir.name}")
        verdict = "NOT RECOVERED"
    return f"INCIDENT dashboard auto-restart -> {inc_dir.name} ({verdict})"


# ----------------------------------------------------------------- main ----

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

    incident_line = None
    state = load_state()
    dash_down = any(a.startswith("dashboard:") for a in h["alerts"])
    if update_dash_state(state, dash_down, h["ts"]):
        first_fail = state.get("dash_first_fail_ts")
        incident_line = remediate_dashboard(first_fail)
        state["dash_fails"] = 0
        state.pop("dash_first_fail_ts", None)
        state["last_incident"] = incident_line
    save_state(state)

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
    if incident_line:
        line += f" | {incident_line}"
    print(f"[{time.strftime('%Y-%m-%d %H:%M')}] sentinel: {line}")
    return 0 if h["healthy"] else 1


if __name__ == "__main__":
    sys.exit(main())
