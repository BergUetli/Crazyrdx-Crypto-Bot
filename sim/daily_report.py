#!/usr/bin/env python3
"""
daily_report.py
Autonomous daily report generator.
Uses model router to decide local vs cloud generation.
"""

import json
import sqlite3
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from cost_tracker.compute_meter import meter, track_cpu
from cost_tracker.cost_model import compute_cost_chf, cloud_token_cost, monthly_estimate

DB_PATH = Path(__file__).parent / "data" / "raw_quotes.db"
REPORT_DIR = Path(__file__).parent / "reports"
LOG_DIR = Path(__file__).parent / "logs"


def get_yesterday_stats() -> dict:
    """Pull yesterday's stats from SQLite."""
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    start_ts = datetime.strptime(yesterday, "%Y-%m-%d").timestamp()
    end_ts = start_ts + 86400

    if not DB_PATH.exists():
        return {
            "date": yesterday,
            "quotes": 0,
            "opportunities": 0,
            "viable": 0,
            "trades": 0,
            "total_pnl": 0.0,
            "max_spread_bps": 0.0,
            "note": "No data yet"
        }

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("SELECT COUNT(*) FROM quotes WHERE ts BETWEEN ? AND ?", (start_ts, end_ts))
    quotes = c.fetchone()[0]

    c.execute("SELECT COUNT(*) FROM opportunities WHERE ts BETWEEN ? AND ?", (start_ts, end_ts))
    opps = c.fetchone()[0]

    c.execute("SELECT COUNT(*) FROM opportunities WHERE ts BETWEEN ? AND ? AND viable=1", (start_ts, end_ts))
    viable = c.fetchone()[0]

    c.execute("SELECT COUNT(*), COALESCE(SUM(net_pnl_usd),0), COALESCE(MAX(spread_bps),0) FROM sim_trades WHERE ts BETWEEN ? AND ?", (start_ts, end_ts))
    row = c.fetchone()
    trades = row[0]
    total_pnl = row[1]
    max_spread = row[2]

    conn.close()

    return {
        "date": yesterday,
        "quotes": quotes,
        "opportunities": opps,
        "viable": viable,
        "trades": trades,
        "total_pnl": round(total_pnl, 4),
        "max_spread_bps": round(max_spread, 2),
    }


def route_model(task_type: str, complexity: str = "low") -> str:
    """Call the model router script."""
    router = Path.home() / ".hermes" / "scripts" / "model_router.sh"
    result = subprocess.run(
        ["bash", str(router), task_type, complexity],
        capture_output=True, text=True, timeout=10
    )
    return result.stdout.strip()


def generate_with_ollama(prompt: str) -> tuple[str, float]:
    """Generate report using local Ollama. Returns (text, cost_chf)."""
    with track_cpu("local:llama3.1:8b"):
        result = subprocess.run(
            ["ollama", "run", "llama3.1:8b", prompt],
            capture_output=True, text=True, timeout=120
        )
        text = result.stdout.strip()
        # Estimate tokens (rough: 1 token ~ 4 chars)
        tokens_in = len(prompt) // 4
        tokens_out = len(text) // 4
        cost = 0.0  # local = electricity only, tracked via compute_meter
        return text, cost


def generate_with_cloud(prompt: str, model: str) -> tuple[str, float]:
    """Generate report using cloud model via hermes chat."""
    with track_cpu(f"cloud:{model}"):
        result = subprocess.run(
            ["hermes", "chat", "-q", prompt, "--model", model],
            capture_output=True, text=True, timeout=120
        )
        text = result.stdout.strip()
        tokens_in = len(prompt) // 4
        tokens_out = len(text) // 4
        cost = cloud_token_cost(model, tokens_in, tokens_out)
        meter.record_tokens(f"cloud:{model}", tokens_in, tokens_out)
        return text, cost


def generate_report(stats: dict, cost_data: dict) -> str:
    """Generate the daily report. Route to local or cloud based on complexity."""
    prompt = f"""Generate a concise daily trading bot report.

Data: {json.dumps(stats)}
Cost: {json.dumps(cost_data)}

Rules:
- P&L first
- Flag if no viable opportunities
- Include compute cost in CHF
- Max 8 lines
- No fluff
"""

    # Route: daily_report = local model
    model = route_model("daily_report", "low")

    if "ollama" in model:
        text, token_cost = generate_with_ollama(prompt)
    else:
        text, token_cost = generate_with_cloud(prompt, model)

    return text, token_cost


def save_report(text: str, stats: dict, cost_data: dict):
    """Save report to file and log."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    date_str = stats["date"]
    report_file = REPORT_DIR / f"report_{date_str}.txt"

    with open(report_file, "w") as f:
        f.write(f"=== Daily Report {date_str} ===\n\n")
        f.write(text)
        f.write(f"\n\n--- Raw Stats ---\n{json.dumps(stats, indent=2)}")
        f.write(f"\n\n--- Cost Data ---\n{json.dumps(cost_data, indent=2)}")

    # Also log to daily summary
    summary_file = LOG_DIR / "daily_summary.jsonl"
    with open(summary_file, "a") as f:
        f.write(json.dumps({
            "date": date_str,
            "stats": stats,
            "cost": cost_data,
            "report_file": str(report_file),
            "timestamp": datetime.now().isoformat()
        }) + "\n")

    return report_file


def main():
    # Get stats
    stats = get_yesterday_stats()

    # Get compute cost from meter
    usage = meter.get_summary()
    cost_data = compute_cost_chf(usage)
    cost_data["monthly_estimate_chf"] = monthly_estimate(cost_data.get("total_chf", 0))

    # Generate report
    report_text, token_cost = generate_report(stats, cost_data)
    cost_data["token_cost_chf"] = token_cost
    current_total = cost_data.get("total_chf", 0.0)
    if isinstance(current_total, str):
        current_total = 0.0
    token_cost_float = token_cost if isinstance(token_cost, (int, float)) else 0.0
    new_total = current_total + token_cost_float
    cost_data["total_chf"] = round(new_total, 6)

    # Save
    report_file = save_report(report_text, stats, cost_data)

    # Output for cron delivery
    print(report_text)
    print(f"\n[Saved to {report_file}]")


if __name__ == "__main__":
    main()
