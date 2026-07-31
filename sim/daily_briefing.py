#!/usr/bin/env python3
"""
daily_briefing.py
Combines news briefing + trading bot evolution results + model performance.
"""

import json
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

SIM_DIR = Path.home() / ".hermes" / "trading-bot" / "sim"
sys.path.insert(0, str(SIM_DIR))

from config import LOG_DIR, MODEL_DIR, DATA_DIR


def get_evolution_results() -> dict:
    """Load latest evolution results."""
    pop_dir = SIM_DIR / "evolution" / "population"
    if not pop_dir.exists():
        return {"status": "no_data"}

    files = sorted(pop_dir.glob("evolution_*.json"))
    if not files:
        return {"status": "no_data"}

    with open(files[-1]) as f:
        data = json.load(f)

    return {
        "date": datetime.fromtimestamp(data["timestamp"]).strftime("%Y-%m-%d"),
        "best_fitness": data.get("best_fitness", 0),
        "generations": data.get("generations_run", 0),
        "trades": data.get("backtest", {}).get("total_trades", 0),
        "win_rate": data.get("backtest", {}).get("win_rate", 0),
        "pnl": data.get("backtest", {}).get("total_pnl", 0),
        "sharpe": data.get("backtest", {}).get("sharpe_ratio", 0),
        "strategy_id": data.get("best_genome", {}).get("genome_id", "unknown"),
    }


def get_model_status() -> dict:
    """Check which models exist and their age."""
    models = {}
    for name in ["tft", "classifier", "risk"]:
        path = MODEL_DIR / f"{name}_final.pt"
        if path.exists():
            age_days = (time.time() - path.stat().st_mtime) / 86400
            models[name] = {
                "exists": True,
                "age_days": round(age_days, 1),
                "size_mb": round(path.stat().st_size / 1e6, 2),
            }
        else:
            models[name] = {"exists": False}
    return models


def get_data_status() -> dict:
    """Check data freshness."""
    db_path = DATA_DIR / "historical_candles.db"
    if db_path.exists():
        conn = sqlite3.connect(str(db_path))
        cursor = conn.execute("SELECT MAX(ts) FROM candles")
        max_ts = cursor.fetchone()[0]
        conn.close()
        age_hours = (time.time() * 1000 - max_ts) / 3600000
        return {"candles": True, "age_hours": round(age_hours, 1)}
    return {"candles": False}


def generate_briefing() -> str:
    """Generate the daily briefing text."""
    now = datetime.now()
    today = now.strftime("%A, %d %B %Y")

    evolution = get_evolution_results()
    models = get_model_status()
    data = get_data_status()

    lines = [
        f"# Daily Briefing — {today}",
        "",
        "## Trading Bot Status",
        "",
    ]

    # Evolution
    if evolution.get("status") == "no_data":
        lines.append("- **Evolution:** No runs yet")
    else:
        lines.append(f"- **Evolution:** {evolution['generations']} generations")
        lines.append(f"  - Best fitness: {evolution['best_fitness']:.1f}")
        lines.append(f"  - Trades: {evolution['trades']} | Win rate: {evolution['win_rate']:.1%} | P&L: ${evolution['pnl']:.2f}")
        lines.append(f"  - Sharpe: {evolution['sharpe']:.1f}")

    lines.append("")

    # Models
    lines.append("### Models")
    for name, info in models.items():
        if info["exists"]:
            lines.append(f"- **{name.upper()}**: {info['age_days']}d old, {info['size_mb']}MB")
        else:
            lines.append(f"- **{name.upper()}**: not trained")

    lines.append("")

    # Data
    lines.append("### Data")
    if data["candles"]:
        lines.append(f"- Candles: {data['age_hours']}h old")
    else:
        lines.append("- Candles: not downloaded")

    lines.append("")

    # Action items
    lines.append("### Action Items")
    actions = []
    if not models["tft"]["exists"]:
        actions.append("Train TFT model")
    if not models["classifier"]["exists"]:
        actions.append("Train classifier")
    if evolution.get("win_rate", 0) < 0.55:
        actions.append("Evolution needs more time to find edge")
    if data.get("age_hours", 999) > 48:
        actions.append("Refresh historical data")

    if actions:
        for a in actions:
            lines.append(f"- {a}")
    else:
        lines.append("- All systems nominal")

    lines.append("")
    lines.append("---")
    lines.append(f"_Generated at {now.strftime('%H:%M')} UTC_")

    return "\n".join(lines)


def main():
    briefing = generate_briefing()

    # Save
    output_path = Path.home() / "hermes-workspace" / "daily_briefing.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(briefing)

    print(briefing)
    print(f"\n[Saved to {output_path}]")


if __name__ == "__main__":
    main()
