#!/usr/bin/env python3
"""
daily_report_generator.py
Generates daily reports using local LLM (Ollama).
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
from cost_tracker.compute_meter import meter
from cost_tracker.cost_model import compute_cost_chf, monthly_estimate

REPORT_DIR = SIM_DIR / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def get_yesterday_stats() -> dict:
    """Pull yesterday's stats from SQLite."""
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    start_ts = datetime.strptime(yesterday, "%Y-%m-%d").timestamp()
    end_ts = start_ts + 86400

    stats = {
        "date": yesterday,
        "quotes": 0,
        "features": 0,
        "models_trained": 0,
        "predictions": 0,
        "trades": 0,
        "alerts": 0,
    }

    # Count quotes
    db_path = DATA_DIR / "raw_quotes.db"
    if db_path.exists():
        conn = sqlite3.connect(str(db_path))
        cursor = conn.execute("SELECT COUNT(*) FROM quotes WHERE ts BETWEEN ? AND ?", (start_ts, end_ts))
        stats["quotes"] = cursor.fetchone()[0]
        conn.close()

    # Count features
    db_path = DATA_DIR / "historical_features.db"
    if db_path.exists():
        conn = sqlite3.connect(str(db_path))
        cursor = conn.execute("SELECT COUNT(*) FROM features WHERE ts BETWEEN ? AND ?", (start_ts * 1000, end_ts * 1000))
        stats["features"] = cursor.fetchone()[0]
        conn.close()

    # Count models
    model_files = list(MODEL_DIR.glob("*.pt")) + list(MODEL_DIR.glob("*.zip"))
    stats["models_trained"] = len(model_files)

    # Count log entries
    log_files = list(LOG_DIR.glob("training_results_*.json"))
    stats["training_runs"] = len(log_files)

    return stats


def get_model_metrics() -> dict:
    """Load latest model metrics."""
    metrics = {}

    # Find latest training results
    result_files = sorted(LOG_DIR.glob("training_results_*.json"))
    if result_files:
        with open(result_files[-1]) as f:
            data = json.load(f)
            metrics["tft"] = {
                "directional_accuracy": data.get("metrics", {}).get("directional_accuracy", 0),
                "precision": data.get("metrics", {}).get("precision", 0),
                "recall": data.get("metrics", {}).get("recall", 0),
                "mse": data.get("metrics", {}).get("mse", 0),
            }

    # Find latest validation results
    val_files = sorted(LOG_DIR.glob("validation_*.json"))
    if val_files:
        with open(val_files[-1]) as f:
            data = json.load(f)
            metrics["validation"] = {
                "strategies_tested": len(data.get("strategies", {})),
                "best_strategy": None,
                "best_pnl": 0,
            }
            # Find best strategy
            best_pnl = float("-inf")
            best_name = None
            for name, res in data.get("strategies", {}).items():
                pnl = res.get("walk_forward", {}).get("avg_pnl", 0)
                if pnl > best_pnl:
                    best_pnl = pnl
                    best_name = name
            metrics["validation"]["best_strategy"] = best_name
            metrics["validation"]["best_pnl"] = best_pnl

    return metrics


def generate_with_ollama(prompt: str) -> str:
    """Generate report using local Ollama."""
    try:
        result = subprocess.run(
            ["ollama", "run", "llama3.1:8b", prompt],
            capture_output=True, text=True, timeout=120
        )
        return result.stdout.strip()
    except Exception as e:
        return f"Error generating report: {e}"


def main():
    stats = get_yesterday_stats()
    metrics = get_model_metrics()
    cost_data = compute_cost_chf(meter.get_summary())
    cost_data["monthly_estimate_chf"] = monthly_estimate(cost_data.get("total_chf", 0))

    # Build prompt for local LLM
    prompt = f"""Generate a concise daily trading bot report.

Date: {stats['date']}
Data collected: {stats['quotes']} quotes, {stats['features']} feature vectors
Models: {stats['models_trained']} saved models, {stats['training_runs']} training runs

Model Performance:
- TFT directional accuracy: {metrics.get('tft', {}).get('directional_accuracy', 0):.1%}
- TFT precision: {metrics.get('tft', {}).get('precision', 0):.1%}
- TFT recall: {metrics.get('tft', {}).get('recall', 0):.1%}

Validation:
- Strategies tested: {metrics.get('validation', {}).get('strategies_tested', 0)}
- Best strategy: {metrics.get('validation', {}).get('best_strategy', 'N/A')}
- Best P&L: {metrics.get('validation', {}).get('best_pnl', 0):.4f} USD

Compute Cost:
- Today: {cost_data.get('total_chf', 0):.4f} CHF
- Monthly estimate: {cost_data.get('monthly_estimate_chf', 0):.2f} CHF

Rules:
- Be direct, no fluff
- Report key metrics first
- Flag if no profitable strategies
- Include cost
- Max 10 lines
"""

    report_text = generate_with_ollama(prompt)

    # Save report
    report_file = REPORT_DIR / f"report_{stats['date']}.txt"
    with open(report_file, "w") as f:
        f.write(f"=== Daily Trading Bot Report ===\n")
        f.write(f"Date: {stats['date']}\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write(f"\n{report_text}\n")
        f.write(f"\n--- Raw Stats ---\n{json.dumps(stats, indent=2)}\n")
        f.write(f"\n--- Metrics ---\n{json.dumps(metrics, indent=2)}\n")
        f.write(f"\n--- Cost ---\n{json.dumps(cost_data, indent=2)}\n")

    print(report_text)
    print(f"\n[Report saved to {report_file}]")


if __name__ == "__main__":
    main()
