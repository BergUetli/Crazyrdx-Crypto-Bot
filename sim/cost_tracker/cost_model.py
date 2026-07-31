"""
cost_model.py
Converts compute usage to CHF.
"""

from typing import Dict

# Constants (Swiss residential electricity pricing)
M4_TDP_WATTS = 30.0          # Apple M4 under ML load
ELECTRICITY_RATE_CHF = 0.25  # CHF per kWh
CPU_KWH_PER_HOUR = M4_TDP_WATTS / 1000.0
GPU_KWH_PER_HOUR = M4_TDP_WATTS / 1000.0  # same TDP on Mac

# Cloud model pricing (per 1M tokens, from OpenRouter live data)
CLOUD_PRICING = {
    "openai/gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "anthropic/claude-sonnet-4": {"input": 3.00, "output": 15.00},
    "moonshotai/kimi-k2": {"input": 0.57, "output": 2.30},
    "moonshotai/kimi-k3": {"input": 3.00, "output": 15.00},
    "deepseek/deepseek-chat": {"input": 0.20, "output": 0.80},
}


def compute_cost_chf(usage: Dict[str, dict]) -> Dict[str, float]:
    """
    Convert compute usage summary to CHF costs.
    Returns dict with per-layer and total costs.
    """
    costs = {}
    total = 0.0

    for layer, u in usage.items():
        # Local compute cost (electricity)
        cpu_cost = (u["cpu_seconds"] / 3600.0) * CPU_KWH_PER_HOUR * ELECTRICITY_RATE_CHF
        gpu_cost = (u["gpu_seconds"] / 3600.0) * GPU_KWH_PER_HOUR * ELECTRICITY_RATE_CHF

        # Cloud token cost (if any)
        token_cost = 0.0
        # Note: token costs are tracked separately via model name in the layer key
        # e.g., "cloud:gpt-4o-mini" or "local:llama3.1:8b"

        layer_total = cpu_cost + gpu_cost
        costs[layer] = round(layer_total, 6)
        total += layer_total

    costs["total_chf"] = round(total, 6)
    return costs


def cloud_token_cost(model: str, tokens_in: int, tokens_out: int) -> float:
    """Calculate CHF cost for a cloud model call."""
    pricing = CLOUD_PRICING.get(model, {"input": 0.50, "output": 1.50})
    in_cost = (tokens_in / 1_000_000) * pricing["input"]
    out_cost = (tokens_out / 1_000_000) * pricing["output"]
    return round(in_cost + out_cost, 6)


def monthly_estimate(daily_cost_chf: float) -> float:
    """Estimate monthly cost from daily average."""
    return round(daily_cost_chf * 30, 2)
