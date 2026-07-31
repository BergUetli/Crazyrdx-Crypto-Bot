"""
config.py
All tunable parameters for the trading bot simulation.
"""

from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
LOG_DIR = BASE_DIR / "logs"
REPORT_DIR = BASE_DIR / "reports"
MODEL_DIR = BASE_DIR / "models"

# Ensure directories exist
for d in [DATA_DIR, LOG_DIR, REPORT_DIR, MODEL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Database paths
DB_RAW_QUOTES = DATA_DIR / "raw_quotes.db"
DB_POOLS = DATA_DIR / "pool_snapshots.db"
DB_CEX = DATA_DIR / "cex_feeds.db"
DB_NETWORK = DATA_DIR / "network_stats.db"
DB_LABELS = DATA_DIR / "labels.db"
DB_FEATURES = DATA_DIR / "features.db"

# Trading pairs
PAIRS = {
    "SOL/USDC": {
        "input_mint": "So11111111111111111111111111111111111111112",
        "output_mint": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
        "input_decimals": 9,
        "output_decimals": 6,
        "trade_amount_raw": 10_000_000_000,  # 10 SOL
    },
    "SOL/USDT": {
        "input_mint": "So11111111111111111111111111111111111111112",
        "output_mint": "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB",
        "input_decimals": 9,
        "output_decimals": 6,
        "trade_amount_raw": 10_000_000_000,
    },
    "BONK/USDC": {
        "input_mint": "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263",
        "output_mint": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
        "input_decimals": 5,
        "output_decimals": 6,
        "trade_amount_raw": 1_000_000_000_000,  # 1M BONK
    },
    "WIF/USDC": {
        "input_mint": "EKpQGSJtjMFqKZ9KQanSqYXRcF8fBopzLHYxdM65zcjm",
        "output_mint": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
        "input_decimals": 6,
        "output_decimals": 6,
        "trade_amount_raw": 100_000_000,  # 100 WIF
    },
}

# API endpoints
JUPITER_QUOTE_URL = "https://api.jup.ag/swap/v1/quote"
RAYDIUM_POOLS_URL = "https://api.raydium.io/v2/main/pairs"
ORCA_WHIRLPOOLS_URL = "https://api.orca.so/v1/whirlpool/list"
BINANCE_TICKER_URL = "https://api.binance.com/api/v3/ticker/price"
COINBASE_TICKER_URL = "https://api.coinbase.com/v2/prices"

# Polling intervals (seconds)
POLL_JUPITER = 5
POLL_POOLS = 60
POLL_CEX = 300  # 5 minutes
POLL_NETWORK = 300

# Feature engineering
ROLLING_WINDOWS = [6, 12, 60]  # 30s, 60s, 300s at 5s intervals
LAG_PERIODS = [1, 2, 3]

# Labeling
LABEL_HORIZONS = [1, 2, 6, 12]  # 5s, 10s, 30s, 60s ahead
SPREAD_THRESHOLDS = {
    "noise": 20,        # < 20 bps
    "transient": 50,    # 20-50 bps, collapses fast
    "exploitable": 50,  # > 50 bps, persists
    "high_value": 200,  # > 200 bps
}

# Simulation
TRADE_SIZE_USD = 25.0
STARTING_CAPITAL_USD = 100.0

# Cost tracking
MONTHLY_COST_CEILING_CHF = 2.00
ELECTRICITY_RATE_CHF = 0.25  # per kWh
M4_TDP_WATTS = 30.0

# Safety
MAX_TRADE_SIZE_USD = 50.0
MAX_DAILY_LOSS_USD = 20.0
MAX_CONSECUTIVE_LOSSES = 5
MIN_OPPORTUNITY_SCORE = 0.70
