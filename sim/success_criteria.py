"""
success_criteria.py
Single source of truth for what "good" means on a $100 book.

Three stages (plain English):
  LAB   = interesting on history + strict exam. Not money.
  PAPER = first real success: forward paper for ~30d with enough trades.
  LIVE  = tiny real capital only after paper pass; kill switches apply.

Nothing is "go live" from search score alone.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Book / costs (shared assumptions)
# ---------------------------------------------------------------------------
BOOK_USD = 100.0
FEE_RATE_BASE = 0.00022  # 2.2 bps Jupiter-measured taker (per side in engine)
FEE_RATE_STRESS_MID = 0.0005  # 5 bps
FEE_RATE_STRESS_HIGH = 0.0010  # 10 bps
MEV_PROB_SEARCH = 0.30
MEV_COST_BPS = 15.0
# Fixed cost per swap (Solana network + priority fees, independent of size).
# On $25-50 positions this is 6-25 bps per side — it is what makes
# high-frequency churning structurally unprofitable on a small book.
FIXED_COST_PER_SIDE_USD = 0.03
# Jupiter is spot: SOL/USDC cannot be shorted there. The simulator only takes
# long entries so discovered edges are actually executable on the live venue.
LONG_ONLY = True
# Benchmark (beta) gate: in a rising market, any long-only strategy shows
# positive PnL just by being exposed. OOS profit must beat what the SAME
# exposure (time-in-market x position size) would have earned from market
# drift alone, by at least this margin.
LAB_BENCH_MIN_EXCESS_USD = 0.25   # at least $0.25 above the exposure benchmark
LAB_BENCH_EXCESS_FACTOR = 1.25    # and at least 25% above it when it is positive

# ---------------------------------------------------------------------------
# Stage 1 — LAB (discovery / funnel)
# ---------------------------------------------------------------------------
LAB_MIN_TRADES_FULL = 30
LAB_MIN_TRADES_OOS = 10
LAB_MIN_TRADES_PREFERRED = 50  # nicer sample, not hard gate
LAB_REQUIRE_POSITIVE_FULL_PNL = True
LAB_REQUIRE_POSITIVE_OOS_PNL = True
LAB_OOS_PNL_RATIO_MIN = 0.50  # OOS pnl >= 50% of IS when IS > 0
LAB_MAX_DRAWDOWN = 0.20  # 20% of book peak-to-trough on full sample
LAB_MAX_DRAWDOWN_FOLD = 0.35  # single WF fold catastrophe
LAB_WF_FOLDS = 5
LAB_WF_MAJORITY = 0.60
LAB_PERTURB_PROFIT_RATE = 0.60  # fraction of ±10% nudges still profitable
LAB_PERTURB_RETENTION = 0.40
LAB_MEV_BREAK_EVEN_MIN = 0.20  # must still work until MEV p > 20%
LAB_DSR_MIN = 0.50
LAB_DSR_COLD_START_TRIALS = 200
LAB_DSR_ALT_OOS_PNL = 5.0  # $ on $100 book
LAB_DSR_ALT_WF_RATE = 0.80

# Search fitness (ranking only — not dollars)
FITNESS_MIN = -250.0
FITNESS_MAX = 250.0
FITNESS_DISPLAY_CAP = 1000.0  # anything beyond = broken legacy score

# ---------------------------------------------------------------------------
# Stage 2 — PAPER (first real success)
# ---------------------------------------------------------------------------
PAPER_MIN_DAYS = 30
PAPER_MIN_TRADES = 20
PAPER_MIN_NET_PNL_USD = 5.0  # +5% on $100
PAPER_TARGET_NET_PNL_USD = 10.0  # +10% stretch
PAPER_MAX_DRAWDOWN = 0.15  # 15% preferred
PAPER_MAX_DRAWDOWN_HARD = 0.20  # 20% fail
PAPER_MAX_WEEKLY_LOSS_FRAC = 0.10  # no week worse than -10% without thesis
PAPER_FEE_STRESS = FEE_RATE_STRESS_MID

# ---------------------------------------------------------------------------
# Stage 3 — LIVE micro (only after paper)
# ---------------------------------------------------------------------------
LIVE_START_USD = 25.0  # not full $100 on day one
LIVE_MAX_BOOK_USD = 100.0
LIVE_MIN_DAYS = 30
LIVE_MIN_NET_PNL_USD = 0.0  # "didn't die" floor
LIVE_TARGET_NET_PNL_USD = 5.0
LIVE_MAX_DRAWDOWN = 0.20
LIVE_KILL_DRAWDOWN = 0.20
LIVE_KILL_CONSEC_LOSS_DAYS = 5

# ---------------------------------------------------------------------------
# What is NOT success
# ---------------------------------------------------------------------------
NOT_SUCCESS = (
    "Search score alone (including huge/legacy numbers)",
    "High win rate with tiny or negative dollar P&L",
    "N < 30 trades on full sample",
    "Funnel pass once with no forward paper",
    "Clone families counted as many independent wins",
)


@dataclass
class LabVerdict:
    passed: bool
    stage: str = "LAB"
    reasons: Optional[List[str]] = None
    metrics: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.reasons is None:
            self.reasons = []
        if self.metrics is None:
            self.metrics = {}

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PaperVerdict:
    passed: bool
    stage: str = "PAPER"
    reasons: Optional[List[str]] = None
    metrics: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.reasons is None:
            self.reasons = []
        if self.metrics is None:
            self.metrics = {}

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LiveVerdict:
    passed: bool
    killed: bool
    stage: str = "LIVE"
    reasons: Optional[List[str]] = None
    metrics: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.reasons is None:
            self.reasons = []
        if self.metrics is None:
            self.metrics = {}

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def evaluate_lab_backtest(
    *,
    total_trades: int,
    total_pnl: float,
    max_drawdown: float,
    oos_trades: Optional[int] = None,
    oos_pnl: Optional[float] = None,
    is_pnl: Optional[float] = None,
) -> LabVerdict:
    """Quick lab-style checks on raw backtest numbers (pre-funnel)."""
    reasons: List[str] = []
    ok = True

    if total_trades < LAB_MIN_TRADES_FULL:
        ok = False
        reasons.append(f"trades {total_trades} < {LAB_MIN_TRADES_FULL}")
    if LAB_REQUIRE_POSITIVE_FULL_PNL and total_pnl <= 0:
        ok = False
        reasons.append(f"full pnl ${total_pnl:.2f} not > 0")
    if max_drawdown is not None and max_drawdown > LAB_MAX_DRAWDOWN:
        ok = False
        reasons.append(f"drawdown {max_drawdown:.0%} > {LAB_MAX_DRAWDOWN:.0%}")

    if oos_trades is not None and oos_trades < LAB_MIN_TRADES_OOS:
        ok = False
        reasons.append(f"oos trades {oos_trades} < {LAB_MIN_TRADES_OOS}")
    if oos_pnl is not None and LAB_REQUIRE_POSITIVE_OOS_PNL and oos_pnl <= 0:
        ok = False
        reasons.append(f"oos pnl ${oos_pnl:.2f} not > 0")
    if (
        oos_pnl is not None
        and is_pnl is not None
        and is_pnl > 0
        and oos_pnl / max(is_pnl, 1e-9) < LAB_OOS_PNL_RATIO_MIN
    ):
        ok = False
        reasons.append(
            f"oos/is pnl ratio {oos_pnl / is_pnl:.2f} < {LAB_OOS_PNL_RATIO_MIN}"
        )

    if ok:
        reasons.append("lab backtest bars clear")

    return LabVerdict(
        passed=ok,
        reasons=reasons,
        metrics={
            "total_trades": total_trades,
            "total_pnl": total_pnl,
            "max_drawdown": max_drawdown,
            "oos_trades": oos_trades,
            "oos_pnl": oos_pnl,
            "is_pnl": is_pnl,
            "book_usd": BOOK_USD,
        },
    )


def evaluate_paper_window(
    *,
    days: float,
    total_trades: int,
    net_pnl_usd: float,
    max_drawdown: float,
    worst_week_frac: Optional[float] = None,
    fee_rate: float = FEE_RATE_BASE,
) -> PaperVerdict:
    """
    First real success bar on $100 book.

    Pass when:
      days >= 30, trades >= 20, net pnl >= +$5, DD <= 20% (prefer 15%),
      worst week not worse than -10% if provided.
    """
    reasons: List[str] = []
    ok = True

    if days < PAPER_MIN_DAYS:
        ok = False
        reasons.append(f"days {days:.1f} < {PAPER_MIN_DAYS}")
    if total_trades < PAPER_MIN_TRADES:
        ok = False
        reasons.append(f"trades {total_trades} < {PAPER_MIN_TRADES}")
    if net_pnl_usd < PAPER_MIN_NET_PNL_USD:
        ok = False
        reasons.append(
            f"net pnl ${net_pnl_usd:.2f} < ${PAPER_MIN_NET_PNL_USD:.0f} target"
        )
    if max_drawdown > PAPER_MAX_DRAWDOWN_HARD:
        ok = False
        reasons.append(
            f"drawdown {max_drawdown:.0%} > hard max {PAPER_MAX_DRAWDOWN_HARD:.0%}"
        )
    elif max_drawdown > PAPER_MAX_DRAWDOWN:
        reasons.append(
            f"drawdown {max_drawdown:.0%} above preferred {PAPER_MAX_DRAWDOWN:.0%} (still under hard cap)"
        )
    if worst_week_frac is not None and worst_week_frac < -PAPER_MAX_WEEKLY_LOSS_FRAC:
        ok = False
        reasons.append(
            f"worst week {worst_week_frac:.0%} < -{PAPER_MAX_WEEKLY_LOSS_FRAC:.0%}"
        )
    if fee_rate > PAPER_FEE_STRESS and net_pnl_usd < PAPER_MIN_NET_PNL_USD:
        ok = False
        reasons.append("failed under fee stress")

    if ok:
        if net_pnl_usd >= PAPER_TARGET_NET_PNL_USD:
            reasons.append(
                f"PAPER PASS (stretch): +${net_pnl_usd:.2f} on ${BOOK_USD:.0f}"
            )
        else:
            reasons.append(
                f"PAPER PASS: +${net_pnl_usd:.2f} on ${BOOK_USD:.0f} over {days:.0f}d"
            )

    return PaperVerdict(
        passed=ok,
        reasons=reasons,
        metrics={
            "days": days,
            "total_trades": total_trades,
            "net_pnl_usd": net_pnl_usd,
            "max_drawdown": max_drawdown,
            "worst_week_frac": worst_week_frac,
            "fee_rate": fee_rate,
            "book_usd": BOOK_USD,
            "min_net_pnl_usd": PAPER_MIN_NET_PNL_USD,
            "target_net_pnl_usd": PAPER_TARGET_NET_PNL_USD,
        },
    )


def evaluate_live_window(
    *,
    days: float,
    net_pnl_usd: float,
    max_drawdown: float,
    consec_loss_days: int = 0,
    book_usd: float = LIVE_START_USD,
) -> LiveVerdict:
    """Live micro sleeve checks + kill switches."""
    reasons: List[str] = []
    killed = False
    ok = True

    if max_drawdown >= LIVE_KILL_DRAWDOWN:
        killed = True
        ok = False
        reasons.append(f"KILL: drawdown {max_drawdown:.0%} >= {LIVE_KILL_DRAWDOWN:.0%}")
    if consec_loss_days >= LIVE_KILL_CONSEC_LOSS_DAYS:
        killed = True
        ok = False
        reasons.append(
            f"KILL: {consec_loss_days} consecutive loss days >= {LIVE_KILL_CONSEC_LOSS_DAYS}"
        )

    if days < LIVE_MIN_DAYS:
        ok = False
        reasons.append(f"days {days:.1f} < {LIVE_MIN_DAYS}")
    if net_pnl_usd < LIVE_MIN_NET_PNL_USD:
        ok = False
        reasons.append(f"net pnl ${net_pnl_usd:.2f} below floor ${LIVE_MIN_NET_PNL_USD:.0f}")
    if max_drawdown > LIVE_MAX_DRAWDOWN:
        ok = False
        reasons.append(f"drawdown {max_drawdown:.0%} > {LIVE_MAX_DRAWDOWN:.0%}")

    if ok and not killed:
        if net_pnl_usd >= LIVE_TARGET_NET_PNL_USD:
            reasons.append(f"LIVE micro target hit: +${net_pnl_usd:.2f} on ${book_usd:.0f}")
        else:
            reasons.append(
                f"LIVE micro survived: +${net_pnl_usd:.2f} on ${book_usd:.0f} (floor only)"
            )

    return LiveVerdict(
        passed=ok and not killed,
        killed=killed,
        reasons=reasons,
        metrics={
            "days": days,
            "net_pnl_usd": net_pnl_usd,
            "max_drawdown": max_drawdown,
            "consec_loss_days": consec_loss_days,
            "book_usd": book_usd,
        },
    )


def sanitize_search_score(v: float) -> float:
    """Display/ranking hygiene for legacy explodey fitness."""
    try:
        x = float(v or 0.0)
    except Exception:
        return 0.0
    if x != x or x in (float("inf"), float("-inf")):
        return 0.0
    if abs(x) > FITNESS_DISPLAY_CAP:
        return 0.0
    return x


def criteria_public_dict() -> Dict[str, Any]:
    """JSON-friendly snapshot for dashboard / API / docs."""
    return {
        "book_usd": BOOK_USD,
        "fee_rate_base": FEE_RATE_BASE,
        "not_success": list(NOT_SUCCESS),
        "lab": {
            "min_trades_full": LAB_MIN_TRADES_FULL,
            "min_trades_oos": LAB_MIN_TRADES_OOS,
            "min_trades_preferred": LAB_MIN_TRADES_PREFERRED,
            "require_positive_full_pnl": LAB_REQUIRE_POSITIVE_FULL_PNL,
            "require_positive_oos_pnl": LAB_REQUIRE_POSITIVE_OOS_PNL,
            "oos_pnl_ratio_min": LAB_OOS_PNL_RATIO_MIN,
            "max_drawdown": LAB_MAX_DRAWDOWN,
            "wf_majority": LAB_WF_MAJORITY,
            "meaning": "Interesting on history + strict exam. Not deployable money.",
        },
        "paper": {
            "min_days": PAPER_MIN_DAYS,
            "min_trades": PAPER_MIN_TRADES,
            "min_net_pnl_usd": PAPER_MIN_NET_PNL_USD,
            "target_net_pnl_usd": PAPER_TARGET_NET_PNL_USD,
            "max_drawdown": PAPER_MAX_DRAWDOWN,
            "max_drawdown_hard": PAPER_MAX_DRAWDOWN_HARD,
            "meaning": (
                f"First real success on ${BOOK_USD:.0f}: "
                f">={PAPER_MIN_DAYS}d paper, >={PAPER_MIN_TRADES} trades, "
                f">=+${PAPER_MIN_NET_PNL_USD:.0f}, DD<=${BOOK_USD * PAPER_MAX_DRAWDOWN_HARD:.0f}."
            ),
        },
        "live": {
            "start_usd": LIVE_START_USD,
            "max_book_usd": LIVE_MAX_BOOK_USD,
            "min_days": LIVE_MIN_DAYS,
            "min_net_pnl_usd": LIVE_MIN_NET_PNL_USD,
            "target_net_pnl_usd": LIVE_TARGET_NET_PNL_USD,
            "max_drawdown": LIVE_MAX_DRAWDOWN,
            "kill_drawdown": LIVE_KILL_DRAWDOWN,
            "kill_consec_loss_days": LIVE_KILL_CONSEC_LOSS_DAYS,
            "meaning": "Tiny real sleeve only after paper pass. Kill at 20% DD.",
        },
        "single_number_target": (
            f"Success on ${BOOK_USD:.0f} = after {PAPER_MIN_DAYS}d paper-live, "
            f"net >= +${PAPER_MIN_NET_PNL_USD:.0f}, max DD <= ${BOOK_USD * PAPER_MAX_DRAWDOWN_HARD:.0f}, "
            f">= {PAPER_MIN_TRADES} trades, fees included."
        ),
    }


def plain_english_summary() -> str:
    c = criteria_public_dict()
    lines = [
        f"Book: ${BOOK_USD:.0f} fake (lab/paper) or tiny live sleeve.",
        f"LAB: >= {LAB_MIN_TRADES_FULL} trades, OOS profit, DD <= {LAB_MAX_DRAWDOWN:.0%}, pass strict exam.",
        f"PAPER WIN: >= {PAPER_MIN_DAYS}d, >= {PAPER_MIN_TRADES} trades, "
        f">= +${PAPER_MIN_NET_PNL_USD:.0f}, DD <= {PAPER_MAX_DRAWDOWN_HARD:.0%}.",
        f"LIVE: start ~${LIVE_START_USD:.0f}, kill if DD >= {LIVE_KILL_DRAWDOWN:.0%} "
        f"or {LIVE_KILL_CONSEC_LOSS_DAYS} loss-days in a row.",
        "NOT success: search score alone, WR theater, thin samples, one-off funnel luck.",
        c["single_number_target"],
    ]
    return "\n".join(lines)
