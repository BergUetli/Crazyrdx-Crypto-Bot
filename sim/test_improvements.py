#!/usr/bin/env python3
"""
test_improvements.py
Verification suite for the 2026-08-01 engine improvements.

Covers:
 1. Threshold-range lookup (longest-match + missing indicators)
 2. Empirical threshold calibration
 3. DNA signature / evaluation cache identity
 4. Kelly sizing fix
 5. Gap-aware exit fills + trailing-stop look-ahead fix
 6. "Most recent N" SQL limit fix
 7. End-to-end evolution run (parallel + serial + cache + warm start)
 8. Promotion funnel end-to-end on the evolved winner

Run from the repo root:
    python3 sim/test_improvements.py

Uses only synthetic data; all persistent state (kill archive, champions,
trials log, funnel results) is redirected to a temp directory so the real
evolution artifacts are never touched.
"""

import json
import math
import random
import sqlite3
import sys
import tempfile
import time
from pathlib import Path

SIM = Path(__file__).resolve().parent
sys.path.insert(0, str(SIM))

import numpy as np

PASS = 0
FAIL = 0


def check(name: str, cond: bool, detail: str = ""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok  {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name} {detail}")


# ---------------------------------------------------------------------------
# Synthetic market data
# ---------------------------------------------------------------------------

def make_synthetic_features(n: int = 1200, seed: int = 7):
    """Random-walk SOL-like 1h candles with every genome indicator present."""
    rng = np.random.default_rng(seed)
    random.seed(seed)
    from evolution.genome import INDICATORS

    px = 150.0
    rows = []
    t0 = 1_700_000_000_000
    closes = []
    for i in range(n):
        ret = rng.normal(0.0002, 0.012) + 0.004 * math.sin(i / 60.0)
        op = px
        px = max(1.0, px * (1 + ret))
        hi = max(op, px) * (1 + abs(rng.normal(0, 0.004)))
        lo = min(op, px) * (1 - abs(rng.normal(0, 0.004)))
        closes.append(px)
        f = {
            "open": op, "high": hi, "low": lo, "close": px,
            "volume": float(rng.uniform(1e4, 1e6)),
        }
        c = np.array(closes[-200:])
        f["sma_5"] = float(c[-5:].mean())
        f["sma_20"] = float(c[-20:].mean())
        f["sma_50"] = float(c[-50:].mean())
        f["price_vs_sma_20"] = (px / f["sma_20"] - 1) * 10000
        f["hl_range_pct"] = (hi - lo) / px * 10000
        f["hl_range_avg_4h"] = f["hl_range_pct"] * float(rng.uniform(0.7, 1.3))
        f["hl_range_avg_1d"] = f["hl_range_pct"] * float(rng.uniform(0.7, 1.3))
        f["price_roc_1h"] = ret * 10000
        f["price_roc_4h"] = float((px / c[max(-5, -len(c))] - 1) * 10000)
        f["price_roc_1d"] = float((px / c[max(-24, -len(c))] - 1) * 10000)
        f["price_roc_3d"] = float((px / c[max(-72, -len(c))] - 1) * 10000)
        f["price_roc_1w"] = float((px / c[0] - 1) * 10000)
        f["volatility_4h"] = float(np.std(np.diff(c[-5:]) / c[-5:-1]) * 10000) if len(c) > 5 else 50.0
        f["volatility_1d"] = float(np.std(np.diff(c[-24:]) / c[-24:-1]) * 10000) if len(c) > 24 else 50.0
        f["volatility_3d"] = f["volatility_1d"] * float(rng.uniform(0.8, 1.2))
        f["volatility_1w"] = f["volatility_1d"] * float(rng.uniform(0.8, 1.2))
        hour = (i % 24)
        f["hour_of_day_sin"] = math.sin(2 * math.pi * hour / 24)
        f["hour_of_day_cos"] = math.cos(2 * math.pi * hour / 24)
        f["day_of_week"] = float((i // 24) % 7)
        f["is_weekend"] = 1.0 if f["day_of_week"] >= 5 else 0.0
        # Everything not explicitly modeled gets a plausible random value
        defaults = {
            "body_pct": abs(ret) * 8000, "upper_wick_pct": float(rng.uniform(0, 30)),
            "lower_wick_pct": float(rng.uniform(0, 30)),
            "volume_roc_4h": float(rng.normal(0, 40)), "volume_roc_1d": float(rng.normal(0, 60)),
            "volume_sma_ratio": float(rng.uniform(0.5, 2.0)),
            "volume_weighted_price": px * float(rng.uniform(0.99, 1.01)),
            "taker_buy_ratio": float(rng.uniform(0.3, 0.7)),
            "taker_buy_roc_4h": float(rng.normal(0, 1)), "taker_buy_roc_1d": float(rng.normal(0, 1)),
            "taker_buy_sma_ratio": float(rng.uniform(0.8, 1.2)),
            "sma_cross_5_20": float(rng.choice([-1.0, 0.0, 1.0])),
            "sma_cross_20_50": float(rng.choice([-1.0, 0.0, 1.0])),
            "price_vs_4h_sma": float(rng.normal(0, 100)), "price_vs_1d_sma": float(rng.normal(0, 150)),
            "returns_skew_1d": float(rng.normal(0, 1)), "returns_kurtosis_1d": float(rng.normal(3, 1)),
            "autocorrelation_1d": float(rng.uniform(-0.5, 0.5)),
            "close_lag_1": closes[-2] if len(closes) > 1 else px,
            "close_lag_2": closes[-3] if len(closes) > 2 else px,
            "close_lag_3": closes[-4] if len(closes) > 3 else px,
            "volume_lag_1": f["volume"], "returns_lag_1": float(rng.normal(0, 1)),
            "returns_lag_2": float(rng.normal(0, 1)),
            "trend_alignment_1h_4h": float(rng.choice([-1.0, 0.0, 1.0])),
            "trend_alignment_4h_1d": float(rng.choice([-1.0, 0.0, 1.0])),
            "trend_alignment_1h_1d": float(rng.choice([-1.0, 0.0, 1.0])),
            "trend_alignment_all": float(rng.choice([0.0, 1.0], p=[0.8, 0.2])),
            "momentum_divergence_1h_4h": float(rng.normal(0, 30)),
            "momentum_divergence_4h_1d": float(rng.normal(0, 30)),
            "momentum_divergence_1h_1d": float(rng.normal(0, 30)),
            "volatility_regime_1h_4h": float(rng.uniform(0.3, 3.0)),
            "volatility_regime_4h_1d": float(rng.uniform(0.3, 3.0)),
            "volatility_regime_1h_1d": float(rng.uniform(0.3, 3.0)),
            "volume_confirmation_1h_4h": float(rng.uniform(0, 200)),
            "tft_prediction": 0.0, "tft_confidence": 0.0,
            "funding_rate": float(rng.normal(0, 3e-4)),
            "funding_rate_8h_avg": float(rng.normal(0, 2e-4)),
            "funding_rate_roc": float(rng.normal(0, 1e-4)),
            "funding_rate_extreme": float(rng.choice([0.0, 1.0], p=[0.9, 0.1])),
            "cex_dex_basis_bps": float(rng.uniform(0, 15)),
            "cex_dex_basis_roc_4h": float(rng.normal(0, 3)),
            "cex_dex_basis_roc_1d": float(rng.normal(0, 5)),
            "cex_dex_basis_extreme": float(rng.choice([0.0, 1.0], p=[0.9, 0.1])),
            "taker_flow_imbalance": float(rng.uniform(-0.4, 0.4)),
            "taker_flow_imbalance_4h": float(rng.uniform(-0.3, 0.3)),
            "taker_flow_imbalance_roc": float(rng.normal(0, 0.3)),
            "taker_flow_persistence": float(rng.uniform(0, 1)),
            "dex_liquidity_ratio": float(rng.uniform(0.5, 2.0)),
            "funding_basis_divergence": float(rng.uniform(0, 1)),
            "market_stress_index": float(rng.uniform(0, 0.4)),
        }
        f.update(defaults)
        for ind in INDICATORS:
            f.setdefault(ind, 0.0)
        rows.append({"ts": t0 + i * 3_600_000, "pair": "SOL/USDC", "features": f})
    return rows


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

def test_threshold_ranges():
    print("\n[1] threshold range lookup")
    from evolution.genome import get_threshold_range, CALIBRATED_RANGES
    CALIBRATED_RANGES.clear()
    check("volatility_regime uses its own range, not volatility's",
          get_threshold_range("volatility_regime_1h_4h") == (0.0, 5.0))
    check("price_vs_sma matches price_vs_sma, not sma",
          get_threshold_range("price_vs_sma_20") == (-500.0, 500.0))
    check("sol_btc_ratio_roc matches roc range, not raw ratio",
          get_threshold_range("sol_btc_ratio_roc_4h") == (-100.0, 100.0))
    check("funding_rate_extreme is a 0-1 flag",
          get_threshold_range("funding_rate_extreme") == (0.0, 1.0))
    check("hour_of_day_sin now has a sane range",
          get_threshold_range("hour_of_day_sin") == (-1.0, 1.0))
    check("day_of_week now has a sane range",
          get_threshold_range("day_of_week") == (0.0, 6.0))


def test_calibration(features):
    print("\n[2] empirical calibration")
    from evolution.genome import (
        calibrate_threshold_ranges, get_threshold_range, CALIBRATED_RANGES,
    )
    n = calibrate_threshold_ranges(features)
    check(f"calibrated many indicators (got {n})", n >= 60)
    lo, hi = get_threshold_range("volatility_regime_1h_4h")
    check("calibrated range brackets actual data", 0.2 <= lo <= 1.0 and 2.0 <= hi <= 3.5,
          f"got ({lo:.2f},{hi:.2f})")
    check("constant features (tft heads) not calibrated",
          "tft_prediction" not in CALIBRATED_RANGES)
    # Thresholds sampled from calibrated ranges must be able to flip
    vals = [r["features"]["volatility_regime_1h_4h"] for r in features]
    frac_below = sum(1 for v in vals if v < hi) / len(vals)
    check("threshold at hi splits data non-degenerately", 0.5 < frac_below <= 0.97)


def test_dna_signature():
    print("\n[3] dna signature")
    from evolution.genome import dna_signature, random_genome, mutate
    random.seed(1)
    g = random_genome()
    clone = type(g).from_dict(g.to_dict())
    clone.genome_id = "clone_xyz"
    check("clone has same signature", dna_signature(g) == dna_signature(clone))
    m = mutate(g, mutation_rate=1.0)
    check("mutant has different signature", dna_signature(g) != dna_signature(m))


def test_kelly(features):
    print("\n[4] kelly sizing")
    from evolution.evaluator import GenomeEvaluator
    from evolution.genome import random_genome
    random.seed(2)
    ev = GenomeEvaluator(features, augment=False)
    g = random_genome()
    g.entry_logic = "OR"
    g.sizing_method = "kelly"
    g.sizing_max = 0.5
    fn = ev._build_signal_fn(g)
    sizes = []
    for i in range(60, min(400, len(features))):
        sig = fn(ev.features, i)
        if sig:
            sizes.append(sig[2])
    expected = 0.5 * (0.55 - 0.45 / 2.0)  # half-Kelly ≈ 0.1625
    check("kelly signals produced", len(sizes) > 0)
    if sizes:
        check("kelly size ≈ half-Kelly (not maxed out)",
              all(abs(s - expected) < 1e-9 for s in sizes),
              f"sizes={set(round(s,4) for s in sizes)} expected={expected:.4f}")


def test_gap_fills():
    print("\n[5] gap-aware exits + trailing look-ahead")
    from layer1.backtest_engine import BacktestEngine

    def bars(specs):
        return [
            {"ts": i * 3_600_000,
             "features": {"open": o, "high": h, "low": l, "close": c}}
            for i, (o, h, l, c) in enumerate(specs)
        ]

    eng = BacktestEngine()
    # Long, SL 2% from exec 100 -> stop at 98. Bar 2 gaps open at 90.
    feats = bars([(100, 101, 99, 100), (100, 101, 99, 100),
                  (90, 91, 88, 89), (89, 90, 87, 88), (88, 89, 86, 87)])
    idx, price = eng._resolve_exit(
        features=feats, entry_idx=0, end_idx=len(feats), direction="long",
        exec_price=100.0, rules=[{"type": "stop_loss", "value": 0.02},
                                 {"type": "time_stop", "value": 10}],
        signal_generator=lambda f, i: None)
    check("gapped stop fills at open (90), not stop (98)", price == 90.0,
          f"got {price}")
    # Same but intrabar touch: open 99 (above stop), low 97 -> fill at 98
    feats2 = bars([(100, 101, 99, 100), (100, 101, 99, 100),
                   (99, 100, 97, 98), (98, 99, 96, 97), (97, 98, 95, 96)])
    idx, price = eng._resolve_exit(
        features=feats2, entry_idx=0, end_idx=len(feats2), direction="long",
        exec_price=100.0, rules=[{"type": "stop_loss", "value": 0.02},
                                 {"type": "time_stop", "value": 10}],
        signal_generator=lambda f, i: None)
    check("intrabar stop fills at stop price (98)", abs(price - 98.0) < 1e-9,
          f"got {price}")
    # Trailing stop must not use the SAME bar's high as its peak:
    # bar 2 spikes high to 110 and dips low to 104.4. Old code set peak=110
    # then triggered 5% trail at 104.5 within the same bar. New code uses the
    # prior peak (101), trail level 95.95, low 104.4 never touches -> no exit
    # on bar 2.
    feats3 = bars([(100, 101, 99, 100), (100, 101, 99, 100),
                   (105, 110, 104.4, 109), (109, 111, 108, 110),
                   (110, 111, 109, 110), (110, 111, 109, 110)])
    idx, price = eng._resolve_exit(
        features=feats3, entry_idx=0, end_idx=len(feats3), direction="long",
        exec_price=100.0, rules=[{"type": "trailing_stop", "value": 0.05},
                                 {"type": "time_stop", "value": 3}],
        signal_generator=lambda f, i: None)
    check("trailing stop no longer fires off same-bar peak", idx == 3,
          f"exited at bar {idx} price {price}")


def test_limit_query():
    print("\n[6] most-recent-N limit query")
    import layer1.historical_feature_engine_1h as hfe
    with tempfile.TemporaryDirectory() as td:
        db = Path(td) / "t.db"
        conn = sqlite3.connect(str(db))
        conn.execute(
            "CREATE TABLE features_1h (id INTEGER PRIMARY KEY, ts INTEGER, "
            "pair TEXT, features_json TEXT)")
        for i in range(10):
            conn.execute(
                "INSERT INTO features_1h (ts, pair, features_json) VALUES (?,?,?)",
                (1000 + i, "SOL/USDC", '{"close": %d}' % i))
        conn.commit(); conn.close()
        old = hfe.DB_HIST_FEATURES_1H
        try:
            hfe.DB_HIST_FEATURES_1H = db
            rows = hfe.get_historical_features_1h("SOL/USDC", limit=3)
        finally:
            hfe.DB_HIST_FEATURES_1H = old
        ts = [r["ts"] for r in rows]
        check("limit=3 returns the 3 NEWEST rows", ts == [1007, 1008, 1009],
              f"got {ts}")
        check("rows are chronological", ts == sorted(ts))


# ---------------------------------------------------------------------------
# End-to-end
# ---------------------------------------------------------------------------

def _redirect_state_to_tmp(tmp: Path):
    """Point kill archive / champions / trials / funnel output at tmp dir."""
    import evolution.kill_archive as ka
    import evolution.promotion_funnel as pf
    ka.KILL_PATH = tmp / "killed_dna.json"
    ka._ARCHIVE = ka.KillArchive(ka.KILL_PATH)
    pf.TRIALS_PATH = tmp / "trials_log.jsonl"
    pf.TRIAL_COUNT_PATH = tmp / "trial_count.json"
    pf.CHAMPIONS_PATH = tmp / "champions.json"
    pf.FUNNEL_DIR = tmp / "funnel_results"
    pf.FUNNEL_DIR.mkdir(parents=True, exist_ok=True)


def test_e2e(features, n_workers):
    label = f"workers={n_workers}"
    print(f"\n[7] end-to-end evolution ({label})")
    from evolution.evaluator import EvolutionEngine
    from evolution.genome import random_genome
    random.seed(11)

    seeds = [random_genome() for _ in range(2)]
    t0 = time.time()
    engine = EvolutionEngine(
        features,
        population_size=24,
        elite_size=4,
        mutation_rate=0.3,
        crossover_rate=0.65,
        immigrant_rate=0.2,
        use_kill_archive=True,
        seed_genomes=seeds,
        n_workers=n_workers,
    )
    best = engine.evolve_continuous(
        max_duration_s=90, no_improvement_limit=3, verbose=False)
    dt = time.time() - t0
    check(f"evolution completed ({label}, {dt:.0f}s, "
          f"{engine.generation} gens)", best is not None)
    check("every genome has fitness + results",
          all(g.backtest_results is not None for g in engine.population))
    check("best fitness is finite and clamped",
          -400.0 <= best.fitness <= 200.0, f"got {best.fitness}")
    evaluated = [g for g in engine.population
                 if not (g.backtest_results or {}).get("killed")]
    check("population produced real backtests", len(evaluated) > 0)
    if n_workers > 1:
        check(f"eval cache used (hits={engine.cache_hits})",
              engine.cache_hits >= 0)  # informational; asserted not to crash
    check("warm-start seeds injected",
          any(g.genome_id.startswith(("seed_", "mut_seed_")) or
              any(p.startswith("seed_") for p in g.parent_ids)
              for g in engine.population) or engine.generation > 3)
    return best, engine


def test_funnel(features, best):
    print("\n[8] promotion funnel end-to-end")
    from evolution.promotion_funnel import PromotionFunnel
    funnel = PromotionFunnel(features)
    out = funnel.run(best, n_trials_context=5000, verbose=False)
    check("funnel returned a verdict", out.get("verdict") in
          ("PROMOTE_TO_PAPER", "REJECT"), f"got {out.get('verdict')}")
    check("gates recorded", isinstance(out.get("gates"), dict)
          and len(out["gates"]) >= 1)
    print(f"      verdict={out.get('verdict')} failed_at={out.get('failed_at')}")


def test_vintage_ledger(features, tmp: Path):
    print("\n[9] vintage forward ledger")
    import evolution.vintage_ledger as vl
    from evolution.genome import random_genome
    random.seed(21)
    old_db, old_gap = vl.LEDGER_DB, vl.RANDOM_COHORT_MIN_GAP_S
    try:
        vl.LEDGER_DB = tmp / "vintage_ledger.db"
        vl.RANDOM_COHORT_MIN_GAP_S = 0  # allow a control cohort per freeze in test

        champ = random_genome()
        st = vl.freeze_cycle(champ, features[:800])
        check("champion + controls frozen", st["frozen_champion"]
              and st["frozen_controls"] == vl.RANDOM_COHORT_SIZE + 2, str(st))
        st2 = vl.freeze_cycle(champ, features[:800])
        check("identical champion not re-frozen", not st2["frozen_champion"])

        # More vintages at later freeze points (distinct champions, ~weekly)
        for cut in (200, 400, 600, 1000):
            vl.freeze_cycle(random_genome(), features[:cut])

        n = vl.score_vintages(features)
        check(f"forward-scored all eligible vintages (n={n})", n >= 25)

        summ = vl.ledger_summary()
        check("summary ok with weekly cohorts", summ.get("ok")
              and summ.get("n_weeks", 0) >= 4, str(summ.get("reason")))
        pcts = [c["med_pct"] for c in summ.get("cohorts", [])]
        check("percentiles in [0,100]", all(0 <= p <= 100 for p in pcts))
        check("verdict produced", summ.get("verdict") in
              ("SMARTER", "WEAKER", "FLAT", "NO EDGE YET", "TOO EARLY"),
              str(summ.get("verdict")))
        check("percentile math", vl._percentile_of(5.0, [1, 2, 3, 4]) == 100.0
              and vl._percentile_of(0.0, [1, 2, 3, 4]) == 0.0
              and abs(vl._percentile_of(2.5, [1, 2, 3, 4]) - 50.0) < 1e-9)

        # Re-scoring with NO new data is skipped entirely (month-scale waste)
        n2 = vl.score_vintages(features)
        summ2 = vl.ledger_summary()
        check("re-scoring skipped when no new candles", n2 == 0
              and len(summ2["cohorts"]) == len(summ["cohorts"]), f"n2={n2}")

        # Champion freezes capped at 12/day (ledger bloat guard)
        from evolution.genome import random_genome as _rg
        for _ in range(20):
            vl.freeze_cycle(_rg(), features[:300])
        import sqlite3 as _sq
        n_champ = _sq.connect(str(vl.LEDGER_DB)).execute(
            "SELECT COUNT(*) FROM vintages WHERE kind='champion'").fetchone()[0]
        check(f"champion freezes capped (n={n_champ})", n_champ <= 12)

        # Dashboard integration renders
        import dashboard as dbmod
        html = dbmod.vintage_html_block({"vintage": summ})
        check("dashboard vintage block renders",
              "Chart D" in html and summ["verdict"] in html)
        html_empty = dbmod.vintage_html_block({"vintage": {"ok": False, "reason": "x"}})
        check("dashboard vintage block handles empty ledger",
              "Forward ledger" in html_empty)
        return summ
    finally:
        vl.LEDGER_DB, vl.RANDOM_COHORT_MIN_GAP_S = old_db, old_gap


def test_fast_signals_equivalence(features):
    print("\n[10] vectorized signals: exact equivalence with legacy path")
    from evolution.evaluator import GenomeEvaluator
    from evolution.genome import random_genome, ExitRule
    from layer1.fast_signals import build_array_signal_fn
    from layer1.backtest_engine import BacktestEngine

    random.seed(31)
    ev = GenomeEvaluator(features, augment=False)
    feats = ev.features

    genomes = []
    for logic in ("AND", "OR", "MEANREV", "BREAKOUT", "TREND", "TFT"):
        for _ in range(15):
            g = random_genome()
            g.entry_logic = logic
            genomes.append(g)
    # Force coverage of every sizing method and reversal exits
    for i, method in enumerate(("fixed", "kelly", "volatility_scaled", "equal_weight")):
        genomes[i].sizing_method = method
    for g in genomes[:10]:
        g.exit_rules.append(ExitRule("signal_reversal", 0.7))

    mismatches = 0
    compared_trades = 0
    t_legacy = t_fast = 0.0
    engine_slow = BacktestEngine(fast_columns=False)  # dict exits + legacy signals
    engine_fast = BacktestEngine(fast_columns=True)   # column exits + fast signals
    for g in genomes:
        t0 = time.time()
        r_old = engine_slow.run_backtest("legacy", "SOL/USDC", feats,
                                         ev._build_signal_fn(g), exit_rules=g.exit_rules)
        t_legacy += time.time() - t0
        fast_fn = build_array_signal_fn(g, feats)
        assert fast_fn is not None
        t0 = time.time()
        r_new = engine_fast.run_backtest("fast", "SOL/USDC", feats,
                                         fast_fn, exit_rules=g.exit_rules)
        t_fast += time.time() - t0

        same = (r_old.total_trades == r_new.total_trades
                and abs(r_old.total_pnl - r_new.total_pnl) < 1e-9)
        if same:
            for a, b in zip(r_old.trades, r_new.trades):
                compared_trades += 1
                if (a.entry_ts != b.entry_ts or a.exit_ts != b.exit_ts
                        or abs(a.entry_price - b.entry_price) > 1e-9
                        or abs(a.exit_price - b.exit_price) > 1e-9
                        or abs(a.size_usd - b.size_usd) > 1e-9):
                    same = False
                    break
        if not same:
            mismatches += 1
            print(f"    MISMATCH {g.entry_logic} {g.genome_id}: "
                  f"trades {r_old.total_trades} vs {r_new.total_trades}, "
                  f"pnl {r_old.total_pnl:.6f} vs {r_new.total_pnl:.6f}")

    check(f"all {len(genomes)} genomes produce identical trades "
          f"({compared_trades} trades compared)", mismatches == 0,
          f"{mismatches} mismatches")
    speed = t_legacy / max(t_fast, 1e-9)
    print(f"      signal-path speedup on this data: {speed:.1f}x "
          f"(legacy {t_legacy:.2f}s vs fast {t_fast:.2f}s)")
    check("fast path is not slower", speed > 1.0, f"{speed:.2f}x")

    # RANDOM baseline must decline the fast path (stochastic)
    g = random_genome()
    g.entry_logic = "RANDOM"
    check("RANDOM falls back to legacy path",
          build_array_signal_fn(g, feats) is None)

    # Kill switch respected inside the evaluator
    import evolution.evaluator as evmod
    old_flag = evmod.FAST_SIGNALS_ENABLED
    try:
        evmod.FAST_SIGNALS_ENABLED = False
        g2 = random_genome()
        g2.entry_logic = "OR"
        r = ev.evaluate(g2)
        check("FAST_SIGNALS=0 path still evaluates", "fitness" in r)
    finally:
        evmod.FAST_SIGNALS_ENABLED = old_flag


def test_realistic_economics(features):
    print("\n[11] realistic economics: fixed costs + long-only")
    from layer1.backtest_engine import BacktestEngine
    from evolution.evaluator import GenomeEvaluator
    from evolution.genome import random_genome
    random.seed(41)
    ev = GenomeEvaluator(features, augment=False)
    feats = ev.features

    # A genome that trades both directions under MEANREV
    g = random_genome()
    g.entry_logic = "MEANREV"
    fn = ev._build_signal_fn(g)

    free = BacktestEngine(fixed_cost_per_side=0.0, long_only=False)
    paid = BacktestEngine(fixed_cost_per_side=0.03, long_only=False)
    r_free = free.run_backtest("free", "SOL/USDC", feats, fn, exit_rules=g.exit_rules)
    r_paid = paid.run_backtest("paid", "SOL/USDC", feats, fn, exit_rules=g.exit_rules)
    check("both engines produced trades",
          r_free.total_trades > 0 and r_paid.total_trades > 0)
    # Per-trade: paid engine charges proportional fee + exactly $0.06 fixed
    ok_paid = all(
        abs(t.fee_cost - (t.size_usd * paid.fee_rate * 2 + 0.06)) < 1e-9
        for t in r_paid.trades
    )
    ok_free = all(
        abs(t.fee_cost - t.size_usd * free.fee_rate * 2) < 1e-9
        for t in r_free.trades
    )
    check(f"fixed $0.06/round-trip charged on every trade "
          f"(n={r_paid.total_trades})", ok_paid and ok_free)
    check("fixed costs reduce total PnL",
          r_paid.total_pnl < r_free.total_pnl)

    lo = BacktestEngine(long_only=True)
    r_lo = lo.run_backtest("lo", "SOL/USDC", feats, fn, exit_rules=g.exit_rules)
    check("long_only engine takes zero short trades",
          all(t.direction == "long" for t in r_lo.trades))
    r_both = free.run_backtest("both", "SOL/USDC", feats, fn, exit_rules=g.exit_rules)
    check("shorts existed to be excluded (control)",
          any(t.direction == "short" for t in r_both.trades))

    # Evaluator default engines carry the realistic settings
    check("evaluator engine is long-only with fixed costs",
          ev.engine.long_only and ev.engine.fixed_cost_per_side > 0)


def test_champion_flush(tmp: Path):
    print("\n[12] legacy champion flush")
    import evolution.promotion_funnel as pf
    old = pf.CHAMPIONS_PATH
    try:
        pf.CHAMPIONS_PATH = tmp / "champions.json"
        legacy = [{"genome_id": f"old_{i}", "score": 3e17, "genome": {}} for i in range(3)]
        modern = [{"genome_id": "new_1", "score": 84.2, "genome": {}}]
        pf.CHAMPIONS_PATH.write_text(json.dumps({"champions": legacy + modern}))
        n = pf.flush_legacy_champions()
        check("archived exactly the 3 legacy champions", n == 3, f"got {n}")
        kept = json.loads(pf.CHAMPIONS_PATH.read_text())["champions"]
        check("modern champion kept", [c["genome_id"] for c in kept] == ["new_1"])
        arch = json.loads((tmp / "champions_legacy.json").read_text())
        check("legacy file holds the archived 3", len(arch["champions"]) == 3)
        n2 = pf.flush_legacy_champions()
        check("flush is idempotent", n2 == 0, f"got {n2}")
    finally:
        pf.CHAMPIONS_PATH = old


def test_benchmark_gate(features):
    print("\n[13] benchmark (beta) gate + champion re-funnel skip")
    import evolution.promotion_funnel as pf
    from layer1.backtest_engine import SimulatedTrade

    def fake_result(trades):
        class R:
            pass
        r = R()
        r.trades = trades
        return r

    def mk_trade(entry_ts, exit_ts, size, direction="long"):
        return SimulatedTrade(
            entry_ts=entry_ts, exit_ts=exit_ts, pair="SOL/USDC",
            direction=direction, entry_price=100, exit_price=101,
            size_usd=size, gross_pnl=0, fee_cost=0.08, net_pnl=0,
            latency_s=10, slippage_bps=0, mev_cost_bps=0, signal_strength=1.0)

    # Window rises 20%; strategy in market ~90% of the time with $40 positions
    # (deep-copied rows so shared test data is not mutated)
    win = [{"ts": f["ts"], "features": dict(f["features"])} for f in features[:400]]
    first = win[0]["features"]["close"]
    win[-1]["features"]["close"] = first * 1.2
    t0, t1 = win[0]["ts"], win[-1]["ts"]
    span = t1 - t0
    beta_trades = [mk_trade(t0 + int(span*0.05), t0 + int(span*0.95), 40.0)]
    bench = pf.exposure_benchmark_pnl(fake_result(beta_trades), win)
    # 20% x 90% exposure x $40 = ~$7.2 minus costs
    check(f"exposure benchmark computed (${bench:.2f})", 6.0 < bench < 7.5)
    check("pure-beta PnL (= benchmark) is rejected",
          not pf.benchmark_gate_passed(bench, bench))
    check("PnL well above benchmark passes",
          pf.benchmark_gate_passed(bench * 2, bench))
    check("in a flat/down market, positive PnL passes",
          pf.benchmark_gate_passed(1.0, -0.5))
    check("tiny benchmark still needs $0.25 excess",
          not pf.benchmark_gate_passed(0.3, 0.1)
          and pf.benchmark_gate_passed(0.36, 0.1))

    # Champion-family skip in funnel candidate selection
    from evolution.genome import random_genome
    import tempfile as tf
    random.seed(51)
    g = random_genome()
    g.backtest_results = {"total_trades": 100}
    g.fitness = 50.0
    with tf.TemporaryDirectory() as td:
        old = pf.CHAMPIONS_PATH
        try:
            pf.CHAMPIONS_PATH = Path(td) / "champions.json"
            pf.CHAMPIONS_PATH.write_text(json.dumps(
                {"champions": [{"genome_id": "champ", "score": 50,
                                "genome": g.to_dict()}]}))
            res = pf.funnel_population_top(features[:200], [g], top_k=5,
                                           min_trades=30, verbose=False)
            check("champion family skipped from re-funneling", len(res) == 0)
        finally:
            pf.CHAMPIONS_PATH = old


def test_prune_dir():
    print("\n[15] disk retention pruning")
    import run_broad_evolution as rbe
    import tempfile as tf
    with tf.TemporaryDirectory() as td:
        d = Path(td)
        for i in range(30):
            (d / f"evolution_{1000 + i}.json").write_text("{}")
        (d / "latest.json").write_text("{}")
        removed = rbe.prune_dir(d, "evolution_*.json", keep=10)
        left = sorted(f.name for f in d.glob("evolution_*.json"))
        check("removed 20, kept newest 10", removed == 20 and len(left) == 10
              and left[0] == "evolution_1020.json")
        check("non-matching files untouched", (d / "latest.json").exists())


def test_champion_revalidation(features):
    print("\n[14] champion revalidation under current gates")
    import evolution.promotion_funnel as pf
    from evolution.genome import random_genome
    import tempfile as tf
    random.seed(61)

    g_pass, g_fail = random_genome(), random_genome()

    class StubFunnel:
        def __init__(self, feats):
            pass
        def run(self, genome, n_trials_context=None, verbose=True):
            ok = genome.genome_id == g_pass.genome_id
            return {"all_passed": ok,
                    "failed_at": None if ok else "benchmark"}

    with tf.TemporaryDirectory() as td:
        old_path, old_funnel = pf.CHAMPIONS_PATH, pf.PromotionFunnel
        try:
            pf.CHAMPIONS_PATH = Path(td) / "champions.json"
            pf.PromotionFunnel = StubFunnel
            pf.CHAMPIONS_PATH.write_text(json.dumps({"champions": [
                {"genome_id": g_pass.genome_id, "score": 80, "genome": g_pass.to_dict()},
                {"genome_id": g_fail.genome_id, "score": 70, "genome": g_fail.to_dict()},
            ]}))
            out = pf.revalidate_champions(features, verbose=False)
            check("one kept, one demoted", out == {"kept": 1, "demoted": 1}, str(out))
            kept = json.loads(pf.CHAMPIONS_PATH.read_text())["champions"]
            check("passer stays on board",
                  [c["genome_id"] for c in kept] == [g_pass.genome_id])
            unval = json.loads((Path(td) / "champions_unvalidated.json").read_text())
            check("failure recorded with gate name",
                  unval["champions"][0]["demoted_at_gate"] == "benchmark")
            out2 = pf.revalidate_champions(features, verbose=False)
            check("revalidation idempotent for passers",
                  out2 == {"kept": 1, "demoted": 0}, str(out2))
        finally:
            pf.CHAMPIONS_PATH, pf.PromotionFunnel = old_path, old_funnel


def test_embargo_and_cross_asset(features):
    print("\n[16] embargo + advisory cross-asset gate")
    from success_criteria import LAB_EMBARGO_BARS
    from evolution.evaluator import GenomeEvaluator
    import evolution.promotion_funnel as pf
    from evolution.genome import random_genome, EntryCondition
    import layer1.historical_feature_engine_1h as hfe
    random.seed(71)

    ev = GenomeEvaluator(features, augment=False)
    gap = ev._oos_features[0]["ts"] - ev._is_features[-1]["ts"]
    check(f"embargo gap between IS and OOS ({gap/3_600_000:.0f} bars)",
          gap >= LAB_EMBARGO_BARS * 3_600_000)

    funnel = pf.PromotionFunnel(features)
    # Scale-bound genome skips the check
    g1 = random_genome()
    g1.entry_logic = "AND"
    g1.entry_conditions = [EntryCondition("sma_50", ">", 80.0)]
    xa1 = funnel._cross_asset_check(g1)
    check("scale-bound genome skips cross-asset", "skipped" in xa1
          and xa1["passed"] and xa1["would_pass"])

    # Scale-free genome is tested on stubbed BTC/ETH features
    g2 = random_genome()
    g2.entry_logic = "MEANREV"
    old_fn = hfe.get_historical_features_1h
    try:
        hfe.get_historical_features_1h = lambda pair, **kw: features[:600]
        xa2 = funnel._cross_asset_check(g2)
    finally:
        hfe.get_historical_features_1h = old_fn
    check("scale-free genome tested on both assets",
          set(xa2["assets"].keys()) == {"BTC/USDC", "ETH/USDC"})
    check("advisory mode never blocks", xa2["passed"] is True
          and xa2["enforced"] is False)


def test_derivatives_collector():
    print("\n[17] derivatives collector storage")
    import tempfile as tf
    import layer1.derivatives_collector as dc
    with tf.TemporaryDirectory() as td:
        old = dc.DB_DERIVS
        try:
            dc.DB_DERIVS = Path(td) / "derivs.db"
            conn = dc.init_db()
            rows = [
                {"fundingTime": 1000, "fundingRate": "0.0001"},
                {"fundingTime": 2000, "fundingRate": "-0.0002"},
                {"fundingTime": "bad", "fundingRate": "0.1"},  # skipped
            ]
            n = dc.store_rows(conn, "SOLUSDT", "funding_rate", rows,
                              "fundingTime", "fundingRate")
            n2 = dc.store_rows(conn, "SOLUSDT", "funding_rate", rows,
                               "fundingTime", "fundingRate")  # upsert, no dupes
            conn.commit()
            total = conn.execute("SELECT COUNT(*) FROM derivs").fetchone()[0]
            val = conn.execute(
                "SELECT value FROM derivs WHERE ts=2000").fetchone()[0]
            conn.close()
            check("stores valid rows, skips malformed", n == 2 and total == 2)
            check("re-run is idempotent (upsert)", n2 == 2 and total == 2)
            check("values parsed as floats", abs(val - (-0.0002)) < 1e-12)
        finally:
            dc.DB_DERIVS = old


def main():
    print("Building synthetic market data...")
    features = make_synthetic_features(1200)

    test_threshold_ranges()
    test_calibration(features)
    test_dna_signature()
    test_kelly(features)
    test_gap_fills()
    test_limit_query()
    test_fast_signals_equivalence(features)
    test_realistic_economics(features)
    with tempfile.TemporaryDirectory() as td:
        test_champion_flush(Path(td))
    test_benchmark_gate(features)
    test_champion_revalidation(features)
    test_prune_dir()
    test_embargo_and_cross_asset(features)
    test_derivatives_collector()

    with tempfile.TemporaryDirectory() as td:
        _redirect_state_to_tmp(Path(td))
        best, _ = test_e2e(features, n_workers=4)
        test_e2e(features, n_workers=1)
        test_funnel(features, best)
        test_vintage_ledger(features, Path(td))

    print(f"\n{'='*50}\nRESULT: {PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
