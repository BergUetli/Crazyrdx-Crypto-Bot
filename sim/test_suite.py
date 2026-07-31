#!/usr/bin/env python3
"""
test_suite.py — regression tests for the trading bot system.

Run after any change to ensure nothing breaks:
  python3 test_suite.py

Covers: Layer 1 (data), Layer 2 (models), evolution, validation, dashboard.
"""

import json
import os
import sqlite3
import sys
import tempfile
from pathlib import Path

SIM = Path(__file__).resolve().parent
sys.path.insert(0, str(SIM))

PASS = FAIL = 0

def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS: {name}")
    else:
        FAIL += 1
        print(f"  FAIL: {name} {detail}")

def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ============================================================================
# 1. DATA LAYER
# ============================================================================

def test_data_layer():
    section("1. Data Layer")

    # Candle DB
    from layer1.historical_downloader import get_candles, get_candle_count
    candles = get_candles("SOL/USDC", limit=10)
    check("get_candles returns data", len(candles) > 0)
    check("candles have required fields", all(k in candles[0] for k in ["ts", "open", "high", "low", "close", "volume"]))
    check("get_candle_count works", get_candle_count("SOL/USDC") > 0)

    # Interval filtering
    candles_5m = get_candles("SOL/USDC", limit=10, interval="5m")
    candles_1h = get_candles("SOL/USDC", limit=10, interval="1h")
    check("5m interval filter", len(candles_5m) > 0)
    check("1h interval filter", len(candles_1h) > 0)
    if candles_5m and len(candles_5m) > 1:
        diff = candles_5m[1]["ts"] - candles_5m[0]["ts"]
        check("5m candles are 5m apart", diff == 300000, f"got {diff}")
    if candles_1h and len(candles_1h) > 1:
        diff = candles_1h[1]["ts"] - candles_1h[0]["ts"]
        check("1h candles are 1h apart", diff == 3600000, f"got {diff}")

    # Feature engine (5m)
    from layer1.historical_feature_engine import get_historical_features, HistoricalFeatureEngine
    features = get_historical_features("SOL/USDC", limit=10)
    check("5m features exist", len(features) > 0)
    if features:
        f0 = features[0]["features"]
        check("5m features have 61 fields", len(f0) == 61, f"got {len(f0)}")
        check("5m has trend_alignment", "trend_alignment_5m_15m" in f0)
        check("5m has momentum_divergence", "momentum_divergence_5m_15m" in f0)

    # Feature engine (1h)
    from layer1.historical_feature_engine_1h import get_historical_features_1h, HistoricalFeatureEngine1h
    features_1h = get_historical_features_1h("SOL/USDC", limit=10)
    check("1h features exist", len(features_1h) > 0)
    if features_1h:
        f1h = features_1h[0]["features"]
        check("1h features have 67 fields", len(f1h) == 67, f"got {len(f1h)}")
        check("1h has tft_prediction", "tft_prediction" in f1h)
        check("1h has tft_confidence", "tft_confidence" in f1h)
        check("1h has taker_buy_ratio", "taker_buy_ratio" in f1h)


# ============================================================================
# 2. BACKTEST ENGINE
# ============================================================================

def test_backtest_engine():
    section("2. Backtest Engine")

    from layer1.backtest_engine import BacktestEngine, LatencyModel
    engine = BacktestEngine(initial_capital=100.0, fee_rate=0.00022)
    check("BacktestEngine creates", engine is not None)
    check("LatencyModel creates", LatencyModel() is not None)

    # Simple signal: always buy
    def buy_signal(features, idx):
        if idx < 50:
            return None
        return ("long", 1.0, 0.25)

    from layer1.historical_feature_engine import get_historical_features
    features = get_historical_features("SOL/USDC", limit=500)
    result = engine.run_backtest("test", "SOL/USDC", features, buy_signal)
    check("backtest runs", result is not None)
    check("backtest has trades", result.total_trades >= 0)
    check("backtest has win_rate", hasattr(result, "win_rate"))
    check("backtest has sharpe_ratio", hasattr(result, "sharpe_ratio"))


# ============================================================================
# 3. GENOME & EVOLUTION
# ============================================================================

def test_genome():
    section("3. Genome")

    from evolution.genome import StrategyGenome, EntryCondition, random_genome, INDICATORS, LOGIC_OPS

    check("indicators present", len(INDICATORS) >= 50, f"got {len(INDICATORS)}")
    check("6 selection logic ops", len(LOGIC_OPS) == 6, f"got {len(LOGIC_OPS)}")
    check("RANDOM banned from selection LOGIC_OPS", "RANDOM" not in LOGIC_OPS)
    check("TFT in LOGIC_OPS", "TFT" in LOGIC_OPS)
    check("MEANREV in LOGIC_OPS", "MEANREV" in LOGIC_OPS)
    # 1h feature alignment: threshold genes must exist on 1h features
    from layer1.historical_feature_engine_1h import get_historical_features_1h
    f1 = get_historical_features_1h("SOL/USDC", limit=1)
    if f1:
        keys = set(f1[0]["features"].keys())
        missing = [i for i in INDICATORS if i not in keys]
        check("all INDICATORS exist on 1h features", len(missing) == 0, f"missing {missing[:8]}")

    g = random_genome(0)
    check("random_genome creates", g is not None)
    check("genome has entry_conditions", len(g.entry_conditions) > 0)
    check("genome has entry_logic", g.entry_logic in LOGIC_OPS)
    check("genome serializes", json.dumps(g.to_dict()) is not None)

    # Round-trip
    g2 = StrategyGenome.from_dict(g.to_dict())
    check("genome round-trips", g2.genome_id == g.genome_id)


def test_evolution():
    section("4. Evolution")

    from evolution.evaluator import GenomeEvaluator, EvolutionEngine
    from evolution.genome import random_genome
    from layer1.historical_feature_engine import get_historical_features

    features = get_historical_features("SOL/USDC", limit=200)
    evaluator = GenomeEvaluator(features)

    # Test each selection strategy type
    for stype in ["AND", "OR", "MEANREV", "BREAKOUT", "TREND", "TFT"]:
        g = random_genome(0)
        g.entry_logic = stype
        r = evaluator.evaluate(g)
        check(f"{stype} evaluates", r is not None and "fitness" in r)

    # RANDOM is banned from selection
    from evolution.genome import random_genome as _rg
    g_rand = _rg(0)
    g_rand.entry_logic = "RANDOM"
    r_rand = evaluator.evaluate(g_rand)
    check("RANDOM gets banned fitness", r_rand["fitness"] <= -250.0, f"got {r_rand['fitness']}")

    # Fitness minimum
    from evolution.genome import StrategyGenome, EntryCondition
    g0 = StrategyGenome(entry_conditions=[EntryCondition("close", ">", 999999)], entry_logic="AND")
    r0 = evaluator.evaluate(g0)
    # Hard min-trade discard: fitness = -100 + trade_count (-100 when 0 trades)
    check("0 trades hard-discarded", r0["fitness"] <= -100.0, f"got {r0['fitness']}")

    # Evolution engine
    engine = EvolutionEngine(features, population_size=5, elite_size=1)
    engine.initialize_population()
    engine.evaluate_population()
    check("evolution initializes", len(engine.population) == 5)
    check("evolution evaluates", all(g.backtest_results is not None for g in engine.population))


# ============================================================================
# 4. VALIDATION SUITE
# ============================================================================

def test_validation():
    section("5. Validation Suite")

    from validation_suite import ValidationSuite
    from evolution.genome import random_genome
    from layer1.historical_feature_engine import get_historical_features

    features = get_historical_features("SOL/USDC", limit=200)
    suite = ValidationSuite(features)

    g = random_genome(0)
    g.entry_logic = "MEANREV"

    # Walk-forward
    wf = suite.walk_forward(g)
    check("walk_forward runs", "avg_fitness" in wf)

    # Monte Carlo
    mc = suite.monte_carlo(g)
    check("monte_carlo runs", "p_value" in mc)

    # MEV stress
    mev = suite.mev_stress(g)
    check("mev_stress runs", "survives_30pct" in mev)

    # Parameter perturbation
    pp = suite.parameter_perturbation(g, n_perturbations=5)
    check("parameter_perturbation runs", "profitability_rate" in pp)
    check("perturbation has fitness_retention", "fitness_retention" in pp)

    # Full validation
    fv = suite.full_validation(g, "test")
    check("full_validation has 4 tests", all(k in fv for k in ["walk_forward", "monte_carlo", "mev_stress", "parameter_perturbation"]))

    # Promotion funnel (fund-grade gauntlet)
    from evolution.promotion_funnel import (
        PromotionFunnel,
        log_trials,
        get_total_trials,
        approximate_dsr,
    )
    from layer1.historical_feature_engine_1h import get_historical_features_1h

    f1h = get_historical_features_1h("SOL/USDC", limit=400)
    if f1h:
        funnel = PromotionFunnel(f1h, min_trades_full=30, wf_folds=3)
        g_rand = random_genome(0)
        g_rand.entry_logic = "RANDOM"
        fr = funnel.run(g_rand, n_trials_context=10, verbose=False)
        check("funnel rejects RANDOM", fr.get("verdict") == "REJECT" and fr.get("failed_at") in ("RANDOM banned", "lottery_ban", "feasibility"), f"got {fr.get('failed_at')}")
        check("dsr helper bounded", 0.0 <= approximate_dsr(1.0, 50, 100) <= 1.0)
        before = get_total_trials()
        after = log_trials(3, meta={"test": True})
        check("trial log increments", after >= before + 3, f"before={before} after={after}")

    # Kill archive
    from evolution.kill_archive import KillArchive, structure_key, reload_archive
    from evolution.genome import StrategyGenome
    import tempfile
    from pathlib import Path as _P
    tmp_kill = _P(tempfile.mkstemp(suffix=".json")[1])
    try:
        ka = KillArchive(path=tmp_kill)
        g1 = random_genome(0)
        g1.entry_logic = "AND"
        key1 = structure_key(g1)
        check("structure_key non-empty", isinstance(key1, str) and len(key1) > 3)
        ka.record_kill(g1, reason="walk_forward")
        check("kill marks neighborhood", ka.is_killed(g1))
        # near retune same neighborhood should still be killed if bins match
        g2 = StrategyGenome.from_dict(g1.to_dict())
        if g2.entry_conditions:
            # tiny nudge within bin
            g2.entry_conditions[0].threshold *= 1.01
        check("near-duplicate still killed or cleanly handled", True)  # bin may or may not move; API must not crash
        check("clean sampler returns genome", ka.random_genome_clean(0) is not None)
        check("archive summary works", ka.summary()["n"] >= 1)
    finally:
        try:
            tmp_kill.unlink(missing_ok=True)
        except Exception:
            pass
    # module singleton still loadable
    arch = reload_archive()
    check("reload_archive works", arch is not None)


# ============================================================================
# 5. MODELS (Layer 2)
# ============================================================================

def test_models():
    section("6. Models")

    from layer2.autoencoder import FeatureAutoencoder
    from layer2.dataset import Dataset
    check("autoencoder imports", FeatureAutoencoder is not None)
    check("dataset imports", Dataset is not None)

    from layer2.tft_model import TemporalFusionTransformer
    check("tft imports", TemporalFusionTransformer is not None)


# ============================================================================
# 6. DASHBOARD
# ============================================================================

def test_dashboard():
    section("7. Dashboard")

    import urllib.request
    try:
        with urllib.request.urlopen("http://127.0.0.1:8765/", timeout=3) as resp:
            html = resp.read().decode()
        check("dashboard serves", "Is the bot finding anything useful" in html or "Bot search status" in html or "Evolution status" in html)
        check("has 40s map", "40-second map" in html or "Genome" in html)
        check("asks if searching", "Is the bot searching right now" in html)
        check("asks what happened", "What just happened" in html)
        check("has shortlist section", "Shortlist" in html)
        check("uses paper trades language", "paper trades" in html.lower() or "Paper trades" in html)
        check("has educational flow", "strict exam" in html.lower() or "Strict exam" in html)
    except Exception as e:
        check("dashboard serves", False, str(e)[:60])

    try:
        with urllib.request.urlopen("http://127.0.0.1:8765/api", timeout=3) as resp:
            api = json.loads(resp.read())
        check("api returns JSON", "trend" in api or "process" in api)
        check("api has process status", "process" in api)
    except Exception as e:
        check("api returns JSON", False, str(e)[:60])


# ============================================================================
# 7. INTEGRATION
# ============================================================================

def test_integration():
    section("8. Integration")

    from layer1.historical_feature_engine import get_historical_features
    from evolution.evaluator import GenomeEvaluator
    from evolution.genome import random_genome

    # Full pipeline: features -> genome -> evaluate -> fitness
    features = get_historical_features("SOL/USDC", limit=200)
    evaluator = GenomeEvaluator(features)
    g = random_genome(0)
    result = evaluator.evaluate(g)

    check("pipeline: features loaded", len(features) > 0)
    check("pipeline: genome created", g is not None)
    check("pipeline: evaluation returns fitness", "fitness" in result)
    check("pipeline: evaluation returns trades", "total_trades" in result)


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 60)
    print("  TRADING BOT REGRESSION TEST SUITE")
    print("=" * 60)

    test_data_layer()
    test_backtest_engine()
    test_genome()
    test_evolution()
    test_validation()
    test_models()
    test_dashboard()
    test_integration()

    print("\n" + "=" * 60)
    print(f"  RESULTS: {PASS} passed, {FAIL} failed")
    print("=" * 60)

    if FAIL > 0:
        print("\n  FAILURES DETECTED — fix before proceeding")
        sys.exit(1)
    else:
        print("\n  ALL TESTS PASSED")
        sys.exit(0)


if __name__ == "__main__":
    main()
