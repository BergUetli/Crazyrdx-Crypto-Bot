#!/usr/bin/env python3
"""
research_agent.py — autonomous strategy research for the evolution engine.

Every few days: builds a compact dossier of the system's current state
(grammar, forward-feedback evidence, coverage gaps, champion sketches), asks
an LLM (Hermes CLI, free-tier model first) for NEW strategy hypotheses, and:

  - Hypotheses EXPRESSIBLE in the existing genome grammar are strictly
    validated and written to evolution/research_seeds.json (max 3). The
    runner injects up to 2 per cycle as ordinary seeds — they face the SAME
    9-gate exam, ledger, and paper stages as everything else. Research gets
    zero shortcuts.
  - Ideas REQUIRING ENGINE MODIFICATIONS are appended to
    docs/research_proposals.md for HUMAN review (per governance: engine
    changes go through the humans).

LLM output is untrusted input: strict schema validation, no code execution,
unknown indicators/logics rejected, everything capped.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SIM = Path(__file__).resolve().parent
sys.path.insert(0, str(SIM))

SEEDS_JSON = SIM / "evolution" / "research_seeds.json"
PROPOSALS_MD = SIM.parent / "docs" / "research_proposals.md"
MAX_SEEDS = 3
MAX_PROPOSALS = 2


def build_dossier() -> str:
    from evolution.genome import INDICATORS, LOGIC_OPS, EXIT_TYPES
    parts = [
        "SYSTEM: evolutionary crypto strategy search, SOL/USDC 1h bars, "
        "long-only spot, costs ~2.2bps+`$0.03`/side.",
        f"GRAMMAR indicators: {', '.join(INDICATORS)}",
        f"logics: {', '.join(l for l in LOGIC_OPS if l != 'RANDOM')} "
        "(KOFN = k-of-n voting; conditions may also be ratio/diff of two "
        "indicators with quantile thresholds 0.05-0.95)",
        f"exits: {', '.join(EXIT_TYPES)}",
    ]
    try:
        from evolution.forward_feedback import load
        fb = load()
        parts.append(f"FORWARD EVIDENCE (real out-of-sample): logic tilts "
                     f"{fb.get('logic_weights')} | baseline random pnl "
                     f"{fb.get('random_baseline_pnl30')}/30d")
    except Exception:
        pass
    try:
        from evolution.strategy_log import indicator_usage
        u = indicator_usage(14.0)
        from evolution.genome import INDICATORS as _I
        never = [i for i in _I if u.get(i, 0) == 0][:12]
        if never:
            parts.append(f"UNDER-EXPLORED indicators: {', '.join(never)}")
    except Exception:
        pass
    try:
        champs = json.loads((SIM / "evolution" / "champions.json").read_text())
        sketches = []
        for c in champs.get("champions", [])[:4]:
            g = c.get("genome") or {}
            sketches.append(f"{g.get('entry_logic')}("
                            + ",".join(x.get("indicator", "?") for x in
                                       g.get("entry_conditions", [])[:3]) + ")")
        parts.append("CURRENT CHAMPIONS: " + "; ".join(sketches))
    except Exception:
        pass
    return "\n".join(parts)


PROMPT_TEMPLATE = """You are the research analyst for an automated trading-strategy laboratory. Based on the dossier below and your knowledge of documented, LEGAL crypto market effects (funding-rate dynamics, flow, volatility structure, seasonality, cross-asset lead-lag), propose NEW strategy hypotheses this system has likely not tried.

Respond with ONLY a JSON object, no other text:
{{"seeds": [up to {max_seeds} objects: {{"entry_logic": one of the allowed logics, "k_of_n": int (only for KOFN), "entry_conditions": [1-5 of {{"indicator": from the grammar list, "operator": ">"|"<"|">="|"<=", "threshold": number, OPTIONAL "combine": "ratio"|"diff", "indicator_b": from grammar}}], "exit_rules": [1-3 of {{"exit_type": from exits, "value": number}}], "sizing_method": "fixed"|"volatility_scaled", "sizing_base": 0.1-0.5, "rationale": "one sentence"}}],
"proposals": [up to {max_proposals} of {{"title": "...", "description": "engine capability needed and why", "rationale": "..."}}]}}

Rules: seeds must use ONLY grammar indicators/logics/exits; thresholds must be plausible for the indicator; derived (combine) conditions use quantile thresholds in 0.05-0.95. Ideas needing new data or engine features go in proposals, NOT seeds.

DOSSIER:
{dossier}"""


def ask_llm(prompt: str) -> Optional[str]:
    """Hermes CLI, free-tier model first; returns raw text or None."""
    cmds = [
        ["zsh", "-ic",
         f"hermes chat -q {json.dumps(prompt)} -Q --provider omniroute "
         f"-m oc/nemotron-3-ultra-free 2>&1"],
        ["zsh", "-ic", f"hermes chat -q {json.dumps(prompt)} -Q 2>&1"],
    ]
    for cmd in cmds:
        try:
            out = subprocess.run(cmd, capture_output=True, text=True,
                                 timeout=300).stdout
            if out and "{" in out:
                return out
        except Exception:
            continue
    return None


def extract_json(text: str) -> Optional[Dict[str, Any]]:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    for candidate in (m.group(0),):
        try:
            return json.loads(candidate)
        except Exception:
            continue
    return None


def validate_seed(s: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Strict schema check; returns a clean genome dict or None."""
    from evolution.genome import (INDICATORS, LOGIC_OPS, EXIT_TYPES,
                                  COMBINE_OPS, StrategyGenome)
    try:
        logic = s.get("entry_logic")
        if logic not in LOGIC_OPS or logic == "RANDOM":
            return None
        conds = s.get("entry_conditions") or []
        if not 1 <= len(conds) <= 5:
            return None
        clean_conds = []
        for c in conds:
            ind = c.get("indicator")
            if ind not in INDICATORS:
                return None
            op = c.get("operator")
            if op not in (">", "<", ">=", "<="):
                return None
            thr = float(c.get("threshold"))
            combine = c.get("combine") or ""
            ind_b = c.get("indicator_b") or ""
            if combine:
                if combine not in COMBINE_OPS or ind_b not in INDICATORS:
                    return None
                if not 0.05 <= thr <= 0.95:
                    return None
            clean_conds.append({"indicator": ind, "operator": op,
                                "threshold": thr, "combine": combine,
                                "indicator_b": ind_b})
        exits = []
        for e in (s.get("exit_rules") or [])[:3]:
            et = e.get("exit_type")
            if et not in EXIT_TYPES:
                return None
            exits.append({"exit_type": et, "value": float(e.get("value"))})
        if not exits:
            return None
        g = {
            "entry_logic": logic,
            "k_of_n": int(s.get("k_of_n") or 2),
            "entry_conditions": clean_conds,
            "exit_rules": exits,
            "sizing_method": s.get("sizing_method")
            if s.get("sizing_method") in ("fixed", "volatility_scaled")
            else "fixed",
            "sizing_base": min(0.5, max(0.1, float(s.get("sizing_base") or 0.25))),
            "genome_id": f"research_{int(time.time())}_{logic.lower()}",
        }
        StrategyGenome.from_dict(g)  # must construct cleanly
        return g
    except Exception:
        return None


def process_response(raw: str) -> Tuple[List[Dict], List[Dict]]:
    d = extract_json(raw) or {}
    seeds = []
    for s in (d.get("seeds") or [])[:MAX_SEEDS]:
        v = validate_seed(s) if isinstance(s, dict) else None
        if v:
            v["rationale"] = str(s.get("rationale") or "")[:300]
            seeds.append(v)
    proposals = []
    for p in (d.get("proposals") or [])[:MAX_PROPOSALS]:
        if isinstance(p, dict) and p.get("title"):
            proposals.append({
                "title": str(p["title"])[:120],
                "description": str(p.get("description") or "")[:800],
                "rationale": str(p.get("rationale") or "")[:400],
            })
    return seeds, proposals


def publish(seeds: List[Dict], proposals: List[Dict]) -> None:
    if seeds:
        tmp = SEEDS_JSON.with_suffix(".tmp")
        tmp.write_text(json.dumps(
            {"updated_ts": time.time(), "seeds": seeds}, indent=2))
        tmp.replace(SEEDS_JSON)
    if proposals:
        PROPOSALS_MD.parent.mkdir(exist_ok=True)
        stamp = time.strftime("%Y-%m-%d")
        block = "".join(
            f"\n## [{stamp}] {p['title']} — NEEDS HUMAN REVIEW\n\n"
            f"{p['description']}\n\nRationale: {p['rationale']}\n"
            for p in proposals)
        with open(PROPOSALS_MD, "a") as f:
            if PROPOSALS_MD.stat().st_size == 0 if PROPOSALS_MD.exists() else True:
                pass
            f.write(block)


def main() -> int:
    dossier = build_dossier()
    prompt = PROMPT_TEMPLATE.format(
        max_seeds=MAX_SEEDS, max_proposals=MAX_PROPOSALS, dossier=dossier)
    raw = ask_llm(prompt)
    if not raw:
        print("research agent: no LLM response (skipped this run)")
        return 0
    seeds, proposals = process_response(raw)
    publish(seeds, proposals)
    print(f"research agent: {len(seeds)} validated seeds "
          f"(face the full exam), {len(proposals)} proposals for human review")
    return 0


if __name__ == "__main__":
    sys.exit(main())
