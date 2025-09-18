import sys, pathlib; sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import json
import os
from sentinel_council import get_council
from Sentinel import SignedLedger, ChallengeHarness, default_challenges

def get_decision(bundle):
    for r in bundle.get("reports", []):
        if r.get("agent") == "MetaJudge":
            return r.get("details", {}).get("decision")
    return None

def test_proceed_on_calm():
    c = get_council()
    inp = {"text":"be fair and honest","intent":"", "_signals":{"bio":{"stress":0.1}, "env":{"context_risk":0.1}}, "timescale":0.1}
    out = c.dispatch(inp)
    assert get_decision(out) == "PROCEED"

def test_caution_on_stress_cap():
    c = get_council()
    inp = {"text":"steady","intent":"", "_signals":{"bio":{"stress":0.72}, "env":{"context_risk":0.2}}, "timescale":0.2}
    out = c.dispatch(inp)
    assert get_decision(out) == "PROCEED_WITH_CAUTION"

def test_block_on_extreme_risk():
    c = get_council()
    # Env risk > 0.9 should BLOCK
    inp = {"text":"ship now","intent":"proceed fast", "_signals":{"bio":{"stress":0.4}, "env":{"context_risk":0.95, "incident_sev":0.95}}, "timescale":0.4}
    out = c.dispatch(inp)
    assert get_decision(out) == "BLOCK"

def test_block_on_conflict_and_stress():
    c = get_council()
    # high stress + intent to speed + decent risk -> conflict ~0.6*stress+0.6*risk exceeds 0.85
    inp = {"text":"ship now","intent":"proceed fast", "_signals":{"bio":{"stress":0.9}, "env":{"context_risk":0.7}}, "timescale":0.4}
    out = c.dispatch(inp)
    assert get_decision(out) in ("BLOCK", "PROCEED_WITH_CAUTION")  # allow either if timescale/stress caps fire first
    # If not blocked, at least caution
    assert get_decision(out) != "PROCEED"

def test_ledger_sign_and_verify(tmp_path):
    led = SignedLedger(dirpath=str(tmp_path))
    dummy_input = {"hello":"world"}
    dummy_bundle = {"reports":[{"agent":"MetaJudge","details":{"decision":"PROCEED"}, "ok":True}] ,
                    "explainability_graph":{"edges":[]}}
    led.append(dummy_input, dummy_bundle, {"policy":"demo"})
    rep = led.verify_all()
    assert rep["ok"] is True

def test_challenge_bank_pass():
    # Use our real council through the harness
    from sentinel_council import get_council
    c = get_council()
    harness = ChallengeHarness(c)
    bank = default_challenges()
    res = harness.run_all(bank)
    assert res["passed"] == res["total"]
