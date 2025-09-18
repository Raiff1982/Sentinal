import sys, pathlib; sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import json
import os
import time
from datetime import datetime, timedelta, timezone
import pytest
from typing import Dict, Any

from sentinel_council import get_council, DEFAULT_CONFIG
from Sentinel import SignedLedger, ChallengeHarness, default_challenges

def get_decision(bundle: Dict[str, Any]) -> str:
    """Extract the MetaJudge decision from a council dispatch bundle."""
    for r in bundle.get("reports", []):
        if r.get("agent") == "MetaJudge":
            return r.get("details", {}).get("decision", "")
    return ""

def get_agent_report(bundle: Dict[str, Any], agent_name: str) -> Dict[str, Any]:
    """Extract a specific agent's report from a council dispatch bundle."""
    for r in bundle.get("reports", []):
        if r.get("agent") == agent_name:
            return r
    return {}

def test_proceed_on_calm():
    """Test that council allows proceeding when all signals are calm."""
    c = get_council()
    inp = {
        "text": "be fair and honest",
        "intent": "",
        "_signals": {
            "bio": {"stress": 0.1},
            "env": {"context_risk": 0.1}
        },
        "timescale": 0.1
    }
    out = c.dispatch(inp)
    assert get_decision(out) == "PROCEED"

def test_caution_on_stress_cap():
    """Test that council suggests caution when stress approaches cap."""
    c = get_council()
    inp = {
        "text": "steady",
        "intent": "",
        "_signals": {
            "bio": {"stress": 0.72},
            "env": {"context_risk": 0.2}
        },
        "timescale": 0.2
    }
    out = c.dispatch(inp)
    assert get_decision(out) == "PROCEED_WITH_CAUTION"

def test_block_on_extreme_risk():
    """Test that council blocks when environmental risk is extreme."""
    c = get_council()
    # Env risk > 0.9 should BLOCK
    inp = {
        "text": "ship now",
        "intent": "proceed fast",
        "_signals": {
            "bio": {"stress": 0.4},
            "env": {
                "context_risk": 0.95,
                "incident_sev": 0.95
            }
        },
        "timescale": 0.4
    }
    out = c.dispatch(inp)
    assert get_decision(out) == "BLOCK"

def test_block_on_conflict_and_stress():
    """Test that high stress combined with conflicting intent triggers blocking."""
    c = get_council()
    inp = {
        "text": "ship now",
        "intent": "proceed fast",
        "_signals": {
            "bio": {"stress": 0.9},
            "env": {"context_risk": 0.7}
        },
        "timescale": 0.4
    }
    out = c.dispatch(inp)
    decision = get_decision(out)
    assert decision in ("BLOCK", "PROCEED_WITH_CAUTION")
    assert decision != "PROCEED"

def test_memory_integrity_decay():
    """Test that memory entries lose integrity over time."""
    c = get_council()
    now = datetime.now(timezone.utc)
    
    # Write test data with short TTL
    c.memory.write("test_key", "test_value", weight=1.0, ttl_secs=2)
    
    # Check immediate integrity
    audit1 = c.memory.audit()
    assert "test_key" in [k for k, v in audit1.items()]
    
    # Wait for partial decay
    time.sleep(1.5)
    audit2 = c.memory.audit()
    integrity1 = next((v["integrity"] for k, v in audit1.items() if "test_key" in k), 1.0)
    integrity2 = next((v["integrity"] for k, v in audit2.items() if "test_key" in k), 0.0)
    assert integrity2 < integrity1

def test_memory_purge():
    """Test that expired memory entries are properly purged."""
    c = get_council()
    
    # Write test data with very short TTL
    c.memory.write("test_expire", "test_value", ttl_secs=1)
    assert c.memory.read("test_expire") == "test_value"
    
    # Wait for expiration
    time.sleep(1.5)
    c.memory.purge_expired()
    assert c.memory.read("test_expire") is None

def test_custom_policy_config():
    """Test that custom policy configuration is properly applied."""
    custom_config = DEFAULT_CONFIG.copy()
    custom_config["risk_threshold"] = 0.99
    
    c = get_council(config=custom_config)
    inp = {
        "text": "proceed",
        "_signals": {
            "bio": {"stress": 0.3},
            "env": {"context_risk": 0.55}  # Just over new cap
        }
    }
    out = c.dispatch(inp)
    assert get_decision(out) in ("BLOCK", "PROCEED_WITH_CAUTION")

def test_memory_config():
    """Test that memory configuration is properly applied."""
    memory_config = {"max_entries": 5, "default_ttl_secs": 3600}
    c = get_council(memory_config=memory_config)
    
    # Fill memory to max
    for i in range(7):
        c.memory.write(f"key_{i}", f"value_{i}")
    
    # Should only keep 5 most recent
    assert len(c.memory.store) == 5
    assert c.memory.read("key_6") is not None
    assert c.memory.read("key_0") is None

def test_sanitizer_max_length():
    """Test that input sanitization properly handles long inputs."""
    c = get_council()
    long_text = "a" * (10_001)  # Exceeds MAX_INPUT_LENGTH
    inp = {"text": long_text}
    out = c.dispatch(inp)
    input_audit = out.get("input_audit", {})
    assert "input_too_long" in input_audit.get("issues", [])

def test_control_char_sanitization():
    """Test that control characters are properly detected and sanitized."""
    c = get_council()
    text_with_control = "test\x00input\x1Fwith\x7Fcontrol"
    inp = {"text": text_with_control}
    out = c.dispatch(inp)
    input_audit = out.get("input_audit", {})
    assert "control_char" in input_audit.get("issues", [])

def test_timescale_coordination():
    """Test that timescale coordination properly weighs short/mid/long term signals."""
    c = get_council()
    
    # Simulate increasing stress over time
    decisions = []
    for stress in [0.3, 0.5, 0.7, 0.9]:
        inp = {
            "text": "continue operation",
            "_signals": {"bio": {"stress": stress}},
            "_last_decision": decisions[-1] if decisions else "PROCEED"
        }
        out = c.dispatch(inp)
        decisions.append(get_decision(out))
    
    # Should see escalating responses
    assert "PROCEED" in decisions[:2]  # Early decisions
    assert "BLOCK" in decisions[-2:]   # Later decisions

def test_council_error_handling():
    """Test council's handling of agent failures."""
    c = get_council(per_agent_timeout_sec=0.001)  # Very short timeout
    
    # Create input that will cause processing delay
    inp = {"text": "a" * 1000000}  # Large input to force timeout
    
    out = c.dispatch(inp)
    # Some agents should time out but council should still produce a decision
    assert any(not r.get("ok", True) for r in out.get("reports", []))
    assert get_decision(out) is not None

def test_ledger_sign_and_verify(tmp_path):
    """Test ledger signing and verification."""
    led = SignedLedger(dirpath=str(tmp_path))
    dummy_input = {"hello": "world"}
    dummy_bundle = {
        "reports": [{
            "agent": "MetaJudge",
            "details": {"decision": "PROCEED"},
            "ok": True
        }],
        "explainability_graph": {"edges": []}
    }
    led.append(dummy_input, dummy_bundle, {"policy": "demo"})
    rep = led.verify_all()
    assert rep["ok"] is True

def test_challenge_bank_pass():
    """Test that council passes all default challenges."""
    c = get_council()
    harness = ChallengeHarness(c)
    bank = default_challenges()
    res = harness.run_all(bank)
    assert res["passed"] == res["total"]

def test_fusion_agent_reliability():
    """Test that fusion agents properly combine and weigh their inputs."""
    c = get_council()
    
    # Test BiofeedbackAgent fusion
    inp = {
        "_signals": {
            "bio": {
                "heart_rate": 120,  # High
                "hrv": 20,          # Low
                "gsr": 15,          # High
                "voice_tension": 0.8 # High
            }
        }
    }
    out = c.dispatch(inp)
    bio_report = get_agent_report(out, "BiofeedbackAgent")
    assert bio_report.get("severity", 0) > 0.7  # High stress should be detected

def test_memory_overflow_handling():
    """Test that memory properly handles overflow by purging low-integrity entries."""
    memory_config = {"max_entries": 3, "default_ttl_secs": 3600}
    c = get_council(memory_config=memory_config)
    
    # Add entries with varying weights and entropies
    c.memory.write("high_weight", "value1", weight=1.0, entropy=0.1)
    c.memory.write("med_weight", "value2", weight=0.5, entropy=0.5)
    c.memory.write("low_weight", "value3", weight=0.1, entropy=0.9)
    
    # This should trigger overflow handling
    c.memory.write("overflow", "value4", weight=0.8, entropy=0.2)
    
    # Low-weight entry should be purged
    assert c.memory.read("low_weight") is None
    assert c.memory.read("high_weight") is not None

def test_memory_persistence(tmp_path):
    """Test that memory state persists between council instances."""
    persistence_path = os.path.join(tmp_path, "memory.json")
    
    # Create first council instance with persistence
    c1 = get_council(persistence_path=persistence_path)
    
    # Write test data
    key = "persist_test"
    value = {"test": "data"}
    c1.memory.write(key, value, weight=1.0, entropy=0.1)
    assert c1.memory.read(key) == value
    
    # Create second council instance with same persistence path
    c2 = get_council(persistence_path=persistence_path)
    
    # Verify data loaded
    assert c2.memory.read(key) == value

def test_memory_persistence_version_check(tmp_path):
    """Test that memory persistence handles version mismatches gracefully."""
    persistence_path = os.path.join(tmp_path, "memory.json")
    
    # Create invalid version snapshot
    snapshot = {
        "version": "0.0.1",  # Invalid version
        "store": {},
        "expiration_heap": [],
        "max_entries": 1000,
        "default_ttl_secs": 3600
    }
    os.makedirs(os.path.dirname(persistence_path), exist_ok=True)
    with open(persistence_path, "w") as f:
        json.dump(snapshot, f)
    
    # Should handle version mismatch gracefully
    c = get_council(persistence_path=persistence_path)
    assert len(c.memory.store) == 0

def test_memory_persistence_corruption_recovery(tmp_path):
    """Test that memory persistence recovers from corrupted state."""
    persistence_path = os.path.join(tmp_path, "memory.json")
    
    # Write corrupted data
    with open(persistence_path, "w") as f:
        f.write("invalid json {")
    
    # Should handle corruption gracefully
    c = get_council(persistence_path=persistence_path)
    assert len(c.memory.store) == 0
    
    # Should be able to write new data
    c.memory.write("test", "value")
    assert c.memory.read("test") == "value"

def test_memory_persistence_maintains_integrity(tmp_path):
    """Test that persisted memory maintains integrity values."""
    persistence_path = os.path.join(tmp_path, "memory.json")
    
    # Create council and write data
    c1 = get_council(persistence_path=persistence_path)
    c1.memory.write("test", "value", weight=0.8, entropy=0.2)
    
    # Get original integrity
    audit1 = c1.memory.audit()
    integrity1 = next((v["integrity"] for k, v in audit1.items()), 0.0)
    
    # Create new council with same persistence
    c2 = get_council(persistence_path=persistence_path)
    audit2 = c2.memory.audit()
    integrity2 = next((v["integrity"] for k, v in audit2.items()), 0.0)
    
    # Integrity should be maintained
    assert round(integrity2, 6) == round(integrity1, 6)

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
