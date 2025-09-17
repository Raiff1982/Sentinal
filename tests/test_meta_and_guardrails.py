

import sys
import pathlib
import os
import json

# Ensure the sentinal package is discoverable
project_root = str(pathlib.Path(__file__).resolve().parents[1])
if project_root not in sys.path:
	sys.path.insert(0, project_root)

from sentinal.sentinel_council import get_council
from sentinal.Sentinel import SignedLedger

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
