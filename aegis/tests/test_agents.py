"""
Tests for agent implementations.
"""
import pytest
from typing import Dict, Any

from ..sentinel_council import (
    AegisAgent,
    ShortTermAgent,
    MidTermAgent,
    LongTermArchivistAgent,
    TimeScaleCoordinator,
    MetaJudgeAgent
)

class MockMemory:
    """Mock memory implementation for testing."""
    def __init__(self):
        self.store = {}
        
    def write(self, key: str, value: Any, **kwargs) -> str:
        self.store[key] = value
        return key
        
    def read(self, key: str) -> Any:
        return self.store.get(key)

def test_short_term_agent(sample_input: Dict[str, Any]):
    """Test ShortTermAgent analysis."""
    memory = MockMemory()
    agent = ShortTermAgent("ShortTerm", memory)
    result = agent.analyze(sample_input)
    
    # Verify structure
    assert "summary" in result
    assert "influence" in result
    assert "reliability" in result
    assert "severity" in result
    assert "details" in result
    assert "explain_edges" in result
    assert "ok" in result
    
    # Verify ranges
    assert 0 <= result["influence"] <= 1
    assert 0 <= result["reliability"] <= 1
    assert 0 <= result["severity"] <= 1
    
def test_meta_judge(sample_input: Dict[str, Any]):
    """Test MetaJudgeAgent decision-making."""
    memory = MockMemory()
    agent = MetaJudgeAgent("MetaJudge", memory)
    
    # Add agent reports
    reports = [
        {
            "agent": "Agent1",
            "ok": True,
            "influence": 0.3,
            "reliability": 0.8,
            "severity": 0.5
        },
        {
            "agent": "Agent2",
            "ok": True,
            "influence": 0.4,
            "reliability": 0.9,
            "severity": 0.7
        }
    ]
    input_with_reports = {**sample_input, "_agent_reports": reports}
    
    result = agent.analyze(input_with_reports)
    
    # Verify structure
    assert "decision" in result["details"]
    assert "agent_metrics" in result["details"]
    assert "weighted_severity" in result["details"]
    
    # Verify decision
    decision = result["details"]["decision"]
    assert decision in ["PROCEED", "PROCEED_WITH_CAUTION", "BLOCK"]
    
    # Verify severity calculation
    severity = result["details"]["weighted_severity"]
    assert 0 <= severity <= 1
    
    # Verify metrics processing
    metrics = result["details"]["agent_metrics"]
    assert len(metrics) == len(reports)
    
def test_timescale_coordinator(sample_input: Dict[str, Any]):
    """Test TimeScaleCoordinator fusion."""
    memory = MockMemory()
    coordinator = TimeScaleCoordinator("TimeScale", memory)
    
    # Add reports from different timescales
    reports = [
        {
            "agent": "ShortTermAgent",
            "severity": 0.3,
            "reliability": 0.9
        },
        {
            "agent": "MidTermAgent",
            "severity": 0.5,
            "reliability": 0.8
        },
        {
            "agent": "LongTermArchivist",
            "severity": 0.4,
            "reliability": 0.7
        }
    ]
    input_with_reports = {**sample_input, "_agent_reports": reports}
    
    result = coordinator.analyze(input_with_reports)
    
    # Verify fusion results
    assert "short_term_severity" in result["details"]
    assert "mid_term_severity" in result["details"]
    assert "long_term_severity" in result["details"]
    assert "patterns" in result["details"]
    
    # Verify severity relationships
    severities = [
        result["details"]["short_term_severity"],
        result["details"]["mid_term_severity"],
        result["details"]["long_term_severity"]
    ]
    assert all(0 <= s <= 1 for s in severities)