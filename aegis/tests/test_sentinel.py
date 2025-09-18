"""
Tests for core Sentinel functionality.
"""
import pytest
from typing import Dict, Any

from ..sentinel import Sentinel
from ..sentinel_council import AegisCouncil

def test_evaluate_basic(
    sentinel_jsonl: Sentinel,
    sentinel_nexus: Sentinel,
    sample_input: Dict[str, Any]
):
    """Test basic evaluation flow."""
    # Test both backends
    for sentinel in [sentinel_jsonl, sentinel_nexus]:
        result = sentinel.evaluate(sample_input)
        
        # Verify structure
        assert "decision" in result
        assert "explanation" in result
        assert "graph" in result
        assert "policy" in result
        
        # Verify decision
        assert result["decision"] in ["PROCEED", "PROCEED_WITH_CAUTION", "BLOCK"]
        
        # Verify explanations
        assert isinstance(result["explanation"], list)
        assert len(result["explanation"]) > 0
        
        # Verify graph
        assert "nodes" in result["graph"]
        assert "edges" in result["graph"]
        
def test_challenge_scenarios(
    sentinel_jsonl: Sentinel,
    sentinel_nexus: Sentinel,
    challenge_scenarios: List[Dict[str, Any]]
):
    """Test decision-making with challenge scenarios."""
    # Test both backends
    for sentinel in [sentinel_jsonl, sentinel_nexus]:
        results = sentinel.challenge(challenge_scenarios)
        
        # Verify structure
        assert "passed" in results
        assert "total" in results
        assert "results" in results
        
        # Verify counts
        assert results["total"] == len(challenge_scenarios)
        assert results["passed"] > 0  # Should pass at least some
        
        # Check individual results
        for result in results["results"]:
            assert "name" in result
            assert "expected" in result
            assert "decision" in result
            assert "ok" in result
            
def test_evolution(
    sentinel_jsonl: Sentinel,
    sentinel_nexus: Sentinel
):
    """Test policy evolution."""
    performance_data = [
        {
            "decision": "PROCEED",
            "outcome": "success",
            "metrics": {"accuracy": 0.95}
        },
        {
            "decision": "BLOCK",
            "outcome": "success", 
            "metrics": {"accuracy": 0.9}
        }
    ]
    
    # Test both backends
    for sentinel in [sentinel_jsonl, sentinel_nexus]:
        # Get initial policy
        initial_policy = sentinel.meta_genes.to_dict()
        
        # Evolve
        sentinel.evolve(performance_data)
        
        # Get evolved policy
        evolved_policy = sentinel.meta_genes.to_dict()
        
        # Verify changes
        assert evolved_policy != initial_policy  # Should adapt
        assert all(0 <= v <= 1 for v in evolved_policy.values())  # Valid ranges