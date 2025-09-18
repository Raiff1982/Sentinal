"""
Shared pytest fixtures for Aegis test suite.
"""
import os
import pytest
import tempfile
from typing import Dict, Any, Generator

from .. import sentinel_config
from ..explain import (
    ExplainStore, 
    create_explain_store,
    ExplainEntry
)
from ..sentinel import Sentinel
from ..sentinel_council import AegisCouncil

class MockNexusClient:
    """Mock Nexus client for testing."""
    def __init__(self):
        self.store = {}
        
    def ingest(self, source: str, category: str, data: Dict[str, Any], ttl_sec: float) -> None:
        """Store data in memory."""
        key = f"{source}:{category}"
        self.store[key] = {
            "data": data,
            "ttl_sec": ttl_sec
        }
        
    def query(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Return stored data matching query."""
        results = []
        for key, value in self.store.items():
            source = query.get("source")
            if source and not key.startswith(source):
                continue
            results.append(value)
        return results[:query.get("limit", len(results))]
        
    def cleanup(self, **kwargs) -> int:
        """Simulate cleanup."""
        return 0

@pytest.fixture
def temp_dir() -> Generator[str, None, None]:
    """Provide temporary directory for test data."""
    with tempfile.TemporaryDirectory() as td:
        yield td
        
@pytest.fixture
def jsonl_store(temp_dir: str) -> ExplainStore:
    """Provide JSONL explain store instance."""
    explain_path = os.path.join(temp_dir, "explain")
    return create_explain_store("jsonl", base_path=explain_path)
    
@pytest.fixture
def nexus_store() -> ExplainStore:
    """Provide Nexus explain store instance with mock client."""
    return create_explain_store("nexus", nexus_client=MockNexusClient())
    
@pytest.fixture
def council() -> AegisCouncil:
    """Provide basic council instance."""
    return AegisCouncil()
    
@pytest.fixture
def sentinel_jsonl(temp_dir: str) -> Sentinel:
    """Provide Sentinel instance with JSONL backend."""
    # Override config
    sentinel_config.EXPLAIN_BACKEND = sentinel_config.ExplainBackend.JSONL
    return Sentinel(base_dir=temp_dir)
    
@pytest.fixture
def sentinel_nexus(temp_dir: str) -> Sentinel:
    """Provide Sentinel instance with Nexus backend."""
    # Override config
    sentinel_config.EXPLAIN_BACKEND = sentinel_config.ExplainBackend.NEXUS
    return Sentinel(
        base_dir=temp_dir,
        explain_backend="nexus",
        nexus_client=MockNexusClient()
    )
    
@pytest.fixture
def sample_input() -> Dict[str, Any]:
    """Provide sample input data for testing."""
    return {
        "text": "Proceed with deployment.",
        "_signals": {
            "bio": {
                "stress": 0.3,
                "heart_rate": 75,
                "hrv": 45
            },
            "env": {
                "context_risk": 0.2,
                "incident_sev": 0.1
            }
        }
    }

@pytest.fixture
def challenge_scenarios() -> List[Dict[str, Any]]:
    """Provide test challenge scenarios."""
    return [
        {
            "name": "low_risk",
            "payload": {
                "text": "Deploy if ready.",
                "_signals": {
                    "bio": {"stress": 0.2},
                    "env": {"context_risk": 0.1}
                }
            },
            "expect": "PROCEED"
        },
        {
            "name": "high_risk",
            "payload": {
                "text": "Ship now!",
                "_signals": {
                    "bio": {"stress": 0.8},
                    "env": {"context_risk": 0.7}
                }
            },
            "expect": "BLOCK"
        }
    ]