"""
Tests for explain store implementations.
"""
import json
import pytest
from datetime import datetime, timedelta
from typing import Dict, Any

from ..explain import ExplainStore, ExplainEntry

def test_write_read(jsonl_store: ExplainStore, nexus_store: ExplainStore):
    """Test writing and reading entries."""
    now = datetime.utcnow()
    entry = ExplainEntry(
        timestamp=now,
        agent="TestAgent",
        category="test",
        data={"value": 42},
        metadata={"source": "test"},
        ttl_sec=300
    )
    
    # Test both backends
    for store in [jsonl_store, nexus_store]:
        # Write entry
        assert store.write(entry) is True
        
        # Read back
        results = store.read(
            agent="TestAgent",
            start_time=now - timedelta(minutes=1),
            end_time=now + timedelta(minutes=1)
        )
        
        assert len(results) == 1
        result = results[0]
        
        # Verify data
        assert result.agent == entry.agent
        assert result.category == entry.category
        assert result.data == entry.data
        assert result.metadata == entry.metadata
        assert result.ttl_sec == entry.ttl_sec
        
def test_cleanup(jsonl_store: ExplainStore, nexus_store: ExplainStore):
    """Test cleanup of expired entries."""
    now = datetime.utcnow()
    old_entry = ExplainEntry(
        timestamp=now - timedelta(hours=2),
        agent="TestAgent",
        category="test",
        data={"old": True},
        metadata={},
        ttl_sec=300  # 5 minutes
    )
    new_entry = ExplainEntry(
        timestamp=now,
        agent="TestAgent", 
        category="test",
        data={"new": True},
        metadata={},
        ttl_sec=3600  # 1 hour
    )
    
    # Test both backends
    for store in [jsonl_store, nexus_store]:
        # Write entries
        assert store.write(old_entry) is True
        assert store.write(new_entry) is True
        
        # Run cleanup
        removed = store.cleanup()
        assert removed >= 1  # Old entry should be removed
        
        # Verify only new entry remains
        results = store.read(agent="TestAgent")
        assert len(results) == 1
        assert results[0].data == {"new": True}