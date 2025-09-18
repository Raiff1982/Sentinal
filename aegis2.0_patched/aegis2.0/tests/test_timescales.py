"""Unit tests for aegis_timescales module."""

import unittest
import json
import os
import tempfile
from datetime import datetime, timedelta
from typing import Dict, Any
from aegis_timescales import (
    NexusMemory,
    InputSanitizer,
    AegisTimescales,
    BaseAgent,
    EchoSeedAgent
)

class TestNexusMemory(unittest.TestCase):
    """Test the NexusMemory class functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.persistence_path = os.path.join(self.temp_dir, "memory.json")
        self.memory = NexusMemory(
            max_entries=5,
            default_ttl_secs=60,
            persistence_path=self.persistence_path
        )
        
    def tearDown(self):
        """Clean up test environment."""
        if os.path.exists(self.persistence_path):
            os.remove(self.persistence_path)
        os.rmdir(self.temp_dir)
        
    def test_basic_operations(self):
        """Test basic memory operations."""
        # Test set and get
        self.memory.set("key1", "value1")
        self.assertEqual(self.memory.get("key1"), "value1")
        
        # Test non-existent key
        self.assertIsNone(self.memory.get("nonexistent"))
        
        # Test overwrite
        self.memory.set("key1", "value2")
        self.assertEqual(self.memory.get("key1"), "value2")
        
    def test_ttl_expiration(self):
        """Test TTL-based expiration."""
        # Set with short TTL
        self.memory.set("temp", "value", ttl=1)
        self.assertEqual(self.memory.get("temp"), "value")
        
        # Wait for expiration
        import time
        time.sleep(1.1)
        
        # Should be expired
        self.assertIsNone(self.memory.get("temp"))
        
    def test_max_entries(self):
        """Test max entries limit."""
        # Add more than max entries
        for i in range(10):
            self.memory.set(f"key{i}", f"value{i}")
            
        # Should only have max_entries items
        store = self.memory.audit()
        self.assertLessEqual(len(store), self.memory.max_entries)
        
    def test_persistence(self):
        """Test persistence to disk."""
        # Add some entries
        self.memory.set("persist1", "value1")
        self.memory.set("persist2", {"nested": "value2"})
        
        # Create new instance with same path
        new_memory = NexusMemory(persistence_path=self.persistence_path)
        
        # Should have same entries
        self.assertEqual(new_memory.get("persist1"), "value1")
        self.assertEqual(new_memory.get("persist2"), {"nested": "value2"})

class TestInputSanitizer(unittest.TestCase):
    """Test the InputSanitizer class functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.sanitizer = InputSanitizer()
        
    def test_basic_text(self):
        """Test basic text sanitization."""
        result = self.sanitizer.audit_text("Hello World")
        self.assertTrue(result["safe"])
        self.assertEqual(result["normalized"], "Hello World")
        self.assertEqual(len(result["issues"]), 0)
        
    def test_max_length(self):
        """Test max length validation."""
        long_text = "x" * (self.sanitizer.max_length + 1)
        result = self.sanitizer.audit_text(long_text)
        self.assertFalse(result["safe"])
        self.assertIn("input_too_long", result["issues"])
        
    def test_control_chars(self):
        """Test control character detection."""
        text_with_control = "Hello\x00World"
        result = self.sanitizer.audit_text(text_with_control)
        self.assertFalse(result["safe"])
        self.assertIn("control_chars_detected:0x0", "".join(result["issues"]))
        
    def test_json_validation(self):
        """Test JSON structure validation."""
        # Valid JSON
        valid_json = '{"key": "value"}'
        result = self.sanitizer.audit_text(valid_json)
        self.assertTrue(result["safe"])
        
        # Deep nested JSON
        deep_json = "{" + "".join('"x":{'*20 + '}' * 20)
        result = self.sanitizer.audit_text(deep_json)
        self.assertTrue(result["safe"])
        self.assertIn("max_depth_exceeded", result["warnings"])
        
    def test_url_validation(self):
        """Test URL validation."""
        # Valid URL
        valid_url = "Check https://example.com"
        result = self.sanitizer.audit_text(valid_url)
        self.assertTrue(result["safe"])
        
        # Invalid scheme
        invalid_url = "Check javascript:alert(1)"
        result = self.sanitizer.audit_text(invalid_url)
        self.assertFalse(result["safe"])
        self.assertIn("disallowed_scheme:javascript", "".join(result["issues"]))

class TestAegisTimescales(unittest.TestCase):
    """Test the AegisTimescales class functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.policy_path = os.path.join(self.temp_dir, "policy.json")
        self.timescales = AegisTimescales(policy_path=self.policy_path)
        
    def tearDown(self):
        """Clean up test environment."""
        if os.path.exists(self.policy_path):
            os.remove(self.policy_path)
        os.rmdir(self.temp_dir)
        
    def test_policy_loading(self):
        """Test policy configuration loading."""
        # Create custom policy
        custom_policy = {
            "thresholds": {
                "risk_block": 0.9,
                "risk_caution": 0.6
            }
        }
        with open(self.policy_path, "w") as f:
            json.dump(custom_policy, f)
            
        # Should use custom thresholds
        policy = self.timescales._load_policy()
        self.assertEqual(policy["thresholds"]["risk_block"], 0.9)
        self.assertEqual(policy["thresholds"]["risk_caution"], 0.6)
        
        # Should keep defaults for unspecified values
        self.assertEqual(policy["thresholds"]["severity_block"], 0.8)
        
    def test_decision_making(self):
        """Test decision making logic."""
        # Create test agent
        class TestAgent(BaseAgent):
            def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
                return {
                    "agent": self.name,
                    "details": {
                        "risk": 0.9,
                        "severity": 0.4
                    },
                    "ok": True
                }
                
        # Add agent and process input
        memory = NexusMemory()
        agent = TestAgent("test", memory)
        self.timescales.add_agent("test", agent)
        
        result = self.timescales.dispatch({"test": "data"})
        self.assertEqual(result.get("decision", "ALLOW"), "BLOCK")
        
    def test_stress_calculation(self):
        """Test stress level calculation."""
        class StressAgent(BaseAgent):
            def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
                return {
                    "agent": self.name,
                    "details": {
                        "operator_stress": 0.7
                    },
                    "ok": True
                }
                
        memory = NexusMemory()
        agent = StressAgent("stress", memory)
        self.timescales.add_agent("stress", agent)
        
        result = self.timescales.dispatch({
            "stress": 0.3,
            "_signals": {
                "bio": {"stress": 0.8},
                "env": {"stress": 0.4}
            }
        })
        
        # Should take maximum of all stress signals
        self.assertEqual(self.timescales._stress, 0.8)

class TestEndToEnd(unittest.TestCase):
    """End-to-end integration tests."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.memory_path = os.path.join(self.temp_dir, "memory.json")
        self.policy_path = os.path.join(self.temp_dir, "policy.json")
        
        self.memory = NexusMemory(persistence_path=self.memory_path)
        self.timescales = AegisTimescales(policy_path=self.policy_path)
        
    def tearDown(self):
        """Clean up test environment."""
        for path in [self.memory_path, self.policy_path]:
            if os.path.exists(path):
                os.remove(path)
        os.rmdir(self.temp_dir)
        
    def test_full_pipeline(self):
        """Test complete processing pipeline."""
        # Set up agents
        agents = [
            EchoSeedAgent("echo", self.memory),
            # Add more agents as needed
        ]
        
        for agent in agents:
            self.timescales.add_agent(agent.name, agent)
            
        # Create test input
        input_data = {
            "text": "Urgent request with potential risk",
            "intent": "rush",
            "_signals": {
                "bio": {"stress": 0.7},
                "env": {"risk": 0.6}
            }
        }
        
        # Process through pipeline
        result = self.timescales.dispatch(input_data)
        
        # Verify results
        self.assertIn("reports", result)
        self.assertIn("input_audit", result)
        self.assertTrue(isinstance(result.get("decision"), str))
        
        # Verify memory persistence
        new_memory = NexusMemory(persistence_path=self.memory_path)
        self.assertEqual(len(new_memory.audit()), len(self.memory.audit()))

if __name__ == '__main__':
    unittest.main()