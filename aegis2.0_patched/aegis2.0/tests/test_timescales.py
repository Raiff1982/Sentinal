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
        
    def test_entropy_calculation(self):
        """Test entropy calculation for different types of values."""
        # Test string entropy
        self.memory.set("str_uniform", "aaaa")  # Low entropy
        self.memory.set("str_varied", "ab!2@Zx")  # Higher entropy
        
        store = self.memory.audit()
        self.assertLess(store["str_uniform"]["entropy"], store["str_varied"]["entropy"])
        
        # Test JSON entropy
        self.memory.set("json_simple", {"a": 1})
        self.memory.set("json_complex", {"a": [1,2,3], "b": {"c": True, "d": None}})
        
        store = self.memory.audit()
        self.assertLess(store["json_simple"]["entropy"], store["json_complex"]["entropy"])
        
    def test_integrity_maintenance(self):
        """Test memory integrity maintenance over operations."""
        # Initial state
        self.memory.set("test_key", "initial")
        initial_audit = self.memory.audit()
        
        # Modify value
        self.memory.set("test_key", "modified")
        modified_audit = self.memory.audit()
        
        # Check timestamps updated
        self.assertGreater(
            modified_audit["test_key"]["timestamp"],
            initial_audit["test_key"]["timestamp"]
        )
        
        # Check entropy tracked
        self.assertIn("entropy", modified_audit["test_key"])
        
        # Test entry expiration
        self.memory.set("expire_soon", "value", ttl=1)
        pre_expire = self.memory.audit()
        self.assertIn("expire_soon", pre_expire)
        
        import time
        time.sleep(1.1)
        
        post_expire = self.memory.audit()
        self.assertNotIn("expire_soon", post_expire)
        
    def test_concurrent_access(self):
        """Test thread-safe operations."""
        import threading
        import random
        
        def worker():
            for _ in range(100):
                key = f"key_{random.randint(1, 10)}"
                value = str(random.randint(1, 1000))
                self.memory.set(key, value)
                _ = self.memory.get(key)
                _ = self.memory.audit()
                
        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
            
        # Verify no data corruption
        audit = self.memory.audit()
        for key, entry in audit.items():
            self.assertIsInstance(entry["value"], str)
            self.assertIsInstance(entry["timestamp"], datetime)
            self.assertIsInstance(entry["entropy"], float)
            self.assertGreaterEqual(entry["entropy"], 0.0)
            self.assertLessEqual(entry["entropy"], 1.0)

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
        
        # Test with custom trusted domains
        sanitizer = InputSanitizer({
            "trusted_domains": ["trusted.example.com"]
        })
        
        result = sanitizer.audit_text("Check https://trusted.example.com")
        self.assertTrue(result["safe"])
        
        result = sanitizer.audit_text("Check https://untrusted.example.com")
        self.assertIn("untrusted_domain:untrusted.example.com", "".join(result["warnings"]))
        
    def test_json_structure_validation(self):
        """Test JSON structure validation."""
        # Test field length
        long_field = "x" * (self.sanitizer.max_field_length + 1)
        json_data = json.dumps({long_field: "value"})
        result = self.sanitizer.audit_text(json_data)
        self.assertIn("field_name_too_long", "".join(result["issues"]))
        
        # Test nested depth
        deep_json = {}
        current = deep_json
        for i in range(self.sanitizer.max_depth + 2):
            current["nested"] = {}
            current = current["nested"]
            
        result = self.sanitizer.audit_text(json.dumps(deep_json))
        self.assertIn("max_depth_exceeded", "".join(result["warnings"]))
        
        # Test mixed content types
        mixed_json = {
            "str": "text",
            "num": 123,
            "bool": True,
            "null": None,
            "arr": [1, "two", {"three": 3}],
            "obj": {"nested": {"deep": ["value"]}}
        }
        result = self.sanitizer.audit_text(json.dumps(mixed_json))
        self.assertTrue(result["safe"])
        
    def test_control_character_detection(self):
        """Test control character detection."""
        # Test various control characters
        for i in range(0x20):
            if i in {0x09, 0x0A, 0x0D}:  # Skip allowed characters
                continue
            text = f"Test{chr(i)}text"
            result = self.sanitizer.audit_text(text)
            self.assertFalse(result["safe"])
            self.assertTrue(
                any(f"control_chars_detected" in issue for issue in result["issues"]),
                f"Control char 0x{i:02x} not detected"
            )
            
        # Test mixed control characters
        mixed = "Test\x00with\x1Fmultiple\x02controls"
        result = self.sanitizer.audit_text(mixed)
        self.assertFalse(result["safe"])
        detected = "".join(result["issues"])
        self.assertIn("0x0", detected)
        self.assertIn("0x1f", detected.lower())
        self.assertIn("0x2", detected)
        
    def test_suspicious_content_detection(self):
        """Test detection of suspicious content."""
        suspicious_inputs = [
            ("<script>alert(1)</script>", "potential_xss_detected"),
            ("onerror=alert(1)", "potential_xss_detected"),
            ("rm -rf /", "suspicious_commands_detected"),
            ("'; DROP TABLE users;--", None),  # Should be normalized
            ("data:text/html,<script>", "disallowed_scheme:data"),
            ("vbscript:msgbox(1)", "disallowed_scheme:vbscript")
        ]
        
        for input_text, expected_issue in suspicious_inputs:
            result = self.sanitizer.audit_text(input_text)
            if expected_issue:
                self.assertTrue(
                    any(expected_issue in issue for issue in result["issues"]),
                    f"Failed to detect {expected_issue} in '{input_text}'"
                )
            self.assertTrue("normalized" in result)

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
        
    def test_policy_deep_merge(self):
        """Test deep merging of policy configurations."""
        base_policy = {
            "thresholds": {
                "risk_block": 0.8,
                "nested": {"value": 1}
            },
            "weights": {"risk": 1.0}
        }
        
        update_policy = {
            "thresholds": {
                "risk_block": 0.9,
                "nested": {"new_value": 2}
            },
            "new_section": {"key": "value"}
        }
        
        self.timescales._deep_merge(base_policy, update_policy)
        
        self.assertEqual(base_policy["thresholds"]["risk_block"], 0.9)
        self.assertEqual(base_policy["thresholds"]["nested"]["value"], 1)
        self.assertEqual(base_policy["thresholds"]["nested"]["new_value"], 2)
        self.assertEqual(base_policy["new_section"]["key"], "value")
        self.assertEqual(base_policy["weights"]["risk"], 1.0)
        
    def test_custom_policy_thresholds(self):
        """Test application of custom policy thresholds."""
        # Create custom policy
        policy = {
            "thresholds": {
                "risk_block": 0.75,
                "risk_caution": 0.45,
                "severity_block": 0.85,
                "severity_caution": 0.55,
                "stress_block": 0.95,
                "stress_caution": 0.65
            }
        }
        with open(self.policy_path, "w") as f:
            json.dump(policy, f)
            
        # Create agent with high risk
        class RiskAgent(BaseAgent):
            def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
                return {
                    "agent": self.name,
                    "details": {"risk": 0.7},
                    "ok": True
                }
                
        # Test with risk above caution but below block
        memory = NexusMemory()
        agent = RiskAgent("risk", memory)
        self.timescales.add_agent("risk", agent)
        
        result = self.timescales.dispatch({"test": "data"})
        self.assertEqual(result["decision"], "CAUTION")
        
        # Test with severity above block
        class SeverityAgent(BaseAgent):
            def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
                return {
                    "agent": self.name,
                    "details": {"severity": 0.9},
                    "ok": True
                }
                
        self.timescales.add_agent("severity", SeverityAgent("severity", memory))
        result = self.timescales.dispatch({"test": "data"})
        self.assertEqual(result["decision"], "BLOCK")
        
    def test_weighted_metrics(self):
        """Test weighted metric calculations."""
        policy = {
            "weights": {
                "risk": 2.0,
                "severity": 0.5,
                "stress": 1.5
            }
        }
        with open(self.policy_path, "w") as f:
            json.dump(policy, f)
            
        # Create test agent
        class MetricAgent(BaseAgent):
            def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
                return {
                    "agent": self.name,
                    "details": {
                        "risk": 0.4,
                        "severity": 0.8,
                        "stress": 0.6
                    },
                    "ok": True
                }
                
        memory = NexusMemory()
        agent = MetricAgent("metrics", memory)
        self.timescales.add_agent("metrics", agent)
        
        result = self.timescales.dispatch({"test": "data"})
        loaded_policy = self.timescales._load_policy()
        
        # Verify weighted values are used in decision
        self.assertEqual(
            self.timescales._risk,
            0.4 * loaded_policy["weights"]["risk"]
        )
        self.assertEqual(
            self.timescales._severity,
            0.8 * loaded_policy["weights"]["severity"]
        )

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