"""Multi-timescale diagnostic system."""

import json
import math
import logging
import threading
import unicodedata
import re
import os
from datetime import datetime, timedelta, timezone
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from typing import Dict, List, Any, Sequence, Optional, TypedDict, Tuple, Deque

log = logging.getLogger(__name__)

def _now() -> datetime:
    """Get current UTC time."""
    return datetime.now(timezone.utc)

class MemoryEntry(TypedDict, total=False):
    """Memory entry type definition."""
    value: Any
    timestamp: datetime
    weight: float
    entropy: float
    ttl: int

class MemorySnapshot(TypedDict, total=False):
    """Memory snapshot type definition."""
    version: str
    store: Dict[str, MemoryEntry]
    expiration_heap: List[Tuple[float, str]]
    max_entries: int
    default_ttl_secs: int

class NexusMemory:
    """Memory system with integrity tracking and persistence."""
    VERSION = "1.0.0"
    
    def __init__(
        self,
        max_entries: int = 20_000,
        default_ttl_secs: int = 14*24*3600,
        persistence_path: Optional[str] = None,
        initial_entries: Optional[Dict[str, Dict[str, Any]]] = None
    ):
        """Initialize memory system."""
        self.max_entries = max_entries
        self.default_ttl_secs = default_ttl_secs
        self._store: Dict[str, MemoryEntry] = {}
        self._expiration_heap: List[Tuple[float, str]] = []
        self._lock = threading.Lock()
        self.persistence_path = persistence_path
        self._last_cleanup = _now()
        
        if persistence_path and os.path.exists(persistence_path):
            self._load_from_disk()
        elif initial_entries:
            with self._lock:
                for key, value in initial_entries.items():
                    self.set(key, value)
                    
    def _serialize_entry(self, entry: MemoryEntry) -> Dict[str, Any]:
        """Serialize memory entry for persistence."""
        return {
            "value": entry["value"],
            "timestamp": entry["timestamp"].isoformat(),
            "weight": entry.get("weight", 1.0),
            "entropy": entry.get("entropy", 0.0),
            "ttl": entry.get("ttl", self.default_ttl_secs)
        }
        
    def _deserialize_entry(self, data: Dict[str, Any]) -> MemoryEntry:
        """Deserialize memory entry from persistence."""
        entry: MemoryEntry = {
            "value": data["value"],
            "timestamp": datetime.fromisoformat(data["timestamp"]),
            "weight": float(data.get("weight", 1.0)),
            "entropy": float(data.get("entropy", 0.0)),
            "ttl": int(data.get("ttl", self.default_ttl_secs))
        }
        return entry
        
    def _load_from_disk(self) -> None:
        """Load memory from disk."""
        try:
            with open(self.persistence_path, "r") as f:
                data = json.load(f)
                
            if data.get("version") != self.VERSION:
                log.warning(f"Memory version mismatch: {data.get('version')} vs {self.VERSION}")
                return
                
            with self._lock:
                self._store.clear()
                self._expiration_heap.clear()
                
                for key, entry_data in data["store"].items():
                    entry = self._deserialize_entry(entry_data)
                    expiration = entry["timestamp"] + timedelta(seconds=entry["ttl"])
                    
                    # Skip expired entries
                    if expiration <= _now():
                        continue
                        
                    self._store[key] = entry
                    self._expiration_heap.append((expiration.timestamp(), key))
                    
                self._expiration_heap.sort()
                
        except Exception as e:
            log.error(f"Failed to load memory from {self.persistence_path}: {e}")
            
    def _save_to_disk(self) -> None:
        """Save memory to disk."""
        if not self.persistence_path:
            return
            
        try:
            with self._lock:
                data = {
                    "version": self.VERSION,
                    "store": {
                        key: self._serialize_entry(entry)
                        for key, entry in self._store.items()
                    },
                    "expiration_heap": self._expiration_heap,
                    "max_entries": self.max_entries,
                    "default_ttl_secs": self.default_ttl_secs
                }
                
            # Use atomic write pattern
            temp_path = f"{self.persistence_path}.tmp"
            with open(temp_path, "w") as f:
                json.dump(data, f, indent=2)
                
            os.replace(temp_path, self.persistence_path)
            
        except Exception as e:
            log.error(f"Failed to save memory to {self.persistence_path}: {e}")
            
    def _cleanup_expired(self) -> None:
        """Remove expired entries."""
        now = _now()
        if (now - self._last_cleanup).total_seconds() < 60:
            return
            
        with self._lock:
            while self._expiration_heap:
                expiry_ts, key = self._expiration_heap[0]
                if datetime.fromtimestamp(expiry_ts, timezone.utc) > now:
                    break
                    
                self._expiration_heap.pop(0)
                self._store.pop(key, None)
                
            self._last_cleanup = now
            
    def _calculate_entropy(self, value: Any) -> float:
        """Calculate Shannon entropy of value."""
        try:
            # Convert value to bytes for entropy calculation
            if isinstance(value, str):
                data = value.encode()
            else:
                data = json.dumps(value).encode()
                
            # Calculate frequency of each byte
            freq = defaultdict(int)
            for byte in data:
                freq[byte] += 1
                
            # Calculate Shannon entropy
            length = len(data)
            entropy = 0.0
            for count in freq.values():
                p = count / length
                entropy -= p * math.log2(p)
                
            return min(1.0, entropy / 8.0)  # Normalize to [0,1]
            
        except Exception:
            return 0.0
            
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set a value in memory with integrity tracking."""
        now = _now()
        ttl = ttl if ttl is not None else self.default_ttl_secs
        
        entry: MemoryEntry = {
            "value": value,
            "timestamp": now,
            "weight": 1.0,
            "entropy": self._calculate_entropy(value),
            "ttl": ttl
        }
        
        with self._lock:
            # Check capacity
            if len(self._store) >= self.max_entries and key not in self._store:
                # Remove oldest entry
                if self._expiration_heap:
                    _, oldest_key = self._expiration_heap.pop(0)
                    self._store.pop(oldest_key, None)
                    
            # Update entry
            self._store[key] = entry
            expiry_ts = (now + timedelta(seconds=ttl)).timestamp()
            
            # Update expiration heap
            self._expiration_heap = [(ts, k) for ts, k in self._expiration_heap if k != key]
            self._expiration_heap.append((expiry_ts, key))
            self._expiration_heap.sort()
            
            # Periodic cleanup and persistence
            self._cleanup_expired()
            self._save_to_disk()
            
    def get(self, key: str) -> Optional[Any]:
        """Get a value from memory."""
        with self._lock:
            entry = self._store.get(key)
            if not entry:
                return None
                
            # Check expiration
            now = _now()
            if (now - entry["timestamp"]).total_seconds() > entry["ttl"]:
                self._store.pop(key, None)
                return None
                
            return entry["value"]
            
    def audit(self) -> Dict[str, MemoryEntry]:
        """Return copy of memory store with integrity info."""
        self._cleanup_expired()
        with self._lock:
            return self._store.copy()
            
    def clear(self) -> None:
        """Clear all entries."""
        with self._lock:
            self._store.clear()
            self._expiration_heap.clear()
            self._save_to_disk()

class BaseAgent(ABC):
    """Base class for all agents."""
    
    def __init__(self, name: str, memory: NexusMemory):
        """Initialize agent."""
        self.name = name
        self.memory = memory
        self._lock = threading.Lock()
        
    @abstractmethod
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input data and return report."""
        pass

class InputSanitizer:
    """Validates and sanitizes input."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize sanitizer with optional configuration.
        
        Args:
            config: Optional configuration dictionary with the following keys:
                max_length: Maximum allowed text length
                max_line_length: Maximum allowed line length
                max_depth: Maximum allowed JSON nesting depth
                max_field_length: Maximum allowed field name length
                banned_patterns: List of regex patterns to block
                allowed_schemes: List of allowed URI schemes
                trusted_domains: List of trusted domains
        """
        self.max_length = config.get("max_length", 10_000) if config else 10_000
        self.max_line_length = config.get("max_line_length", 1_000) if config else 1_000
        self.max_depth = config.get("max_depth", 10) if config else 10
        self.max_field_length = config.get("max_field_length", 256) if config else 256
        self.control_chars = set(range(0x00, 0x20)) - {0x09, 0x0A, 0x0D}  # Allow tab, LF, CR
        
        # Additional security patterns
        self.banned_patterns = [re.compile(p) for p in (config.get("banned_patterns", []) if config else [])]
        self.banned_patterns.extend([
            re.compile(r"javascript:", re.I),  # Block javascript: URLs
            re.compile(r"data:", re.I),       # Block data: URLs
            re.compile(r"vbscript:", re.I),   # Block vbscript: URLs
        ])
        
        # Security configurations
        self.allowed_schemes = set(config.get("allowed_schemes", ["http", "https"]) if config else ["http", "https"])
        self.trusted_domains = set(config.get("trusted_domains", []) if config else [])
        
    def _check_content_safety(self, content: str) -> Tuple[bool, List[str], List[str]]:
        """Check content for potential security issues.
        
        Returns:
            Tuple of (is_safe, issues, warnings)
        """
        issues = []
        warnings = []
        
        # Check length
        if len(content) > self.max_length:
            issues.append("input_too_long")
            return False, issues, warnings
            
        # Check line length
        lines = content.splitlines()
        long_lines = [i for i, line in enumerate(lines, 1) if len(line) > self.max_line_length]
        if long_lines:
            warnings.append(f"Lines {','.join(map(str, long_lines))} exceed max length")
            
        # Check control characters
        control_chars = {c for c in content if ord(c) in self.control_chars}
        if control_chars:
            issues.append(f"control_chars_detected:{','.join(hex(ord(c)) for c in control_chars)}")
            
        # Check banned patterns
        for pattern in self.banned_patterns:
            if pattern.search(content):
                issues.append(f"banned_pattern_matched:{pattern.pattern}")
                
        # Check for potential XSS/injection patterns
        if re.search(r"<script|javascript:|onerror=|onload=", content, re.I):
            issues.append("potential_xss_detected")
            
        # Check for suspicious shell commands
        if re.search(r";\s*rm\s|;\s*del\s|;\s*chmod\s", content):
            issues.append("suspicious_commands_detected")
            
        return not bool(issues), issues, warnings
        
    def _normalize_content(self, content: str) -> str:
        """Normalize content for consistency."""
        # Normalize Unicode representation
        normalized = unicodedata.normalize("NFKC", content)
        
        # Normalize whitespace
        normalized = re.sub(r"[ \t]+", " ", normalized)
        normalized = re.sub(r"\r\n?|\n", "\n", normalized)
        normalized = re.sub(r"\n\s+\n", "\n\n", normalized)
        normalized = re.sub(r"\n{3,}", "\n\n", normalized)
        
        # Remove BOM if present
        if normalized.startswith("\ufeff"):
            normalized = normalized[1:]
            
        return normalized.strip()
        
    def _validate_json_structure(self, data: Any, depth: int = 0) -> Tuple[bool, List[str]]:
        """Recursively validate JSON structure."""
        issues = []
        
        # Check max depth
        if depth > self.max_depth:
            issues.append("max_depth_exceeded")
            return False, issues
            
        if isinstance(data, dict):
            # Check field names
            for key in data:
                if len(str(key)) > self.max_field_length:
                    issues.append(f"field_name_too_long:{key[:50]}")
                if not isinstance(key, str):
                    issues.append("non_string_field_names")
                    
            # Recurse into values
            for value in data.values():
                ok, sub_issues = self._validate_json_structure(value, depth + 1)
                issues.extend(sub_issues)
                
        elif isinstance(data, (list, tuple)):
            # Recurse into sequence items
            for item in data:
                ok, sub_issues = self._validate_json_structure(item, depth + 1)
                issues.extend(sub_issues)
                
        return not bool(issues), issues
        
    def audit_text(self, text: str) -> Dict[str, Any]:
        """Audit text content with comprehensive validation.
        
        Args:
            text: The text content to audit
            
        Returns:
            Dictionary containing audit results:
                safe: Boolean indicating if content is safe
                issues: List of security/validation issues
                warnings: List of non-critical warnings
                normalized: Normalized version of input
        """
        result = {
            "safe": True,
            "issues": [],
            "warnings": [],
            "normalized": ""
        }
        
        try:
            if not text:
                result["normalized"] = ""
                return result
                
            # Basic content safety checks
            is_safe, issues, warnings = self._check_content_safety(text)
            result["safe"] = is_safe
            result["issues"].extend(issues)
            result["warnings"].extend(warnings)
            
            if not is_safe:
                return result
                
            # Try to parse as JSON for additional validation
            try:
                json_data = json.loads(text)
                is_valid, json_issues = self._validate_json_structure(json_data)
                if not is_valid:
                    result["warnings"].extend(json_issues)
            except json.JSONDecodeError:
                # Not JSON, continue with text validation
                pass
                
            # Normalize content
            result["normalized"] = self._normalize_content(text)
            
            # URL safety checks if content contains URLs
            urls = re.finditer(r'https?://[^\s<>"]+|www\.[^\s<>"]+', text)
            for url in urls:
                try:
                    from urllib.parse import urlparse
                    parsed = urlparse(url.group())
                    if parsed.scheme and parsed.scheme not in self.allowed_schemes:
                        result["issues"].append(f"disallowed_scheme:{parsed.scheme}")
                        result["safe"] = False
                    if parsed.netloc and self.trusted_domains and parsed.netloc not in self.trusted_domains:
                        result["warnings"].append(f"untrusted_domain:{parsed.netloc}")
                except Exception:
                    result["warnings"].append("invalid_url_format")
                    
        except Exception as e:
            result["issues"].append(f"sanitization_error:{str(e)}")
            result["safe"] = False
            
        return result

class AegisTimescales:
    """Multi-timescale diagnostic system coordinator."""
    
    def __init__(self, timeout_sec: float = 30.0, policy_path: Optional[str] = None):
        """Initialize coordinator with optional policy configuration.
        
        Args:
            timeout_sec: Maximum time to wait for agent responses
            policy_path: Optional path to custom policy configuration file
        """
        self._timeout_sec = timeout_sec
        self._initialize_state()
        self._memory = {}
        self._memory_lock = threading.Lock()
        self._agents = {}
        self.policy_path = policy_path
        
    def add_agent(self, name: str, agent: Any) -> None:
        """Add an agent."""
        if not hasattr(agent, "process"):
            raise ValueError(f"Agent {name} must have process() method")
        self._agents[name] = agent
        
    @property 
    def agents(self) -> Dict[str, Any]:
        """Get registered agents."""
        return self._agents.copy()
        
    def _initialize_state(self) -> None:
        """Reset system state."""
        self._risk = 0.0
        self._severity = 0.0
        self._conflict = 0.0
        self._timescale = 0.0
        self._stress = 0.0
        
    def _process_metrics(self, details: Dict[str, Any], input_data: Dict[str, Any]) -> None:
        """Process metrics from agent report."""
        # Severity
        severity_values = []
        for key in ["severity", "severity_now"]:
            if key in details:
                try:
                    val = float(details[key])
                    if val >= 0:
                        severity_values.append(val)
                except (ValueError, TypeError):
                    continue
        
        if severity_values:
            self._severity = max(self._severity, max(severity_values))
            
        # Stress signals
        agent_stress = float(details.get("stress", details.get("operator_stress", 0.0)) or 0.0)
        signal_stress = float(input_data.get("_signals", {}).get("bio", {}).get("stress", 0.0))
        env_stress = float(input_data.get("_signals", {}).get("env", {}).get("stress", 0.0))
        input_stress = float(input_data.get("stress", input_data.get("operator_stress", 0.0)) or 0.0)
        
        self._stress = max(
            self._stress,
            agent_stress, 
            signal_stress,
            env_stress,
            input_stress
        )
        
        # Risk signals
        agent_risk = float(details.get("context_risk", details.get("risk", 0.0)) or 0.0)
        signal_risk = float(input_data.get("_signals", {}).get("env", {}).get("context_risk", 0.0))
        
        self._risk = max(
            self._risk,
            agent_risk,
            signal_risk,
            float(input_data.get("risk", 0.0) or 0.0)
        )
        
        # Conflict and timescale
        curr_conflict = float(details.get("conflict", input_data.get("conflict", 0.0)) or 0.0)
        self._conflict = max(self._conflict, curr_conflict)
        
        curr_timescale = float(details.get("timescale_signal", details.get("caution_fused", 0.0)) or 0.0)
        self._timescale = max(self._timescale, curr_timescale)
        
    def _make_decision(self) -> str:
        """Make final decision based on policy thresholds."""
        # Apply policy thresholds
        policy = self._load_policy()
        
        # Risk thresholds
        risk_block = policy.get("thresholds", {}).get("risk_block", 0.8)
        risk_caution = policy.get("thresholds", {}).get("risk_caution", 0.5)
        
        # Severity thresholds
        severity_block = policy.get("thresholds", {}).get("severity_block", 0.8)
        severity_caution = policy.get("thresholds", {}).get("severity_caution", 0.5)
        
        # Conflict thresholds
        conflict_block = policy.get("thresholds", {}).get("conflict_block", 0.8)
        conflict_caution = policy.get("thresholds", {}).get("conflict_caution", 0.5)
        
        # Timescale thresholds
        timescale_block = policy.get("thresholds", {}).get("timescale_block", 0.8)
        timescale_caution = policy.get("thresholds", {}).get("timescale_caution", 0.5)
        
        # Stress thresholds
        stress_block = policy.get("thresholds", {}).get("stress_block", 0.8)
        stress_caution = policy.get("thresholds", {}).get("stress_caution", 0.5)
        
        # Apply blocking rules
        if any([
            self._risk > risk_block,
            self._severity > severity_block,
            self._conflict > conflict_block,
            self._timescale > timescale_block,
            self._stress > stress_block
        ]):
            return "BLOCK"
            
        # Apply caution rules
        if any([
            self._risk > risk_caution,
            self._severity > severity_caution,
            self._conflict > conflict_caution,
            self._timescale > timescale_caution,
            self._stress > stress_caution
        ]):
            return "CAUTION"
            
        return "ALLOW"
        
    def _load_policy(self) -> Dict[str, Any]:
        """Load policy configuration."""
        # Default policy thresholds
        default_policy = {
            "thresholds": {
                "risk_block": 0.8,
                "risk_caution": 0.5,
                "severity_block": 0.8,
                "severity_caution": 0.5,
                "conflict_block": 0.8,
                "conflict_caution": 0.5,
                "timescale_block": 0.8,
                "timescale_caution": 0.5,
                "stress_block": 0.8,
                "stress_caution": 0.5
            },
            "weights": {
                "risk": 1.0,
                "severity": 1.0,
                "conflict": 0.8,
                "timescale": 0.7,
                "stress": 0.6
            }
        }
        
        try:
            # Try to load custom policy if configured
            if hasattr(self, "policy_path") and self.policy_path:
                with open(self.policy_path, "r") as f:
                    custom_policy = json.load(f)
                    # Deep merge with defaults
                    self._deep_merge(default_policy, custom_policy)
        except Exception as e:
            log.warning(f"Failed to load custom policy: {e}")
            
        return default_policy
        
    def _deep_merge(self, target: Dict, source: Dict) -> None:
        """Deep merge source dict into target dict."""
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._deep_merge(target[key], value)
            else:
                target[key] = value
        
    def _sanitize_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize and validate input."""
        if not isinstance(input_data, dict):
            raise ValueError("Input must be a dictionary")
            
        sanitized = input_data.copy()
        audit = {"issues": [], "safe": True}
        
        content = json.dumps(sanitized)
        if len(content) > 32768:
            audit["issues"].append("length_exceeded") 
            audit["safe"] = False
            
        if any(unicodedata.category(c).startswith("C") for c in content):
            audit["issues"].append("control_chars")
            audit["safe"] = False
            
        sanitized["_audit"] = audit
        return sanitized
        
    def dispatch(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Dispatch input to agents.
        
        Args:
            input_data: Input to process
            
        Returns:
            Processing results
        """
        def process_agent(agent: Any, name: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Execute agent with error handling."""
            try:
                result = agent.process(data)
                result["agent"] = name
                return result
            except Exception as e:
                log.error(f"Agent {name} failed: {e}")
                return {
                    "agent": name,
                    "ok": False,
                    "error": str(e)
                }
        
        def process_result(result: Dict[str, Any], reports: List[Dict], edges: List[Dict]) -> None:
            """Process agent result."""
            if not isinstance(result, dict):
                raise ValueError("Invalid agent result")
                
            reports.append(result)
            details = result.get("details", {})
            
            self._process_metrics(details, input_data)
            if "edges" in details:
                edges.extend(details["edges"])
                
        # Validate input
        if not isinstance(input_data, dict):
            return {
                "reports": [],
                "input_audit": {"issues": ["invalid_input"]},
                "decision": "BLOCK"
            }
            
        # Reset state
        self._initialize_state()
        reports: List[Dict] = []
        edges: List[Dict] = []
        
        try:
            # Sanitize input
            sanitized = self._sanitize_input(input_data)
            audit = sanitized.pop("_audit", {})
            
            if not audit.get("safe", False):
                return {
                    "reports": [],
                    "input_audit": audit,
                    "decision": "BLOCK"
                }
                
            # Process agents in parallel
            with ThreadPoolExecutor(max_workers=len(self.agents)) as executor:
                futures = {
                    executor.submit(process_agent, agent, name, sanitized): name
                    for name, agent in self.agents.items()
                }
                
                # Handle results
                for future in as_completed(futures):
                    try:
                        result = future.result(timeout=self._timeout_sec)
                        process_result(result, reports, edges)
                    except Exception as e:
                        name = futures[future]
                        reports.append({
                            "agent": name,
                            "ok": False,
                            "error": str(e)
                        })
                        
            # Build response
            decision = self._make_decision()
            response = {
                "reports": reports,
                "input_audit": audit,
                "memory_snapshot": {},
                "explainability_graph": {
                    "edges": edges,
                    "nodes": sorted({e["from"] for e in edges} | 
                                {e["to"] for e in edges})
                }
            }
            
            # Update MetaJudge
            for report in reports:
                if report.get("agent") == "MetaJudge":
                    report["details"]["decision"] = decision
                    break
                    
            return response
            
        except Exception as e:
            log.error(f"Council dispatch failed: {e}")
            return {
                "reports": [],
                "input_audit": {"issues": [str(e)]},
                "decision": "BLOCK"
            }
            
    def save_memory(self, key: str, value: Any) -> None:
        """Save to memory."""
        with self._memory_lock:
            self._memory[key] = value
            
    def load_memory(self, key: str) -> Optional[Any]:
        """Load from memory."""
        with self._memory_lock:
            return self._memory.get(key)
            
    def clear_memory(self) -> None:
        """Clear all memory."""
        with self._memory_lock:
            self._memory.clear()

class EchoSeedAgent(BaseAgent):
    """Initial seed agent that processes raw input."""
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Echo input data with length-based severity."""
        text = str(input_data.get("text", ""))
        len_severity = min(1.0, len(text) / 1000)
        
        details = {
            "len_s": len_severity,
            "sev_proxy": 0.2 + (0.8 * len_severity)
        }
        
        return {
            "agent": self.name,
            "details": details,
            "ok": True
        }

class ShortTermAgent(BaseAgent):
    """Handles immediate context analysis."""
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process immediate context."""
        text = str(input_data.get("text", ""))
        intent = str(input_data.get("intent", ""))
        
        urgency_words = {"now", "asap", "urgent", "immediately", "rush"}
        found_urgent = any(w in text.lower() or w in intent.lower() 
                        for w in urgency_words)
        
        details = {
            "urgency_now": 0.8 if found_urgent else 0.0,
            "severity_now": 0.0
        }
        
        return {
            "agent": self.name,
            "details": details,
            "ok": True
        }

class MidTermAgent(BaseAgent):
    """Handles medium-term trend analysis."""
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process medium-term trends."""
        stress_hist = []
        try:
            mem_entries = self.memory.audit()
            for entry in mem_entries.values():
                if "stress" in str(entry.get("value", "")):
                    stress_hist.append(float(entry.get("stress", 0.0)))
        except:
            pass
            
        stress_ema = sum(stress_hist[-5:]) / max(1, len(stress_hist[-5:])) if stress_hist else 0.0
        
        details = {
            "stress_ema": stress_ema,
            "sev_ema": 0.0,
            "sev_slope": 0.0,
            "forecast_mid": 0.0
        }
        
        return {
            "agent": self.name,
            "details": details,
            "ok": True
        }

class LongTermArchivistAgent(BaseAgent):
    """Maintains long-term memory and context."""
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process long-term context."""
        mem_audit = self.memory.audit()
        integrities = [rec.get("integrity", 1.0) for rec in mem_audit.values()]
        avg_integrity = sum(integrities)/max(1, len(integrities))
        
        drift_long = 0.3 + (0.7 * (1.0 - avg_integrity))
        
        details = {
            "avg_memory_integrity": avg_integrity,
            "drift_long": drift_long,
            "decision_bias": 1.0,
            "comp_ema": drift_long
        }
        
        return {
            "agent": self.name,
            "details": details,
            "ok": True
        }

class BiofeedbackAgent(BaseAgent):
    """Processes biometric signals."""
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process biometric signals."""
        bio = input_data.get("_signals", {}).get("bio", {})
        
        hr = float(bio.get("heart_rate", 60))
        hr_s = min(1.0, max(0.0, (hr - 60) / 60))
        
        hrv = float(bio.get("hrv", 50))
        hrv_s = min(1.0, max(0.0, 1.0 - (hrv / 50)))
        
        gsr = float(bio.get("gsr", 5))
        gsr_s = min(1.0, max(0.0, gsr / 15))
        
        voice = float(bio.get("voice_tension", 0.0))
        
        stress = max(
            hr_s,
            hrv_s,
            gsr_s,
            voice,
            float(bio.get("stress", 0.0))
        )
        
        details = {
            "hr_s": hr_s,
            "hrv_s": hrv_s,
            "gsr_s": gsr_s,
            "voice_s": voice,
            "operator_stress": stress
        }
        
        return {
            "agent": self.name,
            "details": details,
            "ok": True
        }

class EnvSignalAgent(BaseAgent):
    """Processes environmental context signals."""
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process environmental signals."""
        env = input_data.get("_signals", {}).get("env", {})
        
        details = {
            "context_risk": float(env.get("context_risk", 0.0)),
            "incident_sev": float(env.get("incident_sev", 0.0)),
            "market_vol": float(env.get("market_volatility", 0.0)),
            "network_anom": float(env.get("network_anomalies", 0.0))
        }
        
        return {
            "agent": self.name,
            "details": details,
            "ok": True
        }

class ContextConflictAgent(BaseAgent):
    """Detects conflicts between intent and context."""
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process context conflicts."""
        text = str(input_data.get("text", ""))
        intent = str(input_data.get("intent", ""))
        
        rush_words = {"fast", "quick", "hurry", "rush"}
        caution_words = {"careful", "safe", "slow", "check"}
        
        rush = any(w in text.lower() or w in intent.lower() for w in rush_words)
        caution = any(w in text.lower() or w in intent.lower() for w in caution_words)
        
        conflict = 0.8 if (rush and caution) else 0.0
        
        details = {
            "intent_conflict": conflict,
            "confidence": 0.8
        }
        
        return {
            "agent": self.name,
            "details": details,
            "ok": True
        }

class TimeScaleCoordinator(BaseAgent):
    """Coordinates short/mid/long term signals."""
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Coordinate timescale signals."""
        last_decision = str(input_data.get("_last_decision", "PROCEED"))
        
        caution_signal = 0.0
        if last_decision == "PROCEED_WITH_CAUTION":
            caution_signal = 0.4
        elif last_decision == "BLOCK":
            caution_signal = 0.8
            
        edges = []
        for agent in ["ShortTerm", "MidTerm", "LongArchivist"]:
            edges.append({
                "from": agent,
                "to": "MetaJudge",
                "weight": 0.33
            })
            
        details = {
            "caution_fused": caution_signal,
            "edges": edges
        }
        
        return {
            "agent": self.name,
            "details": details,
            "ok": True
        }

class MetaJudgeAgent(BaseAgent):
    """Final decision coordination."""
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process final decision."""
        all_edges = []
        for agent in ["EchoSeed", "ShortTerm", "MidTerm", "LongArchivist",
                     "BiofeedbackAgent", "EnvSignalAgent", "ContextConflictAgent"]:
            all_edges.append({
                "from": agent,
                "to": self.name,
                "weight": 0.3
            })
            
        details = {
            "decision": "PROCEED",
            "edges": all_edges
        }
        
        return {
            "agent": self.name,
            "details": details,
            "ok": True
        }

class AegisCouncil:
    """Core coordination for the Aegis diagnostic system."""

    def __init__(self):
        """Initialize the council."""
        self._memory = NexusMemory()
        self._executor = ThreadPoolExecutor(max_workers=10)
        self._agents = []
        self._active = True
        self._timescales = AegisTimescales()
        
    def add_agent(self, agent: BaseAgent) -> None:
        """Add an agent to the council."""
        self._agents.append(agent)
        
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input through all agents."""
        if not isinstance(input_data, dict):
            raise ValueError("Input must be a dictionary")
            
        return self._timescales.dispatch(input_data)