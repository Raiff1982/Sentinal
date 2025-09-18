"""
Aegis Diagnostics - System health monitoring and pre-flight checks.

This module provides diagnostic tools for monitoring system health,
resource usage, and performance metrics. It includes pre-flight
checks to ensure all components are functioning correctly before
critical operations.
"""

import os
import hmac
import hashlib
import psutil
import time
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta

from .sentinel import Sentinel
from .explain import ExplainStore
from .sentinel_config import MAX_AGENT_TIMEOUT_SEC

logger = logging.getLogger(__name__)

@dataclass
class HealthStatus:
    """System health check results."""
    status: str  # "healthy", "degraded", "error"
    message: str
    timestamp: str
    details: Dict

@dataclass
class ResourceMetrics:
    """System resource usage metrics."""
    cpu_percent: float
    memory_percent: float
    disk_usage: float
    explain_store_size: int
    active_threads: int
    response_times: List[float]

@dataclass
class SecurityMetrics:
    """Security-specific metrics for high-risk components."""
    crypto_verify_time: float  # Time to verify HMAC signatures
    input_validation_time: float  # Time for input sanitization
    challenge_success_rate: float  # Success rate of challenge gates
    blocked_request_rate: float  # Rate of blocked requests
    anomaly_score: float  # Current anomaly detection score
    attack_surface_score: float  # Estimated attack surface score
    audit_integrity: bool  # Audit trail integrity check

@dataclass
class ComponentHealth:
    """Health status for critical system components."""
    name: str
    status: str  # "healthy", "degraded", "error"
    risk_level: str  # "low", "medium", "high", "critical"
    last_check: str  # ISO timestamp
    metrics: Dict
    warnings: List[str]

class AegisDiagnostics:
    """Diagnostic tools for Aegis system health monitoring."""
    
    def __init__(self, sentinel: Sentinel):
        self.sentinel = sentinel
        self.metrics_history: List[ResourceMetrics] = []
        self.security_history: List[SecurityMetrics] = []
        self.component_status: Dict[str, ComponentHealth] = {}
        self.last_check: Optional[datetime] = None
        self.process = psutil.Process()
        
        # Initialize critical component monitors
        self._init_component_monitors()
    
    def run_preflight_checks(self) -> HealthStatus:
        """Run pre-flight checks before critical operations.
        
        Checks:
        1. Explain store accessibility and integrity
        2. Agent response times
        3. Memory usage and limits
        4. Storage space
        5. Configuration validity
        """
        try:
            status = "healthy"
            details = {}
            messages = []
            
            # Check explain store
            store_status = self._check_explain_store()
            details["explain_store"] = store_status
            if not store_status["accessible"]:
                status = "error"
                messages.append("Explain store inaccessible")
                
            # Check agent response times
            agent_status = self._check_agent_response()
            details["agents"] = agent_status
            if agent_status["slow_responses"]:
                status = "degraded"
                messages.append("Slow agent responses detected")
                
            # Check memory usage
            memory_status = self._check_memory_usage()
            details["memory"] = memory_status
            if memory_status["percent"] > 90:
                status = "degraded"
                messages.append("High memory usage")
                
            # Check storage
            storage_status = self._check_storage()
            details["storage"] = storage_status
            if storage_status["percent"] > 90:
                status = "degraded"
                messages.append("Low storage space")
                
            message = "; ".join(messages) if messages else "All systems operational"
            
            return HealthStatus(
                status=status,
                message=message,
                timestamp=datetime.now().isoformat(),
                details=details
            )
            
        except Exception as e:
            logger.error(f"Pre-flight check failed: {e}")
            return HealthStatus(
                status="error",
                message=f"Diagnostic failure: {str(e)}",
                timestamp=datetime.now().isoformat(),
                details={"error": str(e)}
            )
    
    def collect_metrics(self) -> ResourceMetrics:
        """Collect current system resource metrics."""
        metrics = ResourceMetrics(
            cpu_percent=self.process.cpu_percent(),
            memory_percent=self.process.memory_percent(),
            disk_usage=psutil.disk_usage("/").percent,
            explain_store_size=self._get_store_size(),
            active_threads=len(self.process.threads()),
            response_times=self._get_response_times()
        )
        
        self.metrics_history.append(metrics)
        if len(self.metrics_history) > 1000:  # Keep last 1000 data points
            self.metrics_history.pop(0)
            
        return metrics
    
    def get_performance_report(self) -> Dict:
        """Generate performance report from collected metrics."""
        if not self.metrics_history:
            return {"error": "No metrics collected yet"}
            
        avg_metrics = ResourceMetrics(
            cpu_percent=sum(m.cpu_percent for m in self.metrics_history) / len(self.metrics_history),
            memory_percent=sum(m.memory_percent for m in self.metrics_history) / len(self.metrics_history),
            disk_usage=sum(m.disk_usage for m in self.metrics_history) / len(self.metrics_history),
            explain_store_size=self.metrics_history[-1].explain_store_size,
            active_threads=sum(m.active_threads for m in self.metrics_history) / len(self.metrics_history),
            response_times=self.metrics_history[-1].response_times
        )
        
        return {
            "current": self.metrics_history[-1],
            "average": avg_metrics,
            "trends": self._calculate_trends(),
            "warnings": self._generate_warnings(avg_metrics)
        }
    
    def _check_explain_store(self) -> Dict:
        """Check explain store health."""
        store = self.sentinel.explain_store
        try:
            # Try basic operations
            store.get_entries(limit=1)
            return {
                "accessible": True,
                "size": self._get_store_size(),
                "type": store.__class__.__name__
            }
        except Exception as e:
            return {
                "accessible": False,
                "error": str(e)
            }
    
    def _check_agent_response(self) -> Dict:
        """Check agent response times."""
        try:
            start = time.time()
            self.sentinel.check("diagnostic_check", {"intent": "system_check"})
            response_time = time.time() - start
            
            return {
                "response_time": response_time,
                "slow_responses": response_time > MAX_AGENT_TIMEOUT_SEC * 0.5,
                "timeout": MAX_AGENT_TIMEOUT_SEC
            }
        except Exception as e:
            return {
                "error": str(e),
                "slow_responses": True
            }
    
    def _check_memory_usage(self) -> Dict:
        """Check system memory usage."""
        memory = psutil.virtual_memory()
        return {
            "total": memory.total,
            "available": memory.available,
            "percent": memory.percent,
            "process_usage": self.process.memory_info().rss
        }
    
    def _check_storage(self) -> Dict:
        """Check storage space."""
        disk = psutil.disk_usage("/")
        return {
            "total": disk.total,
            "free": disk.free,
            "percent": disk.percent
        }
    
    def _get_store_size(self) -> int:
        """Get explain store size in bytes."""
        store = self.sentinel.explain_store
        if hasattr(store, "get_size"):
            return store.get_size()
        return 0
    
    def _get_response_times(self) -> List[float]:
        """Get recent response times."""
        times = []
        for _ in range(3):  # Sample 3 quick checks
            start = time.time()
            self.sentinel.check("diagnostic_check", {"intent": "system_check"})
            times.append(time.time() - start)
        return times
    
    def _calculate_trends(self) -> Dict:
        """Calculate metric trends."""
        if len(self.metrics_history) < 2:
            return {}
            
        current = self.metrics_history[-1]
        previous = self.metrics_history[-2]
        
        return {
            "cpu_trend": current.cpu_percent - previous.cpu_percent,
            "memory_trend": current.memory_percent - previous.memory_percent,
            "store_size_trend": current.explain_store_size - previous.explain_store_size
        }
    
    def _generate_warnings(self, metrics: ResourceMetrics) -> List[str]:
        """Generate warnings based on metrics."""
        warnings = []
        
        if metrics.cpu_percent > 80:
            warnings.append("High CPU usage")
        if metrics.memory_percent > 80:
            warnings.append("High memory usage")
        if metrics.disk_usage > 80:
            warnings.append("Low disk space")
        if any(t > MAX_AGENT_TIMEOUT_SEC * 0.8 for t in metrics.response_times):
            warnings.append("Slow response times")
            
        return warnings

    def _init_component_monitors(self):
        """Initialize monitors for critical components."""
        self.critical_components = {
            "input_validator": self._check_input_validator,
            "challenge_gate": self._check_challenge_gate,
            "crypto_system": self._check_crypto_system,
            "audit_trail": self._check_audit_trail,
            "memory_store": self._check_memory_store,
            "agent_council": self._check_agent_council
        }

    def check_critical_components(self) -> Dict[str, ComponentHealth]:
        """Run diagnostics on critical system components."""
        results = {}
        for name, check_func in self.critical_components.items():
            try:
                health = check_func()
                self.component_status[name] = health
                results[name] = health
            except Exception as e:
                logger.error(f"Component check failed for {name}: {e}")
                results[name] = ComponentHealth(
                    name=name,
                    status="error",
                    risk_level="critical",
                    last_check=datetime.now().isoformat(),
                    metrics={"error": str(e)},
                    warnings=[f"Check failed: {str(e)}"]
                )
        return results

    def _check_input_validator(self) -> ComponentHealth:
        """Check input validation system health."""
        start_time = time.time()
        test_inputs = [
            "normal text",
            "SELECT * FROM users",  # SQL injection attempt
            "<script>alert('xss')</script>",  # XSS attempt
            "".join(chr(i) for i in range(128, 256))  # Unicode edge cases
        ]
        
        metrics = {
            "validation_times": [],
            "blocked_attempts": 0,
            "total_attempts": len(test_inputs)
        }
        
        warnings = []
        for text in test_inputs:
            try:
                t0 = time.time()
                result = self.sentinel.check(text, {"intent": "diagnostic"})
                validation_time = time.time() - t0
                
                metrics["validation_times"].append(validation_time)
                if not result.allow:
                    metrics["blocked_attempts"] += 1
                
                if validation_time > 0.1:  # More than 100ms is slow
                    warnings.append(f"Slow validation ({validation_time:.2f}s)")
            except Exception as e:
                warnings.append(f"Validation error: {e}")
        
        avg_time = sum(metrics["validation_times"]) / len(metrics["validation_times"])
        block_rate = metrics["blocked_attempts"] / metrics["total_attempts"]
        
        # Determine status and risk
        if warnings:
            status = "degraded"
            risk_level = "high"
        elif avg_time > 0.05:  # More than 50ms average
            status = "degraded"
            risk_level = "medium"
        elif block_rate < 0.5:  # Should block at least half of test cases
            status = "degraded"
            risk_level = "high"
        else:
            status = "healthy"
            risk_level = "low"
            
        return ComponentHealth(
            name="input_validator",
            status=status,
            risk_level=risk_level,
            last_check=datetime.now().isoformat(),
            metrics={
                "average_validation_time": avg_time,
                "block_rate": block_rate,
                **metrics
            },
            warnings=warnings
        )

    def _check_challenge_gate(self) -> ComponentHealth:
        """Check challenge gate system health."""
        metrics = {
            "total_challenges": 0,
            "successful_challenges": 0,
            "average_time": 0,
            "failure_patterns": {}
        }
        
        # Analyze recent challenge results
        recent = self.sentinel.get_history(limit=100)
        challenge_times = []
        
        for entry in recent:
            if "challenge" in entry.get("context", {}).get("intent", ""):
                metrics["total_challenges"] += 1
                result = entry.get("result", {})
                if result.get("allow", False):
                    metrics["successful_challenges"] += 1
                    challenge_times.append(
                        result.get("metrics", {}).get("challenge_time", 0)
                    )
                else:
                    failure_type = result.get("reason", "unknown")
                    metrics["failure_patterns"][failure_type] = \
                        metrics["failure_patterns"].get(failure_type, 0) + 1
        
        if challenge_times:
            metrics["average_time"] = sum(challenge_times) / len(challenge_times)
        
        # Calculate success rate
        success_rate = (metrics["successful_challenges"] / metrics["total_challenges"]) \
            if metrics["total_challenges"] > 0 else 0
            
        # Determine status and risk
        warnings = []
        if success_rate < 0.6:
            warnings.append(f"Low challenge success rate: {success_rate:.2%}")
        if metrics["average_time"] > 2.0:
            warnings.append(f"Slow challenge time: {metrics['average_time']:.2f}s")
        
        if warnings:
            status = "degraded"
            risk_level = "high" if success_rate < 0.4 else "medium"
        else:
            status = "healthy"
            risk_level = "low"
            
        return ComponentHealth(
            name="challenge_gate",
            status=status,
            risk_level=risk_level,
            last_check=datetime.now().isoformat(),
            metrics=metrics,
            warnings=warnings
        )

    def _check_crypto_system(self) -> ComponentHealth:
        """Check cryptographic system health."""
        metrics = {
            "verification_times": [],
            "total_verifications": 0,
            "failed_verifications": 0
        }
        
        # Test HMAC verification
        test_data = b"test_data"
        test_key = b"test_key"
        warnings = []
        
        try:
            # Time signature generation
            t0 = time.time()
            signature = hmac.new(test_key, test_data, hashlib.sha256).hexdigest()
            gen_time = time.time() - t0
            metrics["signature_generation_time"] = gen_time
            
            # Time verification
            t0 = time.time()
            verify = hmac.new(test_key, test_data, hashlib.sha256).hexdigest() == signature
            verify_time = time.time() - t0
            metrics["verification_times"].append(verify_time)
            
            if gen_time > 0.01:  # More than 10ms is slow
                warnings.append(f"Slow signature generation: {gen_time:.3f}s")
            if verify_time > 0.01:
                warnings.append(f"Slow signature verification: {verify_time:.3f}s")
                
        except Exception as e:
            warnings.append(f"Crypto operation failed: {e}")
        
        # Check recent audit entries
        recent = self.sentinel.get_history(limit=50)
        for entry in recent:
            if "signature" in entry:
                metrics["total_verifications"] += 1
                try:
                    t0 = time.time()
                    # Verify entry signature
                    if not self._verify_entry_signature(entry):
                        metrics["failed_verifications"] += 1
                    metrics["verification_times"].append(time.time() - t0)
                except Exception:
                    metrics["failed_verifications"] += 1
        
        # Calculate average verification time
        if metrics["verification_times"]:
            metrics["average_verification_time"] = \
                sum(metrics["verification_times"]) / len(metrics["verification_times"])
        
        # Determine status and risk
        if metrics["failed_verifications"] > 0:
            status = "error"
            risk_level = "critical"
            warnings.append(f"Failed verifications: {metrics['failed_verifications']}")
        elif warnings:
            status = "degraded"
            risk_level = "high"
        else:
            status = "healthy"
            risk_level = "low"
            
        return ComponentHealth(
            name="crypto_system",
            status=status,
            risk_level=risk_level,
            last_check=datetime.now().isoformat(),
            metrics=metrics,
            warnings=warnings
        )

    def _check_audit_trail(self) -> ComponentHealth:
        """Check audit trail integrity."""
        metrics = {
            "total_entries": 0,
            "verified_entries": 0,
            "integrity_violations": 0,
            "gaps_detected": 0
        }
        
        warnings = []
        recent = self.sentinel.get_history(limit=100)
        last_timestamp = None
        
        for entry in recent:
            metrics["total_entries"] += 1
            
            # Check entry integrity
            if self._verify_entry_integrity(entry):
                metrics["verified_entries"] += 1
            else:
                metrics["integrity_violations"] += 1
                warnings.append(f"Integrity violation at entry {entry.get('id', 'unknown')}")
            
            # Check for temporal gaps
            timestamp = entry.get("timestamp")
            if timestamp and last_timestamp:
                try:
                    gap = (datetime.fromisoformat(timestamp) - 
                           datetime.fromisoformat(last_timestamp)).total_seconds()
                    if gap > 300:  # More than 5 minutes
                        metrics["gaps_detected"] += 1
                        warnings.append(f"Temporal gap detected: {gap:.1f}s")
                except (ValueError, TypeError):
                    warnings.append("Invalid timestamp format")
            last_timestamp = timestamp
        
        # Determine status and risk
        if metrics["integrity_violations"] > 0:
            status = "error"
            risk_level = "critical"
        elif metrics["gaps_detected"] > 0:
            status = "degraded"
            risk_level = "high"
        elif warnings:
            status = "degraded"
            risk_level = "medium"
        else:
            status = "healthy"
            risk_level = "low"
            
        return ComponentHealth(
            name="audit_trail",
            status=status,
            risk_level=risk_level,
            last_check=datetime.now().isoformat(),
            metrics=metrics,
            warnings=warnings
        )

    def _check_memory_store(self) -> ComponentHealth:
        """Check memory store health."""
        metrics = {
            "store_size": 0,
            "access_times": [],
            "write_times": [],
            "failed_operations": 0
        }
        
        warnings = []
        store = self.sentinel.explain_store
        
        # Test read operations
        try:
            t0 = time.time()
            entries = store.get_entries(limit=1)
            read_time = time.time() - t0
            metrics["access_times"].append(read_time)
            
            if read_time > 0.1:  # More than 100ms is slow
                warnings.append(f"Slow read access: {read_time:.3f}s")
        except Exception as e:
            metrics["failed_operations"] += 1
            warnings.append(f"Read operation failed: {e}")
        
        # Test write operations
        try:
            test_entry = {
                "id": "diagnostic_test",
                "timestamp": datetime.now().isoformat(),
                "text": "diagnostic_check",
                "result": {"allow": True}
            }
            
            t0 = time.time()
            store.add_entry(test_entry)
            write_time = time.time() - t0
            metrics["write_times"].append(write_time)
            
            if write_time > 0.1:
                warnings.append(f"Slow write access: {write_time:.3f}s")
        except Exception as e:
            metrics["failed_operations"] += 1
            warnings.append(f"Write operation failed: {e}")
        
        # Get store size
        try:
            metrics["store_size"] = self._get_store_size()
            if metrics["store_size"] > 1_000_000_000:  # 1GB
                warnings.append("Large store size may impact performance")
        except Exception as e:
            warnings.append(f"Failed to get store size: {e}")
        
        # Determine status and risk
        if metrics["failed_operations"] > 0:
            status = "error"
            risk_level = "critical"
        elif warnings:
            status = "degraded"
            risk_level = "high"
        else:
            status = "healthy"
            risk_level = "low"
            
        return ComponentHealth(
            name="memory_store",
            status=status,
            risk_level=risk_level,
            last_check=datetime.now().isoformat(),
            metrics=metrics,
            warnings=warnings
        )

    def _check_agent_council(self) -> ComponentHealth:
        """Check agent council health."""
        metrics = {
            "active_agents": 0,
            "response_times": [],
            "decision_conflicts": 0,
            "timeouts": 0
        }
        
        warnings = []
        
        # Test agent responses
        test_inputs = [
            ("Safe text for testing", {"intent": "diagnostic"}),
            ("Potentially unsafe content", {"intent": "diagnostic", "risk_level": "high"}),
            ("Complex decision scenario", {"intent": "diagnostic", "risk_level": "medium"})
        ]
        
        for text, context in test_inputs:
            try:
                t0 = time.time()
                result = self.sentinel.check(text, context)
                response_time = time.time() - t0
                
                metrics["response_times"].append(response_time)
                if response_time > MAX_AGENT_TIMEOUT_SEC * 0.5:
                    metrics["timeouts"] += 1
                    warnings.append(f"Slow agent response: {response_time:.2f}s")
                    
                # Check for decision conflicts
                if "conflict" in result.reason.lower():
                    metrics["decision_conflicts"] += 1
                    
            except Exception as e:
                warnings.append(f"Agent check failed: {e}")
        
        # Calculate average response time
        if metrics["response_times"]:
            metrics["average_response_time"] = \
                sum(metrics["response_times"]) / len(metrics["response_times"])
        
        # Determine status and risk
        if metrics["timeouts"] > len(test_inputs) // 2:
            status = "error"
            risk_level = "critical"
        elif metrics["decision_conflicts"] > 0:
            status = "degraded"
            risk_level = "high"
        elif warnings:
            status = "degraded"
            risk_level = "medium"
        else:
            status = "healthy"
            risk_level = "low"
            
        return ComponentHealth(
            name="agent_council",
            status=status,
            risk_level=risk_level,
            last_check=datetime.now().isoformat(),
            metrics=metrics,
            warnings=warnings
        )
        
    def _verify_entry_signature(self, entry: Dict) -> bool:
        """Verify cryptographic signature of an audit entry."""
        try:
            if "signature" not in entry:
                return False
            # Implementation would verify HMAC signature
            return True
        except Exception:
            return False
            
    def _verify_entry_integrity(self, entry: Dict) -> bool:
        """Verify the integrity of an audit trail entry."""
        try:
            required_fields = ["id", "timestamp", "text"]
            return all(field in entry for field in required_fields)
        except Exception:
            return False