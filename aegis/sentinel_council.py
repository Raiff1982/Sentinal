"""
Core council implementation for coordinating Aegis agents.
"""
import threading
import concurrent.futures
from typing import Dict, List, Any, Optional
from datetime import datetime

from .sentinel_config import MAX_AGENT_TIMEOUT_SEC

class AegisAgent:
    """Base class for all Aegis agents."""
    def __init__(self, name: str, memory: Any):
        self.name = name
        self.memory = memory
        
    def analyze(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run agent analysis on input data."""
        raise NotImplementedError

class EchoSeedAgent(AegisAgent):
    """Agent that simply echoes input for testing."""
    def analyze(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "summary": "Echo agent response",
            "influence": 0.1,
            "reliability": 1.0,
            "ok": True,
            "details": input_data
        }

class ShortTermAgent(AegisAgent):
    """Analyzes recent context and short-term patterns."""
    
    def analyze(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze short-term signals and patterns.
        
        Examines recent events and immediate context to detect
        rapid changes or concerning patterns.
        
        Args:
            input_data: Current input with signals
            
        Returns:
            Analysis results with influence and reliability
        """
        # Get current signals
        bio = (input_data.get("_signals", {}) or {}).get("bio", {})
        env = (input_data.get("_signals", {}) or {}).get("env", {})
        
        # Calculate stress level
        stress = float(bio.get("stress", 0.3))
        heart_rate = float(bio.get("heart_rate", 70.0))
        stress_rising = heart_rate > 90 or stress > 0.7
        
        # Calculate risk level
        risk = float(env.get("context_risk", 0.2))
        incident_sev = float(env.get("incident_sev", 0.0))
        risk_rising = incident_sev > 0.5 or risk > 0.6
        
        # Get recent memory entries
        recent = []
        if hasattr(self.memory, "nexus"):
            data = self.memory.nexus.query({
                "category": "short_term",
                "limit": 5,
                "order": "desc"
            })
            recent = [d.get("data", {}) for d in data]
        
        # Detect concerning patterns
        pattern_risk = 0.0
        if len(recent) >= 3:
            stress_trend = [float(r.get("stress", 0.0)) for r in recent]
            risk_trend = [float(r.get("risk", 0.0)) for r in recent]
            
            if all(s > 0.5 for s in stress_trend[:3]):
                pattern_risk = max(pattern_risk, 0.7)
            if all(r > 0.4 for r in risk_trend[:3]):
                pattern_risk = max(pattern_risk, 0.6)
        
        # Store current state
        if hasattr(self.memory, "nexus"):
            self.memory.nexus.ingest(
                "short_term",
                "state",
                {"stress": stress, "risk": risk},
                ttl_sec=300  # 5 minute retention
            )
        
        # Calculate severity
        severity = max(
            pattern_risk,
            0.8 if (stress_rising and risk_rising) else 0.0,
            0.6 if (stress_rising or risk_rising) else 0.0,
            0.3 if (stress > 0.5 or risk > 0.4) else 0.0
        )
        
        return {
            "summary": "Short-term pattern analysis",
            "influence": 0.3,  # Medium influence
            "reliability": 0.9,  # High reliability for recent data
            "severity": severity,
            "details": {
                "stress_rising": stress_rising,
                "risk_rising": risk_rising,
                "pattern_risk": pattern_risk,
                "current": {"stress": stress, "risk": risk}
            },
            "explain_edges": [
                {"from": "ShortTerm", "to": "stress", "weight": stress},
                {"from": "ShortTerm", "to": "risk", "weight": risk},
                {"from": "ShortTerm", "to": "patterns", "weight": pattern_risk}
            ],
            "ok": True
        }

class MidTermAgent(AegisAgent):
    """Analyzes medium-term trends and patterns."""
    
    def analyze(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze medium-term trends and patterns.
        
        Examines hour-to-day patterns to identify concerning
        trajectories or accumulated risk.
        
        Args:
            input_data: Current input with signals
            
        Returns:
            Analysis results with influence and reliability
        """
        # Get current state
        signals = input_data.get("_signals", {}) or {}
        bio = signals.get("bio", {})
        env = signals.get("env", {})
        
        # Get historical data
        history = []
        if hasattr(self.memory, "nexus"):
            data = self.memory.nexus.query({
                "category": "mid_term",
                "limit": 24,  # Last 24 entries
                "order": "desc"
            })
            history = [d.get("data", {}) for d in data]
        
        # Calculate stress trajectory
        stress_history = [float(h.get("stress", 0.0)) for h in history]
        current_stress = float(bio.get("stress", 0.3))
        stress_trajectory = 0.0
        if len(stress_history) >= 12:
            recent_avg = sum(stress_history[:6]) / 6
            older_avg = sum(stress_history[6:12]) / 6
            stress_trajectory = max(0.0, min(1.0, 2.0 * (recent_avg - older_avg)))
        
        # Calculate risk trajectory
        risk_history = [float(h.get("risk", 0.0)) for h in history]
        current_risk = float(env.get("context_risk", 0.2)) 
        risk_trajectory = 0.0
        if len(risk_history) >= 12:
            recent_avg = sum(risk_history[:6]) / 6
            older_avg = sum(risk_history[6:12]) / 6
            risk_trajectory = max(0.0, min(1.0, 2.0 * (recent_avg - older_avg)))
        
        # Store current state 
        if hasattr(self.memory, "nexus"):
            self.memory.nexus.ingest(
                "mid_term",
                "state",
                {
                    "stress": current_stress,
                    "risk": current_risk,
                    "stress_trajectory": stress_trajectory,
                    "risk_trajectory": risk_trajectory
                },
                ttl_sec=7200  # 2 hour retention
            )
        
        # Calculate severity
        severity = max(
            0.8 if (stress_trajectory > 0.6 and risk_trajectory > 0.5) else 0.0,
            0.6 if (stress_trajectory > 0.4 or risk_trajectory > 0.3) else 0.0,
            0.4 if (current_stress > 0.6 or current_risk > 0.5) else 0.0,
            min(1.0, (stress_trajectory + risk_trajectory) / 2)
        )
        
        return {
            "summary": "Medium-term trend analysis",
            "influence": 0.25,  # Lower-medium influence
            "reliability": 0.85,  # Good reliability
            "severity": severity,
            "details": {
                "stress_trajectory": stress_trajectory,
                "risk_trajectory": risk_trajectory,
                "current": {
                    "stress": current_stress,
                    "risk": current_risk
                },
                "history_length": len(history)
            },
            "explain_edges": [
                {"from": "MidTerm", "to": "stress_trend", "weight": stress_trajectory},
                {"from": "MidTerm", "to": "risk_trend", "weight": risk_trajectory},
                {"from": "MidTerm", "to": "current", "weight": max(current_stress, current_risk)}
            ],
            "ok": True
        }

class LongTermArchivistAgent(AegisAgent):
    """Maintains long-term memory and analyzes historical patterns."""
    
    def analyze(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze long-term patterns and systemic risks.
        
        Examines days-to-weeks patterns to identify systemic issues,
        cyclical risks, and long-term stability concerns.
        
        Args:
            input_data: Current input with signals
            
        Returns:
            Analysis results with influence and reliability
        """
        # Get current state
        signals = input_data.get("_signals", {}) or {}
        bio = signals.get("bio", {})
        env = signals.get("env", {})
        
        # Get archived data
        archives = []
        if hasattr(self.memory, "nexus"):
            data = self.memory.nexus.query({
                "category": "long_term",
                "limit": 168,  # Last week of hourly data
                "order": "desc" 
            })
            archives = [d.get("data", {}) for d in data]
        
        # Calculate baseline metrics
        stress_baseline = 0.3  # Default baseline
        risk_baseline = 0.2    # Default baseline
        if len(archives) >= 24:
            stress_values = [float(a.get("stress", 0.0)) for a in archives]
            risk_values = [float(a.get("risk", 0.0)) for a in archives]
            stress_baseline = sum(stress_values) / len(stress_values)
            risk_baseline = sum(risk_values) / len(risk_values)
        
        # Current deviations
        current_stress = float(bio.get("stress", 0.3))
        current_risk = float(env.get("context_risk", 0.2))
        stress_deviation = max(0.0, current_stress - stress_baseline)
        risk_deviation = max(0.0, current_risk - risk_baseline)
        
        # Check for cyclical patterns
        cycle_severity = 0.0
        if len(archives) >= 168:  # Full week
            # Simple check for day/night cycles
            day_stress = []
            night_stress = []
            for i, a in enumerate(archives):
                hour = i % 24
                if 8 <= hour <= 20:  # Day hours
                    day_stress.append(float(a.get("stress", 0.0)))
                else:  # Night hours
                    night_stress.append(float(a.get("stress", 0.0)))
                    
            if day_stress and night_stress:
                day_avg = sum(day_stress) / len(day_stress)
                night_avg = sum(night_stress) / len(night_stress)
                cycle_delta = abs(day_avg - night_avg)
                if cycle_delta > 0.4:  # Large day/night difference
                    cycle_severity = 0.6
        
        # Store current state
        if hasattr(self.memory, "nexus"):
            self.memory.nexus.ingest(
                "long_term",
                "state",
                {
                    "stress": current_stress,
                    "risk": current_risk,
                    "stress_baseline": stress_baseline,
                    "risk_baseline": risk_baseline
                },
                ttl_sec=86400  # 24 hour retention
            )
        
        # Calculate severity
        severity = max(
            0.8 if (stress_deviation > 0.4 and risk_deviation > 0.3) else 0.0,
            0.7 if cycle_severity > 0.5 else 0.0,
            0.5 if (stress_deviation > 0.3 or risk_deviation > 0.2) else 0.0,
            min(1.0, (stress_deviation + risk_deviation + cycle_severity) / 3)
        )
        
        return {
            "summary": "Long-term systemic analysis",
            "influence": 0.2,  # Lower influence
            "reliability": 0.75,  # Moderate reliability
            "severity": severity,
            "details": {
                "stress_baseline": stress_baseline,
                "risk_baseline": risk_baseline,
                "deviations": {
                    "stress": stress_deviation,
                    "risk": risk_deviation
                },
                "cycle_severity": cycle_severity,
                "archive_length": len(archives)
            },
            "explain_edges": [
                {"from": "LongTerm", "to": "baseline", "weight": (stress_baseline + risk_baseline) / 2},
                {"from": "LongTerm", "to": "deviation", "weight": max(stress_deviation, risk_deviation)},
                {"from": "LongTerm", "to": "cycles", "weight": cycle_severity}
            ],
            "ok": True
        }

class TimeScaleCoordinator(AegisAgent):
    """Coordinates and fuses analysis across different time scales."""
    
    def analyze(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Fuse multi-timescale analysis results.
        
        Combines reports from short, medium, and long-term agents
        to build a comprehensive risk assessment.
        
        Args:
            input_data: Current input with signals and agent reports
            
        Returns:
            Fused analysis results
        """
        # Get agent reports
        reports = input_data.get("_agent_reports", [])
        
        # Extract by timescale
        short_term = next((r for r in reports if r.get("agent") == "ShortTermAgent"), {})
        mid_term = next((r for r in reports if r.get("agent") == "MidTermAgent"), {})
        long_term = next((r for r in reports if r.get("agent") == "LongTermArchivist"), {})
        
        # Get severity levels
        short_sev = float(short_term.get("severity", 0.0))
        mid_sev = float(mid_term.get("severity", 0.0))
        long_sev = float(long_term.get("severity", 0.0))
        
        # Calculate weighted severity
        severity = max(
            0.9 if (short_sev > 0.8 and mid_sev > 0.6) else 0.0,  # Acute crisis
            0.8 if (mid_sev > 0.7 and long_sev > 0.5) else 0.0,   # Emerging crisis
            0.7 if (short_sev > 0.6 and long_sev > 0.4) else 0.0,  # Unstable situation
            min(1.0, (0.5 * short_sev + 0.3 * mid_sev + 0.2 * long_sev))
        )
        
        # Extract key patterns
        patterns = {
            "stress_rising": short_term.get("details", {}).get("stress_rising", False),
            "risk_rising": short_term.get("details", {}).get("risk_rising", False),
            "stress_trajectory": mid_term.get("details", {}).get("stress_trajectory", 0.0),
            "risk_trajectory": mid_term.get("details", {}).get("risk_trajectory", 0.0),
            "baseline_deviation": long_term.get("details", {}).get("deviations", {})
        }
        
        # Calculate reliability
        reliabilities = [
            float(short_term.get("reliability", 0.0)),
            float(mid_term.get("reliability", 0.0)), 
            float(long_term.get("reliability", 0.0))
        ]
        reliability = sum(r for r in reliabilities if r > 0) / max(1, len([r for r in reliabilities if r > 0]))
        
        return {
            "summary": "Multi-timescale fusion analysis",
            "influence": 0.35,  # High influence
            "reliability": reliability,
            "severity": severity,
            "details": {
                "short_term_severity": short_sev,
                "mid_term_severity": mid_sev,
                "long_term_severity": long_sev,
                "patterns": patterns
            },
            "explain_edges": [
                {"from": "TimeScale", "to": "ShortTerm", "weight": short_sev},
                {"from": "TimeScale", "to": "MidTerm", "weight": mid_sev},
                {"from": "TimeScale", "to": "LongTerm", "weight": long_sev}
            ],
            "ok": True
        }

class MetaJudgeAgent(AegisAgent):
    """Makes final decisions based on all agent reports."""
    
    def analyze(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Make final judgment based on all agent reports.
        
        Combines agent reports using weighted severity calculation,
        incorporating influence and reliability factors.
        
        Args:
            input_data: Input data with agent reports
            
        Returns:
            Final decision and supporting analysis
        """
        reports = input_data.get("_agent_reports", [])
        
        # Extract key metrics
        metrics = []
        for report in reports:
            if not report.get("ok", False):
                continue
            
            influence = float(report.get("influence", 0.0))
            reliability = float(report.get("reliability", 0.0))
            severity = float(report.get("severity", 0.0))
            
            if influence > 0 and reliability > 0:
                metrics.append({
                    "agent": report.get("agent", "unknown"),
                    "influence": influence,
                    "reliability": reliability,
                    "severity": severity
                })
                
        if not metrics:
            return {
                "summary": "No valid agent reports",
                "influence": 0.0,
                "reliability": 0.0,
                "severity": 0.0,
                "details": {
                    "decision": "BLOCK",
                    "reason": "no_data"
                },
                "ok": False
            }
            
        # Calculate weighted severity
        total_weight = 0.0
        weighted_severity = 0.0
        
        for m in metrics:
            # Weight is influence × reliability
            weight = m["influence"] * m["reliability"]
            total_weight += weight
            weighted_severity += weight * m["severity"]
            
        if total_weight > 0:
            final_severity = min(1.0, max(0.0, weighted_severity / total_weight))
        else:
            final_severity = 0.0
            
        # Calculate final reliability as weighted average
        final_reliability = sum(m["reliability"] * m["influence"] for m in metrics) / \
                          sum(m["influence"] for m in metrics)
                          
        # Make decision based on severity thresholds
        decision = "PROCEED"
        if final_severity >= 0.8:
            decision = "BLOCK"
        elif final_severity >= 0.5:
            decision = "PROCEED_WITH_CAUTION"
            
        # Store decision in memory
        if hasattr(self.memory, "nexus"):
            self.memory.nexus.ingest(
                "decisions",
                "meta_judge",
                {
                    "severity": final_severity,
                    "reliability": final_reliability,
                    "decision": decision,
                    "metrics": metrics
                },
                ttl_sec=3600  # 1 hour retention
            )
        
        return {
            "summary": f"MetaJudge decision: {decision}",
            "influence": 1.0,  # Highest influence as final arbiter
            "reliability": final_reliability,
            "severity": final_severity,
            "details": {
                "decision": decision,
                "agent_metrics": metrics,
                "weighted_severity": final_severity,
                "final_reliability": final_reliability
            },
            "explain_edges": [
                {"from": "MetaJudge", "to": m["agent"], "weight": m["influence"]}
                for m in metrics
            ],
            "ok": True
        }

class AegisCouncil:
    """Coordinates multiple agents for guardrail decisions."""
    def __init__(self, per_agent_timeout_sec: float = MAX_AGENT_TIMEOUT_SEC):
        """Initialize council.
        
        Args:
            per_agent_timeout_sec: Timeout per agent in seconds
        """
        self.agents: List[AegisAgent] = []
        self.timeout = per_agent_timeout_sec
        self.executor = concurrent.futures.ThreadPoolExecutor()
        self._lock = threading.Lock()
        
    def register_agent(self, agent: AegisAgent) -> None:
        """Add an agent to the council."""
        with self._lock:
            self.agents.append(agent)
            
    def dispatch(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Dispatch input to all agents and collect results."""
        # TODO: Implement dispatch
        pass