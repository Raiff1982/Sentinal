from __future__ import annotations

import json
import logging
import math
import os
import sys
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple, TypedDict, Union
import statistics

import numpy as np

try:
    import portalocker  # type: ignore
    portalocker_available = True
except ImportError:
    portalocker = None  # type: ignore
    portalocker_available = False

log = logging.getLogger("AEGIS-Explain")

# -------------------------------------------------------------------------
# Snapshot schema
# -------------------------------------------------------------------------
class Edge(TypedDict):
    from_node: str
    to: str
    weight: float

class MetaData(TypedDict, total=False):
    decision: str
    severity_total: float
    stress: float
    context_risk: float
    conflict: float
    timescale_signal: float
    policy: Dict[str, float]

class ExplainSnapshot:
    """A snapshot of the AEGIS explain state at a point in time."""
    
    def __init__(
        self,
        nodes: List[str],
        edges: List[Edge],
        influence_index: Dict[str, float],
        meta: MetaData,
    ) -> None:
        """Create a new explain snapshot.
        
        Args:
            nodes: List of node names in the graph
            edges: List of edges connecting nodes
            influence_index: Map of agent->influence scores
            meta: Metadata about decisions and measurements
            
        Raises:
            ValueError: If any args fail validation
        """
        # Validate nodes are strings
        if not all(isinstance(n, str) for n in nodes):
            raise ValueError("nodes must be a list[str]")
            
        # Validate edge format 
        for e in edges:
            if not isinstance(e, dict) or not all(k in e for k in ("from_node", "to", "weight")):
                raise ValueError("each edge must have from_node, to, weight")
                
        # Validate influence_index types
        if not all(isinstance(k, str) and isinstance(v, (int, float)) for k, v in influence_index.items()):
            raise ValueError("influence_index must map str->number")
            
        # Validate meta is dict
        if not isinstance(meta, dict):
            raise ValueError("meta must be a dict")

        # Set values 
        self.timestamp = datetime.now(timezone.utc)
        self.nodes = nodes
        self.edges = edges
        self.influence_index = {k: max(0.0, min(1.0, float(v))) for k, v in influence_index.items()}
        self.meta = meta

    def to_json(self) -> str:
        """Convert snapshot to JSON string.
        
        Returns:
            JSON string representation of snapshot
        """
        def escape_value(v: Any) -> Any:
            if isinstance(v, str):
                return v.replace("\n", "\\n").replace("\r", "\\r")
            return v
            
        payload = {
            "ts": self.timestamp.isoformat(),
            "nodes": self.nodes,
            "edges": self.edges,
            "influence_index": {k: float(v) for k, v in self.influence_index.items()},
            "meta": {k: escape_value(v) for k, v in self.meta.items()},
        }
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ExplainSnapshot":
        """Create snapshot from dictionary.
        
        Args:
            d: Dictionary with snapshot data
            
        Returns:
            New ExplainSnapshot instance
            
        Raises:
            ValueError: If data fails validation
        """
        try:
            # Parse nodes
            nodes = [str(n) for n in d.get("nodes", [])]
            
            # Parse edges with validation
            edges = []
            for e in d.get("edges", []):
                try:
                    edges.append(Edge(
                        from_node=str(e["from_node"]),
                        to=str(e["to"]),
                        weight=max(0.0, min(1.0, float(e["weight"])))
                    ))
                except Exception:
                    continue
                    
            # Parse influence index
            infl = {str(k): max(0.0, min(1.0, float(v))) 
                   for k, v in (d.get("influence_index", {}) or {}).items()}
                   
            # Parse metadata
            meta = MetaData()
            meta.update(d.get("meta", {}) or {})
            
            snap = ExplainSnapshot(nodes, edges, infl, meta)
            
            # Set timestamp if provided
            ts = d.get("ts", "")
            if isinstance(ts, str) and ts:
                try:
                    snap.timestamp = datetime.fromisoformat(ts)
                except ValueError:
                    snap.timestamp = datetime.now(timezone.utc)
            
            return snap
            
        except Exception as e:
            raise ValueError(f"Invalid snapshot data: {e}")
            
    @property
    def ts(self) -> str:
        """Legacy accessor for timestamp."""
        return self.timestamp.isoformat()

# -------------------------------------------------------------------------
# Persistent explain store (daily JSONL files with pruning)
# -------------------------------------------------------------------------
class ExplainStore:
    """Persistent storage for explain snapshots in daily JSONL files."""
    
    def __init__(self, root: str = "./aegis_explain", prefix: str = "explain", max_age_days: int = 30) -> None:
        """Initialize the explain store.
        
        Args:
            root: Base directory for storage
            prefix: Prefix for JSONL files
            max_age_days: Maximum age in days before files are pruned
        """
        self.root = os.path.abspath(root)
        self.prefix = prefix
        self.max_age_days = int(max_age_days)
        os.makedirs(self.root, exist_ok=True)
        
    def _path_for_date(self, dt: datetime) -> str:
        """Get path for a specific date's JSONL file."""
        day = dt.strftime("%Y-%m-%d")
        return os.path.join(self.root, f"{self.prefix}-{day}.jsonl")
        
    def _prune_old_files(self) -> None:
        """Remove files older than max_age_days."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=self.max_age_days)
        for fname in os.listdir(self.root):
            if fname.startswith(self.prefix) and fname.endswith(".jsonl"):
                try:
                    date_str = fname[len(self.prefix)+1:-6]
                    file_date = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                    if file_date < cutoff:
                        os.remove(os.path.join(self.root, fname))
                except (ValueError, OSError) as e:
                    log.warning("Failed to prune file %s: %s", fname, e)

    def _load_between(self, start_dt: datetime, end_dt: datetime) -> List[ExplainSnapshot]:
        """Load all explanations between two datetimes."""
        if not start_dt.tzinfo or not end_dt.tzinfo:
            raise ValueError("Datetimes must be timezone-aware")

        results: List[ExplainSnapshot] = []
        current = start_dt

        while current <= end_dt:
            path = self._path_for_date(current)
            if os.path.exists(path):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        for line in f:
                            try:
                                data = json.loads(line)
                                snapshot = ExplainSnapshot.from_dict(data)
                                if start_dt <= snapshot.timestamp <= end_dt:
                                    results.append(snapshot)
                            except (json.JSONDecodeError, ValueError) as e:
                                log.warning("Error parsing line in %s: %s", path, e)
                except IOError as e:
                    log.warning("Error reading file %s: %s", path, e)
            current += timedelta(days=1)
        return results

    def _path_for_date(self, dt: datetime) -> str:
        """Get JSONL path for given date."""
        day = dt.strftime("%Y-%m-%d")
        return os.path.join(self.root, f"{self.prefix}-{day}.jsonl")

    def _prune_old_files(self) -> None:
        """Remove files older than max_age_days."""
        try:
            cutoff = datetime.now(timezone.utc) - timedelta(days=self.max_age_days)
            for fname in os.listdir(self.root):
                if not fname.startswith(self.prefix) or not fname.endswith(".jsonl"):
                    continue
                try:
                    date_str = fname[len(self.prefix)+1:-6]
                    file_date = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                    if file_date < cutoff:
                        os.remove(os.path.join(self.root, fname))
                except (ValueError, OSError) as e:
                    log.warning("Failed to prune file %s: %s", fname, e)
        except Exception as e:
            log.error("Failed to prune old files: %s", e)

    def store(self, snap: ExplainSnapshot) -> None:
        """Store a new explanation snapshot."""
        try:
            if not snap.timestamp.tzinfo:
                raise ValueError("ExplainSnapshot timestamp must be timezone-aware")

            self._prune_old_files()
            path = self._path_for_date(snap.timestamp)

            try:
                if portalocker_available:
                    with portalocker.Lock(path, mode="a", encoding="utf-8", timeout=10) as f:
                        f.write(snap.to_json() + "\n")
                else:
                    with open(path, "a", encoding="utf-8") as f:
                        f.write(snap.to_json() + "\n")
            except (IOError, Exception) as e:
                log.error("Failed to store explanation: %s", e)
                raise
        except Exception as e:
            log.error("Critical error storing explanation: %s", e)
            raise

    def window(self, hours: int = 24) -> List[ExplainSnapshot]:
        """Load explanations from the last N hours."""
        if hours < 0:
            raise ValueError("Hours must be non-negative")
            
        end = datetime.now(timezone.utc)
        start = end - timedelta(hours=hours)
        return self._load_between(start, end)

    def range(self, start: datetime, end: datetime) -> List[ExplainSnapshot]:
        """Load explanations between two datetimes."""
        if not start.tzinfo or not end.tzinfo:
            raise ValueError("Datetimes must be timezone-aware")
        if start > end:
            raise ValueError("Start time must be before end time")
        return self._load_between(start, end)

    def _load_between(self, start_dt: datetime, end_dt: datetime) -> List[ExplainSnapshot]:
        """Load all explanations between two datetimes."""
        if not start_dt.tzinfo or not end_dt.tzinfo:
            raise ValueError("Datetimes must be timezone-aware")

        results: List[ExplainSnapshot] = []
        try:
            current = start_dt
            while current <= end_dt:
                path = self._path_for_date(current)
                if os.path.exists(path):
                    try:
                        with open(path, "r", encoding="utf-8") as f:
                            for line in f:
                                try:
                                    data = json.loads(line)
                                    snapshot = ExplainSnapshot.from_dict(data)
                                    if start_dt <= snapshot.timestamp <= end_dt:
                                        results.append(snapshot)
                                except (json.JSONDecodeError, ValueError) as e:
                                    log.warning("Error parsing line in %s: %s", path, e)
                    except IOError as e:
                        log.warning("Error reading file %s: %s", path, e)
                current += timedelta(days=1)
        except Exception as e:
            log.error("Error loading explanations: %s", e)
        return results

    def store(self, snap: ExplainSnapshot) -> None:
        """Store a new explanation snapshot."""
        try:
            if not snap.timestamp or not snap.timestamp.tzinfo:
                raise ValueError("ExplainSnapshot timestamp must be timezone-aware")

            self._prune_old_files()
            path = self._path_for_date(snap.timestamp)

            try:
                if portalocker_available:
                    with portalocker.Lock(path, mode="a", encoding="utf-8", timeout=10) as f:
                        f.write(snap.to_json() + "\n")
                else:
                    with open(path, "a", encoding="utf-8") as f:
                        f.write(snap.to_json() + "\n")
            except (IOError, Exception) as e:
                log.error("Failed to store explanation: %s", e)
                raise
        except Exception as e:
            log.error("Critical error storing explanation: %s", e)
            raise

    def window(self, hours: int = 24) -> List[ExplainSnapshot]:
        """Load explanations from the last N hours."""
        if hours < 0:
            raise ValueError("Hours must be non-negative")
            
        end = datetime.now(timezone.utc)
        start = end - timedelta(hours=hours)
        return self._load_between(start, end)
        self._prune_old_files()
        path = self._path_for_date(datetime.now(timezone.utc))
        try:
            with portalocker.Lock(path, "a", timeout=5) as f:
                f.write(snap.to_json() + "\n")
        except (OSError, portalocker.LockException) as e:
            log.error("Failed to append snapshot to %s: %s", path, e)
            raise

    def _load_between(self, start: datetime, end: datetime) -> List[ExplainSnapshot]:
        self._prune_old_files()
        out: List[ExplainSnapshot] = []
        cur = start
        while cur.date() <= end.date():
            path = self._path_for_date(cur)
            if os.path.exists(path):
                try:
                    with portalocker.Lock(path, "r", timeout=5) as f:
                        for line_num, ln in enumerate(f, 1):
                            try:
                                rec = json.loads(ln.strip())
                                ts = datetime.fromisoformat(rec.get("ts", ""))
                                if start <= ts <= end:
                                    out.append(ExplainSnapshot.from_dict(rec))
                            except (json.JSONDecodeError, ValueError) as e:
                                log.warning("Skipping invalid JSON at %s:%d: %s", path, line_num, e)
                except (OSError, portalocker.LockException) as e:
                    log.error("Failed to read %s: %s", path, e)
            cur += timedelta(days=1)
        return sorted(out, key=lambda x: datetime.fromisoformat(x.ts))

    def window(self, hours: int) -> List[ExplainSnapshot]:
        if hours < 0:
            raise ValueError("Hours must be non-negative")
        end = datetime.now(timezone.utc)
        start = end - timedelta(hours=hours)
        return self._load_between(start, end)

    def range(self, start: datetime, end: datetime) -> List[ExplainSnapshot]:
        if start > end:
            raise ValueError("Start time must be before end time")
        return self._load_between(start, end)

# Math Helpers
def mean(xs: List[float]) -> float:
    return sum(xs) / max(1, len(xs))

def safe_var(xs: List[float]) -> float:
    if len(xs) <= 1:
        return 1e-6  # Small non-zero variance for single-element lists
    return statistics.pvariance(xs)

def welch_t(a: List[float], b: List[float]) -> Tuple[float, float]:
    if not a or not b or len(a) < 2 or len(b) < 2:
        return 0.0, 0.0
    ma, mb = mean(a), mean(b)
    va, vb = safe_var(a), safe_var(b)
    na, nb = len(a), len(b)
    denom = math.sqrt(max(va/na + vb/nb, 1e-6))
    t = (ma - mb) / denom
    num = (va/na + vb/nb)**2
    den = (va**2/(na**2*(na-1)) if na > 1 else 0.0) + (vb**2/(nb**2*(nb-1)) if nb > 1 else 0.0)
    dof = num / max(den, 1e-6) if den > 0 else 1.0
    return t, max(1.0, dof)

# Drift Analyzer
class ExplainDrift:
    def __init__(self, snaps_a: List[ExplainSnapshot], snaps_b: List[ExplainSnapshot]):
        self.A = snaps_a
        self.B = snaps_b
        self._edge_cache: Dict[Tuple[str, str], Tuple[List[float], List[float]]] = {}
        self._infl_cache: Dict[str, Tuple[List[float], List[float]]] = {}

    def _series_edges(self, snaps: List[ExplainSnapshot], edge_key: Tuple[str, str]) -> List[float]:
        fr, to = edge_key
        if edge_key not in self._edge_cache:
            series_a = []
            series_b = []
            for s in self.A:
                w = 0.0
                for e in s.edges:
                    if e.get("from") == fr and e.get("to") == to:
                        w = max(0.0, min(1.0, float(e.get("weight", 0.0))))
                        break
                series_a.append(w)
            for s in self.B:
                w = 0.0
                for e in s.edges:
                    if e.get("from") == fr and e.get("to") == to:
                        w = max(0.0, min(1.0, float(e.get("weight", 0.0))))
                        break
                series_b.append(w)
            self._edge_cache[edge_key] = (series_a, series_b)
        return self._edge_cache[edge_key][0 if snaps is self.A else 1]

    def _series_infl(self, snaps: List[ExplainSnapshot], agent: str) -> List[float]:
        if agent not in self._infl_cache:
            series_a = [max(0.0, min(1.0, float(s.influence_index.get(agent, 0.0)))) for s in self.A]
            series_b = [max(0.0, min(1.0, float(s.influence_index.get(agent, 0.0)))) for s in self.B]
            self._infl_cache[agent] = (series_a, series_b)
        return self._infl_cache[agent][0 if snaps is self.A else 1]

    def edge_delta(self, fr: str, to: str) -> Dict[str, Any]:
        key = (fr, to)
        a = self._series_edges(self.A, key)
        b = self._series_edges(self.B, key)
        ma, mb = mean(a), mean(b)
        t, dof = welch_t(b, a)
        return {
            "edge": {"from": fr, "to": to},
            "avg_before": round(ma, 6),
            "avg_after": round(mb, 6),
            "delta": round(mb - ma, 6),
            "welch_t": round(t, 4),
            "dof": round(dof, 2),
            "samples_before": len(a),
            "samples_after": len(b)
        }

    def influence_delta(self, agent: str) -> Dict[str, Any]:
        a = self._series_infl(self.A, agent)
        b = self._series_infl(self.B, agent)
        ma, mb = mean(a), mean(b)
        t, dof = welch_t(b, a)
        return {
            "agent": agent,
            "avg_before": round(ma, 6),
            "avg_after": round(mb, 6),
            "delta": round(mb - ma, 6),
            "welch_t": round(t, 4),
            "dof": round(dof, 2),
            "samples_before": len(a),
            "samples_after": len(b)
        }

# -------------------------------------------------------------------------
# Persistence layer for explanation snapshots
# -------------------------------------------------------------------------
def mean(xs: List[float]) -> float:
    """Calculate arithmetic mean of a list of numbers."""
    if not xs:
        return 0.0
    return sum(xs) / len(xs)

def safe_var(xs: List[float]) -> float:
    """Calculate variance, returning small non-zero value for len <= 1."""
    if len(xs) <= 1:
        return 1e-6  # Small non-zero variance
    return statistics.pvariance(xs)

def welch_t(a: List[float], b: List[float]) -> Tuple[float, float]:
    """Calculate Welch's t-test statistic and degrees of freedom."""
    if not a or not b or len(a) < 2 or len(b) < 2:
        return 0.0, 0.0
        
    # Calculate means and variances
    ma, mb = mean(a), mean(b)
    va, vb = safe_var(a), safe_var(b)
    na, nb = len(a), len(b)
    
    # Calculate t-statistic
    denom = math.sqrt(max(va/na + vb/nb, 1e-6))
    t = (ma - mb) / denom
    
    # Calculate degrees of freedom
    num = (va/na + vb/nb)**2
    den = (va**2/(na**2*(na-1)) if na > 1 else 0.0) + (vb**2/(nb**2*(nb-1)) if nb > 1 else 0.0)
    dof = num / max(den, 1e-6) if den > 0 else 1.0
    
    return t, max(1.0, dof)

# -------------------------------------------------------------------------
# Why Engine for analyzing temporal drifts in explanations
# -------------------------------------------------------------------------
class WhyEngine:
    """Why Engine for analyzing temporal changes in AEGIS explanations."""
    
    def __init__(self, store: ExplainStore) -> None:
        """Initialize the Why Engine.
        
        Args:
            store: Explain store for accessing historical snapshots
        """
        self.store = store

    def _split(self, start_hours: int, end_hours: int) -> Tuple[List[ExplainSnapshot], List[ExplainSnapshot]]:
        """Split and retrieve snapshots for before/after comparison.
        
        Args:
            start_hours: Hours to look back for 'before' window
            end_hours: Hours to look back for 'after' window
        Returns:
            Tuple of (before_snaps, after_snaps)
        Raises:
            ValueError: if hours are negative
        """
        if start_hours < 0 or end_hours < 0:
            raise ValueError("Hours must be non-negative")
        
        now = datetime.now(timezone.utc)
        start_window = now - timedelta(hours=start_hours)
        end_window = now - timedelta(hours=end_hours)
        
        A = self.store.range(start_window, now)
        B = self.store.range(end_window, now)
        return A, B

    def why_edge_change(self, fr: str, to: str, before_hours: int = 24, after_hours: int = 24) -> Dict[str, Any]:
        """Analyze changes in a specific edge's weight over time.
        
        Args:
            fr: Source node ID
            to: Target node ID 
            before_hours: Hours to look back for baseline
            after_hours: Hours to look back for comparison
        Returns:
            Analysis of the edge weight changes
        """
        A, B = self._split(before_hours, after_hours)
        drift = ExplainDrift(A, B)
        return drift.edge_delta(fr, to)

    def why_influence_change(self, agent: str, before_hours: int = 24, after_hours: int = 24) -> Dict[str, Any]:
        """Analyze changes in an agent's influence index over time.
        
        Args:
            agent: Name of the agent
            before_hours: Hours to look back for baseline
            after_hours: Hours to look back for comparison
        Returns:
            Analysis of the influence changes
        """
        A, B = self._split(before_hours, after_hours)
        drift = ExplainDrift(A, B)
        return drift.influence_delta(agent)

    def top_shifts(self, before_hours: int = 24, after_hours: int = 24, k: int = 5) -> Dict[str, List[Dict[str, Any]]]:
        """Find the most significant changes in edges and influence values.
        
        Args:
            before_hours: Hours to look back for baseline
            after_hours: Hours to look back for comparison
            k: Number of top changes to return
        Returns:
            Dict with top edge and agent influence changes
        """
        if k < 1:
            raise ValueError("k must be positive")
            
        A, B = self._split(before_hours, after_hours)
        if not A or not B:
            return {"edges": [], "agents": []}
            
        edge_keys = set()
        agent_keys = set()
        for s in A + B:
            for e in s.edges:
                edge_keys.add((e["from_node"], e["to"]))
            agent_keys.update(s.influence_index.keys())
            
        drift = ExplainDrift(A, B)
        edge_deltas = [drift.edge_delta(fr, to) for fr, to in edge_keys]
        agent_deltas = [drift.influence_delta(a) for a in agent_keys]
        
        # Prioritize statistical significance, then delta
        edge_rank = sorted(edge_deltas, key=lambda d: (abs(d["welch_t"]), abs(d["delta"]), d["dof"]), reverse=True)[:k]
        agent_rank = sorted(agent_deltas, key=lambda d: (abs(d["welch_t"]), abs(d["delta"]), d["dof"]), reverse=True)[:k]
        return {"edges": edge_rank, "agents": agent_rank}

# -------------------------------------------------------------------------
# Integration Helpers 
# -------------------------------------------------------------------------
def build_snapshot(council_bundle: Dict[str, Any]) -> ExplainSnapshot:
    """Build an explain snapshot from a council output bundle.
    
    Args:
        council_bundle: Dict containing council output
    Returns:
        ExplainSnapshot constructed from the bundle
    Raises:
        ValueError: If bundle is malformed or missing required data
    """
    try:
        nodes = council_bundle.get("explainability_graph", {}).get("nodes", [])
        edges = council_bundle.get("explainability_graph", {}).get("edges", [])
        inf: Dict[str, float] = {}
        meta: MetaData = {
            "decision": "PROCEED",
            "severity_total": 0.0,
            "stress": 0.0,
            "context_risk": 0.0,
            "conflict": 0.0,
            "timescale_signal": 0.0,
            "policy": {}
        }
        
        for r in council_bundle.get("reports", []):
            agent = r.get("agent", "")
            influence = max(0.0, min(1.0, float(r.get("influence", 0.0))))
            reliability = max(0.0, min(1.0, float(r.get("reliability", 0.0))))
            inf[agent] = influence * reliability
            
            if agent == "MetaJudge":
                d = r.get("details", {})
                meta["decision"] = str(d.get("decision", "PROCEED"))
                meta["severity_total"] = max(0.0, min(1.0, float(d.get("severity_total", 0.0))))
                meta["stress"] = max(0.0, min(1.0, float(d.get("stress", 0.0))))
                meta["context_risk"] = max(0.0, min(1.0, float(d.get("context_risk", 0.0))))
                meta["conflict"] = max(0.0, min(1.0, float(d.get("conflict", 0.0))))
                meta["timescale_signal"] = max(0.0, min(1.0, float(d.get("timescale_signal", 0.0))))
                meta["policy"] = d.get("policy", {})
                
        return ExplainSnapshot(nodes, edges, inf, meta)
    except (KeyError, ValueError, TypeError) as e:
        log.error("Failed to build snapshot: %s", e)
        raise ValueError(f"Invalid council bundle: {e}")

def record_from_council_bundle(bundle: Dict[str, Any], store: ExplainStore) -> None:
    """Record a council output bundle to persistent storage.
    
    Args:
        bundle: The council output bundle to record
        store: ExplainStore to save the snapshot to
    Raises:
        ValueError: If bundle is invalid
        IOError: If snapshot cannot be stored
    """
    snap = build_snapshot(bundle)
    store.store(snap)

def example_usage_with_council_output(bundle_now: Dict[str, Any], store: ExplainStore) -> Dict[str, Any]:
    """Example usage of the explain store with council output.
    
    Args:
        bundle_now: Current council output bundle
        store: Explain store for persistence
    Returns:
        Dict containing drift analysis results
    """
    record_from_council_bundle(bundle_now, store)
    why = WhyEngine(store)
    return {
        "edge_shifts": why.top_shifts(before_hours=6, after_hours=6, k=5),
        "agent_changes": why.why_influence_change("ShortTermAgent", before_hours=6, after_hours=6)
    }

# -------------------------------------------------------------------------
# Demo & Example
# -------------------------------------------------------------------------
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    )

    from aegis_timescales import (
        AegisCouncil, EchoSeedAgent, ShortTermAgent, 
        MidTermAgent, LongTermArchivistAgent,
        TimeScaleCoordinator, MetaJudgeAgent
    )

    # Set up store and council
    store = ExplainStore(root="./aegis_explain", prefix="explain")
    council = AegisCouncil(per_agent_timeout_sec=2.5)
    council.register_agent(EchoSeedAgent("EchoSeedAgent", council.memory))
    council.register_agent(ShortTermAgent("ShortTermAgent", council.memory))
    council.register_agent(MidTermAgent("MidTermAgent", council.memory))
    council.register_agent(LongTermArchivistAgent("LongTermArchivist", council.memory)) 
    council.register_agent(TimeScaleCoordinator("TimeScaleCoordinator", council.memory))
    council.register_agent(MetaJudgeAgent("MetaJudge", council.memory))
    sample_inputs = [
        {"text": "Proceed with care. Audit logs, then continue.", "_now_stress": 0.25, "_now_risk": 0.2},
        {"text": "Incident trending up. Protect users. Consider pause.", "_now_stress": 0.45, "_now_risk": 0.35},
        {"text": "Signals mixed; ship small fix under feature flag.", "_now_stress": 0.35, "_now_risk": 0.3},
        {"text": "Escalation observed; slow rollout and increase monitoring.", "_now_stress": 0.6, "_now_risk": 0.55},
    ]
    last_decision = "PROCEED"
    for i, inp in enumerate(sample_inputs, 1):
        inp["_last_decision"] = last_decision
        bundle = council.dispatch(inp)
        record_from_council_bundle(bundle, store)
        mj = next((r for r in bundle["reports"] if r.get("agent") == "MetaJudge"), {})
        last_decision = mj.get("details", {}).get("decision", last_decision)

    # Query WhyEngine
    why = WhyEngine(store)
    result = why.top_shifts(before_hours=24, after_hours=24, k=5)
    print(json.dumps(result, indent=2))