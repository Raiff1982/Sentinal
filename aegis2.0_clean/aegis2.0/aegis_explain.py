import os
import json
import math
import statistics
try:
    import portalocker
except Exception:
    portalocker = None
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, TypedDict, Any
import logging

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("AEGIS-Explain")

# Snapshot Schema
class Edge(TypedDict):
    from_node: str
    to: str
    weight: float

class MetaData(TypedDict):
    decision: str
    severity_total: float
    stress: float
    context_risk: float
    conflict: float
    timescale_signal: float
    policy: Optional[Dict[str, float]]  # From aegis_evolution.py

class ExplainSnapshot:
    def __init__(self, nodes: List[str], edges: List[Edge], influence_index: Dict[str, float], meta: MetaData):
        if not all(isinstance(n, str) for n in nodes):
            raise ValueError("Nodes must be strings")
        if not all(isinstance(e, dict) and all(k in e for k in ("from", "to", "weight")) for e in edges):
            raise ValueError("Edges must be valid dictionaries with from, to, weight")
        if not all(isinstance(k, str) and isinstance(v, (int, float)) for k, v in influence_index.items()):
            raise ValueError("Influence index must map strings to numbers")
        if not isinstance(meta, dict):
            raise ValueError("Meta must be a dictionary")
        self.ts = datetime.utcnow().isoformat()
        self.nodes = nodes
        self.edges = edges
        self.influence_index = {k: max(0.0, min(1.0, float(v))) for k, v in influence_index.items()}
        self.meta = meta

    def to_json(self) -> str:
        def escape_value(v: Any) -> Any:
            if isinstance(v, str):
                return v.replace("\n", "\\n").replace("\r", "\\r")
            return v
        serialized = {
            "ts": self.ts,
            "nodes": self.nodes,
            "edges": [{k: escape_value(v) for k, v in e.items()} for e in self.edges],
            "influence_index": self.influence_index,
            "meta": {k: escape_value(v) for k, v in self.meta.items()}
        }
        return json.dumps(serialized, separators=(",", ":"))

    @staticmethod
    def from_dict(d: Dict) -> "ExplainSnapshot":
        try:
            nodes = d.get("nodes", [])
            edges = d.get("edges", [])
            influence_index = d.get("influence_index", {})
            meta = d.get("meta", {})
            snap = ExplainSnapshot(nodes, edges, influence_index, meta)
            snap.ts = d.get("ts", datetime.utcnow().isoformat())
            return snap
        except (KeyError, ValueError) as e:
            log.error("Invalid snapshot dictionary: %s", e)
            raise ValueError(f"Failed to parse snapshot: {e}")

# Persistent Store
class ExplainStore:
    def __init__(self, root: str = "./aegis_explain", prefix: str = "explain", max_age_days: int = 30):
        if not os.path.abspath(root).startswith(os.getcwd()):
            raise ValueError("Root path must be within current working directory")
        self.root = os.path.abspath(root)
        self.prefix = prefix
        self.max_age_days = max_age_days
        os.makedirs(self.root, exist_ok=True)

    def _path_for_date(self, dt: datetime) -> str:
        day = dt.strftime("%Y-%m-%d")
        return os.path.join(self.root, f"{self.prefix}-{day}.jsonl")

    def _prune_old_files(self) -> None:
        cutoff = datetime.utcnow() - timedelta(days=self.max_age_days)
        for fname in os.listdir(self.root):
            if fname.startswith(self.prefix) and fname.endswith(".jsonl"):
                try:
                    date_str = fname[len(self.prefix)+1:-6]
                    file_date = datetime.strptime(date_str, "%Y-%m-%d")
                    if file_date < cutoff:
                        os.remove(os.path.join(self.root, fname))
                        log.info("Pruned old file: %s", fname)
                except (ValueError, OSError) as e:
                    log.warning("Failed to prune file %s: %s", fname, e)

    def append(self, snap: ExplainSnapshot) -> None:
        self._prune_old_files()
        path = self._path_for_date(datetime.utcnow())
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
        end = datetime.utcnow()
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

# Why Engine
class WhyEngine:
    def __init__(self, store: ExplainStore):
        self.store = store

    def _split(self, start_hours: int, end_hours: int) -> Tuple[List[ExplainSnapshot], List[ExplainSnapshot]]:
        if start_hours < 0 or end_hours < 0:
            raise ValueError("Hours must be non-negative")
        now = datetime.utcnow()
        A = self.store.window(start_hours)
        B = self.store.window(end_hours)
        return A, B

    def why_edge_change(self, fr: str, to: str, before_hours: int = 24, after_hours: int = 24) -> Dict[str, Any]:
        A, B = self._split(before_hours, after_hours)
        drift = ExplainDrift(A, B)
        return drift.edge_delta(fr, to)

    def why_influence_change(self, agent: str, before_hours: int = 24, after_hours: int = 24) -> Dict[str, Any]:
        A, B = self._split(before_hours, after_hours)
        drift = ExplainDrift(A, B)
        return drift.influence_delta(agent)

    def top_shifts(self, before_hours: int = 24, after_hours: int = 24, k: int = 5) -> Dict[str, List[Dict]]:
        if k < 1:
            raise ValueError("k must be positive")
        A, B = self._split(before_hours, after_hours)
        if not A or not B:
            return {"edges": [], "agents": []}
        edge_keys = set()
        agent_keys = set()
        for s in A + B:
            for e in s.edges:
                edge_keys.add((e["from"], e["to"]))
            agent_keys.update(s.influence_index.keys())
        drift = ExplainDrift(A, B)
        edge_deltas = [drift.edge_delta(fr, to) for fr, to in edge_keys]
        agent_deltas = [drift.influence_delta(a) for a in agent_keys]
        # Prioritize statistical significance, then delta
        edge_rank = sorted(edge_deltas, key=lambda d: (abs(d["welch_t"]), abs(d["delta"]), d["dof"]), reverse=True)[:k]
        agent_rank = sorted(agent_deltas, key=lambda d: (abs(d["welch_t"]), abs(d["delta"]), d["dof"]), reverse=True)[:k]
        return {"edges": edge_rank, "agents": agent_rank}

# Integration Helpers
def build_snapshot(council_bundle: Dict) -> ExplainSnapshot:
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
            "policy": None
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
                meta["policy"] = d.get("policy", None)
        return ExplainSnapshot(nodes, edges, inf, meta)
    except (KeyError, ValueError, TypeError) as e:
        log.error("Failed to build snapshot: %s", e)
        raise ValueError(f"Invalid council bundle: {e}")

def record_from_council_bundle(bundle: Dict, store: ExplainStore) -> None:
    snap = build_snapshot(bundle)
    store.append(snap)

def example_usage_with_council_output(bundle_now: Dict, store: ExplainStore) -> Dict[str, Any]:
    record_from_council_bundle(bundle_now, store)
    why = WhyEngine(store)
    return {
        "why_edge_Meta_inputs": why.top_shifts(before_hours=6, after_hours=6, k=5),
        "why_agent_influence": why.why_influence_change("ShortTermAgent", before_hours=6, after_hours=6)
    }

# CLI Demo
if __name__ == "__main__":
    from aegis_timescales import AegisCouncil, EchoSeedAgent, ShortTermAgent, MidTermAgent, LongTermArchivistAgent, TimeScaleCoordinator, MetaJudgeAgent

    store = ExplainStore(root="./aegis_explain", prefix="explain")
    council = AegisCouncil(per_agent_timeout_sec=2.5)
    council.register_agent(EchoSeedAgent("EchoSeedAgent", council.memory))
    council.register_agent(ShortTermAgent("ShortTermAgent", council.memory))
    council.register_agent(MidTermAgent("MidTermAgent", council.memory))
    council.register_agent(LongTermArchivistAgent("LongTermArchivist", council.memory))
    council.register_agent(TimeScaleCoordinator("TimeScaleCoordinator", council.memory))
    council.register_agent(MetaJudgeAgent("MetaJudge", council.memory))

    # Simulate council runs to populate snapshots
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