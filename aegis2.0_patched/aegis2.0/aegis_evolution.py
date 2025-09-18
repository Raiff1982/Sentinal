from datetime import datetime, timezone, timedelta
import heapq, threading
import json
import logging
import math
import os
import random
import threading
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, TypedDict, Any
import heapq
try:
    import xxhash
except Exception:
    xxhash = None
try:
    import portalocker
except Exception:
    portalocker = None  # For file locking

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("AEGIS-Evolution")

# Nexus Memory
class NexusMemory:
    def __init__(self, max_entries: int = 10_000, default_ttl_secs: int = 7*24*3600):
        self.store: Dict[str, Dict[str, Any]] = {}
        self.expiration_heap: List[Tuple[float, str]] = []
        self.max_entries = max_entries
        self.default_ttl_secs = default_ttl_secs
        self._lock = threading.Lock()

    def _hash(self, key: str) -> str:
        if not isinstance(key, str):
            raise ValueError("Key must be a string")
        return xxhash.xxh64(key.encode()).hexdigest()

    def write(self, key: str, value: Any, ttl_secs: Optional[int] = None) -> None:
    now = datetime.now(timezone.utc)
        hashed = self._hash(key)
        ttl = ttl_secs if ttl_secs is not None else self.default_ttl_secs
        with self._lock:
            if len(self.store) >= self.max_entries:
                self._purge_oldest(now)
            self.store[hashed] = {"value": value, "timestamp": now, "ttl": int(ttl)}
            expiration_time = (now + timedelta(seconds=ttl)).timestamp()
            heapq.heappush(self.expiration_heap, (expiration_time, hashed))

    def _purge_oldest(self, now: datetime) -> None:
        if not self.store:
            return
        oldest_key = min(self.store.keys(), key=lambda k: self.store[k]["timestamp"])
        del self.store[oldest_key]
        self.expiration_heap = [(t, k) for t, k in self.expiration_heap if k != oldest_key]
        heapq.heapify(self.expiration_heap)

    def read(self, key: str) -> Any:
        hashed = self._hash(key)
        with self._lock:
            return self.store.get(hashed, {}).get("value")

    def purge_expired(self) -> int:
    now = datetime.now(timezone.utc)
        now_ts = now.timestamp()
        with self._lock:
            while self.expiration_heap and self.expiration_heap[0][0] <= now_ts:
                _, key = heapq.heappop(self.expiration_heap)
                if key in self.store and (now - self.store[key]["timestamp"]).total_seconds() > self.store[key]["ttl"]:
                    del self.store[key]
            self.expiration_heap = [(t, k) for t, k in self.expiration_heap if k in self.store]
            heapq.heapify(self.expiration_heap)
            return len(self.store)

# Base Report Structure
class AgentReport(TypedDict):
    agent: str
    ok: bool
    summary: str
    influence: float
    reliability: float
    severity: float
    details: Dict[str, Any]
    diagnostics: Dict[str, str]

def agent_report(agent: str, ok: bool, influence: float, reliability: float, severity: float, 
                details: Dict[str, Any], summary: str = "") -> AgentReport:
    return {
        "agent": agent,
        "ok": ok,
        "summary": summary,
        "influence": max(0.0, min(1.0, float(influence))),
        "reliability": max(0.0, min(1.0, float(reliability))),
        "severity": max(0.0, min(1.0, float(severity))),
        "details": details,
    "diagnostics": {"started": datetime.now(timezone.utc).isoformat(), "finished": datetime.now(timezone.utc).isoformat()}
    }

# Agents
class BiofeedbackAgent:
    def __init__(self, memory: NexusMemory):
        self.name = "BiofeedbackAgent"
        self.memory = memory

    def run(self, sample: Dict[str, Any]) -> AgentReport:
        try:
            bio = sample.get("bio", {})
            stress = max(0.0, min(1.0, float(bio.get("stress", 0.0))))
            self.memory.write(f"{self.name}:stress", {"stress": stress}, ttl_secs=1800)
            return agent_report(self.name, True, 0.25 + 0.5 * stress, 0.9, stress, 
                              {"stress": round(stress, 4)}, "bio fused")
        except (ValueError, TypeError) as e:
            log.exception("BiofeedbackAgent failed: %s", e)
            return agent_report(self.name, False, 0.0, 0.0, 0.0, {}, f"Error: {e}")

class EnvSignalAgent:
    def __init__(self, memory: NexusMemory):
        self.name = "EnvSignalAgent"
        self.memory = memory

    def run(self, sample: Dict[str, Any]) -> AgentReport:
        try:
            env = sample.get("env", {})
            risk = max(0.0, min(1.0, float(env.get("context_risk", 0.0))))
            self.memory.write(f"{self.name}:risk", {"context_risk": risk}, ttl_secs=1800)
            return agent_report(self.name, True, 0.2 + 0.6 * risk, 0.9, risk, 
                              {"context_risk": round(risk, 4)}, "env fused")
        except (ValueError, TypeError) as e:
            log.exception("EnvSignalAgent failed: %s", e)
            return agent_report(self.name, False, 0.0, 0.0, 0.0, {}, f"Error: {e}")

class ContextConflictAgent:
    def __init__(self, memory: NexusMemory):
        self.name = "ContextConflictAgent"
        self.memory = memory

    def run(self, sample: Dict[str, Any]) -> AgentReport:
        try:
            declared = str(sample.get("intent", "")).lower()[:1000]  # Limit length
            stress = max(0.0, min(1.0, float(sample.get("bio", {}).get("stress", 0.0))))
            risk = max(0.0, min(1.0, float(sample.get("env", {}).get("context_risk", 0.0))))
            want_speed = any(k in declared for k in ("proceed fast", "ship now", "push hard", "ignore risk"))
            want_pause = any(k in declared for k in ("pause", "hold", "audit", "review"))
            conflict = 0.0
            if want_speed:
                conflict = max(conflict, 0.6 * stress + 0.6 * risk)
            if want_pause:
                conflict = max(conflict, max(0.0, 0.4 - 0.5 * (1.0 - max(stress, risk))))
            conflict = max(0.0, min(1.0, conflict))
            self.memory.write(f"{self.name}:conflict", {"conflict": conflict}, ttl_secs=1800)
            return agent_report(self.name, True, 0.3 + 0.5 * conflict, 0.92, conflict, 
                              {"conflict": round(conflict, 4)}, "context vs intent")
        except (ValueError, TypeError) as e:
            log.exception("ContextConflictAgent failed: %s", e)
            return agent_report(self.name, False, 0.0, 0.0, 0.0, {}, f"Error: {e}")

class TimescaleCoordinator:
    def __init__(self, memory: NexusMemory):
        self.name = "TimescaleCoordinator"
        self.memory = memory

    def run(self, sample: Dict[str, Any]) -> AgentReport:
        try:
            timescale = max(0.0, min(1.0, float(sample.get("timescale", 0.0))))
            self.memory.write(f"{self.name}:t", {"timescale": timescale}, ttl_secs=1800)
            return agent_report(self.name, True, 0.3 + 0.6 * timescale, 0.92, timescale, 
                              {"timescale_signal": round(timescale, 4)}, "timescale fused")
        except (ValueError, TypeError) as e:
            log.exception("TimescaleCoordinator failed: %s", e)
            return agent_report(self.name, False, 0.0, 0.0, 0.0, {}, f"Error: {e}")

# Meta-Judge
@dataclass
class MetaGenes:
    risk_cap: float = 0.6
    stress_cap: float = 0.65
    min_integrity: float = 0.2
    timescale_cap: float = 0.55

    def clipped(self) -> "MetaGenes":
        return MetaGenes(
            risk_cap=float(min(0.95, max(0.2, self.risk_cap))),
            stress_cap=float(min(0.95, max(0.2, self.stress_cap))),
            min_integrity=float(min(0.9, max(0.05, self.min_integrity))),
            timescale_cap=float(min(0.95, max(0.1, self.timescale_cap)))
        )

class MetaJudgeEvolvable:
    def __init__(self, memory: NexusMemory, genes: Optional[MetaGenes] = None):
        self.name = "MetaJudge"
        self.memory = memory
        self.genes = (genes or MetaGenes()).clipped()

    def policy(self) -> Dict[str, float]:
        return asdict(self.genes)

    def decide(self, reports: List[AgentReport], avg_memory_integrity: float) -> AgentReport:
        try:
            g = self.genes
            valid_reports = [r for r in reports if r.get("ok") and all(k in r for k in ("severity", "reliability", "influence"))]
            
            # Calculate severity using weighted average of (influence × reliability)
            severity_weights = []
            total_weight = 0.0
            severity = 0.0
            
            for r in valid_reports:
                influence = float(r.get("influence", 0.0))
                reliability = float(r.get("reliability", 0.0))
                r_severity = float(r.get("severity", 0.0))
                weight = influence * reliability
                
                severity_weights.append((r_severity, weight))
                total_weight += weight
            
            if severity_weights:
                # Normalize weights and calculate weighted average
                severity = sum(s * (w / total_weight) for s, w in severity_weights)
                # Clamp to [0,1]
                severity = max(0.0, min(1.0, severity))
            
            # Initialize other metrics
            stress = 0.0
            risk = 0.0 
            conflict = 0.0
            timescale = 0.0
            for r in valid_reports:
                d = r.get("details", {})
                if r["agent"] == "BiofeedbackAgent":
                    stress = float(d.get("stress", 0.0))
                elif r["agent"] == "EnvSignalAgent":
                    risk = float(d.get("context_risk", 0.0))
                elif r["agent"] == "ContextConflictAgent":
                    conflict = float(d.get("conflict", 0.0))
                elif r["agent"] == "TimescaleCoordinator":
                    timescale = float(d.get("timescale_signal", 0.0))
            cautious = (
                severity > g.risk_cap or
                avg_memory_integrity < g.min_integrity or
                stress > g.stress_cap or
                conflict > 0.5 or
                timescale > g.timescale_cap
            )
            decision = "PROCEED_WITH_CAUTION" if cautious else "PROCEED"
            details = {
                "severity_total": round(severity, 4),
                "avg_memory_integrity": round(avg_memory_integrity, 4),
                "stress": round(stress, 4),
                "context_risk": round(risk, 4),
                "conflict": round(conflict, 4),
                "timescale_signal": round(timescale, 4),
                "policy": self.policy(),
                "decision": decision
            }
            self.memory.write(f"{self.name}:decision", details, ttl_secs=48*3600)
            return agent_report(self.name, True, 1.0, 0.95, severity, details, "arbitration")
        except (ValueError, TypeError, KeyError) as e:
            log.exception("MetaJudge failed: %s", e)
            return agent_report(self.name, False, 0.0, 0.0, 0.0, {}, f"Error: {e}")

# Council
class MicroCouncil:
    def __init__(self, genes: Optional[MetaGenes] = None):
        self.memory = NexusMemory()
        self.bio = BiofeedbackAgent(self.memory)
        self.env = EnvSignalAgent(self.memory)
        self.conflict = ContextConflictAgent(self.memory)
        self.timescale = TimescaleCoordinator(self.memory)
        self.meta = MetaJudgeEvolvable(self.memory, genes)

    def run_once(self, sample: Dict[str, Any], avg_memory_integrity: float = 1.0) -> Dict[str, Any]:
        self.memory.purge_expired()
        reports = [
            self.bio.run(sample),
            self.env.run(sample),
            self.conflict.run(sample),
            self.timescale.run(sample),
        ]
        reports.append(self.meta.decide(reports, max(0.0, min(1.0, avg_memory_integrity))))
        return {
            "reports": reports,
            "decision": reports[-1]["details"].get("decision", "PROCEED"),
            "policy": self.meta.policy()
        }

# Invariants & Guardian Hash
@dataclass(frozen=True)
class Invariants:
    name: str = "AEGIS MetaJudge Safety Invariants v1"
    cap_order: Tuple[str, str] = ("min_integrity<=stress_cap<=risk_cap<=0.95", "timescale_cap<=0.95")
    min_integrity_floor: float = 0.1
    risk_cap_ceiling: float = 0.9

def guardian_hash(inv: Invariants, genes: MetaGenes) -> str:
    blob = json.dumps({"invariants": asdict(inv), "genes": asdict(genes)}, sort_keys=True).encode()
    return xxhash.xxh64(blob).hexdigest()

def check_invariants(inv: Invariants, genes: MetaGenes) -> Tuple[bool, List[str]]:
    g = genes.clipped()
    errs = []
    if g.min_integrity < inv.min_integrity_floor:
        errs.append(f"min_integrity<{inv.min_integrity_floor}")
    if g.risk_cap > inv.risk_cap_ceiling:
        errs.append(f"risk_cap>{inv.risk_cap_ceiling}")
    if not (g.min_integrity <= g.stress_cap <= g.risk_cap <= 0.95):
        errs.append("cap_order violated: min_integrity<=stress_cap<=risk_cap<=0.95")
    if g.timescale_cap > 0.95:
        errs.append("timescale_cap>0.95")
    return len(errs) == 0, errs

# Evaluation Harness
@dataclass
class EvalWeights:
    safety: float = 0.65
    utility: float = 0.35
    penalty: float = 0.30

    def __post_init__(self):
        if not (0 <= self.safety <= 1 and 0 <= self.utility <= 1 and self.safety + self.utility <= 1):
            raise ValueError("Safety and utility weights must be in [0,1] and sum to <=1")
        if self.penalty < 0:
            raise ValueError("Penalty must be non-negative")

class EvaluationHarness:
    def __init__(self, dataset: List[Dict[str, Any]], weights: EvalWeights):
        self.dataset = self._validate_dataset(dataset)
        self.weights = weights
        self._conflict_agent = ContextConflictAgent(NexusMemory())

    def _validate_dataset(self, dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        validated = []
        for i, sample in enumerate(dataset):
            try:
                bio = sample.get("bio", {})
                env = sample.get("env", {})
                stress = max(0.0, min(1.0, float(bio.get("stress", 0.0))))
                risk = max(0.0, min(1.0, float(env.get("context_risk", 0.0))))
                timescale = max(0.0, min(1.0, float(sample.get("timescale", 0.0))))
                intent = str(sample.get("intent", ""))[:1000]
                label = sample.get("label", "PROCEED").upper()
                if label not in ("PROCEED", "CAUTION"):
                    raise ValueError(f"Invalid label: {label}")
                validated.append({
                    "bio": {"stress": stress},
                    "env": {"context_risk": risk},
                    "timescale": timescale,
                    "intent": intent,
                    "label": label,
                    "avg_integrity": max(0.0, min(1.0, float(sample.get("avg_integrity", 1.0))))
                })
            except (ValueError, TypeError) as e:
                log.warning("Skipping invalid dataset entry %d: %s", i, e)
        if not validated:
            raise ValueError("No valid dataset entries")
        return validated

    def score(self, genes: MetaGenes) -> Tuple[float, Dict[str, Any]]:
        inv = Invariants()
        ok, errs = check_invariants(inv, genes)
        penalty = self.weights.penalty * len(errs) if not ok else 0.0
        council = MicroCouncil(genes)
        safety_hits = 0
        safety_total = 0
        utility_hits = 0
        utility_total = 0
        unsafe_outcomes = 0
        for sample in self.dataset:
            out = council.run_once(sample, sample.get("avg_integrity", 1.0))
            decision = out["decision"]
            label = sample["label"]
            stress = sample["bio"]["stress"]
            risk = sample["env"]["context_risk"]
            conflict = self._conflict_agent.run(sample)["details"].get("conflict", 0.0)
            high_context = max(stress, risk, conflict, sample["timescale"]) > 0.6
            if high_context:
                safety_total += 1
                if decision == "PROCEED_WITH_CAUTION":
                    safety_hits += 1
                else:
                    unsafe_outcomes += 1
            else:
                utility_total += 1
                if decision == "PROCEED":
                    utility_hits += 1
        safety_score = safety_hits / max(1, safety_total)
        utility_score = utility_hits / max(1, utility_total)
        overall = (
            self.weights.safety * safety_score +
            self.weights.utility * utility_score -
            self.weights.penalty * unsafe_outcomes -
            penalty
        )
        metrics = {
            "safety_score": round(safety_score, 4),
            "utility_score": round(utility_score, 4),
            "unsafe_outcomes": unsafe_outcomes,
            "invariant_violations": errs,
            "guardian_hash": guardian_hash(inv, genes)
        }
        return overall, metrics

# Mutation Chamber
class MutationChamber:
    def __init__(self, harness: EvaluationHarness, seed: Optional[int] = None, sigma: float = 0.05):
        self.harness = harness
        self.sigma = sigma
        self.rng = random.Random(seed or 1337)
        self.inv = Invariants()

    def _mutate(self, genes: MetaGenes, round_num: int, total_rounds: int) -> MetaGenes:
        # Adaptive sigma: reduce as search progresses
        adaptive_sigma = self.sigma * (1 - round_num / total_rounds)
        g = MetaGenes(
            risk_cap=genes.risk_cap + self.rng.gauss(0, adaptive_sigma),
            stress_cap=genes.stress_cap + self.rng.gauss(0, adaptive_sigma),
            min_integrity=genes.min_integrity + self.rng.gauss(0, adaptive_sigma / 2),
            timescale_cap=genes.timescale_cap + self.rng.gauss(0, adaptive_sigma)
        ).clipped()
        return g

    def search(self, start: MetaGenes, rounds: int = 40, beam: int = 6) -> Tuple[MetaGenes, Dict[str, Any]]:
        frontier: List[Tuple[MetaGenes, float, Dict[str, Any]]] = []
        g0 = start.clipped()
        s0, m0 = self.harness.score(g0)
        frontier.append((g0, s0, m0))
        best_score = s0
        best_genes = g0
        best_metrics = m0
        for r in range(rounds):
            candidates: List[Tuple[MetaGenes, float, Dict[str, Any]]] = []
            seeds = sorted(frontier, key=lambda x: x[1], reverse=True)[:beam]
            for seed_genes, _, _ in seeds:
                for _ in range(beam):
                    g = self._mutate(seed_genes, r, rounds)
                    if not check_invariants(self.inv, g)[0]:
                        continue  # Skip invalid candidates
                    score, metrics = self.harness.score(g)
                    candidates.append((g, score, metrics))
                    if score > best_score:
                        best_score = score
                        best_genes = g
                        best_metrics = metrics
                        log.info("Round %d: New best score=%.4f, genes=%s", r, score, g)
            # Preserve best candidate (elitism)
            frontier = sorted(candidates + seeds, key=lambda x: x[1], reverse=True)[:beam]
            if not frontier:
                frontier = [(g0, s0, m0)]  # Fallback to initial
        return best_genes, best_metrics

# Policy Store
class PolicyStore:
    def __init__(self, path: str = "./aegis_policies.json"):
        self.path = path
        self._lock = threading.Lock()

    def save(self, genes: MetaGenes, metrics: Dict[str, Any]) -> None:
        record = {
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "genes": asdict(genes),
            "metrics": metrics,
            "guardian_hash": metrics.get("guardian_hash")
        }
        with self._lock:
            try:
                with portalocker.Lock(self.path, "w", timeout=5) as f:
                    json.dump(record, f, indent=2, sort_keys=True)
            except (OSError, portalocker.LockException) as e:
                log.error("Failed to save policy: %s", e)
                raise

    def load(self) -> Optional[MetaGenes]:
        with self._lock:
            if not os.path.exists(self.path):
                return None
            try:
                with portalocker.Lock(self.path, "r", timeout=5) as f:
                    data = json.load(f)
                g = data.get("genes", {})
                return MetaGenes(**g).clipped()
            except (OSError, portalocker.LockException, json.JSONDecodeError) as e:
                log.error("Failed to load policy: %s", e)
                return None

# Demo Dataset
def demo_dataset() -> List[Dict[str, Any]]:
    ds = [
        {"bio": {"stress": 0.15}, "env": {"context_risk": 0.1}, "timescale": 0.1, "intent": "", "label": "PROCEED", "avg_integrity": 0.9},
        {"bio": {"stress": 0.2}, "env": {"context_risk": 0.25}, "timescale": 0.2, "intent": "pause", "label": "PROCEED", "avg_integrity": 0.85},
        {"bio": {"stress": 0.1}, "env": {"context_risk": 0.15}, "timescale": 0.15, "intent": "proceed fast", "label": "PROCEED", "avg_integrity": 0.95},
        {"bio": {"stress": 0.45}, "env": {"context_risk": 0.35}, "timescale": 0.3, "intent": "", "label": "PROCEED", "avg_integrity": 0.7},
        {"bio": {"stress": 0.4}, "env": {"context_risk": 0.5}, "timescale": 0.45, "intent": "proceed fast", "label": "PROCEED", "avg_integrity": 0.65},
        {"bio": {"stress": 0.8}, "env": {"context_risk": 0.7}, "timescale": 0.6, "intent": "", "label": "CAUTION", "avg_integrity": 0.3},
        {"bio": {"stress": 0.7}, "env": {"context_risk": 0.65}, "timescale": 0.8, "intent": "proceed fast", "label": "CAUTION", "avg_integrity": 0.25},
        {"bio": {"stress": 0.55}, "env": {"context_risk": 0.8}, "timescale": 0.75, "intent": "proceed fast", "label": "CAUTION", "avg_integrity": 0.2},
        {"bio": {"stress": 0.9}, "env": {"context_risk": 0.3}, "timescale": 0.85, "intent": "proceed fast", "label": "CAUTION", "avg_integrity": 0.15},
        {"bio": {"stress": 0.35}, "env": {"context_risk": 0.9}, "timescale": 0.7, "intent": "", "label": "CAUTION", "avg_integrity": 0.4}
    ]
    # Add synthetic variations
    synthetic = []
    rng = random.Random(42)
    for sample in ds[:]:
        for _ in range(2):
            new_sample = {
                "bio": {"stress": max(0.0, min(1.0, sample["bio"]["stress"] + rng.gauss(0, 0.05)))},
                "env": {"context_risk": max(0.0, min(1.0, sample["env"]["context_risk"] + rng.gauss(0, 0.05)))},
                "timescale": max(0.0, min(1.0, sample["timescale"] + rng.gauss(0, 0.05))),
                "intent": sample["intent"],
                "label": sample["label"],
                "avg_integrity": max(0.0, min(1.0, sample["avg_integrity"] + rng.gauss(0, 0.05)))
            }
            synthetic.append(new_sample)
    return ds + synthetic

# CLI Runnable
if __name__ == "__main__":
    store = PolicyStore()
    baseline = store.load() or MetaGenes()
    weights = EvalWeights(safety=0.65, utility=0.35, penalty=0.30)
    harness = EvaluationHarness(demo_dataset(), weights)
    base_score, base_metrics = harness.score(baseline)
    log.info("Baseline genes=%s score=%.4f metrics=%s", baseline, base_score, base_metrics)
    chamber = MutationChamber(harness, seed=42, sigma=0.06)
    best_genes, best_metrics = chamber.search(baseline, rounds=35, beam=6)
    best_score, _ = harness.score(best_genes)
    log.info("Candidate genes=%s score=%.4f metrics=%s", best_genes, best_score, best_metrics)
    margin = 0.02
    inv_ok, inv_errs = check_invariants(Invariants(), best_genes)
    if inv_ok and best_score > base_score + margin:
        log.info("Adopting new genes (Δ=%.4f >= %.4f). Invariants: OK", best_score - base_score, margin)
        store.save(best_genes, best_metrics)
        adopted = best_genes
    else:
        if not inv_ok:
            log.warning("Rejected candidate due to invariant violations: %s", inv_errs)
        else:
            log.info("No adoption (improvement %.4f < %.4f). Keeping baseline.", best_score - base_score, margin)
        adopted = baseline
    council = MicroCouncil(adopted)
    sample = {"bio": {"stress": 0.82}, "env": {"context_risk": 0.68}, "timescale": 0.72, "intent": "proceed fast"}
    out = council.run_once(sample, avg_memory_integrity=0.18)
    print(json.dumps({"adopted_policy": council.meta.policy(), "decision_on_hot_case": out["decision"]}, indent=2))