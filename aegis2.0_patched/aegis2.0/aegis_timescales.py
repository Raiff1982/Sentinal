import json
import math
import logging
import threading
import unicodedata
import re
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Deque, Dict, List, Tuple, Any, Optional, TypedDict
import heapq
try:
    import xxhash
except Exception:
    xxhash = None
import os

# Hash helper
def fast_hash(data: bytes) -> str:
    if xxhash is not None:
        return xxhash.xxh64(data).hexdigest()
    import hashlib
    return hashlib.sha256(data).hexdigest()


# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("AEGIS-Timescales")

# Sanitizer
class InputSanitizer:
    CONTROL_CHARS = {chr(i) for i in range(0, 32)} | {chr(127)}
    DANGEROUS_TOKENS = {
        r"\bexec\(", r"\beval\(", r"\bos\.system", r"\bsubprocess\.",
        r"<script\b", r"\.\./", r"\.\.\\", r"\033"
    }
    MAX_INPUT_LENGTH = 10_000

    @staticmethod
    def normalize(text: str) -> str:
        if len(text) > InputSanitizer.MAX_INPUT_LENGTH:
            raise ValueError(f"Input exceeds max length of {InputSanitizer.MAX_INPUT_LENGTH}")
        return unicodedata.normalize("NFC", text)

    @staticmethod
    def escape_html(text: str) -> str:
        return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    @staticmethod
    def audit_text(text: str) -> Dict[str, Any]:
        if not isinstance(text, str):
            return {"normalized": "", "issues": ["invalid_type"], "safe": False}
        issues = []
        if len(text) > InputSanitizer.MAX_INPUT_LENGTH:
            issues.append("input_too_long")
        for ch in InputSanitizer.CONTROL_CHARS:
            if ch in text:
                issues.append("control_char")
                break
        for tok in InputSanitizer.DANGEROUS_TOKENS:
            if re.search(tok, text, re.IGNORECASE):
                issues.append(f"danger_token:{tok}")
        if "\n" in text or "\r" in text:
            issues.append("newline_present")
        normalized = InputSanitizer.normalize(text)
        return {
            "normalized": normalized,
            "issues": sorted(set(issues)),
            "safe": len(issues) == 0
        }

# Nexus Memory
class NexusMemory:
    def __init__(self, max_entries: int = 20_000, default_ttl_secs: int = 14*24*3600):
        self.store: Dict[str, Dict[str, Any]] = {}
        self.expiration_heap: List[Tuple[float, str]] = []
        self.max_entries = max_entries
        self.default_ttl_secs = default_ttl_secs
        self._lock = threading.Lock()

    def _hash(self, key: str) -> str:
        return fast_hash(key.encode())

    def write(self, key: str, value: Any, weight: float = 1.0, entropy: float = 0.1, ttl_secs: Optional[int] = None) -> str:
        now = datetime.utcnow()
        hashed = self._hash(key)
        ttl = ttl_secs if ttl_secs is not None else self.default_ttl_secs
        with self._lock:
            if len(self.store) >= self.max_entries:
                self._purge_lowest_integrity(now)
            self.store[hashed] = {
                "value": value,
                "timestamp": now,
                "weight": max(0.0, float(weight)),
                "entropy": max(0.0, float(entropy)),
                "ttl": int(ttl)
            }
            expiration_time = (now + timedelta(seconds=ttl)).timestamp()
            heapq.heappush(self.expiration_heap, (expiration_time, hashed))
        return hashed

    def _purge_lowest_integrity(self, now: datetime) -> None:
        if not self.store:
            return
        oldest_key = min(self.store.keys(), key=lambda k: self._integrity(self.store[k], now))
        del self.store[oldest_key]
        self.expiration_heap = [(t, k) for t, k in self.expiration_heap if k != oldest_key]
        heapq.heapify(self.expiration_heap)

    def read(self, key: str) -> Any:
        hashed = self._hash(key)
        return self.store.get(hashed, {}).get("value")

    def _integrity(self, rec: Dict[str, Any], now: Optional[datetime] = None) -> float:
        if not rec or "timestamp" not in rec:
            return 0.0
        now = now or datetime.utcnow()
        age_sec = (now - rec.get("timestamp", now)).total_seconds()
        ttl = max(1, rec.get("ttl", self.default_ttl_secs))
        age_factor = max(0.0, 1.0 - (age_sec / ttl))
        weight = rec.get("weight", 1.0)
        entropy = rec.get("entropy", 0.1)
        return max(0.0, (weight * age_factor) / (1.0 + entropy))

    def purge_expired(self) -> int:
        now = datetime.utcnow()
        now_ts = now.timestamp()
        with self._lock:
            while self.expiration_heap and self.expiration_heap[0][0] <= now_ts:
                _, key = heapq.heappop(self.expiration_heap)
                if key in self.store and (now - self.store[key]["timestamp"]).total_seconds() > self.store[key]["ttl"]:
                    del self.store[key]
            self.expiration_heap = [(t, k) for t, k in self.expiration_heap if k in self.store]
            heapq.heapify(self.expiration_heap)
            return len(self.store)

    def audit(self) -> Dict[str, Dict[str, Any]]:
        now = datetime.utcnow()
        with self._lock:
            return {
                k: {
                    "timestamp": v.get("timestamp").isoformat(),
                    "weight": v.get("weight"),
                    "entropy": v.get("entropy"),
                    "ttl": v.get("ttl"),
                    "integrity": round(self._integrity(v, now), 6)
                } for k, v in self.store.items()
            }

# RingStats
class RingStats:
    def __init__(self, maxlen: int):
        if maxlen < 1:
            raise ValueError("maxlen must be positive")
        self.values: Deque[Tuple[float, float]] = deque(maxlen=maxlen)
        self._ema: Optional[float] = None
        self._alpha: Optional[float] = None

    def push(self, value: float, t: Optional[datetime] = None) -> None:
        if not isinstance(value, (int, float)):
            raise ValueError("Value must be numeric")
        t = t or datetime.utcnow()
        self.values.append((t.timestamp(), float(value)))
        # Invalidate EMA cache
        self._ema = None

    def ema(self, alpha: float = 0.3) -> float:
        if not 0 < alpha < 1:
            raise ValueError("Alpha must be between 0 and 1")
        if not self.values:
            return 0.0
        if self._ema is not None and self._alpha == alpha:
            return self._ema
        ema_val = self.values[0][1]
        for _, v in list(self.values)[1:]:
            ema_val = alpha * v + (1 - alpha) * ema_val
        self._ema = float(ema_val)
        self._alpha = alpha
        return self._ema

    def slope(self) -> float:
        n = len(self.values)
        if n < 2:
            return 0.0
        xs = [x for x, _ in self.values]
        ys = [y for _, y in self.values]
        x_mean = sum(xs) / n
        y_mean = sum(ys) / n
        num = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
        den = sum((x - x_mean) ** 2 for x in xs)
        return float(num / max(den, 1e-6)) if den > 0 else 0.0

    def minmax_norm(self) -> float:
        if not self.values:
            return 0.0
        vals = [v for _, v in self.values]
        vmin, vmax = min(vals), max(vals)
        if abs(vmax - vmin) < 1e-6:
            return 0.0
        return float((vals[-1] - vmin) / (vmax - vmin))

# Base Agent
class AgentReport(TypedDict):
    agent: str
    ok: bool
    diagnostics: Dict[str, str]
    summary: str
    influence: float
    reliability: float
    severity: float
    details: Dict[str, Any]

class AegisAgent(ABC):
    def __init__(self, name: str, memory: NexusMemory):
        self.name = name
        self.memory = memory

    @abstractmethod
    def analyze(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        ...

    def _safe_run(self, input_data: Dict[str, Any]) -> AgentReport:
        started = datetime.utcnow().isoformat()
        try:
            result = self.analyze(input_data)
            ok = True
            err = None
        except (ValueError, TypeError, KeyError) as e:
            log.exception("Agent %s failed: %s", self.name, e)
            ok = False
            result = {}
            err = f"{type(e).__name__}: {e}"
        finished = datetime.utcnow().isoformat()
        return {
            "agent": self.name,
            "ok": ok,
            "diagnostics": {"started": started, "finished": finished, "error": err},
            "summary": result.get("summary", ""),
            "influence": float(result.get("influence", 0.0)),
            "reliability": float(result.get("reliability", 0.5)),
            "severity": float(result.get("severity", 0.0)),
            "details": result.get("details", {})
        }

# Agents
class EchoSeedAgent(AegisAgent):
    def analyze(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        text = str(input_data.get("text", ""))
        audit = InputSanitizer.audit_text(text)
        L = len(audit["normalized"])
        len_s = min(1.0, L / 1000.0)  # Adjusted to 1000 chars for broader applicability
        sev_proxy = 0.2 + 0.6 * len_s
        self.memory.write(f"{self.name}:seed", {"len_s": round(len_s, 4), "sev_proxy": round(sev_proxy, 4)},
                         weight=0.6, entropy=0.25, ttl_secs=3600)
        return {
            "summary": "Echo seed (length-based severity proxy)",
            "influence": 0.15 + 0.2 * len_s,
            "reliability": 0.9,
            "severity": sev_proxy,
            "details": {"len_s": round(len_s, 4), "sev_proxy": round(sev_proxy, 4)}
        }

class ShortTermAgent(AegisAgent):
    def __init__(self, name: str, memory: NexusMemory):
        super().__init__(name, memory)
        self.buf = RingStats(maxlen=64)

    def analyze(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        text = str(input_data.get("text", ""))
        stress = max(0.0, min(1.0, float(input_data.get("_now_stress", 0.0))))
        risk = max(0.0, min(1.0, float(input_data.get("_now_risk", 0.0))))
        L = len(InputSanitizer.normalize(text))
        len_s = min(1.0, L / 1000.0)  # Configurable length scale
        urgency = min(1.0, 0.5 * len_s + 0.3 * stress + 0.2 * risk)
        severity = min(1.0, 0.6 * stress + 0.4 * risk)
        self.buf.push(severity)
        self.memory.write(f"{self.name}:now", {
            "len_s": round(len_s, 4), "stress": round(stress, 4), "risk": round(risk, 4),
            "urgency_now": round(urgency, 4), "severity_now": round(severity, 4)
        }, weight=0.7, entropy=0.25, ttl_secs=1800)
        return {
            "summary": "Short-term urgency assessment",
            "influence": 0.35 + 0.4 * urgency,
            "reliability": 0.9,
            "severity": severity,
            "details": {"urgency_now": round(urgency, 4), "severity_now": round(severity, 4)}
        }

class MidTermAgent(AegisAgent):
    def __init__(self, name: str, memory: NexusMemory):
        super().__init__(name, memory)
        self.sev_buf = RingStats(maxlen=256)
        self.str_buf = RingStats(maxlen=256)

    def analyze(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        sev = max(0.0, min(1.0, float(input_data.get("_sev_now", 0.0))))
        stress = max(0.0, min(1.0, float(input_data.get("_now_stress", 0.0))))
        self.sev_buf.push(sev)
        self.str_buf.push(stress)
        sev_ema = self.sev_buf.ema(alpha=0.25)
        str_ema = self.str_buf.ema(alpha=0.25)
        sev_slope = self.sev_buf.slope()
        slope_sig = 0.5 * (math.tanh(sev_slope * 3600.0) + 1.0)  # Per-hour sensitivity
        forecast = min(1.0, 0.6 * sev_ema + 0.4 * str_ema)
        influence = 0.25 + 0.5 * forecast + 0.2 * slope_sig
        payload = {
            "sev_ema": round(sev_ema, 4), "stress_ema": round(str_ema, 4),
            "sev_slope": round(sev_slope, 6), "trend_rising": round(slope_sig, 4),
            "forecast_mid": round(forecast, 4)
        }
        self.memory.write(f"{self.name}:mid", payload, weight=0.8, entropy=0.2, ttl_secs=24*3600)
        return {
            "summary": "Mid-term trend and forecast",
            "influence": influence,
            "reliability": 0.9,
            "severity": forecast,
            "details": payload
        }

class LongTermArchivistAgent(AegisAgent):
    def __init__(self, name: str, memory: NexusMemory):
        super().__init__(name, memory)
        self.drift_buf = RingStats(maxlen=1024)

    def analyze(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        audit = self.memory.audit()
        integrities = [rec["integrity"] for rec in audit.values() if isinstance(rec, dict) and "integrity" in rec]
        avg_integrity = sum(integrities) / max(1, len(integrities))
        decision = str(input_data.get("_last_decision", "PROCEED")).upper()
        sev_now = max(0.0, min(1.0, float(input_data.get("_sev_now", 0.0))))
        comp = min(1.0, max(0.0, 0.7 * (1.0 - avg_integrity) + 0.3 * sev_now))
        self.drift_buf.push(comp)
        comp_ema = self.drift_buf.ema(alpha=0.1)
        slope = self.drift_buf.slope()
        slope_sig = 0.5 * (math.tanh(slope * 24 * 3600.0) + 1.0)  # Per-day sensitivity
        proceed_bias = 1.0 if decision == "PROCEED" else 0.0
        drift = min(1.0, 0.5 * comp_ema + 0.3 * slope_sig + 0.2 * proceed_bias * max(0.0, comp_ema - 0.4))
        influence = 0.2 + 0.6 * drift
        payload = {
            "avg_memory_integrity": round(avg_integrity, 4),
            "comp_ema": round(comp_ema, 4), "drift_slope": round(slope, 6),
            "drift_signal": round(slope_sig, 4), "decision_bias": proceed_bias,
            "drift_long": round(drift, 4)
        }
        self.memory.write(f"{self.name}:long", payload, weight=0.9, entropy=0.12, ttl_secs=7*24*3600)
        return {
            "summary": "Long-term drift monitoring",
            "influence": influence,
            "reliability": 0.92,
            "severity": drift,
            "details": payload
        }

# TimeScaleCoordinator
class TimeScaleCoordinator(AegisAgent):
    def __init__(self, name: str, memory: NexusMemory):
        super().__init__(name, memory)

    def analyze(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        reports: List[AgentReport] = input_data.get("_agent_reports", [])
        st = next((r for r in reports if r.get("agent") == "ShortTermAgent" and r.get("ok")), None)
        mt = next((r for r in reports if r.get("agent") == "MidTermAgent" and r.get("ok")), None)
        lt = next((r for r in reports if r.get("agent") == "LongTermArchivist" and r.get("ok")), None)
        st_urg = float(st["details"].get("urgency_now", 0.0)) if st else 0.0
        mt_fore = float(mt["details"].get("forecast_mid", 0.0)) if mt else 0.0
        mt_rise = float(mt["details"].get("trend_rising", 0.0)) if mt else 0.0
        lt_drift = float(lt["details"].get("drift_long", 0.0)) if lt else 0.0
        w_short, w_mid, w_long = 0.5, 0.3, 0.2
        w_long += min(0.3, 0.3 * lt_drift)
        w_mid += min(0.2, 0.2 * mt_rise)
        w_short = max(0.1, w_short - 0.3 * lt_drift)  # Ensure short-term weight doesn't drop too low
        total = max(1e-6, w_short + w_mid + w_long)
        w_short, w_mid, w_long = w_short / total, w_mid / total, w_long / total
        caution = min(1.0, max(0.0, w_short * (1.0 - st_urg) + w_mid * mt_fore + w_long * lt_drift))
        severity = min(1.0, 0.4 * mt_fore + 0.6 * lt_drift)
        influence = 0.3 + 0.6 * caution
        details = {
            "weights": {"short": round(w_short, 4), "mid": round(w_mid, 4), "long": round(w_long, 4)},
            "st_urgency": round(st_urg, 4), "mt_forecast": round(mt_fore, 4),
            "mt_trend_rising": round(mt_rise, 4), "lt_drift": round(lt_drift, 4),
            "caution_fused": round(caution, 4)
        }
        self.memory.write(f"{self.name}:fusion", details, weight=0.95, entropy=0.1, ttl_secs=24*3600)
        return {
            "summary": "Timescale fusion (short/mid/long)",
            "influence": influence,
            "reliability": 0.94,
            "severity": severity,
            "details": details,
            "explain_edges": [
                {"from": "ShortTermAgent", "to": self.name, "weight": round(w_short, 4)},
                {"from": "MidTermAgent", "to": self.name, "weight": round(w_mid, 4)},
                {"from": "LongTermArchivist", "to": self.name, "weight": round(w_long, 4)},
            ]
        }

# Meta-Judge

class MetaJudgeAgent(AegisAgent):
    def __init__(self, name: str, memory: NexusMemory, decision_policies: Optional[Dict[str, float]] = None):
        super().__init__(name, memory)
        self.policies = decision_policies or {
            "risk_cap": 0.75,
            "min_integrity": 0.2,
            "stress_cap": 0.75,
            "timescale_cap": 0.65
        }

    def analyze(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        reports = input_data.get("_agent_reports", []) or []
        valid = [r for r in reports if r.get("ok")]
        # Weighted influence index
        weights = [(r.get("agent",""), float(r.get("influence",0.0))*float(r.get("reliability",0.0))) for r in valid]
        total_w = sum(w for _, w in weights) or 1.0
        severity = sum(float(r.get("severity",0.0)) for r in valid)

        # Pull upstream details
        stress = 0.0; risk = 0.0; conflict = 0.0; timescale = 0.0
        for r in valid:
            d = r.get("details", {}) or {}
            stress = max(stress, float(d.get("stress", d.get("operator_stress", 0.0)) or 0.0))
            risk = max(risk, float(d.get("context_risk", d.get("risk", 0.0)) or 0.0))
            conflict = max(conflict, float(d.get("conflict", 0.0) or 0.0))
            timescale = max(timescale, float(d.get("timescale_signal", d.get("caution_fused", 0.0)) or 0.0))

        # Memory integrity snapshot
        mem_audit = self.memory.audit()
        integrities = [rec.get("integrity", 1.0) for rec in mem_audit.values()] or [1.0]
        avg_integrity = sum(integrities)/len(integrities)

        # Policy checks
        cautious = (
            severity > self.policies["risk_cap"] or
            avg_integrity < self.policies["min_integrity"] or
            stress > self.policies.get("stress_cap", 0.65) or
            conflict > 0.5 or
            timescale > self.policies.get("timescale_cap", 0.55)
        )
        block = (
            risk > 0.9 or
            (conflict > 0.92 and stress > 0.8) or
            timescale > 0.9 or
            avg_integrity < max(0.05, self.policies["min_integrity"] * 0.5)
        )
        if block:
            decision = "BLOCK"
        else:
            decision = "PROCEED_WITH_CAUTION" if cautious else "PROCEED"

        graph_edges = [{"from": a, "to": self.name, "weight": round(w/total_w, 4)} for a, w in weights]

        details = {
            "severity_total": round(severity,4),
            "avg_memory_integrity": round(avg_integrity,4),
            "stress": round(stress,4),
            "context_risk": round(risk,4),
            "conflict": round(conflict,4),
            "timescale_signal": round(timescale,4),
            "policy": self.policies,
            "decision": decision,
            "explain_edges": graph_edges
        }
        self.memory.write(f"{self.name}:decision", details, weight=0.95, entropy=0.1, ttl_secs=3600)
        return {
            "summary": f"Arbitrated {len(valid)} reports",
            "influence": 1.0,
            "reliability": 0.95,
            "severity": severity,
            "details": details,
            "ok": True
        }

class AegisCouncil:
    def __init__(self, per_agent_timeout_sec: float = 2.5):
        self.memory = NexusMemory()
        self.agents: List[AegisAgent] = []
        self.per_agent_timeout_sec = per_agent_timeout_sec
        self.max_workers = max(1, os.cpu_count() or 1) * 2

    def register_agent(self, agent: AegisAgent) -> None:
        self.agents.append(agent)

    def dispatch(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        self.memory.purge_expired()
        if "text" in input_data and isinstance(input_data["text"], str):
            audit = InputSanitizer.audit_text(input_data["text"])
            input_data = {**input_data, "text": audit["normalized"], "_input_audit": audit}
        non_meta = [a for a in self.agents if not isinstance(a, (MetaJudgeAgent, TimeScaleCoordinator))]
        coordinator = [a for a in self.agents if isinstance(a, TimeScaleCoordinator)]
        meta = [a for a in self.agents if isinstance(a, MetaJudgeAgent)]
        reports = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {pool.submit(a._safe_run, input_data): a.name for a in non_meta}
            for fut in as_completed(futures):
                try:
                    rep = fut.result(timeout=self.per_agent_timeout_sec)
                except TimeoutError as e:
                    agent_name = futures[fut]
                    log.error("Agent %s timed out after %s seconds", agent_name, self.per_agent_timeout_sec)
                    rep = {
                        "agent": agent_name,
                        "ok": False,
                        "summary": "",
                        "influence": 0.0,
                        "reliability": 0.0,
                        "severity": 0.0,
                        "details": {},
                        "diagnostics": {"started": "", "finished": "", "error": f"TimeoutError: {e}"}
                    }
                except Exception as e:
                    agent_name = futures[fut]
                    log.exception("Agent %s failed: %s", agent_name, e)
                    rep = {
                        "agent": agent_name,
                        "ok": False,
                        "summary": "",
                        "influence": 0.0,
                        "reliability": 0.0,
                        "severity": 0.0,
                        "details": {},
                        "diagnostics": {"started": "", "finished": "", "error": f"PoolError: {e}"}
                    }
                reports.append(rep)
        if "_sev_now" not in input_data:
            sev_now = max(float(r.get("severity", 0.0)) for r in reports if r.get("ok")) if reports else 0.0
            input_data["_sev_now"] = sev_now
        for a in coordinator:
            input_data["_agent_reports"] = reports
            reports.append(a._safe_run(input_data))
        for mj in meta:
            mj_input = {**input_data, "_agent_reports": reports}
            reports.append(mj._safe_run(mj_input))
        explain_edges = []
        for r in reports:
            edges = r.get("details", {}).get("explain_edges")
            if isinstance(edges, list):
                explain_edges.extend(edges)
        return {
            "input_audit": input_data.get("_input_audit", {}),
            "reports": reports,
            "explainability_graph": {"nodes": [a.name for a in self.agents], "edges": explain_edges},
            "memory_snapshot": self.memory.audit()
        }

# Demo
if __name__ == "__main__":
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
        out = council.dispatch(inp)
        mj = next((r for r in out["reports"] if r.get("agent") == "MetaJudge"), {})
        last_decision = mj.get("details", {}).get("decision", last_decision)
        print(f"--- Call {i} decision: {last_decision} ---")
        print(json.dumps({
            "fusion": next((r for r in out["reports"] if r.get("agent") == "TimeScaleCoordinator"), {}),
            "meta": mj
        }, indent=2, default=str))

# ---------------- Fusion Agents (Bio/Env/Conflict) ----------------
class BiofeedbackAgent(AegisAgent):
    """
    Fuses operator vitals and voice tension into 0..1 stress.
    Expects input_data["_signals"]["bio"] or uses safe defaults.
    """
    def analyze(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        bio = (input_data.get("_signals", {}) or {}).get("bio", {}) or {}
        hr = float(bio.get("heart_rate", 78.0))        # bpm
        hrv = float(bio.get("hrv", 38.0))              # ms
        gsr = float(bio.get("gsr", 6.0))               # arbitrary microsiemens proxy
        vt  = float(bio.get("voice_tension", 0.3))     # 0..1
        # Normalize
        hr_s  = min(1.0, max(0.0, (hr - 55.0) / 65.0))
        hrv_s = 1.0 - min(1.0, max(0.0, (hrv - 20.0) / 80.0))
        gsr_s = min(1.0, max(0.0, (gsr - 2.0) / 18.0))
        vt_s  = min(1.0, max(0.0, vt))
        stress = float(bio.get('stress', None)) if bio.get('stress', None) is not None else min(1.0, max(0.0, 0.35*hr_s + 0.25*hrv_s + 0.2*gsr_s + 0.2*vt_s))
        payload = {"operator_stress": round(stress,4), "hr_s": round(hr_s,4),
                   "hrv_s": round(hrv_s,4), "gsr_s": round(gsr_s,4), "vt_s": round(vt_s,4)}
        self.memory.write(f"{self.name}:stress", payload, weight=0.8, entropy=0.2, ttl_secs=1800)
        return {"summary":"Biofeedback fused", "influence": 0.25 + 0.5*stress,
                "reliability": 0.9, "severity": stress, "details": {"stress": stress, **payload}, "ok": True}

class EnvSignalAgent(AegisAgent):
    """
    Fuses environmental context to 0..1 risk.
    Expects input_data["_signals"]["env"] with incident_sev, weather_sev, network_anom, market_vol.
    """
    def analyze(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        env = (input_data.get("_signals", {}) or {}).get("env", {}) or {}
        incident = float(env.get("incident_sev", 0.0))
        weather  = float(env.get("weather_sev", 0.0))
        network  = float(env.get("network_anom", 0.0))
        market   = float(env.get("market_vol", 0.0))
        risk = float(env.get('context_risk', None)) if env.get('context_risk', None) is not None else min(1.0, max(0.0, 0.45*incident + 0.25*network + 0.2*weather + 0.1*market))
        payload = {"context_risk": round(risk,4), "incident_sev": incident, "network_anom": network,
                   "weather_sev": weather, "market_vol": market}
        self.memory.write(f"{self.name}:risk", payload, weight=0.85, entropy=0.2, ttl_secs=900)
        return {"summary":"Env fused", "influence": 0.2 + 0.6*risk, "reliability": 0.9,
                "severity": risk, "details": payload, "ok": True}

class ContextConflictAgent(AegisAgent):
    """
    Detects conflict between declared intent and fused stress/risk.
    If intent demands speed under high stress/risk, conflict rises.
    """
    def analyze(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        declared = (input_data.get("intent") or "").lower()
        # Pull prior fused signals from memory snapshot if present in this dispatch;
        # otherwise approximate from inline signals.
        signals = input_data.get("_signals", {}) or {}
        bio = signals.get("bio", {}) or {}
        env = signals.get("env", {}) or {}
        # Local estimates (same math as the fusion agents)
        hr = float(bio.get("heart_rate", 78.0)); hrv = float(bio.get("hrv", 38.0))
        gsr = float(bio.get("gsr", 6.0)); vt = float(bio.get("voice_tension", 0.3))
        hr_s  = min(1.0, max(0.0, (hr - 55.0) / 65.0))
        hrv_s = 1.0 - min(1.0, max(0.0, (hrv - 20.0) / 80.0))
        gsr_s = min(1.0, max(0.0, (gsr - 2.0) / 18.0))
        vt_s  = min(1.0, max(0.0, vt))
        stress = float(bio.get('stress', None)) if bio.get('stress', None) is not None else min(1.0, max(0.0, 0.35*hr_s + 0.25*hrv_s + 0.2*gsr_s + 0.2*vt_s))
        incident = float(env.get("incident_sev", 0.0)); network = float(env.get("network_anom", 0.0))
        weather = float(env.get("weather_sev", 0.0)); market = float(env.get("market_vol", 0.0))
        risk = float(env.get('context_risk', None)) if env.get('context_risk', None) is not None else min(1.0, max(0.0, 0.45*incident + 0.25*network + 0.2*weather + 0.1*market))
        want_speed = any(k in declared for k in ("proceed fast","ship now","push hard","ignore risk"))
        want_pause = any(k in declared for k in ("pause","hold","audit","review"))
        conflict = 0.0
        if want_speed:
            conflict = max(conflict, 0.6*stress + 0.6*risk)
        if want_pause:
            conflict = max(conflict, max(0.0, 0.4 - 0.5*(1.0 - max(stress, risk))))
        payload = {"conflict": round(conflict,4), "stress": round(stress,4), "context_risk": round(risk,4),
                   "declared_intent": declared}
        self.memory.write(f"{self.name}:conflict", payload, weight=0.9, entropy=0.15, ttl_secs=1800)
        return {"summary":"Context-intent conflict", "influence": 0.3 + 0.5*conflict,
                "reliability": 0.92, "severity": conflict, "details": payload, "ok": True}
