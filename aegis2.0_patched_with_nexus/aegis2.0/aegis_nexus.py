
# aegis_nexus.py
import os, json, time, math, threading, hashlib, random
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
try:
    import portalocker  # type: ignore
    HAVE_PORTALOCKER = True
except Exception:
    HAVE_PORTALOCKER = False
    class _NoLock:
        def __init__(self, f, *a, **k): self.f=f
        def __enter__(self): return self.f
        def __exit__(self, *a): return False
    class portalocker:  # shim
        LOCK_EX = 0
        @staticmethod
        def Lock(f, *a, **k):
            return _NoLock(f)

def utc_ts() -> float:
    return time.time()

def sha256(s: bytes) -> str:
    import hashlib
    return hashlib.sha256(s).hexdigest()

def entropy_of(text: str) -> float:
    if not text:
        return 0.0
    from math import log2
    freq = {}
    n = 0
    for ch in text:
        freq[ch] = freq.get(ch, 0) + 1
        n += 1
    return -sum((c/n) * log2(c/n) for c in freq.values())

@dataclass
class SignalRecord:
    channel: str              # "bio", "env", "intent", "virtue", etc.
    key: str                  # e.g., "stress", "context_risk"
    value: float              # normalized 0..1 where possible
    ts: float                 # epoch seconds
    source: str               # e.g., "agent:BiofeedbackAgent"
    entropy: float = 0.0      # data entropy (if textual) else derived jitter
    valence: float = 0.5      # 0..1 (positive/negative affect proxy)
    arousal: float = 0.5      # 0..1 (activation/energy)
    integrity: float = 1.0    # trust in signal 0..1
    ttl_sec: float = 3600.0   # time-to-live
    meta: Dict[str, Any] = None

    def to_dict(self):
        d = asdict(self)
        if d["meta"] is None: d["meta"] = {}
        return d

class NexusSignalEngine:
    """
    Persistent, entropy-aware signal store with temporal decay and integrity scoring.
    - append-only JSONL files in root/ by day
    - in-memory latest index: (channel,key) -> SignalRecord
    - decay(value,t) = value * exp(-lambda * age_hours) * integrity
    - integrity self-heals via agreement across recent writers
    """
    def __init__(self, root: str = "./nexus", decay_lambda_per_hr: float = 0.06):
        self.root = root
        os.makedirs(self.root, exist_ok=True)
        self.decay_lambda = max(0.0, float(decay_lambda_per_hr))
        self._latest: Dict[Tuple[str,str], SignalRecord] = {}
        self._lock = threading.Lock()

    # -------- persistence helpers --------
    def _path_for_day(self, ts: float) -> str:
        day = datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d")
        return os.path.join(self.root, f"signals-{day}.jsonl")

    def _append(self, rec: SignalRecord):
        path = self._path_for_day(rec.ts)
        line = json.dumps(rec.to_dict(), separators=(",", ":")) + "\n"
        # Lock file for atomic append (best-effort with shim if missing)
        with open(path, "a", encoding="utf-8") as f:
            with portalocker.Lock(f, portalocker.LOCK_EX):
                f.write(line)

    # -------- core API --------
    def ingest(self, channel: str, key: str, value: float, source: str, *, valence: float = 0.5,
               arousal: float = 0.5, integrity: float = 0.9, ttl_sec: float = 3600.0, meta: Optional[Dict[str,Any]] = None) -> SignalRecord:
        ts = utc_ts()
        v = float(max(0.0, min(1.0, value)))
        rec = SignalRecord(channel=channel, key=key, value=v, ts=ts, source=source,
                           entropy=float(meta.get("entropy", 0.0) if meta else 0.0),
                           valence=float(valence), arousal=float(arousal),
                           integrity=float(max(0.0, min(1.0, integrity))), ttl_sec=float(ttl_sec),
                           meta=meta or {})
        with self._lock:
            self._append(rec)
            self._latest[(channel, key)] = rec
        return rec

    def get_latest(self, channel: str, key: str) -> Optional[SignalRecord]:
        with self._lock:
            return self._latest.get((channel, key))

    def value(self, channel: str, key: str, *, now: Optional[float] = None) -> Optional[float]:
        rec = self.get_latest(channel, key)
        if not rec: return None
        now = now or utc_ts()
        age_hr = max(0.0, (now - rec.ts) / 3600.0)
        decay = math.exp(-self.decay_lambda * age_hr)
        alive = (now - rec.ts) <= rec.ttl_sec
        val = rec.value * decay * rec.integrity if alive else None
        return val

    def fuse(self, specs: List[Tuple[str,str,float]]) -> Optional[float]:
        """
        Weighted fusion over (channel,key,weight). Skips missing.
        Returns None if no signals present.
        """
        numer = 0.0
        denom = 0.0
        for ch,k,w in specs:
            v = self.value(ch,k)
            if v is None: continue
            numer += v * float(w)
            denom += float(w)
        if denom == 0.0: return None
        return max(0.0, min(1.0, numer/denom))

    def bump_integrity(self, channel: str, key: str, delta: float):
        with self._lock:
            rec = self._latest.get((channel,key))
            if not rec: return
            rec.integrity = float(max(0.0, min(1.0, rec.integrity + delta)))
            self._latest[(channel,key)] = rec

    def audit(self) -> Dict[str, Any]:
        out = {}
        with self._lock:
            for (ch,k), rec in self._latest.items():
                out[f"{ch}:{k}"] = {
                    "ts": rec.ts,
                    "value": rec.value,
                    "integrity": rec.integrity,
                    "ttl_sec": rec.ttl_sec
                }
        return out
