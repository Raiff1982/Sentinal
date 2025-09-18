from datetime import datetime, timezone, timedelta
import os, threading, hashlib
from typing import Dict
from .council_bundle import CouncilBundle
from .ledger_record import LedgerRecord
import portalocker
import json
import logging
log = logging.getLogger("AEGIS-Ledger")
class SignedLedger:
    def __init__(self, dirpath: str = DEFAULT_LEDGER_DIR, secret_path: str = DEFAULT_SECRET_PATH, retention_days: int = DEFAULT_RETENTION_DAYS):
        self.dir = os.path.abspath(dirpath)
        self.secret = os.environ.get("SENTINEL_HMAC_KEY", load_or_create_secret(secret_path)).encode("utf-8")
        self.key_id = hashlib.sha256(self.secret).hexdigest()[:8]
        self._lock = threading.Lock()
        self.retention_days = retention_days
        self.old_keys: Dict[str, bytes] = {}  # For rotation verification

    def append(self, council_input: Dict, council_bundle: CouncilBundle, policy_snapshot: Dict) -> LedgerRecord:
        self._prune_old_files()
        ts = utcnow()
        decision = council_bundle.get("decision", "")
        edges = council_bundle.get("explainability_graph", {}).get("edges", [])
        rec: LedgerRecord = {
            "ts": ts,
            "decision": decision,
            "policy": policy_snapshot,
            "input_digest": self._input_digest(council_input),
            "edges_digest": self._edges_digest(edges),
            "bundle_digest": self._bundle_digest(council_bundle),
            "key_id": self.key_id,
            "sig": ""
        }
        rec["sig"] = self.sign({k: v for k, v in rec.items() if k != "sig"})
    path = self._path_for_date(datetime.now(timezone.utc))
        with self._lock:
            try:
                with portalocker.Lock(path, "a", timeout=5) as f:
                    f.write(json.dumps(rec, separators=(",", ":")) + "\n")
            except (OSError, portalocker.LockException) as e:
                log.error("Failed to append to ledger %s: %s", path, e)
                raise
        return rec

    def verify_file(self, path: str) -> Dict[str, Any]:
        bad = []
        total = 0
        with open(path, "r", encoding="utf-8") as f:
            for i, ln in enumerate(f, start=1):
                total += 1
                try:
                    rec = json.loads(ln.strip())
                    key_id = rec.pop("key_id", self.key_id)
                    sig = rec.pop("sig", "")
                    key = self.old_keys.get(key_id, self.secret)
                    good = hmac.compare_digest(sig, hmac.new(key, stable_json(rec), hashlib.sha256).hexdigest())
                    if not good:
                        bad.append(i)
                        log.warning("Invalid signature in %s at line %d", path, i)
                except json.JSONDecodeError as e:
                    bad.append(i)
                    log.warning("Invalid JSON in %s at line %d: %s", path, i, e)
        return {"file": path, "lines": total, "bad": bad, "ok": len(bad) == 0}