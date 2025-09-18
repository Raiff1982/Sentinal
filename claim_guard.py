# claim_guard.py
# Fuse hoax heuristics + live SSD data + your Nexis engine verdict.

from __future__ import annotations
from typing import Any, Dict, Optional, List, Tuple
from datetime import datetime
from sentinal.hoax_filter import HoaxFilter
from sentinal.nexis_signal_engine import NexisSignalEngine
from orbit_feeds import SSDClient, extract_designations, best_sbd_identifier

ISO_FMT = "%Y-%m-%d"

class LiveClaimGuard:
    def __init__(self, db_path: str = "signals.db", extraordinary_km: float = 50.0):
        self.engine = NexisSignalEngine(memory_path=db_path)
        self.hoax = HoaxFilter(extraordinary_km=extraordinary_km)
        self.ssd = SSDClient()

    def _cad_min_distance(self, des: str, body: str = "Earth",
                          date_min: str = "2000-01-01", date_max: str = "2100-12-31") -> Optional[Dict[str, Any]]:
        cad = self.ssd.cad(des=des, body=body, date_min=date_min, date_max=date_max, limit=10000)
        data = cad.get("data") or []
        # CAD fields order: des, orbit_id, jd, cd, dist, dist_min, dist_max, v_rel, v_inf, t_sigma_f, body, h, fullname
        best = None
        for row in data:
            try:
                dist = float(row[4])  # au
                if best is None or dist < best["dist_au"]:
                    best = {
                        "cd_utc": row[3],
                        "dist_au": dist,
                        "v_rel_km_s": float(row[7]) if row[7] not in (None, "") else None,
                        "fullname": row[12] if len(row) > 12 else row[0]
                    }
            except Exception:
                continue
        return best

    def evaluate(self, text: str, source_url: Optional[str] = None,
                 earth_window: Tuple[str, str] = ("2000-01-01", "2100-12-31")) -> Dict[str, Any]:
        # 1) Run Nexis base + hoax heuristics
        base = self.engine.process_news(text, source_url=source_url)

        # 2) Try to resolve any small-body mentions to SBDB identifiers
        mentions = extract_designations(text)
        facts: List[Dict[str, Any]] = []
        for m in mentions:
            try:
                fullname, sb = best_sbd_identifier(self.ssd, m)
                # 3) Pull Earth CAD for the analysis window
                cad_best = self._cad_min_distance(fullname, body="Earth",
                                                  date_min=earth_window[0], date_max=earth_window[1])
                facts.append({
                    "query": m,
                    "resolved_fullname": fullname,
                    "sbdb": sb.get("object", {}),
                    "closest_approach": cad_best
                })
            except Exception as e:
                facts.append({"query": m, "error": str(e)})

        # 3I/ATLAS sanity: if text mentions Saturn, pull Mars/Sun perihelion context too (optional)

        # 3) Merge data into the record and adjust verdict if facts contradict scare-claims
        # If any resolved object shows Earth min distance > 0.3 au across window, consider it "no threat".
        threat_flags = []
        for f in facts:
            ca = f.get("closest_approach")
            if ca and ca.get("dist_au") is not None:
                if ca["dist_au"] >= 0.3:
                    threat_flags.append("no_near_earth_encounter")
                else:
                    threat_flags.append("potentially_close")
            else:
                threat_flags.append("cad_unknown")

        if "no_near_earth_encounter" in threat_flags and base["verdict"] != "approved":
            # Downgrade fear, upgrade confidence
            base["verdict"] = "adaptive intervention"
            base["message"] = "Live orbital data shows no close Earth approach; claim requires stronger evidence."

        base["live_orbital_facts"] = {
            "resolved": facts,
            "analysis_window": {"date_min": earth_window[0], "date_max": earth_window[1]}
        }
        return base
