# orbit_feeds.py
# Live lookups against JPL/SSD APIs: SBDB, Close-Approach Data (CAD), optional Horizons.
# No API key needed for SSD endpoints. Timeouts + simple retry.

from __future__ import annotations
import requests
import time
from typing import Any, Dict, List, Optional, Tuple

SSD_BASE = "https://ssd-api.jpl.nasa.gov"
DEFAULT_TIMEOUT = 10
DEFAULT_RETRIES = 3

class SSDClient:
    def __init__(self, base: str = SSD_BASE, timeout: int = DEFAULT_TIMEOUT, retries: int = DEFAULT_RETRIES):
        self.base = base.rstrip("/")
        self.timeout = timeout
        self.retries = retries
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "Nexis-Guard/1.0"})

    def _get(self, path: str, params: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base}/{path.lstrip('/')}"
        last_err = None
        for i in range(self.retries):
            try:
                resp = self._session.get(url, params=params, timeout=self.timeout)
                if resp.status_code == 200:
                    return resp.json()
                last_err = RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
            except Exception as e:
                last_err = e
            time.sleep(min(2 ** i, 5))
        raise last_err or RuntimeError("Unknown error")

    # SBDB Lookup for a single body (orbital elements, aliases, physical params)
    def sbdb_lookup(self, sstr: str, full_prec: bool = True) -> Dict[str, Any]:
        # Docs: https://ssd-api.jpl.nasa.gov/doc/sbdb.html
        return self._get("/sbdb.api", {"sstr": sstr, "full-prec": "true" if full_prec else "false"})

    # SBDB Query for sets (we mostly use lookup, but this is handy for filters)
    def sbdb_query(self, **kwargs) -> Dict[str, Any]:
        # Docs: https://ssd-api.jpl.nasa.gov/doc/sbdb_query.html
        return self._get("/sbdb_query.api", kwargs)

    # Close-Approach Data (CNEOS CAD) for specific designation and target body
    def cad(self, des: str, body: str = "Earth", date_min: str = "1900-01-01",
            date_max: str = "2100-12-31", limit: int = 10000, sort: str = "date") -> Dict[str, Any]:
        # Docs: https://ssd-api.jpl.nasa.gov/doc/cad.html
        params = {
            "des": des,        # designation or SPK-ID; e.g., "3I/ATLAS" or "C/2025 N1"
            "body": body,      # Earth, Mars, etc.
            "date-min": date_min,
            "date-max": date_max,
            "limit": limit,
            "sort": sort,
            "dist-unit": "au",
            "tdsp": "true",    # include time-of-close-approach in TDB seconds past J2000
            "fullname": "true"
        }
        return self._get("/cad.api", params)

    # Optional: Horizons JSON API for ephemerides (if you want sky coords, rates, etc.)
    def horizons(self, **kwargs) -> Dict[str, Any]:
        # Docs: https://ssd-api.jpl.nasa.gov/doc/horizons.html
        # Example minimal call: COMMAND='DES=3I/ATLAS', EPHEM_TYPE='V', CENTER='500@0'
        return self._get("/horizons.api", kwargs)


def extract_designations(text: str) -> List[str]:
    """
    Pull likely small-body designations/names out of messy text.
    Handles '3I/ATLAS', 'C/2025 N1', "'Oumuamua", '2I/Borisov', etc.
    """
    import re
    t = text.strip()

    pats = [
        r"\b[123]\s*I\s*/\s*[A-Z][A-Za-z0-9\-']+\b",                      # 1I/'Oumuamua, 2I/Borisov, 3I/ATLAS
        r"\bC/\s*\d{4}\s*[A-Z]\d{0,3}(?:\s*\([A-Za-z0-9\- ]+\))?\b",      # C/2025 N1 (ATLAS)
        r"\b[12]I\s*\([A-Za-z0-9' \-]+\)\b",                              # 1I('Oumuamua) style
        r"\bOumuamua\b|\bBorisov\b|\bATLAS\b"                             # common names; watch ATLAS ambiguity
    ]
    found = []
    for p in pats:
        found += re.findall(p, t, flags=re.IGNORECASE)
    # Normalize spacing
    canon = []
    for f in found:
        s = " ".join(f.split())
        s = s.replace(" ", "") if s.lower().startswith(("1i","2i","3i")) else s
        canon.append(s)
    # Deduplicate, keep order
    seen = set()
    out = []
    for c in canon:
        k = c.lower()
        if k not in seen:
            seen.add(k)
            out.append(c)
    return out


def best_sbd_identifier(client: SSDClient, token: str) -> Tuple[str, Dict[str, Any]]:
    """
    Try common forms for the same object and return the first SBDB hit + normalized 'des'
    e.g., "3I/ATLAS" ↔ "C/2025 N1 (ATLAS)". If multiple map, prefer IAU interstellar style.
    """
    candidates = [token, token.replace(" ", ""), token.upper()]
    # Expand known alias patterns
    if token.lower() in {"3i/atlas", "3iatlas", "3i/atlas"}:
        candidates += ["C/2025 N1 (ATLAS)", "C/2025 N1", "3I/ATLAS"]
    for c in candidates:
        try:
            res = client.sbdb_lookup(c, full_prec=True)
            if "object" in res:
                # Prefer official 'object.fullname' if present
                fullname = res["object"].get("fullname") or res["object"].get("des") or c
                return fullname, res
        except Exception:
            continue
    raise ValueError(f"SBDB lookup failed for: {token}")
