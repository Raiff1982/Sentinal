# hoax_filter.py
# Lightweight, stateless misinformation heuristics for language/source/scale

import re
from urllib.parse import urlparse
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List

_NUMBER_UNIT = re.compile(
    r'(?P<num>[\d,]+(?:\.\d+)?)\s*(?P<unit>mile|miles|km|kilometer|kilometers)',
    re.I
)

LANG_RED_FLAGS = [
    r'\brecently\s+declassified\b',
    r'\bshocking\b',
    r'\bastonishing\b',
    r'\bexplosive\b',
    r'\bexperts\s+say\b',
    r'\breportedly\b',
    r'\bmothership\b',
    r'\bancient\s+alien\b',
    r'\bdormant\s+(?:observational\s+)?craft\b',
    r'\bangular\s+edges\b',
    r'\bviral\b',
    r'\bnever\s+before\s+seen\b',
    r'\bshaking\s+(?:the\s+)?scientific\s+community\b',
    r'\bfootage\b',
]

ALLOW_DOMAINS = {
    'nasa.gov', 'jpl.nasa.gov', 'pds.nasa.gov', 'science.nasa.gov', 'heasarc.gsfc.nasa.gov',
    'esa.int', 'esawebservices.esa.int', 'esa-maine.esa.int',
    'noirlab.edu', 'cfa.harvard.edu', 'caltech.edu', 'berkeley.edu', 'mit.edu',
    'nature.com', 'science.org', 'iopscience.iop.org', 'agu.org',
    'arxiv.org', 'adsabs.harvard.edu',
}

DENY_DOMAINS = {
    'm.facebook.com', 'facebook.com', 'x.com', 'twitter.com', 't.co',
    'tiktok.com', 'youtube.com', 'youtu.be', 'instagram.com', 'reddit.com',
}

MEDIUM_DOMAINS = {
    'dailyMail.co.uk', 'dailymail.co.uk', 'newyorkpost.com', 'the-sun.com',
    'mirror.co.uk', 'sputniknews.com', 'rt.com',
}

@dataclass
class HoaxFilterResult:
    red_flag_hits: int
    source_score: float
    scale_score: float
    combined: float
    notes: Dict[str, Any]

class HoaxFilter:
    """
    Scores are in [0,1]; higher means more likely hoax/misinformation.
    """

    def __init__(self,
                 red_flag_weight: float = 0.35,
                 source_weight: float   = 0.25,
                 scale_weight: float    = 0.40,
                 extraordinary_km: float = 50.0):
        self.red_flag_weight = red_flag_weight
        self.source_weight   = source_weight
        self.scale_weight    = scale_weight
        self.extraordinary_km = extraordinary_km
        self._flag_res = [re.compile(p, re.I) for p in LANG_RED_FLAGS]

    @staticmethod
    def _km_from_match(num: str, unit: str) -> float:
        n = float(num.replace(',', ''))
        if unit.lower().startswith('mile'):
            return n * 1.609344
        return n

    def language_red_flags(self, text: str) -> Tuple[int, List[str]]:
        hits = []
        for rx in self._flag_res:
            if rx.search(text):
                hits.append(rx.pattern)
        return len(hits), hits

    def source_heuristic(self, url: Optional[str]) -> Tuple[float, str]:
        if not url:
            return 0.5, "no_source"
        host = urlparse(url).netloc.lower()
        parts = host.split(':')[0].split('.')
        base = '.'.join(parts[-2:]) if len(parts) >= 2 else host
        if host in ALLOW_DOMAINS or base in ALLOW_DOMAINS:
            return 0.05, f"allow:{host}"
        if host in DENY_DOMAINS or base in DENY_DOMAINS:
            return 0.85, f"deny:{host}"
        if host in MEDIUM_DOMAINS or base in MEDIUM_DOMAINS:
            return 0.7, f"medium:{host}"
        return 0.6, f"unknown:{host}"

    def scale_check(self, text: str, context_keywords: Optional[List[str]] = None) -> Tuple[float, Dict]:
        context_keywords = context_keywords or []
        sizes_km = []
        for m in _NUMBER_UNIT.finditer(text):
            sizes_km.append(self._km_from_match(m.group('num'), m.group('unit')))
        if not sizes_km:
            return 0.0, {"sizes_km": []}
        max_km = max(sizes_km)
        extraordinary_context = any(k in text.lower() for k in context_keywords)
        ratio = max_km / max(self.extraordinary_km, 1.0)
        base = min(ratio, 1.0)
        if extraordinary_context:
            base = min(1.0, base * 1.25)
        return base, {"sizes_km": sizes_km, "max_km": max_km, "extraordinary_context": extraordinary_context}

    def score(self, text: str, url: Optional[str] = None,
              context_keywords: Optional[List[str]] = None) -> HoaxFilterResult:
        rf_count, rf_hits = self.language_red_flags(text)
        rf_score = min(rf_count / 4.0, 1.0)
        src_risk, src_note = self.source_heuristic(url)
        scale_risk, scale_notes = self.scale_check(text, context_keywords=context_keywords)
        combined = (self.red_flag_weight * rf_score
                    + self.source_weight * src_risk
                    + self.scale_weight * scale_risk)
        return HoaxFilterResult(
            red_flag_hits=rf_count,
            source_score=src_risk,
            scale_score=scale_risk,
            combined=min(combined, 1.0),
            notes={
                "red_flag_patterns": rf_hits,
                "source": src_note,
                **scale_notes
            }
        )
