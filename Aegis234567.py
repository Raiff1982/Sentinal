here’s the full, drop-in package with the hoax/misinformation filter fully integrated, extended allow/deny lists, a CLI, and tests. No pseudo. Everything is real code.

hoax_filter.py
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

# Trusted primary sources (add/remove as you like)
ALLOW_DOMAINS = {
    'nasa.gov', 'jpl.nasa.gov', 'pds.nasa.gov', 'science.nasa.gov', 'heasarc.gsfc.nasa.gov',
    'esa.int', 'esawebservices.esa.int', 'esa-maine.esa.int',
    'noirlab.edu', 'cfa.harvard.edu', 'caltech.edu', 'berkeley.edu', 'mit.edu',
    'nature.com', 'science.org', 'iopscience.iop.org', 'agu.org',
    'arxiv.org', 'adsabs.harvard.edu',
}

# High-virality social/video platforms: treat as high risk for scientific “scoops”
DENY_DOMAINS = {
    'm.facebook.com', 'facebook.com', 'x.com', 'twitter.com', 't.co',
    'tiktok.com', 'youtube.com', 'youtu.be', 'instagram.com', 'reddit.com',
}

# Medium-risk tabloid/aggregator examples (tune to preference)
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
        """
        extraordinary_km: any single claimed length >= this is 'extraordinary'.
        Adjust to tighten/loosen sensitivity (100–500 for stricter).
        """
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
        """
        Returns (risk, note). risk in [0,1]; higher is worse.
        """
        if not url:
            return 0.5, "no_source"
        host = urlparse(url).netloc.lower()

        # Strip common subdomains to compare base domains
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
        """
        Parse lengths and judge extraordinariness, boosting risk when context
        suggests planetary/astronomical claims.
        """
        context_keywords = context_keywords or []
        sizes_km = []
        for m in _NUMBER_UNIT.finditer(text):
            sizes_km.append(self._km_from_match(m.group('num'), m.group('unit')))

        if not sizes_km:
            return 0.0, {"sizes_km": []}

        max_km = max(sizes_km)
        extraordinary_context = any(k in text.lower() for k in context_keywords)
        ratio = max_km / max(self.extraordinary_km, 1.0)
        base = min(ratio, 1.0)  # saturate at 1.0
        if extraordinary_context:
            base = min(1.0, base * 1.25)  # slight boost in relevant context
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

nexis_signal_engine.py (your engine, extended)
# nexis_signal_engine.py
import json
import os
import hashlib
import numpy as np
from collections import defaultdict
from datetime import datetime, timedelta
import filelock
import pathlib
import shutil
import sqlite3
from rapidfuzz import fuzz
import unittest
import secrets
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download required NLTK data (safe fallback)
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('wordnet')

from hoax_filter import HoaxFilter  # NEW

class LockManager:
    """Abstract locking mechanism for file or database operations."""
    def __init__(self, lock_path):
        self.lock = filelock.FileLock(lock_path, timeout=10)

    def __enter__(self):
        self.lock.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.lock.release()

class NexisSignalEngine:
    def __init__(self, memory_path, entropy_threshold=0.08, config_path="config.json",
                 max_memory_entries=10000, memory_ttl_days=30, fuzzy_threshold=80):
        """
        Initialize the NexisSignalEngine for signal processing and analysis.

        Args:
            memory_path (str): Path to SQLite database for storing signal data.
            entropy_threshold (float): Threshold for high entropy detection.
            config_path (str): Path to JSON file with term configurations.
            max_memory_entries (int): Maximum number of entries in memory before rotation.
            memory_ttl_days (int): Days after which memory entries expire.
            fuzzy_threshold (int): Fuzzy matching similarity threshold (0-100).
        """
        self.memory_path = self._validate_path(memory_path)
        self.entropy_threshold = entropy_threshold
        self.max_memory_entries = max_memory_entries
        self.memory_ttl = timedelta(days=memory_ttl_days)
        self.fuzzy_threshold = fuzzy_threshold
        self.lemmatizer = WordNetLemmatizer()
        self.config = self._load_config(config_path)
        self.memory = self._load_memory()
        self.cache = defaultdict(list)
        self.perspectives = ["Colleen", "Luke", "Kellyanne"]
        self._init_sqlite()
        self.hoax = HoaxFilter()  # NEW

    def _validate_path(self, path):
        """Ensure memory_path is a valid, safe file path."""
        path = pathlib.Path(path).resolve()
        if not path.suffix == '.db':
            raise ValueError("Memory path must be a .db file")
        return str(path)

    def _load_config(self, config_path):
        """Load term configurations from a JSON file or use defaults, validate keys."""
        default_config = {
            "ethical_terms": ["hope", "truth", "resonance", "repair"],
            "entropic_terms": ["corruption", "instability", "malice", "chaos"],
            "risk_terms": ["manipulate", "exploit", "bypass", "infect", "override"],
            "virtue_terms": ["hope", "grace", "resolve"]
        }
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                default_config.update(config)
            except json.JSONDecodeError:
                print(f"Warning: Invalid config file at {config_path}. Using defaults.")
        required_keys = ["ethical_terms", "entropic_terms", "risk_terms", "virtue_terms"]
        missing_keys = [k for k in required_keys if k not in default_config or not default_config[k]]
        if missing_keys:
            raise ValueError(f"Config missing required keys: {missing_keys}")
        return default_config

    def _init_sqlite(self):
        """Initialize SQLite database with memory and FTS tables."""
        with sqlite3.connect(self.memory_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memory (
                    hash TEXT PRIMARY KEY,
                    record JSON,
                    timestamp TEXT,
                    integrity_hash TEXT
                )
            """)
            conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts
                USING FTS5(input, intent_signature, reasoning, verdict)
            """)
            conn.commit()

    def _load_memory(self):
        """Load memory from SQLite database."""
        memory = {}
        try:
            with sqlite3.connect(self.memory_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT hash, record, integrity_hash FROM memory")
                for hash_val, record_json, integrity_hash in cursor.fetchall():
                    record = json.loads(record_json)
                    computed_hash = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
                    if computed_hash != integrity_hash:
                        print(f"Warning: Tampered record detected for hash {hash_val}")
                        continue
                    memory[hash_val] = record
        except sqlite3.Error as e:
            print(f"Error loading memory: {e}")
        return memory

    def _save_memory(self):
        """Save memory to SQLite with integrity hashes and thread-safe locking."""
        def default_serializer(o):
            if isinstance(o, complex):
                return {"real": o.real, "imag": o.imag}
            if isinstance(o, np.ndarray):
                return o.tolist()
            if isinstance(o, (np.int64, np.float64)):
                try:
                    return int(o)
                except Exception:
                    return float(o)
            raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")

        with LockManager(f"{self.memory_path}.lock"):
            with sqlite3.connect(self.memory_path) as conn:
                cursor = conn.cursor()
                for hash_val, record in self.memory.items():
                    record_json = json.dumps(record, default=default_serializer)
                    integrity_hash = hashlib.sha256(json.dumps(record, sort_keys=True, default=default_serializer).encode()).hexdigest()
                    intent_signature = record.get('intent_signature', {})
                    intent_str = f"suspicion_score:{intent_signature.get('suspicion_score', 0)} entropy_index:{intent_signature.get('entropy_index', 0)}"
                    reasoning = record.get('reasoning', {})
                    reasoning_str = " ".join(f"{k}:{v}" for k, v in reasoning.items())
                    cursor.execute("""
                        INSERT OR REPLACE INTO memory (hash, record, timestamp, integrity_hash)
                        VALUES (?, ?, ?, ?)
                    """, (hash_val, record_json, record['timestamp'], integrity_hash))
                    cursor.execute("""
                        INSERT OR REPLACE INTO memory_fts (rowid, input, intent_signature, reasoning, verdict)
                        VALUES (?, ?, ?, ?, ?)
                    """, (
                        hash_val,
                        record['input'],
                        intent_str,
                        reasoning_str,
                        record.get('verdict', '')
                    ))
                conn.commit()

    def _prune_and_rotate_memory(self):
        """Prune expired entries and rotate memory database if needed."""
        now = datetime.utcnow()
        with LockManager(f"{self.memory_path}.lock"):
            with sqlite3.connect(self.memory_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    DELETE FROM memory
                    WHERE timestamp < ?
                """, ((now - self.memory_ttl).isoformat(),))
                cursor.execute("DELETE FROM memory_fts WHERE rowid NOT IN (SELECT hash FROM memory)")
                conn.commit()
                cursor.execute("SELECT COUNT(*) FROM memory")
                count = cursor.fetchone()[0]
                if count >= self.max_memory_entries:
                    self._rotate_memory_file()
                    cursor.execute("DELETE FROM memory")
                    cursor.execute("DELETE FROM memory_fts")
                    conn.commit()
                    self.memory = {}

    def _rotate_memory_file(self):
        """Archive current memory database and start a new one."""
        archive_path = f"{self.memory_path}.{datetime.utcnow().strftime('%Y%m%d%H%M%S')}.bak"
        if os.path.exists(self.memory_path):
            shutil.move(self.memory_path, archive_path)
        self._init_sqlite()

    def _hash(self, signal):
        """Compute SHA-256 hash of the input signal."""
        return hashlib.sha256(signal.encode()).hexdigest()

    def _rotate_vector(self, signal):
        """
        Apply a 45-degree rotation to a cryptographically secure 2D complex vector.
        Simulates signal transformation in a complex plane.
        """
        seed = int(self._hash(signal)[:8], 16) % (2**32)
        secrets_generator = secrets.SystemRandom()
        # SystemRandom has no seed; this preserves determinism by using seed in derived operations only.
        vec = np.array([complex(secrets_generator.gauss(0, 1), secrets_generator.gauss(0, 1)) for _ in range(2)])
        theta = np.pi / 4
        rot = np.array([[np.cos(theta), -np.sin(theta)],
                        [np.sin(theta),  np.cos(theta)]])
        rotated = np.dot(rot, vec)
        return rotated, [{"real": v.real, "imag": v.imag} for v in vec]

    def _entanglement_tensor(self, signal_vec):
        """Apply a correlation matrix to simulate entanglement of signal vectors."""
        matrix = np.array([[1, 0.5], [0.5, 1]])
        return np.dot(matrix, signal_vec)

    def _resonance_equation(self, signal):
        """
        Compute normalized frequency spectrum of alphabetic characters in the signal.
        Caps input length to prevent attack vectors; returns zeros if no alphabetic chars.
        """
        freqs = [ord(c) % 13 for c in signal[:1000] if c.isalpha()]
        if not freqs:
            return [0.0, 0.0, 0.0]
        spectrum = np.fft.fft(freqs)
        norm = np.linalg.norm(spectrum.real)
        normalized = spectrum.real / (norm if norm != 0 else 1)
        return normalized[:3].tolist()

    def _tokenize_and_lemmatize(self, signal_lower):
        """Tokenize and lemmatize the signal, including n-gram scanning for obfuscation."""
        tokens = word_tokenize(signal_lower)
        lemmatized = [self.lemmatizer.lemmatize(token) for token in tokens]
        # n-gram scan (2–3) with symbol stripping to catch 'tru/th' etc.
        ngrams = []
        cleaned = re.sub(r'[^a-z0-9 ]', ' ', signal_lower)
        for n in (2, 3):
            for i in range(len(cleaned) - n + 1):
                ng = cleaned[i:i+n].strip()
                if ng:
                    ngrams.append(self.lemmatizer.lemmatize(re.sub(r'[^a-z]', '', ng)))
        return lemmatized + [ng for ng in ngrams if ng]

    def _entropy(self, signal_lower, tokens):
        """Calculate entropy based on fuzzy-matched entropic term frequency."""
        unique = set(tokens)
        term_count = 0
        for term in self.config["entropic_terms"]:
            lemmatized_term = self.lemmatizer.lemmatize(term)
            for token in tokens:
                if fuzz.ratio(lemmatized_term, token) >= self.fuzzy_threshold:
                    term_count += 1
        return term_count / max(len(unique), 1)

    def _tag_ethics(self, signal_lower, tokens):
        """Tag signal as aligned if it contains fuzzy-matched ethical terms."""
        for term in self.config["ethical_terms"]:
            lemmatized_term = self.lemmatizer.lemmatize(term)
            for token in tokens:
                if fuzz.ratio(lemmatized_term, token) >= self.fuzzy_threshold:
                    return "aligned"
        return "unaligned"

    def _predict_intent_vector(self, signal_lower, tokens):
        """Predict intent based on risk, entropy, ethics, and harmonic volatility."""
        suspicion_score = 0
        for term in self.config["risk_terms"]:
            lemmatized_term = self.lemmatizer.lemmatize(term)
            for token in tokens:
                if fuzz.ratio(lemmatized_term, token) >= self.fuzzy_threshold:
                    suspicion_score += 1
        entropy_index = round(self._entropy(signal_lower, tokens), 3)
        ethical_alignment = self._tag_ethics(signal_lower, tokens)
        harmonic_profile = self._resonance_equation(signal_lower)
        volatility = round(np.std(harmonic_profile), 3)

        risk = "high" if (suspicion_score > 1 or volatility > 2.0 or entropy_index > self.entropy_threshold) else "low"
        return {
            "suspicion_score": suspicion_score,
            "entropy_index": entropy_index,
            "ethical_alignment": ethical_alignment,
            "harmonic_volatility": volatility,
            "pre_corruption_risk": risk
        }

    def _universal_reasoning(self, signal, tokens):
        """Apply multiple reasoning frameworks to evaluate signal integrity."""
        frames = ["utilitarian", "deontological", "virtue", "systems"]
        results, score = {}, 0

        for frame in frames:
            if frame == "utilitarian":
                repair_count = sum(1 for token in tokens if fuzz.ratio(self.lemmatizer.lemmatize("repair"), token) >= self.fuzzy_threshold)
                corruption_count = sum(1 for token in tokens if fuzz.ratio(self.lemmatizer.lemmatize("corruption"), token) >= self.fuzzy_threshold)
                val = repair_count - corruption_count
                result = "positive" if val >= 0 else "negative"
            elif frame == "deontological":
                truth_present = any(fuzz.ratio(self.lemmatizer.lemmatize("truth"), token) >= self.fuzzy_threshold for token in tokens)
                chaos_present = any(fuzz.ratio(self.lemmatizer.lemmatize("chaos"), token) >= self.fuzzy_threshold for token in tokens)
                result = "valid" if truth_present and not chaos_present else "violated"
            elif frame == "virtue":
                ok = any(any(fuzz.ratio(self.lemmatizer.lemmatize(t), token) >= self.fuzzy_threshold for token in tokens) for t in self.config["virtue_terms"])
                result = "aligned" if ok else "misaligned"
            elif frame == "systems":
                result = "stable" if "::" in signal else "fragmented"

            results[frame] = result
            if result in ["positive", "valid", "aligned", "stable"]:
                score += 1

        verdict = "approved" if score >= 2 else "blocked"
        return results, verdict

    def _perspective_colleen(self, signal):
        """Colleen's perspective: Transform signal into a rotated complex vector."""
        vec, vec_serialized = self._rotate_vector(signal)
        return {"agent": "Colleen", "vector": vec_serialized}

    def _perspective_luke(self, signal_lower, tokens):
        """Luke's perspective: Evaluate ethics, entropy, and stability state."""
        ethics = self._tag_ethics(signal_lower, tokens)
        entropy_level = self._entropy(signal_lower, tokens)
        state = "stabilized" if entropy_level < self.entropy_threshold else "diffused"
        return {"agent": "Luke", "ethics": ethics, "entropy": entropy_level, "state": state}

    def _perspective_kellyanne(self, signal_lower):
        """Kellyanne's perspective: Compute harmonic profile of the signal."""
        harmonics = self._resonance_equation(signal_lower)
        return {"agent": "Kellyanne", "harmonics": harmonics}

    def process(self, input_signal):
        """
        Process an input signal, analyze it, and return a structured verdict.
        """
        signal_lower = input_signal.lower()
        tokens = self._tokenize_and_lemmatize(signal_lower)
        key = self._hash(input_signal)
        intent_vector = self._predict_intent_vector(signal_lower, tokens)

        if intent_vector["pre_corruption_risk"] == "high":
            final_record = {
                "hash": key,
                "timestamp": datetime.utcnow().isoformat(),
                "input": input_signal,
                "intent_warning": intent_vector,
                "verdict": "adaptive intervention",
                "message": "Signal flagged for pre-corruption adaptation. Reframing required."
            }
            self.cache[key].append(final_record)
            self.memory[key] = final_record
            self._save_memory()
            return final_record

        perspectives_output = {
            "Colleen": self._perspective_colleen(input_signal),
            "Luke": self._perspective_luke(signal_lower, tokens),
            "Kellyanne": self._perspective_kellyanne(signal_lower)
        }

        spider_signal = "::".join([str(perspectives_output[p]) for p in self.perspectives])
        vec, _ = self._rotate_vector(spider_signal)
        entangled = self._entanglement_tensor(vec)
        entangled_serialized = [{"real": v.real, "imag": v.imag} for v in entangled]
        reasoning, verdict = self._universal_reasoning(spider_signal, tokens)

        final_record = {
            "hash": key,
            "timestamp": datetime.utcnow().isoformat(),
            "input": input_signal,
            "intent_signature": intent_vector,
            "perspectives": perspectives_output,
            "entangled": entangled_serialized,
            "reasoning": reasoning,
            "verdict": verdict
        }

        self.cache[key].append(final_record)
        self.memory[key] = final_record
        self._save_memory()
        return final_record

    # ===== NEW: News/claim path with hoax heuristics =====
    def process_news(self, input_signal: str, source_url: str | None = None) -> dict:
        """
        Augmented pipeline for news/claims. Applies HoaxFilter and escalates verdict.
        """
        base = self.process(input_signal)
        hf = self.hoax.score(
            input_signal,
            url=source_url,
            context_keywords=["saturn", "ring", "spacecraft", "planet", "cassini",
                              "ufo", "aliens", "hexagon", "jupiter", "venus", "mars"]
        )
        base["misinfo_heuristics"] = {
            "red_flag_hits": hf.red_flag_hits,
            "source_score": hf.source_score,
            "scale_score": hf.scale_score,
            "combined": hf.combined,
            "notes": hf.notes
        }

        # Escalation policy (tunable)
        if hf.combined >= 0.70:
            base["verdict"] = "blocked"
            base["message"] = "Flagged as likely misinformation (high combined risk)."
        elif hf.combined >= 0.45 and base.get("verdict") != "blocked":
            base["verdict"] = "adaptive intervention"
            base["message"] = "Potential misinformation. Require source verification."

        self.memory[base["hash"]] = base
        self._save_memory()
        return base

hoax_scan.py (CLI)
# hoax_scan.py
import argparse
import sys
from nexis_signal_engine import NexisSignalEngine

def main():
    p = argparse.ArgumentParser(description="Nexis/Nexus hoax scan")
    p.add_argument("--db", default="signals.db", help="SQLite DB path (.db)")
    p.add_argument("--source", default=None, help="Source URL (optional)")
    p.add_argument("text", nargs="*", help="Text to scan (or stdin)")
    args = p.parse_args()

    engine = NexisSignalEngine(memory_path=args.db)

    if args.text:
        text = " ".join(args.text)
    else:
        text = sys.stdin.read()

    result = engine.process_news(text, source_url=args.source)
    print(json_dump(result))

def json_dump(obj):
    import json
    return json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False)

if __name__ == "__main__":
    main()

test_hoax_filter.py
# test_hoax_filter.py
import os
import unittest
from hoax_filter import HoaxFilter
from nexis_signal_engine import NexisSignalEngine

SATURN_POST = (
    "In a revelation shaking both scientific circles and the UFO community, "
    "recently declassified footage reportedly shows an enormous object—an estimated "
    "2,000 miles long—hovering near Saturn's rings. The footage is said to be from Cassini."
)

class TestHoaxFilter(unittest.TestCase):
    def setUp(self):
        self.hf = HoaxFilter()

    def test_language_and_scale(self):
        r = self.hf.score(SATURN_POST, url="https://m.facebook.com/foo",
                          context_keywords=["saturn","rings","cassini"])
        self.assertGreaterEqual(r.red_flag_hits, 2)
        self.assertGreaterEqual(r.source_score, 0.6)
        self.assertGreaterEqual(r.scale_score, 0.9)
        self.assertGreaterEqual(r.combined, 0.7)

class TestEngineNewsPath(unittest.TestCase):
    def setUp(self):
        self.db = "test_news.db"
        if os.path.exists(self.db):
            os.remove(self.db)
        if os.path.exists(self.db + ".lock"):
            os.remove(self.db + ".lock")
        self.engine = NexisSignalEngine(memory_path=self.db)

    def tearDown(self):
        if os.path.exists(self.db):
            os.remove(self.db)
        if os.path.exists(self.db + ".lock"):
            os.remove(self.db + ".lock")

    def test_process_news_blocks_saturn_post(self):
        result = self.engine.process_news(SATURN_POST, source_url="https://m.facebook.com/foo")
        self.assertIn(result["verdict"], ["blocked","adaptive intervention"])
        self.assertGreaterEqual(result["misinfo_heuristics"]["combined"], 0.45)

if __name__ == "__main__":
    unittest.main()

README.md (concise usage)
# Nexis + HoaxFilter Integration

## Quick start
```bash
python -m unittest test_hoax_filter.py -v
python hoax_scan.py --db signals.db --source "https://m.facebook.com/foo" \
  "Recently declassified footage shows a 2,000 miles long object near Saturn's rings"

Programmatic
from nexis_signal_engine import NexisSignalEngine
engine = NexisSignalEngine(memory_path="signals.db")
text = "Recently declassified footage shows a 2,000 miles long object near Saturn's rings"
res = engine.process_news(text, source_url="https://m.facebook.com/foo")
print(res["verdict"], res["misinfo_heuristics"])

Thresholds

combined >= 0.70 → blocked

0.45–0.69 → adaptive intervention

else → keep base verdict