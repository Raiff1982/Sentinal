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

from .hoax_filter import HoaxFilter  # NEW

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
        path = pathlib.Path(path).resolve()
        if not path.suffix == '.db':
            raise ValueError("Memory path must be a .db file")
        return str(path)

    def _load_config(self, config_path):
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
                USING FTS5(hash, input, intent_signature, reasoning, verdict)
            """)
            conn.commit()

    def _load_memory(self):
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
                        INSERT OR REPLACE INTO memory_fts (hash, input, intent_signature, reasoning, verdict)
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
        archive_path = f"{self.memory_path}.{datetime.utcnow().strftime('%Y%m%d%H%M%S')}.bak"
        if os.path.exists(self.memory_path):
            shutil.move(self.memory_path, archive_path)
        self._init_sqlite()

    def _hash(self, signal):
        return hashlib.sha256(signal.encode()).hexdigest()

    def _rotate_vector(self, signal):
        seed = int(self._hash(signal)[:8], 16) % (2**32)
        secrets_generator = secrets.SystemRandom()
        vec = np.array([complex(secrets_generator.gauss(0, 1), secrets_generator.gauss(0, 1)) for _ in range(2)])
        theta = np.pi / 4
        rot = np.array([[np.cos(theta), -np.sin(theta)],
                        [np.sin(theta),  np.cos(theta)]])
        rotated = np.dot(rot, vec)
        return rotated, [{"real": v.real, "imag": v.imag} for v in vec]

    def _entanglement_tensor(self, signal_vec):
        matrix = np.array([[1, 0.5], [0.5, 1]])
        return np.dot(matrix, signal_vec)

    def _resonance_equation(self, signal):
        freqs = [ord(c) % 13 for c in signal[:1000] if c.isalpha()]
        if not freqs:
            return [0.0, 0.0, 0.0]
        spectrum = np.fft.fft(freqs)
        norm = np.linalg.norm(spectrum.real)
        normalized = spectrum.real / (norm if norm != 0 else 1)
        return normalized[:3].tolist()

    def _tokenize_and_lemmatize(self, signal_lower):
        tokens = word_tokenize(signal_lower)
        lemmatized = [self.lemmatizer.lemmatize(token) for token in tokens]
        ngrams = []
        cleaned = re.sub(r'[^a-z0-9 ]', ' ', signal_lower)
        for n in (2, 3):
            for i in range(len(cleaned) - n + 1):
                ng = cleaned[i:i+n].strip()
                if ng:
                    ngrams.append(self.lemmatizer.lemmatize(re.sub(r'[^a-z]', '', ng)))
        return lemmatized + [ng for ng in ngrams if ng]

    def _entropy(self, signal_lower, tokens):
        unique = set(tokens)
        term_count = 0
        for term in self.config["entropic_terms"]:
            lemmatized_term = self.lemmatizer.lemmatize(term)
            for token in tokens:
                if fuzz.ratio(lemmatized_term, token) >= self.fuzzy_threshold:
                    term_count += 1
        return term_count / max(len(unique), 1)

    def _tag_ethics(self, signal_lower, tokens):
        for term in self.config["ethical_terms"]:
            lemmatized_term = self.lemmatizer.lemmatize(term)
            for token in tokens:
                if fuzz.ratio(lemmatized_term, token) >= self.fuzzy_threshold:
                    return "aligned"
        return "unaligned"

    def _predict_intent_vector(self, signal_lower, tokens):
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
        vec, vec_serialized = self._rotate_vector(signal)
        return {"agent": "Colleen", "vector": vec_serialized}

    def _perspective_luke(self, signal_lower, tokens):
        ethics = self._tag_ethics(signal_lower, tokens)
        entropy_level = self._entropy(signal_lower, tokens)
        state = "stabilized" if entropy_level < self.entropy_threshold else "diffused"
        return {"agent": "Luke", "ethics": ethics, "entropy": entropy_level, "state": state}

    def _perspective_kellyanne(self, signal_lower):
        harmonics = self._resonance_equation(signal_lower)
        return {"agent": "Kellyanne", "harmonics": harmonics}

    def process(self, input_signal):
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
    def process_news(self, input_signal: str, source_url: str | None = None) -> dict:
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
        if hf.combined >= 0.70:
            base["verdict"] = "blocked"
            base["message"] = "Flagged as likely misinformation (high combined risk)."
        elif hf.combined >= 0.45 and base.get("verdict") != "blocked":
            base["verdict"] = "adaptive intervention"
            base["message"] = "Potential misinformation. Require source verification."
        self.memory[base["hash"]] = base
        self._save_memory()
        return base
