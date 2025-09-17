# test_hoax_filter.py
import os
import unittest
import nltk
from .hoax_filter import HoaxFilter
from .nexis_signal_engine import NexisSignalEngine

# Ensure NLTK data is present
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/wordnet')
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('punkt_tab')

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
        # Attempt to close DB connections before file removal
        import sqlite3
        try:
            conn = sqlite3.connect(self.db)
            conn.close()
        except Exception:
            pass
        if os.path.exists(self.db):
            try:
                os.remove(self.db)
            except Exception:
                pass
        if os.path.exists(self.db + ".lock"):
            try:
                os.remove(self.db + ".lock")
            except Exception:
                pass

    def test_process_news_blocks_saturn_post(self):
        result = self.engine.process_news(SATURN_POST, source_url="https://m.facebook.com/foo")
        self.assertIn(result["verdict"], ["blocked","adaptive intervention"])
        self.assertGreaterEqual(result["misinfo_heuristics"]["combined"], 0.45)

if __name__ == "__main__":
    unittest.main()
