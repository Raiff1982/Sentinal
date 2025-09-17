# Nexis + HoaxFilter Integration

## Quick start
```bash
python -m unittest test_hoax_filter.py -v
python hoax_scan.py --db signals.db --source "https://m.facebook.com/foo" \
  "Recently declassified footage shows a 2,000 miles long object near Saturn's rings"
```

## Programmatic
```python
from nexis_signal_engine import NexisSignalEngine
engine = NexisSignalEngine(memory_path="signals.db")
text = "Recently declassified footage shows a 2,000 miles long object near Saturn's rings"
res = engine.process_news(text, source_url="https://m.facebook.com/foo")
print(res["verdict"], res["misinfo_heuristics"])
```

## Thresholds
- combined >= 0.70 → blocked
- 0.45–0.69 → adaptive intervention
- else → keep base verdict
