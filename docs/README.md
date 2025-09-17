# Sentinal

A multi-timescale AI guardrails system with hoax/misinformation filter, CLI, and robust test suite.

## Features
- Hoax/misinformation detection
- Allow/deny lists
- CLI for scanning text and sources
- SQLite-based memory and FTS5 search
- NLTK, rapidfuzz, numpy, filelock integration
- Extensible signal engine

## Quickstart
```bash
pip install -e .
hoax-scan --help
```

## Project Structure
- `sentinal/` — Source code and CLI
- `sentinal/tests/` — Unit tests
- `docs/` — Documentation
- `.github/workflows/ci.yml` — CI/CD workflow

## Requirements
- Python 3.8+
- See `setup.py` and `requirements.txt`

## License
MIT