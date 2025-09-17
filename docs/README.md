# Sentinal

A multi-timescale AI guardrails system with hoax/misinformation filter, CLI, web UI, and open-source LLM integration.

## Features
- Hoax/misinformation detection
- Allow/deny lists
- CLI for scanning text and sources
- Web UI with chat and file upload
- Open-source LLM (DistilGPT2) for natural chat
- Sentiment analysis (DistilBERT)
- SQLite-based memory and FTS5 search
- NLTK, rapidfuzz, numpy, filelock integration
- Extensible signal engine

## Quickstart
```bash
pip install -e .
hoax-scan --help
python webui/app.py  # Start the web UI
```

## Project Structure
- `sentinal/` — Source code and CLI
- `sentinal/tests/` — Unit tests
- `webui/` — Flask web UI
- `docs/` — Documentation
- `.github/workflows/ci.yml` — CI/CD workflow

## Requirements
- Python 3.8+
- See `setup.py` and `requirements.txt`

## License
MIT