# Project SENTINAL
# SENTINAL

A multi-timescale AI guardrails system for misinformation detection and filtering, now with:
- Advanced LLM ensemble integration (DistilBERT, RoBERTa, DistilGPT2, GPT2)
- Sentiment analysis: majority vote, average score, all model outputs
- Chat: show all LLM responses
- Web UI with chat, file upload, and admin/user authentication
- Admin dashboard: user management, logs, documentation links

## Features
- Hoax/misinformation detection
- Allow/deny lists
- CLI for scanning text and sources
- Web UI with chat, file upload, and ensemble LLMs
- Open-source LLMs for natural chat
- Sentiment analysis with multiple models
- SQLite-based memory and FTS5 search
- NLTK, rapidfuzz, numpy, filelock integration
- Extensible signal engine
- Comprehensive System Diagnostics
  - Real-time resource monitoring (CPU, memory, disk)
  - Component health tracking
  - Security metrics and audit trail
  - Interactive diagnostic dashboard
  - Pre-flight system checks

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
```

