
# Sentinal

A multi-timescale AI guardrails system with hoax/misinformation filter, CLI, web UI, open-source LLM ensemble integration, and advanced admin features.

## Features
- Hoax/misinformation detection
- Allow/deny lists
- CLI for scanning text and sources
- Web UI with chat, file upload, and model selection
- Open-source LLMs (DistilGPT2, GPT2, DistilBERT, RoBERTa) for chat and sentiment analysis
- Ensemble logic: majority vote, average score, all model outputs
- SQLite-based memory and FTS5 search
- NLTK, rapidfuzz, numpy, filelock integration
- Extensible signal engine
- **Admin Dashboard:** Analytics, batch scan/chat, feedback labeling, export, retrain/deploy

## Quickstart
```bash
pip install -e .
hoax-scan --help
# Run Flask web UI (from workspace root):
& ".venv\Scripts\python.exe" webui/app.py
# Or, with Flask CLI:
$env:FLASK_APP = "webui/app.py"
$env:FLASK_ENV = "development"
flask run
```

## Admin Features
- Analytics dashboard: visualize model usage and compare scan/chat stats
- Batch scan & chat: upload multiple texts for batch analysis
- Feedback & labeling: label scan/chat results for model improvement
- Export: download interactions, feedback, and analytics data
- Retrain & deploy: one-click retraining and deployment of models

## Troubleshooting
- If you see `ModuleNotFoundError: No module named 'sentinal'`, ensure you run from the workspace root or use the provided run commands above.
- For more docs, see the admin dashboard links to PDF and Markdown documentation.

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