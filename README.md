## Testing & Continuous Integration

Run the full test suite:
```bash
python -m pytest -v tests
```

Run the smoke test only:
```bash
python -m pytest -v tests/test_smoke.py
```

For CI, add these commands to your workflow to ensure all guardrails and council logic are validated before release.
## Tamper-Evident Ledger & Integrity Verification

All system actions are logged to a tamper-evident ledger (JSONL) with HMAC signatures.
To verify ledger integrity locally:

```bash
python verify_ledger.py path/to/ledger.jsonl SECRET_KEY
```

This checks every entry for a valid HMAC signature using your secret key. If any entry fails, you'll see a mismatch warning.

**Best practices:**
- Store your ledger securely and back it up regularly.
- Never share your secret key publicly.
- Use the verify script after any suspicious activity or before audits.



# Project SENTINAL

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.16853922.svg)](https://doi.org/10.5281/zenodo.16853922)

**AEGIS Sentinel** is a multi-timescale AI guardrails system for multi-agent governance and safety. Now with advanced LLM ensemble integration, role-based web UI, analytics dashboard, batch scan/chat, feedback labeling, export, and automated retraining/deployment.

## Key Features
- **Shield (pre-attack):** Rate limiting, proof-of-work, injection/Unicode anomaly guard, canaries & honeypots.
- **Challenge Gate (pre-flight):** ChallengeBank for council validation; failed challenges are rejected.
- **Meta-Arbitration (in-flight):** Fused risk, stress, conflict, and timescale analysis for `BLOCK / PROCEED_WITH_CAUTION / PROCEED` decisions.
- **Tamper-Evident Ledger (post-flight):** HMAC-signed JSONL with full verification and rollback.
- **Nexus Signal Engine:** Entropy-aware, arousal-weighted, time-decaying memory.
- **Advanced LLM Ensemble:** Multiple open-source models for sentiment and chat, with majority vote and average scoring.
- **Role-Based Web UI:** Admin/user login, admin-only features, and documentation panels.
- **Admin Dashboard:** Model analytics, batch scan/chat, feedback labeling, export, retrain/deploy.

## Installation

```bash
pip install -r requirements.txt
```


## Quickstart

```bash
# Install runtime dependencies
pip install -r requirements.txt
# For development and testing
pip install -r requirements-dev.txt

# Run tests (minimal pytest suite)
python -m pytest -q tests

# Run CLI and web UI
python sentinal/aegisctl.py challenges
python sentinal/aegisctl.py serve --port 8787

# Run Flask web UI (from workspace root):
& ".venv\Scripts\python.exe" webui/app.py
# Or, with Flask CLI:
$env:FLASK_APP = "webui/app.py"
$env:FLASK_ENV = "development"
flask run
```

## Web UI Features
- Scan text or upload files for misinformation detection
- Chat with ensemble LLMs (DistilGPT2, GPT2, etc.)
- Admin panel for user management, logs, documentation, analytics, batch scan/chat, feedback labeling, export, retrain/deploy
- Results show all model outputs and ensemble verdicts

## Advanced LLM Integration
- Sentiment analysis uses multiple models (DistilBERT, RoBERTa, etc.)
- Chat uses multiple LLMs (DistilGPT2, GPT2, etc.)
- Ensemble logic: majority label, average score, all LLM responses
- Easily extendable in `sentinal/ai_base.py`

## Admin Features
- **Analytics Dashboard:** Visualize model usage and compare scan/chat stats
- **Batch Scan & Chat:** Upload multiple texts for batch analysis and chat
- **Feedback & Labeling:** Label scan/chat results for model improvement
- **Export:** Download interactions, feedback, and analytics data
- **Retrain & Deploy:** One-click retraining and deployment of models


## Secrets & Environment Variables
All secrets and sensitive config are loaded from environment variables only.

**How to use secrets:**
- Copy `.env.example` to `.env` in your project root.
- Edit `.env` and set values for `SECRET_KEY`, `DATABASE_URL`, `API_TOKEN`, etc.
- Never commit `.env` or secrets to source control. `.gitignore` already excludes `.env`.
- The application loads secrets at runtime using `os.environ` or dotenv.

**Example .env:**
```
SECRET_KEY=changeme
DATABASE_URL=sqlite:///sentinal.db
MODEL_PATH=sentinal/models/
API_TOKEN=your_api_token_here
DEBUG=False
```

**Best practices:**
- Use strong, unique values for `SECRET_KEY` and `API_TOKEN`.
- Store production secrets securely (e.g., environment manager, vault).
- Never hardcode secrets in code or commit them to git.

## Troubleshooting
- If you see `ModuleNotFoundError: No module named 'sentinal'`, ensure you run from the workspace root or use the provided run commands above.
- For more docs, see the admin dashboard links to PDF and Markdown documentation.

## Project Structure
```
Sentinal/
│
├── sentinal/                # Main source code package
│   ├── ai_base.py           # LLM ensemble logic
│   ├── hoax_filter.py       # Heuristic filter
│   ├── ...
│
├── webui/                   # Flask web UI
│   ├── app.py
│   ├── auth.py
│   ├── templates/
│   └── ...
│
├── tests/                   # Test suite
│   └── test_meta_and_guardrails.py
│
├── docs/                    # Documentation
│   ├── README.md
│   ├── SECURITY.md
│   ├── CHANGELOG.md
│   └── ...
│
├── requirements.txt         # Python dependencies
├── setup.py                 # (optional) Packaging script
├── .gitignore
└── other metadata/files
```


## Tamper-Evident Ledger Integrity
To verify the integrity of your HMAC-signed ledger:

```bash
python verify_ledger.py path/to/ledger.jsonl SECRET_KEY
```
If all entries pass, you'll see `Ledger integrity: OK`. Otherwise, any mismatches will be reported by line.

## Documentation
- See admin dashboard for links to PDF and Markdown docs
- Key docs: Project_SENTINAL_Readme.pdf, Aegis_Sentinel_Architecture.pdf, RELEASE_NOTES.md

## License
MIT

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
See the Zenodo record for license details.

## Citation
```bibtex
@software{harrison2025sentinal,
  author = {Harrison, Jonathan},
  title  = {Project SENTINAL},
  year   = {2025},
  doi    = {10.5281/zenodo.16853922}
}
```
