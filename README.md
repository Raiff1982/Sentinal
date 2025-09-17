

# Project SENTINAL

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.16853922.svg)](https://doi.org/10.5281/zenodo.16853922)

**AEGIS Sentinel** is a multi-timescale AI guardrails system for multi-agent governance and safety. Now with advanced LLM ensemble integration and role-based web UI features.

## Key Features
- **Shield (pre-attack):** Rate limiting, proof-of-work, injection/Unicode anomaly guard, canaries & honeypots.
- **Challenge Gate (pre-flight):** ChallengeBank for council validation; failed challenges are rejected.
- **Meta-Arbitration (in-flight):** Fused risk, stress, conflict, and timescale analysis for `BLOCK / PROCEED_WITH_CAUTION / PROCEED` decisions.
- **Tamper-Evident Ledger (post-flight):** HMAC-signed JSONL with full verification and rollback.
- **Nexus Signal Engine:** Entropy-aware, arousal-weighted, time-decaying memory.
- **Advanced LLM Ensemble:** Multiple open-source models for sentiment and chat, with majority vote and average scoring.
- **Role-Based Web UI:** Admin/user login, admin-only features, and documentation panels.

## Installation

```bash
pip install -r requirements.txt
```

## Quickstart

```bash
python -m pytest -q tests
python sentinal/aegisctl.py challenges
python sentinal/aegisctl.py serve --port 8787
python webui/app.py  # Start the web UI
```

## Web UI
- Scan text or upload files for misinformation detection
- Chat with ensemble LLMs (DistilGPT2, GPT2, etc.)
- Admin panel for user management, logs, and documentation
- Results show all model outputs and ensemble verdicts

## Advanced LLM Integration
- Sentiment analysis uses multiple models (DistilBERT, RoBERTa, etc.)
- Chat uses multiple LLMs (DistilGPT2, GPT2, etc.)
- Ensemble logic: majority label, average score, all LLM responses
- Easily extendable in `sentinal/ai_base.py`

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
