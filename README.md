
# Project SENTINAL

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.16853922.svg)](https://doi.org/10.5281/zenodo.16853922)

**AEGIS Sentinel** is a multi-timescale AI guardrails system for multi-agent governance and safety. It provides:

- **Shield (pre-attack):** Rate limiting, proof-of-work, injection/Unicode anomaly guard, canaries & honeypots.
- **Challenge Gate (pre-flight):** ChallengeBank for council validation; failed challenges are rejected.
- **Meta-Arbitration (in-flight):** Fused risk, stress, conflict, and timescale analysis for `BLOCK / PROCEED_WITH_CAUTION / PROCEED` decisions.
- **Tamper-Evident Ledger (post-flight):** HMAC-signed JSONL with full verification and rollback.
- **Nexus Signal Engine:** Entropy-aware, arousal-weighted, time-decaying memory.

## Installation

```bash
pip install -r requirements.txt
```

## Quickstart

```bash
python -m pytest -q tests
python sentinal/aegisctl.py challenges
python sentinal/aegisctl.py serve --port 8787
```

## Project Structure

```
Sentinal/
│
├── sentinal/                # Main source code package
│   ├── __init__.py
│   ├── aegis1fix.py
│   ├── aegis2fix.py
│   ├── aegis3fix.py
│   ├── aegis4fix.py
│   ├── aegis5fix.py
│   ├── aegisctl.py
│   ├── aegis_evolution.py
│   ├── aegis_explain.py
│   ├── aegis_timescales.py
│   ├── sentinel_config.py
│   ├── sentinel_council.py
│   ├── Sentinel.py
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
