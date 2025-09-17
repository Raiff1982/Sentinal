# Project SENTINAL

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.16853922.svg)](https://doi.org/10.5281/zenodo.16853922)

**AEGIS Sentinel** is the governance & safety cortex for multi‑agent systems:
- **Shield (pre‑attack)**: rate limiting, proof‑of‑work, injection/Unicode anomaly guard, canaries & honeypots.
- **Challenge Gate (pre‑flight)**: runs a ChallengeBank on the *real* council; fail ⇒ reject.
- **Meta‑Arbitration (in‑flight)**: `BLOCK / PROCEED_WITH_CAUTION / PROCEED` from fused risk, stress, conflict, timescale.
- **Tamper‑Evident Ledger (post‑flight)**: HMAC‑signed JSONL with full verify/rollback.
- **Nexus Signal Engine**: entropy‑aware, arousal‑weighted, time‑decaying memory.

## Install
```bash
pip install -r requirements.txt
```

## Quickstart
```bash
python -m pytest -q tests
python aegisctl.py challenges
python aegisctl.py serve --port 8787
```

## Cite
```bibtex
@software{harrison2025sentinal,
  author = {Harrison, Jonathan},
  title  = {Project SENTINAL},
  year   = {2025},
  doi    = {10.5281/zenodo.16853922}
}
```

