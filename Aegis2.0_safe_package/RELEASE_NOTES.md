# Project SENTINAL v2.0 Release Notes

## 2.1.0 (2025-09-17)

### Overview
This release adds advanced LLM ensemble integration and a role-based web UI with admin/user authentication.

### New Features
- Ensemble support for multiple LLMs (DistilBERT, RoBERTa, DistilGPT2, GPT2)
- Sentiment analysis: majority vote, average score, all model outputs
- Chat: show all LLM responses
- Refactor `ai_base.py` for easy model extension
- Web UI: display ensemble results and all LLM outputs
- Admin/user authentication and role-based feature restriction
- Admin dashboard: user management, logs, documentation links

### Upgrade Notes
- Update requirements.txt to include `transformers` and `torch`
- See README.md and admin dashboard for documentation links

### Security
- No changes to security model; see v2.0 notes

**Date:** 2025-08-18

## Overview
This release delivers the **Aegis Sentinel** with integrated **Nexus Signal Engine**, forming the ethical security cortex of the Aegis architecture.

### Key Features
- **Sentinel Layer**: final decision gate before AI reasoning or action execution.
- **Challenge Bank**: pre-built adversarial tests to detect unsafe/malicious behavior.
- **Agent Council**: multi-perspective evaluators (virtue ethics, risk, temporal coherence).
- **Meta-Judge Arbitration**: resolves agent disagreements via weighted metrics.
- **Audit & Signing**: every decision is logged and explainable.

### Nexus Signal Engine (new)
- Entropy-weighted memory retention
- Emotional-context scoring
- Temporal decay with arousal modulation
- Semantic search & retrieval
- TTL enforcement

### This Release Includes
- Aegis 2.0 design PDF
- Reference implementations (Ada, Python)
- Binary support artifact
- Patched code bundles (with/without Nexus integration)
- Sentinel release kit

## Security
- SHA-256 checksums are provided in MANIFEST.json and must be verified before use.
- All code runs under sandbox and HTTPS-only assumptions.
- See SECURITY.md for full policy.

## Citation
Please cite as:
Harrison, J. (2025). Project SENTINAL v2.0 (Aegis Sentinel with Nexus Signal Engine). Zenodo. https://doi.org/10.5281/zenodo.16894912
