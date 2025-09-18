# Changelog

## 2.2.0 — System Diagnostics & Monitoring
- Added comprehensive diagnostics module for system health monitoring
- Implemented real-time resource usage tracking (CPU, memory, disk)
- Created interactive diagnostic dashboard in Web UI
- Added component health monitoring for critical systems
- Integrated security metrics and audit trail monitoring 
- Implemented pre-flight system checks
- Added API endpoints for diagnostic data
- Enhanced system reliability with early warning detection

## 2.1.0 — Advanced LLM Ensemble & Web UI
- Ensemble support for multiple LLMs (DistilBERT, RoBERTa, DistilGPT2, GPT2)
- Sentiment analysis: majority vote, average score, all model outputs
- Chat: show all LLM responses
- Refactor `ai_base.py` for easy model extension
- Update web UI to display ensemble results and all LLM outputs
- Add admin/user authentication and role-based feature restriction
- Admin dashboard: user management, logs, documentation links

## 2.0.0 — Nexus + Shield + Council wiring
- Add Nexus Signal Engine (entropy‑aware, arousal‑weighted, time‑decaying memory)
- Wire Sentinel to real Council; add fusion agents (Biofeedback/EnvSignal/ContextConflict)
- Meta‑Judge: three‑tier decisions with auditable thresholds
- Pre‑attack Shield: rate‑limit, PoW, canaries/honeypots, injection/Unicode anomaly guard
- Signed Ledger + CLI + Audit API; tests passing
