## 🧪 Testing

Run all tests including integration tests:
```bash
python -m pytest
```

Run specific test categories:
```bash
# Unit tests only
python -m pytest tests/unit/

# Integration tests
python -m pytest tests/integration/

# Quick smoke test
python -m pytest tests/test_smoke.py
```

## 🔍 Integrity Verification

All decisions are stored with HMAC signatures for tamper-evidence:

```bash
# Verify explain store integrity
python -m aegis.tools.verify path/to/store.jsonl

# Check specific time range
python -m aegis.tools.verify --start="2025-01-01" --end="2025-09-18" path/to/store.jsonl
```

## 🌐 Web UI Integration

The Aegis Web UI provides a user-friendly interface for:
- Content safety scanning
- Interactive chat with guardrails
- Batch processing
- Analytics dashboard
- Feedback collection

```bash
# Install web dependencies
pip install -r webui/requirements.txt

# Run development server
python -m webui.app
```

Visit http://localhost:5000 to access the interface.

## ⚙️ Configuration

Configure through `sentinel_config.py` or environment variables:

```python
# sentinel_config.py
EXPLAIN_BACKEND = "jsonl"  # or "nexus"
MAX_AGENT_TIMEOUT_SEC = 30
ENABLE_PERSISTENCE = True
```

Environment variables override config file:
- `AEGIS_EXPLAIN_BACKEND`: Storage backend ("jsonl" or "nexus")
- `AEGIS_MAX_TIMEOUT`: Maximum agent response time
- `AEGIS_ENABLE_PERSISTENCE`: Enable/disable explain store
- `AEGIS_STORE_PATH`: Custom storage path

## 🚀 Advanced Usage

### Persistent Storage

```python
from aegis import Sentinel, NexusExplainStore

# Initialize Nexus storage
store = NexusExplainStore(
    persistence_path="memory.db",
    max_size=10000
)

# Create Sentinel with custom store
sentinel = Sentinel(explain_store=store)
```

### Response Generation

```python
# Generate safe responses
response = sentinel.respond(user_input, {
    "intent": "chat",
    "model": "gpt-4",
    "risk_level": "medium"
})

print(response["text"])
```

### Batch Processing

```python
results = []
for text in texts:
    # Quick safety check
    result = sentinel.check(text, {
        "intent": "batch_scan",
        "risk_level": "low"
    })
    
    if result.allow:
        # Full analysis if safe
        analysis = sentinel.analyze(text)
        results.append(analysis)
```

## 📜 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## 📚 Citation

If you use Aegis in your research, please cite:

```bibtex
@software{aegis_guardrails,
  title = {Aegis: Multi-Timescale AI Guardrails System},
  version = {2.0.0},
  year = {2025},
  doi = {10.5281/zenodo.16853922},
  url = {https://github.com/yourusername/aegis}
}
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests (`pytest`)
5. Submit a pull request

See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## 💬 Support

- 📘 [Documentation](docs/)
- 🐛 [Issue Tracker](https://github.com/yourusername/aegis/issues)
- 💭 [Discord Community](https://discord.gg/aegis)
- 📧 [Email Support](mailto:support@aegis.ai)



# AEGIS: Multi-Timescale AI Guardrails System

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.16853922.svg)](https://doi.org/10.5281/zenodo.16853922)

**AEGIS** is a comprehensive safety system that provides real-time guardrails for AI applications through multi-timescale analysis and persistent memory. Using a coordinated network of specialized agents, it offers robust protection across different time horizons while maintaining auditability and evolutionary learning.

## 🌟 Highlights

- 🛡️ **Unified Sentinel Interface**: Simple, high-level API for safety operations
- 🧠 **Multi-timescale Analysis**: From immediate validation to long-term patterns
- 💾 **Flexible Persistence**: Pluggable explain stores (JSONL, Nexus)
- 📊 **Rich Analytics**: Detailed safety metrics and explanations
- 🔄 **Safety Evolution**: Continuous learning from past decisions

## ⚡ Quick Start

```python
from aegis import Sentinel

# Create a Sentinel instance
sentinel = Sentinel()

# Check if content is safe
result = sentinel.check("Your text here", {
    "intent": "scan",
    "risk_level": "low"
})

if result.allow:
    # Get detailed analysis
    analysis = sentinel.analyze(text)
    print(f"Risk Score: {analysis['risk_score']}")
```

## 🔒 Key Security Features

- **Shield (Pre-Attack):** Rate limiting, proof-of-work, injection guards
- **Validation (Pre-Flight):** Input sanitization and challenge validation
- **Analysis (In-Flight):** Multi-agent risk and safety assessment 
- **Audit (Post-Flight):** HMAC-signed JSONL with full verification
- **Memory (Persistent):** Time-aware, weighted decision storage

## 📚 Documentation

- [API Reference](docs/API.md) - Detailed class and method documentation
- [Examples](docs/EXAMPLES.md) - Common usage patterns and code samples
- [Migration Guide](docs/MIGRATION.md) - Upgrading from older versions

## 🛠️ Installation

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install package
pip install -e .
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
