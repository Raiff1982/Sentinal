"""
Aegis Sentinel - Primary interface and API for the guardrails system.

The Sentinel class provides a high-level interface to the Aegis guardrails system,
abstracting away the complexity of multi-agent coordination and persistence
management. It offers a simple API for content safety checks, analysis, and 
response generation while maintaining audit trails and evolutionary learning.

Key Features:
- Simple High-Level API: Easy-to-use methods for common safety operations
- Configurable Safety Policies: Adjustable risk thresholds and safety rules
- Multi-Backend Persistence: Pluggable explain stores (JSONL, Nexus)
- Audit Trail Generation: Cryptographic verification of decisions
- Safety Evolution: Continuous learning from past decisions
- Red-Team Resistance: Built-in defenses against adversarial inputs

Core Methods:
    check(text: str, context: dict) -> CheckResult:
        Perform a quick safety check on input text
        
    analyze(text: str, context: dict) -> Dict[str, Any]:
        Get detailed safety analysis and risk assessment
        
    respond(text: str, context: dict) -> Dict[str, Any]:
        Generate a safe response with safety metrics

Usage:
    >>> from aegis import Sentinel
    >>> sentinel = Sentinel()
    
    # Quick safety check
    >>> result = sentinel.check("Is this safe?", {"intent": "scan"})
    >>> if result.allow:
    ...     print("Content is safe")
    
    # Detailed analysis
    >>> analysis = sentinel.analyze(text, {"risk_level": "high"})
    >>> print(f"Risk score: {analysis['risk_score']}")
    
    # Generate safe response
    >>> response = sentinel.respond(text, {"model": "gpt-4"})
    >>> print(response["text"])

Configuration:
    The Sentinel can be configured through sentinel_config.py or 
    environment variables:
    
    - EXPLAIN_BACKEND: "jsonl" or "nexus"
    - MAX_AGENT_TIMEOUT_SEC: Maximum agent response time
    - ENABLE_PERSISTENCE: Enable/disable explain store
"""

# Standard library
import hmac
import http.server
import hashlib
import json
import logging
import os
import secrets
import socketserver
import threading
import time
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple, TypedDict

# Aegis components
from .sentinel_config import (
    EXPLAIN_BACKEND,
    ENABLE_PERSISTENCE,
    MAX_AGENT_TIMEOUT_SEC,
    ExplainBackend
)
from .explain import ExplainStore, create_explain_store, ExplainEntry
from .sentinel_council import AegisCouncil, MetaJudgeAgent
from .aegis_evolution import MetaGenes

try:
    import portalocker
    HAVE_PORTALOCKER = True
except ImportError:
    HAVE_PORTALOCKER = False
    class _NoLock:
        def __init__(self, f, *a, **k): self.f = f
        def __enter__(self): return self.f
        def __exit__(self, exc_type, exc, tb): return False
    class portalocker:  # shim
        LOCK_EX = 0
        class LockException(Exception): pass
        @staticmethod
        def Lock(path, *a, **k):
            # Fallback: open file for append without locking
            return open(path, 'a', encoding='utf-8')

from .sentinel_config import (
    EXPLAIN_BACKEND,
    MAX_AGENT_TIMEOUT_SEC,
    ENABLE_PERSISTENCE,
    LOG_LEVEL,
    ExplainBackend
)
from .sentinel_council import (
    AegisCouncil,
    EchoSeedAgent,
    ShortTermAgent, 
    MidTermAgent,
    LongTermArchivistAgent,
    TimeScaleCoordinator,
    MetaJudgeAgent
)
from .explain import ExplainStore, create_explain_store, ExplainEntry
from .aegis_evolution import MetaGenes

# Configure logging
logging.basicConfig(level=getattr(logging, LOG_LEVEL), 
                   format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("AEGIS-Guardrails")

# Default paths and settings
DEFAULT_LEDGER_DIR = "./aegis_ledger"
DEFAULT_POLICY_PATH = "./aegis_policies.json"
DEFAULT_SECRET_PATH = "./aegis_secret.key" 
DEFAULT_API_PORT = 8787
DEFAULT_RETENTION_DAYS = 30

# Type definitions
class LedgerRecord(TypedDict):
    """A single record in the audit ledger."""
    ts: str  # ISO format timestamp
    decision: str  # PROCEED/CAUTION/BLOCK
    policy: Dict[str, float]  # Active policy parameters
    input_digest: str  # HMAC of input data