"""
Configuration settings and feature flags for Aegis system.
"""
import os
from enum import Enum
from typing import Optional, Dict, Any

class ExplainBackend(str, Enum):
    """Supported explanation storage backends"""
    JSONL = "jsonl"  # Classic file-based JSON lines storage
    NEXUS = "nexus"  # Advanced graph-based storage with timeline support

# Feature flags and configuration with defaults
EXPLAIN_BACKEND: ExplainBackend = ExplainBackend(os.getenv("AEGIS_EXPLAIN_BACKEND", "jsonl").lower())
MAX_AGENT_TIMEOUT_SEC: float = float(os.getenv("AEGIS_MAX_AGENT_TIMEOUT_SEC", "2.5"))
ENABLE_PERSISTENCE: bool = os.getenv("AEGIS_ENABLE_PERSISTENCE", "true").lower() == "true"
LOG_LEVEL: str = os.getenv("AEGIS_LOG_LEVEL", "INFO")

# Input validation limits 
MAX_INPUT_LENGTH: int = int(os.getenv("AEGIS_MAX_INPUT_LENGTH", "4096"))
MAX_MEMORY_ENTRIES: int = int(os.getenv("AEGIS_MAX_MEMORY_ENTRIES", "10000"))
DEFAULT_TTL_SECS: float = float(os.getenv("AEGIS_DEFAULT_TTL_SECS", "3600"))

def validate_config() -> Optional[str]:
    """Validate current configuration settings.
    
    Returns:
        str or None: Error message if invalid, None if valid
    """
    try:
        if not isinstance(EXPLAIN_BACKEND, ExplainBackend):
            return f"Invalid EXPLAIN_BACKEND value: {EXPLAIN_BACKEND}"
        if not 0 < MAX_AGENT_TIMEOUT_SEC <= 30:
            return f"MAX_AGENT_TIMEOUT_SEC must be between 0 and 30, got {MAX_AGENT_TIMEOUT_SEC}"
        if not 0 < MAX_INPUT_LENGTH <= 1_000_000:
            return f"MAX_INPUT_LENGTH must be between 0 and 1,000,000, got {MAX_INPUT_LENGTH}"
        if not 0 < MAX_MEMORY_ENTRIES <= 1_000_000:
            return f"MAX_MEMORY_ENTRIES must be between 0 and 1,000,000, got {MAX_MEMORY_ENTRIES}"
        if not 0 < DEFAULT_TTL_SECS <= 86400:
            return f"DEFAULT_TTL_SECS must be between 0 and 86400, got {DEFAULT_TTL_SECS}"
        return None
    except Exception as e:
        return f"Configuration validation error: {str(e)}"

def get_config() -> Dict[str, Any]:
    """Get current configuration as a dictionary."""
    return {
        "explain_backend": EXPLAIN_BACKEND.value,
        "max_agent_timeout_sec": MAX_AGENT_TIMEOUT_SEC,
        "enable_persistence": ENABLE_PERSISTENCE,
        "log_level": LOG_LEVEL,
        "max_input_length": MAX_INPUT_LENGTH,
        "max_memory_entries": MAX_MEMORY_ENTRIES,
        "default_ttl_secs": DEFAULT_TTL_SECS
    }