"""
Aegis - A Multi-Timescale AI Guardrails System

Aegis is a comprehensive safety system for AI applications that provides real-time 
guardrails through multi-timescale analysis and persistent memory. The system uses
a coordinated network of specialized agents to assess safety across different time 
horizons, from immediate content validation to long-term pattern recognition.

Key Features:
- Unified Sentinel Interface: Simple, high-level API for safety checks and analysis
- Multi-timescale Agent Coordination: Network of specialized safety agents  
- Pluggable Persistence: Support for both JSONL and Nexus explain stores
- Input Validation: Comprehensive text analysis with warning system
- Audit Trails: Cryptographic integrity checks and decision explanations
- Evolution: Adaptive learning from past decisions and outcomes

Quick Start:
    >>> from aegis import Sentinel
    >>> sentinel = Sentinel()  # Uses default JSONL explain store
    >>> result = sentinel.check("Your text here", {"intent": "scan"})
    >>> if result.allow:
    ...     analysis = sentinel.analyze(text)

For persistent memory and advanced features:
    >>> from aegis import Sentinel, NexusExplainStore
    >>> store = NexusExplainStore(persistence_path="memory.db")
    >>> sentinel = Sentinel(explain_store=store)

See the README for complete documentation and migration guide from older versions.
"""

from .sentinel import Sentinel
from .sentinel_council import (
    AegisCouncil,
    MetaJudgeAgent,
    EchoSeedAgent,
    ShortTermAgent, 
    MidTermAgent,
    LongTermArchivistAgent,
    TimeScaleCoordinator
)
from .explain import JsonlExplainStore, NexusExplainStore
from .sanitizer import InputSanitizer
from .sentinel_config import *

__version__ = "2.0.0"
__all__ = [
    "Sentinel",
    "AegisCouncil",
    "MetaJudgeAgent",
    "EchoSeedAgent", 
    "ShortTermAgent",
    "MidTermAgent",
    "LongTermArchivistAgent",
    "TimeScaleCoordinator",
    "JsonlExplainStore",
    "NexusExplainStore",
    "InputSanitizer"
]