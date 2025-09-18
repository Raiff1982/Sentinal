"""
Adapter exposing get_council() from aegis_timescales.
Uses classes defined in the shipped aegis_timescales.py.
"""
import os
import logging
from typing import Dict, Any, Optional
from aegis_timescales import (
    AegisCouncil, NexusMemory,
    EchoSeedAgent, ShortTermAgent, MidTermAgent, LongTermArchivistAgent,
    TimeScaleCoordinator, MetaJudgeAgent, BiofeedbackAgent, EnvSignalAgent, ContextConflictAgent
)

log = logging.getLogger("AEGIS-Council")

# Default policy configuration
DEFAULT_POLICIES = {
    "risk_cap": 0.75,          # Risk threshold above which to trigger caution
    "min_integrity": 0.2,      # Minimum required memory integrity
    "stress_cap": 0.75,        # Maximum allowable operator stress
    "timescale_cap": 0.65,     # Threshold for timescale signal
    "conflict_threshold": 0.5,  # Threshold for context-intent conflict
    "max_entries": 20_000,     # Maximum memory entries
    "default_ttl": 14 * 24 * 3600,  # Default TTL in seconds (14 days)
}

def get_council(
    per_agent_timeout_sec: float = 2.5,
    max_workers: Optional[int] = None,
    policies: Optional[Dict[str, float]] = None,
    memory_config: Optional[Dict[str, Any]] = None,
    persistence_path: Optional[str] = None
) -> AegisCouncil:
    """Initialize and configure the Aegis Council."""
    try:
        # Merge configurations
        active_policies = DEFAULT_POLICIES.copy()
        if policies:
            active_policies.update(policies)
        
        # Set up memory config
        memory_cfg = {
            "max_entries": active_policies["max_entries"],
            "default_ttl_secs": active_policies["default_ttl"]
        }
        if memory_config:
            memory_cfg.update(memory_config)
            
        # Add persistence path if provided
        if persistence_path:
            memory_cfg["persistence_path"] = persistence_path
        
        # Initialize council with configured timeout and workers
        council = AegisCouncil(
            per_agent_timeout_sec=float(per_agent_timeout_sec),
            max_workers=max_workers
        )
        
        # Configure shared memory with merged settings
        council.memory = NexusMemory(**memory_cfg)
        
        # Register core seed and timescale agents
        log.info("Registering core agents...")
        council.register_agent(EchoSeedAgent("EchoSeed", council.memory))
        council.register_agent(ShortTermAgent("ShortTerm", council.memory))
        council.register_agent(MidTermAgent("MidTerm", council.memory))
        council.register_agent(LongTermArchivistAgent("LongArchivist", council.memory))
        
        # Register fusion agents
        log.info("Registering fusion agents...")
        council.register_agent(BiofeedbackAgent("BiofeedbackAgent", council.memory))
        council.register_agent(EnvSignalAgent("EnvSignalAgent", council.memory))
        council.register_agent(ContextConflictAgent("ContextConflictAgent", council.memory))
        
        # Register coordination and meta-judgment agents with policies
        log.info("Registering coordination and judgment agents...")
        council.register_agent(TimeScaleCoordinator("TimeScaleCoordinator", council.memory))
        council.register_agent(MetaJudgeAgent("MetaJudge", council.memory, active_policies))
        
        log.info("Council initialization complete with %d agents", len(council.agents))
        return council
        
    except Exception as e:
        log.error("Failed to initialize council: %s", e, exc_info=True)
        raise RuntimeError(f"Council initialization failed: {e}") from e