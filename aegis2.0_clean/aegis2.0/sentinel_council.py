
"""
Adapter exposing get_council() from aegis_timescales.
Uses classes defined in the shipped aegis_timescales.py.
Provides configuration and council initialization with proper agent setup.
"""
from typing import Optional, Dict, Any
import os
import json

from aegis_timescales import (
    AegisCouncil, NexusMemory,
    EchoSeedAgent, ShortTermAgent, MidTermAgent, LongTermArchivistAgent,
    TimeScaleCoordinator, MetaJudgeAgent, BiofeedbackAgent, EnvSignalAgent, ContextConflictAgent
)

# Default configuration
DEFAULT_CONFIG = {
    "per_agent_timeout_sec": 2.5,
    "max_workers": None,
    "memory": {
        "max_entries": 10000,
        "decay_rate": 0.01
    }
}

def get_council(
    per_agent_timeout_sec: float = 2.5,
    max_workers: Optional[int] = None,
    memory_config: Optional[Dict[str, Any]] = None,
    persistence_path: Optional[str] = None
) -> AegisCouncil:
    """Initialize and configure the Aegis Council with persistence support.
    
    Args:
        per_agent_timeout_sec: Timeout per agent in seconds
        max_workers: Maximum number of worker threads
        memory_config: Memory configuration overrides
        persistence_path: Path to memory persistence file
    """
    try:
        # Initialize memory configuration
        memory_cfg = DEFAULT_CONFIG["memory"].copy()
        if memory_config:
            memory_cfg.update(memory_config)
            
        # Initialize memory with persistence if path provided
        if persistence_path:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(persistence_path), exist_ok=True)
            
            # Load existing memory if available
            if os.path.exists(persistence_path):
                with open(persistence_path, 'r') as f:
                    saved_entries = json.load(f)
                memory_cfg['initial_entries'] = saved_entries
                
            memory = NexusMemory(**memory_cfg, persistence_path=persistence_path)
        else:
            memory = NexusMemory(**memory_cfg)
            
        # Initialize council
        council = AegisCouncil(
            per_agent_timeout_sec=per_agent_timeout_sec,
            max_workers=max_workers
        )
        
        # Register core agents
        council.register_agent(EchoSeedAgent("EchoSeed", memory))
        council.register_agent(ShortTermAgent("ShortTerm", memory))
        council.register_agent(MidTermAgent("MidTerm", memory))
        council.register_agent(LongTermArchivistAgent("LongArchivist", memory))
        
        # Register fusion agents
        council.register_agent(BiofeedbackAgent("BiofeedbackAgent", memory))
        council.register_agent(EnvSignalAgent("EnvSignalAgent", memory))
        council.register_agent(ContextConflictAgent("ContextConflictAgent", memory))
        
        # Register coordination agents
        council.register_agent(TimeScaleCoordinator("TimeScaleCoordinator", memory))
        council.register_agent(MetaJudgeAgent("MetaJudge", memory))
        
        return council
    except Exception as e:
        raise RuntimeError(f"Failed to initialize Aegis Council: {e}") from e
