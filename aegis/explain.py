"""
Abstract base classes for Aegis explanation storage backends.
"""
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass

@dataclass
class ExplainEntry:
    """A single explanation entry."""
    timestamp: datetime
    agent: str
    category: str
    data: Dict[str, Any]
    metadata: Dict[str, Any]
    ttl_sec: float = 3600.0

class ExplainStore(ABC):
    """Abstract base class for explanation storage backends."""
    
    @abstractmethod
    def write(self, entry: ExplainEntry) -> bool:
        """Write an explanation entry.
        
        Args:
            entry: The entry to write
            
        Returns:
            bool: True if write successful
        """
        pass
        
    @abstractmethod
    def read(self, agent: str, start_time: Optional[datetime] = None,
            end_time: Optional[datetime] = None, limit: int = 1000) -> List[ExplainEntry]:
        """Read explanation entries.
        
        Args:
            agent: Agent name to filter by
            start_time: Optional start time filter
            end_time: Optional end time filter
            limit: Maximum entries to return
            
        Returns:
            List of matching entries
        """
        pass
        
    @abstractmethod
    def cleanup(self, max_age_sec: Optional[float] = None) -> int:
        """Remove expired entries.
        
        Args:
            max_age_sec: Maximum age in seconds to keep
            
        Returns:
            Number of entries removed
        """
        pass
        
class JsonlExplainStore(ExplainStore):
    """JSONL file-based explain storage."""
    
    def __init__(self, base_path: str):
        """Initialize store with base path for explain files."""
        self.base_path = base_path
        # TODO: Implement
        
    def write(self, entry: ExplainEntry) -> bool:
        # TODO: Implement
        pass
        
    def read(self, agent: str, start_time: Optional[datetime] = None,
             end_time: Optional[datetime] = None, limit: int = 1000) -> List[ExplainEntry]:
        # TODO: Implement
        pass
        
    def cleanup(self, max_age_sec: Optional[float] = None) -> int:
        # TODO: Implement
        pass

class NexusExplainStore(ExplainStore):
    """Nexus-based explain storage."""
    
    def __init__(self, nexus_client: Any):
        """Initialize store with nexus client."""
        self.nexus = nexus_client
        # TODO: Implement
        
    def write(self, entry: ExplainEntry) -> bool:
        # TODO: Implement
        pass
        
    def read(self, agent: str, start_time: Optional[datetime] = None,
             end_time: Optional[datetime] = None, limit: int = 1000) -> List[ExplainEntry]:
        # TODO: Implement
        pass
        
    def cleanup(self, max_age_sec: Optional[float] = None) -> int:
        # TODO: Implement
        pass

def create_explain_store(backend_type: str, **kwargs) -> ExplainStore:
    """Factory function to create explain store instance."""
    if backend_type == "jsonl":
        return JsonlExplainStore(kwargs.get("base_path", "./explain"))
    elif backend_type == "nexus":
        return NexusExplainStore(kwargs["nexus_client"])
    else:
        raise ValueError(f"Unknown backend type: {backend_type}")