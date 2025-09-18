"""
Nexus-based implementation of explain store backend.
"""
import json
import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, cast

from .sentinel_config import DEFAULT_TTL_SECS
from . import explain

log = logging.getLogger(__name__)

class NexusExplainStore(explain.ExplainStore):
    """Nexus-based explain storage with timeline support."""
    
    def __init__(self, nexus_client: Any):
        """Initialize store.
        
        Args:
            nexus_client: Configured nexus client instance
        """
        self.nexus = nexus_client
        if not hasattr(self.nexus, "ingest") or not hasattr(self.nexus, "query"):
            raise ValueError("Invalid nexus client - must support ingest/query API")
            
    def write(self, entry: explain.ExplainEntry) -> bool:
        """Write an explanation entry.
        
        Args:
            entry: Entry to write
            
        Returns:
            bool: True if write successful
        """
        try:
            # Convert to nexus record format
            record = {
                "timestamp": entry.timestamp.isoformat(),
                "agent": entry.agent,
                "category": entry.category,
                "data": entry.data,
                "metadata": entry.metadata
            }
            
            # Ingest to nexus with TTL
            self.nexus.ingest(
                source=f"aegis:{entry.agent}",
                category=entry.category,
                data=record,
                ttl_sec=entry.ttl_sec
            )
            return True
            
        except Exception as e:
            log.error("Failed to write explain entry to Nexus: %s", e)
            return False
            
    def read(self, agent: str, start_time: Optional[datetime] = None,
             end_time: Optional[datetime] = None, 
             limit: int = 1000) -> List[explain.ExplainEntry]:
        """Read explanation entries.
        
        Args:
            agent: Agent name to filter by
            start_time: Optional start time filter
            end_time: Optional end time filter
            limit: Maximum entries to return
            
        Returns:
            List of matching entries
        """
        try:
            # Build nexus query
            query = {
                "source": f"aegis:{agent}",
                "limit": limit
            }
            if start_time:
                query["start_time"] = start_time.isoformat()
            if end_time:
                query["end_time"] = end_time.isoformat()
                
            # Execute query
            results = self.nexus.query(query)
            
            # Convert to ExplainEntry objects
            entries = []
            for r in results:
                try:
                    data = r["data"]
                    entries.append(explain.ExplainEntry(
                        timestamp=datetime.fromisoformat(data["timestamp"]),
                        agent=data["agent"],
                        category=data["category"], 
                        data=data["data"],
                        metadata=data["metadata"],
                        ttl_sec=r.get("ttl_sec", DEFAULT_TTL_SECS)
                    ))
                except Exception as e:
                    log.error("Failed to parse Nexus record: %s", e)
                    continue
                    
            return entries[:limit]
            
        except Exception as e:
            log.error("Failed to read from Nexus: %s", e)
            return []
            
    def cleanup(self, max_age_sec: Optional[float] = None) -> int:
        """Remove expired entries.
        
        Args:
            max_age_sec: Maximum age in seconds to keep
            
        Returns:
            Number of entries removed
        """
        try:
            # Nexus handles TTL expiration automatically
            cutoff = datetime.utcnow() - timedelta(
                seconds=max_age_sec or DEFAULT_TTL_SECS
            )
            return cast(int, self.nexus.cleanup(
                source="aegis:*",
                before=cutoff.isoformat()
            ))
        except Exception as e:
            log.error("Failed to cleanup Nexus entries: %s", e)
            return 0