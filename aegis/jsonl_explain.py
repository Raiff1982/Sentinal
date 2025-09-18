"""
JSONL file-based implementation of explain store backend.
"""
import os
import json
import hashlib
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any

from .sentinel_config import DEFAULT_TTL_SECS
from . import explain

log = logging.getLogger(__name__)

class JsonlExplainStore(explain.ExplainStore):
    """JSONL file-based explain storage with rotation."""
    
    def __init__(self, base_path: str):
        """Initialize store.
        
        Args:
            base_path: Base directory for explain files
        """
        self.base_path = os.path.abspath(base_path)
        os.makedirs(self.base_path, exist_ok=True)
        
    def _path_for_date(self, dt: datetime) -> str:
        """Get explain file path for given date."""
        return os.path.join(
            self.base_path,
            f"explain-{dt.strftime('%Y-%m-%d')}.jsonl"
        )
        
    def _write_entry(self, path: str, entry: explain.ExplainEntry) -> bool:
        """Write single entry to file.
        
        Args:
            path: File path to write to
            entry: Entry to write
            
        Returns:
            True if write successful
        """
        try:
            record = {
                "timestamp": entry.timestamp.isoformat(),
                "agent": entry.agent,
                "category": entry.category,
                "data": entry.data,
                "metadata": entry.metadata,
                "ttl_sec": entry.ttl_sec
            }
            
            # Add integrity check
            content = json.dumps(record, sort_keys=True)
            record["_digest"] = hashlib.sha256(
                content.encode("utf-8")
            ).hexdigest()
            
            # Append to file
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record))
                f.write("\n")
            return True
            
        except Exception as e:
            log.error("Failed to write explain entry: %s", e)
            return False
            
    def write(self, entry: explain.ExplainEntry) -> bool:
        """Write an explanation entry.
        
        Args:
            entry: Entry to write
            
        Returns:
            bool: True if write successful
        """
        path = self._path_for_date(entry.timestamp)
        return self._write_entry(path, entry)
        
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
        entries = []
        
        # Default time range to last 24 hours
        if not start_time:
            start_time = datetime.utcnow() - timedelta(days=1)
        if not end_time:
            end_time = datetime.utcnow()
            
        # Get list of files in date range
        start_date = start_time.date()
        end_date = end_time.date()
        current = start_date
        while current <= end_date:
            path = self._path_for_date(datetime.combine(current, datetime.min.time()))
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        if len(entries) >= limit:
                            break
                        try:
                            record = json.loads(line)
                            ts = datetime.fromisoformat(record["timestamp"])
                            
                            # Apply filters
                            if ts < start_time or ts > end_time:
                                continue
                            if record["agent"] != agent:
                                continue
                                
                            # Validate integrity
                            content = {k:v for k,v in record.items() if k != "_digest"}
                            digest = hashlib.sha256(
                                json.dumps(content, sort_keys=True).encode("utf-8")
                            ).hexdigest()
                            if digest != record.get("_digest"):
                                log.warning("Integrity check failed for entry: %s", record["timestamp"])
                                continue
                                
                            entries.append(explain.ExplainEntry(
                                timestamp=ts,
                                agent=record["agent"], 
                                category=record["category"],
                                data=record["data"],
                                metadata=record["metadata"],
                                ttl_sec=record.get("ttl_sec", DEFAULT_TTL_SECS)
                            ))
                            
                        except Exception as e:
                            log.error("Failed to parse explain entry: %s", e)
                            continue
                            
            current += timedelta(days=1)
            
        return entries[:limit]
        
    def cleanup(self, max_age_sec: Optional[float] = None) -> int:
        """Remove expired entries.
        
        Args:
            max_age_sec: Maximum age in seconds to keep
            
        Returns:
            Number of entries removed
        """
        removed = 0
        cutoff = datetime.utcnow() - timedelta(
            seconds=max_age_sec or DEFAULT_TTL_SECS
        )
        
        for fname in os.listdir(self.base_path):
            if not fname.startswith("explain-") or not fname.endswith(".jsonl"):
                continue
                
            path = os.path.join(self.base_path, fname)
            temp_path = path + ".tmp"
            
            try:
                kept = 0
                with open(path, "r", encoding="utf-8") as fin:
                    with open(temp_path, "w", encoding="utf-8") as fout:
                        for line in fin:
                            try:
                                record = json.loads(line)
                                ts = datetime.fromisoformat(record["timestamp"])
                                ttl = record.get("ttl_sec", DEFAULT_TTL_SECS)
                                
                                if ts + timedelta(seconds=ttl) > cutoff:
                                    fout.write(line)
                                    kept += 1
                                else:
                                    removed += 1
                                    
                            except Exception as e:
                                log.error("Failed to process entry during cleanup: %s", e)
                                fout.write(line)  # Keep corrupted entries
                                kept += 1
                                
                if kept > 0:
                    os.replace(temp_path, path)
                else:
                    os.remove(path)  # Remove empty file
                    try:
                        os.remove(temp_path)
                    except FileNotFoundError:
                        pass
                        
            except Exception as e:
                log.error("Failed to cleanup file %s: %s", fname, e)
                try:
                    os.remove(temp_path)
                except FileNotFoundError:
                    pass
                    
        return removed