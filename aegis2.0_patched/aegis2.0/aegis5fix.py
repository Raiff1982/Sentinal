from datetime import datetime, timezone, timedelta
import threading, heapq
from typing import Dict, Any, List, Tuple, Optional
import queue

class NexusMemory:
    def __init__(self, max_entries: int = 10_000, default_ttl_secs: int = 7*24*3600):
        self.store: Dict[str, Dict[str, Any]] = {}
        self.expiration_heap: List[Tuple[float, str]] = []
        self.max_entries = max_entries
        self.default_ttl_secs = default_ttl_secs
        self._write_queue = queue.Queue()
        self._lock = threading.Lock()
        self._writer_thread = threading.Thread(target=self._process_writes, daemon=True)
        self._writer_thread.start()

    def _process_writes(self) -> None:
        while True:
            key, value, ttl = self._write_queue.get()
            now = datetime.now(timezone.utc)
            hashed = self._hash(key)
            with self._lock:
                if len(self.store) >= self.max_entries:
                    self._purge_oldest(now)
                self.store[hashed] = {"value": value, "timestamp": now, "ttl": int(ttl)}
                expiration_time = (now + timedelta(seconds=ttl)).timestamp()
                heapq.heappush(self.expiration_heap, (expiration_time, hashed))
            self._write_queue.task_done()

    def write(self, key: str, value: Any, ttl_secs: Optional[int] = None) -> None:
        if not isinstance(key, str):
            raise ValueError("Key must be a string")
        ttl = ttl_secs if ttl_secs is not None else self.default_ttl_secs
        self._write_queue.put((key, value, ttl))