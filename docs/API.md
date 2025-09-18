# Aegis API Documentation

## Core Classes

### Sentinel

The main interface to the Aegis guardrails system.

```python
class Sentinel:
    def __init__(self, explain_store: Optional[ExplainStore] = None):
        """Initialize a new Sentinel instance.
        
        Args:
            explain_store: Custom explain store implementation. If None,
                         uses JsonlExplainStore by default.
        """
        
    def check(self, text: str, context: Dict[str, Any]) -> CheckResult:
        """Perform a quick safety check on the input text.
        
        Args:
            text: The input text to check
            context: Additional context about the request
                    {
                        "intent": str,  # Purpose of the check
                        "risk_level": str,  # "low", "medium", "high"
                        "model": str,  # Optional: model name
                    }
                    
        Returns:
            CheckResult with fields:
                allow: bool - Whether the content is safe
                reason: Optional[str] - Reason if blocked
                severity: float - Risk severity score
        """
        
    def analyze(self, text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Get detailed safety analysis of the input text.
        
        Args:
            text: The input text to analyze
            context: Analysis context (same as check())
            
        Returns:
            {
                "risk_score": float,  # Overall risk (0-1)
                "safety_level": str,  # "safe", "warning", "unsafe"
                "confidence": float,  # Confidence in assessment
                "categories": List[str],  # Triggered categories
                "warnings": List[Dict],  # Specific issues found
                "metadata": Dict  # Additional analysis info
            }
        """
        
    def respond(self, text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a safe response to the input.
        
        Args:
            text: The input to respond to
            context: Response context (same as check())
            
        Returns:
            {
                "text": str,  # The generated response
                "risk_score": float,  # Response safety score
                "safety_level": str,  # Response safety level
                "metadata": Dict  # Generation metadata
            }
        """
        
    def get_status(self) -> Dict[str, Any]:
        """Get the current system status.
        
        Returns:
            {
                "version": str,
                "explain_store": str,
                "memory_size": int,
                "persistence_path": Optional[str]
            }
        """
        
    def get_history(self, limit: Optional[int] = None) -> List[Dict]:
        """Get recent history entries.
        
        Args:
            limit: Maximum entries to return
            
        Returns:
            List of history entries with timestamps
        """
```

### ExplainStore

Base interface for persistence backends.

```python
class ExplainStore(Protocol):
    def add_entry(self, entry: ExplainEntry) -> None:
        """Add a new explanation entry."""
        
    def get_entries(self, 
                   limit: Optional[int] = None,
                   filters: Optional[Dict] = None) -> List[ExplainEntry]:
        """Retrieve explanation entries."""
        
    def clear(self) -> None:
        """Clear all entries."""
```

### JsonlExplainStore

Simple file-based storage using JSONL format.

```python
class JsonlExplainStore(ExplainStore):
    def __init__(self, persistence_path: str):
        """Initialize JSONL storage.
        
        Args:
            persistence_path: Path to JSONL file
        """
```

### NexusExplainStore 

Advanced persistent storage with indexing.

```python
class NexusExplainStore(ExplainStore):
    def __init__(self, 
                persistence_path: str,
                max_size: Optional[int] = None):
        """Initialize Nexus storage.
        
        Args:
            persistence_path: Path to storage file
            max_size: Optional maximum entries
        """
```

## Types

### CheckResult

Result of a safety check.

```python
class CheckResult(TypedDict):
    allow: bool  # Whether content is allowed
    reason: Optional[str]  # Reason if blocked
    severity: float  # Risk severity (0-1)
```

### ExplainEntry

An explanation entry for storage.

```python
class ExplainEntry(TypedDict):
    id: str  # Unique entry ID
    text: str  # Input text
    context: Dict[str, Any]  # Request context
    result: CheckResult  # Check result
    analysis: Optional[Dict]  # Full analysis if done
    timestamp: str  # ISO format timestamp
```

## Configuration

The system can be configured through `sentinel_config.py` or environment variables:

```python
# sentinel_config.py
EXPLAIN_BACKEND = "jsonl"  # or "nexus"
MAX_AGENT_TIMEOUT_SEC = 30
ENABLE_PERSISTENCE = True
```

Environment variables override config file settings:
- `AEGIS_EXPLAIN_BACKEND`
- `AEGIS_MAX_TIMEOUT`
- `AEGIS_ENABLE_PERSISTENCE`
- `AEGIS_STORE_PATH`

## Error Handling

The system may raise:

- `ValueError`: Invalid input or configuration
- `RuntimeError`: Internal system error
- `TimeoutError`: Agent timeout
- `IOError`: Storage/persistence error

Example:
```python
try:
    result = sentinel.check(text, context)
except ValueError as e:
    print(f"Invalid input: {e}")
except TimeoutError:
    print("Safety check timed out")
```