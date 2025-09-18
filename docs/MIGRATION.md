# Migrating to Aegis 2.0

This guide helps you transition from the legacy council-based system to the new unified Sentinel interface in Aegis 2.0.

## Key Changes

1. **Simplified Interface**: The new Sentinel class replaces direct council interaction
2. **Improved Persistence**: New pluggable explain store system
3. **Enhanced Safety**: Better input validation and warning system
4. **Unified Configuration**: Centralized settings in sentinel_config.py

## Migration Steps

### 1. Update Imports

Old code:
```python
from aegis_timescales import AegisCouncil, MetaJudgeAgent
from sentinel_council import get_council
```

New code:
```python
from aegis import Sentinel, NexusExplainStore  # For persistent memory
# or simply:
from aegis import Sentinel  # For basic usage
```

### 2. Replace Council Setup

Old code:
```python
council = get_council(persistence_path="memory.json")
result = council.dispatch({
    "text": user_input,
    "intent": "scan",
    "_signals": {
        "bio": {"stress": 0.2},
        "env": {"context_risk": 0.3}
    }
})
decision = next(
    (r for r in result["reports"] if r["agent"] == "MetaJudge"), 
    {}
).get("details", {}).get("decision")
```

New code:
```python
sentinel = Sentinel()  # Uses JSONL store by default
# Or with custom explain store:
store = NexusExplainStore(persistence_path="memory.db")
sentinel = Sentinel(explain_store=store)

result = sentinel.check(user_input, {
    "intent": "scan",
    "risk_level": "low"  # Replaces signal system
})
if result.allow:
    analysis = sentinel.analyze(user_input)
```

### 3. Update Context Format

Old format:
```python
context = {
    "_signals": {
        "bio": {"stress": 0.3},
        "env": {"context_risk": 0.4}
    }
}
```

New format:
```python
context = {
    "intent": "chat",  # Purpose of the check
    "risk_level": "medium",  # "low", "medium", "high"
    "model": "gpt-4"  # Optional: specific model
}
```

### 4. Handle Results

Old code:
```python
if decision == "BLOCK":
    raise ValueError(result["reports"][0]["details"]["reason"])
```

New code:
```python
result = sentinel.check(text, context)
if not result.allow:
    raise ValueError(result.reason)
```

### 5. Access History

Old code:
```python
memory = council.memory.entries
```

New code:
```python
status = sentinel.get_status()  # Get memory size etc
history = sentinel.get_history(limit=10)  # Get recent entries
```

## Feature Mapping

| Old Feature | New Feature | Notes |
|------------|-------------|--------|
| Council dispatch | sentinel.check() | Simpler interface |
| Memory store | ExplainStore | More flexible backends |
| Signal system | Context dict | Cleaner API |
| Agent reports | Analysis results | Structured output |
| Evolution | Auto-managed | Built into Sentinel |

## Configuration Changes

1. **Environment Variables**:
   - Old: `COUNCIL_MEMORY_PATH`
   - New: `EXPLAIN_BACKEND`, `EXPLAIN_STORE_PATH`

2. **Configuration File**:
   - Old: Individual agent configs
   - New: Centralized in sentinel_config.py

## Backwards Compatibility

The AegisCouncil class is still available for backwards compatibility:

```python
from aegis import AegisCouncil
council = AegisCouncil()  # Old interface still works
```

However, we recommend migrating to the new Sentinel interface for:
- Simpler code
- Better error handling
- Enhanced safety features
- Improved persistence
- Future updates and features

## Common Issues

1. **Memory Format**: 
   - The new explain store uses a different format
   - Use `aegis.tools.convert_memory()` to migrate data

2. **Context Changes**:
   - Signal system replaced with risk levels
   - Use context["risk_level"] instead of signals

3. **Error Handling**:
   - More structured errors with proper types
   - Better validation feedback

## Need Help?

- Check the full documentation in docs/
- File issues on GitHub
- Join our Discord community