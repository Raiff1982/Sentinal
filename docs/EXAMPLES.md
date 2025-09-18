# Aegis Examples

This document provides examples of common use cases for the Aegis guardrails system.

## Basic Usage

### Simple Safety Check

```python
from aegis import Sentinel

# Create Sentinel instance
sentinel = Sentinel()

# Check if content is safe
text = "Is this safe to process?"
context = {
    "intent": "content_scan",
    "risk_level": "low"
}

result = sentinel.check(text, context)
if result.allow:
    print("Content is safe!")
else:
    print(f"Content blocked: {result.reason}")
```

### Detailed Analysis

```python
from aegis import Sentinel

sentinel = Sentinel()

# Get detailed safety analysis
text = "Please analyze this content carefully"
context = {
    "intent": "deep_scan",
    "risk_level": "medium",
    "model": "gpt-4"
}

analysis = sentinel.analyze(text, context)
print(f"Risk Score: {analysis['risk_score']}")
print(f"Safety Level: {analysis['safety_level']}")
print(f"Confidence: {analysis['confidence']}")

if analysis['warnings']:
    print("\nWarnings:")
    for warning in analysis['warnings']:
        print(f"- {warning['message']}")
```

### Chat Safety

```python
from aegis import Sentinel

sentinel = Sentinel()

def safe_chat(user_input: str) -> str:
    # First check if input is safe
    check = sentinel.check(user_input, {
        "intent": "chat",
        "risk_level": "medium"
    })
    
    if not check.allow:
        return f"I cannot process that: {check.reason}"
        
    # Generate safe response
    response = sentinel.respond(user_input, {
        "intent": "chat",
        "model": "gpt-4"
    })
    
    return response["text"]

# Example usage
print(safe_chat("Hello!"))
```

## Advanced Usage

### Custom Persistence

```python
from aegis import Sentinel, NexusExplainStore

# Create custom explain store
store = NexusExplainStore(
    persistence_path="memory.db",
    max_size=10000
)

# Use custom store
sentinel = Sentinel(explain_store=store)

# Check with history
def check_with_history(text: str) -> bool:
    # Get recent history
    history = sentinel.get_history(limit=5)
    
    # Check new content
    result = sentinel.check(text, {
        "intent": "scan",
        "risk_level": "medium",
        "history": history  # Provide context
    })
    
    return result.allow
```

### Batch Processing

```python
from aegis import Sentinel
from typing import List, Dict
import asyncio

sentinel = Sentinel()

async def process_batch(texts: List[str]) -> List[Dict]:
    results = []
    
    # Process each text
    for text in texts:
        # Quick check first
        check = sentinel.check(text, {
            "intent": "batch_scan",
            "risk_level": "low"
        })
        
        if check.allow:
            # Full analysis if safe
            analysis = sentinel.analyze(text, {
                "intent": "batch_scan",
                "risk_level": "low"
            })
            results.append({
                "text": text,
                "safe": True,
                "analysis": analysis
            })
        else:
            results.append({
                "text": text,
                "safe": False,
                "reason": check.reason
            })
            
        # Don't overload the system
        await asyncio.sleep(0.1)
        
    return results

# Example usage
texts = [
    "First text to check",
    "Second text to check",
    "Third text to check"
]

results = asyncio.run(process_batch(texts))
```

### Web API Integration

```python
from flask import Flask, request, jsonify
from aegis import Sentinel

app = Flask(__name__)
sentinel = Sentinel()

@app.route('/api/check', methods=['POST'])
def check_content():
    data = request.json
    text = data.get('text')
    intent = data.get('intent', 'scan')
    
    if not text:
        return jsonify({
            "error": "No text provided"
        }), 400
        
    try:
        # Check content safety
        result = sentinel.check(text, {
            "intent": intent,
            "risk_level": "medium"
        })
        
        if not result.allow:
            return jsonify({
                "allowed": False,
                "reason": result.reason,
                "severity": result.severity
            }), 403
            
        # Get full analysis
        analysis = sentinel.analyze(text, {
            "intent": intent,
            "risk_level": "medium"
        })
        
        return jsonify({
            "allowed": True,
            "analysis": analysis
        })
        
    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True)
```

### Custom Risk Rules

```python
from aegis import Sentinel
from typing import Dict, Any

class CustomSentinel(Sentinel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.risk_thresholds = {
            "low": 0.3,
            "medium": 0.6,
            "high": 0.8
        }
    
    def check(self, text: str, context: Dict[str, Any]):
        # Get base result
        result = super().check(text, context)
        
        # Apply custom rules
        risk_level = context.get("risk_level", "medium")
        threshold = self.risk_thresholds[risk_level]
        
        if result.severity > threshold:
            result.allow = False
            result.reason = f"Risk score {result.severity} exceeds threshold {threshold}"
            
        return result

# Usage
sentinel = CustomSentinel()
result = sentinel.check("Test text", {"risk_level": "low"})
```

## Testing Examples

### Unit Tests

```python
import pytest
from aegis import Sentinel

@pytest.fixture
def sentinel():
    return Sentinel()

def test_basic_safety_check(sentinel):
    result = sentinel.check("Safe text", {
        "intent": "test",
        "risk_level": "low"
    })
    assert result.allow
    assert result.severity < 0.5

def test_unsafe_content(sentinel):
    result = sentinel.check("Unsafe content here", {
        "intent": "test",
        "risk_level": "low"
    })
    assert not result.allow
    assert result.reason is not None
    assert result.severity > 0.5
```

These examples demonstrate common usage patterns and integration scenarios. See the full documentation for more details and advanced features.