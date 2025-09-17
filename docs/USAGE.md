# Usage Guide

## CLI
Run the hoax/misinformation scanner from the command line:

```bash
hoax-scan --text "Your text here"
hoax-scan --file path/to/file.txt
```

## Library
Import and use the filter in Python:

```python
from sentinal.hoax_filter import HoaxFilter
hf = HoaxFilter()
result = hf.score("Some text", url="https://example.com")
print(result)
```

## Testing
Run all tests:
```bash
python -m unittest discover sentinal/tests
```
