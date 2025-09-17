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


## Web UI
Start the web interface:
```bash
python webui/app.py
```
Open your browser to `http://localhost:5000`.

### Features
- Enter text or upload a file for scanning
- Use the chat interface for natural LLM responses
- See AI sentiment and Sentinal verdicts for each message

## Testing
Run all tests:
```bash
python -m unittest discover sentinal/tests
```
