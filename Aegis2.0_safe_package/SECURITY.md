# Security Policy for Project SENTINAL

## Supported Versions
Only the latest published Zenodo release is supported for operational use.

## Reporting a Vulnerability
If you discover a vulnerability, please **do not post publicly**. 
Contact the author (Jonathan Harrison) directly via the contact information in the Zenodo record.

## Security Model
- All code that executes model or user input must run inside a **sandboxed environment** (Web Worker, container, or isolated service).
- No API keys or secrets are shipped inside public code or zips. All secrets must be supplied via environment variables at runtime.
- All network communication must use **HTTPS** only.
- Content rendered as HTML must be sanitized through a strict allowlist sanitizer (DOMPurify or equivalent).
- Tokens must **not** be stored in localStorage/sessionStorage; use ephemeral in-memory storage.

## Export Control
Users are responsible for ensuring compliance with their jurisdiction’s laws when deploying this system.
