
# Security Policy

## Reporting Issues
Please report security vulnerabilities by opening an issue or contacting the maintainer directly.

## Dependencies
- All dependencies are open source and regularly updated.
- Automated CI/CD checks for vulnerabilities.

## Data Handling
- SQLite memory is integrity-checked.
- Sensitive data is not stored.

## Admin Features & Data Export
- Admin-only features (analytics, batch, feedback, export, retrain) are restricted to authenticated users with the 'admin' role.
- Data export (interactions, feedback, analytics) is only available to admins.

## Security Model
- All code that executes model or user input runs inside a sandboxed environment (Web Worker, container, or isolated service).
- No API keys or secrets are shipped inside public code or zips. All secrets must be supplied via environment variables at runtime.
- All network communication uses HTTPS only.
- Content rendered as HTML is sanitized through a strict allowlist sanitizer (DOMPurify or equivalent).
- Tokens are not stored in localStorage/sessionStorage; use ephemeral in-memory storage.

## Responsible Disclosure
We appreciate responsible disclosure and will address issues promptly.