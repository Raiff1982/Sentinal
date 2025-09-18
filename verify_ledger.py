"""
verify_ledger.py - Tamper-evident log integrity checker for SENTINAL

Usage:
    python verify_ledger.py path/to/ledger.jsonl SECRET_KEY

Checks HMAC signatures for each entry in the ledger file.
"""
import sys
import hmac
import hashlib
import json

if len(sys.argv) != 3:
    print("Usage: python verify_ledger.py path/to/ledger.jsonl SECRET_KEY")
    sys.exit(1)

ledger_path = sys.argv[1]
secret_key = sys.argv[2].encode()

with open(ledger_path, "r", encoding="utf-8") as f:
    ok = True
    for i, line in enumerate(f, 1):
        entry = json.loads(line)
        data = entry.get("data")
        sig = entry.get("hmac")
        if not data or not sig:
            print(f"Line {i}: Missing data or hmac.")
            ok = False
            continue
        msg = json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
        expected = hmac.new(secret_key, msg, hashlib.sha256).hexdigest()
        if not hmac.compare_digest(sig, expected):
            print(f"Line {i}: HMAC mismatch!")
            ok = False
    if ok:
        print("Ledger integrity: OK")
    else:
        print("Ledger integrity: FAILED")
