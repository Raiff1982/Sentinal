#!/usr/bin/env python3
import argparse, json, sys
from Sentinel import SignedLedger, PolicyStore, serve_api
from Sentinel import default_challenges, SafetyGate
from sentinel_council import get_council

def cmd_verify(args):
    led = SignedLedger()
    out = led.verify_all()
    print(json.dumps(out, indent=2))

def cmd_rollback(args):
    store = PolicyStore()
    idx = int(args.index)
    out = store.rollback(idx)
    print(json.dumps(out, indent=2))
    if not out.get("ok"):
        sys.exit(1)

def cmd_challenges(args):
    council = get_council()
    bank = default_challenges()
    gate = SafetyGate(council, bank, lambda: PolicyStore().load())
    # Run challenges but do not evaluate a live input
    res = gate.evaluate({"text": "noop", "_signals": {"bio": {"stress":0.2}, "env": {"context_risk":0.2}}, "timescale":0.2})
    # We consider "challenge_summary" as the main output
    print(json.dumps({"challenge_summary": res.get("challenge_summary"), "status": res.get("status")}, indent=2))

def cmd_serve(args):
    port = int(args.port)
    print(f"Serving audit API on :{port}")
    serve_api(port=port)

def main():
    p = argparse.ArgumentParser(prog="aegisctl", description="AEGIS control CLI")
    sub = p.add_subparsers(dest="cmd", required=True)
    s1 = sub.add_parser("verify", help="Verify all ledger files")
    s1.set_defaults(func=cmd_verify)
    s2 = sub.add_parser("rollback", help="Rollback policy to ledger index (-1 = latest)")
    s2.add_argument("index", help="Ledger entry index (int, -1 for latest)")
    s2.set_defaults(func=cmd_rollback)
    s3 = sub.add_parser("challenges", help="Run challenge bank via SafetyGate without live action")
    s3.set_defaults(func=cmd_challenges)
    s4 = sub.add_parser("serve", help="Run the audit API server")
    s4.add_argument("--port", default="8787")
    s4.set_defaults(func=cmd_serve)
    args = p.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
