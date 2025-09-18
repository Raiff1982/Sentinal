# nexus_guard_cli.py
import argparse, json, sys
from claim_guard import LiveClaimGuard

def main():
    p = argparse.ArgumentParser(description="Live Codette/Nexis guard with JPL SSD feeds")
    p.add_argument("--db", default="signals.db")
    p.add_argument("--source", default=None, help="Source URL (optional)")
    p.add_argument("--date-min", default="2000-01-01")
    p.add_argument("--date-max", default="2100-12-31")
    p.add_argument("text", nargs="*", help="Text to evaluate (or stdin)")
    args = p.parse_args()

    guard = LiveClaimGuard(db_path=args.db)
    text = " ".join(args.text) if args.text else sys.stdin.read()
    res = guard.evaluate(text, source_url=args.source, earth_window=(args.date_min, args.date_max))

    print(json.dumps(res, indent=2, ensure_ascii=False, sort_keys=True))

if __name__ == "__main__":
    main()
