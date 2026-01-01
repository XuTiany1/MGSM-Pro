#!/usr/bin/env python3
import json
import argparse
from pathlib import Path

def main():
    ap = argparse.ArgumentParser(description="Print IDs missing any combination_1..N.")
    ap.add_argument("--json-file", "-j", type=Path, required=True, help="Path to the JSON file.")
    ap.add_argument("--expected-combos", "-n", type=int, default=10,
                    help="Require combination_1..combination_N to exist. Default: 10")
    args = ap.parse_args()

    with args.json_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    needed = {f"combination_{i}" for i in range(1, args.expected_combos + 1)}

    for rec in data:
        if not isinstance(rec, dict):
            continue
        missing_any = any(k not in rec for k in needed)
        if missing_any:
            rid = rec.get("id", "<no-id>")
            print(rid)

if __name__ == "__main__":
    main()
