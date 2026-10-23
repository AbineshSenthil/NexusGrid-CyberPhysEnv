"""Regenerate the validated Task 2 contingency catalog (server/data/cascade_catalog.json).

Run this after changing the network or the physics:

    python scripts/build_scenario_catalog.py --count 240
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from server.scenarios import CATALOG_PATH, generate_cascade_catalog  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=240)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    start = time.time()
    configs = generate_cascade_catalog(count=args.count, seed=args.seed)
    CATALOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "description": "Validated N-1 contingencies for Task 2 (cascade_overload). The first 80% serve "
                       "training seeds, the last 20% serve held-out evaluation seeds (>= 10000).",
        "generator_seed": args.seed,
        "count": len(configs),
        "configs": configs,
    }
    CATALOG_PATH.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    trips = Counter(c["trip_line"] for c in configs)
    print(f"wrote {len(configs)} configs to {CATALOG_PATH} in {time.time() - start:.1f}s")
    print("trip lines:", dict(trips.most_common()))


if __name__ == "__main__":
    main()
