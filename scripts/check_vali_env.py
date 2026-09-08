"""Audit a validator's .vali_env for settings that cannot do what they look like.

Consensus keys decide points and the weights derived from them. A local OVERRIDE_ on
one is ignored, so leaving it in place is worse than useless: it reads like a pinned
value while the served one is what applies.

    python scripts/check_vali_env.py [path]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from alpharidge_ai import config  # noqa: E402

OVERRIDE = "OVERRIDE_"


def audit(path: Path):
    findings = []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key = stripped.split("=", 1)[0].strip()

        if key.startswith(OVERRIDE) and key[len(OVERRIDE):] in config._CONSENSUS_KEYS:
            findings.append((number, key,
                             "consensus key: ignored at runtime, remove it"))
        elif key in config._CONSENSUS_KEYS:
            findings.append((number, key,
                             "consensus key: the served value applies, remove it"))
    return findings


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else \
        Path(__file__).resolve().parent.parent / ".vali_env"
    if not path.exists():
        print(f"no file at {path}")
        return 1

    findings = audit(path)
    if not findings:
        print(f"{path}: clean")
        return 0

    print(f"{path}: {len(findings)} setting(s) to remove\n")
    for number, key, why in findings:
        print(f"  line {number}: {key}\n      {why}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
