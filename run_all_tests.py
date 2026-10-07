"""
EYEON - Master Test Suite & Validation Runner
Runs all unit test modules and system benchmarks in sequence.
"""

import subprocess
import sys
from pathlib import Path


DIVIDER = "=" * 60

def run_test(script_path: str) -> bool:
    print(f"\n{DIVIDER}")
    print(f"  RUNNING: {script_path}")
    print(f"{DIVIDER}")

    cmd = [sys.executable, script_path]
    res = subprocess.run(cmd)

    if res.returncode == 0:
        print(f"  [PASS] {script_path}")
        return True
    else:
        print(f"  [FAIL] {script_path} (exit code {res.returncode})")
        return False


def main():
    test_scripts = [
        "evaluation/test_database.py",
        "evaluation/test_alerts.py",
        "evaluation/test_zones.py",
        "evaluation/test_positioning.py",
        "evaluation/test_risk.py",
        "evaluation/test_association.py",
        "evaluation/benchmark.py",
    ]

    passed = 0
    failed = 0

    print("=" * 60)
    print("  EYEON MASTER VALIDATION SUITE")
    print("=" * 60)

    for script in test_scripts:
        if run_test(script):
            passed += 1
        else:
            failed += 1

    print("\n" + "=" * 60)
    print(f"  SUMMARY RESULTS: {passed} PASSED, {failed} FAILED out of {len(test_scripts)} suites")
    print("=" * 60)

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
