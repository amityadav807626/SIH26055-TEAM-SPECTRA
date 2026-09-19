"""CLI benchmark utility for the online SPECTRA bandit scheduler."""
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from Backend.simulation_service import run_multi_benchmark


def main():
    result = run_multi_benchmark(30, 260, 0.10, 7, trials=10)
    output = PROJECT_ROOT / "evaluation" / "results.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved benchmark results to {output}")


if __name__ == "__main__":
    main()
