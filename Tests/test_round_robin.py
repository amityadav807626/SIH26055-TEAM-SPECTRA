import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scheduler.round_robin import RoundRobinScheduler


def test_round_robin_cycles():
    scheduler = RoundRobinScheduler([0, 1, 2])
    assert [scheduler.select_band() for _ in range(5)] == [0, 1, 2, 0, 1]
