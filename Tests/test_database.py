import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from Backend.database import clear_database, get_experiment, save_experiment


def test_save_and_read_experiment():
    clear_database()
    experiment_id = save_experiment(
        'Test Scheduler',
        4,
        10,
        {
            'hits': 3,
            'P_D': 0.75,
            'P_FA': 0.05,
            'precision': 0.80,
            'recall': 0.75,
            'F1': 0.774,
            'avg_intercept_time': 1.5,
            'final_rewards': [0.1, 0.2],
            'visits': [5, 5],
            'hits_per_band': [1, 2],
        },
        dataset="Example Normalized Activity",
        dataset_version="v1",
        data_mode="normalized-external",
        data_source="example://normalized-activity",
    )
    row = get_experiment(experiment_id)
    assert row is not None
    assert row['scheduler'] == 'Test Scheduler'
    assert row['hits'] == 3
    assert row['dataset'] == 'Example Normalized Activity'
    assert row['dataset_version'] == 'v1'
    assert row['data_mode'] == 'normalized-external'
    assert row['data_source'] == 'example://normalized-activity'
    clear_database()
