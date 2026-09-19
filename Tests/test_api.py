import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from Backend.api.main import app
from Backend.database import clear_database


@pytest.fixture
def client():
    clear_database()
    with TestClient(app) as test_client:
        yield test_client
    clear_database()


def test_health(client):
    response = client.get('/api/health')
    assert response.status_code == 200
    assert response.json() == {'status': 'ok', 'service': 'SPECTRA API', 'mode': 'simulation-only'}


def test_frontend_is_served_by_fastapi(client):
    response = client.get('/')
    assert response.status_code == 200
    assert 'SPECTRA' in response.text
    assert 'app.js' in response.text


def test_frontend_assets(client):
    assert client.get('/styles.css').status_code == 200
    assert client.get('/app.js').status_code == 200


def test_experiments_endpoint(client):
    response = client.get('/api/experiments')
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_live_simulation_endpoint(client):
    response = client.post('/api/simulation/live', json={'bands': 10, 'time_slots': 100, 'epsilon': 0.1, 'seed': 7})
    assert response.status_code == 200
    data = response.json()
    assert data['experiment_id'] > 0
    assert len(data['band_history']) == 100
    assert len(data['final_rewards']) == 10
    assert len(data['visits']) == 10
    assert data['mode'] == 'simulation-only'
    assert data['avg_decision_latency_us'] > 0
    assert data['scenario'] == 'baseline'
    assert data['decision_latency_samples'] > 0

    detail = client.get(f"/api/experiments/{data['experiment_id']}")
    assert detail.status_code == 200
    assert len(detail.json()['band_learning']) == 10


def test_benchmark_endpoint_and_persistence(client):
    response = client.post('/api/simulation/benchmark', json={'bands': 10, 'time_slots': 100, 'epsilon': 0.1, 'seed': 7, 'trials': 3})
    assert response.status_code == 200
    data = response.json()
    assert data['run_id'] > 0
    assert data['trials_count'] == 3
    assert len(data['summary']) == 2
    for row in data['summary']:
        assert 'P_D_std' in row and row['P_D_std'] >= 0
        assert 'avg_intercept_time_std' in row and row['avg_intercept_time_std'] >= 0
        assert row['avg_intercept_time'] > 0
    assert len(data['trials']) == 6

    latest = client.get('/api/benchmark/latest')
    assert latest.status_code == 200
    assert latest.json()['id'] == data['run_id']



def test_agile_hopping_live_endpoint_exposes_recovery_and_blind_time(client):
    response = client.post('/api/simulation/live', json={
        'bands': 20, 'time_slots': 120, 'epsilon': 0.1, 'seed': 2026,
        'scenario': 'agile_hopping', 'hop_interval': 12
    })
    assert response.status_code == 200
    data = response.json()
    assert data['scenario'] == 'agile_hopping'
    assert data['hop_interval'] == 12
    assert data['avg_hop_recovery_time'] is not None
    assert 0.0 <= data['receiver_blind_time_fraction'] <= 1.0


def test_missing_experiment_is_404(client):
    response = client.get('/api/experiments/999999')
    assert response.status_code == 404

def test_live_runs_use_fresh_seed_by_default(client):
    a = client.post('/api/simulation/live', json={'bands': 10, 'time_slots': 100, 'epsilon': 0.1, 'seed': 0})
    b = client.post('/api/simulation/live', json={'bands': 10, 'time_slots': 100, 'epsilon': 0.1, 'seed': 0})
    assert a.status_code == 200 and b.status_code == 200
    da, db = a.json(), b.json()
    assert da['seed'] != db['seed']
    assert da['run_type'] == 'live-randomized'
    assert db['run_type'] == 'live-randomized'


def test_seeded_live_run_is_reproducible(client):
    a = client.post('/api/simulation/live', json={'bands': 10, 'time_slots': 100, 'epsilon': 0.1, 'seed': 12345}).json()
    b = client.post('/api/simulation/live', json={'bands': 10, 'time_slots': 100, 'epsilon': 0.1, 'seed': 12345}).json()
    assert a['seed'] == b['seed'] == 12345
    assert a['hits'] == b['hits']
    assert a['band_history'] == b['band_history']


def test_external_normalized_dataset_live_endpoint(client):
    response = client.post('/api/simulation/live', json={'bands': 10, 'time_slots': 100, 'epsilon': 0.1, 'seed': 7, 'data_mode': 'normalized-external'})
    assert response.status_code == 200
    data = response.json()
    assert data['data_mode'] == 'normalized-external'
    assert data['dataset'] == 'External Normalized Activity'
    assert data['data_source'].endswith('Backend/datasets/external_activity.csv')
    assert len(data['band_history']) == 100



def test_tsrd_activity_splits_live_and_benchmark(client):
    splits = ["train_scan", "val_scan", "test_scan", "train_stare", "val_stare", "test_stare"]
    for split in splits:
        live = client.post('/api/simulation/live', json={
            'bands': 10, 'time_slots': 100, 'epsilon': 0.1, 'seed': 7,
            'data_mode': 'tsrd-derived', 'tsrd_split': split
        })
        assert live.status_code == 200, (split, live.text)
        ld = live.json()
        assert ld['data_mode'] == 'tsrd-derived'
        assert ld['dataset'] == 'TSRD-Derived Activity Proxy'
        assert ld['data_source'].endswith(f'Backend/datasets/tsrd_activity/{split}.csv')
        bench = client.post('/api/simulation/benchmark', json={
            'bands': 10, 'time_slots': 100, 'epsilon': 0.1, 'seed': 7,
            'trials': 3, 'data_mode': 'tsrd-derived', 'tsrd_split': split
        })
        assert bench.status_code == 200, (split, bench.text)
        bd = bench.json()
        assert bd['data_mode'] == 'tsrd-derived'
        assert bd['tsrd_split'] == split
        assert len(bd['summary']) == 2


def test_jury_csv_upload_and_benchmark(client):
    csv_data = b"band,time_slot,activity\n0,0,1\n1,0,0\n2,1,1\n"
    upload = client.post('/api/datasets/jury-csv', files={'file': ('jury.csv', csv_data, 'text/csv')})
    assert upload.status_code == 200
    assert upload.json()['records'] == 3
    bench = client.post('/api/simulation/benchmark', json={
        'bands': 10, 'time_slots': 100, 'epsilon': 0.1, 'seed': 7,
        'trials': 3, 'data_mode': 'jury-csv'
    })
    assert bench.status_code == 200
    data = bench.json()
    assert data['data_mode'] == 'jury-csv'
    assert data['dataset'] == 'Jury Normalized Activity'
