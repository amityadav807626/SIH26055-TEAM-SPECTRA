import numpy as np

from Backend.simulation_service import create_environment, detector, run_live


def test_recovery_time_after_target_frequency_hops():
    result = run_live(20, 120, 0.10, 2026, scenario="agile_hopping", hop_interval=12)
    recovery = result["hop_recovery_times"]
    assert recovery
    assert all(1.0 <= value <= 12.0 for value in recovery)
    assert result["avg_hop_recovery_time"] > 0


def test_pulse_collision_throughput():
    # Two simultaneous synthetic pulses compete for a single-band receiver.
    truth = np.zeros((1, 4), dtype=np.int8)
    truth[0, 1] = 1
    truth[0, 2] = 1
    rng = np.random.default_rng(7)
    observed = detector(rng, True, pd_=1.0, pfa=0.0)
    scanned_pulses = int(observed)
    throughput = scanned_pulses / int(truth[0].sum())
    assert truth[0].sum() == 2
    assert throughput == 0.5
    assert scanned_pulses <= 1


def test_receiver_blind_time_degradation_after_hopping():
    baseline = run_live(20, 120, 0.10, 2026, scenario="baseline")
    agile = run_live(20, 120, 0.10, 2026, scenario="agile_hopping", hop_interval=12)
    assert baseline["receiver_blind_time_fraction"] == 0.0
    assert 0.0 <= agile["receiver_blind_time_fraction"] <= 1.0
    assert agile["receiver_blind_time_fraction"] > baseline["receiver_blind_time_fraction"]
