from scheduler.smart_scheduler import SmartScheduler


def test_smart_scheduler_selects_valid_band_and_reports_latency():
    scheduler = SmartScheduler(seed=7)
    bands = list(range(10))
    observations = {
        2: {'previous_activity': 1, 'time_since_activity': 0, 'recent_activity_rate': 0.9},
        5: {'previous_activity': 0, 'time_since_activity': 4, 'recent_activity_rate': 0.1},
    }
    selected, scores, latency = scheduler.select_band(bands, 100, observations)
    assert selected in bands
    assert set(scores) == set(bands)
    assert all(0.0 <= item['probability'] <= 1.0 for item in scores.values())
    assert latency > 0
    assert scheduler.avg_decision_latency_us > 0


def test_discounted_beta_posterior_learns_online():
    scheduler = SmartScheduler(seed=1, discount=0.9)
    for _ in range(8):
        scheduler.update_result(3, True)
    for _ in range(2):
        scheduler.update_result(4, False)
    assert scheduler.posterior_mean(3) > 0.75
    assert scheduler.posterior_mean(4) < 0.45
