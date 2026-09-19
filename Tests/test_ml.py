from scheduler.smart_scheduler import SmartScheduler


def test_online_bandit_replaces_offline_model():
    scheduler = SmartScheduler(seed=3)
    scheduler.update_result(1, True)
    scheduler.update_result(1, True)
    assert scheduler.posterior_mean(1) > 0.5
    assert not hasattr(scheduler, "model")
