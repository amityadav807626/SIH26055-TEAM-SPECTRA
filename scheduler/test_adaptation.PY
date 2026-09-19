from scheduler.smart_scheduler import SmartScheduler


def test_scheduler_can_shift_after_live_feedback():
    scheduler = SmartScheduler(seed=11)
    bands = list(range(4))
    for _ in range(12):
        scheduler.update_result(2, True)
    selected, scores, _ = scheduler.select_band(bands, 20, {}, epsilon=0.0)
    assert selected in bands
    assert scores[2]['probability'] > scores[0]['probability']
