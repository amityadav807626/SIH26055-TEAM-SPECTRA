import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from scheduler.smart_scheduler import SmartScheduler
from simulation.config import NUM_BANDS, DEFAULT_HORIZON, DEFAULT_SEED
from simulation.simulator import SyntheticSpectrum


def main():
    bands = list(range(NUM_BANDS))
    world = SyntheticSpectrum(NUM_BANDS, DEFAULT_SEED)
    scheduler = SmartScheduler(seed=DEFAULT_SEED + 17)
    observations = {}

    print("\n================================")
    print("SPECTRA / ADAPTIVE SIMULATION")
    print("================================")

    for time in range(DEFAULT_HORIZON):
        selected, scores, latency_us = scheduler.select_band(bands, time, observations)
        truth = bool(world.truth(time)[selected])
        observed = world.detect(truth)
        scheduler.update_result(selected, observed)
        old = observations.get(selected, {"previous_activity": 0, "time_since_activity": 5, "recent_activity_rate": 0.5})
        old["previous_activity"] = int(observed)
        old["time_since_activity"] = 0 if observed else old["time_since_activity"] + 1
        old["recent_activity_rate"] = 0.75 * old["recent_activity_rate"] + 0.25 * int(observed)
        observations[selected] = old
        if time < 20:
            print(f"Step {time:03d} | Band {selected:02d} | Truth: {'ACTIVE' if truth else 'INACTIVE'} | Observation: {'HIT' if observed else 'MISS'} | ML Probability: {scores[selected]['probability']:.2%} | Priority: {scores[selected]['score']:.3f} | Decision: {latency_us:.2f} µs")

    print("\nSimulation completed. Output is synthetic software data only.")


if __name__ == "__main__":
    main()
