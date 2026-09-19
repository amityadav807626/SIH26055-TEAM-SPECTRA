from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

from scheduler.smart_scheduler import SmartScheduler
from Backend.datasets.activity_adapter import ActivityDatasetAdapter


def _agile_channel_sets(n_bands: int) -> list[list[int]]:
    groups = [
        [2, 9, 17], [4, 11, 18], [5, 13, 22],
        [6, 15, 24], [8, 19, 27], [10, 21, 29],
    ]
    return [[b for b in group if b < n_bands] for group in groups if any(b < n_bands for b in group)]


def create_environment(n_bands: int, horizon: int, seed: int, scenario: str = "baseline", hop_interval: int = 12) -> np.ndarray:
    """Create a deterministic synthetic spectrum world.

    ``agile_hopping`` moves a compact target set to a pseudo-random channel
    group every ``hop_interval`` slots. The scenario is simulation-only and
    represents abstract channel occupancy, not real RF emitter behavior.
    """
    rng = np.random.default_rng(seed)
    truth = np.zeros((horizon, n_bands), dtype=np.int8)
    base = np.full(n_bands, 0.025, dtype=float)
    for band, probability in {5: .16, 13: .12, 22: .20, 31: .08, 2: .70, 9: .55, 17: .60}.items():
        if band < n_bands:
            base[band] = probability
    for band in [4, 11, 18, 25, 33, 37]:
        if band < n_bands:
            base[band] = .35

    for t in range(horizon):
        for band in range(n_bands):
            p = base[band] + .12 * np.sin(t / 30.0 + band * .35)
            if band == 2 and (t - 1) % 17 in range(0, 3):
                p = .92
            elif band == 9 and (t - 2) % 29 in range(0, 4):
                p = .88
            elif band == 17 and (t - 1) % 41 in range(0, 7):
                p = .90
            if band in [4, 11, 18, 25, 33, 37] and t % 5 in [band % 5, (band % 5) + 1]:
                p = max(p, .78)
            truth[t, band] = int(rng.random() < np.clip(p, .01, .95))

    if scenario == "agile_hopping":
        hop_interval = max(1, int(hop_interval))
        groups = _agile_channel_sets(n_bands)
        hop_rng = np.random.default_rng(seed + 7919)
        order = hop_rng.permutation(len(groups))
        for epoch, start in enumerate(range(0, horizon, hop_interval)):
            group = groups[int(order[epoch % len(order)])]
            end = min(start + hop_interval, horizon)
            # Clear the selected target group first, then create a high-activity
            # compact pulse window. This makes hops observable in the truth map.
            for b in group:
                truth[start:end, b] = (hop_rng.random(end - start) < .88).astype(np.int8)
        # Suppress the legacy persistent hot bands during this scenario so the
        # online learner must react to movement rather than memorizing them.
        for b in range(n_bands):
            if b not in {x for g in groups for x in g}:
                truth[:, b] = (rng.random(horizon) < .015).astype(np.int8)
    elif scenario != "baseline":
        raise ValueError(f"Unsupported scenario: {scenario}")
    return truth


def detector(rng: np.random.Generator, truth_value: bool, pd_: float = .90, pfa: float = .05) -> bool:
    return bool(rng.random() < (pd_ if truth_value else pfa))


def metrics(tp: int, fp: int, fn: int, tn: int, delays: list[float]) -> dict[str, Any]:
    total = tp + fp + fn + tn
    actual_pos = tp + fn
    actual_neg = fp + tn
    predicted_pos = tp + fp
    precision = tp / predicted_pos if predicted_pos else 0.0
    recall = tp / actual_pos if actual_pos else 0.0
    pfa = fp / actual_neg if actual_neg else 0.0
    accuracy = (tp + tn) / total if total else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "accuracy": accuracy, "P_D": recall, "P_FA": pfa,
        "precision": precision, "recall": recall, "F1": f1,
        "avg_intercept_time": float(np.mean(delays)) if delays else None,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


class Learner:
    def __init__(self, n: int):
        self.reward = np.zeros(n, dtype=float)
        self.visits = np.zeros(n, dtype=int)
        self.hits = np.zeros(n, dtype=int)

    def update(self, band: int, hit: bool) -> None:
        reward = 1.0 if hit else -0.08
        self.reward[band] = .88 * self.reward[band] + .12 * reward
        self.visits[band] += 1
        self.hits[band] += int(hit)


def score_bands(scheduler: SmartScheduler, learner: "Learner", observations: dict[int, dict[str, Any]], bands: range, t: int, epsilon: float = 0.0) -> tuple[int, dict[int, dict[str, float]], float]:
    """Delegate online band selection to the scheduler and return telemetry."""
    return scheduler.select_band(bands, t, observations, epsilon=epsilon)


def _initial_observations(n: int) -> dict[int, dict[str, Any]]:
    return {b: {"previous_activity": 0, "time_since_activity": 5, "recent_activity_rate": .5} for b in range(n)}


def run_live(n_bands: int, horizon: int, epsilon: float, seed: int, detector_pd: float = .90, detector_pfa: float = .05, progress_callback: Callable[[dict[str, Any]], None] | None = None, scenario: str = "baseline", hop_interval: int = 12, activity_grid: np.ndarray | None = None, source_mode: str | None = None, source_name: str | None = None, source_version: str | None = None, source_detail: str | None = None) -> dict[str, Any]:
    horizon = min(int(horizon), 260)
    n_bands = int(n_bands)
    epsilon = float(epsilon)
    rng = np.random.default_rng(seed)
    if activity_grid is None:
        truth = create_environment(n_bands, horizon, seed, scenario=scenario, hop_interval=hop_interval)
        source_mode = "simulation"
        source_name = "SPECTRA Synthetic Activity"
        source_version = "internal-v1"
        source_detail = "SPECTRA synthetic environment"
    else:
        truth = np.asarray(activity_grid, dtype=np.int8)
        if truth.shape != (horizon, n_bands):
            raise ValueError(f"external activity grid must have shape {(horizon, n_bands)}, got {truth.shape}")
        if not np.isin(truth, [0, 1]).all():
            raise ValueError("external activity grid must contain only 0/1 activity values")
        source_mode = source_mode or "normalized-external"
        source_name = source_name or "External Normalized Activity"
        source_version = source_version or "adapter-v1"
        source_detail = source_detail or "Backend/datasets/external_activity.csv"
        scenario = "external_dataset" if source_mode != "simulation" else scenario
    scheduler = SmartScheduler(seed=seed + 17)
    learner = Learner(n_bands)
    observations = _initial_observations(n_bands)
    last = np.full(n_bands, -1, dtype=int)
    active_start = {b: None for b in range(n_bands)}
    episode_intercepted = {b: False for b in range(n_bands)}
    events: list[dict[str, Any]] = []
    band_history: list[int] = []
    performance_history: list[float] = []
    system_pd_history: list[float] = []
    reward_history: list[list[float]] = []
    visits_history: list[list[int]] = []
    tp = fp = fn = tn = 0
    hits = 0
    delays: list[float] = []
    decision_latencies_us: list[float] = []
    detection_history: list[bool] = []

    for t in range(horizon):
        if t < n_bands:
            band = t
            priority = 1.0
            prediction = None
            decision_latency_us = 0.0
        else:
            band, scores, decision_latency_us = score_bands(scheduler, learner, observations, range(n_bands), t, epsilon=epsilon)
            priority = float(scores[band]["score"])
            prediction = float(scores[band]["probability"])

        actual = bool(truth[t, band])
        if decision_latency_us > 0:
            decision_latencies_us.append(decision_latency_us)
        detected = detector(rng, actual, detector_pd, detector_pfa)
        hits += int(detected)
        detection_history.append(bool(detected))
        last[band] = t
        band_history.append(band)

        # Track the true activity episode for this band, not just the last scan.
        # This makes delay a real first-intercept metric: onset -> first successful scan.
        if actual and (t == 0 or not truth[t - 1, band]):
            active_start[band] = t
            episode_intercepted[band] = False
        if actual and detected and active_start[band] is not None and not episode_intercepted[band]:
            delays.append((t - active_start[band]) + 1)
            episode_intercepted[band] = True

        if actual and detected: tp += 1
        elif not actual and detected: fp += 1
        elif actual: fn += 1
        else: tn += 1

        learner.update(band, detected)
        scheduler.update_result(band, detected)
        for b in range(n_bands):
            observations[b]["time_since_activity"] = min(observations[b]["time_since_activity"] + 1, 50)
        observations[band]["previous_activity"] = int(detected)
        if detected:
            observations[band]["time_since_activity"] = 0
        observations[band]["recent_activity_rate"] = .90 * observations[band]["recent_activity_rate"] + .10 * int(detected)

        event = {"t": t, "band": band, "detected": detected, "actual": actual, "priority": priority, "prediction": prediction}
        events.insert(0, event)
        events = events[:8]
        performance_history.append(hits / (t + 1) * 100)
        active_so_far = int(truth[:t + 1, :].sum())
        system_pd_history.append((tp / active_so_far * 100) if active_so_far else 0.0)
        reward_history.append([round(float(x), 5) for x in learner.reward])
        visits_history.append([int(x) for x in learner.visits])

        if progress_callback is not None:
            progress_callback({
                "type": "progress", "slot": t + 1, "total_slots": horizon,
                "band": band, "detected": bool(detected), "hits": hits,
                "tp": tp, "active_opportunities": active_so_far,
                "system_pd": (tp / active_so_far) if active_so_far else 0.0,
                "coverage": float(np.count_nonzero(learner.visits) / n_bands),
                "reward": float(learner.reward[band]),
                "decision_latency_us": float(scheduler.avg_decision_latency_us),
            })

    m = metrics(tp, fp, fn, tn, delays)
    total_active_opportunities = int(truth.sum())
    system_pd = (tp / total_active_opportunities) if total_active_opportunities else 0.0
    hop_recovery_times: list[float] = []
    receiver_blind_time_fraction = 0.0
    hop_recovery_count = 0
    if scenario == "agile_hopping":
        hop_stats = agile_telemetry(truth, band_history, detection_history, seed, hop_interval)
        hop_recovery_times = [] if hop_stats["avg_hop_recovery_time"] is None else [hop_stats["avg_hop_recovery_time"]]
        receiver_blind_time_fraction = float(hop_stats["receiver_blind_time_fraction"] or 0.0)
        hop_recovery_count = int(hop_stats["hop_recovery_count"])
    return {
        "mode": "simulation-only", "n_bands": n_bands, "horizon": horizon, "epsilon": epsilon, "seed": int(seed),
        "hits": hits, "scans": horizon, "coverage": float(np.count_nonzero(learner.visits) / n_bands),
        "P_D": system_pd, "P_D_scanned": m["P_D"], "P_FA": m["P_FA"], "accuracy": m["accuracy"], "precision": m["precision"], "recall": m["recall"], "F1": m["F1"],
        "avg_intercept_time": m["avg_intercept_time"], "avg_decision_latency_us": float(scheduler.avg_decision_latency_us), "decision_latency_samples": len(decision_latencies_us), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "band_history": band_history, "performance_history": performance_history, "system_pd_history": system_pd_history,
        "reward_history": reward_history, "visits_history": visits_history,
        "events": events,
        "scenario": scenario, "hop_interval": int(hop_interval),
        "dataset": source_name, "dataset_version": source_version, "data_mode": source_mode, "data_source": source_detail,
        "hop_recovery_times": hop_recovery_times,
        "avg_hop_recovery_time": float(np.mean(hop_recovery_times)) if hop_recovery_times else None,
        "hop_recovery_count": int(hop_recovery_count),
        "receiver_blind_time_fraction": float(receiver_blind_time_fraction),
        "useful_detections_per_100_scans": float(tp / max(horizon, 1) * 100.0),
        "final_rewards": learner.reward.tolist(), "visits": learner.visits.tolist(), "hits_per_band": learner.hits.tolist(),
    }


def agile_telemetry(truth: np.ndarray, band_history: list[int], detection_history: list[bool], seed: int, hop_interval: int) -> dict[str, Any]:
    """Measure synthetic target-hop recovery and receiver blind time.

    Recovery is the number of slots from a hop boundary until the first
    successful observation on one of the newly active target channels.
    Blind time is the fraction of target-active slots where the scheduler
    selected outside the current target channel set.
    """
    n_bands = truth.shape[1]
    horizon = truth.shape[0]
    groups = _agile_channel_sets(n_bands)
    if not groups:
        return {"avg_hop_recovery_time": None, "receiver_blind_time_fraction": None, "hop_recovery_count": 0}
    hop_interval = max(1, int(hop_interval))
    hop_rng = np.random.default_rng(seed + 7919)
    order = hop_rng.permutation(len(groups))
    recoveries: list[float] = []
    blind_slots = 0
    target_slots = 0
    for epoch, start in enumerate(range(0, horizon, hop_interval)):
        group = groups[int(order[epoch % len(order)])]
        end = min(start + hop_interval, horizon)
        recovered = None
        for t in range(start, end):
            target_active = bool(truth[t, group].any())
            target_slots += int(target_active)
            if target_active and band_history[t] not in group:
                blind_slots += 1
            if recovered is None and band_history[t] in group and truth[t, band_history[t]] and detection_history[t]:
                recovered = t - start + 1
        if recovered is not None:
            recoveries.append(float(recovered))
    return {
        "avg_hop_recovery_time": float(np.mean(recoveries)) if recoveries else None,
        "receiver_blind_time_fraction": float(blind_slots / max(target_slots, 1)),
        "hop_recovery_count": len(recoveries),
    }


def run_benchmark(n_bands: int, horizon: int, epsilon: float, seed: int, detector_pd: float = .90, detector_pfa: float = .05, scenario: str = "baseline", hop_interval: int = 12, activity_grid: np.ndarray | None = None) -> dict[str, Any]:
    if activity_grid is None:
        truth = create_environment(n_bands, horizon, seed, scenario=scenario, hop_interval=hop_interval)
    else:
        truth = np.asarray(activity_grid, dtype=np.int8)
        if truth.shape != (horizon, n_bands):
            raise ValueError(f"activity_grid shape must be {(horizon, n_bands)}, got {truth.shape}")
        if not np.isin(truth, [0, 1]).all():
            raise ValueError("activity_grid must contain only 0/1 values")
    rng = np.random.default_rng(seed + 10000)
    detection_draws = rng.random((horizon, n_bands))
    false_alarm_draws = rng.random((horizon, n_bands))
    onset = np.full((horizon, n_bands), -1, dtype=int)
    for b in range(n_bands):
        current = -1
        for t in range(horizon):
            if truth[t, b] and (t == 0 or not truth[t - 1, b]): current = t
            onset[t, b] = current
            if not truth[t, b]: current = -1

    rows = []
    for strategy in ["Smart Scan", "Round Robin"]:
        learner = Learner(n_bands)
        observations = _initial_observations(n_bands)
        scheduler = SmartScheduler(seed=seed + 17) if strategy == "Smart Scan" else None
        rr_index = 0
        tp = fp = fn = tn = 0
        delays: list[float] = []
        band_scans = np.zeros(n_bands, dtype=int)
        band_detections = np.zeros(n_bands, dtype=int)
        selected_history: list[int] = []
        detection_history: list[bool] = []
        local_rng = np.random.default_rng(seed + (50000 if strategy == "Smart Scan" else 60000))

        for t in range(horizon):
            if strategy == "Smart Scan":
                if t < n_bands: band = t
                else:
                    band, scores, _latency = score_bands(scheduler, learner, observations, range(n_bands), t, epsilon=epsilon)
            else:
                band = rr_index; rr_index = (rr_index + 1) % n_bands
            band_scans[band] += 1
            selected_history.append(int(band))
            actual = bool(truth[t, band])
            detected = bool(detection_draws[t, band] < detector_pd) if actual else bool(false_alarm_draws[t, band] < detector_pfa)
            band_detections[band] += int(detected)
            detection_history.append(bool(detected))
            if actual and detected and onset[t, band] >= 0: delays.append((t - onset[t, band]) + 1)
            if actual and detected: tp += 1
            elif not actual and detected: fp += 1
            elif actual: fn += 1
            else: tn += 1
            learner.update(band, detected)
            if scheduler: scheduler.update_result(band, detected)
            for b in range(n_bands): observations[b]["time_since_activity"] = min(observations[b]["time_since_activity"] + 1, 50)
            observations[band]["previous_activity"] = int(detected)
            if detected: observations[band]["time_since_activity"] = 0
            observations[band]["recent_activity_rate"] = .90 * observations[band]["recent_activity_rate"] + .10 * int(detected)

        m = metrics(tp, fp, fn, tn, delays)
        total_active_opportunities = int(truth.sum())
        system_pd = (tp / total_active_opportunities) if total_active_opportunities else 0.0
        hop_stats = agile_telemetry(truth, selected_history, detection_history, seed, hop_interval) if scenario == "agile_hopping" else {"avg_hop_recovery_time": None, "receiver_blind_time_fraction": None, "hop_recovery_count": 0}
        rows.append({"Strategy": strategy, "accuracy": m["accuracy"], "P_D": system_pd, "P_D_scanned": m["P_D"], "P_FA": m["P_FA"], "precision": m["precision"], "recall": m["recall"], "F1": m["F1"], "avg_intercept_time": m["avg_intercept_time"], "Scans": int(horizon), "Detections": int(band_detections.sum()), "TP": tp, "FP": fp, "FN": fn, "TN": tn, "Coverage": float(np.count_nonzero(band_scans) / n_bands), "useful_detections_per_100_scans": float(tp / max(horizon, 1) * 100.0), "avg_hop_recovery_time": hop_stats["avg_hop_recovery_time"], "receiver_blind_time_fraction": hop_stats["receiver_blind_time_fraction"], "hop_recovery_count": hop_stats["hop_recovery_count"], "band_scans": band_scans.tolist(), "band_detections": band_detections.tolist()})
    return rows


def run_multi_benchmark(n_bands: int, horizon: int, epsilon: float, seed: int, trials: int = 10, scenario: str = "baseline", hop_interval: int = 12, activity_grid: np.ndarray | None = None) -> dict[str, Any]:
    trial_rows = []
    for trial in range(trials):
        trial_seed = int(seed) + trial * 1009
        for row in run_benchmark(n_bands, horizon, epsilon, trial_seed, scenario=scenario, hop_interval=hop_interval, activity_grid=activity_grid):
            row = dict(row); row["Trial"] = trial + 1; row["Seed"] = trial_seed; trial_rows.append(row)
    trial_df = pd.DataFrame(trial_rows)
    numeric = ["accuracy", "P_D", "P_FA", "precision", "recall", "F1", "avg_intercept_time", "Scans", "Detections", "TP", "FP", "FN", "TN", "Coverage", "useful_detections_per_100_scans", "avg_hop_recovery_time", "receiver_blind_time_fraction", "hop_recovery_count"]
    grouped = trial_df.groupby("Strategy", as_index=False)[numeric]
    summary = grouped.mean()
    # Sample standard deviation (ddof=1) exposes run-to-run variability rather than
    # implying that the mean alone is a statistically complete validation.
    std_df = trial_df.groupby("Strategy")[numeric].std(ddof=1).fillna(0.0).reset_index()
    std_df = std_df.rename(columns={c: f"{c}_std" for c in numeric})
    summary = summary.merge(std_df, on="Strategy", how="left")
    wins = {}
    for metric in ["P_D", "precision", "recall", "F1"]:
        wins[metric] = int(sum(trial_df[trial_df.Trial == t].set_index("Strategy").loc["Smart Scan", metric] > trial_df[trial_df.Trial == t].set_index("Strategy").loc["Round Robin", metric] for t in range(1, trials + 1)))
    pfa_wins = int(sum(trial_df[trial_df.Trial == t].set_index("Strategy").loc["Smart Scan", "P_FA"] < trial_df[trial_df.Trial == t].set_index("Strategy").loc["Round Robin", "P_FA"] for t in range(1, trials + 1)))
    summary_records = []
    for record in summary.to_dict(orient="records"):
        summary_records.append({k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in record.items()})
    trial_records = []
    for record in trial_df.to_dict(orient="records"):
        trial_records.append({k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in record.items()})
    return {"summary": summary_records, "trials": trial_records, "smart_wins": wins, "pfa_wins": pfa_wins, "trials_count": trials, "scenario": scenario, "hop_interval": int(hop_interval), "data_mode": "normalized-external" if activity_grid is not None else "simulation"}
