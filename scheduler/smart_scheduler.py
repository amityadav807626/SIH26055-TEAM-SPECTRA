"""Online bandit scheduler for the software-only SPECTRA simulation."""
from __future__ import annotations

import time
from typing import Any

import numpy as np


class SmartScheduler:
    """Discounted Thompson Sampling (Beta-Bernoulli) scheduler.

    The scheduler starts with an uninformative Beta(1, 1) prior for every band
    and updates that posterior online after each synthetic observation. A
    discount factor lets old observations fade, so the policy can adapt when
    the synthetic environment changes (for example, agile hopping).
    """

    def __init__(self, model_path: Any = None, discount: float = 0.90, seed: int | None = None):
        # model_path is retained as a harmless compatibility argument for older
        # callers; no model is loaded and no .joblib artifact is required.
        self.discount = float(np.clip(discount, 0.50, 0.999))
        self.rng = np.random.default_rng(seed)
        self.last_scan: dict[int, int] = {}
        self.hits: dict[int, int] = {}
        self.scans: dict[int, int] = {}
        self.alpha: dict[int, float] = {}
        self.beta: dict[int, float] = {}
        self.decision_count = 0
        self.total_decision_latency_us = 0.0
        self.avg_decision_latency_us = 0.0

    def _ensure_band(self, band: int) -> None:
        if band not in self.alpha:
            self.alpha[band] = 1.0
            self.beta[band] = 1.0

    def posterior_mean(self, band: int) -> float:
        self._ensure_band(band)
        a, b = self.alpha[band], self.beta[band]
        return float(a / (a + b))

    def predict_activity(self, band: int, time: int, previous_activity: int = 0,
                         time_since_activity: int = 5, recent_activity_rate: float = 0.5) -> float:
        """Return the current Beta posterior mean.

        Observation features are accepted for API compatibility with the
        previous RF implementation. Learning is now entirely online.
        """
        del time, previous_activity, time_since_activity, recent_activity_rate
        return self.posterior_mean(band)

    def calculate_priority(self, probability: float, freshness: float, hit_rate: float,
                           exploration_bonus: float = 0.0) -> float:
        uncertainty = 1.0 - abs(probability - 0.5) * 2.0
        return float(
            0.45 * probability
            + 0.15 * uncertainty
            + 0.15 * freshness
            + 0.10 * hit_rate
            + 0.15 * exploration_bonus
        )

    def select_band(self, bands, time_slot: int, observations: dict[int, dict[str, Any]] | None = None,
                    epsilon: float = 0.0):
        """Select a band with discounted Thompson Sampling.

        Returns ``(selected_band, scores, decision_latency_us)``. Latency is
        measured around the complete decision path and maintained as a running
        average on ``avg_decision_latency_us``.
        """
        # Keep the measurement local to the complete decision path.
        t0 = time.perf_counter()
        band_list = list(bands)
        if not band_list:
            raise ValueError("bands must not be empty")
        observations = observations or {}
        scores: dict[int, dict[str, float]] = {}
        samples: dict[int, float] = {}

        for band in band_list:
            self._ensure_band(int(band))
            obs = observations.get(int(band), {})
            posterior = self.posterior_mean(int(band))
            sample = float(self.rng.beta(self.alpha[int(band)], self.beta[int(band)]))
            samples[int(band)] = sample
            last_scan = self.last_scan.get(int(band), -1)
            freshness = 1.0 if last_scan < 0 else min(max((time_slot - last_scan) / 10.0, 0.0), 1.0)
            scans = self.scans.get(int(band), 0)
            hits = self.hits.get(int(band), 0)
            hit_rate = hits / scans if scans else 0.0
            exploration_bonus = 1.0 / np.sqrt(1.0 + scans)
            score = self.calculate_priority(sample, freshness, hit_rate, exploration_bonus)
            scores[int(band)] = {
                "probability": posterior,
                "sample": sample,
                "score": score,
                "alpha": self.alpha[int(band)],
                "beta": self.beta[int(band)],
            }

        if float(epsilon) > 0 and self.rng.random() < float(epsilon):
            min_visits = min(self.scans.get(int(b), 0) for b in band_list)
            candidates = [int(b) for b in band_list if self.scans.get(int(b), 0) == min_visits]
            selected_band = int(self.rng.choice(candidates))
        else:
            selected_band = max(scores, key=lambda b: scores[b]["score"])

        self.last_scan[selected_band] = int(time_slot)
        self.scans[selected_band] = self.scans.get(selected_band, 0) + 1
        latency_us = (time.perf_counter() - t0) * 1e6
        self.decision_count += 1
        self.total_decision_latency_us += latency_us
        self.avg_decision_latency_us = self.total_decision_latency_us / self.decision_count
        return selected_band, scores, float(latency_us)

    def update_result(self, band: int, activity: bool) -> None:
        """Apply a discounted Beta-Bernoulli update to one band."""
        self._ensure_band(int(band))
        a = 1.0 + self.discount * (self.alpha[int(band)] - 1.0)
        b = 1.0 + self.discount * (self.beta[int(band)] - 1.0)
        if activity:
            a += 1.0
        else:
            b += 1.0
        self.alpha[int(band)] = a
        self.beta[int(band)] = b
        if activity:
            self.hits[int(band)] = self.hits.get(int(band), 0) + 1
