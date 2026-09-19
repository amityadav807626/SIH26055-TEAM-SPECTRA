import numpy as np

from simulation.config import NUM_BANDS, DETECTOR_PD, DETECTOR_PFA


class SyntheticSpectrum:
    """Software-only synthetic spectrum environment used for demonstrations."""

    def __init__(self, n_bands=NUM_BANDS, seed=7):
        self.n_bands = n_bands
        self.rng = np.random.default_rng(seed)
        self.base = np.full(n_bands, 0.025, dtype=float)
        for band, probability in {5: 0.16, 13: 0.12, 22: 0.20, 31: 0.08, 2: 0.70, 9: 0.55, 17: 0.60}.items():
            if band < n_bands:
                self.base[band] = probability
        for band in [4, 11, 18, 25, 33, 37]:
            if band < n_bands:
                self.base[band] = 0.35

    def truth(self, time):
        values = []
        for band in range(self.n_bands):
            p = self.base[band] + 0.12 * np.sin(time / 30.0 + band * 0.35)
            if band == 2 and (time - 1) % 17 in range(0, 3): p = 0.92
            elif band == 9 and (time - 2) % 29 in range(0, 4): p = 0.88
            elif band == 17 and (time - 1) % 41 in range(0, 7): p = 0.90
            if band in [4, 11, 18, 25, 33, 37] and time % 5 in [band % 5, (band % 5) + 1]: p = max(p, 0.78)
            values.append(int(self.rng.random() < np.clip(p, 0.01, 0.95)))
        return np.asarray(values, dtype=np.int8)

    def detect(self, truth_value):
        probability = DETECTOR_PD if truth_value else DETECTOR_PFA
        return bool(self.rng.random() < probability)
