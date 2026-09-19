class RoundRobinScheduler:

    def __init__(self, bands):
        self.bands = bands
        self.current_index = 0

    def select_band(self):
        band = self.bands[self.current_index]

        self.current_index = (
            self.current_index + 1
        ) % len(self.bands)

        return band