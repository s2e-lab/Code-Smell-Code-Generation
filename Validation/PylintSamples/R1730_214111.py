def _compute_std(self, C, mag, stddevs, idx):
        """
        Compute total standard deviation, as explained in table 2, page 67.
        """
        if mag > 8.0:
            mag = 8.0

        for stddev in stddevs:
            stddev[idx] += C['C4'] + C['C5'] * mag