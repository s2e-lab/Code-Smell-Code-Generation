def get_frequency_shift(
            self,
            grid_points,
            temperatures=np.arange(0, 1001, 10, dtype='double'),
            epsilons=None,
            output_filename=None):
        """Frequency shift from lowest order diagram is calculated.

        Args:
            epslins(list of float):
               The value to avoid divergence. When multiple values are given
               frequency shifts for those values are returned.

        """

        if self._interaction is None:
            self.set_phph_interaction()
        if epsilons is None:
            _epsilons = [0.1]
        else:
            _epsilons = epsilons
        self._grid_points = grid_points
        get_frequency_shift(self._interaction,
                            self._grid_points,
                            self._band_indices,
                            _epsilons,
                            temperatures,
                            output_filename=output_filename,
                            log_level=self._log_level)