def min(self, axis=None, skipna=True, *args, **kwargs):
        """
        Return the minimum value of the Array or minimum along
        an axis.

        See Also
        --------
        numpy.ndarray.min
        Index.min : Return the minimum value in an Index.
        Series.min : Return the minimum value in a Series.
        """
        nv.validate_min(args, kwargs)
        nv.validate_minmax_axis(axis)

        result = nanops.nanmin(self.asi8, skipna=skipna, mask=self.isna())
        if isna(result):
            # Period._from_ordinal does not handle np.nan gracefully
            return NaT
        return self._box_func(result)