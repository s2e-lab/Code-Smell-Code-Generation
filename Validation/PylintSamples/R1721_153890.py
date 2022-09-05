def transform(self, maps):
        """ This function transforms from spherical to cartesian spins.

        Parameters
        ----------
        maps : a mapping object

        Examples
        --------
        Convert a dict of numpy.array:

        >>> import numpy
        >>> from pycbc import transforms
        >>> t = transforms.SphericalSpin1ToCartesianSpin1()
        >>> t.transform({'spin1_a': numpy.array([0.1]), 'spin1_azimuthal': numpy.array([0.1]), 'spin1_polar': numpy.array([0.1])})
            {'spin1_a': array([ 0.1]), 'spin1_azimuthal': array([ 0.1]), 'spin1_polar': array([ 0.1]),
             'spin2x': array([ 0.00993347]), 'spin2y': array([ 0.00099667]), 'spin2z': array([ 0.09950042])}

        Returns
        -------
        out : dict
            A dict with key as parameter name and value as numpy.array or float
            of transformed values.
        """
        a, az, po = self._inputs
        data = coordinates.spherical_to_cartesian(maps[a], maps[az], maps[po])
        out = {param : val for param, val in zip(self._outputs, data)}
        return self.format_output(maps, out)