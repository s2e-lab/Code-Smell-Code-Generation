def rmsd(self, other, backbone=False):
        """Calculates the RMSD between two AMPAL objects.

        Notes
        -----
        No fitting operation is performs and both AMPAL objects must
        have the same number of atoms.

        Parameters
        ----------
        other : AMPAL Object
            Any AMPAL object with `get_atoms` method.
        backbone : bool, optional
            Calculates RMSD of backbone only.
        """
        assert type(self) == type(other)
        if backbone and hasattr(self, 'backbone'):
            points1 = self.backbone.get_atoms()
            points2 = other.backbone.get_atoms()
        else:
            points1 = self.get_atoms()
            points2 = other.get_atoms()
        points1 = [x._vector for x in points1]
        points2 = [x._vector for x in points2]
        return rmsd(points1=points1, points2=points2)