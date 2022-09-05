def repertoire(self, direction, mechanism, purview):
        """Return the cause or effect repertoire function based on a direction.

        Args:
            direction (str): The temporal direction, specifiying the cause or
                effect repertoire.
        """
        system = self.system[direction]
        node_labels = system.node_labels

        if not set(purview).issubset(self.purview_indices(direction)):
            raise ValueError('{} is not a {} purview in {}'.format(
                fmt.fmt_mechanism(purview, node_labels), direction, self))

        if not set(mechanism).issubset(self.mechanism_indices(direction)):
            raise ValueError('{} is no a {} mechanism in {}'.format(
                fmt.fmt_mechanism(mechanism, node_labels), direction, self))

        return system.repertoire(direction, mechanism, purview)