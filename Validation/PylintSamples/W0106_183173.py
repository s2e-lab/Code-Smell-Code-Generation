def current_state(self):
        """
        Returns a ``field -> value`` dict of the current state of the instance.
        """
        field_names = set()
        [field_names.add(f.name) for f in self._meta.local_fields]
        [field_names.add(f.attname) for f in self._meta.local_fields]
        return dict([(field_name, getattr(self, field_name)) for field_name in field_names])