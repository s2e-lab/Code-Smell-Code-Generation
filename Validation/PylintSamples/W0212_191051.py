def clear_search_defaults(self, args=None):
        """
        Clear all search defaults specified by the list of parameter names
        given as ``args``.  If ``args`` is not given, then clear all existing
        search defaults.

        Examples::

            conn.set_search_defaults(scope=ldap.SCOPE_BASE, attrs=['cn'])
            conn.clear_search_defaults(['scope'])
            conn.clear_search_defaults()
        """
        if args is None:
            self._search_defaults.clear()
        else:
            for arg in args:
                if arg in self._search_defaults:
                    del self._search_defaults[arg]