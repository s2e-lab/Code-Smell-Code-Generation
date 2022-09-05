def spawn(self, filepath, configuration_alias, replace=False):
        """Spawns uWSGI using the given configuration module.

        :param str|unicode filepath:

        :param str|unicode configuration_alias:

        :param bool replace: Whether a new process should replace current one.

        """
        # Pass --conf as an argument to have a chance to use
        # touch reloading form .py configuration file change.
        args = ['uwsgi', '--ini', 'exec://%s %s --conf %s' % (self.binary_python, filepath, configuration_alias)]

        if replace:
            return os.execvp('uwsgi', args)

        return os.spawnvp(os.P_NOWAIT, 'uwsgi', args)