def parse_ini(self, paths=None, namespace=None, permissive=False):
        """Parse config files and return configuration options.

        Expects array of files that are in ini format.

        :param paths:
            List of paths to files to parse (uses ConfigParse logic).
            If not supplied, uses the ini_paths value supplied on
            initialization.
        """
        namespace = namespace or self.prog
        results = {}
        # DeprecationWarning: SafeConfigParser has been renamed to ConfigParser
        # in Python 3.2. This alias will be removed in future versions. Use
        # ConfigParser directly instead.
        if sys.version_info < (3, 2):
            self.ini_config = configparser.SafeConfigParser()
        else:
            self.ini_config = configparser.ConfigParser()

        parser_errors = (configparser.NoOptionError,
                         configparser.NoSectionError)

        inipaths = list(paths or reversed(self._ini_paths))
        # check that explicitly defined ini paths exist
        for pth in inipaths:
            if not os.path.isfile(pth):
                raise OSError(errno.ENOENT, 'No such file or directory', pth)

        read_ok = self.ini_config.read(inipaths)
        assert read_ok == inipaths
        dicts = (list(self.ini_config._sections.values()) +
                 [self.ini_config.defaults()])
        ini_options = {k for d in dicts for k in d.keys() if k != '__name__'}
        if not ini_options:
            return results

        for option in self._options:
            ini_section = option.kwargs.get('ini_section')
            value = None
            if ini_section:
                try:
                    value = self.ini_config.get(ini_section, option.name)
                    results[option.dest] = option.type(value)
                except parser_errors as err:
                    # this is an ERROR and the next one is a DEBUG b/c
                    # this code is executed only if the Option is defined
                    # with the ini_section keyword argument
                    LOG.error('Error parsing ini file: %r -- Continuing.',
                              err)
            if not value:
                try:
                    value = self.ini_config.get(namespace, option.name)
                    results[option.dest] = option.type(value)
                except parser_errors as err:
                    LOG.debug('Error parsing ini file: %r -- Continuing.',
                              err)
            if option.dest in results:
                ini_options.remove(option.dest)
        if ini_options and not permissive:
            raise simpl.exceptions.SimplConfigUnknownOption(
                'No corresponding Option was found for the following '
                'values in the ini file: %s'
                % ', '.join(["'%s'" % o for o in ini_options]))
        return results