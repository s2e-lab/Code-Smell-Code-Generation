def get_executable_fullpath(name, dirname=None):
        # type: (AnyStr, Optional[AnyStr]) -> Optional[AnyStr]
        """get the full path of a given executable name"""
        if name is None:
            return None
        if is_string(name):
            name = str(name)
        else:
            raise RuntimeError('The input function name or path must be string!')
        if dirname is not None:  # check the given path first
            dirname = os.path.abspath(dirname)
            fpth = dirname + os.sep + name
            if os.path.isfile(fpth):
                return fpth
        # If dirname is not specified, check the env then.
        if sysstr == 'Windows':
            findout = UtilClass.run_command('where %s' % name)
        else:
            findout = UtilClass.run_command('which %s' % name)
        if not findout or len(findout) == 0:
            print("%s is not included in the env path" % name)
            exit(-1)
        first_path = findout[0].split('\n')[0]
        if os.path.exists(first_path):
            return first_path
        return None