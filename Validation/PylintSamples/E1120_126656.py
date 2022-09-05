def create_prefix_dir(self, path, fmt):
        """Create the prefix dir, if missing"""
        create_prefix_dir(self._get_os_path(path.strip('/')), fmt)