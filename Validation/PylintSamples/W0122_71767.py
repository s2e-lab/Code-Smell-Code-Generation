def import_bundle(self):
        """Add the filesystem to the Python sys path with an import hook, then import
        to file as Python"""
        from fs.errors import NoSysPathError

        try:
            import ambry.build
            module = sys.modules['ambry.build']
        except ImportError:
            module = imp.new_module('ambry.build')
            sys.modules['ambry.build'] = module

        bf = self.record

        if not bf.has_contents:
            from ambry.bundle import Bundle
            return Bundle

        try:
            abs_path = self._fs.getsyspath(self.file_name)
        except NoSysPathError:
            abs_path = '<string>'

        exec(compile(bf.contents, abs_path, 'exec'), module.__dict__)

        return module.Bundle