def _copy(self, filename, dir1, dir2):
        """ Private function for copying a file """

        # NOTE: dir1 is source & dir2 is target
        if self._copyfiles:

            rel_path = filename.replace('\\', '/').split('/')
            rel_dir = '/'.join(rel_path[:-1])
            filename = rel_path[-1]

            dir2_root = dir2

            dir1 = os.path.join(dir1, rel_dir)
            dir2 = os.path.join(dir2, rel_dir)

            if self._verbose:
                self.log('Copying file %s from %s to %s' %
                         (filename, dir1, dir2))
            try:
                # source to target
                if self._copydirection == 0 or self._copydirection == 2:

                    if not os.path.exists(dir2):
                        if self._forcecopy:
                            # 1911 = 0o777
                            os.chmod(os.path.dirname(dir2_root), 1911)
                        try:
                            os.makedirs(dir2)
                            self._numnewdirs += 1
                        except OSError as e:
                            self.log(str(e))
                            self._numdirsfld += 1

                    if self._forcecopy:
                        os.chmod(dir2, 1911)  # 1911 = 0o777

                    sourcefile = os.path.join(dir1, filename)
                    try:
                        if os.path.islink(sourcefile):
                            os.symlink(os.readlink(sourcefile),
                                       os.path.join(dir2, filename))
                        else:
                            shutil.copy2(sourcefile, dir2)
                        self._numfiles += 1
                    except (IOError, OSError) as e:
                        self.log(str(e))
                        self._numcopyfld += 1

                if self._copydirection == 1 or self._copydirection == 2:
                    # target to source

                    if not os.path.exists(dir1):
                        if self._forcecopy:
                            # 1911 = 0o777
                            os.chmod(os.path.dirname(self.dir1_root), 1911)

                        try:
                            os.makedirs(dir1)
                            self._numnewdirs += 1
                        except OSError as e:
                            self.log(str(e))
                            self._numdirsfld += 1

                    targetfile = os.path.abspath(os.path.join(dir1, filename))
                    if self._forcecopy:
                        os.chmod(dir1, 1911)  # 1911 = 0o777

                    sourcefile = os.path.join(dir2, filename)

                    try:
                        if os.path.islink(sourcefile):
                            os.symlink(os.readlink(sourcefile),
                                       os.path.join(dir1, filename))
                        else:
                            shutil.copy2(sourcefile, targetfile)
                        self._numfiles += 1
                    except (IOError, OSError) as e:
                        self.log(str(e))
                        self._numcopyfld += 1

            except Exception as e:
                self.log('Error copying file %s' % filename)
                self.log(str(e))