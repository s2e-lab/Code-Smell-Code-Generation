def _update(self, filename, dir1, dir2):
        """ Private function for updating a file based on
        last time stamp of modification """

        # NOTE: dir1 is source & dir2 is target
        if self._updatefiles:

            file1 = os.path.join(dir1, filename)
            file2 = os.path.join(dir2, filename)

            try:
                st1 = os.stat(file1)
                st2 = os.stat(file2)
            except os.error:
                return -1

            # Update will update in both directions depending
            # on the timestamp of the file & copy-direction.

            if self._copydirection == 0 or self._copydirection == 2:

                # Update file if file's modification time is older than
                # source file's modification time, or creation time. Sometimes
                # it so happens that a file's creation time is newer than it's
                # modification time! (Seen this on windows)
                if self._cmptimestamps(st1, st2):
                    if self._verbose:
                        # source to target
                        self.log('Updating file %s' % file2)
                    try:
                        if self._forcecopy:
                            os.chmod(file2, 1638)  # 1638 = 0o666

                        try:
                            if os.path.islink(file1):
                                os.symlink(os.readlink(file1), file2)
                            else:
                                shutil.copy2(file1, file2)
                            self._changed.append(file2)
                            self._numupdates += 1
                            return 0
                        except (IOError, OSError) as e:
                            self.log(str(e))
                            self._numupdsfld += 1
                            return -1

                    except Exception as e:
                        self.log(str(e))
                        return -1

            if self._copydirection == 1 or self._copydirection == 2:

                # Update file if file's modification time is older than
                # source file's modification time, or creation time. Sometimes
                # it so happens that a file's creation time is newer than it's
                # modification time! (Seen this on windows)
                if self._cmptimestamps(st2, st1):
                    if self._verbose:
                        # target to source
                        self.log('Updating file %s' % file1)
                    try:
                        if self._forcecopy:
                            os.chmod(file1, 1638)  # 1638 = 0o666

                        try:
                            if os.path.islink(file2):
                                os.symlink(os.readlink(file2), file1)
                            else:
                                shutil.copy2(file2, file1)
                            self._changed.append(file1)
                            self._numupdates += 1
                            return 0
                        except (IOError, OSError) as e:
                            self.log(str(e))
                            self._numupdsfld += 1
                            return -1

                    except Exception as e:
                        self.log(str(e))
                        return -1

        return -1