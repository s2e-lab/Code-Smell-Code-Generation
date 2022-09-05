def _get_user_processes():
    """ Gets process information owned by the current user.

        Returns generator of tuples: (``psutil.Process`` instance, path).
        """

    uid = os.getuid()

    for proc in psutil.process_iter():
        try:
            # yield processes that match current user
            if proc.uids.real == uid:
                yield (proc, proc.exe)

        except psutil.AccessDenied:
            # work around for suid/sguid processes and MacOS X restrictions
            try:
                path = common.which(proc.name)

                # psutil doesn't support MacOS X relative paths,
                # let's use a workaround to merge working directory with
                # process relative path
                if not path and common.IS_MACOSX:
                    cwd = _get_process_cwd(proc.pid)
                    if not cwd:
                        continue
                    path = os.path.join(cwd, proc.cmdline[0])

                yield (proc, path)

            except (psutil.AccessDenied, OSError):
                pass

        except psutil.NoSuchProcess:
            pass