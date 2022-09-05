def kill(path):
    """
    Kills the process, if it still exists.

    :type  path: str
    :param path: The name of the pidfile.
    """
    # try to read the pid from the pidfile
    pid = read(path)
    if pid is None:
        return

    # Try to kill the process.
    logging.info("Killing PID %s", pid)
    try:
        os.kill(pid, 9)
    except OSError as xxx_todo_changeme2:
        # re-raise if the error wasn't "No such process"
        (code, text) = xxx_todo_changeme2.args
        # re-raise if the error wasn't "No such process"
        if code != errno.ESRCH:
            raise