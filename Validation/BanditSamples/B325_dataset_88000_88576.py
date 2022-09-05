def tempput(local_path=None, remote_path=None, use_sudo=False,
            mirror_local_mode=False, mode=None):
    """Put a file to remote and remove it afterwards"""
    import warnings
    warnings.simplefilter('ignore', RuntimeWarning)
    if remote_path is None:
        remote_path = os.tempnam()
    put(local_path, remote_path, use_sudo, mirror_local_mode, mode)
    yield remote_path
    run("rm '{}'".format(remote_path))