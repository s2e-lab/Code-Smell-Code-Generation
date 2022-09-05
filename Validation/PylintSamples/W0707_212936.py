def link_or_copy(src, dst, verbosity=0):
    """Try to make a hard link from src to dst and if that fails
    copy the file. Hard links save some disk space and linking
    should fail fast since no copying is involved.
    """
    if verbosity > 0:
        log_info("Copying %s -> %s" % (src, dst))
    try:
        os.link(src, dst)
    except (AttributeError, OSError):
        try:
            shutil.copy(src, dst)
        except OSError as msg:
            raise PatoolError(msg)