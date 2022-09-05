def setup_stream_handlers(conf):
    """Setup logging stream handlers according to the options."""
    class StdoutFilter(logging.Filter):
        def filter(self, record):
            return record.levelno in (logging.DEBUG, logging.INFO)

    log.handlers = []

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.WARNING)
    stdout_handler.addFilter(StdoutFilter())
    if conf.debug:
        stdout_handler.setLevel(logging.DEBUG)
    elif conf.verbose:
        stdout_handler.setLevel(logging.INFO)
    else:
        stdout_handler.setLevel(logging.WARNING)
    log.addHandler(stdout_handler)

    stderr_handler = logging.StreamHandler(sys.stderr)
    msg_format = "%(levelname)s: %(message)s"
    stderr_handler.setFormatter(logging.Formatter(fmt=msg_format))
    stderr_handler.setLevel(logging.WARNING)
    log.addHandler(stderr_handler)