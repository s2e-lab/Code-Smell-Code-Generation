def view():
    """Get a dictionary representation of current metrics."""
    if not initialized:
        raise NotInitialized

    marshalled_metrics_mmap.seek(0)
    try:
        uwsgi.lock()
        marshalled_view = marshalled_metrics_mmap.read(
            MAX_MARSHALLED_VIEW_SIZE)
    finally:
        uwsgi.unlock()
    return marshal.loads(marshalled_view)