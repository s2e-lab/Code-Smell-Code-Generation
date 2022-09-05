def reset():
    """
    Clear global data and remove the handlers.
    CAUSION! This method sets as a signal handlers the ones which it has
    noticed on initialization time. If there has been another handler installed
    on top of us it will get removed by this method call.
    """
    global _handlers, python_signal
    for sig, (previous, _) in _handlers.iteritems():
        if not previous:
            previous = SIG_DFL
        python_signal.signal(sig, previous)
    _handlers = dict()