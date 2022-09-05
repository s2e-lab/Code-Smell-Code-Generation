def custom(srcpaths, event_cb=None, poll_interval=1, recurse=True,
           restart_cb=None, restart_func=None, close_fds=True):
    '''Sets up lazarus in custom mode.

    See the :py:func:`default` function for a simpler mode of use.

    The custom mode of lazarus is to watch all modules rooted at any of the
    source paths provided for changes and restart when they take place.

    Keyword arguments:

        event_cb -- Callback invoked when a file rooted at a source path
        changes. Without specifying an event callback, changes to any module
        rooted at a source path will trigger a restart.

        poll_interval -- Rate at which changes will be detected. The default
        value of ``1`` means it may take up to one second to detect changes.
        Decreasing this value may lead to unnecessary overhead.

        recurse -- Whether to watch all subdirectories of every source path for
        changes or only the paths provided.

        restart_cb -- Callback invoked prior to restarting the process; allows
        for any cleanup to occur prior to restarting. Returning anything other
        than *None* in the callback will cancel the restart.

        restart_func -- Function invoked to restart the process. This supplants
        the default behavior of using *sys.executable* and *sys.argv*.

        close_fds -- Whether all file descriptors other than *stdin*, *stdout*,
        and *stderr* should be closed

    An example of using a cleanup function prior to restarting:

        >>> def cleanup():
        ...     pass
        >>> import lazarus
        >>> lazarus.custom(os.curdir, restart_cb=cleanup)
        >>> lazarus.stop()

    An example of avoiding restarts when any ``__main__.py`` changes:

        >>> def skip_main(event):
        ...     if event.src_path == '__main__.py':
        ...         return False
        ...     return True
        >>> import lazarus
        >>> lazarus.custom(os.curdir, event_cb=skip_main)
        >>> lazarus.stop()
    '''
    if _active:
        msg = 'lazarus is already active'
        raise RuntimeWarning(msg)

    if restart_cb and not callable(restart_cb):
        msg = 'restart_cb keyword argument is not callable'
        raise TypeError(msg)

    if restart_func and not callable(restart_func):
        msg = 'restart_func keyword argument is not callable'
        raise TypeError(msg)

    global _close_fds
    _close_fds = close_fds

    try:
        from watchdog.observers import Observer
        from watchdog.events import FileSystemEventHandler
    except ImportError as ie:
        msg = 'no watchdog support (%s)' % str(ie)
        raise RuntimeError(msg)

    class _Handler(FileSystemEventHandler):

        def __init__(self):
            self.active = True

        def dispatch(self, event):
            if not self.active:
                return
            super(_Handler, self).dispatch(event)

        def all_events(self, event):
            # if caller wants event_cb control, defer _restart logic to them
            # (caller decides whether this is a restart event)
            if event_cb:
                if event_cb(event):
                    cancelled = _restart()
                    if not cancelled:
                        self.active = False

            # use default is_restart_event logic
            elif is_restart_event(event):
                cancelled = _restart()
                if not cancelled:
                    self.active = False
                self.active = False

        def on_created(self, event):
            self.all_events(event)

        def on_deleted(self, event):
            self.all_events(event)

        def on_modified(self, event):
            self.all_events(event)

        def on_moved(self, event):
            self.all_events(event)

    global _observer
    kwargs = {'timeout': poll_interval}
    _observer = Observer(**kwargs)
    global _restart_cb
    _restart_cb = restart_cb

    handler = _Handler()
    srcpaths = _as_list(srcpaths)
    kwargs = {}
    if recurse:
        kwargs['recursive'] = True
    for srcpath in srcpaths:
        _observer.schedule(handler, srcpath, **kwargs)
    _activate()
    _observer.start()