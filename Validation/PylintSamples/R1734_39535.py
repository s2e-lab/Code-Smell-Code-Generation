def connection_cache(func: callable):
    """Connection cache for SSH sessions. This is to prevent opening a
     new, expensive connection on every command run."""
    cache = dict()
    lock = RLock()

    @wraps(func)
    def func_wrapper(host: str, username: str, *args, **kwargs):
        key = "{h}-{u}".format(h=host, u=username)
        if key in cache:
            # connection exists, check if it is still valid before
            # returning it.
            conn = cache[key]
            if conn and conn.is_active() and conn.is_authenticated():
                return conn
            else:
                # try to close a bad connection and remove it from
                # the cache.
                if conn:
                    try_close(conn)
                del cache[key]

        # key is not in the cache, so try to recreate it
        # it may have been removed just above.
        if key not in cache:
            conn = func(host, username, *args, **kwargs)
            if conn is not None:
                cache[key] = conn
            return conn

        # not sure how to reach this point, but just in case.
        return None

    def get_cache() -> dict:
        return cache

    def purge(key: str=None):
        with lock:
            if key is None:
                conns = [(k, v) for k, v in cache.items()]
            elif key in cache:
                conns = ((key, cache[key]), )
            else:
                conns = list()

            for k, v in conns:
                try_close(v)
                del cache[k]

    func_wrapper.get_cache = get_cache
    func_wrapper.purge = purge
    return func_wrapper