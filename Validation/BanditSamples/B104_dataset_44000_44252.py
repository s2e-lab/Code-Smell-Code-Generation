def cherrypy_server_runner(
        app, global_conf=None, host='127.0.0.1', port=None,
        ssl_pem=None, protocol_version=None, numthreads=None,
        server_name=None, max=None, request_queue_size=None,
        timeout=None
): # pragma: no cover
    """
    Entry point for CherryPy's WSGI server

    Serves the specified WSGI app via CherryPyWSGIServer.

    ``app``

        The WSGI 'application callable'; multiple WSGI applications
        may be passed as (script_name, callable) pairs.

    ``host``

        This is the ipaddress to bind to (or a hostname if your
        nameserver is properly configured).  This defaults to
        127.0.0.1, which is not a public interface.

    ``port``

        The port to run on, defaults to 8080 for HTTP, or 4443 for
        HTTPS. This can be a string or an integer value.

    ``ssl_pem``

        This an optional SSL certificate file (via OpenSSL) You can
        generate a self-signed test PEM certificate file as follows:

            $ openssl genrsa 1024 > host.key
            $ chmod 400 host.key
            $ openssl req -new -x509 -nodes -sha1 -days 365  \\
                          -key host.key > host.cert
            $ cat host.cert host.key > host.pem
            $ chmod 400 host.pem

    ``protocol_version``

        The protocol used by the server, by default ``HTTP/1.1``.

    ``numthreads``

        The number of worker threads to create.

    ``server_name``

        The string to set for WSGI's SERVER_NAME environ entry.

    ``max``

        The maximum number of queued requests. (defaults to -1 = no
        limit).

    ``request_queue_size``

        The 'backlog' argument to socket.listen(); specifies the
        maximum number of queued connections.

    ``timeout``

        The timeout in seconds for accepted connections.
    """
    is_ssl = False
    if ssl_pem:
        port = port or 4443
        is_ssl = True

    if not port:
        if ':' in host:
            host, port = host.split(':', 1)
        else:
            port = 8080
    bind_addr = (host, int(port))

    kwargs = {}
    for var_name in ('numthreads', 'max', 'request_queue_size', 'timeout'):
        var = locals()[var_name]
        if var is not None:
            kwargs[var_name] = int(var)

    server = None
    try:
        # Try to import from newer CherryPy releases.
        import cheroot.wsgi as wsgiserver
        server = wsgiserver.Server(bind_addr, app,
            server_name=server_name, **kwargs)
    except ImportError:
        # Nope. Try to import from older CherryPy releases.
        # We might just take another ImportError here. Oh well.
        from cherrypy import wsgiserver
        server = wsgiserver.CherryPyWSGIServer(bind_addr, app,
            server_name=server_name, **kwargs)

    server.ssl_certificate = server.ssl_private_key = ssl_pem
    if protocol_version:
        server.protocol = protocol_version

    try:
        protocol = is_ssl and 'https' or 'http'
        if host == '0.0.0.0':
            print('serving on 0.0.0.0:%s view at %s://127.0.0.1:%s' %
                  (port, protocol, port))
        else:
            print('serving on %s://%s:%s' % (protocol, host, port))
        server.start()
    except (KeyboardInterrupt, SystemExit):
        server.stop()

    return server