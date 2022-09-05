def server(service, log=None):
    """
    Creates a threaded http service based on the passed HttpService instance.

    The returned object can be watched via taskforce.poll(), select.select(), etc.
    When activity is detected, the handle_request() method should be invoked.
    This starts a thread to handle the request.  URL paths are handled with callbacks
    which need to be established before any activity might occur.  If no callback
    is registered for a given path, the embedded handler will report a 404 error.
    Any exceptions raised by the callback result in a 500 error.

    This function just instantiates either a TCPServer or UnixStreamServer based
    on the address information in the "host" param.  The UnixStreamServer class
    is used for addresses containing a "/", otherwise the TCPServer class is
    used.  To create a Udom service in the current directory, use './name'.
    If TCP is selected and no port is provided using the ":" syntax, then
    def_port or def_sslport  will be used as appropriate.

    The BaseServer provides the code for registering HTTP handler callbacks.

    Parameters:

      service   - Service configuration.  See the HttpService class above.
      log       - A 'logging' object to log errors and activity.
"""
    if log:
        log = log
    else:                                                                                       # pragma: no cover
        log = logging.getLogger(__name__)
        log.addHandler(logging.NullHandler())
    if not service.timeout:                                                                     # pragma: no cover
        service.timeout = None

    if not service.listen:
        service.listen = def_address

    if service.listen.find('/') >=0 :
        httpd = UnixStreamServer(service.listen, service.timeout, log)
    else:
        port = None
        m = re.match(r'^(.*):(.*)$', service.listen)
        if m:
            log.debug("Matched host '%s', port '%s'", m.group(1), m.group(2))
            host = m.group(1)
            try:
                port = int(m.group(2))
            except:
                raise Exception("TCP listen port must be an integer")
        else:
            host = service.listen
            log.debug("No match, proceding with host '%s'", host)
        if not port:
            port = def_sslport if service.certfile else def_port
        httpd = TCPServer(host, port, service.timeout, log)
    if service.certfile:
        ciphers = ' '.join(ssl_ciphers)
        ctx = None
        try:
            ctx = ssl.SSLContext(ssl.PROTOCOL_SSLv23)
        except AttributeError:                                                                  # pragma: no cover
            log.warning("No ssl.SSLContext(), less secure connections may be allowed")
            pass
        if ctx:
            #  If ssl supports contexts, provide some tigher controls on the ssl negotiation
            #
            if 'OP_NO_SSLv2' in ssl.__dict__:
                ctx.options |= ssl.OP_NO_SSLv2
            else:                                                                               # pragma: no cover
                log.warning("Implementation does not offer ssl.OP_NO_SSLv2 which may allow less secure connections")
            if 'OP_NO_SSLv3' in ssl.__dict__:
                ctx.options |= ssl.OP_NO_SSLv3
            else:                                                                               # pragma: no cover
                log.warning("Implementation does not offer ssl.OP_NO_SSLv3 which may allow less secure connections")
            log.info("Certificate file: %s", service.certfile)
            with open(service.certfile, 'r') as f: pass
            ctx.load_cert_chain(service.certfile)
            ctx.set_ciphers(ciphers)
            httpd.socket = ctx.wrap_socket(httpd.socket, server_side=True)
        else:                                                                                   # pragma: no cover
            httpd.socket = ssl.wrap_socket(httpd.socket, server_side=True,
                certfile=service.certfile, ssl_version=ssl.PROTOCOL_TLSv1, ciphers=ciphers)
    if service.allow_control:
        httpd.allow_control = True
    log.info("HTTP service %s", str(service))
    return httpd