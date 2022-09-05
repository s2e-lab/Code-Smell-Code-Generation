def webapi_request(url, method='GET', caller=None, session=None, params=None):
    """Low level function for calling Steam's WebAPI

    .. versionchanged:: 0.8.3

    :param url: request url (e.g. ``https://api.steampowered.com/A/B/v001/``)
    :type url: :class:`str`
    :param method: HTTP method (GET or POST)
    :type method: :class:`str`
    :param caller: caller reference, caller.last_response is set to the last response
    :param params: dict of WebAPI and endpoint specific params
    :type params: :class:`dict`
    :param session: an instance requests session, or one is created per call
    :type session: :class:`requests.Session`
    :return: response based on paramers
    :rtype: :class:`dict`, :class:`lxml.etree.Element`, :class:`str`
    """
    if method not in ('GET', 'POST'):
        raise NotImplemented("HTTP method: %s" % repr(self.method))
    if params is None:
        params = {}

    onetime = {}
    for param in DEFAULT_PARAMS:
        params[param] = onetime[param] = params.get(param, DEFAULT_PARAMS[param])
    for param in ('raw', 'apihost', 'https', 'http_timeout'):
        del params[param]

    if onetime['format'] not in ('json', 'vdf', 'xml'):
        raise ValueError("Expected format to be json,vdf or xml; got %s" % onetime['format'])

    for k, v in list(params.items()): # serialize some types
        if isinstance(v, bool): params[k] = 1 if v else 0
        elif isinstance(v, dict): params[k] = _json.dumps(v)
        elif isinstance(v, list):
            del params[k]
            for i, lvalue in enumerate(v):
                params["%s[%d]" % (k, i)] = lvalue

    kwargs = {'params': params} if method == "GET" else {'data': params} # params to data for POST

    if session is None: session = _make_session()

    f = getattr(session, method.lower())
    resp = f(url, stream=False, timeout=onetime['http_timeout'], **kwargs)

    # we keep a reference of the last response instance on the caller
    if caller is not None: caller.last_response = resp
    # 4XX and 5XX will cause this to raise
    resp.raise_for_status()

    if onetime['raw']:
        return resp.text
    elif onetime['format'] == 'json':
        return resp.json()
    elif onetime['format'] == 'xml':
        from lxml import etree as _etree
        return _etree.fromstring(resp.content)
    elif onetime['format'] == 'vdf':
        import vdf as _vdf
        return _vdf.loads(resp.text)