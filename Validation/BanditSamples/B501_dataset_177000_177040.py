def _api_get(path, server=None):
    '''
    Do a GET request to the API
    '''
    server = _get_server(server)
    response = requests.get(
            url=_get_url(server['ssl'], server['url'], server['port'], path),
            auth=_get_auth(server['user'], server['password']),
            headers=_get_headers(),
            verify=False
    )
    return _api_response(response)