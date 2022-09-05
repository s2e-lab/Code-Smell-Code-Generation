def _api_post(path, data, server=None):
    '''
    Do a POST request to the API
    '''
    server = _get_server(server)
    response = requests.post(
            url=_get_url(server['ssl'], server['url'], server['port'], path),
            auth=_get_auth(server['user'], server['password']),
            headers=_get_headers(),
            data=salt.utils.json.dumps(data),
            verify=False
    )
    return _api_response(response)