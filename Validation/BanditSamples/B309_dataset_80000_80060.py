def _put_or_post_json(self, method, url, data):
        """
        urlencodes the data and PUTs it to the url
        the response is parsed as JSON and the resulting data type is returned
        """
        if self.parsed_endpoint.scheme == 'https':
            conn = httplib.HTTPSConnection(self.parsed_endpoint.netloc)
        else:
            conn = httplib.HTTPConnection(self.parsed_endpoint.netloc)
        head = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": USER_AGENT,
            API_TOKEN_HEADER_NAME: self.api_token,
        }
        if self.api_version in ['0.1', '0.01a']:
            head[API_VERSION_HEADER_NAME] = self.api_version
        conn.request(method, url, json.dumps(data), head)
        resp = conn.getresponse()
        self._handle_response_errors(method, url, resp)
        return json.loads(resp.read())