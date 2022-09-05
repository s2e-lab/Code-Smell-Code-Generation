def _construct_request(self):
        """
        Utility for constructing the request header and connection
        """
        if self.parsed_endpoint.scheme == 'https':
            conn = httplib.HTTPSConnection(self.parsed_endpoint.netloc)
        else:
            conn = httplib.HTTPConnection(self.parsed_endpoint.netloc)
        head = {
            "Accept": "application/json",
            "User-Agent": USER_AGENT,
            API_TOKEN_HEADER_NAME: self.api_token,
        }
        if self.api_version in ['0.1', '0.01a']:
            head[API_VERSION_HEADER_NAME] = self.api_version
        return conn, head