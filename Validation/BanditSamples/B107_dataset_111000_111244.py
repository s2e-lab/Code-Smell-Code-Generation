def fetch_token(self, client_secret, code, context, scope, redirect_uri,
                    token_url='https://login.bigcommerce.com/oauth2/token'):
        """
        Fetches a token from given token_url, using given parameters, and sets up session headers for
        future requests.
        redirect_uri should be the same as your callback URL.
        code, context, and scope should be passed as parameters to your callback URL on app installation.

        Raises HttpException on failure (same as Connection methods).
        """
        res = self.post(token_url, {'client_id': self.client_id,
                                    'client_secret': client_secret,
                                    'code': code,
                                    'context': context,
                                    'scope': scope,
                                    'grant_type': 'authorization_code',
                                    'redirect_uri': redirect_uri},
                        headers={'Content-Type': 'application/x-www-form-urlencoded'})
        self._session.headers.update(self._oauth_headers(self.client_id, res['access_token']))
        return res