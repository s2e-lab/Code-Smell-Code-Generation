def _get_token_from_cookies(self, request, refresh_token):
        """
        Extract the token if present inside the request cookies.
        """
        if refresh_token:
            cookie_token_name_key = "cookie_refresh_token_name"
        else:
            cookie_token_name_key = "cookie_access_token_name"
        cookie_token_name = getattr(self.config, cookie_token_name_key)
        return request.cookies.get(cookie_token_name(), None)