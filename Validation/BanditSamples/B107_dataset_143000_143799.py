def _urllib_post(self, url,
                     json="",
                     data="",
                     username="",
                     password="",
                     headers=None,
                     timeout=30):

        """This function will POST to the url endpoint using urllib2. returning
        an AdyenResult object on 200 HTTP responce. Either json or data has to
        be provided. If username and password are provided, basic auth will be
        used.

        Args:
            url (str):                  url to send the POST
            json (dict, optional):      Dict of the JSON to POST
            data (dict, optional):      Dict, presumed flat structure of
                                        key/value of request to place as
                                        www-form
            username (str, optional):    Username for basic auth. Must be
                                        uncluded as part of password.
            password (str, optional):   Password for basic auth. Must be
                                        included as part of username.
            headers (dict, optional):   Key/Value pairs of headers to include
            timeout (int, optional): Default 30. Timeout for the request.

        Returns:
            str:    Raw response received
            str:    Raw request placed
            int:    HTTP status code, eg 200,404,401
            dict:   Key/Value pairs of the headers received.
        """
        if headers is None:
            headers = {}

        # Store regular dict to return later:
        raw_store = json

        raw_request = json_lib.dumps(json) if json else urlencode(data)
        url_request = Request(url, data=raw_request.encode('utf8'))
        if json:
            url_request.add_header('Content-Type', 'application/json')
        elif not data:
            raise ValueError("Please provide either a json or a data field.")

        # Add User-Agent header to request so that the
        # request can be identified as coming from the Adyen Python library.
        headers['User-Agent'] = self.user_agent

        # Set regular dict to return as raw_request:
        raw_request = raw_store

        # Adding basic auth is username and password provided.
        if username and password:
            if sys.version_info[0] >= 3:
                basic_authstring = base64.encodebytes(('%s:%s' %
                                                       (username, password))
                                                      .encode()).decode(). \
                    replace('\n', '')
            else:
                basic_authstring = base64.encodestring('%s:%s' % (username,
                                                                  password)). \
                    replace('\n', '')
            url_request.add_header("Authorization",
                                   "Basic %s" % basic_authstring)

        # Adding the headers to the request.
        for key, value in headers.items():
            url_request.add_header(key, str(value))

        # URLlib raises all non 200 responses as en error.
        try:
            response = urlopen(url_request, timeout=timeout)
        except HTTPError as e:
            raw_response = e.read()

            return raw_response, raw_request, e.getcode(), e.headers
        else:
            raw_response = response.read()
            response.close()

            # The dict(response.info()) is the headers of the response
            # Raw response, raw request, status code and headers returned
            return (raw_response, raw_request,
                    response.getcode(), dict(response.info()))