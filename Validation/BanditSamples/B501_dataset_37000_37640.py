def _login_request(self, url_login):
        """Internal function to send login request. """

        expiration_time = self._exp_time
        payload = {'expirationTime': expiration_time}
        # TODO(padkrish), after testing with certificates, make the
        # verify option configurable.
        res = requests.post(url_login,
                            data=jsonutils.dumps(payload),
                            headers=self._req_headers,
                            auth=(self._user, self._pwd),
                            timeout=self.timeout_resp, verify=False)
        session_id = ''
        if res and res.status_code in self._resp_ok:
            session_id = res.json().get('Dcnm-Token')
        self._req_headers.update({'Dcnm-Token': session_id})