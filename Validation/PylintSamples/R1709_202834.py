def all(self, fields=None, include_fields=True, page=None, per_page=None, extra_params=None):
        """Retrieves a list of all the applications.

        Important: The client_secret and encryption_key attributes can only be
        retrieved with the read:client_keys scope.

        Args:
           fields (list of str, optional): A list of fields to include or
              exclude from the result (depending on include_fields). Empty to
              retrieve all fields.

           include_fields (bool, optional): True if the fields specified are
              to be included in the result, False otherwise.

           page (int): The result's page number (zero based).

           per_page (int, optional): The amount of entries per page.

           extra_params (dictionary, optional): The extra parameters to add to
             the request. The fields, include_fields, page and per_page values
             specified as parameters take precedence over the ones defined here.


        See: https://auth0.com/docs/api/management/v2#!/Clients/get_clients
        """
        params = extra_params or {}
        params['fields'] = fields and ','.join(fields) or None
        params['include_fields'] = str(include_fields).lower()
        params['page'] = page
        params['per_page'] = per_page

        return self.client.get(self._url(), params=params)