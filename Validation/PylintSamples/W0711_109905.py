def _get_response(self, endpoint, request_dict):
        """ Returns a dictionary of data requested by each function.

        Arguments:
        ----------
        endpoint: string, mandatory
            Set in all other methods, this is the API endpoint specific to each function.
        request_dict: string, mandatory
            A dictionary of parameters that are formatted into the API call.

        Returns:
        --------
            response: A dictionary that has been dumped from JSON.

        Raises:
        -------
            MesoPyError: Overrides the exceptions given in the requests library to give more custom error messages.
            Connection_error occurs if no internet connection exists. Timeout_error occurs if the request takes too
            long and redirect_error is shown if the url is formatted incorrectly.

        """
        http_error = 'Could not connect to the API. This could be because you have no internet connection, a parameter' \
                     ' was input incorrectly, or the API is currently down. Please try again.'
                     
        json_error = 'Could not retrieve JSON values. Try again with a shorter date range.'
        
        # For python 3.4
        try:
            qsp = urllib.parse.urlencode(request_dict, doseq=True)
            resp = urllib.request.urlopen(self.base_url + endpoint + '?' + qsp).read()

        # For python 2.7
        except AttributeError or NameError:
            try:
                qsp = urllib.urlencode(request_dict, doseq=True)
                resp = urllib2.urlopen(self.base_url + endpoint + '?' + qsp).read()
            except urllib2.URLError:
                raise MesoPyError(http_error)
        except urllib.error.URLError:
            raise MesoPyError(http_error)
        
        try:
            json_data = json.loads(resp.decode('utf-8'))
        except ValueError:
            raise MesoPyError(json_error)
        
        return self._checkresponse(json_data)