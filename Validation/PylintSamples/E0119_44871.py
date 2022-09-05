def get_json_results(self, response):
        '''
        Parses the request result and returns the JSON object. Handles all errors.
        '''
        try:
            # return the proper JSON object, or error code if request didn't go through.
            self.most_recent_json = response.json()
            json_results = response.json()
            if response.status_code in [401, 403]: #401 is invalid key, 403 is out of monthly quota.
                raise PyMsCognitiveWebSearchException("CODE {code}: {message}".format(code=response.status_code,message=json_results["message"]) )
            elif response.status_code in [429]: #429 means try again in x seconds.
                message = json_results['message']
                try:
                    # extract time out seconds from response
                    timeout = int(re.search('in (.+?) seconds', message).group(1)) + 1
                    print ("CODE 429, sleeping for {timeout} seconds").format(timeout=str(timeout))
                    time.sleep(timeout)
                except (AttributeError, ValueError) as e:
                    if not self.silent_fail:
                        raise PyMsCognitiveWebSearchException("CODE 429. Failed to auto-sleep: {message}".format(code=response.status_code,message=json_results["message"]) )
                    else:
                        print ("CODE 429. Failed to auto-sleep: {message}. Trying again in 5 seconds.".format(code=response.status_code,message=json_results["message"]))
                        time.sleep(5)
        except ValueError as vE:
            if not self.silent_fail:
                raise PyMsCognitiveWebSearchException("Request returned with code %s, error msg: %s" % (r.status_code, r.text))
            else:
                print ("[ERROR] Request returned with code %s, error msg: %s. \nContinuing in 5 seconds." % (r.status_code, r.text))
                time.sleep(5)
        return json_results