def publish_asset_ddo(self, ddo):
        """
        Register asset ddo in aquarius.

        :param ddo: DDO instance
        :return: API response (depends on implementation)
        """
        try:
            asset_did = ddo.did
            response = self.requests_session.post(self.url, data=ddo.as_text(),
                                                  headers=self._headers)
        except AttributeError:
            raise AttributeError('DDO invalid. Review that all the required parameters are filled.')
        if response.status_code == 500:
            raise ValueError(
                f'This Asset ID already exists! \n\tHTTP Error message: \n\t\t{response.text}')
        elif response.status_code != 201:
            raise Exception(f'{response.status_code} ERROR Full error: \n{response.text}')
        elif response.status_code == 201:
            response = json.loads(response.content)
            logger.debug(f'Published asset DID {asset_did}')
            return response
        else:
            raise Exception(f'Unhandled ERROR: status-code {response.status_code}, '
                            f'error message {response.text}')