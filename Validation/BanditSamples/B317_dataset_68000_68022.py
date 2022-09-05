def get_location(self):
        """
        Returns the LocationConstraint for the bucket.

        :rtype: str
        :return: The LocationConstraint for the bucket or the empty
                 string if no constraint was specified when bucket
                 was created.
        """
        response = self.connection.make_request('GET', self.name,
                                                query_args='location')
        body = response.read()
        if response.status == 200:
            rs = ResultSet(self)
            h = handler.XmlHandler(rs, self)
            xml.sax.parseString(body, h)
            return rs.LocationConstraint
        else:
            raise self.connection.provider.storage_response_error(
                response.status, response.reason, body)