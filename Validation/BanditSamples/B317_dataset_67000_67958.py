def get_all_parts(self, max_parts=None, part_number_marker=None):
        """
        Return the uploaded parts of this MultiPart Upload.  This is
        a lower-level method that requires you to manually page through
        results.  To simplify this process, you can just use the
        object itself as an iterator and it will automatically handle
        all of the paging with S3.
        """
        self._parts = []
        query_args = 'uploadId=%s' % self.id
        if max_parts:
            query_args += '&max-parts=%d' % max_parts
        if part_number_marker:
            query_args += '&part-number-marker=%s' % part_number_marker
        response = self.bucket.connection.make_request('GET', self.bucket.name,
                                                       self.key_name,
                                                       query_args=query_args)
        body = response.read()
        if response.status == 200:
            h = handler.XmlHandler(self, self)
            xml.sax.parseString(body, h)
            return self._parts