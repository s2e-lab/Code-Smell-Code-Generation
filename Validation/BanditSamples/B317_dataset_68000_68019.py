def copy_key(self, new_key_name, src_bucket_name,
                 src_key_name, metadata=None, src_version_id=None,
                 storage_class='STANDARD', preserve_acl=False,
                 encrypt_key=False, headers=None, query_args=None):
        """
        Create a new key in the bucket by copying another existing key.

        :type new_key_name: string
        :param new_key_name: The name of the new key

        :type src_bucket_name: string
        :param src_bucket_name: The name of the source bucket

        :type src_key_name: string
        :param src_key_name: The name of the source key

        :type src_version_id: string
        :param src_version_id: The version id for the key.  This param
                               is optional.  If not specified, the newest
                               version of the key will be copied.

        :type metadata: dict
        :param metadata: Metadata to be associated with new key.
                         If metadata is supplied, it will replace the
                         metadata of the source key being copied.
                         If no metadata is supplied, the source key's
                         metadata will be copied to the new key.

        :type storage_class: string
        :param storage_class: The storage class of the new key.
                              By default, the new key will use the
                              standard storage class.  Possible values are:
                              STANDARD | REDUCED_REDUNDANCY

        :type preserve_acl: bool
        :param preserve_acl: If True, the ACL from the source key
                             will be copied to the destination
                             key.  If False, the destination key
                             will have the default ACL.
                             Note that preserving the ACL in the
                             new key object will require two
                             additional API calls to S3, one to
                             retrieve the current ACL and one to
                             set that ACL on the new object.  If
                             you don't care about the ACL, a value
                             of False will be significantly more
                             efficient.

        :type encrypt_key: bool
        :param encrypt_key: If True, the new copy of the object will
                            be encrypted on the server-side by S3 and
                            will be stored in an encrypted form while
                            at rest in S3.

        :type headers: dict
        :param headers: A dictionary of header name/value pairs.

        :type query_args: string
        :param query_args: A string of additional querystring arguments
                           to append to the request

        :rtype: :class:`boto.s3.key.Key` or subclass
        :returns: An instance of the newly created key object
        """
        headers = headers or {}
        provider = self.connection.provider
        src_key_name = boto.utils.get_utf8_value(src_key_name)
        if preserve_acl:
            if self.name == src_bucket_name:
                src_bucket = self
            else:
                src_bucket = self.connection.get_bucket(src_bucket_name)
            acl = src_bucket.get_xml_acl(src_key_name)
        if encrypt_key:
            headers[provider.server_side_encryption_header] = 'AES256'
        src = '%s/%s' % (src_bucket_name, urllib.quote(src_key_name))
        if src_version_id:
            src += '?versionId=%s' % src_version_id
        headers[provider.copy_source_header] = str(src)
        # make sure storage_class_header key exists before accessing it
        if provider.storage_class_header and storage_class:
            headers[provider.storage_class_header] = storage_class
        if metadata:
            headers[provider.metadata_directive_header] = 'REPLACE'
            headers = boto.utils.merge_meta(headers, metadata, provider)
        elif not query_args: # Can't use this header with multi-part copy.
            headers[provider.metadata_directive_header] = 'COPY'
        response = self.connection.make_request('PUT', self.name, new_key_name,
                                                headers=headers,
                                                query_args=query_args)
        body = response.read()
        if response.status == 200:
            key = self.new_key(new_key_name)
            h = handler.XmlHandler(key, self)
            xml.sax.parseString(body, h)
            if hasattr(key, 'Error'):
                raise provider.storage_copy_error(key.Code, key.Message, body)
            key.handle_version_headers(response)
            if preserve_acl:
                self.set_xml_acl(acl, new_key_name)
            return key
        else:
            raise provider.storage_response_error(response.status,
                                                  response.reason, body)