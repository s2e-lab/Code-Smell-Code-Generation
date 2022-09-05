def upload_file(self, container, file_or_path, obj_name=None,
            content_type=None, etag=None, content_encoding=None, ttl=None,
            content_length=None, return_none=False, headers=None,
            metadata=None, extra_info=None):
        """
        Uploads the specified file to the container. If no name is supplied,
        the file's name will be used. Either a file path or an open file-like
        object may be supplied. A StorageObject reference to the uploaded file
        will be returned, unless 'return_none' is set to True.

        You may optionally set the `content_type` and `content_encoding`
        parameters; pyrax will create the appropriate headers when the object
        is stored.

        If the size of the file is known, it can be passed as `content_length`.

        If you wish for the object to be temporary, specify the time it should
        be stored in seconds in the `ttl` parameter. If this is specified, the
        object will be deleted after that number of seconds.

        The 'extra_info' parameter is included for backwards compatibility. It
        is no longer used at all, and will not be modified with swiftclient
        info, since swiftclient is not used any more.
        """
        return self.create_object(container, file_or_path=file_or_path,
                obj_name=obj_name, content_type=content_type, etag=etag,
                content_encoding=content_encoding, ttl=ttl, headers=headers,
                metadata=metadata, return_none=return_none)