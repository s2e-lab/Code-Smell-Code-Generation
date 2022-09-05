def list_documents(
        self,
        parent,
        collection_id,
        page_size=None,
        order_by=None,
        mask=None,
        transaction=None,
        read_time=None,
        show_missing=None,
        retry=google.api_core.gapic_v1.method.DEFAULT,
        timeout=google.api_core.gapic_v1.method.DEFAULT,
        metadata=None,
    ):
        """
        Lists documents.

        Example:
            >>> from google.cloud import firestore_v1beta1
            >>>
            >>> client = firestore_v1beta1.FirestoreClient()
            >>>
            >>> parent = client.any_path_path('[PROJECT]', '[DATABASE]', '[DOCUMENT]', '[ANY_PATH]')
            >>>
            >>> # TODO: Initialize `collection_id`:
            >>> collection_id = ''
            >>>
            >>> # Iterate over all results
            >>> for element in client.list_documents(parent, collection_id):
            ...     # process element
            ...     pass
            >>>
            >>>
            >>> # Alternatively:
            >>>
            >>> # Iterate over results one page at a time
            >>> for page in client.list_documents(parent, collection_id).pages:
            ...     for element in page:
            ...         # process element
            ...         pass

        Args:
            parent (str): The parent resource name. In the format:
                ``projects/{project_id}/databases/{database_id}/documents`` or
                ``projects/{project_id}/databases/{database_id}/documents/{document_path}``.
                For example: ``projects/my-project/databases/my-database/documents`` or
                ``projects/my-project/databases/my-database/documents/chatrooms/my-chatroom``
            collection_id (str): The collection ID, relative to ``parent``, to list. For example:
                ``chatrooms`` or ``messages``.
            page_size (int): The maximum number of resources contained in the
                underlying API response. If page streaming is performed per-
                resource, this parameter does not affect the return value. If page
                streaming is performed per-page, this determines the maximum number
                of resources in a page.
            order_by (str): The order to sort results by. For example: ``priority desc, name``.
            mask (Union[dict, ~google.cloud.firestore_v1beta1.types.DocumentMask]): The fields to return. If not set, returns all fields.

                If a document has a field that is not present in this mask, that field
                will not be returned in the response.

                If a dict is provided, it must be of the same form as the protobuf
                message :class:`~google.cloud.firestore_v1beta1.types.DocumentMask`
            transaction (bytes): Reads documents in a transaction.
            read_time (Union[dict, ~google.cloud.firestore_v1beta1.types.Timestamp]): Reads documents as they were at the given time.
                This may not be older than 60 seconds.

                If a dict is provided, it must be of the same form as the protobuf
                message :class:`~google.cloud.firestore_v1beta1.types.Timestamp`
            show_missing (bool): If the list should show missing documents. A missing document is a
                document that does not exist but has sub-documents. These documents will
                be returned with a key but will not have fields,
                ``Document.create_time``, or ``Document.update_time`` set.

                Requests with ``show_missing`` may not specify ``where`` or
                ``order_by``.
            retry (Optional[google.api_core.retry.Retry]):  A retry object used
                to retry requests. If ``None`` is specified, requests will not
                be retried.
            timeout (Optional[float]): The amount of time, in seconds, to wait
                for the request to complete. Note that if ``retry`` is
                specified, the timeout applies to each individual attempt.
            metadata (Optional[Sequence[Tuple[str, str]]]): Additional metadata
                that is provided to the method.

        Returns:
            A :class:`~google.gax.PageIterator` instance. By default, this
            is an iterable of :class:`~google.cloud.firestore_v1beta1.types.Document` instances.
            This object can also be configured to iterate over the pages
            of the response through the `options` parameter.

        Raises:
            google.api_core.exceptions.GoogleAPICallError: If the request
                    failed for any reason.
            google.api_core.exceptions.RetryError: If the request failed due
                    to a retryable error and retry attempts failed.
            ValueError: If the parameters are invalid.
        """
        # Wrap the transport method to add retry and timeout logic.
        if "list_documents" not in self._inner_api_calls:
            self._inner_api_calls[
                "list_documents"
            ] = google.api_core.gapic_v1.method.wrap_method(
                self.transport.list_documents,
                default_retry=self._method_configs["ListDocuments"].retry,
                default_timeout=self._method_configs["ListDocuments"].timeout,
                client_info=self._client_info,
            )

        # Sanity check: We have some fields which are mutually exclusive;
        # raise ValueError if more than one is sent.
        google.api_core.protobuf_helpers.check_oneof(
            transaction=transaction, read_time=read_time
        )

        request = firestore_pb2.ListDocumentsRequest(
            parent=parent,
            collection_id=collection_id,
            page_size=page_size,
            order_by=order_by,
            mask=mask,
            transaction=transaction,
            read_time=read_time,
            show_missing=show_missing,
        )
        iterator = google.api_core.page_iterator.GRPCIterator(
            client=None,
            method=functools.partial(
                self._inner_api_calls["list_documents"],
                retry=retry,
                timeout=timeout,
                metadata=metadata,
            ),
            request=request,
            items_field="documents",
            request_token_field="page_token",
            response_token_field="next_page_token",
        )
        return iterator