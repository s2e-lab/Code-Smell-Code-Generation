def ExecQuery(self, QueryLanguage, Query, namespace=None, **extra):
        # pylint: disable=invalid-name
        """
        Execute a query in a namespace.

        This method performs the ExecQuery operation
        (see :term:`DSP0200`). See :ref:`WBEM operations` for a list of all
        methods performing such operations.

        If the operation succeeds, this method returns.
        Otherwise, this method raises an exception.

        Parameters:

          QueryLanguage (:term:`string`):
            Name of the query language used in the `Query` parameter, e.g.
            "DMTF:CQL" for CIM Query Language, and "WQL" for WBEM Query
            Language.

          Query (:term:`string`):
            Query string in the query language specified in the `QueryLanguage`
            parameter.

          namespace (:term:`string`):
            Name of the CIM namespace to be used (case independent).

            Leading and trailing slash characters will be stripped. The lexical
            case will be preserved.

            If `None`, the default namespace of the connection object will be
            used.

          **extra :
            Additional keyword arguments are passed as additional operation
            parameters to the WBEM server.
            Note that :term:`DSP0200` does not define any additional parameters
            for this operation.

        Returns:

            A list of :class:`~pywbem.CIMInstance` objects that represents
            the query result.

            These instances have their `path` attribute set to identify
            their creation class and the target namespace of the query, but
            they are not addressable instances.

        Raises:

            Exceptions described in :class:`~pywbem.WBEMConnection`.
        """

        exc = None
        instances = None
        method_name = 'ExecQuery'

        if self._operation_recorders:
            self.operation_recorder_reset()
            self.operation_recorder_stage_pywbem_args(
                method=method_name,
                QueryLanguage=QueryLanguage,
                Query=Query,
                namespace=namespace,
                **extra)

        try:

            stats = self.statistics.start_timer(method_name)
            namespace = self._iparam_namespace_from_namespace(namespace)

            result = self._imethodcall(
                method_name,
                namespace,
                QueryLanguage=QueryLanguage,
                Query=Query,
                **extra)

            if result is None:
                instances = []
            else:
                instances = [x[2] for x in result[0][2]]

            for instance in instances:

                # The ExecQuery CIM-XML operation returns instances as any of
                # (VALUE.OBJECT | VALUE.OBJECTWITHLOCALPATH |
                # VALUE.OBJECTWITHPATH), i.e. classes or instances with or
                # without path which may or may not contain a namespace.

                # TODO: Fix current impl. that assumes instance with path.
                instance.path.namespace = namespace

            return instances

        except (CIMXMLParseError, XMLParseError) as exce:
            exce.request_data = self.last_raw_request
            exce.response_data = self.last_raw_reply
            exc = exce
            raise
        except Exception as exce:
            exc = exce
            raise
        finally:
            self._last_operation_time = stats.stop_timer(
                self.last_request_len, self.last_reply_len,
                self.last_server_response_time, exc)
            if self._operation_recorders:
                self.operation_recorder_stage_result(instances, exc)