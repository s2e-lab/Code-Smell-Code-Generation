def find_free(cls, vrf, args):
        """ Finds a free prefix.

            Maps to the function
            :py:func:`nipap.backend.Nipap.find_free_prefix` in the backend.
            Please see the documentation for the backend function for
            information regarding input arguments and return values.
        """

        xmlrpc = XMLRPCConnection()
        q = {
            'args': args,
            'auth': AuthOptions().options
        }

        # sanity checks
        if isinstance(vrf, VRF):
            q['vrf'] = { 'id': vrf.id }
        elif vrf is None:
            q['vrf'] = None
        else:
            raise NipapValueError('vrf parameter must be instance of VRF class')

        # run XML-RPC query
        try:
            find_res = xmlrpc.connection.find_free_prefix(q)
        except xmlrpclib.Fault as xml_fault:
            raise _fault_to_exception(xml_fault)
        pass

        return find_res