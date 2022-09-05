def ssl_context(self, verify=True, cert_reqs=None,
                    check_hostname=False, certfile=None, keyfile=None,
                    cafile=None, capath=None, cadata=None, **kw):
        """Create a SSL context object.

        This method should not be called by from user code
        """
        assert ssl, 'SSL not supported'
        cafile = cafile or DEFAULT_CA_BUNDLE_PATH

        if verify is True:
            cert_reqs = ssl.CERT_REQUIRED
            check_hostname = True

        if isinstance(verify, str):
            cert_reqs = ssl.CERT_REQUIRED
            if os.path.isfile(verify):
                cafile = verify
            elif os.path.isdir(verify):
                capath = verify

        return ssl._create_unverified_context(cert_reqs=cert_reqs,
                                              check_hostname=check_hostname,
                                              certfile=certfile,
                                              keyfile=keyfile,
                                              cafile=cafile,
                                              capath=capath,
                                              cadata=cadata)