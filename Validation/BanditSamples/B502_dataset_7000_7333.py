def proxy_manager_for(self, proxy, **proxy_kwargs):
        """Ensure cipher and Tlsv1"""
        context = create_urllib3_context(ciphers=self.CIPHERS,
                                         ssl_version=ssl.PROTOCOL_TLSv1)
        proxy_kwargs['ssl_context'] = context
        return super(TLSv1Adapter, self).proxy_manager_for(proxy,
                                                           **proxy_kwargs)