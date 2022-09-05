def xmlrpc_run(self, port=24444, bind='127.0.0.1', logRequests=False):
        '''Run xmlrpc server'''
        import umsgpack
        from pyspider.libs.wsgi_xmlrpc import WSGIXMLRPCApplication
        try:
            from xmlrpc.client import Binary
        except ImportError:
            from xmlrpclib import Binary

        application = WSGIXMLRPCApplication()

        application.register_function(self.quit, '_quit')
        application.register_function(self.size)

        def sync_fetch(task):
            result = self.sync_fetch(task)
            result = Binary(umsgpack.packb(result))
            return result
        application.register_function(sync_fetch, 'fetch')

        def dump_counter(_time, _type):
            return self._cnt[_time].to_dict(_type)
        application.register_function(dump_counter, 'counter')

        import tornado.wsgi
        import tornado.ioloop
        import tornado.httpserver

        container = tornado.wsgi.WSGIContainer(application)
        self.xmlrpc_ioloop = tornado.ioloop.IOLoop()
        self.xmlrpc_server = tornado.httpserver.HTTPServer(container, io_loop=self.xmlrpc_ioloop)
        self.xmlrpc_server.listen(port=port, address=bind)
        logger.info('fetcher.xmlrpc listening on %s:%s', bind, port)
        self.xmlrpc_ioloop.start()