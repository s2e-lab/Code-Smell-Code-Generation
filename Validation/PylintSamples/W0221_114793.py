def run_tornado(self, args):
        """ Tornado dev server implementation """
        server = self
        import tornado.ioloop
        import tornado.web
        import tornado.websocket

        ioloop = tornado.ioloop.IOLoop.current()

        class DevWebSocketHandler(tornado.websocket.WebSocketHandler):
            def open(self):
                super(DevWebSocketHandler, self).open()
                server.on_open(self)

            def on_message(self, message):
                server.on_message(self, message)

            def on_close(self):
                super(DevWebSocketHandler, self).on_close()
                server.on_close(self)

        class MainHandler(tornado.web.RequestHandler):
            def get(self):
                self.write(server.index_page)

        #: Set the call later method
        server.call_later = ioloop.call_later
        server.add_callback = ioloop.add_callback

        app = tornado.web.Application([
            (r"/", MainHandler),
            (r"/dev", DevWebSocketHandler),
        ])

        app.listen(self.port)
        print("Tornado Dev server started on {}".format(self.port))
        ioloop.start()