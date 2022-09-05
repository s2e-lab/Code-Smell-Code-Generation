def start(self):
        """
        Creates a SSL connection to the iDigi Server and sends a
        ConnectionRequest message.
        """
        self.log.info("Starting SSL Session for Monitor %s."
                      % self.monitor_id)
        if self.socket is not None:
            raise Exception("Socket already established for %s." % self)

        try:
            # Create socket, wrap in SSL and connect.
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            # Validate that certificate server uses matches what we expect.
            if self.ca_certs is not None:
                self.socket = ssl.wrap_socket(self.socket,
                                              cert_reqs=ssl.CERT_REQUIRED,
                                              ca_certs=self.ca_certs)
            else:
                self.socket = ssl.wrap_socket(self.socket)

            self.socket.connect((self.client.hostname, PUSH_SECURE_PORT))
            self.socket.setblocking(0)
        except Exception as exception:
            self.socket.close()
            self.socket = None
            raise exception

        self.send_connection_request()