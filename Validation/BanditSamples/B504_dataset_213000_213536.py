def build_socket(self):
        """
        Generate either an HTTPS or HTTP socket
        """
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(self.SOCKET_TIMEOUT)
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            # Check if TLS
            if self.request_object.protocol == 'https':
                self.sock = ssl.wrap_socket(self.sock, ciphers=self.CIPHERS)
            self.sock.connect(
                (self.request_object.dest_addr, self.request_object.port))
        except socket.error as msg:
            raise errors.TestError(
                'Failed to connect to server',
                {
                    'host': self.request_object.dest_addr,
                    'port': self.request_object.port,
                    'proto': self.request_object.protocol,
                    'message': msg,
                    'function': 'http.HttpUA.build_socket'
                })