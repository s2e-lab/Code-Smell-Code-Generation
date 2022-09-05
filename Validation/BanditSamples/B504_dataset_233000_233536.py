def authenticate(self):
        '''
        Authenticate the client and return the private
        and signature keys.

        Establish a connection through a secured socket,
        then do the handshake using the napalm-logs
        auth algorithm.
        '''
        log.debug('Authenticate to %s:%d, using the certificate %s',
                  self.address, self.port, self.certificate)
        if ':' in self.address:
            skt_ver = socket.AF_INET6
        else:
            skt_ver = socket.AF_INET
        skt = socket.socket(skt_ver, socket.SOCK_STREAM)
        self.ssl_skt = ssl.wrap_socket(skt,
                                       ca_certs=self.certificate,
                                       cert_reqs=ssl.CERT_REQUIRED)
        try:
            self.ssl_skt.connect((self.address, self.port))
            self.auth_try_id = 0
        except socket.error as err:
            log.error('Unable to open the SSL socket.')
            self.auth_try_id += 1
            if not self.max_try or self.auth_try_id < self.max_try:
                log.error('Trying to authenticate again in %d seconds', self.timeout)
                time.sleep(self.timeout)
                self.authenticate()
            log.critical('Giving up, unable to authenticate to %s:%d using the certificate %s',
                         self.address, self.port, self.certificate)
            raise ClientConnectException(err)

        # Explicit INIT
        self.ssl_skt.write(defaults.MAGIC_REQ)
        # Receive the private key
        private_key = self.ssl_skt.recv(defaults.BUFFER_SIZE)
        # Send back explicit ACK
        self.ssl_skt.write(defaults.MAGIC_ACK)
        # Read the hex of the verification key
        verify_key_hex = self.ssl_skt.recv(defaults.BUFFER_SIZE)
        # Send back explicit ACK
        self.ssl_skt.write(defaults.MAGIC_ACK)
        self.priv_key = nacl.secret.SecretBox(private_key)
        self.verify_key = nacl.signing.VerifyKey(verify_key_hex, encoder=nacl.encoding.HexEncoder)