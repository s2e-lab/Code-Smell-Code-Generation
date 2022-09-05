def establish_connection(self, width=None, height=None):
        """Establish SSH connection to the network device

        Timeout will generate a NetMikoTimeoutException
        Authentication failure will generate a NetMikoAuthenticationException

        width and height are needed for Fortinet paging setting.

        :param width: Specified width of the VT100 terminal window
        :type width: int

        :param height: Specified height of the VT100 terminal window
        :type height: int
        """
        if self.protocol == "telnet":
            self.remote_conn = telnetlib.Telnet(
                self.host, port=self.port, timeout=self.timeout
            )
            self.telnet_login()
        elif self.protocol == "serial":
            self.remote_conn = serial.Serial(**self.serial_settings)
            self.serial_login()
        elif self.protocol == "ssh":
            ssh_connect_params = self._connect_params_dict()
            self.remote_conn_pre = self._build_ssh_client()

            # initiate SSH connection
            try:
                self.remote_conn_pre.connect(**ssh_connect_params)
            except socket.error:
                self.paramiko_cleanup()
                msg = "Connection to device timed-out: {device_type} {ip}:{port}".format(
                    device_type=self.device_type, ip=self.host, port=self.port
                )
                raise NetMikoTimeoutException(msg)
            except paramiko.ssh_exception.AuthenticationException as auth_err:
                self.paramiko_cleanup()
                msg = "Authentication failure: unable to connect {device_type} {ip}:{port}".format(
                    device_type=self.device_type, ip=self.host, port=self.port
                )
                msg += self.RETURN + text_type(auth_err)
                raise NetMikoAuthenticationException(msg)

            if self.verbose:
                print(
                    "SSH connection established to {}:{}".format(self.host, self.port)
                )

            # Use invoke_shell to establish an 'interactive session'
            if width and height:
                self.remote_conn = self.remote_conn_pre.invoke_shell(
                    term="vt100", width=width, height=height
                )
            else:
                self.remote_conn = self.remote_conn_pre.invoke_shell()

            self.remote_conn.settimeout(self.blocking_timeout)
            if self.keepalive:
                self.remote_conn.transport.set_keepalive(self.keepalive)
            self.special_login_handler()
            if self.verbose:
                print("Interactive SSH session established")
        return ""