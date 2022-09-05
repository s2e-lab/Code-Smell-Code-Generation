def connect(self, always_log_errors=True):
        """Connect to an Amazon Fire TV device.

        Will attempt to establish ADB connection to the given host.
        Failure sets state to UNKNOWN and disables sending actions.

        :returns: True if successful, False otherwise
        """
        self._adb_lock.acquire(**LOCK_KWARGS)
        try:
            if not self.adb_server_ip:
                # python-adb
                try:
                    if self.adbkey:
                        signer = Signer(self.adbkey)

                        # Connect to the device
                        self._adb = adb_commands.AdbCommands().ConnectDevice(serial=self.host, rsa_keys=[signer], default_timeout_ms=9000)
                    else:
                        self._adb = adb_commands.AdbCommands().ConnectDevice(serial=self.host, default_timeout_ms=9000)

                    # ADB connection successfully established
                    self._available = True

                except socket_error as serr:
                    if self._available or always_log_errors:
                        if serr.strerror is None:
                            serr.strerror = "Timed out trying to connect to ADB device."
                        logging.warning("Couldn't connect to host: %s, error: %s", self.host, serr.strerror)

                    # ADB connection attempt failed
                    self._adb = None
                    self._available = False

                finally:
                    return self._available

            else:
                # pure-python-adb
                try:
                    self._adb_client = AdbClient(host=self.adb_server_ip, port=self.adb_server_port)
                    self._adb_device = self._adb_client.device(self.host)
                    self._available = bool(self._adb_device)

                except:
                    self._available = False

                finally:
                    return self._available

        finally:
            self._adb_lock.release()