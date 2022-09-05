def login (self):
        """Log into ftp server and check the welcome message."""
        self.url_connection = ftplib.FTP(timeout=self.aggregate.config["timeout"])
        if log.is_debug(LOG_CHECK):
            self.url_connection.set_debuglevel(1)
        try:
            self.url_connection.connect(self.host, self.port)
            _user, _password = self.get_user_password()
            if _user is None:
                self.url_connection.login()
            elif _password is None:
                self.url_connection.login(_user)
            else:
                self.url_connection.login(_user, _password)
            info = self.url_connection.getwelcome()
            if info:
                # note that the info may change every time a user logs in,
                # so don't add it to the url_data info.
                log.debug(LOG_CHECK, "FTP info %s", info)
                pass
            else:
                raise LinkCheckerError(_("Got no answer from FTP server"))
        except EOFError as msg:
            raise LinkCheckerError(
                      _("Remote host has closed connection: %(msg)s") % str(msg))