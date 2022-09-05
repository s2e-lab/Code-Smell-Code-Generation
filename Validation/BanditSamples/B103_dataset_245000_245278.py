def get_mysql_password_on_disk(self, username=None, password=None):
        """Retrieve, generate or store a mysql password for the provided
        username on disk."""
        if username:
            template = self.user_passwd_file_template
            passwd_file = template.format(username)
        else:
            passwd_file = self.root_passwd_file_template

        _password = None
        if os.path.exists(passwd_file):
            log("Using existing password file '%s'" % passwd_file, level=DEBUG)
            with open(passwd_file, 'r') as passwd:
                _password = passwd.read().strip()
        else:
            log("Generating new password file '%s'" % passwd_file, level=DEBUG)
            if not os.path.isdir(os.path.dirname(passwd_file)):
                # NOTE: need to ensure this is not mysql root dir (which needs
                # to be mysql readable)
                mkdir(os.path.dirname(passwd_file), owner='root', group='root',
                      perms=0o770)
                # Force permissions - for some reason the chmod in makedirs
                # fails
                os.chmod(os.path.dirname(passwd_file), 0o770)

            _password = password or pwgen(length=32)
            write_file(passwd_file, _password, owner='root', group='root',
                       perms=0o660)

        return _password