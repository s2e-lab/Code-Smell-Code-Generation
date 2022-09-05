def _make_tmp_path(self, conn):
        ''' make and return a temporary path on a remote box '''

        basefile = 'ansible-%s-%s' % (time.time(), random.randint(0, 2**48))
        basetmp = os.path.join(C.DEFAULT_REMOTE_TMP, basefile)
        if self.sudo and self.sudo_user != 'root':
            basetmp = os.path.join('/tmp', basefile)

        cmd = 'mkdir -p %s' % basetmp
        if self.remote_user != 'root':
            cmd += ' && chmod a+rx %s' % basetmp
        cmd += ' && echo %s' % basetmp

        result = self._low_level_exec_command(conn, cmd, None, sudoable=False)
        rc = utils.last_non_blank_line(result['stdout']).strip() + '/'
        return rc