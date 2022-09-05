def start(self):
        """Start the subprocess."""
        c_out, c_err = (open(self.path('cmd.stdout'), 'w'),
                        open(self.path('cmd.stderr'), 'w'))
        kw = self.kw.copy()
        kw['stdout'] = c_out
        kw['stderr'] = c_err
        if not kw.get('cwd', None):
            kw['cwd'] = os.getcwd()
        pr = subprocess.Popen(self.cmd_args, **kw)
        with open(self.path('cmd.pid'), 'w') as f:
            f.write(str(pr.pid))