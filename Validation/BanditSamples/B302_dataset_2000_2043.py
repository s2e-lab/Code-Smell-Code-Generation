def run(self, cmd, stdin=None, marshal_output=True, **kwargs):
        """Runs a p4 command and returns a list of dictionary objects

        :param cmd: Command to run
        :type cmd: list
        :param stdin: Standard Input to send to the process
        :type stdin: str
        :param marshal_output: Whether or not to marshal the output from the command
        :type marshal_output: bool
        :param kwargs: Passes any other keyword arguments to subprocess
        :raises: :class:`.error.CommandError`
        :returns: list, records of results
        """
        records = []
        args = [self._executable, "-u", self._user, "-p", self._port]

        if self._client:
            args += ["-c", str(self._client)]

        if marshal_output:
            args.append('-G')

        if isinstance(cmd, six.string_types):
            raise ValueError('String commands are not supported, please use a list')

        args += cmd

        command = ' '.join(args)

        startupinfo = None
        if os.name == 'nt':
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

        proc = subprocess.Popen(
            args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            startupinfo=startupinfo,
            **kwargs
        )

        if stdin:
            proc.stdin.write(six.b(stdin))

        if marshal_output:
            try:
                while True:
                    record = marshal.load(proc.stdout)
                    if record.get(b'code', '') == b'error' and record[b'severity'] >= self._level:
                        proc.stdin.close()
                        proc.stdout.close()
                        raise errors.CommandError(record[b'data'], record, command)
                    if isinstance(record, dict):
                        if six.PY2:
                            records.append(record)
                        else:
                            records.append({str(k, 'utf8'): str(v) if isinstance(v, int) else str(v, 'utf8', errors='ignore') for k, v in record.items()})
            except EOFError:
                pass

            stdout, stderr = proc.communicate()
        else:
            records, stderr = proc.communicate()

        if stderr:
            raise errors.CommandError(stderr, command)

        return records