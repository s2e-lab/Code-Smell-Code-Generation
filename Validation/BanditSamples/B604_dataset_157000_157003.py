def _persist(self):
        """
        Run the command inside a thread so that we can catch output for each
        line as it comes in and display it.
        """
        # run the block/command
        for command in self.commands:
            try:
                process = Popen(
                    [command],
                    stdout=PIPE,
                    stderr=PIPE,
                    universal_newlines=True,
                    env=self.env,
                    shell=True,
                )
            except Exception as e:
                retcode = process.poll()
                msg = "Command '{cmd}' {error} retcode {retcode}"
                self.py3.log(msg.format(cmd=command, error=e, retcode=retcode))

            # persistent blocklet output can be of two forms.  Either each row
            # of the output is on a new line this is much easier to deal with)
            # or else the output can be continuous and just flushed when ready.
            # The second form is more tricky, if we find newlines then we
            # switch to easy parsing of the output.

            # When we have output we store in self.persistent_output and then
            # trigger the module to update.

            fd = process.stdout.fileno()
            fl = fcntl.fcntl(fd, fcntl.F_GETFL)
            has_newlines = False
            while True:
                line = process.stdout.read(1)
                # switch to a non-blocking read as we do not know the output
                # length
                fcntl.fcntl(fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)
                line += process.stdout.read(1024)
                # switch back to blocking so we can wait for the next output
                fcntl.fcntl(fd, fcntl.F_SETFL, fl)
                if process.poll():
                    break
                if self.py3.is_python_2():
                    line = line.decode("utf-8")
                self.persistent_output = line
                self.py3.update()
                if line[-1] == "\n":
                    has_newlines = True
                    break
                if line == "":
                    break
            if has_newlines:
                msg = "Switch to newline persist method {cmd}"
                self.py3.log(msg.format(cmd=command))
                # just read the output in a sane manner
                for line in iter(process.stdout.readline, b""):
                    if process.poll():
                        break
                    if self.py3.is_python_2():
                        line = line.decode("utf-8")
                    self.persistent_output = line
                    self.py3.update()
        self.py3.log("command exited {cmd}".format(cmd=command))
        self.persistent_output = "Error\nError\n{}".format(
            self.py3.COLOR_ERROR or self.py3.COLOR_BAD
        )
        self.py3.update()