def run(self, timeout=-1):
        """
        Run the subprocess.

        Arguments:
            timeout (optional) If a positive real value, then timout after
                the given number of seconds.

        Raises:
            SubprocessError If subprocess has not completed after "timeout"
                seconds.
        """
        def target():
            self.process = subprocess.Popen(self.cmd,
                                            stdout=self.stdout_dest,
                                            stderr=self.stderr_dest,
                                            shell=self.shell)
            stdout, stderr = self.process.communicate()

            # Decode output if the user wants, and if there is any.
            if self.decode_out:
                if stdout:
                    self.stdout = stdout.decode("utf-8")
                if stderr:
                    self.stderr = stderr.decode("utf-8")

        thread = threading.Thread(target=target)
        thread.start()

        if timeout > 0:
            thread.join(timeout)
            if thread.is_alive():
                self.process.terminate()
                thread.join()
                raise SubprocessError(("Reached timeout after {t} seconds"
                                       .format(t=timeout)))
        else:
            thread.join()

        return self.process.returncode, self.stdout, self.stderr