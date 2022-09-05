def run(self):
        """Process events such as incoming data.

        This method blocks indefinitely. It will only return after the
        connection to the server is closed.
        """
        self._stop = False # Allow re-starting the event loop
        while not self._stop:
            try:
                self._buffer += self.socket.recv(4096)
            except socket.error:
                raise

            lines = self._buffer.split("\n")
            self._buffer = lines.pop() # Last line may not have been fully read
            for line in lines:
                line = line.rstrip("\r")
                _log.debug("%s --> %s", self.server.host, line)
                self.dispatch_event("LINE", line)
                self.dispatch_event("ACTIVITY")