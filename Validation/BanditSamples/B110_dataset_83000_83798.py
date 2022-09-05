def flush(self):
        """Flush all streams."""
        if self.__logFileStream is not None:
            try:
                self.__logFileStream.flush()
            except:
                pass
            try:
                os.fsync(self.__logFileStream.fileno())
            except:
                pass
        if self.__stdout is not None:
            try:
                self.__stdout.flush()
            except:
                pass
            try:
                os.fsync(self.__stdout.fileno())
            except:
                pass