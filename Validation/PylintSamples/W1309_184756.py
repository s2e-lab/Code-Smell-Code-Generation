def connect_channels(self, channels):
        """Connect the provided channels"""
        self.log.info(f"Connecting to channels...")
        for chan in channels:
            chan.connect(self.sock)
            self.log.info(f"\t{chan.channel}")