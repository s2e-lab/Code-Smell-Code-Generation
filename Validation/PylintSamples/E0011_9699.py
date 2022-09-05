def blob_counter(self):
        """Create a blob counter."""
        import aa    # pylint: disablF0401        # noqa
        from ROOT import EventFile    # pylint: disable F0401

        try:
            event_file = EventFile(self.filename)
        except Exception:
            raise SystemExit("Could not open file")

        num_blobs = 0
        for event in event_file:
            num_blobs += 1

        return num_blobs