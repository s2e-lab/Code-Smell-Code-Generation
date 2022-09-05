def blob_generator(self):
        """Create a blob generator."""

        # pylint: disable:F0401,W0612
        import aa    # pylint: disablF0401        # noqa
        from ROOT import EventFile    # pylint: disable F0401

        filename = self.filename
        log.info("Reading from file: {0}".format(filename))
        if not os.path.exists(filename):
            log.warning(filename + " not available: continue without it")

        try:
            event_file = EventFile(filename)
        except Exception:
            raise SystemExit("Could not open file")

        log.info("Generating blobs through new aanet API...")

        self.print("Reading metadata using 'JPrintMeta'")
        meta_parser = MetaParser(filename=filename)
        meta = meta_parser.get_table()
        if meta is None:
            self.log.warning(
                "No metadata found, this means no data provenance!"
            )

        if self.bare:
            log.info("Skipping data conversion, only passing bare aanet data")
            for event in event_file:
                yield Blob({'evt': event, 'event_file': event_file})

        else:
            log.info("Unpacking aanet header into dictionary...")
            hdr = self._parse_header(event_file.header)
            if not hdr:
                log.info("Empty header dict found, skipping...")
                self.raw_header = None
            else:
                log.info("Converting Header dict to Table...")
                self.raw_header = self._convert_header_dict_to_table(hdr)
                log.info("Creating HDF5Header")
                self.header = HDF5Header.from_table(self.raw_header)
            for event in event_file:
                log.debug('Reading event...')
                blob = self._read_event(event, filename)
                log.debug('Reading header...')
                blob["RawHeader"] = self.raw_header
                blob["Header"] = self.header

                if meta is not None:
                    blob['Meta'] = meta

                self.group_id += 1
                yield blob

        del event_file