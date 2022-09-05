def _read_frame(cls, reader):
        '''
        Read a single frame from a Reader.  Will return None if there is an
        incomplete frame in the stream.

        Raise MissingFooter if there's a problem reading the footer byte.
        '''
        frame_type = reader.read_octet()
        channel_id = reader.read_short()
        size = reader.read_long()

        payload = Reader(reader, reader.tell(), size)

        # Seek to end of payload
        reader.seek(size, 1)

        ch = reader.read_octet()  # footer
        if ch != 0xce:
            raise Frame.FormatError(
                'Framing error, unexpected byte: %x.  frame type %x. channel %d, payload size %d',
                ch, frame_type, channel_id, size)

        frame_class = cls._frame_type_map.get(frame_type)
        if not frame_class:
            raise Frame.InvalidFrameType("Unknown frame type %x", frame_type)
        return frame_class.parse(channel_id, payload)