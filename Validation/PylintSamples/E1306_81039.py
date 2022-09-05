def load(self, f):
        """Load a PNG image"""
        SUPPORTED_COLOR_TYPES = (COLOR_TYPE_TRUECOLOR, COLOR_TYPE_TRUECOLOR_WITH_ALPHA)
        SAMPLES_PER_PIXEL = { COLOR_TYPE_TRUECOLOR: 3,
                              COLOR_TYPE_TRUECOLOR_WITH_ALPHA: 4 }

        assert f.read(8) == SIGNATURE

        chunks = iter(self.chunks(f))
        header = next(chunks)
        assert header[0] == b'IHDR'

        (width, height, bit_depth, color_type, compression,
         filter_type, interlace) = struct.unpack(b"!2I5B", header[1])

        if bit_depth != 8:
            raise ValueError('Unsupported PNG format (bit depth={}; must be 8)'.format(bit_depth))
        if compression != 0:
            raise ValueError('Unsupported PNG format (compression={}; must be 0)'.format(compression))
        if filter_type != 0:
            raise ValueError('Unsupported PNG format (filter_type={}; must be 0)'.format(filter_type))
        if interlace != 0:
            raise ValueError('Unsupported PNG format (interlace={}; must be 0)'.format(interlace))
        if color_type not in SUPPORTED_COLOR_TYPES:
            raise ValueError('Unsupported PNG format (color_type={}; must one of {})'.format(SUPPORTED_COLOR_TYPES))

        self.width = width
        self.height = height
        self.canvas = bytearray(self.bgcolor * width * height)
        bytes_per_pixel = SAMPLES_PER_PIXEL[color_type]
        bytes_per_row = bytes_per_pixel * width
        bytes_per_rgba_row = SAMPLES_PER_PIXEL[COLOR_TYPE_TRUECOLOR_WITH_ALPHA] * width
        bytes_per_scanline = bytes_per_row + 1

        # Python 2 requires the encode for struct.unpack
        scanline_fmt = ('!%dB' % bytes_per_scanline).encode('ascii')

        reader = ByteReader(chunks)

        old_row = None
        cursor = 0
        for row in range(height):
            scanline = reader.read(bytes_per_scanline)
            unpacked = list(struct.unpack(scanline_fmt, scanline))
            old_row = self.defilter(unpacked[1:], old_row, unpacked[0], bpp=bytes_per_pixel)
            rgba_row = old_row if color_type == COLOR_TYPE_TRUECOLOR_WITH_ALPHA else rgb2rgba(old_row)
            self.canvas[cursor:cursor + bytes_per_rgba_row] = rgba_row
            cursor += bytes_per_rgba_row