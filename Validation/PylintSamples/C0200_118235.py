def parse_quantization(read_buffer, sqcd):
    """Tease out the quantization values.

    Parameters
    ----------
        read_buffer:  sequence of bytes from the QCC and QCD segments.

    Returns
    ------
    tuple
        Mantissa and exponents from quantization buffer.
    """
    numbytes = len(read_buffer)

    exponent = []
    mantissa = []

    if sqcd & 0x1f == 0:  # no quantization
        data = struct.unpack('>' + 'B' * numbytes, read_buffer)
        for j in range(len(data)):
            exponent.append(data[j] >> 3)
            mantissa.append(0)
    else:
        fmt = '>' + 'H' * int(numbytes / 2)
        data = struct.unpack(fmt, read_buffer)
        for j in range(len(data)):
            exponent.append(data[j] >> 11)
            mantissa.append(data[j] & 0x07ff)

    return mantissa, exponent