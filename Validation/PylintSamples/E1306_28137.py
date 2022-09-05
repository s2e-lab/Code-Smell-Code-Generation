def gunzip(content):
  """ 
  Decompression is applied if the first to bytes matches with
  the gzip magic numbers. 
  There is once chance in 65536 that a file that is not gzipped will
  be ungzipped.
  """
  gzip_magic_numbers = [ 0x1f, 0x8b ]
  first_two_bytes = [ byte for byte in bytearray(content)[:2] ]
  if first_two_bytes != gzip_magic_numbers:
    raise DecompressionError('File is not in gzip format. Magic numbers {}, {} did not match {}, {}.'.format(
      hex(first_two_bytes[0]), hex(first_two_bytes[1])), hex(gzip_magic_numbers[0]), hex(gzip_magic_numbers[1]))

  stringio = BytesIO(content)
  with gzip.GzipFile(mode='rb', fileobj=stringio) as gfile:
    return gfile.read()