def open(data_dir=nlpir.PACKAGE_DIR, encoding=ENCODING,
         encoding_errors=ENCODING_ERRORS, license_code=None):
    """Initializes the NLPIR API.

    This calls the function :func:`~pynlpir.nlpir.Init`.

    :param str data_dir: The absolute path to the directory that has NLPIR's
        `Data` directory (defaults to :data:`pynlpir.nlpir.PACKAGE_DIR`).
    :param str encoding: The encoding that the Chinese source text will be in
        (defaults to ``'utf_8'``). Possible values include ``'gbk'``,
        ``'utf_8'``, or ``'big5'``.
    :param str encoding_errors: The desired encoding error handling scheme.
        Possible values include ``'strict'``, ``'ignore'``, and ``'replace'``.
        The default error handler is 'strict' meaning that encoding errors
        raise :class:`ValueError` (or a more codec specific subclass, such
        as :class:`UnicodeEncodeError`).
    :param str license_code: The license code that should be used when
        initializing NLPIR. This is generally only used by commercial users.
    :raises RuntimeError: The NLPIR API failed to initialize. Sometimes, NLPIR
        leaves an error log in the current working directory or NLPIR's
        ``Data`` directory that provides more detailed messages (but this isn't
        always the case).
    :raises LicenseError: The NLPIR license appears to be missing or expired.

    """
    if license_code is None:
        license_code = ''
    global ENCODING
    if encoding.lower() in ('utf_8', 'utf-8', 'u8', 'utf', 'utf8'):
        ENCODING = 'utf_8'
        encoding_constant = nlpir.UTF8_CODE
    elif encoding.lower() in ('gbk', '936', 'cp936', 'ms936'):
        ENCODING = 'gbk'
        encoding_constant = nlpir.GBK_CODE
    elif encoding.lower() in ('big5', 'big5-tw', 'csbig5'):
        ENCODING = 'big5'
        encoding_constant = nlpir.BIG5_CODE
    else:
        raise ValueError("encoding must be one of 'utf_8', 'big5', or 'gbk'.")
    logger.debug("Initializing the NLPIR API: 'data_dir': '{}', 'encoding': "
                 "'{}', 'license_code': '{}'".format(
                     data_dir, encoding, license_code))

    global ENCODING_ERRORS
    if encoding_errors not in ('strict', 'ignore', 'replace'):
        raise ValueError("encoding_errors must be one of 'strict', 'ignore', "
                         "or 'replace'.")
    else:
        ENCODING_ERRORS = encoding_errors

    # Init in Python 3 expects bytes, not strings.
    if is_python3 and isinstance(data_dir, str):
        data_dir = _encode(data_dir)
    if is_python3 and isinstance(license_code, str):
        license_code = _encode(license_code)

    if not nlpir.Init(data_dir, encoding_constant, license_code):
        _attempt_to_raise_license_error(data_dir)
        raise RuntimeError("NLPIR function 'NLPIR_Init' failed.")
    else:
        logger.debug("NLPIR API initialized.")