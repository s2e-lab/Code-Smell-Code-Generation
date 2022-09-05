def tempnam():
    ''' returns a temporary file-name '''

    # prevent os.tmpname from printing an error...
    stderr = sys.stderr
    try:
        sys.stderr = cStringIO.StringIO()
        return os.tempnam(None, 'tess_')
    finally:
        sys.stderr = stderr