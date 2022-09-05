def _read_pyc(source, pyc):
    """Possibly read a pytest pyc containing rewritten code.

    Return rewritten code if successful or None if not.
    """
    try:
        fp = open(pyc, "rb")
    except IOError:
        return None
    try:
        try:
            mtime = int(os.stat(source).st_mtime)
            data = fp.read(8)
        except EnvironmentError:
            return None
        # Check for invalid or out of date pyc file.
        if (len(data) != 8 or data[:4] != imp.get_magic() or
                struct.unpack("<l", data[4:])[0] != mtime):
            return None
        co = marshal.load(fp)
        if not isinstance(co, types.CodeType):
            # That's interesting....
            return None
        return co
    finally:
        fp.close()