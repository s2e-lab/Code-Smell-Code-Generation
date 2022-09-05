def FileWriter(path):
    """Context manager for a ninja_syntax object writing to a file."""
    try:
        os.makedirs(os.path.dirname(path))
    except OSError:
        pass
    f = open(path, 'w')
    yield ninja_syntax.Writer(f)
    f.close()