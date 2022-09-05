def sysinfo2magic(version_info=sys.version_info):
    """Convert a list sys.versions_info compatible list into a 'canonic'
    floating-point number which that can then be used to look up a
    magic number.  Note that this can raise an exception.
    """

    # FIXME: DRY with sysinfo2float()
    vers_str = '.'.join([str(v) for v in version_info[0:3]])
    if version_info[3] != 'final':
        vers_str += ''.join([str(v) for v in version_info[3:]])

    if IS_PYPY:
        vers_str += 'pypy'
    else:
        try:
            import platform
            platform = platform.python_implementation()
            if platform in ('Jython', 'Pyston'):
                vers_str += platform
                pass
        except ImportError:
            # Python may be too old, e.g. < 2.6 or implementation may
            # just not have platform
            pass

    return magics[vers_str]