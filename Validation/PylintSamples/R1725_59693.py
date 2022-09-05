def _class_factory(base):
    """Create subclasses of ctypes.

    Positional arguments:
    base -- base class to subclass.

    Returns:
    New class definition.
    """
    class ClsPyPy(base):
        def __repr__(self):
            return repr(base(super(ClsPyPy, self).value))

        @classmethod
        def from_buffer(cls, ba):
            try:
                integer = struct.unpack_from(getattr(cls, '_type_'), ba)[0]
            except struct.error:
                len_ = len(ba)
                size = struct.calcsize(getattr(cls, '_type_'))
                if len_ < size:
                    raise ValueError('Buffer size too small ({0} instead of at least {1} bytes)'.format(len_, size))
                raise
            return cls(integer)

    class ClsPy26(base):
        def __repr__(self):
            return repr(base(super(ClsPy26, self).value))

        def __iter__(self):
            return iter(struct.pack(getattr(super(ClsPy26, self), '_type_'), super(ClsPy26, self).value))

    try:
        base.from_buffer(bytearray(base()))
    except TypeError:
        # Python2.6, ctypes cannot be converted to bytearrays.
        return ClsPy26
    except AttributeError:
        # PyPy on my Raspberry Pi, ctypes don't have from_buffer attribute.
        return ClsPyPy
    except ValueError:
        # PyPy on Travis CI, from_buffer cannot handle non-buffer() bytearrays.
        return ClsPyPy
    return base