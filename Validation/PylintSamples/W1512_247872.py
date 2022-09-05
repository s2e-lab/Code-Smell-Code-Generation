def is_sequence(obj):
    """
    Helper function to determine sequences
    across Python 2.x and 3.x
    """
    try:
        from collections import Sequence
    except ImportError:
        from operator import isSequenceType
        return isSequenceType(obj)
    else:
        return isinstance(obj, Sequence)