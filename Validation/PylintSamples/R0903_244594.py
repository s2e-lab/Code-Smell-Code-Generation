def patch_class(input_class):
    """Create a new class based on the input_class.

    :param class input_class:  The class to patch.
    :rtype class:
    """
    class Instantiator(object):
        @classmethod
        def _doubles__new__(self, *args, **kwargs):
            pass

    new_class = type(input_class.__name__, (input_class, Instantiator), {})

    return new_class