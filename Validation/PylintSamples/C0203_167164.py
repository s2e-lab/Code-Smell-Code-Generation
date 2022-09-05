def create_dummy_class(klass, dependency):
    """
    When a dependency of a class is not available, create a dummy class which throws ImportError when used.

    Args:
        klass (str): name of the class.
        dependency (str): name of the dependency.

    Returns:
        class: a class object
    """
    assert not building_rtfd()

    class _DummyMetaClass(type):
        # throw error on class attribute access
        def __getattr__(_, __):
            raise AttributeError("Cannot import '{}', therefore '{}' is not available".format(dependency, klass))

    @six.add_metaclass(_DummyMetaClass)
    class _Dummy(object):
        # throw error on constructor
        def __init__(self, *args, **kwargs):
            raise ImportError("Cannot import '{}', therefore '{}' is not available".format(dependency, klass))

    return _Dummy