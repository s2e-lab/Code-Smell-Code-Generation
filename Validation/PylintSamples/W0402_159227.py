def _is_dynamic(module):
    """
    Return True if the module is special module that cannot be imported by its
    name.
    """
    # Quick check: module that have __file__ attribute are not dynamic modules.
    if hasattr(module, '__file__'):
        return False

    if hasattr(module, '__spec__'):
        return module.__spec__ is None
    else:
        # Backward compat for Python 2
        import imp
        try:
            path = None
            for part in module.__name__.split('.'):
                if path is not None:
                    path = [path]
                f, path, description = imp.find_module(part, path)
                if f is not None:
                    f.close()
        except ImportError:
            return True
        return False