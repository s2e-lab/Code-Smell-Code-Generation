def get_short_module_name(module_name, obj_name):
    """ Get the shortest possible module name """
    scope = {}
    try:
        # Find out what the real object is supposed to be.
        exec('from %s import %s' % (module_name, obj_name), scope, scope)
        real_obj = scope[obj_name]
    except Exception:
        return module_name

    parts = module_name.split('.')
    short_name = module_name
    for i in range(len(parts) - 1, 0, -1):
        short_name = '.'.join(parts[:i])
        scope = {}
        try:
            exec('from %s import %s' % (short_name, obj_name), scope, scope)
            # Ensure shortened object is the same as what we expect.
            assert real_obj is scope[obj_name]
        except Exception:  # libraries can throw all sorts of exceptions...
            # get the last working module name
            short_name = '.'.join(parts[:(i + 1)])
            break
    return short_name