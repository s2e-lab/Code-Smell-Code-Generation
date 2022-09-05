def get_config_values(config_path, section, default='default'):
    """
    Parse ini config file and return a dict of values.

    The provided section overrides any values in default section.
    """
    values = {}

    if not os.path.isfile(config_path):
        raise IpaUtilsException(
            'Config file not found: %s' % config_path
        )

    config = configparser.ConfigParser()

    try:
        config.read(config_path)
    except Exception:
        raise IpaUtilsException(
            'Config file format invalid.'
        )

    try:
        values.update(config.items(default))
    except Exception:
        pass

    try:
        values.update(config.items(section))
    except Exception:
        pass

    return values