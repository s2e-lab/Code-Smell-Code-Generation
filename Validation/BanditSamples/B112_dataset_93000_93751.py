def create_celerybeat_schedule(apps):
    """Create Celery beat schedule by get schedule from every installed app"""
    beat_schedule = {}
    for app in apps:
        try:
            config = import_object_by_string(app)
            module = importlib.import_module('{}.cron'.format(config.name))
        except Exception:
            try:
                module = importlib.import_module('{}.cron'.format(app))
            except Exception:
                continue

        if not (hasattr(module, 'schedule') and isinstance(module.schedule, dict)):
            logger.warning('{} has no schedule or schedule is not a dict'.format(module.__name__))
            continue

        # Add cron queue option
        for name, schedule in module.schedule.items():
            options = schedule.get('options', {})
            if 'queue' not in options:
                options['queue'] = 'cron'
                schedule['options'] = options

                beat_schedule[name] = schedule

    return beat_schedule