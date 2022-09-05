def enable(identifier = None, *args, **kwargs):
	''' Enables a specific cache for the current session. Remember that is has to be registered. '''

	global cache
	if not identifier:
		for item in (config['default-caches'] + ['NoCache']):
			if caches.has_key(item):
				debug('Enabling default cache %s...' % (item,))
				cache = caches[item](*args, **kwargs)
				if not cache.status():
					warning('%s could not be loaded. Is the backend running (%s:%d)?' % (item, cache.server, cache.port))
					continue
				# This means that the cache backend was set up successfully
				break
			else:
				debug('Cache backend %s is not registered. Are all requirements satisfied?' % (item,))
	elif caches.has_key(identifier):
		debug('Enabling cache %s...' % (identifier,))
		previouscache = cache
		cache = caches[identifier](*args, **kwargs)
		if not cache.status():
			warning('%s could not be loaded. Is the backend running (%s:%d)?' % (identifier, cache.server, cache.port))
			cache = previouscache
	else:
		debug('Cache backend %s is not registered. Are all requirements satisfied?' % (identifier,))