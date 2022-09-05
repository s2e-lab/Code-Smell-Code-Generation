def set_disk_cache(self, results, key=None):
        """Store result in disk cache with key matching model state."""
        if not getattr(self, 'disk_cache_location', False):
            self.init_disk_cache()
        disk_cache = shelve.open(self.disk_cache_location)
        key = self.model.hash if key is None else key
        disk_cache[key] = results
        disk_cache.close()