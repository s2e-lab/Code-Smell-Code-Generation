def _dhash(self, params):
        """Generate hash of the dictionary object."""
        m = hashlib.new('md5')
        m.update(self.hash.encode('utf-8'))
        for key in sorted(params.keys()):
            h_string = ('%s-%s' % (key, params[key])).encode('utf-8')
            m.update(h_string)
        return m.hexdigest()