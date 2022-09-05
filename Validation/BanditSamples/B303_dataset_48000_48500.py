def tagged_item_key(self, key):
        """
        Get a fully qualified key for a tagged item.

        :param key: The cache key
        :type key: str

        :rtype: str
        """
        return '%s:%s' % (hashlib.sha1(encode(self._tags.get_namespace())).hexdigest(), key)