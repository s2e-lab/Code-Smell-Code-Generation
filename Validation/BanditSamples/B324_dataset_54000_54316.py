def _hash(data):
    """ generate a hash from data object to be used as cache key """
    hash_algo = hashlib.new('md5')
    hash_algo.update(pickle.dumps(data))
    # prefix allows possibility of multiple applications
    # sharing same keyspace
    return 'esi_' + hash_algo.hexdigest()