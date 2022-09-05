def get_md5(string):
    """Get a string's MD5"""
    try:
        hasher = hashlib.md5()
    except BaseException:
        hasher = hashlib.new('md5', usedForSecurity=False)
    hasher.update(string)
    return hasher.hexdigest()