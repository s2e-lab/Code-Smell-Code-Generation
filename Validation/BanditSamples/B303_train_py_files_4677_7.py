from hashlib import md5
from itertools import product
def crack(hash):
    digits = [str(x) for x in range(10)]
    return next(s for s in map(lambda tup:"".join(tup),product(digits,repeat=5)) if md5(bytes(s,"utf-8")).hexdigest()==hash)
