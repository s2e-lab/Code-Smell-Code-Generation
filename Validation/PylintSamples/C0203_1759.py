from collections import defaultdict


def setter(prep,k,v,supSetter):
    if callable(v): 
        def wrap(*args):
            f = prep.d[k][len(args)]
            if isinstance(f,int): raise AttributeError()
            return f(*args)
        prep.d[k][v.__code__.co_argcount] = v
        v = wrap
    supSetter(k,v)
        
        
class Prep(dict):
    def __init__(self):         self.d = defaultdict(lambda: defaultdict(int))
    def __setitem__(self,k,v):  setter(self, k, v, super().__setitem__)


class Meta(type):
    @classmethod
    def __prepare__(cls,*args, **kwds): return Prep()
    
    def __new__(metacls, name, bases, prep, **kwargs):
        prep['_Meta__DCT'] = prep
        return super().__new__(metacls, name, bases, prep, **kwargs)
    
    def __setattr__(self,k,v): setter(self.__DCT, k, v, super().__setattr__)
from collections import defaultdict
def setter(prep, k, v, sup):
    if callable(v):
        def wrap(*args):
            a = prep.d[k][len(args)]
            if isinstance(a, int): raise AttributeError()
            return a(*args)
        prep.d[k][v.__code__.co_argcount] = v
        v = wrap
    sup(k, v)
    
class Prep(dict):
    def __init__(self): self.d = defaultdict(lambda: defaultdict(int))
    def __setitem__(self,k,v): setter(self, k, v, super().__setitem__)

class Meta(type):
    @classmethod
    def __prepare__(cls, *args, **kwargs): return Prep()
    
    def __new__(metacls, name, bases, prep, **kwargs):
        prep['_Meta__DCT'] = prep
        return super().__new__(metacls, name, bases, prep, **kwargs)
    
    def __setattr__(self, k, v): setter(self.__DCT, k, v, super().__setattr__)
