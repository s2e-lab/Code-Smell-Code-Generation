import re
class WordDictionary:
    def __init__(self):
        self.data=[]
  
    def add_word(self,x):
        self.data.append(x)
  
    def search(self,x):
        for word in self.data:
            if re.match(x+"Z",word): return True
        return False
class WordDictionary:
  def __init__(self): self.dct = set()
  def add_word(self, word): self.dct.add(word)
  def word_match(self, w, s):
    if len(w) != len(s): return False
    try:
      for i,c in enumerate(s):
        if c != '.' and c != w[i]: return False
      return True
    except: return False
  def search(self, s):
    for w in self.dct:
      if self.word_match(w, s): return True
    return False
import re

class WordDictionary:

    def __init__(self):
        self.d = {} 
  
    def add_word(self, w):
        self.d[w] = True
  
    def search(self, k):
        
        if not '.' in k:
            return self.d.get(k, False)
        
        r = re.compile('^{}$'.format(k))
        return any(r.match(x) for x in self.d.keys())
from re import match

class WordDictionary:
    def __init__(self):
        self.db = []
  
    def add_word(self, word):
        self.db.append(word)
  
    def search(self, word):
        return bool([w for w in self.db if match(rf'^{word}$', w)])
import re
class WordDictionary:
    def __init__(self):
       self.words = []
    
    def add_word(self,v):
       self.words.append(v)
  
    def search(self,rg):
        return any(re.match(r'^{}$'.format(rg),i) for i in self.words)
from collections import defaultdict
from re import compile, match

class WordDictionary(defaultdict):
    def __init__(self):
        super(WordDictionary, self).__init__(list)

    def add_word(self, s):
        self[len(s)].append(s)
  
    def search(self, s):
        p = compile(f"^{s}$")
        return any(match(p, w) for w in self[len(s)])
from itertools import zip_longest
class WordDictionary:
    def __init__(self):
        self.d = set()

    def add_word(self, s):
        return self.d.add(s)

    def search(self, s):
        for w in self.d:
            if all((a==b or b == '.') and a for a,b in zip_longest(w,s)):
                return 1
        return 0
import re


class WordDictionary:

    def __init__(self):
        self.words = set()

    def add_word(self, word):
        self.words.add(word)

    def search(self, pattern):
        pattern += '$'
        return any(re.match(pattern, word) for word in self.words)
from itertools import zip_longest

class WordDictionary:

  def __init__(self):
    self.words = []

  def add_word(self, word):
    self.words.append(word)

  def search(self, pattern):
    return any(all((a and b == '.') or a == b for a, b in zip_longest(word, pattern)) for word in self.words)
