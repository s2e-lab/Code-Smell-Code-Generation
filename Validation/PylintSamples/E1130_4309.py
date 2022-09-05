from itertools import groupby

def replace(s):
    queue, rle = {}, [[i, k, len(list(g))] for i, (k, g) in enumerate(groupby(s))]
    for i, k, l in reversed(rle):
        if l not in queue: queue[l] = {}
        queue[l].setdefault(k, []).append(i)
    for l in queue:
        while sum(map(bool, queue[l].values())) > 1:
            for c in queue[l]: rle[queue[l][c].pop()][1] = ' '
    return ''.join(k * l for i, k, l in rle)
import re

def replace(s):
    dic = {'!': '?', '?': '!'}
    r = re.findall(r'[!]+|[/?]+', s)
    for i in r[:]:
        ii =dic[i[0]] * len(i)
        if  ii in r:
            r[r.index(ii)] = ' ' * len(i)    
    return ''.join(r)
import itertools as it
from collections import Counter, defaultdict
from functools import reduce

def tokenize(s):
    for key, group in it.groupby(s):
        yield key, len(list(group))

def gather(tokens, expected_keys = None):
    stats = defaultdict(Counter)
    tokens = it.chain(tokens, ((key,0) for key in expected_keys or []))
    for key, length in tokens:
        stats[key][length] +=1
    return stats

def intersect(*counters, ignored_keys=None):
    mins = reduce(lambda a, b: a & b, counters)
    for key in ignored_keys:
        mins.pop(key, None)
    return +mins
    
def substitute(tokens, counters, replacement_key):
    for key, length in tokens:
        if counters[key][length]:
            counters[key][length] -= 1
            yield replacement_key, length
        else:
            yield key, length 
def detokenize(tokens):
    return ''.join(key * length for key, length in tokens)
    
def replace(s):
    tokens = list(tokenize(s))
    stats = gather(tokenize(s), expected_keys='!?')
    mins = intersect(*list(stats.values()), ignored_keys = [0])
    replaced_counts = {key: mins.copy() for key in stats}
    replaced_tokens = substitute(tokens, replaced_counts, ' ')
    return detokenize(replaced_tokens)
    

from itertools import groupby

def replace(stg):
    g = ["".join(s) for _, s in groupby(stg)]
    l = len(g)
    for i in range(l):
        for j in range(i+1, l, 2):
            if " " not in f"{g[i]}{g[j]}" and len(g[i]) == len(g[j]):
                g[i] = g[j] = " " * len(g[i])
    return "".join(g)
from itertools import groupby
from collections import defaultdict

def replace(s):
    res, D = [], {'!':defaultdict(list), '?':defaultdict(list)}
    for i, (k, l) in enumerate(groupby(s)):
        s = len(list(l))
        D[k][s].append(i)
        res.append([k, s])
    for v, L1 in D['!'].items():
        L2 = D['?'][v]
        while L1 and L2:
            res[L1.pop(0)][0] = ' '
            res[L2.pop(0)][0] = ' '
    return ''.join(c*v for c,v in res)
from itertools import groupby
from collections import defaultdict

tbl = str.maketrans('!?', '?!')

def replace(s):
    xs = [''.join(grp) for _, grp in groupby(s)]
    stacks = defaultdict(list)
    result = []
    for i, x in enumerate(xs):
        stack = stacks[x]
        if stack:
            result[stack.pop(0)] = x = ' ' * len(x)
        else:
            stacks[x.translate(tbl)].append(i)
        result.append(x)
    return ''.join(result)
from collections import defaultdict, deque
import re

def replace(s):
    chunks = re.findall(r'!+|?+', s)
    cnts   = defaultdict(deque)
    for i,c in enumerate(chunks[:]):
        other = '!?'[c[0]=='!'] * len(c)
        if cnts[other]:
            blank = ' '*len(c)
            chunks[i] = chunks[cnts[other].popleft()] = blank
        else:
            cnts[c].append(i)
    
    return ''.join(chunks)
from itertools import groupby
def replace(s):
    g = ["".join(j) for i, j in groupby(s)]
    for i, j in enumerate(g):
        for k, l in enumerate(g[i+1:], start=i + 1):
            if len(j) == len(l) and j[0] != l[0] and ' ' not in j+l:
                g[i] = g[k] = " " * len(j) 
                break
    return "".join(g)
from collections import Counter
from itertools import chain, groupby, repeat, starmap

C2D = {'!': 1, '?': -1}

def replace(s):
    gs = [(c, sum(1 for _ in g)) for c, g in groupby(s)]
    ds = Counter()
    for c, k in gs:
        ds[k] += C2D[c]
    for i in reversed(list(range(len(gs)))):
        c, k = gs[i]
        if ds[k] * C2D[c] > 0:
            ds[k] -= C2D[c]
        else:
            gs[i] = ' ', k
    return ''.join(chain.from_iterable(starmap(repeat, gs)))

import itertools as it
from collections import Counter, defaultdict
from functools import reduce

def tokenize(string):
    groups = it.groupby(string)
    for key, group in groups:
        yield key, len(list(group))
        
def gather(tokens, expected_keys=None):
    stats = defaultdict(Counter)
    tokens = it.chain(tokens, ((key, 0) for key in expected_keys or []))
    for key, length in tokens:
        stats[key][length] += 1
    return stats
    
def intersect(*counters, ignored_keys=None):
    mins = reduce(lambda a, b: a & b, counters)
    for key in ignored_keys or []:
        mins.pop(key, None)
    return +mins
    
def remove(tokens, counters, replacement_key):
    for key, length in tokens:
        if counters[key][length]:
            counters[key][length] -= 1
            yield replacement_key, length
        else:
            yield key, length
            
def detokenize(tokens):
    return ''.join(key * length for key, length in tokens)

def replace(s):
    tokens = list(tokenize(s))
    stats = gather(tokens, expected_keys='!?')
    mins = intersect(*list(stats.values()), ignored_keys=[0])
    replace_counts = {key: mins.copy()for key in stats}
    replaced_tokens = remove(tokens, replace_counts, ' ')
    return detokenize(replaced_tokens)
    

