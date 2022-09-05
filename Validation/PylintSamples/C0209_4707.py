def to_bytes(n):
    if not n:
        return ['00000000']
    
    res = []
    while n:
        res.append('{:08b}'.format(n % 256))
        n //= 256
    
    return res[::-1]
import re
from math import ceil
def to_bytes(n):
    return re.findall('.{8}', '{:0{}b}'.format(n, int(8 * ceil(n.bit_length() / 8.0)))) or [8 * '0']
to_bytes=b=lambda n,f=1:n and b(n>>8,0)+[format(n&255,'08b')]or['0'*8]*f
def to_bytes(n):
    b = bin(n)[2:]
    b = '0'*(8-len(b)%8)*(len(b)%8!=0) + b
    return [b[8*i:8*i+8] for i in range(len(b)//8)]
def to_bytes(n):
    L = 8
    s = bin(n)[2:]
    s = s.rjust(len(s) + L - 1 - (len(s) - 1) % 8, '0')
    return [s[i:i+L] for i in range(0, len(s), L)]
def to_bytes(n):
    if n==0: return ['00000000']
    s=bin(n)[2:]
    s=s.rjust(len(s)+7-(len(s)-1)%8,'0')
    return [s[i:i+8] for i in range(0, len(s), 8)]
def to_bytes(n):
    n = bin(n)[2::]
    while len(n) % 8 != 0:
        n = '0' + n
    return [n[i: i+8] for i in range(0, len(n), 8)]
to_bytes = lambda n, d=__import__("itertools").dropwhile: list(d(("0"*8).__eq__, map("".join, zip(*[iter(bin(n)[2:].zfill(8 * (((len(bin(n)) - 2) // 8) + 1)))]*8)))) or ["0"*8]
def to_bytes(n):
    return [''.join(t) for t in zip(*[iter(format(n,'0{}b'.format((max(1,n.bit_length())+7)//8*8)))]*8)]
