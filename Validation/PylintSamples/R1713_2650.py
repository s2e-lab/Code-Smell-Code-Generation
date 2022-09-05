n,l = map(int,input().split())
a = []
for i in range(n):
  a.append(input())
  
a.sort()
print("".join(str(i) for i in a))
n,l = [int(x) for x in input().split()]
s = []
for i in range(n):
  s.append(input())
s.sort()

print("".join(s))
n, l = map(int, input().split( ))
t = list()
for i in range(n):
    s = input()
    t.append(s)
t.sort()
ans = ""
for j in range(n):
    ans += t[j]
print(ans)
n, l = list(map(int, input().split()))
s = []
for i in range(n):
    s.append(input())
s.sort()
print((''.join(s)))

N,L=map(int,input().split())
S=[]
for i in range(N):
  S.append(input())
S.sort()
print("".join(S))
#-*-coding:utf-8-*-
import sys
input=sys.stdin.readline

def main():
    n,l = map(int,input().split())
    words=[]
    words=[input().rstrip() for _ in range(n)]
    s_words=sorted(words,key=lambda x:x[0:])
    ans=""

    for s_word in s_words:
        ans+=s_word
    print(ans)
def __starting_point():
    main()
__starting_point()
def main():
    N, L = list(map(int, input().split()))
    Sn = [input() for i in range(N)]
    Sn2 = sorted(Sn)
    ans = ""
    for val in Sn2 :
        ans = ans + val
    print(ans)

def __starting_point():
    main()

__starting_point()
N, L = list(map(int, input().split()))
S = []
for i in range(N):
  S.append(input())

print((''.join(sorted(S))))

n,l = map(int,input().split())
s = [input() for _ in range(n)]
s.sort()
print("".join(s))
import sys
def readint():
    return int(sys.stdin.readline())

def readints():
    return tuple(map(int,sys.stdin.readline().split()))

def readintslist(n):
    return [tuple(map(int,sys.stdin.readline().split())) for _ in range(n)]

def main():
    n,l = readints()

    s = [input() for _ in range(n)]

    s = sorted(s)
    print(*s,sep='')

def __starting_point():
    main()
__starting_point()
_, _, *S = open(0).read().split()
print("".join(sorted(S)))
N, L = map(int,input().split())
l = list()
for i in range(N):
	l.append(input())
print(*sorted(l),sep='')
N, L = input().split()
N, L = int(N), int(L)
S = []
count = 0

while count < N:
	str = input()
	S.append(str)
	count += 1

S_ord = sorted(S)
print(''.join(S_ord))
a,b=map(int,input().split())
s=[]
for i in range(a):
  s.append(input())
s.sort()
for i in range(a):
  print(s[i],end="")
n,l = map(int,input().split())
s = sorted([input() for i in range(n)])
print(*s,sep="")

n, l = map(int, input().split())
s = [input() for _ in range(n)]
s = sorted(s)

for i in s:
    print(i,end = '')
N, L = map(int, input().split())
S = [input() for i in range(N)]
sort = sorted(S)

print(''.join(sort))
nl = list(map(int,input().split()))
N = nl[0]
L = nl[1]
ss = [input() for i in range(N)]
ss.sort()
res = ""
for i in ss:
  res = res+i
print(res)
N, L = map(int, input().split())

S = sorted([input() for i in range(N)])

print(*S, sep="")



N,L=list(map(int,input().split()))
S=[]
for i in range(N):
    S.append(input())

print((''.join(sorted(S))))

nl = input().split()

N = int(nl[0])
L = int(nl[1])

lst = []

for i in range(N):
   lst.append(input())

lst.sort()

ans = ''

for s in lst:
   ans += s

print(ans)
n,l=map(int,input().split())
s=[]
for _ in range(n):
    s.append(input())
s.sort()
ss="".join(s)
print(ss)
N,L = map(int, input().split())
S = [input() for i in range(N)]
S.sort()
print("".join(S))
main = list([int(x) for x in input().split()])
n, l = main[0], main[1]
final = []
for i in range(n):
    string = input()
    final.append(string)

final.sort()
print((''.join(final)))

n,l = map(int,input().split())
s = [input() for _ in range(n)]
an = ''
for _ in range(n):
    an += min(s)
    s.remove(min(s))
print(an)
n,l = map(int, input().split())
print("".join(sorted([input() for _ in range(n)])))
#from statistics import median
#import collections
#aa = collections.Counter(a) # list to list || .most_common(2)u3067u6700u5927u306e2u500bu3068u308au3060u305bu308bu304a a[0][0]
from math import gcd
from itertools import combinations,permutations,accumulate, product # (string,3) 3u56de
#from collections import deque
from collections import deque,defaultdict,Counter
import decimal
import re
import math
import bisect
import heapq
#
#
#
# pythonu3067u7121u7406u306au3068u304du306fu3001pypyu3067u3084u308bu3068u6b63u89e3u3059u308bu304bu3082uff01uff01
#
#
# my_round_int = lambda x:np.round((x*2 + 1)//2)
# u56dbu6368u4e94u5165g
#
# u30a4u30f3u30c7u30c3u30afu30b9u7cfb
# int min_y = max(0, i - 2), max_y = min(h - 1, i + 2);
# int min_x = max(0, j - 2), max_x = min(w - 1, j + 2);
#
#
import sys
sys.setrecursionlimit(10000000)
mod = 10**9 + 7
#mod = 9982443453
#mod = 998244353
INF = float('inf')
from sys import stdin
readline = stdin.readline
def readInts():
  return list(map(int,readline().split()))
def readTuples():
    return tuple(map(int,readline().split()))
def I():
    return int(readline())
n,l = readInts()
S = sorted([input() for _ in range(n)])
print((''.join(S)))

n, l = map(int,input().split())
a = []
for i in range(n):
    a.append(input())
    
a.sort()
print(''.join(a))
N, L = map(int,input().split())

word = []
count = 0
while N > count:
    S = input()
    word.append(S)
    count += 1

word = sorted(word)
ans = ''.join(word)

print(ans)
n,l=map(int,input().split())
s=list(input() for i in range(n))
s.sort()
a=""
for i in range(n):
    a=a+s[i]
print(a)
n, l = list(map(int, input().split()))

strList = []
for _ in range(0, n):
    strList.append(input())

strList.sort()

ans = ""
for i in range(0, n):
    ans += strList[i]

print(ans)

n, l = map(int, input().split())
S = []

for i in range(n):
  s = input()
  S.append(s)
  
S.sort()

ans = ''
for i in range(n):
  ans += S[i]
  
print(ans)
N, L = map(int, input().split())
S = []
for n in range(N):
    s = input()
    S.append(s)
S.sort()
print(''.join(S))
N, L = map(int, input().split())
S = [""] * N
for i in range(N):
    S[i] = input()

S.sort()
for s in S:
    print(s, end = "")
n,l = map(int,input().split())
strings = []
for _ in range(n):
  s = input()
  strings.append(s)
strings.sort()
ans = ''
for string in strings:
  ans += string
print(ans)
N, L = map(int, input().split())
array = [str(input()) for i in range(N)]
array = sorted(array)
ans = ''
for j in array:
  ans = ans + j
print(ans)
def main():
    N, L = list(map(int, input().split()))
    S = []
    for i in range(N):
        s = input()
        S.append(s)
    S.sort()
    ans = ''.join(S)
    print(ans)

def __starting_point():
    main()

__starting_point()
N, L = list(map(int, input().split()))
S = [input() for _ in range(N)]
S.sort()
print(("".join(S)))


n, l = map(int, input().split())
s = []
for i in range(n):
    s.append(input())
s.sort()
print(''.join(s))
N, L = map(int, input().split())

str_list = []

for w in range(N):
  str_list.append(input())

output = "".join(sorted(str_list))
print(output)
#from statistics import median
#import collections
#aa = collections.Counter(a) # list to list || .most_common(2)u3067u6700u5927u306e2u500bu3068u308au3060u305bu308bu304a a[0][0]
from math import gcd
from itertools import combinations,permutations,accumulate, product # (string,3) 3u56de
#from collections import deque
from collections import deque,defaultdict,Counter
import decimal
import re
import math
import bisect
import heapq
#
#
#
# pythonu3067u7121u7406u306au3068u304du306fu3001pypyu3067u3084u308bu3068u6b63u89e3u3059u308bu304bu3082uff01uff01
#
#
# my_round_int = lambda x:np.round((x*2 + 1)//2)
# u56dbu6368u4e94u5165g
#
# u30a4u30f3u30c7u30c3u30afu30b9u7cfb
# int min_y = max(0, i - 2), max_y = min(h - 1, i + 2);
# int min_x = max(0, j - 2), max_x = min(w - 1, j + 2);
#
#
import sys
sys.setrecursionlimit(10000000)
mod = 10**9 + 7
#mod = 9982443453
#mod = 998244353
INF = float('inf')
from sys import stdin
readline = stdin.readline
def readInts():
  return list(map(int,readline().split()))
def readTuples():
    return tuple(map(int,readline().split()))
def I():
    return int(readline())
n,l = readInts()
lis = [input() for _ in range(n)]
lis = sorted(lis)
print((''.join(lis)))

N, L = map(int, input().split())
S = [input() for _ in range(N)]
S.sort()
print(''.join(S))
n, l = map(int,input().split())
words = [input() for i in range(n)]
words = list(sorted(words))
s = ""
for i in range(n):
    s += words[i]
print(s)
N, L = map(int, input().split())
S = [input() for i in range(N)]

S_sorted = sorted(S)
Word = ''.join(S_sorted)

print(Word)
N,L,*Ss =open(0).read().split()

ans = ""
for s in sorted(Ss):
  ans+=s
print(ans)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

a = input().split()
c = int(a[0])
strlist = []
d = ''
for i in range(c):
    b = input().split()
    strlist.append(b[0])

strlist.sort()
for i in range(c):
    d += strlist[i] 
print(d)
n,l = map(int,input().split())
arr = []
for i in range(n):
	arr.append(input())
arr.sort()
print(''.join(arr))
from sys import stdin, stdout
from time import perf_counter

import sys
sys.setrecursionlimit(10**9)
mod = 10**9+7

# import sys
# sys.stdout = open("e:/cp/output.txt","w")
# sys.stdin = open("e:/cp/input.txt","r")

n,l = map(int,input().split())
a = [input() for item in range(n)]
b = sorted(a)
print(''.join(b))
n,l=map(int,input().split())
s=sorted([input() for i in range(n)])
ans=""
for i in range(n):
  ans+=s[i]
print(ans)
n,l=map(int, input().split())

s_list=[input() for i in range(n)]

s_list.sort()

print("".join(s_list))
N,L = map(int,input().split())
S = [input() for _ in range(N)]

S.sort()
print("".join(S))
#Nu3068Lu3092u5165u529b
N, L = map(int, input().split())


# u7a7au306eu30eau30b9u30c8u3092u751fu6210
S = []

#u5165u529b
for i in  range(N):
    array = input()
    S.append(array)

#u8f9eu66f8u9806u306bu4e26u3079u66ffu3048
S.sort()

#pythonu306eu30eau30b9u30c8u306eu51fau529bu3067u306f[u3068,u304cu51fau529bu3055u308cu308bu306eu3067u53d6u308b
S=''.join(S)

#u51fau529b
print(S)
N, L = map(int, input().split())
S = sorted([input() for i in range(N)])
print(*S, sep="")

N, L = map(int, input().split())
text_list = [input() for _ in range(N)]

sorted_list = sorted(text_list)

print(''.join(sorted_list))
N, L = map(int, input().split())
S = [input() for _ in range(N)]
S.sort()
for s in S:
  print(s, end="")
N, L = map(int, input().split()) 
S = [input() for _ in range(N)] 
S.sort()
print(''.join(map(str, S)))
N,L = map(int,input().split())
S = []
for i in range(N):
  S.append(str(input()))
S.sort()
print(''.join(S))
n, l = list(map(int, input().split()))
s = []
for i in range(n):
    s.append(input())

s.sort()
res = ""
for i in range(n):
    res += s[i]
print(res)

N, L = map(int, input().split())
print(''.join(sorted([input() for _ in range(N)])))
from collections import Counter
import math
import statistics
import itertools
a,b=list(map(int,input().split()))
# b=input()
# c=[]
# for i in a:
#     c.append(int(i))
# A,B,C= map(int,input().split())
# f = list(map(int,input().split()))
#g = [list(map(lambda x: '{}'.format(x), list(input()))) for _ in range(a)]
# h = []
# for i in range(a):
#     h.append(list(map(int,input().split())))
# a = [[0] for _ in range(H)]#nizigen
lis=[input() for i in range(a)]
ra=a-1
for i in range(ra):
    for k in range(ra-i):
        if lis[k]>lis[k+1]:
            a=lis[k]
            lis[k]=lis[k+1]
            lis[k+1]=a
print(("".join(lis)))


n,l = map(int,input().split())
s = [input() for _ in range(n)]
ans = sorted(s)
print(''.join(ans))
import collections
import sys

m = collections.defaultdict(int)
line = input()
tokens = line.split()
n = int(tokens[0])
strings = []
for i in range(n):
    s = input()
    strings.append(s)

print(''.join(sorted(strings)))
N, L = map(int,input().split())
mat = []
for i in range(N):
  mat.append(input())
while i <= N:
  FINISH = 0
  for i in range(N-1):
    if mat[i] > mat[i+1]:
      tmp = mat[i]
      mat[i] = mat[i+1]
      mat[i+1] = tmp
      FINISH += 1
  if FINISH == 0:
    break
for i in range(N):
  print(mat[i], end = "")
n,l=map(int, input().split())
s=[input() for i in range(n)]
print("".join(sorted(s)))
# Nu3001Lu306eu5165u529bu53d7u4ed8
N, L = list(map(int, input().split()))
# Nu56deu9023u7d9au3067u5165u529bu6587u5b57u5217u3092u53d7u4ed8
S = []
for i in range(N):
    S.append(input())
# Su5185u3092u30bdu30fcu30c8
S.sort()
# Su306eu8981u7d20u3092u7d50u5408
result = ""
for i in range(N):
    result = result + S[i]
# resultu3092u51fau529b
print(result)

n, l = map(int, input().split())
str = [input() for i in range(n)]
str.sort()
for i in str:
    print(i, end = '')
# coding:UTF-8
import sys


def resultSur97(x):
    return x % (10 ** 9 + 7)


def __starting_point():
    # ------ u5165u529b ------#
    nl = list(map(int, input().split()))     # u30b9u30dau30fcu30b9u533au5207u308au9023u7d9au6570u5b57

    x = nl[0]
    sList = [input() for _ in range(x)]

    # ------ u51e6u7406 ------#
    sListSorted = sorted(sList)
    out = ""
    for s in sListSorted:
        out += s

    # ------ u51fau529b ------#
    print(("{}".format(out)))
    # if flg == 0:
    #     print("YES")
    # else:
    #     print("NO")

__starting_point()
n = list(map(int, input().split()))[0]
ss = list(sorted([input() for _ in range(n)]))
print(("".join(ss)))

n, l = list(map(int, input().split()))
A = [input() for _ in range(n)]
A.sort()
print(("".join(A)))

n, l = map(int,input().split())
lw=[]
for i in range(n):
    lw.append(input())
lw.sort()
print("".join(lw))
N, L = map(int, input().split())
ans = [input() for i in range(N)]
ans.sort()
ans = ''.join(ans)
print(ans)
n,l = map(int, input().split())
print("".join(sorted([input() for _ in range(n)])))
#!/usr/bin/env python3

def main():
    n, l = list(map(int, input().split()))
    s = sorted([input() for i in range(n)])
    print(("".join(s)))


def __starting_point():
    main()

__starting_point()
N, L = map(int, input().split())
ans = [input() for i in range(N)]
ans.sort()
ans = ''.join(ans)
print(ans)
n,l=map(int,input().split())
li=[input() for i in range(n)]

sortli=sorted(li)
print(''.join(sortli))
n, l = map(int, input().split())
sl = sorted(list(input() for _ in range(n)))

print(*sl, sep='')
n, l = map(int, input().split())
sl = list(input() for _ in range(n))
sl_s = sorted(sl)

print(*sl_s, sep='')
n,l = map(int, input().split())
string_list =[input() for i in range(n)]
string_list.sort()
ans = string_list[0]
for i in range(1,n):
    ans += string_list[i]
print(ans)
n, l = map(int, input().split())
s = [input() for _ in range(n)]
s.sort()
print("".join(s))
# -*- coding: utf-8 -*-

n,l = map(int,input().split())
s = [input() for i in range(n)]
s = sorted(s)
print(*s,sep="")


import itertools
n,l = map(int,input().split())
li = [list(input().split()) for i in range(n)]
#print(li)
li.sort()
#print(li)
lis = list(itertools.chain.from_iterable(li))
print(''.join(lis))
n, l = map(int, input().split())
s = [input() for _ in range(n)]
s = sorted(s)
print("".join(s))

N,L = map(int,input().split())

l = []
for _ in range(N):
    s = input()
    l.append(s)

l.sort()

for i in l:
    print(i,end='')

import math
from datetime import date

def main():
		
	line = input().split()
	n = int(line[0])
	k = int(line[1])


	a = []
	for i in range(n):
		s = input()
		a.append(s)

	ans = "";
	for x in sorted(a):
		ans += x

	print(ans)
	
main()

n, l = map(int, input().split())
s = [input() for _ in range(n)]
s = sorted(s)
print(''.join(s))
n,l=map(int,input().split())
a=[]
for i in range(n):
    s=input()
    a.append(s)
a.sort()
print(''.join(a))
#!/usr/bin/env python3

N, L = list(map(int ,input().split()))
S = []
for i in range(N):
    S.append(input())
S.sort()
out = ''
for i in range(N):
    out = out + S[i]

print(out)

n, l = map(int, input().split())
lst = []
for i in range(n):
  s = input()
  lst.append(s)
lst = sorted(lst)
print(''.join(lst))
N, L = map(int, input().split())
a = []

for i in range(N):
    a.append(input())

a.sort()

for i in range(N):
    print(a[i],end="")
n,l = map(int,input().split())
ans = ''
k = []
for i in range(n):
    s = input()
    k.append(s)
k.sort()
for i in k:
    ans +=i
print(ans)
N,L = map(int, input().split())
word_list = [input() for i in range(N)]

word_list.sort()

for i in word_list:
    print(i, end="")

print()
n, l = map(int, input().split())
s_l = [ str(input()) for _ in range(n)  ]
s_l = sorted(s_l)
print(''.join(s_l))
N, L = map(int, input().split())
w = []
for _ in range(N):
  w.append(input())

w.sort()

print("".join(w))
n,l = map(int,input().split())
s = []
for i in range(n):
  s.append(input())
s.sort()
print("".join(s))
N, L = map(int, input().split())
s = []
for i in range(N):
  s.append(input())
print(''.join(sorted(s)))
n,l= [int(x) for x in input().split()]
la = []
for i in range(n):
  la.append(str(input()))
la.sort()
print("".join(str(x) for x in la))
N,L=map(int, input().split())
S=[]

for i in range(N):
    S.append(input())
S.sort()

for i in range(N):
    print(S[i], end="")
n, l = list(map(int, input().split()))

s = []
for i in range(n):
    s.append(input())


print(("".join(sorted(s))))

from typing import List


def answer(n: int, l: int, s: List[str]) -> str:
    return ''.join(sorted(s))


def main():
    n, l = map(int, input().split())
    s = [input() for _ in range(n)]
    print(answer(n, l, s))


def __starting_point():
    main()
__starting_point()
# -*- coding: utf-8 -*-

def __starting_point():
    N, L = list(map(int, input().split()))
    si = []
    for i in range(N):
        si.append(str(input()))
    si.sort()
    print((''.join(si)))

__starting_point()
