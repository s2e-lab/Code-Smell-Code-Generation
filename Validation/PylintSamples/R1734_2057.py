
import sys
#sys.stdin=open("data.txt")
input=sys.stdin.readline

# so the ending sequence is b...ba...a

# find length of ending sequence

extra=0
need=0
for ch in input().strip():
    if ch=='a':
        need=(need*2+1)%1000000007
    else:
        extra=(extra+need)%1000000007

print(extra)
MOD = 1000000007

def main():
    s = input()
    n = len(s)

    # each b contributes 1 flip to the first a before it, 2 flips to the second a before it, etc
    # in general, if there are k 'a's before a b, then add 2^(k + 1) - 1 flips
    ans = 0
    a_ct = 0
    p = 1
    for c in s:
        if c == 'a':
            a_ct += 1
            p *= 2
            p %= MOD
        else:
            ans += p
            ans %= MOD

    ans -= (n - a_ct)
    if ans < 0:
        ans += MOD
    ans %= MOD
    print(ans)


main()

str = input()

bset = []

bno = 0

for c in str:
	if c == 'b':
		bno += 1
	else:
		bset += [bno]
		bno = 0
bset += [bno]

ans = 0
acc = 0

for n in reversed(bset[1:]):
	ans += acc + n
	acc *= 2
	acc += 2*n
	acc %= 1000000007
	ans %= 1000000007

print(ans)


s = input( )
mod = 1000000007
step = 0
p = 1

for x in s:
    if x == 'a':
        p = (p * 2) % mod;
    else:
        step = (step + p - 1) % mod;

print(step)
a = input()
res = 0
cur = 0
for i in reversed(a):
    if (i == 'b'):
        cur+=1
    else:
        res+=cur
        res%=1000000007
        cur= cur*2%1000000007
print(res)
S = input()

num = 1
r = 0
for s in S:
    if s == 'a':
        num = (num * 2) % 1000000007
    else:
        r = (r + num -1) % 1000000007

print(r)
S = input()[::-1]

num = 0
r = 0
for s in S:
    if s == 'a':
        r = (r + num) % 1000000007
        num = (num * 2) % 1000000007
    else:
        num = num + 1
print(r)
S = input()[::-1]
S1 = []

num = 0
for s in S:
    if s == 'b':
        num += 1
    else:
        S1.append(num)

S1 = S1[::-1]
r = 0
num = 1
for s in S1:
    r = (r + s * num) % 1000000007
    num = (num * 2) % 1000000007

print(r)

import sys

s = sys.stdin.read()

n = 1000000007

aCount = 0
bCount = 0

count = 0

increment = 0

for c in s:
    if c == 'a':
        increment = (2 * increment + 1) % n

    if c == 'b':
        count = (count + increment) % n

print(count)

s = input()
cnt = 0
m=10**9 + 7
t = 0

for i in range(len(s)):
	if s[~i] == 'a':
		cnt = (cnt+t)%m
		t = (t*2)%m
	else:
		t += 1
print(cnt)

"""s = raw_input()
m = 0
n = 0
twop = 1
ans = 0
mold = [0,0]
isbool1 = True
isbool2 = False

def twopower(x):
    d = {0:1}
    if x in d:
        return d[x]
    else:
        if x%2 == 0:
            d[x] = twopower(x/2)**2
            return d[x]
        else:
            d[x] = twopower(x-1)*2
            return d[x]

for char in s:
    if char == "a":
        m += 1
    else:
        ans += twopower(m)-1
        ans = ans%(10**9+7)

print ans%(10**9+7)"""
        
"""
for char in s:
    if char == "a":
        m += 1
        if isbool == True:
            twop *= twopower(m-mold[1])
            ans += n*(twop-1)
            isbool = False
            n = 0
    else:
        mold = [mold[1],m]
        n += 1
        isbool = True

if s[-1] == "a":
    print ans
else:
    twop *= twopower(m-mold[0])
    ans += n*(twop-1)
    print ans
"""

s=input()
l=list()
cur=0
for i in range(len(s)):
    if s[i]=='a':
        cur+=1
    else:
        l.append(cur)

mo=10**9+7

cur=0
exp=1
res=0
for i in range(len(l)):
    while l[i]>cur:
        cur+=1
        exp=exp*2%mo
    res=(res+exp-1)%mo
    
    
    
print(res)
s=input()
l=list()
cur=0
for i in range(len(s)):
    if s[i]=='a':
        cur+=1
    else:
        l.append(cur)
mo=10**9+7
cur=0
exp=1
res=0
for i in range(len(l)):
    while l[i]>cur:
        cur+=1
        exp=exp*2%mo
    res=(res+exp-1)%mo
print(res)
s=input()
l=list()
cur=0
for i in range(len(s)):
    if s[i]=='a':
        cur+=1
    else:
        l.append(cur)
mo=10**9+7
cur=0
exp=1
res=0
for i in range(len(l)):
    while l[i]>cur:
        cur+=1
        exp=exp*2%mo
    res=(res+exp-1)%mo
print(res)
N = 10**9+7
s = input().strip()
n = 0
ret = 0
c2n = 2**n-1
for c in s:
  if c == 'a':
    n += 1
    c2n = (2*c2n+1)%N
  else:
    ret += c2n
    ret %= N
print(ret)

b_cnt, res = 0, 0
mod = 10 ** 9 + 7
for c in input()[::-1]:
	if c == 'b':
		b_cnt += 1
	else:
		res += b_cnt
		b_cnt *= 2
		res %= mod
		b_cnt %= mod
print(res)
s = input()[::-1]
a, b = 0, 0
mod = 10 ** 9 + 7
for i in s:
    if i == 'b':
        b += 1
    else:
        a += b
        a %= mod
        b <<= 1
        b %= mod
print(a)

M, a, b=10**9+7, 0, 0
for c in reversed(input()):
	if c=='b':
		b+=1
	else:
		a+=b
		b=(b<<1)%M
print(a%M)

s = input()
cnt = 0
m=10**9 + 7
t = 0

for i in range(len(s)):
	if s[~i] == 'a':
		cnt = (cnt+t)%m
		t = (t*2)%m
	else:
		t += 1
print(cnt)
s = list(input())

p=[1]
for i in range(1000000):
	p.append((p[i]*2)%1000000007)

ctr=0
ans=0
l = len(s)
for i in range(l):
	if s[i]=='a':
		ctr+=1
		continue
	ans=(p[ctr]-1+ans)%1000000007

print(ans)
s=input()
mod = 10 ** 9+7
b=0
res=0
for c in s[::-1]:
    if c=='b':
        b+=1
    else:
        res+=b
        b*=2
        res%=mod
        b%=mod
print(res)
# Description of the problem can be found at http://codeforces.com/problemset/problem/804/B

MOD = 1e9 + 7
s = input()

ans = 0
b = 0
for c in s[::-1]:
    if c == 'a':
        ans = (ans + b) % MOD
        b = (2 * b) % MOD
    else:
        b = (b + 1) % MOD

print(int(ans))
3

def main():
    s = input()

    tpw = [1 for i in range(len(s) + 1)]
    for i in range(1, len(s) + 1):
        tpw[i] = (2 * tpw[i - 1]) % (10 ** 9 + 7)
    
    na = 0
    ans = 0
    for c in s:
        if c == 'a':
            na += 1
        else:
            ans = (ans + tpw[na] - 1) % (10 ** 9 + 7)
    print(ans)

main()

