def find_upper_bound(arr,key):
 low,high = 0,len(arr)-1
 while low<=high:
  mid = (low+high)//2 
  if arr[mid]==key:return mid
  elif arr[mid]>key and mid-1>=0 and arr[mid-1]<key:return mid 
  elif arr[mid]>key:high = mid - 1 
  else:low = mid + 1 
 return mid 
def get_query(l):
 nonlocal prefix_storer,bin_storer
 ind = find_upper_bound(bin_storer,l)
 surplus = (abs(bin_storer[ind]-l)*ind*ind)%limit 
 return (prefix_storer[ind]-surplus+limit)%limit
def fire_query(l,r):
 return (get_query(r)-get_query(l-1)+limit)%limit
golomb,dp,prefix_storer,bin_storer = [],[0,1],[0,1],[0,1]
limit = 10**9+7
for i in range(2,10**6+100):
 dp.append(1 + dp[i-dp[dp[i-1]]])
 bin_storer.append(dp[-1]+bin_storer[-1])
 prefix_storer.append(((prefix_storer[-1] + (dp[-1]*i*i)%limit))%limit)
# print(dp[1:20])
# print(bin_storer[1:20])
# print(prefix_storer[1:20])
# print(get_query(2),get_query(4))
for _ in range(int(input())):
 l,r = map(int,input().split())
 print(fire_query(l,r))
def find_upper_bound(arr,key):
 low,high = 0,len(arr)-1
 while low<=high:
  mid = (low+high)//2 
  if arr[mid]==key:return mid
  elif arr[mid]>key and mid-1>=0 and arr[mid-1]<key:return mid 
  elif arr[mid]>key:high = mid - 1 
  else:low = mid + 1 
 return mid 
def get_query(l):
 nonlocal prefix_storer,bin_storer
 ind = find_upper_bound(bin_storer,l)
 surplus = (abs(bin_storer[ind]-l)*ind*ind)%limit 
 return (prefix_storer[ind]-surplus+limit)%limit
def fire_query(l,r):
 return (get_query(r)-get_query(l-1))%limit
golomb,dp,prefix_storer,bin_storer = [],[0,1],[0,1],[0,1]
limit = 10**9+7
for i in range(2,10**6+100):
 dp.append(1 + dp[i-dp[dp[i-1]]])
 bin_storer.append(dp[-1]+bin_storer[-1])
 prefix_storer.append(((prefix_storer[-1] + (dp[-1]*i*i)%limit))%limit)
# print(dp[1:20])
# print(bin_storer[1:20])
# print(prefix_storer[1:20])
# print(get_query(2),get_query(4))
for _ in range(int(input())):
 l,r = map(int,input().split())
 print(fire_query(l,r))
# cook your dish here
import bisect
MAXR = 100
MAXN = 20000000
T = int(input().strip())
queries = []
for t in range(T):
 L, R = map(int, input().strip().split())
 queries.append((L,R))
 MAXR = max(MAXR, R+1)
MOD = 10**9+7
g = [0,1,2]
p = [0,1,3]
s = [0,1,9]
for n in range(3, MAXN):
 gg = 1 + g[n-g[g[n-1]]]
 pp = p[n-1] + gg
 ss = (s[n-1] + gg*n*n) % MOD
 g.append(gg)
 p.append(pp)
 s.append(ss)
 if pp > MAXR:
  break
def process(m):
 n = bisect.bisect_right(p, m)
 return (s[n-1] + (m-p[n-1])*n*n) % MOD
for L, R in queries:
 print((process(R) - process(L-1))%MOD)
import bisect
M=1000000007
G=[0,1,2,2]
currPos=4
for i in range(3,2000000):
 if(currPos>2000000):
  break
 j=0 
 while(j<G[i] and currPos<2000000):
  G.append(i) 
  currPos+=1 
  j+=1 
prefixSum1=[0]
prefixSum2=[0]
for i in range(1,2000000):
 prefixSum1.append(prefixSum1[i-1]+G[i])
for i in range(1,2000000):
 prefixSum2.append((prefixSum2[i-1] + i*i%M*G[i]%M)%M)
 #print(prefixSum1)
 #print(prefixSum2)

def solve(n):
 nthterm=bisect.bisect_left(prefixSum1, n, lo=0, hi=len(prefixSum1))-0
 ans=0 
 if(nthterm>0):
  ans=prefixSum2[nthterm-1]
 ans = (ans + nthterm*nthterm%M * (n-prefixSum1[nthterm-1])%M)%M
 return ans
for tc in range(int(input())):
 l,r=map(int,input().split())
 print((solve(r)-solve(l-1)+M)%M)
import sys
import bisect as b
input=sys.stdin.readline
li=[]
mod=10**9 + 7
mxr=100
mxn=20000000

for _ in range(int(input())):
 l,r=map(int,input().split())
 li.append([l,r])
 mxr=max(mxr,r+1)

g=[0,1,2]
p=[0,1,3]
s=[0,1,9]

for i in range(3,mxn):
 gg=1+g[i-g[g[i-1]]]
 pp=(p[i-1]+gg)%mod
 ss=(s[i-1]+gg*i*i)%mod
 g.append(gg)
 p.append(pp)
 s.append(ss)
 if pp>mxr:
  break
def solve(m):
 n=b.bisect_right(p,m)
 return (s[n-1] + ((m-p[n-1])*n*n))%mod
for l,r in li:
 ans=(solve(r)-solve(l-1))%mod
 print(ans)
# cook your dish here
import bisect as bi

temp_g = [0, 1, 2]
temp_s = [0, 1, 9]
temp_p = [0, 1, 3]
max_mod = (10 ** 9 + 7)

def find_gn(n):
 return (1 + temp_g[n - temp_g[temp_g[n - 1]]])

def find_sn(n, gn):
 return (temp_s[n - 1] + gn * n**2) % max_mod

def find_pn(n, gn):
 return (temp_p[n - 1] + gn)
 
def bisection(x):
 n = bi.bisect_right(temp_p, x)
 return (temp_s[n - 1] + (x - temp_p[n - 1]) * n**2 ) % max_mod
 
lr, max_ = [], 100

for _ in range(int(input())):
 l, r = list(map(int, input().split()))
 lr.append((l, r))
 max_ = max(max_, r + 1)
 
for n in range(3, 2 * (10 ** 8), 1):
 gn = find_gn(n)
 sn, pn = find_sn(n, gn), find_pn(n, gn)
 temp_g.append(gn)
 temp_p.append(pn)
 temp_s.append(sn)
 if (pn > max_):
  break
 
for pair in lr:
 print(bisection(pair[1]) - bisection(pair[0] - 1) % max_mod)



# cook your dish here
import bisect as bi

temp_g = [0, 1, 2]
temp_s = [0, 1, 9]
temp_p = [0, 1, 3]
max_mod = (10 ** 9 + 7)

def find_gn(n):
 return (1 + temp_g[n - temp_g[temp_g[n - 1]]])

def find_sn(n, gn):
 return (temp_s[n - 1] + gn * n * n) % max_mod

def find_pn(n, gn):
 return (temp_p[n - 1] + gn)
 
def bisection(x):
 n = bi.bisect_right(temp_p, x)
 return (temp_s[n - 1] + (x - temp_p[n - 1]) * n * n ) % max_mod
 
lr, max_ = [], 100

for _ in range(int(input())):
 l, r = list(map(int, input().split()))
 lr.append((l, r))
 max_ = max(max_, r + 1)
 
for n in range(3, 2 * (10 ** 8), 1):
 gn = find_gn(n)
 sn, pn = find_sn(n, gn), find_pn(n, gn)
 temp_g.append(gn)
 temp_p.append(pn)
 temp_s.append(sn)
 if (pn > max_):
  break
 
for pair in lr:
 print(bisection(pair[1]) - bisection(pair[0] - 1) % max_mod)



# cook your dish here
import bisect as bi

temp_g = [0, 1, 2]
temp_s = [0, 1, 9]
temp_p = [0, 1, 3]
max_mod = (10 ** 9 + 7)

def find_gn(n):
 return (1 + temp_g[n - temp_g[temp_g[n - 1]]])

def find_sn(n, gn):
 return (temp_s[n - 1] + gn * n * n) % max_mod

def find_pn(n, gn):
 return (temp_p[n - 1] + gn)
 
def bisection(x):
 n = bi.bisect_right(temp_p, x)
 return (temp_s[n - 1] + (x - temp_p[n - 1]) * n * n ) % max_mod
 
lr, max_ = [], 100

for _ in range(int(input())):
 l, r = list(map(int, input().split()))
 lr.append((l, r))
 max_ = max(max_, r + 1)
 
for n in range(3, 2 * (10 ** 8), 1):
 gn = find_gn(n)
 sn = find_sn(n, gn)
 pn = find_pn(n, gn)
 temp_g.append(gn)
 temp_p.append(pn)
 temp_s.append(sn)
 if (pn > max_):
  break
 
for pair in lr:
 print(bisection(pair[1]) - bisection(pair[0] - 1) % max_mod)



# cook your dish here
import bisect

MAXR = 100
MAXN = 20000000

T = int(input().strip())
queries = []
for t in range(T):
 L, R = list(map(int, input().strip().split()))
 queries.append((L,R))
 MAXR = max(MAXR, R+1)
 
 
MOD = 10**9+7
g = [0,1,2]
p = [0,1,3]
s = [0,1,9]

for n in range(3, MAXN):
 gg = 1 + g[n-g[g[n-1]]]
 pp = p[n-1] + gg
 ss = (s[n-1] + gg*n*n) % MOD
 g.append(gg)
 p.append(pp)
 s.append(ss)
 if pp > MAXR:
  break
 
def process(m):
 n = bisect.bisect_right(p, m)
 return (s[n-1] + (m-p[n-1])*n*n) % MOD
 
 
for L, R in queries:
 print((process(R) - process(L-1))%MOD)

'''
Name : Jaymeet Mehta
codechef id :mj_13
Problem : 
'''
from sys import stdin,stdout
import math
from bisect import bisect_left
mod=1000000007
def mul(a,b):
 nonlocal mod
 return ((a%mod)*(b%mod))%mod
def add(a,b):
 nonlocal mod
 return ((a%mod)+(b%mod))%mod
g=[0,1]
pre=[0,1]
ans=[0,1]
i=2
while(True):
 g.append(1+g[i-g[g[i-1]]])
 pre.append(pre[i-1]+g[i])
 ans.append(add(ans[i-1],mul(mul(i,i),g[i])))
 if pre[i]>10000000000:
  break
 i+=1
test=int(stdin.readline())
for _ in range(test):
 l,r= list(map(int,stdin.readline().split()))
 sm1,sm2=0,0
 
 if l==1:
  sm1=0
 else:
  l-=1
  tl=bisect_left(pre,l)
  sm1=add(ans[tl-1],mul(l-pre[tl-1],mul(tl,tl)))
 tr=bisect_left(pre,r)
 sm2=add(ans[tr-1],mul(r-pre[tr-1],mul(tr,tr)))
 print((sm2-sm1)%mod)

import bisect
import sys
input = sys.stdin.readline

MAXR = 100
MAXN = 20000000

t = int(input().strip())
queries = []
for t in range(t):
 L, R = map(int, input().strip().split())
 queries.append((L,R))
 MAXR = max(MAXR, R + 1)
 
 
MOD = 10 ** 9 + 7
g = [0, 1, 2]
p = [0, 1, 3]
s = [0, 1, 9]

for n in range(3, MAXN):
 gn = 1 + g[n - g[g[n - 1]]]
 pn = p[n - 1] + gn
 sn = (s[n - 1] + gn * n * n) % MOD
 g.append(gn)
 p.append(pn)
 s.append(sn)
 if pn > MAXR:
  break
 
def process(m):
 n = bisect.bisect_right(p, m)
 return (s[n - 1] + (m -p[n - 1]) * n * n) % MOD
 
 
for L, R in queries:
 print(process(R) - process(L - 1) % MOD)
