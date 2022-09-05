import sys
input = sys.stdin.readline

n=int(input())
A=list(map(int,input().split()))

BIT=[0]*(n+1)

def update(v,w):
    while v<=n:
        BIT[v]+=w
        v+=(v&(-v))

def getvalue(v):
    ANS=0
    while v!=0:
        ANS+=BIT[v]
        v-=(v&(-v))
    return ANS

for i in range(1,n+1):
    update(i,i)

ANS=[-1]*n

for i in range(n-1,-1,-1):
    MIN=0
    MAX=n
    k=A[i]

    while True:
        x=(MIN+MAX+1)//2


        if getvalue(x)>k:
            if getvalue(x-1)==k:
                ANS[i]=x
                break
            else:
                MAX=x
        else:
            MIN=x

    update(x,-x)

            
print(*ANS)

class DualBIT():
    def __init__(self, n):
        self.n = n
        self.bit = [0] * (n + 1)

    def get(self, i):
        '''iu756au76eeu306eu8981u7d20u3092u53d6u5f97'''
        i = i + 1
        s = 0
        while i <= self.n:
            s += self.bit[i]
            i += i & -i
        return s

    def _add(self, i, x):
        while i > 0:
            self.bit[i] += x
            i -= i & -i

    def add(self, i, j, x):
        '''[i, j)u306eu8981u7d20u306bxu3092u52a0u7b97u3059u308b'''
        self._add(j, x)
        self._add(i, -x)


n = int(input())
a = list(map(int, input().split()))
                        
bit = DualBIT(n+3)
for i in range(1, n+1):
    bit.add(i+1, n+1, i)

li = []
flag = False
while True:
    if not a:
        break
    ok = n + 1
    ng = 0
    num = a[-1]
    if num == 0 and not flag:
        flag = True
        bit.add(1, n + 2, -1)
        li.append(1)
        del a[-1]
        continue
    while abs(ok - ng) > 1:
        mid = (ok + ng) // 2
        if bit.get(mid) > num:
            ok = mid
        else:
            ng = mid
    tmp = ok - 1
    bit.add(ok, n + 2, -tmp)
    li.append(tmp)
    del a[-1]
print(*li[::-1])
          


NN = 18
BIT=[0]*(2**NN+1)

def addbit(i, x):
    while i <= 2**NN:
        BIT[i] += x
        i += i & (-i)

def getsum(i):
    ret = 0
    while i != 0:
        ret += BIT[i]
        i -= i&(-i)
    return ret

def searchbit(x):
    l, sl = 0, 0
    d = 2**(NN-1)
    while d:
        m = l + d
        sm = sl + BIT[m]
        if sm <= x:
            l, sl = m, sm
        d //= 2
    return l
    
N = int(input())
A = [int(a) for a in input().split()]
for i in range(1, N+1):
    addbit(i, i)

ANS = []
for s in A[::-1]:
    a = searchbit(s) + 1
    addbit(a, -a)
    ANS.append(a)
    
print(*ANS[::-1])
from sys import setrecursionlimit as SRL, stdin

SRL(10 ** 7)
rd = stdin.readline
rrd = lambda: map(int, rd().strip().split())

n = int(rd())

bit = [0] * 200005


def add(x, val):
    while x <= n:
        bit[x] += val
        x += (x & -x)


def query(x):
    num = 0
    for i in range(30, -1, -1):
        if num+(1 << i) <= n and bit[num + (1 << i)] <= x:
           
            x -= bit[num + (1 << i)]
            num += (1 << i)

    return num + 1


for i in range(1, n + 1):
    add(i, i)

s = list(rrd())

ans = []

for i in range(len(s) - 1, -1, -1):
    q = query(s[i])
    ans.append(q)
    add(q, -q)

ans = ans[::-1]
print(*ans)
from sys import setrecursionlimit as SRL, stdin

SRL(10 ** 7)
rd = stdin.readline
rrd = lambda: map(int, rd().strip().split())

n = int(rd())

bit = [0] * 200005


def add(x, val):
    while x <= n:
        bit[x] += val
        x += (x & -x)


def query(x):
    num = 0
    for i in range(30, -1, -1):
        if num+(1 << i) <= n and bit[num + (1 << i)] <= x:

            x -= bit[num + (1 << i)]
            num += (1 << i)

    return num + 1


for i in range(1, n + 1):
    add(i, i)

s = list(rrd())

ans = []

for i in range(len(s) - 1, -1, -1):
    q = query(s[i])
    ans.append(q)
    add(q, -q)

ans = ans[::-1]
print(*ans)
import sys
input = sys.stdin.readline

nn = 18
bit=[0]*(2**nn+1)
 
def addbit(i, x):
    while i <= 2**nn:
        bit[i] += x
        i += i & (-i)
 
def getsum(i):
    ret = 0
    while i != 0:
        ret += bit[i]
        i -= i&(-i)
    return ret
 
def searchbit(x):
    l, sl = 0, 0
    d = 2**(nn-1)
    while d:
        m = l + d
        sm = sl + bit[m]
        if sm <= x:
            l, sl = m, sm
        d //= 2
    return l + 1

n = int(input())
l = list(map(int, input().split()))
for i in range(1, n + 1):
	addbit(i, i)
ans = [0 for _ in range(n)]
for i in range(n - 1, -1, -1):
	a = searchbit(l[i])
	addbit(a, -a)
	ans[i] = a
print(*ans)






n=int(input())
P=[int(i) for i in input().split()]
import math
import bisect

n_max=2*10**5
nn=int(math.log2(n_max))+1
BIT=[0]*(2**nn+1)

def bitsum(BIT,i):
  s=0
  while i>0:
    s+=BIT[i]
    i-=i&(-i)
  return s
def bitadd(BIT,i,x):
  if i<=0:
    return True
  else:
    while i<=2**nn:
      BIT[i]+=x
      i+=i&(-i)
    return BIT
def bitlowerbound(BIT,s):
  if s<=0:
    return 0
  else:
    ret=0
    k=2**nn
    while k>0:
      if k+ret<=2**nn and BIT[k+ret]<s:
        s-=BIT[k+ret]
        ret+=k
      k//=2
    return ret+1
for i in range(n_max):
  bitadd(BIT,i+1,i+1)

Ans=[]
for i in reversed(range(n)):
  p=P[i]
  ans=bitlowerbound(BIT,p+1)
  Ans.append(ans)
  bitadd(BIT,ans,-ans)
Ans=Ans[::-1]
print(*Ans)
# TAIWAN NUMBER ONE!!!!!!!!!!!!!!!!!!!
# TAIWAN NUMBER ONE!!!!!!!!!!!!!!!!!!!
# TAIWAN NUMBER ONE!!!!!!!!!!!!!!!!!!!
from sys import stdin, stdout
from collections import defaultdict
from collections import deque
import math
import copy
 
#T = int(input())
N = int(input())
#s1 = input()
#s2 = input()
#N,Q = [int(x) for x in stdin.readline().split()]
arr = [int(x) for x in stdin.readline().split()]
 
bit = [0]*N

series = [x for x in range(N)]

def lowbit(x):
    return x&(-x)

def update(idx,delta):
    while idx<N:
        bit[idx] += delta
        idx += lowbit(idx)

def query(x):
    s = 0
    while x>0:
        s += bit[x]
        x -= lowbit(x)
    return s
    
# init
for i in range(N):
    bit[i] += series[i]
    y = i + lowbit(i)
    if y<N:
        series[y] += series[i]
        
visited = [0]*N
ans = [0]*N
for i in range(N-1,-1,-1):
    # find
    left = 0
    right = N-1
    target = arr[i]
    while left<=right:
        mid = (left+right)//2
        q = query(mid)
        if q<target:
            left = mid + 1
        elif q>target:
            right = mid - 1
        else:
            if visited[mid]==1:
                left = mid + 1
            else:
                visited[mid] = 1
                ans[i] = mid + 1
                break
    # update
    if mid+1<N:
        update(mid+1,-mid-1)
    
    
print(*ans)

# TAIWAN NUMBER ONE!!!!!!!!!!!!!!!!!!!
# TAIWAN NUMBER ONE!!!!!!!!!!!!!!!!!!!
# TAIWAN NUMBER ONE!!!!!!!!!!!!!!!!!!!
from sys import stdin, stdout
from collections import defaultdict
from collections import deque
import math
import copy
 
#T = int(input())
N = int(input())
#s1 = input()
#s2 = input()
#N,Q = [int(x) for x in stdin.readline().split()]
arr = [int(x) for x in stdin.readline().split()]
 
bit = [0]*(N+1)

series = [0] + [x for x in range(N)]

def lowbit(x):
    return x&(-x)

def update(idx,delta):
    while idx<=N:
        bit[idx] += delta
        idx += lowbit(idx)

def query(x):
    s = 0
    while x>0:
        s += bit[x]
        x -= lowbit(x)
    return s
    
# init
for i in range(1,N+1):
    bit[i] += series[i]
    y = i + lowbit(i)
    if y<=N:
        series[y] += series[i]
        
visited = [0]*(N+1)
ans = [0]*N
for i in range(N-1,-1,-1):
    # find
    left = 1
    right = N
    target = arr[i]
    while left<=right:
        mid = (left+right)//2
        q = query(mid)
        #print(mid,q)
        if q<target:
            left = mid + 1
        elif q>target:
            right = mid - 1
        else:
            if visited[mid]==1:
                left = mid + 1
            else:
                visited[mid] = 1
                ans[i] = mid 
                break
    # update
    update(mid+1,-mid)
    
    
print(*ans)

# TAIWAN NUMBER ONE!!!!!!!!!!!!!!!!!!!
# TAIWAN NUMBER ONE!!!!!!!!!!!!!!!!!!!
# TAIWAN NUMBER ONE!!!!!!!!!!!!!!!!!!!
from sys import stdin, stdout
from collections import defaultdict
from collections import deque
import math
import copy
 
#T = int(input())
N = int(input())
#s1 = input()
#s2 = input()
#N,Q = [int(x) for x in stdin.readline().split()]
arr = [int(x) for x in stdin.readline().split()]
 
bit = [0]*(N+1)

series = [0] + [x for x in range(N)]

def lowbit(x):
    return x&(-x)

def update(idx,delta):
    while idx<=N:
        bit[idx] += delta
        idx += lowbit(idx)

def query(x):
    s = 0
    while x>0:
        s += bit[x]
        x -= lowbit(x)
    return s
    
# init
for i in range(1,N+1):
    bit[i] += series[i]
    y = i + lowbit(i)
    if y<=N:
        series[y] += series[i]
        
visited = [0]*(N+1)
ans = [0]*N

for i in range(N-1,-1,-1):
    # find
    left = 1
    right = N
    target = arr[i]
    
    
    while True:
        L = right - left + 1
        num = left - 1 + 2**int(math.log(L,2))
        
        q = bit[num]
        #print(num,q,target,left,right)
        if q<target:
            target -= q
            left = num + 1
        elif q>target:
            right = num - 1
        else:
            if visited[num]==1:
                target -= q
                left = num + 1
            else:
                visited[num] = 1
                ans[i] = num
                break
            
    # update
    update(num+1,-num)
    
    
print(*ans)

from operator import add

class Stree:
    def __init__(self, f, n, default, init_data):
        self.ln = 2**(n-1).bit_length()
        self.data = [default] * (self.ln * 2)
        self.f = f
        for i, d in init_data.items():
            self.data[self.ln + i] = d
        for j in range(self.ln - 1, 0, -1):
            self.data[j] = f(self.data[j*2], self.data[j*2+1])

    def update(self, i, a):
        p = self.ln + i
        self.data[p] = a
        while p > 1:
            p = p // 2
            self.data[p] = self.f(self.data[p*2], self.data[p*2+1])

    def get(self, i, j):
        def _get(l, r, p):
            if i <= l and j >= r:
                return self.data[p]
            else:
                m = (l+r)//2
                if j <= m:
                    return _get(l, m, p*2)
                elif i >= m:
                    return _get(m, r, p*2+1)
                else:
                    return self.f(_get(l, m, p*2), _get(m, r, p*2+1))
        return _get(0, self.ln, 1)

    def find_value(self, v):
        def _find_value(l, r, p, v):
            if r == l+1:
                return l
            elif self.data[p*2] <= v:
                return _find_value((l+r)//2, r, p*2+1, v - self.data[p*2])
            else:
                return _find_value(l, (l+r)//2, p*2, v)
        return _find_value(0, self.ln, 1, v)


def main():
    n = int(input())
    sums = {i:i for i in range(n+1)}
    stree = Stree(add, n+1, 0, sums)
    ss = list(map(int, input().split()))
    ss.reverse()
    pp = []
    for s in ss:
        sval = stree.find_value(s)
        pp.append(sval)
        stree.update(sval,0)
    print(*(reversed(pp)))

def __starting_point():
    main()
__starting_point()
def update(x,val):
    while x<=n:
        BIT[x]+=val
        x+=(x&-x)
def query(x):
    s=0
    while x>0:
        s=(s+BIT[x])
        x-=(x&-x)
    return s
n=int(input())
BIT=[0]*(n+1)
for i in range(1,n+1):
    update(i,i)
arr=list(map(int,input().split()))
answers=[0]*(n)
#print(BIT)
for i in range(n-1,-1,-1):
    lol=arr[i]
    low=0
    fjf=0
    high=n
   # print(lol)
    while True:
        
        mid=(high+low+1)//2
        j=query(mid)
      #  print(mid,j)
      #  print(answers)
       # break
        if j>lol:
            if query(mid-1)==lol:
                answers[i]=mid
                update(mid,-mid)
                break
            else:
                high=mid
        else:
            low=mid
    
print(*answers)
        

# 1208D
class segTree():
    def __init__(self, n):
        self.t = [0] * (n << 2)

    def update(self, node, l, r, index, value):
        if l == r:
            self.t[node] = value
            return
        mid = (l + r) >> 1
        if index <= mid:
            self.update(node*2, l, mid, index, value)
        else:
            self.update(node*2 + 1, mid + 1, r, index, value)
        self.t[node] = self.t[node*2] + self.t[node*2 + 1]

    def query(self, node, l, r, value):
        if l == r:
            return self.t[node]
        mid = (l + r) >> 1
        if self.t[node*2] >= value:
            return self.query(node*2, l, mid, value)
        return self.query(node*2 + 1, mid + 1, r, value - self.t[node*2])

def do():
    n = int(input())
    nums = [int(i) for i in input().split(" ")]
    res = [0]*n
    weightTree = segTree(n)
    for i in range(1, n+1):
        weightTree.update(1, 1, n, i, i)
    # print(weightTree.t)
    for i in range(n-1, -1, -1):
        res[i] = weightTree.query(1, 1, n, nums[i] + 1)
        weightTree.update(1, 1, n, res[i], 0)
    return " ".join([str(c) for c in res])
print(do())

class FTree:

    def __init__(self, f):

        self.n = len(f)

        self.ft = [0] * (self.n + 1)



        for i in range(1, self.n + 1):

            self.ft[i] += f[i - 1]

            if i + self.lsone(i) <= self.n:

                self.ft[i + self.lsone(i)] += self.ft[i]



    def lsone(self, s):

        return s & (-s)



    def query(self, i, j):

        if i > 1:

            return self.query(1, j) - self.query(1, i - 1)



        s = 0

        while j > 0:

            s += self.ft[j]

            j -= self.lsone(j)



        return s



    def update(self, i, v):

        while i <= self.n:

            self.ft[i] += v

            i += self.lsone(i)



    def select(self, k):

        lo = 1

        hi = self.n



        for i in range(19): ########  30

            mid = (lo + hi) // 2

            if self.query(1, mid) < k:

                lo = mid

            else:

                hi = mid



        return hi


n = int(input())
data = [int(i) for i in input().split()]
ft = FTree(list(range(1, n+1)))
ans = [""]*n

for i in range(n-1, -1, -1):
    val = data[i]
    ind = ft.select(val+1)
    ans[i] = str(ind)
    ft.update(ind, -ind)

print(" ".join(ans))
class FTree:
    def __init__(self, f):
        self.n = len(f)
        self.ft = [0] * (self.n + 1)
        for i in range(1, self.n + 1):
            self.ft[i] += f[i - 1]
            if i + self.lsone(i) <= self.n:
                self.ft[i + self.lsone(i)] += self.ft[i]
    def lsone(self, s):
        return s & (-s)
    def query(self, i, j):
        if i > 1:
            return self.query(1, j) - self.query(1, i - 1)
        s = 0
        while j > 0:
            s += self.ft[j]
            j -= self.lsone(j)
        return s
    def update(self, i, v):
        while i <= self.n:
            self.ft[i] += v
            i += self.lsone(i)
    def select(self, k):
        lo = 1
        hi = self.n
        for i in range(19): ########  30
            mid = (lo + hi) // 2
            if self.query(1, mid) < k:
                lo = mid
            else:
                hi = mid
        return hi
n = int(input())
data = [int(i) for i in input().split()]
ft = FTree(list(range(1, n+1)))
ans = [""]*n
for i in range(n-1, -1, -1):
    val = data[i]
    ind = ft.select(val+1)
    ans[i] = str(ind)
    ft.update(ind, -ind)
print(" ".join(ans))


def sumsegtree(l,seg,st,en,x):
    if st==en:
        seg[x]=l[st]
    else:
        mid=(st+en)>>1
        sumsegtree(l,seg,st,mid,2*x)
        sumsegtree(l,seg,mid+1,en,2*x+1)
        seg[x]=seg[2*x]+seg[2*x+1]

def query(seg,st,en,val,x):
    if st==en:
        return seg[x]
    mid=(st+en)>>1
    if seg[2*x]>=val:
        return query(seg,st,mid,val,2*x)
    return query(seg,mid+1,en,val-seg[2*x],2*x+1)

def upd(seg,st,en,ind,val,x):
    if st==en:
        seg[x]=val
        return
    mid=(st+en)>>1
    if mid>=ind:
        upd(seg,st,mid,ind,val,2*x)
    else:
        upd(seg,mid+1,en,ind,val,2*x+1)
    seg[x]=seg[2*x]+seg[2*x+1]

n=int(input())
l=list(map(int,range(1,n+1)))
s=[0]*n
p=list(map(int,input().split()))


seg=["#"]*(n<<2)
sumsegtree(l,seg,0,len(l)-1,1)

for i in range(n-1,-1,-1):
    s[i]=query(seg,1,n,p[i]+1,1)
    upd(seg,1,n,s[i],0,1)

print (*s)
def sumsegtree(l,seg,st,en,x):
    if st==en:
        seg[x]=l[st]
    else:
        mid=(st+en)>>1
        sumsegtree(l,seg,st,mid,2*x)
        sumsegtree(l,seg,mid+1,en,2*x+1)
        seg[x]=seg[2*x]+seg[2*x+1]
 
def query(seg,st,en,val,x):
    if st==en:
        return seg[x]
    mid=(st+en)>>1
    if seg[2*x]>=val:
        return query(seg,st,mid,val,2*x)
    return query(seg,mid+1,en,val-seg[2*x],2*x+1)
 
def upd(seg,st,en,ind,val,x):
    if st==en:
        seg[x]=val
        return
    mid=(st+en)>>1
    if mid>=ind:
        upd(seg,st,mid,ind,val,2*x)
    else:
        upd(seg,mid+1,en,ind,val,2*x+1)
    seg[x]=seg[2*x]+seg[2*x+1]
 
n=int(input())
l=list(map(int,range(1,n+1)))
s=[0]*n
p=list(map(int,input().split()))
 
 
seg=["#"]*(n<<2)
sumsegtree(l,seg,0,len(l)-1,1)
 
for i in range(len(p)-1,-1,-1):
    s[i]=query(seg,1,n,p[i]+1,1)
    upd(seg,1,n,s[i],0,1)
 
print (*s)
_ = input()
x = [int(i) for i in input().split()]

res = []

from math import log


class SegmentTree(object):

    def __init__(self, nums):
        self.arr = nums
        self.l = len(nums)
        self.tree = [0] * self.l + nums
        for i in range(self.l - 1, 0, -1):
            self.tree[i] = self.tree[i << 1] + self.tree[i << 1 | 1]

    def update(self, i, val):
        n = self.l + i
        self.tree[n] = val
        while n > 1:
            self.tree[n >> 1] = self.tree[n] + self.tree[n ^ 1]
            n >>= 1

    def query(self, i, j):
        m = self.l + i
        n = self.l + j
        res = 0
        while m <= n:
            if m & 1:
                res += self.tree[m]
                m += 1
            m >>= 1
            if n & 1 == 0:
                res += self.tree[n]
                n -= 1
            n >>= 1
        return res

tree = SegmentTree(list(range(1, len(x) + 1)))
org = len(x)
while x:
    q = x.pop()

    lo = 0
    hi = org - 1

    while lo < hi:
        mid = (lo + hi) // 2
        # print(lo, hi, mid)
        sm = tree.query(0, mid)
        # print(sm, mid)
        if sm > q:
            hi = mid
        else:
            lo = mid + 1
    # print(tree.arr, lo, hi)
    idx = tree.arr[lo]
    # print(idx)
    tree.update(lo, 0)
    res.append(idx)

print(' '.join(str(i) for i in res[::-1]))


def sum_number(n,j):
    j[0]=0
    j[1]=0
    for i in range(2,n+1):
        j[i]=j[i-1]+(i-1)
    return(j)
po=int(input())
l=[0]*(po+1)
l1=[int(i) for i in input().split()]
def getsum(BITTree,i):
    s = 0
    while i > 0:
        s += BITTree[i]
        i -= i & (-i) 
    return(s) 
def updatebit(BITTree , n , i ,v):
    #print('n',n)
    while i <= n:
        #print('i',i)
        BITTree[i] += v
        i += i & (-i)
    #print(BITTree)
for i in range(1,po+1):
    updatebit(l,po,i,i)
output=[0]*po
for i in range(po-1,-1,-1):
    min_=0
    max_=po
    k=l1[i]
    while True:
        x=(min_+max_+1)//2
        if getsum(l,x)>k:
            if getsum(l,x-1)==k:
                output[i]=x
                break
            else:
                #print(x)
                max_=x
        else :
            #print(x)
            min_=x
    updatebit(l,po,x,-x)
print(*output)



        
    
    




# https://codeforces.com/contest/1208/problem/D

from sys import stdin, stdout
input = stdin.readline
print = stdout.write

# For every i from N to 1, let's say the value of the si is x. 
# So it means there are k smallest unused numbers whose sum is x.
# We simply put the (k+1)st number in the output permutation at this i, and continue to move left. 

# segment tree and binary search

_ = input()
x = [int(i) for i in input().split()]
 
res = []
 
from math import log
 
 
class SegmentTree(object):
 
    def __init__(self, nums):
        self.arr = nums
        self.l = len(nums)
        self.tree = [0] * self.l + nums
        for i in range(self.l - 1, 0, -1):
            self.tree[i] = self.tree[i << 1] + self.tree[i << 1 | 1]
 
    def update(self, i, val):
        n = self.l + i
        self.tree[n] = val
        while n > 1:
            self.tree[n >> 1] = self.tree[n] + self.tree[n ^ 1]
            n >>= 1
 
    def query(self, i, j):
        m = self.l + i
        n = self.l + j
        res = 0
        while m <= n:
            if m & 1:
                res += self.tree[m]
                m += 1
            m >>= 1
            if n & 1 == 0:
                res += self.tree[n]
                n -= 1
            n >>= 1
        return res
 
tree = SegmentTree(list(range(1, len(x) + 1)))
org = len(x)
while x:
    # from back to forth
    q = x.pop()
 
    lo = 0
    hi = org - 1
 
    while lo < hi:
        mid = (lo + hi) // 2
        # print(lo, hi, mid)
        sm = tree.query(0, mid)
        # print(sm, mid)
        if sm > q:
            hi = mid
        else:
            lo = mid + 1
    # print(tree.arr, lo, hi)
    idx = tree.arr[lo]
    # print(idx)
    tree.update(lo, 0)
    # also from back to forth
    res.append(idx)
 
print(' '.join(str(i) for i in res[::-1]))
# https://codeforces.com/contest/1208/problem/D

from sys import stdin, stdout
input = stdin.readline
# print = stdout.write

# si is the sum of elements before the i-th element that are smaller than the i-th element.

# For every i from N to 1, let's say the value of the si is x.
# So it means there are k smallest unused numbers whose sum is x.
# We simply put the k+1st number in the output permutation at this i, and continue to move left.

# BIT and binary lifting
# https://codeforces.com/contest/1208/submission/59526098


class BIT:
    def __init__(self, nums):
        # we store the sum information in bit 1.
        # so the indices should be 1 based.
        # here we assume nums[0] = 0
        self.nums = [0] * (len(nums))
        for i, x in enumerate(nums):
            if i == 0:
                continue
            self.update(i, x)

    def low_bit(self, x):
        return x & (-x)

    def update(self, i, diff):
        while i < len(self.nums):
            self.nums[i] += diff
            i += self.low_bit(i)

    def prefix_sum(self, i):
        ret = 0
        while i != 0:
            ret += self.nums[i]
            i -= self.low_bit(i)
        return ret

    def search(self, x):
        # find the index i such that prefix_sum(i) == x
        cur_index, cur_sum = 0, 0
        delta = len(self.nums) - 1
        while delta - self.low_bit(delta):
            delta -= self.low_bit(delta)

        while delta:            
            m = cur_index + delta
            if m < len(self.nums):
                sm = cur_sum + self.nums[m]
                if sm <= x:
                    cur_index, cur_sum = m, sm
            delta //= 2
        return cur_index + 1


n = int(input())
bit = BIT(list(range(n+1)))

ans = [0 for _ in range(n)]
nums = list(map(int, input().split()))
for i in range(n - 1, -1, -1):
    index = bit.search(nums[i])
    bit.update(index, -index)
    ans[i] = index
print(*ans)

import sys
input = sys.stdin.readline
class SegTree(object):
	"""docstring for SegTree"""
	def __init__(self, n, arr):
		self.n = n
		self.arr = arr
		self.tree = [0 for i in range(2*n)]

	def construct(self): # Construction
		for i in range(self.n):
			self.tree[n+i] = self.arr[i]
		for i in range(n-1,0,-1):
			self.tree[i] = self.function(self.tree[2*i],self.tree[2*i+1])

	def update(self,index,value):
		start = index+self.n
		self.tree[start] = value
		while start>0:
			start = start//2
			self.tree[start] = self.function(self.tree[2*start],self.tree[2*start+1])

	def calc(self,low,high): # 0-indexed
		low+=self.n
		high+=self.n
		ans = 0 # Needs to initialised
		while low<high:
			if low%2:
				ans = self.function(ans, self.tree[low])
				low += 1
			if high%2:
				high -= 1
				ans = self.function(ans, self.tree[high])
			low = low//2
			high = high//2
		return ans
	
	def function(self,a,b): # Function used to construct Segment Tree
		return a + b


def find(num):
	low = 0
	high = n-1
	while low<high:
		mid = (low+high)//2
		if st.calc(0,mid+1)>num:
			high = mid - 1
		else:
			low = mid + 1
	if st.calc(0,low+1)>num:
		return low
	else:
		return low + 1



n = int(input())
a = list(map(int,input().split()))
arr = [i for i in range(1,n+1)]
st = SegTree(n,arr)
st.construct()
ans = [-1]*n
for i in range(n-1,-1,-1):
	ind = find(a[i])
	# print (a[i],ind,arr)
	ans[i] = arr[ind]
	st.update(ind,0)
print(*ans)

