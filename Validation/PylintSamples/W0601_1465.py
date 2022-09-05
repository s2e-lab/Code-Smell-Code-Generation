import sys

def powc(x,n,m):
  res = 1
  xx=x
  while n:
    if n&1:
      res = (res*xx)%m
    xx=xx*xx%m
    n >>= 1
  return res

def circles(u):
  r = 0
  S = [(u,-1,0)]
  Visited[u] = 0
  for s in S:
    for e in V[s[0]]:
      if e[0] != s[1]:
        if Visited[e[0]]==-1: 
          Visited[e[0]] = s[2]^e[1]
          S.append((e[0], s[0], s[2]^e[1])) 
        elif Visited[e[0]] != s[2]^e[1]:
          return -1
        else:
          r += s[0]<e[0]
  return r

T = int(sys.stdin.readline())
for _ in range(T):
  is_bad = False
  empty = 0
  n,Q = list(map(int, sys.stdin.readline().split()))
  for _ in range(n-1):
    sys.stdin.readline() 
  paths = []
  V=list(map(list,[[]]*n))
  for q in range(Q):
    u,v,x = list(map(int, sys.stdin.readline().split()))
    u-=1
    v-=1
    if (v,x^1) in V[u]:
      is_bad = True
    elif (v,x) in V[u]:
      empty += 1
    elif u!=v:
      V[u].append((v,x))
      V[v].append((u,x))
    elif x==1:
      is_bad = True
    else:
      empty += 1
    paths.append((u,v,x))
  if is_bad:
    print(0)
  elif n<=1:
    print(1)
  else:
    Visited = [-1]*n
    components = 0
    for i in range(n):
      if Visited[i]==-1:
        components += 1
        c = circles(i)
        if c==-1:
          is_bad = True
          break
        empty += c
    if is_bad:
      print(0)
    else:
      print(powc(2,n-1-(Q-empty),10**9+7)) 

import sys

def powc(x,n,m):
  res = 1
  xx=x
  while n:
    if n&1:
      res = (res*xx)%m
    xx=xx*xx%m
    n >>= 1
  return res

def circles(u):
  r = 0
  S = [(u,-1,0)]
  Been = [-1]*n
  Been[u] = 0
  for s in S:
    Visited[s[0]] = 1
    for e in V[s[0]]:
      if e[0] != s[1]:
        if Been[e[0]]==-1: 
          Been[e[0]] = s[2]^e[1]
          S.append((e[0], s[0], s[2]^e[1])) 
        elif Been[e[0]] != s[2]^e[1]:
          return -1
        else:
          r += s[0]<e[0]
  return r

T = int(sys.stdin.readline())
for _ in range(T):
  is_bad = False
  empty = 0
  n,Q = list(map(int, sys.stdin.readline().split()))
  for _ in range(n-1):
    sys.stdin.readline() 
  paths = []
  V=list(map(list,[[]]*n))
  E = []
  for q in range(Q):
    u,v,x = list(map(int, sys.stdin.readline().split()))
    u-=1
    v-=1
    if (v,x^1) in V[u]:
      is_bad = True
    elif (v,x) in V[u]:
      empty += 1
    elif u!=v:
      E.append((u,v,x))
      V[u].append((v,x))
      V[v].append((u,x))
    elif x==1:
      is_bad = True
    else:
      empty += 1
    paths.append((u,v,x))
  if is_bad:
    print(0)
  elif n<=1:
    print(1)
  else:
    Visited = [0]*n
    components = 0
    for i in range(n):
      if Visited[i]==0:
        components += 1
        c = circles(i)
        if c==-1:
          is_bad = True
          break
        empty += c
    if is_bad:
      print(0)
    else:
      print(powc(2,n-1-(Q-empty),10**9+7)) 

import sys

def powc(x,n,m):
  res = 1
  xx=x
  while n:
    if n&1:
      res = (res*xx)%m
    xx=xx*xx%m
    n >>= 1
  return res

def circles(u):
  r = 0
  S = [(u,-1,0)]
  Been = [-1]*n
  for s in S:
    if Been[s[0]]!=-1:
      if Been[s[0]][1] != s[2]:
        return -1
      r += 1
      continue
    Been[s[0]] = (0,s[2])
    Visited[s[0]] = 1
    for e in V[s[0]]:
      if e[0] != s[1]:
        if Been[e[0]]==-1: 
          S.append((e[0], s[0], s[2]^e[1])) 
  return r

T = int(sys.stdin.readline())
for _ in range(T):
  is_bad = False
  empty = 0
  n,Q = list(map(int, sys.stdin.readline().split()))
  for _ in range(n-1):
    sys.stdin.readline() 
  paths = []
  V=list(map(list,[[]]*n))
  E = []
  for q in range(Q):
    u,v,x = list(map(int, sys.stdin.readline().split()))
    u-=1
    v-=1
    if (v,x^1) in V[u]:
      is_bad = True
    elif (v,x) in V[u]:
      empty += 1
    elif u!=v:
      E.append((u,v,x))
      V[u].append((v,x))
      V[v].append((u,x))
    elif x==1:
      is_bad = True
    else:
      empty += 1
    paths.append((u,v,x))
  if is_bad:
    print(0)
  elif n<=1:
    print(1)
  else:
    Visited = [0]*n
    components = 0
    for i in range(n):
      if Visited[i]==0:
        components += 1
        c = circles(i)
        if c==-1:
          is_bad = True
          break
        empty += c
    if is_bad:
      print(0)
    else:
      print(powc(2,n-1-(Q-empty),10**9+7)) 

import sys

def powc(x,n,m):
  res = 1
  xx=x
  while n:
    if n&1:
      res = (res*xx)%m
    xx=xx*xx%m
    n >>= 1
  return res

def findRoot():
  S = [(0,-1)]
  for u in S:
    for w in V[u[0]]:
      if w[0]!=u[1]:
        S.append((w[0],u[0]))
  S = [(S[-1][0],-1,0)]
  D = [0]*n
  for u in S:
    for w in V[u[0]]:
      if w[0]!=u[1]:
        D[w[0]]=u[2]+1
        S.append((w[0],u[0],u[2]+1))
  d = S[-1][2]
  size = d
  u = S[-1][0]
  while size/2<d:
    for w in V[u]:
      if D[w[0]]+1==D[u]:
        u = w[0]
        d -= 1
        break 
  return u
  
class Node:
  def __init__(self, value, edge, parent = None):
    self.value = value
    self.edge = edge
    if parent:
      parent.addChild(self)
    else:
      self.parent = None
    self.children = []
  def addChild(self, node):
    node.parent = self
    self.children.append(node)
  def __repr__(self):
    r = repr(self.value)
    for v in self.children:
      r += ' ' + repr(v)
    return r


def hangTree(root):
  global NodesArray
  NodesArray = [None]*n
  S=[(root, Node(root,-1),-1)]
  NodesArray[root] = S[0][1]
  for u in S:
    for v in V[u[0]]:
      if v[0] != u[2]:
        node = Node(v[0],v[1],u[1])
        NodesArray[v[0]] = node
        S.append((v[0],node,u[0]))

def findPath2(u,v):
  n0 = NodesArray[u]
  n1 = NodesArray[v]
  q = [0]*n
  while n0.parent:
    q[n0.edge] ^= 1
    n0 = n0.parent
  while n1.parent:
    q[n1.edge] ^= 1
    n1 = n1.parent
  return q
         
T = int(sys.stdin.readline())
for _ in range(T):
  n,Q = list(map(int,sys.stdin.readline().split()))
  V = list(map(list,[[]]*n))
  W = [0]*n
  for i in range(n-1):
    u,v = list(map(int,sys.stdin.readline().split()))
    u-=1
    v-=1
    V[u].append((v,i))
    V[v].append((u,i))
    W[u] += 1
    W[v] += 1
  easy = n==1
  root = findRoot()
  hangTree(root)
  M = []
  for _ in range(Q):
    u,v,x = list(map(int,sys.stdin.readline().split()))
    if not easy:
      q = findPath2(u-1,v-1)
      q[-1] = x
      M.append(q)
  if easy:
    print(1)
    continue
  empty = [0]*n
  bad = [0]*n
  bad[-1] = 1
  is_there_bad = False
  empty_cnt = 0
  i = 0
  for q in M:
    i += 1
    if q == empty:
      empty_cnt += 1
      continue
    if q == bad:
      is_there_bad = True
      break
    o = q.index(1)
    for next in range(i,Q):
      if M[next][o]==1:
        for k in range(n):
          M[next][k] ^= q[k]
  if is_there_bad:
    print(0)
  else:
    print(powc(2,n-1-Q+empty_cnt,10**9+7))

import sys

def powc(x,n,m):
  res = 1
  xx=x
  while n:
    if n&1:
      res = (res*xx)%m
    xx=xx*xx%m
    n >>= 1
  return res


def findPath(u,v,x):
  S = [(u,v,x)]
  for s in S:
    if s[0]==v:
      return s[2]
    for e in V[s[0]]: 
      if e[0] != s[1]:
        S.append((e[0],s[0],s[2]^e[1]))
  return None

T = int(sys.stdin.readline())
for _ in range(T):
  is_bad = False
  empty = 0
  n,Q = list(map(int, sys.stdin.readline().split(' ')))
  for _ in range(n-1):
    sys.stdin.readline() 
  paths = []
  V=list(map(list,[[]]*n))
  E = []
  for q in range(Q):
    u,v,x = list(map(int, sys.stdin.readline().split(' ')))
    u-=1
    v-=1
    if (v,x^1) in V[u]:
      is_bad = True
    elif (v,x) in V[u]:
      empty += 1
    else:
      E.append((u,v,x))
      V[u].append((v,x))
      V[v].append((u,x))
    paths.append((u,v,x))
  if is_bad:
    print(0)
  else:
    while E:
      e = E.pop()
      x = findPath(e[0],e[1],e[2]) 
      V[e[0]].remove((e[1],e[2]))
      V[e[1]].remove((e[0],e[2]))
      if x==1:
        is_bad = True
        break
      if x==0:
        empty += 1
    if is_bad:
      print(0)
    else:
      print(powc(2,n-1-(Q-empty),10**9+7))

def modpow(a,x):
	if(x==0):
		return 1;
	elif(x%2==0):
		t=modpow(a,x/2);
		return (t*t)%(1000000007);
	else:
		t=modpow(a,x/2);
		return (t*t*a)%(1000000007);
		
					
T=eval(input());
ans=[0]*T;
for j in range(T):
	[N,Q]=[int(x) for x in (input()).split()];
	for i in range(N-1):
		input();
	comp=list(range(N+1));
	revcomp=[];
	for i in range(N+1):
		revcomp.append([i]);	
	sumcomp=[0]*(N+1);
	flag=True;
	rank=0;
	for i in range(Q):
		if(not(flag)):
			input();
		else:	
			[u,v,x]=[int(x) for x in (input()).split()];
			if(comp[u]==comp[v]):
				if(not((sumcomp[u]+sumcomp[v])%2==(x%2))):
					flag=False;
			else:
				rank=rank+1;
				n1=len(revcomp[comp[u]]);
				n2=len(revcomp[comp[v]]);
				if(n1<n2):
					oldsu=sumcomp[u];
					l=revcomp[comp[v]];
					for w in revcomp[comp[u]]:
						l.append(w);
						comp[w]=comp[v];
						sumcomp[w]=(sumcomp[w]+sumcomp[v]+x+oldsu)%2;
					#revcomp[comp[u]]=[];	
				else:
					oldsv=sumcomp[v];
					l=revcomp[comp[u]];
					for w in revcomp[comp[v]]:
						l.append(w);
						comp[w]=comp[u];
						sumcomp[w]=(sumcomp[w]+sumcomp[u]+x+oldsv)%2;
					#revcomp[comp[v]]=[];	
	if(not(flag)):
		ans[j]=0;
	else:
		ans[j]=modpow(2,(N-rank-1));

for j in range(T):
	print((ans[j]));			

def modpow(a,x):
	if(x==0):
		return 1;
	elif(x%2==0):
		t=modpow(a,x/2);
		return (t*t)%(1000000007);
	else:
		t=modpow(a,x/2);
		return (t*t*a)%(1000000007);
		
			
		
			
T=eval(input());
ans=[0]*T;
for j in range(T):
	[N,Q]=[int(x) for x in (input()).split()];
	for i in range(N-1):
		input();
	comp=list(range(N+1));
	revcomp=[];
	for i in range(N+1):
		revcomp.append([i]);	
	sumcomp=[0]*(N+1);
	flag=True;
	rank=0;
	for i in range(Q):
		if(not(flag)):
			input();
		else:	
			[u,v,x]=[int(x) for x in (input()).split()];
			if(comp[u]==comp[v]):
				if(not((sumcomp[u]+sumcomp[v])%2==(x%2))):
					flag=False;
			else:
				rank=rank+1;
				n1=len(revcomp[comp[u]]);
				n2=len(revcomp[comp[v]]);
				if(n1<n2):
					oldsu=sumcomp[u];
					l=revcomp[comp[v]];
					for w in revcomp[u]:
						l.append(w);
						comp[w]=comp[v];
						sumcomp[w]=(sumcomp[w]+sumcomp[v]+x+oldsu)%2;
				else:
					oldsv=sumcomp[v];
					l=revcomp[comp[u]];
					for w in revcomp[v]:
						l.append(w);
						comp[w]=comp[u];
						sumcomp[w]=(sumcomp[w]+sumcomp[u]+x+oldsv)%2;
	if(not(flag)):
		ans[j]=0;
	else:
		ans[j]=modpow(2,(N-rank-1));

for j in range(T):
	print((ans[j]));			
							
								
			
		


T=eval(input());
ans=[0]*T;
for j in range(T):
	[N,Q]=[int(x) for x in (input()).split()];
	for i in range(N-1):
		input();
	comp=list(range(N+1));
	revcomp=[];
	for i in range(N+1):
		revcomp.append([i]);	
	sumcomp=[0]*(N+1);
	flag=True;
	rank=0;
	for i in range(Q):
		if(not(flag)):
			input();
		else:	
			[u,v,x]=[int(x) for x in (input()).split()];
			if(comp[u]==comp[v]):
				if(not((sumcomp[u]+sumcomp[v])%2==(x%2))):
					flag=False;
			else:
				rank=rank+1;
				n1=len(revcomp[comp[u]]);
				n2=len(revcomp[comp[v]]);
				if(n1<n2):
					oldsu=sumcomp[u];
					l=revcomp[comp[v]];
					for w in revcomp[u]:
						l.append(w);
						comp[w]=comp[v];
						sumcomp[w]=(sumcomp[w]+sumcomp[v]+x+oldsu)%2;
				else:
					oldsv=sumcomp[v];
					l=revcomp[comp[u]];
					for w in revcomp[v]:
						l.append(w);
						comp[w]=comp[u];
						sumcomp[w]=(sumcomp[w]+sumcomp[u]+x+oldsv)%2;
	if(not(flag)):
		ans[j]=0;
	else:
		ans[j]=2**(N-rank-1);

for j in range(T):
	print((ans[j]));			
							
								
			
		


