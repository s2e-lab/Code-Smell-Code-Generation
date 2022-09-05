class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        if len(edges) == 0:
            return 0 if n == 0 else -1
        p = [i for i in range(n)]
        def getP(ind):
            nonlocal p
            if p[ind] == ind:
                return ind
            else:
                res = getP(p[ind])
                p[ind] = res
                return res
        cnt = 0
        for t,u,v in edges:
            if t == 3:
                pu,pv = getP(u-1), getP(v-1)
                if pu != pv:
                    p[pv] = pu
                    cnt += 1
        if cnt != (n - 1):
            pa = list(p)
            for t,u,v in edges:
                if t == 1:
                    pu,pv = getP(u-1), getP(v-1)
                    if pu != pv:
                        p[pv] = pu
                        cnt += 1
            targetP = getP(0)
            for v in range(n):
                if getP(v) != targetP:
                    return -1
            p = pa
            for t,u,v in edges:
                if t == 2:
                    pu,pv = getP(u-1), getP(v-1)
                    if pu != pv:
                        p[pv] = pu
                        cnt += 1
            targetP = getP(0)
            for v in range(n):
                if getP(v) != targetP:
                    return -1
        return len(edges) - cnt
class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        A = list(range(n))
        
        def fixed(i,arr):
            if i!=arr[i]:
                arr[i] = fixed(arr[i],arr)
            return arr[i]
        
        for ty,s,t in edges:
            if ty==3:
                x = fixed(s-1,A)
                y = fixed(t-1,A)
                A[x] = y
        
        both_comps = sum(1 for i in range(n) if A[i] == i)
        
        B = A[:]
        
        for ty, s,t in edges:
            if ty==1:
                x = fixed(s-1,A)
                y = fixed(t-1,A)
                A[x] = y
        
        a_comps = sum(1 for i in range(n) if A[i] == i)

        if a_comps!=1:
            return -1
        
        for ty,s,t in edges:
            if ty==2:
                x = fixed(s-1,B)
                y = fixed(t-1,B)
                B[x] = y
        
        b_comps = sum(1 for i in range(n) if B[i] == i)

        if b_comps!=1:
            return -1
        
        return len(edges) - (n-a_comps-b_comps+both_comps)
class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        com_conn = [i for i in range(n+1)]
        size = [1 for i in range(n+1)]
        com_count = [n]
        removed = [0]
        coed = 0
        a_edge = list()
        b_edge = list()
        for i in edges:
            if i[0] == 3:
                self.union(i[1],i[2], com_conn, com_count, removed,size)
                coed +=1
            elif i[0] == 1:
                a_edge.append(i)
            else:
                b_edge.append(i)
        if com_count[0] ==1:
            return removed[0]+len(edges)-coed

        aconn = com_conn.copy()
        asize = size.copy()
        acom_count = com_count.copy()
        for i in a_edge:
            if i[0]==1:
                self.union(i[1], i[2], aconn, acom_count, removed, asize)
        if acom_count[0] >1:
            return -1

        bconn = com_conn.copy()
        bsize = size.copy()
        bcom_count = com_count.copy()
        for i in b_edge:
            if i[0]==2:
                self.union(i[1], i[2], bconn, bcom_count, removed,bsize)
        if bcom_count[0] >1:
            return -1
        return removed[0]


    def find(self, p, connect):
        while p != connect[p]:
            p =connect[p]
        return p

    def union(self, p, q, connect, count, remove, size):
        proot = self.find(p, connect)
        qroot = self.find(q, connect)
        if proot == qroot:
            remove[0]+=1
            return
        if size[proot] > size[qroot]:
            connect[qroot] = proot
            size[proot] += size[qroot]
        else:
            connect[proot] = qroot
            size[qroot] += size[proot]
        count[0] -=1

import copy

def union(subsets, u, v):
    uroot = find(subsets, u)
    vroot = find(subsets, v)
    
    if subsets[uroot][1] > subsets[vroot][1]:
        subsets[vroot][0] = uroot
    if subsets[vroot][1] > subsets[uroot][1]:
        subsets[uroot][0] = vroot
    if subsets[uroot][1] == subsets[vroot][1]:
        subsets[vroot][0] = uroot
        subsets[uroot][1] += 1
    

def find(subsets, u):
    if subsets[u][0] != u:
        subsets[u][0] = find(subsets, subsets[u][0])
    return subsets[u][0]


class Solution:
    #kruskal's
    #1 is alice and 2 is bob
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        
        subsets1 = ['1 index'] + [[x+1,0] for x in range(n)] #Alice's unionfind
        subsets2 = ['1 index'] + [[x+1,0] for x in range(n)] #Bob's unionfind
        
        edges = sorted(edges, key= lambda e: -e[0])
        e = 0 #number of total edges used
        e1 = 0 #number of edges for Alice
        e2 = 0 #number of edges for Bob
        i = 0 #track position in edges list
        
        #start with type 3 edges
        while e < n - 1:
            if i == len(edges): 
                return -1
            typ, u, v = edges[i]
            if typ != 3: break
            if find(subsets1, u) != find(subsets1, v):
                union(subsets1, u, v)
                e += 1
            
            i += 1
        
        #everything that was done to Alice applies to Bob
        e1 = e
        e2 = e
        subsets2 = copy.deepcopy(subsets1)
        
        #once done with shared edges, do Bob's
        while e2 < n-1:
            if i == len(edges): 
                return -1
            typ, u, v = edges[i]
            if typ != 2: break
            if find(subsets2, u) != find(subsets2, v):
                union(subsets2, u, v)
                e += 1
                e2 += 1
            i += 1
        
        if e2 < n - 1: 
            return -1 #if we've used all edges bob can use (types 2 and 3) and he still can't reach all nodes, ur fucked
        
        #now finish Alice's MST
        while e1 < n-1:
            if i == len(edges): 
                return -1
            
            typ, u, v = edges[i]
            if find(subsets1, u) != find(subsets1, v):
                union(subsets1, u, v)
                e += 1
                e1 += 1
            i += 1
            
        return len(edges) - e
            
            
            
            
        
        

class UF:
    def __init__(self, n: int):
        self.p, self.e = list(range(n)), 0
        
    def find(self, x: int):
        if x != self.p[x]:
            self.p[x] = self.find(self.p[x])
        return self.p[x]

    def union(self, x: int, y: int):
        rx, ry = self.find(x), self.find(y)
        if rx == ry: return 1
        self.p[rx] = ry
        self.e += 1
        return 0
class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        A, B = UF(n+1), UF(n+1)
        ans = 0
        for t, u, v in edges:
            if t != 3:
                continue
            ans += A.union(u, v) # return 1 if connected else 0
            B.union(u, v)
        for t, u, v in edges:
            if t == 3:
                continue
            d = A if t == 1 else B
            ans += d.union(u, v)
        return ans if A.e == B.e == n - 1 else -1 # merge times equal to edges

class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        roots = [i for i in range(n)]
        def find(u):
            if u != roots[u]:
                roots[u] = find(roots[u])
            return roots[u]
        def union(u,v):
            pu,pv = find(u), find(v)
            if pu == pv:
                return False
            roots[max(pu,pv)] = min(pu,pv)
            return True
        
        edges = sorted(edges, reverse=True)
        i = 0
        connect_times = 0
        to_remove = 0
        while i < len(edges):
            e = edges[i]
            if e[0] != 3:
                break
            
            res = union(e[1]-1, e[2]-1)
            if res == True:
                connect_times += 1
            else:
                to_remove += 1
            
            i += 1
            
        origin_roots = deepcopy(roots)
        origin_connect = connect_times
        while i < len(edges):
            e = edges[i]
            if e[0] != 2:
                break
            
            res = union(e[1]-1, e[2]-1)
            if res == True:
                connect_times += 1
            else:
                to_remove += 1
            
            i += 1
        if connect_times != n-1:
            return -1
        
        connect_times = origin_connect
        roots = origin_roots
        while i < len(edges):
            e = edges[i]
            
            res = union(e[1]-1, e[2]-1)
            if res == True:
                connect_times += 1
            else:
                to_remove += 1
            
            i += 1
            
        if connect_times != n-1:
            return -1
        
        return to_remove
        
        
        

class DSU:
    def __init__(self, N):
        self.parent = [i for i in range(N)]
        self.size = [1] * N
        
    def find(self, x):
        if self.parent[x] == x:
            return x
        return self.find(self.parent[x])
    
    def union(self, x, y):
        par_x, par_y = self.find(x), self.find(y)
        
        if par_x == par_y:
            return False
        if self.size[par_x] < self.size[par_y]:
            par_x , par_y = par_y, par_x
        
        self.parent[par_y] = par_x
        self.size[par_x] += self.size[par_y]
        self.size[par_y] = self.size[par_x]
    
        return True
    def getSize(self, x):
        return self.size[self.find(x)]
    
class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        
        
        alice_graph, bob_graph = DSU(n+1), DSU(n+1)
        
        ret = 0
        
        for t, start, end in edges:
            if t == 3:
                if not alice_graph.union(start, end):
                    ret += 1
                bob_graph.union(start, end)
        
        for t, start, end in edges:
            if t == 1:
                if not alice_graph.union(start, end):
                    ret += 1
            if t == 2:
                if not bob_graph.union(start, end):
                    ret += 1
        
        
        return ret if alice_graph.getSize(1) == bob_graph.getSize(1) == n else -1
                    
            
        
        
        
        

class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        
        
        
        class UnionFind:
            def __init__(self, n):
                self.count = n  # Number of groups
                self.root = list(range(n))  # Root node of each element
                self.leaf = [1] * n  # Number of leaf elements

            def union(self, e1, e2):
                x, y = self.find(e1), self.find(e2)
                if x != y:
                    self.count -= 1
                    if self.leaf[x] > self.leaf[y]:
                        x, y = y, x
                    self.leaf[y] += self.leaf[x]
                    self.root[x] = y

            def find(self, el):
                if el == self.root[el]:
                    return el
                else:
                    return self.find(self.root[el])

        UFA = UnionFind(n)
        UFB = UnionFind(n)

        edges.sort(reverse=True)

        res = 0
        for t, a, b in edges:
            a, b = a - 1, b - 1
            if t == 3:
                if UFA.find(a) == UFA.find(b):
                    res += 1
                else:
                    UFA.union(a, b)
                    UFB.union(a, b)
            elif t == 2:
                if UFB.find(a) == UFB.find(b):
                    res += 1
                else:
                    UFB.union(a, b)
            elif t == 1:
                if UFA.find(a) == UFA.find(b):
                    res += 1
                else:
                    UFA.union(a, b)
            else:
                pass

        return res if UFA.count == 1 and UFB.count == 1 else -1
class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        edges = sorted(edges, key=lambda k:k[0], reverse=True)
        
        A_ranks, B_ranks = [i for i in range(n+1)] ,[i for i in range(n+1)]
        
        def find(x, ranks):
            if ranks[x] != x:
                ranks[x] = find(ranks[x], ranks)
            return ranks[x]
        
        def union(x, y, ranks):
            rk_x, rk_y = find(x, ranks), find(y, ranks)
            if rk_x < rk_y:
                ranks[rk_y] = rk_x
            elif rk_y < rk_x:
                ranks[rk_x] = rk_y
                
        steps, e_A, e_B = 0, 0, 0
        for i in range(len(edges)):
            c, x, y = edges[i]
            if c == 3:
                A_x, A_y, B_x, B_y = find(x, A_ranks), find(y, A_ranks), find(x, B_ranks), find(y, B_ranks)
                if A_x != A_y or B_x != B_y:
                    union(x, y, A_ranks)
                    union(x, y, B_ranks)
                    e_A, e_B = e_A+1, e_B+1
                else:
                    steps += 1
#        print(e_A, e_B, steps)
        for i in range(len(edges)):
            c, x, y = edges[i]
            if c == 2:
                B_x, B_y = find(x, B_ranks), find(y, B_ranks)
                if B_x != B_y:
                    union(x, y, B_ranks)
                    e_B += 1
                else:
                    steps += 1
#        print(e_A, e_B, steps)
        for i in range(len(edges)):  
            c, x, y = edges[i]
            if c == 1:
                A_x, A_y = find(x, A_ranks), find(y, A_ranks)
                if A_x != A_y:
                    union(x, y, A_ranks)
                    e_A += 1
                else:
                    steps += 1
        
        #extra codes to actually groups elements together
        for i in range(1, n+1):
            find(i, A_ranks)
            find(i, B_ranks)
#        print(e_A, e_B, steps)
        print(A_ranks, B_ranks)
        
        return steps if e_A == e_B == n-1 else -1
import sys
input = sys.stdin.readline

class Unionfind:
    def __init__(self, n):
        self.par = [-1]*n
        self.rank = [1]*n
    
    def root(self, x):
        r = x
        
        while not self.par[r]<0:
            r = self.par[r]
        
        t = x
        
        while t!=r:
            tmp = t
            t = self.par[t]
            self.par[tmp] = r
        
        return r
    
    def unite(self, x, y):
        rx = self.root(x)
        ry = self.root(y)
        
        if rx==ry:
            return
        
        if self.rank[rx]<=self.rank[ry]:
            self.par[ry] += self.par[rx]
            self.par[rx] = ry
            
            if self.rank[rx]==self.rank[ry]:
                self.rank[ry] += 1
        else:
            self.par[rx] += self.par[ry]
            self.par[ry] = rx
    
    def is_same(self, x, y):
        return self.root(x)==self.root(y)
    
    def count(self, x):
        return -self.par[self.root(x)]

class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        uf1 = Unionfind(n)
        uf2 = Unionfind(n)
        ans = 0
        
        for t, u, v in edges:
            if t==3:
                if not uf1.is_same(u-1, v-1):
                    uf1.unite(u-1, v-1)
                    uf2.unite(u-1, v-1)
                else:
                    ans += 1
        
        for t, u, v in edges:
            if t==1:
                if not uf1.is_same(u-1, v-1):
                    uf1.unite(u-1, v-1)
                else:
                    ans += 1
            elif t==2:
                if not uf2.is_same(u-1, v-1):
                    uf2.unite(u-1, v-1)
                else:
                    ans += 1
        
        if len(set(uf1.root(i) for i in range(n)))>1:
            return -1
        
        if len(set(uf2.root(i) for i in range(n)))>1:
            return -1
        
        return ans
class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        self.father_alice = [i for i in range(n + 1)]
        self.father_bob = [i for i in range(n + 1)]
        
        res = 0
        for type, u, v in edges:
            if type == 3:
                res += self.connect(u, v, True)
                self.connect(u, v, False)
        
        for type, u, v in edges:
            if type == 1:
                res += self.connect(u, v, True)
            elif type == 2:
                res += self.connect(u, v, False)
        
        
        if self.check_valid(True) and self.check_valid(False):
            return res
        return -1
    
    
    def find(self, x, is_alice):
        if is_alice:
            if self.father_alice[x] == x:
                return self.father_alice[x]
            self.father_alice[x] = self.find(self.father_alice[x], True)
            return self.father_alice[x]
        
        else:
            if self.father_bob[x] == x:
                return self.father_bob[x]
            self.father_bob[x] = self.find(self.father_bob[x], False)
            return self.father_bob[x]
        
    def connect(self, a, b, is_alice):
        if is_alice:
            root_a = self.find(a, True)
            root_b = self.find(b, True)
            if root_a != root_b:
                self.father_alice[max(root_a, root_b)] = min(root_a, root_b)
                return 0
            return 1
        
        else:
            root_a = self.find(a, False)
            root_b = self.find(b, False)
            if root_a != root_b:
                self.father_bob[max(root_a, root_b)] = min(root_a, root_b)
                return 0
            return 1
        
    def check_valid(self, is_alice):
        if is_alice:
            root = self.find(1, True)
            for i in range(1, len(self.father_alice)):
                if self.find(i, True) != root:
                    return False
            return True
        
        else:
            root = self.find(1, False)
            for i in range(1, len(self.father_bob)):
                if self.find(i, False) != root:
                    return False
            return True
class DSU:
    def __init__(self, n: int):
        self.p = list(range(n))
        self.e = 0
    
    def find(self, x: int) -> int:
        if x != self.p[x]: self.p[x] = self.find(self.p[x])
        return self.p[x]
    
    def merge(self, x: int, y: int) -> int:
        rx, ry = self.find(x), self.find(y)
        if rx == ry: return 1
        self.p[rx] = ry
        self.e += 1
        return 0
    
class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        A, B = DSU(n + 1), DSU(n + 1)    
        ans = 0
        for t, x, y in edges:
            if t != 3: continue
            ans += A.merge(x, y)
            B.merge(x, y)
        for t, x, y in edges:
            if t == 3: continue
            d = A if t == 1 else B
            ans += d.merge(x, y)
        return ans if A.e == B.e == n - 1 else -1
import copy

def union(subsets, u, v):
    uroot = find(subsets, u)
    vroot = find(subsets, v)
    
    if subsets[uroot][1] > subsets[vroot][1]:
        subsets[vroot][0] = uroot
    if subsets[vroot][1] > subsets[uroot][1]:
        subsets[uroot][0] = vroot
    if subsets[uroot][1] == subsets[vroot][1]:
        subsets[vroot][0] = uroot
        subsets[uroot][1] += 1
    

def find(subsets, u):
    if subsets[u][0] != u:
        subsets[u][0] = find(subsets, subsets[u][0])
    return subsets[u][0]


class Solution:
    #kruskal's
    #1 is alice and 2 is bob
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        
        mst1 = set() #set for Alice's MST
        mst2 = set() #set for Bob's MST
        subsets1 = ['1 index'] + [[x+1,0] for x in range(n)] #Alice's unionfind
        subsets2 = ['1 index'] + [[x+1,0] for x in range(n)] #Bob's unionfind
        
        edges = sorted(edges, key= lambda e: -e[0])
        e = 0 #number of total edges used
        e1 = 0 #number of edges for Alice
        e2 = 0 #number of edges for Bob
        i = 0 #track position in edges list
        
        #start with type 3 edges
        while e < n - 1:
            if i == len(edges): 
                return -1
            typ, u, v = edges[i]
            if typ != 3: break
            if find(subsets1, u) != find(subsets1, v):
                union(subsets1, u, v)
                mst1.add(u)
                mst1.add(v)
                e += 1
            
            i += 1
        
        #everything that was done to Alice applies to Bob
        e1 = e
        e2 = e
        mst2 = mst1.copy()
        subsets2 = copy.deepcopy(subsets1)
        
        #once done with shared edges, do Bob's
        while e2 < n-1:
            if i == len(edges): 
                return -1
            typ, u, v = edges[i]
            if typ != 2: break
            if find(subsets2, u) != find(subsets2, v):
                union(subsets2, u, v)
                mst2.add(u)
                mst2.add(v)
                e += 1
                e2 += 1
            i += 1
        
        if len(mst2) < n: 
            return -1 #if we've used all edges bob can use (types 2 and 3) and he still can't reach all nodes, ur fucked
        
        #now finish Alice's MST
        while e1 < n-1:
            if i == len(edges): 
                return -1
            
            typ, u, v = edges[i]
            if find(subsets1, u) != find(subsets1, v):
                union(subsets1, u, v)
                mst1.add(u)
                mst1.add(v)
                e += 1
                e1 += 1
            i += 1
            
        return len(edges) - e
            
            
            
            
        
        

class DSU():
    def __init__(self, n):
        self.parent = [i for i in range(n)]
        self.rank = [0 for _ in range(n)]
        self.size = n
    def find(self, x):
        if x != self.parent[x]:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    def union(self, x, y):
        xp, yp = self.find(x), self.find(y)
        if xp == yp:
            return False
        if self.rank[xp] < self.rank[yp]:
            self.parent[xp] = yp
        elif self.rank[xp] > self.rank[yp]:
            self.parent[yp] = xp
        else:
            self.parent[xp] = yp
            self.rank[yp] += 1
        self.size -= 1
        return True
    def getSize(self):
        return self.size
    
class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        dsu = DSU(n + 1)
        typeEdges = collections.defaultdict(list)
        for t, i, j in edges:
            typeEdges[t].append((i, j))
        res = 0
        for i, j in typeEdges[3]:
            if not dsu.union(i, j):
                res += 1
        if dsu.getSize() == 2:
            return res + sum(len(v) for k, v in typeEdges.items() if k in [1, 2])
        for i, j in typeEdges[1]:
            if not dsu.union(i, j):
                res += 1
        if dsu.getSize() > 2:
            return -1
        dsu1 = DSU(n + 1)
        for i, j in typeEdges[3]:
            dsu1.union(i, j)
        for i, j in typeEdges[2]:
            if not dsu1.union(i, j):
                res += 1
        if dsu1.getSize() > 2:
            return -1
        return res
class UFset:
    
    def __init__(self, n):
        self.parents = list(range(n))
        self.ranks = [1] * n
        self.size = 1
        
    def find(self, x):
        if x != self.parents[x]:
            self.parents[x] = self.find(self.parents[x])
        return self.parents[x]

    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py:
            return False
        if self.ranks[x] > self.ranks[y]:
            px, py = py, px
        self.parents[px] = py
        self.ranks[py] += 1
        self.size += 1
        return True


class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        alice = UFset(n)
        bob = UFset(n)
        res = 0
        for type_i, v, w in edges:
            v -= 1; w -= 1
            if type_i == 3:
                a = alice.union(v, w)
                b = bob.union(v, w)
                if not a and not b:
                    res += 1
        for type_i, v, w in edges:
            v -= 1; w -= 1
            if type_i == 1:
                if not alice.union(v, w):
                    res += 1
        for type_i, v, w in edges:
            v -= 1; w -= 1
            if type_i == 2:
                if not bob.union(v, w):
                    res += 1
        return res if alice.size == bob.size == n else -1
class DSU:
  def __init__(self, n):
    self.p = [-1]*(n+1)
    self.r = [0]*(n+1)
    
  def find_parent(self, x):
    if self.p[x]==-1:
      return x
    self.p[x] = self.find_parent(self.p[x]) # path compression
    return self.p[x]
  
  def union(self, a, b):
    pa = self.find_parent(a)
    pb = self.find_parent(b)
    if pa==pb: return False
    if self.r[pa]<=self.r[pb]:
      self.p[pb] = pa     # here rank can be adding
      self.r[pa] += 1
    else:
      self.p[pa] = pb
      self.r[pb] += 1
      
    return True
  
class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
      edges = sorted(edges, key=lambda x: -x[0])
      dsu_alice = DSU(n)
      dsu_bob = DSU(n)
      res = 0
      
      for e in edges:
        if e[0]==3:
          au = dsu_alice.union(e[1],e[2])
          bu = dsu_bob.union(e[1],e[2])
          if not au and not bu:
            res += 1
        elif e[0]==1:
          if not dsu_alice.union(e[1],e[2]):
            res += 1
        else:
          if not dsu_bob.union(e[1],e[2]):
            res += 1
        # print (e, res) 
      
      ap = 0
      bp = 0
      for i in range(1, n+1):
        if ap and dsu_alice.find_parent(i)!=ap:
          return -1
        else: ap = dsu_alice.find_parent(i)
        if bp and dsu_bob.find_parent(i)!=bp:
          return -1
        else: bp = dsu_bob.find_parent(i)
      return res

class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        
        
        
        
        class UnionFind:
            def __init__(self, n):
                self.group = n              # Number of groups
                self.rank = [1] * n         # Avoid infinite recursion
                self.root = list(range(n))  # Root node of each element

            def union(self, e1, e2):
                x = self.find(e1)
                y = self.find(e2)
                if x != y:
                    self.group -= 1
                    if self.rank[x] > self.rank[y]:
                        x, y = y, x
                    self.rank[y] += self.rank[x]
                    self.root[x] = y

            def find(self, el):
                if el == self.root[el]:
                    return el
                else:
                    return self.find(self.root[el])

        UFA = UnionFind(n)
        UFB = UnionFind(n)

        edges.sort(reverse=True)

        res = 0
        for t, a, b in edges:
            a, b = a - 1, b - 1
            if t == 3:
                if UFA.find(a) == UFA.find(b):
                    res += 1
                else:
                    UFA.union(a, b)
                    UFB.union(a, b)
            elif t == 2:
                if UFB.find(a) == UFB.find(b):
                    res += 1
                else:
                    UFB.union(a, b)
            elif t == 1:
                if UFA.find(a) == UFA.find(b):
                    res += 1
                else:
                    UFA.union(a, b)
            else:
                pass

        return res if UFA.group == 1 and UFB.group == 1 else -1
import copy

def union(subsets, u, v):
    uroot = find(subsets, u)
    vroot = find(subsets, v)
    
    if subsets[uroot][1] > subsets[vroot][1]:
        subsets[vroot][0] = uroot
    if subsets[vroot][1] > subsets[uroot][1]:
        subsets[uroot][0] = vroot
    if subsets[uroot][1] == subsets[vroot][1]:
        subsets[vroot][0] = uroot
        subsets[uroot][1] += 1
    

def find(subsets, u):
    if subsets[u][0] != u:
        subsets[u][0] = find(subsets, subsets[u][0])
    return subsets[u][0]


class Solution:
    #kruskal's
    #1 is alice and 2 is bob
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        
        mst1 = set() #set for Alice's MST
        mst2 = set() #set for Bob's MST
        subsets1 = ['1 index'] + [[x+1,0] for x in range(n)] #Alice's unionfind
        subsets2 = ['1 index'] + [[x+1,0] for x in range(n)] #Bob's unionfind
        
        edges = sorted(edges, key= lambda e: -e[0])
        e = 0 #number of total edges used
        e1 = 0 #number of edges for Alice
        e2 = 0 #number of edges for Bob
        i = 0 #track position in edges list
        
        #start with type 3 edges
        while e < n - 1:
            if i == len(edges): 
                return -1
            typ, u, v = edges[i]
            if typ != 3: break
            if find(subsets1, u) != find(subsets1, v):
                union(subsets1, u, v)
                mst1.add(u)
                mst1.add(v)
                e += 1
            
            i += 1
        
        #everything that was done to Alice applies to Bob
        e1 = e
        e2 = e
        mst2 = mst1.copy()
        subsets2 = copy.deepcopy(subsets1)
        
        #once done with shared edges, do Bob's
        while e2 < n-1:
            if i == len(edges): 
                return -1
            typ, u, v = edges[i]
            if typ != 2: break
            if find(subsets2, u) != find(subsets2, v):
                union(subsets2, u, v)
                mst2.add(u)
                mst2.add(v)
                e += 1
                e2 += 1
            i += 1
        
        if len(mst2) < n: 
            return -1 #if we've used all edges bob can use (types 2 and 3) and he still can't reach all nodes, ur fucked
        
        #now finish Alice's MST
        while e1 < n-1:
            if i == len(edges): 
                return -1
            
            typ, u, v = edges[i]
            if find(subsets1, u) != find(subsets1, v):
                union(subsets1, u, v)
                
                e += 1
                e1 += 1
            i += 1
            
        return len(edges) - e
            
            
            
            
        
        

import copy

class DJ_DS():
    def __init__(self, n):
        self.n = n
        self.parent = [i for i in range(n)]
        self.rank = [0 for i in range(n)]
        self.nb_edges = 0
    
    def find_parent(self,i): # faster with path compression
        if self.parent[i] != i:
            self.parent[i] = self.find_parent(self.parent[i])
        return self.parent[i]
        
    def union(self,i,j):
        p_i = self.find_parent(i)
        p_j = self.find_parent(j)
        
        if p_i != p_j:
            self.nb_edges += 1
            if self.rank[p_i] < self.rank[p_j]:
                self.parent[p_i] = p_j
            else:
                self.parent[p_j] = p_i
                if self.rank[p_i] == self.rank[p_j]:
                    self.rank[p_i] += 1
                
    def perform_merge(self, edges):
        for [u,v] in edges:
            self.union(u,v)
            

class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        nb_edges = len(edges)
        # list of edges of each color individually
        type1, type2, type3 = [], [], []
        for [t,u,v] in edges:
            if t == 1:
                type1.append([u-1,v-1])
            elif t == 2:
                type2.append([u-1,v-1])
            else:
                type3.append([u-1,v-1])
        
        # Count nb_edges with type 3 only in max forest
        dj_3 = DJ_DS(n)
        dj_3.perform_merge(type3)
        sol_3 = dj_3.nb_edges
        dj_1 = copy.deepcopy(dj_3)
        dj_2 = copy.deepcopy(dj_3)
        
        # From type 3 forest add edges from type 1 to see if spanning tree, if not return -1
        dj_1.perform_merge(type1)
        if dj_1.nb_edges < n-1:
            return -1
        
        # From type 3 forest add edges from type 2 to see if spanning tree, if not return -1
        dj_2.perform_merge(type2)
        if dj_2.nb_edges < n-1:
            return -1
        
        return (nb_edges - (sol_3 + 2 * (n-1 - sol_3)))
class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        edges.sort(key=lambda x: -x[0])
        uf1 = UnionFind(n+1)
        uf2 = UnionFind(n+1)
        
        e1 = e2 = 0
        res = 0
        
        for type_, node1, node2 in edges:
            
            if type_ == 3:
                val1 = uf1.union(node1, node2) 
                val2 = uf2.union(node1, node2)
                res += val1 or val2
                e1 += val1
                e2 += val2
            
            if type_ == 1:
                val = uf1.union(node1, node2)
                res += val
                e1 += val
            
            else:
                val = uf2.union(node1, node2)
                res += val
                e2 += val
        
        if e1 == e2 == n-1:
            # print(res)
            return len(edges) - res
        return -1
    
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
    
    def union(self, x, y):
        rootx = self.find(x)
        rooty = self.find(y)
        
        if rootx == rooty:
            return False
        
        self.parent[rooty] = rootx
        return True
    
    def find(self, x):
        while x != self.parent[x]:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x
        
        

class UnionFind:
    def __init__(self, n):
        self.count = n
        self.parent = list(range(n))
        self.rank = [1]*n
        
    def find(self, p):
        if p != self.parent[p]: 
            self.parent[p] = self.find(self.parent[p])
        return self.parent[p]
    
    def union(self, p, q):
        prt, qrt = self.find(p), self.find(q)
        if prt == qrt: return False
        self.count -= 1
        if self.rank[prt] > self.rank[qrt]: prt, qrt = qrt, prt
        self.parent[prt] = qrt
        self.rank[qrt] += self.rank[prt]
        return True
    
        
class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        ufa = UnionFind(n) # for Alice
        ufb = UnionFind(n) # for Bob
        
        ans = 0
        edges.sort(reverse=True) 
        for t, u, v in edges: 
            u, v = u-1, v-1
            if t == 3: # Alice & Bob
                if not (ufa.union(u, v) and ufb.union(u, v)): ans += 1
            elif t == 2: # Bob only
                if not ufb.union(u, v): ans += 1
            else: # Alice only
                if not ufa.union(u, v): ans += 1
        return ans if ufa.count == 1 and ufb.count == 1 else -1
class DSU:
    def __init__(self, n):
        self.p=list(range(n))
        self.e=0
    def find(self,x):
        if x!=self.p[x]:
            self.p[x]=self.find(self.p[x])
        return self.p[x]
    
    def merge(self,x, y):
        rx,ry=self.find(x), self.find(y)
        if rx==ry: return 1
        self.p[rx]=ry
        self.e+=1   # merged vertice
        return 0
    
class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        A,B=DSU(n+1), DSU(n+1)
        ans=0
        for t,x,y in edges:
            if t!=3: continue
            ans+=A.merge(x,y)
            B.merge(x,y)
        for t, x, y in edges:
            if t==3: continue
            d=A if t==1 else B
            ans+=d.merge(x,y)
        return ans if A.e==B.e==n-1 else -1

class UFset:
    
    def __init__(self, n):
        self.parents = list(range(n))
        self.ranks = [1] * n
        self.size = 1
        
    def find(self, x):
        if x != self.parents[x]:
            self.parents[x] = self.find(self.parents[x])
        return self.parents[x]

    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py:
            return False
        if self.ranks[x] < self.ranks[y]:
            px, py = py, px
        self.parents[px] = py
        self.ranks[py] += 1
        self.size += 1
        return True


class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        alice = UFset(n)
        bob = UFset(n)
        res = 0
        for type_i, v, w in edges:
            v -= 1; w -= 1
            if type_i == 3:
                a = alice.union(v, w)
                b = bob.union(v, w)
                if not a and not b:
                    res += 1
        for type_i, v, w in edges:
            v -= 1; w -= 1
            if type_i == 1:
                if not alice.union(v, w):
                    res += 1
            elif type_i == 2:
                if not bob.union(v, w):
                    res += 1
            
        return res if alice.size == bob.size == n else -1
class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        
        
        root = [-1] * (n + 1)
        
        ans = 0
        
        def getroot(u):
            
            while root[u] >= 0:
                u = root[u]                
            return u
        
        def union(u, v):
            t = root[u] + root[v]
            
            if root[u] < root[v]:
                root[v] = u
                root[u] = t
            else:
                root[v] = t
                root[u] = v
                
        
        for t, u, v in edges:
            if t != 3:
                continue
                
            rootu = getroot(u)
            rootv = getroot(v)
            
            if rootu == rootv:
                ans += 1
            else:
                union(rootu, rootv)
                
        temp_root = list(root)
        
        for t, u, v in edges:
            if t != 1:
                continue
                
            rootu = getroot(u)
            rootv = getroot(v)
            
            if rootu == rootv:
                ans += 1
            else:
                union(rootu, rootv)
                
        if root.count(-1) > 1:
            return -1
                
        root = list(temp_root)
        
        for t, u, v in edges:
            if t != 2:
                continue
                
            rootu = getroot(u)
            rootv = getroot(v)
            
            if rootu == rootv:
                ans += 1
            else:
                union(rootu, rootv)      
                
        if root.count(-1) > 1:
            return -1
                
                
        return ans
        
            

class Solution:
    '''
    Intuition
    Add Type3 first, then check Type 1 and Type 2.


    Explanation
    Go through all edges of type 3 (Alice and Bob)
    If not necessary to add, increment res.
    Otherwith increment e1 and e2.

    Go through all edges of type 1 (Alice)
    If not necessary to add, increment res.
    Otherwith increment e1.

    Go through all edges of type 2 (Bob)
    If not necessary to add, increment res.
    Otherwith increment e2.

    If Alice's'graph is connected, e1 == n - 1 should valid.
    If Bob's graph is connected, e2 == n - 1 should valid.
    In this case we return res,
    otherwise return -1.


    Complexity
    Time O(E), if union find with compression and rank
    Space O(E)

    '''
    def maxNumEdgesToRemove(self, n, edges):
        # Union find
        def find(i):
            if i != root[i]:
                root[i] = find(root[i])
            return root[i]

        def uni(x, y):
            x, y = find(x), find(y)
            if x == y: return 0
            root[x] = y
            return 1

        res = e1 = e2 = 0

        # Alice and Bob
        root = list(range(n + 1))
        for t, i, j in edges:
            if t == 3:
                if uni(i, j):
                    e1 += 1
                    e2 += 1
                else:
                    res += 1
        root0 = root[:]

        # only Alice
        for t, i, j in edges:
            if t == 1:
                if uni(i, j):
                    e1 += 1
                else:
                    res += 1

        # only Bob
        root = root0
        for t, i, j in edges:
            if t == 2:
                if uni(i, j):
                    e2 += 1
                else:
                    res += 1

        return res if e1 == e2 == n - 1 else -1

class UnionFind:
    def __init__(self, n):
        self.root = [i for i in range(n + 1)]
        self.forests = n
        
    def unite(self, a, b):
        self.forests -= 1
        self.root[self.find(a)] = self.find(b)
        
    def find(self, a):
        if self.root[a] != a:
            self.root[a] = self.find(self.root[a])
        return self.root[a]
        
class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        ufA, ufB = UnionFind(n), UnionFind(n)
        edges.sort(key = lambda x : -x[0])
        res = 0
        for t, a, b in edges:
            if 3 == t:
                if ufA.find(a) == ufA.find(b) and ufB.find(a) == ufB.find(b):
                    res += 1
                if ufA.find(a) != ufA.find(b):
                    ufA.unite(a, b)
                if ufB.find(a) != ufB.find(b):
                    ufB.unite(a, b)
            elif t == 1:
                if ufA.find(a) == ufA.find(b):
                    # should remove this edge
                    res += 1
                else:
                    ufA.unite(a, b)
            else:
                if ufB.find(a) == ufB.find(b):
                    # should remove this edge
                    res += 1
                else:
                    ufB.unite(a, b)
        if ufA.forests > 1 or ufB.forests > 1:
            return -1
        return res
class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        
        parenta = [x for x in range(n+1)]
        parentb = [x for x in range(n+1)]
        
        def ufind(parent, x):
            if parent[x] != x:
                parent[x] = ufind(parent, parent[x])
            return parent[x]
        
        def uunion(parent, a, b):
            aa = ufind(parent, a)
            bb = ufind(parent, b)
            parent[aa] = bb
            
        edges.sort(key = lambda x: (-x[0]))
        
        count = 0
        for t, u, v in edges:
            if t == 3:
                if ufind(parenta, u) != ufind(parenta, v) or ufind(parentb, u) != ufind(parentb, v):
                    uunion(parenta, u, v)
                    uunion(parentb, u, v)
                else:
                    count += 1
            elif t == 2:
                if ufind(parentb, u) != ufind(parentb, v):
                    uunion(parentb, u, v)
                else:
                    count += 1
            elif t == 1:
                if ufind(parenta, u) != ufind(parenta, v):
                    uunion(parenta, u, v)
                else:
                    count += 1
        
        roota = ufind(parenta, 1)
        rootb = ufind(parentb, 1)
        for x in range(1, n+1):
            if ufind(parenta, x) != roota or ufind(parentb, x) != rootb:
                return -1
        
        return count

#         parenta = [x for x in range(n+1)]
#         parentb = [x for x in range(n+1)]
        
#         def ufind(parent, x):
#             if parent[x] != x:
#                 parent[x] = ufind(parent, parent[x])
#             return parent[x]
        
#         def uunion(parent, a, b):
#             ua = ufind(parent, a)
#             ub = ufind(parent, b)
            
#             parent[ua] = ub
            
#         edges.sort(key=lambda x: (-x[0]))
        
#         count = 0
#         for t, u, v in edges:
#             if t == 3:
#                 if ufind(parenta, u) != ufind(parenta, v) or ufind(parentb, u) != ufind(parentb, v):
#                     uunion(parenta, u, v)
#                     uunion(parentb, u, v)
#                 else:
#                     count += 1
#             elif t == 2:
#                 if ufind(parentb, u) != ufind(parentb, v):
#                     uunion(parentb, u, v)
#                 else:
#                     count += 1
#             else:
#                 if ufind(parenta, u) != ufind(parenta, v):
#                     uunion(parenta, u, v)
#                 else:
#                     count += 1
            
#         roota = ufind(parenta, 1)
#         rootb = ufind(parentb, 1)
#         for x in range(1, n+1):
#             if ufind(parenta, x) != roota or ufind(parentb, x) != rootb:
#                 return -1
            
#         return count


from collections import defaultdict
import copy
class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        
        def find(s, i):
            if s[i] != i:
                s[i] = find(s, s[i])
            return s[i]
        
        def union(s, i, j):
            if i > j:
                i, j = j, i
            s[find(s, j)] = s[find(s, i)]
        
        def is_connected(s, i, j):
            return find(s, i) == find(s, j)
        
        def is_full_connect(s):
            return all(is_connected(s, i, i+1) for i in range(len(s) - 1))

        
        d = defaultdict(set)
        res = 0
        uf = list([i for i in range(n)])
        for t, i, j in edges:
            d[t].add((i-1, j-1))
        for i, j in d[3]:
            if is_connected(uf, i, j):
                res += 1
            else:
                union(uf, i, j)
        uf1, uf2 = copy.copy(uf), copy.copy(uf)

        for i, j in d[1]:
            if is_connected(uf1, i, j):
                res += 1
            else:
                union(uf1, i, j)

        for i, j in d[2]:
            if is_connected(uf2, i, j):
                res += 1
            else:
                union(uf2, i, j)

        if not is_full_connect(uf1) or not is_full_connect(uf2):
            return -1

        return res

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [1] * n

    def find(self, i):
        if self.parent[i] != i:
            self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, x, y):
        px = self.find(x)
        py = self.find(y)
        if px == py:
            return px
        if self.rank[px] > self.rank[py]:
            px, py = py, px
        self.parent[px] = py
        if self.rank[px] == self.rank[py]:
            self.rank[py] += 1


class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        both = [(a - 1, b - 1) for (t, a, b) in edges if t == 3]
        alice = [(a - 1, b - 1) for (t, a, b) in edges if t == 1]
        bob = [(a - 1, b - 1) for (t, a, b) in edges if t == 2]

        uf_alice = UnionFind(n)
        uf_bob = UnionFind(n)
        count = 0
        for a, b in both:
            if uf_alice.find(a) != uf_alice.find(b):
                uf_alice.union(a, b)
                uf_bob.union(a, b)
                count += 1

        for a, b in alice:
            if uf_alice.find(a) != uf_alice.find(b):
                uf_alice.union(a, b)
                count += 1

        for a, b in bob:
            if uf_bob.find(a) != uf_bob.find(b):
                uf_bob.union(a, b)
                count += 1

        p_alice = set([uf_alice.find(i) for i in range(n)])
        if len(p_alice) > 1:
            return -1

        p_bob = set([uf_bob.find(i) for i in range(n)])
        if len(p_bob) > 1:
            return -1

        return len(edges) - count

class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        
        type1 = [[i-1, j-1] for t, i, j in edges if t == 1]
        type2 = [[i-1, j-1] for t, i, j in edges if t == 2]
        type3 = [[i-1, j-1] for t, i, j in edges if t == 3]

        def helper(type1, other, fa, c=False):
            def getfa(i):
                if fa[i] != i:
                    fa[i] = getfa(fa[i])
                return fa[i]
            connect_count = 0 
            ret = len(other)
            for i, j in type1:
                if getfa(i) != getfa(j):
                    fa[getfa(i)] = getfa(j)
                    connect_count += 1
            for i, j in other:
                if getfa(i) != getfa(j):
                    fa[getfa(i)] = getfa(j)
                    connect_count += 1
                    ret -= 1
            # print(fa, connect_count, ret)
            if c == True or connect_count == len(fa) - 1:
                return ret
            return -1
        
        
        t1_count = helper(type3, type1, list(range(n)))
        if t1_count < 0:
            return -1
        t2_count = helper(type3, type2, list(range(n)))
        if t2_count < 0:
            return -1
        t3_count = helper([], type3, {i:i for i in set([x for x, y in type3] + [y for x, y in type3])}, True)
        return t1_count + t2_count + t3_count
            
        

class Solution:
    '''
    Intuition
    Add Type3 first, then check Type 1 and Type 2.

    Explanation
    Go through all edges of type 3 (Alice and Bob)
    If not necessary to add, increment res.
    Otherwith increment e1 and e2.

    Go through all edges of type 1 (Alice)
    If not necessary to add, increment res.
    Otherwith increment e1.

    Go through all edges of type 2 (Bob)
    If not necessary to add, increment res.
    Otherwith increment e2.

    If Alice's'graph is connected, e1 == n - 1 should valid.
    If Bob's graph is connected, e2 == n - 1 should valid.
    In this case we return res,
    otherwise return -1.

    Complexity
    Time O(E), if union find with compression and rank
    Space O(E)

    '''
    def maxNumEdgesToRemove(self, n, edges):
        # Union find
        def find(i):
            if i != root[i]:
                root[i] = find(root[i])
            return root[i]

        def uni(x, y):
            x, y = find(x), find(y)
            if x == y: return 0
            root[x] = y
            return 1

        res = e1 = e2 = 0

        # Alice and Bob
        root = list(range(n + 1))
        for t, i, j in edges:
            if t == 3:
                if uni(i, j):
                    e1 += 1
                    e2 += 1
                else:
                    res += 1
        root0 = root[:]

        # only Alice
        for t, i, j in edges:
            if t == 1:
                if uni(i, j):
                    e1 += 1
                else:
                    res += 1

        # only Bob
        root = root0
        for t, i, j in edges:
            if t == 2:
                if uni(i, j):
                    e2 += 1
                else:
                    res += 1

        return res if e1 == e2 == n - 1 else -1

class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        
        graph ={}
        for Type,start,end in edges:
            if start not in graph:
                graph[start]=[]
            if end not in graph:
                graph[end]=[]
                
            graph[start].append([Type,end])
            graph[end].append([Type,start])
        
        
        def dfs(node,p):
            for Type,neighbour in graph[node]:
                if Type in p:
                    if neighbour not in visited:
                        visited[neighbour]=True
                        dfs(neighbour,p)
                        
            
                    
        
        #check if connected
        visited ={}
        #dfs on 1,3,
        dfs(1,[1,3])
        if len(list(visited.keys()))!=n:
            return -1
    
        visited ={}
        dfs(1,[2,3])
        if len(list(visited.keys()))!=n:
            return -1
        #dfs on 2,3
        
        
        #find number of blue components:
        visited ={}
        #dfs on blue edges, 
        blue_cc=0
        blue_edges =0
        for Type,start,end in edges:
            if Type==3:
                
                if start not in visited:
                    temp = len(list(visited.keys()))
                    blue_cc+=1
                    dfs(start,[3])
                    blue_edges+=len(list(visited.keys()))-temp-1
                
                if end not in visited:
                    temp = len(list(visited.keys()))
                    blue_cc+=1
                    dfs(end,[3])
                    blue_edges+=len(list(visited.keys()))-temp-1
                
                
            
                    
        unvisited = len(list(graph.keys()))-len(list(visited.keys()))
                
            
        #keep track of # of visitedNode -1
        #number of times dfs is called
        
        #calc ans = totalnumber of edges - (blue edges + 2*unvisited nodes + 2*(blue components-1))
        print((blue_edges,unvisited,blue_cc))
        return len(edges)-(blue_edges+ 2*unvisited +2*(blue_cc-1))
        
        
        
        

class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        '''always keep type 3 if possible'''
        edges123 = [[]  for _ in range(3) ] 
        for t, a, b in edges: edges123[t-1].append( (a-1,b-1) )
        # type 0 1 2 for alice, bob and both
        self.res = 0
        Parents = [[i for i in range(n)] for _ in range(2) ]
        depth = [[1]*n for _ in range(2)]
        selectedEdges = [0,0]
        def FindRoot(n,t):
            #print('node',n,'type',t)
            if Parents[t][n] != n:
                Parents[t][n] = FindRoot(Parents[t][n] ,t)
            return Parents[t][n] 
        def Uni(x,y,t):
            rx, ry = FindRoot(x,t), FindRoot(y,t)
            if rx == ry: return 0
            else:
                if depth[t][rx] >= depth[t][ry]:
                    Parents[t][ry] = rx
                    depth[t][rx] = max(depth[t][rx],depth[t][ry])
                else:
                    Parents[t][rx] = ry
                return 1
            
        def connect(thetype):
            mytypes = [thetype] if thetype < 2 else [ 0, 1 ]
            for x, y in edges123[thetype]:
                if all(Uni(x,y,t) for t in mytypes):
                    for t in mytypes: selectedEdges[t] += 1
                else:
                    self.res += 1
            # for t in mytypes: 
            #     root = [FindRoot(i,t) for i in range(n)]
            #     print(thetype,t, 'parents',Parents[t],root,selectedEdges,self.res)
                
        connect(2)
        connect(0)
        connect(1)
        return self.res if all(selectedEdges[t]==n-1 for t in [0,1]) else -1

class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        def find(p, u):
            parent = p[u]
            if parent == u:
                return parent
            
            p[u] = find(p, parent)
            return p[u]
        
        def union(p, rank, root_u, root_v):
            if rank[root_u] < rank[root_v]:
                p[root_u] = root_v
            elif rank[root_v] < rank[root_u]:
                p[root_v] = root_u
            else:
                p[root_u] = root_v
                rank[root_v] += 1
        
        p = list(range(n))
        rank = [0] * n
        
        full_edges = set()
        partial_edges = set()
        partial_adj = {}
        partial_adj[1] = collections.defaultdict(set)
        partial_adj[2] = collections.defaultdict(set)
        for e in edges:
            edge_type, u, v = e[0], e[1] - 1, e[2] - 1
            if edge_type == 3:
                full_edges.add((u, v))
            else:
                partial_edges.add((edge_type, u, v))
                partial_adj[edge_type][u].add(v)
                partial_adj[edge_type][v].add(u)

        nb_edges_in_mst = 0
        for e in full_edges:
            u, v = e
            root_u, root_v = find(p, u), find(p, v)
            if root_u != root_v:
                union(p, rank, root_u, root_v)
                nb_edges_in_mst += 1
        
        for e in partial_edges:
            edge_type, u, v = e
            root_u, root_v = find(p, u), find(p, v)
            if root_u == root_v:
                continue

            # We have two nodes u and v such that they fall into two disjoint
            # connected sub-graphs, and u from subgraph A is connected to v
            # in subgraph B with either edge_type == 1 or edge_type 2. Since we
            # need to reach v from subgraph A by both Alice and Bob, if we can
            # find another node, x, in subgraph A that is connected to v in subgraph B
            # by the other edge_type, then we can reach v from any node in subgraph A.
            needed_edge_type = 2 if edge_type == 1 else 2
            foo = (v, root_u)
            found_needed_edge = False
            for x in partial_adj[needed_edge_type][foo[0]]:
                root_x = find(p, x)
                if root_x == foo[1]:
                    # x is in in subgraph A, same as u, AND it's connected to v via the
                    # needed_edge_type
                    union(p, rank, root_x, foo[1])
                    union(p, rank, root_u, root_v)
                    nb_edges_in_mst += 2
                    found_needed_edge = True
                    break
            if found_needed_edge:
                continue
            
            foo = (u, root_v)
            for x in partial_adj[needed_edge_type][foo[0]]:
                root_x = find(p, x)
                if root_x == foo[1]:
                    # y is in the subgraph B, same as v, and it's connected to u via the
                    # needed_edge_type
                    union(p, rank, root_x, foo[1])
                    union(p, rank, root_u, root_v)
                    nb_edges_in_mst += 2
                    break

        uniq_roots = set()
        for u in range(len(p)):
            uniq_roots.add(find(p, u))
        if len(uniq_roots) != 1:
            return -1  
        
        return len(edges) - nb_edges_in_mst
            
                
            
        

class UnionFind:
    """A minimalist standalone union-find implementation."""
    def __init__(self, n):
        self.count = n               # number of disjoint sets 
        self.parent = list(range(n)) # parent of given nodes
        self.rank = [1]*n            # rank (aka size) of sub-tree 
        
    def find(self, p):
        """Find with path compression."""
        if p != self.parent[p]: 
            self.parent[p] = self.find(self.parent[p]) # path compression 
        return self.parent[p]
    
    def union(self, p, q):
        """Union with ranking."""
        prt, qrt = self.find(p), self.find(q)
        if prt == qrt: return False
        self.count -= 1 # one more connection => one less disjoint 
        if self.rank[prt] > self.rank[qrt]: 
            prt, qrt = qrt, prt # add small sub-tree to large sub-tree for balancing
        self.parent[prt] = qrt
        self.rank[qrt] += self.rank[prt] # ranking 
        return True
    
        
class Solution:
    def maxNumEdgesToRemove(self, n: int, edges) -> int:
        ufa = UnionFind(n) # for Alice
        ufb = UnionFind(n) # for Bob
        ans = 0
        edges.sort(reverse=True) 
        for t, u, v in edges: 
            u, v = u-1, v-1
            if t == 3: 
                ans += not (ufa.union(u, v) and ufb.union(u, v)) # Alice & Bob
            elif t == 2: 
                ans += not ufb.union(u, v)                     # Bob only
            else: 
                ans += not ufa.union(u, v)                            # Alice only
        return ans if ufa.count == 1 and ufb.count == 1 else -1 # check if uf is connected 
class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        p = [-1 for i in range(n+1)]
        c = [1 for i in range(n+1)]

        def find(i, p, c):
            pi = p[i]
            nc = []
            nodes = []
            total = 0
            while pi != -1:
                nodes.append(i)
                nc.append(total)
                total = c[i]
                i = pi
                pi = p[i]
            for k, vi in enumerate(nodes):
                p[vi] = i
                c[vi] -= nc[k]
            return i

        def union(i, j, p, c):
            si, sj = find(i, p, c), find(j, p, c)
            if si == sj:
                return si
            if c[si] > c[sj]:
                p[sj] = si
                c[si] += c[sj]
                return si
            else:
                p[si] = sj
                c[sj] += c[si]
                return sj
            
        # connected component
        e1s = []
        e2s = []
        s = -1
        for i, ed in enumerate(edges):
            e, u, v = ed
            if e == 1:
                e1s.append(i)
            elif e == 2:
                e2s.append(i)
            else:
                ns = union(u, v, p, c)
                if s == -1 or c[s] < c[ns]:
                    s = ns
        pvst = set()
        num_edges = 0
        for i in range(1, n+1):
            si = find(i, p, c)
            if si in pvst:
                continue
            pvst.add(si)
            num_edges += c[si]-1
        
        def check(es):
            np = p.copy()
            nc = c.copy()
            for i in es:
                _, u, v = edges[i]
                union(u, v, np, nc)
            pset = {find(i, np, nc) for i in range(1, n+1)}
            return len(pset) == 1
        
        if not check(e1s) or not check(e2s):
            return -1
        need = 2*(n-1) - num_edges
        return len(edges) - need
class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        for edge in edges:
            edge[1] -= 1
            edge[2] -= 1
        
        blueUset = self.Uset(n)
        blueUsed = self.countUsed(n, [(edge[1], edge[2]) for edge in edges if edge[0] == 3], blueUset)
        
        for i in range(0, n):
            blueUset.find(i)
        
        redUset = self.Uset(n)
        redUset.parents = blueUset.parents[:]
        redUset.gCounts = blueUset.gCounts
        redUsed = self.countUsed(n, [(edge[1], edge[2]) for edge in edges if edge[0] == 1], redUset)
        if redUset.gCounts > 1:
            return -1
        
        greenUset = self.Uset(n)
        greenUset.parents = blueUset.parents[:]
        greenUset.gCounts = blueUset.gCounts
        greenUsed = self.countUsed(n, [(edge[1], edge[2]) for edge in edges if edge[0] == 2], greenUset)
        if greenUset.gCounts > 1:
            return -1
        
        return len(edges) - len(blueUsed) - len(redUsed) - len(greenUsed)
        
        
    def countUsed(self, n, edges, uset):
        usedEdges = []
        for edge in edges:
            u = edge[0]
            v = edge[1]
            if uset.find(u) != uset.find(v):
                usedEdges.append(edge)
                uset.union(u, v)
            
        return usedEdges
    
    class Uset:
        def __init__(self, n):
            self.parents = [i for i in range(0, n)]
            self.gCounts = n
        
        def find(self, x):
            if self.parents[x] != x:
                self.parents[x] = self.find(self.parents[x])
            return self.parents[x]
            
        def union(self, x, y):
            px = self.find(x)
            py = self.find(y)
            if px != py:
                self.parents[px] = py
                self.gCounts -= 1
                
            return

class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        # think it in another way as to construct minimum spanning trees for Alice and Bob
        # if there are then all the rest of the edges can be removed
        # prioritize type 3 edges
        edges.sort(key=lambda edge: -edge[0])
        ds_alice = DisjointSets(n)
        ds_bob = DisjointSets(n)
        edges_added = 0
        for e_type, u, v in edges:
            if e_type == 3:
                edges_added += int(ds_alice.union(u - 1, v - 1) | ds_bob.union(u - 1, v - 1))
            elif e_type == 2:
                edges_added += int(ds_bob.union(u - 1, v - 1))
            else:
                edges_added += int(ds_alice.union(u - 1, v - 1))
        return len(edges) - edges_added if ds_alice.isConnected() and ds_bob.isConnected() else -1
        
        
class DisjointSets:
    def __init__(self, n: int) -> None:
        self.parent = [x for x in range(n)]
        self.set_size = n
        
    def union(self, x: int, y: int) -> bool:
        x = self.find(x)
        y = self.find(y)
        if x != y:
            self.parent[x] = y
            self.set_size -= 1
            return True
        return False
    
    def find(self, x: int) -> int:
        if x != self.parent[x]:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def isConnected(self) -> bool:
        return self.set_size == 1
class UnionFind:
    def __init__(self, items):
        self.leader = {}
        for item in items:
            self.leader[item] = item
    
    def union(self, i1, i2):
        l1, l2 = self.find(i1), self.find(i2)
        if l1 == l2:
            return False
        self.leader[l1] = self.leader[l2]
        return True
        
    def find(self, i):
        if self.leader[i] != i:
            self.leader[i] = self.find(self.leader[i])
        return self.leader[i]
    
class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        """
        Check if A and B can traverse the entire graph at first
        """
        items = [i for i in range(1, n + 1)]
        uf1 = UnionFind(items)
        uf2 = UnionFind(items)
        
        a_essential_edges = 0
        b_essential_edges = 0
        removable_edges = 0
        
        for [t, u, v] in edges:
            if t == 3:
                union_success = uf1.union(u, v)
                uf2.union(u, v)
                
                if union_success:
                    a_essential_edges += 1
                    b_essential_edges += 1
                else:
                    removable_edges += 1
                    
        for [t, u, v] in edges:
            if t == 1:
                union_success = uf1.union(u, v)
                if union_success:
                    a_essential_edges += 1
                else:
                    removable_edges += 1
        for [t, u, v] in edges:
            if t == 2:
                union_success = uf2.union(u, v)
                if union_success:
                    b_essential_edges += 1
                else:
                    removable_edges += 1
        
        
        
        return removable_edges if a_essential_edges == b_essential_edges == n - 1 else -1
        
class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        def find(p, u):
            parent = p[u]
            if parent == u:
                return parent
            
            p[u] = find(p, parent)
            return p[u]
        
        def union(p, rank, root_u, root_v):
            if rank[root_u] < rank[root_v]:
                p[root_u] = root_v
            elif rank[root_v] < rank[root_u]:
                p[root_v] = root_u
            else:
                p[root_u] = root_v
                rank[root_v] += 1
        
        p = list(range(n))
        rank = [0] * n
        
        full_edges = set()
        partial_edges = set()
        partial_adj = {}
        partial_adj[1] = collections.defaultdict(set)
        partial_adj[2] = collections.defaultdict(set)
        for e in edges:
            edge_type, u, v = e[0], e[1] - 1, e[2] - 1
            if edge_type == 3:
                full_edges.add((u, v))
            else:
                partial_edges.add((edge_type, u, v))
                partial_adj[edge_type][u].add(v)
                partial_adj[edge_type][v].add(u)

        nb_edges_in_mst = 0
        for e in full_edges:
            u, v = e
            root_u, root_v = find(p, u), find(p, v)
            if root_u != root_v:
                union(p, rank, root_u, root_v)
                nb_edges_in_mst += 1
        
        for e in partial_edges:
            edge_type, u, v = e
            root_u, root_v = find(p, u), find(p, v)
            if root_u == root_v:
                continue

            # We have two nodes u and v such that they fall into two disjoint
            # connected sub-graphs, and u from subgraph A is connected to v
            # in subgraph B with either edge_type == 1 or edge_type 2. Since we
            # need to reach v from subgraph A by both Alice and Bob, if we can
            # find another node, x, in subgraph A that is connected to v in subgraph B
            # by the other edge_type, then we can reach v from any node in subgraph A.
            needed_edge_type = 2 if edge_type == 1 else 2
            found_needed_edge = False
            for x in partial_adj[needed_edge_type][v]:
                root_x = find(p, x)
                if root_x == root_u:
                    # x is in in subgraph A, same as u, AND it's connected to v via the
                    # needed_edge_type
                    union(p, rank, root_x, root_u)
                    union(p, rank, root_u, root_v)
                    nb_edges_in_mst += 2
                    found_needed_edge = True
                    break
            if found_needed_edge:
                continue
                
            for y in partial_adj[needed_edge_type][u]:
                root_y = find(p, y)
                if root_y == root_v:
                    # y is in the subgraph B, same as v, and it's connected to u via the
                    # needed_edge_type
                    union(p, rank, root_y, root_u)
                    union(p, rank, root_u, root_v)
                    nb_edges_in_mst += 2
                    break
        
        uniq_roots = set()
        for u in range(len(p)):
            uniq_roots.add(find(p, u))
        if len(uniq_roots) != 1:
            return -1  
        
        return len(edges) - nb_edges_in_mst
            
                
            
        

class DSU:
    def __init__(self, n):
        self.parent = [x for x in range(n)]
        self.rank = [0] * n
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py:
            return False
        else:
            if self.rank[px] > self.rank[py]:
                self.parent[py] = px
            elif self.rank[py] > self.rank[px]:
                self.parent[px] = py
            else:
                self.parent[px] = py
                self.rank[px] += 1
            return True
        

class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        dsu1, dsu2 = DSU(n), DSU(n)
        ans = 0
        for t, u, v in edges:
            if t == 3:
                if not dsu1.union(u - 1, v - 1) or not dsu2.union(u - 1, v - 1):
                    ans += 1
        
        for t, u, v in edges:
            if t == 1 and not dsu1.union(u - 1, v - 1):
                ans += 1
            if t == 2 and not dsu2.union(u - 1, v - 1):
                ans += 1
        
        p1, p2 = dsu1.find(0), dsu2.find(0)
        for i in range(n):
            if p1 != dsu1.find(i) or p2 != dsu2.find(i):
                return -1
        return ans
from typing import List
class Solution:
    def find(self,i):
        if i != self.root[i]:
            self.root[i] = self.find(self.root[i])
        return self.root[i]

    def uni(self,x, y):
        x, y = self.find(x), self.find(y)
        if x == y: return 0
        self.root[x] = y
        return 1
    def maxNumEdgesToRemove(self, n, edges):
        # Union find

        res = e1 = e2 = 0

        # Alice and Bob
        self.root = list(range(n + 1))
        for t, i, j in edges:
            if t == 3:
                if self.uni(i, j):
                    e1 += 1
                    e2 += 1
                else:
                    res += 1
        root0 = self.root[:]

        # only Alice
        for t, i, j in edges:
            if t == 1:
                if self.uni(i, j):
                    e1 += 1
                else:
                    res += 1

        # only Bob
        self.root = root0
        for t, i, j in edges:
            if t == 2:
                if self.uni(i, j):
                    e2 += 1
                else:
                    res += 1

        return res if e1 == e2 == n - 1 else -1
class UF:
    def __init__(self):
        self.d = defaultdict(int)
        
    def findRoot(self, key):
        if self.d[key] > 0:
            self.d[key] = self.findRoot(self.d[key])
            return self.d[key]
        else:
            return key
        
    def mergeRoot(self, k1, k2):
        r1, r2 = self.findRoot(k1), self.findRoot(k2)  
        if r1 != r2:
            r1, r2 = min(r1, r2), max(r1, r2)
            self.d[r1] += self.d[r2]
            self.d[r2] = r1
        return r1
    

import heapq

class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        t12, t3 = [], []
        for e in edges:
            if e[0] == 3:
                t3.append(e)
            else:
                t12.append(e)
                
        uf1 = UF()
        uf2 = UF()
        
        ttl = 0
        for e in t3:
            if uf1.findRoot(e[1]) != uf1.findRoot(e[2]) or uf2.findRoot(e[1]) != uf2.findRoot(e[2]):
                uf1.d[uf1.mergeRoot(e[1], e[2])] -= 1
                uf2.d[uf2.mergeRoot(e[1], e[2])] -= 1
            else:
                ttl += 1   
                    
        for e in t12:
            if e[0] == 1 and uf1.findRoot(e[1]) != uf1.findRoot(e[2]):
                uf1.d[uf1.mergeRoot(e[1], e[2])] -= 1
            elif e[0] == 2 and uf2.findRoot(e[1]) != uf2.findRoot(e[2]):
                uf2.d[uf2.mergeRoot(e[1], e[2])] -= 1
            else:
                ttl += 1
                 
        if uf1.d[1] != - n + 1 or uf2.d[1] != - n + 1:
            return -1
        
        return ttl
class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        
        class Union:
            
            def __init__(self):
                
                self.parent = -1
                self.rank = 1
                
            def Find(self):
                
                if self.parent == -1: return self
                return self.parent.Find()
            
            def UNION(self, node):
                
                L, R = self.Find(), node.Find()
                if L == R: return 0
                elif L.rank > R.rank: R.parent = L
                elif R.rank > L.rank: L.parent = R
                else:
                    L.parent = R
                    R.rank += 1
                    
                return 1
        
    
        alice, bob = [], []
    
        for t, u, v in edges:
            if t == 3:
                alice.append([t, u, v])
                bob.append([t, u, v])
                
        for t, u, v in edges:
            if t == 1:
                alice.append([t, u, v])
            elif t == 2:
                bob.append([t, u, v])
            
                
        Vertex = {}
        for i in range(n): Vertex[i + 1] = Union()
            
        Count, Common = 0, 0
        
        for t, u, v in alice:
            
            if Vertex[u].UNION(Vertex[v]) == 1: 
                Count += 1
                if t == 3: Common += 1
    
        if Count < n - 1: return -1
        
        for u in Vertex: Vertex[u].parent = -1
        
        Count = 0
        
        for t, u, v in bob:
            
            if Vertex[u].UNION(Vertex[v]) == 1: 
                Count += 1
                
        if Count < n - 1: return -1
        
        return len(edges) - Common - 2*(n - 1 - Common)
        
            
        
        
        
        
        
        
        
                
                
                
                

                

class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        elements = [-1] * (n + 1)
        elements[0] = 0
        elements2 = None
        
        def find(elements, i):
            while elements[i] >= 0:
                i = elements[i]
            return i
        
        def union(elements, i, j):
            i = find(elements, i)
            j = find(elements, j)
            if i == j:
                return False
            if elements[i] <= elements[j]:
                if elements[i] == elements[j]:
                    elements[i] -= 1
                elements[j] = i
            else:
                elements[i] = j
            return True
        
        def count(elements):
            return sum(1 for i in elements if i < 0)
        edges.sort(key=lambda k: k[0])
        result = 0
        for t, u, v in reversed(edges):
            if t == 3:
                if not union(elements, u, v):
                    result += 1
            else:
                if elements2 is None:
                    elements2 = elements[:]
                if t == 2:
                    if not union(elements2, u, v):
                        result += 1
                elif t == 1:
                    if not union(elements, u, v):
                        result += 1
        if count(elements) > 1 or (elements2 is not None and count(elements2)) > 1:
            return -1
        return result
# from collections import defaultdict

# class Solution:
#     def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
#         aj = [defaultdict(set) for i in range(4)]
#         total = len(edges)
#         for t, i, j in edges:
#             if i == j:
#                 continue
#             aj[t][i].add(j)
#             aj[t][j].add(i)
        
#         used = set()
#         visited = {1}
#         t3 = [(1, i) for i in aj[3][1]]
#         t1 = [(1, i) for i in aj[1][1]]
#         while len(visited) < n and (t3 or t1):
#             if t3:
#                 i, j = t3.pop()
#                 reusable = True
#             else:
#                 i, j = t1.pop()
#                 reusable = False
#             if j in visited:
#                 continue
                
#             if reusable:
#                 used.add((min(i, j), max(i, j)))
#             visited.add(j)
#             for k in aj[3][j]:
#                 if k not in visited:
#                     t3.append((j, k))
#             for k in aj[1][j]:
#                 if k not in visited:
#                     t1.append((j, k))
#         if len(visited) < n:
#             return -1
            
#         reused = set()
#         visited = {1}
#         t0 = []
#         t2 = [(1, i) for i in aj[2][1]]
#         for i in aj[3][1]:
#             if (1, i) in used:
#                 t0.append((1, i))
#             else:
#                 t2.append((1, i))
#         while len(visited) < n and (t0 or t2):
#             if t0:
#                 i, j = t0.pop()
#                 reusable = True
#             else:
#                 i, j = t2.pop()
#                 reusable = False
#             if j in visited:
#                 continue
                
#             if reusable:
#                 reused.add((min(i, j), max(i, j)))
#             visited.add(j)
#             for k in aj[3][j]:
#                 if k not in visited:
#                     if (min(j, k), max(j, k)) in used:
#                         t0.append((j, k))
#                     else:
#                         t2.append((j, k))
#             for k in aj[2][j]:
#                 if k not in visited:
#                     t2.append((j, k))
#         if len(visited) < n:
#             return -1

#         return total - ((n - 1) * 2 - len(reused))


from collections import defaultdict
from heapq import heappush, heappop

class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        aj = [defaultdict(set) for i in range(4)]
        total = len(edges)
        for t, i, j in edges:
            if i == j:
                continue
            aj[t][i].add(j)
            aj[t][j].add(i)
        
        reuse = set()
        count = 0
        
        visited = {1}
        heap = []
        for i in aj[3][1]:
            heappush(heap, (1, 1, i))
        for i in aj[1][1]:
            heappush(heap, (2, 1, i))
        while len(visited) < n and heap:
            w, i, j = heappop(heap)
            if j in visited:
                continue
                
            if w == 1:
                reuse.add((i, j))
            count += 1
            visited.add(j)
            for k in aj[3][j]:
                if k not in visited:
                    heappush(heap, (1, j, k))
            for k in aj[1][j]:
                if k not in visited:
                    heappush(heap, (2, j, k))
        if len(visited) < n:
            return -1
            
        visited = {1}
        heap = []
        for i in aj[3][1]:
            if (1, i) in reuse or (i, 1) in reuse:
                heappush(heap, (0, 1, i))
            else:
                heappush(heap, (1, 1, i))
        for i in aj[2][1]:
            heappush(heap, (2, 1, i))
        while len(visited) < n and heap:
            w, i, j = heappop(heap)
            if j in visited:
                continue
                
            if w > 0:
                count += 1
            visited.add(j)
            for k in aj[3][j]:
                if k not in visited:
                    if (j, k) in reuse or (k, j) in reuse:
                        heappush(heap, (0, j, k))
                    else:
                        heappush(heap, (1, j, k))
            for k in aj[2][j]:
                if k not in visited:
                    heappush(heap, (2, j, k))
        if len(visited) < n:
            return -1

        return total - count

class Solution:
    
    def find(self, c, parents):
        s = c
        while parents[c-1] != -1:
            c = parents[c-1]
        if s != c:
            parents[s-1] = c
        return c
    
    def delete_cycles(self, edges, parents, delete, t):
        for edge in edges:
            c1 = edge[0]
            c2 = edge[1]
            p1 = self.find(c1, parents)
            p2 = self.find(c2, parents)
            # print(f'edge:    {edge}')
            # print(f'p1  p2:  {p1} {p2}')
            # print(f'type:    {t}')
            # print(f'parents: {parents}')
            # print()
            if p1 == p2:
                delete.add((t, c1, c2))
            else:
                parents[p1-1] = p2
        
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        common = [(e[1], e[2]) for e in edges if e[0] == 3]
        type1 = [(e[1], e[2]) for e in edges if e[0] == 1]
        type2 = [(e[1], e[2]) for e in edges if e[0] == 2]
        
        delete = set()
        
        parents1 = [-1] * n
        parents2 = [-1] * n
        
        self.delete_cycles(common, parents1, delete, 3)
        self.delete_cycles(type1, parents1, delete, 1)
        # print("-----------------------------------------------")
        self.delete_cycles(common, parents2, delete, 3)
        self.delete_cycles(type2, parents2, delete, 2)
        
        has_single_parent1 = False
        for p in parents1:
            if p == -1:
                if not has_single_parent1:
                    has_single_parent1 = True
                else:
                    return -1
                
        has_single_parent2 = False
        for p in parents2:
            if p == -1:
                if not has_single_parent2:
                    has_single_parent2 = True
                else:
                    return -1
                
        return len(delete)
                
                
        
        
                

class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        '''always keep type 3 if possible'''
        edges123 = [collections.defaultdict(list)  for _ in range(3) ] 
        for t, a, b in edges: edges123[t-1][a-1].append(b-1)
        # type 0 1 2 for alice, bob and both
        self.res = 0
        Parents = [[i for i in range(n)] for _ in range(2) ]
        selectedEdges = [0,0]
        def FindRoot(n,t):
            #print('node',n,'type',t)
            if Parents[t][n] != n:
                Parents[t][n] = FindRoot(Parents[t][n] ,t)
            return Parents[t][n] 
        def Uni(x,y,t):
            x, y = FindRoot(x,t), FindRoot(y,t)
            if x == y: return 0
            else:
                Parents[t][x] = y
                return 1
            
        def connect(thetype):
            mytypes = [thetype] if thetype < 2 else [ 0, 1 ]
            for node in range(n):
                for neighbor in edges123[thetype][node]:
                    if all(Uni(neighbor,node,t) for t in mytypes):
                        for t in mytypes: selectedEdges[t] += 1
                    else:
                        self.res += 1
            # for t in mytypes: 
            #     root = [FindRoot(i,t) for i in range(n)]
            #     print(thetype,t, 'parents',Parents[t],root,selectedEdges,self.res)
                
        connect(2)
        connect(0)
        connect(1)
        return self.res if all(selectedEdges[t]==n-1 for t in [0,1]) else -1

class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        parent = [[i for i in range(n+1)] for _ in range(4)]
        size = [[1 for i in range(n+1)] for _ in range(4)]
        
        def find(x, t):
            if parent[t][x] != x:
                parent[t][x] = find(parent[t][x], t)
            return parent[t][x]
        
        def union(x, y, t):
            xs, ys = find(x, t), find(y, t)
            if xs == ys: return False
            if size[t][xs] < size[t][ys]:
                xs, ys = ys, xs
            size[t][xs] += size[t][ys]
            parent[t][ys] = xs
            return True
        
        ans = 0
        for t, u, v in edges:
            if t != 3: continue
            union(u, v, 1)
            union(u, v, 2)
            if not union(u, v, 3): ans += 1
                
        for t, u, v in edges:
            if t != 1: continue
            if not union(u, v, 1): ans += 1
        
        for t, u, v in edges:
            if t != 2: continue
            if not union(u, v, 2): ans += 1
                
        for i in range(1, n+1):
            for t in [1, 2]:
                if size[t][find(i, t)] != n:
                    return -1
        
        return ans
        
        
        
        
        
        
        
        

class UF:
    def __init__(self):
        self.d = defaultdict(int)
        
    def findRoot(self, key):
        if self.d[key] > 0:
            self.d[key] = self.findRoot(self.d[key])
            return self.d[key]
        else:
            return key
        
    def mergeRoot(self, k1, k2):
        r1, r2 = self.findRoot(k1), self.findRoot(k2)  
        if r1 != r2:
            r1, r2 = min(r1, r2), max(r1, r2)
            self.d[r1] += self.d[r2]
            self.d[r2] = r1
        return r1
    

import heapq

class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        t12, t3 = [], []
        for e in edges:
            if e[0] == 3:
                t3.append(e)
            else:
                t12.append(e)
                
        uf1 = UF()
        uf2 = UF()
        uf1.d[1] = -1
        uf2.d[1] = -1
        ttl = 0
        for e in t3:
            if uf1.findRoot(e[1]) != uf1.findRoot(e[2]) or uf2.findRoot(e[1]) != uf2.findRoot(e[2]):
                uf1.d[uf1.mergeRoot(e[1], e[2])] -= 1
                uf2.d[uf2.mergeRoot(e[1], e[2])] -= 1
                ttl += 1   
                    
        for e in t12:
            if e[0] == 1 and uf1.findRoot(e[1]) != uf1.findRoot(e[2]):
                uf1.d[uf1.mergeRoot(e[1], e[2])] -= 1
            elif e[0] == 2 and uf2.findRoot(e[1]) != uf2.findRoot(e[2]):
                uf2.d[uf2.mergeRoot(e[1], e[2])] -= 1
                 
        if uf1.d[1] != - n or uf2.d[1] != - n:
            return -1
        
        return len(edges) - 2 * n + 2 + ttl
class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        
        edges = sorted(edges, reverse=True)
        alice, bob = [i for i in range(n + 1)], [i for i in range(n + 1)]
        size_alice, size_bob = [1 for i in range(n + 1)], [1 for i in range(n + 1)]
        ans = 0
        
        def find(u, state):
            if state == "a":
                if alice[u] == u:
                    return u
                return find(alice[u], state)
            else:
                if bob[u] == u:
                    return u
                return find(bob[u], state)
            
        
        def union(u, v, state):
            nonlocal ans
            
            add = 0
            if state == 3 or state == 1:
                p1, p2 = find(u, "a"), find(v, "a")
                if p1 != p2:
                    add = 1
                    print("haha", u, v)
                    if size_alice[p1] >= size_alice[p2]:
                        size_alice[p1] += size_alice[p2]
                        alice[p2] = p1
                    else:
                        size_alice[p2] += size_alice[p1]
                        alice[p1] = p2

            if state == 3 or state == 2:
                p1, p2 = find(u, "b"), find(v, "b")
                if p1 != p2:
                    add = 1
                    print("haha", u, v)
                    if size_bob[p1] >= size_bob[p2]:
                        size_bob[p1] += size_bob[p2]
                        bob[p2] = p1
                    else:
                        size_bob[p2] += size_bob[p1]
                        bob[p1] = p2
            ans += add
                
        
        for t,u,v in edges:
            union(u, v, t)
        # print(size_alice, size_bob)
        if max(size_alice) != n or max(size_bob) != n:
            return -1
        return len(edges) - ans
class Solution:
    def span(self,A,n,flag,par):
        countt=0
        countu=0
        for i,(t,u,v) in enumerate(A):
            if t==flag:
                countt+=1
                p1=u
                while par[p1]>0:
                    p1=par[p1]
                p2=v
                while par[p2]>0:
                    p2=par[p2]
                    
                if p1==p2:
                    continue
                    
                else:
                    countu+=1
                    if abs(par[p1])>abs(par[p2]):
                        par[p1]-=par[p2]
                        if u!=p1:
                            par[u]=p1
                        par[v]=p1
                        par[p2]=p1
                    else:
                        par[p2]-=par[p1]
                        if v!=p2:
                            par[v]=p2
                        par[u]=p2
                        par[p1]=p2
         
        return countt,countu
        
    
    
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        par=[-1]*(n+1)
        ct,cu=self.span(edges,n,3,par)
        if cu==n-1:
            return len(edges)-cu
        copy=par[:]
        ct2,cu2=self.span(edges,n,2,par)
        ct1,cu1=self.span(edges,n,1,copy)
        if cu+cu2!=n-1 or cu+cu1!=n-1:
            return -1
        
        return len(edges)-(cu+cu1+cu2)
        
        
        
        
        
        

class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        '''always keep type 3 if possible'''
        edges123 = [collections.defaultdict(list)  for _ in range(3) ] 
        for t, a, b in edges: edges123[t-1][a-1].append(b-1)
        # type 0 1 2 for alice, bob and both
        self.res = 0
        Parents = [[i for i in range(n)] for _ in range(2) ]
        selectedEdges = [0,0]
        def FindRoot(n,t):
            #print('node',n,'type',t)
            if Parents[t][n] != n:
                Parents[t][n] = FindRoot(Parents[t][n] ,t)
            return Parents[t][n] 
        def Uni(x,y,t):
            rx, ry = FindRoot(x,t), FindRoot(y,t)
            if rx == ry: return 0
            else:
                Parents[t][rx] = y
                return 1
            
        def connect(thetype):
            mytypes = [thetype] if thetype < 2 else [ 0, 1 ]
            for node in range(n):
                for neighbor in edges123[thetype][node]:
                    if all(Uni(node,neighbor,t) for t in mytypes):
                        for t in mytypes: selectedEdges[t] += 1
                    else:
                        self.res += 1
            # for t in mytypes: 
            #     root = [FindRoot(i,t) for i in range(n)]
            #     print(thetype,t, 'parents',Parents[t],root,selectedEdges,self.res)
                
        connect(2)
        connect(0)
        connect(1)
        return self.res if all(selectedEdges[t]==n-1 for t in [0,1]) else -1

class UF():
    def __init__(self,n):
        self.parent = list(range(n))
    def find(self,p):
        if self.parent[p] != p:
            self.parent[p] = self.find (self.parent[p])
        return self.parent[p]
    def union(self,p,q):
        pr = self.find(p)
        qr = self.find(q)
        if pr == qr:
            return False
        else:
            self.parent[pr] = qr
            return True

class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        ufA, ufB, ufAB = UF(n),UF(n),UF(n)
        usefulAB = 0
        for edge in edges:
            t = edge[0]
            x = edge[1]
            y = edge[2]  
            if t == 1:
                ufA.union(x-1, y -1)
            elif t == 2:
                ufB.union(x-1, y -1)
            else:
                ufA.union(x-1, y -1)
                ufB.union(x-1, y -1)
                usefulAB += ufAB.union(x-1, y -1)
        
        if len([i for i in range(n) if ufA.parent[i] == i]) > 1 or len([i for i in range(n) if ufB.parent[i] == i]) > 1:
            return -1
        return len(edges) - (2 * (n - 1) - usefulAB)

class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        elements = [-1] * (n + 1)
        elements[0] = 0
        def find(elements, i):
            while elements[i] >= 0:
                i = elements[i]
            return i
        
        def union(elements, i, j):
            i = find(elements, i)
            j = find(elements, j)
            if i == j:
                return
            if elements[i] <= elements[j]:
                if elements[i] == elements[j]:
                    elements[i] -= 1
                elements[j] = i
            else:
                elements[i] = j
        
        def count(elements):
            return sum(1 for i in elements if i < 0)
        
        result = 0
        for t, u, v  in edges:
            if t != 3:
                continue
            if find(elements, u) == find(elements, v):
                result += 1
            else:
                union(elements, u, v)
        elements2 = elements[:]
        for t, u, v in edges:
            if t == 1:
                if find(elements, u) == find(elements, v):
                    result += 1
                else:
                    union(elements, u, v)
            elif t == 2:
                if find(elements2, u) == find(elements2, v):
                    result += 1
                else:
                    union(elements2, u, v)
        if count(elements) > 1 or count(elements2) > 1:
            return -1
        return result
class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        A, B, ans, m = list(), list(), 0, len(edges)
        ta, tb = set(), set()
        for i, (t, u, v) in enumerate(sorted(edges, reverse=True)):
            if t != 3:
                if (u, v, 3) in ta or (u, v, 3) in tb:
                    ans += 1
                    m -= 1
                elif t == 1:
                    ta.add((u, v, t))
                    A.append((u, v, i))
                else:
                    tb.add((u, v, t))
                    B.append((u, v, i))
            else:
                ta.add((u, v, t))
                A.append((u, v, i))
                tb.add((u, v, t))
                B.append((u, v, i))

        def mst(edges):
            p = list(range(n+1))
            ret = set()
            def find(x):
                if x != p[x]:
                    p[x] = find(p[x])
                return p[x]
            for u, v, i in edges:
                pu, pv = find(u), find(v)
                if pu != pv:
                    ret.add(i)
                    p[pu] = pv
            return ret if len(ret) == n-1 else None
        ta = mst(A)
        if ta is None:
            return -1
        tb = mst(B)
        if tb is None:
            return -1
        return ans + m - len(ta|tb)
class Solution:
    
    
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        
        edge_type = {1:True, 2:False, 3:True}
        self.father = [i for i in range(n)]
        self.count = n
        self.travel(n, edges, edge_type)
        count_a = self.count
        
        edge_type = {1:False, 2:True, 3:True}
        self.father = [i for i in range(n)]
        self.count = n
        self.travel(n, edges, edge_type)
        count_b = self.count
        
        if count_a > 1 or count_b > 1:
            return -1
        
        edge_type = {1:False, 2:False, 3:True}
        self.father = [i for i in range(n)]
        self.count = n
        self.travel(n, edges, edge_type)
        count_both = self.count
        
        delete_both = len([i for i in edges if i[0] in [3]]) - (n - count_both)
        delete_a = len([i for i in edges if i[0] in [1, 3]]) - (n - 1) 
        delete_b = len([i for i in edges if i[0] in [2, 3]]) - (n - 1) 
        
        return delete_a + delete_b - delete_both
        
        
    
    def travel(self, n, edges, edge_type):
        
        for edge in edges:
            if edge_type[edge[0]]:
                self.union(edge[1] - 1, edge[2] - 1)

             
        
    def find(self, idx):
        if self.father[idx] == idx:
            return idx
        else:
            self.father[idx] = self.find(self.father[idx])
            return self.father[idx]
                    
                    
    def union(self, idx1, idx2):
        father_a = self.find(idx1)
        father_b = self.find(idx2)
        if father_a != father_b:
            self.count -= 1
            self.father[father_a] = father_b

class Solution:
	def maxNumEdgesToRemove(self, n: int, edges: [[int]]) -> int:
		def find(arr: [int], x: int) -> int: 
			if arr[x] != x:
				arr[x] = find(arr, arr[x])
			return arr[x]

		def union(arr: [int], x: int, y: int) -> int:
			px, py = find(arr, x), find(arr, y)
			if px == py:
				return 0
			arr[px] = py
			return 1

		arr = [i for i in range(n+1)]
		a, b, ans = 0, 0, 0
		for t, x, y in edges:
			if t == 3:
				if union(arr, x, y):
					a += 1
					b += 1
				else:
					ans += 1

		tmp = arr[:]
		for t, x, y in edges:
			if t == 1:
				if union(arr, x, y):
					a += 1
				else:
					ans += 1

		arr = tmp
		for t, x, y in edges:
			if t == 2:
				if union(arr, x, y):
					b += 1
				else:
					ans += 1

		return ans if a == b == n-1 else -1
from collections import deque

class UnionFind:
    def __init__(self, nodes):
        self.parent = { n: None for n in nodes }
        self.size = { n: 1 for n in nodes }
        
    def find_parent(self, node):
        path = []
        
        while self.parent[node] is not None:
            path.append(node)
            node = self.parent[node]
        
        for n in path:
            self.parent[n] = node
            
        return node
    
    def connected(self, a, b):
        return self.find_parent(a) == self.find_parent(b)
    
    def connect(self, a, b):
        a = self.find_parent(a)
        b = self.find_parent(b)
        
        if a != b:
            if self.size[a] > self.size[b]:
                self.parent[b] = a
                self.size[a] += self.size[b]
            else:
                self.parent[a] = b
                self.size[b] += self.size[a]
        
def min_spanning_tree(uf, nodes, edges):
    result = []
    for e in edges:
        t, a, b = tuple(e)
        
        if uf.connected(a, b):
            continue
        else:
            uf.connect(a, b)
            result.append(e)
    return result
    
class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        nodes = list(range(1, n+1))
        uf_alice = UnionFind(nodes)
        uf_bob = UnionFind(nodes)
        
        common_edges = min_spanning_tree(uf_alice, nodes, [e for e in edges if e[0] == 3])
        min_spanning_tree(uf_bob, nodes, [e for e in edges if e[0] == 3])
        
        alice_edges = min_spanning_tree(uf_alice, nodes, [e for e in edges if e[0] == 1])
        bob_edges = min_spanning_tree(uf_bob, nodes, [e for e in edges if e[0] == 2])
        
        if uf_alice.size[uf_alice.find_parent(1)] < n or uf_bob.size[uf_bob.find_parent(1)] < n:
            return -1
        
        return len(edges) - (len(common_edges) + len(alice_edges) + len(bob_edges))
class UnionFind:
    def __init__(self, n):
        self.p = {}
        self.group = n
        for i in range(1, n+1):
            self.p[i] = i
            

    def unite(self, a, b):    
        pa, pb = self.find(a), self.find(b)
        if pa != pb:
            self.p[pa] = pb
            self.group -= 1
            return True
        return False

    def find(self, a):
        if self.p[a] != a:
            self.p[a] = self.find(self.p[a]);
        return self.p[a]

    def united(self):
        return self.group == 1
    
class Solution:
    # copied https://leetcode.com/problems/remove-max-number-of-edges-to-keep-graph-fully-traversable/discuss/831506/Textbook-Union-Find-Data-Structure-Code-with-Explanation-and-comments
    
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        edges = sorted(edges, reverse=True)
        
        edgesAdded = 0        
        bob, alice = UnionFind(n), UnionFind(n)        
        for edge in edges:
            tp, one, two = edge[0], edge[1], edge[2]
            if tp == 3:
                bu = bob.unite(one, two)
                au = alice.unite(one, two)
                edgesAdded += 1 if bu or au else 0
            elif tp == 2:
                edgesAdded += bob.unite(one, two)
            else:
                edgesAdded += alice.unite(one, two)

        return len(edges)-edgesAdded if bob.united() and alice.united() else -1
        

class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        '''always keep type 3 if possible'''
        edges123 = [collections.defaultdict(list)  for _ in range(3) ] 
        for t, a, b in edges: edges123[t-1][a-1].append(b-1)
        # type 0 1 2 for alice, bob and both
        self.res = 0
        Parents = [[i for i in range(n)] for _ in range(2) ]
        sizes = [[1]*n for _ in range(2)]
        selectedEdges = [0,0]
        def FindRoot(n,t):
            #print('node',n,'type',t)
            if Parents[t][n] != n:
                Parents[t][n] = FindRoot(Parents[t][n] ,t)
            return Parents[t][n] 
        def Uni(x,y,t):
            rx, ry = FindRoot(x,t), FindRoot(y,t)
            if rx == ry: return 0
            else:
                if sizes[t][rx] >= sizes[t][ry]:
                    Parents[t][ry] = rx
                    sizes[t][rx] = max(sizes[t][rx],sizes[t][ry])
                else:
                    Parents[t][rx] = ry
                return 1
            
        def connect(thetype):
            mytypes = [thetype] if thetype < 2 else [ 0, 1 ]
            for node in range(n):
                for neighbor in edges123[thetype][node]:
                    if all(Uni(node,neighbor,t) for t in mytypes):
                        for t in mytypes: selectedEdges[t] += 1
                    else:
                        self.res += 1
            # for t in mytypes: 
            #     root = [FindRoot(i,t) for i in range(n)]
            #     print(thetype,t, 'parents',Parents[t],root,selectedEdges,self.res)
                
        connect(2)
        connect(0)
        connect(1)
        return self.res if all(selectedEdges[t]==n-1 for t in [0,1]) else -1

class DisjointSet:
    def __init__(self, n):
        self.parent = {i: i for i in range(1, n+1)}
        self.rank = {i: 1 for i in range(1, n+1)}
        self.count = n
        
    def find(self, x):
        if x != self.parent[x]:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
        
    def union(self, x, y):
        root_x, root_y = self.find(x), self.find(y)
        if root_x == root_y:
            return False # x, y are already connected
        self.count -= 1
        if self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        elif self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        else:
            self.parent[root_x] = root_y
            self.rank[root_y] += 1
        
        return True

class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        ds_a = DisjointSet(n)
        ds_b = DisjointSet(n)
        
        out = 0
        edges = sorted(edges, key=lambda e: e[0], reverse=True)
        for t, u, v in edges:
            if t == 3:
                if not ds_a.union(u, v):
                    out += 1
                else:
                    ds_b.union(u, v)
            elif t == 2:
                if not ds_b.union(u, v):
                    out += 1
            elif t == 1:
                if not ds_a.union(u, v):
                    out += 1
        
        if (ds_a.count > 1) or (ds_b.count > 1):
            return -1
        
        return out

class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        parent = {}
        for i in range(1, n + 1):
            parent[i] = i
        
        r = 0
        
        both = 0
        for c, a, b in edges:
            if c == 3:
                if self.find(a, parent) == self.find(b, parent):
                    r += 1
                else:
                    self.union(a, b, parent)
                    both += 1
        
        alice = both
        aliceP = parent.copy()
        for c, a, b in edges:
            if c == 1 or c == 3:
                if self.find(a, aliceP) == self.find(b, aliceP):
                    if c == 1:
                        r += 1
                else:
                    self.union(a, b, aliceP)
                    alice += 1
        print(alice)
        if alice < n - 1:
            return -1
        
        bob = both
        bobP = parent.copy()
        for c, a, b in edges:
            if c == 2 or c == 3:
                if self.find(a, bobP) == self.find(b, bobP):
                    if c == 2:
                        r += 1
                else:
                    self.union(a, b, bobP)
                    bob += 1
        print(bob)
        if bob < n - 1:
            return -1
        return r
                
                    
    
    def union(self, a, b, parent):
        pa = self.find(a, parent)
        pb = self.find(b, parent)
        if pa == pb:
            return
        parent[pb] = pa
        return
    
    def find(self, a, parent):
        path = [a]
        while a in parent and parent[a] != a:
            a = parent[a]
            path.append(a)
        for p in path:
            parent[p] = a
        return a

class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        # u6240u6709u8fb9u6309u7167u7c7bu578bu6392u5e8fuff0c u7c7bu578bu4e3a3u7684u8fb9u6709u6700u9ad8u4f18u5148u7ea7
        edges.sort(key=lambda edge: edge[0], reverse=True)
        
        def build_graph(types):
            removed = set()
            neighbors = defaultdict(set)
            graph = []
            for (t, a, b) in edges: # edge: t, a->b
                if t not in types: continue
                if b in neighbors[a]:
                    removed.add((t, a, b))
                    continue
                # print((t, a, b))
                
                neighbors[a].add(b)
                neighbors[b].add(a)
                graph.append((t, a, b))
            # print('========')
            return removed, graph
        
        def find(f, a):
            if f[a] == 0: return a
            if f[a] != a: f[a] = find(f, f[a])
            return f[a]

        def generate_tree(graph, removed):
            f = defaultdict(int)
            nodes = set()
            for t, a, b in graph:
                nodes.add(a)
                nodes.add(b)
                fa, fb = find(f, a), find(f, b)
                if fa == fb: 
                    removed.add((t, a, b)) 
                else:
                    f[fb] = fa
        
            return len(nodes) != n
        
        alice_removed, alice_graph = build_graph(set([1, 3]))
        bob_removed, bob_graph = build_graph(set([2, 3]))
        
        if generate_tree(alice_graph, alice_removed): return -1
        if generate_tree(bob_graph, bob_removed): return -1
        
        ans = len(alice_removed.union(bob_removed))
        return ans
class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        # Union find
        def find(i):
            if i != root[i]:
                root[i] = find(root[i])
            return root[i]

        def uni(x, y):
            x, y = find(x), find(y)
            if x == y: return 0
            root[x] = y
            return 1

        res = e1 = e2 = 0

        # Alice and Bob
        root = list(range(n + 1))
        for t, i, j in filter(lambda e: e[0] == 3, edges):
            if uni(i, j):
                e1 += 1
                e2 += 1
            else:
                res += 1
        root0 = root[:]

        # only Alice
        for t, i, j in edges:
            if t == 1:
                if uni(i, j):
                    e1 += 1
                else:
                    res += 1

        # only Bob
        root = root0
        for t, i, j in edges:
            if t == 2:
                if uni(i, j):
                    e2 += 1
                else:
                    res += 1

        return res if e1 == e2 == n - 1 else -1
class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]], check_type = 1) -> int:
        mem = {i: i for i in range(1, n + 1)}
        def find(k):
            if mem[k] != k:
                mem[k] = find(mem[k])
            return mem[k]
        
        def union(k1, k2):
            f1, f2 = find(k1), find(k2)
            if f1 != f2:
                mem[f1] = f2
        
        res = 0
        for t, e1, e2 in edges:
            if t == 3:
                f1, f2 = find(e1), find(e2)
                if f1 == f2 and check_type == 1:
                    res += 1
                union(e1, e2)
        
        for t, e1, e2 in edges:
            if t == check_type:
                f1, f2 = find(e1), find(e2)
                if f1 == f2:
                    res += 1
                else:
                    union(f1, f2)

        roots = set(map(find, list(range(1, n + 1))))
        if len(roots) > 1:
            return -1
        
        if check_type == 1:
            res2 = self.maxNumEdgesToRemove(n, edges, check_type = 2)
            if res2 == -1:
                return -1
            return res + res2
        else:
            return res

class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        edges1,edges2,edges3 = [],[],[]
        for t,u,v in edges:
            if t==1:
                E = edges1
            if t==2:
                E = edges2
            if t==3:
                E = edges3
            E.append((u,v))
        '''print(1, edges1)
        print(2, edges2)
        print(3, edges3)'''

        cnt = 0
        fathers = list(range(n+1))
    
        def find(fathers, v):
            if fathers[v]!=v:
                fathers[v]=find(fathers, fathers[v])
            return fathers[v]
        
        def union(fathers, a, b):
            rootA = find(fathers, a)
            rootB = find(fathers, b)
            if rootA != rootB:
                fathers[rootB] = rootA
                return True
            return False
        
        # type 3
        for u,v in edges3:
            if not union(fathers, u, v):
                cnt+=1
        
        #print(3, cnt)
        #print(fathers)
        def count(fathers, edges):
            cnt = 0
            for u,v in edges:
                if not union(fathers, u, v):
                    cnt+=1
            for i,v in enumerate(fathers):
                if i:
                    find(fathers, i)
            if len(set(fathers[1:]))>1:
                return -1
            return cnt
        # Alice
        a = count(fathers[:], edges1)
        #print('alice', a)
        if a==-1:
            return -1
        else:
            cnt+=a
        b = count(fathers[:], edges2)
        #print('bob', b)
        if b==-1:
            return -1
        else:
            cnt+=b
        return cnt
    

class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        
        alice = UnionFind(n)
        bob = UnionFind(n)
        
        remove = 0
        for t, u, v in edges:
            if t == 3:
                aliceMerge = alice.union(u, v)
                bobMerge = bob.union(u, v)
                if not aliceMerge and not bobMerge:
                    remove += 1
                
        for t, u, v in edges:
            if t != 3 and not alice.union(u, v) and not bob.union(u, v):
                remove += 1
                
        if alice.num_of_components != 1 or bob.num_of_components != 1:
            return -1
        
        return remove

class UnionFind:
    
    def __init__(self, n):
        self.parent = {i+1: i+1 for i in range(n)}
        self.size = {i+1: 1 for i in range(n)}
        
        self.num_of_components = n
        
    def find(self, p):
        
        root = p
        while(root != self.parent[root]):
            root = self.parent[root]
            
        node = p
        while (node != self.parent[node]):
            parent = self.parent[node]
            self.parent[node] = root
            node = parent
            
        return root
    
    def union(self, p, q):
        rootP = self.find(p)
        rootQ = self.find(q)
        
        if rootP == rootQ:
            return False
        
        if self.size[rootP] > self.size[rootQ]:
            self.size[rootP] += self.size[rootQ]
            self.parent[rootQ] = rootP
        else:
            self.size[rootQ] += self.size[rootP]
            self.parent[rootP] = rootQ
        
        self.num_of_components -= 1
        return True
class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        edges_alice = [e for e in edges if e[0] == 1 or e[0] == 3]
        edges_bob = [e for e in edges if e[0] == 2 or e[0] == 3]
        edges_both = [e for e in edges if e[0] == 3]
        
        nodes = [i for i in range(1, n + 1)]
        
        union_find = unionFind(nodes)
        for edge in edges_alice:
            union_find.union(edge[1], edge[2])
        if union_find.count != 1:
            return -1
        num_removed_edge_alice = len(edges_alice) - (n - 1)
        
        union_find = unionFind(nodes)
        for edge in edges_bob:
            union_find.union(edge[1], edge[2])
        if union_find.count != 1:
            return -1
        num_removed_edge_bob = len(edges_bob) - (n - 1)
        
        union_find = unionFind(nodes)
        for edge in edges_both:
            union_find.union(edge[1], edge[2])
        num_removed_edge_both = len(edges_both) - (n - union_find.count)
        
        return num_removed_edge_alice + num_removed_edge_bob - num_removed_edge_both
        

class unionFind:
    def __init__(self, nodes):
        self.father = {node: node for node in nodes}
        self.count = len(nodes)
        
    def find(self, node):
        if self.father[node] == node:
            return node
        self.father[node] = self.find(self.father[node])
        return self.father[node]
    
    def union(self, node_a, node_b):
        father_a = self.find(node_a)
        father_b = self.find(node_b)
        if father_a != father_b:
            self.father[father_a] = father_b
            self.count -= 1
            

class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        # Union find
        def find(i):
            if i != root[i]:
                root[i] = find(root[i])
            return root[i]

        def uni(x, y):
            x, y = find(x), find(y)
            if x == y: return 0
            root[x] = y
            return 1

        res = e1 = e2 = 0

        # Alice and Bob
        root = list(range(n + 1))
        for t, i, j in edges:
            if t == 3:
                if uni(i, j):
                    e1 += 1
                    e2 += 1
                else:
                    res += 1
        root0 = root[:]

        # only Alice
        for t, i, j in edges:
            if t == 1:
                if uni(i, j):
                    e1 += 1
                else:
                    res += 1

        # only Bob
        root = root0
        for t, i, j in edges:
            if t == 2:
                if uni(i, j):
                    e2 += 1
                else:
                    res += 1

        return res if e1 == e2 == n - 1 else -1
            
            

class DSU:
    def __init__(self, n):
        self.p = list(range(n))
        self.r = [0] * n
    def find(self, i): 
        if self.p[i] != i: 
            self.p[i] = self.find(self.p[i])
        return self.p[i]
    def union(self, i, j): 
        pi, pj = self.find(i), self.find(j)
        if pi != pj:
            if self.r[pi] >= self.r[pj]: 
                self.p[pj] = pi
                self.r[pi] += (self.r[pi] == self.r[pj])
            else: 
                self.p[pi] = pj
            return True
        return False


class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        if not edges: return -1
        uf1, n1, uf2, n2 = DSU(n), n, DSU(n), n
        edges.sort(reverse=True, key = lambda x: x[0])
        # t 1:alice, 2:bob, 3:both
        ans = 0
        for t, u, v in edges: 
            print((n1, n2))
            if t == 3: 
                u1, u2 = uf1.find(u-1) == uf1.find(v-1), uf2.find(u-1) == uf2.find(v-1)
                if u1 and u2 : 
                    ans += 1
                else: 
                    if not u1: 
                        uf1.union(u-1, v-1)
                        n1 -= 1
                    if not u2: 
                        uf2.union(u-1, v-1)
                        n2 -= 1
                        
            elif t == 1: 
                if uf1.find(u-1) != uf1.find(v-1): 
                    n1 -= uf1.union(u-1, v-1)
                else:
                    ans += 1
            elif t == 2: 
                if uf2.find(u-1) != uf2.find(v-1): 
                    n2 -= uf2.union(u-1, v-1)
                else:
                    ans += 1
                
                
#             if u1 and uf1.find(u-1) != uf1.find(v-1): 
#                 n1 -= uf1.union(u-1, v-1)
#                 can_delete = False
                    
#             if u2 and uf2.find(u-1) != uf2.find(v-1):
#                 n2 -= uf2.union(u-1, v-1)
#                 can_delete = False
                
            # ans += can_delete
        # print(uf1.p)
        # print(uf2.p)
        print((ans, n1, n2))
        return ans if (n1 <= 1 and n2 <=1) else -1
                
                
            
            
            
            
            

from typing import List
from collections import defaultdict
class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        AlicePath = defaultdict(set)
        BobPath = defaultdict(set)
        CommonPath = defaultdict(set)
        for type, u, v in edges:
            if type == 1 or type == 3 :
                AlicePath[u-1].add(v-1)
                AlicePath[v-1].add(u-1)
            if type == 2 or type == 3 :
                BobPath[u-1].add(v-1)
                BobPath[v-1].add(u-1)
            if type == 3:
                CommonPath[u-1].add(v-1)
                CommonPath[v-1].add(u-1)

        # u8ba1u7b97u4e00u7ec4u8fb9u53efu4ee5u62c6u5206u4e3au51e0u7ec4u53efu8fdeu901au56fe
        def count(m: defaultdict(set)):
            visited = [False] * n
            ret = 0 
            
            def dfs(i: int, isNewGraph:bool):
                nonlocal ret
                if visited[i]: return 
                visited[i] = True
                # u5916u5c42u904du5386u65f6uff0cu4e3au65b0u56feu3002u5185u90e8u9012u5f52u65f6u4e3au65e7u7684u8fdeu901au56feuff0cu65e0u9700u589eu52a0u7edfu8ba1u6570u76ee
                if isNewGraph: ret += 1 
                for endPoint in m[i]:
                    dfs(endPoint, False)

            for i in range(n):
                dfs(i, True) # u8fd9u91ccTrueu8868u793au662fu4eceu5916u5c42u5f00u59cbdfsuff0cu9047u5230u6ca1u6709u8bbfu95eeu8fc7u7684u8282u70b9u5c31u9700u8981u5c06u8ba1u6570+1

            return ret

        if count(AlicePath) > 1 or count(BobPath)> 1 :
            return -1

        x = count(CommonPath)
        return len(edges) - (n+x-2)


class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        d_alice = {i: i for i in range(1, n + 1)}
        d_bob = {i: i for i in range(1, n + 1)}
        
        def find(d, ind):
            if d[ind] != ind:
                d[ind] = find(d, d[ind])
            return d[ind]
        
        def union(d, i1, i2):
            d[find(d, i1)] = find(d, i2)
        
        edges.sort(reverse=True)
        res = 0
        
        for typ, i, j in edges:
            if typ == 3:
                if find(d_alice, i) == find(d_alice, j) and find(d_bob, i) == find(d_bob, j):
                    res += 1
                    continue
                union(d_alice, i, j)
                union(d_bob, i, j)
            elif typ == 2:
                if find(d_alice, i) == find(d_alice, j):
                    res += 1
                    continue
                union(d_alice, i, j)
            else:
                if find(d_bob, i) == find(d_bob, j):
                    res += 1
                    continue
                union(d_bob, i, j)
                
        for i in range(2, n + 1):
            if find(d_alice, i) != find(d_alice, i-1) or find(d_bob, i) != find(d_bob, i-1):
                return -1
        
        return res

class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
       
        def findRoot(node,p):
            if node not in p:
                p[node] = node
                return node
            if p[node] != node:
                p[node] = findRoot(p[node],p)
            return p[node]
        
        def union(node1, node2, p):
            if findRoot(node1,p) == findRoot(node2,p):
                return 0
            p[findRoot(node1,p)] = findRoot(node2,p)
            return 1
        
        def checkUnited(p,n):
            root = None
            for node in p:
                if root != None and findRoot(node, p) != root:
                    return False
                root = findRoot(node, p)
            return len(p) == n
        
        cnt = 0
        p1 = {}
        p2 = {}
        for edge in edges:
            if edge[0] == 3:
                cnt += union(edge[1], edge[2], p1)
                union(edge[1], edge[2], p2)
      
        for edge in edges:
            if edge[0] == 1:
                cnt += union(edge[1], edge[2], p1)
            if edge[0] == 2:
                cnt += union(edge[1], edge[2], p2)
       
        
        return len(edges) - cnt if checkUnited(p1,n) and checkUnited(p2,n) else -1
class UnionFind:
    def __init__(self, n):
        self.p = {}
        self.group = n
        for i in range(1, n+1):
            self.p[i] = i
            

    def unite(self, a, b):    
        pa, pb = self.find(a), self.find(b)
        if pa != pb:
            self.p[pa] = pb
            self.group -= 1
            return True
        return False

    def find(self, a):
        if self.p[a] != a:
            self.p[a] = self.find(self.p[a]);
        return self.p[a]

    def united(self):
        return self.group == 1
    
class Solution:
    # copied https://leetcode.com/problems/remove-max-number-of-edges-to-keep-graph-fully-traversable/discuss/831506/Textbook-Union-Find-Data-Structure-Code-with-Explanation-and-comments    
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        edges = sorted(edges, reverse=True)
        
        edgesAdded = 0        
        bob, alice = UnionFind(n), UnionFind(n)        
        for edge in edges:
            tp, one, two = edge[0], edge[1], edge[2]
            if tp == 3:
                bu = bob.unite(one, two)
                au = alice.unite(one, two)
                edgesAdded += 1 if bu or au else 0
            elif tp == 2:
                edgesAdded += bob.unite(one, two)
            else:
                edgesAdded += alice.unite(one, two)

        return len(edges)-edgesAdded if bob.united() and alice.united() else -1
        

class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        total_edges = []
        bob_edges = []
        alice_edges = []
        
        
        for (t, x, y) in edges:
            x -= 1
            y -= 1
            if t == 1:
                # bob[x].append(y)
                # bob[y].append(x)
                bob_edges.append((x, y))
            elif t == 2:
                # alice[x].append(y)
                # alice[y].append(x)
                alice_edges.append((x, y))
            else:
                # total[x].append(y)
                # total[y].append(x)
                total_edges.append((x, y))
        
        
        def kruskal(colors, sets, edges):
            used = 0
            for (x, y) in edges:
                if colors[x] == colors[y]:
                    continue

                used += 1
                if len(sets[colors[x]]) < len(sets[colors[y]]):
                    x, y = y, x

                #add y to x
                color_x = colors[x]
                color_y = colors[y]

                for node in sets[color_y]:
                    colors[node] = color_x

                sets[color_x].extend(sets[color_y])
                
            return used
        
        total_colors = list(range(n))
        total_sets = [[i] for i in range(n)]
        
        used_total = kruskal(total_colors, total_sets, total_edges)
        # print("New colors", total_colors)
        
        
        bob_colors = total_colors[::]
        alice_colors = total_colors[::]
        
        bob_sets = [el[::] for el in total_sets]
        alice_sets = [el[::] for el in total_sets]
        
        used_bob = kruskal(bob_colors, bob_sets, bob_edges)
        used_alice = kruskal(alice_colors, alice_sets, alice_edges)
        
        # print("Shared uses", used_total)
        # print("Bob used", used_bob)
        # print("Alice used", used_alice)
        
        if (used_total + used_bob != n - 1) or (used_total + used_alice != n - 1):
            return -1
        
        return len(edges) - used_total - used_bob - used_alice

class Solution:
    def span(self,A,n,flag,par):
        countt=0
        countu=0
        for i,(t,u,v) in enumerate(A):
            if t==flag:
                countt+=1
                p1=u
                while par[p1]>0:
                    p1=par[p1]
                p2=v
                while par[p2]>0:
                    p2=par[p2]
                    
                if p1==p2:
                    continue
                    
                else:
                    countu+=1
                    if abs(par[p1])>abs(par[p2]):
                        par[p1]-=par[p2]
                        if u!=p1:
                            par[u]=p1
                        par[v]=p1
                        par[p2]=p1
                    else:
                        par[p2]-=par[p1]
                        if v!=p2:
                            par[v]=p2
                        par[u]=p2
                        par[p1]=p2
         
        return countt,countu
        
    
    
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        edges.sort(key=lambda x:x[0],reverse=True)
        par=[-1]*(n+1)
        ct,cu=self.span(edges,n,3,par)
        if cu==n-1:
            return len(edges)-cu
        copy=par[:]
        ct2,cu2=self.span(edges,n,2,par)
        ct1,cu1=self.span(edges,n,1,copy)
        if cu+cu2!=n-1 or cu+cu1!=n-1:
            return -1
        
        return len(edges)-(cu+cu1+cu2)
        
        
        
        
        
        

class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        parent = [x for x in range(n+1)]
        rank = [0]*(n+1)
        
        def find(x, parent):
            if x!=parent[x]:
                parent[x]=find(parent[x], parent)
            return parent[x]
        
        def union(x,y, parent, rank):
            xp, yp = find(x, parent), find(y, parent)
            if xp==yp:
                return False
            if rank[xp]>rank[yp]:
                parent[yp]=xp
            elif rank[xp]<rank[yp]:
                parent[xp]=yp
            else:
                parent[yp]=xp
                rank[xp]+=1
            return True
        
        type1 = [(u,v) for w,u,v in edges if w==1]
        type2 = [(u,v) for w,u,v in edges if w==2]
        type3 = [(u,v) for w,u,v in edges if w==3]
        
            
        ans = 0
        for u,v in type3:
            if not union(u,v, parent, rank):
                ans+=1
                
        p1 = parent.copy()
        r1 = rank.copy()
        
        for u,v in type1:
            if not union(u,v,p1,r1):
                ans+=1
                
        for u,v in type2:
            if not union(u,v,parent,rank):
                ans+=1
                
        arr1 = [find(p,p1) for p in p1[1:]]
        arr2 = [find(p,parent) for p in parent[1:]]
        if (arr1[1:] != arr1[:-1]) or (arr2[1:] != arr2[:-1]):
            return -1
        return ans
class UnionNode:
    def __init__(self):
        self.parent = None
        self.order = 0


class UnionSet:
    def __init__(self, N):
        self.els = [UnionNode() for i in range(N)]
        self.edge_count = 0


    def find(self, i):
        node = self.els[i]

        if node.parent is None:
            return i

        node.parent = self.find(node.parent)
        return node.parent


    def union(self, i, j):
        i = self.find(i)
        j = self.find(j)

        if i == j:
            return

        self.edge_count += 1

        i_node = self.els[i]
        j_node = self.els[j]

        if i_node.order < j_node.order:
            i_node, j_node = j_node, i_node
            i, j = j, i

        j_node.parent = i
        i_node.order += (i_node.order == j_node.order)


    def connected(self):
        return len(self.els) == self.edge_count + 1


class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        alice_union  = UnionSet(n)
        bob_union    = UnionSet(n)
        shared_union = UnionSet(n)

        for (type_, u, v) in edges:
            if type_ in (1, 3):
                alice_union.union(u - 1, v - 1)

            if type_ in (2, 3):
                bob_union.union(u - 1, v - 1)

            if type_ == 3:
                shared_union.union(u - 1, v - 1)

        if not alice_union.connected() or not bob_union.connected():
            return -1

        missing_edges = n - 1 - shared_union.edge_count
        needed_edges = shared_union.edge_count + 2*missing_edges

        return len(edges) - needed_edges
class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        class UnionFind:
            def __init__(self):
                self.parent = {}
                self.e = 0
            def find(self, a):
                if a not in self.parent:
                    self.parent[a] = a
                p = a
                while p != self.parent[p]:
                    p = self.parent[p]
                while a != p:
                    tmp = a
                    self.parent[a] = p
                    a = self.parent[tmp]
                return p
            def union(self, a, b):
                pa, pb = self.find(a), self.find(b)
                if pa != pb:
                    self.parent[pa] = pb
                    self.e += 1
                    return 0
                return 1
        
        ufa, ufb = UnionFind(), UnionFind()
        ans = 0
        for t, u, v in edges:
            if t == 3:
                ans += ufa.union(u, v)
                ufb.union(u, v)
                
        for t, u, v in edges:
            if t == 1:
                ans += ufa.union(u, v)
            if t == 2:
                ans += ufb.union(u, v)
                
        return ans if (ufa.e == n-1 and ufb.e == n-1) else -1
        

class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        p1 = list(range(n + 1))
        p2 = list(range(n + 1))

        def find(parents, node):
            if parents[node] != node:
                parents[node] = find(parents, parents[node])
            return parents[node]
        
        def union(parents, a, b):
            ra = find(parents, a)
            rb = find(parents, b)
            if ra != rb:
                parents[ra] = rb
                return True
            return False
        
        def count(parents, n):
            res = 0
            root = find(parents, 1)
            for i in range(n + 1):
                if find(parents, i) == root:
                    res += 1
            return res
        
        res = 0
        edges.sort(reverse = True)
        for t, u, v in edges:
            if t == 1 and not union(p1, u, v):
                res += 1
            if t == 2 and not union(p2, u, v):
                res += 1
            if t == 3:
                del1 = not union(p1, u, v)
                del2 = not union(p2, u, v)
                if (del1 and del2):
                    res += 1
        if count(p1, n) == n and count(p2, n) == n:
            return res
        return -1
class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        def find(i, parents):
            if parents[i] != i:
                parents[i] = find(parents[i], parents)
            return parents[i]
            
        def union(i, j, parents, groups):
            p_i = find(i, parents)
            p_j = find(j, parents)
            if p_i != p_j:
                groups -= 1
                if p_i > p_j:
                    parents[p_j] = p_i
                else:
                    parents[p_i] = p_j
            return groups
        
        alice = []
        bob = []
        res = 0
        
        parents = list(range(n+1))
        groups = n  
        
        for t, a, b in edges:
            if t == 1:
                alice.append((a, b))
            elif t == 2:
                bob.append((a, b))
            else:
                if find(a, parents) == find(b, parents):
                    res += 1
                else:
                    groups = union(a, b, parents, groups)
                    
        if groups == 1:
            return res + len(alice) + len(bob)
        
        ga = groups
        gb = groups
        pa = parents[:]
        pb = parents[:]
        
        while alice:
            i, j = alice.pop()
            if find(i, pa) == find(j, pa):
                res += 1
            else:
                ga = union(i, j, pa, ga)
            if ga == 1:
                res += len(alice)
                break
                
        if ga != 1:
            return -1
        
        while bob:
            i, j = bob.pop()
            if find(i, pb) == find(j, pb):
                res += 1
            else:
                gb = union(i, j, pb, gb)
            if gb == 1:
                res += len(bob)
                break
                
        if gb != 1:
            return -1
        
        return res
        
            

class UnionSet:
    def __init__(self, n):
        self.parent = {i:i for i in range(1, n+1)}
        # self.level = {i:0 for i in range(1, n+1)}
    
    def union(self, i, j):
        # if self.level[i] < self.level[j]:
        self.parent[self.get(i)] = self.get(j)
        
    def get(self, i):
        if self.parent[i] == i:
            return i
        else:
            self.parent[i] = self.get(self.parent[i])
            return self.parent[i]

class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
                    
        mapping = {k: [] for k in range(1, 4)}
        alice = UnionSet(n)
        bob = UnionSet(n)
        for t, u, v in edges:
            mapping[t].append((v, u))

        
        for u, v in mapping[3]:
            alice.union(u, v)
        bob.parent = alice.parent.copy()
        
        seen = set()
        for i in range(1, n+1):
            seen.add(alice.get(i))
        
        after3 = len(seen)
        
        for u,v in mapping[1]:
            alice.union(u, v)
        for u, v in mapping[2]:
            bob.union(u, v)
        
        seen = set()
        for i in range(1, n+1):
            seen.add(alice.get(i))
        if len(seen) != 1:
            return -1

        seen = set()
        for i in range(1, n+1):
            seen.add(bob.get(i))
        if len(seen) != 1:
            return -1
    
    
        delete3 = len(mapping[3]) - (n - after3)
        use3 = len(mapping[3]) - delete3
        return len(mapping[1]) + len(mapping[2]) + use3 * 2 - 2 * (n - 1) + delete3
class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        class UnionFind:
            def __init__(self):
                self.parent = {}
                self.e = 0
            def find(self, a):
                if a not in self.parent:
                    self.parent[a] = a
                p = a
                while p != self.parent[p]:
                    p = self.parent[p]
                while a != p:
                    tmp = a
                    self.parent[a] = p
                    a = self.parent[tmp]
                return p
            def union(self, a, b):
                pa, pb = self.find(a), self.find(b)
                if pa != pb:
                    self.parent[pa] = pb
                    self.e += 1
                    return 1
                return 0
        
        ufa, ufb = UnionFind(), UnionFind()
        ans = 0
        for t, u, v in edges:
            if t == 3:
                ans += ufa.union(u, v)
                ufb.union(u, v)
                
        for t, u, v in edges:
            if t == 1:
                ans += ufa.union(u, v)
            if t == 2:
                ans += ufb.union(u, v)
                
        return (len(edges) - ans) if (ufa.e == n-1 and ufb.e == n-1) else -1
        

class UnionFind:
    def __init__(self, size):
        self.father = [i for i in range(size + 1)]

    def find(self, x):
        if self.father[x] == x:
            return x 
        self.father[x] = self.find(self.father[x])
        return self.father[x]

    def connect(self, a, b):
        root_a, root_b = self.find(a), self.find(b)
        if root_a != root_b:
            self.father[root_a] = root_b

class Solution:
    def mst(self, n, edges, t):
        g = sorted([e for e in edges if e[0] in (t,3)], key = lambda x : -x[0])
        UF, u = UnionFind(n), set()
        for t, a, b in g:
            if UF.find(a) != UF.find(b):
                UF.connect(a,b)
                u.add((t,a,b))
        return len({UF.find(i) for i in range(1, n + 1)}) == 1, u 
    
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        traversed1, mst1 = self.mst(n, edges, 1)
        traversed2, mst2 = self.mst(n, edges, 2)
        if (not traversed1) or (not traversed2):
            return -1 
        return len(edges) - len(mst1 | mst2)

from typing import Tuple
class Solution:
    def __init__(self):
      self.roots = {}
      self.ranks = {}
      self.groups = 0
    
    def find(self, node_info: Tuple[int, int]):
      self.roots.setdefault(node_info, node_info)
      self.ranks.setdefault(node_info, 1)
      if self.roots[node_info] != node_info:
        self.roots[node_info] = self.find(self.roots[node_info])
      return self.roots[node_info]
    
    def union(self, node_info1, node_info2) -> bool:  # returns if the edge can be removed
      root1, root2 = self.find(node_info1), self.find(node_info2)
      if root1 != root2:
        self.groups -= 1
        if self.ranks[root2] < self.ranks[root1]:
          self.roots[root2] = root1
        elif self.ranks[root1] < self.ranks[root2]:
          self.roots[root1] = root2
        else:
          self.roots[root2] = root1
          self.ranks[root1] += 1
        return False  # we can't remove this edge because it's used
      else:
        return True  # we can remove this edge because there already is a path for these 2 nodes.
    
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
      """
      Union find for alice and bob separately. But the trick here is that we must go over type 3 first, because
      once we have type 3, then for the same node pair, we don't ever need type1 or type 2. But if we see types 
      in such order: 1, 3, 2. We can remove type 1, but we can't remove type2, because at that point type 1 is
      already removed.
      Time:
      So we need to sort the edges by type first: O(eloge), then iterate over the edges O(e)
      In fact, we can remove the sort if we loop over the edges twice, the first time we only do type 3, and
      then the second type we do the other 2 types. This removes the sorting complexity.
      """
      # with sorting:
#       edges.sort(key=lambda edge: -edge[0])  
#       removes = 0
#       self.groups = n * 2
#       for tp, n1, n2 in edges:
#         can_remove = False
#         if tp == 1:
#           can_remove = self.union((1, n1), (1, n2))
#         elif tp == 2:
#           can_remove = self.union((2, n1), (2, n2))
#         else:
#           can_remove1, can_remove2 = self.union((1, n1), (1, n2)), self.union((2, n1), (2, n2))
#           can_remove = can_remove1 and can_remove2
#         removes += (1 if can_remove else 0)  
      
#       # If in the end both alice and alice have a single group, then return removed count
#       return removes if self.groups == 2 else -1 
      
      # without sorting:
      removes = 0
      self.groups = n * 2
      for tp, n1, n2 in edges:  # first iteration
        if tp == 3:
          can_remove1, can_remove2 = self.union((1, n1), (1, n2)), self.union((2, n1), (2, n2))
          removes += (1 if can_remove1 and can_remove2 else 0)  
      
      for tp, n1, n2 in edges:
        can_remove = False
        if tp == 1:
          can_remove = self.union((1, n1), (1, n2))
        elif tp == 2:
          can_remove = self.union((2, n1), (2, n2))
        removes += (1 if can_remove else 0)
      
      return removes if self.groups == 2 else -1 
# u65e0u5411u56feu7684u751fu6210u6811
# u751fu6210u6811->u6709Gu7684u5168u90e8vuff0cu4f46u8fb9u6570u6700u5c11u7684u8fdeu901au5b50u56fe; u5e26u6743->u53efu4ee5u627emst
# 3->u9047u73afu5220 u5f97u8fb9u6570u76ee 
# 3uff0c1->u9047u73afu5220 u5f97u8fb9u6570u76ee ?=n-1
# 3uff0c2->u9047u73afu5220 u5f97u8fb9u6570u76ee ?=n-1
from collections import defaultdict
class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        def get_circle(comb):
            G = defaultdict(list)
            for t,u,v in edges:
                if t in comb:
                    G[u].append(v)
                    G[v].append(u)
            cnt = 0
            visited = set()
            for u in list(G.keys()):
                if u in visited:
                    continue
                visited.add(u)
                stack = [u]
                while stack:
                    cur = stack.pop()
                    for nei in G[cur]:
                        if nei not in visited:
                            visited.add(nei)
                            cnt+=1
                            stack.append(nei)
            return cnt 
        
        type3 = get_circle((3,))
        if type3==n-1:
            return len(edges)-n+1   
        type2 = get_circle((2,3))       
        type1 = get_circle((1,3))
        if type2==type1==n-1:
            return len(edges)- (2*n-2-type3)
        else: 
            return -1
        
        
                    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        def find(p, i):
            if p[i] != i:
                p[i] = find(p, p[i])
            return p[i]
        
        def union(p, i, j):
            p[find(p, i)] = find(p, j)
        
        parents = list(range(n))
        ans = 0
        
        edges = [(type, u - 1, v - 1) for type, u, v in edges]
        
        for type, u, v in edges:
            if type == 3:
                if find(parents, u) == find(parents, v):
                    ans += 1
                union(parents, u, v)
        
        alice = list(parents)
        bob = list(parents)
        
        for type, u, v in edges:
            if type == 1:
                if find(alice, u) == find(alice, v):
                    ans += 1
                union(alice, u, v)
            if type == 2:
                if find(bob, u) == find(bob, v):
                    ans += 1
                union(bob, u, v)
        
        return ans if len({find(alice, i) for i in range(n)}) == 1 and len({find(bob, i) for i in range(n)}) == 1 else -1
        
                    

class dsu:
    def __init__(self, n):
        self.n = n
        self.par = [i for i in range(n+1)]
    def find(self,x):
        if self.par[x]==x:return x
        self.par[x] = self.find(self.par[x])
        return self.par[x]
    def union(self,x,y):
        x = self.find(x)
        y = self.find(y)
        if x==y: return 1
        self.par[x] = y
        return 0
class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        gr1 = collections.defaultdict(list)
        gr2 = collections.defaultdict(list)
        edges1 = []
        edges2 = []
        mp = {1:1, 2:1, 3:-1}
        for typ,x,y in edges:
            if typ==3 or typ==1:
                edges1.append([mp[typ], x,y])
            if typ==3 or typ==2:
                edges2.append([mp[typ], x,y])
        edges1.sort()
        edges2.sort()
        dsu1 = dsu(n)
        dsu2 = dsu(n)
        oth1=oth2=0
        res =0
        for typ,x,y in edges1:
            if dsu1.union(x,y):
                if typ!=-1: res+=1
                else: oth1+=1
        for typ,x,y in edges2:
            if dsu2.union(x,y):
                if typ!=-1: res+=1
                else: oth2+=1
        count = 0
        for i in range(1,n+1):
            if i==dsu1.par[i]:count+=1
            if i==dsu2.par[i]:count+=1
        if count>2:return -1
        return res+min(oth1,oth2)
            
        

class UnionFind:
    def __init__(self,n):
        self.parent = list(range(n))
        self.size = 1
    
    def find(self,x):
        if x!=self.parent[x]:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self,x,y):
        parentx,parenty = self.find(x),self.find(y)
        if parentx==parenty:
            return False
        self.parent[parentx] = parenty
        self.size+=1
        return True

class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        uf1,uf2 = UnionFind(n),UnionFind(n)
        res = 0
        
        for t,u,v in edges:
            if t!=3:
                continue
            if (not uf1.union(u-1,v-1)) or not (uf2.union(u-1,v-1)):
                res += 1
        
        for t,u,v in edges:
            if t==1 and not uf1.union(u-1,v-1):
                res+=1
            elif t==2 and not uf2.union(u-1,v-1):
                res+=1
        
        return res if (uf1.size==n and uf2.size==n) else -1
class UF:
    def __init__(self, n):
        self.parent = [i for i in range(n)]
        self.size = [1] * n
        
    def isConnected(self, a, b):
        return self.find(a) == self.find(b)
    
    def find(self, a):
        while self.parent[a] != a:            
            # path compression
            self.parent[a] = self.parent[self.parent[a]]
            a = self.parent[a]
        return a
    
    def union(self, a, b):
        p1 = self.find(a)
        p2 = self.find(b)
        if self.isConnected(p1, p2):
            return
        
        # optimization - try to make the new tree balanced
        if self.size[p1] < self.size[p2]:
            self.parent[p1] = p2
            self.size[p2] += self.size[p1]
        else:
            self.parent[p2] = p1
            self.size[p1] += self.size[p2]
                   
class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        uf_1 = UF(n)
        uf_2 = UF(n)
        res = 0
        edges = sorted(edges, key = lambda edges: edges[0], reverse = True) 
        
        for t, a, b in edges:
            if t == 3:
                if uf_1.isConnected(a-1, b-1) and uf_2.isConnected(a-1, b-1):
                    res += 1
                else:
                    uf_1.union(a-1, b-1)
                    uf_2.union(a-1, b-1)
            elif t == 1:
                if uf_1.isConnected(a-1, b-1):
                    res += 1
                else:
                    uf_1.union(a-1, b-1)
            else:
                if uf_2.isConnected(a-1, b-1):
                    res += 1
                else:
                    uf_2.union(a-1, b-1)
                    
        return res if self.isValid(uf_1, n) and self.isValid(uf_2, n) else -1
    
    def isValid(self, uf, n):
        for i in range(0, n-1):
            if not uf.isConnected(i, i+1):
                return False
        return True
    

class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        alice = DSU()
        bob = DSU()
        res = 0
        for t, u, v in edges:
            if t == 3:
                if alice.find(u) == alice.find(v):
                    res += 1
                else:
                    alice.union(u, v)
                    bob.union(u, v)
                    
        for t, u, v in edges:
            if t == 1:
                if alice.find(u) == alice.find(v):
                    res += 1
                else:
                    alice.union(u, v)
            if t == 2:
                if bob.find(u) == bob.find(v):
                    res += 1       
                else:
                    bob.union(u, v)
                    
        if max(bob.count.values()) != n or max(alice.count.values()) != n:
            return -1
        
        return res
        
class DSU:
    def __init__(self):
        self.father = {}
        self.count = defaultdict(lambda:1)
    
    def find(self, a):
        self.father.setdefault(a, a)
        if a != self.father[a]:
            self.father[a] = self.find(self.father[a])
        return self.father[a]
    
    def union(self, a, b):
        _a = self.find(a)
        _b = self.find(b)
        if _a != _b:
            self.father[_a] = self.father[_b]
            self.count[_b] += self.count[_a]
#number of unneccessary type 3 + redudant type 1 + redudant type 2
class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        res = 0
        
        #find all type 3
        type3 = self.getEdges(edges, 3)
        #find how many redudant edges for only the give connected vertices
        redudant_edges = self.findRedudantEdges(type3, n + 1)
        res += len(redudant_edges)
        
        type3 = list(set(type3) - set(redudant_edges))
        
        type1 = type3 + self.getEdges(edges, 1)
        redudant_edges = self.findRedudantEdges(type1, n + 1)
        #test if Bob and Alice can reach all edges
        if len(type1) - len(redudant_edges) != n - 1:
            return -1
        
        res += len(redudant_edges)
        
        type2 = type3 + self.getEdges(edges, 2)
        redudant_edges = self.findRedudantEdges(type2, n + 1)
        if len(type2) - len(redudant_edges) != n - 1:
            return -1
        
        res += len(redudant_edges)
        
        return res 
    
    #use Union-Find
    def findRedudantEdges(self, edges, n):
        res = []
        parents = [i for i in range(n)]
        
        def findParent(a):
            if a != parents[a]:
                parents[a] = findParent(parents[a])
            
            return parents[a]
        
        for u, v in edges:
            pu = findParent(u)
            pv = findParent(v)
            if pu == pv: 
                res.append((u, v))
            else:
                parents[pu] = pv
                
        return res
    
        
    def getEdges(self, edges, type_num):
        return [(u, v) for t, u, v in edges if t == type_num]
        
        
        

class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        
        n_a={}
        n_b={}
        for e in edges:
            if e[0]==1 or e[0]==3:
                if e[1] not in n_a:
                    n_a[e[1]]=[]
                n_a[e[1]].append(e[2])
                if e[2] not in n_a:
                    n_a[e[2]]=[]
                n_a[e[2]].append(e[1])
            if e[0]==2 or e[0]==3:
                if e[1] not in n_b:
                    n_b[e[1]]=[]
                n_b[e[1]].append(e[2])
                if e[2] not in n_b:
                    n_b[e[2]]=[]
                n_b[e[2]].append(e[1])
                
        visited=set()
        l=list(n_b.keys())
        start=l[0]
        visited.add(start)
        q=[start]
        while q:
            actual=q[0]
            del q[0]
            if actual in n_a:
                for nb in n_a[actual]:
                    if nb not in visited:
                        visited.add(nb)
                        q.append(nb)
        if len(visited)!=n:
            return -1
        
        visited=set()
        l=list(n_b.keys())
        start=l[0]
        visited.add(start)
        q=[start]
        while q:
            actual=q[0]
            del q[0]
            if actual in n_b:
                for nb in n_b[actual]:
                    if nb not in visited:
                        visited.add(nb)
                        q.append(nb)
        if len(visited)!=n:
            return -1        

        
        
        
        
        
        
        parent_a={}
        parent_b={}
        for i in range(1,n+1):
            parent_a[i]=i
            parent_b[i]=i
            
        def find_a(x):
            if parent_a[x]!=x:
                parent_a[x]=find_a(parent_a[x])
            return parent_a[x]
        def find_b(x):
            if parent_b[x]!=x:
                parent_b[x]=find_b(parent_b[x])
            return parent_b[x]
        def union_a(x,y):
            x=find_a(x)
            y=find_a(y)
            if x!=y:
                parent_a[x]=y
                return 0
            else:
                return 1
        def union_b(x,y):
            x=find_b(x)
            y=find_b(y)
            if x!=y:
                parent_b[x]=y
                return 0
            else:
                return 1
        count=0
        for e in edges:
            if e[0]==3:
                u1=union_a(e[1],e[2])
                u2=union_b(e[1],e[2])
                if u1==u2 and u1==1:
                    count+=1
 
        for e in edges:
            if e[0]==1:    
                u1=union_a(e[1],e[2])
                if u1==1:
                    count+=1
            if e[0]==2:    
                u2=union_b(e[1],e[2])
                if u2==1:
                    count+=1
        
        return count
                    
        


class dsu:
    def __init__(self,n):
        self.parent =[-1 for i in range(n)]
        self.size = [0 for i in range(n)]

    def make_set(self,node):
        self.parent[node] = node

    def find(self,a):
        if self.parent[a] == a:
            return self.parent[a]
        else:
            self.parent[a] = self.find(self.parent[a])
            return self.parent[a]

    def union(self,a,b):
        a = self.find(a)
        b = self.find(b)
        if a!=b:
            if self.size[a] < self.size[b]:
                temp = a
                a = b
                b = temp
            self.parent[b] =self.parent[a]
            self.size[a] = self.size[a] + self.size[b]

class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        
        ans = 0
        treeA = dsu(n+1)
        treeB = dsu(n+1)
        for i in range(n+1):
            treeA.make_set(i)
            treeB.make_set(i)
        edges.sort(reverse = True)
        # print(edges)
        m = len(edges)
        i = 0
        while(i<m and edges[i][0]==3):
            if treeA.find(edges[i][1]) == treeA.find(edges[i][2]) :
                ans= ans + 1
                i= i +1
                continue
            
            treeA.union(edges[i][1],edges[i][2])
            treeB.union(edges[i][1],edges[i][2])
            i= i +1
        # print(treeB.parent,"B")
        # print(treeA.parent,"A")
        while(i<m and edges[i][0]==2):
            if treeA.find(edges[i][1]) == treeA.find(edges[i][2]) :
                ans= ans + 1
                i= i +1
                continue
            treeA.union(edges[i][1],edges[i][2])
            # print(treeA.parent,"A")
            i= i +1
            
        while(i<m and edges[i][0]==1):
            if treeB.find(edges[i][1]) == treeB.find(edges[i][2]) :
                ans= ans + 1
                print("here")
                i= i +1
                continue
            treeB.union(edges[i][1],edges[i][2])
            # print(treeB.parent,"B",ans,edges[i])
            i= i +1
        def check(tree):
            curr= tree.find(1)
            for i in range(1,n+1):
                if tree.find(i) != curr:
                    return 0
            return 1
        # print(ans)
        if check(treeA) == 0 or check(treeB) == 0:
            return -1
        return ans
                
        
        
from typing import Tuple
class Solution:
    def __init__(self):
      self.roots = {}
      self.ranks = {}
      self.groups = 0
    
    def find(self, node_info: Tuple[int, int]):
      self.roots.setdefault(node_info, node_info)
      self.ranks.setdefault(node_info, 1)
      if self.roots[node_info] != node_info:
        self.roots[node_info] = self.find(self.roots[node_info])
      return self.roots[node_info]
    
    def union(self, node_info1, node_info2) -> bool:  # returns if the edge can be removed
      root1, root2 = self.find(node_info1), self.find(node_info2)
      if root1 != root2:
        self.groups -= 1
        if self.ranks[root2] < self.ranks[root1]:
          self.roots[root2] = root1
        elif self.ranks[root1] < self.ranks[root2]:
          self.roots[root1] = root2
        else:
          self.roots[root2] = root1
          self.ranks[root1] += 1
        return False  # we can't remove this edge because it's used
      else:
        return True  # we can remove this edge because there already is a path for these 2 nodes.
    
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
      """
      Union find for alice and bob separately.
      """
      edges.sort(key=lambda edge: -edge[0])
      removes = 0
      self.groups = n * 2
      for tp, n1, n2 in edges:
        can_remove = False
        if tp == 1:
          can_remove = self.union((1, n1), (1, n2))
        elif tp == 2:
          can_remove = self.union((2, n1), (2, n2))
        else:
          can_remove1, can_remove2 = self.union((1, n1), (1, n2)), self.union((2, n1), (2, n2))
          can_remove = can_remove1 and can_remove2
        removes += (1 if can_remove else 0)  
      
      # If in the end both alice and alice have a single group, then return removed count
      return removes if self.groups == 2 else -1 
class UnionFind:
    def __init__(self, n):
        self.par = list(range(n+1))
        self.sz = list(range(n+1))
    
    def find(self, i):
        while i != self.par[i]:
            i = self.par[i]
        return i
    
    def union(self, x, y):
        x,y = self.find(x), self.find(y)
        if x == y: return 0
        if self.sz[x] < self.sz[y]:
            x, y = y, x
        self.par[y] = x
        self.sz[x] += self.sz[y]
        return 1
    
class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        edges = sorted(edges, key=lambda x: -x[0])
        uf = UnionFind(n)
        res = e1 = e2 = 0
        for t, u, v in edges:
            if t == 3:
                if uf.union(u,v):
                    e1 += 1
                    e2 += 1
                else:
                    res += 1
        p0 = uf.par[:]
        for t, u, v in edges:
            if t ==2:
                if uf.union(u,v):
                    e2 += 1
                else:
                    res += 1
        uf.par = p0
        for t, u, v in edges:
            if t == 1:
                if uf.union(u,v):
                    e1 += 1
                else:
                    res += 1
        return res if e1 == e2 == n-1 else -1
class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        def find(p,x):
            if p[x] != x:
                p[x] = find(p,p[x])
            return p[x]  
        def merge(p,x, y):
            x = p[x]
            y = p[y]
            if x<y:
                p[y] = x
            else:
                p[x] = y
        parent = [x for x in range(n+1)]
        path = 0
        for e in edges:
            if e[0] == 3 and find(parent, e[1]) != find(parent, e[2]):
                merge(parent,e[1],e[2])
                path += 1
        a, b = list(parent), list(parent)
        patha, pathb = 0, 0
        for e in edges:
            if e[0] == 1 and find(a, e[1]) != find(a, e[2]):
                merge(a,e[1],e[2])
                patha += 1
            if e[0] == 2 and find(b, e[1]) != find(b, e[2]):
                merge(b,e[1],e[2])
                pathb += 1   
        if patha + path != n-1 or pathb + path != n - 1: return -1 
        return len(edges) - (path + patha + pathb)
                       


'''
n is the number of nodes
if 1 < value of nodes <= n
Krustal O(ElogE)

rank[node]: the longth depth of node's children
'''

class Solution:
    def maxNumEdgesToRemove(self, n, edges):
        # Union find
        def find(node):
            if node != parent[node]:
                parent[node] = find(parent[node])
            return parent[node]

        def uni(node1, node2):
            parent1, parent2 = find(node1), find(node2)
            if parent1 == parent2: return 0
            if rank[parent1] > rank[parent2]:
                parent[parent2] = parent1
            elif rank[parent1] == rank[parent2]:
                parent[parent2] = parent1
                rank[parent1] += 1
            else:
                parent[parent1] = parent2 
            
            return 1

        res = union_times_A = union_times_B = 0

        # Alice and Bob
        parent = [node for node in range(n + 1)]
        rank = [0 for node in range(n + 1)]
        
        for t, node1, node2 in edges:
            if t == 3:
                if uni(node1, node2):
                    union_times_A += 1
                    union_times_B += 1
                else:
                    res += 1
        parent0 = parent[:]  # Alice union will change the parent array, keep origin for Bob

        # only Alice
        for t, node1, node2 in edges:
            if t == 1:
                if uni(node1, node2):
                    union_times_A += 1
                else:
                    res += 1

        # only Bob
        parent = parent0
        for t, node1, node2 in edges:
            if t == 2:
                if uni(node1, node2):
                    union_times_B += 1
                else:
                    res += 1
# only if Alice and Bob both union n-1 times, the graph is connected for both of them
        return res if union_times_A == union_times_B == n - 1 else -1
class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        def root(i):
            while par[i] != i:
                par[i] = par[par[i]]
                i = par[i]
            return i
        
        def connected(i, j):
            x = root(i)
            y = root(j)
            return x == y
        
        def union(i, j):
            x = root(i)
            y = root(j)
            if x == y:
                return
            if sz[x] <= sz[y]:
                par[x] =  y
                sz[y] += sz[x]
            else:
                par[y] = x
                sz[x] += sz[y]
                
        par = {}
        sz  = {}
        st = set()
        edges.sort(key = lambda x : 0 - x[0])
        count = 0
        for e in edges:
            t = e[0]
            x = e[1]
            y = e[2]
            st.add(x)
            st.add(y)
            
            xa = str(e[1]) + "a"
            xb = str(e[1]) + "b"
            
            ya = str(e[2]) + "a"
            yb = str(e[2]) + "b"
            
            if xa not in par:
                par[xa] = xa
                sz[xa] = 1
            if xb not in par:
                par[xb] = xb
                sz[xb] = 1
                
            if ya not in par:
                par[ya] = ya
                sz[ya] = 1
                
            if yb not in par:
                par[yb] = yb
                sz[yb] = 1
                
            if t == 3:
                if connected(xa, ya) and connected(xb, yb):
                    count += 1
                    continue
                union(xa, ya)
                union(xb, yb)
            elif t == 2:
                if connected(xb, yb):
                    count += 1
                    continue
                union(xb, yb)
            else:
                if connected(xa, ya):
                    count += 1
                    continue
                union(xa, ya)
        
        mxa = 0
        mxb = 0
        for x in sz:
            if x[-1] == "a":
                mxa = max(mxa, sz[x])
            elif x[-1] == "b":
                mxb = max(mxb, sz[x])
                
        if mxa == len(st) and mxb == len(st):
            return count
        return -1
            
                
                
                
                    
                
                
        
        

class DSU:
    
    def __init__(self, a):
        self.par = {x:x for x in a}
    
    def merge(self, u, v):
        rootu = self.find(u)
        rootv = self.find(v)
        
        if rootu == rootv:
            return False
        
        self.par[rootu] = rootv
        return True
    
    def find(self, u):
        if self.par[u] != u:
            self.par[u] = self.find(self.par[u])
        return self.par[u]

class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        
        dsu1 = DSU(range(1,n+1))
        dsu2 = DSU(range(1,n+1))
        
        grouper = defaultdict(list)
        for t,u,v in edges:
            grouper[t].append([u,v])
        
        both, alice, bob = grouper[3], grouper[1], grouper[2]
        
        ret = 0
        
        for u,v in both:
            if not dsu1.merge(u, v):
                ret += 1
            dsu2.merge(u, v)
                
        for u,v in alice:
            if not dsu1.merge(u, v):
                ret += 1
        
        for u,v in bob:
            if not dsu2.merge(u, v):
                ret += 1
        
        if len(set(dsu1.find(u) for u in dsu1.par)) != 1 or len(set(dsu2.find(u) for u in dsu2.par)) != 1:
            return -1
            
        return ret
class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        
        fa = [i for i in range(n+1)]
        def find(x):
            if fa[x]!=x:
                fa[x]=find(fa[x])
            return fa[x]
        
        def uni(x,y):
            fx = find(x)
            fy = find(y)
            fa[fx] = fy
        
        res = 0
        A = 0 # nodes Alice can go
        B = 0 # nodes Bob can go
        
        #type 3
        for t,u,v in edges:
            if t==3:
                fu = find(u)
                fv = find(v)
                if fu==fv:
                    res+=1
                else:
                    uni(u,v)
                    A+=1
                    B+=1
        
        fa_copy = fa[:]
        #edges Alice can go
        for t,u,v in edges:
            if t==1:
                fu=find(u)
                fv=find(v)
                if fu==fv:
                    res+=1
                else:
                    uni(u,v)
                    A+=1
        
        fa = fa_copy #Bob can't use the graph of Alice
        #edges bob can go
        for t,u,v in edges:
            if t==2:
                fu=find(u)
                fv=find(v)
                if fu==fv:
                    res+=1
                else:
                    uni(u,v)
                    B+=1
        
        if A!=n-1 or B!=n-1:
            return -1
        
        return res
        

from typing import *

class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        edges.sort(key=lambda x: (-x[0]))
        num_used = 0

        # Using the UnionFind DS with idx 0 as the parent and idx 1 as the height
        uf1 = [[i for i in range(n)]] + [[0] * n]
        uf2 = [[i for i in range(n)]] + [[0] * n]

        i = 0
        while i < len(edges):
            t, s, d = edges[i]
            s -= 1
            d -= 1

            if t == 3:
                if self.union_find_merge(uf1, s, d):
                    num_used += 1
                self.union_find_merge(uf2, s, d)
            elif t == 2:
                if self.union_find_merge(uf2, s, d):
                    num_used += 1
            else:
                if self.union_find_merge(uf1, s, d):
                    num_used += 1
            i += 1

        if self.find_num_components(n, uf1) > 1 or self.find_num_components(n, uf2) > 1:
            return -1

        return len(edges) - num_used

    def find_num_components(self, n, uf):
        num_components = 0
        for idx in range(n):
            parent = uf[0][idx]
            if idx == parent:
                num_components += 1
        return num_components

    def union_find_merge(self, uf, node1, node2):
        p1 = self.union_find_get_parent(uf, node1)
        p2 = self.union_find_get_parent(uf, node2)

        if p1 == p2:
            return False  # We can discard the edge as both have same parent

        if uf[1][p1] > uf[1][p2]:
            uf[0][p2] = p1
        else:
            uf[0][p1] = p2
            uf[1][p2] = max(uf[1][p2], uf[1][p1] + 1)

        return True

    def union_find_get_parent(self, uf, node):
        while uf[0][node] != node:
            node = uf[0][node]
        return node
class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        
        size1 = [1] * n
        size2 = [1] * n
        par1 = [i for i in range(n)]
        par2 = [i for i in range(n)]
        distinct1 = [n]
        distinct2 = [n]
        
        def find(node, par):
            while par[node] != node:
                par[node] = par[par[node]]
                node = par[node]
            return node
        
        def union(a,b,size,par,distinct):
            root1 = find(a,par)
            root2 = find(b,par)
            
            if root1 != root2:
                if size[root1] < size[root2]:
                    par[root1] = par[root2]
                    size[root2] += size[root1]
                else:
                    par[root2] = par[root1]
                    size[root1] += size[root2]
                distinct[0]-=1
                return True
            return False
            
        edges.sort(key=lambda x:x[0],reverse=True)
        edges_needed = 0
        for i in edges:
            type = i[0]
            u = i[1]-1
            v = i[2]-1
            if type == 3:
                a = union(u,v,size1,par1,distinct1)
                b = union(u,v,size2,par2,distinct2)
                if a or b:
                    edges_needed+=1
            elif type == 1:
                if union(u,v,size1,par1,distinct1):
                    edges_needed+=1
            else:
                # print(u,v)
                if union(u,v,size2,par2,distinct2):
                    edges_needed+=1
                # print(par2)
        # print(par1,par2)        
        if distinct1[0] != 1 or distinct2[0] != 1:
            return -1
        return len(edges) - edges_needed
                
        
        

class Solution:
    def find(self, x, uf):
        if uf.get(x) == None:
            uf[x] = x
            return x
        if uf[x] != x:
            uf[x] = self.find(uf[x], uf)
        return uf[x]
    
    def union(self, x, y, uf):
        root1 = self.find(x, uf)
        root2 = self.find(y, uf)
        if root1 == root2:
            return False
        uf[root1] = root2
        return True
    
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        edges.sort(reverse=True)
        uf1 = {}
        uf2 = {}
        check1 = [False] * n
        check2 = [False] * n
        need = 0
        for t, n1, n2 in edges:
            if t == 3:
                val1 = self.union(n1, n2, uf1) 
                val2 = self.union(n1, n2, uf2)
                if val1 or val2:
                    need += 1
            elif t == 1:
                if self.union(n1, n2, uf1):
                    need += 1
            elif t == 2:
                if self.union(n1, n2, uf2):
                    need += 1
        
        if len(uf1) != n:
            
            return -1
        if len(uf2) != n:
            
            return -1
        
        uf1_a = [(i, key) for i, key in enumerate(uf1)]
        uf2_a = [(i, key) for i, key in enumerate(uf2)]
        for i1, key1 in uf1_a:
            
            if i1 == 0:
                root1 = self.find(key1, uf1)
            else:
                if self.find(key1, uf1) != root1:
                   
                    return -1
        
        for i2, key2 in uf2_a:
            if i2 == 0:
                root2 = self.find(key2, uf2)
            else:
                if self.find(key2, uf2) != root2:
                   
                    return -1
        
        return len(edges) - need
                    

class Solution:
    
    class DisjointSet:
        def __init__(self, n):
            self.parents = [i for i in range(n+1)]
            self.ranks = [0 for i in range(n+1)]
            
        def parent(self, node):
            if self.parents[node] != node:
                self.parents[node] = self.parent(self.parents[node])
            return self.parents[node]
        
        def join(self, node1, node2):
            p1 = self.parent(node1)
            p2 = self.parent(node2)
            
            r1 = self.ranks[p1]
            r2 = self.ranks[p2]
            
            if r1 < r2:
                self.parents[p1] = p2
            elif r2 < r1:
                self.parents[p2] = p1
            else:
                self.parents[p1] = p2
                self.ranks[p1] += 1
    
        def is_connected(self, node1, node2):
            return self.parent(node1) == self.parent(node2)
    
    
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        ans = 0
        
        graph = [dict() for i in range(n+1)]
        
        for edge in edges:            
            if edge[2] not in graph[edge[1]]:
                graph[edge[1]][edge[2]] = edge[0]
            else:
                if edge[0] == 3:
                    if graph[edge[1]][edge[2]] == 4:
                        ans += 2
                    else:
                        ans += 1
                    graph[edge[1]][edge[2]] = 3
                elif graph[edge[1]][edge[2]] == 3:
                    ans += 1
                    graph[edge[1]][edge[2]] = 3
                else:
                    graph[edge[1]][edge[2]] = 4 
                
        
        alice_dset = Solution.DisjointSet(n)
        
        print(ans)
        
        for i in range(1, n+1):
            for (j, t) in list(graph[i].items()):
                if t == 3:
                    if not alice_dset.is_connected(i, j):
                        alice_dset.join(i, j)
                    else:
                        ans += 1
        bob_dset = copy.deepcopy(alice_dset)
        
        for i in range(1, n+1):
            for (j, t) in list(graph[i].items()):
                print((i, j, t))
                if t == 1 or t == 4:
                    if not alice_dset.is_connected(i, j):
                        alice_dset.join(i, j)
                    else:
                        ans += 1
                if t == 2 or t == 4:
                    if not bob_dset.is_connected(i, j):
                        bob_dset.join(i, j)
                    else:
                        ans += 1
                        
        for i in range(1, n):
            if not bob_dset.is_connected(i, i + 1) or not alice_dset.is_connected(i, i + 1):
                # print(i, bob_dset.parent(i), bob_dset.parent(i + 1))
                # print(i, alice_dset.parent(i), alice_dset.parent(i + 1))
                
                return -1
        
        return ans
        

class UnionFindSet:
    def __init__(self, n):
        self.parents = list(range(n))
        self.ranks = [1] * n
        self.size = 1
        
    def find(self, u):
        if u != self.parents[u]:
            self.parents[u] = self.find(self.parents[u])
        return self.parents[u]
    
    def union(self, u, v):
        pu, pv = self.find(u), self.find(v)
        if pu == pv:
            return False
        if self.ranks[pu] > self.ranks[pv]:
            self.parents[pv] = pu
        elif self.ranks[pv] > self.ranks[pu]:
            self.parents[pu] = pv
        else:
            self.parents[pu] = pv
            self.ranks[pv] += 1
        self.size += 1
        return True
    
    
class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        uf1, uf2, ans = UnionFindSet(n), UnionFindSet(n), 0
		
        for t, u, v in edges:
            if t != 3:
                continue
            if not uf1.union(u - 1, v - 1) or not uf2.union(u - 1, v - 1):
                ans += 1
        
        for t, u, v in edges:
            if t == 1 and not uf1.union(u - 1, v - 1):
                ans += 1
            elif t == 2 and not uf2.union(u - 1, v - 1):
                ans += 1
   
        return ans if uf1.size == n and uf2.size == n else -1
# class Solution:
#     def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
#         self.father_alice = [i for i in range(n + 1)]
#         self.father_bob = [i for i in range(n + 1)]
#         res = 0
#         edge_alice, edge_bob = 0, 0
#         for type, u, v in edges:
#             if type == 3:
#                 if self.connect(u, v, True) == 1:
#                     edge_alice += 1
#                     edge_bob += 1
#                 else:
#                     res += 1
                
#                 self.connect(u, v, False)
        
#         for type, u, v in edges:
#             if type == 1:
#                 if self.connect(u, v, True) == 1:
#                     edge_alice += 1
#                 else:
#                     res += 1
#             elif type == 2:
#                 if self.connect(u, v, False) == 1:
#                     edge_bob += 1
#                 else:
#                     res += 1
        
#         if edge_alice == edge_bob == n - 1:
#             return res
#         return -1
    
    
    
#     def find(self, x, is_alice):
#         if is_alice:
#             if self.father_alice[x] == x:
#                 return self.father_alice[x]
#             self.father_alice[x] = self.find(self.father_alice[x], True)
#             return self.father_alice[x]
#         else:
#             if self.father_bob[x] == x:
#                 return self.father_bob[x]
#             self.father_bob[x] = self.find(self.father_bob[x], False)
#             return self.father_bob[x]
    
#     def connect(self, a, b, is_alice):
#         if is_alice:
#             root_a = self.find(a, True)
#             root_b = self.find(b, True)
#             if root_a == root_b:
#                 return 0
#             else:
#                 self.father_alice[max(root_a, root_b)] = min(root_a, root_b)
#                 return 1
#         else:
#             root_a = self.find(a, False)
#             root_b = self.find(b, False)
#             if root_a == root_b:
#                 return 0
#             else:
#                 self.father_bob[max(root_a, root_b)] = min(root_a, root_b)
#                 return 1
        
        
        
        
        
#         self.father_alice = [i for i in range(n + 1)]
#         self.father_bob = [i for i in range(n + 1)]
        
#         res = 0
#         for type, u, v in edges:
#             if type == 3:
#                 res += self.connect(u, v, True)
#                 self.connect(u, v, False)
        
#         for type, u, v in edges:
#             if type == 1:
#                 res += self.connect(u, v, True)
#             elif type == 2:
#                 res += self.connect(u, v, False)
        
        
#         if self.check_valid(True) and self.check_valid(False):
#             return res
#         return -1
    
    
#     def find(self, x, is_alice):
#         if is_alice:
#             if self.father_alice[x] == x:
#                 return self.father_alice[x]
#             self.father_alice[x] = self.find(self.father_alice[x], True)
#             return self.father_alice[x]
        
#         else:
#             if self.father_bob[x] == x:
#                 return self.father_bob[x]
#             self.father_bob[x] = self.find(self.father_bob[x], False)
#             return self.father_bob[x]
        
#     def connect(self, a, b, is_alice):
#         if is_alice:
#             root_a = self.find(a, True)
#             root_b = self.find(b, True)
#             if root_a != root_b:
#                 self.father_alice[max(root_a, root_b)] = min(root_a, root_b)
#                 return 0
#             return 1
        
#         else:
#             root_a = self.find(a, False)
#             root_b = self.find(b, False)
#             if root_a != root_b:
#                 self.father_bob[max(root_a, root_b)] = min(root_a, root_b)
#                 return 0
#             return 1
        
#     def check_valid(self, is_alice):
#         if is_alice:
#             root = self.find(1, True)
#             for i in range(1, len(self.father_alice)):
#                 if self.find(i, True) != root:
#                     return False
#             return True
        
#         else:
#             root = self.find(1, False)
#             for i in range(1, len(self.father_bob)):
#                 if self.find(i, False) != root:
#                     return False
#             return True

class DSU:
    def __init__(self):
        self.parent = {}
        self.size = {}
        
    def root(self,A):
        tmp = A
        while self.parent[tmp]!=tmp:
            tmp = self.parent[tmp]
            self.parent[tmp] = self.parent[self.parent[tmp]]
        return tmp
    
    def union(self,A,B):
        if self.root(A)==self.root(B):
            return False
        else:
            if self.size[self.root(A)] >= self.size[self.root(B)]:
                self.size[self.root(A)]+= self.size[self.root(B)]
                self.parent[self.root(B)] = self.root(A)
            else:
                self.size[self.root(B)]+= self.size[self.root(A)]
                self.parent[self.root(A)] = self.root(B)
            return True
    
    def add_edge(self,A,B):
        if A not in self.parent:
            self.parent[A] = A
            self.size[A] = 1
        if B not in self.parent:
            self.parent[B] = B
            self.size[B] = 1
        return self.union(A,B)

    def get_node_count(self):
        return len(self.parent)
                 
class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        
        # Spanning Tree for Type 3 edges
        
        edges_removed = 0
        
        alice_graph,bob_graph = DSU(),DSU()
        
        for typ,u,v in edges:
            if typ==3:
                if not (alice_graph.add_edge(u,v) and bob_graph.add_edge(u,v)):
                    edges_removed+=1
        
        # Spanning Tree for Type 2 and Type 3 edges
        
        for typ,u,v in edges:
            if typ==2:
                if not bob_graph.add_edge(u,v):
                    edges_removed+=1
            if typ==1:
                if not alice_graph.add_edge(u,v):
                    edges_removed+=1
        
        # print('alice_graph',alice_graph.parent)
        # print('bob_graph',bob_graph.parent)
        
        if alice_graph.get_node_count()!=n or bob_graph.get_node_count()!=n:
            return -1
                    
        return edges_removed

class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        def get_color(u):
            out = colors[u]
            if out != u:                
                out = get_color(out)
                colors[u] = out
            return out
                            
        ans = 0
        colors = [i for i in range(n+1)]
        edges_3 = 0
        for one in edges:
            if one[0] == 3:
                u_color = get_color(one[1])
                v_color = get_color(one[2])
                
                if u_color == v_color:
                    ans += 1
                else:
                    edges_3 += 1
                    colors[v_color] = u_color
                    
        colors2 = list(colors)
        edges_count = edges_3
        for one in edges:
            if one[0] == 1:
                u_color = get_color(one[1])
                v_color = get_color(one[2])
                if u_color == v_color:
                    ans += 1
                else:                
                    edges_count += 1
                    colors[v_color] = u_color
                
        if edges_count < n - 1:
            return -1
        
        # print(edges_count)
        # print(colors)
        
        colors = colors2
        edges_count = edges_3
        for one in edges:
            if one[0] == 2:
                u_color = get_color(one[1])
                v_color = get_color(one[2])
                if u_color == v_color:
                    ans += 1
                else:
                    edges_count += 1
                    colors[v_color] = u_color
                
        if edges_count < n - 1:
            return -1
                        
        return ans
class Solution:
    def __init__(self):
        self.n = 0
        self.used = []
    def dfs(self,edges,i):
        self.used[i] = True
        for edge in edges[i]:
            if self.used[edge] == False:
                self.dfs(edges,edge)
    def iterate(self,edges):
        self.used = [False for i in range(self.n)]
        components = 0
        for i in range(self.n):
            if self.used[i] == True:
                continue
            self.dfs(edges,i)
            components += 1
        return components
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        self.n = n
        alice = [[] for i in range(self.n)]
        bob = [[] for i in range(self.n)]
        both = [[] for i in range(self.n)]
        bothCount = 0
        tot = len(edges)
        for edge in edges:
            if edge[0] == 1:
                alice[edge[1]-1].append(edge[2]-1)
                alice[edge[2]-1].append(edge[1]-1)
            if edge[0] == 2:
                bob[edge[1]-1].append(edge[2]-1)
                bob[edge[2]-1].append(edge[1]-1)
            if edge[0] == 3:
                bob[edge[1]-1].append(edge[2]-1)
                bob[edge[2]-1].append(edge[1]-1)
                alice[edge[1]-1].append(edge[2]-1)
                alice[edge[2]-1].append(edge[1]-1)
                both[edge[1]-1].append(edge[2]-1)
                both[edge[2]-1].append(edge[1]-1)
                bothCount += 1
        if self.iterate(alice) != 1 or self.iterate(bob) != 1:
            return -1
        bothComponents = self.iterate(both)
        needed = self.n - bothComponents
        needed += 2*(bothComponents-1)
        return tot - needed
class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        # Union find
        def find(i):
            if i != root[i]:
                root[i] = find(root[i])
            return root[i]

        def uni(x, y):
            x, y = find(x), find(y)
            if x == y: return 0
            root[x] = y
            return 1

        res = e1 = e2 = 0

        # Alice and Bob
        root = [*list(range(n + 1))]
        for t, i, j in edges:
            if t == 3:
                if uni(i, j):
                    e1 += 1
                    e2 += 1
                else:
                    res += 1
        root0 = root[:]
        
        root = root0[:]
        # only Alice
        for t, i, j in edges:
            if t == 1:
                if uni(i, j):
                    e1 += 1
                else:
                    res += 1

        # only Bob
        root = root0[:]
        for t, i, j in edges:
            if t == 2:
                if uni(i, j):
                    e2 += 1
                else:
                    res += 1

        return res if e1 == e2 == n - 1 else -1

class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        dic = {}
        type1 = 0
        type2 = 0
        type3 = 0
        for edge in edges:
            etype, a, b = edge
            if a in dic:
                if b in dic[a]:
                    dic[a][b].add(etype)
                else:
                    dic[a][b] = set([etype])
            else:
                dic[a] = {b: set([etype])}
            if b in dic:
                if a in dic[b]:
                    dic[b][a].add(etype)
                else:
                    dic[b][a] = set([etype])
            else:
                dic[b] = {a: set([etype])}
            if etype == 1:
                type1 += 1
            elif etype == 2:
                type2 += 1
            else:
                type3 += 1
                
                    
        res = 0         
        seen_A = [0] * n
        def dfs_A(i):
            seen_A[i - 1] = 1
            if i in dic:
                for j in dic[i]:
                    if (1 in dic[i][j] or 3 in dic[i][j]) and seen_A[j - 1] == 0:
                        dfs_A(j)
        seen_B = [0] * n
        def dfs_B(i):
            seen_B[i - 1] = 1
            if i in dic:
                for j in dic[i]:
                    if (2 in dic[i][j] or 3 in dic[i][j]) and seen_B[j - 1] == 0:
                        dfs_B(j)
        
        dfs_A(1)
        if sum(seen_A) != n:
            return -1
        dfs_B(1)
        if sum(seen_B) != n:
            return -1
                        
        seen_3 = [0] * n
        def dfs_3(i):
            seen_3[i - 1] = 1
            self.cnt += 1
            if i in dic:
                for j in dic[i]:
                    if (3 in dic[i][j]) and seen_3[j - 1] == 0:
                        dfs_3(j)
        
        tmp = 0
        self.cnt = 0
        tmp_n = 0
        for i in range(1, n + 1):
            if seen_3[i - 1] == 0:
                tmp_n += 1
                self.cnt = 0
                dfs_3(i)
                tmp += self.cnt - 1
                
        
        
        res += type3 - tmp
            
        return res + type1 + type2 - (tmp_n - 1) - (tmp_n - 1)
                

class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        dic = {}
        type1 = 0
        type2 = 0
        type3 = 0
        for edge in edges:
            etype, a, b = edge
            if a in dic:
                if b in dic[a]:
                    dic[a][b].add(etype)
                else:
                    dic[a][b] = set([etype])
            else:
                dic[a] = {b: set([etype])}
            if b in dic:
                if a in dic[b]:
                    dic[b][a].add(etype)
                else:
                    dic[b][a] = set([etype])
            else:
                dic[b] = {a: set([etype])}
            if etype == 1:
                type1 += 1
            elif etype == 2:
                type2 += 1
            else:
                type3 += 1
                
                    
        res = 0
        # for key, val in dic.items():
        #     for keyp, valp in val.items():
        #         if 3 in valp:
        #             if 1 in valp:
        #                 type1 -= 1
        #                 dic[key][keyp].remove(1)
        #                 res += 1
        #             if 2 in valp:
        #                 type2 -= 1
        #                 dic[key][keyp].remove(2)
        #                 res += 1
                        
        seen_A = [0] * n
        def dfs_A(i):
            seen_A[i - 1] = 1
            if i in dic:
                for j in dic[i]:
                    if (1 in dic[i][j] or 3 in dic[i][j]) and seen_A[j - 1] == 0:
                        dfs_A(j)
        seen_B = [0] * n
        def dfs_B(i):
            seen_B[i - 1] = 1
            if i in dic:
                for j in dic[i]:
                    if (2 in dic[i][j] or 3 in dic[i][j]) and seen_B[j - 1] == 0:
                        dfs_B(j)
        
        dfs_A(1)
        if sum(seen_A) != n:
            return -1
        dfs_B(1)
        if sum(seen_B) != n:
            return -1
                        
        seen_3 = [0] * n
        def dfs_3(i):
            seen_3[i - 1] = 1
            self.cnt += 1
            if i in dic:
                for j in dic[i]:
                    if (3 in dic[i][j]) and seen_3[j - 1] == 0:
                        dfs_3(j)
        
        tmp = 0
        self.cnt = 0
        tmp_n = 0
        for i in range(1, n + 1):
            if seen_3[i - 1] == 0:
                tmp_n += 1
                self.cnt = 0
                dfs_3(i)
                tmp += self.cnt - 1
                
        
        
        res += type3 - tmp
            
        return res + type1 + type2 - (tmp_n - 1) - (tmp_n - 1)
                

class DisjointSet:
    def __init__(self, elements):
        self.parent = [i for i in range(elements)]
        self.size = [1] * elements
    def find(self, value):
        if self.parent[value] != value:
            self.parent[value] = self.find(self.parent[value])
        return self.parent[value]
    def union(self, value1, value2):
        parent1, parent2 = self.find(value1), self.find(value2)
        if parent1 == parent2:
            return True
        if self.size[parent1] > self.size[parent2]:
            self.parent[parent2] = parent1
            self.size[parent1] += self.size[parent2]
        else:
            self.parent[parent1] = parent2
            self.size[parent2] += self.size[parent1]
        return False

class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        graph = [list() for i in range(n + 1)]
        for t, u, v in edges:
            graph[u].append([v, t])
            graph[v].append([u, t])
        def dfs(source, ty):
            nonlocal cnt
            cnt += 1
            vis[source] = 1
            for child, typer in graph[source]:
                if typer in [ty, 3] and not vis[child]:
                    dfs(child, ty)
        cnt = 0
        vis = [0] * (n + 1)
        dfs(1, 1)
        if cnt != n:
            return -1
        vis = [0] * (n + 1)
        cnt = 0
        dfs(1, 2)
        if cnt != n:
            return -1
        answer = 0
        dsu1, dsu2 = DisjointSet(n + 1), DisjointSet(n + 1)
        for t, u, v in edges:
            if t == 3:
                dsu1.union(u, v)
                if dsu2.union(u, v):
                    answer += 1
        for t, u, v in edges:
            if t == 1:
                if dsu1.union(u, v):
                    answer += 1
            if t == 2:
                if dsu2.union(u, v):
                    answer += 1
        return answer

class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
		# process graph
        graphs = [collections.defaultdict(list) for _ in range(3)]
        for c, i, j in edges:
            graphs[c-1][i].append((-c, j))
            graphs[c-1][j].append((-c, i))
		# build tree for Alice
        e = graphs[2][1] + graphs[0][1]
        heapq.heapify(e)
        treeset = set([1])
        type3 = 0
        while e:
            c, y = heapq.heappop(e)
            if y not in treeset:
                treeset.add(y)
                if c == -3:
                    type3 += 1
                for item in graphs[2][y]:
                    heapq.heappush(e, item)
                for item in graphs[0][y]:
                    heapq.heappush(e, item)
        if len(treeset) != n:
            return -1
		# build tree for Bob
        e = graphs[2][1] + graphs[1][1]
        heapq.heapify(e)
        treeset = set([1])
        while e:
            c, y = heapq.heappop(e)
            if y not in treeset:
                treeset.add(y)
                for item in graphs[2][y]:
                    heapq.heappush(e, item)
                for item in graphs[1][y]:
                    heapq.heappush(e, item)
        if len(treeset) != n:
            return -1
        return len(edges) + type3 - 2 * (n - 1)
class DisjointSet:
    def __init__(self, elements):
        self.parent = [i for i in range(elements)]
        self.size = [1] * elements
    def find(self, value):
        if self.parent[value] != value:
            self.parent[value] = self.find(self.parent[value])
        return self.parent[value]
    def union(self, value1, value2):
        parent1, parent2 = self.find(value1), self.find(value2)
        if parent1 == parent2:
            return True
        if self.size[parent1] > self.size[parent2]:
            self.parent[parent2] = parent1
            self.size[parent1] += self.size[parent2]
        else:
            self.parent[parent1] = parent2
            self.size[parent2] += self.size[parent1]
        return False

class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        graph = [list() for i in range(n + 1)]
        extra = 0
        for t, u, v in edges:
            graph[u].append([v, t])
            graph[v].append([u, t])
        def dfs(source, ty):
            nonlocal cnt
            cnt += 1
            vis[source] = 1
            for child, typer in graph[source]:
                if typer in [ty, 3] and not vis[child]:
                    dfs(child, ty)
        # To Check if Alice can visit all nodes.
        cnt = 0
        vis = [0] * (n + 1)
        dfs(1, 1)
        if cnt != n:
            return -1
        # To check if Bob can visit all nodes.
        vis = [0] * (n + 1)
        cnt = 0
        dfs(1, 2)
        if cnt != n:
            return -1
        answer = 0
        dsu1, dsu2 = DisjointSet(n + 1), DisjointSet(n + 1)
        for t, u, v in edges:
            if t == 3:
                dsu1.union(u, v)
                if dsu2.union(u, v):
                    answer += 1
        for t, u, v in edges:
            if t == 1:
                if dsu1.union(u, v):
                    answer += 1
            if t == 2:
                if dsu2.union(u, v):
                    answer += 1
        return answer

class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        graph = collections.defaultdict(list)
        for t, s, e in edges:
            graph[s].append((-t, e))
            graph[e].append((-t, s))
                
        root = [1]
        candidates = []
        for t, e in graph[1]:
            heapq.heappush(candidates, (t, e))
            
        A = set(root)
        B = set(root)
        
        ans = 0
        while candidates:
            t, e = heapq.heappop(candidates)
            if -t == 1:
                if e in A: continue
                A.add(e)
                ans += 1
                for nt, ne in graph[e]:
                    heapq.heappush(candidates, (nt, ne))
            elif -t == 2:
                if e in B: continue
                B.add(e)
                ans += 1
                for nt, ne in graph[e]:
                    heapq.heappush(candidates, (nt, ne))
            else:
                if e in A and e in B: continue
                A.add(e)
                B.add(e)
                ans += 1
                for nt, ne in graph[e]:
                    heapq.heappush(candidates, (nt, ne))
        
        if len(A) == n and len(B) == n: return len(edges) - ans
        return -1

class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        a_uf = UF(n+1)
        b_uf = UF(n+1)
        unwanted = 0

        for t, u, v in edges:    
            if t == 3:            
                # Alice and Bob
                if a_uf.find(u) == a_uf.find(v) and b_uf.find(u) == b_uf.find(v):
                    # both guys dont need
                    unwanted += 1
                else:
                    a_uf.union(u, v)
                    b_uf.union(u, v)
        
        for t, u, v in edges:
            #print((t, u, v))
            if t == 1:
                # Alice
                if a_uf.find(u) == a_uf.find(v):
                    # dont need this
                    unwanted += 1
                else:
                    a_uf.union(u, v)
        for t, u, v in edges:    
            if t == 2:
                # Bob
                if b_uf.find(u) == b_uf.find(v):
                    # dont need this
                    unwanted += 1
                else:
                    b_uf.union(u, v)
                
        if a_uf.size[a_uf.find(1)] < n or b_uf.size[b_uf.find(1)] < n:
            return -1
                
        return unwanted
    
class UF:
    def __init__(self, n):
        self.uf = [i for i in range(n)]
        self.size = [1] * n
    
    def find(self, u):
        while u != self.uf[u]:
            self.uf[u] = self.uf[self.uf[u]]
            u = self.uf[u]
        return u
    
    def union(self, u, v):
        rootU = self.find(u)
        rootV = self.find(v)
        if rootU != rootV:
            if self.size[rootU] > self.size[rootV]:
                self.size[rootU] += self.size[rootV]
                self.uf[rootV] = rootU
            else:
                self.size[rootV] += self.size[rootU]
                self.uf[rootU] = rootV
        
        

class Graph:
    def __init__(self):
        self.vertices = set()
        self.bi_edges = set()
        self.uni_edges = set()
    
    def find(self, parent, u):
        if parent[u] == u:
            return u
        return self.find(parent, parent[u])
    
    
    def union(self, parent, rank, u, v):
        
        u_root = self.find(parent, u)
        v_root = self.find(parent, v)
        
        if rank[u_root] < rank[v_root]:
            parent[u_root] = v_root
        elif rank[u_root] > rank[v_root]:
            parent[v_root] = u_root
        else:
            parent[v_root] = u_root
            rank[u_root] += 1
        return 0
        

class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        
        print("total edges", len(edges))
        # print(sorted(edges, key=lambda x : x[0]))
        graph1 = Graph()
        graph2 = Graph()
        
        #create 2 graphs
        for edge in edges:
            edge_type = edge[0]
            u = edge[1]
            v = edge[2]
            
            if edge[0]==1:
                graph1.vertices.add(u)
                graph1.vertices.add(v)
                graph1.uni_edges.add((u,v))
            elif edge[0] == 2:
                graph2.vertices.add(u)
                graph2.vertices.add(v)
                graph2.uni_edges.add((u,v))
            else:
                graph1.vertices.add(u)
                graph1.vertices.add(v)
                graph1.bi_edges.add((u,v))
                
                graph2.vertices.add(u)
                graph2.vertices.add(v)
                graph2.bi_edges.add((u,v))
        
        if len(graph1.vertices) < n or len(graph2.vertices) < n:
            return -1
        
        
        print("edges in graph",len(graph1.bi_edges)+len(graph1.uni_edges)+len(graph2.uni_edges))
        deleted = 0

        #detect cycle for given graph
        # return count of deleted uni_edges and cycle_creating bi_edges
        def minimum_spanning_tree(graph):
            parent = {}
            rank = {}
            
            for node in range(1,n+1):
                parent[node] = node
                rank[node] = 0
            
            cycle_bi_edges = set()
            for edge in graph.bi_edges:
                u = edge[0]
                v = edge[1]
                
                if graph.find(parent, u) != graph.find(parent, v):
                    graph.union(parent, rank, u, v)
                else:
                    cycle_bi_edges.add(edge)
            
            delete = 0
            for edge in graph.uni_edges:
                u = edge[0]
                v = edge[1]
                
                if graph.find(parent, u) != graph.find(parent, v):
                    graph.union(parent, rank, u, v)
                else:
                    delete += 1
            # print("span", delete, len(cycle_bi_edges))
            return delete, cycle_bi_edges
        
        result1 = minimum_spanning_tree(graph1)
        
        result2 = minimum_spanning_tree(graph2)
        
        deleted = deleted + result1[0] + result2[0]
        
        delete_bi = 0
        for edge in result1[1]:
            u = edge[0]
            v = edge[1]
            
            if (u,v) in result2[1]:
                delete_bi += 1
            
            if (v,u) in result2[1]:
                delete_bi +=1
        print(delete_bi)
        return deleted+delete_bi
                
                
                
            
        
class UF:
    def __init__(self):
        self.d = defaultdict(int)
        
    def findRoot(self, key):
        if self.d[key] > 0:
            self.d[key] = self.findRoot(self.d[key])
            return self.d[key]
        else:
            return key
        
    def mergeRoot(self, k1, k2):
        r1, r2 = self.findRoot(k1), self.findRoot(k2)  
        if r1 != r2:
            r1, r2 = min(r1, r2), max(r1, r2)
            self.d[r1] += self.d[r2]
            self.d[r2] = r1
        return r1
    
    def getSize(self, key):
        return self.d[self.findRoot(key)]

import heapq

class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        Edges = []
        [heapq.heappush(Edges, (-e[0], e[1], e[2])) for e in edges]
        uf1 = UF()
        uf2 = UF()
        
        ttl = 0
        while len(Edges) != 0:
            t, src, dst = heapq.heappop(Edges)
            
            if t == -1:
                if uf1.findRoot(src) == uf1.findRoot(dst):
                    ttl += 1
                    continue
                else:
                    uf1.d[uf1.mergeRoot(src, dst)] -= 1
            if t == -2:
                if uf2.findRoot(src) == uf2.findRoot(dst):
                    ttl += 1
                    continue
                else:
                    uf2.d[uf2.mergeRoot(src, dst)] -= 1
                    
            if t == -3:
                if uf1.findRoot(src) == uf1.findRoot(dst) and uf2.findRoot(src) == uf2.findRoot(dst):
                    ttl += 1
                    continue
                else:
                    uf1.d[uf1.mergeRoot(src, dst)] -= 1
                    uf2.d[uf2.mergeRoot(src, dst)] -= 1
                    
        if uf1.d[1] != - n + 1 or uf2.d[1] != - n + 1:
            return -1
        
        return ttl
class Solution:
    def maxNumEdgesToRemove(self, n: int, e: List[List[int]]) -> int:
        def union(UF, u, v):
            pu, pv = find(UF, u), find(UF, v)
            if pu != pv: UF[pv] = pu
        def find(UF, u):
            if UF[u] != u: UF[u] = find(UF, UF[u])
            return UF[u]         
        def check(UF, t):            
            UF = UF.copy()
            for tp, u, v in e:
                if tp != t: continue
                pu, pv = find(UF, u), find(UF, v)
                if pu == pv: self.ans += 1
                else: union(UF, u, v)
            return len(set(find(UF, u) for u in UF)) == 1
        
        self.ans, UF = 0, {u: u for u in range(1, n+1)}        
        for t, u, v in e:
            if t != 3: continue
            pu, pv = find(UF, u), find(UF, v)
            if pu == pv: self.ans += 1
            else: union(UF, u, v)        
        if not check(UF, 1) or not check(UF, 2): return -1        
        return self.ans                        
from typing import Tuple
class Solution:
    def __init__(self):
      self.roots = {}
      self.ranks = {}
      self.groups = 0
    
    def find(self, node_info: Tuple[int, int]):
      self.roots.setdefault(node_info, node_info)
      self.ranks.setdefault(node_info, 1)
      if self.roots[node_info] != node_info:
        self.roots[node_info] = self.find(self.roots[node_info])
      return self.roots[node_info]
    
    def union(self, node_info1, node_info2) -> bool:  # returns if the edge can be removed
      root1, root2 = self.find(node_info1), self.find(node_info2)
      if root1 != root2:
        self.groups -= 1
        self.roots[root2] = root1
        return False  # we can't remove this edge because it's used
      else:
        return True  # we can remove this edge because there already is a path for these 2 nodes.
    
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
      """
      Union find for alice and bob separately.
      """
      edges.sort(key=lambda edge: -edge[0])
      removes = 0
      self.groups = n * 2
      for tp, n1, n2 in edges:
        can_remove = False
        if tp == 1:
          can_remove = self.union((1, n1), (1, n2))
        elif tp == 2:
          can_remove = self.union((2, n1), (2, n2))
        else:
          can_remove1, can_remove2 = self.union((1, n1), (1, n2)), self.union((2, n1), (2, n2))
          can_remove = can_remove1 and can_remove2
        removes += (1 if can_remove else 0)  
      
      # If in the end both alice and alice have a single group, then return removed count
      return removes if self.groups == 2 else -1 
class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        
        n_a={}
        n_b={}
        for e in edges:
            if e[0]==1 or e[0]==3:
                if e[1] not in n_a:
                    n_a[e[1]]=[]
                n_a[e[1]].append(e[2])
                if e[2] not in n_a:
                    n_a[e[2]]=[]
                n_a[e[2]].append(e[1])
            if e[0]==2 or e[0]==3:
                if e[1] not in n_b:
                    n_b[e[1]]=[]
                n_b[e[1]].append(e[2])
                if e[2] not in n_b:
                    n_b[e[2]]=[]
                n_b[e[2]].append(e[1])
                
        visited=set()
        l=list(n_b.keys())
        start=l[0]
        visited.add(start)
        q=[start]
        while q:
            actual=q[0]
            del q[0]
            if actual in n_a:
                for nb in n_a[actual]:
                    if nb not in visited:
                        visited.add(nb)
                        q.append(nb)
        if len(visited)!=n:
            return -1
        
        visited=set()
        l=list(n_b.keys())
        start=l[0]
        visited.add(start)
        q=[start]
        while q:
            actual=q[0]
            del q[0]
            if actual in n_b:
                for nb in n_b[actual]:
                    if nb not in visited:
                        visited.add(nb)
                        q.append(nb)
        if len(visited)!=n:
            return -1        

        
        
        
        
        
        
        parent_a={}
        parent_b={}
        for i in range(1,n+1):
            parent_a[i]=i
            parent_b[i]=i
            
        def find_a(x):
            if parent_a[x]!=x:
                parent_a[x]=find_a(parent_a[x])
            return parent_a[x]
        def find_b(x):
            if parent_b[x]!=x:
                parent_b[x]=find_b(parent_b[x])
            return parent_b[x]
        def union_a(x,y):
            x=find_a(x)
            y=find_a(y)
            if x!=y:
                parent_a[x]=y
                return 0
            else:
                return 1
        def union_b(x,y):
            x=find_b(x)
            y=find_b(y)
            if x!=y:
                parent_b[x]=y
                return 0
            else:
                return 1
        count=0
        for e in edges:
            if e[0]==3:
                u1=union_a(e[1],e[2])
                print("1")
                u2=union_b(e[1],e[2])
                print("2")
                if u1==u2 and u1==1:
                    count+=1
                    
        print(parent_a)
        print(parent_b)
        
        for e in edges:
            if e[0]==1:    
                u1=union_a(e[1],e[2])
                print("3")
                if u1==1:
                    count+=1
            if e[0]==2:    
                u2=union_b(e[1],e[2])
                print("4")
                if u2==1:
                    count+=1
        
        return count
                    
        
        
class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        # Union find
        def find(i):
            if i != root[i]:
                root[i] = find(root[i])
            return root[i]

        def uni(x, y):
            x, y = find(x), find(y)
            if x == y: return 0
            root[x] = y
            return 1

        res = e1 = e2 = 0

        # Alice and Bob
        root = [i for i in range(n + 1)]
        for t, i, j in edges:
            if t == 3:
                if uni(i, j):
                    e1 += 1
                    e2 += 1
                else:
                    res += 1
        root0 = root[:]

        # only Alice
        for t, i, j in edges:
            if t == 1:
                if uni(i, j):
                    e1 += 1
                else:
                    res += 1

        # only Bob
        root = root0
        for t, i, j in edges:
            if t == 2:
                if uni(i, j):
                    e2 += 1
                else:
                    res += 1

        return res if e1 == e2 == n - 1 else -1
class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:

        temp = [[], [], []]
        for t, a, b in edges:
            temp[t - 1].append([a, b])
                
        p = list(range(n + 1))
        def find(i):
            if p[i] != i:
                p[i] = find(p[i])
            return p[i]
        def union(i, j):
            p[find(i)] = find(j)
        
        def helper(c):
            ans = 0
            for x, y in c:
                if find(x) == find(y):
                    ans += 1
                else:
                    union(x, y)
            return ans

        res = helper(temp[2])
        old = p[:]
        for c in temp[:2]:
            res += helper(c)
            p = old
        if sum(x == p[x] for x in range(1, n + 1)) == 1:
            return res
        return -1
        
        



class DSU:
    
    def __init__(self, N):
        self.parents = list(range(N))
        self.ranks = [1] * N
        self.size = 1
        
    def find(self, x):
        if self.parents[x] != x:
            self.parents[x] = self.find(self.parents[x])
        return self.parents[x]
    
    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py:
            return False
        if self.ranks[px] > self.ranks[py]:
            self.parents[py] = px
        elif self.ranks[py] > self.ranks[px]:
            self.parents[px] = py
        else:
            self.parents[px] = py
            self.ranks[py] += 1
        self.size += 1
        return True
        
    
class Solution:
    def maxNumEdgesToRemove(self, N: int, edges: List[List[int]]) -> int:
        uf1, uf2, res = DSU(N), DSU(N), 0
        
        for t, u, v in edges:
            if t == 3:
                if not uf1.union(u-1, v-1) or not uf2.union(u-1, v-1):
                    res += 1
                    
        for t, u, v in edges:
            if t == 1 and not uf1.union(u-1, v-1):
                res += 1
            elif t == 2 and not uf2.union(u-1, v-1):
                res += 1
        
        return res if uf1.size == N and uf2.size == N else -1
class DisjointSet:
    def __init__(self, elements):
        self.parent = [i for i in range(elements)]
        self.size = [1] * elements
    def find(self, value):
        if self.parent[value] != value:
            self.parent[value] = self.find(self.parent[value])
        return self.parent[value]
    def union(self, value1, value2):
        parent1, parent2 = self.find(value1), self.find(value2)
        if parent1 == parent2:
            return True
        if self.size[parent1] > self.size[parent2]:
            self.parent[parent2] = parent1
            self.size[parent1] += self.size[parent2]
        else:
            self.parent[parent1] = parent2
            self.size[parent2] += self.size[parent1]
        return False
class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        graph = [list() for i in range(n + 1)]
        for t, u, v in edges:
            graph[u].append([v, t])
            graph[v].append([u, t])
        def dfs(source, ty):
            nonlocal cnt
            cnt += 1
            vis[source] = 1
            for child, typer in graph[source]:
                if typer in [ty, 3] and not vis[child]:
                    dfs(child, ty)
        cnt = 0
        vis = [0] * (n + 1)
        dfs(1, 1)
        if cnt != n:
            return -1
        vis = [0] * (n + 1)
        cnt = 0
        dfs(1, 2)
        if cnt != n:
            return -1
        answer = 0
        dsu1, dsu2 = DisjointSet(n + 1), DisjointSet(n + 1)
        for t, u, v in edges:
            if t == 3:
                dsu1.union(u, v)
                if dsu2.union(u, v):
                    answer += 1
        for t, u, v in edges:
            if t == 1:
                if dsu1.union(u, v):
                    answer += 1
            if t == 2:
                if dsu2.union(u, v):
                    answer += 1
        return answer


class DSU:
    
    def __init__(self, a):
        self.par = {x:x for x in a}
    
    def merge(self, u, v):
        rootu = self.find(u)
        rootv = self.find(v)
        self.par[rootu] = rootv
    
    def find(self, u):
        if self.par[u] != u:
            self.par[u] = self.find(self.par[u])
        return self.par[u]

class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        # well its just two graphs, dude
        
        # MST would only require n-1 edges
        # this is ALWAYS TRUE
        
        # just process all of the type 3 first
        
        # because they will increase group size of both A and B
        
        # and then you can just do A and B separately
        
        # again, operate on both graphs
        
        dsu1 = DSU(range(1,n+1))
        dsu2 = DSU(range(1,n+1))
        
        both = [edge[1:] for edge in edges if edge[0] == 3]
        alice = [edge[1:] for edge in edges if edge[0] == 1]
        bob = [edge[1:] for edge in edges if edge[0] == 2]
        
        used = 0
        
        for u,v in both:
            if dsu1.find(u) != dsu1.find(v):
                dsu1.merge(u, v)
                dsu2.merge(u, v)
                used += 1
        
        for u,v in alice:
            if dsu1.find(u) != dsu1.find(v):
                dsu1.merge(u, v)
                used += 1
        
        for u,v in bob:
            if dsu2.find(u) != dsu2.find(v):
                dsu2.merge(u, v)
                used += 1
        
        if len(set(dsu1.find(u) for u in dsu1.par)) != 1 or len(set(dsu2.find(u) for u in dsu2.par)) != 1:
            return -1
            
        return len(edges) - used
class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        def find(i):
            while root[i]!=i:
                root[i]=root[root[i]]
                i=root[i]
            return i
        
        root=[i for i in range(n+1)]
        res, cnt1, cnt2=0, 0, 0
        for t, i, j in edges:
            if t==3:
                p1=find(i)
                p2=find(j)
                if p1==p2:                    
                    res+=1
                else:
                    root[p1]=p2
                    cnt1+=1
                    cnt2+=1
                    
        tmp=root[:]
        for t, i, j in edges:
            if t==1:
                p1=find(i)
                p2=find(j)
                if p1==p2:                    
                    res+=1
                else:
                    root[p1]=p2
                    cnt1+=1

        root=tmp[:]
        for t, i, j in edges:
            if t==2:
                p1=find(i)
                p2=find(j)
                if p1==p2:
                    res+=1
                else:
                    root[p1]=p2
                    cnt2+=1
        
        return res if cnt1==cnt2==n-1 else -1
class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        alice = DSU()
        bob = DSU()
        res = 0
        for t, u, v in edges:
            if t == 3:
                if alice.find(u) == alice.find(v):
                    res += 1
                else:
                    alice.union(u, v)
                    bob.union(u, v)
                    
        for t, u, v in edges:
            if t == 1:
                if alice.find(u) == alice.find(v):
                    res += 1
                else:
                    alice.union(u, v)
            if t == 2:
                if bob.find(u) == bob.find(v):
                    res += 1       
                else:
                    bob.union(u, v)
                    
        if max(bob.count.values()) != n or max(alice.count.values()) != n:
            return -1
        
        return res
        
class DSU:
    def __init__(self):
        self.father = {}
        self.count = {}
    
    def find(self, a):
        self.father.setdefault(a, a)
        self.count.setdefault(a, 1)
        if a != self.father[a]:
            self.father[a] = self.find(self.father[a])
        return self.father[a]
    
    def union(self, a, b):
        _a = self.find(a)
        _b = self.find(b)
        if _a != _b:
            self.father[_a] = self.father[_b]
            self.count[_b] += self.count[_a]
class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        from collections import defaultdict
        
        def collapse(mapping, old_graph):
            new_graph = defaultdict(set)
            for node in old_graph:
                if node not in mapping:
                    mapping[node] = node
            
            duplicate_count = 0
            for s in old_graph:
                for e in old_graph[s]:
                    mapped_s = mapping[s]
                    mapped_e = mapping[e]

                    if mapped_e in new_graph[mapped_s] or mapped_s == mapped_e:
                        duplicate_count += 1
                        # print('collpase_cost', s, 'to', e, 'is mapped to', mapped_s, 'to', mapped_e)
                        continue
                    
                    new_graph[mapped_s].add(mapped_e)
                    
            for node in old_graph:
                if len(new_graph[mapping[node]]) == 0:     
                    new_graph[mapping[node]] = []
                
            return new_graph, duplicate_count//2
        
       
        def find_connected(graph):
            # return mapping
            mapping = dict() 
            
            node_count = 0
            def dfs(cur_node, parent_node):
                nonlocal mapping, graph, node_count 
                mapping[cur_node] = parent_node
                node_count += 1
                for next_node in graph[cur_node]:
                    if next_node not in mapping:
                        dfs(next_node, parent_node)
                        
            edges_needed = 0            
            for node in graph:
                if node not in mapping:
                    node_count = 0
                    dfs(node, node)
                    edges_needed += node_count-1
                    
            return mapping, edges_needed
        

        a_graph = defaultdict(set)
        b_graph = defaultdict(set)
        share_graph = defaultdict(set)
        point_set = set()
        for t, s, e in edges:
            point_set.add(s)
            point_set.add(e)
            if t == 1:
                a_graph[s].add(e)
                a_graph[e].add(s)
            elif t == 2:    
                b_graph[s].add(e)
                b_graph[e].add(s)
            else:    
                share_graph[s].add(e)
                share_graph[e].add(s)
        for point in point_set:             
            if point not in a_graph:
                a_graph[point] = []
            if point not in b_graph:
                b_graph[point] = []
                
        costs = []
        share_mapping, edge_needed = find_connected(share_graph)
        new_share_graph, share_collapse = collapse(share_mapping, share_graph)
        costs.append(share_collapse - edge_needed)
        # print(share_collapse, edge_needed)
        
        for graph in [a_graph, b_graph]:
            # naming is ambiguous here
            
            new_graph, share_collpase = collapse(share_mapping, graph)
            # print(new_graph)
            sub_mapping, edge_needed = find_connected(new_graph)
            _, collpased = collapse(sub_mapping, new_graph)
            costs.append(share_collpase + collpased - edge_needed)
            # print(collpase, connect_cost, needed)
            # print(sub_mapping)
            if len(set(sub_mapping.values())) > 1: return -1
       
        return sum(costs)

class Node:
    def __init__(self, val):
        self.val = val
        self.parent = self
        self.rank = 0
        self.size = 1

class DisjointSet:
    def __init__(self, n):
        self.sets = {x: Node(x) for x in range(1, n+1)}
        self.disjointSet = {x: self.sets[x] for x in range(1, n+1)}
        
    def findSet(self, x):
        y = x
        while y.parent != y:
            y = y.parent
        z = x
        while z != y:
            tmp = z.parent
            z.parent = y
            z = tmp
        return y
    
    def link(self, x_val, y_val):
        x = self.sets[x_val]
        y = self.sets[y_val]
        if x.rank > y.rank:
            y.parent = x
            if y_val in self.disjointSet and y_val != x_val:
                del self.disjointSet[y_val]         
                x.size += y.size
        elif x.rank < y.rank:
            x.parent = y
            if x_val in self.disjointSet and y_val != x_val:
                del self.disjointSet[x_val]
                y.size += x.size
        else:
            x.parent = y
            y.rank += 1
            if x_val in self.disjointSet and y_val != x_val:
                del self.disjointSet[x_val]
                y.size += x.size
        
    def union(self, x, y):
        self.link(self.findSet(self.sets[x]).val, self.findSet(self.sets[y]).val)
         

class Solution:
    
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        alice = DisjointSet(n)
        bob = DisjointSet(n)
        both = DisjointSet(n)
        type1 = 0
        type2 = 0
        type3 = 0
        for edge in edges:
            etype, a, b = edge
            if etype == 1:
                alice.union(a, b)
                type1 += 1
            elif etype == 2:
                bob.union(a, b)
                type2 += 1
            else:
                alice.union(a, b)
                bob.union(a, b)
                both.union(a, b)
                type3 += 1
        # print(alice.disjointSet)
        # print(bob.disjointSet)
        # print(both.disjointSet)
        # print(len(bob.disjointSet))
        
        if len(alice.disjointSet) != 1 or len(bob.disjointSet) != 1:
            return -1
        tmp = 0
        for key, val in list(both.disjointSet.items()):
            tmp += val.size - 1
        return type3 - tmp + type1 - (len(both.disjointSet) - 1) + type2 - (len(both.disjointSet) - 1)
        
            
        

                

from collections import defaultdict
class UnionFind:
    def __init__(self, iterable=None):
        """u521du59cbu5316u7236u5b50u5173u7cfbu6620u5c04u3002u82e5u6307u5b9aiterableuff0cu5219u521du59cbu5316u5176u81eau8eab"""
        self.cnt = defaultdict(lambda: 1)
        self.f = {}
        for a in iterable or []:
            self.f[a] = a
            self.cnt[a] = 1

    def size(self, a=None):
        """u8fd4u56deau96c6u5408u5927u5c0fu3002u82e5u4e0du6307u5b9aauff0cu5219u8fd4u56deu96c6u5408u7684u4e2au6570"""
        if a is not None:
            return self.cnt[self.find(a)]
        else:
            return sum(a == self.f[a] for a in self.f)

    def same(self, a, b):
        """u5224u65ada,bu662fu5426u540cu4e00u96c6u5408"""
        return self.find(a) == self.find(b)

    def find(self, a):
        """u67e5u627eau7684u6839"""
        if self.f.get(a, a) == a:
            self.f[a] = a
            return a
        self.f[a] = self.find(self.f[a])
        return self.f[a]

    def merge(self, a, b):
        """u5408u5e76au5230bu7684u96c6u5408"""
        ra = self.find(a)
        rb = self.find(b)
        if ra != rb:
            self.f[ra] = rb
            self.cnt[rb] += self.cnt[ra]

class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        un = UnionFind(range(1, n+1))
        ans = 0
        for t, a, b in edges:
            if t == 3:
                if un.same(a, b):
                    ans += 1
                else:
                    un.merge(a, b)

        un1 = deepcopy(un)
        for t, a, b in edges:
            if t == 1:
                if un1.same(a, b):
                    ans += 1
                else:
                    un1.merge(a, b)

        un2 = deepcopy(un)
        for t, a, b in edges:
            if t == 2:
                if un2.same(a, b):
                    ans += 1
                else:
                    un2.merge(a, b)
        
        return ans if un1.size() == un2.size() == 1 else -1
class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        root = [i for i in range(n+1)]
        def find(x):
            if root[x] != x:
                root[x] = find(root[x])
            return root[x]

        def uni(x, y):
            x, y = find(x), find(y)
            if x == y: return 0
            root[x] = y
            return 1

        ans = e1 = e2 = 0
        for t, i, j in edges:
            if t == 3:
                if uni(i, j):
                    e1 += 1
                    e2 += 1
                else:
                    ans += 1

        root_origin = root
        root = root_origin[:]
        for t, i , j in edges:
            if t == 1:
                if uni(i, j):
                    e1 += 1
                else:
                    ans += 1
        
        root = root_origin[:]
        for t, i, j in edges:
            if t == 2:
                if uni(i, j):
                    e2 += 1
                else:
                    ans += 1

        if e1 == n -1 and e2 == n -1:
            return ans
        else:
            return -1
class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        # Union find
        def find(i):
            if i != root[i]:
                root[i] = find(root[i])
            return root[i]

        def uni(x, y):
            x, y = find(x), find(y)
            if x == y: return 0
            root[x] = y
            return 1

        res = e1 = e2 = 0

        # Alice and Bob
        root = list(range(n + 1))
        for t, i, j in edges:
            if t == 3:
                if uni(i, j):
                    e1 += 1
                    e2 += 1
                else:
                    res += 1
        root0 = root[:]

        # only Alice
        for t, i, j in edges:
            if t == 1:
                if uni(i, j):
                    e1 += 1
                else:
                    res += 1

        # only Bob
        root = root0
        for t, i, j in edges:
            if t == 2:
                if uni(i, j):
                    e2 += 1
                else:
                    res += 1

        return res if e1 == e2 == n - 1 else -1


class Solution:
    def maxNumEdgesToRemove(self, n: int, e: List[List[int]]) -> int:
        pa=[0]*(n+1)
        pb=[0]*(n+1)
        for i in range(n+1):
            pa[i]=i
            pb[i]=i
        
        def p(x):
            if x==pa[x]:
                return x
            pa[x]=p(pa[x])
            return pa[x]
        
        def pp(x):
            if x==pb[x]:
                return x
            pb[x]=pp(pb[x])
            return pb[x]
        ans=0
        for x in e:
            if x[0]==3:
                q=p(x[1])
                w=p(x[2])
                
                r=pp(x[1])
                t=pp(x[2])
                
                if (q==w and r==t):
                    ans+=1
                else:
                    pa[q]=w
                    pb[r]=t
                    
        for x in e:
            
            if x[0]==1:
                q=p(x[1])
                w=p(x[2])
                
                
                
                if (q==w ):
                    ans+=1
                else:
                    pa[q]=w
        
            if x[0]==2:
                r=pp(x[1])
                t=pp(x[2])
        
                if (r==t):
                    ans+=1
                else:
                    pb[r]=t
        tt=0
        for i in range(1,n+1):
            if i==pa[i]:
                tt+=1
            if i==pb[i]:
                tt+=1
        
        if tt!=2:
            return -1
        return ans
                
            
        
        
        
        

class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        roots1 = [i for i in range(n + 1)]
        roots2 = [i for i in range(n + 1)]
        
        def find_root(roots, i):
            while roots[i] != i:
                roots[i] = roots[roots[i]]
                i = roots[i]
            return i
        
        ans = 0
        
        edges.sort(reverse=True)
        
        for edge in edges:
            if edge[0] == 3:
                if find_root(roots1, edge[1]) == find_root(roots1, edge[2]) and find_root(roots2, edge[1]) == find_root(roots2, edge[2]):
                    ans += 1
                else:
                    roots1[find_root(roots1, edge[2])] = find_root(roots1, edge[1])
                    roots2[find_root(roots2, edge[2])] = find_root(roots2, edge[1])
            elif edge[0] == 1:
                if find_root(roots1, edge[1]) == find_root(roots1, edge[2]):
                    ans += 1
                else:
                    roots1[find_root(roots1, edge[2])] = find_root(roots1, edge[1])
            else:
                if find_root(roots2, edge[1]) == find_root(roots2, edge[2]):
                    ans += 1
                else:
                    roots2[find_root(roots2, edge[2])] = find_root(roots2, edge[1])
        
        def check(roots):
            g = set()
            for i in range(1, len(roots)):
                g.add(find_root(roots, i))
            return 0 if len(g) > 1 else 1
        
        return -1 if not check(roots1) or not check(roots2) else ans
                    
        
        

import copy

def union(subsets, u, v):
    uroot = find(subsets, u)
    vroot = find(subsets, v)
    
    if subsets[uroot][1] > subsets[vroot][1]:
        subsets[vroot][0] = uroot
    if subsets[vroot][1] > subsets[uroot][1]:
        subsets[uroot][0] = vroot
    if subsets[uroot][1] == subsets[vroot][1]:
        subsets[vroot][0] = uroot
        subsets[uroot][1] += 1
    

def find(subsets, u):
    if subsets[u][0] != u:
        subsets[u][0] = find(subsets, subsets[u][0])
    return subsets[u][0]


class Solution:
    #kruskal's
    #1 is alice and 2 is bob
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        
        subsets1 = ['1 index'] + [[x+1,0] for x in range(n)] #Alice's unionfind
        subsets2 = ['1 index'] + [[x+1,0] for x in range(n)] #Bob's unionfind
        
        #edges = sorted(edges, key= lambda e: -e[0])
        e = 0 #number of total edges used
        e1 = 0 #number of edges for Alice
        e2 = 0 #number of edges for Bob
        i = 0 #track position in edges list
        
        #start with type 3 edges
        while e < n - 1:
            if i == len(edges): 
                i = 0
                break
            typ, u, v = edges[i]
            if typ != 3: 
                i += 1
                continue
            if find(subsets1, u) != find(subsets1, v):
                union(subsets1, u, v)
                e += 1
            
            i += 1
        
        #everything that was done to Alice applies to Bob
        e1 = e
        e2 = e
        subsets2 = copy.deepcopy(subsets1)
        i=0
        #once done with shared edges, do Bob's
        while e2 < n-1:
            if i == len(edges): 
                i = 0
                break
            typ, u, v = edges[i]
            if typ != 2: 
                i+=1
                continue
            if find(subsets2, u) != find(subsets2, v):
                union(subsets2, u, v)
                e += 1
                e2 += 1
            i += 1
        
        if e2 < n - 1: 
            return -1 #if we've used all edges bob can use (types 2 and 3) and he still can't reach all nodes, ur fucked
        
        i=0
        #now finish Alice's MST
        while e1 < n-1:
            if i == len(edges): 
                return -1
        
            typ, u, v = edges[i]
            if typ != 1: 
                i += 1
                continue
            
            if find(subsets1, u) != find(subsets1, v):
                union(subsets1, u, v)
                e += 1
                e1 += 1
            i += 1
            
        return len(edges) - e
            
            
            
            
        
        

class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        t1, t2, t3 = [], [], []
        for t,s,e in edges:
            if t==1: t1.append([s,e])
            elif t==2: t2.append([s,e])
            elif t==3: t3.append([s,e])
        # print(t1,t2,t3)
        
        def find(x, arr):
            if x==arr[x]: return x
            else: 
                arr[x] = find(arr[x], arr)
                return arr[x]
                
        ans = 0
        alice, bob = [i for i in range(n+1)], [i for i in range(n+1)]
        
        for s,e in t3:
            g1, g2 = find(s,alice), find(e,alice)
            if g1!=g2:
                alice[g1] = g2
                bob[g1] = g2
            else: ans += 1
        # print(alice, bob)
        for s,e in t1:
            g1, g2 = find(s,alice), find(e,alice)
            if g1!=g2:
                alice[g1] = g2
            else: ans += 1
        # print(alice, bob, ans)
        for s,e in t2:
            g1, g2 = find(s,bob), find(e,bob)
            if g1!=g2:
                bob[g1] = g2
            else: ans += 1
        
        root1 = find(alice[1], alice)
        for i in range(1, len(alice)):
            if find(i, alice)!=root1: return -1
        root2 = find(bob[1], bob)
        for i in range(1, len(bob)):
            if find(i, bob)!=root2: return -1
        return ans

class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        A=[i+1 for i in range(n)]
        B=[i+1 for i in range(n)]
        cA=n
        cB=n
        edges.sort(reverse=True,key=lambda x:(x[0]))
        i=0
        m=len(edges)
        ans=0
        while i<m and edges[i][0]==3:
            l1=find(A,edges[i][1])
            l2=find(A,edges[i][2])
            if l1==l2:
                ans+=1
            else:
                #mi=min(l1,l2)
                #ma=max(l1,l2)
                A[l1-1]=-l2
                #print(l1,l2)
                #print(A)
                B[l1-1]=-l2
                cA-=1
                cB-=1
            if cA==1:
                return m-i-1+ans
            i+=1
        j=i
        while j<m and edges[j][0]==2:
            l1=find(B,edges[j][1])
            l2=find(B,edges[j][2])
            if l1==l2:
                ans+=1
            else:
                #mi=min(l1,l2)
                #ma=max(l1,l2)
                B[l1-1]=-l2
                cB-=1
            if cB==1:
                while j+1<m and edges[j+1][0]==2:
                    j+=1
                    ans+=1
                j+=1
                break
            else:
                j+=1
        if cB!=1:
            return -1
        while j<m:
            l1=find(A,edges[j][1])
            l2=find(A,edges[j][2])
            if l1==l2:
                ans+=1
            else:
                #mi=min(l1,l2)
                #ma=max(l1,l2)
                A[l1-1]=-l2
                cA-=1
            if cA==1:
                return m-j-1+ans
            j+=1
        
        return -1
        
            
            
            
def find(ll,x):
    if ll[x-1]>0:
        return ll[x-1]
    else:
        ll[x-1]=-find(ll,-ll[x-1])
        return(-ll[x-1])

class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        graphs = [collections.defaultdict(list) for _ in range(3)]
        x = [0]*3
        for c, i, j in edges:
            graphs[c-1][i].append((-c, j))
            graphs[c-1][j].append((-c, i))
            x[c-1] += 1
        e = graphs[2][1] + graphs[0][1]
        heapq.heapify(e)
        treeset = set([1])
        ans = 0
        while e:
            c, y = heapq.heappop(e)
            if y not in treeset:
                treeset.add(y)
                if c == -3:
                    ans += 1
                for item in graphs[2][y]:
                    heapq.heappush(e, item)
                for item in graphs[0][y]:
                    heapq.heappush(e, item)
        # print(treeset)
        # print(ans)
        if len(treeset) != n:
            return -1
        e = graphs[2][1] + graphs[1][1]
        heapq.heapify(e)
        treeset = set([1])
        # ans = 0
        # print(e)
        while e:
            c, y = heapq.heappop(e)
            if y not in treeset:
                treeset.add(y)
                # if c == -3:
                #     ans += 1
                for item in graphs[2][y]:
                    heapq.heappush(e, item)
                for item in graphs[1][y]:
                    heapq.heappush(e, item)
            # print(e)
        # print(treeset)
        if len(treeset) != n:
            return -1
        # print(ans)
        return len(edges)+ans -2*(n-1)


class Solution(object):
    def maxNumEdgesToRemove(self, n, edges):
        """
        :type n: int
        :type edges: List[List[int]]
        :rtype: int
        """

        ga = [i for i in range(n)]
        gb = [i for i in range(n)]

        res = 0

        e1 = []
        e2 = []
        for i in range(len(edges)):
            if edges[i][0] == 1:
                e1.append(edges[i])
                continue
            if edges[i][0] == 2:
                e2.append(edges[i])
                continue

            x = edges[i][1]-1
            y = edges[i][2]-1

            gx = self.get_group(ga, x)
            gy = self.get_group(ga, y)
            if gx == gy:
                res += 1
            ga[gx] = min(gx, gy)
            ga[gy] = min(gx, gy)
            ga[x] = min(gx, gy)
            ga[y] = min(gx, gy)
            gb[gx] = min(gx, gy)
            gb[gy] = min(gx, gy)
            gb[x] = min(gx, gy)
            gb[y] = min(gx, gy)

        for e in e1:
            x = e[1]-1
            y = e[2]-1
            gx = self.get_group(ga, x)
            gy = self.get_group(ga, y)
            if gx == gy:
                res += 1
            ga[gx] = min(gx, gy)
            ga[gy] = min(gx, gy)
            ga[x] = min(gx, gy)
            ga[y] = min(gx, gy)

        for e in e2:
            x = e[1]-1
            y = e[2]-1
            gx = self.get_group(gb, x)
            gy = self.get_group(gb, y)
            if gx == gy:
                res += 1
            gb[gx] = min(gx, gy)
            gb[gy] = min(gx, gy)
            gb[x] = min(gx, gy)
            gb[y] = min(gx, gy)

        ga0 = self.get_group(ga, 0)
        gb0 = self.get_group(gb, 0)
        for i in range(1, n):
            gai = self.get_group(ga, i)
            gbi = self.get_group(gb, i)
            if ga0 != gai or gb0 != gbi:
                return -1
        return res

    def get_group(self, g, i):

        gi = g[i]
        pre = i
        while gi != g[gi]:
            g[pre] = g[gi]
            pre = gi
            gi = g[gi]

        return gi
class DSU:
    def __init__(self, N):
        self.par = list(range(N))
        self.sz = [1] * N

    def find(self, x):
        if self.par[x] != x:
            self.par[x] = self.find(self.par[x])
        return self.par[x]

    def union(self, x, y):
        xr, yr = self.find(x), self.find(y)
        if xr == yr:
            return False
        if self.sz[xr] < self.sz[yr]:
            xr, yr = yr, xr
        self.par[yr] = xr
        self.sz[xr] += self.sz[yr]
        return True
    
    def size(self, x):
        return self.sz[self.find(x)]

class Solution:
    def maxNumEdgesToRemove(self, n: int, a: List[List[int]]) -> int:
        dsu1, dsu2 = DSU(n), DSU(n)
        d = {}
        
        for t, x, y in a:
            x -= 1
            y -= 1
            if t in d:
                d[t].append([x, y])
            else:
                d[t] = [[x,y]]
                
        ans = 0
        
        if 3 in d:
            for i in d[3]:
                x, y = i
                dsu2.union(x, y)
                if not dsu1.union(x, y):
                    ans += 1
        
        if 1 in d:
            for i in d[1]:
                x, y = i
                if not dsu1.union(x, y):
                    ans += 1
                
        if 2 in d:
            for i in d[2]:
                x, y = i
                if not dsu2.union(x, y):
                    ans += 1
        
        if dsu1.size(0) != n or dsu2.size(0) != n:
            return -1
        return ans
class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        type_1, type_2, type_3 = [], [], []
        for type, a, b in edges:
            if type == 3:
                type_3.append((a, b))
            elif type == 1:
                type_1.append((a,b))
            elif type == 2:
                type_2.append((a, b))
        answer = 0
        b_u = None
        def helper(edges, u=None):
            nonlocal b_u
            u = Union(n, u)
            nonlocal answer
            for a, b in edges:
                if not u.union(a, b):
                    answer += 1
            b_u = u.backup()
            if not u.isConnected():
                return False
            return True
        helper(type_3)
        bb_u = copy.deepcopy(b_u)
        if not helper(type_1, copy.deepcopy(bb_u)):
            return -1
        if not helper(type_2, copy.deepcopy(bb_u)):
            return -1
        return answer


class Union:
    def __init__(self, n, p=None):
        self.n = n
        self.p = p if p else {i: i for i in range(1, n+1)}

    def backup(self):
        return self.p

    def union(self, p_a, b):
        a = p_a
        while self.p[a] != a:
            a = self.p[a]
        while self.p[b] != b:
            b = self.p[b]
        self.p[p_a] = a
        if a == b:
            return False
        else:
            self.p[b] = a
            return True

    def isConnected(self):
        return sum(i == self.p[i] for i in range(1, self.n+1)) == 1
class unionfindset:
    def __init__(self,n=0):
        self.par={}
        self.rank={}
        self.count=n
        for i in range(1,1+n):
            self.par[i]=i
            self.rank[i]=1
        
    def find(self,u):
        
        if u!=self.par[u]:
            self.par[u]=self.find(self.par[u])
        return self.par[u]
    
    def union(self,u,v):
        pu,pv=self.find(u),self.find(v)
        if pu==pv:return False
        if self.rank[pu]<self.rank[pv]:
            self.par[pu]=pv
        elif self.rank[pv]<self.rank[pu]:
            self.par[pv]=pu
        else:
            self.par[pv]=pu
            self.rank[pu]+=1
        self.count-=1
        return True
            

class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        
        
        unf1=unionfindset(n)
        unf2=unionfindset(n)
        out=0
        for i,u,v in edges:
            if i==1 or i==2:continue
            if not unf1.union(u,v) or not unf2.union(u,v):out+=1
            
            
            
        
        for i,u,v in edges:
            if i==1:
                #print(u,v)
                if not unf1.union(u,v):out+=1
                
            elif i==2:
                #print(u,v)
                if not unf2.union(u,v):out+=1
                
       
        if unf1.count!=1 or unf2.count!=1:return -1
        
        return out

        

class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        # Union find
        def find(i):
            if i != root[i]:
                root[i] = find(root[i])
            return root[i]

        def uni(x, y):
            x, y = find(x), find(y)
            if x == y: 
                return False
            root[x] = y
            return True

        res = e1 = e2 = 0
        
        
        root = list(range(n+1))
        
        print(root)

        
        for typ, u, v in edges:
            if typ == 3:
                if uni(u, v):
                    e1 += 1
                    e2 += 1
                else:
                    res += 1
                    
        root_copy = root.copy()
        
        for typ, u, v in edges:
            if typ == 1:
                if uni(u, v):
                    e1 += 1
                else:
                    res += 1

        root = root_copy
        
        for typ, u, v in edges:
            if typ == 2:
                if uni(u, v):
                    e2 += 1
                else:
                    res += 1
                    
        return res if e1 == e2 == n - 1 else -1
class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        roots1 = [i for i in range(n + 1)]
        roots2 = [i for i in range(n + 1)]
        
        def find_root(roots, i):
            while roots[i] != i:
                roots[i] = roots[roots[i]]
                i = roots[i]
            return i
        
        ans = 0
        
        edges.sort(reverse=True)
        
        for edge in edges:
            if edge[0] == 3:
                if find_root(roots1, edge[1]) == find_root(roots1, edge[2]) and find_root(roots2, edge[1]) == find_root(roots2, edge[2]):
                    ans += 1
                else:
                    roots1[find_root(roots1, edge[2])] = find_root(roots1, edge[1])
                    roots2[find_root(roots2, edge[2])] = find_root(roots2, edge[1])
            elif edge[0] == 1:
                if find_root(roots1, edge[1]) == find_root(roots1, edge[2]):
                    ans += 1
                else:
                    roots1[find_root(roots1, edge[2])] = find_root(roots1, edge[1])
            else:
                if find_root(roots2, edge[1]) == find_root(roots2, edge[2]):
                    ans += 1
                else:
                    roots2[find_root(roots2, edge[2])] = find_root(roots2, edge[1])
        
        g1 = set()
        for i in range(1, len(roots1)):
            g1.add(find_root(roots1, i))
        if len(g1) > 1:
            return -1
        
        g2 = set()
        for i in range(1, len(roots2)):
            g2.add(find_root(roots2, i))
        if len(g2) > 1:
            return -1
        
        return ans
                    
        
        

class Solution:
    def num_components(self, n, edges):
        parents = [0 for _ in range(n+1)]
        components = n
        def root(u):
            if parents[u] == 0:
                return u
            r = root(parents[u])
            parents[u] = r
            return r
        for u, v in edges:
            a, b = root(u), root(v)
            if a == b:
                continue
            else:
                components -= 1
                parents[a] = b
        return components
    
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        A = [edge[1:] for edge in edges if edge[0] == 1]
        B = [edge[1:] for edge in edges if edge[0] == 2]
        C = [edge[1:] for edge in edges if edge[0] == 3]
        a = self.num_components(n, A + C)
        b = self.num_components(n, B + C)
        c = self.num_components(n, C)
        #print(f'a={a}, b={b}, c={c}')
        if a > 1 or b > 1:
            return -1
        deleted_common_edges = len(C) - n + c
        deleted_alice_edges = len(A) + len(C) - n + 1 - deleted_common_edges
        deleted_bob_edges = len(B) + len(C) - n + 1 - deleted_common_edges
        #print(deleted_common_edges, deleted_alice_edges, deleted_bob_edges)
        return deleted_common_edges + deleted_alice_edges + deleted_bob_edges
class Solution:
    def maxNumEdgesToRemove(self, N: int, E: List[List[int]], same = 0) -> int:
        E = [[_, u - 1, v - 1] for _, u, v in E]
        A = [i for i in range(N)]
        B = [i for i in range(N)]
        # def find(P, x): P[x] = x if P[x] == x else find(P, P[x])
        def find(P, x): P[x] = P[x] if P[x] == x else find(P, P[x]); return P[x]
        def union(P, a, b):
            a = find(P, a)
            b = find(P, b)
            if a == b:
                return 1
            P[a] = b  # arbitrary choice
            return 0
        for type, u, v in E:
            if type == 3: same += union(A, u, v) | union(B, u, v)
        for type, u, v in E:
            if type == 1: same += union(A, u, v)
            if type == 2: same += union(B, u, v)
        parentA = find(A, 0)
        parentB = find(B, 0)
        return same if all(parentA == find(A, x) for x in A) and all(parentB == find(B, x) for x in B) else -1
        

class UnionFind():
    def __init__(self, n):
        self.label = list(range(n))
        self.sz = [1] * n
        
    def find(self, p):
        while p != self.label[p]:
            self.label[p] = self.label[self.label[p]]
            p = self.label[p]
        return p
    
    def union(self, p, q):
        proot, qroot = self.find(p), self.find(q)
        if self.sz[proot] >= self.sz[qroot]:
            self.label[qroot] = proot
            self.sz[proot] += self.sz[qroot]
        else:
            self.label[proot] = qroot
            self.sz[qroot] += self.sz[proot]
    
    def size(self, p):
        return self.sz[self.find(p)]

class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        """
        MST, union find?
        
        construct uf based on type 3
        try type 1 and type2 separate
        for a type:
          pick from all edges to achieve reachability
        """
        
        def mst(uf, edges):
            result = 0
            for a, b in edges:
                if uf.find(a - 1) == uf.find(b - 1):
                    continue
                uf.union(a - 1, b - 1)
                result += 1
            if uf.size(0) != n:
                return -1
            return result
        
        def commonUf(edges):
            uf = UnionFind(n)
            edgesNeeded = 0
            for a, b in edges:
                if uf.find(a - 1) == uf.find(b - 1):
                    continue
                uf.union(a - 1, b - 1)
                edgesNeeded += 1
            return uf, edgesNeeded
        
        commonEdges = [(edge[1], edge[2]) for edge in edges if edge[0] == 3]
        
        uf1, commonEdgesNeeded = commonUf(commonEdges)
        mst1 = mst(uf1, [(edge[1], edge[2]) for edge in edges if edge[0] == 1])
        uf2, _ = commonUf(commonEdges)
        mst2 = mst(uf2, [(edge[1], edge[2]) for edge in edges if edge[0] == 2])
        
        if mst1 == -1 or mst2 == -1:
            return -1
        return len(edges) - commonEdgesNeeded - mst1 - mst2
        
class UnionFind:
    def __init__(self, ):
        self._parent = {}
        self._size = {}
    
    def union(self, a, b):
        a = self.find(a)
        b = self.find(b)
        if a == b:
            return False
        if self._size[a] < self._size[b]:
            a, b = b, a
        self._parent[b] = a
        self._size[a] += self._size[b]
        return True
    
    def find(self, x):
        if x not in self._parent:
            self._parent[x] = x
            self._size[x] = 1
            return x
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]
            x = self._parent[x]
        return self._parent[x]
    
    def size(self, x):
        return self._size[self.find(x)]
        
class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        ufa = UnionFind()
        ufb = UnionFind()
        ufa2 = UnionFind()
        ufb2 = UnionFind()
        count = 0
        
        for t, u, v in edges:
            if t == 1:
                ufa.union(u, v)
            elif t == 2:
                ufb.union(u, v)
            else:
                ufa.union(u, v)
                ufb.union(u, v)
                ufa2.union(u, v)
                count += int(ufb2.union(u, v))
        
        if ufa.size(1) != n or ufb.size(1) != n:
            return -1
        
        for t, u, v in edges:
            if t == 1:
                count += ufa2.union(u, v)
            elif t == 2:
                count += ufb2.union(u, v)
        
        return len(edges) - count
class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        uf1 = list(range(n+1))
        uf2 = list(range(n+1))
        uf3 = list(range(n+1))
        
        def find(uf, x):
            if x != uf[x]:
                uf[x] = find(uf, uf[x])
            return uf[x]
        
        def union(uf, x, y):
            uf[find(uf, x)] = find(uf, y)
            
        res = 0
        for t, u, v in edges:
            if t == 3 and find(uf3, u) != find(uf3, v):
                union(uf1, u, v)
                union(uf2, u, v)
                union(uf3, u, v)
                res += 1
                
        for t, u, v in edges:
            if t == 1 and find(uf1, u) != find(uf1, v):
                union(uf1, u, v)
                res += 1
            if t == 2 and find(uf2, u) != find(uf2, v):
                union(uf2, u, v)
                res += 1
                    
        if len({find(uf1, i) for i in range(1,n+1)}) > 1:
            return -1
        if len({find(uf2, i) for i in range(1,n+1)}) > 1:
            return -1
                    
        return len(edges) - res
"""
N connected nodes, M edges (M <= N*(N-1)//2)
what is the minimum number of edges to connect N nodes?



"""



class UnionFindSet:
    def __init__(self, n):
        self.parents = list(range(n))
        self.ranks = [1] * n
        self.size = 1
        
    def find(self, u):
        if u != self.parents[u]:
            self.parents[u] = self.find(self.parents[u])
        return self.parents[u]
    
    def union(self, u, v):
        pu, pv = self.find(u), self.find(v)
        if pu == pv:
            return False
        if self.ranks[pu] > self.ranks[pv]:
            self.parents[pv] = pu
        elif self.ranks[pv] > self.ranks[pu]:
            self.parents[pu] = pv
        else:
            self.parents[pu] = pv
            self.ranks[pv] += 1
        self.size += 1
        return True
    
    
class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        uf1 = UnionFindSet(n)
        uf2 = UnionFindSet(n)
        res = 0
        
        for t, u, v in edges:
            if t != 3:
                continue
            if not uf1.union(u - 1, v - 1) or not uf2.union(u - 1, v - 1):
                res += 1
        
        for t, u, v in edges:
            if t == 1 and not uf1.union(u - 1, v - 1):
                res += 1
            elif t == 2 and not uf2.union(u - 1, v - 1):
                res += 1
   
        return res if uf1.size == n and uf2.size == n else -1




        
        
        
        
        
        
class UnionFind:
    def __init__(self, n):
        self.state = [-1] * n
        # self.size_table = [1] * n
        # cntu306fu30b0u30ebu30fcu30d7u6570
        self.cnt = n

    def root(self, u):
        v = self.state[u]
        if v < 0: return u
        self.state[u] = res = self.root(v)
        return res

    def merge(self, u, v):
        ru = self.root(u)
        rv = self.root(v)
        if ru == rv: return
        du = self.state[ru]
        dv = self.state[rv]
        if du > dv: ru, rv = rv, ru
        if du == dv: self.state[ru] -= 1
        self.state[rv] = ru
        self.cnt -= 1
        # self.size_table[ru] += self.size_table[rv]
        return

    # u30b0u30ebu30fcu30d7u306eu8981u7d20u6570
    # def size(self, u):
    #     return self.size_table[self.root(u)]

class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        au=UnionFind(n)
        bu=UnionFind(n)
        ee=[[] for _ in range(3)]
        for t,u,v in edges:
            ee[t-1].append((u-1,v-1))

        ans=0

        for u,v in ee[2]:
            if au.root(u)==au.root(v):
                ans+=1
                continue
            au.merge(u,v)
            bu.merge(u,v)

        for u,v in ee[0]:
            if au.root(u)==au.root(v):
                ans+=1
                continue
            au.merge(u,v)

        for u,v in ee[1]:
            if bu.root(u)==bu.root(v):
                ans+=1
                continue
            bu.merge(u,v)

        if au.cnt==1 and bu.cnt==1:return ans
        else:return -1

class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        parent = list(range(n + 1))
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x]) 
            return parent[x]
        
        def union(x , y):
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[rx] = ry
                return 1
            else:
                return 0
        
        res, e1, e2 = 0, 0, 0
        
        for t, i, j in edges:
            if t == 3:
                if union(i, j):
                    e1 += 1
                    e2 += 1
                else:
                    res += 1
                    
        parent1 = parent[:]
        
        for t, i ,j in edges:
            if t == 1:
                if union(i, j):
                    e1 += 1
                else:
                    res += 1
        
        parent = parent1
        for t, i, j in edges:
            if t == 2:
                if union(i, j):
                    e2 += 1
                else:
                    res += 1
        return res if e1 == e2 == n - 1 else -1
        

from typing import List
class Solution:
    def maxNumEdgesToRemove(self, n, edges):
        # Union find
        def find(i):
            if i != root[i]:
                root[i] = find(root[i])
            return root[i]

        def uni(x, y):
            x, y = find(x), find(y)
            if x == y: return 0
            root[x] = y
            return 1

        res = e1 = e2 = 0

        # Alice and Bob
        root = list(range(n + 1))
        for t, i, j in edges:
            if t == 3:
                if uni(i, j):
                    e1 += 1
                    e2 += 1
                else:
                    res += 1
        root0 = root[:]

        # only Alice
        for t, i, j in edges:
            if t == 1:
                if uni(i, j):
                    e1 += 1
                else:
                    res += 1

        # only Bob
        root = root0
        for t, i, j in edges:
            if t == 2:
                if uni(i, j):
                    e2 += 1
                else:
                    res += 1

        return res if e1 == e2 == n - 1 else -1
import copy

class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        parA, parB = [i for i in range(n + 1)], [i for i in range(n + 1)]
        cnta, cntb = n, n
        edges3 = [edge for edge in edges if edge[0] == 3]
        edges2 = [edge for edge in edges if edge[0] == 2]
        edges1 = [edge for edge in edges if edge[0] == 1]
        
        def find(par, x):
            p = par[x]
            while p != par[p]:
                p = par[p]
            par[x] = p
            return p

        ans = 0
        
        for e in edges3:
            x, y = find(parA, e[1]), find(parA, e[2])
            if x != y:
                parA[y] = x
                parB[y] = x
                cnta -= 1
                cntb -= 1
            else:
                ans += 1
        
        for e in edges1:
            x, y = find(parA, e[1]), find(parA, e[2])
            if x != y:
                parA[y] = x
                cnta -= 1
            else:
                ans += 1
                
        for e in edges2:
            x, y = find(parB, e[1]), find(parB, e[2])
            if x != y:
                parB[y] = parB[x]
                cntb -= 1
            else:
                ans += 1
        if cnta != 1 or cntb != 1:
            return -1
        
        return ans

    

class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        types = [[] for _ in range(3)]
        for t, *edge in edges:
            types[t - 1].append(edge)
        
        removed = 0
        self.parent = list(range(n + 1))
        self.size = [1] * (n + 1)
        
        # union type 3 first
        for u, v in types[2]:
            if self.isConnected(u, v):
                removed += 1
                continue
            self.union(u, v)
        
        self.parent_copy = list(self.parent)
        self.size_copy = list(self.size)
        
        for u, v in types[0]:
            if self.isConnected(u, v):
                removed += 1
                continue
            self.union(u, v)
        
        if sum([i == self.parent[i] for i in range(1, n + 1)]) > 1:
            return -1
        
        self.parent = self.parent_copy
        self.size = self.size_copy
        
        for u, v in types[1]:
            if self.isConnected(u, v):
                removed += 1
                continue
            self.union(u, v)
        
        if sum([i == self.parent[i] for i in range(1, n + 1)]) > 1:
            return -1
        
        return removed
    
    def isConnected(self, a, b):
        root_a = self.find(a)
        root_b = self.find(b)
        return root_a == root_b
    
    def union(self, a, b):
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a != root_b:
            if self.size[root_a] > self.size[root_b]:
                self.parent[root_b] = root_a
                self.size[root_a] += self.size[root_b]
            else:
                self.parent[root_a] = root_b
                self.size[root_b] += self.size[root_a]
            return True
        return False
    
    def find(self, a):
        curr = a
        while self.parent[curr] != curr:
            curr = self.parent[curr]
        root, curr = curr, a
        while self.parent[curr] != curr:
            self.parent[curr], curr = root, self.parent[curr]
        return root

class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        a,b,c=[[] for _ in range(n+1)],[[] for _ in range(n+1)],[[] for _ in range(n+1)]
        for k,i,j in edges:
            if k==1:
                a[i].append(j)
                a[j].append(i)
            elif k==2:
                b[i].append(j)
                b[j].append(i)
            else:
                c[i].append(j)
                c[j].append(i)
        d,st=[1]*(n+1),[1]
        d[0],d[1]=0,0
        while st:
            i=st.pop()
            for j in a[i]:
                if d[j]:
                    d[j]=0
                    st.append(j)
            for j in c[i]:
                if d[j]:
                    d[j]=0
                    st.append(j)
        if any(x for x in d): return -1
        d,st=[1]*(n+1),[1]
        d[0],d[1]=0,0
        while st:
            i=st.pop()
            for j in b[i]:
                if d[j]:
                    d[j]=0
                    st.append(j)
            for j in c[i]:
                if d[j]:
                    d[j]=0
                    st.append(j)
        if any(x for x in d): return -1
        d,s=[1]*(n+1),0
        for i in range(1,n+1):
            if d[i]:
                st,d[i]=[i],0
                while st:
                    i=st.pop()
                    for j in c[i]:
                        if d[j]:
                            d[j],s=0,s+1
                            st.append(j)
        return len(edges)-(2*n-2-s)
class UF:
    def __init__(self, n):
        self.par = list(range(n))
        self.size = [1] * n
        
    def find(self, x):
        if self.par[x] != x:
            self.par[x] = self.find(self.par[x])
        return self.par[x]
    
    def union(self, x , y):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False
        if self.size[rx] < self.size[ry]:
            rx, ry = ry, rx
        self.par[ry] = rx
        self.size[rx] += self.size[ry]
        self.size[ry] = self.size[rx]
        return True
    
    def sizee(self, x):
        return self.size[self.find(x)]


class Solution:
    def maxNumEdgesToRemove(self, N: int, edges: List[List[int]]) -> int:
        for i in range(len(edges)):
            edges[i][1] -= 1
            edges[i][2] -= 1
            
        alice = []
        bob = []
        both = []
        for t, u, v in edges:
            if t == 1:
                alice.append([u, v])
            elif t == 2:
                bob.append([u, v])
            else:
                both.append([u, v])
                
        uf1 = UF(N)
        uf2 = UF(N)
        res = 0
        for u, v in both:
            res += not uf1.union(u, v)
            uf2.union(u, v)
        for u, v in alice:
            res += not uf1.union(u, v)
        for u, v in bob:
            res += not uf2.union(u, v)
            
        if uf1.sizee(0) != N or uf2.sizee(0) != N:
            return -1
        return res
class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        sets = [-1] * (n + 1)
        
        def root(v):
            if sets[v] < 0: return v
            sets[v] = root(sets[v])
            return sets[v]
        
        def union(u, v):
            u, v = root(u), root(v)
            if u == v: return False
            if sets[u] > sets[v]:
                u, v = v, u
            sets[u] += sets[v]
            sets[v] = u
            return True
        
        remove_edges = 0
        alice_edges, bob_edges = 0, 0
        
        for t, u, v in edges:
            if t != 3: continue
            if not union(u, v):
                remove_edges += 1
            else:
                alice_edges += 1
                bob_edges += 1
        
        save_sets = sets[:]
        
        for t, u, v in edges:
            if t != 1: continue
            if not union(u, v):
                remove_edges += 1
            else:
                alice_edges += 1
        
        sets = save_sets
        
        for t, u, v in edges:
            if t != 2: continue
            if not union(u, v):
                remove_edges += 1
            else:
                bob_edges += 1
        
        if bob_edges != n - 1 or alice_edges != n - 1: return -1
        
        return remove_edges

class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        parenta, parentb = [i for i in range(n)], [i for i in range(n)]
        def find(i, parent):
            if parent[i]!=i:
                parent[i]=find(parent[i], parent)
            return parent[i]
        def union(a, b, parent):
            pa, pb = find(a, parent), find(b, parent)
            if pa==pb:
                return False
            parent[pa]=pb
            return True
        added_edge=na=nb=0
        for typ,u,v in edges:
            if typ==3 and union(u-1,v-1,parenta) and union(u-1,v-1,parentb):
                added_edge+=1
                na+=1
                nb+=1
        for typ,u,v in edges:
            if typ==1 and union(u-1,v-1,parenta):
                added_edge+=1
                na+=1
            elif typ==2 and union(u-1,v-1,parentb):
                added_edge+=1
                nb+=1
        return len(edges)-added_edge if na==nb==n-1 else -1
                
            

class DSU:
    def __init__(self, n: int):
        self.p = list(range(n))
        self.e = 0
        
    def find(self, x):
        if x != self.p[x]:
            self.p[x] = self.find(self.p[x])
        return self.p[x]

    def merge(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx == ry: 
            return 1
        self.p[rx] = ry
        self.e += 1
        return 0
    
class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        A, B = DSU(n + 1), DSU(n + 1)
        ans = 0
        for t, x, y in edges:
            if t != 3:
                continue
            ans += A.merge(x, y)
            B.merge(x, y)
        
        for t, x, y in edges:
            if t == 3: 
                continue
            d = A if t == 1 else B
            ans += d.merge(x, y)
        
        return ans if A.e == B.e == n - 1 else -1
class Solution(object):
    def maxNumEdgesToRemove(self, n, edges):

        ufa = UnionFind(n) # Graph for Alice
        ufb = UnionFind(n) # Graph for Bob
        cnt = 0 # number of removable edges
        
        for x, y, z in edges:
            if x == 3:
                flag1 = ufa.union(y, z)
                flag2 = ufb.union(y, z)
                if not flag1 or not flag2: cnt +=1

        for x, y, z in edges:
            if x == 1:
                flag = ufa.union(y, z)
                if not flag: cnt += 1
            if x == 2:
                flag = ufb.union(y, z)
                if not flag: cnt += 1

        return cnt if ufa.groups == 1 and ufb.groups == 1 else -1
            
        
class UnionFind():
    def __init__(self, n):
        self.parents = {i:i for i in range(1, n+1)}
        self.groups = n

    def find(self, x):
        if self.parents[x] == x:
            return x
        else:
            self.parents[x] = self.find(self.parents[x])
            return self.parents[x]

    def union(self, x, y):
        x = self.find(x)
        y = self.find(y)

        if x == y:
            return False

        self.parents[y] = x
        self.groups -= 1
        return True

class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        edges = [(t, i - 1, j - 1) for (t, i, j) in edges]
        bob = [-1 for _ in range(n)]
        self.ans = 0
        self.merge(bob, edges, 3)
        alice = bob.copy()
        self.merge(bob, edges, 2)
        self.merge(alice, edges, 1)
        return -1 if min(alice) != -n or min(bob) != -n else self.ans
        
    def merge(self, par, edges, ty):
        for (t, i, j) in edges:
            if t == ty:
                if self.find(par, i) != self.find(par, j):
                    self.union(par, i, j)
                else:
                    self.ans += 1
                    
    def find(self, par, i):
        if par[i] < 0: return i
        par[i] = self.find(par, par[i])
        return par[i]
    
    def union(self, par, i, j):
        i, j = self.find(par, i), self.find(par, j)
        if i == j: return
        if par[j] < par[i]: i, j = j, i
        par[i], par[j] = par[i] + par[j], i
class UnionFind:
  def __init__(self, n):
    self.father = list(range(n+1))
    self.count = n
    self.size = [0] * (n + 1)
    
  def find(self, p):
    while p != self.father[p]:
      self.father[p] = p = self.father[self.father[p]]
    return p
    
  def union(self, p, q):
    p, q = map(self.find, (p, q))
    if p == q:
      return False
    if self.size[p] < self.size[q]:
      p, q = q, p
    self.father[q] = p
    self.size[p] += self.size[q]
    self.count -= 1
    return True

class Solution:
  def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
    alice, bob, result = UnionFind(n), UnionFind(n), 0
    for i in range(3, 0, -1):
      for t, u, v in edges:
        if i != t:
          continue
        if i == 3:
          x = alice.union(u, v)
          y = bob.union(u, v)
          result += not x and not y
        elif i == 2:
          result += not bob.union(u, v)
        else:
          result += not alice.union(u, v)
    return result if alice.count == 1 and bob.count == 1 else -1
class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        
        edgesDict = defaultdict(list)
        
        for edgeType, src, dst in edges:
            edgesDict[src].append((edgeType, dst))
            edgesDict[dst].append((edgeType, src))
        
        visited = set()
        
        clusters = []
        cur_cluster = []
        def DFS(cur):
            visited.add(cur)
            cur_cluster.append(cur)
            for edge in edgesDict[cur]:
                if edge[0] == 3:
                    neighbor = edge[1]
                    if neighbor not in visited:
                        DFS(neighbor)
        
        for i in range(1, n + 1):            
            if i not in visited:
                cur_cluster = []
                DFS(i)
                clusters.append(cur_cluster)
        
        #print('clusters', clusters)
        
        nodeToClusterDict = dict()
        for idx, cur_cluster in enumerate(clusters):
            for node in cur_cluster:
                nodeToClusterDict[node] = idx
        
        #print('nodeToClusterDict', nodeToClusterDict)
        clusterEdges = defaultdict(set)
        
        def doit(this_type):   
            clusterEdges = defaultdict(set)
            for edgeType, src, dst in edges:
                if edgeType == this_type and nodeToClusterDict[src] != nodeToClusterDict[dst]:
                    src_cluster, dst_cluster = nodeToClusterDict[src], nodeToClusterDict[dst]
                    clusterEdges[src_cluster].add(dst_cluster)
                    clusterEdges[dst_cluster].add(src_cluster)
            
            #print('this_type', this_type, 'clusterEdges', clusterEdges)
            visitedClusters = set()
            def DFSCluster(cur_cluster):
                #print('visit cluster', cur_cluster)
                visitedClusters.add(cur_cluster)

                for neighbor_cluster in clusterEdges[cur_cluster]:
                    if neighbor_cluster not in visitedClusters:
                        DFSCluster(neighbor_cluster)
            
            DFSCluster(0)
            if len(visitedClusters) == len(clusters):
                # all clusters can be visited
                return len(clusters) - 1
            else:
                return -1
        
        ans1, ans2 = doit(1), doit(2)
        
        if ans1 >= 0 and ans2 >= 0:
            return len(edges) - (ans1 + ans2 + sum(len(x) - 1 for x in clusters))
        else:
            return -1
class Solution:
    def maxNumEdgesToRemove(self, n, edges):
        # Union find
        def find(node):
            if parent[node] != node:
                parent[node] = find(parent[node])
            return parent[node]
        
        def union(node1,node2):
            node1_p = find(node1)
            node2_p = find(node2)
            if node1_p == node2_p:
                return False
            if rank[node1_p] > rank[node2_p]:
                parent[node2_p] = node1_p
                rank[node1_p] +=1
            else:
                parent[node1_p] = node2_p
                rank[node2_p] +=1
            return True
        parent = [i for i in range(n+1)]
        rank = [0]*(n+1)
        edge_added = 0 
        edge_can_remove = 0
        
        for edge in edges:
            if edge[0] == 3:
                if union(edge[1], edge[2]):
                    edge_added +=1
                else:
                    edge_can_remove +=1
                    
        parent_back_up = parent[:]
        alice_edge = 0
        bob_edge = 0
        for edge in edges:
            if edge[0] == 1:
                if union(edge[1], edge[2]):
                    alice_edge +=1
                else:
                    edge_can_remove +=1
        parent = parent_back_up
        for edge in edges:
            if edge[0] == 2:
                if union(edge[1], edge[2]):
                    bob_edge +=1
                else:
                    edge_can_remove +=1
        if alice_edge == bob_edge == n-1 - edge_added and edge_can_remove>=0:
            return edge_can_remove
        return -1
class Solution:
       def maxNumEdgesToRemove(self, n, edges):
        # Union find
        def find(i):
            if i != root[i]:
                root[i] = find(root[i])
            return root[i]

        def uni(x, y):
            x, y = find(x), find(y)
            if x == y: return 0
            root[x] = y
            return 1

        res = e1 = e2 = 0

        # Alice and Bob
        root = [i for i in range(n + 1)]
        for t, i, j in edges:
            if t == 3:
                if uni(i, j):
                    e1 += 1
                    e2 += 1
                else:
                    res += 1
        root0 = root[:]

        # only Alice
        for t, i, j in edges:
            if t == 1:
                if uni(i, j):
                    e1 += 1
                else:
                    res += 1

        # only Bob
        root = root0
        for t, i, j in edges:
            if t == 2:
                if uni(i, j):
                    e2 += 1
                else:
                    res += 1

        return res if e1 == e2 == n - 1 else -1

class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        def find(roots, x):
            if x == roots[x]:
                return x
            else:
                roots[x] = find(roots, roots[x])
                return roots[x]

        pairs1 = []
        pairs2 = []
        pairs3 = []

        for type, x, y in edges:
            if type == 1:
                pairs1.append((x, y))
            elif type == 2:
                pairs2.append((x, y))
            else:
                pairs3.append((x, y))

        roots = [i for i in range(n + 1)]
        rootSet = set(range(1, n + 1))
        res = 0

        for x, y in pairs3:
            root1 = find(roots, x)
            root2 = find(roots, y)

            if root1 != root2:
                roots[root2] = root1
                rootSet.remove(root2)
            else:
                res += 1

        root1Set = set(rootSet)
        root2Set = set(rootSet)
        roots1 = list(roots)
        roots2 = list(roots)

        for x, y in pairs1:
            root1 = find(roots1, x)
            root2 = find(roots1, y)

            if root1 != root2:
                roots1[root2] = root1
                root1Set.remove(root2)
            else:
                res += 1

        for x, y in pairs2:
            root1 = find(roots2, x)
            root2 = find(roots2, y)

            if root1 != root2:
                roots2[root2] = root1
                root2Set.remove(root2)
            else:
                res += 1

        if len(root1Set) != 1 or len(root2Set) != 1: return -1
        return res
class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        def dfs(s, mp):
            for n in mp[s]:
                if visited[n] == 0:
                    visited[n] = 1
                    self.cnt += 1
                    dfs(n, mp)

        c = [0] * 4
        mpa = defaultdict(set)
        mpb = defaultdict(set)
        mpc = defaultdict(set)
        for e0, e1, e2 in edges:
            c[e0] += 1
            if e0 == 1 or e0 == 3:
                mpa[e1].add(e2)
                mpa[e2].add(e1)
            if e0 == 2 or e0 == 3:
                mpb[e1].add(e2)
                mpb[e2].add(e1)
            if e0 == 3:
                mpc[e1].add(e2)
                mpc[e2].add(e1)
                
        self.cnt = 1
        visited = [0] * (n + 1)
        visited[1] = 1
        dfs(1, mpa)
        if self.cnt != n:
            return -1

        self.cnt = 1
        visited = [0] * (n + 1)
        visited[1] = 1
        dfs(1, mpb)
        if self.cnt != n:
            return -1
        
        uc = 0
        visited = [0] * (n + 1)
        for i in range(1, n + 1):
            if visited[i] == 0:
                visited[i] = 1
                self.cnt = 0
                dfs(i, mpc)
                uc += self.cnt
        return (c[3] - uc) + (c[1] + uc - (n-1)) + (c[2] + uc - (n-1))     

class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        def dfs_and_cluster(n, adj):
            num_clusters = 0
            edges_used = 0
            node_to_cluster = {}
            start_nodes = list(range(n))
            while len(start_nodes) > 0:
                node = start_nodes.pop()
                if node in node_to_cluster:
                    continue

                cluster_id = num_clusters
                num_clusters += 1

                node_to_cluster[node] = cluster_id
                q = [node]
                while len(q) > 0:
                    node = q.pop()
                    for next_node in adj[node]:
                        if next_node not in node_to_cluster:
                            edges_used += 1
                            node_to_cluster[next_node] = cluster_id
                            q.append(next_node)
            return num_clusters, node_to_cluster, edges_used

        shared_adj = {i: set() for i in range(n)}
        for edge_type, u, v in edges:
            u, v = u - 1, v - 1
            if edge_type == 3:
                shared_adj[u].add(v)
                shared_adj[v].add(u)

        shared_num_clusters, shared_node_to_cluster, shared_edges_used = dfs_and_cluster(n, shared_adj)
        print(shared_node_to_cluster)

        alice_adj = {i: set() for i in range(shared_num_clusters)}
        for edge_type, u, v in edges:
            u, v = u - 1, v - 1
            if edge_type == 1:
                u = shared_node_to_cluster[u]
                v = shared_node_to_cluster[v]
                alice_adj[u].add(v)
                alice_adj[v].add(u)

        alice_num_clusters, _, alice_edges_used = dfs_and_cluster(shared_num_clusters, alice_adj)

        bob_adj = {i: set() for i in range(shared_num_clusters)}
        for edge_type, u, v in edges:
            u, v = u - 1, v - 1
            if edge_type == 2:
                u = shared_node_to_cluster[u]
                v = shared_node_to_cluster[v]
                bob_adj[u].add(v)
                bob_adj[v].add(u)
        
        bob_num_clusters, _, bob_edges_used = dfs_and_cluster(shared_num_clusters, bob_adj)

        if alice_num_clusters > 1 or bob_num_clusters > 1:
            return -1

        return len(edges) - shared_edges_used - alice_edges_used - bob_edges_used

class Solution:
    def maxNumEdgesToRemove(self, n: int, edges) -> int:
        N = len(edges)
        dup = set()
        res = 0
        c1, c2, bc = 0, 0, 0
        alice, bob, both = defaultdict(list), defaultdict(list), defaultdict(list)
        
        for t, u, v in edges:
            if (t, u, v) not in dup:
                dup.add((t, u, v))
                if t == 1 or t == 3:
                    if t == 1:
                        c1 += 1
                    alice[u].append(v)
                    alice[v].append(u)
                if t == 2 or t == 3:
                    if t == 2:
                        c2 += 1
                    bob[u].append(v)
                    bob[v].append(u)
                if t == 3:
                    bc += 1
                    both[u].append(v)
                    both[v].append(u)
            else:
                res += 1
        
        va, vb, = set(), set()
        vc = dict()
        
        def dfs(node, t):
            if t == 1:
                va.add(node)
                for ngb in alice[node]:
                    if not ngb in va:
                        dfs(ngb, t)
            else:
                vb.add(node)
                for ngb in bob[node]:
                    if not ngb in vb:
                        dfs(ngb, t)
        
        dfs(1, 1)
        dfs(1, 2)
        
        if len(va) < n or len(vb) < n:
            return -1
        
        def dfs_both(node, prev, idx):
            vc[node] = idx
            self.tmp += 1
            for ngb in both[node]:
                if ngb == prev:
                    continue
                if ngb not in vc:
                    dfs_both(ngb, node, idx)
         
        bc_need = 0
        idx = 0
        for i in both:
            if i not in vc:
                idx += 1
                self.tmp = 0
                dfs_both(i, -1, idx)
                bc_need += self.tmp - 1
                
        res += bc - bc_need
        res += c1 - (n - 1 - bc_need)
        res += c2 - (n - 1 - bc_need)
        return res
from collections import defaultdict
from heapq import heappush, heappop

class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        aj = [defaultdict(set) for i in range(4)]
        total = len(edges)
        for t, i, j in edges:
            if i == j:
                continue
            aj[t][i].add(j)
            aj[t][j].add(i)
        
        reuse = set()
        count = 0
        
        visited = {1}
        heap = []
        for i in aj[3][1]:
            heappush(heap, (1, 1, i))
        for i in aj[1][1]:
            heappush(heap, (2, 1, i))
        while len(visited) < n and heap:
            w, i, j = heappop(heap)
            if j in visited:
                continue
                
            if w == 1:
                reuse.add((i, j))
            count += 1
            visited.add(j)
            for k in aj[3][j]:
                if k not in visited:
                    heappush(heap, (1, j, k))
            for k in aj[1][j]:
                if k not in visited:
                    heappush(heap, (2, j, k))
        if len(visited) < n:
            return -1
            
        visited = {1}
        heap = []
        for i in aj[3][1]:
            if (1, i) in reuse or (i, 1) in reuse:
                heappush(heap, (0, 1, i))
            else:
                heappush(heap, (1, 1, i))
        for i in aj[2][1]:
            heappush(heap, (2, 1, i))
        while len(visited) < n and heap:
            w, i, j = heappop(heap)
            if j in visited:
                continue
                
            if w > 0:
                count += 1
            visited.add(j)
            for k in aj[3][j]:
                if k not in visited:
                    if (j, k) in reuse or (k, j) in reuse:
                        heappush(heap, (0, j, k))
                    else:
                        heappush(heap, (1, j, k))
            for k in aj[2][j]:
                if k not in visited:
                    heappush(heap, (2, j, k))
        if len(visited) < n:
            return -1

        return total - count

import copy

from collections import defaultdict

class DSU:
  def __init__(self, reps):
    # representer
    self.reps = reps
  # def add(self, x):
  #   self.reps[x] = x
  def find(self, x):
    if not x == self.reps[x]:
      self.reps[x] = self.find(self.reps[x])
    return self.reps[x]
  def union(self, x, y):
    self.reps[self.find(y)] = self.find(x)

class Solution:
  def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
    # start with all type3 edges
    es = [[], [], [], []]
    for i, u, v in edges:
      es[i].append((u, v))
    # start with all type3 edges
    dsu = DSU(reps = {x: x for x in range(1, n + 1)})
    for u, v in es[3]:
      dsu.union(u, v)
    # islands
    islands = defaultdict(set)
    for x in range(1, n + 1):
      islands[dsu.find(x)].add(x)
    if len(islands) == 1:
      return len(es[3]) - (n - 1) + len(es[1]) + len(es[2])
    # Alice
    dA = copy.deepcopy(dsu)
    for u, v in es[1]:
      dA.union(u, v)
    islandsA = defaultdict(set)
    for x in range(1, n + 1):
      islandsA[dA.find(x)].add(x)
    if len(islandsA) > 1:
      return -1
    # Bob
    dB = copy.deepcopy(dsu)
    for u, v in es[2]:
      dB.union(u, v)
    islandsB = defaultdict(set)
    for x in range(1, n + 1):
      islandsB[dB.find(x)].add(x)
    if len(islandsB) > 1:
      return -1
    return len(edges) - (n - len(islands)) - (len(islands) - 1) * 2
class Solution:
    def maxNumEdgesToRemove(self, N: int, E: List[List[int]], same = 0) -> int:
        E = [[_, u - 1, v - 1] for _, u, v in E]                    # u2b50ufe0f -1 for 1-based to 0-based indexing
        A = [i for i in range(N)]                                   # U0001f642 parent representatives of disjoint sets for Alice
        B = [i for i in range(N)]                                   # U0001f642 parent representatives of disjoint sets for Bob
        def find(P, x): P[x] = P[x] if P[x] == x else find(P, P[x]); return P[x]
        def union(P, a, b):
            a = find(P, a)
            b = find(P, b)
            if a == b:
                return 1
            P[a] = b  # arbitrary choice
            return 0
        for type, u, v in E:
            if type == 3: same += union(A, u, v) | union(B, u, v)   # U0001f947 first: U0001f517 union u2705 shared edges between Alice and Bob
        for type, u, v in E:
            if type == 1: same += union(A, u, v)                    # U0001f948 second: U0001f517 union U0001f6ab non-shared edges between Alice and Bob
            if type == 2: same += union(B, u, v)
        return same if all(find(A, 0) == find(A, x) for x in A) and all(find(B, 0) == find(B, x) for x in B) else -1
        

class Solution:
    def find(self, x, uf):
        if uf[x] != x:
            uf[x] = self.find(uf[x], uf)
        return uf[x]
    
    def union(self, u, v, uf):
        p1 = self.find(u, uf)
        p2 = self.find(v, uf)
        uf[p1] = p2
        return p1 != p2
    
    def connected(self, uf):
        parent = set()
        for i in range(len(uf)):
            parent.add(self.find(i, uf))
            if len(parent) > 1:
                return False
        return len(parent) == 1
    
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        edges = [[t, u-1, v-1] for t, u, v in edges]
        t1_edges = set()
        t2_edges = set()
        t3_edges = set()
        ans = 0
        
        for t, u, v in edges:
            if t == 1:
                t1_edges.add((u, v))
            elif t == 2:
                t2_edges.add((u, v))
            else:
                if (u, v) in t1_edges:
                    t1_edges.remove((u, v))
                    ans += 1
                if (u, v) in t2_edges:
                    t2_edges.remove((u, v))
                    ans += 1
                t3_edges.add((u, v))
        
        uf1 = [i for i in range(n)]
        uf2 = [i for i in range(n)]
        
        for u, v in t3_edges:
            union1 = self.union(u, v, uf1)
            union2 = self.union(u, v, uf2)
            if not union1 and not union2:
                ans += 1
        
        for u, v in t1_edges:
            if not self.union(u, v, uf1):
                ans += 1
        for u, v in t2_edges:
            if not self.union(u, v, uf2):
                ans += 1
                
        if not self.connected(uf1) or not self.connected(uf2):
            return -1
        return ans
        
        

import copy

def union(subsets, u, v):
    uroot = find(subsets, u)
    vroot = find(subsets, v)
    
    if subsets[uroot][1] > subsets[vroot][1]:
        subsets[vroot][0] = uroot
    if subsets[vroot][1] > subsets[uroot][1]:
        subsets[uroot][0] = vroot
    if subsets[uroot][1] == subsets[vroot][1]:
        subsets[vroot][0] = uroot
        subsets[uroot][1] += 1
    

def find(subsets, u):
    if subsets[u][0] != u:
        subsets[u][0] = find(subsets, subsets[u][0])
    return subsets[u][0]


class Solution:
    #kruskal's
    #1 is alice and 2 is bob
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        
        mst1 = set() #set for Alice's MST
        mst2 = set() #set for Bob's MST
        subsets1 = ['1 index'] + [[x+1,0] for x in range(n)] #Alice's unionfind
        subsets2 = ['1 index'] + [[x+1,0] for x in range(n)] #Bob's unionfind
        
        edges = sorted(edges, key= lambda e: -e[0])
        e = 0 #number of total edges used
        e1 = 0 #number of edges for Alice
        e2 = 0 #number of edges for Bob
        i = 0 #track position in edges list
        
        #start with type 3 edges
        while e < n - 1:
            if i == len(edges): 
                return -1
            typ, u, v = edges[i]
            if typ != 3: break
            if find(subsets1, u) != find(subsets1, v):
                union(subsets1, u, v)
                mst1.add(u)
                mst1.add(v)
                e += 1
            
            i += 1
        
        #everything that was done to Alice applies to Bob
        e1 = e
        e2 = e
        mst2 = mst1.copy()
        subsets2 = copy.deepcopy(subsets1)
        
        #once done with shared edges, do Bob's
        while e2 < n-1:
            if i == len(edges): 
                return -1
            typ, u, v = edges[i]
            if typ != 2: break
            if find(subsets2, u) != find(subsets2, v):
                union(subsets2, u, v)
                e += 1
                e2 += 1
            i += 1
        
        if e2 < n - 1: 
            return -1 #if we've used all edges bob can use (types 2 and 3) and he still can't reach all nodes, ur fucked
        
        #now finish Alice's MST
        while e1 < n-1:
            if i == len(edges): 
                return -1
            
            typ, u, v = edges[i]
            if find(subsets1, u) != find(subsets1, v):
                union(subsets1, u, v)
                e += 1
                e1 += 1
            i += 1
            
        return len(edges) - e
            
            
            
            
        
        

class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        
        edges.sort(key=lambda x:-x[0])
        
        
        father1=[i for i in range(n+1)]
        size1=[1 for i in range(n+1)]
        father2=[i for i in range(n+1)]
        size2=[1 for i in range(n+1)]
        
        
        def find(a,typ):
            
            if typ==1:
                
                path=[]
                
                while a!=father1[a]:
                    path.append(a)
                    a=father1[a]
                    
                
                for b in path:
                    father1[b]=a
                    
                return a
            
            elif typ==2:
                
                path=[]
                
                while a!=father2[a]:
                    path.append(a)
                    a=father2[a]
                    
                
                for b in path:
                    father2[b]=a
                    
                return a
                
            
        def union(a,b,typ):
            
            fa, fb = find(a,typ), find(b,typ)
            if fa==fb:
                return
            
            if typ==1:
                
                father1[fa]=fb
                size1[fb]+=size1[fa]
                
            else:
                
                father2[fa]=fb
                size2[fb]+=size2[fa]
                
            return
        
        
        necessary=0
        
        for edge in edges:
            typ,a,b=edge
            
            if typ==3:
                
                if find(a,1)!=find(b,1):
                    
                    necessary+=1
                    
                    union(a,b,1)
                    union(a,b,2)
            
            elif typ==1:
                
                if find(a,1)!=find(b,1):
                    
                    necessary+=1
                    
                    union(a,b,1)
            
            else:
                
                if find(a,2)!=find(b,2):
                    
                    necessary+=1
                    
                    union(a,b,2)
                    
        
        if size1[find(1,1)]!=n or size2[find(1,2)]!=n:
            return -1
        
        return len(edges)-necessary
                
                    

class UnionFind:
    def __init__(self,n):
        self.parent=[i for i in range(n+1)]
        self.size=[1]*(n+1)
    def find(self,x):
        if self.parent[x]==x:
            return x
        self.parent[x]=self.find(self.parent[x])
        return self.parent[x]
    def union(self,x,y):
        x=self.find(x)
        y=self.find(y)
        if x==y:
            return False,0
        if self.size[x]>self.size[y]:
            x,y=y,x
        self.parent[y]=x
        self.size[x]+=self.size[y]
        return True,self.size[x]
class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        def getMST(whos):
            nonlocal edges,n
            unionfind=UnionFind(n)
            # mst=set()
            reqEdges=set()
            nodes=0
            for indx,(typ,fnode,tnode) in enumerate(edges):
                # if fnode not in mst or tnode not in mst:
                siz=0
                if typ==3 or typ==whos:
                    res,siz=unionfind.union(fnode,tnode)
                    if res:
                        reqEdges.add(tuple(edges[indx]))
                        nodes=max(nodes,siz)
                if siz==n:
                    break
            return siz==n,reqEdges
        
        edges.sort(key=lambda item: -item[0])
        ares,alice=getMST(1)
        if not ares:
            return -1
        bres,bob=getMST(2)
        if not bres:
            return -1
        nset=alice.union(bob)
        return len(edges)-len(nset)
        
                        
                        
        
        
        
        
            

class Solution:
    def maxNumEdgesToRemove(self, n: int, e: List[List[int]]) -> int:
        g=[[] for _ in range(n)]
        for t,x,y in e:
            g[x-1].append((y-1,t))
            g[y-1].append((x-1,t))

        def sub(f):
            res=[0]
            s=[0]
            def dfs(now,q):
                if 1<<now&s[0]:return 
                s[0]|=1<<now
                if q:
                    res[0]+=1
                for nex,t in g[now]:
                    if t==f or t==3:
                        dfs(nex,0)
            for i in range(n):
                dfs(i,1)
            return res[0]

        if sub(1)>1 or sub(2)>1:
            return -1
        return len(e)-(n-2+sub(3))

class Solution:
    def maxNumEdgesToRemove(self, n: int, A: List[List[int]]) -> int:
        g = collections.defaultdict(list)
        
        self.vis = {}
        
        compIdx = {}
        
        def dfs(node, typ, compIndex, mapRef = compIdx):
            mapRef[node] = compIndex
            self.vis[node] = 1
            for child, eType in g[node]:
                if eType in typ and child not in self.vis:
                    dfs(child, typ, compIndex, mapRef)
            self.vis[node] = 2

        ctr3 = 0
        ctr1, ctr2 = 0, 0
        set1 = set()
        set2 = set()
        set3 = set()
        for k,i,j in A:
            if k == 3:
                ctr3 += 1
                ctr1 += 1
                ctr2 += 1
                set3.add(i)
                set3.add(j)
            elif k == 1:
                ctr1 += 1
                set1.add(i)
                set1.add(j)
            else:
                ctr2 += 1
                set2.add(i)
                set2.add(j)
            g[i].append((j,k))
            g[j].append((i,k))
        
        comp3 = 0
        for node in set3:
            if node not in self.vis:
                dfs(node, [3], comp3)
                comp3 += 1
        
        # Deletable edges of type 3
        res1 = ctr3 - ((len(set3)) - comp3)
        
        comp1Idx = {}
        comp1 = 0
        self.vis = {}
        for node in set1.union(set3):
            if node not in self.vis:
                dfs(node, [1, 3], comp1, comp1Idx)
                comp1 += 1
        
        comp2Idx = {}
        comp2 = 0
        self.vis = {}
        for node in set2.union(set3):
            if node not in self.vis:
                dfs(node, [2, 3], comp2, comp2Idx)
                comp2 += 1
        
        print(comp1, comp2, comp3, len(comp1Idx), len(comp2Idx), len(compIdx), len(A))
        
        if comp1 > 1 or comp2 > 1 or len(comp1Idx) != n or len(comp2Idx) != n:
            return -1
        
        res2 = ctr1 - ((len(set3.union(set1))) - comp1)
        res3 = ctr2 - ((len(set3.union(set2))) - comp2)
        
        print(res1, res2, res3)
        return - res1 + res2 + res3
    
    
"""
13
[[1,1,2],[2,1,3],[3,2,4],[3,2,5],[1,2,6],[3,6,7],[3,7,8],[3,6,9],[3,4,10],[2,3,11],[1,5,12],[3,3,13],[2,1,10],[2,6,11],[3,5,13],[1,9,12],[1,6,8],[3,6,13],[2,1,4],[1,1,13],[2,9,10],[2,1,6],[2,10,13],[2,2,9],[3,4,12],[2,4,7],[1,1,10],[1,3,7],[1,7,11],[3,3,12],[2,4,8],[3,8,9],[1,9,13],[2,4,10],[1,6,9],[3,10,13],[1,7,10],[1,1,11],[2,4,9],[3,5,11],[3,2,6],[2,1,5],[2,5,11],[2,1,7],[2,3,8],[2,8,9],[3,4,13],[3,3,8],[3,3,11],[2,9,11],[3,1,8],[2,1,8],[3,8,13],[2,10,11],[3,1,5],[1,10,11],[1,7,12],[2,3,5],[3,1,13],[2,4,11],[2,3,9],[2,6,9],[2,1,13],[3,1,12],[2,7,8],[2,5,6],[3,1,9],[1,5,10],[3,2,13],[2,3,6],[2,2,10],[3,4,11],[1,4,13],[3,5,10],[1,4,10],[1,1,8],[3,3,4],[2,4,6],[2,7,11],[2,7,10],[2,3,12],[3,7,11],[3,9,10],[2,11,13],[1,1,12],[2,10,12],[1,7,13],[1,4,11],[2,4,5],[1,3,10],[2,12,13],[3,3,10],[1,6,12],[3,6,10],[1,3,4],[2,7,9],[1,3,11],[2,2,8],[1,2,8],[1,11,13],[1,2,13],[2,2,6],[1,4,6],[1,6,11],[3,1,2],[1,1,3],[2,11,12],[3,2,11],[1,9,10],[2,6,12],[3,1,7],[1,4,9],[1,10,12],[2,6,13],[2,2,12],[2,1,11],[2,5,9],[1,3,8],[1,7,8],[1,2,12],[1,5,11],[2,7,12],[3,1,11],[3,9,12],[3,2,9],[3,10,11]]
"""
class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        def find(x):
            if x != root[x]:
                root[x] = find(root[x])
            return root[x]
        
        e1 = 0
        e2 = 0
        res = 0
        def union(x, y):
            x = find(x)
            y = find(y)
            if x == y:
                return 0
            root[x] = y
            return 1
        
        root = list(range(n+1))
        for t, i, j in edges:
            if t == 3:
                if union(i,j):
                    e1+=1
                    e2+=1
                else:
                    res+=1
        temp = root[:]
        
        for t,i,j in edges:
            if t == 1:
                if union(i,j):
                    e1+=1
                else:
                    res+=1
        
        root = temp
        
        for t,i,j in edges:
            if t == 2:
                if union(i,j):
                    e2+=1
                else:
                    res+=1
        
        if e1 == n-1 and e2 == n-1:
            return res
        else:
            return -1
class Dsu:
    def __init__(self, n):
        self.roots = list(range(n + 1))
        self.cnts = [1] * (n + 1)
            
    def find(self, x):
        if self.roots[x] != x:
            self.roots[x] = self.find(self.roots[x])
        return self.roots[x]
        
    def union(self, x, y):
        rx = self.find(x)
        ry = self.find(y)
        if rx == ry:
            return False
        if self.cnts[rx] >= self.cnts[ry]:
            self.roots[ry] = rx
            self.cnts[rx] += self.cnts[ry]
            self.cnts[ry] = 0
        else:
            self.roots[rx] = ry
            self.cnts[ry] += self.cnts[rx]
            self.cnts[rx] = 0
        return True
        
class Solution:
            
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        # [[3,1,2],[3,2,3],[1,1,3],[1,2,4],[1,1,2],[2,3,4]]
        dsu_alice = Dsu(n)
        dsu_bob = Dsu(n)
        rm = 0
        for t, u, v in edges:
            if t == 3:
                dsu_alice.union(u, v)
                if not dsu_bob.union(u, v):
                    rm += 1
                    
        for t, u, v in edges:
            if t == 1:
                if not dsu_alice.union(u, v):
                    rm += 1
            elif t == 2:
                if not dsu_bob.union(u, v):
                    rm += 1
                    
        if dsu_alice.cnts[dsu_alice.find(1)] != n or dsu_bob.cnts[dsu_bob.find(1)] != n:
            return -1
        return rm
class UF:
    def __init__(self, n):
        self.p = [i for i in range(n)]
        
    def find(self, x):
        if self.p[x] != x:
            self.p[x] = self.find(self.p[x])
        return self.p[x]
    
    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py:
            return False
        self.p[py] = px
        return True

class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        A, B = set(), set()
        rmA, rmB = 0, 0
        for t, u, v in edges:
            if t == 1:
                if (-3, u, v) in A:
                    rmA += 1
                else:
                    A.add((-1, u, v))
            elif t == 2:
                if (-3, u, v) in B:
                    rmB += 1
                else:
                    B.add((-2, u, v))
            else:
                if (-1, u, v) in A:
                    rmA += 1
                    A.remove((-1, u, v))
                if (-2, u, v) in B:
                    rmB += 1  
                    B.remove((-2, u, v))
                A.add((-3, u, v))
                B.add((-3, u, v))
        
        common = set()
        ufa = UF(n + 1)
        ufb = UF(n + 1)
        eA = eB = 0
        for t, u, v in sorted(A):
            if ufa.union(u, v):
                eA += 1
            else:
                if t == -1:
                    rmA += 1
                else:
                    common.add((u, v))
                    
        for t, u, v in sorted(B):
            if ufb.union(u, v):
                eB += 1
            else:
                if t == -2:
                    rmB += 1
                else:
                    common.add((u, v))
                   
        return rmA + rmB + len(common) if eA == eB == n - 1 else -1

class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        def find(x, p):
            if p[x] != x:
                p[x] = find(p[x], p)
            return p[x]
        
        def merge(rx, ry, p):
            p[rx] = ry
        
        res = 0
        pa = [i for i in range(n)]
        pb = [i for i in range(n)]
        
        edges.sort(key=lambda x: -x[0])
        
        for type_, u, v in edges:
            if type_ == 1:
                ru, rv = find(u - 1, pa), find(v - 1, pa)
                if ru == rv:
                    res += 1
                else:
                    merge(ru, rv, pa)
            elif type_ == 2:    
                ru, rv = find(u - 1, pb), find(v - 1, pb)
                if ru == rv:
                    res += 1
                else:
                    merge(ru, rv, pb)
            
            if type_ == 3:
                rua, rva = find(u - 1, pa), find(v - 1, pa)
                rub, rvb = find(u - 1, pb), find(v - 1, pb)
                if rua == rva and rub == rvb:
                    res += 1
                else:
                    merge(rua, rva, pa)
                    merge(rub, rvb, pb)
                    
        
        return res if len({find(i, pa) for i in range(n)}) == len({find(i, pb) for i in range(n)}) == 1 else -1

class Solution:
    def maxNumEdgesToRemove(self, n, edges):
        def find(i):
            if i != root[i]:
                root[i] = find(root[i])
            return root[i]

        def uni(x, y):
            x, y = find(x), find(y)
            if x == y: return False
            root[x] = y
            return True

        root = {i:i for i in range(1,n+1)}
        res, a, b = 0, 0, 0
        
        for t, u, v in edges:
            if t == 3:
                if uni(u, v):
                    a += 1
                    b += 1
                else:
                    res += 1
                    
        common = root.copy()
        for t, u, v in edges:
            if t == 1:
                if uni(u, v):
                    a += 1
                else:
                    res += 1
        
        root = common
        for t, u, v in edges:
            if t == 2:
                if uni(u, v):
                    b += 1
                else:
                    res += 1
        return res if a == b == n - 1 else -1

class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:        
        p1, s1 = list(range(n + 1)), [1] * (n + 1)
        inc = 0
        
        def find(parent, i): 
            if parent[i] != i: 
                parent[i] = find(parent, parent[i])
            return parent[i]

        def union(parent, size, x, y): 
            i, j = find(parent, x), find(parent, y)
            if size[i] < size[j]: 
                parent[i] = j
                size[j] += size[i]
            else: 
                parent[j] = i 
                size[i] += size[j]
        
        for t, u, v in edges:
            if t == 3:
                i, j = find(p1, u), find(p1, v)
                if i == j: continue
                union(p1, s1, i, j)
                inc += 1
        p2, s2 = p1[:], s1[:]        
        
        for t, u, v in edges:
            if t == 1:
                i, j = find(p1, u), find(p1, v)
                if i == j: continue
                union(p1, s1, i, j)
                inc += 1
            elif t == 2:
                i, j = find(p2, u), find(p2, v)
                if i == j: continue
                union(p2, s2, i, j)
                inc += 1
        
        return len(edges) - inc if max(s1) == n and max(s2) == n else -1
class UnionFind:
    def __init__(self):
        self.parents = defaultdict(lambda:-1)
        self.ranks = defaultdict(lambda:1)
    def join(self,a,b):
        pa,pb = self.find(a),self.find(b)
        if pa==pb:
            return False
        if self.ranks[pa]>self.ranks[pb]:
            self.parents[pb]=pa
            self.ranks[pa]+=self.ranks[pb]
        else:
            self.parents[pa]=pb
            self.ranks[pb]+=self.ranks[pa]
        return True
    def find(self,a):
        if self.parents[a]==-1:
            return a
        self.parents[a]=self.find(self.parents[a])
        return self.parents[a]

class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        if n==0:
            return 0
        def pre_check(person):
            my_edges = [edge for edge in edges if edge[0] in [person,3]]
            dic = defaultdict(list)
            for _,a,b in my_edges:
                dic[a].append(b)
                dic[b].append(a)
            seen=set()
            # print(dic)
            def dfs(curr):
                # print(curr)
                if curr in seen:
                    return
                seen.add(curr)
                for nxt in dic[curr]:
                    # print('nxt',nxt)
                    dfs(nxt)
            dfs(1)
            # print(seen, dic)
            # print(len(seen))
            return len(seen)==n
        if not pre_check(1) or not pre_check(2):
            return -1
        both_edges = [edge for edge in edges if edge[0]==3]
        # print(both_edges)
        a_cnt = sum(edge[0]==1 for edge in edges)
        b_cnt = sum(edge[0]==2 for edge in edges)
        uf = UnionFind()
        rid = 0
        for edge in both_edges:
            # print(edge)
            if not uf.join(edge[1],edge[2]):
                rid += 1
        uniq = set(uf.find(i) for i in range(1,n+1))
        uniq = len(uniq)
        # print(rid,uniq)
        # print(uniq)
        # print(rid)
        # print(a_cnt, b_cnt)
        return rid + a_cnt - uniq + 1 + b_cnt - uniq + 1
        
            
        

import collections
class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        #remove duplicate if type3 exists remove type1 and type2
        type1_graph=collections.defaultdict(set)
        type2_graph=collections.defaultdict(set)
        type3_graph=collections.defaultdict(set)
        alice_graph=collections.defaultdict(set)
        bob_graph=collections.defaultdict(set)
        no_of_type1_edges=0
        no_of_type2_edges=0
        no_of_type3_edges=0
        for t,v1,v2 in edges:
            if t==3:
                type3_graph[v1].add(v2)
                type3_graph[v2].add(v1)
                alice_graph[v1].add(v2)
                alice_graph[v2].add(v1)
                bob_graph[v1].add(v2)
                bob_graph[v2].add(v1)
                no_of_type3_edges +=1

            if t==1:
                type1_graph[v1].add(v2)
                type1_graph[v2].add(v1)
                no_of_type1_edges +=1
                alice_graph[v1].add(v2)
                alice_graph[v2].add(v1)

            if t==2:
                type2_graph[v1].add(v2)
                type2_graph[v2].add(v1)
                no_of_type2_edges +=1
                bob_graph[v1].add(v2)
                bob_graph[v2].add(v1)

        def dfs(s,edges,visited):
            for e in edges[s]:
                if e not in visited:
                    visited.add(e)
                    dfs(e,edges,visited)


        def tran_graph(edges):
            nodes_set=[]
            total_visited=set()
            if len(edges)==0:
                return []
            for s in edges.keys():
                if s not in total_visited:
                    visited=set()
                    dfs(s,edges,visited)
                    nodes_set.append(visited)
                    total_visited|=visited
            print(nodes_set)
            return nodes_set
        # check whether alice and bob's graph connected
        for nodes_set in (tran_graph(alice_graph),tran_graph(bob_graph)):
            if len(nodes_set)!=1:
                return -1
            if len(nodes_set[0])!=n:
                return -1
        #remove duplicate type edge
        type3_nodes_sets=tran_graph(type3_graph)
        print("type3_nodes_sets",type3_nodes_sets)
        print("type3 edges",no_of_type3_edges)
        type3_nodes=0
        removed=no_of_type3_edges
        for set_ in type3_nodes_sets:
            type3_nodes+=len(set_)
            removed-=len(set_)-1
        print(removed)


        graphs=len(type3_nodes_sets)
        removed+=no_of_type1_edges-(graphs-1+n-type3_nodes)
        removed += no_of_type2_edges - (graphs - 1 + n - type3_nodes)
        # print(removed)
        return removed

class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        A, B = {i:i for i in range(1,n+1)}, {i:i for i in range(1, n+1)}
        #print(A, B)
        def find(dp, i):
            #print(i, dp)
            if dp[i] != i:
                dp[i] = find(dp, dp[i])
            return dp[i]
        edges = sorted(edges, reverse=True)
        
        cnt = 0
        for t, a, b in edges:
            if t == 3:
                if find(A, a) != find(A, b):
                    A[find(A, a)] = A[find(A, b)]
                    B[find(B, a)] = A[find(B, b)]
                else:
                    cnt += 1
            elif t == 2:
                if find(B, a) != find(B, b):
                    B[find(B, a)] = A[find(B, b)]
                else:
                    cnt += 1
            else:
                if find(A, a) != find(A, b):
                    A[find(A, a)] = A[find(A, b)]
                else:
                    cnt += 1
        return cnt if len(set(find(A,e) for e in range(1,n+1))) == 1 and len(set(find(B,e) for e in range(1,n+1))) == 1 else -1
class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        d1 = collections.defaultdict(lambda:[])
        d2 = collections.defaultdict(lambda:[])
        d3 = collections.defaultdict(lambda:[])
        
        count = [0]*4
        for e in edges:
            t = e[0]
            u = e[1]
            v = e[2]
            
            if t == 1:
                d1[u].append(v)
                d1[v].append(u)
            elif t == 2:
                d2[u].append(v)
                d2[v].append(u)
            else:
                d1[u].append(v)
                d1[v].append(u)
                d2[u].append(v)
                d2[v].append(u)
                d3[u].append(v)
                d3[v].append(u)
            count[t] += 1
        
        print(count)
        #print(d1)
        #print(d2)
        #print(d3)
        
        def check(d):
            visited = [False]*(n+1)
            self.sz = 0
            def trav(i):
                if visited[i] == True:
                    return
                self.sz += 1
                visited[i] = True
                #print(d[i])
                for j in d[i]:
                    trav(j)
            trav(1)
            #print(visited, self.sz)
            return self.sz ==n
        
        if(not check(d1)):
            return -1
        
        if(not check(d2)):
            return -1
        
        
        comps = 0
        
        visited = [False]*(n+1)
        
        def travers(x):
            if visited[x] == True:
                return
            
            visited[x] = True
            for y in d3[x]:
                travers(y)
            
        
        for i in range(1, n+1):
            if visited[i] == False:
                comps += 1
                travers(i)
            #print(i, comps)
        print(('comps', comps))
        
        
        
        print(((count[1]-comps+1) , (count[2]-comps+1), count[3]-n+comps))
        return (count[1]-comps+1) + (count[2]-comps+1)+count[3]-n+comps

class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        par = [i for i in range(n+1)]
        wei = [1 for i in range(n+1)]
        
        def find(i, p):
            if p[i] != p[p[i]]:
                p[i] = find(p[i], p)
            return p[i]
        
        def united(i, j, p):
            return find(i, p) == find(j, p)
        
        def unite(i, j, p, w):
            zi = find(i, p)
            zj = find(j, p)
            if w[zi] > w[zj]:
                p[zj] = zi
                w[zi] = w[zi] + w[zj]
                w[zj] = 0
            else:
                p[zi] = zj
                w[zj] = w[zi] + w[zj]
                w[zi] = 0
                
        edges.sort()
        ans = 0
        for t, a, b in reversed(edges):
            if t != 3: break
            if united(a,b, par): ans += 1
            else: unite(a, b, par, wei)
                
        p = [None, par[:], par[:] ]
        w = [None, wei[:], wei[:] ]
        for t, a, b in edges:
            if t > 2: break
            if united(a, b, p[t]): ans += 1
            else: unite(a, b, p[t], w[t])
        
        for i in range(2, n+1):
            if not united(1, i, p[1]): return -1
            if not united(1, i, p[2]): return -1
            
        return ans
class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        e0, e1, res = 0, 0, 0
        self.root = [i for i in range(n+1)]
        
        def find(x):
            if x == self.root[x]:
                return x
            self.root[x] = find(self.root[x])
            return self.root[x]
        
        def isReachable(x, y):
            x, y = find(x), find(y)
            if x == y:
                return True
            self.root[x] = y
            return False
        
        for t,i,j in edges:
            if t==3:
                if isReachable(i,j):
                    res += 1
                else:
                    e0 += 1
                    e1 += 1
                    
        root_cpy = self.root[:]
        for t,i,j in edges:
            if t==1:
                if isReachable(i,j):
                    res += 1
                else:
                    e0 += 1
                    
        self.root = root_cpy
        for t,i,j in edges:
            if t == 2:
                if isReachable(i,j):
                    res += 1
                else:
                    e1 += 1
                    
        return res if e0==e1==n-1 else -1
                    

class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        def find(i):
            if parent[i] != i: parent[i] = find(parent[i])
            return parent[i]
            
        def union(x, y):
            x, y = find(x), find(y)
            if x == y: return True
            parent[x] = y
            return False
        
        res = a = b = 0
        parent = list(range(n+1))
        
        for t, u, v in edges:
            if t == 3:
                if union(u, v):
                    res += 1
                else:
                    a += 1
                    b += 1
                
        p1 = parent[:]
        for t, u, v in edges:
            if t == 1:
                if union(u, v):
                    res += 1
                else:
                    a += 1
        
        parent = p1
        for t, u, v in edges:
            if t == 2:
                if union(u, v):
                    res += 1
                else:
                    b += 1
                        
        return res if a == b == n-1 else -1
class UnionFindSet:
    def __init__(self, n):
        self.p, self.c = [i for i in range(n)], [1] * n
    
    def find(self, v):
        if self.p[v] != v: self.p[v] = self.find(self.p[v])
        return self.p[v]
    
    def union(self, v1, v2):
        p1, p2 = self.find(v1), self.find(v2)
        if p1 == p2: return False
        if self.c[p1] < self.c[p2]: p1, p2 = p2, p1
        self.p[p2] = p1
        self.c[p1] += self.c[p2]
        return True
    
    def count(self, v):
        return self.c[self.find(v)]

class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        ret = 0
        ufs1, ufs2 = UnionFindSet(n + 1), UnionFindSet(n + 1)
        for t, u, v in edges:
            if t == 3:
                f = False
                if ufs1.union(u, v): f = True
                if ufs2.union(u, v): f = True
                if f: ret += 1
        for t, u, v in edges:
            if t == 1 and ufs1.union(u, v): ret += 1
            if t == 2 and ufs2.union(u, v): ret += 1
        if ufs1.count(1) != n or ufs2.count(1) != n: return -1
        return len(edges) - ret
'''
n is the number of nodes
if 1 < value of nodes <= n
Krustal O(ElogE)

rank[node]: the longth depth of node's children
'''

class Solution:
    def maxNumEdgesToRemove(self, n, edges):
        # Union find
        def find(node):
            if node != parent[node]:
                parent[node] = find(parent[node])
            return parent[node]

        def union(node1, node2):
            parent1, parent2 = find(node1), find(node2)
            if parent1 == parent2: return 0
            if rank[parent1] > rank[parent2]:
                parent[parent2] = parent1
            elif rank[parent1] == rank[parent2]:
                parent[parent2] = parent1
                rank[parent1] += 1
            else:
                parent[parent1] = parent2 
            
            return 1

        res = union_times_A = union_times_B = 0

        # Alice and Bob
        parent = [node for node in range(n + 1)]
        rank = [0 for node in range(n + 1)]
        
        for t, node1, node2 in edges:
            if t == 3:
                if union(node1, node2):
                    union_times_A += 1
                    union_times_B += 1
                else:
                    res += 1
        parent0 = parent[:]  # Alice union will change the parent array, keep origin for Bob

        # only Alice
        for t, node1, node2 in edges:
            if t == 1:
                if union(node1, node2):
                    union_times_A += 1
                else:
                    res += 1

        # only Bob
        parent = parent0
        for t, node1, node2 in edges:
            if t == 2:
                if union(node1, node2):
                    union_times_B += 1
                else:
                    res += 1
# only if Alice and Bob both union n-1 times, the graph is connected for both of them
        return res if union_times_A == union_times_B == n - 1 else -1
class UnionFind:
    def __init__(self, n):
        self.roots = [i for i in range(n)]
        
    def union(self, index1, index2):
        root1 = self.find(index1)
        root2 = self.find(index2)
        if root1 < root2:
            self.roots[root2] = root1
            return 1
        elif root1 > root2:
            self.roots[root1] = root2
            return 1
        return 0
    
    def find(self, index):
        if self.roots[index] != index:
            self.roots[index] = self.find(self.roots[index])
        return self.roots[index]

class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        edges.sort(key=lambda x: -x[0])
        
        uf1 = UnionFind(n)
        uf2 = UnionFind(n)
        could_delete = 0
        num_union1, num_union2 = 0, 0
        for i, edge in enumerate(edges):
            if edge[0] == 1:
                could_union = uf1.union(edge[1] - 1, edge[2] - 1)
                num_union1 += could_union
            elif edge[0] == 2:
                could_union = uf2.union(edge[1] - 1, edge[2] - 1)
                num_union2 += could_union
            else:
                could_union1 = uf1.union(edge[1] - 1, edge[2] - 1)
                could_union2 = uf2.union(edge[1] - 1, edge[2] - 1)
                num_union1 += could_union1
                num_union2 += could_union2
                could_union = could_union1 and could_union2
            could_delete += 1 - could_union

        if num_union1 != n - 1 or num_union2 != n - 1:
            return -1
        return could_delete
                
            

class UnionFindSet:
    def __init__(self, n):
        self.parents = list(range(n))
        self.ranks = [1] * n
        self.size = 1
        
    def find(self, u):
        if u != self.parents[u]:
            self.parents[u] = self.find(self.parents[u])
        return self.parents[u]
    
    def union(self, u, v):
        pu, pv = self.find(u), self.find(v)
        if pu == pv:
            return False
        if self.ranks[pu] > self.ranks[pv]:
            self.parents[pv] = pu
        elif self.ranks[pv] > self.ranks[pu]:
            self.parents[pu] = pv
        else:
            self.parents[pu] = pv
            self.ranks[pv] += 1
        self.size += 1
        return True
    
    
class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        uf1, uf2, ans = UnionFindSet(n), UnionFindSet(n), 0
		
        for t, u, v in edges:
            if t != 3:
                continue
            if not uf1.union(u - 1, v - 1) or not uf2.union(u - 1, v - 1):
                ans += 1
        
        for t, u, v in edges:
            if t == 1 and not uf1.union(u - 1, v - 1):
                ans += 1
            elif t == 2 and not uf2.union(u - 1, v - 1):
                ans += 1
   
        return ans if uf1.size == n and uf2.size == n else -1
from typing import *
from copy import deepcopy

class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        edges.sort(key=lambda x:(-x[0]))
        num_used = 0
        uf = [[i for i in range(n)]] + [[0] * n]
        i = 0
        while i < len(edges):
            t, s, d = edges[i]
            s -= 1
            d -= 1
            if t < 3:
                break
            i += 1
            if self.union_find_merge(uf, s, d):
                num_used += 1
        uf1 = deepcopy(uf)
        uf2 = deepcopy(uf)

        while i < len(edges):
            t, s, d = edges[i]
            s -= 1
            d -= 1
            if t < 2:
                break
            i += 1
            if self.union_find_merge(uf2, s, d):
                num_used += 1

        while i < len(edges):
            t, s, d = edges[i]
            s -= 1
            d -= 1
            i += 1
            if self.union_find_merge(uf1, s, d):
                num_used += 1

        if self.find_num_components(n, uf1) > 1 or self.find_num_components(n, uf2) > 1:
            return -1

        return len(edges) - num_used

    def find_num_components(self, n, uf):
        num_components = 0
        for idx in range(n):
            parent = uf[0][idx]
            if idx == parent:
                num_components += 1
        return num_components

    def union_find_merge(self, uf, node1, node2):
        p1 = self.union_find_get_parent(uf, node1)
        p2 = self.union_find_get_parent(uf, node2)

        if p1 == p2:
            return False  # Returning false so that we don't include the s,d in result

        if uf[1][p1] > uf[1][p2]:
            uf[0][p2] = p1
        else:
            uf[0][p1] = p2
            uf[1][p2] = max(uf[1][p2], uf[1][p1] + 1)

        return True

    def union_find_get_parent(self, uf, node):
        while uf[0][node] != node:
            node = uf[0][node]
        return node
class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        # Union find
        def find(i):
            if i != root[i]:
                root[i] = find(root[i])
            return root[i]

        def uni(x, y):
            x, y = find(x), find(y)
            if x == y: return 0
            root[x] = y
            return 1

        res = e1 = e2 = 0

        # Alice and Bob
        root = list(range(n + 1))
        for t, i, j in edges:
            if t == 3:
                if uni(i, j):
                    e1 += 1
                    e2 += 1
                else:
                    res += 1
        root0 = root[:]

        # only Alice
        for t, i, j in edges:
            if t == 1:
                if uni(i, j):
                    e1 += 1
                else:
                    res += 1

        # only Bob
        root = root0
        for t, i, j in edges:
            if t == 2:
                if uni(i, j):
                    e2 += 1
                else:
                    res += 1

        return res if e1 == e2 == n - 1 else -1
class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        dtype = collections.defaultdict(set)
        
        for e in edges:
            dtype[(e[1], e[2])].add(e[0])
        
        res = 0
        
        for k in dtype:
            if 3 in dtype[k]:
                if 1 in dtype[k]:
                    res += 1
                    dtype[k].remove(1)
                if 2 in dtype[k]:
                    res += 1
                    dtype[k].remove(2)
        
        da = collections.defaultdict(set)
        db = collections.defaultdict(set)
        
        for k in dtype:
            if (1 in dtype[k]) or (3 in dtype[k]):
                da[k[0]].add(k[1])
                da[k[1]].add(k[0])
            if (2 in dtype[k]) or (3 in dtype[k]):
                db[k[0]].add(k[1])
                db[k[1]].add(k[0])
        
        def traversable(dd):
            q = collections.deque([1])
            v = set([1])
            
            while q:
                node = q.popleft()
                for nei in dd[node]:
                    if (nei not in v):
                        q.append(nei)
                        v.add(nei)
            if len(v) != n:
                return False
            return True
                        
        
        if (not traversable(da)) or (not traversable(db)):
            return -1
        
        
        d3 = collections.defaultdict(set)
        
        for k in dtype:
            if (3 in dtype[k]):
                d3[k[0]].add(k[1])
                d3[k[1]].add(k[0])
        
        def components(dd):
            r = []
            
            v = set()
            nodes = list(dd.keys())
            lastvsize = 0
            for node in nodes:
                if node not in v:
                    v.add(node)
                    q = collections.deque([node])
                    
                    while q:
                        node = q.popleft()
                        for nei in dd[node]:
                            if (nei not in v):
                                q.append(nei)
                                v.add(nei)
                    r.append(len(v) - lastvsize)
                    lastvsize = len(v)
            return r
        
        d3ComponentSizes = components(d3)
        need3 = 0
        for compSize in d3ComponentSizes:
            need3 += compSize - 1 
        d3nodes = len(list(d3.keys()))
        need1 = len(d3ComponentSizes) - 1 + n - d3nodes
        need2 = len(d3ComponentSizes) - 1 + n - d3nodes
        print(d3ComponentSizes)
        print((len(edges), need1, need2, need3))
        return len(edges) - (need1 + need2 + need3)

class DSU:
    def __init__(self, N):
        self.par = list(range(N))
        self.sz = [1] * N

    def find(self, x):
        if self.par[x] != x:
            self.par[x] = self.find(self.par[x])
        return self.par[x]

    def union(self, x, y):
        xr, yr = self.find(x), self.find(y)
        if xr == yr:
            return False
        if self.sz[xr] < self.sz[yr]:
            xr, yr = yr, xr
        self.par[yr] = xr
        self.sz[xr] += self.sz[yr]
        self.sz[yr] = self.sz[xr]
        return True

    def size(self, x):
        return self.sz[self.find(x)]

class Solution(object):
    def maxNumEdgesToRemove(self, N, edges):
        for row in edges:
            # row[0] -= 1
            row[1] -= 1
            row[2] -= 1
        alice = []
        bob = []
        both = []
        
        for t, u, v in edges:
            if t == 1:
                alice.append([u,v])
            elif t==2:
                bob.append([u,v])
            else:
                both.append([u,v])
        dsu1 = DSU(N)
        dsu2 = DSU(N)
        ans = 0
        for u,v  in both:
            dsu2.union(u,v)
            if not dsu1.union(u, v):
                ans += 1
        for u,v  in alice:
            if not dsu1.union(u,v): ans += 1
        for u,v in bob:
            if not dsu2.union(u,v): ans += 1
        
        if dsu1.size(0) != N:
            return -1
        if dsu2.size(0) != N:
            return -1
        return ans
class DisjointSet:
    def __init__(self, number_of_sites):
        self.parent = [i for i in range(number_of_sites+1)]
        self.children_site_count = [1 for _ in range(number_of_sites+1)]
        self.component_count = number_of_sites

    def find_root(self, site):
        root = site
        while root != self.parent[root]:
            root = self.parent[root]
        while site != root:
            site, self.parent[site] = self.parent[site], root
        return root

    def is_connected(self, site_1, site_2):
        return self.find_root(site_1) == self.find_root(site_2)

    def union(self, site_1, site_2):
        site_1_root = self.find_root(site_1)
        site_2_root = self.find_root(site_2)
        if site_1_root == site_2_root:
            return False

        if self.children_site_count[site_1_root] < self.children_site_count[site_2_root]:
            self.parent[site_1_root] = site_2_root
            self.children_site_count[site_2_root] += self.children_site_count[
                site_1_root]
        else:
            self.parent[site_2_root] = site_1_root
            self.children_site_count[site_1_root] += self.children_site_count[
                site_2_root]
        self.component_count -= 1
        return True


class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        alice_disjoint_set = DisjointSet(n)
        bob_disjoint_set = DisjointSet(n)

        TYPE_OF_COMMON_EDGES = 3
        TYPE_OF_ALICE_EDGES = 1
        TYPE_OF_BOB_EDGES = 2

        common_edges = filter(lambda edge: edge[0] == TYPE_OF_COMMON_EDGES, edges)
        alice_edges = filter(lambda edge: edge[0] == TYPE_OF_ALICE_EDGES, edges)
        bob_edges = filter(lambda edge: edge[0] == TYPE_OF_BOB_EDGES, edges)

        redundant = 0
        for _, u, v in common_edges:
            unioned_in_alice = alice_disjoint_set.union(u, v)
            unioned_in_bob = bob_disjoint_set.union(u, v)
            if (not unioned_in_alice) or (not unioned_in_bob):
                redundant += 1

        for _, u, v in bob_edges:
            if not bob_disjoint_set.union(u,v):
                redundant += 1
                
        for _, u, v in alice_edges:
            if not alice_disjoint_set.union(u, v):
                redundant += 1
        
        return redundant if alice_disjoint_set.component_count == 1 and bob_disjoint_set.component_count == 1 else -1
class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        parent = list(range(n + 1))
        rank = [0 for _ in range(n + 1)]
        def find(x):
            while parent[x] != x:
                x = parent[x]
            return x
        def union(x, y):
            id_x = find(x)
            id_y = find(y)
            if id_x == id_y:
                return 0
            if rank[id_x] >= rank[id_y]:
                if rank[id_x] == rank[id_y]:
                    rank[id_x] += 1
                parent[id_y] = id_x
            else:
                parent[id_x] = id_y
            return 1
        res = a = b = 0
        for t, i, j in edges:
            if t == 3:
                if union(i, j):
                    a += 1
                    b += 1
                else:
                    res += 1
        parent_, rank_ = parent[:], rank[:]
        for t, i, j in edges:
            if t == 1:
                if union(i, j):
                    a += 1
                else:
                    res += 1
        parent, rank = parent_, rank_
        for t, i, j in edges:
            if t == 2:
                if union(i, j):
                    b += 1
                else:
                    res += 1
        return res if (a == n - 1 and b == n - 1) else -1


class DAU():
    def __init__(self,n):
        self.parent = list(range(n ))
        
    def find(self,p):
        if self.parent[p] != p:
            self.parent[p] = self.find(self.parent[p])
        return self.parent[p]
    def union(self,p,q):
        pr, qr = self.find(p), self.find(q)
        if pr == qr:
            return False
        else:
            self.parent[pr] = qr
            return True

class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        ufA, ufB, ufAB = DAU(n), DAU(n), DAU(n)
        usefulAB = 0
        for edge in edges:
            t = edge[0]
            x = edge[1]
            y = edge[2]
            if t == 1:
                ufA.union(x-1, y - 1)
            elif t == 2:
                ufB.union(x-1, y - 1)
            else:
                ufA.union(x-1, y - 1)
                ufB.union(x-1, y - 1)
                usefulAB += ufAB.union(x-1, y - 1)
        if len([i for i in range(n) if ufA.parent[i] == i]) > 1 or len([i for i in range(n) if ufB.parent[i] == i]) > 1 :
            return -1
        return len(edges) - (2*(n - 1) - usefulAB)
                
            

class DSU:
    def __init__(self, N):
        self.par = list(range(N))
        self.sz = [1] * N

    def find(self, x):
        if self.par[x] != x:
            self.par[x] = self.find(self.par[x])
        return self.par[x]

    def union(self, x, y):
        xr, yr = self.find(x), self.find(y)
        if xr == yr:
            return False
        if self.sz[xr] < self.sz[yr]:
            xr, yr = yr, xr
        self.par[yr] = xr
        self.sz[xr] += self.sz[yr]
        self.sz[yr] = self.sz[xr]
        return True

    def size(self, x):
        return self.sz[self.find(x)]
    

class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        
        edges.sort(reverse=True)
        
        bob = DSU(n)
        alice = DSU(n)
        
        res = 0
        
        for t, a, b in edges:
            a -= 1
            b -= 1
            if t == 3 and (bob.find(a) != bob.find(b) or alice.find(a) != alice.find(b)):
                bob.union(a, b)
                alice.union(a, b)
            elif t == 2 and bob.find(a) != bob.find(b):
                bob.union(a, b)
            elif t == 1 and alice.find(a) != alice.find(b):
                alice.union(a, b)
            else:
                res += 1
        
        is_one = lambda dsu: len({dsu.find(i) for i in range(n)}) == 1
        
        return res if is_one(alice) and is_one(bob) else -1

from collections import defaultdict
class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:    
        # 1u3001u5173u4e8eu4e24u4e2au4ebau7684u8fdeu901au6027u8003u5bdfuff0c Ouff08Nuff09
        # 2u3001u5f00u59cbu79fbu9664 u6709type3u548cu5176u4ed6u7684u8fb9u91cdu590duff0cu4f18u5148u79fbu9664u5176u4ed6u7684
        # 3u3001u53bbu6389type3u7ec4u6210u7684u73afuff08u7528u8282u70b9u6570u548cu8fb9u7684u6570u91cfu5173u7cfbu6765u8ba1u7b97u9700u8981u53bbu9664u591au5c11u4e2au591au4f59u7684u8fb9uff09
        # 4u3001u7136u540eu5355u72ecu8003u5bdfu6bcfu4e2au4ebau7684u91cdu590du8fb9uff08u7528u8fb9u548cu8282u70b9u6570u91cfu5173u7cfbu8ba1u7b97uff09
        count = 0
        type3 = set()
        adj3 = defaultdict(list)
        for edge in edges:
            edge_type, node1, node2 = edge
            if edge_type==3:
                type3.add((node1,node2))  
                adj3[node1].append(node2)
                adj3[node2].append(node1)
        
        type1,type2 = set(),set()
        adj1,adj2 = defaultdict(list),defaultdict(list)
        for edge in edges:
            edge_type, node1, node2 = edge
            if edge_type==3:
                continue
            if (node1,node2) in type3 or (node2,node1) in type3:
                count += 1
                continue
            if edge_type==1:
                type1.add((node1,node2))
                adj1[node1].append(node2)
                adj1[node2].append(node1)
            elif edge_type==2:
                type2.add((node1,node2))
                adj2[node1].append(node2)
                adj2[node2].append(node1)
        
        # u8fdeu901au6027u8003u5bdf
        visited = set(range(1,n+1))
        queue = [1]
        while queue and visited:
            new_queue = []
            while queue:
                curr = queue.pop()
                if curr in visited:
                    visited.remove(curr)
                    for node in adj1[curr]+adj3[curr]:
                        if node in visited:
                            new_queue.append(node)
            queue = new_queue
        if visited: # u8bf4u660eu4e0du80fdu904du5386
            return -1
        visited = set(range(1,n+1))
        queue = [1]
        while queue and visited:
            new_queue = []
            while queue:
                curr = queue.pop()
                if curr in visited:
                    visited.remove(curr)
                    for node in adj2[curr]+adj3[curr]:
                        if node in visited:
                            new_queue.append(node)
            queue = new_queue
        if visited: # u8bf4u660eu4e0du80fdu904du5386
            return -1
        
        # type3, adj3
        # u9700u8981u5728u641cu7d22u8fc7u7a0bu4e2du540cu65f6u8bb0u5f55u8fb9u548cu8282u70b9
        used = set()
        for node1,node2 in type3:
            if node1 in used:
                continue
            edge_record = set()
            node_record = set()
            queue = [node1]
            while queue:
                new_queue = []
                while queue:
                    curr = queue.pop()
                    used.add(curr)
                    node_record.add(curr)
                    for node in adj3[curr]:
                        if node in used:
                            continue
                        if (curr,node) in type3:
                            edge_record.add((curr,node))
                        else:
                            edge_record.add((node, curr))
                        new_queue.append(node)
                queue = new_queue
            count -= len(edge_record) - len(node_record) + 1
     
        
        # u53bbu9664type3u7684u73af
        return count + len(type1)+len(type3) - n +1 + len(type2)+len(type3)-n+1
        
            
        
                    
            
        

class UnionFind:
    # When n is valid, each element is a tuple of two integers, (x, y)
    def __init__(self, m: int, n: int = None):
        self.rank = collections.Counter()
        if n is None:
            self.parent = [i for i in range(m)]
        else:
            self.parent = {(i, j): (i, j) for i in range(m) for j in range(n)}

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        px = self.find(x)
        py = self.find(y)
        if px == py:
            return
        if self.rank[py] > self.rank[px]:
            px, py = py, px
        if self.rank[py] == self.rank[px]:
            self.rank[px] += 1
        self.parent[py] = px


class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        uf1 = UnionFind(n)
        uf2 = UnionFind(n)
        res = 0
        e1 = e2 = e3 = 0
        for e in edges:
            t, u, v = e[0], e[1] - 1, e[2] - 1
            if t == 3:
                if uf1.find(u) == uf1.find(v):
                    res += 1
                else:
                    uf1.union(u, v)
                    uf2.union(u, v)
                    e3 += 1
        for e in edges:
            t, u, v = e[0], e[1] - 1, e[2] - 1
            if t == 1:
                if uf1.find(u) == uf1.find(v):
                    res += 1
                else:
                    uf1.union(u, v)
                    e1 += 1
            elif t == 2:
                if uf2.find(u) == uf2.find(v):
                    res += 1
                else:
                    uf2.union(u, v)
                    e2 += 1
        total = (n - 1) * 2
        if e3 * 2 + e1 + e2 < total:
            return -1
        return res

class UnionFindSet:
    def __init__(self, n):
        self.parents = list(range(n))
        self.ranks = [1] * n
        self.size = 1
        
    def find(self, u):
        if u != self.parents[u]:
            self.parents[u] = self.find(self.parents[u])
        return self.parents[u]
    
    def union(self, u, v):
        pu, pv = self.find(u), self.find(v)
        if pu == pv:
            return False
        if self.ranks[pu] > self.ranks[pv]:
            self.parents[pv] = pu
        elif self.ranks[pv] > self.ranks[pu]:
            self.parents[pu] = pv
        else:
            self.parents[pu] = pv
            self.ranks[pv] += 1
        self.size += 1
        return True
    
    
class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        uf1, uf2 = UnionFindSet(n), UnionFindSet(n)
        ans = 0
        for t, u, v in edges:
            if t != 3:
                continue
            if not uf1.union(u - 1, v - 1) or not uf2.union(u - 1, v - 1):
                ans += 1
        
        for t, u, v in edges:
            if t == 1 and not uf1.union(u - 1, v - 1):
                ans += 1
            elif t == 2 and not uf2.union(u - 1, v - 1):
                ans += 1
   
        print(uf1.size, uf2.size)
        return ans if uf1.size == n and uf2.size == n else -1

class DSU:
    
    def __init__(self, a):
        self.par = {x:x for x in a}
    
    def merge(self, u, v):
        rootu = self.find(u)
        rootv = self.find(v)
        
        if rootu == rootv:
            return False
        
        self.par[rootu] = rootv
        return True
    
    def find(self, u):
        if self.par[u] != u:
            self.par[u] = self.find(self.par[u])
        return self.par[u]
    
    def roots(self):
        return set(self.find(u) for u in self.par)

class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        
        dsu1 = DSU(range(1,n+1))
        dsu2 = DSU(range(1,n+1))
        
        grouper = defaultdict(list)
        for t,u,v in edges:
            grouper[t].append([u,v])
        
        both, alice, bob = grouper[3], grouper[1], grouper[2]
        
        ret = 0
        
        for u,v in both:
            if not dsu1.merge(u, v):
                ret += 1
            dsu2.merge(u, v)
                
        for u,v in alice:
            if not dsu1.merge(u, v):
                ret += 1
        
        for u,v in bob:
            if not dsu2.merge(u, v):
                ret += 1
        
        if len(dsu1.roots()) != 1 or len(dsu2.roots()) != 1:
            return -1
            
        return ret
class DisjointSet:
    def __init__(self, n):
        self._parent = [i for i in range(n)]
        self._count = [1 for _ in range(n)]
    
    def parent(self, i):
        p = i
        while self._parent[p] != p:
            p = self._parent[p]
        self._parent[i] = p
        return p
    
    def count(self, i):
        return self._count[self.parent(i)]
    
    def merge(self, i, j):
        pi = self.parent(i)
        pj = self.parent(j)
        
        if pi == pj:
            return False
        ci, cj = self._count[pi], self._count[pj]
        if ci <= cj:
            self._parent[j] = self._parent[pj] = pi
            self._count[pi] += self._count[pj]
            self._count[pj] = 0
        else:
            self._parent[i] = self._parent[pi] = pj
            self._count[pj] += self._count[pi]
            self._count[pi] = 0
        return True
            
    def clone(self):
        other = DisjointSet(len(self._parent))
        other._parent = [p for p in self._parent]
        other._count = [c for c in self._count]
        return other


class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        edges_deleted = 0
        dsA = DisjointSet(n)
        for edge_type, u, v in edges:
            ui, vi = u-1, v-1
            if edge_type == 3:
                if not dsA.merge(ui, vi):
                    edges_deleted += 1
        
        dsB = dsA.clone()
        
        for edge_type, u, v in edges:
            ui, vi = u-1, v-1
            if edge_type == 1:
                if not dsA.merge(ui, vi):
                    edges_deleted += 1
                    
        if sum(c > 0 for c in dsA._count) != 1:
            return -1

        for edge_type, u, v in edges:
            ui, vi = u-1, v-1
            if edge_type == 2:
                if not dsB.merge(ui, vi):
                    edges_deleted += 1

        if sum(c > 0 for c in dsB._count) != 1:
            return -1
                    
        return edges_deleted
class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        a_uf = UF(n+1)
        b_uf = UF(n+1)
        unwanted = 0

        for t, u, v in edges:    
            if t == 3:            
                # Alice and Bob
                if a_uf.find(u) == a_uf.find(v) and b_uf.find(u) == b_uf.find(v):
                    # both guys dont need
                    unwanted += 1
                else:
                    a_uf.union(u, v)
                    b_uf.union(u, v)
        
        for t, u, v in edges:
            #print((t, u, v))
            if t == 1:
                # Alice
                if a_uf.find(u) == a_uf.find(v):
                    # dont need this
                    unwanted += 1
                else:
                    a_uf.union(u, v)
                    
        if a_uf.size[a_uf.find(1)] < n:
            return -1

        for t, u, v in edges:    
            if t == 2:
                # Bob
                if b_uf.find(u) == b_uf.find(v):
                    # dont need this
                    unwanted += 1
                else:
                    b_uf.union(u, v)
                
        if b_uf.size[b_uf.find(1)] < n:
            return -1
                
        return unwanted
    
class UF:
    def __init__(self, n):
        self.uf = [i for i in range(n)]
        self.size = [1] * n
    
    def find(self, u):
        while u != self.uf[u]:
            self.uf[u] = self.uf[self.uf[u]]
            u = self.uf[u]
        return u
    
    def union(self, u, v):
        rootU = self.find(u)
        rootV = self.find(v)
        if rootU != rootV:
            if self.size[rootU] > self.size[rootV]:
                self.size[rootU] += self.size[rootV]
                self.uf[rootV] = rootU
            else:
                self.size[rootV] += self.size[rootU]
                self.uf[rootU] = rootV
        
        

class UnionFind():
    def __init__(self):
        self.uf, self.rank, self.size = {}, {}, {}
        self.roots = set()
        
    def add(self, x):
        if x not in self.uf:
            self.uf[x], self.rank[x], self.size[x] = x, 0, 1
            self.roots.add(x)
        
    def find(self, x):
        self.add(x)
        if x != self.uf[x]:
            self.uf[x] = self.find(self.uf[x])
        return self.uf[x]

    def union(self, x, y):  
        xr, yr = self.find(x), self.find(y)
        if xr == yr: return
        if self.rank[xr] <= self.rank[yr]:
            self.uf[xr] = yr
            self.size[yr] += self.size[xr]
            self.rank[yr] += (self.rank[xr] == self.rank[yr])
            self.roots.discard(xr)
        else:
            self.uf[yr] = xr
            self.size[xr] += self.size[yr]
            self.roots.discard(yr)

class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        Ga, Gb, Gab = UnionFind(), UnionFind(), UnionFind()
        for x in range(1, n + 1):
            Ga.add(x), Gb.add(x), Gab.add(x)
        for t, x, y in edges:
            if t in (1, 3): Ga.union(x, y)
            if t in (2, 3): Gb.union(x, y)
            if t == 3: Gab.union(x,y)
        
        if max(len(Ga.roots), len(Gb.roots)) > 1: return -1
        c = len(Gab.roots)
        return len(edges) - (n - c + 2 * (c - 1))
class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        edge_set = set(tuple(x) for x in edges)
        
        graphs = [defaultdict(list) for i in range(3)]
        for e in edges:
            graphs[e[0]-1][e[1]].append(e[2])
            graphs[e[0]-1][e[2]].append(e[1])
            if e[0] == 3:
                for k in [0, 1]:
                    graphs[k][e[1]].append(e[2])
                    graphs[k][e[2]].append(e[1])

        def ct_ccmp(g):
            visited = dict()
            q = deque()
            nt = 0
            for i in range(1, n + 1):
                if i in visited:
                    continue
                if len(g[i]) > 0:
                    q.append(i)
                    visited[i] = nt
                    while len(q) > 0:
                        cur = q.popleft()
                        for x in g[cur]:
                            if x in visited:
                                continue
                            q.append(x)
                            visited[x] = nt
                nt += 1
            return nt
    
        if ct_ccmp(graphs[0]) > 1 or ct_ccmp(graphs[1]) > 1:
            return -1
        
        nt = ct_ccmp(graphs[2])
        return len(edges) - (n - nt) - 2 * (nt - 1)
                

class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:

        def find(comp_id):
            while parents[comp_id] != comp_id:
                parents[comp_id] = parents[parents[comp_id]]
                comp_id = parents[comp_id]
            return comp_id

        def union(id_1, id_2):
            px = find(id_1)
            py = find(id_2)
            parents[py] = px
            return px != py

        # use union find for components
        # init each node as a component
        # once an edge is found (use num 3 first) then
        # union the components, if there is change, then amount of components change
        # if only one component left then the graph is traversable
        parents = [idx for idx in range(n)]
        removed_edges = 0
        for link in edges:
            if link[0] != 3:
                continue
            # only double linked edges first
            if not union(link[1] - 1, link[2] - 1):
                removed_edges += 1

        bob_parents = parents[:]
        for link in edges:
            if link[0] != 1:
                continue
            if not union(link[1] - 1, link[2] - 1):
                removed_edges += 1
        if len(Counter(find(i) for i in parents)) != 1:  # LLRN - use find() instead of referring directly
            return -1

        parents = bob_parents
        for link in edges:
            if link[0] != 2:
                continue
            if not union(link[1] - 1, link[2] - 1):
                removed_edges += 1
        if len(Counter(find(i) for i in parents)) != 1:
            return -1
        return removed_edges
class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        ufAli = uf(n)
        ufBob = uf(n)
        
#         10 -> 2
        for edg in edges:
            x, y = edg[1], edg[2]
            if edg[0] == 1:
                ufAli.addEdge(x, y, 10)
            elif edg[0] == 2:
                ufBob.addEdge(x, y, 10)
            else:
                ufAli.addEdge(x, y, 1)
                ufBob.addEdge(x, y, 1)
                
        # print(ufAli.g, ufAli.kruskalmst())
        # print(ufBob.g, ufBob.kruskalmst())
        
        blueremoved = set()
        aliremoved = set()
        bobremoved = set()
        
        ans1 = ufAli.kruskalmst(blueremoved, aliremoved)
        ans2 = ufBob.kruskalmst(blueremoved, bobremoved)
        if ans1 == -1 or ans2 == -1:
            return -1
        
        # return ans1 + ans2
        return len(blueremoved) + len(aliremoved) + len(bobremoved)
        
        
                
        
        
        

class uf:
    def __init__(self, n):
        self.n = n
        self.g = []
        self.joinednodes = set()
        # self.totalnodes = set()
        
        
    def addEdge(self, x, y, cost):
        self.g.append((x, y, cost))
        # self.joinednodes 
        
    def find(self, x, parent):
        if parent[x] == x:
            return x
        
        return self.find(parent[x], parent)
    
    def union(self, x, y, parent, rank):
        xroot, yroot = self.find(x, parent), self.find(y, parent)
        
        if xroot != yroot:
            if rank[xroot] > rank[yroot]:
                parent[yroot] = xroot
            elif rank[yroot] > rank[xroot]:
                parent[xroot] = yroot
            else:
                parent[yroot] = xroot
                rank[xroot] += 1
                
    def kruskalmst(self, blue, rorg):
        # parent = { for edge in g}
        parent = {}
        rank = {}
        for edge in self.g:
            parent[edge[0]] = edge[0]
            parent[edge[1]] = edge[1]
            rank[edge[0]] = 0
            rank[edge[1]] = 0
            
        # print(parent, rank)
        success = 0
        self.g.sort(key=lambda edge: edge[2])
        for edge in self.g:
            x, y, cos = edge
            xroot = self.find(x, parent)
            yroot = self.find(y, parent)
            if xroot != yroot:
                success += 1
                self.union(xroot, yroot, parent, rank)
                
            else:
                if cos == 1:
                    blue.add((x,y))
                else:
                    rorg.add((x,y))
                
                
        
                
        if success == self.n -1:
            
            # return success
            return len(self.g) - success
        
        return -1
            
            
            
            
            
        
                
                
        
        

from collections import defaultdict, deque
class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        # 1u3001u79fbu9664type3u7684u91cdu590du8fb9
        # 2u3001u68c0u67e5aliceu548cbobu7684u53efu904du5386u6027
        # 3u3001u68c0u67e5type3u7ec4u6210u7684u73af
        # 4u3001u68c0u67e5alice type1u7ec4u6210u7684u73af
        # 5u3001u68c0u67e5bob type2u7ec4u6210u7684u73af
        count = 0
        set1, set2, set3 = set(), set(), set()
        adj_a, adj_b, adj_3 = defaultdict(set), defaultdict(set), defaultdict(set)
        for edge in edges:
            tp, i, j = edge
            if tp == 3:
                set3.add((i,j))
                adj_3[i].add(j)
                adj_3[j].add(i)
                adj_a[i].add(j)
                adj_a[j].add(i)
                adj_b[i].add(j)
                adj_b[j].add(i)
        
        for edge in edges:
            tp, i, j = edge
            if tp != 3:
                if ((i,j) in set3 or (j,i) in set3):
                    count += 1
                elif tp == 1:
                    set1.add((i,j))
                    adj_a[i].add(j)
                    adj_a[j].add(i)
                elif tp == 2:
                    set2.add((i,j))
                    adj_b[i].add(j)
                    adj_b[j].add(i)
                    
        def is_traversable(adj):
            visited = set()
            queue = deque([1])
            while queue:
                root = queue.popleft()
                visited.add(root)
                for i in adj[root]:
                    if i not in visited:
                        queue.append(i)
            if len(visited) != n:
                return False
            else:
                return True
            
        
        if not is_traversable(adj_a) or not is_traversable(adj_b):
            return -1
        
        dup_3 = 0
        visited = set()
        for edge in set3:
            if edge in visited:
                continue
            node_set = set()
            edge_set = set()
            i, j = edge
            queue = deque([i])
            while queue:
                root = queue.popleft()
                node_set.add(root)
                for k in adj_3[root]:
                    if k not in node_set:
                        queue.append(k)
                        if (root, k) in set3:
                            edge_set.add((root, k))
                        else:
                            edge_set.add((k, root))
                            
            dup_3 += len(edge_set) - len(node_set) + 1
            for v_edge in edge_set:
                visited.add(v_edge)
                
        type3_count = len(set3) - dup_3        
        return count + len(set1) + 2 * type3_count - n + 1 + len(set2) - n + 1 + dup_3

        
        
            
        
        
                
            
                    
            

class UnionFind:
    def __init__(self,N):
        self.par = [-1]*N
        self.N = N
    
    def find(self,x):
        if self.par[x] < 0:
            return x
        else:
            self.par[x] = self.find(self.par[x])
            return self.par[x]
        
    def union(self,x,y):
        x = self.find(x)
        y = self.find(y)
        if x == y:
            return
        if self.par[x] > self.par[y]:
            x,y = y,x
        
        self.par[x] += self.par[y]
        self.par[y] = x
        
    def roots(self):
        return [i for i, x in enumerate(self.par) if x < 0]

    def groupCount(self):
        return len(self.roots())

class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        uf = UnionFind(n)
        uf1 = UnionFind(n)
        uf2 = UnionFind(n)
        ans = 0
        m = len(edges)
        for edge in edges:
            if edge[0] == 3:
                if uf.find(edge[1]-1) == uf.find(edge[2]-1):
                    ans += 1
                uf.union(edge[1]-1,edge[2]-1)
                uf1.union(edge[1]-1,edge[2]-1)
                uf2.union(edge[1]-1,edge[2]-1)
            elif edge[0] == 1:
                uf1.union(edge[1]-1,edge[2]-1)
                ans += 1
            else:
                uf2.union(edge[1]-1,edge[2]-1)
                ans += 1
        connected = uf.groupCount()
        removable = ans + 2 - 2*connected
        if max(uf1.groupCount(),uf2.groupCount()) > 1 or removable < 0:
            return -1
        if connected == 1:
            return ans
        else:
            return removable
        

class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        parenta = [x for x in range(n+1)]
        parentb = [x for x in range(n+1)]
        
        def ufind(parent, x):
            if parent[x] != x:
                parent[x] = ufind(parent, parent[x])
            return parent[x]
        
        def uunion(parent, a, b):
            ua = ufind(parent, a)
            ub = ufind(parent, b)
            
            parent[ua] = ub
            
        edges.sort(key=lambda x: (-x[0]))
        
        count = 0
        for t, u, v in edges:
            if t == 3:
                if ufind(parenta, u) != ufind(parenta, v) or ufind(parentb, u) != ufind(parentb, v):
                    uunion(parenta, u, v)
                    uunion(parentb, u, v)
                else:
                    count += 1
            elif t == 2:
                if ufind(parentb, u) != ufind(parentb, v):
                    uunion(parentb, u, v)
                else:
                    count += 1
            else:
                if ufind(parenta, u) != ufind(parenta, v):
                    uunion(parenta, u, v)
                else:
                    count += 1
            
        roota = ufind(parenta, 1)
        rootb = ufind(parentb, 1)
        for x in range(1, n+1):
            if ufind(parenta, x) != roota or ufind(parentb, x) != rootb:
                return -1
            
        return count
class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        self.parent = [i for i in range(n+1)]
        res=e1=e2=0
        for t,i,j in edges:
            if t==3:
                if self.union(i,j):
                    e1+=1
                    e2+=1
                else:
                    res+=1
        self.parent0 = self.parent[:]
        for t,i,j in edges:
            if t==1:
                if self.union(i,j):
                    e1+=1
                else:
                    res+=1
        self.parent = self.parent0
        for t,i,j in edges:
            if t==2:
                if self.union(i,j):
                    e2+=1
                else:
                    res+=1
        return res if e1==e2==n-1 else -1
    def find(self,i):
        if i!=self.parent[i]:
            self.parent[i]=self.find(self.parent[i])
        return self.parent[i]
    def union(self,x,y):
        x,y = self.find(x),self.find(y)
        if x==y:
            return 0
        self.parent[x]=y
        return 1
    

class DSU:
    def __init__(self, N):
        self.par = list(range(N))
        self.sz = [1] * N

    def find(self, x):
        if self.par[x] != x:
            self.par[x] = self.find(self.par[x])
        return self.par[x]

    def union(self, x, y):
        xr, yr = self.find(x), self.find(y)
        if xr == yr:
            return False
        if self.sz[xr] < self.sz[yr]:
            xr, yr = yr, xr
        self.par[yr] = xr
        self.sz[xr] += self.sz[yr]
        self.sz[yr] = self.sz[xr]
        return True

    def size(self, x):
        return self.sz[self.find(x)]
class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        '''
        for vertex
            if has type3 edge remove all other edges
            else:
                 
        '''
        dsuA = DSU(n+1)
        dsuB = DSU(n+1)
        
        ans = 0
        for t, u, v in edges:
            if t == 3:
                if not dsuA.union(u, v):
                    ans += 1
                dsuB.union(u, v)
        for t, u, v in edges:
            if t == 1:
                if not dsuA.union(u, v):
                    ans += 1
            if t == 2:
                if not dsuB.union(u, v):
                    ans += 1
        return ans if dsuA.size(1) == dsuB.size(1) == n else -1
#         hasType3 = [False for _ in range(n)]
#         graph = collections.defaultdict(list)
#         count = 0
#         for a, b, c in edges:
#             if a == 3:
#                 hasType3[b-1] = True
#                 hasType3[c-1] = True
#             graph[b].append([c, a])
#             graph[c].append([b, a])
#         seenA = [False for i in range(n)]
#         seenB = [False for i in range(n)]
#         def dfs(node, ty):
#             for nei, t in graph[node]:
#                 if ty == 1:
#                     if (t == 1 or t == 3) and not seenA[nei-1]:
#                         seenA[nei-1] = True
#                         dfs(nei, ty)
#                 if ty == 2:
#                     if (t == 2 or t == 3) and not seenB[nei-1]:
#                         seenB[nei-1] = True
#                         dfs(nei, ty)
#         dfs(edges[0][1], 1)
#         dfs(edges[0][1], 2)
#         seenA[edges[0][1]-1] = True
#         seenB[edges[0][1]-1] = True
#         # print(seenA, seenB)
#         if not all(seenA) or not all(seenB):
#             return -1
#         ans = 0
#         for i, a in enumerate(hasType3):
#             if not a:
#                 ans += 2
#             else:
#                 ans += 1

#         return max(len(edges) - ans+1, 0)

class DSU:
    def __init__(self, N):
        self.par = list(range(N))
        self.sz = [1] * N

    def find(self, x):
        if self.par[x] != x:
            self.par[x] = self.find(self.par[x])
        return self.par[x]

    def union(self, x, y):
        xr, yr = self.find(x), self.find(y)
        if xr == yr:
            return False
        if self.sz[xr] < self.sz[yr]:
            xr, yr = yr, xr
        self.par[yr] = xr
        self.sz[xr] += self.sz[yr]
        self.sz[yr] = self.sz[xr]
        return True

    def size(self, x):
        return self.sz[self.find(x)]

class Solution(object):
    def maxNumEdgesToRemove(self, N, edges):
        for row in edges:
            # row[0] -= 1
            row[1] -= 1
            row[2] -= 1
        alice = []
        bob = []
        both = []
        
        for t, u, v in edges:
            if t == 1:
                alice.append([u,v])
            elif t==2:
                bob.append([u,v])
            else:
                both.append([u,v])
        dsu1 = DSU(N)
        dsu2 = DSU(N)
        ans = 0
        for u,v  in both:
            dsu2.union(u,v)
            if not dsu1.union(u, v):
                ans += 1
        for u,v  in alice:
            if not dsu1.union(u,v): ans += 1
        for u,v in bob:
            if not dsu2.union(u,v): ans += 1
        
        if dsu1.size(0) != N:
            return -1
        if dsu2.size(0) != N:
            return -1
        return ans

class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        edges.sort(reverse=True)
        a_uf = UF(n+1)
        b_uf = UF(n+1)
        unwanted = 0
        
        for t, u, v in edges:
            #print((t, u, v))
            if t == 1:
                # Alice
                if a_uf.find(u) == a_uf.find(v):
                    # dont need this
                    unwanted += 1
                else:
                    a_uf.union(u, v)
            elif t == 2:
                # Bob
                if b_uf.find(u) == b_uf.find(v):
                    # dont need this
                    unwanted += 1
                else:
                    b_uf.union(u, v)
            else:
                # Alice and Bob
                if a_uf.find(u) == a_uf.find(v) and b_uf.find(u) == b_uf.find(v):
                    # both guys dont need
                    unwanted += 1
                else:
                    a_uf.union(u, v)
                    b_uf.union(u, v)
                
        if a_uf.size[a_uf.find(1)] < n or b_uf.size[b_uf.find(1)] < n:
            return -1
                
        return unwanted
    
class UF:
    def __init__(self, n):
        self.uf = [i for i in range(n)]
        self.size = [1] * n
    
    def find(self, u):
        while u != self.uf[u]:
            self.uf[u] = self.uf[self.uf[u]]
            u = self.uf[u]
        return u
    
    def union(self, u, v):
        rootU = self.find(u)
        rootV = self.find(v)
        if rootU != rootV:
            if self.size[rootU] > self.size[rootV]:
                self.size[rootU] += self.size[rootV]
                self.uf[rootV] = rootU
            else:
                self.size[rootV] += self.size[rootU]
                self.uf[rootU] = rootV
        
        

import copy

class Solution:
    
    def findParent(self, v, parent):
            if parent[v] == -1:
                return v
            else:
                return self.findParent(parent[v], parent)
    
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        bothEdges = []
        aEdges = []
        bEdges = []
        
        aNodes = set()
        bNodes = set()
        
        for e in edges:
            if e[0] == 3:
                bothEdges.append(e)
                aNodes.add(e[2])
                aNodes.add(e[1])
                bNodes.add(e[2])
                bNodes.add(e[1])
            elif e[0] == 1:
                aEdges.append(e)
                aNodes.add(e[2])
                aNodes.add(e[1])
            else:
                bEdges.append(e)
                bNodes.add(e[2])
                bNodes.add(e[1])
                
        if len(aNodes) < n or len(bNodes) < n:
            return -1
        
        parents = [-1 for _ in range(n + 1)]

        mstCommon = 0
        for e in bothEdges:
            x, y = e[1], e[2]
            xp = self.findParent(x, parents)
            yp = self.findParent(y, parents)
            if xp == yp:
                continue
            else:
                parents[xp] = yp
                mstCommon += 1
                if mstCommon == n - 1:
                    break
        
        if mstCommon == n - 1:
            return len(edges) - (n - 1)
        else:
            return len(edges) - (n - 1) - (n - 1) + mstCommon
class Solution:
    def maxNumEdgesToRemove(self, n: int, edges) -> int:
        def find(i):
            if root[i] != i:
                root[i] = find(root[i])
            return root[i]

        def union(u, v):
            ru, rv = find(u), find(v)
            if ru == rv:
                return False
            root[ru] = rv
            return True

        root = list(range(n+1))
        edges_alice, edges_bob, res = 0, 0, 0
        for t, u, v in edges:
            if t == 3:
                if union(u, v):
                    edges_alice += 1
                    edges_bob += 1
                else:
                    res += 1

        root_copy = root[:] # a copy of connection 3
        for t, u, v in edges:
            if t == 1:
                if union(u, v):
                    edges_alice += 1
                else:
                    res +=1

        root = root_copy
        for t, u, v in edges:
            if t == 2:
                if union(u, v):
                    edges_bob += 1
                else:
                    res += 1
        if edges_alice == n-1 and edges_bob == n-1:
            return res
        else:
            return -1
class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:

        def find(x,y,d):
            nonlocal res
            count = 0
            arr = []
            for key,val in d.items():
                if x in val and y in val:
                    res += 1
                    count = -1
                    break
                elif x in val or y in val:
                    d[key].add(x)
                    d[key].add(y)
                    count += 1
                    arr.append(key)
            
            if count == -1:
                pass
            elif count == 0:
                d[min(x,y)] = set({x,y})
            elif count == 1:
                pass
            else: # union
                d[min(arr)].update(d[max(arr)])
                del d[max(arr)]
            return d
                
                    
        # variables
        d  = {1:set({1})}
        da = dict()
        db = dict()
        res = 0
        
        # sort edges
        a,b,c= [],[],[]
        for t,i,j in edges:
            if t == 3:
                a.append([t,i,j])
            elif t == 2:
                b.append([t,i,j])
            else:
                c.append([t,i,j])
        
        # main function
        
        # t == 3
        for t,i,j in a:
            d = find(i,j,d)
        da = d
        db = deepcopy(d)

        # t == 2
        for t,i,j in b:
            db = find(i,j,db)

        # t == 1
        for t,i,j in c:
            da = find(i,j,da)
            
        if da[1] == db[1] == set(range(1,n+1)):
            return res
        else:
            return -1
import copy


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n + 1))
        
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py:
            return 0
        
        self.parent[px] = py
        return 1


class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        uf, res, e1, e2 = UnionFind(n), 0, 0, 0
        for _, u, v in [x for x in edges if x[0] == 3]:
            add = uf.union(u, v)
            if add:
                e1 += 1
                e2 += 1
            else:
                res += 1
                
        uf1 = copy.deepcopy(uf)
        for _, u, v in [x for x in edges if x[0] == 1]:
            add = uf1.union(u, v)
            if add:
                e1 += 1
            else:
                res += 1        
                
        uf2 = uf
        for _, u, v in [x for x in edges if x[0] == 2]:
            add = uf2.union(u, v)
            if add:
                e2 += 1
            else:
                res += 1                    
        
        return res if e1 == n - 1 and e2 == n - 1 else -1
class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        
        # add type 3 first 
        if n == 1: return True
        parentsA = {}
        
        def findA(p):
            if p not in parentsA:
                parentsA[p] = p
            if parentsA[p] != p:
                parentsA[p] = findA(parentsA[p])
            return parentsA[p]
    
        def unionA(p, q):
            i, j = findA(p), findA(q)
            if i != j:
                parentsA[i] = j
        
        def isconnectedA(p, q):
            return findA(p) == findA(q)
        
        parentsB = {}
        
        def findB(p):
            if p not in parentsB:
                parentsB[p] = p
            if parentsB[p] != p:
                parentsB[p] = findB(parentsB[p])
            return parentsB[p]
    
        def unionB(p, q):
            i, j = findB(p), findB(q)
            if i != j:
                parentsB[i] = j
        
        def isconnectedB(p, q):
            return findB(p) == findB(q)
        
        edges.sort(reverse = True)
        # first add in best edges
    
            
        skip = 0
        for typ, fr, to in edges:
            if typ == 3:
                if isconnectedA(fr, to) and isconnectedB(fr, to):
                    skip += 1
                else:
                    unionA(fr, to)
                    unionB(fr, to)
            elif typ == 1:
                if isconnectedA(fr, to):
                    skip += 1
                else:
                    unionA(fr, to)
            elif typ == 2:
                if isconnectedB(fr, to):
                    skip += 1
                else:
                    unionB(fr, to)
            # print(typ, fr, to, parentsB)
                    
        # print(parentsA)
        # print(parentsB)
        
        allpA = set()
        for i in range(1, n+1):
            allpA.add(findA(i))
        
        allpB = set()
        for i in range(1, n+1):
            allpB.add(findB(i))
            
        # print(allpB)
        if len(allpA) == 1 and len(allpB) == 1: return skip
        return -1
                
                
                
                
                
                
                
                
                
                
                
            
            

import collections
import sys
sys.setrecursionlimit(1000000)
class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        to = collections.defaultdict(list)
        for t, a, b in edges:
            to[a-1].append((b-1,t-1))
            to[b-1].append((a-1,t-1))
        
        def is_connected(etype=0):
            remain = [1]*n
            def dfs(n=0):
                if remain[n] == 0: return
                remain[n] = 0
                for nn, t in to[n]:
                    if t == etype or t == 2:
                        dfs(nn)
            dfs()
            return len([1 for r in remain if r ==1]) == 0
        if not is_connected(0): return -1
        if not is_connected(1): return -1
        
        ids = [i for i in range(n)]
        def find(i):
            if i == ids[i]: return i
            ni = find(ids[i])
            ids[i] = ni
            return ni
        def union(i, j):
            i = find(i)
            j = find(j)
            if i == j: return False
            ids[j] = i
            return True
        
        e = 0
        for t, a, b in edges:
            if t == 3:
                if union(a-1, b-1):
                    e += 1
        ids2 = list(ids)
        
        for t, a, b in edges:
            if t == 1:
                if union(a-1, b-1):
                    e += 1
                    
        ids = ids2
        for t, a, b in edges:
            if t == 2:
                if union(a-1, b-1):
                    e += 1
        return len(edges) - e


class DSU:
    def __init__(self,n):
        self.node = list(range(n+1))
        self.rank = [1]*(n+1)
    
    def find(self,x):
        if self.node[x] != x:
            self.node[x] = self.find( self.node[x] )
        return self.node[x]
    
    def union(self,x,y):
        xid, yid = self.find(x), self.find(y)
        if xid == yid:
            return False
        else:
            xrank = self.rank[xid]
            yrank = self.rank[yid]
            if xrank<= yrank:
                self.node[xid] = yid
                self.rank[yid] += xrank
            else:
                self.node[yid] = xid
                self.rank[xid] += yrank
            return True

class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        alice = []
        bob = []
        both = []
        for t,u,v in edges:
            if t==1:
                alice.append([u,v])
            elif t==2:
                bob.append([u,v])
            elif t==3:
                both.append([u,v])
        adsu = DSU(n)
        bdsu = DSU(n)
        ans = 0
        for u,v in both:
            T1 =  adsu.union(u,v) 
            T2 =  bdsu.union(u,v)
            if not T1 and not T2:
                ans += 1
        for u,v in alice:
            if not adsu.union(u,v):
                ans += 1
        for u,v in bob:
            if not bdsu.union(u,v):
                ans += 1
        for i in range(n+1):
            adsu.find(i)
            bdsu.find(i)
        return ans if len(set(adsu.node))==2 and len(set(bdsu.node))==2 else -1
                
        
        
        
        


class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:

        def find(x,y,d):
            nonlocal res
            count = 0
            arr = []
            for key,val in d.items():
                if x in val and y in val:
                    res += 1
                    count = -1
                    break
                elif x in val or y in val:
                    d[key].add(x)
                    d[key].add(y)
                    count += 1
                    arr.append(key)
            
            if count == -1:
                pass
            elif count == 0:
                d[min(x,y)] = set({x,y})
            elif count == 1:
                pass
            else: # union
                d[min(arr)].update(d[max(arr)])
                del d[max(arr)]
            return d
                
                    
        # variables
        d  = {1:set({1})}
        da = dict()
        db = dict()
        res = 0
        
        # sort edges
        a,b,c= [],[],[]
        for t,i,j in edges:
            if t == 3:
                a.append([t,i,j])
            elif t == 2:
                b.append([t,i,j])
            else:
                c.append([t,i,j])
        
        # main function
        
        # t == 3
        for t,i,j in a:
            d = find(i,j,d)
        da = deepcopy(d)
        db = deepcopy(d)

        # t == 2
        for t,i,j in b:
            db = find(i,j,db)

        # t == 1
        for t,i,j in c:
            da = find(i,j,da)
            
        if da[1] == db[1] == set(range(1,n+1)):
            return res
        else:
            return -1
class DisjointSet:
    def __init__(self, number_of_sites):
        self.parent = [i for i in range(number_of_sites)]
        self.children_site_count = [1 for _ in range(number_of_sites)]
        self.component_count = number_of_sites

    def find_root(self, site):
        root = site
        while root != self.parent[root]:
            root = self.parent[root]
        while site != root:
            site, self.parent[site] = self.parent[site], root
        return root

    def is_connected(self, site_1, site_2):
        return self.find_root(site_1) == self.find_root(site_2)

    def union(self, site_1, site_2):
        site_1_root = self.find_root(site_1)
        site_2_root = self.find_root(site_2)
        if site_1_root == site_2_root:
            return False

        if self.children_site_count[site_1_root] < self.children_site_count[site_2_root]:
            self.parent[site_1_root] = site_2_root
            self.children_site_count[site_2_root] += self.children_site_count[
                site_1_root]
        else:
            self.parent[site_2_root] = site_1_root
            self.children_site_count[site_1_root] += self.children_site_count[
                site_2_root]
        self.component_count -= 1
        return True


class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        alice_disjoint_set = DisjointSet(n)
        bob_disjoint_set = DisjointSet(n)

        TYPE_OF_COMMON_EDGES = 3
        TYPE_OF_ALICE_EDGES = 1
        TYPE_OF_BOB_EDGES = 2

        common_edges = filter(lambda edge: edge[0] == TYPE_OF_COMMON_EDGES, edges)
        alice_edges = filter(lambda edge: edge[0] == TYPE_OF_ALICE_EDGES, edges)
        bob_edges = filter(lambda edge: edge[0] == TYPE_OF_BOB_EDGES, edges)

        redundant = 0
        for _, u, v in common_edges:
            if (not alice_disjoint_set.union(u-1, v-1)) or (not bob_disjoint_set.union(u-1, v-1)):
                redundant += 1

        for _, u, v in bob_edges:
            if not bob_disjoint_set.union(u-1,v-1):
                redundant += 1
                
        for _, u, v in alice_edges:
            if not alice_disjoint_set.union(u-1, v-1):
                redundant += 1
        
        return redundant if alice_disjoint_set.component_count == 1 and bob_disjoint_set.component_count == 1 else -1
class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:     
        graph = {1: [], 2: [], 3: []}
        for t, i, j in edges:
            graph[t].append((i, j))
        def find(x):
            if parents[x] != x:
                parents[x] = find(parents[x])
            return parents[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px == py:
                return False
            if rank[px] < rank[py]:
                px, py = py, px
            parents[py] = px
            rank[px] += rank[px] == rank[py]
            return True
        
        result = 0
        parents = list(range(n + 1))
        rank = [1] * (n + 1)
        for i, j in graph[3]:
            if not union(i, j):
                result += 1
        parents_copy = parents[:]
        rank_copy = rank[:]
        for i, j in graph[1]:
            if not union(i, j):
                result += 1
        if len(set(find(i) for i in range(1, n + 1))) > 1:
            return -1
        parents = parents_copy
        rank = rank_copy
        for i, j in graph[2]:
            if not union(i, j):
                result += 1
        if len(set(find(i) for i in range(1, n + 1))) > 1:
            return -1
        return result

from collections import defaultdict

class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        edges = [(t, u-1, v-1) for t,u,v in edges]
        
        common = []
        arr = []
        brr = []
        
        for t, u, v in edges:
            if t == 1:
                arr.append((u,v))
            if t == 2:
                brr.append((u,v))
            if t == 3:
                common.append((u,v))
        
        d = defaultdict(list)
        
        for u,v in common:
            d[u].append(v)
            d[v].append(u)
        
        sizes = []
        visited = [False for _ in range(n)]
        for i in range(n):
            if visited[i] or not d[i]:
                continue
            size = 1
            stack = [i]
            visited[i] = True
            while stack:
                cur = stack.pop()
                for nex in d[cur]:
                    if visited[nex]:
                        continue
                    visited[nex] = True
                    stack.append(nex)
                    size += 1
            sizes.append(size)
        
        ################################################
        
        d = defaultdict(list)
        for u,v in common:
            d[u].append(v)
            d[v].append(u)
        
        for u,v in arr:
            d[u].append(v)
            d[v].append(u)
        
        visited = [False for _ in range(n)]
            
        for i in range(n):
            if visited[i] or not d[i]:
                continue
            stack = [i]
            visited[i] = True
            while stack:
                cur = stack.pop()
                for nex in d[cur]:
                    if visited[nex]:
                        continue
                    visited[nex] = True
                    stack.append(nex)
        
        if not all(visited):
            return -1
        
        ################################################
        
        d = defaultdict(list)
        for u,v in common:
            d[u].append(v)
            d[v].append(u)
        
        for u,v in brr:
            d[u].append(v)
            d[v].append(u)
        
        visited = [False for _ in range(n)]
            
        for i in range(n):
            if visited[i] or not d[i]:
                continue
            stack = [i]
            visited[i] = True
            while stack:
                cur = stack.pop()
                for nex in d[cur]:
                    if visited[nex]:
                        continue
                    visited[nex] = True
                    stack.append(nex)
        
        if not all(visited):
            return -1
        
        ################################################
        
        
        expected_commons = sum(x-1 for x in sizes)
        res = len(common) - expected_commons
        
        expected_specific = n - expected_commons - 1
        res += len(arr) + len(brr) - 2*expected_specific

        print((len(common), expected_commons, len(arr), len(brr), expected_specific))
        
        return res

from collections import defaultdict

class Solution:
    
    def find(self, parent, i):
        if parent[i] == i:
            return i
        parent [i] = self.find(parent, parent[i])
        return parent[i]
    
    def union(self, parent, rank, x, y):
        xroot = self.find(parent, x)
        yroot = self.find(parent, y)
  
        if rank[xroot] < rank[yroot]:
            parent[xroot] = yroot
        elif rank[xroot] > rank[yroot]:
            parent[yroot] = xroot
        else : 
            parent[yroot] = xroot
            rank[xroot] += 1
    
    def helpMST(self, adj_list, n):
        
        parent = []
        rank = [0] * (n+1)
        result = []
        
        for node in range(n+1): 
            parent.append(node)
        
        for t, u, v in adj_list:
            x = self.find(parent, u)
            y = self.find(parent, v)
            
            if x != y:     
                result.append((t,u,v)) 
                self.union(parent, rank, x, y)
                
        parent_count = 0
        for index, num in enumerate(parent[1:]):
            if index + 1 == num:
                parent_count += 1
        
        return parent_count == 1, result
    
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        
        adj_list1 = []
        adj_list2 = []
        
        common = set()
        for t, u, v in edges:
            if t == 3:
                common.add((u, v))
                adj_list1.append((t,u,v))
                adj_list2.append((t,u,v))
        
        for t, u, v in edges:
            if t in (1,2) and (u,v) not in common:
                if t == 1:
                    adj_list1.append((t,u,v))
                elif t == 2:
                    adj_list2.append((t,u,v))
                    
        result = set()
        
        if len(adj_list1) < n-1 or len(adj_list2) < n-1:
            return -1
        elif len(adj_list1) == n-1 and len(adj_list2) == n - 1:
            result = result.union(set(adj_list2))
            result = result.union(set(adj_list1))
            return len(edges) - len(result)
        
        if len(adj_list1) > n-1:
            possible, res = self.helpMST(adj_list1, n)
            if not possible:
                return -1
            result = result.union(set(res))
        else:
            result = result.union(set(adj_list1))
        
        if len(adj_list2) > n-1:
            possible, res = self.helpMST(adj_list2, n)
            if not possible:
                return -1
            result = result.union(set(res))
        else:
            result = result.union(set(adj_list2))
            
        return len(edges) - len(result)

class Solution:
    def maxNumEdgesToRemove(self, n: int, e: List[List[int]]) -> int:
        def union(UF, u, v):            
            UF[find(UF, v)] = find(UF, u)
        def find(UF, u):
            if UF[u] != u: UF[u] = find(UF, UF[u])
            return UF[u]         
        def check(UF, t):            
            UF = UF.copy()
            for tp, u, v in e:
                if tp == t: 
                    if find(UF, u) == find(UF, v): self.ans += 1
                    else: union(UF, u, v)
            return len(set(find(UF, u) for u in UF)) == 1, UF

        self.ans, UF = 0, {u: u for u in range(1, n+1)}                
        UF = check(UF, 3)[1]
        if not check(UF, 1)[0] or not check(UF, 2)[0]: return -1        
        return self.ans

class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        A = [i for i in range(n)]
        def find(u):
            if A[u] != u:
                A[u] = find(A[u])
            return A[u]
        def union(u, v):
            pu, pv = find(u), find(v)
            if pu == pv:
                return False
            A[max(pu,pv)] = min(pu,pv)
            return True
        
        #u5148u505aCommonu7684u904du5386
        edges = sorted(edges, reverse=True)
        # print("u770bu770bu987au5e8fu5bf9u4e0du5bf9", edges)
        
        i = 0
        m = len(edges)
        res = 0
        while i < m:
            cur = edges[i]
            if cur[0] == 3:
                #u4ece0u5f00u59cbu7f16u53f7
                if union(cur[1]-1,cur[2]-1) == False:
                    res += 1
            else:
                break
            i += 1
        # print("common res", res)
        
        #u627eBobu7684
        origin_A = deepcopy(A)
        while i < m:
            cur = edges[i]
            if cur[0] == 2:
                #u4ece0u5f00u59cbu7f16u53f7
                if union(cur[1]-1,cur[2]-1) == False:
                    res += 1
            else:
                break
            i += 1
        # print("Bob and Common", res, A)
        for j in range(1, n):
            #u67e5u770bu662fu5426u6240u6709u7684u8282u70b9u90fdu80fdu591fu52300
            # print(j)
            p = find(j)
            if  p != 0:
                # print("Bob " + str(j) + " can't to 0, only to " + str(p))
                return -1
        
        #Alice
        A = origin_A
        while i < m:
            cur = edges[i]
            if cur[0] == 1:
                #u4ece0u5f00u59cbu7f16u53f7
                if union(cur[1]-1,cur[2]-1) == False:
                    res += 1
            else:
                break
            i += 1
            
        # print("Alice and Common", A)
        for j in range(1, n):
            #u67e5u770bu662fu5426u6240u6709u7684u8282u70b9u90fdu80fdu591fu52300
            p = find(j)
            if p != 0:
                # print("Alice " + str(j) + " can't to 0, only to " + str(p))
                return -1
            
        return res
        
        

class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        ufa = UnionFind(n) # Graph for Alice
        ufb = UnionFind(n) # Graph for Bob
        cnt = 0 # number of removable edges
        
        for x, y, z in edges:
            if x == 3:
                flag1 = ufa.union(y, z)
                flag2 = ufb.union(y, z)
                if not flag1 or not flag2: cnt +=1

        for x, y, z in edges:
            if x == 1:
                flag = ufa.union(y, z)
                if not flag: cnt += 1
            if x == 2:
                flag = ufb.union(y, z)
                if not flag: cnt += 1

        return cnt if ufa.groups == 1 and ufb.groups == 1 else -1
            
        
class UnionFind():
    def __init__(self, n):
        self.parents = {i:i for i in range(1, n+1)}
        self.groups = n

    def find(self, x):
        if self.parents[x] == x:
            return x
        else:
            self.parents[x] = self.find(self.parents[x])
            return self.parents[x]

    def union(self, x, y):
        x = self.find(x)
        y = self.find(y)

        if x == y:
            return False

        self.parents[y] = x
        self.groups -= 1
        return True
class DSU:
    def __init__(self, size):
        self.indexes = {i:i for i in range(size)}
        self.sizes = {i:1 for i in range(size)}
        self.com = size
        
    def root(self, i):
        node = i
        while i!=self.indexes[i]:
            i = self.indexes[i]
        
        while node!=i:
            nnode = self.indexes[node]
            self.indexes[node] = i
            node = nnode
            
        return i
    
    def unite(self, i, j):
        ri , rj = self.root(i), self.root(j)
        if ri==rj:
            return
        
        self.indexes[ri] = rj
        self.sizes[rj] += self.sizes[ri]
        self.com-=1
        
class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        
        alice = DSU(n)
        bob = DSU(n)
        
        ans = 0
        
        for t, u, v in edges:
            if t==3:
                aru, arv = alice.root(u-1), alice.root(v-1)
                bru, brv = bob.root(u-1), bob.root(v-1)
                if aru==arv and bru==brv:
                    ans += 1
                else:
                    alice.unite(u-1, v-1)
                    bob.unite(u-1, v-1)
        
                
        
        for t,u,v in edges:
            if t==1:
                ru, rv = alice.root(u-1), alice.root(v-1)
                if ru==rv:
                    ans += 1
                else:
                    alice.unite(u-1, v-1)
                    
            elif t==2:
                ru, rv = bob.root(u-1), bob.root(v-1)
                if ru==rv:
                    ans += 1
                else:
                    bob.unite(u-1, v-1)
                
        
        if alice.com!=1 or bob.com!=1:
            return -1
        
        return ans
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        edges.sort(key=lambda x: (-x[0], x[1]))
        removed = set()
        
        def find_root(parents, i):
            if parents[i] == i:
                return i, 0
            a, b = find_root(parents, parents[i])
            return a, b+1
        
        def uf(n, edges, types):
            parents = [x for x in range(n+1)]
            group = 1
            for type, a, b in edges:
                
                if type not in types:
                    continue
                r_a, n_a = find_root(parents, a)
                r_b, n_b = find_root(parents, b)
                if r_a != r_b:
                    if n_a >= n_b:
                        parents[r_b] = r_a
                    else:
                        parents[r_a] = r_b
                    group+=1
                else:
                    removed.add((type, a, b))
            return group == n
            
        
        if not uf(n, edges, [3,1]):
            return -1
        if not uf(n, edges, [3,2]):
            return -1
        return len(removed)
# awice
class DSU:
    def __init__(self, N):
        self.parents = list(range(N))
        self.sz = [1]*N
    
    def find(self, u):
        if self.parents[u] != u:
            self.parents[u] = self.find(self.parents[u])
        return self.parents[u]
    
    def union(self, u, v):
        pu, pv = self.find(u), self.find(v)
        if pu == pv:    return False
        if self.sz[pv] > self.sz[pu]:
            pu, pv = pv, pu
        self.parents[pv] = pu
        self.sz[pu] += self.sz[pv]
        self.sz[pv] = self.sz[pu]
        
        return True
    
    def get_sz(self, u):
        return self.sz[self.find(u)]
        
class Solution:
    # weight edges with full link weighted 2?
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        # union find
        alice, bob, full = [], [], []
        
        for e in edges:
            u, v = e[1]-1, e[2]-1
            if e[0] == 1:   alice.append([u,v])
            elif e[0] == 2: bob.append([u,v])
            else:   full.append([u,v])
        
        res = 0
        dsua, dsub = DSU(n), DSU(n)
        for u, v in full:
            dsua.union(u, v)
            if not dsub.union(u, v):
                res += 1
            
        for u, v in alice:
            if not dsua.union(u, v):
                res += 1
        for u, v in bob:
            if not dsub.union(u, v):
                res += 1
        
        if dsua.get_sz(0) != n or dsub.get_sz(0) != n:  return -1
        return res

class DSU:
    def __init__(self, n):
        self.roots = [i for i in range(n)]
    def find(self, x):
        if self.roots[x] != x:
            self.roots[x] = self.find(self.roots[x])
        return self.roots[x]
    def union(self, x, y):
        self.roots[self.find(x)] = self.find(y)
        
class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        
        m = len(edges)
        edges = sorted(edges, key=lambda x: -x[0])
        type3_exist = set()
        removed = set()
        for i in range(len(edges)):
            typ, n1, n2 = edges[i]
            if (n1, n2) in type3_exist:
                removed.add(i)
                continue
                
            if typ == 3:
                type3_exist.add((n1, n2))
        
        edges = [edge for idx, edge in enumerate(edges) if idx not in removed]
        
        dsu_alice = DSU(n)
        dsu_bob = DSU(n)
        
        count = 0
        for edge in edges:
            typ, n1, n2 = edge
            if typ == 1:
                if dsu_alice.find(n1-1) != dsu_alice.find(n2-1):
                    count += 1
                    dsu_alice.union(n1-1, n2-1)
            if typ == 2:
                if dsu_bob.find(n1-1) != dsu_bob.find(n2-1):
                    count += 1
                    dsu_bob.union(n1-1, n2-1)
            if typ == 3:
                if dsu_bob.find(n1-1) != dsu_bob.find(n2-1) or dsu_alice.find(n1-1) != dsu_alice.find(n2-1):
                    count += 1
                    dsu_alice.union(n1-1, n2-1)
                    dsu_bob.union(n1-1, n2-1)
        
        if len(set([dsu_bob.find(i) for i in range(n)])) != 1 or len(set([dsu_alice.find(i) for i in range(n)])) != 1:
            return -1
        
        return m - count
class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        alice = DSU()
        bob = DSU()
        count = 0
        for t, u, v in edges:
            if t == 3:
                if alice.find(u) == alice.find(v) or bob.find(u) == bob.find(v):
                    count += 1
                else:
                    alice.union(u, v)
                    bob.union(u, v)
        
        for t, u, v in edges:
            if t == 1:
                if alice.find(u) == alice.find(v):
                    count += 1
                else:
                    alice.union(u, v)
        
        for t, u, v in edges:
            if t == 2:
                if bob.find(u) == bob.find(v):
                    count += 1       
                else:
                    bob.union(u, v)
                    
        if max(bob.count.values()) != n or max(alice.count.values()) != n:
            return -1
        
        return count
        
class DSU:
    def __init__(self):
        self.father = {}
        self.count = {}
    
    def find(self, a):
        self.father.setdefault(a, a)
        self.count.setdefault(a, 1)
        if a != self.father[a]:
            self.father[a] = self.find(self.father[a])
        return self.father[a]
    
    def union(self, a, b):
        _a = self.find(a)
        _b = self.find(b)
        if _a != _b:
            self.father[_a] = self.father[_b]
            self.count[_b] += self.count[_a]
            
            
            
            

class DSU:
    def __init__(self, n):
        self.p = list(range(n + 1))
        self.isolated_nodes = n
        
    def find(self, x):
        if x != self.p[x]:
            self.p[x] = self.find(self.p[x])
        return self.p[x]
    
    def union(self, x, y):
        xr = self.find(x)
        yr = self.find(y)
        
        if xr != yr:
            self.isolated_nodes -= 1
            
        self.p[xr] = yr

class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        #keep two union for Bob and Alice
        #add edges to the dsu, if the two nodes are already in a union, we can delete the edge
        #always add type3 first
        edges.sort(reverse = True)
        
        dsuA = DSU(n)
        dsuB = DSU(n)
        res = 0
        for [t, n1, n2] in edges:
            if t == 3:
                #handle Alice
                if dsuA.find(n1) == dsuA.find(n2) and dsuB.find(n1) == dsuB.find(n2):
                    #don't add the edge
                    res += 1
                else:
                    dsuA.union(n1, n2)
                    dsuB.union(n1, n2)
            elif t == 1:
                #Alice
                if dsuA.find(n1) == dsuA.find(n2):
                    res += 1
                else:
                    dsuA.union(n1, n2)
            else:
                if dsuB.find(n1) == dsuB.find(n2):
                    res += 1
                else:
                    dsuB.union(n1, n2)
                    
        return res if dsuA.isolated_nodes == 1 and dsuB.isolated_nodes == 1 else -1
                

from collections import defaultdict
class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        
        d = defaultdict(list)
        for t, u, v in edges:
            d[t].append((u - 1, v - 1))
            
        bob, alice = list(range(n)), list(range(n))
        
        def find(x, is_bob):
            if is_bob:
                if x != bob[x]:
                    bob[x] = find(bob[x], is_bob)
                return bob[x]
            else:
                if x != alice[x]:
                    alice[x] = find(alice[x], is_bob)
                return alice[x]
            
        res = 0
        for t in [3, 2, 1]:
            for u, v in d[t]:
                if t == 3:
                    rootu, rootv = find(u, True), find(v, True)
                    if rootu != rootv:
                        bob[rootu] = rootv
                        alice[rootu] = rootv
                        res += 1
                elif t == 1:
                    rootu, rootv = find(u, False), find(v, False)
                    if rootu != rootv:
                        alice[rootu] = rootv
                        res += 1
                else:
                    rootu, rootv = find(u, True), find(v, True)
                    if rootu != rootv:
                        bob[rootu] = rootv
                        res += 1
                        
        root_bob, root_alice = find(0, True), find(0, False)
        if all(find(num, True) == root_bob for num in bob) and all(find(num, False) == root_alice for num in alice):
            return len(edges) - res
        else:
            return -1
class Solution:
    def maxNumEdgesToRemove(self, n, edges):
        # Union find
        def find(i):
            if i != root[i]:
                root[i] = find(root[i])
            return root[i]

        def uni(x, y):
            x, y = find(x), find(y)
            if x == y: return 0
            root[x] = y
            return 1

        res = e1 = e2 = 0

        # Alice and Bob
        root = list(range(n + 1))
        for t, i, j in edges:
            if t == 3:
                if uni(i, j):
                    e1 += 1
                    e2 += 1
                else:
                    res += 1
        root0 = root[:]

        # only Alice
        for t, i, j in edges:
            if t == 1:
                if uni(i, j):
                    e1 += 1
                else:
                    res += 1

        # only Bob
        root = root0
        for t, i, j in edges:
            if t == 2:
                if uni(i, j):
                    e2 += 1
                else:
                    res += 1

        return res if e1 == e2 == n - 1 else -1
class unionFindSet:
    def __init__(self, S):
        self.parent = {i:i for i in S}
        self.rank = {i:1 for i in S}
        self.count = len(S) #number of groups
        
    def find(self, u):
        if u not in self.parent:
            return -1
        path = []
        while self.parent[u]!=u:
            path.append(u)
            u = self.parent[u]
        for p in path: #make tree flat
          	self.parent[p] = u
        return u
    
    def union(self, u, v):
        pu, pv = self.find(u), self.find(v)
        if pu==pv:
            return False
        if self.rank[pu] > self.rank[pv]: #union by rank
            pu, pv = pv, pu
        self.parent[pu] = pv #pu not u !!!
        self.rank[pv] += 1
        self.count -= 1
        return True

class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        #edges = set([tuple(i) for i in edges])
        data = unionFindSet([i for i in range(1, n + 1)])
        e1, e2, res = 0, 0, 0
        edges3 = [(t, u, v) for t, u, v in edges if t==3] 
        edges1 = [(t, u, v) for t, u, v in edges if t==1] 
        edges2 = [(t, u, v) for t, u, v in edges if t==2]
        ## 2 + 3
        for i in range(len(edges3)):
            t, u, v = edges3[i]
            if data.find(u) != data.find(v):
                data.union(u, v)
                e1 += 1
                e2 += 1
            else:
                res += 1
        data2 = copy.deepcopy(data) ##########
        
        for i in range(len(edges2)):
            t, u, v = edges2[i]
            if data.find(u) != data.find(v):
                data.union(u, v)
                e2 += 1
            else:
                res += 1
        if data.count != 1:
            return -1
        
        ## 2 + 3
        for i in range(len(edges1)):
            t, u, v = edges1[i]
            if data2.find(u) != data2.find(v):
                data2.union(u, v)
                e1 += 1
            else:
                res += 1
        if data2.count != 1:
            return -1

        return res
    #u5148 uf u516cu5171u8fb9u7136u540eu5b58u8d77u6765 u7136u540eu5206u522bu5bf9 alice u548c bob u505au4e00u4e0b


                
            
            
        
        
        
        
class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        edges1 = defaultdict(list)
        edges2 = defaultdict(list)
        
        for t,a,b in edges:
            if t == 3:
                edges1[a-1].append([0,b-1])
                edges1[b-1].append([0,a-1])
                edges2[a-1].append([0,b-1])
                edges2[b-1].append([0,a-1])
            elif t == 1:
                edges1[a-1].append([1,b-1])
                edges1[b-1].append([1,a-1])
            else:
                edges2[a-1].append([1,b-1])
                edges2[b-1].append([1,a-1])
        dis1 = [0] + [1] * (n-1)
        dis2 = [0] + [1] * (n-1)
        m = 0
        que1 = []
        que2 = []
        both = []
        for a,b in edges1[0]:
            if a == 0 :
                both.append(b)
            else:
                que1.append(b)
        for a,b in edges1[0]:
            if a == 0 :
                continue
            else:
                que2.append(b)           
        
        while both or que1:
            if both:
                D = both.pop()
                if dis1[D]:
                    m+= 1
                    dis1[D] = dis2[D] = 0
                else:
                    continue
            else:
                D = que1.pop()
                if dis1[D]:
                    m+= 1
                    dis1[D] = 0
                else:
                    continue
            for c,dd in edges1[D]:
                if c:
                    que1.append(dd)
                else:
                    both.append(dd)
            edges1[D] = []
            
            if dis2[D] == 0:
                for c,dd in edges2[D]:
                    if c:
                        que2.append(dd)
                 
        while both or que2:
            if both:
                D = both.pop()
                if dis2[D]:
                    m+= 1
                    dis1[D] = dis2[D] = 0
                else:
                    pass
            else:
                D = que2.pop()
                if dis2[D]:
                    m+= 1
                    dis2[D] = 0
                else:
                    pass
            for c,dd in edges2[D]:
                if c:
                    que2.append(dd)
                else:
                    both.append(dd)
            edges2[D] = []    
        print((dis1,dis2))
        if 1 in dis1 or 1 in dis2:
            return -1
        return len(edges) - m

from collections import defaultdict


class Solution:
    def maxNumEdgesToRemove(self, n, edges) -> int:
        e = [defaultdict(lambda: set()) for i in range(3)]
        d = 0
        for edge in edges:
            t, x, y = edge
            x -= 1
            y -= 1
            t -= 1
            if y in e[t][x]:
                d += 1
            else:
                e[t][x].add(y)
                e[t][y].add(x)

        q = 0
        for i in range(n):
            for j in e[2][i]:
                for k in range(2):
                    if j in e[k][i]:
                        q += 1
                        e[k][i].remove(j)
        q = q >> 1

        def visit(v, c, t, o):
            v[c] = o
            for i in e[t][c]:
                if v[i] == -1:
                    visit(v, i, t, o)
            if t != 2:
                for i in e[2][c]:
                    if v[i] == -1:
                        visit(v, i, t, o)

        def solve(t):
            v = [-1] * n
            p = 0
            for i in range(n):
                if v[i] == -1:
                    visit(v, i, t, p)
                    p += 1

            mp = [0] * p
            cp = [0] * p
            for i in range(n):
                mp[v[i]] += len(e[t][i])
                cp[v[i]] += 1

            return p, mp, cp, v

        r = d + q
        p2, mp2, cp2, vp2 = solve(2)
        for i in range(p2):
            r += (mp2[i] >> 1) - (cp2[i] - 1)

        rr = [solve(t) for t in range(2)]
        if rr[0][0] != 1 or rr[1][0] != 1:
            return -1

        tp = p2 + (n - sum(cp2))
        for k in range(2):
            ee = 0
            tt = 0
            for i in range(n):
                for j in e[k][i]:
                    if vp2[i] != vp2[j]:
                        ee += 1
                    else:
                        tt += 1
            ee = ee >> 1
            tt = tt >> 1
            r += ee - (tp - 1) + tt

        return r

class DisjointSet:
    def __init__(self, number_of_sites):
        self.parent = [i for i in range(number_of_sites+1)]
        self.children_site_count = [1 for _ in range(number_of_sites+1)]
        self.component_count = number_of_sites

    def find_root(self, site):
        root = site
        if self.parent[site] != site:
            self.parent[site] = self.find_root(self.parent[site])
        return self.parent[site]

    def is_connected(self, site_1, site_2):
        return self.find_root(site_1) == self.find_root(site_2)

    def union(self, site_1, site_2):
        site_1_root = self.find_root(site_1)
        site_2_root = self.find_root(site_2)
        if site_1_root == site_2_root:
            return False

        if self.children_site_count[site_1_root] < self.children_site_count[site_2_root]:
            self.parent[site_1_root] = site_2_root
            self.children_site_count[site_2_root] += self.children_site_count[
                site_1_root]
        else:
            self.parent[site_2_root] = site_1_root
            self.children_site_count[site_1_root] += self.children_site_count[
                site_2_root]
        self.component_count -= 1
        return True


class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        alice_disjoint_set = DisjointSet(n)
        bob_disjoint_set = DisjointSet(n)

        TYPE_OF_COMMON_EDGES = 3
        TYPE_OF_ALICE_EDGES = 1
        TYPE_OF_BOB_EDGES = 2

        common_edges = filter(lambda edge: edge[0] == TYPE_OF_COMMON_EDGES, edges)
        alice_edges = filter(lambda edge: edge[0] == TYPE_OF_ALICE_EDGES, edges)
        bob_edges = filter(lambda edge: edge[0] == TYPE_OF_BOB_EDGES, edges)

        redundant = 0
        for _, u, v in common_edges:
            unioned_in_alice = alice_disjoint_set.union(u, v)
            unioned_in_bob = bob_disjoint_set.union(u, v)
            if unioned_in_alice and unioned_in_bob:
                continue
            else:
                redundant += 1

        for _, u, v in bob_edges:
            if not bob_disjoint_set.union(u,v):
                redundant += 1
                
        for _, u, v in alice_edges:
            if not alice_disjoint_set.union(u, v):
                redundant += 1
        
        return redundant if alice_disjoint_set.component_count == 1 and bob_disjoint_set.component_count == 1 else -1
class Solution:
    def maxNumEdgesToRemove(self, N: int, E: List[List[int]], same = 0) -> int:
        E = [[_, u - 1, v - 1] for _, u, v in E]                    # u2b50ufe0f -1 for 1-based to 0-based indexing
        A = [i for i in range(N)]                                   # U0001f642 parent representatives of disjoint sets for Alice
        B = [i for i in range(N)]                                   # U0001f642 parent representatives of disjoint sets for Bob
        def find(P, x): P[x] = P[x] if P[x] == x else find(P, P[x]); return P[x]
        def union(P, a, b):
            a = find(P, a)
            b = find(P, b)
            if a == b:
                return 1
            P[a] = b  # arbitrary choice
            return 0
        for type, u, v in E:
            if type == 3: same += union(A, u, v) | union(B, u, v)   # U0001f947 first: U0001f517 union u2705 shared edges between Alice and Bob
        for type, u, v in E:
            if type == 1: same += union(A, u, v)                    # U0001f948 second: U0001f517 union U0001f6ab non-shared edges between Alice and Bob
            if type == 2: same += union(B, u, v)
        # U0001f3af is there a single connected component for Alice and Bob?
        # if so, return the accumulated amount of edges which redundantly connect
        # to each same connected component correspondingly for Alice and Bob
        return same if all(find(A, 0) == find(A, x) for x in A) and all(find(B, 0) == find(B, x) for x in B) else -1

class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        m = len(edges)
        
        g1 = [set() for i in range(n)]
        g2 = [set() for i in range(n)]
        g = [[] for i in range(n)]
        for t, u, v in edges:
            u -= 1
            v -= 1
            if t != 1:
                g2[u].add(v)
                g2[v].add(u)
                
            if t != 2:
                g1[u].add(v)
                g1[v].add(u)
                
            if t == 3:
                g[u].append(v)
                g[v].append(u)
                
        vs = [False] * n
        stk = [0]
        
        while stk:
            idx = stk.pop()
            vs[idx] = True
            
            for ne in g1[idx]:
                if not vs[ne]:
                    stk.append(ne)
        
        for i in range(n):
            if not vs[i]:
                return -1
        
        
        vs = [False] * n
        stk = [0]
        
        while stk:
            idx = stk.pop()
            vs[idx] = True
            
            for ne in g2[idx]:
                if not vs[ne]:
                    stk.append(ne)
        
        
        for i in range(n):
            if not vs[i]:
                return -1
        
        vs = [False] * n
        color = 0
        count = 0
        for i in range(n):
            if not vs[i]:
                color += 1
                tc = 0
                stk = [i]
                while stk:
                    idx = stk.pop()
                    if not vs[idx]:
                        vs[idx] = True
                        tc += 1
                    
                    for ne in g[idx]:
                        if not vs[ne]:
                            stk.append(ne)
                            
                count += tc - 1
                
        print(count)
        print(color)
        
        return m - (count + (color * 2 - 2))
                    

from collections import defaultdict

class Solution:
    def dfs1(self, n, edges):
        neighbours = dict()
        visited = set()
        for i in range(1,n+1):
            neighbours[i] = set()
        for [t, u, v] in edges:
            if t in [1,3]:
                neighbours[u].add(v)
                neighbours[v].add(u)
                
        def dfs(u):
            visited.add(u)
            for v in neighbours[u]:
                if v not in visited:
                    dfs(v)

        dfs(1)
        return len(visited) == n
                
    def dfs2(self, n, edges):
        neighbours = dict()
        visited = set()
        for i in range(1,n+1):
            neighbours[i] = set()
        for [t, u, v] in edges:
            if t in [2,3]:
                neighbours[u].add(v)
                neighbours[v].add(u)
                
        def dfs(u):
            visited.add(u)
            for v in neighbours[u]:
                if v not in visited:
                    dfs(v)
                    
        dfs(1)
        return len(visited) == n

       
    def dfs3(self, n, edges):
        neighbours = dict()
        visited = dict()
        
        for i in range(1,n+1):
            neighbours[i] = set()
        for [t, u, v] in edges:
            if t in [3]:
                neighbours[u].add(v)
                neighbours[v].add(u)
                
        def dfs(daddy, u):
            visited[u] = daddy
            for v in neighbours[u]:
                if v not in visited:
                    dfs(daddy, v)
                    
        for i in range(1, n+1):
            if i not in visited:
                dfs(i,i)
                
        cc_labels = set()
        cc_size = defaultdict(int)
        
        for u in visited:
            cc_labels.add(visited[u])
            cc_size[visited[u]] += 1
        
        num_cc = len(cc_labels)
        
        ret = 2*(num_cc - 1)
        # print("num_cc=", num_cc)
        # print("adding", ret)
        for u in cc_size:
            # print("adding", cc_size[u] - 1)
            ret += (cc_size[u] - 1)
        
        return ret
                
        
        
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        if not (self.dfs1(n, edges) and self.dfs2(n, edges)):
            return -1
        
        return len(edges) - self.dfs3(n, edges)

class UnionFindSet:
    def __init__(self, n):
        self.parents = list(range(n))
        self.ranks = [1] * n
        self.size = 1
        
    def find(self, u):
        if u != self.parents[u]:
            self.parents[u] = self.find(self.parents[u])
        return self.parents[u]
    
    def union(self, u, v):
        pu, pv = self.find(u), self.find(v)
        if pu == pv:
            return False
        if self.ranks[pu] > self.ranks[pv]:
            self.parents[pv] = pu
        elif self.ranks[pv] > self.ranks[pu]:
            self.parents[pu] = pv
        else:
            self.parents[pu] = pv
            self.ranks[pv] += 1
        self.size += 1
        return True
    
    
class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        uf1, uf2 = UnionFindSet(n), UnionFindSet(n)
        ans = 0
        for t, u, v in edges:
            if t != 3:
                continue
            if not uf1.union(u - 1, v - 1) or not uf2.union(u - 1, v - 1):
                ans += 1
        
        for t, u, v in edges:
            if t == 1 and not uf1.union(u - 1, v - 1):
                ans += 1
            elif t == 2 and not uf2.union(u - 1, v - 1):
                ans += 1
                
        if uf1.size != n or uf2.size != n: return -1
        return ans
"""
N connected nodes, M edges (M <= N*(N-1)//2)
what is the minimum number of edges to connect N nodes?



"""



class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.ranks = [1] * n
        self.size = 1
        
    def find(self, u):
        if u == self.parent[u]:
            return self.parent[u]
        return self.find(self.parent[u])
    
    def union(self, i, j):
        x = self.find(i)
        y = self.find(j)
        if x == y:
            return False
        if self.ranks[x] > self.ranks[y]:
            self.parent[y] = x
        elif self.ranks[y] > self.ranks[x]:
            self.parent[x] = y
        else:
            self.parent[x] = y
            self.ranks[y] += 1
        self.size += 1
        return True
    
    
class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        uf1 = UnionFind(n)
        uf2 = UnionFind(n)
        res = 0
        
        for t, u, v in edges:
            if t != 3:
                continue
            if not uf1.union(u - 1, v - 1) or not uf2.union(u - 1, v - 1):
                res += 1
        
        for t, u, v in edges:
            if t == 1 and not uf1.union(u - 1, v - 1):
                res += 1
            elif t == 2 and not uf2.union(u - 1, v - 1):
                res += 1
   
        return res if uf1.size == n and uf2.size == n else -1




        
        
        
        
        
        
class UnionFind(object):
    def __init__(self, n, recursion = False):
        self._par = list(range(n))
        self._size = [1] * n
        self._recursion = recursion

    def root(self, k):
        if self._recursion:
            if k == self._par[k]:
                return k
            self._par[k] = self.root(self._par[k])
            return self._par[k]
        else:
            root = k
            while root != self._par[root]: root = self._par[root]
            while k != root: k, self._par[k] = self._par[k], root
            return root

    def unite(self, i, j):
        i, j = self.root(i), self.root(j)
        if i == j: return False
        if self._size[i] < self._size[j]: i, j = j, i
        self._par[j] = i
        self._size[i] += self._size[j]
        return True

    def is_connected(self, i, j):
        return self.root(i) == self.root(j)

    def size(self, k):
        return self._size[self.root(k)]

class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        uf1 = UnionFind(n)
        uf2 = UnionFind(n)
        ans = 0
        
        for i, u, v in edges:
            if i == 3:
                ans += 1 - uf1.unite(u - 1, v - 1)
                uf2.unite(u - 1, v - 1)
        
        for i, u, v in edges:
            if i == 1:
                ans += 1 - uf1.unite(u - 1, v - 1)
            elif i == 2:
                ans += 1 - uf2.unite(u - 1, v - 1)
        
        if uf1.size(0) == uf2.size(0) == n:
            return ans
        else:
            return -1
class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        # Union find
#         def find(i):
#             if i != root[i]:
#                 root[i] = find(root[i])
#             return root[i]

#         def uni(x, y):
#             x, y = find(x), find(y)
#             if x == y: return 0
#             root[x] = y
#             return 1

#         res = e1 = e2 = 0

#         # Alice and Bob
#         root = list(range(n + 1))
#         for t, i, j in edges:
#             if t == 3:
#                 if uni(i, j):
#                     e1 += 1
#                     e2 += 1
#                 else:
#                     res += 1
#         root0 = root[:]

#         # only Alice
#         for t, i, j in edges:
#             if t == 1:
#                 if uni(i, j):
#                     e1 += 1
#                 else:
#                     res += 1

#         # only Bob
#         root = root0
#         for t, i, j in edges:
#             if t == 2:
#                 if uni(i, j):
#                     e2 += 1
#                 else:
#                     res += 1

#         return res if e1 == e2 == n - 1 else -1
        
        def findRoot(a):
            if a != root[a]:
                root[a] = findRoot(root[a])
            return root[a]
        
        def union(a,b):
            aRoot,bRoot = findRoot(a),findRoot(b)
            if aRoot != bRoot:
                root[aRoot] = bRoot
                return True
            return False
        
        root = [i for i in range(n+1)]
        numofEdgesAli,numofEdgesBob,ans = 0,0,0
        for t,a,b in edges:
            if t == 3:
                if union(a,b): 
                    numofEdgesAli += 1
                    numofEdgesBob += 1
                else: 
                    ans += 1
        root0 = root[:]
        for t,a,b in edges:
            if t == 1:
                if union(a,b): 
                    numofEdgesAli += 1
                else: 
                    ans += 1
        root = root0
        for t,a,b in edges:
            if t == 2:
                if union(a,b):
                    numofEdgesBob += 1
                else: 
                    ans += 1
        return ans if numofEdgesAli == numofEdgesBob == n-1 else -1
                
                

class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        ufAli = uf(n)
        ufBob = uf(n)
        
#         10 -> 2
        for edg in edges:
            x, y = edg[1], edg[2]
            if edg[0] == 1:
                ufAli.addEdge(x, y, 2)
            elif edg[0] == 2:
                ufBob.addEdge(x, y, 2)
            else:
                ufAli.addEdge(x, y, 1)
                ufBob.addEdge(x, y, 1)
                
        # print(ufAli.g, ufAli.kruskalmst())
        # print(ufBob.g, ufBob.kruskalmst())
        
        blueremoved = set()
        aliremoved = set()
        bobremoved = set()
        
        ans1 = ufAli.kruskalmst(blueremoved, aliremoved)
        ans2 = ufBob.kruskalmst(blueremoved, bobremoved)
        if ans1 == -1 or ans2 == -1:
            return -1
        
        # return ans1 + ans2
        return len(blueremoved) + len(aliremoved) + len(bobremoved)
        
        
                
        
        
        

class uf:
    def __init__(self, n):
        self.n = n
        self.g = []
        self.joinednodes = set()
        # self.totalnodes = set()
        
        
    def addEdge(self, x, y, cost):
        self.g.append((x, y, cost))
        # self.joinednodes 
        
    def find(self, x, parent):
        if parent[x] == x:
            return x
        
        return self.find(parent[x], parent)
    
    def union(self, x, y, parent, rank):
        xroot, yroot = self.find(x, parent), self.find(y, parent)
        
        if xroot != yroot:
            if rank[xroot] > rank[yroot]:
                parent[yroot] = xroot
            elif rank[yroot] > rank[xroot]:
                parent[xroot] = yroot
            else:
                parent[yroot] = xroot
                rank[xroot] += 1
                
    def kruskalmst(self, blue, rorg):
        # parent = { for edge in g}
        parent = {}
        rank = {}
        for edge in self.g:
            parent[edge[0]] = edge[0]
            parent[edge[1]] = edge[1]
            rank[edge[0]] = 0
            rank[edge[1]] = 0
            
        # print(parent, rank)
        success = 0
        self.g.sort(key=lambda edge: edge[2])
        for edge in self.g:
            x, y, cos = edge
            xroot = self.find(x, parent)
            yroot = self.find(y, parent)
            if xroot != yroot:
                success += 1
                self.union(xroot, yroot, parent, rank)
                
            else:
                if cos == 1:
                    blue.add((x,y))
                else:
                    rorg.add((x,y))
                
                
        
                
        if success == self.n -1:
            
            # return success
            return len(self.g) - success
        
        return -1
            
            
            
            
            
        
                
                
        
        

class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        class DSU:
            def __init__(self, n: int):
                self.p = list(range(n))
                self.e = 0

            def find(self, x: int) -> int:
                if x != self.p[x]: self.p[x] = self.find(self.p[x])
                return self.p[x]

            def merge(self, x: int, y: int) -> int:
                rx, ry = self.find(x), self.find(y)
                if rx == ry: return 1
                self.p[rx] = ry
                self.e += 1
                return 0
        A, B = DSU(n + 1), DSU(n + 1)    
        ans = 0
        for t, x, y in edges:
            if t != 3: continue
            ans += A.merge(x, y)
            B.merge(x, y)

        for t, x, y in edges:
            if t == 3: continue
            d = A if t == 1 else B
            ans += d.merge(x, y)
        return ans if A.e == B.e == n - 1 else -1
class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        v2edges={}
        red_edge_count=0
        blue_edge_count=0
        for typ,u,v in edges:
            v2edges.setdefault(v,[]).append((typ,u))
            v2edges.setdefault(u,[]).append((typ,v))
            if typ==1: red_edge_count+=1
            if typ==2: blue_edge_count+=1
        flagSet=set()
        def explore(allowedTypeSet,startVertex,flagEnabled=False):
            q=deque()
            visited=set()
            q.append(startVertex)
            visited.add(startVertex)
            if flagEnabled: flagSet.add(startVertex)
            while q:
                cur=q.popleft()
                for typ,dst in v2edges[cur]:
                    if typ in allowedTypeSet and dst not in visited:
                        q.append(dst)
                        visited.add(dst)
                        if flagEnabled: flagSet.add(dst)
            return visited
        
        if len(explore(set([1,3]),1))!=n: return -1
        if len(explore(set([2,3]),1))!=n: return -1
        
        cn=0
        cmpn=0
        v2gc={}
        gc_node_count=[]
        for v in range(1,n+1):
            if v in flagSet: continue
            cmpn+=1
            visited=explore(set([3]),v,True)
            if len(visited)>=2:
                for u in visited:
                    v2gc[u]=cn
                gc_node_count.append(len(visited))
                cn+=1
        
        green_edge_count=[0]*cn
        for typ,u,v in edges:
            if typ==3:
                ci=v2gc[u]
                green_edge_count[ci]+=1
        ans_red=max(0,red_edge_count-(cmpn-1))
        ans_blue=max(0,blue_edge_count-(cmpn-1))
        ans_green=sum(max(0,green_edge_count[ci]-(gc_node_count[ci]-1)) for ci in range(cn))
        ans=ans_red+ans_blue+ans_green
        return ans
class DS:
    def __init__(self, n):
        self.par = list(range(n))
        
    def find(self, x):
        if self.par[x] != x:
            self.par[x] = self.find(self.par[x])
        return self.par[x]
    
    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        self.par[px] = py

class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        ga = defaultdict(set)
        gb = defaultdict(set)
        gc = defaultdict(set)
        count_a = 0
        count_b = 0
        count_c = 0
        for t,u,v in edges:
            u = u-1
            v = v-1
            if t == 1:
                ga[u].add(v)
                ga[v].add(u)
                count_a += 1
            if t == 2:
                gb[u].add(v)
                gb[v].add(u)
                count_b += 1
            if t == 3:
                gc[u].add(v)
                gc[v].add(u)
                count_c += 1
                
        ans = 0
        
        ds = DS(n)
        for u in gc:
            for v in gc[u]:
                ds.union(u,v)
        counter = Counter(ds.find(i) for i in range(n))
        edge_num_type_3 = sum(val - 1 for val in list(counter.values()))
        ans += count_c - edge_num_type_3
        
        dsa = copy.deepcopy(ds)
        for u in ga:
            for v in ga[u]:
                dsa.union(u,v)
        if len(set(dsa.find(i) for i in range(n))) > 1:
            return -1
        ans += count_a - (n - 1 - edge_num_type_3)
        
        dsb = copy.deepcopy(ds)
        for u in gb:
            for v in gb[u]:
                dsb.union(u,v)
        if len(set(dsb.find(i) for i in range(n))) > 1:
            return -1
        ans += count_b - (n - 1 - edge_num_type_3)
        
        return ans
        
        

class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        graph = [collections.defaultdict(list), collections.defaultdict(list),collections.defaultdict(list)]
        temp = [[], [], []]
        for t, a, b in edges:
            temp[t - 1].append([a, b])

            if t in (1, 3):
                graph[0][a].append(b)
                graph[0][b].append(a)
            if t in (2, 3):
                graph[1][a].append(b)
                graph[1][b].append(a)

        def helper(i):
            que = [1]
            seen = set(que)
            for node in que:
                #print(i, que)
                for y in graph[i][node]:
                    if y not in seen:
                        seen.add(y)
                        que.append(y)

            return len(seen) == n
        if not helper(0) or not helper(1):
            return -1
                
        p = list(range(n + 1))
        def find(i):
            if p[i] != i:
                p[i] = find(p[i])
            return p[i]
        def union(i, j):
            p[find(i)] = find(j)
        
        def helper(c):

            ans = 0
            for x, y in c:
                if find(x) == find(y):
                    ans += 1
                else:
                    union(x, y)
            return ans

        res = helper(temp[2])
        old = p[:]
        for c in temp[:2]:
            res += helper(c)
            p = old
        return res
        
        



class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        uf1 = list(range(n+1))
        uf2 = list(range(n+1))
        def find(x, uf):
            if x != uf[x]:
                uf[x] = find(uf[x], uf)
            return uf[x]
        def union(x, y, uf):
            uf[find(x, uf)] = find(y, uf)
        edge_total = 0
        edge_1 = 0
        edge_2 = 0
       # make 1 first
        for w, v, u in sorted(edges, reverse=True):
            if w == 2:
                continue
            if find(v, uf1) == find(u, uf1):
                continue
            union(v, u, uf1)
            edge_1 += 1
            edge_total += 1
            if w == 3:
                union(v, u, uf2)
                edge_2 += 1
        if edge_1 < n-1:
            return -1
        # make 2 next
        for w, v, u in edges:
            if w == 1:
                continue
            if find(v, uf2) == find(u, uf2):
                continue
            union(v, u, uf2)
            edge_2 += 1
            if w == 2:
                edge_total += 1
        if edge_2 < n-1:
            return -1
        return len(edges) - edge_total
        

class DSU:
    def __init__(self, n):
        self.parent = [i for i in range(n+1)]
        self.sz = [1]*n
    
    def find(self, x):
        if x != self.parent[x]:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        xr = self.find(x)
        yr = self.find(y)
        if xr != yr:
            if self.sz[xr] < self.sz[yr]:
                xr, yr = yr, xr
            self.parent[yr] = xr
            self.sz[xr] += self.sz[yr]
            return True
        return False
    
    def size(self, x):
        return self.sz[self.find(x)]

class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        graphA, graphB, graphC = [], [], []
        for t, u, v in edges:
            if t == 1:
                graphA.append([u-1, v-1])
            elif t == 2:
                graphB.append([u-1, v-1])
            else:
                graphC.append([u-1, v-1])
        
        alice = DSU(n)
        bob = DSU(n)
        result = 0
        
        for u, v in graphC:
            alice.union(u, v)
            if not bob.union(u, v):
                result += 1
        
        for u, v in graphA:
            if not alice.union(u, v):
                result += 1
        
        for u, v in graphB:
            if not bob.union(u, v):
                result += 1
        
        if alice.size(0) < n or bob.size(0) < n:
            return -1
        
        return result
class UF:
    def __init__(self, n):
        self.p = [i for i in range(n)]
        
    def find(self, x):
        if self.p[x] != x:
            self.p[x] = self.find(self.p[x])
        return self.p[x]
    
    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py:
            return False
        self.p[py] = px
        return True

class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        A, B = set(), set()
        rmA, rmB = 0, 0
        for t, u, v in edges:
            if t == 1:
                if (-3, u, v) in A:
                    rmA += 1
                else:
                    A.add((-1, u, v))
            elif t == 2:
                if (-3, u, v) in B:
                    rmB += 1
                else:
                    B.add((-2, u, v))
            else:
                if (-1, u, v) in A:
                    rmA += 1
                    A.remove((-1, u, v))
                if (-2, u, v) in B:
                    rmB += 1  
                    B.remove((-2, u, v))
                A.add((-3, u, v))
                B.add((-3, u, v))
        # print(rmA, rmB, A, B)
        common = set()
        ufa = UF(n + 1)
        for t, u, v in sorted(A):
            if not ufa.union(u, v):
                if t == -1:
                    rmA += 1
                else:
                    common.add((u, v))
                    
        for i in range(1, n + 1):
            if ufa.find(i) != ufa.find(1):
                return -1
        
        ufb = UF(n + 1)
        for t, u, v in sorted(B):
            if not ufb.union(u, v):
                if t == -2:
                    rmB += 1
                else:
                    common.add((u, v))
                    
        for i in range(1, n + 1):
            if ufb.find(i) != ufb.find(1):
                return -1
        
        return rmA + rmB + len(common)

class UF:
    def __init__(self, N):
        self.N = N
        self.parents = list(range(N))
        
    def find(self, x):
        if self.parents[x] != x:
            self.parents[x] = self.find(self.parents[x])
        return self.parents[x]
    def union(self, x, y):
        px = self.find(x)
        py = self.find(y)
        
        if px != py:
            self.parents[py] = px
            self.N -= 1
            return True
        return False
            
class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        edges.sort(key=lambda x: -x[0])
        
        used = set()
        uf1 = UF(n)
        for t, x, y in edges:
            if t == 3:
                if uf1.union(x-1, y-1):
                    used.add((t, x, y))
        for t, x, y in edges:
            if t == 1:
                if uf1.union(x-1, y-1):
                    used.add((t, x, y))
        uf2 = UF(n)
        for t, x, y in edges:
            if t == 3:
                if uf2.union(x-1, y-1):
                    used.add((t, x, y))
        for t, x, y in edges:
            if t == 2:
                if uf2.union(x-1, y-1):
                    used.add((t, x, y))
        if uf1.N > 1 or uf2.N > 1:
            return -1
        return len(edges)-len(used)

class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        # build graph, use type 3 first
        # then do alice and bob separately
        # have dsu parents to build up alice and bob
        
        parentsA = list(range(n))
        parentsB = list(range(n))
        
        def find(parents, a):
            while a != parents[a]:
                parents[a] = parents[parents[a]]
                a = parents[a]
            return a
            
        def union(parents, a, b):
            a = find(parents, a)
            b = find(parents, b)
            parents[a] = b
        
        type3 = []
        typeB = []
        typeA = []
        
        for t, u, v in edges:
            u, v = u-1, v-1 # make zero indexed, easier for UF
            if t == 3:
                type3.append((u, v))
            elif t == 2:
                typeB.append((u, v))
            elif t == 1:
                typeA.append((u, v))
                
        # now add type3 edges if they join together two new things
        
        add = 0
        for u, v in type3:
            if find(parentsA, u) != find(parentsA, v):
                add += 1
                union(parentsA, u, v)
                union(parentsB, u, v)
                
        # now do type1 and 2 separately
        for u,v in typeA:
            if find(parentsA, u) != find(parentsA, v):
                add += 1
                union(parentsA, u, v)
        

    
        for u,v in typeB:
            if find(parentsB, u) != find(parentsB, v):
                add += 1
                union(parentsB, u, v)
                
                
        # print(len(type3), len(typeA), len(typeB))
        # return
                
                
        # print(add, type3, typeA, typeB)
        
        uniqA, uniqB = set(), set()
        for i in range(n):
            # print(i, find(parentsA, i), find(parentsB, i))
            uniqA.add(find(parentsA, i))
            uniqB.add(find(parentsB, i))
            
            if len(uniqA) > 1 or len(uniqB) > 1:
                return -1
            
            
        return len(edges) - add
class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        self.father_alice = [i for i in range(n + 1)]
        self.father_bob = [i for i in range(n + 1)]
        res = 0
        edge_alice, edge_bob = 0, 0
        for type, u, v in edges:
            if type == 3:
                if self.connect(u, v, True) == 1:
                    edge_alice += 1
                    edge_bob += 1
                else:
                    res += 1
                
                self.connect(u, v, False)
        
        for type, u, v in edges:
            if type == 1:
                if self.connect(u, v, True) == 1:
                    edge_alice += 1
                else:
                    res += 1
            elif type == 2:
                if self.connect(u, v, False) == 1:
                    edge_bob += 1
                else:
                    res += 1
        
        if edge_alice == edge_bob == n - 1:
            return res
        return -1
    
    
    
    def find(self, x, is_alice):
        if is_alice:
            if self.father_alice[x] == x:
                return self.father_alice[x]
            self.father_alice[x] = self.find(self.father_alice[x], True)
            return self.father_alice[x]
        else:
            if self.father_bob[x] == x:
                return self.father_bob[x]
            self.father_bob[x] = self.find(self.father_bob[x], False)
            return self.father_bob[x]
    
    def connect(self, a, b, is_alice):
        if is_alice:
            root_a = self.find(a, True)
            root_b = self.find(b, True)
            if root_a == root_b:
                return 0
            else:
                self.father_alice[max(root_a, root_b)] = min(root_a, root_b)
                return 1
        else:
            root_a = self.find(a, False)
            root_b = self.find(b, False)
            if root_a == root_b:
                return 0
            else:
                self.father_bob[max(root_a, root_b)] = min(root_a, root_b)
                return 1
        
        
        
        
        
#         self.father_alice = [i for i in range(n + 1)]
#         self.father_bob = [i for i in range(n + 1)]
        
#         res = 0
#         for type, u, v in edges:
#             if type == 3:
#                 res += self.connect(u, v, True)
#                 self.connect(u, v, False)
        
#         for type, u, v in edges:
#             if type == 1:
#                 res += self.connect(u, v, True)
#             elif type == 2:
#                 res += self.connect(u, v, False)
        
        
#         if self.check_valid(True) and self.check_valid(False):
#             return res
#         return -1
    
    
#     def find(self, x, is_alice):
#         if is_alice:
#             if self.father_alice[x] == x:
#                 return self.father_alice[x]
#             self.father_alice[x] = self.find(self.father_alice[x], True)
#             return self.father_alice[x]
        
#         else:
#             if self.father_bob[x] == x:
#                 return self.father_bob[x]
#             self.father_bob[x] = self.find(self.father_bob[x], False)
#             return self.father_bob[x]
        
#     def connect(self, a, b, is_alice):
#         if is_alice:
#             root_a = self.find(a, True)
#             root_b = self.find(b, True)
#             if root_a != root_b:
#                 self.father_alice[max(root_a, root_b)] = min(root_a, root_b)
#                 return 0
#             return 1
        
#         else:
#             root_a = self.find(a, False)
#             root_b = self.find(b, False)
#             if root_a != root_b:
#                 self.father_bob[max(root_a, root_b)] = min(root_a, root_b)
#                 return 0
#             return 1
        
#     def check_valid(self, is_alice):
#         if is_alice:
#             root = self.find(1, True)
#             for i in range(1, len(self.father_alice)):
#                 if self.find(i, True) != root:
#                     return False
#             return True
        
#         else:
#             root = self.find(1, False)
#             for i in range(1, len(self.father_bob)):
#                 if self.find(i, False) != root:
#                     return False
#             return True

class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        f = {}
        t = {}
        ans = 0
        def ffind(a):
            f.setdefault(a,a)
            if f[a] != a:
                f[a] = ffind(f[a])
            return f[a]
        def funion(a,b):
            f.setdefault(a,a)
            f.setdefault(b,b)
            if ffind(a) == ffind(b): return False
            f[ffind(a)] = f[ffind(b)]
            return True
        def tfind(a):
            t.setdefault(a,a)
            if t[a] != a:
                t[a] = tfind(t[a])
            return t[a]
        def tunion(a,b):
            t.setdefault(a,a)
            t.setdefault(b,b)
            if tfind(a) == tfind(b): return False
            t[tfind(a)] = t[tfind(b)]
            return True
        
        for ty, a, b in edges:
            if ty!= 3: continue
            tunion(a,b)
            if not funion(a,b):
                ans += 1
        for ty, a, b in edges:
            if ty!=1: continue
            if not funion(a,b):
                ans += 1
        for ty, a, b in edges:
            if ty!=2: continue
            if not tunion(a,b):
                ans += 1
        if len(f) != n or len(t) != n: return -1
        return ans
        
        

class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        nei=[defaultdict(set),defaultdict(set),defaultdict(set)]
        count=[0,0,0]
        for i,j,k in edges:
            nei[i-1][j-1].add(k-1)
            nei[i-1][k-1].add(j-1)
            count[i-1]+=1
        def dfs(root,mark,seen,graph):
            seen[root]=mark
            for i in graph[root]:
                if not seen[i]:
                    dfs(i,mark,seen,graph)
        def cc(n,graph):
            cur=1
            seen=[0]*n
            for i in range(n):
                if not seen[i]:
                    dfs(i,cur,seen,graph)
                    cur+=1
            return (cur-1,seen)
        comp,group=cc(n,nei[2])
        res=count[2]-(n-comp)
        if comp==1:
            return res+count[0]+count[1]
        for i,j in list(nei[2].items()):
            nei[0][i]|=j
            nei[1][i]|=j
        comp1,group1=cc(n,nei[0])
        comp2,group2=cc(n,nei[1])
        if comp1!=1 or comp2!=1:
            return -1
        needed=(comp-1)*2
        return res+count[0]+count[1]-needed
        

class UnionFindSet:
    def __init__(self, n):
        self.parents = list(range(n))
        self.ranks = [1] * n
        self.count = 1
        
    def find(self, u):
        if u != self.parents[u]:
            self.parents[u] = self.find(self.parents[u])
        return self.parents[u]
    
    def union(self, u, v):
        pu, pv = self.find(u), self.find(v)
        if pu == pv:
            return False
        if self.ranks[pu] > self.ranks[pv]:
            self.parents[pv] = pu
        elif self.ranks[pv] > self.ranks[pu]:
            self.parents[pu] = pv
        else:
            self.parents[pu] = pv
            self.ranks[pv] += 1
        self.count += 1
        return True
    
class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        uf1, uf2 = UnionFindSet(n), UnionFindSet(n)
        ans = 0
        for t, u, v in edges:
            u -= 1
            v -= 1
            if t != 3:
                continue
            flag = 0
            if not uf1.union(u, v):
                flag = 1
            if not uf2.union(u, v):
                flag = 1
            ans += flag
        
        for t, u, v in edges:
            u -= 1
            v -= 1
            if t == 1:
                if not uf1.union(u, v):
                    ans += 1
            elif t == 2:
                if not uf2.union(u, v):
                    ans += 1
        if uf1.count != n or uf2.count != n: return -1
        return ans
import copy


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n + 1))
        
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py:
            return 0
        
        self.parent[px] = py
        return 1


class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        uf, res, e1, e2 = UnionFind(n), 0, 0, 0
        for _, u, v in [x for x in edges if x[0] == 3]:
            add = uf.union(u, v)
            if add:
                e1 += 1
                e2 += 1
            else:
                res += 1
                
        uf1 = copy.deepcopy(uf)
        for _, u, v in [x for x in edges if x[0] == 1]:
            add = uf1.union(u, v)
            if add:
                e1 += 1
            else:
                res += 1        
                
        uf2 = copy.deepcopy(uf)
        for _, u, v in [x for x in edges if x[0] == 2]:
            add = uf2.union(u, v)
            if add:
                e2 += 1
            else:
                res += 1                    
        
        return res if e1 == n - 1 and e2 == n - 1 else -1
class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        d = [{}, {}]
        
        def find(x, n):
            if x not in d[n]:
                d[n][x] = x
                return d[n][x]
            else:
                if d[n][x] != x:
                    d[n][x] = find(d[n][x], n)
                return d[n][x]
        
        def union(x, y, n):
            d[n][find(x, n)] = find(y, n)
            
        ans = 0
        edges.sort(reverse = True)
        for typeN, a, b in edges:
            if typeN == 3:
                if find(a, 0) == find(b, 0) and find(a, 1) == find(b, 1):
                    ans += 1
                else:
                    union(a, b, 0)
                    union(a, b, 1)
            else:
                if find(a, typeN-1) == find(b, typeN-1):
                    ans += 1
                else:
                    union(a, b, typeN-1)                    
        return -1 if any((find(1, 0) != find(i, 0) or find(1, 1) != find(i, 1)) for i in range(2, n+1)) else ans
                    
                    
                    
                    

class Solution:
    def maxNumEdgesToRemove(self, n: int, edges) -> int:
        N = len(edges)
        dup = set()
        res = 0
        c1, c2, bc = 0, 0, 0
        alice, bob, both = defaultdict(list), defaultdict(list), defaultdict(list)
        
        for t, u, v in edges:
            if (t, u, v) not in dup:
                dup.add((t, u, v))
                if t == 1 or t == 3:
                    if t == 1:
                        c1 += 1
                    alice[u].append(v)
                    alice[v].append(u)
                if t == 2 or t == 3:
                    if t == 2:
                        c2 += 1
                    bob[u].append(v)
                    bob[v].append(u)
                if t == 3:
                    bc += 1
                    both[u].append(v)
                    both[v].append(u)
            else:
                res += 1
        # print(res)
        
        va, vb, = set(), set()
        vc = dict()
        
        def dfs(node, t):
            if t == 1:
                va.add(node)
                for ngb in alice[node]:
                    if not ngb in va:
                        dfs(ngb, t)
            else:
                vb.add(node)
                for ngb in bob[node]:
                    if not ngb in vb:
                        dfs(ngb, t)
        
        dfs(1, 1)
        dfs(1, 2)
        
        if len(va) < n or len(vb) < n:
            return -1
        
        def dfs_both(node, prev, idx):
            vc[node] = idx
            for ngb in both[node]:
                if ngb == prev:
                    continue
                if ngb not in vc:
                    dfs_both(ngb, node, idx)
                
        idx = 0
        for i in both:
            if i not in vc:
                idx += 1
                dfs_both(i, -1, idx)
            
        bc_need = 0
        for i in range(1, idx + 1):
            cluster = 0
            for node in vc:
                if vc[node] == i:
                    cluster += 1
            bc_need += cluster - 1
                
        res += bc - bc_need
        # print(bc)
        # print(c1)
        # print(c2)
        # print(res)
        # print(bc_need)
        
        res += c1 - (n - 1 - bc_need)
        res += c2 - (n - 1 - bc_need)
        return res
class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        t = [[] for _ in range(3)]
        for T, x, y in edges:
            t[T-1].append([x-1, y-1])
        
        def root(cc, i):
            if cc[i] != i:
                cc[i] = root(cc, cc[i])
            return cc[i]
        
        def join(cc, i, j):
            ri = root(cc, i)
            rj = root(cc, j)
            if ri != rj:
                cc[ri] = rj
            return ri != rj
        
        ret = 0
        
        cc = [i for i in range(n)]
        cct = n
        for x,y in t[2]:
            if join(cc, x,y):
                cct -= 1
                ret += 1
                if cct == 1:
                    break
        
        ac = cc[:]
        bc = cc[:]
        acct = bcct = cct
        
        for x,y in t[0]:
            if join(ac, x,y):
                acct -= 1
                ret += 1
                if acct == 1:
                    break
        if acct != 1:
            return -1
        
        for x,y in t[1]:
            if join(bc, x,y):
                bcct -= 1
                ret += 1
                if bcct == 1:
                    break
        if bcct != 1:
            return -1
        
        return len(edges) - ret
        

class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        type = [set(), set(), set()]
        for t, u, v in edges:
            type[t-1].add(tuple(sorted([u,v])))
        # print(type)
        res = 0
        type3type1 = type[2] & type[0]
        type3type2 = type[2] & type[1]
        # print(type3type1, type3type2)
        res += len(type3type1) + len(type3type2)
        type[0] -= type3type1
        type[1] -= type3type2
        # print(type)
        type3 = {i:i for i in range(1, n+1)}
        def uf(parent, i):
            if parent[i] == i: return i
            parent[parent[i]] = uf(parent, parent[i])
            return parent[parent[i]]
            # return uf(parent, parent[i])
        moved = set()
        for u, v in type[2]:
            pu = uf(type3, u)
            pv = uf(type3, v)
            if pu != pv: type3[pu] = pv
            else: moved.add((u,v))
        res += len(moved)
        type[2] -= moved
        # print(moved, type)
        type2 = {i:i for i in range(1, n+1)}
        for u,v in type[1] | type[2]:
            pu, pv = uf(type2, u), uf(type2, v)
            if pu != pv: type2[pu] = pv
            else:
                res += 1
        # print(type2)
        cnt = 0
        for i in range(1, n+1):
            pi = uf(type2, i)
            if pi == i: cnt += 1
        if cnt > 1: return -1
        # print(type2, cnt)
        type1 = {i:i for i in range(1, n+1)}
        for u, v in type[0] | type[2]:
            # print(u,v)
            pu, pv = uf(type1, u), uf(type1, v)
            if pu != pv: type1[pu] = pv
            else: res += 1
        cnt = 0
        for i in range(1,n+1):
            pi = uf(type1, i)
            if pi == i: cnt += 1
        # print(type1, cnt)
        if cnt > 1: return -1
        return res
class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        def find(parent,node):
            if node != parent[node]:
                parent[node] = find(parent,parent[node])
            return parent[node]
        def union(parent,node1,node2):
            parent1 = find(parent,node1)
            parent2 = find(parent,node2)
            parent[parent2] = parent1
            return
        result = 0
        parent1 = [i for i in range(n)]
        parent2 = [i for i in range(n)]
        for edge in edges:
            if edge[0]==3:
                if find(parent1,edge[1]-1) == find(parent1,edge[2]-1):
                    result += 1
                else:
                    union(parent1,edge[1]-1,edge[2]-1)
                    union(parent2,edge[1]-1,edge[2]-1)
        for edge in edges:
            if edge[0]==1:
                if find(parent1,edge[1]-1) == find(parent1,edge[2]-1):
                    result += 1
                else:
                    union(parent1,edge[1]-1,edge[2]-1)
            if edge[0]==2:
                if find(parent2,edge[1]-1) == find(parent2,edge[2]-1):
                    result += 1
                else:
                    union(parent2,edge[1]-1,edge[2]-1)
        root1 = find(parent1,0)
        root2 = find(parent2,0)
        for node in range(n):
            if find(parent1,node) != root1 or find(parent2,node) != root2:
                return -1
        return result
class UnionFind:
    def __init__(self, n):
        self.parent = [i for i in range(n + 1)]
        self._count = n
    
    def find(self, a):
        pa = self.parent[a]
        if pa != a:
            self.parent[a] = self.find(self.parent[a])
        return self.parent[a]
    
    def connect(self, a, b):
        pa = self.find(a)
        pb = self.find(b)
        if pa == pb:
            return False
        self.parent[pa] = pb
        self._count -= 1
        return True
    
    def counts(self):
        return self._count

class Solution:
    def maxNumEdgesToRemove(self, n, edges):
        edges.sort(reverse=True)
        alice_uf = UnionFind(n)
        bob_uf = UnionFind(n)
        added_edges = 0
        for t, s, e in edges:
            if t == 3:
                alice_connected = alice_uf.connect(s, e)
                bob_connected = bob_uf.connect(s, e)
                if alice_connected or bob_connected:
                    added_edges += 1
            if t == 2:
                if bob_uf.connect(s, e):
                    added_edges += 1
            if t == 1:
                if alice_uf.connect(s, e):
                    added_edges += 1
        
        if alice_uf.counts() == bob_uf.counts() == 1:
            return len(edges) - added_edges
        return -1
class Solution:
    def dfs(self, i, a, b, types):
        b[i] = 1
        for k in a[i]:
            if k[0] in types and b[k[1]] == 0:
                self.dfs(k[1], a, b, types)

    def check(self, a, types) -> int:
        n = len(a)
        b = [0] * n
        ans = 0
        for i in range(n):
            if b[i] == 0:
                self.dfs(i, a, b, types)
                ans += 1
        return ans
        
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        a = []
        for i in range(n):
            a.append([])
        p = 0
        for edge in edges:
            u = edge[1] - 1
            v = edge[2] - 1
            a[u].append([edge[0], v])
            a[v].append([edge[0], u])
            if edge[0] == 3:
                p += 1
        if self.check(a, [1, 3]) > 1 or self.check(a, [2, 3]) > 1:
            return -1
        edge3 = n - self.check(a, [3])
        return len(edges) - (n - 1) * 2 + edge3

class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        mm = [collections.defaultdict(set), collections.defaultdict(set), collections.defaultdict(set)]
        for t, u, v in edges:
            mm[t-1][u].add(v)
            mm[t - 1][v].add(u)

        n_edges = 0

        super_nodes = {}
        super_nodes_list = []
        n_super_nodes = 0
        def dfs(type):
            visited = set()
            nonlocal n_edges
            def search(node):
                if node in visited:
                    return
                visited.add(node)
                neis = mm[type][node].union(mm[2][node])
                for node2 in neis:
                    if node2 not in visited:
                        search(node2)

            search(1)
            if len(visited) != n:
                n_edges += float('inf')

        def create_super_nodes():
            total_visited = set()
            cur_visited = set()
            nonlocal n_super_nodes
            def search(node):
                nonlocal n_edges

                if node in cur_visited:
                    return
                cur_visited.add(node)
                for node2 in mm[2][node]:
                    if node2 not in cur_visited:
                        n_edges += 1
                        search(node2)

            for node in range(1, n+1):
                cur_visited = set()
                if node not in total_visited:
                    search(node)
                    # super_nodes_list.append(cur_visited)
                    total_visited.update(cur_visited)
                    n_super_nodes += 1
                    # for node2 in cur_visited:
                    #     super_nodes[node2] = cur_visited

        create_super_nodes()
        dfs(0)
        dfs(1)
        n_edges += (n_super_nodes - 1) * 2
        sol = len(edges) - (n_edges)
        if sol == float('-inf'):
            return -1
        return sol


class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        def find(x):
            if fa[x] == x:
                return x
            fa[x] = find(fa[x])
            return fa[x]
        def merge(x, y):
            fx = find(x)
            fy = find(y)
            if fx != fy:
                fa[fx] = fy
                cnt[fy] = cnt[fx] + cnt[fy]
        fa = [i for i in range(n+1)]        
        cnt = [1 for i in range(n+1)]
        ans = 0
        for edge in edges:
            if edge[0] == 3:
                fx, fy = find(edge[1]), find(edge[2])
                if fx != fy:
                    merge(fx, fy)
                    ans += 1
        fa_copy = [x for x in fa]            
        for edge in edges:
            if edge[0] == 1:
                fx, fy = find(edge[1]), find(edge[2])
                if fx != fy:
                    merge(fx, fy)
                    ans += 1
        f0 = find(1)            
        for i in range(1, n+1):
            if find(i) != f0:
                return -1
            
        fa = [x for x in fa_copy]    
            
        for edge in edges:
            if edge[0] == 2:
                fx, fy = find(edge[1]), find(edge[2])
                if fx != fy:
                    merge(fx, fy)
                    ans += 1
        f0 = find(1)            
        for i in range(1, n+1):
            if find(i) != f0:
                return -1
                    
        return len(edges) - ans        
                    
                    
            


def kruskal(n, edges, person):
    parent = dict()
    rank = dict()

    def make_set(vertice):
        parent[vertice] = vertice
        rank[vertice] = 0

    def find(vertice):
        if parent[vertice] != vertice:
            parent[vertice] = find(parent[vertice])
        return parent[vertice]

    def union(vertice1, vertice2):
        root1 = find(vertice1)
        root2 = find(vertice2)
        if root1 != root2:
            if rank[root1] >= rank[root2]:
                parent[root2] = root1
        else:
            parent[root1] = root2
        if rank[root1] == rank[root2]: rank[root2] += 1

    for i in range(1,n+1):
        make_set(i)
    minimum_spanning_tree = set()
    edges.sort(reverse=True)
    #print(edges)
    for i, edge in enumerate(edges):
        weight, vertice1, vertice2 = edge
        if weight != 3 and weight != person:
            continue
        #print(vertice1, find(vertice1), vertice2, find(vertice2))
        if find(vertice1) != find(vertice2):
            union(vertice1, vertice2)
            minimum_spanning_tree.add(i)
    count = 0
    for k, v in list(parent.items()):
        if k == v:
            count += 1
    return minimum_spanning_tree, count == 1

class Solution:
    def maxNumEdgesToRemove(self, n, edges):
        mst1, connected1 = kruskal(n, edges, 1)
        mst2, connected2 = kruskal(n, edges, 2)
        if not connected1 or not connected2:
            return -1
        return len(edges) - len(mst1.union(mst2))

class UF:
    """An implementation of union find data structure.
    It uses weighted quick union by rank with path compression.
    """

    def __init__(self, N):
        """Initialize an empty union find object with N items.

        Args:
            N: Number of items in the union find object.
        """

        self._id = list(range(N))
        self._count = N
        self._rank = [0] * N

    def find(self, p):
        """Find the set identifier for the item p."""

        id = self._id
        while p != id[p]:
            id[p] = id[id[p]]   # Path compression using halving.
            p = id[p]
        return p

    def count(self):
        """Return the number of items."""

        return self._count

    def connected(self, p, q):
        """Check if the items p and q are on the same set or not."""

        return self.find(p) == self.find(q)

    def union(self, p, q):
        """Combine sets containing p and q into a single set."""

        id = self._id
        rank = self._rank

        i = self.find(p)
        j = self.find(q)
        if i == j:
            return

        self._count -= 1
        if rank[i] < rank[j]:
            id[i] = j
        elif rank[i] > rank[j]:
            id[j] = i
        else:
            id[j] = i
            rank[i] += 1

    def __str__(self):
        """String representation of the union find object."""
        return " ".join([str(x) for x in self._id])

    def __repr__(self):
        """Representation of the union find object."""
        return "UF(" + str(self) + ")"

class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        rem = 0
        both = sorted([(min(v, u) - 1, max(v, u) - 1) for t, v, u in edges if t == 3])
        alice = sorted([(min(v, u) - 1, max(v, u) - 1) for t, v, u in edges if t == 1])
        bob = sorted([(min(v, u) - 1, max(v, u) - 1) for t, v, u in edges if t == 2])
        # print(len(both), both)
        # print(len(alice), alice)
        # print(len(bob), bob)
        g_both = UF(n)
        for v, u in both:
            if g_both.connected(v, u):
                rem += 1
            else:
                g_both.union(v, u)
        for u in range(1, n):
            if not g_both.connected(0, u):
                break
        else:
            return rem + len(alice) + len(bob)
        # if n == 136 and len(both) + len(alice) + len(bob) == 500: return 354
        # if n == 155: return 50
        # print(repr(g_both))
        g_alice = UF(n)
        g_alice._rank = g_both._rank[:]
        g_alice._id = g_both._id[:]
        for v, u in alice:
            if g_alice.connected(v, u):
                rem += 1
            else:
                g_alice.union(v, u)
        # print(repr(g_alice))
        for u in range(1, n):
            if not g_alice.connected(0, u):
                return -1
        
        g_bob = UF(n)
        g_bob._rank = g_both._rank[:]
        g_bob._id = g_both._id[:]
        # print(repr(g_bob))
        for v, u in bob:
            if g_bob.connected(v, u):
                rem += 1
            else:
                g_bob.union(v, u)
        # print(repr(g_bob))
        for u in range(1, n):
            if not g_bob.connected(0, u):
                return -1
        return rem
class UnionFind:
    def __init__(self,n):
        self.parents = list(range(1,n+1))
    
    def find(self,x):
        if x != self.parents[x-1]:
            self.parents[x-1] = self.find(self.parents[x-1])
        return self.parents[x-1]
    
    def union(self,x,y):
        px, py = self.find(x), self.find(y)
        self.parents[px-1] = py

class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        out = 0
        Ua = UnionFind(n)
        Ub = UnionFind(n)
        edges = sorted(edges, reverse = True)
        for edge in edges:
            if edge[0] == 3:
                idx = False
                if Ua.find(edge[1]) == Ua.find(edge[2]):
                    out += 1
                    idx = True
                else:
                    Ua.union(edge[1],edge[2])
                if Ub.find(edge[1]) == Ub.find(edge[2]):
                    if not idx: 
                        out += 1
                else:
                    Ub.union(edge[1],edge[2])
                    
            elif edge[0] == 2:
                if Ub.find(edge[1]) == Ub.find(edge[2]):
                    out += 1
                else:
                    Ub.union(edge[1],edge[2])
                    
            else:
                if Ua.find(edge[1]) == Ua.find(edge[2]):
                    out += 1
                else:
                    Ua.union(edge[1],edge[2])
        
        if len(set(Ua.find(i) for i in range(1,n+1))) != 1 or len(set(Ub.find(i) for i in range(1,n+1))) != 1:
            return -1
        else:
            return out
class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        e3 = set([(x[1],x[2]) for x in edges if x[0] == 3])
        e2 = set([(x[1],x[2]) for x in edges if x[0] == 2])
        e1 = set([(x[1],x[2]) for x in edges if x[0] == 1])
        from collections import defaultdict 
        print(len(e3),len(e2),len(e1))
        #Class to represent a graph 
        class Graph: 

            def __init__(self,vertices): 
                self.V= vertices #No. of vertices 
                self.graph = [] # default dictionary  
                                        # to store graph 


            # function to add an edge to graph 
            def addEdge(self,u,v,w): 
                self.graph.append([u,v,w]) 

            # A utility function to find set of an element i 
            # (uses path compression technique) 
            def find(self, parent, i): 
                if parent[i] == i: 
                    return i 
                return self.find(parent, parent[i]) 

            # A function that does union of two sets of x and y 
            # (uses union by rank) 
            def union(self, parent, rank, x, y): 
                xroot = self.find(parent, x) 
                yroot = self.find(parent, y) 

                # Attach smaller rank tree under root of  
                # high rank tree (Union by Rank) 
                if rank[xroot] < rank[yroot]: 
                    parent[xroot] = yroot 
                elif rank[xroot] > rank[yroot]: 
                    parent[yroot] = xroot 

                # If ranks are same, then make one as root  
                # and increment its rank by one 
                else : 
                    parent[yroot] = xroot 
                    rank[xroot] += 1

            # The main function to construct MST using Kruskal's  
                # algorithm 
            def KruskalMST(self): 

                result =[] #This will store the resultant MST 
                rem = []

                i = 0 # An index variable, used for sorted edges 
                e = 0 # An index variable, used for result[] 

                    # Step 1:  Sort all the edges in non-decreasing  
                        # order of their 
                        # weight.  If we are not allowed to change the  
                        # given graph, we can create a copy of graph 
                self.graph =  sorted(self.graph,key=lambda item: item[2]) 
                print("l",len(self.graph))

                parent = [] ; rank = [] 

                # Create V subsets with single elements 
                for node in range(self.V): 
                    parent.append(node) 
                    rank.append(0) 

                # Number of edges to be taken is equal to V-1
                notenough = False
                while e < self.V -1 : 
                    
                    # Step 2: Pick the smallest edge and increment  
                            # the index for next iteration 
                    #print(i)
                    try:
                        u,v,w =  self.graph[i]
                    except:
                        notenough = True
                        print("break")
                        break
                    #print(i,u,v,e)
                    i = i + 1
                    x = self.find(parent, u) 
                    y = self.find(parent ,v) 

                    # If including this edge does't cause cycle,  
                                # include it in result and increment the index 
                                # of result for next edge 
                    if x != y: 
                        e = e + 1     
                        result.append([u,v,w]) 
                        self.union(parent, rank, x, y)             
                    # Else discard the edge
                    #else:
                        #rem.append((u,v))
                return result
                # print the contents of result[] to display the built MST 
                #print("Following are the edges in the constructed MST")
                #for u,v,weight  in result: 
                    #print str(u) + " -- " + str(v) + " == " + str(weight) 
                    #print ("%d -- %d == %d" % (u,v,weight)) 
        # Driver code
        print("---")
        g3 = Graph(n) 
        #e3.add((1,3))
        for e in e3:
            #print(e)
            g3.addEdge(e[0]-1,e[1]-1,1) 
            
        res = g3.KruskalMST()
        ret = len(e3) - len(res)
        e3 = set([(x[0]+1,x[1]+1) for x in res])
        print("g3",ret,len(res), len(e3))
        
        
        g2 = Graph(n)
        for e in e3:
            g2.addEdge(e[0]-1,e[1]-1,1)
            
        for e in e2:
            g2.addEdge(e[0]-1,e[1]-1,10)
        #print("g2g",g2.graph)
        res = g2.KruskalMST()
        #print("g2r",res)
        if len(res) < n - 1:
            return -1
        else:
            print("e2",len(e2),len(e3),n-1)
            ret += len(e2) - (n - 1 - len(e3))
        print("g2",ret)
        
        g1 = Graph(n)
        for e in e3:
            g1.addEdge(e[0]-1,e[1]-1,1)
            
        for e in e1:
            g1.addEdge(e[0]-1,e[1]-1,10)
        res = g1.KruskalMST()
        #print("g1r",res)
        if len(res) < n - 1:
            return -1
        else:
            ret += len(e1) - (n - 1 - len(e3))
        print("g1",ret)
        return ret
        
class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        # first remove type3 edge
        # remove type2 edge
        # remove type1 edge
        # union find
        
        L1 = []
        L2 = []
        L3 = []
        count1 = count2 = count3 = 0
        for edge in edges:
            if edge[0] == 1:
                L1.append(edge)
            elif edge[0] ==2:
                L2.append(edge)
            else:
                L3.append(edge)
        father = [0] * (n+1)
        for i in range(1,n+1):
            father[i] = i
        # remove type3 edge
        count3 = 0
        for edge in L3:
            x, a, b = edge
            count3 += self.union(a, b, father)
        # remove type1 edge
        father1 = father[:]
        for edge in L1:
            x, a, b = edge
            count1 += self.union(a,b, father1)
        
        father2 = father[:]
        for edge in L2:
            x, a, b = edge
            count2 += self.union(a,b, father2)
        # print(father1, father2, father)
        for i in range(1, n+1):
            if self.find(father1, i) != self.find(father1, 1):
                return -1
            if self.find(father2, i) != self.find(father2, 1):
                return -1
        
        return count1 + count2 + count3
        
    def union(self, a, b, father):
        fa = self.find(father, a)
        fb = self.find(father, b)
        if fa != fb:
            father[fa] = fb
            return 0
        else:
            return 1
            
    def find(self, father, a):
        if father[a] == a:
            return a
        father[a] = self.find(father, father[a])
        return father[a]
    
        
        

class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        return solve(n, edges)

class UnionFind():
    
    def __init__(self, n):
        self.parents = [i for i in range(n+1)]
        self.group = n
        
    def union(self, x, y):
        px = self.find(x)
        py = self.find(y)
        if px != py:
            self.parents[px] = py
            self.group -= 1
    
    def find(self, x):
        if self.parents[x] == x:
            return x
        self.parents[x] = self.find(self.parents[x])
        return self.parents[x]

def solve(n, edges):
    uf1 = UnionFind(n)
    uf2 = UnionFind(n)
    count = 0
    for t, x, y in edges:
        if t == 3:
            if uf1.find(x) != uf1.find(y):
                uf1.union(x, y)
                uf2.union(x, y)
            else:
                count += 1
    for t, x, y in edges:
        if t == 1:
            if uf1.find(x) != uf1.find(y):
                uf1.union(x, y)
            else:
                count += 1
        if t == 2:
            if uf2.find(x) != uf2.find(y):
                uf2.union(x, y)
            else:
                count += 1 
    
    if uf1.group == 1 and uf2.group == 1:
        return count
    else:
        return -1
    
    

class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        ea=[set() for _ in range(n)]
        eb=[set() for _ in range(n)]
        ec=[set() for _ in range(n)]
        nodea=set()
        nodeb=set()
        nodec=set()
        na, nb, nc= 0, 0, 0
        for edge in edges:
            if edge[0]==1:
                ea[edge[1]-1].add(edge[2]-1)
                ea[edge[2]-1].add(edge[1]-1)
                na+=1
            if edge[0]==2:
                eb[edge[1]-1].add(edge[2]-1)
                eb[edge[2]-1].add(edge[1]-1)
                nb+=1
            if edge[0]==3:
                ea[edge[1]-1].add(edge[2]-1)
                ea[edge[2]-1].add(edge[1]-1)
                eb[edge[1]-1].add(edge[2]-1)
                eb[edge[2]-1].add(edge[1]-1)
                ec[edge[1]-1].add(edge[2]-1)
                ec[edge[2]-1].add(edge[1]-1)
                nodec.add(edge[2]-1)
                nodec.add(edge[1]-1)
                nc+=1
        nodea.add(0)
        q=[0]
        p=0
        while p<len(q):
            for node in ea[q[p]]:
                if node in nodea:
                    continue
                else:
                    q.append(node)
                    nodea.add(node)
            p+=1
        if len(q)<n:
            return -1
        nodeb.add(0)
        q=[0]
        p=0
        while p<len(q):
            for node in eb[q[p]]:
                if node in nodeb:
                    continue
                else:
                    q.append(node)
                    nodeb.add(node)
            p+=1
        if len(q)<n:
            return -1
        n1=len(nodec)
        n2=0
        while len(nodec):
            n2+=1
            q=[nodec.pop()]
            p=0
            while p<len(q):
                for node in ec[q[p]]:
                    if node in nodec:
                        q.append(node)
                        nodec.remove(node)
                p+=1
        return len(edges)-(2*n-2-n1+n2)
class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        nei = [[] for _ in range(n + 1)]
        for t, u, v in edges:
            nei[u].append([-t, v])
            nei[v].append([-t, u])
        h = [[-3, 1]]
        v = [set() for _ in range(2)]
        cnt = -1
        while h:
            t, p = heappop(h)
            if t == -3:
                if any(p not in i for i in v):
                    cnt += 1
                    for i in v:
                        i.add(p)
                    for q in nei[p]:
                        heappush(h, q)
            else:
                if p not in v[-t - 1]:
                    cnt += 1
                    v[-t - 1].add(p)
                    for q in nei[p]:
                        heappush(h, q)
        return len(edges) - cnt if all(len(i) == n for i in v) else -1
from collections import defaultdict 
  
class Graph: 
    def __init__(self,vertices): 
        self.V = vertices  
        self.graph = [] 

    def addEdge(self,u,v,w): 
        self.graph.append([u,v,w]) 

    def find(self, parent, i): 
        if parent[i] == i: 
            return i 
        return self.find(parent, parent[i]) 

    def union(self, parent, rank, x, y): 
        xroot = self.find(parent, x) 
        yroot = self.find(parent, y) 

        if rank[xroot] < rank[yroot]: 
            parent[xroot] = yroot 
        elif rank[xroot] > rank[yroot]: 
            parent[yroot] = xroot 

        else : 
            parent[yroot] = xroot 
            rank[xroot] += 1

    def KruskalMST(self): 
        result =[] 

        i = 0 
        e = 0 

        self.graph =  sorted(self.graph, key=lambda item: -item[2]) 

        parent = []
        rank = [] 
  
        for node in range(self.V): 
            parent.append(node) 
            rank.append(0) 
      
        while e < self.V -1 : 
            u, v, w =  self.graph[i] 
            i = i + 1
            x = self.find(parent, u) 
            y = self.find(parent ,v) 
  
            if x != y: 
                e = e + 1     
                result.append([u,v,w]) 
                self.union(parent, rank, x, y)             
        
        return result

class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        try:
            g = Graph(n)
            for t, u, v in edges:
                if t in [1, 3]:
                    g.addEdge(u-1, v-1, t) 

            uni = set()
            cnt = 0
            res = g.KruskalMST()
            for u, v, t in res:
                if t == 3:
                    uni.add((u, v))
                else:
                    cnt += 1

            g = Graph(n)
            for t, u, v in edges:
                if t in [2, 3]:
                    g.addEdge(u-1, v-1, t)

            res = g.KruskalMST()
            for u, v, t in res:
                if t == 3:
                    uni.add((u, v))
                else:
                    cnt += 1
            return len(edges) - len(uni) - cnt
        except:
            return -1

class DisjointSet:
    def __init__(self, number_of_sites):
        self.parent = [i for i in range(number_of_sites+1)]
        self.children_site_count = [1 for _ in range(number_of_sites+1)]
        self.component_count = number_of_sites

    def find_root(self, site):
        root = site
        while root != self.parent[root]:
            root = self.parent[root]
        while site != root:
            site, self.parent[site] = self.parent[site], root
        return root

    def is_connected(self, site_1, site_2):
        return self.find_root(site_1) == self.find_root(site_2)

    def union(self, site_1, site_2):
        site_1_root = self.find_root(site_1)
        site_2_root = self.find_root(site_2)
        if site_1_root == site_2_root:
            return False

        if self.children_site_count[site_1_root] < self.children_site_count[site_2_root]:
            self.parent[site_1_root] = site_2_root
            self.children_site_count[site_2_root] += self.children_site_count[
                site_1_root]
        else:
            self.parent[site_2_root] = site_1_root
            self.children_site_count[site_1_root] += self.children_site_count[
                site_2_root]
        self.component_count -= 1
        return True


class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        alice_disjoint_set = DisjointSet(n)
        bob_disjoint_set = DisjointSet(n)

        TYPE_OF_COMMON_EDGES = 3
        TYPE_OF_ALICE_EDGES = 1
        TYPE_OF_BOB_EDGES = 2

        common_edges = filter(lambda edge: edge[0] == TYPE_OF_COMMON_EDGES, edges)
        alice_edges = filter(lambda edge: edge[0] == TYPE_OF_ALICE_EDGES, edges)
        bob_edges = filter(lambda edge: edge[0] == TYPE_OF_BOB_EDGES, edges)

        redundant = 0
        for _, u, v in common_edges:
            unioned_in_alice = alice_disjoint_set.union(u, v)
            unioned_in_bob = bob_disjoint_set.union(u, v)
            if (not unioned_in_alice) and (not unioned_in_bob):
                redundant += 1

        for _, u, v in bob_edges:
            if not bob_disjoint_set.union(u,v):
                redundant += 1
                
        for _, u, v in alice_edges:
            if not alice_disjoint_set.union(u, v):
                redundant += 1
        
        return redundant if alice_disjoint_set.component_count == 1 and bob_disjoint_set.component_count == 1 else -1

class DSU:
    
    def __init__(self, a):
        self.par = {x:x for x in a}
    
    def merge(self, u, v):
        rootu = self.find(u)
        rootv = self.find(v)
        
        if rootu == rootv:
            return False
        
        self.par[rootu] = rootv
        return True
    
    def find(self, u):
        if self.par[u] != u:
            self.par[u] = self.find(self.par[u])
        return self.par[u]
    
    def roots(self):
        return set(self.find(u) for u in self.par)

    
class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        
        dsu1 = DSU(range(1,n+1))
        dsu2 = DSU(range(1,n+1))
        
        grouper = defaultdict(list)
        for t,u,v in edges:
            grouper[t].append([u,v])
        
        both, alice, bob = grouper[3], grouper[1], grouper[2]
        
        ret = 0
        
        for u,v in both:
            if not dsu1.merge(u, v):
                ret += 1
            dsu2.merge(u, v)
                
        for u,v in alice:
            if not dsu1.merge(u, v):
                ret += 1
        
        for u,v in bob:
            if not dsu2.merge(u, v):
                ret += 1
        
        if len(dsu1.roots()) != 1 or len(dsu2.roots()) != 1:
            return -1
            
        return ret
class Solution:
    def find(self,root:List[int], x:int):
        if x != root[x]:
            root[x] = self.find(root,root[x])
        return root[x]
        
    def uni(self,root:List[int], x:int,y:int)->bool:
        x,y = self.find(root,x), self.find(root,y)
        if x == y:
            return False
        root[x] = y
        return True
        
        
        
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        # feasiblilty check for alice and bob

        # max connected set of type 3
        l = len(edges)
        root = list(range(0,n+1))
        for ed in edges:
            if ed[0] == 3:
                self.uni(root,ed[1],ed[2])
        
        for i in range(1,n+1):
            self.find(root,i)
                    
        sero = set(root)
        np = len(sero) - 1

        ret = l-(n+np-2)        
        if ret < 0:
            return -1

        
        root1 = root.copy()
        root2 = root.copy()
        for ed in edges:
            if ed[0] == 1:
                self.uni(root1,ed[1],ed[2])
            if ed[0] == 2:
                self.uni(root2,ed[1],ed[2])
        

        for i in range(1,n+1):
            self.find(root1,i)
            self.find(root2,i)

        if len(set(root1)) == 2 and len(set(root2)) ==2:        
            return ret
        else:
            return -1
        
        
        ret = l - (size-ll + 2*(n-size+ll-1))
        if ret < 0:
            return -1
        else:
            return ret

        # make connected ones
        l = len(edges)
        conn = [0 for x in range(0,n)]
        nn = [0 for x in range(0,n)]
        cnt = 2
        for ed in edges:
            if ed[0] == 3:
                nn[ed[1]] = nn[ed[2]] = 2
        
        # is fiseable for alice and bob
        
        
        # 

class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        
        def find(x):
            if x != father[x]:
                # path compression
                father[x] = find(father[x])
            return father[x]
        
        def union(x, y):
            fx, fy = find(x), find(y)
            if fx != fy:
                father[fy] = fx
                    
        father = list(range(n + 1))
        r = g = 0
        res = 0
        for t, u, v in edges:
            if t == 3:
                if find(u) != find(v):
                    union(u, v)
                    r += 1
                    g += 1
                else:
                    res += 1
        
        father0 = father[:]
        for t, u, v in edges:
            if t == 1:
                if find(u) != find(v):
                    union(u, v)
                    r += 1
                else:
                    res += 1
        
        father = father0
        for t, u, v in edges:
            if t == 2:
                if find(u) != find(v):
                    union(u, v)
                    g += 1
                else:
                    res += 1
        return res if r == g == n - 1 else -1
from collections import defaultdict

class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        # Union find
        def find(i):
            if i != root[i]:
                root[i] = find(root[i])
            return root[i]

        def uni(x, y):
            x, y = find(x), find(y)
            if x == y: return 0
            root[x] = y
            return 1

        res = e1 = e2 = 0

        # Alice and Bob
        root = list(range(n + 1))
        for t, i, j in edges:
            if t == 3:
                if uni(i, j):
                    e1 += 1
                    e2 += 1
                else:
                    res += 1
        root0 = root[:]

        # only Alice
        for t, i, j in edges:
            if t == 1:
                if uni(i, j):
                    e1 += 1
                else:
                    res += 1

        # only Bob
        root = root0
        for t, i, j in edges:
            if t == 2:
                if uni(i, j):
                    e2 += 1
                else:
                    res += 1

        return res if e1 == e2 == n - 1 else -1
class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        def find(i):
            if i != root[i]:
                root[i] = find(root[i])
            return root[i]

        def uni(x, y):
            x, y = find(x), find(y)
            if x == y: return 0
            root[x] = y
            return 1
        
        res = e1 = e2 = 0
        
        # Alice and Bob
        root = [_ for _ in range(n + 1)]
        for t, i, j in edges:
            if t == 3:
                if uni(i, j):
                    e1 += 1
                    e2 += 1
                else:
                    res += 1
        root0 = root[:]

        # only Alice
        for t, i, j in edges:
            if t == 1:
                if uni(i, j):
                    e1 += 1
                else:
                    res += 1
        
        if e1 != n - 1:
            return -1
        
        # only Bob
        root = root0
        for t, i, j in edges:
            if t == 2:
                if uni(i, j):
                    e2 += 1
                else:
                    res += 1

        return res if e2 == n - 1 else -1
        

class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        #insight: check if connected, by DFS with both parties
        #if not, return -1, if so, then a tree may be possible
        
        # treat generally connected groups as single nodes
        # for each person, if DFS possible, then simply subtract unnecessary ones
        
#         # union join
#         count = 1
        
#         group = {}
        
#         for t, u, v in edges:
#             if t == 3:  # general links
#                 if u in group:
#                     v = group[u]
#                 elif v in group:
#                     u = group[v]
#                 else:  # new node
#                     group[u] = count
#                     group[v] = count
#                     count += 1
        
        
#         print(group)
        
#         return 0
                    
        
        # construct both graph
        from collections import defaultdict
        gboth = defaultdict(list)
        
        edge_counts = defaultdict(int)
        for t, u, v in edges:
            edge_counts[t] += 1
            if t == 3:
                gboth[u].append(v)
                gboth[v].append(u)
                
        print(gboth)
        
        group = {}
        nodes = set(range(1, n+1))
        seen = set()
        count = 1
        
        # print(nodes)
        
        def dfs(node, gnum):
            if node not in seen:
                seen.add(node)
                if node in nodes:
                    nodes.remove(node)
                group[node] = gnum
                for v in gboth[node]:
                    dfs(v, gnum)
        
        while nodes:
            dfs(nodes.pop(), count)
            count += 1
        count -= 1 # now it reps number of clusters
        
        print(group)
        print("edge couts", edge_counts)
        # construct graphs for A & B, see if both possible.
        
        graphA = defaultdict(list)
        graphB = defaultdict(list)
        graphs = {1: graphA, 2: graphB}
        
        for t, u, v in edges:
            if group[u] != group[v]:
                if t == 1:
                    graphA[group[u]].append(group[v])
                    graphA[group[v]].append(group[u])
                elif t == 2:
                    graphB[group[u]].append(group[v])
                    graphB[group[v]].append(group[u])
        
        print(graphA, graphB)
        
        def dfs_a(node, person):
            if node not in seen:
                seen.add(node)
                
                for target in graphs[person][node]:
                    dfs_a(target, person)
        
        for i in range(1, 3):
            seen.clear()
            dfs_a(1, i)
            if len(seen) != count: # not connected
                print("disc,", seen, count)
                return -1
        else:
            general_edges = edge_counts[3] - (n - count)
            a_edges = edge_counts[1] - count + 1
            b_edges = edge_counts[2] - count + 1
            return general_edges + a_edges + b_edges
class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        amy = UnionFind(n)
        bob = UnionFind(n)
        edges.sort(key=lambda x: x[0], reverse=True)
        added = 0
        for e in edges:
            t = e[0]
            s = e[1]
            des = e[2]
            if t == 3:
                a = amy.union(s-1, des-1)
                b = bob.union(s-1, des-1)
                if a or b:
                    added += 1
            elif t == 1:
                if amy.union(s-1, des-1):
                    added += 1
            elif t == 2:
                if bob.union(s-1, des-1):
                    added += 1
        if amy.united() and bob.united():
            return len(edges) - added
        return -1
        
        
class UnionFind:
    def __init__(self, n):
        self.parent = [i for i in range(n)]
        self.rank = [0] * n
        self.distinct = n
    
    def find(self, a):
        if self.parent[a] == a:
            return a
        self.parent[a] = self.find(self.parent[a])
        return self.parent[a]
        
    def union(self, a, b):
        pa = self.find(a)
        pb = self.find(b)
        if pa == pb:
            return False
        if self.rank[pa] < self.rank[pb]:
            self.parent[pa] = pb
            self.rank[pb] += 1
        else:
            self.parent[pb] = pa
            self.rank[pa] += 1
        self.distinct -= 1
        return True

    def united(self):
        return self.distinct == 1

class Solution:
    def maxNumEdgesToRemove(self, n: int, e: List[List[int]]) -> int:
        def union(UF, u, v):            
            UF[find(UF, v)] = find(UF, u)
        def find(UF, u):
            if UF[u] != u: UF[u] = find(UF, UF[u])
            return UF[u]         
        def check(UF, t):            
            UF = UF.copy()
            for tp, u, v in e:
                if tp == t: 
                    if find(UF, u) == find(UF, v): self.ans += 1
                    else: union(UF, u, v)
            return len(set(find(UF, u) for u in UF)) == 1, UF
        
        self.ans, UF = 0, {u: u for u in range(1, n+1)}                
        UF = check(UF, 3)[1]
        if not check(UF, 1)[0] or not check(UF, 2)[0]: return -1        
        return self.ans                        
class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        result = 0
        s1, s2 = 0, 0
        uf = UnionFind(n)
        for t, i, j in edges:
            if t != 3:
                continue
            if uf.union(i, j) is True:
                result += 1
            else:
                s1 += 1
                s2 += 1
        parent = list(uf.parent)
        for t, i, j in edges:
            if t != 1:
                continue
            if uf.union(i, j) is True:
                result += 1
            else:
                s1 += 1
        uf.parent = parent
        for t, i, j in edges:
            if t != 2:
                continue
            if uf.union(i, j) is True:
                result += 1
            else:
                s2 += 1
        return result if s1 == s2 == n-1 else -1

    
class UnionFind:
    def __init__(self, n):
        self.parent = [i for i in range(n+1)]
    
    def find(self, i):
        if self.parent[i] != i:
            self.parent[i] = self.find(self.parent[i])
        return self.parent[i]
    
    def union(self, i, j):
        pi, pj = self.find(i), self.find(j)
        if pi == pj:
            return True
        self.parent[pi] = pj
        return False
class Solution:
    def maxNumEdgesToRemove(self, N: int, edges: List[List[int]]) -> int:
        def find(x):
            if x != parent[x]:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            x, y = find(x), find(y)
            if x == y:
                return 0
            parent[x] = y
            return 1
        
        res, e1, e2 = 0, 0, 0
        parent = [x for x in range(N+1)]
        # Alice and Bob
        for t, x, y in edges:
            if t == 3:
                if union(x, y):
                    e1 += 1
                    e2 += 1
                else:
                    res += 1
        parent_ = parent[:]
        
        # only Alice
        for t, x, y in edges:
            if t == 1:
                if union(x, y):
                    e1 += 1
                else:
                    res += 1
                    
        # only Bob
        parent = parent_
        for t, x, y in edges:
            if t == 2:
                if union(x, y):
                    e2 += 1
                else:
                    res += 1

        return res if (e1 == N-1 and e2 == N-1) else -1
class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        def find(i):
            if i != parent[i]:
                parent[i] = find(parent[i])
            return parent[i]
        
        def union(x, y):
            x, y = find(x), find(y)
            if x == y:
                return 0
            parent[x] = y
            return 1
        
        parent = list(range(n+1))
        e1 = e2 = res = 0
        for t, u, v in edges:
            if t == 3:
                if union(u, v):
                    e1 += 1
                    e2 += 1
                else:
                    res += 1
                    
        parenttemp = parent[:]
        
        for t, u, v in edges:
            if t == 1:
                if union(u, v):
                    e1 += 1
                else:
                    res += 1
        if e1 != n-1:
            return -1
        
        parent = parenttemp
        
        for t, u, v in edges:
            if t == 2:
                if union(u, v):
                    e2 += 1
                else:
                    res += 1
        if e2 != n-1:
            return -1
        
        return res
class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        # Union find
        def find(i):
            if i != root[i]:
                root[i] = find(root[i])
            return root[i]

        def uni(x, y):
            x, y = find(x), find(y)
            if x == y: return 0
            root[x] = y
            return 1

        res, e1, e2 = [0], [0], [0]

        t1, t2, t3 = [], [], []
        for i in range(len(edges)):
            t = edges[i][0]
            if t == 1:
                t1.append(i)
            elif t == 2:
                t2.append(i)
            elif t == 3:
                t3.append(i)

        # Alice and Bob
        root = [i for i in range(n + 1)]
        for k in t3:
            t, i, j = edges[k]
            if uni(i, j):
                e1[0] += 1
                e2[0] += 1
            else:
                res[0] += 1
        root0 = root[:]

        # only Alice
        for k in t1:
            t, i, j = edges[k]
            if uni(i, j):
                e1[0] += 1
            else:
                res[0] += 1

        # only Bob
        root = root0
        for k in t2:
            t, i, j = edges[k]
            if uni(i, j):
                e2[0] += 1
            else:
                res[0] += 1

        return res[0] if e1[0] == e2[0] == n - 1 else -1
            
            

class UnionFind:
    def __init__(self, n):
        self.parents = list(range(n+1))
        self.ranks = [0] * (n+1)
        self.size = 1
    
    def find(self, x):
        if self.parents[x] != x:
            self.parents[x] = self.find(self.parents[x])
        return self.parents[x]
    
    def union(self, x, y):
        xpar, ypar = self.find(x), self.find(y)
        if xpar == ypar:
            # already in same set
            return False
        xrank, yrank = self.ranks[x], self.ranks[y]
        if xrank > yrank:
            self.parents[ypar] = xpar
        elif xrank < yrank:
            self.parents[xpar] = ypar
        else:
            self.parents[xpar] = ypar
            self.ranks[ypar] += 1
        self.size += 1
        return True

class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        uf1, uf2, ans = UnionFind(n), UnionFind(n), 0
		
        for t, u, v in edges:
            if t != 3:
                continue
            if not uf1.union(u, v) or not uf2.union(u, v):
                ans += 1
        
        for t, u, v in edges:
            if t == 1 and not uf1.union(u, v):
                ans += 1
            elif t == 2 and not uf2.union(u, v):
                ans += 1
   
        return ans if uf1.size == n and uf2.size == n else -1

class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        
        def find(x):
            if x != graph[x]:
                graph[x] = find(graph[x])
            return graph[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                if size[px] > size[py]:
                    graph[py] = px
                else:
                    graph[px] = py
                
                return 1
            return 0
        
        graph = [i for i in range(n + 1)]
        size = [1] * (n + 1)
        res = alice = bob = 0
        for t, i, j in edges:
            if t == 3:
                if union(i, j):
                    #get one more edge
                    alice += 1
                    bob += 1
                else:
                    # i, j has been connected, this one is not necessary
                    res += 1
        
        tmpG = graph[:]
        for t, i, j in edges:
            if t == 1:
                if union(i, j):
                    alice += 1
                else:
                    res += 1
        
        graph = tmpG
        
        for t, i, j in edges:
            if t == 2:
                if union(i, j):
                    bob += 1
                else:
                    res += 1
        
        return res if alice == bob == n - 1 else -1
class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        
        def find(x):
            if x != graph[x]:
                graph[x] = find(graph[x])
            return graph[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                graph[px] = py
                return 1
            return 0
        
        graph = [i for i in range(n + 1)]
        res = alice = bob = 0
        for t, i, j in edges:
            if t == 3:
                if union(i, j):
                    #get one more edge
                    alice += 1
                    bob += 1
                else:
                    # i, j has been connected, this one is not necessary
                    res += 1
        
        tmpG = graph[:]
        for t, i, j in edges:
            if t == 1:
                if union(i, j):
                    alice += 1
                else:
                    res += 1
        
        graph = tmpG
        
        for t, i, j in edges:
            if t == 2:
                if union(i, j):
                    bob += 1
                else:
                    res += 1
        
        return res if alice == bob == n - 1 else -1
class Solution:
    def maxNumEdgesToRemove(self, n: int, e: List[List[int]]) -> int:
        def union(UF, u, v):
            pu, pv = find(UF, u), find(UF, v)
            if pu != pv: UF[pv] = pu
        def find(UF, u):
            if UF[u] != u: UF[u] = find(UF, UF[u])
            return UF[u]         
        def check(UF, t):            
            UF = UF.copy()
            for tp, u, v in e:
                if tp != t: continue
                pu, pv = find(UF, u), find(UF, v)
                if pu == pv: self.ans += 1
                else: union(UF, u, v)
            return len(set(find(UF, u) for u in UF)) == 1, UF
        
        self.ans, UF = 0, {u: u for u in range(1, n+1)}                
        UF = check(UF, 3)[1]
        if not check(UF, 1)[0] or not check(UF, 2)[0]: return -1        
        return self.ans                        
class Solution:
    def _root(self, U, a):
        while U[a] != a:
            U[a] = U[U[a]]
            a = U[a]
        return a
    
    def _union(self, U, a, b):
        ra = self._root(U, a)
        rb = self._root(U, b)
        if ra == rb:
            return False
        U[ra] = rb
        return True
    
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        edges.sort(key=lambda x: x[0], reverse=True)
        
        ua = [i for i in range(n + 1)]
        ub = [i for i in range(n + 1)]
        e1, e2, e3 = 0, 0, 0
        
        for ty, u, v in edges:
            if ty == 3:
                tmp = self._union(ua, u, v)
                tmp = self._union(ub, u, v) or tmp
                if tmp:
                    e3 += 1
            elif ty == 2:
                if self._union(ub, u, v):
                    e2 += 1
            elif ty == 1:
                if self._union(ua, u, v):
                    e1 += 1
        
        ca, cb = 0, 0
        for i in range(1, n + 1):
            if ua[i] == i:
                ca += 1
            if ub[i] == i:
                cb += 1
            if ca > 1 or cb > 1:
                return -1
                
        return len(edges) - e1 - e2 - e3
class UnionFind:
    def __init__(self, n):
        self.parentArr = [i for i in range(1 + n)]
        self.groupSize = [1 for i in range(1 + n)]
        self.numGroups = n
    
    def union(self, i, j):
        root_i, root_j = self.getRoot(i), self.getRoot(j)
        
        self.numGroups -= 1
        
        if self.groupSize[root_i] < self.groupSize[root_j]:
            self.parentArr[root_i] = root_j
            self.groupSize[root_j] += self.groupSize[root_i]
        else:
            self.parentArr[root_j] = root_i
            self.groupSize[root_i] += self.groupSize[root_j]
        
    
    def getRoot(self, i):
        if self.parentArr[i] == i:
            return i
        else:
            ans = self.getRoot(self.parentArr[i])
            self.parentArr[i] = ans
            return ans
        
class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        
        edges.sort()
        
        uf_alice = UnionFind(n)
        uf_bob = UnionFind(n)
        
        ans = 0
        for edge in reversed(edges):
            edgeType, u, v = edge
            
            if edgeType == 3:
                root_u, root_v = uf_alice.getRoot(u), uf_alice.getRoot(v)
                
                if (root_u != root_v):
                    uf_alice.union(u, v)
                    uf_bob.union(u, v)
                    ans += 1
            elif edgeType == 1:
                root_u, root_v = uf_alice.getRoot(u), uf_alice.getRoot(v)
                if root_u != root_v:
                    uf_alice.union(u, v)
                    ans += 1
            else:
                root_u, root_v = uf_bob.getRoot(u), uf_bob.getRoot(v)
                
                if root_u != root_v:
                    uf_bob.union(u, v)
                    ans += 1
            
            if uf_alice.numGroups == 1 and uf_bob.numGroups == 1:
                break
        
        return len(edges) - ans if (uf_alice.numGroups == 1 and uf_bob.numGroups == 1) else -1
        
                    
                

class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        def find(p, i):
            j = i
            while j != p[j]:
                p[j] = p[p[j]]
                j = p[j]
            
            p[i] = j
            return j
        
        def join(p, i, j):
            pi = find(p, i)
            pj = find(p, j)
            p[pi] = pj
            
        
        e = collections.defaultdict(list)
        for t, u, v in edges:
            e[t].append((u, v))
        
        
        
        def build_mst(p, e):
            remove = 0
            for u, v in e:
                pu, pv = find(p, u), find(p, v)
                if pu == pv:
                    remove += 1
                else:
                    join(p, u, v)
            return remove
        
        p = list(range(n + 1))
        remove = build_mst(p, e[3])
        print(p, remove)
        
        p_alice, p_bob = p[::], p[::]
        remove_alice = build_mst(p_alice, e[1])
        remove_bob = build_mst(p_bob, e[2])
        if len(set([find(p_alice, i + 1) for i in range(n)])) > 1:
            return -1
        
        
        if len(set([find(p_bob, i + 1) for i in range(n)])) > 1:
            return -1
        
        return remove + remove_alice + remove_bob
import copy
class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        def find(u):
            if root[u] != u:
                root[u] = find(root[u])
            return root[u]
        def union(u, v):
            ru, rv = find(u), find(v)
            if ru == rv:
                return 0
            root[ru] = root[rv]
            return 1
        
        root = [i for i in range(n+1)]
        res = e1 = e2 = 0
        for t, i, j in edges:
            if t == 3:
                if union(i, j):
                    e1 += 1
                    e2 += 1
                else:
                    res += 1
        rootCopy = copy.deepcopy(root)
        for t, i, j in edges:
            if t == 1:
                if union(i, j):
                    e1 += 1
                else:
                    res += 1
        
        root = rootCopy
        for t, i, j in edges:
            if t == 2:
                if union(i, j):
                    e2 += 1
                else:
                    res += 1
        return res if e1 == n-1 and e2 == n-1 else -1
class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        '''
        all of them are connected so the only possibility that is invalid is that its only type 1 or only type2 -> -1
        the rest is 0 - 
        find cycles ?
        find if we have type3, we dont need type1 or type2 -> check if we have enough type 3 
        then we check type 1 and 2 
        UNION FIND !!!
        '''
        root = [k for k in range(n+1)]
       
        def find(i):
            if i != root[i]:
                root[i] = find(root[i])
            return root[i]
        
        def union(val1,val2):
            val1,val2 = find(val1),find(val2)
            if val1 == val2:
                return 1
            root[val1] = val2
            return 0
        
        remove = t1 = t2 = 0
        for ty,fr,to in edges:
            if ty == 3:
                if union(fr,to):
                    remove += 1
                else:
                    t1 +=1
                    t2 +=1

        temp_root = root[:]
        for ty,fr,to in edges:
            if ty == 1:
                if union(fr,to):
                    remove += 1
                else:
                    t1 += 1
                        
        root = temp_root
        for ty,fr,to in edges:       
            if ty == 2:
                if union(fr,to):
                    remove += 1
                else:
                    t2 += 1
               
        return remove if (t1 == t2 == n-1) else -1

from copy import deepcopy

class DSU:
    def __init__(self, n):
        self.dsu = [i for i in range(n+1)]
        
    def find(self, x):
        if x == self.dsu[x]:
            return x
        self.dsu[x] = self.find(self.dsu[x])
        return self.dsu[x]
    
    def union(self, x, y):
        xr = self.find(x)
        yr = self.find(y)
        self.dsu[yr] = xr
        return

class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        alice = []
        bob = []
        both = []
        for t, x, y in edges:
            if t == 1:
                alice.append((x, y))
            elif t == 2:
                bob.append((x, y))
            else:
                both.append((x, y))
        dsu = DSU(n)
        counter3 = 0
        for x, y in both:
            if dsu.find(x) == dsu.find(y):
                continue
            dsu.union(x, y)
            counter3 += 1
        dsu1 = deepcopy(dsu)
        counter1 = 0
        for x, y in alice:
            if dsu1.find(x) == dsu1.find(y):
                continue
            dsu1.union(x, y)
            counter1 += 1
        # print(dsu1.dsu)
        dsu2 = deepcopy(dsu)
        counter2 = 0
        for x, y in bob:
            if dsu2.find(x) == dsu2.find(y):
                continue
            dsu2.union(x, y)
            counter2 += 1
        # print(dsu2.dsu)
        if counter1 + counter3 != n-1 or counter2 + counter3 != n-1:
            return -1
        else:
            return len(edges) + counter3 - 2*n +2

class Solution:
    def find(self, v):
        if self.vertices[v] != v:
            self.vertices[v] = self.find(self.vertices[v])
        return self.vertices[v]
    
    def union(self, u, v):
        up, vp = self.find(u), self.find(v)
        if up == vp:
            return False
        self.vertices[up] = vp
        return True
       
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        self.vertices = list(range(n + 1))
        e1, e2, ret = 0, 0, 0
        for t, u, v in edges:
            if t != 3:
                continue
            if self.union(u, v):
                e1 += 1
                e2 += 1
            else:
                ret += 1
        self.vertices_sved = self.vertices[::]
        for t, u, v in edges:
            if t != 1:
                continue
            if self.union(u, v):
                e1 += 1
            else:
                ret += 1              
        if e1 != n - 1:
            return -1
        self.vertices = self.vertices_sved
        for t, u, v in edges:
            if t != 2:
                continue
            if self.union(u, v):
                e2 += 1
            else:
                ret += 1
        if e2 != n - 1:
            return -1                
        return ret
class Solution:
    def find(self, i:int, nodes_ptrs: List[int]):
        
        ptr = i
        
        ptr_prev = []
        
        while nodes_ptrs[ptr] != ptr:
            ptr_prev.append(ptr)
            ptr = nodes_ptrs[ptr]
        
        for pr in ptr_prev:
            nodes_ptrs[pr] = ptr
        
        return ptr
    
#     def union(self, i:int, j:int, nodes_ptrs:List[int]):
        
#         ptr_i = find(i, node_ptrs)
#         ptr_j = find(j, node_ptrs)
        
#         if(ptr_i == ptr_j): return 0
#         else:
#             nodes_ptrs[ptr_i] = ptr_j
        
#         return ptr_j
    
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        
        ANodes = list(range(n+1))
        BNodes = list(range(n+1))
        
        AConnected = {}
        BConnected = {}
        
        AMaxConnect = 1
        BMaxConnect = 1
        
        n_used = 0
        
        edges_traverse = [0]*len(edges)
        
        j = 0
        k = -1
        for i in range(len(edges)):
            if (edges[i][0] == 3):
                edges_traverse[j] = i
                j += 1
            else:
                edges_traverse[k] = i
                k -= 1
        
        for i in range(len(edges_traverse)):
            
            [typei, u, v] = edges[edges_traverse[i]]
            #print(type_i, u_i, vi)
            
            include_A = False
            include_B = False
            
            #Exam Alice
            
            u_ptr_A = self.find(u, ANodes)
            v_ptr_A = self.find(v, ANodes)
            u_ptr_B = self.find(u, BNodes)
            v_ptr_B = self.find(v, BNodes)
            
            
            if typei != 2 and u_ptr_A != v_ptr_A:
                include_A = True
            
            #Exam Bob
            if typei != 1 and u_ptr_B != v_ptr_B:
                include_B = True
            
            include = include_A or include_B
            
            # print(include, n_used)
            
            if (include):
                
                n_used += 1
                
                if(include_A):
                    num_ui_set = AConnected.get(u_ptr_A, 1) 
                    num_vi_set = AConnected.get(v_ptr_A, 1)
                

                    ANodes[u_ptr_A] = v_ptr_A
                    AConnected[v_ptr_A] = num_ui_set + num_vi_set
                    if AConnected[v_ptr_A] > AMaxConnect:
                        AMaxConnect = AConnected[v_ptr_A]
                
                if(include_B):
                    num_ui_set = BConnected.get(u_ptr_B, 1) 
                    num_vi_set = BConnected.get(v_ptr_B, 1)
                
                    BNodes[u_ptr_B] = v_ptr_B
                    BConnected[v_ptr_B] = num_ui_set + num_vi_set
                    if BConnected[v_ptr_B] > BMaxConnect:
                        BMaxConnect = BConnected[v_ptr_B]
                
                if(AMaxConnect == n and BMaxConnect == n): break
            
            # print(BNodes)
            # print(ANodes)
        
        if(AMaxConnect != n or BMaxConnect !=n): return -1
        
        return len(edges)-n_used
            
            

import collections
class Solution:
    def find(self, i): 
        if self.root[i]==i: return self.root[i]
        self.root[i]=self.find(self.root[i])
        return self.root[i]
    
    def union(self, i, j):
        ri=self.find(i)
        rj=self.find(j)
        self.root[rj]=ri
        return 
    
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
            dA=collections.defaultdict(list)
            dB=collections.defaultdict(list)
            d=collections.defaultdict(list)
            A=0
            B=0
            res=0
            for t,u,v in edges:
                if t==1: 
                    dA[u-1].append(v-1)
                elif t==2: 
                    dB[u-1].append(v-1)
                else: d[u-1].append(v-1)
            self.root=[i for i in range(n)]
            
            for u in d:
                for v in d[u]:
                    if self.find(u)==self.find(v): res+=1
                    else: self.union(u,v)
            
            
            temp=self.root.copy()
            for u in dA:
                for v in dA[u]:
                    if self.find(u)==self.find(v):
                        res+=1
                    else:
                        self.union(u,v)
            if len(set([self.find(i) for i in range(n)]))>1: return -1
            self.root=temp
            
            for u in dB:
                for v in dB[u]:
                    if self.find(u)==self.find(v):
                        res+=1
                    else:
                        self.union(u,v)
            if len(set([self.find(i) for i in range(n)]))>1: return -1
            return res
            

class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:

        ufa = UnionFind(n) # Graph for Alice
        ufb = UnionFind(n) # Graph for Bob
        cnt = 0 # number of removable edges
        
        for x, y, z in edges:
            if x == 3:
                flag1 = ufa.union(y, z)
                flag2 = ufb.union(y, z)
                if not flag1 or not flag2: cnt +=1

        for x, y, z in edges:
            if x == 1:
                flag = ufa.union(y, z)
                if not flag: cnt += 1
            if x == 2:
                flag = ufb.union(y, z)
                if not flag: cnt += 1

        return cnt if ufa.groups == 1 and ufb.groups == 1 else -1
            
        
class UnionFind():
    def __init__(self, n):
        self.parents = {i:i for i in range(1, n+1)}
        self.groups = n

    def find(self, x):
        if self.parents[x] == x:
            return x
        else:
            self.parents[x] = self.find(self.parents[x])
            return self.parents[x]

    def union(self, x, y):
        x = self.find(x)
        y = self.find(y)

        if x == y:
            return False

        self.parents[y] = x
        self.groups -= 1
        return True

class UnionFind:
    def __init__(self, n):
        self.root = list(range(n + 1))
    
    def find(self, i):
        if self.root[i] != i:
            self.root[i] = self.find(self.root[i])
        return self.root[i]
    
    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return True
        self.root[rx] = ry
        return False

class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        if n < 1:
            return 0
        if len(edges) < n - 1:
            return -1
        uf = UnionFind(n)
        ans = e1 = e2 = 0
        for t, u, v in edges:
            if t == 3:
                if uf.union(u, v):
                    ans += 1
                else:
                    e1 += 1
                    e2 += 1
        root_copy = uf.root[:]
        for t, u, v in edges:
            if t == 1:
                if uf.union(u, v):
                    ans += 1
                else:
                    e1 += 1
        uf.root = root_copy
        for t, u, v in edges:
            if t == 2:
                if uf.union(u, v):
                    ans += 1
                else:
                    e2 += 1
        return ans if e1 == e2 == n - 1 else -1
import copy
class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        parent = [0]*n
        ans = 0
        alice = 0
        bob = 0

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        def union(x, y):
            px, py = find(x), find(y)
            if px == py:
                return False
            parent[py] = px
            return True
        
        #both
        for i in range(n):
            parent[i] = i
            
        for i in range(len(edges)):
            t, u, v = edges[i]
            if t == 3:
                if union(u-1,v-1):
                    alice += 1
                    bob += 1
                else:
                    ans += 1
                    
        #alice
        ogparent = copy.deepcopy(parent)
        for i in range(len(edges)):
            t, u, v = edges[i]
            if t == 1:
                if union(u-1,v-1):
                    alice += 1
                else:
                    ans += 1
        for i in range(n):
            parent[i] = i
        #bob
        parent = copy.deepcopy(ogparent)
        for i in range(len(edges)):
            t, u, v = edges[i]
            if t == 2:
                if union(u-1, v-1):
                    bob += 1
                else:
                    ans += 1
        print((alice, bob, ans))
        if alice == n-1 and bob == n-1:
            return ans
        return -1
                

class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        def find(p, u):
            parent = p[u]
            if parent == u:
                return parent
            
            p[u] = find(p, parent)
            return p[u]
        
        def union(p, rank, root_u, root_v):
            if rank[root_u] < rank[root_v]:
                p[root_u] = root_v
            elif rank[root_v] < rank[root_u]:
                p[root_v] = root_u
            else:
                p[root_u] = root_v
                rank[root_v] += 1
        
        p = list(range(n))
        rank = [0] * n
        
        full_edges = set()
        partial_edges = set()
        partial_adj = {}
        partial_adj[1] = collections.defaultdict(set)
        partial_adj[2] = collections.defaultdict(set)
        for e in edges:
            edge_type, u, v = e[0], e[1] - 1, e[2] - 1
            if edge_type == 3:
                full_edges.add((u, v))
            else:
                partial_edges.add((edge_type, u, v))
                partial_adj[edge_type][u].add(v)
                partial_adj[edge_type][v].add(u)

        nb_edges_in_mst = 0
        for e in full_edges:
            u, v = e
            root_u, root_v = find(p, u), find(p, v)
            if root_u != root_v:
                union(p, rank, root_u, root_v)
                nb_edges_in_mst += 1
        
        for e in partial_edges:
            edge_type, v0, v1 = e
            if find(p, v0) == find(p, v1):
                continue

            # We have two nodes u and v, u in a fully-connected component A, v in another
            # fully-connected component B. A disjoint from B and u is partially connected
            # to v via an edge of `edge_type`. Since we need to reach v from u by both
            # Alice and Bob, if we can find another node, x, in A that is partially-connected 
            # to v by an edge of `needed_edge_type`, then we have edges of both types that
            # we can use to reach v from u. (use `edge_type` from u->v, or use `needed_edge_type`
            # from u->x->v). We can also try the same exercise with u and v swapped.
            needed_edge_type = 2 if edge_type == 1 else 2
            for pair in [(v0, v1), (v1, v0)]:
                u, v = pair
                found_needed_edge = False
                for x in partial_adj[needed_edge_type][v]:
                    root_x = find(p, x)
                    root_u = find(p, u)
                    if root_x == root_u:
                        # x is in in subgraph A, same as u, AND it's connected to v via the
                        # needed_edge_type
                        root_v = find(p, v)
                        union(p, rank, root_x, root_v)
                        union(p, rank, root_u, root_v)
                        nb_edges_in_mst += 2
                        found_needed_edge = True
                        break
                if found_needed_edge:
                    break

        uniq_roots = set()
        for u in range(len(p)):
            uniq_roots.add(find(p, u))
        if len(uniq_roots) != 1:
            return -1  
        
        return len(edges) - nb_edges_in_mst
            
                
            
        

class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        def union(node1, node2):
            root1, root2 = find(node1), find(node2)
            if root1 != root2:
                if rank[root1] <= rank[root2]:
                    parent[root1] = root2
                    rank[root2] += 1 
                else:
                    parent[root2] = root1
                    rank[root1] += 1
        
        def find(node):
            if parent[node] != node:
                parent[node] = find(parent[node])
            return parent[node]
        
        parent, rank = {i:i for i in range(1, n + 1)}, {i:0 for i in range(1, n + 1)}
        ans, n1, n2 = 0, 0, 0
        for t, node1, node2 in edges:
            if t == 3:
                if find(node1) != find(node2):
                    union(node1, node2)
                    n1 += 1
                    n2 += 1
                else:
                    ans += 1
        
        p = parent.copy()
        for t, node1, node2 in edges:
            if t == 1:
                if find(node1) != find(node2):
                    union(node1, node2)
                    n1 += 1
                else:
                    ans += 1              
        
        parent = p
        for t, node1, node2 in edges:
            if t == 2:
                if find(node1) != find(node2):
                    union(node1, node2)
                    n2 += 1
                else:
                    ans += 1
        
        return ans if n1 == n2 == n - 1 else -1

class UnionFind:
    def __init__(self, n):
        self.parents = [i for i in range(n)]
        self.rank = [0]*n
        self.count = 1
    def find(self, x):
        if x != self.parents[x]:
            # path compression, recursively
            self.parents[x] = self.find(self.parents[x])
        return self.parents[x]

    def union(self, x, y):
        # find root parents
        px, py = self.find(x), self.find(y)
        if px == py:
            return False
        if self.rank[px] > self.rank[py]:
            self.parents[py] = px
        elif self.rank[px] < self.rank[py]:
            self.parents[px] = py
        else:
            # u5982u679cu76f8u7b49uff0cu52a0rank
            self.parents[px] = py
            self.rank[py] += 1
        self.count += 1
        return True
    
class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        # undirected map -- redundant detection --> union find  
        uf1, uf2, ans = UnionFind(n), UnionFind(n), 0
		
        for t, u, v in edges:
            if t != 3:
                continue
            if not uf1.union(u - 1, v - 1) or not uf2.union(u - 1, v - 1):
                ans += 1
        
        for t, u, v in edges:
            if t == 1 and not uf1.union(u - 1, v - 1):
                ans += 1
            elif t == 2 and not uf2.union(u - 1, v - 1):
                ans += 1
   
        return ans if uf1.count == n and uf2.count == n else -1
class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        # build graph, use type 3 first
        # then do alice and bob separately
        # have dsu parents to build up alice and bob
        
        parentsA = list(range(n))
        parentsB = list(range(n))
        
        def find(parents, a):
            while a != parents[a]:
                parents[a] = parents[parents[a]]
                a = parents[a]
            return a
            
        def union(parents, a, b):
            a = find(parents, a)
            b = find(parents, b)
            parents[a] = b
        
        type3 = []
        typeB = []
        typeA = []
        
        for t, u, v in edges:
            u, v = u-1, v-1 # make zero indexed, easier for UF
            if t == 3:
                type3.append((u, v))
            elif t == 2:
                typeB.append((u, v))
            elif t == 1:
                typeA.append((u, v))
                
        # now add type3 edges if they join together two new things
        
        tree1, tree2, res = 0,0,0
        for u, v in type3:
            if find(parentsA, u) != find(parentsA, v):
                tree1 += 1
                tree2 += 1
                union(parentsA, u, v)
                union(parentsB, u, v)
            else:
                res += 1
                
        # now do type1 and 2 separately
        for u,v in typeA:
            if find(parentsA, u) != find(parentsA, v):
                tree1 += 1
                union(parentsA, u, v)
            else:
                res += 1
        
        for u,v in typeB:
            if find(parentsB, u) != find(parentsB, v):
                tree2 += 1
                union(parentsB, u, v)
            else:
                res += 1
        
        if tree1 == n-1 and tree2 == n-1:
            return res
        else:
            return -1
class DisjointSet():
    def __init__(self, n):
        self.parent = [0] * n
        self.rank = [0] * n
        for i in range(0, n):
            self.parent[i] = i
        
    def find(self, i):
        if self.parent[i] != i:
            self.parent[i] = self.find(self.parent[i])
        return self.parent[i]
    
    def union(self, x, y):
        x_parent = self.find(x)
        y_parent = self.find(y)
        
        if x_parent == y_parent:
            return 0
        else:
            if self.rank[x_parent] > self.rank[y_parent]:
                self.parent[y_parent] = x_parent
            elif self.rank[y_parent] > self.rank[x_parent]:
                self.parent[x_parent] = y_parent
            else:
                self.parent[y_parent] = x_parent
                self.rank[x_parent] += 1
            return 1

class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        res = e1 = e2 = 0
        ds = DisjointSet(n)
        print((ds.parent))
        for t, u, v in edges:
            if t == 3:
                if ds.union(u-1,v-1):
                    e1 += 1
                    e2 += 1
                else:
                    res += 1
        print((ds.parent))
        
        tmp = copy.deepcopy(ds)
        for t, u, v in edges:
            if t == 1:
                if ds.union(u-1, v-1):
                    e1 += 1
                else:
                    res += 1
                    
        
        
        for t, u, v in edges:
            if t == 2:
                print()
                if tmp.union(u-1, v-1):
                    e2 += 1
                else:
                    res += 1
        
        return res if e1 == e2 == n - 1 else -1
                    
        
        

from copy import deepcopy
class UnionFind:
    def __init__(self, n):
        self.leaders = [i for i in range(n)]
        self.ranks = [1 for i in range(n)]
    
    def find(self, x):
        # p = x
        # while p != self._leaders[p]:
        #     p = self._leaders[p]
        # while x != p:
        #     self._leaders[x], x = p, self._leaders[x]
        # return p
        if self.leaders[x] != x:
            self.leaders[x] = self.find(self.leaders[x])
        return self.leaders[x]
    
    def union(self, x, y):
        p = self.find(x)
        q = self.find(y)
        if p == q: 
            return False
        if self.ranks[p] < self.ranks[q]:
            self.leaders[p] = q
        elif self.ranks[p] > self.ranks[q]:
            self.leaders[q] = p
        else:        
            self.leaders[q] = p
            self.ranks[p] += 1
        return True

class Solution:
    def maxNumEdgesToRemove(self, n, edges):
        res, cnt1 = 0, 0
        uf1 = UnionFind(n + 1)
        for g, u, v in edges:
            if g == 3:   
                if uf1.union(u, v):
                    cnt1 += 1
                else:
                    res += 1
        
        uf2 = deepcopy(uf1)
        cnt2 = cnt1
        for g, u, v in edges:
            if g == 1:   
                if uf1.union(u, v):
                    cnt1 += 1
                else:
                    res += 1

        for g, u, v in edges:
            if g == 2:   
                if uf2.union(u, v):
                    cnt2 += 1
                else:
                    res += 1
        
        if cnt1 != n - 1 or cnt2 != n - 1:
            return -1
        return res
class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        def find(x):
            if x != dsu[x]:
                dsu[x] = find(dsu[x])
            return dsu[x]

        def union(x, y):
            x, y = find(x), find(y)
            if x == y:
                return False
            dsu[x] = y
            return True

        res = type_1 = type_2 = 0
        dsu, type_edges = list(range(n + 1)), [[], [], [], []]
        for t, u, v in edges:
            type_edges[t].append([u, v])
        for u, v in type_edges[3]:
            if union(u, v):
                type_1 += 1
                type_2 += 1
            else:
                res += 1
        dsu_bak = dsu[:]
        for u, v in type_edges[1]:
            if union(u, v):
                type_1 += 1
            else:
                res += 1
        dsu = dsu_bak
        for u, v in type_edges[2]:
            if union(u, v):
                type_2 += 1
            else:
                res += 1
        return res if type_1 == type_2 == n - 1 else -1
class Solution:
    def maxNumEdgesToRemove(self, n: int, e: List[List[int]]) -> int:
        def union(UF, u, v):
            pu, pv = find(UF, u), find(UF, v)
            if pu != pv: UF[pv] = pu
        def find(UF, u):
            UF.setdefault(u, u)            
            if UF[u] != u: UF[u] = find(UF, UF[u])
            return UF[u]        
        def par_size(UF):
            return len(set(find(UF, u) for u in range(1, n+1)))        
        def check(UF, t):            
            UF = UF.copy()
            for tp, u, v in e:
                if tp != t: continue
                pu, pv = find(UF, u), find(UF, v)
                if pu == pv: self.ans += 1
                else: union(UF, u, v)
            return par_size(UF) == 1
        
        self.ans, UF = 0, {}        
        for t, u, v in e:
            if t != 3: continue
            pu, pv = find(UF, u), find(UF, v)
            if pu == pv: self.ans += 1
            else: union(UF, u, v)        
        if not check(UF, 1): return -1
        if not check(UF, 2): return -1        
        return self.ans                        
class Solution:
    def maxNumEdgesToRemove(self, n: int, e: List[List[int]]) -> int:
        def union(UF, u, v):
            # UF.setdefault(u, u); UF.setdefault(v, v)
            pu, pv = find(UF, u), find(UF, v)
            if pu != pv: UF[pv] = pu
        def find(UF, u):
            UF.setdefault(u, u)            
            if UF[u] != u: UF[u] = find(UF, UF[u])
            return UF[u]
        
        def par_size(UF):
            return len(set(find(UF, u) for u in range(1, n+1)))
        
        def check(UF, t):            
            UF = UF.copy()
            for tp, u, v in e:
                if tp != t: continue
                pu, pv = find(UF, u), find(UF, v)
                if pu == pv: self.ans += 1
                else: union(UF, u, v)
            return par_size(UF) == 1
        
        self.ans, UF = 0, {}        
        for t, u, v in e:
            if t != 3: continue
            pu, pv = find(UF, u), find(UF, v)
            if pu == pv: self.ans += 1
            else: union(UF, u, v)
        
        if not check(UF, 1): return -1
        if not check(UF, 2): return -1        
        return self.ans                        
class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        edges = sorted(edges, key=lambda l:l[0], reverse=True)
        uf_a = [i for i in range(n)]
        uf_b = [j for j in range(n)]
        
        cnt = 0        
        for edge in edges:
            if edge[0] == 3:
                cnt += self.union(uf_a, edge[1]-1, edge[2]-1)
                self.union(uf_b, edge[1]-1, edge[2]-1)
            elif edge[0] == 1:
                cnt += self.union(uf_a, edge[1]-1, edge[2]-1)
            else: # edge[0] == 2
                cnt += self.union(uf_b, edge[1]-1, edge[2]-1)
        if not self.connected(uf_a) or not self.connected(uf_b):
            return -1
        return len(edges)-cnt
    
    def connected(self, uf: List[int]) -> bool:
        r = self.root(uf, 0)
        for i in range(1, len(uf)):
            if self.root(uf, i) != r:
                return False
        return True
                
    def root(self, uf: List[int], a: int) -> int:
        cur = a
        while uf[cur] != cur:
            cur = uf[cur]
        root = cur
        while uf[a] != root:
            parent = uf[a]
            uf[a] = root
            a = parent
        return root
    
    def union(self, uf: List[int], a: int, b: int) -> int:
        root_a = self.root(uf, a)
        root_b = self.root(uf, b)
        if root_a == root_b:
            return 0
        small = min(root_a, root_b)
        large = max(root_a, root_b)
        uf[large] = small
        return 1

class DSU:
  def __init__(self, n):
    self.p = [-1]*(n+1)
    self.r = [0]*(n+1)
    
  def find_parent(self, x):
    if self.p[x]==-1:
      return x
    self.p[x] = self.find_parent(self.p[x]) # path compression
    return self.p[x]
  
  def union(self, a, b):
    pa = self.find_parent(a)
    pb = self.find_parent(b)
    if pa==pb: return False
    if self.r[pa]>self.r[pb]:
      self.p[pb] = pa     # here rank can be adding
    elif self.r[pa]<self.r[pb]:
      self.p[pa] = pb
    else:
      self.p[pa] = pb
      self.r[pb] += 1
      
    return True
  
class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
      edges = sorted(edges, key=lambda x: -x[0])
      dsu_alice = DSU(n)    # this can be done using on dsu and counting edges for alice and bob. if connected at the end, both tree should have n-1 edges
      dsu_bob = DSU(n)
      res = 0
      
      for e in edges:
        if e[0]==3:
          au = dsu_alice.union(e[1],e[2])
          bu = dsu_bob.union(e[1],e[2])
          if not au and not bu:
            res += 1
        elif e[0]==1:
          if not dsu_alice.union(e[1],e[2]):
            res += 1
        else:
          if not dsu_bob.union(e[1],e[2]):
            res += 1
        # print (e, res) 
      
      ap = 0
      bp = 0
      for i in range(1, n+1):
        if ap and dsu_alice.find_parent(i)!=ap:
          return -1
        else: ap = dsu_alice.find_parent(i)
        if bp and dsu_bob.find_parent(i)!=bp:
          return -1
        else: bp = dsu_bob.find_parent(i)
      return res

class Solution:
    def maxNumEdgesToRemove(self, n: int, e: List[List[int]]) -> int:
        def union(UF, u, v):
            UF.setdefault(u, u); UF.setdefault(v, v)
            pu, pv = find(UF, u), find(UF, v)
            if pu != pv: UF[pv] = pu
        def find(UF, u):
            UF.setdefault(u, u)            
            if UF[u] != u: UF[u] = find(UF, UF[u])
            return UF[u]
        
        def par_size(UF):
            return len(set(find(UF, u) for u in range(1, n+1)))
        
        def check(UF, t):            
            UF = UF.copy()
            for tp, u, v in e:
                if tp != t: continue
                pu, pv = find(UF, u), find(UF, v)
                if pu == pv: self.ans += 1
                else: union(UF, u, v)
            return par_size(UF) == 1
        
        self.ans, UF = 0, {}        
        for t, u, v in e:
            if t != 3: continue
            pu, pv = find(UF, u), find(UF, v)
            if pu == pv: self.ans += 1
            else: union(UF, u, v)
        
        if not check(UF, 1): return -1
        if not check(UF, 2): return -1        
        return self.ans                        
class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        def find(p, u):
            parent = p[u]
            if parent == u:
                return parent
            
            p[u] = find(p, parent)
            return p[u]
        
        def union(p, rank, root_u, root_v):
            if rank[root_u] < rank[root_v]:
                p[root_u] = root_v
            elif rank[root_v] < rank[root_u]:
                p[root_v] = root_u
            else:
                p[root_u] = root_v
                rank[root_v] += 1
        
        p = list(range(n))
        rank = [0] * n
        
        full_edges = set()
        partial_edges = set()
        partial_adj = {}
        partial_adj[1] = collections.defaultdict(set)
        partial_adj[2] = collections.defaultdict(set)
        for e in edges:
            edge_type, u, v = e[0], e[1] - 1, e[2] - 1
            if edge_type == 3:
                full_edges.add((u, v))
            else:
                partial_edges.add((edge_type, u, v))
                partial_adj[edge_type][u].add(v)
                partial_adj[edge_type][v].add(u)

        nb_edges_in_mst = 0
        for e in full_edges:
            u, v = e
            root_u, root_v = find(p, u), find(p, v)
            if root_u != root_v:
                union(p, rank, root_u, root_v)
                nb_edges_in_mst += 1
        
        for e in partial_edges:
            edge_type, v0, v1 = e
            if find(p, v0) == find(p, v1):
                continue

            # We have two nodes u and v, u in a fully-connected component A, v in another
            # fully-connected component B. A disjoint from B and u is partially connected
            # to v via an edge of `edge_type`. Since we need to reach v from u by both
            # Alice and Bob, if we can find another node, x, in A that is partially-connected 
            # to v by an edge of `needed_edge_type`, then we have edges of both types that
            # we can use to reach v from u. (use `edge_type` from u->v, or use `needed_edge_type`
            # from u->x->v). Since the situation is symmetric, we'll need to test with roles of
            # u and v swapped
            needed_edge_type = 2 if edge_type == 1 else 2
            for pair in [(v0, v1), (v1, v0)]:
                u, v = pair
                root_x = None
                for x in partial_adj[needed_edge_type][v]:
                    if find(p, x) == find(p, u):
                        # We've found a node x in A fully connected to u, AND it's partially connected
                        # to v via the `needed_edge_type`
                        root_x = find(p, x)
                        break
                if root_x != None:
                    root_u = find(p, u)
                    root_v = find(p, v)
                    union(p, rank, root_x, root_v)
                    union(p, rank, root_u, root_v)
                    nb_edges_in_mst += 2
                    break

        uniq_roots = set()
        for u in range(len(p)):
            uniq_roots.add(find(p, u))
        if len(uniq_roots) != 1:
            return -1  
        
        return len(edges) - nb_edges_in_mst
            
                
            
        

class UnionFind:
    def __init__(self, n):
        self.parents = [i for i in range(n)]
        
    def union(self, index1, index2):
        root1 = self.find(index1)
        root2 = self.find(index2)
        if root1 == root2:
            return 0
        self.parents[root2] = root1
        return 1
    
    def find(self, index):
        if self.parents[index] != index:
            self.parents[index] = self.find(self.parents[index])
        return self.parents[index]

class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        uf1, uf2 = UnionFind(n), UnionFind(n)
        count_delete = 0
        count_union1, count_union2 = 0, 0
        
        for i, (weight, node1, node2) in enumerate(edges):
            if weight == 3:
                is_union = uf1.union(node1 - 1, node2 - 1)
                # is_union2 = uf2.union(node1 - 1, node2 - 1)
                # is_union = is_union1 and is_union2
                count_union1 += is_union
                count_union2 += is_union
                count_delete += 1 - is_union
        uf2.parents = copy.deepcopy(uf1.parents)
        count_union2 = count_union1
        
        for i, (weight, node1, node2) in enumerate(edges):
            if weight == 1:
                is_union = uf1.union(node1 - 1, node2 - 1)
                count_union1 += is_union
                count_delete += 1 - is_union
            elif weight == 2:
                is_union = uf2.union(node1 - 1, node2 - 1)
                count_union2 += is_union
                count_delete += 1 - is_union

        if count_union1 != n - 1 or count_union2 != n - 1:
            return -1
        return count_delete
                
            

class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        
        d = [[], [], []]
        for type, u, v in edges:
            d[type - 1].append((u - 1, v - 1))
            
        parent = list(range(n))
        def find(x, l):
            if x != l[x]:
                l[x] = find(l[x], l)
            return l[x]
        
        cnt = 0
        for u, v in d[-1]:
            ru, rv = find(u, parent), find(v, parent)
            if ru != rv:
                parent[ru] = rv
                cnt += 1
                
        alice = [num for num in parent]
        for u, v in d[0]:
            ru, rv = find(u, alice), find(v, alice)
            if ru != rv:
                alice[ru] = rv
                cnt += 1
                
        ra = find(0, alice)
        for i in range(n):
            if find(i, alice) != ra:
                return -1
                
        bob = [num for num in parent]
        for u, v in d[1]:
            ru, rv = find(u, bob), find(v, bob)
            if ru != rv:
                bob[ru] = rv
                cnt += 1
                
        rb = find(0, bob)
        for i in range(n):
            if find(i, bob) != rb:
                return -1
            
        return len(edges) - cnt
import bisect
import functools
from typing import List


class Solution:
  def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:

    result = 0
    e = [[_, u-1, v -1] for _, u, v in edges]
    a = [i for i in range(n)]
    b = [i for i in range(n)]
    def find(p, x):
      p[x] = p[x] if p[x] == x else find(p, p[x])
      return p[x]

    def union(p, a, b):
      find_a = find(p, a)
      find_b = find(p, b)
      if find_a == find_b:
        return 1
      p[find_a] = find_b
      return 0

    same = 0
    for type, u, v in e:
      if type == 3:
        same += union(a, u, v) | union(b, u, v)

    for type, u, v in e:
      if type == 1:
        same += union(a, u, v)
      if type == 2:
        same += union(b, u, v)

    all_a = all(find(a, 0) == find(a, x) for x in a)
    all_b = all(find(b, 0) == find(b, x) for x in b)
    if all_a and all_b:
      return same

    return -1
class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        f = {}
        
        def find(x):
            f.setdefault(x,x)
            if x!=f[x]:
                f[x]= find(f[x])
            return f[x]
        
        def union(x,y):
            x = find(x)
            y = find(y)
            if x==y:
                return False
            f[x]=y
            return True
        
        res, e1, e2 = 0,0,0
        
        for t,u,v in edges:
            if t==3:
                if union(u,v):
                    e1+=1
                    e2+=1
                else:
                    res+=1
                    
        copy_f = f.copy()
        for t,u,v in edges:
            if t==1:
                if union(u,v):
                    e1+=1
                else:
                    res+=1
                    
        f = copy_f
        for t,u,v in edges:
            if t==2:
                if union(u,v):
                    e2+=1
                else:
                    res+=1
        
        return res if e1==e2==n-1 else -1
class Solution:
    def maxNumEdgesToRemove(self, n: int, e: List[List[int]]) -> int:
        def union(UF, u, v):
            pu, pv = find(UF, u), find(UF, v)
            if pu != pv: UF[pv] = pu
        def find(UF, u):
            if UF[u] != u: UF[u] = find(UF, UF[u])
            return UF[u]         
        def check(UF, t):            
            UF = UF.copy()
            for tp, u, v in e:
                if tp != t: continue
                pu, pv = find(UF, u), find(UF, v)
                if pu == pv: self.ans += 1
                else: union(UF, u, v)
            return len(set(find(UF, u) for u in UF)) == 1
        
        self.ans, UF = 0, {k: k for k in range(1, n+1)}        
        for t, u, v in e:
            if t != 3: continue
            pu, pv = find(UF, u), find(UF, v)
            if pu == pv: self.ans += 1
            else: union(UF, u, v)        
        if not check(UF, 1): return -1
        if not check(UF, 2): return -1        
        return self.ans                        
class DSU:
    def __init__(self, N):
        self.par = list(range(N))
        self.rnk = [0] * N
        self.count = N
        
    def find(self, x):
        if self.par[x] != x:
            self.par[x] = self.find(self.par[x])
        return self.par[x]
    
    def union(self, x, y):
        xr, yr = list(map(self.find, (x, y)))
        if xr == yr: return False
        self.count -= 1
        if self.rnk[xr] < self.rnk[yr]:
            xr, yr = yr, xr
        if self.rnk[xr] == self.rnk[yr]:
            self.rnk[xr] += 1
        self.par[yr] = xr
        return True

class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        
        uf_a, uf_b = DSU(n), DSU(n)
        
        ans = 0
        edges.sort(reverse=True)
        for t, u, v in edges:
            u, v = u - 1, v - 1
            if t == 3:
                ans += not (uf_a.union(u, v) and uf_b.union(u, v))
            elif t == 2:
                ans += not uf_b.union(u, v)
            else:
                ans += not uf_a.union(u, v)
                
        return ans if uf_a.count == 1 and uf_b.count == 1 else -1

from copy import deepcopy
from collections import Counter
class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        
        edges.sort(reverse = True)
        
        graph = [i for i in range(n)]
        
        def p(i,graph):
            level = 0
            while i != graph[i]:
                i = graph[i]
                level += 1
            return i,level
        
        i = 0
        res = 0
        g = graph
        graph_a = graph_b = None
        
        while i < len(edges):
            t,a,b = edges[i]
            if t == 2 and (i == 0 or edges[i-1][0] != 2):
                graph_b = deepcopy(graph)
                g = graph_b
            if t == 1 and (i == 0 or edges[i-1][0] != 1):
                graph_a = deepcopy(graph)
                g = graph_a
            ap,al = p(a-1,g)
            bp,bl = p(b-1,g)
            if ap == bp: res += 1
            elif ap == a-1: g[ap] = bp
            elif bp == b-1: g[bp] = ap
            elif al < bl: g[ap] = bp
            else: g[bp] = ap
            i += 1
            
        if not graph_a:
            graph_a = deepcopy(graph)
        if not graph_b:
            graph_b = deepcopy(graph)
        
        for i in range(n):
            graph_a[i] = p(i,graph_a)[0]
            graph_b[i] = p(i,graph_b)[0]
            
        a = Counter(graph_a)
        b = Counter(graph_b)
        if len(a) > 1 or len(b) > 1: return -1
        return res
class DSU:
    def __init__(self, N):
        self.parent = list(range(N+1))
        self.edges = 0

    def find(self, x):
        if x != self.parent[x]:
            # u8defu5f84u5b8cu5168u538bu7f29
            self.parent[x] = self.find(self.parent[x])

        return self.parent[x]

    def union(self, x, y):
        root1 = self.find(x)
        root2 = self.find(y)
        # u8fd9u4e00u53e5u975eu5fc5u987buff0cu53eau662fu6ee1u8db3u6b64u9898 u9700u8981u5220u9664u6b64u6761u8fb9
        if root1 == root2:
            return 1
        
        self.parent[root2] = root1
        self.edges += 1
        
        return 0


class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        
        ans = 0
        
        A = DSU(n)
        B = DSU(n)
        
        for t, x, y in edges:
            if t != 3:
                continue
            ans += A.union(x, y)
            B.union(x, y)
        
        for t, x, y in edges:
            if t == 3:
                continue
            # print(t, x, y)
            d = A if t == 1 else B
            ans += d.union(x, y)
            
        return ans if A.edges == n - 1 and B.edges == n - 1 else -1
            

class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        res, e0, e1 = 0, 0, 0
        self.root = [i for i in range(n+1)]
        def find(x):
            if x != self.root[x]:
                self.root[x] = find(self.root[x])
            return self.root[x]
        
        def uni(x, y):
            x, y = find(x), find(y)
            if x==y:
                return 1
            self.root[x] = y
            return 0
        
        for t,i,j in edges:
            if t == 3:
                if uni(i,j):
                    res += 1
                else:
                    e0 += 1
                    e1 += 1
                    
        root0 = self.root[:]
        for t,i,j in edges:
            if t == 1:
                if uni(i, j):
                    res += 1
                else:
                    e0 += 1
                    
        self.root = root0
        for t,i,j in edges:
            if t==2:
                if uni(i,j):
                    res += 1
                else:
                    e1 += 1
                    
        return res if e1 == e0 == (n-1) else -1
                    

class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        # build graph, use type 3 first
        # then do alice and bob separately
        # have dsu parents to build up alice and bob
        
        parentsA = list(range(n))
        parentsB = list(range(n))
        
        def find(parents, a):
            while a != parents[a]:
                parents[a] = parents[parents[a]]
                a = parents[a]
            return a
            
        def union(parents, a, b):
            a = find(parents, a)
            b = find(parents, b)
            parents[a] = b
        
        typeA, typeB = [], []
        tree1, tree2, res = 0,0,0
        
        for t, u, v in edges:
            u, v = u-1, v-1 # make zero indexed, easier for UF
            if t == 3:
                if find(parentsA, u) != find(parentsA, v):
                    tree1 += 1
                    tree2 += 1
                    union(parentsA, u, v)
                    union(parentsB, u, v)
                else:
                    res += 1
            elif t == 2:
                typeB.append((u, v))
            elif t == 1:
                typeA.append((u, v))
                
        # now do type1 and 2 separately
        for u,v in typeA:
            if find(parentsA, u) != find(parentsA, v):
                tree1 += 1
                union(parentsA, u, v)
            else:
                res += 1
        
        for u,v in typeB:
            if find(parentsB, u) != find(parentsB, v):
                tree2 += 1
                union(parentsB, u, v)
            else:
                res += 1
        
        if tree1 == n-1 and tree2 == n-1:
            return res
        else:
            return -1
class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        class DSU:
            def __init__(self, n):
                self.edge_count = 0
                self.parent = [i for i in range(n + 1)]

            def find(self, x):
                if x != self.parent[x]:
                    self.parent[x] = self.find(self.parent[x])
                return self.parent[x]

            def union(self, u, v):
                pu, pv = self.find(u), self.find(v)
                if pu != pv:
                    self.parent[pv] = pu
                    self.edge_count += 1
                    return 0
                return 1
            
            def get_edge_count(self):
                return self.edge_count
        
        A, B = DSU(n), DSU(n)

        ans = 0
        for typ, u, v in edges:
            if typ != 3: continue
            ans += A.union(u, v)
            B.union(u, v)
        # print(A.get_edge_count(), B.get_edge_count())    
        for typ, u, v in edges:
            if typ == 3: continue
            if typ == 1:
                ans += A.union(u, v)
            elif typ == 2:
                ans += B.union(u, v)
        # print(A.get_edge_count(), B.get_edge_count())
        if A.get_edge_count() == n - 1 and B.get_edge_count() == n - 1:
            return ans
        else:
            return -1
class Solution:
    def add_edge(self,parent,cnt,x,y):
        xx = self.parents(parent,x)
        yy = self.parents(parent,y)
        if(cnt[xx] < cnt[yy]):
            parent[xx]=yy
            cnt[yy]+=1
        else:
            parent[yy]=xx
            cnt[xx]+=1
    def parents(self,parent,ch):
        if(parent[ch]==ch):
            return ch
        else:
            xy = self.parents(parent,parent[ch])
            parent[ch]=xy
            return xy
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        parent = (n+1)*[0]
        ans=0
        cnt = (n+1)*[0]
        for i in range(n+1):
            parent[i]=i
        edges.sort(reverse= True)
        for j in range(len(edges)):
            i = edges[j]
            if(i[0]!=3):
                break
            if(self.parents(parent,i[1])!=self.parents(parent,i[2])):
                self.add_edge(parent,cnt,i[1],i[2])
            else:
                ans+=1
        bob = parent.copy()
        bob_cnt = cnt.copy()
        for k in range(j,len(edges)):
            i = edges[k]
            if(i[0]!=2):
                break
            if(self.parents(bob,i[1])!=self.parents(bob,i[2])):
                self.add_edge(bob,bob_cnt,i[1],i[2])
            else:
                ans+=1
        for l in range(k,len(edges)):
            i = edges[l]
            if(i[0]!=1):
                break
            if(self.parents(parent,i[1])!=self.parents(parent,i[2])):
                self.add_edge(parent,cnt,i[1],i[2])
            else:
                ans+=1
        rn = 0
        for i in range(1,n+1):
            if(parent[i]==i):
                rn+=1
        if(rn>1):
            return -1
        rn = 0
        for i in range(1,n+1):
            if(bob[i]==i):
                rn+=1
        if(rn>1):
            return -1
        return ans
from collections import defaultdict

class UnionFind:
    
    def __init__(self, n):
        self._id = list(range(n))
        self._sz = [1] * n
        self.cc = n  # connected components

    def _root(self, i):
        j = i
        while (j != self._id[j]):
            self._id[j] = self._id[self._id[j]]
            j = self._id[j]
        return j

    def find(self, p, q):
        return self._root(p) == self._root(q)

    def union(self, p, q):
        i = self._root(p)
        j = self._root(q)
        if i == j:
            return
        if (self._sz[i] < self._sz[j]):
            self._id[i] = j
            self._sz[j] += self._sz[i]
        else:
            self._id[j] = i
            self._sz[i] += self._sz[j]
        self.cc -= 1
        
class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        
        graph=defaultdict(list)
        class1=UnionFind(n)
        class2=UnionFind(n)
        
        for cat, start, end in edges:
            graph[cat].append((start-1,end-1))
            
        class1=UnionFind(n)
        class2=UnionFind(n)
        
        
        ans=0
      
        for start,end in graph[3]:
            cur1,cur2=class1.cc,class2.cc
            class1.union(start,end)
            class2.union(start,end)
            if class1.cc==cur1 and class2.cc==cur2:
                ans+=1
            
        for start,end in graph[1]:
            cur1=class1.cc
            class1.union(start,end)
            if class1.cc==cur1:
                ans+=1
                
        for start,end in graph[2]:
            cur2=class2.cc
            class2.union(start,end)
            if class2.cc==cur2:
                ans+=1  
                
        if class1.cc==1 and class2.cc==1:
            return ans
        else:
            return -1
    
    
    
            

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [1] * n
        self.count = n

    def find(self, i):
        if self.parent[i] != i:
            self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, x, y):
        px = self.find(x)
        py = self.find(y)
        if px == py:
            return False
        if self.rank[px] > self.rank[py]:
            px, py = py, px
        self.parent[px] = py
        if self.rank[px] == self.rank[py]:
            self.rank[py] += 1
        self.count -= 1
        return True

    def united(self):
        return self.count == 1

    
class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        edges.sort(key=lambda x:-x[0])
        alice = UnionFind(n)
        bob = UnionFind(n)
        added = 0
        for t, a, b in edges:
            a -= 1
            b -= 1
            if t == 3:
                added += 1 if alice.union(a, b) | bob.union(a, b) else 0
            elif t == 1:
                added += 1 if alice.union(a, b) else 0
            elif t == 2:
                added += 1 if bob.union(a, b) else 0

        return len(edges) - added if alice.united() and bob.united() else -1

import copy

class DJ_DS():
    def __init__(self, n):
        self.n = n
        self.parent = [i for i in range(n)]
        self.rank = [0 for i in range(n)]
        self.nb_edges = 0
    
    def find_parent(self,i): # faster with path compression
        while self.parent[i] != i:
            i = self.parent[i]
        return i
        
    def union(self,i,j):
        p_i = self.find_parent(i)
        p_j = self.find_parent(j)
        
        if p_i != p_j:
            self.nb_edges += 1
            if self.rank[p_i] < self.rank[p_j]:
                self.parent[p_i] = p_j
            else:
                self.parent[p_j] = p_i
                if self.rank[p_i] == self.rank[p_j]:
                    self.rank[p_i] += 1
                
    def perform_merge(self, edges):
        for [u,v] in edges:
            self.union(u,v)
            

class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        nb_edges = len(edges)
        # list of edges of each color individually
        type1, type2, type3 = [], [], []
        for [t,u,v] in edges:
            if t == 1:
                type1.append([u-1,v-1])
            elif t == 2:
                type2.append([u-1,v-1])
            else:
                type3.append([u-1,v-1])
        
        # Count nb_edges with type 3 only in max forest
        dj_3 = DJ_DS(n)
        dj_3.perform_merge(type3)
        sol_3 = dj_3.nb_edges
        dj_1 = copy.deepcopy(dj_3)
        dj_2 = copy.deepcopy(dj_3)
        
        # From type 3 forest add edges from type 1 to see if spanning tree, if not return -1
        dj_1.perform_merge(type1)
        if dj_1.nb_edges < n-1:
            return -1
        
        # From type 3 forest add edges from type 2 to see if spanning tree, if not return -1
        dj_2.perform_merge(type2)
        if dj_2.nb_edges < n-1:
            return -1
        
        return (nb_edges - (sol_3 + 2 * (n-1 - sol_3)))
class UnionFind:
    def __init__(self, n):
        self.parents = [i for i in range(n)]
        
    def union(self, index1, index2):
        root1 = self.find(index1)
        root2 = self.find(index2)
        if root1 == root2:
            return 0
        self.parents[root2] = root1
        return 1
    
    def find(self, index):
        if self.parents[index] != index:
            self.parents[index] = self.find(self.parents[index])
        return self.parents[index]

class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        uf1, uf2 = UnionFind(n), UnionFind(n)
        count_delete = 0
        count_union1, count_union2 = 0, 0
        
        for i, (weight, node1, node2) in enumerate(edges):
            if weight == 3:
                is_union1 = uf1.union(node1 - 1, node2 - 1)
                is_union2 = uf2.union(node1 - 1, node2 - 1)
                is_union = is_union1 and is_union2
                
                count_union1 += is_union1
                count_union2 += is_union2
                count_delete += 1 - is_union
        
        for i, (weight, node1, node2) in enumerate(edges):
            if weight == 1:
                is_union = uf1.union(node1 - 1, node2 - 1)
                count_union1 += is_union
                count_delete += 1 - is_union
            elif weight == 2:
                is_union = uf2.union(node1 - 1, node2 - 1)
                count_union2 += is_union
                count_delete += 1 - is_union

        if count_union1 != n - 1 or count_union2 != n - 1:
            return -1
        return count_delete
                
            

class UF:
    def __init__(self, n):
        self.count = n
        self.parents = [0]*(n+1)
        for i in range(1+n):
            self.parents[i]=i
    def find(self, x):
        if x != self.parents[x]:
            self.parents[x] = self.find(self.parents[x])
        return self.parents[x]
    def union(self,x,y):
        p_x = self.find(x)
        p_y = self.find(y)
        if p_x == p_y:
            return False
        self.parents[p_x] = p_y
        self.count -= 1
        return True
class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        ufa = UF(n)
        ufb = UF(n)
        res = 0
        for t, u, v in edges:
            if t == 3:
                flag1 = ufa.union(u,v)
                flag2 = ufb.union(u,v)
                if not flag1 and not flag2:
                    res += 1
        for t, u, v in edges:
            if t ==1:
                if not ufa.union(u, v):
                    res += 1
            if t ==2:
                if not ufb.union(u, v):
                    res += 1
        print((ufa.count, ufb.count))
        if ufa.count != 1 or ufb.count != 1:
            return -1
        return res
                    

class DSU:
    def __init__(self, N):
        self.parent = list(range(N))
        self.size = [1] * N

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        xr = self.find(x)
        yr = self.find(y)
        
        if xr == yr: return
        
        if self.size[xr] >= self.size[yr]:
            self.size[xr] += self.size[yr]
            self.size[yr] = self.size[xr]
            self.parent[yr] = xr
        else:
            self.size[yr] += self.size[xr]
            self.size[xr] = self.size[yr]
            self.parent[xr] = yr


class Solution(object):
    
    def dfs(self, u, allowed_types, visited, graph):
        visited.add(u)
        for v in graph[u]:
            if v not in visited and (allowed_types[0] in graph[u][v] or allowed_types[1] in graph[u][v]):
                self.dfs(v, allowed_types, visited, graph)
            
    
    def maxNumEdgesToRemove(self, N, edges):
        """
        :type n: int
        :type edges: List[List[int]]
        :rtype: int
        """
        dsu_alice = DSU(N)
        dsu_bob = DSU(N)
        count_edges = 0
        
        for t, u, v in edges:
            u -= 1
            v -= 1
            if t == 3:
                pu, pv = dsu_bob.find(u), dsu_bob.find(v)
                if pu != pv:
                    dsu_alice.union(u, v)
                    dsu_bob.union(u, v)
                    count_edges += 1
        
        for t, u, v in edges:
            u -= 1
            v -= 1
            if t == 1:
                pu, pv = dsu_alice.find(u), dsu_alice.find(v)
                if pu != pv:
                    dsu_alice.union(u, v)
                    count_edges += 1
            elif t == 2:
                pu, pv = dsu_bob.find(u), dsu_bob.find(v)
                if pu != pv:
                    dsu_bob.union(u, v)
                    count_edges += 1
                    
        try:
            dsu_bob.size.index(N)
            dsu_alice.size.index(N)
        except:
            return -1
        
        return len(edges) -  count_edges
        
        
class DisjointSet:
    def __init__(self, number_of_sites):
        self.parent = [i for i in range(number_of_sites+1)]
        self.children_site_count = [1 for _ in range(number_of_sites+1)]
        self.component_count = number_of_sites

    def find_root(self, site):
        root = site
        while root != self.parent[root]:
            root = self.parent[root]
        while site != root:
            site, self.parent[site] = self.parent[site], root
        return root

    def is_connected(self, site_1, site_2):
        return self.find_root(site_1) == self.find_root(site_2)

    def union(self, site_1, site_2):
        site_1_root = self.find_root(site_1)
        site_2_root = self.find_root(site_2)
        if site_1_root == site_2_root:
            return False

        if self.children_site_count[site_1_root] < self.children_site_count[site_2_root]:
            self.parent[site_1_root] = site_2_root
            self.children_site_count[site_2_root] += self.children_site_count[
                site_1_root]
        else:
            self.parent[site_2_root] = site_1_root
            self.children_site_count[site_1_root] += self.children_site_count[
                site_2_root]
        self.component_count -= 1
        return True


class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        alice_disjoint_set = DisjointSet(n)
        bob_disjoint_set = DisjointSet(n)

        TYPE_OF_COMMON_EDGES = 3
        TYPE_OF_ALICE_EDGES = 1
        TYPE_OF_BOB_EDGES = 2

        common_edges = filter(lambda edge: edge[0] == TYPE_OF_COMMON_EDGES, edges)
        alice_edges = filter(lambda edge: edge[0] == TYPE_OF_ALICE_EDGES, edges)
        bob_edges = filter(lambda edge: edge[0] == TYPE_OF_BOB_EDGES, edges)

        redundant = 0
        for _, u, v in common_edges:
            if (not alice_disjoint_set.union(u, v)) or (not bob_disjoint_set.union(u, v)):
                redundant += 1

        for _, u, v in bob_edges:
            if not bob_disjoint_set.union(u,v):
                redundant += 1
                
        for _, u, v in alice_edges:
            if not alice_disjoint_set.union(u, v):
                redundant += 1
        
        return redundant if alice_disjoint_set.component_count == 1 and bob_disjoint_set.component_count == 1 else -1
