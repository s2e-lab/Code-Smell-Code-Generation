class Solution:
    def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
        pr=[i for i in range(len(s))]
        def union(x,y):
            p1=find(x)
            p2=find(y)
            if p1!=p2:
                pr[p1]=p2
        def find(x):
            while pr[x]!=x:
                pr[x]=pr[pr[x]]
                x=pr[x]
            return x
        
        for i in pairs:
            union(i[0],i[1])
            
        from collections import defaultdict
        dp=defaultdict(list)
        for i in range(len(s)):
            ld=find(i)
            dp[ld].append(i)
        ans=[0]*len(s)
        for i in dp:
            dp[i].sort()
            st=''
            for j in dp[i]:
                st+=s[j]
            st=sorted(st)
            c=0
            for j in dp[i]:
                ans[j]=st[c]
                c+=1
        return ''.join(ans)
        
        
        
        
        
        
                

class Solution:
    def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
        n = len(s)
        
        parents = [i for i in range(n)]

        def find_parent(i):
            while parents[i] != i:
                # path compression
                parents[i] = parents[parents[i]]
                i = parents[i]
            return i

        def do_union(p, q):
            i = find_parent(p)
            j = find_parent(q)
            parents[i] = j

        for p, q in pairs:
            do_union(p, q)
       
        groups = defaultdict(list)
        for i in range(len(s)):
            root = find_parent(i)
            groups[root].append(i)
        
        s = list(s)
        for indices in sorted(groups.values()):
            vals = sorted([s[i] for i in indices])
            for i, c in enumerate(vals):
                s[indices[i]] = c
            
        return ''.join(s)
class UnionFind:
    def __init__(self, N):
        self.par = list(range(N))
        self.rank = [0]*N
        
    def find(self, x):
        if self.par[x] != x:
            self.par[x] = self.find(self.par[x])
        return self.par[x]
    
    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px==py: return
        if self.rank[px]<self.rank[py]:
            self.par[px] = py
        elif self.rank[px]>self.rank[py]:
            self.par[py] = px
        else:
            self.par[py] = px
            self.rank[px] += 1

class Solution:
    def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
        N = len(s)
        uf = UnionFind(N)
        res = ['']*N
        indices = defaultdict(list)
        
        for u, v in pairs:
            uf.union(u, v)
        
        for i in range(N):
            indices[uf.find(i)].append(i)
        
        for i in indices:
            chars = sorted([s[j] for j in indices[i]])
            for j in range(len(indices[i])):
                res[indices[i][j]] = chars[j]

        return ''.join(res)
class Solution:
    def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
        self.parent = {i: i for i in range(len(s))}
        
        for pos1, pos2 in pairs:
            self.merge(pos1, pos2)
        
        connected_component = defaultdict(list)
        for i in range(len(s)):
            connected_component[self.find(i)].append(s[i])
        for k in connected_component.keys():
            connected_component[k].sort(reverse=True)
            
        res = []
        for i in range(len(s)):
            res.append(connected_component[self.find(i)].pop())
        
        return ''.join(res)
    
    
    def find(self, e):
        while e != self.parent[e]:
            self.parent[e] = self.parent[self.parent[e]]
            e = self.parent[e]
        return e
    
    def merge(self, e1, e2):
        root1, root2 = self.find(e1), self.find(e2)
        if root1 != root2:
            self.parent[root1] = root2
        
        return 
class Solution:
    def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
        n = len(s)
        parent = [i for i in range(n)]
        group = collections.defaultdict(list)
        res = []
        def find(a):
            p = parent[a]
            while p!= parent[p]:
                p = parent[p]
            parent[a] = p
            return p
        def union(a,b):
            pa,pb = find(a),find(b)
            if pa!=pb:
                parent[pa] = pb
        for u,v in pairs:
            union(u,v)
        for i in range(n):
            group[find(i)].append(s[i])
        for k in group:
            group[k].sort(reverse=True)
        for i in range(n):
            res.append(group[find(i)].pop())
        return ''.join(res)
import heapq
from copy import copy, deepcopy
class MaxHeapObj(object):
    def __init__(self, val): self.val = val
    def __lt__(self, other): return self.val > other.val
    def __eq__(self, other): return self.val == other.val
    def __str__(self): return str(self.val)
 
class MinHeap(object):
    def __init__(self, arr=[]):
        self.h = deepcopy(arr)
        heapq.heapify(self.h)
 
    def heappush(self, x): heapq.heappush(self.h, x)
    def heappop(self): return heapq.heappop(self.h)
    def __getitem__(self, i): return self.h[i]
    def __len__(self): return len(self.h)
 
class MaxHeap(object):
    def __init__(self, arr=[]):
        self.h = [MaxHeapObj(x) for x in arr]
        heapq.heapify(self.h)
 
    def heappush(self, x): heapq.heappush(self.h, MaxHeapObj(x))
    def heappop(self): return heapq.heappop(self.h).val
    def __getitem__(self, i): return self.h[i].val
    def __len__(self): return len(self.h)

class Solution:
    def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
        
        res, n, adj, component_num, curr, comp_vals = [], len(s), defaultdict(list), defaultdict(lambda: -1), 0, defaultdict(lambda: MinHeap())
        for edge in pairs:
            adj[edge[0]].append(edge[1])
            adj[edge[1]].append(edge[0])
        
        def DFS(i, comp):
            
            component_num[i] = comp 
            for j in adj[i]:
                if component_num[j] == -1: DFS(j, comp)
        
        for i in range(n):
            if component_num[i] == -1: 
                DFS(i, curr)
                curr += 1 
            comp_vals[component_num[i]].heappush(s[i])
        
        for i in range(n):
            res.append(comp_vals[component_num[i]].heappop())
        
        return "".join(res)
class Solution:
    def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
        def dfs(i):
            visited[i] = True
            component.append(i)
            for adj in adj_list[i]:
                if not visited[adj]:
                    dfs(adj)
        
        adj_list = collections.defaultdict(list)
        visited = [False] * len(s)
        for pair in pairs:
            adj_list[pair[0]].append(pair[1])
            adj_list[pair[1]].append(pair[0])
            
        ans = list(s)
    
        for index in range(len(s)):
            if not visited[index]:
                component = []
                dfs(index)
                component.sort()
                lst_chars = [ans[i] for i in component]
                lst_chars.sort()
                for i in range(len(lst_chars)): ans[component[i]] = lst_chars[i]
        
        return "".join(ans)
                
        
        
            
class DSU:
    def __init__(self,n):
        self.parent = [i for i in range(n)]
        self.rank = [1 for i in range(n)]
        
    def find(self,x):
        if x!=self.parent[x]:
            self.parent[x]=self.find(self.parent[x])
        return self.parent[x]

    def union(self,x,y):
        px = self.find(x)
        py = self.find(y)
        if px==py:
            return
        if self.rank[px]>self.rank[py]:
            self.parent[py] = px
            self.rank[px]+=self.rank[py]
        else:
            self.parent[px]=py
            self.rank[py]+=self.rank[x]
        
class Solution:
    def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
        n = len(s)
        dsu = DSU(n)
        for x,y in pairs:
            dsu.union(x,y)
        # Club everyone with the same parent
        hm = collections.defaultdict(list)
        for i in range(n):
            hm[dsu.find(i)].append(s[i])
        for key in hm:
            hm[key].sort()
        res = []
        for i in range(n):
            res.append(hm[dsu.find(i)].pop(0))
        return "".join(res)
            
        
from typing import List
from collections import defaultdict


class DisjointSet:
    def __init__(self, n):
        self.makeSet(n)

    def makeSet(self, n):
        self.parent = [i for i in range(n)]

    def union(self, i, j):
        self.parent[self.find(i)] = self.find(j)

    def find(self, i):
        if self.parent[i] != i:
            self.parent[i] = self.find(self.parent[i])
        return self.parent[i]


class Solution:
    def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
        sa = list(s)
        lens = len(s)
        ds = DisjointSet(lens)
        for pair in pairs:
            i = pair[0]
            j = pair[1]
            ip = ds.find(i)
            jp = ds.find(j)
            if ip != jp:
                ds.union(i, j)
        cm = defaultdict(lambda: [])
        for i in range(lens):
            ip = ds.find(i)
            cm[ip].append((i, sa[i]))
        for _, li in list(cm.items()):
            if li:
                lsv = sorted(li, key=lambda t: t[1])
                for i in range(len(li)):
                    sa[li[i][0]] = lsv[i][1]

        return ''.join(sa)

class Solution:
    def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
        if len(s) < 2 or len(pairs) == 0:
            return s
        
        class UnionFind:
            def __init__(self, N):
                self.arr = [i for i in range(N)]
                
            def find(self, x):
                if self.arr[x] == x:
                    return x
                else:
                    self.arr[x] = self.find(self.arr[x])
                    return self.arr[x]
            
            def union(self, x1, x2):
                self.arr[self.find(x1)] = self.find(x2)
            
        uf = UnionFind(len(s))
        for pair in pairs:
            uf.union(pair[0], pair[1])
        
        g = defaultdict(list)
        for i in range(len(s)):
            g[uf.find(i)].append(s[i])
            
        for k in g:
            g[k].sort(reverse=True)
        
        res = []
        for i in range(len(s)):
            res.append(g[uf.find(i)].pop())
        
        return "".join(res)
class Solution:
    def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
            class UF:
                def __init__(self, n): self.p = list(range(n))
                def union(self, x, y): self.p[self.find(x)] = self.find(y)
                def find(self, x):
                    if x != self.p[x]: self.p[x] = self.find(self.p[x])
                    return self.p[x]
            uf, res, m = UF(len(s)), [], defaultdict(list)
            for x,y in pairs: 
                uf.union(x,y)
            for i in range(len(s)): 
                m[uf.find(i)].append(s[i])
            for comp_id in list(m.keys()): 
                m[comp_id].sort(reverse=True)
            for i in range(len(s)): 
                res.append(m[uf.find(i)].pop())
            return ''.join(res)

# from collections import defaultdict

# class Solution:
#     def find(self,x):
#         if(x!=self.parent[x]):
#             x=self.find(self.parent[x])
#         return x
        
        
#     def union(self,x,y):
#         x_find=self.find(x)
#         y_find=self.find(y)
#         self.parent[x_find]=y_find
        
    
    
#     def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
#         n=len(s)
#         self.parent=list(range(n))
        
#         for x,y in pairs:
#             self.union(x,y)
        
#         # print(self.parent)
        
#         groups=defaultdict(list)
#         for i in range(n):
#             tem=self.find(i)
#             # self.parent[i]=tem
#             groups[tem].append(s[i])    
#             # print(tem)
#         # print(self.parent)
        
#         ans=""
#         for comp_id in groups.keys(): 
#             groups[comp_id].sort(reverse=True)
            
#         # print(groups)
        
#         for i in range(n): 
#             ans+=groups[self.find(i)].pop()
#         return ans
        
        
# # #         for i in range(len(s)):
# # #             if(i not in added):
# # #                 groups[i]=[i]
        
# #         # print(groups)
# #         ls=dict()
# #         for i,j in groups.items():
# #             ls[tuple(j)]=sorted([s[ele] for ele in j])
# #         # print(ls)
        
# #         ans=""
# #         for i in range(len(s)):
# #             ans+=ls[tuple(groups[self.parent[i]])].pop(0)
        
# #         return ans
                
        
            
        
        
        
        
# # #         self.ans=s
# # #         visited=set()
# # # #         def traverse(st,pair,i):
# # # #             print(st,i)
# # # #             if(st in visited):
# # #                 return
# # #             visited.add(st)
# # #             a,b=pair[i][0],pair[i][1]
# # #             st=list(st)
# # #             st[a],st[b]=st[b],st[a]
# # #             st="".join(st)
# # #             self.ans=min(self.ans,st)
# # #             # tem=st[:]
# # #             for j in range(len(pair)):
# # #                 if(i!=j):
# # #                     traverse(st,pair,j)
        
        
        
        
# #             # traverse(s,pairs,i)
        
# #         q=[s]
# #         while(q!=[]):
# #             tem=q.pop(0)
# #             if(tem in visited):
# #                 continue
# #             visited.add(tem)
# #             self.ans=min(self.ans,tem)
# #             for i in range(len(pairs)):
# #                 a,b=pairs[i][0],pairs[i][1]
# #                 tem=list(tem)
# #                 tem[a],tem[b]=tem[b],tem[a]
# #                 tem="".join(tem)
# #                 q.append(tem)
            
        
# #         return self.ans

class Solution:
    def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
        p = list(range(len(s)))
        d = defaultdict(list)
        def find(x):
            if x!=p[x]:
                p[x]=find(p[x])
            return p[x]
        def union(x,y):
            p[find(x)]=find(y)
        for a,b in pairs:
            union(a,b)
        for i in range(len(p)):
            d[find(i)].append(s[i])
        for i in d:
            d[find(i)].sort(reverse=True)
        ret=''
        for i in range(len(s)):
            ret+=d[find(i)].pop()
        return ret
class UF:
    def __init__(self, n):
        self.f = list(range(n))
        self.cc = [1] * n
        
    def find(self, x):
        while x != self.f[x]: #
            x = self.f[x]
        return x 
    
    def union(self, x, y):
        fx, fy = self.find(x), self.find(y)
        if fx != fy:
            if self.cc[fx] <= self.cc[fy]: # path compression
                self.f[fx] = fy
                self.cc[fx], self.cc[fy] = 0,  self.cc[fx] + self.cc[fy]
            else:
                self.f[fy] = fx
                self.cc[fx], self.cc[fy] = self.cc[fx] + self.cc[fy], 0
                
class Solution:
    def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
        if not s or not pairs:
            return s 
        
        n = len(s)
        uf = UF(n)
        
        for p in pairs:
            a, b = p[0], p[1]
            uf.union(a, b)
            
        f2c = collections.defaultdict(list)
        for i in range(n):
            f = uf.find(i)
            f2c[f].append(i)
            
        ls = [''] * n
        for _, comp in list(f2c.items()):
            if not comp: continue
            tmp = [s[c] for c in comp]
            comp.sort()
            tmp.sort()
            for i in range(len(comp)):
                ls[comp[i]] = tmp[i]
        
        return ''.join(ls)
                

from heapq import *
class Solution:
    def smallestStringWithSwaps(self, s: str, prs: List[List[int]]) -> str:
        f = {}
        for p in prs:
            r_a, r_b = self.fnd(f, p[0]), self.fnd(f, p[1])
            if r_a != r_b:
                f[r_b] = r_a
        
        m, res = defaultdict(list), []
        for i in range(len(s)):
            m[self.fnd(f, i)].append(s[i])
        for v in list(m.values()):
            heapify(v)
        for i in range(len(s)):
            res.append(heappop(m[self.fnd(f, i)]))
        return ''.join(res)
            
        
    def fnd(self, f, n):
        f[n] = f.get(n, n)
        if f[n] == n:
            return n
        f[n] = self.fnd(f, f[n])
        
        return f[n]

from heapq import *
class Solution:
    def smallestStringWithSwaps(self, s: str, prs: List[List[int]]) -> str:
        f = {}
        for p in prs:
            r_a, r_b = self.fnd(f, p[0]), self.fnd(f, p[1])
            if r_a != r_b:
                f[r_b] = r_a
        
        m, res = defaultdict(list), []
        for i in range(len(s)):
            m[self.fnd(f, i)].append(s[i])
        for v in list(m.values()):
            heapify(v)
            print(v)
        for i in range(len(s)):
            res.append(heappop(m[self.fnd(f, i)]))
        return ''.join(res)
            
        
    def fnd(self, f, n):
        f[n] = f.get(n, n)
        if f[n] == n:
            return n
        f[n] = self.fnd(f, f[n])
        
        return f[n]
        

class Solution:
    
    def dfs(self, i):
        self.tmp.append(self.ls[i])
        self.idx.append(i)
        self.visited.add(i)
        for j in self.d[i]:
            if j not in self.visited:
                self.dfs(j)

    def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:       
        self.ls = list(s)
        self.visited = set()
        self.d = [[] for _ in range(len(self.ls))]

        for i,j in pairs:
            self.d[i].append(j)
            self.d[j].append(i)

        for i in range(len(self.ls)):
            if i not in self.visited:
                self.tmp = []
                self.idx = []
                self.dfs(i)

                sorted_tmp = sorted(self.tmp)
                sorted_idx = sorted(self.idx)
                #print(sorted_tmp, sorted_idx,"CONNECTED", self.visited)

                for index in range(len(sorted_idx)):
                    self.ls[sorted_idx[index]] = sorted_tmp[index]

        return ''.join(self.ls)

class Solution:
    def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
        parent = {i:i for i in range(len(s))}
        def find(x):
            if x != parent[x]:
                parent[x] = find(parent[x])
            return parent[x]
        
        for edge in pairs:  # Union
            parent[find(edge[0])] = find(edge[1])
        
        parent_table = collections.defaultdict(list)
        for i in list(parent.keys()):
            parent_table[find(i)].append(i)
        
        ans = list(s)
        for i in list(parent_table.keys()):
            ids = sorted(parent_table[i])
            t = sorted(ans[j] for j in ids)
            for j in range(len(ids)):
                ans[ids[j]] = t[j]
        
        return ''.join(ans)

from collections import defaultdict

class Solution:
    def find(self,x):
        if(x!=self.parent[x]):
            self.parent[x]=self.find(self.parent[x])
        return self.parent[x]
        
        
    def union(self,x,y):
        x_find=self.find(x)
        y_find=self.find(y)
        self.parent[x_find]=y_find
        
    
    
    def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
        n=len(s)
        self.parent=list(range(n))
        
        for x,y in pairs:
            self.union(x,y)
        
        # print(self.parent)
        
        groups=defaultdict(list)
        for i in range(n):
            tem=self.find(i)
            # self.parent[i]=tem
            groups[tem].append(s[i])    
            # print(tem)
        # print(self.parent)
        
        ans=[]
        for comp_id in groups.keys(): 
            groups[comp_id].sort(reverse=True)
            
        # print(groups)
        
        for i in range(n): 
            ans.append(groups[self.find(i)].pop())
        return "".join(ans)
        
class Solution:
    def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
        
        class UF:
            def __init__(self,n):
                self.p = list(range(n))
            def union(self, x, y):
                self.p[self.find(x)] = self.find(y)
            def find(self, x):
                if x != self.p[x]:
                    self.p[x] = self.find(self.p[x])
                return self.p[x]
        
        uf, res, m = UF(len(s)), [], collections.defaultdict(list)
        
        for x, y in pairs:
            uf.union(x,y)
        for i in range(len(s)):
            m[uf.find(i)].append(s[i])
        for comp_id in list(m.keys()):
            m[comp_id].sort(reverse=True)
        for i in range(len(s)):
            res.append(m[uf.find(i)].pop())
        return ''.join(res)
        
        
        

class DisjSet:
    def __init__(self, n):
        self.disj_set = [-1] * n
        
    def find(self, x):
        while self.disj_set[x] >= 0:
            x = self.disj_set[x]
        return x
    
    def union(self, x, y):
        i = self.find(x)
        j = self.find(y)
        if i == j:
            return
        if self.disj_set[i] < self.disj_set[j]:
            self.disj_set[j] = i
        else:
            if self.disj_set[i] == self.disj_set[j]:
                self.disj_set[j] -= 1
            self.disj_set[i] = j

class Solution:
    def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
        n = len(s)
        result = []
        disj = DisjSet(n)
        for a, b in pairs:
            disj.union(a, b)
        g = collections.defaultdict(collections.Counter)
        for i in range(n):
            g[disj.find(i)][s[i]] += 1
        for i in g:
            g[i] = [[k, v] for k, v in g[i].items()]
            g[i].sort(reverse=True)
        for i in range(n):
            j = disj.find(i)
            chars = g[j]
            result.append(chars[-1][0])
            chars[-1][1] -= 1
            if chars[-1][1] == 0:
                chars.pop()
        return ''.join(result)
def dfs(index,s,edges,visited):
    indices = []
    vals = []
    stack = [index]
    while stack:
        index = stack.pop()
        if index not in visited:
            visited.add(index)
            indices.append(index)
            vals.append(s[index])
            for kid in edges[index]:
                stack.append(kid)
    
    indices.sort()
    vals.sort()
    for index in indices:
        s[index] = vals.pop(0)

class Solution:
    def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
        visited = set()
        s = list(s)
        edges = {}
        for a,b in pairs:
            if a not in edges.keys():
                edges[a] = []
            
            if b not in edges.keys():
                edges[b] = []
                
            edges[a].append(b)
            edges[b].append(a)
        
        for i in edges.keys():
            if i not in visited:
                dfs(i,s,edges,visited)
        
        return ''.join(s)
from collections import defaultdict
class Solution:
    def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
        class UF:
            def __init__(self,n):
                self.p=list(range(n))
            def union(self,x,y):
                self.p[self.find(x)]=self.find(y)
            def find(self,x):
                if self.p[x] is not x:
                    self.p[x]=self.find(self.p[x])
                return self.p[x]
        uf,res,m=UF(len(s)),[],defaultdict(list)
        for x,y in pairs:
            uf.union(x,y)
        for i in range(len(s)):
            m[uf.find(i)].append(s[i])
        for i in m.keys():
            m[i].sort(reverse=True)
        for i in range(len(s)):
            res.append(m[uf.find(i)].pop())
        return ''.join(res)
class Solution:
    def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
        g = collections.defaultdict(list)
        for a, b in pairs:
            g[a].append(b)
            g[b].append(a)
        n = len(s)
        s = list(s)
        while g:
            i, j = g.popitem()
            visited = {i}
            visited.update(j)
            chars = collections.Counter()
            q = collections.deque(j)
            while q:
                i = q.popleft()
                if i in g:
                    j = g.pop(i)
                    q.extend(j)
                    visited.update(j)
            visited = sorted(visited)
            for i in visited:
                chars[s[i]] += 1
            j = 0
            for c in sorted(chars):
                for k in range(chars[c]):
                    s[visited[j]] =  c
                    j += 1
        return ''.join(s)
class Solution:
    def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
        parents = [i for i,_ in enumerate(s)]
        def find(i):
            if i == parents[i]:
                return i
            parents[i] = find(parents[i])
            return parents[i]
        
        def union(i, j):
            i, j = find(i), find(j)
            if i != j:
                parents[i] = parents[j]
            
        for i,j in pairs:
            union(i,j)
                           
        groups = reduce((lambda group, i:
                group[find(i)].append(i)
               or group),
               range(len(s)),
              defaultdict(list))
        
        res = [0] * len(s)
        for items in groups.values():
            for i,c in zip(items, sorted(s[i] for i in items)):
                res[i] = c
            
        return ''.join(res)
class Solution:
    def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
        class UnionFind:
            def __init__(self, N):
                self.arr = [i for i in range(N)]
                
            def find(self, x):
                if self.arr[x] == x:
                    return x
                else:
                    self.arr[x] = self.find(self.arr[x])
                    return self.arr[x]
            
            def union(self, x1, x2):
                self.arr[self.find(x1)] = self.find(x2)
            
        uf = UnionFind(len(s))
        for pair in pairs:
            uf.union(pair[0], pair[1])
        
        g = defaultdict(list)
        for i in range(len(s)):
            g[uf.find(i)].append(s[i])
            
        for k in g:
            g[k].sort(reverse=True)
        
        res = []
        for i in range(len(s)):
            res.append(g[uf.find(i)].pop())
        
        return "".join(res)
class Solution:
    def __init__(self):
      self.roots = {}
      self.ranks = {}
      self.idx2chars = {}
      
    def find(self, idx) -> int:
      self.roots.setdefault(idx, idx)
      self.ranks.setdefault(idx, 1)
      self.idx2chars.setdefault(idx, [self.s[idx]])
      if self.roots[idx] != idx:
        self.roots[idx] = self.find(self.roots[idx])
      return self.roots[idx]
    
    def union(self, idx1, idx2) -> None:
      root1, root2 = self.find(idx1), self.find(idx2)
      if root1 != root2:
        if self.ranks[root2] < self.ranks[root1]:
          self.roots[root2] = root1
          self.idx2chars[root1].extend(self.idx2chars[root2])
        elif self.ranks[root2] > self.ranks[root1]:
          self.roots[root1] = root2
          self.idx2chars[root2].extend(self.idx2chars[root1])
        else:
          self.roots[root2] = root1
          self.idx2chars[root1].extend(self.idx2chars[root2])
          self.ranks[root1] += 1
    
    def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
      """
      Union find, and for each union root, keep a queue of letters in sorted order. Then finally iterate over the string indices to re-build the string
      """
      self.s = s
      for idx1, idx2 in pairs:
        root1, root2 = self.find(idx1), self.find(idx2)
        if root1 != root2:
          self.union(idx1, idx2)
      
      for idx in self.idx2chars:
        self.idx2chars[idx].sort(reverse=True)  # so we can pop the last
      
      # print("roots: ", self.roots)
      # print("idx2chars: ", self.idx2chars)
      
      ordered_s = ''
      for idx in range(len(s)):
        root = self.find(idx)
        # print("idx, root: ", idx, root)
        ordered_s += self.idx2chars[root].pop()
      
      return ordered_s
class Solution:
    def __init__(self):
      self.roots = {}
      self.ranks = {}
      self.idx2chars = {}
      
    def find(self, idx) -> int:
      self.roots.setdefault(idx, idx)
      self.ranks.setdefault(idx, 1)
      self.idx2chars.setdefault(idx, [self.s[idx]])
      if self.roots[idx] != idx:
        self.roots[idx] = self.find(self.roots[idx])
      return self.roots[idx]
    
    def union(self, idx1, idx2) -> None:
      root1, root2 = self.find(idx1), self.find(idx2)
      if root1 != root2:
        if self.ranks[root2] < self.ranks[root1]:
          self.roots[root2] = root1
          self.idx2chars[root1].extend(self.idx2chars[root2])
        elif self.ranks[root2] > self.ranks[root1]:
          self.roots[root1] = root2
          self.idx2chars[root2].extend(self.idx2chars[root1])
        else:
          self.roots[root2] = root1
          self.idx2chars[root1].extend(self.idx2chars[root2])
          self.ranks[root1] += 1
    
    def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
      """
      Union find, and for each union root, keep a queue of letters in sorted order. Then finally iterate over the string indices to re-build the string
      """
      self.s = s
      for idx1, idx2 in pairs:
        root1, root2 = self.find(idx1), self.find(idx2)
        if root1 != root2:
          self.union(idx1, idx2)
      
      for idx in self.idx2chars:
        self.idx2chars[idx].sort(reverse=True)  # so we can pop the last
      
      # print("roots: ", self.roots)
      # print("idx2chars: ", self.idx2chars)
      
      ordered_chars = [''] * len(s)
      for idx in range(len(s)):
        root = self.find(idx)
        # print("idx, root: ", idx, root)
        ordered_chars[idx] = self.idx2chars[root].pop()
      
      return ''.join(ordered_chars)
class Solution:
    def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
        def dfs(i):
            visited[i] = True
            component.append(i)
            for j in adj_lst[i]:
                if not visited[j]:
                    dfs(j)
            
        n = len(s)
        adj_lst = [[] for _ in range(n)]
        for i, j in pairs:
            adj_lst[i].append(j)
            adj_lst[j].append(i)
        visited = [False for _ in range(n)]
        lst = list(s)
        for i in range(n):
            if not visited[i]:
                component = []
                dfs(i)
                component.sort()
                chars = [lst[k] for k in component]
                chars.sort()
                for i in range(len(component)):
                    lst[component[i]] = chars[i]
        return ''.join(lst)
import collections
class Solution:
    def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
        s=list(s)
        path=collections.defaultdict(list)
        for u,v in pairs:
            path[u].append(v)
            path[v].append(u)
        seen=set()
        group=[] # group of nodes
        for i in range(len(s)):
            if i in seen: continue 
            else: # start to search for connected points
                cur=[i]
                connect=[i]
                while cur:
                    temp=[]
                    for c in cur:
                        if c not in seen: 
                            seen.add(c)
                            temp+=[x for x in path[c] if x not in seen]
                    cur=temp
                    connect+=cur
                group.append(connect)
        for g in group:
            temp=sorted([s[i] for i in set(g)])
            for i in sorted(set(g)):
                s[i]=temp[0]
                temp.pop(0)
        return "".join(s)
             
                
            
            
        
        
class Solution:
    def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
        d = {}
        for a,b in pairs:
            if a not in d:
                d[a] = []
            if b not in d:
                d[b] = []
            d[a].append(b)
            d[b].append(a)
        
        def dfs(x, result):
            if x in d:
                result.append(x)
                for y in d.pop(x):
                    dfs(y,result)
        
        s = list(s)
        while d:
            x = next(iter(d))
            result = []
            dfs(x, result)
            result = sorted(result)
            B = sorted([ s[i] for i in result ])
            for i,b in enumerate(B):
                s[result[i]] = b
        return ''.join(s)
class Solution:
    def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
        n=len(s)
        self.p=[i for i in range(n)]
        for x,y in pairs:
            self.union(x,y)
        
        dic=collections.defaultdict(list)
        
        for i in range(n):
            dic[self.find(i)].append(s[i])
            
        for k in dic.keys():
            dic[k]=sorted(dic[k])
            
        res=[]
        for i in range(n):
            res.append(dic[self.find(i)].pop(0))
        
        return ''.join(res)
        
    def find(self, x):
        if self.p[x] != x:
            self.p[x] = self.find(self.p[x])
        return self.p[x]
    
    def union(self,x,y):
        self.p[self.find(x)]=self.find(y)
class DSU:
    def __init__(self, n):
        self.dsu = [i for i in range(n)]
        
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
    def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
        n = len(s)
        dsu = DSU(n)
        for x, y in pairs:
            dsu.union(x, y)
        groups = defaultdict(list)
        for i in range(n):
            key = dsu.find(i)
            groups[key].append(s[i])
        for k in groups:
            groups[k] = sorted(groups[k])
        ans = []
        for i in range(n):
            key = dsu.find(i)
            ans.append(groups[key].pop(0))
        return "".join(ans)
        
class Solution:
    def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
        class UF:
            def __init__(self, n): 
                self.p = list(range(n))
                
            def union(self, x, y): 
                self.p[self.find(x)] = self.find(y)
                
            def find(self, x):
                if x != self.p[x]: 
                    self.p[x] = self.find(self.p[x])
                return self.p[x]
            
        uf, res, m = UF(len(s)), [], defaultdict(list)
        
        for x,y in pairs: 
            uf.union(x,y)
            
        for i in range(len(s)): 
            m[uf.find(i)].append(s[i])
            
        for comp_id in m.keys(): 
            m[comp_id].sort(reverse=True)
            
        for i in range(len(s)): 
            res.append(m[uf.find(i)].pop())
            
        return ''.join(res)
from typing import List


class Solution:
    def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
        """
        find union

        u628a pairs u8996u70ba graph edge, u7528 find union alg u5206u7fa4
        u627eu51fau6bcfu500bu7fa4u7684 idxes / chars
        u500bu5225u6392u5e8fuff0cu586bu56de
        """
        parents = [i for i in range(len(s))]  # u9019u908au8981u7528u4e0du540cu7684 iduff0cu5f8cu9762u624du597du8655u7406

        def find_parent(v) -> int:
            if parents[v] != v:
                parents[v] = find_parent(parents[v])
            return parents[v]

        # union
        for edge in pairs:
            p1, p2 = find_parent(edge[0]), find_parent(edge[1])
            parents[p1] = p2

        # group idx/char in each group as map
        group_idx_mapping = {}
        group_char_mapping = {}
        for i in range(len(s)):  # for each idx
            group_id = find_parent(i)
            try:
                group_idx_mapping[group_id].append(i)
            except KeyError:
                group_idx_mapping[group_id] = [i]
            try:
                group_char_mapping[group_id].append(s[i])
            except KeyError:
                group_char_mapping[group_id] = [s[i]]

        # sort idx/chars in each group
        ans = [''] * len(s)
        for i in range(len(s)):  # for each group
            if i not in group_idx_mapping:
                continue
            # idxes = sorted(group_idx_mapping[i])
            idxes = group_idx_mapping[i]  # already sorted
            chars = sorted(group_char_mapping[i])
            for j in range(len(idxes)):
                ans[idxes[j]] = chars[j]

        return ''.join(ans)

        # """
        # DFS
        # """
        # # build graph
        # g = [[] for _ in range(len(s))]
        # for edge in pairs:
        #     g[edge[0]].append(edge[1])
        #     g[edge[1]].append(edge[0])  # bug: u61c9u70ba indirected graph
        #
        # # DFS, find components
        # visited = set()
        #
        # def f(i):  # fulfill idexes and chars
        #     if i in visited:
        #         return
        #     else:
        #         visited.add(i)
        #
        #     # visit
        #     idxes.append(i)
        #     chars.append(s[i])
        #
        #     for adj in g[i]:
        #         f(adj)
        #
        # ans = [''] * len(s)
        # for i in range(len(s)):
        #     if i in visited:
        #         continue
        #     idxes = []
        #     chars = []
        #     f(i)
        #
        #     # sort each components
        #     idxes.sort() # bug
        #     chars.sort()
        #
        #     for j in range(len(chars)):
        #         ans[idxes[j]] = chars[j]
        #
        # return ''.join(ans)



class Solution:
    def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
        if not pairs:
            return s
        
        adj_list = {}
        visited = set()
        sub_string = ''
        sub_indices = []
        
        for i in range(len(pairs)):
            adj_list[pairs[i][0]] = adj_list.get(pairs[i][0], []) + [pairs[i][1]]
            adj_list[pairs[i][1]] = adj_list.get(pairs[i][1], []) + [pairs[i][0]]
        
        def dfs(index):
            nonlocal sub_string
            sub_string += s[index]
            sub_indices.append(index)
            visited.add(index)
            
            if index in adj_list:
                for neighbor in adj_list[index]:
                    if neighbor not in visited:
                        dfs(neighbor)
                    
        
        for i in range(len(s)):
            sub_string = ''
            sub_indices = []
            if i not in visited:
                dfs(i)
                sub_string = sorted(sub_string)
                sub_indices.sort()
                
                for i in range(len(sub_indices)):
                    s = s[:sub_indices[i]] + sub_string[i] + s[sub_indices[i] + 1 : ]
        
        return s
class DUS:
    def __init__(self, N):
        self.N = N
        self.p = [i for i in range(self.N)]
    
    def find(self, x):
        if self.p[x] != x:
            self.p[x] = self.find(self.p[x])
        return self.p[x] 

    def union(self, x, y):
        rootx = self.find(x)
        rooty = self.find(y)
        if rootx != rooty:
            self.p[rooty] = rootx

class Solution:
    def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
        ans = ""   
        dict_components = collections.defaultdict(list)
        N = len(s)
        dus = DUS(N)
        
        for x, y in pairs: 
            dus.union(x, y)

        for i in range(len(s)):
            dict_components[dus.find(i)].append(s[i])
            
        for comp_id in dict_components.keys():
            dict_components[comp_id].sort()

            
        for i in range(len(s)):
            char = dict_components[dus.find(i)].pop(0)
            ans = ans + char
        
        return ans
            
      
            
            
            
            
            
            
            
            
            
            
            
            
            
        
from collections import defaultdict
class Solution:
    def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
        class UF:
            def __init__(self,n):
                self.p = list(range(n))
            def union(self,x,y):
                self.p[self.find(x)] = self.find(y)
            def find(self,i):
                if self.p[i] != i:
                    self.p[i] = self.find(self.p[i])
                return self.p[i]
            
        uf = UF(len(s))
        m = defaultdict(list)
        res = []
        
        for x,y in pairs:
            uf.union(x,y)
        for i in range(len(s)):
            m[uf.find(i)].append(s[i])
        for key_name in list(m.keys()):
            m[key_name].sort(reverse=True)
        for i in range(len(s)):
            res.append(m[uf.find(i)].pop())
        return ''.join(res)
                
                
                
            
                

class DUS:
    def __init__(self, N):
        self.N = N
        self.p = [i for i in range(self.N)]
    
    def find(self, x):
        if self.p[x] != x:
            self.p[x] = self.find(self.p[x])
        return self.p[x] 

    def union(self, x, y):
        rootx = self.find(x)
        rooty = self.find(y)
        if rootx != rooty:
            self.p[rooty] = rootx

class Solution:
    def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
        ans = ""   
        dict_components = collections.defaultdict(list)
        N = len(s)
        dus = DUS(N)
        
        for x, y in pairs: 
            dus.union(x, y)

        for i in range(len(s)):
            dict_components[dus.find(i)].append(s[i])
            
        for comp_id in dict_components.keys():
            dict_components[comp_id].sort()

            
        for i in range(len(s)):
            char = dict_components[dus.find(i)].pop(0)
            ans += char
        
        return ans
            
      
            
            
            
            
            
            
            
            
            
            
            
            
            
        
class Solution:
    def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
        
        def find(x):
            parent.setdefault(x,x)
            if x!=parent[x]:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x,y):
            px = find(x)
            py = find(y)
            parent[px] = py
            
        
        parent = dict()
        
        
        for i,j in pairs:
            # if i==j:
            #     continue
            x = i
            y = j
            px = find(x)
            py = find(y)
            if px!=py:
                union(x,y)
        
        graph = collections.defaultdict(list)
        
        for i in range(len(s)):
            px = find(i)
            heapq.heappush(graph[px],s[i])
           
  
        res = ''
        mem = collections.defaultdict(int)
        for i in range(len(s)):
            px = find(i)
            res += heapq.heappop(graph[px])
        return res

class Solution:
    def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
        par = {i:i for i in range(len(s))}
        def find(x):
            if par[x] != x:
                par[x] = find(par[x])
            return par[x]
        def union(x,y):
            rx, ry = find(x), find(y)
            if rx != ry:
                par[rx] = ry
        for x,y in pairs:
            union(x,y)
        
        group2chars = defaultdict(list)
        for idx, char in enumerate(s):
            gid = find(idx)
            group2chars[gid].append(char) # collect chars for each group (connected component)
        for gid in group2chars:
            group2chars[gid].sort(reverse=True) # sort the chars (in reverse order) in one connected component
        
        outstr = ''
        for idx, char in enumerate(s):
            gid = find(idx)
            outchar = group2chars[gid].pop() # pop from the order of a to z (since sorted in a reverse way)
            outstr += outchar
        return outstr

class Solution:
    def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
        class UF:
            def __init__(self, n): self.p = list(range(n))
            def union(self, x, y): self.p[self.find(x)] = self.find(y)
            def find(self, x):
                if x != self.p[x]: self.p[x] = self.find(self.p[x])
                return self.p[x]
        uf, res, m = UF(len(s)), [], defaultdict(list)
        for x,y in pairs: 
            uf.union(x,y)
        for i in range(len(s)): 
            m[uf.find(i)].append(s[i])
        for comp_id in list(m.keys()): 
            m[comp_id].sort()
        print(m)
        for i in range(len(s)): 
            res.append(m[uf.find(i)].pop(0))
        return ''.join(res)


from collections import defaultdict
from typing import List


class DSU:
    def __init__(self):
        self.parent = {}
        self.size = {}

    def make_set(self, val):
        self.parent[val] = val
        self.size[val] = 1

    def get_parent(self, val):
        if self.parent[val] == val:
            return val
        parent = self.get_parent(self.parent[val])
        self.parent[val] = parent
        return parent

    def union(self, val1, val2):
        parent1 = self.get_parent(val1)
        parent2 = self.get_parent(val2)

        if parent1 != parent2 and self.size[parent1] >= self.size[parent2]:
            self.parent[parent2] = parent1
            self.size[parent1] += self.size[parent2]
        elif parent1 != parent2 and self.size[parent1] < self.size[parent2]:
            self.parent[parent1] = parent2
            self.size[parent2] += self.size[parent1]

        return True


class Solution:
    def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
        dsu = DSU()
        #   create set for every letter
        for i in range(len(s)):
            dsu.make_set(i)

        #   union connected letters
        for x, y in pairs:
            dsu.union(x, y)

        m = defaultdict(list)
        #   map dsu parent to list of valid letters
        for i in range(len(s)):
            parent = dsu.get_parent(i)
            m[parent].append(s[i])

        #   sort lists of strings
        for key, list_val in m.items():
            m[key] = sorted(list_val)

        res = []
        for j in range(len(s)):
            parent = dsu.get_parent(j)
            smallest_letter = m[parent].pop(0)
            res.append(smallest_letter)

        return ''.join(res)
class Solution:
    def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
        graph = defaultdict(list)
        visited = [False]*len(s)
        out = [None]*len(s)

        for u, v in pairs:
            graph[u].append(v)
            graph[v].append(u)


        def dfs(i, stash):
            visited[i] = True
            stash.append(i)

            for vertice in graph[i]:
                if not visited[vertice]:
                    stash = dfs(vertice, stash)
            return stash

        for i in range(len(s)):
            if not visited[i]:

                indices = sorted(dfs(i, []))
                letters = sorted([s[i] for i in indices])

                for j in range(len(indices)):
                    out[indices[j]] = letters[j]

                # print(out)
        return ''.join(out)

import collections
class Solution:
    def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
        self.uf_table = [idx for idx in range(len(s))]
        def find(p):
            if p != self.uf_table[p]:
                self.uf_table[p] = find(self.uf_table[p])
            return self.uf_table[p]
        
        for p, q in pairs:
            rp = find(p)
            rq = find(q)
            if rp != rq:
                self.uf_table[rp] = rq
                
        conn = collections.defaultdict(list)
        for idx, p in enumerate(self.uf_table):
            conn[find(p)].append(s[idx])
        for k, v in conn.items():
            conn[k] = sorted(v)
        result = []
        for idx in range(len(s)):
            result.append(conn[find(idx)].pop(0))
        
        return ''.join(result)

        
class UF(object):
    
    def __init__(self):
        self.parent = [i for i in range(100001)]
        self.rank = [0]*100001
        
    def find(self, x):
        if self.parent[x] != x: 
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        setx, sety = self.find(x), self.find(y)
        if setx == sety: return False
        elif self.rank[setx] < self.rank[sety]:
            self.parent[setx] = sety
        elif self.rank[sety] < self.rank[setx]:
            self.parent[sety] = setx
        else: 
            self.parent[sety] = setx
            self.rank[setx] += 1
        return True 

class Solution(object):
    def smallestStringWithSwaps(self, s, pairs):
        """
        :type s: str
        :type pairs: List[List[int]]
        :rtype: str
        """
        u, graph, groups, res = UF(), collections.defaultdict(int), collections.defaultdict(list), []
        for x, y in pairs: 
            u.union(x, y)
        for i in range(len(s)):
            groups[u.find(i)].append(s[i])
            graph[i] = u.find(i)
        print(groups)
        print(graph)
        for k in groups.keys():
            groups[k] = collections.deque(sorted(groups[k]))
        return "".join([groups[graph[i]].popleft() for i in range(len(s))]) 
            
        
class Solution:
    def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
        class UF:
            def __init__(self, n): 
                self.p = list(range(n))
            def union(self, x, y): 
                self.p[self.find(x)] = self.find(y)
            def find(self, x):
                if x != self.p[x]: 
                    self.p[x] = self.find(self.p[x])
                return self.p[x]
        uf, res, m = UF(len(s)), [], defaultdict(list)
        for x,y in pairs: 
            uf.union(x,y)
        for i in range(len(s)): 
            m[uf.find(i)].append(s[i])
        for comp_id in m.keys(): 
            m[comp_id].sort(reverse=True)
        for i in range(len(s)): 
            res.append(m[uf.find(i)].pop())
        return ''.join(res)  
class DSU:
    
    def __init__(self, n):
        self.parent = list(range(n))

    def getP(self, i):
        if self.parent[i] != i:
            self.parent[i] = self.getP(self.parent[i])
            return self.parent[i]
        else:
            return i

    def rewrire(self):
        tops = set()
        for i in range(len(self.parent)):
            if i != self.parent[i]:
                self.parent[i] = self.getP(i)
    
    def merge(self, a, b):
        pa, pb = self.getP(a), self.getP(b)
        if pa == pb: return
        if pa < pb:
            self.parent[pb] = pa
        else:
            self.parent[pa] = pb

    def connect(self, xs):
        for a, b in xs:
            self.merge(a, b)

        self.rewrire()
        
class Solution:
    def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
        n = len(s)
        uf = DSU(n)
        uf.connect(pairs)
        
        group = defaultdict(list)
        for i, pindex in enumerate(uf.parent):
            group[pindex].append((s[i], i))
            
        r = [None] * n
        
        for gk, gv in list(group.items()):
            cxs = []
            ixs = []
            for c, index in gv:
                cxs.append(c)
                ixs.append(index)
            
            cxs.sort()
            ixs.sort()
            
            for i in range(len(ixs)):
                r[ixs[i]] = cxs[i]
        
        
        return ''.join(r)

class Solution:
    def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
        n=len(s)
        m=len(pairs)
        
        ## If string length is 1 or less, or if there are no swappable pairs, return original string
        if n<=1 or m==0: return s
        
        ## Parent array for storing group parent ids in union find
        ## For each node, traversing upward would lead to the group leader.
        ## All connected indices share the same group leader
        parent= [i for i in range(n) ] 
        
        ## Returns the group leader index for the given index
        def find(i):
            pi = parent[i]
            while parent[pi] != pi :
                pi = parent[pi]
            parent[i] = pi
            return parent[i]
        
        ## Merges two indices into same group
        def union(i,j):
            pi=find(i)
            pj=find(j)
            if pi!=pj:
                parent[pj]=pi
        
        for index1,index2 in pairs:
            union(index1,index2)

        ## Forming groups or connected components
        groups={}
        for index in range(n):
            leader = find(index)
            groups[leader] = groups.get(leader,[])
            groups[leader].append(s[index])

        for leader in groups.keys():
            groups[leader].sort()

        group_index={}
        result=''
        for index in range(n):
            leader=find(index)
            group_index[leader]=group_index.get(leader,0)+1
            result+=groups[leader][group_index[leader]-1]
        return result
class Solution:
    def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
        n=len(s)
        m=len(pairs)
        
        ## If string length is 1 or less, or if there are no swappable pairs, return original string
        if n<=1 or m==0: return s
        
        ## Parent array for storing group parent ids in union find
        ## For each node, traversing upward would lead to the group leader.
        ## All connected indices share the same group leader
        parent= [i for i in range(n) ] 
        
        ## Returns the group leader index for the given index
        def find(i):
            pi = parent[i]
            while parent[pi] != pi :
                pi = parent[pi]
            parent[i] = pi
            return parent[i]
        
        ## Merges two indices into same group
        def union(i,j):
            pi=find(i)
            pj=find(j)
            if pi!=pj:
                parent[pj]=pi
        
        for index1,index2 in pairs:
            union(index1,index2)

        ## Forming groups or connected components
        groups={}
        for index in range(n):
            leader = find(index)
            groups[leader] = groups.get(leader,[])
            groups[leader].append(s[index])

        for leader in groups.keys():
            groups[leader].sort()

        group_index={}
        result=''
        for index in range(n):
            leader=find(index)
            group_index[leader]=group_index.get(leader,0)+1
            result+=groups[leader][group_index[leader]-1]
        return result     
class Solution:
    def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
        n = len(s)
        parent = [i for i in range(n)]
        def find(i):
            if not i ==parent[i]:
                parent[i]=find(parent[i])
            return parent[i]    
        def union(i, j):
            a = find(i)
            b = find(j)
            parent[a] = b
        for i, j in pairs:
            union(i, j)
        memo = collections.defaultdict(list)
        for i in range(n):
            memo[find(i)].append(s[i])
        for k in memo.keys():
            memo[k].sort(reverse=True)
        res = []
        for i in range(n):
            res.append(memo[find(i)].pop())
        return "".join(res) 
from collections import defaultdict
class Solution:
    def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
        p = list(range(len(s)))
        d = defaultdict(list)
        def find(x):
            if p[x]!=x:
                p[x]=find(p[x])
            return p[x]
        def union(x,y):
            x,y = find(x),find(y)
            p[x]=y
            return p[x] 
        for a,b in pairs:
            union(a,b)
        for i in range(len(s)):
            d[find(i)].append(s[i])
        for x in d:
            d[find(x)].sort(reverse=True)
        ret=''
        for i in range(len(s)):
            ret+=d[find(i)].pop()
        return ret
class Solution:
    def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
        from collections import defaultdict
        n = len(s)
        roots = list(range(n))
        sizes = [1]*n
        
        def find(node):
            root = node
            while root != roots[root]:
                root = roots[root]
                
            while node != root:
                next_node = roots[node]
                roots[node] = root
                node = next_node
            
            return root
        
        def union(node1, node2):
            root1, root2 = find(node1), find(node2)
            
            if root1 == root2:
                return False
            
            if sizes[root2] > sizes[root1]:
                root1, root2 = root2, root1
            
            sizes[root1] += sizes[root2]
            roots[root2] = root1
            
        for x,y in pairs:
            union(x, y)
        
        for i in range(n):
            find(i)
        
        indices = defaultdict(lambda: [])
        chars = defaultdict(lambda: [])
        for i, root in enumerate(roots):
            c = s[i]
            indices[root].append(i)
            chars[root].append(c)
        
        result = [0]*n
        for key in indices.keys():
            for i, v in zip(indices[key], sorted(chars[key])):
                result[i] = v
                
        return "".join(result)
        
        
                
class UnionFind:
    
    def __init__(self, n):
        self.reps = [i for i in range(n)]
    
    def find(self, x):
        while x != self.reps[x]:
            self.reps[x] = self.reps[self.reps[x]]
            x = self.reps[x]
        return x
    def union(self, x, y):
        x_root = self.find(x)
        y_root = self.find(y)
        if x_root != y_root:
            self.reps[x_root] = y_root
class Solution:
    def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
        uf = UnionFind(len(s))
        
        for i, j in pairs:
            uf.union(i, j)
            
        mp = collections.defaultdict(list)
        
        for i in range(len(s)):
            mp[uf.find(i)].append(s[i])
            
        for comp_id in mp.keys(): 
            mp[comp_id].sort(reverse=True)
        
        
        ret = []
        for i in range(len(s)): 
            ret.append(mp[uf.find(i)].pop())
        return "".join(ret)
        
class Solution:
    def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
        p = {}

        def getP(i: int) -> int:
            if i not in p:
                return -1
            return i if p[i] == i else getP(p[i])

        uf = {}
        index = 0
        for (x, y) in pairs:
            px = getP(x)
            py = getP(y)
            if px == -1 and py == -1:
                p[x] = min(x, y)
                p[y] = min(x, y)
                uf[min(x, y)] = pairs[index]
            elif px == -1:
                uf[py].append(x)
                p[x] = py
            elif py == -1:
                uf[px].append(y)
                p[y] = px
            elif px != py:
                p[px] = min(px, py)
                p[py] = min(px, py)
                uf[min(px, py)] += uf.pop(max(px, py))
            index += 1
        ans = list(s)
        for k in list(uf.keys()):
            st = sorted(set(uf[k]))
            tmp = [s[i] for i in st]
            tmp.sort()
            idx = 0
            for i in st:
                ans[i] = tmp[idx]
                idx += 1
        return ''.join(ans)

from typing import List


class Solution:
    def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
        # build graph
        g = [[] for _ in range(len(s))]
        for edge in pairs:
            g[edge[0]].append(edge[1])
            g[edge[1]].append(edge[0])  # bug: u61c9u70ba indirected graph

        # DFS, find components
        visited = set()

        def f(i):  # fulfill idexes and chars
            if i in visited:
                return
            else:
                visited.add(i)

            # visit
            idxes.append(i)
            chars.append(s[i])

            for adj in g[i]:
                f(adj)

        ans = [''] * len(s)
        for i in range(len(s)):
            if i in visited:
                continue
            idxes = []
            chars = []
            f(i)

            # sort each components
            idxes.sort()
            chars.sort()

            for j in range(len(chars)):
                ans[idxes[j]] = chars[j]

        return ''.join(ans)



from collections import defaultdict
class Solution:
    def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
        p=list(range(len(s)))
        d=defaultdict(list)
        def find(x):
            if p[x]!=x:
                p[x]=find(p[x])
            return p[x]
        def union(x,y):
            x,y = find(x),find(y)
            if x==y:
                return False
            p[x]=y
            return p[x]
        for a,b in pairs:
            union(a,b)
        for i in range(len(p)):
            d[find(i)].append(s[i])
        for x in d:
            d[find(x)].sort(reverse=True)
        ret=''
        for i in range(len(s)):
            ret+=d[find(i)].pop()
        return ret

import collections
import functools
import heapq
import itertools
import sys
from functools import lru_cache
from typing import List
from fractions import gcd

'''
u7ed9u4f60u4e00u4e2au5b57u7b26u4e32u00a0suff0cu4ee5u53cau8be5u5b57u7b26u4e32u4e2du7684u4e00u4e9bu300cu7d22u5f15u5bf9u300du6570u7ec4u00a0pairsuff0cu5176u4e2du00a0pairs[i] =u00a0[a, b]u00a0u8868u793au5b57u7b26u4e32u4e2du7684u4e24u4e2au7d22u5f15uff08u7f16u53f7u4ece 0 u5f00u59cbuff09u3002
u4f60u53efu4ee5 u4efbu610fu591au6b21u4ea4u6362 u5728u00a0pairsu00a0u4e2du4efbu610fu4e00u5bf9u7d22u5f15u5904u7684u5b57u7b26u3002
u8fd4u56deu5728u7ecfu8fc7u82e5u5e72u6b21u4ea4u6362u540euff0csu00a0u53efu4ee5u53d8u6210u7684u6309u5b57u5178u5e8fu6700u5c0fu7684u5b57u7b26u4e32u3002

u6ce8u610fu8fd9u4e0du662fu4e00u9053dfsu6216bfsu9898u3002

u5982u679cu4e24u4e2au4f4du7f6eu53efu4ee5u4efbu610fu4ea4u6362uff0cu5219u4e24u4e2au4f4du7f6eu6392u5e8fu5373u53efu3002

u5982u679c(1,2), (2,3), u52191 2 3u4f4du7f6eu6392u5e8fu5373u53efu3002

u6240u4ee5u95eeu9898u662fu627eu5230u5e76u67e5u96c6, u7136u540eu76f4u63a5u6392u5e8fu5373u53efu3002
'''


class UFSet:
    def __init__(self, n):
        self.dp = [-1 for _ in range(n)]

    def find(self, x):
        if self.dp[x] < 0:
            return x
        self.dp[x] = self.find(self.dp[x])
        return self.dp[x]

    def union(self, x, y):
        root_x, root_y = self.find(x), self.find(y)
        if root_x == root_y: return
        self.dp[root_x] += self.dp[root_y]
        self.dp[root_y] = root_x

    def get_group(self):
        ret = collections.defaultdict(list)
        for i in range(len(self.dp)):
            ret[self.find(i)].append(i)
        return ret


class Solution:
    def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
        n = len(s)
        ufs = UFSet(n)
        for i, j in pairs:
            ufs.union(i, j)
        group_map = ufs.get_group()
        ret = [i for i in s]
        for group in list(group_map.values()):
            sort_group = sorted([s[i] for i in group])
            for i, j in zip(group, sort_group):
                ret[i] = j
        return ''.join(ret)


class Solution:
    def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
        n = len(s)
        root = { i:i for i in range(n) }
        
        def find(i): # path compression
            if root[i] != i:
                root[i] = find(root[i])
            return root[i]
            
            
        def union(i,j):
            ri = find(i)
            rj = find(j)
            if ri != rj:
                # root[j] = root[i]
                root[ri] = root[rj] # root(rj)
            return
        
        for i,j in pairs:
            union(i,j)
        
        d = collections.defaultdict( list )
        for i in range(n):
            d[find(i)].append(i)
        
        print(root)
        
        res = list(s)
        for k in d:
            tmp = sorted([s[i] for i in d[k]])
            for i in range(len(tmp)):
                res[ d[k][i] ] = tmp[i]
        
        return "".join(res)
        
        
        
        
        
class Solution:
    def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
        n = len(s)
        parent = [i for i in range(n)]
        
        for u, v in pairs:
            self.union(parent, u, v)
        
        m = collections.defaultdict(list)
        for i in range(n):
            m[self.find_root(parent, i)].append(i)
        
        ans = list(s)
        for _, indices in m.items():
            temp = []
            for i in indices:
                temp.append(s[i])
            temp.sort()
            for i in range(len(temp)):
                ans[indices[i]] = temp[i]
        
        return "".join(ans)
    
    def find_root(self, parent, x):
        if parent[x] != x:
            parent[x] = self.find_root(parent, parent[x])
        return parent[x]
    
    def union(self, parent, x, y):
        x_root = self.find_root(parent, x)
        y_root = self.find_root(parent, y)
        if x_root != y_root:
            parent[x_root] = y_root
class UnionFind():
    def __init__(self,n):
        self.parent = list(range(n))
        self.size = [1]*n
    
    def find(self,A):
        root = A
        
        while root != self.parent[root]:
            root = self.parent[root]
        
        while A!=root:
            old_parent = self.parent[A]
            self.parent[A] = root
            A = old_parent
        return root
    
        
    def union(self,A,B):
        root_A = self.find(A)
        root_B = self.find(B)
        
        if self.size[root_A]>self.size[root_B]:           
            self.parent[root_B] = root_A
            self.size[root_A] += self.size[root_B]
        else:
            self.parent[root_A] = root_B
            self.size[root_B] += self.size[root_A]
            
        
class Solution:
    def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
        
        uf = UnionFind(len(s))
        d = collections.defaultdict(list)
        res = []
        
        for i,j in pairs:
            uf.union(i,j)
        
        for i in range(len(s)):
            d[uf.find(i)].append(s[i])
        
        for parent in d:
            d[parent].sort(reverse = True)
        
        for i in range(len(s)):
            res.append(d[uf.find(i)].pop())
            
        return "".join(res)
            
        
class Solution:
    def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
        n = len(s)
        uf = UF(n)
        for pair in pairs:
            uf.union(pair[0], pair[1])
        
        groups = collections.defaultdict(list)
        for i in range(n):
            r = uf.root(i)
            groups[r].append(i)
        
        res = ['' for _ in range(n)]
        for r, group in list(groups.items()):
            if len(group) == 1:
                res[r] = (s[group[0]])
            else:
                temp = [s[i] for i in group]
                temp.sort()
                for index, idx in enumerate(sorted(group)):
                    res[idx] = temp[index]
        return ''.join(res)
    
class UF:
    def __init__(self, n):
        self.parents = list(range(n))
    
    def find(self, x):
        if x != self.parents[x]:
            self.parents[x] = self.find(self.parents[x])
        return self.parents[x]
    
    def union(self, x, y):
        px = self.find(x)
        py = self.find(y)
        if px != py:
            self.parents[px] = py
    
    def root(self, x):
        return self.find(x)

class Solution:
    def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
        class DSU:
            def __init__(self,n):
                self.n = n
                self.l = [i for i in range(n)]
            def get(self,c):
                if self.l[c] != c:
                    self.l[c] = self.get(self.l[c])
                return self.l[c]
            def merge(self,c,d):
                t1 = self.get(c)
                t2 = self.get(d)
                self.l[t1] = t2
                
        g = DSU(len(s))
        for a,b in pairs:
            g.merge(a,b)
        
        res = {}
        for i,c in enumerate(s):
            cl = g.get(i)
            if cl in res:
                res[cl][0].append(i)
                res[cl][1].append(c)
            else:
                res[cl] = [[i],[c]]
        
        result = ['']*len(s)
        for cl in res:
            l = sorted(res[cl][1],reverse=True)
            for i in res[cl][0]:
                result[i] = l.pop()
        return "".join(result)
class Solution:
    def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
        parents = [i for i in range(len(s))]
        ranks = [1 for _ in range(len(s))]
        components = collections.defaultdict(list)
        def find(x):
            while x != parents[x]:
                parents[x] = parents[parents[x]]
                x = parents[x]
            return x
        def union(x, y):
            px, py = find(x), find(y)
            if px == py:
                return False
            if ranks[px] > ranks[py]:
                parents[py] = px
            elif ranks[px] < ranks[py]:
                parents[px] = py
            else:
                parents[py] = px
                ranks[px] += 1
            return True
        for a, b in pairs:
            union(a, b)
        for i in range(len(parents)):
            components[find(i)].append(s[i])
        for comp_id in components:
            components[comp_id].sort(reverse=True)
        res = ''
        for i in range(len(s)):
            res += components[find(i)].pop()
        return res
class Solution:
    def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
        s = list(s)
        graph = [[] for _ in range(len(s))]
        for u,v in pairs:
            graph[u].append(v)
            graph[v].append(u)
        stack = []
        vis = [False]*len(s)
        self.res = ""
        self.index = []
        def dfs(graph,u):
            vis[u] = True
            self.res += s[u]
            self.index.append(u)
            for v in graph[u]:
                if not vis[v]:
                    dfs(graph,v)
        for i in range(len(graph)):
            if not vis[i]:
                self.res = ""
                self.index = []
                dfs(graph,i)
                self.index.sort()
                self.res = sorted(list(self.res))
                for i in range(len(self.index)):
                    s[self.index[i]] = self.res[i]
        return ''.join(s)

class Solution:
    def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
        
        def find(x):
            if x!=parent[x]:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x,y):
            px = find(x)
            py = find(y)
            parent[px] = py
            
        parent = list(range(len(s)+1))
        
        for x,y in pairs:
            px = find(x)
            py = find(y)
            if px!=py:
                union(x,y)
        
        graph = collections.defaultdict(list)
        
        for i in range(len(s)):
            px = find(i)
            heapq.heappush(graph[px],s[i]) # We are using priority queue to keep track of the lexicographical ordering
        
        res = ''
        for i in range(len(s)):
            px = find(i)
            res += heapq.heappop(graph[px])
        return res
from collections import defaultdict
class Solution:
    def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
        s = list(s)
        graph = [[] for _ in range(len(s))]
        for u,v in pairs:
            graph[u].append(v)
            graph[v].append(u)
        stack = []
        vis = [False]*len(s)
        self.res = ""
        self.index = []
        def dfs(graph,u):
            vis[u] = True
            self.res += s[u]
            self.index.append(u)
            for v in graph[u]:
                if not vis[v]:
                    dfs(graph,v)
        for i in range(len(graph)):
            if not vis[i]:
                self.res = ""
                self.index = []
                dfs(graph,i)
                self.index.sort()
                self.res = sorted(list(self.res))
                for i in range(len(self.index)):
                    s[self.index[i]] = self.res[i]
        return ''.join(s)
class Solution:
    def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
        graph = dict()
        for i in range(len(s)):
            graph[i]=set()
            
        for pair in pairs:
            fro = pair[0]
            to = pair[1]
            graph[fro].add(to)
            graph[to].add(fro)
            
        def dfs(node, graph,explored,s,path):
            path.append(node)
            explored[node]=True
            string = s[node]
            
            for neighbour in graph[node]:
                if not explored[neighbour]:
                    res = dfs(neighbour,graph,explored,s,path)
                    string+=res[0]
                    
            return (string,path)
        
        connected=[]
        explored = [False]*len(s)
        
        for node in graph:
            if not explored[node]:
                connected.append(dfs(node,graph,explored,s,[]))
                
                
            
        stringList = [""]*len(s)
        
        for conn in connected:
            st = "".join(sorted(conn[0]))
            path = sorted(conn[1])
            
            for i in range(len(st)):
                char = st[i]
                ind = path[i]
                stringList[ind]=char
                
        return "".join(stringList)
                
            
        
                
        
                
            
class DSU:
    def __init__(self, n):
        self.parent = [i for i in range(n)]
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        xroot, yroot = self.find(x), self.find(y)
        if xroot != yroot:
            self.parent[yroot] = xroot
            

class Solution:
    def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
        n = len(s)
        dsu = DSU(n)
        ans = []
        m = collections.defaultdict(list)
        
        for i, j in pairs:
            dsu.union(i, j)
        
        for i in range(n):
            m[dsu.find(i)].append(s[i])
        
        for key in list(m.keys()):
            m[key].sort(reverse = True)
            
        for i in range(n):
            ans.append(m[dsu.find(i)].pop())
        
        return ''.join(ans)
        

from collections import defaultdict

class Solution:
    def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
        class UF:
            def __init__(self, s): self.p = [i for i in range(len(s))]

            def union(self, x, y): self.p[self.find(x)] = self.find(y)

            def find(self, x):
                if x != self.p[x]: self.p[x] = self.find(self.p[x])
                return self.p[x]
        res, cc, uf = [], defaultdict(list), UF(s)
        for x, y in pairs:
            uf.union(x, y)
        for i, c in enumerate(s):
            cc[uf.find(i)].append(c)
        for cc_id in cc.keys():
            cc[cc_id].sort(reverse=True)
        for i in range(len(s)):
            res.append(cc[uf.find(i)].pop())
        return ''.join(res)
class Solution:
    def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
        class UF:
            def __init__(self, n): self.p = list(range(n))
            def union(self, x, y): self.p[self.find(x)] = self.find(y)
            def find(self, x):
                if x != self.p[x]: self.p[x] = self.find(self.p[x])
                return self.p[x]
        uf, res, m = UF(len(s)), [], defaultdict(list)
        for x,y in pairs: 
            uf.union(x,y)
        for i in range(len(s)): 
            m[uf.find(i)].append(s[i])
        for comp_id in m.keys(): 
            m[comp_id].sort(reverse=True)
        for i in range(len(s)): 
            res.append(m[uf.find(i)].pop())
        return ''.join(res)
from collections import defaultdict

class Solution:
    def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
#         def find(uf, node):        
#             if uf[node] != node:
#                 return find(uf, uf[node])
                
#             return node
            
#         def union(uf, node1, node2):
#             uf[find(uf, node1)] = find(uf, node2)
            
#         uf = list(range(len(s)))
        
#         if not pairs:
#             return s
        
#         for pair in pairs:
#             union(uf, pair[0], pair[1])
        
#         groups = defaultdict(list)
        
#         for i in range(len(s)):
#             groups[find(uf, i)].append(s[i])
                        
#         for group in groups:
#             groups[group].sort(reverse=True)
        
#         res = []

#         for i in range(len(s)):
#             res.append(groups[find(uf, i)].pop())
            
#         return ''.join(res)
    
        class UF:
            def __init__(self, n): self.p = list(range(n))
            def union(self, x, y): self.p[self.find(x)] = self.find(y)
            def find(self, x):
                if x != self.p[x]: self.p[x] = self.find(self.p[x])
                return self.p[x]
        uf, res, m = UF(len(s)), [], defaultdict(list)
        for x,y in pairs: 
            uf.union(x,y)
        for i in range(len(s)): 
            m[uf.find(i)].append(s[i])
        for comp_id in list(m.keys()): 
            m[comp_id].sort(reverse=True)
        for i in range(len(s)): 
            res.append(m[uf.find(i)].pop())
        return ''.join(res)

class Solution:
    def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
        class UF:
            def __init__(self,n):
                self.p=list(range(n))
            def find(self,x):
                if x!=self.p[x]:
                    self.p[x]=self.find(self.p[x])
                return self.p[x]
            def union(self,x,y):
                self.p[self.find(x)]=self.find(y)
        d=defaultdict(list)
        uf=UF(len(s))
        ans=[]
        for x,y in pairs:
            uf.union(x,y)
        for i in range(len(s)):
            d[uf.find(i)].append(s[i])
        for key in d:
            d[key].sort(reverse=True)
        for i in range(len(s)):
            ans.append(d[uf.find(i)].pop())
        return "".join(ans)
    
class DSU:
    
    def __init__(self, N):
        self.par = list(range(N))
        self.sz = [1] * N
    
    def find(self, x):
        if self.par[x] != x:
            self.par[x] = self.find(self.par[x])
        return self.par[x]
    
    def union(self, x, y):
        x, y = self.find(x), self.find(y)
        if x == y:
            return False
        if self.sz[x] < self.sz[y]:
            x, y = y, x
        self.sz[x] += self.sz[y]
        self.par[y] = x
        return True
    
class Solution:
    def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
        N = len(s)
        dsu = DSU(N)
        for a, b in pairs:
            dsu.union(a, b)
        s = list(s)
        components = [[] for i in range(N)]
        for i in range(N):
            components[dsu.find(i)].append(i)
        for i in range(N):
            chars = [s[j] for j in components[i]]
            chars.sort()
            for j, v in enumerate(components[i]):
                s[v] = chars[j]
        
        return "".join(s)
        
class UnionFind(object):
    def __init__(self,n):
        self._parent = [0]*n
        self._size = [1]*n
        self.count = n
        for i in range(n):
            self._parent[i] = i
            
    def union(self, p, q):
        rootp = self.find(p)
        rootq = self.find(q)
        if rootp == rootq:
            return
        self.count -= 1
        if self._size[rootp] > self._size[rootq]:
            self._size[rootp] += self._size[rootq]
            self._parent[rootq] = self._parent[q] = rootp
        else:
            self._size[rootq] += self._size[rootp]
            self._parent[rootp] = self._parent[p] = rootq
    
    def find(self, n):
        while self._parent[n] != n:
            self._parent[n] = self._parent[self._parent[n]]
            n = self._parent[n]
        return n
    
    def connected(self, p, q):
        return self.find(p) == self.find(q)
    
class Solution:
    def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
        N = len(s)
        if N < 2: return s
        uf = UnionFind(N)
        for pair in pairs:
            uf.union(pair[0],pair[1])
        
        dict = defaultdict(list)
        for i in range(N):
            r = uf.find(i)
            dict[r].append(i)
            
        res = [' ']*N
        for lst in list(dict.values()):            
            subs = []
            for idx in lst:
                subs.append(s[idx])
            subs.sort()
            i2 = 0
            for idx in lst:
                res[idx]=subs[i2]
                i2+=1
        return ''.join(res)
            
        
        
            

class Solution:
    def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
        
        dd = {x:x for x in [i for v in pairs for i in v]}
        
        def merge(x,y):
            x,y = find(x), find(y)
            if x!=y:
                dd[x] = y
        
        def find(x):
            if dd[x] != x:
                dd[x] = find(dd[x])
            return dd[x]
        
        for i,x in pairs:
            merge(i,x)
            
        agg = collections.defaultdict(list)
        
        for i in dd:
            agg[find(i)].append(s[i])
            
        for i in agg:
            agg[i] = sorted(agg[i], reverse = 1)
            
        s = list(s)
        
        for i in range(len(s)):
            if i in dd:
                s[i] = agg[find(i)].pop()
        
        return ''.join(s)
class Solution:
    def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
        
        edges = collections.defaultdict(list)
        for p in pairs:
            edges[p[0]].append(p[1])
            edges[p[1]].append(p[0])
            
        # print(edges)
        
        ans = list(s)
        seen = set()
        for i,c in enumerate(s):
            if i in seen: continue
            seen.add(i)
            h1,h2 = [],[]
            frontier = [i]
            while frontier:
                cur = frontier.pop()
                heapq.heappush(h1,cur)
                heapq.heappush(h2,ans[cur])
                for j in edges[cur]:
                    # print(cur,j)
                    if j not in seen:
                        # print('New index')
                        seen.add(j)
                        frontier.append(j)
            # print(f' Current-{cur}, h1-{h1}, h2-{h2}')
            while h1: ans[heapq.heappop(h1)] = heapq.heappop(h2)
        
        return ''.join(ans)
class DSU:
    def __init__(self,n):
        self.parent = [i for i in range(n)]
        self.rank = [1 for i in range(n)]
        self.count = n
        
    def find(self,x):
        if x!=self.parent[x]:
            self.parent[x]=self.find(self.parent[x])
        return self.parent[x]

    def union(self,x,y):
        px = self.find(x)
        py = self.find(y)
        if px==py:
            return
        self.count-=1
        if self.rank[px]>self.rank[py]:
            self.parent[py] = px
            self.rank[px]+=self.rank[py]
        else:
            self.parent[px]=py
            self.rank[py]+=self.rank[x]
        
class Solution:
    def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
        n = len(s)
        dsu = DSU(n)
         # [[0,3],[1,2],[0,2]]
        for x,y in pairs:
            dsu.union(x,y)
        # [0,1,2,3]
        
        # Club everyone with the same parent
        if dsu.count == 1:
            return "".join(sorted(s))
        hm = collections.defaultdict(list)
        for i in range(n):
            hm[dsu.find(i)].append(s[i])
        for key in hm:
            hm[key].sort(reverse=True)
        res = []
        for i in range(n):
            res.append(hm[dsu.find(i)].pop())
        return "".join(res)
            
        
from heapq import heappush, heappop

class Solution:
    def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
        def parent(a):
            val, _ = d[a]
            if val == a:
                return a
            return parent(val)
        
        def union(a, b):
            parent1 = parent(a)
            parent2 = parent(b)
            
            if parent1 == parent2:
                return
            
            a, n1 = d[parent1]
            b, n2 = d[parent2]
            if n1 > n2:
                d[b] = (a, n2)
                d[a] = (a, n1 + n2)
            else:
                d[a] = (b, n1)
                d[b] = (b, n1 + n2)
                
        d = [(i, 1) for i in range(len(s))]
        for i, j in pairs:
            union(i, j)
        arrs = [[] for _ in range(len(s))]
        for i in range(len(s)):
            heappush(arrs[parent(i)], s[i])
        sol = list()
        for i in range(len(s)):
            sol.append(heappop(arrs[parent(i)]))
        return ''.join(sol)
        

class Solution:
    def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
        res = list(s)
        adj = defaultdict(set)
        for a, b in pairs:
            adj[a].add(b)
            adj[b].add(a)
            
        while adj:
            i = next(iter(adj))
            v = []
            self.dfs(i, adj, v)
            v = sorted(v)
            chars = sorted([s[i] for i in v])
            for i, c in enumerate(chars):
                res[v[i]] = c
                
        return ''.join(res)
    
    def dfs(self, i, adj, res):
        if i in adj:
            res.append(i)
            for j in adj.pop(i):
                self.dfs(j, adj, res)
        
                

class Solution:
    def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
        def union(x, y):
            graph[find(x)] = find(y)
        
        def find(x):
            path = []
            if x not in graph:
                return x
            while graph[x] != x:
                path.append(x)
                x = graph[x]
            for n in path:
                graph[n] = x
            return x

        graph = {}
        
        for u, v in pairs:
            if u not in graph:
                graph[u] = u
            if v not in graph:
                graph[v] = v
            union(u, v)
            
        ans = []
        comp = collections.defaultdict(list)
        for i, c in enumerate(s):
            comp[find(i)].append(c)
        for v in comp.values():
            v.sort(reverse=True)
        for i in range(len(s)):
            ans.append(comp[find(i)].pop())
        return "".join(ans)
class Solution:
    # 21:50
    """
    s = "dcab", pairs = [[0,3],[1,2],[0,2]]
    dcab -[1, 2]->dacb
    dacb -[0, 3]->bacd
    not possible anymore
    """
    def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
        res = list(s)
        groups = self.groups(pairs, len(s))
        for g in groups:
            g = sorted(g)
            chars = sorted([s[i] for i in g])
            for i, c in enumerate(chars):
                res[g[i]] = c
                
        return ''.join(res)
    
    def groups(self, pairs, n):
        adj = collections.defaultdict(set)
        for a, b in pairs:
            adj[a].add(b)
            adj[b].add(a)
            
        arr = []
        while adj:
            i = next(iter(adj))
            v = []
            self.dfs(i, adj, v)
            arr.append(v)
                
        return arr
    
    def dfs(self, i, adj, res):
        if i in adj:
            res.append(i)
            for j in adj.pop(i):
                self.dfs(j, adj, res)
        
                
        
from collections import defaultdict

class DisjointSet:
    class Node:
        def __init__(self, x):
            self.val = x
            self.parent = self
            self.rank = 0
            
    def __init__(self, node_num):
        self.val_to_node = {}
        
        for val in range(node_num):
            self.val_to_node[val] = DisjointSet.Node(val)
            
    def find(self, x):
        return self._find(self.val_to_node[x]).val
    
    def _find(self, node):
        if node.parent is node:
            return node
        
        node.parent = self._find(node.parent)
        return node.parent
    
    def union(self, val1, val2):
        root1 = self._find(self.val_to_node[val1])
        root2 = self._find(self.val_to_node[val2])
        
        if root1 is root2:
            return
        
        if root2.rank > root1.rank:
            root1, root2 = root2, root1
            
        if root1.rank == root2.rank:
            root1.rank += 1
        
        root2.parent = root1
            

class Solution:
    def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
        if not pairs:
            return s
        
        disjoint_set = DisjointSet(len(s))
        
        for u, v in pairs:
            disjoint_set.union(u, v)
        
        connected_components = defaultdict(list)
        
        for i in range(len(s)):
            connected_components[disjoint_set.find(i)].append(i)
        
        res = [None] * len(s)
        for group in list(connected_components.values()):
            sorted_chars = sorted([s[i] for i in group])
            
            for idx, s_i in enumerate(sorted(group)):
                res[s_i] = sorted_chars[idx]
                
        return ''.join(res)

class Solution:
    # 21:50
    """
    s = "dcab", pairs = [[0,3],[1,2],[0,2]]
    dcab -[1, 2]->dacb
    dacb -[0, 3]->bacd
    not possible anymore
    """
    def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
        res = list(s)
        groups = self.groups(pairs, len(s))
        for g in groups:
            g = sorted(g)
            chars = sorted([s[i] for i in g])
            for i, c in enumerate(chars):
                res[g[i]] = c
                
        return ''.join(res)
    
    def groups(self, pairs, n):
        adj = collections.defaultdict(set)
        for p in pairs:
            adj[p[0]].add(p[1])
            adj[p[1]].add(p[0])
            
        arr = []
        while adj:
            i = next(iter(adj))
            v = []
            self.dfs(i, adj, v)
            arr.append(v)
                
        return arr
    
    def dfs(self, i, adj, res):
        if i in adj:
            res.append(i)
            for j in adj.pop(i):
                self.dfs(j, adj, res)
        
                
        
class Solution:
    def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
        # create joined sets 
        # [0, 3] and [1, 2]
        # [0, 3, 2, 1] = "abcd"
        # a at position 0, b at 1 ...
        # [0 1 2 3]
        # [0 1 2 0]
        # [0 1 1 0]
        # [0 0 0 0]
        # let the value at the index represent it's parent
        # Create a set
        parentList = [i for i in range(len(s))]
        # letterList = [[] for i in range(len(s))]
        def setRoot(parent, child):
            if parentList[parent] == child or parentList[child] == parent:
                return
            if parentList[parent] > parentList[child]:
                return setRoot(child, parent)
            if parent == child:
                return
            childParent = parentList[child]
            parentList[child] = parent
            setRoot(parent, childParent)
            
        def findParent(child, i):
            # if i == 0:
                # print(child)
            parent = parentList[child]
            if child != parentList[child]:
                parent = findParent(parentList[child], i+1)
                parentList[child] = parent
            return parent
            
        for pair in pairs:
            # print(parentList)
            # if parentList[pair[0]] < parentList[pair[1]]:
            setRoot(pair[0], pair[1])
            # else:
            #     setRoot(pair[1], pair[0])
        letterSets = {}
        
        # print(parentList)
        for i in range(len(s)):
            l = s[i]
            parent = findParent(i, 0)
            
            entry = letterSets.get(parent, [[], []])
            entry[0].append(l)
            entry[1].append(i)
            letterSets[parent] = entry
        retList = ['a' for i in range(len(s))]
        for _, entry in letterSets.items():
            entry[0].sort()
            entry[1].sort()
            for i in range(len(entry[1])):
                retList[entry[1][i]] = entry[0][i]
        return "".join(retList)
            
            
par=[0]*(100005)
rank=[1]*(100005)

def find(x):
    if par[x]!=x:
        par[x]=find(par[x])
    return par[x]
def union(x,y):
    xs = find(x)
    ys = find(y)
    if xs!=ys:
        if rank[xs]<rank[ys]:
            par[xs]=ys
        elif rank[xs]>rank[ys]:
            par[ys]=xs
        else:
            par[ys] = xs
            rank[xs]+=1
class Solution:
    def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
        n = len(s)
        for i in range(n+3):
            rank[i]=1
            par[i]=i
        for i in pairs:
            union(i[0],i[1])
            #print(par[:5])
        #print(par[:10])
        poi=[0]*(n+3)
        #a=[0]*(n+3)
        for i in range(n):
            par[i]=find(par[i])
        ans=[[]for i in range(n+3)]
        for i in range(n):
            ans[par[i]].append(s[i])
        for i in range(n):
            ans[i].sort()
        fin=''
        for i in range(n):
            tmp = par[i]
            fin+=ans[tmp][poi[tmp]]
            poi[tmp]+=1
        return fin
            
        
        
        

class Solution:
    def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
        S=s
        N = len(S)
        graph = [[] for _ in range(N)]
        for u, v in pairs:
            graph[u].append(v)
            graph[v].append(u)
        ans = [None] * N
        
        seen = [False] * N
        for u in range(N):
            if not seen[u]:
                seen[u] = True
                stack = [u]
                component = []
                while stack:
                    node = stack.pop()
                    component.append(node)
                    for nei in graph[node]:
                        if not seen[nei]:
                            seen[nei] = True
                            stack.append(nei)
                
                component.sort()
                letters = sorted(S[i] for i in component)
                for ix, i in enumerate(component):
                    letter = letters[ix]
                    ans[i] = letter
        return "".join(ans)
        
from collections import defaultdict

class DisjointSet:
    class Node:
        def __init__(self, x):
            self.val = x
            self.parent = self
            self.rank = 0
            
    def __init__(self, node_num):
        self.val_to_node = {}
        
        for val in range(node_num):
            self.val_to_node[val] = DisjointSet.Node(val)
            
    def find(self, x):
        return self._find(self.val_to_node[x]).val
    
    def _find(self, node):
        if node.parent is node:
            return node
        
        node.parent = self._find(node.parent)
        return node.parent
    
    def union(self, val1, val2):
        root1 = self._find(self.val_to_node[val1])
        root2 = self._find(self.val_to_node[val2])
        
        if root1 is root2:
            return
        
        if root2.rank > root1.rank:
            root1, root2 = root2, root1
            
        if root1.rank == root2.rank:
            root1.rank += 1
        
        root2.parent = root1
            

class Solution:
    def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
        '''
        cba
        
        bca
        bac
        abc
        '''
        if not pairs:
            return s
        
        disjoint_set = DisjointSet(len(s))
        
        for u, v in pairs:
            disjoint_set.union(u, v)
        
        connected_components = defaultdict(list)
        
        for i in range(len(s)):
            connected_components[disjoint_set.find(i)].append(i)
        
        res = [None] * len(s)
        for group in list(connected_components.values()):
            sorted_chars = sorted([s[i] for i in group])
            
            for idx, s_i in enumerate(sorted(group)):
                res[s_i] = sorted_chars[idx]
                
        return ''.join(res)

class Solution:
    # 21:50
    """
    s = "dcab", pairs = [[0,3],[1,2],[0,2]]
    dcab -[1, 2]->dacb
    dacb -[0, 3]->bacd
    not possible anymore
    """
    def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
        res = list(s)
        adj = collections.defaultdict(set)
        for a, b in pairs:
            adj[a].add(b)
            adj[b].add(a)
            
        def dfs(i, v):
            if i in adj:
                v.append(i)
                for j in adj.pop(i):
                    dfs(j, v)
            
            
        while adj:
            i = next(iter(adj))
            v = []
            dfs(i, v)
            v = sorted(v)
            chars = sorted([s[i] for i in v])
            for i, c in enumerate(chars):
                res[v[i]] = c
                
        return ''.join(res)
    
        
                
        
class Solution:
    def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
        class UF:
            def __init__(self, n):
                self.p = list(range(n))
                self.sizes = [0 for i in range(n)]
            def union(self, x, y):
                x = self.find(x)
                y = self.find(y)
                if x == y:
                    return
                if self.sizes[x] < self.sizes[y]:
                    x, y = y, x
                self.p[y] = x
                self.sizes[x] += self.sizes[y]
            def find(self, x):
                if x != self.p[x]:
                    self.p[x] = self.find(self.p[x])
                return self.p[x]
    
        uf = UF(len(s))
        for x,y in pairs: 
            uf.union(x,y)
        m = defaultdict(list)
        for i in range(len(s)):
            m[uf.find(i)].append(s[i])
        for k in list(m.keys()):
            m[k].sort(reverse=True)
        res = []
        for i in range(len(s)):
            res.append(m[uf.find(i)].pop())
        return ''.join(res)
                

class Solution:
    def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
        class UF:
            def __init__(self, n): self.p = list(range(n))
            # set one number's parent to other number's parent
            def union(self, x, y): self.p[self.find(x)] = self.find(y)
            # find current num's root node using recursive call 
            def find(self, x):
                if x != self.p[x]:
                    self.p[x] = self.find(self.p[x])
                return self.p[x]
        
        union_find, result, group = UF(len(s)), [], defaultdict(list)
        # join the groups
        for pair in pairs:
            union_find.union(pair[0], pair[1])
         
        #for i in range(len(s)):
        #    union_find.p[i] = union_find.find(i)
        # append list of num to the parent node 
        for i in range(len(s)):
            group[union_find.find(i)].append(s[i])
        # sort the keys in the group 
        for comp_id in list(group.keys()):
            group[comp_id].sort(reverse=True)
        # using pop to append
        for i in range(len(s)):
            result.append(group[union_find.find(i)].pop())
        return ''.join(result)

class Solution:
    def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
        
        
        node = {i : i for i in range(len(s))}
        

        def find(x):
            if x != node[x]:
                node[x] = find(node[x])   
                
            return node[x]
        
        def union(x, y):
            if find(x) != find(y):
                node[find(x)] = find(y)
                
        ans = []
        m = defaultdict(list)
        
        for x, y in pairs:
            union(x, y)
        for i in range(len(s)):
            m[find(i)].append(s[i])
           
        for cid in m.keys():
            m[cid].sort(reverse=True)
        print(m.items())
            
        for  i in range(len(s)):
            ans.append(m[find(i)].pop())
            
        return "".join(ans)
            
            
                
        
        
        
        
     
from collections import defaultdict
class UF:
    
    def __init__(self,n):
        self.parent=[x for x in range(n)]
        
    def find(self,x):
        if self.parent[x]!=x:
            self.parent[x]=self.find(self.parent[x])
        return self.parent[x]
    
    def union(self,x,y):
        self.parent[self.find(x)]=self.find(y)


class Solution:       
    def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
        """
        0 1 2 3
        3 2
        2
        """
        n=len(s)
        uf=UF(n)
        for item in pairs:
            uf.union(item[0],item[1])
        
        root_dict=defaultdict(list)
        
        for i,x in enumerate(list(s)):
            root_dict[uf.find(i)].append(x)
            
        for key in root_dict:
            root_dict[key]=sorted(root_dict[key],reverse=True)
            
        output=[]
        for i,x in enumerate(list(s)):
            output.append(root_dict[uf.find(i)].pop())
        
        return "".join(output)
        
class union:
    def __init__(self):
        self.par = None
        self.rank = 0
    @staticmethod
    def parent(A):
        temp = A
        while A.par:
            A = A.par
        if temp != A:
            temp.par = A
        return A
    @staticmethod
    def fun(A,B):
        pA = union.parent(A)
        pB = union.parent(B)
        if pA != pB:
            if pA.rank > pB.rank:
                pB.par = pA
                pA.rank+=1
                return pB
            pA.par = pB
            pB.rank+=1
            return pA
        return None
            
class Solution:
    def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
        s = list(s)
        dic = {}
        par = set()
        for i in range(len(s)):
            temp =  union()
            dic[i] = temp
            par.add(temp)
        for i in pairs:
            temp = union.fun(dic[i[0]],dic[i[1]])
            if temp in par:
                par.remove(temp)
        
        if len(par) == 1:
            s.sort()
            return "".join(s)
        new = {}
        for i in par:
            new[i] = [[],0]
        for i in range(len(s)):
            par = union.parent(dic[i])
            new[par][0].append(s[i])
        for i in new.values():
            i[0].sort()
        ans =[]
        for i in range(len(s)):
            par = union.parent(dic[i])
            ind = new[par][1]
            new[par][1]+=1
            ans.append(new[par][0][ind])
        return "".join(ans)
        
        
class Solution:
    def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
        n = len(s)
        edges = defaultdict(list)
        
        for p in pairs:
            edges[p[0]].append(p[1])
            edges[p[1]].append(p[0])
        visited = [False for i in range(n)]
        
        def dfs(node):
            visited[node] = True
            component.append(node)
            for neighbor in edges[node]:
                if not visited[neighbor]:
                    dfs(neighbor)
                    
        connected_components = []
        for i in range(n):
            if not visited[i]:
                component = []
                dfs(i)
                connected_components.append(component)
        #print(connected_components)
        ans = [" " for i in range(n)]
        for component in connected_components:
            indexes = sorted(component)
            chars = [s[i] for i in indexes]
            chars = sorted(chars)
            for i in range(len(indexes)):
                ans[indexes[i]] = chars[i]
        #print(ans)
        return ''.join(ans)
class Solution:
    def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
        uf = {}
        
        def find(x):
            uf.setdefault(x, x)
            if uf[x] != x:
                uf[x] = find(uf[x])
            return uf[x]
        
        def union(x, y):
            uf[find(x)] = find(y)
            
        for a, b in pairs:
            union(a, b)
            
        ans = [""] * len(s)
        dic = collections.defaultdict(list)
        for i, ss in enumerate(s):
            dic[find(i)].append((ss, i))
        print(dic)
        for d in dic:
            for ss, i in zip(sorted(ss for ss, i in dic[d]), sorted(i for ss, i in dic[d])):
                ans[i] = ss
        return "".join(ans)
                
                
            
            
            
            
            
            
            
class Solution:
    def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
        l = len(s)
        union_find = [i for i in range(l)]
        union_set = {}
        for i in range(l):
            union_set[i] = [i]
        for p in pairs:
            x = p[0]
            y = p[1]
            t, t1 = union_find[y], union_find[x]
            if union_find[x] < union_find[y]:
                t, t1 = union_find[x], union_find[y]
            if t == t1:
                continue
            for i in union_set[t1]:
                union_set[t].append(i)
                union_find[i] = t
            union_set[t1] = []
        print(union_find)
        print(union_set)
        res_cand = {}
        for k, v in list(union_set.items()):
            union_set = []
            for i in v:
                union_set.append(s[i])
            union_set.sort(reverse=True)
            res_cand[k] = union_set

        print(res_cand)
        res = ''
        for i in range(l):
            res = res + res_cand[union_find[i]][-1]
            res_cand[union_find[i]].pop()
        return res

class Solution:
    def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
        def dfs(u, cc):
            cc.append(u)
            visited_set.add(u)
            if u in graph:
                for v in graph[u]:
                    if v not in visited_set:
                        dfs(v, cc)

        if len(pairs) == 0:
            return s

        graph = {}
        for u,v in pairs:
            if u in graph:
                graph[u].append(v)
            else:
                graph[u] = [v]
            if v in graph:
                graph[v].append(u)
            else:
                graph[v] = [u]

        result = [c for c in s]
        visited_set = set()
        for u in graph:
            if u not in visited_set:
                cc = []
                dfs(u, cc)
                cc.sort()
                auxr = [s[i] for i in cc]
                auxr.sort()
                for i, index in enumerate(cc):
                    result[index] = auxr[i]
        return ''.join(result)
class Solution:
    def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
        def dfs(u, cc):
            visited_set[u] = cc
            if u in graph:
                for v in graph[u]:
                    if visited_set[v] == 0:
                        dfs(v, cc)

        if len(pairs) == 0:
            return s

        graph = {}
        for u,v in pairs:
            if u in graph:
                graph[u].append(v)
            else:
                graph[u] = [v]
            if v in graph:
                graph[v].append(u)
            else:
                graph[v] = [u]

        visited_set = [0 for i in range(len(s))]
        cc = 0
        for u in graph:
            if visited_set[u] == 0:
                cc += 1
                dfs(u, cc)

        dd = defaultdict(list)
        result = [c for c in s]
        for i,key in enumerate(visited_set):
            if key != 0:
                dd[key].append(s[i]);
        for key in dd:
            dd[key].sort(reverse=True)
        for i, key in enumerate(visited_set):
            if key != 0:
                result[i] = dd[key].pop()

        return ''.join(result)
class Solution:
    def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
        g = collections.defaultdict(list)
        for a, b in pairs:
            g[a].append(b)
            g[b].append(a)
        
        def find(i, idx):
            for k in g[i]:
                if not visited[k]:
                    visited[k] = 1
                    idx.append(k)
                    find(k, idx)
        n = len(s)
        s = list(s)
        visited = [0] * n
        for i in range(n):
            if visited[i]:
                continue
            visited[i] = 1
            idx = [i]
            find(i, idx)
            idx.sort()
            chars = [s[j] for j in idx]
            chars.sort()
            for j, c in zip(idx, chars):
                s[j] =  c
        return ''.join(s)
# class Solution:
#     def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
#             class UF:
#                 def __init__(self, n): self.p = list(range(n))
#                 def union(self, x, y): self.p[self.find(x)] = self.find(y)
#                 def find(self, x):
#                     if x != self.p[x]: self.p[x] = self.find(self.p[x])
#                     return self.p[x]
#             uf, res, m = UF(len(s)), [], defaultdict(list)
#             for x,y in pairs: 
#                 uf.union(x,y)
#             for i in range(len(s)): 
#                 m[uf.find(i)].append(s[i])
#             for comp_id in m.keys(): 
#                 m[comp_id].sort(reverse=True)
#             for i in range(len(s)): 
#                 res.append(m[uf.find(i)].pop())
#             return ''.join(res)

from collections import defaultdict

class Solution:
    def find(self,x):
        if(x!=self.parent[x]):
            self.parent[x]=self.find(self.parent[x])
        return self.parent[x]
        
        
    def union(self,x,y):
        x_find=self.find(x)
        y_find=self.find(y)
        self.parent[x_find]=y_find
        
    
    
    def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
        n=len(s)
        self.parent=list(range(n))
        
        for x,y in pairs:
            self.union(x,y)
        
        # print(self.parent)
        
        groups=defaultdict(list)
        for i in range(n):
            tem=self.find(i)
            # self.parent[i]=tem
            groups[tem].append(s[i])    
            # print(tem)
        # print(self.parent)
        
        ans=[]
        for comp_id in groups.keys(): 
            groups[comp_id].sort(reverse=True)
            
        # print(groups)
        
        for i in range(n): 
            ans.append(groups[self.find(i)].pop())
        return "".join(ans)
        
from collections import defaultdict
class Solution:
    def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
        def dfs(x):
            visited.add(x)
            component.append(x)
            for ele in adjacencyList[x]:
                if ele not in visited:
                    dfs(ele)
            
        adjacencyList = [[] for x in range(len(s))]
        for pair in pairs:
            # print(pair, adjacencyList, len(s))
            adjacencyList[pair[0]].append(pair[1])
            adjacencyList[pair[1]].append(pair[0])
        
        
        visited = set()
        ans = list(s)
        for x in range(len(s)):
            if x not in visited:
                component = []
                dfs(x)
                lst = []
                component.sort()
                for y in component:
                    lst.append(s[y])
                lst.sort()
                i = 0
                for y in component:
                    ans[y] = lst[i]
                    i+=1
        return "".join(ans)
                    
                
        
    
class Solution:
    def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
        class UF:
            def __init__(self, n): self.p = list(range(n))
            def union(self, x, y): self.p[self.find(x)] = self.find(y)
            def find(self, x):
                if x != self.p[x]:
                    self.p[x] = self.find(self.p[x])
                return self.p[x]
        
        union_find, result, group = UF(len(s)), [], defaultdict(list)
        for pair in pairs:
            union_find.union(pair[0], pair[1])
        for i in range(len(s)):
            union_find.p[i] = union_find.find(i)
        for i in range(len(s)):
            group[union_find.find(i)].append(s[i])
        for comp_id in list(group.keys()):
            group[comp_id].sort(reverse=True)
        for i in range(len(s)):
            result.append(group[union_find.find(i)].pop())
        return ''.join(result)

class Solution:
    def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
        def dfs(u, cc):
            visited_set[u] = cc
            if u in graph:
                for v in graph[u]:
                    if visited_set[v] == 0:
                        dfs(v, cc)

        if len(pairs) == 0:
            return s

        graph = {}
        for u,v in pairs:
            if u in graph:
                graph[u].append(v)
            else:
                graph[u] = [v]
            if v in graph:
                graph[v].append(u)
            else:
                graph[v] = [u]

        visited_set = [0 for i in range(len(s))]
        cc = 0
        for u in graph:
            if visited_set[u] == 0:
                cc += 1
                dfs(u, cc)

        dd = {}
        result = [c for c in s]
        for i,key in enumerate(visited_set):
            if key != 0:
                if key in dd:
                    heapq.heappush(dd[key], s[i]);
                else:
                    dd[key] = [s[i]]
        for i, key in enumerate(visited_set):
            if key != 0:
                result[i] = heapq.heappop(dd[key])

        return ''.join(result)
class Solution:
    def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
        def dfs(root, res):
            if letters[root] == 1:
                return
            letters[root] = 1
            res.append(root)
            for node in graph[root]:
                dfs(node, res)
                
        letters = [0]*len(s)
        graph = [[] for _ in range(len(s))]
        for a, b in pairs:
            graph[a].append(b)
            graph[b].append(a)
        res = list(s)
        for i in range(len(s)):
            if letters[i] == 0:
                visited = []
                dfs(i, visited)
                nodes = []
                for node in visited:
                    nodes.append(res[node])
                nodes.sort()
                visited.sort()
                # print(nodes)
                # print(visited)
                for node, index in zip(nodes, visited):
                    res[index] = node
        return ''.join(res)
class Solution:
    def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
        def dfs(u, cc):
            visited_set[u] = cc
            if u in graph:
                for v in graph[u]:
                    if visited_set[v] == 0:
                        dfs(v, cc)

        if len(pairs) == 0:
            return s

        graph = {}
        for u,v in pairs:
            if u in graph:
                graph[u].append(v)
            else:
                graph[u] = [v]
            if v in graph:
                graph[v].append(u)
            else:
                graph[v] = [u]

        visited_set = [0 for i in range(len(s))]
        cc = 0
        for u in graph:
            if visited_set[u] == 0:
                cc += 1
                dfs(u, cc)

        dd = {}
        result = [c for c in s]
        for i,key in enumerate(visited_set):
            if key != 0:
                if key in dd:
                    dd[key].append(s[i]);
                else:
                    dd[key] = [s[i]]
        for key in dd:
            dd[key].sort(reverse=True)
        for i, key in enumerate(visited_set):
            if key != 0:
                result[i] = dd[key].pop()

        return ''.join(result)
class Solution:
    def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
        graph = defaultdict(list)
        for node, neigbor in pairs:
            graph[node].append(neigbor)
            graph[neigbor].append(node)

        visited = set()

        def dfs(index):
            if index not in visited:
                visited.add(index)
                newgroup.add(index)
                for neigbor in graph[index]:
                    dfs(neigbor)

        n = len(s)
        result = [None] * n
        for i in range(n):
            newgroup = set()
            dfs(i)
            subseq = [s[i] for i in newgroup]
            subseq.sort()
            for letter, index in zip(subseq, sorted(newgroup)):
                result[index] = letter        

        return ''.join(result)
from heapq import *
class Solution:
    def smallestStringWithSwaps(self, s: str, prs: List[List[int]]) -> str:
        f, m, ans = {}, defaultdict(list), []
        for p in prs:
            r_a, r_b = self.fnd(f, p[0]), self.fnd(f, p[1])
            if r_a != r_b:
                f[r_b] = r_a
        
        for i in range(len(s)):
            m[self.fnd(f, i)].append(s[i])
        for v in list(m.values()):
            heapify(v)
        for i in range(len(s)):
            ans.append(heappop(m[self.fnd(f, i)]))
        return ''.join(ans)
    
    def fnd(self, f, n):
        f[n] = f.get(n, n)
        if f[n] == n:
            return n
        f[n] = self.fnd(f, f[n])
        
        return f[n]

import collections

class Solution:
  def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
    swap_sets = [i for i in range(len(s))]
    
    for [a,b] in pairs:
      if not self.find(a,b,swap_sets):
        self.union(a,b, swap_sets)
        
    swap_sets = [self.root(s, swap_sets) for s in swap_sets]
    
    groups = collections.defaultdict(list)
    for i, charIndex in enumerate(swap_sets):
      groups[charIndex].append(i)
    
    for k,v in groups.items():
      chars = [s[i] for i in v]
      groups[k] = sorted(v), sorted(chars)
      
    subs = [v for k,v in groups.items()]
    newStr = ['' for i in range(len(s))]
    for indices, letters in subs:
      for i in range(len(indices)):
        newStr[indices[i]] = letters[i]
        
    return "".join(newStr)
      
      
  def union(self,a,b, roots):
    ra, rb = self.root(a, roots), self.root(b, roots)
    roots[rb] = ra
    
  def root(self,a, roots):
    while(roots[a] != a):
      roots[a] = roots[roots[a]]
      a = roots[a]
    return a
  
  def find(self,a,b, roots):
    return self.root(a, roots) == self.root(b, roots)
    
    
      
class DSU:
    def __init__(self, n):
        self.p = list(range(n))
    
    def find(self, x):
        if(self.p[x] != x):
            self.p[x] = self.find(self.p[x])
        return self.p[x]
    
    def union(self, x, y):
        xr = self.find(x)
        yr = self.find(y)
        self.p[xr] = yr
        
class Solution:
    def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
        N = len(s)
        dsu = DSU(N)
        
        for x, y in pairs:
            dsu.union(x, y)
            
        dic = collections.defaultdict(list)
        for i in range(N):
            k = dsu.find(i)
            dic[k].append(i)
        
        res = [' ']*N
        for v in dic.values():
            for i, j in zip(v, sorted(v, key=lambda idx:s[idx])):
                res[i] = s[j]
        return "".join(res)
class Solution:
    def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
        def dfs(u, cc):
            visited_set[u] = cc
            if u in graph:
                for v in graph[u]:
                    if visited_set[v] == 0:
                        dfs(v, cc)

        if len(pairs) == 0:
            return s

        graph = {}
        for u,v in pairs:
            if u in graph:
                graph[u].append(v)
            else:
                graph[u] = [v]
            if v in graph:
                graph[v].append(u)
            else:
                graph[v] = [u]

        visited_set = [0 for i in range(len(s))]
        cc = 0
        for i in range(len(s)):
            if visited_set[i] == 0:
                cc += 1
                dfs(i, cc)

        dd = {}
        for i,key in enumerate(visited_set):
            if key in dd:
                heapq.heappush(dd[key], s[i]);
            else:
                dd[key] = [s[i]]

        return ''.join(heapq.heappop(dd[key]) for key in visited_set)
from collections import defaultdict
class Solution:
    def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
        p = list(range(len(s)))
        # r = [1]*len(s)
        d = defaultdict(list)
        def find(x):
            if x!=p[x]:
                p[x]=find(p[x])
            return p[x]
        
        def union(x,y):
            x,y = find(x), find(y)
            if x==y:
                return False
            p[x]=y
            
        for a,b in pairs:
            union(a,b)
        for i in range(len(p)):
            d[find(i)].append(s[i])
        for i in d:
            d[i].sort(reverse=True)
        ret=''
        for i in range(len(s)):
            ret+=d[find(i)].pop()
        return ret
class UnionFind(object):
    def __init__(self,n):
        self._parent = [0]*n
        self._size = [1]*n
        self.count = n
        for i in range(n):
            self._parent[i] = i
            
    def union(self, p, q):
        rootp = self.find(p)
        rootq = self.find(q)
        if rootp == rootq:
            return
        self.count -= 1
        if self._size[rootp] > self._size[rootq]:
            self._size[rootp] += self._size[rootq]
            self._parent[rootq] = self._parent[q] = rootp
        else:
            self._size[rootq] += self._size[rootp]
            self._parent[rootp] = self._parent[p] = rootq
    
    def find(self, n):
        while self._parent[n] != n:
            self._parent[n] = self._parent[self._parent[n]]
            n = self._parent[n]
        return n
    
    def connected(self, p, q):
        return self.find(p) == self.find(q)
    
class Solution:
    def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
        N = len(s)
        if N < 2: return s
        uf = UnionFind(N)
        for pair in pairs:
            uf.union(pair[0],pair[1])
        
        dict = defaultdict(list)
        for i in range(N):
            r = uf.find(i)
            dict[r].append(i)
            
        res = [' ']*N
        for lst in list(dict.values()):
            lst.sort()
            subs = ''
            for idx in lst:
                subs += s[idx]
            s2 = sorted(subs)
            i2 = 0
            for idx in lst:
                res[idx]=s2[i2]
                i2+=1
        return ''.join(res)
            
        
        
            

class Solution:
    def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
        
        def find(x):
            parent.setdefault(x,x)
            if x!=parent[x]:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x,y):
            px = find(x)
            py = find(y)
            parent[px] = py
            
        
        parent = dict()
        
        
        for i,j in pairs:
            # if i==j:
            #     continue
            x = i
            y = j
            px = find(x)
            py = find(y)
            if px!=py:
                union(x,y)
        
        graph = collections.defaultdict(list)
        
        
        
        for i in range(len(s)):
            px = find(i)
            heapq.heappush(graph[px],s[i])
            #bisect.insort(graph[px],s[i])
        # for char in s:
        #     px = find()
        #     bisect.insort(graph[px],char)
        
        print(graph)
        
#         res = ''
#         for char in s:
#             px = find(char)
#             res += graph[px].pop(0) # this is o(n)
#         return(res)
        res = ''
        mem = collections.defaultdict(int)
        for i in range(len(s)):
            px = find(i)
            
            res += heapq.heappop(graph[px])
            #res += graph[px][mem[px]] # this is o(n)
            #mem[px]+=1
        return res

class Solution:
    def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
        def dfs(u, cc):
            cc.append(u)
            visited_set.add(u)
            if u in graph:
                for v in graph[u]:
                    if v not in visited_set:
                        dfs(v, cc)

        M = len(pairs)
        if M == 0:
            return s

        graph = defaultdict(list)
        for u,v in pairs:
            graph[u].append(v)
            graph[v].append(u)

        result = [c for c in s]
        visited_set = set()
        for u in graph:
            if u not in visited_set:
                cc = []
                dfs(u, cc)
                cc.sort()
                auxr = [s[i] for i in cc]
                auxr.sort()
                for i, index in enumerate(cc):
                    result[index] = auxr[i]
        return ''.join(result)
class Solution:
    def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
        
        def find(x):
            parent.setdefault(x,x)
            if x!=parent[x]:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x,y):
            px = find(x)
            py = find(y)
            parent[px] = py
            
        
        parent = dict()
        
        
        for i,j in pairs:
            if i==j:
                continue
            x = i
            y = j
            px = find(x)
            py = find(y)
            if px!=py:
                union(x,y)
        
        graph = collections.defaultdict(list)
        
        for i in range(len(s)):
            px = find(i)
            heapq.heappush(graph[px],s[i])
           
  
        res = ''
        mem = collections.defaultdict(int)
        for i in range(len(s)):
            px = find(i)
            res += heapq.heappop(graph[px])
        return res

class Solution:
    def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
        
        def find(x):
            #parent.setdefault(x,x)
            #print(x)
            if x!=parent[x]:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x,y):
            px = find(x)
            py = find(y)
            parent[px] = py
            
        
        parent = list(range(len(s)+1))
        
        
        for i,j in pairs:
            if i==j:
                continue
            x = i
            y = j
            px = find(x)
            py = find(y)
            if px!=py:
                union(x,y)
        
        graph = collections.defaultdict(list)
        
        for i in range(len(s)):
            px = find(i)
            heapq.heappush(graph[px],s[i])
           
  
        res = ''
        mem = collections.defaultdict(int)
        for i in range(len(s)):
            px = find(i)
            res += heapq.heappop(graph[px])
        return res

class DisjointSet:
    
    def __init__(self):
        self.parent = dict()
    
    def union(self, a, b):
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a == root_b:
            return False
        self.parent[root_a] = root_b
        return True
    
    def find(self, a):
        if a not in self.parent:
            self.parent[a] = a
            return a
        
        if a != self.parent[a]:
            self.parent[a] = self.find(self.parent[a])
        return self.parent[a]
        

class Solution:
    def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
        ds = DisjointSet()
        
        for u, v in pairs:
            ds.union(u, v)
        
        hashmap = collections.defaultdict(list)
        for i, c in enumerate(s):
            root = ds.find(i)
            hashmap[root].append((c, i))
        
        for root in hashmap:
            hashmap[root].sort(reverse=True)
        
        ans = []
        for i in range(len(s)):
            root = ds.find(i)
            c, _ = hashmap[root].pop()
            ans.append(c)
        return "".join(ans)
        
class Solution:
    # 21:50
    """
    s = "dcab", pairs = [[0,3],[1,2],[0,2]]
    dcab -[1, 2]->dacb
    dacb -[0, 3]->bacd
    not possible anymore
    """
    def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
        res = list(s)
        groups = self.groups(pairs, len(s))
        for g in groups:
            chars = sorted([s[i] for i in g])
            j = 0
            for i in g:
                res[i] = chars[j]
                j += 1
                
        return ''.join(res)
    
    def groups(self, pairs, n):
        adj = collections.defaultdict(set)
        for p in pairs:
            adj[p[0]].add(p[1])
            adj[p[1]].add(p[0])
            
        arr = []
        while adj:
            i = list(adj.keys())[0]
            v = []
            self.dfs(i, adj, v)
            arr.append(sorted(list(v)))
                
        return arr
    
    def dfs(self, i, adj, res):
        if i in adj:
            res.append(i)
            for j in adj.pop(i):
                self.dfs(j, adj, res)
        
                
        
from collections import defaultdict

class Solution:
    def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
        array = [char for char in s]
        edges = defaultdict(list)
        for i, j in pairs:
            edges[i].append(j)
            edges[j].append(i)
        parents = {}
        components = defaultdict(list)
        for i in range(len(pairs)):
            for j in edges[i]:
                self.union(i, j, parents)
        for child, parent in list(parents.items()):
            components[self.find(parent, parents)].append(child)
        print(parents)
        print(components)
        for parent, children in list(components.items()):
            tmp = sorted([array[child] for child in children])
            print(tmp)
            components[parent] = sorted(children)
            for k in range(len(children)):
                array[components[parent][k]] = tmp[k]
        return ''.join(array)

        
    
    def union(self, i, j, parents):
        if not parents.get(i):
            parents[i] = i
        if not parents.get(j):
            parents[j] = j
        pi, pj = self.find(i, parents), self.find(j, parents)
        parents[pj] = pi
        parents[i] = pi
    
    def find(self, i, parents):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return parents[i]
        
                
                
                    
            
    
        
        
                    
                    
            
        

from collections import defaultdict
class Solution:
    def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
        
        def find(node):
            path = []
            while root[node] != node:
                path.append(node)
                node = root[node]
            
            for n in path:
                root[n] = node
            return node
        
        
        def union(a, b):
            r1, r2 = find(a), find(b)
            if r1 != r2:
                root[r1] = r2
        
        
        root = {i: i for i in range(len(s))}
        for a, b in pairs:
            union(a, b)
            
        root_to_char = defaultdict(list)
        for k in list(root.keys()):
            root_to_char[find(k)].append(s[k])
            
        for v in list(root_to_char.values()):
            v.sort(reverse=True)
        
        res = []
        for i in range(len(s)):
            res.append(root_to_char[root[i]].pop())
        
        return ''.join(res)

from collections import defaultdict
class Solution:
    def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
        p = list(range(len(s)))
        d = defaultdict(list)
        def find(x):
            if p[x]!=x:
                p[x]=find(p[x])
            return p[x]
        def union(x,y):
            x,y = find(x),find(y)
            p[x]=y
            return p[x] 
        for a,b in pairs:
            union(a,b)
        for i in range(len(p)):
            d[find(i)].append(s[i])
        for x in d:
            d[find(x)].sort(reverse=True)
        ret=''
        for i in range(len(s)):
            ret+=d[find(i)].pop()
        return ret
class Solution:
    def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
        
        def dfs(node):
            seen.add(node)
            idx.append(node)
            ch.append(s[node])
            
            for nei in graph[node]:
                if nei not in seen:
                    dfs(nei)
            
        
        
        graph = collections.defaultdict(list)
        
        for u, v in pairs:
            graph[u].append(v)
            graph[v].append(u)
        
        seen = set()
        idxCh = [""]*len(s)
        
        for i in range(len(s)):
            if i not in seen:
                idx = []
                ch = []
                dfs(i)
            for i, c in zip(sorted(idx), sorted(ch)):
                idxCh[i] = c
        return "".join(idxCh)
            
        
     #    0 - 3
     #    |
     #1 - 2
    
    # [0,3] bd
    # [1,2] ac
    
    
# class Solution:
#     def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
#             class UF:
#                 def __init__(self, n): self.p = list(range(n))
#                 def union(self, x, y): self.p[self.find(x)] = self.find(y)
#                 def find(self, x):
#                     if x != self.p[x]: self.p[x] = self.find(self.p[x])
#                     return self.p[x]
#             uf, res, m = UF(len(s)), [], defaultdict(list)
#             for x,y in pairs: 
#                 uf.union(x,y)
#             for i in range(len(s)): 
#                 m[uf.find(i)].append(s[i])
#             for comp_id in m.keys(): 
#                 m[comp_id].sort(reverse=True)
#             for i in range(len(s)): 
#                 res.append(m[uf.find(i)].pop())
#             return ''.join(res)

from collections import defaultdict

class Solution:
    def find(self,x):
        if(x!=self.parent[x]):
            self.parent[x]=self.find(self.parent[x])
        return self.parent[x]
        
        
    def union(self,x,y):
        x_find=self.find(x)
        y_find=self.find(y)
        self.parent[x_find]=y_find
        
    
    
    def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
        n=len(s)
        self.parent=list(range(n))
        
        for x,y in pairs:
            self.union(x,y)
        
        # print(self.parent)
        
        groups=defaultdict(list)
        for i in range(n):
            tem=self.find(i)
            # self.parent[i]=tem
            groups[tem].append(s[i])    
            # print(tem)
        # print(self.parent)
        
        ans=[]
        for comp_id in groups.keys(): 
            groups[comp_id].sort(reverse=True)
            
        # print(groups)
        
        for i in range(n): 
            ans.append(groups[self.find(i)].pop())
        return "".join(ans)
        
        
# # #         for i in range(len(s)):
# # #             if(i not in added):
# # #                 groups[i]=[i]
        
# #         # print(groups)
# #         ls=dict()
# #         for i,j in groups.items():
# #             ls[tuple(j)]=sorted([s[ele] for ele in j])
# #         # print(ls)
        
# #         ans=""
# #         for i in range(len(s)):
# #             ans+=ls[tuple(groups[self.parent[i]])].pop(0)
        
# #         return ans
                
        
            
        
        
        
        
# # #         self.ans=s
# # #         visited=set()
# # # #         def traverse(st,pair,i):
# # # #             print(st,i)
# # # #             if(st in visited):
# # #                 return
# # #             visited.add(st)
# # #             a,b=pair[i][0],pair[i][1]
# # #             st=list(st)
# # #             st[a],st[b]=st[b],st[a]
# # #             st="".join(st)
# # #             self.ans=min(self.ans,st)
# # #             # tem=st[:]
# # #             for j in range(len(pair)):
# # #                 if(i!=j):
# # #                     traverse(st,pair,j)
        
        
        
        
# #             # traverse(s,pairs,i)
        
# #         q=[s]
# #         while(q!=[]):
# #             tem=q.pop(0)
# #             if(tem in visited):
# #                 continue
# #             visited.add(tem)
# #             self.ans=min(self.ans,tem)
# #             for i in range(len(pairs)):
# #                 a,b=pairs[i][0],pairs[i][1]
# #                 tem=list(tem)
# #                 tem[a],tem[b]=tem[b],tem[a]
# #                 tem="".join(tem)
# #                 q.append(tem)
            
        
# #         return self.ans
class Solution:
    def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
        def dfs(i):
            visited[i] = True
            component.append(i)
            for j in adj_list[i]:
                if not visited[j]:
                    dfs(j)
        n = len(s)
        adj_list = [[] for _ in range(n)]
        for i, j in pairs:
            adj_list[i].append(j)
            adj_list[j].append(i)
        visited = [False for _ in range(n)]
        s = list(s)
        for i in range(n):
            if not visited[i]:
                component = []
                dfs(i)
                component.sort()
                chars = [s[k] for k in component]
                chars.sort()
                for i in range(len(component)):
                    s[component[i]] = chars[i]
        return ''.join(s)

class UnionFind:
    def __init__(self, n):
        self.parents = [i for i in range(n)]
        self.rank = [0] * n
        
    def find(self, x):
        if self.parents[x] != x:
            self.parents[x] = self.find(self.parents[x])
        return self.parents[x]
    
    def union(self, x, y):
        self.parents[self.find(x)] = self.find(y)

class Solution:
    def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
        UF = UnionFind(len(s))
        res = []
        for x, y in pairs:
            UF.union(x, y)
        parent_to_heap = collections.defaultdict(list)
        for i, p in enumerate(UF.parents):
            heapq.heappush(parent_to_heap[UF.find(p)], s[i])
        for i, p in enumerate(UF.parents):
            res.append(heapq.heappop(parent_to_heap[UF.find(p)]))
        return ''.join(res)
class Solution:
    # 21:50
    """
    s = "dcab", pairs = [[0,3],[1,2],[0,2]]
    dcab -[1, 2]->dacb
    dacb -[0, 3]->bacd
    not possible anymore
    """
    def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
        res = list(s)
        groups = self.groups(pairs, len(s))
        for g in groups:
            chars = sorted([s[i] for i in g])
            j = 0
            for i in g:
                res[i] = chars[j]
                j += 1
                
        return ''.join(res)
    
    def groups(self, pairs, n):
        adj = collections.defaultdict(set)
        for p in pairs:
            adj[p[0]].add(p[1])
            adj[p[1]].add(p[0])
            
        arr = []
        while adj:
            i = list(adj.keys())[0]
            v = []
            self.dfs(i, adj, v)
            arr.append(sorted(v))
                
        return arr
    
    def dfs(self, i, adj, res):
        if i in adj:
            res.append(i)
            for j in adj.pop(i):
                self.dfs(j, adj, res)
        
                
        
class UF:
    def __init__(self, N):
        self.N = N
        self.parent = [i for i in range(N)]
        self.rank = [0 for i in range(N)]
        
    def union(self, a ,b):
        ra = self.find(a)
        rb = self.find(b)
        if ra != rb:
            self.parent[ra] = rb
    
    def find(self, a):
        while a!= self.parent[a]:
            a = self.parent[a]
        return a

class Solution:
    def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
        class UF:
            def __init__(self, n): self.p = list(range(n))
            def union(self, x, y): self.p[self.find(x)] = self.find(y)
            def find(self, x):
                if x != self.p[x]: self.p[x] = self.find(self.p[x])
                return self.p[x]
        uf, res, m = UF(len(s)), [], defaultdict(list)
        for x,y in pairs: 
            uf.union(x,y)
        for i in range(len(s)): 
            m[uf.find(i)].append(s[i])
        for comp_id in list(m.keys()): 
            m[comp_id].sort(reverse=True)
        for i in range(len(s)): 
            res.append(m[uf.find(i)].pop())
        return ''.join(res)
        
        '''
        cbad
        [[0,3],[1,2],[0,2]]
        
        '''

# this is version after checking the discussions
class Solution:
    def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
        class unionFind:
            def __init__(self, n):
                #here index = component id, initiallly point to eachself
                self.parents = list(range(n))
                
            def union (self, x, y):
                # make y's parent x's
                self.parents[self.find(x)] = self.find(y)
                
            def find (self, x):
                # parent's parent == themself
                # if this x is not a parent, find and record his final parent
                if self.parents[x]!=x: self.parents[x] = self.find(self.parents[x])
                return self.parents[x]
            
        uf1 = unionFind(len(s))
        
        for x,y in pairs:
            uf1.union(x,y)
        # every index of parents (implied index of s) found his parents now
        
        # because old parents won't be updated if their parent have a new parent
        # do find to all indices to make sure the parents list is clean
        for i in range(len(s)):
            uf1.find(i)
        
        groupList = defaultdict(list)
        
        # make indices with same parents group together, get char directly from string
        for i in range(len(s)):
            groupList[uf1.parents[i]].append(s[i])
        
        # sort each group (list), reversely
        for key in groupList.keys():
            groupList[key].sort(reverse = True)
        
        #pop out char from behind of each group
        result = []
        for i in range(len(s)):
            result.append(groupList[uf1.parents[i]].pop())
        
        # the way convert list to str
        return "".join(result)
        
        
class Solution:
    def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
        p = list(range(len(s)))
        d = defaultdict(list)
        def find(x):
            if x!=p[x]:
                p[x]=find(p[x])
            return p[x]
        def union(x,y):
            p[find(x)]=find(y)
        for a,b in pairs:
            union(a,b)
        for i in range(len(p)):
            d[find(i)].append(s[i])
        for x in d:
            d[find(x)].sort(reverse=True)
        ret=''
        for i in range(len(s)):
            ret+=d[find(i)].pop()
        return ret
