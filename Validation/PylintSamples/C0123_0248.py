class Solution:
    def containsCycle(self, grid: List[List[str]]) -> bool:
        n = len(grid)
        m = len(grid[0])
        
        F = [i for i in range(m * n)]
        def find(x):
            if x == F[x]:
                return x
            else:
                F[x] = find(F[x])
                return F[x]
            
        for i in range(n):
            for j in range(m):
                if i > 0 and grid[i-1][j] == grid[i][j]:
                    f1 = find((i-1)*m+j)
                    f2 = find((i)*m+j)
                    if f1 == f2:
                        return True
                    F[f1] = f2
                if j > 0 and grid[i][j-1] == grid[i][j]:
                    f1 = find((i)*m+j-1)
                    f2 = find((i)*m+j)
                    if f1 == f2:
                        return True
                    F[f1] = f2
        return False

class Solution:
    def containsCycle(self, grid: List[List[str]]) -> bool:
        visited = set()
        m = len(grid)
        if m==1: return False
        n = len(grid[0])
        dirs = [(0,-1),(-1,0),(0,1),(1,0)]
        def dfs(prev, curr):
            if curr in visited: return True
            visited.add(curr)
            for dirn in dirs:
                nei = (dirn[0]+curr[0], dirn[1]+curr[1])
                if 0<=nei[0]<m and 0<=nei[1]<n and nei != prev and grid[nei[0]][nei[1]] == grid[curr[0]][curr[1]]:
                    if dfs(curr, nei): return True
            return False
        for i in range(m):
            for j in range(n):
                if (i,j) not in visited:
                    if dfs(None, (i,j)): return True
        return False



import collections
class Solution:
    def containsCycle(self, grid: List[List[str]]) -> bool:
        def find(pos):
            if parents[pos] != pos:
                parents[pos] = find(parents[pos])
            return parents[pos]

        def union(pos1, pos2):
            parent1, parent2 = find(pos1), find(pos2)
            if parent1 != parent2:
                if ranks[parent2] > ranks[parent1]:
                    parents[parent1] = parent2
                else:
                    parents[parent2] = parent1
                    if ranks[parent1] == ranks[parent2]:
                        ranks[parent1] += 1

        rows, cols = len(grid), len(grid[0])
        parents = {(i, j): (i, j) for i in range(rows) for j in range(cols)}
        ranks = collections.Counter()
        for i, row in enumerate(grid):
            for j, letter in enumerate(row):
                if i > 0 and j > 0 and grid[i-1][j] == grid[i][j-1] == letter and find((i-1, j)) == find((i, j-1)):
                    return True
                for r, c in (i - 1, j), (i, j - 1):
                    if 0 <= r < rows and 0 <= c < cols and grid[r][c] == letter:
                        union((i, j), (r, c))
        return False
class Solution:
    def containsCycle(self, grid: List[List[str]]) -> bool:
        def find(pos):
            if parents[pos] != pos:
                parents[pos] = find(parents[pos])
            return parents[pos]

        def union(pos1, pos2):
            parent1, parent2 = find(pos1), find(pos2)
            if parent1 != parent2:
                if ranks[parent2] > ranks[parent1]:
                    parents[parent1] = parent2
                else:
                    parents[parent2] = parent1
                    if ranks[parent1] == ranks[parent2]:
                        ranks[parent1] += 1

        rows, cols = len(grid), len(grid[0])
        parents = {(i, j): (i, j) for i in range(rows) for j in range(cols)}
        ranks = collections.Counter()
        for i, row in enumerate(grid):
            for j, letter in enumerate(row):
                if i > 0 and j > 0 and grid[i-1][j] == grid[i][j-1] == letter and find((i-1, j)) == find((i, j-1)):
                    return True
                for r, c in (i - 1, j), (i, j - 1):
                    if 0 <= r < rows and 0 <= c < cols and grid[r][c] == letter:
                        union((i, j), (r, c))
        return False
class UnionFind:
    def __init__(self,n):
        self.parent = [i for i in range(n)]
        self.size = [1]*n
    def find(self,A):
        root=A
        while root!=self.parent[root]:
            root=self.parent[root]
            
        while A!=root:
            old_parent=self.parent[A]
            self.parent[A]=root
            A=old_parent
        return(root)
    
    def union(self,A,B):
        root_A = self.find(A)
        root_B = self.find(B)
        if root_A==root_B:
            return(False)
        
        if self.size[root_A]<self.size[root_B]:
            self.parent[root_A]=root_B
            self.size[root_B]+=self.size[root_A]
        else:
            self.parent[root_B]=root_A
            self.size[root_A]+=self.size[root_B]
            
        return(True)
class Solution:
    def containsCycle(self, grid: List[List[str]]) -> bool:
        R = len(grid) ; C = len(grid[0])
        dsu = UnionFind(R*C)
        def encode(r,c):
            return(r*C+c)
        
        for r in range(R):
            for c in range(C):
                if c+1<C and grid[r][c]==grid[r][c+1]:
                    if not dsu.union( encode(r,c),encode(r,c+1) ):
                        return(True)
                if r+1<R and grid[r][c]==grid[r+1][c]:
                    if not dsu.union( encode(r,c),encode(r+1,c) ):
                        return(True)
                    
        return(False)
    
                    
            
        

class Solution:
    def containsCycle(self, grid: List[List[str]]) -> bool:
        vis = [[False for i in range(len(grid[0]))] for j in range(len(grid))] 
        
        dirx = [0,0,-1,+1]
        diry = [-1,+1, 0,0]
        
        def check(i,j):
            if 0<= i < len(grid) and 0<= j< len(grid[0]): return True
            return False
        
        def dfs(r,c,pr,pc, no):
            flag = False
            vis[r][c] = True
            for i in range(4):
                nr,nc = r+dirx[i], c+diry[i]
                if not (nr == pr and nc == pc) and check(nr,nc) and grid[nr][nc] == grid[r][c]:
                    if vis[nr][nc] : print((nr,nc)); return True
                    if dfs(nr,nc,r,c,no+1): return True
            return False
                    
        
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if not vis[i][j]:
                    if(dfs(i,j, -1,-1, 0)):
                        return True
        return False

class UnionFind:
    def __init__(self,n):
        self.parent = [i for i in range(n)] # i is the parent of i , initially
        self.size = [1 for i in range(n)]
    
    def find(self,A):
        root = A
        while root!=self.parent[root]:
            root = self.parent[root]
            
        while A!=root:
            old_parent = self.parent[A]
            self.parent[A]=root
            A = old_parent    
        return(root)
    
    def union(self,A,B):
        root_A = self.find(A)
        root_B = self.find(B)
        
        if root_A == root_B:
            return(False)
        if self.size[root_A] < self.size[root_B]:
            self.parent[A]= root_B
            self.size[root_B]+=self.size[root_A]
        else:
            self.parent[B]= root_A
            self.size[root_A]+=self.size[root_B]
        return(True)
            
class Solution:
    def containsCycle(self, grid: List[List[str]]) -> bool:
        R = len(grid) ; C = len(grid[0])
        dsu = UnionFind(R*C)
        def encode(r,c):
            return(r*C+c)
        
        for r in range(R):
            for c in range(C):
                if c+1<C and grid[r][c]==grid[r][c+1]:
                    if not dsu.union( encode(r,c),encode(r,c+1) ):
                        return(True)
                if r+1<R and grid[r][c]==grid[r+1][c]:
                    if not dsu.union( encode(r,c),encode(r+1,c) ):
                        return(True)
                    
        return(False)
    
                    
            
        

class Solution:
    def containsCycle(self, grid: List[List[str]]) -> bool:
        disjoint = {}
        def find(x):
            while disjoint[x] != x:
                x = disjoint[x]
            return disjoint[x]
        def union(x, y):
            disjoint[find(y)] = find(x)
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                disjoint[(i, j)] = (i, j)
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                for di, dj in [[0, 1], [1, 0]]:
                    if 0 <= i + di < len(grid) and 0 <= j + dj < len(grid[0]) and grid[i][j] == grid[i + di][j + dj]:
                        if find((i, j)) == find((i + di, j + dj)):
                            return True
                        union((i, j), (i + di, j + dj))
        return False

class Solution:
    def containsCycle(self, grid: List[List[str]]) -> bool:
        visited = set()
        m = len(grid)
        n = len(grid[0])
        if m == 1 or n == 1:
            return False
       
        dir = [(0,-1), (-1,0), (0,1), (1,0)]
        
        def dfs(prev, curr):
            if (curr in visited):
                return True
            visited.add(curr)
            for d in dir:
                x = (curr[0] + d[0], curr[1] + d[1])
                if (x != prev and 0<= x[0] < m and 0<=x[1]<n and grid[x[0]][x[1]] == grid[curr[0]][curr[1]]):
                    if dfs(curr, x):
                        return True
            else:
                return False
            
            
            
            
            
        for i in range(m):
            for j in range(n):
                node = (i,j)
                if node not in visited:
                    if dfs(None, node) == True:
                        return True
        return False
                
                

class Solution:
    def containsCycle(self, grid: List[List[str]]) -> bool:
        visited = set()
        m = len(grid)
        if m==1: return False
        n = len(grid[0])
        dirs = [(0,-1),(-1,0),(0,1),(1,0)]
        def dfs(prev, curr):
            if curr in visited: return True
            visited.add(curr)
            for dirn in dirs:
                nei = (dirn[0]+curr[0], dirn[1]+curr[1])
                if 0<=nei[0]<m and 0<=nei[1]<n and nei != prev and grid[nei[0]][nei[1]] == grid[curr[0]][curr[1]]:
                    if dfs(curr, nei): return True
            return False
        for i in range(m):
            for j in range(n):
                if (i,j) not in visited:
                    if dfs(None, (i,j)): return True
        return False
        
#     for each cell, if not in visited, do a DFS
    
#     DFS (prev->(x,y), curx, cury, visited):
#         if cur is visited: return True
#         for each cell, 
#             if nei is not prev and nei == cur: 
#                 if (DFS on the nei) return True
#         return False

class UnionFind:
    def __init__(self,n):
        self.parent = [i for i in range(n)]
        self.size = [1 for i in range(n)]
    def find(self,A):
        root = A
        while root !=self.parent[root]:
            root = self.parent[root]
        
        while A!=root:
            old_parent = self.parent[A]
            self.parent[A]=root
            A=old_parent
        return(root)
    def union(self,A,B):
        root_A= self.find(A)
        root_B= self.find(B)
        if root_A==root_B:
            return(False)
        
        if self.size[root_A]<self.size[root_B]:
            self.parent[root_A] = root_B
            self.size[root_B]+=self.size[root_A]
        else:
            self.parent[root_B]=root_A
            self.size[root_A]+=self.size[root_B]
        return(True)
            
class Solution:
    def containsCycle(self, grid: List[List[str]]) -> bool:
        R = len(grid) ; C = len(grid[0])
        dsu = UnionFind(R*C)
        def encode(r,c):
            return(r*C+c)
        
        for r in range(R):
            for c in range(C):
                if c+1<C and grid[r][c]==grid[r][c+1]:
                    if not dsu.union( encode(r,c),encode(r,c+1) ):
                        return(True)
                if r+1<R and grid[r][c]==grid[r+1][c]:
                    if not dsu.union( encode(r,c),encode(r+1,c) ):
                        return(True)
                    
        return(False)
    
                    
            
        

class UnionFind:
    def __init__(self, row_size, col_size):
        self.roots = [[(i, j) for j in range(col_size)]
                      for i in range(row_size)]

    def get_rank(self, node):
        return -node[0] * len(self.roots[0]) - node[1]

    def union(self, node1, node2):
        root1 = self.find(node1)
        root2 = self.find(node2)
        if root1 != root2:
            if self.get_rank(root1) > self.get_rank(root2):
                self.roots[root2[0]][root2[1]] = root1
            else:
                self.roots[root1[0]][root1[1]] = root2

    def find(self, node):
        if self.roots[node[0]][node[1]] != node:
            self.roots[node[0]][node[1]] = self.find(self.roots[node[0]][node[1]])
        return self.roots[node[0]][node[1]]


class Solution:
    def containsCycle(self, grid: List[List[str]]) -> bool:
        row_size, col_size = len(grid), len(grid[0])
        uf = UnionFind(row_size, col_size)
        for i in range(row_size):
            for j in range(col_size):
                for (x, y) in [(i - 1, j), (i, j - 1)]:
                    if x >= 0 and y >= 0 and grid[x][y] == grid[i][j]:
                        if uf.find((i, j)) == uf.find((x, y)):
                            return True
                        uf.union((i, j), (x, y))
        return False
                        
            

class Solution:
    def containsCycle(self, grid: List[List[str]]) -> bool:
        parent = {}

        def find(u):
            parent.setdefault(u, u)
            if parent[u] != u:
                parent[u] = find(parent[u])
            return parent[u]

        def union(u, v):
            x, y = find(u), find(v)
            if x != y:
                parent[y] = x 
            return x != y

        m, n = len(grid), len(grid[0])
        for i, row in enumerate(grid):
            for j, cell in enumerate(row):
                if i + 1 < m and grid[i][j] == grid[i + 1][j]:
                    if not union((i, j), (i + 1, j)):
                        return True
                if j + 1 < n and grid[i][j] == grid[i][j + 1]:
                    if not union((i, j), (i, j + 1)):
                        return True 
        return False
class DSU:
    def __init__(self,m,n):
        self.par = {(i,j):(i,j) for i in range(m) for j in range(n)}
    
    def find(self,x):
        if self.par[x]!=x:
            self.par[x] = self.find(self.par[x])
        return self.par[x]
    
    def union(self,x,y):
        xp,yp = self.find(x),self.find(y)
        if xp == yp:
            return False
        self.par[xp] = yp
        return True

dirs = [(0,1),(1,0)]
class Solution:
    def containsCycle(self, grid):
        R,C = len(grid),len(grid[0])
        dsu = DSU(R,C)        
        for r in range(R):
            for c in range(C):
                for x,y in dirs:
                    nr,nc = r+x,c+y
                    if 0<=nr<R and 0<=nc<C and grid[r][c] == grid[nr][nc]:
                        if dsu.union((r,c),(nr,nc)) == False:
                            return True
        return False
                
                
                        

class Solution:
    x = [0,0,-1,1]
    y = [-1,1,0,0]
    def findCycle(self,grid,i,j,li,lj,path,vis):
        
        if vis[i][j]:
            return False
        
        for k in range(4):
            nx = i + Solution.x[k]
            ny = j + Solution.y[k]
            if nx == li and ny == lj:
                continue
            if nx < 0 or ny < 0 or nx >= len(grid) or ny >= len(grid[0]):
                continue
            if (nx,ny) in path and path[(nx,ny)] == 1:
                vis[i][j] = 1
                return True
            if grid[nx][ny] == grid[i][j]:
                path[(nx,ny)] = 1
                isfind = self.findCycle(grid,nx,ny,i,j,path,vis)
                if isfind:
                    vis[i][j] = 1
                    return True
                path[(nx,ny)] = 0
        vis[i][j] = 1
        return False
    def containsCycle(self, grid: List[List[str]]) -> bool:
        h = len(grid)
        if h == 0:
            return False
        w = len(grid[0])
        vis = [ [0 for i in range(w)] for j in range(h)]
        path = {}
        for i in range(h):
            for j in range(w):
                if not vis[i][j]:
                    isfind = self.findCycle(grid,i,j,-1,-1,path,vis)
                    if isfind:
                        return True
        return False
class DSU:
  def __init__(self):
    # representer
    self.reps = {}
  def add(self, x):
    self.reps[x] = x
  def find(self, x):
    if not x == self.reps[x]:
      self.reps[x] = self.find(self.reps[x])
    return self.reps[x]
  def union(self, x, y):
    self.reps[self.find(y)] = self.find(x)

class Solution:
  def containsCycle(self, grid: List[List[str]]) -> bool:
    # detect cycle, often dfs, but use dsu in this problem due to its special graph structure.
    m, n = len(grid), len(grid[0])
    dsu = DSU()
    for i in range(m):
      for j in range(n):
        dsu.add((i, j))
        if i - 1 >= 0 and j - 1 >= 0 and grid[i - 1][j] == grid[i][j - 1] == grid[i][j] and dsu.find((i - 1, j)) == dsu.find((i, j - 1)):
          return True
        if i - 1 >= 0 and grid[i - 1][j] == grid[i][j]:
          dsu.union((i - 1, j), (i, j))
        if j - 1 >= 0 and grid[i][j - 1] == grid[i][j]:
          dsu.union((i, j - 1), (i, j))
    return False
class Solution:
    def containsCycle(self, grid: List[List[str]]) -> bool:
        m = len(grid)
        n = len(grid[0])
        visited = {}
        
        def dfs(i, j, k):
            c = i * n + j
            if c in visited:
                return k - visited[c] >= 4
            visited[c] = k
            a = grid[i][j]
            for di, dj in [[0, -1], [0, 1], [-1, 0], [1, 0]]:
                ii = i + di
                jj =j + dj
                if not (0 <= ii < m and 0 <= jj < n) or grid[ii][jj] != a:
                    continue
                if dfs(ii, jj, k + 1):
                    return True
            return False
                
                
        for i in range(m):
            for j in range(n):
                c = i * n + j
                if c in visited:
                    continue
                if dfs(i, j, 0):
                    return True
        return False

import collections

class Solution:
    def containsCycle(self, grid):
        if not grid:
            return False
        
        M = len(grid)
        N = len(grid[0])
        parent = {}
        
        def find(x, y):
            if parent[(x, y)] != (x, y):
                parent[(x, y)] = find(parent[(x, y)][0], parent[(x, y)][1])
            return parent[(x, y)]
            
        def union(x1, y1, x2, y2):
            p1 = find(x1, y1)
            p2 = find(x2, y2)
            
            if p1 == p2:
                return True
            
            parent[p1] = p2
            return False
        
        def move(x, y, grid):
            for i, j in [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]:
                if 0 <= i < M and 0 <= j < N:
                    yield i, j
        
        seen = set()
        for i in range(M):
            for j in range(N):
                seen.add((i, j))
                for x1, y1 in move(i, j, grid):
                    if (i, j) not in parent:
                        parent[(i, j)] = (i, j)
                    if (x1, y1) not in parent:
                        parent[(x1, y1)] = (x1, y1)

                    if grid[i][j] == grid[x1][y1] and (x1, y1) not in seen:
                        if union(i, j, x1, y1):
                            return True
        return False
        

class Solution:
    def containsCycle(self, grid: List[List[str]]) -> bool:
        
        
        
        # public solution ... 3384 ms ... 0 % ... 216 MB ... 0 %
        #  time: O(n*m)
        # space: O(n*m)
        
        def dfs(pi, pj, i, j):
            visited[i][j] = True
            for ni, nj in [(i-1, j), (i, j-1), (i, j+1), (i+1, j)]:
                if 0 <= ni < n and 0 <= nj < m and grid[ni][nj] == grid[i][j]:
                    if visited[ni][nj]:
                        if (ni,nj) != (pi,pj):
                            return True
                    else:
                        if dfs(i, j, ni, nj):
                            return True
            return False
        
        n, m = len(grid), len(grid[0])
        if n < 2 or m < 2:
            return False
        visited = [[False]*m for _ in range(n)]
        for i in range(n):
            for j in range(m):
                if not visited[i][j] and dfs(-1, -1, i, j):
                    return True
        return False
        
        

class Solution:
    def containsCycle(self, grid: List[List[str]]) -> bool:
        m, n = len(grid), len(grid[0])
        dirs = [[0,-1], [0,1], [-1,0], [1,0]]
        visited = [[False]*n for i in range(m)]

        def dfs(y, x, py, px, c):

            visited[y][x] = True
            for d in dirs:
                ny, nx = y+d[0], x+d[1]
                if ny < 0 or ny >= m or nx < 0 or nx >= n or (ny == py and nx == px) or grid[ny][nx] != c:
                    continue
                if visited[ny][nx] or dfs(ny, nx, y, x, c):
                    return True
            
            return False
        
        for i in range(m):
            for j in range(n):
                if not visited[i][j] and dfs(i, j, -1, -1, grid[i][j]):
                    return True

        return False
sys.setrecursionlimit(10000000)
class Solution:
    def containsCycle(self, grid: List[List[str]]) -> bool:
        visited = set()
        m = len(grid)
        if m==1: return False
        n = len(grid[0])
        dirs = [(0,-1),(-1,0),(0,1),(1,0)]
        def dfs(prev, curr):
            if curr in visited: return True
            visited.add(curr)
            for dirn in dirs:
                nei = (dirn[0]+curr[0], dirn[1]+curr[1])
                if 0<=nei[0]<m and 0<=nei[1]<n and nei != prev and grid[nei[0]][nei[1]] == grid[curr[0]][curr[1]]:
                    if dfs(curr, nei): return True
            return False
        for i in range(m):
            for j in range(n):
                if (i,j) not in visited:
                    if dfs(None, (i,j)): return True
        return False
#from collections import deque
#from random import randint

class Solution:
    def containsCycle(self, grid: List[List[str]]) -> bool:
        
        h, w = len(grid), len(grid[0])
        
        #------------------------------------------
        
        def in_range(x, y):
            
            return (0 <= x < w) and (0 <= y < h) 
        
        #------------------------------------------
        
        def four_neighbors(x, y):
            
            for dx, dy in {(+1, 0), (-1, 0), (0, +1), (0, -1)}:
                
                next_x, next_y = x + dx, y + dy
                
                if in_range(next_x, next_y):
                    yield (next_x, next_y)
            

        
        #------------------------------------------
        
        def dfs(x, y, prev_x, prev_y, grid):
            
            if grid[y][x] == dfs.cur_symbol:
                # this grid has a cycle with current symbol
                return True
            
            
            # mark to uppercase letter as visited
            grid[y][x] = dfs.cur_symbol
            
            
            for next_x, next_y in four_neighbors(x, y):
                
                if (next_x, next_y) == (prev_x, prev_y):
                    # avoid backward visit
                    continue
                    
                elif grid[next_y][next_x].upper() != dfs.cur_symbol:
                    # different symbol
                    continue
                    
                if dfs(next_x, next_y, x, y, grid): return True
            
            #print(f'terminate with {x} {y} {grid[y][x]}')
            return False
            
        
        #------------------------------------------
        
        failed_symbol = set()
        
        for y in range(h):
            for x in range(w):
                
                dfs.cur_symbol = grid[y][x]
                
                if dfs.cur_symbol in failed_symbol:
                    # skip search on failed symbol
                    continue
                
                dfs.cur_symbol = grid[y][x].upper()
                
                if dfs(x,y,-1,-1, grid):
                    return True
                else:
                    failed_symbol.add( dfs.cur_symbol )
        
        return False
class Solution:
    def containsCycle(self, grid: List[List[str]]) -> bool:
        m = len(grid)
        n = len(grid[0])
        visited = [[0] * n for i in range(m)]
        
        def dfs(i, j, k):
            if visited[i][j]:
                return k - visited[i][j] >= 4
            visited[i][j] = k
            a = grid[i][j]
            for di, dj in [[0, -1], [0, 1], [-1, 0], [1, 0]]:
                ii = i + di
                jj =j + dj
                if not (0 <= ii < m and 0 <= jj < n) or grid[ii][jj] != a:
                    continue
                if dfs(ii, jj, k + 1):
                    return True
            return False
                
                
        for i in range(m):
            for j in range(n):
                if visited[i][j]:
                    continue
                if dfs(i, j, 1):
                    return True
        return False

class Solution:
    def containsCycle(self, grid: List[List[str]]) -> bool:
        
        m, n = len(grid), len(grid[0])
        root = {}
        size = {}
        
        def find(t):
            if root[t] != t:
                root[t] = find(root[t])
            return root[t]
        
        for i in range(m):
            for j in range(n):
                root[(i,j)] = (i,j)
                size[(i,j)] = 1
                
                if 0<=i-1 and grid[i-1][j] == grid[i][j]:
                    top = (i-1, j)
                    rt = find(top)
                    root[(i,j)] = rt
                    size[rt] += 1
                if 0<=j-1 and grid[i][j-1] == grid[i][j]:
                    left = (i, j-1)
                    rl = find(left)
                    root[(i,j)] = rl
                    size[rl] += 1
                if 0<=i-1 and 0<=j-1 and grid[i-1][j] == grid[i][j] and grid[i][j-1] == grid[i][j]:
                    rl = root[(i,j-1)]
                    rt = root[(i-1,j)]
                    if rl == rt:
                        return True
                    if size[rt] >= size[rl]:
                        root[rl] = rt
                        size[rt] += size[rl]
                    else:
                        root[rt] = rl
                        size[rl] += size[rt]
        #print(root)
        return False
                    
                    
                        

class Solution:
    def containsCycle(self, grid: List[List[str]]) -> bool:
        for x, row in enumerate(grid):
            for y,num in enumerate(row):
                if self.containsCycleInComponent(grid, (x,y)):
                    return True
        
        return False
    
    def adjCells(self, x, y):
        yield x-1, y
        yield x+1, y
        yield x, y-1
        yield x, y+1
    
    def containsCycleInComponent(self, grid, startPoint):
        # startPoint is the position (x,y)
        startX,startY = startPoint
        value = grid[startX][startY]
        if value is None:
            return False
        
        checkedPoints = set()
        uncheckedPoints = [startPoint]
        
        while uncheckedPoints:
            point = uncheckedPoints.pop()
            checkedPoints.add(point)
            
            x,y = point
            grid[x][y] = None
            adjKnownPoints = 0
            for nextPoint in self.adjCells(x,y):
                if nextPoint in checkedPoints or self.hasValue(grid, nextPoint, value):
                    if nextPoint not in checkedPoints:
                        uncheckedPoints.append(nextPoint)
                    else:
                        adjKnownPoints += 1
            if adjKnownPoints > 1:
                return True
            
        return False
            
    
    def hasValue(self, grid, point, value):
        x, y = point
        return 0<=x<len(grid) and 0<=y<len(grid[x]) and grid[x][y]==value
        

class Solution:
    def containsCycle(self, grid: List[List[str]]) -> bool:
        R = len(grid)
        C = len(grid[0])
        N = R * C
        parents = [i for i in range(N)]
        
        def ufind(a):
            if parents[a] == a:
                return a
            parents[a] = ufind(parents[a])
            return parents[a]
        
        def uunion(a, b):
            aa = ufind(a)
            bb = ufind(b)
            if aa == bb:
                return False
            parents[bb] = aa
            return True
        
        def decode(row, col):
            return row * C + col
        
        for row in range(R):
            for col in range(C):
                if row + 1 < R and grid[row + 1][col] == grid[row][col]:
                    if not uunion(decode(row, col), decode(row + 1, col)):
                        return True
                if col + 1 < C and grid[row][col + 1] == grid[row][col]:
                    if not uunion(decode(row, col), decode(row, col + 1)):
                        return True
                    
        return False
class Solution:
    def containsCycle(self, grid: List[List[str]]) -> bool:
        m = len(grid)
        n = len(grid[0])
        visited = [[0] * n for i in range(m)]
        
        def dfs(i, j, pi, pj):
            if visited[i][j]:
                return True
            visited[i][j] = 1
            a = grid[i][j]
            for di, dj in [[0, -1], [0, 1], [-1, 0], [1, 0]]:
                ii = i + di
                jj =j + dj
                if not (0 <= ii < m and 0 <= jj < n) or grid[ii][jj] != a or (ii == pi and jj == pj):
                    continue
                if dfs(ii, jj, i, j):
                    return True
            return False
                
                
        for i in range(m):
            for j in range(n):
                if visited[i][j]:
                    continue
                if dfs(i, j, -1, -1):
                    return True
        return False

class Solution:
    def containsCycle(self, g: List[List[str]]) -> bool:            
        def find(u):
            if u != UF[u]: u = find(UF[u])
            return UF[u]        
        UF, m, n = {}, len(g), len(g[0])                
        for i in range(m):
            for j in range(n):
                u = (i, j); UF.setdefault(u, u)
                for x, y in [(i, j-1), (i-1, j)]:                    
                    if not (0 <= x < m and 0 <= y < n): continue                    
                    if g[x][y] == g[i][j]:                      
                        v = (x, y)
                        UF.setdefault(v, v)
                        pu, pv = find(u), find(v)                            
                        if pu != pv: UF[pv] = pu
                        else: return True
        return False                
class Solution:
    def containsCycle(self, grid: List[List[str]]) -> bool:
        
        
        
        # public solution ... 
        
        def dfs(pi, pj, i, j):
            visited[i][j] = True
            for ni, nj in [(i-1, j), (i, j-1), (i, j+1), (i+1, j)]:
                if 0 <= ni < n and 0 <= nj < m and grid[ni][nj] == grid[i][j]:
                    if visited[ni][nj]:
                        if (ni,nj) != (pi,pj):
                            return True
                    else:
                        if dfs(i, j, ni, nj):
                            return True
            return False
        
        n, m = len(grid), len(grid[0])
        if n < 2 or m < 2:
            return False
        visited = [[False]*m for _ in range(n)]
        for i in range(n):
            for j in range(m):
                if not visited[i][j] and dfs(-1, -1, i, j):
                    return True
        return False
        
        

import collections
class Solution:
    def containsCycle(self, grid: List[List[str]]) -> bool:
        def find(pos):
            if parents[pos] != pos:
                parents[pos] = find(parents[pos])
            return parents[pos]

        def union(pos1, pos2):
            parent1, parent2 = find(pos1), find(pos2)
            if parent1 != parent2:
                if ranks[parent1] > ranks[parent2]:
                    parents[parent2] = parent1
                else:
                    parents[parent1] = parent2
                    if ranks[parent1] == ranks[parent2]:
                        ranks[parent1] += 1

        rows, cols = len(grid), len(grid[0])
        parents = {(i, j): (i, j) for i in range(rows) for j in range(cols)}
        ranks = collections.Counter()
        for i, row in enumerate(grid):
            for j, letter in enumerate(row):
                if i > 0 and j > 0 and grid[i-1][j] == grid[i][j-1] == letter and find((i-1, j)) == find((i, j-1)):
                    return True
                for r, c in (i - 1, j), (i, j - 1):
                    if 0 <= r < rows and 0 <= c < cols and grid[r][c] == letter:
                        union((i, j), (r, c))
        return False

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

class Solution:
    def containsCycle(self, A: List[List[str]]) -> bool:
        R, C = len(A), len(A[0])
        def encode(r, c):
            return r * C + c
        
        dsu = DSU(R * C)
        for r in range(R):
            for c in range(C):
                if c + 1 < C and A[r][c] == A[r][c+1]:
                    if not dsu.union(encode(r, c), encode(r, c + 1)):
                        return True
                if r + 1 < R and A[r][c] == A[r+1][c]:
                    if not dsu.union(encode(r, c), encode(r + 1, c)):
                        return True
        return False

class Solution:
    def containsCycle(self, grid: List[List[str]]) -> bool:
        
        def find(pos):
            if parents[pos] != pos:
                parents[pos] = find(parents[pos])
            return parents[pos]

        def union(pos1, pos2):
            parent1, parent2 = find(pos1), find(pos2)
            if parent1 != parent2:
                if ranks[parent2] > ranks[parent1]:
                    parents[parent1] = parent2
                else:
                    parents[parent2] = parent1
                    if ranks[parent1] == ranks[parent2]:
                        ranks[parent1] += 1

        rows, cols = len(grid), len(grid[0])
        parents = {(i, j): (i, j) for i in range(rows) for j in range(cols)}
        ranks = collections.Counter()
        for i, row in enumerate(grid):
            for j, letter in enumerate(row):
                if i > 0 and j > 0 and grid[i-1][j] == grid[i][j-1] == letter and find((i-1, j)) == find((i, j-1)):
                    return True
                for r, c in (i - 1, j), (i, j - 1):
                    if 0 <= r < rows and 0 <= c < cols and grid[r][c] == letter:
                        union((i, j), (r, c))
        return False

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
#         def dfs(node, parent):
#             if node in visited: return True
#             visited.add(node)
#             nx,ny = node
#             childs = [(cx,cy) for cx,cy in [[nx+1,ny],[nx-1, ny],[nx,ny+1],[nx,ny-1]] 
#                       if 0 <= cx < m and 0 <= cy < n 
#                       and grid[cx][cy] == grid[nx][ny] and (cx,cy) != parent]
#             for x in childs:
#                 if dfs(x, node): return True 
#             return False  
    
#         m, n = len(grid), len(grid[0])
#         visited = set()
#         for i in range(m):
#             for j in range(n):
#                 if (i,j) in visited: continue 
#                 if dfs((i,j), None): return True
#         return False 

class Solution:
    def containsCycle(self, grid) -> bool:
        if not grid or not grid[0]: return False
        
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        visited=[[False] * len(grid[i]) for i in range(len(grid))]
                
        for i in range(len(grid)):
            for j in range(len(grid[i])):
                if not visited[i][j]:
                    stack = [(i, j, 0, 0)]
                    visited[i][j] = True
                    symbol = grid[i][j]
                    
                    while stack:
                        row, col, prev_row, prev_col = stack.pop()

                        for direction in directions:
                            nr = row + direction[0]
                            nc = col + direction[1]

                            if nr == prev_row and nc == prev_col:
                                continue
                            if len(grid) > nr >= 0 and len(grid[nr]) > nc >= 0 and grid[nr][nc] == symbol:
                                if visited[nr][nc]:
                                    return True
                                else:
                                    visited[nr][nc] = True
                                    stack.append((nr, nc, row, col))
                      
        return False
class Solution:
    def containsCycle(self, grid: List[List[str]]) -> bool:
        for x, row in enumerate(grid):
            for y,num in enumerate(row):
                if self.containsCycleInComponent(grid, (x,y)):
                    return True
                else:
                    self.eraseComponent(grid, (x,y))
        
        return False
    
    def eraseComponent(self, grid, startPoint):
        # startPoint is the position (x,y)
        startX,startY = startPoint
        value = grid[startX][startY]
        if value is None:
            return
        
        pointsToErase = [startPoint]
        while pointsToErase:
            point = pointsToErase.pop()
            x,y = point
            grid[x][y] = None
            for nextPoint in [(x-1,y), (x+1,y), (x,y-1), (x,y+1)]:
                if self.hasValue(grid, nextPoint, value):
                    pointsToErase.append(nextPoint)
    
    def containsCycleInComponent(self, grid, startPoint):
        # startPoint is the position (x,y)
        startX,startY = startPoint
        value = grid[startX][startY]
        if value is None:
            return False
        
        checkedPoints = set()
        uncheckedPoints = [startPoint]
        componentPointsCount = 0
        componentEdgesDoubleCount = 0
        
        while uncheckedPoints:
            point = uncheckedPoints.pop()
            componentPointsCount += 1
            checkedPoints.add(point)
            
            x,y = point
            for nextPoint in [(x-1,y), (x+1,y), (x,y-1), (x,y+1)]:
                if self.hasValue(grid, nextPoint, value):
                    componentEdgesDoubleCount += 1
                    if nextPoint not in checkedPoints:
                        uncheckedPoints.append(nextPoint)
        
            
        return componentPointsCount <= componentEdgesDoubleCount // 2
                    
            
    
    def hasValue(self, grid, point, value):
        x, y = point
        return 0<=x<len(grid) and 0<=y<len(grid[x]) and grid[x][y]==value
        

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
    def containsCycle(self, grid: List[List[str]]) -> bool:
        n = len(grid)
        m = len(grid[0])
        uf = Unionfind(n*m)
        
        for i in range(n):
            for j in range(m):
                if i+1<n and grid[i][j]==grid[i+1][j]:
                    if uf.is_same(m*i+j, m*(i+1)+j):
                        return True
                    
                    uf.unite(m*i+j, m*(i+1)+j)
                
                if j+1<m and grid[i][j]==grid[i][j+1]:
                    if uf.is_same(m*i+j, m*i+j+1):
                        return True
                    
                    uf.unite(m*i+j, m*i+j+1)
        
        return False
class Solution:
    def containsCycle(self, grid: List[List[str]]) -> bool:
        m = len(grid)
        n = len(grid[0])
        visited = set()
        
        def dfs(i, j, pi, pj):
            k = i * n + j
            if k in visited:
                return True
            visited.add(k)
            a = grid[i][j]
            for di, dj in [[0, -1], [0, 1], [-1, 0], [1, 0]]:
                ii = i + di
                jj =j + dj
                if not (0 <= ii < m and 0 <= jj < n) or grid[ii][jj] != a or (ii == pi and jj == pj):
                    continue
                if dfs(ii, jj, i, j):
                    return True
            return False
                
                
        for i in range(m):
            for j in range(n):
                if (i * n + j) in visited:
                    continue
                if dfs(i, j, -1, -1):
                    return True
        return False

class Solution:
    def containsCycle(self, grid: List[List[str]]) -> bool:
        def find(coords):
            if coords != parents[coords]:
                parents[coords] = find(parents[coords])
            return parents[coords]
        
        def union(point_one, point_two):
            parx, pary = find(point_one), find(point_two)
            if parx != pary:
                if rank[parx] > rank[pary]:
                    parents[pary] = parx
                else:
                    parents[parx] = pary
                    if rank[parx] == rank[pary]:
                        rank[parx] += 1

        if not grid or not grid[0]:
            return False
        m, n = len(grid), len(grid[0])
        
        parents = {(i, j): (i, j) for i in range(m) for j in range(n)}
        rank = collections.Counter()
        
        for i in range(m):
            for j in range(n):
                if i > 0 and j > 0 and grid[i-1][j] == grid[i][j-1] == grid[i][j] and find((i-1, j)) == find((i, j-1)):
                    return True
                for r, c in (i-1, j), (i, j-1):
                    if r >= 0 and c >= 0 and grid[r][c] == grid[i][j]:
                        union((r, c), (i, j))
                        
        return False
class Solution:
    def containsCycle(self, grid: List[List[str]]) -> bool:
        visited = set()
        def dfs(i, j, pre_i, pre_j):
            visited.add((i, j))
            for x, y in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                if 0 <= x + i < len(grid) and 0 <= y + j < len(grid[0]) and grid[i][j] == grid[i + x][j + y] and (i + x != pre_i or j + y != pre_j):
                    if (i + x, j + y) in visited or dfs(i + x, j + y, i, j):
                        return True
            return False
    
        for i in range(len(grid)):
            for j in range(len(grid[i])):
                if (i, j) not in visited:
                    if dfs(i, j, -1, -1):
                        return True
        return False
class UnionFind:
    def __init__(self, n):
        self.parents = [i for i in range(n)]
        
    def find(self, i):
        if self.parents[i] == i:
            return i
        self.parents[i] = self.find(self.parents[i])
        return self.parents[i]
    def unite(self, a, b):
        pa = self.find(a)
        pb = self.find(b)
        self.parents[pb] = pa
        return
    def same(self, a, b):
        pa = self.find(a)
        pb = self.find(b)
        return pa==pb
    
class Solution:
    def containsCycle(self, grid: List[List[str]]) -> bool:
        dx = [ 1, 0]
        dy = [ 0, 1]
        N = len(grid)
        M = len(grid[0])
        ALL = N*M
        # print("size", N, M, ALL)
        tree = UnionFind(ALL)
        for i in range(N):
            for j in range(M):
                for k in range(2):
                    if 0<=i +dx[k] <N and 0<= j+dy[k]<M:
                        if grid[i][j] == grid[i+dx[k]][j+dy[k]]:
                            # print(i, j, k)
                            # print((i+dx[k])*M+j+dy[k])
                            if tree.same(i*M+j, (i+dx[k])*M+j+dy[k]):
                                return True
                            tree.unite(i*M+j, (i+dx[k])*M+j+dy[k])
        return False
class Solution:
    def containsCycle(self, grid: List[List[str]]) -> bool:
        '''
        [["a","b","b"],
         ["b","z","b"],
         ["b","b","a"]]
        
        [["a","b","b"],
         ["b","b","b"],
         ["b","b","a"]]
        
        dfs(pre_i,pre_j, i,j):-> bool: hasLoop
        '''
        label = [[0 for _ in range(len(grid[0]))] for _ in range(len(grid))]
        
        def dfs(pre_i,pre_j, i,j):
            if label[i][j] == 1: # is being visited
                return True
            if label[i][j] == -1:# visited
                return False
            label[i][j] = 1
            for ii, jj in [(i+1,j),(i-1,j),(i,j+1),(i,j-1)]:
                if 0<=ii<len(grid) and 0<=jj<len(grid[ii]):
                    if grid[ii][jj] == grid[i][j] and (not (ii==pre_i and jj==pre_j)):
                        if dfs(i,j,ii,jj): return True
            label[i][j] = -1
            return False
        
        for i in range(len(grid)):
            for j in range(len(grid[i])):
                if dfs(-1,-1, i,j): return True
        return False

class Solution:
    def containsCycle(self, grid: List[List[str]]) -> bool:
        dirs = [[-1, 0], [0, 1], [1, 0], [0, -1]]
        def dfs(m, n, pm, pn):
            if m < 0 or m >= len(grid) or n < 0 or n >= len(grid[m]):
                return False
            if grid[m][n].lower() != grid[pm][pn].lower():
                return False
            if grid[m][n].isupper():
                return True
            grid[m][n] = grid[m][n].upper()
            for dir in dirs:
                if m + dir[0] != pm or n + dir[1] != pn:
                    if (dfs(m + dir[0], n + dir[1], m, n)):
                        return True
            return False
        for m in range(len(grid)):
            for n in range(len(grid[m])):
                if grid[m][n].islower():
                    if dfs(m, n, m, n):
                        return True
        return False

class Solution:
    def containsCycle(self, grid: List[List[str]]) -> bool:
        if not grid:
            return(False)
        visited = [[0 for i in grid[0]] for j in grid]
 
        def ExplorePath(rowIn,colIn, value, previous):
            current = [rowIn, colIn] 
            #print(rowIn,colIn)
            if visited[rowIn][colIn]==1:
                return(True)
                #print('hi')
    
            else:
                
                visited [rowIn][colIn] =1
                
                #print(temp1[0][0])
                output = False
                if rowIn<len(grid)-1 and previous != [rowIn+1,colIn] and grid[rowIn+1][colIn]==value:
                    output = ExplorePath(rowIn+1,colIn,value, current)
                if colIn<len(grid[0])-1 and previous != [rowIn,colIn+1] and grid[rowIn][colIn+1]==value:
                    output = output or ExplorePath(rowIn,colIn+1, value, current) 
                if rowIn>0 and previous != [rowIn-1,colIn] and grid[rowIn-1][colIn]==value:
                    output = output or ExplorePath(rowIn-1,colIn,  value, current)
                if colIn>0 and previous != [rowIn,colIn-1] and grid[rowIn][colIn-1]==value:
                    output = output or ExplorePath(rowIn,colIn-1, value, current) 
                    
            return(output)
            

        
        for rowIn in range(len(grid)-1):
            for colIn in range(len(grid[0])-1):
                if grid[rowIn+1][colIn]== grid[rowIn][colIn]:
                    if grid[rowIn+1][colIn+1]== grid[rowIn][colIn]:
                        if grid[rowIn][colIn+1]== grid[rowIn][colIn]:
                            #print(rowIn,colIn)
                            return(True) 
        for rowIn in range(len(grid)):
            for colIn in range(len(grid[0])):
                
                if visited[rowIn][colIn]==0:
                    #print(grid[rowIn][colIn])
                    tempVisited = []
                    #print(tempVisited[0][0])
                    length = 0
                    if (ExplorePath(rowIn,colIn, grid[rowIn][colIn], [rowIn,colIn])):
                        return(True)
                    
        return(False)
    
    def containsCycle2(self, grid: List[List[str]]) -> bool:
        if not grid:
            return(False)
        visited = [[0 for i in grid[0]] for j in grid]
 
        def ExplorePath(rowIn,colIn, length, tempV, value, previous):
            current = [rowIn, colIn] 
            #print(rowIn,colIn)
            if grid[rowIn][colIn] != value:
                return(False)
            if [rowIn,colIn] in tempV:
                #print('hi')
                if length >= 3:
                    return(True)
                else:
                    return(False)
            else:
                tempV.append([rowIn,colIn])
                visited [rowIn][colIn] =1
                temp1, temp2, temp3, temp4 = deepcopy(tempV), deepcopy(tempV), deepcopy(tempV), deepcopy(tempV)
                #print(temp1[0][0])
                output = False
                if rowIn<len(grid)-1 and previous != [rowIn+1,colIn]:
                    output = ExplorePath(rowIn+1,colIn, length+1, temp1, value, current)
                if colIn<len(grid[0])-1 and previous != [rowIn,colIn+1]:
                    output = output or ExplorePath(rowIn,colIn+1, length+1, temp2, value, current) 
                if rowIn>0 and previous != [rowIn-1,colIn]:
                    output = output or ExplorePath(rowIn-1,colIn, length+1, temp3, value, current)
                if colIn>0 and previous != [rowIn,colIn-1]:
                    output = output or ExplorePath(rowIn,colIn-1, length+1, temp4, value, current) 
                    
            return(output)
            

        
        for rowIn in range(len(grid)-1):
            for colIn in range(len(grid[0])-1):
                if grid[rowIn+1][colIn]== grid[rowIn][colIn]:
                    if grid[rowIn+1][colIn+1]== grid[rowIn][colIn]:
                        if grid[rowIn][colIn+1]== grid[rowIn][colIn]:
                            #print(rowIn,colIn)
                            return(True) 
        for rowIn in range(len(grid)):
            for colIn in range(len(grid[0])):
                
                if visited[rowIn][colIn]==0:
                    #print(grid[rowIn][colIn])
                    tempVisited = []
                    #print(tempVisited[0][0])
                    length = 0
                    if (ExplorePath(rowIn,colIn, length, tempVisited, grid[rowIn][colIn], [rowIn,colIn])):
                        return(True)
                    
        return(False)

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
    def containsCycle(self, A):
        R, C = len(A), len(A[0])
        def encode(r, c):
            return r * C + c
        
        dsu = DSU(R * C)
        for r in range(R):
            for c in range(C):
                if c + 1 < C and A[r][c] == A[r][c+1]:
                    if not dsu.union(encode(r, c), encode(r, c + 1)):
                        if dsu.size(encode(r, c)) >= 4:
                            return True
                if r + 1 < R and A[r][c] == A[r+1][c]:
                    if not dsu.union(encode(r, c), encode(r + 1, c)):
                        if dsu.size(encode(r, c)) >= 4:
                            return True
        return False

# bac
# cac
# ddc
# bcc

class UF:
    def __init__(self, m, n):    
        self.p = {(i, j): (i, j) for i in range(m) for j in range(n)}
        
    def union(self, ti, tj):
        pi, pj = self.find(*ti), self.find(*tj)
        if pi != pj:
            self.p[pj] = pi
            return False
        return True
            
    def find(self, i, j):
        if (i, j) != self.p[i,j]:  
            self.p[i,j] = self.find(*self.p[i,j])
        return self.p[i,j]    
                
class Solution:
    def containsCycle(self, grid: List[List[str]]) -> bool:
        # u5982u4f55u5b9au4e49cycleuff1au540cu4e00u4e2au70b9u7684u4e24u4e2au4e0du540cu65b9u5411u7684pathuff0cu6700u7ec8u6c47u805au5230u975eu81eau8eabu7684u540cu4e00u70b9
        m, n = len(grid), len(grid[0])
        uf = UF(m, n)
        for i in range(m):
            for j in range(n):
                if i > 0 and grid[i][j] == grid[i-1][j]:
                    uf.union((i-1, j), (i, j))
                if j > 0 and grid[i][j] == grid[i][j-1]:
                    if uf.union((i, j-1), (i, j)): return True
        return False            
class Solution:
    def containsCycle(self, grid) -> bool:
        def find(pos):
            if parents[pos] != pos:
                parents[pos] = find(parents[pos])
            return parents[pos]

        def union(pos1, pos2):
            parent1, parent2 = find(pos1), find(pos2)
            if parent1 != parent2:
                if ranks[parent2] > ranks[parent1]:
                    parents[parent1] = parent2
                else:
                    parents[parent2] = parent1
                    if ranks[parent1] == ranks[parent2]:
                        ranks[parent1] += 1

        rows, cols = len(grid), len(grid[0])
        parents = {(i, j): (i, j) for i in range(rows) for j in range(cols)}
        ranks = collections.Counter()
        for i, row in enumerate(grid):
            for j, letter in enumerate(row):
                if i > 0 and j > 0 and grid[i-1][j] == grid[i][j-1] == letter and find((i-1, j)) == find((i, j-1)):
                    return True
                for r, c in (i - 1, j), (i, j - 1):
                    if r >= 0 and c >= 0 and grid[r][c] == letter:
                        union((i, j), (r, c))
        return False
class Solution:
    def containsCycle(self, grid: List[List[str]]) -> bool:
        M, N = len(grid), len(grid[0])
        dp = [[set() for j in range(N)] for i in range(M)]
        dv = [(0,-1),(-1,0)]
        for i in range(M):
            for j in range(N): 
                for x,y in dv:
                    if 0 <= i+x < M and 0 <= j+y < N:
                        if grid[i][j] == grid[i+x][j+y]:
                            if not dp[i+x][j+y]:
                                dp[i][j].add(((i+x,j+y),1))
                            else:
                                for entry in dp[i+x][j+y]:
                                    prnt, dist = entry
                                    if (prnt, dist+1) in dp[i][j] and dist+1 >= 2:
                                        return True
                                    dp[i][j].add((prnt, dist+1))
                                    #print(i,j, prnt, dist, dp[i][j])
        return False

    

class Solution:
    def containsCycle(self, grid) -> bool:
        if not grid or not grid[0]: return False
        
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        visited=[[False] * len(grid[i]) for i in range(len(grid))]

        def dfs(grid, symbol, row, col, prev_row, prev_col):
            nonlocal directions
            nonlocal visited
            
            visited[row][col] = True
            
            for direction in directions:
                nr = row + direction[0]
                nc = col + direction[1]

                if nr == prev_row and nc == prev_col:
                    continue
                if len(grid) > nr >= 0 and len(grid[nr]) > nc >= 0:
                    if grid[nr][nc] == symbol and (visited[nr][nc] or dfs(grid, symbol, nr, nc, row , col)):
                        return True            
            return False
                
        for i in range(len(grid)):
            for j in range(len(grid[i])):
                if not visited[i][j]:
                    if dfs(grid, grid[i][j], i, j, 0, 0):
                        return True
        return False
class Solution:
    def containsCycle(self, grid: List[List[str]]) -> bool:
        m = len(grid)
        n = len(grid[0])

        class roots():
            def __init__(self, val):
                self.val = val
                self.par = None
            def root(self):
                if self.par == None: return self.val
                return self.par.root()

        chains = {}
        for i in range(m):
            for j in range(n):
                if i > 0 and grid[i-1][j] == grid[i][j]:
                    chains[(i,j)]=roots((i,j))
                    chains[(i,j)].par = chains[(i-1,j)]

                if j > 0 and grid[i][j-1] == grid[i][j]:
                    if (i,j) in chains:
                        if chains[(i,j)].root() == chains[(i,j-1)].root(): return True
                        else: chains[chains[(i,j-1)].root()].par = chains[(i,j)]
                    else:
                        chains[(i,j)]=roots((i,j))
                        chains[(i,j)].par = chains[(i,j-1)]
                if (i,j) not in chains:
                    chains[(i,j)]=roots((i,j))
        return False

class Solution:
    def containsCycle(self, grid: List[List[str]]) -> bool:
        n,m = len(grid),len(grid[0])
        root = {(i,j):(i,j) for i in range(n) for j in range(m)}   
        rank = collections.defaultdict(int)
        def find(x):
            if root[x]==x:
                return x
            root[x] == find(root[x])
            return root[x]
        
        def union(a,b):
            ra,rb = find(a),find(b)
            if ra!=rb:
                if rank[rb]>rank[ra]:
                    root[ra] = rb
                else:
                    root[rb] = ra
                    if rank[ra]==rank[rb]:
                        rank[ra]+=1
                
                
        
        for i in range(n):
            for j in range(m):
                #print(i,j)
                val = grid[i][j]
                # parent: i-1,j and i,j-1
                if i>0 and j>0 and grid[i-1][j] == grid[i][j-1]==val and find((i-1,j))==find((i,j-1)):
                    #print(find((i-1,j)),find((i,j-1)))
                    return True
                for ni,nj in [(i-1,j),(i,j-1)]:
                    if 0<=ni<n and 0<=nj<m and grid[ni][nj]==val:
                        #print((i,j),(ni,nj),val,grid[ni][nj])
                        union((i,j),(ni,nj))
                        #print(find((ni,nj)))
                
        return False

class Solution:
    def containsCycle(self, grid: List[List[str]]) -> bool:
        
        row , col = len(grid), len(grid[0])
        
        
        track = {}
        dx, dy = [0,0,-1,1], [-1,1,0,0]
        def check(i,j, parent):
            
            # print('outside',i,j)
            
            for k in range(4):
                xx, yy = dx[k] + i, dy[k] + j
                if xx in range(row) and yy in range(col) and parent!=(xx,yy) and grid[i][j]== grid[xx][yy] and grid[i][j]!='checked':
                    if (xx,yy) in track:
                        return True
                    track[(xx,yy)] = True
                    if check(xx,yy,(i,j)): return True
                    grid[xx][yy]='checked'
            return False
            
            
            
            
            
        for i in range(row):
            for j in range(col):
                track[(i,j)] = True
                if check(i,j,None): return True
                grid[i][j] = 'checked'
        return False
class Solution:
    def find(self, n1):
        if n1 == self.par[n1[0]][n1[1]]:
            return n1
        else:
            return self.find(self.par[n1[0]][n1[1]])
        
    def union(self, n1, n2):
        p1 = self.find(n1)
        p2 = self.find(n2)
        self.par[p1[0]][p1[1]] = p2
        
    def containsCycle(self, grid: List[List[str]]) -> bool:
        n_row, n_col = len(grid), len(grid[0])
        self.par = []
        for r in range(n_row):
            self.par.append([(r,c) for c in range(n_col)])
            
        for r in range(n_row):
            for c in range(n_col):
                # check right
                if r+1 < n_row and grid[r][c]==grid[r+1][c]:
                    if self.find((r,c)) == self.find((r+1,c)):
                        return True
                    self.union((r,c), (r+1,c))
                # check down
                if c+1 < n_col and grid[r][c]==grid[r][c+1]:
                    if self.find((r,c)) == self.find((r,c+1)):
                        return True
                    self.union((r,c), (r,c+1))
            
       
        return False
            

class Solution:
    def containsCycle(self, grid: List[List[str]]) -> bool:
        def find(pos):
            if parents[pos] != pos:
                parents[pos] = find(parents[pos])
            return parents[pos]

        def union(pos1, pos2):
            parent1, parent2 = find(pos1), find(pos2)
            if parent1 != parent2:
                parents[parent2] = parent1

        rows, cols = len(grid), len(grid[0])
        parents = {(i, j): (i, j) for i in range(rows) for j in range(cols)}
        for i, row in enumerate(grid):
            for j, letter in enumerate(row):
                if i > 0 and j > 0 and grid[i-1][j] == grid[i][j-1] == letter and find((i-1, j)) == find((i, j-1)):
                    return True
                for r, c in (i - 1, j), (i, j - 1):
                    if 0 <= r < rows and 0 <= c < cols and grid[r][c] == letter:
                        union((i, j), (r, c))
        return False
    
        

class Solution:
    def containsCycle(self, grid: List[List[str]]) -> bool:
        def find(coords):
            if coords != parents[coords]:
                parents[coords] = find(parents[coords])
            return parents[coords]
        
        def union(point_one, point_two):
            parx, pary = find(point_one), find(point_two)
            if parx != pary:
                if rank[parx] > rank[pary]:
                    parents[pary] = parx
                else:
                    parents[parx] = pary
                    if rank[parx] == rank[pary]:
                        rank[parx] += 1
                # elif rank[parx] < rank[pary]:
                #     parents[parx] = pary
                # else:
                #     parents[parx] = pary
                #     rank[pary] += 1

        if not grid or not grid[0]:
            return False
        m, n = len(grid), len(grid[0])
        
        parents = {(i, j): (i, j) for i in range(m) for j in range(n)}
        rank = collections.Counter()
        
        for i in range(m):
            for j in range(n):
                if i > 0 and j > 0 and grid[i-1][j] == grid[i][j-1] == grid[i][j] and find((i-1, j)) == find((i, j-1)):
                    return True
                for r, c in (i-1, j), (i, j-1):
                    if r >= 0 and c >= 0 and grid[r][c] == grid[i][j]:
                        union((r, c), (i, j))
                        
        return False
class Solution:
    def containsCycle(self, grid: List[List[str]]) -> bool:
        
        
        
        # public solution ... 3020 ms ... 0 % ... 215 MB ... 0 %
        #  time: O(n*m)
        # space: O(n*m)
        
        def dfs(pi, pj, i, j):
            visited[i][j] = True
            for ni, nj in [(i-1, j), (i, j-1), (i, j+1), (i+1, j)]:
                if 0 <= ni < n and 0 <= nj < m and grid[ni][nj] == grid[i][j]:
                    if visited[ni][nj]:
                        if (ni,nj) != (pi,pj):
                            return True
                    else:
                        if dfs(i, j, ni, nj):
                            return True
            return False
        
        n, m = len(grid), len(grid[0])
        if n < 2 or m < 2:
            return False
        visited = [[False]*m for _ in range(n)]
        for i in range(n):
            for j in range(m):
                if not visited[i][j] and dfs(-1, -1, i, j):
                    return True
        return False
        
        

class Solution:
    
    def detect_cycle(self, grid, visited, rec_stack, i, j, M, N, prev):
        visited[i][j] = True
        rec_stack.append([i, j])
        for k, l in [[i, j-1], [i-1, j], [i, j+1], [i+1, j]]:
            if k >= 0 and l >= 0 and k < M and l < N and grid[i][j] == grid[k][l] and [k, l] != prev:
                if not visited[k][l]:
                    if self.detect_cycle(grid, visited, rec_stack, k, l, M, N, [i, j]) == True:
                        return True
                elif [k, l] in rec_stack:
                    return True
        rec_stack.pop()
        
    def containsCycle(self, grid: List[List[str]]) -> bool:
        M, N = len(grid), len(grid[0])
        visited = [[False]*N for _ in range(M)]
        rec_stack = []
        for i in range(M):
            for j in range(N):
                if not visited[i][j]:
                    if self.detect_cycle(grid, visited, rec_stack, i, j, M, N, [i, j]) == True:
                        return True
        return False



class Solution:
    def containsCycle(self, grid: List[List[str]]) -> bool:
        n,m = len(grid),len(grid[0])
        root = {(i,j):(i,j) for i in range(n) for j in range(m)}   

        def find(x):
            i,j = x
            if root[i,j]==x:
                return x
            root[i,j] = find(root[i,j])
            return root[i,j]
        
        def union(a,b):
            ra,rb = find(a),find(b)
            if ra!=rb:
                root[rb] = ra
                return False
            return True
        
        for i in range(n):
            for j in range(m):
                val = grid[i][j]
                if i > 0 and grid[i][j] == grid[i-1][j]:
                    union((i-1, j), (i, j))
                if j > 0 and grid[i][j] == grid[i][j-1]:
                    if union((i, j-1), (i, j)): 
                        return True                
        return False

from collections import defaultdict
class Solution:
    DELTA = [(0,-1),(0,1),(-1,0),(1,0)]
    def containsCycle(self, grid: List[List[str]]) -> bool:
        def dfsCycle(grid, R, C, p_r, p_c, r, c, visited, grp, grp_num):
            if (r, c) in grp[grp_num]:
                return True
            
            # check 4 directions
            visited.add((r,c))
            grp[grp_num].add((r,c))
            result = False
            for d in Solution.DELTA:
                n_r = r + d[0]
                n_c = c + d[1]
                if 0 <= n_r < R and 0 <= n_c < C and not (p_r == n_r and p_c == n_c) and grid[n_r][n_c] == grid[r][c]:
                    result |= dfsCycle(grid, R, C, r, c, n_r, n_c, visited, grp, grp_num)
                    if result:
                        break
                    
            return result
            
        R = len(grid)
        C = len(grid[0])
        visited = set()
        grp_num = 0
        grp = defaultdict(set)
        for r in range(R):
            for c in range(C):
                if (r,c) not in visited:
                    grp_num += 1
                    if dfsCycle(grid, R, C, r, c, r, c, visited, grp, grp_num):
                        return True
                    
        return False


class Solution:
    def containsCycle(self, grid: List[List[str]]) -> bool:
        n,m = len(grid),len(grid[0])
        root = {(i,j):(i,j) for i in range(n) for j in range(m)}
        
        def find(x):
            if root[x]==x:
                return x
            root[x] = find(root[x])
            return root[x]
        
        def union(a,b):
            ra,rb = find(a),find(b)
            if ra!=rb:
                root[ra] = rb
                return False
            return True
        
        for i in range(n):
            for j in range(m):
                val = grid[i][j]
                for ni,nj in [(i+1,j),(i,j+1)]:
                    if ni<n and nj<m and grid[ni][nj]==val:
                        t = union((ni,nj),(i,j))
                        if t:
                            return True
        return False
class Solution:
    def containsCycle(self, grid: List[List[str]]) -> bool:
        def ff(grid,i,j,n,c):
            grid[i][j]=c
            c+=1
            if i>0 :
                    #print(grid)
                    if type(grid[i-1][j])==int and c-grid[i-1][j]>2 and c//250000==grid[i-1][j]//250000:
                        return True
                    elif type(grid[i-1][j])==str and grid[i-1][j]==n:
                        
                        #print(grid)
                        if ff(grid,i-1,j,n,c):
                            return True
            if j<len(grid[i])-1: 
                    #print(grid)
                    if type(grid[i][j+1])==int and c-grid[i][j+1]>2 and c//250000==grid[i][j+1]//250000:
                        return True
                    elif type(grid[i][j+1])==str and grid[i][j+1]==n:
                        
                        if ff(grid,i,j+1,n,c):
                            return True
            if j>0:
                    #print(grid,n)
                    if type(grid[i][j-1])==int and c-grid[i][j-1]>2 and c//250000==grid[i][j-1]//250000:
                        return True
                    elif type(grid[i][j-1])==str and grid[i][j-1]==n:
                        
                        if ff(grid,i,j-1,n,c):
                            return True
            if i<len(grid)-1  :
                    #print(grid,n)
                    
                    if type(grid[i+1][j])==int and c-grid[i+1][j]>2 and c//250000==grid[i+1][j]//250000:
                        return True
                    elif type(grid[i+1][j])==str and grid[i+1][j]==n:
                        #print(grid)
                        
                        if ff(grid,i+1,j,n,c):
                            return True
            return False
        cc=0
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if type(grid[i][j])!=int:
                    cc+=1
                    if ff(grid,i,j,grid[i][j],cc*250000):
                        return True
        return False

import collections
class Solution:
    def containsCycle(self, grid: List[List[str]]) -> bool:
        def find(pos):
            if parents[pos] != pos:
                parents[pos] = find(parents[pos])
            return parents[pos]

        def union(pos1, pos2):
            parent1, parent2 = find(pos1), find(pos2)
            if parent1 != parent2:
                if ranks[parent2] > ranks[parent1]:
                    parents[parent1] = parent2
                else:
                    parents[parent2] = parent1
                    if ranks[parent1] == ranks[parent2]:
                        ranks[parent1] += 1

        rows, cols = len(grid), len(grid[0])
        parents = {(i, j): (i, j) for i in range(rows) for j in range(cols)}
        ranks = collections.Counter()
        for i, row in enumerate(grid):
            for j, letter in enumerate(row):
                if i > 0 and j > 0 and grid[i-1][j] == grid[i][j-1] == letter and find((i-1, j)) == find((i, j-1)):
                    return True
                for r, c in (i - 1, j), (i, j - 1):
                    if 0 <= r < rows and 0 <= c < cols and grid[r][c] == letter:
                        union((i, j), (r, c))
        return False
class UnionFindSet:
    def __init__(self, n):
        self.parents = list(range(n))
        self.ranks = [1] * n
        
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
        return True
    
    
class Solution:
    def containsCycle(self, grid: List[List[str]]) -> bool:
        m, n = len(grid), len(grid[0])
        dirs = ((0, 1), (0, -1), (1, 0), (-1, 0))
        uf = UnionFindSet(m*n)
        for i in range(m):
            for j in range(n):
                if i > 0 and grid[i][j] == grid[i-1][j] and not uf.union(i*n+j, (i-1)*n+j):
                    return True
                if j > 0 and grid[i][j] == grid[i][j-1] and not uf.union(i*n+j, i*n+j-1):
                    return True
        return False
class Solution:
    def containsCycle(self, grid: List[List[str]]) -> bool:
        parents = {}
        def find(p):
            if p not in parents:
                parents[p] = p
            if parents[p] != p:
                parents[p] = find(parents[p])
            return parents[p]
        def is_connected(p, q):
            return find(p) == find(q)
        def union(p, q):
            i, j = find(p), find(q)
            parents[j] = i
            
        R, C = len(grid), len(grid[0])
        
        for r in range(R):
            for c in range(C):
                for nr, nc in [r+1, c], [r, c+1]:
                    if nr < R and nc < C and grid[r][c] == grid[nr][nc]:
                        if is_connected((r, c), (nr, nc)):
                            return True
                        union((r, c), (nr, nc))
        
        return False
class Solution:
    def containsCycle(self, grid: List[List[str]]) -> bool:
        
        visited = {}
        
        def dfs(x, y, u, v):
            
            if not 0 <= x < len(grid): return False
            
            if not 0 <= y < len(grid[x]): return False
            
            if grid[x][y] != grid[u][v]: return False 
            
            if (x,y) in visited:
                return True
            
            visited[(x,y)] = True
            
            if (x, y+1) != (u,v) and dfs(x, y+1,x,y):
                return True
                
            if (x-1, y) != (u,v) and dfs(x-1, y,x,y):
                return True
            
            if (x+1, y) != (u,v) and dfs(x+1, y, x, y):
                return True
                
            if (x, y-1) != (u,v) and dfs(x, y-1, x, y):
                return True
            
            return False
            
        for i in range(len(grid)):
            for j in range(len(grid[i])):
                if (i,j) not in visited and dfs(i,j,i,j):
                    return True
                
        return False
class Solution:
    def containsCycle(self, grid: List[List[str]]) -> bool:
        if not grid: return False
        m, n = len(grid), len(grid[0])
        
        nbrs = [(-1,0), (0,1),(0,-1),(1, 0)]
        def isValid(x,y):
            return x>=0 and x<m and y>=0 and y<n
        
        def hasCycle(x, y, vis, parentX, parentY):
            vis.add((x,y))
            for nbr in nbrs:
                newX, newY = x + nbr[0], y+nbr[1]
                if isValid(newX, newY) and grid[newX][newY] == grid[x][y] and not(parentX == newX and parentY == newY):
                    if (newX, newY) in vis:
                        return True
                    else:
                        if hasCycle(newX, newY, vis,x,y):
                            return True
            return False
        vis = set()
        for i in range(m):
            for j in range(n):
                if (i,j) not in vis:
                    ans = hasCycle(i,j, vis, -1, -1)
                if ans: return True
        
        return False

class Solution:
    def containsCycle(self, grid) -> bool:
        if not grid or not grid[0]: return False
        
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
                
        for i in range(len(grid)):
            for j in range(len(grid[i])):
                if '#' not in grid[i][j]:
                    stack = [(i, j, 0, 0)]
                    symbol = grid[i][j]
                    grid[i][j] = '#' + symbol
                    
                    while stack:
                        row, col, prev_row, prev_col = stack.pop()

                        for direction in directions:
                            nr = row + direction[0]
                            nc = col + direction[1]

                            if nr == prev_row and nc == prev_col:
                                continue
                            if len(grid) > nr >= 0 and len(grid[nr]) > nc >= 0 and symbol in grid[nr][nc]:
                                if '#' in grid[nr][nc]:
                                    return True
                                else:
                                    grid[nr][nc] = '#' + symbol
                                    stack.append((nr, nc, row, col))
                      
        return False
class Solution:
    def find(self, n):
        if n == self.par[n[0]][n[1]]:
            return n
        else:
            p = self.find(self.par[n[0]][n[1]])
            # path compression
            self.par[n[0]][n[1]] = p
            return p
        
    def union(self, n1, n2):
        p1 = self.find(n1)
        p2 = self.find(n2)
        
        # union by rank
        if self.rank[p1[0]][p1[1]] > self.rank[p2[0]][p2[1]]:
            self.par[p2[0]][p2[1]] = p1
        else:
            self.par[p1[0]][p1[1]] = p2
            self.rank[p1[0]][p1[1]] += 1
        
    def containsCycle(self, grid: List[List[str]]) -> bool:
        n_row, n_col = len(grid), len(grid[0])

        self.rank = [[1]*n_col for _ in range(n_row)]
        
        self.par = []
        for r in range(n_row):
            self.par.append([(r,c) for c in range(n_col)])
            
        for r in range(n_row):
            for c in range(n_col):
                # check right
                if r+1 < n_row and grid[r][c]==grid[r+1][c]:
                    if self.find((r,c)) == self.find((r+1,c)):
                        return True
                    self.union((r,c), (r+1,c))
                # check down
                if c+1 < n_col and grid[r][c]==grid[r][c+1]:
                    if self.find((r,c)) == self.find((r,c+1)):
                        return True
                    self.union((r,c), (r,c+1))
            
       
        return False
class Solution:
    def containsCycle(self, grid: List[List[str]]) -> bool:
        dsu = DSU()
        m, n = len(grid), len(grid[0])
        for i in range(m):
            for j in range(n):
                if i != 0 and grid[i - 1][j] == grid[i][j]:
                    dsu.union((i - 1, j), (i, j))
                if grid[i][j - 1] == grid[i][j]:
                    if dsu.find((i, j - 1)) == dsu.find((i, j)):
                        return True
                    dsu.union((i, j - 1), (i, j))
        return False
        
class DSU:
    def __init__(self):
        self.father = {}
    
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
class Solution:
    def isCycle(self, grid, r, c, visited, pr, pc):
        nrow, ncol = len(grid), len(grid[0])
        direcs = [(0,1),(0,-1),(-1,0),(1,0)]
        visited.add((r, c))
        for dr, dc in direcs:
            nr, nc = r + dr, c + dc
            if 0 <= nr < nrow and 0 <= nc < ncol and grid[nr][nc] == grid[r][c] and not (pr==nr and pc==nc):
                if (nr, nc) in visited:
                    return True
                if self.isCycle(grid, nr, nc, visited, r, c):
                    return True
                
        return False
            
        
    def containsCycle(self, grid: List[List[str]]) -> bool:
        nrow, ncol = len(grid), len(grid[0])
        visited = set()
        for r in range(nrow):
            for c in range(ncol):
                if (r, c) in visited:
                    continue
                if self.isCycle(grid, r, c, visited, -1, -1):
                    return True
        return False
class UF:
    def __init__(self, m, n):    
        self.p = {(i, j): (i, j) for i in range(m) for j in range(n)}
        
    def union(self, ti, tj):
        pi, pj = self.find(*ti), self.find(*tj)
        if pi != pj:
            self.p[pj] = pi
            return False
        return True
            
    def find(self, i, j):
        if (i, j) != self.p[i,j]:  
            self.p[i,j] = self.find(*self.p[i,j])
        return self.p[i,j]    
                
class Solution:
    def containsCycle(self, grid: List[List[str]]) -> bool:
        # u5982u4f55u5b9au4e49cycleuff1au540cu4e00u4e2au70b9u7684u4e24u4e2au4e0du540cu65b9u5411u7684pathuff0cu6700u7ec8u6c47u805au5230u975eu81eau8eabu7684u540cu4e00u70b9
        m, n = len(grid), len(grid[0])
        uf = UF(m, n)
        for i in range(m):
            for j in range(n):
                if i > 0 and grid[i][j] == grid[i-1][j]:
                    if uf.union(tuple([i-1, j]), tuple([i, j])): return True
                if j > 0 and grid[i][j] == grid[i][j-1]:
                    if uf.union(tuple([i, j-1]), tuple([i, j])): return True
        return False            
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

class Solution:
    def containsCycle(self, A):
        R, C = len(A), len(A[0])
        def encode(r, c):
            return r * C + c
        
        dsu = DSU(R * C)
        for r in range(R):
            for c in range(C):
                if c + 1 < C and A[r][c] == A[r][c+1]:
                    if not dsu.union(encode(r, c), encode(r, c + 1)):
                        return True
                if r + 1 < R and A[r][c] == A[r+1][c]:
                    if not dsu.union(encode(r, c), encode(r + 1, c)):
                        return True
        return False

import collections
class Solution:
    def containsCycle(self, grid: List[List[str]]) -> bool:
        def find(pos):
            if parents[pos] != pos:
                parents[pos] = find(parents[pos])
            return parents[pos]

        def union(pos1, pos2):
            parent1, parent2 = find(pos1), find(pos2)
            if parent1 != parent2:
                if ranks[pos1] > ranks[pos2]:
                    parents[pos2] = pos1
                else:
                    parents[pos1] = pos2
                    if ranks[pos1] == ranks[pos2]:
                        ranks[pos1] += 1

        rows, cols = len(grid), len(grid[0])
        parents = {(i, j): (i, j) for i in range(rows) for j in range(cols)}
        ranks = collections.Counter()
        for i, row in enumerate(grid):
            for j, letter in enumerate(row):
                if i > 0 and j > 0 and grid[i-1][j] == grid[i][j-1] == letter and find((i-1, j)) == find((i, j-1)):
                    return True
                for r, c in (i - 1, j), (i, j - 1):
                    if 0 <= r < rows and 0 <= c < cols and grid[r][c] == letter:
                        union((i, j), (r, c))
        return False

class Solution:
    def rec(self,i,j,char,prev):
        if self.grid[i][j] != char:
            return False
        if self.matrix[i][j]:
            return True
        else:
            self.matrix[i][j] = True
            
        l = (i,j-1)
        r = (i,j+1)
        u = (i-1,j)
        d = (i+1,j)
        
        for c in [l,r,u,d]:
            if 0<=c[0]<self.row and 0<=c[1]<self.col and c != prev:
                if self.rec(c[0],c[1], char, (i,j)):
                    return True
        return False
            

    def containsCycle(self, grid: List[List[str]]) -> bool:
        self.grid = grid
        self.row = len(grid)
        self.col = len(grid[0])

        self.matrix = [ [False for i in range(self.col)] for j in range(self.row)  ]

        for i in range(self.row):
            for j in range(self.col):
                if not self.matrix[i][j]:
                    if self.rec(i,j, grid[i][j],0):
                        return True
        return False
class Solution:
    def containsCycle(self, grid: List[List[str]]) -> bool:
        m, n = len(grid), len(grid[0])
        dsu = DUS()
        for i in range(m):
            for j in range(n):
                if i == 0 and j == 0:
                    continue
                if j != 0 and grid[i][j] == grid[i][j - 1]:
                    dsu.union((i, j), (i, j - 1))
                    
                if i != 0 and grid[i][j] == grid[i - 1][j]:
                    if dsu.find((i,j)) == dsu.find((i - 1, j)):
                        return True
                    dsu.union((i, j), (i - 1, j))
        return False
        
class DUS:
    def __init__(self):
        self.father = {}
    
    def find(self, a):
        self.father.setdefault(a, a)
        if a != self.father[a]:
            self.father[a] = self.find(self.father[a])
        return self.father[a]
    
    def union(self, a, b):
        _a = self.find(a)
        _b = self.find(b)
        if _a != _b:
            self.father[_a] = _b
class Solution:
    def containsCycle(self, grid: List[List[str]]) -> bool:
        m, n = len(grid), len(grid[0])
        uf = {(i, j): (i, j) for i in range(m) for j in range(n)}
        
        def find(pos):
            if pos != uf[pos]:
                uf[pos] = find(uf[pos])
            return uf[pos]
        
        def union(pos1, pos2):
            root1 = find(pos1)
            root2 = find(pos2)
            if root1 != root2:
                uf[root1] = root2
            
        for i in range(m):
            for j in range(n):
                if i > 0 and j > 0 and find((i - 1, j)) == find((i, j - 1)) and grid[i][j] == grid[i-1][j] == grid[i][j-1]:
                    return True
                if i > 0 and grid[i][j] == grid[i-1][j]:
                    union((i, j), (i - 1, j))
                if j > 0 and grid[i][j] == grid[i][j-1]:
                    union((i, j), (i, j - 1))
        return False
class UnionFind:
    def __init__(self, m: int, n: int):
        self.rank = collections.Counter()
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
    def containsCycle(self, grid: List[List[str]]) -> bool:
        rows, cols = len(grid), len(grid[0])
        uf = UnionFind(rows, cols)
        for i, row in enumerate(grid):
            for j, letter in enumerate(row):
                if (i > 0 and j > 0 and
                        grid[i - 1][j] == grid[i][j - 1] == letter and
                        uf.find((i - 1, j)) == uf.find((i, j - 1))):
                    return True
                for r, c in (i - 1, j), (i, j - 1):
                    if 0 <= r and 0 <= c and grid[r][c] == letter:
                        uf.union((i, j), (r, c))
        return False


class Solution:
    def containsCycle(self, grid: List[List[str]]) -> bool:
        # dfs, remember parent
        self.m, self.n = len(grid), len(grid[0])
        nei = [(0, 1), (0, -1),(-1, 0), (1, 0)]
        memo = set()
        def dfs(i, j, p):
            # print(i, j, p, memo)
            if (i, j) in memo: return True
            memo.add((i, j))
            for ne in nei:
                x, y = i + ne[0], j + ne[1]
                if 0 <= x < self.m and 0 <= y < self.n and (x, y) != p and grid[i][j] == grid[x][y]:
                    if dfs(x, y, (i, j)): return True
            return False
            
        for i in range(self.m):
            for j in range(self.n):
                if (i, j) not in memo:
                    if dfs(i, j, (-1, -1)): return True
        return False

from collections import defaultdict
class Solution:
    DELTA = [(0,-1),(0,1),(-1,0),(1,0)]
    def containsCycle(self, grid: List[List[str]]) -> bool:
        def dfsCycle(grid, R, C, p_r, p_c, r, c, visited, grp, grp_num):
            if (r, c) in grp[grp_num]:
                return True
            # print('p_r=', p_r, 'p_c=', p_c)
            
            # check 4 directions
            visited.add((r,c))
            grp[grp_num].add((r,c))
            result = False
            for d in Solution.DELTA:
                n_r = r + d[0]
                n_c = c + d[1]
                # print('n_r=', n_r, 'n_c=', n_c)
                if 0 <= n_r < R and 0 <= n_c < C and not (p_r == n_r and p_c == n_c) and grid[n_r][n_c] == grid[r][c]:
                    result |= dfsCycle(grid, R, C, r, c, n_r, n_c, visited, grp, grp_num)
                    if result:
                        break
                    
            return result
            
        R = len(grid)
        C = len(grid[0])
        visited = set()
        grp_num = 0
        grp = defaultdict(set)
        for r in range(R):
            for c in range(C):
                if (r,c) not in visited:
                    grp_num += 1
                    # print('r=', r, 'c=', c, grid[r][c])
                    if dfsCycle(grid, R, C, r, c, r, c, visited, grp, grp_num):
                        return True
                    # print(grid)
                    
        return False
class Solution:
    def containsCycle(self, grid: List[List[str]]) -> bool:
        
        h, w = len(grid), len(grid[0])
        
        #------------------------------------------
        
        def in_range(x, y):
            
            return (0 <= x < w) and (0 <= y < h) 
        
        #------------------------------------------
        
        def four_neighbors(x, y):
            
            for dx, dy in {(+1, 0), (-1, 0), (0, +1), (0, -1)}:
                
                next_x, next_y = x + dx, y + dy
                
                if in_range(next_x, next_y):
                    yield (next_x, next_y)
            
        #------------------------------------------
        
        def dfs(x, y, prev_x, prev_y, grid):
            
            if grid[y][x] == dfs.cur_symbol:
                # this grid has a cycle with current symbol
                return True
            
            
            # mark to uppercase letter as visited
            grid[y][x] = dfs.cur_symbol
            
            
            for next_x, next_y in four_neighbors(x, y):
                
                if (next_x, next_y) == (prev_x, prev_y):
                    # avoid backward visit
                    continue
                    
                elif grid[next_y][next_x].upper() != dfs.cur_symbol:
                    # different symbol
                    continue
                    
                if dfs(next_x, next_y, x, y, grid): return True
            
            #print(f'terminate with {x} {y} {grid[y][x]}')
            return False
            
        
        #------------------------------------------
        
        failed_symbol = set()
        
        for y in range(h):
        #for y in reversed(range(h)):
            for x in range(w):
            #for x in reversed(range(w)):
                
                dfs.cur_symbol = grid[y][x]
                
                if dfs.cur_symbol in failed_symbol:
                    # skip search on failed symbol
                    continue
                
                dfs.cur_symbol = grid[y][x].upper()
                
                if dfs(x,y,-1,-1, grid):
                    return True
                
                else:
                    failed_symbol.add( dfs.cur_symbol )
        
        return False
class Solution:
    def containsCycle(self, grid: List[List[str]]) -> bool:
        m, n = len(grid), len(grid[0])
        ve = collections.defaultdict(set)
        for i, row in enumerate(grid):
            for j, val in enumerate(row):
                ve[val].add((i,j))
                
        # print(ve)
        dxy = [(-1,0), (1,0), (0,-1), (0,1)]
                
        def check(k):
            visiting = set()
            visited = set()
            v = ve[k]
            def dfs(curr, prev):
                if curr in visiting: return True
                visiting.add(curr)
                x,y = curr
                for dx, dy in dxy:
                    x2,y2 = x+dx, y+dy
                    if 0 <= x2 < m and 0 <= y2 < n and (x2,y2) != prev and (x2,y2) in v:
                        # print((x2,y2), curr)
                        if dfs((x2,y2), curr): return True
                visiting.remove(curr)
                visited.add(curr)
                return False
                
            for a in v:
                if a not in visited:
                    if dfs(a, None): return True
            return False
        
        for k in ve:
            if check(k): return True
        return False
class Solution:
    def containsCycle(self, grid: List[List[str]]) -> bool:
        
        visited = set()
        def dfs(x, y, parent):
            if (x, y) in visited:
                return True
            visited.add((x, y))
            c = grid[x][y]
            for dx, dy in [[0, -1], [0, 1], [-1, 0], [1, 0]]:
                nx, ny = dx + x, dy + y
                if 0 <= nx < len(grid) and 0 <= ny < len(grid[0]) and grid[nx][ny] == c and [nx, ny] != parent:
                    if dfs(nx, ny, [x, y]):
                        return True
            return False
        
        for x in range(len(grid)):
            for y in range(len(grid[0])):
                if (x, y) not in visited and dfs(x, y, [-1, -1]):
                    return True
        return False
class Solution:
    def containsCycle(self, grid: List[List[str]]) -> bool:
        visited = set()
        n, m = len(grid), len(grid[0])
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]

        def dfs(i, j, pre_i, pre_j):
            visited.add((i, j))
            for x, y in directions:
                if 0 <= x + i < n and 0 <= y + j < m and grid[i][j] == grid[i + x][j + y] and (i + x != pre_i or j + y != pre_j):
                    if (i + x, j + y) in visited or dfs(i + x, j + y, i, j):
                        return True
            return False
    
        for i in range(n):
            for j in range(m):
                if (i, j) not in visited:
                    if dfs(i, j, -1, -1):
                        return True
        return False
class Solution:
    def containsCycle(self, grid: List[List[str]]) -> bool:
        n = len(grid)
        m = len(grid[0])
        
        node_par = {}
        for i in range(n):
            for j in range(m):
                node_par[(i, j)] = (i, j)
                
        dxdys = ((0, 1), (0, -1), (1, 0), (-1, 0))
        
        def find_par(par):
            path = []
            while node_par[par] != par:
                path.append(par)
                par = node_par[par]
            for tmp in path:
                node_par[tmp] = par
            return par
        
        for x in range(n):
            for y in range(m):
                if (x + y) % 2:
                    continue
                    
                for dx, dy in dxdys:
                    x_new, y_new = x + dx, y + dy
                    if not (0 <= x_new < n and 0 <= y_new < m) or grid[x_new][y_new] != grid[x][y]:
                        continue
                    
                    curr_par = find_par((x, y))
                    new_par = find_par((x_new, y_new))
                    if curr_par == new_par:
                        return True
                    node_par[curr_par] = new_par
                    
        return False

from collections import deque
class Solution:
    
    def __init__(self):
        self.locs = defaultdict(set)
        self.grid = []
        
    def isValid(self, i, j):
        return i >= 0 and i < len(self.grid) and j >= 0 and j < len(self.grid[i])
    
    def hasCycle(self, l):
        seen = set()
        todo = deque([])
        around = [[0, 1], [0, -1], [1, 0], [-1, 0]]
        for x in l:
            if x not in seen:
                todo.append([(-1, -1), x])
                while todo:
                    node = todo.popleft()
                    cur = node[1]
                    fr = node[0]
                    if cur in seen:
                        
                        return True
                    seen.add(cur)
                    for x in around:
                        test = (cur[0] + x[0], cur[1] + x[1])
                        if self.isValid(test[0], test[1]) and test in l and not test == fr:
                           
                            todo.append([cur, test])
        return False
                    
        
        
    def containsCycle(self, grid: List[List[str]]) -> bool:
        self.grid = grid
        for x in range(len(grid)):
            for y in range(len(grid[x])):
                self.locs[grid[x][y]].add((x, y))
        for x in list(self.locs.keys()):
            if self.hasCycle(self.locs[x]):
                return True
        return False

class Solution:
    def containsCycle(self, grid: List[List[str]]) -> bool:
        # union find
        W, H = len(grid[0]), len(grid)
        
        parent = list(range(W * H))
        rank = [0] * (W * H)
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                if rank[px] > rank[py]:
                    parent[py] = px
                elif rank[px] < rank[py]:
                    parent[px] = py
                else:
                    parent[px] = py
                    rank[py] += 1
        
        for x in range(H):
            for y in range(W):
                if x and y and grid[x][y] == grid[x - 1][y] == grid[x][y - 1] and find((x - 1) * W + y) == find(x * W + y - 1):
                    return True
                
                for dx, dy in [(0, -1), (-1, 0)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < H and 0 <= ny < W and grid[nx][ny] == grid[x][y]:
                        union(x * W + y, nx * W + ny)
        
        return False
        
    def containsCycle_dfs(self, grid: List[List[str]]) -> bool:
        W, H = len(grid[0]), len(grid)
        
        visited = [[0] * W for _ in range(H)]
        def search(x, y, target, px, py):
            nonlocal grid, W, H, visited
            
            visited[x][y] = 1
            
            for dx, dy in [(0, 1), (1, 0), (-1, 0), (0, -1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < H and 0 <= ny < W and grid[nx][ny] == target:
                    
                    if visited[nx][ny]:
                        if nx == px and ny == py:
                            continue
                        return True

                    if search(nx, ny, target, x, y):
                        return True
            
            return False
        
        for x in range(H):
            for y in range(W):
                if not visited[x][y]:
                    if search(x, y, grid[x][y], None, None):
                        return True
        
        return False
class Solution:
    def containsCycle(self, grid: List[List[str]]) -> bool:
        visited = set()
        m = len(grid)
        if m==1: return False
        n = len(grid[0])
        dirs = [(0,-1),(-1,0),(0,1),(1,0)]
        def dfs(prev, curr):
            if curr in visited: return True
            visited.add(curr)
            for dirn in dirs:
                nei = (dirn[0]+curr[0], dirn[1]+curr[1])
                if 0<=nei[0]<m and 0<=nei[1]<n and nei != prev and grid[nei[0]][nei[1]] == grid[curr[0]][curr[1]]:
                    if dfs(curr, nei): return True
            return False
        for i in range(m):
            for j in range(n):
                if (i,j) not in visited:
                    if dfs(None, (i,j)): return True
        return False
class DSU:
    def __init__(self, n):
        self.par = list(range(n))
        self.count = [1]*n
    
    def find(self, u):
        if u != self.par[u]:
            self.par[u] = self.find(self.par[u])
        return self.par[u]
    
    def union(self, u, v):
        p1, p2 = self.find(u), self.find(v)
        if p1 == p2:
            return False
        
        if self.count[p1] < self.count[p2]:
            p1, p2 = p2, p1
        
        self.count[p1] += self.count[p2]
        self.count[p2] = self.count[p1]
        self.par[p2] = p1
        return True
    

class Solution:
    def containsCycle(self, grid: List[List[str]]) -> bool:
        
        # dsu for each cell
        m, n = len(grid), len(grid[0])
        dsu = DSU(m*n)
        
        for i in range(m):
            for j in range(n):
                for p,q in [(i+1,j), (i,j+1)]:
                    if 0 <= p < m and 0 <= q < n and grid[i][j] == grid[p][q]:
                        if not dsu.union(i*n+j, p*n+q):
                            return True
        
        print((dsu.count))
        return False
        
                            
                        
                

class Solution:
    
    def search_cycle(self, grid, i, j, parents):
        key = grid[i][j]
        parents[i, j] = (None, None)
        nodes = [(i, j)]
        visited = set()
        while len(nodes):
            i, j = nodes.pop()
            visited.add((i, j))
            pi, pj = parents[i, j]
            for ci, cj in [
                (i+1, j),
                (i-1, j),
                (i, j+1),
                (i, j-1),
            ]:
                in_range = 0<=ci<len(grid) and 0<=cj<len(grid[ci])
                is_same_key = in_range and grid[ci][cj] == key
                if ci == pi and cj == pj:
                    continue
                if in_range and is_same_key:
                    if (ci, cj) in visited:
                        return True
                    parents[ci, cj] = (i, j)
                    nodes.append((ci, cj))
        return False
        
                
    def containsCycle(self, grid: List[List[str]]) -> bool:
        
        parents = {}
        for i in range(len(grid)):
            for j in range(len(grid[i])):
                if (i, j) in parents:
                    continue
                is_cycle = self.search_cycle(grid, i, j, parents)
                if is_cycle:
                    return True
        return False
class Solution:
    def containsCycle(self, g: List[List[str]]) -> bool:
        r = len(g)
        c = len(g[0])
        
        cc = {}
        
        def root(x):
            if cc[x] != x:
                cc[x] = root(cc[x])
            return cc[x]
        
        def join(x, y):
            rx = root(x)
            ry = root(y)
            if rx != ry:
                cc[rx] = ry
            return rx != ry
        
        for i in range(r):
            for j in range(c):
                cc[i,j] = (i,j)
                
        for i in range(r):
            for j in range(c):
                for di, dj in [[0,1], [1,0]]:
                    ni,nj = i + di, j + dj
                    if 0 <= ni < r and 0 <= nj < c and g[i][j] == g[ni][nj]:
                        if not join((i,j), (ni,nj)):
                            return True
        return False
                        
                        
        

class Solution:
    def containsCycle(self, grid: List[List[str]]) -> bool:
        R, C = len(grid), len(grid[0])
        UF = {}
        def find(u):
            if UF[u] != u:
                UF[u] = find(UF[u])
            return UF[u]
        
        def union(u, v):
            UF.setdefault(u, u)
            UF.setdefault(v, v)
            UF[find(u)] = find(v)
        
        for i in range(R):
            for j in range(C):
                if i > 0 and grid[i][j] == grid[i - 1][j]:
                    if (i, j) in UF and (i - 1, j) in UF and find((i, j)) == find((i - 1, j)): return True
                    union((i, j), (i - 1, j))
                if j > 0 and grid[i][j] == grid[i][j - 1]:
                    if (i, j) in UF and (i, j - 1) in UF and find((i, j)) == find((i, j - 1)): return True
                    union((i, j), (i, j - 1))
        return False
import sys
sys.setrecursionlimit(250000)
class Solution:
    def containsCycle(self, grid: List[List[str]]) -> bool:
        m, n = len(grid), len(grid[0])
        
        def valid(x, y):
            if x < m and x >= 0 and y < n and y >= 0:
                return True
            else:
                return False
        
        def dfs(x, y, parent_x, parent_y):
            visit[x][y] = 1
            
            for d in ((0, 1), (0, -1), (-1, 0), (1, 0)):
                new_x = x + d[0]
                new_y = y + d[1]
                if valid(new_x, new_y) and grid[new_x][new_y] == grid[x][y] and (not (parent_x == new_x and parent_y == new_y)):
                    if visit[new_x][new_y] == 1:
                        return True
                    else:
                        cur = dfs(new_x, new_y, x, y)
                        if cur:
                            return True
            
            return False
        
        visit = [[0 for _ in range(n)] for _ in range(m)]
        
        res = False
        
        for i in range(m):
            if res:
                return True
            
            for j in range(n):
                if visit[i][j] == 0:
                    res = dfs(i, j, -1, -1)
                
                if res:
                    return True
        
        return False
import collections
class Solution:
    def containsCycle(self, grid: List[List[str]]) -> bool:
        '''
        # dfs, TLE. I think bfs will leads to TLE as well.
        rows, cols = len(grid), len(grid[0])
        def dfs(ch, s_r, s_c, row, col, seen, leng):
            for dr, dc in [[1, 0],[0, -1], [0, 1], [-1, 0]]:
                r, c = row + dr, col + dc
                if leng >= 4 and (r, c) == (s_r, s_c):
                    return True
                if 0 <= r < rows and 0 <= c < cols and grid[r][c] == ch and (r, c) not in seen and dfs(ch, s_r, s_c, r, c, seen | set([(r, c)]), leng + 1):
                    return True
            return False
                    
        for r in range(rows - 1):
            for c in range(cols - 1):
                if grid[r][c] == grid[r + 1][c] == grid[r][c + 1]:
                    if dfs(grid[r][c], r, c, r, c, set([(r, c)]), 1):
                        return True
        return False
        '''
        # Union Find, when you reach a char which is the same as current char and the two share the same
        # ancestor, then there is a ring
        rows, cols = len(grid), len(grid[0])
        seen = set()
        ancestors = dict()
        for r in range(rows):
            for c in range(cols):
                ancestors[(r, c)] = (r, c)
        def find(x, y):
            if ancestors[(x, y)] != (x, y):
                xx, yy = ancestors[(x, y)] 
                ancestors[(x, y)] = find(xx, yy)
            return ancestors[(x, y)]
        
        def union(x1, y1, x2, y2):
            # (x2, y2) is the new char that should be added to the group that (x1, y1) belongs to
            ancestors[find(x2, y2)] = find(x1, y1)
            
        for r in range(rows):
            for c in range(cols):
                if r == 0 and c == 0:
                    continue
                if r > 0 and c > 0 and grid[r - 1][c] == grid[r][c - 1] == grid[r][c] and find(r - 1, c) == find(r, c - 1):
                    return True
                if c > 0 and grid[r][c - 1] == grid[r][c]:
                    union(r, c - 1, r, c)
                if r > 0 and grid[r - 1][c] == grid[r][c]:
                    union(r, c, r - 1, c)
        return False        
                        
        
'''
[["a","a","a","a"],["a","b","b","a"],["a","b","b","a"],["a","a","a","a"]]
[["c","c","c","a"],["c","d","c","c"],["c","c","e","c"],["f","c","c","c"]]
[["a","b","b"],["b","z","b"],["b","b","a"]]
[["d","b","b"],["c","a","a"],["b","a","c"],["c","c","c"],["d","d","a"]]
'''
class Solution:
    def containsCycle(self, grid: List[List[str]]) -> bool:
        
        return self.unionfind(grid)
        
        '''
        self.grid = grid
        self.m = len(grid)
        self.n = len(grid[0])
        self.visited = set()
        
        for i in range(self.m):
            for j in range(self.n):
                
                if (i,j) not in self.visited:
                    if self.bfs((i,j), (-1,-1)):
                        return True
        return False

        '''
                
    def find(self, node):
        if self.parent[node[0]][node[1]] == node:
            return node
        else:
            p = self.find(self.parent[node[0]][node[1]])
            self.parent[node[0]][node[1]] = p
            return(p)
    
    def union(self, node1, node2):
        
        p1 = self.find(node1)
        p2 = self.find(node2)
        
        if self.rank[p1[0]][p1[1]] > self.rank[p2[0]][p2[1]]:
            self.parent[p2[0]][p2[1]] = p1
        elif self.rank[p2[0]][p2[1]] > self.rank[p1[0]][p1[1]]:
            self.parent[p1[0]][p1[1]] = p2
        else:
            self.parent[p1[0]][p1[1]] = p2
            self.rank[p2[0]][p2[1]] += 1
    
    def unionfind(self,g):
        
        nrow, ncol = len(g), len(g[0])
        
        self.parent = []
        self.rank = [[1]*ncol for _ in range(nrow)]
        
        for i in range(nrow):
                self.parent.append([(i,j) for j in range(ncol)])
        
        for i in range(nrow):
            for j in range(ncol):
                
                if i+1 < nrow and g[i][j] == g[i+1][j]:
                    
                    if self.find((i,j)) == self.find((i+1, j)):
                        return True
                    self.union((i,j), (i+1, j))
                
                if j+1 < ncol and g[i][j] == g[i][j+1]:
                    if self.find((i,j)) == self.find((i, j+1)):
                        return True
                    self.union((i,j), (i, j+1))
        return False
        
    
    def cycle(self, current, parent):
        
        if current in self.visited:
            return True
        
        self.visited.add(current)
        i,j = current
        neb = [(i+1,j), (i-1,j), (i, j+1), (i, j-1)]
        
        for ne in neb:
            ni, nj = ne
            if ne != parent and ni >= 0 and ni < self.m  and nj >=0 and nj < self.n and self.grid[ni][nj] == self.grid[i][j]:
                #print(ne)
                if self.cycle((ni, nj), current):
                    return True
        return False
                
    
    def bfs(self, current, parent):
        
        if current in self.visited:
            return True
        
        q = []
        q.append((current, parent))
        self.visited.add(current)
        
        while q:
            
            node, par = q.pop()
            #print(node)
            #print(par)
            i,j = node
            neb = [(i+1,j),(i-1,j), (i, j+1), (i,j-1)]
            
            for ni,nj in neb:
                if ni >= 0 and ni < self.m and nj >=0 and nj < self.n and self.grid[ni][nj] == self.grid[i][j] and (ni, nj) != par:
                    
                    if (ni, nj) in self.visited:
                        return True
                    q.append(((ni,nj), (i,j)))
                    self.visited.add((ni,nj))
        return False
                        
            
            
            
        
                

