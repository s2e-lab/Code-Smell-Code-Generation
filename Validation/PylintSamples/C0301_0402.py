import heapq

def solve(b,s,t):
    def create_priority_item(c, t):
        dx = c[0]-t[0]
        dy = c[1]-t[1]
        d2 = dx*dx + dy*dy
        return (d2, c)

    b = set(tuple(_b) for _b in b)
    s = tuple(s)
    t = tuple(t)
    # heap = [(-1,s)]
    heap = [s]
    visited = set()
    iter = -1
    while heap:
        iter += 1
        if iter > 1.1e6:
            return False
        # _, c = heapq.heappop(heap)
        c = heap.pop()
        if c in visited or c in b or c[0] < 0 or c[0] >=1e6 or c[1]<0 or c[1]>=1e6:
            continue
        if c == t:
            # found!
            return True
        # search neighbors:
        dx = c[0] - s[0]
        dy = c[1] - s[1]
        if dx*dx + dy*dy > 200*200:
            return True

        visited.add(c)


        # heapq.heappush(heap, create_priority_item((c[0]+1, c[1]  ), t))
        # heapq.heappush(heap, create_priority_item((c[0]-1, c[1]  ), t))
        # heapq.heappush(heap, create_priority_item((c[0]  , c[1]+1), t))
        # heapq.heappush(heap, create_priority_item((c[0]  , c[1]-1), t))
        heap.append((c[0]+1, c[1]  ))
        heap.append((c[0]-1, c[1]  ))
        heap.append((c[0]  , c[1]+1))
        heap.append((c[0]  , c[1]-1))
    # we live in a cavity :(
    return False

def solve_both(b,s,t):
    return solve(b,s,t) and solve(b,t,s)




class Solution:
    def isEscapePossible(self, blocked: List[List[int]], source: List[int], target: List[int]) -> bool:
        return solve_both(blocked, source, target)

class Solution:
     def isEscapePossible(self, blocked, source, target):
        blocked = {tuple(p) for p in blocked}

        def bfs(source, target):
            bfs, seen = [source], {tuple(source)}
            for x0, y0 in bfs:
                for i, j in [[0, 1], [1, 0], [-1, 0], [0, -1]]:
                    x, y = x0 + i, y0 + j
                    if 0 <= x < 10**6 and 0 <= y < 10**6 and (x, y) not in seen and (x, y) not in blocked:
                        if [x, y] == target: return True
                        bfs.append([x, y])
                        seen.add((x, y))
                if len(bfs) == 20000: return True
            return False
        return bfs(source, target) and bfs(target, source)

class Solution:
    def isEscapePossible(self, blocked: List[List[int]], source: List[int], target: List[int]) -> bool:
        blocked = {tuple(p) for p in blocked}

        def bfs(source, target):
            bfs, seen = [source], {tuple(source)}
            for x0, y0 in bfs:
                for i, j in [[0, 1], [1, 0], [-1, 0], [0, -1]]:
                    x, y = x0 + i, y0 + j
                    if 0 <= x < 10**6 and 0 <= y < 10**6 and (x, y) not in seen and (x, y) not in blocked:
                        if [x, y] == target: return True
                        bfs.append([x, y])
                        seen.add((x, y))
                if len(bfs) == 20000: return True
            return False
        return bfs(source, target) and bfs(target, source)
class Solution:
    def isEscapePossible(self, blocked: List[List[int]], source: List[int], target: List[int]) -> bool:
        def bfs(source, target):
            seen = set()
            queue = collections.deque([source])
            while queue:
                if len(seen) >= 19901: return True
                for _ in range(len(queue)):
                    pos = queue.popleft()
                    if pos == target: return True
                    seen.add(pos)
                    i, j = pos
                    for ni, nj in ((i+1, j), (i-1, j), (i, j+1), (i, j-1)):
                        if 0 <= ni < 10**6 and 0 <= nj < 10**6 and (ni, nj) not in blocked and (ni, nj) not in seen:
                            if (ni, nj) == target: return True
                            seen.add((ni, nj))
                            queue.append((ni, nj))
            return False

            
        blocked = set(map(tuple, blocked))
        source, target = tuple(source), tuple(target)
        return bfs(source, target) and bfs(target, source)
            

class Solution:
    def isEscapePossible(self, blocked: List[List[int]], source: List[int], target: List[int]) -> bool:
        def method1(blocked):
            R=C=10**6
            if not (0<=source[0]<R and 0<=source[1]<C):
                return False
            
            if not (0<=target[0]<R and 0<=target[1]<C):
                return False
            
            if not blocked:
                return True
            
            blocked=set(map(tuple,blocked))
            seen=set()
            
            def neighbors(r,c):
                for nr,nc in [(r-1,c),(r+1,c),(r,c-1),(r,c+1)]:
                    if 0<=nr<R and 0<=nc<C:
                        yield nr,nc
            
            def dfs(r,c):
                if (r,c) in seen:
                    return False
                
                if [r,c]==target:
                    return True
                
                if (r,c) in blocked:
                    return False
                
                seen.add((r,c))
                for nr,nc in neighbors(r,c):
                    if dfs(nr,nc):
                        return True
                return False
            
            return dfs(*source)
        
        #return method1(blocked)
    
        def method2(blocked):
            R=C=10**6
            if not (0<=source[0]<R and 0<=source[1]<C):
                return False
            
            if not (0<=target[0]<R and 0<=target[1]<C):
                return False
            
            if not blocked:
                return True
            
            blocked = set(map(tuple, blocked))
            
            def neighbors(r,c):
                for nr,nc in [(r-1,c),(r+1,c),(r,c-1),(r,c+1)]:
                    if 0<=nr<R and 0<=nc<C:
                        yield nr,nc
        
            def check(blocked, source, target):
                si, sj = source
                ti, tj = target
                level = 0
                q = collections.deque([(si,sj)])
                vis = set()
                while q:
                    for _ in range(len(q)):
                        i,j = q.popleft()
                        if i == ti and j == tj: return True
                        for x,y in neighbors(i,j):
                            if (x,y) not in vis and (x,y) not in blocked:
                                vis.add((x,y))
                                q.append((x,y))
                    level += 1
                    if level == len(blocked): 
                        return True
                    
                return False
        
            return check(blocked, source, target) and check(blocked, target, source)
        
        return method2(blocked)
            
                

class Solution:
    def isEscapePossible(self, blocked: List[List[int]], source: List[int], target: List[int]) -> bool:
        def method1(blocked):
            R=C=10**6
            if not (0<=source[0]<R and 0<=source[1]<C):
                return False
            
            if not (0<=target[0]<R and 0<=target[1]<C):
                return False
            
            if not blocked:
                return True
            
            blocked=set(map(tuple,blocked))
            seen=set()
            
            def neighbors(r,c):
                for nr,nc in [(r-1,c),(r+1,c),(r,c-1),(r,c+1)]:
                    if 0<=nr<R and 0<=nc<C:
                        yield nr,nc
            
            def dfs(r,c):
                if (r,c) in seen:
                    return False
                
                if [r,c]==target:
                    return True
                
                if (r,c) in blocked:
                    return False
                
                seen.add((r,c))
                for nr,nc in neighbors(r,c):
                    if dfs(nr,nc):
                        return True
                return False
            
            return dfs(*source)
        
        #return method1(blocked)
    
        def method2(blocked):
            R=C=10**6
            if not (0<=source[0]<R and 0<=source[1]<C):
                return False
            
            if not (0<=target[0]<R and 0<=target[1]<C):
                return False
            
            if not blocked:
                return True
            
            blocked = set(map(tuple, blocked))
            
            def neighbors(r,c):
                for nr,nc in [(r-1,c),(r+1,c),(r,c-1),(r,c+1)]:
                    if 0<=nr<R and 0<=nc<C:
                        yield nr,nc
        
            def check(source, target):
                sr, sc = source
                tr, tc = target
                level = 0
                q = collections.deque([(sr,sc)])
                vis = set()
                while q:
                    for _ in range(len(q)):
                        r,c = q.popleft()
                        if r == tr and c == tc: return True
                        for nr,nc in neighbors(r,c):
                            if (nr,nc) not in vis and (nr,nc) not in blocked:
                                vis.add((nr,nc))
                                q.append((nr,nc))
                    level += 1
                    if level == len(blocked): 
                        return True
                    
                return False
        
            return check(source, target) and check(target, source)
        
        return method2(blocked)
            
                

class Solution:
    def isEscapePossible(self, blocked: List[List[int]], source: List[int], target: List[int]) -> bool:
        def method1(blocked):
            R=C=10**6
            if not (0<=source[0]<R and 0<=source[1]<C):
                return False
            
            if not (0<=target[0]<R and 0<=target[1]<C):
                return False
            
            if not blocked:
                return True
            
            blocked=set(map(tuple,blocked))
            seen=set()
            
            def neighbors(r,c):
                for nr,nc in [(r-1,c),(r+1,c),(r,c-1),(r,c+1)]:
                    if 0<=nr<R and 0<=nc<C:
                        yield nr,nc
            
            def dfs(r,c):
                if (r,c) in seen:
                    return False
                
                if [r,c]==target:
                    return True
                
                if (r,c) in blocked:
                    return False
                
                seen.add((r,c))
                for nr,nc in neighbors(r,c):
                    if dfs(nr,nc):
                        return True
                return False
            
            return dfs(*source)
        
        #return method1(blocked)
    
        def method2(blocked):
            if not blocked: 
                return True
                
            blocked = set(map(tuple, blocked))
            
            def neighbors(r,c):
                for nr,nc in [(r-1,c),(r+1,c),(r,c-1),(r,c+1)]:
                    if 0<=nr<R and 0<=nc<C:
                        yield nr,nc
        
            def check(blocked, source, target):
                si, sj = source
                ti, tj = target
                level = 0
                q = collections.deque([(si,sj)])
                vis = set()
                while q:
                    for _ in range(len(q)):
                        i,j = q.popleft()
                        if i == ti and j == tj: return True
                        for x,y in ((i+1,j),(i-1,j),(i,j+1),(i,j-1)):
                            if 0<=x<10**6 and 0<=y<10**6 and (x,y) not in vis and (x,y) not in blocked:
                                vis.add((x,y))
                                q.append((x,y))
                    level += 1
                    if level == len(blocked): 
                        return True
                    
                return False
        
            return check(blocked, source, target) and check(blocked, target, source)
        
        return method2(blocked)
            
                

class Solution:
    def isEscapePossible(self, blocked: List[List[int]], source: List[int], target: List[int]) -> bool:
        

        #Column and row of source and target (used for escape condition)
        sc, sr = source
        tc, tr = target
        R, C = 10**6, 10**6

        def dist(r1,c1,r2,c2):
            '''Calculates Manhattan distance from (r1,c1) to (r2,c2)'''
            return abs(r2-r1)+abs(c2-c1)

        #Two priority queues: one starting from target and one from source
        #Two visited sets: one for nodes visited by path from target and the other from source
        q = [[(0,*source[::-1])], [(0,*target[::-1])]]
        v = [set(),set()]
        b = set((tuple(b[::-1]) for b in blocked))

        if (tuple(source) in b) or (tuple(target) in b):
            return False

        #if source and target can reach 200 distance from their origin
        #it is safe to say 200 blocked spaces cannot contain them
        source_escape = False
        target_escape = False

        while q[0] and q[1]:

            index = 0 if len(q[0]) <= len(q[1]) else 1

            d, r, c = heapq.heappop(q[index])

            for i, j in ((r+1, c), (r-1, c), (r, c+1), (r, c-1)):
                if (0 <= i < R) and (0 <= j < C) and ((i,j) not in b) and ((i,j) not in v[index]):

                    if (i,j) in v[1-index]:
                        return True

                    v[index].add((i,j))
                    r_target, c_target = q[1-index][0][1:]
                    heapq.heappush(q[index], (dist(i,j,r_target,c_target), i, j))

            if not source_escape and not index:
                source_escape = dist(r, c, sr, sc) > 200
            if not target_escape and index:
                target_escape = dist(r, c, tr, tc) > 200

            if source_escape and target_escape:
                return True

        return False



class Solution:
    def isEscapePossible(self, blocked, source, target):
        blocked = {tuple(p) for p in blocked}

        def bfs(source, target):
            queue, seen = [source], {tuple(source)}
            for x0, y0 in queue:
                for i, j in [[0, 1], [1, 0], [-1, 0], [0, -1]]:
                    x, y = x0 + i, y0 + j
                    if 0 <= x < 10**6 and 0 <= y < 10**6 and (x, y) not in seen and (x, y) not in blocked:
                        if [x, y] == target: return True
                        queue.append([x, y])
                        seen.add((x, y))
                if len(queue) == 20000: 
                    return True
            return False
        
        return bfs(source, target) and bfs(target, source)

    
    
    
class Solution1:
    def isEscapePossible(self, blocked: List[List[int]], source: List[int], target: List[int]) -> bool:
        
        if not blocked:
            return True
        
        rows = cols = 10**6
        directions = {(0,1), (0,-1), (1,0), (-1,0)}
        q1, q2 = [source], [target]
        seen = set()
        
        while q1:
            next_q = []
            # current level
            while q1:
                r, c = q1.pop()
                # print(r, c, q1)
                for dr, dc in directions:
                    if 0 <= r+dr < rows and 0 <= c+dc < cols:
                        if [r+dr, c+dc] in q2:
                            return True
                        if [r+dr, c+dc] not in blocked:
                            next_q.append([r+dr, c+dc])
                            seen.add((r+dr, c+dc))
            # print('hi', next_q)
            # update level queue
            q1 = next_q
            if len(q1) > len(q2):
                q1, q2 = q2, q1
                    
        return False
                        
                
                
                

class Solution:
    def isEscapePossible(self, blocked, source, target):
        blocked = {tuple(p) for p in blocked}

        def bfs(source, target):
            bfs, seen = [source], {tuple(source)}
            for x0, y0 in bfs:
                for i, j in [[0, 1], [1, 0], [-1, 0], [0, -1]]:
                    x, y = x0 + i, y0 + j
                    if 0 <= x < 10**6 and 0 <= y < 10**6 and (x, y) not in seen and (x, y) not in blocked:
                        if [x, y] == target: return True
                        bfs.append([x, y])
                        seen.add((x, y))
                if len(bfs) == 20000: return True
            return False
        return bfs(source, target) and bfs(target, source)
class Solution:
    def isEscapePossible(self, blocked: List[List[int]], source: List[int], target: List[int]) -> bool:
        
        blocked = {tuple(p) for p in blocked}

        def bfs(source, target):
            bfs, seen = [source], {tuple(source)}
            for x0, y0 in bfs:
                for i, j in [[0, 1], [1, 0], [-1, 0], [0, -1]]:
                    x, y = x0 + i, y0 + j
                    if 0 <= x < 10**6 and 0 <= y < 10**6 and (x, y) not in seen and (x, y) not in blocked:
                        if [x, y] == target: return True
                        bfs.append([x, y])
                        seen.add((x, y))
                if len(bfs) == 20000: return True
            return False
        return bfs(source, target) and bfs(target, source)

from collections import deque 

class Solution:
    def isEscapePossible(self, blocked, source, target):
        blocked = {tuple(p) for p in blocked}

        def bfs(source, target):
            bfs, seen = [source], {tuple(source)}
            for x0, y0 in bfs:
                for i, j in [[0, 1], [1, 0], [-1, 0], [0, -1]]:
                    x, y = x0 + i, y0 + j
                    if 0 <= x < 10**6 and 0 <= y < 10**6 and (x, y) not in seen and (x, y) not in blocked:
                        if [x, y] == target: return True
                        bfs.append([x, y])
                        seen.add((x, y))
                if len(bfs) == 20000: return True
            return False
        return bfs(source, target) and bfs(target, source)
class Solution:
    def isEscapePossible(self, blocked: List[List[int]], source: List[int], target: List[int]) -> bool:
        blocked = {tuple(p) for p in blocked}

        def bfs(source, target):
            bfs, seen = [source], {tuple(source)}
            for x0, y0 in bfs:
                for i, j in [[0, 1], [1, 0], [-1, 0], [0, -1]]:
                    x, y = x0 + i, y0 + j
                    if 0 <= x < 10**6 and 0 <= y < 10**6 and (x, y) not in seen and (x, y) not in blocked:
                        if [x, y] == target: return True
                        bfs.append([x, y])
                        seen.add((x, y))
                if len(bfs) == 20000: return True
            return False
        return bfs(source, target) and bfs(target, source)

class Solution:
    def isEscapePossible(self, blocked: List[List[int]], source: List[int], target: List[int]) -> bool:
        
        blocked = set([tuple(b) for b in blocked])
        dirs = [[-1,0],[1,0],[0,-1],[0,1]]   
        def valid(r, c):
            if r>=0 and r<1000000 and c>=0 and c<1000000:
                return True
            return False
        
        def bfs(source, target):
            q = [tuple(source)]
            vis = set([tuple(source)])
            while q:

                if len(q) > len(blocked):
                    return True            
                temp = []
                for r, c in q:
                    if (r, c) == tuple(target): #must do this cast
                        return True
                    for d in dirs:
                        nr = r+d[0]
                        nc = c+d[1]
                        if valid(nr, nc):
                            if (nr,nc) not in vis and (nr, nc) not in blocked:
                                temp.append((nr,nc))
                                vis.add((nr,nc))
                q = temp
                
            return False
            
        return bfs(source, target) and bfs(target, source)
            
                
                

import collections

class Solution:
    def isEscapePossible(self, blocked: List[List[int]], source: List[int], target: List[int]) -> bool:
        if not blocked: 
            return True
         
        blocked = set(map(tuple, blocked)) 
        print(len(blocked))
        
        return self.bfs(blocked, source, target) and self.bfs(blocked, target, source)
        
    def bfs(self, blocked, source, target):
        si, sj = source
        ti, tj = target
        
        queue = collections.deque([(si, sj)])
        visited = set()
        
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        step = 0
        while queue:
            size = len(queue)
            for _ in range(size):
                curX, curY = queue.popleft() 
                 
                if curX == ti and curY == tj:
                    return True
                
                for dx, dy in directions:
                    newX = curX + dx
                    newY = curY + dy
                    if self.isValid(newX, newY, blocked, visited):
                        queue.append((newX, newY))
                        visited.add((newX, newY))
            step += 1
            if step == len(blocked):
                break
                
        else:
            return False
        
        return True 
        
    def isValid(self, newX, newY, blocked, visited):
        return 0 <= newX < 1000000 and  0 <= newY < 1000000 and (newX, newY) not in blocked and (newX, newY) not in visited
class Solution:
        def isEscapePossible(self, blocked, source, target):
            blocked = {tuple(p) for p in blocked}

            def bfs(source, target):
                bfs, seen = [source], {tuple(source)}
                for x0, y0 in bfs:
                    for i, j in [[0, 1], [1, 0], [-1, 0], [0, -1]]:
                        x, y = x0 + i, y0 + j
                        if 0 <= x < 10**6 and 0 <= y < 10**6 and (x, y) not in seen and (x, y) not in blocked:
                            if [x, y] == target: return True
                            bfs.append([x, y])
                            seen.add((x, y))
                    if len(bfs) == 20000: return True
                return False
            return bfs(source, target) and bfs(target, source)

class Solution:
    def isEscapePossible(self, blocked: List[List[int]], source: List[int], target: List[int]) -> bool:
        blocked = set(map(tuple, blocked))
        src, target = tuple(source), tuple(target)
        
        return self.dfs(src, target, set(), blocked) and self.dfs(target, src, set(), blocked)
    
    def dfs(self, src, target, seen, blocked):
        
        if len(seen) > 20000 or src == target: return True 
        
        x0, y0 = src
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            x = x0 + dx
            y = y0 + dy
            if (0<=x<10**6 and 0<=y<10**6 and ((x, y) not in seen) and ((x, y) not in blocked)):
                seen.add((x, y))
                if self.dfs((x, y), target, seen, blocked):
                    return True 
                
        return False

class Solution:
    def isEscapePossible(self, blocked: List[List[int]], source: List[int], target: List[int]) -> bool:

        #Column and row of source and target (used for escape condition)
        sc, sr = source
        tc, tr = target
        R, C = 10**6, 10**6

        def dist(r1,c1,r2,c2):
            '''Calculates Manhattan distance from (r1,c1) to (r2,c2)'''
            return abs(r2-r1)+abs(c2-c1)

        #Two priority queues: one starting from target and one from source
        #Two visited sets: one for nodes visited by path from target and the other from source
        q = [[(0,*source[::-1])], [(0,*target[::-1])]]
        v = [set(),set()]
        b = set((tuple(b[::-1]) for b in blocked))

        if (tuple(source) in b) or (tuple(target) in b):
            return False

        #if source and target can reach 200 distance from their origin
        #it is safe to say 200 blocked spaces cannot contain them
        source_escape = False
        target_escape = False

        while q[0] and q[1]:

            index = 0 if len(q[0]) <= len(q[1]) else 1

            d, r, c = heapq.heappop(q[index])

            for i, j in ((r+1, c), (r-1, c), (r, c+1), (r, c-1)):
                if (0 <= i < R) and (0 <= j < C) and ((i,j) not in b) and ((i,j) not in v[index]):

                    if (i,j) in v[1-index]:
                        return True

                    v[index].add((i,j))
                    r_target, c_target = q[1-index][0][1:]
                    heapq.heappush(q[index], (dist(i,j,r_target,c_target), i, j))

            if not source_escape and not index:
                source_escape = dist(r, c, sr, sc) > 200
            if not target_escape and index:
                target_escape = dist(r, c, tr, tc) > 200

            if source_escape and target_escape:
                return True

        return False
class Solution:
    def isEscapePossible(self, blocked: List[List[int]], se: List[int], target: List[int]) -> bool:
        d={}
        d[(se[0],se[1])]=1
        e=[se]
        v=1
        bl={tuple(i) for i in blocked}
        while(e!=[]):
            f=[]
            for i in e:
                if(i==target):
                    return True
                x=i[0]
                y=i[1]
                if(x>0):
                    if((x-1,y) not in bl)and((x-1,y) not in d):
                        f.append([x-1,y])
                        d[(x-1,y)]=1
                if(x<10**6):
                    if((x+1,y) not in bl)and((x+1,y) not in d):
                        f.append([x+1,y])
                        d[(x+1,y)]=1
                if(y>0):
                    if((x,y-1) not in bl)and((x,y-1) not in d):
                        f.append([x,y-1])
                        d[(x,y-1)]=1
                if(y<10**6):
                    if((x,y+1) not in bl)and((x,y+1) not in d):
                        f.append([x,y+1])
                        d[(x,y+1)]=1
            e=f
            v+=len(f)
            if(v>=20000):
                break
        if(e==[]):
            return False
        
        d={}
        d[(target[0],target[1])]=1
        e=[target]
        v=0
        while(e!=[]):
            f=[]
            for i in e:
                if(i==se):
                    return True
                x=i[0]
                y=i[1]
                if(x>0):
                    if((x-1,y) not in bl)and((x-1,y) not in d):
                        f.append([x-1,y])
                        d[(x-1,y)]=1
                if(x<10**6):
                    if((x+1,y) not in bl)and((x+1,y) not in d):
                        f.append([x+1,y])
                        d[(x+1,y)]=1
                if(y>0):
                    if((x,y-1) not in bl)and((x,y-1) not in d):
                        f.append([x,y-1])
                        d[(x,y-1)]=1
                if(y<10**6):
                    if((x,y+1) not in bl)and((x,y+1) not in d):
                        f.append([x,y+1])
                        d[(x,y+1)]=1
            e=f
            v+=len(f)
            if(v>=20000):
                return True
        if(e==[]):
            return False
        
        
        

class Solution:
    def isEscapePossible(self, blocked: List[List[int]], source: List[int], target: List[int]) -> bool:
        
        blocks = { tuple(b) for b in blocked }
        
        def bfs(start, end):
            queue, used = [start], { tuple(start) }
            for x, y in queue:
                for dx, dy in [[-1, 0], [1, 0], [0, 1], [0, -1]]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < 10**6 and 0 <= ny < 10**6 and (nx, ny) not in blocks and (nx, ny) not in used:
                        if [nx, ny] == end:
                            return True
                        queue.append([nx, ny])
                        used.add((nx, ny))
                if len(queue) == 20000:
                    return True
                    
            return False
        
        return bfs(source, target) and bfs(target, source)

class Solution:
    def isEscapePossible(self, blocked: List[List[int]], source: List[int], target: List[int]) -> bool:
        if not blocked: return True
        blocked = set(map(tuple, blocked))
        
        def check(blocked, source, target):
            si, sj = source
            ti, tj = target
            level = 0
            q = collections.deque([(si,sj)])
            vis = set()
            while q:
                for _ in range(len(q)):
                    i,j = q.popleft()
                    if i == ti and j == tj: return True
                    for x,y in ((i+1,j),(i-1,j),(i,j+1),(i,j-1)):
                        if 0<=x<10**6 and 0<=y<10**6 and (x,y) not in vis and (x,y) not in blocked:
                            vis.add((x,y))
                            q.append((x,y))
                level += 1
                if level == len(blocked): break
            else:
                return False
            return True
        
        return check(blocked, source, target) and check(blocked, target, source)
class Solution:
    def isEscapePossible(self, blocked: List[List[int]], source: List[int], target: List[int]) -> bool:
#         blocked_set = set([tuple(l) for l in blocked])
#         visited = set(tuple(source))
#         visited_block_set = set()
#         step = 0
        
#         directions = [(0,1),(1,0),(0,-1),(-1,0)]
#         cur_queue = [tuple(source)]
#         next_queue = []
#         while True:
    
#             for c in cur_queue:
#                 for d in directions:
#                     if c[0]+d[0] >= 0 and c[1]+d[1] >= 0 and (c[0]+d[0], c[1]+d[1]) not in visited and (c[0]+d[0], c[1]+d[1]) not in blocked_set:
#                         visited.add((c[0]+d[0], c[1]+d[1]))
#                         next_queue.append((c[0]+d[0], c[1]+d[1]))
#                         if (c[0]+d[0], c[1]+d[1]) == tuple(target):
#                             return True
#                     elif (c[0]+d[0], c[1]+d[1]) in blocked_set:
#                         visited_block_set.add((c[0]+d[0], c[1]+d[1]))
                        
#             step += 1
#             cur_queue = next_queue
#             next_queue = []
            
#             if not cur_queue or step > 200:
#                 break
        
#         if step == 201:
#             return True
#         return False

        blocked = {tuple(p) for p in blocked}

        def bfs(source, target):
            bfs, seen = [source], {tuple(source)}
            for x0, y0 in bfs:
                for i, j in [[0, 1], [1, 0], [-1, 0], [0, -1]]:
                    x, y = x0 + i, y0 + j
                    if 0 <= x < 10**6 and 0 <= y < 10**6 and (x, y) not in seen and (x, y) not in blocked:
                        if [x, y] == target: return True
                        bfs.append([x, y])
                        seen.add((x, y))
                if len(bfs) == 20000: return True
            return False
        return bfs(source, target) and bfs(target, source)
import collections

class Solution:
    def isEscapePossible(self, blocked: List[List[int]], source: List[int], target: List[int]) -> bool:
        if not blocked: 
            return True
         
        blocked = set(map(tuple, blocked)) 
        # for b in blocked:
        #     block.add((b[0], b[1]))
        
        return self.bfs(blocked, source, target) and self.bfs(blocked, target, source)
        
    def bfs(self, blocked, source, target):
        si, sj = source
        ti, tj = target
        
        queue = collections.deque([(si, sj)])
        visited = set()
        
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        step = 0
        while queue:
            size = len(queue)
            for _ in range(size):
                curX, curY = queue.popleft() 
                 
                if curX == ti and curY == tj:
                    return True
                
                for dx, dy in directions:
                    newX = curX + dx
                    newY = curY + dy
                    if self.isValid(newX, newY, blocked, visited):
                        queue.append((newX, newY))
                        visited.add((newX, newY))
            step += 1
            if step == len(blocked):
                break
                
        else:
            return False
        
        return True 
        
    def isValid(self, newX, newY, blocked, visited):
        return 0 <= newX < 1000000 and  0 <= newY < 1000000 and (newX, newY) not in blocked and (newX, newY) not in visited
    
    
    
    
    
    
#     def isEscapePossible(self, blocked, source, target):
#         blocked = {tuple(p) for p in blocked}

#         def bfs(source, target):
#             bfs, seen = [source], {tuple(source)}
#             for x0, y0 in bfs:
#                 for i, j in [[0, 1], [1, 0], [-1, 0], [0, -1]]:
#                     x, y = x0 + i, y0 + j
#                     if 0 <= x < 10**6 and 0 <= y < 10**6 and (x, y) not in seen and (x, y) not in blocked:
#                         if [x, y] == target: return True
#                         bfs.append([x, y])
#                         seen.add((x, y))
#                 if len(bfs) == 20000: return True
#             return False
#         return bfs(source, target) and bfs(target, source)


class Solution:
    def isEscapePossible(self, blocked: List[List[int]], source: List[int], target: List[int]) -> bool:

        #Column and row of source and target (used for escape condition)
        sc, sr = source
        tc, tr = target
        R, C = 10**6, 10**6

        def dist(r1,c1,r2,c2):
            '''Calculates Manhattan distance from (r1,c1) to (r2,c2)'''
            return abs(r2-r1)+abs(c2-c1)

        #Two priority queues: one starting from target and one from source
        #Two visited sets: one for nodes visited by path from target and the other from source
        q = [[(0,*source[::-1])], [(0,*target[::-1])]]
        v = [set(),set()]
        b = set((tuple(b[::-1]) for b in blocked))

        if (tuple(source) in b) or (tuple(target) in b):
            return False

        #if source and target can reach 200 distance from their origin
        #it is safe to say 200 blocked spaces cannot contain them
        source_escape = False
        target_escape = False

        while q[0] and q[1]:

            index = 0 if len(q[0]) <= len(q[1]) else 1

            d, r, c = heapq.heappop(q[index])

            for i, j in ((r+1, c), (r-1, c), (r, c+1), (r, c-1)):
                if (0 <= i < R) and (0 <= j < C) and ((i,j) not in b) and ((i,j) not in v[index]):

                    if (i,j) in v[1-index]:
                        return True

                    v[index].add((i,j))
                    r_target, c_target = q[1-index][0][1:]
                    heapq.heappush(q[index], (dist(i,j,r_target,c_target), i, j))

            if not source_escape and not index:
                source_escape = dist(r, c, sr, sc) > 200
            if not target_escape and index:
                target_escape = dist(r, c, tr, tc) > 200

            if source_escape and target_escape:
                return True

        return False

class Solution:
    def isEscapePossible(self, blocked: List[List[int]], source: List[int], target: List[int]) -> bool:

###BFS
        blocked = {tuple(p) for p in blocked}

        def bfs(source, target):
            bfs, seen = [source], {tuple(source)}
            for x0, y0 in bfs:
                for i, j in [[0, 1], [1, 0], [-1, 0], [0, -1]]:
                    x, y = x0 + i, y0 + j
                    if [x, y] == target: return True
                    if 0 <= x < 10**6 and 0 <= y < 10**6 and (x, y) not in seen and (x, y) not in blocked:    
                        bfs.append([x, y])
                        seen.add((x, y))
                if len(bfs) == 20000: return True
            return False
        return bfs(source, target) and bfs(target, source)
        
    
    
###DFS
#         blocked = set(map(tuple, blocked))

#         def dfs(x, y, target, seen):
#             if not (0 <= x < 10**6 and 0 <= y < 10**6) or (x, y) in blocked or (x, y) in seen: return False
#             seen.add((x, y))
#             if len(seen) > 20000 or [x, y] == target: return True
#             return dfs(x + 1, y, target, seen) or 
#                 dfs(x - 1, y, target, seen) or 
#                 dfs(x, y + 1, target, seen) or 
#                 dfs(x, y - 1, target, seen)
#         return dfs(source[0], source[1], target, set()) and dfs(target[0], target[1], source, set())

class Solution:
    def isEscapePossible(self, blocked: List[List[int]], source: List[int], target: List[int]) -> bool:
        blocked = {tuple(p) for p in blocked}
        src_q = []
        src_visited = set()
        tgt_q = []
        tgt_visited = set()
        src_q.append((source[0], source[1]))
        tgt_q.append((target[0], target[1]))
        while src_q and tgt_q:
            src_node_x, src_node_y = src_q.pop(0)
            tgt_node_x, tgt_node_y = tgt_q.pop(0)
            if (src_node_x, src_node_y) in tgt_visited:
                return True
            if (tgt_node_x, tgt_node_y) in src_visited:
                return True
            if len(tgt_visited) > 20000 or len(src_visited) > 20000:
                return True
            if (src_node_x, src_node_y) not in src_visited:
                src_visited.add((src_node_x, src_node_y))
                src_neighboring_nodes = [(src_node_x, src_node_y-1), (src_node_x, src_node_y+1), (src_node_x-1, src_node_y), (src_node_x+1, src_node_y)]
                for each_node in src_neighboring_nodes:
                    if 0<=each_node[0]<10**6 and 0<=each_node[1]<10**6 and each_node not in blocked and each_node not in src_visited:
                        src_q.append((each_node[0], each_node[1]))
            if (tgt_node_x, tgt_node_y) not in tgt_visited:
                tgt_visited.add((tgt_node_x, tgt_node_y))
                tgt_neighboring_nodes = [(tgt_node_x, tgt_node_y-1), (tgt_node_x, tgt_node_y+1), (tgt_node_x-1, tgt_node_y), (tgt_node_x+1, tgt_node_y)]
                for each_node in tgt_neighboring_nodes:
                    if 0<=each_node[0]<10**6 and 0<=each_node[1]<10**6 and each_node not in blocked and each_node not in tgt_visited:
                        tgt_q.append((each_node[0], each_node[1]))
        return False
                        
                
                
        

class Solution:
    def isEscapePossible(self, blocked: List[List[int]], source: List[int], target: List[int]) -> bool:
        
        blocked = {tuple(p) for p in blocked}

        def bfs(source, target):
            bfs, seen = [source], {tuple(source)}
            for x0, y0 in bfs:
                for i, j in [[0, 1], [1, 0], [-1, 0], [0, -1]]:
                    x, y = x0 + i, y0 + j
                    if 0 <= x < 10**6 and 0 <= y < 10**6 and (x, y) not in seen and (x, y) not in blocked:
                        if [x, y] == target: return True
                        bfs.append([x, y])
                        seen.add((x, y))
                if len(bfs) == 20000: return True
            return False
        
        return bfs(source, target) and bfs(target, source)
class Solution:
    def isEscapePossible(self, blocked: List[List[int]], source: List[int], target: List[int]) -> bool:
        m = 10 ** 6
        n = len(blocked)
        if target in blocked or source in blocked: return False
        if n <= 1: return True
        dxy = [[0, -1], [0, 1], [-1, 0], [1, 0]]
        blocked = set(map(tuple, blocked))
        threshold = 200 * 200
        def bfs(pos, target):
            q = collections.deque([pos])
            visited = {tuple(pos)}
            cnt = 0
            while q:
                x, y = q.popleft()
                if x == target[0] and y == target[1]:
                    return 1
                cnt += 1
                if cnt > threshold:
                    return 2
                for dx, dy in dxy:
                    x_, y_ = x + dx, y + dy
                    if 0 <= x_ < m and 0 <= y_ < m:
                        p = (x_, y_)
                        if p not in visited and p not in blocked:
                            q.append(p)
                            visited.add(p)
            return -1
        
        i = bfs(source, target)
        # print(i)
        if i == 1:
            return True
        if i == -1:
            return False
        j = bfs(target, source)
        # print(j)
        return j == 2
                     
        

class Solution:
    def isEscapePossible(self, blocked: List[List[int]], source: List[int], target: List[int], maxNum = 10e6, bound = 200) -> bool:
        
        maxArea = bound*bound//2 + 1
        directions = [0, 1, 0, -1, 0]
        
        def dfs(x, y, t): 
            if x<0 or x>maxNum or y<0 or y>maxNum or tuple([x, y]) in block: 
                return False 
            seen.add(tuple([x, y]))
            block.add(tuple([x, y]))
            if len(seen)>maxArea or (x == t[0] and y == t[1]): 
                return True 
            for d in range(4): 
                if dfs(x+directions[d], y+directions[d+1], t): 
                    return True
            return False        
        seen = set()
        block = set(map(tuple, blocked))
        if not dfs(source[0], source[1], target): 
            return False 
        seen = set()
        block = set(map(tuple, blocked))
        return dfs(target[0], target[1], source)
class Solution:
    def isEscapePossible(self, blocked: List[List[int]], source: List[int], target: List[int]) -> bool:

###BFS
        blocked = {tuple(p) for p in blocked}

        def bfs(source, target):
            bfs, seen = [source], {tuple(source)}
            for x0, y0 in bfs:
                for i, j in [[0, 1], [1, 0], [-1, 0], [0, -1]]:
                    x, y = x0 + i, y0 + j
                    if [x, y] == target: return True
                    if len(bfs) == 20000: return True
                    if 0 <= x < 10**6 and 0 <= y < 10**6 and (x, y) not in seen and (x, y) not in blocked:    
                        bfs.append([x, y])
                        seen.add((x, y))
                
            return False
        return bfs(source, target) and bfs(target, source)
        
    
    
###DFS
#         blocked = set(map(tuple, blocked))

#         def dfs(x, y, target, seen):
#             if not (0 <= x < 10**6 and 0 <= y < 10**6) or (x, y) in blocked or (x, y) in seen: return False
#             seen.add((x, y))
#             if len(seen) > 20000 or [x, y] == target: return True
#             return dfs(x + 1, y, target, seen) or 
#                 dfs(x - 1, y, target, seen) or 
#                 dfs(x, y + 1, target, seen) or 
#                 dfs(x, y - 1, target, seen)
#         return dfs(source[0], source[1], target, set()) and dfs(target[0], target[1], source, set())

class Solution:
    def isEscapePossible(self, blocked: List[List[int]], source: List[int], target: List[int]) -> bool:
        def bfs(start, is_source=True):
            queue = [start]
            pending = set({start})
            visited = set()
            while len(queue) > 0:
                temp = queue.pop(0)
                x1, y1 = temp
                if temp == tuple(target) and is_source:
                    return True
                if temp == tuple(source) and not is_source:
                    return True
                visited.add(temp)
                if len(visited) > len(blocked)**2//2:
                    return True
                pending.remove(temp)
                for x, y in [(x1+1, y1), (x1-1, y1), (x1, y1+1), (x1, y1-1)]:
                    if 0<=x<10**6 and 0<=y<10**6:
                        if (x, y) not in pending and (x, y) not in blocked and (x, y) not in visited:
                            queue.append((x, y))
                            pending.add((x, y))
            return False
        blocked = list(map(tuple, blocked))
        return bfs(tuple(source), True) and bfs(tuple(target), False)
class Solution:
    def isEscapePossible(self, blocked: List[List[int]], source: List[int], target: List[int]) -> bool:
        
        blocked = set([tuple(b) for b in blocked])
        dirs = [[-1,0],[1,0],[0,-1],[0,1]]   
        def valid(r, c):
            if r>=0 and r<1000000 and c>=0 and c<1000000:
                return True
            return False
        
        def bfs(source, target):
            q = [tuple(source)]
            vis = set([tuple(source)])
            while q:

                if len(q) >= len(blocked):
                    return True            
                temp = []
                for r, c in q:
                    if (r, c) == tuple(target): #must do this cast
                        return True
                    for d in dirs:
                        nr = r+d[0]
                        nc = c+d[1]
                        if valid(nr, nc):
                            if (nr,nc) not in vis and (nr, nc) not in blocked:
                                temp.append((nr,nc))
                                vis.add((nr,nc))
                q = temp
                
            return False
            
        return bfs(source, target) and bfs(target, source)
            
                
                

class Solution:
    def isEscapePossible(self, b: List[List[int]], s: List[int], t: List[int]) -> bool:
        def find(source,  target, blocked):
            queue = [source]
            visited = set()

            while queue:
                x, y = queue.pop()

                if len(visited) > 20000:
                    return True

                if x == target[0] and y == target[1]:
                    return True

                visited.add((x, y))

                next_options = []
                for x_delta, y_delta in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                    new_x, new_y = x + x_delta, y + y_delta

                    if new_x < 0 or new_x >= 10 ** 6 or new_y < 0 or new_y >= 10 ** 6:
                        continue

                    if (new_x, new_y) in visited or (new_x, new_y) in blocked:
                        continue

                    next_options.append((new_x, new_y))

                next_options.sort(key=lambda point: (point[0] - target[0]) ** 2 + (point[1] - target[1]) ** 2, reverse=True)
                queue += next_options
        
            return False
    
        bl = set()
        for bi in b:
            bl.add(tuple(bi))
            
        return find(s, t, bl) and find(t, s, bl)
class Solution:
    def isEscapePossible(self, blocked: List[List[int]], source: List[int], target: List[int]) -> bool:
        maxlen = 10**6
        #maxarea = (4/3.14) * 17000 # maxarea = (4/3.14) * 10000 does not work!
        maxarea = 40000
        dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        
        blocked = set(map(tuple, blocked)) # use a set to make it faster for retrieving
        
        def bfs(source, target):
            q = collections.deque()
            aset = set()
            q.append(source)
            while q and len(aset) < maxarea:
                row, col = q.popleft()
                if row == target[0] and col == target[1]:
                    return True
                aset.add((row,col))
                
                for dir in dirs:
                    row2 = row + dir[0]
                    col2 = col + dir[1]
                    if 0<=row2<maxlen and 0<=col2 < maxlen and not (row2, col2) in aset and not (row2,col2) in blocked:
                        q.append([row2, col2])
                        aset.add((row2, col2))
            return len(aset) >= maxarea # evaluate by maxarea
        return bfs(source, target) and bfs(target, source)
class Solution:
    def isEscapePossible(self, blocked: List[List[int]], source: List[int], target: List[int]) -> bool:
        maxlen = 10**6
        #maxarea = (4/3.14) * 17000 # maxarea = (4/3.14) * 10000 does not work!
        maxarea = 40000
        dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        
        blocked = set(map(tuple, blocked)) # use a set to make it faster for retrieving
        
        def bfs(source, target):
            q = collections.deque()
            visited = set()
            q.append(source)
            while q and len(visited) < maxarea:
                row, col = q.popleft()
                if row == target[0] and col == target[1]:
                    return True
                visited.add((row,col))
                
                for dir in dirs:
                    row2 = row + dir[0]
                    col2 = col + dir[1]
                    if 0<=row2<maxlen and 0<=col2 < maxlen and not (row2, col2) in visited and not (row2,col2) in blocked:
                        q.append([row2, col2])
                        visited.add((row2, col2))
            return len(visited) >= maxarea # evaluate by maxarea
        return bfs(source, target) and bfs(target, source)
class Solution:
    def isEscapePossible(self, blocked: List[List[int]], source: List[int], target: List[int]) -> bool:
        directions = [[-1, 0], [0, -1], [1, 0], [0, 1]]
        block_set = {tuple(t) for t in blocked}
        source, target = tuple(source), tuple(target)
        if source in block_set or target in block_set:
            return False
        
        def findPath(source, target, block_set):
            from collections import deque
            queue = deque()
            covered, covered_block = set(), set()
            queue.append(source)
            covered.add(source)
            while len(queue) > 0:
                count = len(queue)
                while count > 0:
                    head = queue.popleft()
                    count -= 1
                    for dir1 in directions:
                        x, y = head[0] + dir1[0], head[1] + dir1[1]
                        if x < 0 or x == int(1e6) or y < 0 or y == int(1e6):
                            continue
                        pos = (x, y)
                        if pos == target:
                            return True                    
                        elif pos in covered:
                            continue
                        elif pos in block_set:
                            covered.add(pos)
                            covered_block.add(pos)
                        else:
                            queue.append(pos)
                            covered.add(pos)
                
                if len(queue) + len(covered_block) > len(block_set):
                    return True
            return False
        
        if not findPath(source, target, block_set):
            return False
        
        return findPath(target, source, block_set)
            



class Solution:
    def isEscapePossible(self, blocked: List[List[int]], se: List[int], target: List[int]) -> bool:
        d={}
        d[(se[0],se[1])]=1
        e=[se]
        v=1
        bl={tuple(i) for i in blocked}
        g=0
        while(e!=[]):
            f=[]
            for i in e:
                if(i==target):
                    return True
                x=i[0]
                y=i[1]
                if(abs(x-se[0])+abs(y-se[1])>200):
                    g=2
                    break
                if(x>0):
                    if((x-1,y) not in bl)and((x-1,y) not in d):
                        f.append([x-1,y])
                        d[(x-1,y)]=1
                if(x<10**6):
                    if((x+1,y) not in bl)and((x+1,y) not in d):
                        f.append([x+1,y])
                        d[(x+1,y)]=1
                if(y>0):
                    if((x,y-1) not in bl)and((x,y-1) not in d):
                        f.append([x,y-1])
                        d[(x,y-1)]=1
                if(y<10**6):
                    if((x,y+1) not in bl)and((x,y+1) not in d):
                        f.append([x,y+1])
                        d[(x,y+1)]=1
            if(g==2):
                break
            e=f
        if(e==[]):
            return False
        
        d={}
        d[(target[0],target[1])]=1
        e=[target]
        v=0
        while(e!=[]):
            f=[]
            for i in e:
                if(i==se):
                    return True
                x=i[0]
                y=i[1]
                if(abs(x-se[0])+abs(y-se[1])>200):
                    return True
                if(x>0):
                    if((x-1,y) not in bl)and((x-1,y) not in d):
                        f.append([x-1,y])
                        d[(x-1,y)]=1
                if(x<10**6):
                    if((x+1,y) not in bl)and((x+1,y) not in d):
                        f.append([x+1,y])
                        d[(x+1,y)]=1
                if(y>0):
                    if((x,y-1) not in bl)and((x,y-1) not in d):
                        f.append([x,y-1])
                        d[(x,y-1)]=1
                if(y<10**6):
                    if((x,y+1) not in bl)and((x,y+1) not in d):
                        f.append([x,y+1])
                        d[(x,y+1)]=1
            e=f
            
        if(e==[]):
            return False
        
        
        

class Solution:
    def isEscapePossible(self, blocked: List[List[int]], source: List[int], target: List[int]) -> bool:
        if not blocked: return True
        blocked = set(map(tuple, blocked))
        
        def check(blocked, source, target):
            si, sj = source
            ti, tj = target
            level = 0
            q = collections.deque([(si,sj)])
            vis = set()
            while q:
                for _ in range(len(q)):
                    i,j = q.popleft()
                    if i == ti and j == tj: return True
                    for x,y in ((i+1,j),(i-1,j),(i,j+1),(i,j-1)):
                        if 0<=x<10**6 and 0<=y<10**6 and (x,y) not in vis and (x,y) not in blocked:
                            vis.add((x,y))
                            q.append((x,y))
                level += 1
                if level == 2 * len(blocked): break
            else:
                return False
            return True
        
        return check(blocked, source, target) and check(blocked, target, source)
import heapq

def solve(b,s,t):
    def create_priority_item(c, t):
        dx = c[0]-t[0]
        dy = c[1]-t[1]
        d2 = dx*dx + dy*dy
        return (d2, c)

    b = set(tuple(_b) for _b in b)
    s = tuple(s)
    t = tuple(t)
    heap = [(-1,s)]
    visited = set()
    iter = -1
    while heap:
        iter += 1
        if iter > 1.1e5:
            return False
        _, c = heapq.heappop(heap)
        if c in visited:
            continue
        if c[0] < 0 or c[0] >=1e6 or c[1]<0 or c[1]>=1e6:
            # outside!
            continue
        if c in b:
            # blocked!
            continue
        if c == t:
            # found!
            return True
        # search neighbors:

        visited.add(c)
        x = c[0]
        while t[0] > x and (x+1,c[1]) not in b:
            x += 1
        heapq.heappush(heap, create_priority_item((x, c[1]  ), t))
        x = c[0]
        while t[0] < x and (x-1,c[1]) not in b:
            x -= 1
        heapq.heappush(heap, create_priority_item((x, c[1]  ), t))

        y = c[1]
        while t[1] > y and (c[0],y) not in b:
            y += 1
        heapq.heappush(heap, create_priority_item((c[0], y  ), t))
        y = c[1]
        while t[1] < y and (c[0],y) not in b:
            y -= 1
        heapq.heappush(heap, create_priority_item((c[0], y  ), t))


        heapq.heappush(heap, create_priority_item((c[0]+1, c[1]  ), t))
        heapq.heappush(heap, create_priority_item((c[0]-1, c[1]  ), t))
        heapq.heappush(heap, create_priority_item((c[0]  , c[1]+1), t))
        heapq.heappush(heap, create_priority_item((c[0]  , c[1]-1), t))
    # we live in a cavity :(
    return False


class Solution:
    def isEscapePossible(self, blocked: List[List[int]], source: List[int], target: List[int]) -> bool:
        return solve(blocked, source, target)

class Solution:
    def isEscapePossible(self, blocked: List[List[int]], source: List[int], target: List[int]) -> bool:
        def bfs(s,t,b):
            q,v=[s],{s}
            while len(q)>0:
                i,j=q.pop(0)
                if (i,j)==t:return True
                if i in (s[0]+210,s[0]-210) or j in (s[1]+210,s[1]-210):return True
                for d,e in ((i+1,j),(i,j+1),(i-1,j),(i,j-1)):
                    if d>=0 and d<10**6 and e>=0 and e<10**6 and (d,e) not in v and (d,e) not in b:
                        v.add((d,e))
                        q.append((d,e))
            return False
        
        b=set(tuple(i) for i in blocked)
        return bfs(tuple(source),tuple(target),b) and bfs(tuple(target),tuple(source),b)
        
        # blocked_map = collections.defaultdict(set)
        # for r,c in blocked:
        #     blocked_map[r].add(c)
        # queue_s = collections.deque()
        # queue_s.append(source)
        # visited_s = set()
        # visited_s.add(tuple(source))
        # queue_t = collections.deque()
        # queue_t.append(target)
        # visited_t = set()
        # visited_t.add(tuple(target))
        # while queue_s and queue_t:
        #     curr = queue_s.popleft()
        #     if curr==target or tuple(curr) in visited_t:
        #         return True
        #     for dr,dc in [(0,1),(1,0),(-1,0),(0,-1)]:
        #         nei_r = curr[0]+dr
        #         nei_c = curr[1]+dc
        #         if nei_r>=0 and nei_r<10**6 and nei_c>=0 and nei_c<10**6:
        #             if nei_r not in blocked_map or (nei_r in blocked_map and nei_c not in blocked_map[nei_r]):
        #                 if tuple([nei_r,nei_c]) not in visited_s:
        #                     visited_s.add(tuple([nei_r,nei_c]))
        #                     queue_s.append([nei_r,nei_c])
        #     curr = queue_t.popleft()
        #     if curr == source or tuple(curr) in visited_s:
        #         return True
        #     for dr,dc in [(0,1),(1,0),(-1,0),(0,-1)]:
        #         nei_r = curr[0]+dr
        #         nei_c = curr[1]+dc
        #         if nei_r>=0 and nei_r<10**6 and nei_c>=0 and nei_c<10**6:
        #             if nei_r not in blocked_map or (nei_r in blocked_map and nei_c not in blocked_map[nei_r]):
        #                 if tuple([nei_r,nei_c]) not in visited_t:
        #                     visited_t.add(tuple([nei_r,nei_c]))
        #                     queue_t.append([nei_r,nei_c])
        # return False

class Solution:
    def isEscapePossible(self, blocked: List[List[int]], source: List[int], target: List[int]) -> bool:
        DELTAS = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        CANNOT_MEET = 0
        CAN_WALK_200_STEPS = 1
        CAN_MEET = 2
        blocked_set = set(tuple(element) for element in blocked)
        
        def is_valid(i, j, seen):
            return 0 <= i < 10 ** 6 and 0 <= j < 10 ** 6 and (i, j) not in blocked_set and (i, j) not in seen
        
        def can_meet(source, target):
            i, j = source
            ti, tj = target
            q = deque([(i, j, 0)])
            seen = set()
            seen.add((i, j))
            while q:
                i, j, step = q.popleft()
                if (i == ti and j == tj):
                    return CAN_MEET
                if step == 200:
                    return CAN_WALK_200_STEPS
                for di, dj in DELTAS:
                    i1, j1 = i + di, j + dj
                    if is_valid(i1, j1, seen):
                        seen.add((i1, j1))
                        q.append((i1, j1, step + 1))
            return CANNOT_MEET
        
        result1 = can_meet(source, target)
        if result1 == CANNOT_MEET:
            return False
        if result1 == CAN_MEET:
            return True
        
        # result1 == CAN_WALK_200_STEPS
        result2 = can_meet(target, source)
        if result2 == CAN_WALK_200_STEPS:
            return True
        return False

class Solution:
    def isEscapePossible(self, blocked: List[List[int]], source: List[int], target: List[int]) -> bool:
        DELTAS = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        blocked_set = set(tuple(element) for element in blocked)
        
        def is_valid(i, j, seen):
            return 0 <= i < 10 ** 6 and 0 <= j < 10 ** 6 and (i, j) not in blocked_set and (i, j) not in seen
        
        def can_meet(source, target):
            i, j = source
            ti, tj = target
            q = deque([(i, j, 0)])
            seen = set()
            seen.add((i, j))
            while q:
                i, j, step = q.popleft()
                if (i == ti and j == tj) or step == 200:
                    return True
                for di, dj in DELTAS:
                    i1, j1 = i + di, j + dj
                    if is_valid(i1, j1, seen):
                        seen.add((i1, j1))
                        q.append((i1, j1, step + 1))
            return False
        
        return can_meet(source, target) and can_meet(target, source)

def bfs(s,t,b):
    q,v=[s],{s}
    while len(q)>0:
        i,j=q.pop(0)
        if (i,j)==t:return True
        if i in (s[0]+210,s[0]-210) or j in (s[1]+210,s[1]-210):return True
        for d,e in ((i+1,j),(i,j+1),(i-1,j),(i,j-1)):
            if d>=0 and d<10**6 and e>=0 and e<10**6 and (d,e) not in v and (d,e) not in b:
                v.add((d,e))
                q.append((d,e))
    return False
class Solution:
    def isEscapePossible(self, b: List[List[int]], s: List[int], t: List[int]) -> bool:
        b=set(tuple(i) for i in b)
        return bfs(tuple(s),tuple(t),b) and bfs(tuple(t),tuple(s),b)
class Solution:
    def isEscapePossible(self, blocked: List[List[int]], source: List[int], target: List[int]) -> bool:
        if not blocked: return True
        blocked = set(map(tuple, blocked))
        
        def check(blocked, source, target):
            si, sj = source
            ti, tj = target
            level = 0
            q = collections.deque([(si,sj)])
            vis = set()
            while q:
                for _ in range(len(q)):
                    i,j = q.popleft()
                    if i == ti and j == tj: return True
                    for x,y in ((i+1,j),(i-1,j),(i,j+1),(i,j-1)):
                        if 0<=x<10**6 and 0<=y<10**6 and (x,y) not in vis and (x,y) not in blocked:
                            vis.add((x,y))
                            q.append((x,y))
                level += 1
                if level == 2*len(blocked): 
                    break
            if len(q)==0:
                return False
            return True
        
        return check(blocked, source, target) and check(blocked, target, source)
import heapq


def solve(b,s,t):
    def create_priority_item(c, t):
        dx = c[0]-t[0]
        dy = c[1]-t[1]
        d2 = dx*dx + dy*dy
        return (d2, c)

    b = set(tuple(_b) for _b in b)
    s = tuple(s)
    t = tuple(t)
    heap = [(-1,s)]
    visited = set()
    iter = -1
    while heap:
        iter += 1
        if iter > 1.1e5:
            return False
        _, c = heapq.heappop(heap)
        if c in visited:
            continue
        if c[0] < 0 or c[0] >=1e6 or c[1]<0 or c[1]>=1e6:
            # outside!
            continue
        if c in b:
            # blocked!
            continue
        if c == t:
            # found!
            return True
        # search neighbors:

        visited.add(c)
        x = c[0]
        while t[0] > x and (x+1,c[1]) not in b:
            x += 1
        heapq.heappush(heap, create_priority_item((x, c[1]  ), t))
        x = c[0]
        while t[0] < x and (x-1,c[1]) not in b:
            x -= 1
        heapq.heappush(heap, create_priority_item((x, c[1]  ), t))

        y = c[1]
        while t[1] > y and (c[0],y) not in b:
            y += 1
        heapq.heappush(heap, create_priority_item((c[0], y  ), t))
        y = c[1]
        while t[1] < y and (c[0],y) not in b:
            y -= 1
        heapq.heappush(heap, create_priority_item((c[0], y  ), t))


        heapq.heappush(heap, create_priority_item((c[0]+1, c[1]  ), t))
        heapq.heappush(heap, create_priority_item((c[0]-1, c[1]  ), t))
        heapq.heappush(heap, create_priority_item((c[0]  , c[1]+1), t))
        heapq.heappush(heap, create_priority_item((c[0]  , c[1]-1), t))
    # we live in a cavity :(
    return False






class Solution:
    def isEscapePossible(self, blocked: List[List[int]], source: List[int], target: List[int]) -> bool:
        return solve(blocked, source, target)



def get_children(point, target):
    children = []
    y, x = point
    size_y, size_x = target
    
    if y-1 >= 0:
        children.append((y-1, x))
    
    if x-1 >= 0:
        children.append((y,x-1))
        
    if y+1 <= size_y:
        children.append((y+1,x))
        
    if x+1 <= size_x:
        children.append((y,x+1))
        
    return children


def bfs(source, target, is_blocked):
    queue = []
    marked = set()

    queue.append((source,1))
    marked.add(source)

    while queue:

        node_id, depth = queue.pop(0)
        
        if depth > 200:
            return True

        for child_id in get_children(node_id, (1000000, 1000000)):

            if child_id in is_blocked:
                continue

            if child_id == target:
                return True

            if child_id not in marked:
                queue.append((child_id,depth+1))
                marked.add(child_id)

    return False
    


class Solution:
    def isEscapePossible(self, blocked: List[List[int]], source: List[int], target: List[int]) -> bool:
        is_blocked = set()
        for item in blocked:
            is_blocked.add(tuple(item))
            
        target = tuple(target)
        source = tuple(source)
        
        if bfs(source, target, is_blocked) and bfs(target,source,is_blocked):
            return True
        
        return False
from queue import PriorityQueue
class Solution:
    def isEscapePossible(self, blocked: List[List[int]], source: List[int], target: List[int]) -> bool:
        
       #Column and row of source and target (used for escape condition)
        sc, sr = source
        tc, tr = target
        R, C = 10**6, 10**6

        def dist(r1,c1,r2,c2):
            '''Calculates Manhattan distance from (r1,c1) to (r2,c2)'''
            return abs(r2-r1)+abs(c2-c1)

        #Two priority queues: one starting from target and one from source
        #Two visited sets: one for nodes visited by path from target and the other from source
        q = [[(0,*source[::-1])], [(0,*target[::-1])]]
        v = [set(),set()]
        b = set((tuple(b[::-1]) for b in blocked))

        if (tuple(source) in b) or (tuple(target) in b):
            return False

        #if source and target can reach 200 distance from their origin
        #it is safe to say 200 blocked spaces cannot contain them
        source_escape = False
        target_escape = False

        while q[0] and q[1]:

            index = 0 if len(q[0]) <= len(q[1]) else 1

            d, r, c = heapq.heappop(q[index])

            for i, j in ((r+1, c), (r-1, c), (r, c+1), (r, c-1)):
                if (0 <= i < R) and (0 <= j < C) and ((i,j) not in b) and ((i,j) not in v[index]):

                    if (i,j) in v[1-index]:
                        return True

                    v[index].add((i,j))
                    r_target, c_target = q[1-index][0][1:]
                    heapq.heappush(q[index], (dist(i,j,r_target,c_target), i, j))

            if not source_escape and not index:
                source_escape = dist(r, c, sr, sc) > 200
            if not target_escape and index:
                target_escape = dist(r, c, tr, tc) > 200

            if source_escape and target_escape:
                return True

        return False
class Solution:
    def isValidStep(self, blocked, visited, step) -> bool:
        return tuple(step) not in blocked and tuple(step) not in visited and step[0] >= 0 and step[0] < 1000000 and step[1] >= 0 and step[1] < 1000000
    
    def isEscapePossibleHelper(self, blocked: set, source: List[int], target: List[int]) -> bool:
        directions = [[0, 1], [1, 0], [0, -1], [-1, 0]]
        nextSteps = []
        visited = set([tuple(source)])
        
        for d in directions:
            step = [source[0] + d[0], source[1] + d[1]]
            if self.isValidStep(blocked, visited, step):
                nextSteps.append(step)
                visited.add(tuple(step))

        while nextSteps:
            step = nextSteps.pop()
            if step == target or abs(step[0] - source[0]) + abs(step[1] - source[1]) >= 200:
                return True
            
            for d in directions:
                nextStep = [step[0] + d[0], step[1] + d[1]]
                if self.isValidStep(blocked, visited, nextStep):
                    nextSteps.append(nextStep)
                    visited.add(tuple(step))
                    if len(visited) > 20000:
                        return True

        return False
    
    def isEscapePossible(self, blocked: List[List[int]], source: List[int], target: List[int]) -> bool:
        blocked = set(tuple(b) for b in blocked)
        return self.isEscapePossibleHelper(blocked, source, target) and self.isEscapePossibleHelper(blocked, target, source)
class Solution:
    def isEscapePossible(self, b: List[List[int]], s: List[int], t: List[int]) -> bool:
        def dis(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])
        
        def find(source,  target, blocked):
            queue = [source]
            visited = set()

            while queue:
                x, y = queue.pop()

                if dis((x, y), source) > 200:
                    return True

                if x == target[0] and y == target[1]:
                    return True

                visited.add((x, y))

                next_options = []
                for x_delta, y_delta in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                    new_x, new_y = x + x_delta, y + y_delta

                    if new_x < 0 or new_x >= 10 ** 6 or new_y < 0 or new_y >= 10 ** 6:
                        continue

                    if (new_x, new_y) in visited or (new_x, new_y) in blocked:
                        continue

                    next_options.append((new_x, new_y))

                next_options.sort(key=lambda point: (point[0] - source[0]) ** 2 + (point[1] - source[1]) ** 2)
                queue += next_options
        
            return False
    
        bl = set()
        for bi in b:
            bl.add(tuple(bi))
            
        return find(s, t, bl) and find(t, s, bl)
class Solution:
    def isEscapePossible(self, blocked: List[List[int]], source: List[int], target: List[int]) -> bool:
        
        
        m = n = 10**6
        
        max_area = len(blocked)**2/2
        blocked = set(tuple(x) for x in blocked)

        def dfs(r, c, dst, visited, count, steps):
            # print(r, c)
            if (r, c) in visited:
                return False
            if (r, c) == dst:
                return True
            
            visited.add((r,c))
            if (r,c) in blocked:
                count[0] += 1
                return False



            if 0<=r<m and 0<=c<n:
                if count[0] >= 200:
                    return True
                if steps[0] >= max_area:
                    return True
                
                steps[0] += 1

                for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)]:
                    if dfs(r+dr, c+dc, dst, visited, count, steps):
                        return True

            return False


        return dfs(source[0], source[1], tuple(target), set(), [0], [0]) and dfs(target[0], target[1], tuple(source), set(), [0], [0])

import heapq

def solve(b,s,t):
    def create_priority_item(c, t):
        dx = c[0]-t[0]
        dy = c[1]-t[1]
        d2 = dx*dx + dy*dy
        return (d2, c)

    b = set(tuple(_b) for _b in b)
    s = tuple(s)
    t = tuple(t)
    heap = [s]
    visited = set()
    iter = -1
    while heap:
        iter += 1
        if iter > 1.1e6:
            return False
        c = heap.pop()
        if c in visited or c in b or c[0] < 0 or c[0] >=1e6 or c[1]<0 or c[1]>=1e6:
            continue
        if c == t:
            # found!
            return True
        # search neighbors:
        dx = c[0] - s[0]
        dy = c[1] - s[1]
        if dx*dx + dy*dy > 200*200:
            return True

        visited.add(c)
        heap.append((c[0]+1, c[1]  ))
        heap.append((c[0]-1, c[1]  ))
        heap.append((c[0]  , c[1]+1))
        heap.append((c[0]  , c[1]-1))
    # we live in a cavity :(
    return False

def solve_both(b,s,t):
    return solve(b,s,t) and solve(b,t,s)




class Solution:
    def isEscapePossible(self, blocked: List[List[int]], source: List[int], target: List[int]) -> bool:
        return solve_both(blocked, source, target)

class Solution:
    def isEscapePossible(self, blocked: List[List[int]], source: List[int], target: List[int]) -> bool:
        return self.helper(blocked, source, target) and self.helper(blocked, target, source)
    def helper(self, blocked, source, target):
        if not blocked: return True
        dq = collections.deque([tuple(source)])
        l = len(blocked)
        seen = {tuple(source)}
        blocked = set(map(tuple, blocked))
        while dq:
            sz = len(dq)
            for _ in range(sz):
                x, y = dq.popleft()
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    xx, yy = x + dx, y + dy
                    if 0 <= xx < 10 ** 6 and 0 <= yy < 10 ** 6 and (xx, yy) not in seen and (xx, yy) not in blocked:
                        seen.add((xx, yy))
                        dq.append((xx, yy))
                        # the maximum area covered by blocks will be an isosceles right triangle with area less than l * l // 2
                        # if we can cover more cells than l * l // 2, we will be bound to break the block
                        if (xx, yy) == tuple(target) or len(seen) >= l * l // 2: return True
        return False
import heapq

def solve(b,s,t):
    def create_priority_item(c, t):
        dx = c[0]-t[0]
        dy = c[1]-t[1]
        d2 = dx*dx + dy*dy
        return (d2, c)

    b = set(tuple(_b) for _b in b)
    s = tuple(s)
    t = tuple(t)
    heap = [(-1,s)]
    visited = set()
    iter = -1
    while heap:
        iter += 1
        if iter > 1.1e5:
            return False
        _, c = heapq.heappop(heap)
        if c in visited or c in b or c[0] < 0 or c[0] >=1e6 or c[1]<0 or c[1]>=1e6:
            continue
        if c == t:
            # found!
            return True
        # search neighbors:
        dx = c[0] - s[0]
        dy = c[1] - s[1]
        if dx*dx + dy*dy > 200*200:
            print(('found!', c, t))
            return True
        visited.add(c)
        heapq.heappush(heap, create_priority_item((c[0]+1, c[1]  ), t))
        heapq.heappush(heap, create_priority_item((c[0]-1, c[1]  ), t))
        heapq.heappush(heap, create_priority_item((c[0]  , c[1]+1), t))
        heapq.heappush(heap, create_priority_item((c[0]  , c[1]-1), t))
    # we live in a cavity :(
    return False

def solve_both(b,s,t):
    return solve(b,s,t) and solve(b,t,s)




class Solution:
    def isEscapePossible(self, blocked: List[List[int]], source: List[int], target: List[int]) -> bool:
        return solve_both(blocked, source, target)

class Solution:
    def isEscapePossible(self, blocked: List[List[int]], source: List[int], target: List[int]) -> bool:
        q = collections.deque([source])
        visited = set([tuple(source)])
        blocked = set([tuple(b) for b in blocked])
        lb = len(blocked)*3
        r = math.ceil(lb/(math.pi)/2.0)
        Max = math.ceil(r*r*math.pi)
        dx,dy = [0,1,0,-1],[1,0,-1,0]
        breakflag = False
        while q:
            if len(visited) > Max:
                breakflag = True
                break
            l = len(q)
            for i in range(l):
                x,y = q.popleft()
                for j in range(4):
                    newx,newy = x+dx[j],y+dy[j]
                    if 0<=newx<1000000 and 0<=newy<1000000 and (newx,newy) not in visited and (newx,newy) not in blocked:
                        if newx == target[0] and newy == target[1]:
                            return True
                        visited.add((newx,newy))
                        q.append((newx,newy))
        if breakflag == False:
            return False
        breakflag = False
        q = collections.deque([target])
        visited = set([tuple(target)])
        while q:
            if len(visited) > Max:
                breakflag = True
                break
            l = len(q)
            for i in range(l):
                x,y = q.popleft()
                for j in range(4):
                    newx,newy = x+dx[j],y+dy[j]
                    if 0<=newx<1000000 and 0<=newy<1000000 and (newx,newy) not in visited and (newx,newy) not in blocked:
                        visited.add((newx,newy))
                        q.append((newx,newy))
        if breakflag == False:
            return False
        return True
                    

import heapq

def solve(b,s,t):
    def create_priority_item(c, t):
        dx = c[0]-t[0]
        dy = c[1]-t[1]
        d2 = dx*dx + dy*dy
        return (d2, c)

    b = set(tuple(_b) for _b in b)
    s = tuple(s)
    t = tuple(t)
    heap = [(-1,s)]
    visited = set()
    iter = -1
    while heap:
        iter += 1
        if iter > 1.1e5:
            return False
        _, c = heapq.heappop(heap)
        if c in visited or c in b or c[0] < 0 or c[0] >=1e6 or c[1]<0 or c[1]>=1e6:
            continue
        if c == t:
            # found!
            return True
        # search neighbors:
        dx = c[0] - s[0]
        dy = c[1] - s[1]
        if dx*dx + dy*dy > 201*201:
            print(('found!', c, t))
            return True

        visited.add(c)
        # x = c[0]
        # while t[0] > x and (x+1,c[1]) not in b:
        #     x += 1
#         heapq.heappush(heap, create_priority_item((x, c[1]  ), t))
#         x = c[0]
#         while t[0] < x and (x-1,c[1]) not in b:
#             x -= 1
#         heapq.heappush(heap, create_priority_item((x, c[1]  ), t))

#         y = c[1]
#         while t[1] > y and (c[0],y+1) not in b:
#             y += 1
#         heapq.heappush(heap, create_priority_item((c[0], y  ), t))
#         y = c[1]
#         while t[1] < y and (c[0],y-1) not in b:
#             y -= 1
#         heapq.heappush(heap, create_priority_item((c[0], y  ), t))


        heapq.heappush(heap, create_priority_item((c[0]+1, c[1]  ), t))
        heapq.heappush(heap, create_priority_item((c[0]-1, c[1]  ), t))
        heapq.heappush(heap, create_priority_item((c[0]  , c[1]+1), t))
        heapq.heappush(heap, create_priority_item((c[0]  , c[1]-1), t))
    # we live in a cavity :(
    return False

def solve_both(b,s,t):
    return solve(b,s,t) and solve(b,t,s)




class Solution:
    def isEscapePossible(self, blocked: List[List[int]], source: List[int], target: List[int]) -> bool:
        return solve_both(blocked, source, target)

class Solution:
    def isEscapePossible(self, blocked: List[List[int]], source: List[int], target: List[int]) -> bool:
        bl = {tuple(b) for b in blocked}
        s0, s1 = source
        t0, t1 = target
        s_vis = {(s0, s1)}
        t_vis = {(t0, t1)}
        s_q = [(s0, s1)]
        t_q = [(t0, t1)]
        while s_q and len(s_vis) < 20010:
            n0, n1 = s_q.pop()
            if (n0, n1) == (t0, t1):
                return True
            if n0 < 10**6-1 and (n0+1, n1) not in s_vis and (n0+1, n1) not in bl:
                s_q.append((n0+1, n1))
                s_vis.add((n0+1, n1))
            if n0 > 0 and (n0-1, n1) not in s_vis and (n0-1, n1) not in bl:
                s_q.append((n0-1, n1))
                s_vis.add((n0-1, n1))
            if n1 < 10**6-1 and (n0, n1+1) not in s_vis and (n0, n1+1) not in bl:
                s_q.append((n0, n1+1))
                s_vis.add((n0, n1+1))
            if n1 > 0 and (n0, n1-1) not in s_vis and (n0, n1-1) not in bl:
                s_q.append((n0, n1-1))
                s_vis.add((n0, n1-1))
        while t_q and len(t_vis) < 20010:
            n0, n1 = t_q.pop()
            if (n0, n1) == (s0, s1):
                return True
            if n0 < 10**6-1 and (n0+1, n1) not in t_vis and (n0+1, n1) not in bl:
                t_q.append((n0+1, n1))
                t_vis.add((n0+1, n1))
            if n0 > 0 and (n0-1, n1) not in t_vis and (n0-1, n1) not in bl:
                t_q.append((n0-1, n1))
                t_vis.add((n0-1, n1))
            if n1 < 10**6-1 and (n0, n1+1) not in t_vis and (n0, n1+1) not in bl:
                t_q.append((n0, n1+1))
                t_vis.add((n0, n1+1))
            if n1 > 0 and (n0, n1-1) not in t_vis and (n0, n1-1) not in bl:
                t_q.append((n0, n1-1))
                t_vis.add((n0, n1-1))

        return bool(t_q and s_q)

import heapq

def solve(b,s,t):
    def create_priority_item(c, t):
        dx = c[0]-t[0]
        dy = c[1]-t[1]
        d2 = dx*dx + dy*dy
        return (d2, c)

    b = set(tuple(_b) for _b in b)
    s = tuple(s)
    t = tuple(t)
    heap = [s]
    visited = set()
    iter = -1
    while heap:
        iter += 1
        if iter > 1.1e5:
            return False
        c = heap.pop()
        if c in visited or c in b or c[0] < 0 or c[0] >=1e6 or c[1]<0 or c[1]>=1e6:
            continue
        if c == t:
            # found!
            return True
        # search neighbors:
        dx = c[0] - s[0]
        dy = c[1] - s[1]
        if dx*dx + dy*dy > 200*200:
            return True

        visited.add(c)
        heap.append((c[0]+1, c[1]  ))
        heap.append((c[0]-1, c[1]  ))
        heap.append((c[0]  , c[1]+1))
        heap.append((c[0]  , c[1]-1))
    # we live in a cavity :(
    return False

def solve_both(b,s,t):
    return solve(b,s,t) and solve(b,t,s)




class Solution:
    def isEscapePossible(self, blocked: List[List[int]], source: List[int], target: List[int]) -> bool:
        return solve_both(blocked, source, target)

class Solution:
    def isEscapePossible(self, blocked: List[List[int]], source: List[int], target: List[int]) -> bool:
        blocked = set(map(tuple, blocked))
        
        def dfs(x, y, sink, visited):
            if not (0 <= x < 10**6 and 0 <= y < 10**6) or (x, y) in blocked or (x, y) in visited:
                return False
            
            visited.add((x, y))
            # max blocked cell = 200
            if len(visited) > 20000 or [x, y] == sink:
                return True
            return dfs(x + 1, y, sink, visited) or dfs(x - 1, y, sink, visited) or dfs(x, y + 1, sink, visited) or dfs(x, y - 1, sink, visited)
        
        return dfs(source[0], source[1], target, set()) and dfs(target[0], target[1], source, set())
class Solution:
    def isEscapePossible(self, blocked: List[List[int]], source: List[int], target: List[int]) -> bool:
        blocked=set(map(tuple,blocked))
       
        def dfs(x,y,target,blocked,seen):
            if not (0<=x<1e6 and 0<=y<1e6) or (x,y) in blocked or (x,y) in seen:
                return False
            seen.add((x,y))
            if len(seen)>20000 or [x,y]==target:
                return True
            return dfs(x+1,y,target,blocked,seen) or dfs(x-1,y,target,blocked,seen) or dfs(x,y+1,target,blocked,seen) or dfs(x,y-1,target,blocked,seen)
        return dfs(source[0],source[1],target,blocked,set()) and dfs(target[0],target[1],source,blocked,set())

class Solution:
    def isEscapePossible(self, blocked: List[List[int]], source: List[int], target: List[int]) -> bool:
        blocked = set(map(tuple, blocked))
        
        def inbounds(x, y):
            return (x >= 0 and x < 10**6 and y >= 0 and y < 10**6)

        def dfs(x, y, target, seen):
            if (x,y) in blocked or not inbounds(x,y) or (x,y) in seen:
                return False
            seen.add((x,y))
            
            if len(seen) > 20000 or [x, y] == target:
                return True
            
            return dfs(x+1, y, target, seen) or dfs(x-1, y, target, seen) or dfs(x, y+1, target, seen) or dfs(x, y-1,target,seen)
            if not (0 <= x < 10**6 and 0 <= y < 10**6) or (x, y) in blocked or (x, y) in seen: return False
        return dfs(source[0], source[1], target, set()) and dfs(target[0], target[1], source, set())
class Solution:
    def isEscapePossible(self, blocked: List[List[int]], source: List[int], target: List[int]) -> bool:
        blocked = set(map(tuple, blocked))
        
        def inbounds(x, y):
            return (x >= 0 and x < 10**6 and y >= 0 and y < 10**6)

        def dfs(x, y, target, seen):
            if (x,y) in blocked or not inbounds(x,y) or (x,y) in seen:
                return False
            seen.add((x,y))
            
            if len(seen) > (200*199/2) or [x, y] == target:
                return True
            
            return dfs(x+1, y, target, seen) or dfs(x-1, y, target, seen) or dfs(x, y+1, target, seen) or dfs(x, y-1,target,seen)
            if not (0 <= x < 10**6 and 0 <= y < 10**6) or (x, y) in blocked or (x, y) in seen: return False
        return dfs(source[0], source[1], target, set()) and dfs(target[0], target[1], source, set())
class Solution:
    def isEscapePossible(self, blocked: List[List[int]], source: List[int], target: List[int]) -> bool:
        blocked = set(map(tuple, blocked))
        src, target = tuple(source), tuple(target)
        
        # srcu548ctargetu8981u5206u522bu4f5cu4e3au8d77u70b9u8bd5u4e00u4e0buff0cu9632u6b62targetu88abu5305u56f4u4e86u4f46u662fsourceu8fd8u662fu53efu4ee5u8d70u5f88u8fdcu7684u60c5u51b5
        # dfsu91ccu7684seenu8981u5355u72ecu8f93u5165set(), u5426u5219u4f1au6cbfu7528u4e0au4e00u6b21dfsu7684seenu800cu5f71u54cdu7ed3u679c
        return self.dfs(src, target, set(), blocked) and self.dfs(target, src, set(), blocked)
    
    def dfs(self, src, target, seen, blocked):
        
        if len(seen) > 20000 or src == target: return True 
        
        x0, y0 = src
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            x = x0 + dx
            y = y0 + dy
            if (0<=x<10**6 and 0<=y<10**6 and ((x, y) not in seen) and ((x, y) not in blocked)):
                seen.add((x, y))
                if self.dfs((x, y), target, seen, blocked):
                    return True 
                
        return False

class Solution:
    def isEscapePossible(self, blocked: List[List[int]], source: List[int], target: List[int]) -> bool:
        def method1(blocked):
            R=C=10**6
            if not (0<=source[0]<R and 0<=source[1]<C):
                return False
            
            if not (0<=target[0]<R and 0<=target[1]<C):
                return False
            
            if not blocked:
                return True
            
            blocked=set(map(tuple,blocked))
            seen=set()
            
            def neighbors(r,c):
                for nr,nc in [(r-1,c),(r+1,c),(r,c-1),(r,c+1)]:
                    if 0<=nr<R and 0<=nc<C:
                        yield nr,nc
            
            def dfs(r,c):
                if (r,c) in seen:
                    return False
                
                if [r,c]==target:
                    return True
                
                if (r,c) in blocked:
                    return False
                
                seen.add((r,c))
                for nr,nc in neighbors(r,c):
                    if dfs(nr,nc):
                        return True
                return False
            
            return dfs(*source)
        
        #return method1(blocked)
    
        def method2(blocked):
            if not blocked: 
                return True
                
            blocked = set(map(tuple, blocked))
        
            def check(blocked, source, target):
                si, sj = source
                ti, tj = target
                level = 0
                q = collections.deque([(si,sj)])
                vis = set()
                while q:
                    for _ in range(len(q)):
                        i,j = q.popleft()
                        if i == ti and j == tj: return True
                        for x,y in ((i+1,j),(i-1,j),(i,j+1),(i,j-1)):
                            if 0<=x<10**6 and 0<=y<10**6 and (x,y) not in vis and (x,y) not in blocked:
                                vis.add((x,y))
                                q.append((x,y))
                    level += 1
                    if level == len(blocked): 
                        return True
                #else:
                return False
                #return True
        
            return check(blocked, source, target) and check(blocked, target, source)
        
        return method2(blocked)
            
                

class Solution:
    def isEscapePossible(self, blocked: List[List[int]], source: List[int], target: List[int]) -> bool:
        def bfs(source, target):
            r0, c0 = source
            rd, cd = target
            seen = {(r0, c0)}
            dq = deque([(r0, c0)])
            N = 10**6
            cnt = 0
            while dq:
                r, c = dq.popleft()
                cnt += 1
                if r==rd and c==cd or cnt>19900: return True
                
                for ro, co in [(1,0), (-1,0), (0,1), (0,-1)]:
                    nr, nc = r+ro, c+co
                    if 0<=nr<N and 0<=nc<N and (nr, nc) not in seen and (nr, nc) not in bset:
                        seen.add((nr, nc))
                        dq.append((nr, nc))
                
            return False
        
        bset = {tuple(b) for b in blocked}
        return bfs(source, target) and bfs(target, source)
class Solution:
    def isEscapePossible(self, blocked: List[List[int]], source: List[int], target: List[int]) -> bool:
        def method1(blocked):
            R=C=10**6
            if not (0<=source[0]<R and 0<=source[1]<C):
                return False
            
            if not (0<=target[0]<R and 0<=target[1]<C):
                return False
            
            if not blocked:
                return True
            
            blocked=set(map(tuple,blocked))
            seen=set()
            
            def neighbors(r,c):
                for nr,nc in [(r-1,c),(r+1,c),(r,c-1),(r,c+1)]:
                    if 0<=nr<R and 0<=nc<C:
                        yield nr,nc
            
            def dfs(r,c):
                if (r,c) in seen:
                    return False
                
                if [r,c]==target:
                    return True
                
                if (r,c) in blocked:
                    return False
                
                seen.add((r,c))
                for nr,nc in neighbors(r,c):
                    if dfs(nr,nc):
                        return True
                return False
            
            return dfs(*source)
        
        #return method1(blocked)
    
        def method2(blocked):
            if not blocked: 
                return True
                
            blocked = set(map(tuple, blocked))
        
            def check(blocked, source, target):
                si, sj = source
                ti, tj = target
                level = 0
                q = collections.deque([(si,sj)])
                vis = set()
                while q:
                    for _ in range(len(q)):
                        i,j = q.popleft()
                        if i == ti and j == tj: return True
                        for x,y in ((i+1,j),(i-1,j),(i,j+1),(i,j-1)):
                            if 0<=x<10**6 and 0<=y<10**6 and (x,y) not in vis and (x,y) not in blocked:
                                vis.add((x,y))
                                q.append((x,y))
                    level += 1
                    if level == len(blocked): 
                        break
                else:
                    return False
                return True
        
            return check(blocked, source, target) and check(blocked, target, source)
        
        return method2(blocked)
            
                

from collections import deque

class Solution:
    def isEscapePossible(self, blocked: List[List[int]], source: List[int], target: List[int]) -> bool:
        if not blocked: return True
        blocked = set(map(tuple, blocked))
        
        def check(source, target):
            si, sj = source
            ti, tj = target
            level = 0
            q = deque([(si,sj)])
            seen = set()
            while q:
                for _ in range(len(q)):
                    i, j = q.popleft()
                    if i == ti and j == tj: return True
                    for x,y in ((i+1,j),(i-1,j),(i,j+1),(i,j-1)):
                        if 0<=x<10**6 and 0<=y<10**6 and (x,y) not in seen and (x,y) not in blocked:
                            seen.add((x,y))
                            q.append((x,y))
                level += 1
                if level == len(blocked): break
            else:
                return False
            return True
        
        return check(source, target) and check(target, source)
class Solution:
    def isEscapePossible(self, blocked: List[List[int]], source: List[int], target: List[int]) -> bool:
        
        q1 = [tuple(source)]
        q2 = [tuple(target)]
        vis1 = set([tuple(source)])
        vis2 = set([tuple(target)])
        blocked = set([tuple(b) for b in blocked])
        dirs = [[-1,0],[1,0],[0,-1],[0,1]]
        
        def valid(r, c):
            if r>=0 and r<1000000 and c>=0 and c<1000000:
                return True
            return False
        
        while q1 and q2:
            
            if len(q1) > len(blocked) and len(q2) > len(blocked):
                return True
            
            temp = []
            for r, c in q1:
                if (r, c) in vis2:
                    return True
                for d in dirs:
                    nr = r+d[0]
                    nc = c+d[1]
                    if valid(nr, nc):
                        if (nr,nc) not in vis1 and (nr, nc) not in blocked:
                            temp.append((nr,nc))
                            vis1.add((nr,nc))
            q1 = temp
            
            
            
            temp = []
            for r, c in q2:
                if (r, c) in vis1:
                    return True
                for d in dirs:
                    nr = r+d[0]
                    nc = c+d[1]
                    if valid(nr, nc):
                        if (nr,nc) not in vis2 and (nr, nc) not in blocked:
                            temp.append((nr,nc))
                            vis2.add((nr,nc))
            q2 = temp
            
        return False
            
                
                

class Solution:
    def isEscapePossible(self, blocked: List[List[int]], source: List[int], target: List[int]) -> bool:
        m = 10 ** 6
        n = len(blocked)
        if target in blocked or source in blocked: return False
        if n <= 1: return True
        dxy = [[0, -1], [0, 1], [-1, 0], [1, 0]]
        blocked = set(map(tuple, blocked))
        threshold = 100 * 200
        def bfs(pos, target):
            q = collections.deque([pos])
            visited = {tuple(pos)}
            cnt = 0
            while q:
                x, y = q.popleft()
                if x == target[0] and y == target[1]:
                    return 1
                cnt += 1
                if cnt > threshold:
                    return 2
                for dx, dy in dxy:
                    x_, y_ = x + dx, y + dy
                    if 0 <= x_ < m and 0 <= y_ < m:
                        p = (x_, y_)
                        if p not in visited and p not in blocked:
                            q.append(p)
                            visited.add(p)
            return -1
        
        i = bfs(source, target)
        print(i)
        if i == 1:
            return True
        if i == -1:
            return False
        j = bfs(target, source)
        print(j)
        return j == 2
                     
        

from collections import deque
class Solution:
    def isEscapePossible(self, blocked: List[List[int]], source: List[int], target: List[int]) -> bool:
        if not blocked or not blocked[0]:
            return True
        
        blocked = set(map(tuple, blocked))
        print(blocked)
        return self.bfs(tuple(source), tuple(target), blocked) and self.bfs(tuple(target), tuple(source), blocked)
        
        
        
    def bfs(self, source, target, blocked):
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        
        queue = deque([source])
        
        visited = set([source])
        
        while queue:
            x, y = queue.popleft()
            for (dx, dy) in directions:
                x_ , y_ = x + dx, y + dy
                if 0 <= x_ < 10 ** 6 and 0 <= y_ < 10 ** 6 and (x_, y_) not in visited and (x_, y_) not in blocked:
                    if x_ == target[0] and y_ == target[1]:
                        return True
                    queue.append((x_, y_))
                    visited.add((x_, y_))
            if len(visited) > 20000:
                return True
        return False
    
                    
                 
        
        
        
        
        
        
        
        
        
        
#     def isEscapePossible(self, blocked, source, target):
#         blocked = set(map(tuple, blocked))

#         def dfs(x, y, target, seen):
#             if not (0 <= x < 10**6 and 0 <= y < 10**6) or (x, y) in blocked or (x, y) in seen: return False
#             seen.add((x, y))
#             if len(seen) > 20000 or [x, y] == target: return True
#             return dfs(x + 1, y, target, seen) or 
#                 dfs(x - 1, y, target, seen) or 
#                 dfs(x, y + 1, target, seen) or 
#                 dfs(x, y - 1, target, seen)
#         return dfs(source[0], source[1], target, set()) and dfs(target[0], target[1], source, set())
# Python, BFS:
#     def isEscapePossible(self, blocked, source, target):
#         blocked = {tuple(p) for p in blocked}

#         def bfs(source, target):
#             bfs, seen = [source], {tuple(source)}
#             for x0, y0 in bfs:
#                 for i, j in [[0, 1], [1, 0], [-1, 0], [0, -1]]:
#                     x, y = x0 + i, y0 + j
#                     if 0 <= x < 10**6 and 0 <= y < 10**6 and (x, y) not in seen and (x, y) not in blocked:
#                         if [x, y] == target: return True
#                         bfs.append([x, y])
#                         seen.add((x, y))
#                 if len(bfs) == 20000: return True
#             return False
#         return bfs(source, target) and bfs(target, source)

from collections import deque
class Solution:
    def isEscapePossible(self, blocked: List[List[int]], source: List[int], target: List[int]) -> bool:
        if not blocked or not blocked[0]:
            return True
        
        blocked = set(map(tuple, blocked))

        return self.bfs(tuple(source), tuple(target), blocked) and self.bfs(tuple(target), tuple(source), blocked)
        
        
    def bfs(self, source, target, blocked):
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        
        queue = deque([source])
        visited = set([source])
        
        while queue:
            x, y = queue.popleft()
            for (dx, dy) in directions:
                x_ , y_ = x + dx, y + dy
                if 0 <= x_ < 10 ** 6 and 0 <= y_ < 10 ** 6 and (x_, y_) not in visited and (x_, y_) not in blocked:
                    if x_ == target[0] and y_ == target[1]:
                        return True
                    queue.append((x_, y_))
                    visited.add((x_, y_))
            if len(visited) > 20000:
                return True
        return False
    
                    

class Solution:
    
    def isEscapePossible(self, blocked, source, target):
        blocked = set(map(tuple, blocked))
        source = tuple(source)
        target = tuple(target)
        
        def neighbors(node):
            i, j = node
            return (
                (i+di, j+dj)
                for (di, dj) in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                if 0 <= i+di < 10**6
                if 0 <= j+dj < 10**6
                if (i+di, j+dj) not in blocked
            )
        
        def mhat_dist(node0, node1):
            i0, j0 = node0
            i1, j1 = node1
            return abs(i0-i1) + abs(j0-j1)
        
        stack = [source]
        visited = {source}
        exceeded_threshold = False
        while stack:
            node = stack.pop()
            if node == target:
                return True
            if mhat_dist(source, node) > 200:
                exceeded_threshold = True
                break
            new_nodes = set(neighbors(node)) - visited
            visited.update(new_nodes)
            stack.extend(new_nodes)
            
        if not exceeded_threshold:
            return False
        
        stack = [target]
        visited = {target}
        while stack:
            node = stack.pop()
            if mhat_dist(source, node) > 200:
                return True
            new_nodes = set(neighbors(node)) - visited
            visited.update(new_nodes)
            stack.extend(new_nodes)
        return False
        
                

class Solution:
    def isEscapePossible(self, blocked: List[List[int]], source: List[int], target: List[int]) -> bool:
        if not blocked:
            return True
        M, N = 10 ** 6, 10 ** 6
        blocked = set(map(tuple, blocked))
        
        def bfs(src, tgt):
            q = collections.deque([tuple(src)])
            k = 0
            seen = set(tuple(src))
            while q:
                i, j = q.popleft()
                k += 1
                if [i, j] == tgt or k == 20000:
                    return True
                for x, y in [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]:
                    if 0 <= x < M and 0 <= y < N and (x, y) not in seen and (x, y) not in blocked:
                        seen.add((x, y))
                        q.append((x, y))
            return False
        
        return bfs(source, target) and bfs(target, source)
from collections import deque

class Solution:
    def isEscapePossible(self, blocked: List[List[int]], source: List[int], target: List[int]) -> bool:
        if not blocked: return True
        blocked = set(tuple(b) for b in blocked)
        
        def util(source, target):
            si, sj = source
            ti, tj = target
            step = 0
            q = deque([(si, sj)])
            seen = {(si, sj)}
            
            while q:
                step += 1
                if step == len(blocked): return True
                for _ in range(len(q)):
                    i, j = q.popleft()
                    if i == ti and j == tj: return True
                    for r, c in [(i-1, j), (i, j+1), (i+1,j), (i,j-1)]:
                        if 0 <= r < 1000000 and 0 <= c < 1000000 and (r, c) not in blocked and (r, c) not in seen:
                            seen.add((r, c))
                            q.append((r, c))
                            
            return False
        
        return util(source, target) and util(target, source)
class Solution:
    def isEscapePossible(self, blocked: List[List[int]], source: List[int], target: List[int]) -> bool:
        if not blocked:
            return True
        blocked = set(map(tuple, blocked))

        def bfs(start: list, final: list):
            seen, stack = {tuple(start)}, [start]
            for x, y in stack:
                for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                    if 0 <= nx < 10 ** 6 and 0 <= ny < 10 ** 6 and (nx, ny) not in seen and (nx, ny) not in blocked:
                        if [nx, ny] == final:
                            return True
                        seen.add((nx, ny))
                        stack.append((nx, ny))
                if len(stack) == 20000:
                    return True
            return False

        return bfs(source, target) and bfs(target, source)

class Solution:
    def isEscapePossible(self, blocked: List[List[int]], source: List[int], target: List[int]) -> bool:
        blocked = {tuple(p) for p in blocked}
        direction = [[0,1],[0,-1],[1,0],[-1,0]]
        
        def bfs(source, target):
            bfs, seen = [source], {tuple(source)}
            
            for x0, y0 in bfs:
                for i, j in direction:
                    x, y = x0 + i, y0 + j
                    if 0 <= x < 10**6 and 0 <= y < 10**6 and (x, y) not in seen and (x, y) not in blocked:
                        if [x, y] == target: 
                            return True
                        
                        bfs.append([x, y])
                        seen.add((x, y))
                        
                if len(bfs) == 20000: 
                    return True
                
            return False
        
        return bfs(source, target) and bfs(target, source)
    
    
class Solution_:
    def isEscapePossible_(self, blocked: List[List[int]], source: List[int], target: List[int]) -> bool:
        
        queue = collections.deque()
        direction = [[0,1],[0,-1],[1,0],[-1,0]]
        
        queue.append(source)
        
        s = set()
        for i,j in blocked:
            s.add((i,j))
  
        while queue:
            x, y = queue.popleft()
            
            if [x,y] == target:
                return True
            
            for xx, yy in direction:
                if x+xx >= 0 and x+xx < 10**6 and y+yy >= 0 and y+yy < 10**6 and (x+xx,y+yy) not in s:
                    queue.append([x+xx,y+yy])
                    s.add((x+xx,y+yy))
            
        return False
import collections

class Solution:
    def isEscapePossible(self, blocked: List[List[int]], source: List[int], target: List[int]) -> bool:
        if not blocked: 
            return True
         
        blocked = set(map(tuple, blocked))
        
        
        return self.bfs(blocked, source, target) and self.bfs(blocked, target, source)
        
    def bfs(self, blocked, source, target):
        si, sj = source
        ti, tj = target
        
        queue = collections.deque([(si, sj)])
        visited = set()
        
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        level = 0
        while queue:
            size = len(queue)
            for _ in range(size):
                curX, curY = queue.popleft()
                
                 
                if curX == ti and curY == tj:
                    return True
                
                for dx, dy in directions:
                    newX = curX + dx
                    newY = curY + dy
                    if 0 <= newX < 1000000 and  0 <= newY < 1000000 and (newX, newY) not in blocked and (newX, newY) not in visited:
                        queue.append((newX, newY))
                        visited.add((newX, newY))
            level += 1
            if level == len(blocked):
                break
                
        else:
            return False
        return True

class Solution:
    def isEscapePossible(self, blocked: List[List[int]], source: List[int], target: List[int]) -> bool:
        blocked = {tuple(i) for i in blocked}
        def bfs(source,target):
            que = [source]
            seen = {tuple(source)}
            for x,y in que:
                for i,j in [(0,1),(1,0),(-1,0),(0,-1)]:
                    m,n = x+i,y+j
                    if 0<=m<10**6 and 0<=n<10**6 and (m,n) not in seen and (m,n) not in blocked:
                        if m == target[0] and n==target[1]:
                            return True
                        que.append((m,n))
                        seen.add((m,n))
                    if len(que)>=20000:
                        return True
            return False
        return bfs(source,target) and bfs(target,source)


# 1036. Escape a Large Maze

class Solution:
    def isEscapePossible(self, blocked, source, target):
        blocked = {tuple(p) for p in blocked}
        directions = {(0, 1), (1, 0), (-1, 0), (0, -1)}
        
        def bfs(source, target):
            queue, seen = [source], {tuple(source)}
            for x0, y0 in queue:
                for i, j in directions:
                    x, y = x0 + i, y0 + j
                    if 0 <= x < 10**6 and 0 <= y < 10**6 and (x, y) not in seen and (x, y) not in blocked:
                        if [x, y] == target: 
                            return True
                        queue.append([x, y])
                        seen.add((x, y))
                if len(queue) == 20000: 
                    return True
            return False
        
        return bfs(source, target) and bfs(target, source)

    
    
    
class Solution1:
    def isEscapePossible(self, blocked: List[List[int]], source: List[int], target: List[int]) -> bool:
        
        if not blocked:
            return True
        
        rows = cols = 10**6
        directions = {(0,1), (0,-1), (1,0), (-1,0)}
        q1, q2 = [tuple(source)], [tuple(target)]
        seen = set()
        blocked = {tuple(p) for p in blocked}
        
        while q1:
            next_q = []
            # current level
            while q1:
                r, c = q1.pop()
                # print(r, c, q1)
                for dr, dc in directions:
                    if 0 <= r+dr < rows and 0 <= c+dc < cols and (r+dr, c+dc) not in blocked and (r+dr, c+dc) not in seen:
                        if (r+dr, c+dc) in q2 or [r+dr, c+dc] == target:
                            return True
                        next_q.append((r+dr, c+dc))
                        seen.add((r+dr, c+dc))

            q1 = next_q
            if len(q1) > len(q2):
                q1, q2 = q2, q1
                    
        return False
                        
                


# class Solution:
#     def isEscapePossible(self, blocked: List[List[int]], source: List[int], target: List[int]) -> bool:

#         block = set()
#         for b in blocked:
#             block.add((b[0], b[1]))
            
#         directions = [[-1, 0], [1, 0], [0, -1], [0, 1]]
#         def dfs(sx, sy, tx, ty, cx, cy, visited):
#             if cx == tx and cy == ty:
#                 return True
#             if (abs(sx - cx) + abs(sy - cy)) > 200:
#                 return True
#             visited.add((cx, cy))
#             for d in directions:
#                 r = cx + d[0]
#                 c = cy + d[1]
                
#                 if r >= 0 and r < 1000000 and c >=0 and r < 1000000:
#                     if (r,c) not in block and (r, c) not in visited:
#                         if  dfs(sx, sy, tx, ty, r, c, visited):
#                             return True
#             return False
#         v1 = set()  
#         v2 = set()
#         r1 = dfs(source[0], source[1], target[0], target[1], source[0], source[1], v1)
#         r2 = dfs(target[0], target[1], source[0], source[1], target[0], target[1], v2)
#         return r1 and  r2                
                

class Solution:
    def isEscapePossible(self, blocked: List[List[int]], source: List[int], target: List[int]) -> bool:
        # blocked = set(map(tuple, blocked))

        # def dfs(x, y, target, seen):
        #     if not (0 <= x < 10**6 and 0 <= y < 10**6) or (x, y) in blocked or (x, y) in seen:
        #         return False

        #     seen.add((x, y))

        #     if len(seen) > 20000 or [x, y] == target:
        #         return True

        #     return dfs(x + 1, y, target, seen) 
        #         or dfs(x - 1, y, target, seen) 
        #         or dfs(x, y + 1, target, seen) 
        #         or dfs(x, y - 1, target, seen)

        # return dfs(source[0], source[1], target, set()) 
        #    and dfs(target[0], target[1], source, set())


        blocked = {tuple(p) for p in blocked}

        def bfs(source, target):
            bfs, seen = [source], {tuple(source)}
            for x0, y0 in bfs:
                for i, j in (0, 1), (1, 0), (-1, 0), (0, -1):
                    x, y = x0 + i, y0 + j
                    if 0 <= x < 10**6 and 0 <=y < 10**6 and (x, y) not in seen and (x, y) not in blocked:
                        if [x, y] == target:
                            return True
                        bfs.append([x, y])
                        seen.add((x, y))
                if len(bfs) > 20000:
                    return True

            return False

        return bfs(source, target) and bfs(target, source)

class Solution:
    def isEscapePossible(self, blocked: List[List[int]], source: List[int], target: List[int]) -> bool:
        m = 10 ** 6
        n = len(blocked)
        if target in blocked or source in blocked: return False
        if n <= 1: return True
        dxy = [[0, -1], [0, 1], [-1, 0], [1, 0]]
        blocked = set(map(tuple, blocked))
        threshold = 150 * 150
        def bfs(pos, target):
            q = collections.deque([pos])
            visited = {tuple(pos)}
            cnt = 0
            while q:
                x, y = q.popleft()
                if x == target[0] and y == target[1]:
                    return 1
                cnt += 1
                if cnt > threshold:
                    return 2
                for dx, dy in dxy:
                    x_, y_ = x + dx, y + dy
                    if 0 <= x_ < m and 0 <= y_ < m:
                        p = (x_, y_)
                        if p not in visited and p not in blocked:
                            q.append(p)
                            visited.add(p)
            return -1
        
        i = bfs(source, target)
        print(i)
        if i == 1:
            return True
        if i == -1:
            return False
        j = bfs(target, source)
        print(j)
        return j == 2
                     
        

class Solution:
    def isEscapePossible(self, blocked: List[List[int]], source: List[int], target: List[int]) -> bool:
        blocked = set(map(tuple, blocked))
        
        def inbounds(x, y):
            return (x >= 0 and x < 10**6 and y >= 0 and y < 10**6)

        def dfs(x, y, target, seen):
            if (x,y) in blocked or not inbounds(x,y) or (x,y) in seen:
                return False
            seen.add((x,y))
            
            if len(seen) > 30000 or [x, y] == target:
                return True
            
            return dfs(x+1, y, target, seen) or dfs(x-1, y, target, seen) or dfs(x, y+1, target, seen) or dfs(x, y-1,target,seen)
            if not (0 <= x < 10**6 and 0 <= y < 10**6) or (x, y) in blocked or (x, y) in seen: return False
        return dfs(source[0], source[1], target, set()) and dfs(target[0], target[1], source, set())
'''
Using BFS

Idea and Facts

Another Reference:
https://leetcode.com/problems/escape-a-large-maze/discuss/282870/python-solution-with-picture-show-my-thoughts
https://assets.leetcode.com/users/2017111303/image_1556424333.png

FAQ
Question I think the maximum area is 10000?
Answer
The maximum area is NOT 10000. Even it's accepted with bound 10000, it's WRONG.
The same, the bfs with just block.size steps, is also wrong.

In the following case, the area is 19900.
The sum of the area available equals 1+2+3+4+5+...+198+199=(1+199)*199/2=19900 (trapezoid sum) --> Area = B*(B-1)/2
X -> blocking points

0th      _________________________
         |O O O O O O O X
         |O O O O O O X
         |O O O O O X
         |O O O O X
         .O O O X
         .O O X
         .O X
200th    |X

Question I think the maximum area is area of a sector.
Answer
All cells are discrete, so there is nothing to do with pi.


Question What is the maximum area?
Answer
It maximum blocked is achieved when the blocked squares,
surrounding one of the corners as a 45-degree straight line.

And it's easily proved.

If two cells are connected horizontally,
we can slide one part vertically to get bigger area.

If two cells are connected vertically,
we can slide one part horizontally to get bigger area.


Question Can we apply a BFS?
Answer
Yes, it works.
BFS in 4 directions need block.length * 2 as step bounds,
BFS in 8 directions need block.length as step bounds.

It needs to be noticed that,
The top voted BFS solution is WRONG with bound,
though it's accpected by Leetcode.

But compared with the complexity:
Searching with limited area is O(0.5B*B).
BFS with steps can be O(2B^B).


Intuition
Simple search will get TLE, because the big search space.
Anyway, we don't need to go further to know if we are blocked or not.
Because the maximum area blocked are 19900.


Explanation
Search from source to target,
if find, return true;
if not find, return false;
if reach 20000 steps, return true.

Then we do the same thing searching from target to source.

Complexity
Time complexity depends on the size of blocked
The maximum area blocked are B * (B - 1) / 2.
As a result, time and space complexity are both O(B^2)
In my solution I used a fixed upper bound 20000.
'''
class Solution:
    def isEscapePossible(self, blocked: List[List[int]], source: List[int], target: List[int]) -> bool:
        
        blocked = set(map(tuple, blocked))
        dirs = [(0,1), (1,0), (0,-1), (-1,0)]
        
        def bfs(s, t):
            bfsQ = [tuple(s)]
            visited = {tuple(s)}
            areaSum = 0
            
            for x, y in bfsQ:
                for i,j in dirs:
                    r = x+i
                    c = y+j
                    
                    if 0 <= r < 10**6 and 0 <= c < 10**6 and (r,c) not in visited and (r,c) not in blocked:
                        if (r,c) == tuple(t):
                            return True
                        bfsQ.append((r,c))
                        visited.add((r,c))
                        
                if len(bfsQ) >= 20000: # max block area upper bound
                    return True
            
            return False
    
        return bfs(source, target) and bfs(target, source)
class Solution:
    def isEscapePossible(self, blocked: List[List[int]], source: List[int], target: List[int]) -> bool:
        maxlen = 10**6
        #maxarea = (4/3.14) * 17000 # maxarea = (4/3.14) * 10000 does not work!
        maxarea = 20000
        dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        
        blocked = set(map(tuple, blocked)) # use a set to make it faster for retrieving
        
        def bfs(source, target):
            q = collections.deque()
            aset = set()
            q.append(source)
            while q and len(aset) < maxarea:
                row, col = q.popleft()
                if row == target[0] and col == target[1]:
                    return True
                aset.add((row,col))
                
                for dir in dirs:
                    row2 = row + dir[0]
                    col2 = col + dir[1]
                    if 0<=row2<maxlen and 0<=col2 < maxlen and not (row2, col2) in aset and not (row2,col2) in blocked:
                        q.append([row2, col2])
                        aset.add((row2, col2))
            return len(aset) >= maxarea # evaluate by maxarea
        return bfs(source, target) and bfs(target, source)
