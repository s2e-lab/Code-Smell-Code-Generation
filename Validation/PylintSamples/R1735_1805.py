class Solution:
    def watchedVideosByFriends(self, watchedVideos: List[List[str]], friends: List[List[int]], id: int, level: int) -> List[str]:
        vis = [0 for i in range(len(friends))]
        index = id
        lis = set([id])
        while level>0:
            temp = []
            for i in lis:
                if vis[i] == 0:
                    #print(i)
                    temp += friends[i]
                    vis[i] = 1
            lis = set(temp)
            level -= 1
        dic = dict()
        for i in lis:
            if vis[i] == 0:
                for j in watchedVideos[i]:
                    if j in dic:
                        dic[j]+=1
                    else:
                        dic[j] = 1
        dic2 = dict()
        for i in dic:
            if dic[i] in dic2:
                dic2[dic[i]].append(i)
            else:
                dic2[dic[i]] = [i]
        lis = []
        for i in sorted(dic2.keys()):
            lis += sorted(dic2[i])
        return lis

from collections import deque, Counter
class Solution:
    def watchedVideosByFriends(self, watchedVideos: List[List[str]], friends: List[List[int]], id: int, level: int) -> List[str]:
        # Var def
        all_video = 0
        videos_dict = {}
        queue = deque([id])
        visited = set()
        visited.add(id)
        current_level = 0
        
        # Create friends dict (for BFS)
        friends_dict = {id: []}
        
        
        
        # Get all friends at K level
        while queue: # 1, 2
            level_size = len(queue) # len=2
            level_friends = queue
            for _ in range(level_size): # [1, 2]
                cur_node = queue.popleft() # 2
                for friend in friends[cur_node]: #3
                    if friend in visited:
                        continue
                    queue.append(friend) # 
                    visited.add(friend)  # 3
            current_level += 1 # 2
            print(current_level)
            
            if current_level == level:
                break
        print(('friends', queue))    
        for friend in queue:
            
            for video in watchedVideos[friend]:
                if video in videos_dict:
                    videos_dict[video] += 1
                else:
                    videos_dict[video] = 1
            
        #print(videos_dict)    
        #print(sorted(videos_dict.items(), key=lambda x:(videos_dict[x], x[1]))   )
        #print(sorted(videos_dict, key=lambda x: videos_dict[x], reverse=False))
        return sorted(videos_dict, key=lambda x: (videos_dict[x], x) , reverse=False)
        
        #videos_dict = Counter('abcdaab')
        
                
                
            
        
        
        
        
        # Get the movies watch by that level of friends
        
        
        # Ordering
        
        
        
        
        

class Solution:
    def watchedVideosByFriends(self, watchedVideos: List[List[str]], friends: List[List[int]], id: int, level: int) -> List[str]:
        
        n = len(watchedVideos)
        
        dist = [9999]*n
        
        queue = [(id, 0)]
        while queue:
            idx, depth = queue.pop(0)
            if depth < dist[idx]:
                dist[idx] = depth
                for c in friends[idx]:
                    queue.append((c, depth+1))
                
        # print(dist)
        
        freq = defaultdict(int)
        for i, l in enumerate(dist):
            if l == level:
                for vid in watchedVideos[i]:
                    freq[vid] += 1
        
        pairs = sorted([(v, k) for k,v in list(freq.items())])
        # print(pairs)
        
        return [x[-1] for x in pairs]
        

class Solution:
    def watchedVideosByFriends(self, watchedVideos: List[List[str]], friends: List[List[int]], id: int, level: int) -> List[str]:
        frontier = [id]
        seen = {id}
        depth = 0
        while frontier:
            next_frontier = []
            if depth == level-1:
                count = collections.defaultdict(lambda: 0)
            for p in frontier:
                for fr in friends[p]:
                    if fr not in seen:
                        seen.add(fr)
                        if depth == level-1:
                            for video in watchedVideos[fr]:
                                count[video] += 1
                        next_frontier.append(fr)
            frontier = next_frontier
            depth += 1
            if depth == level:
                return sorted(count.keys(), key = lambda x:[count[x], x])
            
        return []
class Solution:
    def watchedVideosByFriends(self, watchedVideos: List[List[str]], friends: List[List[int]], id: int, level: int) -> List[str]:
        bfs,visited={id},{id}
        for _ in range(level):
            bfs={j for i in bfs for j in friends[i] if j not in visited}
            visited|=bfs
        freq=collections.Counter([v for idx in bfs for v in watchedVideos[idx]])
        return sorted(list(freq.keys()),key=lambda x:(freq[x],x))
#         d = defaultdict(list)
#         for i in range(len(friends)):
#             d[i].extend(friends[i])
        
#         videoCounts = []
#         for i in range(len(watchedVideos)):
#             videoCounts.append(len(watchedVideos[i]))
        
#         output = defaultdict(int)
#         q = []
#         q.append((id, 0))
#         while q:
#             node, lev = q.pop(0)
#             if lev > level:
#                 continue
#             if lev == level:
#                 for movie in watchedVideos[node]:
#                     output[movie] += 1
#                 continue
#             else:
#                 for nei in d[node]:
#                     q.append((nei, lev + 1))
        
#         ans = []
#         for key in output:
#             ans.append((key, output[key]))
#         ans.sort(key=lambda x: x[1])
#         return list(map(lambda x: x[0], ans))

class Solution:
    def bfs(self, id: int, target_level:int, friends:List[List[int]], watchedVideos:List[List[str]], ans:dict):
        queue = []
        visited = []
        valid_index = 0
        queue.insert(0, id)
        visited.append(id)
        for l in range(target_level):
            new_q =[]
            while len(queue) != 0:
                k = queue[0]
                #visited.append(k)
                del queue[0]
                for i in friends[k]:
                    if i not in visited:
                        visited.append(i)
                        new_q.append(i)
            queue[:] = new_q
            
            
        for f in queue:
            for v in watchedVideos[f]:
                if v not in ans:
                    ans[v] = 1
                else:
                    ans[v] += 1
        
    def dfs(self, id:int, cur_level:int, target_level:int, friends:List[List[int]], watchedVideos:List[List[str]], ans:dict, visited:List[int]):
        #can't work even if we have visited 
        # A -> B,C
        # B -> C
        # to A, C is level 1
        # but DFS will traverse A->B->C, and think C is level 2
        if id in visited:
            return
        visited.append(id)
        if cur_level == target_level:
            
            for v in watchedVideos[id]:
                if v not in ans:
                    ans[v] = 1
                else:
                    ans[v] += 1
            return
        
        
        for f in friends[id]:
            self.dfs(f, cur_level+1, target_level, friends, watchedVideos, ans, visited)
            
    def watchedVideosByFriends(self, watchedVideos: List[List[str]], friends: List[List[int]], id: int, level: int) -> List[str]:
        ans = {}
        # visited = []
        # self.dfs(id, 0, level, friends, watchedVideos, ans, visited)
        self.bfs(id, level, friends, watchedVideos, ans)
        ret = [k for k, v in sorted(list(ans.items()), key=lambda item: (item[1], item[0]))]
        
        return ret
        

class Solution:
    def watchedVideosByFriends(self, watchedVideos: List[List[str]], friends: List[List[int]], id: int, level: int) -> List[str]:
        
        n = len(watchedVideos)
        
        dist = [9999]*n
        
        queue = [(id, 0)]
        while queue:
            idx, depth = queue.pop(0)
            if depth < dist[idx]:
                dist[idx] = depth
                for c in friends[idx]:
                    queue.append((c, depth+1))
                
        print(dist)
        
        freq = defaultdict(int)
        for i, l in enumerate(dist):
            if l == level:
                for vid in watchedVideos[i]:
                    freq[vid] += 1
        
        pairs = sorted([(v, k) for k,v in list(freq.items())])
        print(pairs)
        
        return [x[-1] for x in pairs]
        

from collections import defaultdict
class Solution:
    def watchedVideosByFriends(self, watchedVideos: List[List[str]], friends: List[List[int]], id: int, level: int) -> List[str]:
        
        graph=defaultdict(list)
        
        for i,friend_list in enumerate(friends):
            for friend in friend_list:
                graph[i].append(friend)
                graph[friend].append(i)           
        
        queue=[id]
        visited=set()
        visited.add(id)
        current_level=0
        videos=defaultdict(lambda:0)
        while queue:
            if current_level==level:
                for friend in queue:
                    for video in watchedVideos[friend]:
                        videos[video]+=1  
                break
            size=len(queue)
            for i in range(size):
                
                node=queue.pop(0)
                
                for friend in graph[node]:
                    if friend not in visited:
                        visited.add(friend)
                        queue.append(friend)
            
            current_level+=1

        answer=sorted(list(videos.keys()),key= lambda x:(videos[x],x))
        return answer
            
                    
                    
            
            
            
            
        
        
            
            
        
        
        
        
        
        
        
        
        
        
        

class Solution:
    def watchedVideosByFriends(self, watchedVideos: List[List[str]], friends: List[List[int]], id: int, level: int) -> List[str]:
        '''
        bfs + heap
        time: nlogn
        space: n
        '''
        def bfs(start):
            queue = [start]
            current_level = 0
            visited = set()
            while queue:
                if current_level >= level:
                    break
                subqueue = queue[:]
                queue = []
                for i in subqueue:
                    visited.add(i)
                    for j in friends[i]:
                        if j not in visited and j not in queue and j not in subqueue:
                            queue.append(j)
                current_level += 1
            return queue
        from collections import Counter
        import heapq
        videos = []
        for i in bfs(id):
            videos += watchedVideos[i]
        freq = dict(Counter(videos))
        heap = []
        for key in freq:
            heapq.heappush(heap, (freq[key], key))
        ans = []
        while heap:
            ans.append(heapq.heappop(heap)[1])
        return ans
from collections import defaultdict
import heapq
class Solution:
    def watchedVideosByFriends(self, watchedVideos: List[List[str]], friends: List[List[int]], id: int, level: int) -> List[str]:
        # (count, c)
        adj = {}
        for i in range(len(friends)):
            adj[i] = friends[i]
        
        queue = [id]
        visited = set()
        visited.add(id)
        dis = 0
        while queue:
            if dis == level:
                break
            for _ in range(len(queue)):
                cur = queue.pop(0)
                for friend in adj[cur]:
                    if friend not in visited:
                        queue.append(friend)
                        visited.add(friend)
            dis += 1
        count = defaultdict(int)
        for people in queue:
            for video in watchedVideos[people]:
                count[video] += 1
        heap = []
        for key in count:
            heapq.heappush(heap, (count[key], key))
        res = []
        while heap:
            res.append(heapq.heappop(heap)[1])
        return res
class Solution:
    def watchedVideosByFriends(self, watchedVideos: List[List[str]], friends: List[List[int]], id: int, level: int) -> List[str]:
        def bfs(start):
            queue = [start]
            current_level = 0
            visited = set()
            while queue:
                if current_level >= level:
                    break
                subqueue = queue[:]
                queue = []
                for i in subqueue:
                    visited.add(i)
                    for j in friends[i]:
                        if j not in visited and j not in queue and j not in subqueue:
                            queue.append(j)
                current_level += 1
            return queue
        from collections import Counter
        import heapq
        videos = []
        for i in bfs(id):
            videos += watchedVideos[i]
        freq = dict(Counter(videos))
        heap = []
        for key in freq:
            heapq.heappush(heap, (freq[key], key))
        ans = []
        while heap:
            ans.append(heapq.heappop(heap)[1])
        return ans
        
                            
                            

from collections import defaultdict


class Movie:
    def __init__(self, name):
        self.name = name
        self.freq = 0
    
    def __lt__(self, other):
        if not other:
            return True
        if self.freq != other.freq:
            return self.freq < other.freq
        
        return self.name < other.name

class Solution:
    def watchedVideosByFriends(self, watchedVideos: List[List[str]], friends: List[List[int]], id: int, level: int) -> List[str]:
        if not watchedVideos or not friends or level == 0:
            return []
        
        
        movies_freq = {}
        queue = deque()
        queue.append(id)
        cur_level = 0
        visited = set()
        visited.add(id)
        while queue:
            n = len(queue)
            if cur_level == level:
                break
            cur_level += 1
            for _ in range(n):
                node = queue.popleft()
                if node >= len(friends):
                    continue
                
                for child in friends[node]:
                    if child in visited:
                        continue
                    visited.add(child)
                    queue.append(child)
        print(queue)
        while queue:
            friend = queue.popleft()
            if friend >= len(watchedVideos):
                continue
            
            for video in watchedVideos[friend]:
                if video not in movies_freq:
                    video_obj = Movie(video)
                    movies_freq[video] = video_obj
                movies_freq[video].freq += 1
        
        if not movies_freq:
            return []
        
        *result, = [x.name for x in sorted(movies_freq.values())]
        return result

class Solution:
    def watchedVideosByFriends(self, watchedVideos: List[List[str]], friends: List[List[int]], id: int, level: int) -> List[str]:
        
        w,f = {},{}
        for i in range(len(friends)):
            w[i] = watchedVideos[i]
            f[i] = friends[i]
        
        cand = []
        frontier = [(id,0)]
        seen ={id}
        while frontier:
            cur,d = frontier.pop()
            if d==level:
                cand.append(cur)
                continue
            
            for friend in f[cur]:
                if friend not in seen:
                    seen.add(friend)
                    frontier.append((friend,d+1))
        
        count = collections.Counter()
        for c in cand:
            for m in w[c]:
                count[m]+=1
        
        
        ans = [(v,k) for k,v in count.items()]
        ans.sort()
        ans = [x[1] for x in ans]
        return ans
from collections import defaultdict, deque


class Solution:
    def watchedVideosByFriends(self, watchedVideos: List[List[str]], friends: List[List[int]], id: int, level: int) -> List[str]:
        visited = set()
        queue = deque([(id, 0)])
        visited.add(id)
        chosenFriends = list()
        while queue:
            person, lv = queue.popleft()
            if lv == level:
                chosenFriends.append(person)
            if lv > level:
                break
            for friend in friends[person]:
                if friend not in visited:
                    queue.append((friend, lv + 1))
                    visited.add(friend)
        videoFrequency = defaultdict(int)
        for friend in chosenFriends:
            for video in watchedVideos[friend]:
                videoFrequency[video] += 1
        return [video[0] for video in sorted(list(videoFrequency.items()), key = lambda x : (x[1], x[0]))]
                

from collections import defaultdict


class Movie:
    def __init__(self, name):
        self.name = name
        self.freq = 0
    
    def __lt__(self, other):
        if not other:
            return True
        if self.freq != other.freq:
            return self.freq < other.freq
        
        return self.name < other.name

class Solution:
    def watchedVideosByFriends(self, watchedVideos: List[List[str]], friends: List[List[int]], id: int, level: int) -> List[str]:
        if not watchedVideos or not friends or level == 0:
            return []
        
        
        movies_freq = {}
        queue = deque()
        queue.append(id)
        cur_level = 0
        visited = set()
        visited.add(id)
        while queue:
            n = len(queue)
            if cur_level == level:
                break
            cur_level += 1
            for _ in range(n):
                node = queue.popleft()
                if node >= len(friends):
                    continue
                
                for child in friends[node]:
                    if child in visited:
                        continue

                    visited.add(child)
                    queue.append(child)

        while queue:
            friend = queue.popleft()
            if friend >= len(watchedVideos):
                continue
            
            for video in watchedVideos[friend]:
                if video not in movies_freq:
                    video_obj = Movie(video)
                    movies_freq[video] = video_obj
                movies_freq[video].freq += 1
        
        if not movies_freq:
            return []
        
        *result, = [x.name for x in sorted(movies_freq.values())]
        return result

from collections import defaultdict


class Movie:
    def __init__(self, name):
        self.name = name
        self.freq = 0
    
    def __lt__(self, other):
        if not other:
            return True
        if self.freq != other.freq:
            return self.freq < other.freq
        
        return self.name < other.name

class Solution:
    def watchedVideosByFriends(self, watchedVideos: List[List[str]], friends: List[List[int]], id: int, level: int) -> List[str]:
        if not watchedVideos or not friends or level == 0:
            return []
        
        
        movies_freq = {}
        queue = deque()
        queue.append(id)
        cur_level = 0
        visited = set()
        visited.add(id)
        while queue:
            n = len(queue)
            if cur_level == level:
                break
            cur_level += 1
            for _ in range(n):
                node = queue.popleft()
                if node >= len(friends):
                    continue
                
                for child in friends[node]:
                    if child in visited:
                        continue
                    visited.add(child)
                    queue.append(child)

        while queue:
            friend = queue.popleft()
            if friend >= len(watchedVideos):
                continue
            
            for video in watchedVideos[friend]:
                if video not in movies_freq:
                    video_obj = Movie(video)
                    movies_freq[video] = video_obj
                movies_freq[video].freq += 1
        
        if not movies_freq:
            return []
        
        *result, = [x.name for x in sorted(movies_freq.values())]
        return result

class Solution:
    def watchedVideosByFriends(self, watchedVideos: List[List[str]], friends: List[List[int]], id: int, level: int) -> List[str]:
        graph = collections.defaultdict(list)
        for u, v in enumerate(friends):
            for i in v:
                graph[u].append(i)
        queue = collections.deque()
        queue.append((id, 0))
        visited = set()
        visited.add(id)
        res = collections.defaultdict(int)
        while queue:
            id, l = queue.popleft()
            if l == level:
                for j in watchedVideos[id]:
                    res[j] += 1
            for v in graph[id]:
                if l+1 <= level and v not in visited:
                    visited.add(v)
                    queue.append((v, l+1))
        from functools import cmp_to_key
        def func(x, y):
            if res[x] > res[y]:
                return -1
            elif res[y] > res[x]:
                return 1
            else:
                if x > y:
                    return -1
                elif y > x:
                    return 1
                else:
                    return 0
        return (sorted(res.keys(), key=cmp_to_key(func)))[::-1]
from collections import Counter
class Solution:
    def watchedVideosByFriends(self, watchedVideos: List[List[str]], friends: List[List[int]], id: int, level: int) -> List[str]:
        current_level = 0
        nodes_in_current_level = [id]
        visited = set(nodes_in_current_level)
        while current_level < level:
            next_level = []
            while len(nodes_in_current_level) > 0:
                node = nodes_in_current_level.pop()

                for neighbor in friends[node]:
                    if neighbor in visited:
                        continue
                    visited.add(neighbor)
                    next_level.append(neighbor)
            nodes_in_current_level = next_level
            current_level += 1
        counter = Counter()
        for id in nodes_in_current_level:
            counter.update(watchedVideos[id])
        return [item for item, count in sorted(list(counter.items()), key=lambda x: (x[1], x[0]))]

class Solution:
    def watchedVideosByFriends(self, watchedVideos: List[List[str]], friends: List[List[int]], id: int, level: int) -> List[str]:
        stack = [(0, id)]
        visited = set([id])
        friendsSet = set()
        while stack:
            lNow, fi = heapq.heappop(stack)
            if lNow == level:
                friendsSet.add(fi)
            else:
                for fi1 in friends[fi]:
                    if fi1 not in visited:
                        visited.add(fi1)
                        heapq.heappush(stack, (lNow+1, fi1))
        videoCounter = Counter()
        for f in friendsSet:
            videoCounter += Counter(watchedVideos[f])
            
        ans = sorted([key for key in videoCounter], key = lambda x: (videoCounter[x],x) )
        # print(videoCounter)
        return ans        
#         friendsSet = set()
#         visited = set()
#         visited.add(id)
#         def findFriends(fi0, l0):
#             if l0 == level:
#                 if fi0 not in visited:
#                     visited.add(fi0)
#                     friendsSet.add(fi0)
#                 return
#             visited.add(fi0)
#             for fi1 in friends[fi0]:
#                 if fi1 not in visited:
#                     findFriends(fi1, l0 + 1)
        
#         findFriends(id, 0)
#         videoCounter = Counter()
#         for f in friendsSet:
#             videoCounter += Counter(watchedVideos[f])
            
#         ans = sorted([key for key in videoCounter], key = lambda x: (videoCounter[x],x) )
#         # print(videoCounter)
#         return ans

class Solution:
    def watchedVideosByFriends(self, watchedVideos: List[List[str]], friends: List[List[int]], id: int, level: int) -> List[str]:
    
        q = deque([id])
        visited = set()
        visited.add(id)
        
        # q contains friends at level k
        for _ in range(level):
            for _ in range(len(q)):
                curr = q.popleft()
                for friend in friends[curr]:
                    if friend not in visited:
                        q.append(friend)
                        visited.add(friend)
                        
        
        # get videos for friends at level k
        video_dict = {}
        for friend in q:
            for video in watchedVideos[friend]:
                video_dict[video] = video_dict.get(video, 0) + 1
              
        res = [k for k, v in sorted(video_dict.items(), key=lambda item: (item[1],item[0]))]
        
        return res
class Solution:
    def watchedVideosByFriends(self, watchedVideos: List[List[str]], friends: List[List[int]], id: int, level: int) -> List[str]:
        # BFS to find the k-related friends
        visited = {id: 1}
        queue = deque()
        queue.append((id, 0))
        result = []
        while queue:
            u, dist = queue.popleft()
            if dist == level:
                result.append(u)
                continue
            
            for v in friends[u]:
                if v in visited:
                    continue
                visited[v] = 1
                queue.append((v, dist + 1))
                
        # collect the movies
        counter = {}
        for u in result:
            for video in watchedVideos[u]:
                counter[video] = counter.get(video, 0) + 1
                
        # sort the movies
        result = sorted([(times, videos) for videos, times in list(counter.items())])
        return [val[1] for val in result]
        

class Solution:
    def watchedVideosByFriends(self, watchedVideos: List[List[str]], friends: List[List[int]], id: int, level: int) -> List[str]:
        
        n = len(watchedVideos)
        visited = [False]*n
        
        que = collections.deque([id])
        visited[id] = True
        cur = 0
        while que and cur<level:
            size = len(que)
            cur += 1
            for _ in range(size):
                node = que.popleft()
                for f in friends[node]:
                    if not visited[f]:
                        visited[f] = True
                        que.append(f)
        
        cnt = collections.Counter()
        for node in que:
            for m in watchedVideos[node]:
                cnt[m] += 1
        
        videos = list(cnt.keys())
        videos.sort(key=lambda x: [cnt[x], x] )
        return videos
        
        

class Solution:
    def watchedVideosByFriends(self, watchedVideos: List[List[str]], friends: List[List[int]], id: int, level: int) -> List[str]:
        visited = set()
        queue = deque([(id, 0)])
        visited.add(id)
        requiredFriends = []
        while queue:
            person, lev = queue.popleft()
            if lev == level:
                requiredFriends.append(person)
            if lev > level:
                break
            for friend in friends[person]:
                if friend not in visited:
                    queue.append((friend, lev + 1))
                    visited.add(friend)
        videosFrequency = defaultdict(int)
        for friend in requiredFriends:
            for video in watchedVideos[friend]:
                videosFrequency[video] += 1
        return [video[0] for video in sorted(videosFrequency.items(), key = lambda x : (x[1], x[0]))]
class Solution:
    def watchedVideosByFriends(self, watchedVideos: List[List[str]], friends: List[List[int]], id: int, level: int) -> List[str]:
        g_friends = collections.defaultdict(list)
        for i in range(len(friends)):
            for f in friends[i]:
                g_friends[i].append(f)
        queue = collections.deque()
        queue.append((id, 0))
        visited = set([id])
        videos = collections.defaultdict(int)
        while queue:
            size = len(queue)
            for _ in range(size):
                curr_id, curr_level = queue.popleft()
                if curr_level == level:
                    for v in watchedVideos[curr_id]:
                        videos[v] += 1
                else:
                    for f in g_friends[curr_id]:
                        if f not in visited:
                            visited.add(f)
                            queue.append((f, curr_level + 1))
        
        return [v for _, v in sorted([(f, v) for v, f in list(videos.items())])]

"""


"""


class Solution:
    def watchedVideosByFriends(self, watchedVideos: List[List[str]], friends: List[List[int]], id: int, level: int) -> List[str]:
        n = len(watchedVideos)
        visited = [0] * n
        visited[id] = 1
        queue = collections.deque()
        queue.append(id)
        persons = []
        step = 0
        while queue:
            size = len(queue)
            step += 1
            for _ in range(size):
                node = queue.popleft()
                for nei in friends[node]:
                    if visited[nei] == 1:
                        continue
                    visited[nei] = 1
                    queue.append(nei)
                    if step == level:
                        persons.append(nei)
            if step == level:
                break
        
        VideoSet = set()
        freq = collections.defaultdict(int)
        for person in persons:
            for v in watchedVideos[person]:
                VideoSet.add(v)
                freq[v] += 1        
        
        temp = []
        for v in VideoSet:
            temp.append([freq[v], v])
        
        temp.sort()
        
        res = []
        for x in temp:
            res.append(x[1])
        return res
        
        
        
        
        
        
        
        
class Solution:
    def watchedVideosByFriends(self, watchedVideos: List[List[str]], friends: List[List[int]], id: int, level: int) -> List[str]:
        #return self.fast(watchedVideos, friends, id, level)
    
        q = deque([id])
        visited = set()
        visited.add(id)
        
        for _ in range(level):
            for _ in range(len(q)):
                curr = q.popleft()
                for friend in friends[curr]:
                    if friend not in visited:
                        q.append(friend)
                        visited.add(friend)
                        
        # q contains friends at level k
        
        # get videos for friends at level k
        video_dict = defaultdict(int)
        for friend in q:
            for video in watchedVideos[friend]:
                video_dict[video] += 1
              
        res = [k for k, v in sorted(list(video_dict.items()), key=lambda item: (item[1],item[0]))]
        
        return res
    
    def fast(self, watchedVideos: List[List[str]], friends: List[List[int]], id: int, level: int):
        res = []
        visited = [False]*len(friends)
        fr_set = set()
        q = deque([id])
        for _ in range(level):
            size = len(q)
            for n in range(size):
                fId = q.popleft()
                for f in friends[fId]:
                    if visited[f]!=True and f !=id: 
                        q.append(f)
                        visited[f] = True
                        
        dic = {}
        while q:
            fid = q.popleft()
            for video in watchedVideos[fid]:
                dic[video] = dic.get(video,0)+1  
                        
                    
        res = [k for k, v in sorted(list(dic.items()), key=lambda item: (item[1],item[0]))]

        return res;

class Solution:
    def watchedVideosByFriends(self, watchedVideos: List[List[str]], friends: List[List[int]], id: int, level: int) -> List[str]:
        
        graph = defaultdict(list)
        
        #build graph
        for i, v in enumerate(friends):
            graph[i].extend(v)
            for r in v:
                graph[r].append(i)
        
        
        
        videos = defaultdict(int)
        def bfs(source):
            dq = deque([source])
            visited = set()
            visited.add(source)
            nlevels = 0
            
            while dq:
                for _ in range(len(dq)):
                    node = dq.popleft()
                    if nlevels == level:
                        for v in watchedVideos[node]:
                            videos[v] += 1
                    for nei in graph[node]:
                        if nei not in visited:
                            visited.add(nei)
                            dq.append(nei)
                    
                nlevels += 1
        bfs(id)            
        minheap = []
        for v in sorted(videos.keys()):
            heappush(minheap, (videos[v], v)) 

        res = []
        while minheap:
            res.append(heappop(minheap)[1])
        return res
        

class Solution:
    def watchedVideosByFriends(self, watchedVideos: List[List[str]], friends: List[List[int]], id: int, level: int) -> List[str]:
        n = len(friends)
        friend_dist = [math.inf] * n
        
        friend_dist[id] = 0
        deque = collections.deque()
        deque.append(id)
        
        watched = dict()
        
        while deque:
            person = deque.popleft()
            
            if friend_dist[person] == level:
                for video in watchedVideos[person]:
                    watched[video] = watched.get(video,0) +1
                continue

            for friend in friends[person]:
                if friend_dist[friend] == math.inf:
                    friend_dist[friend] = friend_dist[person] + 1
                    deque.append(friend)
        
        return sorted(watched.keys(), key=lambda v: (watched[v],v))
class Solution:
    def watchedVideosByFriends(self, watchedVideos: List[List[str]], friends: List[List[int]], id: int, level: int) -> List[str]:
        stack = deque([id])
        seen = set()
        seen.add(id)
        for _ in range(level):
            for i in range(len(stack)):
                f = stack.popleft()
                for j in friends[f]:
                    if j not in seen:
                        stack.append(j)
                        seen.add(j)
        
        count = Counter()
        for lf in stack:
            count += Counter(watchedVideos[lf])
        return [x[0] for x in sorted(count.items(), key=lambda item: (item[1], item[0]))]
class Solution:
    def watchedVideosByFriends(self, watchedVideos: List[List[str]], friends: List[List[int]], ID: int, level: int) -> List[str]:
        # friends is an adjacency list: person -> list<friends of person>
        # watchedvideos: person -> list of videos
        
        # find subset of friends with provided level
        freqs = dict()
        visited = set()
        visited.add(ID)
        q = set()
        q.add(ID)
        for _ in range(level):
            q = {j for i in q for j in friends[i] if j not in visited}
            visited |= q
        for p in q:
            for v in watchedVideos[p]:
                freqs[v] = freqs.get(v, 0) + 1
            

        sortedfreqs = sorted([(n, v) for v, n in list(freqs.items())])
        
        return [v for _, v in sortedfreqs]
            
        
            
        

class Solution:
    def watchedVideosByFriends(self, watchedVideos: List[List[str]], friends: List[List[int]], id: int, level: int) -> List[str]:
        visited = set()
        visited.add(id)
        frequency = {}
        dictionary = {}
        q = deque()
        q.append(id)
        running_level = 1
        solution = []
        
        while q and running_level <= level:
            length = len(q)
            
            for index in range(length):
                node = q.popleft()
                
                for friend in friends[node]:
                    if friend in visited: continue
                    visited.add(friend)

                    if running_level == level:
                        for movie in watchedVideos[friend]:
                            frequency[movie] = frequency.get(movie, 0)  + 1
                    else: q.append(friend)
            
            running_level+=1
        
        
        order = set(sorted(frequency.values()))
        
        for key in frequency:
            dictionary[frequency[key]] = dictionary.get(frequency[key], []) + [key]
        
        for rank in order:
            solution.extend(sorted(dictionary[rank]))

        return solution
class Solution:
    def watchedVideosByFriends(self, watchedVideos: List[List[str]], friends: List[List[int]], id: int, level: int) -> List[str]:
        n=len(watchedVideos)
        a,d,w=[id],[1]*n,{}
        d[id]=0
        for _ in range(level):
            b=[]
            for i in a:
                for j in friends[i]:
                    if d[j]:
                        b.append(j)
                        d[j]=0
            a=b
        for i in a:
            for x in watchedVideos[i]: w[x]=w.get(x,0)+1
        return sorted(w.keys(),key=lambda x:(w[x],x))
from collections import deque, Counter

class Solution:
    def watchedVideosByFriends(self, watchedVideos: List[List[str]], friends: List[List[int]], id: int, level: int) -> List[str]:
        
        # BFS search
        persons = len(friends)
        visited = [False]*persons
        source = id
        visited[source] = True
        
        queue = deque()
        queue.append(source)
        k = 0
        
        videos_counter = Counter()
        
        while queue and k < level:
            
            videos_counter = Counter()
            q_len = len(queue)
            for _ in range(q_len):
                person = queue.popleft()
                
                # search all the watched videos of your friend
                for friend in friends[person]:
                    if visited[friend]:
                        continue
                    
                    visited[friend] = True
                    queue.append(friend)
                    videos = watchedVideos[friend]
                    # update the videos_counter
                    for video in videos:
                        videos_counter[video] += 1

            
            k += 1  # update the level
        
        # collect the videos with their frequencies
        frequencies = [(frequency, video) for video, frequency in list(videos_counter.items())]
        
        # sort them in increasing order with respect to frequency
        result = [video for freq, video in sorted(frequencies)]
        return result

class Solution:
    def watchedVideosByFriends(self, videos: List[List[str]], friends: List[List[int]], me: int, level: int) -> List[str]:
        visit = set()
        queue, arr = [me], []
        while level:
            level -= 1
            size = len(queue)
            for i in range(size):
                curr = queue.pop(0)
                if curr not in visit:
                    visit.add(curr)
                    for f in friends[curr]:
                        queue.append(f)
        v = []
        for i in queue:
            if i not in visit:
                visit.add(i)
                v += videos[i]
        c = collections.Counter(v)
        return sorted(sorted(list(set(v))), key=lambda x:c[x])
class Solution:
    def watchedVideosByFriends(self, watchedVideos: List[List[str]], friends: List[List[int]], id: int, level: int) -> List[str]:
        friends = getFriends(friends, id, 1, level, set([id]))
        videoCounts = collections.Counter([val for i in friends for val in watchedVideos[i]])
        sortedCounts = sorted(list(videoCounts.items()), key=lambda video: (video[1], video[0]))
        return [count[0] for count in sortedCounts]
        
        
        
        
def getFriends(friends: List[List[int]], index: int, level: int, targetLevel: int, knownFriends):
    currentFriends = set(friends[index]) - knownFriends
    if (level == targetLevel):
        return currentFriends
    else:
        newKnownFriends = knownFriends | currentFriends
        return set([val for i in currentFriends for val in getFriends(friends, i, level + 1, targetLevel, newKnownFriends)])

from collections import deque, defaultdict
class Solution:
    def watchedVideosByFriends(self, watchedVideos: List[List[str]], friends: List[List[int]], id: int, level: int) -> List[str]:
        stack = deque()
        nbr = dict()
        stack.append(id)
        seen = set()
        seen.add(id)
        lvl = 0
        while stack:
            n = len(stack)
            lvl += 1
            if lvl > level:
                break
            for i in range(n):
                person_id = stack.pop()
                for p in friends[person_id]:
                    if p in seen:
                        continue
                    seen.add(p)
                    stack.append(p)
                    if p not in nbr:
                        nbr[p] = lvl

                        
        res = defaultdict(int)
        for person_id in nbr:
            if nbr[person_id] == level:
                for video in watchedVideos[person_id]:
                    res[video] += 1
        res_sorted = sorted(list(res.items()), key = lambda x: (x[1], x[0]))
        return [x[0] for x in res_sorted]
                    

from collections import Counter 
class Solution:
    def watchedVideosByFriends(self, watchedVideos: List[List[str]], friends: List[List[int]], id: int, level: int) -> List[str]:
        friendsSeen = {id}
        k = 0
        
        connections = {id}
        for k in range(0, level):
            nextDegree = [friends[i] for i in connections]
            connections = set([friend for friends in nextDegree for friend in friends if friend not in friendsSeen])
            friendsSeen = friendsSeen.union(connections)
            
        videosWatched = [video for friend in connections for video in watchedVideos[friend]]
        videoCounter = dict(Counter(videosWatched))
        result = sorted(videoCounter.keys(), key = lambda x: videoCounter[x])
        freqUnique = [videoCounter[video] for video in result]
        freqMovieDict = dict()
        for i in range(0, len(freqUnique)):
            if freqUnique[i] not in freqMovieDict:
                freqMovieDict[freqUnique[i]] = [result[i]]
            else:
                freqMovieDict[freqUnique[i]] += [result[i]]
        for key in freqMovieDict:
            freqMovieDict[key] = sorted(freqMovieDict[key])
        ans = []
        for key in sorted(freqMovieDict.keys()):
            ans += freqMovieDict[key]
        return ans
class Solution:
    def watchedVideosByFriends(self, watchedVideos: List[List[str]], friends: List[List[int]], id: int, level: int) -> List[str]:
        # BFS to find the k-related friends
        visited = {id: 1}
        queue = deque()
        queue.append((id, 0))
        result = []
        while queue:
            u, dist = queue.popleft()
            if dist == level:
                result.append(u)

            if dist > level:
                break
                
            for v in friends[u]:
                if v in visited:
                    continue
                
                queue.append((v, dist + 1))
                visited[v] = 1
                
        # collect the movies
        counter = {}
        for u in result:
            for video in watchedVideos[u]:
                counter[video] = counter.get(video, 0) + 1
                
        # sort the movies
        result = sorted([(times, videos) for videos, times in list(counter.items())])
        return [val[1] for val in result]
        

from collections import Counter, deque
from typing import List

class Solution:
    def watchedVideosByFriends(self, watchedVideos: List[List[str]], friends: List[List[int]], id: int, level: int) -> List[str]:
        q = deque([(0, id)])
        seen = [False] * len(friends)
        d = dict()

        while q:
            stage, person = q.popleft()
            seen[person] = True
            if stage not in d: d[stage] = set()

            d[stage].add(person)

            for friend in friends[person]:
                if seen[friend]: continue
                seen[friend] = True
                q.append((stage + 1, friend))
        
        shared = dict()
        for friend in d[level]:
            for video in watchedVideos[friend]:
                if video in shared:
                    shared[video] += 1
                else:
                    shared[video] = 1
        return list(x for _, x in sorted((freq, val) for val, freq in shared.items()))
class Solution:
    def watchedVideosByFriends(self, watchedVideos: List[List[str]], friends: List[List[int]], id: int, level: int) -> List[str]:
        k_friends = self.findFriends(friends, [id], {id}, level)
        watched = {}
        for id in k_friends:
            for vid in watchedVideos[id]:
                if vid in watched:
                    watched[vid] += 1
                else:
                    watched[vid] = 1
            
        out = []
        for key in watched:
            out.append((watched[key], key))
        out.sort()
        
        return list([x[1] for x in out])
    
    def findFriends(self, friends, ids, seen, level):
        if level == 0 or len(ids) == 0:
            return ids
        
        new = []
        for id in ids:
            for friend in friends[id]:
                if friend not in seen:
                    seen.add(friend)
                    new.append(friend)
                    
        return self.findFriends(friends, new, seen, level - 1)
                

class Solution:
    def watchedVideosByFriends(self, watchedVideos: List[List[str]], friends: List[List[int]], id: int, level: int) -> List[str]:
        bfs,visited={id},{id}
        for _ in range(level):
            bfs={j for i in bfs for j in friends[i] if j not in visited}
            visited|=bfs
        freq=collections.Counter([v for idx in bfs for v in watchedVideos[idx]])
        return sorted(freq.keys(),key=lambda x:(freq[x],x))
from collections import deque, Counter

class Solution:
    def watchedVideosByFriends(self, watchedVideos: List[List[str]], friends: List[List[int]], id: int, level: int) -> List[str]:
        
        n = len(friends)
            
        stack = deque([id])
        visited = set([id])
        
        for _ in range(level):
            for j in range(len(stack)): 
                cur = stack.popleft()
                for nxt in friends[cur]:
                    if nxt not in visited:
                        visited.add(nxt)
                        stack.append(nxt)
                        
        dic = Counter()
        for ppl in stack:
            dic.update(watchedVideos[ppl])

        items = sorted(list(dic.items()), key=lambda x:(x[1],x[0]))
        return [k for k, _ in items]

class Solution:
    def watchedVideosByFriends(self, watchedVideos: List[List[str]], friends: List[List[int]], id: int, level: int) -> List[str]:
        
        q = deque([id])
        visited = set()
        visited.add(id)
        
        while level > 0:
            level -= 1
            for _ in range(len(q)):
                curr = q.popleft()
                for friend in friends[curr]:
                    if friend not in visited:
                        q.append(friend)
                        visited.add(friend)
                        
        # q contains friends at level k
        
        # get videos for friends at level k
        video_dict = defaultdict(int)
        for friend in q:
            for video in watchedVideos[friend]:
                video_dict[video] += 1
              
        res = [k for k, v in sorted(list(video_dict.items()), key=lambda item: (item[1],item[0]))]
        
        return res
    
    def fast():
        res = []
        visited = [False]*len(friends)
        fr_set = set()
        q = deque([id])
        for _ in range(level):
            size = len(q)
            for n in range(size):
                fId = q.popleft()
                for f in friends[fId]:
                    if visited[f]!=True and f !=id: 
                        q.append(f)
                        visited[f] = True
                        
        dic = {}
        while q:
            fid = q.popleft()
            for video in watchedVideos[fid]:
                dic[video] = dic.get(video,0)+1  
                        
                    
        res = [k for k, v in sorted(list(dic.items()), key=lambda item: (item[1],item[0]))]

        return res;

from collections import defaultdict, deque
class Solution:
    def watchedVideosByFriends(self, watchedVideos: List[List[str]], friends: List[List[int]], id: int, level: int) -> List[str]:
        n=len(friends)
        adj=[[] for i in range(n)]
        for i in range(n):
            adj[i]=friends[i]
        req=[]
        visited={}
        q=deque()
        q.append((id,0))
        visited[id]=True
        while q:
            u,lev=q.popleft()
            if lev==level:
                req.append(u)
                continue
            if lev>level:
                break
            for f in adj[u]:
                if f not in visited:
                    q.append((f,lev+1))
                    visited[f]=True
        res=defaultdict(lambda:0)
        for  f in req:
            for mov in watchedVideos[f]:
                res[mov]+=1
        res = sorted(res.items(),key = lambda x:(x[1], x[0]))
        movies = [i[0] for i in res]
        return movies
class Solution:
    def watchedVideosByFriends(self, watchedVideos: List[List[str]], friends: List[List[int]], id: int, level: int) -> List[str]:
        
        queue = deque()
        queue.append([id, 0])
        ans = defaultdict(int)
        visited = {id:1}
        while(len(queue) > 0):
            now, l = queue.popleft()
            
            if l == level:
                for v in watchedVideos[now]:
                    ans[v] += 1
            
                continue
            
            for friend in friends[now]:
                if friend not in visited:
                    queue.append([friend, l+1])
                    visited[friend] = 1
        
        ansid = sorted(ans.items(), key=lambda x: (x[1], x[0]))
        
        ansid = [x for x,_ in ansid]
        return ansid
import collections

class Solution:
    def watchedVideosByFriends(self, watchedVideos: List[List[str]], friends: List[List[int]], id: int, level: int) -> List[str]:
        # First: walk friends and summarize k-level info
        krem = set(range(len(friends)))
        krem.remove(id)
        
        curlevel = 0
        klevel = set([id])
        while True:
            #print("level: {} members: {}".format(curlevel, klevel))
            if curlevel == level:
                break
            
            newlevel = set()
            
            # We will only walk each 'pid' exactly once
            for pid in klevel:
                # Therefore we will only walk friends-of-pid exactly once
                for fid in friends[pid]:
                    # Not shortest path to 'fid'
                    if not fid in krem:
                        continue
                
                    krem.remove(fid)
                    newlevel.add(fid)
            
            klevel = newlevel
            curlevel += 1
        
        krem = None
        
        # klevel now is the set of ids reachable at shortest-path 'k' from 'id'
        # So we just need to identify the videos.
        vids = collections.defaultdict(int)
        for pid in klevel:
            for vid in watchedVideos[pid]:
                vids[vid] += 1
        
        return [x[0] for x in sorted(vids.items(), key=lambda x: (x[1], x[0]))]
class Solution:
    def createAdjMatrix(self,friends):
        adj_mat = {}
        for i in range(len(friends)):
            adj_mat[i] = friends[i]
        return adj_mat
    
    def calculateFreq(self,level_friend,watchedVideos):
        movies_freq = {}
        for friend in level_friend:
            movies = watchedVideos[friend]
            for movie in movies:
                movies_freq[movie] = movies_freq.get(movie,0)+1
                
        movies_freq = sorted(movies_freq.items(),key = lambda x:(x[1], x[0]))
        movies = [i[0] for i in movies_freq]
        return movies
    
    def watchedVideosByFriends(self, watchedVideos: List[List[str]], friends: List[List[int]], id: int, level: int) -> List[str]:
        
        adj_mat = self.createAdjMatrix(friends)
        visited = set()
        Q = [(id,0)]
        visited.add(id)
        level_friend = []
        while Q and level:
            id_,lev = Q.pop(0)
            if lev==level:
                level_friend.append(id_)
            if lev>level:
                break
            friends = adj_mat[id_]
            for friend in friends:
                if friend not in visited:
                    Q.append((friend,lev+1))
                    visited.add(friend)
                    
        return self.calculateFreq(level_friend,watchedVideos)
class Solution:
    def watchedVideosByFriends(self, watchedVideos: List[List[str]], friends: List[List[int]], id: int, level: int) -> List[str]:
        graph = collections.defaultdict(set)
        
        for i in range(len(friends)):
            for friend in friends[i]:
                graph[i].add(friend)
                graph[friend].add(i)
        
        levelOfFriends = [id]
        visited = set([id])
        l = 0

        while levelOfFriends and l < level:
            size = len(levelOfFriends)
            l += 1
            
            for _ in range(size):
                f = levelOfFriends.pop(0)
                
                for otherF in graph[f]:
                    if otherF not in visited:
                        levelOfFriends.append(otherF)
                        visited.add(otherF)
                    
        w = collections.defaultdict(int)
        
        for f in levelOfFriends:
            videos = watchedVideos[f]
            
            for v in videos:
                w[v] += 1
                
        heap = []
        
        for v in list(w.keys()):
            heapq.heappush(heap, (w[v], v))
            
        res = []
        for _ in range(len(heap)):
            _, v = heapq.heappop(heap)
            res.append(v)
            
        return res
                

class Solution:
    def watchedVideosByFriends(self, watchedVideos: List[List[str]], friends: List[List[int]], id: int, level: int) -> List[str]:
        vis={id}
        q=deque()
        q.append(id)
        while q and level:
            ln=len(q)
            for _ in range(ln):
                x=q.popleft()
                #print(x)
                for i in friends[x]:
                    if i not in vis: 
                        q.append(i)
                        vis.add(i)
            level-=1
        #print(q)
        d=defaultdict(int)
        for i in q:
            for j in watchedVideos[i]:
                d[j]+=1
        '''d2=defaultdict(list)
        for i in d:
            d2[d[i]].append(i)
        for i in d2: d2[i].sort()
        x=sorted(list(d2.keys()))
        res=[]
        for i in x: 
            for j in d2[i]: res.append(j)
        return res'''
    
        pq=[]
        res=[]
        for i in d:
            heapq.heappush(pq,(d[i],i))
        while pq:
            res.append(heapq.heappop(pq)[1])
        return res
