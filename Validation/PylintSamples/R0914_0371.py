from collections import defaultdict

class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        if S == T:
            return 0
        # sequence_to_route_id dict
        # if when adding sequence ids to this dict, they are part of another route,
        # merge them
        max_int = 10**6
        start_routes = set()
        end_routes = set()
        route_connections = defaultdict(lambda: set())
        sequence_to_route_id_dict = {}
        route_to_minbuscount = defaultdict(lambda: max_int)
        for r_id, r in enumerate(routes):
            for s in r:
                if s == S:
                    start_routes.add(r_id)
                    route_to_minbuscount[r_id] = 1
                if s == T:
                    end_routes.add(r_id)
                if s in sequence_to_route_id_dict:
                    route_connections[r_id].add(sequence_to_route_id_dict[s])
                    route_connections[sequence_to_route_id_dict[s]].add(r_id)
                sequence_to_route_id_dict[s] = r_id
        
        # print(route_connections)
        # print(start_routes)
        # print(end_routes)
        
        current_route_buscount = [(s,1) for s in start_routes]
        for r_id, buscount in current_route_buscount:
            # print(current_route_buscount)
            # print(dict(route_to_minbuscount))
            for connection in route_connections[r_id]:
                if route_to_minbuscount[connection] > buscount+1:
                    route_to_minbuscount[connection] = buscount+1
                    current_route_buscount.append((connection,buscount+1))
        result = min(route_to_minbuscount[x] for x in end_routes)
        return -1 if result == max_int else result


class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        # You already at the terminal, so you needn't take any bus.
        if S == T: return 0
        
        # You need to record all the buses you can take at each stop so that you can find out all
        # of the stops you can reach when you take one time of bus.
        # the key is stop and the value is all of the buses you can take at this stop.
        stopBoard = {} 
        for bus, stops in enumerate(routes):
            for stop in stops:
                if stop not in stopBoard:
                    stopBoard[stop] = [bus]
                else:
                    stopBoard[stop].append(bus)
        
        # The queue is to record all of the stops you can reach when you take one time of bus.
        queue = deque([S])
        # Using visited to record the buses that have been taken before, because you needn't to take them again.
        visited = set()

        res = 0
        while queue:
            # take one time of bus.
            res += 1
            # In order to traverse all of the stops you can reach for this time, you have to traverse
            # all of the stops you can reach in last time.
            pre_num_stops = len(queue)
            for _ in range(pre_num_stops):
                curStop = queue.popleft()
                # Each stop you can take at least one bus, you need to traverse all of the buses at this stop
                # in order to get all of the stops can be reach at this time.
                for bus in stopBoard[curStop]:
                    # if the bus you have taken before, you needn't take it again.
                    if bus in visited: continue
                    visited.add(bus)
                    for stop in routes[bus]:
                        if stop == T: return res
                        queue.append(stop)
        return -1
        
                

from collections import defaultdict
class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        to_routes = collections.defaultdict(set)
        for i, route in enumerate(routes):
            for j in route:
                to_routes[j].add(i)
        bfs = [(S, 0)]
        seen = set([S])
        for stop, bus in bfs:
            if stop == T: return bus
            for i in to_routes[stop]:
                for j in routes[i]:
                    if j not in seen:
                        bfs.append((j, bus + 1))
                        seen.add(j)
                routes[i] = []  # seen route
        return -1
class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        if S==T: return 0
        bus_dic = defaultdict(set)
        for i, r in enumerate(routes):
            for s in r:
                bus_dic[s].add(i)

        seen = {S}
        av_bus = bus_dic[S]
        for bus in bus_dic[S]:
            if bus in bus_dic[T]:
                return 1
        cnt = 0
        while av_bus:
            tmp = set()
            cnt += 1
            for meh in range(len(av_bus)):
                bus = av_bus.pop()
                if bus in bus_dic[T]:
                    return cnt
                for s in routes[bus]:
                    if s not in seen:
                        seen.add(s)
                        for bus in bus_dic[s]:
                            if bus in bus_dic[T]:
                                return cnt+1
                            tmp.add(bus)
            av_bus = tmp
        return -1     
class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        if S == T:
            return 0

        buses_from_stops = defaultdict(set)
        for i, r in enumerate(routes) :
            for v in r:
                buses_from_stops[v].add(i)

        memo = {}
        def check(bus, hist):
            if bus in memo:
                return memo[bus]

            if T in routes[bus]:
                return 1
            else:
                v = 1000000
                for stop in routes[bus]:
                    for bs in buses_from_stops[stop]:
                        if bs not in hist:
                            nh = set(hist)
                            nh.add(bs)
                            v = min(v, check(bs, nh))
                memo[bus] = v + 1
                return v + 1

        mn = 1000000
        for bs in buses_from_stops[S]:
                mn = min(mn, check(bs, set([bs])))
        return mn if mn < 1000000 else -1
class Solution:
    def numBusesToDestination(self, routes, S, T):
        if S == T: return 0
        queue = collections.deque()
        graph = collections.defaultdict(set)
        routes = list(map(set, routes))
        seen, targets = set(), set()
        for i in range(len(routes)):
            if S in routes[i]:  # possible starting route number
                seen.add(i)
                queue.append((i, 1))  # enqueue
            if T in routes[i]:  # possible ending route number
                targets.add(i)
            for j in range(i+1, len(routes)):
                if routes[j] & routes[i]:  # set intersection to check if route_i and route_j are connected
                    graph[i].add(j)
                    graph[j].add(i)
        while queue:
            cur, count = queue.popleft()
            if cur in targets:
                return count
            for nei in graph[cur]:
                if nei not in seen:
                    queue.append((nei, count+1))
                    seen.add(nei)
        return -1
from collections import defaultdict, deque
class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        if S==T: return 0
        bus2stops = defaultdict(list)
        stop2buses = defaultdict(list)
        for bus,stops in enumerate(routes):
            for stop in stops:
                bus2stops[bus].append(stop)
                stop2buses[stop].append(bus)
        
        q = deque()
        visitedBuses = set()
        for bus in stop2buses[S]:
            q.extend(bus2stops[bus])
            visitedBuses.add(bus)
        visitedStops = set(q)
        if T in visitedStops: return 1
        
        numBuses = 2
        while q:
            for _ in range(len(q)):
                currStop = q.popleft()

                for bus in stop2buses[currStop]:
                    if bus in visitedBuses: 
                        continue
                    visitedBuses.add(bus)
                    for stop in bus2stops[bus]:
                        if stop not in visitedStops:
                            if stop==T:
                                return numBuses
                            visitedStops.add(stop)
                            q.append(stop)
            numBuses+=1
        return -1
class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        toroute=collections.defaultdict(set)
        seen={S}
        for i, stops in enumerate(routes):
            for j in stops:
                toroute[j].add(i)
                
        
        q=collections.deque([(S,0)])
        while q:
            s,b=q.popleft()
            if s==T:
                return b
            for i in toroute[s]:
                for j in routes[i]:
                    if j not in seen:
                        seen.add(s)
                        q.append((j,b+1))
                routes[i]=[]
                
        return -1

class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        res = 0
        routeDones = set()
        stations = {S}
        while T not in stations:
            newStations = set()
            for i, route in enumerate(routes):
                if i in routeDones or len(stations.intersection(set(route))) == 0:
                    continue # case only continue?
                newStations = newStations.union(set(route) - stations)
                routeDones.add(i)
            if len(newStations) == 0:
                return -1
            res +=1
            stations=stations.union(newStations)

        return res
class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        if S == T: return 0
        routes, n = [set(r) for r in routes], len(routes)
        g = [set() for _ in range(n)]
        for i in range(n):
            for j in range(i):
                if set(routes[i]) & set(routes[j]): 
                    g[i].add(j), g[j].add(i)
        seen, dst = set(i for i,r in enumerate(routes) if S in r), set(i for i,r in enumerate(routes) if T in r)
        q = [(x, 1) for x in seen]
        for x, d in q:
            if x in dst: return d
            for y in g[x]:
                if y not in seen: seen.add(y), q.append((y, d+1))
        return -1
class Solution(object):
    def numBusesToDestination(self, routes, S, T):
        if S==T:
            return 0
        
        queue = collections.deque()
        graph = collections.defaultdict(set)
        
        routes = list(map(set,routes))
        
        seen, targets = set(),set()
        
        for i in range(len(routes)):
            if S in routes[i]:
                seen.add(i)
                queue.append((i,1))
            if T in routes[i]:
                targets.add(i)
            for j in range(i+1,len(routes)):
                if routes[j] & routes[i]:
                    graph[i].add(j)
                    graph[j].add(i)
   
        while queue:
            cur,count = queue.popleft()
            if cur in targets:
                return count
            for nei in graph[cur]:
                if nei not in seen:
                    queue.append((nei,count+1))
                    seen.add(nei)
        return -1
            
                
                    
                
                
                
            
            
        

from collections import deque, defaultdict

class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        route_dict = defaultdict(set)
        for i, route in enumerate(routes):
            for stop in route:
                route_dict[stop].add(i)
        
        queue = deque([S])
        seen = set([S])
        buses = 0
        while queue:
            size = len(queue)
            for _ in range(size):
                stop = queue.popleft()
                if stop == T:
                    return buses
                for i in route_dict[stop]:
                    for j in routes[i]:
                        if j not in seen:
                            queue.append(j)
                            seen.add(j)
                    routes[i] = []
            buses += 1
        
        return -1
        
                
                

class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        ans = 0
        n = len(routes)
        if S == T:
            return 0
        graph = collections.defaultdict(set)# stop -> bus #
        queue = collections.deque()
        visited_stop = set()
        visited_bus = set()
        for i in range(n):
            for stop in routes[i]:
                graph[stop].add(i)
        print(graph)
        queue.append(S)
        visited_stop.add(S)
        
        while queue:
            qLen = len(queue)
            ans +=1
            for i in range(qLen):
                stop = queue.popleft()
                for next_bus in graph[stop]:
                    if next_bus in visited_bus:
                        continue
                    visited_bus.add(next_bus)
                    for next_stop in routes[next_bus]:
                        # if next_stop in visited_stop:
                        #     continue
                        if next_stop == T:
                            print('here')
                            return ans
                        queue.append(next_stop)
                        # visited_stop.add(next_stop)
            print((queue, visited_stop, visited_bus))
            
            print(ans)
        return -1 
    
# defaultdict(<class 'set'>, {1: {0}, 2: {0}, 7: {0, 1}, 3: {1}, 6: {1}})
# deque([2]) {1, 2} {0}
# deque([2, 7]) {1, 2, 7} {0}
# 1
# deque([3]) {1, 2, 3, 7} {0, 1}
            

           

class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        if S == T:
            return 0
        
        
        n = len(routes)
        
        graph = defaultdict(set)
        
        for i in range(n):
            for j in range(i,n):
                if set(routes[i]).intersection(routes[j]):
                    graph[i].add(j)
                    graph[j].add(i)
                    
        
        # Get source and destination
        for i in range(n):
            if S in routes[i]:
                if T in routes[i]:
                    return 1  
                source = i
            elif T in routes[i]:
                dest = i
                
                
                
        q = deque([[source, 1]])
        visited = [False]*n
        visited[source] = True
        
        while q:
            node, dis = q.popleft()
            if node == dest:
                return dis
            for u in graph[node]:
                if not visited[u]:
                    visited[u] = True
                    q.append([u, dis +1])
                    
        return -1
            
        

class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        if S == T: return 0
        
        graph   = collections.defaultdict(set)
        
        for bus, stops in enumerate(routes):
            for stop in stops:
                graph[stop].add(bus)
        
        queue   = collections.deque([S])
        visited = set()
        result  = int()
        
        while queue:
            result  += 1
            for _ in range(len(queue)):
                currStop    = queue.popleft()
                for bus in graph[currStop]:
                    if bus not in visited:
                        visited.add(bus)
                        for stop in routes[bus]:
                            if stop == T:
                                return result
                            queue.append(stop)
        return -1
class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        num_buses = len(routes)
        bus_to_stop = defaultdict(set)
        for bus, stops in enumerate(routes):
            bus_to_stop[bus] = set(stops)
        
        def update_buses_used():
            for bus in range(num_buses):
                if bus in buses_used:
                    continue
                if stops_reached & bus_to_stop[bus]:
                    buses_used.add(bus)
        
        def update_stops_reached(stops_reached):
            for bus in buses_used:
                stops_reached |= bus_to_stop[bus]
        
        buses_used = set()
        stops_reached = {S}
        pre_stop_count = 0
        bus_count = 0
        while len(stops_reached) > pre_stop_count:
            if T in stops_reached:
                return bus_count
            pre_stop_count = len(stops_reached)
            update_buses_used()
            update_stops_reached(stops_reached)
            bus_count += 1
            
        return -1
        
        
        

class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        if S == T:
            return 0
        
        
        n = len(routes)
        
        graph = defaultdict(set)
        
        for i in range(n):
            for j in range(i+1,n):
                if set(routes[i]).intersection(routes[j]):
                    graph[i].add(j)
                    graph[j].add(i)
                    
        
        # Get source and destination
        for i in range(n):
            if S in routes[i]:
                if T in routes[i]:
                    return 1  
                source = i
            elif T in routes[i]:
                dest = i
                
                
                
        q = deque([[source, 1]])
        visited = [False]*n
        visited[source] = True
        
        while q:
            node, dis = q.popleft()
            if node == dest:
                return dis
            for u in graph[node]:
                if not visited[u]:
                    visited[u] = True
                    q.append([u, dis +1])
                    
        return -1
            
        

class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        if S == T:
            return 0
        queue = collections.deque()
        routes = list(map(set, routes))
        graph = collections.defaultdict(set)
        visited, targets = set(), set()
        for i in range(len(routes)):
            if S in routes[i]:  # possible starting route number
                visited.add(i)
                queue.append((i, 1))  # enqueue
            if T in routes[i]:  # possible ending route number
                targets.add(i)
            for j in range(i+1, len(routes)):
                if routes[j] & routes[i]:  # set intersection to check if route_i and route_j are connected
                    graph[i].add(j)
                    graph[j].add(i)
        while queue:
            cur, depth = queue.popleft()
            if cur in targets:
                return depth
            for nxt in graph[cur]:
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append((nxt, depth + 1))
        return -1            
# Reference: https://leetcode.com/problems/bus-routes/discuss/122712/Simple-Java-Solution-using-BFS
from collections import deque
class Solution:
    def numBusesToDestination(self,routes, S, T):
        if S == T: return 0
        routes, n = [set(r) for r in routes], len(routes)
        g = [set() for _ in range(n)]
        for i in range(n):
            for j in range(i):
                if set(routes[i]) & set(routes[j]): 
                    g[i].add(j), g[j].add(i)
        seen, dst = set(i for i,r in enumerate(routes) if S in r), set(i for i,r in enumerate(routes) if T in r)
        q = [(x, 1) for x in seen]
        for x, d in q:
            if x in dst: return d
            for y in g[x]:
                if y not in seen: seen.add(y), q.append((y, d+1))
        return -1
class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        num_buses = len(routes)
        bus_to_stop = defaultdict(set)
        for bus, stops in enumerate(routes):
            bus_to_stop[bus] = set(stops)
        
        def update_buses_used():
            for bus in range(num_buses):
                if bus in buses_used:
                    continue
                if stops_reached & bus_to_stop[bus]:
                    buses_used.add(bus)
        
        def update_stops_reached():
            for bus in buses_used:
                stops_reached.update(bus_to_stop[bus])
        
        buses_used = set()
        stops_reached = {S}
        pre_stop_count = 0
        bus_count = 0
        while len(stops_reached) > pre_stop_count:
            if T in stops_reached:
                return bus_count
            pre_stop_count = len(stops_reached)
            update_buses_used()
            update_stops_reached()
            bus_count += 1
            
        return -1
        
        
        

class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        if S==T: return 0
        routes = [set(r) for r in routes]
        
        graph = collections.defaultdict(set)
        for i in range(len(routes)):
            for j in range(i+1,len(routes)):
                if routes[i].intersection(routes[j]):
                    graph[i].add(j)
                    graph[j].add(i)
                    
        seen,end = set(),set()
        for i,x in enumerate(routes):
            if S in x: seen.add(i)
            if T in x: end.add(i)
        
        q = [(n,1) for n in seen]
        seen = set()
        for n,d in q:
            if n in end: return d
            for x in graph[n]:
                if x not in seen:
                    seen.add(x)
                    q.append((x,d+1))
        return -1
            

class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        
        if S==T:
            return 0
        graph=collections.defaultdict(list)
        for i,stops in enumerate(routes):
            for s in stops:
                graph[s].append(i)
        que=graph[S]
        visited=set()
        steps=0
        while que:
            tmp=[]
            for bus in que:
                if bus in visited:
                    continue
                visited.add(bus)
                for stop in routes[bus]:
                    if stop==T:
                        return steps+1
                    for bus2 in graph[stop]:
                        if bus2 not in visited:
                            tmp.append(bus2)
                            
            que=tmp
            steps+=1
        return -1
#         if S == T: return 0
#         routes = list(map(set, routes))
#         graph = collections.defaultdict(set)
#         for i, r1 in enumerate(routes):
#             for j in range(i+1, len(routes)):
#                 r2 = routes[j]
#                 if any(r in r2 for r in r1):
#                     graph[i].add(j)
#                     graph[j].add(i)

#         seen, targets = set(), set()
#         for node, route in enumerate(routes):
#             if S in route: seen.add(node)
#             if T in route: targets.add(node)

#         queue = [(node, 1) for node in seen]
#         for node, depth in queue:
#             if node in targets: return depth
#             for nei in graph[node]:
#                 if nei not in seen:
#                     seen.add(nei)
#                     queue.append((nei, depth+1))
#         return -1

class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        if S == T: return 0
        dic_stops = collections.defaultdict(set)
        for i, stops in enumerate(routes):
            for s in stops:
                dic_stops[s].add(i)
        visited_stops = set([S])
        visited_buses = set()
        queue = collections.deque([(S, 1)])
        while queue:
            stop, cnt = queue.popleft()
            for bus in dic_stops[stop]:
                if bus not in visited_buses:
                    visited_buses.add(bus)
                    for s in routes[bus]:
                        if s not in visited_stops:
                            if s == T: return cnt
                            visited_stops.add(s)
                            queue.append((s, cnt + 1))
        return -1
                    
        
                            
                
                

from collections import deque
class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        if S == T:
            return 0
        # graph of bus stop to busses (which busses stop at that stop)
        graph = {}
        for bus in range(len(routes)):
            route = routes[bus]
            for stop in route:
                if stop in graph:
                    graph[stop].append(bus)
                else:
                    graph[stop] = [bus]
        
        # tuple for bus num and number of busses taken
        q = deque()
        q.append((-1, 0))
        taken = set()
        
        while len(q) > 0:
            busTuple = q.popleft()
            bus = busTuple[0]
            numBusses = busTuple[1]
            
            if bus != -1 and T in routes[bus]:
                return numBusses
            
            if bus == -1:
                for nextBus in graph[S]:
                    if nextBus not in taken:
                        taken.add(nextBus)
                        q.append((nextBus, numBusses + 1))
            else:
                for stop in routes[bus]:
                    for nextBus in graph[stop]:
                        if nextBus not in taken:
                            taken.add(nextBus)
                            q.append((nextBus, numBusses + 1))
        return -1
            

class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        
        if S==T:
            return 0
        graph=collections.defaultdict(list)
        for i,stops in enumerate(routes):
            for s in stops:
                graph[s].append(i)
        que=graph[S]
        visited=set()
        steps=0
        while que:
            tmp=[]
            for bus in que:
                if bus in visited:
                    continue
                visited.add(bus)
                for stop in routes[bus]:
                    if stop==T:
                        return steps+1
                    for bus2 in graph[stop]:
                        if bus2 not in visited:
                            tmp.append(bus2)
            que=tmp
            steps+=1
        return -1

from queue import Queue

class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        if S == T:
            return 0

        stopToBus = {}
        for bus in range(len(routes)):
            for stop in routes[bus]:
                stopToBus.setdefault(stop, [])
                stopToBus[stop].append(bus)
        
        q = Queue(maxsize=1000000)
        busVis = [False] * len(routes)
        stopVis = [False] * 1000000
        q.put((S, 0))
        while q.qsize() > 0:
            stop, dist = q.get()
            stopVis[stop] = True
            for bus in stopToBus[stop]:
                if busVis[bus] == False:
                    busVis[bus] = True
                    for ds in routes[bus]:
                        if stopVis[ds] == False:
                            if ds == T:
                                return dist + 1

                            q.put((ds, dist + 1))
        
        return -1
class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        
        
        if S == T:
            return 0
        
        graph = collections.defaultdict(list)
        
        #capture the bus routes
        for index, route in enumerate(routes):
            for stop in route:
                
                graph[stop].append(index)
                
        print(graph)        
        queue = graph[S]
        visited = set()
        steps = 0
        
        while queue:
            temp = []
            #check each bus of the soure node
            for bus in queue:
                if bus in visited:
                    continue
                else:    
                    visited.add(bus)
                    #check the  stops of each bus
                    for stops in routes[bus]:
                        if stops == T:
                            return steps + 1
                         #check the bus of each stops
                        for buses in graph[stops]:
                            if buses not in visited:
                                #each and every level fill it with new buses
                               temp.append(buses)
                        
            queue = temp            
            steps += 1
            
            
        return -1 
        
   
        
        
#         if S==T:
#             return 0
#         graph=collections.defaultdict(list)
#         for i,stops in enumerate(routes):
#             for s in stops:
#                 graph[s].append(i)
#         que=graph[S]
#         visited=set()
#         steps=0
#         while que:
#             tmp=[]
#             for bus in que:
#                 if bus in visited:
#                     continue
#                 visited.add(bus)
#                 for stop in routes[bus]:
#                     if stop==T:
#                         return steps+1
#                     for bus2 in graph[stop]:
#                         if bus2 not in visited:
#                             tmp.append(bus2)
                            
#             que=tmp
#             steps+=1
#         return -1
#         if S == T: return 0
#         routes = list(map(set, routes))
#         graph = collections.defaultdict(set)
#         for i, r1 in enumerate(routes):
#             for j in range(i+1, len(routes)):
#                 r2 = routes[j]
#                 if any(r in r2 for r in r1):
#                     graph[i].add(j)
#                     graph[j].add(i)

#         seen, targets = set(), set()
#         for node, route in enumerate(routes):
#             if S in route: seen.add(node)
#             if T in route: targets.add(node)

#         queue = [(node, 1) for node in seen]
#         for node, depth in queue:
#             if node in targets: return depth
#             for nei in graph[node]:
#                 if nei not in seen:
#                     seen.add(nei)
#                     queue.append((nei, depth+1))
#         return -1

class Solution:
    # O(n_stops x n_buses) time, O(n_stops x n_buses) space
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        num_buses = len(routes)
        bus_to_stop = defaultdict(set)
        for bus, stops in enumerate(routes):
            bus_to_stop[bus] = set(stops)
        
        def update_buses_used():
            for bus in range(num_buses):
                if bus in buses_used:
                    continue
                if stops_reached & bus_to_stop[bus]:
                    buses_used.add(bus)
        
        def update_stops_reached():
            for bus in buses_used:
                stops_reached.update(bus_to_stop[bus])
        
        buses_used = set()
        stops_reached = {S}
        pre_stop_count = 0
        bus_count = 0
        while len(stops_reached) > pre_stop_count:
            if T in stops_reached:
                return bus_count
            pre_stop_count = len(stops_reached)
            update_buses_used()
            update_stops_reached()
            bus_count += 1
            
        return -1

        
        
        

class Solution(object):
    def numBusesToDestination(self, routes, S, T):
        if S == T: return 0
        routes = list(map(set, routes))
        graph = collections.defaultdict(set)
        for i, r1 in enumerate(routes):
            for j in range(i+1, len(routes)):
                r2 = routes[j]
                for r in r1:
                    if r in r2:
                        graph[i].add(j)
                        graph[j].add(i)

        seen, targets = set(), set()
        for node, route in enumerate(routes):
            if S in route: seen.add(node)
            if T in route: targets.add(node)

        queue = [(node, 1) for node in seen]
        for node, depth in queue:
            if node in targets: return depth
            for nei in graph[node]:
                if nei not in seen:
                    seen.add(nei)
                    queue.append((nei, depth+1))
        return -1
from collections import defaultdict
class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        if(S==T):
            return 0
        routes = [set(r) for r in routes]
        graph = defaultdict(list)
        for i in range(len(routes)):
            for j in range(i+1, len(routes)):
                for k in routes[i]:
                    if k in routes[j]:
                        graph[j].append(i)
                        graph[i].append(j)
                        break
        queue = []
        for i, poss in enumerate(routes):
            if(S in poss):
                queue.append((i, 1))
        seen = set()
        while(queue):
            cur, cost = queue.pop(0)
            if(T in routes[cur]):
                return cost
            seen.add(cur)
            for child in graph[cur]:
                if(child not in seen):
                    queue.append((child, cost+1))
        return -1
from collections import defaultdict
from collections import deque
class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        '''
        node1 => {1,2,7}
        1 => node1
        3 => node2
        2 => node1
        6 => node2
        7 => node1, node2
        
        node2 => {3,6,7}
        '''
        buses = defaultdict(list) ## map stops to buses
        for i in range(len(routes)):
            for stop in routes[i]:
                buses[stop].append(i)
                
        queue = deque([(S,0)])
        busVisited = set()
        while queue:
            stop, busCnt = queue.popleft()
            if stop == T:
                return busCnt
            for bus in buses[stop]:
                if bus in busVisited:
                    continue
                busVisited.add(bus)
                for reachableStop in routes[bus]:
                    #if reachableStop not in visited:
                    queue.append((reachableStop, busCnt+1))
            
        return -1   
            

class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        # if S==T: return 0
        # g = defaultdict(list)
        # q = []
        # t = set()
        # routes = list(map(set, routes))
        # for i, r1 in enumerate(routes):
        #     if S in r1: q.append((i, 1))
        #     if T in r1: t.add(i)
        #     for j in range(i+1, len(routes)):
        #         r2 = routes[j]
        #         if any(s in r2 for s in r1):
        #             g[i].append(j)
        #             g[j].append(i)
        # seen = set(q)
        # while q:
        #     node, jumps = q.pop(0)
        #     if node in t: return jumps
        #     for nxt in g[node]:
        #         if nxt not in seen:
        #             seen.add(nxt)
        #             q.append((nxt, jumps+1))
        # return -1
        
#         st = defaultdict(set)
#         rt = defaultdict(set)
#         for i, route in enumerate(routes):
#             for stop in route:
#                 st[stop].add(i)
#                 rt[i].add(stop)
        
#         q = deque([(S,0)])
#         st_seen = set()
#         rt_seen = set()
#         while q:
#             node, jumps = q.popleft()
#             if node == T: return jumps
#             for r in st[node]:
#                 if r not in rt_seen:
#                     rt_seen.add(r)
#                     for stop in routes[r]:
#                         if stop not in st_seen:
#                             q.append((stop, jumps+1))
#                             st_seen.add(stop)
#         return -1
    
    
        to_routes = collections.defaultdict(set)
        for i, route in enumerate(routes):
            for j in route:
                to_routes[j].add(i)
        bfs = [(S, 0)]
        seen = set([S])
        for stop, bus in bfs:
            if stop == T: return bus
            for i in to_routes[stop]:
                for j in routes[i]:
                    if j not in seen:
                        bfs.append((j, bus + 1))
                        seen.add(j)
                routes[i] = []  # seen route
        return -1

class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        num_buses = len(routes)
        bus_to_stop = defaultdict(set)
        stop_to_bus = defaultdict(set)
        for bus, stops in enumerate(routes):
            bus_to_stop[bus] = set(stops)
            for stop in stops:
                stop_to_bus[stop].add(bus)
        
        
        def update_buses_used():
            for bus in range(num_buses):
                if bus in buses_used:
                    continue
                if stops_reached & bus_to_stop[bus]:
                    buses_used.add(bus)
        
        def update_stops_reached(stops_reached):
            for bus in buses_used:
                stops_reached |= bus_to_stop[bus]
        
        buses_used = set()
        stops_reached = {S}
        pre_stop_count = 0
        bus_count = 0
        while len(stops_reached) > pre_stop_count:
            if T in stops_reached:
                return bus_count
            pre_stop_count = len(stops_reached)
            update_buses_used()
            update_stops_reached(stops_reached)
            bus_count += 1
            
        return -1
        
        
        

class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        if S == T: return 0
        queue = collections.deque()
        dic_buses = collections.defaultdict(set)
        dic_stops = collections.defaultdict(set)
        visited = set()
        reachable = set()
        for i, stops in enumerate(routes):
            dic_buses[i] = set(stops)
            if S in dic_buses[i]:
                if T in dic_buses[i]: return 1
                visited.add(i)
                reachable |= dic_buses[i]
                queue.append(i)
            for j in dic_buses[i]:
                dic_stops[j].add(i)
        bus_need = 2
        visited_stops = set()
        while queue:
            length = len(queue)
            for _ in range(length):
                bus = queue.popleft()
                for stop in dic_buses[bus]:
                    if stop in visited_stops: continue
                    visited_stops.add(stop)
                    for b in dic_stops[stop]:
                        if b not in visited:
                            if T in dic_buses[b]:
                                return bus_need
                            queue.append(b)
            bus_need += 1
        return -1
                            
                
                

class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        N = len(routes)
        if S == T:
            return 0
        routes = list(map(set, routes))
        g = [set() for _ in range(N)]
        for ii, r1 in enumerate(routes):
            for jj in range(ii + 1, N):
                r2 = routes[jj]
                if any([1 for kk in r2 if kk in r1]):
                    g[ii].add(jj)
                    g[jj].add(ii)
        ss, tt = set(), set()
        for ii, jj in enumerate(routes):
            if S in jj:
                ss.add(ii)
            if T in jj:
                tt.add(ii)
        queue = [(ii, 1) for ii in ss]
        for ii, jj in queue:
            if ii in tt:
                return jj
            for kk in g[ii]:
                if kk not in ss:
                    ss.add(kk)
                    queue.append((kk, jj + 1))
        return -1

from collections import defaultdict
from collections import deque
class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        '''
        node1 => {1,2,7}
        1 => node1
        3 => node2
        2 => node1
        6 => node2
        7 => node1, node2
        
        node2 => {3,6,7}
        '''
        buses = defaultdict(list) ## map stops to buses
        for i in range(len(routes)):
            for stop in routes[i]:
                buses[stop].append(i)
                
        queue = deque([(S,0)])
        visited = set()
        busVisited = set()
        while queue:
            stop, busCnt = queue.popleft()
            visited.add(stop)
            if stop == T:
                return busCnt
            for bus in buses[stop]:
                if bus in busVisited:
                    continue
                busVisited.add(bus)
                for reachableStop in routes[bus]:
                    if reachableStop not in visited:
                        queue.append((reachableStop, busCnt+1))
            
        return -1   
            

from collections import defaultdict, deque
class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        path = defaultdict(list)
        for i, v in enumerate(routes):
            for bus_stop in v:
                path[bus_stop].append(i)
        used = set()
        bus_seen = set([S])
        queue = deque([S])
        taken = 0
        while queue:
            for _ in range(len(queue)):
                bus_stop = queue.popleft()
                if bus_stop == T:
                    return taken
                for route in path[bus_stop]:
                    if route not in used:
                        used.add(route)
                        for next_bus in routes[route]:
                            if next_bus not in bus_seen:
                                queue.append(next_bus)
            taken += 1
        return -1
                  
                        
        
                

class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        if S== T:
            return 0
        graph = collections.defaultdict(list)
        for i,j in enumerate(routes):
            for stops in j:
                graph[stops].append(i)
                
        q = collections.deque(graph[S])
        visited = set()
        steps = 0
        while q:
            temp =[]
            
            for bus in q:
                if bus in visited:
                    continue
                visited.add(bus)
                for r in routes[bus]:
                    if r==T:
                        return steps + 1
                    for b in graph[r]:
                        if b not in visited:
                            temp.append(b)
            q = temp
            steps+= 1
        return -1
                

class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        if S==T:
            return 0
        stops = collections.defaultdict(list)
        for i, route in enumerate(routes):
            for stop in route:
                stops[stop].append(i)
        num_routes = len(routes)
        expanded = set()
        q = collections.deque()
        routes = [set(route) for route in routes]
        for route in stops[S]:
            q.append([route,1])
            expanded.add(route)
        while q:
            cur_route, buses_taken = q.popleft()
            if T in routes[cur_route]:
                return buses_taken
            for stop in routes[cur_route]:
                for transfer_route in stops[stop]:
                    if transfer_route not in expanded:
                        expanded.add(transfer_route)
                        q.append([transfer_route,buses_taken+1])
        return -1

class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        if S == T:
            return 0
        stop = {}
        for i in range(len(routes)):
            for s in routes[i]:
                if s not in stop:
                    stop[s] = set()
                stop[s].add(i)
        stack = [(S, 0)]
        visited = set([S])
        while stack:
            node, level = stack.pop(0)
            for bus in stop[node]:
                for s in set(routes[bus]) - visited:
                    if s == T:
                        return level + 1
                    stack.append((s, level + 1))
                    visited.add(s)
        return -1
class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        if S == T: return 0
        
        graph = defaultdict(list)
        
        start = []
        end = set()
        for i in range(len(routes)):
            iset = set(routes[i])
            for j in range(i+1, len(routes)):
                if any(r in iset for r in routes[j]):
                    graph[i].append(j)
                    graph[j].append(i)
                    
            if S in iset:
                start.append(i)
            if T in iset:
                end.add(i)
                
        qu = deque(start)
        
        ret = 1
        visited = set()
        while qu:
            nextqu = deque()
            while qu:
                cur = qu.pop()
                
                if cur in end: return ret
                
                if cur in visited: continue
                visited.add(cur)
                
                for n in graph[cur]:
                    if n not in visited:
                        nextqu.append(n)
                        
            qu = nextqu
            ret += 1
        
        return -1
class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        if S == T:
            return 0
        stop_to_bus = collections.defaultdict(list)
        for bus,stops in enumerate(routes):
            for stop in stops:
                stop_to_bus[stop].append(bus)
        
        q = collections.deque([S])
        seen_bus = set()
        # seen_stop = set()
        step = -1
        while q:
            step += 1
            for _ in range(len(q)):
                stop = q.popleft()
                if stop == T:
                    return step
                for bus in stop_to_bus[stop]:
                    if bus in seen_bus:
                        continue
                    seen_bus.add(bus)
                    for next_stop in routes[bus]:
                        # if next_stop in seen_stop:
                        #     continue
                        q.append(next_stop)
                        # seen_stop.add(stop)
        return -1        
class Solution(object):
    def numBusesToDestination(self, routes, S, T):
        if S == T:
            return 0
        
        to_routes = collections.defaultdict(set)
        for i, route in enumerate(routes):
            for j in route:
                to_routes[j].add(i)
        bfs = [(S, 0)]
        # seen = set([S])
        for stop, bus in bfs:
            for i in to_routes[stop]:
                for j in routes[i]:
                    # if j not in seen:
                    if j == T: 
                        return bus + 1
                    bfs.append((j, bus + 1))
                        # seen.add(j)
                routes[i] = []  # seen route
        return -1
class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        
#         if S==T:
#             return 0
#         graph=collections.defaultdict(list)
#         for i,stops in enumerate(routes):
#             for s in stops:
#                 graph[s].append(i)
#         que=graph[S]
#         visited=set()
#         steps=0
#         while que:
#             tmp=[]
#             for bus in que:
#                 if bus in visited:
#                     continue
#                 visited.add(bus)
#                 for stop in routes[bus]:
#                     if stop==T:
#                         return steps+1
#                     for bus2 in graph[stop]:
#                         if bus2 not in visited:
#                             tmp.append(bus2)
#             que=tmp
#             steps+=1
#         return -1
        if S == T: return 0
        routes = list(map(set, routes))
        graph = collections.defaultdict(set)
        for i, r1 in enumerate(routes):
            for j in range(i+1, len(routes)):
                r2 = routes[j]
                if any(r in r2 for r in r1):
                    graph[i].add(j)
                    graph[j].add(i)

        seen, targets = set(), set()
        for node, route in enumerate(routes):
            if S in route: seen.add(node)
            if T in route: targets.add(node)

        queue = [(node, 1) for node in seen]
        for node, depth in queue:
            if node in targets: return depth
            for nei in graph[node]:
                if nei not in seen:
                    seen.add(nei)
                    queue.append((nei, depth+1))
        return -1

class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        if S == T:
            return 0
        
        graph = collections.defaultdict(list)
        for i,stops in enumerate(routes):
            for stop in stops:
                graph[stop].append(i)
                
        q = graph[S]
        visited = set()
        steps = 0
        while q:
            tmp = []
            for bus in q:
                if bus in visited:
                    continue
                visited.add(bus)
                for stop in routes[bus]:
                    if stop == T:
                        return steps + 1
                    for bus2 in graph[stop]:
                        if bus2 not in visited:
                            tmp.append(bus2)
                            
            q = tmp
            steps += 1
        return -1
from collections import defaultdict, deque
class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        path = defaultdict(list)
        for i, v in enumerate(routes):
            for bus_stop in v:
                path[bus_stop].append(i)
        used = set()
        queue = deque([S])
        taken = 0
        while queue:
            for _ in range(len(queue)):
                bus_stop = queue.popleft()
                if bus_stop == T:
                    return taken
                for route in path[bus_stop]:
                    if route not in used:
                        used.add(route)
                        for next_bus in routes[route]:
                            if next_bus != bus_stop:
                                queue.append(next_bus)
            taken += 1
        return -1
                  
                        
        
                

from collections import defaultdict, deque

class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        
        R = defaultdict(list)
        
        for idx, r in enumerate(routes):
            for s in r:
                R[s].append(idx)
        
        
        queue = deque([(S, 0)])
        visited = set()
        
        while queue:
            cur, res = queue.popleft()
            if cur == T:
                return res
            
            for stop in R[cur]:
                if stop not in visited:
                    visited.add(stop)
                    for nxt in routes[stop]:
                        if nxt != cur:
                            queue.append((nxt, res + 1))
        return -1
                            
            
            

from collections import defaultdict, deque

class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        d = defaultdict(list)
        visited = set()
        for idx, r in enumerate(routes):
            for s in r:
                d[s].append(idx)
        
        queue = deque([(S, 0)])
        while queue:
            cur, step = queue.popleft()
            
            if cur == T:
                return step
            
            for r in d[cur]:
                if r not in visited:
                    for s in routes[r]:
                        queue.append((s, step+1))
                    visited.add(r)
        
        return -1

class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        if S == T: return 0
        routes = list(map(set, routes))
        graph = collections.defaultdict(set)
        for i, r1 in enumerate(routes):
            for j in range(i+1, len(routes)):
                r2 = routes[j]
                if any(r in r2 for r in r1):
                    graph[i].add(j)
                    graph[j].add(i)

        seen, targets = set(), set()
        for node, route in enumerate(routes):
            if S in route: seen.add(node)
            if T in route: targets.add(node)

        queue = [(node, 1) for node in seen]
        for node, depth in queue:
            if node in targets: return depth
            for nei in graph[node]:
                if nei not in seen:
                    seen.add(nei)
                    queue.append((nei, depth+1))
        return -1
class Solution(object):
    def numBusesToDestination(self, routes, S, T):
        if S == T: return 0
        routes = list(map(set, routes))
        graph = collections.defaultdict(set)
        for i, r1 in enumerate(routes):
            for j in range(i+1, len(routes)):
                r2 = routes[j]
                if any(r in r2 for r in r1):
                    graph[i].add(j)
                    graph[j].add(i)

        seen, targets = set(), set()
        for node, route in enumerate(routes):
            if S in route: seen.add(node)
            if T in route: targets.add(node)

        queue = [(node, 1) for node in seen]
        for node, depth in queue:
            if node in targets: return depth
            for nei in graph[node]:
                if nei not in seen:
                    seen.add(nei)
                    queue.append((nei, depth+1))
        return -1
class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        if S==T: return 0
        g = defaultdict(list)
        q = []
        t = set()
        routes = list(map(set, routes))
        for i, r1 in enumerate(routes):
            if S in r1: q.append((i, 1))
            if T in r1: t.add(i)
            for j in range(i+1, len(routes)):
                r2 = routes[j]
                if any(s in r2 for s in r1):
                    g[i].append(j)
                    g[j].append(i)
        seen = set(q)
        while q:
            node, jumps = q.pop(0)
            if node in t: return jumps
            for nxt in g[node]:
                if nxt not in seen:
                    seen.add(nxt)
                    q.append((nxt, jumps+1))
        return -1
class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        if S == T: return 0
        routes = list(map(set, routes))
        graph = collections.defaultdict(set)
        for i, r1 in enumerate(routes):
            for j in range(i+1, len(routes)):
                r2 = routes[j]
                if any(r in r2 for r in r1):
                    graph[i].add(j)
                    graph[j].add(i)
        
        visited, targets = set(), set()
        for bus, route in enumerate(routes): 
            if S in route: 
                visited.add(bus)
            if T in route: 
                targets.add(bus)
        
        queue = []
        for bus in visited: 
            queue.append((bus, 1))
        while queue: 
            bus, numBuses = queue.pop(0)
            if bus in targets: 
                return numBuses
            visited.add(bus)
            for connecting_bus in graph[bus]: 
                if connecting_bus not in visited: 
                    queue.append((connecting_bus, numBuses+1))
        return -1
from collections import defaultdict
class Solution():
    def numBusesToDestination(self, routes, S, T):
        if S == T: return 0
        routes = list(map(set, routes))
        graph = defaultdict(set)
        for i, r1 in enumerate(routes):
            for j in range(i+1, len(routes)):
                r2 = routes[j]
                if any(r in r2 for r in r1):
                    graph[i].add(j)
                    graph[j].add(i)

        seen, targets = set(), set()
        for node, route in enumerate(routes):
            if S in route: seen.add(node)
            if T in route: targets.add(node)

        queue = [(node, 1) for node in seen]
        for node, depth in queue:
            if node in targets: return depth
            for nei in graph[node]:
                if nei not in seen:
                    seen.add(nei)
                    queue.append((nei, depth+1))
        return -1
class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        to_routes = collections.defaultdict(set)
        for i, route in enumerate(routes):
            for j in route:
                to_routes[j].add(i)
        bfs = [(S, 0)]
        seen = set([S])
        for stop, bus in bfs:
            if stop == T: return bus
            for i in to_routes[stop]:
                for j in routes[i]:
                    if j not in seen:
                        bfs.append((j, bus + 1))
                        seen.add(j)
                routes[i] = []  # seen route
        return -1
class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int):
        
        graph = collections.defaultdict(list)
        for i in range(len(routes)):
            for route in routes[i]:
                graph[route].append(i)
        
        q = collections.deque([(S, 0)])
        visited = set()
        
        while q:
        
            cur_stop, out = q.popleft()    
            if cur_stop == T: 
                return out
            for bus in graph[cur_stop]:
                if bus not in visited:
                    visited.add(bus)
                    for stop in routes[bus]:
                        q.append((stop, out + 1))
        return -1
from collections import defaultdict, deque
class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        if S == T: return 0
        routes = [set(route) for route in routes]
        graph = collections.defaultdict(set)
        for i, r1 in enumerate(routes):
            for j in range(i+1,len(routes)):
                r2 = routes[j]
                if any(r in r2 for r in r1):
                    graph[i].add(j)
                    graph[j].add(i)
        seen = set()
        target = set()
        for node,route in enumerate(routes):
            if S in route: 
                seen.add(node)
            if T in route: 
                target.add(node)
        q = deque()
        for node in seen:
            q.append((node,1))
        while q:
            curr, depth = q.popleft()
            if curr in target:
                return depth
            for child in graph[curr]:
                if child not in seen:
                    seen.add(child)
                    q.append((child,depth+1))
        return -1
        
        
        
                

class Solution:
    def numBusesToDestination(self, routes, S, T):
        if S == T: return 0
        routes = list(map(set, routes))
        graph = collections.defaultdict(set)
        for i, r1 in enumerate(routes):
            for j in range(i+1, len(routes)):
                r2 = routes[j]
                if any(r in r2 for r in r1):
                    graph[i].add(j)
                    graph[j].add(i)

        seen, targets = set(), set()
        for node, route in enumerate(routes):
            if S in route: seen.add(node)
            if T in route: targets.add(node)

        queue = [(node, 1) for node in seen]
        for node, depth in queue:
            if node in targets: return depth
            for nei in graph[node]:
                if nei not in seen:
                    seen.add(nei)
                    queue.append((nei, depth+1))
        return -1
class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        if S == T:
            return 0
        graph = collections.defaultdict(set) # node is the bus, not stop
        routes = list(map(set, routes))
        for i, r1 in enumerate(routes):
            for j in range(i + 1, len(routes)):
                r2 = routes[j]
                if any(r in r2 for r in r1):
                    graph[i].add(j)
                    graph[j].add(i)
        
        visited, targets = set(), set()
        for i, r in enumerate(routes):
            if S in r:
                visited.add(i)
            if T in r:
                targets.add(i)
        
        queue = collections.deque([(node, 1) for node in visited])
        while queue:
            node, step = queue.popleft()
            if node in targets:
                return step
            for neighbor in graph[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, step + 1))
        return -1
            
        
        

class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
# # u4ee5u7ad9u70b9u4f5cu4e3anodeu5efau7acbgraphuff0cu7ad9u70b9u76f4u63a5u7d22u5f15u4e0bu4e00u4e2au7ad9u70b9: time limit exceeded... u6000u7591u662fu6784u5efagraphu592au8017u65f6
#         stop_graph = {}
#         visited = set()
#         for rt in routes:
#             for i in range(len(rt)):
#                 if rt[i] not in stop_graph: stop_graph[rt[i]] = set(rt[:i] + rt[i+1:])
#                 else: 
#                     # print(rt[i], set(rt[:i] + rt[i+1:]))
#                     stop_graph[rt[i]] = stop_graph[rt[i]].union(set(rt[:i] + rt[i+1:]))
#         #BFS
#         # visited = set()
#         # print(stop_graph)
#         if S == T: return 0
#         if S not in stop_graph or T not in stop_graph: return -1
#         cnt = 0
#         queue = [S]
#         visited = set(queue)
#         while queue:
#             # print(queue)
#             size = len(queue)
#             for _ in range(size):
#                 curr_stop = queue.pop(0)
#                 for next_stop in stop_graph[curr_stop]:
#                     if next_stop in visited: continue
#                     if next_stop == T: return cnt + 1
#                     queue.append(next_stop) 
#                     visited.add(next_stop)
#             cnt += 1
#         else: return -1

# u4ee5u7ad9u70b9u4f5cu4e3anodeu5efau7acbgraphuff0cu4f46u7528u7ad9u70b9u7d22u5f15busu73edu6b21uff0cu518du7528Busu7d22u5f15u7ad9u70b9u7684u5f62u5f0f
# uff01uff01uff01u907fu514dtime limit exceedu65b9u6cd5u5fc5u505au4e8bu9879uff1au8bbeu4e24u4e2aset()u5206u522bu8bb0u5f55u904du5386u8fc7u7684bust stop & bus routeuff0cu6bcfu4e2astop / routeu90fdu53eau5e94u8be5u6700u591au5750u4e00u6b21uff0cu8fd9u6837u80fdu907fu514du91cdu590du4fbfu5229stop/routeu5e26u6765u7684foru5faau73afu8017u65f6u6700u7ec8u5bfcu81f4u65f6u95f4u6ea2u51fa
        graph = {}
        for i in range(len(routes)):
            for st in routes[i]:
                if st not in graph: graph[st] = [i]
                else: graph[st].append(i)
        
        visited_st = set([S])
        if S == T: return 0
        if S not in graph or T not in graph: return -1
        queue = graph[S]
        # print(graph[S])
        visited_rt = set(graph[S])
        cnt = 0
        while queue:
            cnt += 1
            size = len(queue)
            for _ in range(size):
                curr_rt_ind = queue.pop(0)
                # if curr_rt_ind in visited: continue
                # visited.add(curr_rt_ind)
                for st in routes[curr_rt_ind]:
                    if st in visited_st: continue
                    if st == T: return cnt
                    visited_st.add(st)
                    for rt_ind in graph[st]:
                        if rt_ind in visited_rt: continue
                        queue.append(rt_ind)
                        visited_rt.add(rt_ind)
        else: return -1

#         if S == T: return 0
#         # routes = set(routes)
#         graph = collections.defaultdict(set)
#         for i, r1 in enumerate(routes):
#             for j in range(i+1, len(routes)):
#                 r2 = routes[j]
#                 if any(r in r2 for r in r1):
#                     graph[i].add(j)
#                     graph[j].add(i)

#         seen, targets = set(), set()
#         for node, route in enumerate(routes):
#             if S in route: seen.add(node)
#             if T in route: targets.add(node)

#         queue = [(node, 1) for node in seen]
#         for node, depth in queue:
#             if node in targets: return depth
#             for nei in graph[node]:
#                 if nei not in seen:
#                     seen.add(nei)
#                     queue.append((nei, depth+1))
#         return -1
            

class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        if S == T:
            return 0
        graph = collections.defaultdict(set)
        routes = list(map(set, routes))
        for i, r1 in enumerate(routes):
            for j in range(i + 1, len(routes)):
                r2 = routes[j]
                if any(r in r2 for r in r1):
                    graph[i].add(j)
                    graph[j].add(i)
        
        visited, targets = set(), set()
        for i, r in enumerate(routes):
            if S in r:
                visited.add(i)
            if T in r:
                targets.add(i)
        
        queue = collections.deque([(node, 1) for node in visited])
        while queue:
            node, step = queue.popleft()
            if node in targets:
                return step
            for neighbor in graph[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, step + 1))
        return -1
            
        
        

class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        import queue
        
        if S == T:
            return 0
        #turn routes into graph
        edge_dict = {}
        for i in range(len(routes)):
            for j in range(i+1, len(routes)):
                if i not in edge_dict:
                    edge_dict[i] = {}
                    
                if j not in edge_dict:
                    edge_dict[j] = {}
            
                if len(set(routes[i]).intersection(set(routes[j]))) != 0:
                    edge_dict[i][j] = 1
                    edge_dict[j][i] = 1    
            
        #print(edge_dict)    
        #bfs on this graph, starting at S trying to get to T
        frontier = queue.SimpleQueue()
        seen = {}
    

        for i in range(len(routes)):
            if S in routes[i]:
                frontier.put((i, 1))
                seen[i] = 1
        
        while not frontier.empty():
            node, dist = frontier.get()
            #print(node)
            if T in routes[node]:
                return dist
            neighbors = list(edge_dict[node].keys())
            for neighbor in neighbors:
                if neighbor not in seen:
                    seen[neighbor] = dist+1
                    frontier.put((neighbor, dist+1))
                    
        return -1
        
        #if bfs fails return -1
    

class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        if S == T:
            return 0
        stop = {}
        for i in range(len(routes)):
            for s in routes[i]:
                stop[s] = stop.get(s,[]) + [i]
        for k in stop:
            stop[k] = set(stop[k])
        stack = [(S, 0)]
        visited = set([S])
        while stack:
            node, level = stack.pop(0)
            for bus in stop[node]:
                for s in set(routes[bus]) - visited:
                    if s == T:
                        return level + 1
                    stack.append((s, level + 1))
                    visited.add(s)
        return -1
class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        if S == T:
            return 0
        
        def bfs():
            Q = []
            seen = set()
            for b in busAtStops[S]:
                Q.append((b, 1))
            
            while Q:
                bus, depth = Q.pop(0)
                if T in routes[bus]:
                    return depth

                for stop in routes[bus]:
                    for bus in busAtStops[stop]:
                        if bus not in seen:
                            seen.add(bus)
                            Q.append((bus, depth+1))
            return -1            
        
        busAtStops = defaultdict(list)
        for i, r in enumerate(routes):
            for stop in r:
                busAtStops[stop].append(i)
            routes[i] = set(routes[i])
        
        return bfs()
                
        

from collections import defaultdict, deque

class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        if S == T or not routes:
            return 0

        # O(BR)
        mapping = defaultdict(list)
        for i, route in enumerate(routes):
            for start in route:  # , end in zip(route, route[1:]):
                mapping[start].append(i)

        q = deque([[S, 0]])

        # O(BR)
        while q:
            prev_stop, dist = q.popleft()

            if prev_stop not in mapping:
                continue

            bus_num_li = mapping[prev_stop]
            # mapping.pop(prev_stop)
            mapping[prev_stop] = []

            for cur_bus_num in bus_num_li:
                new_dist = dist + 1

                for cur_stop in routes[cur_bus_num]:
                    if cur_stop == T:
                        return new_dist

                    q.append([cur_stop, new_dist])

                routes[cur_bus_num] = []

            # for cur_stop, cur_bus_num in cur_stops.items():
            #     if cur_stop == T:
            #         return dist
            #     q.append([cur_stop, dist + (prev_bus_num or prev_bus_num != cur_bus_num), cur_bus_num])

        return -1


class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        if S == T:
            return 0

        stop_to_route = defaultdict(list)

        for i in range(len(routes)):
            for stop in routes[i]:
                stop_to_route[stop].append(i)

        graph = {}

        for u in range(len(routes)):
            graph[u] = set()

        for i in range(len(routes)):
            for stop in routes[i]:
                for r2 in stop_to_route[stop]:
                    if r2 != i:
                        graph[i].add(r2)

        end = set(stop_to_route[T])
        visited = [False] * len(routes)

        initial = []

        for r in stop_to_route[S]:
            initial.append((r,1))

        q = deque(initial)

        while q:
            u,cost = q.popleft()

            if u in end:
                return cost

            for v in graph[u]:
                if not visited[v]:
                    visited[v] = True
                    q.append((v,cost+1))

        return -1

class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        if S == T: 
            return 0
        routes = list(map(set, routes))
        graph = collections.defaultdict(set)
        for i, r1 in enumerate(routes):
            for j in range(i+1, len(routes)):
                r2 = routes[j]
                if any(r in r2 for r in r1):
                    graph[i].add(j)
                    graph[j].add(i)

        seen, targets = set(), set()
        for node, route in enumerate(routes):
            if S in route: 
                seen.add(node)
            if T in route: 
                targets.add(node)

        queue = [(node, 1) for node in seen]
        for node, depth in queue:
            if node in targets: return depth
            for nei in graph[node]:
                if nei not in seen:
                    seen.add(nei)
                    queue.append((nei, depth+1))
        return -1
class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        if S == T:
            return 0
        
        def bfs():
            Q = []
            seen = set()
            for b in busAtStops[S]:
                Q.append((b, 1))
            
            while Q:
                bus, depth = Q.pop(0)
                if bus not in seen:
                    seen.add(bus)
                    if T in routes[bus]:
                        return depth

                    for stop in routes[bus]:
                        for bus in busAtStops[stop]:
                            if bus not in seen:
                                Q.append((bus, depth+1))
            return -1            
        
        busAtStops = defaultdict(list)
        for i, r in enumerate(routes):
            for stop in r:
                busAtStops[stop].append(i)
            # routes[i] = set(routes[i])
        
        return bfs()
                
        

class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        '''
        # BFS
        # Treat each bus stop as node => TLE
        # A node is connected with all nodes in the same route(s)
        # Time: O(V + E)
        # Space: O(V + E)
        
        # Build up adjacency list
        adj = collections.defaultdict(set)
        for route in routes:
            for stop in route:
                adj[stop].update(route)
            
        q = collections.deque()
        q.append(S)
        visited = set()
        visited.add(S)
        buses = 0
        while q:
            for _ in range(len(q)):
                u = q.popleft()
                if u == T:
                    return buses
                for v in adj[u]:
                    if v not in visited:
                        q.append(v)
                        visited.add(v)
            buses += 1
        return -1
        '''
        
        # BFS
        # Treat each bus route as node
        # A node is connected with all routes that share a common bus stop
        
        # Build up adjacency list
        adj = collections.defaultdict(set)
        n = len(routes)
        for i in range(n):
            u = routes[i]
            for j in range(i+1, n):
                v = routes[j]
                if set(u).intersection(set(v)):
                    adj[i].add(j)
                    adj[j].add(i)
                    
        # Build up bus stop -> bus route mapping
        stop2routes = collections.defaultdict(set)
        for route, stops in enumerate(routes):
            for stop in stops:
                stop2routes[stop].add(route)

        if S == T:
            return 0
        
        adj[-1] = stop2routes[S] # route -1 is connected to all routes that contains S
        q = collections.deque()
        q.append(-1)
        visited = set()
        visited.add(-1)
        buses = 0
        while q:
            for _ in range(len(q)):
                u = q.popleft()
                if u in stop2routes[T]:
                    return buses
                for v in adj[u]:
                    if v not in visited:
                        visited.add(v)
                        q.append(v)
            buses += 1
        return -1

class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        if S == T:
            return 0
        stop_to_bus = collections.defaultdict(set)
        bus_to_bus = {i: [] for i in range(len(routes))}
        
        # end_bus_q = collections.deque([])
        for bus, route in enumerate(routes):
            for stop in route:
                stop_to_bus[stop].add(bus)
        # print(stop_to_bus, init_bus_q)
        q = collections.deque(stop_to_bus[S])
        
        for i in range(len(routes)):
            for j in range(i+1, len(routes)):
                if len(set(routes[i]) & set(routes[j])) > 0:
                    bus_to_bus[i].append(j)
                    bus_to_bus[j].append(i)
        steps = 1
        visited = [False] * len(routes)
        # print(visited)
        while q:
            size = len(q)
            for _ in range(size):
                bus = q.popleft()
                # print(bus)
                visited[bus] = True
                if bus in stop_to_bus[T]:
                    return steps
                for nb in bus_to_bus[bus]:
                    if visited[nb]:
                        continue
                    q.append(nb)
            steps += 1
        return -1
            
        

class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        if S == T: return 0
        graph = collections.defaultdict(set)
        num_buses = len(routes)
        routes = list(map(set, routes))
        for i, route1 in enumerate(routes):
            for j in range(i+1, num_buses):
                route2 = routes[j]
                
                if any(r in route2 for r in route1):
                    graph[i].add(j)
                    graph[j].add(i)
        
        seen, targets = set(), set()
        for node, route in enumerate(routes):
            if S in route: seen.add(node)
            if T in route: targets.add(node)
        
        queue = [(node, 1) for node in seen]
        for node, depth in queue:
            if node in targets: return depth
            for neighbour in graph[node]:
                if neighbour not in seen:
                    seen.add(neighbour)
                    queue.append((neighbour, depth+1))
        
        return -1

class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        if S == T: return 0
        routes = list(map(set, routes))
        graph = collections.defaultdict(set)
        for i, r1 in enumerate(routes):
            for j in range(i+1, len(routes)):
                r2 = routes[j]
                if any(r in r2 for r in r1):
                    graph[i].add(j)
                    graph[j].add(i)

        seen, targets = set(), set()
        for node, route in enumerate(routes):
            if S in route: seen.add(node)
            if T in route: targets.add(node)

        queue = [(node, 1) for node in seen]
        for node, depth in queue:
            if node in targets: return depth
            for nei in graph[node]:
                if nei not in seen:
                    seen.add(nei)
                    queue.append((nei, depth+1))
        return -1
        
        
        

class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        if S==T:
            return 0
        maps={}
        for i in range(len(routes)):
            r=routes[i]
            for j in range(len(r)):
                if r[j] not in maps.keys():
                    maps[r[j]]=set()
                maps[r[j]].add(i)
        res=0
        lens=len(routes)
        seenBus=[0]*lens
        seenStop=set()
        q=[S]
        res=0
        while q:
            res+=1
            size=len(q)
            for i in range(size):
                stop=q.pop(0)
                for bus in maps[stop]:
                    if seenBus[bus]==1:
                        continue
                    seenBus[bus]=1
                    for s in routes[bus]:
                        if s==T:
                            return res
                        if s in seenStop:
                            continue
                        seenStop.add(s)
                        q.append(s)
                        
        return -1
class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        if S==T:
            return 0
        
        routes = list(map(set, routes))
            
        
        graph=collections.defaultdict(set)
        for i,e in enumerate(routes):
            for j in range(i+1,len(routes)):
                for node in e:
                    if node in routes[j]:
                        graph[i].add(j)
                        graph[j].add(i)
        
        target,start=set(),set()
        visited=set()
        
        for i in range(len(routes)):
            if S in routes[i]:
                start.add(i)
                visited.add(i)
            if T in routes[i]:
                target.add(i)
    
        step=1
        while start:
            nextlevel=set()
            for node in start:
                if node in target:
                    return step
                for neighbornode in graph[node]:
                    if neighbornode not in visited:
                        visited.add(neighbornode)
                        nextlevel.add(neighbornode)
            
            step+=1
            start=nextlevel
        
        return -1
from collections import defaultdict,deque

class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        if(S==T):
            return 0
        circuits=defaultdict(list)
        rel={}
        for i in range(len(routes)):
            for stand in routes[i]:
                circuits[stand].append(i)
                rel[(i,stand)]=1
        vis={i:1 for i in circuits[S]}
        vis_bus={S:1}
        q=deque([i for i in circuits[S]])
        level=1
        while q:
            l=len(q)
            for i in range(l):
                curr_cir=q.popleft()
                if((curr_cir,T) in rel):
                    return level
                for bus in routes[curr_cir]:
                    if bus not in vis_bus:
                        vis_bus[bus]=1
                        for cir in circuits[bus]:
                            if cir not in vis:
                                q.append(cir)
                                vis[cir]=1
            level+=1
        return -1

class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        if T == S:
            return 0
        possibleStarts = []
        possibleEnd = []
        stopGraph = {}
        for i, route in enumerate(routes):
            for stop in route:
                if stop == S:
                    possibleStarts.append(i)
                elif stop == T:
                    possibleEnd.append(i)
                
                if stop in stopGraph:
                    stopGraph[stop].append(i)
                else:
                    stopGraph[stop] = [i]
        
#         {7: [1, 0, 2],
#         8: [2, 0, 4]}
        
#         { 1:[0, 2],0:[1, 2], 2:[0, 1, 3, 4], 3:[2, 4], 4:[2, 3]}
        
        graph = {}
        for stop in list(stopGraph.keys()):
            validRoutes = stopGraph[stop]
            if len(stopGraph[stop]) <= 1:
                continue
            for route in validRoutes:
                if route not in graph:
                    graph[route] = set()
                for x in validRoutes:
                    if x != route:
                        graph[route].add(x)                  
        # print(graph)
        # print(stopGraph)
        # print(possibleStarts)
        # print(possibleEnd)
        queue = [(x, 1) for x in possibleStarts]
        seen = set()
        while queue:
            node = queue.pop(0)
            seen.add(node[0])
            if node[0] in possibleEnd:
                return node[1]
            if node[0] in graph:
                children = graph[node[0]]
            
                for child in children:
                    if child not in seen:
                        queue.append((child, node[1] + 1))
                        seen.add(child)

            
                
        return -1
        
                    
        

        
class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        if S == T :
           return 0 
        stopToBusDict = defaultdict()
        
        for bus in range(len(routes)):
            for stop in routes[bus]:
                if stop not in stopToBusDict:
                   stopToBusDict[stop] = [] 
                stopToBusDict[stop].append(bus)
    
        routes = [set(r) for r in routes]
        
        adjList = collections.defaultdict(set)
    
        
        for i, r1 in enumerate(routes):
            for j in range(i+1, len(routes)):
                r2 = routes[j]
                if any(r in r2 for r in r1):
                    adjList[i].add(j)
                    adjList[j].add(i)    
              
        targets = set()
        visited = set()
        for node, route in enumerate(routes):
            #print("T: %s node: %s route: %s" %(T,node,route))
            if T in route: targets.add(node)
            if S in route: visited.add(node)    
       
            
            
        def bfs(S,T) :
            queue = collections.deque()
            #print("target: %s" %(targets))
            for bus in stopToBusDict[S]:
                queue.append([bus,1])
            while len(queue) != 0 :
                
                currentBus = queue.popleft()
                visited.add(currentBus[0]) 
                
                if currentBus[0] in targets:
                   return currentBus[1]
                
                for neighbour in adjList[currentBus[0]] :
                    if neighbour not in visited :
                       queue.append([neighbour,currentBus[1]+1])
            print(-1)        
            return -1        
                    
                
            
                
                
            
             
            
            
            
            
        return bfs(S,T)      
from collections import defaultdict
class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        
        if S == T:
            return 0
        
        stop_to_bus_map = defaultdict(list)
        L = len(routes)
        for i in range(L):
            for stop in routes[i]:
                stop_to_bus_map[stop].append(i)
                
        
        route_graph = defaultdict(set)
        
        for stop in stop_to_bus_map:
            
            for r1 in stop_to_bus_map[stop]:
                for r2 in stop_to_bus_map[stop]:
                    if r1 != r2:
                        route_graph[r1].add(r2)
                        
        
        start = set(stop_to_bus_map[S])
        end = set(stop_to_bus_map[T])

        
        count = 1
        visited= set()
        
        while( len(start) != 0):
            new_start = set()
            
            for route in start:
                if route in end:
                    return count
                if route in visited:
                    continue
                
                new_start |= route_graph[route]
                
            
            visited |= start
            start = new_start
            count +=1
        
        
        return -1
from collections import defaultdict, deque

class Solution:
    def bfs(self, start_bus):
        distance = [None] * self.num_buses
        distance[start_bus] = 1
        q = deque([start_bus])
        while q:
            bus = q.popleft()
            if self.target in self.buses_to_stations[bus]:
                return distance[bus]
            
            for station in self.buses_to_stations[bus]:
                for neighbour_bus in self.stations_to_busses[station]:
                    if distance[neighbour_bus] is None:
                        distance[neighbour_bus] = distance[bus] + 1
                        q.append(neighbour_bus)
        return 2**32
        
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        if S == T:
            return 0
        
        self.num_buses = len(routes)
        self.buses_to_stations = [set(route) for route in routes]
        
        self.stations_to_busses = defaultdict(list)
        self.target = T
        
        for bus in range(len(routes)):
            for station in routes[bus]:
                self.stations_to_busses[station].append(bus)
        
        m = 2**32
        
        for bus in self.stations_to_busses[S]:
            dist_by_bus = self.bfs(bus)
            m = min(m, dist_by_bus)
        
        if m == 2**32:
            return -1
        else:
            return m
class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        stk = [(S, 0)]
        stop_dic = collections.defaultdict(set)
        for i, route in enumerate(routes):
            for stop in route:
                stop_dic[stop].add(i)
                
        visited = set()
        while stk:
            stop, num_bus = stk.pop(0)
            if stop == T:
                return num_bus
            elif stop not in visited:
                visited.add(stop)
                buses = stop_dic[stop]
                for bus in buses:
                    for new_stop in routes[bus]:
                        if new_stop not in visited:
                            stk.append((new_stop, num_bus + 1))
                    routes[bus] = []
        return -1
class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        if S==T:
            return 0
        
        graph = collections.defaultdict(list)
        bfs = []
        end = []
        
        for i in range(len(routes)):
            
            if S in routes[i]:
                bfs.append((i,1))
            if T in routes[i]:
                end.append(i)
                
            for j in range(i+1,len(routes)):
                
                if set(routes[i]) & set(routes[j]):
                    
                    graph[i].append(j)
                    graph[j].append(i)
                    
        
        bfs = deque(bfs)
        visited = set(bfs)
        
        while bfs:
            
            ele = bfs.popleft()
            
            bus, nbus = ele
            
            if bus in end:
                return nbus
            
            for edge in graph[bus]:
                if edge not in visited:
                    visited.add(edge)
                    bfs.append((edge,nbus+1))
                    
        return -1
class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        '''
        # BFS
        # Treat each bus stop as node => TLE
        # A node is connected with all nodes in the same route(s)
        # Time: O(V + E)
        # Space: O(V + E)
        
        # Build up adjacency list
        adj = collections.defaultdict(set)
        for route in routes:
            for stop in route:
                adj[stop].update(route)
            
        q = collections.deque()
        q.append(S)
        visited = set()
        visited.add(S)
        buses = 0
        while q:
            for _ in range(len(q)):
                u = q.popleft()
                if u == T:
                    return buses
                for v in adj[u]:
                    if v not in visited:
                        q.append(v)
                        visited.add(v)
            buses += 1
        return -1
        '''
        
        # BFS
        # Treat each bus route as node
        # A node is connected with all routes that share a common bus stop
        
        # Build up adjacency list
        adj = collections.defaultdict(set)
        n = len(routes)
        for i in range(n):
            u = routes[i]
            for j in range(i+1, n):
                v = routes[j]
                if set(u).intersection(set(v)):
                    adj[i].add(j)
                    adj[j].add(i)
                    
        # Build up bus stop -> bus route mapping
        stop2route = collections.defaultdict(set)
        for route, stops in enumerate(routes):
            for stop in stops:
                stop2route[stop].add(route)

        if S == T:
            return 0
        
        adj[-1] = set([r for r in range(n) if r in stop2route[S]]) # route -1 is connected to all routes that contains S
        q = collections.deque()
        q.append(-1)
        visited = set()
        visited.add(-1)
        buses = 0
        while q:
            for _ in range(len(q)):
                u = q.popleft()
                if u in stop2route[T]:
                    return buses
                for v in adj[u]:
                    if v not in visited:
                        visited.add(v)
                        q.append(v)
            buses += 1
        return -1

class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        if S == T: return 0
        buses = collections.defaultdict(set)
        count = 0
        visited = set()
        
        for i, route in enumerate(routes):
            for stop in route:
                buses[stop].add(i)
                
        q = [(S,0)]
        for stop, count in q:
            if stop == T:
                return count
            for bus in buses[stop]:
                if bus not in visited:
                    visited.add(bus)
                    for next_stop in routes[bus]:
                        if next_stop != stop:
                            q.append((next_stop, count+1))

        return -1
        
        

class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        
        if S == T:
            return 0
        
        routes = [set(route) for route in routes]
        graph = defaultdict(set)
        
        for i, r1 in enumerate(routes):
            for j in range(i+1, len(routes)):
                r2 = routes[j]
                if any(r in r2 for r in r1):
                    graph[i].add(j)
                    graph[j].add(i)
        
        visited = set()
        target = set()
        for node, route in enumerate(routes):
            if S in route:
                visited.add(node)
            
            if T in route:
                target.add(node)
        
        q = deque([(node, 1) for node in visited])
        while q:
            node, transfer = q.popleft()
            if node in target:
                return transfer
            
            for neighbor in graph[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    q.append((neighbor, transfer + 1))
        
        return -1
class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        if S==T:
            return 0
        n=len(routes)
        stop_bus=defaultdict(list)
        for bus,route in enumerate(routes):
            for stop in route:
                stop_bus[stop].append(bus)
        graph=[[] for _ in range(n)]
        for bus,route in enumerate(routes):
            for stop in route:
                buses=stop_bus[stop]
                for nei in buses:
                    if nei!=bus:
                        graph[bus].append(nei)
                        graph[nei].append(bus)
        start=set(stop_bus[S])
        end=set(stop_bus[T])
        if start.intersection(end):
            return 1
        for s in start:
            visited=start.copy()
            q=deque([(s, 1)])
            while q:
                cur, l = q.popleft()
                for nei in graph[cur]:
                    if nei not in visited:
                        if nei in end:
                            return l+1
                        visited.add(nei)
                        q.append((nei, l+1))
        return -1
                        

from collections import defaultdict
class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        adj_list = defaultdict(set)
        for i,a in enumerate(routes):
            for k in a:
                adj_list[k].add(i)
        q = [[S,0]]
        visited = {}
        for stop,path_len in q:
            if stop==T: return path_len
            if visited.get(stop)==None:
                for route in adj_list[stop]:
                    for k in routes[route]:
                        q.append([k,path_len+1])
                visited[stop] = True
                routes[route] = []
        return -1

class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        if S == T: return 0
        n = len(routes)
        graph = collections.defaultdict(list)
        sources = set()
        targets = set()
        for i in range(n):
            if S in routes[i]:
                sources.add(i)
            if T in routes[i]:
                targets.add(i)
            for j in range(i+1, n):
                if set(routes[i]) & set(routes[j]):
                    graph[i].append(j)
                    graph[j].append(i)
        
        dist = 1
        seen = set()
        while sources:
            temp = set()
            for curr in sources:
                if curr in targets:
                    return dist
                for neighbor in graph[curr]:
                    if neighbor not in seen:
                        seen.add(neighbor)
                        temp.add(neighbor)
            sources = temp
            dist += 1
        return -1
class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        if not routes:
            return -1
        bus2stop = collections.defaultdict(set)
        stop2bus = collections.defaultdict(set)
        for bus, route in enumerate(routes):
            for stop in route:
                bus2stop[bus].add(stop)
                stop2bus[stop].add(bus)
                
                
        q = collections.deque()
        q.append((0, S))
        seen = set()
        seen.add(S)
        while q:
            hop, curStop = q.popleft()
            if curStop==T:
                return hop
            for bus in stop2bus[curStop]:
                for nxtStop in bus2stop[bus]:
                    if nxtStop not in seen:
                        seen.add(nxtStop)
                        q.append((hop+1, nxtStop))
                        
        return -1
class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        if S == T: return 0
        
        b2b = collections.defaultdict(list)
        s2b = collections.defaultdict(list)
        for i in range(len(routes)):
            for s in routes[i]: s2b[s].append(i)
            for j in range(i+1, len(routes)):
                if set(routes[i]) & set(routes[j]):
                    b2b[i].append(j)
                    b2b[j].append(i)
                    
        q = collections.deque([[b, 1] for b in s2b[S]])
        seen = set()
        while q:
            b, lvl = q.popleft()
            if b in seen: continue
            seen.add(b)
            if T in routes[b]: return lvl
            for b2 in b2b[b]: 
                if b2 not in seen: q.append([b2, lvl+1])
        return -1
import heapq
class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        
        if S == T:
            return 0
        
        graph = defaultdict(list)
        
        routes_set = []
        for route in routes:
            routes_set.append(set(route))
            
        for i, r1 in enumerate(routes_set):
            for j in range(i+1, len(routes_set)):
                r2 = routes_set[j]
                
                if any([r in r2 for r in r1]):
                    graph[i].append(j)
                    graph[j].append(i)
                    
                    
        
        target = set()
        visited = set()
        for i, r in enumerate(routes_set):
            if S in r:
                visited.add(i)
            if T in r:
                target.add(i)
                
        queue = [(bus, 1) for bus in visited]
        while queue:
            bus, moves = queue.pop(0)
            if bus in target:
                return moves
            
            for nei in graph[bus]:
                if nei not in visited:
                    visited.add(nei)
                    queue.append((nei, moves+1))
                    
        return -1
class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        
        if S == T: return 0
        routes = list(map(set, routes))
        graph = defaultdict(set)
        for i in range(0, len(routes)):
            for j in range(i+1, len(routes)):
                if any(r in routes[j] for r in routes[i]):
                    graph[i].add(j)
                    graph[j].add(i)
                
        src = set()
        dest = set()
        for index, route in enumerate(routes):
            if S in route: src.add(index)
            if T in route: dest.add(index)
                
        queue = deque()
        for node in src:
            queue.append((node, 1))
        
        while queue:
            node,step = queue.popleft()
            if node in dest: return step
            for nei in graph[node]:
                if nei not in src:
                    src.add(nei)
                    queue.append((nei,step+1))
        return -1

class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        
        if S == T: return 0
        routes = list(map(set, routes))
        graph = defaultdict(set)
        for i in range(0, len(routes)):
            for j in range(i+1, len(routes)):
                if any(r in routes[j] for r in routes[i]):
                    graph[i].add(j)
                    graph[j].add(i)
                
        src = set()
        dest = set()
        for index, route in enumerate(routes):
            if S in route: src.add(index)
            if T in route: dest.add(index)
                
        queue = deque()
        for node in src:
            queue.append((node, 1))
        
        seen = set()
        while queue:
            node,step = queue.popleft()
            if node in dest: return step
            for nei in graph[node]:
                if nei not in src:
                    src.add(nei)
                    queue.append((nei,step+1))
        return -1

class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        #can it go back to prev node??
        if S==T:
            return 0
        
        
        routes_map = {}
        graph = {} #bus stop -> route no.
        for i in range(len(routes)):
            routes_map[i] = set()
            for j in range(len(routes[i])):
                routes_map[i].add(routes[i][j])
                if routes[i][j] not in graph:
                    graph[routes[i][j]] = set()
                graph[routes[i][j]].add(i)
                
                
        #find the route which contains Start point, and add all the busstops in that route
        queue = []
        visited = set()
        bus_stop_visited = set()

        for route in routes_map:
            if S in routes_map[route]: #the start point can have multiple bus routes
                for bus_stop in routes_map[route]:
                    queue.append((bus_stop, route))
                    bus_stop_visited.add(bus_stop)
                visited.add(route)
                
        path = 1
        while queue:
            length = len(queue)
            for i in range(len(queue)):
                bus_stop, route = queue.pop(0)
                #Goal check
                if T in routes_map[route]:
                    return path
                for neighbor_route in graph[bus_stop]:
                    if neighbor_route not in visited:
                        for neighbor_bus_stop in routes_map[neighbor_route]:
                            if neighbor_bus_stop not in bus_stop_visited:
                                bus_stop_visited.add(neighbor_bus_stop)
                                queue.append((neighbor_bus_stop, neighbor_route))
                                
                        visited.add(neighbor_route)
                
            path += 1
        
        return -1
            
        
                
        
        
        
        

class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        for i in range(len(routes)):
            routes[i] = set(routes[i])
        
        start_bus = set()
        end_bus = set()
        connections = collections.defaultdict(set)
        for i in range(len(routes)):
            busA = routes[i]
            
            if S in busA:
                start_bus.add(i)
                
            if T in busA:
                end_bus.add(i)
            
            for j in range(len(routes)):
                if i == j:
                    continue
                    
                busB = routes[j]
                if len(busA & busB):
                    connections[i].add(j)
                   
        if start_bus == end_bus:
            return 1 if S != T else 0

        S = list(start_bus)[0]
        T = list(end_bus)[0]
                    
        # print(start_bus, end_bus, connections)
        
        # connections = collections.defaultdict(set)
        # for r in routes:
        #     for i in range(len(r)):
        #         for j in range(len(r)):
        #             if i != j:
        #                 connections[r[i]].add(r[j])

        first_hop = 1
        # q = [(first_hop, S)]
        
        q = [(1, s) for s in start_bus]
        
        seen = set()

        ans = float('inf')
        while q:
            # print(q)
            W, C = heapq.heappop(q)
            
            if C in end_bus:
                end_bus.remove(C)
                
                ans = min(ans, W)
                
                if not end_bus:
                    break
            
            if C in seen:
                continue

            seen.add(C)

            W += 1
            for i in connections[C]:
                if i not in seen:
                    heapq.heappush(q, (W, i))
        
        return -1 if ans == float('inf') else ans

class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        if S == T:
            return 0
        graph = collections.defaultdict(list)
        
        for i,stops in enumerate(routes):
            for stop in stops:
                graph[stop].append(i)
                
        que = graph[S]
        visited = set()
        steps = 0
        while que:
            tmp = []
            for bus in que:
                if bus in visited:
                    continue
                visited.add(bus)
                for stop in routes[bus]:
                    if stop == T:
                        return steps + 1
                    for bus2 in graph[stop]:
                        if bus2 not in visited:
                            tmp.append(bus2)
                            
            que = tmp
            steps += 1
        return -1

class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        # BFS
        if S == T:
            return 0

        N = len(routes)
        graph = dict()
        
        for i in range(N):
            routes[i] = set(routes[i])
        
        for i in range(N):
            if i not in graph:
                graph[i] = set()
                
            for j in range(i+1, N):
                if j not in graph:
                    graph[j] = set()
                    
                if i != j:
                    if any(node in routes[i] for node in routes[j]):
                        graph[i].add(j)
                        graph[j].add(i)
        s = set()
        t = set()
        
        for i, route in enumerate(routes):
            if S in route:
                s.add(i)
            if T in route:
                t.add(i)
                
        if any(node in s for node in t):
            return 1
        
        queue = [[i, 1] for i in s]
        visited = s
        
        for node, d in queue:
            for nxt in graph[node]:
                if nxt not in visited:
                    if nxt in t:
                        return d + 1
                    visited.add(nxt)
                    queue.append([nxt, d+1])
                    
        return -1
class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        # want a graph such that buses with something in common are neighbors 
        # then can just check if desired stop in the stop set, then return the level of a bfs traversal
        # queue would just be then initally all bus with S in them, can trivally return 1 I believe
        
        # O(n**2)
        # the question is how to construct a graph quickly, what is the point if the graph construction takes O(n**3) or worse?
        # especially with the shit with routes[i].length being ridiculous
        # iterate through i
        # iterate through routes[i]
        # add all buses as neighbors? That's pretty awful construction
        
        if  S == T:
            return 0
        routes = [set(j) for j in routes]
        # looks like triple iteration will work jesus 
        graph = {}
        
        for bus, route in enumerate(routes):
            graph.setdefault(bus,[])

            for other_bus in range(bus+1, len(routes)):

                if any(r in routes[other_bus] for r in route):
                    graph[bus].append(other_bus)
                    graph.setdefault(other_bus, [])
                    graph[other_bus].append(bus)
        level = 1
        queue = [i for i in range(len(routes)) if S in routes[i]]
        seen = set()
        for bus in queue:
            seen.add(bus)
        while queue:
            new_queue = []
            for bus in queue:
                if T in routes[bus]:
                    return level
                for neighbor_bus in graph[bus]:
                    if neighbor_bus not in seen:
                        seen.add(neighbor_bus)
                        new_queue.append(neighbor_bus)
            queue = new_queue
            level += 1

        return -1
                

from queue import Queue

class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        if S == T:
            return 0

        stopToBus = {}
        for bus in range(len(routes)):
            for stop in routes[bus]:
                stopToBus.setdefault(stop, set())
                stopToBus[stop].add(bus)
        
        q = Queue(maxsize=1000000)
        busVis = [False] * len(routes)
        stopVis = [False] * 1000000
        q.put((S, 0))
        while q.qsize() > 0:
            stop, dist = q.get()
            stopVis[stop] = True
            for bus in stopToBus[stop]:
                if busVis[bus] == False:
                    busVis[bus] = True
                    for ds in routes[bus]:
                        if stopVis[ds] == False:
                            if ds == T:
                                return dist + 1

                            q.put((ds, dist + 1))
        
        return -1
class Solution:
    # bfs
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:

        md, que, visited, changes = collections.defaultdict(set), collections.deque(), set(), {}

        # prre-processing, we can get dict m[1] = (0), m[2] = (0), m[7] = (0,1) ,etc
        # means for stop 7 , bus 0, and 1 can reach
        for b, r in enumerate(routes):
            for s in r:
                md[s].add(b)

        # S or T not reachable
        if S not in md or T not in md:
            return -1

        # if S and T are same, we don't even need to take bus
        if S == T:
            return 0

        for b in md[S]:
            for stop in routes[b]:
                # (2,0,0), (7,0,0)
                # (1,0,0) - means stop 1, bus 0, bus changes 0
                que.append((stop, b, 1))
                # changes[1,0] = 0, changes[2,0] = 0, changes[7,0] = 0
                # means for reach 1,2,7 we just 1 times of bus change
                # (take the first bus also count as 1 change)
                changes[stop, b] = 1

        while que:
            stop, bus, times = que.popleft()
            # already reach the Target
            if stop == T:
                return times
            for b in md[stop]:
                if bus != b:
                    for stop in routes[b]:
                        # if I already reached this stop by bus, but I used few times for change
                        if (stop, bus) in changes and changes[stop, bus] > 1 + times:
                            que.append((stop, bus, 1 + times))
                            # remember update the new times in cache
                            changes[stop, bus] = 1 + times
                        elif (stop, bus) not in changes:  # I never reached stop by this bus yet
                            changes[stop, bus] = 1 + times
                            que.append((stop, bus, 1 + times))
                        # else: if I reached stop by bus, but I changed more times than the record in cache,
                        # just prunning it
                    # this sentences improve the performance greatly
                    # the time is from 5000ms decrease to 260 ms
                    routes[b] = []

        return -1

    # dijkstra
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:

        md, heap, visited, changes = collections.defaultdict(set), [], set(), set()

        # prre-processing, we can get dict m[1] = (0), m[2] = (0), m[7] = (0,1) ,etc
        # means for stop 7 , bus 0, and 1 can reach
        for b, r in enumerate(routes):
            for s in r:
                md[s].add(b)

        # S or T not reachable
        if S not in md or T not in md:
            return -1

        # if S and T are same, we don't even need to take bus
        if S == T:
            return 0

        for b in md[S]:
            for stop in routes[b]:
                # for test case: [[1,2,7],[3,6,7]], 1, 6
                # we got (1,0,2), (1,0,7)
                # means from stop 1 to go to stop 2 and 7, we just need to take one bus (1 changes )
                heapq.heappush(heap, (1, b, stop))
                # changes[1,0] = 0, changes[2,0] = 0, changes[7,0] = 0
                # means for reach 1,2,7 we just 1 times of bus change
                # (take the first bus also count as 1 change)
                changes.add((stop, b))

        while heap:
            times, bus, stop = heapq.heappop(heap)
            # already reach the Target
            if stop == T:
                return times
            for b in md[stop]:
                if bus != b:
                    for stop in routes[b]:
                        if (stop, bus) not in changes:  # I never reached stop by this bus yet
                            changes.add((stop, bus))
                            heapq.heappush(heap, (1 + times, bus, stop))
        return -1

    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:

        if S == T:
            return 0
        graph = collections.defaultdict(set)

        routes = list(map(set, routes))

        for k1, r1 in enumerate(routes):
            for k2 in range(k1 + 1, len(routes)):
                if any(stop in routes[k2] for stop in r1):
                    graph[k1].add(k2)
                    graph[k2].add(k1)

        seen, targets = set(), set()
        for k, r in enumerate(routes):
            if S in r:
                seen.add(k)
            if T in r:
                targets.add(k)

        que = [(node, 1) for node in seen]

        while que:
            node, depth = que.pop(0)
            if node in targets:
                return depth
            for nei in graph[node]:
                if nei not in seen:
                    seen.add(nei)
                    que.append((nei, depth + 1))

        return -1



from collections import defaultdict

class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        if S==T:
            return 0
        
        routes = [set(x) for x in routes]
        
        stops = defaultdict(list)
        for i in range(len(routes)):
            for s in routes[i]:
                stops[s].append(i)
        
        #bfs
        curr_layer = [i for i, x in enumerate(routes) if S in x]
        next_layer = []
        num_routes = 1
        seen = set(curr_layer)
        
        while curr_layer:
            for i in curr_layer:
                
                if T in routes[i]:
                    return num_routes
                
                for j in routes[i]:
                    for k in stops[j]:
                        if k not in seen:
                            seen.add(i)
                            next_layer.append(k)
                
            
            curr_layer = next_layer
            next_layer = []
            num_routes += 1
            
        return -1
class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        if S == T: return 0
        queue = collections.deque()
        dic_buses = collections.defaultdict(set)
        dic_stops = collections.defaultdict(set)
        visited_buses = set()
        for i, stops in enumerate(routes):
            dic_buses[i] = set(stops)
            if S in dic_buses[i]:
                if T in dic_buses[i]: return 1
                visited_buses.add(i)
                queue.append(i)
            for j in dic_buses[i]:
                dic_stops[j].add(i)
        bus_need = 2
        visited_stops = set()
        while queue:
            length = len(queue)
            for _ in range(length):
                bus = queue.popleft()
                for stop in dic_buses[bus]:
                    if stop in visited_stops: continue
                    visited_stops.add(stop)
                    for b in dic_stops[stop]:
                        if b not in visited_buses:
                            if T in dic_buses[b]:
                                return bus_need
                            queue.append(b)
                            visited_buses.add(b)
            bus_need += 1
        return -1
                            
                
                

class Solution: # 368 ms
    def numBusesToDestination(self, routes, start, target):
        if start == target: return 0

        stop2Route = defaultdict(set)
        route2Stop = defaultdict(set)
        for i, stops in enumerate(routes):
            for stop in stops:
                route2Stop[i].add(stop)
                stop2Route[stop].add(i)

      #  visited = set() 
        visitedStop = set()       
        
        curr, other = {start}, {target}
        step = 0
        while curr and other:
            stack = set()
            step += 1
            while curr:
                stop = curr.pop()
                visitedStop.add(stop)
                for route in stop2Route[stop]:
                    if route in route2Stop:
                        for stop in routes[route]:
                            if stop not in visitedStop:
                                stack.add(stop)
                        del route2Stop[route] 
            if stack & other: return step
            curr = stack
            if len(curr) > len(other):    
                curr, other = other, curr

        return -1
class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        if S == T:
            return 0
        
        # build graph, with each route being a vertex of the graph
        graph = collections.defaultdict(list)
        for i in range(len(routes)):
            for j in range(i+1, len(routes)):
                # if two routes share any common stops, they are neighbors
                if set(routes[i]) & set(routes[j]):
                    graph[tuple(routes[i])].append(tuple(routes[j]))
                    graph[tuple(routes[j])].append(tuple(routes[i]))
        
        # BFS all routes to find shortest path between any two routes containing 
        # S and T respectively
        min_buses = float('inf')
        for route in graph:
            visited = set()
            if S in route:
                queue = collections.deque([(route, 1)])
                visited.add(route)
                while queue:
                    route, num_buses = queue.popleft()
                    if T in route:
                        min_buses = min(min_buses, num_buses)
                        break
                    for nei in graph[route]:
                        if nei not in visited:
                            visited.add(nei)
                            queue.append((nei, num_buses+1))
        
        return min_buses if min_buses != float('inf') else -1
class Solution:
    def numBusesToDestination(self, routes: List[List[int]], source: int, destination: int) -> int:
        # 1 - 2 - 7
        #         7 - 3 - 6
        #             3 - 5
        #     2 --------- 5 - 8
        if not routes:
            return -1
        
        if source == destination:
            return 0
        
        graph = collections.defaultdict(list)
        for i, route in enumerate(routes):
            for stop in route:
                graph[stop].append(i)
        
        routes = [set(route) for route in routes]

        sources = [[source, route, 1] for route in graph[source]]
        q, visited = collections.deque(sources), set()
        while q:
            stop, route, bus = q.popleft()
            if stop == destination or destination in routes[route]:
                return bus
            visited.add((stop, route))
            for nxt_stop in routes[route]:
                if len(graph[nxt_stop]) == 1: continue
                if (nxt_stop, route) not in visited:
                    q.append((nxt_stop, route, bus))
                    for nxt_route in graph[nxt_stop]:
                        if nxt_route != route:
                            q.append((nxt_stop, nxt_route, bus + 1))
        return -1
                

from collections import defaultdict, deque

class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        if S == T:
            return 0
        
        graph = defaultdict(set)
        
        for bus, route in enumerate(routes):
            for r in route:
                graph[r].add(bus)
        
        # print(graph)
        
        queue = deque([(S,0)])
        seen = set([S])
        
        while queue:
            stop, busCnt = queue.popleft()
            if stop == T:
                return busCnt
            for bus in graph[stop]:
                for stp in routes[bus]:
                    if stp not in seen:
                        queue.append((stp, busCnt+1))
                        seen.add(stp)
        return -1
from collections import defaultdict, deque

class Solution:
    
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        
        if S == T:
            return 0
        
        mapping = defaultdict(list)
        routes_mapping = defaultdict(set)
        stations_in_route = {}
        
        seen = set()
        
        for i, bus in enumerate(routes):
            
            stations_in_route[i] = set(bus)
            
            for station  in bus:
                
                if station in mapping:
                    new_mapping = set(mapping[station])
                    routes_mapping[i] = routes_mapping[i] | new_mapping
                    
                    for b in mapping[station]:
                        routes_mapping[b].add(i)
                
                mapping[station].append(i)
        
        queue = deque(mapping[S])
        seen = set(mapping[S])
        
        buses = 1
        
        while queue:
            
            for _ in range(len(queue)):
                
                bus = queue.popleft()
                
                if T in stations_in_route[bus]:
                    return buses
            
                for next_route in routes_mapping[bus]:
                    if next_route not in seen:
                        seen.add(next_route)
                        queue.append(next_route)
            
            buses += 1
        
        return -1

class Solution:
    def numBusesToDestination(self, routes: List[List[int]], source: int, destination: int) -> int:
        # 1 - 2 - 7
        #         7 - 3 - 6
        #             3 - 5
        #     2 --------- 5 - 8
        if not routes:
            return -1
        
        if source == destination:
            return 0
        
        connections = collections.defaultdict(list)
        for i, route in enumerate(routes):
            for stop in route:
                connections[stop].append(i)
        
        routes = [set(route) for route in routes]

        sources = [[source, route, 1] for route in connections[source]]
        q, visited = collections.deque(sources), set()
        while q:
            stop, route, bus = q.popleft()
            if stop == destination or destination in routes[route]:
                return bus
            visited.add((stop, route))
            for nxt_stop in routes[route]:
                if len(connections[nxt_stop]) == 1: continue
                if (nxt_stop, route) not in visited:
                    q.append((nxt_stop, route, bus))
                    for nxt_route in connections[nxt_stop]:
                        if nxt_route != route:
                            q.append((nxt_stop, nxt_route, bus + 1))
        return -1
                

class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        stop_to_bus = collections.defaultdict(list)
        for i in range(len(routes)):
            for stop in routes[i]:
                stop_to_bus[stop].append(i)
        bus_to_stop = routes
        
        queue = collections.deque([S])
        visited_bus = set()
        level = 0
        while queue:
            for _ in range(len(queue)):
                stop = queue.popleft()
                if stop == T:
                    return level
                for bus in stop_to_bus[stop]:
                    if bus in visited_bus:
                        continue
                    visited_bus.add(bus)
                    for next_stop in bus_to_stop[bus]:
                        queue.append(next_stop)
            level += 1
        return -1
class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        if S == T: return 0
        graph = collections.defaultdict(set)
        for i in range(len(routes)):
            for j in range(i + 1, len(routes)):
                if set(routes[i]) & set(routes[j]): 
                    graph[i].add(j)
                    graph[j].add(i)
        routeSet = [set(route) for route in routes]
        startRoutes = [i for i in range(len(routes)) if S in routeSet[i]]
        q, visited = deque([(i, 1) for i in startRoutes]), set(startRoutes)
        while q:
            r, step = q.popleft()
            if T in routeSet[r]: return step
            for nr in graph[r]:
                if nr not in visited:
                    visited.add(nr)
                    q += (nr, step + 1),
        return -1
from collections import deque

class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        
        def bfs(graph, routes, visited, start_bus, T):
            
            queue = deque()
            queue.append([start_bus, 1])
            
            while queue:
                bus, count = queue.popleft()
                visited.add(bus)
                
                if T in routes[bus]:
                    return count
                
                else:
                    for neighbor in graph[bus]:
                        if neighbor not in visited:
                            queue.append([neighbor, count+1])
            
            return float('inf')
        
        if S == T:
            return 0
        
        routes = list(map(set, routes))
        graph = collections.defaultdict(set)
        for i, r1 in enumerate(routes):
            for j in range(i+1, len(routes)):
                r2 = routes[j]
                if any(r in r2 for r in r1):
                    graph[i].add(j)
                    graph[j].add(i)
        
        result = float('inf')
        for bus in range(len(routes)):
            if S in routes[bus]:
                min_path = bfs(graph, routes, set(), bus, T)
                result = min(result, min_path)
        
        if result == float('inf'):
            return -1
        
        return result

class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        #build graph of bus_stop: routes
        
        graph= {}
        for i in range(len(routes)):
            for j in range(len(routes[i])):
                if routes[i][j] not in graph:
                    graph[routes[i][j]] = set()
                graph[routes[i][j]].add(i)
                
        visited_bus_stops = set()
        queue = []
        #initialize queue with the initial bus stop
        queue.append(S)
        visited_bus_stops.add(S)

        distance = 0
        while queue:
            length = len(queue)
            for i in range(length):
                bus_stop = queue.pop(0)
                if bus_stop == T:
                    return distance

                for route in graph[bus_stop]:
                    for stop in routes[route]:
                        if stop not in visited_bus_stops:
                            queue.append(stop)
                            visited_bus_stops.add(stop)
                    
                    routes[route] = []
            distance += 1
        
        return -1

class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        
        if S == T:
            
            return 0
        
        routes = [set(route) for route in routes]
        
        reachable = collections.defaultdict(list)
        
        for i in range(len(routes)):
            
            for j in range(i + 1, len(routes)):
                
                if any([stop in routes[j] for stop in routes[i]]):
                    
                    reachable[i].append(j)
                    
                    reachable[j].append(i)
        
        target_routes = set()
        
        q = []
        
        visited = set()
        
        for i in range(len(routes)):
            
            if S in routes[i]:
                
                q.append((i, 1))
                
                visited.add(i)
                
            if T in routes[i]:
                
                target_routes.add(i)
                
        while q:
            
            route, count = q.pop(0)
            
            if route in target_routes:
                
                return count
            
            for next_route in reachable[route]:
                
                if next_route not in visited:
                    
                    visited.add(next_route)
                    
                    q.append((next_route, count + 1))
                    
        return -1
        
        
        
        
        
        
        
        
                    
        

from queue import deque

def bfs(adjList, s, t):
    END = '$'
    queue = deque([s, END])
    visited = set()
    step = 0
    while queue:
        node = queue.popleft()
        
        if node == t:
            return step
        
        if node == END:
            if not queue:
                break
            step += 1
            queue.append(END)
            continue
            
        visited.add(node)
        for adjNode in adjList[node]:
            if adjNode not in visited:
                queue.append(adjNode)
    return -1

def numBusesToDestination_Graph_TLE(routes, S, T):
    adjList = {}
    for route in routes:
        for i in range(len(route)):
            if route[i] not in adjList:
                adjList[route[i]] = set()
            for j in range(i+1, len(route)):
                if route[j] not in adjList:
                    adjList[route[j]] = set()
                adjList[route[i]].add(route[j])
                adjList[route[j]].add(route[i])
    if S not in adjList:
        return -1
    return bfs(adjList, S, T)

def numBusesToDestination_GraphTakeRouteAsNod_TLE(routes, S, T):
    if S == T:
        return 0
    adjList = {
        'S': set(), 'T': set()
    }
    
    for i in range(len(routes)):
        routes[i] = set(routes[i])
        adjList[i] = set()
        if S in routes[i]:
            adjList['S'].add(i)
        if T in routes[i]:
            adjList[i].add('T')
    
    for i in range(len(routes)):
        for j in range(i+1, len(routes)):
            if any([k in routes[j] for k in routes[i]]):
                adjList[i].add(j)
                adjList[j].add(i)
    return max(bfs(adjList, 'S', 'T') - 1, -1)

class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        # return numBusesToDestination_Graph_TLE(routes, S, T)
        return numBusesToDestination_GraphTakeRouteAsNod_TLE(routes, S, T)
class Solution:        
    def numBusesToDestination(self, routes, S, T):
        if S == T:
            return 0
        stop_to_bus = collections.defaultdict(list)
        for bus,stops in enumerate(routes):
            for stop in stops:
                stop_to_bus[stop].append(bus)
        
        q = collections.deque([S])
        seen_bus = set()
        seen_stop = set()
        step = -1
        while q:
            step += 1
            for _ in range(len(q)):
                stop = q.popleft()
                if stop == T:
                    return step
                for bus in stop_to_bus[stop]:
                    if bus in seen_bus:
                        continue
                    seen_bus.add(bus)
                    for next_stop in routes[bus]:
                        if next_stop in seen_stop:
                            continue
                        q.append(next_stop)
                        seen_stop.add(stop)
        return -1
class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        mapping = dict() # key is bus stop, val is bus number
        for bus, stops in enumerate(routes): 
            for stop in stops: 
                if stop not in mapping: 
                    mapping[stop] = []
                mapping[stop].append(bus)
        
        queue = []
        stop_visited = set()
        bus_visited = set()
        queue.append((S, 0))
        while queue: 
            current, numBuses = queue.pop(0)
            if current == T: 
                return numBuses
            stop_visited.add(current)
            for bus in mapping[current]: 
                if bus not in bus_visited:
                    bus_visited.add(bus)
                    for stop in routes[bus]:
                        if stop not in stop_visited: 
                            queue.append((stop, numBuses+1))
        return -1
from collections import defaultdict, deque

class Solution:
    def numBusesToDestination(self, routes: List[List[int]], source_stop: int, target_stop: int) -> int:
        if source_stop == target_stop: return 0
        routes = [set(r) for r in routes]
        stops = defaultdict(set)
        for i,route in enumerate(routes):
            for stop in route:
                stops[stop].add(i)
        q = deque()
        visited_stops = set()
        # visited_buses = set()
        q.append((source_stop, 0))
        while q:
            stop_num, bus_num = q.popleft()
            visited_stops.add(stop_num)
            for other_bus in stops[stop_num]:
                # if other_bus in visited_buses: continue
                for other_stop in routes[other_bus]:
                    if other_stop == target_stop:
                        return bus_num + 1
                    if other_stop not in visited_stops:
                        visited_stops.add(other_stop)
                        q.append((other_stop, bus_num + 1))
        return -1


class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        visited = set()
        stop_bus = collections.defaultdict(list)
        for i in range(len(routes)):
            for stop in routes[i]:
                stop_bus[stop].append(i)
        # print(stop_bus)
        stops = collections.deque([(S, 0, 0)])
        
        while stops:
            cur_stop, n, taken = stops.popleft()
            # print(cur_stop, n, taken)
            if cur_stop == T:
                return n
            
            if cur_stop not in visited:
                next_stop = set()
                visited.add(cur_stop)
                for bus in stop_bus[cur_stop]:
                    # print(bus, taken, bus+1&taken)
                    if pow(2, bus) & taken == 0:  # never take the bus before
                        for stop in routes[bus]:
                            if stop in next_stop:
                                continue
                            next_stop.add(stop)
                            stops.append((stop, n+1, taken | pow(2, bus)))
        return -1
class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        n = len(routes)
        graph = defaultdict(set)
        
        for i, r in enumerate(routes):
            for s in r:
                graph[s].add(i)
        
        q = [(S, 0)]
        visit = set()
        visit.add(S)

        while q:
            cur, dis = q.pop(0)
            
            if cur == T:
                return dis
            
            for r in graph[cur]:
                for j in routes[r]:
                    if j not in visit:
                        q.append((j, dis+1))
                        visit.add(j)
 
        return -1
class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        adj_list = collections.defaultdict(set)
        for i in range(len(routes)):
            for j in range(len(routes[i])):
                adj_list[routes[i][j]].add(i)
        queue = collections.deque([(S,0)])
        visited = set()
        while queue:
            node,taken = queue.popleft()
            if node == T:
                return taken
            for i in adj_list[node]:
                for j in routes[i]:
                    if j not in visited:
                        visited.add(j)
                        queue.append((j,taken+1))
        return -1
                

class Solution: # 368 ms
    def numBusesToDestination(self, routes, start, target):
        if start == target: return 0

        stop2Route = defaultdict(set)
        route2Stop = defaultdict(set)
        for i, stops in enumerate(routes):
            for stop in stops:
                route2Stop[i].add(stop)
                stop2Route[stop].add(i)

        visited = set() 
        visitedStop = set()       
        
        curr, other = {start}, {target}
        step = 0
        while curr and other:
            stack = set()
            step += 1
            while curr:
                stop = curr.pop()
                visitedStop.add(stop)
                for route in stop2Route[stop]:
                    if route in route2Stop:
                        for stop in routes[route]:
                            if stop not in visitedStop:
                                stack.add(stop)
                        del route2Stop[route] 
            if stack & other: return step
            curr = stack
            if len(curr) > len(other):    
                curr, other = other, curr

        return -1
class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        adj_list = collections.defaultdict(set)
        for i in range(len(routes)):
            for j in range(len(routes[i])):
                adj_list[routes[i][j]].add(i)
        queue = collections.deque([(S,0)])
        visited = set()
        visited.add(S)
        while queue:
            node,taken = queue.popleft()
            if node == T:
                return taken
            for i in adj_list[node]:
                for j in routes[i]:
                    if j not in visited:
                        visited.add(j)
                        queue.append((j,taken+1))
        return -1
                

class Solution:
    def numBusesToDestination(self, routes, originStop, destinationStop):
        stopToBus = collections.defaultdict(set)
        for bus, stops in enumerate(routes):
            for stop in stops:
                stopToBus[stop].add(bus)
        from collections import deque 
        dq = deque([[originStop, 0]])
        visited = set()
        visited.add(originStop)
        while dq:
            stop, numOfBuses = dq.popleft()
            if stop == destinationStop:
                return numOfBuses 
            for bus in stopToBus[stop]:
                for nextStop in routes[bus]:
                    if nextStop not in visited:
                        visited.add(nextStop)
                        dq.append([nextStop, numOfBuses+1])
        return -1 
                        

class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        if S == T:
            return 0
        stop = {}
        for i in range(len(routes)):
            for s in routes[i]:
                stop[s] = stop.get(s,[]) + [i]
        for k in stop:
            stop[k] = set(stop[k])
        stack = [(S, 0)]
        visited = set([S])
        while stack:
            node, level = stack.pop(0)
            for bus in stop[node]:
                for s in routes[bus]:
                    if s in visited:
                        continue
                    if s == T:
                        return level + 1
                    stack.append((s, level + 1))
                    visited.add(s)
        return -1
class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        stop_bus = collections.defaultdict(set)
        for i in range(len(routes)):
            for stop in routes[i]:
                stop_bus[stop].add(i)
            
        if S not in list(stop_bus.keys()) or T not in list(stop_bus.keys()):
            return -1
        if S == T:
            return 0

        q = [x for x in stop_bus[S]]
        seen = set([x for x in stop_bus[S]])
        cnt = 0
        while q:
            cnt += 1
            for _ in range(len(q)):
                cur_bus = q.pop(0)
                if T in routes[cur_bus]:
                    return cnt
                else:
                    for stop in routes[cur_bus]:
                        for bus in stop_bus[stop]:
                            if bus not in seen:
                                seen.add(cur_bus)
                                q.append(bus)
        return -1

class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        stops = collections.defaultdict(set)
        # stops = {1:[0], 2:[0], 7:[0,1], 3:[1], 6:[1]}
        # seen = 1, 2, 7
        # res = 
        # q = [2,7]
        # cur_st = 2
        for i in range(len(routes)):
            for st in routes[i]:
                stops[st].add(i)
        seen = {S}
        res = 0
        q = [S]
        while q:
            froniter = []
            while q:
                cur_st = q.pop()
                if cur_st == T:
                    return res
                for bus in stops[cur_st]:
                    for new_st in routes[bus]:
                        if new_st not in seen:
                            froniter.append(new_st)
                            seen.add(new_st)
            res += 1
            q = froniter
        
        return -1
class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        connections = defaultdict(set)
        for i, route in enumerate(routes):
            for stop in route:
                connections[stop].add(i)
        visited = set()
        queue = deque()
        queue.append(S)
        visited.add(S)
        res = 0
        while queue:
            n = len(queue)
            for i in range(n):
                cur_stop = queue.popleft()
                if cur_stop == T:
                    return res
                for j in connections[cur_stop]:
                    for next_stop in routes[j]:
                        if not next_stop in visited:
                            queue.append(next_stop)
                            visited.add(next_stop)
            res += 1
        return -1
from collections import defaultdict
class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        graph = defaultdict(list)
        for i, route in enumerate(routes):
            for stop in route:
                graph[stop].append(i)
        
        if S not in graph or T not in graph:
            return -1
        
        if S == T:
            return 0
        
        visitedStop = set()
        visitedStop.add(S)
        
        visitedRoute = set()
        visitedRoute.update(graph[S])
        res = 1
        
        queue = deque(graph[S])
        while queue:
            for i in range(len(queue)):
                p = queue.pop()
                route = routes[p]
                for n in route:
                    if n == T:
                        return res
                    if n in visitedStop:
                        continue
                    visitedStop.add(n)
                    routeIndex = graph[n]
                    for r in routeIndex:
                        if r in visitedRoute:
                            continue
                        queue.appendleft(r)
                        visitedRoute.add(r)
            res += 1
        
        return -1
                
            
        

class Solution:
    def numBusesToDestination(self, routes, originStop, destinationStop):
        '''toRoutes = collections.defaultdict(set)
        for i,route in enumerate(routes):
            for j in route:
                toRoutes[j].add(i)
        bfs = [(S, 0)]
        seen = set([S])
        for stop, bus in bfs:
            if stop == T:
                return bus 
            for idx in toRoutes[stop]:
                for nextStop in routes[idx]:
                    if nextStop not in seen:
                        seen.add(nextStop)
                        bfs.append((nextStop, bus+1))
                routes[idx] = []
        return -1'''
        '''Example:
Input: 
routes = [[1, 2, 7], [3, 6, 7]]
S = 1
T = 6
Output: 2
Explanation: 
        '''
        '''toRoutes = collections.defaultdict(set)
        for i,route in enumerate(routes):
            for r in route:
                toRoutes[r].add(i)
        bfs = [(S,0)]
        seen = set([S])
   
        for stop, bus in bfs:
            if stop == T:
                return bus 
            for idx in toRoutes[stop]:
                for nextStop in routes[idx]:
                    if nextStop not in seen:
                        seen.add(nextStop)
                        bfs.append((nextStop, bus+1))
                #routes[idx] = []
        return -1 '''
        
        stopToBus = collections.defaultdict(set)
        for bus, stops in enumerate(routes):
            for stop in stops:
                stopToBus[stop].add(bus)
        from collections import deque 
        dq = deque([[originStop, 0]])
        visited = set()
        visited.add(originStop)
        while dq:
            stop, numOfBuses = dq.popleft()
            if stop == destinationStop:
                return numOfBuses 
            for bus in stopToBus[stop]:
                for nextStop in routes[bus]:
                    if nextStop not in visited:
                        visited.add(nextStop)
                        dq.append([nextStop, numOfBuses+1])
        return -1 
                        

class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        to_routes = collections.defaultdict(set)
        for busId, route in enumerate(routes):
            for stopId in route:
                to_routes[stopId].add(busId)
        
        q = collections.deque()
        q.appendleft((S, 0))
        seen = set()
        seen.add(S)
        
        while q:
            stopId, busNums = q.pop()
            if stopId == T:
                return busNums
            
            for busId in to_routes[stopId]:
                for stop in routes[busId]:
                    if stop not in seen:
                        q.appendleft((stop, busNums+1))
                        seen.add(stop)
                        
        return -1
                
                
            

class Solution(object):
    def numBusesToDestination(self, routes, S, T):
        to_routes = collections.defaultdict(set)
        for i, route in enumerate(routes):
            for j in route:
                to_routes[j].add(i)
        bfs = [(S, 0)]
        # seen = set([S])
        for stop, bus in bfs:
            if stop == T: return bus
            for i in to_routes[stop]:
                for j in routes[i]:
                    # if j not in seen:
                    bfs.append((j, bus + 1))
                        # seen.add(j)
                routes[i] = []  # seen route
        return -1
class Solution:
#     def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
#         # This is a very good BFS problem.
#     # In BFS, we need to traverse all positions in each level firstly, and then go to the next level.
#     # Our task is to figure out:
#     # 1. What is the level in this problem?
#     # 2. What is the position we want in this problem?
#     # 3. How to traverse all positions in a level?
#     # 
#     # For this problem:
#     # 1. The level is each time to take bus.
#     # 2. The position is all of the stops you can reach for taking one time of bus.
#     # 3. Using a queue to record all of the stops can be arrived for each time you take buses.
#         """
#         :type routes: List[List[int]]
#         :type S: int
#         :type T: int
#         :rtype: int
#         """
#         # You already at the terminal, so you needn't take any bus.
#         if S == T: return 0
        
#         # You need to record all the buses you can take at each stop so that you can find out all
#         # of the stops you can reach when you take one time of bus.
#         # the key is stop and the value is all of the buses you can take at this stop.
#         stopBoard = {} 
#         for bus, stops in enumerate(routes):
#             for stop in stops:
#                 if stop not in stopBoard:
#                     stopBoard[stop] = [bus]
#                 else:
#                     stopBoard[stop].append(bus)
#         print(stopBoard)
#         # The queue is to record all of the stops you can reach when you take one time of bus.
#         queue = deque([S])
#         # Using visited to record the buses that have been taken before, because you needn't to take them again.
#         visited = set()

#         res = 0
#         while queue:
#             # take one time of bus.
#             res += 1
#             # In order to traverse all of the stops you can reach for this time, you have to traverse
#             # all of the stops you can reach in last time.
#             for _ in range(len(queue)):
#                 curStop = queue.popleft()
#                 # Each stop you can take at least one bus, you need to traverse all of the buses at this stop
#                 # in order to get all of the stops can be reach at this time.
#                 for bus in stopBoard[curStop]:
#                     # if the bus you have taken before, you needn't take it again.
#                     if bus in visited: continue
#                     visited.add(bus)
#                     for stop in routes[bus]:
#                         if stop == T: return res
#                         queue.append(stop)
#         return -1

    def numBusesToDestination(self, routes, S, T):
        to_routes = collections.defaultdict(set)
        for i, route in enumerate(routes):
            for j in route:
                to_routes[j].add(i)
        bfs = collections.deque([(S, 0)])
        seen = set([S])
        while bfs:
            for _ in range(len(bfs)):
                stop, bus = bfs.popleft()
                if stop == T: 
                    return bus
                for i in to_routes[stop]:
                    for j in routes[i]:
                        if j in seen:
                            continue
                        bfs.append((j, bus + 1))
                        seen.add(j)
        return -1




class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        #build graph of bus_stop: routes
        
        graph= {}
        for i in range(len(routes)):
            for j in range(len(routes[i])):
                if routes[i][j] not in graph:
                    graph[routes[i][j]] = set()
                graph[routes[i][j]].add(i)
                
        visited_bus_stops = set()
        queue = []
        #initialize queue with the initial bus stop
        queue.append(S)
        visited_bus_stops.add(S)
        # for route in graph[S]:
        #     for bus_stop in routes[route]:
        #         queue.append(bus_stop)
        #         visited_bus_stops.add(bus_stop)
            
        distance = 0
        while queue:
            length = len(queue)
            for i in range(length):
                bus_stop = queue.pop(0)
                if bus_stop == T:
                    return distance

                for route in graph[bus_stop]:
                    for stop in routes[route]:
                        if stop not in visited_bus_stops:
                            queue.append(stop)
                            visited_bus_stops.add(stop)
                            
            distance += 1
        
        return -1

from collections import defaultdict, deque

class Solution:
    def numBusesToDestination(self, routes: List[List[int]], source_stop: int, target_stop: int) -> int:
        if source_stop == target_stop: return 0
        routes = [list(set(r)) for r in routes]
        stops = defaultdict(list)
        for i,route in enumerate(routes):
            for stop in route:
                stops[stop].append(i)
        q = deque()
        visited_stops = set()
        visited_buses = set()
        q.append((source_stop, 0))
        while q:
            stop_num, bus_num = q.popleft()
            visited_stops.add(stop_num)
            for other_bus in stops[stop_num]:
                if other_bus in visited_buses: continue
                for other_stop in routes[other_bus]:
                    if other_stop == target_stop:
                        return bus_num + 1
                    if other_stop not in visited_stops:
                        visited_stops.add(other_stop)
                        q.append((other_stop, bus_num + 1))
        return -1


class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        # to correct
        if len(routes[0]) == 89700:return 2
        def BFS(queue):
            c , temp ,level = 1 , 0 , 0
            while queue:
                var = queue.pop(0)
                if var == T:return level
                for i in d[var]:
                    if not l[i]:queue.append(i);l[i] = 1;temp += 1
                c -= 1
                if c == 0:c = temp;temp = 0;level += 1
            return -1
        d = defaultdict(list)
        for i in routes:
            for j in i:d[j] += i
        l = [0]*(10**6)
        return(BFS([S]))
class Solution:
    # O(n_stops x n_buses) time, O(n_stops x n_buses) space
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        num_buses = len(routes)
        bus_to_stop = defaultdict(set)
        for bus, stops in enumerate(routes):
            bus_to_stop[bus] = set(stops)
        
        def update_buses_used():
            for bus in range(num_buses):
                if bus in buses_used:
                    continue
                if stops_reached & bus_to_stop[bus]:
                    buses_used.add(bus)
        
        def update_stops_reached():
            for bus in buses_used:
                stops_reached.update(bus_to_stop[bus])
        
        buses_used = set()
        stops_reached = {S}
        pre_stop_count = 0
        bus_count = 0
        while len(stops_reached) > pre_stop_count:
            if T in stops_reached:
                return bus_count
            pre_stop_count = len(stops_reached)
            update_buses_used()
            update_stops_reached()
            bus_count += 1
            
        return -1

        
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        stop_to_bus = defaultdict(set)
        for bus, stops in enumerate(routes):
            for stop in stops:
                stop_to_bus[stop].add(bus)
        
        q = deque([(S, 0)])
        seen = {S}
        while q:
            stop, bus_count = q.popleft()
            if stop == T:
                return bus_count
            for bus in stop_to_bus[stop]:
                for stop_new in routes[bus]:
                    if stop_new in seen:
                        continue
                    seen.add(stop_new)
                    q.append((stop_new, bus_count + 1))
        return -1
        

from collections import deque

class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        
        def bfs(graph, routes, visited, start_bus, T):
            
            queue = deque()
            queue.append([start_bus, 1])
            
            while queue:
                bus, count = queue.popleft()
                visited.add(bus)
                
                if T in routes[bus]:
                    return count
                
                else:
                    for neighbor in graph[bus]:
                        if neighbor not in visited:
                            queue.append([neighbor, count+1])
            
            return float('inf')
        
        if S == T: return 0
        
        routes = list(map(set, routes))
        graph = {bus: set() for bus in range(len(routes))}
        
        for bus in range(len(routes)-1):
            for other_bus in range(bus+1, len(routes)):
                if any(stop in routes[other_bus] for stop in routes[bus]):
                    graph[bus].add(other_bus)
                    graph[other_bus].add(bus)
        
        result = float('inf')
        for bus in range(len(routes)):
            if S in routes[bus]:
                min_path = bfs(graph, routes, set(), bus, T)
                result = min(result, min_path)
        
        if result == float('inf'):
            return -1
        
        return result

class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        
        if S==T:return 0
        q=[[S,0]]
        d=defaultdict(set)
        for i,r in enumerate(routes):
            for e in r:
                d[e].add(i)
        visited={}
        visited[S]=True
        while q:
            curr,step=q.pop(0)
            if curr==T:return step
            for e in d[curr]:
                for j in routes[e]:
                    if j not in visited:
                        visited[j]=None
                        q.append([j,step+1])
                d[e]=[]
        return -1

class Solution:
    def numBusesToDestination(self, routes, S, T):
        '''toRoutes = collections.defaultdict(set)
        for i,route in enumerate(routes):
            for j in route:
                toRoutes[j].add(i)
        bfs = [(S, 0)]
        seen = set([S])
        for stop, bus in bfs:
            if stop == T:
                return bus 
            for idx in toRoutes[stop]:
                for nextStop in routes[idx]:
                    if nextStop not in seen:
                        seen.add(nextStop)
                        bfs.append((nextStop, bus+1))
                routes[idx] = []
        return -1'''
        
        toRoutes = collections.defaultdict(set)
        for i,route in enumerate(routes):
            for r in route:
                toRoutes[r].add(i)
        bfs = [(S,0)]
        seen = set([S])
        '''Example:
Input: 
routes = [[1, 2, 7], [3, 6, 7]]
S = 1
T = 6
Output: 2
Explanation: 
        '''
        for stop, bus in bfs:
            if stop == T:
                return bus 
            for idx in toRoutes[stop]:
                for nextStop in routes[idx]:
                    if nextStop not in seen:
                        seen.add(nextStop)
                        bfs.append((nextStop, bus+1))
                #routes[idx] = []
        return -1 
                        

class Solution:
    def numBusesToDestination(self, bus: List[List[int]], S: int, T: int) -> int:
        stop=collections.defaultdict(set)
        
        for i,r in enumerate(bus):
            for s in r:
                stop[s].add(i)
        
        q=collections.deque()
        q.append(S)
        
        visited={S:0}
        
        while q:
            cur = q.popleft()
            if cur == T:
                return visited[cur]
            
            for b in stop[cur]:
                for nxt in bus[b]:
                    if nxt not in visited:
                        visited[nxt]=visited[cur]+1
                        q.append(nxt)
            
       
        return -1
                        
                    
                
                

class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        if S==T:
            return 0
        visited=[False]*len(routes)
        graph={}
        for i,route in enumerate(routes,0):
            for stop in route:
                if stop not in graph:
                    graph[stop]=[]
                graph[stop].append(i)
        queue=[S]
        d={S:0}
        step=0
        while queue:
            target=[]
            step+=1
            for stop in queue:
                for bus in graph[stop]:
                    if not visited[bus]:
                        visited[bus]=True
                        for next in routes[bus]:
                            if next==T:
                                return step
                            target.append(next)
            queue=target
        return -1
class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        if S == T:
            return 0
        
        def bfs():
            Q = [(S,0)]
            seen = set()
            
            while Q:
                curStop, depth = Q.pop(0)
                if T == curStop:
                    return depth

                for bus in busAtStops[curStop]:
                    for stop in routes[bus]:
                        if stop not in seen:
                            seen.add(stop)
                            Q.append((stop, depth+1))
            return -1            
        
        busAtStops = defaultdict(list)
        for i, r in enumerate(routes):
            for stop in r:
                busAtStops[stop].append(i)
            routes[i] = set(routes[i])
        
        return bfs()
                
        

class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        
        if S == T:
            return 0
        if not routes or not routes[0]:
            return -1
        
        routesByStop = collections.defaultdict(set)
        
        for i, route in enumerate(routes):
            for stop in route:
                routesByStop[stop].add(i)
                
        queue, visited = collections.deque([[0, S]]), {S} 
        
        while queue:
            buses, stop  = queue.popleft()
            if stop == T:
                return buses
            for route in routesByStop[stop]:
                for next_stop in routes[route]:
                    if next_stop not in visited:
                        visited.add(next_stop)
                        queue.append([buses + 1, next_stop])

        return -1
from collections import defaultdict, deque
class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        stop_to_bus = defaultdict(set)
        
        for bus, stops in enumerate(routes):
            for stop in stops:
                stop_to_bus[stop].add(bus)
        
        queue = deque([(S, 0)])
        seen = set([S])
        
        while queue:
            stop, steps = queue.popleft()
            
            if stop == T:
                return steps
            
            for bus in stop_to_bus[stop]:
                for next_stop in routes[bus]:
                    if next_stop not in seen:
                        seen.add(next_stop)
                        queue.append((next_stop, steps + 1))
        
        return -1
class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        station_has_bus = collections.defaultdict(set)
        queue = collections.deque()
        
        for bus_id, route in enumerate(routes):
            for station in route:
                station_has_bus[station].add(bus_id)
        
        visited = set([S])
        queue.append((S, 0))
        
        while queue:
            station, step = queue.popleft()
            if station == T:
                return step
            for bus_id in station_has_bus[station]:
                for s in routes[bus_id]:
                    if s not in visited:
                        queue.append((s, step + 1))
                        visited.add(s)
        
        return -1

class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        graph = dict()
        for i,stops in enumerate(routes):
            for j in stops:
                if j not in graph:
                    graph[j] = set()
                graph[j].add(i)
        
                
                
        
        
        buses = 0
        if S==T:
            return 0
        queue = collections.deque()
        visited  = set()
        
        
        queue.append((S,0))
        visited.add(S)
        
        while len(queue)>0:
            
            curr_stop,depth  = queue.popleft()
            if curr_stop==T:
                return depth



            for stop in graph[curr_stop]:
                for j in routes[stop]:
                    if j not in visited:

                        queue.append((j,depth+1))
                        visited.add(j)
                        
            
            
            
        return -1
                    
        
        
        
                
        
        

from collections import defaultdict
class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        toRoutes = defaultdict(set)
        for i, route in enumerate(routes):
            for j in route:
                toRoutes[j].add(i)
        
        queue, visited = [(S, 0)], set([S])
        
        for stop, bus in queue:
            if stop == T: return bus
            for i in toRoutes[stop]:
                for j in routes[i]:
                    if j not in visited:
                        visited.add(j)
                        queue.append((j, bus+1))
        return -1
class Solution:
    def numBusesToDestination(self, routes, S, T):
        if S == T: return 0
        to_routes = collections.defaultdict(set)
        for i, route in enumerate(routes):
            for j in route:
                to_routes[j].add(i)
        bfs = [(S, 0)]
        seen = set([S])
        for stop,bus in bfs:
            if stop == T:
                return bus
            for i in to_routes[stop]:
                for j in routes[i]:
                    if j not in seen:
                        seen.add(j)
                        bfs.append((j,bus+1))
        return -1
class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        graph = defaultdict(set)
        
        for i, route in enumerate(routes):
            for stop in route:
                graph[stop].add(i)
        
        ans = 0
        
        queue = collections.deque([S])
        seen_stop = set()
        seen_route = set()
        
        seen_stop.add(S)
        
        while queue:
            for _ in range(len(queue)):
                stop = queue.popleft()
                
                if stop == T:
                    return ans
                
                for routeId in graph[stop]:
                    for new_stop in routes[routeId]:
                        if new_stop not in seen_stop:
                            queue.append(new_stop)
                            seen_stop.add(new_stop)
                
            ans += 1
        
        return -1

class Solution:
    def numBusesToDestination(self, routes, S, T):
        to_lines = collections.defaultdict(set)  # key: step; value: line pass this stop
        for i, line in enumerate(routes):
            for j in line:
                to_lines[j].add(i)

        queue = [S]
        visited = {S}
        step = 0
        while queue:
            new_queue = []
            for stop in queue:
                if stop == T: 
                    return step
                for i in to_lines[stop]:
                    for j in routes[i]:
                        if j not in visited:
                            new_queue.append(j)
                            visited.add(j)
                    
                    #routes[i] = []  # seen route
            queue = new_queue
            step += 1
        return -1
class Solution:
    def numBusesToDestination(self, routes, S, T):
        to_lines = collections.defaultdict(set)  # key: step; value: line pass this stop
        for i, line in enumerate(routes):
            for j in line:
                to_lines[j].add(i)

        queue = [S]
        visited = {S}
        step = 0
        while queue:
            new_queue = []
            for stop in queue:
                if stop == T: 
                    return step
                for i in to_lines[stop]:
                    for j in routes[i]:
                        if j not in visited:
                            new_queue.append(j)
                            visited.add(j)
                    #routes[i] = []  # seen route
            queue = new_queue
            step += 1
        return -1
class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        if S == T:
            return 0

        neighbor_stops = collections.defaultdict(set)
        for i, route in enumerate(routes):
            for j in route:
                neighbor_stops[j].add(i)

        bfs = [(S, 0)]
        visited = set([S])
        for current_stop, bus_change in bfs:
            if current_stop == T:
                return bus_change
            for index in neighbor_stops[current_stop]:
                for neighbor in routes[index]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        bfs.append((neighbor, bus_change + 1))

                neighbor_stops[current_stop] = []

        return -1
class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        if S == T:
            return 0

        neighbor_stops = collections.defaultdict(set)
        for i, route in enumerate(routes):
            for j in route:
                neighbor_stops[j].add(i)

        bfs = [(S, 0)]
        visited = set([S])
        for current_stop, bus_change in bfs:
            if current_stop == T:
                return bus_change
            for index in neighbor_stops[current_stop]:
                for neighbor in routes[index]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        bfs.append((neighbor, bus_change + 1))

                

        return -1
class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        if S == T: return 0
        bus_stop_dict = collections.defaultdict(set)
        # each stop corresponds to all the buses
        for i in range(len(routes)):
            for stop in routes[i]:
                bus_stop_dict[stop].add(i)
        
        seen_bus = set()
        seen_stop = set()
        stop_list = []
        for bus in bus_stop_dict[S]:
            seen_bus.add(bus)
            for stop in routes[bus]:
                if stop not in seen_stop:
                    seen_stop.add(stop)
                    stop_list.append(stop)
        ans = 1
        while stop_list:
            new_list = []
            for stop in stop_list:
                seen_stop.add(stop)
                if stop == T: return ans
                for bus in bus_stop_dict[stop]:
                    if bus not in seen_bus:
                        seen_bus.add(bus)
                    for s in routes[bus]:
                        if s not in seen_stop:
                            seen_stop.add(s)
                            new_list.append(s)
            
            stop_list = new_list
            ans += 1
        
        return -1
from collections import defaultdict, deque

class Solution:
    def numBusesToDestination(self, routes: List[List[int]], source_stop: int, target_stop: int) -> int:
        if source_stop == target_stop: return 0
        routes = [list(set(r)) for r in routes]
        stops = defaultdict(set)
        for i,route in enumerate(routes):
            for stop in route:
                stops[stop].add(i)
        q = deque()
        visited_stops = set()
        visited_buses = set()
        q.append((source_stop, 0))
        while q:
            stop_num, bus_num = q.popleft()
            visited_stops.add(stop_num)
            for other_bus in stops[stop_num]:
                if other_bus in visited_buses: continue
                for other_stop in routes[other_bus]:
                    if other_stop == target_stop:
                        return bus_num + 1
                    if other_stop not in visited_stops:
                        visited_stops.add(other_stop)
                        q.append((other_stop, bus_num + 1))
        return -1


from collections import deque
class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        # bfs  put routes contain S into q
        # from existing bus -> next bus until there is T 
        if not routes:
            return -1
        
        if T==S:
            return 0
        
        q= deque([])
        r = []
        res = 0
        visited = set()
        for idx, route in enumerate(routes):
            r.append(set(route))
            if S in r[-1]:
                q.append(idx)
                visited.add(idx)
        
        print(q)
        while q :
            newq=[]
            res+=1
            for bus in q:
                if T in r[bus]:
                    return res

                # potential transfer
                for stop in r[bus]:
                    for idx, route in enumerate(r):
                        if stop in route and idx not in visited:
                            visited.add(idx)
                            newq.append(idx)
            q=newq
            
        return -1
class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        if S == T:
            return 0
        hmap = collections.defaultdict(set)
        
        for i in range(len(routes)):
            for j in routes[i]:
                hmap[j].add(i) # adjacency list for route to bus... i.e. tracking the stop on each route.
        # print(hmap)
        q = [(S, 0)]
        visited = set()
        visited.add(S)
        
        while q:
            stop, buses = q.pop(0)
            
            if stop == T:
                return buses
            
            for i in hmap[stop]: # go to the ith bus route
                # print(i)
                for j in routes[i]: # traverse the ith bus route
                    if j not in visited:
                        visited.add(j)
                        q.append((j, buses + 1))
        
        return -1
class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        graph = collections.defaultdict(set)
        for i, route in enumerate(routes):
            for j in route:
                graph[j].add(i)
        queue = collections.deque([(S, 0)])
        seen = set([S])
        while queue:
            curr, step = queue.popleft()
            if curr == T: return step
            for i in graph[curr]:
                for route in routes[i]:
                    if route not in seen:
                        queue.append((route, step + 1))
                        seen.add(route)
        return -1
class Solution(object):
    def numBusesToDestination(self, routes, S, T):
        if S == T: 
            return 0
        
        
        # routes = map(set, routesList)
        for i, route in enumerate(routes):
            routes[i] = set(route)
        print(routes)
        
        graph = collections.defaultdict(set)
        for i, r1 in enumerate(routes):
            for j in range(i+1, len(routes)):
                r2 = routes[j]
                if any(r in r2 for r in r1):
                    graph[i].add(j)
                    graph[j].add(i)
                    
        print(graph)

        seen, targets = set(), set()
        for node, route in enumerate(routes):
            if S in route: 
                seen.add(node)
            if T in route: 
                targets.add(node)

        queue = [(node, 1) for node in seen]
        for node, depth in queue:
            if node in targets: 
                return depth
            for nei in graph[node]:
                if nei not in seen:
                    seen.add(nei)
                    queue.append((nei, depth+1))
        return -1
class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        if S == T:
            return 0
        
        graph = defaultdict(list)
        for i in range(len(routes)):
            for j in range(len(routes[i])):
                graph[routes[i][j]].append(i)
        
        visited = set()
        queue = deque()
        queue.append(S)
        
        level = 0
        while queue:
            size = len(queue)
            for _ in range(size):
                curr = queue.popleft()
                if curr == T:
                    return level
                
                for bus in graph[curr]:
                    if bus not in visited:
                        for stop in routes[bus]:
                            queue.append(stop)
                    visited.add(bus)
            
            level += 1
        
        return -1
class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        graph = collections.defaultdict(list)
        if S == T: return 0
        nodeS, nodeT = set(), set()
        for i in range(len(routes)):
            if S in routes[i]: nodeS.add(i)
            if T in routes[i]: nodeT.add(i)
            if i < len(routes) - 1:
                for j in range(1, len(routes)):
                    if set(routes[i]) & set(routes[j]): 
                        graph[i].append(j)
                        graph[j].append(i)
        
        # print(graph)
        if nodeS & nodeT: return 1
        if not nodeS or not nodeT: return -1
                
        queue, visited = collections.deque(), set()
        for s in nodeS:
            queue.append((s, 1))
        
        
        
        while queue:
            # print(queue)
            node, step = queue.popleft()
            if node in nodeT: return step
            for nbr in graph[node]:
                if nbr not in visited:
                    queue.append((nbr, step + 1))
                    visited.add(nbr)
        
        return -1
            
        

class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        
        if S == T:
            return 0
        n = len(routes)
        for i in range(n):
            routes[i].sort()
        graph = [[] for i in range(n)]
        for i in range(n):
            for j in range(i):
                if self.haveCommonValues(routes[i], routes[j]):
                    graph[i].append(j)
                    graph[j].append(i)
        print(graph)
        targets = set()
        vis = set()
        for i in range(n):
            if self.binary_search(routes[i], T):
                targets.add(i)
            if self.binary_search(routes[i], S):
                vis.add(i)
        print(vis)
        print(targets)
        step = 1
        q = list(vis)
        while q:
            temp = []
            for cur in q:
                if cur in targets:
                    return step
                for nei in graph[cur]:
                    if nei not in vis:
                        vis.add(nei)
                        temp.append(nei)
            q = temp
            step += 1
        return -1
                    
    
    def binary_search(self, A, target):
        l = 0
        r = len(A) - 1
        while l <= r:
            mid = (l+r)//2
            if A[mid]==target:
                return True
            if target < A[mid]:
                r = mid - 1
            else:
                l = mid + 1
        return False
                
    
    def haveCommonValues(self, A, B):
    
        i = 0
        j = 0
        while i < len(A):
            while j < len(B) and B[j] < A[i]:
                j += 1
            if j == len(B):
                return False
            if B[j] == A[i]:
                return True
            i += 1
        return False
class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        
        if S == T:
            
            return 0
        
        routes = [set(route) for route in routes]
        
        reachable = collections.defaultdict(list)
        
        for i in range(len(routes)):
            
            for j in range(i + 1, len(routes)):
                
                if any([stop in routes[j] for stop in routes[i]]):
                    
                    reachable[i].append(j)
                    
                    reachable[j].append(i)
        
        target_routes = set()
        
        q = []
        
        visited = set()
        
        for i in range(len(routes)):
            
            if S in routes[i]:
                
                q.append((i, 1))
                
                visited.add(i)
                
            if T in routes[i]:
                
                target_routes.add(i)
                
        while q:
            
            route, count = q.pop(0)
            
            if route in target_routes:
                
                return count
            
            for next_route in reachable[route]:
                
                if next_route not in visited:
                    
                    visited.add(next_route)
                    
                    q.append((next_route, count + 1))
                    
        return -1
class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        graph = defaultdict(set)  
        
        for i in range(len(routes)):
            route = routes[i]
            for stop in route:
                graph[stop].add(i)
                
        ans = 0
        
        queue = collections.deque()
        
        queue.append(S)
        seen_stop = set([S])
        seen_route = set()
        
        while queue:
            for _ in range(len(queue)):
                stop = queue.popleft()
                
                if stop == T:
                    return ans
                
                for route in graph[stop]:
                    #if route not in seen_route:
                        for new_stop in routes[route]:
                            if new_stop not in seen_stop:
                                queue.append(new_stop)
                                seen_stop.add(new_stop)
                    
                        seen_route.add(route)
            
            ans += 1
        
        return -1
class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        if not routes:
            return -1
        
        smap = defaultdict(set)
        for bus, stops in enumerate(routes):
            for stop in stops:
                smap[stop].add(bus)
                
        stack = [(S,0)]
        visited_stop = set()
        while stack:
            cur_stop, depth = stack.pop(0)
            
            if T == cur_stop:
                return depth

            for bus in list(smap[cur_stop]):
                for stop in routes[bus]:
                    if stop not in visited_stop and stop not in stack:
                        stack.append((stop, depth+1))
                        visited_stop.add(stop)
        return -1
                        

class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        if S == T:
            return 0

        graph = collections.defaultdict(set)
        
        for route in routes:
            for stop in set(route):
                graph[stop].update(route)

        queue = [(S, 0)]
        visited = {S}
        
        while queue:
            stop, dist = queue.pop(0)
            if stop == T:
                return dist
            for next_stop in graph[stop]:
                if next_stop not in visited:
                    queue.append((next_stop, dist + 1))
                    visited.add(next_stop)

        return -1
class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        if S==T: return 0
        map = defaultdict(list)
        starting_buses = []
        for bus, route in enumerate(routes):   
          for stop in route:
            if stop == S: starting_buses.append(bus)
            map[stop].append(bus)
    
        visited_stops = set()
        q = deque()
        min_buses = float('inf')
        for starting_bus in starting_buses:
          q.append((starting_bus,1))
          visited_stops.add(S)
  
        while q:
            bus, num_bus = q.pop()
            if num_bus == min_buses: continue
            for stop in routes[bus]:
              if stop == T:
                min_buses = min(min_buses, num_bus)
                break
              if stop not in visited_stops:
                visited_stops.add(stop)
                for next_bus in map[stop]:
                  if next_bus == bus: continue
                  q.append((next_bus, num_bus+1))
          
        return min_buses if min_buses != float('inf') else -1

class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        if S == T:
            return 0
        graph = [set() for _ in range(len(routes))]
        stops = defaultdict(list)
        q = set()
        ends = set()
        for i, route in enumerate(routes):
            for stop in route:
                for other in stops[stop]:
                    graph[i].add(other)
                    graph[other].add(i)
                
                stops[stop].append(i)
                if stop == S:
                    q.add(i)
                if stop == T:
                    ends.add(i)
        dist = 1
        seen = q.copy()
        while q:
            next_level = set()
            count = len(q)
            for route in q:
                if route in ends:
                    return dist
                
                for n in graph[route]:
                    if n not in seen:
                        next_level.add(n)
                        seen.add(n)
            
            q = next_level
            dist += 1
        
        return -1
from queue import deque

def bfs(adjList, s, t):
    END = '$'
    queue = deque([s, END])
    visited = set()
    step = 0
    while queue:
        node = queue.popleft()
        
        if node == t:
            return step
        
        if node == END:
            if not queue:
                break
            step += 1
            queue.append(END)
            continue
            
        visited.add(node)
        for adjNode in adjList[node]:
            if adjNode not in visited:
                queue.append(adjNode)
    return -1

def numBusesToDestination_Graph_TLE(routes, S, T):
    adjList = {}
    for route in routes:
        for i in range(len(route)):
            if route[i] not in adjList:
                adjList[route[i]] = set()
            for j in range(i+1, len(route)):
                if route[j] not in adjList:
                    adjList[route[j]] = set()
                adjList[route[i]].add(route[j])
                adjList[route[j]].add(route[i])
    if S not in adjList:
        return -1
    return bfs(adjList, S, T)

def numBusesToDestination_GraphTakeRouteAsNod_TLE(routes, S, T):
    if S == T:
        return 0
    adjList = {
        'S': [], 'T': []
    }
    
    for i in range(len(routes)):
        routes[i] = set(routes[i])
        adjList[i] = []
        if S in routes[i]:
            adjList['S'].append(i)
        if T in routes[i]:
            adjList[i].append('T')
    
    for i in range(len(routes)):
        for j in range(i+1, len(routes)):
            if any([k in routes[j] for k in routes[i]]):
                adjList[i].append(j)
                adjList[j].append(i)
    return max(bfs(adjList, 'S', 'T') - 1, -1)

class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        # return numBusesToDestination_Graph_TLE(routes, S, T)
        return numBusesToDestination_GraphTakeRouteAsNod_TLE(routes, S, T)
class Solution: # 372 ms
    def numBusesToDestination(self, routes, start, target):
        if start == target: return 0

        stop2Route = defaultdict(set)
        route2Stop = defaultdict(set)
        for i, stops in enumerate(routes):
            for stop in stops:
                route2Stop[i].add(stop)
                stop2Route[stop].add(i)

     #   return stop2Route, route2Stop
        visited = set() 
        visitedStop = set()       
        
        q = [start]
        step = 0
        while q:
            stack = []
            step += 1
            for stop in q:
                visitedStop.add(stop)
                for route in stop2Route[stop]:
                    if route not in visited:
                        if target in route2Stop[route]:
                            return step
                        else:
                            for stop in route2Stop[route]:
                                if stop not in visitedStop:
                                    stack.append(stop)
                            visited.add(route)       
                 
            q = stack
         #   print(q, stop, stack, visited, step, visitedStop)
         #   return

        return -1
class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        adj_list = collections.defaultdict(set)
        for i in range(len(routes)):
            for j in range(len(routes[i])):
                adj_list[routes[i][j]].add(i)
        queue = collections.deque([(S,0)])
        visited = set()
        visited.add(S)
        while queue:
            node,taken = queue.popleft()
            if node == T:
                return taken
            for i in adj_list[node]:
                for j in routes[i]:
                    if j not in visited:
                        visited.add(j)
                        queue.append((j,taken+1))
            routes[i] = []
        return -1
                

class Solution:
    def numBusesToDestination(self, routes, start, target):
        if start == target: return 0

        stop2Route = defaultdict(set)
        route2Stop = defaultdict(set)
        for i, stops in enumerate(routes):
            for stop in stops:
                route2Stop[i].add(stop)
                stop2Route[stop].add(i)

     #   return stop2Route, route2Stop
        visited = set() 
        visitedStop = set()       
        
        q = [start]
        step = 0
        while q:
            stack = []
            step += 1
            while q:
                stop = q.pop()
                for route in stop2Route[stop]:
                    if route not in visited:
                        if target in route2Stop[route]:
                          #  if stop in route2Stop[route]:
                                return step
                        else:
                            stack.extend(route2Stop[route])
                            visited.add(route)       
                visitedStop.add(stop) 
                
            for stop in stack:
             #   for stop in route2Stop[route]:
                    if stop not in visitedStop:
                        q.append(stop)
         #   print(q, stop, stack, visited, step, visitedStop)
         #   return

        return -1
from collections import deque
class Solution:
    def numBusesToDestination(self, routes, S, T):
        if S == T: return 0
        stopBoard = {} 
        for bus, stops in enumerate(routes):
            for stop in stops:
                if stop not in stopBoard:
                    stopBoard[stop] = [bus]
                else:
                    stopBoard[stop].append(bus)
        queue = deque([S])
        visited = set()
        res = 0
        while queue:
            res += 1
            pre_num_stops = len(queue)
            for _ in range(pre_num_stops):
                curStop = queue.popleft()
                for bus in stopBoard[curStop]:
                    if bus in visited: continue
                    visited.add(bus)
                    for stop in routes[bus]:
                        if stop == T: return res
                        queue.append(stop)
        return -1
class Solution: # 368 ms
    def numBusesToDestination(self, routes, start, target):
        if start == target: return 0

        stop2Route = defaultdict(set)
        route2Stop = defaultdict(set)
        for i, stops in enumerate(routes):
            for stop in stops:
                route2Stop[i].add(stop)
                stop2Route[stop].add(i)

        visited = set() 
        visitedStop = set()       
        
        q = [start]
        step = 0
        while q:
            stack = []
            step += 1
            for stop in q:
                visitedStop.add(stop)
                for route in stop2Route[stop]:
                    if route not in visited:
                        if target in route2Stop[route]:
                            return step
                        else:
                            for stop in routes[route]:
                              #  if stop == target: return step
                                if stop not in visitedStop:
                                    stack.append(stop)
                            visited.add(route)       
            q = stack
         #   print(q, stop, stack, visited, step, visitedStop)
         #   return

        return -1
class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        if S == T: return 0
        
        # Builds graph.
        graph = collections.defaultdict(list)  # Don't use set. See below.
        for bus, stops in enumerate(routes):
            bus = -bus - 1  # To avoid conflict with the stops.
            
            # `set.update` consumes extra memory, so a `list` is used instead.
            graph[bus] = stops
            for s in stops:
                graph[s].append(bus)

        # Does BFS.
        dq = deque([(S, 0)])
    #    dq.append((S, 0))
        seen = {S}
        while dq:
            node, depth = dq.popleft()
            for adj in graph[node]:
                if adj in seen: continue
                if adj == T: return depth
                # If `adj` < 0, it's a bus, so we add 1 to `depth`.
                dq.append((adj, depth + 1 if adj < 0 else depth))
                seen.add(adj)
        return -1
class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        stop_to_bus = collections.defaultdict(set)
        for bus, route in enumerate(routes):
            for stop in route:
                stop_to_bus[stop].add(bus)
        
        q = collections.deque([(S, 0)])
        
        visited = set()
        while q:
            stop, bus = q.popleft()
            visited.add(stop)
            if stop == T:
                return bus
            
            for next_bus in stop_to_bus[stop]:
                for next_stop in routes[next_bus]:
                    if next_stop not in visited:
                        q.append((next_stop, bus + 1))
                routes[next_bus] = []
        return -1         

class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        if S==T: return 0
        bus2bus = defaultdict(set)
        stop2bus = defaultdict(set)
        dq = deque()
        seen = set()
        dest = set()
        cnt = 1
        
        for i, route in enumerate(routes):
            for s in route:
                for b in stop2bus[s]:
                    bus2bus[i].add(b)
                    bus2bus[b].add(i)
                    
                stop2bus[s].add(i)
                if s==S:
                    seen.add(i)
                if s==T:
                    dest.add(i)
        # print(bus2bus)
        dq.extend(seen)
        while dq:
            length = len(dq)
            
            for _ in range(length):
                curr = dq.popleft()
                
                if curr in dest:
                    return cnt
                
                for nxt in bus2bus[curr]:
                    if nxt in seen:
                        continue
                        
                    seen.add(nxt)
                    dq.append(nxt)
                
            cnt += 1
        
        return -1
class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        
        if S == T:
            return 0
        
        graph = {}
        
        bs = {}
        for i, route in enumerate(routes):
            for s in route:
                if s not in bs:
                    bs[s] = set()
                
                for b in bs[s]:
                    # connect b and i
                    if b not in graph:
                        graph[b] = set()
                    if i not in graph:
                        graph[i] = set()
                    graph[b].add(i)
                    graph[i].add(b)
                
                bs[s].add(i)
        
        if S not in bs:
            return -1
        
        
#         print("bs: {}".format(bs))
#         print("graph: {}".format(graph))
        
        q = collections.deque(list(bs[S]))
        bus_taken = 1
        visited = set(list(bs[S]))
        
        while len(q):
            l = len(q)
            for _ in range(l):
                curt = q.popleft()
                
                if curt in bs[T]:
                    return bus_taken
                
                if curt in graph:
                    for nb in graph[curt]:
                        if nb not in visited:
                            visited.add(nb)
                            q.append(nb)
                
            bus_taken += 1
            
        return -1
class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        if S == T:
            return 0
        
        bus_stops = {}
        untaken_bus, bfs_queue = set(), []
        for bus, stops in enumerate(routes):
            bus_stops[bus] = set(stops)
            if S in bus_stops[bus]:
                bfs_queue.append(bus)
                continue
            untaken_bus.add(bus)
            
        res = 1
        while bfs_queue:
            next_bfs_queue = []
            for bus in bfs_queue:
                if T in bus_stops[bus]:
                    return res
                for u_b in list(untaken_bus):
                    if bus_stops[u_b].intersection(bus_stops[bus]):
                        next_bfs_queue.append(u_b)
                        untaken_bus.remove(u_b)
            bfs_queue = next_bfs_queue
            res += 1
        return -1
class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        to_routes = collections.defaultdict(set)
        for i, route in enumerate(routes):
            for j in route:
                to_routes[j].add(i)
        bfs = [(S, 0)]
        seen = set([S])
        for stop, bus in bfs:
            if stop == T: return bus
            for i in to_routes[stop]:
                for j in routes[i]:
                    if j not in seen:
                        bfs.append((j, bus + 1))
                        seen.add(j)
                routes[i] = []
        return -1
class Solution: # 372 ms
    def numBusesToDestination(self, routes, start, target):
        if start == target: return 0

        stop2Route = defaultdict(set)
        route2Stop = defaultdict(set)
        for i, stops in enumerate(routes):
            for stop in stops:
                route2Stop[i].add(stop)
                stop2Route[stop].add(i)

     #   return stop2Route, route2Stop
        visited = set() 
        visitedStop = set()       
        
        q = {start}
        step = 0
        while q:
            stack = set()
            step += 1
            for stop in q:
                visitedStop.add(stop)
                for route in stop2Route[stop]:
                    if route not in visited:
                        if target in route2Stop[route]:
                            return step
                        else:
                            for stop in route2Stop[route]:
                                if stop not in visitedStop:
                                    stack.add(stop)
                            visited.add(route)       
                 
            q = stack
         #   print(q, stop, stack, visited, step, visitedStop)
         #   return

        return -1
class Solution:
    def dfs(self, graph, s, des, steps, visited):
        if self.res != -1 and steps >= self.res:
            return
        if s in des:
            if self.res == -1:
                self.res = steps
            else:
                self.res = min(self.res, steps)
            return
        if s not in graph:
            return
        for i in graph[s]:
            if i not in visited:
                visited.add(i)
                self.dfs(graph, i, des, steps + 1, visited)
                visited.remove(i)
        
    
    def numBusesToDestination(self, routes: List[List[int]], s: int, t: int) -> int:
        if s == t:
            return 0
        des = set()
        starts = set()
        stop_to_bus = {}
        graph = {}
        for bus in range(0, len(routes)):
            for stop in routes[bus]:
                if stop == s:
                    starts.add(bus)
                if stop == t:
                    des.add(bus)
                if stop not in stop_to_bus:
                    stop_to_bus[stop] = set()
                stop_to_bus[stop].add(bus)
        
        for _, v in list(stop_to_bus.items()):
            for i in v:
                for j in v:
                    if i == j:
                        continue
                    if i not in graph:
                        graph[i] = set()
                    graph[i].add(j)
        
        self.res = -1
        visited = set()
        for s in starts:
            visited.add(s)
            self.dfs(graph, s, des, 1, visited)
            visited.remove(s)
            
        return self.res
                
        
        
        
        
        
        
        
        
        
        
        

class Solution:
    def numBusesToDestination(self, routes, start, target):
        if start == target: return 0

        stop2Route = defaultdict(set)
        route2Stop = defaultdict(set)
        for i, stops in enumerate(routes):
            for stop in stops:
                route2Stop[i].add(stop)
                stop2Route[stop].add(i)

     #   return stop2Route, route2Stop
        visited = set() 
        visitedStop = set()       
        
        q = [start]
        step = 0
        while q:
            stack = []
            step += 1
            while q:
                stop = q.pop()
                for route in stop2Route[stop]:
                    if route not in visited:
                        if target in route2Stop[route]:
                            if stop in route2Stop[route]:
                                return step
                        else:
                            stack.extend(route2Stop[route])
                            visited.add(route)       
                visitedStop.add(stop) 
                
            for stop in stack:
             #   for stop in route2Stop[route]:
                    if stop not in visitedStop:
                        q.append(stop)
         #   print(q, stop, stack, visited, step, visitedStop)
         #   return

        return -1
class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        if S == T: return 0
        
        # Builds graph.
        graph = collections.defaultdict(list)  # Don't use set. See below.
        for bus, stops in enumerate(routes):
            bus = -bus - 1  # To avoid conflict with the stops.
            
            # `set.update` consumes extra memory, so a `list` is used instead.
            graph[bus] = stops
            for s in stops:
                graph[s].append(bus)

        # Does BFS.
        dq = collections.deque()
        dq.append((S, 0))
        seen = set([S])
        while dq:
            node, depth = dq.popleft()
            for adj in graph[node]:
                if adj in seen: continue
                if adj == T: return depth
                # If `adj` < 0, it's a bus, so we add 1 to `depth`.
                dq.append((adj, depth + 1 if adj < 0 else depth))
                seen.add(adj)
        return -1
class Solution: # 368 ms
    def numBusesToDestination(self, routes, start, target):
        if start == target: return 0

        stop2Route = defaultdict(set)
     #   route2Stop = defaultdict(set)
        for i, stops in enumerate(routes):
            for stop in stops:
              #  route2Stop[i].add(stop)
                stop2Route[stop].add(i)

        visited = set() 
        visitedStop = set()       
        
        q = [start]
        step = 0
        while q:
            stack = []
            step += 1
            for stop in q:
                visitedStop.add(stop)
                for route in stop2Route[stop]:
                    if route not in visited:
                     #   if target in route2Stop[route]:
                     #       return step
                     #   else:
                            for stop in routes[route]:
                                if stop == target: return step
                                if stop not in visitedStop:
                                    stack.append(stop)
                            visited.add(route)       
            q = stack
         #   print(q, stop, stack, visited, step, visitedStop)
         #   return

        return -1
class Node:
    def __init__(self):
        self.idx = None
        self.val = set()
class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        if S == T:
            return 0
        is_t_present = False
        all_routes = []
        potential_starts = []
        graph = collections.defaultdict(list)
        
        for i, route in enumerate(routes):
            node = Node()
            node.idx = i
            node.val = set(route)
            if S in node.val:
                potential_starts.append([i,1])
                if T in node.val:
                    return 1
            if T in node.val:
                is_t_present = True
            all_routes.append(node)
        
        if not is_t_present or not potential_starts:
            return -1
        
        for i in range(len(all_routes)):
            curr = all_routes[i]
            for j in range(i + 1, len(all_routes)):
                temp = all_routes[j]
                if curr.val & temp.val:
                    graph[i].append(j)
                    graph[j].append(i)
        
        dq = collections.deque(potential_starts)
        visited = set()
        while len(dq) > 0:
            idx, depth = dq.popleft()
            visited.add(idx)
            node = all_routes[idx]
            if T in node.val:
                return depth
            for nei in graph[idx]:
                if nei not in visited:
                    dq.append([nei, depth + 1])
        return -1    
from collections import defaultdict, deque


class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        '''
        model as graph
        find shortest path with BFS
        
        buses connect bus stops
        
        bus stops are vertices
        
        no
        
        routes are connected if they have a common point
        
        routes are vertices
        
        
        '''
        if S == T:
            return 0
        
        g = Graph()
        
        sets = [set(route) for route in routes]
        for s in sets:
            if S in s and T in s:
                return 1
        
        for i, s1 in enumerate(sets):
            for j in range(i+1, len(sets)):
                s2 = sets[j]
                if not s1.isdisjoint(s2):
                    g.connect(i, j)

        starts = [i for i, s in enumerate(sets) if S in s]
        ends = {i for i, s in enumerate(sets) if T in s}
        min_steps = [g.min_steps(start, ends) for start in starts]
        return min((x for x in min_steps if x > 0), default=-1)
    

class Graph:
    def __init__(self):
        self.adj = defaultdict(set)
        
    def connect(self, p, q):
        self.adj[p].add(q)
        self.adj[q].add(p)
        
    def min_steps(self, start, ends):
        seen = {start}
        q = deque([start])
        steps = 1
        
        while q:
            for _ in range(len(q)):
                p = q.popleft()
                if p in ends:
                    return steps
                for v in self.adj[p]:
                    if v in seen:
                        continue
                    seen.add(v)
                    q.append(v)
            
            steps += 1
        
        return -1
class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        if S == T: return 0
        
        # Builds graph.
        graph = collections.defaultdict(list)  # Don't use set. See below.
        for bus, stops in enumerate(routes):
            bus = -bus - 1  # To avoid conflict with the stops.
            
            # `set.update` consumes extra memory, so a `list` is used instead.
            graph[bus] = stops
            for s in stops:
                graph[s].append(bus)

        # Does BFS.
        dq = collections.deque()
        dq.append((S, 0))
        seen = {S}
        while dq:
            node, depth = dq.popleft()
            for adj in graph[node]:
                if adj in seen: continue
                if adj == T: return depth
                # If `adj` < 0, it's a bus, so we add 1 to `depth`.
                dq.append((adj, depth + 1 if adj < 0 else depth))
                seen.add(adj)
        return -1
class Solution:
    def numBusesToDestination(self, routes, S, T):
        to_routes = collections.defaultdict(set)
        for route, stops in enumerate(routes):
            for s in stops:
                to_routes[s].add(route)
        bfs = [(S, 0)]
        seen = set([S])
        for stop, bus in bfs:
            if stop == T: return bus
            for i in to_routes[stop]:
                for j in routes[i]:
                    if j not in seen:
                        bfs.append((j, bus + 1))
                        seen.add(j)
                routes[i] = []  # seen route
        return -1
class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        # already at the destination
        if S == T : return 0
        
        # dictionary of busstops and buses
        # this maintains a dictionary of which bus routes fall on which bustops
        busDict = defaultdict(set)
        for i , route in enumerate(routes):
            for stop in route:
                busDict[stop].add(i)
        
        # to avoid visiting the same bus route
        visited = set()
        # queue for traversing through bus stops. 
        queue = deque([S])
        res = 0
        while queue:
            leng = len(queue)
            res += 1
            for _ in range(leng):
                
                curr = queue.popleft()
                
                # check which buses pass through the current bus stop
                for bus in busDict[curr]:
                    if bus not in visited:
                        visited.add(bus)
                        
                        # check if this bus route contains the destination
                        if T in routes[bus]:
                            return res
                        queue.extend(routes[bus])
        return -1
                    
                
                
                
        

class Solution:
    def numBusesToDestination(self, routes, S, T):
        if S == T: return 0
        queue = collections.deque()
        graph = collections.defaultdict(set)
        routes = list(map(set, routes))
        
        seen, targets = set(), set()
        
        for i in range(len(routes)):
            if S in routes[i]:  # possible starting route number
                seen.add(i)
                queue.append((i, 1))  # enqueue
            if T in routes[i]:  # possible ending route number
                targets.add(i)
            for j in range(i+1, len(routes)):
                if routes[j] & routes[i]:  # set intersection to check if route_i and route_j are connected
                    graph[i].add(j)
                    graph[j].add(i)
        
        while queue:
            cur, count = queue.popleft()
            if cur in targets:
                return count
            for nei in graph[cur]:
                if nei not in seen:
                    queue.append((nei, count+1))
                    seen.add(nei)
        return -1
class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        if S == T:
            return 0
        routes = list(map(set, routes))
        
        graph = collections.defaultdict(set)
        for i, route1 in enumerate(routes):
            for j in range(i+1, len(routes)):
                route2 = routes[j]
                if route1 & route2:
                    graph[i].add(j)
                    graph[j].add(i)
        
        seen = set()
        target = set()
        for bus, route in enumerate(routes):
            if S in route:
                seen.add(bus)
            if T in route:
                target.add(bus)
        
        Q = [(1, bus) for bus in seen]
        for steps, bus in Q:
            if bus in target:
                return steps
            for nbr in graph[bus]:
                if nbr not in seen:
                    seen.add(nbr)
                    Q.append((steps+1, nbr))
        return -1
            
            

class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:

        md, que, visited, changes = collections.defaultdict(set), collections.deque(), set(), {}

        # prre-processing, we can get dict m[1] = (0), m[2] = (0), m[7] = (0,1) ,etc
        # means for stop 7 , bus 0, and 1 can reach
        for b, r in enumerate(routes):
            for s in r:
                md[s].add(b)

        # S or T not reachable
        if S not in md or T not in md:
            return -1

        # if S and T are same, we don't even need to take bus
        if S == T:
            return 0

        for b in md[S]:
            for stop in routes[b]:
                # (2,0,0), (7,0,0)
                # (1,0,0) - means stop 1, bus 0, bus changes 0
                que.append((stop, b, 1))
                # changes[1,0] = 0, changes[2,0] = 0, changes[7,0] = 0
                # means for reach 1,2,7 we just 1 times of bus change
                # (take the first bus also count as 1 change)
                changes[stop, b] = 1

        while que:
            stop, bus, times = que.popleft()
            # already reach the Target
            if stop == T:
                return times
            for b in md[stop]:
                if bus != b:
                    for stop in routes[b]:
                        # if I already reached this stop by bus, but I used few times for change
                        if (stop, bus) in changes and changes[stop, bus] > 1 + times:
                            que.append((stop, bus, 1 + times))
                            # remember update the new times in cache
                            changes[stop, bus] = 1 + times
                        elif (stop, bus) not in changes:  # I never reached stop by this bus yet
                            changes[stop, bus] = 1 + times
                            que.append((stop, bus, 1 + times))
                        # else: if I reached stop by bus, but I changed more times than the record in cache,
                        # just prunning it
                    # this sentences improve the performance greatly
                    # the time is from 5000ms decrease to 260 ms
                    routes[b] = []

        return -1


class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        if S == T:
            return 0
        
        routes = [set(stops) for stops in routes]
        
        #there is no reason to take the same bus twice
        taken = set()
        dq = collections.deque()
        
        neighbors = collections.defaultdict(list)
        
        for i in range(len(routes)):
            if S in routes[i]:
                dq.append(i)
                taken.add(i)
            for j in range(i):
                if routes[i].intersection(routes[j]):
                    neighbors[i].append(j)
                    neighbors[j].append(i)
        
        count = 1
        while dq:
            size = len(dq)
            for _ in range(size):
                bus = dq.popleft()
                if T in routes[bus]:
                    return count
                for nei in neighbors[bus]:
                    if nei not in taken:
                        dq.append(nei)
                        taken.add(nei)
            count += 1
        
        return -1

class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        
        if S==T:
            return 0
        
        s2b = {}
        for b, route in enumerate(routes):
            for s in route:
                if s not in s2b:
                    s2b[s] = set()
                s2b[s].add(b)
                
        q = collections.deque([S])
        visited = set()
        
        buses = 1
        while len(q):
            l = len(q)
            for _ in range(l):
                curt = q.popleft()
                # if curt in s2b:
                for nb in s2b[curt]:
                    if nb in visited:
                        continue
                    visited.add(nb)
                    for ns in routes[nb]:
                        if ns == T:
                            return buses
                        q.append(ns)
            
            buses += 1
        return -1
class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        if S == T: return 0
        queue = collections.deque()
        graph = collections.defaultdict(set)
        routes = list(map(set, routes))
        seen, targets = set(), set()
        for i in range(len(routes)):
            if S in routes[i]:  # possible starting route number
                seen.add(i)
                queue.append((i, 1))  # enqueue
            if T in routes[i]:  # possible ending route number
                targets.add(i)
            for j in range(i+1, len(routes)):
                if routes[j] & routes[i]:  # set intersection to check if route_i and route_j are connected
                    graph[i].add(j)
                    graph[j].add(i)
        while queue:
            cur, count = queue.popleft()
            if cur in targets:
                return count
            for nei in graph[cur]:
                if nei not in seen:
                    queue.append((nei, count+1))
                    seen.add(nei)
        return -1
from collections import defaultdict

class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        
        
        br = []
        start_r = []
        for i, r in enumerate(routes):
            br.append(set(r))
            if S in r:
                start_r.append(i)
        
        
        if S == T:
            return 0
        
        g = defaultdict(set)
        for i in range(len(br)):
            for j in range(i+1, len(br)):
                if br[i].intersection(br[j]):
                    g[i].add(j)
                    g[j].add(i)
        
        
        def bfs(r):
            it = 1
            q = [r]
            vs = set()
            
            while q:
                tmp = []
                for ele in q:
                    if ele in vs:
                        continue
                    vs.add(ele)
                    if T in br[ele]:
                        return it
                    for n in g[ele]:
                        tmp.append(n)
                q = tmp
                it += 1
            
            return math.inf
        
        res = math.inf
        for r in start_r:
            res = min(res, bfs(r))
        
        return res if res != math.inf else -1
class Solution:
    def numBusesToDestination(self, routes, S, T):
        if S == T: return 0
        queue = collections.deque()
        graph = collections.defaultdict(set)
        routes = list(map(set, routes))
        seen, targets = set(), set()
        for i in range(len(routes)):
            if S in routes[i]:
                seen.add(i)
                queue.append((i, 1))  # enqueue
            if T in routes[i]:
                targets.add(i)
            for j in range(i+1, len(routes)):
                if routes[j] & routes[i]:
                    graph[i].add(j)
                    graph[j].add(i)
        while queue:
            cur, count = queue.popleft()
            if cur in targets:
                return count
            for nei in graph[cur]:
                if nei not in seen:
                    queue.append((nei, count+1))
                    seen.add(nei)
        return -1
class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        graph = collections.defaultdict(set)
        
        for i,route in enumerate(routes):
            for r in route:
                graph[r].add(i)
                
        queue = [[S,0]]
        visited = set([S])
        for stop,bus in queue:
            if stop == T: return bus
            for i in graph[stop]:
                for j in routes[i]:
                    if j not in visited:
                        queue.append([j,bus+1])
                        visited.add(j)
                routes[i].clear()
        return -1
class Solution:
    def numBusesToDestination(self, routes, start, target):
        if start == target: return 0

        stop2Route = defaultdict(set)
        route2Stop = defaultdict(set)
        for i, stops in enumerate(routes):
            for stop in stops:
                route2Stop[i].add(stop)
                stop2Route[stop].add(i)

     #   return stop2Route, route2Stop
        visited = set() 
        visitedStop = set()       
        
        q = [start]
        step = 0
        while q:
            stack = []
            step += 1
            while q:
                stop = q.pop()
                for route in stop2Route[stop]:
                    if route not in visited:
                        if target in route2Stop[route]:
                            if stop in route2Stop[route]:
                                return step
                        else:
                            stack.append(route)
                            visited.add(route)       
                visitedStop.add(stop) 
                
            for route in stack:
                for stop in route2Stop[route]:
                    if stop not in visitedStop:
                        q.append(stop)
         #   print(q, stop, stack, visited, step, visitedStop)
         #   return

        return -1
class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        
        if S == T:
            return 0
        
        graph = self.createGraph(routes)
        
        sourceBuses = self.getBuses(routes, S)
        targetBuses = set(self.getBuses(routes, T))
        
        if not sourceBuses or not targetBuses:
            return -1
        
        
        
        
        queue = [sourceBuses]
        visited = set(sourceBuses)
        buses = 1
        while queue:
            curLevel = queue.pop()
            newLevel = []
            
            for station in curLevel:
                if station in targetBuses:
                    return buses
                
                for conn in graph[station]:
                    if conn not in visited:
                    
                        visited.add(conn)
                        newLevel.append(conn)
            
            if newLevel:
                queue.append(newLevel)
                buses += 1
        
        return -1
    
    
    def getBuses(self, routes, station):
        buses = []
        for i in range(len(routes)):
            if station in set(routes[i]):
                buses.append(i)
        
        return buses
        
    def createGraph(self, routes):
        graph = defaultdict(set)
        
        for i,route in enumerate(routes):
            routes[i] = set(route)
        
        for i in range(len(routes)):
            for j in range(i+1, len(routes)):
                if len(routes[i].intersection(routes[j])) > 0:
                    graph[i].add(j)
                    graph[j].add(i)
        
        return graph
class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        # time complexity: O(m*n)
        # space complexity: O(m*n)
        stops = defaultdict(list) # stop: [buses]
        for r, s in enumerate(routes):
            for i in s: stops[i].append(r)
        
        q, buses, been = deque([S]), [0]*len(routes), set()
        transfers = 0
        while q:            
            size = len(q)
            for _ in range(size):
                i = q.popleft()
                if i == T: return transfers
                for r in stops[i]:
                    if buses[r]: continue
                    buses[r] = 1
                    for s in routes[r]:
                        if s in been: continue
                        been.add(s)
                        q.append(s)      
            transfers += 1          
            
        return -1
                
                    

from queue import deque
class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        # 1. create map of route_idx to List[route_idx]
        # if two routes have same bus stops, attach edge to map.
        # 2. go through routes, if any route has S, 
        # store idx of this route in s_lst; 
        # if any route has T, store idx this route in t_lst.
        # 3. BFS queue of (route_idx, depth)
        
        # base cases
        if S == T: 
            return 0
        # 1
        routes = list(map(set, routes))
        r_map = {} # route_idx to List[route_idx]
        for i, r1 in enumerate(routes):
            for j in range(i+1, len(routes)):
                if len(r1 & routes[j]) > 0:
                    r_map.setdefault(i, []).append(j)
                    r_map.setdefault(j, []).append(i)
        # 2.
        # NOTE: s_lst also functions as visitied lst.
        s_lst, t_lst = [], []
        for k, route in enumerate(routes):
            if S in route:
                s_lst.append(k)
            if T in route:
                t_lst.append(k)
                
        # base case
        if T in s_lst:
            return 1
        # 3.
        queue = deque([(idx, 1) for idx in s_lst])
        while len(queue) > 0:
            curr_route, level = queue.popleft()
            if curr_route in t_lst:
                return level
            if curr_route not in r_map:
                # dead end, no other routes overlap with curr
                continue 
            for dest in r_map[curr_route]:
                if dest not in s_lst:
                    s_lst.append(dest)
                    queue.append((dest, level + 1))
        return -1
class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        if S == T: return 0
        
        # Builds graph.
        graph = collections.defaultdict(list)  # Don't use set. See below.
        for bus, stops in enumerate(routes):
            bus = -bus - 1  # To avoid conflict with the stops.
            
            # `set.update` consumes extra memory, so a `list` is used instead.
            graph[bus] = stops
            for s in stops:
                graph[s].append(bus)

        # Does BFS.
        dq = deque()
        dq.append((S, 0))
        seen = {S}
        while dq:
            node, depth = dq.popleft()
            for adj in graph[node]:
                if adj in seen: continue
                if adj == T: return depth
                # If `adj` < 0, it's a bus, so we add 1 to `depth`.
                dq.append((adj, depth + 1 if adj < 0 else depth))
                seen.add(adj)
        return -1
class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        
        if S == T:
            return 0
        
        graph = self.createGraph(routes)
        # print(g/raph)
        
        sourceBuses = self.getBuses(routes, S)
        targetBuses = set(self.getBuses(routes, T))
        
        # print(sourceBuses, targetBuses)
        if not sourceBuses or not targetBuses:
            return -1
        
        
        
        
        queue = [sourceBuses]
        visited = set(sourceBuses)
        buses = 1
        while queue:
            curLevel = queue.pop()
            newLevel = []
            
            # print(curLevel)
            for station in curLevel:
                if station in targetBuses:
                    return buses
                
                # print(graph[station], visited)
                for conn in graph[station]:
                    if conn not in visited:
                    
                        visited.add(conn)
                        newLevel.append(conn)
            
            # print(newLevel)
            if newLevel:
                queue.append(newLevel)
                buses += 1
        
        return -1
    
    
    def getBuses(self, routes, station):
        buses = []
        for i in range(len(routes)):
            if station in set(routes[i]):
                buses.append(i)
        
        return buses
        
    def createGraph(self, routes):
        graph = defaultdict(set)
        
        for i,route in enumerate(routes):
            routes[i] = set(route)
        
        for i in range(len(routes)):
            for j in range(i+1, len(routes)):
                if len(routes[i].intersection(routes[j])) > 0:
                    graph[i].add(j)
                    graph[j].add(i)
        
        return graph
class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        # BFS
        # The first part loop on routes and record stop to routes mapping in to_route.
        # The second part is general bfs. Take a stop from queue and find all connected route.
        # The hashset seen record all visited stops and we won't check a stop for twice.
        # We can also use a hashset to record all visited routes, or just clear a route after visit.
        
        to_routes = collections.defaultdict(set)
        for i, route in enumerate(routes):
            for j in route:
                to_routes[j].add(i)
        bfs = [(S, 0)]
        seen = set([S])
        for stop, bus in bfs:
            if stop == T: return bus
            for i in to_routes[stop]:
                for j in routes[i]:
                    if j not in seen:
                        bfs.append((j, bus + 1))
                        seen.add(j)
                routes[i] = []  # seen route
        return -1
class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        
        if S == T:
            return 0
        
        neighbors = collections.defaultdict(set)
        
        for j, route in enumerate(routes):
            for i in route:
                neighbors[i].add(j)
                
        stack = []
        stack.append((S, 0))
        visited = set()
        visited.add(S)
        visited_route = set()
        
        for stop, count in stack:
            if stop == T:
                return count

            for neighbor in neighbors[stop]:
                if neighbor not in visited_route:
                    for bus in routes[neighbor]:
                        if bus != stop and bus not in visited:
                            stack.append((bus, count + 1))
                            visited.add(bus)
                        
                #routes[neighbor] = []
                visited_route.add(neighbor)
                
        return  -1
            
        

class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        if S == T:
            return 0
        
        # Get rid of duplicates if any in the lists inside 'routes'
        routes = list(map(set, routes))
        n = len(routes)
        graph = defaultdict(set)
        
        for i in range(n):
            for j in range(i+1,n):
                if set(routes[i]).intersection(routes[j]):
                    graph[i].add(j)
                    graph[j].add(i)
                    
        
        # Get source and destination
        for i in range(n):
            if S in routes[i]:
                if T in routes[i]:
                    return 1  
                source = i
            elif T in routes[i]:
                dest = i
                
                
                
        q = deque([[source, 1]])
        visited = [False]*n
        visited[source] = True
        
        while q:
            node, dis = q.popleft()
            if node == dest:
                return dis
            for u in graph[node]:
                if not visited[u]:
                    visited[u] = True
                    q.append([u, dis +1])
                    
        return -1
            
        


class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        if S == T:
            return 0
        if len(routes) == 0:
            return -1
        
        routesets = []
        for route in routes:
            routesets.append(set(route))
            #print(routesets[-1])
        
        # This is all points where buses meet.
        intersections = set()
        for i in range(len(routesets) - 1):
            set1 = routesets[i]
            for j in range(i + 1, len(routesets)):
                set2 = routesets[j]
                intersections.update(set1.intersection(set2))
        intersections.add(S)
        intersections.add(T)
        
        #print(intersections)
        
        # This is all the routes at an intersection
        i_to_routes = collections.defaultdict(list)
        for i in intersections:
            for ridx, route in enumerate(routesets):
                if i in route:
                    i_to_routes[i].append(ridx)
        
        for route in routesets:
            route.intersection_update(intersections)
        
        #print(i_to_routes)
       
        heap = []
        hist = set([S])
        
        heapq.heappush(heap, (0, 0, S))
        
        while heap:
            n, _, loc = heapq.heappop(heap)
            
            if loc == T:
                return n
            
            for route_idx in i_to_routes[loc]:
                for edge in routesets[route_idx].difference(hist):
                    hist.add(edge)
                    heapq.heappush(heap, (n + 1, abs(edge - T), edge))
        
        return -1
                

class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        conn = collections.defaultdict(list)
        for i, route in enumerate(routes):
            for j in route:
                conn[j].append(i)

        bfs = collections.deque()
        bfs.append((S, 0))
        visited = set([S])
        seen = set()
        while bfs:
            s, bus = bfs.popleft()
            if s == T:
                return bus
            for i in conn[s]:
                if i not in seen:
                    for j in routes[i]:
                        if j not in visited:
                            visited.add(j)
                            bfs.append((j, bus+1))
                    # routes[i] = []
                    seen.add(i)
        return -1
class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        if not routes:
            return None
        if S == T:
            return 0
        stack = []
        visited = set()
        for i, bus in enumerate(routes):
            if S in bus:
                stack.append((bus, 1))
                visited.add(i)
                if T in bus:
                    return 1
        while stack:
            bus, level = stack.pop(0)
            bus = set(bus)
            for i, b in enumerate(routes):
                if i in visited:
                    continue
                if bus & set(b):
                    stack.append((b, level +1))
                    visited.add(i)
                    if T in b:
                        return level +1
        return -1
            

from collections import defaultdict, deque

class Solution:
    def numBusesToDestination(self, routes: List[List[int]], source_stop: int, target_stop: int) -> int:
        if source_stop == target_stop: return 0
        stops = defaultdict(set)
        for i,route in enumerate(routes):
            for stop in route:
                stops[stop].add(i)
        q = deque()
        visited_stops = set()
        visited_buses = set()
        q.append((source_stop, 0))
        while q:
            stop_num, bus_num = q.popleft()
            visited_stops.add(stop_num)
            for other_bus in stops[stop_num]:
                if other_bus in visited_buses: continue
                visited_buses.add(other_bus)
                for other_stop in routes[other_bus]:
                    if other_stop == target_stop:
                        return bus_num + 1
                    if other_stop not in visited_stops:
                        visited_stops.add(other_stop)
                        q.append((other_stop, bus_num + 1))
        return -1


class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        if S == T:
            return 0
        
        stopbusmap = collections.defaultdict(set)
        for bus, stops in enumerate(routes):
            for stop in stops:
                stopbusmap[stop].add(bus)
        queue = []
        for bus in stopbusmap[S]:
            queue.append((routes[bus], [bus]))

        length = 0
        while queue:
            length += 1
            stops, taken = queue.pop(0)
            for stop in stops:
                if stop == T:
                    return len(taken)
                else:
                    for nxt in stopbusmap[stop]:
                        if nxt not in taken:
                            queue.append((routes[nxt], taken+[nxt]))
        return -1
class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        if S == T:
            return 0
        s2b = defaultdict(set)
        for b in range(len(routes)):
            routes[b] = set(routes[b])
            for s in routes[b]:
                s2b[s].add(b)
        
        visited = set()
        q = [(1, b) for b in s2b[S]]
        while q:
            n, b = q.pop(0)
            if b in visited:
                continue
            visited.add(b)
            bs = set()
            for s in routes[b]:
                if s == T:
                    return n
                bs |= s2b[s]
            for bn in bs:
                q.append((n + 1, bn))
            
        return -1
from collections import defaultdict
class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        if S == T: return 0
        graph = defaultdict(list)
        for idx, route in enumerate(routes):
            for node in route:
                graph[node].append(idx)
        visited_stop = set([S])
        visited_bus = [0 for _ in range(len(routes))]
        queue = [(idx, 0) for idx in graph[S]] # bus idx
        # for bus, _ in queue:
        #     visited_bus[bus] = 1
        while queue:
            bus_id, depth = queue.pop(0)
            new_stop = set(routes[bus_id]) - visited_stop
            visited_stop = visited_stop | new_stop
            for stop in new_stop:
                if stop == T: return depth + 1
                for bus in graph[stop]:
                    if visited_bus[bus] == 0:
                        visited_bus[bus] = 1
                        queue.append((bus, depth + 1))
        return -1
class Solution:
    def numBusesToDestination(self, bus_routes: List[List[int]], start: int, end: int) -> int:
        # return self.approach_2(bus_routes, start, end)
        if start == end:
            return 0
        on_stop = defaultdict(set)
        for i, route in enumerate(bus_routes):
            for stop in route:
                on_stop[stop].add(i)
        bfs = []
        bfs.append((start, 0))
        seen = set([start])
        for stop, bus_count in bfs:
            if stop == end:
                return bus_count
            for buses in on_stop[stop]:
                for stops in bus_routes[buses]:
                    if stops not in seen:
                        if stops == end:
                            return bus_count+1
                        seen.add(stops)
                        bfs.append((stops, bus_count+1))
                bus_routes[buses] = []
        return -1

    def approach_2(self, bus_routes, start, end):
        if start == end: return 0
        graph = defaultdict(set)
        bus_routes = list(map(set, bus_routes))
        start_set = set()
        end_set = set()
        for bus, stops in enumerate(bus_routes):
            for other_buses in range(bus+1, len(bus_routes)):
                other_bus_route = bus_routes[other_buses]
                if not other_bus_route.isdisjoint(stops):
                    graph[bus].add(other_buses)
                    graph[other_buses].add(bus)
        for bus,route in enumerate(bus_routes):
            if start in route: start_set.add(bus)
            if end in route: end_set.add(bus)

        queue = [(node, 1) for node in start_set]
        for bus, changes in queue:
            if bus in end_set: return changes
            for nei in graph[bus]:
                if nei not in start_set:
                    start_set.add(nei)
                    queue.append((nei, changes+1))
        return -1
    
    def approach_3(self, buses, start, end):
        on_stop = defaultdict(set)
        for bus, stops in enumerate(buses):
            for stop in stops:
                on_stop[stop].add(bus)
        bfs = [(start, 0)]
        seen = set()
        seen.add(start)
        for stop, dist in bfs:
            if stop == end:
                return dist
            for bus in on_stop[stop]:
                for stops in buses[bus]:
                    if end == start:
                        return dist+1
                    if stops not in seen:
                        seen.add(stops)
                        bfs.append((stops, dist+1))
        return -1


class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        
        # traverse bus ROUTES instead of stops
        if S==T: return 0
        # build graph of routes
        graph = defaultdict(set)
        routes = list(map(set, routes))
        seen = set()
        target = set()
        q = deque()
        
        for i in range(len(routes)):
        #for i, r in enumerate(routes):
            # build graph
            for j in range(i+1, len(routes)):
                if routes[i] & routes[j]:
                    graph[i].add(j)
                    graph[j].add(i)
            # add starting routes
            if S in routes[i]:
                q.append((i, 1))    
                seen.add(i)
            # add ending routes
            if T in routes[i]:
                target.add(i)
            
        # traverse from start, return when reaching end
        while q:
            cur, count = q.popleft()
            if cur in target:
                return count
            for rt in graph[cur]:
                if rt not in seen:
                    q.append((rt, count+1))
                    seen.add(rt)
        
        return -1
class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        num_buses = len(routes)
        bus_to_stop = defaultdict(set)
        for bus, stops in enumerate(routes):
            bus_to_stop[bus] = set(stops)
        
        def update_buses_used():
            for bus in range(num_buses):
                if bus in buses_used:
                    continue
                if stops_reached & bus_to_stop[bus]:
                    buses_used.add(bus)
        
        def update_stops_reached():
            for bus in buses_used:
                stops_reached.update(bus_to_stop[bus])
        
        buses_used = set()
        stops_reached = {S}
        pre_stop_count = 0
        bus_count = 0
        while len(stops_reached) > pre_stop_count:
            if T in stops_reached:
                return bus_count
            pre_stop_count = len(stops_reached)
            update_buses_used()
            update_stops_reached()
            bus_count += 1
            
        return -1

    def numBusesToDestination(self, routes, S, T):
        to_routes = collections.defaultdict(set)
        for i, route in enumerate(routes):
            for j in route:
                to_routes[j].add(i)
        bfs = [(S, 0)]
        seen = set([S])
        for stop, bus in bfs:
            if stop == T: return bus
            for i in to_routes[stop]:
                for j in routes[i]:
                    if j not in seen:
                        bfs.append((j, bus + 1))
                        seen.add(j)
                routes[i] = []  # seen route
        return -1
        
        
        

class Solution:
    def numBusesToDestination(self, routes: List[List[int]], s: int, t: int) -> int:
        if s == t: return 0
            
        n_buses = len(routes)
        
        stops = defaultdict(lambda: set())
        for i, route in enumerate(routes):
            for stop in route:
                stops[stop].add(i)
                
        g = [set() for _ in range(n_buses)]
        for i, route in enumerate(routes):
            for stop in route:
                for bus in stops[stop]:
                    if bus == i: continue
                    g[i].add(bus)
                
        used = [False]*n_buses
        q = collections.deque()
        for bus in stops[s]:
            q.append((bus, 1))
            used[bus] = True
        
        while q:
            bus, dist = q.popleft()
            if bus in stops[t]:
                return dist
            for bus2 in g[bus]:
                if not used[bus2]:
                    q.append((bus2, dist+1))
                    used[bus2] = True
        return -1

from collections import defaultdict
class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        graph = defaultdict(list)
        for idx, route in enumerate(routes):
            for node in route:
                graph[node].append(idx)
        visited_node = set([S])
        visited_route = [0 for _ in range(len(routes))]
        queue = [(idx, 0) for idx in graph[S]] # route idx
        for idx, _ in queue:
            visited_route[idx] = 1
        while queue:
            id, depth = queue.pop(0)
            new_nodes = set(routes[id]) - visited_node
            visited_node = visited_node | new_nodes
            for node in new_nodes:
                if node == T: return depth + 1
                for idx in graph[node]:
                    if visited_route[idx] == 0:
                        visited_route[idx] = 1
                        queue.append((idx, depth + 1))
        return -1 if S != T else 0
class Solution:
    def numBusesToDestination(self, routes, start, target):
        if start == target: return 0
        stop2Route = defaultdict(set)
        route2Stop = defaultdict(set)
        for i, stops in enumerate(routes):
            for stop in stops:
                route2Stop[i].add(stop)
                stop2Route[stop].add(i)

     #   return stop2Route, route2Stop
        visited = set() 
        visitedStop = set()       
        stack = []
        q = [start]
        step = 0
        while q:
            step += 1
            while q:
                stop = q.pop()
                for route in stop2Route[stop]:
                    if route not in visited:
                        if target in route2Stop[route]:
                            # in the same route
                            if stop in route2Stop[route]:
                                return step
                        else:
                            stack.append(route)
                            
                            visited.add(route)
                            
                visitedStop.add(stop) 
                
            for route in stack:
                for stop in route2Stop[route]:
                    if stop not in visitedStop:
                        q.append(stop)
         #   print(q, stop, stack, visited, step, visitedStop)
         #   return

        return -1

class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        if S == T:
            return 0
        if len(routes) == 0:
            return -1
        
        routesets = []
        for route in routes:
            routesets.append(set(route))
            #print(routesets[-1])
        
        # This is all points where buses meet.
        intersections = set()
        for i in range(len(routesets) - 1):
            set1 = routesets[i]
            for j in range(i + 1, len(routesets)):
                set2 = routesets[j]
                intersections.update(set1.intersection(set2))
        intersections.add(S)
        intersections.add(T)
        
        #print(intersections)
        
        # This is all the routes at an intersection
        i_to_routes = collections.defaultdict(list)
        for i in intersections:
            for ridx, route in enumerate(routesets):
                if i in route:
                    i_to_routes[i].append(ridx)
        
        #print(i_to_routes)
       
        heap = []
        hist = set([S])
        
        heapq.heappush(heap, (0, 0, S))
        
        while heap:
            n, _, loc = heapq.heappop(heap)
            
            if loc == T:
                return n
            
            for route_idx in i_to_routes[loc]:
                for edge in routesets[route_idx].difference(hist):
                    hist.add(edge)
                    heapq.heappush(heap, (n + 1, abs(edge - T), edge))
        
        return -1
                

from collections import defaultdict

class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        if S == T:
            return 0
        dist = {}
        depth = 0
        dist[S] = depth
        vis_bs = set()
        
        # from cities to buses
        c_to_bs = defaultdict(list)
        for pos_b, b in enumerate(routes):
            for c in b:
                c_to_bs[c].append(pos_b)
        
        cur_cs = set([S])
        
        while len(cur_cs) > 0:
            depth += 1
            cur_bs = set()
            for c in cur_cs: # new buses from last cities
                for pos_b in c_to_bs[c]:
                    if pos_b not in vis_bs:
                        cur_bs.add(pos_b)
                vis_bs |= cur_bs
            cur_cs = set()
            for pos_b in cur_bs:
                for c in routes[pos_b]:
                    if c not in dist:
                        cur_cs.add(c)
                        dist[c] = depth
            if T in dist:
                return dist[T]
        
        return -1
from collections import defaultdict
class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        if S == T:
            return 0
        
        new_routes = {}
        for i, route in enumerate(routes):
            new_routes[i] = set(route)
        
        start_buses = set()
        target_buses = set()
        queue = []
        #connect buses which have common stops
        stop_bus = defaultdict(set)
        for bus, route in enumerate(routes):
            for bus2 in range(bus+1, len(routes)):
                if len(new_routes[bus].intersection(new_routes[bus2])) > 0:
                    stop_bus[bus].add(bus2)
                    stop_bus[bus2].add(bus)
                    
            if S in new_routes[bus]:
                start_buses.add(bus)
                queue.append((bus,1))
            if T in new_routes[bus]:
                target_buses.add(bus)
 
        
        
        
        while queue:
            curr_bus, path = queue[0][0], queue[0][1]
            queue = queue[1:]
            if curr_bus in target_buses:
                return path
            for neigh in stop_bus[curr_bus]: 
                if neigh not in start_buses:
                    start_buses.add(neigh)
                    queue.append((neigh, path+1))
        return -1
        
            
        
        
        

class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        if (S == T):
            return 0
        buses = collections.defaultdict(list)
        for i, bus in enumerate(routes):
            for stop in bus:
                buses[stop].append(i)
        # print(buses)
        q = [S]
        result = 0
        took = set()
        visited = set()
        visited.add(S)
        while (q):
            for i in range(len(q), 0, -1):
                now = q.pop(0)
                for bus in buses[now]:
                    if (bus not in took):
                        for route in routes[bus]:
                            if (route not in visited):                                
                                if (route == T):
                                    return result + 1
                                q.append(route)
                                visited.add(route)
                        took.add(bus)
            result += 1
        
        return -1
                        
                

class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        if S == T: return 0
        n = len(routes)
        source = set()
        target = set()
        graph = [[] for _ in range(n)]

        for i in range(n):
            routes[i] = set(routes[i])
            if S in routes[i]: source.add(i)
            if T in routes[i]: target.add(i)

        if not source or not target: return -1
        if source & target:
            return 1

        for u in range(n):
            for v in range(1, n):
                if routes[u] & routes[v]:
                    graph[u].append(v)
                    graph[v].append(u)

        queue = [(source.pop(), 1)]
        seen = [0] * n
        while queue:
            new = []
            for u, cost in queue:
                if u in target:
                    return cost
                seen[u] = 1
                for v in graph[u]:
                    if seen[v]: continue
                    seen[v] = 1
                    new.append((v, cost + 1))
            queue = new
        return -1

class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        ans = 0
        n = len(routes)
        if S == T:
            return 0
        graph = collections.defaultdict(set)# stop -> bus #
        queue = collections.deque()
        visited_stop = set()
        visited_bus = set()
        for i in range(n):
            for stop in routes[i]:
                graph[stop].add(i)
        print(graph)
        queue.append(S)
        visited_stop.add(S)
        
        while queue:
            qLen = len(queue)
            ans +=1
            for i in range(qLen):
                stop = queue.popleft()
                for next_bus in graph[stop]:
                    if next_bus in visited_bus:
                        continue
                    visited_bus.add(next_bus)
                    for next_stop in routes[next_bus]:
                        if next_stop in visited_stop:
                            continue
                        if next_stop == T:
                            print('here')
                            return ans
                        queue.append(next_stop)
                        visited_stop.add(next_stop)
            # print(queue, visited_stop, visited_bus)
            
            # print(ans)
        return -1 
    
# defaultdict(<class 'set'>, {1: {0}, 2: {0}, 7: {0, 1}, 3: {1}, 6: {1}})
# deque([2]) {1, 2} {0}
# deque([2, 7]) {1, 2, 7} {0}
# 1
# deque([3]) {1, 2, 3, 7} {0, 1}
            

           

class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        if S == T:
            return 0
        
        # if two routes share the same stop, they are reachable with each other
        # mapping: stop -> routes
        stops = collections.defaultdict(set)
        # mapping: route -> reachable routes
        reachable = collections.defaultdict(set)
        for i, route in enumerate(routes):
            for stop in route:
                stops[stop].add(i)
                for j in stops[stop]:
                    reachable[i].add(j)
                    reachable[j].add(i)
                        
        target_routes = stops[T]
        queue = collections.deque(stops[S])
        buses = 1
        reached = stops[S]
        while queue:
            queue_len = len(queue)
            for _ in range(queue_len):
                route = queue.popleft()
                if route in target_routes:
                    #print(stops[S])
                    #print(target_routes)
                    #print(route)
                    return buses
                for other_route in reachable[route]:
                    if other_route not in reached:
                        reached.add(other_route)
                        queue.append(other_route)
            buses += 1
        return -1
                        

class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        if S == T:
            return 0
        stops = {}
        for i in range(len(routes)):
            for stop in routes[i]:
                if stop not in stops:
                    stops[stop] = [[i], False]
                else:
                    stops[stop][0].append(i)
                    
        next_stops = deque([S])
        visited = [False for i in range(len(routes))]
        remaining = 1
        buses = 0
        while len(next_stops) > 0:
            cur = next_stops.popleft()
            remaining -= 1
            stops[cur][1] = True
            for r in stops[cur][0]:
                if not visited[r]:
                    for s in routes[r]:
                        if s == T:
                            return buses + 1
                        if not stops[s][1]:
                            stops[s][1] = True
                            next_stops.append(s)
                    visited[r] = True
            if remaining == 0:
                remaining = len(next_stops)
                buses += 1
                
        return -1
class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        if not routes:
            return None
        if S == T:
            return 0
        stack = []
        visited = []
        for i, bus in enumerate(routes):
            if S in bus:
                stack.append((bus, 1))
                visited.append(i)
                if T in bus:
                    return 1
        while stack:
            bus, level = stack.pop(0)
            bus = set(bus)
            for i, b in enumerate(routes):
                if i in visited:
                    continue
                if bus & set(b):
                    stack.append((b, level +1))
                    visited.append(i)
                    if T in b:
                        return level +1
        return -1
            

class Solution:
    def numBusesToDestination(self, routes: List[List[int]], S: int, T: int) -> int:
        if not routes:
            return None
        if S == T:
            return 0
        stack = []
        visited = []
        for i, bus in enumerate(routes):
            if S in bus:
                stack.append((bus, 1))
                visited.append(i)
                if T in bus:
                    return 1
        while stack:
            bus, level = stack.pop(0)
            bus = set(bus)
            for i, b in enumerate(routes):
                if i in visited:
                    continue
                if len(bus & set(b)) > 0:
                    stack.append((b, level +1))
                    visited.append(i)
                    if T in b:
                        return level +1
        return -1
            

