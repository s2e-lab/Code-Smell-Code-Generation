class Solution:
    def avoidFlood(self, rains: List[int]) -> List[int]:
        fullLake = {}
        dry = {}

        for day, lake in enumerate(rains):
            if lake not in fullLake:
                if lake:
                    fullLake[lake]=day
            else:
                if lake:
                    dry[fullLake[lake]] = day
                    fullLake[lake]=day
        heap=[]
        for day, lake in enumerate(rains):
            if heap and day >= heap[0][0]:
                return []
            if lake:
                if day in dry:
                    heapq.heappush(heap, (dry[day], lake))
                rains[day] = -1
            else:
                if heap:
                    rains[day] = heapq.heappop(heap)[1]
                else:
                    rains[day] = 1
        return rains
class Solution:
    def avoidFlood(self, rains: List[int]) -> List[int]:
        filled = {}
        deadline = {}

        for day, lake in enumerate(rains):
            if lake not in filled:
                if lake:
                    filled[lake] = day
            else:
                if lake:
                    deadline[filled[lake]] = day
                    filled[lake] = day

        heap = []

        for day, lake in enumerate(rains):
            if heap and day >= heap[0][0]:
                return []
            if lake:
                if day in deadline:
                    heapq.heappush(heap, (deadline[day], lake))
                rains[day] = -1
            else:
                if heap:
                    rains[day] = heapq.heappop(heap)[1]
                else:
                    rains[day] = 1
        return rains
from heapq import heappush,heappop
class Solution:
    def avoidFlood(self, rains: List[int]) -> List[int]:
         d=collections.defaultdict(list);last=[]
         for idx,lake in enumerate(rains):
              d[lake].append(idx)
         ans=[]
         for idx,lake in enumerate(rains):
             if lake:
                if last and last[0]==idx:
                    return []
                arr=d[lake]
                arr.pop(0)
                if arr:
                    heappush(last,arr[0])
                ans.append(-1)
             else:
                 if last:
                    ans.append(rains[heappop(last)])
                 else:
                    ans.append(1)
         return ans
from sortedcontainers import SortedSet 

class Solution:
    def avoidFlood(self, rains: List[int]) -> List[int]:
        could_dry = SortedSet()
        last_rain = {}
        dry = {}
        for idx, rain in enumerate(rains):
            if rain == 0:
                could_dry.add(idx)
            else:
                if rain not in last_rain:
                    last_rain[rain] = idx
                else:  # need to find a dry
                    i = could_dry.bisect_left(last_rain[rain])
                    if i == len(could_dry):
                        return []  # could not find a good dry day
                    else:
                        day = could_dry[i]
                        dry[day] = rain
                        could_dry.remove(day)
                    last_rain[rain] = idx
        res = []
        for idx, rain in enumerate(rains):
            if rain == 0:
                res.append(dry.get(idx, 1))
            else:
                res.append(-1)
        return res
            

'''
class Solution:
    def avoidFlood(self, rains: List[int]) -> List[int]:
        n = len(rains)
        ret = []
        def dfs(i, has_water, ans):
            i#print(i, has_water,ans)
            nonlocal ret
            if ret: return
            if i == n:  ret = ans; return 
            if rains[i] in has_water:  return 
            if rains[i]:  dfs(i+1, has_water|set([rains[i]]), ans+[-1])
            else:
                if not has_water: dfs(i+1, has_water, ans+[1])
                for lake in has_water:
                    has_water.remove(lake)
                    dfs(i+1, has_water, ans+[lake])
                    has_water.add(lake)
        dfs(0, set(), [])
        return ret
                    
'''            
            
class Solution:
    def avoidFlood(self, rains: List[int]) -> List[int]:
        # min heap to store the days when flooding would happen (if lake not dried)
        nearest = []
        # dict to store all rainy days
        # use case: to push the subsequent rainy days into the heap for wet lakes
        locs = collections.defaultdict(collections.deque)
        # result - assume all days are rainy
        res = [-1] * len(rains)
        
        # preprocessing - {K: lake, V: list of rainy days}
        for i, lake in enumerate(rains):
            locs[lake].append(i)
            
        for i, lake in enumerate(rains):
            # the nearest lake got flooded (termination case)
            if nearest and nearest[0] == i:
                return []
            
            # lake got wet
            if lake != 0:
                # pop the wet day
                locs[lake].popleft()
                
                # priotize the next rainy day for this lake
                if locs[lake]:
                    nxt = locs[lake][0]
                    heapq.heappush(nearest, nxt)
            # a dry day
            else:
                # no wet lake, append an arbitrary value
                if not nearest:
                    res[i] = 1
                else:
                    # dry the lake that has the highest priority
                    # since that lake will be flooded in nearest future otherwise (greedy property)
                    next_wet_day = heapq.heappop(nearest)
                    wet_lake = rains[next_wet_day]
                    res[i] = wet_lake
        
        return res
class Solution:
    def avoidFlood(self, rains: List[int]) -> List[int]:
        closest = []
        locations = collections.defaultdict(collections.deque)
        for index, lake in enumerate(rains):
            locations[lake].append(index)
        res = []
        for index, lake in enumerate(rains):
            if closest and closest[0] == index:
                return []
            if not lake:
                if not closest:
                    res.append(1) 
                    continue
                nxt = heapq.heappop(closest)
                res.append(rains[nxt])
            else:
                l = locations[lake]
                l.popleft()
                if l:
                    nxt = l[0]
                    heapq.heappush(closest, nxt)
                res.append(-1)
        return res
from collections import deque, defaultdict
import heapq

class Solution:
    def avoidFlood(self, rains: List[int]) -> List[int]:
        rq = defaultdict(deque)
        for i, r in enumerate(rains):
            rq[r].append(i)
        for r, q in list(rq.items()):
            q.popleft()
        ans = [-1] * len(rains)
        urgent = []
        for i, r in enumerate(rains):
            if r != 0:
                q = rq[r]
                if q:
                    heapq.heappush(urgent, q.popleft())
            elif urgent:
                d = heapq.heappop(urgent)
                if d < i:
                    return []
                ans[i] = rains[d]
            else:
                ans[i] = 1
        return ans if not urgent else []

class Solution:
    def avoidFlood(self, rains: List[int]) -> List[int]:
        lakes = {}
        dries = []
        res = []
        for i, rain in enumerate(rains):
            if rain == 0:
                dries.append(i)
                res.append(1)
            else:
                if rain in lakes:
                    if len(dries) == 0:
                        return []
                    idx = bisect_left(dries, lakes[rain])
                    if idx == len(dries):
                        return []
                    res[dries.pop(idx)] = rain
                lakes[rain] = i
                res.append(-1)
        return res
class Solution:
    def avoidFlood(self, rains: List[int]) -> List[int]:
        lakes = {}
        dries = []
        res = [-1] * len(rains)
        for i, rain in enumerate(rains):
            if rain == 0:
                if lakes:
                    dries.append(i)
                else:
                    res[i] = 1
            elif rain in lakes:
                if len(dries) == 0:
                    return []
                idx = bisect_right(dries, lakes[rain])
                if idx == len(dries):
                    return []
                res[dries.pop(idx)] = rain
                lakes[rain] = i
            else:
                lakes[rain] = i
        for d in dries:
            if lakes:
                res[d] = lakes.popitem()[0]
            else:
                res[d] = 1
        return res
from collections import defaultdict, deque
import heapq

class Solution:
    def avoidFlood(self, rains: List[int]) -> List[int]:
        
        res = [-1] * len(rains)
        dic = defaultdict(deque)
        
        for idx, rain in enumerate(rains):
            dic[rain].append(idx)
        
        pq = []
        full = set()
        for i in range(len(rains)):
            
            if rains[i]:
                if rains[i] in full:
                    return []
                else:
                    full.add( rains[i] )
                    dic[ rains[i] ].popleft()
                if dic[ rains[i] ]:
                    heapq.heappush( pq, dic[ rains[i] ][0] )
                    
            else:
                if not pq:
                    res[i] = 1
                else:
                    lake = rains[ heapq.heappop(pq) ]
                    res[i] = lake
                    full.discard(lake)
        
        return res
                

class Solution:
    def avoidFlood(self, rains: List[int]) -> List[int]:
        floods, n, visited, water_heap = None, len(rains), defaultdict(int), []
        
        for i in range(n - 1, -1, -1):
            if rains[i]:
                if rains[i] in visited:
                    floods = i, visited[rains[i]], floods
                visited[rains[i]] = i
        
        for i in range(n):
            if not rains[i]:
                while floods and floods[0] < i:
                    start, end, floods = floods
                    heapq.heappush(water_heap, (end, start))
                
                if not water_heap:
                    rains[i] = 1
                else:
                    end,start = heappop(water_heap)
                    if end < i: return []
                    rains[i] = rains[end]
            else:
                rains[i] = -1
        
        
        if water_heap or floods:
            return []
        return rains
class Solution:
    def avoidFlood(self, rains: List[int]) -> List[int]:
        full= set()
        priority = []
        toReturn = []
        for i in rains:
            if i != 0:
                if i in full: priority.append(i)
                else: full.add(i)
        full = set()
        for i in rains:
            if i == 0:
                done = False
                for x in range(len(priority)):
                    if priority[x] in full:
                        a = priority.pop(x)
                        toReturn.append(a)
                        full.remove(a)
                        done = True
                        break
                if not done:
                    toReturn.append(1)
            elif i in full: return []
            else:
                full.add(i)
                toReturn.append(-1)
        return toReturn
                

class Solution:
    def avoidFlood(self, rains: List[int]) -> List[int]:
        """
        #O(n^2) working sol
        ans = [1 for i in range(len(rains))]
        d = collections.defaultdict(int)
        d[0]=0
        
        for i in range(len(rains)):
            d[rains[i]]+=1
            if rains[i]==0:
                #look for the nearest value that exists in the dict we got
                for x in range(i+1,len(rains)):
                    if rains[x] in d and not rains[x]==0:
                        #print(d,d[rains[x]],rains[x])
                        d[0]-=1
                        ans[i] = rains[x]
                        d[rains[x]]-=1
                        if d[rains[x]]==0: del d[rains[x]]
                        break
            else:
                #you gotta get out early of a bad pattern that cannot be salvaged
                if d[rains[i]]>1:
                    return []
                ans[i] = -1
        
        return ans
        """
        
        ans = [1 for i in range(len(rains))]
        d = collections.defaultdict(int)
        d[0]=0
        #preprosess, find all  #:0#0#0...
        # as d grows, put corresponding value here in a heap
        # every time heap pops, we get the nearest value that exists in the dict we got
        p = {}
        x = collections.defaultdict(int)
        x[0] = 0
        for i in range(len(rains)):
            if rains[i] in p:
                #print(x[0],rains[i],x[rains[i]])
                if x[0]>=x[rains[i]]:
                    p[rains[i]] += [i]
            else:
                p[rains[i]] = []
            x[rains[i]]+=1
        p[0] = []
            
        #print(p)       
            
        s= set()
        h = []
        for i in range(len(rains)):

            d[rains[i]]+=1

            if rains[i]!=0 and rains[i] not in s:
                if rains[i] in p and p[rains[i]] != []:
                    for j in p[rains[i]]:
                        heappush(h,j)
                s.add(rains[i])
            #print(d,h)
             
            if rains[i]==0:
                #look for the nearest value that exists in the dict we got
                if h:
                    pop = heappop(h)
                    d[0]-=1
                    
                    ans[i] = rains[pop]
                    if rains[pop] not in d:
                        rains[pop] = 1
                    else:
                        d[rains[pop]]-=1
                    if d[rains[pop]]==0: del d[rains[pop]] 
            else:
                
                #you gotta get out early of a bad pattern that cannot be salvaged
                if d[rains[i]]>1:
                    return []
                #find the next equal closest value past a zero.
                ans[i] = -1
            #print(h,i,"heap at end")
        
        return ans
        
        
        
        
                
        
                
            
            
        
            
        
                        
                        
        
                
       
        
                
                
                
class Solution:
    def avoidFlood(self, rains: List[int]) -> List[int]:
        N=len(rains)
        ans=[-1]*N
        drydays=[]
        last={}
        for i,e in enumerate(rains):
            if e==0:
                drydays+=i,
            else:
                if e in last:
                    lastIndex=last[e]
                    j=bisect_right(drydays,lastIndex)
                    if j<len(drydays):
                        ans[drydays[j]]=e
                        del drydays[j]
                    else:
                        return []
                last[e]=i
                
        #populate drydays
        for d in drydays:
            ans[d]=1
        return ans
def avoid(rains):
    zeros = []
    full = {}
    sol = [-1]*len(rains)

    for ix,r in enumerate(rains):
        if r == 0:
            zeros.append(ix)
        elif r not in full:
            full[r] = ix
        else:
            # we're gonna have a flood
            if not zeros:
                return []

            zix = None
            rix = full[r]
            for i in range(len(zeros)):
                if zeros[i] > rix:
                    zix = zeros.pop(i)
                    break

            if not zix:
                return []

            sol[zix] = r
            full[r] = ix # update filling day

    while zeros:
        sol[zeros.pop()] = 1

    return sol



class Solution:
    def avoidFlood(self, rains: List[int]) -> List[int]:
        return avoid(rains)

class Solution:
    def avoidFlood(self, rains: List[int]) -> List[int]:
        ans = [-1] * len(rains)
        lake_to_days = collections.defaultdict(list)
        full_lakes = set()
        to_empty = []
        
        for day, lake in enumerate(rains):
            lake_to_days[lake].append(day)
        
        for day in range(len(rains)):
            lake = rains[day]
            if lake:
                if lake in full_lakes:
                    return []
                full_lakes.add(lake)
                lake_to_days[lake].pop(0)
                if lake_to_days[lake]:
                    heapq.heappush(to_empty, lake_to_days[lake][0])
            else:
                if to_empty:
                    ans[day] = rains[heapq.heappop(to_empty)]
                    full_lakes.remove(ans[day])
                else:
                    ans[day] = 1
        return ans
        
        
                    
        
        
        
            

class Solution:
    def avoidFlood(self, rains: List[int]) -> List[int]:
        fill = {}
        dry = []
        ans = [0]*len(rains)
        for i,v in enumerate(rains):
            if v == 0:
                dry.append(i)
                continue
            if v not in fill:
                fill[v] = i
                ans[i] = -1
            elif v in fill:
                idx = bisect.bisect_left(dry,fill[v])
                if idx == len(dry):
                    return []
                else:
                    ans[dry[idx]] = v
                    dry.pop(idx)
                fill[v] = i
                ans[i] = -1
        for i in range(len(ans)):
            if ans[i] == 0:
                ans[i] = 1
        return ans
# 1488. Avoid Flood in The City

import heapq

'Your country has an infinite number of lakes.'
'Initially, all lakes are empty, but when it rains over the nth lake, the nth lake becomes full of water.'
'If it rains over a full lake, there will be a flood.'
'Your goal is to avoid flood.'
'On each dry day you may choose to dry one lake.'

def drying_strategy (rains):
    # Idea: greedy method
    # Always dry the lake which is most urgent.
    # Never dry already-dry lakes, unless no lakes are full.

    # Alternate method: orthogonal.
    # Consider each lake and how many times you should dry it.
    # Ignore any lake that is only filled once.

    # Maintain a heap of (lake, urgency) for filled lakes.
    # Whenever a lake is filled, its urgency is pushed into the heap.
    # Whenever we can dry a lake, we take the most urgent task on the top of the heap.

    last_rain = {} # updating
    chain_rain = {} # persistent, link from one rain to next

    for time, rain in enumerate (rains):
        if rain > 0:
            if rain in last_rain:
                chain_rain[last_rain[rain]] = time
            last_rain[rain] = time

    del last_rain

    urgency = []
    filled = set ()

    solution = []

    for time, rain in enumerate (rains):
        if rain > 0:
            if rain in filled:
                # flooded
                return []
            else:
                filled.add (rain)
                if time in chain_rain: # has next rain
                    heapq.heappush (urgency, chain_rain[time])
                    # add next rain
            solution.append (-1) # wait
        else:
            # clear day. find the next urgent and dry.
            solved = False
            while not solved:
                if not urgency:
                    solution.append (1)
                    if 1 in filled: filled.remove (1)
                    solved = True
                else:
                    time = heapq.heappop (urgency)
                    rain = rains[time]
                    if rain not in filled:
                        pass # nothing to worry about
                    else:
                        solution.append (rain)
                        filled.remove (rain)
                        solved = True

    return solution


class Solution:
    def avoidFlood(self, rains: List[int]) -> List[int]:
        return drying_strategy(rains)
class Solution:
    def avoidFlood(self, rains: List[int]) -> List[int]:
        '''
Logics:
We dry lakes in the order of urgency - Greedy.
Iterating through days, when day i is raining on lake lake, if lake is already full, simply return []; else, push the next raining day for lake to to_empty to queue it for drying.
When day i is sunny, dry the most urgent lake referring to to_empty. Remember to remove it from full.
        '''
        # dic stores the raining day for each lake in ascending order.
        dic = collections.defaultdict(list)
        for day,lake in enumerate(rains):
            dic[lake].append(day)
            
        res = [-1] * len(rains)
        to_empty = [] # index,Min-heap and records the lakes that are full and sorted in urgency order.
        for i in range(len(rains)):
            lake = rains[i]
            if lake:
                if dic[lake] and dic[lake][0] < i:
                    return []
                if dic[lake] and len(dic[lake])>1:
                    heapq.heappush(to_empty,dic[lake][1])
            else:
                if to_empty:
                    res[i] = rains[heapq.heappop(to_empty)]
                    dic[res[i]].pop(0)
                else:
                    res[i] = 1
        return res
class Solution:
    def avoidFlood(self, rains: List[int]) -> List[int]:
        
        output = [-1] * len(rains)
        
        dic = dict()
        stack = []
        
        for i, n in enumerate(rains):
            if n == 0:
                stack.append(i)
            elif n not in dic:
                dic[n] = i
            else:
                if not stack:
                    return []
                
                for j in stack:
                    if j > dic[n]:
                        break
                
                if j < dic[n]:
                    return []
                # print(stack)
                stack.pop(stack.index(j))
                output[j] = n
                dic[n] = i
        
        for j in stack:
            output[j] = 1
        
        return output
"""
[1,2,3,4]
rain[i] = 0 -> u62bdu6c34u65e5
    dryDay.insert(i)
rain[i] = x
1) x is empty: fill[x] = i
2) x is full: when to drain x?
    must be in dryDays
    must be later than fill[x]
    
      1  2  3  4  5  6
fill  x     y     x 
dryD     -     x 


"""



class Solution:
    def avoidFlood(self, rains):
        filled = {}
        dryDays = []
        res = [1] * len(rains)

        for day, lake in enumerate(rains):
            if not lake:
                dryDays.append(day)
                continue 

            res[day] = -1
            if lake in filled:
                if not dryDays: return []
                # use the first dry day after the lake was filled (stored in filled[lake])
                idx = bisect.bisect_left(dryDays, filled[lake])
                if idx >= len(dryDays): 
                    return []
                dry_on_day = dryDays.pop(idx)
                res[dry_on_day] = lake

            filled[lake] = day # we fill it on day

        return res
        
        
        
        
        
class Solution:
    def avoidFlood(self, rains: List[int]) -> List[int]:

        res = []
        free_day = [] # days without rain
        filled = {}   # map of cities that are full (but not flooded) -> day that they were filled

        for day,city in enumerate(rains):
            if city: 
                res.append(-1)
                if city not in filled:                                       # 1
                    filled[city] = day                                       # 1
                else:
                    if free_day and (free_day[-1] > filled[city]):           # 3.1
                        dry_day = bisect.bisect_left(free_day, filled[city]) # 3.3
                        res[free_day.pop(dry_day)] = city                    # 3.3
                        filled[city] = day                                   # 3.3
                    else:
                        return []                                            # 3.2
            else:
                res.append(1)                                                # 2 (we will fill in rain free days later ~ use city 1 as a place holder)
                free_day.append(day)                                         # 2

        return res
                        

class Solution:
    def avoidFlood(self, rains: List[int]) -> List[int]:
        """
            using deque canDry to store the possible days that can be used to dry a lake
            using hasRain to store the lake : day pairs
            update rules:
                1) if lake rains[i] rains on day i, check if it has rained before or not
                    if it has rained before, check if there is a way to dry it 
                        binary search the interval between two rain days
                    if there is no way to dry it, return []
                2) if there is no rain on day i, put i in canDry
        """
        dry, res, rained = [], [], {}
        
        for i, lake in enumerate(rains):
            if lake > 0: # lake rains on day i
                res.append(-1)
                if lake in rained: # lake has rained before
                    idx = bisect.bisect(dry, rained[lake]) # search for the index of the 
                    if idx < len(dry): # a valid day is found to dry lake
                        day = dry[idx]
                        res[day] = lake
                        dry.pop(idx)
                        rained[lake] = i
                    else:
                        return []
                else: # lake has not rained before
                    rained[lake] = i
            else: # no rain on day i
                dry.append(i)
                res.append(0)
        
        for day in dry:
            res[day] = 1
        return res
       
class Solution:
    def avoidFlood(self, rains: List[int]) -> List[int]:
        n = len(rains)
        ans = [-1] * n
        
        last = {}
        
        dry_days = []
        
        for idx, r in enumerate(rains):
            if r == 0:
                dry_days.append(idx)
            else:
                if r in last:
                    found = False
                    j = 0
                    while j < len(dry_days):
                        if dry_days[j] > last[r]:
                            ans[dry_days[j]] = r
                            found = True
                            break
                        j += 1
                    
                    if not found:
                        return []
                    dry_days.pop(j)
                last[r] = idx
        
        while dry_days:
            dry_day = dry_days.pop()
            ans[dry_day] = 1
        
        return ans

class Solution:
    def avoidFlood(self, rains: List[int]) -> List[int]:
        ans = [-1] * len(rains)
        nearest = []
        locs = defaultdict(deque)
        for i, lake in enumerate(rains):
            locs[lake].append(i)
        for i, lake in enumerate(rains):
            if nearest and nearest[0] == i:
                return []
            if lake == 0:
                if not nearest:
                    ans[i] = 1
                else:
                    n = heappop(nearest)
                    ans[i] = rains[n]
            else:
                locs[lake].popleft()
                if locs[lake]:
                    heappush(nearest, locs[lake][0])
        return ans
from collections import Counter, defaultdict, OrderedDict, deque
from bisect import bisect_left, bisect_right
from functools import reduce, lru_cache
from typing import List
import itertools
import math
import heapq
import string
true = True
false = False
MIN, MAX, MOD = -0x3f3f3f3f, 0x3f3f3f3f, 1000000007


#
# @lc app=leetcode id=1488 lang=python3
#
# [1488] Avoid Flood in The City
#
# https://leetcode.com/problems/avoid-flood-in-the-city/description/
#
# algorithms
# Medium (25.27%)
# Total Accepted:    9.6K
# Total Submissions: 38.1K
# Testcase Example:  '[1,2,3,4]'
#
# Your country has an infinite number of lakes. Initially, all the lakes are
# empty, but when it rains over the nth lake, the nth lake becomes full of
# water. If it rains over a lake which is full of water, there will be a flood.
# Your goal is to avoid the flood in any lake.
#
# Given an integer array rains where:
#
#
# rains[i] > 0 means there will be rains over the rains[i] lake.
# rains[i] == 0 means there are no rains this day and you can choose one lake
# this day and dry it.
#
#
# Return an array ans where:
#
#
# ans.length == rains.length
# ans[i] == -1 if rains[i] > 0.
# ans[i] is the lake you choose to dry in the ith dayu00a0if rains[i] == 0.
#
#
# If there are multiple valid answers return any of them. If it is impossible
# to avoid flood return an empty array.
#
# Notice that if you chose to dry a full lake, it becomes empty, but if you
# chose to dry an empty lake, nothing changes. (see example 4)
#
#
# Example 1:
#
#
# Input: rains = [1,2,3,4]
# Output: [-1,-1,-1,-1]
# Explanation: After the first day full lakes are [1]
# After the second day full lakes are [1,2]
# After the third day full lakes are [1,2,3]
# After the fourth day full lakes are [1,2,3,4]
# There's no day to dry any lake and there is no flood in any lake.
#
#
# Example 2:
#
#
# Input: rains = [1,2,0,0,2,1]
# Output: [-1,-1,2,1,-1,-1]
# Explanation: After the first day full lakes are [1]
# After the second day full lakes are [1,2]
# After the third day, we dry lake 2. Full lakes are [1]
# After the fourth day, we dry lake 1. There is no full lakes.
# After the fifth day, full lakes are [2].
# After the sixth day, full lakes are [1,2].
# It is easy that this scenario is flood-free. [-1,-1,1,2,-1,-1] is another
# acceptable scenario.
#
#
# Example 3:
#
#
# Input: rains = [1,2,0,1,2]
# Output: []
# Explanation: After the second day, full lakes are  [1,2]. We have to dry one
# lake in the third day.
# After that, it will rain over lakes [1,2]. It's easy to prove that no matter
# which lake you choose to dry in the 3rd day, the other one will flood.
#
#
# Example 4:
#
#
# Input: rains = [69,0,0,0,69]
# Output: [-1,69,1,1,-1]
# Explanation: Any solution on one of the forms [-1,69,x,y,-1], [-1,x,69,y,-1]
# or [-1,x,y,69,-1] is acceptable where 1 <= x,y <= 10^9
#
#
# Example 5:
#
#
# Input: rains = [10,20,20]
# Output: []
# Explanation: It will rain over lake 20 two consecutive days. There is no
# chance to dry any lake.
#
#
#
# Constraints:
#
#
# 1 <= rains.length <= 10^5
# 0 <= rains[i] <= 10^9
#
#
class DisjointSet():
    def __init__(self, size=10):
        self.data = list(range(size))

    def find(self, i):
        if i == self.data[i]: return i
        else:
            j = self.find(self.data[i])
            self.data[i] = j
            return j

    def merge(self, i):
        self.data[i] = i + 1


class Solution:
    def method2(self, rains: List[int]) -> List[int]:
        # Using disjoint set to sole the problem. check the C++ solution
        # to see how this method work
        n = len(rains)
        cc = {}
        ds = DisjointSet(n)
        i = 0
        while i < n:
            if rains[i] == 0:
                rains[i] = 1
            else:
                lake = rains[i]
                if lake in cc:
                    available_dry_day = ds.find(cc[lake])
                    # otherwise there is no available day to dry a lake before i
                    if available_dry_day < i:
                        rains[available_dry_day] = lake
                        ds.merge(available_dry_day)
                    else:
                        return []
                cc[lake] = i
                ds.merge(i)
                # Modify rain in place and return it on exiting the program to avoid
                # allocating extra space.
                rains[i] = -1
            i+=1
        return rains

    def avoidFlood(self, rains: List[int]) -> List[int]:
        return self.method2(rains)

        n = len(rains)
        cc = defaultdict(int)
        dry = list()
        res = []
        for i, r in enumerate(rains):
            if r == 0:
                dry.append(i)
                res.append(1)
            else:
                if r not in cc:
                    cc[r] = i
                else:
                    last_idx = cc[r]
                    j = bisect_left(dry, last_idx)
                    if j == len(dry): return []
                    day = dry[j]
                    del dry[j]
                    res[day] = r
                    cc[r] = i
                res.append(-1)
        return res


sol = Solution()

rains = [1, 2, 3, 4]
# rains = [1,2,0,0,2,1]
# rains = [1,2,0,1,2]
rains = [69, 0, 0, 0, 69]
# rains = [10,20,20]
rains = [1, 0, 2, 3, 0, 1, 2]
rains = [1, 2, 0, 0, 2, 1]

import heapq
from collections import deque, defaultdict

class Solution:
    def avoidFlood(self, rains: List[int]) -> List[int]:
        ret = [-1 for i in rains]
        
        rain_days = defaultdict(deque)
        
        for i, lake in enumerate(rains):
            if lake != 0:
                rain_days[lake].append(i)
        
        to_drain = []
        
        for i, lake in enumerate(rains):
            if lake == 0:
                if len(to_drain) == 0:
                    ret[i] = 1
                else:
                    day, lake = heapq.heappop(to_drain)
                    if day < i:
                        return []
                    ret[i] = lake
            else:
                if len(rain_days[lake]) > 1:
                    rain_days[lake].popleft()
                    heapq.heappush(to_drain, (rain_days[lake][0], lake))
        
        if len(to_drain) > 0:
            return []
        
        return ret
class Solution:
    def avoidFlood(self, rains: List[int]) -> List[int]:
        lakes = defaultdict(deque)
        
        for day, lake in enumerate(rains):
            lakes[lake].append(day)
        
        N = len(rains)
        res = [1]*N
        heap = [] # days in order of indicies
        
        for day, lake in enumerate(rains):
            if lake:
                lakes[lake].popleft() # current
                if lakes[lake]:
                    next_lake_day = lakes[lake][0]
                    heappush(heap, next_lake_day)
                res[day] = -1
            else:
                if heap:
                    chosen_day = heappop(heap)
                    if chosen_day < day:
                        return []
                    res[day] = rains[chosen_day]
        
        return res if not heap else []
                
                

class Solution:
    def avoidFlood(self, rains: List[int]) -> List[int]:
        n = len(rains)
        res = [-1] * n
        dry = []
        rained = {}
        for i, r in enumerate(rains):
            if r:
                if r not in rained:
                    rained[r] = i
                else:
                    if not dry:
                        return []
                    else:
                        idx = bisect.bisect_left(dry, rained[r])
                        if idx == len(dry):
                            return []
                        res[dry[idx]] = r
                        dry.pop(idx)
                        rained[r] = i
                        
            else:
                dry.append(i)
        for i in dry:
            res[i] = 1
        return res
class Solution:
    def avoidFlood(self, rains: List[int]) -> List[int]:
        a = rains
        n = len(a)
        j = 0
        ans = [-1] * n
        v = {}
        q = []
        for i in range(n):
            c = a[i]
            if c:
                if c in v:
                   # print(i, q)
                    j = bisect.bisect(q, v[c])
                    if j == len(q):
                        return []
                    else:
                        ans[q[j]] = c
                        q.pop(j)
                v[c] = i
            else:
                q.append(i)
        while q:
            ans[q.pop()] = 1
        return ans
class Solution:
    def avoidFlood(self, rains: List[int]) -> List[int]:
        from collections import defaultdict, deque
        closest = []
        locs = defaultdict(deque)
        res = []
        for i, lake in enumerate(rains):
            locs[lake].append(i)
        for i,rain in enumerate(rains):
            # print(closest, rain)
            if closest and closest[0] == i:
                return []
            if rain:
                locs[rain].popleft()
                if locs[rain]:
                    heapq.heappush(closest, locs[rain][0])
                res.append(-1)
            else:
                if closest:
                    dry = heapq.heappop(closest)
                    res.append(rains[dry])
                else:
                    res.append(1)
        return res
# rains: u6bcfu4e2au6570u4ee3u8868u4e00u573au96e8uff0cu7b2ciu573au96e8u4e0bu5728rains[i]u7684u8fd9u4e2alakeu4e0a
# res: -1u8868u793au4e0bu96e8uff0cu4e0du80fdu6392u6c34u3002res[i]u8868u793au8981u6392u5e72u7684u90a3u4e2alake
class Solution:
    def avoidFlood(self, rains: List[int]) -> List[int]:
        # brute force
        lakesFull = {}
        n = len(rains)
        dry = []
        res = []        
        for i in range(n):
            #print ("res:" + str(res))
            #print("dry:" + str(dry))
            #print("lakesFull:" + str(lakesFull))
            l = rains[i]
            if l == 0:
                dry.append(i)
                res.append(-10000000)
                continue
            else:
                if l in lakesFull:
                    if len(dry) > 0:
                        di = -1
                        for dd in range(len(dry)):
                            if dry[dd] > lakesFull[l]:
                                di = dry[dd]
                                dry.pop(dd)
                                break
                        #if i == 10: print("di" + str(di))
                        if di >= 0:
                            res[di] = l
                            del lakesFull[l]
                        else: 
                            return []
                    else:
                        return []
                lakesFull[l] = i
                res.append(-1)
        #print (res)
        #print (dry)
        #print (lakesFull)
        for i in range(n):
            if res[i] == -10000000: res[i] = 1
        return res
class Solution:
    def avoidFlood(self, rains: List[int]) -> List[int]:
        full= set()
        drys = []
        filled = {}
        for i in range(len(rains)):
            if not rains[i]:
                drys.append(i)
            else:
                if rains[i] in full:
                    if not drys: return []
                    if drys[-1] < filled[rains[i]]: return []
                    index = bisect.bisect(drys, filled[rains[i]])
                    rains[drys.pop(index)] = rains[i]
                else:
                    full.add(rains[i])
                filled[rains[i]] = i
                rains[i] = -1
        rains= [1 if i == 0 else i for i in rains]
        return rains
                

class Solution:
    def avoidFlood(self, rains):
        filled, dry_days = {}, []
        ans = [1] * len(rains)

        for day, lake in enumerate(rains):
            if not lake:
                dry_days.append(day)
                continue 

            ans[day] = -1
            if lake in filled:
                if not dry_days: return []
                # use the first dry day after the lake was filled (stored in filled[lake])
                dry_on_day_index = bisect.bisect_left(dry_days, filled[lake])
                if dry_on_day_index >= len(dry_days): return []
                dry_on_day = dry_days.pop(dry_on_day_index)
                ans[dry_on_day] = lake

            filled[lake] = day # we fill it on day

        return ans

class Solution:
    def avoidFlood(self, rains: List[int]) -> List[int]:
        d = {}
        dry = []
        res = []
        for day,rain in enumerate(rains):
            if rain !=0:
                if rain in d:
                    p = d[rain]
                    flag = -1
                    for dry_day in dry:
                        if dry_day > p:
                            flag = dry_day
                            break
                    if flag == -1:
                        return []
                    res[flag] = rain
                    dry.remove(flag)
                    d[rain] = day
                else:
                    d[rain] = day
                res.append(-1)
            else:
                dry.append(day)
                res.append(56)
        return res
from typing import Set, List
from copy import deepcopy
import bisect


class Solution:
    def avoidFlood(self, rains: List[int]) -> List[int]:
        full_lakes = dict()
        dry_days = list()
        ans = list()
        
        for day, lake in enumerate(rains):
            if lake == 0:
                # print(f"Append {day} to dry days")
                dry_days.append(day)
                ans.append(1)
            else:
                if lake not in full_lakes:
                    # print(f"Fill lake {lake} at day {day}")
                    full_lakes[lake] = day
                elif dry_days:
                    filled_at = full_lakes[lake]
                    index = bisect.bisect_right(dry_days, filled_at)
                    # print(f"Dry days are {dry_days}")
                    # print(f"Try to empty lake {lake} filled at day {filled_at}")
                    
                    if index == len(dry_days):
                        # print(f"Can't find a dry day to empty lake {lake}")
                        return list()
                    
                    # print(f"Use day {dry_days[index]} to empty lake {lake}")
                    
                    ans[dry_days[index]] = lake
                    del dry_days[index]
                    full_lakes[lake] = day
                else:
                    return list()
                    
                ans.append(-1)
                
        return ans
        
        
            
        
        

class Solution:
    def avoidFlood(self, rains: List[int]) -> List[int]:
        rains_over_city = {}
        lake_drying_days = []
        
        ind = 0
        for rain in rains:
            if rain > 0:
                rain_index = rains_over_city.get(rain, -1)
                if rain_index != -1:
                    len_lak = len(lake_drying_days)
                    j = 0
                    while j < len_lak and lake_drying_days[j] <= rain_index:
                        j += 1
                    if j >= len_lak:
                        return []
                    rains[lake_drying_days[j]] = rain
                    lake_drying_days.remove(lake_drying_days[j])
                    rains_over_city.pop(rain)
                rains_over_city[rain] = ind
                rains[ind] = -1
            else:
                lake_drying_days.append(ind)
                rains[ind] = 1
            ind += 1
        return rains
class Solution:
    def avoidFlood(self, rains: List[int]) -> List[int]:
        ans = []
        freeday = []
        filled = {}
        
        for day, city in enumerate(rains):
            if city:
                ans.append(-1)
                if city not in filled:
                    filled[city] = day
                
                else:
                    if freeday and freeday[-1] > filled[city]:
                        dry_day = bisect.bisect_left(freeday, filled[city])
                        ans[freeday.pop(dry_day)] = city
                        filled[city] = day
                    else:
                        return []
            
            else:
                ans.append(1)
                freeday.append(day)
            
        return ans
            

class Solution:
    def avoidFlood(self, rains: List[int]) -> List[int]:
        ans = []
        zero = []
        full = {}
        for day, city in enumerate(rains):
            if city:
                ans.append(-1)
                if city in full:
                    if zero and zero[-1] > full[city]:
                        dry = bisect.bisect_left(zero, full[city])
                        ans[zero.pop(dry)] = city
                        full[city] = day
                    else:
                        return []
                else:
                    full[city] = day
            else:
                ans.append(1)
                zero.append(day)
        return ans
class Solution:
    #Version 1: Greedy
    #Use binary search to find the first dry day after the city got wet.
    #TC: O(n^2), SC: O(n)
    '''
    def avoidFlood(self, rains: List[int]) -> List[int]:
        from bisect import bisect_left
        wet = {}
        ans = [1]*len(rains)
        dry = []
        for k in range(len(rains)):
            if not rains[k]:
                dry.append(k)
            else:
                ans[k] = -1
        for k in range(len(rains)):
            if rains[k] > 0:
                if rains[k] not in wet:
                    wet[rains[k]] = k
                else:
                    index = bisect_left(dry, wet[rains[k]])
                    if index == len(dry) or dry[index] > k:
                        return []
                    wet[rains[k]] = k
                    ans[dry[index]] = rains[k]
                    dry.pop(index)
        return ans
    '''
    
    #Version 2: Improved version 1
    #Use SortedList to accelerate remove part
    #TC: O(nlogn), SC: O(n)
    '''
    def avoidFlood(self, rains: List[int]) -> List[int]:
        from sortedcontainers import SortedList
        wet = {}
        ans = [1]*len(rains)
        dry = SortedList()
        for k in range(len(rains)):
            if not rains[k]:
                dry.add(k)
            else:
                ans[k] = -1
        for k in range(len(rains)):
            if rains[k] > 0:
                if rains[k] not in wet:
                    wet[rains[k]] = k
                else:
                    index = dry.bisect_left(wet[rains[k]])
                    if index == len(dry) or dry[index] > k:
                        return []
                    wet[rains[k]] = k
                    ans[dry[index]] = rains[k]
                    dry.pop(index)
        return ans
    '''
    
    #Version 3: Greedy
    #Store the next position of wet cities in the heap and pop out by urgency
    #TC: O(nlogn), SC: O(n)
    def avoidFlood(self, rains: List[int]) -> List[int]:
        from collections import deque
        import heapq
        city = {}
        ans = [1]*len(rains)
        for k in range(len(rains)):
            if rains[k]:
                if rains[k] not in city:
                    city[rains[k]] = deque()
                city[rains[k]].append(k)
                ans[k] = -1
        option = []
        wet = {}
        for k in range(len(rains)):
            if rains[k]:
                if rains[k] in wet:
                    return []
                else:
                    wet[rains[k]] = k
                    city[rains[k]].popleft()
                    if city[rains[k]]:
                        heapq.heappush(option, (city[rains[k]][0], rains[k]))
            else:
                if option:
                    _, c = heapq.heappop(option)
                    ans[k] = c
                    wet.pop(c)
                else:
                    ans[k] = 1
        return ans
                    
        

class Solution:
    def avoidFlood(self, rains: List[int]) -> List[int]:
        from bisect import bisect_left
        wet = {}
        ans = [1]*len(rains)
        dry = []
        for k in range(len(rains)):
            if not rains[k]:
                dry.append(k)
            else:
                ans[k] = -1
        for k in range(len(rains)):
            if rains[k] > 0:
                if rains[k] not in wet:
                    wet[rains[k]] = k
                else:
                    index = bisect_left(dry, wet[rains[k]])
                    if index == len(dry) or dry[index] > k:
                        return []
                    wet[rains[k]] = k
                    ans[dry[index]] = rains[k]
                    dry.pop(index)
        return ans

from sortedcontainers import SortedList
class Solution:
    def avoidFlood(self, rains: List[int]) -> List[int]:
        res = [1 for i in range(len(rains))]
        ls, zs = {}, SortedList()
        for i, l in enumerate(rains):
            if l in ls:
                if len(zs) == 0:
                    return []
                else:
                    p = zs.bisect(ls[l])
                    if p == len(zs):
                        return []
                    res[zs[p]] = l
                    zs.pop(p)
                    res[i] = -1
                    ls[l] = i
            elif l != 0:
                ls[l] = i
                res[i] = -1
            else:
                zs.add(i)
        return res
            
        
            
        

class Solution:
    def avoidFlood(self, rains: List[int]) -> List[int]:
        
        def closest(i, arr):
            for j in arr:
                if j > i:
                    return j
            return -1
        
        ans = [1]*len(rains)
        visited = dict()
        zeros = deque()
        x = 0
        while x<len(rains) and rains[x] == 0:
            x += 1
        
        for i in range(x, len(rains)):
            if rains[i] in visited:
                if not zeros:
                    return []
                else:
                    r = visited[rains[i]]
                    c = closest(r, zeros)
                    if c == -1:
                        return []
                    ans[c] = rains[i]
                    zeros.remove(c)
                    ans[i] = -1
                    visited[rains[i]] = i
            elif rains[i]:
                ans[i] = -1
                visited[rains[i]] = i
            else:
                zeros.append(i)
        return ans
        

class Solution:
    #Version 1: Greedy
    #Use binary search to find the first dry day after the city got wet.
    #TC: O(n^2), SC: O(n)
    '''
    def avoidFlood(self, rains: List[int]) -> List[int]:
        from bisect import bisect_left
        wet = {}
        ans = [1]*len(rains)
        dry = []
        for k in range(len(rains)):
            if not rains[k]:
                dry.append(k)
            else:
                ans[k] = -1
        for k in range(len(rains)):
            if rains[k] > 0:
                if rains[k] not in wet:
                    wet[rains[k]] = k
                else:
                    index = bisect_left(dry, wet[rains[k]])
                    if index == len(dry) or dry[index] > k:
                        return []
                    wet[rains[k]] = k
                    ans[dry[index]] = rains[k]
                    dry.pop(index)
        return ans
    '''
    
    def avoidFlood(self, rains: List[int]) -> List[int]:
        from sortedcontainers import SortedList
        wet = {}
        ans = [1]*len(rains)
        dry = SortedList()
        for k in range(len(rains)):
            if not rains[k]:
                dry.add(k)
            else:
                ans[k] = -1
        for k in range(len(rains)):
            if rains[k] > 0:
                if rains[k] not in wet:
                    wet[rains[k]] = k
                else:
                    index = dry.bisect_left(wet[rains[k]])
                    if index == len(dry) or dry[index] > k:
                        return []
                    wet[rains[k]] = k
                    ans[dry[index]] = rains[k]
                    dry.pop(index)
        return ans

from collections import deque
class Solution:
    def avoidFlood(self, rains: List[int]) -> List[int]:
        ret = [-1 for i in range(0, len(rains))]
        
        filled_lakes = dict()
        dry_days = list()
        
        i = 0
        while i < len(rains):
            rain = rains[i]
            
            if rain == 0:
                dry_days.append(i)
            if rain > 0 and rain in filled_lakes:
                prev_pos = filled_lakes[rain]
                if len(dry_days) == 0:
                    return []
                
                j = 0
                while j < len(dry_days) and prev_pos > dry_days[j]:
                    j += 1
                
                if j == len(dry_days):
                    return []
                
                k = dry_days.pop(j)
                
                ret[k] = rain
            
            if rain != 0:
                filled_lakes[rain] = i
            
            i += 1
        
        while len(dry_days) > 0:
            j = dry_days.pop()
            ret[j] = 1
        return ret

from sortedcontainers import SortedList
class Solution:
    def avoidFlood(self, rains: List[int]) -> List[int]:
        lakes = defaultdict(deque)
        
        for day, lake in enumerate(rains):
            lakes[lake].append(day)
        
        N = len(rains)
        need = SortedList()
        res = [1]*N
        
        for curr_day, lake in enumerate(rains):
            if lake:
                lakes[lake].popleft()
                
                if len(lakes[lake]) > 0:
                    next_lake_index = lakes[lake][0]
                    need.add(next_lake_index)
                
                res[curr_day] = -1
            elif need:
                chosen_lake_index = need.pop(0)
                if chosen_lake_index < curr_day:
                    return []
                
                res[curr_day] = rains[chosen_lake_index]
        
        return res if not need else []
                
                
                
        
                
                

class Solution:
    def avoidFlood(self, rains: List[int]) -> List[int]:
        drying = []
        out = []
        lakes = {}
        v = 1
        for c,i in enumerate(rains):
            out.append(-1)
            if i <= 0:
                drying.append(c)
            elif i in lakes and drying:
                found = -1
                for index, val in enumerate(drying):
                    if val > lakes[i]:
                        found = index
                        break
                if found > -1:
                    out[drying.pop(found)] = i
                    lakes[i] = c
                else:
                    return []
            elif i in lakes:
                return []
            else:
                lakes[i] = c
                v = i
        for j in drying:
            out[j] = v
        return out
                

class Solution:
    def avoidFlood(self, rains: List[int]) -> List[int]:
        return_arr = [-1 for i in range(len(rains))]
        rains_dict = {}
        zeros_indices = []
        for i in range(len(rains)):
            if(rains[i]) == 0:
                return_arr[i] = 1
                zeros_indices.append(i)
            elif rains[i] not in rains_dict:
                rains_dict[rains[i]] = i
            else:
                if len(zeros_indices) == 0:
                    return []
                
                #find index of dry day to use
                index = 0
                while(zeros_indices[index] < rains_dict[rains[i]]):
                    index += 1
                    print(index)
                    if(index == len(zeros_indices)):
                        return []
                return_arr[zeros_indices[index]] = rains[i]
                rains_dict[rains[i]] = i
                del zeros_indices[index]
        return return_arr
from sortedcontainers import SortedList
class Solution:
    def avoidFlood(self, rains: List[int]) -> List[int]:
        lakes = defaultdict(deque)
        
        for day, lake in enumerate(rains):
            lakes[lake].append(day)
        
        N = len(rains)
        need = SortedList()
        res = [1]*N
        
        for curr_day, lake in enumerate(rains):
            if lake:
                lakes[lake].popleft()
                
                if len(lakes[lake]) > 0:
                    next_lake_day = lakes[lake][0]
                    need.add(next_lake_day)
                
                res[curr_day] = -1
            elif need:
                chosen_lake_day = need.pop(0)
                if chosen_lake_day < curr_day:
                    return []
                
                res[curr_day] = rains[chosen_lake_day]
        
        return res if not need else []
                
                
                
        
                
                

class Solution:
    def avoidFlood(self, rains: List[int]) -> List[int]:
        ans = [-1]*len(rains)
        spares = []
        full = {}
        for i in range(len(rains)):
            if rains[i] > 0:
                if rains[i] in full:
                    for j in range(len(spares)):
                        if spares[j] > full[rains[i]]:
                            ans[spares.pop(j)] = rains[i]
                            full[rains[i]] = i
                            break
                    else:
                        return []
                else:
                    full[rains[i]] = i
            else:
                spares.append(i)
        for i in spares:
            ans[i] = 1
        return ans
class Solution:
    def avoidFlood(self, rains: List[int]) -> List[int]:
        '''
        [1,2,0,0,2,1]
        day0 rains on lake1
        day1 rains on lake2
        day2 sunny
        day3 sunny
        day4 rains on lake1
        day5 rains on lake2
        
        '''
        sunny_day_idx=[]
        res=[-1]*len(rains)
        for day,lake in enumerate(rains):
            if lake==0:
                sunny_day_idx.append(day)
                res[day]=1
            
                
        last_day_rains_over_lake={}
        '''
        [1,0,2,0,2,1]
        for lake2 we need to find sunny_day_idx between 2 - 4
        sunny_day_idx=[1,4] is a increasing array
        use binary search to find minimum value between prev_day and curr_day
        
        '''
        def binary_search(sunny_day_idx,prev):
            low,high=0,len(sunny_day_idx)
            while low<high:
                mid=(low+high)//2
                if sunny_day_idx[mid]>prev:
                    high=mid
                else:
                    low=mid+1
            return low if low<len(sunny_day_idx) and sunny_day_idx[low]>prev else None
            
        print((binary_search([2,3],0)))    
                
                
                
                
        for day,lake in enumerate(rains):
            if lake!=0 and lake not in last_day_rains_over_lake:
                last_day_rains_over_lake[lake]=day
            elif lake!=0 and lake in last_day_rains_over_lake:
                if not sunny_day_idx:
                    return []
                idx=binary_search(sunny_day_idx,last_day_rains_over_lake[lake])
                #print(idx)
                if idx==None:
                    return []
                if sunny_day_idx[idx]>day:
                    return []
                #print(sunny_day_idx[0],day)
                res[sunny_day_idx[idx]]=lake
                last_day_rains_over_lake[lake]=day
                sunny_day_idx.pop(idx)
        return res
                
            

class Solution:
    def avoidFlood(self, rains: List[int]) -> List[int]:
        # loop  day , lk ,   if 0 ,  clean the immediate next overflow,  if not 0 , overflow or fill empty  depends on if in next schedule  
        # https://leetcode.com/problems/avoid-flood-in-the-city/discuss/698328/python-faster-than-10000-of-python-online-submissions-for-avoid-flood-in-the-city
        # https://leetcode.jp/leetcode-avoid-flood-in-the-city-%e8%a7%a3%e9%a2%98%e6%80%9d%e8%b7%af%e5%88%86%e6%9e%90/
        
        days_next = [] 
        n = len(rains) 
        dct_lk2day = defaultdict(int) # lake:day ,  last time happened day 
        for day_rev, e in enumerate(rains[::-1]): 
            if e in dct_lk2day:
                days_next.append(dct_lk2day[e]) # existing day 
            else:
                days_next.append(n)     # last day + 1  , next will be outside 
            dct_lk2day[e] = n -1 - day_rev  # update current day 
        days_next = days_next[::-1]     # reverse 
        
        # loop again,  put next day to minheap, 0 , clean, non0 fill or overflow 
        minHp = [] 
        rst = [] 
        for day, lk in enumerate(rains):
            if not lk:  # 0 
                if minHp:
                    d_next = heappop(minHp)  # 1st ele 
                    rst.append(rains[d_next]) # that lake 
                else:   
                    rst.append(1)  # 1st lake 
            else:
                if minHp and day == minHp[0]: # not cleaned up , overflow  , eval and left first 
                    return [] 
                else:       # cleaned , will fill 
                    if days_next[day]<n:    # if needs to be cleaned in future 
                        heappush(minHp, days_next[day]) # next coming time 
                    rst.append(-1) 
        
        return rst 
                    
            
        
                    
                
                
            
                    
                    
                

from sortedcontainers import SortedList as sl
class Solution:
    def avoidFlood(self, rains: List[int]) -> List[int]:
        ans = [-1]*(len(rains))
        has_rained = dict()
        free = sl()
        for i in range(len(rains)):
            # print(free)
            if rains[i]==0:
                free.add(i)
            else:
                if rains[i] not in has_rained:
                    has_rained[rains[i]]=i
                else:
					# no free days are available
                    if len(free)==0:
                        return []
					#finding the index of the free day that came after the first occurance of
					# rains[i]
                    idx = free.bisect_left(has_rained[rains[i]])
                    # print(free,idx,i,has_rained)
                    if idx<len(free):
                        ans[free[idx]]=rains[i]
						# updating the index of rains[i] for future it's future occurances
                        has_rained[rains[i]] = i
                        free.remove(free[idx])
                    else:
						#if no such index exists then return
                        return []
        if len(free):
            while free:
				# choosing some day to dry on the remaining days
                ans[free.pop()]=1
        return ans

class Solution:
    def avoidFlood(self, rains: List[int]) -> List[int]:
        '''
        [1,2,0,0,2,1]
        day0 rains on lake1
        day1 rains on lake2
        day2 sunny
        day3 sunny
        day4 rains on lake1
        day5 rains on lake2
        
        '''
        sunny_day_idx=[]
        res=[-1]*len(rains)
        # for day,lake in enumerate(rains):
        #     if lake==0:
        #         sunny_day_idx.append(day)
        #         res[day]=1
            
                
        last_day_rains_over_lake={}
        '''
        [1,0,2,0,2,1]
        for lake2 we need to find sunny_day_idx between 2 - 4
        sunny_day_idx=[1,4] is a increasing array
        use binary search to find minimum value between prev_day and curr_day
        
        '''
        def binary_search(sunny_day_idx,prev):
            low,high=0,len(sunny_day_idx)
            while low<high:
                mid=(low+high)//2
                if sunny_day_idx[mid]>prev:
                    high=mid
                else:
                    low=mid+1
            return low if low<len(sunny_day_idx) and sunny_day_idx[low]>prev else None
            
        #print(binary_search([2,3],0))    
                
                
                
                
        for day,lake in enumerate(rains):
            if lake==0:
                sunny_day_idx.append(day)
                res[day]=1
            if lake!=0 and lake not in last_day_rains_over_lake:
                last_day_rains_over_lake[lake]=day
            elif lake!=0 and lake in last_day_rains_over_lake:
                
                if not sunny_day_idx:
                    return []
                idx=binary_search(sunny_day_idx,last_day_rains_over_lake[lake])
                #print(idx)
                if idx==None or sunny_day_idx[idx]>day:
                    return []
                #print(sunny_day_idx[0],day)
                res[sunny_day_idx[idx]]=lake
                last_day_rains_over_lake[lake]=day
                sunny_day_idx.pop(idx)
        return res
                
            

class Solution:
    def avoidFlood(self, rains: List[int]) -> List[int]:
        q=[] # list for zeros positions
        ans=[]
        hashmap={}
        for i in range(len(rains)):
            if rains[i] == 0:
                q.append(i)
                ans.append(1)  # as per example 4
            else:
                if rains[i] in hashmap:    
                    if len(q) == 0:
                        return []
                    else:
                        index = hashmap[rains[i]]
                        # find a zero position just greater than previous occurrence of rains[i]
                        pos=bisect.bisect_right(q, index) 
                        if pos<len(q): # no zero exists in between occurrence
                            ans[q[pos]]=rains[i]
                            q.pop(pos)
                        else:
                            return []
                hashmap[rains[i]]=i
                ans.append(-1)  
            
        return ans
from bisect import bisect_left

class Solution:
    def avoidFlood(self, rains: List[int]) -> List[int]:
        ans = [-1] * len(rains)
        last_appear = {}
        dry_days = []
        
        for idx, lake in enumerate(rains):
            if lake == 0:
                dry_days.append(idx)
                continue
            if lake in last_appear:
                # find 0
                first_0 = bisect_left(dry_days, last_appear[lake])
                if first_0 == len(dry_days): # not found
                    return []
                ans[dry_days[first_0]] = lake
                dry_days.pop(first_0)
                last_appear.pop(lake)
            last_appear[lake] = idx
            
        for day in dry_days: ans[day] = 1 
        
        return ans

def nex(arr, target): 
	start = 0;
	end = len(arr) - 1

	ans = -1; 
	while (start <= end): 
		mid = (start + end) // 2; 

		# Move to right side if target is 
		# greater. 
		if (arr[mid] <= target): 
			start = mid + 1; 

		# Move left side. 
		else: 
			ans = mid; 
			end = mid - 1; 

	return ans;
            
def find(ind,dind,flag):
    n=len(dind)
    if n==0:
        flag[0]=1
        return -1
    ans = nex(dind,ind)
    if ans==-1:
        flag[0]=1
        return ans
    else:
        return dind[ans]
class Solution:
    def avoidFlood(self, rains: List[int]) -> List[int]:
        lind={}
        n = len(rains)
        ans = [1 for i in range(n)]
        dind = []
        for i in range(n):
            if rains[i] >0:
                ans[i]=-1
                if rains[i] not in lind:
                    lind[rains[i]]=i
                    
                else:
                    flag=[0]
                    ind = lind[rains[i]]
                    dry = find(ind,dind,flag)
                    if flag[0]==1:
                        return []
                    else:
                        ans[dry] = rains[i]
                        lind[rains[i]]=i
                        dind.remove(dry)
            else:
                dind.append(i)
                
        return ans
            
from sortedcontainers import SortedList
class Solution:
    def avoidFlood(self, rains: List[int]) -> List[int]:
        ans = [-1]*len(rains)
        zeros = SortedList()
        todry = {}
        for i, r in enumerate(rains):
            if r == 0:
                zeros.add(i)
                continue
            if r in todry:
                di = zeros.bisect_left(todry[r])
                if di >= len(zeros):
                    return []
                ans[zeros[di]] = r
                zeros.pop(di)
            todry[r] = i
        for i in range(len(rains)):
            if rains[i] == 0 and ans[i] == -1:
                ans[i] = 1
        return ans

class Solution:
    def avoidFlood(self, rains: List[int]) -> List[int]:
        n = len(rains) 
        M = max(rains)
        ans = [-1]*n
        
        l = set()
                
        r = collections.defaultdict(list)
        for j in range(n):
            if rains[j] > 0: 
                r[rains[j]] += [j] 
                                
        i = 0 
        tbd = []
        
        
        while(i<n):
            if rains[i]>0:
                if rains[i] in l:
                    return []
                else:
                    l.add(rains[i])
                    r[rains[i]].pop(0)
                    
                    if len(r[rains[i]]) > 0: 
                        heapq.heappush(tbd, r[rains[i]][0])
            
            elif rains[i] == 0:
                if len(tbd) > 0:
                    get = heapq.heappop(tbd)
                    ans[i] = rains[get] 
                    l.remove(rains[get])
                                    
                else:
                    ans[i] = M + 1            
            i += 1
                    
        
        # print(ans) 
        
        return ans
        
                
            

from sortedcontainers import SortedList


class Solution:
    def avoidFlood(self, rains: List[int]) -> List[int]:
        ans = [-1]*len(rains)
        zeros, todry = SortedList(), {}
        for i, r in enumerate(rains):
            if r == 0:
                zeros.add(i)
            elif r not in todry:
                todry[r] = i
            else:
                di = zeros.bisect_left(todry[r])
                if di == len(zeros):
                    return []
                ans[zeros[di]] = r
                zeros.pop(di)
                todry[r] = i
        for i in range(len(rains)):
            if rains[i] == 0 and ans[i] == -1:
                ans[i] = 1
        return ans

from bisect import bisect_left

class Solution:
    def avoidFlood(self, rains: List[int]) -> List[int]:
        flooded = {}
        dry_days = []
        out = [-1 for _ in range(len(rains))]
        
        for i in range(len(rains)):
            day = rains[i]
            if day > 0:
                if day in flooded:
                    if dry_days:
                        # found = False
                        # for d in range(len(dry_days)):
                        #     dry_day = dry_days[d]
                        #     if dry_day > flooded[day]:
                        #         out[dry_day] = day
                        #         dry_days.pop(d)
                        #         found = True
                        #         break
                        # if not found:
                        #     return []
                        dry = bisect_left(dry_days, flooded[day])
                        if dry == len(dry_days):
                            return []
                        else:
                            dry_day = dry_days.pop(dry)
                            out[dry_day] = day
                    else:
                        return []
                flooded[day] = i
            else:
                dry_days.append(i)
        for dry_day in dry_days:
            out[dry_day] = 1
        return out
from collections import deque


class Solution:
    def avoidFlood(self, rains: List[int]) -> List[int]:
        zero_indices = deque()
        last_flood = {}
        result = [-1] * len(rains)
        
        def find_zero_for_lake(idx, r):
            if not zero_indices:
                return None

            for zero_index in zero_indices:
                if last_flood[r] < zero_index < idx:
                    return zero_index
                
            return None
        
        for idx, r in enumerate(rains):
            if r > 0:
                if r in last_flood:
                    found_zero = find_zero_for_lake(idx, r)            
                    if found_zero is None:
                        return []
                    else:
                        result[found_zero] = r
                        zero_indices.remove(found_zero)  
                        
                last_flood[r] = idx
            else:
                zero_indices.append(idx)
                
        while zero_indices:
            zero_index = zero_indices.pop()
            result[zero_index] = 1
                
        return result

class Solution:
    def avoidFlood(self, rains: List[int]) -> List[int]:
        # To be improved with binary search for faster determination of idx for dry
        full, idxDry, res = dict(), [], [-1] * len(rains)
        for i, x in enumerate(rains):
            if not x: idxDry.append(i); continue
            if x in full:
                if not idxDry or full[x] > idxDry[-1]: return []
                # Improve here
                for idx in idxDry:
                    if idx > full[x]:
                        res[idx] = x
                        idxDry.remove(idx)
                        del full[x]
                        break
            full[x] = i
        for i in idxDry:
            res[i] = 1
        return res
from bisect import bisect_left 
class Solution:
    def avoidFlood(self, rains: List[int]) -> List[int]:
        
        filled = {}
        ans = [-1] * len(rains)
        rem = []
        
        def get(r): # idx for first zero b4 right
            low, high = 0, len(rem) - 1
            while low <= high:
                mid = low + (high-low)//2
                if rem[mid] > r:
                    low = mid + 1
                else:
                    high = mid - 1
            # print(low)
            return rem[low] if 0 <= low < len(rem) else -1
        
        for i in range(len(rains)-1, -1, -1):
            curr = rains[i]
            # if i == 8:
            #     print(rem)
            if curr == 0:
                rem.append(i)
                continue
            elif curr in filled:
                if not rem:
                    return []
                
                idx = filled.pop(curr)
                zero = get(idx)                          
                
                if not i < zero < idx:
                    # print(rem)
                    # print(curr, i, zero, idx)
                    return []
                ans[zero] = curr                
                rem.remove(zero)     
            
            filled[curr] = i
        while rem:
            ans[rem.pop()] = 1
        return ans
            

class Solution:
    def avoidFlood(self, rains: List[int]) -> List[int]:
        n = len(rains)
        idx = defaultdict(list)
        
        for i in range(n):
            if rains[i]>0:
                idx[rains[i]].append(i)
        
        nex = defaultdict(lambda: -1)
        
        for k in idx.keys():
            for i in range(len(idx[k])-1):
                nex[idx[k][i]] = idx[k][i+1]
        
        cnt = defaultdict(int)
        pq = []
        ans = []
        
        for i in range(n):
            if rains[i]>0:
                if cnt[rains[i]]==1:
                    return []
                
                cnt[rains[i]] = 1
                
                if nex[i]!=-1:
                    heappush(pq, (nex[i], rains[i]))
                    
                ans.append(-1)
            else:
                if len(pq)==0:
                    ans.append(1)
                else:
                    _, lake = heappop(pq)
                    cnt[lake] -= 1
                    ans.append(lake)
        
        return ans
class Solution:
    
    def backtrack(self, rains, full, position, seq):
        if position >= len(rains):
            return True
        if rains[position] > 0:
            if rains[position] in full:
                return False
            seq.append(-1)
            full.add(rains[position])
            if self.backtrack(rains, full, position + 1, seq):
                return True
            full.remove(rains[position])
            seq.pop()
        elif rains[position] == 0:
            # must choose one lake that is full to dry
            for lake in full:
                seq.append(lake)
                full.remove(lake)
                if self.backtrack(rains, full, position + 1, seq):
                    return True
                full.add(lake)
                seq.pop()
            if len(full) < 1:
                seq.append(1) # random lake
                if self.backtrack(rains, full, position + 1, seq):
                    return True
                seq.pop()
    
    def avoidFloodBacktrack(self, rains: List[int]) -> List[int]:
        seq = []
        full = set()
        if not self.backtrack(rains, full, 0, seq):
            return []
        return seq
    
    def avoidFlood(self, rains: List[int]) -> List[int]:
        spares = []
        recent = dict()
        full = set()
        ans = []
        for i in range(len(rains)):
            if rains[i] > 0:
                ans.append(-1)
                if rains[i] in full:
                    # we need to have dried this lake for sure
                    if len(spares) < 1:
                        return []
                    valid = False
                    for d in range(len(spares)):
                        dry = spares[d]
                        if dry > recent[rains[i]]:
                            ans[dry] = rains[i]
                            full.remove(rains[i])
                            del spares[d]
                            valid = True
                            break
                    if not valid:
                        return []
                elif rains[i] in recent and recent[rains[i]] == i - 1:
                    # no chance to dry this one
                    return []
                else:
                    full.add(rains[i])
                recent[rains[i]] = i
            else:
                # we can dry one lake
                # greedy chooses some random lake
                # that will be replaced if needed
                ans.append(1)
                spares.append(i)
        return ans
    
if False:
    assert Solution().avoidFlood([69,0,0,0,69]) == [-1, 69, 1, 1, -1]
    assert Solution().avoidFlood([1,2,0,0,2,1]) == [-1,-1,2,1,-1,-1]
    assert Solution().avoidFlood([1,2,0,1,2]) == []
    assert Solution().avoidFlood([10,20,20]) == []
from collections import defaultdict
from sortedcontainers import SortedSet
class Solution:
    def avoidFlood(self, rains: List[int]) -> List[int]:
        def getCeil(a, v):
            if not a:
                return None
            b = a.bisect_right(v)

            if b == 0:
                if a[b]>=v:
                    return a[b]
                return None
            if b==len(a):
                return None
            return a[b]
        
        res = [0]*len(rains)
        zeros = SortedSet()
        m = {}
        for  i, val in enumerate(rains):
            # print(i, val, m, zeros, res)
            if val == 0:
                zeros.add(i)
            else:
                if val in m:
                    n = getCeil(zeros, m[val])
                    print(n)
                    if not n: return []
                    res[n] = val
                    zeros.remove(n)
                res[i] = -1
                m[val] = i
        for i in zeros:
            res[i] = 1
        return res
            
                        

class Solution:
    def avoidFlood(self, rains: List[int]) -> List[int]:
        
        queue = []
        lakes = dict()
        
        days = []
        for i in range(len(rains)):
            if rains[i] == 0:
                queue.append(i)
            elif rains[i] in lakes:
                if not queue:
                    return []
                
                for queue_index in range(len(queue)):
                    if queue[queue_index] < lakes[rains[i]]:
                        if queue_index == len(queue) - 1:
                            return []
                        continue
                    else:
                        days[queue.pop(queue_index)] = rains[i]
                        break
                lakes[rains[i]] = i
            else:
                lakes[rains[i]] = i
            days.append(-1 if rains[i] != 0 else 999)
            
        return days
                
        # 1,0 ,1, 0,1
        # [1,2,0,0,2,1]
        # (1: 2, 2: 2, 0:2)
        
        # [1,2,0,1,2]
        # (1: 2, 2: 2, 0:1) (4 - 2) > 1
        
        # O(n) space
        # keep queue of indexes which represents days where there is no rain
        # keep a set of values of n that have been rained on
        # for each entry in the array (O(n))
        #       If value is zero:
        #           add current index to end of queue 
        #       elif: Whenever we encounter value that is in the set and nonzero- 
        #           if queue is empty, return empty array   
        #           pop from the queue and emplace encountered value in output array at popped index (O(1))
        #       else:
        #           add encountered value to set
        #       add -1 to output array if value is not 0, else 999
        #
    
    # rains = [1, 0 ,1, 0]
    # rains = [1, 1, 0]
    # rains = [1, 0, 1, 0, 1]
    # rains = [0, 1, 0, 1, 0, 1]
    # rains = [0 , 0, 0... , 1, 0, 1]
    # rains = [1, 0, 2, 0 , 2, 1]
    
    

class Solution:
    def avoidFlood(self, rains: List[int]) -> List[int]:
        
        res=[]
        for rain in rains:
            if rain>0:
                res.append(-1)
            else:
                res.append(1)
        
        #full_lakes=[]
        full_lakes=defaultdict(list)
        
        #index of 0
        sunny_days=[]
        
        
        #remove leading and trailing zeros
        start=0
        end=len(rains)-1
        while rains[start]==0:
            start+=1
        while rains[end]==0:
            end-=1

        def binarySearch(sunny_days,prev_rain,curr):
            #print('binary seaarch: ',sunny_days,prev_rain,curr)
            if not sunny_days:
                return -1
            low=0
            high=len(sunny_days)
            while low<high:
                mid=(low+high)//2

                if sunny_days[mid]<=prev_rain:
                    low=mid+1
                else:
                    high=mid
            #print(low)
            if low>=len(sunny_days) or sunny_days[low]<=prev_rain or sunny_days[low]>=curr:
                return -1
            else:
                return low
                
        # a=[8]
        # b=9
        # c=10
        # print(binarySearch(a,b,c))

            
            
        
        for i in range(start,end+1):
            if rains[i]!=0:
                if rains[i] not in full_lakes or not full_lakes[rains[i]]:
                    full_lakes[rains[i]].append(i)
                else:
                    prev_rain=full_lakes[rains[i]][-1]

                    
                    #print(prev_rain,i,sunny_days)
                    #print('#####')
                    if not sunny_days:
                        return []
                    
                    idx=binarySearch(sunny_days,prev_rain,i)
                    #print(idx)
                    #print(sunny_days,prev_rain,i)
                    # for j in range(len(sunny_days)):
                    #     if prev_rain<sunny_days[j]<i:
                    #         idx=sunny_days[j]
                    #         break
                    if idx==-1:
                        return []
                    else:
                        
                        res[sunny_days[idx]]=rains[i]
                        full_lakes[rains[i]].pop()
                        full_lakes[rains[i]].append(i)
                        sunny_days.pop(idx)
                        
            else:
                sunny_days.append(i)
        return res
        
                    
                        
        

class Solution:
    
    def find_min_greater(self,a,x):
        start = 0
        end = len(a)-1
        while(start<end):
            mid = start+end
            mid = mid//2
            if(a[mid]<x and a[mid+1]>x):
                return mid+1
            elif(a[mid]<x):
                start = mid
            else:
                end = mid
        return start
    
    def avoidFlood(self, rains: List[int]) -> List[int]:
        last_rained = {}
        no_rain = []
        ans = []
        for i in range(len(rains)):
            if(rains[i]==0):
                no_rain.append(i)
                ans.append(1)
            else:
                lake = rains[i]
                ans.append(-1)
                if(lake not in last_rained.keys()):
                    last_rained[lake] = i
                else:
                    lr = last_rained[lake]
                    zl = len(no_rain)
                    if(len(no_rain)==0 or no_rain[zl-1]<lr):
                        return []
                    else:
                        empty = self.find_min_greater(no_rain,lr)
                        ans[no_rain[empty]] = lake
                        last_rained[lake]=i
                        no_rain.pop(empty)
        return ans
from collections import deque

class Solution:
    def avoidFlood(self, rains: List[int]) -> List[int]:
        dry_sequence = [1 for _ in range(len(rains))]
        full_lakes = {}
        dry_days = deque()
        for i,lake in enumerate(rains):
            if lake:
                dry_sequence[i] = -1
                if lake in full_lakes:
                    if dry_days:
                        index = 0
                        while index < len(dry_days) and dry_days[index] <= full_lakes[lake]:
                            index += 1
                        if index < len(dry_days):
                            dry_sequence[dry_days[index]] = lake
                            del dry_days[index]
                        else:
                            return []
                    else:
                        return []
                full_lakes[lake] = i
            else:
                dry_days.append(i)
        return dry_sequence
                                        

from collections import defaultdict
class Solution:
    def avoidFlood(self, rains: List[int]) -> List[int]:
        dry_days = []
        filled_lakes = {}
        ans = [1]*len(rains)
        for i in range(len(rains)):
            if rains[i] != 0:
                ans[i] = -1
            if rains[i] in filled_lakes:
                if not dry_days:
                    return []
                else:
                    day_to_dry_lake = bisect.bisect_left(dry_days,filled_lakes[rains[i]])
                    if day_to_dry_lake >= len(dry_days):
                        return []
                    ans[dry_days[day_to_dry_lake]] = rains[i]
                    dry_days.pop(day_to_dry_lake)
                    filled_lakes.pop(rains[i],None)
            if rains[i] == 0:
                dry_days.append(i)        
            else:
                filled_lakes[rains[i]] = i
        return ans
class Solution:
    def avoidFlood(self, rains: List[int]) -> List[int]:
        
        availables = [] # available days, will be sorted ascendingly 
        n = len(rains)
        ans = [-1]*n
        prev_rain = dict() # prev_rain[lake] tells us when it last rained on lake, assuming lake is full.
        
        for day in range(n):
            if rains[day] == 0:
                availables.append(day)
            else:
                lake = rains[day]
                if lake not in prev_rain:
                    prev_rain[lake] = day
                else:
                    # we must find the earliest available day to empty this lake
                    # after prev_rain[lake] then remove it from availables and 
                    # remove lake from prev_rain, and indicate this in the answer
                    if len(availables) == 0 or availables[-1] < prev_rain[lake]:
                        return []
                    low = 0
                    high = len(availables)-1
                    while low < high:
                        med = (low+high)//2
                        if availables[med] < prev_rain[lake]:
                            low = med+1
                        else:
                            high = med
                    chosen_day = availables[low]
                    availables.remove(chosen_day)
                    prev_rain[lake] = day
                    ans[chosen_day] = lake
        
        while availables:
            ans[availables[-1]] = 20
            availables.pop()
        
        return ans              
from collections import defaultdict
class Solution:
    def avoidFlood(self, rains: List[int]) -> List[int]:
        lake=defaultdict(list)
        ans=[1]*len(rains)
        for i in range(len(rains)):
            rain=rains[i]
            if rain==0:
                lake[0].append(i)
            else:
                ans[i]=-1
                if len(lake[0])==0 and len(lake[rain])!=0:
                    return []
                elif len(lake[0])!=0 and len(lake[rain])!=0:
                    for k in lake[0]:
                        if k > lake[rain][0]:
                            ans[k]=rain
                            lake[0].remove(k)
                            lake[rain]=[i]
                            break
                    else:
                        return []
                elif len(lake[rain])==0:
                    lake[rain].append(i)
        return ans
#https://blog.csdn.net/pfdvnah/article/details/106897444

class Solution:
    def avoidFlood(self, rains: List[int]) -> List[int]:
        sol = [53456 for _ in range(len(rains))]
        
        pos = collections.defaultdict(list)
        for idx, n in enumerate(rains):
            if n>0:
                pos[n].append(idx)
        for key in pos:
            pos[key].reverse()
            
        q = []
        used = set()
        for idx, n in enumerate(rains):
            # print (q, used)
            if n>0:
                if n in used:
                    return []
                else:
                    pos[n].pop()
                    if pos[n]:
                        heapq.heappush(q, (pos[n][-1], n))
                    else:
                        heapq.heappush(q, (math.inf, n))
                    used.add(n)
                sol[idx] = -1
            elif n==0:
                if q:
                    _, val = heapq.heappop(q)
                    sol[idx] = val
                    used.remove(val)
        return sol
from queue import PriorityQueue
class Solution:
    def avoidFlood(self, rains: List[int]) -> List[int]:
        pq = PriorityQueue()
        rainOnLake = {}        
        for i, v in enumerate(rains):
            if v > 0:
                if v not in rainOnLake: rainOnLake[v] = []
                rainOnLake[v].append(i)
        
        lakes = {}
        ans = [-1] * len(rains)
        for i , v in enumerate(rains):
            if v > 0:
                if lakes.get(v, 0) > 0: 
                    return []
                lakes[v] = 1
                rainSched = rainOnLake[v]
                rainSched.pop(0)
                if rainSched:
                    nextRain = rainSched[0]
                    pq.put((nextRain, v))
            elif v == 0:
                if pq.qsize() <= 0:
                    ans[i] = 111111111
                else:
                    (d, l) = pq.get()
                    ans[i] = l
                    lakes[l] = 0
        return ans
                

class Solution:
    def avoidFlood(self, rains: List[int]) -> List[int]:
        ret = [1]*len(rains)
        filled = dict()
        dryDays = list()
        for ind,rain in enumerate(rains):
            if rain>0:
                ret[ind] = -1
                if rain not in filled:
                    ret[ind]=-1
                    filled[rain]=ind
                    continue
                else:
                    if not dryDays:
                        return []
                    found = False
                    print(ind)
                    for day in dryDays:
                        if day > filled[rain]:
                            ret[day]=rain
                            dryDays.remove(day)
                            found = True
                            filled[rain] = ind
                            break
                    if not found:
                        return []
            else:
                dryDays.append(ind)
        return ret
                        
            
            

from bisect import bisect, bisect_left
from typing import List


class Solution:
  def avoidFlood(self, rains: List[int]) -> List[int]:

    result = []
    filled = {}
    free_day = []
    for index, city in enumerate(rains):
      if city:
        result.append(-1)
        if city not in filled:
          filled[city] = index
        else:
          if len(free_day) > 0 and free_day[-1] > filled[city]:
            dry_day = bisect_left(free_day, filled[city])
            result[free_day.pop(dry_day)] = city
            filled[city] = index
            pass
          else:
            return []
          pass

        pass
      else:
        result.append(1)
        free_day.append(index)
        pass
      pass

    return result
from bisect import bisect_left

class Solution:
    def avoidFlood(self, rains: List[int]) -> List[int]:
        """
        We only care about the lakes that receive rain multiple times
        
        Those lakes will need to, ideally, be drained prior
        
        When we see a repeat --- we need to find the first "Dry" after the last
        and pop it
        """
        ans = [float('-inf') for _ in rains]
        dry = []
        tracker = {}
        for i, r in enumerate(rains):
            if r != 0:
                ans[i] = -1
                if r in tracker:
                    idx = bisect_left(dry, tracker[r]+1)
                    if idx == len(dry):
                        return []
                    else:
                        res = dry[idx]
                        ans[res] = r
                        dry.pop(idx)
                tracker[r] = i
            else:
                dry.append(i)
        for i in range(len(ans)):
            if ans[i] == float('-inf'):
                ans[i] = 1
        return ans
# from bisect import bisect_left

# class Solution:
#     def avoidFlood(self, rains: List[int]) -> List[int]:
#         flooded = {}
#         dry_days = []
#         out = [-1 for _ in range(len(rains))]
        
#         for i in range(len(rains)):
#             day = rains[i]
#             if day > 0:
#                 if day in flooded:
#                     if dry_days:
#                         # found = False
#                         # for d in range(len(dry_days)):
#                         #     dry_day = dry_days[d]
#                         #     if dry_day > flooded[day]:
#                         #         out[dry_day] = day
#                         #         dry_days.pop(d)
#                         #         found = True
#                         #         break
#                         # if not found:
#                         #     return []
#                         dry = bisect_left(dry_days, flooded[day])
#                         if dry == len(dry_days):
#                             return []
#                         else:
#                             dry_day = dry_days.pop(dry)
#                             out[dry_day] = day
#                     else:
#                         return []
#                 flooded[day] = i
#             else:
#                 dry_days.append(i)
#         for dry_day in dry_days:
#             out[dry_day] = 1
#         return out

from sortedcontainers import SortedList as sl
class Solution:
    def avoidFlood(self, rains: List[int]) -> List[int]:
        ans = [-1]*(len(rains))
        has_rained = dict()
        free = sl()
        for i in range(len(rains)):
            # print(free)
            if rains[i]==0:
                free.add(i)
            else:
                if rains[i] not in has_rained:
                    has_rained[rains[i]]=i
                else:
					# no free days are available
                    if len(free)==0:
                        return []
					#finding the index of the free day that came after the first occurance of
					# rains[i]
                    idx = free.bisect_left(has_rained[rains[i]])
                    # print(free,idx,i,has_rained)
                    if idx<len(free):
                        ans[free[idx]]=rains[i]
						# updating the index of rains[i] for future it's future occurances
                        has_rained[rains[i]] = i
                        free.remove(free[idx])
                    else:
						#if no such index exists then return
                        return []
        if len(free):
            while free:
				# choosing some day to dry on the remaining days
                ans[free.pop()]=1
        return ans
class Solution:
  def avoidFlood(self, rains: List[int]) -> List[int]:
    last_ids = {}
    next_ids = [0] * len(rains)
    for i in reversed(range(len(rains))):
      if rains[i] > 0:
        if rains[i] in last_ids:
          next_ids[i] = last_ids[rains[i]]
        last_ids[rains[i]] = i
    
    prio = []
    result = [-1] * len(rains)
    #print(next_ids)
    for i in range(len(rains)):
      if rains[i] > 0:
        if len(prio) > 0 and prio[0] <= i:
          #print('exit', prio)
          return []
        if next_ids[i] > 0:
          #print(i, 'push', next_ids[i])
          heapq.heappush(prio, next_ids[i])
      else:
        if len(prio) > 0:
          result[i] = rains[heapq.heappop(prio)]
          #print(i, 'pop', result)
        else:
          result[i] = 1
    return result
class Solution:
    def avoidFlood(self, rains: List[int]) -> List[int]:
        free_days = []
        lake_tracker = {}
        free_days_balance = 0
        ans = [ -1 if rains[i] > 0 else 1 for i in range( len(rains) ) ]
        for i in range( len(rains) ):
            if rains[i] > 0:
                lake = rains[i]
                if lake in lake_tracker:
                    if free_days_balance > 0:
                        index = lake_tracker[lake]
                        
                        fnd = None
                        for free_day in free_days:
                            if index < free_day:
                                fnd = free_day
                                break
                        
                        if not fnd:
                            return []
                        
                        free_days_balance = free_days_balance - 1
                        lake_tracker[lake] = i
                        
                        ans[ fnd ] = lake
                        free_days.remove(fnd)
                        
                    else:
                        return []
                else:
                    lake_tracker[ lake ] = i
            else:
                free_days_balance = free_days_balance + 1
                free_days.append(i)
        return ans
                
                    

class Solution:
    def avoidFlood(self, rains: List[int]) -> List[int]:
        op=[]
        lakes=dict()
        days=[]
        for i,r in enumerate(rains):
            if r>0:
                op.append(-1)
                if r in lakes.keys():
                    day=lakes[r]
                    v=-1
                    for d in days:
                        if d>day:
                            v=d
                            break
                    if v>0:
                        days.remove(v)
                        op[v]=r
                    else:
                        return []
                lakes[r]=i
            else:
                op.append(99999)
                days.append(i)
            
        return op
class Solution(object):
    def avoidFlood(self, rains):
        aux = dict()
        for i in range(len(rains)):
            if rains[i] not in aux:
                aux[rains[i]] = deque()
            aux[rains[i]].append(i)
        q = []
        ans = [1] * len(rains)
        for i in range(len(rains)):
            if rains[i] == 0:
                if q:
                    index, val = heapq.heappop(q)
                    if index < i:
                        return []
                    else:
                        ans[i] = val
            else:
                ans[i] = -1
                if len(aux[rains[i]]) > 1:
                    aux[rains[i]].popleft()
                    heapq.heappush(q, (aux[rains[i]][0], rains[i]))
        if len(q):  return []
        return ans
class Solution:
    def avoidFlood(self, rains: List[int]) -> List[int]:
        ans=[]
        
        lakes={}
        freeDays=[]

        for i in range(0,len(rains)):
            if(rains[i]!=0):
                if(rains[i] not in lakes):
                    lakes[rains[i]]=[]
                    lakes[rains[i]].append(0)
                    lakes[rains[i]].append(i)
                elif(lakes[rains[i]][0]==1 and len(freeDays)>0):
                    for j in range(0,len(freeDays)):
                        if(freeDays[j]>lakes[rains[i]][1]):
                            ans[freeDays[j]]=rains[i]
                            lakes[rains[i]][0]-=1
                            lakes[rains[i]][1]=i
                            freeDays.pop(j)
                            break
                lakes[rains[i]][0]+=1
                lakes[rains[i]][1]=i
                if(lakes[rains[i]][0]>1):
                    return []
                ans.append(-1)
            else:
                freeDays.append(i)
                ans.append(1)

                    
        
        
        
        
        
        return ans
class Solution:
    def avoidFlood(self, rains: List[int]) -> List[int]:
        lakes = collections.defaultdict(bool)
        lastrain = collections.defaultdict(int)
        res = []
        dry = []
        for k,v in enumerate(rains):
            if v > 0:
                if not lakes[v]:
                    lakes[v] = True
                    res.append(-1)
                    lastrain[v] = k
                else:
                    # lakes[v] == True
                    if dry == []:
                        return []
                    else:
                        # check if there is a dry day we can use
                        i = 0
                        found = False
                        while i < len(dry) and not found:
                            if dry[i] > lastrain[v]:
                                found = True
                                dry_day = dry[i]
                            else:
                                i += 1
                        if found:
                            res[dry_day] = v
                            lastrain[v] = k
                            dry.pop(i)
                            res.append(-1)
                        else:
                            return []
            elif v == 0:
                res.append(1)
                dry.append(k)
        return res

class Solution:
    def avoidFlood(self, rains: List[int]) -> List[int]:
        # we can do this in O(n) space 
        lakes = {}
        zeroes = []
        
        length = len(rains)
        
        
        for i, rain in enumerate(rains):
            if rain == 0:
                zeroes.append(i)
                continue
            
            if rain in lakes: 
                lake_index = lakes[rain]
                
                found = False
                
                for j, zero in enumerate(zeroes):
                    if zero > lake_index:
                        rains[zero] = rain
                        found = True
                        del zeroes[j]
                        break
                
                if not found: return []

                lakes[rain] = i
                rains[i] = -1
            else:
                lakes[rain] = i
                rains[i] = -1
        
        for zero in zeroes: rains[zero] = 1
                
        return rains

import bisect
class Solution:
    def avoidFlood(self, rains: List[int]) -> List[int]:
        dry_days=[]
        last_rainy_day={}
        ans=[-1 if rain>0 else 1 for rain in rains]
        for i,rain in enumerate(rains): 
            if not rain:
                dry_days.append(i)
            else:
                if rain not in last_rainy_day:
                    last_rainy_day[rain]=i 
                else:
                    if not dry_days:
                        return []
                    else:
                        index=bisect.bisect_left(dry_days,last_rainy_day[rain])
                        if index>=len(dry_days):
                            return []
                        ans[dry_days.pop(index)]=rain
                        last_rainy_day[rain]=i
        return  ans
class Solution:
    def avoidFlood(self, rains: List[int]) -> List[int]:
        nex = [-1] * len(rains)
        last = {}
        for i, n in enumerate(rains):
            if n in last:
                nex[last[n]] = i
            last[n] = i

        prio, answer = [], []
        for i, event in enumerate(rains):
            if prio and prio[0] <= i:
                return []
            
            if event != 0:
                if nex[i] != -1:
                    heapq.heappush(prio, nex[i])
                answer.append(-1)
            else:
                answer.append(rains[heapq.heappop(prio)] if prio else 1)
        
        return answer
class Solution:
    def avoidFlood(self, rains: List[int]) -> List[int]:
        full_lakes = {}
        
        sunny_days = []
        
        res = []
        
        for i, rain in enumerate(rains):
            if rain == 0:
                res.append(1)
                sunny_days.append(i)
            else:
                if rain in full_lakes:
                    last_rain = full_lakes[rain]
                    
                    earliest_sunny_day = -1
                    for day in sunny_days:
                        if day > last_rain and day < i:
                            earliest_sunny_day = day
                            sunny_days.remove(day)
                            break
                            
                    if earliest_sunny_day == -1:
                        return []
                    
                    res[earliest_sunny_day] = rain
                    res.append(-1)

                else:
                    res.append(-1)
                full_lakes[rain] = i
        return res
class Solution:
    
    def backtrack(self, rains, full, position, seq):
        if position >= len(rains):
            return True
        if rains[position] > 0:
            if rains[position] in full:
                return False
            seq.append(-1)
            full.add(rains[position])
            if self.backtrack(rains, full, position + 1, seq):
                return True
            full.remove(rains[position])
            seq.pop()
        elif rains[position] == 0:
            # must choose one lake that is full to dry
            for lake in full:
                seq.append(lake)
                full.remove(lake)
                if self.backtrack(rains, full, position + 1, seq):
                    return True
                full.add(lake)
                seq.pop()
            if len(full) < 1:
                seq.append(1) # random lake
                if self.backtrack(rains, full, position + 1, seq):
                    return True
                seq.pop()
    
    def avoidFloodBacktrack(self, rains: List[int]) -> List[int]:
        seq = []
        full = set()
        if not self.backtrack(rains, full, 0, seq):
            return []
        return seq
    
    def avoidFlood(self, rains: List[int]) -> List[int]:
        spares = []
        recent = dict()
        full = set()
        ans = []
        for i in range(len(rains)):
            if rains[i] > 0:
                ans.append(-1)
                if rains[i] in full:
                    # we need to have dried this lake for sure
                    valid = False
                    for d in range(len(spares)):
                        dry = spares[d]
                        if dry > recent[rains[i]]:
                            ans[dry] = rains[i]
                            full.remove(rains[i])
                            del spares[d]
                            valid = True
                            break
                    if not valid:
                        return []
                elif rains[i] in recent and recent[rains[i]] == i - 1:
                    # no chance to dry this one
                    return []
                else:
                    full.add(rains[i])
                recent[rains[i]] = i
            else:
                # we can dry one lake
                # greedy chooses some random lake
                # that will be replaced if needed
                ans.append(1)
                spares.append(i)
        return ans
    
if False:
    assert Solution().avoidFlood([69,0,0,0,69]) == [-1, 69, 1, 1, -1]
    assert Solution().avoidFlood([1,2,0,0,2,1]) == [-1,-1,2,1,-1,-1]
    assert Solution().avoidFlood([1,2,0,1,2]) == []
    assert Solution().avoidFlood([10,20,20]) == []
import sortedcontainers

class Solution:
    def avoidFlood(self, rains: List[int]) -> List[int]:
        n = len(rains)
        
        ans = [-1] * n
        
        track = {}
        ptr = 0
        dries = sortedcontainers.SortedList()
        
        while ptr < n:
            lake = rains[ptr]
            
            if lake > 0:
                if lake in track:
                    last_rain = track[lake]
                    
                    if not dries: return []
                    
                    idx = dries.bisect_right(last_rain)
                    if idx >= len(dries): return []
                    
                    dry_pos = dries[idx]
                    if dry_pos < last_rain: return []
                    
                    dries.discard(dry_pos)
                    
                    ans[dry_pos] = lake
                    track[lake] = ptr
                else:
                    track[lake] = ptr
            else:
                dries.add(ptr)
                
            ptr += 1
        
        for i in dries: ans[i] = 1
            
        return ans
        

from heapq import heappush, heappop

class Solution:
    def avoidFlood(self, rains: List[int]) -> List[int]:
        appears = dict()
        for i in range(len(rains)):
            if not rains[i]:
                continue
            if not rains[i] in appears:
                appears[rains[i]] = []
            appears[rains[i]].append(i)
            
        next_rain = dict()
        for v in appears.values():
            for i in range(len(v) - 1):
                next_rain[v[i]] = v[i + 1]
        
        h = []
        ans = [-1] * len(rains)
        for i in range(len(rains)):
            if rains[i]:
                if i in next_rain:
                    heappush(h, (next_rain[i], rains[i]))
            else:
                if h:
                    day, idx = heappop(h)
                    if day < i:
                        return []
                    else:
                        ans[i] = idx
                else:
                    ans[i] = 1
        
        if h:
            return []
        
        return ans
class Solution:
    def avoidFlood(self, rains: List[int]) -> List[int]:
        # we can do this in O(n) space 
        lakes = {}
        zeroes = []
        length = len(rains)
        
        for i, rain in enumerate(rains):
            if rain == 0:
                zeroes.append(i)
                continue
            
            if rain in lakes: 
                lake_index = lakes[rain]
                
                found = False
                
                for j, zero in enumerate(zeroes):
                    if zero > lake_index:
                        rains[zero] = rain
                        found = True
                        del zeroes[j]
                        break
                
                if not found: return []

                lakes[rain] = i
                rains[i] = -1
            else:
                lakes[rain] = i
                rains[i] = -1
        
        for zero in zeroes: rains[zero] = 1
                
        return rains

class Solution:
    def avoidFlood(self, rains: List[int]) -> List[int]:
        zeros = []
        last = {}
        ans = [-1]*len(rains)
        
        for i in range(len(rains)):
            if rains[i] == 0:
                ans[i] = 1
        
        for i, lake in enumerate(rains):
            if lake == 0:
                zeros.append(i)
            else:
                if lake in last:
                    prev_idx = last[lake]
                    zero = bisect_left(zeros, prev_idx)
                    if zero < len(zeros):
                        # found a valid lake
                        zero_lake = zeros.pop(zero)
                        last[lake] = i
                        ans[zero_lake] = lake
                    else:
                        return []
                else:
                    last[lake] = i
        return ans
class Solution:
    def avoidFlood(self, rains: List[int]) -> List[int]:
        counter=collections.Counter()
        firstSeen={}
        # day=0
        empty=[-1]*len(rains)
        # stack=collections.deque()
        # slow=0
        sunny={}
        for day,lake in enumerate(rains):
            if lake>0:
                if counter[lake]>=1:        
                    for index in sunny:
                        if index>firstSeen[lake]:
                            empty[index]=lake
                            counter[lake]-=1
                            del sunny[index]
                            break
                    if counter[lake]>=1:
                        return []
                counter[lake]+=1
                firstSeen[lake]=day
            else:
                sunny[day]=1
        # print(sunny)
        for day in sunny:
            empty[day]=1
        return empty
       
            

class Solution:
    def avoidFlood(self, rains: List[int]) -> List[int]:
        closest = []
        locs = collections.defaultdict(collections.deque)
        for i, lake in enumerate(rains):
            locs[lake].append(i)
        ret = []
        for i, lake in enumerate(rains):
            if closest and closest[0] == i:
                return []
            if not lake:
                if not closest:
                    ret.append(1) 
                    continue
                nxt = heapq.heappop(closest)
                ret.append(rains[nxt])
            else:
                l = locs[lake]
                l.popleft()
                if l:
                    nxt = l[0]
                    heapq.heappush(closest, nxt)
                ret.append(-1)
        return ret

class Solution:
    def avoidFlood(self, rains: List[int]) -> List[int]:
        """
        #O(n^2) working sol
        ans = [1 for i in range(len(rains))]
        d = collections.defaultdict(int)
        d[0]=0
        
        for i in range(len(rains)):
            d[rains[i]]+=1
            if rains[i]==0:
                #look for the nearest value that exists in the dict we got
                for x in range(i+1,len(rains)):
                    if rains[x] in d and not rains[x]==0:
                        #print(d,d[rains[x]],rains[x])
                        d[0]-=1
                        ans[i] = rains[x]
                        d[rains[x]]-=1
                        if d[rains[x]]==0: del d[rains[x]]
                        break
            else:
                #you gotta get out early of a bad pattern that cannot be salvaged
                if d[rains[i]]>1:
                    return []
                ans[i] = -1
        
        return ans
        """
        
        ans = [1 for i in range(len(rains))]
        d = collections.defaultdict(int)
        d[0]=0
        #preprosess, find all  #:0#0#0...
        # as d grows, put corresponding value here in a heap
        # every time heap pops, we get the nearest value that exists in the dict we got
        p = {}
        x = collections.defaultdict(int)
        x[0] = 0
        for i in range(len(rains)):
            if rains[i] in p:
                #print(x[0],rains[i],x[rains[i]])
                if x[0]>=x[rains[i]]:
                    p[rains[i]] += [i]
            else:
                p[rains[i]] = []
            x[rains[i]]+=1
        p[0] = []
            
        #print(p)       
            
        s= set()
        h = []
        for i in range(len(rains)):

            d[rains[i]]+=1

            if rains[i]!=0 and rains[i] not in s:
                if rains[i] in p and p[rains[i]] != []:
                    for j in p[rains[i]]:
                        heappush(h,j)
                s.add(rains[i])
            #print(d,h)
             
            if rains[i]==0:
                #look for the nearest value that exists in the dict we got
                """
                for x in range(i+1,len(rains)):
                    print(x," is x")
                    if rains[x] in d and not rains[x]==0:
                        #print(d,d[rains[x]],rains[x])
                        if h: 
                            pop = heappop(h)
                            print(pop,x,"compare")
                        
                        d[0]-=1
                        ans[i] = rains[x]
                        d[rains[x]]-=1
                        if d[rains[x]]==0: del d[rains[x]]
                        break
                """
                
                if h:
                    pop = heappop(h)
                    d[0]-=1
                    
                    ans[i] = rains[pop]
                    if rains[pop] not in d:
                        rains[pop] = 1
                    else:
                        d[rains[pop]]-=1
                    if d[rains[pop]]==0: del d[rains[pop]]
                
                
            else:
                
                #you gotta get out early of a bad pattern that cannot be salvaged
                if d[rains[i]]>1:
                    return []
                #find the next equal closest value past a zero.
                ans[i] = -1
            #print(h,i,"heap at end")
        
        return ans
        
        
        
        
                
        
                
            
            
        
            
        
                        
                        
        
                
       
        
                
                
                
