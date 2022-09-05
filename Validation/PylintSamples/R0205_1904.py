class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        points.sort(key = lambda x: x[0]*x[0] + x[1]*x[1])
        return points[:K]
class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        points.sort(key=lambda x:x[0]*x[0]+x[1]*x[1])
        return points[:K]
class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        return (sorted(points,key=lambda x : x[0]**2 + x[1]**2))[:K]
class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        # nlog(n)
        # points.sort(key=lambda item: item[0]**2 + item[1]**2)
        # return points[:K]
        
        # nlog(k)
        # return heapq.nsmallest(K, points, key=lambda item: item[0]**2 + item[1]**2)
        
        # O(n)
        def compare(p1, p2):
            return p1[0]**2 + p1[1]**2 - (p2[0]**2 + p2[1]**2)
        
        def partition(points, l, r):
            pivot = points[l]
            while l < r:
                while l < r and compare(points[r], pivot) >=0:
                    r -= 1
                points[l] = points[r]
                while l < r and compare(points[l], pivot) <=0:
                    l += 1
                points[r] = points[l]
            points[l] = pivot
            return l
        
        l, r = 0, len(points) - 1
        while l < r:
            p = partition(points, l, r)
            if p == K:
                break
            if p < K:
                l = p + 1
            else:
                r = p - 1
        return points[:K]
class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        heap = []
        for x,y in points:
            dist = - (x**2 + y**2)
            if len(heap) >= K:
                heapq.heappushpop(heap, (dist, x, y))
            else:
                heapq.heappush(heap, (dist, x, y))
        return [[x,y] for dist, x, y in heap]
class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        def partition(start, end):
            pivot = start
            left = start+1
            right = end
            while True:
                while left < right and not comparer(points[left], points[pivot]):
                    left += 1
                while left <= right and comparer(points[right], points[pivot]):
                    right -= 1
                if left >= right: 
                    break
                points[left], points[right] = points[right], points[left]
            points[pivot], points[right] = points[right], points[pivot]
            return right
        def comparer(point1, point2):
            return (point1[0]**2+point1[1]**2)>=(point2[0]**2+point2[1]**2)
        
        left, right, mid = 0, len(points)-1, 0
        while left<=right:
            mid = partition(left, right)
            if mid == K-1:
                break
            elif mid > K-1:
                right = mid - 1
            else:
                left = mid + 1
        return points[:K]
class Solution(object):
    def kClosest(self, points, K):
        points.sort(key = lambda P: P[0]**2 + P[1]**2)
        return points[:K]
from queue import PriorityQueue

class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        pq = PriorityQueue()
        
        for i, point in enumerate(points):
            point_distance = point[0]**2 + point[1]**2
            if i < K:
                pq.put((-point_distance, point))
            
            if i >=  K:
                min_point_distance, min_point = pq.queue[0]
                
                if point_distance < -min_point_distance:
                    pq.get()
                    pq.put((-point_distance, point))
            
        results = []
        while not pq.empty():
            _, point = pq.get()
            results.append(point)
        return results
import heapq
class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        hp = []
        heapq.heapify(hp)
        for x,y in points:
            ds = x**2 + y**2  # distance square
            if len(hp) < K:
                heapq.heappush(hp, (-ds, [x, y]))
            else:
                heapq.heappushpop(hp, (-ds, [x, y]))
        return [xy for ds, xy in hp]
class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        
        def compare(point1, point2):
            return (point1[0] ** 2 + point1[1] ** 2 - point2[0] ** 2 - point2[1] ** 2) >= 0
            
        def partition(l, r, arr):
            pivot = arr[l]
            
            while l < r:
                while l < r and compare(arr[r], pivot):
                    r -= 1
                arr[l] = arr[r]
                
                while l < r and compare(pivot, arr[l]):
                    l += 1
                arr[r] = arr[l]
            
            arr[l] = pivot
            
            return l
        
        lo = 0
        hi = len(points) - 1
        
        while lo <= hi:
            mid = partition(lo, hi, points)
            if mid == K:
                break
            if mid < K:
                lo = mid + 1
            else:
                hi = mid - 1
        
        return points[0:K]
                
            

import random

class Solution:
    
    # problem: https://leetcode.com/problems/k-closest-points-to-origin/
    # type : divide and conquer
    # will keep on partitioning and partially sorting part of the values
    
    # T = O(N) on expectation becuase of the the random pivots
    # S = O(1)
    
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        
        def dist(x):
            return points[x][0]**2 + points[x][1]**2
        
        def partial_sort(i,j,K):
            # i : the start index to sort from, initally will be 0
            # j : the end index, initialy will be len(points) - 1
            # K: # elements to be partially sorted
            
            if i >= j:
                return
            
            # 1. pick a random pivot:
            pivot = random.randint(i,j) # inclusive of j
            
            # switch with index i, initially will swap with i = 0
            points[i], points[pivot] = points[pivot], points[i]
            
            # partitions function,
            # will find elements that are smaller than pivot and put to left
            # elements larger than pivot will go to right
            # will return the #sorted elements
            num_sorted = partition(i,j)
            
            # will have to check if K values are sorted
            # continue after the partition func
            
            # 1. in case we have sorted more than K elements
            # initially i ==0, but i will change for each iteration
            if K == num_sorted:
                return
            
            if K < num_sorted:
                partial_sort(i,num_sorted - 1, K)
            elif K > num_sorted:
                partial_sort(num_sorted + 1, j, K)
            
        def partition(i,j):
            
            # pivot will be at i
            oi = i # pivot index, old-i
            i += 1 # will keep on checking for the values after oi
            
            while True: 
                # pass 1 -->>
                while i < j and dist(i) < dist(oi):
                    i += 1
                # pass 2 <<-- of j
                while j >= i and dist(j) >= dist(oi):
                    j -= 1
                # stopping condition
                if i >= j:
                    break
                # if it didnt stop, swap i,j
                points[i], points[j] = points[j], points[i]
                
            # swap pivot with j
            points[oi],points[j] = points[j], points[oi]
            
            return j
        
        partial_sort(0,len(points)-1,K)
        return points[:K]
                
                
            
            
        
        
            
        

class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        # sort
#         s_p = sorted(points, key = lambda p:p[0]**2 + p[1]**2)
        
#         return s_p[:K]
    
        # Divide and Conquer
        dist = lambda i: points[i][0]**2 + points[i][1]**2

        def sort(i, j, K):
            # Partially sorts A[i:j+1] so the first K elements are
            # the smallest K elements.
            if i >= j: return

            # Put random element as A[i] - this is the pivot
            k = random.randint(i, j)
            points[i], points[k] = points[k], points[i]

            mid = partition(i, j)
            if K < mid - i + 1:
                sort(i, mid - 1, K)
            elif K > mid - i + 1:
                sort(mid + 1, j, K - (mid - i + 1))

        def partition(i, j):
            # Partition by pivot A[i], returning an index mid
            # such that A[i] <= A[mid] <= A[j] for i < mid < j.
            oi = i
            pivot = dist(i)
            i += 1

            while True:
                while i < j and dist(i) < pivot:
                    i += 1
                while i <= j and dist(j) >= pivot:
                    j -= 1
                if i >= j: break
                points[i], points[j] = points[j], points[i]

            points[oi], points[j] = points[j], points[oi]
            return j

        sort(0, len(points) - 1, K)
        return points[:K]

class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        # nlog(n)
        # points.sort(key=lambda item: item[0]**2 + item[1]**2)
        # return points[:K]
        
        # nlog(k)
        # return heapq.nsmallest(K, points, key=lambda item: item[0]**2 + item[1]**2)
        
        # O(n)
        def compare(p1, p2):
            return p1[0]**2 + p1[1]**2 - (p2[0]**2 + p2[1]**2)
        
        def partition(points, l, r):
            pivot = points[l]
            while l < r:
                while l < r and compare(points[r], pivot) >=0:
                    r -= 1
                points[l] = points[r]
                while l < r and compare(points[l], pivot) <=0:
                    l += 1
                points[r] = points[l]
            points[l] = pivot
            return l
        
        l, r = 0, len(points) - 1
        while l < r:
            p = partition(points, l, r)
            if p == K - 1:
                break
            if p < K - 1:
                l = p + 1
            else:
                r = p - 1
        return points[:K]
import operator
class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        points.sort(key = lambda p : p[0] ** 2 + p[1] ** 2)
        
        return points[:K]
class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        def partition(start, end):
            pivot = start
            left = start+1
            right = end
            while True:
                while left < right and comparer(points[pivot], points[left]):
                    left += 1
                while left <= right and comparer(points[right], points[pivot]):
                    right -= 1
                if left >= right: 
                    break
                points[left], points[right] = points[right], points[left]
            points[pivot], points[right] = points[right], points[pivot]
            return right
        def comparer(point1, point2):
            return (point1[0]**2+point1[1]**2)>=(point2[0]**2+point2[1]**2)
        
        left, right, mid = 0, len(points)-1, 0
        while left<=right:
            mid = partition(left, right)
            if mid == K-1:
                break
            elif mid > K-1:
                right = mid - 1
            else:
                left = mid + 1
        return points[:K]
import random

class Solution:
    
    # problem: https://leetcode.com/problems/k-closest-points-to-origin/
    # type : divide and conquer
    # will keep on partitioning and partially sorting part of the values
    
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        
        def dist(x):
            return points[x][0]**2 + points[x][1]**2
        
        def partial_sort(i,j,K):
            # i : the start index to sort from, initally will be 0
            # j : the end index, initialy will be len(points) - 1
            # K: # elements to be partially sorted
            
            if i >= j:
                return
            
            # 1. pick a random pivot:
            pivot = random.randint(i,j) # inclusive of j
            
            # switch with index i, initially will swap with i = 0
            points[i], points[pivot] = points[pivot], points[i]
            
            # partitions function,
            # will find elements that are smaller than pivot and put to left
            # elements larger than pivot will go to right
            # will return the #sorted elements
            num_sorted = partition(i,j)
            
            # will have to check if K values are sorted
            # continue after the partition func
            
            # 1. in case we have sorted more than K elements
            # initially i ==0, but i will change for each iteration
            if K < num_sorted + 1 - i:
                partial_sort(i,num_sorted - 1, K)
            elif K > num_sorted + 1 - i:
                partial_sort(num_sorted + 1, j, K - (num_sorted + 1 - i))
            
        def partition(i,j):
            
            # pivot will be at i
            oi = i # pivot index, old-i
            i += 1 # will keep on checking for the values after oi
            
            while True: 
                # pass 1 -->>
                while i < j and dist(i) < dist(oi):
                    i += 1
                # pass 2 <<-- of j
                while j >= i and dist(j) >= dist(oi):
                    j -= 1
                # stopping condition
                if i >= j:
                    break
                # if it didnt stop, swap i,j
                points[i], points[j] = points[j], points[i]
                
            # swap pivot with j
            points[oi],points[j] = points[j], points[oi]
            
            return j
        
        partial_sort(0,len(points)-1,K)
        return points[:K]
                
                
            
            
        
        
            
        

class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        heap = []
        for x,y in points:
            dist = - (x**2 + y**2)
            if len(heap) >= K:
                heapq.heappushpop(heap, (dist, x, y))
            else:
                heapq.heappush(heap, (dist, x, y))
        return [[x,y] for dist, x, y in heap]
        
#         heap = []
        
#         for (x, y) in points:
#             dist = -(x*x + y*y)
#             if len(heap) == K:
#                 heapq.heappushpop(heap, (dist, x, y))
#             else:
#                 heapq.heappush(heap, (dist, x, y))
        
#         return [(x,y) for (dist,x, y) in heap]

from queue import PriorityQueue

class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        pq = PriorityQueue()
        
        for p in points: #O(nlogk)
            new_priority = -1* (p[0]*p[0] + p[1]*p[1])
            if pq.qsize() == K:
                old_priority, old_point = pq.get()
                if new_priority > old_priority:
                    pq.put((new_priority, p))
                else:
                    pq.put((old_priority, old_point))
            else:
                pq.put((new_priority, p))
            
        res = []
        for i in range(K):
            priority, p = pq.get()
            res.append(p)
            
        return res
import heapq

class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        
        heap = []
        
        for point in points:
            dist = point[0] * point[0] + point[1] * point[1]
            if len(heap) < K:
                heapq.heappush(heap, (-dist, point))
                continue
            
            if -heap[0][0] > dist:
                heapq.heappush(heap, (-dist, point))
                heapq.heappop(heap)
            
        return [x[1] for x in heap]
class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        
        
        distance = [   ((x[0]**2 + x[1]**2)**(1/2),i) for i,x in enumerate(points)    ]
        heapq.heapify(distance)
        ans = []
        
        while K > 0 :
            element = heapq.heappop(distance)
            ans.append(points[element[1]])
            K -= 1
        return ans
import heapq
class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        if K >= len(points): return points
        
        heap = []
                
        #k 
        for i in range(K):
            distance = -1 * self.getDistance(points[i])
            heapq.heappush(heap, (distance, i))

        # n
        for i in range(K, len(points)): 
            distance = -1 * self.getDistance(points[i])
            if distance > heap[0][0]: 
                heapq.heappop(heap)
                heapq.heappush(heap, (distance, i))
            
        result = []
        # O(k)
        for p in heap:
            result.append(points[p[1]])
        
        return result
        
        
    
    def getDistance(self, point):
        return point[0]**2 + point[1]**2
        
        
        
        
        
        
        

import heapq
class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        return heapq.nsmallest(K, points, lambda p: p[0]**2 + p[1]**2)
import random

class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        
        def sort(left, right, K):
            if left < right:
                pivot = partition(left, right)
                
                if pivot == K:
                    return
                elif pivot < K:
                    sort(pivot+1, right, K)
                else:
                    sort(left, pivot-1, K)
        
        def partition(left, right):
            pivot = points[right]
            anchor = left
            
            for i in range(left, right):
                if points[i][0]**2+points[i][1]**2 <= pivot[0]**2+pivot[1]**2:
                    points[anchor], points[i] = points[i], points[anchor]
                    anchor += 1
                    
            points[anchor], points[right] = points[right], points[anchor]
            
            return anchor
        
        sort(0, len(points)-1, K)
        
        return points[:K]
from queue import PriorityQueue

class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        pq = PriorityQueue()
        
        for p in points: #O(nlogk)
            new_priority = -1* (pow(p[0],2) + pow(p[1], 2))
            if pq.qsize() == K:
                old_priority, old_point = pq.get()
                if new_priority > old_priority:
                    pq.put((new_priority, p))
                else:
                    pq.put((old_priority, old_point))
            else:
                pq.put((new_priority, p))
            
        res = []
        for i in range(K):
            priority, p = pq.get()
            res.append(p)
            
        return res
class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        points.sort(key= lambda x: x[0]**2+x[1]**2)
        points= points[0:K]
        return points

from queue import PriorityQueue

class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        kPq = PriorityQueue()
        cnt = 0
        for point in points:
            w = point[0] * point[0] + point[1] * point[1]
            kPq.put([-w, point])
            cnt += 1
            if cnt > K:
                kPq.get()
        r = []
        while not kPq.empty():
            item = kPq.get()
            r.append(item[1])
        return r


import queue
class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        # priority queue (minheap)
        # put distance: index of points
        if not points: return []
        q = queue.PriorityQueue()
        for l in points:
            # put negative distance to make heap as max heap
            q.put((-(l[0]**2 + l[1]**2), l))
            if q.qsize() > K:
                q.get()
        res = []
        while K > 0:
            res.append(q.get()[1])
            K -= 1
        return res
            

class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        # get_distance = lambda x: x[0] **2 + x[1] **2
        # tmp = list(map(lambda x: [x[0], x[1], get_distance(x)], points))
        # # print(tmp)
        # tmp.sort(key=lambda x: x[2])
        # res = list(map(lambda x: [x[0], x[1]], tmp[:K]))
        # return res
        
        distance_map = list(map(lambda x: (x, x[0] **2 + x[1] **2), points))
        from queue import PriorityQueue
        heap = PriorityQueue(K)
        
        for points, dist in distance_map:
            if not heap.full():
                heap.put((-dist, points))
                # print(heap.queue)
                # print(heap.queue[0][0])
            elif dist < - heap.queue[0][0]:
                # print('heap', heap.queue)
                _ = heap.get()
                heap.put((-dist, points))
            
        res = []
        while not heap.empty():
            res.append(heap.get()[1])
        return res
class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        points.sort(key=lambda x: x[0]**2 + x[1]**2)
        return points[:K]
import math
class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        points.sort(key = lambda p: p[0]**2+p[1]**2)
        return points[:K]
class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        distance = lambda x, y: sqrt(x**2 + y**2)
        k = 0
        output = []
        for s in sorted(points, key=lambda x: distance(x[0], x[1])):
            if k < K:
                output.append(s)
            else:
                break
            k += 1
        return output
class Solution:
    def kClosest(self, points, K: int):
      
        return [points[i] for d,i in sorted([[x[0]**2+x[1]**2,i] for i,x in enumerate(points)])[0:K]]   
import math 
class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        
        distances = {}
        
        d = []
        
#         large = -float(inf)
#         small = float(inf)
        
        
        for i in range(len(points)):
            
            dist = math.sqrt(points[i][0] ** 2 + points[i][1] ** 2)
            # print(dist)
            distances[dist] = distances.get(dist, []) + [points[i]]
            d.append(dist)
            
        # print(distances)
        d.sort()
        ans = []
        
        for i in range(K):
            if i != 0 and d[i] == d[i-1]: continue
            ans = ans + distances[d[i]]
            
        return ans
            
        
        
        

#     Version I sort:
#       Step1: write a comparator
#       Step2: write a distance calculator
#       Step3: sort
#       T: O(nlgn) S: O(n)
#     Version II: k sort
#       T: O(nlgk) S: O(k)
#     Version III: quick select
#       Step1: start, end
#       Step2: find mid and seperate to left part and right part
#       Step3: check if mid == k: return [:mid] or check mid again in right part or left part again
#       T: O(n) S: O(lgn) the height of stack
class Solution:
    def _getDistance(self, point):
        dis = math.sqrt(point[0]**2 + point[1]**2)
        return dis
    
    def _partition(self, start, end, points):
        target = points[start]
        while start < end:
            while start < end and self._getDistance(points[end]) >= self._getDistance(target):
                end -= 1
            points[start] = points[end]
            while start < end and self._getDistance(points[start]) <= self._getDistance(target):
                start += 1
            points[end] = points[start]
        points[start] = target
        return start
    
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        if not points:
            return []
        start = 0
        end = len(points) - 1
        while start <= end:
            mid = self._partition(start, end, points)
            if mid == K:
                return points[:K]
            elif mid > K:
                end = mid - 1
            else:
                start = mid + 1
        return points[:K]

class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        from math import sqrt
        points.sort(key=lambda x: sqrt(x[0]**2+x[1]**2))
        return points[:K]
from queue import PriorityQueue

class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        if points is None:
            return []
        
        q = PriorityQueue()
        
        for point in points:
            q.put((-self.distanceOrigin(point), point))
            if q.qsize() > K:
                q.get()
                
        res = []
        while not q.empty():
            res.append(q.get()[1])
            
        return res

    
    def distanceOrigin(self, a):
        return sqrt(a[0]**2 + a[1]**2)
class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        distances = []
        
        for idx, point in enumerate(points):
            eucliDist = point[0]**2 + point[1]**2
            heapq.heappush(distances, (eucliDist, point))
        
        distances = heapq.nsmallest(K, distances)
        result = []
        for res in distances:
            result.append(res[1])
            
        return result
            

class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        dic = collections.defaultdict(list)
        for p in points:
            d = p[0]**2 + p[1]**2
            dic[d].append(p)
        dic2 = list(dic.keys())
        print(dic2)
        heapq.heapify(dic2)
        res = []
        count = 0
        while(count<K):
            res.extend(dic[heapq.heappop(dic2)])
            count = len(res)
        return res
def compare(p1, p2):
    return (p1[0] ** 2 + p1[1] ** 2) - (p2[0] ** 2 + p2[1] ** 2)

def swap(arr, i, j):
    arr[i], arr[j] = arr[j], arr[i]

def partition(A, l, r):
    pivot = A[l]
    while l < r:
        while l < r and compare(A[r], pivot) >= 0:
            r -= 1
        
        A[l] = A[r]
        while l < r and compare(A[l], pivot) <= 0:
            l += 1
        
        A[r] = A[l]
    
    A[l] = pivot;
    return l
#     pivot = points[high]
    
#     i = low - 1
    
#     for j in range(low, high):
#         if compare(points[j], pivot) < 0:
#             i += 1
#             swap(points, i, j)
#     swap(points, i + 1, high)
#     return i + 1
    
def quick_select(points, k, low, high):
    while low < high:
        pivot = partition(points, low, high)
        if pivot == k:
            break
        elif pivot < k:
            # need more elements from the right
            low = pivot + 1
        else:
            # reduce element from the left
            high = pivot - 1


class Solution:
    def kClosest(self, points: List[List[int]], k: int) -> List[List[int]]:
        quick_select(points, k, 0, len(points) - 1)
        return points[:k]
class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        pointDistances = [
            [point, self.calculateDistanceToOrigin(point)]
            for point in points
        ]
        
        sortedPoints = sorted(pointDistances, key=lambda pd: pd[1])
        return [pd[0] for pd in sortedPoints[:K]]
                
    
    def calculateDistanceToOrigin(self, point):
        return point[0] * point[0] + point[1] * point[1] 
"""
quick select to find k

quick select method
arr, k (kth)
l = -1
r = len(arr)

def square(e):
    return e[0] * e[0] + e[1] * e[1]


def partition(arr, l, r):
    p = r
    firstHigh = l
    for i in range(r):
        if square(arr[i]) < square(arr[p]):
            arr[i], arr[firstHigh] = arr[firstHigh], arr[i]
    arr[firstHigh], arr[p] = arr[p], arr[firsthHigh]
    return firstHigh
            


def getKClosest(arr, k):
    while l + 1 < r:
        v = partition(arr, l, r)
        if v < k:
            l = v
        else:
            r = v

        retrun arr[:k]

"""
def square(e):
    return e[0] * e[0] + e[1] * e[1]


def partition(arr, l, r):
    p = r
    firstHigh = l
    for i in range(l, r):
        if square(arr[i]) < square(arr[p]):
            arr[i], arr[firstHigh] = arr[firstHigh], arr[i]
            firstHigh += 1
    arr[firstHigh], arr[p] = arr[p], arr[firstHigh]
    return firstHigh
            
class Solution:
    def kClosest(self, arr: List[List[int]], k: int) -> List[List[int]]:
        l = 0
        r = len(arr) - 1
        while l < r:
            v = partition(arr, l, r)
            if v < k:
                l = v + 1
            else:
                r = v - 1

        return arr[:k]
class Solution(object):
    def kClosest(self, points, K):
        dist = lambda i: points[i][0]**2 + points[i][1]**2

        def sort(i, j, K):
            # Partially sorts A[i:j+1] so the first K elements are
            # the smallest K elements.
            if i >= j: return

            # Put random element as A[i] - this is the pivot
            k = random.randint(i, j)
            points[i], points[k] = points[k], points[i]

            mid = partition(i, j)
            if K < mid - i + 1:
                sort(i, mid - 1, K)
            elif K > mid - i + 1:
                sort(mid + 1, j, K - (mid - i + 1))

        def partition(i, j):
            # Partition by pivot A[i], returning an index mid
            # such that A[i] <= A[mid] <= A[j] for i < mid < j.
            oi = i
            pivot = dist(i)
            i += 1

            while True:
                while i < j and dist(i) < pivot:
                    i += 1
                while i <= j and dist(j) >= pivot:
                    j -= 1
                if i >= j: break
                points[i], points[j] = points[j], points[i]

            points[oi], points[j] = points[j], points[oi]
            return j

        sort(0, len(points) - 1, K)
        return points[:K]
class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        distanceList = []
        for x,y in points:
            distanceList.append((x,y,self.distanceFromOrigin(x, y)))
            
        return [[p[0], p[1]] for p in heapq.nsmallest(K, distanceList, key=lambda x:x[2])]
    
    def distanceFromOrigin(self, x, y):
        return sqrt(pow(x - 0, 2) + pow(y - 0, 2))
from queue import PriorityQueue
import math

class Solution:
    def kClosest(self, points, K):
        pq = PriorityQueue()
        
        for point in points:
            d = self.getDistance(point)
            pq.put((-1 * d, point))
            
            if pq.qsize() > K:
                pq.get()
                
        out = []
        while pq.qsize() > 0:
            out.append(pq.get()[1])
            
        return out
        
    def getDistance(self, point):
        return math.sqrt(point[0] ** 2 + point[1] ** 2)
class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        if not points or len(points) < 1:
            return []
        distances = []
        
        for point in points:
            dist = point[0]*point[0] + point[1]*point[1]
            if len(distances) < K:
                heapq.heappush(distances, (-dist, point))
            else:
                if -dist > distances[0][0]:
                    heapq.heappushpop(distances, (-dist, point))
        
        return [point for _, point in distances]
class Node:
    def __init__(self, c, sum):
        self.sum = sum
        self.c = c

class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        if not points:
            return []
        heap = []
        ans = []
        for i in points:
            heap.append(Node(i, i[0]**2 + i[1]**2))
        n = len(heap)
        def heapify(arr, i, n):
            smallest = i
            left = i*2 + 1
            right = i*2 + 2
            if left < n and arr[i].sum > arr[left].sum:
                smallest = left
            if right < n and arr[right].sum < arr[smallest].sum:
                smallest = right
            if smallest != i:
                arr[smallest], arr[i] = arr[i], arr[smallest]
                heapify(arr, smallest, n)
        for i in range((n-1)//2,-1,-1):
            heapify(heap, i, n)
        while K > 0 and len(ans) < len(points):
            h = heap[0]
            ans.append(h.c)
            heap[0].sum = sys.maxsize
            heapify(heap, 0, len(heap))
            K -= 1
        return ans
## Time Complexity: O(NlogK)
## Space Complexity: O(K)

from heapq import *

## Approach 1 - Succinct code
# class Solution:
#     def distance(self, point):
#         # ignoring sqrt to calculate the distance
#         return point[0] ** 2 + point[1] ** 2
#
#     def kClosest(self, points, K):
#         '''
#         :type points: List[List[int]]
#         :type K: int
#         :rtype: List[List[int]]
#         '''
#         max_heap = []
#         for point in points:
#             heappush(max_heap, (-self.distance(point), point))
#             if len(max_heap) > K:
#                 heappop(max_heap)
#
#         res = []
#         while max_heap:
#             res.append(heappop(max_heap)[1])
#         return res


## Approach 2 - Define a Point class
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def distance_from_origin(self):
        # for comparison we ignore the sqrt part
        return self.x ** 2 + self.y ** 2

    def __lt__(self, other):
        return self.distance_from_origin() < other.distance_from_origin()

    def __eq__(self, other):
        return self.distance_from_origin() == other.distance_from_origin()


class Solution:
    def kClosest(self, points, K):
        '''
        :type points: List[List[int]]
        :type K: int
        :rtype: List[List[int]]
        '''
        max_heap = []
        for p in points:
            point = Point(p[0], p[1])
            dist = point.distance_from_origin()
            heappush(max_heap, (-dist, point))
            if len(max_heap) > K:
                heappop(max_heap)

        res = []
        while max_heap:
            pt = heappop(max_heap)[1]
            res.append([pt.x, pt.y])
        return res


from queue import PriorityQueue
class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        pq = PriorityQueue()
        for i in range(len(points)):
            distance = points[i][0]**2+points[i][1]**2
            if i < K:
                pq.put((-distance, points[i]))
            else:
                pq.put((-distance, points[i]))
                pq.get()
        ans = []
        while pq.qsize() > 0:
            ans.append(pq.get()[1])
        return ans
class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        closest = lambda x: points[x][0]**2 + points[x][1]**2
        
        
            
        def partition(i, j):
            oi = i
            pivot = closest(i)
            i += 1
            
            while True:
                while i < j and closest(i) < pivot:
                    i += 1
                while i <= j and closest(j) >= pivot:
                    j -= 1
                if i >= j:
                    break
                points[i], points[j] = points[j], points[i]
            points[oi], points[j] = points[j], points[oi]
            return j
        
        def kclosest(i, j, k):
            if i >= j:
                return
            
            mid = partition(i, j)
            if k > mid - i + 1:
                return kclosest(mid+1, j, k-(mid - i + 1))
            elif k < mid - i + 1:
                return kclosest(i, mid-1, k)
        kclosest(0, len(points)-1, K)
        return points[:K]
class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        
        def compare(p1, p2):
            return (p1[0]**2 + p1[1]**2)-(p2[0]**2 + p2[1]**2)
        
        def quickSelect(l,r):
            pivot = points[r]
            j=l
            for i in range(l,r):
                if compare(points[i], pivot) <= 0:
                    points[i], points[j] = points[j], points[i]
                    j+=1
            
            points[r], points[j]  =points[j], points[r]
            return j
                
            
            
            
            
        l,r = 0, len(points)-1
        while l<=r:
            pivot = quickSelect(l,r)
            if pivot == K:
                break
            elif pivot < K:
                l = pivot+1
            else:
                r = pivot-1
        return points[:K]
        
        
        
        

import heapq

class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        def dist(p):
            return p[0]**2 + p[1]**2
        
        def qs(l, r, k):
            if l >= r:
                return
            pivotd = dist(points[r])
            ll = l
            for rr in range(l, r):
                if dist(points[rr]) < pivotd:
                    points[ll], points[rr] = points[rr], points[ll]
                    ll += 1
            ppos = ll
            points[ppos], points[r] = points[r], points[ppos]
            if ppos == k:
                return
            if ppos < k:
                qs(ppos + 1, r, k)
            qs(l, ppos - 1, k)
        
        qs(0, len(points) - 1, K - 1)
        return points[:K]
class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        dictionary = {}
        for i in points:
            dictionary[(i[0],i[1])] = math.sqrt( ((i[0]-0)**2)+((i[1]-0)**2) )
        heap = []
        heapq.heapify(heap)
        for i in list(dictionary.keys()):
            heapq.heappush(heap,(dictionary[i],i))
        result = []

        for i in range(K):
            result.append(heapq.heappop(heap)[1])
        return result

class Solution:
    """
    Sort everything and then get the first k items.
    T: O(NlogN).
    """
    def kClosest(self, points: List[List[int]], k: int) -> List[List[int]]:
        return sorted(points, key = lambda p: p[0]**2 + p[1]**2)[:k]    
    
    """
    Use a max-heap to store the k smallest items.
    T: O(N log K).
    """
    def kClosest(self, points: List[List[int]], k: int) -> List[List[int]]:
        heap = []
        
        for x,y in points:
            # We put the -ve sign becase we want this to be a max heap.
            key = -(x**2 + y**2)
            heapq.heappush(heap, (key, x, y))
            if len(heap) > k:
                heapq.heappop(heap)
        
        return [[x,y] for distance, x, y in heap]
    
    """
    Use QuickSelect.
    T: O(N) in the average case; O(N^2) in the worst case.
    """
    def kClosest(self, points: List[List[int]], k: int) -> List[List[int]]:
        self.quickSelect(points, k - 1, 0, len(points) - 1)
        return points[:k]
    
    def quickSelect(self, points, k, p, q):
        partitionIndex  = self.partition(points, p, q)
        if partitionIndex == k:
            return partitionIndex
        elif partitionIndex > k:
            return self.quickSelect(points, k, p, partitionIndex  - 1)
        else:
            return self.quickSelect(points, k, partitionIndex + 1, q)
                
    def getDistance(self, p):
        x,y = p 
        return x**2 + y**2
    
    # p <= k <= boundary: items that are smaller than the pivot.
    # boundary + 1 <= < q: items that are bigger than the pivot.
    def partition(self, points, p, q):
        distancePivot = self.getDistance(points[q])
        boundary = p - 1
        
        for j in range(p, q):
            if self.getDistance(points[j]) < distancePivot:
                boundary += 1
                points[boundary], points[j] = points[j], points[boundary]                
        
        # Insert the pivot in the correct position.
        boundary += 1 
        points[boundary], points[q] = points[q], points[boundary]
        
        return boundary
import math
class Solution:
    def __init__(self):
        self.dist_coord = dict()
        self.dist_heap = [None]
        self.final_result = list()
        
    def _calc_dist(self, x, y):
        return math.sqrt(x*x + y*y)

    def _calc_dist_coord(self):
        for point in self.coords:
            dist = self._calc_dist(point[0], point[1])
            if self.dist_coord.get(dist) is None:
                self.dist_coord[dist] = [point]
                self.dist_heap.append(dist)
            else:
                self.dist_coord[dist].append(point)
        
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        self.coords = points
        self._calc_dist_coord()
        for i in reversed(list(range(1, len(self.dist_heap)//2 + 1))):
            self.shift_down(self.dist_heap, i)
        
        while K > 0 and len(self.dist_heap) > 1:
            # print(self.dist_coord, self.dist_heap)
            for point in self.dist_coord[self.dist_heap[1]]:
                self.final_result.append(point)
                K -= 1
            self.del_elem_from_heap(self.dist_heap)
            # K -= 1
        return self.final_result
            
    def shift_up(self, h_list):
        i = len(h_list) - 1
        while int(i/2) > 0:
            if h_list[i] > h_list[int(i/2)]:
                h_list[i], h_list[int(i/2)] = h_list[int(i/2)], h_list[i]
            i = int(i/2)

    def shift_down(self, h_list, start):
        i = start
        l = 2* i
        r = 2* i + 1
        if l < len(h_list) and h_list[i] > h_list[l]:
            i = l

        if r < len(h_list) and h_list[i] > h_list[r]:
            i = r

        if i != start:
            h_list[start], h_list[i] = h_list[i], h_list[start]
            self.shift_down(h_list, i)

    def add_elem_to_heap(self, h_list, val):
        h_list.append(val)
        self.shift_up(h_list)

    def del_elem_from_heap(self, h_list):
        h_list[1] = h_list[-1]
        h_list.pop()
        self.shift_down(h_list, 1)
        

class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        n1 = len(points)
        newList = [0] * n1
        for i in range(n1):
            pointX = points[i][0]
            pointY = points[i][1]
            d2 = pointX * pointX + pointY * pointY
            newList[i] = d2
        newList.sort()
        res = newList[K - 1]
        i = 0
        t = 0
        ans = []
        while i < n1 and t < K:
            if res >= points[i][0] * points[i][0] + points[i][1] * points[i][1]:
                ans = ans + [points[i]]
                t += 1
            i += 1
        return ans
class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        def helper(p):
            return p[0]**2 + p[1]**2
        return sorted(points, key=helper)[:K]
from queue import PriorityQueue



class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
           
        def dst(point):
            return (point[0] ** 2 + point[1] ** 2) ** (1/2)
        
        n = 0
        q = PriorityQueue()
        for point in points:
            point = [-dst(point)] + point[:]  
            if n < K:
                q.put(point)
            
            else:
                least = q.get()
    
                if -point[0] < -least[0]:
                    q.put(point)
                else:
                    q.put(least)   
            n += 1
        
        closest = []
        while not q.empty():
            closest.append(q.get()[1:])
        
        return closest
        
        
            

class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        
        
        li = []
        
        class Point:
            def __init__(self, x, y):
                self.x = x
                self.y = y
            def distance_square(self):
                return self.x * self.x + self.y * self.y
        
            def __lt__(self, other):
                return self.distance_square() < other.distance_square()
            
        
        
        
        for p in points:
            x = p[0]
            y = p[1]
            pt = Point(x, y)
            heapq.heappush(li,pt)
        
        result_p = [heapq.heappop(li) for i in range(K)]
        result = [ [p.x, p.y] for p in result_p ]
        return result

class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        def get_distance(point):
            return point[0] ** 2 + point[1] ** 2
        
        def helper(i, j, K):
            if i >= j:
                return
            pivot_index = random.randint(i, j)
            points[j], points[pivot_index] = points[pivot_index], points[j]
            mid = partition(i, j)
            if mid == K - 1:
                return
            elif mid < K - 1:
                helper(mid + 1, j, K)
            else:
                helper(i, mid - 1, K)
        
        def partition(i, j):
            left, right = i, j
            pivot_distance = get_distance(points[j])
            for k in range(left, right):
                if get_distance(points[k]) < pivot_distance:
                    points[i], points[k] = points[k], points[i]
                    i += 1
            points[i], points[j] = points[j], points[i]
            return i
        
        helper(0, len(points) - 1, K)
        return points[:K]
        
        
        
#         def get_distance(point):
#             return point[0] ** 2 + point[1] ** 2
        
#         def helper(i, j, K):
#             if i >= j:
#                 return
#             pivot_index = random.randint(i, j)
#             points[j], points[pivot_index] = points[pivot_index], points[j]
#             mid = partition(i, j)
#             if mid - i + 1 == K:
#                 return
#             elif mid - i + 1 < K:
#                 helper(mid + 1, j, K - (mid - i + 1))
#             else:
#                 helper(i, mid - 1, K)
        
#         def partition(i, j):
#             left, right = i, j
#             pivot_distance = get_distance(points[j])
#             for k in range(left, right):
#                 if get_distance(points[k]) < pivot_distance:
#                     points[i], points[k] = points[k], points[i]
#                     i += 1
#             points[i], points[j] = points[j], points[i]
#             return i
        
#         helper(0, len(points) - 1, K)
#         return points[:K]

from queue import PriorityQueue
class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        pq = PriorityQueue()
        # furthest = 0
        for p in points:
            distance = -1 * ((p[0] * p[0]) + (p[1] * p[1]))
            # print(distance, p)
            if pq.qsize() < K:
                pq.put((distance, p))
                # furthest = min(furthest, distance)
            else:
                # if distance < furthest:
                pq.put((distance, p))
                pq.get()
        
        ans = []
        while pq.qsize():
            (distance, p) = pq.get()
            ans.append(p)
        
        return ans
class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        
        
        def norm(point):
            return (point[0]**2+point[1]**2)
        
        dis=[]
        for point in points:
            dis.append(norm(point))
      
        dis, points=list(zip(*sorted(zip(dis,points))))
        
        res=[]
        for i in range(K):
            res.append(points[i])
        return res
            

import heapq
def getDistance(point):
        return point[0]**2 + point[1]**2
    
class Solution:
    def swap(self, i, j):
        tmp = self.points[i]
        tmp_dist = self.dist[i]
        
        self.points[i] = self.points[j]
        self.points[j] = tmp
        
        self.dist[i] = self.dist[j]
        self.dist[j] = tmp_dist
        
    def partition(self, start, end, pivot):
        self.swap(0, pivot)
        
        left = 1
        right = end
        while left <= right:
            if self.dist[left] > self.dist[0]:
                self.swap(left, right)
                right -= 1
            else:
                left += 1
                
        self.swap(0, right)
        
        return right
    
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        n = len(points)
        if n <= K: return points
        
        self.points = points
        self.dist = []
        
        for i in range(n):
            self.dist.append(getDistance(points[i]))
        # print(self.dist)
        start = 0
        end = n - 1
        pivot = self.partition(0, n - 1, K)
        # print(pivot, self.dist)
        while pivot != K:
            if pivot < K:
                start = pivot + 1
            else:
                end =  pivot - 1
            pivot = self.partition(start, end, start)
        
            # print(pivot)
        
        res = []
        
        # print(self.dist)
        return self.points[0:K]
                
            
            

import heapq
class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        heap = []
        for (x,y) in points:
            dist = -(x**2 + y**2)
            if len(heap) == K:
                heapq.heappushpop(heap, (dist, x, y))
            else:
                heapq.heappush(heap, (dist, x, y))
        return [(x,y) for (dist, x, y) in heap]
        
        
        
# import heapq

# class Solution:
#     def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        
#         heap = []
        
#         for (x, y) in points:
#             dist = -(x*x + y*y)
#             if len(heap) == K:
#                 heapq.heappushpop(heap, (dist, x, y))
#             else:
#                 heapq.heappush(heap, (dist, x, y))
        
#         return [(x,y) for (dist,x, y) in heap]

class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        d={}
        ans=[]
        points = sorted(points, key= lambda x:((x[0]**2)+(x[1]**2)))
        for i in range(K):
            ans.append(points[i])
        return ans

import math  

class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        a = []
        for i in range(len(points)):
            a.append([math.sqrt(pow(points[i][0], 2) + pow(points[i][1], 2)),                  [points[i][0], points[i][1]]])
        a.sort(key=lambda x:x[0])
        print(a)
        l = []
        for j in range(K):
            l.append(a[j][1])
        return l
            
            
            

import heapq
def getDistance(point):
        return point[0]**2 + point[1]**2
    
class Solution:
    def swap(self, i, j):
        tmp = self.points[i]
        tmp_dist = self.dist[i]
        
        self.points[i] = self.points[j]
        self.points[j] = tmp
        
        self.dist[i] = self.dist[j]
        self.dist[j] = tmp_dist
        
    def partition(self, start, end, pivot):
        self.swap(0, pivot)
        
        left = 1
        right = end
        while left <= right:
            if self.dist[left] > self.dist[0]:
                self.swap(left, right)
                right -= 1
            else:
                left += 1
                
        self.swap(0, right)
        
        return right
    
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        n = len(points)
        if n <= K: return points
        
        self.points = points
        self.dist = []
        
        for i in range(n):
            self.dist.append(getDistance(points[i]))
        # print(self.dist)
        start = 0
        end = n - 1
        pivot = self.partition(0, n - 1, K)
        # print(pivot, self.dist)
        while pivot != (K-1):
            if pivot < (K-1):
                start = pivot + 1
            else:
                end =  pivot - 1
            pivot = self.partition(start, end, start)
        
            # print(pivot)
        
        res = []
        
        # print(self.dist)
        return self.points[0:K]
                
            
            

import heapq
class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        def distance(point):
            return (point[0] ** 2 + point[1] ** 2)
        
        return sorted(points, key=distance)[:K]
        
#         h = []
#         for i, p in enumerate(points):
#             if len(h) == K:
#                 heapq.heappushpop(h, (distance(p), i, p))
#             else:
#                 heapq.heappush(h, (distance(p), i, p))
                
#         out = []
#         for tup in h:
#             out.append(tup[2])
            
#         return out

class DistanceSort(List[int]):
    
    def __lt__(lhs, rhs):
        ldiff = lhs[0]*lhs[0] + lhs[1]*lhs[1]
        rdiff = rhs[0]*rhs[0] + rhs[1]*rhs[1]
        return ldiff < rdiff
    
class Solution:
    
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        return sorted(points, key=DistanceSort)[:K]
        
"""
"""
class Solution:
    def kClosest(self, points: List[List[int]], k: int) -> List[List[int]]:
        from queue import PriorityQueue as pq
        distance_points = [(-1 * ((x**2 + y**2) ** .5), [x,y]) for x,y in points]
        q = pq()
        for item in distance_points:
            if q.qsize() < k:
                q.put(item) 
            else:
                distance, _ = item
                popped = q.get()
                popped_distance, _ = popped
                if distance > popped_distance:
                    q.put(item)
                else:
                    q.put(popped)
        return [q.get()[1] for _ in range(k)]
                
            

import heapq
def getDistance(point):
        return point[0]**2 + point[1]**2
    
class Solution:
    def swap(self, i, j):
        tmp = self.points_idx[i]
        tmp_dist = self.dist[i]
        
        self.points_idx[i] = self.points_idx[j]
        self.points_idx[j] = tmp
        
        self.dist[i] = self.dist[j]
        self.dist[j] = tmp_dist
        
    def partition(self, start, end, pivot):
        self.swap(0, pivot)
        
        left = 1
        right = end
        while left <= right:
            if self.dist[left] > self.dist[0]:
                self.swap(left, right)
                right -= 1
            else:
                left += 1
                
        self.swap(0, right)
        
        return right
    
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        n = len(points)
        if n <= K: return points
        
        self.points_idx = [i for i in range(n)]
        self.dist = []
        
        for i in range(n):
            self.dist.append(getDistance(points[i]))
        # print(self.dist)
        start = 0
        end = n - 1
        pivot = self.partition(0, n - 1, K)
        # print(pivot, self.dist)
        while pivot != (K-1):
            if pivot < (K-1):
                start = pivot + 1
            else:
                end =  pivot - 1
            pivot = self.partition(start, end, start)
        
            # print(pivot)
        
        res = [points[idx] for idx in self.points_idx[0:K]]
        

        return res
                
            
            

from heapq import heappop, heappush, heapify
class Solution:
    def distFromOrigin(self,x,y):
        return sqrt(x**2 + y**2)
    
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        heap = []
        for point in points:
            dist = self.distFromOrigin(point[0],point[1])
            if len(heap) == K:
                if dist < -heap[0][0]:
                    heappop(heap)
                    heappush(heap, (-dist, point))
            else:
                heappush(heap, (-dist, point))
        
        output = []
        for item in heap:
            output.append(item[1])
        return output

class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        if not points or not K:
            return []
        
        distances = []
        disMap = defaultdict(list)
        maxKDis = float('-inf')
        for point in points:
            dis = (point[0] ** 2 + point[1] ** 2) ** 0.5
            disMap[dis].append(point)
            if len(distances) < K:
                distances.append(dis)
                maxKDis = max(maxKDis, dis)
            elif len(distances) >= K and dis < maxKDis:
                distances.sort()
                distances.pop()
                distances.append(dis)
                maxKDis = max(distances[-2:])

        res = []
        for dis in distances:
            res.extend(disMap[dis])
            
        return res[:K]
class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        point_distance = dict()
        distance_list = []
        
        for x, y in points:
            distance = sqrt(x**2 + y**2)
            heappush(distance_list, distance)
            if distance in list(point_distance.keys()):
                point_distance[distance].append([x,y])
            else:
                point_distance[distance] = [[x,y]]
            
        result = []
        while K > 0:
            tmp = point_distance[heappop(distance_list)]
            K-= len(tmp)
            result = result + tmp
            
        return result

class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        heap = []
        for x,y in points:
            dis2 = x**2 + y**2
            if len(heap)<K:
                heapq.heappush(heap, (-dis2, (x,y)))
            else:
                if -dis2>heap[0][0]:
                    heapq.heappop(heap)
                    heapq.heappush(heap,(-dis2,(x,y)))
        ans = []
        for i in range(K):
            x,y = heapq.heappop(heap)[1]
            ans = [[x, y]] + ans
        return ans
from queue import PriorityQueue
import math
class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        queue = PriorityQueue()
        
        for point in points:
            queue.put((math.sqrt((point[0])**2 + (point[1])**2), point))
            
        ans = []
        
        for i in range(K):
            ans.append(queue.get()[1])
        return ans
class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        point_distance = dict()
        distance_list = []
        
        for x, y in points:
            distance = -sqrt(x**2 + y**2)
            if len(distance_list) < K:
                heappush(distance_list, distance)
                if distance in list(point_distance.keys()):
                        point_distance[distance].append([x,y])
                else:
                    point_distance[distance] = [[x,y]]
            else:
                if distance_list[0] < distance:
                    to_remove = distance_list[0]
                    if len(point_distance[to_remove]) == 1:
                        del point_distance[to_remove]
                    else:
                        point_distance[to_remove].pop()
                    heappop(distance_list)
                    heappush(distance_list, distance)
                    if distance in list(point_distance.keys()):
                        point_distance[distance].append([x,y])
                    else:
                        point_distance[distance] = [[x,y]]

        result = []
        while K > 0:
            tmp = point_distance[heappop(distance_list)]
            K-= len(tmp)
            result = result + tmp
            
        return result

from heapq import heappop,heappush,heapify
class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        l1=[]
        l2=[]
        heapify(l1)
        for i in range(0,len(points)):
            heappush(l1, (-1*(points[i][0]*points[i][0] + points[i][1]*points[i][1]),i))
            if len(l1) > K:
                heappop(l1)
        while(len(l1) > 0):
            i = heappop(l1)
            l2.append(points[i[1]])
        return l2
                    
        

import heapq
class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:

        heap = []

        for (x, y) in points:
            dist = -(x*x + y*y)
            if len(heap) == K:
                heapq.heappushpop(heap, (dist, x, y))
            else:
                heapq.heappush(heap, (dist, x, y))

        return [(x,y) for (dist,x, y) in heap]
class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        def distance(point):
            return (point[0]**2 + point[1]**2)**(0.5)

        def partition(left, right, pivot_index):
            pivot = points[pivot_index]
            # move the pivot to the position of right first
            points[pivot_index], points[right] = points[right], points[pivot_index]

            store_index = left
            for i in range(left, right):
                if distance(points[i]) < distance(pivot):
                    points[store_index], points[i] = points[i], points[store_index]
                    store_index += 1
            
            # move the pivot back to its correct position:
            # such that all elements on the left are smaller, and elements on the right have larger distance to origin.
            points[store_index], points[right] = points[right], points[store_index]
            
            return store_index
        
        left, right = 0, len(points)-1
        while True:
            pivot_index = random.randint(left, right)
            M = partition(left, right, pivot_index)
            if M == K-1:
                break
            elif M < K-1:
                left = M + 1
            else:
                right = M - 1
                
        return points[:K]
import heapq

class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        
        heap = []
        
        for x, y in points:
            dist = -(x*x + y*y)
            
            if len(heap) == K:
                heapq.heappushpop(heap, (dist, x, y))
            else:
                heapq.heappush(heap, (dist, x, y))
            
        return [[x,y] for dist, x, y in heapq.nlargest(K, heap)]

class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        dist = lambda i: points[i][0]**2 + points[i][1]**2
        def quicksort(points,K): # points: list of index
            print(points, K)
            if not points:
                return
            # pivot = points[random.randint(0,len(points)-1)]

            pivot = points[-1]
            l,r,m=[],[],[]
            for i in points:
                if dist(i) == dist(pivot):
                    m.append(i)
                elif dist(i) < dist(pivot):
                    l.append(i)
                else:
                    r.append(i)
            if len(l) >= K:
                return quicksort(l, K)
            elif len(l) < K <= len(l)+len(m):
                return l+m[:K-len(l)]
            else:
                return l+m+quicksort(r, K-len(l)-len(m))
        return [points[i] for i in quicksort([i for i in range(len(points))], K)]
class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        self.sort(0, len(points) - 1, points, K)
        return points[:K]

    def getDistance(self, a):
        return a[0] ** 2 + a[1] ** 2

    def sort(self, l, r, points, K):
        if l >= r:
            return 
        mid = self.partition(l, r, points)
        if (mid - l + 1) < K:
            self.sort(mid + 1, r, points, K - (mid - l + 1))
        else:
            self.sort(l, mid - 1, points, K)
    
    def partition(self, i, j, points):
        pivot = self.getDistance(points[i])
        l = i
        r = j
        while True:
            while l <= r and self.getDistance(points[l]) <= pivot:
                l += 1
            while l <= r and self.getDistance(points[r]) > pivot:
                r -= 1
            if l >= r:
                break
            points[l], points[r] = points[r], points[l]
        points[i], points[r] = points[r], points[i]
        return r

class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        
        # edge case
        if not points:
            return []
        
        def distance(x, y):
            
            # Euclidean distance
            return ((x - 0)**2 + (y - 0)**2)**0.5
        
        def binsea_insert(dist, point):
            
            # if empty, just insert and return
            if not nearest:
                nearest.append((dist, point))
                return
            
            left, right = 0, len(nearest) -1
            
            while left <= right:
                
                # unpack tuple 
                d, coor = nearest[right]
                if dist >= d:
                    nearest.insert(right + 1, (dist, point))
                    return
                    
                # unpack tuple 
                d, coor = nearest[left]
                    
                if dist <= d:
                    nearest.insert(left, (dist, point))
                    return
                    
                    
                
                mid = left + (right - left) // 2
                
                # unpack tuple 
                d, coor = nearest[mid]
                
                if dist < d:
                    right = mid - 1
                
                else:
                    left = mid + 1
            

        # sorted list of tuples
        nearest = []
        
        for point in points:
            x, y = point
            binsea_insert(distance(x, y), point)
            
        print(nearest)            
        results = []
        for k in range(K):
            if k < len(nearest):
                d, point = nearest[k]
                results.append(point)
                
        return results
            
            
            

class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        heap = [(float('-Inf'), 0, 0)]*K
        for x, y in points:
            heapq.heappushpop(heap, (-x**2-y**2, x, y))
        res = []
        while heap:
            val, row, col = heapq.heappop(heap)
            res.append([row, col])
        return res
from collections import defaultdict

class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        dic = defaultdict(list)
        for i in range(len(points)):
            distance = points[i][0] ** 2 + points[i][1] ** 2
            dic[distance].append(points[i])

        dis_list = sorted(dic.keys())
        ans = []
        
        for i in dis_list:
            ans.extend(dic[i])
        
        return ans[:K]
class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        points = [[i ** 2 + j ** 2, i, j] for i, j in points]
        
        def quickSelect(l, r):
            p = l
            for i in range(l, r):
                if points[i][0] < points[r][0]:
                    points[p], points[i] = points[i], points[p]
                    p += 1
            points[r], points[p] = points[p], points[r]
            if p == K - 1: return
            elif p > K - 1: quickSelect(l, p - 1)
            else: quickSelect(p + 1, r)
        quickSelect(0, len(points) - 1)
        return [[i, j] for _, i, j in points[:K]]

class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        
        for i in range(len(points)):
            points[i].insert(0,(points[i][0]**2+points[i][1]**2)**0.5)
        points.sort()
        ans=[]
        for i in range(K):
            ans.append(points[i][1:])
            
        return(ans)
import math
class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        points.sort(key = lambda p : p[0]**2+p[1]**2)
        return points[:K]
class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        heap = []
        
        for x,y in points:
            if len(heap)<K:
                heapq.heappush(heap,[-(x*x+y*y),[x,y]])
            else:
                heapq.heappushpop(heap,[-(x*x+y*y),[x,y]])
        return [pair for value, pair in heap]
import math
from queue import PriorityQueue
class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        # dist_map = dict()
        # for point in points:
        #     dist_map[tuple(point)] = math.sqrt((0-abs(point[0]))**2 + (0-abs(point[1]))**2)
        # closest_points = sorted(dist_map.items(), key=lambda x: x[1])
        # return [i[0] for i in list(closest_points)[:K]]
        pq = PriorityQueue()
        for point in points:
            dist = math.sqrt((0-abs(point[0]))**2 + (0-abs(point[1]))**2)
            pq.put((dist, tuple(point)))
        i = 0
        res = []
        while pq and i < K:
            res.append(pq.get()[1])
            i += 1
        return res
class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        
        # points = sorted(points, key = lambda x: x[0]**2 + x[1]**2)
        # return points [:K]
        shuffle(points)
        
        distance_dict = dict()
        
        def distance(xy):
            xy = tuple(xy)
            if xy in distance_dict:
                return distance_dict[xy]
            ans = xy[0]**2 + xy[1]**2
            distance_dict[xy] = ans
            return ans
    
        def _quicksort(points, left, right):
                        
            if left>right:
                return
            
            pivot = right
            lower = left
            for i in range(left, right):
                if distance(points[i]) <= distance(points[pivot]):
                    points[lower], points[i] = points[i], points[lower]
                    lower = lower + 1
            
            pivot = lower
            points[pivot], points[right] = points[right], points[pivot]
                        
            if (K-1) <= pivot:
                _quicksort(points, left, pivot-1)
            else:
                _quicksort(points, left, pivot-1)
                _quicksort(points, pivot+1, right)
            
            return
        
        _quicksort(points, 0, len(points)-1)

        return points[:K]
import heapq
class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        max_heap =[]
        for i in range(len(points)):
            sqrt = (points[i][0]**2 + points[i][1]**2)
            heapq.heappush(max_heap,[sqrt,points[i]])
    
        res = []
        for i in range(K):
            temp = heapq.heappop(max_heap)
            res.append(temp[1])
        return res
            
            

class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        heap = []
        output = []
        
        for i in range(len(points)): 
            distance = (points[i][0]**2) + (points[i][1]**2)
            heapq.heappush(heap, (distance, points[i]))
        
        for i in range(K): 
            add = (heapq.heappop(heap))
            output.append(add[1])
        
        return output
        
                  

import random
import math

class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        
        euclidean = lambda x: math.sqrt(x[0]**2 + x[1]**2)
        
        def topK(arr, l, r, k):
            if l >= r:
                return
            
            choice = random.randint(l, r)
            arr[l], arr[choice] = arr[choice], arr[l]
            
            tl, tr = l+1, r
            while tl <= tr:
                if euclidean(arr[tl]) < euclidean(arr[l]):
                    tl += 1
                    continue
                if euclidean(arr[tl]) >= euclidean(arr[l]):
                    arr[tl], arr[tr] = arr[tr], arr[tl]
                    tr -= 1
                    continue
            arr[l], arr[tr] = arr[tr], arr[l]
            
            partition = tr
            
            if (partition-l+1) <= k:
                topK(arr, partition+1, r, k-(partition-l+1))
            else:
                topK(arr, l, partition-1, k)
                
        
        topK(points, 0, len(points)-1, K)
        return points[:K]
class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        
        # points = sorted(points, key = lambda x: x[0]**2 + x[1]**2)
        # return points [:K]
        
        distance_dict = dict()
        
        def distance(xy):
            xy = tuple(xy)
            if xy in distance_dict:
                return distance_dict[xy]
            ans = xy[0]**2 + xy[1]**2
            distance_dict[xy] = ans
            return ans
    
        def _quicksort(points, left, right):
                        
            if left>right:
                return
            
            pivot = right
            lower = left
            for i in range(left, right):
                if distance(points[i]) <= distance(points[pivot]):
                    points[lower], points[i] = points[i], points[lower]
                    lower = lower + 1
            
            pivot = lower
            points[pivot], points[right] = points[right], points[pivot]
                        
            if (K-1) <= pivot:
                _quicksort(points, left, pivot-1)
            else:
                _quicksort(points, left, pivot-1)
                _quicksort(points, pivot+1, right)
            
            return
        
        _quicksort(points, 0, len(points)-1)

        return points[:K]
class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        distant = [[point,point[0]**2+point[1]**2] for point in points]
        distant.sort(key = itemgetter(1))
        res = [distant[i][0] for i in range(K)]
        return res

import random
import math

class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        
        euclidean = lambda x: math.sqrt(x[0]**2 + x[1]**2)
        
        def topK(arr, l, r, k):
            if l >= r:
                return
            
            choice = random.randint(l, r)
            arr[l], arr[choice] = arr[choice], arr[l]
            
            tl, tr = l+1, r
            # print(tl, tr, arr[l])
            # print(arr)
            while tl <= tr:
                if euclidean(arr[tl]) < euclidean(arr[l]):
                    tl += 1
                    continue
                if euclidean(arr[tl]) >= euclidean(arr[l]):
                    # print("Swapping", tl, tr)
                    arr[tl], arr[tr] = arr[tr], arr[tl]
                    tr -= 1
                    continue
            arr[l], arr[tr] = arr[tr], arr[l]
            # print("Swapping", l, tr)
            # print(arr)
            
            partition = tr
            
            if (partition-l+1) <= k:
                topK(arr, partition+1, r, k-(partition-l+1))
            else:
                topK(arr, l, partition-1, k)
                
                        
            
                    
                
        
        topK(points, 0, len(points)-1, K)
        return points[:K]
class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        self.sort(0, len(points) - 1, points, K)
        return points[:K]

    def getDistance(self, a):
        # print(a)
        return a[0] ** 2 + a[1] ** 2

    def sort(self, l, r, points, K):
        if l >= r:
            return 
        mid = self.partition(l, r, points)
        if (mid - l + 1) < K:
            self.sort(mid + 1, r, points, K - (mid - l + 1))
        else:
            self.sort(l, mid - 1, points, K)
    
    def partition(self, i, j, points):
        pivot = self.getDistance(points[i])
        l = i
        r = j
        while True:
            while l <= r and self.getDistance(points[l]) <= pivot:
                l += 1
            while l <= r and self.getDistance(points[r]) > pivot:
                r -= 1
            if l >= r:
                break
            points[l], points[r] = points[r], points[l]
        points[i], points[r] = points[r], points[i]
        return r

class Solution:
    def calculateEuclidean(self, point):
        return (point[0]) ** 2 + (point[1]) ** 2
    
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        heap = []
        
        
        for point in points:
            heapq.heappush(heap, (-self.calculateEuclidean(point), point))
            
            if len(heap) > K:
                heapq.heappop(heap)
        
        return [point[1] for point in heap]
import heapq
class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        arr = [(-1 * (p[0] ** 2 + p[1] ** 2), p) for p in points]
        heap = arr[:K]
        heapq.heapify(heap)
        for p in arr[K:]:
            heapq.heappushpop(heap, p)
        return [coord for distance, coord in heap]
class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        dist = [(x**2 + y**2, [x,y]) for x, y in points]
        
        self.partition(dist, K, 0, len(dist) - 1)
        
        return [elem[1] for elem in dist[:min(K, len(dist))]]
    
    """
    [035476]
    pivot = 6
    
    
    
    """
    def partition(self, dist, K, start, end):
        if start == end:
            return
        mid = (start + end)//2
        pivot = dist[mid][0]
        left, right = start, end
    
        while left <= right:
            while left <= right and dist[left][0] < pivot:
                left += 1
            while left <= right and dist[right][0] > pivot:
                right -= 1
            if left <= right:    
                dist[left], dist[right] = dist[right], dist[left]
                left += 1
                right -= 1
            
        if right - start >= K :
            self.partition(dist, K, start, right)
        if left - start < K:
            self.partition(dist, K - (left - start), left, end)
        
        
        
class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        arr = self.sortPoints(points)
        return arr[:K]
    
    def sortPoints(self, points):
        if len(points) <= 1:
            return points
        mid = len(points)//2
        left = self.sortPoints(points[:mid])
        right = self.sortPoints(points[mid:])
        return self.merge(left,right)
    
    def merge(self, left,right):
        arr, i, j = [], 0, 0
        while i < len(left) and j < len(right):
            xLeft, yLeft = left[i][0], left[i][1]
            xRight, yRight = right[j][0], right[j][1]
            leftDis = xLeft*xLeft + yLeft*yLeft
            rightDis = xRight*xRight + yRight*yRight

            if leftDis < rightDis:
                arr.append(left[i])
                i += 1
            elif rightDis < leftDis:
                arr.append(right[j])
                j += 1
            else:
                arr.append(left[i])
                arr.append(right[j])
                i, j = i+1, j+1
        while i < len(left):
            arr.append(left[i])
            i += 1
        while j < len(right):
            arr.append(right[j])
            j += 1
        return arr
        
    
    
    
    

import heapq
class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        point_map = {}
        for point in points:
            point_map[(point[0], point[1])] = (point[0]**2 + point[1]**2)

        return [list(k) for k, v in sorted(point_map.items(), key=lambda kv: kv[1])[:K]]
class Solution:
    """
    Solution #1.
    May 2020.
    
    Sort everything and then get the first k items.
    T: O(NlogN).
    """
    def kClosest(self, points: List[List[int]], k: int) -> List[List[int]]:
        return sorted(points, key = lambda p: p[0]**2 + p[1]**2)[:k]    
    
    """
    Solution #2.
    May 2020.
    
    Use a max-heap to store the k smallest items.
    T: O(N log K).
    """
    def kClosest(self, points: List[List[int]], k: int) -> List[List[int]]:
        heap = []
        
        for x,y in points:
            # We put the -ve sign becase we want this to be a max heap.
            key = -(x**2 + y**2)
            heapq.heappush(heap, (key, x, y))
            if len(heap) > k:
                heapq.heappop(heap)
        
        return [[x,y] for distance, x, y in heap]
    
    """
    Solution #3.
    June 2020.
    
    Use QuickSelect.
    T: O(N) in the average case; O(N^2) in the worst case.
    """
    def kClosest(self, points: List[List[int]], k: int) -> List[List[int]]:
        # random.shuffle(points)
        self.quickSelect(points, k - 1, 0, len(points) - 1)
        return points[:k]
    
    def quickSelect(self, points, k, p, q):
        partitionIndex  = self.partition(points, p, q)
        if partitionIndex == k:
            return partitionIndex
        elif partitionIndex > k:
            return self.quickSelect(points, k, p, partitionIndex  - 1)
        else:
            return self.quickSelect(points, k, partitionIndex + 1, q)
                
    # p <= k <= boundary: items that are smaller than the pivot.
    # boundary + 1 <= < q: items that are bigger than the pivot.
    def partition(self, points, p, q):
        distancePivot = self.getSortingKey(points[q])
        boundary = p - 1
        
        for j in range(p, q):
            if self.getSortingKey(points[j]) < distancePivot:
                boundary += 1
                points[boundary], points[j] = points[j], points[boundary]
        
        # Insert the pivot in the correct position.
        boundary += 1
        points[boundary], points[q] = points[q], points[boundary]
        
        return boundary

    def getSortingKey(self, p):
        return p[0] ** 2 + p[1] ** 2
import heapq
class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        add_index = [tup + [idx] for (tup, idx) in zip(points,list(range(len(points))))]
        print(add_index)
        sorted_by_second = sorted(add_index, key=lambda tup: sqrt(tup[1]**2 + tup[0]**2))
        res = []
        for i in range(K):
            res += [sorted_by_second[i][:2]]
        return res


class Solution:
    
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        squared = [round((x[0]**2 + x[1]**2)**(1/2),2) for x in points]
        self.heapSort(points, squared)
        result = []
        
        for i in range(K):
            result.append(points[i])
        return result
    
    def heapify(self, points, squared, n, i):
        
        largest = i
        l = 2 * i + 1
        r = 2 * i + 2
        
        if l < n and squared[l] > squared[largest]:
            largest = l
        
        if r < n and squared[r] > squared[largest]:
            largest = r
            
        if largest != i:
            squared[largest], squared[i] = squared[i], squared[largest]
            points[largest], points[i] = points[i], points[largest]
            
            self.heapify(points, squared, n, largest)
            
    def heapSort(self, points, squared):
        n = len(squared)
        
        for i in range( n // 2 - 1, -1, -1):
            self.heapify(points, squared, n, i)
            
        for i in range(n-1, 0, -1):
            squared[i], squared[0] = squared[0], squared[i]
            points[i], points[0] = points[0], points[i]
            self.heapify(points, squared, i, 0)
            
        # Solution_1
#         min_dist = []
        
#         def calculate_sqrt(x):
#             return round(math.sqrt(x[0]**2 + x[1]**2), 2)
        
#         for i in range(0, len(points)):
            
#             min_dist.append((calculate_sqrt(points[i]), points[i]))
        
#         min_dist = sorted(min_dist)
        
       
        
        
#         return [min_dist[i][1] for i in range(K) ]

import heapq
class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        
        heap = []
        
        for p in points:
            dist = -(p[0]*p[0] + p[1]*p[1])
            heapq.heappush(heap, (dist, p[0], p[1]))
            
            if len(heap) > K:
                heapq.heappop(heap)
        
        res = [[p[1], p[2]] for p in heap]
        return res
        

class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        points.sort(key=lambda p:(p[0]*p[0] + p[1]*p[1]))
        return points[:K]

import heapq

class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        heap = []
        for point in points:
            dist = point[0] * point[0] + point[1] * point[1]
            heapq.heappush(heap, (-dist, point))
            if len(heap) > K:
                heapq.heappop(heap)
        
        return [tuple[1] for tuple in heap]

#         left = 0
#         right = len(points) - 1
#         while left <= right:
#             mid = self.helperPivot(left, right, points)
#             if mid == K:
#                 break
#             if mid < K:
#                 left = mid+1
#             else:
#                 right = mid - 1
#         return points[:K]
             
    
#     def helperPivot(self,start, end, points):
#         def dist(point):
#             return point[0]**2 + point[1]**2
#         pivot = start
#         left = start + 1
#         right = end
#         while left <= right:
#             if dist(points[left]) > dist(points[pivot]) and dist(points[right]) < dist(points[pivot]):
#                 #swap
#                 points[left], points[right] = points[right], points[left]
#             if dist(points[left]) <= dist(points[pivot]):
#                 left += 1
#             if dist(points[right]) >= dist(points[pivot]):
#                 right -=1
#         #right is the correct position of pivot
#         points[pivot], points[right] = points[right], points[pivot]
#         return right
            
    
                    
        

class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        # given a list of x, y, return the smallest Kth  x^2 +y^2 pairs
        # need to maintain a Max heap of length K
        # BUT, Python does not support Max heap from heapq
        # We are going to x -1 to the distance and use a min heapp

        pq=[]
        for x, y in points:
            if len(pq)<K:
                heapq.heappush(pq, (-x*x-y*y,[x,y])) # (distance, [coordinate])
            else: 
                small_distance, small_coordinate = heapq.heappop(pq)
                if -x*x-y*y > small_distance:
                    heapq.heappush(pq, (-x*x-y*y,[x,y]))
                else:
                    heapq.heappush(pq, (small_distance,small_coordinate))

        
        return [_[1] for _ in pq]
import operator
class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        res = []
        dict_m = {}
        for i in range(len(points)):
            pt = points[i]
            tmp = (pt[0] ** 2 + pt[1] ** 2)
            dict_m[i] = tmp
        sorted_m = sorted(list(dict_m.items()), key = operator.itemgetter(1))
        tmp = [t[0] for t in sorted_m]
        res = [points[tmp[i]] for i in range(K)]
        print(res)
        return res 
        # sorted_m.keys()
        # return res
        # res = sorted(res)
        # print(res)
        # return res[:K]

class Solution:
    from collections import defaultdict
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        distDict = defaultdict(list)
        
        for point in points:
            dist = point[0]**2 + point[1]**2
            distDict[dist].append(point)
            
        sortedKey = sorted(distDict.keys())
        
        rst = []
        for key in sortedKey:
            numOfPairs = len(distDict[key])
            if(numOfPairs < K):
                K -= numOfPairs
                rst += distDict[key]
            elif(numOfPairs >= K):
                rst += distDict[key][:K]
                return rst
import random

class Solution:
    def kClosest(self, nums: List[List[int]], k: int) -> List[List[int]]:
        dist =[]
        for i in range(len(nums)):
            dist.append(nums[i][0] **2 + nums[i][1] **2)
        #Create a hMap
        hMap={}
        for i in range(len(dist)):
            hMap[i] = dist[i]
            
        unique_ids = list(hMap.keys())
        
        def helper(unique_ids,k,start,end):
            if start >= end:
                return
            pivot_idx = random.randint(start,end)
            
            unique_ids[start],unique_ids[pivot_idx] = unique_ids[pivot_idx], unique_ids[start]
            
            pivot = hMap[unique_ids[start]]
            
            smaller = start
            
            for bigger in range(start+1, end+1):
                if hMap[unique_ids[bigger]] <= pivot:
                    smaller +=1
                    unique_ids[smaller], unique_ids[bigger] = unique_ids[bigger],unique_ids[smaller]
                    
            unique_ids[start], unique_ids[smaller] = unique_ids[smaller],unique_ids[start]
            
            if k == smaller:
                return 
            elif k < smaller:
                helper(unique_ids,k,start,smaller-1)
            else:
                helper(unique_ids,k,smaller+1,end)
                
                
        
        helper(unique_ids,k,0,len(unique_ids)-1)
        return [nums[i] for i in unique_ids[:k]]

class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        
        pq = []
        
        def distance(point):
            return ( point[0]*point[0] + point[1]*point[1])
        for index, point in enumerate(points):
            heapq.heappush(pq,(distance(point),index,point))
            
        ans = []  
        k=0
        while pq and k<K:
            _,_,point = heapq.heappop(pq)
            ans.append(point)
            k+=1
        
        return ans

from heapq import *
class Solution:
    def kClosest(self, points: List[List[int]], k: int) -> List[List[int]]:
        maxHeap = []
        for i in range(k):
            heappush(maxHeap, [-self.distance(points[i]), points[i]])
        for j in range(k, len(points)):
            if maxHeap[0][0] < -self.distance(points[j]):
                heappop(maxHeap)
                heappush(maxHeap, [-self.distance(points[j]), points[j]])
        return [points for (distance, points) in list(maxHeap)]
            
    def distance(self, coord):
        return coord[0]**2 + coord[1]**2
# use maxheap for nlogk

# Time - O(NlogN), Space - O(N)
# class Solution:
#     def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
#         points.sort(key = lambda points: points[0]**2 + points[1]**2)
#         return points[:K]
    
# Time - O(Nlogn), Space - O(N)
import heapq
class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        ans = []
        min_heap = []
        for pnt in points:
            heapq.heappush(min_heap, (pnt[0]**2+pnt[1]**2, pnt))
        for i in range(K):
            ans.append(heapq.heappop(min_heap)[1])

        return ans


class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        heap = []
        for i, p in enumerate(points):
            x, y = p
            dist = x**2 + y**2
            heapq.heappush(heap, (-dist, i))
            if len(heap) > K:
                heapq.heappop(heap)
                
        res = []
        for _, i in heap:
            res.append(points[i])
            
        return res
            

import heapq
class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        def get_distance(points):
            return sqrt(points[0]**2 + points[1] **2)
        dist = [[get_distance(point), point] for point in points]
        heapq.heapify(dist)
        ans = []
        for i in range(K):
            ans.append(heapq.heappop(dist)[1])
        return ans        
class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        points.sort(key=lambda point: (point[0]*point[0])+(point[1]*point[1]))
        return points[:K]
from heapq import *
class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        hp = []
        n = len(points)
        if K >= n:
            return points
        if K == 0:
            return []
        for x, y in points:
            key = -(x**(2) + y**(2))
            if len(hp) < K:
                heappush(hp,(key,x,y))
            else:
                if -hp[0][0] > -key:
                    heappop(hp)
                    heappush(hp,(key,x,y))
        ans = []
        for _ in range(K):
            key, x, y = heappop(hp)
            ans.append([x,y])
        return ans

class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        nums=[]
        heapq.heapify(nums)
        def dist(x,y):
            return (x**2+y**2)**0.5
        for x,y in points:
            heapq.heappush(nums,(-dist(x,y),[x,y]))
            if len(nums)>K:
                heapq.heappop(nums)
        return [k[1] for k in nums]
from collections import defaultdict
from heapq import heapify, heappush, heappop
class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        temp = []
        for i in points:
            dist = i[0]**2 + i[1]**2
            temp.append([dist, [i[0],i[1]]])
        heapify(temp)
        res= []
        for i in range(K):
            res.append(heappop(temp)[1])
        return res      

class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        
        def edistance(point):
            d=sqrt(point[0]**2 + point[1]**2)
            return d
        result=[]
        stk=[]
        for point in points:
            dis=edistance(point)
            heapq.heappush(stk,[dis,point])
        
        while K>0:
            result.append(heapq.heappop(stk)[1])
            K-=1
        return result

from math import sqrt
class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        sqrt_d = {}
        
        for x,y in points:
            sqrt_d[(x,y)] = sqrt(x**2 + y**2)
            
        #print(sqrt_d)
        return heapq.nsmallest(K, sqrt_d, key=lambda x: sqrt_d[x])
class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        heap = []
        for x,y in points:
            dis2 = x**2 + y**2
            if len(heap)<K:
                heapq.heappush(heap, (-dis2, (x,y)))
            else:
                if -dis2>heap[0][0]:
                    heapq.heappop(heap)
                    heapq.heappush(heap,(-dis2,(x,y)))
        ans = []
        for i in range(K):
            x,y = heapq.heappop(heap)[1]
            # ans = [[x, y]] + ans
            ans.append([x,y])
        return ans[::-1]
import math
import heapq

class Solution:
    def getEuclideanDistance(self, p1):
        return math.sqrt((p1[0])**2 + (p1[1])**2)
    
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        
        kheap = []
        heapq.heapify(kheap)
        size = 0
        
        for i,point in enumerate(points):
            heapq.heappush(kheap,(-self.getEuclideanDistance(point), i))
            size += 1
            if size > K:
                heapq.heappop(kheap)
                
        kpoints = []
        while kheap:
            dist, ind = heapq.heappop(kheap)
            kpoints.append(points[ind])
        return kpoints

class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        # data structure: heap
        h = []
        for point in points:
            d = 0
            for cor in point:
                d += cor**2
            h.append((d, point))
            
        ksmallests = heapq.nsmallest(K,h)
        ret = []
        for val,point in ksmallests:
            ret.append(point)
        return ret
class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        
        heap = []
        indices = {}
        
        for i in range(len(points)):
            curr_point = points[i]
            distance = math.sqrt(curr_point[0]**2 + curr_point[1]**2)
            heapq.heappush(heap, distance)
            if distance in list(indices.keys()):
                indices[distance].append(i)
            else:
                indices[distance] = [i]
        
        result = []
        i = 0
        while i < K:
            curr_distance = heapq.heappop(heap)
            curr_indices = indices[curr_distance]
            for index in curr_indices:
                result.append(points[index])
                i += 1
        
        return result
                
        
        
        
        

class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        return sorted(points, key=lambda x: (x[0]**2+x[1]**2))[:K]
from collections import defaultdict
import math
import heapq
class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        heap = []
        lookup = defaultdict(list)
        
        for x, y in points:
            distance = math.sqrt(x ** 2 + y ** 2)
            lookup[distance].append([x, y])
            heapq.heappush(heap, -distance)
            if len(heap) > K:
                heapq.heappop(heap)
        result = []
        for distance in heap:
            result.append(lookup[-distance].pop())
        return result
class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        heap = []
        for a,b in points:
            if len(heap) < K:
                heapq.heappush(heap, (-1*(a**2+b**2),[a,b]))
            else:
                heapq.heappushpop(heap, (-1*(a**2+b**2),[a,b]))
        op = []
        print(heap)
        for i in range(K):
            print(i)
            op.append(heap[i][1])
        return op
class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        length = len(points)
        if length <= K:
            return points

        # key: distance to origin
        # value: list index in points
        distances: Dict[int, List[int]] = defaultdict(list)
        for i, li in enumerate(points):
            distance = (li[0] ** 2 + li[1] ** 2) ** 0.5
            distances[distance].append(i)

        order = OrderedDict(sorted(distances.items()))
        ans = []
        for i, (key, value) in enumerate(order.items()):
            for j in value:
                ans.append(points[j])
            if len(ans) >= K:
                break
        return ans[:K]
class myObj:
    def __init__(self, val, p):
        self.val = val
        self.p = p
    def __lt__(self, other):
        return self.val < other.val
    
    
class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        
        def partition(arr, left, right):
            i = left
            for j in range(left, right):
                if arr[j]<arr[right]:
                    arr[i], arr[j] = arr[j], arr[i]
                    i += 1
            arr[i], arr[right] = arr[right], arr[i]
            return i
        
        def sort(arr, left, right, K):
            if left<right:
                p = partition(arr, left, right)
                if p==K:
                    return
                elif p<K:
                    sort(arr, p+1, right, K)
                else:
                    sort(arr, left, p-1, K)
        
        dis = [0]*len(points)
        for i in range(len(points)):
            dis[i] = myObj(points[i][0]*points[i][0] + points[i][1]*points[i][1], points[i])
        sort(dis, 0, len(points)-1, K)
        
        ans = []
        for i in range(K):
            ans.append(dis[i].p)
        return ans
from scipy.spatial import distance
class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
            points.sort(key = lambda P: P[0]**2 + P[1]**2)
            return points[:K]
'''

- preprocess, store (distance, (x,y))
- init heap w/ k
- push pop for the rest
- return heap

- O(k) + O(N-K log K) / O(n)
- O(K log N) / O(n)

'''
# O(nlogk) / O(k)
class Point:
    def __init__(self, pt):
        self.distance = -1*sqrt(pt[0]**2+pt[1]**2)
        self.pt = pt
        
    def __lt__(self, other):
        return self.distance < other.distance
        
    def __eq__(self, other):
        return self.distance == other.distance

class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        
        heap = []
        heapq.heapify(heap)
        
        for pt in points:
            curr_pt = Point(pt)
            
            if len(heap) < K:
                heapq.heappush(heap, curr_pt)
            else:
                heapq.heappushpop(heap, curr_pt)
                
        return [pt.pt for pt in heap]
class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:    
        self.quickselect(points, K, 0, len(points)-1)
        return points[:K]
    
    def quickselect(self, points, k, lo, high):
        if lo >= high:
            return
        
        pivot = self.partition(points, lo, high)
        
        if pivot+1 == k:
            return
        elif pivot < k:
            self.quickselect(points, k, pivot+1, high)
        else: 
            self.quickselect(points, k, lo, pivot-1)
        
    def partition(self, points, lo, high):
        pivot = self.dist(points[high])
        w = lo
        
        for r in range(lo, high):
            if self.dist(points[r]) < pivot:
                points[w], points[r] = points[r], points[w]
                w+=1
                
        points[w], points[high] = points[high], points[w]
        return w
    
    def dist(self, pt):
        return pt[0]**2 + pt[1]**2

class Solution:    
    def partition(self, nums, l, r):
        pivot = nums[r]
        i = l - 1
        for j in range(l, r):
            if nums[j] <= pivot:
                i += 1
                nums[i], nums[j] = nums[j], nums[i]
        nums[i+1], nums[r] = nums[r], nums[i+1]
        return i+1
    
    def selectSort(self, nums, l, r, k):
        if l < r:
            p = self.partition(nums, l, r)
            if p == k:
                return
            elif k < p :
                self.selectSort(nums, l, p-1, k)
            else:
                self.selectSort(nums, p+1, r, k)
        
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        dists = [math.sqrt(p[0] ** 2 + p[1] ** 2)  for p in points]
        self.selectSort(dists, 0 , len(dists)-1, K)
        print(dists)
        out= []
        if K == len(points):
            return points
        for p in points:
            if math.sqrt(p[0] ** 2 + p[1] ** 2) < dists[K]:
                out.append(p)
        
        return out
                

class Solution:
    import math
    def distance(self, point):
        return math.sqrt(point[0]**2 + point[1]**2)
    
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        distances = {}
        for point in points:
            distances[tuple(point)] = self.distance(point)
        print(distances)
        points_sorted = sorted(list(distances.keys()), key=lambda x: distances[x])
        
        return [list(i) for i in points_sorted[:K]]

class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        points.sort(key = lambda x: x[0]*x[0] + x[1]*x[1])
        return points[:K]

from numpy import argsort
from numpy import array
from numpy import sqrt
from numpy import sum

class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        p = array(points)
        q = sqrt(sum(p ** 2, axis=1))
        return p[argsort(q)[:K]]
import numpy as np

class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        p = np.array(points)
        q = np.sqrt(np.sum(p ** 2, axis=1))
        return p[np.argsort(q)[:K]]
from collections import OrderedDict
class Solution:
    def partition(self, nums, l, r):
        piv = nums[r]
        i = l -1
        for j in range(l, r):
            if nums[j] <= piv:
                i += 1
                nums[i], nums[j] = nums[j], nums[i]
        nums[r], nums[i+1] = nums[i+1], nums[r]
        return (i+1)
    
    def selectSort(self, nums, left, right, k):
        if left < right:
            p = self.partition(nums, left, right)
            if p == k:
                return 
            elif k < p:
                self.selectSort(nums, left, p-1, k)
            else:
                self.selectSort(nums, p+1, right, k)    
    
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        dists = [math.sqrt(p[0] ** 2 + p[1] ** 2)  for p in points]
        self.selectSort(dists, 0 , len(dists)-1, K)
        print(dists)
        out= []
        if K == len(points):
            return points
        for p in points:
            if math.sqrt(p[0] ** 2 + p[1] ** 2) < dists[K]:
                out.append(p)
        
        return out
                

import heapq

class Distance:
    
    def __init__(self, p, d):
        self.d = d
        self.p = p
    
    def __lt__(self, other):

        return self.d >= other.d
        

class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        
        
        def distance(p):
            return math.sqrt(p[0]**2+p[1]**2)
        
        distances_k = []
        for p in points:
            
            d = Distance(p, distance(p))
            
            if len(distances_k) < K:
                heapq.heappush(distances_k, d)
            elif d.d < distances_k[0].d:
                heapq.heappop(distances_k)
                heapq.heappush(distances_k, d)
                
        return [x.p for x in distances_k]
                
                
                
                
        
        
        
        
        
        

class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        K -= 1
        start, end = 0, len(points) - 1
        while start <= end:
            mid = self.__partition(points, start, end)
            if mid == K:
                break
            elif mid < K:
                start = mid + 1
            else:
                end = mid - 1
        return points[: K + 1]
    
    def __partition(self, points, lo, hi):
        __dist = lambda points, i : points[i][0] ** 2 + points[i][1] ** 2
        d = __dist(points, lo)
        i, j = lo, hi + 1
        while True:
            while i < hi and __dist(points, i + 1) < d:
                i += 1
            while j > lo and __dist(points, j - 1) > d:
                j -= 1
            i, j = i + 1, j - 1
            if i >= j:
                break
            points[i], points[j] = points[j], points[i]
        points[lo], points[j] = points[j], points[lo]
        return j
class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        
        # need to get the distance of each points
        # need to keep the K smallest, then pq can pop the samllest
        # need to negative the input and pushpop so that i can do the k operation
        
        # other approach would be quick_select
        # by randomize the input, you can get the average O(n) worst case O(n**2)
        # get the parition of input arr, 
        # if partition < k: quick_select(st, partition-1)
        # else: quick_select(partition+1, end)
        
        def parition(st, end):
            
            lo, i, j = st, st, end
            while True:
                
                while i < end and arr[i][0] <= arr[lo][0]:
                    i+=1
                
                while j > st and arr[lo][0] <= arr[j][0]:
                    j-=1
                    
                if i >= j:
                    break
                arr[i], arr[j] = arr[j], arr[i]
            arr[lo], arr[j] = arr[j], arr[lo]
            return j
        
        def quick_select(st, end):
            
            # this gurantee 0,1. 1,1 is not possible
            if st > end:
                return
            
            par = parition(st, end)
            
            if par == K:
                return
            
            if par > K:
                quick_select(st, par-1)
            else:
                quick_select(par+1, end)
            return 
        
        arr = []
        for pt in points:
            arr.append( [math.sqrt(sum([x**2 for x in pt])), pt] )
        #print(arr)
        random.shuffle(arr)
        #print(arr)
        quick_select(0, len(arr)-1)
        
        return [x[1] for x in arr[:K]]
import math
import heapq
class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        sol = []
        for point in points:
            dist = math.sqrt(point[0]**2 + point[1] **2)
            heapq.heappush(sol, (dist, point))
        print(sol)
        return [point[1] for point in heapq.nsmallest(K, sol)]
            

"""
10, 14, 11, 15 12
10, 11, 14, 15
"""

class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        def swap(arr, idx1, idx2):
            arr[idx1], arr[idx2] = arr[idx2], arr[idx1]
        
        def swapElement(idx1, idx2):
            swap(self.array, idx1, idx2)
            swap(self.distance, idx1, idx2)
            
        def Partition(low, high, index) -> int:
            """
            Returns the new index
            """
            partDistance = self.distance[index]
            swapElement(high, index)
            newIndex = low
            for i in range(low, high):
                if (self.distance[i] < partDistance):
                    swapElement(newIndex, i)
                    newIndex += 1
            
            swapElement(newIndex, high)
            return newIndex
        
        def Util(k):
            low = 0
            high = len(self.distance) - 1
            while (low <= high):
                index = random.randint(low, high)
                newIndex = Partition(low, high, index)
                if (newIndex == k):
                    return self.array[:k]
                if (newIndex > k):
                    high = newIndex - 1
                else:
                    low = newIndex + 1
        if (len(points) == K):
            return points
        self.array = points
        self.distance = []
        for point in points:
            self.distance.append(math.sqrt(point[0]**2 + point[1]**2))
        print(self.distance)
        return Util(K)
            
            
import math

class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        if len(points) == 0:
            return []
        
        points.sort(key = lambda x: math.sqrt(x[0]*x[0] + x[1]*x[1]))
        
        return points[:K]

class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:   
        def distance(point):
            x, y = point
            return x ** 2 + y ** 2
        
        # negative values to make a min-heap a max-heap
        distances = [(distance(p), p) for p in points]
        heapq.heapify(distances)

        
        for point in points:
            heapq.heappushpop(distances, (-distance(point), point))
        return [p for _, p in heapq.nsmallest(K, distances)]
class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        left = 0
        right = len(points) - 1
        target = K - 1
        while (left <= right):
            rand_pivot = random.randint(left, right)
            pivot = self.partition(left, right, rand_pivot, points) 
            if (pivot == target):
                return points[0 : pivot + 1]
            elif (pivot < target):
                left = pivot + 1
            else:
                right = pivot - 1
    
    def partition(self, i, j, pivot, arr):
        pivot_element = self.distFromOrigin(arr[pivot])
        self.swap(pivot, j, arr)
        result = i
        for x in range(i, j):
            if self.distFromOrigin(arr[x]) < pivot_element:
                self.swap(x, result, arr)
                result += 1
        self.swap(result, j, arr)
        return result
        
    
    def swap(self, i, j , arr):
        temp = arr[i]
        arr[i] = arr[j]
        arr[j] = temp
    
    def distFromOrigin(self, point):
        return math.sqrt((point[0] ** 2 + point[1] ** 2))

class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        dist = lambda i: points[i][0]**2 + points[i][1]**2

        def sort(i, j, K):
            # Partially sorts A[i:j+1] so the first K elements are
            # the smallest K elements.
            if i >= j: return

            # Put random element as A[i] - this is the pivot
            k = random.randint(i, j)
            points[i], points[k] = points[k], points[i]

            mid = partition(i, j)
            if K < mid - i + 1:
                sort(i, mid - 1, K)
            elif K > mid - i + 1:
                sort(mid + 1, j, K - (mid - i + 1))

        def partition(i, j):
            # Partition by pivot A[i], returning an index mid
            # such that A[i] <= A[mid] <= A[j] for i < mid < j.
            oi = i
            pivot = dist(i)
            i += 1

            while True:
                while i < j and dist(i) < pivot:
                    i += 1
                while i <= j and dist(j) >= pivot:
                    j -= 1
                if i >= j: break
                points[i], points[j] = points[j], points[i]

            points[oi], points[j] = points[j], points[oi]
            return j

        sort(0, len(points) - 1, K)
        return points[:K]
class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        def getDistance(p: List[int]):
            return p[0] ** 2 + p[1] ** 2

        def helper(points: List[List[int]], K: int):
            if len(points) == K:
                return points

            left, right = [], []
            import random
            random.shuffle(points)

            pivot = points[0][0]

            for point in points:
                curr_distance = point[0]
                if curr_distance > pivot:
                    right.append(point)
                else:
                    left.append(point)

            if len(left) >= K:
                return helper(left, K)
            else:
                return left + helper(right, K - len(left))
        distances = []
        for point in points:
            distances.append([getDistance(point), point])

        top_results = helper(distances, K)
        for i, result in enumerate(top_results):
            top_results[i] = result[1]
        return top_results

class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        dist_square = lambda i: points[i][0] ** 2 + points[i][1] ** 2

        def sort(i: int, j: int, K: int):
            # partially sorts points[i...j+1] so the first K elements are the
            # smallest K elements
            if i >= j:
                return

            # put random element as points[i] as the pivot
            k = random.randint(i, j)
            points[i], points[k] = points[k], points[i]  # ???

            mid = partition(i, j)
            if K < mid - i + 1:
                sort(i, mid - 1, K)
            elif K > mid - i + 1:
                sort(mid + 1, j, K - (mid - i + 1))

        def partition(i: int, j: int) -> int:
            # partition by pivot points[i], returning an index mid such that
            # points[i] <= points[mid] <= points[j] for i < mid < j
            oi = i
            pivot = dist_square(i)
            i += 1

            while True:
                while i < j and dist_square(i) < pivot:
                    i += 1
                while i <= j and dist_square(j) >= pivot:
                    j -= 1
                if i >= j:
                    break
                points[i], points[j] = points[j], points[i]
            points[oi], points[j] = points[j], points[oi]
            return j

        sort(0, len(points) - 1, K)
        return points[:K]
class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        def dist(point):
            return sum([e**2 for e in point])
        
        def partition(nums, left, right):
            x = dist(nums[right])
            i = left - 1
            for j in range(left, right):
                if dist(nums[j]) < x:
                    i += 1
                    nums[i], nums[j] = nums[j], nums[i]
            i += 1
            nums[right], nums[i] = nums[i], nums[right]
            return i
        
        p = -1
        left = 0
        right = len(points) - 1
        
        
        while p != K:
            
            p = partition(points, left, right)
            
            if p + 1 < K:
                left = p + 1
            elif p + 1 > K:
                right = p - 1
            else:
                return points[:K]
            
        return points[:K]

# def distSqr(point: List[int]) -> int:
#     d = 0
#     for p in point:
#         d += p ** 2
#     return d

# def partition(points: List[List[int]], left: int, right: int) -> int:
#     if left > right:
#         return -1
#     pivot = left
#     left = left + 1
#     pivotDistSqr = distSqr(points[pivot])
#     while left <= right:
#         if distSqr(points[left]) > pivotDistSqr and distSqr(points[right]) < pivotDistSqr:
#             points[left], points[right] = points[right], points[left]
#             left += 1
#             right -= 1
#         if distSqr(points[left]) <= pivotDistSqr:
#             left += 1
#         if distSqr(points[right]) >= pivotDistSqr:
#             right -= 1
#     points[pivot], points[right] = points[right], points[pivot]
#     pivot = right
#     return pivot

# class Solution:
#     def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
#         pivot = -1 
#         n = len(points)
#         left, right = 0, n - 1
        
#         while pivot != K - 1:
            
#             pivot = partition(points, left, right)
#             if pivot < K - 1:
#                 left = pivot + 1
#             elif pivot == K - 1:
#                 return points[0:K]
#             elif pivot > K - 1:
#                 right = pivot - 1
#         return points[0:K]


# import heapq
# class Solution:
#     def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
#         idx = defaultdict(list)
#         heap = []
        
#         for i, p in enumerate(points):
#             d = (p[0] ** 2 + p[1] ** 2)
#             if heap and len(heap) >= K:
#                 if - heap[0] > d:
#                     larger_dist = - heap[0]
#                     idx[larger_dist].pop()
#                     heapq.heappushpop(heap, - d)
#                     idx[d].append(i)
#             else:
#                 heapq.heappush(heap, -d)
#                 idx[d].append(i)
        
#         res = []
#         for indicies in idx.values():
#             for i in indicies:
#                 res.append(points[i])
#         return res

class Solution(object):
    def kClosest(self, points, K):
        
        # create a function that calculates the euclidean distance
        dist = lambda i: sqrt(points[i][0]**2 + points[i][1]**2)

        def sort(i, j, K):
            # return when recursion tree reaches the leaf
            if i >= j: return
            
            # get the random index between ith and jth index
            k = random.randint(i, j)
                        
            # swap the values in the array  
            points[i], points[k] = points[k], points[i]

            # return the partitioned index A[i] <= A[mid] <= A[j] for i < mid < j.
            mid = partition(i, j)
            
            # if the number of Kth element is smaller than the partitioned index 
            if K < mid - i + 1:
                # sort again in the left array 
                sort(i, mid - 1, K)
            # if the number of kth element is bigger than the partitioned index 
            elif K > mid - i + 1:
                # sort again in the right array 
                sort(mid + 1, j, K - (mid - i + 1))

        def partition(i, j):
            # set the orgiinal i 
            oi = i
            # get the distance of the random pivot  
            pivot = dist(i)
            i += 1

            while True:
                # increment i until the current distance reaches the pivot
                while i < j and dist(i) < pivot:
                    i += 1
                
                # decrement j until the current distance reaches the pivot 
                while i <= j and dist(j) >= pivot:
                    j -= 1
                    
                # break i and j meets  
                if i >= j: break
                
                # swap the ith and jth index 
                points[i], points[j] = points[j], points[i]
            
            # swap back the original index and the jth index
            points[oi], points[j] = points[j], points[oi]
            
            # return jth index 
            return j

        # invoke sort function 
        sort(0, len(points) - 1, K)
        # return the kth amount of sorted result
        return points[:K]
class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        
        points.sort(key=lambda x: x[0]**2+x[1]**2)
        return points[:K]

from heapq import *

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.distance = x ** 2 + y ** 2
    
    def __lt__(self, other):
        return self.distance > other.distance
    

class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        distance_heap = []
        for x, y in points:
            if len(distance_heap) >= K:
                if Point(x, y) > distance_heap[0]:
                    heappop(distance_heap)
                    heappush(distance_heap, Point(x, y))
            else:
                heappush(distance_heap, Point(x, y))
        res = []
        
        while distance_heap:
            point =  heappop(distance_heap)
            res.append([point.x, point.y])
        
        return res
class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        def dist(point):
            return sum([e**2 for e in point])
        
        def partition(nums, left, right):
            x = dist(nums[right])
            i = left - 1
            for j in range(left, right):
                if dist(nums[j]) < x:
                    i += 1
                    nums[i], nums[j] = nums[j], nums[i]
            i += 1
            nums[right], nums[i] = nums[i], nums[right]
            return i
        
        p = -1
        left = 0
        right = len(points) - 1
        
        
        while p + 1 != K:
            
            p = partition(points, left, right)
            
            if p + 1 < K:
                left = p + 1
            elif p + 1 > K:
                right = p - 1
        return points[:K]

# def distSqr(point: List[int]) -> int:
#     d = 0
#     for p in point:
#         d += p ** 2
#     return d

# def partition(points: List[List[int]], left: int, right: int) -> int:
#     if left > right:
#         return -1
#     pivot = left
#     left = left + 1
#     pivotDistSqr = distSqr(points[pivot])
#     while left <= right:
#         if distSqr(points[left]) > pivotDistSqr and distSqr(points[right]) < pivotDistSqr:
#             points[left], points[right] = points[right], points[left]
#             left += 1
#             right -= 1
#         if distSqr(points[left]) <= pivotDistSqr:
#             left += 1
#         if distSqr(points[right]) >= pivotDistSqr:
#             right -= 1
#     points[pivot], points[right] = points[right], points[pivot]
#     pivot = right
#     return pivot

# class Solution:
#     def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
#         pivot = -1 
#         n = len(points)
#         left, right = 0, n - 1
        
#         while pivot != K - 1:
            
#             pivot = partition(points, left, right)
#             if pivot < K - 1:
#                 left = pivot + 1
#             elif pivot == K - 1:
#                 return points[0:K]
#             elif pivot > K - 1:
#                 right = pivot - 1
#         return points[0:K]


# import heapq
# class Solution:
#     def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
#         idx = defaultdict(list)
#         heap = []
        
#         for i, p in enumerate(points):
#             d = (p[0] ** 2 + p[1] ** 2)
#             if heap and len(heap) >= K:
#                 if - heap[0] > d:
#                     larger_dist = - heap[0]
#                     idx[larger_dist].pop()
#                     heapq.heappushpop(heap, - d)
#                     idx[d].append(i)
#             else:
#                 heapq.heappush(heap, -d)
#                 idx[d].append(i)
        
#         res = []
#         for indicies in idx.values():
#             for i in indicies:
#                 res.append(points[i])
#         return res

class Solution(object):
    def kClosest(self, points, K):
        # create a lambda function that calculates the euclidean distance
        dist = lambda i: sqrt(points[i][0]**2 + points[i][1]**2)

        def sort(i, j, K):
            # return when recursion tree reaches the leaf
            if i >= j: return            
            # get the pivot index that will sort array A[ i...pivot...j ]
            k = random.randint(i, j)
            # place the pivot in the front of the array A[ pivot.i....j ]
            points[i], points[k] = points[k], points[i]
            # return the partitioned index A[i] <= A[mid] <= A[j] 
            mid = partition(i, j)
            # if the number of Kth element is smaller than the partitioned index 
            if K < mid - i + 1:
                # sort recursively from the left array 
                sort(i, mid - 1, K)
            # if the number of kth element is bigger than the partitioned index 
            elif K > mid - i + 1:
                # sort recursively from the right array 
                sort(mid + 1, j, K - (mid - i + 1))

        def partition(i, j):
            # save the pivot index
            oi = i
            # get the distance value of the pivot  
            pivot = dist(i)
            # move to the index that needs to be sorted 
            i += 1
            # loop until the sorting is complete
            while True:
                # increment from left if the distance of ith index is smaller than the pivot
                while i < j and dist(i) < pivot:
                    i += 1
                # increment from right if the distance of ith index is bigger than the pivot
                while i <= j and dist(j) >= pivot:
                    j -= 1
                # break if the sorting is complete 
                if i >= j: break                
                # place the smaller value to the leftside and bigger value to the rigtside of the pivot
                points[i], points[j] = points[j], points[i]
            # move the pivot value to the middle index
            points[oi], points[j] = points[j], points[oi]
            # return middle index 
            return j

        # invoke sort function 
        sort(0, len(points) - 1, K)
        # return the kth amount of sorted result
        return points[:K]
class myObj:
    def __init__(self, val, p):
        self.val = val
        self.p = p
    def __lt__(self, other):
        return self.val > other.val
        

class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        h = []
        heapq.heapify(h)
        for p in points:
            dis = p[0]*p[0] + p[1]*p[1]
            heapq.heappush(h, myObj(dis, p))
            if len(h)>K:
                heapq.heappop(h)
        
        ans = []
        while h:
            obj = heapq.heappop(h)
            ans.append(obj.p)
        return ans
class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        self.countK(points, 0, len(points)-1, K)
        return points[:K]

    
    def countK(self, points, l, r, k):
        if l < r:
            p = self.createPartition(points, l, r, k)
            if p == k:
                return
            elif p < k:
                self.countK(points, p+1, r, k)
            else:
                self.countK(points, l, p-1, k)
    
    def createPartition(self, points, l, r, k):
        pivot = points[r]
        count = l
        for i in range(l, r):
            if (points[i][0]**2 + points[i][1]**2) <= (pivot[0]**2 + pivot[1]**2):
                points[i], points[count] = points[count], points[i]
                count += 1
        points[count], points[r] = points[r], points[count]
        return count

from queue import PriorityQueue
class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        pq = PriorityQueue()
        for i, c in enumerate(points):
            pq.put((c[0]*c[0] + c[1]*c[1], i))
        res = []
        for j in range(K):
            pair = pq.get()
            res.append(points[pair[1]])
        return res

class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        return (sorted(points,key=lambda point:point[0]*point[0]+point[1]*point[1]))[:K]

class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        return sorted(points,key = lambda x : x[0]**2 + x[1]**2)[:K]
# Quick select: time = O(N)
# Logic to understand https://leetcode.com/problems/k-closest-points-to-origin/discuss/220235/Java-Three-solutions-to-this-classical-K-th-problem.
# clean code https://leetcode.com/problems/k-closest-points-to-origin/discuss/219442/Python-with-quicksort-algorithm

class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        
        self.qkSel(points, 0, len(points)-1, K)
        return points[:K]
    
    def qkSel(self, points, l, r, K):
        if l > r: return 
        
        idx = random.randint(l, r)
        points[idx], points[r] = points[r], points[idx]  
        i = l
        
        for j in range(l, r):
            if self.cmp(points[j], points[r]):
                points[i], points[j] = points[j], points[i]
                i += 1
                
        points[i], points[r] = points[r], points[i] 
        
        if i == K: return 
        elif i < K: return self.qkSel(points, i+1, r, K)
        elif i > K: return self.qkSel(points, l, i-1, K)
        
    
    def cmp(self, p1, p2):
        return (p1[0]**2 + p1[1]**2) - (p2[0]**2 + p2[1]**2) < 0  
        
                
        
        
        
        
        
        

# class Solution:
#     def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
#         points.sort(key=lambda x: math.sqrt(x[0]*x[0] + x[1]*x[1]))
#         return points[:K]
    
class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        dist = lambda x: math.sqrt(points[x][0]**2 + points[x][1]**2)
        def dc(i, j, K):
            if i >= j: return
            oi, oj = i, j
            i += 1
            pivot = dist(oi)
            while True:
                while i < j and dist(i) < pivot: i += 1
                while i <= j and dist(j) >= pivot: j -= 1
                if i >= j: break
                points[i], points[j] = points[j], points[i]
            points[oi], points[j] = points[j], points[oi]
            if K-1 == j: return
            elif K-1 < j: dc(oi, j, K) 
            else: dc(j+1, oj, K)
        dc(0, len(points)-1, K)
        return points[:K]
            

class MyHeapObj():
    def __init__(self, x, y, dist):
        self.x = x
        self.y = y
        self.dist = dist
    
    def __gt__(self, other):
        return self.dist > other.dist

import heapq
class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        pointsHeap = []
        for p in points:
            dist = (p[0]**2+p[1]**2)**0.5
            heapq.heappush(pointsHeap, MyHeapObj(p[0], p[1], dist))
            
        
        ret = []
        for i in range(K):
            point = heapq.heappop(pointsHeap)
            ret.append([point.x, point.y])
        return ret

import random

class Solution:
    # def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
    #     values = [(points[i][0] ** 2 + points[i][1] ** 2, i) for i in range(len(points))]
    #     values.sort()
    #     return [points[i] for _, i in values[:K]]
    def _value(self, point):
        return point[0] ** 2 + point[1] ** 2
    
    def _quickSelect(self, start: int, end: int, points: List[List[int]]) -> int:
        if start == end:
            return start
        # random pivot
        p = random.randint(start, end)
        points[p], points[start] = points[start], points[p]
        j = start + 1
        for i in range(start + 1, end + 1):
            if self._value(points[i]) <= self._value(points[start]):
                points[i], points[j] = points[j], points[i]
                j += 1
        # swap i to its correct position
        points[j - 1], points[start] = points[start], points[j - 1]
        return j - 1
                
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        
        start = 0
        end = len(points) - 1
        # select the pivot using quick select
        while True:
            p = self._quickSelect(start, end, points)
            # if the pivot <= K: quick select for K on the right
            # print(p)
            if p < K - 1:
                start = p + 1
            elif p > K - 1:
                end = p - 1
            else:
                return points[:p+1]
        return None
           

class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        
        def dist(i):
            return points[i][0]** 2 + points[i][1]**2
        
        def sort(i,j,K):
            """partially sort A[i:j+1] so that first K elements are the                      smallest K
            """
            if i >= j: 
                return
            k = random.randint(i,j)
            points[i],points[k] = points[k],points[i]
            mid = partition(i,j)
            if K < mid - i + 1:
                sort(i,mid-1,K)
            elif K > mid - i + 1:
                sort(mid + 1, j, K - (mid - i + 1))
            
        def partition(i,j):
            """partition by pivot A[i], returning an index mid
                such that A[i] <= A[mid] <= A[j] for i < mid < j
            """
            oi = i
            pivot = dist(i)
            i += 1
            
            while True:
                while i < j and dist(i) < pivot:
                    i += 1
                while i <= j and dist(j) >= pivot:
                    j -= 1
                if i >= j: break
                points[i], points[j] = points[j], points[i]
            points[oi], points[j] = points[j], points[oi]
            return j
        
        sort(0,len(points)-1, K)
        return points[:K]
        
        
class Solution:
    def kClosest(self, points, K):
        self.sort(points, 0, len(points)-1, K)
        return points[:K]
    
    def sort(self, points, l, r, K):
        if l < r:
            p = self.partition(points, l, r)
            if p == K:
                return
            elif p < K:
                self.sort(points, p+1, r, K)
            else:
                self.sort(points, l, p-1, K)
            
    def partition(self, points, l, r):
        pivot = points[r]
        a = l
        for i in range(l, r):
            if (points[i][0]**2 + points[i][1]**2) <= (pivot[0]**2 + pivot[1]**2):
                points[a], points[i] = points[i], points[a]
                a += 1
        points[a], points[r] = points[r], points[a]                
        return a
class Solution:
    def kClosest(self, points: List[List[int]], k: int) -> List[List[int]]:
        
        def distance(point):
            return point[0]**2 + point[1]**2
        
        def find_pivot_index(low, high, p_index):
            ppoint = points[p_index]
            points[p_index], points[high] = points[high], points[p_index]
            
            i = low
            j = low
            
            while j < high:
                if distance(points[j]) < distance(points[high]):
                    points[i], points[j] = points[j], points[i]
                    i+=1
                j+=1
                
            points[i], points[high] = points[high], points[i]
            return i
            
        def recursion(low, high):
            pivot_index = random.randint(low, high)
            p = find_pivot_index(low, high, pivot_index)
            
            if p == k-1:
                return points[:p+1]
            elif p<k-1:
                return recursion(p+1, high)
            else:
                return recursion(low, p-1)
            
        return recursion(0, len(points)-1)
class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        res=[]
        def insert(d, pos):
            l=0
            r=len(res)
            while l<r:
                mid=(l+r)//2
                if res[mid][0]>d:
                    r=mid
                else:
                    l=mid+1
            res.insert(l, [d,pos])
            Max=res[-1][0]
        Max=float('inf')
        
        for i in range(len(points)):
            dis=(points[i][0])**2+(points[i][1])**2
            print(len(res))
            if i>K:
                res.pop()
            insert(dis, points[i])
        ans=[]
        for i in range(K):
            ans.append(res[i][1])
        return ans
class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        # quick sort with partition
        dist = lambda x:x[0]**2 + x[1]**2
        def partition(start, end):
            ran = random.randint(start, end)
            pivot = end
            points[pivot], points[ran] = points[ran], points[pivot]

            border = start
            for cur in range(start, end):
                # sort in descending order
                if dist(points[cur]) <= dist(points[pivot]):
                    points[cur], points[border] = points[border], points[cur]
                    border += 1
            points[border], points[pivot] = points[pivot], points[border]
            return border  
        
        def quick_sort(left, right, k):
            if left >= right: return
            p = partition(left, right)
            if p == k - 1:
                return
            if p < k - 1:
                quick_sort(p + 1, right, k)
            else:
                quick_sort(left, p - 1, k)
        
        quick_sort(0, len(points) - 1, K)
        return points[:K]

class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        points.sort(key = lambda x : x[0] ** 2 + x[1] ** 2)
        return points[:K]
class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        def dist(x) -> int:
            return x[0]**2 + x[1] ** 2
        
        def partition(i,j)->int:
            pivot = points[j]
            l = i - 1
            for r in range(i,j):
                rv = points[r]
                # print(r, points)
                if dist(rv) < dist(pivot):
                    l += 1
                    points[l], points[r] = points[r], points[l]
            points[j], points[l+1] = points[l+1], points[j]
            return l+1
        
        
        def sort(i,j,K):
            if i >= j: return 
            
            mid = partition(i,j)
            if (mid - i + 1) == K:
                return
            elif (mid - i + 1) < K:
                sort(mid + 1, j, K - (mid - i + 1))
            else:
                sort(i, mid - 1, K)
        
        sort(0, len(points)-1, K)
        return points[:K]

class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        # quick sort with partition
        dist = lambda x:x[0]**2 + x[1]**2
        def partition(start, end):
            ran = random.randint(start, end)
            pivot = end
            points[pivot], points[ran] = points[ran], points[pivot]

            border = start
            for cur in range(start, end):
                # sort in descending order
                if dist(points[cur]) >= dist(points[pivot]):
                    points[cur], points[border] = points[border], points[cur]
                    border += 1
            points[border], points[pivot] = points[pivot], points[border]
            return border  
        
        def quick_sort(left, right, k):
            if left >= right: return
            p = partition(left, right)
            if p == k - 1:
                return
            if p < k - 1:
                quick_sort(p + 1, right, k)
            else:
                quick_sort(left, p - 1, k)
        
        quick_sort(0, len(points) - 1, len(points) - K + 1)
        return points[len(points) - K:]

class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        start, end = 0, len(points) - 1
        pivot = -1
        while start <= end:
            pivot = self.partition(points, start, end)
            if pivot == K:
                return points[0:pivot]
            elif pivot > K:
                end = pivot - 1
            else:
                start = pivot + 1
       
        return points
            
    
    def partition(self, points, start, end):
        pivot = start
        for i in range(start, end+1):
            if self.dist(points, i) < self.dist(points, end):
                points[i], points[pivot] = points[pivot], points[i]
                pivot += 1
        
        points[end], points[pivot] = points[pivot], points[end]
        return pivot
    
    def dist(self, points, i):
        return points[i][0]**2 + points[i][1]**2

class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        
        def dist(x)->int:
            return x[0]**2 + x[1]**2
        
        def partition(i, j)->int:
            
            # pivotu7684u9009u62e9u5728u6700u540eu4e00u4e2au70b9
            pivot = points[j]
            l = i - 1
            for r in range(i, j):
                rv = points[r]
                if dist(rv) < dist(pivot):
                    l += 1
                    points[l], points[r] = points[r], points[l]
            points[j], points[l+1] = points[l+1], points[j]
            
            return l+1
        
        def sort(i, j, K):
            # u6b64u65f6u8bf4u660eu5df2u7ecfu6392u597du5e8fu4e86
            if i >= j: return 
            
            mid = partition(i,j)
            
            if (mid - i + 1) == K:
                return
                
            elif (mid - i + 1) < K:
                sort(mid + 1, j, K - (mid - i + 1))
            else:
                sort(i, mid - 1, K)
            
        sort(0, len(points)-1, K)
        
        return points[:K]
import math
import heapq
class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        
        # distance type is euclidean
        # we can iterate over the points and compute dist for each point O(n)
        # we can then heapify the list with dist being the priority in the priority queue O(n)
        # we can can return the k_smallest O(klogn)
        # using extra O(n) space
        
        # test cases
        # empty points
        # < k points
        # all same values and len > k
        if len(points) == 0:
            return []
        if K >= len(points):
            return points
        
        def euclidean(point):
            x, y = point
            return math.sqrt(math.pow(x,2) + math.pow(y,2))
        
        mod_points = []
        for point in points:
            dist = euclidean(point)
            mod_points.append((dist, point[0], point[1]))
        
        heapq.heapify(mod_points)
        
        ret_values = heapq.nsmallest(K, mod_points)
        ret_values = [[x,y] for dist, x, y in ret_values]
        
        return ret_values
        

class Solution:
    
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        
        max_heap = []
        
        for point in points:
            x, y = point[0], point[1]
            dist = x**2 + y**2
            if len(max_heap) < K:               
                heapq.heappush(max_heap, (-dist, x, y))
            else:
                if dist < -max_heap[0][0]:
                    heapq.heappop(max_heap)
                    heapq.heappush(max_heap, (-dist, x, y))
            
        return [[x, y] for (dist, x,y) in max_heap]
class Solution(object):
    def kClosest(self, points, K):

        # minHeap #
        def distance(p):
            return (p[0]**2 + p[1]**2)**0.5 
        
        def partition(l, r):
            pivot = points[r]
            mover = l
            ## u628au5de6u8fb9u7684u548cu53f3u8fb9u6bd4, u628au5c0fu7684u653eu5230u5de6u8fb9, u5927u7684u653eu5230u53f3u8fb9
            for i in range(l, r):
                if distance(points[i]) <= distance(pivot):
                    points[mover], points[i] = points[i], points[mover]
                    mover += 1
            points[mover], points[r] = points[r], points[mover]
            return mover # such that all the numbers are smaller than the number at the mover
        
        def sort(l, r):
            if l < r:
                p = partition(l, r)
                if p == K:
                    return
                elif p < K:
                    sort(p+1, r)
                else:
                    sort(l, p-1)
                    
        sort(0, len(points)-1)
        return points[:K]

# Max heap
class Node:
    def __init__(self,x,y,d=None):
        self.x = x
        self.y = y
        if d:
            self.distance = d
        else:
            self.distance = x*x + y*y
        
        
def parent(i):
    return (i-1)//2
def left(i):
    return 2*i +1
def right(i):
    return 2*i +2
def swap(heap,i,j):
    t = heap[i]
    heap[i] = heap[j]
    heap[j] = t
    
def heapify(heap):
    n = len(heap)
    for i in range(1,n):
        p = parent(i)
        
        while i > 0 and heap[p].distance < heap[i].distance:
            # print(heap[i].distance,i,p)
            swap(heap,i,p)
            i = p
            p = parent(i)
        
    
def replace_top(heap,node):
    t = heap[0]
    heap[0] = node
    n = len(heap)
    # swap(heap,0,n-1)
    i = 0
    l = left(i)
    r = right(i)
    while i < n and not((l >= n or heap[l].distance <= heap[i].distance) and (r >= n or heap[r].distance <= heap[i].distance)):
        if l < n and r < n:
            if  heap[l].distance < heap[r].distance:
                swap(heap,i,r)
                i = r
            else:
                swap(heap,i,l)
                i = l
        elif l < n:
            swap(heap,i,l)
            i = l
        else:
            swap(heap,i,r)
            i = r
            
        l = left(i)
        r = right(i)
            
class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        n = len(points)
        heap = []
        for i in range(K):
            p = points[i]
            node = Node(p[0],p[1])
            heap.append(node)
            # print(node.distance)
        heapify(heap)
        top = 0
        # print(heap)
        # print("max",heap[top].distance)
        for i in range(K,n):
            p = points[i]
            
            distance = p[0]*p[0] + p[1]*p[1]
            # print(p,distance)
            if heap[top].distance > distance:
                node = Node(p[0], p[1], distance)
                # print(p,node.distance)
                replace_top(heap,node)
        return [ [heap[i].x,heap[i].y] for i in range(K)]
                
        

import random

class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> int:
        l, h, k = 0, len(points) - 1, K - 1
        
        dist = lambda i : points[i][0] ** 2 + points[i][1] ** 2
        
        def partition(l: int, h: int) -> int:
            t = random.randint(l, h)
            points[t], points[h] = points[h], points[t]
            for i, val in enumerate(points[l:h+1], l):
                if dist(i) < dist(h):
                    points[i], points[l] = points[l], points[i]
                    l += 1
            points[l], points[h] = points[h], points[l]
            return l
        
        
        while True:
            pos = partition(l, h)
            if pos < k:
                l = pos + 1
            elif pos > k:
                h = pos - 1
            else:
                return points[:K]
                    

