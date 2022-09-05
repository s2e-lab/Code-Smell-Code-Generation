class Solution:
    def minCost(self, houses: List[int], Cost: List[List[int]], m: int, n: int, target: int) -> int:
        @lru_cache(None)
        def dfs(i, j, k):
            if i == len(houses):
                if j == target:
                    return 0
                else:
                    return float('inf')
                
            if houses[i] != 0:
                return dfs(i + 1, int(houses[i] != k) + j, houses[i])
            
            cost = float('inf')
            for index, c in enumerate(Cost[i]):
                cost = min(cost, dfs(i + 1, int(index + 1 != k) + j, index + 1) + c)
                
            return cost
        
        return dfs(0, 0, 0) if dfs(0, 0, 0) != float('inf') else -1
    

class Solution:
    def minCost(self, houses: List[int], cost: List[List[int]], m: int, n: int, target: int) -> int:
        
        @lru_cache(None)
        def dp(i, prev, count):
            if count>target:
                return float('inf')
            if i == m:
                return 0 if count==target else float('inf')
            
            return min((cost[i][c-1] if c!=houses[i] else 0)+dp(i+1, c, count+(prev!=c)) for c in range(1,n+1) if not houses[i] or c==houses[i])
        ans = dp(0, houses[0], 1 if houses[0] else 0)
        return -1 if ans==float('inf') else ans
from functools import lru_cache
import math

class Solution:
    def minCost(self, houses: List[int], cost: List[List[int]], m: int, n: int, target: int) -> int:
        @lru_cache(None)
        def paint(i, k, color):
            # print(i, k, color)
            if k == 1 and i == m:
                return 0
            if k == 0 or i == m:
                return math.inf
            total_cost = math.inf
            if houses[i] != 0:
                if houses[i] == color:
                    return paint(i + 1, k, color)
                else:
                    return paint(i + 1, k - 1, houses[i])
            for c in range(1, n + 1):
                kk = k
                if c != color:
                    kk -= 1
                cost_ = cost[i][c - 1] + paint(i + 1, kk, c)    
                total_cost = min(total_cost, cost_)
            # print(i, k, color, total_cost)
            return total_cost
        
        res = paint(0, target + 1, -1)
        return res if res != math.inf else -1
class Solution:
    def minCost(self, houses: List[int], cost: List[List[int]], m: int, n: int, target: int) -> int:
        #dp[house][target][color]
        dp = [[[float('inf')]*(n+1) for j in range(target+1)] for i in range(m)]
        
        if houses[0] == 0:
            for idx, c in enumerate(cost[0]):
                dp[0][1][idx+1] = c
        else:
            dp[0][1][houses[0]] = 0
            
        for i in range(1, m):
            if houses[i] != 0:
                for t in range(1, target+1):
                    for cidx in range(1, n+1):
                        pre_cost = dp[i-1][t][cidx]
                        if pre_cost == float('inf'):
                            continue
                        if houses[i] == cidx:
                            dp[i][t][cidx] = min(dp[i][t][houses[i]], pre_cost)
                        elif t + 1 <= target:
                            dp[i][t + 1][houses[i]] = min(dp[i][t+1][houses[i]], pre_cost)
            else:
                for t in range(1, target+1):
                    for cidx in range(1, n+1):
                        pre_cost = dp[i-1][t][cidx]
                        if pre_cost == float('inf'):
                            continue
                        for cidx2, c in enumerate(cost[i]):
                            cidx2+=1
                            if cidx == cidx2:
                                dp[i][t][cidx2] = min(dp[i][t][cidx2], pre_cost + c)
                            elif t +1 <= target:
                                dp[i][t+1][cidx2] = min(dp[i][t+1][cidx2], pre_cost + c)
        #print(dp)
        res = float('inf')
        for cidx in range(1, n+1):
            res = min(res, dp[-1][target][cidx])
            
        if res == float('inf'):
            return -1
        return res
                                

class Solution:
    def minCost(self, houses: List[int], cost: List[List[int]], m: int, n: int, target: int) -> int:
        #dp[house][target][color]
        dp = [[[float('inf')]*(n+1) for j in range(target+1)] for i in range(m)]
        
        if houses[0] == 0:
            for idx, c in enumerate(cost[0]):
                dp[0][1][idx+1] = c
        else:
            dp[0][1][houses[0]] = 0
            
        for i in range(1, m):
            if houses[i] != 0:
                for t in range(1, target+1):
                    for cidx in range(1, n+1):
                        pre_cost = dp[i-1][t][cidx]
                        if pre_cost == float('inf'):
                            continue
                        if houses[i] == cidx:
                            dp[i][t][cidx] = min(dp[i][t][houses[i]], pre_cost)
                        elif t + 1 <= target:
                            dp[i][t + 1][houses[i]] = min(dp[i][t+1][houses[i]], pre_cost)
            else:
                for t in range(1, target+1):
                    for cidx in range(1, n+1):
                        pre_cost = dp[i-1][t][cidx]
                        if pre_cost == float('inf'):
                            continue
                        for cidx2, c in enumerate(cost[i]):
                            cidx2+=1
                            if cidx == cidx2:
                                dp[i][t][cidx2] = min(dp[i][t][cidx2], pre_cost + c)
                            elif t +1 <= target:
                                dp[i][t+1][cidx2] = min(dp[i][t+1][cidx2], pre_cost + c)
        #print(dp)
        res = min(dp[-1][target])
        if res == float('inf'):
            return -1
        return res
                                

class Solution:
    def minCost(self, houses: List[int], cost: List[List[int]], m: int, n: int, target: int) -> int:
        @lru_cache(None)
        def recur(index, prev_color, neighbor_count):
            if index == m:
                return 0 if neighbor_count == target else float('inf')
            
            if houses[index] != 0:                    
                return recur(index + 1, houses[index], neighbor_count + int(prev_color != houses[index]))
            
            total = float('inf')
            for color in range(1, n + 1):
                total = min(total, cost[index][color - 1] + recur(index + 1, color, neighbor_count + int(prev_color != color)))
            return total
                            
        final_ans = recur(0, -1, 0)
        return final_ans if final_ans != float('inf') else -1
class Solution:
    def minCost(self, houses: List[int], cost: List[List[int]], m: int, n: int, target: int) -> int:
        dp = [[[float('inf') for _ in range (n)] for _ in range(target+1)] for _ in range(m)]
        for c in range(1, n+1):
            if houses[0] == 0: dp[0][1][c-1] = cost[0][c-1]
            elif houses[0] == c: dp[0][1][c-1] = 0
        for i in range(1, m):
            for k in range (1, min(i+1,target)+1):
                for c in range(1, n+1):
                        prev = min (dp[i-1][k][c-1], min([dp[i-1][k-1][c_-1] for c_ in range(1, n+1) if c_ != c]))
                        if houses[i]==0 or houses[i]==c:
                            dp[i][k][c-1] = prev + cost[i][c-1] *(houses[i] == 0)
        res = min(dp[-1][-1])
        return -1 if res == float('inf') else res
                

class Solution:
    def minCost(self, houses: List[int], cost: List[List[int]], m: int, n: int, target: int) -> int:
        maxc = 10**9
        N = len(houses)
        dp = [[[None]*(target+1) for _ in range(n+1)] for _ in range(m)]
        
        def solve(i,j,k):
            if k < 0: return maxc
            if dp[i][j][k] is None:
                if i == N - 1:
                    if k != 0:
                        dp[i][j][k] = maxc
                    else:
                        if houses[i] == 0:
                            dp[i][j][k] = cost[i][j-1]
                        else:
                            dp[i][j][k] = 0
                else:
                    dp[i][j][k] = cost[i][j-1] if houses[i] == 0 else 0
                    if houses[i+1] == 0:
                        dp[i][j][k] += min([solve(i+1, jj, k-1 if jj != j else k) for jj in range(1, n+1)])
                    elif houses[i+1] == j:
                        dp[i][j][k] += solve(i+1, j, k)
                    else:
                        dp[i][j][k] += solve(i+1, houses[i+1], k-1)
            return dp[i][j][k]
        
        if houses[0] == 0:
            result = min([solve(0, j, target-1) for j in range(1, n+1)])
        else:
            result = solve(0, houses[0], target-1)
        
        return result if result < maxc else -1

from functools import lru_cache
import math

class Solution:
    def minCost(self, houses: List[int], cost: List[List[int]], m: int, n: int, target: int) -> int:
        @lru_cache(None)
        def paint(i, color, k):
            # print(i, k, color)
            if k == 0 and i == m:
                return 0
            if k < 0 or i == m:
                return math.inf
            total_cost = math.inf
            if houses[i] != 0:
                return paint(i + 1, houses[i], k - (1 if houses[i] != color else 0))
            for c in range(1, n + 1):
                cost_ = cost[i][c - 1] + paint(i + 1, c, k - (1 if c != color else 0))    
                total_cost = min(total_cost, cost_)
            # print(i, k, color, total_cost)
            return total_cost
        
        # neighbors = 0
        # prev = 0
        # for h in houses:
        #     if h == 0: 
        #         continue
        #     if h != prev:
        #         neighbors += 1
        #         prev = h
        # if neighbors > target:
        #     return -1
        res = paint(0, -1, target)
        return res if res != math.inf else -1
class Solution:
    def minCost(self, houses: List[int], cost: List[List[int]], m: int, n: int, target: int) -> int:
        # dp[i][c][k]: i means the ith house, c means the cth color, k means k neighbor groups
        dp = [[[math.inf for _ in range(n)] for _ in range(target + 1)] for _ in range(m)]
        
        for c in range(1, n + 1):
            if houses[0] == c: dp[0][1][c - 1] = 0
            elif not houses[0]: dp[0][1][c - 1] = cost[0][c - 1]
                
        for i in range(1, m):
            for k in range(1, min(target, i + 1) + 1):
                for c in range(1, n + 1):
                    if houses[i] and c != houses[i]: continue
                    same_neighbor_cost = dp[i - 1][k][c - 1]
                    diff_neighbor_cost = min([dp[i - 1][k - 1][c_] for c_ in range(n) if c_ != c - 1] or [math.inf])
                    paint_cost = cost[i][c - 1] * (not houses[i])
                    dp[i][k][c - 1] = min(same_neighbor_cost, diff_neighbor_cost) + paint_cost
        res = min(dp[-1][-1])
        return res if res < math.inf else -1
            
        
        
        

class Solution:
    def minCost(self, houses: List[int], costs: List[List[int]], m: int, n: int, target: int) -> int:
        @lru_cache(None)
        def helper(hIdx, prevColor, groups):
            if hIdx == m:
                return 0 if groups == target else float('inf')
            if houses[hIdx]: # painted
                return helper(hIdx + 1, houses[hIdx], groups + int(houses[hIdx] != prevColor))
            total = float('inf')
            for c in range(1, n+1):
                total = min(total, costs[hIdx][c-1] + helper(hIdx+1, c, groups + int(c != prevColor)))
            return total
        
        res = helper(0, 0, 0)
        return res if res < 10 ** 9 else -1

# k: num of neighborhoods, i: num of houses, c: color of the last house
# dp[k][i][c]: min cost to form k neighborhoods using the first i houses and the i-th house's color is c
# init: dp[0][0][*] = 0, else is inf
# if the i - 1 house's color != i house's color, means they are not in the same neighborhood
# dp[k][i][ci] = dp[k - 1][i - 1][ci-1] + cost
# else if the i - 1house's color == i house's color, means they are in the same neighborhood
# dp[k][i][ci] = dp[k][i - 1][ci-1] + cost
# if houses[i] == ci, house i is painted by color i, cost = 0, no need to paint
# else, cost = cost[i][ci]
# if ci != houses[i], means house i is painted by another color, dp[k][i][ci] = inf
# ans = min(dp[target][m][*])
# u56e0u4e3au9700u8981u679au4e3eu6240u6709u7684 house i - 1 u548c house i u7684u989cu8272u7684u7ec4u5408, u6240u4ee5u65f6u95f4u590du6742u5ea6u4e58u4ee5 n * n
# u56e0u4e3a dp[k] u53eau548c dp[k] u548c dp[k - 1] u6709u5173, u6240u4ee5u65f6u95f4u590du6742u5ea6u53efu4ee5u4ece O(target * m * n) -> O(m * n)
# O(target * m * n * n) time compleixty, O(m * n) space complexity
class Solution:
    def minCost(self, houses: List[int], costs: List[List[int]], m: int, n: int, target: int) -> int:
        dp = [[[float('inf') for _ in range(n + 1)] for _ in range(m + 1)] for _ in range(target + 1)]
        for c in range(n + 1):
            dp[0][0][c] = 0

        for k in range(1, target + 1):
            for i in range(k, m + 1):  # u56e0u4e3a i < k u7684u60c5u51b5u4e0du5408u6cd5, u7ec4u6210 k u4e2a neighborhoods u81f3u5c11u4e5fu9700u8981 k u4e2a house
                hi = houses[i - 1]
                hj = 0  # u521du59cbu5316u524du4e00u4e2au623fu5b50u7684u989cu8272u4e3a 0, u5982u679c i < 2, u5219u5b83u5c31u662fu7b2cu4e00u4e2au623fu5b50, u6240u4ee5u8bbeu524du4e00u4e2au623fu5b50u53efu4ee5u662fu4efbu4f55u989cu8272u7684, u5373 0
                if i >= 2:
                    hj = houses[i - 2]
                for ci in range(1, n + 1):
                    if hi != 0 and hi != ci:
                        dp[k][i][ci] = float('inf')
                        continue
                    cost = 0
                    if hi != ci:
                        cost = costs[i - 1][ci - 1]
                    for cj in range(1, n + 1):
                        dp[k][i][ci] = min(dp[k][i][ci], dp[k - (ci != cj)][i - 1][cj] + cost)
        
        res = min(dp[target][m][c] for c in range(1, n + 1))
        if res == float('inf'):
            return -1
        return res

# https://zxi.mytechroad.com/blog/dynamic-programming/leetcode-1473-paint-house-iii/
# k: num of neighborhoods, i: num of houses, c: color of the last house
# dp[k][i][c]: min cost to form k neighborhoods using the first i houses and the i-th house's color is c
# init: dp[0][0][*] = 0, else is inf
# if the i - 1 house's color != i house's color, means they are not in the same neighborhood
# dp[k][i][ci] = dp[k - 1][i - 1][ci-1] + cost
# else if the i - 1house's color == i house's color, means they are in the same neighborhood
# dp[k][i][ci] = dp[k][i - 1][ci-1] + cost
# if houses[i] == ci, house i is painted by color i, cost = 0, no need to paint
# else, cost = cost[i][ci]
# if ci != houses[i], means house i is painted by another color, dp[k][i][ci] = inf
# ans = min(dp[target][m][*])
# u56e0u4e3au9700u8981u679au4e3eu6240u6709u7684 house i - 1 u548c house i u7684u989cu8272u7684u7ec4u5408, u6240u4ee5u65f6u95f4u590du6742u5ea6u4e58u4ee5 n * n
# u56e0u4e3a dp[k] u53eau548c dp[k] u548c dp[k - 1] u6709u5173, u6240u4ee5u65f6u95f4u590du6742u5ea6u53efu4ee5u4ece O(target * m * n) -> O(m * n)
# O(target * m * n * n) time compleixty, O(m * n) space complexity
class Solution:
    def minCost(self, houses: List[int], costs: List[List[int]], m: int, n: int, target: int) -> int:
        dp = [[[float('inf') for _ in range(n + 1)] for _ in range(m + 1)] for _ in range(target + 1)]
        for c in range(n + 1):
            dp[0][0][c] = 0

        for k in range(1, target + 1):
            for i in range(k, m + 1):  # u56e0u4e3a i < k u7684u60c5u51b5u4e0du5408u6cd5, u7ec4u6210 k u4e2a neighborhoods u81f3u5c11u4e5fu9700u8981 k u4e2a house
                hi = houses[i - 1]
                hj = 0  # u521du59cbu5316u524du4e00u4e2au623fu5b50u7684u989cu8272u4e3a 0, u5982u679c i < 2, u5219u5b83u5c31u662fu7b2cu4e00u4e2au623fu5b50, u6240u4ee5u8bbeu524du4e00u4e2au623fu5b50u53efu4ee5u662fu4efbu4f55u989cu8272u7684, u5373 0
                if i >= 2:
                    hj = houses[i - 2]
                for ci in range(1, n + 1):
                    if hi != 0 and hi != ci:  # u5f53u8fd9u4e2au623fu5b50 i u5df2u7ecfu6709u989cu8272u4e14u4e0du662fu5f53u524du60f3u7ed9u5b83u7684u989cu8272u65f6, u76f4u63a5u8df3u8fc7
                        dp[k][i][ci] = float('inf')
                        continue
                    cost = 0  # u5982u679cu8fd9u4e2au623fu5b50u5df2u7ecfu6709u989cu8272u4e14u548cu5f53u524du60f3u7ed9u5b83u7684u989cu8272u4e00u6837, cost u4e3a 0
                    if hi != ci:  # u5982u679cu8fd9u4e2au623fu5b50u6ca1u6709u989cu8272, cost u4e3a costs u77e9u9635u4e2du7684u503c
                        cost = costs[i - 1][ci - 1]
                    for cj in range(1, n + 1):
                        dp[k][i][ci] = min(dp[k][i][ci], dp[k - (ci != cj)][i - 1][cj] + cost)
        
        res = min(dp[target][m][c] for c in range(1, n + 1))
        if res == float('inf'):
            return -1
        return res

class Solution:
    def minCost(self, houses: List[int], cost: List[List[int]], m: int, n: int, target: int) -> int:
        memo = {}
        
        @lru_cache(None)
        def dfs(i, k, t):
            
            if t < 0 and t > m-i:
                return float('inf')
            
            if m == i:
                return 0 if t == 0 else float('inf')
            
            if (i, k, t) not in memo:
                if houses[i]:
                    memo[i,k,t] = dfs(i+1, houses[i], t - int(houses[i] != k))
                else:
                    
                    memo[i,k,t] = min(cost[i][j-1] + dfs(i+1, j, t - int(j != k)) for j in range(1, n+1))
            return memo[i,k,t]
        
        ans = dfs(0, 0, target)
        return ans if ans != float('inf') else -1
class Solution:
    def minCost(self, houses: List[int], cost: List[List[int]], m: int, n: int, target: int) -> int:
        memo = [[[-1 for i in range(m+1)] for j in range(n+1)] for _ in range(m+1)]
        def dp(i, prevColor, nbrs):
            if nbrs > target : return float('inf')
            if i == m:
                if nbrs == target: return 0
                else: return float('inf')
            
            if memo[i][prevColor][nbrs] != -1: return memo[i][prevColor][nbrs]
            ans = float('inf')
            if houses[i] == 0:
                for j in range(n):
                    if j+1 == prevColor:
                        temp = dp(i+1, j+1, nbrs)
                    else:
                        temp = dp(i+1, j+1, nbrs+1)
                    ans = min(ans, cost[i][j]+ temp)
            else:
                if prevColor == houses[i]:
                    ans = min(ans, dp(i+1, houses[i], nbrs))
                else:
                    ans = min(ans, dp(i+1, houses[i], nbrs+1))
            memo[i][prevColor][nbrs] = ans
            return ans
        ans = dp(0,0,0)
        if ans == float('inf'): return -1
        else: return ans

class Solution:
    def minCost(self, houses: List[int], cost: List[List[int]], m: int, n: int, target: int) -> int:
        # dp[color][blocks]
        
        dp, dp2 = {(0, 0): 0}, {}
        
        for index, house in enumerate(houses):
            for cj in (range(1, n + 1) if house == 0 else [house]):
                for ci, b in dp:
                    b2 = b + (ci != cj)
                    if b2 > target:
                        continue
                    dp2[cj, b2] = min(dp2.get((cj, b2), float('inf')), dp[ci, b] + (cost[index][cj-1] if cj != house else 0))
            dp, dp2 = dp2, {}
        return min([dp[c, b] for c, b in dp if b == target] or [-1])
class Solution:
    def minCost(self, houses: List[int], cost: List[List[int]], m: int, n: int, target: int) -> int:
        if n == 1 and target >= 2: return -1
        if target > m: return -1
        
        c = 0
        p = -1
        for i in range(m):
            if houses[i] > 0 and houses[i] != p:
                c += 1
                p = houses[i]
        if c > target:
            return -1
        
        cache = {}
        MAX_VAL = float('inf')
        
        def process(i, p, t):
            if t < 0:
                return MAX_VAL
            if i == m:
                if t == 0:
                    return 0
                else:
                    return MAX_VAL
            else:
                if not (i, p, t) in cache:
                    ans = MAX_VAL
                    if houses[i] > 0:
                        if houses[i] == p:
                            ans = process(i+1, p, t)
                        else:
                            ans = process(i+1, houses[i], t-1)
                    else:
                        ans = MAX_VAL
                        for j in range(n):
                            if p == j+1:
                                ans = min(ans, cost[i][j] + process(i+1, p, t))
                            else:
                                ans = min(ans, cost[i][j] + process(i+1, j+1, t-1))
                    cache[(i, p, t)] = ans
            return cache[(i, p, t)]
        
        x = process(0, -1, target)
        return x if x != MAX_VAL else -1
            
            
            
            
            
            
        
        

class Solution:
    def minCost(self, houses: List[int], cost: List[List[int]], m: int, n: int, target: int) -> int:
        # dp[color][target]
        dp = {(0,0):0}
        for i in range(len(houses)):
            tmp = {}
            color = range(1, n+1) if houses[i] == 0 else [houses[i]]
            for curr in color:
                for prev, j in dp:
                    t = j if curr == prev else j+1
                    if t > target:
                        continue
                    tmp[curr,t] = min(tmp.get((curr, t), float('inf')), dp[prev, j]+(cost[i][curr-1] if curr != houses[i] else 0))
            dp = tmp
        return min([dp[c, b] for c, b in dp if b == target] or [-1])
class Solution:
        def minCost(self, A, cost, m, n, target):
            dp, dp2 = {(0, 0): 0}, {}
            for i, a in enumerate(A):
                for cj in (list(range(1, n + 1)) if a == 0 else [a]):
                    for ci, b in dp:
                        b2 = b + (ci != cj)
                        if b2 > target: continue
                        dp2[cj, b2] = min(dp2.get((cj,b2), float('inf')), dp[ci, b] + (cost[i][cj - 1] if cj != a else 0))
                dp, dp2 = dp2, {}
            return min([dp[c, b] for c, b in dp if b == target] or [-1])
        
        
                            

class Solution:
  def minCost(self, houses: List[int], cost: List[List[int]], m: int, n: int, target: int) -> int:
    # TC: O(MNNT), SC: (NT)
    # dp: (n-color, t-blocks): min-cost
    dp0, dp1 = {(0, 0): 0}, {}
    for i, k in enumerate(houses):
      # assume painted houses can NOT be repainted..
      for ik in ([k] if k > 0 else range(1, n + 1)):
        # previous color and blocks
        for pk, pb in dp0:
          bb = pb + (ik != pk)
          if bb > target:
            continue
          dp1[ik, bb] = min(dp1.get((ik, bb), float('inf')), dp0[pk, pb] + (0 if k > 0 else cost[i][ik - 1]))
      dp0, dp1 = dp1, {}
    return min([dp0[k, b] for k, b in dp0 if b == target] or [-1])
from functools import lru_cache
import math

class Solution:
    def minCost(self, houses: List[int], cost: List[List[int]], m: int, n: int, target: int) -> int:
        @lru_cache(None)
        def paint(i, k, color):
            # print(i, k, color)
            if k == 1 and i == m:
                return 0
            if k == 0 or i == m:
                return math.inf
            total_cost = math.inf
            if houses[i] != 0:
                if houses[i] != color:
                    k -= 1
                return paint(i + 1, k, houses[i])
            for c in range(1, n + 1):
                kk = k
                if c != color:
                    kk -= 1
                cost_ = cost[i][c - 1] + paint(i + 1, kk, c)    
                total_cost = min(total_cost, cost_)
            # print(i, k, color, total_cost)
            return total_cost
        
        # neighbors = 0
        # prev = 0
        # for h in houses:
        #     if h == 0: 
        #         continue
        #     if h != prev:
        #         neighbors += 1
        #         prev = h
        # if neighbors > target:
        #     return -1
        res = paint(0, target + 1, -1)
        return res if res != math.inf else -1
class Solution:
    def minCost(self, houses: List[int], cost: List[List[int]], m: int, n: int, target: int) -> int:
        houses = [c - 1 for c in houses]
        ans = [[[float('inf') for k in range(target+1)] for j in range(n)] for i in range(m)]
        
        
        for j in range(n):
            if houses[0] == -1:
                ans[0][j][1] = cost[0][j]
            else:
                ans[0][houses[0]][1] = 0

        for i in range(1, m):
            if houses[i] == -1:
                for j in range(n):
                    for l in range(n):
                        for k in range(1, min(target+1, i+2)):
                            if j == l:
                                ans[i][j][k] = min(ans[i][j][k], ans[i-1][j][k] + cost[i][j])
                            else:
                                ans[i][j][k] = min(ans[i][j][k], ans[i-1][l][k-1] + cost[i][j])
            else:
                for k in range(1, min(target+1, i+2)):
                    for l in range(n):
                        if houses[i] == l:
                            ans[i][houses[i]][k] = min(ans[i][houses[i]][k], ans[i-1][l][k])
                        else:
                            ans[i][houses[i]][k] = min(ans[i][houses[i]][k], ans[i-1][l][k-1])
                            
        res = float('inf')
        for j in range(n):
            res = min(res, ans[m-1][j][target])
        if res == float('inf'):
            res = -1
        return res
            
        
        
        

class Solution:
    def minCost(self, houses: List[int], cost: List[List[int]], m: int, n: int, target: int) -> int:
        dp, dp2 = {(0, 0): 0}, {}
        for i, a in enumerate(houses):
            for cj in (range(1, n + 1) if a == 0 else [a]):
                for ci, b in dp:
                    b2 = b + (ci != cj)
                    if b2 > target: continue
                    dp2[cj, b2] = min(dp2.get((cj, b2), float('inf')), dp[ci, b] + (cost[i][cj - 1] if cj != a else 0))
            dp, dp2 = dp2, {}
        return min([dp[c, b] for c, b in dp if b == target] or [-1])
class Solution:
    def minCost(self, A, cost, m, n, target):
        dp, dp2 = {(0, 0): 0}, {}
        for i, a in enumerate(A):
            for cj in (list(range(1, n + 1)) if a == 0 else [a]):
                for ci, b in dp:
                    b2 = b + (ci != cj)
                    if b2 > target: continue
                    dp2[cj, b2] = min(dp2.get((cj,b2), float('inf')), dp[ci, b] + (cost[i][cj - 1] if cj != a else 0))
            dp, dp2 = dp2, {}
        return min([dp[c, b] for c, b in dp if b == target] or [-1])

class Solution:
    def minCost(self, houses: List[int], cost: List[List[int]], m: int, n: int, target: int) -> int:
        A=houses
        dp, dp2 = {(0, 0): 0}, {}
        for i, a in enumerate(A):
            for cj in (range(1, n + 1) if a == 0 else [a]):
                for ci, b in dp:
                    b2 = b + (ci != cj)
                    if b2 > target: continue
                    dp2[cj, b2] = min(dp2.get((cj,b2), float('inf')), dp[ci, b] + (cost[i][cj - 1] if cj != a else 0))
            dp, dp2 = dp2, {}
        return min([dp[c, b] for c, b in dp if b == target] or [-1])
class Solution:
    def minCost(self, A, cost, m, n, target):
        dp = {(0, 0): 0}
        for i, a in enumerate(A):
            dp2 = {}
            for cj in (range(1, n + 1) if a == 0 else [a]):
                for ci, b in dp:
                    b2 = b + (ci != cj)
                    if b2 > target: 
                        continue
                    dp2[cj, b2] = min(dp2.get((cj,b2), float('inf')), dp[ci, b] + (cost[i][cj - 1] if cj != a else 0))
            dp = dp2
        return min([dp[c, b] for c, b in dp if b == target] or [-1])
class Solution:
    def minCost(self, houses: List[int], cost: List[List[int]], m: int, n: int, target: int) -> int:
        dp = {(0, 0) : 0}
        dp2 = {}
        # dp (x,y) = z, x is the color, y is the number of neighbors and z is the min cost we get so far
        
        for index, house in enumerate(houses):
            for color in (list(range(1, n + 1)) if house == 0 else [house]):
                for preColor, block in dp:
                    newBlock = 0
                    if preColor == color:
                        newBlock = block
                    else:
                        newBlock = block + 1 
                    if newBlock > target:
                        continue 
                    dp2[(color, newBlock)] = min(dp2.get((color, newBlock), float('inf')),  dp[(preColor, block)] + (cost[index][color - 1] if color != house else 0))
            dp, dp2 = dp2, {}
        
            print(dp)
        return min([dp[(i,color)] for i, color in dp if color == target] or [-1])
                
        


class Solution:
    def minCost(self, houses: List[int], cost: List[List[int]], m: int, n: int, target: int) -> int:
        dp = []
        for i in range(m):
            dp.append([[-1]*(n+1) for _ in range(m+1)])
        for i in range(m):
            for j in range(1, i+2):
                for k in range(1, n+1):
                    if i == 0:
                        if houses[0] == 0:
                            dp[0][1][k] = cost[0][k-1]
                        elif houses[0] == k:
                            dp[0][1][k] = 0
                    else:
                        if houses[i] == 0:
                            options = []
                            for last_color in range(1, n+1):
                                if last_color == k:
                                    if dp[i-1][j][k] != -1:
                                        options.append(cost[i][k-1] + dp[i-1][j][k])
                                else:
                                    if dp[i-1][j-1][last_color] != -1:
                                        options.append(cost[i][k-1] + dp[i-1][j-1][last_color])
                            if len(options) != 0:
                                dp[i][j][k] = min(options)
                        elif houses[i] == k:
                            options = []
                            for last_color in range(1, n+1):
                                if last_color == k:
                                    if dp[i-1][j][k] != -1:
                                        options.append(dp[i-1][j][k])
                                else:
                                    if dp[i-1][j-1][last_color] != -1:
                                        options.append(dp[i-1][j-1][last_color])
                            if len(options) != 0:
                                dp[i][j][k] = min(options)
        #print(dp)
        costs = list([x for x in dp[m-1][target] if x!=-1])
        if len(costs) == 0:
            return -1
        else:
            return min(costs)
                            

from functools import lru_cache
import math

class Solution:
    def minCost(self, houses: List[int], cost: List[List[int]], m: int, n: int, target: int) -> int:
        MAX_COST = 10 ** 7
        
        @lru_cache(None)
        def paint(i, color, k):
            # print(i, k, color)
            if k == 0 and i == m:
                return 0
            if k < 0 or i == m:
                return MAX_COST
            if m - i < k - 1:
                return MAX_COST
            if houses[i] != 0:
                return paint(i + 1, houses[i], k - (1 if houses[i] != color else 0))
            return min((cost[i][c - 1] + paint(i + 1, c, k - (1 if c != color else 0)) for c in range(1, n + 1)))
            # print(i, k, color, total_cost)
            # return total_cost
        
        # neighbors = 0
        # prev = 0
        # for h in houses:
        #     if h == 0: 
        #         continue
        #     if h != prev:
        #         neighbors += 1
        #         prev = h
        # if neighbors > target:
        #     return -1
        res = paint(0, -1, target)
        return res if res < MAX_COST else -1
class Solution:
    def minCost(self, houses: List[int], cost: List[List[int]], m: int, n: int, target: int) -> int:
        def dp(i, color, target):
            if target < 0: return float('inf')
            if cache[i][color][target] < 0:
                if houses[i] > 0 and color != houses[i] - 1:
                    cache[i][color][target] = float('inf')
                elif i == m - 1:
                    if target > 0:
                        cache[i][color][target] = float('inf')
                    else:
                        cache[i][color][target] = cost[i][color] if houses[i] == 0 else 0
                else:
                    cost1 = cost[i][color] if houses[i] == 0 else 0
                    cost2 = min(dp(i+1, c, target - int(c != color)) for c in range(n))
                    cache[i][color][target] = cost1 + cost2
            return cache[i][color][target]
        
        cache = [[[-1] * target for _ in range(n)] for _ in range(m)]
        res = min(dp(0, c, target-1) for c in range(n))
        return -1 if res == float('inf') else res
class Solution:
    def minCost(self, houses: List[int], cost: List[List[int]], m: int, n: int, target: int) -> int:
        memo = {}
        def dfs(idx, groups, prev_color):
            nonlocal target, m, n
            if (idx, groups, prev_color) in memo:
                return memo[(idx, groups, prev_color)]
            if groups > target:
                return sys.maxsize 
            if idx == len(houses):
                if groups == target:
                    return 0
                return sys.maxsize 
            else:
                if houses[idx] != 0:
                    return dfs(idx + 1, groups + (1 if houses[idx] != prev_color else 0), houses[idx])
                else:
                    low = sys.maxsize
                    for i in range(n):
                        low = min(low,cost[idx][i] + dfs(idx + 1, groups + (1 if (i + 1) != prev_color else 0), i + 1))
                    memo[(idx, groups, prev_color)] = low
                    return memo[(idx, groups, prev_color)]
        ans = dfs(0, 0, -1)
        return -1 if ans == sys.maxsize else ans
from functools import lru_cache
import math

class Solution:
    def minCost(self, houses: List[int], cost: List[List[int]], m: int, n: int, target: int) -> int:
        @lru_cache(None)
        def paint(i, k, color):
            # print(i, k, color)
            if k == 1 and i == m:
                return 0
            if k == 0 or i == m:
                return math.inf
            total_cost = math.inf
            if houses[i] != 0:
                if houses[i] != color:
                    k -= 1
                return paint(i + 1, k, houses[i])
            for c in range(1, n + 1):
                kk = k
                if c != color:
                    kk -= 1
                cost_ = cost[i][c - 1] + paint(i + 1, kk, c)    
                total_cost = min(total_cost, cost_)
            # print(i, k, color, total_cost)
            return total_cost
        
        neighbors = 0
        prev = 0
        for h in houses:
            if h == 0: 
                continue
            if h != prev:
                neighbors += 1
                prev = h
        if neighbors > target:
            return -1
        res = paint(0, target + 1, -1)
        return res if res != math.inf else -1
class Solution:
    def minCost(self, houses: List[int], cost: List[List[int]], m: int, n: int, t: int) -> int:
        mem={}
        
        def recurse(i,lastc,target):
            if target<0:
                return float('inf')
            if i==m:
                return float('inf') if target!=0 else 0
            if (i,lastc,target) not in mem:
                if houses[i]>0:
                    mem[(i,lastc,target)]=recurse(i+1,houses[i],target-1 if lastc!=houses[i] else target)
                else:
                    if lastc>0:
                        mem[(i,lastc,target)]=recurse(i+1,lastc,target)+cost[i][lastc-1]
                    else:
                        mem[(i,lastc,target)]=float('inf')
                    for j in range(1,n+1):
                        if j!=lastc:
                            mem[(i,lastc,target)]=min(recurse(i+1,j,target-1)+cost[i][j-1],mem[(i,lastc,target)])
            return mem[(i,lastc,target)]
        
        result=recurse(0,0,t)
        # print(mem)
        if result==float('inf'):
            return -1
        return result
class Solution:
    def minCost(self, houses: List[int], cost: List[List[int]], m: int, n: int, target: int) -> int:
        dp={}
        def dfs(i, t, p):
            key = (i, t, p)
            
            if i == len(houses):
                return 0 if t == 0 else float('inf')
            if key not in dp:                
                if houses[i] == 0:
                    dp[key] = min(dfs(i+1, t-(p != nc), nc)+cost[i][nc-1] for nc in range(1, n+1))
                else:
                    dp[key] = dfs(i+1, t-(p != houses[i]), houses[i])
            return dp[key]
        ret =  dfs(0, target, -1)
        return -1 if  ret == float('inf') else ret

class Solution:
    def minCost(self, houses: List[int], cost: List[List[int]], m: int, n: int, target: int) -> int:
        '''
        houses = [0,0,0,0,0], cost = [[1,10],[10,1],[10,1],[1,10],[5,1]], m = 5, n = 2, target = 3
        Output: 9
        
        [0,2,1,2,0], cost = [[1,10],[10,1],[10,1],[1,10],[5,1]], m = 5, n = 2, target = 3
        
        if cur_t == target: same as pre
        if cur_t < target: new or same as pre
        
        dp(i, cur_t) = 
        if memo
        res = float('inf')
        if i == len(houses):
            if cur_t == target: return 0
            else: return res
        if cur_t > target: return res
        
        if houses[i] != 0:
            if i>0 and houses[i] == houses[i-1]:
                res = dp(i+1, cur_t)
            else: res = dp(i+1, cur_t+1)
        else:
            for color in range(1,n+1):
                if i>0 and color = houses[i-1]:
                    houses[i] = color
                    res = min(res, cost[i][color-1] + dp(i+1, cur_t))
                    houses[i] = 0
                else:
                    houses[i] = color
                    res = min(res, cost[i][color-1] + dp(i+1, cur_t+1))
                    houses[i] = 0
            
        
        
        '''
        memo = {}
        def dp(i, pre_col, cur_t):
            # print(i, cur_t)
            res = float('inf')
            if i == len(houses):
                if cur_t == 0: return 0
                else: return res
            if cur_t < 0: return res
            if (i, pre_col,cur_t) in memo.keys(): return memo[(i, pre_col,cur_t)]
            
            if houses[i] != 0:
                if i>0 and houses[i] == pre_col:
                    res = dp(i+1, pre_col, cur_t)
                else: res = dp(i+1, houses[i], cur_t-1)
            else:
                for color in range(1,n+1):
                    if i>0 and color == pre_col:
                        # houses[i] = color
                        res = min(res, cost[i][color-1] + dp(i+1, pre_col,cur_t))
                        # houses[i] = 0
                    else:
                        # houses[i] = color
                        res = min(res, cost[i][color-1] + dp(i+1,color, cur_t-1))
                        # houses[i] = 0
            memo[(i,pre_col, cur_t)] = res
            return res
        ans = dp(0, houses[0],target)
        return ans if ans != float('inf') else -1
class Solution:
    def minCost(self, A, cost, m, n, target):
        dp, dp2 = {(0, 0): 0}, {}
        for i, a in enumerate(A):
            for cj in (range(1, n + 1) if a == 0 else [a]):
                for ci, b in dp:
                    b2 = b + (ci != cj)
                    if b2 > target: continue
                    dp2[cj, b2] = min(dp2.get((cj,b2), float('inf')), dp[ci, b] + (cost[i][cj - 1] if cj != a else 0))
            dp, dp2 = dp2, {}
        return min([dp[c, b] for c, b in dp if b == target] or [-1])
class Solution:
    def minCost(self, houses: List[int], cost: List[List[int]], m: int, n: int, t: int) -> int:
        mem={}
        
        def recurse(i,lastc,target):
            if target<0:
                return float('inf')
            if i==m:
                return float('inf') if target!=0 else 0
            if houses[i]>0:
                return recurse(i+1,houses[i],target-1 if lastc!=houses[i] else target)
            if (i,lastc,target) not in mem:
                if lastc>0:
                    mem[(i,lastc,target)]=recurse(i+1,lastc,target)+cost[i][lastc-1]
                else:
                    mem[(i,lastc,target)]=float('inf')
                for j in range(1,n+1):
                    if j!=lastc:
                        mem[(i,lastc,target)]=min(recurse(i+1,j,target-1)+cost[i][j-1],mem[(i,lastc,target)])
            return mem[(i,lastc,target)]
        
        result=recurse(0,0,t)
        # print(mem)
        if result==float('inf'):
            return -1
        return result
class Solution:
    def minCost(self, houses: List[int], cost: List[List[int]], m: int, n: int, target: int) -> int:
        dp={}
        def dfs(i, t, p):
            key = (i, t, p)            
            if i == len(houses) or t<0:
                return 0 if t == 0 else float('inf')
            if key not in dp:                
                if houses[i] == 0:
                    dp[key] = min(dfs(i+1, t-(p != nc), nc)+cost[i][nc-1] for nc in range(1, n+1))
                else:
                    dp[key] = dfs(i+1, t-(p != houses[i]), houses[i])
            return dp[key]
        ret =  dfs(0, target, -1)
        return -1 if  ret == float('inf') else ret

dp = [[[0]*102 for j in range(23)] for i in range(102)]
def dfs(i,house,cost,prev,tar):
    if i>=len(house):
        if tar==0:
            return 0
        else:
            return 1000000000000
    if tar<0:
        return 1000000000000
    if(dp[i][prev][tar]>0):
        return dp[i][prev][tar]
    res = 1000000000000000000
    if house[i]==0:
        for j in range(len(cost[i])):
            res = min(res , cost[i][j] + dfs(i+1,house,cost,j+1,tar - ((j+1)!=prev)))
    else:
        res = min(res , dfs(i+1,house,cost,house[i],tar- (house[i]!=prev)))
    #print(i,prev,tar,res)
    dp[i][prev][tar]=res
    return dp[i][prev][tar]
class Solution:
    def minCost(self, house: List[int], cost: List[List[int]], m: int, n: int, tar: int) -> int:
        for i in range(101):
            for j in range(21):
                for k in range(101):
                    dp[i][j][k]=0
        res = dfs(0,house,cost,n+1,tar)
        #for i in dp:
        #    print(i)
        if res>=1000000000000:
            return -1
        else:
            return res
        

class Solution:
    def minCost(self, houses: List[int], cost: List[List[int]], m: int, n: int, target: int) -> int:
        @lru_cache(None)
        def min_cost_helper(i, prev_color, groups):
            if i == m:
                return 0 if groups == target else float('inf')
            
            if houses[i] != 0:
                return min_cost_helper(i + 1, houses[i], groups + int(prev_color != houses[i]))
            
            total = float('inf')
            for color in range(1, n + 1):
                total = min(total, cost[i][color - 1] + min_cost_helper(i + 1, color, groups + int(prev_color != color)))
            
            return total
        
        ans = min_cost_helper(0, -1, 0)
        return ans if ans != float('inf') else -1
class Solution:
    def minCost(self, houses: List[int], cost: List[List[int]], m: int, n: int, target: int) -> int:
        @lru_cache(None)
        def dp(i, g, p):
            if i == m:
                return 0 if g == 0 else float('inf')
            if m - i < g:
                return float('inf')
            if houses[i]:
                return dp(i + 1, g - (p != houses[i]), houses[i])
            else:
                return min(dp(i + 1, g - (p != nc), nc) + cost[i][nc - 1] for nc in range(1, n + 1))
        
        ret = dp(0, target, -1)
        return ret if ret != float('inf') else -1
            
        
        
#         dp = [[[float('inf') for _ in range(target + 1)] for _ in range(1 + n)] for _ in range(m)]
#         if houses[0] != 0:
#             dp[0][houses[0]][1] = 0
#         else:
#             for i in range(1, n + 1):
#                 dp[0][i][1] = cost[0][i - 1]
        
#         for house in range(1, m):
#             if houses[house] > 0:
#                 for neigh in range(1, target + 1):
#                     c1= houses[house]
#                     dp[house][c1][neigh] = min(min(dp[house - 1][c2][neigh - 1] for c2 in range(1, n + 1) if c2 != c1), dp[house - 1][c1][neigh])
#                 continue
#             for c1 in range(1, n + 1):
#                 for neigh in range(1, target + 1):
#                     for c2 in range(1, n + 1):
#                         if c1 == c2:
#                             dp[house][c1][neigh] = min(dp[house][c1][neigh], dp[house - 1][c2][neigh] + cost[house][c1 - 1])
#                         else:
#                             dp[house][c1][neigh] = min(dp[house][c1][neigh], dp[house - 1][c2][neigh - 1] + cost[house][c1 - 1])
#         ans = min(k[target] for k in dp[-1])  
#         return ans if ans != float('inf') else -1

class Solution:
    def minCost(self, houses: List[int], cost: List[List[int]], m: int, n: int, target: int) -> int:
        m_ = {}
        
        def dp(i, t, color):
            key =  (i, t, color)
            if key in m_: return m_[key]
            if t == 0 and i == len(houses): return 0
            if t < 0 or t > m - i: return float('inf')
            if houses[i] == 0:
                m_[key] = min(dp(i + 1, t - (c != color), c) + cost[i][c - 1] for c in range(1, n + 1))
            else:
                m_[key] = dp(i + 1, t - (houses[i] != color), houses[i])
            
            return m_[key]
        
        ans = dp(0, target, -1)
        
        return ans if ans < float('inf') else -1
from functools import lru_cache
class Solution:
    def minCost(self, houses: List[int], cost: List[List[int]], m: int, n: int, target: int) -> int:
        @lru_cache(None)
        def dp(g, i, c):
            if g > target or target - g > m - i: return sys.maxsize
            if i == m: return 0 if g == target else sys.maxsize 
            ans = sys.maxsize 
            if houses[i] != 0:
                ans = min(ans, dp(g + (houses[i]!=c), i+1, houses[i]))
            else:
                for j in range(n):
                    ans = min(ans, cost[i][j] + dp(g + (j+1!=c), i+1, j+1))
            return ans
        
        ans = dp(0, 0, 0)
        return ans if ans < sys.maxsize else -1

class Solution:
    def minCost(self, houses: List[int], cost: List[List[int]], m: int, n: int, target: int) -> int:
        if not houses: return 0
        size = len(houses)
        memo = dict()
        def dfs(index, t, p):
            key = (index, t, p)
            if key in memo: return memo[key]
            if index == size or t > size - index or t < 0:
                if index == size and t == 0: return 0
                else: return float('inf')
            temp = float('inf')
            if houses[index] == 0:
                for nc in range(1, n + 1):
                    temp = min(temp, dfs(index + 1, t - (nc != p), nc) + cost[index][nc - 1])
            else:
                temp = dfs(index + 1, t - (p != houses[index]), houses[index])
            memo[key] = temp
            return temp
                
        res = dfs(0, target, -1)
        return res if res < float('inf') else -1

class Solution:
    def minCost(self, houses: List[int], cost: List[List[int]], m: int, n: int, target: int) -> int:
        memo = {}
        def dfs(i, j, target):
            if target < 0 or target > m-i:
                return float('inf')
            if i == m:
                return 0 if target == 0 else float('inf')
            if (i, j, target) not in memo:
                if houses[i]:
                    memo[i,j,target] = dfs(i+1, houses[i], target-(houses[i]!=j))
                else:
                    memo[i,j,target] = min(cost[i][a-1] + dfs(i+1, a, target-(a!=j)) for a in range(1, n+1))
            return memo[i,j,target]
        ans = dfs(0, 0, target)
        return ans if ans < float('inf') else -1
class Solution:
    def minCost(self, houses: List[int], cost: List[List[int]], m: int, n: int, target: int) -> int:
        dp = {}
        def dfs(i, t, p):
            key = (i,t,p)
            if i == len(houses) or t < 0 or m-i < t:
                return 0 if t == 0 and i == m else float('inf')
            
            if key not in dp:
                if houses[i] == 0:
                    dp[key] = min(dfs(i+1, t-(nc!=p), nc) + cost[i][nc-1] for nc in range(1, n+1))
                else:
                    dp[key] = dfs(i+1, t-(houses[i]!=p), houses[i])
                    
            return dp[key]
        
        ret = dfs(0, target, -1)
            
        return ret if ret < float('inf') else -1
            

class Solution:
    def minCost(self, houses: List[int], cost: List[List[int]], m: int, n: int, target: int) -> int:
        dp={}
        def dfs(i, t, p):
            key = (i, t, p)            
            if i == len(houses) or t<0 or m-i<t:
                return 0 if t == 0 else float('inf')
            if key not in dp:                
                if houses[i] == 0:
                    dp[key] = min(dfs(i+1, t-(p != nc), nc)+cost[i][nc-1] for nc in range(1, n+1))
                else:
                    dp[key] = dfs(i+1, t-(p != houses[i]), houses[i])
            return dp[key]
        ret =  dfs(0, target, -1)
        return -1 if  ret == float('inf') else ret

class Solution:
    def minCost(self, houses: List[int], cost: List[List[int]], m: int, n: int, target: int) -> int:
        
        memo = {}
        
        def dfs(i, b ,c):
            
            if i == m and b == target:
                return 0
            if m - i < target - b or i == m or b > target:
                return float('inf')
            
            key = (i, b, c)
            if key not in memo:
                if houses[i] != 0:
                    memo[key] = dfs(i+1, b + (houses[i] != c), houses[i])
                else:
                    memo[key] = min( dfs(i+1, b + (nc != c), nc) + cost[i][nc-1] for nc in range(1, n+1))
            
            return memo[key]
        
        res = dfs(0, 0, -1)
        if res == float('inf'):
            return -1
        else:
            return res
class Solution:
    def minCost(self, houses: List[int], cost: List[List[int]], m: int, n: int, target: int) -> int:
        dp = {}
        # def func(i,t,p):
        #     key = (i, t, p)
        #     if i == len(houses) or t < 0 or m - i < t :
        #         return 0 if t == 0 and i == len(houses) else float('inf')
        #     if key not in dp:
        #         if houses[i]==0:
        #             dp[key] = min(func(i + 1, t - (nc != p), nc) + cost[i][nc - 1] for nc in range(n+1))
        #         else:
        #             dp[key] = func(i + 1, t - (houses[i]!=p), houses[i])
        #     return dp[key]
        # ret = func(0, target, -1)
        # return ret if ret < float('inf') else -1
        def dfs(i, t, p):
            key = (i, t, p)
            if i == len(houses) or t < 0 or m - i < t:
                return 0 if t == 0 and i == len(houses) else float('inf')
            if key not in dp:
                if houses[i] == 0:
                    dp[key] = min(dfs(i + 1, t - (nc != p), nc) + cost[i][nc - 1] for nc in range(1, n + 1))
                else:
                    dp[key] = dfs(i + 1, t - (houses[i] != p), houses[i])
            return dp[key]
        ret = dfs(0, target, -1)
        return ret if ret < float('inf') else -1
class Solution:
    def minCost(self, houses: List[int], cost: List[List[int]], m: int, n: int, target: int) -> int:
        dp={}
        def dfs(i, t, p):
            key = (i, t, p)            
            if i == len(houses) or t<0 or m-i<t:
                return 0 if t == 0 else float('inf')
            if key not in dp:      
                if houses[i] == 0:
                    dp[key] = min(dfs(i+1, t-(p != nc), nc)+cost[i][nc-1] for nc in range(1, n+1))
                else:
                    dp[key] = dfs(i+1, t-(p != houses[i]), houses[i])
            return dp[key]
        ret =  dfs(0, target, 0)
        return -1 if  ret == float('inf') else ret

import numpy as np
from collections import defaultdict

class Solution:
    def minCost(self, houses: List[int], cost: List[List[int]], m: int, n: int, target: int) -> int:
        
        # DP(number of painted houses from left, last house color, number of groups in painted houses)
        # DP(i, color, groups)
        #       = min(DP(i - 1, x != color, groups - 1), DP(i - 1, color, groups)) + cost[i][x]

        # m houses <= 100
        # n colors <= 20
        # number of states = m * n * target = 100 * 20 * 100 = 2e5
        
        INF = int(1e9)
        DP = defaultdict(lambda: INF)  # (groups, last_color) -> min cost

        if houses[0] == 0:
            for col in range(1, n + 1):
                DP[(1, col)] = cost[0][col - 1]
        else:
            DP[(1, houses[0])] = 0
        
        for i in range(1, m):
            NDP = defaultdict(lambda: INF)
            
            ByGroups = defaultdict(list)  # (groups) -> [(cost, prev color)]
            for key, min_cost in DP.items():
                groups_prev, col_prev = key
                ByGroups[groups_prev].append((min_cost, col_prev))

            curr_colors = range(1, n + 1) if houses[i] == 0 else [houses[i]]
                
            for groups_prev, prev_colors_array in ByGroups.items():
                prev_colors_array.sort()
                prev_colors = [col_prev for min_cost, col_prev in prev_colors_array[:2]]
                
                for col_curr in curr_colors:
                    paint_cost = cost[i][col_curr - 1] if houses[i] == 0 else 0

                    for col_prev in [col_curr] + prev_colors:
                        groups_curr = groups_prev + (col_prev != col_curr)
                        curr_cost = DP[(groups_prev, col_prev)] + paint_cost
                        if groups_curr <= target and curr_cost < INF:
                            key = (groups_curr, col_curr)
                            NDP[key] = min(NDP[key], curr_cost)
                            
            DP = NDP
        
        ans = -1
        for key, min_cost in DP.items():
            if key[0] == target and (ans < 0 or ans > min_cost):
                ans = min_cost
        return ans
from functools import lru_cache
class Solution:
    def minCost(self, houses: List[int], cost: List[List[int]], m: int, n: int, target: int) -> int:
        @lru_cache(None)
        def dp(g, i, c):
            if g > target or target - g > m - i: return sys.maxsize
            if i == m: return 0 if g == target else sys.maxsize 
            ans = sys.maxsize 
            if houses[i] != 0:
                ans = min(ans, dp(g+(houses[i]!=c), i+1, houses[i]))
            else:
                for j in range(n):
                    ans = min(ans, cost[i][j] + dp(g+(j+1!=c), i+1, j+1))
            return ans
        
        ans = dp(0, 0, 0)
        return -1 if ans >= sys.maxsize else ans

class Solution:
    def onTrack(self,neighbours, visited, m, target):
        return neighbours <= target and neighbours + m - visited >= target
        
    def minCost(self, houses, cost, m: int, n: int, target: int) -> int:
        dp = {}
        if not self.onTrack(1, 1, m, target):
            return -1
        if houses[0]!=0:
            dp[houses[0]] = {1:0}
        else:
            for color in range(1,n+1):
                dp[color] = {1:cost[0][color-1]}
        # Start iteration
        for each in range(1,m):
            new = {}
            if houses[each]!=0:
                colors = list(range(houses[each], houses[each]+1))
            else:
                colors = list(range(1, n+1))
            for new_color in colors:
                for color in list(dp.keys()):
                    isNew = int(color!=new_color)
                    for neighbours in list(dp[color].keys()):
                        if self.onTrack(neighbours+isNew, each+1, m, target):
                            if new_color not in list(new.keys()):
                                new[new_color] = {}
                            new_cost = dp[color][neighbours]+cost[each][new_color-1]*int(houses[each]==0)
                            last = new_cost
                            if neighbours+isNew in list(new[new_color].keys()):
                                last = min(new_cost, new[new_color][neighbours+isNew])
                            new[new_color][neighbours+isNew]=last
            if not new:
                return -1
            dp = new
        result = float('inf')
        for color in list(dp.keys()):
            if target in list(dp[color].keys()):
                result = min(result, dp[color][target])
        if result!=float('inf'):
            return result
        else:
            return -1

class Solution:
    def minCost(self, houses: List[int], cost: List[List[int]], m: int, n: int, target: int) -> int:
        memo = {}
        def dfs(i, j, target):
            if target < 0 or target > m-i:
                return float('inf')
            if i == m:
                if target == 0:
                    return 0
                else:
                    return float('inf')
            if (i, j, target) not in memo:
                if houses[i]:
                    memo[i,j,target] = dfs(i+1, houses[i], target-(houses[i]!=j))
                else:
                    memo[i,j,target] = min(cost[i][k-1] + dfs(i+1, k, target-(k!=j)) for k in range(1, n+1))
            return memo[i,j,target]
        ans = dfs(0, 0, target)
        return ans if ans < float('inf') else -1
class Solution:
    def minCost(self, houses: List[int], cost: List[List[int]], m: int, n: int, target: int) -> int:
        memo = {}
        def dfs(i, k, target):
            if target < 0 or target > m-i:
                return float('inf')
            if i == m:
                return 0 if target == 0 else float('inf')
            if (i, k, target) not in memo:
                if houses[i]:
                    memo[i,k,target] = dfs(i+1, houses[i], target-(houses[i]!=k))
                else:
                    memo[i,k,target] = min(cost[i][a-1] + dfs(i+1, a, target-(a!=k)) for a in range(1, n+1))
            return memo[i,k,target]
        ans = dfs(0, 0, target)
        return ans if ans < float('inf') else -1
class Solution:
    def minCost(self, houses: List[int], cost: List[List[int]], m: int, n: int, target: int) -> int:
        
        # top-down approach
        # key: (i, t, c) -> cost for paint i house left t group with c color
        
        cache = {}
        
        # dfs search
        def dfs(i, t, c):
            key = (i, t, c)
            
            # boundary condition
            # 1. we paint all house and no target left
            # 2. t must in  0 < t < m - i  
            if i == len(houses) or t < 0 or m - i < t:
                return 0 if i == len(houses) and t == 0 else float('inf')
            
            if key not in cache:
                print(key)
                if houses[i] == 0: 
                    # compte 
                    cache[key] = min(dfs(i + 1, t - int(nc != c), nc) + cost[i][nc - 1]
                                   for nc in range(1, n + 1)) 
                else:
                    # if already paint no extra cost add
                    cache[key] = dfs(i + 1, t - int(c != houses[i]), houses[i])
                    
                
            return cache[key]
        
        result = dfs(0, target, 0)
        print(cache.keys())
        
        return result if result < float('inf') else -1
from collections import defaultdict
class Solution:
    def minCost(self, houses: List[int], cost: List[List[int]], m: int, n: int, target: int) -> int:
        '''
        
        m houses - n colors (1~n)
        neighborhoods of same color
        0 = not colored yet
        cost[i][j] - cost to paint i with color j+1
        
        return minimal cost of painting remaining houses to get exactly target neighborhoods
        
        -have to keep track of min cost
        -num neighbors==target at end
        
        dp => cost,numNeighbors
        dp[house][lastneighbor] = min(dp[house][lastneighbor], min(cost + dp[house-1][neighbors])
        
        best now = last house with same color, correct num of neighbors or last house diff color, 1 less num of neighbors
        
        dp[h][numNeighbors][color] = colorCost + min(
                dp[h-1][numNeighbors][color],
                dp[h-1][numNeighbors-1][diffColor] <- iterate
            )
        
        at house h,
        dp[neighbors][color] = colorCost (if it's 0)
                                + prev[neighbors][color]
                                + prev[neighbors-1][anothercolor]
        neighbors: [j-target,j+1]
        
        edge cases:
        -if # of color neighborhoods > target: return -1
        -num houses < target: return -1
        '''
        prev = [[0]*n for _ in range(target+1)]
        for h,house in enumerate(houses):
            dp = [[float('inf')]*n for _ in range(target+1)]
            for numNeighbors in range(1,min(h+2,target+1)):
                if house==0:
                    for color in range(n):
                        colorCost = cost[h][color]
                        dp[numNeighbors][color] = min(
                            dp[numNeighbors][color],
                            colorCost + prev[numNeighbors][color],
                            colorCost + min(prev[numNeighbors-1][c] for c in range(n) if c!=color)
                        )
                else:
                    color = house-1
                    dp[numNeighbors][color] = min(
                        dp[numNeighbors][color],
                        prev[numNeighbors][color],
                        min(prev[numNeighbors-1][c] for c in range(n) if c!=color)
                    )
            prev = dp
        out = min(prev[target][c] for c in range(n)) if houses[-1]==0 else prev[target][houses[-1]-1]
        return out if out!=float('inf') else -1
                

from collections import defaultdict
class Solution:
    def minCost(self, houses: List[int], cost: List[List[int]], m: int, n: int, target: int) -> int:
        '''
        
        m houses - n colors (1~n)
        neighborhoods of same color
        0 = not colored yet
        cost[i][j] - cost to paint i with color j+1
        
        return minimal cost of painting remaining houses to get exactly target neighborhoods
        
        -have to keep track of min cost
        -num neighbors==target at end
        
        dp => cost,numNeighbors
        dp[house][lastneighbor] = min(dp[house][lastneighbor], min(cost + dp[house-1][neighbors])
        
        best now = last house with same color, correct num of neighbors or last house diff color, 1 less num of neighbors
        
        dp[h][numNeighbors][color] = colorCost + min(
                dp[h-1][numNeighbors][color],
                dp[h-1][numNeighbors-1][diffColor] <- iterate
            )
        
        at house h,
        dp[neighbors][color] = colorCost (if it's 0)
                                + prev[neighbors][color]
                                + prev[neighbors-1][anothercolor]
        neighbors: [j-target,j+1]
        
        edge cases:
        -if # of color neighborhoods > target: return -1
        -num houses < target: return -1
        '''
        prev = [[0]*n for _ in range(2)]
        for h,house in enumerate(houses):
            dp = [[float('inf')]*n for _ in range(h+2)]
            for numNeighbors in range(1,h+2):
                if house==0:
                    for color in range(n):
                        colorCost = cost[h][color]
                        if numNeighbors==h+1:
                            dp[numNeighbors][color] = min(
                                dp[numNeighbors][color],
                                colorCost + min(prev[numNeighbors-1][c] for c in range(n) if c!=color)
                            )
                        else:
                            dp[numNeighbors][color] = min(
                                dp[numNeighbors][color],
                                colorCost + prev[numNeighbors][color],
                                colorCost + min(prev[numNeighbors-1][c] for c in range(n) if c!=color)
                            )
                else:
                    color = house-1
                    if numNeighbors==h+1:
                        dp[numNeighbors][color] = min(
                            dp[numNeighbors][color],
                            min(prev[numNeighbors-1][c] for c in range(n) if c!=color)
                        )

                    else:
                        dp[numNeighbors][color] = min(
                            dp[numNeighbors][color],
                            prev[numNeighbors][color],
                            min(prev[numNeighbors-1][c] for c in range(n) if c!=color)
                        )
            prev = dp
        
        out = min(prev[target][c] for c in range(n))
        return out if out!=float('inf') else -1
                

from collections import defaultdict
class Solution:
    def minCost(self, houses: List[int], cost: List[List[int]], m: int, n: int, target: int) -> int:
        '''
        
        m houses - n colors (1~n)
        neighborhoods of same color
        0 = not colored yet
        cost[i][j] - cost to paint i with color j+1
        
        return minimal cost of painting remaining houses to get exactly target neighborhoods
        
        -have to keep track of min cost
        -num neighbors==target at end
        
        dp => cost,numNeighbors
        dp[house][lastneighbor] = min(dp[house][lastneighbor], min(cost + dp[house-1][neighbors])
        
        best now = last house with same color, correct num of neighbors or last house diff color, 1 less num of neighbors
        
        dp[h][numNeighbors][color] = colorCost + min(
                dp[h-1][numNeighbors][color],
                dp[h-1][numNeighbors-1][diffColor] <- iterate
            )
        
        at house h,
        dp[neighbors][color] = colorCost (if it's 0)
                                + prev[neighbors][color]
                                + prev[neighbors-1][anothercolor]
        neighbors: [j-target,j+1]
        
        edge cases:
        -if # of color neighborhoods > target: return -1
        -num houses < target: return -1
        '''
        prev = [[0]*n for _ in range(m+1)]
        for h,house in enumerate(houses):
            dp = [[float('inf')]*n for _ in range(m+1)]
            for numNeighbors in range(1,h+2):
                if house==0:
                    for color in range(n):
                        colorCost = cost[h][color]
                        dp[numNeighbors][color] = min(
                            dp[numNeighbors][color],
                            colorCost + prev[numNeighbors][color],
                            colorCost + min(prev[numNeighbors-1][c] for c in range(n) if c!=color)
                        )
                else:
                    color = house-1
                    dp[numNeighbors][color] = min(
                        dp[numNeighbors][color],
                        prev[numNeighbors][color],
                        min(prev[numNeighbors-1][c] for c in range(n) if c!=color)
                    )
            prev = dp
        out = min(prev[target][c] for c in range(n)) if houses[-1]==0 else prev[target][houses[-1]-1]
        return out if out!=float('inf') else -1
                

from collections import defaultdict
class Solution:
    def minCost(self, houses: List[int], cost: List[List[int]], m: int, n: int, target: int) -> int:
        '''
        
        m houses - n colors (1~n)
        neighborhoods of same color
        0 = not colored yet
        cost[i][j] - cost to paint i with color j+1
        
        return minimal cost of painting remaining houses to get exactly target neighborhoods
        
        -have to keep track of min cost
        -num neighbors==target at end
        
        dp => cost,numNeighbors
        dp[house][lastneighbor] = min(dp[house][lastneighbor], min(cost + dp[house-1][neighbors])
        
        best now = last house with same color, correct num of neighbors or last house diff color, 1 less num of neighbors
        
        dp[h][numNeighbors][color] = colorCost + min(
                dp[h-1][numNeighbors][color],
                dp[h-1][numNeighbors-1][diffColor] <- iterate
            )
        
        at house h,
        dp[neighbors][color] = colorCost (if it's 0)
                                + prev[neighbors][color]
                                + prev[neighbors-1][anothercolor]
        neighbors: [j-target,j+1]
        
        edge cases:
        -if # of color neighborhoods > target: return -1
        -num houses < target: return -1
        '''
        prev = [[0]*n for _ in range(target+1)]
        out = float('inf')
        for h,house in enumerate(houses):
            dp = [[float('inf')]*n for _ in range(target+1)]
            for numNeighbors in range(1,min(h+2,target+1)):
                if house==0:
                    for color in range(n):
                        colorCost = cost[h][color]
                        dp[numNeighbors][color] = min(
                            dp[numNeighbors][color],
                            colorCost + prev[numNeighbors][color],
                            colorCost + min(prev[numNeighbors-1][c] for c in range(n) if c!=color)
                        )
                else:
                    color = house-1
                    dp[numNeighbors][color] = min(
                        dp[numNeighbors][color],
                        prev[numNeighbors][color],
                        min(prev[numNeighbors-1][c] for c in range(n) if c!=color)
                    )
            prev = dp
        
        out = None
        if houses[-1]==0:
            out = min(prev[target][c] for c in range(n))
        else:
            out = prev[target][houses[-1]-1]
        return out if out!=float('inf') else -1
                

from collections import defaultdict
class Solution:
    def minCost(self, houses: List[int], cost: List[List[int]], m: int, n: int, target: int) -> int:
        '''
        
        m houses - n colors (1~n)
        neighborhoods of same color
        0 = not colored yet
        cost[i][j] - cost to paint i with color j+1
        
        return minimal cost of painting remaining houses to get exactly target neighborhoods
        
        -have to keep track of min cost
        -num neighbors==target at end
        
        dp => cost,numNeighbors
        dp[house][lastneighbor] = min(dp[house][lastneighbor], min(cost + dp[house-1][neighbors])
        
        best now = last house with same color, correct num of neighbors or last house diff color, 1 less num of neighbors
        
        dp[h][numNeighbors][color] = colorCost + min(
                dp[h-1][numNeighbors][color],
                dp[h-1][numNeighbors-1][diffColor] <- iterate
            )
        
        at house h,
        dp[neighbors][color] = colorCost (if it's 0)
                                + prev[neighbors][color]
                                + prev[neighbors-1][anothercolor]
        neighbors: [j-target,j+1]
        
        edge cases:
        -if # of color neighborhoods > target: return -1
        -num houses < target: return -1
        '''
        prev = [[0]*n for _ in range(target+1)]
        out = float('inf')
        for h,house in enumerate(houses):
            dp = [[float('inf')]*n for _ in range(target+1)]
            for numNeighbors in range(1,min(h+2,target+1)):
                if house==0:
                    for color in range(n):
                        colorCost = cost[h][color]
                        dp[numNeighbors][color] = min(
                            dp[numNeighbors][color],
                            colorCost + prev[numNeighbors][color],
                            colorCost + min(prev[numNeighbors-1][c] for c in range(n) if c!=color)
                        )
                        if numNeighbors==target and h==m-1:
                            # print(h,numNeighbors,color,out,dp[numNeighbors][color])
                            out = min(out,dp[numNeighbors][color])
                else:
                    color = house-1
                    dp[numNeighbors][color] = min(
                        dp[numNeighbors][color],
                        prev[numNeighbors][color],
                        min(prev[numNeighbors-1][c] for c in range(n) if c!=color)
                    )
                    if numNeighbors==target and h==m-1:
                        out = min(out,dp[numNeighbors][color])
            prev = dp
        
        # out = None
        # if houses[-1]==0:
        #     out = min(prev[target][c] for c in range(n))
        # else:
        #     prev[target][houses[-1]-1]
        return out if out!=float('inf') else -1
                

from collections import defaultdict
class Solution:
    def minCost(self, houses: List[int], cost: List[List[int]], m: int, n: int, target: int) -> int:
        '''
        
        m houses - n colors (1~n)
        neighborhoods of same color
        0 = not colored yet
        cost[i][j] - cost to paint i with color j+1
        
        return minimal cost of painting remaining houses to get exactly target neighborhoods
        
        -have to keep track of min cost
        -num neighbors==target at end
        
        dp => cost,numNeighbors
        dp[house][lastneighbor] = min(dp[house][lastneighbor], min(cost + dp[house-1][neighbors])
        
        best now = last house with same color, correct num of neighbors or last house diff color, 1 less num of neighbors
        
        dp[h][numNeighbors][color] = colorCost + min(
                dp[h-1][numNeighbors][color],
                dp[h-1][numNeighbors-1][diffColor] <- iterate
            )
        
        at house h,
        dp[neighbors][color] = colorCost (if it's 0)
                                + prev[neighbors][color]
                                + prev[neighbors-1][anothercolor]
        neighbors: [j-target,j+1]
        
        edge cases:
        -if # of color neighborhoods > target: return -1
        -num houses < target: return -1
        '''
        prev = [[0]*n for _ in range(2)]
        for h,house in enumerate(houses):
            dp = [[float('inf')]*n for _ in range(h+2)]
            for numNeighbors in range(1,h+2):
                if house==0:
                    for color in range(n):
                        colorCost = cost[h][color]
                        if numNeighbors==h+1:
                            dp[numNeighbors][color] = min(
                                dp[numNeighbors][color],
                                colorCost + min(prev[numNeighbors-1][c] for c in range(n) if c!=color)
                            )
                        else:
                            dp[numNeighbors][color] = min(
                                dp[numNeighbors][color],
                                colorCost + prev[numNeighbors][color],
                                colorCost + min(prev[numNeighbors-1][c] for c in range(n) if c!=color)
                            )
                            
                    
                else:
                    color = house-1
                    if numNeighbors==h+1:
                        dp[numNeighbors][color] = min(
                            dp[numNeighbors][color],
                            min(prev[numNeighbors-1][c] for c in range(n) if c!=color)
                        )

                    else:
                        # print(numNeighbors,color)
                        # print(len(dp),len(dp[0]))
                        dp[numNeighbors][color] = min(
                            dp[numNeighbors][color],
                            prev[numNeighbors][color],
                            min(prev[numNeighbors-1][c] for c in range(n) if c!=color)
                        )
            prev = dp
        
        out = min(prev[target][c] for c in range(n))
        return out if out!=float('inf') else -1
                

from collections import defaultdict
class Solution:
    def minCost(self, houses: List[int], cost: List[List[int]], m: int, n: int, target: int) -> int:
        '''
        
        m houses - n colors (1~n)
        neighborhoods of same color
        0 = not colored yet
        cost[i][j] - cost to paint i with color j+1
        
        return minimal cost of painting remaining houses to get exactly target neighborhoods
        
        -have to keep track of min cost
        -num neighbors==target at end
        
        dp => cost,numNeighbors
        dp[house][lastneighbor] = min(dp[house][lastneighbor], min(cost + dp[house-1][neighbors])
        
        best now = last house with same color, correct num of neighbors or last house diff color, 1 less num of neighbors
        
        dp[h][numNeighbors][color] = colorCost + min(
                dp[h-1][numNeighbors][color],
                dp[h-1][numNeighbors-1][diffColor] <- iterate
            )
        
        at house h,
        dp[neighbors][color] = colorCost (if it's 0)
                                + prev[neighbors][color]
                                + prev[neighbors-1][anothercolor]
        neighbors: [j-target,j+1]
        
        edge cases:
        -if # of color neighborhoods > target: return -1
        -num houses < target: return -1
        '''
        prev = [[0]*n for _ in range(m+1)]
        for h,house in enumerate(houses):
            dp = [[float('inf')]*n for _ in range(m+1)]
            for numNeighbors in range(1,h+2):
                if house==0:
                    for color in range(n):
                        colorCost = cost[h][color]
                        dp[numNeighbors][color] = min(
                            dp[numNeighbors][color],
                            colorCost + prev[numNeighbors][color],
                            colorCost + min(prev[numNeighbors-1][c] for c in range(n) if c!=color)
                        )
                else:
                    color = house-1
                    dp[numNeighbors][color] = min(
                        dp[numNeighbors][color],
                        prev[numNeighbors][color],
                        min(prev[numNeighbors-1][c] for c in range(n) if c!=color)
                    )
            prev = dp
        
        out = min(prev[target][c] for c in range(n))
        return out if out!=float('inf') else -1
                

class Solution:
    def minCost(self, houses: List[int], cost: List[List[int]], m: int, n: int, target: int) -> int:
        # dp[i][j][k] is the minimum cost to paint {0, 1, ..., `i`} houses of `j` neighborhoods, and the `i`th house is painted as color `k`
        
        dp = [[[math.inf for k in range(n)] for j in range(target)] for i in range(m)]
        
        pre = None
        min_neighborhoods = 1
        for i in range(m):
            if houses[i]:
                if pre is not None and houses[i] != pre:
                    min_neighborhoods += 1
                pre = houses[i]
            if i == 0:
                if houses[i]:
                    k = houses[i]-1
                    dp[i][0][k] = 0
                else:
                    for k in range(n):
                        dp[i][0][k] = cost[i][k]
                continue
            for j in range(min_neighborhoods-1, min(i+1, target)):
                if houses[i]:
                    k = houses[i]-1
                    dp[i][j][k] = min(min(dp[i-1][j-1][p] for p in range(n) if p != k), dp[i-1][j][k])
                else:
                    for k in range(n):
                        dp[i][j][k] = cost[i][k] + min(min(dp[i-1][j-1][p] for p in range(n) if p != k), dp[i-1][j][k])

        ans = min(dp[-1][-1])
        if ans == math.inf:
            return -1
        else:
            return ans
class Solution:
    def minCost(self, houses: List[int], cost: List[List[int]], m: int, n: int, target: int) -> int:
        # dp[i][j][k] is the minimum cost to paint {0, 1, ..., `i`} houses of `j` neighborhoods, and the `i`th house is painted as color `k`
        
        dp = [[[math.inf for k in range(n)] for j in range(target)] for i in range(m)]
        if houses[0]:
            dp[0][0][houses[0]-1] = 0
        else:
            for k in range(n):
                dp[0][0][k] = cost[0][k]
        
        pre = houses[0]
        min_neighborhoods = 1
        for i in range(1, m):
            if houses[i]:
                if pre and houses[i] != pre:
                    min_neighborhoods += 1
                pre = houses[i]
            for j in range(min_neighborhoods-1, min(i+1, target)):
                for k in (houses[i]-1,) if houses[i] else range(n):
                    dp[i][j][k] = min(min(dp[i-1][j-1][p] for p in range(n) if p != k), dp[i-1][j][k])
                    if not houses[i]:
                        dp[i][j][k] += cost[i][k]
                        
        ans = min(dp[-1][-1])
        return -1 if ans == math.inf else ans
class Solution:
    def minCost(self, houses: List[int], cost: List[List[int]], m: int, n: int, target: int) -> int:
        

        @lru_cache(None)
        def dp(idx: int, pc: int, k: int) -> int:
            if idx >= m:
                return 0 - int(k != 0)
            if k < 0:
                return -1
            if houses[idx] != 0:
                return dp(idx+1, houses[idx], k - int(pc != houses[idx]))
            
            ans = -1
            for i in range(1, n+1):
                new_cost = cost[idx][i-1]
                prev_cost = dp(idx+1, i, k - int(pc!=i))
                if prev_cost >= 0:
                    ans = min(ans if ans >0 else float('inf'), new_cost + prev_cost)
            
            return ans
        
        return dp(0, 0, target)
    
#         @lru_cache(None)
#         def dp(idx: int, prev_color: int, k: int) -> int:

#             # No more houses left to paint, return true if there are no more neighborhoods
#             # to paint, false otherwise
#             if idx >= m:
#                 return 0 - int(k != 0)

#             # No more neighborhoods colors available
#             if k < 0:
#                 return -1

#             # Check if this house is already painted, if so, go to the next house
#             # Note: `k - int(prev_color != new_color)` decreases the number of
#             # neighborhoods left to paint if the current house color is different than the
#             # previous one
#             if houses[idx] != 0:
#                 return dp(idx + 1, houses[idx], k - int(prev_color != houses[idx]))

#             # Decide on the best color to paint current house
#             best = -1

#             # Try all possible colors
#             for new_color in range(1, n + 1):
#                 new_color_cost = cost[idx][new_color - 1]
#                 other_costs = dp(idx + 1, new_color, k - int(prev_color != new_color))

#                 # Check if painting this house with `new_color` will give us a lower cost
#                 if other_costs >= 0:
#                     best = min(best if best > 0 else float("inf"),
#                                new_color_cost + other_costs)
#             return best

#         # Start with the first house
#         return dp(0, 0, target)

class Solution:
    def minCost(self, houses: List[int], cost: List[List[int]], m: int, n: int, target: int) -> int:
        m = len(houses)
        n += 1
        @lru_cache(None)
        def minCost(i, n_nei, prev_c):
            if i == m:
                if n_nei == target:
                    return 0
                return float('inf')
            if n_nei > target:
                return float('inf')
            if houses[i] != 0:
                if houses[i] == prev_c:
                    return minCost(i+1, n_nei, prev_c)
                return minCost(i+1, n_nei+1, houses[i])
            return min(minCost(i+1, n_nei if c==prev_c else (n_nei+1), c)+cost[i][c-1] for c in range(1, n))
        if houses[0] == 0:
            res = min(minCost(1, 1, c) + cost[0][c-1] for c in range(1, n))
        else:
            res = minCost(1, 1, houses[0])
        return res if res != float('inf') else -1
import functools

class Solution:
    def minCost(self, houses: List[int], cost: List[List[int]], m: int, n: int, target: int) -> int:
        
        @functools.lru_cache(None)
        def dfs(i, last_color, hoods):
            # print(i, last_color, hoods)
            if hoods > target: return float('inf')
            
            if i >= m: 
                if hoods != target: return float('inf')
                return 0
            
            if houses[i] != 0:
                if houses[i]-1 == last_color:
                    return dfs(i+1, houses[i]-1, hoods)
                else:
                    return dfs(i+1, houses[i]-1, hoods + 1)
            else:
                cands = []
                for color in range(n):
                    if color == last_color:
                        cands.append(cost[i][color] + dfs(i+1, color, hoods))
                    else:
                        cands.append(cost[i][color] + dfs(i+1, color, hoods + 1))
                return min(cands)
                
        
        ans = dfs(0, -1, 0)
        
        if ans == float('inf'): return -1
        return ans
class Solution:
    def minCost(self, houses: List[int], cost: List[List[int]], m: int, n: int, target: int) -> int:
        

        @lru_cache(None)
        def dp(idx, pc, k):
            if idx >= m:
                return 0 - int(k != 0)
            if k < 0:
                return -1
            if houses[idx] != 0:
                return dp(idx+1, houses[idx], k - int(pc != houses[idx]))
            
            ans = -1
            for i in range(1, n+1):
                new_cost = cost[idx][i-1]
                prev_cost = dp(idx+1, i, k - int(pc!=i))
                if prev_cost >= 0:
                    ans = min(ans if ans >0 else float('inf'), new_cost + prev_cost)
            
            return ans
        
        return dp(0, 0, target)
    
#         @lru_cache(None)
#         def dp(idx: int, prev_color: int, k: int) -> int:

#             # No more houses left to paint, return true if there are no more neighborhoods
#             # to paint, false otherwise
#             if idx >= m:
#                 return 0 - int(k != 0)

#             # No more neighborhoods colors available
#             if k < 0:
#                 return -1

#             # Check if this house is already painted, if so, go to the next house
#             # Note: `k - int(prev_color != new_color)` decreases the number of
#             # neighborhoods left to paint if the current house color is different than the
#             # previous one
#             if houses[idx] != 0:
#                 return dp(idx + 1, houses[idx], k - int(prev_color != houses[idx]))

#             # Decide on the best color to paint current house
#             best = -1

#             # Try all possible colors
#             for new_color in range(1, n + 1):
#                 new_color_cost = cost[idx][new_color - 1]
#                 other_costs = dp(idx + 1, new_color, k - int(prev_color != new_color))

#                 # Check if painting this house with `new_color` will give us a lower cost
#                 if other_costs >= 0:
#                     best = min(best if best > 0 else float("inf"),
#                                new_color_cost + other_costs)
#             return best

#         # Start with the first house
#         return dp(0, 0, target)

class Solution:
    def minCost(self, houses: List[int], cost: List[List[int]], m: int, n: int, target: int) -> int:
       # f[house i][j neighborhoods][color k] 1<=j<=i
       # =min(f[house i-1][j neighborhoods][color k],f[house i-1][j-1 neighborhoods][color l!=k]) + cost[i][k] if houses[i] == 0
        
        f = [[-1] * n for i in range(target + 1)]
        if houses[0]:
            f[1][houses[0] - 1] = 0
        else:
            for k in range(n):
                f[1][k] = cost[0][k]
        
        for i in range(1, m):
            g = [[-1] * n for i in range(target + 1)]
            for j in range(1, min(i + 1, target) + 1):
                for k in range(n):
                    if houses[i] and houses[i] - 1 != k:
                        continue
                    g[j][k] = f[j][k]
                    for l in range(n):
                        if l != k and f[j - 1][l] != -1:
                            if  g[j][k] == -1 or f[j - 1][l] < g[j][k]:
                                g[j][k] = f[j - 1][l]
                    if g[j][k] != -1 and not houses[i]:
                        g[j][k] += cost[i][k]
            f = g
            
        ans = list(filter(lambda x: x != -1, f[target]))
        if not ans:
            return -1
        else:
            return min(ans)
class Solution:
    def minCost(self, houses: List[int], cost: List[List[int]], m: int, n: int, target: int) -> int:
        @lru_cache(None)
        def dfs(i: int, t: int, pc: int) -> int:
            if i == m:
                return math.inf if t != 0 else 0
            if houses[i] != 0: return dfs(i + 1, t - (pc != houses[i]), houses[i])
            else:
                return min(dfs(i + 1, t - (pc != c), c) + cost[i][c - 1] for c in range(1, n + 1))
        ans = dfs(0, target, -1)
        return ans if ans != math.inf else -1

from functools import lru_cache
class Solution:
    def minCost(self, houses: List[int], cost: List[List[int]], m: int, n: int, target: int) -> int:
        
        @lru_cache(None)
        def dfs(i,prev,k):
            if i>=m:
                return 0 - int(k!=0)
                
            if k<0:
                return -1
                
            if houses[i]!=0:
                return dfs(i+1, houses[i], k - int(prev!=houses[i]))
            
            else:
                temp = float('inf')
                for c in range(1,n+1):
                    c_cost = cost[i][c-1]
                    other = dfs(i+1,c, k-int(prev!=c))
                    
                    if other>=0:
                        temp = min(temp, c_cost+other)
                if temp == float('inf'):
                    return -1
                return temp
        return dfs(0,0,target)
            
            

class Solution:
    def minCost(self, houses: List[int], cost: List[List[int]], m: int, n: int, target: int) -> int:
        # dp[i][c][k]: i means the ith house, c means the cth color, k means k neighbor groups
        dp = [[[math.inf for _ in range(n)] for _ in range(target + 1)] for _ in range(m)]
        
        for c in range(1, n + 1):
            if houses[0] == c: dp[0][1][c - 1] = 0
            elif not houses[0]: dp[0][1][c - 1] = cost[0][c - 1]
                
        for i in range(1, m):
            for k in range(1, min(target, i + 1) + 1):
                for c in range(1, n + 1):
                    if houses[i] and c != houses[i]: continue
                    same_neighbor_cost = dp[i - 1][k][c - 1]
                    diff_neighbor_cost = min([dp[i - 1][k - 1][c_] for c_ in range(n) if c_ != c - 1] or [math.inf])
                    paint_cost = cost[i][c - 1] * (not houses[i])
                    dp[i][k][c - 1] = min(same_neighbor_cost, diff_neighbor_cost) + paint_cost
        res = min(dp[-1][-1])
        return res if res < math.inf else -1
class Solution:
    def minCost(self, houses: List[int], cost: List[List[int]], m: int, n: int, target: int) -> int:
        t = target
        dp = [[[0 for i in range(t+1)] for j in range(n+1)] for i in range(m+1)]
        for k in range(1,t+1):
            for j in range(1,n+1):
                dp[0][j][k] = 10**9
        for i in range(1,m+1):    
            for j in range(1,n+1):
                dp[i][j][0] = 10**9
        for i in range(1,m+1):
            for j in range(1,n+1):
                for k in range(1,t+1):
                    if houses[i-1] == j and k<=i:
                        dp[i][j][k] = min(dp[i-1][j][k],min([dp[i-1][p][k-1] if p!=j else 10**9 for p in range(1,n+1)]))
                    elif houses[i-1]==0 and k<=i:
                        dp[i][j][k] = cost[i-1][j-1]+min(dp[i-1][j][k],min([dp[i-1][p][k-1] if p!=j else 10**9 for p in range(1,n+1)]))
                    else:
                        dp[i][j][k] = 10**9
                        
        ans = 10**9
        for j in range(1,n+1):
            ans = min(ans, dp[m][j][t])
        return ans if ans<10**9 else -1
class Solution:
    def minCost(self, houses: List[int], cost: List[List[int]], m: int, n: int, target: int) -> int:
        m = len(houses)
        n += 1
        @lru_cache(None)
        def minCost(i, n_nei, prev_c):
            if i == m:
                if n_nei == target:
                    return 0
                return float('inf')
            if n_nei > target:
                return float('inf')
            if houses[i] != 0:
                if houses[i] == prev_c:
                    return minCost(i+1, n_nei, prev_c)
                return minCost(i+1, n_nei+1, houses[i])
            return min(minCost(i+1, n_nei if c==prev_c else (n_nei+1), c)+cost[i][c-1] for c in range(1, n))
        res = minCost(0, 0, -1)
        return res if res != float('inf') else -1
class Solution:
    def minCost(self, houses: List[int], cost: List[List[int]], m: int, n: int, target: int) -> int:
        @lru_cache(None)
        def dp(idx, prev_color, k):
            if idx >= m:
                return 0 - int(k != 0)
            if k < 0:
                return -1
            
            if houses[idx] != 0:
                return dp(idx + 1, houses[idx], k - int(prev_color != houses[idx]))
            
            best = math.inf
            
            for new_color in range(1, n + 1):
                new_cost = cost[idx][new_color - 1]
                other_cost = dp(idx + 1, new_color, k - int(new_color != prev_color))
                
                if other_cost >= 0:
                    best = min(best, new_cost + other_cost)
                    
            return best if best is not math.inf else -1
        res = dp(0, 0, target)
        return res 

from functools import lru_cache

class Solution:
    def minCost(self, houses: List[int], cost: List[List[int]], m: int, n: int, target: int) -> int:
        
        INF = 10**9

        @lru_cache(None)
        def dp(i, c, k):
            # print(i, c, k, i == m)
            if k > target:
                return INF
            if i >= m:
                return 0 if k == target else INF
            if houses[i] > 0:
                k_next = k if houses[i] == c else k + 1
                return dp(i + 1, houses[i], k_next)

            result = INF
            for j in range(1, n + 1):
                k_next = k if j == c else k + 1
                result = min(result, cost[i][j - 1] + dp(i + 1, j, k_next))

            return result


        ans = dp(0, 0, 0)
        return ans if ans < INF else -1


from functools import lru_cache
class Solution:
    def minCost(self, houses: List[int], cost: List[List[int]], m: int, n: int, target: int) -> int:
        @lru_cache(None)
        def dp(g, i, c):
            if g > target: return sys.maxsize
            if i == m: return 0 if g == target else sys.maxsize 
            ans = sys.maxsize 
            if houses[i] != 0:
                ans = min(ans, dp(g+(houses[i]!=c), i+1, houses[i]))
            else:
                for j in range(n):
                    ans = min(ans, cost[i][j] + dp(g+(j+1!=c), i+1, j+1))
            return ans
        
        ans = dp(0, 0, 0)
        return -1 if ans >= sys.maxsize else ans

class Solution:
    def minCost(self, houses: List[int], cost: List[List[int]], m: int, n: int, target: int) -> int:
        # dp[i][k][c]
        dp = [[[float('inf') for _ in range(n)] for _ in range(target+1)] for _ in range(m)]
        
        for c in range(1, n+1): # houses[i] == 0 u662fu6307u989cu8272u5f85u5237
            if houses[0] == c:
                dp[0][1][c-1] = 0 #luckyu4e0du7528u5237
            elif not houses[0]:
                dp[0][1][c-1] = cost[0][c-1] #u8981u5237
                
        for i in range(1, m):
            for k in range(1, min(target, i+1) + 1):
                for c in range(1, n+1):
                    if houses[i] and c != houses[i]: continue
                    same_neighbor_cost = dp[i-1][k][c-1]
                    diff_neighbor_cost = min([dp[i-1][k-1][c_] for c_ in range(n) if c_ != c-1])
                    paint_cost = cost[i][c-1] * (not houses[i])
                    dp[i][k][c-1] = min(same_neighbor_cost, diff_neighbor_cost) + paint_cost
        res = min(dp[-1][-1])
        
        return res if res < float('inf') else -1
                
                
        

from functools import lru_cache
class Solution:
    def minCost(self, houses: List[int], cost: List[List[int]], m: int, n: int, target: int) -> int:
        @lru_cache(None)
        def dp(g, i, c):
            if g > target: return sys.maxsize
            if i == m: return 0 if g == target else sys.maxsize 
            ans = sys.maxsize 
            if houses[i] != 0:
                if houses[i] == c: ans = min(ans, dp(g, i+1, c))
                else: ans = min(ans, dp(g+1, i+1, houses[i]))
            else:
                for j in range(n):
                    if j + 1 == c: ans = min(ans, cost[i][j] + dp(g, i+1, c))
                    else: ans = min(ans, cost[i][j] + dp(g+1, i+1, j+1))
            return ans
        
        ans = dp(0, 0, 0)
        return -1 if ans >= sys.maxsize else ans

from functools import lru_cache
class Solution:
    def minCost(self, houses: List[int], cost: List[List[int]], m: int, n: int, target: int) -> int:
        @lru_cache(None)
        def dp(g, i, c):
            if i == m: return 0 if g == target else sys.maxsize 
            ans = sys.maxsize 
            if houses[i] != 0:
                if houses[i] == c: ans = min(ans, dp(g, i+1, c))
                else: ans = min(ans, dp(g+1, i+1, houses[i]))
            else:
                for j in range(n):
                    if j + 1 == c: ans = min(ans, cost[i][j] + dp(g, i+1, c))
                    else: ans = min(ans, cost[i][j] + dp(g+1, i+1, j+1))
            return ans
        
        ans = dp(0, 0, 0)
        return -1 if ans >= sys.maxsize else ans

class Solution:
    def minCost(self, houses: List[int], cost: List[List[int]], m: int, n: int, target: int) -> int:
        dp = [[float('inf')]*(target+1) for _ in range(n+1)]
        if houses[0] != 0:
            dp[houses[0]][1] = 0
        else:
            for c in range(n):
                dp[c+1][1] = cost[0][c]
        for i in range(1, len(houses)):
            tmp = [[float('inf')]*(target+1) for _ in range(n+1)]
            for j in range(1, target+1):
                if houses[i] != 0:
                    if houses[i-1] == 0:
                        tmp[houses[i]][j] = min(dp[houses[i]][j], min([dp[color+1][j-1] for color in range(n) if color+1!=houses[i]]))
                    elif houses[i-1] == houses[i]:
                        tmp[houses[i]][j] = dp[houses[i]][j]
                    else:
                        tmp[houses[i]][j] = dp[houses[i-1]][j-1]
                else:
                    if houses[i-1]:
                        for c in range(n):
                            if c+1 == houses[i-1]:
                                tmp[c+1][j] = dp[c+1][j] + cost[i][c]
                            else:
                                tmp[c+1][j] = dp[houses[i-1]][j-1] + cost[i][c]
                    else:
                        for c in range(n):
                            tmp[c+1][j] = min(dp[c+1][j], min([dp[color+1][j-1] for color in range(n) if color!=c])) + cost[i][c]
            dp = tmp
        res = min([dp[c+1][target] for c in range(n)])
        return res if res != float('inf') else -1
class Solution:
    def minCost(self, houses: List[int], cost: List[List[int]], m: int, n: int, target: int) -> int:
        @lru_cache(None)
        def min_cost_helper(i, prev_color, groups):
            if i == m:
                return 0 if groups == target else float('inf')
            
            if houses[i] != 0:
                return min_cost_helper(i + 1, houses[i], groups + int(prev_color != houses[i]))
            
            total = float('inf')
            for color in range(1, n + 1):
                total = min(total, cost[i][color - 1] + min_cost_helper(i + 1, color, groups + int(prev_color != color)))
            
            return total
        
        ans = min_cost_helper(0, -1, 0)
        return ans if ans != float('inf') else -1

class Solution:
    def minCost(self, houses: List[int], cost: List[List[int]], m: int, n: int, target: int) -> int:
        @lru_cache(None)
        def dp(idx, prev_color, k):
            if idx >= m:
                return 0 - int(k != 0)
            if k < 0:
                return -1
            
            if houses[idx] != 0:
                return dp(idx + 1, houses[idx], k - int(prev_color != houses[idx]))
            
            best = math.inf
            
            for new_color in range(1, n + 1):
                new_cost = cost[idx][new_color - 1]
                other_cost = dp(idx + 1, new_color, k - int(new_color != prev_color))
                
                if other_cost >= 0:
                    best = min(best, new_cost + other_cost)
                    
            return best if best is not math.inf else -1
        res = dp(0, 0, target)
        return res if res != math.inf else -1

class Solution:
    def minCost(self, houses: List[int], Cost: List[List[int]], m: int, n: int, target: int) -> int:
        @lru_cache(None)
        def dfs(i, j, k):
            if j > target:
                return float('inf')
            if i == len(houses):
                if j == target:
                    return 0
                else:
                    return float('inf')
                
            cost = float('inf')
            if houses[i] == 0:
                for index, c in enumerate(Cost[i]):
                    if i == 0:
                        cost = min(cost, dfs(i + 1, 1, index + 1) + c)
                    else:
                        if index + 1 == k:
                            cost = min(cost, dfs(i + 1, j, index + 1) + c)
                        else:
                            cost = min(cost, dfs(i + 1, j + 1, index + 1) + c)     
                    houses[i] = 0
            else:
                if i == 0:
                    cost = dfs(i + 1, 1, houses[i])
                else:
                    if houses[i] == k:
                        cost = dfs(i + 1, j, houses[i])
                    else:
                        cost = dfs(i + 1, j + 1, houses[i])
            return cost
        
        ans = dfs(0, 0, 0)
        return ans if ans != float('inf') else -1
    

