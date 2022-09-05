from math import comb
from math import pow
class Solution:

        
    
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        if(target < d*1 or target > d*f ):
            return 0
        target = target - d
        sum = 0
        i = 0
        j=0
        while(i <= target):
            y = target - i
            if(j%2 == 0):
            
                sum =int( (sum +  comb(d, j) * comb(y+d-1,y)) )
            else:
                sum =int( (sum -  comb(d, j) * comb(y+d-1,y)))
            #print( comb(d, j) * comb(y+d-1,y))
            #print('i ={} y= {} sum={}  '.format(i,y,sum))
            j=j+1
            i = i + f
            
        #print(sum)
        return int(sum) % 1000000007

class Solution:
    
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        if target < d or target > d * f:
            return 0
        
        total_sum = 0
        cache_sum = []
        
        if target > d * (1 + f) / 2:
            target = d * (1 + f) - target
        
        for i in range(target, -1, -1):
            num = 0 if i > f else 1
            total_sum += num; 
            cache_sum.append(total_sum)
        cache_sum = cache_sum[::-1]
        cache_sum[0] = cache_sum[1]
        
        for i in range(2, d+1):
            total_sum = 0
            for j in range(target, -1, -1):
                total_sum += cache_sum[max(j-f, 0)] - cache_sum[j]
                cache_sum[j] = total_sum
        return cache_sum[-1] % (10**9+7)

class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        @lru_cache(None)
        def solve(s,t):
            if s == 0:
                if t == 0:return 1
                return 0
            ans = 0
            for i in range(1,f+1):
                if t >= i:
                    ans += solve(s - 1,t - i)
            return ans
        return solve(d,target) % (10**9+7)
class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        # dp[d][target] = sum(dp[d-1][target - w]) w in 1-f
        dp = {0: 1}
        MOD = 10 ** 9 + 7
        for i in range(d):
            ndp = {}
            for k, v in list(dp.items()):
                for w in range(1, f+1):
                    if k + w <= target: 
                        ndp[k+w] = (ndp.get(k+w, 0) + dp[k] ) % MOD
            dp = ndp.copy()
        return dp.get(target, 0)

class Solution:
    seen = {}
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        results = {}
        results[1] = {}
        for i in range(1, min(f, target) + 1):
            results[1][i] = 1
        for i in range(2, d+1):
            results[i] = {}
            for val, times in list(results[i-1].items()):
                for j in range(1, min(f, target) + 1):
                    results[i][j+val] = results[i].get(j+val, 0) + times
                    
        return results[d].get(target, 0) % 1000000007

class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
      return memoization(d,f,target)

# thank goodness that dices have orders (they are different dices)      
# d = 2, f = 3, t = 4
# (1,3), (2,2), (3,1)
# note that (1,3) and (3,1) are different ways (by definition in this example)
# TODO: what if they are exchangable?

# sol:
# f(d, f, t) = f(d-1, f, t-v) for v ...

from functools import lru_cache
mod = 10**9 + 7
@lru_cache(maxsize=None)
def memoization(d, f, t):
  if d == 0: 
    return int(t==0)
  return sum(memoization(d-1, f, t-v) for v in range(1,f+1)) % mod
class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        
        @lru_cache(maxsize=None)
        def dp(d, t):
            print((d,t))
            if d == 0:
                return 1 if t == 0 else 0
            return sum(dp(d-1, t - x) for x in range(1, f+1))
        return dp(d, target)%(10**9 + 7)
        

class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        mem = collections.defaultdict(int)
        mem[0] = 1
        for i in range(d):
            curr = collections.defaultdict(int)
            for face in range(1,f+1):
                for val,count in list(mem.items()):
                    newVal = face + val
                    if newVal <= target:
                        curr[newVal] += count 
            mem = curr
        return mem[target]%(10**9+7)
            

class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        dp = [[0]*(target+1) for i in range(d+1)]
        MOD = 1000000007
        dp[0][0]=1
        for dice in range(1,d+1):
            for target in range(1,target+1):
                if(target>dice*f):
                    continue
                else:
                    face=1
                    while(face<=f and face<=target):
                        dp[dice][target]=(dp[dice][target]+dp[dice-1][target-face])%MOD
                        face+=1
        return dp[-1][-1]
class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        MOD = 10 ** 9 + 7
        dp = [0] * (target + 1)
        dp[0] = 1
        for i in range(1, d + 1):
            if target < i:
                break
            t = min(i * f, target)
            dp[t] = sum(dp[t - j] for j in range(max(1, t - (i - 1) * f), min(f, t - i + 1) + 1)) % MOD
            for t in reversed(range(i,  t)):
                dp[t] = dp[t + 1] - dp[t] + (dp[t - f] if t - f >= i - 1 else 0) 
        return dp[target]
class Solution:
    def numRollsToTarget(self, d, f, target):
        dp = [0] * (target+1)
        dp[0] = 1
        for i in range(d):
            for j in range(target, -1, -1):
                tot = 0
                for k in range(1, 1 + min(f, j)):
                    tot += dp[j-k]
                dp[j] = tot
        return dp[target] % (10**9 + 7)
class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        if target > d*f:
            return 0
        if target == d * f:
            return 1
        if d==1 :
            return 1
        if f > target:
            f = target
        MOD = 10**9+7
        dp = [[0 for j in range( d*f+1 +1) ]for i in range(d+1)]


        for i in range(1,f+1):
            dp[1][i] = 1
        for i in range(d+1):
            for j in range(d*f+1+1):
                for t in range(1,f+1):
                    if j-t>=i-1:
                        dp[i][j] += dp[i-1][j-t]%MOD
        return dp[d][target]%MOD
class Solution:
    @lru_cache(None)
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        if d == 0 or target <= 0:
            return d == 0 and target == 0
        res = 0
        for i in range(1, f+1):
            res = (res + self.numRollsToTarget(d - 1, f, target - i)) % (10**9 + 7)
        return res

class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        memo = {}
        def helper(rolls_left, t):
            if not t and not rolls_left:
                return 1
            if not rolls_left or not t:
                return 0
            ret = 0
            for v in range(1, min(f+1, t+1)):
                if (rolls_left -1, t-v) not in memo:
                    memo[(rolls_left -1, t-v)] = helper(rolls_left - 1, t-v)
                ret += memo[(rolls_left -1, t-v)]
            return ret
        return helper(d,target) % (10**9 + 7)
class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        @lru_cache(None)
        def solve(s,t):
            if s == 0:
                if t == 0:return 1
                return 0
            ans = 0
            for i in range(1,f+1):
                if t >= i:
                    ans += solve(s - 1,t - i)
            return ans
        
        return solve(d,target) % (10**9+7)
class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        modulo = 1000000007
        memo = {}
        def recurseRoll(d, target):
            if d < 0:
                return 0
            if target < 0:
                return 0
            if target == 0 and d == 0:
                return 1
            s = 0
            for i in range(1, f+1):
                if ((d-1,target-i)) in memo:
                    s+=memo[(d-1,target-i)]
                else:
                    s+=recurseRoll(d-1, target-i)
            memo[(d, target)] = s
            return s
        return recurseRoll(d, target) % modulo
from functools import lru_cache
class Solution:
    @lru_cache
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        
        dp = [[0] * (target+1) for _ in range(d+1)]
        dp[0][0] = 1
        for i in range(1, d+1):
            for j in range(target+1):
                if dp[i-1][j] > 0:
                    for k in range(1, f+1):
                        if j + k <= target:
                            dp[i][j+k] += dp[i-1][j] 
                            dp[i][j+k] %= (10**9 + 7)
        #print(dp)
        return dp[d][target]
class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        dpMatrix = [[0 for i in range(target+1)] for i in range(d)]
        
        for i in range(1,min(f+1, len(dpMatrix[0]))):
            dpMatrix[0][i] = 1
        
        for i in range(1,len(dpMatrix)):
            for j in range(1, len(dpMatrix[0])):
                calcSum=0
                for k in range(j-1, max(-1,j-f-1), -1):
                    calcSum+= dpMatrix[i-1][k]
                
                dpMatrix[i][j] = calcSum
        
        return dpMatrix[-1][-1]%(10**9 + 7)
class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        memo = {}
        def helper(d, target):
            if d == 0:
                 return 0 if target > 0 else 1
            if (d, target) in memo:
                return memo[(d,target)]
            to_return = 0
            for k in range(max(0, target-f), target):
                to_return += helper(d-1, k)
            memo[(d, target)] = to_return
            return to_return
        return helper(d, target) % (10**9 + 7)

class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        mem = {(0, 0): 1}
        def find(d: int, target: int):
            if (d, target) in mem:
                return mem[(d, target)]
            
            if d == 0:
                return 0
            
            res = 0
            d -= 1
            for i in range(1, min(target - d, f)+1):
                res += find(d, target-i)
            mem[(d+1, target)] = res
            return res
        
        return find(d, target) % (10**9 + 7)

class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        def solve(s,t):
            if s == 0:
                if t == 0:return 1
                return 0
            if dp[s][t] != -1:return dp[s][t]
            ans = 0
            for i in range(1,f+1):
                if t >= i:
                    ans += solve(s - 1,t - i)
            dp[s][t] = ans
            return dp[s][t]
        dp = [[-1]*(target+1) for _ in range(d+1)]
        return solve(d,target) % (10**9+7)
class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        dp_map ={}
        
        for i in range(target):
            dp_map[(1,i+1)] = 0
        for i in range(f):
            dp_map[(1,i+1)] = 1
        
        def recurse_with_mem(d,target):
            
            if (d,target) in dp_map:
                return dp_map[(d,target)]

            if target<=0:
                return 0
            
            num_ways = 0
            for i in range(f):
                num_ways += recurse_with_mem(d-1,target-i-1)
        
            dp_map[(d,target)] = num_ways
            return num_ways
        
        return recurse_with_mem(d,target)%(10**9+7)
class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        memo = {}
        def dp(d,target):
            if d == 0:
                return 0 if target>0 else 1
            if (d,target) in memo:
                return memo[(d,target)]
            res = 0
            for k in range(max(0, target-f),target):
                res += dp(d-1, k)
            memo[(d,target)] = res
            return res
                
        return dp(d,target) % (10**9 + 7)
class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        memo = {}
        def dp(d, target):
            nonlocal memo, f
            if d == 0:
                return 1 if target == 0 else 0
            if (d, target) in memo:
                return memo[(d, target)]
            res = 0
            for i in range(max(0, target-f), target):
                res += dp(d-1, i)
            memo[(d, target)] = res
            return res
        return dp(d, target)%(10**9+7)
class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        memo = {}
        def helper(dice, tar):
            if dice == 0:
                return 1 if tar == 0 else 0
            
            if (dice, tar) in memo:
                return memo[(dice, tar)]
            
            res = 0
            for k in range(max(0, tar-f), tar):
                res += helper(dice-1, k)
            
            memo[(dice, tar)] = res
            return res
        
        
        return  helper(d, target) % (10**9 + 7)
class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        memo = {}
        
        def helper(dice, t):
            if dice == 0:
                return 1 if t == 0 else 0
            
            if (dice, t) in memo:
                return memo[(dice, t)]
            
            res = 0
            for k in range(max(0, t-f), t):
                res += helper(dice-1, k)
            
            memo[(dice, t)] = res
            return res
        
        return helper(d, target) %(10**9 + 7)
class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        memo = {}
        def dp(d,target):
            if d == 0:
                return 1 if target == 0 else 0
            if (d,target) in memo:
                return memo[(d,target)]
            result = 0
            for k in range(max(0,target-f), target):
                result += dp(d-1,k)
            memo[(d,target)] = result
            return memo[(d,target)]
        return dp(d, target) % (10**9 + 7)
class Solution:
    # Recursive memoized solution
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        memo = {}
        def num_rolls_util(level, target):
            if level == 0:
                return target == 0

            if (level, target) in memo:
                return memo[(level, target)]
            else:
                res = 0
                for i in range(max(0, target - f), target):
                    res += num_rolls_util(level - 1, i)

                memo[(level, target)] = res

                return res % (10 ** 9 + 7)
        
        return num_rolls_util(d, target)
class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        if d>target or d*f<target: return 0
        
        if f>target: f = target
        
        memo = [[0 for _ in range(target)] for _ in range(d)]
        
        # set first line
        for i in range(f):
            memo[0][i] = 1
        
        # run algo
        for i,row in enumerate(memo[1:]):
            for j,_2 in enumerate(row):
                if j-f<0:
                    memo[i+1][j] = sum(memo[i][0:j]) % 1000000007
                else:
                    memo[i+1][j] = sum(memo[i][j-f:j]) % 1000000007
                
        return memo[-1][-1] % 1000000007
class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        self.mapx = {}
        return self.rec(d, f, target)
       
    def rec(self, d, f, target):
        if d == 0:
            if target == 0:
                return 1
            return 0
        
        count = 0
        
        for num in range(1,f+1):
            if target-num >= 0:
                nextState = (d-1, target-num)
                if nextState not in self.mapx:
                    count += self.rec(d-1, f, target-num)
                else:
                    count += self.mapx.get(nextState,0)
                    
        count = count % 1000000007
        self.mapx[(d, target)] = count 
        return count
class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:

        cache = {}
        
        def helper(d, target):
            if d == 0 and target == 0:
                return 1
            if d == 0:
                return 0
            if (d, target) in cache:
                return cache[(d, target)]
            
            result = 0
            for i in range(max(0, target-f), target):
                result += helper(d-1, i)
                
            cache[(d, target)] = result
            return result
    
        
        return helper(d, target) % ((10**9)+7)

class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        self.dic = {}
        self.f = f
        return self.dfs(d, target) % (10**9 + 7)
        
        
    def dfs(self, dices, tar):
        if (dices, tar) in self.dic:
            return self.dic[(dices, tar)]
        elif tar < 0:
            return 0
        elif dices == 1 and tar <= self.f:
            return 1
        elif dices == 1:
            return 0
        else:
            r = 0
            for i in range(1, min(self.f+1, tar)):
                r += self.dfs(dices-1, tar-i)
            
            self.dic[(dices, tar)] = r
            return r

class Solution:
    def numRollsToTarget(self, k: int, f: int, target: int) -> int:
        d = {}
        def h(x,t):
            v = (x,t)
            if v in d:
                return d[v]
            if x == 0 and t == 0:
                return 1
            if x == 0 or t == 0:
                return 0
            s = 0
            for i in range(1,f+1):
                s+=h(x-1,t-i)
            d[v] = s
            return s
        x = h(k,target)
        return x%(10**9+7)
class Solution:
    def numRollsToTarget(self, d, f, target):
        dp = [1] + [0]*target
        for i in range(d):
            for j in range(target, -1, -1):
                dp[j] = sum([dp[j-k] for k in range(1, 1+min(f, j))] or [0])
        return dp[target] % (10**9 + 7)
class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        D = {}
        def func(d,f, target):
            res = 0
            if d==0 and target==0:
                return 1
            if d==0 and target!=0:
                return 0
            if target<0:
                return 0
            
            for i in range(1,f+1):
                if (d-1,target-i) not in D:
                    D[d-1,target-i] = func(d-1,f,target-i)
                res+=D[d-1,target-i]
                res = res%(10**9 + 7)
            return res
        
        #d=2
        #f=6
        #target=7
        return func(d,f,target)
            

class Solution:
    def numRollsToTarget(self, d, f, target):
        dp = [0] * (target+1)
        dp[0] = 1
        for i in range(d):
            for j in range(target, -1, -1):
                dp[j] = sum([dp[j-k] for k in range(1, 1+min(f, j))] or [0])
        return dp[target] % (10**9 + 7)
class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        # bottom up DP
        prev_dp = [0] * (target + 1)
        prev_dp[0] = 1
        for num_dice in range(1, d + 1):
            curr_dp = [0] * (target + 1)
            for face in range(1, f+1):
                for total in range(face, target + 1):
                    curr_dp[total] = (curr_dp[total] + prev_dp[total - face]) % (10 ** 9 + 7)
            prev_dp = curr_dp
        return prev_dp[-1]
class Solution:
    # @lru_cache(maxsize=None)
    # def numRollsToTarget(self, d: int, f: int, target: int) -> int:
#         mod = 1e9 + 7
#         if target < 0 or d < 0:
#             return 0

#         if target == 0 and d == 0:
#             return 1

#         ways = 0
#         for dice in range(1, 1 + f):
#             ways += self.numRollsToTarget(d - 1, f, target - dice
#             )
#         return int(ways % mod)
    
#         dp = [0] * (1 + target)
#         dp[0] = 1
#         mod = 1e9 + 7

#         for rep in range(d):
#             newDP = [0] * (1 + target)
#             for i in range(1, 1 + f):
#                 for j in range(1, 1 + target):
#                     if i <= j:
#                         newDP[j] += dp[j - i]
#                         newDP[j] %= mod
#             dp = newDP
#         return int(dp[-1])

    # @lru_cache(maxsize=None)
    def numRollsToTarget(self, d, f, target, result=0):
#         MOD = 7 + 1e9
#         if d == 0:
#             return target == 0

#         for i in range(1, f + 1):
#             result = (result + self.numRollsToTarget(d - 1, f, target - i)) % MOD

#         return int(result)
        
        MOD = 7 + 1e9
        dp = [0] * (1 + target)
        dp[0] = 1
        for i in range(1, 1 + d):
            newDP = [0] * (1 + target)
            prev = dp[0]
            for j in range(1, 1 + target):
                newDP[j] = prev
                prev = (prev + dp[j]) % MOD
                if j >= f:
                    prev = (prev - dp[j - f] + MOD) % MOD
            dp = newDP
        return int(dp[-1])
class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        
        # we can use memoization
        DP = {}
        def get_rolls(n, t):
            if n == 1:
                if 0 < t <= f:
                    return 1
                return 0
            if (n, t) in DP:
                return DP[(n, t)]
            
            total = 0
            
            for i in range(1, f+1):
                total += get_rolls(n-1, t-i)
            
            DP[(n, t)] = total
            return total
        
        return get_rolls(d, target) % (10**9 + 7)
class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        dp1 = [0]*(target + 1)
        dp2 = [0]*(target + 1)
        dp1[0] = 1
        for i in range(1, d+1):
            for j in range(1, target+1):
                for k in range(1, min(j, f) + 1):
                    dp2[j] =  dp2[j] + dp1[j-k]
            dp1 = dp2
            dp2 = [0]*(target + 1)
                    
        return dp1[target] % (10**9 + 7)

class Solution:
    def __init__(self):
        self.dp = []
        
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        # Initiate memoization array
        for dice in range(0, d + 1):
            self.dp.append([-1] * (target + 1))
        
        return self.rollDice(d, f, target) % (10 ** 9 + 7)
    
    def rollDice(self, d: int, f: int, target: int) -> int:
        if self.dp[d][target] != -1:
            return self.dp[d][target]
        
        if d == 1:
            if f >= target:
                return 1
            else:
                return 0
        
        self.dp[d][target] = 0
        for nextRoll in range(1, f + 1):
            if target - nextRoll > 0:
                self.dp[d][target] += self.rollDice(d - 1, f, target - nextRoll)
        
        return self.dp[d][target]
class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        def find_next(dice_left, current_val):
            if current_val > target:
                return 0
            if dice_left == 0:
                if current_val == target:
                    return 1
                return 0
            
            if (dice_left, current_val) not in seen:
                combinations = 0
                for i in range(1, f + 1):
                    combinations += find_next(dice_left - 1, current_val + i)
                seen[(dice_left, current_val)] = combinations % 1000000007
                
            return seen[(dice_left, current_val)]
        
        seen = {}
        return find_next(d, 0) 

class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        bottomWidth = f + ((d-1)*(f-1))
        currentLayer = [0 for i in range(bottomWidth)] + [1 for i in range(f)] + [0 for i in range(bottomWidth)]
        nextLayer = []
        
        for run in range(d-1): # runs as many times as we traverse down the triangle
            # update each next layer number to be sum of f numbers above it
            for i in range(f, len(currentLayer)-f):
                localSum = 0
                for j in range(i, f+i):
                    localSum += currentLayer[j]
                nextLayer += [localSum]
                
            neededZeros = [0 for i in range((len(currentLayer) - len(nextLayer))//2)]
            currentLayer = neededZeros + nextLayer + neededZeros
            nextLayer = []
            
        # at this point, shave off the zeros
        while 0 in currentLayer:
            currentLayer.remove(0)
            
        # at this point, the answer must be less than or equal to d, and greater than or equal to f*d.
        if target-d >= len(currentLayer): 
            return 0
        else: 
            return currentLayer[target-d] % ((10**9) + 7)
class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        # bottom up DP
        prev_dp = [0] * (target + 1)
        prev_dp[0] = 1
        for num_dice in range(1, d + 1):
            curr_dp = [0] * (target + 1)
            for face in range(1, f+1):
                for total in range(face, target + 1):
                    curr_dp[total] = (curr_dp[total] + prev_dp[total - face]) % (10 ** 9 + 7)
            prev_dp = curr_dp
            print(prev_dp)
        return prev_dp[-1]
class Solution:
    def numRolls(self, d, f, target, memo):
        if target < d:
            # print(f"d {d} f {f} target {target} ans {0}")
            return 0
        if d == 1:
            # print(f"d {d} f {f} target {target} ans {int(target <= f)}")
            return int(target <= f)
        
        if (d, f, target) in memo:
            return memo[(d, f, target)]

        total = 0
        for new_target in range(target - f, target):
            total += self.numRolls(d - 1, f, new_target, memo)
            total = total % (10**9 + 7)
        # print(f"d {d} f {f} target {target} ans {total}")
        memo[(d, f, target)] = total
        return total

    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        memo = {}
        ans = self.numRolls(d, f, target, memo)
        # print(memo)
        return ans

class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        
        memo = defaultdict(int)

        def solve(d, f, target):
            if (d,f,target) in memo:
                return memo[(d,f,target)]
            if d==0:
                if target!=0:
                    return 0
                else:
                    return 1
            dp = 0
            for i in range(1,f+1):
                dp += solve(d-1,f,target-i)
            memo[(d,f,target)] = dp
            return dp
        
        return solve(d,f,target)%(10**9+7)
class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        dp = [1 if i < f else 0 for i in range(target)]
        for n in range(2, d+1):
            new_dp = [0 for _ in range(target)]
            cumsum = 0
            for i in range(target-1):
                cumsum += dp[i]
                if i >= f:
                    cumsum -= dp[i-f]
                new_dp[i+1] = cumsum 
            dp = new_dp
            # print(n, dp)
        return dp[-1] % (10**9 + 7)

class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        def helper(h, d, target):
            # if target is too small or if it is out of range
            if not d and not target:
                return 1

            if (d, target) in h:
                return h[(d, target)]        # directly access from hash table
            res = 0
            for i in range(1, f + 1):
                if target - i >= 0 and d > 0:
                    res += helper(h, d - 1, target - i)       # check all possible combinations
            h[(d, target)] = res
            return h[(d, target)]
        
        h = {}
        return helper(h, d, target) % (10 ** 9 + 7)
class Solution:
    def helper(self,d,f,target):
        if(d == 0):
            if(target == 0):
                return 1
            else:
                return 0
        
        if((d,target) in self.memo):
            return self.memo[(d,target)]
        
        count = 0
        for i in reversed(list(range(1,min(target+1,f+1)))):
            count+=self.helper(d-1,f,target-i)
        
        self.memo[(d,target)]=count
        return count
    
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        self.memo = dict()
        return self.helper(d,f,target)%(10**9 +7)

class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        return self.recurse(d,0,f,0,target,{})%(10**9+7)
    
    def recurse(self,dices,dno,faces,cursum,target,cache):
        if cursum>target:
            return 0
        
        if dno==dices:
            if cursum==target:
                return 1
            return 0
        
        if (dno,cursum) in cache:
            return cache[(dno,cursum)]
        
        ways=0
        for curface in range(1,faces+1):
            ways+=self.recurse(dices,dno+1,faces,cursum+curface,target,cache)
        cache[(dno,cursum)]=ways
        return ways
class Solution:
    # Recursive memoized solution
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        memo = {}
        return self.num_rolls_util(memo, d, f, target, 0, 0)
    
    
    def num_rolls_util(self, memo, d, f, target, level, cur_sum):
        if level == d:
            return cur_sum == target
        
        if (level, cur_sum) in memo:
            return memo[(level, cur_sum)]
        
        res = 0
        for i in range(1, f + 1):
            if cur_sum + i <= target:
                res += self.num_rolls_util(memo, d, f, target, level + 1, cur_sum + i)
            
        memo[(level, cur_sum)] = res
        
        return res % (10 ** 9 + 7)
class Solution:
    
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        
        #[1,2,3,4,5,6], target=7
        #[1,2,3,4,5,6]
        #[1,2,3,4,5,6]
        #[]
        
        store = {}
        
        def helper(d, f, target):
            if d==0 or f==0 or target<=0:
                return 0
            if d==1 and target>f:
                return 0
            if d==1 and target<=f:
                return 1
        
            if (d, f, target) in store:
                 return store[(d,f,target)]
        
            n = 0
            for i in range(1, f+1):
                n += helper(d-1, f, target-i)
        
            store[(d, f, target)] = n
            return n
    
        #d=2, f=6, t=7
        #i=1, 1,6,6
        
        return (helper(d, f, target))%(10**9 + 7)
            
        #2,6,7
        #(1,6,6), (1,6,5), (1,6,4), (1,6,3), (1,6,2),(1,6,1)
        

class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        memo = {}
        
        def dfs(n, t):
            if n == 0:
                return t == 0
            if (n, t) in memo:
                return memo[n, t]
            
            ret = 0
            for face in range(1, f+1):
                ret += dfs(n-1, t-face)
            
            memo[n, t] = ret % (10**9 + 7)
            return memo[n, t]
        
        return dfs(d, target)
class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        def backtrack(index=d-1,target=target,dp={}):
            if index==-1 and target==0:return 0
            if index<0 or target<=0:return None
            if (index,target) in dp:return dp[(index,target)]
            if target>f:start=f
            else:start=target
            count=0
            for i in range(start,0,-1):
                res=backtrack(index-1,target-i,dp)
                if res!=None:
                    count+=res if res else 1
            dp[(index,target)]=count if count else None
            return dp[(index,target)]
        count=backtrack()
        return (count%(10**9+7)) if count!=None else 0

class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        memo = {}
        def helper(d, target):
            if d == 0 and target == 0:
                return 1
            if d == 0:
                return 0
            if (target, d) in memo:
                return memo[(target, d)]
            num_ways = 0
            for i in range(1, f+1):
                if target - i >= 0:
                    num_ways += helper(d-1, target-i)
            memo[(target, d)] = num_ways
            return num_ways
        
        return helper(d, target) % (10**9 + 7)
class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        dp = [0] * (target+1)
        dp[0] = 1
        for i in range(1, d+1):
            temp = [0] * (target+1)
            # iterate each tot from 1 to target
            for j in range(1, target+1):
                # k is each face 
                temp[j] = sum(dp[j-k] if k <= j else 0 for k in range(1, f+1))
            dp = temp
        return dp[target] % (10**9 + 7)
class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        
        if target > 900:
            return 0
        memo = {}
        
        def dfs(dice, t):
            if dice == 0 and t == 0:
                return 1
            if dice == 0 or t == 0:
                return 0
            
            if (dice, t) in memo:
                return memo[(dice, t)]
            count = 0
            for n in range(1, f+1):
                count += dfs(dice-1, t-n)
            memo[(dice, t)] = count % ((10 ** 9) + 7)
            return memo[(dice, t)]
        
        return dfs(d, target)
class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        modulo = 10**9 + 7
        dp = [[0 for i in range(target+1)] for j in range(d+1)]
        
        for dd in range(1, d+1):
            for tt in range(dd, min(f * dd, target) + 1 ):
                if dd == 1:
                    dp[dd][tt] = 1
                else:
                    end   = tt - 1
                    start = max(1, tt - f)
                    dp[dd][tt] = sum(dp[dd-1][start:end+1])
    
        return dp[d][target] % modulo   
class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        memo = {}
        return self.num_rolls_util(memo, d, f, target, 0, 0)
    
    
    def num_rolls_util(self, memo, d, f, target, level, cur_sum):
        if level == d:
            return cur_sum == target
        
        if (level, cur_sum) in memo:
            return memo[(level, cur_sum)]
        
        res = 0
        for i in range(1, f + 1):
            res += self.num_rolls_util(memo, d, f, target, level + 1, cur_sum + i)
            
        memo[(level, cur_sum)] = res
        
        return res % (10 ** 9 + 7)
class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        
        self.dp = {}
        
        def recur_top_down(d, f, target):
            
            if d == 0 or target <= 0: 
                return d == 0 and target == 0

            if (d, target) in self.dp: return self.dp[(d, target)]
            
            res = 0
            for i in range(f):
                res += recur_top_down(d-1, f, target-(i+1)) % (10**9+7)

            res = res % (10**9+7)
            self.dp[(d, target)] = res 
            
            return res 
            
        
        return recur_top_down(d, f, target)
class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        mem = {}
        def dfs(l, target):
            if((l, target) in mem):
                return mem[(l, target)]
            if(l==d):
                return int(target==0)
            MOD = 1e9+7
            ans=0
            for i in range(1, f+1):
                ans = (ans+dfs(l+1, target-i))%MOD
            mem[(l, target)] = ans
            return ans
        return int(dfs(0, target))
class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        
        self.dp = {}
        
        def recur_top_down(d, f, target):
            
            if d == 0 : 
                return 1 if target == 0 else 0

            if (d, target) in self.dp: return self.dp[(d, target)]
            
            res = 0
            for i in range(f):
                res += recur_top_down(d-1, f, target-(i+1)) % (10**9+7)

            res = res % (10**9+7)
            self.dp[(d, target)] = res 
            
            return res 
            
        
        return recur_top_down(d, f, target)
class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        
        def count_for_roll(num_dice, curr_target):
            
            if num_dice == 0 and curr_target == 0:
                return 1
            
            if num_dice == 0 or curr_target == 0:
                return 0
            
            if (num_dice, curr_target) in memo:
                return memo[(num_dice, curr_target)]
            
            curr_count = 0
            
            for face in range(1, f+1):
                
                new_target = curr_target - face
                
                if new_target < 0:
                    break
                    
                curr_count += count_for_roll(num_dice - 1, new_target)
            
            memo[(num_dice, curr_target)] = curr_count
            return curr_count % mod
        
        memo = {}
        mod = 10**9 + 7
        
        return count_for_roll(d, target)

'''
d = 2, f = 6, t = 7

1
    1
    2
    3
    4
    5
    6 -> inc
    
2
    1
    2
    3
    4
    5 -> inc
    6 -> stop?
    
3 
    1
    2
    3
    4 -> inc
    5 -> stop
    
    

'''
class Solution:
    def helper(self,d,f,target,cursum,dp):
        if(d==0):
            if(target==cursum):
                return 1
            else:
                return 0
        if(cursum>target):
            return 0
        cnt = 0
        if((d,cursum) in dp):
            return dp[(d,cursum)]
        for i in range(1,f+1):
            cnt+=self.helper(d-1,f,target,cursum+i,dp)
        dp[(d,cursum)]=cnt%(pow(10,9)+7)
        return cnt%(pow(10,9)+7)
    
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        dp={}
        return self.helper(d,f,target,0,dp)
class Solution:
    # Recursive memoized solution
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        memo = {}
        return self.num_rolls_util(memo, d, f, target, 0, 0)
    
    
    def num_rolls_util(self, memo, d, f, target, level, cur_sum):
        if level == d:
            return cur_sum == target
        
        if (level, cur_sum) in memo:
            return memo[(level, cur_sum)]
        
        res = 0
        for i in range(1, f + 1):
            res += self.num_rolls_util(memo, d, f, target, level + 1, cur_sum + i)
            
        memo[(level, cur_sum)] = res
        
        return res % (10 ** 9 + 7)
class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        memo = {}
        def helper(d, target):
            if d==0 and target == 0:
                return 1
            if d==0 or target == 0:
                return 0
            if (d,target) in memo:
                return memo[(d,target)]
            else:
                ans = 0
                for i in range(1, f+1):
                    ans+= helper(d-1, target-i)
                memo[(d,target)] = ans%(10**9+7)
                return memo[(d,target)]
        return helper(d, target)
from collections import defaultdict

class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        memo = defaultdict(int)
        
        def helper(d, target):
            if d == 0:
                return 1 if target == 0 else 0
            if (d, target) in memo:
                return memo[(d, target)]


            #for c in range(1, f + 1):
            for c in range(max(0, target-f), target):
                #memo[(d, target)] += helper(d - 1, target - c, memo)
                memo[(d, target)] += helper(d - 1, c)   
            return memo[(d, target)]
        return helper(d, target) % (10**9 + 7)
    

class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        # f(n) = f(n - 1) + f(n-2) + ...
        dp = [[0 for i in range(target+1)] for t in range(d+1)]
        for dd in range(1, d+1):
            for tt in range(dd, min(f*dd, target)+1):
                if dd == 1:
                    dp[dd][tt] = 1
                else:
                    end = tt -1
                    start = max(1, tt-f)
                    dp[dd][tt] = sum(dp[dd-1][start:end+1])
        return dp[d][target]%(10**9 + 7)
        
    

class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        cache = {}
        def helper(d,target):
            # print(d,target)
            if target == 0 and d == 0:
                return 1
            if d == 0:
                return 0
            if (d,target)  in cache:
                return cache[(d,target)]
            ways = 0
            for i in range(1,f+1):
                if target - i >= 0:
                    ways = (ways + helper(d-1,target-i)) % 1000000007
            cache[(d,target)] = ways
            return cache[(d,target)]
        
        return helper(d,target)

class Solution:
    '''
    similar to coin Change problem.
        
    '''
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        dp1 = [0 for _ in range(target+1)]
        dp2 = dp1[:]
        dp1[0] = 1
        mod = 10 ** 9 + 7
        for i in range(1, d+1):
            for j in range(target+1):
                for k in range(1, min(j,f) + 1):
                    dp2[j] = (dp2[j] + dp1[j-k]) % mod
            dp1 = dp2
            dp2 = [0 for _ in range(target+1)]
        return dp1[target] % mod
    
# https://www.youtube.com/watch?time_continue=872&v=UiYVToWORMY&feature=emb_logo

class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        memo = {}
        MOD = pow(10, 9) + 7
        def dfs(d_left=d, target_left=target):
            if (d_left, target_left) in memo:
                return memo[(d_left, target_left)]
            if not d_left:
                return 1 if not target_left else 0
            else:
                memo[(d_left, target_left)] = 0
                for face in range(1, min(f+1, target_left+1)):
                    memo[(d_left, target_left)] += dfs(d_left-1, target_left-face)
                return memo[(d_left, target_left)]
        return dfs() % MOD
class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        
        BASE = 10**9 + 7
        
        memo = {(0,0): 1}
        def dfs(num, tar):
            
            nonlocal BASE
            
            if (num, tar) in memo:
                return memo[(num, tar)]
            
            if num == 0 or tar == 0:
                return -1
            
            cnt = 0
            for i in range(1, f+1):
                sub = dfs(num - 1, tar-i)
                if sub != -1: cnt += sub
            memo[(num, tar)] = cnt % BASE
            
            return memo[(num, tar)]
    
        dfs(d, target)
        #print(memo)
        return memo[(d, target)]
# from (D * F * target) -> O(D * target)
class Solution:
    def numRollsToTarget(self, D: int, F: int, target: int) -> int:
        # dp[d][t] -> how many ways to form t using d dices
        dp = [[0] * (target + 1) for _ in range(2)]
        dp[0][0] = 1
        for d in range(1, D + 1):
            cd = d & 0x1
            pd = (d - 1) & 0x1
            dp[cd][0] = 0
            for t in range(1, target + 1):
                dp[cd][t] = (dp[cd][t - 1] + dp[pd][t - 1] - (dp[pd][t - F - 1] if t - F - 1 >= 0 else 0)) % 1000000007
        return dp[D & 0x1][target]
    
    def numRollsToTarget(self, D: int, F: int, target: int) -> int:
        dp = [0] * (target + 1)
        dp[0] = 1
        for d in range(1, D + 1):
            ndp = [0] * (target + 1)
            for i in range(1, target + 1):    
                ndp[i] = sum(dp[i - f] for f in range(1, F + 1) if i - f >= 0) % 1000000007
            dp = ndp
        return dp[-1]
class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        self.map = collections.defaultdict(int)
        return self.noOfWays(d,f, target)
        
    def noOfWays(self,d, f, target):
           
        if(d==0 and target==0):
            return 1
        elif(d==0):
            return 0

        if((d,target) in self.map):
            return self.map[(d,target)]
        res = 0
        for i in range(1, f+1):
            res+= (self.noOfWays(d-1,f, target-i))
        self.map[(d,target)] = res  
        return res%(10**9 + 7)

MOD = 10 ** 9 + 7

class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        ways = [0] + [1] * min(f, target) + [0] * max(target - f, 0)
        for _ in range(d - 1):
            for i in reversed(range(1, target + 1)):
                ways[i] = 0
                for j in range(1, min(i, f + 1)):
                    ways[i] = (ways[i] + ways[i - j]) % MOD
        return ways[target]
class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        memo = {}
        mod = int(math.pow(10,9)+7)
        def recur(d,target):
            if (d,target) in memo:
                return memo[(d,target)]
            if d<0 or target<0:
                return 0
            if d == 0 and target == 0:
                return 1
            ways = 0
            for i in range(1,f+1):
                if target-i < 0:
                    break
                ways = int(ways + recur(d-1,target-i))%mod
            memo[(d,target)] = ways
            return ways
        return recur(d,target)

class Solution:
    @lru_cache(None)
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        # dp[i][k]: the number of ways to get target "k" using "i" dices
        # dp[0][0] = 1
        # dp[i][k] = dp[i-1][k] + sum(dp[i-1][k - ff] for ff in range(1, f + 1))
        dp = [1] + [0] * target
        mod = 10**9 + 7
        for _ in range(d):
            for i in range(target, -1, -1):
                dp[i] = sum(dp[i-ff] for ff in range(1, f + 1) if i >= ff) % mod
        return dp[-1] % mod
        # for i in range(1, d + 1):
        #     for j in range(1, target + 1):
                # dp[i]
        
        dp = [[0] * (target + 1) for _ in range(d + 1)]
        mod = 10 ** 9 + 7
        dp[0][0] = 1
        for i in range(1, d + 1):
            for j in range(1, target + 1):
                # if i == 0 and j == 0:
                #     dp[i][j] = 1
                # elif j == 0:
                #     dp[i][j] = 1
                # elif i == 0:
                #     continue
                # else:
                dp[i][j] = (sum(dp[i-1][j-ff] for ff in range(1, f + 1) if j >= ff))
        return dp[-1][-1] % mod
        
        
        # if d == 0 or target <= 0:
        #     return d == 0 and target == 0
        # res = 0
        # for i in range(1, f+1):
        #     res = (res + self.numRollsToTarget(d - 1, f, target - i)) % (10**9 + 7)
        # return res

class Solution:
    def __init__(self):
        self.memo = {}
        
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        res = 0
        if (d, target) in self.memo:
            return self.memo[(d, target)]
        for i in range(1, f + 1):
            if target - i == 0 and d == 1:
                res += 1
            elif target - i > 0 and d > 1:
                res += self.numRollsToTarget(d - 1, f, target - i)
        self.memo[(d, target)] = res
        return res % (10 ** 9 + 7)
class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        dp = [[0 for _ in range(target+1)] for _ in range(d+1)]
        
        for dd in range(1, d+1):
            for tt in range(dd, min(f*dd, target)+1):
                if dd == 1:
                    dp[dd][tt] = 1
                else:
                    dp[dd][tt] = sum(dp[dd-1][max(1, tt-f):tt])
                    
        return dp[d][target] % (10**9+7)
class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        self.cache = {}    
        return self.helper(0, 0, d, f, target) % (10**9 + 7)
    
    def helper(self, currD, currSum, d, f, target):
        if currD == d:
            return currSum == target
        
        if currD in self.cache and currSum in self.cache[currD]:
            return self.cache[currD][currSum]
        
        ways = 0
        for i in range(1, f+1):
            ways += self.helper(currD+1, currSum + i, d, f, target)
        
        if currD not in self.cache:
            self.cache[currD] = {}
        self.cache[currD][currSum] = ways
        return ways
class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        
        dp = [[0] * (target+1) for _ in range(d+1)]
        dp[0][0] = 1
        for i in range(1, d+1):
            for j in range(1, target+1):
                dp[i][j] = sum(dp[i-1][j-k] for k in range(1, min(f,j)+1) )            
        return dp[-1][-1] % (10**9 + 7)
    
#         if target > d*f:
#             return 0
#         dicti = collections.defaultdict(int)
#         def dice_target(rem_dice, summ):
#             if rem_dice == 0:
#                 return 1 if summ == target else 0
#             if summ > target:
#                 return 0
#             if (rem_dice, summ) in dicti:
#                 return dicti[rem_dice, summ]

#             for i in range(1, f+1):
#                 dicti[rem_dice, summ] += dice_target(rem_dice-1, summ+i)
#             return dicti[rem_dice, summ]
        
        
#         return dice_target(d, 0) % (10**9 + 7)


from math import comb

class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        dic = {}
        return self.helper(d,f,target,dic)
    
    def helper(self, d2, f2, target2,dic):
        if target2 == 0 and d2 == 0:
            return 1
        if (d2,f2,target2) in dic:
            return dic[(d2,f2,target2)]
        if target2 < d2 or d2*f2 < target2:
            dic[(d2,f2,target2)] = 0
            return 0
        elif d2 == 1:
            dic[(d2,f2,target2)] = 1
            return 1
        tot = 0
        for i in range(0,d2+1):
            num_poss = self.helper(d2-i,f2-1,target2 - i * f2,dic) * comb(d2,i)
            tot += num_poss % (10**9 + 7)
        dic[(d2,f2,target2)] = tot % (10**9 + 7)
        return tot % (10**9 + 7)
                

class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        dp = [[0 for x in range(d + 1)]for y in range(target + 1)]
        for i in range(1, min(f + 1, target + 1)):
            dp[i][1] = 1
        for i in range(1, target + 1):
            for j in range(2, d + 1):
                val = 0
                for k in range(1, f + 1):
                    if i - k >= 0:
                        val += dp[i - k][j - 1]
                dp[i][j] = val
        return dp[target][d]%(10**9 + 7)
class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        #dp = [[0 for _ in range(target+1)] for _ in range(d+1)]
        dp = [0 for _ in range(target+1)]
        dp[0] = 1
        #dp[0][0] = 1
        
        for i in range(1, d+1):
            new = [0]
            for j in range(1, target+1):
                new.append(0)
                for k in range(1, f+1):
                    if j-k>=0:
                        new[-1]+=dp[j-k]
                        #dp[i][j]+=dp[i-1][j-k]
            dp = new             
        return dp[-1]%(10**9+7)  
        return dp[-1][-1]%(10**9+7)
                
                
    
    '''
    f = 6
    
     01234567
    010000000
    101111110
    200123456
         
         
         
    '''
class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        dp = [[0]*(target+1) for i in range(d+1)]
        for i in range(1,d+1):
            for j in range(1,target+1):
                if i ==1:
                    if j<=f:
                        dp[i][j]=1
                else:
                    if j>=i:
                        for k in range(1,min(j+1,f+1)):
                            dp[i][j] += dp[i-1][j-k]
        
        return (dp[-1][-1]%((10**9)+7))
class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        dp = [[0 for x in range(target+1)] for y in range(d+1)]
        dp[0][0] = 1
        
        for i in range(1, d+1):
            for j in range(1, target+1):
                total_sum = 0
                for k in range(1, f+1):
                    if j-k >= 0:
                        total_sum += dp[i-1][j-k]
                dp[i][j] = total_sum
        return dp[d][target] % (10**9 + 7)
            
        
        
        
        
        # memo = {}
        # def dp(d, target):
        #     if d == 0:
        #         return 0 if target > 0 else 1
        #     if (d, target) in memo:
        #         return memo[(d, target)]
        #     to_return = 0
        #     for k in range(max(0, target-f), target):
        #         to_return += dp(d-1, k)
        #     memo[(d, target)] = to_return
        #     return to_return
        # return dp(d, target) % (10**9 + 7)

class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        dp = [[0 for j in range(target+1)] for i in range(d+1)]
        dp[0][0] = 1
        for i in range(1, d+1):
            for j in range(1, target+1):
                dp[i][j] = sum([dp[i-1][j-k] for k in range(1, 1+min(f, j))])
        return dp[d][target] % (10**9 + 7)
class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        dp = [[0 for _ in range(target+1)]]
        #dp.append([0 for _ in range(target+1)])
        for i in range(1,min(f+1,target+1)):
            dp[0][i] = 1 
        for i in range(2,d+1):
            dp.append([0 for _ in range(target+1)])
            for t in range(i,target+1):
                for d in range(1,min(t,f+1,target+1)):
                    if t - d >= 0 and dp[-2][t-d]:
                        dp[-1][t] += dp[-2][t - d]
                    else:
                        #print('')
                        0
            dp.pop(0)
        for row in dp:
            #print(row)
            0
        return (dp[0][-1]) % (10**9 + 7)
class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        memo = [0]*(target+1)
        memo[0] = 1
        maxV = 10**9+7
        for _ in range(d):
            for tv in range(target,-1,-1):
                memo[tv] = 0 #since have must use all dices! So reaching tv with less than d dices doesn't count.
                for fv in range(1, f+1):
                    memo[tv] += memo[tv-fv] if tv-fv>=0 else 0
                memo[tv] %= maxV 
        return memo[-1]
        
        '''
        memo = {}
        x = 10**9+7
        def bt(remD, remT):
            if remT<0 or remD<0:
                return 0
            if (remD, remT) in memo:
                return memo[(remD, remT)]
            if remD==0:
                return 1 if remT==0 else 0
            temp = 0
            for i in range(1, f+1):
                temp += bt(remD-1, remT-i)
            temp %= x
            memo[(remD, remT)] = temp
            return temp
        
        return bt(d, target)
        '''
    
    '''
    (a+b)%c = (a%c+b%c)%c
    
    '''

class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        modulo = 10**9 + 7
        dp = [[0 for i in range(target+1)] for j in range(d+1)]
        for dd in range(1, d+1):
            for tt in range(dd, min(target, dd * f)+1):
                if dd == 1:
                    dp[dd][tt] = 1
                else:
                    start =max( tt - f, 1)
                    end = tt - 1
                    dp[dd][tt] = sum(dp[dd-1][start:end+1])
                    
        return dp[d][target] % modulo
'''
class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        modulo = 10**9 + 7
        #don't use row 0 and all column 0 
        dp = [[0 for i in range(target+1)] for j in range(d+1)]
        
        #from die No. 1 to d
        for dd in range(1, d+1):
            #for target from 0 to min(f*dd, target)
            for tt in range(dd, min(f * dd, target) + 1 ):
                if dd == 1:
                    dp[dd][tt] = 1
                else:
                    end   = tt - 1
                    start = max(1, tt - f)
                    dp[dd][tt] = sum(dp[dd-1][start:end+1])
    
        return dp[d][target] % modulo                          
        
  
        #f(d, target) = f(d-1, target-1) + f(d-1, target-2) + ... + f(d-1, target-f)  assuming target > f
        modulo = 10**9 + 7
        cache = {}
        def numRollsToTargetHelper(dd, tt):
            if cache.get((dd,tt)) != None:
                return cache[(dd,tt)]
            nonlocal f
            if dd == 1:
                if tt <= f:
                    return 1
                else:
                    return 0
            
            ret = 0
            for i in range(1, f+1):
                if tt - i > 0:
                    ret += numRollsToTargetHelper(dd-1, tt-i)
            cache[(dd,tt)] = ret
            return ret
        
        ret = numRollsToTargetHelper(d, target)
        return ret % modulo
        '''       
class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        
        n = target
        ways = [[0 for i in range(n+1)] for j in range(d+1)]
        
        ways[0][0] = 1
        
        for i in range(1,d+1):
            for j in range(n+1):
                c = 0
                for k in range(1,f+1):
                    c += ways[i-1][j-k] if j-k>=0 else 0
                    
                ways[i][j] = c
                
        #print(ways)
        return ways[d][n]%(10**9+7)
class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        dp = [[0 for _ in range(target+1)]]
        #dp.append([0 for _ in range(target+1)])
        for i in range(1,min(f+1,target+1)):
            dp[0][i] = 1 
        for i in range(2,d+1):
            dp.append([0 for _ in range(target+1)])
            for t in range(i,target+1):
                for d in range(1,min(t,f+1,target+1)):
                    if t - d >= 0 and dp[-2][t-d]:
                        dp[-1][t] += dp[-2][t - d]
            dp.pop(0)
        return (dp[0][-1]) % (10**9 + 7)
class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        MOD = 10 ** 9 + 7
        dp = [0] * (target + 1)
        dp[0] = 1
        for i in range(1, d + 1):
            for t in reversed(range(target + 1)):
                dp[t] = sum(dp[t - j] if t - j >= 0 else 0 for j in range(1, f + 1)) % MOD
        return dp[target]
class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        dp = [0] * (target + 1)
        dp[0] = 1
        M = 10 ** 9  + 7
        for i in range(d):
            n_dp = [0] * (target + 1)
            for j in range(target + 1):
                for m in range(1, f + 1):
                    if j + m > target:
                        break
                    
                    n_dp[j + m] += dp[j]
            
            dp = [k % M for k in n_dp]
        
        return dp[-1]
from collections import defaultdict

class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        memo = defaultdict(int)
        
        def helper(d, target):
            if d == 0:
                return 1 if target == 0 else 0
            if (d, target) in memo:
                return memo[(d, target)]


            for c in range(1, f + 1):
            #for c in range(max(0, target-f), target):
                memo[(d, target)] += helper(d - 1, target - c)
                #memo[(d, target)] += helper(d - 1, c)   
            return memo[(d, target)]
        return helper(d, target) % (10**9 + 7)
    

class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        
        dp = [[0] * (target+1) for _ in range(d+1)]
        dp[0][0] = 1
        for i in range(1, d+1):
            for j in range(1, target+1):
                for k in range(1, min(f,j)+1):
                    dp[i][j] += dp[i-1][j-k]         
            # print(dp)
        return dp[-1][-1] % (10**9 + 7)
    
#         if target > d*f:
#             return 0
#         dicti = collections.defaultdict(int)
#         def dice_target(rem_dice, summ):
#             if rem_dice == 0:
#                 return 1 if summ == target else 0
#             if summ > target:
#                 return 0
#             if (rem_dice, summ) in dicti:
#                 return dicti[rem_dice, summ]

#             for i in range(1, f+1):
#                 dicti[rem_dice, summ] += dice_target(rem_dice-1, summ+i)
#             return dicti[rem_dice, summ]
        
        
#         return dice_target(d, 0) % (10**9 + 7)


class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        dp = [0] * (target + 1)
        dp[0] = 1
        for i in range(d):
            dp_new = [0] * len(dp)
            for i in range(1, len(dp)):
                for j in range(1, f + 1):
                    if i - j >= 0:
                        dp_new[i] = (dp_new[i] + dp[i - j]) % (10**9 + 7)
            dp = dp_new
        return dp[-1]  % (10**9 + 7)

class Solution:
    cache = {}
    
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        if d == 1:
            if f < target:
                return 0
            else:
                return 1

        if (d, f, target) in self.cache:
            return self.cache[(d, f, target)]
        else:
            num = sum(self.numRollsToTarget(d-1, f, target-i) for i in range(1, min(f+1, target))) % (10**9 + 7)
            self.cache[(d, f, target)] = num
        
        return num
        

class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        memo = {}
        def dp(d, target):
            if d == 0:
                return 0 if target > 0 else 1
            if (d, target) in memo:
                return memo[(d, target)]
            to_return = 0
            for k in range(max(0, target-f), target):
                to_return += dp(d-1, k)
            memo[(d, target)] = to_return
            return to_return
        return dp(d, target) % (10**9 + 7)
class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        if target < d or target > d * f:
            return 0
        
        total_sum = 0
        cache_sum = []
        
        if target > d * (1 + f) / 2:
            target = d * (1 + f) - target
        
        for i in range(target, -1, -1):
            num = 0 if i > f else 1
            total_sum += num; 
            cache_sum.append(total_sum)
        cache_sum = cache_sum[::-1]
        cache_sum[0] = cache_sum[1]
        
        for i in range(2, d+1):
            total_sum = 0
            for j in range(target, -1, -1):
                total_sum += cache_sum[max(j-f, 0)] - cache_sum[j]
                cache_sum[j] = total_sum
        return cache_sum[-1] % (10**9+7)
class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        modulo = 10**9 + 7
        dp = [[0 for i in range(target+1)] for j in range(d+1)]
        
        for dd in range(1, d+1):
            for tt in range(dd, min(f*dd, target) + 1):
                if dd == 1:
                    dp[dd][tt] = 1
                else:
                    end = tt - 1
                    start = max(1, tt-f)
                    dp[dd][tt] = sum(dp[dd-1][start:end+1])
        
        return dp[d][target] % modulo
class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        dp = [[0] * (d + 1) for _ in range(target+1)]
        dp[0][0] = 1
        for i in range(1, target+1):
            for j in range(1, d + 1):
                if i >= j:
                    for num in range(1, min(i, f) + 1):
                        dp[i][j] += dp[i-num][j - 1]
        return dp[-1][-1] % (10 ** 9 + 7)

class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        dp = [[0 for j in range(target+1)] for i in range(d)]
        for j in range(1, min(f, target)+1):
            dp[0][j] = 1
        for i in range(1, d):
            for j in range(1, target+1):
                for k in range(1, f+1):
                    if j < k:
                        break
                    dp[i][j] += dp[i-1][j-k]
        return dp[d-1][target] % (10**9 + 7)
                
                

from collections import defaultdict

class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        memo = defaultdict(int)
        
        def helper(d, target):
            if d == 0:
                return 1 if target == 0 else 0
            if (d, target) in memo:
                return memo[(d, target)]
            else:
                for c in range(1, f + 1):
                    memo[(d, target)] += helper(d - 1, target - c)
                return memo[(d, target)]
        return helper(d, target) % (10**9 + 7)
    

class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        dp={}
        for k in range(1,f+1):
            dp[(1,k)]=1
            
        def dfs_num_rolls(dice_left,tot_left):

            if tot_left>f**(dice_left) or tot_left<0 or dice_left==0:
                dp[(dice_left,tot_left)]=0
                return 0
            if (dice_left,tot_left) in dp:
                return dp[(dice_left,tot_left)]
            
            total=0
            for k in range(1,f+1):
                total+=dfs_num_rolls(dice_left-1,tot_left-k)%(10**9+7)
            
            total=total%(10**9+7)
            dp[(dice_left,tot_left)]=total
            return total
        
        dfs_num_rolls(d,target)
        return dp[(d,target)]

class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        dp = [[0 for _ in range(target+1)] for _ in range(d+1)]
        dp[0][0] = 1 #d=0 and target=0
        
        for i in range(1,d+1):
            for j in range(1,target+1):
                for k in range(1,min(j,f)+1):
                    dp[i][j] += dp[i-1][j-k]
                    
        return dp[d][target]%(10**9+7)
        

class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        memo = {}
        def rec(d,target):
            if d==0:
                if target==0: 
                    return 1
                else:
                    return 0
            if target<=0:
                return 0
            if (d,target) in memo:
                return memo[(d,target)]
            count = 0
            for i in range(1,f+1):
                count += rec(d-1, target-i)
            memo[(d,target)] = count   
            return count
        return rec(d,target)%((10**9)+7)
        

class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        #dp = [[0 for _ in range(target+1)] for _ in range(d+1)]
        dp = [0 for _ in range(target+1)]
        dp[0] = 1
        #dp[0][0] = 1
        
        for i in range(1, d+1):
            new = [0]
            for j in range(1, target+1):
                new.append(0)
                for k in range(1, f+1):
                    if j-k>=0:
                        new[-1]=(new[-1]+dp[j-k])%(10**9+7)
                        #dp[i][j]+=dp[i-1][j-k]
                    else:
                        break
            dp = new             
        return dp[-1]
        return dp[-1][-1]%(10**9+7)
                
                
    
    '''
    f = 6
    
     01234567
    010000000
    101111110
    200123456
         
         
         
    '''
class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        
        
        dp = [ [0]*(target+1) for _ in range(d) ]
        
        l = min(target+1, f+1)
        
        for i in range(1,l):
            dp[0][i] = 1
        
        for level in range(1,d):
            
            #print(dp[level-1], dp[level])
            for face in range(1,f+1):
                for t in range(1,target+1):
                
                
                #for face in range(1,f+1):
                    
                    if t - face > 0:
                        dp[level][t] += dp[level-1][t-face]
        
        #print(dp)
        return dp[-1][-1] % (10**9 + 7)
        
        

class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        dp = [[0 for _ in range(target+1)] for _ in range(d+1)]
        dp[0][0] = 1 #d=0 and target=0
        
        for i in range(1,d+1):
            for j in range(1,target+1):
                for k in range(1,min(j,f)+1):
                    dp[i][j] += dp[i-1][j-k]
                    
        return dp[d][target]%(10**9+7)
        
        

class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        dp = [[0 for x in range(target + 1)] for y in range(d + 1)]
        dp[0][0] = 1
        for i in range(1, d + 1):
            for j in range(1, target + 1):
                for k in range(1, min(j, f) + 1):
                    dp[i][j] += dp[i - 1][j - k]
        return dp[d][target] % (10**9+7)
class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        dp = [[0 for _ in range(target+1)] for _ in range(d+1)]
        dp[0][0] = 1
        
        for i in range(1,d+1):
            for j in range(1, target+1):
                for k in range(1, f+1):
                    if k<=j:
                        dp[i][j] += dp[i-1][j-k]
        
        
        return dp[d][target] % (10**9 + 7)
class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        def helper(d, target, map):
            if (d,target) in map:
                return map[(d,target)]
            if d == target:
                return 1
            elif d == 0 or target < d:
                return 0
            map[(d,target)] = 0
            for num in range(1,f+1):
                map[(d,target)] += helper((d-1), (target-num), map)
            return map[(d,target)]
        helperMap = {}
        return (helper(d,target,helperMap)%(10**9+7))
class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        
        dp = [[0 for _ in range(target+1)] for _ in range(d+1)]
        dp[0][0] = 1
        m = (10**9 + 7)
        
        for i in range(1, d+1):
            for j in range(1, target+1):
                for k in range(1, f+1):
                    if k<=j:
                        dp[i][j] += dp[i-1][j-k]

        return dp[d][target]%m;
                        
        

class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        mod = 10**9 + 7
        dp = [[0] * (target + 1) for i in range(d + 1)]
        
        dp[0][0] = 1
        for i in range(1, d + 1):
            nxt = [[0] * (target + 1) for i in range(d + 1)]
            for j in range(1, target + 1):
                nxt[i][j] = sum(dp[i-1][j - x] for x in range(1, f + 1) if j >= x) % mod
            dp = nxt
        return dp[-1][-1]
                
        
        
        # mod = 10**9 + 7
        # @lru_cache(None)
        # def dfs(d, curr):
        #     if d == 0:
        #         return curr == 0
        #     if d < 0 or curr < 0:
        #         return 0
        #     return sum(dfs(d - 1, curr - x) for x in range(1, f + 1)) % mod
        # return dfs(d, target)

class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        
        
        dp = [ [0]*(target+1) for _ in range(d) ]
        
        l = min(target+1, f+1)
        
        for i in range(1,l):
            dp[0][i] = 1
        
        for level in range(1,d):

            #for face in range(1,f+1):
            for t in range(1,target+1):
                
                
                for face in range(1,f+1):
                    
                    if t - face > 0:
                        dp[level][t] += dp[level-1][t-face]
        
        #print(dp)
        return dp[-1][-1] % (10**9 + 7)
        
        

class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        #dp = [[0 for _ in range(target+1)] for _ in range(d+1)]
        dp = [0 for _ in range(target+1)]
        dp[0] = 1
        #dp[0][0] = 1
        
        for i in range(1, d+1):
            new = [0]
            for j in range(1, target+1):
                new.append(0)
                for k in range(1, f+1):
                    if j-k>=0:
                        new[-1]=(new[-1]+dp[j-k])%(10**9+7)
                        #dp[i][j]+=dp[i-1][j-k]
            dp = new             
        return dp[-1]
        return dp[-1][-1]%(10**9+7)
                
                
    
    '''
    f = 6
    
     01234567
    010000000
    101111110
    200123456
         
         
         
    '''
class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        # # 1-d
        # dp = [0] * (target+1)
        # for i in range(1,min(f+1,target+1)):
        #     dp[i] = 1
        # for _ in range(d-1):
        #     for j in reversed(range(1,len(dp))):
        #         dp[j] = 0
        #         for num in range(1,f+1):
        #             if 0<=j-num:
        #                 dp[j] += dp[j-num]
        # return dp[-1] % (10**9 + 7)
        
        
        # 2-d
        dp = [[0]*(target+1) for i in range(d)]
        for i in range(1,min(target+1,f+1)):
            dp[0][i] = 1
        for i in range(1,d):
            for j in range(1,target+1):
                for num in range(1,f+1):
                    if j-num >= 0:
                        dp[i][j] += dp[i-1][j-num]
        return dp[-1][-1] % (10**9 + 7)
    
    
    
    



class Solution:
    # mod = 1e9+7

        
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        dp = dict()
        
        def fun(d, target):
            mod = int(1e9+7)
            if(target==0 and d==0):
                return 1
            if(target==0 and d!=0):
                return 0
            if(target<0):
                return 0
            if(d==0):
                return 0
            if((d, target) in dp):
                return dp[d, target]
            tmp = 0
            for i in range(1,f+1):
                tmp += (fun(d-1, target-i))%mod
                tmp %= mod
            dp[d, target] = tmp
            return tmp
        return fun(d, target)
class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        dp = [[0]*(target + 1) for i in range(d+1)]
        dp[0][0] = 1
        for i in range(1, d+1):
            for j in range(1, target+1):
                for k in range(1, min(j, f) + 1):
                    dp[i][j] =  dp[i][j] + dp[i-1][j-k]
                    
        return dp[i][j] % (10**9 + 7)

class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        arr = [[0 for i in range(target+1)] for i in range(d)]
        
        for i in range(1, min(f, target)+1):
            arr[0][i] = 1
            
        for row in range(1, d):
            temp = 0
            for col in range(row+1, target+1):
                temp += arr[row-1][col-1]
                if col >= f+1:
                    temp -= arr[row-1][col-f-1]

                arr[row][col] = temp
                
        # for i in range(len(arr)):
        #     print(i, arr[i])
            
        return arr[d-1][target]%(1000000007)
class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        if(d>target or (d==1 and target>f)):
            return 0
        if(d == target or d == 1):
            return 1
        dp = [[0 for i in range(target+1)] for j in range(2)]
        dp_1 = [0 for i in range(target+1)]
        for i in range(1,min(target+1,f+1)):
            dp_1[i] = 1
        dp[0] = dp_1
        for i in range(2,d+1):
            dp[1] = [0 for i in range(target+1)]
            for j in range(i,target+1):
                if(i==j):
                    dp[1][j] = 1
                    continue
                for k in range(1,min(j+1,f+1)):
                    dp[1][j] += dp[0][j-k]*dp_1[k]
            dp[0] = dp[1]
        # print(dp,dp_1)
        return dp[1][-1]%1000000007
class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        dp=[[0 for i in range(target+1)] for j in range(d+1)]
        dp[0][0]=1
        for i in range(1,d+1):
            for j in range(1,target+1):
                for k in range(1,f+1):
                    
                    if(j>=k):
                        dp[i][j]+=dp[i-1][j-k]
                        
        return dp[-1][-1]%((10**9)+7)

class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        mod = int(10**9 + 7)
        dp = [[0] * (target+1) for _ in range(d+1)] 
        for j in range(1, min(f+1, target+1)): dp[1][j] = 1
        for i in range(2, d+1):
            for j in range(1, target+1):
                for k in range(1, f+1):
                    if j - k >= 0: dp[i][j] += dp[i-1][j-k]
                dp[i][j] %= mod        
        return dp[-1][-1]

class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        dp = [[0 for _ in range(target + 1)] for _ in range(d + 1)]
        
        dp[0][0] = 1
        
        # Iterate through every dice
        for i in range(1, d+1):
            
            # iterate through every target
            for j in range(1, target+1):
                
                # go through every possible value on dice
                for k in range(1, f+1):
                    if k <= j:
                        dp[i][j] += dp[i-1][j - k]
                        
        return dp[-1][-1] % (10**9 + 7)
class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        dp = [[0]*(target+1) for i in range(d+1)]
        dp[0][0] = 1
        mod = (10**9)+7
        for i in range(1,d+1):
            for j in range(target+1):
                for k in range(1,f+1):
                    if j-k >=0:
                        dp[i][j] += dp[i-1][j-k]
        return dp[d][target] % mod

class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        dp = [[0] * target for _ in range(d)]
        
        for i in range(d):
            for j in range(target):
                if i == 0 and j+1 <= f:
                    dp[i][j] = 1
                elif i == 0:
                    dp[i][j] = 0
                elif j >= i:
                    for k in range(1, f+1):
                        if j - k >= 0:
                            dp[i][j] += dp[i-1][j-k]
        
        # print(dp)
        return dp[d-1][target-1] % 1000000007
class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        if target == 0:
            return 0
        dp = [[0] * (target + 1) for _ in range(d + 1)]
        dp[0][0] = 1
        for i in range(1, d + 1):
            for j in range(1, target + 1):
                for num in range(1, f + 1):
                    if j - num >= 0:
                        dp[i][j] += dp[i-1][j-num]
        
        return dp[d][target] % (10**9 + 7)
class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        
        memo = {}
        
        def dpWays(d, target):
            
            if d == 0:
                
                return(0 if target > 0 else 1)
            
            if (d, target) in memo:
                
                return(memo[(d, target)])
            
            to_return = 0
            
            for k in range(max(0, target - f), target):
                
                to_return += dpWays(d-1, k)
                
            memo[(d, target)] = to_return
            
            return(to_return)
        
        return(dpWays(d, target) % (10**9 + 7))
class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        # dp(depth, f, target) = 1+dp(depth-1, f, target-face_chosen) for f_c in range(1,f+1)
        
        opt = [[0]*(target+1) for _ in range(d+1)]
        opt[0][0]=1
        mod = 10**9 +7
        for i in range(1, d+1):
            for j in range(1, target+1):
                # we need to get sum of all arrows out
                sum_children = 0
                face_val = 1
                while face_val <=f and j-face_val>=0 :
                    sum_children+= opt[i-1][j-face_val]%mod
                    face_val+=1
                opt[i][j] = sum_children%mod
        # print(opt[1])
        # print(opt[0])
        return opt[d][target]
                    
        
        
#         for d in range()
        
        
        
        
        
#         opt(i, j) = 1+sum(opt(i-1, j-face) for face in range(1, f+1))
        
        
        
#         we want opt(d=d, f=f, sum_=target)
        
        
        
#         prev_map = dict()
#         for face in range(2, f+1):
#             prev_map[face]=1
        
#         for dice in range(1, d+1):
#             new_map = dict()
#             for face in range(1, face):
#                 for i, count in prev_map.items():
#                     new_map[face+i]=count+i
#             new_map = prev_map
            
#         print(prev_map)
#         return prev_map[target]
                
            

class Solution:
    # Recursive memoized solution
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        memo = {}
        def num_rolls_util(level, target):
            if level * f < target or target < level:
                return 0
            if level == 0:
                return 1
            
            res = 0
            for i in range(max(0, target - f), target):
                if (level-1, i) in memo:
                    res += memo[(level-1, i)]
                else:
                    tmp = num_rolls_util(level - 1, i)
                    memo[(level-1, i)] = tmp
                    res += tmp

            return res % (10 ** 9 + 7)
        
        return num_rolls_util(d, target)
    
    
    '''
    if target < d or target > d * f:
            return 0
        if target > (d*(1+f)/2):
            target = d * (1 + f) - target
        dp = [0] * (target + 1) 
        for i in range(1, min(f, target) + 1):
            dp[i] = 1
        for i in range(2, d + 1):
            new_dp = [0] * (target + 1)
            for j in range(i, min(target, i * f) + 1):
                new_dp[j] = new_dp[j - 1] + dp[j - 1]
                if j - 1 > f:
                    new_dp[j] -= dp[j - f - 1]
            dp = new_dp

        return dp[target] % (10 ** 9 + 7)
    '''
class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        dp = [[0 for _ in range(max(target+1,f+1))] for _ in range(d+1)]
        # boundary condition dp[1][1],dp[1][2],...,dp[1][f]
        for i in range(1,f+1):
            dp[1][i] = 1
        # dp[dice][values]= dp[dice-1][values-1] + 
        for dice in range(1,d+1):
            for values in range(1,target+1):
                for i in range(1,f+1):
                    if values-i <1:
                        break
                    dp[dice][values]+=dp[dice-1][values-i]
        return dp[d][target] % (10**9+7)
                

class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        # dp[i][j]: u7528iu4e2au9ab0u5b50u6254u51fau548cu4e3aju7684u65b9u6cd5u603bu6570
        dp = [[0] * (target+1) for _ in range(d+1)]
        m = 10**9+7
        dp[0][0] = 1 # u7531u4e8eu6bcfu4e2au9ab0u5b50u5fc5u9009uff0cdp[i][0], i != 0u7684u610fu601du662fuff0cu524diu4e2au9ab0u5b50u90fdu4e0du9009uff0cu662fu4e0du53efu4ee5u7684uff0cu6240u4ee5u4e3a0
        for i in range(1, d+1): 
            for j in range(i, target-d+i+1): # u7531u4e8eu540eu9762u6709d-ju4e2au9ab0u5b50uff0cu6240u4ee5u6211u4eecu524diu4e2au4e0du7528u6254u6ee1target
                for k in range(1, min(j, f)+1): # u53eau53efu80fdu6254u5230juff0cu5982u679cfu8d85u8fc7u4e86juff0cu6211u4eecu4e0du8981
                    dp[i][j] += dp[i-1][j-k] % m

        return dp[d][target] % m


class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        z = 10**9+7
        def sol(n,m,t):
            
            dp=[[0 for i in range(t+1)] for j in range(n+1)]
            
            for i in range(1,min(t+1,m+1)):
                dp[1][i]=1
            
            for i in range(1,n+1):
                for j in range(1,t+1):
                    for k in range(1,min(j,m+1)):
                           dp[i][j]+=(dp[i-1][j-k])% z
                    
            return dp[n][t]%z
                          
        return sol(d,f,target)%z
                          

class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        
        dp = [[0 for j in range(target + 1)] for i in range(d + 1)]
        dp[0][0] = 1
        for i in range(d):
            for ff in range(1, f + 1):
                for sm in range(target):
                    if sm + ff <= target:
                        dp[i + 1][sm + ff] += dp[i][sm]
        return dp[d][target]%(10**9 + 7)


class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        dp = [[0 for _ in range(target+1)] for _ in range(d+1)]
        
        dp[0][0] = 1
        
        for i in range(1, d+1):
            for j in range(1, target+1):
                for k in range(1, f+1):
                    if j-k>=0:
                        dp[i][j]+=dp[i-1][j-k]
        return dp[-1][-1]%(10**9+7)
                
                
    
    '''
    f = 6
    
     01234567
    010000000
    101111110
    200123456
         
         
         
    '''
class Solution:
    def numRollsToTarget(self, d, f, target):
        
        if d == 0:
            return 0
        
        dp = [[0 for _ in range(target+1)] for _ in range(d+1)]
        
        
        
        for i in range(1, target+1):
            dp[0][i] = 0
        for j in range(1, d+1):
            dp[j][0] = 0
            
        dp[0][0] = 1
        
        for i in range(1, d+1):
            for j in range(1, target+1):
                for k in range(1, f+1):
                    if j-k >= 0:
                        dp[i][j] += dp[i-1][j-k]
                        
        return dp[d][target] % ((10 ** 9) + 7)
                
                
#         def recursive(d, f, target):
            
#             if target == 0 and d == 0:
#                 return 1
            
#             if target < 0 or d == 0:
#                 return 0
            
#             if dp[d][target] != -1:
#                 return dp[d][target]
#             temp = 0
#             for i in range(1, f+1):
#                 temp += recursive(d - 1, f, target - i)
                
#             dp[d][target] =  temp
#             return dp[d][target]

#         return (recursive(d, f, target) % ((10 ** 9) + 7))

class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        def dfs(d, target):
            if (d, target) in cache:
                return cache[(d, target)]
            if target < 0 or d < 0:
                return 0
            if target == 0 and d == 0:
                return 1
            ways = 0
            for i in range(1, f + 1):
                ways += dfs(d - 1, target - i)
            cache[(d, target)] = ways
            return ways

        cache = {}
        mod = 10 ** 9 + 7
        return dfs(d, target) % mod

        # mod = 10 ** 9 + 7
        # dp = [[0] * (target + 1) for _ in range(d + 1)]
        # # dp[0][0] = 1
        # for i in range(1, min(f, target) + 1):
        #     dp[1][i] = 1
        # for i in range(2, d + 1):
        #     # the numbers that can be reached by rolling i dice
        #     for num in range(i, min(i * f, target) + 1):
        #         for diceNum in range(1, (f + 1)):
        #             if num - diceNum >= 1:
        #                 dp[i][num] += dp[i - 1][num - diceNum]
        #             else:
        #                 break
        # return dp[d][target] % mod


class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        
        MOD=10**9+7
        dp=[[0 for i in range(target+1)] for i in range(d+1)]
        
        for j in range(1,min(f+1,target+1)):
            dp[1][j]=1
        
        for i in range(1,d+1):
            for j in range(target+1):
                for k in range(1,min(f+1,j)):
                    dp[i][j]+=dp[i-1][j-k]%MOD
        
        return dp[d][target]%MOD
class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        memo={}
        def dp(dice,target):
            if((dice,target) in memo):
                return memo[(dice,target)]
            if(dice<=0 or target<=0):
                return 0
            if(dice==1):
                if(target>=1 and target<=f):
                    return 1
                else:
                    return 0
            rolls=0
            for i in range(1,f+1):
                rolls+=dp(dice-1,target-i)%1000000007
            memo[(dice,target)]=rolls%1000000007
            return memo[(dice,target)]%1000000007
        return dp(d,target)%1000000007
class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        dp1 = [0 for _ in range(target+1)]
        dp2 = [0 for _ in range(target+1)]
        dp1[0] = 1
        mod = 10 ** 9 + 7
        for i in range(1, d+1):
            for j in range(target+1):
                for k in range(1, min(j,f) + 1):
                    dp2[j] = (dp2[j] + dp1[j-k]) % mod
            dp1 = dp2
            dp2 = [0 for _ in range(target+1)]
        return dp1[target] % mod
class Solution:
    # Recursive memoized solution
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        memo = {}
        def num_rolls_util(level, target):
            if level * f < target or target < level:
                return 0
            if level == 0:
                return 1
            
            res = 0
            for i in range(max(0, target - f), target):
                if (level-1, i) in memo:
                    res += memo[(level-1, i)]
                else:
                    tmp = num_rolls_util(level - 1, i)
                    memo[(level-1, i)] = tmp
                    res += tmp

            return res % (10 ** 9 + 7)
        
        return num_rolls_util(d, target)
class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        if(d==1):
            return int(f>=target)
        dp = [[0]*target for _ in range(d)]
        f = min(f, target)
        for i in range(f):
            dp[0][i] = 1
        for i in range(1, d):
            for j in range(i, target):
                for face in range(1, f+1):
                    if(face > j):
                        break
                    dp[i][j] = (dp[i-1][j-face]+dp[i][j])%(1e9+7)
        return int(dp[-1][-1])
class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        dp = [ [ 0 for j in range(target + 1) ] for i in range(d) ]
        
        for i in range(f):
            if i+1 <= target:
                dp[0][i+1] = 1
        
        for i in range(1, d):
            for j in range(1, target + 1):
                k = 1
                while j - k > 0 and k <= f:
                    dp[i][j] += dp[i-1][j-k]
                    k += 1
        return dp[-1][-1] % (10**9 + 7)
class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        A = [[0 for _ in range(target+1)] for _ in range(d+1)]
        A[0][0] = 1
        mod = 10 ** 9 + 7
        for i in range(1, d+1):
            for j in range(1, target+1):
                for k in range(1, min(j,f)+1):
                    A[i][j] = (A[i][j] + A[i-1][j-k]) % mod
        return A[-1][-1]
class Solution:    
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        dpt = [[1 if (i == 0 and 0<j and j <= f) else 0 for j in range(target+1)] for i in range(d)]
        for i in range(1,d):
            for j in range(1,target+1):
                for k in range(1,f+1):
                    if(j-k >= 0):
                        dpt[i][j] += dpt[i-1][j-k] % (10**9+7)
        return dpt[-1][target] % (10**9+7)
        

class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        dp = [[0 for j in range(d + 1)] for i in range(target + 1)]
        dp[0][0] = 1    
        for i in range(1, target + 1):
            for j in range(1, d + 1):
                for k in range(1, f + 1):
                    if target - k >= 0:
                        dp[i][j] += dp[i - k][j - 1]
                        
                    
        
        return dp[target][d] % (10**9 + 7)      
                    

class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        self.sols = {}
        self.sols[(d, f, target)] = self.iter(d, f, target)
        return self.sols[(d, f, target)] % (10**9 + 7)
    def iter(self, d, f, target):
        if d == 0:
            return 1 if target == 0 else 0
        if (d, f, target) in self.sols: return self.sols[(d, f, target)]
        self.sols[(d, f, target)] =  sum([self.iter(d-1, f, target-i) for i in range(1,f+1)])
        return self.sols[(d, f, target)]
class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:        
        # dp
        dp = [[0 for i in range(target + 1)] for j in range(d + 1)]
        dp[0][0] =  1
        mod =  10**9 + 7
        for i in range(1, d+1):
            for j in range(1, target + 1):
                for k in range(1, min(f, j) + 1):
                    dp[i][j] = (dp[i][j] + dp[i - 1][j - k]) % mod
        return dp[d][target] % mod
        
        
        

class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        MOD=(10**9)+7
        mem=[]
        for di in range(d+1):
            cr=[0 for i in range(target+1)]
            #cr=[1 if (i>0 and di==1 and i<=f) else 0 for i in range(target+1)]
            #cr=[0 if (i>f or di!=1 or i<1) else 1 for i in range(target+1)]
            mem.append(cr)
        mem[0][0]=1
        print(mem)
        for ti in range(1,target+1):
            for di in range(1,d+1):
                for fi in range(1,f+1):
                    if(ti-fi>=0):
                        mem[di][ti]+=mem[di-1][ti-fi];
        
        #print(mem)
        return mem[d][target]%MOD
class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        def numRollsHelper(level, target):
            if f * level < target or target < level:
                return 0
            
            if level == 0:
                return 1
            
            res = 0
            for i in range(max(target - f, 0), target):
                if (level - 1, i) in memo:
                    res += memo[(level - 1, i)]
                else:
                    tmp = numRollsHelper(level - 1, i)
                    memo[(level - 1, i)] = tmp % (10 ** 9 + 7)
                    res += tmp % (10 ** 9 + 7)
            
            return res
        
        memo = {}
        return numRollsHelper(d, target) % (10 ** 9 + 7)
class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        M = 10**9+7
        dp = [[0 for j in range(max(target, f)+1)] for i in range(d+1)]
        for j in range(1, f+1):
            dp[1][j] = 1
            
        for i in range(2, d+1):
            for j in range(i, max(target, f)+1):
                # print("i", i, "j", j)
                for k in range(1, f+1):
                    if j>=k:
                        dp[i][j] += dp[i-1][j-k]
        
            
        # for i in range(d+1):
        #     print(dp[i])
            
        return dp[-1][target] % M
    
# 1
# 6
# 3
# 2
# 6
# 7
# 2
# 5
# 10
# 30
# 30
# 500

class Solution:
    
    def __init__(self):
        self.cache = dict()
    
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        
        def helper (d, r):
            
            if d < 0 or r < 0:
                return 0
            
            if d == 0 and r == 0:
                return 1
                
            elif d != 0 and r != 0:
                
                if (d, r) not in self.cache:
                
                    self.cache[(d, r)] = 0
                
                    for i in range(1, f+1):
                        self.cache[(d, r)] += helper(d - 1, r - i)
                        
                return self.cache[(d, r)]
            
            else:
                return 0
            
        
        return helper(d, target) % (10**9 + 7)

# factorials = [1]

# def choose(n,r):
#     nonlocal factorials
#     return factorials[n]/(factorials[n-r]*factorials[r])

# def counter(d,f,target,facesUsed, d_og, t_og):
#     if(d==0 and target==0):
#         return factorials[d_og]
#     if(d<0 or target<0 or f<1):
#         return 0
#     # print(d,f,target,facesUsed, min(target//f + 1,d))
#     c = 0
#     for i in range(min(target//f,d)+1):
#         a = counter(d-i,f-1,target-f*i,facesUsed + 0 if i==0 else 1,d_og,t_og) / factorials[i]
#         c += a
#     return c
        
class Solution:    
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        # nonlocal factorials
        # for i in range(1,d+1):
        #     factorials.append(i*factorials[i-1])
        # return int(counter(d,f,target,0,d,target))
        dpt = [[1 if (i == 0 and 0<j and j <= f) else 0 for j in range(target+1)] for i in range(d)]
        # for i in range(d):
            # for j in range(min(,target):
        for i in range(1,d):
            for j in range(1,target+1):
                for k in range(1,f+1):
                    if(j-k >= 0):
                        dpt[i][j] += dpt[i-1][j-k] % (10**9+7)
        # print(dpt)
        return dpt[-1][target] % (10**9+7)
        

class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        dp = [1]+[0]*target
        
        for i in range(1, d+1):
            dp2 = [0]*(target+1)
            for j in range(target, i-1, -1):
                for k in range(1, f+1):
                    if j-k<i-1: break
                    
                    dp2[j] += dp[j-k]
                dp2[j]%=(10**9+7)
            dp = dp2
        # print(dp)
        return dp[-1]
class Solution:
    def numRollsToTarget(self, dice: int, faces: int, target: int) -> int:
        
        table = [[0 for _ in range(target + 1)] for _ in range(dice + 1)]
        table[0][0] = 1
        mod = 10 ** 9 + 7
        
        for d in range(1, dice + 1):
            for i in range(1, target + 1):
                for f in range(1, min(i + 1, faces + 1)):
                    table[d][i] = (table[d][i] + table[d-1][i-f]) % mod
                
        return table[-1][-1]
class Solution:
    def numRollsToTarget(self, N: int, faces: int, total: int) -> int:
        DP = {}
        DP[0] = [0] * (total+1)
        DP[1] = [0] + [1]*(faces) + [0] *(total-faces)

        for die in range(2,N+1):
            DP[die] = [0] * (total+1)
            for i in range(1,total+1):#the subtotal
                count  = 0
                for j in range(1,faces+1):#this die's contribution
                    if (i-j) >= 1 and DP[die-1][i-j] > 0:
                        if die==2:
                            print((i,j))
                        count+= DP[die-1][i-j]
                DP[die][i] = count
                print(count)
        ans = DP[N][total]
        return ans %(10**9 + 7)

class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        
        dp=[[0 for i in range(target+1)] for _ in range(d+1)]
        dp[0][0]=1
        for i in range(1,d+1):
            for j in range(1,target+1):
                for k in range(1,f+1):
                    if j-k>=0:
                        dp[i][j]+=dp[i-1][j-k]
                        
        return dp[-1][-1]%(10**9+7)

class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        dp = [[0 for _ in range(d)] for _ in range(target+1)]
        
        for i in range(1, target+1):
            dp[i][0] = 1 if i <= f else 0

        for di in range(1, d):
            for t in range(1, target+1):
                sub_targets = [t-face if t >= face else 0 for face in range(1, f+1)]
                dp[t][di] = sum([dp[sub][di-1] for sub in sub_targets])
        
        return dp[-1][-1] % (10 ** 9 + 7)
            
            

class Solution:
    # Recursive memoized solution
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        memo = {}
        def num_rolls_util(level, target):
            if level * f < target or target < level:
                return 0
            if level == 0:
                return 1
            
            res = 0
            for i in range(max(0, target - f), target):
                if (level-1, i) in memo:
                    res += memo[(level-1, i)]
                else:
                    tmp = num_rolls_util(level - 1, i)
                    memo[(level-1, i)] = tmp
                    res += tmp

            return res % (10 ** 9 + 7)
        
        return num_rolls_util(d, target)
    
    # O(d * f * target) iterative
#     def numRollsToTarget(self, d: int, f: int, target: int) -> int:
#         mod = 10 ** 9 + 7
#         dp = [[0] * (target + 1) for _ in range(d+1)]
#         dp[0][0] = 1
        
#         for i in range(1, d+1):
#             for j in range(1, min(target, i * f) + 1):
#                 for k in range(1, min(j, f) + 1):
#                     dp[i][j] += dp[i-1][j-k] % mod
        
#         return dp[d][target] % mod        
    
    # def numRollsToTarget(self, d: int, f: int, target: int) -> int:
    #     # row: target value, col: number of dices
    #     dp: List[List[int]] = [[0] * (d + 1) for _ in range(target + 1)]
    #     for i in range(1, f + 1):  # initialize first col
    #         if i <= target:
    #             dp[i][1] = 1
    #         else:
    #             break
    #     for j in range(2, d + 1):  # populate rest of dp matrix
    #         for i in range(j, target + 1):
    #             dp[i][j] = dp[i - 1][j - 1] + dp[i - 1][j] - dp[i - min(i - 1, f) - 1][j - 1]  # line *
    #     return dp[target][d] % (10**9 + 7)

class Solution:
    def numRollsToTarget(self, d, f, target):
        '''
        :type d: int
        :type f: int
        :type target: int
        :rtype: int
        '''
        dp = [[0 for _ in range(target+1)] for _ in range(d+1)]
        dp[0][0] = 1
        mod = 10**9+7
        for i in range(d+1):
            for t in range(1, target+1):
                for val in range(1, min(f, t)+1):
                    dp[i][t] = (dp[i][t] + dp[i-1][t-val]) % mod
        return dp[d][target]
class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        # key = (num_rolls, target); value = (num_combinations)
        memo = {}
        def num_rolls_recur(d, target):
            key = (d, target)
            if key in memo:
                return memo[key]
            if target < 0:
                return 0
            if d == 0:
                if target == 0:
                    return 1
                return 0
            num_ways = 0
            for i in range(1, f + 1):
                num_ways = (num_ways + num_rolls_recur(d - 1, target - i)) % (pow(10, 9) + 7)
            memo[key] = num_ways
            return num_ways
        return num_rolls_recur(d, target)
class Solution:
    def helper(self, d, f, target, dp):
        if d == 0:
            return 1 if target == 0 else 0
        if (target, d) in dp:
            # use cache | memoization
            return dp[(target, d)]
        for i in range(1, f + 1):
            # we are branching dp into all of the sub cases
            # sub cases are additive
            dp[(target, d)] += self.helper(d - 1, f, target - i, dp)
        return dp[(target, d)]
        
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        dp = Counter()
        return self.helper(d, f, target, dp) % (pow(10, 9) + 7)

class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        dp=[[0 for _ in range(target+1)] for _ in range(d+1)]
        for i in range(1,min(f+1, target+1)):
            dp[1][i]=1
        for i in range(1, d+1):
            for j in range(2, target+1):
                for k in range(1,f+1):
                    if j-k>=0:
                        dp[i][j]+=dp[i-1][j-k]
        #print(dp)
        return dp[-1][-1]%1000000007
            

class Solution:
    # Top down solution for dice sim
    def helper(self, d, f, target, dp):
        if d == 0:
            return 1 if target == 0 else 0
        if (target, d) in dp:
            # use cache | memoization
            return dp[(target, d)]
        for i in range(1, f + 1):
            # we are branching dp into all of the sub cases
            # sub cases are additive
            dp[(target, d)] += self.helper(d - 1, f, target - i, dp)
        return dp[(target, d)]
        
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        dp = Counter()
        return self.helper(d, f, target, dp) % (pow(10, 9) + 7)

class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        if target == d or target == d*f:
            return 1
        T = [0 for i in range(target+1)]
        m = 10**9 + 7
        T[0]=1
        for i in range(0,d):
            for j in range(target,-1,-1):
                T[j] = 0
                for k in range(1,f+1):
                    if j>=k:
                        T[j]+= T[j-k] % m
                        T[j]%=m
        
        return T[target]

class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        mod = 10**9 + 7
        dp = [[0]*d for i in range(target + 1)]
        
        for i in range(1, target + 1):
            dp[i][0] = int(i <= f)
                
                   
        
        for dd in range(1, d):
            for ff in range(1, f+1):
                for i in range(target + 1):
                    if(i+ff <= target):
                        dp[i+ff][dd] = ( dp[i+ff][dd] + dp[i][dd-1] ) % mod
                        
            
        return dp[-1][-1]
MOD = 1000000007

class Solution:
    dyn = {}
    def _get_res(self, d,f,target):
        if d == 1:
            return 1 if 0 < target and target <= f else 0
        if (d, f, target) in self.dyn:
            return self.dyn[(d, f, target)]
        res = 0
        for i in range(1, f + 1):
            res += self._get_res(d - 1, f, target - i)

        self.dyn[(d, f, target)] = res 
        
        return res
    
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        return self._get_res(d, f, target) % MOD
class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        MOD = (10 ** 9) + 7
        # print(MOD)
        
        f = min(f, target)
        
        dp = [[0 for t in range(target + 1)] for r in range(d)]
        
        # initialize for first roll (using just one die)
        for i in range(1, f+1):
            dp[0][i] = 1
        
        
        # with dp, number of states can be 1 - 1000
        
        # how many ways can I get from a to b in less than 7 rolls?
        # how many ways can I get from 0 to target in less than d rolls?
        
        # with transition cost as # of dice: 
        
        # fill table
        # simulate number of ways using 2 dice, then 3, etc.
        for r in range(1, d):
            # for each possible running sum
            for i in range(1, target + 1):
                # check if this sum is reachable by some roll of the new die
                for val in range(1, f+1):
                    if i-val >= 0:
                        # add the number of ways to get to the current sum from the
                        # previous sum, use modulus at this point to reduce computation costs
                        dp[r][i] = (dp[r-1][i-val] + dp[r][i]) % MOD

        # return the number of ways to reach target, using ALL dice
        return dp[-1][target]
        
        

class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        maxAmount = 10**9 + 7
        
        '''
        dp[d][v] = dp[d-1][v-1] + dp[d-1][v-2] + ... + dp[d-1][v-f]
        '''
        if target < 1 or d * f < target:
            return 0
        
        dpPrev = [(i, 1) for i in range(1, f+1)]
        for roundNo in range(2, d+1):
            numWays = defaultdict(int)
            for i, v in dpPrev:
                for num in range(1, f+1):
                    numWays[i + num] += v
            dpPrev = []
            for k in numWays:
                dpPrev.append((k, numWays[k]))
        
        for pair in dpPrev:
            if pair[0] == target:
                return pair[1] % maxAmount
        
        return 0                    

class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        mod = 10**9 + 7
        dp = [[0] * (target + 1) for _ in range(d+1)]
        dp[0][0] = 1
        for i in range(1, d+1):
            for j in range(i, target+1):
                
                #dp[i][j] = dp[i-1][j]
                for k in range(1, min(j, f)+1):
                    if 0 <= j - k:
                        dp[i][j] += dp[i-1][j-k]
                dp[i][j] %= mod
                        
        return dp[d][target] % mod
        

class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        
        MOD = (10 ** 9) + 7
        # ways to roll target with d dice:
        # roll target - 1, -2, -3, .. , -f with d - 1 dice
        f = min(target, f)
        dp = [[0 for val in range(target+1)] for die in range(d)]
        
        for i in range(1, f+1):
            dp[0][i] = 1
            
        for i in range(1, d):
            for val in range(target+1):
                for face in range(1, f+1):
                    if val - face >= 0:
                        dp[i][val] = (dp[i-1][val-face] + dp[i][val]) % MOD
        return dp[-1][target]
class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        mod = 10**9+7
        dp = [[0 for i in range(target+1)] for j in range(d+1)]
        
        dp[1] = [0] + [1 for i in range(f)] + [0 for i in range(target-f)]
        
        for i in range(1, d+1):
            if i==1 or i==0:
                continue
            for j in range(1, target+1):
                for k in range(1, f+1):
                    if j-k > 0:
                        dp[i][j] = (dp[i][j] + dp[i-1][j-k]) % mod
                        #dp[i][j] = (dp[i][j] + dp[i-1][j-k])
        
        #print(dp)
        return dp[-1][-1]
class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        dp = [0] * (target+1)
        dp[0] = 1
        for i in range(1, d+1):
            temp = [0]
            # iterate each tot from 1 to target
            for j in range(1, target+1):
                # k is each face 
                x = sum(dp[j-k] if k <= min(j, f) else 0 for k in range(1, f+1))
                temp.append(x)
            # print(temp)
            dp = temp

        return dp[target] % (10**9 + 7)
class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        dp = [[0 for _ in range(target+1)] for _ in range(d+1)]
        
        for dCand in range(1,d+1):
            for targetCand in range(1,target+1):
                if dCand == 1:
                    dp[dCand][targetCand] = 1 if targetCand <= f else 0
                    continue
                allCombos = 0
                for option in range(1,f+1):
                    allCombos += dp[dCand-1][max(0,targetCand-option)]
                dp[dCand][targetCand] = allCombos % (10 ** 9 + 7)
        return dp[d][target]
                
        
        
        '''
        dp[d][t]:
            for each side:
                smallerAns = dp[d-1][target-side]
            dp[d][f] = sum of smaller answers
        
        base case:
            dp[1][t] = 1 if t <= f, 0 otherwise
            dp[d][<=0] = 0
        '''
        

class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        modu = 10**9+7
        dp = [[0]*d for i in range(target + 1)]
        for i in range(1, target + 1):
            dp[i][0] = int(i <= f)
            
            
        for dd in range(1, d):
            for ff in range(1, f+1):
                for i in range(target + 1):
                    if(i+ff <= target):
                        dp[i+ff][dd] = ( dp[i+ff][dd] + dp[i][dd-1]) % modu
                        
        return dp[-1][-1]
class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        dp = [[0 for j in range(target + 1)] for i in range(d + 1)]
        dp[0][0] = 1
        for i in range(1, d + 1):
            for j in range(1, target + 1):
                for val in range(1, f + 1):
                    if j - val >= 0:
                        dp[i][j] = (dp[i][j] + dp[i - 1][j - val]) % (10 ** 9 + 7)
        return dp[-1][-1] % (10 ** 9 + 7)
        
        

class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        
        # target < d -> impossible 
        
        # target > f*d -> impossible 
        def auxNumRolls(d, f, target, memoization):
            if target < d or target > d*f:
                return 0

            # one die condition: only one possibility targe == die roll
            if d == 1:
                return 1

            ans = 0
            for i in range(1, f+1):
                if (d-1, target-i) not in memoization:
                    memoization[(d-1, target-i)] = auxNumRolls(d-1, f, target-i, memoization)
                ans += memoization[(d-1, target-i)]
            return ans
        
        
        return auxNumRolls(d, f, target, {})%(10**9 + 7)
        

class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        mod = 10**9 + 7
        state = [ 1 ] + [ 0 ] * ( target )
        for i in range( d ) :
            for j in range( target , -1 , -1 ) :
                state[ j ] = 0
                for k in range( 1 , min( f , j ) + 1 ) :
                    state[ j ] += state[ j - k ]
                    state[ j ] %= mod
        return state[-1]
#Sanyam Rajpal

class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        
        dp = [[0] * (target+1) for _ in range(d)] # dp[i][j]: solution for target j with i + 1 dices
        BASE = 10**9 + 7
        for j in range(1, min(f, target)+1):
            dp[0][j] = 1
        for i in range(1, d):
            for j in range(target+1):
                for k in range(1, f+1):
                    if k >= j: break
                    dp[i][j] += dp[i-1][j-k] % BASE
        
        return dp[d-1][target] % BASE
            

# DP, either recursion or iteration works (recursion is faster than iteration, O(dt) vs. O(dft))
class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        memo = [[0] * (target + 1) for _ in range(d + 1)]
        for i in range(1, d+1):
            for j in range(1, target+1):
                if i == 1:
                    if 1 <=j <= f:
                        memo[i][j] = 1
                    continue
                for t in range(1, f+1):
                    if j >= t:
                        memo[i][j] += memo[i-1][j-t]
        return memo[d][target] % (10 ** 9 + 7)
class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        # NRT(d, f, t) = (sum over f' = {1, 2, ..., f}) numRollsToTarget(d - 1, f, t - f')
        
        dp = [] # dice x target
        mod = math.pow(10, 9) + 7
        
        for _ in range(d + 1):
            dp.append((target + 1) * [0])
            
        for i in range(target + 1):
            dp[0][i] = 0
            
        for i in range(d + 1):
            dp[i][0] = 0
            
        for i in range(min(f, target)):
            dp[1][i + 1] = 1
        
        for i in range(2, d + 1):
            for j in range(1, target + 1):
                for roll in range(1, f + 1):
                    if (j - roll >= 0):
                        dp[i][j] = ((dp[i][j] % mod) + (dp[i - 1][j - roll] % mod)) % mod
                        
        return int(dp[d][target])
class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        a = [[0] * (d+1) for _ in range(target+1)]
        a[0][0] = 1
        for i in range(1, target+1):
            for j in range(1, d+1):
                for k in range(1, f+1):
                    if i >= k and j >= 1:
                        a[i][j] =  (a[i][j] + a[i-k][j-1]) % (10**9+7)
        return a[target][d]
class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        mod = int(1e9) + 7
        
        dp = [[0 for _ in range(d + 1)] for _ in range(target + 1)]
        for i in range(1, min(target + 1, f + 1)):
            dp[i][1] = 1
            
        for i in range(2, target + 1):
            for j in range(2, d + 1):
                for k in range(1, min(f + 1, i + 1)):
                    dp[i][j] += dp[i - k][j - 1]
            
        return dp[target][d] % mod
class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        if d == 0:
            return 0
        
        dp = [[0 for i in range(target + 1)] for i in range(d + 1)]
        
        for i in range(1, min(f + 1, target + 1)):
            dp[1][i] = 1
        
        for i in range(2, d + 1):
            for j in range(1, target + 1):
                for k in range(1, f + 1):
                    if j - k >= 0:
                        dp[i][j] += dp[i - 1][j - k]
                        
                        if dp[i][j] >= 1000000007:
                            dp[i][j] -= 1000000007
        
        return dp[d][target]
class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        
        if d*f < target or d > target:
            return 0
        
        prev = {0:1}
        
        for x in range(d):
            curr = {}
            for y in range(1,f+1):
                
                for prev_sum,count in list(prev.items()):
                    curr_sum = prev_sum+y
                    if curr_sum <= target:
                        curr[curr_sum] = curr.get(curr_sum,0)+count
            prev = curr
            # print(prev)
        return prev[target]%(10**9+7) if target in prev else 0

class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        DP = [[0]*(target+1) for _ in range(d+1)]
        
        for i in range(f+1):
            if i <= target:
                DP[1][i] = 1
        if d == 1 :
            return DP[d][target]
        
        for j in range(2,d+1):
            for i in range(1,target+1):
                for val in range(1,f+1):
                    if 0< i-val <len(DP[j]):
                        DP[j][i] += DP[j-1][i-val]
        # print(DP)
        return DP[d][target] % (10**9 + 7)
class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        mod = 10**9 + 7
        # dp[i][j] use i dices to get target
        dp = [[0]* (target+1) for _ in range(d+1)]
        dp[0][0] = 1
        for i in range(1, d+1):
            for j in range(target+1):
                for k in range(1, f+1):
                    if j >= k:
                        dp[i][j] += dp[i-1][j-k]
                        dp[i][j] %= mod
        return dp[d][target]
class Solution:
    def numRollsToTarget(self, d: int, f: int, n: int) -> int:
        mod = 10**9 + 7
        dp = [0 for i in range(n+1)]
        dp[0] = 1
        temp = [0 for i in range(n+1)]
        lol = [0 for i in range(n+1)]
        for i in range(1,d+1):

            for j in range(i,n+1):
                for k in range(1,min(f+1,j+1)):
                    temp[j]+=dp[j-k]
                    temp[j] = temp[j]%mod
            dp = temp.copy()
            temp = lol.copy()
        # print(dp)
        return dp[-1]
class Solution:

            
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        mod = 1000000000+7
        dp =[[0 for i in range(target+1)] for j in range(d)]
        for i in range(d):
            for j in range(target+1):
                if i == 0:
                    dp[i][j] = 1 if j>=1 and j<=f else 0
                else:
                    for l in range(1,f+1):
                        if j-l>0:
                            dp[i][j]+=dp[i-1][j-l]
                            dp[i][j]%=mod
        return dp [d-1][target] % mod

class Solution:
    '''
    DP formula:
    calc(d, target) = calc(d-1, target - 1) + calc(d-1, target - 2) + ... + calc(d-1, target - f)
    example:
    d = 2, target = 7, f = 6
    calc(1, 1) = 1  -> one way to get 1 out of one 6 faced dice
    calc(1, 2) = 1  -> one way to get 2 out of one 6 faced dice
    calc(1, 3) = 1  -> one way to get 3 out of one 6 faced dice
    ...
    calc(1, 6) = 1 -> one way to get 6 out of one 6 faced dice
    calc(1, 7) = 0 -> NO way to get 7 from one 6 faced dice
    similarly, no way to get value > f using just one dice

    calc(2, 1) = 0 -> 2 dice, target 1. impossible to get 1 using 2 six faced dice.
    calc(2, 2) = 1 -> calc(1, 1) = 1. ways getting 2 out of 2 dice is similar to getting 1 out of 1 dice
    calc(2, 3) = calc(1, 2) + calc(1, 1) = 1 + 1 = 2
                 ways of getting 3 out of 2 dice meaning, ways of getting 2 using one dice [calc(1, 2)] 
                 and getting 1 from other dice [calc(1, 1)]
    calc(2, 6) = calc(1, 1) + calc(1, 2) + calc(1, 3) + calc(1, 4) + calc(1, 5) + calc(1, 6)
                ways to get 1 out of 1 dice then getting 5 from second dice +
                ways to get 2 from first dice and then getting 4 from second dice +
                ways to get 3 from first dice and getting 3 from second dice +
                ways to get 4 from first dice and getting 2 from second dice +
                ways to get 5 from first dice and getting 1 from second dice
    '''
    def numRollsToTarget(self, d, f, target):
        mod_v = 10 ** 9 + 7 
        dp = [[0] * (target + 1) for _ in range(d+1)]

        for i in range(1, f+1):
            if i > target:
                break
            #print(i)
            dp[1][i] = 1
        #print(dp)
        for i in range(2, d+1):
            for j in range(i, target+1):
                for k in range(1, f+1):
                    if j - k >= 0:
                        dp[i][j] += dp[i-1][j-k]
                        dp[i][j] %= mod_v
                    else:
                        break
        #print(dp)
        return dp[d][target]

class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        dp = [[None for _ in range(target+1)] for _ in range(d+1)]
        
        for dCand in range(d+1):
            for targetCand in range(target+1):
                if targetCand == 0 or dCand == 0:
                    dp[dCand][targetCand] = 0
                    continue
                if dCand == 1:
                    dp[dCand][targetCand] = 1 if targetCand <= f else 0
                    continue
                allCombos = 0
                for option in range(1,f+1):
                    allCombos += dp[dCand-1][max(0,targetCand-option)]
                    allCombos %= (10 ** 9 + 7)
                dp[dCand][targetCand] = allCombos
        return dp[d][target]
                
        
        
        '''
        dp[d][t]:
            for each face:
                smallerAns = dp[d-1][target-face]
            dp[d][f] = sum of smaller answers
        
        base case:
            dp[1][t] = 1 if t <= f, 0 otherwise
            dp[d][<=0] = 0
        '''
        

class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        opt = [[0 for i in range(target+1)] for x in range(d)]
        
        for i in range(1,target+1):
            if (i <= f):
                opt[0][i] = 1
                
        for i in range(1,d):
            for j in range(1, target+1):
                for h in range(1,f+1):
                    if (j-h > -1):
                        opt[i][j] += opt[i-1][j-h]
        
        return opt[d-1][target] % (10**9 + 7)

class Solution(object):
   def numRollsToTarget(self, d, f, t):
      mod = 1000000000+7
      dp =[[0 for i in range(t+1)] for j in range(d)]
      for i in range(d):
         for j in range(t+1):
            if i == 0:
               dp[i][j] = 1 if j>=1 and j<=f else 0
            else:
               for l in range(1,f+1):
                  if j-l>0:
                     dp[i][j]+=dp[i-1][j-l]
                     dp[i][j]%=mod
      return dp [d-1][t] % mod

import itertools

class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
      mod = 1000000000+7
      dp =[[0 for i in range(target+1)] for j in range(d)]
      for i in range(d):
         for j in range(target+1):
            if i == 0:
               dp[i][j] = 1 if j>=1 and j<=f else 0
            else:
               for l in range(1,f+1):
                  if j-l>0:
                     dp[i][j]+=dp[i-1][j-l]
                     dp[i][j]%=mod
      return dp [d-1][target] % mod
class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        if d * f < target or d > target:
            return 0
        
        if d == 1:
            if target <=f:
                return 1
            else:
                return 0
        
        M = 10**9 + 7
        
        dp = [[0 for _ in range(target+1)] for _ in range(d+1)]
        
        dp[0][0] = 1
        
        for i in range(1, d + 1):
            for j in range(1, target + 1):
                
                if j == 1:
                    dp[i][j] = dp[i - 1][0] 
                elif j <= f:
                    dp[i][j] = dp[i][j - 1] + dp[i - 1][j - 1]                
                else:
                    dp[i][j] = (dp[i][j - 1] + dp[i - 1][j - 1] - dp[i - 1][j -f - 1])
                    
        return dp[d][target] % M
class Solution:
    # Recursive memoized solution
#     def numRollsToTarget(self, d: int, f: int, target: int) -> int:
#         memo = {}
#         def num_rolls_util(level, target):
#             if level * f < target or target < level:
#                 return 0
#             if level == 0:
#                 return 1
            
#             res = 0
#             for i in range(max(0, target - f), target):
#                 if (level-1, i) in memo:
#                     res += memo[(level-1, i)]
#                 else:
#                     tmp = num_rolls_util(level - 1, i)
#                     memo[(level-1, i)] = tmp
#                     res += tmp

#             return res % (10 ** 9 + 7)
        
#         return num_rolls_util(d, target)
    
    # O(d * f * target) iterative
#     def numRollsToTarget(self, d: int, f: int, target: int) -> int:
#         mod = 10 ** 9 + 7
#         dp = [[0] * (target + 1) for _ in range(d+1)]
#         dp[0][0] = 1
        
#         for i in range(1, d+1):
#             for j in range(1, min(target, i * f) + 1):
#                 for k in range(1, min(j, f) + 1):
#                     dp[i][j] += dp[i-1][j-k] % mod
        
#         return dp[d][target] % mod        
    
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        # row: target value, col: number of dices
        dp: List[List[int]] = [[0] * (d + 1) for _ in range(target + 1)]
        for i in range(1, f + 1):  # initialize first col
            if i <= target:
                dp[i][1] = 1
            else:
                break
        for j in range(2, d + 1):  # populate rest of dp matrix
            for i in range(j, target + 1):
                dp[i][j] = dp[i - 1][j - 1] + dp[i - 1][j] - dp[i - min(i - 1, f) - 1][j - 1]  # line *
        return dp[target][d] % (10**9 + 7)
class Solution:
    def numRollsToTarget(self, d: int, f: int, t: int) -> int:
      mod = 1000000000+7
      dp =[[0 for i in range(t+1)] for j in range(d)]
      for i in range(d):
         for j in range(t+1):
            if i == 0:
               dp[i][j] = 1 if j>=1 and j<=f else 0
            else:
               for l in range(1,f+1):
                  if j-l>0:
                     dp[i][j]+=dp[i-1][j-l]
                     dp[i][j]%=mod
      return dp [d-1][t] % mod
class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
           
        mod = int(10**9 + 7)
        dp = [[0] * (target+1) for _ in range(d+1)] 
        for j in range(1, min(f+1, target+1)): dp[1][j] = 1
        for i in range(2, d+1):
            for j in range(1, target+1):
                for k in range(1, f+1):
                    if j - k >= 0: dp[i][j] += dp[i-1][j-k]
                dp[i][j] %= mod        
        return dp[-1][-1] 

class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        '''
        d*f = total possible roll outcomes
        
        
        d=2 f=6 target=7
        
        1+6 2+5 3+4 4+3 5+2 6+1
        
            0 1 2 3 4 5 6 7
            0 1 1 1 1 1 1 0
            0 0 1 2 3 4 5 6
        dp[d][i] += dp[d-1][i-j] when j = 1...6 and i >= j
        
        '''
        
        dp = [[ 0 for _ in range(target+1)] for _ in range(d)]
        
        # initialize the first row with up to f
        for i in range(1, min(f+1, target+1)): dp[0][i] = 1
        
        for i in range(1, d):
            for j in range(1, target+1):
                for k in range(1, f+1):
                    if j-k >= 0:
                        dp[i][j] += dp[i-1][j-k]
                        dp[i][j] %= (10**9)+7
                    
        # print(dp)
        return dp[d-1][target]

class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        dp = [[0 for _ in range(target+1)] for _ in range(d+1)]
        
        for dCand in range(1,d+1):
            for targetCand in range(1,target+1):
                if dCand == 1:
                    dp[dCand][targetCand] = 1 if targetCand <= f else 0
                    continue
                allCombos = 0
                for option in range(1,f+1):
                    allCombos += dp[dCand-1][max(0,targetCand-option)]
                    allCombos %= (10 ** 9 + 7)
                dp[dCand][targetCand] = allCombos
        return dp[d][target]
                
        
        
        '''
        dp[d][t]:
            for each side:
                smallerAns = dp[d-1][target-side]
            dp[d][f] = sum of smaller answers
        
        base case:
            dp[1][t] = 1 if t <= f, 0 otherwise
            dp[d][<=0] = 0
        '''
        

class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        dp = [[0 for i in range(target+1)] for j in range(d+1)]
        for i in range(1,d+1):
            for j in range(1,target+1):
                if i == 1:
                    for k in range(1,min(f+1, target+1)):
                        dp[i][k] = 1
                else:
                    num_permutations = 0
                    for k in range(1,f+1):
                        if j-k >= 0:
                            num_permutations += dp[i-1][j-k]
                    dp[i][j] = num_permutations
        return int(dp[d][target] % (10**9+7))
class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        dp = [[None for _ in range(d+1)] for _ in range(target+2)]
        for i in range(target+2):
            dp[i][d] = 0
        dp[target][d] = 1
        for i in range(d+1):
            dp[target+1][i] = 0
        for i in range(target, -1, -1):
            for j in range(d-1, -1, -1):
                s = 0
                for k in range(1, f+1):
                    s+= dp[min(i+k, target+1)][j+1] % 1000000007
                dp[i][j] = s
        return dp[0][0] % 1000000007

class Solution(object):
   def numRollsToTarget(self, d, f, t):
      mod = 1000000000+7
      dp =[[0 for i in range(t+1)] for j in range(d)]
      for i in range(d):
         for j in range(t+1):
            if i == 0:
               dp[i][j] = 1 if j>=1 and j<=f else 0
            else:
               for l in range(1,f+1):
                  if j-l>0:
                     dp[i][j]+=dp[i-1][j-l]
                     dp[i][j]%=mod
      return dp [d-1][t] % mod
ob = Solution()
print(ob.numRollsToTarget(2,6,7))
class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        MOD = 1000000007
        dp = [[0] * 1001 for _ in range(31)]
        mint = min(f, target)
        targetMax = d*f
        for i in range(1, mint+1):
            dp[1][i] = 1
        
        for i in range(2,d+1):
            for j in range(i, targetMax+1):
                for k in range(1, min(j, f)+1):
                    dp[i][j] = (dp[i][j] + dp[i-1][j-k]) % MOD
        return dp[d][target]


class Solution:
    def numRollsToTarget(self, d: int, f: int, s: int) -> int:
        mem = [[0 for _ in range(s + 1)] for _ in range(d + 1)]
        mem[0][0] = 1
        
        for i in range(1,d + 1):
            for j in range(1,s + 1):
                mem[i][j] = mem[i][j - 1] + mem[i - 1][j - 1]
                if j - f - 1 >= 0:
                    mem[i][j] -= mem[i - 1][j - f - 1]
                    
        return mem[-1][-1] % (10 ** 9 + 7)

class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        m = pow(10,9)+7
        T = [[0 for i in range(max(target+1, f+1))] for i in range(d)]
        for i in range(f):
            T[0][i+1] = 1
        for i in range(1,d):
            for j in range(1,target+1):
                for k in range(1,f+1):
                    if j-k>0:
                        T[i][j] += T[i-1][j-k]
                        T[i][j] = T[i][j]%m
            #print(T)
        return T[d-1][target]
class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        mod=10**9 + 7
        dp=[[0 for i in range(target+1)] for j in range (d)]
        for i in range(d):
            for j in range(target+1):
                if i == 0:
                    dp[i][j] = 1 if j>=1 and j<=f else 0
                else:
                    for l in range(1,f+1):
                        if j-l>0:
                            dp[i][j]+=dp[i-1][j-l]
                            dp[i][j]%=mod
        print(dp)
        return dp[-1][-1]
class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        dp = [[0 for _ in range(target + 1)] for _ in range(d + 1)]
        dp[0][0] = 1
        mod = 10 ** 9 + 7
        for i in range(1, d + 1):
            for j in range(1, target + 1):
                ff = 1
                highest_die = min(j, f)
                while ff <= highest_die:
                    dp[i][j] += dp[i-1][j-ff]
                    dp[i][j] %= mod
                    ff += 1
                    
        return dp[d][target] % mod

class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        if target == d or target == d*f:
            return 1
        T = [[0 for j in range(target+1)] for i in range(d+1)]
        m = 10**9 + 7

        T[0][0] = 1
        for i in range(1,d+1):
            T[i][0] = 0 
            for j in range(1,target+1,1):
                for k in range(1,f+1):
                    if j>=k:
                        T[i][j]+= T[i-1][j-k] % m
                        T[i][j]%=m

        return T[d][target] % m

class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        t = target
        mod = 1000000000+7
        dp =[[0 for i in range(t+1)] for j in range(d)]
        for i in range(d):
            for j in range(t+1):
                if i == 0:
                    dp[i][j] = 1 if j>=1 and j<=f else 0
                else:
                    for l in range(1,f+1):
                        if j-l>0:
                            dp[i][j]+=dp[i-1][j-l]
                            dp[i][j]%=mod
        return dp [d-1][t] % mod
class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        
        dp = [[0]+[0]*target for _ in range(d)]
        
        for i in range(d):
            for j in range(1,target+1):
                for k in range(1,f+1):
                    if k>j: break
                    if i==0: dp[i][j] = int(j==k)
                    else: dp[i][j] += dp[i-1][j-k]
                    if dp[i][j]>10**9: dp[i][j] = (dp[i][j]%(10**9))-(7*(dp[i][j]//(10**9)))
        
        return dp[-1][-1]
class Solution:
    
    @lru_cache(maxsize=None)
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        
        dp = [[0]+[0]*target for _ in range(d)]
        for i in range(d):
            for j in range(1,target+1):
                for k in range(1,f+1):
                    if k>j: break
                    if i==0: dp[i][j] = int(j==k)
                    else: dp[i][j] += dp[i-1][j-k]
                    if dp[i][j]>10**9: dp[i][j] = (dp[i][j]%(10**9))-(7*(dp[i][j]//(10**9)))
        
        return dp[-1][-1]
class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        mod=int(10**9+7)
        dp=[[0 for i in range(d+1)] for j in range(target+1)]
        dp[0][0]=1
        for i in range(1,target+1):
            for j in range(1,d+1):
                for k in range(1,f+1):
                    if i>=k:
                        dp[i][j]+=(dp[i-k][j-1]%mod)
                    dp[i][j]=dp[i][j]%mod
        return dp[-1][-1]%mod
class Solution:
    def numRollsToTarget(self, d, f, target):
        
        if d == 0:
            return 0
        
        dp = [[0 for _ in range(target+1)] for _ in range(d+1)]
        
        for i in range(1, target+1):
            dp[0][i] = 0
        for j in range(1, d+1):
            dp[j][0] = 0
            
        dp[0][0] = 1
        
        for i in range(1, d+1):
            for j in range(1, target+1):
                for k in range(1, f+1):
                    if j-k >= 0:
                        dp[i][j] += dp[i-1][j-k]
                        dp[i][j] %= ((10 ** 9) + 7)
                        
        return dp[d][target]
                
                
#         def recursive(d, f, target):
            
#             if target == 0 and d == 0:
#                 return 1
            
#             if target < 0 or d == 0:
#                 return 0
            
#             if dp[d][target] != -1:
#                 return dp[d][target]
#             temp = 0
#             for i in range(1, f+1):
#                 temp += recursive(d - 1, f, target - i)
                
#             dp[d][target] =  temp
#             return dp[d][target]

#         return (recursive(d, f, target) % ((10 ** 9) + 7))

from functools import lru_cache
class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        
        if d*f < target:
            return 0
        
        @lru_cache(maxsize = None)
        def dfs(index, target):
            if index == d:
                if target == 0:
                    return 1
                return 0
            
            if target <= 0:
                return 0
            
            ret = [0]*(f + 1)
            for num in range(1,f+1):
                ret[num] = dfs(index + 1, target - num)
            
            return sum(ret) % (10**9 + 7)
        
        return dfs(0, target)

class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        m = collections.defaultdict(int)
        m[(0, 0)] = 0
        for t in range(1, 1+f):
            m[(1, t)] = 1
        for i in range(1,1+d):
            for j in range(1, 1+target):
                for k in range(1, min(1+f, j)):
                    m[(i,j)] += m[(i-1, j-k)]
        return m[(d,target)]%(10**9+7)
class Solution:
    def helper(self, d, f, target, num_ways):
        if d == 0:
            if target == 0:
                return 1
            else:
                return 0
        
        if (d, target) in num_ways:
            return num_ways[(d, target)]
        
        for face in range(1, f+1):
            num_ways[(d, target)] += self.helper(d-1, f, target - face, num_ways)
            
        return num_ways[(d, target)]
    
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        num_ways = collections.defaultdict(int)
        
        self.helper(d, f, target, num_ways)
        
        return num_ways[(d, target)] % (1000000000 + 7)

class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        dp = [[None for _ in range(d+1)] for _ in range(target+2)]
        for i in range(target+2):
            dp[i][d] = 0
        dp[target][d] = 1
        for i in range(d+1):
            dp[target+1][i] = 0
        for i in range(target, -1, -1):
            for j in range(d-1, -1, -1):
                s = 0
                for k in range(1, f+1):
                    s+= dp[min(i+k, target+1)][j+1] % 1000000007
                dp[i][j] = s
        return dp[0][0] % 1000000007
            
        
        
        
        def solve(x, i):
            if i == d:
                if x == target:
                    return 1
                return 0
            if x >= target:
                return 0
            s = 0
            for j in range(1, f+1):
                s+= solve(x+j, i+1) % 1000000007
            return s
        
        return solve(0, 0) % 1000000007

class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        # initialization
        f = min(target, f)
        MOD = 10**9 + 7
        dp = [[None for _ in range(target + 1)] for _ in range(d + 1)]
        for i in range(1, f + 1):
            dp[1][i] = 1
            
        # computing
        for i in range(2, d + 1):
            for j in range(1, target + 1):
                dp[i][j] = 0
                for k in range(1, f + 1):
                    if j - k >= 1 and dp[i - 1][j - k] is not None:
                        dp[i][j] += dp[i - 1][j - k]
                        
                dp[i][j] %= MOD
                
        return dp[d][target] if dp[d][target] is not None else 0
class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        
        if target > f*d:
            return 0
        
        @lru_cache(None)
        def dp(target, f, d):
            if target == 0 and d == 0:
                return 1
            
            if target <= 0:
                return 0
            
            count = 0
            
            for i in range(1, f+1):
                count += dp(target - i, f, d-1)
                count %= (10**9 + 7)
                
            return count
    
        return int(dp(target, f, d) % (10**9 + 7))
            

# # ############
# f = 3
# ans = [1,1,2],[1,2,1],[2,1,1]
# count = 3
# dp(3, 4)
# d = 3
# target = 4
# i = 2
# -------
# dp(2, 2)
# d = 2
# target = 2
# i = 1
# ------
# dp(1, 1)
# d = 1
# target = 1
# i = 1
# ------
# dp(0, 0)
# d = 0
# target = 0
# ------

class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        dp = [[0] * (target + 1) for _ in range(d + 1)]
        dp[0][0] = 1
        
        for i in range(1, d + 1):
            for j in range(1, target + 1):
                for k in range(1, f + 1):
                    if (j - k) < 0:
                        break
                    dp[i][j] += dp[i-1][j-k]
                    dp[i][j]  = dp[i][j]%(10**9 + 7)
        
        return dp[-1][-1]
class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        mod = 10**9 + 7
        dp = [[0] * (target + 1) for _ in range(d+1)]
        dp[0][0] = 1
        for i in range(1, d+1):
            for j in range(1, target+1):
                
                #dp[i][j] = dp[i-1][j]
                for k in range(1, f+1):
                    if 0 <= j - k:
                        dp[i][j] = (dp[i][j] + dp[i-1][j-k]) % mod
                
                        
        return dp[d][target]
        

class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        mod = 1000000007
        dp = [[0] * (d+1) for i in range(target+1)]
        for tar in range(1, target+1):
            if f >= tar:
                dp[tar][1] = 1
        for dice in range(2, d+1):
            for tar in range(1, target+1):
                for possVal in range(1, f+1):
                    if tar - possVal >= 1:
                        dp[tar][dice] = (dp[tar][dice] + dp[tar-possVal][dice-1]) % mod
        print(dp)
        return dp[-1][-1]
class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        
        if not d or not f or not target: return 0
        
        dp = [[0]*(target+1) for _ in range(d+1)]
        dp[0][0] = 1
        for i in range(1,len(dp)):
            for j in range(1,len(dp[0])):
                k = 1
                while k <= min(j,f):
                    dp[i][j] += dp[i-1][j-k]
                    k += 1
                
        return dp[-1][-1] % (10**9+7)

class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        table=[[0]*(target+1) for i in range(d+1)]  
      
        for j in range(1,min(f+1,target+1)):  
            table[1][j]=1
          
        for i in range(2,d+1): 
            for j in range(1,target+1): 
                for k in range(1,min(f+1,j)): 
                    table[i][j]+=table[i-1][j-k] 
     
        return table[-1][-1] % 1000000007
        

class Solution:
    def numRollsToTarget(self, d: int, f: int, s: int) -> int:
        mem = [[0 for _ in range(s + 1)] for _ in range(d + 1)]
        mem[0][0] = 1
        for i in range(1,d + 1):
            for j in range(1,s + 1):
                mem[i][j] = mem[i][j - 1] + mem[i - 1][j - 1]
                if j - f - 1 >= 0:
                    mem[i][j] -= mem[i - 1][j - f - 1]
        return mem[-1][-1] % (10 ** 9 + 7)
class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        dp = [[0] * (target + 1) for _ in range(d + 1)]
        MOD = 10 ** 9 + 7
        dp[0][0] = 1

        # for j in range(target + 1):
        #     dp[0][j] = 1

        for j in range(1, d + 1):
            for i in range(1, target + 1):
                for k in range(1, f + 1):
                    if i - k < 0:
                        break
                    dp[j][i] = (dp[j][i] + dp[j - 1][i - k]) % MOD

        return dp[d][target]


class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        dp = [[0 for i in range(target+1)] for j in range(d)]
        print(dp)
        for i in range(f):
            if i + 1 <= target:
                dp[0][i+1] = 1
                
        for i in range(1,d):
            for j in range(1,target+1):
                #start with k = 1, accumulate all possible number sum before f
                #if this dice presents 1, then the sum up to the current j is related to the result in j-k, the previous sum in i-1 try.
                
                k = 1
                while j-k > 0 and k <= f:
                    
                    dp[i][j] += dp[i-1][j-k]
                    k+=1
                    
        return dp[-1][-1] % (1000000000+7)
class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        # time O(dft), space O(dt) --> optimize space O(t)
        dp = [[0] * (target + 1) for _ in range(d + 1)]
        dp[0][0] = 1
        
        for i in range(1, d + 1):
            for j in range(1, target + 1):
                for k in range(1, f + 1):
                    if (j - k) < 0:
                        break
                    dp[i][j] += dp[i-1][j-k]
                    dp[i][j]  = dp[i][j]%(10**9 + 7)
        
        return dp[-1][-1]
class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        
        dp=[[0]*(target+1) for _ in range(d+1)]
        md=pow(10,9)+7
        dp[0][0]=1
        for trial in range(1,d+1):
            for i in range(target):
                for face in range(1,f+1):
                    if i+face<=target:
                        dp[trial][i+face]+=dp[trial-1][i]
                        dp[trial][i+face]%=md
        
        return dp[-1][target]
            

class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        dp = [[0 for _ in range(d+1)] for _ in range(target+1)]
        if target < d: return 0
        for i in range(2,d+1): dp[1][i] = 0
        for i in range(1,min(f+1,target+1)): dp[i][1] = 1            
        for D in range(2,d+1):
            for s in range(1,target+1):
                for k in range(1,f+1):
                    dp[s][D] += dp[s-k][D-1] if s-k >= 1 and D > 1  else 0
                    dp[s][D] = dp[s][D]%1000000007
        return dp[target][d]
class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        
        dp = [[0] * (target+1) for i in range(d+1)]
        
      
       
            
        # for i in range(d+1):
        #     if 
        #     dp[1][0] = 1
        
        dp[0][0] = 1
        
        for i in range(d+1):
            for j in range(target+1):
                
                for k in range(1,f+1):
                    if j-k>=0:
                        
                        dp[i][j] += dp[i-1][j-k]
                    
                   
                    
        return dp[d][target] % (10 ** 9 + 7)
class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        
        dp=[[0]*(target+1) for i in range(d+1)]
        dp[0][0]=True
        
        for i in range(1,d+1):
            for j in range(1,target+1):
                k=1
                while k<=min(j,f):
                    dp[i][j]+=dp[i-1][j-k]
                    k+=1
        
        mod=10**9+7
        return dp[-1][-1]%mod
class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        dice = [k for k in range(1,f+1)]
        dp = [[0 for _ in range(d+1)]for _ in range(target+1)]
        
        def numRolls(d,f,target):
            for j in range(1,min(f+1,target+1)): dp[j][1] = 1
            for i in range(2,d+1):
                for j in range(1,target+1):
                    for new_t in list(map(lambda k: j-k,dice)):
                        if new_t >= 0 :
                            dp[j][i] += dp[new_t][i-1]
                        dp[j][i] 
            return dp[target][d]% (10**9 + 7)
        
                        
        return numRolls(d,f,target)
class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        dp=[[0]*(target+1) for i in range(d+1)]
        
        dp[0][0]=1
        mod=10**9+7
        for i in range(1,d+1):
            for j in range(1,target+1):
                k=1
                while k<=min(j,f):
                    dp[i][j]+=dp[i-1][j-k]
                    k+=1
        
        return dp[-1][-1]%mod
class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        mod = 10 ** 9 + 7
        if target > d * f:
            return 0
        
        dp = [[0 for j in range(target+1)] for i in range(d+1)]
        for j in range(1, min(f, target)+1):
            dp[1][j] = 1
        for i in range(2, d+1):
            for j in range(1, target+1):
                for k in range(max(1, j-f), j):
                    dp[i][j] += dp[i-1][k]
        return dp[d][target] % mod
class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        
        mod = 10 ** 9 + 7
        dp = [[0 for j in range(target + 1)] for i in range(d + 1)]
        dp[0][0] = 1
        for i in range(d):
            for ff in range(1, f + 1):
                for sm in range(target):
                    if sm + ff <= target:
                        dp[i + 1][sm + ff] += dp[i][sm]
                        dp[i + 1][sm + ff] %= mod
        return dp[d][target]


class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        dp = [[0]*(target+1) for _ in range(d+1)]
        dp[0][0] = 1
        
        for i in range(1, d+1):
            for j in range(target, i-1, -1):
                for k in range(1, f+1):
                    if j-k<0: break
                    
                    dp[i][j] += dp[i-1][j-k]
        # print(dp)
        return dp[d][target]%(10**9+7)
class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        dp=[[0]*(target+1) for i in range(d+1)]
        
        dp[0][0]=1
        
        for i in range(1,d+1):
            for j in range(1,(target+1)):
                k =1
                while k <= min(f,j):
                    dp[i][j] = (dp[i][j] + dp[i-1][j-k])
                    k +=1
        return dp[d][target]% (10**9 + 7)

class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        mod = 10**9 + 7
        dp = [[0 for j in range(target + 1)] for i in range(d + 1)]
        dp[0][0] = 1
        for i in range(1, d + 1):
            for ff in range(1, f + 1):
                for sm in range(target):
                    if sm + ff <= target:
                        dp[i][sm + ff] += dp[i - 1][sm]
                        dp[i][sm + ff] %= mod
        return dp[d][target]
class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        dp = [[None for _ in range(d+1)] for _ in range(target+2)]
        for i in range(target+2):
            dp[i][d] = 0
        dp[target][d] = 1
        for i in range(d+1):
            dp[target+1][i] = 0
        for i in range(target, -1, -1):
            for j in range(d-1, -1, -1):
                dp[i][j] = 0
                for k in range(1, f+1):
                    dp[i][j]+= dp[min(i+k, target+1)][j+1] % 1000000007
        return dp[0][0] % 1000000007

MODULE = 10**9 + 7
class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        if target < d:
            return 0
        #u7528 j u4e2au6570u6765u51d1u51fau548cu4e3a j u7684u5168u90e8u7ec4u5408u53efu80fdu6027
        dp = [[0] * (target + 1) for _ in range(d + 1)]
        for i in range(1, f + 1):
            if i > target:
                break
            dp[1][i] = 1
            
        for i in range(1, d + 1):
            for j in range(1, target + 1):
                for face in range(1, f + 1):
                    if j >= face:
                        dp[i][j] = (dp[i][j] + dp[i - 1][j - face]) % MODULE
        # print (dp)
        return dp[d][target]

class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        dp=[[0]*(target+1) for i in range(d+1)]
        
        dp[0][0]=1
        
        for i in range(1,d+1):
            for j in range(1,target+1):
                k=1
                while k<=min(j,f):
                    dp[i][j]+=dp[i-1][j-k]
                    k+=1
        
        mod=10**9+7
        return dp[-1][-1]%mod
class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        dp=[[0]*(target+1) for i in range(d+1)]
        
        dp[0][0]=1
        
        for i in range(1,d+1):
            for j in range(1,target+1):
                k=1
                while k<=min(j,f):
                    dp[i][j]+=dp[i-1][j-k]
                    k+=1
        
        mod=10**9+7
        return dp[-1][-1]%(mod)
class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        dp = [[0 for _ in range(max(target+1,f+1))] for _ in range(d+1)]
        for i in range(1,f+1):
            dp[1][i] = 1
        break_flag = False
        for i in range(1,d):
            break_flag = False
            
            for j in range(1,max(target+1,f+1)):
                for n in range(1,f+1):
                    if j+n <= max(target,f):
                        dp[i+1][j+n] += dp[i][j] 
                    else:
                        break
                        break_flag = True
                if break_flag:
                    break
                        
                
        return dp[d][target] % (10**9 + 7)
class Solution:
    def numRollsToTarget(self, d: int, f: int, s: int) -> int:
        mem = [[0 for _ in range(s + 1)] for _ in range(d + 1)]
        mem[0][0] = 1
        for i in range(1,d + 1):
            for j in range(1,s + 1):
                mem[i][j] = mem[i][j - 1] + mem[i - 1][j - 1]
                if j - f - 1 >= 0:
                    mem[i][j] -=  mem[i - 1][j - f - 1]
                    
        return mem[-1][-1] % (10 ** 9 + 7)
                    

class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        dp = [[0 for _ in range(max(target+1,f+1))] for _ in range(d+1)]
        for i in range(1,f+1):
            dp[1][i] = 1
        for i in range(1,d):
            for j in range(1,max(target+1,f+1)):
                for n in range(1,f+1):
                    if j+n <= max(target,f):
                        dp[i+1][j+n] += dp[i][j] 
                    else:
                        break
        return dp[d][target] % (10**9 + 7)
class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        dp = [[0]*(target+1) for _ in range(d+1)]
        dp[0][0] = 1
        MOD = 10**9+7
        
        for i in range(d):
            for j in range(target+1):
                for k in range(1, f+1):
                    if j+k<=target:
                        dp[i+1][j+k] += dp[i][j]
                        dp[i+1][j+k] %= MOD

        return dp[d][target]

class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        mod = 10**9 + 7
        dp = [ [ 0 for _ in range( target + 1 ) ] for _ in range( d + 1 ) ]
        dp[ 0 ][ 0 ] = 1
        for i in range( d ) :
            for j in range( 1 , f + 1 ) :
                for k in range( target ) :
                    if k + j <= target :
                        dp[ i + 1 ][ k + j ] += dp[ i ][ k ]
                        dp[ i + 1 ][ k + j ] %= mod
        return dp[-1][-1]
#Sanyam Rajpal

class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        if d == 0:
            return 0
        q = collections.deque([(i+1, 1) for i in range(f)])
        visited = {(0,0): 1}
        while(q):
            s, di = q.popleft()
            if (s, di) in visited:
                continue
            npath = sum(visited.get((s-i-1, di-1), 0) for i in range(f))
            if (di==d):
                if (s==target):
                    return npath % (1000000007)
                else:
                    continue
            visited[(s, di)] = npath
            q += [(s+i+1, di+1) for i in range(f)]
        return 0
        
            

class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        if not d or not f or not target: return 0
        
        dp = [[0]*(target+1) for _ in range(d+1)]
        
        dp[0][0] = 1
        for i in range(1,len(dp)):
            for j in range(1,len(dp[0])):
                k=1
                while k <= min(j,f):
                    dp[i][j] += dp[i-1][j-k] % (10**9 + 7)
                    k+=1
                
        
        return dp[-1][-1] % (10**9 + 7)

class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        dp = [[0]*(target+1) for _ in range(d+1)]
        dp[0][0] = 1
        mod = 10**9 + 7
        for i in range(1, d+1):
            for j in range(1, target+1):
                k = 1
                while k <= min(j, f):
                    dp[i][j] = (dp[i][j] + dp[i-1][j-k]) % mod
                    k += 1
        return dp[d][target] % mod            

from itertools import product

class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        dp = [[0 for i in range(target + 1)] for j in range(d + 1)]
        dp[0][0] = 1
        mod = 10 ** 9 + 7
        for i in range(1, d + 1):
            for j in range(1, target + 1):
                k = 1
                while k <= min(j, f):
                    dp[i][j] = (dp[i][j] + dp[i - 1][j - k]) % mod
                    k += 1
        return dp[d][target] % mod

class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        mod = 10**9 + 7
        dp = [[0 for j in range(target + 1)] for i in range(d + 1)]
        dp[0][0] = 1  # * 1 way to make 0 with 0 dice
        for i in range(d):
            for ff in range(1, f + 1):
                for sm in range(target):
                    if sm + ff <= target:
                        dp[i + 1][sm + ff] += dp[i][sm]
                        dp[i + 1][sm + ff] %= mod
                    else:
                        break
        return dp[d][target]

class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        dp = [[0 for i in range(target + 1)] for j in range(d + 1)]
        dp[0][0] = 1
        mod = 10 ** 9 + 7
        for i in range(1, d + 1):
            for j in range(1, target + 1):
                k = 1
                while k <= min(j, f):
                    dp[i][j] = (dp[i][j] + dp[i - 1][j - k]) % mod
                    k += 1
        return dp[d][target] % mod
class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        #dp, tree -> check which leaf has val = target 
        # recursion : memory overload
        # ways of d dices with f faces to sum up to target = ways of d-1 dices to sum to target-1 + ways of d-1 dices to sum to target-2 +..
        # rolls[d][target] 
        
        rolls = []
        for i in range(d+1):
            rolls.append([0]*(target+1))
        
        for i in range(target+1):
            rolls[0][i] = 0
            
        for i in range(d+1):
            rolls[i][0] = 0
            
        for i in range(1,target+1):
            if i<=f:
                rolls[1][i] = 1
        
        for i in range(2,d+1):
            for j in range(1, target+1):
                k = 1
                while(k<=f):
                    if(j-k) < 0:
                        break
                    rolls[i][j] += rolls[i-1][j-k]
                    k += 1
        
        #print(10**9)
        return (rolls[d][target] %(10**9 + 7))
class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        # d = 5, f=3, target=11
        # d=4, f=3, target=10 + d=4, f=3, target=9, d=4 f=3 target=9
        def rec_find_target_sum(d, f, target):
            if (d,f,target) in memo:
                return memo[(d,f,target)]
            if d == 0 or d*f < target or target <= 0:
                return 0
            if d == 1:
                return 1
            
            possibs = 0
            for face in range(f):
                possibs += rec_find_target_sum(d-1, f, target-face-1)
            memo[(d,f,target)] = possibs
            return possibs
        
        memo = {}
        return rec_find_target_sum(d, f, target) % (10 ** 9 + 7)
class Solution:
    def numRollsToTarget(self, d,f,target):
        dp = [[0]*(target+1) for _ in range(d+1)]
        
        dp[0][0] = 1
        mod = 10**9 + 7
        
        for i in range(1 , d+1):
            for j in range(1 , target+ 1):
                k = 1 
                while k <= min(f , j):
                    dp[i][j] = (dp[i][j]  + dp[i-1][j-k])%mod
                    k +=  1
        return dp[d][target]
         
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        mod = 10 ** 9 + 7
        dp = [[0 for j in range(target + 1)] for i in range(d + 1)]
        dp[0][0] = 1
        for i in range(d):
            for ff in range(1, f + 1):
                for sm in range(target):
                    if sm + ff <= target:
                        dp[i + 1][sm + ff] += dp[i][sm]
                        dp[i + 1][sm + ff] %= mod
        return dp[d][target]

class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        
        dp = [[0 for i in range(target + 1)] for j in range(d + 1)]
        dp[0][0] = 1
        
        
        for i in range(1, d + 1):
            for j in range(1, target + 1):
                k = 1
                while k <= min(j, f):
                    dp[i][j] = (dp[i][j] + dp[i - 1][j - k]) % (10 ** 9 + 7)
                    k += 1
        
        return dp[d][target] % (10 ** 9 + 7)
    

class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        mod = 10**9 + 7
        dp = [[0] * (target + 1) for _ in range(d+1)]
        dp[0][0] = 1
        for i in range(1, d+1):
            for j in range(i, target+1):
                
                #dp[i][j] = dp[i-1][j]
                for k in range(1, f+1):
                    if 0 <= j - k:
                        dp[i][j] += dp[i-1][j-k]
                dp[i][j] %= mod
                        
        return dp[d][target] % mod
        

class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        dp = [[0 for i in range(target + 1)] for j in range(d + 1)]
        dp[0][0] = 1
        mod = 10 ** 9 + 7
        for i in range(1, d + 1):
            for j in range(1, target + 1):
                k = 1
                while k <= min(j, f):
                    dp[i][j] = (dp[i][j] + dp[i - 1][j - k]) 
                    k += 1
        return dp[d][target] % mod
class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        dice = [k for k in range(1,f+1)]
        dp = [[0 for _ in range(d+1)]for _ in range(target+1)]
        
        def numRolls(d,f,target):
            for j in range(1,min(f+1,target+1)): dp[j][1] = 1
            for i in range(2,d+1):
                for j in range(1,target+1):
                    for new_t in list([j-k for k in dice]):
                        if new_t >= 0 :
                            dp[j][i] += dp[new_t][i-1]
                        dp[j][i] %= (10**9 + 7)
            return dp[target][d]
        
                        
        return numRolls(d,f,target)
    
    # def numRollsToTarget(self, d: int, f: int, target: int) -> int:
    #     if d*f < target: return 0        # Handle special case, it speed things up, but not necessary
    #     elif d*f == target: return 1     # Handle special case, it speed things up, but not necessary
    #     mod = int(10**9 + 7)
    #     dp = [[0] * (target+1) for _ in range(d+1)] 
    #     for j in range(1, min(f+1, target+1)): dp[1][j] = 1
    #     for i in range(2, d+1):
    #         for j in range(1, target+1):
    #             for k in range(1, f+1):
    #                 if j - k >= 0: dp[i][j] += dp[i-1][j-k]
    #             dp[i][j] %= mod        
    #     return dp[-1][-1] 

class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        # Dynamic programming on the number of dice?
        # if i have n - 1 dice, and know the number of ways to get to all possible targets
        # then i should be able to compute the number of way to get to a target with n dice
        
       # keep a vector of number of ways to get to targets between 1 and "target" for d = 1 etc
    
        dp = [[0 for i in range(target + 1)] for j in range(d + 1)]
        dp[0][0] = 1
        mod = 10 ** 9 + 7
        for i in range(1, d + 1):
            for j in range(1, target + 1):
                k = 1
                while k <= min(j, f):
                    dp[i][j] = (dp[i][j] + dp[i - 1][j - k]) % mod
                    k += 1
        return dp[d][target] % mod

class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        dp = [[0 for _ in range(target+1)] for _ in range(d+1)]
        dp[0][0] = 1
        mod = 10 **9 + 7
        for i in range(1,len(dp)):
            for j in range(1,len(dp[0])):
                k = 1
                while k <= min(j,f):
                    dp[i][j] = (dp[i][j]+ dp[i-1][j-k]) % mod
                    k += 1
        return dp[-1][-1] % mod
        

class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        memo = {} # num of ways for combo of [# of rolls, target]
        def rollDice(d, target):
            if target < 0 or target > f * d: # def 0 way
                memo[d, target] = 0
                return 0
            if d == 0: # base case 
                return 0 if target > 0 else 1
            if (d, target) in memo: # if already computed
                return memo[d, target]
            total = 0  # actual computation
            for num in range(1, f + 1):
                total += rollDice(d - 1, target - num) #dp based on previous roll 
            memo[d, target] = total
            return total
            
        return rollDice(d, target) % (10**9 + 7)
        
        
    # for each roll, f doesn't change, dp[d, target] = dp[d-1, target-1] + dp[d-1, target-2] + ... + dp[d-1, target-f]. That is, cur total # of steps is sum of total # of steps of all possible last steps (determined by # of faces last dice has)
    # base case is when d is 0 (no roll left), target is 0, there is one way. If target > 0 and d is 0, there is no way. 
    # bcs for each layer we have, the num of operations for the same d value is timed by f, there's a lot of repetition value there. Therefore we use memoization

class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        def helper(h, d, target):
            # if target is too small or if it is out of range
            if target <= 0 or target > (d * f):
                return 0
            if d == 1:
                return 1        # no need to check if target is within reach; already done before
            if (d, target) in h:
                return h[(d, target)]        # directly access from hash table
            res = 0
            for i in range(1, f + 1):
                res += helper(h, d - 1, target - i)       # check all possible combinations
            h[(d, target)] = res
            return h[(d, target)]
        
        h = {}
        return helper(h, d, target) % (10 ** 9 + 7)

class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        dp = [1]+[0]*target
        for i in range(d):
            for j in range(target,-1,-1):
                dp[j] = sum(dp[max(0,j-f):j])
                
        return dp[target] % (10 ** 9 + 7)
class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        if target < d or target > d * f:
            return 0
        if target > (d*(1+f)/2):
            target = d * (1 + f) - target
        dp = [0] * (target + 1) 
        for i in range(1, min(f, target) + 1):
            dp[i] = 1
        for i in range(2, d + 1):
            new_dp = [0] * (target + 1)
            for j in range(i, min(target, i * f) + 1):
                new_dp[j] = new_dp[j - 1] + dp[j - 1]
                if j - 1 > f:
                    new_dp[j] -= dp[j - f - 1]
            dp = new_dp

        return dp[target] % (10 ** 9 + 7)
class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        def helper(h, d, target):
            # if target is too small or if it is out of range
            if target <= 0 or target > (d * f):
                return 0
            if d == 1:
                return 1        # no need to check if target is within reach; already done before
            if (d, target) in h:
                return h[(d, target)]        # directly access from hash table
            res = 0
            for i in range(1, f + 1):
                res += helper(h, d - 1, target - i)       # check all possible combinations
            h[(d, target)] = res
            return h[(d, target)]
        
        h = {}
        return helper(h, d, target) % (10 ** 9 + 7)
class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        if d*f < target: return 0        # Handle special case, it speed things up, but not necessary
        elif d*f == target: return 1     # Handle special case, it speed things up, but not necessary
        mod = int(10**9 + 7)
        dp = [[0] * (target+1) for _ in range(d+1)] 
        for j in range(1, min(f+1, target+1)): dp[1][j] = 1
        for i in range(2, d+1):
            for j in range(1, target+1):
                for k in range(1, f+1):
                    if j - k >= 0: dp[i][j] += dp[i-1][j-k]
                dp[i][j] %= mod        
        return dp[-1][-1] 
class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        # if d*f < target:
        #     return 0
        # elif d*f==target:
        #     return 1
        
        faces = [i for i in range(1,f+1)]
        cache = {}
        def dfs(left,numDice):
            re = 0
            if (left,numDice) in cache:
                return cache[(left,numDice)]
            if left>f*(d-numDice):
            #if left>f*d:
                cache[(left,numDice)] = 0
                return 0
            if numDice == d:
            # if d*f==
                return 1 if left == 0 else 0
            else:
                for face in faces:
                    numDice += 1
                    if left-face>=0:
                        re += dfs(left-face,numDice)
                    numDice -= 1
                
                cache[(left,numDice)] = re
                return re
        return dfs(target,0)%(10**9 + 7)
from functools import lru_cache

MOD = 10 ** 9 + 7

class Solution:
    
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        sol = [0] * (target + 1)
        sol[0] = 1
        for _ in range(1,d+1):
            nxt = [0] * (target + 1)
            for i in range(target + 1):
                start, end = max(i-f, 0), i-1
                nxt[i] = sum(sol[start:end+1]) % MOD
            sol = nxt
        return sol[target]
class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        if d*f < target: 
            return 0        # Handle special case, it speed things up, but not necessary
        elif d*f == target: 
            return 1     # Handle special case, it speed things up, but not necessary
        mod = int(10**9 + 7)
        dp = [[0] * (target+1) for _ in range(d+1)] 
        for j in range(1, min(f+1, target+1)): 
            dp[1][j] = 1
        for i in range(2, d+1):
            for j in range(1, target+1):
                for k in range(1, f+1):
                    if j - k >= 0: dp[i][j] += dp[i-1][j-k]
                dp[i][j] %= mod        
        return dp[-1][-1] 
                    

class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        res = [[0 for _ in range(target + 1)] for _ in range(d + 1)]
        modula = (10**9)+7
        res[0][0] = 1
        for dice in range(1, d + 1):
            for i in range(dice, target + 1):
                interestMinRange = max(0, i - f)
                res[dice][i] = sum(res[dice - 1][interestMinRange:i]) % modula
        return res[-1][-1]
                

class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
    
        memo = {}
        
        def helper(dices, total):
            # print(dices, total)
            if total < 0:
                return 0
            if dices == 0 and total == 0:
                return 1
            if dices > 0 and total == 0:
                return 0
            if dices * f < total:
                memo[(dices, total)] = 0
                return 0

            if (dices, total) in memo:
                return memo[(dices, total)]
            
            ways = 0
            for num in range(1, f + 1):
                ways += helper(dices - 1, total - num)
            
            memo[(dices, total)] = ways
            # print(ways)
            return ways
        
        return helper(d, target) % (10 ** 9 + 7)
class Solution:
    def numRollsToTarget(self, d, f, target):
        if (target < d) or (target > d*f): return 0
        dp = [[0 for _ in range(d*f+1)] for _ in range(d+1)]
        for v in range(1, f+1):
            dp[1][v] = 1
        for dcnt in range(2, d+1):
            for total in range(dcnt, target+1):
                for v in range(1, f+1):
                    dp[dcnt][total] += dp[dcnt-1][total-v]
        return dp[d][target] % (10**9 + 7)
    
class Solution:
    def numRollsToTarget(self, d, f, target):
        if (target < d) or (target > d*f): return 0
        dp = [[0 for _ in range(d*f+1)] for _ in range(d+1)]
        for v in range(1, f+1):
            dp[1][v] = 1
        for dcnt in range(2, d+1):
            for total in range(dcnt, target+1):
                for v in range(1, f+1):
                    dp[dcnt][total] += dp[dcnt-1][total-v]
        return dp[d][target] % (10**9 + 7)
class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        if target < d or target > d * f:
            return 0
        mod_val = 10 ** 9 + 7
        dp = [0] * (target + 1)
        dp[0] = 1
        for r in range(1, d + 1):
            if r > 1:
                dp[r - 2] = 0
            for i in range(target, r - 1, -1):
                dp[i] = sum(dp[max(0, i - f) : i]) % mod_val
        return dp[target]

class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
           if target<d or target>d*f:
                return 0
           if target==d*f:
              return 1
           mem={};mod=int(1e9+7)
           def dice(k,target):
                if target==0:
                    return 0
                if k==1:
                    if target<=f:
                       return 1
                    else:
                        return 0
                if (k,target) in mem:
                    return mem[(k,target)]
                ans=0 
                for i in range(1,f+1):
                    if target>=i and target<=f*k:
                        ans+=dice(k-1,target-i)%mod
                ans%=mod
                mem[(k,target)]=ans
                return ans
           return dice(d,target) 
class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        
        module = 10 ** 9 + 7
        
        
        
        self.m = {}
        
        def dfs(d,f,target):
            
            if d == 0 and target == 0:
                return 1
            
            if d == 0 and target != 0:
                return 0
            if d*f < target or d > target:
                return 0
            
            if (d, target) in self.m:
                return self.m[(d, target)]
            
            
            res  = 0
            for i in range(1, f+1):
                res = (res + dfs(d - 1, f, target - i)) % module

            self.m[(d,target)] = res
            return res

        return dfs(d,f,target)
            
            
        
       
            
            
        

class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        module = 10 ** 9 + 7
        
        
        
        self.m = {}
        
        def dfs(d,f,target):
            
            if d == 0 and target == 0:
                return 1
            
            if d == 0 and target != 0:
                return 0
            if d*f < target or d > target:
                return 0
            
            if (d, target) in self.m:
                return self.m[(d, target)]
            
            
            res  = 0
            for i in range(1, f+1):
                res = (res + dfs(d - 1, f, target - i)) % module

            self.m[(d,target)] = res
            return res

        return dfs(d,f,target)
            

class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        if target > d*f or target < d:
            return 0
    
        targets = {}
        faces = [i for i in range(1,f+1)]
        
        def solve(target,numDice):
            ways = 0
            if (target,numDice) in targets:
                return targets[(target,numDice)]
            if target > f*(d-numDice):
                targets[(target,numDice)]=0
                return 0
            if numDice==d:
                return 1 if target==0 else 0
            else:
                for face in faces:
                    if target-face>=0:
                        ways += solve(target-face,numDice+1)
                
                targets[(target,numDice)]=ways
                return ways
        return solve(target,0)%(10**9+7)
        
        

class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        def foo(dices, need):
            if dices == 0: return int(need==0)
            elif (dices, need) in m: return m[(dices, need)]
            elif need > f*dices: m[(dices, need)] = 0; return 0
            ret = 0
            for i in range(1,f+1):
                ret += foo(dices-1, need-i)
            m[(dices, need)] = ret
            return ret
        m = {}
        return foo(d, target)%(10**9 + 7)
class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        if target > f*d:
            return 0
        
        dp = [[0 for _ in range(target + 1)] for _ in range(d + 1)]
        dp[0][0] = 1
        
        for i in range(d + 1):
            for j in range(1, target + 1):
                for k in range(1, f + 1):
                    if k > j:
                        continue
                    dp[i][j] = (dp[i][j] + dp[i - 1][j - k])%(10 ** 9 + 7)
        
        return dp[-1][-1]

MOD = (10 ** 9) + 7
from collections import defaultdict

class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        memo = {}
        
        if target > f*d: return 0
        if target == f*d: return 1
        
        def F(j,memo,left):
            if (j,left) in memo.keys(): return memo[(j,left)]
            
            if left < 0: return 0
            
            if j == d-1:
                memo[(j,left)] = int(left <= f and left >= 1)
                return memo[(j,left)]
            
            ans = 0
            for i in range(1,f+1):
                ans += F(j+1,memo,left - i)
                
            memo[(j,left)] = ans
            return memo[(j,left)]
            
            
        F(0,memo,target)
        
        return memo[(0,target)] % MOD
class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        
        if d*f < target:
            return 0
        elif d*f == target:
            return 1
        
        dp = [[0]*(target+1) for _ in range(d+1)]
        for i in range(1, min(f+1, target+1)):
            dp[1][i] = 1
        
        print(dp)
        
        for i in range(1,d+1):
            for j in range(1,target+1):
                for k in range(1,f+1):
                    if k <= j:
                        dp[i][j] += dp[i-1][j-k]
                    dp[i][j] %= (10**9+7) 
            
        print(dp)
        return dp[-1][-1] 
class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        if d*f < target: return 0
        if d*f == target: return 1
        
        def dfs(j,left,memo):
            
            if (j,left) in memo.keys():
                return memo[(j,left)]
            
            if left < 0:
                return 0
            
            if j == d-1:
                return int(left > 0 and left <= f)
            
            ans = 0
            for i in range(1,f+1):
                ans += dfs(j+1,left-i,memo)
            
            memo[(j,left)] = ans
            
            return memo[(j,left)]
        
        return dfs(0,target,{}) % (10 ** 9 + 7)
class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        matrix = [[1 if 1 <= x <= f and y == 0 else 0 for x in range(target+1) ] for y in range(0, d)]
        for row in range(1, d):
            for col, val in enumerate(matrix[row]):
                matrix[row][col] = sum(matrix[row-1][max(0, col-f):max(0, col)])
        
        return matrix[d-1][target] % (10**9 + 7)
class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        if target < d or target > d * f:
            return 0
        dp = [0] * (target + 1)
        dp[0] = 1
        mod_val = 10 ** 9 + 7
        for r in range(1, d + 1):
            if r > 1:
                dp[r - 2] = 0
            for i in range(target, r - 1, -1):
                dp[i] = sum(dp[max(0, i - f) : i]) % mod_val
        return dp[target]
# REMEMBER: WHERE TO PUT THE MEMOIZE SAVE AND LOOKUP LINES IN THE RECURSIVE FUNCTION.
# 1. BEFORE any work is done + AFTER all the work is done and we're about to return the answer.

class Solution:
    def __init__(self):
        self.memo = {} # maps (d, target) to num ways.
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        # obvious base case.
        if d * f < target:
            return 0
        if d == 1:
            return 1
        
        if (d, target) in self.memo:
            return self.memo[(d, target)]
        # we do it recursively
        numWays = 0
        #for i in range(f, 0, -1):
        for i in range(1, f+1):
            # we roll with i, then we
            if target - i > 0:            
                ways = self.memo.get((d-1, target - i), self.numRollsToTarget(d - 1, f, target - i))
                numWays += ways
        self.memo[(d, target)] = numWays
        return numWays % (10**9 + 7)
class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        if d*f < target:
            return 0
        if d*f == target:
            return 1
        dp = [[0]*(target+1) for i in range(d+1)]
        for i in range(1, min(f+1, target+1)):
            dp[1][i] = 1
        for i in range(2,d+1):
            for j in range(1,target+1):
                for k in range(1,f+1):
                    if j-k >= 0:
                        dp[i][j] += dp[i-1][j-k]
                    dp[i][j] %= 10**9+7
        return dp[-1][-1]
# REMEMBER: WHERE TO PUT THE MEMOIZE SAVE AND LOOKUP LINES IN THE RECURSIVE FUNCTION.
# 1. BEFORE any work is done + AFTER all the work is done and we're about to return the answer.

class Solution:
    def __init__(self):
        self.memo = {} # maps (d, target) to num ways.
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        # obvious base case.
        if d * f < target:
            return 0
        if d == 1:
            return 1
        
        if (d, target) in self.memo:
            return self.memo[(d, target)]
        # we do it recursively
        numWays = 0
        for i in range(f, 0, -1):
        #for i in range(1, f+1):
            # we roll with i, then we
            if target - i > 0:            
                ways = self.memo.get((d-1, target - i), self.numRollsToTarget(d - 1, f, target - i))
                numWays += ways
        self.memo[(d, target)] = numWays
        return numWays % (10**9 + 7)
class Solution:
    
    MODULO = 10**9 + 7
    table = {}
    
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        return self.recursive(d, f, target) % self.MODULO
        
    def recursive(self, d: int, f: int, target: int):
        if target < d or target > d*f: 
            return 0
        if d == 1 and target <= f:
            return 1
        sum_ = 0
        for i in range(1, f+1):
            idx = (d - 1, f, target - i)
            if idx not in self.table:
                self.table[idx] = self.recursive(*idx)
            sum_ += self.table[idx] % self.MODULO
        return sum_
class Solution:
    def __init__(self):
        self.memo = {} # maps (d, target) to num ways.
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        # obvious base case.
        if d * f < target:
            return 0
        if d == 1:
            return 1
        
        if (d, target) in self.memo:
            return self.memo[(d, target)]
        # we do it recursively
        numWays = 0
        for i in range(f, 0, -1):
            # we roll with i, then we
            if target - i > 0:            
                ways = self.memo.get((d-1, target - i), self.numRollsToTarget(d - 1, f, target - i))
                numWays += ways
        self.memo[(d, target)] = numWays
        return numWays % (10**9 + 7)
class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        memo = {}
        def solve(dice, target):
            if target > dice*f:
                memo[dice, target] = 0
                return 0
            if dice == 0:
                return target == 0
            if target == 0:
                return 0
            if (dice, target) in memo:
                return memo[dice, target]
            ways = 0
            for num in range(1, f+1):
                ways += solve(dice-1, target-num)
            memo[dice, target] = ways
            return ways
            
            
        return solve(d, target)%(10**9+7)
class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        if target > d*f:
            return 0
        dp = [[0 for _ in range(d*f+1)] for _ in range(d+1)]
        dp[0][0] = 1
        for i in range(1, d+1):
            for j in range(i, i*f+1):
                for t in range(1, f+1):
                    if (j-t) >= i-1 and (j-t) <= (i-1)*f:
                        dp[i][j] += dp[i-1][j-t]
        return dp[d][target] % (10**9+7)
class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        memo= {}
        def rolls(dice, t):
            if t > f * dice:
                memo[(dice, t)] = 0
                return 0
            if dice == 0:
                return t == 0
            if target < 0:
                return 0
            if (dice, t) in memo:
                return memo[(dice, t)]
            solu = 0
            for num in range(1, f + 1):
                solu += rolls(dice - 1, t - num)
            memo[(dice, t)] = solu
            return solu
        return rolls(d, target) % (10**9 + 7)
                
        

class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        @lru_cache(None)
        def dfs(dice_left, curr_sum):
            if curr_sum > target:
                return 0
            if dice_left == 0:
                return 1 if curr_sum == target else 0
            count = 0
            for i in range(1, f + 1):
                count += dfs(dice_left - 1, curr_sum + i)
            return count % 1000000007
        return dfs(d, 0)
class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        MOD = 10 ** 9 + 7

        @lru_cache(None)
        def dp(dice, target):
            # number of ways to form target with `dice` remaining
            if target == 0:
                return int(dice == 0)
            elif target < 0 or dice <= 0:
                return 0
            else: # target >= 0 and dice >= 0
                res = 0
                for x in range(1, f + 1):
                    res += dp(dice - 1, target - x)
                return res % MOD

        return dp(d, target)
class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        if target < d or target > d * f:
            return 0
        dp = [0] * (target + 1)
        dp[0] = 1
        mod_val = 10 ** 9 + 7
        for r in range(1, d + 1):
            if r > 1:
                dp[r - 2] = 0
            for i in range(target, r-1, -1):
                dp[i] = sum(dp[max(0, i - f) : i]) % mod_val
        return dp[target]
import functools
class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        @functools.lru_cache(None)
        def helper(idx, curr):
            if curr > target: return 0
            if idx == d:
                return int(curr == target)
            res = 0
            for num in range(1, f + 1):
                res += helper(idx + 1, curr + num)
            return res % (10 ** 9 + 7)
        return helper(0, 0)
class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        MOD = 10 ** 9 + 7

        @lru_cache(None)
        def dp(dice, target):
            if dice == 0:
                return int(target == 0)
            elif target <= 0: # and dice > 0
                return 0
            else: # target > 0 and dice > 0
                res = 0
                for x in range(1, f + 1):
                    res += dp(dice - 1, target - x)
                return res % MOD

        return dp(d, target)
class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
            @lru_cache(maxsize=None)
            def nrolls(d, t):
                if d == 0:  
                    if t == 0:
                        return 1    
                    else:
                        return 0
                if d > t or t < 0: 
                    return 0
                res = 0
                for i in range(1, f+1):
                    res += nrolls(d-1, t-i)
                return res % ((10 ** 9) + 7)
            return nrolls(d, target) 
class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        
        if target > d*f:
            return 0
        dicti = collections.defaultdict(int)
        def dice_target(rem_dice, summ):
            if rem_dice == 0:
                return 1 if summ == target else 0
            if summ > target:
                return 0
            if (rem_dice, summ) in dicti:
                return dicti[rem_dice, summ]

            for i in range(1, f+1):
                dicti[rem_dice, summ] += dice_target(rem_dice-1, summ+i)
            return dicti[rem_dice, summ]
        
        
        return dice_target(d, 0) % (10**9 + 7)


class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        if d * f < target:
            return 0
        elif d * f == target:
            return 1
        
        dp = [[0] * (target+1) for _ in range(d)] # dp[i][j]: solution for target j with i + 1 dices
        BASE = 10**9 + 7
        for j in range(1, min(f, target)+1):
            dp[0][j] = 1
        for i in range(1, d):
            for j in range(target+1):
                for k in range(1, f+1):
                    if k >= j: break
                    dp[i][j] += dp[i-1][j-k] % BASE
        
        return dp[d-1][target] % BASE
class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        
        if target > d*f:
            return 0
        
        dicti = collections.defaultdict(int)
        def dice_target(rem_dice, summ):
            if rem_dice == 0:
                return 1 if summ == target else 0
            if summ > target:
                return 0
            if (rem_dice, summ) in dicti:
                return dicti[rem_dice, summ]

            for i in range(1, f+1):
                dicti[rem_dice, summ] += dice_target(rem_dice-1, summ+i)
            return dicti[rem_dice, summ]
        
        
        return dice_target(d, 0) % (10**9 + 7)


class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        dp = [0]*(target+1)
        
        for i in range(1,f+1):
            if i<=target:
                dp[i] = 1
        temp = dp[:]
        for i in range(1,d):
            dp = [0]*(target+1)

            for t in range(1, target+1):              
                if temp[t]>0:
                    for k in range(1,f+1):
                        if t+k <= target:
                            dp[t+k]+= temp[t]%(10**9 + 7) 
           
            for t in range(1,target+1):
                dp[t]=dp[t]  %(10**9 + 7) 
            temp = dp[:]
    #    print(929256393%(10**9 + 7) )
        return dp[target]%(10**9 + 7) 

class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        if d * f < target or d > target:
            return 0
    
        if d == 1 and f >= target:
            return 1
        
        rows = d+1
        cols = target+1
        dp = [[0 for _ in range(cols)] for _ in range(rows)]
            
        for i in range(1,min(f+1, cols)):
            dp[1][i] = 1

            
        for i in range(2, rows):
            for j in range(i, cols):
                
                start = max(1, j - f)
                dp[i][j] = sum(dp[i-1][start:j]) % 1000000007
                
                
        return dp[d][target]
                

class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        # 2. Bottom-up Dynamic Programming
        dp = [[0 for _ in range(target + 1)] for _ in range(d+1)]
        dp[0][0] = 1
        for i in range(1, d + 1):
            for j in range(i*1, min(target+1, i*f + 1)):
                for r in range(1, f+1):
                    if j - r >= 0:
                        dp[i][j] += dp[i-1][j-r]
        return dp[-1][-1] % (10**9 + 7)
                
                
            
#             1 dice -> 1 , f
#             2 dices -> 2, 2f

   
        
        # 1. Recursion: O(F^D) Time, O(F^D) Space
        self.res = 0
        def dfs(dice, sum):
            for i in range(1, f+1):
                if dice == d:
                    if sum + i == target:
                        self.res += 1
                elif sum + i < target:
                    dfs(dice + 1, sum + i)
        dfs(1, 0)
        return self.res

class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        
        @lru_cache(None)
        def dfs(rsum,d):
            
            if d==0 and rsum==0:
                return 1
            
            elif rsum==0 or d==0:
                return 0
            
            res=0
            for i in range(1,f+1):
                if rsum-i>=0:
                    res+=dfs(rsum-i,d-1)
                    
            return res
        
        
        return dfs(target,d)%(10**9+7)

class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        dp = [[0 for j in range(target + 1)] for i in range(d + 1)]
        for dd in range(1, d + 1):
            for tt in range(dd, min(dd * f, target) + 1):
                if dd == 1:
                    dp[dd][tt] = 1
                else:
                    for i in range(1, f + 1):
                        if tt - i >= 1:
                            dp[dd][tt] += dp[dd - 1][tt - i]
        return dp[dd][target] % (10**9 + 7)
class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        # 2. Bottom-up Dynamic Programming
        #   DP Array where target is the column and numDices is the row
        
        #   Range of values:
        #   1 dice -> (1 , f)
        #   2 dices -> (2, 2f)
        #   d dices -> (d, fd)
        dp = [[0 for _ in range(target + 1)] for _ in range(d+1)]
        dp[0][0] = 1
        for i in range(1, d + 1):
            for j in range(i*1, min(target+1, i*f + 1)):
                for r in range(1, f+1):
                    if j - r >= 0:
                        dp[i][j] += dp[i-1][j-r]
        return dp[-1][-1] % (10**9 + 7)

    
        # 1. Recursion: O(F^D) Time, O(F^D) Space
        self.res = 0
        def dfs(dice, sum):
            for i in range(1, f+1):
                if dice == d:
                    if sum + i == target:
                        self.res += 1
                elif sum + i < target:
                    dfs(dice + 1, sum + i)
        dfs(1, 0)
        return self.res

class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        if not d or not f or target == 0:
            return 0
            
        def dfs(d, f, hash_map, target):
            
            
            if target == 0 and d == 0:
                return 1
            if target<=0 or target>d*f or target<d:
                return 0
            
            # logic
            if (d,target) in hash_map:
                return hash_map[(d,target)]
            
            cnt = 0
            for i in range(1, f+1):
                if target-i<0:
                    break
                cnt += dfs(d-1, f,hash_map, target-i)
                cnt %= 1000000007

            hash_map[(d,target)] = cnt
            return hash_map[(d,target)]
        
        return dfs(d,f,{},target)
    

  
    

        

class Solution:
    # Recursive memoized solution
#     def numRollsToTarget(self, d: int, f: int, target: int) -> int:
#         memo = {}
#         def num_rolls_util(level, target):
#             if level * f < target or target < level:
#                 return 0
#             if level == 0:
#                 return 1
            
#             res = 0
#             for i in range(max(0, target - f), target):
#                 if (level-1, i) in memo:
#                     res += memo[(level-1, i)]
#                 else:
#                     tmp = num_rolls_util(level - 1, i)
#                     memo[(level-1, i)] = tmp
#                     res += tmp

#             return res % (10 ** 9 + 7)
        
#         return num_rolls_util(d, target)
    
    
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        mod = 10 ** 9 + 7
        dp = [[0] * (target + 1) for _ in range(d+1)]
        dp[0][0] = 1
        
        for i in range(1, d+1):
            for j in range(1, min(target, i * f) + 1):
                for k in range(1, min(j, f) + 1):
                    dp[i][j] += dp[i-1][j-k] % mod
        
        return dp[d][target] % mod        
class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        if d <= target <= d*f:
            mod = 10**9 + 7
            def rec(d, f, t):
                
                if d == 1:
                    return 1 if 0 < t <= f else 0
                elif (d, t) in memo:
                    return memo[(d, t)]
                else:
                    temp = sum([rec(d-1, f, t-x) for x in range(1, f+1)])
                    memo[(d, t)] = temp
                    return temp
            memo = {}
            return rec(d, f, target) % mod
        return 0
'''
1155. Number of Dice Rolls With Target Sum.  Medium

You have d dice, and each die has f faces numbered 1, 2, ..., f.

Return the number of possible ways (out of fd total ways) modulo 10^9 + 7 
to roll the dice so the sum of the face up numbers equals target.

Example 1:
Input: d = 1, f = 6, target = 3
Output: 1
Explanation: 
You throw one die with 6 faces.  There is only one way to get a sum of 3.

Example 2:
Input: d = 2, f = 6, target = 7
Output: 6
Explanation: 
You throw two dice, each with 6 faces.  There are 6 ways to get a sum of 7:
1+6, 2+5, 3+4, 4+3, 5+2, 6+1.

Example 3:
Input: d = 2, f = 5, target = 10
Output: 1
Explanation: 
You throw two dice, each with 5 faces.  There is only one way to get a sum of 10: 5+5.

Example 4:
Input: d = 1, f = 2, target = 3
Output: 0
Explanation: 
You throw one die with 2 faces.  There is no way to get a sum of 3.

Example 5:
Input: d = 30, f = 30, target = 500
Output: 222616187
Explanation: 
The answer must be returned modulo 10^9 + 7.

Constraints:
1 <= d, f <= 30
1 <= target <= 1000

Accepted
49,213
Submissions
101,895
'''
class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        if d <= target <= d*f:
            mod = 10**9 + 7
            def rec(d, f, t):
                if d == 1:
                    return 1 if 0 < t <= f else 0
                elif (d, t) in memo:
                    return memo[(d, t)]
                else:
                    temp = sum([rec(d-1, f, t-x) for x in range(1, f+1)])
                    memo[(d, t)] = temp
                    return temp
            memo = {}
            return rec(d, f, target) % mod
        return 0
class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        dp = {}
        
        for i in range(1, f+1):
            dp[i] = 1
            
        for i in range(d-1):
            temp = {}
            for val in dp:
                for dice in range(1, f+1):
                    if val + dice <= target:
                        temp[val+dice] = temp.get(val+dice, 0) + dp[val]
            dp = temp
                
        if not target in dp:
            return 0
        return dp[target]  % (10**9 + 7)
                    

class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        # 2. Bottom-up Dynamic Programming
        #   DP Array where target is the column and numDices is the row
        
        #   Range of values:
        #   1 dice -> (1 , f)
        #   2 dices -> (2, 2f)
        #   d dices -> (d, fd)
        dp = [[0 for _ in range(target + 1)] for _ in range(d+1)]
        dp[0][0] = 1
        for n in range(1, d + 1):
            for i in range(1*n, min(target+1, f*n + 1)):
                for j in range(1, f+1):
                    if i - j >= 0:
                        dp[n][i] += dp[n-1][i - j]
        return dp[-1][-1] % (10**9+7)

    
        # 1. Recursion: O(F^D) Time, O(F^D) Space
        self.res = 0
        def dfs(dice, sum):
            for i in range(1, f+1):
                if dice == d:
                    if sum + i == target:
                        self.res += 1
                elif sum + i < target:
                    dfs(dice + 1, sum + i)
        dfs(1, 0)
        return self.res

class Solution:
    def __init__(self):
        self.mem = {}
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        if target < d or target > d*f:
            return 0
        if d == 0 or target == 0:
            return 0
        if d==1 and target <= f:
            return 1
        key = (d,f,target)
        if key in self.mem:
            return self.mem[key]
        Sum = 0
        for i in range(1,f+1):
            Sum += self.numRollsToTarget(d-1, f, target-i)
            Sum %= (10**9+7)
        self.mem[key] = Sum
        return (self.mem[key])
        

class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        @lru_cache(None)
        def solve(s,t):
            if s <= 0:
                if t == 0:return 1
                return 0
            ans = 0
            for i in range(1,f+1):
                if t >= i:
                    ans += solve(s - 1,t - i)
            return ans
        
        return solve(d,target) % (10**9+7)
class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        
        
        @lru_cache(None)
        def dfs(left, k):
            if left == k:
                return 1
            
            if left > k:
                return 0
            
            if k == 0:
                if left == 0:
                    return 1
                else:
                    return 0
            
            if left < 0 :
                return 0
            
            
            s = 0
            for i in range(1, f+1):
                if k - i < 0:
                    break
                s += dfs(left-1, k - i)
            
            return s % (10 ** 9 + 7)
        
        return dfs(d, target) % (10 ** 9 + 7)
class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        current_sums = {}
        current_sums[0] = 1
        new_sums = {}
        out = 0
        
        for die in range(1, d+1):
            for face in range(1, f+1):
                for summ in list(current_sums.keys()):
                    curr = summ+face
                    if curr == target and die == d:
                        out += current_sums[summ]
                    elif curr < target:
                        if curr in list(new_sums.keys()):
                            new_sums[curr] += current_sums[summ]
                        else:
                            new_sums[curr] = current_sums[summ] 
                    
            current_sums = new_sums
            # print(current_sums)
            new_sums = {}
        
        return out % (10**9 + 7)

        # need to go through from target backwards suubtracting face everytime and adding to possibilities
    
    
#         def helper(h, d, target):
#             # if target is too small or if it is out of range
#             if target <= 0 or target > (d * f):
#                 return 0
#             if d == 1:
#                 print(d,f,target)
#                 return 1        # no need to check if target is within reach; already done before
#             if (d, target) in h:
#                 return h[(d, target)]        # directly access from hash table
#             res = 0
#             for i in range(1, f + 1):
#                 res += helper(h, d - 1, target - i)       # check all possible combinations
#             h[(d, target)] = res
#             return h[(d, target)]
        
#         h = {}
#         return helper(h, d, target) % (10 ** 9 + 7)

class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        @lru_cache(maxsize=None)
        def func(d, target):
            if d == 0:
                return 1 if target==0 else 0
            if d>target or target<0:
                return 0
            return sum([func(d-1, target-i)%(1e9+7) for i in range(1, f+1)])%(1e9+7)
        
        if d<1 or f<1 or target<1:
            return 0
        return int(func(d, target))
class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        @lru_cache(None)
        def solve(s,t):
            if s <= 0 or t < 0:
                if t == 0:return 1
                return 0
            ans = 0
            for i in range(1,f+1):
                ans += solve(s - 1,t - i)
            return ans
        return solve(d,target) % (10**9+7)
from functools import lru_cache
MOD=10**9+7
class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        @lru_cache(None)
        def dp(i,k):
            if i == d:
                return k==0
            return sum(dp(i+1,k-face) for face in range(1,f+1))
        return dp(0, target) % MOD
class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        
        if d == 1:
            return 1 if f >=target else 0
        
        dp = [[0 for _ in range(target+1)] for _ in range(d+1)]
        dp[0][0] = 1
        
        for i in range(1, d+1):
            for j in range(i, min(i*f, target)+1):
                # to ensure j-k >=0 means, k <=j, u81f3u591au4e3aj,min(j, f)
                for k in range(1, min(j, f)+1):
                    dp[i][j] += dp[i-1][j-k]
        
        return dp[d][target] % (pow(10, 9)+7)
    
    def __numRollsToTarget(self, d: int, f: int, target: int) -> int:
        mem = {}
        return self.dp(d, f, target, mem) % (pow(10, 9) + 7)
    
    def dp(self, d, f, target, mem):
        if d==0:
            return 1 if target == 0 else 0
        
        if d*f < target:
            return 0
        
        if (d, target) in mem:
            return mem[(d, target)]
        
        res = 0
        for i in range(1, f+1):
            if target -i < 0:
                break
            res += self.dp(d-1, f, target-i, mem)
        
        mem[(d, target)] = res
        return res
