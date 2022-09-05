import math

class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        dp: List[int] = [0] * (n+1)
        candidates: List[int] = []
        for j in range(1, int(math.sqrt(n))+1):
            candidates.append(j*j)
        for i in range(n):
            if not dp[i]:
                for can in candidates:
                    if i + can < n:
                        dp[i+can] = 1
                    elif i + can == n:
                        return 1
        return dp[-1]
import math
class Solution:
    
    def winnerSquareGame(self, n: int) -> bool:
        @lru_cache(None)
        def canWin(n):
            bound = math.floor(math.sqrt(n))
            if bound * bound == n:
                return True
            for i in range(bound, 0, -1):
                if not canWin(n - i * i):
                    return True
            return False
        return canWin(n)
    
    # def winnerSquareGame(self, n: int) -> bool:
    #     result = [None] * (n + 1)
    #     def canWin(n):
    #         bound = math.ceil(math.sqrt(n))
    #         if bound * bound == n:
    #             result[n] = True
    #             return True
    #         if result[n] != None:
    #             return result[n]
    #         for i in range(1, bound):
    #             if not canWin(n - i * i):
    #                 result[n] = True
    #                 return True
    #         result[n] = False
    #         return False
    #     return canWin(n)

class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        dp = [False]*(n+1)
        for i in range(n+1):
            if dp[i]:
                continue
            for k in range(1, int(n**0.5)+1):
                if i+k*k <= n:
                    dp[i+k*k] = True
                else:
                    break
        return dp[n]
class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        
        table = [False] * (n + 1)
        
        for index in range(n + 1):
            table[index] = any(not table[index - (lose * lose)]
                               for lose in range(1, 1 + math.floor(math.sqrt(index))))
            
        return table[-1]
class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        dp = [False] * (max(n, 2)+1)
        dp[0] = False
        dp[1] = True
        dp[2] = False
        #dp[3] = True
        #dp[4] = 

        squares = [i**2 for i in range(1, floor(sqrt(n))+1)]
        
        for i in range(3, n+1):
            for square in squares:
                if i - square < 0:
                    break
                if dp[i - square] == False:
                    dp[i] = True
                    break
        #print(dp)
        return dp[n]

class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        # 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30
        # t f t t f t f t t f  t  f  t  t  f  t  f  t  t  f  t  f  t  t  t  t  t  t  t  t
        
        if n == 1:
            return True
        if n == 2:
            return False
        
        dp = [0] * n
        dp[0] = True
        dp[1] = False
        count = 2
        
        for i in range(2, n):
            if (i+1) == count ** 2:
                dp[i] = True
                count += 1
            else:
                if dp[i-1] == False:
                    dp[i] = True
                else:
                    cur = 0
                    for j in range(count - 1, 1, -1):
                        if dp[i - j ** 2] == False:
                            dp[i] = True
                            cur = 1
                            break
                    if cur == 0:
                        dp[i] = False
        
        return dp[n-1]

class Solution:
  def winnerSquareGame(self, n: int) -> bool:
    # dpi = true if exists dpj for j = 1 -> square i that dpj = false
    # else dpi = false
    
    wins = [False] * (n + 1)
    for i in range(1, n + 1):
      j = 1
      canWin = False
      while j * j <= i and not canWin:
        canWin = not wins[i - j * j]
        j += 1
      wins[i] = canWin
      
    return wins[n]
    

import math
class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        dp = [False for i in range(n+1)]
        dp[0] = True
        dp[1] = True
        for i in range(3, n+1):
            if math.sqrt(i) == int(math.sqrt(i)):
                dp[i] = True
                continue
            start = 1
            while(start*start < i):
                if not dp[i-start*start]:
                    dp[i] = True
                    break
                start += 1
        print(dp)
        return dp[n]
import math
class Solution:
    squares=[]
    dp={}
    def util(self,n):
        if n==0:
            return False
        if n in self.dp:
            return self.dp[n]
        
        for i in range(1,int(math.sqrt(n))+1):
          #  print("square we subtract is",self.squares[i],"in util of",n,"and checking if util of",n-self.squares[i],"is true or false")
            if not self.util(n-self.squares[i]):
        #        print("util of",n-self.squares[i],"turned out to be false,so util of",n,"is a win!")
                self.dp[n]=True
                return True
       # print("all paths from util of",n,"led to failure, and we lost GG")
        self.dp[n]=False
        return False
                
    def winnerSquareGame(self, n: int) -> bool:
        self.squares=[]
        for i in range(int(math.sqrt(n))+1):
            self.squares.append(i*i)
        return self.util(n)
class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        if n == 1 or n==3 or n==4 or n==6: return True
        if n==0 or n == 2 or n==5: return False
        
        dp = [0,1,0,1,1,0,1,0]
        i = 8
        while i<=n:
            j = 1
            add= False
            while j*j <= i:
                if dp[i-j*j]==0:
                    dp.append(1)
                    add = True
                    break
                j += 1
            if not add: dp.append(0)
            i += 1
        #print(dp)
        #print([i for i in range(n+1)])
        return dp[n]
            
            
            
            

class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        dp = [0 for _ in range(n+1)]
        for i in range(1, n+1):
            for j in range(1, i+1):
                if j*j > i:
                    break
                if dp[i-j*j] == 0:
                    dp[i] = 1
                    break
        return dp[-1]
class Solution:
    @lru_cache(None)
    def winnerSquareGame(self, n: int) -> bool:
        if n == 0:
            return False
        if n == 1:
            return True
        
        # note range is right-end non-inclusive
        xsqrt = int(n**0.5)
        for i in range(xsqrt, 0, -1):
            if not self.winnerSquareGame(n - i * i):
                return True
        
        return False
            

class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        dp, sqs = [False] * (n + 1), []
        i = 1
        while i * i <= n:
            sqs.append(i * i)
            i += 1
        for i in range(1, n + 1):
            for sq in sqs:
                if sq > i:
                    break
                dp[i] = dp[i] or not dp[i - sq]
                if dp[i]:
                    break
        return dp[n]

class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        # W(a) = n - W(b) if W(a) is a square number
        # dp[i]: for n = i, if first one can win or not
        # if there's a "k" that can make dp[i - k*k] == False, then the other guy lose, and by making dp[i] = True
        dp = [False]*(n+1)
        for i in range(1, n+1):
            k = 1
            while k*k <= i:
                if dp[i-k*k] == False:
                    dp[i] = True
                    break
                k += 1
        return dp[n]
class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        dp = [False] * (n+1)
        for i in range(1,n+1):
            j = 1
            while j*j <= i:
                if dp[i-j*j]==False:
                    dp[i] = True
                    break
                j+=1
        return dp[n]
from functools import lru_cache
class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        @lru_cache(maxsize=None)
        def win(cur):
            if cur == 0: return False            
            for i in range(1, int(cur**0.5)+1):
                if not win(cur - i*i):
                    return True                
            return False
        
        return win(n)

class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        dp = [False] * (160000)
        m = 400
        for i in range(1, m):
            dp[i*i] = True
        # print(dp[0:n])
        
        for i in range(1, n + 1):
            for j in range(1, m):
                if i > j * j:
                    if not dp[i - j*j]:
                        dp[i] = True
                        break
                    # dp[i] = max(-dp[i - j*j], dp[i])
        # print(dp[0:n])
        return dp[n]

class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        squares = [i**2 for i in range(1,int(sqrt(n)) + 1)]
        A = [False for _ in range(n + 1)]
        
        for i in range(1, n + 1):
            if i in squares: 
                A[i] = True
                continue
                
            for square in squares:
                if square > i:
                    break
        
                if not A[i - square]:
                    A[i] = True
                    break
        
        return A[n]

class Solution:
    def winnerSquareGame(self, n: int) -> bool:

        @lru_cache(maxsize=None)
        def dfs(remain):
            if remain == 0:
                return False

            sqrt_root = int(remain**0.5)
            for i in range(1, sqrt_root+1):
                # if there is any chance to make the opponent lose the game in the next round,
                #  then the current player will win.
                if not dfs(remain - i*i):
                    return True

            return False

        return dfs(n)
class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        
        arr = [0] * (n+1)
        
        arr[0] = False
        
        for i in range(1, n+1):
            
            for j in range(1, n+1):
                square = j * j
                
                if square > i:
                    break
                elif square == i:
                    arr[i] = True
                    break
                else:
                    rest = i - square
                    if arr[rest] == False:
                        arr[i] = True
                        break
        
        return arr[n]

 
class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        @lru_cache(maxsize=None)
        def dfs(n):
            if n == 0:
                return False
            square_root = int(sqrt(n))
            for i in range(1, square_root + 1):
                if not dfs(n - i * i):
                    return True
            return False                  
        return dfs(n)

class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        mem=[None]*(n+1)
        def dp(k):    #u6709ku4e2au77f3u5934u65f6uff0cu80fdu5426u8d62
            if k==0:
                return -1
            if mem[k]!=None:
                return mem[k]
            for i in range(int(math.sqrt(k)),0,-1):
                if dp(k-i*i)<0:    #u5982u679cu6211u53d6i*iu4e2au77f3u5934uff0cu5269u4e0bk-i*iu4e2au77f3u5934u65f6uff0cu6211u6700u540eu6ca1u77f3u5934u53d6u8f93u4e86uff0cu5219u4ee3u8868u6709k
                                   # u4e2au77f3u5934u80dcu5229u4e86uff0cu6211u53efu4ee5u53d6i*iu4e2au77f3u5934
                    mem[k]=1
                    return 1
            mem[k]=-1
            return -1
        return dp(n)>0
import math
class Solution:
    
    def winnerSquareGame(self, n: int) -> bool:
        # result = [None] * (n + 1)
        @lru_cache(None)
        def canWin(n):
            bound = math.ceil(math.sqrt(n))
            if bound * bound == n:
                # result[n] = True
                return True
            # if result[n] != None:
            #     return result[n]
            for i in range(1, bound):
                if not canWin(n - i * i):
                    # result[n] = True
                    return True
            # result[n] = False
            return False
        return canWin(n)
    
    # def winnerSquareGame(self, n: int) -> bool:
    #     result = [None] * (n + 1)
    #     def canWin(n):
    #         bound = math.ceil(math.sqrt(n))
    #         if bound * bound == n:
    #             result[n] = True
    #             return True
    #         if result[n] != None:
    #             return result[n]
    #         for i in range(1, bound):
    #             if not canWin(n - i * i):
    #                 result[n] = True
    #                 return True
    #         result[n] = False
    #         return False
    #     return canWin(n)

class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        @lru_cache(maxsize=None)
        def helper(i):
            if i == 0: return False
            sr = int(i**0.5)
            for k in range(1, sr+1):
                if not helper(i-k*k):
                    return True
            return False
        
        return helper(n)
import functools

class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        @functools.lru_cache(maxsize = None)
        def dfs(remaining: int) -> bool:
            if remaining == 0:
                return False
            else:
                sqroot: int = int(remaining ** 0.5)
                for i in range(1, sqroot + 1):
                    if dfs(remaining - i * i):
                        continue
                    else:
                        return True
                return False
        
        return dfs(n)
from collections import deque
class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        
        dp=[False]*(n+1)
        dp[1]=True
        
        
        for x in range(2,n+1):
            i=1
            while i*i<=x:
                if not dp[x-i*i]:
                    dp[x]=True
                    break
                i+=1
                    
        return dp[n]
        
        
        
            
            
            
        
        

class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        
        def getSquareNumbers(n: int) -> List[int]:
            # less than or equal to n
            return [index * index for index in range(1, 1 + math.floor(math.sqrt(n)))]
        
        table = [False] * (n + 1)
        
        for index in range(n + 1):
            table[index] = any(not table[index - (lose * lose)]
                               for lose in range(1, 1 + math.floor(math.sqrt(index))))
            
        return table[-1]
class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        # dp[i] if the player can win after i pile has been taken
        # dp[i] = one of dp[j1], dp[j2], is false ... where i - j_x is a square number
        
        dp = [False] * (n + 1)
        
        for i in reversed(range(n)):
            temp = 1
            while i + temp * temp <= n:
                j = i + temp * temp
                temp += 1
                if not dp[j]:
                    dp[i] = True
                    break
        
        return dp[0]
class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        #Lee code for speed comparison
        dp = [False] * (n + 1)
        for i in range(1, n + 1):
            dp[i] = not all(dp[i - k * k] for k in range(1, int(i**0.5) + 1))
        return dp[-1]

class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        
        @lru_cache(maxsize=None)
        def dfs(remain):
            if remain == 0:
                return False

            sqrt_root = int(remain**0.5)
            for i in range(1, sqrt_root+1):
                # if there is any chance to make the opponent lose the game in the next round,
                #  then the current player will win.
                if not dfs(remain - i*i):
                    return True

            return False

        return dfs(n)

class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        # Solution 1: DP by myself (9532ms: 5.06%)
        '''
        ele = [item**2 for item in range(1, int(math.sqrt(n))+1)]
        memo = {}
        def helper(amount, person):
            if amount == 0:
                memo[(amount, person)] = False
                return False
            if amount<0:
                return
            if (amount, person) in memo:
                return memo[(amount, person)]
            for item in ele:
                if item<=amount:
                    if not helper(amount-item, -person):
                        memo[(amount, person)] = True
                        return True
                else:
                    break
            memo[(amount, person)] = False
            return False
        return helper(n, 1)
        '''
        # Solution 2: TLE!!
        '''
        dp = [False]*(n+1)
        def check(n):
            if n==0:
                return False
            if n==1:
                dp[1] = True
                return True
            for i in range(int(math.sqrt(n)), 0, -1):
                if not check(n-i*i):
                    dp[n] = True
                    return True
            return False
        return check(n)
        '''
        # Solution 3: DP from discussion (2132ms: 44.64%)
        dp = [False] * (n + 1)
        for i in range(1, n+1):
            for j in range(1, int(i**0.5)+1):
                if not dp[i-j*j]:
                    dp[i] = True
                    break
        return dp[-1]
class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        
        @lru_cache(maxsize = None)
        def dfs(remain):
            
            sqrts = int(sqrt(remain))
            if sqrts ** 2 == remain:
                return True
            
            for i in range(1, sqrts + 1):
                
                if not dfs(remain - i*i):
                    return True
            return False
        
        return dfs(n)
def is_winning (n):
    winning = []
    for val in range (0, n+1):
        if val == 0:
            ans = False
        else:
            ans = False
            i = 1
            while val - i * i >= 0:
                if not winning[val - i * i]:
                    ans = True
                    break
                i += 1
        winning.append (ans)
    return winning[n]

class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        return is_winning(n)
import functools

class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        @functools.lru_cache(None)
        def can_force_win_from(m):
            if m == 0:
                return False
            if m == 1:
                return True

            for root in range(1, int(m ** 0.5) + 1):
                if not can_force_win_from(m - root * root):
                    return True
            return False
        return can_force_win_from(n)

class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        @lru_cache(maxsize=None)
        def dfs(remain):
            if remain == 0:
                return False
            sqrt_root = int(remain**0.5)
            for i in range(1, sqrt_root+1):
                if not dfs(remain - i*i):
                    return True
            return False
        return dfs(n)

class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        dp = [False for _ in range(n+1)]
        dp[1] = True
        for i in range(2, n+1):  # just to make it more clear
            k = 1
            while k*k <= i and not dp[i]:  # try to find at least one way to win
                dp[i] = not dp[i-k*k]
                k += 1
        return dp[-1]

class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        def is_square(num):
            a = int(num**0.5)
            return a * a == num
        dp = [False] * (n + 1)
        dp[0] = False
        dp[1] = True
        for i in range(2, n + 1):
            dp[i] = False
            if is_square(i):
                dp[i] = True
                continue
            limit = int(i ** 0.5)
            for j in range(1, limit + 1):
                if dp[i - j * j] == False:
                    dp[i] = True
                    break
        #print(dp)
        return dp[n]
class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        
        table = [False] * (n + 1)
        
        for index in range(n + 1):
            table[index] = not all(table[index - (lose * lose)]
                               for lose in range(1, 1 + math.floor(math.sqrt(index))))
            
        return table[-1]
class Solution:
    def winnerSquareGame(self, n: int) -> bool:

        @lru_cache(maxsize=None)
        def dfs(remain):
            sqrt_root = int(remain**0.5)
            if sqrt_root ** 2 == remain:
                return True

            for i in range(1, sqrt_root+1):
                # if there is any chance to make the opponent lose the game in the next round,
                #  then the current player will win.
                if not dfs(remain - i*i):
                    return True

            return False

        return dfs(n)

class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        
        alice = [False]*(n+1)
        bob = [False]*(n+1)
                
        for i in range(1, n+1):
            for x in range(1, int(sqrt(i))+1):
                if not bob[i-x*x]:
                    alice[i] = True
                    break
            for x in range(1, int(sqrt(i))+1):
                if not alice[i-x*x]:
                    bob[i] = True
                    break
            
        return alice[n]

class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        res = [0] * (n + 1)
        res[1] = 1
        for i in range(2, n + 1):
            for j in range(1, int(math.sqrt(i)) + 1):
                if not res[i-j*j]:
                    res[i] = 1
                    break
        print(res)
        return res[-1]
class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        
        @lru_cache(maxsize=None)
        def dfs(remain):
            sqrt_root = int(sqrt(remain))
            
            if sqrt_root ** 2 == remain:
                return True
            
            for i in range(1, sqrt_root+1):
                if not dfs(remain - i*i):
                    return True
    
            return False
    
        
        return dfs(n)

class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        dp = [False for _ in range(n+1)]
        for i in range(1, n+1):
            k = 1
            while k*k <= i and not dp[i]:
                dp[i] = not dp[i-k*k]
                k += 1
        return dp[-1]

class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        @lru_cache(None)
        def dfs(idx):
            if idx == 0:
                return False
            
            for i in range(1,int(math.sqrt(idx))+1):
               if dfs(idx - i*i) == False:
                    return True
            return False 
        return dfs(n)
class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        wins = [False for i in range(n+1)]
        wins[1] = True ## True u8868u793aaliceu8d62
        for i in range(2,n+1):
            j = 1
            while i - j*j >= 0:
                if wins[i-j*j] == False:
                    wins[i] = True
                    break
                j += 1
        return wins[n]

class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        dp = [False for _ in range(n+1)]
        i = 0
        while i*i <= n:
            dp[i*i] = True  # next round is Alice, she wins immediately
            i += 1
        for i in range(1, n+1):
            k = 1
            while k*k <= i and not dp[i]:
                dp[i] = not dp[i-k*k]
                k += 1
        return dp[-1]

class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        

        dp = [False for i in range(n + 1)]

        for num in range(1, n + 1):
            for j in range(1, num + 1):
                if j * j > num:
                    break
                dp[num] = dp[num] or not dp[num - j * j]
                if(dp[num]):
                    break

        print(dp)
        return dp[n]
class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        # u6bcfu6b21u79fbu9664u5e73u65b9u6570
        # aliceu6bcfu6b21u79fbu9664u5b8cu4e0du80fdu5269u4f59u5e73u65b9u6570uff0cu5426u5219u5c31u8f93
        # u5f53u524du6570u5b57u51cfu53bbu4e00u4e2au5e73u65b9u6570u540euff0cu4e0du80fdu662fu5e73u65b9u6570
        # u4eceu5c0fu5230u5927u8ba1u7b97uff0cn=1uff0cu3002u3002u3002uff0cn
        # dp[i]: u6709iu4e2au77f3u5934u65f6uff0caliceu80fdu4e0du80fdu8d62
        # dp[i] = u4eceiu5411u4e0bu51cfu53bbu5e73u65b9u6570uff0cu5bf9u5e94u7684dp[j]u6709u4e00u4e2au662ffalseu5373u53ef
        dp = [False for _ in range(n+1)]
        for i in range(1,n+1):
            base = 1
            while i-base**2>=0:
                if not dp[i-base**2]:
                    dp[i] = True
                    break
                else:
                    base += 1
            if i<n and ((n-i)**0.5)%1==0 and not dp[i]:
                return True                
        return dp[n]

class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        
        dp = [False]*(n+1)
        dp[1] = True
        for i in range(1,n+1):
            for j in range(1,i):
                if j*j > i: break
                elif not dp[i-j*j]:
                    dp[i] = True
                    break
        return dp[n]

import math

class Solution:
    @lru_cache(None)
    def winnerSquareGame(self, n: int) -> bool:
        if n == 0:
            return False
        i = int(math.sqrt(n))
        while i >= 1:
            if self.winnerSquareGame(n-i*i) == False:
                return True
            i -= 1
        return False
class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        res = [False] * (n+1)
        
        for i in range(1, n+1):
            j = 1
            while j * j <= i:
                res[i] |= not res[i-j*j]
                if res[i]: break
                j += 1
        return res[n]

class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        from functools import lru_cache
        @lru_cache(None)
        def dp(i):
            if i == 0:
                return False
            if i == 1:
                return True
            for k in range(1, i+1):
                if k * k <= i:
                    if not dp(i-k*k):
                        return True
                else:
                    break
            return False
        return dp(n)
class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        @lru_cache(None)
        def dfs(n: int) -> bool:
            if n == 0:
                return False
            if n == 1:
                return True
            k = 1
            while (k * k) <= n:
                if not dfs(n - (k*k)):
                    return True
                k += 1
            return False
        return dfs(n)
class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        dp = [False] * (160000)
        m = 400
        for i in range(1, m):
            dp[i*i] = True
        # print(dp[0:n])
        
        for i in range(1, n + 1):
            for j in range(1, m):
                if i > j * j:
                    if not dp[i - j*j]:
                        dp[i] = True
                        break

                else:
                    break
        return dp[n]

class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        @lru_cache(None)
        def win(n):
            if n == 1:
                return True
            if n == 0:
                return False
            v = 1
            while v*v <= n:
                if not win(n - v*v):
                    return True
                v += 1
            return False
        return win(n)
from functools import lru_cache

class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        
        @lru_cache(None)
        def dp(m):
            if m == 0:
                return True
            if int(math.sqrt(m)) ** 2 == m:
                return True
            
            i = 1
            while i*i <= m:
                if not dp(m - i*i):
                    return True
                i += 1
                
            return False
        
        return dp(n)
class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        # cost = [-1 for _ in range(n+1)]
        
        @lru_cache(None)
        def helper(n):
            if n == 0:
                return 0
            i = 1
            sq = 1
            while sq <= n:
                if helper(n-sq) == 0:
                    return 1
                i += 1
                sq = i*i
            return 0
        
        return helper(n)

class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        dp = [False]
        for i in range(1, n + 1):
            s = False
            for j in range(1, int(i ** 0.5) + 1):
                if not dp[i - j ** 2]:
                    s = True
                    break
            dp.append(s)
        return dp[-1]
import math
class Solution:
    
    def winnerSquareGame(self, n: int) -> bool:
        dp = [False] * (n + 1)
        i = 1
        for i in range(1, n + 1):
            dp[i] = not all(dp[i - j*j] for j in range(1, int(math.sqrt(i)) + 1))
        return dp[-1]
class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        # time complexity: O(N*logN)
        # space complexity: O(N)
        dp = [False]
        for i in range(1, n + 1):
            s = False
            for j in range(1, int(i ** 0.5) + 1):
                if not dp[i - j ** 2]:
                    s = True
                    break
            dp.append(s)
        return dp[-1]
class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        dp = [False for _ in range(n+1)]
        for i in range(1, n+1):
            for e in range(1, int(i ** 0.5)+1):
                if not dp[i - e ** 2]:
                    dp[i] = True
                    break
        return dp[-1]
class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        
        dp = [False]*(n+1)
        dp[1] = True
        for i in range(1, n+1):
            x = 1
            while x * x <= i:
                
                if dp[i-x*x] == False:
                    dp[i] = True
                    break
                
                x+=1
                
        return dp[n]
            

import math
class Solution:
    
    def winnerSquareGame(self, n: int) -> bool:
        dp = [False] * (n + 1)
        i = 1
        for i in range(1, n + 1):
            dp[i] = not all(dp[i - j*j] for j in range(1, int(math.sqrt(i)) + 1))
        print(dp)
        return dp[-1]
from math import floor, sqrt
class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        dp = [0 for _ in range(n+1)]
        for i in range(1, floor(sqrt(n))+1):
            dp[i*i] = 1
            great = 1
        for i in range(2, n+1):
            if dp[i] == 0:
                for j in range(1, great+1):
                    res = (dp[j*j] +dp[i-j*j]) % 2
                    # print(i, j, res)
                    if res == 1:
                        dp[i] = 1
                        break
            else:
                great +=1
        print(dp)
        return True if dp[n] else False
class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        losers = {0}
        for x in range(1,n+1):
            flag = True
            for y in range(1,int(x**0.5)+1):
                if (x - y**2) in losers:
                    flag = False
                    break
            if flag: #Its a loser position because you cant send the next player to a loser position
                losers.add(x)
        return False if n in losers else True

class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        dp = [False]
        for i in range(1, n+1):
            for b in range(1, math.floor(math.sqrt(i))+1):
                if not dp[i-b**2]:
                    dp.append(True)
                    break
            else:
                dp.append(False)
        return dp[-1]
class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        sqrs = []
        for j in range(1, n + 1):
            if j * j <= n:
                sqrs.append(j * j)
            else:
                break
        dp = [False] * (1 + n)
        for s in sqrs: dp[s] = True
        
        for i in range(1, n + 1):
            if dp[i]: continue
            for s in sqrs:
                if s > i: break
                dp[i] = not dp[i - s]
                if dp[i]: break
        return dp[-1]
        

class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        dp = [False]*(n+1)
        dp[1] = True
        for i in range(2,n+1):
            j = 1
            flag = False
            while j*j <= i:
                if not dp[i-j*j]:
                    flag = True
                    break
                j += 1
            dp[i] = flag
                    
        return dp[-1]
import math

class Solution:
    
    cache = {}
        
    
    def winnerSquareGame(self, n: int) -> bool:
        if n in self.cache:
            return self.cache[n]
        for i in range(1, int(math.sqrt(n)) + 1):
            if n - i*i == 0:
                self.cache[n] = True
                return True
            if not self.winnerSquareGame(n - i*i):
                self.cache[n] = True
                return True
        self.cache[n] = False
        return False

from functools import lru_cache
class Solution:
    def winnerSquareGame(self, n: int) -> bool:

        @lru_cache(None)
        def dp(k):
            if k == 1:
                return True
            if k == 0:
                return False
            for i in range(1, k):
                if i * i > k:
                    break
                if not dp(k - i * i):
                    return True
            return False
        
        return dp(n)
class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        dp = [None] * (n + 1)
        squares = []
        for i in range(1, n+1):
            if i ** 2 > n:
                break
            squares.append(i**2)
            dp[i**2] = True
        for i in range(1, n + 1):
            if dp[i] is not None:
                continue
            cur = True
            for s in squares:
                if s > i:
                    break
                cur = cur and dp[i - s]
            dp[i] = not cur
        return dp[n]
class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        l=[1]
        pos=[0]*(n+1)
        
        for i in range(1,n+1):
            if i==(math.sqrt(l[-1])+1)**2:
                l.append(int((math.sqrt(l[-1])+1)**2))
            for square_num in l:
                if not pos[i-square_num]:
                    pos[i]=1
        return pos[n]
class Solution:
    #   1(T)   2(F)   3(T)   4(T)   5   6   7
    def winnerSquareGame(self, n: int) -> bool:
        mem = {}
        squares = []
        if int(pow(n,0.5)) == pow(n,0.5):
            return True
        def helper(n):
            if n in mem:
                return mem[n]
            for i in range(1,int(pow(n,0.5))+1):
                if not helper(n-i*i):  # try that move and won
                    mem[n] = True
                    return True
            mem[n] = False
            return False
        return helper(n)
    
    def calcSquares(self,mem,squares,n):
        for i in range(1,n):
            i = pow(i,0.5)
            if int(i) == i:
                squares.append(i)
                mem[i] = True
class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        res = [False] * (n+1)
        for i in range(n+1):
            if res[i]: continue
            j = 1
            while i + j * j <= n:
                res[i + j*j] = True
                j += 1
        return res[n]

class Solution:
    #   1(T)   2(F)   3(T)   4(T)   5   6   7
    def winnerSquareGame(self, n: int) -> bool:
        mem = {}
        if int(pow(n,0.5)) == pow(n,0.5):
            return True
        def helper(n):
            if n in mem:
                return mem[n]
            for i in range(1,int(pow(n,0.5))+1):
                if not helper(n-i*i):  # try that move and won
                    mem[n] = True
                    return True
            mem[n] = False
            return False
        return helper(n)
class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        s = [i**2 for i in range(1, int(n**(1/2))+1)]
        record = {}
        
        def helper(i):
            if i in record:
                return record[i]
            if i == 0:
                return False
            for j in s:
                if j > i:
                    record[i] = False
                    return False
                if not helper(i-j):
                    record[i] = True
                    return True
        
        res = helper(n)
        #print(record)
        return res
class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        sol = [False] * (n + 1)
        
        for i in range(1, n+1):
            j = pwr = 1
            sol[i] = False 
            while pwr <= i:
                if not sol[i-pwr]:
                    sol[i] = True
                    break
                j+=1
                pwr = j**2
                
        return sol[-1]
            

class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        ans = [False]*(n+1)
        for i in range(1, n+1):
            ans[i] = not all(ans[i-j**2] for j in range(1, 1+int(i**.5)))
        return ans[-1]

class Solution:
    
    def winnerSquareGame2(self, n):
        dp = [False] * (n + 1)
        for i in range(1, n + 1):
            dp[i] = not all(dp[i - k * k] for k in range(1, int(i**0.5) + 1))
        return dp[-1]
    
    def winnerSquareGame(self, n: int) -> bool:
        # u5c1du8bd5u540cu6837u7684u89e3u6cd5uff1f
        # solve(s), s=u5f53u524du7684u5269u4f59u7684u6570u5b57u4e0buff0cscore diff? u9700u8981u7684u6700u5c0fu6b65u6570uff1f
        # u5982u679cs=0uff0cu5219u8fd4u56deFalse
        # u5982u679cs=u4efbu610fu4e00u4e2au5e73u65b9u6570uff0cu5219return True, u56e0u4e3aaliceu53efu4ee5u90fdu62ffu8d70
        # u5426u5219uff0caliceu53efu4ee5u4eceu4e2du5207u8d70u4efbu610fu5927u5c0fu7684u5e73u65b9u6570xuff0cu7136u540eu628au5269u4e0bu7684s-xu6254u7ed9bob
        
        # u53eau8981u5176u4e2du6709u4e00u4e2au5207u5272uff0cbobu65e0u6cd5u5728s-xu4e0bu83b7u80dcuff0cu90a3u5c31u662faliceu83b7u80dc
        # u5982u679cbobu5728u6240u6709u5207u5272u4e0bu90fdu83b7u80dcuff0cu90a3alice lose
        
        # u53eau4ecealiceu7684u89d2u5ea6u51fau53d1uff0cu662fu5426u8db3u591fuff1f
        cache = dict()
        
        def solve(s):
            if s in cache: return cache[s]
            if s == 0: 
                cache[s] = False
                return False
            
            if pow(int(sqrt(s)), 2) == s: 
                cache[s] = True
                return True # s is a square number and current player can take it directly, so win
            
            iswin = False
            #for x in range(s-1, 0, -1): # from 1 to s-1, since s is not a square number
            #    if pow(int(sqrt(x)), 2) == x:
            #        if not solve(s-x):
            #            iswin = True
            #            break
            for k in range(1, int(sqrt(s))+1):
                if not solve(s - k*k):
                    iswin = True
                    break
                
            cache[s] = iswin
            return iswin
        return solve(n) # u65b9u6cd5u662fu5bf9u7684uff0cu4f46u662fu8d85u65f6u4e86uff0cn=31250u7684u65f6u5019

def canWin(n,squares,memo):
    
    if n in squares:
        #print(n,True)
        return True
    
    if n in memo:
        return memo[n]
    
    res = False
    for i in reversed(squares):
        if i>n: continue
        #if n==13: print('here',n-i)
        if not canWin(n-i,squares,memo):
            res = True
            break
            
    memo[n] = res
    return res
    

class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        squares = [i**2 for i in range(1,floor(sqrt(n))+1)]
        memo = dict()
        #print(squares)
        return canWin(n,squares,memo)
        

class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        return self.dfs(n)
    
    @lru_cache(None)
    def dfs(self, remain):
        sqrt_root = int(math.sqrt(remain))
        for i in range(1, sqrt_root+1):
            if not self.dfs(remain - i*i):
                return True

        return False

        
    '''
        return self.helper(n, True)
    
    def helper(self, n, label):
        value = math.sqrt(n)
        if value == int(value):
            return label
        re = False
        
        for i in range(n,0, -1):
            ii = math.sqrt(i)
            if ii == int(ii):
                print(ii, label)
                re = self.helper(n-int(ii*ii), not label)
        return re 
    '''
class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        
        def dp_search(dp, n):
            if n == 0:
                return False
            if n == 1:
                return True
            
            if n in dp:
                return dp[n]
            
            dp[n] = False
            i = int(sqrt(n))
            while i>=1:
                if dp_search(dp, n-i*i)==False:
                    dp[n] = True
                    return True
                i-=1
            
            return False
        
        
        dp = {}
        return dp_search(dp, n)

class Solution:
    @lru_cache(None)
    def winnerSquareGame(self, n: int) -> bool:
        if n == 0:
            return False
        if n == 1:
            return True
        
        # note range is right-end non-inclusive
        for i in range(1, int(n**0.5) + 1):
            if not self.winnerSquareGame(n - i * i):
                return True
        
        return False
            

class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        dp = [False] * (n + 1)
        for i in range(1, n+1):
            dp[i] = any(not dp[i - j ** 2] for j in range(1, int(i ** 0.5) + 1))
        return dp[-1]
class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        choices = [1]
        memo = {}
        for i in range(2,n):
            if i*i > n:
                break
            choices.append(i*i)
                
        def find(n):
            if n == 0:
                return False
            if n in memo:
                return memo[n]
            for i in choices:
                if i > n:
                    break
                if not find(n-i):
                    memo[n] = True
                    return True
            memo[n] = False
            return False
        return find(n)

def canWin(n,squares,memo):
    
    if n in squares:
        #print(n,True)
        return True
    
    if n in memo:
        return memo[n]
    
    res = False
    for i in reversed(squares):
        if i>n: continue
        #if n==13: print('here',n-i)
        if not canWin(n-i,squares,memo):
            res = True
            break
            
    memo[n] = res
    return res
    

class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        squares = [i**2 for i in range(1,floor(sqrt(n))+1)]
        memo = dict()
        print(squares)
        return canWin(n,squares,memo)
        

class Solution:
    @lru_cache(None)    
    def winnerSquareGame(self, n: int) -> bool:
        for i in range(1, int(sqrt(n))+1):
            if not self.winnerSquareGame(n-i*i):
                return True
        return False
class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        memo = {0: False}
        def dp(i):
            if i in memo: return memo[i]
            m = int(i ** 0.5)
            if i ** 0.5 == m: return True
            res = False
            for j in range(1, m + 1):
                res |= not dp(i - j * j)
                if res: 
                    memo[i] = True
                    return True
            memo[i] = False
            return False
        
        return dp(n)
            

class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        mem={}
        def game(n):
            if n in mem:
                return mem[(n)]
            if n==0:
                return False
            k=1
            mem[n]=False
            while k*k<=n:
                if not game(n-k*k):
                    mem[n]=True
                    break
                k+=1
            return mem[n]
        
        game(n)
        return mem[n]

class Solution:
    @lru_cache(None)
    def winnerSquareGame(self, n: int) -> bool:
        if n == 0:
            return False
        if n == 1:
            return True
        
        # note range is right-end non-inclusive
        xsqrt = int(n**0.5) + 1
        for i in range(1, xsqrt):
            if not self.winnerSquareGame(n - i * i):
                return True
        
        return False
            

class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        cost = [-1 for _ in range(n+1)]
        
        @lru_cache(None)
        def helper(n):
            if n == 0:
                return 0
            if cost[n] != -1:
                return cost[n]
            i = 1
            sq = 1
            while sq <= n:
                if cost[n-sq] != -1:
                    return cost[n-sq]
                if helper(n-sq) == 0:
                    cost[n-sq] = 1
                    return 1
                i += 1
                sq = i*i
            cost[n] = 0
            return 0
        
        return helper(n)

class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        @lru_cache(None)
        def win(amt):
            x = int(math.sqrt(amt))
            for i in range(x, 0, -1):
                if not win(amt - i*i):
                    return True
            return False
        return win(n)
class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        
        self.memo = dict()
        return self.dfs(n)

        
    def dfs(self, stonesRemain):
        if stonesRemain <= 0:
            return False
        
        if stonesRemain in self.memo:
            return self.memo[stonesRemain]
        
        squareRoot = int(stonesRemain ** 0.5)
        res = False
        for i in reversed(range(1, squareRoot + 1)):
            if not self.dfs(stonesRemain - i * i):
                res = True
                break
        
        self.memo[stonesRemain] = res
        return res
class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        x = 1
        s = []
        st = set()
        dp = [False] * (n+1)
        while x * x <= n:
            s.append(x * x)
            st.add(x * x)
            dp[x * x] = True
            x += 1
        if n in s:
            return True
        for i in range(1, n+1):
            if dp[i] == False:
                start = 0
                while start < len(s) and i - s[start] > 0:
                    if dp[i - s[start]] == False:
                        dp[i] = True
                        break
                    start += 1
        return dp[n]
                    

class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        cache = {}
        def helper(n):
            if n in cache:
                return cache[n]
            s = sqrt(n)
            if s.is_integer():
                cache[n] = True
                return True
            i = 1
            while i<s:
                j = i*i
                if not helper(n-j):
                    cache[n] = True
                    return True
                i += 1
            cache[n] = False
            return False
        return helper(n)
class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        dp = [False] * (n + 1)
        dp[1] = True
        for i in range(2, n + 1):
            dp[i] = not all(dp[i - j * j] for j in range(1, int(math.sqrt(i)) + 1) if i >= j * j)
        return dp[-1]
import math
from collections import defaultdict

class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        def dfs(position, player):
            if position in cache:
                if cache[position] == True:
                    return player
                return not player

            for sq in squares:
                if sq == position:
                    cache[position] = player
                    return player
                if sq > position:
                    break

                if player == dfs(position - sq, not player):
                    cache[position] = player
                    return player

            cache[position] = not player
            return not player

        cache = defaultdict(bool)
        max_val = int(math.sqrt(n))
        squares = [1]
        for i in range(2, max_val + 1):
            squares.append(i ** 2)
        cache[1] = True
        cache[2] = False
        cache[3] = True
        cache[4] = True
        for i in range(5, n+1):
            cache[i] = dfs(i, True)
        return cache[n]
import math
class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        dp = [False for _ in range(n+1)] # init dp[0] as False since it implies no move to make.
        dp[1] = True # known corner case
        for i in range(2,n+1): # for every i in [2,n]
            sqr = int(i**0.5) # calculate upper bound for integer square root less than i
            for j in range(1, sqr+1): # for every integer square root less than sqrt(i)
                dp[i] |= not dp[i-j**2] # if there is any n == (i-j**2) that is doomed to lose, i should be true.
                                        # because Alice can make that move(remove j**2 stones) and make Bob lose.
                                        # otherwise i should be false since there is no any choice that will lead to winning.
                if dp[i]: # Optimization due to test case TLE: if it is already true, break out.
                    break
        return dp[n]
                
                
                
        
        

class Solution:
    @lru_cache(None)
    def winnerSquareGame(self, n: int) -> bool:
        if n <= 1:
            return n == 1
        
        for i in range(1, n+1):
            sq = i*i
            if sq > n:
                break
            
            if not self.winnerSquareGame(n-sq):
                return True
        
        return False
import math

class Solution:
    # A 
    # B 15 1 4 9
    def helper(self, n: int, dp: dict) -> bool:
        if n in dp:
            return dp[n]
        if n == 0:
            return False
        i = 1
        while i*i <= n:
            if (self.helper(n-i*i, dp) == False):
                dp[n] = True
                return True
            i += 1
        dp[n] = False
        return False
    
    def winnerSquareGame(self, n: int) -> bool:
        dp = {}
        return self.helper(n, dp)
class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        if n==1:
            return True
        dp=[0,1]
        for i in range(2,n+1):
            root=int(i**0.5)
            if root**2==i:
                dp.append(1)
            else:
                for j in range(1,root+1):
                    if not dp[i-j**2]:
                        dp.append(1)
                        break
            if len(dp)==i:
                dp.append(0)
        return dp[n]      
                

class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        
        self.memo = dict()
        return self.dfs(n)

        
    def dfs(self, left):
        if left <= 0:
            return False
        
        if left in self.memo:
            return self.memo[left]
        
        squareRoot = int(left ** 0.5)
        res = False
        for i in reversed(list(range(1, squareRoot + 1))):
            if not self.dfs(left - i * i):
                res = True
                break
        
        self.memo[left] = res
        return res
        
        
        
                

class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        
        if n == 0 or n == 2:
            return False
        if n == 1:
            return True
        dp = [-1] * (n+1)
        dp[0] = 0
        dp[1] = 1
        dp[2] = 0
        
        def pick(n):
            
            if dp[n] != -1:
                return dp[n]
            i = 1
            while i * i <= n:
                if i * i == n:
                    dp[n] = 1
                    return True
                if not pick(n-i*i):
                    dp[n] = 1
                    return True
                i = i+1
            dp[n] = 0
            return dp[n]
        pick(n)
        return dp[n] == 1
        
                
                
        
                
                
            
                
            
            
            
        
        
        
        
        

import math
class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        return self.alice_wins(n, {})
        
    def alice_wins(self, n, dp):
        if n in dp:
            return dp[n]
        x = 1
        dp[n] = False
        while x * x <= n:
            if not self.alice_wins(n - (x * x), dp):
                dp[n] = True
                break
            x += 1
        return dp[n]
   
'''
Alice tries to get to 0 first
Bob tries to get to get to 0 first/ Bob tries to not allow Alice get to zero first



37


Alice tries all perfect squares
If current num is perfect square, Bob picks it, else he picks 1
'''
class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        
        memo = {}
        def helper(n):
            if n in memo:
                return memo[n]
            if n==0:
                return False
            if n==1:
                return True
            i=1
            while i*i<=n:
                win = helper(n-i*i)
                if not win:
                    memo[n] = True
                    return True 
                i+=1
            memo[n] = False
            return False
        res = helper(n)
        return res
    
        

class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        
        @lru_cache(maxsize=None)

        def dfs(r):
          s = int(sqrt(r))
          if s**2 == r: return True
          for i in range(1, s+1):
            if not dfs(r - i **2): return True
          
          return False  # There is no i such that removing i**2 stones will win the game
          
        return dfs(n)

class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        dp = [False] * (n + 1)
        for i in range(1, n + 1):
            dp[i] = any(not dp[i - j**2] for j in range(1, int(i ** 0.5) + 1))
        return dp[-1]
            

from functools import lru_cache

class Solution:
  @lru_cache(None)
  def winnerSquareGame(self, n: int) -> bool:
    for i in range(1, n + 1):
      if i * i > n:
        break
      if i * i == n:
        return True
      if not self.winnerSquareGame(n - i * i):
        return True
    return False

from functools import lru_cache

class Solution:
    
    @lru_cache(None)
    def dp(self, n):
        if n == 0:
            return False
        for s in self.s:
            if n-s >= 0:
                if not self.dp(n-s):
                    return True
        return False
        
    
    def winnerSquareGame(self, n: int) -> bool:
        self.s, i = [1], 1
        while self.s[-1] < 10**5:
            i += 1
            self.s.append(i*i)
        
        return self.dp(n)

class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        
        @lru_cache(maxsize=None)
        def dfs(r):
          s = int(sqrt(r))
          if s**2 == r: return True
          for i in range(1, s+1):
            if not dfs(r - i **2): return True
          
          return False  # There is no i such that removing i**2 stones will win the game
          
        return dfs(n)

class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        # Approach: dfs with memoization - O(nn**0.5)
        self.memo = dict()
        return self.dfs(n)

    def dfs(self, stonesRemain):
        if stonesRemain <= 0:
            return False
        
        if stonesRemain in self.memo:
            return self.memo[stonesRemain]
        
        squareRoot = int(stonesRemain ** 0.5)
        res = False
        
        # iterate from the largest square
        for i in reversed(range(1, squareRoot + 1)):
            posRes = self.dfs(stonesRemain - i * i)
            # if there's a way such that opponent loses, we know
            # that we can win with current number of stones
            # So, we terminate early
            if not posRes:
                res = True
                break
        
        self.memo[stonesRemain] = res
        return res
class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        dp = [False]*(n+1)
        for i in range(1, n+1):
            for j in range(1, int(sqrt(i))+1):
                if not dp[i-j*j]:
                    dp[i] = True
        
        return dp[n]

class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        dp = [0]*(n+1)
        for i in range(1, n+1):
            if i**0.5 == int(i**0.5):
                dp[i] = 1
                continue
            else:
                for j in range(1, int(i**0.5)+1):
                    if dp[i - j**2] == 0:
                        dp[i] = 1
                        break
        return bool(dp[n])
class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        mem = {}
        def get_dp(i):
            if i == 0:
                return False
            elif i not in mem:
                root = 1
                while True:
                    if root*root > i:
                        mem[i] = False
                        break
                    else:
                        if not get_dp(i-root*root):
                            mem[i] = True
                            break
                        root += 1
            return mem[i]
        return get_dp(n)

class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        roots = [i*i for i in range(1, int(sqrt(n))+1)]
        dp = [False]*(n+1)
        for i in range(1, n+1):
            for j in roots:
                if i < j:
                    break
                    
                if not dp[i-j]:
                    dp[i] = True
        
        return dp[n]

class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        
        @lru_cache(maxsize=None)
        def dfs(remain):
            sqrt_root = int(sqrt(remain))
            # current player will win immediately by taking the square number tiles
            if sqrt_root ** 2 == remain:
                return True
            
            for i in range(1, sqrt_root+1):
                # if there is any chance to make the opponent lose the game in the next round,
                #  then the current player will win.
                if not dfs(remain - i*i):
                    return True
    
            return False
    
        return dfs(n)

class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        dp = {}
        
        def rec_win(n):
            if n == 0:
                return False
            if n == 1:
                return True
            if n in dp:
                return dp[n]
            for i in range(1,n):
                if i*i > n:
                    break
                if not rec_win(n-i*i):
                    dp[n] = True
                    return True
            dp[n] = False
            return False
        return rec_win(n)

class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        
        dp = {0:False}
                
        for i in range(1, n+1):
            
            # get dp[i]
            for j in range(1, int(i ** 0.5) + 1):
                sn = j ** 2
                if not dp[i-sn]:
                    dp[i] = True
                    break
            else:
                dp[i] = False
                
        return dp[n]

class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        
        #dp = {0:False}
        dp = [False] * (n + 1)
        
        for i in range(1, n+1):
            
            # get dp[i]
            for j in range(1, int(i ** 0.5) + 1):
                sn = j ** 2
                if not dp[i-sn]:
                    dp[i] = True
                    break
            else:
                dp[i] = False
                
        return dp[n]

class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        dp = [False]*(n+1)
        for i in range(1, len(dp)):
            for j in range(1, int(i**0.5)+1):
                if not dp[i-j*j]:
                    dp[i]=True
                    continue
        return dp[-1]
from functools import lru_cache

class Solution:
    @lru_cache(None)
    def winnerSquareGame(self, n: int) -> bool:
        if n == 0:
            return False
        x = 1
        while x*x <= n:
            if not self.winnerSquareGame(n-x*x):
                return True
            x += 1
        return False


import functools
from math import sqrt
class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        @functools.lru_cache(maxsize=10**5)
        def WSG(k):
            if k == 0:
                return False

            i = int(sqrt(k))
            while i >= 1:
                if not WSG(k-i**2):
                    return True
                i -= 1
            return False
        return WSG(n)
class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        memo = {}
        def isSquare(n):
            root = int(n ** 0.5)
            return root * root == n
        def dfs(n):
            if n in memo:
                return memo[n]
            if isSquare(n):
                memo[n] = True
                return memo[n]
            root = int(n ** 0.5)
            memo[n] = False
            for i in range(1, root + 1):
                if not dfs(n - i * i):
                    memo[n] = True
                    break
            return memo[n]
        return dfs(n)
class Solution:
    def stone_game(self, memo, n):
        if n in memo:
            return memo[n]
        
        memo[n] = False
        for cand in range(1, int(math.sqrt(n)) + 1):
            i = cand * cand
            memo[n] = memo[n] or not self.stone_game(memo, n-i)
            if memo[n]:
                return True
        
        return memo[n]
    
                
    def winnerSquareGame(self, n: int) -> bool:
        '''
        - implementation 
        '''
        can_i_win = defaultdict(bool)
        
        can_i_win[1] = True
        can_i_win[0] = False
        self.stone_game(can_i_win, n)
        
        return can_i_win[n]
class Solution:
    @lru_cache(None)
    def winnerSquareGame(self, n: int) -> bool:
        if n==1:
            return True
        s = 1
        while s*s<=n:
            flag = self.winnerSquareGame(n-s*s)
            if flag == False:
                return True
            s+=1
        return False
class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        
        def getSquareNumbers(n: int) -> List[int]:
            # less than or equal to n
            return [index * index for index in range(1, 1 + math.floor(math.sqrt(n)))]
        
        table = [False] * (n + 1)
        
        for index in range(n + 1):
            table[index] = any(not table[index - lose] for lose in getSquareNumbers(index))
            
        return table[-1]
import math
class Solution:
    
    def winnerSquareGame(self, n: int) -> bool:
        @lru_cache(None)
        def helper(n):
            if n == 0:
                return False
            i = 1
            while i*i <= n:
                if not helper(n - i*i):
                    return True
                i += 1
            return False
        
        return helper(n)
class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        
        winmap = {}
        winmap[0] = False
        winmap[1] = True
        
        def fill_map(n):
            if n in winmap:
                return winmap[n]
            i = 1
            winmap[n] = 0
            while i*i <= n:
                winmap[n] = winmap[n] or not fill_map(n-i*i) 
                if winmap[n]:
                    break
                i += 1
            return winmap[n]
        
        for i in range(1, n):
            fill_map(n)
            
        return winmap[n]
        
        
        
        
        

class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        memo = {0: False}
        def dfs(i):
            if i not in memo:
                memo[i] = False
                j = 1
                while not memo[i] and j * j <= i:
                    memo[i] = not dfs(i - j * j)
                    j += 1
            return memo[i]
        
        return dfs(n)
        
        
        
        
#         dp = [False] * (n + 1)
#         for i in range(1, n + 1):
#             j = 1
#             while not dp[i] and j * j <= i:
#                 dp[i] = not dp[i - j * j]
#                 j += 1
        
#         return dp[n]
                
            

class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        self.memo = {}
        return self.helper(n)
    
    def helper(self, n):
        if n == 0:
            return False
        if n in self.memo:
            return self.memo[n]
        i = 1
        while i * i <= n:
            if not self.helper(n - i * i):
                self.memo[n] = True
                return True
            i += 1
        self.memo[n] = False
        return False

class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        squares = [i * i for i in range(1, int(n ** 0.5) + 1)]
        
        DP = [0] * (1 + n)
        for i in range(1, n + 1):
            can_win = False
            for s in squares:
                if s > i:
                    break
                can_win |= not DP[i - s]
            DP[i] = can_win
        return DP[-1]
class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        # time O(n*sqrt(n)), space O(n)
        dp = [False] * (n + 1) # dp[i] means Alice can win with i
        dp[1] = True
        for i in range(2, n + 1):
            for k in range(1, int(i**0.5) + 1):
                if dp[i - k*k] == False: # if Bob can't win, then Alice wins
                    dp[i] = True
        return dp[n]
class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        dp = [False] * (n + 1)
        dp[1] = True
        for i in range(2, n + 1):
            for k in range(1, int(i**0.5) + 1):
                if dp[i - k*k] == False:
                    dp[i] = True
        return dp[n]
class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        dp = [False]*(n+1)
        for i in range(1, n+1):
            for j in range(1, int(sqrt(i))+1):
                if dp[i-j**2] is False:
                    dp[i] = True 
                    break 
        return dp[n]
class Solution:
    @lru_cache(None)
    def winnerSquareGame(self, n: int) -> bool:
        if n == 0:
            return False
        else:
            for i in range(1, int(n ** 0.5) + 1):
                if not self.winnerSquareGame(n - i**2):
                    return True  
            return False
        

class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        m = {0:False, 1:True}
        def helper(n):
            if n in m:
                return m[n]
            ans, start = False, 1
            while start*start<=n:
                ans = ans or (not helper(n-start*start))
                if ans:
                    m[n]=ans
                    return ans
                start+=1
            m[n]=ans
            return ans
        
        return helper(n)

class Solution:
    @lru_cache(None)
    @lru_cache(maxsize=None)
    def winnerSquareGame(self, n: int) -> bool:
        if n == 0:
            return False
        else:
            for i in range(1, int(n ** 0.5) + 1):
                if not self.winnerSquareGame(n - i**2):
                    return True  
            return False
        

class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        dp = [False]*(n+1)
        for i in range(1,n+1):
            cur = int(i**0.5)
            if cur ** 2 == i:
                dp[i] = True
            else:
                f = True
                for j in range(1,int(i**0.5)+1):
                    f &= dp[i-j*j]
                if not f:
                    dp[i] = True
        return dp[-1]
class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        
        M = {}
        M[0] = False
        M[1] = True
        for i in range(2, n+1):
            M[i] = False
            sq = int(math.sqrt(i))
            if sq**2 == i:
                M[i] = True
            for j in range(1, sq + 1):
                M[i] = M[i] or not M[i-(j*j)]
        return M[n]
            
        

from functools import lru_cache
class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        @lru_cache(None)
        def dp(num):
            sqr = int(num ** (1/2))
            if sqr ** 2 == num:
                return True
            
            way = False
            for i in range(1, sqr + 1):
                way = not dp(num - i ** 2)
                if way:
                    return True
            return way
        
        return dp(n)
class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        memo = {}
        def dfs(n):
            if n == 0:
                return False
            if n in memo:
                return memo[n]
            can_win = False
            for i in range(1, int(n ** 0.5) + 1):
                if not dfs(n - i ** 2):
                    can_win = True
                    break
            memo[n] = can_win
            return memo[n]
        return dfs(n)
class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        dp = [0] * (n + 1)
        squares = []
        i = 1
        nxt_sqrt = 1
        nxt = 1
        for i in range(1, n + 1):
            if i == nxt:
                dp[i] = 1
                squares.append(nxt)
                nxt_sqrt += 1
                nxt = nxt_sqrt ** 2
            else:
                dp[i] = max(-dp[i - s] for s in squares)
        return dp[n] == 1
        
#         squares = [i**2 for i in range(1, int(sqrt(n)) + 1)]

#         for s in squares:
#             dp[s] = 1

        
        # @lru_cache(None)
        # def play(n):
        #     sq = sqrt(n)
        #     if int(sq) == sq:
        #         return 1
        #     best = -1
        #     for i in range(int(sq), 0, -1):
        #         best = max(best, -play(n - i ** 2))
        #     return best
        # return play(n) == 1

class Solution:
    # dp(i): remain i piles
    def winnerSquareGame(self, n: int) -> bool:
        if n == 0: return False
        if n == 1: return True
        squares = [i**2 for i in range(1, int(sqrt(n))+1)]
        @lru_cache(None)
        def dp(i): 
            nonlocal squares
            canWin = False
            for sq in squares:
                if i < sq:
                    break
                if i == sq:
                    return True
                canWin = canWin or not dp(i - sq)
            return canWin
        return dp(n)
import math
class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        result = [None] * (n + 1)
        def canWin(n):
            bound = math.ceil(math.sqrt(n))
            if bound * bound == n:
                result[n] = True
                return True
            if result[n] != None:
                return result[n]
            for i in range(1, bound):
                if not canWin(n - i * i):
                    result[n] = True
                    return True
            result[n] = False
            return False
        return canWin(n)
class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        
        @lru_cache(None)
        def rec(n):
            if n < 2: return n == 1
            return any(not rec(n-i*i) for i in range(int(n**0.5), 0, -1))
        return rec(n)
class Solution:
    @lru_cache(None)
    def winnerSquareGame(self, n: int) -> bool:
        if n == 0:
            return False
        for i in range(1, int(n ** 0.5) + 1):
            if not self.winnerSquareGame(n - i ** 2):
                return True
        return False

class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        dp = [False]
        for i in range(1, n+1):
            b = 1
            winnable = False
            while b ** 2 <= i:
                if not dp[i-b**2]:
                    winnable = True
                    break
                b += 1
            dp.append(winnable)
        return dp[-1]
class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        cache = {}

        def helper(number):
            if number == 0:
                return False
            
            if number not in cache:
                flag = False
                for i in range(1, 317):
                    val = i ** 2
                    if val > number:
                        break

                    if not helper(number - val):
                        flag = True
                        break
                
                cache[number] = flag
            
            return cache[number]
        
        return helper(n)
from math import sqrt
class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        dp = [False] * (n+1)
        dp[1] = True
        
        for m in range(2,n+1) :
#             if sqrt(m) == int(sqrt(m)) : 

#                 dp[m] = True
#                 continue 
            i = 1
            while i**2 < m : 
                if not dp[m-i**2] : 

                    dp[m] = True
                    break
                i +=1 
            if i**2 == m : dp[m] = True

        return dp[n]
                
            
        
'''
True,False, True, True, False, True, 
Brute force:

Bactracking method:
proceed with removing a certain number of stones which are a square number
and then have another state
(stonesRemaining: int, AliceTurn: bool)

so base case, if n == 0 or person can't make any moves: return not AliceTurn

construct dp table of two dimensions 

at any subproblem m, if alice wins, then bob necessarily loses and vice versa 

n =1 
n is a square, return True

n= 2, 
n is not square so only option is take 1 stone
(2) -> (1) -> False

n= 4
if n is a square: return True

n = 7 
dp(1) : return True meaning alice wins
dp(n: int) -> bool :
    if sqrt(n) == int(sqrt(n)) : return True
    i = 1
    while i**2 < n : 
        if dp[n-i] : return True
        i +=1 
    return True

return not dp(n)

'''
class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        
        table = [False] * (n+1)
        table[1] = True
        for i in range(2, n+1):
            num = 1
            while num ** 2 <= i:
                square = num ** 2
                if not table[ i - square ]:
                    table[i] = True
                    break
                num += 1
                
        return table[-1]
class Solution:
    @lru_cache(None)
    def winnerSquareGame(self, n: int) -> bool:
        if n == 1:
            return True
        sqrt = math.floor(math.sqrt(n))
        if sqrt * sqrt == n:
            return True
        for num in range(1, sqrt+1):
            if not self.winnerSquareGame(n-num**2):
                return True
        return False

class Solution:
    @lru_cache(None)
    def winnerSquareGame(self, n: int) -> bool:
        table = [False for x in range(n+1)]
        for x in range(1, n+1):
            flag = False
            c = 1
            while c**2 <= x:
                if not table[x-c**2]:
                    flag = True
                    break
                c += 1
            table[x] = flag
        return table[-1]
from functools import lru_cache
class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        sq = [x*x for x in range(1, 317)]
        dp = [False] * (n+1)
        
        for i in range(n+1):
            for x in sq:
                if i - x < 0:
                    break
                if dp[i - x] == False:
                    dp[i] = True
        
        return dp[n]

    # T F T F F

class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        dic =dict()
        dic[1] = True
        dic[2] = False
        dic[3] = True
        dic[4] = True
        def helper(n):
            if n in dic:
                return dic[n]
            i = int(n ** 0.5)
            while i >= 1:
                if not helper(n - i**2):
                    dic[n] = True
                    return True
                i -=1
            dic[n] = False
            return False
        return helper(n)
from math import sqrt
class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        dp = [False] * (n+1)
        dp[1] = True
        
        for m in range(2,n+1) :
            if sqrt(m) == int(sqrt(m)) : 

                dp[m] = True
                continue 
            i = 1
            while i**2 < m : 
                if not dp[m-i**2] : 
                    # print(i**2, m)
                    dp[m] = True
                    break
                i +=1 
        # print(dp)
        return dp[n]
                
            
        
'''
True,False, True, True, False, True, 
Brute force:

Bactracking method:
proceed with removing a certain number of stones which are a square number
and then have another state
(stonesRemaining: int, AliceTurn: bool)

so base case, if n == 0 or person can't make any moves: return not AliceTurn

construct dp table of two dimensions 

at any subproblem m, if alice wins, then bob necessarily loses and vice versa 

n =1 
n is a square, return True

n= 2, 
n is not square so only option is take 1 stone
(2) -> (1) -> False

n= 4
if n is a square: return True

n = 7 
dp(1) : return True meaning alice wins
dp(n: int) -> bool :
    if sqrt(n) == int(sqrt(n)) : return True
    i = 1
    while i**2 < n : 
        if dp[n-i] : return True
        i +=1 
    return True

return not dp(n)

'''
class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        dp = [False] * (n + 1)
        for i in range(1, n+1):
            for k in range(1, i+1):
                if i - k**2 < 0: break
                if not dp[i - k**2]: dp[i] = True;break
            #print(dp)        
        return dp[-1]

# O(n) dp[n] = !dp[n-1] or !dp[n-4] ...
class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        dp = [-1] * (n+1)
        dp[0] = 0
        return self.can_win(n, dp)
    
    def can_win(self, n, dp):
        if dp[n] != -1:
            return dp[n]
        root = 1
        cur = n - root ** 2
        while cur >= 0:
            dp[cur] = self.can_win(cur, dp)
            if dp[cur] == 0:
                dp[n] = 1
                return 1
            root += 1
            cur = n - root ** 2
        dp[n] = 0
        return dp[n]
class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        # This is an even simpler and more efficient way to reason about the state of 
        # f(x). Instead of defining f(x) for whether Alice wins or loses, let f(x)
        # return True/False for if someone, Alice or Bob wins/loses given the value x.
        # In other words, f(x) tells us whether we'll win or lose if we start with x.
        # If "we" are Alice starting with x, then our opponent is Bob, if "we" are
        # Bob then the opponent is Alice.
        memo = {}
        def f(x):
            if x == 0:
                return False
            if x == 1:
                return True
            
            if x in memo:
                return memo[x]
            
            i = 1
            while i * i <= x:
                # If we choose to remove i*i and our opponent ends up losing then
                # we win. Why? Since we are playing optimally, as long as there's a
                # choice we can make that forces the opponent to lose, we will make
                # that choice and guarantee the win.
                if f(x - i * i) == False:
                    memo[x] = True
                    return True
                i += 1
            
            # If no matter the choice we make, our opponent ends up winning, i.e
            # f(x - i * i) == True for all i, i*i <= x, then we are guaranteed to lose
            memo[x] = False
            return False
        
        return f(n)
            

class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        # u6bcfu6b21u79fbu9664u5e73u65b9u6570
        # aliceu6bcfu6b21u79fbu9664u5b8cu4e0du80fdu5269u4f59u5e73u65b9u6570uff0cu5426u5219u5c31u8f93
        # u5f53u524du6570u5b57u51cfu53bbu4e00u4e2au5e73u65b9u6570u540euff0cu4e0du80fdu662fu5e73u65b9u6570
        # u4eceu5c0fu5230u5927u8ba1u7b97uff0cn=1uff0cu3002u3002u3002uff0cn
        # dp[i]: u6709iu4e2au77f3u5934u65f6uff0caliceu80fdu4e0du80fdu8d62
        # dp[i] = u4eceiu5411u4e0bu51cfu53bbu5e73u65b9u6570uff0cu5bf9u5e94u7684dp[j]u6709u4e00u4e2au662ffalseu5373u53ef
        dp = [False for _ in range(n+1)]
        for i in range(1,n+1):
            base = 1
            while i-base**2>=0:
                if not dp[i-base**2]:
                    dp[i] = True
                    break
                else:
                    base += 1
        return dp[n]

class Solution:
    
    table = [-1] * 1000001
    table[0] = False
    table[1] = True
    idx = 1
    
    def winnerSquareGame(self, n: int) -> bool:
        
        if n > self.idx:
            for i in range(self.idx+1, n+1):
                num = 1
                while num ** 2 <= i:
                    square = num ** 2
                    if not self.table[ i - square ]:
                        self.table[i] = True
                        break
                    num += 1
                if self.table[i] == -1:
                    self.table[i] = False
            self.idx = i
        return self.table[n]
class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        dp = [False] * (n + 1)
        for i in range(1, n+1):
            for k in range(1,int(i**0.5)+1):
                if i - k**2 < 0: break
                if not dp[i - k**2]: dp[i] = True;break
            #print(dp)        
        return dp[-1]
        '''
        dp = [False] * (n + 1)
        for i in range(1, n + 1):
            dp[i] = not all(dp[i - k * k] for k in range(1, int(i**0.5) + 1))
        return dp[-1]   
        '''
class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        dp = [False]*(n+1)
        dp[1] = True 
        for i in range(2, n+1):
            j = 1 
            while i - j**2 >= 0:
                if dp[i-j**2] is False:
                    dp[i] = True
                    break 
                j += 1 
            
        return dp[n]
import math
class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        result = dict()
        def canWin(n):
            bound = math.ceil(math.sqrt(n))
            if bound * bound == n:
                result[n] = True
                return True
            if n in result:
                return result[n]
            for i in range(1, bound):
                if not canWin(n - i * i):
                    result[n] = True
                    return True
            result[n] = False
            return False
        return canWin(n)
class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        
        @lru_cache(None)
        def dp(i):
            if i == 0:
                return False
            return any(not dp(i-x*x) for x in range(floor(sqrt(i)), 0, -1))

        return dp(n)

import math
class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        dp = [False] * (n+1)
        for i in range(1, n+1):
            curr = 1
            while curr**2 <= i:
                if dp[i-curr**2] == False:
                    dp[i] = True
                    break
                curr += 1
        return dp[-1]

class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        # if n == 1:
        #     return True
        # if n == 2:
        #     return False
        
        dp = [0] * (n+1)
        # dp[1] = 1
        
        for i in range(1, n+1):
            base = 1
            while i - base**2 >= 0: 
                if dp[i - base**2] == False:
                    dp[i] = 1
                    break
                base += 1

        return dp[-1]
class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        if n == 1:
            return True
        if n == 2:
            return False
        
        dp = [0] * (n+1)
        dp[1] = 1
        
        for i in range(3, n+1):
            base = 1
            while i - base**2 >= 0: 
                if dp[i - base**2] == False:
                    dp[i] = 1
                    break
                base += 1

        return dp[-1]

class Solution:
    def __init__(self):
        self.memo = {0: False}
        
    def winnerSquareGame(self, n: int) -> bool:
        def rwsg(n,picks):
            if n not in self.memo:
                out = False
                for p in picks:
                    if p > n: break
                    out = out or not rwsg(n-p,picks)
                self.memo[n] = out
            return self.memo[n]
        picks = []
        m = 1
        while m*m <= n:
            picks.append(m*m)
            m += 1
        rwsg(n,picks)
        return rwsg(n,picks)
import math
class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        result = dict()
        def canWin(n):
            bound = math.ceil(math.sqrt(n))
            if n in result:
                return result[n]
            result[n] = False
            for i in range(1, bound + 1):
                if n == i * i:
                    result[n] = True
                    return True
                if n > i * i:
                    if not canWin(n - i * i):
                        result[n] = True
                        return True
            return result[n]
        return canWin(n)
class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        # Square numbers: 1, 4, 9, 16, 25...

        def removeSquare(n: int, memo: [int]):
            
            if n == 0: return False
            if memo[n]: return memo[n]

            i = 1
            
            while i*i <= n:
                
                if memo[n-(i*i)] == False: return True
                
                memo[n-(i*i)] = removeSquare(n-(i*i), memo)
                
                # If you can make a move, that will result
                # in the case: n==0
                # Then you can win
                if memo[n-(i*i)] == False:
                    return True
                
                i += 1
                
            return False
        
        memo = [None] * (n+1)
        
        return removeSquare(n, memo)
class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        dp = [False] * (n + 1)
        for i in range(1, n + 1):
            j = 1
            while j * j <= i:
                if not dp[i - j * j]:
                    dp[i] = True
                j += 1
        return dp[n]
import functools
import math


class Solution:
  def winnerSquareGame(self, n: int) -> bool:
    @functools.lru_cache(None)
    def dp(k):
      if k <= 0:
        return False

      sq = math.sqrt(k)
      if sq.is_integer():
        return True

      for m in range(1, k):
        if m ** 2 > k:
          break

        ans = dp(k - m ** 2)
        if not ans:
          return True

      return False

    return dp(n)
class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        from functools import lru_cache
        @lru_cache(None)
        def dp(n):
            if math.sqrt(n) == round(math.sqrt(n)):
                return True
            i = 1
            while i**2 < n:
                if not dp(n - i**2):
                    return True
                i += 1
            return False
        return dp(n)
class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        
        
        n2p = []
        p2n = {}
        for x in range(n+1): 
            n2p.append(x**2)
            p2n[x**2] = x
        
        #print(n2p)
        #print(p2n)
        
        @lru_cache(None)
        def recur(rem):
            if rem == 0: return False
            if round(math.sqrt(rem)) ** 2 == rem: return True
            
            
            #print("rem", rem)
            max_rm_val = math.floor(math.sqrt(rem))**2
            #print("val", max_rm_val)
            max_rm_ind = p2n[max_rm_val]
            
            for ind in range(max_rm_ind, 0, -1):
                
                # hope that at least one next call returns False
                if not recur(rem - n2p[ind]): return True
            
            return False
        
        return recur(n)
'''
    1 - Alice
    2 - Bob
    3 - Alice
    4 - Alice
    5 - [1 (Bob), 4(Bob)]
    6 - [1 (Alice), 4(Bob)] 
    7 - [1, 4]
    n - Alice
    n+1 -
'''
class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        if n == 1:
            return True
        dp = [False] * (n+1)
        squares = set()
        for i in range(1, n//2+1):
            square = i * i
            if square <= n:
                squares.add(square)
                dp[square] = True   
        for i in range(1, n+1):
            if i not in squares:
                possible = [not dp[i-square] for square in squares if square < i]
                dp[i] = any(possible)
        return dp[-1]

import functools
class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        @functools.lru_cache(maxsize=None)
        def WSG(k):
            if k == 0:
                return False

            i=1
            while i**2 <= k:
                if not WSG(k-i**2):
                    return True
                i += 1
            return False
        return WSG(n)

class Solution:
  def winnerSquareGame(self, n: int) -> bool:
    squart_num = []
    dp = [False] * (n + 1)
    i = 1

    for i in range(1, n + 1):
      start_index = 1
      while True:
        if start_index * start_index > i:
          break
        if not dp[i - start_index * start_index]:
          dp[i] = True
        start_index += 1

        pass
    if dp[n] == 1:
      return True
    return False
    pass

'''
    1 - Alice
    2 - Bob
    3 - Alice
    4 - Alice
    5 - [1 (Bob), 4(Bob)]
    6 - [1 (Alice), 4(Bob)] 
    7 - [1, 4]
    n - Alice
    n+1 -
'''
class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        if n == 1:
            return True
        dp = [False] * (n+1)
        squares = set()
        for i in range(1, n//2+1):
            square = i * i
            if square <= n:
                squares.add(square)
                dp[square] = True   
        # print(squares)
        for i in range(1, n+1):
            if i not in squares:
                possible = [not dp[i-square] for square in squares if square < i]
                # print(i, possible)
                dp[i] = any(possible)
        # print(dp)
        return dp[-1]
class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        
        from functools import lru_cache
        @lru_cache(None)
        def can_win(remain):
            if remain == 0:
                return False
            
            root = 1
            while root**2 <= remain:
                if not can_win(remain - root**2):
                    return True
                root += 1
            return False
        
        return can_win(n)
class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        
        @lru_cache(None)
        def test(n):
            if n == 0:
                return False
        
            for i in range(1, int(n ** 0.5) + 1):
                sn = i ** 2
                if not test(n - sn):
                    return True
        
            return False
        
        return test(n)

class Solution:
    def __init__(self):
        self.isGood = {}
    
    
    def winnerSquareGame(self, n: int) -> bool:
        if n <= 0:
            return False
        
        if n in self.isGood:
            return self.isGood[n]
        
        self.isGood[n] = False
        i = 1
        while i*i <= n:
            if not self.winnerSquareGame(n - i*i):
                self.isGood[n] = True
                return True
            i += 1
            
        return self.isGood[n]
class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        memo = {}
        memo[0] = False
        memo[1] = True
        def dfs(n) -> bool:
            if n in memo:
                return memo[n]
            i = 1
            while i*i <= n:
                res = dfs(n - i*i) 
                if not res:
                    memo[n] = True
                    return True
                i += 1
            memo[n] = False
            return False
        return dfs(n) 
class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        @lru_cache(None)
        def helper(n):
            if n == 0:
                return -1
            for i in range(int(sqrt(n)), 0, -1):
                # print(i)
                if(helper(n - i*i) < 0):
                    return 1
            return -1
        return helper(n) > 0
class Solution:
    @lru_cache(None)
    def winnerSquareGame(self, n: int) -> bool:
        return True if n == 1 else any(not self.winnerSquareGame(n - i ** 2) for i in range(int(n ** 0.5), 0, -1))
               
            
            

class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        dp=[False]*(n+1)
        dp[0]=False
        dp[1]=True
        for i in range(2,n+1):
            k=1
            while k**2<=i:
                if dp[i-k**2]==False:
                    dp[i]=True
                    break
                k+=1
        return dp[n]
class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        
        dp = defaultdict(bool)
        
        def helper(n):
            if n == 0:
                return False
            
            if n in dp:
                return dp[n]
            
            for i in range(1, int(n**0.5)+1):
                if not helper(n-i**2):
                    dp[n] = True
                    return True
            
            dp[n] = False
            return False
        
        return helper(n)
            


import functools
class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        @functools.lru_cache(maxsize=10**5)
        def WSG(k):
            if k == 0:
                return False

            i=1
            while i**2 <= k:
                if not WSG(k-i**2):
                    return True
                i += 1
            return False
        return WSG(n)
class Solution:
    @lru_cache(maxsize=None)
    def winnerSquareGame(self, n: int) -> bool:
        # if a perfect square is available, you win
        if int(sqrt(n)) ** 2 == n:
            return True

        for i in range(1, int(sqrt(n)) + 1):
            if not self.winnerSquareGame(n - i * i):
                return True

        return False
class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        dp=[False]*(n+1)
        dp[0]=False
        for i in range(1,n+1):
            k=1
            while k**2<=i:
                if dp[i-k**2]==False:
                    dp[i]=True
                    break
                k+=1
        return dp[n]
class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        dp = [-1 for i in range(n+1)]
        
        def help(n):
            if dp[n]==-1:            
                if n<1:
                    result = False
                else:
                    result = False
                    i=1
                    while i**2 <= n:
                        move = i**2
                        result = result or not(help(n-move))
                        if result:
                            break
                        i+=1
                        
                dp[n] = result
            return dp[n]
        return help(n)
class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        
        memo = {0:False}
        def wing(x):
            if math.sqrt(x) == int(math.sqrt(x)):
                return True
            if x in memo:
                return memo[x]
            i = 1
            ans = False
            while i*i <= x:
                if wing(x-i*i) == False:
                    ans = True
                    break
                i += 1
            memo[x] = ans
            return ans
        
        return wing(n)
class Solution:
    @lru_cache(1024**2)
    def winnerSquareGame(self, n: int) -> bool:
        if n == 0: return False
        for i in range(1, int(sqrt(n))+1):
            if i*i > n: continue
            if not self.winnerSquareGame(n - (i*i)):
                return True
        return False
class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        square = []
        for i in range(1, n+1):
            if i*i <= n:
                square.append(i*i)
            else:
                break
        
        from functools import lru_cache
        @lru_cache(None)
        def dp(n, state):
            if n == 0:
                if state == 1:
                    return False
                else:
                    return True
            
            if state == 1:
                tmp = False
                for num in square:
                    if num <= n:
                        tmp = tmp or dp(n-num, 1-state)
                        if tmp == True:
                            return tmp
                    else:
                        break
            else:
                tmp = True
                for num in square:
                    if num <= n:
                        tmp = tmp and dp(n-num, 1-state)
                        if tmp == False:
                            return tmp
                    else:
                        break
                
            return tmp
        
        return dp(n, 1)

class Solution:
    @lru_cache(None)
    def winnerSquareGame(self, n: int) -> bool:
        if n == 1:
            return True
        return any(not self.winnerSquareGame(n - i ** 2) for i in range(int(n ** 0.5), 0, -1))
               
            
            

class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        dp = [False] * (n + 1)
        for i in range(1, len(dp)):
            j = 1
            while j * j <= i:
                if dp[i - j * j] == False:
                    dp[i] = True
                j += 1
        return dp[-1]

import functools
class Solution:
    @functools.lru_cache(maxsize=None)
    def winnerSquareGame(self, n: int) -> bool:
        if n == 0:
            return False
        
        i=1
        while i**2 <= n:
            if not self.winnerSquareGame(n-i**2):
                return True
            i += 1
        return False
class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        dp = [0] * (n+3)
        dp[1] = 1 #Alice
        dp[2] = 0 #Bob
        
        for i in range(3,n+1):
            for j in range(1,int(i**0.5) + 1):
                if j**2<=i:
                    if dp[i-j**2]==0:
                        dp[i] = 1
                        break
                        
        print((dp[n]))
        return dp[n]==1

class Solution:
    @lru_cache(None)
    def winnerSquareGame(self, n: int) -> bool:
        if n == 0 or n == 1: return True
        i = 1
        while i ** 2 < n:
            if not self.winnerSquareGame(n - i ** 2): return True
            i += 1
        return i**2 == n
class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        
        lookup = {}
        def getSquareNumbers(n: int) -> List[int]:
            if n in lookup:
                return lookup[n]
            # less than or equal to n
            lookup[n] = [index * index for index in range(1, 1 + math.floor(math.sqrt(n)))]
            return lookup[n]
        
        table = [False] * (n + 1)
        
        for index in range(n + 1):
            table[index] = any(not table[index - lose] for lose in getSquareNumbers(index))
            
        return table[-1]
class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        dp = [0,1]
        
        for s in range(2,n+1):
            dp.append(0)
            i = 1
            while i**2 <= s:
                dp[-1] = max(dp[-1], 1-dp[s-i**2])
                if dp[-1] == 1: break
                i += 1
        return dp[-1]
class Solution:
    @lru_cache(None)
    def winnerSquareGame(self, n: int) -> bool:
        if n in [1, 0]: return True
        i = 1
        while i ** 2 < n:
            if not self.winnerSquareGame(n - i ** 2): return True
            i += 1
        return i**2 == n
class Solution:
    @lru_cache(None)
    def winnerSquareGame(self, n: int) -> bool:
        if n == 0 or n == 1: return True
        i = int(sqrt(n))
        if i ** 2 == n: return True
        while i > 0:
            if not self.winnerSquareGame(n - i ** 2): return True
            i -= 1
        return False
class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        dp = [False]*(n+1)
        for i in range(1,n+1):
            j = 1
            while j*j <= i:
                dp[i] |= not dp[i-j*j]
                j+=1
        return dp[n]

class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        @lru_cache(None)
        def dp(n,people):
            for i in range(1,400):
                if i*i<=n:
                    result=dp(n-i*i,0 if people is 1 else 1)
                    if people==0:# Alice
                        if result:
                            return result
                    else:
                        if not result:
                            return result
                else:
                    break
            if people==0:# Alice
                return False
            else:
                return True
        return dp(n,0)
            

class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        @lru_cache(None)
        def dfs(m, p):
            if m == 0:
                #print(m, p, p == 1)
                return p == 1
            if m == 1:
                #print(m, p, p == 0)
                return p == 0
            i = 1
            while i * i <= m:
                if p == 0 and dfs(m - i * i, 1):
                    #print(m, p, True)
                    return True
                elif p == 1 and not dfs(m - i * i, 0):
                    #print(m, p, False)
                    return False
                i += 1
            #print(m, p, p == 1)
            return p == 1
        return dfs(n, 0)
class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        win = [False]
        for i in range(n):
            j = 1
            can_win = False
            while j ** 2 <= len(win):
                if not win[-j ** 2]:
                    can_win = True
                    break
                j += 1
            win.append(can_win)
        return win[-1]
class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        @lru_cache(None)
        def get(k):
            if not k: return False
            return not all([get(k-i*i) for i in range(1, int(k**0.5)+1)])
        return get(n)

class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        m = {0:False, 1:True}
        def helper(n):
            if n in m:
                return m[n]
            ans, start = False, 1
            while start*start<=n:
                ans = ans or (not helper(n-start*start))
                start+=1
            m[n]=ans
            return ans
        
        return helper(n)

class Solution:
    @lru_cache(None)
    def winnerSquareGame(self, n: int) -> bool:
        return True if n == 1 else any(not self.winnerSquareGame(n - i ** 2) for i in range(int(n ** 0.5), 0, -1))
class Solution:
    @lru_cache(None)
    def winnerSquareGame(self, n: int) -> bool:
        return True if n == 1 else any(not self.winnerSquareGame(n - i ** 2) for i in range(int(n ** 0.5), 0, -1))

class Solution:
    @lru_cache(None)
    def winnerSquareGame(self, n: int) -> bool:
        if n == 0:
            return False
        return any(not self.winnerSquareGame(n - x ** 2) for x in range(int(n ** 0.5), 0, -1))
class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        dp = {0:False}
        for i in range(0, n):
            if i in dp and i!=0: 
                continue
            for j in range(1, n+1):
                if i+j*j > n:
                    break
                dp[i+j*j]=True
        return False if n not in dp else dp[n]
import math
class Solution:
    squares=[0]
    dp=[False]            
    def winnerSquareGame(self, n: int) -> bool:
        sqt=int(math.sqrt(n))
        for i in range(len(self.squares),sqt+1):
            self.squares.append(i*i)
        
        if n+1<=len(self.dp):
            return self.dp[n]
        
        old_len=len(self.dp)
        for i in range(old_len,n+1):
            self.dp.append(False)
        
        for i in range(old_len,n+1):
            flag=0
           # print("in loop i")
            for j in range(1,int(math.sqrt(i))+1):
               # print("i and j are",i,j)
                if not self.dp[i-self.squares[j]]:
                    self.dp[i]=True
                    flag=1
                    break
            if flag==0:
                self.dp[i]=False
       # print(dp)
        return self.dp[n]
                
                

class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        @lru_cache(None)
        def dp(n,people):
            for i in range(floor(math.sqrt(n)),0,-1):
                if i*i<=n:
                    result=dp(n-i*i,0 if people is 1 else 1)
                    if people==0:# Alice
                        if result:
                            return result
                    else:
                        if not result:
                            return result
            if people==0:# Alice
                return False
            else:
                return True
        return dp(n,0)
            

class Solution:
    @lru_cache(None)    
    def winnerSquareGame(self, n: int) -> bool:
        for i in reversed(range(1, int(sqrt(n))+1)):
            if not self.winnerSquareGame(n-i*i):
                return True
        return False
class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        win = [False] * (n + 1)
        for i in range(n):
            if not win[i]:
                j = 1
                while i + j ** 2 <= n:
                    if i + j ** 2 == n:
                        return True
                    win[i + j ** 2] = True
                    j += 1
        return False
class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        to_ret = [False]
        for i in range(1, n+1) :
            for j in range(int(i**0.5), 0, -1) :
                if not to_ret[i-j*j] :
                    to_ret.append(True)
                    break
            if len(to_ret) == i :
                to_ret.append(False)
        return to_ret[-1]
class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        base = 1
        steps = []
        dp = [False for i in range(n+1)]
        for i in range(1, n+1):
            if base * base <= i:
                steps.append(base*base)
                base += 1
            for step in steps:  
                if not dp[i-step]:
                    dp[i] = True
                    break
        return dp[-1]
from functools import lru_cache
class Solution:
    @lru_cache(None)
    def winnerSquareGame(self, n: int) -> bool:
        k = int(math.sqrt(n))
        if k*k ==n or n==3:
            return True
        if n==2:
            return False
        if all(self.winnerSquareGame(n-i**2) for i in range(k,0,-1)):
            return False
        return True
        
        

class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        def dfs(n):
            if n == 0:
                return False
            if int(n ** 0.5) ** 2 == n:
                return True
            if n not in dic:
                dic[n] = any(not dfs(n-i) for i in poss if i <= n)
            return dic[n]
        poss = []
        dic = {}
        for i in range(1, int(n**0.5)+1):
            poss.append(i*i)
        poss = poss[::-1]
        return dfs(n)

class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        square = [i**2 for i in range(1,int(n**0.5)+1)]
        dp = [False for i in range(n+1)]
        dp[1] = True
        dp[0] = False
        
        for i in range(2,n+1):
            for sq in square:
                if sq > i:
                    break
                if not dp[i-sq]:
                    dp[i] = True
                    break
                    
        return dp[-1]
class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        
        dp = [False]*(n+1)
        
        for i in range(1, n+1):
            for x in range(1, int(sqrt(i))+1):
                if not dp[i-x*x]:
                    dp[i] = True
                    break
                    
        return dp[n]

from math import sqrt
class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        from functools import lru_cache
        @lru_cache(None)
        def df(x):
            if int(sqrt(x)) **2 == x: return True
            if not x: return False
            for i in range(int(sqrt(x)),0,-1):
                if not df(x-i*i): return True
            return False
        return df(n)

class Solution:
    
    def winnerSquareGame(self, n):
        dp = [False] * (n + 1)
        for i in range(1, n + 1):
            #dp[i] = not all(dp[i - k * k] for k in range(1, int(i**0.5) + 1))
            iswin = False
            for k in range(1, int(sqrt(i))+1):
                if not dp[i-k*k]:
                    iswin = True
                    break
            dp[i] = iswin
        return dp[-1]
    
    def winnerSquareGame1(self, n: int) -> bool:
        # u5c1du8bd5u540cu6837u7684u89e3u6cd5uff1f
        # solve(s), s=u5f53u524du7684u5269u4f59u7684u6570u5b57u4e0buff0cscore diff? u9700u8981u7684u6700u5c0fu6b65u6570uff1f
        # u5982u679cs=0uff0cu5219u8fd4u56deFalse
        # u5982u679cs=u4efbu610fu4e00u4e2au5e73u65b9u6570uff0cu5219return True, u56e0u4e3aaliceu53efu4ee5u90fdu62ffu8d70
        # u5426u5219uff0caliceu53efu4ee5u4eceu4e2du5207u8d70u4efbu610fu5927u5c0fu7684u5e73u65b9u6570xuff0cu7136u540eu628au5269u4e0bu7684s-xu6254u7ed9bob
        
        # u53eau8981u5176u4e2du6709u4e00u4e2au5207u5272uff0cbobu65e0u6cd5u5728s-xu4e0bu83b7u80dcuff0cu90a3u5c31u662faliceu83b7u80dc
        # u5982u679cbobu5728u6240u6709u5207u5272u4e0bu90fdu83b7u80dcuff0cu90a3alice lose
        
        # u53eau4ecealiceu7684u89d2u5ea6u51fau53d1uff0cu662fu5426u8db3u591fuff1f
        cache = dict()
        
        def solve(s): # u9012u5f52u8c03u7528u672cu8eabuff0cu4f1au82b1u8d39u6bd4u8f83u591au7684u65f6u95f4uff01
            if s in cache: return cache[s]
            if s == 0: 
                cache[s] = False
                return False
            
            if pow(int(sqrt(s)), 2) == s: 
                cache[s] = True
                return True # s is a square number and current player can take it directly, so win
            
            iswin = False
            #for x in range(s-1, 0, -1): # from 1 to s-1, since s is not a square number, too slow if write in this way!
            #    if pow(int(sqrt(x)), 2) == x:
            #        if not solve(s-x):
            #            iswin = True
            #            break
            for k in range(1, int(sqrt(s))+1): # this can pass! great! 2612 ms, 37%
                if not solve(s - k*k):
                    iswin = True
                    break
                
            cache[s] = iswin
            return iswin
        return solve(n) # u65b9u6cd5u662fu5bf9u7684uff0cu4f46u662fu8d85u65f6u4e86uff0cn=31250u7684u65f6u5019

class Solution:
    
    def winnerSquareGame(self, n):
        dp = [False] * (n+1)
        for i in range(1, n+1):
            iswin = False
            for k in range(1, int(sqrt(i)) + 1):
                if not dp[i-k*k]:
                    iswin = True
                    break
            dp[i] = iswin
        return dp[-1]
    
    def winnerSquareGame2(self, n): # fast speed, 888ms, 77.55%
        dp = [False] * (n + 1)
        for i in range(1, n + 1):
            #dp[i] = not all(dp[i - k * k] for k in range(1, int(i**0.5) + 1))
            iswin = False
            for k in range(1, int(sqrt(i))+1):
                if not dp[i-k*k]: # if there is one (one is enough) path that the other player lose, then current player win
                    iswin = True
                    break
            dp[i] = iswin # when no path the other player lose, then iswin=False; otherwise, iswin=True!
        return dp[-1]
    
    def winnerSquareGame1(self, n: int) -> bool:
        # u5c1du8bd5u540cu6837u7684u89e3u6cd5uff1f
        # solve(s), s=u5f53u524du7684u5269u4f59u7684u6570u5b57u4e0buff0cscore diff? u9700u8981u7684u6700u5c0fu6b65u6570uff1f
        # u5982u679cs=0uff0cu5219u8fd4u56deFalse
        # u5982u679cs=u4efbu610fu4e00u4e2au5e73u65b9u6570uff0cu5219return True, u56e0u4e3aaliceu53efu4ee5u90fdu62ffu8d70
        # u5426u5219uff0caliceu53efu4ee5u4eceu4e2du5207u8d70u4efbu610fu5927u5c0fu7684u5e73u65b9u6570xuff0cu7136u540eu628au5269u4e0bu7684s-xu6254u7ed9bob
        
        # u53eau8981u5176u4e2du6709u4e00u4e2au5207u5272uff0cbobu65e0u6cd5u5728s-xu4e0bu83b7u80dcuff0cu90a3u5c31u662faliceu83b7u80dc
        # u5982u679cbobu5728u6240u6709u5207u5272u4e0bu90fdu83b7u80dcuff0cu90a3alice lose
        
        # u53eau4ecealiceu7684u89d2u5ea6u51fau53d1uff0cu662fu5426u8db3u591fuff1f
        cache = dict()
        
        def solve(s): # u9012u5f52u8c03u7528u672cu8eabuff0cu4f1au82b1u8d39u6bd4u8f83u591au7684u65f6u95f4uff01
            if s in cache: return cache[s]
            if s == 0: 
                cache[s] = False
                return False
            
            if pow(int(sqrt(s)), 2) == s: 
                cache[s] = True
                return True # s is a square number and current player can take it directly, so win
            
            iswin = False
            #for x in range(s-1, 0, -1): # from 1 to s-1, since s is not a square number, too slow if write in this way!
            #    if pow(int(sqrt(x)), 2) == x:
            #        if not solve(s-x):
            #            iswin = True
            #            break
            for k in range(1, int(sqrt(s))+1): # this can pass! great! 2612 ms, 37%
                if not solve(s - k*k):
                    iswin = True
                    break
                
            cache[s] = iswin
            return iswin
        return solve(n) # u65b9u6cd5u662fu5bf9u7684uff0cu4f46u662fu8d85u65f6u4e86uff0cn=31250u7684u65f6u5019

class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        opt = [False] * (n+1)
        for i in range(1, n+1):
            for j in range(1, int(i**0.5)+1):
                if not opt[i - j*j]:
                    opt[i] = True
                    break
        return opt[n]
                    
        

class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        
        
        dp = [False]*(n+1)
        
        for i in range(n+1):
            # we search downwards, alice can win (i.e. = True)
            # only we can take a square number away and hit a dp[False]
            # otherwise it's false
            
            # if square, we auto win
            if i == int(sqrt(i))**2:
                # print('sq', i)
                dp[i] = True
            else:
                for j in range(1, int(i**0.5)+1):
                    if not dp[i-j*j]:
                        # print(i, j*j)
                        dp[i] = True
                        break
                
        # print(dp)
        return dp[n]
class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        self.sqnums = [x ** 2 for x in range(1, 317)]
        self.cache = dict((sn, True) for sn in self.sqnums)
        
        def dp(n):
            if n in self.cache:
                return self.cache[n]
            
            x = int(sqrt(n))
            while x > 0:
                sn = self.sqnums[x-1]
                if sn >= n:
                    break

                if not dp(n - sn):
                    self.cache[n] = True
                    return True
                x -= 1
                
            return False
                
        return dp(n)

class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        ans_list=[0 for _ in range(n+1)]
        for i in range(1,n+1):
            for j in range(1,int(i**0.5)+1):
                if ans_list[i-j*j]==0:
                        ans_list[i]=1
                        break
        #print(ans_list)
        return ans_list[-1]
                
            

class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        dp=[False]*(n+1)
        dp[1]=True
        for i in range(1,n+1):
            for k in range(1,int(i**0.5)+1):
                if dp[i-k*k]==False:
                    dp[i]=True
                    break
        return dp[n]

class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        s = [i**2 for i in range(1, int(n**(1/2))+1)]
        dp = [0 for i in range(n+1)]
        dp[0], dp[1] = False, True
        
        for i in range(2, n+1):
            for j in s:
                if j > i:
                    dp[i] = False
                    break
                if dp[i-j] == False:
                    dp[i] = True
                    break
        
        return dp[-1]

class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        dp = [False] * (n + 1)
        for i in range(1, n + 1):
            for base in range(1, int(math.sqrt(i)) + 1):
                take = base * base
                if not dp[i - take]:
                    dp[i] = True
                    break
        return dp[n]
class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        @lru_cache(None)
        def dfs(stones):
            if stones <= 0:
                return False
            
            for i in reversed(list(range(1, int(sqrt(stones)) + 1))):
                square = i*i
                if stones - square == 0 or not dfs(stones - square):
                    return True
                
            return False
        
        return dfs(n)
    
    

class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        dp=[False]*(n+1)

        for i in range(1,n+1):
            for j in range(1,int(math.sqrt(i))+1):
                if dp[i-j*j]==False: dp[i]=True; break
            
        return dp[-1]
class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        
        dp = [False for i in range(n+1)]
        sqs = [i*i for i in range(1,1+int(math.sqrt(n)))]
        for i in range(n+1):
            t = False
            for sq in sqs:
                if i-sq < 0:
                    break
                if not dp[i-sq]:
                    t = True
                    break
            dp[i] = t                    
        
        return dp[n]
class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        dp=[False]*(n+1)
        for i in range(1,n+1):
            for j in range(1,int(i**0.5)+1):
                if dp[i-j*j]==False:
                    dp[i]=True
                    break
        return dp[n]
        
        
        
        
        
        
        
        
        
        

class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        dp: List[bool] = [False for _ in range(n + 1)]
        for i in range(1, n + 1):
            for j in range(1, int(i ** 0.5) + 1):
                if dp[i - j * j] == False:
                    dp[i] = True
                    break
        return dp[n]
class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        dp = [False]*(n+1)
        for i in range(1,n+1):
            for j in range(1, int(i**0.5)+1):
                if not dp[i-j*j]:
                    dp[i] = True
                    break
        return dp[n]

class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        squares = [i * i for i in range(1, int(n ** 0.5) + 1)]
        
        DP = [0] * (1 + n)
        for i in range(1, n + 1):
            can_win = False
            for s in squares:
                if s > i:
                    break
                can_win |= not DP[i - s]
                if can_win:
                    break
            DP[i] = can_win
        return DP[-1]
import math
class Solution:
    squares=[]
                
    def winnerSquareGame(self, n: int) -> bool:
        sqt=int(math.sqrt(n))
        for i in range(len(self.squares),sqt+1):
            self.squares.append(i*i)
            
        dp=[False]*(n+1)
        
        for i in range(1,n+1):
            flag=0
            for j in range(1,int(math.sqrt(i))+1):
                if not dp[i-self.squares[j]]:
                    dp[i]=True
                    flag=1
                    break
            if flag==0:
                dp[i]=False
        print(dp)
        return dp[n]
                
                

class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        dp = [False]*(n+1)
        for i in range(1, n+1):
            for k in range(1, int(i**0.5)+1):
                if dp[i-k*k] == False:
                    dp[i] = True
                    break
        return dp[n]
from functools import lru_cache
class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        sq = [x*x for x in range(1, 317)]
        dp = [False] * (n+1)
        
        for i in range(n+1):
            for x in sq:
                if i - x < 0:
                    break
                if dp[i - x] == False:
                    dp[i] = True
                    break
        
        return dp[n]

    # T F T F F

class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        @lru_cache(None)
        def helper(i):
            if i == 0:
                return False
            
            base = int(i ** 0.5)
            return any(not helper(i - j * j) for j in range(base, 0, -1))
        
        return helper(n)
class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        if int(n**0.5)**2 == n:
            return True
        
        sqdict = {i*i:1 for i in range(1,n+1)}
        dp = [False for i in range(n+1)]
        dp[:4] = [False, True, False, True]
        
        for i in range(4, n+1):
            if sqdict.get(i,0) == 1:
                dp[i] = True
            else:
                for j in sqdict:
                    if j>i: break
                    
                    if dp[i-j] == False:
                        dp[i] = True
                        break
        
        return dp[n]
                        
        

class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        if n == 1: return True
        dp = [None]*(n+1)
        
        dp[0],dp[1],dp[2] = False, True, False
        
        for i in range(3,n+1):
            for j in range(1,n*n):
                y = i - j*j
                if y<0 : 
                    break
                if not dp[y]:
                    dp[i] = True
                    break
        return dp[-1]
                
        

class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        dp = [False, True, False]
        for x in range(3, n+1):
            dp.append(False)
            for y in range(1, n):
                a = x - y * y
                if a < 0: break
                if not dp[a]:
                    dp[x] = True
                    break
        return dp[n]
class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        #squares = [i * i for i in range(1, int(n ** 0.5) + 1)]
        
        DP = [0] * (1 + n)
        for i in range(1, n + 1):
            DP[i] = not all(DP[i - j * j] for j in range(1, int(i ** 0.5) + 1))
        return DP[-1]
class Solution:
    def findWinner(self, res, sqlist, n, t, turn):
        if turn == 'Alice':
            if res[t] == 1:
                res[n] = 1
                return res
            if res[t] == 2:
                return res
            if res[t] == 0:
                ind = len(sqlist) - 1
                while ind >= 0:
                    temp = t - sqlist[ind]
                    if temp >= 0:
                        fw1 = Solution()
                        res = fw1.findWinner(res, sqlist, n, temp, 'Bob')
                        if res[temp] == 2:
                            res[t] = 1
                            res[n] = 1
                            return res
                    ind -= 1
                res[t] = 2
                return res
        if turn == 'Bob':
            if res[t] == 2:
                res[n] = 1
                return res
            if res[t] == 1:
                return res
            if res[t] == 0:
                ind = len(sqlist) - 1
                while ind >= 0:
                    temp = t - sqlist[ind]
                    if temp >= 0:
                        fw2 = Solution()
                        res = fw2.findWinner(res, sqlist, n, temp, 'Alice')
                        if res[temp] == 2:
                            res[t] = 1
                            res[n] = 2
                            return res
                    ind -= 1
                res[t] = 2
                return res
        return res
    def winnerSquareGame(self, n: int) -> bool:
        if n == 0:
            return False
        res = []
        for i in range(n + 1):
            res.append(0)
        sqlist = []
        i = 1
        isq = 1
        while isq <= n:
            sqlist.append(isq)
            res[isq] = 1
            i += 1
            isq = i ** 2
        fw = Solution()
        finres = fw.findWinner(res, sqlist, n, n, 'Alice')
        if finres[n] == 1:
            return True
        else:
            return False
class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        @lru_cache(None)
        def dfs(stones):
            if stones < 0:
                return False
            
            for i in reversed(list(range(1, int(sqrt(stones)) + 1))):
                square = i*i
                if stones - square == 0 or not dfs(stones - square):
                    return True
                
            return False
        
        return dfs(n)
    
    

class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        dp = [False] * (n + 1)
        for i in range(1, n + 1):
            dp[i] = not all(dp[i - k * k] for k in range(1, int(i**0.5 + 1)))
            
        return dp[-1]
class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        dp = [False] * (n + 1)
        for i in range(1, n + 1):
            dp[i] = not all(dp[i - k * k] for k in range(1, int(i**0.5) + 1))
        return dp[-1]
                

class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        dp = [False] * (n + 1)
        for i in range(1, n + 1):
            dp[i] = not all(dp[i - k * k] for k in range(1, int(i**0.5) + 1))
        return dp[-1]
class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        '''
        dp = [False] * (n + 1)
        for i in range(1, n+1):
            for k in range(1, i+1):
                if i - k**2 < 0: break
                if not dp[i - k**2]: dp[i] = True;break
            #print(dp)        
        return dp[-1]
        '''
        dp = [False] * (n + 1)
        for i in range(1, n + 1):
            dp[i] = not all(dp[i - k * k] for k in range(1, int(i**0.5) + 1))
        return dp[-1]   
class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        dp = [False] * (n + 1)
        for i in range(1, n + 1):
            dp[i] = any(not dp[i - j * j] for j in range(1, int(sqrt(i)) + 1))
        return dp[-1]
class Solution:
    def winnerSquareGame(self, n):
        dp = [False] * (n + 1)
        for i in range(1, n + 1):
            dp[i] = not all(dp[i - k * k] for k in range(1, int(i ** 0.5) + 1))
        return dp[-1]
from functools import lru_cache

class Solution:
    def winnerSquareGame(self, n: int) -> bool:
   
        @lru_cache(None)
        def dp(i):
            sq = int(math.sqrt(i))
            if sq**2==i:
                return True
            
            ans=False
            for m in [n**2 for n in range(1,sq+1)][::-1]:
                ans = ans or not dp(i-m)
                if ans:
                    return True
            return ans
        
        return dp(n)
            
            
        
        

class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        @lru_cache(None)
        def helper(i):
            if i in [0, 2]:
                return False
            
            base = int(i ** 0.5)
            if base * base == i:
                return True
            
            return any(not helper(i - j * j) for j in range(base, 0, -1))
        
        return helper(n)
class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        
        
        dp = [False] * (n + 1)
        for i in range(1, n + 1):
            dp[i] = not all(dp[i - k * k] for k in range(1, int(i**0.5) + 1))
        return dp[-1]
        
#         if n == 0 or n == 2:
#             return False
#         if n == 1:
#             return True
#         dp = [-1] * (n+1)
#         dp[0] = 0
#         dp[1] = 1
#         dp[2] = 0
        
#         def pick(n):
            
#             if dp[n] != -1:
#                 return dp[n]
#             i = 1
#             while i * i <= n:
#                 if i * i == n:
#                     dp[n] = 1
#                     return True
#                 if not pick(n-i*i):
#                     dp[n] = 1
#                     return True
#                 i = i+1
#             dp[n] = 0
#             return dp[n]
#         pick(n)
#         return dp[n] == 1
        
                
                
        
                
                
            
                
            
            
            
        
        
        
        
        

class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        dp = [False] * (n + 1) # true is on peut gagner le jeu avec i pierre
        for i in range(1, n + 1):
            dp[i] = not all(dp[i - k * k] for k in range(1, int(i**0.5) + 1)) # si tout gagne apru00e8s notre coup (quel qu'il soit), on perd
        return dp[-1]
class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        
        dp = [False]*(n+1)
        for i in range(n+1):
            for j in range(1,i+1):
                if j*j > i: break
                elif not dp[i-j*j]:
                    dp[i] = True
                    break
        return dp[n]

class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        dp = [False] * (n + 1)
        for i in range(1, n + 1):
            dp[i] = any(not dp[i - k * k] for k in range(1, int(i**0.5) + 1))
        return dp[n]

class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        if n == 1:
            return True
        num = 0
        dp = [False] * (1 + n)
        while num * num < n:
            dp[num * num] = True
            num += 1
        if num * num == n:
            return True
            
        for i in range(1, n + 1):
            if dp[i]:
                continue
            j = 1
            while j * j <= i:
                if not dp[i - j * j]:
                    dp[i] = True
                    break
                j += 1
        return dp[n]
                
        
        

class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        def is_square(n):
            root = math.sqrt(i)
            return root == int(root)
            
            
        dp = [False] * (n + 1)
        squares = []
        for i in range(1, n + 1):
            if is_square(i):
                squares.append(i)
            
            dp[i] = any(not dp[i - square] for square in squares)
        return dp[-1]
class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        dp = [False for _ in range(n + 1)]
        #dp[i] denotes can a player win when i stones are present in the pile
        dp[1] = True
        for i in range(2, n + 1):
            j = 1
            while j*j <= i:
                if not dp[i - j*j]:
                    dp[i] = True
                    break
                    
                j += 1
        
        # print(dp)
        return dp[n]
class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        result = []
        result.append(False)
        for i in range(1,n+1):
            j = 1
            flag = True
            while(j*j<=i):
                if not result[i - j*j]:
                    flag = False
                    break
                j += 1
            if flag==True:
                result.append(False)
            else:
                result.append(True)
        
        print(result)
        return result[-1]
from functools import lru_cache
class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        dp = [None] * (n + 1)
        dp[0] = False
        i = 1
        while i ** 2 <= n:
            dp[i**2] = True
            i += 1
        # print(dp)
        sq = 1
        for i in range(2, n + 1):
            if dp[i]: continue
                
            dp[i] = False
            sq = int(math.sqrt(i))
            for k in range(sq, 0, -1):
                if not dp[i - k**2]:
                    dp[i] = True
                    break
                        
        return dp[n]
    
#     def winnerSquareGame(self, n: int) -> bool:
#         def dp(i):
#             if i == 0: return False
#             sq = int(math.sqrt(i))
#             if sq ** 2 == i: return True
#             if i not in memo:
#                 memo[i] = False
#                 for k in range(1, sq + 1):
#                     if not dp(i - k**2):
#                         memo[i] = True
#             return memo[i]
        
#         memo = {}
        
#         return dp(n)

