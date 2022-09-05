import math
class Solution:
    def numMusicPlaylists(self, N: int, L: int, K: int) -> int:
        s=0
        c=0
        r=0
        x=math.factorial(N)
        while(True):
            c=x*((N-r-K)**(L-K))*(-1)**(r)//(math.factorial(N-r-K)*math.factorial(r))
            if(c!=0):
                s=(s+c)%(10**9+7)
                r+=1
            else:
                return s

import math
class Solution:
    def numMusicPlaylists(self, N: int, L: int, K: int) -> int:
        s=0
        c=0
        r=0
        x=math.factorial(N)
        while(True):
            c=x*((N-r-K)**(L-K))*(-1)**(r)//(math.factorial(N-r-K)*math.factorial(r))
            if(c!=0):
                s=(s+c)%(10**9+7)
                r+=1
            else:
                return s
            

class Solution:
    def numMusicPlaylists(self, N: int, L: int, K: int) -> int:
        MOD = 10 ** 9 + 7
        @lru_cache(None)
        def dp(i, j):
            if i < j: return 0
            if i == 0:
                return 1 if j == 0 else 0
            # if i == j:
            #     return math.factorial
            a = dp(i - 1, j - 1) * (N - j + 1)
            a += dp(i - 1, j) * (j - K if j > K else 0)
            return a % MOD
        return dp(L, N)
            

class Solution:
    def numMusicPlaylists(self, N: int, L: int, K: int) -> int:
        mod = 10**9+7
        def memo(f):
            dic = {}

            def f_alt(*args):
                if args not in dic:
                    dic[args] = f(*args)
                return dic[args]
            return f_alt

        @memo
        def play(N, L):
            if L == 0:
                return 1 if N == 0 else 0
            if N > L:
                return 0
            return (N*play(N-1, L-1) + max(0, N-K)*play(N, L-1))%mod

        return play(N, L)
class Solution:
    def numMusicPlaylists(self, N: int, L: int, K: int) -> int:
        dp = [[0] * (N + 1) for _ in range(L + 1)]
        dp[0][0] = 1
        for i in range(1, L + 1):
            for j in range(1, N + 1):
                dp[i][j] = dp[i - 1][j - 1] * (N - j + 1) #play new song
                if j > K: #play old song
                    dp[i][j] += dp[i - 1][j] * (j - K)
        return dp[-1][-1]%(10 ** 9 + 7)
                      
                

class Solution:
    def numMusicPlaylists(self, N: int, L: int, K: int) -> int:
        mod = 10 ** 9 + 7
        dp = [[0] * (L + 1) for _ in range(N + 1)]
        
        for i in range(1, N + 1):
            for j in range(i, L + 1):
                if i == K + 1:# or i == j:
                    dp[i][j] = math.factorial(i)
                else:
                    dp[i][j] = dp[i - 1][j - 1] * i
                    if j > i:
                        dp[i][j] += dp[i][j - 1] * (i - K)
                dp[i][j] %= mod
        # print(dp)
        return dp[N][L]

class Solution:
    def numMusicPlaylists(self, N: int, L: int, K: int) -> int:
        dp = [[0]*(N+1) for _ in range(L+1)]
        dp[0][0] = 1
        
        for i in range(1,L+1):
            for j in range(1,N+1):
                dp[i][j] = dp[i-1][j-1]*(N-j+1)%(10**9+7)
                if j > K:
                    dp[i][j] = (dp[i][j] + dp[i-1][j] * (j-K))%(10**9+7)
                    
        return dp[L][N]            
    
        #T=O(NL) S=O(NL) 
        memo = {}
        def DFS(i,j):
            if i == 0:
                return j==0
            if (i,j) in memo:
                return memo[(i,j)]
            ans = DFS(i-1, j-1)*(N-j+1)
            ans += DFS(i-1, j)* max(j-K,0)
            memo[(i,j)] = ans%(10**9+7)
            return memo[(i,j)]
        
        return DFS(L,N)
class Solution:
    def numMusicPlaylists(self, N: int, L: int, K: int) -> int:
        mod = 10 ** 9 + 7
        dp = [[0] * (L + 1) for _ in range(N + 1)]
        
        for i in range(1, N + 1):
            for j in range(i, L + 1):
                if i == K + 1 or i == j:
                    dp[i][j] = math.factorial(i)
                else:
                    dp[i][j] = dp[i - 1][j - 1] * i
                    if j > i:
                        dp[i][j] += dp[i][j - 1] * (i - K)
                dp[i][j] %= mod
        # print(dp)
        return dp[N][L]

class Solution:
    def numMusicPlaylists(self, N: int, L: int, K: int) -> int:
        dp = [[0 for _ in range(N+1)] for _ in range(L+1)]
        dp[0][0] = 1
        mod = 10**9 + 7
        for i in range(1, L+1):
            for j in range(1, N+1):
                dp[i][j] = dp[i-1][j-1] * (N-j+1)
                if j > K:
                    dp[i][j] += dp[i-1][j]*(j-K)
                dp[i][j] %= mod
                    
        return dp[L][N]
class Solution:
    def numMusicPlaylists(self, N: int, L: int, K: int) -> int:
        dp = [1] * (L-N+1)
        for p in range(2, N-K+1):
            for i in range(1, L-N+1):
                dp[i] += dp[i-1] * p
                
        ans = dp[-1]
        for k in range(2, N+1):
            ans *= k
            
        return ans % (10 ** 9 + 7)
class Solution:
    def numMusicPlaylists(self, N: int, L: int, K: int) -> int:
        # 11:35
        # pick k+1 songs out of N Songs
        # k+1 factorial
        # you can pick a new song or old song from this k+1 => basically you got N options now
        
        mod=10**9+7
        @lru_cache(None)
        def helper(i,notplayed):
            nonlocal mod
            if i==L+1:
                return 0 if notplayed!=0 else 1   
            ans=(max((N-notplayed)-K,0)*helper(i+1,notplayed))%mod
            if notplayed!=0:
                ans+=(notplayed)*helper(i+1,notplayed-1)
            return ans%mod
        return helper(1,N)
    
    
    
            
        
                
                
           
            
        
        
        
        
        

class Solution:
    def numMusicPlaylists(self, N: int, L: int, K: int) -> int:
        BIG = 10**9+7
        
        @lru_cache(None)
        def dp(r,n):
            if r == 0: return 1 if n == 0 else 0
            return ( dp(r-1,n-1) * (N-(n-1)) + dp(r-1,n) * max(0, n-K) ) % BIG
        
        
        return dp(L, N)
class Solution:
    def numMusicPlaylists(self, N: int, L: int, K: int) -> int:
        @lru_cache(None)
        def unique(length, uni):
            if uni == 0:
                return 0
            if length == 1:
                if uni == 1:
                    return N
                else:
                    return 0
            
            ret = unique(length - 1, uni - 1) * (N - uni + 1)
            ret += unique(length -1, uni) * max(0, uni - K)
            
            return ret % (10**9+7)
        
        return unique(L, N)
class Solution:
    def numMusicPlaylists(self, N: int, L: int, K: int) -> int:
        m = 10 ** 9 + 7
        dp = [[0 for _ in range(N+1)] for _ in range(L+1)]
        dp[0][0] = 1
        for i in range(1,L+1):
            for j in range(1,N+1):
                dp[i][j] = (dp[i-1][j-1] * (N-j+1) + dp[i-1][j] * max(j-K,0))%m
        
        return int(dp[L][N])
class Solution:
    def numMusicPlaylists(self, N: int, L: int, K: int) -> int:
        mod = 10 ** 9 + 7
        
        @lru_cache(None)
        def dp(l, n):
            if not l:
                return not n
            return dp(l - 1, n - 1) * (N - n + 1) + dp(l - 1, n) * max(n - K, 0)
        
        return dp(L, N) % mod
from functools import lru_cache

class Solution:
    def numMusicPlaylists(self, N, L, K):
        @lru_cache(None)
        def dp(i, j):
            if i == 0:
                return +(j == 0)
            ans = dp(i-1, j-1) * (N-j+1)
            ans += dp(i-1, j) * max(j-K, 0)
            return ans % (10**9+7)

        return dp(L, N)
class Solution:
    def numMusicPlaylists(self, N: int, L: int, K: int) -> int:
        dp = [[0 for i in range(L+1)] for j in range(N+1)]
        for i in range(1, len(dp)):
            for j in range(1, len(dp[0])):
                if i == j:
                    dp[i][j] = math.factorial(i)
                else:
                    dp[i][j] = dp[i-1][j-1]*i + dp[i][j-1]*max((i-K), 0)
        print(dp)
        return dp[N][L]%(10**9+7)
class Solution:
    def numMusicPlaylists(self, N: int, L: int, K: int) -> int:
        dp = [0 for _ in range(L + 1)]
        dp[0] = 1
        for i in range(1, N + 1):
            dp2 = [0 for _ in range (L + 1)]
            for j in range(1, L + 1):
                dp2[j] = dp[j - 1] * (N - i + 1) 
                dp2[j] += dp2[j - 1] * max(i - K, 0)
            dp = dp2
            
        return dp[L] % (10**9 + 7)
import math
class Solution:
    def numMusicPlaylists(self, N: int, L: int, K: int) -> int:
        mod = 10**9+7
        dp = [[0 for _ in range(L+1)] for _ in range(N+1)]
        dp[0][0]=1
        for i in range(1,N+1):
            for j in range(1,L+1):
                    dp[i][j] = dp[i-1][j-1]*(N-i+1)%mod
                    dp[i][j] += dp[i][j-1]*max(i-K,0)%mod
        return dp[-1][-1]%mod
class Solution:
    def numMusicPlaylists(self, N: int, L: int, K: int) -> int:
        memo = {}
        def dp(i, j):
            if i == 0:
                return j == 0
            if (i, j) in memo: return memo[i, j]
            memo[i, j] = dp(i - 1, j - 1) * (N - j + 1) + dp(i - 1, j) * max(j - K, 0)
            return memo[i, j]
        
        return dp(L, N)%(10**9 + 7)
                      
                

class Solution:
    def numMusicPlaylists(self, N: int, L: int, K: int) -> int:
        # N = 10   L = 25    K = 4
        # 1~10         24        3
        # 2~9          23        2
        #  Unique : L - N + 1       Extra : N-K+1
        
        dp = [1] * (L-N+1)
        
        for p in range(2,N-K+1):
            for i in range(1,L-N+1):
                dp[i] += p*dp[i-1]
        
        ans = dp[-1]
        for k in range(2,N+1):
            ans = ans * k
        return ans% (10**9 + 7)
        
        
        
        
        
      
        
        
        
        
        
        
        
        
        dp = [1] * (L-N+1)
        print(dp)
        for p in range(2, N-K+1):
            for i in range(1, L-N+1):
                
                dp[i] += dp[i-1] * p
                print((p,i, dp))
        # Multiply by N!
        ans = dp[-1]
        for k in range(2, N+1):
            ans *= k
        return ans % (10**9 + 7)
            

class Solution:
    def numMusicPlaylists(self, N: int, L: int, K: int) -> int:
        cache = {}
        def dp(i, j):
            if i == 0:
                return +(j == 0)
            if (i,j) in cache:
                return cache[(i,j)]
            ans = dp(i-1, j-1) * (N-j+1)
            ans += dp(i-1, j) * max(j-K, 0)
            ans %= (10**9+7)
            cache[(i,j)] = ans
            return ans

        return dp(L, N)
class Solution:
    def numMusicPlaylists(self, N: int, L: int, K: int) -> int:
        
        @lru_cache(None)
        def dp(i, j):
            if i == 0:
                return j == 0
            
            return (dp(i-1, j) * max(0, j - K) + dp(i-1, j-1) * (N - j + 1)) % (10**9 + 7)
        
        return dp(L, N)
class Solution:
    def numMusicPlaylists(self, N: int, L: int, K: int) -> int:
        
        dp = [[0 for n in range(N+1) ] for l in range(L+1)]
        dp[0][0] = 1
        for l in range(1, L+1):
            for n in range(1, N+1):
                dp[l][n] += dp[l-1][n-1] * (N-n+1)
                dp[l][n] += dp[l-1][n] * max(n-K, 0)
                dp [l][n] = dp [l][n] %  (10 **9+7)
        return dp[L][N]
class Solution:
    def numMusicPlaylists(self, N: int, L: int, K: int) -> int:
        dp = [[0] * (N + 1) for _ in range(L + 1)]
        dp[0][0] = 1;
        for l in range(1, L + 1):
            for n in range(1, N + 1):
                dp[l][n] += dp[l - 1][n - 1] * (N - n + 1)
                dp[l][n] += dp[l - 1][n] * max(n - K, 0)
                dp[l][n] = dp[l][n] % (1000000007)
        return dp[L][N]
class Solution:
    def numMusicPlaylists(self, N: int, L: int, K: int) -> int:
        mod = 10**9 + 7
        memo = {}
        # helper(i,j) returns the number of playlists of len i
        # that have exactly j unique songs
        def helper(i, j):
            nonlocal N, K
            if i==0:
                if j==0:
                    # base case
                    # helper(0,0) returns 1
                    return 1
                else:
                    return 0
            if (i,j) in memo:
                return memo[(i,j)]
            ans = 0
            # the jth song is unique,
            # then the jth song has (N-(j-1)) possibilities
            ans += helper(i-1, j-1)*(N-(j-1))
            # the jth song is not unique
            # it is the same as one of the previous songs
            # then the jth song has max(0, j-K) possibilities
            # since it can be the same as the previous K songs
            ans += helper(i-1, j)*max(0, j-K)
            memo[(i,j)]=ans%mod
            return ans%mod
        return helper(L, N)
class Solution:
    def numMusicPlaylists(self, N: int, L: int, K: int) -> int:
        memo = {}
        def dp(i, j):
            if i == 0: return 1 if j == 0 else 0
            if (i, j) in memo: return memo[(i, j)]
            # non repeat
            ans = dp(i - 1, j - 1) * (N - (j - 1))
            # repeat
            ans += dp(i - 1, j) * max(0, j - K)
            memo[(i, j)] = ans % (10 ** 9 + 7)
            return memo[(i, j)]
        return dp(L, N)
class Solution:
    def numMusicPlaylists(self, N: int, L: int, K: int) -> int:
        memo = {}
        def dp(i, j):
            if i == 0: return 1 if j == 0 else 0
            if (i, j) in memo: return memo[(i, j)]
            # non repeat
            ans = dp(i - 1, j - 1) * (N - (j - 1))
            # repeat
            ans += dp(i - 1, j) * max(0, j - K)
            memo[(i, j)] = ans % (10 ** 9 + 7)
            return memo[(i, j)]
        return dp(L, N)
# from functools import lru_cache

# class Solution:
#     def numMusicPlaylists(self, N, L, K):
#         @lru_cache(None)
#         def dp(i, j):
#             if i == 0:
#                 return +(j == 0)
#             ans = dp(i-1, j-1) * (N-j+1)
#             ans += dp(i-1, j) * max(j-K, 0)
#             return ans % (10**9+7)

#         return dp(L, N)

class Solution:
    def numMusicPlaylists(self, N: int, L: int, K: int) -> int:
        @lru_cache(None)
        def dp(i, j):
            if i == 0:
                return +(j == 0)
            ans = dp(i-1, j-1) * (N-j+1)
            ans += dp(i-1, j) * max(j-K, 0)
            return ans % (10**9+7)

        return dp(L, N)
from functools import lru_cache
class Solution:
    def numMusicPlaylists(self, N, L, K):
        @lru_cache(None)
        def dp(i, j):
            if i == 0:
                return +(j == 0)
            ans = dp(i-1, j-1) * (N-j+1)
            ans += dp(i-1, j) * (j-min(K, i-1))
            return ans % (10**9+7)

        return dp(L, N)
            
            

from functools import lru_cache

class Solution:
    def numMusicPlaylists(self, N: int, L: int, K: int) -> int:
        # Dynamic Programming
        # Let dp[i][j] be the number of playlists of length i that have exactly j unique songs.
        # Time  complexity: O(NL)
        # Space complexity: O(NL)
        @lru_cache(None)
        def dp(i, j):
            if i == 0:
                return +(j == 0)
            ans = dp(i - 1, j - 1) * (N - j + 1)
            ans += dp(i - 1, j) * max(j - K, 0)
            return ans % (10**9 + 7)

        return dp(L, N)


class Solution:
    def numMusicPlaylists(self, N: int, L: int, K: int) -> int:
        dp = [[0 for j in range(L+1)] for i in range(N+1)]
        for i in range(K+1, N+1):
            for j in range(i, L+1):
                if i == j or i == K+1:
                    dp[i][j] = math.factorial(i)
                else:
                    dp[i][j] = dp[i-1][j-1]*i + dp[i][j-1]*(i-K)
        return dp[N][L]%(10**9+7)

class Solution:
    def numMusicPlaylists(self, N: int, L: int, K: int) -> int:
        '''
         construct a 2d dp[i][j] where i is i different songs and
         j is the length of the playlist, also track the remaining songs r:
         for dp update, we have two options:
         if i <= k:
            1. add a new song to the list, r -= 1
         else:
            if r > L-j
                1. add a new song to the list, r -= 1
                2. add an existing song
            else:
                1. add a new song
         ''' 
        @lru_cache(None)
        def dp(unique, total, r):
            if total == L:
                return 1
            if unique <= K:
                return r * dp(unique+1, total+1, r-1)
            else:
                ans = 0
                if r < L-total:
                    # add an existing song
                    ans += (unique-K) * dp(unique, total+1, r)
                # add a new song
                ans += r * dp(unique+1, total+1, r-1)
                return ans

        return dp(0, 0, N) % (10**9+7)
class Solution:
    def numMusicPlaylists(self, N: int, L: int, K: int) -> int:
        dp = [[0 for i in range(L + 1)] for j in range(N + 1)]
        for i in range(K + 1, N + 1):
            for j in range(i, L + 1):
                if i == j or i == K + 1:
                    dp[i][j] = math.factorial(i)
                else:
                    dp[i][j] = dp[i - 1][j - 1] * i + dp[i][j - 1] * (i - K)
        return dp[N][L] % (10**9 + 7)
class Solution:
    import math
    from functools import lru_cache
    def numMusicPlaylists(self, N: int, L: int, K: int) -> int:
        MOD = 1000000007

        @lru_cache(maxsize=None)
        def rec(n, l):
            if l < n or n <= K:
                return 0
            elif l == n:
                return math.factorial(n) % MOD
            return (n * rec(n - 1, l - 1) + (n - K) * rec(n, l - 1)) % MOD

        return rec(N, L)
class Solution:
    def numMusicPlaylists(self, N: int, L: int, K: int) -> int:
        MOD = 10 ** 9 + 7
        @lru_cache(None)
        def dp(i, j):
            if i < j: return 0
            if i == 0:
                return 1 if j == 0 else 0
            if i == j:
                return (math.factorial(N) // math.factorial(N - j)) % MOD
            a = dp(i - 1, j - 1) * (N - j + 1)
            a += dp(i - 1, j) * (j - K if j > K else 0)
            return a % MOD
        return dp(L, N)
            

