from collections import Counter
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        c = dict(Counter(A).most_common())
        # print(c)
        m1 = max(c.values())
        # A = list(set(A))
        # A.sort()
        index = {}
        # for i in range(len(A)):
            # index[A[i]]=i
        dp = [[2] * len(A) for i in A]
        m = 2
        for i in range(len(A)):
            # print("I=", i)
            # index[A[i+1]]=(i+1)
            for j in range(i+1, len(A)):
                # index[A[j]]=(j)
                a = A[i]
                
                c = A[j]
                b = 2 * a - c
                # print(b,a,c)
                if b in index :
                    # print("B {} in index ".format(b))
                    # print(b,a,c,i,j)
                    dp[i][j] = dp[index[b]][i] + 1
            index[A[i]]=i
            m = max(m, max(dp[i]))
        # # print(A)
        # for i,d in enumerate(dp):
        #     print(A[i],d)
        return max(m,m1)
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        length = len(A)
        onleft = set()
        onleftl = []
        onright = [0 for _ in range(501)]
        toextend = [{} for _ in range(501)]
        res = 2
        for v in A:
            onright[v] += 1
        for i in range(0, length):
            val = A[i]
            tex = toextend[val]
            onright[val] -= 1
            for lval in onleftl:
                diff = val - lval
                nextval = val + diff
                if nextval > 500 or nextval < 0 or onright[nextval] == 0:
                    continue
                if diff in tex:
                    c = tex[diff] + 1
                else:
                    c = 3
                if c > res:
                    res = c
                toextend[nextval][diff] = c
            if val not in onleft:
                onleftl.append(val)
                onleft.add(val)
        return res
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        n = len(A)
        
        dp = [[2 for c in range(n)] for r in range(n)]
        
        visited = {}
        res = 2
        
        for i in range(n - 1):
            for j in range(i + 1, n):
                
                prev = A[i] * 2 - A[j]
                
                if prev < 0 or prev not in visited:
                    continue
                
                dp[i][j] = dp[visited[prev]][i] + 1
            
                res = max(res, dp[i][j])
            
            visited[A[i]] = i
        
        return res

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        res, n = 1, len(A)
        dp = [{} for _ in range(n)]
        for i in range(1, n):
            for j in range(i-1, -1, -1):
                d = A[i] - A[j]
                if d in dp[i]: continue
                if d in dp[j]:
                    dp[i][d] = dp[j][d]+1
                else:
                    dp[i][d] = 2 
                res = max(res, dp[i][d])
        # print(dp)        
        return res

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        dp = {}
        maxValue = 1;
        for i in range(len(A)):
            for j in range(0, i):
                dp[i, A[i] - A[j]] = dp.get((j, A[i] - A[j]), 0) + 1
                maxValue = max(maxValue, dp[i, A[i] - A[j]])
        return maxValue + 1
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        dp=[dict() for a in A]
        for idx, a in enumerate(A):
            for j in range(idx):
                diff=a-A[j]
                dp[idx][diff]=dp[j].get(diff,1)+1
        
        def get_len(d):
            if not d:
                return 0
            return max(d.values())
        
        return max(map(get_len,dp))
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        a = len(A)
        dp = [[0]*a for _ in range(a)] # dp array
        index = [-1]*20001#index array
        maximum = 2
        for i in range(0,a):
            dp[i] = [2]*a
            for j in range(i+1, a):
                first = A[i]*2-A[j]
                if first < 0 or index[first]==-1:
                    continue
               
                dp[i][j] = dp[index[first]][i]+1
                maximum = max(maximum,dp[i][j] ) 
                
            index[A[i]] = i
        return maximum

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        if not A:
            return 0
        if len(A) == 1 or len(A) == 2:
            return len(A)
        D = [dict() for _ in range(len(A))]
        ans = 0
        for i, a in enumerate(A[1:], 1):
            for j in range(i):
                if a - A[j] not in D[j]:
                    D[i][a - A[j]] = 2
                else:
                    D[i][a - A[j]] = D[j][a - A[j]] + 1
            ans = max(ans, max(D[i].values()))
        return ans
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        
        if len(A) < 3:
            return len(A)
        
        dp = [ {} for i in range(len(A))]
        m = 2
        
        for i in range(1, len(A)):
            for j in range(i):  # here we have to iterate from 0 to i-1 and not i-1 to 0.
                
                diff = A[i] - A[j]
                if diff in dp[j]:
                    dp[i][diff] = dp[j][diff]+1
                    if m < dp[i][diff]:
                        m = dp[i][diff]
                else:
                    dp[i][diff] = 2

        return m
                
                
                
                
                
                
                
                
                
                

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        a = len(A)
        dp = [[0]*a for _ in range(a)]
        index = [-1]*20001
        maximum = 2
        for i in range(0,a):
            dp[i] = [2]*a
            for j in range(i+1, a):
                first = A[i]*2-A[j]
                if first < 0 or index[first]==-1:
                    continue
                else:
                    dp[i][j] = dp[index[first]][i]+1
                    maximum = max(maximum,dp[i][j] ) 
            index[A[i]] = i
        return maximum
    
    

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        n = len(A)
        
        dp = [{} for _ in range(n)]
        
        ans = 0
        
        for i in range(n):
            dp[i][0] = 1
            for j in range(i):
                diff = A[i] - A[j]
                
                if diff not in dp[j]:
                    dp[i][diff] = 2
                else:
                    dp[i][diff] = dp[j][diff] + 1
            
            ans = max(ans, max([dp[i][key] for key in dp[i]]))
        
        return ans
from collections import defaultdict
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        highest = 0
        offsets = [None] * len(A)
        for i in range(len(A)):
            offsets[i] = defaultdict(int)
        for i in range(len(A) - 1,-1,-1):
            # go forwards
            for j in range(i,-1,-1):
                offset = A[i] - A[j]
                if offset == 0:
                    continue
                offsets[i][offset] = 1
            # go backwards
            seen_offsets = set()
            for j in range(i,len(A)):
                offset = (A[i] - A[j]) * -1
                if offset == 0 or offset in seen_offsets:
                    continue
                seen_offsets.add(offset)
                # increment only for the first time we've seen this offset going back
                offsets[i][offset] += offsets[j][offset]
                if offsets[i][offset] > highest:
                    highest = offsets[i][offset]
        #for offset in offsets:
        #    print(offset)
        return highest + 1
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int: 
        dp = []
        
        for i, x in enumerate(A):
            nd = collections.defaultdict(int)
            dp.append(nd)
            for j in range(i):
                curr_diff = x - A[j]
                dp[i][curr_diff] = dp[j][curr_diff] + 1
          
        return max(max(y.values()) for y in dp) + 1
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        ALen = len(A)
        dictList = [defaultdict(lambda: 1) for _ in range(ALen)]
        ans = 2
        for i, num in enumerate(A):
            for j in range(i-1, -1, -1):
                delta = A[i] - A[j]
                if delta not in dictList[i]:
                    dictList[i][delta] = 1 + dictList[j][delta]
                    ans = max(ans, dictList[i][delta])
        return ans
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        
        from collections import defaultdict

        d = [{} for _ in range(len(A))]
        res = 2

        for i, x in enumerate(A):
            for j in range(i):
                diff = x - A[j]
                if diff in d[j]:
                    d[i][diff] = d[j][diff] + 1
                    # d[j].pop(diff)
                    
                    res = max(res, d[i][diff])
                    
                else:
                    d[i][diff] = 2

                
        return res


class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        
        dp = []
        
        for i, x in enumerate(A):
            nd = collections.defaultdict(int)
            dp.append(nd)
            for j in range(i):
                curr_diff = x - A[j]
                dp[i][curr_diff] = dp[j][curr_diff] + 1
          
        return max(max(y.values()) for y in dp) + 1
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        n = len(A)

        if n <= 1:
            return n 
        
        ap = [None] * n
        for i in range(n):
            ap[i] = dict()

        for j in range(1, n):
            for i in range(0, j):
                diff = A[j] - A[i]
                ap[j][diff] = ap[i].get(diff, 1) + 1

        ans = 0

        for item in ap[1:]:
            vals = max(item.values())
            ans = max(ans, vals)

        return ans 
class Solution:
    def longestArithSeqLength(self, nums: List[int]) -> int:
        n = len(nums)
        
        if n == 2:
            return n
        
        dp = [{} for i in range(n)]
        res = 0
        for i in range(1, n):
            for j in range(i):
                dis = nums[i] - nums[j]
                # u5728u524du9762u7684dp[j]u4e2d u7528get()u5bfbu627edpu4e2du5df2u6709u7684u6570u636eu3002
                # u5982u679cu6ca1u6709uff0cu8bf4u660eu662fu57fau672cu72b6u6001uff0cu75281u521du59cbu5316uff0c+1u4e3a2
                x = dp[j].get(dis, 1)+1
                dp[i][dis] = x
            res = max(res, max(dp[i].values()))

        return res


from collections import defaultdict

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        N = len(A)
        dp = [defaultdict(lambda: 1) for _ in range(N)]
        for i in range(N):
            for j in range(i+1, N):
                diff = A[j] - A[i]
                dp[j][diff] = dp[i][diff] + 1 
        return max([max(d.values()) for d in dp])
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        length = len(A)
        onleft = set()
        onleftl = []
        onright = [0 for _ in range(501)]
        toextend = [{} for _ in range(501)]
        res = 2
        for v in A:
            onright[v] += 1
        for i in range(0, length):
            val = A[i]
            tex = toextend[val]
            onright[val] -= 1
            for lval in onleft:
                diff = val - lval
                nextval = val + diff
                if nextval > 500 or nextval < 0 or onright[nextval] == 0:
                    continue
                if diff in tex:
                    c = tex[diff] + 1
                else:
                    c = 3
                if c > res:
                    res = c
                toextend[nextval][diff] = c
            if val not in onleft:
                onleftl.append(val)
                onleft.add(val)
        return res
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        memo = [{} for _ in range(len(A))] 
        for i in range(len(A)):
            for j in range(i):
                diff = A[i] - A[j]
                memo[i][diff] = 1 + memo[j].get(diff, 1)
                #result = max(result, memo[i][diff])
        return max(d[diff] for d in memo for diff in d)
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        
        table = []
        for i, z in enumerate(A):
            table.append(collections.defaultdict(lambda: 1))
            for j in range(i):
                diff = z - A[j]
                table[i][diff] = table[j][diff] + 1
        
        return max(max(y.values()) for y in table)
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int: 
        dp = []
        
        for i, x in enumerate(A):
            nd = collections.defaultdict(int)
            dp.append(nd)
            for j in range(i):
                curr_diff = x - A[j]
                dp[i][curr_diff] = dp[j][curr_diff] + 1
          
        return max(max(y.values()) for y in dp) + 1

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        dp = dict()

        for i in range(len(A)):
            for j in range(i):
                diff = A[i] - A[j]
                if diff not in dp:
                    dp[diff] = {i: 2}
                else:
                    dic = dp[diff]
                    if j in dic:
                        dic[i] = dic[j] + 1
                    else:
                        dic[i] = 2

        return max(max(v1 for k1, v1 in v.items()) for k, v in dp.items())
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        if len(A)<2: return len(A)
        table = []
        result = 1
        
        for i, z in enumerate(A):
            table.append(collections.defaultdict(lambda: 1))
            for j in range(i):
                diff = z - A[j]
                table[i][diff] = table[j][diff] + 1
                #if table[i][diff] > result: result = table[i][diff]
        
        return max([max(y.values()) for y in table])
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int: 
        dp = []
        
        for i, x in enumerate(A):
            nd = collections.defaultdict(lambda: 1)
            dp.append(nd)
            for j in range(i):
                curr_diff = x - A[j]
                dp[i][curr_diff] = dp[j][curr_diff] + 1
          
        return max(max(y.values()) for y in dp)
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        cache = [{} for i in range(len(A))]
        
        n = len(A)
        for i in range(1,n):
            for j in range(i):
                diff = A[i] - A[j]
                if diff not in cache[j]:
                    cache[i][diff] = 2
                else:
                    cache[i][diff] = cache[j][diff] + 1

        m = 0
        for dictionary in cache:
            if dictionary:
                m = max(m, max(dictionary.values()))
        return m
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        n = len(A)
        # out = [1] * n
        outdict = {}
        longseq = 0
        for i in range(n):
            for j in range(0, i):
                diff = A[i] - A[j]
                if diff not in outdict:
                    outdict[diff] = [1] * n
                pointer = outdict[diff] 
                pointer[i] = max(pointer[i], pointer[j] + 1)
                longseq = max(longseq, pointer[i])
        # print(longseq)
        return longseq
                

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        memo = [{} for _ in range(len(A))] 
        for i in range(len(A)):
            for j in range(i):
                diff = A[i] - A[j]
                memo[i][diff] = 1 + memo[j].get(diff, 1)
        return max(d[diff] for d in memo for diff in d)
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        d = {}
        dd = [{} for i in range(len(A))]
        m = 0
        for i in range(len(A)):
            
            for j in range(i + 1, len(A)):
                diff = A[j] - A[i]
                #if diff not in d:
                #    d[diff] = 0
                if diff not in dd[i]:
                    dd[j][diff] = 1
                else:
                    dd[j][diff] = dd[i][diff] + 1
                    
                if dd[j][diff] > m:
                    m = dd[j][diff]
                
                #d[diff] += 1
        #if not d:
        #    return 0
        #print(d)
        #return max(d.values()) + 1
        return m + 1
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        length = len(A)
        onleft = set()
        onleftl = []
        onright = [0] * 501 #[0 for _ in range(501)]
        toextend = [{} for _ in range(501)]
        res = 2
        for v in A:
            onright[v] += 1
        for i in range(0, length):
            val = A[i]
            tex = toextend[val]
            onright[val] -= 1
            for lval in onleftl:
                diff = val - lval
                nextval = val + diff
                if nextval > 500 or nextval < 0 or onright[nextval] == 0:
                    continue
                if diff in tex:
                    c = tex[diff] + 1
                else:
                    c = 3
                if c > res:
                    res = c
                toextend[nextval][diff] = c
            if val not in onleft:
                onleftl.append(val)
                onleft.add(val)
        return res
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        dp = dict()

        for i in range(len(A)):
            for j in range(i):
                diff = A[i] - A[j]
                if diff not in dp:
                    dp[diff] = {i: 2}
                else:
                    dic = dp[diff]
                    if j in dic:
                        dic[i] = dic[j] + 1
                    else:
                        dic[i] = 2

        return max(max(v1 for k1, v1 in list(v.items())) for k, v in list(dp.items()))

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        cache = [{} for i in range(len(A))]
        m = 0

        n = len(A)
        for i in range(1,n):
            for j in range(i):
                diff = A[i] - A[j]
                if diff not in cache[j]:
                    cache[i][diff] = 2
                else:
                    cache[i][diff] = cache[j][diff] + 1
                if cache[i][diff] > m:
                    m = cache[i][diff]

        return m
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        dp = [{} for _ in range(len(A))]
        maxSequence = 0
        for i in range(len(A)):
            for j in range(i):
                diff = A[i] - A[j]
                dp[i][diff] = val = dp[j].get(diff,1) + 1
                if val > maxSequence:
                    maxSequence = val
        return maxSequence
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        dp = []
        for i in range(len(A)):
            nd = collections.defaultdict(int)
            dp.append(nd)
            for j in range(i):
                curr_diff = A[i] - A[j]
                dp[i][curr_diff] = dp[j][curr_diff] + 1
        maxVal = -99999
        for dt in dp:
            maxVal = max(maxVal, max(dt.values()) + 1)
        return maxVal
class Solution(object):
    def longestArithSeqLength(self, A):
        if not A:
            return None
        dp = [{} for i in range(0,len(A))]
        for i in range(len(A)):
            if i == 0:
                dp[i][0]= 1
            else:
                for j in range(0,i):
                    diff = A[i]-A[j]
                    if diff in dp[j]:
                        dp[i][diff] = dp[j][diff]+1
                    else:
                        dp[i][diff] = 2
        mx = 2
        for j in range(len(A)):
            for i in dp[j]:
                mx = max(dp[j][i],mx)
        return mx
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        if not A:
            return 0
        
        dp = []
        for _ in range(len(A)):
            dp.append(dict())
        dp[0][0] = 1
        
        for i in range(1, len(A)):
            dp[i][0] = 1
            for j in range(i):
                # continue subsequence
                diff = A[i] - A[j]
                if diff in dp[j]:
                    if diff not in dp[i]:
                        dp[i][diff] = dp[j][diff] + 1
                    else:
                        dp[i][diff] = max(dp[i][diff], dp[j][diff] + 1)
                    
                # start new subsequence
                else:
                    dp[i][diff] = 2
                    
        # for x in dp:
        #     print(str(x))
        return max([max(x.values()) for x in dp])

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        # subsequence problem -> dp
        # dp[i][j] -- length of arithmetic subsequence ending at ith and jth element
        ans = 2
        n = len(A)
        index = {}
        dp = [[2] * n for i in range(n)]
        
        for i in range(n-1):
            for j in range(i+1, n):
                first = A[i] * 2 - A[j]
                if first in index:
                    dp[i][j] = dp[index[first]][i] + 1
                    ans = max(ans, dp[i][j])
            index[A[i]] = i
        return ans
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        answer = 2
        
        L = len(A)
        table = [dict() for _ in range(L)]
        
        for i in range(1, L):
            for j in range(0, i):
                diff = A[i] - A[j] 
                if not diff in table[j]:
                    table[i][diff] = 2
                else:
                    
                    table[i][diff] = table[j][diff] + 1
                    answer = max(answer, table[i][diff])
                

        return answer
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        # dp[i][j]: for 0-ith elements, the length of subsquence when step = j
        # dp[i][j] = dp[i-1][A[i]-A[k]] + 1 where k = 0, 1, ...i-1
        # return max(dp[n-1][j])
        # base case dp[0][0] = 1
        
        N = len(A)
        dp = [{} for _ in range(N)]
        for i in range(1, N):
            for j in range(0, i):
                dp[i][A[i]-A[j]] = dp[j].get(A[i]-A[j], 0) + 1
        max_len = 0
        for i in range(1, N):
            max_len = max(max_len, max(dp[i].values()))
        return max_len + 1
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        length = len(A)
        onleft = [False for _ in range(501)]
        onleftl = []
        onright = [0 for _ in range(501)]
        toextend = [{} for _ in range(501)]
        res = 2
        for v in A:
            onright[v] += 1
        for i in range(0, length):
            val = A[i]
            tex = toextend[val]
            onright[val] -= 1
            for lval in onleftl:
                diff = val - lval
                nextval = val + diff
                if nextval > 500 or nextval < 0 or onright[nextval] == 0:
                    continue
                if diff in tex:
                    c = tex[diff] + 1
                else:
                    c = 3
                if c > res:
                    res = c
                toextend[nextval][diff] = c
            if not onleft[val]:
                onleftl.append(val)
                onleft[val] = True
        return res
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        
        if len(A) < 2:
            return len(A)
        
        dp = [ {} for i in range(len(A))]
        m = 2
        
        for i in range(1, len(A)):
            for j in range(0, i):
                
                diff = A[i] - A[j]
                if diff in dp[j]:
                    dp[i][diff] = dp[j][diff]+1
                    m = max(m, dp[i][diff])
                
                else:
                    dp[i][diff] = 2

        return m
                
                
                
                
                
                
                
                
                
                

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        dp = []
        for _ in range(len(A)):
            dp.append({})
        max_ = 0
        dp[0][0] = 1
        for i in range(1, len(A)):
            for j in range(0, i):
                diff = A[i] - A[j]
                if diff in dp[j]:
                    dp[i][diff] = dp[j][diff] + 1
                else:
                    dp[i][diff] = 2
                if dp[i][diff] > max_:
                    max_ = dp[i][diff]
        return max_
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        
        '''
        9  4     7     2     10
          -5     4 2  -5 2   8
                -2 2  -2 3   3
                      -7 2   6
                             1
           j      i 
        
        
        could sort array. 
        
        could iterate through jump sizes
        could transition each elem to distance from other.  
        '''
        if len(A) < 3:
            return len(A)
        
        best = 2
        sequences = [ {} for _ in A]
        for right in range(1, len(A)):
            for left in range(right):
                diff = A[right] - A[left]
                #print(diff, sequences[left])
                if diff in sequences[left]:
                    count = sequences[left][diff] + 1
                    sequences[right][diff] = count
                    best = max(best, count)
                else: 
                    sequences[right][diff] = 2
        
        return best

        
        '''
        
        best = 2
        for i in range(len(A)-2):
            for j in range(len(A)-1):
                jump = A[j] - A[i]
                last = A[j]
                thisCount = 2
                for k in range(j+1, len(A)):
                    if A[k] == last + jump:
                        thisCount += 1
                        last = A[k]
                best = max(best, thisCount)
        return best
        '''
                        

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        if len(A) == 1:
            return 1
        max_l = 2
        dp = [{0: 1}]
        for i in range(1, len(A)):
            dp.append({})
            for j in range(0, i):
                idx = A[i] -A[j]
                if idx in dp[j]:
                    dp[i][idx] = dp[j][idx] + 1
                    max_l = max(max_l, dp[i][idx])
                else:
                    dp[i][idx] = 2
        return max_l
from collections import defaultdict
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        '''
        # Note:
            A.length >= 2
        # Analysis:
            Arithmetic sequence
        
        '''
        n = len(A)
        res = 2
        dif_arr = [defaultdict(int) for _ in range(n)]
        for i in range(1, n):
            for j in range(i):
                dif = A[i] - A[j]
                if dif in dif_arr[j]:
                    dif_arr[i][dif] = dif_arr[j][dif] + 1
                    res = max(res, dif_arr[i][dif]+1)
                else:
                    dif_arr[i][dif] = 1
                
        return res
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        
        if len(A) < 2:
            return len(A)
        
        dp = [ {} for i in range(len(A))]
        m = 2
        
        for i in range(1, len(A)):
            for j in range(0, i):
                
                diff = A[i] - A[j]
                if diff in dp[j]:
                    dp[i][diff] = dp[j][diff]+1
                    m = max(m, dp[i][diff])
                
                else:
                    if diff not in dp[i]:
                        dp[i][diff] = 2

        return m
                
                
                
                
                
                
                
                
                
                

from collections import defaultdict

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        # mapping: idx -> (mapping: arithmetic_step -> longest_arithmetic_subsequence_with_this_step_ending_at_idx
        d = defaultdict(lambda: defaultdict(lambda: 1))
        
        for i,a in enumerate(A):
            # LAS: A[i]
            d[i][0] = 1
            for j in range(i):
                # Consider each subsequence that ends at i ~        A[?] ... A[??] A[j] A[i]
                # A[i] - A[j] denotes the step
                # LSA(j, step) := length of LSA ending at j with progression equal to step
                # We only care about count, not the actual sequence, so length of such subsequence will be: 1 + LSA(j, step)
                step = A[i] - A[j]
                d[i][step] = d[j][step] + 1
        return max([max(dn.values()) for dn in d.values()])
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        # d subproblem
        # index, difference
        D = []
        for i in range(1001):
            D.append([0]* 1002) # first 501 is pos, second 501 is neg difference
        best = 0
        for second in range(len(A)-1, -1, -1):
            for first in range(second-1, -1, -1):
                diff = A[second]-A[first]
                if diff < 0:
                    diff = 500 + -1 * diff
                D[first][diff] = D[second][diff] + 1
                if D[first][diff] > best:
                    best = D[first][diff]
                    # print(f'best: {best}, first: {first}, diff: {diff}')
        # print(D[0][501+5])
        return best + 1

from collections import defaultdict

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        # mapping: 
        d = defaultdict(lambda: defaultdict(lambda: 1))
        
        for i,a in enumerate(A):
            d[i][0] = 1
            for j in range(i):
                step = A[i] - A[j]
                d[i][step] = d[j][step] + 1
        return max([max(dn.values()) for dn in d.values()])
from itertools import repeat

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        length = len(A)
        onleft = 0
        onleftl = []
        onright = {}
        toextend = [{} for _ in range(501)]
        res = 2
        for v in A:
            onright[v] = onright.get(v, 0) + 1
        for i in range(0, length):
            val = A[i]
            tex = toextend[val]
            onright[val] -= 1
            for lval in onleftl:
                diff = val - lval
                nextval = val + diff
                if onright.get(nextval, 0) == 0:
                    continue
                if diff in tex:
                    c = tex[diff] + 1
                else:
                    c = 3
                if c > res:
                    res = c
                toextend[nextval][diff] = c
            b = (1<<val)
            if not (onleft & b):
                onleftl.append(val)
                onleft = (onleft | b)
        return res
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        memo = {} # storing next Item -> (diff pattern, length so far)
        # [a, b, c]
        maxLength = 2
        
        if len(A) < 3:
            return len(A)
            
        
        for i in range(len(A)): # iterating over A
            if A[i] in memo:
                toIter = [(i, j) for i, j in list(memo[A[i]].items())]
                del memo[A[i]]
                for k in toIter:
                    diff, length = k
                    if length > maxLength:
                        maxLength = length
                    length += 1

                    newKey = A[i] + diff
                    if newKey not in memo:
                        memo[newKey] = {}
                    if diff in memo[newKey]:
                        memo[newKey][diff] = max(length, memo[newKey][diff])
                    else:
                        memo[newKey][diff] = length
            for j in range(i):
                diff = A[i] - A[j]
                newKey = A[i] + diff
                if A[i] + diff not in memo:
                    memo[newKey] = {}
                if diff not in memo[newKey]:
                    memo[newKey][diff] = 3
            
                    
        return maxLength

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        dp = [defaultdict(int) for a in A]
        for i,a in enumerate(A):
            for j in range(i):
                dp[i][a-A[j]]=dp[j][a-A[j]]+1
        #print(dp)
        m = 0
        for d in dp:
            x = d.values()
            if x: m=max(m, max(x))
        return m+1
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        dp = {}
        
        for i in range(1, len(A)):
            item = A[i]
            for j in range(0, i):
                d = item - A[j]
                
                if (j, d) in dp:
                    dp[i, d] = dp[j, d] + 1
                else:
                    dp[i, d] = 2
        
        return max(dp.values())
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        d = collections.defaultdict(dict)
        n = len(A)
        res = 1
        for i in range(n):
            for j in range(i):
                v = A[i] - A[j]
                if v not in d[j]: d[j][v] = 1
                if v not in d[i]: d[i][v] = 0
                d[i][v] = max(d[i][v] ,d[j][v] + 1)
                res = max(res, d[i][v])
        return res
from collections import defaultdict

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        d = defaultdict(lambda: defaultdict(lambda: 1))
        
        for i,a in enumerate(A):
            d[a][0] = 1
            for j in range(i):
                d[a][A[i] - A[j]] = d[A[j]][A[i]-A[j]] + 1
        return max([max(dn.values()) for dn in d.values()])
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        dp = {}
        for i, Ai in enumerate(A):
            for j in range(i+1, len(A)):
                b = A[j] - Ai
                if (i,b) not in dp: dp[j,b] = 2
                else              : dp[j,b] = dp[i,b] + 1
        return max(dp.values())
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        
        def dp_matrix_based():
            item_dict = collections.defaultdict(list)
            for i, x in enumerate(A):
                item_dict[x].append(i)

            C = max(A) - min(A)
            n = len(A)
            maxlen = -math.inf
            dp =[[-math.inf]*(2*C+1) for _ in range(n)]

            for i in range(n):
                dp[i][0 + C] = 1

            for i in range(n):
                for j in range(i+1, n):
                    g = A[j] - A[i] + C 
                    dp[j][g] = 2

            for i in range(1,n):
                for gap in range(2*C+1):
                    candidate = A[i] - gap + C
                    if candidate in item_dict:
                        for t in item_dict[candidate]:
                            if t < i:
                                dp[i][gap] = max(dp[i][gap], dp[t][gap] + 1)
                                maxlen = max(maxlen, dp[i][gap])

            return maxlen
        
        # return dp_matrix_based()
        
        def dict_based():
            '''
            Less space and simpler
            '''
            
            dp = defaultdict(defaultdict)
            for i in range(len(A)):
                for j in range(i):
                    diff = A[i] - A[j]
                    if diff not in dp:
                        #save an inner dictionary with the higher index
                        dp[diff] = { i: 2 }
                    else:
                        dic = dp[diff]
                        if j not in dic:
                            dic[i] = 2
                        else:
                            dic[i] = dic[j] + 1
            maxlen = 0
            for k,v in list(dp.items()):
                for k1, v1 in list(v.items()):
                    maxlen = max(maxlen, v1)
            return maxlen
        return dict_based()
            
            

from array import array
from itertools import repeat

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        length = len(A)
        onleft = set()
        onleftl = []
        onright = array('H', repeat(0, 501))
        toextend = [{} for _ in range(501)]
        res = 2
        for v in A:
            onright[v] += 1
        for i in range(0, length):
            val = A[i]
            tex = toextend[val]
            onright[val] -= 1
            for lval in onleftl:
                diff = val - lval
                nextval = val + diff
                if nextval > 500 or nextval < 0 or onright[nextval] == 0:
                    continue
                if diff in tex:
                    c = tex[diff] + 1
                else:
                    c = 3
                if c > res:
                    res = c
                toextend[nextval][diff] = c
            if val not in onleft:
                onleftl.append(val)
                onleft.add(val)
        return res
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        dp = []
        for _ in range(len(A)):
            dp.append({})
        max_ = 0
        for i in range(len(A)):
            for j in range(0, i):
                if i == 0:
                    dp[i][0] == 1
                else:
                    diff = A[i] - A[j]
                    if diff in dp[j]:
                        dp[i][diff] = dp[j][diff] + 1
                    else:
                        dp[i][diff] = 2
                    if dp[i][diff] > max_:
                        max_ = dp[i][diff]
        return max_
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        # dp = defaultdict(int)
        dp = {}
        for i in range(1, len(A)):
            for j in range(i):
                diff = A[i] - A[j]
                if (j,diff) in dp:
                    dp[(i,diff)] = dp[(j,diff)] + 1
                else:
                    dp[(i,diff)] = 1
        return max(dp.values()) + 1
#dp
#d[(i, diff)] = len: end at i with diff has maximum subsequence length len
#i: right num idx, j: left num idx
#d[(i, diff)] = d[(j, diff)] + 1 if (j, diff) in d else 2, j = 0...i-1
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        d = dict()
        for i, a in enumerate(A):
            for j in range(i):
                diff = a - A[j]
                if (j, diff) in d:
                    d[(i, diff)] = d[(j, diff)] + 1
                else:
                    d[(i, diff)] = 2
        return max(d.values())
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        
        # Question: list is arithmetic with constant diff bet elements
        # dict of dict -> for each index key have dict with diff as key and count as value
        cur = collections.defaultdict(dict)
        
        maxSeq = 0
        for i, v in enumerate(A):
            for j in range(i):
                val = v - A[j]
                cur[i][val] = 1 + cur[j].get(val, 1)  # def is 1 and not 0 since first time its 2 no's diff
                # print(cur)
                if maxSeq < cur[i][val] : maxSeq = cur[i][val]
                
        
        return maxSeq
        # return max(cur[i][j] for i in cur for j in cur[i])
        
# [3,6,9,12]   
# defaultdict(<class 'dict'>, {})
# defaultdict(<class 'dict'>, {0: {}, 1: {3: 2}})
# defaultdict(<class 'dict'>, {0: {}, 1: {3: 2}, 2: {6: 2}})
# defaultdict(<class 'dict'>, {0: {}, 1: {3: 2}, 2: {6: 2, 3: 3}})
# defaultdict(<class 'dict'>, {0: {}, 1: {3: 2}, 2: {6: 2, 3: 3}, 3: {9: 2}})
# defaultdict(<class 'dict'>, {0: {}, 1: {3: 2}, 2: {6: 2, 3: 3}, 3: {9: 2, 6: 2}})
# defaultdict(<class 'dict'>, {0: {}, 1: {3: 2}, 2: {6: 2, 3: 3}, 3: {9: 2, 6: 2, 3: 4}})



class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        dp = {}
        N = len(A)
        for i, n in enumerate(A):
            for j in range(i+1, N):
                b = A[j] - n
                if (i, b) in dp:
                    dp[j, b] = dp[i, b] + 1
                else:
                    dp[j, b] = 2
                    
        return max(dp.values())
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        
        # Question: list is arithmetic with constant diff bet elements
        # dict of dict -> for each index key have dict with diff as key and count as value
        cur = collections.defaultdict(dict)
        
        maxSeq = 0
        
        # this is DP problem but this is one way to solve without DP, by storing all
        for i, v in enumerate(A):
            for j in range(i):
                val = v - A[j]    # end -start
                # diff cnt + previous diff cnt from start(j)
                cur[i][val] = 1 + cur[j].get(val, 1)  # def is 1 and not 0 since first time its 2 no's diff
                # print(cur)
                if maxSeq < cur[i][val] : maxSeq = cur[i][val]
                
        
        return maxSeq
        # return max(cur[i][j] for i in cur for j in cur[i])
        
# [3,6,9,12]   
# defaultdict(<class 'dict'>, {})
# defaultdict(<class 'dict'>, {0: {}, 1: {3: 2}})
# defaultdict(<class 'dict'>, {0: {}, 1: {3: 2}, 2: {6: 2}})
# defaultdict(<class 'dict'>, {0: {}, 1: {3: 2}, 2: {6: 2, 3: 3}})
# defaultdict(<class 'dict'>, {0: {}, 1: {3: 2}, 2: {6: 2, 3: 3}, 3: {9: 2}})
# defaultdict(<class 'dict'>, {0: {}, 1: {3: 2}, 2: {6: 2, 3: 3}, 3: {9: 2, 6: 2}})
# defaultdict(<class 'dict'>, {0: {}, 1: {3: 2}, 2: {6: 2, 3: 3}, 3: {9: 2, 6: 2, 3: 4}})



class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        dp = [{} for _ in A]
        max_l = 1
        for i in range(1, len(A)):
            for j in range(i):
                diff = A[i] - A[j]
                dp[i][diff] = dp[j].get(diff, 1) + 1
                if dp[i][diff] > max_l:
                    max_l = dp[i][diff]
        return max_l

from itertools import repeat

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        length = len(A)
        onleft = 0
        onleftl = []
        onright = {}
        toextend = [{} for _ in range(501)]
        res = 2
        for v in A:
            onright[v] = onright.get(v, 0) + 1
        for i in range(0, length):
            val = A[i]
            tex = toextend[val]
            onright[val] -= 1
            for lval in onleftl:
                diff = val - lval
                nextval = val + diff
                if nextval not in onright or onright[nextval] == 0:
                    continue
                if diff in tex:
                    c = tex[diff] + 1
                else:
                    c = 3
                if c > res:
                    res = c
                toextend[nextval][diff] = c
            b = (1<<val)
            if not (onleft & b):
                onleftl.append(val)
                onleft = (onleft | b)
        return res
import bisect
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        
        dp = {}
        
        for i, num in enumerate(A):
            for j in range(i+1, len(A)):
                diff = A[j]-num
                if (i, diff) not in dp:
                    dp[(j, diff)] = 2
                else:
                    dp[(j, diff)] = dp[(i, diff)] + 1
        
        return max(dp.values())
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:

    
        f = {}
        maxlen = 0
        for i in range(len(A)):
            for j in range(i):
                diff = A[i] - A[j]
                #fff[(A[i], diff)] = max(fff[(A[i], diff)], fff.get((A[j], diff), 1) + 1)
                #f[(i, diff)] = max(f[(i, diff)], f.get((j, diff), 1) + 1)
                #f[(i, diff)] = f.get((j, diff), 1) + 1
                
                if (j, diff) not in f:
                    f[i, diff] = 2
                else:
                    f[i, diff] = f[j, diff] + 1          
                                    
                #maxlen = max(maxlen, f[(i, diff)])

        return max(f.values())

from collections import OrderedDict

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        # When adding a new number A[j], we look at all previous numbers A[i]:
        # (1) If A[j] can extend any arithmetic subsequence currently ends at A[i]: LAS += 1
        # (2) Otherwise, LAS = 2
        subseq_lengths = {}
        for j in range(1, len(A)):
            for i in range(j):
                diff = A[j] - A[i]
                if (diff, i) in subseq_lengths:
                    subseq_lengths[diff, j] = subseq_lengths[diff, i] + 1
                else:
                    subseq_lengths[diff, j] = 2
        return max(subseq_lengths.values())
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        
        n = len(A)
        d = {}
        for i in range(n):
            for j in range(i+1,n):
                diff = A[j] - A[i]
                
                if (i,diff) in d:
                    d[(j,diff)] = d[(i,diff)] + 1
                    
                else:
                    d[(j,diff)] = 2
                    
        #print(d)
        return max(d.values())
    
        # n=len(A)
        # dp={}
        # for i in range(n):
        #     for j in range(i+1,n):
        #         dif = A[j]-A[i]
        #         if (i,dif) in dp :
        #             dp[(j,dif)]=dp[(i,dif)]+1
        #         else:
        #             dp[(j,dif)]=2
        # return max(dp.values())

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        n=len(A)
        if n<=1: return A
        t=collections.defaultdict(int)
        dp=[None]*n
        for i in range(n):
            dp[i]=collections.defaultdict(int)
        for i in range(n):
            for j in range(i):
                diff=A[i]-A[j]
                dp[i][diff]=dp[j][diff]+1
                
        ret=0
        for i in range(n):
            ret=max(ret,max(dp[i].values())+1)
        return ret
        

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        n=len(A)
        dp={}
        for i in range(n):
            for j in range(i+1,n):
                dif = A[j]-A[i]
                if (i,dif) in dp :
                    dp[(j,dif)]=dp[(i,dif)]+1
                else:
                    dp[(j,dif)]=2
        return max(dp.values())
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        if not A: return 0
        if len(A)<3: return 2
        dp = {}
        for i, a1 in enumerate(A[1:], 1):
            for j, a2 in enumerate(A[:i]):
                diff = a1 - a2
                if (j, diff) in dp:
                    dp[i, diff] = dp[j, diff] + 1
                else:
                    dp[i, diff] = 2
                    
        return max(dp.values())
                
            
        
        
        
                        

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        length = len(A)
        onleft = 0
        onleftl = []
        onright = [0 for _ in range(501)]
        toextend = [{} for _ in range(501)]
        res = 2
        for v in A:
            onright[v] += 1
        for i in range(0, length):
            val = A[i]
            tex = toextend[val]
            onright[val] -= 1
            for lval in onleftl:
                diff = val - lval
                nextval = val + diff
                if nextval > 500 or nextval < 0 or onright[nextval] == 0:
                    continue
                c = tex.get(diff, 2) + 1
                if c > res:
                    res = c
                else:
                    ending = (res - c) * diff + nextval
                    if ending > 500 or ending < 0 or onright[ending] == 0:
                        continue
                toextend[nextval][diff] = c
            b = (1 << val)
            if not (onleft & b):
                onleft = (onleft | b)
                onleftl.append(val)
        return res
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        
#         d = {}
#         for i in range(len(A)):
#             for j in range(i+1,len(A)):
#                 diff = A[j] - A[i]
                
#                 if (i,diff) in d:
#                     d[(j,diff)] = d[(i,diff)] + 1
                    
#                 else:
#                     d[(j,diff)] = 2
                    
#         print(d)
#         return max(d.values())
    
        n=len(A)
        dp={}
        for i in range(n):
            for j in range(i+1,n):
                dif = A[j]-A[i]
                if (i,dif) in dp :
                    dp[(j,dif)]=dp[(i,dif)]+1
                else:
                    dp[(j,dif)]=2
        return max(dp.values())
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        
        if len(A) <= 2:
            return len(A)
        
        n = len(A)
        memo = {}
        
        for i in range(n):
            for j in range(i + 1,n):
                diff = A[j] - A[i]
                memo[(j, diff)] = memo[(i, diff)] + 1 if (i, diff) in memo else 2
        
        return max(memo.values())
class Solution:
    def longestArithSeqLength(self, A):
        dp = dict()
        for endi, endv in enumerate(A[1:], start = 1):
            for starti, startv in enumerate(A[:endi]):
                diff = endv - startv
                if (starti, diff) in dp:
                    dp[(endi, diff)] = dp[(starti, diff)] + 1
                else:
                    dp[(endi, diff)] = 2
        return max(dp.values())
        


class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        dp = {}
        for i, a2 in enumerate(A[1:], start=1):
            for j, a1 in enumerate(A[:i]):
                d = a2 - a1
                if (j, d) in dp:
                    dp[i, d] = dp[j, d] + 1
                else:
                    dp[i, d] = 2
        return max(dp.values())
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        n = len(A)
        dp = {}
        for i in range(1, n):
            for j in range(i):
                dif = A[i] - A[j]
                if (j,dif) in dp:
                    dp[(i, dif)] = dp[(j, dif)] + 1
                else:
                    dp[(i, dif)] = 2
                #print(dp)
        return max(dp.values())

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        dp = {}
        for i, a2 in enumerate(A[1:], start=1):
            for j, a1 in enumerate(A[:i]):
                d = a2 - a1
                if (j, d) in dp:
                    dp[i, d] = dp[j, d] + 1
                else:
                    dp[i, d] = 2
        return max(dp.values())
                

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        s = len(A)
        dp = {}
        for i in range(s):
            for j in range(i+1, s):
                diff = A[j] - A[i]
                if (i, diff) in dp:
                    dp[(j, diff)] = dp[(i, diff)] + 1
                else:
                    dp[(j, diff)] = 2
        return max(dp.values())
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        
        dp = collections.defaultdict(int)
        
        for i in range(0, len(A)):
            for j in range(i + 1, len(A)):
                diff = A[j] - A[i]
                if (i, diff) in dp:
                    dp[(j, diff)] = dp[(i,diff)] + 1
                else:
                    dp[(j,diff)] = 2
                
            
        return max(dp.values())
                    
            

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        length = len(A)
        onleft = set()
        onleftl = [0 for _ in range(501)]
        onleftlen = 0
        onright = [0 for _ in range(501)]
        toextend = [{} for _ in range(501)]
        res = 2
        for v in A:
            onright[v] += 1
        for i in range(0, length):
            val = A[i]
            tex = toextend[val]
            onright[val] -= 1
            for lvali in range(onleftlen):
                lval = onleftl[lvali]
                diff = val - lval
                nextval = val + diff
                if nextval > 500 or nextval < 0 or onright[nextval] == 0:
                    continue
                if diff in tex:
                    c = tex[diff] + 1
                else:
                    c = 3
                if c > res:
                    res = c
                toextend[nextval][diff] = c
            if val not in onleft:
                onleftl[onleftlen] = val
                onleftlen += 1
                onleft.add(val)
        return res
from functools import lru_cache
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        
        if len(A) <= 2:
            return len(A)
        
        n = len(A)
        memo = {}
        
        for i in range(n):
            for j in range(i + 1,n):
                diff = A[j] - A[i]
                memo[(j, diff)] = memo[(i, diff)] + 1 if (i, diff) in memo else 2
        
        return max(memo.values())
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        maximum = 0
        n=len(A)
        dp={}
        for i in range(n):
            for j in range(i+1,n):
                dif = A[j]-A[i]
                if (i,dif) in dp :
                    dp[(j,dif)]=dp[(i,dif)]+1
                else:
                    dp[(j,dif)]=2
        return max(dp.values())
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        dp = {}
        for i in range (len(A)):
            for j in range (i+1,len(A)):
                d = A[j] - A[i]
                if (i,d) in dp:
                    dp[j,d] = dp[i,d]+1
                else:
                    dp[j,d] = 2
        return max(dp.values())
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:  
        memo = dict()
        n = len(A)
        for i in range(n-1,-1,-1):
            for j in range(n-1,i,-1):
                d = A[j] - A[i]
                if (j,d) in memo:
                    memo[(i,d)] = memo[(j,d)] + 1
                else:
                    memo[(i,d)] = 2
        return max(memo.values())

from heapq import heappop, heapify


class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        
        # KEEP TRACK OF THE MAXIMUM COUNT
        count = 0
        
        # dp[i][d] = Length of arithmetic subsequence ending at A[i] (inclusive), with diff = d
        dp = defaultdict(lambda: defaultdict(int))
        
        for i in range(1, len(A)):
            seen = set()
            for j in range(i - 1, -1, -1):
                diff = A[i] - A[j]
                if diff not in seen:
                    dp[i][diff] += dp[j][diff] + 1
                    count = max(count, dp[i][diff])
                    seen.add(diff)
                    
        # for k, v in dp.items():
        #     print(k, v)
                                  
        return count + 1

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        
        dp = {}
        for i in range(len(A)):
            for j in range(i + 1, len(A)):
                gap = A[j] - A[i]
                if (i, gap) in dp:
                    dp[(j, gap)] = dp[(i, gap)] + 1
                else:
                    dp[(j, gap)] = 2
                    
        return max(dp.values())
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        dic = {}
        for i, val2 in enumerate(A[1:], start = 1):
            for j, val in enumerate(A[:i]):
                diff = val2 - val
                if (j,diff) in dic:
                    dic[i,diff] = dic[j,diff] + 1
                else:
                    dic[i,diff] = 2
                
        return max(dic.values())

from itertools import repeat

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        length = len(A)
        onleft = 0
        onleftl = []
        onright = [*repeat(0, 501)]
        toextend = [{} for _ in range(501)]
        res = 2
        for v in A:
            onright[v] += 1
        for i in range(0, length):
            val = A[i]
            tex = toextend[val]
            onright[val] -= 1
            for lval in onleftl:
                diff = val - lval
                nextval = val + diff
                if nextval > 500 or nextval < 0 or onright[nextval] == 0:
                    continue
                if diff in tex:
                    c = tex[diff] + 1
                else:
                    c = 3
                if c > res:
                    res = c
                toextend[nextval][diff] = c
            b = (1<<val)
            if not (onleft & b):
                onleftl.append(val)
                onleft = (onleft | b)
        return res
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        length = len(A)
        onleft = set()
        onright = [0 for _ in range(501)]
        toextend = [{} for _ in range(501)]
        res = 2
        for v in A:
            onright[v] += 1
        for i in range(0, length):
            val = A[i]
            tex = toextend[val]
            onright[val] -= 1
            for lval in onleft:
                diff = val - lval
                nextval = val + diff
                if nextval > 500 or nextval < 0 or onright[nextval] == 0:
                    continue
                if diff in tex:
                    c = tex[diff] + 1
                else:
                    c = 3
                if c > res:
                    res = c
                toextend[nextval][diff] = c
            onleft.add(val)
        return res
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        di={}
        for i in range(1,len(A)):
            for j in range(i):
                d=A[i]-A[j]
                if (j,d) in di:
                    di[i,d]=di[j,d]+1
                else:
                    di[i,d]=2
        return max(di.values())
                    

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        
        n = len(A)
        
        # 2 4 6
        # 2 1 4 6
        # 2 3 4 6 8
        
        increment = 0
        num_steps = 0
        dp = {}
        for i in range(len(A)):
            for j in range(i+1, len(A)):
                diff = A[j] - A[i]
                
                if (i,diff) not in dp:
                    dp[j,diff] = 2
                else:
                    dp[j,diff] = dp[i,diff] + 1
        
        return max(dp.values())
            
            
                
                
            
            

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        dp = {}
        for i in range(len(A)):
            for j in range(i + 1, len(A)):
                delta = A[j] - A[i]
                dp[(j, delta)] = dp[(i, delta)] + 1 if (i, delta) in dp else 2
        return max(dp.values())
from collections import OrderedDict

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        # When adding a new number A[j], we look at all previous numbers A[i]:
        # (1) If A[j] can extend any arithmetic subsequence currently ends at A[i]: LAS += 1
        # (2) Otherwise, LAS = 2
        subseq_lengths = {}
        for j in range(1, len(A)):
            for i in range(j):
                diff = A[j] - A[i]
                subseq_lengths[diff, j] = subseq_lengths.get((diff, i), 1) + 1
        return max(subseq_lengths.values())
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        dp = {}
        for i in range(1, len(A)):
            for j in range(i):
                diff = A[i] - A[j]
                if (j, diff) in dp:
                    dp[(i, diff)] = dp[(j, diff)] + 1
                else:
                    dp[(i, diff)] = 2
        return max(dp.values())
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        n = len(A)
        if n < 3:
            return n
        d = {}
        for i in range(n - 1):
            for j in range(i + 1, n):
                diff = A[j] - A[i]
                if (i, diff) in d:
                    d[(j, diff)] = d[(i, diff)] + 1
                else:
                    d[(j, diff)] = 2
        return max(d.values())
#Memorize: arithmetic sequence has >=2 length.
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        memo = {}
        for i in range(len(A)):
            for j in range(i):
                interval = A[i]-A[j]
                if (interval, j) in memo:
                    memo[(interval, i)] = memo[(interval, j)]+1
                else:
                    memo[(interval, i)] = 2
        return max(memo.values())

'''
9 4 7 2 10

u62bdu4e24u4e2au6570 u5f97u5230u7b49u5dee u8ba1u7b97u6240u6709u7684u7ec4u5408
dp[(j, dif)]u6307u7684u662fu4ee5difu4e3au7b49u5deeuff0c u622au81f3uff08u5305u542buff09A[j], u7684u6700u957fu7b49u5deeu5e8fu5217
'''
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        dp = {}
        for i in range(len(A)):
            for j in range(i + 1, len(A)):
                dif = A[j] - A[i]
                dp[(j, dif)] = dp.get((i, dif), 1) + 1
        return max(dp.values())

            

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        n = len(A)
        mark = [{}]
        res = 0
        
        for i in range(1, n):
            mark.append({})
            for j in range(i):
                delta = A[i] - A[j]
                if delta in mark[j]:
                    if delta in mark[i]:
                        mark[i][delta] = max(mark[j][delta] + 1, mark[i][delta])
                    else:
                        mark[i][delta] = mark[j][delta] + 1
                else:
                    if delta not in mark[i]:
                        mark[i][delta] = 1
                    
                if mark[i][delta] > res:
                    res = mark[i][delta]
                    
        return res+1

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        dp = {}
        for i in range(len(A)):
            for j in range(i + 1, len(A)):
                d = A[j] - A[i]
                dp[j, d] = dp.get((i, d), 1) + 1
        return max(dp.values())
from array import array
from itertools import repeat

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        length = len(A)
        onleft = 0
        onleftl = []
        onright = array('H', repeat(0, 501))
        toextend = [{} for _ in range(501)]
        res = 2
        for v in A:
            onright[v] += 1
        for i in range(0, length):
            val = A[i]
            tex = toextend[val]
            onright[val] -= 1
            for lval in onleftl:
                diff = val - lval
                nextval = val + diff
                if nextval > 500 or nextval < 0 or onright[nextval] == 0:
                    continue
                if diff in tex:
                    c = tex[diff] + 1
                else:
                    c = 3
                if c > res:
                    res = c
                toextend[nextval][diff] = c
            b = (1<<val)
            if not (onleft & b):
                onleftl.append(val)
                onleft = (onleft | b)
        return res
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        dp = dict()
        n = len(A)
        for i in range(n):
            for j in range(i+1,n):
                delta = A[j]-A[i]
                dp[(j, delta)] = dp.get((i, delta), 1) + 1
        return max(dp.values())
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        dp = [[1] * 1001 for _ in range(501)]
        res = 2
        for i, a in enumerate(A):
            flag = True
            for j in range(i):
                d = a - A[j]
                if d == 0 and not flag: continue
                dp[a][d] = dp[A[j]][d] + 1
                res = max(res, dp[a][d])
                if d == 0: flag = False
                # if res == dp[a][d]: print(a, d, dp[a][d])
        return res
        

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        dp = {} 
        for i in range(len(A)): 
            for j in range(i+1, len(A)): 
                delta = A[j]-A[i] 
                if (delta, i) in dp:  
                    dp[(delta, j)] = dp[(delta, i)] + 1 
                else: 
                    dp[(delta, j)] = 2
                 
        return max(dp.values()) 
                
         
         
         
            
        

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        DP={}
        for i in range(1,len(A)):
            for j in range(len(A[:i])):
                d=A[i]-A[j]
                if (j,d) in DP: DP[i,d]=DP[j,d]+1
                else: DP[i,d]=2
        return max(DP.values())
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        dp = {}; n = len(A)
        for i in range(n):
            for j in range(i+1, n):
                b = A[j] - A[i]
                if (i,b) not in dp: dp[j,b] = 2
                else              : dp[j,b] = dp[i,b] + 1
        return max(dp.values())

from collections import defaultdict
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        d = defaultdict(int)
        maxVal = 2
        for i in range(1,len(A)):
            for j in range(i):
                diff = A[i] - A[j]
                # print(d)
                if (j,diff) not in d:
                    d[(i,diff)] = 2
                else:
                    # if (j,diff) in d:
                    d[(i,diff)] = d[(j,diff)] + 1 
                    
                    
                        
        # print(d)
        return max(d.values())
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        dp = {}
        #dictionary of pairs (i pointer, dist)
        
        # i pointer iterates from 1index to the end
        for i in range(1, len(A)):
            # j pointer iterates from 0 to just left of i pointer then resets
            for j in range(0, len(A[:i])):
                
                #finds the difference of the two values
                d = A[i] - A[j]
                
                #checks to see if the same diff exists at j
                if (j, d) in dp:
                    #if j,d is in dp then add 1 because the value at i has the same difference and set that as i,d
                    dp[i, d] = dp[j, d] + 1
                #if not then its set to two because that accounts for the i,j as two integers
                else:
                    dp[i, d] = 2
                    
        #return what ever is the highest value of all the keys in the dictionary is
        return max(dp.values())

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        length = len(A)
        onleft = set()
        onright = [0 for _ in range(501)]
        toextend = [{} for _ in range(501)]
        res = 2
        for v in A:
            onright[v] += 1
        for i in range(0, length):
            val = A[i]
            tex = toextend[val]
            onright[val] -= 1
            for lval in onleft:
                diff = val - lval
                nextval = val + diff
                if nextval > 500 or nextval < 0 or onright[nextval] == 0:
                    continue
                c = tex.get(diff, 2) + 1
                if c > res:
                    res = c
                toextend[nextval][diff] = c
            onleft.add(val)
        return res
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        dp = {}
        for i in range(len(A)):
            for j in range(i + 1, len(A)):
                diff = A[j] - A[i]
                dp[j, diff] = dp.get((i, diff), 1) + 1
        return max(dp.values())
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        mapp=collections.defaultdict(int)
        if not A:
            return -1
        
        for i in range(1,len(A)):
            for j in range(0,len(A[:i])):
                d=A[i]-A[j]
                if (d,j) in mapp:
                    mapp[(d,i)]=mapp[d,j]+1
                else:
                    mapp[d,i]=2
        return max(mapp.values())
                

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
          n=len(A);dp={};s=set(A)
          for i in range(1,n):
               for j in range(i):
                    d=A[i]-A[j]
                    if j!=0 and (j,d) in dp:
                        dp[(i,d)]=1+dp[(j,d)]
                    else:
                        dp[(i,d)]=2
          l=dp.values()
          return max(l)
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        # dp[(idx, diff)]: length of arithmetic sequence at index with difference diff.
        dp = {}
        for i in range(len(A)):
            for j in range(i + 1, len(A)):
                diff = A[j] - A[i]
                dp[(j, diff)] = dp.get((i, diff), 1) + 1
        return max(dp.values())
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        n = len(A)
        diff = {}
        for i in range(n):
            for j in range(i+1, n):
                d = A[j] - A[i]
                if (i, d) in diff:
                    diff[(j, d)] = diff[(i, d)] + 1
                else:
                    diff[(j, d)] = 2
        return max(diff.values())

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        if len(A) <= 1:
            return len(A)
        
        memo = [(1 + 2 * 500) * [1] for _ in range(1 + len(A))]
        
        res = 2
        
        for i in range(len(A)-2, -1, -1):
            Ai = A[i] - 500
            mi = memo[i]
            
            for j in range(i+1, len(A)):
                diff = A[j] - Ai
                mi[diff] = max(mi[diff], memo[j][diff] + 1)
        
            res = max(res, max(mi))
        
        return res
        
#         h = dict()
        
#         res = 2
        
#         for i, ai in enumerate(A):
#             for j in range(i):
#                 diff = A[j] - ai
#                 aux = (j, diff)
#                 if aux in h:
#                     h[(i, diff)] = h[aux] + 1
#                     # res = max(res, h[(i, diff)])
#                 else:
#                     h[(i, diff)] = 2
    
#         return max(h.values())

from itertools import repeat

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        length = len(A)
        onleft = 0
        onleftl = []
        onright = {}
        toextend = [{} for _ in range(501)]
        res = 2
        for v in A:
            if v not in onright:
                onright[v] = 1
            else:
                onright[v] += 1
        for i in range(0, length):
            val = A[i]
            tex = toextend[val]
            onright[val] -= 1
            for lval in onleftl:
                diff = val - lval
                nextval = val + diff
                if nextval not in onright or onright[nextval] == 0:
                    continue
                if diff in tex:
                    c = tex[diff] + 1
                else:
                    c = 3
                if c > res:
                    res = c
                toextend[nextval][diff] = c
            b = (1<<val)
            if not (onleft & b):
                onleftl.append(val)
                onleft = (onleft | b)
        return res
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        dp = {}
        for i in range(len(A)):
            for j in range(i+1, len(A)):
                b = A[j] - A[i]
                if (i,b) not in dp: 
                    dp[j,b] = 2
                else: 
                    dp[j,b] = dp[i,b] + 1
        return max(dp.values())
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        diffdict = {}
        max = 0
        
        for i in range(len(A)):
            
            for j in range(i):
                
                diff = A[i]-A[j]
                if diff not in diffdict:
                    diffdict[diff] = {i: 2}
                else:
                    if j in diffdict[diff]: 
                        diffdict[diff][i] = diffdict[diff][j] + 1
                    else:
                        diffdict[diff][i] = 2
                if diffdict[diff][i] > max:
                        max = diffdict[diff][i]
                    
        return  max
        

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        res = 0
        dic = {}
        for i in range(len(A)):
            for j in range(i+1,len(A)):
                d = A[j]-A[i]
                dic[j,d]=dic.get((i,d),1)+1
                # if (i,d) in dic:
                #     dic[(j,d)]=dic[(i,d)]+1
                # else:
                #     dic[(j,d)]=2
                
        return max(dic.values())

class Solution:
    def longestArithSeqLength(self, A):
        dp = collections.defaultdict(int)
        for i in range(len(A)):
            for j in range(i):
                diff = A[i] - A[j]
                j_key = (j, diff)
                i_key = (i, diff)
                if j_key in dp:
                    dp[i_key] = dp[j_key] + 1
                else:
                    dp[i_key] = 2
        return max(dp.values())

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        dp = {}
        for i, a2 in enumerate(A[1:], start=1):
            for j, a1 in enumerate(A[:i]):
                d = a2 - a1
                if (j, d) in dp:
                    dp[i, d] = dp[j, d] + 1
                else:
                    dp[i, d] = 2
        return max(dp.values())
    
    
        f = collections.defaultdict(int)
        maxlen = 0
        for i in range(len(A)):
            for j in range(i):
                diff = A[i] - A[j]
                #fff[(A[i], diff)] = max(fff[(A[i], diff)], fff.get((A[j], diff), 1) + 1)
                f[(i, diff)] = max(f[(i, diff)], f.get((j, diff), 1) + 1)
                '''
                if (j, diff) not in f:
                    f[(i, diff)] = 2
                else:
                    f[(i, diff)] = max(f[(i, diff)],  f[(j, diff)] + 1)                
                '''                    
                maxlen = max(maxlen, f[(i, diff)])

        return maxlen

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        mem = [[0]*1001 for i in range(len(A))]
        ret = 0
        for i in range(len(A)):
            for j in range(i):
                diff = A[i] - A[j]
                mem[i][diff] = mem[j][diff] + 1
                ret = max(ret, mem[i][diff])
        return ret+1
'''
subseq: non contiguous

B[i+1] - B[i] is always the same 
for each we need the last elt in seq 

goal: return the length of longest 

state:
i, last in seq, diff, length of the seq 

helper(i, last, diff, l)

choices:
 - use (condition A[i] - last == diff)
 - skip 
 - start looking for a new arith seq ? 

use: 
if A[i] - last == diff:
    helper(i + 1, A[i], diff, l + 1)
    
helper(i+1, last, diff, l)

if i > 0:
    helper(i + 1, A[i], A[i] - A[i - 1], 2)
   
dp[(i, diff)]: length of longest arith seq ending at i with difference diff 


'''
from functools import lru_cache

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        n = len(A)
        dp = {}
        
        for i in range(n):
            for j in range(i + 1, n):
                d = A[j] - A[i]
                dp[(j, d)] = dp.get((i, d), 1) + 1
        return max(dp.values())
           
                
                
                
                
                

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        dp = {}
        
        for i in range(len(A)):
            for j in range(i):
                d = A[j] - A[i]
                dp[i, d] = dp.get((j, d), 0) + 1
                    
        return max(dp.values()) + 1
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        dp = {}
        
        for i in range(len(A)):
            samecnt = 1
            for j in range(i):
                diff = A[i] - A[j]
                if diff == 0:
                    samecnt += 1
                else:
                    dp[A[i], diff] = dp.get((A[j], diff), 1) + 1
            dp[A[i], 0] = samecnt
        
        # print(dp)
        key = max(dp, key=dp.get)
        return dp[key]
from itertools import repeat

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        length = len(A)
        onleft = set()
        onleftl = []
        onright = list(repeat(0, 501))
        toextend = [{} for _ in range(501)]
        res = 2
        for v in A:
            onright[v] += 1
        for i in range(0, length):
            val = A[i]
            tex = toextend[val]
            onright[val] -= 1
            for lval in onleftl:
                diff = val - lval
                nextval = val + diff
                if nextval > 500 or nextval < 0 or onright[nextval] == 0:
                    continue
                c = tex.get(diff, 2) + 1
                if c > res:
                    res = c
                toextend[nextval][diff] = c
            if val not in onleft:
                onleftl.append(val)
                onleft.add(val)
        return res
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        dp = {}
        for i in range(1, len(A)):
            for j in range(i):
                num1 = A[j]
                num2 = A[i]
                d = num2 - num1
                if (j, d) in dp:
                    dp[(i, d)] = dp[(j, d)] + 1
                else:
                    dp[(i, d)] = 2
        return max(dp.values())

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        dp = {}
        L = len(A)
        for i in range(L):
            for j in range(i):
                diff = A[i] - A[j]
                dp[(i, diff)] = dp.get((j, diff), 1) + 1
        
        return max(dp.values())
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        # When adding a new number A[j], we look at all previous numbers A[i]:
        # (1) If A[j] can extend any arithmetic subsequence currently ends at A[i]: LAS += 1
        # (2) Otherwise, LAS = 2
        subseq_lengths = {}
        for j in range(1, len(A)):
            for i in range(j):
                diff = A[j] - A[i]
                if (diff, i) in subseq_lengths:
                    subseq_lengths[diff, j] = subseq_lengths[diff, i] + 1
                else:
                    subseq_lengths[diff, j] = 2
        return max(subseq_lengths.values())
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        dp = {}
        
        for i, a in enumerate(A[1:], 1):
            for j, b in enumerate(A[:i]):
                d = a - b
                dp[i, d] = dp.get((j, d), 0) + 1
                    
        return max(dp.values()) + 1
class Solution:
    def longestArithSeqLength(self, nums: List[int]) -> int:
        dp=dict()
        for i in range(len(nums)):
            for j in range(i+1,len(nums)):
                diff=nums[j]-nums[i]
                dp[(j,diff)]=dp.get((i,diff),1)+1

        # print(dp)
        return max(dp.values())
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        memo = {}
        
        for i in range(len(A)):
            for j in range(i+1, len(A)):
                diff = A[j] - A[i]
                memo[j, diff] = memo.get((i, diff), 1) + 1
                
        return max(memo.values())
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        dp, n = {}, len(A)
        for i in range(n):
            for j in range(i+1, n):
                diff = A[j] - A[i]
                dp[(j, diff)] = dp.get((i, diff), 1) + 1
                
        return max(dp.values())

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        dp = {}
        
        for i in range(len(A)):
            samecnt = 1
            for j in range(i):
                diff = A[i] - A[j]
                if diff == 0:
                    samecnt += 1
                else:
                    dp[A[i], diff] = dp.get((A[j], diff), 1) + 1
            dp[A[i], 0] = samecnt
        
        # print(dp)
        key = max(dp, key=dp.get)
        # print(key)
        return dp[key]
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        # O(N ** 2) DP
        dp = {}
        n = len(A)
        for i in range(n):
            for j in range(i + 1, n):
                diff = A[j] - A[i]
                dp[(j, diff)] = dp.get((i, diff), 1) + 1
        return max(dp.values())
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        # dp[index][diff] := the longest arithmetic subsequence in A[:index+1] with difference = diff
        dp = collections.defaultdict(int)
        for i in range(len(A)):
            for j in range(i+1, len(A)):
                diff = A[j] - A[i]
                dp[(j, diff)]  = dp.get((i,diff), 1) + 1
        return max(dp.values())
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        '''
        Maintain a dictionary of differences at each position.
        The keys are going to be (position, diff)
        
        Compare each value with all the values after it
        calculate the diff and store in the dictionary using the equation in the solution.
        This equation is the crux of the solution.
        
        Do d dry run with example for better understanding
        https://leetcode.com/problems/longest-arithmetic-subsequence/discuss/274611/JavaC%2B%2BPython-DP
        '''
        if not A:
            return 0
        
        dp = {}
        for i in range(len(A)):
            for j in range(i+1, len(A)):
                diff = A[j] - A[i]
                dp[j, diff] = dp.get((i, diff), 1) + 1
        
        return max(dp.values())
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        if not A:
            return 0
        
        dp = [None] * len(A)
        dp[0] = {0:1}
        
        max_val = 1
        
        for i in range(1, len(A)):
            dp[i] = {}
                        
            for j in range(i):
                diff = A[i] - A[j]
                diff_val = 2
                
                if diff in dp[j]:
                    diff_val = dp[j][diff] + 1
                
                dp[i][diff] = diff_val

                max_val = max(max_val, diff_val)
                    
        return max_val
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        dp = {}
        
        for i in range(len(A)):
            for j in range(i):
                step = A[i]-A[j]
                dp[(i,step)] = dp.get((j,step),1)+1
        
        return max(dp.values())
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        diffs = set(y - x for i, x in enumerate(A) for y in A[i+1:])
        ans = 0
        for diff in diffs:
            data = {}
            for num in A:
                if num - diff not in data:
                    if num not in data:
                        data[num] = [num]
                    continue
                if len(data[num - diff]) < len(data.get(num, [])):
                    continue
                seq = data.pop(num - diff)
                seq.append(num)
                ans = max(ans, len(seq))
                data[num] = seq
        return ans
import numpy as np

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        length = len(A)
        onleft = set()
        onright = np.zeros(501, dtype=np.int16)
        toextend = [{} for _ in range(501)]
        res = 2
        for v in A:
            onright[v] += 1
        for i in range(0, length):
            val = A[i]
            tex = toextend[val]
            onright[val] -= 1
            for lval in onleft:
                diff = val - lval
                nextval = val + diff
                if nextval > 500 or nextval < 0 or onright[nextval] == 0:
                    continue
                if diff in tex:
                    c = tex[diff] + 1
                else:
                    c = 3
                if c > res:
                    res = c
                toextend[nextval][diff] = c
            onleft.add(val)
        return res
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        dp = [{} for _ in range(len(A))]
        # maxSequence = 0
        for i in range(len(A)):
            for j in range(i):
                diff = A[i] - A[j]
                dp[i][diff] = dp[j].get(diff,1) + 1
                # if val > maxSequence:
                #     maxSequence = val
        return max(v1 for dic in dp for v1 in dic.values())
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        if len(A) <= 2:
            return len(A)
        n, ans = len(A), 0
        dp = [[0] * 1001
              for _ in range(n)]
        for j in range(n):
            for i in range(0, j):
                diff = A[j] - A[i] + 500
                dp[j][diff] = dp[i][diff] + 1
                ans = max(ans, dp[j][diff])
        
        return ans + 1
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        length = len(A)
        onleft = set()
        onleftl = []
        onright = [0 for _ in range(501)]
        toextend = [{} for _ in range(501)]
        res = 2
        for v in A:
            onright[v] += 1
        for i in range(0, length):
            val = A[i]
            tex = toextend[val]
            onright[val] -= 1
            for lval in onleft:
                diff = val - lval
                nextval = val + diff
                if nextval > 500 or nextval < 0 or onright[nextval] == 0:
                    continue
                c = tex.get(diff, 2) + 1
                if c > res:
                    res = c
                toextend[nextval][diff] = c
            if val not in onleft:
                onleftl.append(val)
                onleft.add(val)
        return res
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        if not A:
            return 0
        
        dp = [{0: 1}]
        
        max_val = 1
        
        for i in range(1, len(A)):
            dp.append({0: 1})
                        
            for j in range(i):
                diff = A[i] - A[j]
                diff_val = 2
                
                if diff in dp[j]:
                    diff_val = dp[j][diff] + 1
                
                dp[-1][diff] = diff_val

                max_val = max(max_val, diff_val)
                    
        return max_val
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        # table[index][diff] equals to the length of 
        # arithmetic sequence at index with difference diff.
        table = dict()
        max_v = 0
        for i in range(len(A)):
            for j in range(i+1,len(A)):
                
                _diff = A[j] - A[i]
                if (i,_diff) in table.keys():
                    table[j,_diff] = table[i,_diff] + 1
                else:
                    table[j,_diff] = 2 # the first diff
                    # will corrspond to two values [v1,v2]
                #max_v = max(max_v,table[j,_diff])
                    
        return max(table.values())
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        
        n = len(A)
        if n < 3: return n
        
        dp = [{} for _ in range(n)]
        max_ = 0
        for i in range(1, n):
            for j in range(i):
                
                diff = A[i] - A[j]
                if diff in dp[j]:
                    dp[i][diff] = dp[j][diff] + 1
                else:
                    dp[i][diff] = 2
                
                max_ = max(max_, dp[i][diff])
        
        return max_

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
          n=len(A);dp={};s=set(A)
          for i in range(1,n):
               for j in range(i):
                    d=A[i]-A[j]
                    if (j,d) in dp:
                        dp[(i,d)]=1+dp[(j,d)]
                    else:
                        dp[(i,d)]=2
          l=dp.values()
          return max(l)
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        diffs = [collections.defaultdict(int) for _ in range(len(A))]
        diffs[0][0] = 1
        max_len = 1
        for i in range(1, len(A)):
            for j in range(i):
                diff = A[i] - A[j]
                if diff in diffs[j]:
                    diffs[i][diff] = diffs[j][diff] + 1
                else:
                    diffs[i][diff] = 2
                
                max_len = max(max_len, diffs[i][diff])
        
        
        return max_len
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        
        table = dict()
        
        for i in range(len(A)):
            for j in range(i+1,len(A)):
                
                _diff = A[j] - A[i]
                if (i,_diff) in table.keys():
                    table[j,_diff] = table[i,_diff] + 1
                else:
                    table[j,_diff] = 2
                    
        return max(table.values())
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        dp = {}
        for i in range(len(A)):
            for j in range(i + 1, len(A)):
                dp[j, A[j] - A[i]] = dp.get((i, A[j] - A[i]), 1) + 1
        return max(dp.values())
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        dp = [{} for _ in range(len(A))]
        for i in range(len(A) - 1):
            for j in range(i + 1, len(A)):
                diff = A[j] - A[i]
                diffMap = dp[i]
                dp[j][diff] = diffMap.get(diff, 1) + 1

        maxLength = 0

        for i in range(len(A)):
            for diff in dp[i]:
                if dp[i][diff] > maxLength: maxLength = dp[i][diff]

        return maxLength        
# O(n2) time, O(n2) space   

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        length = len(A)
        onleft = set()
        #onleftl = []
        onright = [0 for _ in range(501)]
        toextend = [{} for _ in range(501)]
        res = 2
        for v in A:
            onright[v] += 1
        for i in range(0, length):
            val = A[i]
            tex = toextend[val]
            onright[val] -= 1
            for lval in onleft:
                diff = val - lval
                nextval = val + diff
                if nextval > 500 or nextval < 0 or onright[nextval] == 0:
                    continue
                if diff in tex:
                    c = tex[diff] + 1
                else:
                    c = 3
                if c > res:
                    res = c
                toextend[nextval][diff] = c
            if val not in onleft:
                #onleftl.append(val)
                onleft.add(val)
        return res
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        dp = {}
        
        for i in range(len(A)):
            for j in range(i+1, len(A)):
                dp[j, A[j]-A[i]] = dp.get((i, A[j]-A[i]), 1) + 1
        
        return max(dp.values())
                
                    
                
                

#from collections import Counter
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        if len(A)<=2:
            return len(A)
        dp = [{} for _ in range(len(A))]
        ans = 2
        for i in range(len(A)):
            for j in range(i+1,len(A),1):
                diff = A[j]-A[i]
                if diff in dp[i]:
                    dp[j][diff] = dp[i][diff] + 1
                else:
                    dp[j][diff] = 2
                ans = max(ans, dp[j][diff])
        return ans
                

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        n = len(A)
        ans = 2
        
        if n < 2:
            return n
        h = max(A)
        dp = [ {} for j in range(n)]
        # good idea, we did not know d, and we use dict here
        ans = 0
        for i in range(n):
            for j in range(i):
                d = A[i] - A[j]
                if d in dp[j]:
                    
                    dp[i][d] =  dp[j][d] + 1
                else:
                    dp[i][d] = 2
                ans = max(ans, dp[i][d])
        
        return ans


class Solution:
    def longestArithSeqLength(self, A):
        n = len(A)
        dp = [{} for _ in range(n)]
        res = 0
        for j in range(n):
            for i in range(j):
                d = A[j] - A[i]
                if d in dp[i]:
                    dp[j][d] = dp[i][d] + 1
                else:
                    dp[j][d] = 2
                res = max(res, dp[j][d])
        return res
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        dp = {}
        for i in range(len(A)):
            for j in range(i+1, len(A)):
                dp[j, A[j] - A[i]] = dp.get((i,A[j] - A[i]) , 1) + 1
        return max(dp.values())
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        
        if not A:
            return 0
        
        n = len(A)
        dp = dict()
        
        for r in range(1, n):
            for l in range(r):
                # pr
                dp[r, A[r] - A[l]] = dp.get((l, A[r] - A[l]), 1) + 1
                
        return max(dp.values())
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        dp = []
        for i in range(len(A)):
            dp.append({})
             
        # print(dp)
        res = 2
        for i in range(1, len(A)):
            for j in range(i):
                diff = A[i] - A[j]
                if diff in dp[j]:
                    dp[i][diff] = dp[j][diff] + 1
                else:
                    dp[i][diff] = 2
                res = max(res,dp[i][diff])
                
        return res
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        
        table = dict()
        
        for i in range(len(A)):
            for j in range(i+1,len(A)):
                
                _diff = A[j] - A[i]
                if (i,_diff) in list(table.keys()):
                    table[j,_diff] = table[i,_diff] + 1
                else:
                    table[j,_diff] = 2
                    
        return max(table.values())
                

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        
        res = 0
        dp = [{} for _ in range(len(A))]
        for i in range(1, len(A)):
            for j in range(i):
                diff = A[i] - A[j]
                if diff in dp[j]:
                    dp[i][diff] = dp[j][diff] + 1
                else:
                    dp[i][diff] = 2
                res = max(res, dp[i][diff])
                
        return res

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        dict1 = {}
        for i in range(len(A)):
            for j in range(i+1, len(A)):
                dict1[(A[j] - A[i], j)] = dict1.get((A[j] - A[i], i), 1) + 1
        return max(dict1.values())

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        dp = {}
        for i in range(len(A)):
            for j in range(i):
                dp[(i, A[i] - A[j])] = dp.get((j, A[i] - A[j]), 1) + 1

        return max(dp.values())        
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        dp = {}
        
        for i in range(len(A)):
            for j in range(i + 1, len(A)):
                dp[j, A[j] - A[i]] = dp.get((i, A[j] - A[i]), 1) + 1
                
        return max(dp.values())
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
#         Explanation
#         dp[index][diff] equals to the length of arithmetic sequence at index with difference diff.

#         Complexity
#         Time O(N^2)
#         Space O(N^2)
        
        n = len(A)
        if n< 2:
            return n
        dp = {}
        for i in range(n):
            for j in range(i+1, n):
                dp[(j, A[j]-A[i])] = dp.get((i, A[j]-A[i]), 1) + 1 # if there is no such key, len is 1 ==> A[i]
        return max(dp.values())
                    

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        dp = {}
        for i in range(len(A)):
            for j in range(i):
                dp[(i, A[i] - A[j])] = dp.get((j, A[i] - A[j]), 1) + 1
        return max(dp.values())
class Solution:

    def longestArithSeqLength(self, A: List[int]) -> int:
        
        n=len(A)
        DP={}
        for i in range(1,len(A)):
            for j in range(0,i):
                if (j, A[i]-A[j]) in DP:
                    DP[(i,A[i]-A[j])]= DP[(j, A[i]-A[j])]+1
                else:
                    DP[(i,A[i]-A[j])]=2
                
        return max(DP.values())
                
        
        n=len(A)
        dp={}
        for i in range(n):
            for j in range(i+1,n):
                dif = A[j]-A[i]
                if (i,dif) in dp :
                    dp[(j,dif)]=dp[(i,dif)]+1
                else:
                    dp[(j,dif)]=2
        return max(dp.values())
from itertools import repeat

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        length = len(A)
        onleft = 0
        onleftl = []
        onright = [*repeat(0, 501)]
        toextend = [{} for _ in range(501)]
        res = 2
        for v in A:
            onright[v] += 1
        for i in range(0, length):
            val = A[i]
            tex = toextend[val]
            onright[val] -= 1
            for lval in onleftl:
                diff = val - lval
                nextval = val + diff
                if nextval > 500 or nextval < 0 or onright[nextval] == 0:
                    continue
                c = tex.get(diff, 2) + 1
                if c > res:
                    res = c
                toextend[nextval][diff] = c
            b = (1<<val)
            if not (onleft & b):
                onleftl.append(val)
                onleft = (onleft | b)
        return res
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        # dp[index][diff] equals to the length of arithmetic sequence at index with difference diff.
        dp = {}
        for i in range(len(A)):
            for j in range(i + 1, len(A)):
                dp[j, A[j]-A[i]] = dp.get((i, A[j]-A[i]), 1) + 1
        return max(dp.values())
        
            
            

#[Runtime: 4284 ms, faster than 10.74%] DP
#O(N^2)
#NOTE: diff can be either positive or negative
#f[i]: the longest length of arithmetic subsequences who takes A[i] as the tail.
#f[i] = defaultdict(lambda: 1)
#f[i] = {diff: longest length}
#f[i] = max(f[i][d], f[j][d] += 1) for j < i and d:=A[i]-A[j]
from collections import defaultdict
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        dp = [defaultdict(lambda: 1) for _ in range(len(A))]
        for i, a in enumerate(A):
            for j in range(i):
                diff = A[i] - A[j]
                if dp[i][diff] < dp[j][diff] + 1: 
                    dp[i][diff] = dp[j][diff] + 1
        return max(max(lens.values()) for lens in dp)
# from collections import defaultdict
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        dp = defaultdict(int)
        for i in range(1, len(A)):
            for j in range(i):
                diff = A[i] - A[j]
                dp[(i,diff)] = dp[(j,diff)] + 1
        # print(dp)
        return max(dp.values()) + 1
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        
#         n = len(A);
#         res = 1;
        
#         for i in range(1,n):
#             for j in range(i):
#                 count = 2; 
#                 x = i + 1;
#                 y = i;
                
#                 while x < n and y < n:
#                     if A[x] - A[y] == A[i] - A[j]:
#                         count += 1;
#                         y = x
#                     x += 1;
#                 res = max(res, count);
        
#         return res;

        
        dp = dict();
        n = len(A);
        for i in range(n):
            for j in range(i+1,n):
                dp[(j, A[j] - A[i])] = dp.get((i, A[j] - A[i]), 1) + 1;
        
        return max(dp.values());
                
        
                        
                
                

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:

        d = {}
        for i in range(len(A)):
            for j in range(i + 1, len(A)):
                d[j, A[j] - A[i]] = d.get((i, A[j] - A[i]), 1) + 1
        return max(d.values())

class Solution:
    def longestArithSeqLength(self, A):
        dp = {}
        for i in range(len(A)):
            for j in range(i + 1, len(A)):
                dp[j, A[j] - A[i]] = dp.get((i, A[j] - A[i]), 1) + 1
        return max(dp.values())
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        dp = {}
        for i in range(len(A)):
            for j in range(i+1, len(A)):
                dp[(j, A[j]-A[i])] = dp.get((i, A[j]-A[i]), 1) + 1
        return max(dp.values())
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        
        dp={}
        for i in range(len(A)):
            for j in range(i+1,len(A)):
                
                dp[j,A[j]-A[i]]=dp.get((i,A[j]-A[i]),1)+1
        
        return max(dp.values())
from itertools import repeat
from collections import Counter

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        length = len(A)
        onleft = set()
        onleftl = []
        onright = dict(Counter(A))
        toextend = [{} for _ in range(501)]
        res = 2
        for i in range(0, length):
            val = A[i]
            tex = toextend[val]
            onright[val] -= 1
            for lval in onleftl:
                diff = val - lval
                nextval = val + diff
                if onright.get(nextval, 0) == 0:
                    continue
                c = tex.get(diff, 2) + 1
                if c > res:
                    res = c
                toextend[nextval][diff] = c
            if val not in onleft:
                onleftl.append(val)
                onleft.add(val)
        return res
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        #9,4,7,2,10
        #0,5,2,7,-1
        N=len(A)
        dp={}
        for i in range(N):
            for j in range(i+1,N):
                dp[j,A[j]-A[i]]=dp.get((i,A[j]-A[i]),1)+1
        return max(dp.values())

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        dp = {}
        ret = 0
        for i in range(len(A)):
            for j in range(i+1, len(A)):
                dp[j, A[j]-A[i]] = dp.get((i, A[j]-A[i]), 1) + 1

        return max(dp.values())
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        if len(A) < 2: return len(A)
        dp = [{} for _ in range(len(A))]
        res = 0
        for i in range(1, len(A)):
            for j in range(i):
                diff = A[i] - A[j] 
                if diff in dp[j]:
                    dp[i][diff] = dp[j][diff] + 1
                else:
                    dp[i][diff] = 2
                res = max(res, dp[i][diff])
        return res

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        dp = {}
        for i in range(len(A)):
            for j in range(i+1, len(A)):
                dp[j, A[j]-A[i]] = dp.get((i, A[j]-A[i]), 1) + 1
        return max(dp.values())
class Solution:
    def longestArithSeqLength(self, A):
        dp = {}
        for i in range(len(A)):
            for j in range(i + 1, len(A)):
                dp[j, A[j] - A[i]] = dp.get((i, A[j] - A[i]), 1) + 1
                
        #print(dp)
        return max(dp.values())
class Solution:
    def longestArithSeqLength(self, nums: List[int]) -> int:
        n = len(nums)
        if n <= 2:
            return n
        dp = dict()
        for i in range(n):
            for j in range(i + 1, n):
                d = nums[j] - nums[i]
                if (i, d) in list(dp.keys()):
                    dp[(j, d)] = dp[(i, d)] + 1
                else:
                    dp[(j, d)] = 2
        return max(dp.values())

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        res, d = 0, [{} for _ in range(len(A))]
        for i in range(1, len(A)):
            for j in range(i):
                diff = A[i] - A[j]
                if diff in d[j]:
                    d[i][diff] = d[j][diff] + 1
                else:
                    d[i][diff] = 2
                res = max(res, d[i][diff])
        return res

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        N = len(A)
        dp = [[2] * N for i in range(N)]
        ans = 0
        for i in range(N):
            pos = {}
            for j in range(i):
                x = 2*A[j] - A[i]
                if x in pos:
                    dp[i][j] = max(dp[i][j], 1 + dp[j][pos[x]])
                ans = max(ans, dp[i][j])
                pos[A[j]] = j
        
        return ans
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        dp = {}
        for i in range(len(A)):
            for j in range(i+1, len(A)):
                dp[(j, A[j]-A[i])] = dp.get((i, A[j]-A[i]), 1) + 1
        return max(dp.values())

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        #446. Arithmetic Slices II - Subsequence
        '''
        https://leetcode.com/problems/longest-arithmetic-sequence/discuss/274611/JavaC++Python-DP
        dp[diff][index] + 1 equals to the length of arithmetic sequence at index with difference diff.
        '''
        '''
        Input: [3,6,9,12]
Output: 4
Explanation: 
The whole array is an arithmetic sequence with steps of length = 3.
Input: [9,4,7,2,10]
Output: 3
Explanation: 
The longest arithmetic subsequence is [4,7,10].
Input: [20,1,15,3,10,5,8]
Output: 4
Explanation: 
The longest arithmetic subsequence is [20,15,10,5].
        '''
        '''
Input: [3,6,9,12]
Output: 4
Explanation: 
The whole array is an arithmetic sequence with steps of length = 3.
        defaultdict(<class 'int'>, {})
defaultdict(<class 'int'>, {(3, 0): 0, (3, 1): 1})
defaultdict(<class 'int'>, {(3, 0): 0, (3, 1): 1, (6, 0): 0, (6, 2): 1})
defaultdict(<class 'int'>, {(3, 0): 0, (3, 1): 1, (6, 0): 0, (6, 2): 1, (9, 0): 0, (9, 3): 1})
defaultdict(<class 'int'>, {(3, 0): 0, (3, 1): 1, (6, 0): 0, (6, 2): 1, (9, 0): 0, (9, 3): 1, (3, 2): 2})
defaultdict(<class 'int'>, {(3, 0): 0, (3, 1): 1, (6, 0): 0, (6, 2): 1, (9, 0): 0, (9, 3): 1, (3, 2): 2, (6, 1): 0, (6, 3): 1})
defaultdict(<class 'int'>, {(3, 0): 0, (3, 1): 1, (6, 0): 0, (6, 2): 1, (9, 0): 0, (9, 3): 1, (3, 2): 2, (6, 1): 0, (6, 3): 1, (3, 3): 3})
      ''' 
        #longest arithmetic subseq
    
        '''
        Input:
[24,13,1,100,0,94,3,0,3]
Output:
3
Expected:
2
        '''
        '''
        Input:
[0,8,45,88,48,68,28,55,17,24]
Output:
4
Expected:
2
        '''
      
  
        
        '''Len = len(A)
        res = 0
        for i in range(aLen):
            for j in range(i+1, aLen):
                diff = A[j]-A[i]
                target = A[j] + diff 
                count = 2 
                idx = j+1
                while idx < aLen:
                    if A[idx] == target:
                        count += 1
                        target = target + diff 
                    idx += 1
                res = max(res, count)
        return res'''
        
       
                
        '''aLen =len(A)
        res = 0 
        dp = [{} for _ in range(aLen)]
        for i in range(aLen):
            for j in range(i):
                diff = A[i]-A[j]
                if diff in dp[j]:
                    dp[i][diff] = 1 + dp[j][diff]
                else:
                    dp[i][diff] = 2
                res = max(res, dp[i][diff])
        return res '''
        
        '''aLen = len(A)
        res = 0 
        dp = [{} for _ in range(aLen)]
        for i in range(aLen):
            for j in range(i):
                diff = A[i] - A[j]
                if diff in dp[j]:
                    dp[i][diff] = 1+ dp[j][diff]
                else:
                    dp[i][diff] = 2 
                res = max(res, dp[i][diff])
        return res '''
        
        aLen = len(A)
        res = 0 
        dp = [{} for _ in range(aLen)]
        for i in range(aLen):
            for j in range(i):
                diff = A[i]-A[j]
                if diff in dp[j]:
                    dp[i][diff] = dp[j][diff] + 1
                else:
                    dp[i][diff] = 2 
                res = max(res, dp[i][diff])
        return res 
                    
            
 

                    
                    
 

        
        '''
        #why is this solution the fastest??????
        nums_map = {}
        for i,n in enumerate(A):
            nums_map[n] = nums_map.get(n, [])
            nums_map[n].append(i)
        max_length = 2
        for i, n in enumerate(A):
            for j in range(i+1, len(A)):
                m = A[j]
                target = m + (m-n)
                length = 2
                last_index = j
                found = True
                while target in nums_map and found:
                    found = False
                    for index in nums_map[target]:
                        if index > last_index:
                            last_index = index
                            length += 1
                            target += m-n
                            max_length = max(max_length, length)
                            found = True
                            break
                    if not found:
                        break
        return max_length'''
                
                

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        # dp[index][diff]  equals to the length of arithmetic sequence at index with difference diff.
        # O(n^2)
        # A is unsorted 
        
        dp = {}
        for i in range(len(A)):
            for j in range(i+1, len(A)):
                dp[j, A[j] - A[i]] = dp.get((i, A[j] - A[i]), 1) + 1  # A is unsorted, so it is dp.get((i, A[j]-A[i])) not dp.get(j-1, diff)
        
        
        return max(dp.values())
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        ##pretty sure this soln works, it just takes too long.
#         def searchList(num: int, lst: List[int]) -> bool:
#             for curr in lst:
#                 if curr == num:
#                     return True
#             return False
        
#         longest = 2 #shorest array we're given is 2        
#         info = []
#         subseq = {
#             'diff': A[1] - A[0],
#             'seq': A[0:2]
#         }
#         info.append(subseq)
        
        
#         for i in range(2, len(A)):
#             seen = []
#             curr = A[i]
#             prev = A[i - 1]
#             for sub in info:
#                 if curr - sub['seq'][-1] == sub['diff']:
#                     sub['seq'].append(curr)
#                     seen.append(sub['seq'][-1])
#                     if len(sub['seq']) > longest:
#                         longest = len(sub['seq'])            
#             for num in A[0:i]:
#                 #if an element hasn't been seen, append another info subseq dict
#                 #with the current element and the unseen one
#                 if not searchList(num, seen):
#                     diff = curr - num
#                     #if curr + diff < 0, then we know that the subseq will not continue, so don't
#                     #bother keeping it
#                     if curr + diff >= 0:
#                         info.append({
#                             'diff': curr - num,
#                             'seq': [num, curr]
#                         })
#         return longest

        dp = {}
        for i in range(len(A)):
            for j in range(i + 1, len(A)):
                dp[j, A[j] - A[i]] = dp.get((i, A[j] - A[i]), 1) + 1
        return max(dp.values())
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        # 9 ,4    ,     7,         10
        #    ^
#             (1,-5): 2  (2, 3): 2   (3, 3):3
#                                    (3, -5):3
        
        dict = collections.defaultdict(lambda: 0)
        
        for i in range(len(A) - 1):    
            for j in range(i + 1, len(A)):
                dict[(j, A[j] - A[i])] = dict.get((i, A[j] - A[i]), 1) + 1
        
        return max(dict.values())
        


class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        dp = {}
        for i in range(len(A)):
            for j in range(i + 1, len(A)):
                dp[(j, A[j] - A[i])] = dp.get((i, A[j] - A[i]), 1) + 1 
        return max(dp.values())

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        dp = [{} for i in range(len(A))]
        longest = 0

        for i in range(len(A)):
            for j in range(0, i):
                diff = A[i] - A[j]
                if diff in dp[j]:
                    dp[i][diff] = dp[j][diff] + 1
                else:
                    dp[i][diff] = 2
                longest = max(longest, dp[i][diff])

        return longest
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        #446. Arithmetic Slices II - Subsequence
        '''
        https://leetcode.com/problems/longest-arithmetic-sequence/discuss/274611/JavaC++Python-DP
        dp[diff][index] + 1 equals to the length of arithmetic sequence at index with difference diff.
        '''
        '''
        Input: [3,6,9,12]
Output: 4
Explanation: 
The whole array is an arithmetic sequence with steps of length = 3.
Input: [9,4,7,2,10]
Output: 3
Explanation: 
The longest arithmetic subsequence is [4,7,10].
Input: [20,1,15,3,10,5,8]
Output: 4
Explanation: 
The longest arithmetic subsequence is [20,15,10,5].
        '''
        '''
Input: [3,6,9,12]
Output: 4
Explanation: 
The whole array is an arithmetic sequence with steps of length = 3.
        defaultdict(<class 'int'>, {})
defaultdict(<class 'int'>, {(3, 0): 0, (3, 1): 1})
defaultdict(<class 'int'>, {(3, 0): 0, (3, 1): 1, (6, 0): 0, (6, 2): 1})
defaultdict(<class 'int'>, {(3, 0): 0, (3, 1): 1, (6, 0): 0, (6, 2): 1, (9, 0): 0, (9, 3): 1})
defaultdict(<class 'int'>, {(3, 0): 0, (3, 1): 1, (6, 0): 0, (6, 2): 1, (9, 0): 0, (9, 3): 1, (3, 2): 2})
defaultdict(<class 'int'>, {(3, 0): 0, (3, 1): 1, (6, 0): 0, (6, 2): 1, (9, 0): 0, (9, 3): 1, (3, 2): 2, (6, 1): 0, (6, 3): 1})
defaultdict(<class 'int'>, {(3, 0): 0, (3, 1): 1, (6, 0): 0, (6, 2): 1, (9, 0): 0, (9, 3): 1, (3, 2): 2, (6, 1): 0, (6, 3): 1, (3, 3): 3})
      ''' 
        #longest arithmetic subseq
    
        '''
        Input:
[24,13,1,100,0,94,3,0,3]
Output:
3
Expected:
2
        '''
        '''
        Input:
[0,8,45,88,48,68,28,55,17,24]
Output:
4
Expected:
2
        '''
      
  
        
        '''Len = len(A)
        res = 0
        for i in range(aLen):
            for j in range(i+1, aLen):
                diff = A[j]-A[i]
                target = A[j] + diff 
                count = 2 
                idx = j+1
                while idx < aLen:
                    if A[idx] == target:
                        count += 1
                        target = target + diff 
                    idx += 1
                res = max(res, count)
        return res'''
        
       
                
        '''aLen =len(A)
        res = 0 
        dp = [{} for _ in range(aLen)]
        for i in range(aLen):
            for j in range(i):
                diff = A[i]-A[j]
                if diff in dp[j]:
                    dp[i][diff] = 1 + dp[j][diff]
                else:
                    dp[i][diff] = 2
                res = max(res, dp[i][diff])
        return res '''
        
        '''aLen = len(A)
        res = 0 
        dp = [{} for _ in range(aLen)]
        for i in range(aLen):
            for j in range(i):
                diff = A[i] - A[j]
                if diff in dp[j]:
                    dp[i][diff] = 1+ dp[j][diff]
                else:
                    dp[i][diff] = 2 
                res = max(res, dp[i][diff])
        return res '''
        
        aLen = len(A)
        dp = [{} for _ in range(aLen)]
        res = 0 
        for i in range(aLen):
            for j in range(i):
                diff = A[i] - A[j]
                if diff in dp[j]:
                    dp[i][diff] = dp[j][diff] + 1
                else:
                    dp[i][diff] = 2 
                res= max(res, dp[i][diff])
        return res 
                    
            
 

                    
                    
 

        
        '''
        #why is this solution the fastest??????
        nums_map = {}
        for i,n in enumerate(A):
            nums_map[n] = nums_map.get(n, [])
            nums_map[n].append(i)
        max_length = 2
        for i, n in enumerate(A):
            for j in range(i+1, len(A)):
                m = A[j]
                target = m + (m-n)
                length = 2
                last_index = j
                found = True
                while target in nums_map and found:
                    found = False
                    for index in nums_map[target]:
                        if index > last_index:
                            last_index = index
                            length += 1
                            target += m-n
                            max_length = max(max_length, length)
                            found = True
                            break
                    if not found:
                        break
        return max_length'''
                
                

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        dp = {}
        n = len(A)
        for i in range(n):
            for j in range(i + 1, n):
                dp[j, A[j] - A[i]] = dp.get((i, A[j] - A[i]), 1) + 1
        return max(dp.values())
from collections import defaultdict

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        dp = {}
        for i in range(len(A)):
            for j in range(i + 1, len(A)):
                dp[(j, A[j] - A[i])] = dp.get((i, A[j] - A[i]), 1) + 1

        return max(dp.values())
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        stored = [{} for i in range(len(A))]
        best = 0
        for index, value in enumerate(A):
            if index == 0:
                continue
            for compare in range(index):
                difference = value - A[compare]
                stored[index][difference] = 1 + stored[compare].get(difference, 1)
                best = max(best, stored[index][difference])
        return best
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        dp = {}
        #dictionary of pairs (i pointer, dist)
        
        # i pointer iterates from 1index to the end
        for i in range(1, len(A)):
            # j pointer iterates from 0 to just left of i pointer then resets
            for j in range(0, len(A[:i])):
                
                #finds the difference of the two values
                d = A[i] - A[j]
                
                #checks to see if the same diff exists at j
                if (j, d) in dp:
                    #if j,d is in dp then add 1 because the value at i has the same difference and set that as i,d
                    dp[i, d] = dp[j, d] + 1
                #if not then its set to two because that accounts for the i,j as two integers
                else:
                    dp[i, d] = 2
                    
        #return what ever is the highest value of all the keys in the dictionary is
        return max(dp.values())
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        # dp[(index, diff)] equals to the length of arithmetic sequence at index with difference diff.
        dp = collections.defaultdict(int)
        
        for i in range(len(A)):
            for j in range(i + 1, len(A)):
                dp[(j, A[j] - A[i])] = dp[(i, A[j] - A[i])] + 1
        
        return max(dp.values()) + 1
class Solution:
    def longestArithSeqLength(self, A):
        n = len(A)
        dp = [{} for _ in range(n)]
        res = 0
        for i in range(n):
            for j in range(i + 1, n):
                d = A[j] - A[i]
                if d in dp[i]:
                    dp[j][d] = dp[i][d] + 1
                else:
                    dp[j][d] = 2
                res = max(res, dp[j][d])
        return res
class Solution:
    def longestArithSeqLength(self, A):
        dp = {}
        for i in range(len(A)):
            for j in range(i + 1, len(A)):
                dp[j, A[j] - A[i]] = dp.get((i, A[j] - A[i]), 1) + 1
        return max(dp.values())
#     def longestArithSeqLength(self, A: List[int]) -> int:
#         m = {}
        
#         m[A[1] + A[1]-A[0]] = (2,A[1]-A[0])
#         for i in range(2, len(A)):
#             if A[i] in m:
#                 counter, d = m[A[i]]
#                 del m[A[i]]
#                 m[A[i]+d] = (counter+1, d)
#             else:
#                 for j in range(0, i):
#                     d = A[i]-A[j]
#                     m[A[i]+d] = (2,d)
#         # print(m)
#         return max([counter for counter,_ in list(m.values())])

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        n , dp = len(A) , {}
        for i in range(n):
            for j in range(i + 1 , n):
                dp[j , A[j] - A[i]] = dp.get((i , A[j] - A[i]) , 1) + 1
        return max(dp.values())
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        dp = [dict() for _ in range(len(A))]
        res = 2
        for i in range(len(A)):
            for j in range(i):
                diff = A[i] - A[j]
                dp[i][diff] = dp[j].get(diff, 1) + 1
                res = max(dp[i][diff], res)
        
        return res
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        dp = []
        
        for i, x in enumerate(A):
            Dict = collections.defaultdict(int)
            dp.append(Dict)
            for j in range(i):
                diff = x - A[j]
                dp[i][diff] = dp[j][diff] + 1
        
        return max(max(y.values()) for y in dp) + 1
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        n = len(A)
        dp = [collections.defaultdict(int) for _ in range(n)] 
        res = 0
        for i in range(1, n):
            for j in range(0, i):
                diff = A[i]-A[j]
                dp[i][diff] = dp[j][diff] + 1
                    
                res = max(res, dp[i][diff])
        return res + 1
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        N = len(A) 
        dp = [dict() for _ in range(N)]
        ret = 1
        for i in range(N):
            for j in range(i + 1, N):
                diff = A[j] - A[i]
                dp[j][diff] = dp[i].get(diff, 1) + 1
                ret = max(ret, dp[j][diff])
        return ret
        # def calc(A):
        #     for i in range(len(A) - 1, -1, -1):
        #         for j in range(i + 1, len(A)):
        #             if A[j] < A[i]:
        #                 continue
        #             diff = A[j] - A[i]
        #             memo[i][diff] = max(memo[i].get(diff, 0), memo[j].get(diff, 1) + 1)
        #             ret = max(ret, memo[i][diff])
        #     return ret
        # 
        # return max(
        #     calc(A), calc(list(reversed(A)))
        # )

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:

        dp = {}
        for i in range(len(A)):
            for j in range(i + 1, len(A)):
                dp[j, A[j] - A[i]] = dp.get((i, A[j] - A[i]), 1) + 1
        return max(dp.values())
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        dp = {}
        for i in range(len(A)):
            for j in range(i + 1, len(A)):
                dp[j, A[j] - A[i]] = dp.get((i, A[j] - A[i]), 1) + 1
        return max(dp.values())     
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        N = len(A)
        
        table = []
        out = 0
        for i in range(N):
            table.append(defaultdict(dict))
            for j in range(0, i):
                js_table = table[j]
                diff = A[i]-A[j]
                if diff not in js_table:
                    table[i][diff] = 2
                else:
                    table[i][diff] = table[j][diff]+1
                
                out = max(table[i][diff], out)
                
        # print(table)
        return out
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        dp = {}
        for i in range(len(A)):
            for j in range(i+1, len(A)):
                dp[(j, A[j] - A[i])] = dp.get((i,A[j]-A[i]),1) + 1
        return max(dp.values())
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        dp = {}
        n = len(A)
        for i in range(n):
            for j in range(i+1, n):
                dp[(j, A[j] - A[i])] = dp.get((i, A[j] - A[i]), 1) + 1
                
        return max(dp.values())
                    

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        ans = 0
        n = len(A)
        dp = [collections.defaultdict(int) for _ in range(n)]
        for i in range(1, n):
            for j in range(i):
                diff = A[i] - A[j]
                if diff in dp[j]:
                    dp[i][diff] = dp[j][diff] + 1
                else:
                    dp[i][diff] = 2
                ans = max(ans, dp[i][diff])
        return ans
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        dic={}
        for i in range(len(A)):
            for j in range(i+1,len(A)):
                dic[(j,A[j]-A[i])]=dic.get((i,A[j]-A[i]),1)+1
        return max(dic.values())
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        
        dp = {}
        for i in range(len(A)):
            for j in range(i + 1, len(A)):
                dp[j, A[j] - A[i]] = dp.get((i, A[j] - A[i]), 1) + 1
        return max(dp.values())

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        dp = {}
        for i in range(len(A)):
            for j in range(i+1, len(A)):
                d = A[j]-A[i]
                dp[j,d] = dp.get((i,d),1)+1
                
                
        # print(dp)
        return max(dp.values())
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        dp = {}
        N = len(A) 
        for i in range(N):
            for j in range(i + 1, N):
                dp[j, A[j] - A[i]] = dp.get((i, A[j] - A[i]), 1) + 1
        return max(dp.values())
        # def calc(A):
        #     for i in range(len(A) - 1, -1, -1):
        #         for j in range(i + 1, len(A)):
        #             if A[j] < A[i]:
        #                 continue
        #             diff = A[j] - A[i]
        #             memo[i][diff] = max(memo[i].get(diff, 0), memo[j].get(diff, 1) + 1)
        #             ret = max(ret, memo[i][diff])
        #     return ret
        # 
        # return max(
        #     calc(A), calc(list(reversed(A)))
        # )

'''
[9,4,7,2,10]
 i
          j

:  (freq, diff)
{9: {0:1}, 
{4: {0,1}, {-5, 2}}
{7: {0,1}, {-2,2}, {3,2}}
{2: {0,1}, {-7:2}, {-2, 2}, {-5,2}}
{10:{0,1}, {1, 2}, {6,2}, {3,3}}
 
 [20,1,15,3,10,5,8]
  i       j
  
  {
  20:(1,0)
  1:(2,-19)
  15:(2,-5)
  3(4,5)
  
}

So we keep running hashmap of all numbers we meet along with its available arith difference along with its frequency. Then we do n^2 loop through all numbers, each time calculating the difference between number i and number j and trying to see if there is that difference avaible in the hashmap of j pointer. At this point if there is a difference match, we add on, else we start at frequency of.

then we can keep track of max throughout and return the max

O(n^2)
O(n^2)


'''
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        d = collections.defaultdict(dict)
        sol = 0
        for i in range(1, len(A)):
            for j in range(i):
                curDiff = A[i] - A[j]
                if curDiff in d[j]:
                    d[i][curDiff] = d[j][curDiff] + 1
                else:
                    d[i][curDiff] = 2
                    
                sol = max(sol, d[i][curDiff])
        return sol
        
            

            

class Solution:
    def longestArithSeqLength(self, nums: List[int]) -> int:
        n = len(nums)
        
        if n == 2:
            return n
        
        dp = [{} for i in range(n)]
        max_len = 0
        for i in range(1, n):
            for j in range(i):
                diff = nums[i] - nums[j] 
                if diff in dp[j]:
                    dp[i][diff] = max(2, 1 + dp[j][diff])
                else:
                    dp[i][diff] = 2
                max_len = max(max_len, dp[i][diff])
        
        return max_len
class Solution:
    def longestArithSeqLength(self, A):
        dp = {}
        for i in range(len(A)):
            for j in range(i + 1, len(A)):
                dp[j, A[j] - A[i]] = dp.get((i, A[j] - A[i]), 1) + 1
        return max(dp.values())

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        res = 0
        dic = {}
        for i in range(len(A)):
            dic[i] = {}
            for j in range(i):
                key = A[i]-A[j]
                if key in dic[j]:
                    dic[i][key] = dic[j][key] + 1
                else:
                    dic[i][key] = 2
                res = max(res, dic[i][key])
        return res

#print (Solution.longestArithSeqLength(Solution, [20,1,15,3,10,5,8]))

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        if len(A)<=2:
            return len(A)
        dp=[{} for i in range(len(A))]
        ans=2
        for i in range(1,len(A)):
            for j in range(i):
                diff=A[i]-A[j]
                if diff in dp[j]:
                    dp[i][diff]=dp[j][diff]+1
                
                else:
                    dp[i][diff]=2
                
                ans=max(ans,dp[i][diff])
        
        return ans
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        dp = {}
        for i in range(len(A)):
            for j in range(i+1,len(A)):
                diff = A[j]-A[i]
                dp[j,diff] = dp.get((i,diff),1)+1
        return max(dp.values())

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        dp = {} #stores the longest len of arithmetic subsequence for each pair of (idx, diff)
        for i in range(len(A)):
            for j in range(i+1, len(A)):
                dp[j, A[j] - A[i]] = dp.get((i, A[j] - A[i]), 1) + 1 
        return max(dp.values())
    
    #for A[i] and A[j], we store by ith idx and their diff the length of subsequence that follows diff A[j]-A[i]. Next time when cur A[j] becomes the first num and we find another item in A that's A[j]-A[i] away from cur A[j], we increment this subsequence's length by 1

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        n = len(A)
        dp = [dict() for _ in range(n)] 
        max_len = 0
        for i in range(n):
            for j in range(i):
                diff = A[i] - A[j]
                dp[i][diff] = dp[j].get(diff, 1) + 1
                # print(dp[j][diff])
                max_len = max(max_len, dp[i][diff])
        return max_len
                
                

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        # dp[index][diff]
        dp = {}
        for i in range(len(A)):
            for j in range(i+1, len(A)):
                dp[j, A[j]- A[i]] = dp.get((i, A[j] - A[i]), 1) + 1
        return max(dp.values())
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        dp = collections.defaultdict(int)
        
        for i in range(len(A)):
            for j in range(i + 1, len(A)):
                dp[(j, A[j] - A[i])] = dp[(i, A[j] - A[i])] + 1
        
        return max(dp.values()) + 1
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        n = len(A)
        longest = 2
        dp = [{} for _ in range(n)]
        
        for i in range(1, n):
            for j in range(i):
                diff = A[i] - A[j]
                if diff in dp[j]:
                    dp[i][diff] = dp[j][diff] + 1
                else:
                    dp[i][diff] = 2
                
                longest = max(longest, dp[i][diff])
        
        return longest

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        N = len(A)
        ans = 2
        dp = [{} for _ in range(N)]
        for i in range(N):
            for j in range(i):
                diff = A[j] - A[i]
                dp[i][diff] = dp[j].get(diff,1)+1
                ans = max(ans,dp[i][diff])
        return ans
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        cache = dict()
        maxi = 0
        for i in range(len(A)):
            for j in range(i + 1, len(A)):
                diff = A[j]-A[i]
                if (diff, i) in cache:
                    cache[diff, j]  = 1 + cache[diff, i]
                else:
                    cache[diff, j] = 2
                if maxi < cache[diff, j]:
                    maxi = cache[diff, j]
        return maxi
# class Solution:
#     def longestArithSeqLength(self, arr: List[int]) -> int:
#         if not arr:
#             return 0
        
#         dp = [collections.defaultdict(lambda : 1) for _ in range(len(arr))]
#         ret = 1
#         for i in range(len(arr)):
#             for j in range(i):
#                 dp[i][arr[i] - arr[j]] = 1 + dp[j][arr[i] - arr[j]]
#                 ret = max(ret, dp[i][arr[i] - arr[j]])
        
#         return ret

class Solution:
   def longestArithSeqLength(self, A):
        dp = {}
        for i in range(len(A)):
            for j in range(i + 1, len(A)):
                dp[j, A[j] - A[i]] = dp.get((i, A[j] - A[i]), 1) + 1
        return max(dp.values())
from collections import defaultdict

class Solution:
    def longestArithSeqLength(self, arr: List[int]) -> int:
        dp = defaultdict(dict)
        n = len(arr)
        max_len = 0
        for i in range(n):
            for j in range(i):
                diff = arr[i] - arr[j]
                dp[i][diff] = dp[j].get(diff, 0) + 1
                max_len = max(max_len, dp[i][diff])
        
        return max_len + 1

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        # dp[i][k] = longestArithSeqLength(A[:i+1]) with step size k
        dp = []
        res = 0
    
        for i in range(len(A)):
            step2len = defaultdict(int)
            dp.append(step2len)
            for prev_i in range(i):
                step = A[i] - A[prev_i]
                prev_step = dp[prev_i][step]
                dp[i][step] = prev_step + 1
                res = max(res, dp[i][step])
        
        return res + 1
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        dp = []
        n = len(A)
        ans = 0
        
        for _ in range(n):
            dp.append({})
        
        for i in range(n):
            for j in range(0, i):
                if i == 0:
                    dp[i][0] = 1
                else:   
                    diff = A[i] - A[j]

                    if diff in dp[j]:
                        dp[i][diff] = dp[j][diff] + 1
                    else:
                        dp[i][diff] = 2
                        
                    ans = max(ans, dp[i][diff])
        
        
        return ans
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        N = len(A)
        dp = [[2] * N for i in range(N)]
        ans = 0
        for i in range(N):
            pos = {}
            for j in range(i):
                x = 2 * A[j] - A[i]
                if x in pos:
                    dp[i][j] = max(dp[i][j],1 + dp[j][pos[x]])
                ans = max(ans,dp[i][j])
                pos[A[j]] = j
        return ans
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        idx_diff_count = {}
        for i in range(1,len(A)):
            for j in range(i):
                diff = A[i] - A[j]
                if (j,diff) in idx_diff_count:
                    idx_diff_count[i,diff] = idx_diff_count[j,diff] + 1
                else:
                    idx_diff_count[i,diff] = 2
        return max(idx_diff_count.values())
class Solution:
    def longestArithSeqLength(self, A: 'List[int]') -> int:
        n = len(A)
        if n == 2:
            return 2
        dic = {0:{0:1}}
        longest = 0
        for i in range(1,n):
            if i not in dic:
                dic[i] = {0:1}
            for j in range(i):
                diff = A[i] - A[j]
                if diff not in dic[j]:
                    dic[i][diff] = 2
                else:
                    dic[i][diff] = dic[j][diff] + 1
                longest = max(longest,dic[i][diff])
        return longest
from collections import defaultdict

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        count = defaultdict(lambda: 1)
        n = len(A)
        for i in range(n):
            for j in range(i + 1, n):
                count[j, (A[j] - A[i])] = count[i, (A[j] - A[i])] + 1
        return max(count.values())

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        maxVal = 1
        cur = collections.defaultdict(dict)
        for i, val in enumerate(A):
            for j in range(i):
                dist = A[i] - A[j]
                
                cur[i][dist] = 1 + cur[j].get(dist, 1)
                maxVal = max(maxVal, cur[i][dist])
        return maxVal

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        
        n = len(A)
        dp = {}
        for i in range(n):
            for j in range(i+1, n):
                diff = A[j] - A[i]
                if (i, diff) in dp:
                    dp[(j, diff)] = dp[(i, diff)] + 1
                else:
                    dp[(j, diff)] = 2
        
        return max(dp.values())
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        dp = defaultdict(dict)
        
        ans = 0
        
        dp[0][0] = 1
        
        for i in range(1, len(A)):
            for j in range(i):
                diff = A[i]-A[j]
                dp[i][diff] = dp[j].get(diff, 1) + 1
                ans = max(ans, dp[i][diff])
                
        return ans
            

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        d = collections.defaultdict(int)
        for i in range(1,len(A)):
            for j in range(i):
                dif = A[i] - A[j]
                if (j,dif) in d:
                    d[i,dif] = d[j,dif] + 1
                else:
                    d[i,dif] = 2
        return max(d.values())
from collections import defaultdict
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        # 3, 6, 9, 12
        # 1,3 -> 0
        # 2,6 -> 0
        # 2,3 -> 1
        # 3,9 -> 0
        # 3,6 -> 1
        # 3,3 -> 2
        count = 2
        diff_map = defaultdict(dict)
        for i in range(len(A)):
            for j in range(0, i):
                #print(i, j, diff_map[i])
                diff = A[i] - A[j]
                diff_map[i][diff] = 1 + diff_map[j].get(diff, 1)
                count = max(count, diff_map[i][diff])
        return count

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        if(len(A)==2):
            return(len(A))
        d=[{} for i in range(len(A))]
        m=0
        for i in range(1,len(A)):
            for j in range(i):
                k=A[i]-A[j]
                if k in d[j]:
                    d[i][k]=max(2,1+d[j][k])
                else:
                    d[i][k]=2
                m=max(m,d[i][k])
        return(m)
        
        

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        self.results = {}
        # self.results[i][diff] is the length of the longest subsequence that includes index i to the end, with difference diff
        
        self.results[len(A) - 1] = {}
        for i in range(len(A) - 2, -1, -1):
            self.results[i] = {}

            for j in range(i + 1, len(A), 1):
                diff = A[i] - A[j]

                possibility = self.results[j].get(diff, 1)                
                if 1 + possibility > self.results[i].get(diff, 0):
                    self.results[i][diff] = 1 + possibility

        result = 1
        for i in range(0, len(A) - 1, 1):
            for value in list(self.results[i].values()):
                result = max(result, value)
        
        return result
            
        
        
        
        
        
        
        
    

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        
        sequences = [defaultdict(int) for _ in range(len(A))]
        
        for i in range(1, len(A)):
            for j in range(i):
                diff = A[i] - A[j]
                sequences[i][diff] = max(sequences[j][diff]+1, sequences[i][diff])
        
        return max(max(mapping.values()) for mapping in sequences[1:]) + 1
                

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        # find step between number, and count max of consequence index.
        # step: [arr], # preserve order?
        
        if not A:
            return 0
        if len(A) == 2:
            return 2
        
        dp = [{} for a in range(len(A))]
        max_len = 0
        for i in range(1, len(A)):
            for j in range(i):
                diff = A[i] - A[j]
                if diff in dp[j]:
                    dp[i][diff] = max(2, 1+dp[j][diff])
                else:
                    dp[i][diff] = 2
                max_len = max(max_len, dp[i][diff])
        return max_len
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        nums = A
        n = len(nums)
        if n == 2:
            return n
        dp = [{} for i in range(n)]
        max_len = 0
        for i in range(1, n):
            for j in range(i):
                diff = nums[i] - nums[j]
                if diff in dp[j]:
                    dp[i][diff] = max(2, 1 + dp[j][diff])
                else:
                    dp[i][diff] = 2
                max_len = max(max_len, dp[i][diff])
        return max_len
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        if len(A) < 3:
            return len(A)
        d = {k:{} for k in set(A)}
        for j in range(len(A)):
            zero = d[A[j]].get(0, 0) + 1
            for step, l in list(d[A[j]].items()):
                d.setdefault(A[j] + step, {})
                prev = d[A[j] + step].get(step, 0)
                d[A[j] + step][step] = max(prev, l + 1)
            for i in range(j):
                d.setdefault(2 * A[j] - A[i], {})
                d[2 * A[j] - A[i]].setdefault(A[j] - A[i], 2)
                # print(i, j, d[2 * A[j] - A[i]])
            d[A[j]] = {0: zero}
            # print(d)
        res = max([max(v.values()) for v in list(d.values())])
        return res

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        if len(A) <= 2:
            return len(A)
        
        dp = [{} for _ in range(len(A))]
        ans = 0
        
        for i in range(1, len(A)):
            for j in range(i):
                diff = A[i] - A[j]
                
                if diff in dp[j]: # see if i's distance from j  is possible from a prev number to j
                    dp[i][diff] = max(2, 1 + dp[j][diff])
                else:
                    dp[i][diff] = 2 # len 2 is always possible
                    
                ans = max(ans, dp[i][diff])
                
        return ans
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        from collections import Counter, defaultdict
        
        min_val = min(A)
        max_val = max(A)
        
        global_best = -1
        dp = {}
        
        for i, v in enumerate(A):
            dp[i] = {}
            
            # print(f'--------- PROCESSING INDEX {i}')
            for j, w in enumerate(A[:i]):
                d = v - w
                dp[i][d] = dp[j].get(d, 1) + 1
                global_best = max(global_best, dp[i][d])
            
        # print(dp)
        
        return global_best
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        # dp = {}
        # for i, a2 in enumerate(A[1:], start=1):
        #     for j, a1 in enumerate(A[:i]):
        #         print('a2: ' + str(a2) + '; a1: ' + str(a1))
        #         d = a2 - a1
        #         if (j, d) in dp:
        #             dp[i, d] = dp[j, d] + 1
        #         else:
        #             dp[i, d] = 2
        #         print(dp)
        # return max(dp.values())
        dp = {}
        for i, a2 in enumerate(A[1:], start=1):
            for j, a1 in enumerate(A[:i]):
                d = a2 - a1
                if (j, d) in dp:
                    dp[i, d] = dp[j, d] + 1
                else:
                    dp[i, d] = 2
        return max(dp.values())
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        #nested dict: k1 = difference, k2 = end idx of difference, val = count
        #max_len
        record=defaultdict(lambda : defaultdict(int))
        max_len = 0
        #go over list
        #   for num, get difference with all previous nums
        #       if difference in dict and second key is the same as previous num, add to dict[dif][self] = count+1, update max_len
        #return max_len
        for i, v_i in enumerate(A):
            for j in range(0, i):
                dif = v_i - A[j]
                if dif in record and j in record[dif]: record[dif][i] = record[dif][j]+1
                else: record[dif][i] = 1
                max_len = max(max_len, record[dif][i])
        return max_len+1
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        n = len(A)

        dp = collections.defaultdict(lambda: 1)
        for i in range(n):
            for j in range(i+1, n):
                dp[(j, A[j]-A[i])] = dp[(i, A[j]-A[i])] + 1
        return max(dp.values())
        

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        dp = {}
        n = len(A)
        for i in range(n):
            for j in range(i+1, n):
                dp[(j, A[j] - A[i])] = dp.get((i, A[j] - A[i]), 1) + 1
        return max(dp.values())

from collections import OrderedDict

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        # 1. Compute pairwise differences d_ij: O(n^2)
        #   [0 -5 -2 -7 1]
        #   [0  0  3 -2 6]
        #   [0  0  0 -5 3]
        #   [0  0  0  0 8]
        # 2. For each target node j in [1, n), record (diff, i) pairs where i in [0, n-1)
        # 3. For each target node j in [1, n), LAS[j][diff] = LAS[i].get(diff, 0) + 1
        # 4. Output max(LAS[n-1])
        diffs = {}
        for src in range(len(A) - 1):
            for tgt in range(len(A) - 1, src, -1):
                diff = A[tgt] - A[src]
                # Only record the closest j to maximize LAS
                diffs[src, diff] = tgt
        
        inbound_edges = {}
        for edge, tgt in diffs.items():
            inbound_edges.setdefault(tgt, []).append(edge)

        max_length = 0
        memo = {0: {}}
        for tgt in range(1, len(A)):
            memo[tgt] = {}
            for src, diff in inbound_edges[tgt]:
                seq_length = memo[src].get(diff, 1) + 1
                if seq_length > memo[tgt].get(diff, 0):
                    memo[tgt][diff] = seq_length
                    max_length = max(seq_length, max_length)

        return max_length
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        dp = defaultdict(lambda : 1)
        for i in range(len(A)):
            for j in range(i+1,len(A)):
                dp[j,A[j]-A[i]] = dp[i,A[j]-A[i]]+1
        return max(dp.values())
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        dp=[{} for i in A]
        if len(A)<=2:
            return len(A)
        ans=0
        
        for i in range(1,len(A)):
            for j in range(i):
                diff=A[i]-A[j]
                if diff in dp[j]:
                    dp[i][diff]=max(2,dp[j][diff]+1)
                
                else:
                    dp[i][diff]=2
                
                ans=max(ans,dp[i][diff])
        return ans
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        if len(A) < 2:
            return 0
        la = len(A)
        dp = {}
        curr = 0
        for i in range(1, la):
            for j in range(i):
                d = A[i] - A[j]
                dp[(i, d)] = dp.get((j, d), 1) + 1
                # if dp[(i, d)] > curr:
                    # curr = dp[(i, d)]
        return max(dp.values())
from itertools import repeat

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        length = len(A)
        onleft = set()
        onleftl = []
        onright = list(repeat(0, 501))
        toextend = [{} for _ in range(501)]
        res = 2
        for v in A:
            onright[v] += 1
        for i in range(0, length):
            val = A[i]
            tex = toextend[val]
            onright[val] -= 1
            for lval in onleftl:
                diff = val - lval
                nextval = val + diff
                if nextval > 500 or nextval < 0 or onright[nextval] == 0:
                    continue
                if diff in tex:
                    c = tex[diff] + 1
                else:
                    c = 3
                if c > res:
                    res = c
                toextend[nextval][diff] = c
            if val not in onleft:
                onleftl.append(val)
                onleft.add(val)
        return res
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        if A is None or not A:
            return 0
        
        N = len(A)
        
        f = [{} for _ in range(N)]
        
        ret = 0
        
        for i in range(1, N):
            for j in range(i):
                diff = A[i] - A[j]
                if diff in f[j]:
                    f[i][diff] = f[j][diff] + 1
                else:
                    f[i][diff] = 2
                
                ret = max(ret, f[i][diff])
        
        return ret
class Solution:
    def longestArithSeqLength(self, A):
        n=len(A)
        dp={}
        for i in range(n):
            for j in range(i+1,n):
                dif = A[j]-A[i]
                if (i,dif) in dp :
                    dp[(j,dif)]=dp[(i,dif)]+1
                else:
                    dp[(j,dif)]=2
        return max(dp.values())
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        dp = [{}]
        max_length = 1
        for i in range(1, len(A)):
            dp.append({})
            for j in range(i):
                diff = A[i] - A[j]
                if diff in dp[j]:
                    dp[i][diff] = dp[j][diff] + 1
                else:
                    dp[i][diff] = 2
                max_length = max(max_length, dp[i][diff])

        return max_length
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        max_len = 2
        alen = len(A)
        
        len_index = [{} for index in range(alen)]
        len_index[1] = {A[1]-A[0]: 2}
        for idx in range(2, alen):
            val = A[idx]
            indices = len_index[idx]
            for preidx in range(idx):
                diff = val - A[preidx]
                if diff in len_index[preidx]:
                    new_len = len_index[preidx][diff] + 1
                else:
                    new_len = 2
                if diff in indices:
                    indices[diff] = max(indices[diff], new_len)
                else:
                    indices[diff] = new_len
                max_len = max(max_len, new_len)
        return max_len
                

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        dp = {}
        for i, a2 in enumerate(A[1:], start=1): #enumerate start=1 index u8d77u59cbu6578u503c
            for j, a1 in enumerate(A[:i]):
                d = a2 - a1
                if (j, d) in dp:
                    dp[i, d] = dp[j, d] + 1
                else:
                    dp[i, d] = 2 #u5169u500bu6578 u69cbu6210u4e00u500bd
        return max(dp.values())

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        dp = defaultdict(lambda: 1)
        for first_index in range(len(A) - 1):
            for second_index in range(first_index + 1, len(A)):
                dp[(second_index, A[second_index] - A[first_index]
                    )] = dp[(first_index, A[second_index] - A[first_index])] + 1
        max_length = max(dp.values())
        return max_length

class Solution:
    def longestArithSeqLength(self, A):
        n = len(A)
        dp = {}
        for i in range(1, n):
            for j in range(i):
                diff = A[i] - A[j]
                if (j, diff) in dp:
                    dp[(i, diff)] = dp[(j, diff)] + 1
                else:
                    dp[(i, diff)] = 2
        return max(dp.values())
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        
        #dp[i] = {delta:len} 
        
        n = len(A)
        if n < 2 : return n
        
        dp = [ {} for _ in range(n)]
        ans = 1
        dp[0] = {0:1}
        for i in range(1,n): 
            for j in range(i):
                delta =  A[i] - A[j]
                if delta in list(dp[j].keys()):
                    dp[i][delta] = dp[j][delta] + 1
                else:
                    dp[i][delta] = 2
                ans = max(ans,dp[i][delta])
        
        return ans
                    
                    
                

class Solution:
    def longestArithSeqLength(self, nums: List[int]) -> int:
        if not nums:
            return 0
        
        n = len(nums)
        # dp = [{}] * n # This is incorrect. It makes just one dictionary object, not n of them.
        dp = [{} for _ in range(n)]
        result = 2
        
        for i in range(1, n):
            for j in range(i):
                delta = nums[i] - nums[j]
                longestArithSeq = 2
                
                # If we're adding on to the longest arithmetic sequence seen thus far.
                if delta in dp[j]:
                    longestArithSeq = dp[j].get(delta) + 1
                    
                # Add it to the dictionary.
                dp[i][delta] = longestArithSeq
                
                # Update the result.
                result = max(result, longestArithSeq)
                
                if result == 3:
                    print('dim')
        
        return result
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        
        dp = defaultdict(dict)
        
        n = len(A)
        best = 0
        
        for i in range(1, n):
            for j in range(0, i):
                de = A[i] - A[j]
                dp[i][de] = dp[j].get(de, 1) + 1
                best = max(best, dp[i][de])
                
                
        return best
class Solution:
    def longestArithSeqLength(self, nums: List[int]) -> int:
        if not nums:
            return 0
        # Write your code here
        memo = [0] * len(nums)
        graph = collections.defaultdict(lambda: collections.defaultdict(int))
        
        res = 0
        for i in range(1, len(nums)):
            for j in range(i):
                diff = nums[i] - nums[j]
                prev_diffs = graph[j]
    
                prev_diff = prev_diffs[diff]
                graph[i][diff] = prev_diff + 1
                res = max(res, graph[i][diff])
        return res + 1
from collections import defaultdict

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        # lengths (i, d) longest arithmetic subsequence starting at i
        # with difference d
        lengths = defaultdict(lambda: 1)
        
        for i in range(len(A) - 2, -1, -1):
            for j in range(len(A) - 1, i, -1):
                lengths[i, A[j] - A[i]] = lengths[j, A[j] - A[i]] + 1
        return max(lengths.values())
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        from collections import defaultdict
        dp = []
        max_length = 0
        for i in range(len(A)):
            nd = defaultdict(int)
            nd[0] = 1
            dp.append(nd)
            for j in range(i):
                diff = A[i]-A[j]
                dp[i][diff] = max(dp[j][diff],1)+1
                if dp[i][diff] > max_length:
                    max_length = dp[i][diff]
        return max_length
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        if not A:
            return 0
        n = len(A)
        dp = [{} for i in range(n)]
        ans = 0
        for i in range(1, n):
            for j in range(i):
                diff = A[i] - A[j]
                tmp = dp[j].get(diff, 0)
                dp[i][diff] = tmp + 1
                ans = max(ans, tmp+1)
        return ans+1
                    
                    
        

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        if len(A) < 3:
            return len(A)
        res = 2
        d = {k:{} for k in set(A)}
        for j in range(len(A)):
            zero = d[A[j]].get(0, 0) + 1
            res = max(res, zero)
            for step, l in list(d[A[j]].items()):
                d.setdefault(A[j] + step, {})
                prev = d[A[j] + step].get(step, 0)
                d[A[j] + step][step] = max(prev, l + 1)
                res = max(res, l + 1)
            for i in range(j):
                d.setdefault(2 * A[j] - A[i], {})
                d[2 * A[j] - A[i]].setdefault(A[j] - A[i], 2)
                # print(i, j, d[2 * A[j] - A[i]])
            d[A[j]] = {0: zero}
            # print(d)
        # res = max([max(v.values()) for v in d.values()])
        return res

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        
        if len(A)==1:
            return 1
        if len(A)==2:
            return 2
        N=len(A)
        ans=2
        seen={}
        for i in range(N):
            for j in range(i+1,N):
                diff=A[j]-A[i]
                # now for the rest can we do this diff?
                
                if (i,diff) in seen:
                    ans=max(ans,seen[i,diff]+1)
                    seen[j,diff]=seen[i,diff]+1
                else:
                    seen[j,diff]=2
                
                
                # # prev=A[j]
                # # for k in range(j+1,N):
                # #     if A[k]-prev==diff:
                # #         curr+=1
                # #         prev=A[k]
                # ans=max(ans,curr)
                
            # seen.add(A[i])
        return ans
                    
        
#         if N<=2:
#             return N
        
#         dp=[0]*N
        
#         dp[2]=3 if A[2]-A[1]==A[1]-A[0] else 0
        
#         ans=dp[2]
        
#         for i in range(3,N):
#             if A[i]-A[i-1]==A[i-1]-A[i-2]:
#                 ans=max(ans,dp[i-1]+1,3)
#                 dp[i]=max(1+dp[i-1],3)
        
#         return ans
        
#         N=len(A)
#         def rec(diff,i,m):
#             if (i,diff) in m:
#                 return m[i,diff]
            
#             if i==N:
#                 return 0
            
#             m[i,diff]=0
            
#             ans=0
            
#             for k in range(i+1,N):
#                 if A[k]-A[i]==diff:
#                     ans=1+rec(diff,k+1,m)
#                     break
            
#             # make a new diff
#             # ans=max(ans,rec(diff,i+1,m))
            
#             for k in range(i+1,N):
#                 ans=max(ans,2+rec(A[k]-A[i],k+1,m))
            
#             m[i,diff]=ans
#             return ans
        
#         return rec(float('inf'),0,{})
            
#             if A[i]-diff in seen and seen[A[i]-diff]==i-:
#                 seen[A[i]]=i
#                 print('found',A[i],diff,i,seen)
#                 m[i,diff]=1+rec(diff,i+1,seen,m)
#             else:
#                 m[i,diff]=rec(diff,i+1,seen,m)
            
#             return m[i,diff]
                
            
            
        
        
#         N=len(A)
#         ans=0
#         m={}
#         for i in range(N):
#             for j in range(i+1,N):
#                 diff=A[j]-A[i]
#                 if diff!=0:
#                     seen={A[j]:1, A[i]:0}
#                     ans=max(ans,2+rec(diff,j+1,seen,m))
        
#         return ans

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        max_len = 0
        # index_of_array: {diff: curr_max_len}
        dp = [{} for _ in range(len(A))]
        
        for i in range(1, len(A)):
            for j in range(0, i):
                diff = A[i] - A[j]
                
                if diff in dp[j]:
                    dp[i][diff] = dp[j][diff] + 1
                else:
                    dp[i][diff] = 2
                
                max_len = max(max_len, dp[i][diff])
        
        return max_len

from typing import Dict, Tuple

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        n: int = len(A)
        dp: Dict[Tuple[int, int], int] = {}
        answer: int = 0
        for i in range(n):
            for j in range(i + 1, n):
                diff: int = A[j] - A[i]
                length: int = dp.get((i, diff), 1) + 1
                dp[(j, diff)] = length
                answer = max(answer, length)
        return answer
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        m = max(A) - min(A)+1
        n = len(A)
        dp = [[1 for _ in range(2*m)] for _ in range(n)]
        
        ans = 0
        for i in range(n):
            for j in range(i+1, n):
                d = A[j] - A[i]
                
                if d < 0:
                    d = m + abs(d)
                dp[j][d] = dp[i][d] + 1
                ans = max(ans, dp[j][d])
        
        return ans
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        # dp[i][d] = longestArithSeqLength(A[:i]) with difference d
        # dp[i][d] = max(1 + dp[j][A[i]-A[j]] for j=[0..i-1])
        # n^2
        dp = dict()
        max_len = 0
        # [3,6,9,12]
        # {1: {3: 2}, 2: {6: 2, 3: 3}, 3: {9: 2, 6: 2, 3: 4}}
        # d = 3
        for i in range(len(A)):
            # dp[i] = {diff: max_len}
            dp[i] = dict()
            for j in range(i):
                d = A[i]-A[j]
                if d in dp[j]:
                    dp[i][d] = max(dp[i][d], 1 + dp[j][d]) if d in dp[i] else 1 + dp[j][d]
                else:
                    dp[i][d] = 2
                max_len = max(max_len, dp[i][d])
    
        return max_len

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        n = len(A)
        dp = [{} for i in range(n)] #i: {step: length}
        for i in range(1, n):
            for j in range(i):
                step = A[i] - A[j]
                dp[i][step] = max(
                    dp[i].get(step, 0), dp[j].get(step, 1) + 1)
                
        return max(max(dp[i].values(), default=0)
                   for i in range(n))
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        if not A:
            return 0
        
        dp = [{0: 1}]
        
        max_val = 1
        
        for i in range(1, len(A)):
            dp.append({0: 1})
            
            i_val = A[i]
            last_map = dp[-1]
            
            for j in range(i):
                j_val = A[j]
                diff = i_val - j_val
                
                diff_val = 2
                
                if diff in dp[j]:
                    diff_val = dp[j][diff] + 1
                
                if diff not in last_map:
                    last_map[diff] = diff_val
                else:
                    last_map[diff] = max(last_map[diff], diff_val)
                
                max_val = max(max_val, diff_val)
                    
        return max_val
            
                
                
        
        # every sequence is trivially 1 for itself.
        
        
        # start at index 1
        
            # loop up to this index
                # get the difference of i and j
                    
                    # have we seen that difference before? if so, add a 1 to that amount.
                    # if we have, take the max of the two times we've seen differences
        
        
        # return the max difference we've seen so far.

class Solution:
    def longestArithSeqLength(self, arr: List[int]) -> int:
        n = len(arr)
        if n <= 2:
            return n
        dp = defaultdict(dict)
        max_len = 2
        for i in range(n):
            for j in range(i + 1, n):
                diff = arr[j] - arr[i]
                if diff in dp[i]:
                    dp[j][diff] = dp[i][diff] + 1
                else:
                    dp[j][diff] = 2
                max_len = max(max_len, dp[j][diff])
        return max_len
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        n = len(A)
        
        m = {} # expect: {step: length}
        for i in range(1, n):
            e = m.pop(A[i], {})
            for step, length in e.items():
                e1 = m.setdefault(A[i] + step, {})
                e1[step] = max(e1.get(step, 0), length + 1)
            for j in range(i):
                step = A[i] - A[j]
                e1 = m.setdefault(A[i] + step, {})
                e1[step] = max(e1.get(step, 0), 2)
                
        return max(max(e.values()) for e in m.values())
class Solution:
    def longestArithSeqLength(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 2:
            return n
        dp = [{} for i in range(n)]
        max_len = 0
        for i in range(1, n):
            for j in range(i):
                diff = nums[i] - nums[j]
                if diff in dp[j]:
                    dp[i][diff] = max(2, 1 + dp[j][diff])
                else:
                    dp[i][diff] = 2
                max_len = max(max_len, dp[i][diff])
        return max_len
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        cur = collections.defaultdict(dict)
        for i, v in enumerate(A):
            for j in range(i):
                val = v - A[j]
                cur[i][val] = 1 + cur[j].get(val, 1)
        return max(cur[i][j] for i in cur for j in cur[i])
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:

        d=[{}]
        ans=0
        for i in range(1,len(A)):
            d.append({})
            for j in range(i):
                if (A[j]-A[i]) in d[j]:
                    d[i][A[j]-A[i]]=d[j][A[j]-A[i]]+1
                else:
                    d[i][A[j]-A[i]]=1
                ans=max(ans,d[i][A[j]-A[i]])
        return ans+1
            
                
                
                
                

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        N = len(A)
        if N <= 2:
            return N
        counters = [None] * N
        ret_val = 2
        for idx in range(N):
            if idx == 0:
                counters[idx] = Counter()
                continue
            counter = Counter()
            for prev_idx in range(idx):
                prev_counter = counters[prev_idx]
                delta = A[idx] - A[prev_idx]
                counter[delta] = max(
                    counter[delta],
                    max(prev_counter[delta] + 1, 2)
                )
                ret_val = max(ret_val, counter[delta])
                counters[idx] = counter
            # return
        return ret_val

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        if not A:
            return 0
        h = {}
        h[0] = defaultdict(int)
        res = 1
        for i in range(1, len(A)):
            h[i] = defaultdict(int)
            for j in range(i):
                diff = A[i] - A[j]
                h[i][diff] = h[j][diff] + 1
                res = max(res, h[i][diff])
        return res + 1
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        if not A:
            return 0
        h, res = {}, 1
        for i in range(len(A)):
            h[i] = defaultdict(int)
            for j in range(i):
                diff = A[i] - A[j]
                h[i][diff] = h[j][diff] + 1
                res = max(res, h[i][diff])
        return res + 1
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        N = len(A)
        dp = [defaultdict(int) for _ in range(N)]
        for j in range(N):
            for i in range(j):
                t = A[j] - A[i]
                dp[j][t] = max(dp[j][t], dp[i][t] + 1)
        return max(max(col.values()) for col in dp) + 1
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        dp = {}
        m = 2
        for i in range(1, len(A)):
            for j in range(0, i):
                if (j, A[i] - A[j]) in dp:
                    dp[(i, A[i] - A[j])] = dp[(j, A[i] - A[j])] + 1
                    m = max(m, dp[(i, A[i] - A[j])])
                else:
                    dp[(i, A[i] - A[j])] = 2
        return m
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        dp = {}
        n = len(A)
        
        res = 0
        for r in range(1, n):
            for l in range(r):
                diff = A[r] - A[l]
                
                if (diff, l) in dp:
                    dp[(diff, r)] = dp[(diff, l)] + 1
                    res = max(res, dp[(diff, l)] + 1)
                else:
                    dp[(diff,r)] = 2
                    res = max(res, 2)
        return res
                

from collections import defaultdict

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        mx = 0
        dp = [defaultdict(int) for _ in range(len(A))]
        for i,a in enumerate(A):
            for j,b in enumerate(A[:i]):
                if a - b in dp[j].keys():
                    dp[i][a-b] = dp[j][a-b] + 1
                else:
                    dp[i][a-b] = 2
                mx = max(mx, dp[i][a-b])
        return mx
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        
        dp = [{} for _ in range(len(A))]
        res = 0
        for i in range(1, len(A)):
            for j in range(i):
                diff = A[i] - A[j]
                if diff not in dp[i]:
                    dp[i][diff] = 2
                if diff in dp[j]:
                    dp[i][diff] = max(dp[i][diff], dp[j][diff] + 1)
                res = max(res, dp[i][diff])
        return res
                

                    
        

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        dp = {}
        N = len(A)
        for i in range(N):
            for j in range(i+1, N):
                diff = A[j]-A[i]
                if (i, diff) in dp:
                    dp[(j, diff)] = dp[(i, diff)]+1
                else:
                    dp[(j, diff)] = 2
                
        return max(dp.values())
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        # dp[i][d]: the length of longest arithmetic d subsequence in A[:i]
        
        dp = [collections.defaultdict(lambda: 1) for _ in range(len(A) + 1)]
        
        for i in range(1, len(A) + 1):
            for j in range(1, i):
                d = A[i-1] - A[j-1]
                dp[i][d] = max(dp[i][d], dp[j][d] + 1)
        
        return max(max(list(item.values()) or [1]) for item in dp)

from collections import defaultdict
class Solution:
    
    def longestArithSeqLength(self, A: List[int]) -> int:
        def gen():
            x = [0 for i in range(l)]
            x[0] = 1
            return x
        
        # dp[diff][index]
        l = len(A)
        dp = defaultdict(lambda: [1 for i in range(l)])
        ans = 0
        for i in range(l):
            for j in range(i+1, l):
                dp[A[j]-A[i]][j] = dp[A[j]-A[i]][i] + 1
                ans = max(ans,dp[A[j]-A[i]][j] )

            
        return ans
        

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        memo = collections.defaultdict(dict)
        ans = 0
        
        for i, a2 in enumerate(A):
            for j in range(i):
                a1 = A[j]
                diff = a2 - a1
                memo[i][diff] = 2
                if diff in memo[j]:
                    memo[i][diff] = max(memo[i][diff], memo[j][diff] + 1)
                ans = max(ans, memo[i][diff])
        return ans
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        seen = set()
        dp = [defaultdict(int) for _ in range(len(A))]
        mx = 0
        
        for i in range(len(A)):
            seen.add(A[i])
            for j in range(0, i):
                ap = A[i] + -A[j]
            
                if A[i] + -ap in seen:
                    dp[i][ap] = dp[j][ap] + 1
                    mx = max(mx, dp[i][ap])
                    
        return mx + 1

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        N = len(A) 
        dp = [[2] * N for i in range(N)]
        ans = 0
        for i in range(N):
            pos = {}
            for j in range(i):
                x = 2*A[j] - A[i]
                if x in pos:
                    dp[i][j] = max(dp[i][j], 1 + dp[j][pos[x]])
                ans = max(ans, dp[i][j])
                pos[A[j]] = j
        
        return ans
    
        dp = [dict() for _ in range(N)]
        ret = 1
        for i in range(N):
            for j in range(i + 1, N):
                diff = A[j] - A[i]
                dp[j][diff] = dp[i].get(diff, 1) + 1
                ret = max(ret, dp[j][diff])
        return ret
        # def calc(A):
        #     for i in range(len(A) - 1, -1, -1):
        #         for j in range(i + 1, len(A)):
        #             if A[j] < A[i]:
        #                 continue
        #             diff = A[j] - A[i]
        #             memo[i][diff] = max(memo[i].get(diff, 0), memo[j].get(diff, 1) + 1)
        #             ret = max(ret, memo[i][diff])
        #     return ret
        # 
        # return max(
        #     calc(A), calc(list(reversed(A)))
        # )

from collections import defaultdict
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        dp = defaultdict(dict) # { index: {difference: steps} }
        max_cnt = 0
        for i in range(1, len(A)):
            for j in range(0, i):
                difference = A[i] - A[j]
                length = dp[j].get(difference, 1) + 1
                dp[i][difference] = dp[j].get(difference, 1) + 1
                max_cnt = max(max_cnt, length)
        
        return max_cnt
                

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        ans=0
        dd=[[1]*1001 for i in range(len(A))]
        for i in range(1,len(A)):
            for j in range(i):
                diff=A[i]-A[j]+500
                dd[i][diff]=max(dd[i][diff],dd[j][diff]+1)
                ans=max(ans,dd[i][diff])
        return ans
class Solution:
    def longestArithSeqLength(self, A):
        dp = {}
        #dictionary of pairs (i pointer, dist)
        
        # i pointer iterates from 1index to the end
        for i in range(1, len(A)):
            # j pointer iterates from 0 to just left of i pointer then resets
            for j in range(0, len(A[:i])):
                
                #finds the difference of the two values
                d = A[i] - A[j]
                
                #checks to see if the same diff exists at j
                if (j, d) in dp:
                    #if j,d is in dp then add 1 because the value at i has the same difference and set that as i,d
                    dp[i, d] = dp[j, d] + 1
                #if not then its set to two because that accounts for the i,j as two integers
                else:
                    dp[i, d] = 2
                    
        #return what ever is the highest value of all the keys in the dictionary is
        return max(dp.values())
class Solution:
                    
    def longestArithSeqLength(self, A: List[int]) -> int:
        memo = dict()
        res = 0
        for j in range(len(A)):
            for i in range(j):
                diff = A[j]-A[i]
                if (diff, i) in memo:
                    memo[(diff, j)] = memo[(diff, i)]+1
                else:
                    memo[(diff, j)] = 2
        max_val = max(memo.values())
        return max_val
                    
                    

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        length = len(A)

        # Stores [difference, index] -> length
        # for each position j:
        #   the arithmetic sequence 
        hash = {}

        for i in range(length):
            for j in range(i + 1, length):
                hash[A[j] - A[i], j] = hash.get((A[j] - A[i], i), 1) + 1

        # O(n)
        return max(hash.values())
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        memo = {}
        for j, y in enumerate(A):
            for i, x in enumerate(A[:j]):
                d = y - x
                memo[d, j] = memo.get((d, i), 1) + 1
        return max(memo.values())
#[] DP
#O(N^2)
#NOTE: diff can be either positive or negative
#f[i]: the longest length of arithmetic subsequences who takes A[i] as the tail.
#f[i] = defaultdict(lambda: 1)
#f[i] = {diff: longest length}
#f[i] = max(f[i][d], f[j][d] += 1) for j < i and d:=A[i]-A[j]
from collections import defaultdict
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        dp = [defaultdict(lambda: 1) for _ in range(len(A))]
        for i, a in enumerate(A):
            for j in range(i):
                diff = A[i] - A[j]
                dp[i][diff] = max(dp[i][diff], dp[j][diff] + 1)
        return max(max(lens.values()) for lens in dp)
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        if len(A) < 2: return len(A)
        dp = {}
        
        for i in range(len(A)):
            for j in range(i+1, len(A)):
                dp[j, A[j]-A[i]] = dp.get((i, A[j]-A[i]), 1) + 1
        return max(dp.values())

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        n = len(A)
        dp = []
        maxv = 0
        for i in range(n):
            cur = A[i]
            dp.append({})
            for j in range(i):
                pre = A[j]
                dif = cur-pre
                if dif not in dp[i]:
                    dp[i][dif] = 0
                if dif in dp[j]:
                    dp[i][dif] = dp[j][dif] + 1
                else:
                    dp[i][dif] = 1
                maxv = max(maxv, dp[i][dif])
        return maxv+1
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        dp = {}
        n = len(A)
        for i in range(n):
            for j in range(i+1, n):
                dp[(j, A[j]-A[i])] = dp.get((i, A[j]-A[i]), 1)+1
        return max(dp.values())
from collections import defaultdict
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        n = len(A)
        dp = [defaultdict(lambda: 1) for _ in range(n)]
        for i in range(n):
            for j in range(i):
                dp[i][A[i]-A[j]]=max(dp[j][A[i]-A[j]]+1,2)
        return max([val for _ in dp for __,val in _.items()])
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        dp = {}
        for i,a1 in enumerate(A):
            for j,a2 in enumerate(A[:i]):
                diff = a1-a2
                if (j,diff) in dp:
                    dp[i,diff] = 1+dp[j,diff]
                else:
                    dp[i,diff]=2
        return max(dp.values())
                    

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        dp = []
        for i in range(len(A)):
            dp.append(collections.defaultdict(lambda: 1))
            for j in range(i):
                # calculate the differences
                diff = A[i] - A[j]
                # check the existing sequence length
                dp[i][diff] = max(dp[i][diff], dp[j][diff]+1)
        return max([max(d.values()) for d in dp])

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:

    
        f = {}
        maxlen = 0
        for i in range(len(A)):
            for j in range(i):
                diff = A[i] - A[j]
                #fff[(A[i], diff)] = max(fff[(A[i], diff)], fff.get((A[j], diff), 1) + 1)
                #f[(i, diff)] = max(f[(i, diff)], f.get((j, diff), 1) + 1)
                #f[(i, diff)] = f.get((j, diff), 1) + 1
                
                if (j, diff) not in f:
                    f[i, diff] = 2
                else:
                    f[i, diff] = f[j, diff] + 1          
                                    
                maxlen = max(maxlen, f[(i, diff)])

        return maxlen

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        n=len(A)
        dp={}
        ans=0
        for i in range(n):
            for j in range(i):
                diff=A[i]-A[j]
                if (diff,j) not in dp:
                    
                    dp[(diff,i)]=2
                else:
                    
                    dp[(diff,i)]=dp[(diff,j)]+1
                ans=max(ans,dp[(diff,i)])       
        return ans
                    
                    
        
        
        

class Solution:
    def longestArithSeqLength(self, A):
        dp = {}
        for i in range(len(A)):
            for j in range(i + 1, len(A)):
                dif = A[j]-A[i]
                if (i,dif) in dp :
                    dp[(j,dif)]=dp[(i,dif)]+1
                else:
                    dp[(j,dif)]=2
        return max(dp.values())
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        res = 0
        dic = {}
        for i in range(len(A)):
            for j in range(i+1,len(A)):
                d = A[j]-A[i]
                if (i,d) in dic:
                    dic[(j,d)]=dic[(i,d)]+1
                else:
                    dic[(j,d)]=2
                res = max(res,dic[(j,d)])
        return res

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        '''
        
        '''
        dp = {}
        ans = 0
        for i in range(len(A)):
            for j in range(i):
                diff = A[i] - A[j]
                if (diff, j) not in dp:
                    dp[(diff, i)] = 2
                else:
                    dp[(diff, i)] = dp[(diff, j)] + 1
                ans = max(ans, dp[(diff, i)])
        return ans

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        if not A:
            return 0
        
        dp = [{} for _ in range(len(A))]
                
        max_seq = 1
        
        for i in range(1, len(A)):
            dp[i] = {0:1}
            
            for j in range(i):
                diff = A[i] - A[j]
                
                if diff not in dp[j]:
                    dp[i][diff] = 2
                else:
                    dp[i][diff] = dp[j][diff] + 1
                
                max_seq = max(max_seq, dp[i][diff])
                        
        return max_seq
            
                
                
            
        

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        
        dp = {}
        ans = 0
        for i, n in enumerate(A):
            for j in range(i):
                diff = A[j] - A[i]
                if (diff, j) not in dp:
                    dp[(diff, i)] = 2
                else:
                    dp[(diff, i)] = dp[(diff, j)] +1
                ans = max(ans, dp[(diff, i)])
        return ans

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        dp = []
        
        for i, x in enumerate(A):
            nd = collections.defaultdict(int)
            dp.append(nd)
            for j in range(i):
                curr_diff = x - A[j]
                dp[i][curr_diff] = dp[j][curr_diff] + 1
          
        return max(max(y.values()) for y in dp) + 1
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        dp = defaultdict(int)
        for i in range(len(A)):
            for j in range(i + 1, len(A)):
                span = A[i] - A[j]
                dp[j, span] = dp.get((i, span), 1) + 1
        return max(dp.values())
from collections import defaultdict
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        d = {}
        mval = -1
        for i, n in enumerate(A):
            for j in range(i):
                if (j, n-A[j]) not in d:
                    val = 2
                else:
                    val = d[(j, n-A[j])] + 1
                mval = max(mval, val)
                d[(i, n-A[j])] = val
        return mval

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        if len(A) <= 2:
            return len(A)
        dict = {}
        maxLength = 2
        for i in range(len(A) - 1):
            sameNumEncountered = False
            for j in range(i + 1, len(A)):
                dif = A[j] - A[i]
                if dif not in dict:
                    dict[dif] = {}
                
                if A[i] not in dict[dif]:
                    dict[dif][A[j]] = 2
                elif dif != 0 or not sameNumEncountered:
                    dict[dif][A[j]] = dict[dif][A[i]] + 1
                
                if dif == 0:
                    sameNumEncountered = True
                
                maxLength = max(maxLength, dict[dif][A[j]])
        return maxLength

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        res = 0
        
        cache = dict()
        
        for i in range(len(A)):
            for j in range(i + 1, len(A)):
                diff = A[j] - A[i]
                
                if (i, diff) not in cache:
                    cache[(j, diff)] = 2
                else:
                    cache[(j, diff)] = 1 + cache[(i, diff)]
                
                
        return max(cache.values())
'''
dp[i][d] = longest subsequence ending at i with difference d

dp[j][d] = 1 + max(
    dp[j][A[j] - A[i]]
) for j < i


'''
class Solution:
    def longestArithSeqLength(self, A):
        dp = collections.defaultdict(int)
        mx = 0
        for i in range(len(A)):
            for j in range(i):
                diff = A[i] - A[j]
                j_key = (j, diff)
                i_key = (i, diff)
                if j_key in dp:
                    dp[i_key] = dp[j_key] + 1
                else:
                    dp[i_key] = 2
                mx = max(mx, dp[i_key])
        return mx#max(dp.values())

'''

brute force - 2 for loops, find diff of i and i+1, then find next elem j where j-(i+i) is the same
keep track of max len found
  t=O(n^3)
  s=O(1)



[20,1,15,3,10,5,8]
                {}
              {3:2}
           {-5:2,-2:2}
         {7:2,2:2,5:2}
      {-12:2,-5:3,-10:2,-7:2}
    
                

  |       |
0,1,0,1,0,1
          {}
        {1:2}
      {-1:2, 0:2}
    {1:2,0:2}
  {-1:2,0:3,}
max = 4
-19


'''

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        maps = [{} for i in range(len(A))]
        maxlen = 0
        for i in range(len(A)-1, -1, -1):
            for j in range(i+1, len(A)):
                diff = A[j] - A[i]
                if diff in maps[j]:
                    length = maps[j][diff]+1
                else:
                    length = 2
                maxlen = max(maxlen, length)
                if diff in maps[i]:
                    prev_max = maps[i][diff]
                    maps[i][diff] = max(prev_max, length)
                else:
                    maps[i][diff] = length
        return maxlen
                
def search(vals, start, target, length):
    for i in range(start+1, len(vals)):
        if vals[i] - vals[start] == target:
            return search(vals, i, target, length+1)
    return length
from collections import OrderedDict

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        # When adding a new number A[j], we look at all previous numbers A[i]:
        # (1) If A[j] can extend any arithmetic subsequence currently ends at A[i]: LAS += 1
        # (2) Otherwise, LAS = 2
        max_length = 0
        subseq_lengths = {}
        for j in range(1, len(A)):
            for i in range(j):
                diff = A[j] - A[i]
                if (diff, i) in subseq_lengths:
                    subseq_lengths[diff, j] = subseq_lengths[diff, i] + 1
                else:
                    subseq_lengths[diff, j] = 2
                max_length = max(max_length, subseq_lengths[diff, j])
        return max_length
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        from collections import defaultdict
        dp = [Counter() for _ in range(len(A))]
        
        
        max_length = 0
        for i in range(len(A)):
            for j in range(i):
                diff = A[i] - A[j]
                # Was this difference earlier seen at index j? Then continue that chain
                if diff in dp[j]:
                    dp[i][diff] = dp[j][diff] + 1
                else: # If no, create a new chain of length 2
                    dp[i][diff] = 2
                    
                max_length = max(max_length, dp[i][diff])
        # Print this table for any input for better understanding of the approach
        # print table

        return max_length
        
    

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        dp = {}
        for i in range(1,len(A)):
            for j in range(i):
                delta = A[i] - A[j]
                if (j, delta) in dp:
                    dp[(i, delta)] = dp[(j, delta)] + 1
                else:
                    dp[(i,delta)] = 2
        return max(dp.values())


class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        if A==[]: return 0
        if len(A)==1: return 1
        dp={}
        for i in range(0,len(A)):
            for j in range(0,i):
                dp[i,A[i]-A[j]]=dp.get((j,A[i]-A[j]),0)+1
        
        return max(dp.values())+1

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        dp = {}
        ans = 0
        for i in range(len(A)):
            for j in range(i+1, len(A)):
                diff = A[i] - A[j]
                if (i, diff) in dp:
                    dp[(j, diff)] = dp[(i, diff)] + 1
                else:
                    dp[(j, diff)] = 2
                ans = max(ans, dp[(j, diff)])
        return ans

class Solution:
 

    def longestArithSeqLength(self, A: List[int]) -> int:
        n = len(A)
        dp = [{} for i in range(n)]
        result = 2
        
        for i in range(1, n):
            for j in range(i):
                delta = A[i] - A[j]
                
                # If we've seen this delta with dp[j], then increase the length of the subseq by 1.
                # This is equivalent of dp[i] 'adding on' to the subsequence.
                if delta in dp[j]:
                    currentLength = dp[j].get(delta)
                    dp[i][delta] = currentLength + 1
                
                else:
                    dp[i][delta] = 2
                
                result = max(result, dp[i][delta])        
        return result
    
    def longestArithSeqLength2(self, A: List[int]) -> int:
        from collections import Counter
        cnt = Counter()
        cnt
        arith = [Counter() for i in range(len(A))]
        longest = 0
        for i in range(len(A)-2, -1,-1):
            for j in range(i+1, len(A)):
                diff = A[j]-A[i]
                arith[i][diff] = max(1 + arith[j][diff], arith[i][diff])
                longest = max(longest, arith[i][diff])
        #print(arith)
        
        #         for i in range(len(A)):
        #             #print(arith[i])
        #             most_common = arith[i].most_common()

        #             longest = max(most_common[0][1] if most_common else 0, longest)
        return longest + 1
        # for i in range(len(A)):
        #     for j in range(i+1, len(A)):
        #         cnt[A[j]-A[i]] += 1
        #     print(A[i], cnt)
        # print(cnt)
        # val = cnt.most_common()[0][1]
        # return val + 1 
            
        
        #         self.arith = [dict() for i in range(len(A))]

        #         def helper(i, diff):
        #             if diff in self.arith[i]:
        #                 return self.arith[i][diff]

        #             val = 0
        #             for j in range(i+1, len(A)):
        #                 if A[j] - A[i] == diff:
        #                     val = 1 + helper(j, diff)
        #                     break
        #             self.arith[i][diff] = val        
        #             return self.arith[i][diff]
        

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        dp = {}
        ret = 0
        for i in range(len(A)):
            for j in range(i):
                diff = A[i] - A[j]
                dp[i, diff] = dp.get((j, diff), 1) + 1
                ret = max(ret, dp[i, diff])
        return ret
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        n = len(A)
        dp = [[0]*501 for i in range(n)]
        max_val = 0
        for i in range(n):
            for j in range(i):
                dif = A[i] - A[j]
                dp[i][dif] = max(dp[i][dif], dp[j][dif] + 1)
                max_val = max(dp[i][dif], max_val)
        #print(dp)
        return max_val + 1
    
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        n = len(A)
        dp = collections.defaultdict(dict)
        max_val = 0
        for i in range(n):
            for j in range(i):
                dif = A[i] - A[j]
                dp[dif].setdefault(i, 0)
                dp[dif][i] = dp[dif].get(j,0) + 1
                max_val = max(dp[dif][i], max_val)
        #print(dp)
        return max_val + 1    

class Solution1:
    def longestArithSeqLength(self, A: List[int]) -> int:
        n = len(A)
        dp = [collections.defaultdict(int) for _ in range(n)] 
        res = 0
        for i in range(1, n):
            for j in range(0, i):
                diff = A[i]-A[j]
                dp[i][diff] = dp[j][diff] + 1
                    
                res = max(res, dp[i][diff])
        return res + 1
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        # dp[i][k] = longestArithSeqLength(A[:i+1]) with step size k
        largest = max(A)
        smallest = min(A)
    
        dp = dict()
        res = 0
    
        for i in range(1, len(A)):
            for prev_i in range(i):
                step = A[i] - A[prev_i]
                prev_step = dp[(prev_i, step)] if (prev_i, step) in dp else 1
                dp[(i, step)] = prev_step + 1
                
                res = max(res, dp[(i, step)])
        
        return res
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:

    
        f = collections.defaultdict(int)
        maxlen = 0
        for i in range(len(A)):
            for j in range(i):
                diff = A[i] - A[j]
                #fff[(A[i], diff)] = max(fff[(A[i], diff)], fff.get((A[j], diff), 1) + 1)
                #f[(i, diff)] = max(f[(i, diff)], f.get((j, diff), 1) + 1)
                f[(i, diff)] = f.get((j, diff), 1) + 1
                '''
                if (j, diff) not in f:
                    f[(i, diff)] = 2
                else:
                    f[(i, diff)] = max(f[(i, diff)],  f[(j, diff)] + 1)                
                '''                    
                maxlen = max(maxlen, f[(i, diff)])

        return maxlen

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        dp = list(dict() for i in range(len(A)))
        maxsize = 0
        for i in range(1,len(A)):
            for j in range(0,i):
                if(A[j] - A[i] in dp[j]):
                    dp[i][A[j] - A[i]] = dp[j][A[j] - A[i]] + 1
                else:
                    dp[i][A[j] - A[i]] = 1
            
                maxsize = max(maxsize, dp[i][A[j] - A[i]])
        
        return maxsize + 1

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        dp = {}
        ret = 0
        for i in range(len(A)):
            for j in range(i+1, len(A)):
                diff = A[i] - A[j]
                dp[j, diff] = dp.get((i, diff), 1) + 1
                ret = max(ret, dp[j, diff])
        return ret
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        maxd = max(A) - min(A)
        dp = [[1 for j in range(2*maxd+1)] for i in range(len(A))]
        maxv = 1
        for i in range(1, len(dp)):
            for j in range(i-1, -1, -1):
                diff = A[i]- A[j]
                dp[i][diff] = max(dp[i][diff], dp[j][diff] + 1)
                maxv = max(maxv, dp[i][diff])
        return maxv

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        dp = dict()
        
        for i in range(len(A)):
            for j in range(i+1, len(A)):
                dp[j, A[j]-A[i]] = dp.get((i, A[j]-A[i]), 1) + 1   # dp ending at j with common diff A[j]-A[i]
                
        return max(dp.values())
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        dp = {}
        for i in range(len(A)):
            for j in range(i+1, len(A)):
                dp[j, A[j] - A[i]] = dp.get((i, A[j]-A[i]), 1) + 1
        return max(dp.values())
        

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        dp = {}
        for i in range(len(A)):
            for j in range(i+1, len(A)):
                diff = A[j] - A[i]
                dp[j, diff] = dp.get((i, diff), 1) + 1
        return max(dp.values())
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:

    
        f = {}
        maxlen = 0
        for i in range(len(A)):
            for j in range(i):
                diff = A[i] - A[j]
                #fff[(A[i], diff)] = max(fff[(A[i], diff)], fff.get((A[j], diff), 1) + 1)
                #f[(i, diff)] = max(f[(i, diff)], f.get((j, diff), 1) + 1)
                f[(i, diff)] = f.get((j, diff), 1) + 1
                '''
                if (j, diff) not in f:
                    f[(i, diff)] = 2
                else:
                    f[(i, diff)] = max(f[(i, diff)],  f[(j, diff)] + 1)                
                '''                    
                maxlen = max(maxlen, f[(i, diff)])

        return maxlen

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        dir = {}
        res = 0
        for i in range(len(A)):
            
            for j in range(0, i):
                diff = A[i] - A[j]
                if (j, diff) not in dir:
                    dir[(i, diff)] = 2
                else:
                    dir[(i, diff)] = dir[(j, diff)] + 1
                    
                res = max(res, dir[(i, diff)])
        return res
                    

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        if len(A) <= 2:
            return len(A)
        
        dict = {}
        maxLength = 2
        for i in range(len(A) - 1):
            sameNumEncountered = False
            for j in range(i + 1, len(A)):
                dif = A[j] - A[i]
                
                if (i, dif) not in dict:
                    dict[j, dif] = 2
                else:
                    dict[j, dif] = dict[i, dif] + 1
                maxLength = max(maxLength, dict[j, dif])
                
        return maxLength

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        mapping = {}
        n = len(A)
        if n < 2:
            return n
        max_ = 0
        for i in range(n):
            mapping[i] = {}
            for j in range(i):
                diff = A[i]-A[j]
                if diff not in mapping[j]:
                    mapping[i][diff] = 2
                else:
                    mapping[i][diff] = mapping[j][diff] + 1
                max_ = max(max_, mapping[i][diff])
        return max_
            

#SELF TRY 9/20/2020
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        #Problem asks -> array A 
        #Want -> return the length of longest arithmetic subseq in A 
        
        #Two cases basically.
        #When building up to to your table 
        #When looking at previous values i.e from (col, change) we see that there is a PREVIOUS value already there. (With the same change) so you know you can extend that length by 1
        #This "1" indicates the CUR number is being added because it's change is the same. 
        #If it's never been seen that the base case is at most length 2 
        #i.e some "subseq" whatever it is with 2 elements would always be "valid" and so would be length 2 
        table = dict()
        longest_subseq = float('-inf')
        
        for row in range(len(A)): 
            for col in range(0, row): 
                change = A[row] - A[col]
                if (col, change) in table: 
                    table[(row, change)] = 1 + table[(col, change)]
                    
                
                
                else: 
                    table[(row, change)] = 2 
                    
                longest_subseq = max(longest_subseq, table[(row, change)])
                
                
                
        return longest_subseq
    
    
# class Solution:
#     def longestArithSeqLength(self, A):
#         d = {}
#         for i in range(1, len(A)):
#             for j in range(len(A)):
#                 delta = A[i] - A[j]
#                 d[(i, delta)] = d[(j, delta)] + 1 if (j, delta) in d else 2
#         return max(d.values())
# class Solution:
#     def longestArithSeqLength(self, A: List[int]) -> int:
#         #Array A 
#         #Want return the (Len of longest arithmetic subseq in A)
#         #Recall that a subseq of A is a list A[i_1], A[i_2]... A[i_k] s.t
#         # 0 <= i_1 < i_2 < i_k <= A.len() - 1 
    
#         #ROW
#         # [9,4,7,2,10]
        
#         #COL
#         #[
#         #9
#         #4
#         #7 
#         #2
#         #10
#         #]
#         table = dict()
#         for row in range(1, len(A)):
#             for col in range(0, row): 
#                 change = A[row] - A[col]
                
#                 if (col, change) in table: 
#                     table[(row, change)] = table[(col, change)] + 1 
#                 else: 
#                     #table(1, -5) = 2  
#                     #table(1, 0) = 2 
#                     #table(1, -3) = 2 + 1 
#                     #table(1, -2) = 2
#                     #table(1, -6) = 2
#                     #table(0, -2) = 2
#                     #(2, 0) = 2 
#                     #(3,5) = 2 
#                     #(2, -3) = 2 
                    
#                     table[(row, change)] = 2 
            
#         return max(table.values())

#SELF TRY 9/20/2020
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        #Problem asks -> array A 
        #Want -> return the length of longest arithmetic subseq in A 
        
        #Two cases basically.
        #When building up to to your table 
        #When looking at previous values i.e from (col, change) we see that there is a PREVIOUS value already there. (With the same change) so you know you can extend that length by 1
        #This "1" indicates the CUR number is being added because it's change is the same. 
        #If it's never been seen that the base case is at most length 2 
        #i.e some "subseq" whatever it is with 2 elements would always be "valid" and so would be length 2 
        table = dict()
        longest_subseq = float('-inf')
        
        for row in range(len(A)): 
            for col in range(0, row): 
                change = A[row] - A[col]
                if (col, change) in table: 
                    table[(row, change)] = 1 + table[(col, change)]
                    
                
                
                else: 
                    table[(row, change)] = 2 
                longest_subseq = max(longest_subseq, table[(row, change)])
        return longest_subseq

from collections import defaultdict
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        d = defaultdict(dict)
        ans = 0
        for i in range(len(A)):
            for j in range(i+1, len(A)):
                diff = A[j]-A[i]
                if(diff in d[i]):
                    d[j][diff] = d[i][diff]+1
                else:
                    d[j][diff] = 1
                ans = max(ans, d[j][diff])
        return ans+1
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        # arr = {i:[1 for i in range(len(A))] for i in range(min(A)-max(A), max(A)-min(A)+1)}  # No need to initialize initially, will save lots of space
        arr = {}
        ans = 0
        for i in range(1, len(A)):
            for j in range(i):
                dif = A[i]-A[j]
                if not arr.get(dif):
                    arr[dif] = [1 for i in range(len(A))]
                arr[dif][i] = arr[dif][j] + 1
                ans = max(ans, arr[dif][i])
        
        # for x in arr:
        #     print(x, arr[x])
        
        return ans




# TLE
#         arr = {i:[1 for i in range(len(A))] for i in range(min(A)-max(A), max(A)-min(A)+1)}
#         ans = 0
#         for dif in range(min(A)-max(A), max(A)-min(A)+1):
#             for j in range(1, len(arr[0])):
#                 for k in range(j-1, -1, -1):
#                     if A[j]-A[k] == dif:
#                         arr[dif][j] = arr[dif][k] + 1
#                         ans = max(ans, arr[dif][j])
#                         break

#         # for x in arr:
#         #     print(x, arr[x])
#         return ans
        
                        

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        dp = []
        maximum = 1
        for i in range(len(A)):
            d = {}
            dp.append({})
            for j in range(i):
                diff = A[i]-A[j]
                if(diff not in dp[i]):
                    dp[i][diff] = 0
                if(diff not in dp[j]):
                    dp[j][diff] = 0
                dp[i][diff] = dp[j][diff]+1
                maximum = max(maximum,dp[i][diff])
        
        return maximum+1
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        res = 0
        dic = {}
        for i in range(len(A)):
            for j in range(i+1,len(A)):
                d = A[j]-A[i]
                dic[(j,d)]=dic.get((i,d),1)+1
                # if (i,d) in dic:
                #     dic[(j,d)]=dic[(i,d)]+1
                # else:
                #     dic[(j,d)]=2
                res = max(res,dic[(j,d)])
        return res

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        d = {}
        dp = {i:defaultdict(int) for i in range(len(A))}
        mx = 0
        
        for i in range(len(A)):
            if A[i] not in d: d[A[i]] = 1

            for j in range(0, i):
                ap = A[i] + -A[j]
            
            
                if A[i] + -ap in d:
                    dp[i][ap] = dp[j][ap] + 1
                    mx = max(mx, dp[i][ap])
                    
        return mx+1 
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        ans = 0
        dp = {}
        #dp[i,d] = max length subseq up to i with diff d
        for i in range(len(A)):
            for j in range(i):
                d = A[i] - A[j]
                if (j,d) in dp:
                    dp[i,d] = dp[j,d] + 1
                else:
                    dp[i,d] = 2
                ans = max(ans, dp[i,d])
        return ans
        

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
                    
        dp = [{} for _ in range(len(A))]
        res = 0
        for i in range(len(A)):            
            for j in range(i):  
                x = A[i] - A[j]                
                dp[i][x] = max(dp[j][x]+1 if x in dp[j] else 0, 2 if x not in dp[i] else dp[i][x])
                res = max(dp[i][x],res)
                
        #print(dp)
        return res
                
        
                
        

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        dp = []
        for i, x in enumerate(A):
            nd = collections.defaultdict(int)
            dp.append(nd)
            for j in range(i):
                diff = x - A[j]
                dp[i][diff] = dp[j][diff] + 1
        return max(max(y.values()) for y in dp) + 1
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
#         dp = defaultdict(int)
        
#         for i in range(len(A)):
#             for j in range(i+1, len(A)):
#                 d = A[j] - A[i]
                
#                 if (i, d) in dp:
#                     dp[j, d] = dp[i, d] + 1
#                 else:
#                     dp[j, d] = 2
        
#         return max(dp.values())

        dp = {}
        for i, Ai in enumerate(A):
            for j in range(i+1, len(A)):
                b = A[j] - Ai
                if (i,b) not in dp: dp[j,b] = 2
                else              : dp[j,b] = dp[i,b] + 1
        return max(dp.values())

class Solution2:
    def longestArithSeqLength(self, A: List[int]) -> int:
        n = len(A)
        dp = [[0]*501 for i in range(n)]
        max_val = 0
        for i in range(n):
            for j in range(i):
                dif = A[i] - A[j]
                dp[i][dif] = max(dp[i][dif], dp[j][dif] + 1)
                max_val = max(dp[i][dif], max_val)
        #print(dp)
        return max_val + 1
    
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        n = len(A)
        dp = collections.defaultdict(dict)
        max_val = 0
        for i in range(n):
            for j in range(i):
                dif = A[i] - A[j]
                dp[dif].setdefault(i, 0)
                dp[dif][i] = dp[dif].get(j,0) + 1
                max_val = max(dp[dif][i], max_val)
        #print(dp)
        return max_val + 1    

class Solution1:
    def longestArithSeqLength(self, A: List[int]) -> int:
        n = len(A)
        dp = [collections.defaultdict(int) for _ in range(n)] 
        res = 0
        for i in range(1, n):
            for j in range(0, i):
                diff = A[i]-A[j]
                dp[i][diff] = dp[j][diff] + 1
                    
                res = max(res, dp[i][diff])
        return res + 1
class Solution:
    def longestArithSeqLength(self, nums: List[int]) -> int:
        if not nums:
            return 0
        graph = collections.defaultdict(lambda: collections.defaultdict(int))
        
        res = 0
        for i in range(1, len(nums)):
            for j in range(i):
                diff = nums[i] - nums[j]
                prev_diffs = graph[j]
    
                prev_diff = prev_diffs[diff]
                graph[i][diff] = prev_diff + 1
                res = max(res, graph[i][diff])
        return res + 1
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        dp = {}
        ans = 0
        for i in range(len(A)):
            for j in range(i+1, len(A)):
                b = A[j] - A[i]
                if (i,b) not in dp: 
                    dp[(j,b)] = 2
                else: 
                    dp[(j,b)] = dp[(i,b)] + 1
                ans = max(ans, dp[(j, b)])
        return ans
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        dp = defaultdict(int)
        
        for i in range(1, len(A)):
            for j in range(i):
                d = A[i] - A[j]
                
                if (j, d) in dp:
                    dp[i, d] = dp[j, d] + 1
                else:
                    dp[i, d] = 2
        
        return max(dp.values())
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        n = len(A)
        tracker = {}
        max_length = 1
        for i in range(n):
            for j in range(i):
                diff = A[i] - A[j]
                if (diff, j) in tracker:
                    tracker[(diff, i)] = tracker[(diff, j)] + 1
                else:
                    tracker[(diff, i)] = 2
                max_length = max(max_length, tracker[(diff, i)])
        return max_length

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        if not A:
            return 0
        if len(A) <= 2:
            return len(A)
        
        dic = {}
        
        for idx, val in enumerate(A):
            for j in range(idx + 1, len(A)):
                diff = A[j] - val
                if (diff, idx) in dic:
                    dic[diff, j] = dic[(diff, idx)] + 1
                else:
                    dic[diff, j] = 2
        return max(dic.values())
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        dp = {}
        for i in range(len(A)):
            for j in range(i + 1, len(A)):
                diff = A[j] - A[i]
                dp[(j, diff)] = dp.get((i, diff), 1) + 1
        return max(dp.values())
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        dp = [[1 for i in range(1001)] for j in range(len(A))]
        lastIndex = {}
        ans = 1
        for i,num in enumerate(A):
            for j in range(1001):
                commonDiff = j-500
                lastNumIndex = lastIndex.get(num-commonDiff,-1)
                if(lastNumIndex >= 0 and lastNumIndex < i):
                    dp[i][j] = dp[lastNumIndex][j] + 1
                    ans = max(ans,dp[i][j])
            if(num not in lastIndex):
                lastIndex[num] = -1
            lastIndex[num] = i
        return ans
        

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        cache = dict()
        
        for i in range(len(A)):
            for j in range(i + 1, len(A)):
                diff = A[j] - A[i]
                
                if (i, diff) not in cache:
                    cache[(j, diff)] = 2
                else:
                    cache[(j, diff)] = 1 + cache[(i, diff)]
                
                
        return max(cache.values())
'''
dp[i][d] = longest subsequence ending at i with difference d

dp[j][d] = 1 + max(
    dp[j][A[j] - A[i]]
) for j < i


'''
class Solution:
    #O(n^2) Bottom-up DP
    def longestArithSeqLength(self, A):
        dp = []
        ans,n = 2,len(A)
        for i in range(n):
            dp.append(dict())
            for j in range(i-1,-1,-1):
                diff = A[i]-A[j]# create a start with A[i],A[j]
                if diff not in dp[i]:
                    dp[i][diff] = 2
                # going backward to make sure information can be used!
                if diff in dp[j]:
                    dp[i][diff] = max(dp[j][diff]+1,dp[i][diff])
                ans = max(ans,dp[i][diff])
        return ans
        

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        M=max(A)+1
        dp=[]
        for i in range(len(A)):
            temp=[1]*(M*2)
            dp.append(temp)
        m=0
        for i in range(len(A)):
            
            for j in range(i):
                delta=A[i]-A[j]
                k=delta+M
            #    print(k,M)
                dp[i][k]=dp[j][k]+1
                m=max(m,dp[i][k])
  #      m=0
  #      for i in range(2*M):
  #          for j in range(len(A)):
  #              m=max(dp[j][i],m)
  #    #  print(dp)
        return m
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        # let dp[i][k] is the length of the arith seq with step size k
        # range is -500, 500
        n = len(A)
        m = 1001
        dp = [ [1] * (m+1) for _ in range(n)] # can replace it by a dict, will be quicker
        ans = float('-inf')
        for i in range(1,n):
            for k in range(i):
                dp[i][A[i]-A[k]+500] = max(dp[i][A[i]-A[k]+500], 1 + dp[k][A[i]-A[k]+500])  ## a_i - a_{i-1} = j 
            ans = max(max(dp[i]),ans)
        return ans

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        dp = {}; n = len(A); res = 2
        for i in range(n):
            for j in range(i+1, n):
                b = A[j] - A[i]
                if (i,b) not in dp: dp[j,b] = 2
                else              : dp[j,b] = dp[i,b] + 1
                res = max(res, dp[j,b])
        return res
        #return max(dp.values())

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        cache = {}
        ans = 0
        for i in range(len(A)):
            for j in range(i+1, len(A)):
                # print(cache)
                diff = A[j] - A[i]
                if (i, diff) in cache:
                    cache[j, diff] = cache[i, diff] + 1
                    ans = max(cache[j, diff], ans)
                else:
                    cache[j, diff] = 2
                    ans = max(cache[j, diff], ans)
        return ans


class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        dp, n, ans = {}, len(A), 0
        for i in range(n):
            for j in range(i+1, n):
                diff = A[j] - A[i]
                dp[(j, diff)] = dp.get((i, diff), 1) + 1
                ans = max(ans, dp[(j, diff)])
                
        return ans

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        n = len(A)
        dp = {}
        mostFreq = -1
        for i in range(1,n):
            for j in range(i-1, -1,-1):
                diff = A[i] - A[j]
                prevVal = 1
                if (i,diff) in dp: prevVal = dp[ (i,diff)]
                if (j,diff) in dp: 
                    dp[(i,diff)] = max(dp[(j,diff)], prevVal-1) + 1
                else:
                    if (i,diff) not in dp:
                        dp[(i,diff)] = 1
                # mostFreq = max(dp[(i,diff)], mostFreq)
        ret = -1
        for k,v in dp.items():
            ret = max(ret,v)
        return ret+1
        return mostFreq+1
class Solution:
    def longestArithSeqLength(self, A):
        dp = collections.defaultdict(int)
        mx = 0
        for i in range(len(A)):
            for j in range(i):
                diff = A[i] - A[j]
                j_key = (j, diff)
                i_key = (i, diff)
                
                dp[i_key] = max(2, dp[j_key]+1)
        return max(dp.values())
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        dp = {}
        n = len(A)
        res = 2
        for i in range(n):
            for j in range(i):
                d = A[i] - A[j]
                dp[d, i] = dp.get((d, j), 1) + 1
                res = max(res, dp[d, i])
        return res
                

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        if A is None or not A:
            return 0
        
        N = len(A)
        
        f = [{} for _ in range(N + 1)]
        
        ret = 0
        
        for i in range(2, N + 1):
            for j in range(1, i):
                diff = A[i - 1] - A[j - 1]
                if diff in f[j]:
                    f[i][diff] = f[j][diff] + 1
                else:
                    f[i][diff] = 2
                
                ret = max(ret, f[i][diff])
        
        return ret
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        n = len(A)
        dp = {}
        for j in range(1,n):
            for i in range(j):
                diff = A[j] - A[i]
                dp[j,diff] = max(dp.get((j,diff), 2), dp.get((i,diff),1)+1)
                
        return max(dp.values())
class Solution:
  def longestArithSeqLength(self, A: List[int]) -> int:
    dp = {}
    res = 0

    for i in range(len(A)):
      for j in range(0, i):
        d = A[i] - A[j]
        
        dp[i, d] = dp.get((j, d), 1) + 1

        res = max(res, dp[i, d])

    return res        

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        # DP?
        if len(A) <= 2: return len(A)
        DP = [{} for i in range(len(A))]
        ret = 0
        for i in range(1, len(A)):
            # j < i
            for j in range(i):
                diff = A[i] - A[j]
                l = 0
                if diff in DP[j]:
                    l = DP[j][diff] + 1
                else:
                    l = 2 # A[j] and A[i]
                new_longest = max(l, DP[i].get(diff, 0))
                ret = max(ret, new_longest)
                DP[i][diff] = new_longest
        return ret
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        
        dp = {}
        m = 0
        for i in range(len(A)):
            for j in range(i):
                d = A[i] - A[j]
                j1 = (j,d)
                i1 = (i,d)
                if dp.get(j1) != None:
                    dp[i1] = 1 + dp[j1]
                else:
                    dp[i1] = 1
                m = max(m,dp[i1])
        
        return m+1

from collections import defaultdict
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:    
        count = 0
        memo = defaultdict(int)
        for i in range(len(A)):
            for j in range(i+1,len(A)):
                num1 = A[i]
                num2 = A[j]
                diff = A[j] - num1
                val = memo[(i,diff)] + 1
                memo[(j, diff)] = val
                count = max(val, count)
        return count + 1 if count > 0 else count

                    
                    

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        dp = {}
        for i in range(len(A)):
            for j in range(i+1, len(A)):
                b = A[j] - A[i]
                if (i,b) not in dp: 
                    dp[(j,b)] = 2
                else: 
                    dp[(j,b)] = dp[(i,b)] + 1
        return max(dp.values())
class Solution:
    #natural algorithm is O(n^3) by test for each i<j all remaining sequence
    #O(n^2) Bottom-up DP
    def longestArithSeqLength(self, A):
        dp = []
        ans,n = 2,len(A)
        for i in range(n):
            dp.append(dict())
            for j in range(i-1,-1,-1):
                diff = A[i]-A[j]# create a start with A[i],A[j]
                if diff not in dp[i]:
                    dp[i][diff] = 2
                # going backward to make sure information can be used!
                if diff in dp[j]:
                    dp[i][diff] = max(dp[j][diff]+1,dp[i][diff])
                ans = max(ans,dp[i][diff])
        return ans
        

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        dp = {}
        for i in range(len(A)):
            for j in range(i+1, len(A)):
                dp[(j, A[j]- A[i])] = dp.get((i, A[j] - A[i]),1) + 1
        return max(dp.values())
            

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        dp = dict()
        max_val = 0
        for i in range(len(A)):
            for j in range(i+1, len(A)):
                dp[j, A[j] - A[i]] = dp.get((i, A[j] - A[i]), 1) + 1
                if dp[j, A[j] - A[i]] > max_val:
                    max_val = dp[j, A[j] - A[i]]
        return max_val
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        # table[index][diff] equals to the length of 
        # arithmetic sequence at index with difference diff.
        table = dict()
        max_v = 0
        for i in range(len(A)):
            for j in range(i+1,len(A)):
                
                _diff = A[j] - A[i]
                if (i,_diff) in table.keys():
                    table[j,_diff] = table[i,_diff] + 1
                else:
                    table[j,_diff] = 2 # the first diff
                    # will corrspond to two values [v1,v2]
                max_v = max(max_v,table[j,_diff])
                    
        return max_v
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        dp = {}
        for i in range(1, len(A)):
            for j in range(i):
                diff = A[i]-A[j]
                dp[(i, diff)] = max(dp.get((i, diff), 2), dp.get((j, diff), 1)+1)

        
        # print(dp)
        return max(dp.values())
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        mp = defaultdict(list)
        for i, a in enumerate(A):
            mp[a].append(i)
        
        @lru_cache(None)
        def dp(i, diff=None):
            res = 1
            if diff is None:
                for j in range(i):
                    res = max(res, 1+ dp(j, A[i] - A[j]))
            else:
                for j in mp[A[i] - diff]:
                    if j < i: res = max(res, 1+ dp(j, diff))
            return res
        
        return max([dp(i) for i in range(len(A))])

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        '''
        dp = {}
        for i in xrange(len(A)):
            for j in xrange(i + 1, len(A)):
                dp[j, A[j] - A[i]] = dp.get((i, A[j] - A[i]), 1) + 1
        return max(dp.values())
        '''
        l=len(A)
        dp={}
        c=collections.defaultdict(list)
        for i in range(l-1):
            for j in range(i+1,l):
                #c[(i,j)]=A[i]-A[j]
                c[i].append(A[i]-A[j])
                dp[j,A[j]-A[i]]=dp.get((i,A[j]-A[i]),1)+1
        #print(c)
        return max(dp.values())
        res=2
        #for i in range(l-1):
            #analyze c[i]
            

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        dp = {}
        n = len(A)
        for i in range(n):
            for j in range(i+1,n):
                diff = A[i]-A[j]
                dp[(diff, j)] = dp.get((diff,i), 1) + 1
        #print(dp)
        return max(dp.values())
class Solution:
    def longestArithSeqLength(self, A):
        max_len, dp = 0, [{} for _ in range(len(A))]  # index_of_array: {diff: curr_max_len}
        for i in range(1, len(A)):
            for j in range(0, i):
                diff = A[i] - A[j]
                if diff in dp[j]:
                    dp[i][diff] = dp[j][diff] + 1
                else:
                    dp[i][diff] = 2
                max_len = max(max_len, dp[i][diff] )
        return max_len

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        dp = {}  # key = (index, diff), value = len of sequences
        for i in range(len(A)):
            for j in range(i + 1, len(A)):
                curr_diff = A[j] - A[i]
                dp[(j, curr_diff)] = dp.get((i, curr_diff), 1) + 1
        return max(dp.values())

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        dp = [{} for i in range(len(A))]
        res = 0
        for i in range(len(A)):
            for j in range(i):
                # in the case that we are attaching to a single element
                if A[i]-A[j] in dp[j]:
                    dp[i][A[i]-A[j]] = max(2, 1 + dp[j][A[i]-A[j]])
                else:
                    dp[i][A[i]-A[j]] = 2
                res = max(res, dp[i][A[i]-A[j]])
        return res
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        dp = collections.defaultdict(int)
        
        for i in range(len(A)):
            for j in range(i):
                diff = A[i] - A[j]
                dp[i, diff] = max(2, dp[j, diff] + 1)
        
        return max(dp.values())
from collections import defaultdict
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        n = len(A)
        if n < 3:
            return n 
        
        dp = defaultdict(lambda: defaultdict(int))
        res = 2
        for i in range(1, n):
            for j in range(i):
                gap = A[i] - A[j]
                dp[i][gap] = max(dp[j][gap] + 1, 2)
                res = max(res, dp[i][gap])
        return res
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        dp = defaultdict(int)
        
        for i in range(len(A)):
            for j in range(i+1, len(A)):
                d = A[j] - A[i]
                
                if (i, d) in dp:
                    dp[j, d] = dp[i, d] + 1
                else:
                    dp[j, d] = 2
        
        return max(dp.values())


#[Runtime: 4284 ms, faster than 10.74%] DP
#O(N^2)
#NOTE: diff can be either positive or negative
#f[i]: the longest length of arithmetic subsequences who takes A[i] as the tail.
#f[i] = defaultdict(lambda: 1)
#f[i] = {diff: longest length}
#f[i] = max(f[i][d], f[j][d] + 1) for j < i and d:=A[i]-A[j]
from collections import defaultdict
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        dp = defaultdict(lambda: 1)
        for i, a in enumerate(A):
            for j in range(i):
                diff = A[i] - A[j]
                if dp[i, diff] < dp[j, diff] + 1:
                    dp[i, diff] = dp[j, diff] + 1
        return max(dp.values())
from collections import defaultdict
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:    
        count = 0
        memo = defaultdict(int)
        for i in range(len(A)):
            for j in range(i+1,len(A)):
                num1 = A[i]
                num2 = A[j]
                diff = A[j] - num1
                val = memo[(num1,diff,i)] + 1
                memo[(num2, diff,j)] = val
                count = max(val, count)
        return count + 1 if count > 0 else count

                    
                    

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        if not A:
            return 0
        dp = collections.defaultdict(int)
        for i in range(len(A)):
            for j in range(i+1, len(A)):
                dp[(j, A[j]-A[i])] = dp[(i, A[j]-A[i])] + 1
        return max(dp.values()) + 1

class Solution:
    #    [9, 4, 7, 2, 10]
    #     1  2  2  2   3
    # (4, 5): 2
    # (7, 2): 2
    # (7, 3): 2
    # (2, 7): 2
    # (2, 2): 2
    # (2, 5): 2
    # (10, 1): 2
    # (10, 6): 2
    # (10, 3): 3
    # (10, 8): 2
    # store the last val in a sequence formed, its difference and length
    
    # [24, 13, 1, 100, 0, 94, 3, 0, 3] result = 2
    # (13, 11): 2
    # (1, 23): 2
    # (1, 12): 2
    # (100, 86): 2
    # (100, 87): 2
    # (100, 99): 2
    # (0, -24): 2
    # (0, -13): 2
    # (0, -1): 2
    # (0, -100): 2
    # (94, 70): 2
    # (94, 81): 2
    # (94, 93): 2
    # (94, -6): 2
    # (94, 94): 2
    # (3, -11): 2
    # (3, -10): 2
    # (3, -2): 2
    # (3, -97): 2
    # (3, 3, 7): 2
    # (3, -91): 2
    
    def longestArithSeqLength(self, A: List[int]) -> int:
        
        seqs = {} # stores mapping of (last_num_in_seq, diff) to length
        
        result = 2
        
        for i in range(1, len(A)):
            for j in range(i):
                diff = A[i] - A[j]
                
                if (A[j], diff, j) in seqs:
                    seqs[(A[i], diff, i)] = seqs[(A[j], diff, j)] + 1
                else:
                    seqs[(A[i], diff, i)] = 2
                    
                result = max(result, seqs[(A[i], diff, i)])
                
        return result
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        dp = list(dict() for i in range(len(A)))
        maxsize = 0
        for i in range(1,len(A)):
            for j in range(0,i):
                diff = A[j] - A[i]
                if(diff in dp[j]):
                    dp[i][diff] = dp[j][diff] + 1
                else:
                    dp[i][diff] = 1
            
                maxsize = max(maxsize, dp[i][diff])
        
        return maxsize + 1

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        dp, ans = [], 2
        for i in range(len(A)):
            dp.append({})
            for j in range(i-1, -1, -1):                
                dif = A[i]-A[j]
                if dif not in dp[i]: dp[i][dif] = 2
                if dif in dp[j]: dp[i][dif] = max(dp[j][dif]+1, dp[i][dif])
                ans = max(ans, dp[i][dif])
        return ans
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        dp = {}
        for i in range(len(A)):
            for j in range(i):
                dp[i, A[i] - A[j]] = dp.get((j, A[i] - A[j]), 1) + 1
        return max(dp.values())
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        if len(A)==0:
            return 0
        
        ld=[{} for i in range(len(A))]
        ans=0
        ind=-1
        for i in range(1,len(A)):
            for j in range(i-1,-1,-1):
                diff=A[i]-A[j]
                if diff in ld[j]:
                    # print(i,j,diff,ld[j][diff])
                    if diff in ld[i]:
                        ld[i][diff]=max(ld[j][diff]+1,ld[i][diff])
                    else:
                        ld[i][diff]=ld[j][diff]+1
                else:
                    if diff in ld[i]:
                        ld[i][diff]=max(2,ld[i][diff])
                    else:
                        ld[i][diff]=2
                if ld[i][diff]>ans:
                    ind=i
                ans=max(ans,ld[i][diff])
                
        # print(ind,ld[ind],A[ind],len(A))
        
        return ans

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        maxd = max(A) - min(A)
        dp = [[1 for j in range(2*maxd+1)] for i in range(len(A))]
        maxv = 1
        for i in range(1, len(dp)):
            for j in range(i-1, -1, -1):
                diff = A[i]- A[j]
                dp[i][diff+maxd] = max(dp[i][diff+maxd], dp[j][diff+maxd] + 1)
                maxv = max(maxv, dp[i][diff+maxd])
        return maxv

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        if len(A) <= 2:
            return len(A)
        idx_diff_to_longest = {}
        longest = 0
        for i in range(1, len(A)):
            for j in range(i):
                diff = A[i] - A[j]
                prev_len = idx_diff_to_longest[(j, diff)] if (j, diff) in idx_diff_to_longest else 1
                idx_diff_to_longest[(i, diff)] = prev_len + 1
                longest = max(longest, prev_len + 1)
        return longest
from collections import defaultdict

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        A_indices = defaultdict(list)
        for i, a in enumerate(A):
            A_indices[a].append(i)
        lengths = defaultdict(lambda: 2)
        best = 2
        
        for i in range(len(A) - 3, -1, -1):
            for j in range(i + 1, len(A) - 1):
                if 2 * A[j] - A[i] in A_indices:
                    indices = A_indices[2 * A[j] - A[i]]
                    # find earliest occurrence of 2 * A[j] + A[i] after j
                    if indices[-1] <= j:
                        continue
                    if indices[0] > j:
                        r = 0
                    else:
                        l = 0
                        r = len(indices) - 1
                        while l < r - 1:
                            mid = (l + r) // 2
                            if indices[mid] <= j:
                                l = mid
                            else:
                                r = mid
                    lengths[i, j] = 1 + lengths[j, indices[r]]
                    best = max(best, lengths[i, j])
        return best
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        #u8fd9u9898bottom upu5199u8d77u6765u5feb?
        dp = defaultdict(lambda: 1) #default u662f 1uff0c u81eau5df1u80afu5b9au662fu81eau5df1u7684seq
        for i in range(len(A)):
            for j in range(i): #u4e4bu524du7684
                diff = A[i] - A[j]
                dp[(i, diff)] = 1+ dp[(j, diff)]#u52a0u901f u8fd9u91ccu4e0du9700u8981u7528maxu6765u53d6u56e0u4e3au662fu4eceu524du5f80u540eu904du5386uff0cu6700u540evalidu7684diffu80afu5b9au662fu6700u5927u7684
                
        return max(dp.values())
class Solution:
    #    [9, 4, 7, 2, 10]
    #     1  2  2  2   3
    # (4, 5): 2
    # (7, 2): 2
    # (7, 3): 2
    # (2, 7): 2
    # (2, 2): 2
    # (2, 5): 2
    # (10, 1): 2
    # (10, 6): 2
    # (10, 3): 3
    # (10, 8): 2
    # store the last val in a sequence formed, its difference and length
    
    # [24, 13, 1, 100, 0, 94, 3, 0, 3] result = 2
    # (13, 11): 2
    # (1, 23): 2
    # (1, 12): 2
    # (100, 86): 2
    # (100, 87): 2
    # (100, 99): 2
    # (0, -24): 2
    # (0, -13): 2
    # (0, -1): 2
    # (0, -100): 2
    # (94, 70): 2
    # (94, 81): 2
    # (94, 93): 2
    # (94, -6): 2
    # (94, 94): 2
    # (3, -11): 2
    # (3, -10): 2
    # (3, -2): 2
    # (3, -97): 2
    # (3, 3, 7): 2
    # (3, -91): 2
    
    def longestArithSeqLength(self, A: List[int]) -> int:
        
        seqs = {} # stores mapping of (last_num_in_seq, diff) to length
        
        result = 2
        
        for i in range(1, len(A)):
            for j in range(0, i):
                diff = A[i] - A[j]
                
                if (A[j], diff, j) in seqs:
                    seqs[(A[i], diff, i)] = seqs[(A[j], diff, j)] + 1
                else:
                    seqs[(A[i], diff, i)] = 2
                    
                result = max(result, seqs[(A[i], diff, i)])
                
        return result
from collections import Counter
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        if len(A)<=2:
            return len(A)
        dp = [Counter() for _ in range(len(A))]
        ans = 2
        for i in range(len(A)):
            for j in range(i+1,len(A),1):
                diff = A[j]-A[i]
                if dp[i][diff] != 0:
                    dp[j][diff] = dp[i][diff] + 1
                else:
                    dp[j][diff] = 2
                ans = max(ans, dp[j][diff])
        return ans
                

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        dp = {}
        n = len(A)
        for i in range(n):
            for j in range(i+1, n):
                dp[(j, A[j]-A[i])] = dp.get((i, A[j]-A[i]), 1) + 1
        
        return max(dp.values())
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        d = [{} for _ in range(len(A))]
        longest = 0
        for i in range(1, len(A)):
            for j in range(i):
                diff = A[i] - A[j]
                d[i][diff] = max(d[i].get(diff, 0), 1 + d[j].get(diff, 1))
                longest = max(longest, d[i][diff])
        return longest

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        n = len(A)
        if n < 3:
            return n

        counts = [{} for _ in A]
        max_count = 0

        for i in range(0, len(A)):
            for j in range(0, i):
                diff = A[i] - A[j]
                counts[i][diff] = max(counts[i].get(diff, 1), counts[j].get(diff, 1) + 1)
                max_count = max(max_count, counts[i][diff])

        return max_count

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        N = len(A)
        dp = defaultdict(lambda: 1)
        for j in range(N):
            nxt = defaultdict(lambda: 1)
            for i in range(j):
                y, z = A[i], A[j]
                d = delta = z - y
                nxt[z, d] = max(nxt[z, d], dp[y, d] + 1)
            dp.update(nxt)
        return max(dp.values())

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        dp = {}
        for i in range(len(A)):
            for j in range(i+1, len(A)):
                dp[j, A[j] - A[i]] = dp.get((i, A[j] - A[i]), 1) + 1;
        return max(dp.values())
from collections import defaultdict

class Solution:
    def longestArithSeqLength(self, nums: List[int]) -> int:
        dp = defaultdict(int)
        n = len(nums)
        max_val = 0
        for i in range(n-1):
            for j in range(i+1, n):
                diff = nums[i] - nums[j]
                dp[(j, diff)] = 1 + dp[(i, diff)]
                max_val = max(max_val, dp[(j,diff)])
        return max_val + 1
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        dp = {}
        for i in range(len(A)):
            for j in range(0, i):
                dp[i, A[i] - A[j]] = dp.get((j, A[i] - A[j]), 1) + 1
        return max(dp.values())
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        result = 1
        n = len(A)
        umap = [dict() for i in range(n)]
        
        for i in range(1, n):
            for j in range(i):
                diff = A[i] - A[j]
                jval = (umap[j][diff] if diff in umap[j] else 0)
                if diff not in umap[i]:
                    umap[i][diff] = 0
                umap[i][diff] = max(umap[i][diff], jval + 1)
                result = max(result, umap[i][diff])
        return result+1
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        d = [{} for _ in range(len(A))]
        longest = 0
        for i in range(1, len(A)):
            for j in range(i):
                diff = A[i] - A[j]
                d[i][diff] = max(d[i].get(diff, 0), 1 + d[j].get(diff, 1))
                longest = max(longest, d[i][diff])
        return longest
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        N = len(A)
        dp = defaultdict(lambda: 1)
        for j in range(N):
            dp2 = defaultdict(lambda: 1)
            for i in range(j):
                y, z = A[i], A[j]
                d = delta = z - y
                dp2[z, d] = max(dp2[z, d], dp[y, d] + 1)
            dp.update(dp2)
            # for k, v in dp.items():
            #     print(k, v)
            # print()
        return max(dp.values())

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        # dp[i][k] = longestArithSeqLength(A[:i+1]) with step size k
    
        dp = dict()
        res = 0
    
        for i in range(1, len(A)):
            for prev_i in range(i):
                step = A[i] - A[prev_i]
                prev_step = dp[(prev_i, step)] if (prev_i, step) in dp else 1
                dp[(i, step)] = prev_step + 1
                
                res = max(res, dp[(i, step)])
        
        return res
from collections import defaultdict

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        A_indices = defaultdict(list)
        for i, a in enumerate(A):
            A_indices[a].append(i)
        # lengths (i, d) longest arithmetic subsequence starting at i
        # with difference d
        lengths = defaultdict(lambda: 1)
        best = 0
        
        for i in range(len(A) - 2, -1, -1):
            for j in range(len(A) - 1, i, -1):
                diff = A[j] - A[i]
                lengths[i, diff] = lengths[j, diff] + 1
                best = max(best, lengths[i, diff])
        return best
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        if len(A)<=2:
            return len(A)
        
        dp = [{} for _ in range(len(A))]
        dp[1][A[1]-A[0]] = 2
        
        maxLen = 0
        for i in range(2, len(A)):
            for j in range(i):
                diff = A[i]-A[j]
                dp[i][diff] = max(dp[i][diff] if diff in dp[i] else 0, dp[j][diff]+1 if diff in dp[j] else 2)
                maxLen = max(maxLen, dp[i][diff])
        
        return maxLen
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        dp = {}
        
        for i in range(len(A)):
            for j in range(i):
                d = A[j] - A[i]
                if (j, d) in dp:
                    dp[(i, d)] = dp[(j, d)] + 1
                else: 
                    dp[(i, d)] = 2
                    
        return max(dp.values())
class Solution:
    def longestArithSeqLength(self, A):
        dp = {}
        for i in range(len(A)):
            for j in range(i + 1, len(A)):
                if (i, A[j] - A[i]) in dp:
                    dp[(j, A[j] - A[i])] = dp[(i, A[j] - A[i])] + 1
                else:
                    dp[(j, A[j] - A[i])] = 2
        return max(dp.values())
# dict.get(key,number) return number if get None
# u7528dict u7d00u9304 dict[(id,diff)]: count
# dict[(new id,diff)] = dict[(old id,diff)] || 1 + 1
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        dp = {}
        for i in range(len(A)):
            for j in range(i + 1, len(A)):
                x = dp.get((i, A[j] - A[i]), 1)
                # print(x)
                dp[j, A[j] - A[i]] = x + 1
            # print(dp)
        return max(dp.values())
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        dp = {}
        for i in range(1, len(A)):
            for j in range(i):
                diff = A[i]-A[j]
                dp[(i, diff)] = max(dp.get((i, diff), 2), dp.get((j, diff), 1)+1)

        return max(dp.values())
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        n = len(A)
        dp = [{} for i in range(n)]
        result = 0
        for i, a in enumerate(A):
            for j in range(i):
                l = dp[j].get(a - A[j], 1) + 1
                dp[i][a - A[j]] = max(dp[i].get(a - A[j], 0) ,l)
                result = max(result, l)
        return result
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int: 
        if not A:
            return 0
        result = 0
        N = len(A)
        d =  collections.defaultdict(int)
        for i in range(1,N):
            for j in range(i):
                diff = A[i] - A[j]
                d[(i,diff)] = d[(j,diff)] +1
                result = max(result, d[(i,diff)] )
        return result+1
        
        
 
        
            

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        
        
        diff_dict = {}
        max_ = 0
        for i in range(len(A)):
            diff_dict[i] = {}
            for j in range(i):
                diff = A[i] - A[j]
                if diff in diff_dict[j]:
                    diff_dict[i][diff] = diff_dict[j][diff] + 1
                else:
                    diff_dict[i][diff] = 1
                    
                max_ = max(max_, diff_dict[i][diff])
             
        return max_ + 1
                
                

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
#         diff_dict = {}
#         max_ = 0
#         for i in range(len(A)):
#             diff_dict[i] = {}
#             for j in range(i):
#                 diff = A[i] - A[j]
                
#                 if diff in diff_dict[j]:
#                     diff_dict[i][diff] = diff_dict[j][diff] + 1
                
#                 else:
#                     diff_dict[i][diff] = 1
                
                
#                 max_ = max(max_, diff_dict[i][diff])
                
#         return max_
                
                
                
                
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
#         diff_dict = {}
#         max_ = 0
#         for i in range(len(A)):
#             diff_dict[i] = {}
#             for j in range(i):
#                 curr = A[i]
#                 prev = A[j]

#                 diff = curr - prev 
#                 if diff in diff_dict[j]:
#                     diff_dict[i][diff] = diff_dict[j][diff] + 1
#                 else:
#                     diff_dict[i][diff] = 1

#                 max_ = max(max_, diff_dict[i][diff])

#         return (max_ + 1)

                
            
            

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
#         diff_dict = {}
#         max_ = 0
#         for i in range(len(A)):
#             diff_dict[i] = {}
#             for j in range(i):
#                 curr = A[i]
#                 prev = A[j]
                
#                 diff = curr - prev
                
#                 if diff in diff_dict[j]:
#                     diff_dict[i][diff] =  diff_dict[j][diff] + 1
#                 else:
#                     diff_dict[i][diff] = 1
                    
#                 max_ = max(max_, diff_dict[i][diff])
#         print(diff_dict)
#         return (max_ + 1)
            

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
#         mapping = {}
#         if len(A) < 2:
#             return len(A)
#         max_ = 0
    
#         for i in range(len(A)):
#             mapping[i] = {}
#             for j in range(0, i):
#                 diff = A[i] - A[j]
                
#                 if diff not in mapping[j]:
#                     mapping[i][diff] = 2
#                 else:
#                     mapping[i][diff] = mapping[j][diff] + 1
#                 max_ = max(max_, mapping[i][diff])
            
#         return max_
             
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
#         track = [0 for i in range(len(A))]
        
#         for i in range(1, len(A)):
#             track[i] = A[i] - A[i-1]
#         print(track)
        
#         num = {}
        
#         for i in range(1, len(track)):
#             val = (track[i] - track[i-1])
#             if val in num:
#                 num[val] += 1
#             else:
#                 num[val] = 1
                
#         print(num)
#         return max(num.values()) + 1
        

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        n = len(A)
        sol = 1
        l_d = [{} for _ in range(n)]
        for j in range(1,n):
            dj = l_d[j]
            for i in range(j):
                diff = A[j] - A[i]
                di = l_d[i]
                dj[diff] = max(dj.get(diff, 2), di.get(diff,1)+1)
                sol = max(sol, dj[diff])
                
        return sol
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        dp = {}
        for i in range(len(A)):
            for j in range(i + 1, len(A)):
                dp[j, A[j]-A[i]] = dp.get((i, A[j] - A[i]), 1) + 1
        return max(dp.values())
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        # want subsequence that is longest arithmetic 
        # dictionary with len longest subsequence at j
        # for i in range(A):
        #    for j in range(i):
        #        diff = j - i # difference between the two
        #        dictionary[(diff, i)] = max(dictionary[(diff, i)], dictionary[(diff, j)]+1)
        #        
        
        # (-5, 1) = 1
        # (-2, 2) = 1
        # (3, 2) = 1
        # (-7,3) = 1
        # (-2,3) = 1
        long_len_sub_at_pos = {}
        
        for i in range(len(A)):
            for j in range(i):
                diff = A[i] - A[j]
                
                # if (diff, j) in long_len_sub_at_pos:
                #     long_len_sub_at_pos[(diff, i)] = max(long_len_sub_at_pos[(diff, i)], long_len_sub_at_pos[(diff, j)] + 1)
                # else:
                #     long_len_sub_at_pos[(diff, i)] = 2

                sub_len_at_j = long_len_sub_at_pos.get((diff,j), 1) 
                long_len_sub_at_pos[(diff, i)] = max(long_len_sub_at_pos.get((diff,i), 0), sub_len_at_j + 1)
        
        #values in the dictionary would be the length of hte subseq
        #loop over and find the max subseq
        
        return max(long_len_sub_at_pos.values())
class Solution:
    def longestArithSeqLength(self, A):
        dp = dict()
        for endi, endv in enumerate(A[1:], start = 1):
            for starti, startv in enumerate(A[:endi]):
                diff = endv - startv
                if (starti, diff) in dp:
                    dp[(endi, diff)] = dp[(starti, diff)] + 1
                else:
                    dp[(endi, diff)] = 2
        return max(dp.values())
    
class Solution:
    def longestArithSeqLength(self, A):
        dp = dict()
        for starti, startv in enumerate(A):
            for endi, endv in enumerate(A[starti+1:], start = starti+1):
                diff = endv - startv
                if (starti, diff) in dp:
                    dp[(endi, diff)] = dp[(starti, diff)] + 1
                else:
                    dp[(endi, diff)] = 2
        return max(dp.values())
    
class Solution:
    def longestArithSeqLength(self, A):
        N = len(A)
        dp = [{0:1} for _ in range(N)]
        for end in range(1, N):
            for start in range(end):
                diff = A[end] - A[start]
                if diff in dp[start]:
                    dp[end][diff] = dp[start][diff] + 1
                else:
                    dp[end][diff] = 2
        return max(max(dp[end].values()) for end in range(1, N))
    
class Solution:
    def longestArithSeqLength(self, A):
        ans = 2
        n = len(A)
        index = {}
        dp = [[2] * n for i in range(n)]
        
        for i, val in enumerate(A[:-1]):
            for j in range(i+1, n):
                first = val + val - A[j]
                if first in index:
                    dp[i][j] = dp[index[first]][i] + 1
                    #ans = max(ans, dp[i][j])
                    if dp[i][j] > ans: ans = dp[i][j]
            index[val] = i
        return ans

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        seen = set()
        for i in range(len(A)):
            for j in range(i,len(A)):
                seen.add(A[j] - A[i])
        ans = 0
        for j in seen:
            d = defaultdict(int)
            for i in A:
                d[i] = d[i-j] + 1
                ans = max(ans,d[i])
        return ans
        

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        dp = {}
        n = len(A)
        ans = 0
        for i in range(n - 1):
            for j in range(i + 1, n):
                d = A[j] - A[i]
                dp[j, d] = dp.get((i, d), 1) + 1
                ans = max(ans, dp[j, d])
        
        return ans
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        dp = {}
        for i in range(len(A)):
            for j in range(i + 1, len(A)):
                dp[j, A[j] - A[i]] = dp.get((i, A[j] - A[i]), 1) + 1
        return max(dp.values())

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        n = len(A)
        if n < 3:
            return n
        self.memo = {}
        self.res = 0
        for i in range(n - 1):
            for j in range(i + 1, n):
                diff = A[j] - A[i]
                if (i, diff) in self.memo:
                    self.memo[(j, diff)] = self.memo[(i, diff)] + 1
                else:
                    self.memo[(j, diff)] = 2
                self.res = max(self.res, self.memo[(j, diff)])
        return self.res

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        dp = {}
        ans = 1
        
        for i in range(len(A)):
            for j in range(i + 1, len(A)):
                dp[(j, A[j] - A[i])] = dp.get((i, A[j] - A[i]), 1) + 1
                ans = max(ans, dp[(j, A[j] - A[i])])
        
        return ans
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        from collections import Counter, defaultdict
        
        min_val = min(A)
        max_val = max(A)
        
        global_best = -1
        dp = {}
        prev = defaultdict(list)
        
        for i, v in enumerate(A):
            dp[i] = {}
            
            # print(f'--------- PROCESSING INDEX {i}')
            for d in range(min_val-v, max_val-v+1):
                # print(f'PROCESSING DIFF {d}')
                if v+d < min_val or v+d > max_val:
                    raise Exception()
                
                best = 0
                
                if v+d in prev:
                    for j in prev[v+d]:
                        best = max(best, dp[j].get(d, 1))
            
                dp[i][d] = best + 1
                global_best = max(global_best, dp[i][d])
            
            prev[v].append(i)
        # print(dp)
        
        return global_best
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        
        lim = len(A)
        
        d = dict()
        ans = []
        for i in range(0,lim):
            d[i]=dict()
            ans.append(2)
        
        for i in range(1,lim):
            for j in range(0,i):
                if A[i]-A[j] in d[j]:
                    d[i][A[i]-A[j]]=d[j][A[i]-A[j]]+1
                    if d[i][A[i]-A[j]]>ans[i]:ans[i] = d[i][A[i]-A[j]]
                        
                else:
                    d[i][A[i]-A[j]] = 2
        ###print(d)            
        return max(ans)
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        # dp approach
        
        dp = {}
        
        for i in range(len(A)):
            for j in range(i + 1, len(A)):
                dp[j, A[j] - A[i]] = dp.get((i, A[j] - A[i]), 1) + 1
                
        return max(dp.values())
from collections import defaultdict

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        
        largestLen = 0
        
        longestArithSequence = [defaultdict(lambda: 1) for _ in range(len(A))]
        
        for i in range(1, len(A)):
            
            for j in range(i):
                diff = A[i] - A[j]
                
                seqLen = longestArithSequence[i][diff] = max(longestArithSequence[i][diff], longestArithSequence[j][diff] + 1)
                
                largestLen = max(largestLen, seqLen)
                
        
        return largestLen
                
        

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        dp = {}
        N = len(A)
        for i in range(N):
            for j in range(i+1, N):
                dp[(j, A[j]-A[i])] = dp.get((i, A[j]-A[i]), 1)+1
                
        return max(dp.values())
from collections import defaultdict
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        d = {}
        mval = -1
        for i, n in enumerate(A):
            for j in range(i):
                if (j, n-A[j]) not in d:
                    d[(i, n-A[j])] = 2
                else:
                    d[(i, n-A[j])] = d[(j, n-A[j])] + 1
                mval = max(mval, d[(i, n-A[j])])
        return mval

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        res = 0
        dp = collections.defaultdict(lambda: 1)
        
        for i in range(len(A)-1, -1, -1):
            for j in range(i+1, len(A)):
                dist = A[j] - A[i]
                dp[(i, dist)] = max(dp[(i, dist)], dp[(j, dist)] + 1)
        
        return max(dp.values())
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        table = [collections.defaultdict(int) for _ in range(len(A))]
        ans = 0
        for i in range(len(A)):
            for j in range(i):
                if A[i] - A[j] in table[j]:
                    currLen = table[j][A[i] - A[j]] + 1
                else:
                    currLen = 2
                
                ans = max(ans, currLen)
                table[i][A[i]-A[j]] = max(table[i][A[i]-A[j]], currLen)
        # print(table)
        return ans
                    

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        
        
        if len(A) < 3:
            return len(A)
        
        sub_lens = [{} for i in A]
        
        max_len = 0
        
        for i in range(0, len(A)):
            for j in range(0, i):
                diff = A[i] - A[j]
                sub_lens[i][diff] = max(sub_lens[i].get(diff, 1), sub_lens[j].get(diff, 1) + 1)
                max_len = max(max_len, sub_lens[i][diff])
                
        return max_len
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        dp = collections.defaultdict(int)
        for i in range(len(A)):
            for j in range(i):
                dp[(i, A[i] - A[j])] = dp[(j, A[i] - A[j])] + 1
        return max(dp.values())+1

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        
        memo = {}
        
        for i in range(len(A)):
            for j in range(i+1, len(A)):
                if (i, A[j] - A[i]) in memo:
                    memo[j, A[j] - A[i]] = memo[i, A[j]-A[i]] + 1
                else:
                    memo[j, A[j] - A[i]] = 2
                   
        return max(memo.values())
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        dp = collections.defaultdict(lambda: collections.defaultdict(int))
        
        maxSoFar = 0
        
        for curr in range(len(A)):
            for prev in range(curr):
                difference = A[curr] - A[prev]
                dp[curr][difference] = max(dp[curr][difference], dp[prev][difference] + 1)
                maxSoFar = max(maxSoFar, dp[curr][difference])
        
        return maxSoFar + 1

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        mem = [collections.defaultdict(int) for _ in A]
        res = 1
        for i in range(len(A)):
            for j in range(i+1, len(A)):
                v = A[j] - A[i]
                mem[j][v]=max(mem[i][v] + 1, mem[j][v])
                res = max(res, mem[j][v])
        return res + 1
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        d = [collections.defaultdict(int) for _ in A]
        res = 1
        for i in range(0,len(A)):
            for j in range(i):
                v = A[i]-A[j]
                d[i][v]=max(d[j][v]+1,d[i][v])
                res = max(d[i][v],res)
        return res+1

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        dp=[]
        m=0
        for i,a in enumerate(A):        
            dp.append(defaultdict(int))
            for j in range(i):
                dif=a-A[j]
                dp[i][dif]=dp[j][dif]+1
                m=max(m,dp[i][dif])
        return m+1
          

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        N = len(A)
        dp = [collections.defaultdict(int) for _ in A]
        res = 1
        for i in range(N):
            for j in range(i):
                diff = A[i] - A[j]
                if dp[i][diff] == None: dp[i][diff] = 0
                if dp[j][diff] == None: dp[j][diff] = 0
                
                if dp[j][diff] == 0:
                    dp[i][diff] = 2
                else:
                    dp[i][diff] = dp[j][diff] + 1
                
                res = max(res, dp[i][diff])
        return res
from collections import defaultdict

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        n = len(A)
        # lengths of AM seqs after i with a diff d
        mem = [defaultdict(int) for _ in range(n)]
        res = 0
        for i in reversed(range(n)):
            for j in range(i + 1, n):
                d = A[j] - A[i]
                mem[i][d] = max(1 + mem[j].get(d, 0), mem[i][d])
                res = max(mem[i][d], res)
        return res + 1
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        if len(A) < 2:
            return 0
        la = len(A)
        dp = {}
        curr = 0
        for i in range(1, la):
            for j in range(i):
                d = A[i] - A[j]
                dp[(i, d)] = dp.get((j, d), 1) + 1
                if dp[(i,d)] > curr:
                    curr = dp[(i,d)]

        return curr
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        d = [collections.defaultdict(int) for _ in A]
        res = 1
        for i in range(0,len(A)):
            for j in range(i):
                v = A[i]-A[j]
                d[i][v]=max(d[j][v]+1,d[i][v])
                res = max(d[i][v],res)
        return res+1
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        memo = {}
        for j, y in enumerate(A):
            for i, x in enumerate(A[:j]):
                d = y - x
                memo[d, j] = memo.setdefault((d, i), 1) + 1
        return max(memo.values())
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        N = len(A)
        dp = [collections.defaultdict(int) for _ in range(N)]
        res = 0
        for i in range(1, N):
            for j in range(i):
                delta = A[i] - A[j]
                dp[i][delta] = max(dp[i][delta], dp[j][delta] + 1, 2)
                res = max(res, dp[i][delta])
        return res

class Solution:
    def longestArithSeqLength(self, A):
        d = [collections.defaultdict(int) for _ in A]
        res = 1
        for i in range(0,len(A)):
            for j in range(i):
                v = A[i]-A[j]
                d[i][v]=max(d[j][v]+1,d[i][v])
                res = max(d[i][v],res)
        return res+1
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        dp = {}
        if len(A) == 0:
            return 0
        if len(A) == 1:
            return 1
        for i in range(1, len(A)):
            for j in range(0, i):
                diff = A[i] - A[j]
                
                if (j, diff) in dp:
                    dp[(i, diff)] = 1 + dp[(j, diff)]
                else:
                    dp[i, diff] = 2
        return max(dp.values())
from collections import defaultdict

class Solution:
    # @param A : tuple of integers
    # @return an integer
    def longestArithSeqLength(self, A: List[int]) -> int:
        sol = defaultdict(int)
        
        n = len(A)
        for i in range(n):
            for j in range(i):
                d = A[i] - A[j]
                sol[(i, d)] = max(sol[(i, d)], 1 + sol[(j, d)])
                
        return max(list(sol.values()), default=0) + 1
    
#         rows = len(A)
#         cols = len(A)
#         dp = [[2 for c in range(cols)] for r in range(rows)]
        
#         for c in range(cols): 
#             for r in range(0, c):
                
                
#         for c in range(cols): 
#             for r in range(0, c):
#                 diff = A[c] - A[r]             
#                 x = A[r] - diff 
#                 # search for k such that dp[k][r] A[k]=x
#                 for k in reversed(range(0, r)): # put this in doc
#                     if(A[k] == x):
#                         dp[r][c] = max(dp[k][r] + 1, dp[r][c])
#                         break
#         max_so_far = dp[0][0]
        
#         for c in range(cols): 
#             for r in range(0, c):
#                 max_so_far = max(max_so_far, dp[r][c])
        
#         return max_so_far
                
                

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        N = len(A)
        dp = defaultdict(int)
        for j in range(N):
            for i in range(j):
                t = A[j] - A[i]
                dp[j, t] = max(dp[j, t], dp[i, t] + 1)
        return max(dp.values()) + 1
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        #u8fd9u9898bottom upu5199u8d77u6765u5feb?
        dp = defaultdict(lambda: 1) #default u662f 1uff0c u81eau5df1u80afu5b9au662fu81eau5df1u7684seq
        for i in range(len(A)):
            for j in range(i): #u4e4bu524du7684
                diff = A[i] - A[j]
                dp[(i, diff)] = max(dp[(i, diff)], 1+ dp[(j, diff)])
                
        return max(dp.values())
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        if not A: return 0
        dp = [{} for _ in range(len(A))]
        res = 0
        for j in range(1, len(A)):
            for i in range(j):
                if A[j]-A[i] in dp[i]:
                    dp[j][A[j]-A[i]] = dp[i][A[j]-A[i]]+1
                else:
                    dp[j][A[j]-A[i]] = 1
                res = max(dp[j][A[j]-A[i]], res)
        return res+1

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        if not A:
            return 0
        res = 0
        records = [collections.defaultdict(int) for _ in range(len(A))]
        for i in range(len(A)):
            for j in range(i):
                diff = A[i] - A[j]
                current = records[i].get(diff, 0)
                prev = records[j].get(diff, 0) + 1
                records[i][diff] = max(prev, current, 2)
                res = max(res, records[i][diff])
        return res
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        n = len(A)
        dp = [collections.defaultdict(lambda: 1) for _ in range(n)]
        ans = 0
        for i in range(1, n):
            for j in range(0, i):
                diff = A[i] - A[j]
                dp[i][diff] = max(dp[i][diff], dp[j][diff] + 1)
                ans = max(ans, dp[i][diff])
        return ans
                

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        # dp = {}
        # n = len(A)
        # for i in range(1, n):
        #     for j in range(i):
        #         d = A[i] - A[j]
        #         if (j, d) in dp:
        #             dp[i, d] = dp[j, d] + 1
        #         else:
        #             dp[i, d] = 2
        # return max(dp.values())
        d = [collections.defaultdict(int) for _ in A]
        res = 1
        n = len(A)
        for i in range(n):
            for j in range(i):
                v = A[i]-A[j]
                d[i][v]=max(d[j][v]+1,d[i][v])
                res = max(d[i][v],res)
        return res+1
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        dp = {}
        ans = 1
        
        for x in range(len(A)):
            for y in range(x + 1, len(A)):
                dp[y, A[y] - A[x]] = dp.get((x, A[y] - A[x]), 1) + 1
                ans = max(dp[y, A[y] - A[x]], ans)
                
        return ans

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        length = len(A)
        onright = [0 for _ in range(501)]
        toextend = [{} for _ in range(501)]
        res = 2
        for v in A:
            onright[v] += 1
        for i in range(0, length):
            val = A[i]
            tex = toextend[val]
            onright[val] -= 1
            doneself = False
            for lval in A[:i]:
                diff = val - lval
                nextval = val + diff
                if nextval == val:
                    if doneself:
                        continue
                    doneself = True
                if nextval > 500 or nextval < 0 or onright[nextval] == 0:
                    continue
                if diff in tex:
                    c = tex[diff] + 1
                else:
                    c = 3
                if c > res:
                    res = c
                toextend[nextval][diff] = c
        return res
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        if len(A)<=2:
            return len(A)
            
        seen = {}
        ans = 0
        for i in range(len(A)-1):
            for j in range(i+1,len(A)):
                seen[j, A[j]-A[i]] = seen.get((i, A[j]-A[i]), 0) + 1
                ans = max(ans, seen[j, A[j]-A[i]])
        
        return ans+1
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
#         d = [collections.defaultdict(lambda:0) for _ in A]
#         res = 1
#         for i in range(len(A)):
#             for j in range(i):
#                 v = A[i]-A[j]
#                 d[i][v]=max(d[j][v]+1,d[i][v])
#                 res = max(d[i][v],res)
#         return res+1
    
        d = [collections.defaultdict(lambda:0) for _ in A]
        res = 1
        for i in range(len(A)):
            for j in range(i+1,len(A)):
                v = A[i]-A[j]
                d[j][v]=max(d[i][v]+1,d[j][v])
                res = max(d[j][v],res)
        return res+1
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        ans = 0
        dp = {}
        for i in range(len(A)):
            for j in range(i):
                diff = A[i] - A[j]
                j1 = (j , diff)
                i1 = (i, diff)
                if dp.get(j1) != None:
                    dp[i1] = 1 + dp[j1]
                    ans = max(ans, dp[i1])
                else:
                    dp[i1] = 2
                    ans = max(ans, dp[i1])

        return ans
            
                    
                


class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        def helper():
            return 1

        dp = defaultdict(helper)
        for i in range(len(A)):
            for j in range(i):
                step = A[i] - A[j]
                dp[(i, step)] = max(dp[(j, step)] + 1, dp[(i, step)])
        return max(dp.values())

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        dp = collections.defaultdict(int)
        n = len(A)
        for i in range(n):
            for j in range(i + 1, n):
                dp[j, A[j] - A[i]] = dp.get((i, A[j] - A[i]), 1) + 1
        return max(dp.values())

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        # longest length with end index i and diff j
        if not A:
            return 0
        
        dp = {}
        n = len(A)
        
        max_len = 0
        
        for j in range(1, n):
            for i in range(0, j):
                if (i, A[j] - A[i]) in dp:
                    dp[(j, A[j] - A[i])] = dp[(i, A[j] - A[i])] + 1
                else:
                    dp[(j, A[j] - A[i])] = 2
                
                max_len = max(max_len, dp[(j, A[j] - A[i])])
                
        return max_len
                
        
                    
                

class Solution:
    def longestArithSeqLength(self, A):
        dp = collections.defaultdict(int)
        for i in range(len(A)):
            for j in range(i):
                diff = A[i] - A[j]
                dp[(i, diff)] = max(2, dp[(i, diff)], dp[(j, diff)]+1)
        return max(dp.values())

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        dp = {}
        for i, v in enumerate(A):
            for j in range(i):
                diff = v - A[j]
                dp.setdefault((diff, i), 1)
                dp[diff, i] = max(dp[diff, i], dp.get((diff, j), 1) + 1)
        return max(dp.values(), default=0)
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        dp = [{} for _ in range(len(A))]
        max_len = 0
        for i in range(len(A)-1):
            
            for j in range(i+1, len(A)):
                diff = A[j] - A[i]
                
                if dp[i].get(diff) == None:
                    dp[j][diff] = 2
                else:
                    dp[j][diff] = dp[i][diff] + 1
                max_len = max(dp[j][diff], max_len)
                
        return max_len
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        dp = []
        maximum = 1
        
        for i in range(len(A)):
            d={}
            dp.append(d)
            for j in range(i):
                dp[i][A[i]-A[j]] = dp[j][A[i]-A[j]] = 0
                
        for i in range(len(A)):
            d = {}
            dp.append({})
            for j in range(i):
                diff = A[i]-A[j]
                
                dp[i][diff] = dp[j][diff]+1
                maximum = max(maximum,dp[i][diff])
        
        return maximum+1
from collections import defaultdict
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        n = len(A)
        dp = [defaultdict(lambda: 1) for _ in range(n)]
        ans = 2
        for i in range(n):
            for j in range(i):
                dp[i][A[i]-A[j]]=max(dp[j][A[i]-A[j]]+1,2)
                ans = max(ans,dp[i][A[i]-A[j]])
        return ans
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        n = len(A)
        dp = {}
        mostFreq = -1
        for i in range(1,n):
            for j in range(i-1, -1,-1):
                diff = A[i] - A[j]
                prevVal = 1
                if (i,diff) in dp: prevVal = dp[ (i,diff)]
                if (j,diff) in dp: 
                    dp[(i,diff)] = max(dp[(j,diff)], prevVal-1) + 1
                else:
                    if (i,diff) not in dp:
                        dp[(i,diff)] = 1
                mostFreq = max(dp[(i,diff)], mostFreq)
        # ret = -1
        # for k,v in dp.items():
        #     ret = max(ret,v)
        # return ret+1
        return mostFreq+1
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        n = len(A)
        dp = {}
        mostFreq = -1
        for i in range(1,n):
            for j in range(i-1, -1,-1):
                diff = A[i] - A[j]
                prevVal = 1
                if (i,diff) in dp: prevVal = dp[ (i,diff)]
                if (j,diff) in dp: 
                    dp[(i,diff)] = max(dp[(j,diff)], prevVal-1) + 1
                else:
                    if (i,diff) not in dp:
                        dp[(i,diff)] = 1
                mostFreq = max(dp[(i,diff)], mostFreq)
        return mostFreq+1
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        from collections import defaultdict
        opt = defaultdict(lambda : defaultdict(int))
        l = len(A)
        
        sol = 0
        # print(opt[-4])
        
        for i in range(l): # 6
            # print(opt[-17])
            for j in range(i + 1, l): # 8
                diff = A[j] - A[i] # 0
                
                sub_l = 2                
                if diff in opt[i]:
                    sub_l = opt[i][diff] + 1
                
                opt[j][diff] = max(opt[j][diff], sub_l)
                sol = max(sol, opt[j][diff])
                # if opt[A[i]][diff] == 7:
                #     print(i, A[i], diff)
        # for i, row in enumerate(opt):
        #      print(i, row)
        
        # print(dict(opt))
        # print(opt[-4])
        # for k, v in opt.items():
        #     print(k, v) 
            # if k < 0:                
            #     print(k, v) 
        #     pass
        return sol
                        

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
            
        dp = {}
        for i in range(len(A)):
            for j in range(i + 1, len(A)):
                dp[j, A[j] - A[i]] = dp.get((i, A[j] - A[i]), 1) + 1
                
        return max(dp.values())
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        # dp[i][k] = longestArithSeqLength(A[:i+1]) with step size k
        largest = max(A)
        smallest = min(A)
    
        dp = dict()
        res = 0
    
        for i in range(1, len(A)):
            for prev_i in range(i):
                step = A[i] - A[prev_i]
                prev_step = dp[(prev_i, step)] if (prev_i, step) in dp else 1
                if (i, step) not in dp:
                    dp[(i, step)] = prev_step + 1
                else:
                    dp[(i, step)] = max(prev_step + 1, dp[(i, step)])
                
                res = max(res, dp[(i, step)])
        
        return res
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        from collections import Counter
        cnt = Counter()
        cnt
        arith = [Counter() for i in range(len(A))]
        for i in range(len(A)-2, -1,-1):
            for j in range(i+1, len(A)):
                diff = A[j]-A[i]
                arith[i][diff] = max(1 + arith[j][diff], arith[i][diff])
        #print(arith)
        longest = 0
        for i in range(len(A)):
            #print(arith[i])
            most_common = arith[i].most_common()
            
            longest = max(most_common[0][1] if most_common else 0, longest)
        return longest + 1
        # for i in range(len(A)):
        #     for j in range(i+1, len(A)):
        #         cnt[A[j]-A[i]] += 1
        #     print(A[i], cnt)
        # print(cnt)
        # val = cnt.most_common()[0][1]
        # return val + 1 
            
        
#         self.arith = [dict() for i in range(len(A))]
        
#         def helper(i, diff):
#             if diff in self.arith[i]:
#                 return self.arith[i][diff]
            
#             val = 0
#             for j in range(i+1, len(A)):
#                 if A[j] - A[i] == diff:
#                     val = 1 + helper(j, diff)
#                     break
#             self.arith[i][diff] = val        
#             return self.arith[i][diff]
        

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        n = len(A)
        dp = collections.defaultdict(dict)
        max_val = 0
        for i in range(n):
            for j in range(i):
                dif = A[i] - A[j]
                dp[dif].setdefault(i, 0)
                dp[dif][i] = max(dp[dif][i], dp[dif].get(j,0) + 1)
                max_val = max(dp[dif][i], max_val)
        #print(dp)
        return max_val + 1
class Solution:
    def longestArithSeqLength(self, A):
        dp = dict()
        for endi, endv in enumerate(A[1:], start = 1):
            for starti, startv in enumerate(A[:endi]):
                diff = endv - startv
                if (starti, diff) in dp:
                    dp[(endi, diff)] = dp[(starti, diff)] + 1
                else:
                    dp[(endi, diff)] = 2
        return max(dp.values())
    
class Solution:
    def longestArithSeqLength(self, A):
        dp = dict()
        for starti, startv in enumerate(A):
            for endi, endv in enumerate(A[starti+1:], start = starti+1):
                diff = endv - startv
                if (starti, diff) in dp:
                    dp[(endi, diff)] = dp[(starti, diff)] + 1
                else:
                    dp[(endi, diff)] = 2
        return max(dp.values())
    
class Solution:
    def longestArithSeqLength(self, A):
        N = len(A)
        dp = [{0:1} for _ in range(N)]
        for end in range(1, N):
            for start in range(end):
                diff = A[end] - A[start]
                if diff in dp[start]:
                    dp[end][diff] = dp[start][diff] + 1
                else:
                    dp[end][diff] = 2
        return max(max(dp[end].values()) for end in range(1, N))
    
class Solution:
    def longestArithSeqLength(self, A):
        ans = 2
        n = len(A)
        index = {}
        dp = [[2] * n for i in range(n)]
        
        for i in range(n-1):
            for j in range(i+1, n):
                first = A[i] * 2 - A[j]
                if first in index:
                    dp[i][j] = dp[index[first]][i] + 1
                    #ans = max(ans, dp[i][j])
                    if dp[i][j] > ans: ans = dp[i][j]
            index[A[i]] = i
        return ans

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        #{#next num: cur length}
        dp = [{} for a in A]
        imax = 1
        for i in range(len(A)):
            for j in range(i-1,-1,-1):
                diff = A[i]-A[j]
                prev = dp[i].get(diff, 1)
                dp[i][diff] = max(dp[j].get(diff, 1)+1, prev)
                imax = max(imax, dp[i][diff])
        return imax

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        if len(A) <= 1:
            return len(A)
        
#         memo = [(1 + 2 * 500) * [1] for _ in range(1 + len(A))]
        
#         res = 0
        
#         for i in range(len(A)-2, -1, -1):
#             for j in range(i+1, len(A)):
#                 diff = A[j] - A[i] + 500
#                 memo[i][diff] = max(memo[i][diff], memo[j][diff] + 1)
#                 res = max(res, memo[i][diff])
        
        h = dict()
        
        res = 0
        
        for i in range(1, len(A)):
            for j in range(i):
                diff = A[j] - A[i]
                h.setdefault((j, diff), 1)
                h[(i, diff)] = h[(j, diff)] + 1
                res = max(res, h[(i, diff)])
    
        return res
from collections import defaultdict
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        
        dp=dict()
        mlen=0
        for i in range(len(A)):
            dp[A[i]]=defaultdict(lambda :1)
            for j in range (i-1,-1,-1):
                
                d=A[i]-A[j]
                if dp[A[i]][d]<=dp[A[j]][d]+1:
                    dp[A[i]][d]=dp[A[j]][d]+1
                    mlen=max(mlen,dp[A[i]][d])
                    
        return mlen

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        #diffs = set(y - x for i, x in enumerate(A) for y in A[i+1:])
        ans = 0
        for diff in range(-500, 501):
            data = {}
            for num in A:
                if num - diff not in data:
                    if num not in data:
                        data[num] = [num]
                    continue
                if len(data[num - diff]) < len(data.get(num, [])):
                    continue
                seq = data.pop(num - diff)
                seq.append(num)
                ans = max(ans, len(seq))
                data[num] = seq
        return ans
import itertools

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        d = {}
        for i in range(len(A)):
            for j in range(i+1,len(A)):
                check = (i,A[j]-A[i])
                if check in d:
                    d[(j,A[j]-A[i])] = d[check]+[A[j]]
                else:
                    d[(j,A[j]-A[i])] = [A[i],A[j]]
        return len(max([i for i in list(d.values())],key=len))
                
                

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:    
        dp = [{} for i in range(len(A))]
        res = 0
        for i in range(1, len(A)):
            for j in range(i):
                dp[i][A[i]-A[j]] = max(dp[i].get(A[i]-A[j], 0), dp[j].get(A[i]-A[j], 0) + 1)
                res = max(res, dp[i][A[i]-A[j]])
        return res + 1
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        n = len(A)
        dp = [{} for _ in range(n)]
        ans = 2
        for i in range(1, n):
            for j in range(i):
                key_i = (A[i], A[i] - A[j])
                key_j = (A[j], A[i] - A[j])
                if key_i not in dp[i]:
                    dp[i][key_i] = 2
                if key_j in dp[j]:
                    dp[i][key_i] = max(dp[i][key_i], dp[j][key_j] + 1)
                ans = max(dp[i][key_i], ans)
        return ans
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        M=max(A)+1
        dp=[]
        for i in range(len(A)):
            temp=[1]*(M*2)
            dp.append(temp)
        for i in range(len(A)):
            
            for j in range(i):
                delta=A[i]-A[j]
                k=delta+M
            #    print(k,M)
                dp[i][k]=dp[j][k]+1
        m=0
        for i in range(2*M):
            for j in range(len(A)):
                m=max(dp[j][i],m)
      #  print(dp)
        return m
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        res = 2
        n = len(A)
        dp = [collections.defaultdict(int) for _ in range(n)]
        for i in range(1, n):
            for j in range(i):
                d = A[j] - A[i]
                dp[i][d] = dp[j][d] + 1
                res = max(res, dp[i][d]+1)
        return res
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        seen = {}
        res = 0
        for i in range(len(A)):
            for j in range(i-1, -1, -1):
                diff = A[i] - A[j]
                if diff not in seen:
                    seen[diff] = {}
                old_val = seen[diff][i] if i in seen[diff] else 0
                if j not in seen[diff]:
                    seen[diff][i] = 2
                else:
                    seen[diff][i] = seen[diff][j] + 1
                seen[diff][i] = max(old_val, seen[diff][i])
                if seen[diff][i] > res:
                    res = seen[diff][i]
        return res
from collections import defaultdict
class Solution:
    def find(self, A):
        dp = []
        ans = 1
        for _ in A: dp.append({})
        for i, ai in enumerate(A):
            for j in range(i):
                aj = A[j]
                d = ai - aj
                if d < 0: continue
                if d not in dp[i]: dp[i][d] = 1
                if d not in dp[j]: dp[j][d] = 1
                temp = max(dp[i][d], dp[j][d] + 1)
                dp[i][d] = temp
                ans = max(ans, temp)
        return ans
    def longestArithSeqLength(self, A: List[int]) -> int:
        ans = self.find(A)
        ans = max(ans, self.find(A[::-1]))
        return ans;
        

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        ans = 0
        #A = sorted(A)
        for diff in range(-500,501):
            dp = defaultdict(int)
            for e in A:
                dp[e] = dp[e-diff]+1
                ans = max(ans,dp[e])
            
        return ans
                    

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        
#         dp = {} 
#         res = -1
#         for i in range(0, len(A)):
#             for j in range(i-1, -1, -1):
#                 d = A[i] - A[j]
#                 dp[(i, d)] = max(dp.get((i, d), -1), dp.get((j, d), 1) + 1)
#                 res = max(res, dp[(i, d)])
            
#         return res
        dp = [[-1 for i in range(1001)] for j in range(len(A))]
        res = -1
        for i in range(0, len(A)):
            for j in range(i-1, -1, -1):
                d = A[i] - A[j] + 500
                if dp[j][d] == -1: dp[j][d] = 1
                dp[i][d] = max(dp[i][d], dp[j][d] + 1)
                res = max(res, dp[i][d])
            
        return res
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        dp = {}
        hm = {A[0]:0}
        for i in range(1, len(A)):
            if A[i] not in hm:
                hm[A[i]] = i
            for j in range(i):
                diff = A[i]-A[j]
                # print(i, j, diff, dp, hm)
                if (A[j]-diff) in hm:
                    dp[(i, diff)] = max(dp.get((i, diff), 2), dp.get((j, diff), 1)+1)
                else:
                    dp[(i, diff)] = max(dp.get((i, diff), 2), 2)
        
        # print(dp)
        return max(dp.values())
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:    
        _max, _min = 0, 0
        
        for ele in A:
            _max = max(_max, ele)
            _min = max(_min, ele)
            
        diff = _min-_max
        
        dp = [ {} for i in range(len(A))]
        res = 0
        for i in range(1, len(A)):
            for j in range(i):
                dp[i][A[i]-A[j]] = max(dp[i].get(A[i]-A[j], 0), dp[j].get(A[i]-A[j], 0) + 1)
                res = max(res, dp[i][A[i]-A[j]])
        return res + 1
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
     
        f = collections.defaultdict(int)
        fff = collections.defaultdict(int)
        maxlen = 0
        for i in range(len(A)):
            for j in range(i):
                diff = A[i] - A[j]
                #fff[(A[i], diff)] = max(fff[(A[i], diff)], fff.get((A[j], diff), 1) + 1)
                f[(i, diff)] = max(f[(i, diff)], f.get((j, diff), 1) + 1)
                '''
                if (j, diff) not in f:
                    f[(i, diff)] = 2
                else:
                    f[(i, diff)] = max(f[(i, diff)],  f[(j, diff)] + 1)                
                '''                    
                maxlen = max(maxlen, f[(i, diff)])

        return maxlen

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        dp = {}
        globalMax = 1
        for i, a1 in enumerate(A):
            for j, a2 in enumerate(A[:i]):
                x = a1 - a2
                if (j,x) in dp:
                    dp[(i,x)] = dp[(j,x)] + 1
                else:
                    dp[(i,x)] = 2
                globalMax = max(globalMax, dp[(i,x)])
        return globalMax
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        dp = {}
        N = len(A)
        mx = 0
        for i, n in enumerate(A):
            for j in range(i+1, N):
                b = A[j] - n
                if (i, b) in dp:
                    dp[j, b] = dp[i, b] + 1
                else:
                    dp[j, b] = 2
                
                mx = max(mx, dp[j, b])
                    
        return mx
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        res = 0
        dp = {i:collections.defaultdict(lambda: 1) for i in range(len(A))}
        for i in range(1, len(A)):
            for j in range(i):
                diff = A[i]-A[j]
                dp[i][diff] = max(dp[i][diff], dp[j][diff] + 1)
                res = max(res, dp[i][diff])
        return res
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        # dp[i][k] = longestArithSeqLength(A[:i+1]) with step size k
        dp = dict()
        res = 0
    
        for i in range(1, len(A)):
            for prev_i in range(i):
                step = A[i] - A[prev_i]
                prev_step = dp[(prev_i, step)] if (prev_i, step) in dp else 1
                dp[(i, step)] = prev_step + 1
                
                res = max(res, dp[(i, step)])
        
        return res
from collections import defaultdict
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        def calc(A):
            memo = [dict() for _ in range(len(A))]
            ret = 1
            for i in range(len(A) - 1, -1, -1):
                for j in range(i + 1, len(A)):
                    if A[j] < A[i]:
                        continue
                    diff = A[j] - A[i]
                    memo[i][diff] = max(memo[i].get(diff, 0), memo[j].get(diff, 1) + 1)
                    ret = max(ret, memo[i][diff])
            return ret
        
        return max(
            calc(A), calc(list(reversed(A)))
        )
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        res = 0
        dp = [collections.defaultdict(lambda: 1) for _ in range(len(A))]
        for i in range(len(A)):
            for k in range(i-1, -1, -1):
                dp[i][A[i] - A[k]] = max(dp[i][A[i]-A[k]], dp[k][A[i]-A[k]] + 1) # remember the MAX here!!!
                res = max(res, dp[i][A[i]-A[k]])
        return res

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        dp = {}
        length = len(A)
        
        for i in range(length):
            for j in range(i + 1, length):
                dp[j, A[j] - A[i]] = dp.get((i, A[j] - A[i]), 1) + 1
        
        return max(dp.values())
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        res = 0
        dp = [collections.defaultdict(lambda: 1) for _ in range(len(A))]
        for i in range(len(A)):
            for k in range(i-1, -1, -1):
                dp[i][A[i] - A[k]] = max(dp[i][A[i]-A[k]], dp[k][A[i]-A[k]] + 1)
                res = max(res, dp[i][A[i]-A[k]])
        return res

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        if len(A) < 2:
            return 0
        la = len(A)
        dp = {}
        curr = 0
        for i in range(1, la):
            for j in range(i):
                d = A[i] - A[j]
                dp[(i, d)] = dp.get((j, d), 1) + 1
                curr = max(curr, dp.get((i,d), 1))

        return curr
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        n = len(A)
        if n == 2:
            return n
        dp = [{} for i in range(n)]
        max_len = 0
        for i in range(1,n):
            for j in range(i):
                diff = A[i]-A[j]
                if diff in dp[j]:
                    dp[i][diff] = max(2, 1+dp[j][diff])
                else:
                    dp[i][diff] = 2
                max_len = max(max_len, dp[i][diff])
        return max_len
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        n = len(A)
        
        table = [ defaultdict(int)  for i in range(n) ]
        out = 0
        curr = A[1]-A[0]
        full = True
        for i in range(2,n):
            if curr != A[i]-A[i-1]:
                full = False
                break
        if full: return n
        
        # print(table)f
        for i in range(n):
            for j in range(0,i):
                diff = A[i]-A[j]
                if table[j][diff] == 0:
                    table[j][diff] = 1
                    
                table[i][diff] = max(table[i][diff],table[j][diff] + 1)
                
                out = max(table[i][diff], out)
        
        # for i in range(n):
        #     print(A[i], table[i])
        return out

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        dp = defaultdict(lambda:1)
        
        n = len(A)
        for i in range(n):
            for j in range(i):
                dp[i, A[i] - A[j]] = max(dp[i, A[i] - A[j]], dp[j, A[i] - A[j]] + 1)
        
        return max(dp.values())
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        d = [collections.Counter() for _ in A]
        res = 1
        for i in range(0,len(A)):
            for j in range(i):
                v = A[i]-A[j]
                d[i][v]=max(d[j][v]+1,d[i][v])
                res = max(d[i][v],res)
        return res+1

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        d = {}
        mx = 0
        for i in range(len(A)):
            for j in range(i):
                diff = A[i] - A[j]
                if (diff, i) not in d:
                    d[(diff, i)] = 2
                if (diff, j) in d:
                    d[(diff, i)] = max(d[(diff,i)], d[(diff, j)] + 1)
                mx = max(mx, d[(diff, i)])
        return mx
                    
        
                    
                
        
        
                

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        
        dp = collections.defaultdict(dict)
        ans = 0
        
        for i in range(len(A)):
            for j in range(i):
                diff = A[i] - A[j]
                dp[i][diff] = dp[i].get(diff, 0)
                dp[i][diff]  = max(dp[i][diff], dp[j].get(diff,0)+1)
                # if dp[i][diff]==6: print(A[i],A[j], diff)
                ans = max(ans, dp[i][diff])
        return ans+1
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        if len(A) < 3:
            return len(A)

        n = len(A)

        diff_c = collections.Counter()

        for i in range(n):
            for j in range(i):
                if diff_c[(j, A[i]-A[j])] == 0:
                    diff_c[(i, A[i]-A[j])] = 2
                else:
                    diff_c[(i, A[i] - A[j])] = diff_c[(j, A[i]-A[j])] + 1


        return max(diff_c.values())

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        #{#next num: cur length}
        dp = {}
        imax = 1
        for i in range(len(A)):
            for j in range(i-1,-1,-1):
                diff = A[i]-A[j]
                prev = dp.get((diff, i), 1)
                saved = dp[(diff, i)] = max(dp.get((diff, j),1)+1, prev)
                imax = max(imax, saved)
        return imax

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        n = len(A)
        sol = 1
        dp = {}
        for j in range(1,n):
            for i in range(j):
                diff = A[j] - A[i]
                dp[j,diff] = max(dp.get((j,diff), 2), dp.get((i,diff),1)+1)
                sol = max(sol, dp[j,diff])
                
        return sol
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        pos = defaultdict(int)
        best = 1
        for i in range(len(A) - 1, -1, -1):
            for j in range(i + 1, len(A)):
                diff = A[j] - A[i]
                if (j, diff) in pos:
                    pos[(i, diff)] = max(pos[(i, diff)], pos[(j, diff)] + 1)
                else:
                    pos[(i, diff)] = max(pos[(i, diff)], 2)
                best = max(best, pos[(i, diff)])
        return best
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        
        
        minimum = min(A)
        maximum = max(A)
        
        length = 2*(maximum-minimum)+1
        
        dp = [[1 for i in range(length)] for j in range(len(A))]
        
        diff = maximum-minimum
        ans = 0
        for i in range(len(dp)):
            for j in range(i):
                dp[i][A[i]-A[j]+diff]=max(dp[i][A[i]-A[j]+diff],dp[j][A[i]-A[j]+diff]+1)
                ans = max(ans,dp[i][A[i]-A[j]+diff])
        return ans 
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        
        if len(A)==0:
            return 0
        
        maxx=1
        arSubCounts = dict()
        
        for i in range(len(A)):
            for j in range(i+1,len(A)):
                diff = A[j]-A[i]
                arSubCounts[(j,diff)]=max(arSubCounts.get((j,diff),1),arSubCounts.get((i,diff),1)+1)
                maxx = max(arSubCounts[(j,diff)],maxx)
                
        return maxx

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        if not A:
            return 0
        seen = {}
        longest = 1
        for i in range(len(A)):
            for j in range(i+1, len(A)):
                diff = A[j]-A[i]
                if (i, diff) not in seen:
                    seen[(i, diff)] = 1
                    
                if (j, diff) not in seen:
                    seen[(j,diff)] = 1
                
                seen[(j,diff)] = max(seen[(j,diff)], seen[(i,diff)]+1)
        
        return max([v for k,v in seen.items()])
import collections
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        dp = {}
        for i in range(len(A)):
            for j in range(i + 1, len(A)):
                dp[j, A[j] - A[i]] = dp.get((i, A[j] - A[i]), 1) + 1
        return max(dp.values())

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        n = len(A)
        d = {}
        res = 0
        for i in range(n):
            d[i] = {}
            for j in range(i):
                diff = A[i] - A[j]
                if diff not in d[j]:
                    d[i][diff] = 2
                else:
                    d[i][diff] = d[j][diff] + 1
                res = max(res, d[i][diff])
        return res
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        dp = collections.defaultdict(int)
        ans = 0
        
        for i in range(len(A)):
            for j in range(i):
                diff = A[i] - A[j]
                dp[i, diff] = max(2, dp[j, diff] + 1)
                ans = max(ans, dp[i, diff])
        
        return ans
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        
        
        dp = {}
        ans = 0
        
        for i in range(len(A)):
            for j in range(i+1, len(A)):
                diff = A[j] - A[i]
                if (i, diff) not in dp: 
                    dp[(j, diff)] = 2 
                else:
                    dp[(j, diff)] = dp[(i, diff)] + 1
                ans = max(ans, dp[(j, diff)])
                    
        return ans

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        
        
        # if there's no A, 
        if not A:
            return 0
        
        # create a dp array,
        # key would be a tuple of indices, 
        dp = {}
        
        # iterate over the length of A, 
        for it1 in range (1, len (A)):
            
            # iterate upto length of it1, 
            for it2 in range (it1):
                
                # create a tuple by difference, 
                key_tuple = (it2, A[it1] - A[it2])
                
                # if key_tuple doesn't exist, 
                if key_tuple not in dp:
                    dp[(key_tuple)] = 1
                    
                dp [(it1, A[it1] - A[it2])] =  dp[(key_tuple)] + 1
        #print (dp)  
        return max (dp.values())
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        dp = [collections.defaultdict(int) for _ in range(len(A))]
        res = 0
        for i in range(1, len(A)):
            for j in range(i):
                dp[i][A[i]-A[j]] = max(dp[i][A[i]-A[j]], dp[j].get(A[i]-A[j], 1)+1)
                res = max(res, dp[i][A[i]-A[j]])
        return res
'''

[9,4,7,2,10]

0, 



[3,6,9,12]

0 -> 1

3 -> 2

3 -> 3

6 -> 1



'''

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        
        # if there's no list
        if not A:
            return 0
        
        if len(A) == 1:
            return 1
        
        diff_map = {}
        
        # iterate over indexes
        for it1 in range (1, len(A)):
            
            num1 = A[it1]
            
            for it2 in range (it1):
                
                num2 = A[it2]
                
                 # check the difference
                diff = num1 - num2
                
                if (it2, diff) not in diff_map:
                    diff_map[(it2, diff)] = 1
                
                diff_map[(it1, diff)] = diff_map[(it2, diff)] + 1
                
        #print (diff_map)   
        # return the maximum of values
        return max (diff_map.values())

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        dp,res = {},0
        for i in range(len(A)):
            for j in range(i+1,len(A)):
                dp[j,A[j]-A[i]] = dp.get((i,A[j]-A[i]),1)+1
                res = max(res,dp[(j,A[j]-A[i])])
        return res

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        dp = {}
        
        for i in range(1, len(A)):
            for j in range(i):
                diff = A[i]-A[j]
                dp[(i, diff)] = max(dp.get((i, diff), 2), dp.get((j, diff), 1)+1)
                
        return max(dp.values())
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        # -5:1
        # -2:1 3:1 -5:1
        # -7:1 -2:2 3:1 -5:2
        # 
        if not A:
            return 0
        result = 0
        N = len(A)
        d= {}
        for i in range(1,N):
            for j in range(0,i):
                diff = A[i] - A[j]
                d[i,diff] = d.get((j,diff), 0 )+ 1
                result = max(result,d[i,diff] )
                
        return result+1
            

class Solution:
    def longestArithSeqLength(self, A):
        dp = dict()
        for endi, endv in enumerate(A[1:], start = 1):
            for starti, startv in enumerate(A[:endi]):
                diff = endv - startv
                if (starti, diff) in dp:
                    dp[(endi, diff)] = dp[(starti, diff)] + 1
                else:
                    dp[(endi, diff)] = 2
        return max(dp.values())
    
class Solution:
    def longestArithSeqLength(self, A):
        dp = dict()
        for starti, startv in enumerate(A):
            for endi, endv in enumerate(A[starti+1:], start = starti+1):
                diff = endv - startv
                if (starti, diff) in dp:
                    dp[(endi, diff)] = dp[(starti, diff)] + 1
                else:
                    dp[(endi, diff)] = 2
        return max(dp.values())
    
class Solution:
    def longestArithSeqLength(self, A):
        N = len(A)
        dp = [{0:1} for _ in range(N)]
        for end in range(1, N):
            for start in range(end):
                diff = A[end] - A[start]
                if diff in dp[start]:
                    dp[end][diff] = dp[start][diff] + 1
                else:
                    dp[end][diff] = 2
        return max(max(dp[end].values()) for end in range(1, N))
    
class Solution:
    def longestArithSeqLength(self, A):
        ans = 2
        n = len(A)
        index = {}
        dp = [[2] * n for i in range(n)]
        
        for i in range(n-1):
            for j in range(i+1, n):
                first = A[i] + A[i] - A[j]
                if first in index:
                    dp[i][j] = dp[index[first]][i] + 1
                    #ans = max(ans, dp[i][j])
                    if dp[i][j] > ans: ans = dp[i][j]
            index[A[i]] = i
        return ans

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        # this is very tricky, A inside can have duplicates
        mem = {}
        prev_values = set()
        for idx, num in enumerate(A):
            for pval in prev_values:
                diff = num - pval
                mem[num, diff] = max(mem.get((num, diff), 1), 1 + mem.get((pval, diff), 1))
            prev_values.add(num)
        return max(mem.values(), default = 0)
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        if len(A) < 2:
            return len(A)
        sequenceEnds = dict()
        maxSequenceLen = 2
        for i in range(len(A)):
            seenDiffs = set()
            for j in range(i):
                diff = A[i] - A[j]
                if diff in seenDiffs:
                    continue
                elif diff in sequenceEnds:
                    sequencesWithDiff = sequenceEnds[diff]
                    if A[j] in sequencesWithDiff:
                        sequenceLength = sequencesWithDiff[A[j]]
                        sequencesWithDiff[A[i]] = sequenceLength + 1
                        maxSequenceLen = max(maxSequenceLen, sequenceLength + 1)
                    else:
                        sequencesWithDiff[A[i]] = 2
                else:
                    sequencesWithDiff = dict()
                    sequencesWithDiff[A[i]] = 2
                    sequenceEnds[diff] = sequencesWithDiff
                seenDiffs.add(diff)
        return maxSequenceLen

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        ans = 2
        n = len(A)
        index = {}
        dp = [[2] * n for i in range(n)]
        
        for i in range(n-1):
            for j in range(i+1, n):
                first = A[i] * 2 - A[j]
                if first in index:
                    dp[i][j] = dp[index[first]][i] + 1
                    ans = max(ans, dp[i][j])
            index[A[i]] = i
        return ans
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        dp = {}
        for i in range(1, len(A)):
            for j in range(i):
                d = A[i] - A[j]
                if (j, d) in dp:
                    dp[i, d] = dp[j, d] + 1
                else:
                    dp[i, d] = 2
        return max(dp.values())
        
        # dp = {}
        # for i, a2 in enumerate(A[1:], start=1):
        #     for j, a1 in enumerate(A[:i]):
        #         d = a2 - a1
        #         if (j, d) in dp:
        #             dp[i, d] = dp[j, d] + 1
        #         else:
        #             dp[i, d] = 2
        # return max(dp.values())

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        # subsequence problem -> dp
        # dp[i][j] -- length of arithmetic subsequence ending at ith and jth element
        ans = 2
        n = len(A)
        index = {}
        dp = [[2] * n for i in range(n)]
        
        for i in range(n-1):
            for j in range(i+1, n):
                first = A[i] * 2 - A[j]
                if first in index:
                    dp[i][j] = dp[index[first]][i] + 1
                    ans = max(ans, dp[i][j])
            index[A[i]] = i
        return ans

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        length = len(A)
        onleft = set()
        onleftl = []
        onright = [0 for _ in range(501)]
        toextend = [{} for _ in range(501)]
        res = 2
        for v in A:
            onright[v] += 1
        for i in range(0, length):
            val = A[i]
            tex = toextend[val]
            onright[val] -= 1
            for lval in onleftl:
                diff = val - lval
                nextval = val + diff
                if nextval > 500 or nextval < 0 or onright[nextval] == 0:
                    continue
                c = tex.get(diff, 2) + 1
                if c > res:
                    res = c
                else:
                    ending = (res - c) * diff + nextval
                    if ending > 500 or ending < 0 or onright[ending] == 0:
                        continue
                toextend[nextval][diff] = c
            if val not in onleft:
                onleft.add(val)
                onleftl.append(val)
        return res
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        
        dp=[{} for i in range(len(A))]
        for i in range(len(A)):
            dp[i]={0:1}
            for j in range(i):
                diff=A[i]-A[j]
                if diff in dp[j]:
                    dp[i][diff]=dp[j][diff]+1
                else:
                    dp[i][diff]=2
            
        ans=0
        for dic in dp:
            if dic:
                ans=max(ans,max(dic.values()))
            
        #print(dp)
        return ans
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        n = len(A)
        
        dp = [{} for _ in range(n)]
        
        ans = 0
        
        for i in range(n):
            dp[i][0] = 1
            for j in range(i):
                diff = A[i] - A[j]
                
                if diff not in dp[j]:
                    dp[i][diff] = 2
                else:
                    dp[i][diff] = dp[j][diff] + 1
            
            ans = max(ans, max(dp[i].values()))
        
        return ans

class Solution:
    def longestArithSeqLength(self, A):
        dp = dict()
        for endi, endv in enumerate(A[1:], start = 1):
            for starti, startv in enumerate(A[:endi]):
                diff = endv - startv
                if (starti, diff) in dp:
                    dp[(endi, diff)] = dp[(starti, diff)] + 1
                else:
                    dp[(endi, diff)] = 2
        return max(dp.values())
    
class Solution:
    def longestArithSeqLength(self, A):
        dp = dict()
        for starti, startv in enumerate(A):
            for endi, endv in enumerate(A[starti+1:], start = starti+1):
                diff = endv - startv
                if (starti, diff) in dp:
                    dp[(endi, diff)] = dp[(starti, diff)] + 1
                else:
                    dp[(endi, diff)] = 2
        return max(dp.values())
    
class Solution:
    def longestArithSeqLength(self, A):
        N = len(A)
        dp = [{0:1} for _ in range(N)]
        for end in range(1, N):
            for start in range(end):
                diff = A[end] - A[start]
                if diff in dp[start]:
                    dp[end][diff] = dp[start][diff] + 1
                else:
                    dp[end][diff] = 2
        return max(max(dp[end].values()) for end in range(1, N))
    
class Solution:
    def longestArithSeqLength(self, A):
        ans = 2
        n = len(A)
        index = {}
        dp = [[2] * n for i in range(n)]
        
        for i in range(n-1):
            for j in range(i+1, n):
                first = A[i] * 2 - A[j]
                if first in index:
                    dp[i][j] = dp[index[first]][i] + 1
                    ans = max(ans, dp[i][j])
            index[A[i]] = i
        return ans

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        biggest = max(A)
        
        def findLen(A: List[int]) -> int:
            seen = set()
            dp = defaultdict(lambda: defaultdict(lambda: 0)) 
            
            for a in A:
                for prev in seen:
                    gap = a - prev
                    newLen = 2 if dp[gap][prev] == 0 else 1 + dp[gap][prev]
                    dp[gap][a] = max(dp[gap][a], newLen)
                    
                seen.add(a)
            
            return max([l for gaps in dp.values() for l in gaps.values()])
                
        
        return findLen(A)
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        if not A:
            return 0
        n = len(A)
        if n < 2:
            return 0

        res = 2
        dp = [{} for i in range(n)]
        dp[1] = {A[1]-A[0]: 2}
        for k in range(2, n):
            for i in range(k):
                diff = A[k] - A[i]
                if diff in dp[i]:
                    dp[k][diff] = dp[i][diff] + 1
                else:
                    dp[k][diff] = 2
        return max(max(item.values()) for item in dp if item)
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        length = len(A)
        onright = [0 for _ in range(501)]
        toextend = [{} for _ in range(501)]
        res = 2
        for v in A:
            onright[v] += 1
        for i in range(0, length):
            val = A[i]
            tex = toextend[val]
            onright[val] -= 1
            doneself = False
            for lval in A[:i]:
                diff = val - lval
                nextval = val + diff
                if nextval > 500 or nextval < 0 or onright[nextval] == 0:
                    continue
                if nextval == val:
                    if doneself:
                        continue
                    doneself = True
                if diff in tex:
                    c = tex[diff] + 1
                else:
                    c = 3
                if c > res:
                    res = c
                toextend[nextval][diff] = c
        return res
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        
        from collections import defaultdict

        d = [{} for _ in range(len(A))]
        res = 2

        for i, x in enumerate(A):
            for j in range(i):
                diff = x - A[j]
                if diff in d[j]:
                    d[i][diff] = max(d[j][diff] + 1, d[i][diff]) if diff in d[i] else d[j][diff] + 1
                    d[j].pop(diff)

                    res = max(res, d[i][diff])
                    
                else:
                    d[i][diff] = 2

                
        return res


class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        a = A
        n = len(a)
        
        count = [{} for i in range(n)]
        
        for i in range(1,n):
            for j in range(i):
                diff = a[i] - a[j]
                
                if diff in count[j]:
                    count[i][diff] = 1 + count[j][diff]
                else:
                    count[i][diff] = 1
                
        max_val = 0
        for item in count:
            if item:
                max_val = max(max_val, max(item.values()))
        
        return max_val+1
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        length = len(A)
        onleft = set()
        onright = [0 for _ in range(501)]
        toextend = [{} for _ in range(501)]
        res = 2
        for v in A:
            onright[v] += 1
        for i in range(0, length):
            val = A[i]
            tex = toextend[val]
            onright[val] -= 1
            for lval in onleft:
                diff = val - lval
                nextval = val + diff
                if nextval > 500 or nextval < 0 or onright[nextval] == 0:
                    continue
                c = tex.get(diff, 2) + 1
                if c > res:
                    res = c
                else:
                    ending = (res - c) * diff + nextval
                    if ending > 500 or ending < 0 or onright[ending] == 0:
                        continue
                toextend[nextval][diff] = c
            onleft.add(val)
        return res
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        n = len(A)
        dp = [[2] * n for i in range(n)]
        index = [-1] * 501
        res = 2
        for i in range(n):
            for j in range(i+1, n):
                first = 2 * A[i] - A[j]
                if first < 0 or first >= 500 or index[first] == -1:
                    continue
                dp[i][j] = dp[index[first]][i] + 1
                res = max(res, dp[i][j])
            index[A[i]] = i
        return res
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        size = len(A)
        if size <= 1:
            return size
        
        nums = [{} for _ in range(size)]
        for i in range(1, size):
            for j in range(0, i):
                diff = A[i] - A[j]
                if diff in nums[j]:
                    nums[i][diff] = nums[j][diff]+1
                else:
                    nums[i][diff] = 2
            
        max_num = 0
        for i in range(1, size):
            max_num = max(max_num, max(nums[i].values()))
        return max_num
            

class Solution:
    def longestArithSeqLength(self, A):
        dp = dict()
        for endi, endv in enumerate(A[1:], start = 1):
            for starti, startv in enumerate(A[:endi]):
                diff = endv - startv
                if (starti, diff) in dp:
                    dp[(endi, diff)] = dp[(starti, diff)] + 1
                else:
                    dp[(endi, diff)] = 2
        return max(dp.values())
    
class Solution:
    def longestArithSeqLength(self, A):
        dp = dict()
        for starti, startv in enumerate(A):
            for endi, endv in enumerate(A[starti+1:], start = starti+1):
                diff = endv - startv
                if (starti, diff) in dp:
                    dp[(endi, diff)] = dp[(starti, diff)] + 1
                else:
                    dp[(endi, diff)] = 2
        return max(dp.values())
    
class Solution:
    def longestArithSeqLength(self, A):
        N = len(A)
        dp = [{0:1} for _ in range(N)]
        for end in range(1, N):
            for start in range(end):
                diff = A[end] - A[start]
                if diff in dp[start]:
                    dp[end][diff] = dp[start][diff] + 1
                else:
                    dp[end][diff] = 2
        return max(max(dp[end].values()) for end in range(1, N))

from collections import defaultdict, Counter
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:        
        dp = defaultdict(lambda: defaultdict(lambda: 1))
        best = 0
        for i in range(len(A)):
            num = A[i]

            for prev_num in list(dp.keys()):
                step = num - prev_num
                dp[num][step] = max(dp[prev_num][step] + 1, dp[num][step])
                best = max(best, dp[num][step])
            dp[num][0] = max(dp[num][0], 1)
            best = max(dp[num][0], best)
        return best
                

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        dp = {}
        for i, a2 in enumerate(A[1:], start=1):
            for j, a1 in enumerate(A[:i]):
                d = a2 - a1
                if (j, d) in dp:
                    dp[i, d] = dp[j, d] + 1
                else:
                    dp[i, d] = 2
        return max(dp.values())
    
#         dp = {}
#         for i, a2 in enumerate(A[1:], start=1):
#             for j, a1 in enumerate(A[:i]):
#                 d = a2 - a1
#                 if (j, d) in dp:
#                     dp[i, d] = dp[j, d] + 1
#                 else:
#                     dp[i, d] = 2
#         return max(dp.values())

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        result = 2
        
        L = len(A)
        index = [-1] * 2001 
        
        dp = [[2] * L for _ in range(L)]

        
        for i in range(L - 1):
            for j in range(i + 1, L):
                prevVal = 2 * A[i] - A[j]
                
                if index[prevVal] == -1:
                    continue
                else:
                    idx = index[prevVal]
                    if idx == -1:
                        dp[i][j] = 2
                    else:
                        dp[i][j] = dp[idx][i] + 1
                        result = max(result, dp[i][j])
            
            index[A[i]] = i
        
        return result
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        a = len(A)
        dp = [[0]*a for _ in range(a)] # dp array
        #print(dp)
        index = [-1]*20001#index array
        maximum = 2
        for i in range(0,a):
            dp[i] = [2]*a
            for j in range(i+1, a):
                #print("A[i]",A[i],"A[j]",A[j] )
                first = A[i]*2-A[j]
                #print("first",first)
                if first < 0 or index[first]==-1:
                    continue
                else:
                    #print("index[first]",index[first])
                    #print("dp[index[first]][i]",dp[index[first]][i])
                    dp[i][j] = dp[index[first]][i]+1
                    #print("dp[i][j]",dp[i][j])
                    maximum = max(maximum,dp[i][j] ) 
                    #print("max", maximum)
            #print(dp)
            index[A[i]] = i
        return maximum
    
    

class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        
        pos_diff_to_length = {}
        result = 0
        for i in range(1, len(A)):
            for j in range(i):
                if (j, A[i] - A[j]) in pos_diff_to_length:
                    pos_diff_to_length[(i, A[i] - A[j])] = pos_diff_to_length[(j, A[i] - A[j])] + 1 
                else:
                    pos_diff_to_length[(i, A[i] - A[j])] = 2
                result = max(result, pos_diff_to_length[(i, A[i] - A[j])])
                
        return result
class Solution:
    def longestArithSeqLength(self, A: List[int]) -> int:
        DP = [0]*len(A)
        for i in range(len(A)):
            temp = {}
            for j in range(i):
                diff = A[i]-A[j]
                temp[diff] = DP[j].get(diff, 0)+1
            DP[i] = temp
        return max(max(d.values()) for d in DP if d)+1
