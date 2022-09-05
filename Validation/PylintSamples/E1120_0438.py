class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        A = arr
        if m == len(A): 
            return m
        length = [0] * (len(A) + 2)
        res = -1
        for i, a in enumerate(A):
            left, right = length[a - 1], length[a + 1]
            if left == m or right == m:
                res = i
            length[a - left] = length[a + right] = left + right + 1
        return res
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        
        if m == len(arr):
            return m
        
        groupLen = [0 for i in range(len(arr) + 2)]
        
        latestStep = -1
        
        for step in range(len(arr)):
            index = arr[step] - 1
            left = groupLen[index - 1] 
            right = groupLen[index + 1]
            groupLen[index - left] = groupLen[index + right] = 1 + left + right
            
            if left == m or right == m:
                latestStep = step
            
        return latestStep
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        sizes = [0] * (len(arr) + 2)
        res = -1
        cnt = 0
        for step, cur in enumerate(arr, start=1):
            l, r = sizes[cur - 1], sizes[cur + 1]
            new_sz = l + 1 + r
            sizes[cur - l] = sizes[cur + r] = new_sz
            if l == m:
                cnt -= 1
            if r == m:
                cnt -= 1
            if new_sz == m:
                cnt += 1
            if cnt:
                res = step
        return res
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        n=len(arr)
        count=[0]*(n+2)
        lens=[0]*(n+2)
        res=-1
        for i,a in enumerate(arr):
            if lens[a]:
                continue
            l=lens[a-1]
            r=lens[a+1]
            t=l+r+1
            lens[a]=lens[a-l]=lens[a+r]=t
            count[l]-=1
            count[r]-=1
            count[t]+=1
            if count[m]:
                res=i+1
        return res
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        res = -1
        cnt = 0
        lm = {i: 0 for i in range(len(arr) + 2)}
        for i, idx in enumerate(arr):
            length = lm[idx - 1] + 1 + lm[idx + 1]
            if lm[idx - 1] == m:
                cnt -= 1
            if lm[idx + 1] == m:
                cnt -= 1
            if length == m:
                cnt += 1
            if cnt > 0:
                res = i + 1
            lm[idx - lm[idx - 1]] = length
            lm[idx + lm[idx + 1]] = length
        return res
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        N = len(arr)
        L = [None for i in range(N)]
        
        num_m_groups = 0
        latest_step = -1
        
        for i in range(N):
            idx = arr[i]-1
            
            if( (idx==0 or L[idx-1]==None) and (idx==N-1 or L[idx+1]==None)):
                L[idx] = (idx,idx)
                diff = 1
                if(m==diff):
                    num_m_groups += 1
            elif((idx==0 or L[idx-1]==None)):
                x,y = L[idx+1]
                if(y-x+1==m):
                    num_m_groups -= 1
                new_pair = (idx,y)
                L[idx] = new_pair
                L[y] = new_pair
                if(y-idx+1==m):
                    num_m_groups += 1
            elif((idx==N-1 or L[idx+1]==None)):
                x,y = L[idx-1]
                if(y-x+1==m):
                    num_m_groups -= 1
                new_pair = (x,idx)
                L[idx] = new_pair
                L[x] = new_pair
                if(idx-x+1==m):
                    num_m_groups += 1
            else:
                x1,y1 = L[idx-1]
                x2,y2 = L[idx+1]
                if(y1-x1+1==m):
                    num_m_groups -= 1
                if(y2-x2+1==m):
                    num_m_groups -= 1
                new_pair = (x1,y2)
                if(y2-x1+1==m):
                    num_m_groups += 1
                L[x1] = new_pair
                L[y2] = new_pair
            if(num_m_groups>0):
                latest_step = i+1
        return latest_step
                
               

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        nums = []
        max_index = -1
        correct_blocks = 0
        latest_index = -1
        for _ in range(len(arr)):
            nums.append(0)
        for i in range(len(arr)):
            index = arr[i]-1
            
            if index == 0:
                try:
                    nums[index] = 1 + nums[index+1]
                    if nums[index+1] == m:
                        correct_blocks -= 1
                    if 1 + nums[index+1] == m:
                        correct_blocks += 1
                    if nums[index+1] != 0:
                        val = 1 + nums[index+1]
                        nums[index + nums[index+1]] = val
                        nums[index+1] = val
                except:
                    return 1
            elif index == len(arr)-1:
                try:
                    nums[index] = 1 + nums[index-1]
                    if nums[index-1] == m:
                        correct_blocks -= 1
                    if 1 + nums[index-1] == m:
                        correct_blocks += 1
                    if nums[index-1] != 0:
                        val = 1 + nums[index - 1]
                        nums[index - nums[index-1]] = val
                        nums[index-1] = val
                except:
                    return 1
            else:
                try:
                    val = 1 + nums[index-1] + nums[index+1]
                    if nums[index-1] == m:
                        correct_blocks -= 1
                    if nums[index+1] == m:
                        correct_blocks -= 1
                    if 1 + nums[index-1] + nums[index+1] == m:
                        correct_blocks += 1
                    nums[index] = val
                    if nums[index-1] != 0:
                        nums[index - nums[index-1]] = val
                        nums[index-1] = val
                    if nums[index+1] != 0:
                        nums[index + nums[index+1]] = val
                        nums[index+1] =va;
                except:
                    pass
            if correct_blocks > 0:
                latest_index = i+1
        return latest_index
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        if (m == len(arr)):
            return len(arr)
        
        bit_information = [0] * (len(arr) + 2)
        target_group_size_counter = 0
        ret = -2
        
        
        for i in range(len(arr)):
            total_length = 1 + bit_information[arr[i] - 1] + bit_information[arr[i] + 1]   
            bit_information[arr[i]] = total_length
            target_group_size_counter -= 1 if bit_information[arr[i] - 1] == m else 0
            bit_information[arr[i] - bit_information[arr[i] - 1]] = total_length
            target_group_size_counter -= 1 if bit_information[arr[i] + 1] == m else 0
            bit_information[arr[i] + bit_information[arr[i] + 1]] = total_length
            target_group_size_counter += 1 if total_length == m else 0
            ret = i if target_group_size_counter > 0 else ret
            
        return ret + 1       
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        l = len(arr)
        x = [[] for x in range(l+1)]
        if l == m:
            return l

        last = -1
        lens = [0 for x in range(l+1)]
        for i in range(0,l):
            cur = arr[i]
            right = []
            left = []
            if cur+1 < l+1:
                right = x[cur+1]
            if cur-1 >= 0:
                left = x[cur-1]

            lv = rv = cur
            ss = 1
            if left:
                lv = left[1]
                ss += left[0]
                lens[left[0]] -=1     
                if left[0] == m:
                    last = i
            if right:
                rv = right[2]
                ss += right[0]
                lens[right[0]] -=1
                if right[0] == m:
                    last = i
            lens[ss]+=1
            x[lv] = [ss,lv,rv]
            x[rv] = [ss,lv,rv]


        
        return last
        
        for i in range(l-1, 0, -1):
            cur = arr[i]
            if lC[cur] or rC[cur]:  return i
            if cur+m+1 <= l:
                temp = True
                for j in range(cur+1,cur+m+1):
                    if rC[j]:
                        rC[j] = False
                        temp = False
                        break
                if temp: rC[cur+m+1]=True
            if cur-m-1 >= 0:
                temp = True
                for j in range(cur-m, cur):
                    if lC[j]: 
                        lC[j] = False
                        temp = False
                        break
                if temp: lC[cur-m-1]=True
        return -1
            
        
        
        
        
        mx = l
        mxcount = 1
        ls = [l]
        for i in range(l-1,-1,-1):
            cur = arr[i]
            prev = 0
            j = self.bisearch(cur,done,0,len(done))
            val = done[j]
            prev = done[j-1]

            if m == val-cur-1 or m == cur-prev-1:
                return i

            done = done[:j] +[cur] + done[j:]
            
            
        return -1
    
    def bisearch(self, cur: List[int], arr:List[int], i:int, j:int) -> int:
        if i==j:
            return j
        if j-i <= 1:
            if arr[i] < cur:
                return j
            else:
                return i
        mid = (j-i)//2
        if cur < arr[i+mid]:
            return self.bisearch(cur, arr, i, i+mid)
        else:
            return self.bisearch(cur, arr, i+mid,j)

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        N = len(arr)
        count = defaultdict(int)
        left = {}
        right = {}
        lastStep = -1
        for step, n in enumerate(arr):
            i = n-1
            # print('----')
            # print(step, i)
            newLeft = i
            newRight = i

            if i > 0:
                if i-1 in right:
                    newLeft = right[i-1]
                    del right[i-1]
                    count[i-newLeft] -= 1
            if i < N-1:
                if i+1 in left:
                    newRight = left[i+1]
                    del left[i+1]                    
                    count[newRight-i] -= 1


            left[newLeft] = newRight
            right[newRight] = newLeft
            count[newRight - newLeft + 1] += 1
            # print('left:',left)
            # print('right:',right)
            # print('count:',count)
            if count[m] > 0:
                lastStep = step + 1

        return lastStep
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        start_dic, end_dic = {}, {}
        m_len_interval = []
        latest = -1
        for step, position in enumerate(arr):
            if position -1 not in end_dic and position + 1 not in start_dic:
                start_dic[position] = 1
                end_dic[position] = 1
                if m == 1: m_len_interval.append((position, position))
            if position - 1 in end_dic and position + 1 not in start_dic:
                length = end_dic[position - 1]
                old_start_index = position - 1 - length + 1
                old_end_index = position - 1
                if length == m:
                    m_len_interval.remove((old_start_index, old_end_index))
                new_start_index = old_start_index
                new_end_index = position
                start_dic[new_start_index] = length + 1
                del end_dic[old_end_index]
                end_dic[new_end_index] = length + 1
                if length + 1 == m:
                    m_len_interval.append((new_start_index, new_end_index))
            if position - 1 not in end_dic and position + 1 in start_dic:
                length = start_dic[position + 1]
                old_start_index = position + 1
                old_end_index = old_start_index + length - 1
                if length == m:
                    m_len_interval.remove((old_start_index, old_end_index))
                new_start_index = position
                new_end_index = old_end_index
                del start_dic[old_start_index]
                start_dic[new_start_index] = length + 1
                end_dic[new_end_index] = length + 1
                if length + 1 == m:
                    m_len_interval.append((new_start_index, new_end_index))
            if position - 1 in end_dic and position + 1 in start_dic:
                old_len_1 = end_dic[position - 1]
                old_start_index_1 = position - 1 - old_len_1 + 1
                old_end_index_1 = position - 1
                if old_len_1 == m: m_len_interval.remove((old_start_index_1, old_end_index_1))
                old_len_2 = start_dic[position + 1]
                old_start_index_2 = position + 1
                old_end_index_2 = position + 1 + old_len_2 - 1
                if old_len_2 == m: m_len_interval.remove((old_start_index_2, old_end_index_2))
                new_start = old_start_index_1
                new_end = old_end_index_2
                new_len = old_len_1 + 1 + old_len_2
                if new_len == m: m_len_interval.append((new_start, new_end))
                start_dic[new_start] = new_len
                end_dic[new_end] = new_len
                del start_dic[old_start_index_2]
                del end_dic[old_end_index_1]
            if m_len_interval: latest = step
        return latest + 1 if latest != -1 else -1
                
                
                
                
                
            
            

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        h = defaultdict(int)
        result = -1
        counter = defaultdict(int)
        
        for i in range(len(arr)):
            n = arr[i] - 1

            l = h[n - 1]
            r = h[n + 1]

            if l > 0 or r > 0:
                newL = l + r + 1
                if l > 0:
                    h[n - l] = newL
                    counter[l] -= 1
                if r > 0:
                    h[n + r] = newL
                    counter[r] -= 1
                h[n] = newL
                counter[newL] += 1
            else:
                h[n] = 1
                counter[1] += 1
            if counter[m] > 0:
                result = i + 1
            # print(counter)
        return result

class Solution:
    def isSorted(self, arr):
        for i in range(1, len(arr)):
            if arr[i] < arr[i-1]:
                return False
        return True
    
    def findLatestStep(self, arr: List[int], m: int) -> int:
        length = len(arr)
        if m == length:
            # Full string of 1s can only be found at last step
            return m
        
        if self.isSorted(arr):
            return m
        
        # Pad front and back to make boundary conditions easier
        binary = [0] * (len(arr) + 2)
        latest_step = -1
        
        for step in range(length):
            pos = arr[step]
            binary[pos] = 1
            
            # Examine positions directly to the left and right i.e., group boundaries
            # Find/store the new group size at the new boundaries
            left_len = binary[pos-1]
            right_len = binary[pos+1]
            new_len = left_len + right_len + 1
            
            if left_len == m or right_len == m:
                # Target length persistent until prev step
                latest_step = step
                
            binary[pos-left_len] = new_len
            binary[pos+right_len] = new_len
                
        return latest_step
            
    def findLatestStepNaive2(self, arr: List[int], m: int) -> int:
        length = len(arr)
        if m == length:
            # Full string of 1s can only be found at last step
            return m
        
        if self.isSorted(arr):
            return m
        
        binary = [1] * len(arr)
        
        for step in range(length-1, m-1, -1):
            pos = arr[step] - 1
            binary[pos] = 0
            
            # Note that at each step, only one group is getting split into two
            # All other groups stay the same, so no need to iterate through entire string
            left_len = 0
            right_len = 0
            for i in range(pos-1, -1, -1):
                if binary[i]:
                    left_len += 1
                else:
                    break
            for i in range(pos+1, length):
                if binary[i]:
                    right_len += 1
                else:
                    break
            
            # Check only newly-formed groups.
            if left_len == m or right_len == m:
                return step
        return -1

    def findLatestStepNaive(self, arr: List[int], m: int) -> int:
        length = len(arr)
        if m == length:
            # Full string of 1s can only be found at last step
            return m
        
        if self.isSorted(arr):
            return m
        
        binary = [1] * len(arr)
        
        for step in range(length-1, m-1, -1):
            binary[arr[step] - 1] = 0

            # Iterate through current binary string
            current_group_len = 0
            max_group_len = 0
            for j in range(length):
                if binary[j]:
                    current_group_len += 1
                elif current_group_len > 0:
                    if current_group_len == m:
                        return step
                    max_group_len = max(max_group_len, current_group_len)
                    current_group_len = 0
            if current_group_len == m:
                return step
            max_group_len = max(max_group_len, current_group_len)
            # If max drops below m, stop; subsequent passes will only add more 0s
            if max_group_len < m:
                return -1
        return -1
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        
        
        n = len(arr)
        if m == n:
            print(n)

        def dfs(start: int, end: int, step: int, target: int):
            if end > len(arr) or end < 1:
                return -1
            if start > len(arr) or start < 1:
                return -1
            if end < start:
                return -1
            if end - start + 1 < target:
                return -1

            if end - start + 1 == target:
                return step
            bp = arr[step - 1]
            res = -1
            if start <= bp <= end:
                res = max(dfs(start, bp - 1, step - 1, target),
                          dfs(bp + 1, end, step - 1, target))
            else:
                res = max(res, dfs(start, end, step - 1, target))
            return res


        return dfs(1, n, n, m)

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        
        
        
        # my solution ... 1424 ms ... 77 % ... 27.7 MB ... 92 %
        #  time: O(n)
        # space: O(n)
        
        res = -1
        e2s = {}  # e2s[5] = 3 u8868u793a 3-4-5 u4f4du7f6eu73b0u5728u5747u4e3a1u3002u6b64u65f6u5fc5u6709s2e[3] = 5
        s2e = {}
        cnt = {}  # cnt[3] = 8 u8868u793au6b64u65f6u6709 8 u4e2a '111'
        for i, v in enumerate(arr):  # i+1 u662fu5f53u524d stepuff0cvu662fu5f53u524dstepu4ece0u53d8u4e3a1u7684u4f4du7f6euff0cu5176u4e24u4fa7u7684u7d22u5f15u4e3av-1u548cv+1
            if v-1 not in e2s and v+1 not in s2e:
                l, r = v, v
            elif v-1 not in e2s and v+1 in s2e:
                l, r = v, s2e[v+1]
                del s2e[v+1]
                cnt[r-v] -= 1
                if not cnt[r-v]:
                    del cnt[r-v]
            elif v-1 in e2s and v+1 not in s2e:
                l, r = e2s[v-1], v
                del e2s[v-1]
                cnt[v-l] -= 1
                if not cnt[v-l]:
                    del cnt[v-l]
            else:
                l, r = e2s[v-1], s2e[v+1]  # u4f4du7f6e v u53d8u4e3a 1 u540euff0cv u5de6u4fa7u6700u8fdc 1 u53ca v u53f3u4fa7u6700u8fdc 1
                del e2s[v-1]
                del s2e[v+1]
                cnt[v-l] -= 1  # u66f4u65b0u65e7u7684u9891u6b21
                if not cnt[v-l]:
                    del cnt[v-l]
                cnt[r-v] -= 1
                if not cnt[r-v]:
                    del cnt[r-v]
            s2e[l] = r
            e2s[r] = l
            cnt[r-l+1] = cnt.get(r-l+1, 0) + 1  # u589eu52a0u65b0u7684u9891u6b21
            if m in cnt:
                res = i + 1
        return res
        
        

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        
        
        
        # my solution ... 1332 ms ... 80 % ... 27.5 MB ... 98 %
        #  time: O(n)
        # space: O(n)
        
        if m == len(arr):
            return m
        res = -1
        e2s = {}  # e2s[5] = 3 u8868u793a 3-4-5 u4f4du7f6eu73b0u5728u5747u4e3a1u3002u6b64u65f6u5fc5u6709s2e[3] = 5
        s2e = {}
        cnt = {}  # cnt[3] = 8 u8868u793au6b64u65f6u6709 8 u4e2a '111'
        for i, v in enumerate(arr):  # i+1 u662fu5f53u524d stepuff0cvu662fu5f53u524dstepu4ece0u53d8u4e3a1u7684u4f4du7f6euff0cu5176u4e24u4fa7u7684u7d22u5f15u4e3av-1u548cv+1
            if v-1 not in e2s and v+1 not in s2e:
                l, r = v, v
            elif v-1 not in e2s and v+1 in s2e:
                l, r = v, s2e[v+1]
                del s2e[v+1]
                cnt[r-v] -= 1
                if not cnt[r-v]:
                    del cnt[r-v]
            elif v-1 in e2s and v+1 not in s2e:
                l, r = e2s[v-1], v
                del e2s[v-1]
                cnt[v-l] -= 1
                if not cnt[v-l]:
                    del cnt[v-l]
            else:
                l, r = e2s[v-1], s2e[v+1]  # u4f4du7f6e v u53d8u4e3a 1 u540euff0cv u5de6u4fa7u6700u8fdc 1 u53ca v u53f3u4fa7u6700u8fdc 1
                del e2s[v-1]
                del s2e[v+1]
                cnt[v-l] -= 1  # u66f4u65b0u65e7u7684u9891u6b21
                if not cnt[v-l]:
                    del cnt[v-l]
                cnt[r-v] -= 1
                if not cnt[r-v]:
                    del cnt[r-v]
            s2e[l] = r
            e2s[r] = l
            cnt[r-l+1] = cnt.get(r-l+1, 0) + 1  # u589eu52a0u65b0u7684u9891u6b21
            if m in cnt:
                res = i + 1
        return res
        
        

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        len_arr = len(arr)
        tot = 0
        latest = -1
        i1 = 0
        lis = [[] for x in range(len_arr)]
        for i in arr:
            index = i-1
            if(index > 0 and lis[index-1] and index < len_arr-1 and lis[index+1]):
                if(lis[index-1][2] == m):
                    tot -= 1
                if(lis[index+1][2] == m):
                    tot -= 1
                start = lis[index-1][0]
                end = lis[index+1][1]
                lis1 = [start, end, lis[index-1][2]+1+lis[index+1][2]]
                if(lis1[2] == m):
                    tot += 1
                lis[start] = lis1
                lis[end] = lis1
            elif(index > 0 and lis[index-1]):
                if(lis[index-1][2] == m):
                    tot -= 1
                start = lis[index-1][0]
                end = index
                if(lis[index-1][2] + 1 == m):
                    tot += 1
                lis1 = [start, end, lis[index-1][2]+1]
                lis[start] = lis1
                lis[end] = lis1
            elif(index < len_arr - 1 and lis[index+1]):
                if(lis[index+1][2] == m):
                    tot -= 1
                start = index
                end = lis[index+1][1]
                if(lis[index+1][2] + 1 == m):
                    tot += 1
                lis1 = [start, end, lis[index+1][2]+1]
                lis[end] = lis1
                lis[start] = lis1
            else:
                lis[index] = [index, index, 1]
                if(m == 1):
                    tot += 1
            if(tot > 0):
                latest = i1+1
            i1 += 1
        return latest
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        hd, td = {}, {}
        ld = {}
        n = -1
        for k, i in enumerate(arr):
            if i-1 in td and i+1 in hd:
                h = td[i-1]
                t = hd[i+1]
                hd[h] = t
                td[t] = h
                del hd[i+1]
                del td[i-1]
                ld[t-i] = ld[t-i]-1
                ld[i-h] = ld[i-h]-1
                if t-h+1 not in ld:
                    ld[t-h+1] = 0
                ld[t-h+1] = ld[t-h+1] + 1
            elif i-1 in td:
                h = td[i-1]
                hd[h] = i
                td[i] = h
                del td[i-1]
                ld[i-h] = ld[i-h] - 1
                if i-h+1 not in ld:
                    ld[i-h+1] = 0
                ld[i-h+1] = ld[i-h+1] + 1
            elif i+1 in hd:
                t = hd[i+1]
                hd[i] = t
                td[t] = i
                del hd[i+1]
                ld[t-i] = ld[t-i] - 1
                if t-i+1 not in ld:
                    ld[t-i+1] = 0
                ld[t-i+1] = ld[t-i+1] + 1
            else:
                hd[i] = i
                td[i] = i
                if 1 not in ld:
                    ld[1] = 0
                ld[1] = ld[1] + 1
            if m in ld and ld[m] > 0:
                n = k+1
        return n

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        n = len(arr)
        f = [ii for ii in range(n)]     # union find
        b = [0 for ii in range(n)]      # whether turns to 1
        s = [0 for ii in range(n)]      # sum
        def findunion(n):
            if f[n] == n:
                return n
            else:
                f[n] = findunion(f[n])
                return f[n]
        
        ans = -1
        m_set = 0
        for i in range(len(arr)):
            item = arr[i]-1
            b[item] = 1
            s[item] = 1
            tmp = 1
            
            if item < n-1 and b[item+1] == 1:
                f[item+1] = item
                s[item] = s[item+1]+1
                if s[item+1] == m:
                    m_set -= 1
                tmp = s[item]
                
            if item > 0 and b[item-1] == 1:
                fa = findunion(item-1)
                f[item] = fa
                if s[fa] == m:
                    m_set -= 1
                s[fa] = s[item] + s[fa]
                tmp = s[fa]
                
            if tmp == m:
                m_set += 1
            if m_set > 0:  
                ans = i+1
        
        return ans

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        l = len(arr)
        x = [[] for x in range(l+1)]
        if l == m:
            return l

        last = -1
        lens = [0 for x in range(l+1)]
        for i in range(0,l):
            cur = arr[i]
            right = []
            left = []
            if cur+1 < l+1:
                right = x[cur+1]
            if cur-1 >= 0:
                left = x[cur-1]

            lv = rv = cur
            ss = 1
            if left:
                lv = left[1]
                ss += left[0]
                lens[left[0]] -=1     
                if left[0] == m:
                    last = i
            if right:
                rv = right[2]
                ss += right[0]
                lens[right[0]] -=1
                if right[0] == m:
                    last = i
            lens[ss]+=1
            x[lv] = [ss,lv,rv]
            x[rv] = [ss,lv,rv]

        return last
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        states = [None]*len(arr)
        numSizeM = 0
        latestM = -1
        for i in range(len(arr)):
            ind = arr[i]-1
            if ind != 0 and ind != len(arr)-1 and states[ind-1] != None and states[ind+1] != None:
                leftInd = ind - states[ind-1][0]
                rightInd = ind + states[ind+1][1]
                if states[leftInd][1] == m:
                    numSizeM -= 1
                if states[rightInd][0] == m:
                    numSizeM -= 1
                groupSize = states[leftInd][1] + 1 + states[rightInd][0]
                states[leftInd][1] = groupSize
                states[rightInd][0] = groupSize
                if groupSize == m:
                    numSizeM += 1
            elif ind != 0 and states[ind-1] != None:
                leftInd = ind - states[ind-1][0]
                if states[leftInd][1] == m:
                    numSizeM -= 1
                groupSize = states[leftInd][1] + 1
                states[leftInd][1] = groupSize
                states[ind] = [groupSize, 1]
                if groupSize == m:
                    numSizeM += 1
            elif ind != len(arr)-1 and states[ind+1] != None:
                rightInd = ind + states[ind+1][1]
                if states[rightInd][0] == m:
                    numSizeM -= 1
                groupSize = states[rightInd][0] + 1
                states[rightInd][0] = groupSize
                states[ind] = [1, groupSize]                
                if groupSize == m:
                    numSizeM += 1
            else:
                states[ind] = [1, 1]
                if m == 1:
                    numSizeM += 1
            if numSizeM > 0:
                latestM = i+1
        return latestM
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        
        
        sizes = collections.defaultdict(int)
        has = [0]*len(arr)
        ends = {}
        out = 0
        for i, a in enumerate(arr):
            has[a-1] += 1
            if a-1 in ends and a+1 in ends:
                length1 = a-1 - ends[a-1] + 1
                length2 = ends[a+1] + 1 - (a+1)
                temp1 = ends[a+1]
                temp2 = ends[a-1]
                ends[temp2] = temp1
                ends[temp1] = temp2
                sizes[length1] -= 1
                sizes[length2] -= 1
                sizes[length1 + 1 + length2] += 1
            elif a-1 in ends:
                length1 = a-1 - ends[a-1] + 1
                ends[a] = ends[a-1]
                ends[ends[a-1]] = a
                sizes[length1] -= 1
                sizes[length1+1] += 1
            elif a + 1 in ends:
                length1 = ends[a+1] - (a+1) + 1
                ends[a] = ends[a+1]
                ends[ends[a+1]] = a
                sizes[length1] -= 1
                sizes[length1+1] += 1
            else:
                ends[a] = a
                sizes[1] += 1
            
            if sizes[m] > 0:
                out = i+1
        return out if out else -1
class Solution:         
    def findLatestStep(self, arr: List[int], m: int) -> int:
        n = len(arr)
        if n == m:
            return n
        lengths = [0 for _ in range(n + 2)]
        last_seen = -1
        for step, i in enumerate(arr):
            left_length = lengths[i - 1]
            right_length = lengths[i + 1]
            new_length = left_length + right_length + 1
            lengths[i] = new_length
            lengths[i - left_length] = new_length
            lengths[i + right_length] = new_length
            if left_length == m or right_length == m:
                last_seen = step
        return last_seen
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        ans = -1
        l = collections.defaultdict(int)
        count = collections.defaultdict(int)
        step = 0
        for num in arr:
            step += 1
            left, right = l[num-1], l[num+1]
            l[num] = l[num-left] = l[num+right] = left+right+1
            count[left+right+1] += 1
            count[left] -= 1
            count[right] -= 1
            if count[m]:
                ans = step
        return ans

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        forward = [0] * len(arr)
        backward = [0] * len(arr)
        counter = collections.defaultdict(int)
        res = -1
        step = 1
        
        for i in arr:
            i = i - 1
            #print(counter)
            if i - 1 >= 0 and i + 1 < len(arr):
                curr = forward[i-1] + 1 + backward[i+1]
                counter[forward[i-1]] -= 1
                if counter[forward[i-1]] == 0:
                    del counter[forward[i-1]]
                counter[backward[i+1]] -= 1
                if counter[backward[i+1]] == 0:
                    del counter[backward[i+1]]
                backward[i-forward[i-1]] = curr
                forward[i+backward[i+1]] = curr
                counter[curr] += 1
            elif i == 0 and i + 1 == len(arr):
                if m == 1:
                    return step
                else:
                    break
            elif i == 0:
                curr = 1 + backward[i+1]
                counter[backward[i+1]] -= 1
                if counter[backward[i+1]] == 0:
                    del counter[backward[i+1]]
                backward[i] = curr
                forward[i+backward[i+1]] = curr
                counter[curr] += 1
            else:
                curr = forward[i-1] + 1
                counter[forward[i-1]] -= 1
                if counter[forward[i-1]] == 0:
                    del counter[forward[i-1]]
                forward[i] = curr
                backward[i-forward[i-1]] = curr
                counter[curr] += 1
            
            if counter[m] >= 1:
                    res = step
            
            step += 1
        
        return res

from collections import defaultdict
from typing import *
from bisect import *

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        n = len(arr)
        Start = dict()
        End = dict()
        NumWithSize = defaultdict(int)
        for i in range(n):
            NumWithSize[0] += 1
        step = 0
        res = -1
        for x in arr:
            step += 1
            x -= 1
            NumWithSize[0] -= 1
            if (x-1) in End:
                leftSize = End[x-1] - Start[x-1] + 1
                NumWithSize[leftSize] -= 1
                if (x+1) in End:
                    rightSize = End[x+1] - Start[x+1] + 1
                    NumWithSize[rightSize] -= 1
                    NumWithSize[leftSize + rightSize + 1] += 1
                    Start[End[x+1]] = Start[x-1]
                    End[Start[x-1]] = End[x+1]
                else:
                    # only merge with left
                    NumWithSize[leftSize + 1] += 1
                    Start[x] = Start[x-1]
                    End[x] = x
                    End[Start[x-1]] = x
            elif (x+1) in End:
                # only merge with right
                rightSize = End[x+1] - Start[x+1] + 1
                NumWithSize[rightSize] -= 1
                NumWithSize[rightSize + 1] += 1
                Start[x] = x
                End[x] = End[x+1]
                Start[End[x+1]] = x
            else:
                # make only 1 block siez
                NumWithSize[1] += 1
                Start[x] = End[x] = x
            if NumWithSize[m] > 0:
                res = step
        return res

print((Solution().findLatestStep(arr = [3,5,1,2,4], m = 1))) # 4
print((Solution().findLatestStep(arr = [3,1,5,4,2], m = 2))) # -1
print((Solution().findLatestStep(arr = [1], m = 1))) # 1
print((Solution().findLatestStep(arr = [2,1], m = 2))) # 2

class Solution:
    
    def findLatestStep(self, arr: List[int], m: int) -> int:
        n = len(arr)
        barr = [None for i in range(n + 1)]
        comp_l = defaultdict(lambda: 0)
        
        last = -1
        for i, num in enumerate(arr):
            low, high = num, num
            
            if num > 1 and barr[num - 1] is not None:
                llow, lhigh = barr[num - 1]
                comp_l[lhigh - llow + 1] -= 1
                low = llow
                
            if num < len(barr) - 1 and barr[num + 1] is not None:
                rlow, rhigh = barr[num + 1]
                comp_l[rhigh - rlow + 1] -= 1
                high = rhigh
            
            comp_l[high - low + 1] += 1
            barr[low] = (low, high)
            barr[high] = (low, high)
            
            if comp_l[m] > 0:
                last = max(last, i+1)
        return last

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        start_dict = {}
        end_dict = {}
        m_dict = {}
        last_step = -2
        for i, num in enumerate(arr):
            if num + 1 not in start_dict and num - 1 not in end_dict:
                start = num
                end = num
            if num + 1 in start_dict and num-1 not in end_dict:
                end = start_dict.pop(num+1)
                start = num
                m_dict[end - num].remove(num+1)
            if num + 1 not in start_dict and num - 1 in end_dict:
                start = end_dict.pop(num-1)
                end = num
                m_dict[num-start].remove(start)
            if num + 1 in start_dict and num - 1 in end_dict:
                end = start_dict.pop(num+1)
                start = end_dict.pop(num-1)
                m_dict[end-num].remove(num+1)
                m_dict[num-start].remove(start)
            start_dict[start] = end
            end_dict[end] = start
            m_dict.setdefault(end - start + 1, set()).add(start)
            if m in m_dict and m_dict[m]:
                last_step = i
        return last_step+1

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        
        
        
        # my solution ... 1332 ms ... 80 % ... 27.5 MB ... 98 %
        #  time: O(n)
        # space: O(n)
        
        res = -1
        e2s = {}  # e2s[5] = 3 u8868u793a 3-4-5 u4f4du7f6eu73b0u5728u5747u4e3a1u3002u6b64u65f6u5fc5u6709s2e[3] = 5
        s2e = {}
        cnt = {}  # cnt[3] = 8 u8868u793au6b64u65f6u6709 8 u4e2a '111'
        for i, v in enumerate(arr):  # i+1 u662fu5f53u524d stepuff0cvu662fu5f53u524dstepu4ece0u53d8u4e3a1u7684u4f4du7f6euff0cu5176u4e24u4fa7u7684u7d22u5f15u4e3av-1u548cv+1
            if v-1 not in e2s and v+1 not in s2e:
                l, r = v, v
            elif v-1 not in e2s and v+1 in s2e:
                l, r = v, s2e[v+1]
                del s2e[v+1]
                cnt[r-v] -= 1
                if not cnt[r-v]:
                    del cnt[r-v]
            elif v-1 in e2s and v+1 not in s2e:
                l, r = e2s[v-1], v
                del e2s[v-1]
                cnt[v-l] -= 1
                if not cnt[v-l]:
                    del cnt[v-l]
            else:
                l, r = e2s[v-1], s2e[v+1]  # u4f4du7f6e v u53d8u4e3a 1 u540euff0cv u5de6u4fa7u6700u8fdc 1 u53ca v u53f3u4fa7u6700u8fdc 1
                del e2s[v-1]
                del s2e[v+1]
                cnt[v-l] -= 1  # u66f4u65b0u65e7u7684u9891u6b21
                if not cnt[v-l]:
                    del cnt[v-l]
                cnt[r-v] -= 1
                if not cnt[r-v]:
                    del cnt[r-v]
            s2e[l] = r
            e2s[r] = l
            cnt[r-l+1] = cnt.get(r-l+1, 0) + 1  # u589eu52a0u65b0u7684u9891u6b21
            if m in cnt:
                res = i + 1
        return res
        
        

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        if len(arr) == m:
            return m
        
        lengths = [0] * (len(arr) + 2)
        
        ans = -1
        
        for (step, value) in enumerate(arr):
            left, right = lengths[value-1], lengths[value+1]
            
            if left == m or right == m:
                ans = step
            
            lengths[value - left] = lengths[value + right] = left + right + 1
            
        return ans
        
        
#         N = len(arr) + 1
        
#         starts = dict()
#         ends = dict()
        
#         num_groups = 0
#         ans = -1
        
#         for (step, i) in enumerate(arr):
            
#             cur_range = [i, i]
#             if i + 1 in starts:
#                 cur_range[1] = starts[i+1]
                
#                 if starts[i+1] - i == m:
#                     num_groups -= 1
                
#                 del starts[i+1]
                
#             if i - 1 in ends:
#                 cur_range[0] = ends[i-1]
                
#                 if i - ends[i-1] == m:
#                     num_groups -= 1
#                 del ends[i-1]
            
#             starts[cur_range[0]] = cur_range[1]
#             ends[cur_range[1]] = cur_range[0]
            
#             if cur_range[1] - cur_range[0] + 1 == m:
#                 num_groups += 1
            
#             if num_groups > 0:
#                 ans = step + 1
#         return ans
            
        
        
        

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        if m==len(arr): return m
        
        border=[0]*(len(arr)+2)
        ans=-1
        
        for i in range(len(arr)):
            left=right=arr[i]
            if border[right+1]>0: right=border[right+1]
            if border[left-1]>0: left=border[left-1]
            border[left], border[right] = right, left
            if (right-arr[i]==m) or (arr[i]-left ==m): ans=i
        
        return ans

from collections import defaultdict


class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        s = [0 for _ in arr]
        ans = -1
        for i, n in enumerate(arr):
            num = n-1
            left = s[num-1] if num > 0 else 0
            right = s[num+1] if num < len(s)-1 else 0
            total = left + right + 1
            s[num-left] = total
            s[num+right] = total
            if left == m or right == m:
                ans = i
        if m == max(s):
            return i+1
        return ans
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        length_at_index = [0]*(len(arr)+2)
        count_of_length = [0]*(len(arr)+1)
        res = -1
        for i,index in enumerate(arr):
            left_length = length_at_index[index-1]
            right_length = length_at_index[index+1]

            new_length = left_length + right_length + 1
            length_at_index[index] = new_length
            length_at_index[index-left_length] = new_length
            length_at_index[index+right_length] = new_length

            count_of_length[left_length] -= 1
            count_of_length[right_length] -= 1
            count_of_length[new_length] += 1

            if count_of_length[m] > 0:res = i+1
        return res
class Solution:
    def findLatestStep(self, A: List[int], m: int) -> int:
        if m == len(A): return m
        length = [0] * (len(A) + 2)
        res = -1
        for i, a in enumerate(A):
            left, right = length[a - 1], length[a + 1]
            if left == m or right == m:
                res = i
            length[a - left] = length[a + right] = left + right + 1
        return res

import bisect 

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        N = len(arr)
        last = N - 1
        groupStartWith = {0: last}
        groupEndWith = {last: 0}
        groupLength = {N: 1}
        def decreaseGroupLength(oldlen):
            if oldlen >= m:
                oldlencount = groupLength.pop(oldlen)
                if oldlencount > 1 :
                    groupLength[oldlen] = oldlencount - 1
                    
        def increaseGroupLength(newlen):
            if newlen >= m:
                if newlen in groupLength:
                    groupLength[newlen]  = groupLength[newlen] + 1
                else :
                    groupLength[newlen] = 1
             
        def getGroup(num):
            if num  in groupStartWith: 
                right = groupStartWith.pop(num) 
                groupEndWith.pop(right)
                return (num, right)
            elif num in groupEndWith:
                left = groupEndWith.pop(num)
                groupStartWith.pop(left)
                return (left, num)
            
            starts = sorted(list(groupStartWith.keys()))
            index = bisect.bisect_left(starts, num) - 1
            if index < 0:
                return ()
            left = starts[index]
            right = groupStartWith[left]
            if left <= num and num <= right:
                groupStartWith.pop(left)
                groupEndWith.pop(right)
                return (left, right)
            return ()
                
        def updateGroup(left, right): 
            if right - left + 1 >= m:
                groupStartWith[left] = right
                groupEndWith[right] = left
            
            
        def removeNumber(num):
            group = getGroup(num)
            if len(group) == 0:
                return ()
            
            left , right = group
            res = ()
            oldlen = right - left + 1
            if oldlen < m:
                return ()
            decreaseGroupLength(oldlen)
            
            if num == left:
                newlen = oldlen - 1
                updateGroup(left + 1, right)
                increaseGroupLength(newlen)
                return (newlen,)
            
            if num == right:
                newlen = oldlen - 1
                updateGroup(left, right - 1)
                increaseGroupLength(newlen)
                return (newlen,)
            
            newLeftLen = num - left
            newRightLen = right - num
            
            if newLeftLen >= m: 
                updateGroup(left, num - 1)
                increaseGroupLength(newLeftLen)
                res = res + (newLeftLen,)
            
            if newRightLen >= m:
                updateGroup(num + 1, right)
                increaseGroupLength(newRightLen)
                res = res + (newRightLen,)
             
            return res

        if m == N:
            return m
        
        for i in range(N, 0 , -1): 
            #print(groupStartWith, i - 1, arr[i-1] - 1)
            if m in removeNumber(arr[i-1] - 1): 
                #print(groupStartWith, i - 1, arr[i-1] - 1, '=')
                return i - 1 
            #print(groupStartWith, i - 1, arr[i-1] - 1, '-') 
                
        return - 1 
            

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        n = len(arr)
        if m>n: return -1
        if m==n: return n
        helper = [0]*(n+1)
        res = -1
        
        for k, i in enumerate(arr):
            helper[i] = l = r = i
            if l>1 and helper[l-1]:
                l = helper[l-1]
                if r - l == m:
                    res = k
                helper[l], helper[r] = r, l
            if r<n and helper[r+1]:
                r = helper[r+1]
                if r - i == m:
                    res = k
                helper[l], helper[r] = r, l
                
        return res
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        bits = [0]*(len(arr)+2)
        answ = -1
        num_groups = 0
        for step, idx in enumerate(arr):
            before = bits[idx-1]
            after = bits[idx+1]
            group = 1
            if before + after == 0:
                bits[idx] = 1
            elif before == 0:
                bits[idx] = after + 1
                bits[idx+1] = 0
                bits[idx+after] = after + 1
                group = after + 1
            elif after == 0:
                bits[idx] = before + 1
                bits[idx-1] = 0
                bits[idx - before] = before + 1
                group = before + 1
            else:
                bits[idx-1], bits[idx+1] = 0, 0
                bits[idx-before], bits[idx+after] = before + after + 1, before + after + 1
                group = before + after + 1
            
            if group == m:
                num_groups += 1
            if before == m:
                num_groups -= 1
            if after == m:
                num_groups -= 1
            
            if num_groups > 0:
                answ = step + 1
                
            
        return answ
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        if m == len(arr):
            return m
        
        groups = [(0, len(arr)+1)] # both start and end are exclusive
        for i in range(len(arr)-1, -1, -1):
            temp = []
            for start, end in groups:
                if start <= arr[i] < end:
                    if end - arr[i] - 1 == m or arr[i] - 1 - start == m:
                        return i
                    if end - 1 - arr[i] > m:
                        temp.append((arr[i], end))
                    if arr[i] - 1 - start > m:
                        temp.append((start, arr[i]))
                elif end - 1 - start >= m:
                    temp.append((start, end))
            groups = temp
        return -1
    
# [3,5,1,2,4]
# 1
# [3,5,1,2,4]
# 2
# [3,5,1,2,4]
# 3
# [1]
# 1
# [2,1]
# 2

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        
        startwith = dict()
        endwith = dict()
        length = dict()
        ans = -1
        for i in range(len(arr)):
            if arr[i]+1 in startwith:
                if arr[i]-1 in endwith:
                    new_start = endwith[arr[i]-1]
                    new_end = startwith[arr[i]+1]
                    old_start = endwith[arr[i]-1]
                    old_end = arr[i]-1
                    del endwith[arr[i]-1]
                    length[old_end - old_start + 1] -= 1
                else:
                    new_start = arr[i]
                    new_end = startwith[arr[i]+1]
                old_start = arr[i]+1
                old_end = startwith[arr[i]+1]
                del startwith[arr[i]+1]
                length[old_end - old_start + 1] -= 1 
            elif arr[i]-1 in endwith:
                new_start = endwith[arr[i]-1]
                new_end = arr[i]
                old_start = endwith[arr[i]-1]
                old_end = arr[i]-1             
                length[old_end - old_start + 1] -= 1
                del endwith[arr[i]-1]
            else:
                new_start = arr[i]
                new_end = arr[i]
            length[new_end - new_start + 1] = length.get(new_end - new_start + 1, 0) + 1
            startwith[new_start] = new_end
            endwith[new_end] = new_start
            if m in length and length[m] > 0: ans = i+1
        return ans

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        last_found = -1
        #(start,end,size)
        hash_size = {}
        hash_chunks = {}
        for i in range (len(arr)):
            num = arr[i] - 1
            if num == 0:
                if (num + 1) not in hash_chunks:
                    hash_chunks[num] = (num,num,1)
                    hash_size[1] = 1 + hash_size.get(1,0)
                else:
                    (start,end,size) = hash_chunks[num + 1]
                    new_start = num
                    end = end
                    new_size = size + 1
                    hash_chunks[end] = (new_start,end,new_size)
                    hash_chunks[num] = (new_start,end,new_size)
                    hash_size[size] -= 1
                    hash_size[new_size] = 1 + hash_size.get(new_size,0)
            elif (num == len(arr) - 1):
                if (num - 1) not in hash_chunks:
                    hash_chunks[num] = (num,num,1)
                    hash_size[1] = 1 + hash_size.get(1,0)
                else:
                    (start,end,size) = hash_chunks[num - 1]
                    start = start
                    new_end = num
                    new_size = size + 1
                    hash_chunks[start] = (start,new_end,new_size)
                    hash_chunks[num] = (start,new_end,new_size)
                    hash_size[size] -= 1
                    hash_size[new_size] = 1 + hash_size.get(new_size,0)
            else:
                if ((num + 1) in hash_chunks) and ((num - 1) in hash_chunks):
                    (f_start,f_end,f_size) = hash_chunks[num - 1]
                    (b_start,b_end,b_size) = hash_chunks[num + 1]
                    new_front = f_start
                    new_end = b_end
                    new_size = f_size + b_size + 1
                    hash_chunks[f_start] = (new_front,new_end,new_size)
                    hash_chunks[b_end] = (new_front,new_end,new_size)
                    hash_size[f_size] -= 1
                    hash_size[b_size] -= 1
                    hash_size[new_size] = 1 + hash_size.get(new_size,0)
                elif (num + 1) in hash_chunks:
                    (start,end,size) = hash_chunks[num + 1]
                    new_start = num
                    end = end
                    new_size = size + 1
                    hash_chunks[end] = (new_start,end,new_size)
                    hash_chunks[num] = (new_start,end,new_size)
                    hash_size[size] -= 1
                    hash_size[new_size] = 1 + hash_size.get(new_size,0)
                elif (num - 1) in hash_chunks:
                    (start,end,size) = hash_chunks[num - 1]
                    start = start
                    new_end = num
                    new_size = size + 1
                    hash_chunks[start] = (start,new_end,new_size)
                    hash_chunks[num] = (start,new_end,new_size)
                    hash_size[size] -= 1
                    hash_size[new_size] = 1 + hash_size.get(new_size,0)
                else:
                    hash_chunks[num] = (num,num,1)
                    hash_size[1] = 1 + hash_size.get(1,0)
            if (m in hash_size) and (hash_size[m] > 0):
                last_found = i + 1
        return last_found

    
                    
                    

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        n = len(arr)
        left = [-1] * (n+2)
        right = [-1] * (n+2)
        
        ans = -1
        cnt = collections.Counter()
        for i, v in enumerate(arr):
            left[v] = right[v] = v
            l = r = v
            if left[v-1] != -1:
                l = left[v-1]
                cnt[v - l] -= 1
            if right[v+1] != -1:
                r = right[v+1]
                cnt[r - v] -= 1
            right[l] = r
            left[r] = l
            cnt[r - l + 1] += 1
            if cnt[m] > 0:
                ans = i + 1
        return ans
class Solution:
    def findLatestStep(self, A, m):
        if m == len(A): return m
        length = [0] * (len(A) + 2)
        res = -1
        for i, a in enumerate(A):
            left, right = length[a - 1], length[a + 1]
            if left == m or right == m:
                res = i
            length[a - left] = length[a + right] = left + right + 1
        return res
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        def find(x):
            while x!=par[x]:
                par[x] = par[par[x]]
                x = par[x]
            return x
        
        def union(x,y):
            nonlocal count
            
            px, py = find(x), find(y)
            if px==py:  return
            if size[px]==m: count-=1
            if size[py]==m: count-=1
                
            if size[px] > size[py]:    
                par[py] = par[px]
                size[px] += size[py]
                if size[px]==m: count+=1
                
            else:
                par[px] = par[py]
                size[py] += size[px]
                if size[py]==m: count+=1
                    
        count = 0
        n = len(arr)+2
        par = list(range(n))
        size = [0]*n
        res = -1
        for i, el in enumerate(arr, 1):
            size[el] = 1
            if m==1:    count+=1
            if size[el-1]:   union(el, el-1)
            if size[el+1]:   union(el, el+1)
            if count>0:    res = i
            
        return res
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        ret = -1
        n = len(arr)
        length = [0]*(n+2)
        count = [0]*(n+1)
        
        for i in range(n):
            a = arr[i]
            left = length[a-1]
            right = length[a+1]
            length[a] = length[a-left] = length[a+right] = left+right+1
            count[left] -= 1
            count[right] -= 1
            count[length[a]] += 1
            if count[m] > 0:
                ret = i+1
        return ret

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        n = len(arr)
        if m == n: return m
        length = [0] *(n + 2)
        res = -1
        for i, a in enumerate(arr):
            left, right = length[a-1], length[a+1]
            if left == m or right == m:
                res = i
            length[a-left] = length[a+right] = left + right + 1
        return res
                
                    
        

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        a,b,s,t=[0]*(len(arr)+2),[0]*(len(arr)+1),-1,0
        for p,i in enumerate(arr,1):
            j,k=a[i-1],a[i+1]
            a[i]=a[i-j]=a[i+k]=j+k+1
            if a[i]==m: t+=1
            if j==m: t-=1
            if k==m: t-=1
            if t: s=p
        return s
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        ans = -1
        size = collections.Counter()
        comp = [0 for _ in range(len(arr) + 2)]
        for i, n in enumerate(arr):
            left, right = comp[n - 1], comp[n + 1]
            comp[n] = comp[n - left] = comp[n + right] = left + right + 1
            size[left] -= 1
            size[right] -= 1
            size[comp[n]] += 1
            if size[m] > 0:
                ans = i + 1
        return ans
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        lengthStartingWith = {}
        lengthEndingWith = {}
        relevantStartingIndices = {}
        
        bestIndex = -1
        for index,j in enumerate(arr):
            print(index)
            i = j-1
            leftNeighbor = i-1
            rightNeighbor = i+1
            if(leftNeighbor in lengthEndingWith and rightNeighbor in lengthStartingWith):
                leftLength = lengthEndingWith[leftNeighbor]
                rightLength = lengthStartingWith[rightNeighbor]
                lengthEndingWith.pop(leftNeighbor)
                lengthStartingWith.pop(rightNeighbor)
                if(rightNeighbor in relevantStartingIndices):
                    relevantStartingIndices.pop(rightNeighbor)
                lengthStartingWith[leftNeighbor-leftLength+1] = leftLength + rightLength + 1
                if(leftLength + rightLength + 1 == m):
                    relevantStartingIndices[leftNeighbor-leftLength+1] = True
                else:
                    if(leftNeighbor-leftLength+1 in relevantStartingIndices):
                        relevantStartingIndices.pop(leftNeighbor-leftLength+1)
                lengthEndingWith[rightNeighbor+rightLength-1] = leftLength + rightLength + 1
                
            
            elif(leftNeighbor in lengthEndingWith):
                leftLength = lengthEndingWith[leftNeighbor]
                lengthEndingWith.pop(leftNeighbor)
                lengthStartingWith[leftNeighbor-leftLength+1] = leftLength + 1
                lengthEndingWith[i] = leftLength + 1
                
                if(leftLength + 1 == m):
                    relevantStartingIndices[leftNeighbor-leftLength+1] = True
                else:
                    if(leftNeighbor-leftLength+1 in relevantStartingIndices):
                        relevantStartingIndices.pop(leftNeighbor-leftLength+1)
                
            elif(rightNeighbor in lengthStartingWith):
                rightLength = lengthStartingWith[rightNeighbor]
                lengthStartingWith.pop(rightNeighbor)
                lengthEndingWith[rightNeighbor+rightLength-1] = rightLength + 1
                lengthStartingWith[i] = rightLength + 1
                
                if(rightNeighbor in relevantStartingIndices):
                    relevantStartingIndices.pop(rightNeighbor)
                if(rightLength + 1 == m):
                    relevantStartingIndices[i] = True
                else:
                    if(i in relevantStartingIndices):
                        relevantStartingIndices.pop(i)
                
            
            else:
                #print("here4")
                #print(i)
                lengthEndingWith[i] = 1
                lengthStartingWith[i] = 1
                
                if(m == 1):
                    relevantStartingIndices[i] = True
                else:
                    if(i in relevantStartingIndices):
                        relevantStartingIndices.pop(i)
            
            if(len(relevantStartingIndices) > 0):
                bestIndex = index + 1
        return bestIndex
            

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        if m == len(arr): return m
        length = [0] * (len(arr) + 2)
        res = -1
        for i, a in enumerate(arr):
            left, right = length[a - 1], length[a + 1]
            if left == m or right == m:
                res = i
            length[a - left] = length[a + right] = left + right + 1
        return res
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        n = len(arr)
        ta = [-1] * n
        sg = [-1] * n
        eg = [-1] * n
        g = {}
        step = 1
        ans = -1
        gc = {}
        for v in arr:
            idx = v - 1
            wl = False
            wr = False
            if idx > 0 and eg[idx-1] > -1:
                sgi = eg[idx-1]
                ngl = g[sgi]
                sg[sgi] = idx
                eg[idx] = sgi
                g[sgi] += 1
                gc[ngl] -= 1
                if ngl+1 not in gc:
                    gc[ngl+1] = 0
                gc[ngl+1] += 1
                wl = True
            if idx < n-1 and sg[idx+1] > -1:
                sgi = idx+1
                egi = sg[sgi]
                ngl = g[sgi]
                eg[egi] = idx
                sg[idx] = egi
                g[idx] = g[sgi]+1
                l = g.pop(sgi)
                gc[ngl] -= 1
                if ngl+1 not in gc:
                    gc[ngl+1] = 0
                gc[ngl+1] += 1
                wr = True
            if not wl and not wr:
                sg[idx] = idx
                eg[idx] = idx
                g[idx] = 1
                if 1 not in gc:
                    gc[1] = 0
                gc[1] += 1
            elif wl and wr:
                sgi = eg[idx]
                ngl = g[sgi]
                ngr = g[idx]
                l = g.pop(idx)
                gc[ngl] -= 1
                gc[ngr] -= 1
                if ngl+ngr-1 not in gc:
                    gc[ngl+ngr-1] = 0
                gc[ngl+ngr-1] += 1
                g[sgi] = g[sgi] + l - 1
                egi = sg[idx]
                eg[egi] = sgi
                sg[sgi] = egi
            ta[idx] = v

            if m in gc and gc[m] > 0:
                ans = step

            step += 1
        return ans
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        n = len(arr)
        d = dict() # max length of 1u2018s at index i
        counter = collections.Counter() # count of occurence of len
        ans = -1
        for i, num in enumerate(arr, 1):
            left = d.get(num-1, 0)
            right = d.get(num+1, 0)
            total = left+right+1
            d[num] = total
            d[num-left] = total
            d[num+right] = total
            counter[total] += 1
            counter[left] -= 1
            counter[right] -= 1
            if counter[m]:
                ans = i
        return ans
            

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        ans = -1
        comps = [0 for _ in range(len(arr) + 2)]
        size = collections.Counter()
        for i, a in enumerate(arr):
            l, r = comps[a - 1], comps[a + 1]
            comps[a] = comps[a - l] = comps[a + r] = l + r + 1
            size[l] -= 1
            size[r] -= 1
            size[comps[a]] += 1
            if size[m] > 0:
                ans = i + 1
        return ans
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        n = len(arr)
        ta = [-1] * n
        sg = [-1] * n
        eg = [-1] * n
        g = {}
        ans = -1
        gc = {}
        for step, v in enumerate(arr):
            idx = v - 1
            wl = False
            wr = False
            if idx > 0 and eg[idx-1] > -1:
                sgi = eg[idx-1]
                ngl = g[sgi]
                sg[sgi] = idx
                eg[idx] = sgi
                g[sgi] += 1
                gc[ngl] -= 1
                if ngl+1 not in gc:
                    gc[ngl+1] = 0
                gc[ngl+1] += 1
                wl = True
            if idx < n-1 and sg[idx+1] > -1:
                sgi = idx+1
                egi = sg[sgi]
                ngl = g[sgi]
                eg[egi] = idx
                sg[idx] = egi
                g[idx] = g[sgi]+1
                l = g.pop(sgi)
                gc[ngl] -= 1
                if ngl+1 not in gc:
                    gc[ngl+1] = 0
                gc[ngl+1] += 1
                wr = True
            if not wl and not wr:
                sg[idx] = idx
                eg[idx] = idx
                g[idx] = 1
                if 1 not in gc:
                    gc[1] = 0
                gc[1] += 1
            elif wl and wr:
                sgi = eg[idx]
                ngl = g[sgi]
                ngr = g[idx]
                l = g.pop(idx)
                gc[ngl] -= 1
                gc[ngr] -= 1
                if ngl+ngr-1 not in gc:
                    gc[ngl+ngr-1] = 0
                gc[ngl+ngr-1] += 1
                g[sgi] = g[sgi] + l - 1
                egi = sg[idx]
                eg[egi] = sgi
                sg[sgi] = egi
            ta[idx] = v

            if m in gc and gc[m] > 0:
                ans = step+1

            step += 1
        return ans
class Solution:
    def findLatestStep(self, a: List[int], m: int) -> int:
        index2len = defaultdict(int)
        cnt = Counter()
        last = -1
        for i, p in enumerate(a):    
            left_len, right_len = index2len[p-1], index2len[p+1]
            new_len = left_len + 1 + right_len
            index2len[p-left_len] = index2len[p+right_len] = new_len
            cnt[left_len] -= 1
            cnt[right_len] -= 1                
            cnt[new_len] += 1 
            if cnt[m] > 0: last = i + 1            
        return last
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        n = len(arr)
        start = {}
        end = {}
        groups = collections.defaultdict(set)
        ans = -1
        for idx,a in enumerate(arr):
            new_start,new_end = a,a
            if a + 1 in start:
                new_end = start[a + 1]
                del start[a + 1]
                groups[new_end - (a + 1 )+ 1].remove((a+1,new_end))
            if a - 1 in end:
                new_start = end[a-1]
                del end[a-1]
                groups[a-1 - new_start + 1].remove((new_start,a-1))
            start[new_start] = new_end
            end[new_end] = new_start
            groups[new_end - new_start + 1].add((new_start,new_end))
            if len(groups[m])>0:ans = idx+1
        return ans
                
            

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        n = len(arr)
        if m==n: return m
        res = -1
        length = [0 for _ in range(n+2)]
        
        for i,val in enumerate(arr):
            left, right = length[val-1], length[val+1]
            if left==m or right==m:
                res = i
            length[val-left] = length[val+right] = left+right+1
        return res   
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        dp = [0] * (len(arr) + 2)
        g = collections.Counter()
        ans = -1
        for i in range(len(arr)):
            l, r = dp[arr[i] - 1], dp[arr[i] + 1]
            dp[arr[i]] = l + r + 1
            dp[arr[i] - l] = dp[arr[i] + r] = l + r + 1
            g[l] -= 1
            g[r] -= 1
            g[dp[arr[i]]] += 1
            if g[m] > 0:
                ans = i + 1
        return ans
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        n = len(arr)
        if n == 1 : return 1
        dic, groups = {i:0 for i in range(1, n+1)}, {i:0 for i in range(1, n+1)}
        right = {i:0 for i in range(1, n+1)}
        laststep = -1
        for idx, i in enumerate(arr):
            # single 1
            if (i == 1 or dic[i-1] == 0) and (i == n or dic[i+1] == 0):
                groups[1] += 1
                dic[i] = i
                right[i] = i
            # add 1 to right
            elif (i == n or dic[i+1] == 0) and (i > 0 and dic[i-1] > 0):
                leftmost = dic[i-1]
                dic[i] = leftmost
                right[leftmost] = i
                right[i] = i
                groups[i-leftmost] -= 1
                groups[i-leftmost+1] += 1
            # add 1 to left
            elif (i == 1 or dic[i-1] == 0) and (i < n and dic[i+1] > 0):
                rightmost = right[i+1]
                dic[rightmost] = i
                dic[i] = i
                right[i] = rightmost
                groups[rightmost - i] -= 1
                groups[rightmost - i + 1] += 1
            else:
                leftmost = dic[i-1]
                rightmost = right[i+1]
                right[leftmost] = rightmost
                dic[rightmost] = leftmost
                groups[rightmost - i] -= 1
                groups[i - leftmost] -= 1
                groups[rightmost-leftmost+1] += 1

            if groups[m] > 0:
                laststep = idx+1
            # print("step:", idx+1, ":",groups)
        return laststep
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        '''
        for arr[k:k+m], it forms m-group if max(time[x] for x in range(k,k+m) < min(time[k-1],time[k+m]),  )
        cand(k)=
        '''
        turnedOnTime={}
        for i,x in enumerate(arr):
            turnedOnTime[x]=i+1
        n=len(arr)
        q=deque()
        ans=-1
        for i in range(1,n+1):
            while q and q[0][0]<i-m+1: q.popleft()
            while q and q[-1][1]<=turnedOnTime.get(i,n+1): q.pop()
            q.append((i,turnedOnTime.get(i,n+1)))
            if i>=m:
                vanishTime=min(turnedOnTime.get(i+1,n+1),turnedOnTime.get(i-m,n+1))
                cur=-1 if q[0][1]>=vanishTime else (vanishTime-1)
                ans=max(ans,cur)
        return ans

class Solution:
    def findLatestStep(self, a: List[int], m: int) -> int:
        n = len(a)
        l = [0] * (n + 2)
        cnt = [0] * (n + 2)
        ans = -1
        for i, x in enumerate(a):
            left, right = l[x - 1], l[x + 1]
            l[x] = l[x - left] = l[x + right] = left + right + 1
            cnt[left] -= 1
            cnt[right] -= 1
            cnt[l[x]] += 1
            if cnt[m]:
                ans = i + 1
        return ans
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        arr_len = len(arr)
        d = {i: [i-1, 0, i+1] for i in range(arr_len + 1)}  # d[i] = (previous i, length ones start i)
        last_round = -1  # last round where we had string of length m
        count_m = 0  # count of current number of such strings

        for cur_round, n in enumerate(arr):
            prev_idx, this_len, next_idx = d[n]
            if this_len == m:
                count_m -= 1
            if d[prev_idx][1] == m:
                count_m -= 1
            new_len = d[prev_idx][1] + this_len + 1
            d[prev_idx][1] = new_len
            d[prev_idx][2] = next_idx
            
            if next_idx <= arr_len:  # only set if still in rang
                d[next_idx][0] = prev_idx
                
            d[n] = None   # so generate error if reuse
            
            if new_len == m:
                count_m += 1
            if count_m > 0:
                last_round = cur_round + 1
        return last_round


class Solution:
    def findLatestStep(self, a: List[int], m: int) -> int:
        
        n = len(a)
        
        sz = {}
        cnt = Counter()
        
        ret = -1
        
        for i,x in zip(list(range(1, n+1)), a):
            
            # merge
            left = sz[x-1] if x-1 in sz else 0
            right = sz[x+1] if x+1 in sz else 0
            tot = left + right + 1
            
            sz[x-left] = tot
            sz[x+right] = tot
            
            cnt[left] -= 1
            cnt[right] -= 1
            cnt[tot] += 1
            
            if cnt[m]:
                ret = i
        
        return ret

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        from collections import Counter
        n = len(arr)
        mem = [0] * n
        counter = Counter()
        res = -1
        for j, v in enumerate(arr):
            i = v - 1
            mem[i] = 1
            # 0 1 2 3 4 5
            # 0 0 1 1 1 0
            l = mem[i - 1] if i - 1 >= 0 else 0
            r = mem[i + 1] if i + 1 <  n else 0
            counter[l] -= 1
            counter[r] -= 1
            cur = l + r + 1
            mem[i - l], mem[i + r] = cur, cur
            counter[cur] += 1
            # (j, v, cur, mem).p()
            if counter[m] > 0:
                res = j + 1
        return res
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        parent = [0] * (len(arr) + 2)
        size = [1] * (len(arr) + 1)
        
        count = collections.defaultdict(int)
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
                count[size[py]] -= 1
                count[size[px]] -= 1
                size[py] += size[px]
                count[size[py]] += 1
        
        answer = -1
        for i, value in enumerate(arr):
            parent[value] = value
            count[1] += 1
            
            if parent[value - 1]:
                union(value - 1, value)
            if parent[value + 1]:
                union(value, value + 1)
            
            if count[m]:
                answer = i + 1
        
        return answer
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        gLefts = dict()
        gRights = dict()
        gSizeCounter = collections.Counter()
        result = -1
        for step, x in enumerate(arr, 1):
            try:
                rGroup = gLefts[x + 1]
            except KeyError:
                rGroup = None
            try:
                lGroup = gRights[x - 1]
            except KeyError:
                lGroup = None
                
            if lGroup is not None and rGroup is not None:
                lSize = lGroup[2]
                rSize = rGroup[2]
                del gLefts[rGroup[0]]
                del gRights[lGroup[1]]
                
                gSizeCounter[lSize] -= 1
                    
                gSizeCounter[rSize] -= 1
                    
                lSize += 1 + rSize
                gSizeCounter[lSize] += 1
                lGroup[2] = lSize
                lGroup[1] = rGroup[1]
                gRights[lGroup[1]] = lGroup
            elif lGroup is not None:
                lSize = lGroup[2]
                
                gSizeCounter[lSize] -= 1
                    
                lSize += 1
                gSizeCounter[lSize] += 1
                lGroup[2] = lSize
                del gRights[lGroup[1]]
                lGroup[1] = x
                gRights[x] = lGroup
            elif rGroup is not None:
                rSize = rGroup[2]
                

                gSizeCounter[rSize] -= 1
                    
                rSize += 1
                gSizeCounter[rSize] += 1
                rGroup[2] = rSize
                del gLefts[rGroup[0]]
                rGroup[0] = x
                gLefts[x] = rGroup
            else:
                gSizeCounter[1] += 1
                gLefts[x] = gRights[x] = [x, x, 1]

            if gSizeCounter[m] > 0:
                result = step
                
        return result
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        n, ans = len(arr), -1
        if m == n: return m
        cnts = [0] * (n + 2)
        for i, x in enumerate(arr):
            cl, cr = cnts[x - 1], cnts[x + 1]
            if cl == m or cr == m: ans = i
            cnts[x - cl] = cnts[x + cr] = cl + cr + 1
        return ans

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        end = {}
        n = len(arr)
        count = collections.defaultdict(int)
        string = [0]*(n+1)
        ans = -1
        for i in range(len(arr)):
            start = arr[i]
            string[start] = 1
            left = start
            right = start
            flag1, flag2 = False, False
            if arr[i]-1 > 0 and string[arr[i]-1] == 1:
                l, r = end[arr[i]-1]
                left = l
                count[r-l+1] -= 1
                flag1 = True
            if arr[i]+1 <= n and string[arr[i]+1] == 1:
                l2, r2 = end[arr[i]+1]
                right = r2
                count[r2-l2+1] -= 1
                flag2 = True
            end[arr[i]] = (left, right)
            if flag1:
                end[l] = (l, right)
            if flag2:
                end[r2] = (left, r2)

            count[right - left + 1] += 1
            # print(i, right, left, count, end)
            if count[m] > 0:
                ans = i + 1
        return ans
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        N = len(arr)
        m_cnt = 0
        parent = [i for i in range(N)]
        cnt = [1]*N
        mark = [0]*N
        
        def find(i):
            if parent[i]==i:
                return i
            else:
                parent[i] = find(parent[i])
                return parent[i]
            
        def union(i,j,m_cnt):
            x = find(i)
            y = find(j)
            if cnt[x] == m:
                m_cnt -= 1
            if cnt[y] == m:
                m_cnt -= 1
            if cnt[x]+cnt[y] == m:
                m_cnt += 1
            if x<y:
                parent[y] = x
                cnt[x] += cnt[y]
            else:
                parent[x] = y
                cnt[y] += cnt[x]
            
            return m_cnt
        
        ans = -1
        
        
        
        for i,x in enumerate(arr):
            mark[x-1]=1
            l = False
            r = False
            
            if m==1:
                m_cnt+=1
            
            if x>1 and mark[x-2]==1:
                m_cnt = union(x-1,x-2,m_cnt)
            else:
                l =True
            if x<N and mark[x]==1:
                m_cnt = union(x-1,x,m_cnt)
            else:
                r = True
                
            
            
            if m_cnt>0:
                ans = i+1
                
            #print(m_cnt)
        return ans
            
            

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        lens = defaultdict(int) #len of union, count
        
        unions = {} #start:end, or vice versa
        
        latest = -1
        
        for step, i in enumerate(arr):
            start = unions.pop(i-1, i)
            end = unions.pop(i+1, i)
            
            unions[start] = end
            unions[end] = start
            
            left_size = i - start
            right_size = end - i
            
            lens[left_size] -= 1
            lens[right_size] -= 1
            lens[left_size + right_size + 1] += 1
            
            if lens[m]:
                latest = step + 1
        
        return latest
            
        
        # 1 0 1 0 1

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        res, n = -1, len(arr)
        # length of group
        length = [0] * (n + 2)
        # count of length
        count = [n] + [0] * n
        
        for i, v in enumerate(arr):
            left, right = length[v - 1], length[v + 1]
            length[v] = length[v - left] = length[v + right] = left + right + 1
            count[left] -= 1
            count[right] -= 1
            count[length[v]] += 1
            if count[m]:
                res = i + 1        
        return res
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        if m == len(arr):
            return m
        length = [0] * (len(arr) + 2)
        res = -1
        
        for i, pos in enumerate(arr):
            left, right = length[pos - 1], length[pos + 1]
            if left == m or right == m:
                res = i
            length[pos - left], length[pos + right] = left + right + 1, left + right + 1
            
        return res
class Solution:
    def findLatestStep(self, A, m):
        if m == len(A): 
            return m
        
        length = [0] * (len(A) + 2)
        res = -1
        
        for i, a in enumerate(A):
            left, right = length[a - 1], length[a + 1]
            if left == m or right == m:
                res = i
            length[a - left] = length[a + right] = left + right + 1
        return res

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        if len(arr) ==m:
            return m
        n = len(arr)
        max_length = [0]*(n+2)
        result = -1
        for step, i in enumerate(arr):
            left, right = max_length[i-1],  max_length[i+1]
            if left == m or right == m:
                result = step 
            
            max_length[i-left] = max_length[i+right] = left + right + 1
        
        return result
                

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        
        def helper(start, end, curStep):
            if curStep == 1: return curStep if m == 1 else -1
            
            if end - start + 1 < m: return -1
            
            elif end - start + 1 == m: return curStep
            
            else:    
                idx = arr[curStep - 1]

                if idx < start or idx > end: return helper(start, end, curStep - 1)

                else: return max(helper(start, idx - 1, curStep - 1), helper(idx + 1, end, curStep - 1))
                
        return helper(1, len(arr), len(arr))
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        if len(arr) == m:
            return m
        length = [0] * (len(arr) + 2)
        cnt = [0] * (len(arr) + 1)
        res = -1
        for idx, i in enumerate(arr):
            l = length[i - 1]
            r = length[i + 1]
            length[i] = length[i - l] = length[i + r] = l + r + 1
            cnt[l] -= 1
            cnt[r] -= 1
            cnt[length[i]] += 1
            if cnt[m]:
                res = idx + 1
        return res
import collections

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:        
        arr = [0] + arr + [0]
        N = len(arr)
        left  = [0]*N
        right = [0]*N
        sizes = collections.Counter()
        ans = 0
        for i in range(1, N-1):            
            l = left[arr[i]-1]
            r = right[arr[i]+1]
           
            sizes[l] -= 1
           
            sizes[r] -= 1
            sizes[l + r + 1] += 1
            left[arr[i] + 1 + r - 1]  = l + r + 1
            right[arr[i] - 1 - l + 1] = l + r + 1
            if sizes[m] >= 1:
                ans = i   
                
        return  ans if ans >= 1 else -1

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        group_size = [0 for _ in range(len(arr)+1)] 
        parent_idx = [-1 for _ in range(len(arr)+1)] # same for self leading, 0 for not init
        
        res = -1
        num_same = 0
        for i in range(1, len(arr)+1):
            num = arr[i-1]
            this_group = 1
            group_start = num
            group_end = num
            if num > 0 and group_size[num-1] > 0:
                this_group += group_size[num-1]
                group_start = parent_idx[num-1]
                if (group_size[num-1] == m):
                    num_same -= 1
                
            if num < len(arr) and group_size[num+1] > 0:
                this_group += group_size[num+1]
                group_end = num+group_size[num+1]
                if (group_size[num+1] == m):
                    num_same -= 1
            
            group_size[num] = this_group
            group_size[group_start] = this_group
            group_size[group_end] = this_group
            
            parent_idx[num] = group_start
            parent_idx[group_end] = group_start
            
            if (this_group == m):
                res = i
                num_same += 1
            elif (num_same > 0):
                res = i
        return res
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        ans = -1
        r = []
        b = [0,]*len(arr)
        i = 0
        while i<len(arr):
            v1 = arr[i] > 1 and b[arr[i]-2]
            v2 = arr[i] < len(arr) and b[arr[i]]
            if v1 and v2:
                h = b[arr[i]]
                t = b[arr[i]-2]
                b[arr[i]] = 0
                b[arr[i]-2] = 0
            elif v1:
                h = arr[i]
                t = b[arr[i]-2]
                b[arr[i]-2] = 0
            elif v2:
                h = b[arr[i]]
                t = arr[i]
                b[arr[i]] = 0
            else:
                h = arr[i]
                t = h

            b[t-1] = h
            b[h-1] = t

            i+=1

            if h-t+1 == m:
                ans = i
                r.append((t,h))
            elif r:
                while r and not (b[r[-1][0]-1] == r[-1][1]):
                    r.pop()

                if r:
                    ans = i

        return ans
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        ## u5c31u662fu80fdu591fu5f97u77e5u5230u5b58u5728u67d0u4e2au4e8cu8fdbu5236str u91ccu9762111 u957fu5ea6u7684u72b6u6001
        ## u6bcfu4e00u6b21u66f4u65b0u90fdu662fu4f1au6539u53d8 u5de6u8fb9uff0c u53f3u8fb9u7684u72b6u6001
        if m == len(arr): return m
        length = [0] * (len(arr) + 2)
        res = -1
        for i, a in enumerate(arr):
            left, right = length[a - 1], length[a + 1]
            if left == m or right == m:
                res = i
            length[a - left] = length[a + right] = left + right + 1
        return res
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        if (m > len(arr)):
            return -1
        elif (m == len(arr)):
            return len(arr)
        else:
            pass
        
        bit_information = [0] * (len(arr) + 1)
        target_group_size_counter = 0
        ret = -2
        
        
        for i in range(len(arr)):
            group_sizes = []
            total_length = 1
            neighbor_group_exists = [False, False]
            
            
            if (arr[i] > 1 and bit_information[arr[i] - 1] != 0):
                total_length += bit_information[arr[i] - 1]
                neighbor_group_exists[0] = True
                
                
            if (arr[i] < len(arr) and bit_information[arr[i] + 1] != 0):
                total_length += bit_information[arr[i] + 1]
                neighbor_group_exists[1] = True
                  
            bit_information[arr[i]] = total_length
            
            
            if (neighbor_group_exists[0]):
                target_group_size_counter -= 1 if bit_information[arr[i] - 1] == m else 0
                bit_information[arr[i] - bit_information[arr[i] - 1]] = total_length
            
            
            if (neighbor_group_exists[1]):
                target_group_size_counter -= 1 if bit_information[arr[i] + 1] == m else 0
                bit_information[arr[i] + bit_information[arr[i] + 1]] = total_length

            target_group_size_counter += 1 if total_length == m else 0
            ret = i if target_group_size_counter > 0 else ret
            
        return ret + 1       
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
#         dic = {}
#         def count_cluster(y,x,cur_count):
#             length = 0
#             if y[x-1] == 0:
#                 if x <n:
#                     if y[x+1] ==0:
#                         # ...0x0... 
#                         y[x] = 1
#                         dic[x] = x
#                         if m==1:
#                             cur_count+=1
#                     else:
#                         # ...0x#... 
#                         oldr = y[x+1]
#                         y[x] = 1 + y[x+1]
#                         y[dic[x+1]] = y[x]
#                         dic[x] = dic[x+1]
#                         dic[dic[x+1]] = x
#                         if oldr == m-1:
#                             cur_count +=1
#                         if oldr == m:
#                             cur_count -=1
#                 else:
#                     # ...0x 
#                     y[x] = 1
#                     dic[x] = x
#                     if m==1:
#                         cur_count+=1
#             else:
#                 if x <n:
#                     if y[x+1] ==0:
#                         # ...#x0... 
#                         oldl = y[x-1]
#                         y[x] = y[x-1] +1
#                         y[dic[x-1]] = y[x]
#                         dic[x] = dic[x-1]
#                         dic[dic[x-1]] = x
#                         if oldl == m-1:
#                             cur_count +=1
#                         if oldl == m:
#                             cur_count -=1
#                     else:
#                         # ...#x#... 
#                         oldr = y[x+1]
#                         oldl = y[x-1]
#                         y[x] = y[x-1] + 1 + y[x+1]
#                         temp = dic[x-1]
                        
#                         y[dic[x-1]] = y[x]
#                         dic[dic[x-1]] = dic[x+1]
                        
#                         y[dic[x+1]] = y[x]
#                         dic[dic[x+1]] = temp
                        
#                         if oldr==m:
#                             cur_count -= 1
#                         if oldl ==m:
#                             cur_count-=1
#                         if oldr+oldl == m-1:
#                             cur_count+=1
#                 else:
#                     # ...#x 
#                     oldl = y[x-1]
#                     y[x] = y[x-1] +1
#                     y[dic[x-1]] = y[x]
#                     dic[x] = dic[x-1]
#                     dic[dic[x-1]] = x
#                     if oldl == m-1:
#                         cur_count +=1
#                     if oldl == m:
#                         cur_count -=1
                
#             return cur_count     
#         n = len(arr)
#         s = [0] * (n+1)
#         last = -1
#         cur_count = 0
#         for idx,x in enumerate(arr):
#             cur_count=count_cluster(s,x,cur_count)
#             if cur_count>0:
#                 last = idx+1
#         return last

    #gonna try the union-find method
        def find(parent, x):
            if x == parent[x]:
                return x
            parent[x] = find(parent,parent[x])
            return parent[x]
        n = len(arr)
        parent = [0 for x in range(n+1)]
        size = [0] * (n+1)
        count = [0] * (n+1)
        ans = -1
        for i,pos in enumerate(arr):
            
            size[pos] = 1
            count[1] += 1
            parent[pos] = pos
            for j in [-1,1]:
                if (pos+j <=n) and (pos+j>0) and (parent[pos+j]!=0):
                    x = find(parent,pos+j)
                    y = find(parent,pos)
                    if x!=y:
                        
                        count[size[x]] -=1
                        count[size[y]] -=1
                        parent[x] = y
                        size[y] += size[x]
                        count[size[y]] +=1
            if count[m]>0:
                ans = i+1
        return ans
# class UF:
#     def __init__(self, n, m):
#         self.p = [i for i in range(n+1)]  # parent for each position
#         self.c = [0 for _ in range(n+1)]  # length of group for each position
#         self.m_cnt = 0                    # count of group with length m
#         self.m = m                        # m
        
#     def union(self, i, j):
#         pi, pj = self.find(i), self.find(j)
#         if pi != pj:
#             if self.c[pi] == self.m: self.m_cnt -= 1  # if previous length at pi is m, decrement m_cnt by 1
#             if self.c[pj] == self.m: self.m_cnt -= 1  # if previous length at pj is m, decrement m_cnt by 1
#             self.p[pj] = pi                           # union, use pi at parent for pj
#             self.c[pi] += self.c[pj]                  # update new length at pi
#             if self.c[pi] == self.m: self.m_cnt += 1  # if new length at pi == m, increment m_cnt by 1
            
#     def mark(self, i):                                
#         self.c[i] = 1                                 # when first visit a point, mark length as 1
#         if self.m == 1: self.m_cnt += 1               # if self.m == 1, increment m_cnt by 1
        
#     def find(self, i):                                # find parent of i
#         if self.p[i] != i:
#             self.p[i] = self.find(self.p[i])
#         return self.p[i]
    
# class Solution:
#     def findLatestStep(self, arr: List[int], m: int) -> int:
#         n = len(arr)
#         uf, ans = UF(n, m), -1                                   # create union find and answer
#         for i, num in enumerate(arr, 1):
#             uf.mark(num)
#             if num-1 >= 1 and uf.c[num-1]: uf.union(num-1, num)  # if left neighbor is marked, union the two
#             if num+1 < n+1 and uf.c[num+1]: uf.union(num+1, num) # if right neighbor is marked, union the two
                
#             if uf.m_cnt > 0: ans = i                             # if m_cnt > 0, meaning there exists some group with length m, update ans
#         return ans
    
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        def find(x):
            while x!=par[x]:
                par[x] = par[par[x]]
                x = par[x]
            return x
        
        def union(x,y):
            nonlocal count
            
            px, py = find(x), find(y)
            if px==py:  return
            if size[px]==m: count-=1
            if size[py]==m: count-=1
                
            if size[px] > size[py]:    
                par[py] = par[px]
                size[px] += size[py]
                if size[px]==m: count+=1
                
            else:
                par[px] = par[py]
                size[py] += size[px]
                if size[py]==m: count+=1
                    
        count = 0
        n = len(arr)+1
        par = list(range(n))
        size = [0]*n
        res = -1
        for i, el in enumerate(arr, 1):
            size[el] = 1
            if m==1:    count+=1
            if el-1>0 and size[el-1]:   union(el, el-1)
            if el+1<n and size[el+1]:   union(el, el+1)
            if count>0:    res = i
            
        return res
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        n = len(arr)
        res = -1
        if n==m:
            return n
        set_range = [0]*(n+2)
        for i in range(n):
            set_bit = arr[i]
            left_range = set_range[set_bit-1]
            right_range = set_range[set_bit+1]
            set_range[set_bit] = left_range+right_range+1
            set_range[set_bit-left_range] = set_range[set_bit+right_range] = set_range[set_bit]
            if left_range==m or right_range==m:
                res = i
        
        return res
class Solution:
    def findLatestStep(self, A: List[int], m: int) -> int:
        left = {}
        right = {}
        tot = {}
        cands = set()
        N = len(A)
        res = -1
        for i, a in enumerate(A):
            size = 1
            newLeft = newRight = a
            if a-1 in tot:  size += tot[a-1]
            if a+1 in tot:  size += tot[a+1]
            if a-1 in tot:
                for c in [a-1, left[a-1]]:
                    if c in cands: cands.remove(c)
                newLeft = left[a-1]
                right[left[a-1]] = right[a+1] if a+1 in tot else a
                tot[left[a-1]] = size
            if a+1 in tot:
                for c in [a+1, right[a+1]]:
                    if c in cands: cands.remove(c)
                newRight = right[a+1]
                left[right[a+1]] = left[a-1] if a-1 in tot else a
                tot[right[a+1]] = size
            tot[a] = size
            left[a] = newLeft
            right[a] = newRight
            if size == m:
                cands.add(newLeft)
                cands.add(newRight)
            if cands:
                res = i+1
        return res
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        
        uf = list(range(len(arr)))
        sz = [0] * len(arr)
        steps = set()
        
        def find(x):
            while x != uf[x]:
                # uf[x] = uf[uf[x]]
                x = uf[x]
            return x
        
        def union(p, q):
            pid, qid = find(p), find(q)
            if pid == qid:
                return 
            if sz[pid] == m and pid in steps:
                steps.remove(pid)
            if sz[qid] == m and qid in steps:
                steps.remove(qid)
            if sz[pid] < sz[qid]:
                uf[pid] = qid
                sz[qid] += sz[pid]
            else:
                uf[qid] = pid
                sz[pid] += sz[qid]
        last_step = -1
        for i in range(len(arr)):
            idx = arr[i] - 1
            sz[idx] = 1
            if idx - 1 >= 0 and sz[idx-1]:
                union(idx-1, idx)
            if idx + 1 < len(arr) and sz[idx+1]:
                union(idx+1, idx)
            if sz[find(idx)] == m:
                steps.add(find(idx))
            if steps:
                last_step = i + 1
            # print(steps)
        return last_step

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        length = [0]*(len(arr)+2)
        count = [0]*(len(arr)+1)
        latest = -1
        for idx, pos in enumerate(arr):
            left, right = length[pos-1], length[pos+1]
            length[pos] = length[pos - left] = length[pos + right] = left + right + 1
            count[left] -= 1
            count[right] -= 1
            count[length[pos]] += 1
            if count[m]:
                latest = idx + 1
        return latest

                
                
                

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
      
        n= len(arr)
        return self.dfs_helper(arr, n, 1, n, m)
    def dfs_helper(self, arr, step, left, right, target):
        if left > right or right-left+1<target :
            return -1 
        if right - left + 1 == target:
            return step
        breakpoint = arr[step-1]
        if left<=breakpoint<=right:
            res = max(self.dfs_helper(arr, step-1, left, breakpoint-1, target), self.dfs_helper(arr,step-1,breakpoint+1, right, target))
        else:
            res = self.dfs_helper(arr, step-1, left, right,target)
        return res 
import collections

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:        
        arr = [0] + arr + [0]
        N = len(arr)
        left  = [0]*N
        right = [0]*N
        sizes = collections.Counter()
        ans = 0
        for i in range(1, N-1):
            # print()
            # print("i: ", i)
            l = left[arr[i]-1]
            r = right[arr[i]+1]
            if sizes[l] >= 1:
                sizes[l] -= 1
            if sizes[r] >= 1:
                sizes[r] -= 1
            sizes[l + r + 1] += 1
            left[arr[i] + 1 + r - 1]  = l + r + 1
            right[arr[i] - 1 - l + 1] = l + r + 1
            if sizes[m] >= 1:
                ans = i
                
            # print(left)
            # print(right)
            # print("sizes: ", sizes)
                
        return  ans if ans >= 1 else -1

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        bit_information = [0] * (len(arr) + 1)
        target_group_size_counter = 0
        ret = -2
        
        
        for i in range(len(arr)):
            group_sizes = []
            total_length = None
            
            
            if (arr[i] > 1 and bit_information[arr[i] - 1] != 0 and arr[i] < len(arr) and bit_information[arr[i] + 1] != 0):
                group_sizes = [bit_information[arr[i] - 1], bit_information[arr[i] + 1]]
                total_length = 1 + bit_information[arr[i] - 1] + bit_information[arr[i] + 1]
                bit_information[arr[i] - bit_information[arr[i] - 1]] = total_length
                bit_information[arr[i] + bit_information[arr[i] + 1]] = total_length
            elif (arr[i] > 1 and bit_information[arr[i] - 1] != 0):
                group_sizes = [bit_information[arr[i] - 1]]
                total_length = bit_information[arr[i] - 1] + 1
                bit_information[arr[i] - bit_information[arr[i] - 1]] = total_length
                bit_information[arr[i]] = total_length
            elif (arr[i] < len(arr) and bit_information[arr[i] + 1] != 0):
                group_sizes = [bit_information[arr[i] + 1]]
                total_length = bit_information[arr[i] + 1] + 1
                bit_information[arr[i] + bit_information[arr[i] + 1]] = total_length
                bit_information[arr[i]] = total_length
            else:
                bit_information[arr[i]] = 1
                total_length = 1
                
            target_group_size_counter -= group_sizes.count(m)
            target_group_size_counter += 1 if total_length == m else 0
            

            if (target_group_size_counter > 0):
                ret = i
                
        return ret + 1       
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        n, uf, cs, ones, cache, res = len(arr), list(range(len(arr))), set(), [0] * len(arr), [0] * len(arr), -1
        def find(i):
            while i != uf[i]:
                i = uf[i]
            return i
        def union(nb, cur):
            p = find(nb)
            uf[p] = cur
            cache[cur] += cache[p]
            cache[p] = 0
            if p in cs:
                cs.remove(p)
        for i, v in enumerate(arr):
            l, cur, r = v - 2, v - 1, v
            ones[cur] = 1
            if l >= 0 and ones[l] == 1:
                union(l, cur)
            if r < n and ones[r] == 1:
                union(r, cur)
            cache[cur] += 1
            if cache[cur] == m:
                cs.add(cur)
            if len(cs) > 0:
                res = i + 1
            # print(f'{ones} {cs} {cache} {uf}')
        return res
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        
        if m == len(arr): return m
        d = {v: i + 1 for i, v in enumerate(arr)}
        latest = -1
        
        i, j = 1, m
        max_stack = collections.deque()
        
        for t in range(i, j + 1):
            while max_stack and max_stack[-1] < d[t]:
                max_stack.pop()
            max_stack.append(d[t])
        
        while j <= len(arr):
            in_max = max_stack[0]
            
            if in_max < d.get(i - 1, float('inf')) and in_max < d.get(j + 1, float('inf')):
                latest = max(latest, min(d.get(i - 1, float('inf')), d.get(j + 1, float('inf'))) - 1)
            
            if d[i] == max_stack[0]:
                max_stack.popleft()
            
            i += 1
            j += 1
            
            if j <= len(arr):
                while max_stack and max_stack[-1] < d[j]:
                    max_stack.pop()
                max_stack.append(d[j])

        return latest
from collections import deque 
class Solution:
    def findLatestStep(self, lis: List[int], m: int) -> int:
        n = len(lis)
        if m>n:
            return 0
        if m==n:
            return n
        if n==1:
            return 1
        lis = [[lis[i],i] for i in range(n)]
        lis.sort()
        #print(lis)
        q = deque()
        ans=0
        for i in range(n):
            while q and lis[i][1]>=q[-1][0]:
                q.pop()
            q.append([lis[i][1],i])
            while q and q[0][1]<=i-m:
                q.popleft()
            if i>=m-1:
                aa = q[0][0]
          #      print(aa,i,m)
                if i==m-1:
                    if i+1<n and aa<lis[i+1][1]:
                        ans = max(ans,lis[i+1][1])
                if i==n-1:
                    #print(lis[i-m][1],i,m)
                    if aa<lis[i-m][1]:
                        ans = max(ans,lis[i-m][1])
                else:
                    if i+1<n and aa<lis[i+1][1] and aa<lis[i-m][1]:
                        ans = max(ans,min(lis[i+1][1],lis[i-m][1]))
                #print(ans)
        return -1 if ans==0 else ans
            
            
            
        

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        if len(arr) == 1:
            if m == 0:
                return 0
            elif m == 1:
                return 1
            else:
                return -1
        
        if len(arr) == m:
            return m
        
        freq_indices = []
        last_freq = -1
        goal_freq = 0
        arr = [el - 1 for el in arr]
        left = [None] * len(arr)
        right = [None] * len(arr)
        
        idx = 0
        while idx < len(arr):
            change = arr[idx]
            if change == 0:
                left_end = change
                right_end = right[change + 1]
                left_size = 0
                if right_end is None:
                    right_end = change
                    right_size = 0
                else:
                    right_size = right_end - change
                
                right[change] = right_end
                left[right_end] = left_end
                
                if right_size == m:
                    goal_freq -= 1
                
                new_size = right_end - left_end + 1
                if new_size == m:
                    goal_freq += 1
                
                # print(new_size, goal_freq)
                if goal_freq > 0:
                    last_freq = idx
                    
            elif change == len(arr) - 1:
                right_end = len(arr) - 1
                left_end = left[change - 1]
                right_size = 0
                if left_end is None:
                    left_end = change
                    left_size = 0
                else:
                    left_size = change - left_end
                
                left[change] = left_end
                right[left_end] = right_end
                
                if left_size == m:
                    goal_freq -= 1
                
                new_size = right_end - left_end + 1
                if new_size == m:
                    goal_freq += 1
                
                if goal_freq:
                    last_freq = idx
                    
            else:
                left_end = left[change - 1]
                right_end = right[change + 1]
                if right_end is None:
                    right_end = change
                    right_size = 0
                else:
                    right_size = right_end - change
                    
                if left_end is None:
                    left_end = change
                    left_size = 0
                else:
                    left_size = change - left_end
                
                right[left_end] = right_end
                left[right_end] = left_end
                
                if right_size == m:
                    goal_freq -= 1
                    
                if left_size == m:
                    goal_freq -= 1
                
                
                new_size = right_end - left_end + 1
                if new_size == m:
                    goal_freq += 1
                
                if goal_freq:
                    last_freq = idx
            idx += 1
        
        if last_freq != -1:
            return last_freq + 1
        return -1
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        status = [0]*len(arr)
        cnt = collections.Counter()
        
        last = -1
        for step, idx in enumerate(arr):
            i = idx-1
            status[i] = 1
            
            left = not (i == 0 or status[i-1] == 0)
            right = not (i >= len(arr)-1 or status[i+1] == 0)
            
            if not left and not right:
                cnt[1] += 1
                status[i] = 1
            
            elif left and right:
                j, k = status[i-1], status[i+1]
                full = 1 + j + k
                status[i-j] = full
                status[i+k] = full
                cnt[full] += 1
                cnt[j] -= 1
                cnt[k] -= 1
            
            elif left:
                j = status[i-1]
                full = 1+j
                status[i-j] = full
                status[i] = full
                cnt[j] -= 1
                cnt[full] += 1
                
            elif right:
                k = status[i+1]
                full = 1+k
                status[i] = full
                status[i+k] = full
                cnt[k] -= 1
                cnt[full] += 1
                
            # print(step, status, cnt)
            if cnt[m] > 0:
                last = step+1
                
        return last

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        N = len(arr) + 1
        
        starts = dict()
        ends = dict()
        
        num_groups = 0
        ans = -1
        
        for (step, i) in enumerate(arr):
            
            cur_range = [i, i]
            if i + 1 in starts:
                cur_range[1] = starts[i+1]
                
                if starts[i+1] - i == m:
                    num_groups -= 1
                
                del starts[i+1]
                
            if i - 1 in ends:
                cur_range[0] = ends[i-1]
                
                if i - ends[i-1] == m:
                    num_groups -= 1
                del ends[i-1]
            
            starts[cur_range[0]] = cur_range[1]
            ends[cur_range[1]] = cur_range[0]
            
            if cur_range[1] - cur_range[0] + 1 == m:
                num_groups += 1
            
            if num_groups > 0:
                ans = step + 1
        return ans
            
        
        
        

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        n = len(arr)
        aux = [[0, -1, -1] for _ in range(n)]
        
        #      Caso 1  
        # arr = [3, 5, 1, 2, 4]
        # [[0,0,0], [0,0,0], [0,0,0], [0,0,0], [0,0,0]]
        # [Group size, startIdx, endIdx]

        if n == 1:
            return 1 if m == n else -1

        mCounter = 0
        result = -1
        for i in range(n):
            idx = arr[i] - 1
            
            if idx == 0: # Caso 1
                if aux[idx+1][0] == m:
                    mCounter -= 1
                aux[idx][0] = aux[idx+1][0] + 1
                if aux[idx][0] == m:
                    mCounter += 1
                endIdx = idx if aux[idx+1][2] == -1 else aux[idx+1][2]
                aux[idx][2] = endIdx
                aux[idx][1] = 0
                
                aux[endIdx][1] = 0
                aux[endIdx][0] = aux[idx][0]
                
            elif idx == n-1: # Caso 2
                if aux[idx-1][0] == m:
                    mCounter -= 1
                aux[idx][0] = aux[idx-1][0] + 1
                if aux[idx][0] == m:
                    mCounter += 1
                startIdx = idx if aux[idx-1][1] == -1 else aux[idx-1][1]
                aux[idx][1] = startIdx
                aux[idx][2] = n-1
                
                aux[startIdx][2] = n-1
                aux[startIdx][0] = aux[idx][0]
                
            else:
                if aux[idx-1][0] == m:
                    mCounter -= 1
                if aux[idx+1][0] == m:
                    mCounter -= 1
                groupSize = aux[idx+1][0] + aux[idx-1][0] + 1
                if groupSize == m: mCounter += 1

                aux[idx][0] = groupSize
                startIdx = idx if aux[idx-1][1] == -1 else aux[idx-1][1]
                endIdx = idx if aux[idx+1][2] == -1 else aux[idx+1][2]

                aux[idx][1] = startIdx
                aux[idx][2] = endIdx
                
                # Updating first element of group
                aux[startIdx][0] = groupSize
                aux[startIdx][1] = startIdx
                aux[startIdx][2] = endIdx
                
                # Updating last element of group
                aux[endIdx][0] = groupSize
                aux[endIdx][1] = startIdx
                aux[endIdx][2] = endIdx
            
            if mCounter > 0:
                result = i+1

        return result
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        if (m == len(arr)):
            return len(arr)
        
        bit_information = [0] * (len(arr) + 2)
        target_group_size_counter = 0
        ret = -2
        
        
        for i in range(len(arr)):
            total_length = 1 + bit_information[arr[i] - 1] + bit_information[arr[i] + 1]   
            bit_information[arr[i]] = total_length
            
            
            if (total_length != 1):
                target_group_size_counter -= 1 if bit_information[arr[i] - 1] == m else 0
                bit_information[arr[i] - bit_information[arr[i] - 1]] = total_length
                target_group_size_counter -= 1 if bit_information[arr[i] + 1] == m else 0
                bit_information[arr[i] + bit_information[arr[i] + 1]] = total_length
            
            target_group_size_counter += 1 if total_length == m else 0
            ret = i if target_group_size_counter > 0 else ret
            
        return ret + 1       
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        last_step = -1
        n = len(arr)
        left, right = [0] * n, [0] * n
        tmp_arr = [0] * n
        memo = collections.defaultdict(lambda: 0)
        for i, num in enumerate(arr):
            tmp_arr[num - 1] = 1
            left[num - 1] = 1 + (left[num - 2] if num >= 2 else 0)
            right[num - 1] = 1 + (right[num] if num < n else 0)
            
            if num >= 2 and tmp_arr[num - 2]:
                memo[left[num - 2] + right[num - 2] - 1] -= 1
                right[num - 2] += right[num - 1]
                if (num - 1 - (left[num - 1] - 1)) != (num - 2): 
                    right[num - 1 - (left[num - 1] - 1)] += right[num - 1]
                
                
            if num < n and tmp_arr[num]:
                memo[left[num] + right[num] - 1] -= 1
                left[num] += left[num - 1]
                # print("haha", tmp_arr, left, right)
                if (num - 2 + right[num - 1]) != num: 
                    left[num - 2 + right[num - 1]] += left[num - 1]
                
            memo[left[num - 1] + right[num - 1] - 1] += 1
                
            if memo[m] > 0:
                
                last_step = i + 1
            # print(memo, tmp_arr, left, right)
        return last_step
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        groupCount = dict()
        n = len(arr)
        parents = [0] * n
        
        lastM = -1
        mCnt = 0
        for i,p in enumerate(arr):
            leftParent = self.getParent(p - 1, parents)
            rightParent = self.getParent(p + 1, parents)
            parents[p-1] = p    # its own parent
            if leftParent == 0 and rightParent == 0:
                groupCount[p] = 1
                newCnt = 1
            elif leftParent != 0 and rightParent != 0:
                newCnt = groupCount[leftParent] + groupCount[rightParent] + 1
                self.mergeGroups(leftParent, p, parents)
                self.mergeGroups(rightParent, p, parents)
            elif leftParent != 0:
                newCnt = groupCount[leftParent] + 1
                self.mergeGroups(leftParent, p, parents)
            else:
                newCnt = groupCount[rightParent] + 1
                self.mergeGroups(rightParent, p, parents)
            
            if leftParent != 0 and groupCount[leftParent] == m:
                mCnt -= 1
            
            if rightParent != 0 and groupCount[rightParent] == m:
                mCnt -= 1
            
            groupCount[p] = newCnt
            
            if newCnt == m:
                mCnt += 1
            
            if mCnt > 0:
                lastM = i + 1
        
        return lastM
    
    def getParent(self, p: int, parents: List[int]) -> int:
        if p <= 0 or p > len(parents):
            return 0
        
        if p == parents[p-1]:
            return p
        
        return self.getParent(parents[p-1], parents)
    
    def mergeGroups(self, pp, p, parents):
        parents[pp-1] = p

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        ans = -1
        start_set = {}
        end_set = {}
        counter = 0
        for idx, n in enumerate(arr):
            if n - 1 in end_set and n + 1 in start_set:
                st = end_set.pop(n-1)
                ed = start_set.pop(n+1)
                end_set[ed] = st
                start_set[st] = ed
                if n - st == m:
                    counter -= 1
                if ed - n == m:
                    counter -= 1
                if ed - st + 1 == m:
                    counter += 1
            elif n - 1 in end_set:
                st = end_set.pop(n-1)
                end_set[n] = st
                start_set[st] = n
                if n - st == m:
                    counter -= 1
                elif n-st+1 == m:
                    counter += 1
            elif n + 1 in start_set:
                ed = start_set.pop(n+1)
                start_set[n] = ed
                end_set[ed] = n
                if ed - n == m:
                    counter -= 1
                elif ed-n+1 == m:
                    counter += 1
            else:
                start_set[n] = n
                end_set[n] = n
                if m == 1:
                    counter += 1
            if counter > 0:
                ans = idx + 1
        return ans
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        
        if m == len(arr): return m
        length = [0] * (len(arr) + 2)
        res = -1
        for i, v in enumerate(arr):
            left, right = length[v - 1], length[v + 1]
            if left == m or right == m:
                res = i
            length[v - left] = length[v + right] = left + right + 1
        return res
    

class Solution:
    def findLatestStep(self, A: List[int], m: int) -> int:
        if m == len(A): return m
        length = [0] * (len(A) + 2)
        res = -1
        for i, a in enumerate(A):
            left, right = length[a - 1], length[a + 1]
            if left == m or right == m:
                res = i
            length[a - left] = length[a + right] = left + right + 1
        return res
from collections import Counter
class Solution:
    def findLatestStep(self, arr1: List[int], m: int) -> int:
        if m== len(arr1):
          return len(arr1)
        
        def find(arr,i):
          if i== arr[i]: return i
          arr[i] = find(arr,arr[i])
          return arr[i]
        
        def union(arr,i,j,c,mCount):
          u= find(arr,i)
          v= find(arr,j)
          if u != v and v in c and c[v] == m: mCount -=1
          if u != v and u in c and c[u] == m: mCount -=1
          if u != v:
            arr[v] = u
          return u,mCount
            
        ret = -1
        t = [0]*len(arr1)
        par = [i for i in range(len(arr1))]
        c = dict()
        mCount = 0
        for i in range(len(arr1)):
          pri = arr1[i]-1
          t[pri] = 1
          p1, p2 = arr1[i]-2, arr1[i]
          if( p1 < 0 or not t[p1] ) and (p2 >= len(arr1) or not t[p2]):
              c[pri] = 1
              if c[pri] == m: mCount+=1
          if p1 >=0 and t[p1]:
            u, mCount = union(par, p1, pri,c,mCount)
            c[u] += 1
            if  u in c and c[u] == m: mCount +=1 
          if p2 <len(arr1) and p1>=0 and t[p1] and t[p2]:
            u , mCount = union(par, p1, p2,c,mCount)
            c[u] += c[p2]
            if  u in c and c[u] == m: mCount +=1 
            del c[p2]
          elif p2<len(arr1) and t[p2]:
            u,mCount = union (par,pri, p2,c,mCount)
            c[u] = c[p2]+1
            if  u in c and c[u] == m: mCount +=1 
            del c[p2]
          if mCount:
            ret = i+1
        return ret
        
                       

from collections import Counter
class Solution:
    def findLatestStep(self, arr1: List[int], m: int) -> int:
        if m== len(arr1):
          return len(arr1)
        
        def find(arr,i):
          if i== arr[i]: return i
          arr[i] = find(arr,arr[i])
          return arr[i]
        
        def union(arr,i,j,c,mCount):
          u= find(arr,i)
          v= find(arr,j)
          #print(mCount,u,v,c,"9999999")
          if u != v and v in c and c[v] == m: mCount -=1
          if u != v and u in c and c[u] == m: mCount -=1
          if u != v:
            arr[v] = u
          #print(mCount,u,v,c,par)
          return u,mCount
            
        ret = -1
        t = [0]*len(arr1)
        par = [i for i in range(len(arr1))]
        c = dict()
        mCount = 0
        for i in range(len(arr1)):
          pri = arr1[i]-1
          t[pri] = 1
          p1, p2 = arr1[i]-2, arr1[i]
          if( p1 < 0 or not t[p1] ) and (p2 >= len(arr1) or not t[p2]):
              c[pri] = 1
              if c[pri] == m: mCount+=1
          if p1 >=0 and t[p1]:
            u, mCount = union(par, p1, pri,c,mCount)
            c[u] += 1
            if  u in c and c[u] == m: mCount +=1 
          if p2 <len(arr1) and p1>=0 and t[p1] and t[p2]:
            u , mCount = union(par, p1, p2,c,mCount)
            c[u] += c[p2]
            if  u in c and c[u] == m: mCount +=1 
            del c[p2]
          elif p2<len(arr1) and t[p2]:
            u,mCount = union (par,pri, p2,c,mCount)
            c[pri] = c[p2]+1
            if  pri in c and c[pri] == m: mCount +=1 
            del c[p2]
          if mCount:
            ret = i+1
          #print(mCount,";;;",c)
        return ret
        
                       

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        if m == len(arr):
            return m
        
        counts = [0] * (len(arr)+2)
        # m_nums = 0
        ans = -1
        
        for index,i in enumerate(arr):
            left = counts[i-1]
            right = counts[i+1]
            total = 1+left+right
            counts[i] = total
            counts[i-left] = total
            counts[i+right] = total
            
            if left==m or right ==m:
                ans = index
            # if left==m:
            #     m_nums-=1
            # if right == m:
            #     m_nums-=1
            # if total == m:
            #     m_nums+=1
            # if m_nums >0:
            #     ans = index+1
                
        return ans
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        
        
        n = len(arr)
        dp = [1]*(n+1)
        res = -1
        if n == m:
            return n
        
        for i in range(len(arr)-1, -1, -1):
            dp[arr[i]] = 0
            
            j = arr[i]+1
            count = 0
            while j < len(dp) and dp[j]==1:
                count+=1
                if count > m:
                    break
                j+=1
            
            if count == m:
                return i
            
            j = arr[i]-1
            count = 0
            while j >= 1 and dp[j]==1:
                count+=1
                if count > m:
                    break
                j-=1
                
            if count == m:
                return i
        
        
        return -1
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        n = len(arr)
        count = [0]*(n + 1)
        length = [0]*(n + 1)
        p = list(range(1 + n))
        cur = [0]*(n + 2)
        
        def find(x):
            if x != p[x]:
                p[x] = find(p[x])
            return p[x]
        def union(x, y):
            t1, t2 = find(x), find(y)
            a, b = length[t1], length[t2]
            p[t1] = t2
            length[t2] = a + b
            count[a] -= 1
            count[b] -= 1
            count[a + b] += 1
        ans = -1
        for i, x in enumerate(arr):
            #print('in', i, x, cur, length, count)
            cur[x] += 1
            length[x] = 1
            count[1] += 1
            if cur[x - 1]:
                union(x, x - 1)
            if cur[x + 1]:
                union(x, x + 1)
            if count[m]:
                ans = i + 1
        return ans

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        n = len(arr)
        arr.append(n+1)
        start = defaultdict(dict)
        finish = defaultdict(dict)
        last = -1
        for level,i in enumerate(arr):
            if i-1 not in finish: finish[i-1] = i 
            if i+1 not in start: start[i+1] = i

            s, f = finish[i-1], start[i+1]
            start[s] = f 
            finish[f] = s
            
            for os, of in [[i+1, start[i+1]], [finish[i-1], i-1]]:
                if of-os+1 == m: last = level
        
        return last
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        n = len(arr)
        if n == m: return m
        
        fa = [i for i in range(n)]
        sz = [0 for i in range(n)]
        
        def gf(i) -> int:
            if fa[i] != i: fa[i] = gf(fa[i])
            return fa[i]
        
        def merge(x, y):
            fx, fy = gf(x), gf(y)
            if fx != fy:
                if sz[fx] < sz[fy]: 
                    fx, fy = fy, fx
                fa[fy] = fx
                sz[fx] += sz[fy]
        ans = -1
        for i in range(n):
            a = arr[i] - 1
            sz[a] = 1
            for j in (a - 1, a + 1):
                if 0 <= j < n and sz[j] > 0:
                    if sz[gf(j)] == m: ans = i
                    merge(j, a)
        return ans
                    

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        bit_information = [0] * (len(arr) + 1)
        target_group_size_counter = 0
        ret = -2
        
        
        for i in range(len(arr)):
            group_sizes = []
            total_length = 1
            neighbor_group_exists = [False, False]
            
            
            if (arr[i] > 1 and bit_information[arr[i] - 1] != 0):
                total_length += bit_information[arr[i] - 1]
                neighbor_group_exists[0] = True
                
                
            if (arr[i] < len(arr) and bit_information[arr[i] + 1] != 0):
                total_length += bit_information[arr[i] + 1]
                neighbor_group_exists[1] = True
                  
            bit_information[arr[i]] = total_length
            
            
            if (neighbor_group_exists[0]):
                target_group_size_counter -= 1 if bit_information[arr[i] - 1] == m else 0
                bit_information[arr[i] - bit_information[arr[i] - 1]] = total_length
            
            
            if (neighbor_group_exists[1]):
                target_group_size_counter -= 1 if bit_information[arr[i] + 1] == m else 0
                bit_information[arr[i] + bit_information[arr[i] + 1]] = total_length

            target_group_size_counter += 1 if total_length == m else 0
            ret = i if target_group_size_counter > 0 else ret
            
        return ret + 1       
class Solution:
    def isSorted(self, arr):
        for i in range(1, len(arr)):
            if arr[i] < arr[i-1]:
                return False
        return True
    
    def findLatestStep(self, arr: List[int], m: int) -> int:
        length = len(arr)
        if m == length:
            # Full string of 1s can only be found at last step
            return m
        
        if self.isSorted(arr):
            return m
        
        # Pad front and back to make boundary conditions easier
        binary = [0] * (len(arr) + 2)
        latest_step = -1
        
        for step in range(length):
            pos = arr[step]
            binary[pos] = 1
            
            # Examine positions directly to the left and right i.e., group boundaries
            # Find/store the new group size at the new boundaries
            left_len = binary[pos-1]
            right_len = binary[pos+1]
            new_len = left_len + right_len + 1
            
            if left_len == m or right_len == m:
                # Target length persistent until prev step
                latest_step = step
                
            binary[pos-left_len] = new_len
            binary[pos+right_len] = new_len
                
        return latest_step
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        n = len(arr)
        dic = collections.Counter()
        cnt = collections.Counter()
        res = -1
        for i, a in enumerate(arr):
            l = dic[a - 1]
            r = dic[a + 1]
            dic[a - l] = dic[a + r] = dic[a] = l + r + 1
            cnt[l + r + 1] += 1
            cnt[l] -= 1
            cnt[r] -= 1
            if cnt[m]:
                res = i + 1
        return res

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        length = [0] * (len(arr) + 2)
        count = [0] * (len(arr) + 1)
        res = -1
        for i, a in enumerate(arr):
            left, right = length[a - 1], length[a + 1]
            length[a] = length[a - left] = length[a + right] = left + right + 1
            count[left] -= 1
            count[right] -= 1
            count[length[a]] += 1
            if count[m]:
                res = i + 1
        return res
class Solution:
        
    def findLatestStep(self, arr: List[int], m: int) -> int:
        self.mcnt = 0
        par = {}
        sz = {}
        
        def find(i):
            while i != par[i]:
                par[i] = par[par[i]]
                i = par[i]
            return i
        
        def union(i, j):
            x = find(i)
            y = find(j)
            if x == y:
                return
            if sz[x] == m:
                self.mcnt -= 1
            if sz[y] == m:
                self.mcnt -= 1
                    
            if sz[x] <= sz[y]:
                sz[y] += sz[x]
                par[x] = y
                
                if sz[y] == m:
                    self.mcnt += 1
            else:
                sz[x] += sz[y]
                par[y] = x
                
                if sz[x] == m:
                    self.mcnt += 1
        
        count = 1
        ans = -1
        target = set()
        for i in arr:
            if i not in par:
                par[i] = i
                sz[i] = 1
                if m == 1:
                    self.mcnt += 1
                
            if i - 1 in par:
                union(i-1, i)
                
            if i + 1 in par:
                union(i, i+1)
            
            if self.mcnt > 0:
                ans = count
            count += 1
                
        return ans
            

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        n, res = len(arr), -1
        if m == n:
            return n
        if m == 1 and n == 1:
            return 1
        string, size, root = [0] * n, [1] * n, [-1] * n
        
        def find(a):
            if root[a] != a:
                root[a] = find(root[a])
            return root[a]
            
        def union(a, b):
            find_a, find_b = find(a), find(b)
            root[find_a] = find_b
            size[find_b] += size[find_a]
            
        for step in range(n):
            idx = arr[step] - 1
            string[idx] = 1
            root[idx] = idx
            
            # we check the sizes of its two neigbor sets before we merge them with it
            if idx - 1 >= 0 and string[idx - 1] == 1:
                if m == size[find(idx - 1)]:
                    res = step
            if idx + 1 < n and string[idx + 1] == 1:
                if m == size[find(idx + 1)]:
                    res = step
            if idx - 1 >= 0 and string[idx - 1] == 1:
                union(idx - 1, idx)
            if idx + 1 < n and string[idx + 1] == 1:
                union(idx + 1, idx)
        return res
'''
[3,5,1,2,4]
1
[3,1,5,4,2]
2
[1]
1
[2, 1]
2
[3,2,5,6,10,8,9,4,1,7]
3
[4,3,2,1]
1
'''

class Solution:
    def findLatestStep(self, arr, m) -> int:

        n = length = len(arr)
        if m == length:
            return m
        parent = [i for i in range(length)]
        size = [0 for _ in range(length)]

        cur = [0] * length
        ans = -1

        def root(p):
            while p != parent[p]:
                parent[p] = parent[parent[p]]
                p = parent[p]
            return p

        def union(p, q):
            root_p = root(p)
            root_q = root(q)

            if root_p != root_q:
                if size[root_p] > size[root_q]:
                    parent[root_q] = root_p
                    size[root_p] += size[root_q]

                else:
                    parent[root_p] = root_q
                    size[root_q] += size[root_p]

        for idx, i in enumerate(arr):
            i -= 1
            size[i] = 1
            for j in (i - 1, i + 1):
                if 0 <= j < n:
                    if size[root(j)] == m:
                        ans = idx
                    if size[j]:
                        union(i, j)
                    
        return ans
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        length = [0]*(len(arr)+2)
        count=[0]*(len(arr)+1)
        res=-1
        for i,a in enumerate(arr):
            left,right = length[a-1],length[a+1]
            length[a]=length[a-left]=length[a+right]=left+right+1
            count[left]-=1
            count[right]-=1
            count[length[a]]+=1
            if count[m]>0:
                res=i+1
        return res

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        #u4e24u4e2au6570u7ec4uff0cu4e00u4e2au8bb0u5f55u957fu5ea6uff0cu53e6u4e00u4e2au8bb0u5f55u957fu5ea6u7684u4e2au6570
        length = [0 for _ in range(len(arr)+2)]
        count = [0 for _ in range(len(arr)+1)]
        res = -1
        for i, a in enumerate(arr):
            #u5148u628au5de6u8fb9u7684u957fu5ea6u548cu53f3u8fb9u76841u957fu5ea6u53d6u51fau6765
            left, right = length[a-1], length[a+1]
            #u73b0u5728u8fd9u4e2au4f4du7f6eu7684u957fu5ea6u5c31u662fu5de6u8fb9u7684u957fu5ea6u52a0u4e0au53f3u8fb9u7684u957fu5ea6u52a0u4e0au81eau5df1
            #u8dddu79bbau4f4du7f6eu7684u5de6u53f3u4e24u8fb9u7684u8fb9u89d2u5904u7684u7d22u5f15u4e5fu4f1au88abu9644u4e0au65b0u7684u503cuff0cu4e4bu540eu7684u8ba1u7b97u53efu80fdu7528u5f97u4e0a
            length[a] = length[a-left] = length[a+right] = left + right + 1
            
            #u7136u540eu5c31u662fu66f4u65b0countu4e86
            count[left] -= 1
            count[right] -= 1
            count[length[a]] += 1
            
            #u5224u65admu662fu5426u8fd8u5b58u5728uff0cu53eau8981mu5b58u5728u90a3u5c31u662fu6ee1u8db3u6761u4ef6u7684u6700u540eu4e00u6b65
            if count[m]:
                res = i+1
        return res
class Solution:
    def findLatestStep(self, A: List[int], m: int) -> int:
        length = [0] * (len(A) + 2)
        count = [0] * (len(A) + 1)
        res = -1
        for i, a in enumerate(A):
            left, right = length[a - 1], length[a + 1]
            length[a] = length[a - left] = length[a + right] = left + right + 1
            count[left] -= 1
            count[right] -= 1
            count[length[a]] += 1
            if count[m]:
                res = i + 1
        return res
from bisect import bisect_left

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        N = len(arr)
        if m == N: return m
        bits = [(1, N)]
        ans = 0
        for i in range(N-1, -1, -1):
            a = arr[i]
            idx = bisect_left(bits, (a, a))
            if idx < len(bits) and bits[idx][0] == a:
                if bits[idx][1] == a:
                    bits.pop(idx)
                else:
                    s, e = bits[idx]
                    if (e - s) == m: return i
                    bits[idx] = (s+1, e)
                continue
            idx -= 1
            if bits[idx][1] == a:
                bits[idx] = (bits[idx][0], a-1)
                if bits[idx][1] - bits[idx][0] + 1 == m:
                    return i
            else:
                before = (bits[idx][0], a-1)
                after = (a+1, bits[idx][1])
                if (before[1] - before[0] + 1) == m: return i
                if (after[1] - after[0] + 1) == m: return i
                bits[idx:idx+1] = [before, after]
        return -1
class Solution:
    def findLatestStep2(self, arr: List[int], m: int) -> int:
        N = len(arr)
        spans = [(1, N)]
        step = N
        
        if m == N:
            return m
        
        while arr:
            #print(step, spans)
            d = arr.pop()
            step -= 1
            for span in spans:
                if span[0] <= d <= span[1]:
                    if d-span[0] == m or span[1] - d == m:
                        return step
                    
                    spans.remove(span)
                    if d - span[0] > m:
                        spans.append((span[0], d-1))
                    if span[1] - d > m:
                        spans.append((d+1, span[1]))
            
        return -1
    
    def findLatestStep(self, A, m):
        if m == len(A): return m
        length = [0] * (len(A) + 2)
        res = -1
        for i, a in enumerate(A):
            left, right = length[a - 1], length[a + 1]
            if left == m or right == m:
                res = i
            length[a - left] = length[a + right] = left + right + 1
        return res
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        length = len(arr)
        length_to_index = defaultdict(set)
        state = [(0,0) for i in range(length+2)]
        ans = -1
        
        for step,index in enumerate(arr):
            if state[index-1] == (0,0) and state[index+1] == (0,0):
                state[index] = (1,1)
                length_to_index[1].add(index)
            elif state[index+1] == (0,0):
                L = state[index-1][0]+1
                state[index] = (L,1)
                state[index-L+1] = (1,L)
                length_to_index[L-1].remove(index-L+1)
                length_to_index[L].add(index-L+1)
            elif state[index-1] == (0,0):
                L = state[index+1][1]+1
                state[index] = (1,L)
                state[index+L-1] = (L,1)
                length_to_index[L-1].remove(index+1)
                length_to_index[L].add(index)
            else:
                l_left = state[index-1][0]
                l_right = state[index+1][1]
                L = l_left+ l_right + 1
                state[index-l_left] = (1,L)
                state[index+l_right]= (L,1)
                length_to_index[l_right].remove(index+1)

                length_to_index[l_left].remove(index-l_left)
                length_to_index[L].add(index-l_left)
            if length_to_index[m] :
                ans = step + 1
        return ans

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        seen = dict()
        
        ans = -1
        ct = 0
        latestGood = dict()
        for pos in arr:
            ct+=1
            mi, ma = pos, pos
            
            if pos-1 in seen:
                mi = min(seen[pos-1][0], mi)
            
            if pos+1 in seen:
                ma = max(seen[pos+1][1], ma)                                    
            
            seen[pos] = (mi, ma)
            
            if pos-1 in seen:
                seen[pos-1] = (mi, ma)
            
            if pos+1 in seen:
                seen[pos+1] = (mi, ma)
                
            if mi in seen:
                seen[mi] = (mi, ma)
            
            if ma in seen:
                seen[ma] = (mi, ma)
            
            if ma-mi+1==m:
                ans=ct
                
                latestGood[mi] = ma
                latestGood[ma] = mi
            else:                                
                if pos-1 in latestGood:
                    if latestGood[pos-1] in latestGood:
                        latestGood.pop(latestGood[pos-1])
                        
                    if pos-1 in latestGood:
                        latestGood.pop(pos-1)
                    
                if pos+1 in latestGood:
                    if latestGood[pos+1] in latestGood:
                        latestGood.pop(latestGood[pos+1])
                    
                    if pos+1 in latestGood:
                        latestGood.pop(pos+1)
                
                if len(latestGood)>0:
                    ans=ct
            
                
        return ans

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        
        par = {}
        sz = {}
        target = set()
        
        def find(i):
            while i != par[i]:
                par[i] = par[par[i]]
                i = par[i]
            return i
        
        def union(i, j):
            x = find(i)
            y = find(j)
            if x == y:
                return
            if sz[x] <= sz[y]:
                sz[y] += sz[x]
                par[x] = y
                
                if sz[y] == m:
                    target.add(y)
                    
                if sz[x] == m and x in target:
                    target.remove(x)
            else:
                sz[x] += sz[y]
                par[y] = x
                
                if sz[x] == m:
                    target.add(x)
                    
                if sz[y] == m and y in target:
                    target.remove(y)
        
        count = 1
        ans = -1
        target = set()
        for i in arr:
            if i not in par:
                par[i] = i
                sz[i] = 1
                if m == 1:
                    target.add(i)
                
            if i - 1 in par and i + 1 in par:
                union(i-1, i+1)
                union(i-1, i)
            elif i - 1 in par:
                union(i-1, i)
            elif i + 1 in par:
                union(i, i+1)
            
            t = set(target)
            for x in t:
                if sz[x] != m:
                    target.remove(x)
            if len(target):
                ans = count
            count += 1
                
        return ans
            

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        lenArr = [0] * (len(arr) + 2)        
        
        count = 0
        steps = 1
        res = -1
        for k in arr:            
            left, right = lenArr[k-1], lenArr[k+1]
            if lenArr[k-left] == m:
                count -= 1 
                
            if lenArr[k+right] == m:
                count -= 1 
                
            lenArr[k] = lenArr[k-left] = lenArr[k+right] = left + right + 1
            if lenArr[k] == m:                
                count += 1
                
            if count > 0:
                res = steps
            
            steps += 1
            
        return res
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        
        
        def find(parents, u):
            if u != parents[u]:
                parents[u] = find(parents, parents[u])
            
            return parents[u]
        
        def union(parents, ranks, u, v):
            pu = find(parents, u)
            pv = find(parents, v)
            if ranks[pu] >= ranks[pv]:
                parents[pv] = pu
                ranks[pu] += ranks[pv]
            else:
                parents[pu] = pv
                ranks[pv] += ranks[pu]
        
        n = len(arr)
        if n == m:
            return n
        
        laststep = -1
        parents = list(range(n))
        ranks = [0] * (n)
        for i in range(n):
            num = arr[i] - 1
            ranks[num] = 1
            if num-1 >= 0:
                pleft = find(parents, num-1)
                #print('left', num, num-1, pleft)
                if ranks[pleft] == m:
                    laststep = i
                if ranks[pleft] > 0:
                    union(parents, ranks, num-1, num)
            if num+1 < n:
                pright = find(parents, num+1)
                #print('right', num, num+1, pright)
                if ranks[pright] == m:
                    laststep = i
                if ranks[pright] > 0:
                    union(parents, ranks, num+1, num)
            
            #print('p:', parents)
            #print('r:', ranks)
        return laststep
        
        

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        
        n = len(arr)
        if m == n:
            return m
        parent = [-1]*n
        size = [0]*n
        def find(x):
            if parent[x] == -1:
                return x
            parent[x] = find(parent[x])
            return parent[x]
        def union(x,y):
            px = find(x)
            py = find(y)
            if px == py:
                return False
            if size[px] > size[py]:
                size[px] += size[py]
                parent[py] = px
            else:
                size[py] += size[px]
                parent[px] = py
            return True
        

        ans = -1
        
        for step, i in enumerate(arr):
            i -=1
            size[i] = 1
            for j in (i-1, i+1):
                if 0 <= j < n:
                    if size[find(j)] == m:
                        ans = step
                    if size[j]:
                        union(i,j)
                        
                    
        return ans
        
        
        
                
       
        

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        from collections import defaultdict
        cnt = defaultdict(int)
        
        N = len(arr)
        rank = [0] * N
         
        
        res = -1
        for i, n in enumerate(arr):
            n -= 1
            
            if n!=0 and n!=N-1 and rank[n-1] and rank[n+1]:
                cnt[rank[n-1]] -= 1
                cnt[rank[n+1]] -= 1
                r = 1+rank[n-1]+rank[n+1]
                cnt[r] += 1
                rank[n+1+rank[n+1]-1] = r
                rank[n-1-rank[n-1]+1] = r
            
            elif n!=0 and rank[n-1]:
                cnt[rank[n-1]] -= 1
                cnt[rank[n-1]+1] += 1
                rank[n] = rank[n-1] + 1
                rank[n-1-rank[n-1]+1] = rank[n]
                
            elif n!=N-1 and rank[n+1]:
                cnt[rank[n+1]] -= 1
                cnt[rank[n+1]+1] += 1
                rank[n] = rank[n+1] + 1
                rank[n+1+rank[n+1]-1] = rank[n]
            else:
                cnt[1] += 1
                rank[n] = 1
            
            
            if cnt[m]:
                res = i + 1
                
        return res
        

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        A=arr
        if m == len(A): return m
        length = [0] * (len(A) + 2)
        res = -1
        for i, a in enumerate(A):
            left, right = length[a - 1], length[a + 1]
            if left == m or right == m:
                res = i
            length[a - left] = length[a + right] = left + right + 1
            
        return res
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        n = len(arr)
        if n == m:
            return n
        length = [0]*(n+2)
        ans = -1
        for i, a in enumerate(arr):
            left, right = length[a-1], length[a+1]
            if left == m or right == m:
                ans = i
            length[a-left] = length[a+right] = left + right + 1
        return ans

class Solution:
    def findLatestStep(self, A: List[int], T: int, last = -1) -> int:
        seen, ok = set(), set()
        A = [i - 1 for i in A]
        N = len(A)
        P = [i for i in range(N)]
        L = [1] * N
        def find(x):
            if x != P[x]:
                P[x] = find(P[x])
            return P[x]
        def union(a, b):
            a = find(a)
            b = find(b)
            P[b] = a
            L[a] += L[b]
            return L[a]
        step = 1
        for i in A:
            seen.add(i)
            if 0 < i     and find(P[i - 1]) in ok: ok.remove(find(P[i - 1]))
            if i + 1 < N and find(P[i + 1]) in ok: ok.remove(find(P[i + 1]))
            if i - 1 in seen: L[i] = union(i, i - 1)
            if i + 1 in seen: L[i] = union(i, i + 1)
            if L[i] == T:
                ok.add(i)
            if len(ok):
                last = step
            step += 1
        return last
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        # length[i]u8868u793au4f4du7f6eiu7684u8fdeu7eed1u957fu5ea6u4e3au591au957fuff0cu56e0u4e3au6bcfu6b21u90fdu662fu63d2u5165u4e00u4e2au65b0u7684u5143u7d20u4f7fu539fu672cu4e0du8fdeu901au7684u5de6u53f3u8fdeu901a
        # u6240u4ee5u5bf9u4e8eu5de6u8fb9u7684length[i -1]u8868u793au4ee5i-1u4e3au7ed3u675fu7684u8fdeu7eed1u957fu5ea6u6709u591au957f
        # u5bf9u4e8eu53f3u8fb9u7684length[i + 1]u8868u793au4ee5i+1u4e3au5f00u59cbu7684u8fdeu7eed1u957fu5ea6u6709u591au957f
        length = [0] * (len(arr) + 2)
        # count[i]u8868u793au957fu5ea6u4e3aiu7684u5e8fu5217u6709u51e0u4e2a
        count = [0] * (len(arr) + 1)
        res = -1
        for step, i in enumerate(arr):
            left, right = length[i - 1], length[i + 1]
            length[i] = length[i - left] = length[i + right] = left + right + 1
            count[left] -= 1
            count[right] -= 1
            count[length[i]] += 1
            if count[m]:
                res = step + 1
        return res
            

class UF:
    def __init__(self, n, m):
        self.p = [i for i in range(n+1)]  # parent for each position
        self.c = [0 for _ in range(n+1)]  # length of group for each position
        self.m_cnt = 0                    # count of group with length m
        self.m = m                        # m
        
    def union(self, i, j):
        pi, pj = self.find(i), self.find(j)
        if pi != pj:
            if self.c[pi] == self.m: self.m_cnt -= 1  # if previous length at pi is m, decrement m_cnt by 1
            if self.c[pj] == self.m: self.m_cnt -= 1  # if previous length at pj is m, decrement m_cnt by 1
            self.p[pj] = pi                           # union, use pi at parent for pj
            self.c[pi] += self.c[pj]                  # update new length at pi
            if self.c[pi] == self.m: self.m_cnt += 1  # if new length at pi == m, increment m_cnt by 1
            
    def mark(self, i):                                
        self.c[i] = 1                                 # when first visit a point, mark length as 1
        if self.m == 1: self.m_cnt += 1               # if self.m == 1, increment m_cnt by 1
        
    def find(self, i):                                # find parent of i
        if self.p[i] != i:
            self.p[i] = self.find(self.p[i])
        return self.p[i]
    
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        n = len(arr)
        uf, ans = UF(n, m), -1                                   # create union find and answer
        for i, num in enumerate(arr, 1):
            uf.mark(num)
            if num-1 >= 1 and uf.c[num-1]: uf.union(num-1, num)  # if left neighbor is marked, union the two
            if num+1 < n+1 and uf.c[num+1]: uf.union(num+1, num) # if right neighbor is marked, union the two
                
            if uf.m_cnt > 0: ans = i                             # if m_cnt > 0, meaning there exists some group with length m, update ans
        return ans

class UnionFindSet:
    def __init__(self, n):
        self.parents = list(range(n))
        self.ranks = [1] * n
        
    def find(self, u):
        if u != self.parents[u]:
            self.parents[u] = self.find(self.parents[u])
        return self.parents[u]
    
    def union(self, u, v):
        pu, pv = self.find(u), self.find(v)
        if pu == pv:
            return False
        if self.ranks[pu] > self.ranks[pv]:
            self.parents[pv] = pu
            self.ranks[pu] += self.ranks[pv]
        elif self.ranks[pv] > self.ranks[pu]:
            self.parents[pu] = pv
            self.ranks[pv] += self.ranks[pu]
        else:
            self.parents[pu] = pv
            self.ranks[pv] += self.ranks[pu]
        return True
    

class Solution:
    def findLatestStep(self, A: List[int], m: int) -> int:
        length = [0] * (len(A) + 2)
        count = [0] * (len(A) + 1)
        res = -1
        for i, a in enumerate(A):
            left, right = length[a - 1], length[a + 1]
            length[a] = length[a - left] = length[a + right] = left + right + 1
            count[left] -= 1
            count[right] -= 1
            count[length[a]] += 1
            if count[m]:
                res = i + 1
        return res
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        N = len(arr)
        parents = [i for i in range(N + 1)]
        sizes = [1 for i in range(N + 1)]
        
        def ufind(a):
            if parents[a] == a:
                return a
            parents[a] = ufind(parents[a])
            return parents[a]
        
        def uunion(a, b):
            ra = ufind(a)
            rb = ufind(b)
            if ra != rb:
                parents[rb] = parents[ra]
                sizes[ra] += sizes[rb]
                
        def usize(a):
            return sizes[ufind(a)]
        
        ans = -1
        seen = set()
        counter = collections.defaultdict(int)
        
        for i, x in enumerate(arr):
            lft = 0
            if x - 1 > 0 and (x - 1) in seen:
                lft = usize(x - 1)
                counter[lft] -= 1
                uunion(x, x - 1)
                
            rgt = 0
            if x + 1 <= N and (x + 1) in seen:
                rgt = usize(x + 1)
                counter[rgt] -= 1
                uunion(x, x + 1)
                
            grp = lft + 1 + rgt
            counter[grp] += 1
            
            if counter[m] > 0:
                ans = max(ans, i + 1)
                
            seen.add(x)
                
        return ans
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:           
        d = set([0,len(arr) + 1])
        if m == len(arr): return m
        for i in range(len(arr)-1,-1,-1):
            if arr[i] - m - 1 in d:
                exit = True
                for j in range(arr[i] - m , arr[i]):
                    if j in d:
                        exit = False
                        break
                if exit:
                    return i
            if arr[i] + m+1 in d:
                exit = True
                for j in range(arr[i]+1,arr[i]+m+1):
                    if j in d:
                        exit = False
                        break
                if exit:
                    return i
            d.add(arr[i])
        
        return -1
                

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        count = [0] * (len(arr)+1)
        length = [0] * (len(arr)+2)
        res = -1
        for i, n in enumerate(arr):
            left, right = length[n-1], length[n+1]
            length[n] = length[n-left] = length[n+right] = left+right+1
            count[left] -= 1
            count[right] -= 1
            count[left+right+1] += 1
            if count[m] > 0:
                res = i + 1
        return res
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        group_members = {}
        in_group = [-1 for i in range(len(arr))]
        existing_ms = 0
        lastm = -1
        for i, pos in enumerate(arr):
            pos -= 1
            group_at_left = (pos > 0) and (in_group[pos - 1] != -1)
            group_at_right = (pos < len(arr) - 1) and (in_group[pos + 1] != -1)
            len_left = len(group_members[in_group[pos - 1]]) if group_at_left else 0
            len_right = len(group_members[in_group[pos + 1]]) if group_at_right else 0
            if len_left == m:
                existing_ms -= 1
            if len_right == m:
                existing_ms -= 1
            if (not group_at_left) and (not group_at_right):
                in_group[pos] = pos
                group_members[pos] = [pos]
                if m == 1:
                    existing_ms += 1
            elif group_at_left and group_at_right:
                if (len_left + len_right + 1) == m:
                    existing_ms += 1
                merge_group = in_group[pos - 1]
                in_group[pos] = merge_group
                group_members[merge_group].append(pos)
                group_members[merge_group] += group_members[in_group[pos + 1]]
                for pos_right in group_members[in_group[pos + 1]]:
                    in_group[pos_right] = merge_group
            elif group_at_left:
                if (len_left + 1) == m:
                    existing_ms += 1
                merge_group = in_group[pos - 1]
                in_group[pos] = merge_group
                group_members[merge_group].append(pos)
            else:
                if (len_right + 1) == m:
                    existing_ms += 1
                merge_group = in_group[pos + 1]
                in_group[pos] = merge_group
                group_members[merge_group].append(pos)
            if existing_ms > 0:
                lastm = i + 1
        return lastm
class Solution:
    def findLatestStep(self, A, m):
        if m == len(A): return m
        length = [0] * (len(A) + 2)
        res = -1
        for i, a in enumerate(A):
            left, right = length[a - 1], length[a + 1]
            if left == m or right == m:
                res = i
            length[a - left] = length[a + right] = left + right + 1
        return res      
class UF:
    def __init__(self, n, m):
        self.p = [i for i in range(n+1)]  # parent for each position
        self.c = [0 for _ in range(n+1)]  # length of group for each position
        self.m_cnt = 0                    # count of group with length m
        self.m = m                        # m
        
    def union(self, i, j):
        pi, pj = self.find(i), self.find(j)
        if pi != pj:
            if self.c[pi] == self.m: self.m_cnt -= 1  # if previous length at pi is m, decrement m_cnt by 1
            if self.c[pj] == self.m: self.m_cnt -= 1  # if previous length at pj is m, decrement m_cnt by 1
            self.p[pj] = pi                           # union, use pi at parent for pj
            self.c[pi] += self.c[pj]                  # update new length at pi
            if self.c[pi] == self.m: self.m_cnt += 1  # if new length at pi == m, increment m_cnt by 1
            
    def mark(self, i):                                
        self.c[i] = 1                                 # when first visit a point, mark length as 1
        if self.m == 1: self.m_cnt += 1               # if self.m == 1, increment m_cnt by 1
        
    def find(self, i):                                # find parent of i
        if self.p[i] != i:
            self.p[i] = self.find(self.p[i])
        return self.p[i]
    
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        n = len(arr)
        uf, ans = UF(n, m), -1                                   # create union find and answer
        for i, num in enumerate(arr, 1):
            uf.mark(num)
            if num-1 >= 1 and uf.c[num-1]: uf.union(num-1, num)  # if left neighbor is marked, union the two
            if num+1 < n+1 and uf.c[num+1]: uf.union(num+1, num) # if right neighbor is marked, union the two
                
            if uf.m_cnt > 0: ans = i                             # if m_cnt > 0, meaning there exists some group with length m, update ans
        return ans
# 22:00
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        # 1 - n
        ret = -1
        cnt = [[] for _ in range(len(arr))]
        counter = collections.Counter()
        for (i, n) in enumerate(arr):
            n -= 1
            cnt[n].extend([n, n])
            if n - 1 >= 0 and cnt[n - 1]:
                cnt[n][0] = cnt[n - 1][0]
                counter[cnt[n - 1][1] - cnt[n - 1][0] + 1] -= 1
            if n + 1 < len(arr) and cnt[n + 1]:
                cnt[n][1] = cnt[n + 1][1]
                counter[cnt[n + 1][1] - cnt[n + 1][0] + 1] -= 1
            
            cnt[cnt[n][0]][1] = cnt[n][1]
            cnt[cnt[n][1]][0] = cnt[n][0]
            counter[cnt[n][1] - cnt[n][0] + 1] += 1
            
            if counter[m] > 0:
                ret = i + 1
        
        return ret

# 2, 4, 0, 1, 3
# [0, 0] [] [2, 2] [] [4, 4]

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        n = len(arr)
        arr.append(n+1)
        start = {}
        finish = {}
        last = -1
        for level,i in enumerate(arr):
            if i-1 not in finish: finish[i-1] = i 
            if i+1 not in start: start[i+1] = i

            s, f = finish[i-1], start[i+1]
            start[s] = f 
            finish[f] = s
            
            for os, of in [[i+1, start[i+1]], [finish[i-1], i-1]]:
                if of-os+1 == m: last = level
                
            del start[i+1]
            del finish[i-1]
            
        return last
class Solution:
    def findLatestStep(self, arr, k):
        n=len(arr)
        par=[-1]*(n+2)
        count=[0]*(n+1)
        ind=[0]*(n+1)
        ans=-1
        def find(node):
            if par[node]==node:
                return node
            node=par[node]
            return find(node)
        for i in range(n):
            cur=arr[i]
            left=cur-1
            right=cur+1
            par[cur]=cur
            count[cur]+=1
            if par[left]!=-1:
                p=find(left)
                ind[count[p]]-=1
                par[p]=cur
                count[cur]+=count[p]
            if par[right]!=-1:
                p=find(right)
                ind[count[p]]-=1
                par[p]=cur
                count[cur]+=count[p]
            
            ind[count[cur]]+=1
            if ind[k]:
                ans=i+1
        return ans
            

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        n = len(arr)
        A = [0] * n
        B = [[i, i+1] for i in range(n)]
        
        def get_len(i) :
            return B[i][1] - B[i][0]
        def merge(i, j) :
            left = min(B[i][0], B[j][0])
            right = max(B[i][1], B[j][1])
            
            B[left][1] = right
            B[right-1][0] = left
            
            B[i][0] = B[j][0] = left
            B[i][1] = B[j][1] = right
                        
        ret = -1
        for i in range(n) :
            j = arr[i] - 1
            A[j] = 1
            if j and A[j-1] :
                if get_len(j-1) == m :
                    ret = i
                merge(j, j-1)
            if j + 1 < n and A[j+1] :
                if get_len(j+1) == m :
                    ret = i
                merge(j, j+1)
            if B[j][1] - B[j][0] == m :
                ret = i+1
        return ret
        
                

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        
        
        def find(u):
            if parent[u] == u:
                return u
            else:
                parent[u] = find(parent[u])
                return parent[u]
            
        def union(u,v):
            pu = find(u)
            pv = find(v)
            
            if pv!=pu:
                store[size[pv]]-=1
                store[size[pu]]-=1
                size[pu] += size[pv]
                size[pv] = 0
                store[size[pu]]+=1
                parent[pv] = pu
                
            return size[pu]
                
                
        
        n = len(arr)
        
        parent = [0]*n ;size = [0]*n;val = [0]*n
        store = defaultdict(int)
        
        for i in range(n):
            arr[i]-=1
            parent[i] = i
        
        ans = -1
        for i in range(n):
            size[arr[i]] =1
            val[arr[i]] = 1
            store[1]+=1
            curr = 0
            
            if arr[i] - 1 >= 0 and val[arr[i]-1] == 1:
                curr = union(arr[i],arr[i]-1)
                
            if arr[i]+1 < n and val[arr[i]+1] == 1:
                curr = union(arr[i],arr[i]+1)
                
            if store[m] > 0:
                ans = i+1
                
        
        return ans

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        vals = [0 for _ in range(len(arr)+2)]
        numGroups = 0
        res = -1
        for p in range(len(arr)):
            i = arr[p]
            if vals[i-1] == 0 and vals[i+1] == 0:
                vals[i] = 1
                if m == 1:
                    numGroups += 1
            else:
                if vals[i-1] == 0:
                    groupStart = i
                else:
                    groupStart = i - vals[i-1]
                    if vals[i-1] == m:
                        numGroups -= 1
                if vals[i+1] == 0:
                    groupEnd = i
                else:
                    groupEnd = i + vals[i+1]
                    if vals[i+1] == m:
                        numGroups -= 1
                groupLength = groupEnd - groupStart + 1
                vals[groupStart] = vals[groupEnd] = groupLength
                if groupLength == m:
                    numGroups += 1
            if numGroups > 0:
                res = p + 1
        return res
class Solution:
    def findLatestStep(self, A: List[int], m: int) -> int:
        if m == len(A): return m
        length = [0] * (len(A) + 2)
        res = -1
        for i, a in enumerate(A):
            left, right = length[a - 1], length[a + 1]
            if left == m or right == m:
                res = i
            length[a - left] = length[a + right] = left + right + 1
        return res

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        n = len(arr)
        parents = [-1] * n
        ranks = [0]*n
        if m == n:
            return m
        ans = -1
        
        def find(n):
            if parents[n] >= 0:
                parents[n] = find(parents[n])
            else:
                return n       
            return parents[n]
        
        def union(m, n):
            pm, pn = find(m), find(n)
            #print(pm, pn)
            if ranks[pm] > ranks[pn]:
                parents[pn] = pm
                ranks[pm] += ranks[pn]
            else:
                parents[pm] = pn
                ranks[pn] += ranks[pm]
            return True
        visited = set([])
        for i, a in enumerate(arr):
            a -= 1
            ranks[a] = 1
            for j in [a-1, a+1]:
                if 0 <= j < n:
                    if ranks[find(j)] == m:
                        ans = i
                    if j in visited:
                        union(a, j)
                        #print(parents)
                        #print(ranks)
            visited.add(a)
        #print("="*20)
        return ans
                    


    
class Solution:
    def findLatestStep(self, arr, m):
        n = len(arr)
        if n == m:
            return m
        size = [0] * (n + 2)
        res = -1
        for i, x in enumerate(arr):
            if size[x - 1] == m or size[x + 1] == m:
                res = i
            size[x - size[x - 1]] = size[x + size[x + 1]] = size[x - 1] + size[x + 1] + 1

        return res
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        n = len(arr)
        uf = [-1 for i in range(n+1)]
        size = [0 for i in range(n+1)]
        counts = [0 for i in range(n+1)]
        fliped = set()
        
        res = -1
        for i in range(0, len(arr)):
            val = arr[i]
            uf[val] = val
            size[val] =1
            counts[1] +=1
            fliped.add(val)
            if val-1 in fliped:
                self.union(uf, val-1, val, size, counts)
            if val+1 in fliped:
                self.union(uf, val, val+1, size, counts)
            if counts[m] > 0:
                res = max(res, i+1)
        return res
            
    
    def root(self, uf: List[int], a:int)-> int:
        root = a
        while uf[root] != root:
            root = uf[root]

        next = a
        while next != root:
            next = uf[a]
            uf[a] = root
        return root
        
    def union(self, uf: List[int], a: int, b: int, size: List[int], counts: List[int]):
        roota = self.root(uf, a)
        rootb = self.root(uf, b)
    
        
        large = max(roota, rootb)
        small = min(roota, rootb)
        uf[large]=small
        counts[size[large]] -=1
        counts[size[small]] -=1
        
        size[small] = size[large] + size[small]
        counts[size[small]] +=1

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        n = len(arr)
        ids = [i for i in range(n + 1)]
        weights = [0 for i in range(n + 1)]
        size_set = collections.defaultdict(int)
        ans = -1
        
        def find(i: int) -> int:
            while ids[i] != i:
                ids[i] = ids[ids[i]]
                i = ids[i]
            return i
        
        def union(i: int, j: int):
            i_id, j_id = find(i), find(j)
            i_weight, j_weight = weights[i_id], weights[j_id]
            new_weight = weights[i_id] + weights[j_id]
            if weights[i_id] > weights[j_id]:
                weights[i_id] = new_weight
                ids[j_id] = i_id
            else:
                weights[j_id] = new_weight
                ids[i_id] = j_id
            size_set[i_weight] -= 1
            size_set[j_weight] -= 1
            size_set[new_weight] += 1
        
        for i, index in enumerate(arr):
            weights[index] = 1
            size_set[1] += 1
            if index > 1:
                prev_id = find(index - 1)
                if weights[prev_id] > 0:
                    union(prev_id, index)
            if index < n:
                next_id = find(index + 1)
                if weights[next_id] > 0:
                    union(index, next_id)
            if size_set[m] > 0:
                ans = i + 1
        return ans
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        if(len(arr)==1): return 1
        res = [[0 for i in range(2)]]*len(arr)
        mx = -1
        for ind, i in enumerate(arr):
            i-=1
            #Current Element
            current = res[i]
            prev = nxt = 0
            #Last element of previous sequence
            if i-1>-1 and res[i-1][0]==1:
                prev = res[i-1][1]
            #first element and last element of next sequence
            if i<len(res)-1 and res[i+1][0]==1:
                nxt = res[i+1][1]
            
            res[i] = [1,prev + nxt + 1]
            if i-1>-1 and res[i-1][0]==1:
                if res[i-1][1]==m or res[i-res[i-1][1]][1]==m:
                    mx = max(ind, mx)
                res[i-res[i-1][1]][1] = res[i][1]
                res[i-1][1] = res[i][1]
            if i<len(res)-1 and res[i+1][0]==1:
                if res[i+1][1]==m or res[i+res[i+1][1]][1]==m:
                    mx = max(ind, mx)
                res[i+res[i+1][1]][1] = res[i][1]
                res[i+1][1] = res[i][1]
            if res[i][1]==m:
                mx = max(ind+1, mx)
        return mx            
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        parents = [i for i in range(len(arr) + 1)]
        cnt = [1] * (len(arr) + 1)#initial 1
        groupCnt = [0] * (len(arr) + 1)
        rank = [0] * (len(arr) + 1)
        def find(x):
            if x != parents[x]:
                parents[x] = find(parents[x])
            return parents[x]
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                groupCnt[cnt[px]] -= 1
                groupCnt[cnt[py]] -= 1
                cnt[px] = cnt[py] = cnt[px] + cnt[py]
                groupCnt[cnt[px]] += 1
                if rank[px] > rank[py]:
                    parents[py] = px
                elif rank[px] < rank[py]:
                    parents[px] = py
                else:
                    parents[py] = px
                    rank[px] += 1
        visited = [False] * (len(arr) + 1)
        res = -1
        for i, num in enumerate(arr):
            groupCnt[1] += 1
            if num - 1 > 0 and visited[num - 1]:
                union(num, num - 1)
            if num + 1 < len(arr) + 1 and visited[num + 1]:
                union(num, num + 1)
            visited[num] = True
            if groupCnt[m] > 0:
                res = i + 1
        return res
from collections import defaultdict as d
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        #define a range as [l,r]
        left=dict() #left[r]=l
        right=dict() #right[l]=r
        cntLengths=d(lambda :0) #cntLengths[length]=cnts
        ans=-1
        for i in range(len(arr)):
            num=arr[i]
            lower=num
            upper=num
            
            if num+1 in right.keys():
                upper=right[num+1]
                right.pop(num+1)
                cntLengths[upper-num]-=1
            if num-1 in left.keys():
                lower=left[num-1]
                left.pop(num-1)
                cntLengths[num-lower]-=1
            left[upper]=lower
            right[lower]=upper
            cntLengths[upper-lower+1]+=1
            
            if cntLengths[m]>0:
                ans=i+1
        
        return ans
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        '''
        n = len(arr)
        parents = list(range(n))
        ranks = [0] * n
        groupCounts = [0] * (n+1)
        counts = [1] * n
        visited = [False] * n
        
        def find(x):
            if parents[x] != x:
                parents[x] = find(parents[x])
            
            return parents[x]
    
        def union(x, y):
            r1 = find(x)
            r2 = find(y)
            
            if r1 != r2:
                groupCounts[counts[r1]] -= 1
                groupCounts[counts[r2]] -= 1
                counts[r1] = counts[r2] = counts[r1] + counts[r2]
                groupCounts[counts[r1]] += 1
                
                if ranks[r1] >= ranks[r2]:
                    parents[r2] = r1
                    ranks[r1] += 1
                else:
                    parents[r1] = r2
                    ranks[r2] += 1
        
        last = -1
        
        for step, index in enumerate(arr):
            index -= 1
            groupCounts[1] += 1
            if index-1 >= 0 and visited[index-1]:
                union(index, index-1)
            
            if index+1 < n and visited[index+1]:
                union(index, index+1)
            
            visited[index] = True
            
            if groupCounts[m]:
                last = step + 1

        return last
        '''
        n = len(arr)
        length = [0] * (n+2)
        count = [0] * (n+1)
        ans = -1
        
        for step, index in enumerate(arr):
            left = length[index-1]
            right = length[index+1]
            length[index-left] = length[index+right] = left + right + 1
            count[left] -= 1
            count[right] -= 1
            count[length[index-left]] += 1
            
            if count[m]:
                ans = step + 1
        
        return ans
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        steps = [0] * (len(arr) + 2)
        count = defaultdict(int)
        ans = -1
        for i in range(len(arr)):
            c = arr[i]
            l = c - 1
            r = c + 1
            count[steps[l]] -= 1
            count[steps[r]] -= 1
            steps[c] = steps[l - steps[l] + 1] = steps[r + steps[r] - 1] = steps[l] + steps[r] + 1
            count[steps[c]] += 1
            if count[m] > 0:
                ans = i + 1
        return ans

    

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        n = len(arr)
        count = [0] * (n+1)
        length = [0] * ( n + 2)
        ans = -1
        for i,v in enumerate(arr):
            left = length[v - 1]
            right = length[v + 1]
            length[v] = length[v - left] = length[v + right] = left + right + 1
            count[left] -= 1
            count[right] -= 1
            count[length[v]] += 1
            if count[m] >0:
                ans = i + 1
            
        return ans
            
        

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        
        n = len(arr)
        UF = list(range(n + 2))
        SZ = [0] * (n + 2)
        
        def find(x):
            if UF[x] != x:
                UF[x] = find(UF[x])
            return UF[x]
        
        def union(a, b):
            a = find(a)
            b = find(b)
            if a != b:
                if SZ[a] < SZ[b]:
                    a, b = b, a
                UF[b] = a
                SZ[a] += SZ[b]
            
        ans = -1
        cnt = 0
        
        for step, x in enumerate(arr):
            UF[x] = x
            SZ[x] = 1
            cnt -= (SZ[find(x - 1)] == m) + (SZ[find(x + 1)] == m)
                
            if SZ[x - 1]: union(x, x - 1)
            if SZ[x + 1]: union(x, x + 1)

            if SZ[find(x)] == m: cnt += 1

            if cnt > 0: ans = step + 1
                
            # print(step, x, UF, SZ, cnt, ans)
                
        return ans
class Solution:
    


    #def findLatestStep(self, arr: List[int], m: int) -> int:
    def fstep(self, arr, start_idx, end_idx , step, m):
        # u can't hit end idx
        n = end_idx - start_idx
        if n == m:
            return step+1
        turnoff = arr[step]-1
        if turnoff < start_idx or turnoff >= end_idx:
            return self.fstep(arr, start_idx, end_idx , step-1, m)
        
        left = turnoff - start_idx
        right = n - left -1

        

        lr = -1
        rr = -1
        if left >= m:
            lr = self.fstep(arr, start_idx, start_idx+left, step-1, m)
        if right >= m:
            rr = self.fstep(arr, start_idx +left+1, end_idx,step-1, m)
        
        return max(lr,rr) 

    def findLatestStep(self, arr: List[int], m: int) -> int:
        return self.fstep(arr,0, len(arr), len(arr)-1, m)

                
            
            
            

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        groups = {}
        cnt = 0
        result = -1
        for i, a in enumerate(arr):
            b = a - 1
            if b in groups:
                l = groups.pop(b)
                groups[a] = groups[b - l + 1] = l + 1
            else:
                groups[a] = 1
            if groups[a] == m:
                cnt += 1
            elif groups[a] - 1 == m:
                cnt -= 1
            c = a + 1
            if c in groups:
                l = groups.pop(a)
                r = groups.pop(c)
                groups[c + r - 1] = groups[a - l + 1] = l + r
                cnt += (l + r) == m
                cnt -= (l == m) + (r == m)
            if cnt != 0:
                result = i + 1
        return result
# 1562. Find Latest Group of Size M

def find_latest_step (arr, group_size):
    n = len (arr)
    left_end = [-1] * (n + 2)
    right_end = [-1] * (n + 2)
    value = [0] * (n + 2)

    def merge (a, b, removes, appends):
        if value[a] == 0 or value[b] == 0:
            return
        lend, rend = left_end[a], right_end[b]
        removes.append (a - lend + 1)
        removes.append (rend - b + 1)
        appends.append (rend - lend + 1)
        left_end[rend] = lend
        right_end[lend] = rend

    right_size_group_count = 0

    latest_step = -1

    step = 1
    for elem in arr:
        removes = []
        appends = [1]
        value[elem] = 1
        left_end[elem] = elem
        right_end[elem] = elem
        merge (elem - 1, elem, removes, appends)
        merge (elem, elem + 1, removes, appends)
        right_size_group_count += - sum (1 for x in removes if x == group_size) + sum (1 for x in appends if x == group_size)
        if right_size_group_count > 0:
            latest_step = max (latest_step, step)
        step += 1

    return latest_step


class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        return find_latest_step(arr, m)
class Solution:
    
    def find(self, d, x):
        while x != d[x]:
            x = d[x]
        return x
    
    def union(self, d, x, y):
        px, py = self.find(d, x), self.find(d, y)
        if px != py:
            d[px] = py
    
    def findLatestStep(self, arr: List[int], m: int) -> int:
        n, step = len(arr), -1
        s, d, d_len, d_rec = [0]*n, {i:i for i in range(n)}, [1]*n, dict()
        for i in range(n):
            num = arr[i] - 1
            s[num] = 1
            if num-1 >= 0 and s[num-1] == 1:
                temp = d_len[self.find(d, num-1)]
                self.union(d, num-1, num)
                d_rec[temp] -= 1
                d_len[num] += temp
            if num+1 < n and s[num+1] == 1:
                temp = d_len[self.find(d, num+1)]
                self.union(d, num+1, num)
                d_rec[temp] -= 1
                d_len[num] += temp
            d_rec[d_len[num]] = d_rec[d_len[num]]+1 if d_len[num] in list(d_rec.keys()) else 1
            if m in list(d_rec.keys()) and d_rec[m] > 0:
                step = i+1
        return step

from collections import defaultdict

class Solution:
    def merge(self, x, y, f):
        f[y] = x
        t = y
        while f[t] != t:
            t = f[t]
        l = y
        while f[l] != l:
            f[l], l = t, f[l]
        self.d[self.len[t]] -= 1
        self.d[self.len[y]] -= 1
        self.len[t] += self.len[y]
        self.d[self.len[t]] += 1

    def findLatestStep(self, arr: List[int], m: int) -> int:
        self.d = defaultdict(int)
        self.f = list(range(len(arr)))
        self.len = [0] * len(arr)
        state = [0] * len(arr)
        ans = -1
        for i, num in enumerate(arr):
            num = num - 1
            self.len[num] = 1
            self.d[1] += 1
            if num > 0 and state[num - 1] == 1:
                self.merge(num - 1, num, self.f)
            if num < len(arr) - 1 and state[num + 1]:
                self.merge(num, num + 1, self.f)
            state[num] = 1
            if m in self.d and self.d[m] > 0:
                ans = i + 1
        return ans

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        mc = 0
        step = -1
        tuples = {}
        for i in range(len(arr)):
            pos = arr[i]
            minPos, maxPos = pos, pos
            if pos - 1 in tuples:
                minPos = tuples[pos - 1][0]
                if tuples[pos - 1][1] - minPos + 1 == m:
                    mc -= 1
                    if mc == 0:
                        step = i
            if pos + 1 in tuples:
                maxPos = tuples[pos + 1][1]
                if maxPos - tuples[pos + 1][0] + 1 == m:
                    mc -= 1
                    if mc == 0:
                        step = i
            tuples[minPos] = (minPos, maxPos)
            tuples[maxPos] = tuples[minPos]
            if maxPos - minPos + 1 == m:
                mc += 1
        if mc > 0:
            step = len(arr)
        return step
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        count = {}
        for i in range(1,len(arr)+1):
            count[i] = 0
        segment = {}
        res = -1
        for step, a in enumerate(arr):
            #print(count)
            if a-1 in segment and a+1 in segment:
                count[abs(a - 1 - segment[a - 1]) + 1] -= 1
                count[abs(a + 1 - segment[a + 1]) + 1] -= 1
                count[abs(segment[a+1] - segment[a-1]) + 1] += 1
                left = segment[a-1]
                right = segment[a+1]
                del segment[a-1]
                del segment[a+1]
                segment[left] = right
                segment[right] = left
                
            elif a-1 in segment:
                count[abs(a - 1 - segment[a - 1]) + 1] -= 1
                count[abs(a - 1 - segment[a - 1]) + 2] += 1
                left = segment[a-1]
                right = a
                del segment[a-1]
                segment[left] = right
                segment[right] = left
                
            elif a+1 in segment:
                count[abs(a + 1 - segment[a + 1]) + 1] -= 1
                count[abs(a + 1 - segment[a + 1]) + 2] += 1
                left = a
                right = segment[a+1]
                del segment[a+1]
                segment[left] = right
                segment[right] = left
                
            else:
                count[1] += 1
                segment[a] = a
            
            if count[m] > 0:
                
                res = step+1
        #print(count)
        return res
            
            
                
                

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        def helper(start, end, curStep):
            if curStep == 1: return curStep if m == 1 else -1
            
            if end - start + 1 < m: return -1
            
            elif end - start + 1 == m: return curStep
            
            else:    
                idx = arr[curStep - 1]

                if idx < start or idx > end: return helper(start, end, curStep - 1)

                else: return max(helper(start, idx - 1, curStep - 1), helper(idx + 1, end, curStep - 1))
                
        return helper(1, len(arr), len(arr))

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        size = len(arr)
        if m == size:
            return m

        length, ans = [0] * (len(arr) + 2), -1
        for i, n in enumerate(arr):
            left, right = length[n - 1], length[n + 1]
            if left == m or right == m:
                ans = i
            length[n - left] = length[n + right] = (left + right + 1)
        return ans
# class Solution:
#     def findLatestStep(self, arr: List[int], m: int) -> int:
#         arr = arr[::-1]
#         s = "1"*len(arr)
#         step = len(arr)
#         for j in arr:
#             group = s.split("0")
#             group = list(set(group))
#             if "1"*m in group:
#                 return step
#             step -=1
#             s = s[:j-1] + "0" + s[j:]
#         return -1
            
            
class UF:
    def __init__(self, n, m):
        self.p = [i for i in range(n+1)]  # parent for each position
        self.c = [0 for _ in range(n+1)]  # length of group for each position
        self.m_cnt = 0                    # count of group with length m
        self.m = m                        # m
        
    def union(self, i, j):
        pi, pj = self.find(i), self.find(j)
        if pi != pj:
            if self.c[pi] == self.m: self.m_cnt -= 1  # if previous length at pi is m, decrement m_cnt by 1
            if self.c[pj] == self.m: self.m_cnt -= 1  # if previous length at pj is m, decrement m_cnt by 1
            self.p[pj] = pi                           # union, use pi at parent for pj
            self.c[pi] += self.c[pj]                  # update new length at pi
            if self.c[pi] == self.m: self.m_cnt += 1  # if new length at pi == m, increment m_cnt by 1
            
    def mark(self, i):                                
        self.c[i] = 1                                 # when first visit a point, mark length as 1
        if self.m == 1: self.m_cnt += 1               # if self.m == 1, increment m_cnt by 1
        
    def find(self, i):                                # find parent of i
        if self.p[i] != i:
            self.p[i] = self.find(self.p[i])
        return self.p[i]
    
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        n = len(arr)
        uf, ans = UF(n, m), -1                                   # create union find and answer
        for i, num in enumerate(arr, 1):
            uf.mark(num)
            if num-1 >= 1 and uf.c[num-1]: uf.union(num-1, num)  # if left neighbor is marked, union the two
            if num+1 < n+1 and uf.c[num+1]: uf.union(num+1, num) # if right neighbor is marked, union the two
                
            if uf.m_cnt > 0: ans = i                             # if m_cnt > 0, meaning there exists some group with length m, update ans
        return ans
from collections import defaultdict

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        max_time = 0
        n = len(arr)
        
        curr = [0 for _ in range(n)]
        
        parent = [i for i in range(n)]
        size = [1 for i in range(n)]
        
        size_tracker = defaultdict(lambda: 0)
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y, time):
            xs, ys = find(x), find(y)
            if xs == ys: return False
            if size[xs] < size[ys]:
                xs, ys = ys, xs
            size_tracker[size[xs]] -= 1
            size_tracker[size[ys]] -= 1
            size[xs] += size[ys]
            size_tracker[size[xs]] += 1
            if size_tracker[m] > 0: max_time = time + 1
            parent[ys] = xs
            return True
        
        for t in range(n):
            x = arr[t] - 1
            curr[x] = 1
            size_tracker[1] += 1
            if x > 0 and curr[x-1] == 1:
                union(x, x-1, t)
            if x < len(curr)-1 and curr[x+1] == 1:
                union(x, x+1, t)
            if size_tracker[m] > 0:
                max_time = t + 1
                
        return max_time if max_time > 0 else -1
        
        
        
            
            
            
            

class Solution:
    class UnionFind():
        def __init__(self, n):
            self.parents = list(range(n))
            self.sizes = [0] * n
        
        def find(self, i):
            #print(i, self.parents)
            if i != self.parents[i]:
                self.parents[i] = self.find(self.parents[i])
            return self.parents[i]
        
        def union(self, i, j):
            pi = self.find(i)
            pj = self.find(j)
            
            if pi == pj:
                return
            
            if self.sizes[pi] > self.sizes[pj]:
                self.parents[pj] = pi
                self.sizes[pi] += self.sizes[pj]
            else:
                self.parents[pi] = pj
                self.sizes[pj] += self.sizes[pi]
            
            
    def findLatestStep(self, arr: List[int], m: int) -> int:
        n = len(arr)
        last_seen = -1
        if n == m:
            return n
        union_find = self.UnionFind(n)
        
        for step, num in enumerate(arr):
            i = num - 1
            #print(i)
            #print(union_find.parents)
            #print(union_find.sizes)
            union_find.sizes[i] += 1
            for j in [i - 1, i + 1]:
                if 0 <= j < n:
                    pj = union_find.find(j)
                    if union_find.sizes[pj] == m:
                        last_seen = step
                    if union_find.sizes[pj] > 0:
                        union_find.union(i, pj)
        return last_seen

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        counter = collections.defaultdict(int)
        count = collections.defaultdict(int)
        ans = -1
        term = 1
        for i in arr:
            if i - 1 in counter and i + 1 in counter:
                left_most = counter[i - 1]
                right_most = counter[i + 1]
                counter[left_most] = right_most
                counter[right_most] = left_most
                count[right_most - left_most + 1] += 1
                count[i - left_most] -= 1
                count[right_most - i] -= 1
                if i - 1 != left_most:
                    del counter[i - 1]
                if i + 1 != right_most:
                    del counter[i + 1]
                
                
            elif i - 1 in counter:
                left_most = counter[i - 1]
                counter[left_most] = i
                counter[i] = left_most
                count[i - left_most] -= 1
                count[i - left_most + 1] += 1
                if i - 1 != left_most:
                    del counter[i - 1]
            
            elif i + 1 in counter:
                right_most = counter[i + 1]
                counter[right_most] = i
                counter[i] = right_most
                count[right_most - i] -= 1
                count[right_most - i + 1] += 1
                if i + 1 != right_most:
                    del counter[i + 1]
                
            else:
                counter[i] = i
                count[1] += 1
            
            if m in count and count[m] > 0:
                ans = term
            
            term += 1

        return ans
                
                

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
#         groups = ["0" for _ in range(len(arr))]
#         dp = defaultdict(lambda:0)
#         ctn = False
#         pdx = -1
        
#         for i in range(len(arr)): groups[arr[i]-1] = "1"
        
#         for i in range(len(groups)):
#             if groups[i] == "0":
#                 ctn = False
#             else:
#                 if not ctn:
#                     dp[i] += 1
#                     ctn = True; pdx = i
#                 else:
#                     dp[pdx] += 1
        
#         for key in dp:
#             if dp[key] == m: return len(arr)
#         keys = list(sorted(dp.keys()))
#         for i in range(len(arr))[::-1]:
#             idx = bisect.bisect_left(keys, arr[i]-1)
#             if ((0 < idx < len(keys) and keys[idx] != arr[i]-1)) or (idx == len(keys)): idx -= 1
#             key = keys[idx]
#             dif = arr[i]-1 - key
#             if dp[key]-dif-1 != 0:
#                 dp[arr[i]] = dp[key]-dif-1
#                 bisect.insort(keys, arr[i])
#                 if dp[arr[i]] == m: return i
#             dp[key] = dif
#             if dp[key] == m:
#                 return i
#             if dp[key] == 0: del dp[key]
    
#         return -1
        
        #u53cc list
        # length = [0] * (len(arr) + 2)
        # count = [0] * (len(arr) + 1)
        # res = -1
        # for i, a in enumerate(arr):
        #     left, right = length[a - 1], length[a + 1]
        #     length[a] = length[a - left] = length[a + right] = left + right + 1
        #     count[left] -= 1
        #     count[right] -= 1
        #     count[length[a]] += 1
        #     if count[m]:
        #         res = i + 1
        # return res
        
        #u7b80u5355u7248 u901au8fc7u8bb0u5f55u5de6u53f3border
        if m==len(arr): return m
        
        border=[-1]*(len(arr)+2)
        ans=-1
        
        for i in range(len(arr)):
            left=right=arr[i]
            #left = arr[i]-1; right = arr[i]+1
            if border[right+1]>=0: right=border[right+1]
            if border[left-1]>=0: left=border[left-1]
            border[left], border[right] = right, left
            if (right-arr[i]==m) or (arr[i]-left==m):
                ans=i
        
        return ans
        
        
        
        
        
        
        
        
        
        

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        a = [(1, len(arr))]
        if m == len(arr): return len(arr)
        
        def binSearch(ar, num) -> int:
            if len(ar) == 1:
                if ar[0][0] <= num <= ar[0][1]:
                    return 0
                
            lo, hi = 0, len(ar)-1
            while lo <= hi:
                # print(ar, num, lo, hi)
                mid = (lo+hi)//2
                if ar[mid][1] < num:
                    lo = mid+1
                elif num < ar[mid][0]:
                    hi = mid
                elif ar[mid][0] <= num <= ar[mid][1]:
                    return mid
                else:
                    return -1
            return -1
                    
            
        for i, n in enumerate(arr[::-1]):
            idx = binSearch(a, n)
            # print('binSearch', a, n, idx)
            el = a[idx]
            if el[0] == n: # left border
                if el[1] == n: # (1,1)
                    del a[idx]
                else:
                    a[idx] = (n+1, el[1])
            elif el[1] == n: # right border
                if el[0] == n: # (1,1)
                    del a[idx]
                else:
                    a[idx] = (el[0], n-1)
            else: # middle
                a[idx] = (el[0], n-1)
                a.insert(idx+1, (n+1, el[1]))
            # print(a, n, el)
            if n-el[0] == m or el[1]-n == m:
                return len(arr)-i-1
                    
        return -1
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        ans = -1
        da = {}
        db = {}
        n = len(arr)
        s = [0 for i in range(len(arr))]
        cnts = 0
        step = 0
        for e in arr:
            step += 1
            x = e-1
            s[x] = 1
            st = x
            ed = x
            da[x] = (x, x)
            db[x] = (x, x)
            if x > 0 and s[x-1] == 1:
                p = db[x-1]
                if (p[1]+1-p[0] == m):
                    cnts -= 1
                st = p[0]
                da[st] = (st, ed)
                del db[p[1]]
                db[ed] = (st, ed)
                del da[x]
            if x < n-1 and s[x+1] == 1:
                q = da[x+1]
                if (q[1]+1-q[0] == m):
                    cnts -= 1
                ed = q[1]
                da[st] = (st, ed)
                del da[q[0]]
                db[ed] = (st, ed)
                del db[x]

            if (ed+1-st) == m:
                cnts += 1
            if cnts > 0:
                ans = step

        return ans
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        count = [0] * (len(arr) + 2)
        length = [0] * (len(arr) + 2)
        ans = -1
        for i, a in enumerate(arr):
            left, right = length[a-1], length[a+1]
            length[a] = left + right + 1
            length[a - left] = length[a]
            length[a + right] = length[a]
            count[left] -= 1
            count[right] -= 1
            count[left + right + 1] += 1
            if count[m] > 0:
                ans = i + 1
        return ans        

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        if len(arr) == m : return m
        res = -1
        ln = [0] * (len(arr) + 2)
        
        for i, a in enumerate(arr):
            left, right = ln[a - 1], ln[a + 1]
            if left == m or right == m : res = i
            ln[a - left] = ln[a + right] = left + right + 1
            
        return res
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        
        def find(node):
            if parent[node] < 0:
                return node
            else:
                return find(parent[node])
    
        def union(n1,n2):
            p1 = find(n1)
            p2 = find(n2)
            p1_size = abs(parent[p1])
            p2_size = abs(parent[p2])
            if p1_size == m or p2_size == m:
                ans = count
            if p1_size < p2_size:
                tmp = parent[p1]
                parent[p1] = p2
                parent[p2] += tmp
            else:
                tmp = parent[p2]
                parent[p2] = p1
                parent[p1] += tmp
        
        n = len(arr)
        ans = -1
        
        parent = [-1]*(n+1)
        bitvalue = [0]*(n+1)
        for count,i in enumerate(arr,1):
            if i+1 <=n and bitvalue[i+1] == 1 and i-1 > 0 and bitvalue[i-1] == 1:
                if abs(parent[find(i+1)]) == m or abs(parent[find(i-1)]) == m:
                    ans = count-1
                union(i,i+1)
                union(i,i-1)
            elif i+1 <= n and bitvalue[i+1] == 1:
                if abs(parent[find(i+1)]) == m:
                    ans = count-1
                union(i,i+1)
            elif i-1 > 0 and bitvalue[i-1] == 1:
                if abs(parent[find(i-1)]) == m:
                    ans = count-1
                union(i,i-1)
            bitvalue[i] = 1
            if abs(parent[find(i)]) == m:
                ans = count
            # print(parent)
        return ans
                

class Solution:
    def findLatestStep(self, a: List[int], m: int) -> int:
        index2len, len_cnt, last_index = defaultdict(int), Counter(), -1        
        for i, p in enumerate(a):    
            left_len, right_len = index2len[p-1], index2len[p+1]
            new_len = left_len + 1 + right_len
            index2len[p-left_len] = index2len[p+right_len] = new_len
            len_cnt[left_len] -= 1
            len_cnt[right_len] -= 1                
            len_cnt[new_len] += 1 
            if len_cnt[m] > 0: last_index = i + 1            
        return last_index
class Solution:
    def findLatestStep(self, A, m):
        length = [0] * (len(A) + 2)
        count = [0] * (len(A) + 1)
        res = -1
        for i, a in enumerate(A):
            left, right = length[a - 1], length[a + 1]
            length[a] = length[a - left] = length[a + right] = left + right + 1
            count[left] -= 1
            count[right] -= 1
            count[length[a]] += 1
            if count[m]:
                res = i + 1
        return res
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        alls=[]
        n=len(arr)
        alls.append([1,n])
        
        count=n
        
        if m==n:
            return n
        
        for j in range(len(arr)):
            a=arr[n-j-1]
            
            count-=1
            for i in range(len(alls)):
                if a>=alls[i][0] and a<=alls[i][1]:
                    left=[alls[i][0],a-1]
                    right=[a+1,alls[i][1]]
                    
                    del alls[i]
                    
                    if left[1]-left[0]==m-1:
                        return count
                    
                    if right[1]-right[0]==m-1:
                        return count
                    
                    if left[1]>=left[0] and left[1]-left[0]>m-1:
                        alls.append(left)
                    if right[1]>=right[0] and right[1]-right[0]>m-1:
                        alls.append(right)
                        
                    break
            #print(alls)
                        
        
        return -1
                        
                    
                

class UF:
    def __init__(self, n, m):
        self.p = [i for i in range(n+1)]
        self.c = [0 for _ in range(n+1)]
        self.m_cnt = 0
        self.m = m
        
    def union(self, i, j):
        pi, pj = self.find(i), self.find(j)
        if pi != pj:
            self.p[pj] = pi
            if self.c[pj] == self.m: self.m_cnt -= 1
            if self.c[pi] == self.m: self.m_cnt -= 1
            #self.c[pj] += self.c[pi]
            self.c[pi] += self.c[pj]
            if self.c[pi] == self.m: self.m_cnt += 1 
            
    def mark(self, i):
        self.c[i] = 1
        if self.c[i] == self.m: self.m_cnt += 1
        
    def find(self, i):
        if self.p[i] != i:
            self.p[i] = self.find(self.p[i])
        return self.p[i]
    
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        n = len(arr)
        uf = UF(n, m)
        ans = -1
        for i, num in enumerate(arr, 1):
            uf.mark(num)
            if num-1 >= 1 and uf.c[num-1]:
                uf.union(num-1, num)
            if num+1 < n+1 and uf.c[num+1]:
                uf.union(num+1, num)
            if uf.m_cnt > 0:    
                ans = i
        return ans                
class Solution:
    def findLatestStep(self, A, m):
        length = [0] * (len(A) + 2)
        count = [0] * (len(A) + 1)
        res = -1
        for i, a in enumerate(A):
            left, right = length[a - 1], length[a + 1]
            length[a] = length[a - left] = length[a + right] = left + right + 1
            count[left] -= 1
            count[right] -= 1
            count[length[a]] += 1
            if count[m]:
                res = i + 1
        return res        
from collections import defaultdict
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        intervals = {}
        lengths = defaultdict(int)
        
        found = -1
        for i, val in enumerate(arr):
            if (val-1) not in intervals and (val+1) not in intervals:
                intervals[val] = val
                lengths[1] += 1
                
            if (val-1) in intervals and (val+1) not in intervals:
                prev_left = intervals[val-1]
                prev_right = val-1
                lengths[prev_right - prev_left + 1] -= 1
                lengths[prev_right - prev_left + 2] += 1
                
                intervals[prev_left] = val
                intervals[val] = prev_left
                
            if (val-1) not in intervals and (val+1) in intervals:
                prev_right = intervals[val+1]
                prev_left = val+1
                lengths[prev_right - prev_left + 1] -= 1
                lengths[prev_right - prev_left + 2] += 1
                
                intervals[prev_right] = val
                intervals[val] = prev_right
            
            if (val-1) in intervals and (val+1) in intervals:
                prev_right = intervals[val+1]
                prev_left = intervals[val-1]
                
                lengths[prev_right-val] -= 1
                lengths[val-prev_left] -= 1
                lengths[prev_right-prev_left+1] += 1
                
                intervals[prev_left] = prev_right
                intervals[prev_right] = prev_left
                if val+1 != prev_right:
                    del intervals[val+1]
                if val-1 != prev_left:
                    del intervals[val-1]
                    
            if lengths[m] != 0:
                found = i+1
        
        return found
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        N    = len(arr)
        par  = [0] * N
        rank = [0] * N
        sz   = [0] * N
        ops = {}
        
        for n in range(N):
            par[n] = n
        
        def find(u):
            if par[u] == u:
                return u
            par[u] = find(par[u])
            return par[u]
        
        def merge(u, v):
            up, vp = find(u), find(v)
            
            if rank[up] < rank[vp]:
                par[up] = vp
                sz[vp] += sz[up]
                sz[up] = 0    
            elif rank[vp] < rank[up]:
                par[vp] = up
                sz[up] += sz[vp]
                sz[vp] = 0
            else:
                par[up] = vp
                sz[vp] += sz[up]
                rank[vp] += 1
                sz[up] = 0
                
        
        snap = [0] * (N)
        last = -1
        
        for p, n in enumerate(arr):
            n -= 1
            snap[n] = 1
            sz[n] = 1
            mark = False
            
            if (n - 1) >= 0 and snap[n - 1] == 1:
                p1 = find(n - 1)
                if p1 in ops:
                    del ops[p1]
                mark = True
                merge(n - 1, n)
                
            if n + 1 < N and snap[n + 1] == 1:
                p1 = find(n + 1)
                if p1 in ops:
                    del ops[p1]
                mark = True
                merge(n, n + 1)
            
            para = find(n)
            if sz[para] == m:
                ops[para] = 1
            
            if ops:
                last = p
        
        if last == -1:
            return -1
        return last + 1
            
            

from collections import defaultdict
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        s_d = defaultdict(list)
        e_d = defaultdict(list)
        lengths = defaultdict(int)
        latest = -1
        step = 1
        for i in arr:
            
            end = i - 1
            start = i + 1
            
            if end in e_d and start in s_d:
                s_l = s_d[start]
                e_l = e_d[end]
                x = s_l[1] - s_l[0] + 1
                y = e_l[1] - e_l[0] + 1
                lengths[x] -= 1
                lengths[y] -= 1
                if lengths[x] == 0:
                    del lengths[x]
                if lengths[y] == 0:
                    del lengths[y]
                del s_d[start]
                del e_d[end]
                
                l = [e_l[0],s_l[1]]
                length = l[1] - l[0] + 1                
                
                lengths[length] += 1
                s_d[l[0]] = l 
                e_d[l[1]] = l
            elif end in e_d:
                e_l = e_d[end]
                x = e_l[1] - e_l[0] + 1 
                lengths[x] -= 1
                if lengths[x] == 0:
                    del lengths[x]
                    
                del e_d[end]
                
                e_l[1] = i 
                e_d[e_l[1]] = e_l
                length = e_l[1] - e_l[0] + 1
                lengths[length] += 1
                
            elif start in s_d:
                s_l = s_d[start]
                x = s_l[1] - s_l[0] + 1 
                lengths[x] -= 1 
                if lengths[x] == 0:
                    del lengths[x]
                    
                del s_d[start]
                s_l[0] = i 
                s_d[i] = s_l
                length = s_l[1] - s_l[0] + 1
                lengths[length] += 1
            else:
                
                l = [i,i]
                s_d[i] = l 
                e_d[i] = l 
                
                lengths[1] += 1
            # print(i,s_d,lengths)
            if m in lengths:
                latest = step
            step += 1
        return latest
                

from collections import defaultdict

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        n = len(arr)
        bits = [0] * n
        
        groups = defaultdict(int)
        
        
        corner = defaultdict(int)
        
        result = -1
        for step, i in enumerate(arr):
            i = i -1
            bits[i] = 1
            group = 1
            
            j = i-1
            group += corner[j]
            groups[corner[j]] -= 1
                
            j = i+1
            group += corner[j]
            groups[corner[j]] -= 1
            
            # print(corner)
            
            groups[group] += 1
            corner[i - corner[i-1]] = group
            corner[i + corner[i+1]] = group
            
            # print(corner)
            
            # print(groups)
            
            if groups[m]:
                result = step
                
        return result + 1 if result >= 0 else -1
                

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        
        n = len(arr)
        
        rank = [0 for i in range(n + 1)]
        fa  =  [i for i in range(n + 1)]
        def getfa(x):
            if fa[x] == x: return x
            fa[x] = getfa(fa[x])
            return fa[x]
        
        def union(a, b):
            p, q = getfa(a), getfa(b)
            if p != q:
                if rank[p] >= rank[q]:
                    fa[q] = p
                    rank[p] += rank[q]
                    return p
                else:
                    fa[p] = q
                    rank[q] += rank[p]
                    return q
                
            return False
        
        cc = Counter()
        last = -1
        for i, num in enumerate(arr):
            l, r = num - 1, num + 1
            rank[num] = 1
            cc[1] += 1
            if rank[l]:
                rl = getfa(l)
                cc[1] -= 1
                cc[rank[rl]] -= 1
                newroot = union(num, rl)
                cc[rank[newroot]] += 1
            if r <= n and rank[r]:
                rl = getfa(num)
                cc[rank[rl]] -= 1
                rr = getfa(r)
                cc[rank[rr]] -= 1
                newroot = union(rl, rr)
                cc[rank[newroot]] += 1
            if cc[m] > 0:
                last = i + 1
                
        return last
                
                
                

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        n = len(arr)
        self.parent = [i for i in range(n + 1)]
        self.rank = [1 for _ in range(n + 1)]
        self.result = 0
        self.groups = set()
        def find_parent(a):
            if a != self.parent[a]:
                self.parent[a] = find_parent(self.parent[a])
            return self.parent[a]
        
        def union(a, b, check=False):
            parent_a = find_parent(a)
            parent_b = find_parent(b)
            
            if parent_a == parent_b:
                if self.rank[parent_a] == m:
                    self.groups.add(parent_a)
                return
            if self.rank[parent_a] < self.rank[parent_b]:
                parent_a, parent_b = parent_b, parent_a
            self.parent[parent_b] = parent_a
            self.rank[parent_a] += self.rank[parent_b]
            
            if parent_a in self.groups:
                self.groups.remove(parent_a)
            if parent_b in self.groups:
                self.groups.remove(parent_b)
            
            if check:
                if self.rank[parent_a] == m:
                    self.groups.add(parent_a)
                
        self.binary = [0 for _ in range(n + 2)]
        result = -1
        for idx in range(n):
            num = arr[idx]
            if self.binary[num-1] == 1 and self.binary[num + 1] == 1:
                union(num-1, num)
                union(num, num + 1, True)
                #print(self.rank[num-1], self.rank[num], self.rank[num + 1])
                
                
            elif self.binary[num - 1] == 1:
                union(num, num - 1, True)
            elif self.binary[num + 1] == 1:
                union(num, num + 1, True)
            else:
                union(num, num, True)
            if len(self.groups) > 0:
                result = idx + 1
            self.binary[num] = 1
            #print(self.groups, self.binary, self.parent)
            
        return result 
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        dp = [(-1, -1)] * (len(arr) + 1)
        groups = collections.defaultdict(int)
        if len(arr) == 1:
            return 1
        res = -1
        for i, a in enumerate(arr):
            leftPos, rightPos = a, a
            if a == 1:
                rightPos = a if dp[a+1][1] == -1 else dp[a+1][1]
            elif a == len(arr):
                leftPos = a if  dp[a-1][0] == -1 else dp[a-1][0]
            else:
                leftPos = a if  dp[a-1][0] == -1 else dp[a-1][0]
                rightPos = a if dp[a+1][1] == -1 else dp[a+1][1]
            
            dp[a] = (leftPos, rightPos)
            dp[leftPos] = (leftPos, rightPos)
            dp[rightPos] = (leftPos, rightPos)

            groups[rightPos - leftPos + 1] += 1
            groups[a - leftPos] -= 1
            groups[rightPos - a] -= 1
            if groups[m] >= 1:
                res = i + 1
        return res

        # length = [0] * (len(arr) + 2)
        # groups = [0] * (len(arr) + 1)
        # res = -1
        # for i, a in enumerate(arr):
        #     left, right = length[a - 1], length[a + 1]
        #     total = left + right + 1
        #     length[a] = length[a - left] = length[a + right] = total
        #     groups[left] -= 1
        #     groups[right] -= 1
        #     groups[total] += 1
        #     if groups[m] > 0:
        #         res = i + 1
        # return res

class Solution:
    
    def find_root(self, arr, idx):
        assert arr[idx][0] != -1
        while(arr[idx][0] != idx):
            idx = arr[idx][0]
        return idx
        
    def findLatestStep(self, arr: List[int], m: int) -> int:
        tree_root = [[-1,-1] for i in range(len(arr))]
        
        if m == len(arr):
            return m
        
        last_t = -1
        for i in range(len(arr)):
            bit_idx = arr[i] - 1
            for j in range(bit_idx-1, bit_idx+2):
                if 0 <= j < len(arr):
                    if tree_root[j][0] != -1 and tree_root[self.find_root(tree_root, j)][1] == m:
                        last_t = i
                        
            tree_root[bit_idx][0] = bit_idx
            tree_root[bit_idx][1] = 1
            if bit_idx > 0 and tree_root[bit_idx-1][0] != -1:
                left_node_root = self.find_root(tree_root, bit_idx-1)
                tree_root[left_node_root][0] = bit_idx
                tree_root[bit_idx][1] += tree_root[left_node_root][1]
            if bit_idx < len(arr)-1 and tree_root[bit_idx+1][0] != -1:
                right_node_root = self.find_root(tree_root, bit_idx+1)
                tree_root[right_node_root][0] = bit_idx
                tree_root[bit_idx][1] += tree_root[right_node_root][1]
            
            # for j in range(len(arr)):
            #     if (tree_root[j][0] == j and tree_root[j][1] == m):
            #         last_t = i + 1
            
            # if (tree_root[bit_idx][1] == m):
            #     last_t = i + 1
        
        return last_t
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
#         n = len(arr)
#         groups = defaultdict(set)
#         parents = [i for i in range(n)]
#         size = [0] * n
        
#         def find(node):
#             if parents[node] == node:
#                 return node
#             parent = find(parents[node])
#             return parent
        
#         def union(a, b):
#             para = find(a)
#             parb = find(b)
#             if para != parb:
#                 groups[parb].update(groups[para])
#                 groups.pop(para)
#                 parents[para] = parb
                
#         def get_size(a):
#             parent = find(parents[a])
#             return len(groups[parent])
        
#         def update(i):
#             check = get_size(i)
#             sizes[check] -= 1
#             if sizes[check] == 0:
#                 sizes.pop(check)
        
#         arr = [i-1 for i in arr]
#         step = 0
#         ans = -1
#         sizes = Counter()
#         for i in arr:
#             step += 1
#             size[i] += 1
#             groups[i].add(i)
#             sizes[1] += 1
#             if i-1 >= 0 and i+1 < n and size[i-1] and size[i+1]:
#                 update(i-1)
#                 update(i+1)
#                 update(i)
#                 union(i, i-1)
#                 union(i+1, i-1)
#                 new_size = get_size(i-1)
#                 sizes[new_size] += 1
#             elif i-1 >= 0 and size[i-1]:
#                 update(i-1)
#                 update(i)
#                 union(i, i-1)
#                 new_size = get_size(i-1)
#                 sizes[new_size] += 1
#             elif i+1 < n and size[i+1]:
#                 update(i+1)
#                 update(i)
#                 union(i, i+1)
#                 new_size = get_size(i+1)
#                 sizes[new_size] += 1
#             if m in sizes:
#                 ans = step
#         return ans
        N = len(arr)
        arr = [x - 1 for x in arr]
        parent = [x for x in range(N)]
        size = [1 for _ in range(N)]
        used = [False for _ in range(N)]
        
        def ufind(a):
            if a == parent[a]:
                return a
            parent[a] = ufind(parent[a])
            return parent[a]
        
        def uunion(a, b):
            sa = ufind(a)
            sb = ufind(b)
            
            if sa != sb:
                parent[sa] = parent[sb]
                size[sb] += size[sa]
        
        def usize(a):
            return size[ufind(a)]
            
        counts = [0] * (N+1)
        
        latest = -1
        for index, x in enumerate(arr):
            left = 0
            if x - 1 >= 0 and used[x - 1]:
                left = usize(x - 1)
                
            right = 0
            if x + 1 < N and used[x + 1]:
                right = usize(x + 1)
                
            current = 1
            counts[1] += 1
            if left > 0:
                counts[left] -= 1
            if right > 0:
                counts[right] -= 1
            counts[1] -= 1
            used[x] = True
            
            new_size = left + right + current
            #print(x, left, right)
            counts[new_size] += 1
            if left > 0:
                uunion(x, x - 1)
            if right > 0:
                uunion(x, x + 1)
            
            #print(counts)
            if counts[m] > 0:
                latest = max(latest, index + 1)
        return latest
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        # [3,1,5,4,2]
        uf = UF(len(arr) + 1)
        res, step = -1, 1
        ok = set()
        for i in arr:
            uf.isOne[i] = True
            if i - 1 > 0 and uf.isOne[i - 1]:
                uf.union(i - 1, i)
            if i + 1 <= len(arr) and uf.isOne[i + 1]:
                uf.union(i, i + 1)
            curok = set()
            curf = uf.find(i)
            curones = uf.rank[curf]
            if curones == m:
                curok.add(curf)
                
            for f in ok:
                newf = uf.find(f)
                if uf.rank[newf] == m:
                    curok.add(newf)
            ok = curok
            if len(ok) > 0:
                res = step
            step += 1
        return res
        
        
        
        
class UF:
    def __init__(self, n):
        self.n = n
        self.fathers = [i for i in range(n)]
        self.rank = [1 for _ in range(n)]
        self.isOne = [False for _ in range(n)]
        
    def find(self, x):
        if x != self.fathers[x]:
            self.fathers[x] = self.find(self.fathers[x])
        return self.fathers[x]
    
    def union(self, x, y):
        fx, fy = self.find(x), self.find(y)
        if fx != fy:
            count = self.rank[fx] + self.rank[fy]
            if self.rank[fx] > self.rank[fy]:
                self.fathers[fx] = fy
                self.rank[fy] = count
            else:
                self.fathers[fy] = fx
                self.rank[fx] = count
    
        

class Solution:
    def findLatestStep(self, arr: List[int], target: int) -> int:
        b = [0] * len(arr)
        m = {}
        c = {}
        res = -1
        step = 1
        for i in arr:
            i -= 1
            newl = i
            newr = i
            if i >= 1 and b[i-1] == 1:
                l = i - m[i-1]
                newl = m[i-1]
                del m[m[i-1]]
                if i-1 in m:
                    del m[i-1]
                c[l] -= 1
            if i < len(arr) - 1 and b[i+1] == 1:
                l = m[i+1] - i
                newr = m[i+1]
                del m[m[i+1]]
                if i+1 in m:
                    del m[i+1]
                c[l] -= 1
            m[newl] = newr
            m[newr] = newl
            l = newr - newl + 1
            c[l] = c.get(l, 0) + 1
            b[i] = 1
            if c.get(target, 0):
                res = step
            step += 1
        return res
        

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        N = len(arr)
        parent = [x for x in range(N)]
        size = [1 for _ in range(N)]
        used = [False for _ in range(N)]
        
        def ufind(a):
            if a == parent[a]:
                return a
            parent[a] = ufind(parent[a])
            return parent[a]
        
        def uunion(a, b):
            sa = ufind(a)
            sb = ufind(b)
            
            if sa != sb:
                parent[sa] = parent[sb]
                size[sb] += size[sa]
        
        def usize(a):
            return size[ufind(a)]
            
        counts = [0] * (N+1)
        
        latest = -1
        for index, X in enumerate(arr):
            x = X - 1
            left = 0
            if x - 1 >= 0 and used[x - 1]:
                left = usize(x - 1)
                
            right = 0
            if x + 1 < N and used[x + 1]:
                right = usize(x + 1)
                
            current = 1
            counts[1] += 1
            if left > 0:
                counts[left] -= 1
            if right > 0:
                counts[right] -= 1
            counts[1] -= 1
            used[x] = True
            
            new_size = left + right + current
            counts[new_size] += 1
            if left > 0:
                uunion(x, x - 1)
            if right > 0:
                uunion(x, x + 1)
            
            if counts[m] > 0:
                latest = max(latest, index + 1)
        return latest
class Solution:
    # https://leetcode.com/problems/find-latest-group-of-size-m/discuss/806786/JavaC%2B%2BPython-Count-the-Length-of-Groups-O(N)
    # https://www.youtube.com/watch?v=2NCfiCpv1OA
    def findLatestStep(self, arr: List[int], m: int) -> int:
        if len(arr) == m:
            return m
        # number [0, n + 1]
        length = [0 for _ in range(len(arr) + 2)]
        res = -1
        # n starts from 1.
        for i, n in enumerate(arr):
            left, right = length[n - 1], length[n + 1]
            if left == m or right == m:
                #update res for each time satisfying conditiong. so return the latest one.
                res = i
            # update edge. [0111010], change middle 0 t0 1. left = 3, right = 1.total length = 3 + 1 + 1 = 5. edge, length[1] = 5, length[6] = 5
            length[n - left] = length[n + right] = left + right + 1
        return res
            

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        if len(arr) == m:
            return m
        result = len(arr) - self.helper(arr,1,len(arr),len(arr)-1,m)
        
        if result < 0: 
            return -1
        return result
    
    
    
    def helper(self,arr,left,right,index,m):
        val=arr[index]
        steps= 1
        
        while val > right or val < left:
            steps+=1
            if index -1 >= 0:
                index -=1 
                val=arr[index]
            else:
                return float('inf')
        
        if val-left  == m or  right-val == m:
            return steps 
        
        # print(left,val,right,index,m,steps)
        
        left_bound =  self.helper(arr,left,val -1,index-1,m) if (val-left > m)  else float('inf')
        right_bound = self.helper(arr,val +1,right,index-1,m) if (right-val > m) else float('inf')
        
        # print(left_bound,right_bound,left,right)
        return steps + min(left_bound,right_bound)

class Solution:
    def findLatestStep(self, A: List[int], m: int) -> int:
        L, R, C, res = {}, {}, 0, -1
        for i, x in enumerate(A):
            l = r = x
            if x-1 in L:
                l = L[x-1]
                if x-l == m:
                    C -= 1
                del L[x-1]
            if x+1 in R:
                r = R[x+1]
                if r-x == m:
                    C -= 1
                del R[x+1]
            R[l], L[r] = r, l
            if r-l+1 == m:
                C += 1
            if C:
                res = i+1
        return res

class Solution:
    def find(self,node1: int, group:List[int]) -> int:
        head  =  node1
        while head != group[head]:
            head = group[head]
        
        # path compression
        # while node1 != group[node1]:
        #     next_node = group[node1]
        #     group[node1] = head 
        #     node1 = next_node
        return head
        
    def union(self,node1: int, node2:int, group:List[int], size:dict) -> int:
        # find head for both node 1 and node 2
        head1 , head2 = self.find(node1,group) , self.find(node2,group)
        
        if head1 == head2:
            return head1
        
        if size[head1] < size[head2]:
            # merge head 1 into head2
            group[head1] = head2
            size[head2] += size[head1]
            size.pop(head1)
            return head2

        # merge 2 into one
        group[head2] = head1
        size[head1] += size[head2]
        size.pop(head2)
        return head1
    
    def findLatestStep(self, arr: List[int], m: int) -> int:
        if m == len(arr):
            return m
        group = [-1 for _ in range(len(arr))]
        sizes = {}
        ans = -1
        arr_size= len(arr)
        for ix, val in enumerate(arr):
            if group[val-1] != -1:
                # already visited
                continue
            
            # first time visiting
            group[val-1] = val-1
            sizes[val-1] = 1
            for neighbor in (val-2,val):
                if 0<=neighbor<arr_size:
                    head = self.find(neighbor,group)
                    if head in sizes and sizes[head] == m:
                        ans = ix
                    if group[neighbor] != -1:
                        self.union(val-1,neighbor,group,sizes)
            # if head and sizes[head] == m:
            #     ans = max(ix+1,ans)
        return ans

from sortedcontainers import SortedList
class Solution:
    def findLatestStep(self, A: List[int], m: int) -> int:
        length = [0] * (len(A) + 2)
        count = [0] * (len(A) + 1)
        res = -1
        for i, a in enumerate(A):
            left, right = length[a - 1], length[a + 1]
            length[a] = length[a - left] = length[a + right] = left + right + 1
            count[left] -= 1
            count[right] -= 1
            count[length[a]] += 1
            if count[m]:
                res = i + 1
        return res
from collections import defaultdict

class DSU:
    def __init__(self, n):
        self.n = n
        self.fa = list(range(n))
        self.sz = [1 for _ in range(n)]
        self.cnt = defaultdict(int)

    def find(self, x):
        r = x
        while self.fa[r] != r:
            r = self.fa[r]
        i = x
        while i != r:
            i, self.fa[i] = self.fa[i], r
        return r
    
    def join(self, x, y):
        x = self.find(x)
        y = self.find(y)
        if x != y:
            if self.sz[x] > self.sz[y]:
                x, y = y, x
            sx = self.sz[x]
            sy = self.sz[y]
            self.cnt[sx] -= 1
            self.cnt[sy] -= 1
            self.cnt[sx + sy] += 1
            
            self.fa[x] = y
            self.sz[y] += self.sz[x]

class Solution:
    def findLatestStep(self, a: List[int], m: int) -> int:
        n = len(a)
        b = [0 for _ in range(n + 2)]
        dsu = DSU(n + 1)
        
        ans = -1
        valid = set()
        for step, j in enumerate(a, 1):
            b[j] = 1
            dsu.cnt[1] += 1

            if b[j - 1]:
                dsu.join(j, j - 1)
            if b[j + 1]:
                dsu.join(j, j + 1)
            if dsu.cnt.get(m, 0):
                ans = step

        return ans
class UF:
  def __init__(self, e):
    self.parents = list(range(e))
    self.ranks = [0]*e

  def findP(self, r):
    if r==self.parents[r]:
      return r
    self.parents[r] = self.findP(self.parents[r])
    return self.parents[r]

  def union(self, u, v):
    up = self.findP(u)
    vp = self.findP(v)

    if up!=vp:
      if self.ranks[up]>=self.ranks[vp]:
        self.parents[vp] = up

        self.ranks[up] += self.ranks[vp]
      else:
        self.parents[up] = vp
        self.ranks[vp] += self.ranks[up]
      return False
    return True
  
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
      
      if m==len(arr): return m
      
      n = len(arr)
      u = UF(n)
      
      res = -1
  
      for step, v in enumerate(arr):
        
        v = v - 1
        u.ranks[v] = 1
        
        for i in (v-1, v+1):
          if 0<=i<n:
            if u.ranks[u.findP(i)]==m:
              res = step
              
            if u.ranks[i]:
              u.union(v, i)
            
        # print (step, u.ranks, u.parents)
        
        # if u.ranks[u.findP(v)]==m:
        #   res = step
        
      return res
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        ids = [0]*(len(arr)+2)
        d = []
        lst = -1
        def parent(a):
            # print(a)
            if(ids[a]<0):
                return a
            ids[a] = parent(ids[a])
            return ids[a]
        
        def SF(a, b):
            return parent(a)==parent(b)
        
        def SU(a,b):
            a = parent(a)
            b = parent(b)
            # print(a," ",b)
            if(a==b):
                return
            if(ids[a]<=ids[b]):
                ids[a] += ids[b]
                ids[b] = a
            else:
                ids[b] += ids[a]
                ids[a] = b
        
        def size(a):
            return -ids[parent(a)]
        
        for j,i in enumerate(arr):
            # print("toto ",j, "  ",i)
            ids[i]=-1
            if(ids[i-1]!=0):
                SU(i-1,i)
            if(ids[i+1]!=0):
                SU(i, i+1)
            # print(i," ",size(i))
            if(size(i)==m):
                d.append(i)
            for t in range(len(d)-1,-1, -1):
                x = d.pop(t)
                if(size(x)==m):
                    d.append(x)
                    lst = j+1
            # print(d)
        
        return lst
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        n = len(arr)
        F = [0] * n
        d = collections.defaultdict(int)
        
        def find(x):
            if F[x] < 0:
                return x
            else:
                F[x] = find(F[x])
                return F[x]
        
        t = [0] * n
        ans = -1
        
        for i in range(n):
            ind = arr[i] - 1
            d[1] += 1
            t[ind] = 1
            F[ind] = -1
            if ind > 0 and t[ind-1] == 1:
                new = find(ind-1)
                d[-F[ind]] -= 1
                d[-F[new]] -= 1
                d[-F[ind]-F[new]] += 1
                F[ind] += F[new]
                F[new] = ind
            if ind < n-1 and t[ind+1] == 1:
                new = find(ind+1)
                d[-F[ind]] -= 1
                d[-F[new]] -= 1
                d[-F[ind]-F[new]] += 1
                F[ind] += F[new]
                F[new] = ind
            if d[m] > 0:
                ans = i + 1
        return ans
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        if len(arr) == m:
            return len(arr)
        
        visited = set([0, len(arr)+1]) 
        for i in range(len(arr)-1, -1, -1):
            index = arr[i]
            if index + m +1 in visited:
                for n in range(index, index+m+1):
                    if n in visited:
                        break 
                else:
                    return i
            
            if index - m - 1 in visited:
                for n in range(index-1, index-m-1, -1):
                    if n in visited:
                        break 
                else:
                    return i
            visited.add(index)
            
        return -1 
            
            

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        self.groups_start = {} # start pos: length
        self.groups_end = {} # end pos: start pos
        
        last_pos = -1
        m_start_pos = set()
        n = len(arr)
        for k, i in enumerate(arr):
            start_pos = i
            if i + 1 <= n:
                if i + 1 in self.groups_start:
                    # merge
                    length = self.groups_start[i+1]
                    del self.groups_start[i+1]
                    self.groups_start[i] = length + 1
                    self.groups_end[i+length] = i
                    if i + 1 in m_start_pos:
                        m_start_pos.remove(i+1)
                else:
                    self.groups_start[i] = 1
                    self.groups_end[i] = i
            else:
                self.groups_start[i] = 1
                self.groups_end[i] = i
            if i - 1 >= 1:
                if i - 1 in self.groups_end:
                    # merge
                    start_pos = self.groups_end[i-1]
                    if start_pos in m_start_pos:
                        m_start_pos.remove(start_pos)
                    new_length = self.groups_start[start_pos] + self.groups_start[i]
                    self.del_group(i)
                    self.del_group(start_pos)
                    self.groups_start[start_pos] = new_length
                    self.groups_end[start_pos+new_length-1] = start_pos
            if self.groups_start[start_pos] == m:
                m_start_pos.add(start_pos)
            if len(m_start_pos) > 0:
                last_pos = k + 1
                    
        return last_pos
        
    def del_group(self, start_pos):
        del self.groups_end[start_pos+self.groups_start[start_pos]-1]
        del self.groups_start[start_pos]
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        n = len(arr)
        if n == m:
            return n
        
        def helper(rec, k):
            temp_rec = []
            for i, j in rec:
                if k < i or k > j:
                    if j-i+1 > m:
                        temp_rec.append([i, j])
                    continue
                if k-i == m or j-k == m:
                    return True
                else:
                    if k-i > m:
                        temp_rec.append([i, k-1])
                    if j-k > m:
                        temp_rec.append([k+1, j])
#            print(temp_rec)
            return temp_rec
        
        rec = [(1, n)]
        for ind in range(n, 0, -1):
            rec = helper(rec, arr[ind-1])
#            print(rec)
            if rec == True:
                return ind-1
            elif not rec:
                return -1
            
            

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        def func(i):
            if d[i]!=i: d[i]=func(d[i])
            return d[i]
        d,dd,a,s={},{},{},-1
        for p,i in enumerate(arr,1):
            if i-1 in d:
                if i+1 in d:
                    j,k=func(i-1),func(i+1)
                    d[k],d[i]=j,j
                    a[dd[j]]-=1
                    a[dd[k]]-=1
                    dd[j]+=dd[k]+1
                    a[dd[j]]=a.get(dd[j],0)+1
                else:
                    j=func(i-1)
                    d[i]=j
                    a[dd[j]]-=1
                    dd[j]+=1
                    a[dd[j]]=a.get(dd[j],0)+1
            elif i+1 in d:
                j=func(i+1)
                d[i]=j
                a[dd[j]]-=1
                dd[j]+=1
                a[dd[j]]=a.get(dd[j],0)+1
            else:
                d[i]=i
                dd[i]=1
                a[1]=a.get(1,0)+1
            if a.get(m,0): s=p
        return s
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        result = dict()
        total = len(arr)
        buffer = [-1] * total

        return self.do_find(arr, m, total, 0, buffer, result)

    def do_find(self, arr, m, total, index, buffer, result):
        if index == total:
            return -1

        arr_idx = arr[index] - 1


        if arr_idx > 0 and buffer[arr_idx-1] != -1:
            start_idx = buffer[arr_idx-1]
            result[arr_idx - start_idx] -= 1
        else:
            start_idx = arr_idx

        if arr_idx < total - 1 and buffer[arr_idx + 1] != -1:
            end_idx = buffer[arr_idx+1]
            result[end_idx - arr_idx] -= 1
        else:
            end_idx = arr_idx

        new_len = end_idx - start_idx + 1

        if new_len in result:
            result[new_len] += 1
        else:
            result[new_len] = 1

        buffer[end_idx] = start_idx
        buffer[start_idx] = end_idx

        current_result = index+1 if result.get(m, 0) > 0 else -1
        next_result = self.do_find(arr, m, total, index + 1, buffer, result)
        if next_result > 0:
            return next_result
        return current_result

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        a = len(arr)
        if a == m:
            return m
        arr_set = set(arr)
        arr.reverse()
        for i in range(a):
            arr_set.remove(arr[i])
            back_i = arr[i]+1
            if back_i in arr_set:
                cur_streak = 1
                while back_i+1 in arr_set:
                    back_i += 1
                    cur_streak += 1
                    if cur_streak > m:
                        break
                if cur_streak == m:
                    return a-1-i
            front_i = arr[i]-1
            if front_i in arr_set:
                cur_streak = 1
                while front_i-1 in arr_set:
                    front_i -= 1
                    cur_streak += 1
                    if cur_streak > m:
                        break
                if cur_streak == m:
                    return a-1-i
        return -1
from collections import Counter
class Union:
    def __init__(self, n):
        self.groups = list(range(n))
        self.sizes = [1] * n
    def find(self, node):
        while node != self.groups[node]:
            node = self.groups[node]
        return node
    
    def union(self, node1, node2):
        root1, root2 = self.find(node1), self.find(node2)
        if self.sizes[root1] < self.sizes[root2]:
            root1, root2 = root2, root1
        self.sizes[root1] += self.sizes[root2]
        self.groups[root2] = root1
        return self.sizes[root1]
    
    def getSize(self, node):
        node = self.find(node)
        return self.sizes[node]
    
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        n = len(arr)
        union = Union(n)
        bits = [0] * n
        counter = Counter()
        res = -1
        for i, x in enumerate(arr):
            curr_size = 1
            x -= 1
            bits[x] = 1
            if x and bits[x-1] == 1:
                l_size = union.getSize(x - 1)
                counter[l_size] -= 1
                curr_size = union.union(x-1, x)
            
            if x < n - 1 and bits[x+1] == 1:
                r_size = union.getSize(x + 1)
                counter[r_size] -= 1
                curr_size = union.union(x, x+1)
            
            counter[curr_size] += 1
            #print(counter)
            if counter[m] > 0:
                res = i + 1
        return res
                
                

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        n = len(arr)
        a = [0]*n
        c = [0]*n
        for i in range(n):
            a[i] = -1
            c[i] = -1
        res = -1
        have_m = 0
        def find_root(i):
            while a[i] != i:
                i = a[i]
            return i
        step = 0
        for pos in arr:
            step+=1
            pos -= 1
            if pos<n-1 and a[pos+1]!=-1:
                a[pos+1] = pos
                a[pos] = pos
                c[pos] = c[pos+1]+1
                if c[pos+1] == m:
                    have_m-=1
                if c[pos] == m:
                    have_m+=1
            else:
                a[pos] = pos
                c[pos] = 1
                if c[pos] == m:
                    have_m+=1
            if pos>0 and a[pos-1]!=-1:
                a[pos] = find_root(pos-1)
                if c[pos] == m:
                    have_m -= 1
                if c[a[pos]] == m:
                    have_m -= 1
                c[a[pos]] += c[pos]
                if c[a[pos]] == m:
                    have_m+=1
            if have_m:
                res = step
        return res
                

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        pos = [None] * (len(arr) + 3)
        num = 0
        ans = -1
        for idx in range(1, len(arr) + 1):
            l, r, p = [arr[idx - 1]] * 3
            if pos[p-1]:
                if pos[p-1][1] - pos[p-1][0] + 1 == m:
                    num -= 1
                l = pos[p-1][0]
            if pos[p + 1]:
                if pos[p + 1][1] - pos[p+1][0] + 1 == m:
                    num -= 1
                r = pos[p+1][1]
            pos[l] = pos[r] = (l, r)
            if r - l + 1 == m:
                num += 1
            if num != 0:
                ans = idx
        return ans
class UnionFind:

    def __init__(self, values: int):
        self.values = list(range(values))
        self.connectedComponents = [1] * values

    def root(self, key):
        root = self.values[key]

        while root != self.values[root]:
            root = self.values[root]

        return root

    def union(self, keyA, keyB):
        rootA = self.root(keyA)
        rootB = self.root(keyB)

        self.values[rootB] = rootA
        self.connectedComponents[rootA] += self.connectedComponents[rootB]
        self.connectedComponents[rootB] = 0

    def find(self, keyA, keyB):
        return self.root(keyA) == self.root(keyB)


class Solution:
    def findLatestStep(self, arr: list, m: int) -> int:

        uf = UnionFind(len(arr))

        step = -1
        groupM = set()
        binaryString = [0] * len(arr)

        for i, val in enumerate(arr):

            if i == 9:
                print('')

            val = val - 1

            binaryString[val] = 1

            if val != 0 and binaryString[val - 1] == 1:
                if not uf.find(val, val - 1):
                    root = uf.root(val - 1)
                    uf.union(val, val - 1)

                    if root in groupM:
                        groupM.remove(root)

            if val != (len(arr) - 1) and binaryString[val + 1] == 1:
                if not uf.find(val, val + 1):
                    root = uf.root(val + 1)
                    uf.union(val, val + 1)

                    if root in groupM:
                        groupM.remove(root)

            if uf.connectedComponents[val] == m:
                groupM.add(val)
            elif uf.connectedComponents[val] != m and val in groupM:
                groupM.remove(val)

            if len(groupM):
                step = i + 1

        return step
# class Solution:
#     def findLatestStep(self, arr: List[int], m: int) -> int:
#         n = len(arr)
#         if n == m:
#             return n
#         temp = [1]*n
#         j = n
#         while j > 0:
#             temp[arr[j-1]-1] = 0
#             count = 0
#             for i, y in enumerate(temp):
#                 if y == 1:
#                     count += 1
#                 else:
#                     count = 0
#                 if count == m and (i+1 >= n or (i+1 < n and temp[i+1] == 0)):
#                     return j - 1
#             j -= 1
#         return -1

# class Solution:
#     def findLatestStep(self, A, m):
#         if m == len(A): return m
#         length = [0] * (len(A) + 2)
#         res = -1
#         for i, a in enumerate(A):
#             left, right = length[a - 1], length[a + 1]
#             if left == m or right == m:
#                 res = i
#             length[a - left] = length[a + right] = left + right + 1
#         return res

class UnionFindSet:
    def __init__(self, n):
        self.parents = list(range(n))
        self.ranks = [0] * n
        
    def find(self, u):
        if u != self.parents[u]:
            self.parents[u] = self.find(self.parents[u])
        return self.parents[u]
    
    def union(self, u, v):
        pu, pv = self.find(u), self.find(v)
        if pu == pv:
            return False
        if self.ranks[pu] > self.ranks[pv]:
            self.parents[pv] = pu
            self.ranks[pu] += self.ranks[pv]
        else:
            self.parents[pu] = pv
            self.ranks[pv] += self.ranks[pu]
        return True

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        if m == len(arr):
            return m
        
        n, ans = len(arr), -1
        uf = UnionFindSet(n)
        
        for step, i in enumerate(arr):
            i -= 1
            uf.ranks[i] = 1
            for j in (i - 1, i + 1):
                if 0 <= j < n:
                    if uf.ranks[uf.find(j)] == m:
                        ans = step
                    if uf.ranks[j]:
                        uf.union(i, j)
        
        return ans
        

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        n = len(arr)
        if m == n: return n 
        arr.append(n+1)
        start = {}
        finish = {}
        last = -1
        for level,i in enumerate(arr):
            if i-1 not in finish: finish[i-1] = i 
            if i+1 not in start: start[i+1] = i

            s, f = finish[i-1], start[i+1]
            start[s] = f 
            finish[f] = s
            
            os, of = i+1, start[i+1]
            if of-os+1 == m: last = level
                
            os, of = finish[i-1], i-1
            if of-os+1 == m: last = level
            
            del start[i+1]
            del finish[i-1]
            
        return last
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        if m == len(arr):
            return m
        lengths = [0] * (len(arr) + 2)
        result = -1
        for i, a in enumerate(arr):
            left, right = lengths[a - 1], lengths[a + 1]
            if left == m or right == m:
                result = i
            lengths[a - left] = lengths[a + right] = left + right + 1
        return result
            

class UnionFindSet:
    def __init__(self, n):
        self.parents = list(range(n))
        self.ranks = [0] * n
        
    def find(self, u):
        while u != self.parents[u]:
            self.parents[u] = self.parents[self.parents[u]]
            u = self.parents[u]
        return self.parents[u]
    
    def union(self, u, v):
        pu, pv = self.find(u), self.find(v)
        if pu == pv:
            return False
         
        if self.ranks[pu] > self.ranks[pv]:
            a, b = pu, pv
        else:
            a, b = pv, pu
            
        self.parents[b] = a
        self.ranks[a] += self.ranks[b]
        return True

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        if m == len(arr):
            return m
        
        n, ans = len(arr), -1
        uf = UnionFindSet(n)
        
        for step, i in enumerate(arr):
            i -= 1
            uf.ranks[i] = 1
            for j in (i - 1, i + 1):
                if 0 <= j < n:
                    if uf.ranks[uf.find(j)] == m:
                        ans = step
                    if uf.ranks[j]:
                        uf.union(i, j)
        
        return ans
                
                    
        

class UnionFindSet:
    def __init__(self, n):
        self.parents = list(range(n))
        self.ranks = [0] * n
        
    def find(self, u):
        if u != self.parents[u]:
            self.parents[u] = self.find(self.parents[u])
        return self.parents[u]
    
    def union(self, u, v):
        pu, pv = self.find(u), self.find(v)
        if pu == pv:
            return False
        if self.ranks[pu] > self.ranks[pv]:
            self.parents[pv] = pu
            self.ranks[pu] += self.ranks[pv]
        #elif self.ranks[pv] > self.ranks[pu]:
        #    self.parents[pu] = pv
        #    self.ranks[pv] += self.ranks[pu]
        else:
            self.parents[pu] = pv
            self.ranks[pv] += self.ranks[pu]
        return True

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        n, ans = len(arr), -1
        uf = UnionFindSet(n)
        if m == n:
            return n
        for step, i in enumerate(arr):
            i -= 1
            uf.ranks[i] = 1
            for j in (i - 1, i + 1):
                if 0 <= j < n:
                    if uf.ranks[uf.find(j)] == m:
                        ans = step
                    if uf.ranks[j]:
                        uf.union(i, j)
        return ans
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        n = len(arr)
        left = [0]*(n+2)
        right = [0]*(n+2)
        def getlength(x):
            return right[x]-x+1
        count = 0
        res = -1
        for i in range(n):
            x = arr[i]
            if left[x-1] and right[x+1]:
                if getlength(left[x-1]) == m:
                    count -= 1
                if getlength(x+1) == m:
                    count -= 1
                right[left[x-1]] = right[x+1]
                left[right[x+1]] = left[x-1]
                if getlength(left[x-1]) == m:
                    count += 1
            elif left[x-1]:
                if getlength(left[x-1]) == m:
                    count -= 1
                right[left[x-1]] = x
                left[x] = left[x-1]
                if getlength(left[x-1]) == m:
                    count += 1
            elif right[x+1]:
                if getlength(x+1) == m:
                    count -= 1
                left[right[x+1]] = x
                right[x] = right[x+1]
                if getlength(x) == m:
                    count += 1
            else:
                left[x] = x
                right[x] = x
                if m == 1:
                    count += 1
            if count > 0:
                res = i + 1
        return res
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        d = {}
        res = set()
        ans = -1
        for (s,j) in enumerate(arr):
            i = j-1
            #print(i,d)
            if i+1 not in d and i-1 not in d:
                #print(i,d)
                d[i] = 1
                if d[i] == m: res.add(i)
            elif i+1 in d and i-1 not in d:
                if i+1 in res:res.remove(i+1)
                if i+d[i+1] in res:res.remove(i+d[i+1])
                if d[i+1] != 1:
                    temp = d.pop(i+1)
                    d[i] = 1+temp
                    d[i+temp] += 1
                else:
                    d[i] = 2
                    d[i+1] = 2
                if d[i] == m: 
                    res.add(i)

                #print(i,d)
            elif i-1 in d and i+1 not in d:
                #print(i,d)
                if i-1 in res: res.remove(i-1)
                if i-d[i-1] in res:res.remove(i-d[i-1])
                if d[i-1] != 1:
                    temp = d.pop(i-1)
                    d[i] = 1+temp
                    d[i-temp] += 1
                else:
                    d[i] = 2
                    d[i-1] = 2
                if d[i] == m: 
                    res.add(i)

                         
            else:
                a,b = i-d[i-1],i+d[i+1]
                if d[i-1] != 1: d.pop(i-1)
                if d[i+1] != 1: d.pop(i+1)
                if i-1 in res: res.remove(i-1)
                if i+1 in res: res.remove(i+1)
                if a in res: res.remove(a)
                if b in res: res.remove(b)
                d[a] = b-a+1
                d[b] = b-a+1
                #print(i,a,b,d[i-1],d[i+1])
                if b-a+1 == m: 
                    res.add(a)
                    res.add(b)        

            #print(d,res)
            if res: ans = s+1
        return ans
class UnionFindSet:
    def __init__(self, n):
        self.parents = list(range(n))
        self.ranks = [0] * n
        
    def find(self, u):
        if u != self.parents[u]:
            self.parents[u] = self.find(self.parents[u])
        return self.parents[u]
    
    def union(self, u, v):
        pu, pv = self.find(u), self.find(v)
        if pu == pv:
            return False
        if self.ranks[pu] > self.ranks[pv]:
            self.parents[pv] = pu
            self.ranks[pu] += self.ranks[pv]
        else:
            self.parents[pu] = pv
            self.ranks[pv] += self.ranks[pu]
        return True

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        if m == len(arr):
            return m
        
        n, ans = len(arr), -1
        uf = UnionFindSet(n)
        
        for step, i in enumerate(arr):
            i -= 1
            uf.ranks[i] = 1
            for j in (i - 1, i + 1):
                if 0 <= j < n:
                    if uf.ranks[uf.find(j)] == m:
                        ans = step
                    if uf.ranks[j]:
                        uf.union(i, j)
        
        return ans
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        n = len(arr)
        mcount = 0
        lastindex = -1
        ends = {}
        starts = {}
        for i in range(n):
            index = arr[i]
            if index-1 in ends and index+1 in starts:
                start = ends[index-1]
                end = starts[index+1]
                old1 = index-start
                old2 = end-index
                if old1==m:
                    mcount-=1
                if old2==m:
                    mcount-=1
                if end-start+1==m:
                    mcount+=1
                starts[start], ends[end] = end, start
                starts.pop(index+1)
                ends.pop(index-1)
            elif index-1 in ends:
                start = ends[index-1]
                old1 = index-start
                if old1==m:
                    mcount-=1
                elif old1==m-1:
                    mcount+=1
                starts[start] = index
                ends[index] = start
                ends.pop(index-1)
            elif index+1 in starts:
                end = starts[index+1]
                old1 = end-index
                if old1==m:
                    mcount-=1
                elif old1==m-1:
                    mcount+=1
                starts[index] = end
                ends[end] = index
                starts.pop(index+1)
            else:
                starts[index] = index
                ends[index] = index
                if m==1:
                    mcount+=1
            if mcount != 0:
                lastindex = i+1
        return lastindex
                
                

class UnionFindSet:
    def __init__(self, n):
        self.parents = list(range(n))
        self.ranks = [0] * n
        
    def find(self, u):
        if u != self.parents[u]:
            self.parents[u] = self.find(self.parents[u])
        return self.parents[u]
    
    def union(self, u, v):
        pu, pv = self.find(u), self.find(v)
        if pu == pv:
            return False
        if self.ranks[pu] > self.ranks[pv]:
            self.parents[pv] = pu
            self.ranks[pu] += self.ranks[pv]
        else:
            self.parents[pu] = pv
            self.ranks[pv] += self.ranks[pu]
        return True

    
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        if m == len(arr):
            return m
        
        n, ans = len(arr), -1
        uf = UnionFindSet(n)
        
        for step, i in enumerate(arr):
            i -= 1
            uf.ranks[i] = 1
            for j in (i - 1, i + 1):
                if 0 <= j < n:
                    if uf.ranks[uf.find(j)] == m:
                        ans = step
                    if uf.ranks[j]:
                        uf.union(i, j)
        
        return ans
class UnionFindSet:
    def __init__(self, n):
        self.parents = list(range(n))
        self.ranks = [0] * n
        
    def find(self, u):
        if u != self.parents[u]:
            self.parents[u] = self.find(self.parents[u])
        return self.parents[u]
    
    def union(self, u, v):
        pu, pv = self.find(u), self.find(v)
        if pu == pv:
            return False
        if self.ranks[pu] > self.ranks[pv]:
            self.parents[pv] = pu
            self.ranks[pu] += self.ranks[pv]
        else:
            self.parents[pu] = pv
            self.ranks[pv] += self.ranks[pu]
        return True
    
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        if m == len(arr):
            return m
        
        n, ans = len(arr), -1
        uf = UnionFindSet(n)
        
        for step, i in enumerate(arr):
            i -= 1
            uf.ranks[i] = 1
            for j in (i - 1, i + 1):
                if 0 <= j < n:
                    if uf.ranks[uf.find(j)] == m:
                        ans = step
                    if uf.ranks[j]:
                        uf.union(i, j)
        
        return ans
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        N = len(arr)
        spans = [(1, N)]
        step = N
        
        if m == N:
            return m
        
        while arr:
            #print(step, spans)
            d = arr.pop()
            step -= 1
            for span in spans:
                if span[0] <= d <= span[1]:
                    if d-span[0] == m or span[1] - d == m:
                        return step
                    
                    spans.remove(span)
                    if d - span[0] > m:
                        spans.append((span[0], d-1))
                    if span[1] - d > m:
                        spans.append((d+1, span[1]))
            
        return -1
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        start_mp, end_mp = {}, {}
        max_step = -1
        m_cnt = 0
        # use the merge's mind
        for idx, num in enumerate(arr): 
            l = 1
            if num-1 not in end_mp and num+1 not in start_mp: 
                start_mp[num] = end_mp[num] = 1
                start_index = end_index = num
            elif num-1 in end_mp and num +1 in start_mp:
                # merge
                old_l = end_mp[num-1]
                old_r = start_mp[num+1]
                if old_l == m: 
                    m_cnt -= 1
                if old_r == m: 
                    m_cnt -= 1
                start_index = num-1 - end_mp[num-1] + 1
                end_index = num+1 + start_mp[num+1] -1
                l = end_mp[num-1] + start_mp[num+1] + 1
                del end_mp[num-1], start_mp[num-1-old_l+1], start_mp[num+1], end_mp[num+1+old_r-1]
                start_mp[start_index] = l
                end_mp[end_index] = l
            elif num-1 in end_mp: 
                # extend to the left
                old_l = end_mp[num-1]
                if old_l == m: 
                    m_cnt -= 1
                l = old_l + 1
                del end_mp[num-1], start_mp[num-1-old_l+1]
                start_index = num-l+1 
                end_index = m
                end_mp[num] = start_mp[num-l+1] = l
            elif num+1 in start_mp: 
                old_l = start_mp[num+1]
                if old_l == m: 
                    m_cnt -= 1
                l = old_l + 1
                del end_mp[num+1+old_l-1], start_mp[num+1]
                start_mp[num] = end_mp[num+l-1] = l
                start_index = num
                end_index = num+l-1
            
            if l == m: 
                m_cnt += 1
            if m_cnt > 0: 
                max_step = max(max_step, idx + 1)
        return max_step
                
    def findLatestStep(self, arr: List[int], m: int) -> int:
        # actually the above code could be much simpler 
        # keyed by length, cnt is the value
        lcnt = collections.Counter()
        l = [0 for _ in range(len(arr)+2)]
        max_step = -1
        for idx, num in enumerate(arr): 
            left, right = l[num-1], l[num+1]
            new_l = left + right + 1
            l[num-left] = l[num] = l[num+right] = new_l # this step is the key
            lcnt[left] -=1
            lcnt[right] -= 1
            lcnt[new_l] += 1
            if lcnt[m]: 
                max_step = idx + 1
        return max_step
            
            

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        segment = [0]*len(arr)
        count = {}
        res = -1
        for i in range(len(arr)):
            left,right = 0,0
            index = arr[i]-1
            if index-1 >= 0:
                left = segment[index-1]
            if index+1 < len(arr):
                right = segment[index+1]
            segment[index-left] = left+right+1
            segment[index+right] = left+right+1
            
            if left in count and count[left] != 0 :
                count[left] -= 1
            if right in count and count[right] != 0 :
                count[right] -= 1
            
            if left+right+1 in count:
                count[left+right+1] += 1
            else:
                count[left+right+1] = 1
                
            if m in count and count[m] != 0 :
                res = i+1
        return res
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        if m == len(arr):
            return len(arr)    
        
        u = list(range(len(arr)+1))
        size = [0] * (len(arr)+1)
        
        def find(a):
            if u[a] != a:
                u[a] = find(u[a])
            return u[a]
        
        def union(a, b):
            u1 = find(a)
            u2 = find(b)
            if u1 != u2:
                if size[u1] < size[u2]:
                    u[u1] = u2
                    size[u2] += size[u1]
                else:
                    u[u2] = u1
                    size[u1] += size[u2]
        ans = -1
        for a_idx, a in enumerate(arr):
            # print(a)
            # print("u", u)
            # print("s", size)
            size[a] = 1
            for i in [a-1, a+1]:
                if 1 <= i <= len(arr):
                    if size[find(i)] == m:
                        ans = a_idx
                    if size[i] > 0:
                        union(a, i)
                    # if size[find(a)] == m:
                    #     ans = a_idx+1
                        
        # print("u", u)
        # print("s", size)
                        
        for i in range(1, len(arr)+1):
            # print(i, find(i))
            if size[find(i)] == m:
                return len(arr)
        # print("not last")
        return ans
        
#         sections = [
#             [1, len(arr)]
#         ]
#         arr = arr[::-1]
#         for a_idx, a in enumerate(arr):
#             # print(a, sections)
#             l = 0
#             r = len(sections)
#             while l < r:
#                 c = l + (r-l)//2
#                 if sections[c][0] <= a <= sections[c][1]:
#                     # a at left
#                     if a == sections[c][0]:
#                         sections[c][0] = a+1
#                         if sections[c][0] > sections[c][1]:
#                             sections = sections[:c] + sections[c+1:]
#                         else:
#                             if sections[c][1] - sections[c][0] + 1 == m:
#                                 return len(arr) - a_idx - 1 
#                     elif a == sections[c][1]:
#                         sections[c][1] = a-1
#                         if sections[c][0] > sections[c][1]:
#                             sections = sections[:c] + sections[c+1:]
#                         else:
#                             if sections[c][1] - sections[c][0] + 1 == m:
#                                 return len(arr) - a_idx - 1 
#                     else:
#                         tmp = sections[c][1]
#                         sections[c][1] = a-1
#                         sections = sections[:c+1] + [[a+1, tmp]] + sections[c+1:]
#                         # heapq.heappush(sections, [a+1, tmp])
#                         if sections[c][1] - sections[c][0] + 1 == m:
#                             return len(arr) - a_idx - 1 
#                         if sections[c+1][1] - sections[c+1][0] + 1 == m:
#                             return len(arr) - a_idx - 1
#                     break   
#                 elif a < sections[c][0]:
#                     r = c
#                 elif a > sections[c][1]:
#                     l = c+1
#             # print(sections)
        
#         return -1
        
#         ans = -1
#         dp = [0] * (len(arr)+1)
#         for idx, a in enumerate(arr):
#             if a > 1:
#                 dp[a] += dp[a-1]
#             if a < len(arr):
#                 dp[a] += dp[a+1]
#             dp[a] += 1
#             if dp[a] == m:
#                 ans = idx+1
#             print(dp)
                
#         return ans
        

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        if len(arr) == m:
            return m
        n = len(arr)
        uf = UnionFind(n)
        ans = -1
        for step, i in enumerate(arr):
            index = i - 1
            uf.ranks[index] = 1
            for j in [index - 1, index + 1]:
                if 0 <= j < n:
                    if uf.ranks[uf.find(j)] == m:
                        ans = step
                    if uf.ranks[j]:
                        uf.union(index, j)
        return ans
        

class UnionFind():
    def __init__(self, n):
        self.parents = list(range(n))
        self.ranks = [0] * n
    
    def find(self, u):
        if u != self.parents[u]:
            self.parents[u] = self.find(self.parents[u])
        return self.parents[u]
    
    def union(self, n1, n2):
        p1, p2 = self.find(n1), self.find(n2)
        if p1 == p2:
            return
        if self.ranks[p1] > self.ranks[p2]:
            self.parents[p2] = p1
            self.ranks[p1] += self.ranks[p2]
            return
        else:
            self.parents[p1] = p2
            self.ranks[p2] += self.ranks[p1]
            return

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        
        par = {}
        sz = {}
        target = set()
        
        def find(i):
            while i != par[i]:
                par[i] = par[par[i]]
                i = par[i]
            return i
        
        def union(i, j):
            x = find(i)
            y = find(j)
            if x == y:
                return
            if sz[x] <= sz[y]:
                sz[y] += sz[x]
                par[x] = y
                
                if sz[y] == m:
                    target.add(y)
                    
                if sz[x] == m and x in target:
                    target.remove(x)
            else:
                sz[x] += sz[y]
                par[y] = x
                
                if sz[x] == m:
                    target.add(x)
                    
                if sz[y] == m and y in target:
                    target.remove(y)
        
        count = 1
        ans = -1
        for i in arr:
            if i not in par:
                par[i] = i
                sz[i] = 1
                if m == 1:
                    target.add(i)
                
            if i - 1 in par and i + 1 in par:
                union(i-1, i+1)
                union(i-1, i)
            elif i - 1 in par:
                union(i-1, i)
            elif i + 1 in par:
                union(i, i+1)
                
            t = set()
            for i in target:
                if sz[i] == m:
                    ans = count
                    t.add(i)
            target = t
            count += 1
                
        return ans
            

class UnionFindSet:
    def __init__(self, n):
        self.parents = list(range(n))
        self.ranks = [0] * n
        
    def find(self, u):
        if u != self.parents[u]:
            self.parents[u] = self.find(self.parents[u])
        return self.parents[u]
    
    def union(self, u, v):
        pu, pv = self.find(u), self.find(v)
        if pu == pv:
            return False
        if self.ranks[pu] > self.ranks[pv]:
            self.parents[pv] = pu
            self.ranks[pu] += self.ranks[pv]
        else:
            self.parents[pu] = pv
            self.ranks[pv] += self.ranks[pu]
        return True

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        if len(arr)==m:
            return m
        n,  ans = len(arr), -1
        uf = UnionFindSet(n)
        
        for step, i in enumerate(arr):
            i -= 1
            uf.ranks[i] = 1
            for j in (i - 1, i + 1):
                if 0 <= j < n:
                    if uf.ranks[uf.find(j)] == m:
                        ans = step
                    if uf.ranks[j]:
                        uf.union(i, j)
        
        #for i in range(n):
         #   if uf.ranks[uf.find(i)] == m:
          #      return n
            
        return ans

class Solution:
    def findLatestStep(self, a: List[int], m: int) -> int:
        g = [0] * (len(a)+1)
        cnt = Counter()
        last = -1
        for i, p in enumerate(a):    
            l = g[p-1] if p > 1 else 0
            r = g[p+1] if p < len(g)-1 else 0
            new_l = l+1+r
            g[p-l] = g[p+r] = new_l
            if l > 0: 
                cnt[l] -= 1
                if cnt[l] == 0: del cnt[l]
            if r > 0: 
                cnt[r] -= 1
                if cnt[r] == 0: del cnt[r]
            cnt[new_l] += 1 
            if m in cnt:
                last = i + 1
            # print(i, l, r, g)
        return last
class UnionFindSet:
    def __init__(self, n):
        self.parents = list(range(n))
        self.ranks = [0] * n
        
    def find(self, u):
        if u != self.parents[u]:
            self.parents[u] = self.find(self.parents[u])
        return self.parents[u]
    
    def union(self, u, v):
        pu, pv = self.find(u), self.find(v)
        if pu == pv:
            return False
        if self.ranks[pu] > self.ranks[pv]:
            self.parents[pv] = pu
            self.ranks[pu] += self.ranks[pv]
        else:
            self.parents[pu] = pv
            self.ranks[pv] += self.ranks[pu]
        return True

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        if m == len(arr):
            return m
        
        n, ans = len(arr), -1
        uf = UnionFindSet(n)
        
        for step, i in enumerate(arr):
            i -= 1
            uf.ranks[i] = 1
            for j in (i - 1, i + 1):
                if 0 <= j < n:
                    if uf.ranks[uf.find(j)] == m:
                        ans = step
                    if uf.ranks[j]:
                        uf.union(i, j)
        
        return ans
                
                    
        

class UF:
    def __init__(self, n):
        self.parents = [i for i in range(n)]
        self.ranks = [0] * n
    
    def find(self, u):
        if u != self.parents[u]:
            self.parents[u] = self.find(self.parents[u])
        return self.parents[u]
    
    def union(self, u, v):
        pu, pv = self.find(u), self.find(v)
        if pu == pv:
            return False
        if self.ranks[pu] > self.ranks[pv]:
            self.parents[pv] = pu
            self.ranks[pu] += self.ranks[pv]
        else:
            self.parents[pu] = pv
            self.ranks[pv] += self.ranks[pu]
            
        return True

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        ans = -1
        if m == len(arr):
            return m
        
        n = len(arr)
        uf = UF(n)
        
        for step, i in enumerate(arr):
            i -= 1 # because of 1-index
            uf.ranks[i] = 1
            
            for j in (i-1, i+1):
                if 0<=j<n:
                    # if j's parent to j constintutes a m-length array
                    if uf.ranks[uf.find(j)] == m:
                        ans = step
                    if uf.ranks[j] > 0:
                        uf.union(i, j)
        return ans
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        n = len(arr)
        if m == n: 
            return m
        
        border = [0]*(n+2)
        ans = -1
        for i in range(n):
            left = right = arr[i]
            if border[right+1] > 0: 
                right = border[right+1]
            if border[left-1] > 0:
                left = border[left-1]
            border[left], border[right] = right, left
            if right-arr[i] == m or arr[i]-left == m:
                ans=i
        
        return ans
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        res = -1
        
        # pos: length
        lenMap = dict()
        # length: cnt
        cntMap = collections.defaultdict(lambda: 0)
        
        for i,x in enumerate(arr):
            left = 0
            right = 0
            
            if x - 1 in lenMap: 
                left = lenMap[x - 1]
                cntMap[left] -= 1
            if x + 1 in lenMap: 
                right = lenMap[x + 1]
                cntMap[right] -= 1
                
            newLen = 1 + left + right
            lenMap[x] = lenMap[x - left] = lenMap[x + right] = newLen
            cntMap[newLen] += 1
            
            if cntMap[m] != 0:
                res = i + 1
                
        return res
                

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        n = len(arr)
        uf = UnionFind(n)
        res = [0]*n
        ans = -1
        for i in range(n):
            step = i+1
            index = arr[i]-1
            res[index] = 1
            uf.size[index] = 1
            uf.count[1] += 1
            
            if index-1 >= 0 and res[index-1] == 1:
                uf.union(index-1, index)
            
            if index+1 < n and res[index+1] == 1:
                uf.union(index, index+1)
            
            if uf.count[m] > 0:
                ans = step
        
        return ans
            
        

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.size = [0]*n
        self.count = Counter()
        
    def union(self, x, y):
        rootx = self.find(x)
        rooty = self.find(y)
        
        if rootx == rooty:
            return False
        
        size_x = self.size[rootx]
        size_y = self.size[rooty]
        
        self.count[size_x] -= 1
        self.count[size_y] -= 1
        
        new_size = size_x + size_y
        
        self.parent[rooty] = rootx
        self.size[rootx] = new_size
        self.count[new_size] += 1
        
        return True
    
    def find(self, x):
        while x != self.parent[x]:
            # self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        start_mp, end_mp = {}, {}
        max_step = -1
        m_cnt = 0
        # use the merge's mind
        for idx, num in enumerate(arr): 
            l = 1
            if num-1 not in end_mp and num+1 not in start_mp: 
                start_mp[num] = end_mp[num] = 1
                start_index = end_index = num
            elif num-1 in end_mp and num +1 in start_mp:
                # merge
                old_l = end_mp[num-1]
                old_r = start_mp[num+1]
                if old_l == m: 
                    m_cnt -= 1
                if old_r == m: 
                    m_cnt -= 1
                start_index = num-1 - end_mp[num-1] + 1
                end_index = num+1 + start_mp[num+1] -1
                l = end_mp[num-1] + start_mp[num+1] + 1
                del end_mp[num-1], start_mp[num-1-old_l+1], start_mp[num+1], end_mp[num+1+old_r-1]
                start_mp[start_index] = l
                end_mp[end_index] = l
            elif num-1 in end_mp: 
                # extend to the left
                old_l = end_mp[num-1]
                if old_l == m: 
                    m_cnt -= 1
                l = old_l + 1
                del end_mp[num-1], start_mp[num-1-old_l+1]
                start_index = num-l+1 
                end_index = m
                end_mp[num] = start_mp[num-l+1] = l
            elif num+1 in start_mp: 
                old_l = start_mp[num+1]
                if old_l == m: 
                    m_cnt -= 1
                l = old_l + 1
                del end_mp[num+1+old_l-1], start_mp[num+1]
                start_mp[num] = end_mp[num+l-1] = l
                start_index = num
                end_index = num+l-1
            
            if l == m: 
                m_cnt += 1
            if m_cnt > 0: 
                max_step = max(max_step, idx + 1)
        return max_step
                
    def findLatestStep(self, arr: List[int], m: int) -> int:
        # actually the above code could be much simpler 
        # keyed by length, cnt is the value
        lcnt = collections.Counter()
        l = collections.Counter()
        max_step = -1
        for idx, num in enumerate(arr): 
            left, right = l[num-1], l[num+1]
            new_l = left + right + 1
            l[num-left] = l[num] = l[num+right] = new_l # this step is the key
            lcnt[left] -=1
            lcnt[right] -= 1
            lcnt[new_l] += 1
            if lcnt[m]: 
                max_step = idx + 1
        return max_step
            
            

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        n = len(arr)
        parent = [i for i in range(n)]
        cnt = [0]*n
        self.m_cnt = 0
        def find(node):
            if parent[node] != node:
                parent[node] = find(parent[node])
            return parent[node]
        
        def union(n1, n2):
            old = (cnt[find(n1)] == m) + (cnt[find(n2)] == m)
            cnt[find(n2)] += cnt[find(n1)]
            parent[find(n1)] = find(n2)
            new = (cnt[find(n2)] == m)
            self.m_cnt += new-old
            
        string = [0]*n
        ans = -1
        for i in range(n):
            j = arr[i]-1
            string[j] = 1
            cnt[j] = 1
            if m == 1:
                self.m_cnt += 1
            if j>0 and string[j-1] == 1:
                union(j-1, j)
            if j<n-1 and string[j+1] == 1:
                union(j, j+1)
            if self.m_cnt > 0:
                ans = i+1
        return ans
        

class Solution:
    def findLatestStep(self, a: List[int], m: int) -> int:
        n = len(a)
        left, right = [0] * (n + 2), [0] * (n + 2)
        count = [0] * (n + 2)
        ans = -1
        for i, x in enumerate(a):
            left[x], right[x] = x, x
            count[1] += 1
            if left[x - 1] != 0:
                count[1] -= 1
                right[left[x - 1]] = x
                left[x] = left[x - 1]
                count[x - left[x - 1]] -= 1
                count[x - left[x - 1] + 1] += 1
            if right[x + 1] != 0:
                right[x] = right[x + 1]
                right[left[x]] = right[x + 1]
                left[right[x + 1]] = left[x]
                count[right[x + 1] - x] -= 1
                count[x - left[x] + 1] -= 1
                count[right[x] - left[x] + 1] += 1
            if count[m] > 0:
                ans = i + 1
        return ans 
class Solution:
    
    def find(self, a):
        if self.parent[a] != a:
            self.parent[a] = self.find(self.parent[a])
        return self.parent[a]
    
    def union(self, a, b):
        r1 = self.find(a)
        r2 = self.find(b)
        if r1 == r2:
            return
        if self.size[r1] == self.m:
            self.num_m -= 1
        if self.size[r2] == self.m:
            self.num_m -= 1
        ns = self.size[r1] + self.size[r2]
        if ns == self.m:
            self.num_m += 1
            
        if self.rank[r1] > self.rank[r2]:
            self.parent[r2] = r1
            self.size[r1] = ns
        else:
            if self.rank[r2] > self.rank[r1]:
                self.parent[r1] = r2
                self.size[r2] = ns
            else:
                self.parent[r1] = r2
                self.rank[r2] += 1
                self.size[r2] = ns
    
    def findLatestStep(self, arr: List[int], m: int) -> int:
        self.parent = {}
        self.rank = {}
        self.size = {}
        self.m = m
        self.num_m = 0
        ans = -1
        for i, v in enumerate(arr):
            self.parent[v] = v
            self.rank[v] = 1
            self.size[v] = 1
            if self.size[v] == m:
                self.num_m += 1
            if v-1 in self.parent:
                self.union(v-1, v)
            if v+1 in self.parent:
                self.union(v+1, v)
            if self.num_m > 0:
                ans = i+1
            # print(i,self.num_m,ans)
        return ans

from collections import Counter
from typing import List

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:        
        n = len(arr)
        last = -2        
        sizes = Counter()        
        starts, ends = {}, {}
            
        def dec_sizes(size):
            sizes[size] -= 1
            if sizes[size] <= 0:
                sizes.pop(size)
    
        for i, x in enumerate(arr):
            x -=1                        
            count = 1            
            start = x
            end = x
            
            left = x - 1
            if left >= 0 and left in ends:
                len_ = ends.pop(left)
                j = left - len_ + 1                
                starts.pop(j)
                start = j
                count += len_                
                dec_sizes(len_)
                            
            right = x + 1
            if right < n and right in starts:
                len_ = starts.pop(right)
                j = right + len_ - 1
                ends.pop(j)
                end = j
                count += len_
                dec_sizes(len_)
                
            starts[start] = count
            ends[end] = count                                             
            sizes[count] += 1
            
            if m in sizes:
                last = i
        
        return last + 1

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        
        if m==len(arr): return m
        
        border=[0]*(len(arr)+2)
        ans=-1
        
        for i in range(len(arr)):
            left=right=arr[i]
            if border[right+1]>0: right=border[right+1]
            if border[left-1]>0: left=border[left-1]
            border[left], border[right] = right, left
            if (right-arr[i]==m) or (arr[i]-left ==m): ans=i
        
        return ans
class UnionFindSet:
    def __init__(self, n):
        self.parents = [i for i in range(n)]
        self.ranks = [0]*n
    
    def find(self, u):
        if u != self.parents[u]:
            self.parents[u] = self.find(self.parents[u])
        return self.parents[u]

    def union(self, u, v):
        pu, pv = self.find(u), self.find(v)
        if pu == pv:
            return False
        if self.ranks[pu] > self.ranks[pv]:
            self.parents[pv] = pu
            self.ranks[pu] += self.ranks[pv]
        else:
            self.parents[pu] = pv
            self.ranks[pv] += self.ranks[pu]
        return True
    
class Solution:    
    def findLatestStep(self, arr: List[int], m: int) -> int:
        if m == len(arr):
            return m
        
        n, ans = len(arr), -1
        uf = UnionFindSet(n)
        
        for step, i in enumerate(arr):
            i -= 1
            uf.ranks[i] = 1
            for j in (i - 1, i + 1):
                if 0 <= j < n:
                    if uf.ranks[uf.find(j)] == m:
                        ans = step
                    if uf.ranks[j]:
                        uf.union(i, j)
        print((uf.ranks))
        return ans

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        N = len(arr)
        parents = list(range(N))
        sizes = [0] * N
        
        islands = set()
        
        def union(one, two):
            r1 = findroot(one)
            r2 = findroot(two)
            
            if r1 == r2: return sizes[r1]
            if sizes[r1] == m and r1 in islands:
                islands.remove(r1)
            if sizes[r2] == m and r2 in islands:
                islands.remove(r2)
            big, small = (r1, r2) if sizes[r1] > sizes[r2] else (r2, r1)
            parents[small] = big
            sizes[big] += sizes[small]
            return sizes[big]
        
        def findroot(pos):
            if parents[pos] != pos:
                parents[pos] = findroot(parents[pos])
            return parents[pos]
        
        last_round = -1
        for i, pos in enumerate(arr, 1):
            pos -= 1
            sizes[pos] += 1
            sz = sizes[pos]
            if pos < N - 1 and sizes[pos + 1]:
                sz = union(pos, pos+1)
            if pos > 0 and sizes[pos - 1]:
                sz = union(pos-1, pos)
            if sz == m:
                islands.add(findroot(pos))
            if islands:
                last_round = i
        
        return last_round
            

class UnionFind:
    
    def __init__(self, n):
        # every node's parent is itself
        self.parents = list(range(n))
        self.ranks = [0] * n
        
    def find(self, u):
        if u != self.parents[u]:
            self.parents[u] = self.find(self.parents[u])
        return self.parents[u]
    
    def union(self, u, v):
        pu, pv = self.find(u), self.find(v)
        
        if pu == pv:
            return 
        
        if self.ranks[pu] > self.ranks[pv]:
            self.parents[pv] = pu
            self.ranks[pu] += self.ranks[pv]
        
        elif self.ranks[pv] > self.ranks[pu]:
            self.parents[pu] = pv
            self.ranks[pv] += self.ranks[pu]
            
        else:
            self.parents[pu] = pv
            self.ranks[pv] += self.ranks[pu]

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        n, ans = len(arr), -1
        uf = UnionFind(n)
        
        if m == n:
            return m
        
        for step, i in enumerate(arr):    
            i -= 1
            uf.ranks[i] = 1
            for j in (i - 1, i + 1):
                if 0 <= j < n:
                    if uf.ranks[uf.find(j)] == m:
                        ans = step    
                    if uf.ranks[j]:
                        uf.union(i, j)
            
        return ans
# class Solution:
#     def findLatestStep(self, arr: List[int], m: int) -> int:
#         bits = [0]*len(arr)
        
#         #2 choice for key: step value or index locations
        
#         for i in reversed(range(len(arr))):
            
from collections import Counter
class DSU:
    def __init__(self,n):
        self.par = [x for x in range(n)]
        self.sz = [1]*n
    
    def find(self,x):
        if self.par[x]!=x:
            self.par[x] = self.find(self.par[x])
        return self.par[x]
    
    def union(self,x,y):
        xp,yp = self.find(x),self.find(y)
        if xp == yp:
            return False
        if self.sz[xp] < self.sz[yp]:
            xp,yp = yp,xp
            
        self.par[yp] = xp
        self.sz[xp] += self.sz[yp]
        self.sz[yp] = self.sz[xp]
        return True
        
    def size(self,x):
        xp = self.find(x)
        return self.sz[xp]
    
class Solution:
    def findLatestStep(self, arr,m):
        res = -1
        n = len(arr)
        dsu = DSU(n)
        
        A = [0]*n
        count = Counter()
        for i,x in enumerate(arr,start=1):
            x -= 1
            A[x] = 1
            this = 1
            if x-1>=0 and A[x-1]:
                left = dsu.size(x-1)
                dsu.union(x,x-1)
                this += left
                count[left] -= 1
            if x+1<n and A[x+1]:
                right = dsu.size(x+1)
                dsu.union(x,x+1)
                this += right
                count[right] -=1
                
            count[this] += 1
            if count[m] >0:
                res = i
        return res
class Solution:
    
    def find(self, d, x):
        while x != d[x]:
            x = d[x]
        return x
    
    def union(self, d, x, y):
        px, py = self.find(d, x), self.find(d, y)
        if px != py:
            d[px] = py
    
    def findLatestStep(self, arr: List[int], m: int) -> int:
        n, step, rec = len(arr), -1, 0
        s, d, d_len = [0]*n, {i:i for i in range(n)}, [1]*n
        for i in range(n):
            num = arr[i] - 1
            s[num] = 1
            for j in (num-1, num+1):
                if j >= 0 and j < n and s[j] == 1:
                    temp = d_len[self.find(d, j)]
                    if temp == m:
                        rec -= 1
                    self.union(d, j, num)
                    d_len[num] += temp
            if d_len[num] == m:
                rec += 1
            if rec > 0:
                step = i+1
        return step

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        class DSU:
            def __init__(self, n, m):
                self.parent = [i for i in range(n)]
                self.size = [0] * n
                self.m = m
                self.cntm = 0
                
            def add(self, x):
                self.size[x] = 1
                if self.m == 1:
                    self.cntm += 1
                self.unite(x - 1, x)
                self.unite(x, x + 1)

            def find(self, x):
                if self.parent[x] != x:
                    self.parent[x] = self.find(self.parent[x])
                return self.parent[x]
            
            def unite(self, x, y):
                if x < 0 or self.size[x] == 0 or y == len(self.size) or self.size[y] == 0:
                    return 
                px, py = self.find(x), self.find(y)
                self.cntm -= self.size[px] == self.m
                self.cntm -= self.size[py] == self.m

                if self.size[px] < self.size[py]:
                    px, py = py, px
                self.size[px] += self.size[py]
                self.parent[py] = px
                self.cntm += self.size[px] == self.m
                
            
        n = len(arr)
        dsu = DSU(n, m)
        latest = -1
        
        for i in range(n):
            dsu.add(arr[i] - 1)
            if dsu.cntm:
                latest = i + 1
        return latest

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        n = len(arr)
        if m == n:
            return n
        
        def dfs(start: int, end: int, step: int, target: int):
            if end > len(arr) or end < 1:
                return -1
            if start > len(arr) or start < 1:
                return -1
            if end < start:
                return -1
            if end - start + 1 < target:
                return -1

            if end - start + 1 == target:
                return step
            bp = arr[step - 1]
            res = -1
            if start <= bp <= end:
                res = max(dfs(start, bp - 1, step - 1, target), dfs(bp + 1, end, step - 1, target))
            else:
                res = max(res, dfs(start, end, step - 1, target)) 
            return res
        
        return dfs(1, n, n, m)
        
        

from collections import Counter
class DSU:
    def __init__(self,n):
        self.par = [x for x in range(n)]
        self.sz = [1]*n
    
    def find(self,x):
        if self.par[x]!=x:
            self.par[x] = self.find(self.par[x])
        return self.par[x]
    
    def union(self,x,y):
        xp,yp = self.find(x),self.find(y)
        if xp == yp:
            return False
        if self.sz[xp] < self.sz[yp]:
            xp,yp = yp,xp
            
        self.par[yp] = xp
        self.sz[xp] += self.sz[yp]
        self.sz[yp] = self.sz[xp]
        return True
        
    def size(self,x):
        xp = self.find(x)
        return self.sz[xp]
    
class Solution:
    def findLatestStep(self, arr,m):
        res = -1
        n = len(arr)
        dsu = DSU(n)
        
        A = [0]*n
        count = Counter()
        for i,x in enumerate(arr,start=1):
            x -= 1
            A[x] = 1
            this = 1
            if x-1>=0 and A[x-1]:
                left = dsu.size(x-1)
                dsu.union(x,x-1)
                this += left
                count[left] -= 1
            if x+1<n and A[x+1]:
                right = dsu.size(x+1)
                dsu.union(x,x+1)
                this += right
                count[right] -=1
                
            count[this] += 1
            if count[m] >0:
                res = i
        return res
class Solution:
    def findLatestStep(self, a: List[int], m: int) -> int:
        n = len(a)
        ans = -1
        d = {}
        ls = [0]*(n+3)
        for ind, i in enumerate(a):
            j,k = i-1, i+1
            ls[i] = 1
            l,r = ls[j], ls[k]
            # while j<n+2 and ls[j] == 1:
            #     l += 1
            #     j += 1
            # while k>0 and ls[k] == 1:
            #     r += 1
            #     k -= 1
            d[l] = d.get(l, 0) - 1
            d[r] = d.get(r, 0) - 1
            d[l+r+1] = d.get(l+r+1, 0) + 1
            ls[i-l] = l+r+1
            ls[i+r] = l+r+1
            # print(l, r, d, ls)
            if m in d and d[m]>0:
                # print(d, a, m, ls)
                ans = ind+1
            
        return ans
            

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        if len(arr) == m:
            return m 
        groups = [0] * (len(arr) + 2)
        res = -1
        for i, num in enumerate(arr):
            left, right = groups[num - 1], groups[num + 1]
            groups[num-groups[num-1]] = left + right + 1
            groups[num+groups[num+1]] = left + right + 1
            if left == m or right == m :
                res = i
        return res

class UnionFindSet:
    def __init__(self, n):
        self.parents = list(range(n))
        self.ranks = [0] * n
        self.size = [0] * n
        
    def find(self, u):
        if u != self.parents[u]:
            self.parents[u] = self.find(self.parents[u])
        return self.parents[u]
    
    def union(self, u, v):
        pu, pv = self.find(u), self.find(v)
        if pu == pv:
            return False
        if self.ranks[pu] > self.ranks[pv]:
            self.parents[pv] = pu
            self.size[pu] += self.size[pv]
        elif self.ranks[pu] < self.ranks[pv]:
            self.parents[pu] = pv
            self.size[pv] += self.size[pu]
        else:
            self.parents[pu] = pv
            self.ranks[pv] += 1
            self.size[pv] += self.size[pu]
        return True

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        if m == len(arr):
            return m
        
        n, ans = len(arr), -1
        uf = UnionFindSet(n)
        
        for step, i in enumerate(arr):
            i -= 1
            uf.ranks[i] = 1
            uf.size[i] = 1
            for j in (i - 1, i + 1):
                if 0 <= j < n:
                    # if uf.ranks[uf.find(j)] == m:
                    if uf.size[uf.find(j)] == m:
                        ans = step
                    if uf.ranks[j]:
                        uf.union(i, j)
        
        return ans
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        n = len(arr)
        if m == n: return n 
        arr.append(n+1)
        start = defaultdict(dict)
        finish = defaultdict(dict)
        last = -1
        for level,i in enumerate(arr):
            if i-1 not in finish: finish[i-1] = i 
            if i+1 not in start: start[i+1] = i

            s, f = finish[i-1], start[i+1]
            start[s] = f 
            finish[f] = s
            
            for os, of in [[i+1, start[i+1]], [finish[i-1], i-1]]:
                if of-os+1 == m: last = level
        
        return last
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        start_mp, end_mp = {}, {}
        max_step = -1
        m_cnt = 0
        for idx, num in enumerate(arr): 
            l = 1
            if num-1 not in end_mp and num+1 not in start_mp: 
                start_mp[num] = end_mp[num] = 1
                start_index = end_index = num
            elif num-1 in end_mp and num +1 in start_mp:
                # merge
                old_l = end_mp[num-1]
                old_r = start_mp[num+1]
                if old_l == m: 
                    m_cnt -= 1
                if old_r == m: 
                    m_cnt -= 1
                start_index = num-1 - end_mp[num-1] + 1
                end_index = num+1 + start_mp[num+1] -1
                l = end_mp[num-1] + start_mp[num+1] + 1
                del end_mp[num-1], start_mp[num-1-old_l+1], start_mp[num+1], end_mp[num+1+old_r-1]
                start_mp[start_index] = l
                end_mp[end_index] = l
            elif num-1 in end_mp: 
                # extend to the left
                old_l = end_mp[num-1]
                if old_l == m: 
                    m_cnt -= 1
                l = old_l + 1
                del end_mp[num-1], start_mp[num-1-old_l+1]
                start_index = num-l+1 
                end_index = m
                end_mp[num] = start_mp[num-l+1] = l
            elif num+1 in start_mp: 
                old_l = start_mp[num+1]
                if old_l == m: 
                    m_cnt -= 1
                l = old_l + 1
                del end_mp[num+1+old_l-1], start_mp[num+1]
                start_mp[num] = end_mp[num+l-1] = l
                start_index = num
                end_index = num+l-1
            
            if l == m: 
                m_cnt += 1
            if m_cnt > 0: 
                max_step = max(max_step, idx + 1)
        return max_step
                
                
            
            
            
                
            
            
            
            
            

def getGroup(groupsByNum, n):
    path = []
    while groupsByNum[n] != n:
        path.append(n)
        n = groupsByNum[n]
    for x in path:
        groupsByNum[x] = n
    return n

def joinGroups(a, b, groupsByNum, sizeByGroup, groupBySize):
    try:
        b = getGroup(groupsByNum, b)
        a = getGroup(groupsByNum, a)
        if a != b:
            aSize = sizeByGroup[a]
            bSize = sizeByGroup[b]
            if aSize > bSize:
                a, b = b, a
                aSize, bSize = bSize, aSize
            
            groupsByNum[a] = b
            del sizeByGroup[a]
            sizeByGroup[b] += aSize
            groupBySize[aSize].remove(a)
            groupBySize[bSize].remove(b)
            try:
                groupBySize[sizeByGroup[b]].add(b)
            except KeyError:
                groupBySize[sizeByGroup[b]] = {b}
            
            
            return True
        else:
            return False
    except KeyError:
        return False


class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        sizeByGroup = dict()
        groupBySize = {1: set()}
        groupsByNum = dict()
        

        result = -1
        for step, x in enumerate(arr, 1):
            groupsByNum[x] = x
            groupBySize[1].add(x)
            sizeByGroup[x] = 1
            joinGroups(x, x + 1, groupsByNum, sizeByGroup, groupBySize)
            joinGroups(x, x - 1, groupsByNum, sizeByGroup, groupBySize)
            try:
                if len(groupBySize[m]) > 0:
                    result = step
            except KeyError:
                pass
                
            
        return result
        
        

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        res = -1
        
        # pos: length
        lenMap = dict()
        # length: cnt
        cntMap = collections.defaultdict(lambda: 0)
        
        for i,x in enumerate(arr):
            left = 0
            right = 0
            
            if x - 1 in lenMap: 
                left = lenMap[x - 1]
                cntMap[left] -= 1
            if x + 1 in lenMap: 
                right = lenMap[x + 1]
                cntMap[right] -= 1
                
            newLen = 1 + left + right
            lenMap[x] = newLen
            lenMap[x - left] = newLen
            lenMap[x + right] = newLen
            cntMap[newLen] += 1
            
            if cntMap[m] != 0:
                res = i + 1
                
        return res
                

class UnionFind:

    def __init__(self):
        self.parents = {}
        self.size = {}
        self.counts = defaultdict(int)

    def add(self,val):
        self.parents[val] = val
        self.size[val] = 1
        self.counts[1] += 1

    def find(self,u):
        if self.parents[u] != u:
            self.parents[u] = self.find(self.parents[u])

        return self.parents[u]

    def union(self,u,v):

        if not u in self.parents or not v in self.parents:
            return

        pU = self.find(u)
        pV = self.find(v)

        if pU == pV:
            return

        self.counts[self.size[pU]]-=1
        self.counts[self.size[pV]]-=1

        if self.size[pU] < self.size[pV]:
            self.parents[pU] = self.parents[pV]
            self.size[pV] += self.size[pU]
            self.counts[self.size[pV]] += 1
        else:
            self.parents[pV] = self.parents[pU]
            self.size[pU] += self.size[pV]
            self.counts[self.size[pU]] += 1

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        uf = UnionFind()
        iter = 1
        res = -1

        for num in arr:
            uf.add(num)

            uf.union(num,num-1)
            uf.union(num,num+1)

            if uf.counts[m] > 0:
                res = iter

            iter += 1

        return res
from collections import defaultdict

class Solution:
    def findLatestStep(self, arr_: List[int], m: int) -> int:
        arr = [a - 1 for a in arr_]
        n = len(arr)
        last_round = -1

        length = defaultdict(int)
        count = defaultdict(int)

        for i, a in enumerate(arr):
            left, right = length[a-1], length[a+1]
            new_length = left + right + 1

            length[a-left] = new_length
            length[a+right] = new_length

            count[new_length] += 1
            count[left] -= 1
            count[right] -= 1

            if count[m] > 0:
                last_round = i + 1
        
        return last_round
        

# class UnionFind:
#     def __init__(self, n):
#         self.parent = {}
#         self.rank = [0] * (n+1)
#         self.group_size = defaultdict(list)
    
#     def find(self, x):
#         if x not in self.parent:
#             self.parent[x] = x
#             self.rank[x] = 1
#             self.group_size[1].append(x)

class UnionFind:
        def __init__(self, m, n):
            self.m = m
            self.parents = [i for i in range(n+1)]
            # self.ranks = [1 for _ in range(n)]
            self.group_size = defaultdict(set)
            # self.group_size[1] = set([i+1 for i in range(n)])
            self.sizes = defaultdict(int)
            # for i in range(n):
            #     self.sizes[i+1] = 1

        def find(self, x):
            if self.parents[x]!=x:
                self.parents[x] = self.find(self.parents[x])
            return self.parents[x]

        def union(self, x, y):
            root_x, root_y = self.find(x), self.find(y)
            # print("x", x ,"y", y)
            # print("root_x", root_x ,"root_y", root_y)
            self.parents[root_x] = root_y
            size_root_x = self.sizes[root_x]
            self.sizes[root_x] = 0
            self.group_size[size_root_x].remove(root_x)

            size_root_y = self.sizes[root_y]
            self.group_size[size_root_y].remove(root_y)
            self.sizes[root_y] = size_root_y + size_root_x
            self.group_size[self.sizes[root_y]].add(root_y)
            
            
            # print("len(self.group_size[self.m])", len(self.group_size[self.m]))
            if len(self.group_size[self.m])>0:
                return True
            else:
                return False
            

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        n = len(arr)
        uf = UnionFind(m, n)
        # print(uf.group_size)
        # print(uf.sizes)
        seen = set()
        res = -1
        for idx, x in enumerate(arr):
            seen.add(x)
            uf.sizes[x] = 1
            uf.group_size[1].add(x)
            if x-1 in seen:
                uf.union(x, x-1)
                # if len(uf.group_size[m])>0:
                #     res = idx+1
            if x+1 in seen:        
                uf.union(x+1, x)
                
            if len(uf.group_size[m])>0:
                res = idx+1
            # print("uf.group_size", uf.group_size)
            # print("uf.sizes", uf.sizes)
        return res
            

#https://leetcode.com/problems/find-latest-group-of-size-m/discuss/836441/Very-Easy

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        n = len(arr)
        # if n==m:
        #     return n
        
        memo = collections.defaultdict(lambda : 0)
        res = -1
        for idx, i in enumerate(arr):
            left = memo[i-1]
            right = memo[i+1]
            if left==m or right==m:
                res = idx
            memo[i-left] = left+right+1
            memo[i+right] = left+right+1
            
        # print (dict(memo))
        if m==memo[1]:
            res = n
        
        return res
            

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        '''
        maintain a DS to record all segments (start, end).
        For each arr[i], test if i-1 or i+1 are in the DS.
        If so, merge the interval and insert to the DS again.
        
        DS:
        - Insert interval
        - Delete interval
        '''
        
        starts = {}
        ends = {}
        intervals = {}
        interval_count = 0
        
        ans = -1
        
        for i, x in enumerate(arr):
            s, e = x, x
            if x - 1 in ends:
                iid = ends[x - 1]
                s = intervals[iid][0]
                if intervals[iid][1] - intervals[iid][0] + 1 == m:
                    ans = max(ans, i)
                del starts[intervals[iid][0]]
                del ends[intervals[iid][1]]
            if x + 1 in starts:
                iid = starts[x + 1]
                e = intervals[iid][1]
                if intervals[iid][1] - intervals[iid][0] + 1 == m:
                    ans = max(ans, i)
                del starts[intervals[iid][0]]
                del ends[intervals[iid][1]]
            iid = interval_count
            interval_count += 1
            intervals[iid] = (s, e)
            starts[s] = iid
            ends[e] = iid
            # print(iid, s, e)
            # print(starts, ends, intervals)
            if e - s + 1 == m:
                ans = max(ans, i + 1)
        
        return ans
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        
        parent = {}
        size = {}
        sizes = collections.defaultdict(int)
        
        def union(a, b):
            fa, fb = find(a), find(b)
            parent[fa] = fb
            sizes[size[fa]] -= 1
            sizes[size[fb]] -= 1
            size[fb] += size[fa]
            sizes[size[fb]] += 1
            
        def get_size(a):
            return size[find(a)]
            
        def find(a):
            if a not in parent:
                parent[a] = a
                size[a] = 1
                sizes[1] += 1
                return parent[a]
            if parent[a] != a:
                parent[a] = find(parent[a])
            return parent[a]
        
        bits = [0] * len(arr)
        res = -1
        for i, n in enumerate(arr):
            idx = n-1
            bits[idx] = 1
            if idx - 1 >= 0 and bits[idx-1] == 1:
                union(idx, idx-1)
            if idx + 1 < len(bits) and bits[idx + 1] == 1:
                union(idx, idx+1)
            sz = get_size(idx)
            if sizes[m] > 0:
                res = i + 1
        return res
            
    
    
    
    def findLatestStep1(self, arr: List[int], m: int) -> int:
        res, n = -1, len(arr)
        length, cnts = [0] * (n+2), [0] * (n+1)
        for i in range(n):
            a = arr[i]
            left, right = length[a-1], length[a+1]
            newlen = left + right + 1
            length[a] = newlen
            length[a - left] = newlen
            length[a + right] = newlen
            cnts[left] -= 1
            cnts[right] -= 1
            cnts[length[a]] += 1
            if cnts[m] > 0:
                res = i + 1
        return res
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        n = len(arr)
        if m == n:
            return n
        size = [0] * (n + 1)
        parents = list(range(n + 1))
        def find(x):
            if parents[x] != x:
                parents[x] = find(parents[x])
            return parents[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if size[px] < size[py]:
                px, py = py, px
            parents[py] = px
            size[px] += size[py]
        
        result = -1
        for j, i in enumerate(arr):
            size[i] = 1
            if i > 1 and size[find(i - 1)] == m:
                result = j
            if i < n and size[find(i + 1)] == m:
                result = j
            if i > 1 and size[i - 1]: 
                union(i, i - 1)
            if i < n and size[i + 1]:
                union(i, i + 1)
        return result
        
        
        
#         n = len(arr)
#         if m == n:
#             return n
#         segments = set([(1, n)])
#         not_relevant = set()
#         for i in range(n - 1, -1, -1):
#             if arr[i] in not_relevant:
#                 continue
#             to_add = []
#             to_remove = None
#             for left, right in segments:
#                 if left <= arr[i] <= right:
#                     to_remove = (left, right)
#                     left_len = arr[i] - left
#                     right_len = right - arr[i]
#                     if left_len == m or right_len == m:
#                         return i
#                     if left_len > m:
#                         to_add.append((left, arr[i] - 1))
#                     else:
#                         for j in range(left, arr[i]):
#                             not_relevant.add(j)
#                     if right_len > m:
#                         to_add.append((arr[i] + 1, right))
#                     else:
#                         for j in range(arr[i] + 1, right + 1):
#                             not_relevant.add(j)
#                     break
#             for segment in to_add:
#                 segments.add(segment)
#             if to_remove:
#                 segments.discard(to_remove)
#             if not segments:
#                 return -1
#         return -1

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        #Union find
        records = []
        self.num_m = 0
        for i in range(len(arr)):
            records.append([-1,0])
        def find(i):
            if records[i][0] == -1:
                return -1
            elif records[i][0] == i:
                return i
            else:
                ii = find(records[i][0])
                records[i][0] = ii
                return ii
        def union(i,j):
            ans_i = find(i)
            ans_j = find(j)
            if records[ans_i][1] > records[ans_j][1]:
                records[ans_j][0] = ans_i
                if m == records[ans_i][1]:
                    self.num_m -= 1
                if m == records[ans_j][1]:
                    self.num_m -= 1
                records[ans_i][1] += records[ans_j][1]
                if m == records[ans_i][1]:
                    self.num_m += 1
            else:
                records[ans_i][0] = ans_j
                if m == records[ans_i][1]:
                    self.num_m -= 1
                if m == records[ans_j][1]:
                    self.num_m -= 1
                records[ans_j][1] += records[ans_i][1]
                if m == records[ans_j][1]:
                    self.num_m += 1

        last = -1
        for i, n in enumerate(arr):
            num = n - 1
            records[num] = [num,1]
            if m == 1:
                self.num_m += 1
            if num >= 1 and records[num - 1][0] != -1:
                if records[num-1][1] == 1:
                    union(num - 1, num)
                else:
                    union(num - 1, num)
            if num < len(arr) - 1 and records[num + 1][0] != -1:
                if records[num+1][1] == 1:
                    union(num + 1, num)
                else:
                    union(num + 1, num)
            if self.num_m > 0:
                last = i+1
        return last
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        n = len(arr)
        from collections import defaultdict 
        parents = [i for i in range(n)]
        ranks = [0] * n
        size = [1] * n 
        group_size = [0] * (n+1)
        visited = [False] * n
        
        def find(x):
            if x != parents[x]:
                parents[x] = find(parents[x])
            return parents[x]
        
        def union(x, y):
            r1 = find(x)
            r2 = find(y)
         
            if r1 != r2:
                group_size[size[r1]] -= 1 
                group_size[size[r2]] -= 1 
                size[r1] = size[r2] = size[r1] + size[r2]
                group_size[size[r1]] += 1 
            
                if ranks[r2] > ranks[r1]:
                    r1, r2 = r2, r1 
                
                parents[r2] = r1 
                if ranks[r1] == ranks[r2]:
                    ranks[r1] += 1
        
        ans = -1
        for step, idx in enumerate(arr):
            idx -= 1 
            left, right = idx - 1, idx + 1
            group_size[1] += 1 
            if left >= 0 and visited[left]:
                union(idx, left)
            if right < n  and visited[right]:
                union(idx, right)
            
            visited[idx] = True
            if group_size[m] > 0:
                ans = step + 1 
        
        return ans
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        n = len(arr)
        g_right = {}
        g_left = {}
        lengths = [0] * n # number of cells with length 1, 2, ... , n
        res = -1
        for itr,i in enumerate(arr):
            pos = i-1
            rb = i-1
            lb = i-1
            
            if pos + 1 in g_right:
                rb = g_right[pos + 1]
                end = g_right.pop(pos + 1)
                lengths[end - (pos + 1)] -= 1
            
            if pos - 1 in g_left:
                lb = g_left[pos - 1]
                end = g_left.pop(pos - 1)
                lengths[(pos - 1) - end] -= 1
            
            g_left.update({rb:lb})
            g_right.update({lb:rb})
            lengths[rb - lb] += 1
            # print(lengths)
            if lengths[m-1] > 0:
                res = itr + 1

        return res
        #     for i in g_
        #     if m in s:
        #         res = itr + 1
        # return res
            
            

class Solution:
    def find(self, n):
        if self.par[n] == n:
            return n
        else:
            self.par[n] = self.find(self.par[n])
            return self.par[n]
        
    def union(self, n1, n2):
        p1 = self.find(n1)
        p2 = self.find(n2)
        if self.rank[p1] < self.rank[p2]:
            self.par[p1] = p2
            self.rank[p2] += self.rank[p1]
        elif self.rank[p1] > self.rank[p2]:
            self.par[p2] = p1
            self.rank[p1] += self.rank[p2]    
        else:
            self.par[p2] = p1
            self.rank[p1] += self.rank[p2]
            
        
    def findLatestStep(self, arr: List[int], m: int) -> int:
        N = len(arr)
        self.N = N
        if m == N:
            return N
        
        self.par = list(range(N+1))
        self.rank = [0]*(N+1)
        
        result = -1
        for i, v in enumerate(arr, 1):
            self.rank[v] = 1
            for j in [v-1, v+1]:
                if 1<=j<=N and self.rank[j]:
                    if self.rank[self.find(j)] == m:
                        result = i-1
                    self.union(j, v)

            
        for i in range(1, N+1):
            if self.rank[self.find(i)] == m:
                return N
            
        return result
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        n = len(arr)
        start = {}
        end = {}
        groups = collections.defaultdict(set)
        ans = -1
        for idx,a in enumerate(arr):
            new_start,new_end = a,a
            if a + 1 in start:
                new_end = start[a + 1]
                del start[a + 1]
                groups[new_end - (a + 1 )+ 1].remove((a+1,new_end))
            if a - 1 in end:
                new_start = end[a-1]
                del end[a-1]
                groups[a-1 - new_start + 1].remove((new_start,a-1))
            start[new_start] = new_end
            end[new_end] = new_start
            groups[new_end - new_start + 1].add((new_start,new_end))
            #print(groups)
            if len(groups[m])>0:ans = idx+1
            #print(groups)
        return ans
                
            

# class Solution:
#     def findLatestStep(self, arr: List[int], m: int) -> int:
#         #u4e24u4e2au6570u7ec4uff0cu4e00u4e2au8bb0u5f55u957fu5ea6uff0cu53e6u4e00u4e2au8bb0u5f55u957fu5ea6u7684u4e2au6570
#         length = [0 for _ in range(len(arr)+2)]
#         count = [0 for _ in range(len(arr)+1)]
#         res = -1
#         for i, a in enumerate(arr):
#             #u5148u628au5de6u8fb9u7684u957fu5ea6u548cu53f3u8fb9u76841u957fu5ea6u53d6u51fau6765
#             left, right = length[a-1], length[a+1]
#             #u73b0u5728u8fd9u4e2au4f4du7f6eu7684u957fu5ea6u5c31u662fu5de6u8fb9u7684u957fu5ea6u52a0u4e0au53f3u8fb9u7684u957fu5ea6u52a0u4e0au81eau5df1
#             #u8dddu79bbau4f4du7f6eu7684u5de6u53f3u4e24u8fb9u7684u8fb9u89d2u5904u7684u7d22u5f15u4e5fu4f1au88abu9644u4e0au65b0u7684u503cuff0cu4e4bu540eu7684u8ba1u7b97u53efu80fdu7528u5f97u4e0a
#             length[a] = length[a-left] = length[a+right] = left + right + 1
            
#             #u7136u540eu5c31u662fu66f4u65b0countu4e86
#             count[left] -= 1
#             count[right] -= 1
#             count[length[a]] += 1
            
#             #u5224u65admu662fu5426u8fd8u5b58u5728uff0cu53eau8981mu5b58u5728u90a3u5c31u662fu6ee1u8db3u6761u4ef6u7684u6700u540eu4e00u6b65
#             if count[m]:
#                 res = i+1
#         return res

class UnionFindSet:
    def __init__(self, n):
        self.parents = [i for i in range(n)]
        self.ranks = [0 for _ in range(n)] #u8868u793aindexu4f4du7f6eu76841u7684u957fu5ea6

    def find(self, u):
        if u != self.parents[u]:
            self.parents[u] = self.find(self.parents[u])
        return self.parents[u]
    
    def union(self, u, v):
        pu, pv = self.find(u), self.find(v)
        if pu == pv:
            return False
        
        #u4f5cu4e3au7236u4eb2u8282u70b9u7684u6761u4ef6uff0cu8c01u7684u957fu5ea6u6bd4u8f83u5927u3002u5c31u8ddfu7740u8c01u3002 u5982u679cu957fu5ea6u76f8u540cu65f6u4f1au4e0du4f1au6709u95eeu9898? u611fu89c9u8fd9u91ccu5199u5f97u4e0du5bf9uff0cu957fu5ea6u76f8u540cu65f6u5e94u8be5u770bpuu548cpvu54eau4e2au6bd4u8f83u5927(u5c0f). u4e5fu8bb8u6309u957fu5ea6u6765u4f5cu4e3au7236u4eb2u8282u70b9u5e76u6ca1u6709u5f71u54cduff0cu4e5fu53efu4ee5u5c1du8bd5u627eu7d22u5f15u503cu8f83u5c0fu7684u5143u7d20u4f5cu4e3au7236u4eb2u8282u70b9
        # if self.ranks[pu] > self.ranks[pv]:
        #     self.parents[pv] = self.parents[pu]
        #     self.ranks[pu] += self.ranks[pv] #u8fd9u91ccu53eau4feeu6539puu76841u957fu5ea6u539fu56e0u662fuff0cu4ee5u540eu67e5u8be2pvu7684u957fu5ea6u90fdu4f1au6307u5411puu7684u957fu5ea6
        # else:
        #     self.parents[pu] = self.parents[pv]
        #     self.ranks[pv] += self.ranks[pu]
        self.parents[max(pu,pv)] = min(pu, pv)
        self.ranks[min(pu,pv)] += self.ranks[max(pu,pv)]
             
        return True
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        if len(arr) == m: #u6700u540eu4e00u6b65
            return m
        
        uf = UnionFindSet(len(arr))
        n, ans = len(arr), -1
        
        for step, a in enumerate(arr):
            a -= 1 #u5b57u7b26u4e32u7d22u5f15u4ece0u5f00u59cb
            
            uf.ranks[a] = 1
            #u627eu5de6u53f3u8282u70b9u8fdbu884cu8054u7ed3
            for j in (a-1, a+1):
                if 0 <= j < n:
                    if uf.ranks[uf.find(j)] == m:
                        #u4e0au4e00u6b65u8fd8u5b58u5728mu957fu5ea6u76841u7fa4. u4e5fu8bb8u5728u5176u4ed6u5730u65b9u8fd8u6709u5b58u5728mu957fu5ea6u7684uff0cu4f46u662fu540eu9762u4f1au904du5386u5230. u6ce8u610fu8fd9u91ccu904du5386u4e0du5230u6700u540eu4e00u6b65
                        ans = step
                    #u5982u679cju4f4du7f6eu4e0au662f1,u53731u7684u7fa4u957fu5ea6u5927u4e8e0u3002 u5219u53efu4ee5u8fdbu884cu8fdeu63a5
                    if uf.ranks[j] > 0:
                        uf.union(a , j)
                        
        return ans
#https://leetcode.com/problems/find-latest-group-of-size-m/discuss/836441/Very-Easy

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        n = len(arr)
        if n==m:
            return n
        
        memo = collections.defaultdict(lambda : 0)
        res = -1
        for idx, i in enumerate(arr):
            left = memo[i-1]
            right = memo[i+1]
            if left==m or right==m:
                res = idx
            memo[i-left] = left+right+1
            memo[i+right] = left+right+1
        # if m in memo.values():
        #     res = n
        return res
            

class UnionFind:
    
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0]*n
        self.size = [0]*n
        self.groupCount = [0]*(n+1)
    
    def find(self, x):
        if x != self.parent[x]:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def add(self, x):
        self.size[x] = 1
        self.groupCount[1] += 1
    
    def union(self, x, y):
        px = self.find(x)
        py = self.find(y)
        if px == py:
            return False
        if self.groupCount[self.size[px]] > 0: 
            self.groupCount[self.size[px]] -= 1
        if self.groupCount[self.size[py]] > 0:
            self.groupCount[self.size[py]] -= 1
        if self.rank[px] > self.rank[py]:
            self.parent[py] = px
            self.size[px] += self.size[py]
            self.size[py] = 0
        elif self.rank[py] > self.rank[px]:
            self.parent[px] = py
            self.size[py] += self.size[px]
            self.size[px] = 0
        else:
            self.parent[px] = py
            self.size[py] += self.size[px]
            self.size[px] = 0
            self.rank[py] += 1
        self.groupCount[self.size[px]] += 1
        self.groupCount[self.size[py]] += 1
        return True
    
    def getSize(self, i):
        px = self.find(i)
        return self.size[px]
    
class Solution:
    
    def findLatestStep(self, arr: List[int], m: int) -> int:
        disjoint = UnionFind(len(arr))
        ans = - 1
        val = [0]*len(arr)
        for k in range(len(arr)):
            index = arr[k] - 1
            val[index] += 1
            disjoint.add(index)
            if index > 0 and val[index] == val[index-1]:
                disjoint.union(index, index - 1)
            if index + 1 < len(val) and val[index] == val[index+1]:
                disjoint.union(index, index + 1)
            #print(k, disjoint.groupCount)
            if disjoint.groupCount[m] > 0:
                ans = k + 1
            '''
            i = 0
            while i < len(arr):
                if val[i] == 1 and disjoint.getSize(i) == m:
                    i += disjoint.getSize(i)
                    ans = k + 1
                    continue
                i += 1
            '''
            #print(k, disjoint.size, val)
        return ans 
    
    '''
    def findLatestStep(self, arr: List[int], m: int) -> int:
        def check(i):
            val = [0]*len(arr)
            for k in range(i+1):
                val[arr[k]-1] = 1
            count = 0
            success = False
            for k in range(len(val)):
                if val[k] > 0:
                    count += 1
                else:
                    if count == m:
                        success = True
                        break
                    count = 0
            if count == m:
                success = True
            return success                
            
        left = 0
        right = len(arr)
        while left < right:
            mid = left + (right - left) //2
            if not check(mid):
                right = mid
            else:
                left = mid + 1
        print(left)
        if left == 0 and not check(left):
            return -1
        else:
            return left
    '''

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        
        temp = [0]*len(arr)
        rn = [[None, None] for i in range(len(arr))]
        fans = None
        ans = set()
        st = 0
        for ind in arr:
            st += 1
            i = ind - 1
            temp[i] = 1
            
            if i == len(arr)-1:
                rn[i] = [i, i]
                if temp[i-1] == 1:
                    if tuple(rn[i-1]) in ans:
                        ans.remove(tuple(rn[i-1]))
                    mn, mx = rn[i-1]
                    rn[i] = [min(mn, rn[i][0]), max(mx, rn[i][1])]
                    rn[rn[i][0]] = rn[rn[i][1]] = rn[i]
                    
            elif i == 0:
                rn[i] = [i, i]
                if temp[i+1] == 1:
                    if tuple(rn[i+1]) in ans:
                        ans.remove(tuple(rn[i+1]))
                    mn, mx = rn[i+1]
                    rn[i] = [min(mn, rn[i][0]), max(mx, rn[i][1])]
                    rn[rn[i][0]] = rn[rn[i][1]] = rn[i]
                    
                
            else:
                rn[i] = [i, i]
                if temp[i-1] == 1:
                    if tuple(rn[i-1]) in ans:
                        ans.remove(tuple(rn[i-1]))
                    mn, mx = rn[i-1]
                    rn[i] = [min(mn, rn[i][0]), max(mx, rn[i][1])]
                    rn[rn[i][0]] = rn[rn[i][1]] = rn[i]
                
                if temp[i+1] == 1:
                    if tuple(rn[i+1]) in ans:
                        ans.remove(tuple(rn[i+1]))
                    mn, mx = rn[i+1]
                    rn[i] = [min(mn, rn[i][0]), max(mx, rn[i][1])]
                    rn[rn[i][0]] = rn[rn[i][1]] = rn[i]
            
            if rn[i][1]-rn[i][0] == m-1:
                ans.add(tuple(rn[i]))
            if ans:
                fans = st
        if fans:
            return fans
        return -1
                
            
          
        
                
                

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        
        indeces = {}
        for i in range(len(arr)):
            indeces[arr[i]] = i
            
        leftToRight = {}
        rightToLeft = {}
        result = -1
        numOfGroup = 0
        
        for i in range(len(arr)):
            num = arr[i]
            if rightToLeft.get(num-1) is not None and leftToRight.get(num+1) is not None:
                leftToRight[rightToLeft[num-1]] = leftToRight[num+1]
                rightToLeft[leftToRight[num+1]] = rightToLeft[num-1]
                if leftToRight[num+1] - rightToLeft[num-1] + 1 == m:
                    numOfGroup += 1
                if leftToRight.get(num+1) - (num+1) + 1 == m:
                    numOfGroup -= 1
                if num-1 - rightToLeft.get(num-1) + 1 == m:
                    numOfGroup -= 1
                leftToRight[num+1] = None
                rightToLeft[num-1] = None
            elif rightToLeft.get(num-1) is not None:
                rightToLeft[num] = rightToLeft[num-1]
                leftToRight[rightToLeft[num-1]] = num
                if num - rightToLeft[num-1] + 1 == m:
                    numOfGroup += 1
                if num - 1 - rightToLeft[num-1] + 1 == m:
                    numOfGroup -= 1
                rightToLeft[num-1] = None
            elif leftToRight.get(num+1) is not None:
                leftToRight[num] = leftToRight[num+1]
                rightToLeft[leftToRight[num+1]] = num
                if leftToRight[num+1] - num + 1 == m:
                    numOfGroup += 1
                if leftToRight[num+1] - (num + 1) + 1 == m:
                    numOfGroup -= 1
                leftToRight[num+1] = None
            else:
                leftToRight[num] = num
                rightToLeft[num] = num
                if m == 1:
                    numOfGroup += 1
            if numOfGroup > 0:
                result = i+1
                
        return result
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        table = collections.defaultdict(int)
        size = [0] * len(arr)
        parent = [-1] * len(arr)
        res = 0
        for i in range(len(arr)):
            pos = arr[i] - 1
            idx = self.find(pos, parent)
            if idx == -1:
                parent[pos] = pos
                size[pos] = 1
                table[size[pos]] = table[size[pos]] + 1
            self.unionAround(pos, arr, parent, size, table)
            if m in table:
                res = i + 1
        if res == 0:
            return -1
        return res
        
    
    def unionAround(self, i, arr, parent, size, table):
        if i > 0:
            self.union(i, i-1, parent, size, table)
        if i < len(arr) - 1:
            self.union(i, i+1, parent, size, table)
    
    def union(self, i, j, parent, size, table):
        x = self.find(i, parent)
        y = self.find(j, parent)
        if y == -1:
            return
        if x != y:
            table[size[y]] = table[size[y]]-1
            if table[size[y]] == 0:
                del table[size[y]]
                
            table[size[x]] = table[size[x]]-1
            if table[size[x]] == 0:
                del table[size[x]]
            
            size[y] += size[x]
            parent[x] = y
            
            table[size[y]] = table[size[y]] + 1
    
    def find(self, i, parent):
        if parent[i] == -1:
            return -1
        if parent[i] == i:
            return i
        return self.find(parent[i], parent)
        
        
        
        
        
        

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        parents = {}
        weights = {}
        
        def find(p): 
            if p not in parents:
                parents[p] = p
                weights[p] = 1
            if parents[p] != p:
                parents[p] = find(parents[p])
            return parents[p]
        
        def union(p, q):
            i, j = find(p), find(q)
            if i == j:
                return
            if weights[i] >= weights[j]:
                parents[j] = i
                weights[i] += weights[j] # do not need to reset weights[j] to 0 since we only care about the weights at the root.
            else:
                parents[i] = j
                weights[j] += weights[i]
            
        def connected(p, q):
            return find(p) == find(q)
        
        status = [0]*len(arr)
        
        cnt = collections.Counter()
        last = -1
        
        for step, info in enumerate(arr):
            i = info-1
            status[i] = 1
            
            if i == 0 or status[i-1] == 0:
                left = False
            else:
                left = True
            
            if i >= len(arr)-1 or status[i+1] == 0:
                right = False
            else:
                right = True
            
            # print(left, right)
            if (not left) and (not right):
                cnt[1] += 1
                p = find(arr[i])
                
            elif left and right:
                # combine all three
                pleft = find(arr[i-1])
                pright = find(arr[i+1])
                size_left = weights[pleft]
                size_right = weights[pright]
                cnt[size_left] -= 1
                cnt[size_right] -= 1
                cnt[size_left + size_right + 1] += 1
                union(arr[i-1], arr[i])
                union(arr[i], arr[i+1])
                
            elif left:
                pleft = find(arr[i-1])
                
                size_left = weights[pleft]
                
                cnt[size_left] -= 1
                cnt[size_left + 1] += 1
                union(arr[i-1], arr[i])
                # print(parents, weights, pleft)
                # print("size_left:", size_left)
            elif right:
                pright = find(arr[i+1])
                size_right = weights[pright]
                cnt[size_right] -= 1
                cnt[size_right+1] += 1
                union(arr[i], arr[i+1])
                
            if cnt[m] > 0:
                last = step+1
            # print(cnt)
            
        return last
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        output = -1
        uf = UF()
        seen = set()
        for step, idx in enumerate(arr, 1):
            uf.add(idx)
            seen.add(idx)
            if idx - 1 in seen:
                uf.join(idx - 1, idx)
            if idx + 1 in seen:
                uf.join(idx, idx + 1)
            if uf.size_counts[m]:
                output = step
        return output

class UF:
    
    def __init__(self):
        self.parents = {}
        self.sizes = {}
        self.size_counts = collections.Counter()
        
    def add(self, n):
        self.parents[n] = n
        self.sizes[n] = 1
        self.size_counts[1] += 1
    
    def find(self, n):
        p = self.parents
        while p[n] != n:
            p[n] = p[p[n]]
            n = p[n]
        return n
    
    def join(self, a, b):
        a = self.find(a)
        b = self.find(b)
        if a == b:
            return

        s = self.sizes
        sc = self.size_counts
        sc[s[a]] -= 1
        sc[s[b]] -= 1
        sc[s[a] + s[b]] += 1
        
        p = self.parents
        if s[a] < s[b]:
            p[a] = b
            s[b] += s[a]
        else:
            p[b] = a
            s[a] += s[b]

class Subset:
    def __init__(self,n):
        self.parent = [i for i in range(n)]
        self.rank = [0] * n
    
    def find(self,i):
        if self.parent[i] != i:
            self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self,u,v):
        if self.rank[u] > self.rank[v]:
            self.parent[v] = self.find(u)
        if self.rank[v] > self.rank[u]:
            self.parent[u] = self.find(v)
        if self.rank[u] == self.rank[v]:
            self.parent[v] = self.find(u)
            self.rank[u] += self.rank[v]
            
class Solution:       
    def findLatestStep(self, arr: List[int], m: int) -> int:
        n = len(arr)
        subset = Subset(n)
        ls = [0] * n
        size = [1] * n
        res = -1
        cnt = 0
        for i in range(n):
            idx = arr[i] - 1
            ls[idx] = 1
            sizeMiddle = 1
            if idx > 0:
                if ls[idx-1] == 1:
                    p = subset.find(idx-1)
                    sizeLeft = size[p]
                    subset.union(idx,p)
                    if sizeLeft == m:
                        cnt -= 1
                    sizeMiddle += sizeLeft
            if idx < n-1:
                if ls[idx+1] == 1:
                    p2 = subset.find(idx+1)
                    sizeRight = size[p2]
                    subset.union(idx,p2)
                    if sizeRight == m:
                        cnt -= 1
                    sizeMiddle += sizeRight
            finalP = subset.find(idx)
            size[finalP] = sizeMiddle
            if sizeMiddle == m:
                cnt += 1
            if cnt > 0:
                res = max(res,i+1)
        return res
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        dp = [(-1, -1)] * (len(arr) + 1)
        groups = collections.defaultdict(int)
        if len(arr) == 1:
            return 1
        res = -1
        for i, a in enumerate(arr):
            leftPos, rightPos = a, a
            if a == 1:
                rightPos = a if dp[a+1][1] == -1 else dp[a+1][1]
            elif a == len(arr):
                leftPos = a if  dp[a-1][0] == -1 else dp[a-1][0]
            else:
                leftPos = a if  dp[a-1][0] == -1 else dp[a-1][0]
                rightPos = a if dp[a+1][1] == -1 else dp[a+1][1]
            
            dp[a] = (leftPos, rightPos)
            dp[leftPos] = (leftPos, rightPos)
            dp[rightPos] = (leftPos, rightPos)

            groups[rightPos - leftPos + 1] += 1
            groups[a - leftPos] -= 1
            groups[rightPos - a] -= 1
            if groups[m] >= 1:
                res = i + 1
        return res
class UnionFindSet:
    def __init__(self, n):
        self.parents = list(range(n))
        self.ranks = [0] * n
        
    def find(self, u):
        while u != self.parents[u]:
            self.parents[u] = self.parents[self.parents[u]]
            u = self.parents[u]
        return self.parents[u]
    
    def union(self, u, v):
        pu, pv = self.find(u), self.find(v)
        if pu == pv:
            return False
        if self.ranks[pu] > self.ranks[pv]:
            self.parents[pv] = pu
            self.ranks[pu] += self.ranks[pv]
        else:
            self.parents[pu] = pv
            self.ranks[pv] += self.ranks[pu]
        return True

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        if m == len(arr):
            return m
        
        n, ans = len(arr), -1
        uf = UnionFindSet(n)
        
        for step, i in enumerate(arr):
            i -= 1
            uf.ranks[i] = 1
            for j in (i - 1, i + 1):
                if 0 <= j < n:
                    if uf.ranks[uf.find(j)] == m:
                        ans = step
                    if uf.ranks[j]:
                        uf.union(i, j)
        
        return ans
                
                    
        

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        arr = [i - 1 for i in arr]
        n = len(arr)
        uf = UnionFind(n)
        s = ['0'] * n
        steps = -1
        if m == 1:
            steps = 1
        for i, idx in enumerate(arr):
            s[idx] = '1'
            uf.sz_count[1] += 1
            if idx > 0 and s[idx-1] == '1':
                uf.find_and_union(idx, idx-1)
            if idx < n-1 and s[idx+1] == '1':
                uf.find_and_union(idx, idx+1)
            if uf.sz_count[m] > 0:
                steps = i + 1
        return steps

class UnionFind:
    def __init__(self, n):
        self.component_count = n
        self.parents = list(range(n))
        self.size = [1] * n
        self.sz_count = Counter()

    def find(self, x):
        if self.parents[x] != x:
            self.parents[x] = self.find(self.parents[x])
        return self.parents[x]

    def union(self, x, y):
        if self.size[x] < self.size[y]:
            x, y = y, x
        self.parents[y] = x
        self.sz_count[self.size[x]] -= 1
        self.sz_count[self.size[y]] -= 1
        self.size[x] += self.size[y]
        self.sz_count[self.size[x]] += 1
        self.component_count -= 1

    # return true if two are newly unioned, false if already unioned.
    def find_and_union(self, x, y):
        x0 = self.find(x)
        y0 = self.find(y)
        if x0 != y0:
            return self.union(x0, y0)
        return 0
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        size = [0] * (len(arr) + 2)
        count = collections.defaultdict(int)
        
        answer = -1
        for i, value in enumerate(arr):
            left, right = size[value - 1], size[value + 1]
            size[value] = size[value - left] = size[value + right] = left + right + 1
            count[left] -= 1
            count[right] -= 1
            count[size[value]] += 1
            
            if count[m]:
                answer = i + 1
        
        return answer
        
    def findLatestStep_my(self, arr: List[int], m: int) -> int:
        parent = [0] * (len(arr) + 2)
        size = [1] * (len(arr) + 1)
        
        count = collections.defaultdict(int)
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
                count[size[py]] -= 1
                count[size[px]] -= 1
                size[py] += size[px]
                count[size[py]] += 1
        
        answer = -1
        for i, value in enumerate(arr):
            parent[value] = value
            count[1] += 1
            
            if parent[value - 1]:
                union(value - 1, value)
            if parent[value + 1]:
                union(value, value + 1)
            
            if count[m]:
                answer = i + 1
        
        return answer
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        n = len(arr)
        
        steps = n
        a1 = [[1, n]]
        a2 = [n]
        if m == n:
            return steps
        
        while steps > 0:
            steps -= 1
            i = arr[steps]
            
            l = 0
            r = len(a1) - 1
            
            while l <= r:
                mid = (l + r) // 2
                if a1[mid][0] > i:
                    r = mid - 1
                elif a1[mid][1] < i:
                    l = mid + 1
                else:
                    left = [a1[mid][0], i - 1]
                    right = [i + 1, a1[mid][1]]
                    
                    if left[0] > left[1]:
                        if right[0] > right[1]:
                            a1.pop(mid)
                            a2.pop(mid)
                            break
                        else:
                            a1[mid] = right
                            a2[mid] = right[1] - right[0] + 1
                            if a2[mid] == m:
                                return steps
                    else:
                        if right[0] > right[1]:
                            a1[mid] = left
                            a2[mid] = left[1] - left[0] + 1
                            if a2[mid] == m:
                                return steps
                        else:
                            a1[mid] = right
                            a2[mid] = right[1] - right[0] + 1
                            a1.insert(mid, left)
                            a2.insert(mid, left[1] - left[0] + 1)
                            if a2[mid] == m or a2[mid + 1] == m:
                                return steps
        return -1
class DSU:
    def __init__(self, N):
        self.par = list(range(N))
        self.sz = [1] * N

    def find(self, x):
        if self.par[x] != x:
            self.par[x] = self.find(self.par[x])
        return self.par[x]
    def get_size(self, x):
        return self.sz[self.find(x)]
    
    def union(self, x, y):
        xr, yr = self.find(x), self.find(y)
        if xr == yr:
            return False
        if self.sz[xr] < self.sz[yr]:
            xr, yr = yr, xr
        self.par[yr] = xr
        self.sz[xr] += self.sz[yr]
        self.sz[yr] = self.sz[xr]
        return True
    
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        n = len(arr)
        dsu = DSU(n)
        cur_string = [0] * n
        ans = -1
        cur_m_count = 0
        for step_0indexed, pos_1indexed in enumerate(arr):
            pos_to_be1 = pos_1indexed-1
            cur_string[pos_to_be1] = 1 
            if pos_to_be1 >= 1 and cur_string[pos_to_be1-1] == 1:
                if dsu.get_size(pos_to_be1-1) == m:
                    cur_m_count -= 1
                dsu.union(pos_to_be1, pos_to_be1-1)
            if pos_to_be1 < n-1 and cur_string[pos_to_be1+1] == 1:
                if dsu.get_size(pos_to_be1+1) == m:
                    cur_m_count -= 1
                dsu.union(pos_to_be1, pos_to_be1+1)
                
            if dsu.get_size(pos_to_be1) == m:
                cur_m_count += 1
            if cur_m_count > 0:
                ans = step_0indexed + 1
        return ans
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        n=len(arr)
        d=[[i-1,i+1] for i in range(0,n+2)]
        latest = -1
        for i,j in enumerate(arr):
            a,b = d[j]
            if d[a][1]-a==m+1 or a-d[a][0]==m+1 or d[b][1]-b==m+1 or b-d[b][0]==m+1:
                latest=i
            if b-a==m+1:
                latest=i+1
            if a>=0:
                d[a][1]=b
            if b<=n+1:
                d[b][0]=a            
        return latest

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        n = len(arr)
        p = list(range(n))
        size = [0]*n

        def find(x):
            if x!=p[x]:
                p[x] = find(p[x])
            return p[x]

        def union(x,y):
            px,py = find(x),find(y)
            if px == py:
                return False
            if size[px]>size[py]:
                p[py] = px
                size[px]+=size[py]
            else:
                p[px] =py
                size[py] += size[px]
            return True

        if m == len(arr):
            return m
        ans = -1
        for step,i in enumerate(arr):
            i -= 1
            for j in range(i-1,i+2):
                if 0<=j<n:
                    if size[find(j)]==m:
                        ans = step
            if ans == m:
                break
            size[i] = 1
            for j in range(i-1,i+2):
                if 0<=j<n:
                    if size[j]:
                        union(i,j)
        return ans
class UnionFind:
    def __init__(self):
        self.parent = {}
        self.rank = {}

    def add(self, x):
        self.parent[x] = x
        self.rank[x] = 0

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        xRoot, yRoot = self.find(x), self.find(y)

        if xRoot == yRoot:
            return

        xRank, yRank = self.rank[xRoot], self.rank[yRoot]
        if xRank < yRank:
            yRoot, xRoot = xRoot, yRoot

        self.parent[yRoot] = xRoot
        self.rank[xRoot] += self.rank[yRoot]
        # if self.rank[ yRoot] == self.rank[xRoot]:
        #     self.rank[xRoot] += 1

        return


class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        if m == len(arr):
            return len(arr)
        if m > len(arr):
            return -1
        uf = UnionFind()
        
        for i in range(1, len(arr)+1):
            uf.add(i)
        ans = -1
        seen = set()
        
        for i, n in enumerate(arr):
            uf.rank[n] = 1
            if n - 1 >= 1 and uf.rank[n-1] != 0:
                if uf.rank[uf.find(n-1)] == m:
                    ans = i
                uf.union(n, n-1)
            if n + 1 <= len(arr) and uf.rank[n+1] != 0:
                if uf.rank[uf.find(n+1)] == m:
                    ans = i
                uf.union(n, n + 1)
                
        return ans
        
        
        
        
        

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        # length = [0] * (len(arr) + 2)
        # cnt = [0] * (len(arr) + 1)
        # ans = -1
        # for i, a in enumerate(arr):
        #     left, right = length[a - 1], length[a + 1]
        #     length[a] = length[a - left] = length[a + right] = left + right + 1
        #     cnt[left] -= 1
        #     cnt[right] -= 1
        #     cnt[length[a]] += 1
        #     if cnt[m]:
        #         ans = i + 1
        # return ans
        
        # Union-Find
        uf = {}
        seen = [0] * (len(arr) + 1)

        def find(x):
            uf.setdefault(x, x)
            if uf[x] != x:
                uf[x] = find(uf[x])
            return uf[x]

        def union(x, y):
            seen[find(y)] += seen[find(x)]
            uf[find(x)] = find(y)

        ans, n = -1, len(arr)
        for i, a in enumerate(arr, 1):
            seen[a] = 1
            for b in [a - 1, a + 1]:
                if 1 <= b <= n and seen[b]:
                    if seen[find(b)] == m:
                        ans = i - 1
                    union(a, b)
        if m == n:
            ans = n
        return ans
                
                
                
                
                

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            
            return parent[x]
        
        def union(x, y):
            xr, yr = find(x), find(y)
            if xr == yr:
                return False
            if sizes[xr] < sizes[yr]:
                (xr, yr) = (yr, xr)
            
            parent[yr] = xr
            size_counter[sizes[xr]] -= 1
            size_counter[sizes[yr]] -= 1
            sizes[xr] += sizes[yr]
            size_counter[sizes[xr]] += 1
            
            
        n = len(arr)
        parent = list(range(n + 1))
        sizes = [1] * (n + 1)
        size_counter = [0] * (n + 1)
        last = -2
        status = [False] * (n + 2)
        for i, x in enumerate(arr):
            status[x] = True
            size_counter[1] += 1
            prev = status[x - 1]
            next = status[x + 1]
            if prev:
                union(x, x -1)
            if next:
                union(x, x + 1)
            if size_counter[m]:
                last = i
        
        return last + 1
class Solution:
    # https://leetcode.com/problems/find-latest-group-of-size-m/discuss/806786/JavaC%2B%2BPython-Count-the-Length-of-Groups-O(N)
    def findLatestStep(self, arr: List[int], m: int) -> int:
        if len(arr) == m:
            return m
        length = [0 for _ in range(len(arr) + 2)]
        res = -1
        # n starts from 1.
        for i, n in enumerate(arr):
            left, right = length[n - 1], length[n + 1]
            if left == m or right == m:
                #update res for each time satisfying conditiong. so return the latest one.
                res = i
            # update edge. [0111010], change middle 0 t0 1. left = 3, right = 1.total length = 3 + 1 + 1 = 5. edge, length[1] = 5, length[6] = 5
            length[n - left] = length[n + right] = left + right + 1
        return res
            

class Solution:
    def findLatestStep(self, arr, m):
        D = dict() # D[x] records the index of the end in the interval, +: right end, -: left end
        
        c, ret = 0, -1 # c: count of m-intervals, ret: return index
        for k, x in enumerate(arr, 1):
            D[x], S = 0, 0 # S: shift
            
            # discuss in cases
            if x-1 in D and x+1 in D:
                i, j = D[x-1], -D[x+1]
                if i+1 == m: # i+1 is the length
                    c -= 1
                if j+1 == m:
                    c -= 1
                S = i+j+2
                D[x-i-1], D[x+j+1] = -S, S
            elif x-1 in D:
                i = D[x-1]
                if i+1 == m:
                    c -= 1
                S = i+1
                D[x-i-1], D[x] = -S, S
            elif x+1 in D:
                j = -D[x+1]
                if j+1 == m:
                    c -= 1
                S = j+1
                D[x+j+1], D[x] = S, -S
                
            if S+1 == m: # find a m-inteval
                c += 1
            if c > 0: # no m-interval in this round
                ret = k
        
        return ret
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        N = len(arr)
        
        parents = list(range(N))
        sizes = [0 for _ in range(N)]
        m_count = 0
        
        def find(i):
            if i != parents[i]:
                parents[i] = find(parents[i])
            return parents[i]

        def merge(i, j):
            nonlocal m_count
            parent_i = find(i)
            parent_j = find(j)
            if parent_i != parent_j:
                if sizes[parent_j] == m:
                    m_count -= 1
                if sizes[parent_i] == m:
                    m_count -= 1
                sizes[parent_j] += sizes[parent_i]
                if sizes[parent_j] == m:
                    m_count += 1
                parents[parent_i] = parent_j
                
        groups = [0]*N
        latest_round = -1
        for i, a in enumerate(arr, start=1):
            a -= 1
            groups[a] = 1
            sizes[a] = 1
            if m == 1:
                m_count += 1
            if a-1 >= 0 and groups[a-1] == 1:
                merge(a-1, a)
            if a+1 < N and groups[a+1] == 1:
                merge(a, a+1)
            if m_count > 0:
                latest_round = i
        return latest_round
from typing import List
class UnionFindSet:
    def __init__(self, n):
        self.parent = list(range(n+2))
        self.rank = [0]*(n+2)
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a, b):
        pa, pb = self.find(a), self.find(b)
        if pa == pb:
            return False
        if self.rank[pa] > self.rank[pb]:
            self.parent[pb] = pa
            self.rank[pa] += self.rank[pb]
        else:
            self.parent[pa] = pb
            self.rank[pb] += self.rank[pa]
        return True

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        n = len(arr)
        uf = UnionFindSet(n)
        ans = -1

        for step, idx in enumerate(arr):
            uf.rank[idx] = 1
            for j in [idx+1, idx-1]:
                if uf.rank[uf.find(j)] == m:
                    ans = step
                if uf.rank[j]:
                    uf.union(idx,j)
        for i in range (1,n+1):
            if uf.rank[uf.find(i)] == m:
                return n 
        return ans

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        n = len(arr)
        visited = [0 for _ in range(n)]
        d = [0 for _ in range(n)]
        
        cnt = 0
        ans = -1
        for step, i in enumerate(arr):
            i = i-1
            lhs, rhs = 0, 0
            start, end = i, i
            visited[i] = 1
            if i+1 < n and visited[i+1]:
                rhs = d[i+1]
                end = i + rhs
                if rhs == m:
                    cnt -= 1
            if i- 1 >= 0 and visited[i-1]:
                lhs = d[i-1]
                start = i - lhs
                if lhs == m:
                    cnt -= 1
                    
            length = lhs + rhs + 1
            d[start] = length
            d[end] = length
            # print(start, end, length)
            if length == m:
                cnt += 1
            if cnt > 0:
                ans = step+1
            # print(cnt)
        
        return ans

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        N = len(arr)
        parents = list(range(N))
        sizes = [0] * N
        
        islands = set()
        
        def union(one, two):
            r1 = findroot(one)
            r2 = findroot(two)
            
            if r1 == r2: return sizes[r1]
            if sizes[r1] == m and r1 in islands:
                islands.remove(r1)
            if sizes[r2] == m and r2 in islands:
                islands.remove(r2)
            big, small = (r1, r2) if sizes[r1] > sizes[r2] else (r2, r1)
            parents[small] = big
            sizes[big] += sizes[small]
            return sizes[big]
        
        def findroot(pos):
            if parents[pos] != pos:
                parents[pos] = findroot(parents[pos])
            return parents[pos]
        
        last_round = -1
        for i, pos in enumerate(arr, 1):
            pos -= 1
            sizes[pos] += 1
            sz = sizes[pos]
            if pos < N - 1 and sizes[pos + 1]:
                sz = union(pos, pos+1)
            if pos > 0 and sizes[pos - 1]:
                sz = union(pos-1, pos)
            if sz == m:
                islands.add(findroot(pos))
            if islands:
                last_round = i
        
        return last_round
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        dic = {}
        def count_cluster(y,x,cur_count):
            length = 0
            if y[x-1] == 0:
                if x <n:
                    if y[x+1] ==0:
                        y[x] = 1
                        dic[x] = x
                        if m==1:
                            cur_count+=1
                    else:
                        oldr = y[x+1]
                        y[x] = 1 + y[x+1]
                        y[dic[x+1]] = y[x]
                        dic[x] = dic[x+1]
                        dic[dic[x+1]] = x
                        if oldr == m-1:
                            cur_count +=1
                        if oldr == m:
                            cur_count -=1
                else:
                    y[x] = 1
                    dic[x] = x
                    if m==1:
                        cur_count+=1
            else:
                if x <n:
                    if y[x+1] ==0:
                        oldl = y[x-1]
                        y[x] = y[x-1] +1
                        y[dic[x-1]] = y[x]
                        dic[x] = dic[x-1]
                        dic[dic[x-1]] = x
                        if oldl == m-1:
                            cur_count +=1
                        if oldl == m:
                            cur_count -=1
                    else:
                        oldr = y[x+1]
                        oldl = y[x-1]
                        y[x] = y[x-1] + 1 + y[x+1]
                        temp = dic[x-1]
                        
                        y[dic[x-1]] = y[x]
                        # dic[x] = dic[x-1]
                        dic[dic[x-1]] = dic[x+1]
                        
                        y[dic[x+1]] = y[x]
                        # dic[x] = dic[x+1]
                        dic[dic[x+1]] = temp
                        
                        if oldr==m:
                            cur_count -= 1
                        if oldl ==m:
                            cur_count-=1
                        if oldr+oldl == m-1:
                            cur_count+=1
                else:
                    oldl = y[x-1]
                    y[x] = y[x-1] +1
                    y[dic[x-1]] = y[x]
                    dic[x] = dic[x-1]
                    dic[dic[x-1]] = x
                    if oldl == m-1:
                        cur_count +=1
                    if oldl == m:
                        cur_count -=1
                
            return cur_count     
        n = len(arr)

        s = [0] * (n+1)
        # narr = [(x,idx) for idx, x in enumerate(arr)]
        # x = sorted(narr,key=lambda x: x[0])
        last = -1
        cur_count = 0
        for idx,x in enumerate(arr):
            # print(s,idx,x,cur_count)
            # s[x] = 1
            cur_count=count_cluster(s,x,cur_count)
            if cur_count>0:
                last = idx+1
        print(last)
        return last

class Solution:
    def findLatestStep(self, A: List[int], m: int) -> int:
        res=-1
        x=0
        n=len(A)
        l2c={i:0 for i in range(1,n+1)}
        d={}
        for i in range(1,n+1):
            j=A[i-1]
            start=end=j
            if j-1 in d:
                start=d[j-1][0]
                l=d[j-1][-1]-d[j-1][0]+1
                l2c[l]-=1
            if j+1 in d:
                end = d[j+1][1]
                l=d[j+1][-1]-d[j+1][0]+1
                l2c[l]-=1
            d[start]=[start,end]
            d[end]=[start,end]
            l2c[end-start+1]+=1
            if l2c[m]>0:res=i
        return res
        
        
        
    def findLatestStep1(self, A: List[int], m: int) -> int:
        res=-1
        x=0
        n=len(A)
        for i in range(1,n+1):
            j=A[i-1]
            k=1<<(n-j)
            x+=k
            s=bin(x)[2:]
            ss=s.split('0')
            if any(len(s2)==m for s2 in ss):res=i
        return res
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        n = len(arr)
        A = [0 for _ in range(n+2)]
        left = [-1 for _ in range(n+2)]
        right = [-1 for _ in range(n+2)]
        count = 0
        res = -1
        
        for i,a in enumerate(arr):
            if A[a-1] == 0 and A[a+1]==0:
                left[a] = a
                right[a] = a
            elif A[a-1] == 0:
                if abs(left[a+1]-right[a+1])+1 == m:
                    count -= 1
                left[a] = a
                right[a] = right[a+1]
                left[right[a]] = a
            elif A[a+1] == 0:
                if abs(left[a-1]-right[a-1])+1==m:
                    count -= 1
                left[a] = left[a-1]
                right[a] = a
                right[left[a]] = a
            else:
                if abs(left[a+1]-right[a+1])+1 == m:
                    count -= 1
                if abs(left[a-1]-right[a-1])+1==m:
                    count -= 1
                left[a] = left[a-1]
                right[a] = right[a+1]
                right[left[a]] = right[a]
                left[right[a]] = left[a]
            
            A[a] = 1    
            if abs(left[a]-right[a])+1 == m:
                count += 1
            if count >= 1:
                res = i+1
                
            # print(left, right, res, count)
        
        return res
                

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        n = len(arr) + 1
        group = [i for i in range(n)]
        bits = [False] * n
        root_to_size = [0] * n
        
        size_count = collections.defaultdict(int)
        size_count[0] = n
        ans = -1
        for step, num in enumerate(arr, start=1):
            g1 = self.find(group, num)
            bits[num] = True
            
            size_count[root_to_size[g1]] -= 1
            root_to_size[g1] += 1
            size_count[root_to_size[g1]] += 1
            
            if num+1 <= len(arr) and bits[num+1]:
                g2 = self.find(group, num+1)
                group[g2] = g1
                combined_size = root_to_size[g1] + root_to_size[g2]
                size_count[root_to_size[g1]] -= 1
                size_count[root_to_size[g2]] -= 1
                root_to_size[g1] = combined_size
                size_count[root_to_size[g1]] += 1
                
            if num-1 >= 1 and bits[num-1]:
                g2 = self.find(group, num-1)
                group[g2] = g1
                combined_size = root_to_size[g1] + root_to_size[g2]
                size_count[root_to_size[g1]] -= 1
                size_count[root_to_size[g2]] -= 1
                root_to_size[g1] = combined_size
                size_count[root_to_size[g1]] += 1
            
            if m in size_count and size_count[m] > 0:
                ans = step
            # print(ans, step, size_count)
        return ans
        
        
    def find(self, group, i):
        while group[i] != i:
            group[i] = group[group[i]]
            i = group[i]
        return i
class Solution:
    def findLatestStep(self, a: List[int], m: int) -> int:
        
        n = len(a)
        a = [i-1 for i in a]
        val = [[i,i] for i in range(n)]
        par = [i for i in range(n)]
        lis = [0]*n
        def find(chi):
            if par[chi] == chi:
                return chi
            temp = find(par[chi])
            par[chi] = temp
            return temp
        
        def union(i,j):
            pari = find(i)
            parj = find(j)
            
            par[parj] = pari
            val[pari][0] = min(val[parj][0], val[pari][0])
            val[pari][1] = max(val[parj][1], val[pari][1])
        
        ans = -1
        cnt = 0
        for i in range(len(a)):
            lis[a[i]] = 1
            if a[i]-1 >= 0 and lis[a[i]-1] == 1:
                tval = val[find(a[i]-1)]
                if tval[1]-tval[0] +1 == m:
                    cnt-=1
                union(a[i]-1, a[i])
            if a[i]+1 < n and lis[a[i]+1] == 1:
                tval = val[find(a[i]+1)]
                if tval[1]-tval[0] +1 == m:
                    cnt-=1
                union(a[i]+1, a[i])
            tval = val[find(a[i])]
            if tval[1]-tval[0] +1 == m:
                cnt+=1
            if cnt >=1 :
                ans = i+1
            # print(par, cnt)
        return ans
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        
        n = len(arr)
        que = collections.deque([(1, n)])
        
        for i in range(n-1, -1, -1):
            
            k = len(que)
            
            for j in range(k):
                l, r = que.popleft()
                if r - l + 1 == m:
                    return i+1
                
                if l <= arr[i] <= r:
                    if arr[i]-l >= m:
                        que.append((l, arr[i]-1))
                    if r-arr[i] >= m:
                        que.append((arr[i]+1, r))
                else:
                    que.append((l, r))
        return -1
            
            
            
        
        
        
        # change endpoint .
        # n = len(arr)
        # if m == n:
        #     return m
        # length = [0] * (n+2)
        # # count = [0] * (n+1)
        # res = -1
        # for i, num in enumerate(arr):
        #     left = length[num-1]
        #     right = length[num + 1]
        #     # I almost came up with this, change the endpoint.
        #     length[num-left] = length[num+right] = left+right+1
        #     if left == m or right == m:
        #         res = i
        # return res
    
    
    
    # if m == len(A): return m
    #     length = [0] * (len(A) + 2)
    #     res = -1
    #     for i, a in enumerate(A):
    #         left, right = length[a - 1], length[a + 1]
    #         if left == m or right == m:
    #             res = i
    #         length[a - left] = length[a + right] = left + right + 1
    #     return res
                
            
            
                
            
            
        # Union-find
#         n = len(arr)
#         p = [i for i in range(n+1)]
#         count = [0] * (n+1)
#         ### I didn't come up with this groups at first. It shouldn't be hard.
#         groups = [0] * (n+1)
#         def findp(x):
#             while x != p[x]:
#                 x = p[x]
#             return x
        
#         def union(x, y):
            
#             groups[count[y]] -= 1
#             groups[count[x]] -= 1
#             if count[x] >= count[y]:
#                 p[y] = x
#                 count[x] += count[y]
#                 groups[count[x]] += 1
#             else:
#                 p[x] = y
#                 count[y] += count[x]
#                 groups[count[y]] += 1
        
#         res = -1
        
#         for i, num in enumerate(arr):
#             # print(p)
#             # print(count)
#             count[num] = 1
#             left = num-1
#             right = num + 1
#             groups[1] += 1
#             if left >= 1 and count[left] != 0:
#                 pl = findp(left)
#                 pm = findp(num)
#                 if pl != pm:
#                     union(pl, pm)
#             if right <= n and count[right] != 0:
#                 pr = findp(right)
#                 pm = findp(num)
#                 if pr != pm:
#                     union(pr, pm)
            
#             if groups[m] > 0:
#                 res = i+1
#         return res
                    
                
                
            
        
                
        
        
        
        
         
        
            
            
            

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        s,lastOcc,countsDict,n=[0]*(len(arr)+1),-1,defaultdict(int),len(arr)
        for traversei,ind in enumerate(arr):
            i=ind-1
            newSize=s[i-1]+s[i+1]+1
            countsDict[s[i+1]]-=1
            countsDict[s[i-1]]-=1
            countsDict[newSize]+=1
            s[i-s[i-1]]=s[i+s[i+1]]=newSize
            if countsDict[m]>0:
                    lastOcc=traversei+1
        return lastOcc
class UnionFind:
    def __init__(self, n):
        self.parent = [-1 for i in range(n)]
        self.size = [0 for _ in range(n)]
        self.size_count = [0 for _ in range(n + 1)]
        self.size_count[0] = n
        
    def init(self, x):
        self.parent[x] = x
        self.size[x] = 1
        self.size_count[1] += 1
        self.size_count[0] -= 1
        
    def find(self, x):
        if self.parent[x] == -1:
            return -1
        if x != self.parent[x]:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        p = self.find(x)
        q = self.find(y)
        if p == q:
            return
        if p == -1 or q == -1:
            return
        father, son = p, q
        self.parent[son] = father
        self.size_count[self.size[son]] -= 1
        self.size_count[self.size[father]] -= 1
        self.size[father] += self.size[son]
        self.size_count[self.size[father]] += 1
    
    def get_size(self, x):
        if self.find(x) == -1:
            return 0
        return self.size[self.find(x)]

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        n = len(arr)
        uf = UnionFind(n + 1)
        step = -1
        for idx, num in enumerate(arr):
            left, right = num - 1, num + 1
            uf.init(num)
            if left >= 1:
                uf.union(left, num)
            if right <= n:
                uf.union(num, right)
            if uf.size_count[m]:
                step = max(step, idx + 1)
        return step
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        dp = [(-1, -1)] * (len(arr) + 1)
        groups = collections.defaultdict(int)
        if len(arr) == 1:
            return 1
        res = -1
        for i, a in enumerate(arr):
            left, right = -1, -1
            if a == 1:
                right = a + 1
            elif a == len(arr):
                left = a - 1
            else:
                left = a - 1
                right = a + 1
            leftPos = a if left == -1 or dp[left][0] == -1 else dp[left][0]
            rightPos = a if right == -1 or dp[right][1] == -1 else dp[right][1]
            
            dp[a] = (leftPos, rightPos)
            if leftPos != a:
                dp[leftPos] = (leftPos, rightPos)
            if rightPos != a:
                dp[rightPos] = (leftPos, rightPos)

            groups[rightPos - leftPos + 1] += 1
            groups[a - leftPos] -= 1
            groups[rightPos - a] -= 1
            if groups[m] >= 1:
                res = i + 1
            #print(a, left, right, leftPos, rightPos, groups, dp)
        return res
class UnionFind:
    def __init__(self, n):
        self.parent = [-1 for i in range(n)]
        self.size = [0 for _ in range(n)]
        self.size_count = [0 for _ in range(n + 1)]
        self.size_count[0] = n
        
    def init(self, x):
        self.parent[x] = x
        self.size[x] = 1
        self.size_count[1] += 1
        self.size_count[0] -= 1
        
    def find(self, x):
        if self.parent[x] == -1:
            return -1
        if x != self.parent[x]:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        p = self.find(x)
        q = self.find(y)
        if p == q:
            return
        if p == -1 or q == -1:
            return
        father, son = p, q
        self.parent[son] = father
        self.size_count[self.size[son]] -= 1
        self.size_count[self.size[father]] -= 1
        self.size[father] += self.size[son]
        self.size_count[self.size[father]] += 1
    
    def get_size(self, x):
        if self.find(x) == -1:
            return 0
        return self.size[self.find(x)]

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        n = len(arr)
        uf = UnionFind(n + 1)
        res = -1
        for step, num in enumerate(arr):
            left, right = num - 1, num + 1
            uf.init(num)
            if left >= 1:
                uf.union(left, num)
            if right <= n:
                uf.union(num, right)
            if uf.size_count[m]:
                res = max(res, step + 1)
        return res
from collections import defaultdict

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        ans = -1
        forLib = defaultdict(int)
        backLib = defaultdict(int)
        memo = {0: len(arr)}
        for i in range(len(arr)):
            index = arr[i]
            val = backLib[index - 1] + 1 + forLib[index + 1]
            if val not in memo:
                memo[val] = 0
            memo[val] += 1
            memo[backLib[index - 1]] -= 1
            memo[forLib[index + 1]] -= 1
            if val == m:
                ans = i + 1
            if (backLib[index - 1] == m or forLib[index + 1] == m) and memo[m] == 0:
                ans = i  
            forLib[index - backLib[index - 1]] = val
            backLib[index + forLib[index + 1]] = val
        return ans

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        res = -1
        mem = {}
        count = collections.Counter()
        group_size = {}
        def find(k):
            if mem[k] != k:
                mem[k] = find(mem[k])
            return mem[k]

        def union(n1, n2):
            f1, f2 = find(n1), find(n2)
            if f1 != f2:
                count[group_size[f1]] -= 1
                count[group_size[f2]] -= 1
                group_size[f1] += group_size[f2]
                count[group_size[f1]] += 1
                mem[f2] = f1

        for idx, v in enumerate(arr, 1):
            mem[v] = v
            group_size[v] = 1
            count[1] += 1
            left = v - 1 if v - 1 in mem else v
            right = v + 1 if v + 1 in mem else v
            union(left, v)
            union(v, right)
            if count[m] > 0:
                res = idx
        return res
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        n=len(arr)
        d={i:[i-1,i+1] for i in range(0,n+2)}
        latest = -1
        for i,j in enumerate(arr):
            a,b = d.pop(j)
            if d[a][1]-a==m+1 or a-d[a][0]==m+1 or d[b][1]-b==m+1 or b-d[b][0]==m+1:
                latest=i
            if b-a==m+1:
                latest=i+1
            if a>=0:
                d[a][1]=b
            if b<=n+1:
                d[b][0]=a            
        return latest

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        L = len(arr)
        parent = [0] * (L+1)
        size = [1] * (L+1)

        def find(x):
            if parent[x] == 0:
                return x
            parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            if find(x) == find(y):
                return
            px = find(x)
            py = find(y)
            if size[px] < size[py]:
                px, py = py, px
                
            if size[px] == m:
                good.discard(px)
            if size[py] == m:
                good.discard(py)
                
            parent[py] = px
            size[px] += size[py]

        bs = [0] * (L+1)
        ret = -1
        step = 0
        good = set()
        
        for a in arr:
            step += 1
            bs[a] = 1
            if a-1>=0 and bs[a-1] == 1:
                union(a, a-1)
            if a+1 <= L and bs[a+1] == 1:
                union(a, a+1)
            
            if size[find(a)] == m:
                good.add(find(a))
            # print(step, good)
            if len(good) > 0:
                ret = step
                
        return ret
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        n = len(arr)
        if n == m:
            return n
        arr.insert(0, 0)
        day = [0] * (n + 1)
        for i in range(1, n+1):
            day[arr[i]] = i
        ans = -1
        max_q = MaxQueue(m)
        for i in range(1, n+1):
            max_q.pop_expired(i)
            max_q.push(day[i], i)
            if i < m:
                continue
            left = right = math.inf
            if i - m >= 1:
                left = day[i-m]
            if i + 1 <= n:
                right = day[i+1]
            if max_q.max() < (d := min(left, right)):
                ans = max(ans, d - 1)
        return ans

class MaxQueue:
    def __init__(self, size):
        self.queue = deque()
        self.size = size

    def push(self, x, pos):
        while self.queue and self.queue[-1][0] < x:
            self.queue.pop()
        self.queue.append([x, pos])

    def pop_expired(self, pos):
        if self.queue and pos - self.queue[0][1] >= self.size:
            self.queue.popleft()

    def max(self):
        return self.queue[0][0]


import collections
class Node:
    def __init__(self, val):
        self.size = 1
        self.val = val
        self.parent = self
        
class UnionFind:
    def __init__(self):
        self.map = {}
        self.sizes = collections.defaultdict(int)
        
    def find(self, node):
        if node.val not in self.map:
            self.map[node.val] = node
            self.sizes[node.size] += 1
        elif node.parent != node:
            node = self.find(node.parent)
        return node
    
    
    def merge(self, node1, node2):
        parent1, parent2 = self.find(node1), self.find(node2)
        if parent1 != parent2:
            if parent1.size >= parent2.size:
                self.sizes[parent1.size] -= 1
                self.sizes[parent2.size] -= 1
                parent2.parent = parent1
                parent1.size += parent2.size
                self.sizes[parent1.size] += 1
            else:
                self.sizes[parent1.size] -= 1
                self.sizes[parent2.size] -= 1
                parent1.parent = parent2
                parent2.size += parent1.size
                self.sizes[parent2.size] += 1

class Solution:
    def findLatestStep(self, arr, m: int) -> int:
        uf = UnionFind()
        ans = -1
        for i, val in enumerate(arr):
            node = Node(val)
            uf.find(node)
            if val - 1 in uf.map:
                uf.merge(node, uf.map[val - 1])
            if val + 1 in uf.map:
                uf.merge(node, uf.map[val + 1])
                
            if uf.sizes[m] > 0:
                ans = i + 1
        return ans
class UnionFindSet:
    def __init__(self, n):
        self.parents = list(range(n))
        self.ranks = [0] * n
        
    def find(self, u):
        # u3082u3057uu304cu81eau5206u306eu89aau3068u9055u3048u3070uu306bu81eau5206u306eu89aau306eu89aau3092u89aau3068u3059u308b
        # u3069u3053u304bu306bu65e2u306bu6240u5c5eu3057u3066u308cu3070u3053u3053u304cTrueu3064u307eu308aparentu3068u81eau5206u81eau8eabu304cu9055u3046u3053u3068u306bu306au308b
        if u != self.parents[u]:
            self.parents[u] = self.find(self.parents[u])
            
        return self.parents[u]
    def union(self, u, v):
        pu, pv = self.find(u), self.find(v)
        
        if pu == pv:
            return False
        
        if self.ranks[pu] > self.ranks[pv]:
            self.parents[pv] = pu
            self.ranks[pu] += self.ranks[pv]
        else:
            self.parents[pu] = pv
            self.ranks[pv] += self.ranks[pu]
        return True
    
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        n, ans = len(arr), -1
        uf = UnionFindSet(n)

        for step, i in enumerate(arr):
            i -= 1
            uf.ranks[i] = 1

            for j in (i - 1, i + 1):
                if 0 <= j < n:
                    if uf.ranks[uf.find(j)] == m:
                        ans = step
                    if uf.ranks[j]:
                        uf.union(i, j)

        for i in range(n):
            if uf.ranks[uf.find(i)] == m:
                return n

        return ans
# class UnionFindSet:
#     def __init__(self, n):
#         self.parents = list(range(n))
#         self.ranks = [0] * n
        
#     def find(self, u):
#         if u != self.parents[u]:
#             self.parents[u] = self.find(self.parents[u])
#         return self.parents[u]
    
#     def union(self, u, v):
#         pu, pv = self.find(u), self.find(v)
#         if pu == pv:
#             return False
#         if self.ranks[pu] > self.ranks[pv]:
#             self.parents[pv] = pu
#             self.ranks[pu] += self.ranks[pv]
#         else:
#             self.parents[pu] = pv
#             self.ranks[pv] += self.ranks[pu]
#         return True

# class Solution:
#     def findLatestStep(self, arr: List[int], m: int) -> int:
#         n, ans = len(arr), -1
#         uf = UnionFindSet(n)
        
#         for step, i in enumerate(arr):
#             i -= 1
#             uf.ranks[i] = 1
#             for j in (i - 1, i + 1):
#                 if 0 <= j < n:
#                     if uf.ranks[uf.find(j)] == m:
#                         ans = step
#                     if uf.ranks[j]:
#                         uf.union(i, j)
        
#         for i in range(n):
#             if uf.ranks[uf.find(i)] == m:
#                 return n
            
#         return ans

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:

        if len(arr) == m:
            return m
        
        move = -1
        l = {x : (0,0) for x in range(len(arr)+2)}
        
        for i, a in enumerate(arr):
    
            l[a] = (a,a)
            b,c = a,a
            #print(i,a,l)
        
            # Check Left
            if l[a-1][0]:
                
                #Check Prev Length
                if l[a-1][1] - l[a-1][0] + 1 == m:
                    move = i
                
                # Left Boarder 
                b = l[a-1][0]
                
            # Check Right
            if l[a+1][0]:
                #Check Prev Length
                if l[a+1][1] - l[a+1][0] + 1 == m:
                    move = i
                
                # Right Boarder
                c = l[a+1][1]
                
                            
            # Update   
            l[a] = (b,c)
            l[b] = (b,c)
            l[c] = (b,c)
                   
            # Check Current Length 
            if l[a][1] - l[a][0] + 1 == m:
                move = i+1
           

        return move

        
        
     

class UnionFindSet:
    def __init__(self, n):
        self.parents = list(range(n))
        self.ranks = [0] * n
        
    def find(self, u):
        if u != self.parents[u]:
            self.parents[u] = self.find(self.parents[u])
        return self.parents[u]
    
    def union(self, u, v):
        pu, pv = self.find(u), self.find(v)
        if pu == pv:
            return False
        if self.ranks[pu] > self.ranks[pv]:
            self.parents[pv] = pu
            self.ranks[pu] += self.ranks[pv]
        else:
            self.parents[pu] = pv
            self.ranks[pv] += self.ranks[pu]
        return True

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        n, ans = len(arr), -1
        uf = UnionFindSet(n)
        
        for step, i in enumerate(arr):
            i -= 1
            uf.ranks[i] = 1
            for j in (i - 1, i + 1):
                if 0 <= j < n:
                    if uf.ranks[uf.find(j)] == m:
                        ans = step
                    if uf.ranks[j]:
                        uf.union(i, j)
        
        for i in range(n):
            if uf.ranks[uf.find(i)] == m:
                return n
            
        return ans
class Subsets:
    def __init__(self,parent,rank):
        self.parent=parent
        self.rank=rank

def find(subsets,node):
    if subsets[node].parent!=node:
        subsets[node].parent=find(subsets,subsets[node].parent)
    return subsets[node].parent

def union(subsets,x,y):
    
    xr=find(subsets,x)
    yr=find(subsets,y)
    
    if xr==yr:
        return False
    else:
        
        xr=subsets[xr]
        yr=subsets[yr]
        
        r1=xr.rank
        r2=yr.rank
        
        if r1<r2:
            xr.parent=yr.parent
            yr.rank+=xr.rank
        elif r2<r1:
            yr.parent=xr.parent
            xr.rank+=yr.rank
        else:
            xr.parent=yr.parent
            yr.rank+=xr.rank
            
        return True
            
        
class Solution:
    
    def findLatestStep(self, arr: List[int], m: int) -> int:
        
        subsets=[Subsets(i,1) for i in range(len(arr))]
        lst=[0]*len(arr)
        parents = set()
        ans=-1
        time=1
        
        for index in arr:
            
            index-=1
            lst[index]=1
            
            if index+1<len(arr) and lst[index+1]==1:
                p=find(subsets,index+1)
                if p in parents:
                    parents.remove(p)
                union(subsets,index+1,index)
                
                    
            if index-1>=0 and lst[index-1]==1:
                p=find(subsets,index-1)
                if p in parents:
                    parents.remove(p)
                union(subsets,index-1,index)
                
                        
            if subsets[find(subsets,index)].rank==m:
                parents.add(find(subsets,index))
                
            if len(parents):
                ans=time
                
            #print(parents)
            
            time+=1
        
        return ans
            
                
                
            
            
            
            
            
            
            
            
            

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        
        
            
        
        n = len(arr)
        p = [i for i in range(n+1)]
        count = [0] * (n+1)
        groups = [0] * (n+1)
        def findp(x):
            while x != p[x]:
                x = p[x]
            return x
        
        def union(x, y):
            
            groups[count[y]] -= 1
            groups[count[x]] -= 1
            if count[x] >= count[y]:
                p[y] = x
                count[x] += count[y]
                groups[count[x]] += 1
            else:
                p[x] = y
                count[y] += count[x]
                groups[count[y]] += 1
        
        res = -1
        
        for i, num in enumerate(arr):
            # print(p)
            # print(count)
            count[num] = 1
            left = num-1
            right = num + 1
            groups[1] += 1
            if left >= 1 and count[left] != 0:
                pl = findp(left)
                pm = findp(num)
                if pl != pm:
                    union(pl, pm)
            if right <= n and count[right] != 0:
                pr = findp(right)
                pm = findp(num)
                if pr != pm:
                    union(pr, pm)
            
            if groups[m] > 0:
                res = i+1
        return res
                    
                
                
            
        
                
        
        
        
        
         
        
            
            
            

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        n=len(arr)
        count=[0]*(n+2)
        lens=[0]*(n+2)
        res=-1
        for i,a in enumerate(arr):
            if lens[a]:
                continue
            l=lens[a-1]
            r=lens[a+1]
            t=l+r+1
            lens[a-l]=lens[a+r]=lens[a]=t
            count[l]-=1
            count[r]-=1
            count[t]+=1
            if count[m]:
                res=i+1
        return res

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        arr = [a - 1 for a in arr]
        N = len(arr)
        state = [False for _ in range(N)]
        heads = [i for i in range(N)]
        children = {i: [i] for i in range(N)}
        counts = {0: N}

        last = -1
        for it, x in enumerate(arr):
            state[x] = True
            # print(state, heads, children, counts)
            neighbors = [i for i in [x + 1, x - 1] if 0 <= i < N and state[i]]
            counts[0] -= 1
            if not neighbors:
                counts[1] = counts.get(1, 0) + 1
                if counts.get(m, 0) > 0:
                    last = it
                continue
            neighbors.sort(key=lambda x: len(children[heads[x]]))
            h = heads[neighbors[0]]
            heads[x] = h
            counts[len(children[h])] -= 1
            children[h].append(x)
            if len(neighbors) == 2:
                h2 = heads[neighbors[1]]
                for y in children[h2]:
                    heads[y] = h
                    children[h].append(y)
                counts[len(children[h2])] -= 1
            counts[len(children[h])] = counts.get(len(children[h]), 0) + 1
            if counts.get(m, 0) > 0:
                last = it
        if last == -1: return -1
        return last + 1
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        answer = -1
        n = len(arr)
        parent = {}
        size = {}
        sizes = 0
        def root(x):
            return x if parent[x] == x else root(parent[x])
        def merge(x, y):
            nonlocal sizes
            x = root(x)
            y = root(y)
            if size[y] == m: sizes -= 1
            if size[x] < size[y]: x, y = y, x
            parent[y] = x
            size[x] += size[y]
            del size[y]
        for t, x in enumerate(arr):
            parent[x] = x
            size[x] = 1
            if x+1 in parent: merge(x, x+1)
            if x-1 in parent: merge(x, x-1)
            if size[root(x)] == m: sizes += 1
            if sizes: answer = t+1
        return answer
class UnionFindSet:
    def __init__(self, n):
        self.parents = list(range(n))
        self.ranks = [0] * n
        
    def find(self, u):
        if u != self.parents[u]:
            self.parents[u] = self.find(self.parents[u])
        return self.parents[u]
    
    def union(self, u, v):
        pu, pv = self.find(u), self.find(v)
        if pu == pv:
            return False
        if self.ranks[pu] > self.ranks[pv]:
            self.parents[pv] = pu
            self.ranks[pu] += self.ranks[pv]
        else:
            self.parents[pu] = pv
            self.ranks[pv] += self.ranks[pu]
        return True

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        if m == len(arr):
            return m
        
        n, ans = len(arr), -1
        uf = UnionFindSet(n)
        
        for step, i in enumerate(arr):
            i -= 1
            uf.ranks[i] = 1
            for j in (i - 1, i + 1):
                if 0 <= j < n:
                    if uf.ranks[uf.find(j)] == m:
                        ans = step
                    if uf.ranks[j]:
                        uf.union(i, j)
        
        return ans

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        bit = [None] * (len(arr)+1)
        res = -1
        step = 1
        group = 0
        for i in range(len(arr)):
            #print(step, arr[i], group)
            temp = 1
            right = 0
            left = 0
            #print(bit)
            if arr[i] - 1 > 0 and bit[arr[i]-1] != None:
                if bit[arr[i]-1] == 'True':
                    bit[arr[i]-m] = False
                    bit[arr[i]-1] = False
                    bit[arr[i]] = False
                    group -= 1
                elif bit[arr[i]-1] == False:
                    bit[arr[i]] = False
                else:
                    right += bit[arr[i]-1]
                
            if arr[i] + 1 <= len(arr) and bit[arr[i]+1] != None:
                
                    
                if bit[arr[i]+1] == 'True':
                    bit[arr[i]+m] = False
                    bit[arr[i]+1] = False
                    bit[arr[i]] = False
                    group -= 1
                elif bit[arr[i]] == False:
                    if bit[arr[i]+1]:
                        bit[arr[i]+bit[arr[i]+1]] = False
                    bit[arr[i]+1] = False
                elif bit[arr[i]+1] == False:
                    bit[arr[i]] = False
                else:
                    left += bit[arr[i]+1]
            if bit[arr[i]] == None:
                #print(arr[i],right , left)
                temp += right + left
                bit[arr[i]] = temp
                if right:
                    bit[arr[i]-right] += left + 1
                if left:
                    bit[arr[i]+left] += right + 1
                if temp == m:
                    bit[arr[i]-right] = 'True'
                    bit[arr[i]+left] = 'True'
                    group += 1
            #print(bit)
            if group > 0:
                res = step
            step += 1
        return res
class UnionFind:
    def __init__(self):
        self.parent = dict()
        self.rank = dict()
        self.size = dict()
        self.sizes = collections.defaultdict(int)
        
        return
    
    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
            self.size[x] = 1
            self.sizes[1] += 1
        
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        
        while self.parent[x] != root:
            parent = self.parent[x]
            self.parent[x] = root
            x = parent
        
        return root
    
    def union(self, x, y):
        x_root = self.find(x)
        y_root = self.find(y)
        
        if x_root == y_root:
            return
        
        if self.size[x_root] < self.size[y_root]:
            x_root, y_root = y_root, x_root
            
        y_size = self.size[y_root]
        x_size = self.size[x_root]
        
        self.sizes[x_size] -= 1
        self.sizes[y_size] -= 1
        
        
        self.parent[y_root] = x_root        
        self.size[x_root] += self.size[y_root]
        
        self.sizes[self.size[x_root]] += 1
        
        return

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        n = len(arr)
        
        binary_string = [0] * n
        
        uf = UnionFind()
        
        max_step = -1
        step = 1
        
        for i in arr:
            index = i - 1
            binary_string[index] = 1
            
            root = uf.find(i)
            
            if index - 1 >= 0 and binary_string[index - 1] == 1:
                uf.union(root, i - 1)
                
            if index + 1 < len(binary_string) and binary_string[index + 1] == 1:
                uf.union(root, i + 1)  
            
            if uf.sizes[m] > 0:
                max_step = step
                
            step += 1
        return max_step
        

class DisjointUnionSets:
    def __init__(self,n):
        self.rank = [0] * n 
        self.parent = [0] * n
        self.n = n
        self.makeSet()

    def makeSet(self):
        for i in range(self.n):
            self.parent[i] = i
        
    def find(self ,x): 
        if (self.parent[x] != x):
            self.parent[x] = self.find(self.parent[x]); 
        return self.parent[x]; 
    
    def union(self, x, y): 
        xRoot = self.find(x)
        yRoot = self.find(y)
   
        if (xRoot == yRoot): 
            return; 
  
        if (self.rank[xRoot] < self.rank[yRoot]):
            self.parent[xRoot] = yRoot
            self.rank[yRoot]+=self.rank[xRoot]
        else: 
            self.parent[yRoot] = xRoot; 
            self.rank[xRoot]+=self.rank[yRoot]



class Solution:
    def findLatestStep(self, arr, m: int) -> int:
        n, ans = len(arr), -1
        uf = DisjointUnionSets(n)
        
        for step, i in enumerate(arr):
            i-=1
            uf.rank[i] = 1
            for j in (i - 1, i + 1):
                if 0 <= j < n:
                    if uf.rank[uf.find(j)] == m:
                        ans = step
                    if uf.rank[j]:
                        uf.union(i, j)
        
        for i in range(n):
            if uf.rank[uf.find(i)] == m:
                return n
        return ans

class UnionSet:
    def __init__(self,n):
        self.parent = list(range(n))
        self.rank = [0] * n
        
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self,x,y):
        xParent = self.find(x)
        yParent = self.find(y)
        if xParent == yParent:
            return
        
        if self.rank[xParent] > self.rank[yParent]:
            self.parent[yParent] = xParent
            self.rank[xParent]+=self.rank[yParent]
        else:
            self.parent[xParent] = yParent
            self.rank[yParent]+=self.rank[xParent]


class Solution:
    def findLatestStep(self, arr, m: int) -> int:
        us = UnionSet(len(arr))
        ans = -1
        for step,idx in enumerate(arr):
            idx-=1
            us.rank[idx] = 1      
            for j in (-1, 1):
                neighbour = idx+j
                if 0<=neighbour<len(arr):
                    if us.rank[us.find(neighbour)] == m:
                        ans = step
                    if us.rank[neighbour]:
                        us.union(neighbour,idx)
        
        for i in range(len(arr)):
            if us.rank[us.find(i)] == m:
                return len(arr)
        return ans
class UnionFindSet:
    def __init__(self, n):
        self.parents = list(range(n))
        self.ranks = [0] * n
        
    def find(self, u):
        if u != self.parents[u]:
            self.parents[u] = self.find(self.parents[u])
        return self.parents[u]
    
    def union(self, u, v):
        pu, pv = self.find(u), self.find(v)
        if pu == pv:
            return False
        if self.ranks[pu] > self.ranks[pv]:
            self.parents[pv] = pu
            self.ranks[pu] += self.ranks[pv]
        elif self.ranks[pv] > self.ranks[pu]:
            self.parents[pu] = pv
            self.ranks[pv] += self.ranks[pu]
        else:
            self.parents[pu] = pv
            self.ranks[pv] += self.ranks[pu]
        return True

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        n, ans = len(arr), -1
        uf = UnionFindSet(n)
        
        for step, i in enumerate(arr):
            i -= 1
            uf.ranks[i] = 1
            for j in (i - 1, i + 1):
                if 0 <= j < n:
                    if uf.ranks[uf.find(j)] == m:
                        ans = step
                    if uf.ranks[j]:
                        uf.union(i, j)
        
        for i in range(n):
            if uf.ranks[uf.find(i)] == m:
                return n
            
        return ans
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        n=len(arr)
        index=[0]*(n+2)
        length=[0]*(n+1)
        ans=-1
        for i in range(n):
            x=arr[i]
            l_l=index[x-1]
            r_l=index[x+1]
            new_l=1+l_l+r_l
            
            index[x]=new_l
            index[x-l_l]=new_l
            index[x+r_l]=new_l
            
            
            
            if length[l_l]:
                length[l_l]-=1
            if length[r_l]:
                length[r_l]-=1
            length[new_l]+=1
            
            if length[m]>0:
                ans=i+1
                
        return ans
            
            
            
            
            
            
                
            
            
            
            
                    
        
        
        
            
            
            
            
            
            
        
        
        

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        dsu = DSU()
        used = set()
        candidates = set()
        ans = -1
        for index, i in enumerate(arr):
            if i + 1 in used:
                dsu.union(i, i + 1)
            if i - 1 in used:
                dsu.union(i, i - 1)
            used.add(i)
                        
            if dsu.get_count(i) == m:
                candidates.add(i)
            
            cur_candidates = set()
            for c in candidates:
                if dsu.get_count(c) == m:
                    cur_candidates.add(c)
                    ans = max(ans, index + 1)
            candidates = cur_candidates
            
        return ans
        
class DSU:
    def __init__(self):
        self.father = {}
        self.count = defaultdict(lambda x: 1)
    
    def find(self, a):
        self.father.setdefault(a, a)
        self.count.setdefault(a, 1)
        if a != self.father[a]:
            self.father[a] = self.find(self.father[a])
        return self.father[a]
    
    def union(self, a, b):
        _a = self.find(a)
        _b = self.find(b)
        if _a != _b:
            self.father[_a] = self.father[_b]
            self.count[_b] += self.count[_a]
    
    def get_count(self, a):
        return self.count[self.find(a)]
class Solution:
    def __init__(self):
        self.ans = -1
    
    def findLatestStep(self, arr: List[int], m: int) -> int:
            n = len(arr)
            if n == m: return n
            
            sz = [1]*n
            group = [i for i in range(n)]
            
            def find(i):
                root = i
                while group[root] != root:
                    root = group[root]
                while i != root:
                    nxt = group[i]
                    group[i] = root
                    i = nxt
                return root
            
            def union(i, j):
                root1 = find(i)
                root2 = find(j)
                if(sz[root1] > sz[root2]):
                    sz[root1] += sz[root2]
                    group[root2] = root1
                else:
                    sz[root2] += sz[root1]
                    group[root1] = root2
            
            nums = [0]*n
            cnt = 0
            for i in range(n):
                nums[arr[i]-1] = 1
                if arr[i]-2 >= 0 and nums[arr[i]-2] == 1:
                    if sz[find(arr[i]-2)] == m:
                        cnt -= 1
                    union(arr[i]-1, arr[i]-2)
                if arr[i] < n and nums[arr[i]] == 1:
                    if sz[find(arr[i])] == m:
                        cnt -= 1
                    union(arr[i]-1, arr[i])
                if sz[find(arr[i]-1)] == m:
                    cnt += 1
                if cnt:
                    self.ans = i+1
            return self.ans
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        class UF:
            def __init__(self):
                self.size = 1
                self.parent = self

            def find(self):
                if self.parent is not self:
                    self.parent = self.parent.find()
                return self.parent

            def union(self, other):
                p1, p2 = self.find(), other.find()
                if p1 is p2:
                    return

                if p2.size < p1.size:
                    p1, p2 = p2, p1

                p2.size += p1.size
                p1.parent = p2

        groups = {}
        sizes = collections.Counter()

        res = -1
        for i, n in enumerate(arr):
            n -= 1
            groups[n] = g = UF()
            sizes[1] += 1

            if n - 1 in groups:
                sizes[groups[n-1].find().size] -= 1
                sizes[g.find().size] -= 1
                groups[n - 1].union(g)
                sizes[g.find().size] += 1

            if n + 1 in groups:
                sizes[groups[n+1].find().size] -= 1
                sizes[g.find().size] -= 1
                groups[n + 1].union(g)
                sizes[g.find().size] += 1

            if sizes[m] > 0:
                res = i + 1

        return res
# class Solution(object):
#     def findLatestStep(self, arr, m):
#         """
#         :type arr: List[int]
#         :type m: int
#         :rtype: int
#         """
#         self.rank = collections.defaultdict(int)
#         self.p = collections.defaultdict(int)
        
#         def find(i):
#             if i not in self.p:
#                 self.p[i] = i
#                 self.rank[i] = 1
#                 return i
#             p = self.p[i]
#             while p != self.p[p]:
#                 p = self.p[p]
#             self.p[i] = p
#             return p
        
#         def union(i, j):
#             ip, jp = find(i), find(j)
#             if ip == jp:
#                 return False
#             ir, jr = self.rank[ip], self.rank[jp]
#             if ir > jr:
#                 self.p[jp] = ip
#                 self.rank[ip] += self.rank[jp]
#                 self.rank.pop(jp)
#             else:
#                 self.p[ip] = jp
#                 self.rank[jp] += self.rank[ip]
#             return True
        
        
#         status = [0] * len(arr)
#         res = -1
#         l = len(arr)
#         for step, i in enumerate(arr):
#             i -= 1
#             status[i] = 1
#             self.p[i] = i
#             self.rank[i] = 1
#             for j in [i-1, i+1]:
#                 if 0<= j < l:
#                     if self.rank[find(j)] == m:
#                         res = step
#                         print(self.p)
#                         print(self.rank)
#                     if status[j] == 1:
#                         union(i, j)
                        
#         for i in range(l):
#             if self.rank[find(i)] == m:
#                 return l
            
#         return res

class UnionFindSet:
    def __init__(self, n):
        self.parents = list(range(n))
        self.ranks = [0] * n
        
    def find(self, u):
        if u != self.parents[u]:
            self.parents[u] = self.find(self.parents[u])
        return self.parents[u]
    
    def union(self, u, v):
        pu, pv = self.find(u), self.find(v)
        if pu == pv:
            return False
        if self.ranks[pu] > self.ranks[pv]:
            self.parents[pv] = pu
            self.ranks[pu] += self.ranks[pv]
        elif self.ranks[pv] > self.ranks[pu]:
            self.parents[pu] = pv
            self.ranks[pv] += self.ranks[pu]
        else:
            self.parents[pu] = pv
            self.ranks[pv] += self.ranks[pu]
        return True

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        n, ans = len(arr), -1
        uf = UnionFindSet(n)
        
        for step, i in enumerate(arr):
            i -= 1
            uf.ranks[i] = 1
            for j in (i - 1, i + 1):
                if 0 <= j < n:
                    if uf.ranks[uf.find(j)] == m:
                        ans = step
                    if uf.ranks[j]:
                        uf.union(i, j)
        
        for i in range(n):
            if uf.ranks[uf.find(i)] == m:
                return n
            
        return ans
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        if m == len(arr):
            return m
        
        dsu = DSU()
        used = set()
        ans = -1
        for index, i in enumerate(arr):
            if i + 1 in used and dsu.get_count(i + 1) == m or i - 1 in used and dsu.get_count(i - 1) == m:
                ans = index
            if i + 1 in used:
                dsu.union(i, i + 1)
            if i - 1 in used:
                dsu.union(i, i - 1)
            used.add(i)
            
        return ans
        
class DSU:
    def __init__(self):
        self.father = {}
        self.count = {}
    
    def find(self, a):
        self.father.setdefault(a, a)
        self.count.setdefault(a, 1)
        if a != self.father[a]:
            self.father[a] = self.find(self.father[a])
        return self.father[a]
    
    def union(self, a, b):
        _a = self.find(a)
        _b = self.find(b)
        if _a != _b:
            self.father[_a] = self.father[_b]
            self.count[_b] += self.count[_a]
    
    def get_count(self, a):
        return self.count[self.find(a)]
class UnionFind():
    def __init__(self, n, m):
        self.parents={}
        self.size={}
        self.rev_map={}
        self.m=m
        for i in range(n):
            self.parents[i]=i
            self.size[i]=0
            self.rev_map[i]=0
        self.rev_map[i+1]=0
        self.rev_map[0]=0
    def union(self, n1, n2):
        
        p1=self.find_parent(n1)
        p2=self.find_parent(n2)
        self.parents[p1]=p2
        self.rev_map[self.size[p1]]-=1
        self.rev_map[self.size[p2]]-=1
        self.size[p2]=self.size[p1]+self.size[p2]
        self.rev_map[self.size[p2]]+=1
        
        
    def find_parent(self, n1):
        if self.parents[n1]==n1:
            return n1
        self.parents[n1]=self.find_parent(self.parents[n1])
        return self.parents[n1]
        
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        u=UnionFind(len(arr),m)
        d=[0]*max(arr)
        best=-1
        # arr=[i-1 for i in arr]
        for i in range(len(arr)):
            
            d[arr[i]-1]=1
            u.size[u.find_parent(arr[i]-1)]=1

            u.rev_map[u.size[u.find_parent(arr[i]-1)]]+=1
            # u.rev_map[u.size[u.find_parent(arr[i])]]+=1
            if arr[i]-2>=0 and d[arr[i]-2]==1:
                u.union(arr[i]-2,arr[i]-1)
            if arr[i]<len(arr) and d[arr[i]]==1:
                u.union(arr[i]-1,arr[i])
            if u.rev_map[m]>=1:
                best=i
            # print(u.rev_map)
        if best==-1:
              return -1
        return best+1
import bisect
import math

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        # print("-----"*5)
        arr = [i-1 for i in arr]
        splitted = [(0,len(arr)-1),] # indices "1"s started with
        step = len(arr)
        if len(arr) == m: return step
        for n in reversed(arr):
            step -= 1
            i = bisect.bisect_right(splitted, (n,math.inf))
            range_ = splitted[i-1]
            left = (range_[0], n-1)
            right = (n+1, range_[1])
            if left[1]-left[0]+1 == m or right[1]-right[0]+1 == m:
                return step
            replace = []
            if left[1] >= left[0]:
                replace.append(left)
            if right[1] >= right[0]:
                replace.append(right)
            
            splitted[i-1:i] = replace
            # print(splitted)
        return -1
import sys
input = sys.stdin.readline

class Unionfind:
    def __init__(self, n):
        self.par = [-1]*n
        self.rank = [1]*n
    
    def root(self, x):
        r = x
        
        while not self.par[r]<0:
            r = self.par[r]
        
        t = x
        
        while t!=r:
            tmp = t
            t = self.par[t]
            self.par[tmp] = r
        
        return r
    
    def unite(self, x, y):
        rx = self.root(x)
        ry = self.root(y)
        
        if rx==ry:
            return
        
        if self.rank[rx]<=self.rank[ry]:
            self.par[ry] += self.par[rx]
            self.par[rx] = ry
            
            if self.rank[rx]==self.rank[ry]:
                self.rank[ry] += 1
        else:
            self.par[rx] += self.par[ry]
            self.par[ry] = rx
    
    def is_same(self, x, y):
        return self.root(x)==self.root(y)
    
    def count(self, x):
        return -self.par[self.root(x)]

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        arr = list(map(lambda x: x-1, arr))
        n = len(arr)
        now = [0]*n
        ans = -1
        uf = Unionfind(n)
        cnt = [0]*(n+1)
        
        for i in range(n):
            p = arr[i]
            now[p] = 1
            cnt[1] += 1
            
            if p-1>=0 and now[p-1]==1:
                cnt[uf.count(p)] -= 1
                cnt[uf.count(p-1)] -= 1
                uf.unite(p-1, p)
                cnt[uf.count(p)] += 1
            
            if p+1<n and now[p+1]==1:
                cnt[uf.count(p)] -= 1
                cnt[uf.count(p+1)] -= 1
                uf.unite(p, p+1)
                cnt[uf.count(p)] += 1
            
            if cnt[m]>0:
                ans = i+1
        
        return ans

class Solution:
  def findLatestStep(self, arr: List[int], m: int) -> int:
    if len(arr) == m:
      return m

    result = -1
    length = []
    for i in range(len(arr) + 2):
      length.append(0)

    for index, value in enumerate(arr):
      left = length[value - 1]
      right = length[value + 1]
      if left == m or right == m:
        result = index

      length[value-left] = left + right + 1
      length[value + right] = left + right + 1
      pass

    return result
class DSU:
    def __init__(self, count):
        self.parent = [i for i in range(count)]
        self.size = [1 for _ in range(count)]
    
    def find(self, x):
        root = x
        while root != self.parent[root]:
            root = self.parent[root]
        while x != root:
            next_node = self.parent[x]
            self.parent[x] = root
            x = next_node
        return root
    
    def union(self, x, y):
        r1, r2 = self.find(x), self.find(y)
        if r1 == r2:
            return
        if self.size[r1] < self.size[r2]:
            self.size[r2] += self.size[r1]
            self.parent[r1] = r2
        else:
            self.size[r1] += self.size[r2]
            self.parent[r2] = r1
    
    def get_size(self, x):
        return self.size[self.find(x)]
        
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        components = [0 for _ in range(len(arr))]
        size_count = collections.Counter()
        dsu = DSU(len(arr))
        ans = -1
        for i, num in enumerate(arr, 1):
            num -= 1
            components[num] = 1
            for adj in (num - 1, num + 1):
                if 0 <= adj < len(arr) and components[adj]:
                    size_count[dsu.get_size(adj)] -= 1
                    dsu.union(num, adj)
            size_count[dsu.get_size(num)] += 1
            if size_count[m] > 0:
                ans = i # step
        return ans
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        memo = [[1, len(arr)]]
        if m == len(arr):
            return len(arr)
        for j in range(len(arr) - 1,- 1, -1):
            i = arr[j]
            left,right = 0, len(memo) - 1
            while left <= right:
                mid = (left + right) // 2
                if memo[mid][0] <= i:
                    left = mid + 1
                else:
                    right = mid - 1
            a,b = memo[right][0], memo[right][1]
            if i - a == m or a + b - i - 1 == m:
                return j
            flag = True
            if i - a > 0:
                memo[right][1] = i -a
            else:
                memo.pop(right)
                flag = False
            if a + b - i - 1 > 0:
                memo[right + flag: right + flag] = [[i + 1, a + b - i - 1]]
            
        return -1

class DSU:
    
    def __init__(self, N):
        self.parents = list(range(N))
        self.size = [1] * N
    
    def find(self, x):
        if x != self.parents[x]:
            self.parents[x] = self.find(self.parents[x])
        
        return self.parents[x]
    
    def union(self, x, y):
        xr = self.find(x)
        yr = self.find(y)
        
        if xr != yr:
            if self.size[xr] < self.size[yr]:
                xr, yr = yr, xr
            self.parents[yr] = xr
            self.size[xr] += self.size[yr]
            self.size[yr] = self.size[xr]
    
    def sz(self, x):
        return self.size[self.find(x)]

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        count = Counter()
        N = len(arr)
        S = [0] * N
        dsu = DSU(N)
        
        ans = -1
        for i, a in enumerate(arr, 1):
            a -= 1
            S[a] = 1
            
            for b in (a - 1, a + 1):
                if 0 <= b < N and S[b]:
                    count[dsu.sz(b)] -= 1
                    dsu.union(a, b)
            
            count[dsu.sz(a)] += 1
            if count[m] > 0:
                ans = i
        
        return ans
        

class UnionFindSet:
    def __init__(self, n):
        self.parents = list(range(n))
        self.ranks = [0] * n

    def find(self, u):
        if u != self.parents[u]:
            self.parents[u] = self.find(self.parents[u])
        return self.parents[u]

    def union(self, u, v):
        pu, pv = self.find(u), self.find(v)
        if pu == pv:
            return False
        if self.ranks[pu] > self.ranks[pv]:
            self.parents[pv] = pu
            self.ranks[pu] += self.ranks[pv]
        elif self.ranks[pv] > self.ranks[pu]:
            self.parents[pu] = pv
            self.ranks[pv] += self.ranks[pu]
        else:
            self.parents[pu] = pv
            self.ranks[pv] += self.ranks[pu]
        return True

class Solution:

    def findLatestStep(self, arr: List[int], m: int) -> int:
        
        '''
        length_dict = {}
        
        length_edge = [0]*len(arr)
        
        result = -1
        
        for i in range(len(arr)):
            index = arr[i] - 1
            
            left_length = 0
            right_length = 0
            if index>0:
                left_length = length_edge[index - 1]
            if index<len(arr)-1:
                right_length = length_edge[index + 1]
            length_edge[index+right_length] = 1 + left_length + right_length
            length_edge[index-left_length] = 1 + left_length + right_length
            
            if left_length in length_dict:
                length_dict[left_length] -= 1
            if right_length in length_dict:
                length_dict[right_length] -= 1
            if 1 + left_length + right_length not in length_dict:
                length_dict[1 + left_length + right_length] = 0
            length_dict[1 + left_length + right_length] += 1
            
            if m in length_dict and length_dict[m]>0:
                result = i + 1
        return result
        '''
        n, ans = len(arr), -1
        uf = UnionFindSet(n)
        
        groups = {}
        groups[0] = len(arr)
        
        result = -1
        for step, i in enumerate(arr):
            i -= 1
            uf.ranks[i] = 1
            
            for j in (i - 1, i + 1):
                if 0 <= j < n:
                    '''
                    if uf.ranks[uf.find(j)] == m:
                        ans = step
                    '''
                    groups[uf.ranks[uf.find(j)]] -= 1
                    if uf.ranks[j]:
                        uf.union(i, j)
            
            '''
            if uf.ranks[uf.find(i)] == m:
                ans = step + 1
            '''
            group = uf.ranks[uf.find(i)]
            if group not in groups:
                groups[group] = 0
            groups[group] += 1
            if m in groups and groups[m]>0:
                result = step + 1
        '''
        for i in range(n):
            if uf.ranks[uf.find(i)] == m:
                return n
        return ans
        '''
        
        return result
            

class DSU:
    
    def __init__(self, count):
        self.parent = [i for i in range(count)]
        self.size = [1 for _ in range(count)]
    
    def find(self, x):
        root = x
        while root != self.parent[root]:
            root = self.parent[root]
        while x != root:
            next_node = self.parent[x]
            self.parent[x] = root
            x = next_node
        return root
    
    def union(self, x, y):
        r1, r2 = self.find(x), self.find(y)
        if r1 == r2:
            return
        if self.size[r1] < self.size[r2]:
            self.size[r2] += self.size[r1]
            self.parent[r1] = r2
        else:
            self.size[r1] += self.size[r2]
            self.parent[r2] = r1
            
    def get_size(self, x):
        return self.size[self.find(x)]

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        components = [0 for _ in range(len(arr))]
        ans = -1
        dsu = DSU(len(arr))
        size_count = collections.Counter()
        for i, n in enumerate(arr, 1):
            n -= 1
            components[n] = 1
            for adj in (n - 1, n + 1):
                if 0 <= adj < len(arr) and components[adj]:
                    size_count[dsu.get_size(adj)] -= 1
                    dsu.union(n, adj)
            size_count[dsu.get_size(n)] += 1
            if size_count[m] > 0:
                ans = i
        return ans
                

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        
        d = {}
        step = 1
        ans = -1
        s = set()
        
        for a in arr:
            
            if a not in d:
                
                left = d.get(a-1, [])
                right = d.get(a+1, [])
                
                
                s = 1
                
                if left:
                    s += left[0]
                    if left[0] == m:
                        ans = max(ans, step-1)
                
                if right:
                    s += right[0]
                    if right[0] == m:
                        ans = max(ans, step-1)
                
                # print(s, step, left, right)
                
                if s == m:
                    ans = max(ans, step)
                    
                d[a] = [s,step]
                
                if left:
                    d[a- left [0]] = [s,step]
                
                if right:
                    d[a+ right[0]] = [s,step]
                
                # print(step, d)
            step += 1
        
        

        
        
        return ans
class Solution:
    def findLatestStep(self, A: List[int], T: int, last = -1) -> int:
        seen, ok = set(), set()
        A = [i - 1 for i in A]
        N = len(A)
        P = [i for i in range(N)]
        L = [1] * N
        def find(x):
            if x != P[x]:
                P[x] = find(P[x])
            return P[x]
        def union(a, b):
            a = find(a)
            b = find(b)
            P[b] = a
            L[a] += L[b]
            return L[a]
        step = 1
        for i in A:
            seen.add(i)
            if 0 < i and find(P[i - 1]) in ok: ok.remove(find(P[i - 1]))
            if i + 1 < N and find(P[i + 1]) in ok: ok.remove(find(P[i + 1]))
            if i - 1 in seen: L[i] = union(i, i - 1)
            if i + 1 in seen: L[i] = union(i, i + 1)
            if L[i] == T:
                ok.add(i)
            if len(ok):
                last = step
            step += 1
        return last
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        n = len(arr)
        if m==n: return m
        length = [0 for _ in range(n+2)]
        res = -1
        
        for step, pos in enumerate(arr):
            left, right = length[pos-1], length[pos+1]
            if left==m or right==m:
                res = step
            length[pos-left]=length[pos+right] = left+right+1
        return res
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        N = len(arr)
        parent = {}
        size = {}
        ds_sizes = Counter()

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def decrement_size(x):
            ds_sizes[x] -= 1
            if ds_sizes[x] == 0:
                del ds_sizes[x]

        def union(x, y):
            px, py = find(x), find(y)

            if px is py:
                return

            if size[px] < size[py]:
                px, py = py, px

            parent[py] = px
            decrement_size(size[px])
            decrement_size(size[py])
            size[px] += size[py]
            ds_sizes[size[px]] += 1

        def make_set(x):
            if x in parent:
                return
            parent[x] = x
            size[x] = 1
            ds_sizes[1] += 1

        steps = 0
        last_step = -1

        for n in arr:
            make_set(n)

            for neighbor in (n + 1, n - 1):
                if neighbor in parent:
                    union(n, neighbor)

            steps += 1
            if m in ds_sizes:
                last_step = steps

        return last_step
import copy
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        n = len(arr)
        opts = [i for i in range(n+1)] # 0...n+1
        if m > n:
            return -1
        if m == n:
            return n
        left,right = [i for i in range(n+2)],[i for i in range(n+2)]
        def find_l(node):
            if left[node]!=node:
                left[node] = find_l(left[node])
            return left[node]
        def find_r(node):
            if right[node]!=node:
                right[node] = find_r(right[node])
            return right[node]
        ret = -1
        cnt = collections.defaultdict(int)
        for i,ind in enumerate(arr):
            left[ind] = l_par = find_l(ind-1)
            right[ind] = r_par = find_r(ind+1)
            if ind - l_par == 1 and r_par - ind == 1:
                # print('1')
                cnt[1] += 1
            elif ind - l_par != 1 and r_par - ind != 1:
                # print('2')
                l_dis = ind - l_par - 1
                r_dis = r_par - ind - 1
                cnt[l_dis] -= 1
                if cnt[l_dis] == 0:
                    del cnt[l_dis]
                cnt[r_dis] -= 1
                if cnt[r_dis] == 0:
                    del cnt[r_dis]
                # print(l_dis,r_dis,cnt)
                cnt[l_dis+r_dis+1] += 1
            else:
                # print('3')
                dis = 0
                if ind - l_par == 1:
                    dis = r_par - ind
                elif r_par - ind == 1:
                    dis = ind - l_par
                cnt[dis-1] -= 1
                if cnt[dis-1] == 0:
                    del cnt[dis-1]
                cnt[dis] += 1
            if m in cnt:
                ret = i+1
            # print('aaaaaaaaaa',left,right,cnt)
        
        return ret
                    
                
            

class Subset:
    def __init__(self,n):
        self.parent = [i for i in range(n)]
        self.rank = [0] * n
    
    def find(self,i):
        if self.parent[i] != i:
            self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self,u,v):
        if self.rank[u] > self.rank[v]:
            self.parent[v] = self.find(u)
        if self.rank[v] > self.rank[u]:
            self.parent[u] = self.find(v)
        if self.rank[u] == self.rank[v]:
            self.parent[v] = self.find(u)
            self.rank[u] += self.rank[v]
            
class Solution:       
    def findLatestStep(self, arr: List[int], m: int) -> int:
        n = len(arr)
        subset = Subset(n)
        ls = [0] * n
        size = [1] * n
        res = -1
        cnt = 0
        for i in range(n):
            idx = arr[i] - 1
            ls[idx] = 1
            sizeMiddle = 1
            if idx > 0:
                if ls[idx-1] == 1:
                    p = subset.find(idx-1)
                    sizeLeft = size[p]
                    subset.union(min(idx,p),max(idx,p))
                    if sizeLeft == m:
                        cnt -= 1
                    sizeMiddle += sizeLeft
            if idx < n-1:
                if ls[idx+1] == 1:
                    p2 = subset.find(idx+1)
                    sizeRight = size[p2]
                    subset.union(min(idx,p2),max(idx,p2))
                    if sizeRight == m:
                        cnt -= 1
                    sizeMiddle += sizeRight
            finalP = subset.find(idx)
            size[finalP] = sizeMiddle
            if sizeMiddle == m:
                cnt += 1
            if cnt > 0:
                res = max(res,i+1)
        return res
class UnionFind:
    def __init__(self):
        self.sets = {}
        self.size = {}
        self.sizes = collections.defaultdict(int)
    def make_set(self, s):
        self.sets[s] = s
        self.size[s] = 1
        self.sizes[1] += 1
    def find(self, s):
        if self.sets[s] != s:
            self.sets[s] = self.find(self.sets[s])
        return self.sets[s]
    def union(self, s1, s2):
        a, b = self.find(s1), self.find(s2)
        if a == b:
            return
        self.sizes[self.size[a]] -= 1
        self.sizes[self.size[b]] -= 1
        if self.sizes[self.size[a]] == 0:
            self.sizes.pop(self.size[a], None)
        if self.sizes[self.size[b]] == 0:
            self.sizes.pop(self.size[b], None)
        self.sets[a] = b
        self.size[b] += self.size[a]
        self.sizes[self.size[b]] += 1
    def get_size(self, m):
        return m in self.sizes
        
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        seen = set()
        uf = UnionFind()
        res = -1
        for i, v in enumerate(arr, 1):
            uf.make_set(v)
            if v + 1 in seen:
                uf.union(v, v + 1)
            if v - 1 in seen:
                uf.union(v, v - 1)
            seen.add(v)
            if uf.get_size(m):
                res = i
        return res
            
            

class UnionFind:
    def __init__(self, n):
        self.parents = list(range(n))
        self.ranks = [0] * n

    def find(self, src):
        if self.parents[src] == src:
            return src
        self.parents[src] = self.find(self.parents[src])
        return self.parents[src]
    
    def union(self, src, dest):
        rootSrc, rootDest = self.find(src), self.find(dest)
        if rootDest == rootSrc:
            return False
        
        if self.ranks[rootSrc] > self.ranks[rootDest]:
            self.parents[rootDest] = rootSrc
            self.ranks[rootSrc] += self.ranks[rootDest]
        else:
            self.parents[rootSrc] = rootDest
            self.ranks[rootDest] += self.ranks[rootSrc]
        return True
    
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        n, result = len(arr), -1
        uf = UnionFind(n)

        for step, idx in enumerate(arr):
            idx -= 1
            uf.ranks[idx] = 1
            for j in (idx - 1, idx + 1):
                if 0 <= j < n:
                    if uf.ranks[uf.find(j)] == m:
                        result = step
                    if uf.ranks[j]:
                        uf.union(idx, j)

        for i in range(n):
            if uf.ranks[uf.find(i)] == m:
                return n
        return result
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        def root(x):
            if x == group[x]:
                return x
            group[x] = root(group[x])
            return group[x]
            
        def same(x, y):
            return root(x) == root(y)
            
        def unite(x, y):
            nonlocal cnt
            x = root(x)
            y = root(y)
            cnt -= (sz[x] == m)
            #print(x, y, sz[x], sz[y])
            if sz[x] < sz[y]:
                x, y = y, x
            group[y] = x
            sz[x] += sz[y]
            
            
            
        group = [-1 for i in range(len(arr) + 1)]
        sz = [0 for i in range(len(arr) + 1)]
        ones = [False for i in range(len(arr) + 1)]
        cnt = 0
        latest = -1
        for i in range(len(arr)):
            index = arr[i]
            ones[index] = True
            sz[index] = 1
            group[index] = arr[i]
            if index - 1 >= 1 and ones[index-1]:
                unite(index - 1, index)
            if index + 1 <= len(arr) and ones[index+1]:
                unite(index + 1, index)
            if sz[root(index)] == m:
                cnt += 1
            if cnt > 0:
                latest = i + 1
            #print(group, sz, ones, cnt)
        return latest
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        n = len(arr)
        F = [0] * n
        d = collections.defaultdict(int)
        
        def find(x):
            if F[x] < 0:
                return x
            else:
                F[x] = find(F[x])
                return F[x]
        
        t = [0] * n
        ans = -1
        
        for i in range(n):
            ind = arr[i] - 1
            d[1] += 1
            t[ind] = 1
            F[ind] = -1
            for newind in [ind-1, ind+1]:
                if newind < 0 or newind >= n or t[newind] == 0:
                    continue
                new = find(newind)
                d[-F[ind]] -= 1
                d[-F[new]] -= 1
                d[-F[ind]-F[new]] += 1
                F[ind] += F[new]
                F[new] = ind

            if d[m] > 0:
                ans = i + 1
        return ans
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        def dfs(start, end, cur):
            if end - start + 1 < m: return -2
            if end - start + 1 == m: return cur
            while arr[cur] < start or arr[cur] > end:
                cur -= 1
            return max(dfs(start, arr[cur] - 1, cur - 1), dfs(arr[cur] + 1, end, cur - 1))
        return dfs(1, len(arr), len(arr) - 1) + 1


class Solution:
    def findLatestStep(self, a: List[int], m: int) -> int:
        
        n = len(a)
        
        c = Counter()
        par = {}
        sz = {}
        
        def add(u):
            
            c[1] += 1
            sz[u] = 1
            par[u] = u
        
        def merge(u, v):
            
            ru = find(u)
            rv = find(v)
            
            if ru != rv:
                
                c[sz[ru]] -= 1
                c[sz[rv]] -= 1
                c[sz[ru] + sz[rv]] += 1
                
                par[rv] = ru
                sz[ru] += sz[rv]
                
        def find(u):
            
            if par[u] != u:
                par[u] = find(par[u])
            
            return par[u]
        
        ret = -1
        
        for i,x in zip(list(range(1, n+1)), a):
            
            add(x)
            
            if x-1 in par:
                merge(x-1, x)
            
            if x+1 in par:
                merge(x+1, x)
            
            #print(c[m])
            
            if c[m]:
                #print("hi")
                ret = i
        
        return ret

# union find problem
class UnionNode:
    def __init__(self,value,parent=None):
        self.value = value
        self.parent = parent
        self.size = 1
        
class UnionFind:
    def __init__(self):
        return
    
    def findGroup(self,curNode):
        while(curNode!=curNode.parent):
            curNode = curNode.parent
        return curNode
    
    def merge(self,node1,node2):
        root1,root2 = self.findGroup(node1),self.findGroup(node2)
        if(root1==root2):
            return -1
        if(root1.size>root2.size):
            root2.parent = root1
            root1.size += root2.size
            return root1.size
        else:
            root1.parent = root2
            root2.size += root1.size
            return root2.size
            
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        numOfm = 0
        res = -1
        string = [UnionNode(0) for _ in range(len(arr))]
        theUnionFind = UnionFind()
        for i in range(len(arr)):
            step = i+1
            loc = arr[i]-1
            thisUnionNode = string[loc]
            thisUnionNode.value = 1
            thisUnionNode.parent = thisUnionNode
            thisSize = 1
            if(loc-1>=0 and string[loc-1].value==1):
                # merge with left nei
                
                # if left nei has size m, numOfm -= 1
                newSize = theUnionFind.merge(string[loc-1],string[loc])
                if(newSize-thisSize==m):
                    numOfm -= 1
                thisSize = newSize
            if(loc+1<len(string) and string[loc+1].value==1):
                # merge with right nei
                
                # if right nei has size m, numOfm -= 1
                newSize = theUnionFind.merge(string[loc+1],string[loc])
                if(newSize-thisSize==m):
                    numOfm -= 1
                thisSize = newSize
            #print(thisSize)
            if(thisSize==m):
                numOfm += 1
            if(numOfm > 0):
                res = step
        
        return res
        
        
        
        

class Solution:
    def findLatestStep(self, A: List[int], m: int) -> int:
        n = len(A)
        arr = [0]*n
        parent = [i for i in range(n)]
        rank = [1]*n
        groupSize = [0]*n
        groupMap = set()
        ans = -1
        
        def ugm(x):
            nonlocal m
            if groupSize[x] == m:
                groupMap.add(x)
            else:
                if x in groupMap:
                    groupMap.remove(x)
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def join(x,y):
            px = find(x)
            py = find(y)
            if px != py:
                if px < py:
                    parent[py] = px
                    groupSize[px]+=groupSize[py]
                    groupSize[py]=0
                else:
                    parent[px] = py
                    groupSize[py]+=groupSize[px]
                    groupSize[px]=0
                ugm(px)
                ugm(py)
            
        
        for ind, num in enumerate(A):
            num-=1
            arr[num]=1
            groupSize[num]=1
            ugm(num)
            # print(arr)
            if num-1 >= 0 and arr[num-1]:
                join(num-1,num)
            
            if num+1 < n and arr[num+1]:
                join(num,num+1)
            # print(groupMap)
            if len(groupMap) > 0:
                ans = ind+1
        return ans
            
            

class UF:
    def __init__(self, N):
        self.N = N
        self.size = [0]*N
        self.stat = [False]*N
        self.id = list(range(N))
        self.sizes = collections.Counter()
        
    def find(self, x):
        if self.id[x] != x:
            self.id[x] = self.find(self.id[x])
        return self.id[x]
    
    def union(self, x, y):
        px = self.find(x)
        py = self.find(y)
        
        if px != py:
            self.size[px] += self.size[py]
            self.id[py] = px
            self.sizes[self.size[py]] -= 1

    def set(self, x):
        self.stat[x] = True
        self.size[x] += 1
        if x-1 >= 0 and self.stat[x-1]:
            self.union(x, x-1)
        if x+1 < self.N and self.stat[x+1]:
            self.union(x, x+1)
        self.sizes[self.size[x]] += 1
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        n = len(arr)
        uf = UF(n)
        ans = -1
        for step, idx in enumerate(arr, 1):
            uf.set(idx-1)
            if uf.sizes[m] > 0:
                ans = step

        return ans
class Group:
    def __init__(self, x, y):
        self.left = x
        self.right = y
        # self.update()
        self.n = y - x + 1
        
#     def update(self):
#         self.n = self.right - self.left + 1
        
#     def add_left(self, x):
#         self.left = x
#         self.update()
        
#     def add_right(self, y):
#         self.right = y
#         self.update()
    
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        groups = [None] * (1+len(arr))
        group_size = collections.defaultdict(int)
        visited = set()
        res = -1
        for step, x in enumerate(arr, 1):
            visited.add(x)
            left = right = x
            lsize = rsize = 0
            if x>1 and x-1 in visited:
                left = groups[x-1].left
                lsize = groups[x-1].n
            if x<len(arr) and x+1 in visited:
                right = groups[x+1].right
                rsize = groups[x+1].n
            g = Group(left, right)
            groups[left] = g
            groups[right] = g
            group_size[lsize+rsize+1] += 1
            if lsize != 0:
                group_size[lsize] -= 1
            if rsize != 0:
                group_size[rsize] -= 1
                
            if group_size[m] > 0:
                res = step
        return res
                
        
#         def find(parent, i):
#             if parent[i] == -1:
#                 return i 
#             if parent[i] != -1:
#                 return find(parent, parent[i]) 

#         def union(parent, x, y): 
#             px = find(parent, x) 
#             py = find(parent, y) 
#             parent[px] = py
            
#         parent = [-1] * (1+len(arr))
#         for x in arr:
#             parent[x] = x
#             if x > 1 and parent[x-1] != -1:
#                 union(parent, x, y)
#                 parent[x] = x
            
        

class UF:
    def __init__(self, n):
        self.p = [-1 for _ in range(n+1)]
        self.size = [0 for _ in range(n+1)]
        
    def find(self, x):
        if self.p[x] == -1:
            return -1
        if self.p[x] != x:
            self.p[x] = self.find(self.p[x])
        return self.p[x]

    def union(self, a, b):
        pa = self.find(a)
        pb = self.find(b)
        self.p[pa] = pb
        self.size[pb] += self.size[pa]
        
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        n = len(arr)
        uf = UF(len(arr))
        ans = -1

        for i, x in enumerate(arr):
            uf.p[x] = x
            uf.size[x] = 1

            if x > 0 and uf.find(x - 1) != -1:
                if uf.size[uf.find(x-1)] == m:
                    ans = i
                uf.union(x, x-1)

            if x < n and uf.find(x + 1) != -1:
                if uf.size[uf.find(x+1)] == m:
                    ans = i
                uf.union(x, x+1)
            if uf.size[uf.find(x)] == m:
                ans = i+1

        return ans


class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        n = len(arr)
        if m == n: return n 
        arr.append(n+1)
        start = {}
        finish = {}
        last = -1
        for level,i in enumerate(arr):
            if i-1 not in finish: finish[i-1] = i 
            if i+1 not in start: start[i+1] = i

            s, f = finish[i-1], start[i+1]
            start[s] = f 
            finish[f] = s
            
            for os, of in [[i+1, start[i+1]], [finish[i-1], i-1]]:
                if of-os+1 == m: last = level
                
            del start[i+1]
            del finish[i-1]
            
        return last
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        ranges = [None for _ in range(len(arr))]
        lefts = set([])
        
        best = -1
        for rnd, flipped in enumerate(arr):
            i = flipped - 1
            left = right = i

            if i > 0 and ranges[i-1] is not None:
                left = ranges[i-1][0]
                if left in lefts:
                    lefts.remove(left)
            if i < len(ranges)-1 and ranges[i+1] is not None:
                right = ranges[i+1][1]
                if ranges[i+1][0] in lefts:
                    lefts.remove(ranges[i+1][0])

            ranges[i] = [left, right]
            ranges[left] = [left, right]
            ranges[right] = [left, right]
            if right - left + 1 == m:
                lefts.add(left)
            if len(lefts) > 0:
                best = rnd + 1
        return best
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        length, count = [0 for i in range(len(arr) + 2)], [0 for i in range(len(arr) + 2)]
        ans = -1
        for i, num in enumerate(arr):
            left, right = length[num - 1], length[num + 1]
            length[num - left], length[num + right] = left + right + 1, left + right + 1
            count[left] -= 1
            count[right] -= 1
            count[left + right + 1] += 1
            if count[m] > 0:
                ans = i + 1
        return ans
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        n = len(arr)
        uf = UnionFind(n + 1, m)
        ans = -1
        visited = set()
        for i in range(n):
            a = arr[i]
            uf.add(a)
            if a - 1 in visited:
                uf.union(a, a - 1)
            if a + 1 in visited:
                uf.union(a, a + 1)
            if uf.cnt > 0:
                ans = i + 1
            visited.add(a)
        return ans
        
        
class UnionFind:
    def __init__(self, n, m):
        self.id = [-1 for _ in range(n)]
        self.size = [0 for _ in range(n)]
        self.cnt = 0
        self.m = m
        
    def add(self, i):
        self.id[i] = i
        self.size[i] = 1
        if self.get_size(i) == self.m:
            self.cnt += 1
        
    def find(self, i):
        root = i
        while root != self.id[root]:
            root = self.id[root]
        while root != i:
            j = self.id[i]
            self.id[i] = root
            i = j
        return root
    
    def union(self, i, j):
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i == root_j:
            return
        if self.get_size(i) == self.m:
            self.cnt -= 1
        if self.get_size(j) == self.m:
            self.cnt -= 1
        if self.size[root_i] < self.size[root_j]:
            self.id[root_i] = root_j
            self.size[root_j] += self.size[root_i]
        else:
            self.id[root_j] = root_i
            self.size[root_i] += self.size[root_j]
        if self.get_size(root_i) == self.m:
            self.cnt += 1
    
    def get_size(self, i):
        return self.size[self.find(i)]
class DSU:
    def __init__(self, size):
        self.parent = [i for i in range(size)]
        self.size = [1] * size
    def find(self, value):
        if value == self.parent[value]:
            return value
        self.parent[value] = self.find(self.parent[value])
        return self.parent[value]
    def merge(self, value1, value2):
        p1, p2 = self.parent[value1], self.parent[value2]
        if p1 == p2:
            return
        self.parent[p1] = p2
        self.size[p2] += self.size[p1]

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        n = len(arr)
        c = Counter()
        d = DSU(n + 2)
        vis = [0] * (n + 2)
        answer = -1
        for i in range(n):
            c[1] += 1
            vis[arr[i]] = 1
            if vis[arr[i]-1] and d.find(arr[i]) != d.find(arr[i]-1):
                c[d.size[d.find(arr[i])]] -= 1
                c[d.size[d.find(arr[i]-1)]] -= 1
                d.merge(arr[i], arr[i]-1)
                c[d.size[d.find(arr[i])]] += 1
            if vis[arr[i]+1] and d.find(arr[i]) != d.find(arr[i]+1):
                c[d.size[d.find(arr[i])]] -= 1
                c[d.size[d.find(arr[i]+1)]] -= 1
                d.merge(arr[i], arr[i]+1)
                c[d.size[d.find(arr[i])]] += 1
            if c[m] > 0:
                answer = i + 1
        return answer

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        from collections import defaultdict
        n = len(arr)
        parent = [i for i in range(0, n + 1)]
        group_count = defaultdict(int)
        l = [0] * n
        dd = defaultdict(int)
        def union(left, right):
            left_p = find(left)
            right_p = find(right)
            if left_p != right_p:
                parent[right_p] = left_p
                dd[group_count[right_p - 1]] -= 1
                dd[group_count[left_p - 1]] -= 1
                group_count[left_p - 1] += group_count[right_p - 1]
                group_count[right_p - 1] = 0
                dd[group_count[left_p - 1]] += 1
            # print(left, right, group_count)
            
        def find(i):
            p = parent[i]
            if parent[p] != p:
                pp = find(p)
                parent[i] = pp
            return parent[i]
        
        last = -1
        for idx, num in enumerate(arr):
            l[num - 1] = 1
            group_count[num - 1] = 1
            dd[1] += 1
            if num > 1 and l[num - 2] == 1:
                union(num - 1, num)
            if num != n and l[num] == 1:
                union(num, num + 1)
                
            # print(group_count)
                
            if m in dd and dd[m] > 0:
                last = idx + 1
            # print(idx, num, l, parent)
            # print(q)
        return last

class UnionFind:
    def __init__(self, n):
        self.parent = {}
        self.size = [0] * (n + 1)
        self.groups = collections.defaultdict(int)
        for i in range(1, n + 1):
            self.parent[i] = i
            
    def find(self, cur):
        if self.parent[cur] == cur:
            return cur
        self.parent[cur] = self.find(self.parent[cur])
        return self.parent[cur]
    
    def union(self, a, b):
        root_a = self.find(a)
        root_b = self.find(b)
        
        if root_a != root_b:
            self.parent[root_b] = root_a
            self.groups[self.size[root_a]] -= 1
            self.groups[self.size[root_b]] -= 1
            self.size[root_a] += self.size[root_b]
            self.groups[self.size[root_a]] += 1
            
            
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        
        result = 0
        UF = UnionFind(len(arr))
        for i in range(len(arr)):
            cur = arr[i]
            UF.size[cur] = 1
            UF.groups[1] += 1
            if cur - 1 >= 1 and UF.size[cur - 1] > 0:
                UF.union(cur, cur - 1)
            if cur + 1 <= len(arr) and UF.size[cur + 1] > 0:
                UF.union(cur, cur + 1)
            if m in UF.groups and UF.groups[m] > 0:
                    result = i + 1
        return -1 if result == 0 else result

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        mystr =[0]*len(arr)
        latest = -1
        mydict = {}
        count = 0
        for i in range(len(arr)):
            mystr[arr[i]-1] = 1
            if arr[i]-2 not in mydict.keys() and arr[i] not in mydict.keys():
                mydict[arr[i]-1] = [arr[i]-1,arr[i]-1, False]
                if m == 1:
                    count += 1
                    mydict[arr[i]-1] = [arr[i]-1,arr[i]-1, True]
                
            elif arr[i]-2 in mydict.keys() and arr[i] not in mydict.keys():
                head = mydict[arr[i]-2][0]
                if mydict[arr[i]-2][2] == True:
                    count -= 1
                del mydict[arr[i]-2]
                mydict[head] = [head,arr[i]-1, False]
                mydict[arr[i]-1] = [head,arr[i]-1, False]        
                if arr[i]-head == m:
                    count += 1
                    mydict[head] = [head,arr[i]-1, True]
                    mydict[arr[i]-1] = [head,arr[i]-1, True]     

            elif arr[i]-2 not in mydict.keys()  and arr[i] in mydict.keys():
                tail = mydict[arr[i]][1]
                if mydict[arr[i]][2] == True:
                    count -= 1
                del mydict[arr[i]]
                mydict[tail] = [arr[i]-1,tail, False]
                mydict[arr[i]-1] = [arr[i]-1,tail, False]             
                if tail - (arr[i]-1) + 1 == m:
                    count += 1
                    mydict[tail] = [arr[i]-1,tail, True]
                    mydict[arr[i]-1] = [arr[i]-1,tail, True]   
                
            else:
                head = mydict[arr[i]-2][0]
                tail = mydict[arr[i]][1]
                if mydict[arr[i]-2][2] == True:
                    count -= 1
                if mydict[arr[i]][2] == True:
                    count -= 1
                del mydict[arr[i]-2]
                del mydict[arr[i]]
                
                mydict[head] = [head,tail, False]
                mydict[tail] = [head,tail, False]           
                if tail - head + 1 == m:
                    count += 1
                    mydict[head] = [head,tail, True]
                    mydict[tail] = [head,tail, True]   
            if count > 0:
                latest = i+1
        return(latest)
class Solution:
    class UnionFind:
        def __init__(self, n, m):
            self.n = n
            self.m = m
            self.rank = [0] * n
            self.parent = [n for i in range(n)]
            self.counts = [0] * n
            self.counts_num = [0] * 100005
            
        def set(self, idx):
            self.rank[idx] = 1
            self.parent[idx] = idx
            self.counts[idx] = 1
            self.counts_num[self.counts[idx]] += 1
            if self.find(idx-1) != self.n:
                self.unite(idx, idx-1)
            if self.find(idx+1) != self.n:
                self.unite(idx, idx+1)
            
        def find(self, idx):
            if idx == self.n or self.parent[idx] == idx:
                return idx
            self.parent[idx] = self.find(self.parent[idx])
            return self.parent[idx]
            
        def unite(self, idx, idx2):
            if idx < 0 or idx2 < 0 or idx >= self.n or idx2 >= self.n:
                return
            root = self.find(idx)
            root2 = self.find(idx2)
            if root == root2:
                return
            self.counts_num[self.counts[root]] -= 1
            self.counts_num[self.counts[root2]] -= 1
            if self.rank[root] > self.rank[root2]:
                self.parent[root2] = root
                self.rank[root] += 1
                self.counts[root] += self.counts[root2]
                self.counts[root2] = 0
            else:
                self.parent[root] = root2
                self.rank[root2] += 1
                self.counts[root2] += self.counts[root]
                self.counts[root] = 0
            self.counts_num[self.counts[root]] += 1
            self.counts_num[self.counts[root2]] += 1

    def findLatestStep(self, arr: List[int], m: int) -> int:
        n = len(arr)
        uf = self.UnionFind(n, m)
        ans = -2
        for i, num in enumerate(arr):
            uf.set(num - 1)
            if uf.counts_num[m] > 0:
                ans = max(ans, i)
            
        return ans + 1

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        def union(x, y):
            mp[right[y] - left[y] + 1] -= 1
            mp[right[x] - left[x] + 1] -= 1
            ll = left[x]
            rr = right[y]
            left[ll] = left[rr] = ll
            right[rr] = right[ll] = rr
            mp[rr - ll + 1] += 1
            
        res = -1
        mp = Counter()
        n = len(arr)
        left = [-1] * (n + 1)
        right = [-1] * (n + 1)
        for i, a in enumerate(arr):
            mp[1] += 1
            left[a] = right[a] = a
                        
            if a - 1 > 0 and left[a - 1] != -1:
                union(a-1, a)                
            if a + 1 <= n and left[a + 1] != -1:
                union(a, a + 1)
                
            if mp[m] != 0:
                res = i + 1
        return res
            

class UnionFind:
    
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
        
    def find(self, u):
        if u != self.parent[u]:
            # path compression
            self.parent[u] = self.find(self.parent[u])
        return self.parent[u]
    
    def union(self, u, v):
        pu, pv = self.find(u), self.find(v)
        if pu == pv: # ALREADY in the same set
            return
        
        if self.rank[pu] > self.rank[pv]:
            self.parent[pv] = pu
            self.rank[pu] += self.rank[pv]
            
        elif self.rank[pv] > self.rank[pu]:
            self.parent[pu] = pv
            self.rank[pv] += self.rank[pu]
            
        else:
            self.parent[pu] = pv
            self.rank[pv] += self.rank[pu]
            

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:   
        n, ans = len(arr), -1
        uf = UnionFind(n)
        
        for step, i in enumerate(arr):
            i -= 1
            uf.rank[i] = 1
            for j in (i - 1, i + 1):
                if 0 <= j < n:
                    if uf.rank[uf.find(j)] == m:
                        ans = step
                    if uf.rank[j]:
                        uf.union(i, j)
        
        for i in range(n):
            if uf.rank[uf.find(i)] == m:
                return n
            
        return ans
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        n = len(arr)
        if n == m:
            return n
        day = [0] * (n + 1)
        for i, a in enumerate(arr):
            day[a] = i + 1
        ans = -1
        max_q = MaxQueue(m)
        for i in range(1, n+1):
            max_q.pop_expired(i)
            max_q.push(day[i], i)
            if i < m:
                continue
            left = right = math.inf
            if i - m >= 1:
                left = day[i-m]
            if i + 1 <= n:
                right = day[i+1]
            if max_q.max() < (d := min(left, right)):
                ans = max(ans, d - 1)
        return ans

class MaxQueue:
    def __init__(self, size):
        self.queue = deque()
        self.size = size

    def push(self, x, pos):
        while self.queue and self.queue[-1][0] < x:
            self.queue.pop()
        self.queue.append([x, pos])

    def pop_expired(self, pos):
        if self.queue and pos - self.queue[0][1] >= self.size:
            self.queue.popleft()

    def max(self):
        return self.queue[0][0]
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        res = -1
        mp = Counter()
        n = len(arr)
        v = [False] * (n + 1)
        left = [0] * (n + 1)
        right = [0] * (n + 1)
        for i, a in enumerate(arr):
            v[a] = True
            mp[1] += 1
            left[a] = a
            right[a] = a
            if a - 1 > 0 and v[a - 1]:
                mp[right[a] - left[a] + 1] -= 1
                mp[right[a-1] - left[a-1] + 1] -= 1
                ll = left[a - 1]
                rr = right[a]
                left[ll] = left[rr] = ll
                right[rr] = right[ll] = rr
                mp[rr - ll + 1] += 1
            if a + 1 <= n and v[a + 1]:
                mp[right[a] - left[a] + 1] -= 1
                mp[right[a+1] - left[a+1] + 1] -= 1
                ll = left[a]
                rr = right[a+1]
                left[ll] = left[rr] = ll
                right[rr] = right[ll] = rr
                mp[rr - ll + 1] += 1
                
            if mp[m] != 0:
                res = i + 1
        return res
            

class UnionFindSet:
    def __init__(self, n):
        self.parents = list(range(n))
        self.ranks = [0] * n
        
    def find(self, u):
        if u != self.parents[u]:
            self.parents[u] = self.find(self.parents[u])
        return self.parents[u]
    
    def union(self, u, v):
        pu, pv = self.find(u), self.find(v)
        if pu == pv:
            return False
        if self.ranks[pu] > self.ranks[pv]:
            self.parents[pv] = pu
            self.ranks[pu] += self.ranks[pv]
        else:
            self.parents[pu] = pv
            self.ranks[pv] += self.ranks[pu]
        return True

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        n, ans = len(arr), -1
        uf = UnionFindSet(n)
        
        for step, i in enumerate(arr):
            i -= 1
            uf.ranks[i] = 1
            for j in (i - 1, i + 1):
                if 0 <= j < n:
                    if uf.ranks[uf.find(j)] == m:
                        ans = step
                    if uf.ranks[uf.find(j)]:
                        uf.union(i, j)
        
        for i in range(n):
            if uf.ranks[uf.find(i)] == m:
                return n
            
        return ans
class UnionFind:
    def __init__(self, N):
        self.par = list(range(N))
        self.rank = [0]*N
        self.size = [0]*N
    
    def find(self, x):
        if self.par[x]!=x:
            self.par[x]=self.find(self.par[x])
        return self.par[x]
    
    def union(self, x, y):
        px = self.find(x)
        py = self.find(y)
        
        if px==py: return
        if self.rank[px]<self.rank[py]:
            self.par[px]=py
            self.size[py]+=self.size[px]
        elif self.rank[px]>self.rank[py]:
            self.par[py]=px
            self.size[px]+=self.size[py]
        else:
            self.par[py]=px
            self.size[px]+=self.size[py]
            self.rank[px]+=1

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        arr = [0]+arr
        N = len(arr)
        uf = UnionFind(N)
        res = []
        seen = set()
        matched = set()
        
        for i in range(1, N):
            seen.add(arr[i])
            matched.add(arr[i])
            uf.size[arr[i]]=1
            if arr[i]-1>=0 and arr[i]-1 in seen:
                uf.union(arr[i], arr[i]-1)
            if arr[i]+1<N and arr[i]+1 in seen:
                uf.union(arr[i], arr[i]+1)
                
            for j in list(matched):
                idx = uf.find(j)
                if uf.size[idx]!=m:
                    matched.remove(j)

            if matched: 
                res.append(i)

        return res[-1] if res else -1
        
        
        
        

class UnionFindSet:
    def __init__(self, n):
        self.parents=[i for i in range(n)]
        self.ranks=[0]*n  ## check whether the position is 1
        
    def find(self, x):
        if x !=self.parents[x]:
            self.parents[x]=self.find(self.parents[x])
        return self.parents[x]
    
    def union(self,x,y):
        px,py=self.find(x), self.find(y)
        if px==py: return False
        if self.ranks[px]>self.ranks[py]:
            self.parents[py]=px
            self.ranks[px]+=self.ranks[py]
        elif self.ranks[px]<self.ranks[py]:
            self.parents[px]=py
            self.ranks[py]+=self.ranks[px]
        else:
            self.parents[py]=px
            self.ranks[px]+=self.ranks[py]
        return True
        
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        if m==len(arr):
            return m 
        uf,ans=UnionFindSet(len(arr)),-1
        for step, idx in enumerate(arr):
            idx-=1
            uf.ranks[idx]=1
            for j in [idx-1,idx+1]:
                if 0<=j<len(arr):
                    if uf.ranks[uf.find(j)]==m:
                        ans=step
                    if uf.ranks[j]:  ### j is 1
                        uf.union(idx,j)
                        
        #for i in range(n):
        #    if uf.ranks[uf.find(i)]==m:
        #        return n
        return ans
        

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        n = len(arr)
        parents = list(range(n+1))
        ranks = [0] * (n+1)
        groupCounts = [0] * (n+1)
        counts = [1] * (n+1)
        visited = [False] * (n+1)
        
        def find(x):
            if parents[x] != x:
                parents[x] = find(parents[x])
            
            return parents[x]
    
        def union(x, y):
            r1 = find(x)
            r2 = find(y)
            
            if r1 != r2:
                groupCounts[counts[r1]] -= 1
                groupCounts[counts[r2]] -= 1
                counts[r1] = counts[r2] = counts[r1] + counts[r2]
                groupCounts[counts[r1]] += 1
                
                if ranks[r1] >= ranks[r2]:
                    parents[r2] = r1
                    ranks[r1] += ranks[r2]
                else:
                    parents[r1] = r2
                    ranks[r2] += ranks[r1]
        
        last = -1
        
        for step, index in enumerate(arr):
            groupCounts[1] += 1
            if index-1 > 0 and visited[index-1]:
                union(index, index-1)
            
            if index+1 <= n and visited[index+1]:
                union(index, index+1)
            
            visited[index] = True
            
            if groupCounts[m]:
                last = step + 1

        return last

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        def find_set(x):
            if parents[x][0] == x:
                return x
            else:
                return find_set(parents[x][0])


        def union_set(x, y):
            x_root = find_set(x)
            y_root = find_set(y)
            parents[y_root][1] += parents[x_root][1]
            parents[x_root][0] = y_root


        n = len(arr)
        parents = [[i, 1] for i in range(n)]
        visited = [False for i in range(n)]
        answer = -1
        d = {}
        for i in range(n):
            num = arr[i] - 1
            visited[num] = True
            if num > 0 and visited[num - 1]:
                d[parents[find_set(num - 1)][1]] -= 1
                union_set(num - 1, num)
            if num + 1 < n and visited[num + 1]:
                d[parents[find_set(num + 1)][1]] -= 1
                union_set(num + 1, num)
            d[parents[num][1]] = 1 if parents[num][1] not in d else d[parents[num][1]] + 1
            if m in d and d[m] > 0:
                answer = i + 1
        return answer
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        res = -1
        n = len(arr)
        
        if n == m: return n;
        
        g = [0] * (n + 2)
        for i, x in enumerate(arr):
            l = g[x - 1]
            r = g[x + 1]
            
            if l == m or r == m: res = i
            
            g[x - l] = g[x + r] = l + r + 1;
        return res
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        n = len(arr)
        degree = [0] * n
        par = list(range(n))

        def find(x: int) -> int:
            if par[x] == x:
                return x
            return find(par[x])

        def union(x: int, y: int) -> None:
            px, py = find(x), find(y)
            if px == py:
                return
            if degree[px] > degree[py]:
                par[py] = px
                degree[px] += degree[py]
            else:
                par[px] = py
                degree[py] += degree[px]

        res = -1
        for i, num in enumerate(arr):
            num -= 1
            degree[num] = 1
            for nei in (num - 1, num + 1):
                if 0 <= nei < n:
                    if degree[find(nei)] == m:
                        res = i
                    if degree[nei]:
                        union(nei, num)

        # Check the last
        for i in range(n):
            if degree[find(i)] == m:
                return n
        return res
from collections import defaultdict
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        size_records = defaultdict(int)
        range_records = defaultdict(list)
        ans = -1
        for index, i in enumerate(arr):
            new_range = [i, i]
            if range_records[i-1]:
                new_range = [range_records[i-1][0], i]
                size_records[range_records[i-1][1] - range_records[i-1][0] + 1] -= 1
            if range_records[i+1]:
                new_range = [new_range[0], range_records[i+1][1]]
                size_records[range_records[i+1][1] - range_records[i+1][0] + 1] -= 1
            # print(new_range)
            size_records[new_range[1] - new_range[0] + 1] += 1
            range_records[new_range[0]] = new_range
            range_records[new_range[1]] = new_range
            if size_records[m]:
                ans = index + 1
        return ans 
        

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        n = len(arr)
        if m == n:
            return n
        
        groups = defaultdict(set)
        parents = [i for i in range(n)]
        size = [0] * n
        
        def find(node):
            if parents[node] == node:
                return node
            parent = find(parents[node])
            return parent
        
        def union(a, b):
            para = find(a)
            parb = find(b)
            if para != parb:
                groups[parb].update(groups[para])
                groups.pop(para)
                parents[para] = parb
                
        def get_size(a):
            parent = find(parents[a])
            return len(groups[parent])
        
        def update(i):
            check = get_size(i)
            sizes[check] -= 1
            if sizes[check] == 0:
                sizes.pop(check)
        
        arr = [i-1 for i in arr]
        step = 0
        ans = -1
        sizes = Counter()
        for i in arr:
            step += 1
            size[i] += 1
            groups[i].add(i)
            sizes[1] += 1
            if i-1 >= 0 and i+1 < n and size[i-1] and size[i+1]:
                update(i-1)
                update(i+1)
                union(i, i-1)
                union(i+1, i-1)
                sizes[1] -= 1
                if sizes[1] == 0:
                    sizes.pop(1)
                new_size = get_size(i-1)
                sizes[new_size] += 1
            elif i-1 >= 0 and size[i-1]:
                update(i-1)
                union(i, i-1)
                sizes[1] -= 1
                if sizes[1] == 0:
                    sizes.pop(1)
                new_size = get_size(i-1)
                sizes[new_size] += 1
            elif i+1 < n and size[i+1]:
                update(i+1)
                union(i, i+1)
                sizes[1] -= 1
                if sizes[1] == 0:
                    sizes.pop(1)
                new_size = get_size(i+1)
                sizes[new_size] += 1
            if m in sizes:
                ans = step
        return ans
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        bit_to_parent = [-1] * len(arr)
        size = [0] * len(arr)
        last_step = -1
        groups_of_size_m = 0

        def root(bit):
            if bit_to_parent[bit] == -1:
                return -1

            root = bit
            while root != bit_to_parent[root]:
                root = bit_to_parent[root]
            
            curr = bit
            while curr != root:
                tmp = bit_to_parent[bit]
                bit_to_parent[bit] = root
                curr = tmp

            # print("root", bit, root)
            return root
        
        def union(b1, b2):
            # print("union", b1, b2)
            nonlocal size
            nonlocal groups_of_size_m
            
            if b1 == b2:
                size[b1] = 1
                if m == 1:
                    groups_of_size_m += 1
                return
            
            root_b1 = root(b1)
            root_b2 = root(b2)
            
            if root_b1 == -1 or root_b2 == -1:
                # Can't union
                return
            
            if size[root_b1] >= size[root_b2]:
                parent = root_b1
                child = root_b2
            else:
                parent = root_b2
                child = root_b1
            
            old_parent_size = size[parent]
            old_child_size = size[child]
            
            size[parent] += size[child]
            bit_to_parent[child] = parent
            
            # print("union", b1, b2, parent, child, old_parent_size, old_child_size, size[parent])
            if old_parent_size == m:
                groups_of_size_m -= 1
            
            if old_child_size == m:
                groups_of_size_m -= 1
            
            if size[parent] == m:
                groups_of_size_m += 1
            
            return parent
        
        for i in range(len(arr)):
            bit = arr[i] - 1
            bit_to_parent[bit] = bit
            
            union(bit, bit)
            if bit - 1 >= 0:
                union(bit, bit - 1)
            if bit + 1 < len(arr):
                union(bit, bit + 1)
            
            if groups_of_size_m > 0:
                last_step = i + 1
        
        return last_step
import heapq
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        if m == len(arr):
            return m
        times = [0] * len(arr)
        for t, a in enumerate(arr):
            times[a - 1] = t
        # print(times)
        res = -1
        h = [(-a, i) for i, a in enumerate(times[:m])]
        heapq.heapify(h)
        maxtime = [-h[0][0]]
        for i in range(m, len(times)):
            heapq.heappush(h, (-times[i], i))
            while h[0][1] <= i - m:
                heapq.heappop(h)
            maxtime.append(-h[0][0])
        # print(maxtime)
        if maxtime[0] < times[m]:
            res = times[m]
        if maxtime[-1] < times[-m - 1]:
            res = max(res, times[-m - 1])
        for i in range(1, len(times) - m):
            if times[i - 1] > maxtime[i] and times[i + m] > maxtime[i]:
                res = max(res, min(times[i - 1], times[i + m]))
        return res
class UnionFind:
    
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0]*n
        self.size = [0]*n
        self.groupCount = [0]*(n+1)
    
    def find(self, x):
        if x != self.parent[x]:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def add(self, x):
        self.size[x] = 1
        self.groupCount[1] += 1
    
    def union(self, x, y):
        px = self.find(x)
        py = self.find(y)
        if px == py:
            return False
        self.groupCount[self.size[px]] -= 1
        self.groupCount[self.size[py]] -= 1
        if self.rank[px] > self.rank[py]:
            self.parent[py] = px
            self.size[px] += self.size[py]
            self.size[py] = 0
        elif self.rank[py] > self.rank[px]:
            self.parent[px] = py
            self.size[py] += self.size[px]
            self.size[px] = 0
        else:
            self.parent[px] = py
            self.size[py] += self.size[px]
            self.size[px] = 0
            self.rank[py] += 1
        self.groupCount[self.size[px]] += 1
        self.groupCount[self.size[py]] += 1
        return True
    
    def getSize(self, i):
        px = self.find(i)
        return self.size[px]
    
class Solution:
    
    def findLatestStep(self, arr: List[int], m: int) -> int:
        disjoint = UnionFind(len(arr))
        ans = - 1
        val = [0]*len(arr)
        for k in range(len(arr)):
            index = arr[k] - 1
            val[index] += 1
            disjoint.add(index)
            if index > 0 and val[index] == val[index-1]:
                disjoint.union(index, index - 1)
            if index + 1 < len(val) and val[index] == val[index+1]:
                disjoint.union(index, index + 1)
            #print(k, disjoint.groupCount)
            if disjoint.groupCount[m] > 0:
                ans = k + 1
            '''
            i = 0
            while i < len(arr):
                if val[i] == 1 and disjoint.getSize(i) == m:
                    i += disjoint.getSize(i)
                    ans = k + 1
                    continue
                i += 1
            '''
            #print(k, disjoint.size, val)
        return ans 
    
    '''
    def findLatestStep(self, arr: List[int], m: int) -> int:
        def check(i):
            val = [0]*len(arr)
            for k in range(i+1):
                val[arr[k]-1] = 1
            count = 0
            success = False
            for k in range(len(val)):
                if val[k] > 0:
                    count += 1
                else:
                    if count == m:
                        success = True
                        break
                    count = 0
            if count == m:
                success = True
            return success                
            
        left = 0
        right = len(arr)
        while left < right:
            mid = left + (right - left) //2
            if not check(mid):
                right = mid
            else:
                left = mid + 1
        print(left)
        if left == 0 and not check(left):
            return -1
        else:
            return left
    '''

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        n = len(arr)
        groups = defaultdict(set)
        parents = [i for i in range(n)]
        size = [0] * n
        
        def find(node):
            if parents[node] == node:
                return node
            parent = find(parents[node])
            return parent
        
        def union(a, b):
            para = find(a)
            parb = find(b)
            if para != parb:
                groups[parb].update(groups[para])
                groups.pop(para)
                parents[para] = parb
                
        def get_size(a):
            parent = find(parents[a])
            return len(groups[parent])
        
        def update(i):
            check = get_size(i)
            sizes[check] -= 1
            if sizes[check] == 0:
                sizes.pop(check)
        
        arr = [i-1 for i in arr]
        step = 0
        ans = -1
        sizes = Counter()
        for i in arr:
            step += 1
            size[i] += 1
            groups[i].add(i)
            sizes[1] += 1
            if i-1 >= 0 and i+1 < n and size[i-1] and size[i+1]:
                update(i-1)
                update(i+1)
                union(i, i-1)
                union(i+1, i-1)
                sizes[1] -= 1
                if sizes[1] == 0:
                    sizes.pop(1)
                new_size = get_size(i-1)
                sizes[new_size] += 1
            elif i-1 >= 0 and size[i-1]:
                update(i-1)
                union(i, i-1)
                sizes[1] -= 1
                if sizes[1] == 0:
                    sizes.pop(1)
                new_size = get_size(i-1)
                sizes[new_size] += 1
            elif i+1 < n and size[i+1]:
                update(i+1)
                union(i, i+1)
                sizes[1] -= 1
                if sizes[1] == 0:
                    sizes.pop(1)
                new_size = get_size(i+1)
                sizes[new_size] += 1
            if m in sizes:
                ans = step
        return ans
class UnionSet:
    def __init__(self, n):
        self.par = list(range(n))
        self.ed = list(range(n))
    def find(self, i):
        if self.par[i] != i:
            par = self.find(self.par[i])
            self.par[i] = par
        return self.par[i]
    def merge(self, i, j):
        par1 = self.find(i)
        par2 = self.find(j)
        ed1 = self.ed[par1]
        ed2 = self.ed[par2]
        self.par[max(par1, par2)] = min(par1, par2)
        self.ed[par1] = self.ed[par2] = max(ed1, ed2)
    def get_ed(self, i):
        return self.ed[i]

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        n = len(arr)
        us = UnionSet(n)
        bits = [0] * n
        result = -1
        cnt = 0
        for i, pos in enumerate(arr):
            # print(bits)
            # print(us.par)
            # print(us.ed)
            # print(cnt)
            # print('-------')
            pos -= 1
            bits[pos] = 1
            if pos > 0 and bits[pos - 1] == 1:
                st = us.find(pos - 1)
                ed = us.get_ed(st)
                if ed - st + 1 == m:
                    cnt -= 1
                us.merge(pos, pos - 1)
            if pos < n - 1 and bits[pos + 1] == 1:
                st = us.find(pos + 1)
                ed = us.get_ed(st)
                if ed - st + 1 == m:
                    cnt -= 1
                us.merge(pos, pos + 1)
            st = us.find(pos)
            ed = us.get_ed(st)
            if ed - st + 1 == m:
                cnt += 1
            if cnt > 0:
                result = i + 1
        return result
from collections import defaultdict
class group:
    def __init__(self,n,m):
        self.groups=[i for i in range(n+1)]
        self.group_size={}
        self.m=m
        self.m_sizes={}
        for i in range(1,n+1):
            self.group_size[i]=1
        
        
    def union(self,i,j):
        if i==j:
            self.group_size[i]=1
            gi=gj=i
        else:
            gi=self.get_group(i)
            gj=self.get_group(j)
            if self.group_size[gi]>self.group_size[gj]:
                gj,gi=gi,gj
            self.groups[gi]=gj
        if self.group_size[gj]==self.m and gj in self.m_sizes  :
            del(self.m_sizes[gj])
        if self.group_size[gi]==self.m and gi in self.m_sizes :
            del(self.m_sizes[gi])
        if i!=j:
            self.group_size[gj]+=self.group_size[gi]
        
        if self.group_size[gj]==self.m:
            self.m_sizes[gj]=1
        return self.group_size[gj]

    def get_group(self,i):
        if self.groups[i]!=i:
            self.groups[i]=self.get_group(self.groups[i])
        return self.groups[i]
        
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        bits=[0]*(len(arr)+2)
        GG=group(len(arr),m)
        steps=0
        latest=-1
        for i in arr:
            steps+=1
            bits[i]=1
            sz=1
            if bits[i-1]==1:
                sz=GG.union(i-1,i)
            if bits[i+1]==1:
                sz=GG.union(i,i+1)
            if bits[i-1]==bits[i+1]==0:
                sz=GG.union(i,i)
            if GG.m_sizes:
                latest=steps
        return latest
                
            
        

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        cnt, ans = 0, -1
        start_end, end_start = {}, {}
        for i, n in enumerate(arr, start=1):
            start, end = n, n
            if n-1 in end_start:
                start = end_start[n-1] 
                del end_start[n-1]
                if n-start == m:
                    cnt -= 1
            if n+1 in start_end:
                end = start_end[n+1]
                del start_end[n+1]
                if end-n == m:
                    cnt -= 1
            start_end[start] = end
            end_start[end] = start
            if end-start+1 == m:
                cnt += 1
            if cnt >= 1:
                ans = i
        return ans

#Work backwards using a sorted dictionary to keep track of starting index and group length
#Note: the key idea is to use dictionary bisect Java -> treemap.floorkey()

from sortedcontainers import SortedDict
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        if m == len(arr) : return m
        
        pos2length = SortedDict()
        
        #means starting from index 0, there is group length of n
        pos2length[0] = len(arr)
        
        #traverse the arr / break points backwards
        for i in range(len(arr) - 1, -1, -1):
            index = arr[i] - 1
            interval_pos = pos2length.bisect_right(index) - 1
            floor, length = pos2length.popitem(index = interval_pos)
            
            if floor != index:
                pos2length[floor] = index - floor
                if pos2length[floor] == m: return i
            if floor + length - 1 != index:
                pos2length[index + 1] = length - (index - floor) - 1
                if pos2length[index + 1] == m: return i
                
        return -1

class Solution:
    def findLatestStep(self, arr, m):
        l = [[] for _ in range(len(arr)+2)]; step = 1; cnt = 0; ans = -1
        
        for idx in arr:
            l[idx].append(1)
            
            if l[idx][0] == m:
                cnt += 1
            
            if l[idx-1] and l[idx+1]:
                if l[idx-1][0] == m:
                    cnt -= 1
                if l[idx+1][0] == m:
                    cnt -= 1
                if l[idx][0] == m:
                    cnt -= 1
                    
                _sum = l[idx-1][0] + l[idx][0] + l[idx+1][0]
                i = 1
                while l[idx-i]:
                    l[idx-i] = l[idx]
                    i+= 1

                i = 1
                while l[idx+i]:
                    l[idx+i] = l[idx]
                    i+= 1

                l[idx].pop(); l[idx].append(_sum)
                if l[idx][0] == m:
                    cnt += 1
                
            elif l[idx-1]:
                if l[idx-1][0] == m:
                    cnt -= 1
                if l[idx][0] == m:
                    cnt -= 1
                    
                _sum = l[idx-1][0] + l[idx][0]
                l[idx] = l[idx-1]
                l[idx].pop(); l[idx].append(_sum)
                
                if l[idx][0] == m:
                    cnt += 1
                
            elif l[idx+1]:
                if l[idx+1][0] == m:
                    cnt -= 1
                if l[idx][0] == m:
                    cnt -= 1
                    
                _sum = l[idx+1][0] + l[idx][0]
                l[idx] = l[idx+1]
                l[idx].pop(); l[idx].append(_sum)
                
                if l[idx][0] == m:
                    cnt += 1
                
            if cnt > 0:
                ans = step
                
            step += 1
                
        return ans
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
    
        class DisjointSet:

            def __init__(self, n):
                self.parent = [-1] * n
                self.size = [0] * n
            
            def make_set(self, v):
                self.parent[v] = v
                self.size[v] = 1

            def find_set(self, v):
                if self.parent[v] != v:
                    self.parent[v] = self.find_set(self.parent[v])
                return self.parent[v]

            def union_sets(self, a, b):
                a = self.find_set(a)
                b = self.find_set(b)
                if self.size[a] < self.size[b]:
                    a, b = b, a
                self.parent[b] = a
                self.size[a] += self.size[b]
        
        n = len(arr)
        if n == m:
            return n
        ans = -1
        ds = DisjointSet(n)
        for step, i in enumerate(arr):
            i -= 1
            ds.make_set(i)
            for j in (i-1, i+1):
                if 0 <= j <= n-1 and ds.parent[j] > -1:
                    rep = ds.find_set(j)
                    if ds.size[rep] == m:
                        ans = step
                    ds.union_sets(i, j)
        return ans
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        # union found and counter
        counter = collections.Counter()
        hs = {}  # record size of strings, with keys as starting index
        n = len(arr)
        # union find
        def find(x, union):
            root = x
            if union[x] != x:
                union[x] = find(union[x], union)
            return union[x]
                
        union = {}
        
        res = -1
        for i in range(n):
            idx = arr[i]
            union[idx] = idx
            hs[idx] = 1 # size of 1
            counter[1] += 1
            if idx-1 in union:
                left = find(idx-1, union)
                union[idx] = find(idx-1, union)
                # substract from counter
                counter[hs[left]] -= 1
                counter[hs[idx]] -= 1
                counter[hs[left]+hs[idx]] += 1
                hs[left] += hs[idx]
                
            if idx+1 in union:
                right = find(idx+1, union)
                union[idx+1] = find(idx, union)
                # substract from counter
                t = find(idx, union)
                counter[hs[right]] -= 1
                counter[hs[t]] -= 1
                counter[hs[right] + hs[t]] += 1
                hs[t] += hs[right]
                
            if counter[m] > 0:
                res = i+1
                
        return res
            

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        n = len(arr)
        sele = [0]*n
        repe = [i for i in range(n)]
        sis = [1]*n
        def getRepe(a):
            if repe[a]==a:
                return a
            repe[a] = getRepe(repe[a])
            return repe[a]
        def join(a,b):
            ra,rb = getRepe(a), getRepe(b)
            # print(ra,rb)
            repe[b] = ra
            sis[ra]+=sis[rb]
        def sete(x):
            # print('-',x)
            if x>0 and sele[x-1]:
                join(x-1,x)
            if x < n-1 and sele[x+1]:
                join(x,x+1)
            sele[x]=1
        res = -1
        for i,v in enumerate(arr,1):
            if v>1 and sele[v-2] and sis[getRepe(v-2)]==m:
                res = i - 1
            if v<n and sele[v] and sis[getRepe(v)]==m:
                res = i-1
            sete(v-1)
            if sis[getRepe(v-1)]==m:
                res = i
        return res
class DSU():
    
    def __init__(self):
        self.size={}
        self.parent={}
    
    def find(self,x):
        if x!=self.parent[x]:
            self.parent[x]=self.find(self.parent[x])
        
        return self.parent[x]
    
    def union(self,x,y):
        xp,yp=self.find(x),self.find(y)
        if xp==yp:
            return False
        
        if self.size[xp]<self.parent[yp]:
            xp,yp=yp,xp
        
        self.size[xp]+=self.size[yp]
        self.parent[yp]=xp
        return True
    
    def add_node(self,x):
        self.parent[x]=x
        self.size[x]=1
    
    def get_size(self,x):
        return self.size[self.find(x)]

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        
        dsu=DSU()
        sizes=collections.Counter()
        ans=-1
        for i,n in enumerate(arr,1):
            dsu.add_node(n)
            for nn in [n-1,n+1]:
                if nn in dsu.parent:
                    sizes[dsu.get_size(nn)]-=1
                    dsu.union(n,nn)
            sizes[dsu.get_size(n)]+=1
            if sizes[m]>0:
                ans=i
        return ans
                
            

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        counts = defaultdict(int)
        root = [i for i in range(len(arr))]
        size = [0 for i in range(len(arr))]
        rank = [1 for i in range(len(arr))]
        def find(i):
            if root[i-1] != i-1:
                root[i-1] = find(root[i-1]+1)
            return root[i-1]
        def union(i,j):
            pi = find(i)
            pj = find(j)
            length = size[pi]+size[pj]
            if pi != pj:
                if rank[pi] <= rank[pj]:
                    root[pi] = pj
                    if rank[pi] == rank[pj]:
                        rank[pj] += 1
                else:
                    root[pj] = pi
                size[root[pi]] = length
        step = -1
        for i in range(len(arr)):
            size[arr[i]-1] += 1
            if arr[i] - 1 != 0 and size[find(arr[i]-1)] != 0:
                counts[size[find(arr[i]-1)]] -= 1
                union(arr[i]-1, arr[i])
            if arr[i] + 1 != len(arr)+1 and size[find(arr[i]+1)] != 0:
                counts[size[find(arr[i]+1)]] -= 1
                union(arr[i]+1, arr[i])
            counts[size[find(arr[i])]] += 1
            if counts[m] != 0:
                step = i+1
        return step
    '''
    [5,3,4,7,8,14,11,9,2,12,1,13,10,6]
6
    '''
class UnionFindSet:
    def __init__(self, n):
        self.parents = list(range(n))
        self.ranks = [0] * n
        
    def find(self, u):
        if u != self.parents[u]:
            self.parents[u] = self.find(self.parents[u])
        return self.parents[u]
    
    def union(self, u, v):
        pu, pv = self.find(u), self.find(v)
        if pu == pv:
            return False
        if pu < pv:
            self.parents[pv] = pu
            self.ranks[pu] += self.ranks[pv]
        else:
            self.parents[pu] = pv
            self.ranks[pv] += self.ranks[pu]
        return True

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        n, res = len(arr), -1
        uf = UnionFindSet(n)
        
        for step, i in enumerate(arr):
            i -= 1
            uf.ranks[i] = 1
            for j in [i - 1, i + 1]:
                if 0 <= j < n:
                    # u67e5u770bu672au5408u5e76u524du4e0au4e00u6b65u65f6uff0cu662fu5426u4e3am
                    if uf.ranks[uf.find(j)] == m:
                        res = step
                    if uf.ranks[j]:
                        uf.union(i, j)
        for i in range(n):                
            if uf.ranks[uf.find(i)] == m:
                return n
            
        return res
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        length = [0] * (len(arr) + 2)
        cnt = [0] * (len(arr) + 1)
        res = -1
        for i, num in enumerate(arr):
            left, right = length[num - 1], length[num + 1]
            length[num] = length[num - left] = length[num + right] = left + right + 1
            cnt[length[num]] += 1
            cnt[left] -= 1
            cnt[right] -= 1
            if cnt[m] > 0:
                res = i + 1
        return res
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        n=max(arr)

        dp=[[0,[i,i]] for i in range(n+2)]
        
        memo={}
        res=-1
        
        for j,i in enumerate(arr):  
            
            dp[i][0]=1
            val=1
            
            if dp[i-1][0]:
                memo[dp[i-1][0]]-=1
                left,right,val=dp[i-1][1][0],i,val+dp[i-1][0]
                dp[left]=dp[right]=[val,[left,right]]


            if dp[i+1][0]:
                memo[dp[i+1][0]]-=1
                left,right,val=dp[i][1][0],dp[i+1][1][1],val+dp[i+1][0]
                dp[left]=dp[right]=[val,[left,right]]

            memo[val]=memo.get(val,0)+1


            if memo.get(m,0):
                res=j+1
                
        return res
class Solution:
    def findLatestStep(self, A: List[int], T: int, last = -1) -> int:
        seen, ok = set(), set()
        A = [i - 1 for i in A]      # u2b50ufe0f -1 for 0-based indexing
        N = len(A)
        P = [i for i in range(N)]   # U0001f642 parent representative sets
        L = [1] * N                 # U0001f925 length of each representative set
        def find(x):
            if x != P[x]:
                P[x] = find(P[x])
            return P[x]
        def union(a, b):
            a = find(a)
            b = find(b)
            P[b] = a                # U0001f517 arbitrary choice for parent representative
            return L[a] + L[b]
        step = 1
        for i in A:
            seen.add(i)
            if 0 < i     and find(P[i - 1]) in ok: ok.remove(find(P[i - 1]))
            if i + 1 < N and find(P[i + 1]) in ok: ok.remove(find(P[i + 1]))
            if i - 1 in seen: L[i] = union(i, i - 1)
            if i + 1 in seen: L[i] = union(i, i + 1)
            if L[i] == T:
                ok.add(i)          # u2705 i is the parent reprentative of the set with U0001f3af target T length
            if len(ok):
                last = step
            step += 1
        return last
#Work backwards using a sorted dictionary to keep track of starting index and group length
#Note: the key idea is to use dictionary bisect Java -> treemap.floorkey()

from sortedcontainers import SortedDict
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        if m == len(arr) : return m
        
        pos2length = SortedDict()
        
        #means starting from index 0, there is group length of n
        pos2length[0] = len(arr)
        
        #traverse the arr / break points backwards
        for i in range(len(arr) - 1, -1, -1):
            index = arr[i] - 1
            
            #this is equivalent to Java -> treemap.floorkey() function
            interval_index = pos2length.bisect_right(index) - 1
            floor, length = pos2length.popitem(index = interval_index)
            
            if floor != index:
                pos2length[floor] = index - floor
                if pos2length[floor] == m: return i
            if floor + length - 1 != index:
                pos2length[index + 1] = length - (index - floor) - 1
                if pos2length[index + 1] == m: return i
                
        return -1
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        n = len(arr)
        parent = [i for i in range(n + 1)]
        size = [0] * (n + 1)
        counter = collections.Counter()
        def find(x):
            if x != parent[x]:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            x, y = min(x, y), max(x, y)
            px, py = find(x), find(y)
            if px != py:
                counter[size[px]] -= 1
                counter[size[py]] -= 1
                parent[py] = px
                size[px] += size[py]
                counter[size[px]] += 1
        A = [0] * (n + 2)
        res = -1
        for i, cur in enumerate(arr):
            A[cur] = 1
            size[cur] = 1
            counter[1] += 1
            if A[cur - 1] == 1:
                union(cur - 1, cur)
            if A[cur + 1] == 1:
                union(cur, cur + 1)
            if counter[m] > 0:
                res = i + 1
        return res
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        N = len(arr)
        arr = [x - 1 for x in arr]
        parent = [x for x in range(N)]
        size = [1 for _ in range(N)]
        used = [False for _ in range(N)]
        
        def ufind(a):
            if a == parent[a]:
                return a
            parent[a] = ufind(parent[a])
            return parent[a]
        
        def uunion(a, b):
            sa = ufind(a)
            sb = ufind(b)
            
            if sa != sb:
                parent[sa] = parent[sb]
                size[sb] += size[sa]
        
        def usize(a):
            return size[ufind(a)]
            
        counts = [0] * (N+1)
        
        latest = -1
        for index, x in enumerate(arr):
            left = 0
            if x - 1 >= 0 and used[x - 1]:
                left = usize(x - 1)
                
            right = 0
            if x + 1 < N and used[x + 1]:
                right = usize(x + 1)
                
            current = 1
            counts[1] += 1
            if left > 0:
                counts[left] -= 1
            if right > 0:
                counts[right] -= 1
            counts[1] -= 1
            used[x] = True
            
            new_size = left + right + current
            #print(x, left, right)
            counts[new_size] += 1
            if left > 0:
                uunion(x, x - 1)
            if right > 0:
                uunion(x, x + 1)
            
            #print(counts)
            if counts[m] > 0:
                latest = max(latest, index + 1)
        return latest
from collections import defaultdict
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        if m == 0:
            return 0
        if not arr:
            return -1
        step = 0
        uf = UF(len(arr), m)
        res = -1
        for num in arr:
            step += 1
            # uf.add(num)
            if uf.add(num):
                res = step - 1
#             print(uf.cnt.items(), step, num)
#             print(uf.length_list, res)
        
        if uf.cnt.get(m):
            return step
        return res
        
class UF:
    def __init__(self, n, m):
        self.max_length = n + 1
        self.target = m
        self.length_list = [0 for _ in range(self.max_length)] # length of each idx. Always check root node length.
        self.cnt = defaultdict(int) # Save length we have.
        self.par = defaultdict(int)
        
    def find(self, x):
        if self.par[x] != x:
            self.par[x] = self.find(self.par[x])
        return self.par[x]
        
    def union(self, x, y):
        par_x = self.find(x)
        par_y = self.find(y)
        if par_x != par_y:
            self.par[par_x] = par_y
        self.cnt[self.length_list[par_x]] -= 1
        self.cnt[self.length_list[par_y]] -= 1
        self.length_list[par_y] += self.length_list[par_x]
        self.cnt[self.length_list[par_y]] += 1
        return False

    def add(self, x):
        tmp = self.cnt.get(self.target)
        self.par[x] = x
        self.length_list[x] = 1
        self.cnt[1] += 1
        if x >= 2 and self.length_list[x-1] > 0:
            self.union(x, x-1)
        if x <= self.max_length - 2 and self.length_list[x+1] > 0:
            self.union(x, x+1)
        # print(tmp, self.cnt[self.target], self.target)
        if tmp and self.cnt[self.target] == 0:
            return True
        return False
        

# class Solution:
#     def findLatestStep(self, A, m):
#         length = [0] * (len(A) + 2)
#         count = [0] * (len(A) + 1)
#         res = -1
#         for i, a in enumerate(A):
#             left, right = length[a - 1], length[a + 1]
#             length[a] = length[a - left] = length[a + right] = left + right + 1
#             count[left] -= 1
#             count[right] -= 1
#             count[length[a]] += 1
#             if count[m]:
#                 res = i + 1
#         return res        


class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        n = len(arr)
        sz, cnt = [0]*n, defaultdict(int)
        res = -2
        parent = [i for i in range(n)]

        def find(x):
            if x != parent[x]:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            rx, ry = find(x), find(y)
            cnt[sz[rx]] -= 1
            cnt[sz[ry]] -= 1
            cnt[sz[rx]+sz[ry]] += 1
            parent[ry] = rx
            sz[rx] += sz[ry]
            return

        for idx, a in enumerate(arr):
            a -= 1
            sz[a] = 1
            cnt[1] += 1
            if a-1 >= 0 and sz[a-1] > 0:
                union(a-1,a)
            if a+1 < n and sz[a+1] > 0:
                union(a,a+1)
            if cnt[m] > 0:
                res = idx
        return res+1
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        n = len(arr)
        parents = list(range(n))
        ranks = [0] * n
        groupCounts = [0] * (n+1)
        counts = [1] * n
        visited = [False] * n
        
        def find(x):
            if parents[x] != x:
                parents[x] = find(parents[x])
            
            return parents[x]
    
        def union(x, y):
            r1 = find(x)
            r2 = find(y)
            
            if r1 != r2:
                groupCounts[counts[r1]] -= 1
                groupCounts[counts[r2]] -= 1
                counts[r1] = counts[r2] = counts[r1] + counts[r2]
                groupCounts[counts[r1]] += 1
                
                if ranks[r1] >= ranks[r2]:
                    parents[r2] = r1
                    ranks[r1] += 1
                else:
                    parents[r1] = r2
                    ranks[r2] += 1
        
        last = -1
        
        for step, index in enumerate(arr):
            index -= 1
            groupCounts[1] += 1
            if index-1 >= 0 and visited[index-1]:
                union(index, index-1)
            
            if index+1 < n and visited[index+1]:
                union(index, index+1)
            
            visited[index] = True
            
            if groupCounts[m]:
                last = step + 1

        return last

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        n = len(arr)
        parents = list(range(n))
        ranks = [0] * n
        counts = [1] * n
        groupCounts = [0] * (n+1)
        cur_string = [0] * n
        
        def find(x):
            if parents[x] != x:
                parents[x] = find(parents[x])
            
            return parents[x]
        
        def union(x,y):
            r1 = find(x)
            r2 = find(y)
            
            groupCounts[counts[r1]] -= 1
            groupCounts[counts[r2]] -= 1
            counts[r1] = counts[r2] = counts[r1] + counts[r2]
            groupCounts[counts[r1]] += 1
            
            if r1 != r2:
                if ranks[r1] >= ranks[r2]:
                    parents[r2] = r1
                    ranks[r1] += 1
                else:
                    parents[r1] = r2
                    ranks[r2] += 1
        
        ans = -1
        
        for step, i in enumerate(arr):
            i -= 1
            groupCounts[1] += 1
            
            if i - 1 >= 0 and cur_string[i-1] == 1:
                union(i, i-1)
            
            if i + 1 < n and cur_string[i+1] == 1:
                union(i, i+1)
            
            cur_string[i] = 1
            
            if groupCounts[m]:
                ans = step + 1
        
        return ans

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        comps = [0 for _ in range(len(arr) + 2)]
        size = collections.defaultdict(int)
        ans = -1
        for i, n in enumerate(arr):
            left, right = comps[n - 1], comps[n + 1]
            comps[n - left] = comps[n + right] = comps[n] = left + right + 1
            # print(comps)
            
            size[comps[n]] += 1
            size[left] -= 1
            size[right] -= 1
            if size[m] > 0:
                ans = i + 1
        return ans

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        n = len(arr)
        dp = [[0,0] for _ in range(n+2)]
        num = [0]*(n+2)
        d = defaultdict(int)
        ans = -1
        for i in range(1,n+1):
            x = arr[i-1]
            num[x] = 1
            l = dp[x-1][1]
            r = dp[x+1][0]

            # print(i,l,r)
            d[l] -= 1
            d[r] -= 1
            d[dp[x-1][0]] -= 1
            d[dp[x+1][1]] -= 1
            dp[x-1] = [0,0]
            dp[x+1] = [0,0]
            d[0] += 4

            d[dp[x-l][0]] -= 1
            d[dp[x+r][1]] -= 1
            dp[x-l][0] = r+l+1
            dp[x+r][1] = r+l+1
            d[r+l+1] += 2
#             if num[x+1] == 1 and num[x-1] == 1:
#                 l = dp[x-1][1]
#                 r = dp[x+1][0]
                
#                 # print(i,l,r)
#                 d[l] -= 1
#                 d[r] -= 1
#                 d[dp[x-1][0]] -= 1
#                 d[dp[x+1][1]] -= 1
#                 dp[x-1] = [0,0]
#                 dp[x+1] = [0,0]
#                 d[0] += 4
                
#                 d[dp[x-l][0]] -= 1
#                 d[dp[x+r][1]] -= 1
#                 dp[x-l][0] = r+l+1
#                 dp[x+r][1] = r+l+1
#                 d[r+l+1] += 2
#             elif num[x+1] == 1:
#                 r = dp[x+1][0]
                
#                 d[r] -= 1
#                 d[dp[x+1][1]] -= 1
#                 dp[x+1] = [0,0]
#                 d[0] += 2
                
#                 d[dp[x-l][0]] -= 1
#                 d[dp[x+r][1]] -= 1
#                 dp[x-l][0] = r+l+1
#                 dp[x+r][1] = r+l+1
#                 d[r+l+1] += 2
#             elif num[x-1] == 1:
#                 l = dp[x-1][1]
                
#                 d[l] -= 1
#                 d[dp[x-1][0]] -= 1
#                 dp[x][1] = dp[x-1][1]+1
#                 dp[x-1] = [0,0]
#                 d[0] += 2
                
#                 d[dp[x-l][0]] -= 1
#                 dp[x-l][0] = l+1
#                 d[l+1] += 1
                
#             else:
#                 dp[x] = [1,1]
#                 d[1] += 2
            
            # print(num)
            # print(dp)
            # print(d)
            
            if d[m] != 0:
                ans = i
        return ans
            

class DSU:
    def __init__(self):
        self.p = {}
        self.size = {}
        self.size_to_node = collections.defaultdict(set)

    def exists(self, x): return x in self.p
    
    def size_exists(self, x): return len(self.size_to_node[x]) > 0
    
    def len(self, x):
        if self.exists(x): return self.size[self.find(x)]
        else:              return -1

    def make_set(self, x):
        self.p[x] = x
        self.size[x] = 1
        self.size_to_node[1].add(x)
        
    def find(self, x):
        if not self.exists(x): return None
        
        if self.p[x] != x:
            self.p[x] = self.find(self.p[x])

        return self.p[x]

    def union(self, x, y):
        xr = self.find(x)
        yr = self.find(y)
        
        if xr is None or yr is None: return
        
        self.p[xr] = yr
        
        self.size_to_node[self.size[yr]].remove(yr)
        self.size_to_node[self.size[xr]].remove(xr)
        
        self.size[yr] += self.size[xr]
        self.size_to_node[self.size[yr]].add(yr)
        
        del self.size[xr]

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        dsu = DSU()
        latest_group = -1
        
        for i, x in enumerate(arr):
            dsu.make_set(x)
            dsu.union(x, x-1)
            dsu.union(x, x+1)
            
            if dsu.size_exists(m):
                latest_group = i+1
        
        return latest_group
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        def find_set(x):
            if parents[x][0] == x:
                return x
            else:
                return find_set(parents[x][0])


        def union_set(x, y):
            # print(x, y)
            x_root = find_set(x)
            y_root = find_set(y)
            # print(x_root, y_root)
            # print(parents[x_root], parents[y_root])
            parents[y_root][1] += parents[x_root][1]
            parents[x_root][0] = y_root


        n = len(arr)
        parents = [[i, 1] for i in range(n)]
        # print(parents)
        visited = [False for i in range(n)]
        answer = -1
        d = {}
        for i in range(n):
            num = arr[i] - 1
            visited[num] = True
            # if num > 0 and visited[num - 1]:
            #     d[parents[num - 1][1]] -= 1
            # if num + 1 < n and visited[num + 1]:
            #     d[parents[num + 1][1]] -= 1
            if num > 0 and visited[num - 1]:
                d[parents[find_set(num - 1)][1]] -= 1
                # print(parents)
                union_set(num - 1, num)
                # print(parents)
            if num + 1 < n and visited[num + 1]:
                d[parents[find_set(num + 1)][1]] -= 1
                # print(parents)
                union_set(num + 1, num)
                # print(parents)
            d[parents[num][1]] = 1 if parents[num][1] not in d else d[parents[num][1]] + 1
            if m in d and d[m] > 0:
                # print(i + 1, num + 1, d)
                answer = i + 1
        return answer
class DSU:
    def __init__(self):
        self.p = {}
        self.r = {}
        self.count = collections.Counter()

    def add(self, x: int):
        self.p[x] = x
        self.r[x] = 1
        self.count[1] += 1
        
    def parent(self, x: int) -> int:
        if self.p[x] != x:
            self.p[x] = self.parent(self.p[x])
        return self.p[x]
    
    def unite(self, x: int, y: int) -> int:
        x = self.parent(x)
        y = self.parent(y)
        if x == y:
            return self.r[x]
        if self.r[x] > self.r[y]:
            x, y = y, x
        self.count[self.r[x]] -= 1
        self.count[self.r[y]] -= 1
        self.count[self.r[x] + self.r[y]] += 1
        self.p[x] = y
        self.r[y] += self.r[x]
        return self.r[y]
        
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        n = len(arr)
        dsu = DSU()
        bits = [0] * (n + 2) 
        ans = -1
        
        for i, bit in enumerate(arr, 1):
            dsu.add(bit)
            bits[bit] = 1
            if bits[bit - 1] == 1:
                dsu.unite(bit, bit - 1)
            if bits[bit + 1] == 1:
                dsu.unite(bit, bit + 1)
            if dsu.count[m] > 0:
                ans = i
            
        return ans

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.number_of_groups = n
        
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        x_parent, y_parent = self.find(x), self.find(y)
        if x_parent == y_parent:
            return False
        if self.rank[x_parent] < self.rank[y_parent]:
            x_parent, y_parent = y_parent, x_parent
            
        self.rank[x_parent] += self.rank[y_parent]
        
        self.parent[y_parent] = x_parent
        
        self.number_of_groups -= 1
        return True

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        n = len(arr)
        ans = -1
        uf = UnionFind(n)
        
        for i, x in enumerate(arr):
            x -= 1
            uf.rank[x] = 1
            for j in [x+1, x-1]:
                if 0 <= j < n:
                    if uf.rank[uf.find(j)] == m:
                        ans = i
                    if uf.rank[j]:
                        uf.union(x, j)
        
        if any(uf.rank[uf.find(i)]==m for i in range(n)):
            ans = n
            
        return ans

_ROOT = object()
class UnionSet:
    def __init__(self):
        self.parents = {}
        self.sizes = {}
        self.size_histogram = collections.defaultdict(int)
    
    def has_size(self, size):
        return self.size_histogram[size] != 0
    
    def add(self, x):
        self.parents[x] = _ROOT
        self.sizes[x] = 1
        self.size_histogram[1] += 1

        self.merge(x, x-1)
        self.merge(x, x+1)
        
    def root(self, x):
        path = []
        while self.parents[x] != _ROOT:
            path.append(x)
            x = self.parents[x]
        for y in path:
            self.parents[y] = x
        return x
    
    def merge(self, x, y):
        if y not in self.parents:
            return
        x = self.root(x)
        y = self.root(y)

        self.size_histogram[self.sizes[x]] -= 1
        self.size_histogram[self.sizes[y]] -= 1
        merged_size = self.sizes[x] + self.sizes[y]
        self.size_histogram[merged_size] += 1
        if self.sizes[x] < self.sizes[y]:
            del self.sizes[x]
            self.parents[x] = y
            self.sizes[y] = merged_size
        else:
            del self.sizes[y]
            self.parents[y] = x
            self.sizes[x] = merged_size
            

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        union_set = UnionSet()
        latest_step = -1
        for step, pos in enumerate(arr):
            union_set.add(pos)
            if union_set.has_size(m):
                latest_step = step + 1
        return latest_step
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        n = len(arr)
        groups = defaultdict(set)
        parents = [i for i in range(n)]
        size = [0] * n
        
        def find(node):
            if parents[node] == node:
                return node
            parent = find(parents[node])
            return parent
        
        def union(a, b):
            para = find(a)
            parb = find(b)
            if para != parb:
                groups[parb].update(groups[para])
                groups.pop(para)
                parents[para] = parb
                
        def get_size(a):
            parent = find(parents[a])
            return len(groups[parent])
        
        def update(i):
            check = get_size(i)
            sizes[check] -= 1
            if sizes[check] == 0:
                sizes.pop(check)
        
        arr = [i-1 for i in arr]
        step = 0
        ans = -1
        sizes = Counter()
        for i in arr:
            step += 1
            size[i] += 1
            groups[i].add(i)
            sizes[1] += 1
            if i-1 >= 0 and i+1 < n and size[i-1] and size[i+1]:
                update(i-1)
                update(i+1)
                update(i)
                union(i, i-1)
                union(i+1, i-1)
                new_size = get_size(i-1)
                sizes[new_size] += 1
            elif i-1 >= 0 and size[i-1]:
                update(i-1)
                update(i)
                union(i, i-1)
                new_size = get_size(i-1)
                sizes[new_size] += 1
            elif i+1 < n and size[i+1]:
                update(i+1)
                update(i)
                union(i, i+1)
                new_size = get_size(i+1)
                sizes[new_size] += 1
            if m in sizes:
                ans = step
        return ans
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        n = len(arr)
        if m == n:
            return n
        uf = UnionFindHelper()
        res = -1
        for i, curr in enumerate(arr):
            uf.add(curr)
            step = i + 1
            for neighbor in [curr - 1, curr + 1]:
                if uf.contains(neighbor):
                    if uf.getrank(neighbor) == m:
                        res = step - 1
                    uf.union(neighbor, curr)
            # if uf.getrank(curr) == m:
            #     res = step
        return res
        
class UnionFindHelper:
    def __init__(self):
        self.parent = {}
        self.ranks = {}
        self.count = 0
        
    def contains(self, item):
        return item in self.parent
    
    def add(self, item):
        if item not in self.parent:
            self.parent[item] = item
            self.ranks[item] = 1
            self.count += 1
    
    def getrank(self, item):
        return self.ranks[self.find(item)]
    
    def find(self, item):
        if item != self.parent[item]:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]
    
    def union(self, item1, item2):
        item1 = self.find(item1)
        item2 = self.find(item2)
        rank1 = self.ranks[item1]
        rank2 = self.ranks[item2]
        if item1 != item2:
            self.parent[item1] = item2
            self.ranks[item2] = rank1 + rank2
            self.count -= 1
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        n = len(arr)
        if n == m:
            return n
        
        self.parent = list(range(n + 1))
        self.size = [0] * (n + 1)
        
        res = -1
        for i, num in enumerate(arr):
            if self.size[num] == 0:
                self.size[num] = 1
            if num - 1 >= 1 and self.size[num - 1] >= 1:
                if self.size[self.find(num - 1)] == m: res = i
                self.union(num - 1, num)
            if num + 1 <= n and self.size[num + 1] >= 1:
                if self.size[self.find(num + 1)] == m: res = i
                self.union(num, num + 1)
            # print(i, self.size, self.parent, res)
        return res
        
    def union(self, a, b):
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a != root_b:
            self.parent[root_a] = root_b
            self.size[root_b] += self.size[root_a]
    
    def find(self, a):
        curr = a
        while self.parent[curr] != curr:
            curr = self.parent[curr]
        curr, root = a, curr
        while self.parent[curr] != curr:
            self.parent[curr], curr = root, self.parent[curr]
        return root

class DSU:
  def __init__(self, m):
    self.reps = {}
    self.size = {}
    self.m = m
    self.count = 0
  def add(self, x):
    self.reps[x] = x
    self.size[x] = 1
    if self.m == 1:
        self.count += 1
  def find(self, x):
    if not x == self.reps[x]:
      self.reps[x] = self.find(self.reps[x])
    return self.reps[x]
  def union(self, x, y):
    hX = self.find(x)
    hY = self.find(y)
    if not hX == hY:
      h = min(hX, hY)
      if self.size[hX] == self.m:
        self.count -= 1
      if self.size[hY] == self.m:
        self.count -= 1
      if h == hX:
        self.reps[hY] = h
        self.size[hX] += self.size[hY]
        if self.size[hX] == self.m:
          self.count += 1
        self.size.pop(hY)
      else:
        self.reps[hX] = h
        self.size[hY] += self.size[hX]
        if self.size[hY] == self.m:
          self.count += 1
        self.size.pop(hX)

class Solution:
  def findLatestStep(self, arr: List[int], m: int) -> int:
    # dsu
    dsu, s = DSU(m = m), -1
    for i, x in enumerate(arr):
      dsu.add(x)
      if x - 1 in dsu.reps:
        dsu.union(x - 1, x)
      if x + 1 in dsu.reps:
        dsu.union(x + 1, x)
      if dsu.count > 0:
        s = max(s, i + 1)
    return s
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        comps = [0 for _ in range(len(arr) + 2)]
        size = collections.defaultdict(int)
        ans = -1
        for i, n in enumerate(arr):
            left, right = comps[n - 1], comps[n + 1]
            comps[n - left] = comps[n + right] = comps[n] = left + right + 1
            size[comps[n]] += 1
            size[left] -= 1
            size[right] -= 1
            if size[m] > 0:
                ans = i + 1
        return ans

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        if m==1 and len(arr)==1: return(1)
        if m>len(arr):return(-1)
        Bins=[0]*len(arr)
        Gs={}#start,len
        Ge={}#end,start ix
        Res=-1
        Rescnt=0
        for y in range(len(arr)):
            i=arr[y]-1
            if (i==0 or Bins[i-1]==0) and (i==len(arr)-1 or Bins[i+1]==0):
                #new group
                # Gl[i]=1
                # Gs[i]=i
                Gs.update({i:1})
                Ge.update({i:i})
                Bins[i]=1
                # ix=len(Gs)-1 
                if m==1:
                    Res=y+1
                    Rescnt=Rescnt+1
                    # print(Res,Rescnt)
            else:
                if (i==0 or Bins[i-1]==0) and (i<len(arr)-1 and Bins[i+1]==1):
                    #extend left i+1
                    # ix=[k for k,j in enumerate(Gs) if j[0]==(i+1)][0]
                    Gs.update({i:Gs[i+1]+1})
                    Ge[Gs[i+1] + i] = i
                    tmp=Gs.pop(i+1)
                    # tmpe=Ge.pop(i+1)
                    Bins[i]=1
                    if Gs[i]==m:
                        Res=y+1
                        Rescnt=Rescnt+1
                    else: 
                        if tmp==m: Rescnt-=1
                else:
                    if (i>0 and Bins[i-1]==1) and (i==len(arr)-1 or Bins[i+1]==0):
                        #extend right i-1
                        # strt=i-1
                        # while(strt>0 and Bins[strt]==1):strt-=1
                        # if(Bins[strt]==0):strt+=1
                        
                        # ix=[k for k in Gs.keys() if k+Gs[k]==i][0]
                        # ix=[k for k,j in enumerate(Gs) if j[0]==strt][0]
                        ix=Ge[i-1]
                        tmp=Gs[ix]
                        Gs[ix]=Gs[ix]+1 
                        tmpe=Ge.pop(i-1)
                        Ge.update({i:ix})
                        Bins[i]=1
                        # if(Gs[ix][1]==m):Res=i
                        if Gs[ix]==m:
                            Res=y+1
                            Rescnt=Rescnt+1
                        else: 
                            if tmp==m: Rescnt-=1
                    else:
                        if (i>0 and Bins[i-1]==1) and (i<len(arr)-1 and Bins[i+1]==1):
                            #merge Len=Sum+1

                            # ix=[k for k in Gs.keys() if k+Gs[k]==i][0]
                            # print('merge',i)
                            ix=Ge[i-1]
                            # print('ix',ix,i)
                            tmp0=Gs[ix]
                            Gs[ix]=Gs[ix]+Gs[i+1]+1
                            tmp=Gs.pop(i+1)
                            tmpe=Ge.pop(i-1)
                            # print('ix2',tmp+i)
                            Ge[tmp+i]=ix
                            Bins[i]=1
                            if Gs[ix]==m:
                                Res=y+1
                                Rescnt=Rescnt+1
                            else: 
                                if tmp==m: Rescnt-=1
                                if tmp0==m: Rescnt-=1                                
            if Rescnt>0: Res=y+1
            # print(i,'E',Ge)
            # print('S',Gs)
            
            # for r in range(len(Gs)):
            #         if Gs[r][1]==m:
            #             Res=y+1
            #             break                    
        return(Res)
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        count = collections.Counter()
        length = collections.defaultdict(int)
        ans = -1
        for i, a in enumerate(arr):
            left, right = length[a - 1], length[a + 1]
            length[a] = length[a - left] = length[a + right] = left + right + 1
            count[left] -= 1
            count[right] -= 1
            count[length[a]] += 1
            if count[m]:
                ans = i + 1
        return ans

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        parent = [i for i in range(len(arr))]
                
        def union(a, b):
            par_a = find(a)
            par_b = find(b)
            if par_a == par_b:
                return
                
            if(par_a > par_b):
                par = par_a
            else:
                par = par_b
                
            parent[par_a] = par
            parent[par_b] = par
        def find(a):
            if parent[a] == a:
                return a
            else :
                parent[a] = find(parent[a])
                return parent[a]
        
        ans = -1 
        contigAtI = [0] * len(arr)
        counts = [0] * (len(arr) + 1)
        s = [0] * len(arr)
        for idx, item in enumerate(arr):
            s[item-1] = 1
            
            contigAtI[item-1]= 1 
            counts[contigAtI[item-1]]+=1
            
            if item - 2 >= 0 and s[item-2] == 1:
                counts[contigAtI[item-2]]-=1
                counts[contigAtI[item-1]]-=1
                contigAtI[item-1]+= contigAtI[item-2]
                counts[contigAtI[item-1]]+=1
                
                union(item-2, item-1)
                                
            if item < len(arr) and s[item] == 1:
                endPtr = find(item)
                
                counts[contigAtI[item-1]]-=1
                counts[contigAtI[endPtr]]-=1
                
                contigAtI[endPtr] += contigAtI[item-1]
                
                counts[contigAtI[endPtr]]+=1
                
                
                union(item-1, endPtr)
            if counts[m] > 0:
                ans = max(ans, idx+1)
            
        return ans
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        
        f = {}
        n = len(arr)
        ans = -1
        ranks = [0] * (n+1)
        
        if m == n:
            return m
        
        def find(x):
            f.setdefault(x,x)
            if f[x] != x:
                f[x] = find(f[x])
            
            return f[x]
        
        def union(x,y):
            px,py = find(x), find(y)
            
            if ranks[px] > ranks[py]:
                ranks[px] += ranks[py]
                f[py] = px
            else:
                ranks[py] += ranks[px]
                f[px] = py

            
        for i,a in enumerate(arr):
            ranks[a] = 1
            
            for j in [a-1,a+1]:
                if 1<= j <= n:
                    if ranks[find(j)] == m:
                        ans = i
                    if ranks[j]:
                        union(a,j)
        
        return ans

class Subset:
    
    def __init__(self,parent,rank):
        self.parent=parent
        self.rank=rank
        
def find(subsets,node):
    if subsets[node].parent!=node:
        subsets[node].parent=find(subsets,subsets[node].parent)
    return subsets[node].parent

def union(subsets,x,y):
    
    xr=find(subsets,x)
    yr=find(subsets,y)
    if xr==yr:
        return True
    else:
        xr=subsets[xr]
        yr=subsets[yr]
        
        if xr.rank<yr.rank:
            xr.parent=yr.parent
            yr.rank+=xr.rank
        elif xr.rank>yr.rank:
            yr.parent=xr.parent
            xr.rank+=yr.rank
        else:
            xr.parent=yr.parent
            yr.rank=2*yr.rank
            
        return False
    
class Solution:
    
    def findLatestStep(self, arr: List[int], m: int) -> int:
        
        
        
        a=[0 for ii in range(len(arr))]
        subsets=[Subset(i,1) for i in range(len(arr))]
        groups=set()
        ans=-1
        for j in range(len(arr)):
            i=arr[j]-1
            p=find(subsets,i)
            a[i]=1
            if i-1>=0 and a[i-1]==1:
                if find(subsets,i-1) in groups:
                   groups.remove(find(subsets,i-1))
                union(subsets,i-1,i)
            if i+1<=len(arr)-1 and a[i+1]==1:
                if find(subsets,i+1) in groups:
                    groups.remove(find(subsets,i+1))
                union(subsets,i+1,i)
            if subsets[find(subsets,i)].rank==m:
                groups.add(find(subsets,i))
            if subsets[find(subsets,i)].rank!=m and find(subsets,i) in groups:
                groups.remove(find(subsets,i))
            if len(groups):
                ans=j+1
            
        return ans
            
            

class UFDS:
    def __init__(self, n):
        self.parents = list(range(n))
        self.ranks = [0] * n
        self.sizes = [1] * n
        self.numdisjoint = n

    def find(self, x):
        xp = x
        children = []
        while xp != self.parents[xp]:
            children.append(xp)
            xp = self.parents[xp]
        for c in children:
            self.parents[c] = xp
        return xp

    def union(self, a, b):
        ap = self.find(a)
        bp = self.find(b)
        if ap == bp:
            return

        if self.ranks[ap] < self.ranks[bp]:
            self.parents[ap] = bp
            self.sizes[bp] += self.sizes[ap]
        elif self.ranks[bp] < self.ranks[ap]:
            self.parents[bp] = ap
            self.sizes[ap] += self.sizes[bp]
        else:
            self.parents[bp] = ap
            self.ranks[ap] += 1
            self.sizes[ap] += self.sizes[bp]

        self.numdisjoint -= 1

    def size(self, x):
        return self.sizes[self.find(x)]

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        s = set()
        res = -1
        d = [0] * (len(arr) + 1)
        arr = [i-1 for i in arr]
        u = UFDS(len(arr))
        for i, val in enumerate(arr):
            s.add(val)
            if (val > 0) and (val-1 in s):
                d[u.size(val-1)] -= 1
                u.union(val-1, val)
            if (val < len(arr)-1) and (val+1 in s):
                d[u.size(val+1)] -= 1
                u.union(val, val+1)
            d[u.size(val)] += 1
            #print(d)
            if d[m] >= 1:
                res = max(res, i+1)
        #print("------------")
        return res
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        N = len(arr)
        parent = [i for i in range(N)]
        rank = [0] * N
        size = [1] * N
        groupCount = Counter()
        visited = set()
        ans = -1
        
        for i, num in enumerate(arr, 1):
            num -= 1
            visited.add(num)
            groupCount[1] += 1
            
            if num - 1 in visited:
                self.union(parent, rank, size, groupCount, num, num - 1)
                
            if num + 1 in visited:
                self.union(parent, rank, size, groupCount, num, num + 1)
                
            if groupCount[m] > 0:
                ans = i
                
        return ans
            
    def find(self, parent, x):
        if parent[x] != x:
            parent[x] = self.find(parent, parent[x])

        return parent[x]

    def union(self, parent, rank, size, groupCount, x, y):
        xRoot = self.find(parent, x)
        yRoot = self.find(parent, y)
        groupCount[size[xRoot]] -= 1
        groupCount[size[yRoot]] -= 1
        size[xRoot] = size[yRoot] = size[xRoot] + size[yRoot]
        groupCount[size[xRoot]] += 1

        if rank[xRoot] > rank[yRoot]:
            parent[yRoot] = xRoot
        elif rank[xRoot] < rank[yRoot]:
            parent[xRoot] = yRoot
        else:
            parent[yRoot] = xRoot
            rank[xRoot] += 1
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        d = {}
        sizes = {}
        last_seen = -1
        def find(x):
            if d[x] != x:
                d[x] = find(d[x])
            return d[x]
        for i in range(len(arr)):
            I = arr[i]
            d[I] = I
            size = 1
            if I - 1 in d:
                if sizes[find(I - 1)] == m:
                    last_seen = i
                d[find(I)] = I - 1
                sizes[find(I)] += size
                size = sizes[find(I)]
            if I + 1 in d:
                if sizes[find(I + 1)] == m:
                    last_seen = i
                d[find(I)] = I + 1
                sizes[find(I)] += size
                size = sizes[find(I)]
            sizes[find(I)] = size
            if size == m:
                last_seen = i + 1
        return last_seen
            

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        N = len(arr)
        arr = [x - 1 for x in arr]
        parent = [x for x in range(N)]
        size = [1 for _ in range(N)]
        used = [False for _ in range(N)]
        
        def ufind(a):
            if a == parent[a]:
                return a
            parent[a] = ufind(parent[a])
            return parent[a]
        
        def uunion(a, b):
            sa = ufind(a)
            sb = ufind(b)
            
            if sa != sb:
                parent[sa] = parent[sb]
                size[sb] += size[sa]
                
        def usize(a):
            return size[ufind(a)]
        
        counts = [0] * (N + 1)
        latest = -1
        print('arr', arr)
        
        for index, x in enumerate(arr):
            left = 0
            if x - 1 >= 0 and used[x - 1]:
                left = usize(x - 1)
            right = 0
            if x + 1 < N and used[x + 1]:
                right = usize(x + 1)
            current = 1
            counts[1] += 1
            if left > 0:
                counts[left] -= 1
            if right > 0:
                counts[right] -= 1
            counts[1] -= 1
            used[x] = True
            new_size = left + right + current
            counts[new_size] += 1
            if left > 0:
                uunion(x, x - 1)
            if right > 0:
                uunion(x, x + 1)
            if counts[m] > 0:
                latest = max(latest, index + 1)
                
        return latest
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        length = [0]*(len(arr)+2)
        count = [0]*(len(arr)+1)
        ans = 0
        for step, index in enumerate(arr):
            left = length[index-1]
            right = length[index+1]
            length[index-left] = length[index+right] = right + left + 1
            count[left] -= 1
            count[right] -= 1
            count[right+left+1] += 1
            if count[m] > 0:
                ans = step+1 
        if ans == 0:
            return -1
        return ans
                
                

class Solution:
    def findLatestStep(self, A: List[int], m: int) -> int:
        n = len(A)
        if m == n: return n
        dic = {0:[0, 0]}
        def union(a, b):
            fa = find(a)
            fb = find(b)
            if fa != fb:
                dic[fa[0]] = [fb[0], fa[1] + fb[1]]
                dic[fb[0]] = dic[fa[0]]
                dic[a] = dic[fa[0]]
                dic[b] = dic[fa[0]]
        
        def find(a):
            if dic[a][0] != a:
                dic[a] = find(dic[a][0])
            return dic[a]
                
        
        ans, t, ret = set(), 0, -1
        for i, v in enumerate(A):
            t = t | (1 << (v - 1))
            dic[v] = [v, 1]
            if v - 1 in dic:
                if find(v - 1)[1] == m:
                    ans.remove(dic[v - 1][0])
                union(v, v - 1)
            if v + 1 in dic:
                if find(v + 1)[1] == m:
                    ans.remove(dic[v + 1][0])
                union(v, v + 1)
            if dic[v][1] == m:
                ans.add(dic[v][0])
            if ans: ret = i + 1
            # print(dic, ans)        
        return ret
class dsu:
    def __init__(self,n):
        self.par=[i for i in range(n)]
        self.len=[1]*n
        self.size=[0]*n
        self.store=[0]*(n+1)

    def unio(self,a,b):
        a=self.find(a)
        b=self.find(b)
        
        if self.len[a]>self.len[b]:
            self.par[b]=a
            self.size[a]+=self.size[b]
            
        elif self.len[a]<self.len[b]:
            self.par[a]=b
            self.size[b]+=self.size[a]
        else:
            self.par[b]=a
            self.len[a]+=1
            self.size[a]+=self.size[b]

    def find(self,a):
        if(a!=self.par[a]):
            self.par[a]=self.find(self.par[a])
        return self.par[a]

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        ds=dsu(len(arr))
        n=len(arr)
        t=[0]*n
        ans=-1
        tpp=0
        for i in arr:
            tpp+=1
            curr=i-1
            t[curr]=1
            if(ds.size[curr]==0):
                ds.size[curr]=1
            flag=0
            if(curr>=1 and t[curr-1]==1):
                jm=ds.find(curr-1)
                ds.store[ds.size[jm]]-=1
                flag=1
            if(curr<(n-1) and t[curr+1]==1):
                jm=ds.find(curr+1)
                ds.store[ds.size[jm]]-=1
                flag=1
            # if(flag):
            #     ds.store[1]-=1
            if(curr>=1 and t[curr-1]==1):
                ds.unio(curr,curr-1)                
            if(curr<(n-1) and t[curr+1]==1):
                ds.unio(curr,curr+1)
            jm=ds.find(curr)
            ds.store[ds.size[jm]]+=1
            # print(ds.store)
            if(ds.store[m]):
                ans=tpp
                
        return ans

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        n = len(arr)
        if n == m: #u6700u540eu4e00u6b65u624du51fau73b0u6ee1u8db3u6761u4ef6
            return m
        
        A = [i for i in range(n)]
        length = [0 for _ in range(n)]
        ans = -1
        def find(u):
            if u != A[u]:
                A[u] = find(A[u])
            return A[u]
        
        def union(u, v):
            pu, pv = find(u), find(v)
            if pu == pv:
                return False
            
            A[max(pu, pv)] = min(pu, pv)
            #u53eau4feeu6539u65b0u7236u8282u70b9u7684u957fu5ea6uff0cu8282u7701u65f6u95f4u3002u5176u4ed6u7684u7528u4e0du5230
            length[min(pu,pv)] += length[max(pu,pv)]
            
        for i, a in enumerate(arr):
            a -= 1
            length[a] = 1
            for j in [a-1, a+1]:
                #u67e5u627eu4e24u8fb9u662fu5426u662f1uff0cu5982u679cu662f1uff0cu5219u8054u5408u8d77u6765
                if 0 <= j < n:
                    #u5982u679cju4f4du7f6eu7684u957fu5ea6u662fmuff0cu5219u8bb0u5f55u4e0au4e00u6b65u662fu6ee1u8db3u6761u4ef6u7684(u6700u540eu4e00u6b65u65e0u6cd5u8bb0u5f55)
                    if length[find(j)] == m:
                        ans = i
                    if length[j]:
                        union(j, a)
                        # print(length)

        # print(length)
                        
        return ans
class DS:
    def __init__(self, n):
        self.par = list(range(n))
        self.rank = [1] * n
        
    def find(self, x):
        if self.par[x] != x:
            self.par[x] = self.find(self.par[x])
        return self.par[x]
    
    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        
        if px == py:
            return
        self.par[px] = py
        self.rank[py] += self.rank[px]
    
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        ds = DS(len(arr))
        ans = -1
        b_arr = [0] * len(arr)
        
        for i, num in enumerate(arr):
            idx = num-1
            b_arr[idx] = 1
            
            if idx > 0 and b_arr[idx-1]:
                p = ds.find(idx-1)
                if ds.rank[p] == m:
                    ans = i
                ds.union(idx, idx-1)
                
            if idx < len(arr)-1 and b_arr[idx+1]:
                p = ds.find(idx+1)
                if ds.rank[p] == m:
                    ans = i
                ds.union(idx, idx+1)
                
            p = ds.find(idx)
            if ds.rank[p] == m:
                ans = i+1
        return ans
class UnionFindSet:
    def __init__(self, n):
        self.parent = [i for i in range(n)]
        self.rank = [0] * n
        
    def find(self, u):
        if u != self.parent[u]:
            self.parent[u] = self.find(self.parent[u])
        return self.parent[u]
    
    def union(self, u, v):
        pu = self.find(u)
        pv = self.find(v)
        if pv == pu: return False
        if self.rank[pu] < self.rank[pv]:
            self.rank[pv] += self.rank[pu]
            self.parent[pu] = pv
        else:
            self.rank[pu] += self.rank[pv]
            self.parent[pv] = pu
        return True
            

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        n = len(arr)
        ans = -1
        uf = UnionFindSet(n)
        
        for step, i in enumerate(arr):
            i -= 1
            uf.rank[i] = 1
            for j in (i-1, i+1):
                if  0 <= j < n:
                    if uf.rank[uf.find(j)] == m:
                        ans = step
                    print(( uf.rank[uf.find(j)]))
                    if uf.rank[j]:
                        uf.union(i, j)
                    
                    
                        
        for i in range(n):
            if uf.rank[uf.find(i)] == m:
                return n
        return ans
        

class union(object):
    def __init__(self,n):
        self.parent = [-1]*n
        self.size = [0] *n
        self.count = collections.defaultdict(int)
    def find(self,p):
        if self.parent[p] == -1:
            return -1
        if self.parent[p] != p:
            self.parent[p] = self.find(self.parent[p])
        return self.parent[p]
    def union(self,p, q):
        if self.find(p) != self.find(q):
            pr = self.find(p)
            qr = self.find(q)
            
            #print(pr,qr)
            #print(self.count)
            #print(self.size)
            self.count[self.size[pr]] -= 1
            self.count[self.size[qr]] -= 1
            self.parent[pr] = qr
            self.size[qr] += self.size[pr]
            self.count[self.size[qr]] += 1
            
            #print(pr,qr)
            #print(self.count)
            #print(self.size)
            
            
    def add(self,p):
        self.parent[p] = p
        self.size[p] = 1
        self.count[1] += 1
    def get_size(self , m):
        
        return self.count[m]>0
    

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        if   m == len(arr):
            return m
        uds = union(len(arr) + 1)       
        ans = -1
        for i in range(len(arr)):
            uds.add(arr[i]  ) 
            if arr[i] + 1 <= len(arr)  and uds.find(arr[i] +1  ) != -1:
                uds.union(arr[i],arr[i] + 1)
            if arr[i] - 1 >= 1 and uds.find(arr[i] - 1  ) != -1:
                uds.union(arr[i] - 1,arr[i]  ) 
            if     uds.get_size( m)  :
                ans = i + 1
             
            #print(uds.parent)
            #print(uds.size)
            #print(uds.count)
        return ans
            
        

class UnionFindSet:
    def __init__(self, n):
        self.parents = list(range(n))
        self.ranks = [0] * n
        
    def find(self, u):
        if u != self.parents[u]:
            self.parents[u] = self.find(self.parents[u])
            
        return self.parents[u]
    def union(self, u, v):
        pu, pv = self.find(u), self.find(v)
        
        if pu == pv:
            return False
        
        if self.ranks[pu] > self.ranks[pv]:
            self.parents[pv] = pu
            self.ranks[pu] += self.ranks[pv]
        else:
            self.parents[pu] = pv
            self.ranks[pv] += self.ranks[pu]
        return True
    
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        n, ans = len(arr), -1
        uf = UnionFindSet(n)

        for step, i in enumerate(arr):
            i -= 1
            uf.ranks[i] = 1

            for j in (i - 1, i + 1):
                if 0 <= j < n:
                    if uf.ranks[uf.find(j)] == m:
                        ans = step
                    if uf.ranks[j]:
                        uf.union(i, j)

        for i in range(n):
            if uf.ranks[uf.find(i)] == m:
                return n

        return ans

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        uf = {}
        n, ans = len(arr), -1
        seen = [0] * (n + 1)

        def find(x):
            uf.setdefault(x, x)
            if uf[x] != x:
                uf[x] = find(uf[x])
            return uf[x]

        def union(x, y):
            seen[find(y)] += seen[find(x)]
            uf[find(x)] = find(y)
        
        if m == n:
            return n
        for i, a in enumerate(arr):
            seen[a] = 1
            for b in [a - 1, a + 1]:
                if 1 <= b <= n and seen[b]:
                    if seen[find(b)] == m:
                        ans = i
                    union(a, b)
        return ans
                
                
                
                
                

class Solution:
   def findLatestStep(self, A, m):
        starts ={}
        ends ={}
        res = -1
        count = 0
        for i, a in enumerate(A):
            if a-1 in ends and a+1 in starts:
                end = starts.pop(a+1,None)
                if (end - (a+1) +1) == m: count-=1
                    
                st = ends.pop(a-1,None)
                if ((a-1)-st+1)==m: count-=1
                
                starts[st] = end
                ends[end] = st
                if(end-st+1) ==m: count+=1
                
            elif a-1 in ends:
                st = ends.pop(a-1, None)
                if ((a-1)-st+1)==m: count-=1
                    
                if a-1 in starts:
                    starts.pop(a-1,None)
                    
                ends[a]=st
                starts[st]=a
                if(a-st+1) ==m: count+=1
                
            elif a+1 in starts:
                end = starts.pop(a+1, None)
                if (end - (a+1) +1) == m: count-=1
                
                if a+1 in ends:
                    ends.pop(a+1,None)
                    
                starts[a]=end
                ends[end]=a
                if(end -a +1) ==m: count+=1
            else:
                ends[a]=a
                starts[a]=a
                if m ==1: count+=1
            if count: res = i+1    
        return res

class UF:
    def __init__(self, n):
        self.cnt = [1] * n
        self.fa = [i for i in range(n)]
        
    def find(self, x):
        if x == self.fa[x]:
            return x
        
        self.fa[x] = self.find(self.fa[x])
        return self.fa[x]
    
    def unit(self, a, b):
        a, b = self.find(a), self.find(b)
        if a == b:
            return False
        
        if self.cnt[a] < self.cnt[b]:
            a, b = b, a
            
        self.cnt[a] += self.cnt[b]
        self.fa[b] = a
        return True
    
    def count(self, x):
        return self.cnt[self.find(x)]
    

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        # initialization
        n = len(arr)
        uf = UF(n + 1)
        bits = [0] * (n + 1)
        
        # computing
        result = -1
        count = 0
        for idx, val in enumerate(arr):
            if val > 1 and bits[val - 1] == 1:
                if uf.count(val - 1) == m:
                    count -= 1
                uf.unit(val, val - 1)
                
            if val < n and bits[val + 1] == 1:
                if uf.count(val + 1) == m:                
                    count -= 1
                uf.unit(val, val + 1)
                
            if uf.count(val) == m:
                count += 1
                
            if count > 0:
                result = idx
                
            bits[val] = 1
            
                
        return result + 1 if result != -1 else -1
                
        
        

class UnionFind(object):
    def __init__(self):
        self.parents = dict()
        self.sizes = dict()
    
    def __contains__(self, i):
        return self.parents.__contains__(i)
    
    def insert(self, i):
        self.parents[i] = i
        self.sizes[i] = 1

    def find(self, i):
        while i != self.parents[i]:
            self.parents[i] = self.find(self.parents[i])  
            i = self.parents[i]
        return i

    def union(self, p, q):
        root_p, root_q = list(map(self.find, (p, q)))
        if root_p == root_q: return
        small, big = sorted([root_p, root_q], key=lambda x: self.sizes[x])
        self.parents[small] = big
        self.sizes[big] += self.sizes[small]    


class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        uf = UnionFind()
        step_wanted = -1
        n = len(arr)
        for step, pos in enumerate(arr, 1):
            uf.insert(pos)
            for neighbor in [pos-1, pos+1]:
                if neighbor not in uf: continue
                if uf.sizes[uf.find(neighbor)] == m: step_wanted = step - 1
                uf.union(pos, neighbor)
        for i in range(1, n+1):
            if uf.sizes[uf.find(i)] == m:
                step_wanted = n
        return step_wanted

class DSU:
    def __init__(self, m):
        self.p = {}
        self.islands = {}
        self.m = m
        self.hasm = 0
    
    def make_island(self, x):
        self.p[x] = x
        self.islands[x] = 1
        if self.m == 1:
            self.hasm += 1
    
    def exist(self, x):
        if not x in self.p:
            return False
        return True
    
    def find(self, x):
        if x != self.p[x]:
            self.p[x] = self.find(self.p[x])
        return self.p[x]
    
    def union(self, x, y):
        if not self.exist(x) or not self.exist(y):
            return
        
        xr = self.find(x)
        yr = self.find(y)
        
        self.p[xr] = yr
        
        #both islands will be mutated after the following assignment
        if self.islands[yr] == self.m:
            self.hasm -= 1
        if self.islands[xr] == self.m:
            self.hasm -= 1
            
        self.islands[yr] = self.islands[xr] + self.islands[yr]
        
        if self.islands[yr] == self.m:
            self.hasm += 1
            
        del self.islands[xr]
        

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        #DSU
        res = -1
        dsu = DSU(m)
        for i, x in enumerate(arr):
            if not dsu.exist(x):
                dsu.make_island(x)
                dsu.union(x, x + 1)
                dsu.union(x, x - 1)
                if dsu.hasm != 0:
                    res = i + 1
        return res

class Tree:
    def __init__(self, lo, hi):
        self.lo = lo
        self.hi = hi
        self.mid = None
        self.left = self.right = None
        
    def size(self):
        return self.hi - self.lo + 1
        
    def split(self, val, m):
        if self.mid != None:
            if val < self.mid:
                if self.left.size() == 1:
                    self.left = None
                else:
                    return self.left.split(val, m)
            elif val > self.mid:
                if self.right.size() == 1:
                    self.right = None
                else:
                    return self.right.split(val, m)
            return False
        
        if val == self.lo:
            self.lo = val + 1
            if self.hi - self.lo + 1 == m:
                return True
            
        elif val == self.hi:
            self.hi = val - 1
            if self.hi - self.lo + 1 == m:
                return True
        else:
            self.mid = val
            check = False
            if val - 1 >= self.lo:
                if val - self.lo == m:
                    return True
                self.left = Tree(self.lo, val - 1)
            if val + 1 <= self.hi:
                if self.hi - val == m:
                    return True
                self.right = Tree(val + 1, self.hi)
        return False
    

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        if m == len(arr):
            return len(arr)
        
        root = Tree(1, len(arr))
        step = len(arr) - 1
        for i in range(len(arr)-1, -1, -1):
            if root.split(arr[i], m):
                return step
            step -= 1
        return -1

class UF:
    def __init__(self, n):
        self.arr = list(range(n + 1))
        self.rank = [1] * n
        
    def root(self, x): 
        curr = x    
        while curr != self.arr[curr]:
            curr = self.arr[curr]
    
        return curr
    
    def union(self, x, y):
        root_x = self.root(x)
        root_y = self.root(y)
        
        if root_x == root_y:
            return
        
        rank_x = self.rank[root_x]
        rank_y = self.rank[root_y]

        if rank_x >= rank_y:
            self.arr[root_y] = root_x
            self.rank[root_x] += rank_y
        else:
            self.arr[root_x] = root_y
            self.rank[root_y] += rank_x
            
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        uf = UF(len(arr))
        
        sizes = {}
        
        seen = set()
        last_time = None
        for i, elt in enumerate(arr):
            elt -= 1
            
            seen.add(elt)
            left = elt - 1
            right = elt + 1
            if left >= 0 and left in seen and right < len(arr) and right in seen:
                old_left_root = uf.root(left)
                old_right_root = uf.root(right)

                sizes[uf.rank[old_left_root]].remove(old_left_root)
                if len(sizes[uf.rank[old_left_root]]) == 0:
                    del sizes[uf.rank[old_left_root]]
                sizes[uf.rank[old_right_root]].remove(old_right_root)
                if len(sizes[uf.rank[old_right_root]]) == 0:
                    del sizes[uf.rank[old_right_root]]
                    
                uf.union(left, elt)
                uf.union(right, elt)

            elif left >= 0 and left in seen:
                old_left_root = uf.root(left) 
                sizes[uf.rank[old_left_root]].remove(old_left_root)
                if len(sizes[uf.rank[old_left_root]]) == 0:
                    del sizes[uf.rank[old_left_root]]
                    
                uf.union(left, elt)
            
            elif right < len(arr) and right in seen:
                old_right_root = uf.root(right)
                sizes[uf.rank[old_right_root]].remove(old_right_root)
                if len(sizes[uf.rank[old_right_root]]) == 0:
                    del sizes[uf.rank[old_right_root]]
                    
                uf.union(right, elt)
            
            new_root = uf.root(elt)
            new_rank = uf.rank[new_root]
            
            if new_rank not in sizes:
                sizes[new_rank] = set()
            sizes[new_rank].add(new_root)

            if m in sizes:
                last_time = i
            
        if last_time is None:
            return -1
        else:
            return last_time + 1
                


class Node:
    def __init__(self, parent, value):
        self.value = value
        self.parent = parent
        self.size = 1
        self.rank = 0

class UnionFind:
    def __init__(self, nodes):
        self.subsets = [Node(i, v) for i, v in enumerate(nodes)]
        self.maxSubSetSize = 1

    def union(self, i, j):
        irep = self.find(i)
        jrep = self.find(j)
        if irep == jrep:
            return
        if self.subsets[irep].rank > self.subsets[jrep].rank:
            self.subsets[jrep].parent = irep
            self.subsets[irep].size += self.subsets[jrep].size
            self.maxSubSetSize = max(self.maxSubSetSize, self.subsets[irep].size)
        else:
            self.subsets[irep].parent = jrep
            self.subsets[jrep].size += self.subsets[irep].size
            if self.subsets[irep].rank == self.subsets[jrep].rank:
                self.subsets[jrep].rank += 1
            self.maxSubSetSize = max(self.maxSubSetSize, self.subsets[jrep].size)
    
    def find(self, index):
        if self.subsets[index].parent != index:
            self.subsets[index].parent = self.find(self.subsets[index].parent)
        return self.subsets[index].parent
        

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        arr0 = [a - 1 for a in arr]
        uf = UnionFind(list(range(len(arr0))))
        lengthMSets = set()
        last_step = -1
        if m == 1:
            lengthMSets.add(arr0[0])
            last_step = 1
        visited = [False for _ in arr0]
        visited[arr0[0]] = True
        for i in range(1, len(arr0), 1):
            num = arr0[i]
            visited[num] = True
            if  num - 1 >= 0 and visited[num-1]:
                left_rep = uf.find(num-1)
                if left_rep in lengthMSets:
                    lengthMSets.remove(left_rep)
                uf.union(left_rep, num)
            if num + 1 < len(visited)and  visited[num+1]:
                right_rep = uf.find(num+1)
                if right_rep in lengthMSets:
                    lengthMSets.remove(right_rep)
                uf.union(right_rep, num)
            if uf.subsets[uf.find(num)].size == m:
                lengthMSets.add(uf.find(num))
            if len(lengthMSets) > 0:
                last_step = i + 1
        return last_step

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        n = len(arr)
        if m == n:
            return n
        uf = UnionFindHelper()
        res = -1
        for i, curr in enumerate(arr):
            uf.add(curr)
            step = i + 1
            for neighbor in [curr - 1, curr + 1]:
                if uf.contains(neighbor):
                    if uf.getrank(neighbor) == m:
                        res = step - 1
                    uf.union(neighbor, curr)
            if uf.getrank(curr) == m:
                res = step
        return res
        
class UnionFindHelper:
    def __init__(self):
        self.parent = {}
        self.ranks = {}
        self.count = 0
        
    def contains(self, item):
        return item in self.parent
    
    def add(self, item):
        if item not in self.parent:
            self.parent[item] = item
            self.ranks[item] = 1
            self.count += 1
    
    def getrank(self, item):
        return self.ranks[self.find(item)]
    
    def find(self, item):
        if item != self.parent[item]:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]
    
    def union(self, item1, item2):
        item1 = self.find(item1)
        item2 = self.find(item2)
        rank1 = self.ranks[item1]
        rank2 = self.ranks[item2]
        if item1 != item2:
            self.parent[item1] = item2
            self.ranks[item2] = rank1 + rank2
            self.count -= 1
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        L=len(arr)
        S=1+2**(L+1)
        count=1
        account=L//m-1
        C=2**m
        if m==L:
            return L
        
        while account:
            k=arr.pop()
            if m+1<=k:
                s=S>>(k-m-1)
                if s&(2*C-1)==1 :
                    return L-count
                s=s>>(m+2)
            else:
                s=(S>>k+1)&(2*C-1)
            if s&(2*C-1)==C:
                return L-count
            S+= 1 << k
            count+=1
            if arr==[]:
                break
        Max=L+1
        Min=0
        while account==0:
            k=arr.pop()
            if L-m>=k>Min:
                Min=k
            elif k<Max:
                Max=k
            if Max-Min==m+1:
                return L-count
            elif Max-Min<m+1:
                break
            count+=1
            if arr==[]:
                break
        return -1
class DSU:
    def __init__(self, n):
        self.parent = []
        for i in range(n + 2):
            self.parent.append(i)
        self.size = []
        for i in range(n + 2):
            self.size.append(1)
    def union(self, u, v):
        pu = self.find(u)
        pv  = self.find(v)
        if pu == pv:
            return
        if self.size[pv] > self.size[pu]:
            pu, pv = pv, pu
        #pu is bigger
        self.parent[pv] = pu
        self.size[pu] += self.size[pv]
    def find(self, u):
        if self.parent[u] != u:
            self.parent[u] = self.find(self.parent[u])
        return self.parent[u]
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        current_arr = [-1] * (len(arr) + 2)
        dsu = DSU(len(arr)+2)
        res = -1
        cur_sol = set()
        for i, val in enumerate(arr):
            if current_arr[val] == 1:
                continue
            else:
                current_arr[val] = 1
                if current_arr[val - 1] == 1:
                    dsu.union(val, val - 1)
                if current_arr[val + 1] == 1:
                    dsu.union(val, val + 1)
                pv = dsu.find(val)
                if dsu.size[pv] == m:
                    res = i + 1
                    cur_sol.add(pv)
                found = False
                for cs in cur_sol:
                    pcs = dsu.find(cs)
                    if dsu.size[pcs] == m:
                        found = True
                if found == True:
                    res = i + 1
                #res = i + 1
        return res

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        n=len(arr)
        lens=[0]*(n+2)
        count=[0]*(n+2)
        res=-1
        for i,num in enumerate(arr):
            if lens[num]:
                continue
            l=lens[num-1]
            r=lens[num+1]
            t=l+r+1
            lens[num-l]=lens[num+r]=lens[num]=t
            count[l]-=1
            count[r]-=1
            count[t]+=1
            if count[m]:
                res=i+1
        return res
# https://leetcode.com/problems/find-latest-group-of-size-m/discuss/806718/Python-Clean-Union-Find-solution-with-explanation-O(N)
# This solution utilize union by rank with 2 aims: count number of one as well as reduce the TC. One main point is that last step of group of ones of length m means we need bit set to ruin the last group of ones with length of m, or the this type of group will last until last step, so we can determine the result by determining the rank of left and right of currrent bit set. After one pass of arr, we can check if ranks of our union equal m.
class UnionFind:
    def __init__(self, n):
        self.parents = list(range(n))
        self.ranks = [0] * n

    def find(self, src: int) -> int:
        if self.parents[src] == src:
            return src
        self.parents[src] = self.find(self.parents[src])
        return self.parents[src]
    
    def union(self, src: int, dest: int) -> bool:
        rootSrc, rootDest = self.find(src), self.find(dest)
        if rootDest == rootSrc:
            return False
        
        if self.ranks[rootSrc] > self.ranks[rootDest]:
            self.parents[rootDest] = rootSrc
            self.ranks[rootSrc] += self.ranks[rootDest]
        else:
            self.parents[rootSrc] = rootDest
            self.ranks[rootDest] += self.ranks[rootSrc]
        return True
    
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        n, result = len(arr), -1
        uf = UnionFind(n)

        for step, idx in enumerate(arr):
            idx -= 1
            uf.ranks[idx] = 1
            for j in (idx - 1, idx + 1):
                if 0 <= j < n:
                    if uf.ranks[uf.find(j)] == m:
                        result = step
                    if uf.ranks[j]:
                        uf.union(idx, j)

        for i in range(n):
            if uf.ranks[uf.find(i)] == m:
                return n
        return result
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        parents = list(range(len(arr) + 1))
        group = [0] * (len(arr)  + 1)
        counter = collections.Counter()
        def find(x):
            if x != parents[x]:
                parents[x] = find(parents[x])
            return parents[x]
        
        def union(x, y):
            px = find(x)
            py = find(y)
            if px != py:
                parents[py] = px
                counter[group[px]] -= 1
                counter[group[py]] -= 1
                group[px] += group[py]
                counter[group[px]] += 1
            return
        
        visited = set()
        ans = -1
        
        for i in range(len(arr)):
            x = arr[i]
            group[x] = 1
            counter[1] += 1
            for y in [x - 1, x + 1]:
                if y in visited:
                    union(x, y)
            visited.add(x)
            
            if counter[m] > 0:
                ans = i + 1
        return ans
class UnionFind:
    def __init__(self, N):
        self.par = list(range(N))
        self.rank = [0]*N
        self.size = [0]*N
    
    def find(self, x):
        if self.par[x]!=x:
            self.par[x]=self.find(self.par[x])
        return self.par[x]
    
    def union(self, x, y):
        px = self.find(x)
        py = self.find(y)
        
        if px==py: return
        if self.rank[px]<self.rank[py]:
            self.par[px]=py
            self.size[py]+=self.size[px]
        elif self.rank[px]>self.rank[py]:
            self.par[py]=px
            self.size[px]+=self.size[py]
        else:
            self.par[py]=px
            self.size[px]+=self.size[py]
            self.rank[px]+=1

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        arr = [0]+arr
        N = len(arr)
        uf = UnionFind(N)
        res = []
        seen = set()
        matched = set()
        
        for i in range(1, N):
            seen.add(arr[i])
            matched.add(arr[i])
            uf.size[arr[i]]=1
            if arr[i]-1>=0 and arr[i]-1 in seen:
                uf.union(arr[i], arr[i]-1)
            if arr[i]+1<N and arr[i]+1 in seen:
                uf.union(arr[i], arr[i]+1)
                
            all_bigger = True
            for j in list(matched):
                idx = uf.find(j)
                if uf.size[idx]!=m:
                    matched.remove(j)

            if matched: 
                res.append(i)

        return res[-1] if res else -1
        
        
        
        

class DSU:
    def __init__(self, n):
        self.n = n
        self.fa = list(range(n))
        self.sz = [1 for _ in range(n)]

    def find(self, x):
        r = x
        while self.fa[r] != r:
            r = self.fa[r]
        i = x
        while i != r:
            i, self.fa[i] = self.fa[i], r
        return r
    
    def join(self, x, y):
        x = self.find(x)
        y = self.find(y)
        if x != y:
            self.fa[x] = y
            self.sz[y] += self.sz[x]
    
    def size(self, x):
        x = self.find(x)
        return self.sz[x]

class Solution:
    def findLatestStep(self, a: List[int], m: int) -> int:
        n = len(a)
        b = [0 for _ in range(n)]
        dsu = DSU(n)
        
        ans = -1
        valid = set()
        for k, i in enumerate(a, 1):
            j = i - 1
            b[j] = 1
            if j > 0 and b[j - 1]:
                dsu.join(j, j - 1)
            if j + 1 < n and b[j + 1]:
                dsu.join(j, j + 1)

            if m == dsu.size(j):
                valid.add(j)
            
            valid = set(p for p in valid if m == dsu.size(p))
            if valid:
                ans = k

        return ans
class UnionFind:
    def __init__(self):
        self.parent = {}
        self.rank = {}

    def add(self, x):
        self.parent[x] = x
        self.rank[x] = 1

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        xRoot, yRoot = self.find(x), self.find(y)

        if xRoot == yRoot:
            return

        xRank, yRank = self.rank[xRoot], self.rank[yRoot]
        if xRank < yRank:
            yRoot, xRoot = xRoot, yRoot

        self.parent[yRoot] = xRoot
        self.rank[xRoot] += self.rank[yRoot]
        # if self.rank[ yRoot] == self.rank[xRoot]:
        #     self.rank[xRoot] += 1

        return


class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        if m == len(arr):
            return len(arr)
        if m > len(arr):
            return -1
        uf = UnionFind()
        
        for i in range(1, len(arr)+1):
            uf.add(i)
        ans = -1
        seen = set()
        
        for i, n in enumerate(arr):
            uf.rank[n] = 1
            if n - 1 >= 1 and n - 1 in seen:
                if uf.rank[uf.find(n-1)] == m:
                    ans = i
                uf.union(n, n-1)
            if n + 1 <= len(arr) and n+1 in seen:
                if uf.rank[uf.find(n+1)] == m:
                    ans = i
                uf.union(n, n + 1)
            seen.add(n)
        return ans
        
        
        
        
        

class UnionFindSet:
    def __init__(self, n):
        self.parent = [i for i in (list(range(n)))]
        self.rank = [0] * n
        
    def find(self, u):
        if u != self.parent[u]:
            self.parent[u] = self.find(self.parent[u])
        return self.parent[u]
    
    def union(self, u, v):
        pu = self.find(u)
        pv = self.find(v)
        if pu == pv:
            return False
        if self.rank[pu] < self.rank[pv]:
            self.rank[pv] += self.rank[pu]
            self.parent[pu] = pv
        else:
            self.rank[pu] += self.rank[pv]
            self.parent[pv] = pu
        return True
            

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        n = len(arr)
        ans = -1
        uf = UnionFindSet(n)
        
        for step, i in enumerate(arr):
            i -= 1
            uf.rank[i] = 1
            for j in (i-1, i+1):
                if  0 <= j < n:
                    if uf.rank[uf.find(j)] == m:
                        ans = step
                    if uf.rank[j]:
                        uf.union(i, j)
                        
        for i in range(n):
            if uf.rank[uf.find(i)] == m:
                return n
        return ans
        

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        n = len(arr)
        s = '1' * n
        sm = '1' * m
        if m == n:
            return n
        for ind, i in enumerate(arr[::-1]):
            #print(s, i, sm)
            s = s[:i-1] +  '0' + s[i:]
   # print((i - 1 + m < n ))
   #          print(s[i : i  + m] , sm)
   #          print(i, m, n)
            if (i - 1 - m >= 0 and s[i - 1 - m: i - 1] == sm and ((i - 1 - m == 0) or s[i - 2 - m] == '0')) or  (i  + m <= n and s[i : i  + m] == sm and ((i + m == n ) or s[i + m ] == '0')):
                return n - ind - 1
        return -1

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        
        f = {}
        n = len(arr)
        ans = -1
        ranks = [0] * (n+1)
        
        if m == n:
            return m
        
        def find(x):
            f.setdefault(x,x)
            if f[x] != x:
                f[x] = find(f[x])
            
            return f[x]
        
        def union(x,y):
            px,py = find(x), find(y)
            
            if ranks[px] > ranks[py]:
                ranks[px] += ranks[py]
                f[py] = px
            else:
                ranks[py] += ranks[px]
                f[px] = py

            
        for i,a in enumerate(arr):
            ranks[a] += 1
            
            for j in [a-1,a+1]:
                if 1<= j <= n:
                    if ranks[find(j)] == m:
                        ans = i
                    if ranks[j]:
                        union(a,j)
        
        return ans
class UnionFind:

    def __init__(self, n):
        self.parent = [i for i in range(n)]
        self.count = [1]*n

    def find(self, x):
        parent = self.parent[x]
        if parent != x:
            # reassign its parent to the root
            self.parent[x] = self.find(parent)
        return self.parent[x]
    
    def get_count(self, x):
        return self.count[self.find(x)]
        
    def union(self, x, y):
        xparent, yparent = self.find(x), self.find(y)
        if xparent == yparent:
            return
        self.parent[yparent] = xparent  # assign yparent parent to xparent
        self.count[xparent] += self.count[yparent]

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        n = max(arr)
        bits = [0]*(100005)
        disjoint = UnionFind(n+1)
        mapping = collections.defaultdict(int)
        ans = -1

        for ind in range(len(arr)):
            pos = arr[ind]
            bits[pos] = 1
            mapping[1] += 1
            i, j = pos-1, pos+1
            if bits[i] and disjoint.find(i) != disjoint.find(pos):
                mapping[disjoint.get_count(i)] -= 1
                mapping[disjoint.get_count(pos)] -= 1
                disjoint.union(i, pos)
                mapping[disjoint.get_count(pos)] += 1
            if bits[j] and disjoint.find(j) != disjoint.find(pos):
                mapping[disjoint.get_count(j)] -= 1
                mapping[disjoint.get_count(pos)] -= 1
                disjoint.union(j, pos)
                mapping[disjoint.get_count(pos)] += 1
            if mapping[m] > 0:
                ans = ind+1
        return ans
class UnionFind:
    def __init__(self, n):
        self.parents = [i for i in range(n)]
        self.size = [1] * n

    def find(self, i):
        while i != self.parents[i]:
            self.parents[i] = self.parents[self.parents[i]] # path halving
            i = self.parents[i]
        return i
    
    def union(self, a, b):
        aPar = self.find(a)
        bPar = self.find(b)
        
        if aPar == bPar:
            return
        
        # union by size
        if self.size[aPar] > self.size[bPar]:
            self.parents[bPar] = aPar
            self.size[aPar] += self.size[bPar]
        else:
            self.parents[aPar] = bPar
            self.size[bPar] += self.size[aPar]

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        N = len(arr)
        uf = UnionFind(N)
        isAlive = [False] * N
        numM = 0
        latest = -1
        
        for index, i in enumerate(arr):
            isAlive[i-1] = True
            if i != 1 and isAlive[i-2]:
                if uf.size[uf.find(i-2)] == m:
                    numM -= 1
                uf.union(i-1, i-2)
            if i != N and isAlive[i]:
                if uf.size[uf.find(i)] == m:
                    numM -= 1
                uf.union(i-1, i)
            if uf.size[uf.find(i-1)] == m:
                numM += 1
            if numM > 0:
                latest = index + 1
                
        return latest
                
            
            
            

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        starts, ends = {}, {}
        res = None
        contributors = set()
        for i, pos in enumerate(arr):
            if pos-1 not in ends:
                l = pos
            else:
                l = ends[pos-1]
                del ends[pos-1]
                if l in contributors:
                    contributors.remove(l)
            if pos+1 not in starts:
                r = pos
            else:
                r = starts[pos+1]
                del starts[pos+1]
                if pos+1 in contributors:
                    contributors.remove(pos+1)
            if m == r - l + 1:
                contributors.add(l)
            if contributors:
                res = i+1
            starts[l] = r
            ends[r] = l
        return res if res is not None else -1
class UnionFindSet:
    def __init__(self, n):
        self.parents = list(range(n))
        self.ranks = [0] * n
        
    def find(self, u):
        if u != self.parents[u]:
            self.parents[u] = self.find(self.parents[u])
        return self.parents[u]
    
    def union(self, u, v):
        pu, pv = self.find(u), self.find(v)
        if pu != pv:
            self.parents[pv] = pu
            self.ranks[pu] += self.ranks[pv]

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        n, ans = len(arr), -1
        uf = UnionFindSet(n)
        
        for step, i in enumerate(arr):
            i -= 1
            uf.ranks[i] = 1
            for j in (i - 1, i + 1):
                if 0 <= j < n:
                    if uf.ranks[uf.find(j)] == m:
                        ans = step
                    if uf.ranks[j]:
                        uf.union(i, j)
            if uf.ranks[uf.find(i)] == m:
                ans = step + 1
            # print(step, i, uf.ranks, ans)
        return ans
from sortedcontainers import SortedList

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        n = len(arr)
        s = SortedList([0, n + 1])
        
        if n == m: return n
        
        for i, x in enumerate(reversed(arr)):
            j = s.bisect_left(x)
            s.add(x)
            if m == x - s[j-1] - 1 or m == s[j + 1] - x - 1:
                return n - i - 1
            
        return -1
class UnionFindSet:
    def __init__(self, n):
        self.parent = [i for i in range(n)]
        self.rank = [0 for _ in range(n)]
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        xroot, yroot = self.find(x), self.find(y)
        if xroot == yroot:
            return
        
        if self.rank[xroot] > self.rank[yroot]:
            self.rank[xroot] += self.rank[yroot]
            self.parent[yroot] = xroot
        else:
            self.rank[yroot] += self.rank[xroot]
            self.parent[xroot] = yroot
        return True

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        n = len(arr)
        if n == m:
            return m
        
        uf = UnionFindSet(n)
        ans = -1
        
        for step, idx in enumerate(arr):
            idx -= 1
            uf.rank[idx] = 1
            for j in (idx - 1, idx + 1):
                if 0 <= j < n:
                    if uf.rank[uf.find(j)] == m:
                        ans = step
                    if uf.rank[j]:
                        uf.union(idx, j)
        
        return ans
from typing import List


class UnionFind:
  def __init__(self, n):
    self.parent = list(range(n))
    self.rank = [0] * n

  def find(self, u):
    if u != self.parent[u]:
      self.parent[u] = self.find(self.parent[u])
    return self.parent[u]

  def union(self, u, v):
    pu, pv = self.find(u), self.find(v)
    if pu == pv:
      return False
    if self.rank[pu] > self.rank[pv]:
      self.parent[pv] = pu
      self.rank[pu] += self.rank[pv]
    elif self.rank[pv] > self.rank[pu]:
      self.parent[pu] = pv
      self.rank[pv] += self.rank[pu]
    else:
      self.parent[pu] = pv
      self.rank[pv] += self.rank[pu]
    return True


class Solution:
  def findLatestStep(self, arr: List[int], m: int) -> int:
    ans = -1
    n = len(arr)
    uf = UnionFind(n)

    for step, k in enumerate(arr):
      i = k - 1
      uf.rank[i] = 1
      for j in (i - 1, i + 1):
        if 0 <= j < n:
          if uf.rank[uf.find(j)] == m:
            ans = step
          if uf.rank[j]:
            uf.union(i, j)

    for i in range(n):
      if uf.rank[uf.find(i)] == m:
        return n

    return ans


class Node:
    def __init__(self):
        self.left = None
        self.right = None
        self.length = 0

class Solution:
    def findLatestStep(self, arr, m):
        def remove(node):
            left = node.left
            node.right.left = left
            left.right = node.right
            
        count = 0
        node = {}
        M= max(arr)
        for i in range(M+2):
            n = Node()
            node[i] = n
            if i == 0:
                continue
            
            node[i-1].right = n
            n.left = node[i-1]
        
        ans = -1
        for step, i in enumerate(arr):
            node[i].length = 1
            if node[i].left.length > 0:
                # merge with left
                node[i].length += node[i].left.length
                if node[i].left.length == m:
                    count -= 1
                remove(node[i].left)
            if node[i].right.length > 0:
                if node[i].right.length == m:
                    count -= 1
                node[i].length += node[i].right.length
                remove(node[i].right)
            if node[i].length == m:
                count += 1
            if count > 0:
                ans = step + 1
            #print(step, count, ans)
        return ans
                

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        n = len(arr)
        counter = [0] * (n+1)
        parent = [i for i in range(n+1)]
        size = [0 for i in range(n+1)]
        
        def find(u):
            if parent[u] == u:
                return u
            parent[u] = find(parent[u])
            return parent[u]
        
        def union(u, v):
            ru, rv = find(u), find(v)
            if size[ru] < size[rv]:
                parent[ru] = rv
                size[rv] += size[ru]
            else:
                parent[rv] = ru
                size[ru] += size[rv]
        
        def getSize(p):
            return size[find(p)]
        
        res = -1
        for i, pos in enumerate(arr):
            size[pos] = 1
            # counter[1] += 1
            if pos > 0:
                if getSize(pos-1) > 0:
                    s_last = getSize(pos-1)
                    union(pos-1, pos)
                    counter[s_last] -= 1
            if pos < n:
                if getSize(pos+1) > 0:
                    s_next = getSize(pos+1)
                    union(pos, pos+1)
                    counter[s_next] -= 1
            counter[getSize(pos)] += 1
            if counter[m] > 0:
                res = i+1
        
        return res

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        n = len(arr)
        size = {i:1 for i in range(n)}
        dp = [*range(n)]
        res = -1
        
        def union(i, j):
            size_count[size[find(i)]] -= 1
            size_count[size[find(j)]] -= 1
            tmp = size[find(i)] + size[find(j)]
            size_count[tmp] += 1
            size[find(i)] = size[find(j)] = tmp
            dp[find(i)] = dp[find(j)]
        
        def find(i):            
            if i != dp[i]: dp[i] = find(dp[i])
            return dp[i]                
            
        curr = [0] * n                
        size_count = [0] * (n + 1)
        for k, i in enumerate([x - 1 for x in arr]):
            curr[i] = 1
            size_count[1] += 1
            if i < n - 1 and curr[i + 1]:
                union(i, i + 1)
            if i and curr[i - 1]:
                union(i, i - 1)
            if size_count[m] > 0:
                res = k + 1
       
        return res
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        mem = {}
        gsizes = 0
        inc = False
        lastpos = -1
        changed = False
        
        for i, num in enumerate(arr):
            # print(mem)
            gsize = 1
            if num -1 in mem:
                if mem[num-1][0] == m:
                    gsizes -= 1
                    changed = True
                gsize += mem[num-1][0]
            
            if num +1 in mem:
                if mem[num+1][0] == m:
                    gsizes -= 1
                    changed = True
                gsize += mem[num+1][0]
            
            # print(gsize, m)
            if gsize == m:
                inc = True
                changed = True
                gsizes += 1
            # print(gsizes)

            # print(mem)
            # if inc:
            #     print(gsizes)
            #     print(gsize)
            #     print(num)
            
            # print(gsizes)
            if gsizes == 0 and inc and changed:
                changed = False
                lastpos = i
                # print('end')
                # return i
            
            # mem[num] = (gsize, num)
            if num +1 not in mem and num -1 not in mem:
                end = num
            elif num + 1 in mem:
                end = mem[num+1][1]
            else:
                end = mem[num-1][1]
            mem[num] = (gsize, end)
            
            if num - 1 in mem:
                old = mem[num-1][1]
                mem[mem[num-1][1]] = (gsize, mem[num+1][1] if num +1 in mem else num)
                
            if num + 1 in mem:
                mem[mem[num+1][1]] = (gsize, old if num -1 in mem else num)
                
#         if gsizes:
#             return len(arr)
        # print(gsizes)
            
        # return -1 if gsizes == 0 else len(arr)
        return len(arr) if gsizes else lastpos
        
            
                

from collections import defaultdict 
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        n=len(arr)

        if m==n:
            return m

        # number at index i means the length of 1's array starting or ending at i
        b=[0]*(n+2)
        cnt=defaultdict(lambda : 0)
        latest = -1
        for i in range(n):
            left = b[arr[i]-1]
            right = b[arr[i]+1]
            
            # if left is non-zero, arr[i]-1 must be the end of a 1's array
            # it cannot be a start because arr[i] is not yet 1
            # in this case we merge with left, arr[i] will be the new end point
            if left > 0 and right == 0:
                # b[arr[i]-1]+=1
                b[arr[i]]+=1+left
                # arr[i]-1-left-1 is the start of the 1's array
                b[arr[i]-1-left+1]+=1
                cnt[left] -= 1
                cnt[left+1] += 1
            elif left == 0 and right > 0:
                # b[arr[i]+1]+=1
                b[arr[i]]+=1+right
                # arr[i]+1+right-1 is the end of the 1's array
                b[arr[i]+1+right-1]+=1
                cnt[right] -= 1
                cnt[right+1] += 1
            # if both are non zero, we can merge the left array and right array
            # creating an array with length left + right + 1
            elif left > 0 and right > 0:
                # b[arr[i]-1]+=1+right
                # b[arr[i]+1]+=1+left
                # b[arr[i]]+=1+left+right

                # arr[i]-1-left-1 is the start of the 1's array
                b[arr[i]-1-left+1]+=1+right
                # arr[i]+1+right-1 is the end of the 1's array
                b[arr[i]+1+right-1]+=1+left
                cnt[right] -= 1
                cnt[left] -= 1
                cnt[1+left+right] += 1
            else:# final case where both left and right are zero
                b[arr[i]]+=1
                cnt[1]+=1
            if m in cnt:
                if cnt[m]>0:
                    latest = i+1
        return latest
class DSU:
    
    def __init__(self, n, m):
        self.parents = [-1] * n
        self.sizes = [0] * n
        self.target = m
        self.matches = 0
        
    def find(self, x):
        parent = self.parents[x]
        if parent in [-1, x]:
            return parent
        self.parents[x] = self.find(parent)
        return self.parents[x]
    
    def union(self, x, y):
        if x == y:
            self.parents[x] = x
            self.sizes[x] = 1
            if 1 == self.target:
                self.matches += 1
        else:
            px, py = self.find(x), self.find(y)
            sx, sy = self.sizes[px], self.sizes[py]
            if sy > sx:
                px, py = py, px
                sx, sy = sy, sx
            self.parents[py] = px
            self.sizes[px] = sx + sy
            if sx == self.target:
                self.matches -= 1
            if sy == self.target:
                self.matches -= 1
            if sx + sy == self.target:
                self.matches += 1     

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        dsu = DSU(len(arr), m)
        last_good = -1
        for i, v in enumerate(arr):
            v -= 1
            dsu.union(v, v)
            if v-1 >= 0 and dsu.find(v-1) != -1:
                dsu.union(v, v-1)
            if v+1 < len(arr) and dsu.find(v+1) != -1:
                dsu.union(v, v+1)
            if dsu.matches > 0:
                last_good = i+1
        return last_good
                

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        n = len(arr)
        p = [i for i in range(n)]
        l = [0 for i in range(n)]
        s = [0 for i in range(n)]
        counts = collections.Counter()
        res = []
        def find(a):
            if p[a] == a: return a
            p[a] = find(p[a])
            
            return p[a]
        def union(a, b):
            p_a, p_b = find(a), find(b)
            if p[p_b] != p[p_a]:
                p[p_b] = p[p_a]
                l[p_a] += l[p_b]
        for v in arr:
            i = v - 1
            s[i] = 1
            l[i] = 1
            f_a = f_b = False
            if i + 1 < n and s[i + 1] == 1:
                counts[l[find(i + 1)]] -= 1
                union(i, i + 1)
                f_a = True
            if i - 1 >= 0 and s[i - 1] == 1:
                counts[l[find(i - 1)]] -= 1
                union(i - 1, i)
                f_b = True
            if f_a and f_b:
                counts[l[find(i - 1)]] += 1
            elif f_a:
                counts[l[find(i)]] += 1
            elif f_b:
                counts[l[find(i - 1)]] += 1
            else:
                counts[l[find(i)]] += 1
            res.append(counts[m])
            
        for i in range(n - 1, -1, -1):
            if res[i] > 0:
                return i + 1
        
        return -1

from sortedcontainers import SortedList

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        sl = SortedList()
        n = len(arr)
        if n == m:
            return n
        s = n
        while s >= 1:
            s -= 1
            if len(sl) == 0:
                if n-arr[s] == m or arr[s] - 1 == m:
                    return s
            else:
                idx = sl.bisect_left(arr[s])
                if idx == 0:
                    if arr[s] - 1 == m or sl[idx] - arr[s] - 1 == m:
                        return s
                elif idx == len(sl):
                    if n - arr[s] == m or arr[s] - sl[idx-1] - 1 == m:
                        return s
                else:
                    if arr[s] - sl[idx-1] - 1 == m or sl[idx] - arr[s] - 1 == m:
                        return s
            sl.add(arr[s])
        return -1
class UnionFindSet:
    def __init__(self, n):
        self.parents = list(range(n))
        self.ranks = [0] * n
        
    def find(self, u):
        if u != self.parents[u]:
            self.parents[u] = self.find(self.parents[u])
        return self.parents[u]
    
    def union(self, u, v):
        pu, pv = self.find(u), self.find(v)
        if pu == pv:
            return False
        if self.ranks[pu] > self.ranks[pv]:
            self.parents[pv] = pu
            self.ranks[pu] += self.ranks[pv]
        elif self.ranks[pv] > self.ranks[pu]:
            self.parents[pu] = pv
            self.ranks[pv] += self.ranks[pu]
        else:
            self.parents[pu] = pv
            self.ranks[pv] += self.ranks[pu]
        return True


class Solution:
    def findLatestStep(self, A: List[int], m: int) -> int:
        if m == len(A):
            return len(A)
        
        n, ans = len(A), -1
        uf = UnionFindSet(n)
        
        for step, i in enumerate(A):
            i -= 1
            uf.ranks[i] = 1
            for j in (i-1, i+1):
                if 0 <= j < n:
                    if uf.ranks[uf.find(j)] == m:
                        ans = step
                    if uf.ranks[j]:
                        uf.union(i,j)
        return ans
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        #union find
        
        
        dsu = [False] * (len(arr) +1)
        dsu[0] = 0
        
        
        def find(a):
            if dsu[a] == False:
                dsu[a] = -1
                return a
            x = []
            while dsu[a] >=0:
                x.append(a)
                a=dsu[a]
            for b in x:
                dsu[b] = a        
            return a
        def size(a):
            pa=find(a)
            return abs(dsu[pa])
        def union(a,b):
            pa,pb = find(a),find(b)
            if pa != pb:
                if dsu[pa] < dsu[pb]:
                    dsu[pa] += dsu[pb]
                    dsu[pb] = pa
                else:
                    dsu[pb] += dsu[pa]
                    dsu[pa] = pb
                return True
            return False
        ans=-1
        for i,x in enumerate(arr,1):
            find(x)
            if x> 1 and dsu[x-1] != False:
                if size(x-1) == m:
                    dsu[0] -=1
                union(x,x-1)
            if x<len(arr) and dsu[x+1] != False:
                if size(x+1) == m:
                    dsu[0]-=1
                union(x,x+1)
            if size(x) == m:
                dsu[0]+=1
            #print(dsu[0])
            if dsu[0] > 0:
                ans = i
        return ans

from sortedcontainers import SortedList

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        sizes = Counter((len(arr),))
        groups = SortedList([[1, len(arr)]])
        for i in range(len(arr)-1, -1, -1):
            if m in sizes:
                return i + 1
            n = arr[i]
            j = groups.bisect_left([n, n]) 
            if j == len(groups) or j > 0 and groups[j-1][1] >= n:
                j -= 1
            h, t = groups.pop(j)
            sizes[t - h + 1] -= 1
            if h < n: 
                groups.add([h, n-1])
                sizes[n-1 - h + 1] += 1 
            if t > n:    
                groups.add([n+1, t])
                sizes[t - n - 1 + 1] += 1 
        return -1 
class UnionFindSet:
    def __init__(self, n):
        self.parents = list(range(n))
        self.ranks = [0] * n
        
    def find(self, u):
        if u != self.parents[u]:
            self.parents[u] = self.find(self.parents[u])
        return self.parents[u]
    
    def union(self, u, v):
        pu, pv = self.find(u), self.find(v)
        if pu == pv:
            return False
        if self.ranks[pu] > self.ranks[pv]:
            self.parents[pv] = pu
            self.ranks[pu] += self.ranks[pv]
        elif self.ranks[pv] > self.ranks[pu]:
            self.parents[pu] = pv
            self.ranks[pv] += self.ranks[pu]
        else:
            self.parents[pu] = pv
            self.ranks[pv] += self.ranks[pu]
        return True

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        n, ans = len(arr), -1
        uf = UnionFindSet(n)
        
        for step, i in enumerate(arr):
            i -= 1
            uf.ranks[i] = 1
            for j in (i - 1, i + 1):
                if 0 <= j < n:
                    if uf.ranks[uf.find(j)] == m:
                        ans = step
                    if uf.ranks[j]:
                        uf.union(i, j)
        
        for i in range(n):
            if uf.ranks[uf.find(i)] == m:
                return n
            
        return ans
        
 

class UnionFindSet:
    def __init__(self, n):
        self.parents = list(range(n))
        self.ranks = [1] * n
        
    def find(self, u):
        if u != self.parents[u]:
            self.parents[u] = self.find(self.parents[u])
        return self.parents[u]
    
    def union(self, u, v):
        pu, pv = self.find(u), self.find(v)
        if pu == pv:
            return False
        if self.ranks[pu] > self.ranks[pv]:
            self.parents[pv] = pu
            self.ranks[pu] += self.ranks[pv]
        elif self.ranks[pv] > self.ranks[pu]:
            self.parents[pu] = pv
            self.ranks[pv] += self.ranks[pu]
        else:
            self.parents[pu] = pv
            self.ranks[pv] += self.ranks[pu]
        return True
    

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        n = len(arr)
        uf = UnionFindSet(n)
        b_arr = [0] * n
        ans = -1
        for step, num in enumerate(arr):
            idx = num - 1
            b_arr[idx] = 1
            if idx > 0 and b_arr[idx - 1]:
                p = uf.find(idx - 1)
                if uf.ranks[p] == m:
                    ans = step
                uf.union(idx, idx - 1)
            if idx < n - 1 and b_arr[idx + 1]:
                p = uf.find(idx + 1)
                if uf.ranks[p] == m:
                    ans = step
                uf.union(idx, idx + 1)
            p = uf.find(idx)
            if uf.ranks[p] == m:
                ans = step + 1
                
        for idx in range(n):
            p = uf.find(idx)
            if uf.ranks[p] == m:
                return n
            
        return ans
from collections import deque

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        pieces = {}
        goodpieces = {}
        latest = -1
        for i in range(1, len(arr) + 1):
            n = arr[i-1]
            start, end = 0,0
            if n+1 in pieces and n-1 in pieces:
                start = pieces.pop(n-1)
                end = pieces.pop(n+1)
            elif n+1 in pieces:
                start = n
                end = pieces.pop(n+1)
            elif n-1 in pieces:
                start = pieces.pop(n-1)
                end = n
            else:
                start = n
                end = n
            if (end - start + 1) == m:
                goodpieces[start] = end
            pieces[start] = end
            pieces[end] = start
            bad = []
            for piece in goodpieces:
                if (piece in pieces) and pieces[piece] == goodpieces[piece]:
                    latest = i
                else:
                    bad.append(piece)
            for b in bad:
                del goodpieces[b]

            #print(pieces)
        return latest
                    

from collections import Counter
class DSU:
    def __init__(self, n):
        self.dic = [i for i in range(n)]
    def find(self, n1):
        if self.dic[n1] != n1:
            self.dic[n1] = self.find(self.dic[n1])
        return self.dic[n1]
    def union(self, n1, n2):
        s1 = self.find(n1)
        s2 = self.find(n2)
        self.dic[s2] = s1
        
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        cnt = Counter()        
        area = Counter()
        fliped = [False for _ in arr]
        dsu = DSU(len(arr))        
        res = []
        def union_bits(b, bi):
            if not (0 <= bi < len(arr) and fliped[bi]):
                return 0
            s = dsu.find(bi)
            sa = area[s] 
            cnt[sa] -= 1
            dsu.union(s, b)     
            return sa
        for i, b in enumerate(arr):
            b -= 1
            fliped[b] = True             
            ba = 1
            ba += union_bits(b, b - 1)
            ba += union_bits(b, b + 1)
            s = dsu.find(b)
            area[s] = ba
            cnt[ba] += 1
            if cnt[m] > 0:
                res.append(i)
        # print(res)
        return res[-1] + 1 if res else -1
            

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        bits = [0] * len(arr)
        uf = UnionFind(bits)
        step = 0
        ans = -1
        for real_n in arr:
            step += 1
            # print("step", step)
            n = real_n-1
            bits[n] = 1
            uf.father[n] = n
            uf.cnt[n] = 1
            uf.cntFreq[1] += 1
            # print(bits)
            if n-1 >= 0 and bits[n-1] == 1:
                uf.union(n, n-1)  
            if n+1 < len(bits) and bits[n+1] == 1:
                uf.union(n, n+1) 
            # print(uf.cntFreq)
            if uf.cntFreq[m] > 0:
                ans = step
        return ans
            
class UnionFind:
    def __init__(self, bits):
        self.len = len(bits)
        self.father = [-1] * (self.len)
        self.cnt = [0] * (self.len)
        self.cntFreq = collections.Counter()

                    
    def union(self, p, q):
        rootP = self.find(p)
        rootQ = self.find(q)
        cntP = self.cnt[rootP]
        cntQ = self.cnt[rootQ]
        if rootP != rootQ:
            self.father[rootP] = rootQ
            self.cntFreq[self.cnt[rootP]] -= 1
            self.cntFreq[self.cnt[rootQ]] -= 1
            self.cntFreq[cntP+cntQ] += 1
            self.cnt[rootQ] = cntP+cntQ
        
    def find(self, p):
        rootP = self.father[p]
        while rootP != self.father[rootP]:
            rootP = self.father[rootP]
        self.father[p] = rootP
        return rootP

class DSU:
    def __init__(self, n):
        self.r = collections.defaultdict(int)
        self.p = collections.defaultdict(int)
        self.s = collections.defaultdict(int)
        self.scount = collections.defaultdict(int)
    
    def find(self, x):
        if self.p[x] != x:
            self.p[x] = self.find(self.p[x])
        return self.p[x]
    
    def union(self, x, y):
        px = self.find(x)
        py = self.find(y)
        
        if self.r[px] > self.r[py]:
            self.p[py] = px
            self.scount[self.s[py]] -= 1
            self.scount[self.s[px]] -= 1
            self.s[px] += self.s[py]
            self.s[py] = 0
            self.scount[self.s[px]] += 1
            
        if self.r[px] < self.r[py]:
            self.p[px] = py
            self.scount[self.s[py]] -= 1
            self.scount[self.s[px]] -= 1
            self.s[py] += self.s[px]
            self.s[px] = 0
            self.scount[self.s[py]] += 1
        else:
            self.p[py] = px
            self.scount[self.s[py]] -= 1
            self.scount[self.s[px]] -= 1
            self.r[px] += 1
            self.s[px] += self.s[py]
            self.s[py] = 0
            self.scount[self.s[px]] += 1

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        dsu = DSU(len(arr))
        seen = set()
        ret = -1
        for i in range(len(arr)):
            dsu.p[arr[i]] = arr[i]
            dsu.s[arr[i]] = 1
            dsu.r[arr[i]] = 0
            dsu.scount[1] += 1
            if arr[i] + 1 in seen:
                dsu.union(arr[i], arr[i] + 1)
            if arr[i] - 1 in seen:
                dsu.union(arr[i], arr[i] - 1)
                
            if dsu.scount[m] > 0:
                ret = max(ret, i)
            seen.add(arr[i])
            
        return ret + 1 if ret != -1 else -1

        
        
        
        
        

class UnionFindSet:
    def __init__(self, n):
        self.parents = list(range(n))
        self.ranks = [0] * n
        
    def find(self, u):
        if u != self.parents[u]:
            self.parents[u] = self.find(self.parents[u])
        return self.parents[u]
    
    def union(self, u, v):
        pu, pv = self.find(u), self.find(v)
        if pu == pv:
            return False
        if self.ranks[pu] > self.ranks[pv]:
            self.parents[pv] = pu
            self.ranks[pu] += self.ranks[pv]
        else:
            self.parents[pu] = pv
            self.ranks[pv] += self.ranks[pu]
        return True
    
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        if m == len(arr):
            return m
        
        n, ans = len(arr), -1
        uf = UnionFindSet(n)
        
        for step, i in enumerate(arr):
            i -= 1
            uf.ranks[i] = 1
            for j in (i - 1, i + 1):
                if 0 <= j < n:
                    if uf.ranks[uf.find(j)] == m:
                        ans = step
                    if uf.ranks[j]:
                        uf.union(i, j)
        
        return ans

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        n = max(arr)
        dp = [1] * n
        steps = n
        if m == n:
            return steps

        for a in reversed(arr):
            steps -= 1
            i = a - 1
            dp[i] = 0
            j = i + 1
            cnt = 0
            while j < n and dp[j] == 1:
                cnt += 1
                if cnt > m:
                    break
                j += 1
            else:
                if cnt == m:
                    return steps
            j = i - 1
            cnt = 0
            while j >= 0 and dp[j] == 1:
                cnt += 1
                if cnt > m:
                    break
                j -= 1
            else:
                if cnt == m:
                    return steps
        return -1
from bisect import bisect_left
class Solution:
    def findLatestStep(self, arr, m: int) -> int:
        n = len(arr)
        segments = [(1, n)]
        if m == n:
            return n
        for cur_iter, zero in enumerate(arr[::-1]):
            index = bisect_left(segments, (zero, 9999999999)) - 1
            #print(segments, zero, index)
            seg = segments[index]
            if seg[1] == 1 and seg[0] == zero:
                del segments[index]
            elif seg[1] == 1:
                assert False
            else:
                del segments[index]
                first_length = zero-seg[0]
                second_length = seg[0]+seg[1]-1-zero
                if first_length == m or second_length == m:
                    return n - cur_iter - 1
                if second_length >= 1:
                    segments.insert(index, (zero+1, second_length))
                if first_length >= 1:
                    segments.insert(index, (seg[0], first_length))
            # print(segments)
        return -1

class Solution:
    def find(self, n):
        if self.par[n] == n:
            return n
        else:
            self.par[n] = self.find(self.par[n])
            return self.par[n]
        
    def union(self, n1, n2):
        p1 = self.find(n1)
        p2 = self.find(n2)
        if self.rank[p1] < self.rank[p2]:
            self.par[p1] = p2
            self.rank[p2] += self.rank[p1]
        elif self.rank[p1] > self.rank[p2]:
            self.par[p2] = p1
            self.rank[p1] += self.rank[p2]    
        else:
            self.par[p2] = p1
            self.rank[p1] += self.rank[p2]
            
        
    def findLatestStep(self, arr: List[int], m: int) -> int:
        N = len(arr)
        self.N = N
        if m == N:
            return N
        
        self.par = list(range(N+1))
        self.rank = [0]*(N+1)
        result = -1
        s = '0'*(N+1)
        for i, v in enumerate(arr, 1):
            self.rank[v] = 1
            for j in [v-1, v+1]:
                if 1<=j<=N and self.rank[j]:
                    if self.rank[self.find(j)] == m:
                        result = i-1
                    self.union(j, v)

            
        for i in range(1, N+1):
            if self.rank[self.find(i)] == m:
                return N
            
        return result
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        res,matchGroups=-1,set()
        uf={}
        groupSize=defaultdict(lambda: 1)
        def find(x):
            uf.setdefault(x,x)
            if uf[x]!=x:
                uf[x]=find(uf[x])
            return uf[x]
        def union(x,y):
            nonlocal groupId
            gx,gy=find(x),find(y)
            if gx==gy:
                return
            if gx in matchGroups:
                matchGroups.remove(gx)
            if gy in matchGroups:
                matchGroups.remove(gy)
            size=groupSize[find(x)]+groupSize[find(y)]
            uf[find(x)]=find(y)
            groupSize[find(x)]=size
        cur=[0]*(len(arr)+2)
        for i,num in enumerate(arr):
            cur[num]=1
            if cur[num-1]==1:
                union(num,num-1)
            if cur[num+1]==1:
                union(num,num+1)
            groupId=find(num)
            if groupSize[find(num)]==m:
                matchGroups.add(groupId)
            if matchGroups:
                res=i+1
        return res
class UnionFind:
    def __init__(self, n):
        self.roots = [i for i in range(n)]
        self.sizes = [1 for i in range(n)]
        
    def root(self, a):
        c = a
        while self.roots[c] != c:
            c = self.roots[c]
        self.roots[a] = c                
        return c
    
    def add(self, a, b):
        a = self.root(a)
        b = self.root(b)
        if self.sizes[a] < self.sizes[b]:
            a, b = b, a
        self.roots[b] = a            
        self.sizes[a] += self.sizes[b]
            
class Solution:
    def findLatestStep(self, arr: List[int], M: int) -> int:
        uf = UnionFind(len(arr))
        m = [0 for i in range(len(arr))]
        good = set()
        day = 1
        result = -1            
        for a in arr:
            a -= 1
            m[a] = 1
            
            if a > 0 and m[a-1] == 1:
                if uf.root(a-1) in good:
                    good.remove(uf.root(a-1))
                uf.add(a, a-1)
            if a < len(arr)-1 and m[a+1] == 1:
                if uf.root(a+1) in good:
                    good.remove(uf.root(a+1))
                uf.add(a, a+1)
            if uf.sizes[uf.root(a)] == M:
                good.add(uf.root(a))
            if good:
                result = day
            day += 1                
        return result                
            



from sortedcontainers import SortedList
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        n = len(arr)
        if m == n:
            return n
        sl = SortedList([0, n+1])
        for i in range(n-1, -1, -1):
            a = arr[i]
            up = sl.bisect(a)
            if up != len(sl) and sl[up] - a - 1 == m:
                return i
            lp = up - 1
            if lp >= 0 and a - sl[lp] - 1 == m:
                return i
            sl.add(a)
        return -1
from bisect import bisect_left

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        res = -2
        
        groups = []
        n = len(arr)
        
        m_cnt = 0
        for step, x in enumerate(arr):
            k = groups
            if not groups:
                groups.append([x,x])
                if m == 1:
                    m_cnt += 1
            else:
                gn = len(groups)
                left = x
                right = x
                idx = bisect_left(groups, [x,x])
                #print(step, x, idx, groups)

                if idx < gn:
                    if groups[idx][0] == x + 1:
                        right = groups[idx][1]
                        if groups[idx][1] - groups[idx][0] + 1 == m:
                            m_cnt -= 1
                        groups.pop(idx)
                        

                if idx - 1 >= 0:
                    if groups[idx-1][1] == x - 1:
                        left = groups[idx-1][0]
                        if groups[idx-1][1] - groups[idx-1][0] + 1 == m:
                            m_cnt -= 1
                        groups.pop(idx-1)
                        idx -= 1
                
                groups.insert(idx, [left, right])
                if right - left + 1 == m:
                    m_cnt += 1
                
            if m_cnt:
                res = step
            #print(x, groups)
        return res + 1

# class UnionFind:
#     def __init__(self, n):
#         self.parent = {}
#         self.rank = [0] * (n+1)
#         self.group_size = defaultdict(list)
    
#     def find(self, x):
#         if x not in self.parent:
#             self.parent[x] = x
#             self.rank[x] = 1
#             self.group_size[1].append(x)

class UnionFind:
        def __init__(self, m, n):
            self.m = m
            self.parents = [i for i in range(n+1)]
            # self.ranks = [1 for _ in range(n)]
            self.group_size = defaultdict(set)
            # self.group_size[1] = set([i+1 for i in range(n)])
            self.sizes = defaultdict(int)
            # for i in range(n):
            #     self.sizes[i+1] = 1

        def find(self, x):
            if self.parents[x]!=x:
                self.parents[x] = self.find(self.parents[x])
            return self.parents[x]

        def union(self, x, y):
            root_x, root_y = self.find(x), self.find(y)
            # print("x", x ,"y", y)
            # print("root_x", root_x ,"root_y", root_y)
            self.parents[root_x] = root_y
            size_root_x = self.sizes[root_x]
            self.sizes[root_x] = 0
            self.group_size[size_root_x].remove(root_x)

            size_root_y = self.sizes[root_y]
            self.group_size[size_root_y].remove(root_y)
            self.sizes[root_y] = size_root_y + size_root_x
            self.group_size[self.sizes[root_y]].add(root_y)
            
            
            # print("len(self.group_size[self.m])", len(self.group_size[self.m]))
            if len(self.group_size[self.m])>0:
                return True
            else:
                return False
            

class Solution:
    ## my own solution: union find
    ## mapping between sizes and positions
    def findLatestStep(self, arr: List[int], m: int) -> int:
        n = len(arr)
        uf = UnionFind(m, n)
        # print(uf.group_size)
        # print(uf.sizes)
        seen = set()
        res = -1
        for idx, x in enumerate(arr):
            seen.add(x)
            uf.sizes[x] = 1
            uf.group_size[1].add(x)
            if x-1 in seen:
                uf.union(x, x-1)
                # if len(uf.group_size[m])>0:
                #     res = idx+1
            if x+1 in seen:        
                uf.union(x+1, x)
                
            if len(uf.group_size[m])>0:
                res = idx+1
            # print("uf.group_size", uf.group_size)
            # print("uf.sizes", uf.sizes)
        return res
            

class UF:
    def __init__(self):
        self.p = {}
        self.m = {}

    def find_make_set(self, x):
        if not x in self.p:
            self.p[x] = None
            self.m[x] = 1
            return x
        
        if self.p[x] is None:
            return x
        return self.find_make_set(self.p[x])
    
    def union(self, a, b):
        repa = self.find_make_set(a)
        repb = self.find_make_set(b)
        if repa != repb:
            self.p[repb] = repa
            self.m[repa] += self.m[repb]
        return repa



class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:

        n = len(arr)
        ones = {}
        ans = -1
        uf = UF()

        import collections
        cnt = collections.defaultdict(int)
        for i, a in enumerate(arr):
            ones[a] = 1
            uf.find_make_set(a)
            cnt[1] += 1

            if a - 1 >= 1 and a-1 in ones:
                rep = uf.find_make_set(a-1)
                cnt[uf.m[rep]] -= 1

                cnt[1] -= 1

                rep = uf.union(a-1, a)
                cnt[uf.m[rep]] += 1

            if a + 1 <= n and a+1 in ones:
                rep = uf.find_make_set(a+1)
                cnt[uf.m[rep]] -= 1

                rep = uf.find_make_set(a)
                cnt[uf.m[rep]] -= 1

                rep = uf.union(a+1, a)
                cnt[uf.m[rep]] += 1

            if cnt[m] > 0:
                ans = i + 1

        return ans


class UnionFindSet:
    def __init__(self, n):
        self.parents = list(range(n))
        self.ranks = [0] * n
        
    def find(self, u):
        if u != self.parents[u]:
            self.parents[u] = self.find(self.parents[u])
        return self.parents[u]
    
    def union(self, u, v):
        pu, pv = self.find(u), self.find(v)
        if pu == pv:
            return False
        if self.ranks[pu] > self.ranks[pv]:
            self.parents[pv] = pu
            self.ranks[pu] += self.ranks[pv]
        elif self.ranks[pv] > self.ranks[pu]:
            self.parents[pu] = pv
            self.ranks[pv] += self.ranks[pu]
        else:
            self.parents[pu] = pv
            self.ranks[pv] += self.ranks[pu]
        return True
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        n, ans = len(arr), -1
        uf = UnionFindSet(n)
        
        for step, i in enumerate(arr):
            i -= 1
            uf.ranks[i] = 1
            for j in (i - 1, i + 1):
                if 0 <= j < n:
                    if uf.ranks[uf.find(j)] == m:
                        ans = step
                    if uf.ranks[j]:
                        uf.union(i, j)
        
        for i in range(n):
            if uf.ranks[uf.find(i)] == m:
                return n
            
        return ans
    
        d = set([0,len(arr) + 1])
        if m == len(arr): return m
        for i in range(len(arr)-1,-1,-1):
            if arr[i] - m-1 in d:
                exit = True
                for j in range(arr[i] -m,arr[i]):
                    if j in d:
                        exit = False
                        break
                if exit:
                    return i
            if arr[i] + m+1 in d:
                exit = True
                for j in range(arr[i]+1,arr[i]+m+1):
                    if j in d:
                        exit = False
                        break
                if exit:
                    return i
            d.add(arr[i])
        
        return -1
                

class UnionFindSet:
    def __init__(self, n):
        self.parents = list(range(n))
        self.ranks = [0] * n

    def find(self, u):
        if u != self.parents[u]:
            self.parents[u] = self.find(self.parents[u])
        return self.parents[u]

    def union(self, u, v):
        pu, pv = self.find(u), self.find(v)
        if pu == pv:
            return False
        if self.ranks[pu] > self.ranks[pv]:
            self.parents[pv] = pu
            self.ranks[pu] += self.ranks[pv]
        elif self.ranks[pv] > self.ranks[pu]:
            self.parents[pu] = pv
            self.ranks[pv] += self.ranks[pu]
        else:
            self.parents[pu] = pv
            self.ranks[pv] += self.ranks[pu]
        return True

class Solution:

    def findLatestStep(self, arr: List[int], m: int) -> int:
        
        '''
        length_dict = {}
        
        length_edge = [0]*len(arr)
        
        result = -1
        
        for i in range(len(arr)):
            index = arr[i] - 1
            
            left_length = 0
            right_length = 0
            if index>0:
                left_length = length_edge[index - 1]
            if index<len(arr)-1:
                right_length = length_edge[index + 1]
            length_edge[index+right_length] = 1 + left_length + right_length
            length_edge[index-left_length] = 1 + left_length + right_length
            
            if left_length in length_dict:
                length_dict[left_length] -= 1
            if right_length in length_dict:
                length_dict[right_length] -= 1
            if 1 + left_length + right_length not in length_dict:
                length_dict[1 + left_length + right_length] = 0
            length_dict[1 + left_length + right_length] += 1
            
            if m in length_dict and length_dict[m]>0:
                result = i + 1
        return result
        '''
        n, ans = len(arr), -1
        uf = UnionFindSet(n)
        
        groups = {}
        groups[0] = len(arr)
        
        result = -1
        for step, i in enumerate(arr):
            i -= 1
            uf.ranks[i] = 1
            
            for j in (i - 1, i + 1):
                if 0 <= j < n:
                    '''
                    if uf.ranks[uf.find(j)] == m:
                        ans = step
                    '''
                    groups[uf.ranks[uf.find(j)]] -= 1
                    if uf.ranks[j]:
                        uf.union(i, j)
            
            if uf.ranks[uf.find(i)] == m:
                ans = step + 1
            group = uf.ranks[uf.find(i)]
            if group not in groups:
                groups[group] = 0
            groups[group] += 1
            if m in groups and groups[m]>0:
                result = step + 1
        '''
        for i in range(n):
            if uf.ranks[uf.find(i)] == m:
                return n
        return ans
        '''
        
        return result
            

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        dsu = DSU()
        ans = -1
        visited = set()
        previous_index = []
        for i, num in enumerate(arr):
            visited.add(num)
            if num - 1 in visited:
                dsu.union(num, num - 1)
            if num + 1 in visited:
                dsu.union(num, num + 1)
            
            current_index = [i for i in previous_index if dsu.getCount(i) == m]
            previous_index = current_index
            
            if dsu.getCount(num) == m:
                current_index.append(num)
            
            if previous_index:
                ans = i + 1
                
        return ans
            

class DSU:
    def __init__(self):
        self.father = {}
        self.count = {}
    
    def find(self, a):
        self.father.setdefault(a, a)
        self.count.setdefault(a, 1)
        if a != self.father[a]:
            self.father[a] = self.find(self.father[a])
        return self.father[a]
    
    def union(self, a, b):
        _a = self.find(a)
        _b = self.find(b)
        if _a != _b:
            self.father[_a] = _b
            self.count[_b] += self.count[_a]
    
    def getCount(self, a):
        return self.count[self.find(a)]
            

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        n = len(arr)
        size = [0] * (n + 2)
        size_count = [0] * (n + 1)
        res = -1
        for step, num in enumerate(arr):
            left, right = size[num - 1], size[num + 1]
            size[num] = size[num - left] = size[num + right] = left + right + 1
            size_count[left] -= 1
            size_count[right] -= 1
            size_count[size[num]] += 1
            if size_count[m]:
                res = max(res, step + 1)
        return res
from sortedcontainers import SortedList
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        sizes = Counter((len(arr),))
        groups = SortedList([[1, len(arr)]])
        for i in range(len(arr)-1, -1, -1):
            if m in sizes:
                return i + 1
            n = arr[i]
            j = groups.bisect_left([n, n]) 
            if j == len(groups) or j > 0 and groups[j-1][1] >= n:
                j -= 1
            h, t = groups.pop(j)
            sizes[t - h + 1] -= 1
            if h < n: 
                groups.add([h, n-1])
                sizes[n-1 - h + 1] += 1 
            if t > n:    
                groups.add([n+1, t])
                sizes[t - n - 1 + 1] += 1 
        return -1        
class UnionFind:
    def __init__(self):
        self.parents = defaultdict(lambda:-1)
        self.ranks = defaultdict(lambda:1)
    def join(self, a, b):
        pa, pb = self.find(a), self.find(b)
        if pa == pb:
            return
        if self.ranks[pa] > self.ranks[pb]:
            self.parents[pb] = pa
            self.ranks[pa] += self.ranks[pb]
        else:
            self.parents[pa] = pb
            self.ranks[pb] += self.ranks[pa]
    def find(self, a):
        if self.parents[a] == -1:
            return a
        self.parents[a] = self.find(self.parents[a])
        return self.parents[a]

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        uf = UnionFind()
        cnt=0
        ret=-1
        lst = [0]*len(arr)
        for idx,i in enumerate(arr):
            i-=1
            lst[i]=1
            if i-1>=0 and lst[i-1]:
                if uf.ranks[uf.find(i-1)]==m:
                    cnt-=1
                uf.join(i,i-1)
            if i+1<len(lst) and lst[i+1]:
                if uf.ranks[uf.find(i+1)]==m:
                    cnt-=1
                uf.join(i,i+1)
            if uf.ranks[uf.find(i)]==m:
                cnt+=1
            if cnt>0:
                ret=idx+1
        return ret
class UnionFindSet:
    def __init__(self, n):
        self.parents = list(range(n))
        self.ranks = [0] * n
        
    def find(self, u):
        if u != self.parents[u]:
            self.parents[u] = self.find(self.parents[u])
        return self.parents[u]
    
    def union(self, u, v):
        pu, pv = self.find(u), self.find(v)
        if pu == pv:
            return False
        if self.ranks[pu] > self.ranks[pv]:
            self.parents[pv] = pu
            self.ranks[pu] += self.ranks[pv]
        elif self.ranks[pv] > self.ranks[pu]:
            self.parents[pu] = pv
            self.ranks[pv] += self.ranks[pu]
        else:
            self.parents[pu] = pv
            self.ranks[pv] += self.ranks[pu]
        return True

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        n, ans = len(arr), -1
        uf = UnionFindSet(n)
        
        for step, i in enumerate(arr):
            i -= 1
            uf.ranks[i] = 1
            for j in (i - 1, i + 1):
                if 0 <= j < n:
                    if uf.ranks[uf.find(j)] == m:
                        ans = step
                    if uf.ranks[j]:
                        uf.union(i, j)
        
        for i in range(n):
            if uf.ranks[uf.find(i)] == m:
                return n
            
        return ans
                
                
                
                
                

class Node:
    def __init__(self, left, right):
        
        self.leftvalue = left
        self.rightvalue = right        
        self.nextleft = None
        self.nextright = None
        
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        
        n = len(arr)
        
        root = Node(1, n)
        
        chk = False
                
        def dfs(u, node):
            
            # nonlocal ans
            nonlocal chk
            nonlocal m
            
            # print (u, node.leftvalue, node.rightvalue)
            
            if chk:
                return
            
            if not node.nextleft and not node.nextright:
                
                # print ('x')
                
                if u == node.leftvalue:
                    node.leftvalue += 1
                    
                    if node.rightvalue - node.leftvalue + 1 == m:
                        chk = True
                    
                    return
                
                if u == node.rightvalue:
                    node.rightvalue -= 1
                    
                    if node.rightvalue - node.leftvalue + 1 == m:
                        chk = True
                    
                    return
                
                
                if u - node.leftvalue == m or node.rightvalue - u == m:
                    chk = True
                    return
                
                node.nextleft = Node(node.leftvalue, u - 1)
                node.nextright = Node(u + 1, node.rightvalue)

                
                return
            
            
            if node.nextleft.leftvalue <= u <= node.nextleft.rightvalue:
                dfs(u, node.nextleft)
            elif node.nextright.leftvalue <= u <= node.nextright.rightvalue:
                dfs(u, node.nextright)
        
        
        if m == len(arr):
            return n
        
        # if arr == sorted(arr):
        #     return m
        
        ans = n
        for i in range(len(arr) - 1, -1, -1):
            
            dfs(arr[i], root)
            ans -= 1
            
            if chk:
                return ans
            
        return -1
                

from sortedcontainers import SortedSet
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        n = len(arr)
        # 11111 -> 11101
        # [(start, end)]  , (1,5) => (1,3), 4, (5,5)
        ans = n
        if m == n:
            return ans
        ss = SortedSet([(1, n)])
        s0 = set()
        # print(ss)
        # 1110 1111
        
        for i in range(n-1, -1, -1):
            v = arr[i]
            i_itv = ss.bisect_right((v,n)) - 1
            # print("idx:", i_itv)
            start, end = ss[i_itv]
            # print(start, end, "v:",v)
            if (v-start == m) or (end-v == m):
                return i
            ss.discard((start,end))
            if v-1 >= start:
                ss.add((start,v-1))
            if end >= v+1:
                ss.add((v+1,end))
            # print(ss)
        
        return -1
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        if len(arr) == m:
            return m
        
        intervals = [[1, len(arr)]]
        
        for i in range(len(arr)-1, -1, -1):
            remove = arr[i]
            l, r = 0, len(intervals)-1
            while l <= r:
                mid = l + (r - l)//2
                if intervals[mid][0] > remove:
                    r = mid - 1
                else:
                    l = mid + 1
            interval = list(intervals[r])
            if interval[0] == remove:
                intervals[r][0] = intervals[r][0] + 1
                if intervals[r][1] - intervals[r][0] + 1 == m:
                    return i
            elif interval[1] == remove:
                intervals[r][1] = intervals[r][1] - 1
                if intervals[r][1] - intervals[r][0] + 1 == m:
                    return i
            else:
                intervals.insert(r, list(interval))
                intervals[r][1] = remove - 1
                intervals[r+1][0] = remove + 1
                if (intervals[r][1] - intervals[r][0] + 1 == m) or (intervals[r+1][1] - intervals[r+1][0] + 1 == m):
                    return i
        return -1
                
            
            

class UnionFindSet:
    def __init__(self, n):
        self.parents = list(range(n))
        self.ranks = [0] * n
        
    def find(self, u):
        if u != self.parents[u]:
            self.parents[u] = self.find(self.parents[u])
        return self.parents[u]
    
    def union(self, u, v):
        pu, pv = self.find(u), self.find(v)
        if pu == pv:
            return False
        if self.ranks[pu] > self.ranks[pv]:
            self.parents[pv] = pu
            self.ranks[pu] += self.ranks[pv]
        elif self.ranks[pv] > self.ranks[pu]:
            self.parents[pu] = pv
            self.ranks[pv] += self.ranks[pu]
        else:
            self.parents[pu] = pv
            self.ranks[pv] += self.ranks[pu]
        return True

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        n, ans = len(arr), -1
        uf = UnionFindSet(n)
        
        if m == n:
            return m
        for step, i in enumerate(arr):
            i -= 1
            uf.ranks[i] = 1
            for j in (i - 1, i + 1):
                if 0 <= j < n:
                    if uf.ranks[uf.find(j)] == m:
                        ans = step
                    if uf.ranks[j]:
                        uf.union(i, j)
            if uf.ranks[i] == m == 1 and uf.parents[i] == i:
                ans = step + 1
            # print(step, i, uf.ranks, ans)
        return ans
class Union_find:
    def __init__(self, MAX: int,target):
        self.fa = [i for i in range(MAX)]
        self.cnt = [1 for _ in range(MAX)]
        self.exist = 0
        self.target = target
        self.root_map = collections.defaultdict(set)

    def find(self, u: int) -> int:
        if self.fa[u] == u:
            return u

        self.fa[u] = self.find(self.fa[u])
        return self.fa[u]

    def union(self, u: int, v: int):
        u, v = self.find(u), self.find(v)
        if u == v:
            return None

        if self.cnt[u] < self.cnt[v]:
            u, v = v, u
        vn = int(self.cnt[v])
        un = int(self.cnt[u])
        try:
            self.root_map[vn].remove(v)
        except:
            pass
        self.cnt[u] = vn + un
        try:
            self.root_map[un].remove(u)
        except:
            pass
        self.root_map[vn+un].add(u)
        self.fa[v] = u
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        ct = 0
        l = [0 for i in arr]
        n = 0
        res = -1
        uf = Union_find(len(arr),m)
        for i in arr:
            l[i-1] = 1
            ct += 1
            flag = False
            if i-2>-1 and l[i-2] == 1:
                uf.union(i-1,i-2)
                flag = True
            if i<len(arr) and l[i] == 1:
                uf.union(i-1,i)
                flag = True
            if not flag:
                uf.root_map[1].add(i-1)
            if len(uf.root_map[m])>0:
                res = ct
        return res
import collections
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        dic={}
        n=len(arr)
        for i in range(1,n+1):
            dic[i]=(-1,-1,0)
        res=-1    
        memo=collections.defaultdict(int)
        for i in range(n):
            left=arr[i]
            right=arr[i]
            l=1
            if arr[i]+1<=n and dic[arr[i]+1][2]!=0:
                right=dic[arr[i]+1][1]
                l+=dic[arr[i]+1][2]
                memo[dic[arr[i]+1][2]]-=1
                
            if arr[i]-1>=1 and dic[arr[i]-1][2]!=0:
                left=dic[arr[i]-1][0]
                l+=dic[arr[i]-1][2]
                memo[dic[arr[i]-1][2]]-=1
                
            for x in [left, right]:
                dic[x]=(left, right, l)  
            memo[l]+=1
            if memo[m]>0: res=i+1
            
        return res
        
                
                
                
            
       
            
            

from sortedcontainers import SortedDict
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        n = len(arr)
        a = [0] * n
        for i in arr:
            a[i-1] = 1
        last = -1
        lastcnt = 0
        parentcnts = SortedDict()
        for i, val in enumerate(a):
            if val == 1:
                if i == 0 or a[i-1] == 0:
                    last = i
                    lastcnt = 0
                    parentcnts[i] = 0
                lastcnt += 1
                parentcnts[last] += 1
        for key in parentcnts:
            if parentcnts[key] == m:
                return n
        
        for x in range(n-1, -1, -1):
            # print(parentcnts)
            ind = arr[x] - 1
            leftone, rightone = True, True
            if ind == 0 or a[ind-1] == 0:
                leftone = False
            if ind == n-1 or a[ind+1] == 0:
                rightone = False
            if not leftone and not rightone:
                parentcnts.pop(ind)
                continue
            if not leftone:
                parentcnts[ind + 1] = parentcnts[ind] - 1
                parentcnts.pop(ind)
                if m == parentcnts[ind + 1]:
                    return x
                continue
            
            ins_ind = parentcnts.peekitem( parentcnts.bisect_right(ind) - 1 )[0]
            if not rightone:
                parentcnts[ins_ind] -= 1
                if m == parentcnts[ins_ind]:
                    return x
                continue
            
            parentcnts[ind + 1] = parentcnts[ins_ind] - (ind - ins_ind + 1)
            parentcnts[ins_ind] = ind - ins_ind
            if m == parentcnts[ind + 1] or m == parentcnts[ins_ind]:
                return x
        return -1
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        self.m = m
        n = len(arr)
        self.parents = [i for i in range(n + 1)]
        self.grpcounts = {}
        self.countMgrps = 0
        
        last = -1
        for step, pos in enumerate(arr):
            self.grpcounts[pos] = 1
            if m == 1:
                self.countMgrps += 1
            
            if pos + 1 <= n and self.find(pos + 1) in self.grpcounts:
                self.union(pos, pos + 1)
            if pos - 1 > 0 and self.find(pos - 1) in self.grpcounts:
                self.union(pos, pos - 1)
            
            # print(self.countMgrps)
            if self.countMgrps > 0:
                last = step + 1
        
        return last
        
        
    def find(self, pos):
        path = []
        while pos != self.parents[pos]:
            path.append(pos)
            pos = self.parents[pos]
        
        for p in path:
            self.parents[p] = pos
        
        return pos
    
    
    def union(self, a, b):
        p1 = self.find(a)
        p2 = self.find(b)
        if p1 != p2:
            self.parents[p1] = p2
            if self.grpcounts[p1] == self.m:
                self.countMgrps -= 1
            if self.grpcounts[p2] == self.m:
                self.countMgrps -= 1
            
            self.grpcounts[p2] += self.grpcounts[p1]
            if self.grpcounts[p2] == self.m:
                self.countMgrps += 1
            
            del self.grpcounts[p1]

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        n = len(arr)
        p = list(range(n))
        size = [0]*n

        def find(x):
            if x!=p[x]:
                p[x] = find(p[x])
            return p[x]

        def union(x,y):
            px,py = find(x),find(y)
            if px == py:
                return False
            if size[px]>size[py]:
                p[py] = px
                size[px]+=size[py]
            else:
                p[px] =py
                size[py] += size[px]
            return True

        if m == len(arr):
            return m
        ans = -1
        for step,i in enumerate(arr):
            i -= 1
            
            for j in range(i-1,i+2):
                if 0<=j<n:
                    if size[find(j)]==m:
                        ans = step
            size[i] = 1
            for j in range(i-1,i+2):
                if 0<=j<n:
                    if size[j]:
                        union(i,j)
        return ans
    
    # 4 1 3
    # 4 5 8

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        if m == len(arr): return m
        groups = {}
        result = -1
        for i, a in enumerate(arr):
            b = a - 1
            if b in groups:
                l = groups.pop(b)
                groups[a] = groups[b - l + 1] = l + 1
            else:
                groups[a] = 1
            if groups[a] - 1 == m:
                result = i
            c = a + 1
            if c in groups:
                l = groups.pop(a)
                r = groups.pop(c)
                groups[c + r - 1] = groups[a - l + 1] = l + r
                if r == m:
                    result = i
        return result
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        
        if m == len(arr):
            return m
        
        def find(vals, index):
            if vals[index] != -1:
                vals[index] = find(vals, vals[index])
                return vals[index]
            else:
                return index
        
        # def union(vals, a, b):
        #     root_a = find(vals, a)
        #     root_b = find(vals, b)
        #     if root_a < root_b:
        #         vals[root_b] = root_a
        #     else:
        #         vals[root_a] = root_b
        
        def unionStart(a, b):
            root_a = find(start, a)
            root_b = find(start, b)
            if root_a < root_b:
                start[root_b] = root_a
            else:
                start[root_a] = root_b
        
        def unionEnd(a, b):
            root_a = find(end, a)
            root_b = find(end, b)
            if root_a > root_b:
                end[root_b] = root_a
            else:
                end[root_a] = root_b
        
        def getLength(index):
            start_curr = find(start, index)
            end_curr = find(end, index)
            return end_curr - start_curr + 1
            
        res = -1
        nums = [0 for i in range(len(arr))]
        start = [-1 for i in range(len(arr))]
        end = [-1 for i in range(len(arr))]
        mem = dict() # start, length
        lengths = collections.Counter()
        for i in range(len(arr)):
            index = arr[i] - 1
            # print(index)
            nums[index] += 1
            # found group of m
            # check left
            if index > 0 and nums[index - 1] == 1:
                # union find start
                old_length = getLength(index - 1)
                lengths[old_length] -= 1
                unionStart(index - 1, index)
                unionEnd(index - 1, index)
                
            # check right
            if index < len(arr) - 1 and nums[index + 1] == 1:
                old_length = getLength(index + 1)
                lengths[old_length] -= 1
                unionStart(index, index + 1)
                unionEnd(index, index + 1)
                
            start_curr = find(start, index)
            end_curr = find(end, index)
            # print(start)
            # print(end, start_curr, end_curr)
            length = getLength(index)
            # plus 1
            lengths[length] += 1
            if lengths[m] > 0:
                res = i + 1
            # print(lengths)
        return res
from collections import Counter
class DSU:
    def __init__(self, n):
        self.dic = [i for i in range(n)]
    def find(self, n1):
        if self.dic[n1] != n1:
            self.dic[n1] = self.find(self.dic[n1])
        return self.dic[n1]
    def union(self, n1, n2):
        s1 = self.find(n1)
        s2 = self.find(n2)
        self.dic[s2] = s1
        
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        cnt = Counter()        
        area = Counter()
        fliped = [False for _ in arr]
        dsu = DSU(len(arr))        
        res = []
        for i, b in enumerate(arr):
            b -= 1
            fliped[b] = True             
            bl = 1
            if b > 0 and fliped[b - 1]:
                s = dsu.find(b - 1)
                sa = area[s] 
                cnt[sa] -= 1
                dsu.union(s, b)
                bl += sa
            if b < len(arr) - 1 and fliped[b + 1]:
                s = dsu.find(b + 1)
                sa = area[s] 
                cnt[sa] -= 1
                dsu.union(s, b)
                bl += sa     
            s = dsu.find(b)
            area[s] = bl
            cnt[bl] += 1
            if cnt[m] > 0:
                res.append(i)
        # print(res)
        return res[-1] + 1 if res else -1
            

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        import numpy as np
        def max_rolling1(a, window,axis =1):
            shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
            strides = a.strides + (a.strides[-1],)
            rolling = np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
            return np.max(rolling,axis=axis)
        
        n=len(arr)
        L=[0]*n
        for i in range(n):
            L[arr[i]-1]=i
            
        A =np.array(L)
        K = m
        RM=(max_rolling1(A,K))
        hh=-1
        if m==n:
            return n
        for i in range(n-m+1):
            temp =[L[x] for x in [i-1, i+m] if x in range(n)]
            
            if min(temp)>RM[i]:
                hh=max(hh, min(temp))
        return hh
class Union_Find():
    def __init__(self):
        self.father = {}
        self.count = collections.defaultdict(int)
    
    def find(self, a):
        if self.father[a] == a:
            return a
        self.father[a] = self.find(self.father[a])
        return self.father[a]
    
    def union(self, a, b):
        father_a = self.find(a)
        father_b = self.find(b)
        if father_a != father_b:
            self.father[father_b] = father_a
            self.count[father_a] += self.count[father_b]


class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        uf = Union_Find()
        result = -1
        for i in range(len(arr)):
            uf.father[arr[i]] = arr[i]
            uf.count[arr[i]] = 1
            if arr[i] - 1 in uf.father:
                if uf.count[uf.find(arr[i] - 1)] == m:
                    result = i
                uf.union(arr[i], arr[i] - 1)
            if arr[i] + 1 in uf.father:
                if uf.count[uf.find(arr[i] + 1)] == m:
                    result = i
                uf.union(arr[i], arr[i] + 1)
        n = len(arr)
        for i in range(n):
            if uf.count[uf.find(i + 1)] == m:
                return n
        return result

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        a = [x - 1 for x in arr]
        n = len(a)
        parent = list(range(n))
        size = [0] * n
        count = Counter()

        def find(u):
            if parent[u] != u:
                parent[u] = find(parent[u])
            return parent[u]

        def union(u, v):
            x, y = find(u), find(v)
            if x != y:
                parent[y] = x
                size[x] += size[y]

        res = -1
        bits = [0] * n

        for i, u in enumerate(a, 1):
            bits[u] = 1
            size[u] = 1
            count[1] += 1
            if u > 0 and bits[u - 1]:
                count[size[find(u - 1)]] -= 1
                union(u - 1, u)
            if u + 1 < n and bits[u + 1]:
                count[size[find(u + 1)]] -= 1
                union(u, u + 1)
            if size[find(u)] != 1:
                count[1] -= 1
                count[size[find(u)]] += 1
            if count[m] > 0:
                res = i
            # print(i, size)
            # print(i, count)

        return res
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        if m == len(arr):
            return m
        ends, starts = {}, {}
        last_found = -1
        
        for ind, val in enumerate(arr):
            left = val - 1 if val - 1 in ends else None
            right = val + 1 if val + 1 in starts else None
            
            if left and right:
                length1, length2 = left - ends[left], starts[right] - right
                if length1 == m - 1 or length2 == m - 1:
                    last_found = ind
                l, r = ends[left], starts[right]
                starts[l], ends[r] = r, l
                del ends[left]
                del starts[right]
            elif left:
                length = left - ends[left]
                if length == m - 1:
                    last_found = ind
                ends[val] = ends[left]
                starts[ends[left]] = val
                del ends[left]
            elif right:
                length = starts[right] - right
                if length == m - 1:
                    last_found = ind
                starts[val] = starts[right]
                ends[starts[right]] = val
                del starts[right]
            else:
                starts[val] = val
                ends[val] = val
        return last_found

from bisect import *
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        N = len(arr)
        arr = [i - 1 for i in arr]
        p = [i for i in range(len(arr))]
        gsize = [0 for _ in range(len(arr))]
        
        def fp(n):
            nonlocal p
            if n == p[n]:
                return p[n]
            else:
                p[n] = fp(p[n])
                return p[n]
            
        def gs(n):
            return gsize[fp(n)]
        
        ms = set()
        def uu(a, b):
            nonlocal ms
            pa = fp(a)
            pb = fp(b)
            
            
            if gs(pa) == m:
                ms.add(pa)
            if gs(pb) == m:
                ms.add(pb)
                
            if pa == pb:
                return
            try:
                ms.remove(pa)
            except:
                pass
            try:
                ms.remove(pb)
            except:
                pass
            
            gsize[pb] += gsize[pa]
            p[pa] = p[pb]
            if gs(pb) == m:
                ms.add(pb)
            
        
        filled = set()
        ans = -2
        for i, n in enumerate(arr):
            gsize[n] = 1
            uu(n, n)
            if n > 0 and n - 1 in filled:
                uu(n, n - 1)
                
                
            if n < N - 1 and n + 1 in filled:
                uu(n, n + 1)

            filled.add(n)
            if len(ms) > 0:
                ans = i
        return ans + 1
                
        

from collections import defaultdict
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        parent = {}
        rank = defaultdict(int)
        size = {}
        
        def find(x):
            if x != parent[x]:
                parent[x] = find(parent[x])
            return parent[x]
    
        def union(x, y):
            nx, ny = find(x), find(y)
            
            if nx == ny:
                return False
            
            if rank[nx] > rank[ny]:
                parent[ny] = nx
                size[nx] += size[ny]
                return size[nx]
            elif rank[nx] > rank[ny]:
                parent[nx] = ny
                size[ny] += size[nx]
                return size[ny]
            else:
                parent[nx] = ny
                rank[ny] += 1
                size[ny] += size[nx]
                return size[ny]
        
        seen = set()
        last = -1
        size_counts = Counter()
        
        for i, num in enumerate(arr):
            parent[num] = num
            size[num] = 1
            if num - 1 in seen and num + 1 in seen:
                size_counts[size[find(num - 1)]] -= 1
                size_counts[size[find(num + 1)]] -= 1
                union(num - 1, num)
                res = union(num + 1, num)
                size_counts[res] += 1
            elif num - 1 in seen:
                size_counts[size[find(num - 1)]] -= 1
                res = union(num - 1, num)
                size_counts[res] += 1
            elif num + 1 in seen:
                size_counts[size[find(num + 1)]] -= 1
                res = union(num + 1, num)
                size_counts[res] += 1
            else:
                size_counts[size[num]] += 1

            if m in size_counts and size_counts[m] > 0:
                last = i + 1
            # for _, l in parent.items():
            #     if size[find(l)] == m:
            #         last = i + 1

            seen.add(num)
        return last
        

import collections

class Union:
    
    def __init__(self):
        self.parent = {}
        self.rank = {}
        self.rank_count = collections.Counter()
        self.count = 0
        
    def add(self, pos):
        self.parent[pos] = pos
        self.rank[pos] = 1
        self.rank_count[1] += 1
        self.count += 1
    
    def find(self, pos): #recursively find parent
        if self.parent[pos] != pos:
            self.parent[pos] = self.find(self.parent[pos])
        return self.parent[pos]
    
    def unite(self, p, q):
        i, j = self.find(p), self.find(q)
        if i == j:
            return
        if self.rank[i] > self.rank[j]:
            i, j = j, i
        self.parent[i] = j # i is smaller tree, attach it to larger tree j with j as parent
        self.rank_count[self.rank[j]] -= 1
        self.rank_count[self.rank[i]] -= 1
        self.rank[j] += self.rank[i]
        self.rank_count[self.rank[j]] += 1
        self.count -= 1

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        res = -1
        bi = Union()
        for step, idx in enumerate(arr, 1):
            bi.add(idx)
            for neighbor in (idx + 1), (idx - 1):
                if neighbor in bi.parent:
                    bi.unite(idx, neighbor)
            if bi.rank_count.get(m, 0) > 0:  res = step
        return res

class DS:
    def __init__(self):
        self.intervals = {}
        self.par = {}
        self.all_vals = {}
        
    def find(self, x):
        if self.par[x] != x:
            self.par[x] = self.find(self.par[x])
        return self.par[x]
    
    def query(self, x):
        px = self.find(x)
        x, y = self.intervals[px]
        return y - x + 1
    
    def add(self, x, m):
        self.par[x] = x
        self.intervals[x] = [x,x]
        
        if 1 not in self.all_vals:
            self.all_vals[1] = 0
        self.all_vals[1] += 1
        
        if x+1 in self.par:
            px = self.find(x)
            py = self.find(x+1)
            y1,y2 = self.intervals[py]
            x1,x2 = self.intervals[px]
            self.par[py] = px
            x, y = min(x1,y1), max(x2,y2)
            self.intervals[px] = [x, y]
            
            self.all_vals[y2 - y1 + 1] -= 1
            self.all_vals[x2 - x1 + 1] -= 1
            if y - x + 1 not in self.all_vals:
                self.all_vals[y - x + 1] = 0
            self.all_vals[y - x + 1] += 1
            
        if x-1 in self.intervals:
            px = self.find(x)
            py = self.find(x-1)
            y1,y2 = self.intervals[py]
            x1,x2 = self.intervals[px]
            self.par[py] = px
            x, y = min(x1,y1), max(x2,y2)
            self.intervals[px] = [x, y]
            
            self.all_vals[y2 - y1 + 1] -= 1
            self.all_vals[x2 - x1 + 1] -= 1
            if y - x + 1 not in self.all_vals:
                self.all_vals[y - x + 1] = 0
            self.all_vals[y - x + 1] += 1
            
            
        return m in self.all_vals and self.all_vals[m] > 0

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        ds = DS()
        ans = -1
        for i,num in enumerate(arr):
            if ds.add(num, m):
                ans = i+1
        return ans
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        n = len(arr)
        uf = UnionFind(n + 1, m)
        ans = -1
        for i, a in enumerate(arr):
            a = arr[i]
            uf.add(a)
            if a > 1: uf.union(a, a - 1)
            if a < n: uf.union(a, a + 1)
            if uf.cnt > 0:
                ans = i + 1
        return ans
        
        
class UnionFind:
    def __init__(self, n, m):
        self.id = [-1 for _ in range(n)]
        self.size = [0 for _ in range(n)]
        self.cnt = 0
        self.m = m
        
    def add(self, i):
        self.id[i] = i
        self.size[i] = 1
        if self.get_size(i) == self.m:
            self.cnt += 1
        
    def find(self, i):
        if self.id[i] == -1:
            return -1
        root = i
        while root != self.id[root]:
            root = self.id[root]
        while root != i:
            j = self.id[i]
            self.id[i] = root
            i = j
        return root
    
    def union(self, i, j):
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i < 0 or root_j < 0 or root_i == root_j:
            return
        if self.get_size(i) == self.m:
            self.cnt -= 1
        if self.get_size(j) == self.m:
            self.cnt -= 1
        if self.size[root_i] < self.size[root_j]:
            self.id[root_i] = root_j
            self.size[root_j] += self.size[root_i]
        else:
            self.id[root_j] = root_i
            self.size[root_i] += self.size[root_j]
        if self.get_size(root_i) == self.m:
            self.cnt += 1
    
    def get_size(self, i):
        return self.size[self.find(i)]
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        counts = [0]*(len(arr)+1)
        root = [i for i in range(len(arr))]
        size = [0 for i in range(len(arr))]
        rank = [1 for i in range(len(arr))]
        def find(i):
            if root[i-1] != i-1:
                root[i-1] = find(root[i-1]+1)
            return root[i-1]
        def union(i,j):
            pi = find(i)
            pj = find(j)
            length = size[pi]+size[pj]
            if pi != pj:
                if rank[pi] <= rank[pj]:
                    root[pi] = pj
                    if rank[pi] == rank[pj]:
                        rank[pj] += 1
                else:
                    root[pj] = pi
                size[root[pi]] = length
        step = -1
        for i in range(len(arr)):
            size[arr[i]-1] += 1
            if arr[i] - 1 != 0 and size[find(arr[i]-1)] != 0:
                counts[size[find(arr[i]-1)]] -= 1
                union(arr[i]-1, arr[i])
            if arr[i] + 1 != len(arr)+1 and size[find(arr[i]+1)] != 0:
                counts[size[find(arr[i]+1)]] -= 1
                union(arr[i]+1, arr[i])
            counts[size[find(arr[i])]] += 1
            if counts[m] != 0:
                step = i+1
        return step
from sortedcontainers import SortedList
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        zeros = SortedList()
        zeros.add(0)
        zeros.add(len(arr) + 1)
        if len(arr) == m:
            return len(arr)
        for inverse_step, i in enumerate(arr[::-1], 1):
            head = zeros.bisect_left(i)
            # print(zeros, i, zeros.bisect_left(i), zeros.bisect_right(i))
            if  i - zeros[head - 1] - 1 == m:
                print((len(arr), inverse_step))
                return len(arr) - inverse_step
            
            tail = zeros.bisect_right(i)
            if  zeros[tail] - i - 1 == m:
                # print(len(arr), inverse_step)
                return len(arr) - inverse_step
            
            zeros.add(i)
        return -1
                
        
        
#         def is_length_existed(string):
#             last_zero = -1
#             for i, bit in string:
#                 if bit == 1:
#                     continue
#                 ones = i - last_zero
#                 if ones == m:
#                     return True
#                 last_zero = i
#             return False
        
#         def build(t):
#             string = [0] * len(arr)
#             for i in arr[:t]:
#                 string[i-1] = 1
        
#         p, q = 1, len(arr)
#         while p < q:
#             mid = (p + q) >> 1
#             string = build(mid)
#             if is_length_existed(string):
#                 p = mid
#             else:
#                 q = mid - 1
            
            
            
                    

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        n = len(arr)
        
        res = -1
        length = [0] * (n + 2)
        count = [0] * (n + 1)
        for i in range(n):
            b = arr[i]
            
            left, right = length[b - 1], length[b + 1]
            
            length[b] = length[b - left] = length[b + right] = left + right + 1
            count[left] -= 1
            count[right] -= 1
            count[length[b]] += 1
            
            if count[m] > 0:
                res = i + 1
        
        return res
            
            
            
            
        

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        uf = {}
        cnt = Counter()

        def find(x):
            if uf[x][0] != x:
                uf[x][0] = find(uf[x][0])[0]
            uf[x][1] = uf[uf[x][0]][1]
            return uf[x]

        def union(x, y):
            t1, t2 = find(y)[1], find(x)[1]
            cnt[t1] -= 1
            cnt[t2] -= 1
            uf[find(x)[0]][1] += t1
            uf[find(y)[0]][1] += t2
            uf[find(x)[0]][0] = find(y)[0]
            cnt[find(y)[1]] += 1

        seen = [0] * (len(arr) + 1)
        n = len(arr)
        ans = -1
        for i, a in enumerate(arr, 1):
            seen[a] = 1
            uf.setdefault(a, [a, 1])
            cnt[1] += 1
            if a > 1 and seen[a - 1]:
                union(a, a - 1)
            if a < n and seen[a + 1]:
                union(a, a + 1)
            if cnt[m]:
                ans = i
        return ans
                
                
                
                
                

class UnionFindSet:
    def __init__(self, n):
        self.parents = list(range(n))
        self.ranks = [0] * n
        
    def find(self, u):
        if u != self.parents[u]:
            self.parents[u] = self.find(self.parents[u])
        return self.parents[u]
    
    def union(self, u, v):
        pu, pv = self.find(u), self.find(v)
        if pu == pv:
            return False
        if self.ranks[pu] > self.ranks[pv]:
            self.parents[pv] = pu
            self.ranks[pu] += self.ranks[pv]
        elif self.ranks[pv] > self.ranks[pu]:
            self.parents[pu] = pv
            self.ranks[pv] += self.ranks[pu]
        else:
            self.parents[pu] = pv
            self.ranks[pv] += self.ranks[pu]
        return True

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        n, ans = len(arr), -1
        uf = UnionFindSet(n)
        
        for step, i in enumerate(arr):
            i -= 1
            uf.ranks[i] = 1
            for j in (i - 1, i + 1):
                if 0 <= j < n:
                    if uf.ranks[uf.find(j)] == m:
                        ans = step
                    if uf.ranks[j]:
                        uf.union(i, j)
        
        for i in range(n):
            if uf.ranks[uf.find(i)] == m:
                return n
            
        return ans


import bisect
import collections
from typing import List
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        intervals=[]
        lengths=collections.defaultdict(int)
        ans=-1
        for step,x in enumerate(arr):
            ind=bisect.bisect(intervals,[x,x])
            intervals.insert(ind,[x,x])
            lengths[1]+=1
            merge_left=merge_right=ind
            if ind-1>=0 and intervals[ind-1][1]==x-1:
                merge_left=ind-1
            if ind+1<len(intervals) and intervals[ind+1][0]==x+1:
                merge_right=ind+1
            # print(intervals)
            if merge_right>merge_left:
                for i in range(merge_left, merge_right + 1):
                    lengths[intervals[i][1] - intervals[i][0] + 1] -= 1
                lengths[intervals[merge_right][1]-intervals[merge_left][0]+1]+=1
                intervals[merge_left:merge_right+1]=[[intervals[merge_left][0],intervals[merge_right][1]]]
            if lengths[m]>0:
                ans=step+1
            # print(step, x)
            # print(intervals,lengths)


        return ans
class DSU:
    def __init__(self, n):
        self.parent = [i for i in range(n)]
        self.depth = [0 for i in range(n)]
        self.size = [0 for i in range(n)]
        self.count = collections.defaultdict(int)
    
    def findParent(self, n):
        if self.parent[n] != n:
            self.parent[n] = self.findParent(self.parent[n])
        return self.parent[n]

    def union(self, x, y):
        x_parent = self.findParent(x)
        y_parent = self.findParent(y)
        
        if x_parent == y_parent:
            return
            
        self.count[self.size[y_parent]] -= 1
        self.count[self.size[x_parent]] -= 1
        
        if self.depth[x_parent] >= self.depth[y_parent]:
            self.parent[y_parent] = x_parent
            self.size[x_parent] += self.size[y_parent]
            self.depth[x_parent] += (self.depth[x_parent] == self.depth[y_parent])
            
            self.count[self.size[x_parent]] += 1
            
        else:
            self.parent[x_parent] = y_parent
            self.size[y_parent] += self.size[x_parent]
            self.depth[y_parent] += 1
            
            self.count[self.size[y_parent]] += 1
        
        
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        last_step = -1
        bits = [0] * len(arr)
        dsu = DSU(len(arr))
        
        for i in range(len(arr)):
            idx = arr[i] - 1
            bits[idx] = 1
            
            dsu.size[idx] = 1
            dsu.count[1] += 1
            
            cur_size = 1
            
            if idx > 0:
                if bits[idx - 1]:
                    dsu.union(idx-1, idx)
            if idx < len(arr) - 1:
                if bits[idx + 1]:
                    dsu.union(idx+1, idx)
            if dsu.count[m] > 0:
                last_step = i+1
            
            # print(dsu.parent)
            # print(dsu.size)
            # print(dsu.count)
            # print()
        
        return last_step

class UF:
    def __init__(self, n):
        self.p = [i for i in range(n + 1)]
        self.counts = Counter()
        self.rank = [0 for i in range(n + 1)]
    
    def getParent(self, i):
        if self.p[i] == i:
            return i
        self.p[i] = self.getParent(self.p[i])
        return self.p[i]
    
    def set(self, i):
        self.counts[1] += 1
        self.rank[i] = 1
        
    def isSet(self, i):
        return 1 <= i < len(self.p) and self.rank[i] != 0
    
    def getCount(self, i):
        return self.counts[i]
    
    def connect(self, i, j):
        pi = self.getParent(i)
        pj = self.getParent(j)
        if pi != pj:
            self.p[pi] = pj
            ri, rj = self.rank[pi], self.rank[pj]
            self.counts[ri] -= 1
            self.counts[rj] -= 1
            self.counts[ri + rj] += 1
            self.rank[pj] = ri + rj

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        n = len(arr)
        uf = UF(n)
        
        result = -1
        for i, e in enumerate(arr, start = 1):
            uf.set(e)
            if uf.isSet(e-1):
                uf.connect(e-1, e)
            if uf.isSet(e+1):
                uf.connect(e, e+1)
            if uf.getCount(m) != 0:
                result = i
        
        return result

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        n = len(arr)
        p = list(range(n))
        size = [0]*n
 
        def find(x):
            if x!=p[x]:
                p[x] = find(p[x])
            return p[x]
 
        def union(x,y):
            px,py = find(x),find(y)
            if px == py:
                return False
            if size[px]>size[py]:
                p[py] = px
                size[px]+=size[py]
            else:
                p[px] =py
                size[py] += size[px]
            return True
 
        if m == len(arr):
            return m
        ans = -1
        for step,i in enumerate(arr):
            i -= 1
            for j in range(i-1,i+2):
                if 0<=j<n:
                    if size[find(j)]==m:
                        ans = step
            size[i] = 1
            for j in range(i-1,i+2):
                if 0<=j<n:
                    if size[j]:
                        union(i,j)
        return ans
class BinaryIndexedTree:
    ''' index from 1'''
    def __init__(self, n):
        self.n = n
        self.data = [0]*(n+1)
        #todo: init all 1
        for i in range(1, n+1):
            self.data[i]+=1
            tmp = i + (i&-i)
            if tmp <= n:
                self.data[i+(i&-i)] += self.data[i]
    def add(self, index, value):
        while(index<=self.n):
            self.data[index]+=value
            index += index & -index
    def prefix(self, index):
        res = 0
        while(index):
            res += self.data[index]
            index -= index & -index
        return res
    

class Solution:
    def findLatestStep(self, arr: List[int], k: int) -> int:
        n = len(arr)
        if k == n:
            return n
        bit = BinaryIndexedTree(n)
        for no, p in enumerate(arr[::-1]):
            bit.add(p, -1)
            if p-k >= 1:
                s1 = bit.prefix(p-1)
                s2 = bit.prefix(p-k-1)
                if s1 - s2 == k and (p-k-1==0 or bit.prefix(p-k-2)==s2):                    
                    return n-no-1
            if p+k <= n:
                s1 = bit.prefix(p)
                s2 = bit.prefix(p+k)
                if s2 - s1 == k and (p+k==n or bit.prefix(p+k+1)==s2):
                    print('b',p,s1,s2)
                    return n-no-1
        return -1
class UnionFind:
    def __init__(self, n):
        self.leaders = {}
        self.ranks = {}
        self.size = {}
        
    def add(self, x):
        if x in self.leaders:
            return
        self.leaders[x] = x
        self.ranks[x] = 1
        self.size[x] = 1
    
    def find(self, x):
        # p = x
        # while p != self._leaders[p]:
        #     p = self._leaders[p]
        # while x != p:
        #     self._leaders[x], x = p, self._leaders[x]
        # return p
        if self.leaders[x] != x:
            self.leaders[x] = self.find(self.leaders[x])
        return self.leaders[x]
    
    def union(self, x, y):
        p = self.find(x)
        q = self.find(y)
        if p == q: 
            return False
        if self.ranks[p] < self.ranks[q]:
            self.leaders[p] = q
            self.size[q] += self.size[p]
        elif self.ranks[p] > self.ranks[q]:
            self.leaders[q] = p
            self.size[p] += self.size[q]
        else:        
            self.leaders[q] = p
            self.ranks[p] += 1
            self.size[p] += self.size[q]
        return True
    
class Solution:
    def findLatestStep(self, arr, m):
        n = len(arr)
        if n == m:
            return m
        uf = UnionFind(n)
        state = 0
        res = -1
        for i, x in enumerate(arr):
            uf.add(x)
            state ^= (1 << x)
            if x - 1 >= 1 and state & (1 << (x - 1)) != 0:
                if uf.size[uf.find(x - 1)] == m:
                    res = i
                uf.union(x, x - 1)
            if x + 1 <= n and state & (1 << (x + 1)) != 0:
                if uf.size[uf.find(x + 1)] == m:
                    res = i
                uf.union(x, x + 1)
        return res


class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        # limit for array size  
        N = len(arr)+2;

        # Max size of tree  
        tree = [0] * (2 * N);  

        # function to build the tree  
        def build(arr) : 

            # insert leaf nodes in tree  
            for i in range(n) :  
                tree[n + i] = arr[i];  

            # build the tree by calculating parents  
            for i in range(n - 1, 0, -1) :  
                tree[i] = tree[i << 1] + tree[i << 1 | 1];  

        # function to update a tree node  
        def updateTreeNode(p, value) :  

            # set value at position p  
            tree[p + n] = value;  
            p = p + n;  

            # move upward and update parents  
            i = p; 

            while i > 1 : 

                tree[i >> 1] = tree[i] + tree[i ^ 1];  
                i >>= 1;  

        # function to get sum on interval [l, r)  
        def query(l, r) :  

            res = 0;  

            # loop to find the sum in the range  
            l += n; 
            r += n; 

            while l < r : 

                if (l & 1) : 
                    res += tree[l];  
                    l += 1

                if (r & 1) : 
                    r -= 1; 
                    res += tree[r];  

                l >>= 1; 
                r >>= 1

            return res;
        
        if m == len(arr):
            return len(arr)
        arr.reverse()
        n = len(arr)+2
        init = [0] * (n+1)
        init[0] = init[n-1] = 1
        build(init)
        for i, e in enumerate(arr):
            if 0 <= e - (m+1) and init[e - (m+1)] == 1 and query(e - m, e) == 0:
                return len(arr) - i - 1
            if e + (m+1) <= n-1 and init[e + (m+1)] == 1 and query(e, e + m + 1) == 0:
                return len(arr) - i - 1
            updateTreeNode(e, 1)
            init[e] = 1
        return -1

from typing import List
from heapq import heappush, heappop, heapify

class Node:
    def __init__(self, parent, value):
        self.parent = parent
        self.rank = 0
        self.size = 1
        self.value = value

class UnionFind:
    def __init__(self, nodes):
        self.subsets = [Node(i, v) for i, v in enumerate(nodes)]
        self.maxSubsetSize = 1

    def union(self, i, j):
        irep = self.find(i)
        jrep = self.find(j)
        if irep == jrep:
            return
        # union by rank
        if self.subsets[irep].rank > self.subsets[jrep].rank:
            self.subsets[jrep].parent = irep
            self.subsets[irep].size += self.subsets[jrep].size
        elif self.subsets[jrep].rank > self.subsets[irep].rank:
            self.subsets[irep].parent = jrep
            self.subsets[jrep].size += self.subsets[irep].size
        else:
            self.subsets[irep].parent = jrep
            self.subsets[jrep].rank += 1
            self.subsets[jrep].size += self.subsets[irep].size
        self.maxSubsetSize = max(self.maxSubsetSize, max(self.subsets[irep].size, self.subsets[jrep].size))

    def find(self, i):
        if self.subsets[i].parent != i:
            # path compression
            self.subsets[i].parent = self.find(self.subsets[i].parent)
        return self.subsets[i].parent

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        uf = UnionFind(list(range(len(arr))))
        lengthMsets = set()
        last_step = -1
        if m == 1:
            lengthMsets.add(arr[0]-1)
            last_step = 1
        numsets = set([arr[0]-1])
        for i in range(1, len(arr), 1):
            num = arr[i] - 1
            numsets.add(num)
            if num - 1 in numsets:
                if uf.find(num - 1) in lengthMsets:
                    lengthMsets.remove( uf.find(num - 1))
                uf.union(num-1, num)
            if num + 1 in numsets:
                if  uf.find(num + 1) in lengthMsets:
                    lengthMsets.remove(uf.find(num + 1))
                uf.union(num+1, num)
            if uf.subsets[uf.find(num)].size == m:
                lengthMsets.add(uf.find(num))
            if len(lengthMsets) > 0:
                last_step = i + 1
        return last_step

from sortedcontainers import SortedList
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        n = len(arr)
        if n == m:
            return n
        
        sl = SortedList([0, n+1])
        for i in range(n)[::-1]:
            k = arr[i]
            pos = sl.bisect(k)
            left, right = sl[pos - 1], sl[pos]
            if right - k - 1 == m or k - left - 1 == m:
                return i
            sl.add(k)
            
        return -1
class DisjointSet:
    def __init__(self, nums,p='left'):
        self.parent = {i: i for i in nums}
        self.height = {i: 0 for i in nums}
        self.count = len(nums)
        self.p=p

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        xRoot = self.find(x)
        yRoot = self.find(y)
        if xRoot == yRoot:
            return xRoot
        if self.p=='left':
            (lower, higher) = (
                xRoot, yRoot) if xRoot > yRoot else (yRoot, xRoot)
        else:
            (lower, higher) = (
                xRoot, yRoot) if xRoot < yRoot else (yRoot, xRoot)
        self.parent[lower] = higher
        if self.height[higher] == self.height[lower]:
            self.height[higher] += 1
        self.count -= 1
        return higher
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        n=len(arr)
        left=DisjointSet(list(range(len(arr))),'left')
        right=DisjointSet(list(range(len(arr))),'right')
        state=[0]*(len(arr))
        count=collections.defaultdict(int)
        latest=-1
        for step,idx in enumerate(arr):
            idx-=1
            state[idx]=1
            l=r=idx
            if idx>0 and state[idx-1]>0:
                left.union(idx,idx-1)
                right.union(idx,idx-1)
                
            if idx<n-1 and state[idx+1]>0:
                right.union(idx,idx+1)
                left.union(idx,idx+1)
            l=left.find(idx)
            r=right.find(idx)
            count[idx-l]-=1
            count[r-idx]-=1
            count[r-l+1]+=1
            if count[m]>=1:
                latest=step+1
            # print(idx,state,count,left.parent,right.parent,l,r)
        return latest

import collections
class Solution:
    def findLatestStep(self, placed, target):
        N = len(placed)
        A = [0] * N
        dsu = DSU(N)
        sizes = collections.Counter()
        ans = -1

        for i, x in enumerate(placed, 1):
            x -= 1
            A[x] = 1
            for y in (x - 1, x + 1):
                if 0 <= y < N and A[y]:
                    sizes[dsu.size(y)] -= 1
                    dsu.union(x, y)
            
            sizes[dsu.size(x)] += 1
            if sizes[target] > 0:
                ans = i
        
        return ans

class DSU:
    def __init__(self, N):
        self.par = list(range(N))
        self.sz = [1] * N

    def find(self, x):
        if self.par[x] != x:
            self.par[x] = self.find(self.par[x])
        return self.par[x]

    def union(self, x, y):
        xr, yr = self.find(x), self.find(y)
        if xr == yr:
            return False
        if self.sz[xr] < self.sz[yr]:
            xr, yr = yr, xr
        self.par[yr] = xr
        self.sz[xr] += self.sz[yr]
        self.sz[yr] = self.sz[xr]
        return True

    def size(self, x):
        return self.sz[self.find(x)]
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        if m == len(arr):
            return m
        vals = [0] * (len(arr) + 2)
        intervals = [0, len(arr) + 1]
        for idx in reversed(list(range(len(arr)))):
            bit = arr[idx]
            lo = 0
            hi = len(intervals)
            while lo < hi:
                mid = lo + (hi - lo) // 2
                if intervals[mid] >= bit:
                    hi = mid
                else:
                    lo = mid + 1
                    
            leftLen = bit - intervals[lo - 1] - 1
            rightLen = intervals[lo] - bit - 1
            if leftLen == m or rightLen == m:
                return idx
            if intervals[lo] - intervals[lo - 1] - 1 > m:
                intervals.insert(lo, bit)
        return -1

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        uf = {}
        seen = [0] * (len(arr) + 1)

        def find(x):
            uf.setdefault(x, x)
            if uf[x] != x:
                uf[x] = find(uf[x])
            return uf[x]

        def union(x, y):
            seen[find(y)] += seen[find(x)]
            uf[find(x)] = find(y)

        ans, n = -1, len(arr)
        for i, a in enumerate(arr, 1):
            seen[a] = 1
            for b in [a - 1, a + 1]:
                if 1 <= b <= n and seen[b]:
                    if seen[find(b)] == m:
                        ans = i - 1
                    union(a, b)
        for i in range(1, n + 1):
            if seen[find(i)] == m:
                ans = n

        return ans
                
                
                
                
                

class Solution:
    def findLatestStep(self, A: List[int], m: int) -> int:
        n = len(A)
        length = [0] * (n + 2)
        cnt = [0] * (n + 2)
        res = -1
        for i,v in enumerate(A):
            l = length[v - 1]
            r = length[v + 1]
            length[v] = length[v - l] = length[v + r] = l + r + 1
            cnt[l] -= 1
            cnt[r] -= 1
            cnt[l + r + 1] += 1
            if cnt[m] != 0:
                res = i + 1
        return res
from bisect import bisect
from collections import defaultdict

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        ans = -1
        count = defaultdict(int)
        tmp = []
        
        for i, a in enumerate(arr):
            index = bisect(tmp, a)
            add = 0

            if index != 0 and tmp[index-1] + 1 == a:
                count[tmp[index-1] - tmp[index-2] + 1] -= 1
                k = tmp.pop(index-1)
                tmp.insert(index-1, a)
                count[a - tmp[index-2] + 1] += 1
                add += 1
                # print('<', tmp, count)
            if index < len(tmp) - 1 and tmp[index] -1 == a:
                count[tmp[index+1] - tmp[index] + 1] -= 1
                k = tmp.pop(index)
                tmp.insert(index, a)
                count[tmp[index+1] - a + 1] += 1
                add += 1
                # print('>', tmp, count)
                
            if add == 0:
                tmp.insert(index, a)
                tmp.insert(index, a)
                count[1] += 1
            elif add == 2:
                count[a - tmp[index-2] + 1] -= 1
                count[tmp[index+1] - a + 1] -= 1
                count[tmp[index+1] - tmp[index-2] + 1] += 1
                tmp.pop(index)
                tmp.pop(index-1)
            
            # print(tmp, count)
            if count[m] > 0:
                ans = i + 1
        
        # print('-' * 20)
        # s = ['0' for _ in range(len(arr))]
        # def exist(s, m):
        #     p = ''.join(s).split('0')
        #     return any(len(x) == m for x in p)
        # for i, a in enumerate(arr):
        #     s[a-1] = '1'
        #     if exist(s, m):
        #         ans = i + 1
        return ans
from collections import Counter
class UnionFind():
    def __init__(self):
        self.uf, self.rank, self.size = {}, {}, {}
        self.roots = set()
        self.size_cnt = Counter()
        
    def add(self, x):
        if x not in self.uf:
            self.uf[x], self.rank[x], self.size[x] = x, 0, 1
            self.roots.add(x)
            self.size_cnt[1] += 1
        
    def find(self, x):
        self.add(x)
        if x != self.uf[x]:
            self.uf[x] = self.find(self.uf[x])
        return self.uf[x]

    def union(self, x, y):  
        xr, yr = self.find(x), self.find(y)
        if xr == yr: return
        self.size_cnt[self.size[xr]] -= 1
        self.size_cnt[self.size[yr]] -= 1
        if self.rank[xr] <= self.rank[yr]:
            self.uf[xr] = yr
            self.size[yr] += self.size[xr]
            self.size_cnt[self.size[yr]] += 1
            self.rank[yr] += (self.rank[xr] == self.rank[yr])
            self.roots.discard(xr)
        else:
            self.uf[yr] = xr
            self.size[xr] += self.size[yr]
            self.size_cnt[self.size[xr]] += 1
            self.roots.discard(yr)

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        uf, ans = UnionFind(), -1
        for step, i in enumerate(arr, 1):
            if i not in uf.uf: uf.add(i)
            if i - 1 in uf.uf: uf.union(i, i - 1)
            if i + 1 in uf.uf: uf.union(i, i + 1)
            if uf.size_cnt[m] > 0: ans = step
        return ans

from bisect import bisect
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        n = len(arr)
        if n == m:
            return n
        l = [0,n+1]
        for i in range(n-1,-1,-1):
            index = bisect(l,arr[i])
            front, end = l[index-1], l[index]
            if end - front <= m:
                continue
            if arr[i] - front == m+1 or end-arr[i] == m+1:
                return i
            else:
                l.insert(index,arr[i])
        return -1
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        from sortedcontainers import SortedList
        n = len(arr)
        if n == m:
            return n
        s = SortedList()
        s.add(0)
        s.add(n+1)
        for j in range(n-1, -1, -1):
            a = arr[j]
            s.add(a)
            i = s.bisect_left(a)
            le, ri = a - s[i-1] - 1, s[i+1] - a - 1
            if le == m or ri == m:
                return j
        return -1
# union find idea. But no need to implement union find, since only two sides can extend
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        count_m = 0
        n = len(arr)
        string = [0]*(n+1)
        res = -1
        step = 0
        for loc in arr:
            step += 1
            l,r = loc,loc
            string[loc] = loc
            if(loc-1>=0 and string[loc-1]!=0):
                # merge with left
                if((loc-1)-string[loc-1]+1==m):  # one sequence with length m disapper
                    count_m -= 1
                    
                string[r]= l = string[loc-1]
                string[l] = r 
            if(loc+1<=n and string[loc+1]!=0):
                # merge with right
                if(string[loc+1]-(loc+1)+1==m):  # one sequence with length m disapper
                    count_m -= 1
                
                string[l] = r = string[loc+1]
                string[r] = l 
            
            if(r-l+1==m):
                count_m += 1
            if(count_m>0):
                res = step
            #print(string)
        
        return res
class DSU:
    def __init__(self):
        self.N = 10**5 + 1
        self.parent = [i for i in range(self.N)]
        self.rank = [0 for _ in range(self.N)]
        self.comp = [1 for _ in range(self.N)]
        
    def find(self, i):
        if self.parent[i] == i: return i
        else: return self.find(self.parent[i])
        
    def union(self, i, j):
        x, y = self.find(i), self.find(j)
        
        if self.rank[x] > self.rank[y]:
            self.parent[y] = x
            self.comp[x] += self.comp[y]
            self.comp[y] = 0
            
        elif self.rank[x] < self.rank[y]:
            self.parent[x] = y
            self.comp[y] += self.comp[x]
            self.comp[x] = 0
            
        else:
            self.parent[y] = x
            self.comp[x] += self.comp[y]
            self.comp[y] = 0
            self.rank[x] += 1
        

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        N = 10**5 + 1
        nums = [False for _ in range(N+2)]
        d = DSU()
        
        cnt = [0 for _ in range(N+2)]
        ret = -1
        
        for index, a in enumerate(arr):
            cnt[d.comp[d.find(a)]] += 1
            
            if nums[a-1]: 
                cnt[d.comp[d.find(a-1)]] -= 1
                cnt[d.comp[d.find(a)]] -= 1
                d.union(a-1, a)
                cnt[d.comp[d.find(a)]] += 1

                
                
            if nums[a+1]:
                cnt[d.comp[d.find(a+1)]] -= 1
                cnt[d.comp[d.find(a)]] -= 1
                d.union(a, a+1)
                cnt[d.comp[d.find(a)]] += 1

                

            nums[a] = True
                
            if cnt[m]: ret = index+1
                
            #print(index, a, cnt)
                
        return ret
                
        

from collections import Counter
class UnionFind():
    def __init__(self):
        self.uf, self.rank, self.size, self.cnt = {}, {}, {}, Counter()
    def add(self, x):
        if x not in self.uf:
            self.uf[x], self.rank[x], self.size[x] = x, 0, 1
            self.cnt[1] += 1
    def find(self, x):
        self.add(x)
        if x != self.uf[x]:
            self.uf[x] = self.find(self.uf[x])
        return self.uf[x]
    def union(self, x, y):  
        xr, yr = self.find(x), self.find(y)
        if xr == yr: return
        self.cnt[self.size[xr]] -= 1
        self.cnt[self.size[yr]] -= 1
        if self.rank[xr] <= self.rank[yr]:
            self.uf[xr] = yr
            self.size[yr] += self.size[xr]
            self.cnt[self.size[yr]] += 1
            self.rank[yr] += (self.rank[xr] == self.rank[yr])
        else:
            self.uf[yr] = xr
            self.size[xr] += self.size[yr]
            self.cnt[self.size[xr]] += 1
            
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        uf = UnionFind()
        ans = -1
        for s, ind in enumerate(arr):
            if ind not in uf.uf:
                uf.add(ind)
            if ind - 1 in uf.uf:
                uf.union(ind, ind - 1)
            if ind + 1 in uf.uf:
                uf.union(ind, ind + 1)
            if uf.cnt[m] > 0:
                ans = s+1
        return ans
import bisect

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        n = len(arr)
        s = [0, n + 1]
        
        if n == m: return n
        
        for i, x in enumerate(reversed(arr)):
            j = bisect.bisect_right(s, x)
            s.insert(j, x)
            if m == x - s[j-1] - 1 or m == s[j + 1] - x - 1:
                return n - i - 1
            
        return -1
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        if m == len(arr): return m;
        length = [0] * (len(arr) + 2)
        res = -1
        for i, a in enumerate(arr):
            left, right = length[a - 1], length[a + 1]
            if left == m or right == m:
                res = i
            length[a - left] = length[a + right] = left + right + 1
        return res
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        if m == len(arr): return m
        length = [0] * (len(arr) + 2)
        res = -1
        for i, a in enumerate(arr):
            left, right = length[a - 1], length[a + 1]
            if left==m or right == m: res = i
            length[a] = length[a - left] = length[a + right] = left + right + 1
        return res
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        if m == len(arr):
            return m
        
        from bisect import bisect_left
        zeros = [0,len(arr)+1]
        ans = -1
        
        for i in reversed(list(range(len(arr)))):
            index = bisect_left(zeros,arr[i])
            zeros.insert(index,arr[i])
            if m in (zeros[index+1]-arr[i]-1,arr[i]-zeros[index-1]-1):
                return i
        return -1



class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        # limit for array size  
        N = len(arr)+2;

        # Max size of tree  
        tree = [0] * (2 * N);  

        # function to build the tree  
        def build(arr) : 

            # insert leaf nodes in tree  
            for i in range(n) :  
                tree[n + i] = arr[i];  

            # build the tree by calculating parents  
            for i in range(n - 1, 0, -1) :  
                tree[i] = tree[i << 1] + tree[i << 1 | 1];  

        # function to update a tree node  
        def updateTreeNode(p, value) :  

            # set value at position p  
            tree[p + n] = value;  
            p = p + n;  

            # move upward and update parents  
            i = p; 

            while i > 1 : 

                tree[i >> 1] = tree[i] + tree[i ^ 1];  
                i >>= 1;  

        # function to get sum on interval [l, r)  
        def query(l, r) :  

            res = 0;  

            # loop to find the sum in the range  
            l += n; 
            r += n; 

            while l < r : 

                if (l & 1) : 
                    res += tree[l];  
                    l += 1

                if (r & 1) : 
                    r -= 1; 
                    res += tree[r];  

                l >>= 1; 
                r >>= 1

            return res;
        
        if m == len(arr):
            return len(arr)
        
        n = len(arr)+2
        init = [0] * (n+1)
        init[0] = init[n-1] = 1
        build(init)
        for i in range(len(arr)-1, -1, -1):
            e = arr[i]
            print(e)
            if 0 <= e - (m+1) and init[e - (m+1)] == 1 and query(e - m, e) == 0:
                return i
            if e + (m+1) <= n-1 and init[e + (m+1)] == 1 and query(e, e + m + 1) == 0:
                return i
            updateTreeNode(e, 1)
            init[e] = 1
        return -1

from bisect import bisect_left
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        n = len(arr)
        if (m == n):
            return n
        r = []
        r.append(0)
        r.append(n + 1)
        
        for i in range(n - 1, -1, -1):
            j = bisect_left(r, arr[i])
            if (r[j] - arr[i] - 1 == m or arr[i] - r[j-1] - 1 == m):
                return i
            r.insert(j, arr[i])
        return -1
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        # expected group len is equal to len(arr)
        if(m == len(arr)):
            return m
        '''
        group_edge = [0] * (len(arr)+2)
        ans = 0 
        for i in range(0, len(arr)):
            print(arr[i])
            left=right=arr[i]
            if group_edge[right+1]>0: 
                right=group_edge[right+1]
            if group_edge[left-1]>0: 
                left=group_edge[left-1]
            group_edge[left], group_edge[right] = right, left
            if (right-arr[i]==m) or (arr[i]-left ==m): 
                ans=i
            print(group_edge)
        '''
        group_len = [0] * (len(arr) + 2)
        cnt_group_len = [0] * (len(arr) + 1)
        ans = -1
        for i in range(0, len(arr)):
            
            #print(arr[i])
            
            left_most = arr[i] - 1
            right_most = arr[i] + 1
            new_len = group_len[left_most] + group_len[right_most] + 1
            group_len[arr[i]] = new_len
            
            cnt_group_len[new_len] += 1
            cnt_group_len[group_len[left_most]] -= 1
            cnt_group_len[group_len[right_most]] -= 1
            
            #if(group_len[left_most] > 0):
            group_len[arr[i] - group_len[left_most]] = new_len
            
            #if(group_len[right_most] > 0):
            group_len[arr[i] + group_len[right_most]] = new_len
            
            #print(group_len)
            #print(cnt_group_len)
            
            if(cnt_group_len[m] > 0):
                ans = i + 1
            
        return ans
                

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        length = [0 for i in range(len(arr) + 2)]
        count = [0 for i in range(len(arr) + 1)]
        res = -1
        for i, v in enumerate(arr):
            left, right = length[v-1], length[v+1]
            length[v] = left + right + 1
            length[v-left] = left + right + 1
            length[v+right] = left + right + 1
            if count[left] > 0:
                count[left] -= 1
            if count[right] > 0:
                count[right] -= 1
            count[length[v]] += 1
            if count[m]:
                res = i + 1
        return res
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:   
        length = [0] * (len(arr)+2)
        count = [0]*(len(arr)+1)
        res = -1
        for i, pos in enumerate(arr):
            left, right = length[pos-1], length[pos+1]
            length[pos] = length[pos-left] = length[pos+right] = left+right+1
            count[left+right+1] += 1 
            count[left]-=1 
            count[right]-=1 
            
            if count[m]:
                res = i+1 
                
                
        return res 

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        # expected group len is equal to len(arr)
        if(m == len(arr)):
            return m
        '''
        group_edge = [0] * (len(arr)+2)
        ans = 0 
        for i in range(0, len(arr)):
            print(arr[i])
            left=right=arr[i]
            if group_edge[right+1]>0: 
                right=group_edge[right+1]
            if group_edge[left-1]>0: 
                left=group_edge[left-1]
            group_edge[left], group_edge[right] = right, left
            if (right-arr[i]==m) or (arr[i]-left ==m): 
                ans=i
            print(group_edge)
        '''
        group_len = [0] * (len(arr) + 2)
        cnt_group_len = [0] * (len(arr) + 1)
        ans = -1
        for i in range(0, len(arr)):
            
            #print(arr[i])
            
            left_most = arr[i] - 1
            right_most = arr[i] + 1
            new_len = group_len[left_most] + group_len[right_most] + 1
            group_len[arr[i]] = new_len
            
            cnt_group_len[new_len] += 1
            cnt_group_len[group_len[left_most]] -= 1
            cnt_group_len[group_len[right_most]] -= 1
            
            if(group_len[left_most] > 0):
                group_len[arr[i] - group_len[left_most]] = new_len
            
            if(group_len[right_most] > 0):
                group_len[arr[i] + group_len[right_most]] = new_len
            
            #print(group_len)
            #print(cnt_group_len)
            
            if(cnt_group_len[m] > 0):
                ans = i + 1
            
        return ans
                

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        lookup = {}
        lastStep = -1
        valid = set()
        for idx, num in enumerate(arr):
            left = num
            right = num
            if num-1 in lookup:
                left = lookup[num-1][0]
                if num-1 in valid:
                    valid.remove(num-1)
                
            if num+1 in lookup:
                right = lookup[num+1][1]
                if num+1 in valid:
                    valid.remove(num+1)
            
            if left in valid:
                valid.remove(left)
            
            if right in valid:
                valid.remove(right)
                
            lookup[left] = (left, right)
            lookup[right] = (left, right)
            
            if right-left+1 == m:
                valid.add(left)
                valid.add(right)
            
            if valid:
                lastStep = idx+1
        return lastStep
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        index = [-1 for i in range(len(arr))]
        groups = [0 for i in range(len(arr))]
        count, lastindex = 0, -1
        for i, num in enumerate(arr):
            a = num -1
            groups[a], index[a], length = 1, a, 1
            if a-1 >= 0 and groups[a-1] == 1:
                left = index[a-1] # left index
                if a - left == m:
                    count -= 1
                index[left], index[a] = a, left # left end
                length += (a - left)
            if a+1 < len(arr) and groups[a+1] == 1:
                left, right = index[a], index[a+1]
                if right - a == m:
                    count -= 1
                index[right], index[left] = left, right
                length += (right - a)
            if length == m: count += 1
            if count > 0: lastindex = i+1
        return lastindex
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        l=len(arr)
        left=[i for i in range(l+1)]
        right=[i for i in range(l+2)]
        count=[0]*(l+1)
        res=-1
        step=0
        for a in arr:
            step+=1
            lt=left[a-1]
            rt=right[a+1]
            tlen=rt-lt-1
            
            templeft=a-lt-1
            tempright=rt-a-1
            count[templeft]-=1
            count[tempright]-=1
            count[tlen]+=1
            
            if count[m]>0:
                res=step
            right[lt+1]=rt
            left[rt-1]=lt
        return res
                   

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        sizes = [0] * (len(arr) + 2)
        res = -1
        cnt = 0
        for step, cur in enumerate(arr, start=1):
            l, r = sizes[cur - 1], sizes[cur + 1]
            new_sz = l + 1 + r
            sizes[cur - l] = sizes[cur + r] = new_sz
            cnt += (new_sz == m) - (l == m) - (r == m)
            if cnt:
                res = step
        return res
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        a = [0] * len(arr) 
        heads = {}
        ends = {}
        ans = -1
        for step, i in enumerate(arr):
            a[i - 1] = 1
            if self.mergeOne(a, i - 1, heads, ends, m) == 1:
                ans = step 
        for i in heads:
            if heads[i] - i + 1 == m:
                return len(arr)
        return ans
            
    def mergeOne(self, ls, index, heads, ends, m):
        left, right = index - 1, index + 1 
        lefthead = rightend = index
        ext = -1
        if left in ends:
            lefthead = ends[left]
            if left - lefthead + 1 == m:
                ext = 1
            del ends[left]
        if right in heads:
            rightend = heads[right]
            if rightend - right + 1 == m:
                ext = 1
            del heads[right]
        heads[lefthead] = rightend
        ends[rightend] = lefthead
        return ext
            

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        res, n = -1, len(arr)
        # length of group
        length = [0] * (n + 2)
        # count of length
        count = [0] * (n + 1)
        
        for i, v in enumerate(arr):
            left, right = length[v - 1], length[v + 1]
            length[v] = length[v - left] = length[v + right] = left + right + 1
            count[left] -= 1
            count[right] -= 1
            count[length[v]] += 1
            if count[m]:
                res = i + 1        
        return res
class Solution:
    def findLatestStep2(self, arr: List[int], m: int) -> int:
        n = len(arr)
        dic = collections.Counter()
        cnt = collections.Counter()
        res = -1
        for i, a in enumerate(arr):
            l = dic[a - 1]
            r = dic[a + 1]
            dic[a - l] = dic[a + r] = dic[a] = l + r + 1
            cnt[l + r + 1] += 1
            cnt[l] -= 1
            cnt[r] -= 1
            if cnt[m]:
                res = i + 1
        return res
    
    def findLatestStep(self, arr: List[int], m: int) -> int:
        n = len(arr)
        leng = [0]  * (n + 2)
        cnt = [0] * (n + 1)
        res = -1
        for i, a in enumerate(arr):
            l = leng[a - 1]
            r = leng[a + 1]
            leng[max(0, a - l)] = leng[min(n + 1, a + r)] = l + r + 1
            cnt[l] -= 1
            cnt[r] -= 1
            cnt[l + r + 1] += 1
            if cnt[m]:
                res = i + 1
        return res

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        if len(arr) == m: return len(arr)
        days = [0] * len(arr)
        for i, x in enumerate(arr,1):
            days[x-1] = i
        deq = collections.deque()
        def insert_deq(val,pop):
            if deq and deq[0] == pop:
                deq.popleft()
            while deq and deq[-1] < val:
                deq.pop()
            deq.append(val)
        latest = -1
        for i, x in enumerate(days):
            insert_deq(x, days[i-m] if i >= m else None )
            if i < m - 1:
                continue
            left = days[i-m] if(i - m >= 0) else float('inf')
            right = days[i+1] if(i + 1 < len(days)) else float('inf') 
            max_day_turn1 = deq[0]
            if left > max_day_turn1 and right > max_day_turn1:
                latest = max(latest,min(left,right) - 1)
        return latest

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        nums = []
        max_index = -1
        correct_blocks = 0
        latest_index = -1
        for _ in range(len(arr)):
            nums.append(0)
        for i in range(len(arr)):
            index = arr[i]-1
            
            if index == 0:
                try:
                    nums[index] = 1 + nums[index+1]
                    if nums[index+1] == m:
                        correct_blocks -= 1
                    if 1 + nums[index+1] == m:
                        correct_blocks += 1
                    if nums[index+1] != 0:
                        val = 1 + nums[index+1]
                        nums[index + nums[index+1]] = val
                        nums[index+1] = val
                except:
                    return 1
            elif index == len(arr)-1:
                try:
                    nums[index] = 1 + nums[index-1]
                    if nums[index-1] == m:
                        correct_blocks -= 1
                    if 1 + nums[index-1] == m:
                        correct_blocks += 1
                    if nums[index-1] != 0:
                        val = 1 + nums[index - 1]
                        nums[index - nums[index-1]] = val
                        nums[index-1] = val
                except:
                    return 1
            else:
                try:
                    val = 1 + nums[index-1] + nums[index+1]
                    if nums[index-1] == m:
                        correct_blocks -= 1
                    if nums[index+1] == m:
                        correct_blocks -= 1
                    if 1 + nums[index-1] + nums[index+1] == m:
                        correct_blocks += 1
                    nums[index] = val
                    if nums[index-1] != 0:
                        nums[index - nums[index-1]] = val
                        nums[index-1] = val
                    if nums[index+1] != 0:
                        nums[index + nums[index+1]] = val
                except:
                    pass
            if correct_blocks > 0:
                latest_index = i+1
        return latest_index
class Solution:
    def findLatestStep(self, A, m):
        if m == len(A): return m
        length = [0] * (len(A) + 2)
        res = -1
        for i, a in enumerate(A):
            left, right = length[a - 1], length[a + 1]
            if left == m or right == m:
                res = i
            length[a - left] = length[a + right] = left + right + 1
        return res

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        dic = {}
        endpoints = {}
        startpoints = {}
        latest = -1
        for i in range(len(arr)):
            step = arr[i]
            if step-1 in endpoints:
                if step+1 not in startpoints:
                    j = endpoints[step-1]
                    startpoints[j] = step
                    endpoints[step] = j
                    del endpoints[step-1]
                    l = step-j
                    dic[l] -= 1
                    if l+1 in dic:
                        dic[l+1] += 1
                    else: dic[l+1] = 1
                else:
                    j = endpoints[step-1]
                    k = startpoints[step+1]
                    startpoints[j] = k
                    endpoints[k] = j
                    del startpoints[step+1]
                    del endpoints[step-1]
                    l1 = step-j
                    l2 = k-step
                    dic[l1] -= 1
                    dic[l2] -= 1
                    if l1+l2+1 in dic:
                        dic[l1+l2+1] += 1
                    else: dic[l1+l2+1] = 1
            elif step+1 in startpoints:
                k = startpoints[step+1]
                endpoints[k] = step
                startpoints[step] = k
                del startpoints[step+1]
                l = k-step
                dic[l] -= 1
                if l+1 in dic:
                    dic[l+1] += 1
                else: dic[l+1] = 1
            else:
                endpoints[step] = step
                startpoints[step] = step
                if 1 in dic:
                    dic[1] += 1
                else: dic[1] = 1
            if m in dic and dic[m]!=0:
                latest = i+1
        return latest

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        nums = []
        max_index = -1
        correct_blocks = 0
        latest_index = -1
        for _ in range(len(arr)):
            nums.append(0)
        for i in range(len(arr)):
            index = arr[i]-1
            
            if index == 0:
                try:
                    nums[index] = 1 + nums[index+1]
                    if nums[index+1] == m:
                        correct_blocks -= 1
                    if 1 + nums[index+1] == m:
                        correct_blocks += 1
                    if nums[index+1] != 0:
                        val = 1 + nums[index+1]
                        nums[index + nums[index+1]] = val
                except:
                    return 1
            elif index == len(arr)-1:
                try:
                    nums[index] = 1 + nums[index-1]
                    if nums[index-1] == m:
                        correct_blocks -= 1
                    if 1 + nums[index-1] == m:
                        correct_blocks += 1
                    if nums[index-1] != 0:
                        val = 1 + nums[index - 1]
                        nums[index - nums[index-1]] = val
                except:
                    return 1
            else:
                try:
                    val = 1 + nums[index-1] + nums[index+1]
                    if nums[index-1] == m:
                        correct_blocks -= 1
                    if nums[index+1] == m:
                        correct_blocks -= 1
                    if 1 + nums[index-1] + nums[index+1] == m:
                        correct_blocks += 1
                    nums[index] = val
                    if nums[index-1] != 0:
                        nums[index - nums[index-1]] = val
                        nums[index-1] = val
                    if nums[index+1] != 0:
                        nums[index + nums[index+1]] = val
                except:
                    pass
            if correct_blocks > 0:
                latest_index = i+1
        return latest_index
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        d = defaultdict(int)
        ans = 0
        n = len(arr)
        l = [-1]*n
        r = [-1]*n
        for i in range(n):
            x = y = arr[i]-1
            if x and l[x-1] != -1:
                d[(x-1)-l[x-1]+1] -= 1
                x = l[x-1]
            if y < n-1 and r[y+1] != -1:
                d[r[y+1]-(y+1)+1] -= 1
                y = r[y+1]
            d[y-x+1] += 1
            if d[m]:
                ans = i+1
            l[y] = x
            r[x] = y
        return ans if ans else -1
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        start = {}
        end = {}
        lengths = {}
        ans = -1
        for i in range(len(arr)):
            st = arr[i]
            ending = st + 1
            if st in end:
                tempStart = st
                st = end[st]
                end.pop(st,None)
            else:
                tempStart = st
            if ending in start:
                ed = ending
                ending = start[ending]
                start.pop(ed,None)
            else:
                ed = ending
            if st != tempStart:
                lengths[tempStart-st] -= 1
            if ed != ending:
                print('d')
                lengths[ending - ed] -= 1
            if ending-st not in lengths:
                lengths[ending-st] = 1
            else:
                lengths[ending-st] = lengths[ending-st]+1
            start[st] = ending
            end[ending] = st
            if m in lengths and lengths[m] > 0:
                ans = i+1
        return ans

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        l = len(arr)
        if l == m:
            return l
        
        x = [0 for x in range(l+2)]
        last = -1
        for i in range(l):
            cur = arr[i]
            left, right = x[cur-1], x[cur+1]
            if left == m or right == m:
                last = i
            x
            x[cur-left] = x[cur+right] = left+right+1

        return last
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        length = len(arr)
        if length == m:
            return m
        if length < m:
            return -1

        count_m = 0
        array2 = [0 for _ in range(length + 2)]
        result = -1
        for i in range(length):
            index = arr[i]
            if array2[index - 1] == m:
                count_m -= 1
            if array2[index + 1] == m:
                count_m -= 1            
            array2[index] = array2[index - 1] + array2[index + 1] + 1
            if array2[index - 1] > 0:
                array2[index - array2[index - 1]] = array2[index]
            if array2[index + 1] > 0:
                array2[index + array2[index + 1]] = array2[index]
            if array2[index] == m:
                count_m += 1
            if count_m > 0:
                result = i + 1
        return result
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        n = len(arr)
        mask = [0] * (n + 2)
        ans = -1
        
        i = 1
        count = 0
        for index in arr:
            total_len = 1 + mask[index - 1] + mask[index + 1]
            change_set = {index + mask[index + 1], index - mask[index - 1]}
            for ind in change_set:
                if mask[ind] == m:
                    count -= 1
            mask[index - mask[index - 1]] = total_len
            mask[index + mask[index + 1]] = total_len
            
            if total_len == m:
                count += 1
            if count > 0:
                ans = i
            i += 1
        return ans
            

class Solution:
    def findLatestStep(self, a: List[int], m: int) -> int:
        if m == len(a): return len(a)
        index2len, ans = defaultdict(int), -1        
        for i, p in enumerate(a):    
            l, r = index2len[p-1], index2len[p+1]      
            if l == m or r == m: ans = i
            index2len[p-l] = index2len[p+r] = l + 1 + r            
        return ans
class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        latest = -1
        
        if len(arr) == 1:
            return 1 if arr[0] == m else -1
        
        ## (length -> appear times)
        length_count = [0 for _ in range(len(arr) + 1)]
        ## store consecutive 1's count ((start, end) -> length)
        binary_str = [0 for _ in range(len(arr))]
        for idx in range(len(arr)):
            num = arr[idx] - 1
            ## left boundary
            if num <= 0:
                if binary_str[num+1] > 0:
                    right_count = binary_str[num+1]
                    binary_str[num] = 1 + right_count
                    binary_str[num+right_count] += 1
                    length_count[1+right_count] += 1
                    length_count[right_count] -= 1
                else:
                    binary_str[num] = 1
                    length_count[1] += 1
            ## right boundary
            elif num >= len(arr) - 1:
                if binary_str[num-1] > 0:
                    left_count = binary_str[num-1]
                    binary_str[num-left_count] += 1
                    binary_str[num] = 1 + left_count
                    length_count[1+left_count] += 1
                    length_count[left_count] -= 1
                else:
                    binary_str[num] = 1
                    length_count[1] += 1
            ## in the middle
            else:
                if binary_str[num+1] > 0 and binary_str[num-1] > 0:
                    left_count = binary_str[num-1]
                    right_count = binary_str[num+1]
                    binary_str[num-left_count] += (right_count + 1)
                    binary_str[num+right_count] += (left_count + 1)
                    length_count[left_count + right_count + 1] += 1
                    length_count[left_count] -= 1
                    length_count[right_count] -= 1
                elif binary_str[num+1] > 0 and binary_str[num-1] <= 0:
                    right_count = binary_str[num+1]
                    binary_str[num] = 1 + right_count
                    binary_str[num+right_count] += 1
                    length_count[1 + right_count] += 1
                    length_count[right_count] -= 1
                elif binary_str[num+1] <= 0 and binary_str[num-1] > 0:
                    left_count = binary_str[num-1]
                    binary_str[num-left_count] += 1
                    binary_str[num] = 1 + left_count
                    length_count[1 + left_count] += 1
                    length_count[left_count] -= 1
                else:
                    binary_str[num] = 1
                    length_count[1] += 1
                    
            #print(num, binary_str, latest, idx, length_count)
            if length_count[m] > 0:
                latest = idx
                
        return latest + 1 if latest > -1 else -1

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        if len(arr) == m: return len(arr)
        days = [0] * len(arr)
        for i, x in enumerate(arr,1):
            days[x-1] = i
        deq = collections.deque()
        def insert_deq(val,pop):
            if deq and deq[0] == pop:
                deq.popleft()
            while deq and deq[-1] < val:
                deq.pop()
            deq.append(val)
        latest = -1
        for i, x in enumerate(days):
            insert_deq(x, days[i-m] if i >= m else None )
            if i < m - 1 or i == len(arr) - 1:
                continue
            left = days[i-m] if(i - m >= 0) else float('inf')
            right = days[i+1] if(i + 1 < len(days)) else float('inf') 
            max_day_turn1 = deq[0]
            if i == m-1: #checking just right, sliding window start
                latest = max(latest,right - 1) if right > max_day_turn1 else latest
            else: # making sure left and right side turn after sliding window max, and the min will be the latest
                if left > max_day_turn1 and right > max_day_turn1:
                    latest = max(latest,min(left,right) - 1)
        left = days[-1 - m]
        latest = max(latest,left - 1) if left > deq[0] else latest
        return latest

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        slots = [0 for _ in arr]
        num_m = 0
        last_m = -1 
        if len(arr) == 1:
            if m == 1:
                return 1
            else:
                return -1 
        for n in range(len(arr)):
            i = arr[n]
            idx = i -1 
            if idx == len(arr) - 1:
                slots[idx] = slots[idx - 1] + 1
                if slots[idx] == m:
                    num_m += 1
                if slots[idx - 1] == m:
                    num_m -= 1                    
                if slots[idx - 1] > 0:
                    slots[idx - slots[idx - 1]] = slots[idx]
                
            elif idx == 0:
                slots[idx] = slots[idx + 1] + 1
                if slots[idx] == m:
                    num_m += 1
                if slots[idx + 1] == m:
                    num_m -= 1  
                if slots[idx + 1] > 0:
                    slots[idx + slots[idx + 1]] = slots[idx]
            else:
                slots[idx] = slots[idx- 1] + slots[idx + 1]+ 1
                if slots[idx] == m:
                    num_m += 1
                if slots[idx + 1] == m:
                    num_m -= 1
                if slots[idx - 1] == m:
                    num_m -= 1    
                slots[idx - slots[idx - 1]] = slots[idx]
                slots[idx + slots[idx + 1]] = slots[idx]
            if num_m > 0:
                last_m = n + 1

        return last_m
                

class Solution:
    def findLatestStep(self, arr: List[int], m: int) -> int:
        length = [0] * (len(arr)+2)
        count = [0] * (len(arr)+1)
        ans = -1
        for i, a in enumerate(arr):
            left, right = length[a-1], length[a+1]
            length[a] = length[a-left] = length[a+right] = left + right + 1
            count[left] -= 1
            count[right] -= 1
            count[length[a]] += 1
            # print(length, count)
            if count[m]:
                ans = i + 1
        return ans
