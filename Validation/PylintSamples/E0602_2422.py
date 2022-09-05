class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        if nums[0] > nums[1]:
            largest = nums[0]
            second_largest = nums[1]
        else:
            largest = nums[1]
            second_largest = nums[0]
        for i in range(2,len(nums)):
            if nums[i] > largest:
                second_largest = largest
                largest = nums[i]
            elif nums[i] > second_largest:
                second_largest = nums[i]
        return (largest-1) * (second_largest -1)
                
                
                

class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        max_1 = max(nums)
        nums.remove(max_1)
        max_2 = max(nums)
        return (max_2-1)*(max_1-1)
class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        i=nums[0];j=nums[1]
        for num in range(2,len(nums)):
            if(nums[num]>i or nums[num]>j):
                if(nums[num]>i):
                    if(i>j):
                        j=i    
                    i=nums[num]
                    #print(i)
                else:
                    if(j>i):
                        i=j    
                    j=nums[num]
                    #print(j)
        #print (i," ",j)
        return (i-1)*(j-1)
                
            

class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        prods=[]
        for i in range(0,len(nums)-1):
            for j in range(i+1, len(nums)):
                prods.append((nums[i]-1)*(nums[j]-1))
        return max(prods)
class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        maximum_value=[]
        for i in range(len(nums)):
            for j in range(i+1,len(nums)):
                maximum_value.append((nums[i]-1)*(nums[j]-1))
        return max(maximum_value)
from itertools import combinations
class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        comb=list(combinations(nums,2))
        maxv=0
        for x in comb :
            if (x[0]-1)*(x[1]-1)>maxv :
                maxv=(x[0]-1)*(x[1]-1)
        return maxv

class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        sum = 0
        for i, num in enumerate(nums):
            j = i + 1
            while j < len(nums):
                tempSum = (nums[i]-1)*(nums[j]-1)
                if sum < tempSum:
                    sum = tempSum
                j = j + 1
        return sum

class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        products = []
        
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                products.append((nums[i] - 1) * (nums[j] -1)) 
        
        return max(products)
class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        n = []
        for i in range(len(nums)-1):
            for j in range(i+1, len(nums)):
                n.append((nums[i]-1)*(nums[j]-1))
        return max(n)
class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        maximum = 0
        for i in range(len(nums)):
            for j in range(i+1, len(nums)):
                product = (nums[i]-1)*(nums[j]-1)
                if maximum < product:
                    maximum = product
        return maximum
class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        l=[]
        for i in range(len(nums)-1):
            for j in range(i+1,len(nums)):
                s=(nums[i]-1)*(nums[j]-1)
                l.append(s)
        return max(l)
                

class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        k=[]
        for i in range(len(nums)):
            for j in range(i+1,len(nums)):
                l=(nums[i]-1)*(nums[j]-1)
                k.append(l)
        return max(k)
class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        max_pro=0
        for i in range(len(nums)-1):
            for j in range(i+1,len(nums)):
                temp=(nums[i]-1)*(nums[j]-1)
                if temp>max_pro:
                   max_pro=temp
        return(max_pro)
                
                

class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        k=[]
        for i in range(len(nums)):
            for j in range(i+1,len(nums)):
                
                #print((nums[i]-1),(nums[j]-1))
                k.append((nums[i]-1)*(nums[j]-1))
                
        print((max(k)))
        return(max(k))

class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        new = []
        
        for i in range(len(nums)):
            for j in range(i+1, len(nums)):
                new.append(((nums[i] - 1)*(nums[j] - 1)))
                
        return max(new)
class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        lst = []
        for i in range(len(nums)):
            for j in range(i+1,len(nums)):
                x = (nums[i]-1)*(nums[j]-1)
                lst.append(x)
        return max(lst)
class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        if nums == []:
            return None
        max = None
        for idx, i in enumerate(nums):
            j = idx + 1
            while j < len(nums):
                product = (i - 1) * (nums[j] - 1)
                if max == None:
                    max = product
                elif product > max:
                    max = product
                j += 1
        return max

class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        mp = float('-inf')
        for i in range(len(nums)):
            for j in range(i+1, len(nums)):
                mp = max(mp, (nums[i]-1)*(nums[j]-1))
        return mp
class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        ans = 0
        for i in range(len(nums)):
            for j in range(i+1,len(nums)):
                ans = max(ans,(nums[i]-1)*(nums[j]-1))
        return ans
class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        max_product = 0
        for i in range(len(nums) - 1):
            for j in range(i + 1, len(nums)):
                max_product = max(max_product, (nums[i] - 1) * (nums[j] - 1))

        return max_product
class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        ans = float('-inf')
        for i in range(0, len(nums)):
            for j in range(i+1, len(nums)):
                ans = max(ans, (nums[i]-1)*(nums[j]-1))
        
        return ans

class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        ans = float('-inf')
        n = len(nums)
        for i in range(n):
            for j in range(i + 1, n):
                ans = max((nums[i] - 1) * (nums[j] - 1), ans)
        return ans
class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        arr = sorted(nums)
        
        idx1 = nums.index(arr[-1])
        idx2 = nums.index(arr[-2])
        
        return (nums[idx1]-1)*(nums[idx2]-1)
class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        result = 0
        for i in range(len(nums)-1):
            for j in range(i+1, len(nums)):
                # prod = (nums[i]-1) * (nums[j]-1)
                # if prod > result:
                    # result = prod
                result = max(result, (nums[i]-1) * (nums[j]-1))
        return result

class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        sol = 0
        
        for i in range(len(nums)):
            for j in range(i+1, len(nums)):
                sol = max(sol, (nums[i]-1)*(nums[j]-1))
        
        return sol
class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        m = 0
        for i in range(len(nums)):
            for j in range(i+1, len(nums)):
                m = max(m, (nums[i] - 1) * (nums[j] - 1))
        return m
class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        max_val = sys.maxsize * -1
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                max_val = max(max_val, (nums[i] - 1) * (nums[j] - 1))
        return max_val
class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        #m = (nums[0]-1)*(nums[1]-1)
        m=0
        for i in range(0,len(nums)-1):
            for j in range(i+1,len(nums)):
                m = max(m, (nums[i]-1)*(nums[j]-1))
        return m

class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        r=0
        for i in range(len(nums)-1):
            for j in range(i+1,len(nums)):
                r=max(r,(nums[i]-1)*(nums[j]-1))
        return r
class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        
        max_val = float('-inf') 
        for i in range(len(nums)):
            for j in range(i+1,len(nums)):
                max_val = max(max_val,((nums[i]-1) * (nums[j]-1)))
        return max_val

class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        maxNum = 0
        for i in range(len(nums)):
            for j in range(len(nums)):
                if (nums[i]-1)*(nums[j]-1) >= maxNum and i != j:
                    maxNum = (nums[i]-1)*(nums[j]-1)
        return maxNum

class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        swapped = True
        while swapped:
            swapped = False
            for i in range(len(nums)-1):
                if nums[i+1]>nums[i]:
                    nums[i],nums[i+1]=nums[i+1],nums[i]
                    swapped = True
        return (nums[0]-1)*(nums[1]-1)
class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        max_prod = 0
        for i in range(0, len(nums)-1):
            for j in range(i+1, len(nums)):
                prod=((nums[i]-1)*(nums[j]-1))
                max_prod = max(max_prod, prod)
        return max_prod
class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        res = 0
        for i in range(len(nums)):
            for j in range(i+1, len(nums)):
                if i !=j:
                    res = max(res, (nums[i]-1) * (nums[j]-1))
        return res
class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        req = 0
        for i in range(1,len(nums)):
            for j in range(i):
                req = max(req, (nums[i]-1)*(nums[j]-1))
        return req

class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        n=len(nums)
        ans=float('-inf')
        for i in range(n-1):
            for j in range(i+1,n):
                ans=max((nums[i]-1)*(nums[j]-1),ans)
        return ans

class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        b = 0
        for i in range(len(nums)):
            for j in range(len(nums)):
                a = (nums[i] - 1)*(nums[j]-1)
                if b<a and i!=j:
                    b=a
        return(b)
            

class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        max=0
        for x in range(len(nums)):
            for y in range(len(nums)):
                if(y>x):
                    if((nums[x]-1)*(nums[y]-1)>max):
                        max=(nums[x]-1)*(nums[y]-1)
        return max

from itertools import combinations


class Solution:
    def maxProduct(self, nums: List[int]) -> int:

        combs = list(combinations(nums, 2))
        return max(list([(x[0]-1)*(x[1]-1) for x in combs]))

class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        for i,v in enumerate(nums):
            nums[i]=nums[i]-1
            nums[i]=abs(nums[i])
        nums.sort(reverse=True)
        return (nums[0])*(nums[1])
class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        first, second = 0, 0
        
        for number in nums:

            if number > first:
                # update first largest and second largest
                first, second = number, first

            else:
                # update second largest
                second = max( number, second)

        return (first - 1) * (second - 1)
            

class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        max1 = nums.pop(nums.index(max(nums)))
        max2 = nums.pop(nums.index(max(nums)))
        return ((max1-1)*(max2-1))
class Solution:
    def maxProduct(self, nums):
        result = 0
        altnums = []
        for num in nums:
            altnums.append(num)
        for i in range(len(nums)):
            if nums[i] == max(nums):
                altnums[i] = 0
                break
        print(altnums, nums)
        result = (max(nums)-1) * (max(altnums)-1)
        return result
class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        s = sorted(nums, reverse = True)
        return (s[0]-1)*(s[1]-1)

class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        min_val = None
        max_val = None
        
        for i, val in enumerate(nums):
            if min_val is None or min_val[1] > val:
                min_val = [i, val]
            if max_val is None or max_val[1] < val:
                max_val = [i, val]
            
        answer = 0
        for i, val in enumerate(nums):
            if i != min_val[0]:
                answer = max(answer, (min_val[1]-1)*(val-1))
            if i != max_val[0]:
                answer = max(answer, (max_val[1]-1)*(val-1))
        return answer
from itertools import combinations
import numpy
class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        nums.sort()
        res = (nums[-1]-1)*(nums[-2]-1)
        return res

class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        Max = max(nums)
        nums.remove(Max)
        return (Max-1)*(max(nums)-1)
class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        highest = max(nums)
        nums.pop(nums.index(max(nums)))
        second_highest = max(nums)
        
        return((highest-1)*(second_highest-1))
class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        nums.sort()
        return (nums[- 1] - 1) * (nums[- 2] - 1)
class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        
        heap = []
        
        for num in nums:
            heapq.heappush(heap,-num)
            
        m1 = heapq.heappop(heap)
        m2 = heapq.heappop(heap)
        
        return -(m1+1) * -(m2+1)
class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        a = max(nums)
        nums.remove(a)
        return (a-1)*(max(nums)-1)

class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        if len([i for i, x in enumerate(nums) if x == max(nums)])>1:
            return (max(nums)-1)**2
        else:
            maxi = max(nums)
            other = nums
            nums.remove(maxi)
            return (maxi-1) * (max(other)-1)
class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        print(sorted(nums)[-1]-1*sorted(nums)[-2]-1)
        if len([i for i, x in enumerate(nums) if x == max(nums)])>1:
            return (max(nums)-1)**2
        else:
            maxi = max(nums)
            other = nums
            nums.remove(maxi)
            return (maxi-1) * (max(other)-1)
class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        max_nums=[]

        for num in nums:
            if num==max(nums):

                max_nums.append(num-1)
                nums.remove(num)
        for num in nums:
            if num == max(nums):
                max_nums.append(num - 1)
                nums.remove(num)
        return max_nums[0]*max_nums[1]

class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        s = set([])
        m = 0
        for num in nums:
            x = num-1
            for y in s:
                p = x*y
                if m<p:
                    m = p
            s.add(x)
        return m

class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        nums = sorted(nums)
        l = len(nums)
        return (nums[l-1]-1)*(nums[l-2]-1) 

class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        
        largeNumber = 0
        secondNumber = 0
        
        for i in nums:
            if i > largeNumber:
                largeNumber = i 
        for j in nums:
            if j > secondNumber and j != largeNumber:
                secondNumber = j
                
        for x in range(0, len(nums)):
            for y in range(x+1, len(nums)):
                if nums[x] == nums[y] and nums[x] == largeNumber:
                    secondNumber = largeNumber
                    
        return int((largeNumber - 1) * (secondNumber - 1))

class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        max_nums = []
        for i in range(2):
            for num in nums:
                if num == max(nums):
                    max_nums.append(num - 1)
                    nums.remove(num)
        return max_nums[0] * max_nums[1]

class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        k = nums[0]
        n = [i for i in range(2)]
        for i in range(len(nums)):
            for j in range(i):
                if nums[i]*nums[j]>k:
                    n[0] =  nums[j]
                    n[1] = nums[i]
                    k = nums[i]*nums[j]
        m = ((n[0]-1)*(n[1]-1))
        return m
import itertools

class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        return max(x*y for x, y in itertools.combinations(map(lambda x: x-1, nums), 2))
class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        candidates = [0, 0]
        highest = nums[0]
        
        for i in range(len(nums)):
            for j in range(i+1, len(nums)):
                product = nums[i] * nums[j]
                if product > highest:
                    highest = product
                    candidates[0] = i
                    candidates[1] = j
        
        return (nums[candidates[0]] - 1) * (nums[candidates[1]] - 1)
            
    # i = 1
    # j = 3
    # product = 25
    # highest = 25
    # candidates = [1, 3]

class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        max_prod = 0
        
        nums = [x-1 for x in nums]
        for first in range(len(nums)-1):
            for second in range(first+1, len(nums)):
                prod = nums[first] * nums[second]
                if prod > max_prod:
                    max_prod = prod
        
        return max_prod

class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        sorted_nums= sorted(nums)
        return (sorted_nums[-1]-1) *  (sorted_nums[-2]-1)

class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        
        ix = 0
        iy = 0
        ma = 0
        
        for i in range(0,len(nums)):
            for j in range(i+1,len(nums)):
                if (nums[i]*nums[j]) > ma:
                    ma = nums[i]*nums[j]
                    ix = i
                    iy = j
                    
        result = (nums[ix]-1)*(nums[iy]-1)
        
        return result
class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        second_largest = 0
        largest = 0
        max_product = 0
        for index1 in range(len(nums)-1):
            for index2 in range(index1+1, len(nums)):
                product = nums[index1] * nums[index2]
                if product > max_product:
                    max_product = product
                    if nums[index1] > nums[index2]:
                        largest, second_largest = nums[index1], nums[index2]
                    else:
                        largest, second_largest = nums[index2], nums[index1]
        return ((largest-1) * (second_largest-1))

class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        from itertools import combinations
        answer = None
        max_value =  float('-inf')
        for x,y in combinations(list(range(len(nums))), 2):
            if (nums[x] * nums[y]) > max_value:
                answer = (nums[x]-1)*(nums[y]-1)
                max_value= nums[x] * nums[y]
        return answer
                

from itertools import combinations

class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        currentMaximum = 0
        for num_i, num_j in combinations(nums, 2):
            if (product := (num_i - 1)*(num_j - 1)) > currentMaximum:
                currentMaximum = product
        
        # for i, num_i in enumerate(nums):
        #     for j, num_j in enumerate(nums):
        #         if (currentProduct := (num_i - 1)*(num_j - 1)) > currentMaximum:
        #             print(num_i, num_j)
        #             currentMaximum = currentProduct
        return currentMaximum        
class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        
        return (sorted(nums)[-1]-1) *  (sorted(nums)[-2]-1)

class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        maxi = 0
        
        for i in range(len(nums)):
            for j in range(i+1, len(nums)):
                if ((nums[i]-1) * (nums[j]-1)) > maxi:
                    maxi = ((nums[i]-1) * (nums[j]-1))
        
        return maxi

class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        maxx=0
        for i in range(len(nums)):
            for j in range(i+1,len(nums)):
                if(((nums[i]-1)*(nums[j]-1))>maxx):
                    maxx=(nums[i]-1)*(nums[j]-1)
        return maxx
class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        max = 0;
        for i in range(len(nums)):
            for j in range(i+1,len(nums)):
                if max < (nums[i]-1)*(nums[j]-1):
                    max = (nums[i]-1)*(nums[j]-1)
        return max

class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        max_product=0
        for i in range(len(nums)):
            for j in range(i+1,len(nums)):
                if (nums[i]-1)*(nums[j]-1) > max_product:
                    max_product = (nums[i]-1)*(nums[j]-1)
        return max_product

class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        res = (nums[0]-1)*(nums[1]-1)
        for i in range(len(nums)-1):
            for j in range(i+1, len(nums)):
                if res < (nums[i]-1)*(nums[j]-1):
                    res = (nums[i]-1)*(nums[j]-1)
        return res

class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        maxim = 0
        for i in range(len(nums)): 
            for j in range(i + 1, len(nums)): 
                if (nums[i]-1)*(nums[j]-1) > maxim: 
                    print(i)
                    print(j)
                    maxim = (nums[i]-1)*(nums[j]-1)
                    
        return maxim
class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        high = 0
        for i in range(len(nums)):
            for j in range(i+1,len(nums)):
                if (nums[i]-1)*(nums[j]-1) > high:
                    high = (nums[i]-1)*(nums[j]-1)
        return high 

class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        n = len(nums)
        current_max = -2147483647
        for i in range(n-1):
            for j in range(i+1, n):
                temp = (nums[i]-1)*(nums[j]-1)
                if(temp > current_max):
                    current_max = temp
        return current_max
class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        maxNum = 0
        for i in range(len(nums)):
            for j in range(i+1,len(nums)):
                temp = (nums[i]-1)*(nums[j]-1)
                if temp > maxNum:
                    maxNum = temp
        return maxNum
#optimal, one-pass, without sorting

class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        if nums[0] > nums[1]:
            max1 = nums[0]
            max2 = nums[1]
        else:
            max1 = nums[1]
            max2 = nums[0]
            
        for i in range(2, len(nums)):
            if nums[i] > max1:
                max2 = max1
                max1 = nums[i]
            elif nums[i] > max2:
                max2 = nums[i]
                
        return (max1 - 1)*(max2 - 1)
class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        max = 0
        for i in range(0 , len(nums)-1):
            for j in range(i+1 , len(nums)):
                product = (nums[i]-1) * (nums[j]-1)
                if product > max:
                    max = product
        return max
                
        # 0 1 2 3 4 5 len=6 
        # 2 3 5 6 8 9

class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        result = 0
        for n in range(len(nums)-1):
            for m in range(n+1, len(nums)):
                product = (nums[n] - 1) * (nums[m] -1)
                if product > result:
                    result = product
        return result
class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        highest = (nums[0]-1) * (nums[1]-1)
        c = 0
        for i in range(len(nums)-1):
            for j in range(i+1,len(nums)):
                c = (nums[i] - 1) * (nums[j] -1)
                if c >= highest:
                    highest = c
                
        return highest
            

class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        maxRes = float('-inf')
        for x in range(len(nums)):
            for y in range(x + 1, len(nums)):
                output = (nums[x] - 1) * (nums[y] - 1)
                if output > maxRes:
                    maxRes = output
        return maxRes

class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        ans=0
        for i in range(0,len(nums)):
            for j in range(i+1,len(nums)):
                if ans<(nums[i]-1)*(nums[j]-1):
                    ans=(nums[i]-1)*(nums[j]-1)
        return ans
            

class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        prod = 0
        for i in range(len(nums)):
            for j in range(i+1, len(nums)):
                if (nums[i]-1) * (nums[j]-1) > prod:
                    prod = (nums[i]-1) * (nums[j]-1)
        return prod
class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        n = len(nums)
        
        max_product = (nums[0] - 1) * (nums[1] - 1)
        
        for i in range(n-1):
            for j in range(i+1, n):
                product = (nums[i] - 1) * (nums[j] - 1)
                if(product > max_product):
                    max_product = product
        
        return max_product       
class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        my_list = []
        for index, num in enumerate(nums):
            for n in nums[index+1:]:
                my_list.append((num-1)*(n-1))
        return max(my_list)
class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        prod = 0
        for index in range(len(nums) - 1):
            for subindex in range(index + 1, len(nums)):
                cache = (nums[index] - 1) * (nums[subindex] - 1)
                if cache > prod:
                    prod = cache
        return prod

class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        return max([(nums[i] - 1) * (nums[j] - 1) 
                    for i in range(len(nums))
                    for j in range(i + 1, len(nums))
                   ])
class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        if nums[0] > nums[1]:
            max1 = nums[0]
            max2 = nums[1]
        else:
            max1 = nums[1]
            max2 = nums[0]
            
        for i in range(2, len(nums)):
            if nums[i] > max1:
                max2 = max1
                max1 = nums[i]
            elif nums[i] > max2:
                max2 = nums[i]
                
        return (max1 - 1)*(max2 - 1)
class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        max_sum = 0
        n = len(nums)
        
        for i in range(n):
            for j in range(i+1, n):
                sum = (nums[i] - 1) * (nums[j] - 1)
                if sum > max_sum:
                    max_sum = sum
        return max_sum

class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        return max((nums[i]-1)*(nums[j]-1) for i in range(len(nums)-1) for j in range(i+1, len(nums)))
class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        d = len(nums)
        m = None
        
        for i in range(d):
            for j in range(i+1, d):
                mn = (nums[i] - 1) * (nums[j] - 1)
                if m is None:
                    m = mn
                elif m < mn:
                    m = mn
        return m
        

class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        
        n = len(nums)
        ans = 0
        
        for i in range(n-1):
            for j in range(i+1,n):
                
                temp = (nums[i] -1) * (nums[j] -1)
                
                if temp > ans:
                    ans = temp
        
        return ans
class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        list1 = [((nums[i]-1)*(nums[j]-1)) for i in range(len(nums)) for j in range(i+1,len(nums))]
        return max(list1)

class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        return max((nums[i]-1)*(nums[j]-1) for i in range(len(nums)) for j in range(i+1, len(nums)))
class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        
        for i in range(len(nums)): 
            for j in range(0, len(nums)-i-1): 
                if nums[j] > nums[j+1] : 
                    nums[j], nums[j+1] = nums[j+1], nums[j] 
        
        a = nums.pop(len(nums)-1)-1
        b = nums.pop(len(nums)-1)-1    
        return a*b
class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        curr_product = 0
        max_product = 0
        for i in range(len(nums)-1):
            for j in range(i+1, len(nums)):
                if i == 0 and j == 0:
                    max_product = (nums[i]-1)*(nums[j]-1)
                else:
                    curr_product = (nums[i]-1)*(nums[j]-1)
                    if curr_product > max_product:
                        max_product = curr_product
        
        return max_product
class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        
        max_val = 0
        curr_val = 0
        for i in range(len(nums)):
            for j in range(i+1, len(nums)):
                if i == 0 and j == 0:
                    max_val = (nums[i]-1) * (nums[j]-1)
                else:
                    curr_val = (nums[i]-1) * (nums[j]-1)
                    if curr_val > max_val:
                        max_val = curr_val
                        
        return max_val
class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        nums.sort()
        
        return (nums[-1]-1)*(nums[-2]-1)
class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        maxValue = 0
        max_return = 0
        for nr_idx in range(len(nums)):
            for nr_id in range(nr_idx + 1, len(nums)):
                if nr_id != nr_idx:
                    max_Value = (nums[nr_idx] - 1) * (nums[nr_id] - 1)
                    if max_Value > max_return:
                        max_return = max_Value
        return max_return
class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        max=  0
        
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                curr = (nums[i] - 1) * (nums[j] -1)
                if curr > max:
                    max = curr
    
        return max
class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        maxx=float('-inf')
        for i in range(0,len(nums)-1):
            for j in range(i+1,len(nums)):
                if maxx<(nums[i]-1)*(nums[j]-1):
                    maxx=(nums[i]-1)*(nums[j]-1)
        return maxx

class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        result = 0
        for i in range(len(nums)):
            for k in range(i+1,len(nums)):
                if(nums[i]-1)*(nums[k]-1) > result:
                    result = (nums[i]-1) * (nums[k]-1)
        return result
class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        maximum = (nums[0] - 1)*(nums[1] - 1)
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                if (nums[i] - 1)*(nums[j] - 1) > maximum:
                    maximum = (nums[i] - 1)*(nums[j] - 1)
        return maximum

class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        if len(set(nums))==1 and set(nums)=={1}:
            return 0
        maxi=1
        for i in range(len(nums)-1):
            for j in range(i+1,len(nums)):
                if (nums[i]-1)*(nums[j]-1)>maxi:
                    maxi=(nums[i]-1)*(nums[j]-1)
        return maxi
class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        result = [] # variabel untuk menampung hasil perhitungan rumus
        for i in range(len(nums)-1):
            for j in range(i+1, len(nums)):
                result.append((nums[i]-1)*(nums[j]-1))
        return max(result) # mengambil nilai maximum dari rumus

class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        
        product_list = []
        
        for i in range(len(nums)):
            for j in range(i+1,len(nums)):
                product_list.append((nums[i]-1)*(nums[j]-1))
        return max(product_list)
        

class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        mpr = []
        
        for i in range(len(nums)):
            for j in range(i+1, len(nums)):
                mpr.append((nums[i]-1)*(nums[j]-1))
        return max(mpr)
class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        p = []
        for count, n in enumerate(nums):
            for i in range(count+1,len(nums)):
                p.append((nums[i] -1)* (n-1))
        return max(p)
class Solution:
    def maxProduct(self, L: List[int]) -> int:
        max=0
        x=0
        y=0
        for i in range(len(L)):
            for j in range(len(L)):
                if L[i]*L[j]>max and i!=j:
                    max=L[i]*L[j]
                    x,y=i,j
        return (L[x]-1)*(L[y]-1)
class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        max_product = 0
        num_len = len(nums)
        for i in range(num_len):
            for j in range(i, num_len):
                product = (nums[i] - 1) * (nums[j] - 1)
                if i != j and product > max_product:
                    max_product = product
        return max_product
class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        res=0
        for i in range(len(nums)-1):
            for j in range(i+1,len(nums)):
                if (nums[i]-1)*(nums[j]-1)>res:
                    res=(nums[i]-1)*(nums[j]-1)
        return res

class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        mularray = []
        for i in range(len(nums)-1):
            for j in range(i+1, len(nums)):
                mularray.append((nums[i]-1)*(nums[j]-1))
        return max(mularray)

class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        products = []
        for i in range(len(nums)):
            for j in range(i+1,len(nums)):
                products.append((nums[i]-1)*(nums[j]-1))
        return max(products)
class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        l=[]
        for i in range(0,len(nums)):
            for j in range(i+1,len(nums)):
                l.append((nums[i]-1)*(nums[j]-1))
        return max(l)
            

class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        if not nums: return 0
        max_val=i_value=y_value = 0
        for i_idx, i_val in enumerate(nums):
            for y_idx, y_val in enumerate(nums):
                if i_idx == y_idx:
                    continue
                if i_val * y_val > max_val : 
                    max_val = i_val * y_val
                    i_value,y_value=i_val,y_val
        return (i_value-1)*(y_value-1)
            

class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        n=[i-1 for i in nums]
        res=0
        for i,j in itertools.combinations(n, 2):
            res=max(res,i*j)
        return res

from itertools import combinations

class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        
        arr = combinations(nums, 2)
        res = []
        
        for x in arr:
            res.append((x[0] - 1) * (x[1] - 1))
        
        return max(res)
