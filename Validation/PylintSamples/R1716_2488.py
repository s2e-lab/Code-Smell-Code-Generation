class Solution:
  
    def findNumbers(self, nums: List[int]) -> int:
        def has_even_digits(number: int):
            if number < 10:
                return False
            elif number < 100:
                return True
            elif number < 1000:
                return False
            elif number < 10000:
                return True
            elif number < 100000:
                return False
            return True

        return sum([1 for num in nums if has_even_digits(num)])

class Solution:
    def findNumbers(self, nums: List[int]) -> int:
        return len([i for (i) in nums if len(str(i))%2==0 ])

class Solution:
    def findNumbers(self, nums: List[int]) -> int:
        return sum([1 if(len(str(num))%2 == 0) else 0 for num in nums])
class Solution:
    def findNumbers(self, nums: List[int]) -> int:
        count = 0
        for i in nums:
            if len(str(i)) % 2==0:
                count += 1
        return count
            

class Solution:
    def findNumbers(self, nums: List[int]) -> int:
        even_cnt = 0
        for num in nums :
            if (len(str(num)) % 2 == 0) : even_cnt += 1
                
        return even_cnt
class Solution:
    def findNumbers(self, nums: List[int]) -> int:
        ec = 0
        digit = 0
        for value in nums:
            digit = len(str(value))
            if digit % 2 == 0:
                ec += 1                
        return ec
class Solution:
    def findNumbers(self, nums: List[int]) -> int:
        count = 0
        track = 0
        for num in nums:
            count = 0
            if len(str(num)) % 2 ==0:
                track += 1
        return track
class Solution:
    def findNumbers(self, nums: List[int]) -> int:
        count = 0
        string_map = map(str, nums)
        for i in string_map:
            if len(i) % 2 == 0:
                count+=1
        return count
class Solution:
    def findNumbers(self, nums: List[int]) -> int:
        nums.sort()
        # pt = [ 10 ** m for m in range(6)]
        ans = 0
        for i in nums:
            if (i >= 10 and i < 100) or (i >= 1000 and i < 10000) or (i >= 100000 and i < 1000000):
                ans += 1
        return ans
class Solution:
    def findNumbers(self, nums: List[int]) -> int:
        count = 0
        for n in nums:
            if len(str(n)) % 2 == 0:
                count += 1
        return count
class Solution:
    def findNumbers(self, nums: List[int]) -> int:
        result = 0
        for num in nums:
            count = 0 
            while num:
                num = num//10
                count +=1
            if count%2 == 0:
                result+=1
                
                
        return result

class Solution:
    def digits(self,num):
        count=0
        while num:
            num=num//10
            count+=1
        if count%2==0:
            return True
        else:
            return False
        
    def findNumbers(self, nums: List[int]) -> int:
        c=0
        for i in nums:
            if Solution().digits(i):
                c+=1
        return c
        

class Solution:
    def findNumbers(self, nums: List[int]) -> int:
        count = 0
        for num in nums:
            if len(str(num)) % 2 == 0:
                count += 1
        return count
