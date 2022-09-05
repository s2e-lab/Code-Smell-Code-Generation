class Solution:
     def findDuplicate(self, nums):
         """
         :type nums: List[int]
         :rtype: int
         """
         if len(nums) == 0:
             return None
         slow = fast = nums[0]
         while True:
             slow = nums[slow]
             fast = nums[nums[fast]]
             if slow == fast:
                 break
         fast = nums[0]
         while slow != fast:
             slow = nums[slow]
             fast = nums[fast]
         return slow
class Solution:
     def findDuplicate(self, nums):
         """
         :type nums: List[int]
         :rtype: int
         """
 
         # similar to find cycle in the linked list
         # actually, here we have more than 1 logical linked list(with or without cycle)
         # begin with nums[0] can help us determining at least one cycled linked list of them
         fast, slow = nums[0], nums[0]
         while(True):
             slow = nums[slow]
             fast = nums[fast]
             fast = nums[fast]
             if slow == fast:
                 break
 
         ptr1 = nums[0]
         ptr2 = fast
         while(ptr1 != ptr2):
             ptr1 = nums[ptr1]
             ptr2 = nums[ptr2]
 
         return ptr1
class Solution:
     def findDuplicate(self, nums):
         """
         :type nums: List[int]
         :rtype: int
         """
         st = set()
         for i in range(len(nums)):
             n = nums[i]
             if n not in st:
                 st.add(n)
             else:
                 return n

class Solution:
     def findDuplicate(self, nums):
         """
         :type nums: List[int]
         :rtype: int
         """
         slow = nums[0]
         fast = nums[0]
         while True:
             slow = nums[slow]
             fast = nums[nums[fast]]
             if slow == fast:
                 break
         
         ptr1 = nums[0]
         ptr2 = slow
         while ptr1 != ptr2:
             ptr1 = nums[ptr1]
             ptr2 = nums[ptr2]
         
         return ptr1
         
         

class Solution:
     def findDuplicate(self, nums):
         """
         :type nums: List[int]
         :rtype: int
         """
         from collections import Counter
         c = Counter(nums)
         return list(filter(lambda x: x[1] > 1, c.items()))[0][0]
class Solution:
     def findDuplicate(self, nums):
         """
         :type nums: List[int]
         :rtype: int
         """
         low, high = 1, len(nums) - 1
         while (low < high):
             mid = (low + high)//2
             count = 0
             for i in nums:
                 if i <= mid:
                     count += 1
             if count <= mid:
                 low = mid + 1
             else:
                 high = mid
         return low
class Solution:
     def findDuplicate(self, nums):
         """
         :type nums: List[int]
         :rtype: int
         """
         nums.sort()
         i=0
         while i < len(nums):
             if nums[i]==nums[i+1]:
                 return nums[i]
             i+=1

class Solution:
     def findDuplicate(self, nums):
         """
         :type nums: List[int]
         :rtype: int
         """
         for n in nums:
             if n < 0:
                 i = -n
             else:
                 i = n
             ind = i - 1
             if nums[ind] < 0:
                 return i
             else:
                 
                 nums[ind] = -nums[ind]
         
 
         return extra

class Solution:
     def findDuplicate(self, nums):
         nums.sort()
         for i in range(1, len(nums)):
             if nums[i] == nums[i-1]:
                 return nums[i]

class Solution:
     def findDuplicate(self, nums):
         """
         :type nums: List[int]
         :rtype: int
         """
         nums = sorted(nums)
         
         for i in range(1, len(nums)):
             if nums[i] == nums[i - 1]:
                 return nums[i]

class Solution:
     def findDuplicate(self, nums):
         """
         :type nums: List[int]
         :rtype: int
         """
         nums.sort()
         print(nums)
         for i in range(len(nums)-1):
             if nums[i] == nums[i+1]:
                 return nums[i]
             

class Solution:
     def findDuplicate(self, nums):
         """
         :type nums: List[int]
         :rtype: int
         """
 #        d = {}
 #        for i in nums:
 #            if i in d:
 #                return i
 #            d[i] = 0
         return (sum(nums)-sum(set(nums))) // (len(nums)-len(set(nums)))
class Solution:
     def findDuplicate(self, nums):
         """
         :type nums: List[int]
         :rtype: int
         """
         # set_sum = sum(list(set(nums)))
         # set_len = len(set(nums))
         # pro_sum = sum(nums)
         # pro_len = len(nums)
         # return (pro_sum - set_sum) // (pro_len - set_len) 
         
         left, right = 0, len(nums) - 1
         mid = (left + right) // 2
         while right - left > 1:
             count = 0
             for num in nums:
                 if mid < num <= right:
                     count += 1
             if count > (right - mid):
                 left = mid
             else:
                 right = mid
             mid = (left + right) // 2
         return right

class Solution:
     def findDuplicate(self, nums):
         """
         :type nums: List[int]
         :rtype: int
         """
         nums.sort()
         prev = None
         for i in nums:
             if prev == i:
                 return prev
             else:
                 prev = i
class Solution:
     def findDuplicate(self, nums):
         """
         :type nums: List[int]
         :rtype: int
         """
         
         n = len(nums) - 1
         
         a = 1
         b = n
         
         while a < b:
             m = (a+b)//2
             #print(a,m,b)
             lCount = 0
             hCount = 0
             for k in nums:
                 if a <= k <= m:
                     lCount += 1
                 elif m < k <= b:
                     hCount += 1
             #print(lCount, hCount)
             if lCount > m-a+1:
                 b = m
             else:
                 a = m+1
         
         return a

