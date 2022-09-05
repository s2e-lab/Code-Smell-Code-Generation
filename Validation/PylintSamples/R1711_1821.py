class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        nums.sort()
        return nums
import heapq
class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        heap = []
        
        for num in nums:
            heapq.heappush(heap, num)
        
        
        sorted_nums = []        
        
        while heap:
            cur_min = heapq.heappop(heap)
            sorted_nums.append(cur_min)
            
            
        
        return sorted_nums
            
            
            

class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        if len(nums) <= 1:
            return nums
        mid = int(len(nums)/2)
        left = self.sortArray(nums[0:mid])
        right = self.sortArray(nums[mid:])
        return self.merge(left, right)
    def merge(self, left, right):
        i = j = 0
        ret = []
        while i < len(left) and j < len(right):
            if left[i] < right[j]:
                ret.append(left[i])
                i += 1
            elif right[j] <= left[i]:
                ret.append(right[j])
                j += 1
        ret.extend(left[i:])
        ret.extend(right[j:])
        return ret
class Node:
    def __init__(self, val):
        self.val = val
	
	# lt means less than, le means less or equal than etc.
    def __lt__(self, other):
        return self.val < other.val
    
class Solution:    
    def sortArray(self, nums: List[int]) -> List[int]:
        nodes = [Node(n) for n in nums]
        return [node.val for node in sorted(nodes)]
class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        if len(nums) <= 1:
            return nums
        n = len(nums)
        left = self.sortArray(nums[:n//2])
        right = self.sortArray(nums[n//2:])
        n_left = len(left)
        n_right = len(right)
        combine = []
        while n_left > 0 or n_right > 0:
            if n_left and n_right:
                if left[0] > right[0]:
                    combine.append(right.pop(0))
                    n_right -= 1
                else:
                    combine.append(left.pop(0))
                    n_left -= 1
            elif n_left and not n_right:
                combine += left
                n_left = 0
            elif n_right and not n_left:
                combine += right
                n_right = 0
        return combine

class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        def merge(a, b):
            l1 = l2 = 0
            r1, r2 = len(a), len(b)
            i, j = l1, l2
            
            out = []
            while i < r1 or j < r2:
                if j == r2 or i < r1 and a[i] < b[j]:
                    out.append(a[i])
                    i += 1
                elif j < r2:
                    out.append(b[j])
                    j += 1
            return out
        
        skip_interval = 1
        while skip_interval < len(nums):
            for i in range(0, len(nums), 2*skip_interval):
                middle = i + skip_interval
                nums[i: i + 2*skip_interval] = merge(nums[i: middle], nums[middle : middle + skip_interval])
            
            skip_interval *= 2
            
        return nums
class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        
        # recursive merge sort
        
        def merge_sort(nums):
            
            if len(nums) <= 1:
                return nums
            
            mid = len(nums)//2
            
            left_list = merge_sort(nums[:mid])
            right_list = merge_sort(nums[mid:])
            #print(left_list, right_list)
            return merge(left_list, right_list)
            
        def merge(left_list, right_list):
            
            if not left_list:
                return right_list
            if not right_list:
                return left_list
            
            left_ptr = right_ptr = 0
            out = []
            while left_ptr < len(left_list) and right_ptr < len(right_list):
                
                if left_list[left_ptr] <= right_list[right_ptr]:
                    out.append(left_list[left_ptr])
                    left_ptr += 1
                else:
                    out.append(right_list[right_ptr])
                    right_ptr += 1
                    
            out.extend(left_list[left_ptr:])
            out.extend(right_list[right_ptr:])
            return out
            
        return merge_sort(nums)

class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        return self.quicksort(nums)
        
    def quicksort(self, nums):
        if len(nums) == 1 or len(nums) == 0:
            return nums
        pivot = nums[len(nums)//2]
        left = [x for x in nums if x<pivot]
        mid = [x for x in nums if x==pivot]
        right = [x for x in nums if x>pivot]
        return self.quicksort(left) + mid + self.quicksort(right)
        

class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        if len(nums)<=1:
            return nums
        else:
            numA = self.sortArray(nums[:len(nums)//2])
            numB = self.sortArray(nums[len(nums)//2:])
            i,j = 0,0
            nums = []
            while i<len(numA) and j<len(numB):
                if numA[i]<=numB[j]:
                    nums.append(numA[i])
                    i+=1
                else:
                    nums.append(numB[j])
                    j+=1
            if i==len(numA):
                nums+=numB[j:]
            else:
                nums+=numA[i:]
        return nums

class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        def merge(arr1, arr2):
            if not arr1:
                return arr2
            if not arr2:
                return arr1
            
            res = []
            a1 = 0
            a2 = 0
            
            while a1 < len(arr1) or a2 < len(arr2):
                if a1 == len(arr1):
                    res.append(arr2[a2])
                    a2 += 1
                    continue
                if a2 == len(arr2):
                    res.append(arr1[a1])
                    a1 += 1
                    continue
                
                if arr1[a1] < arr2[a2]:
                    res.append(arr1[a1])
                    a1 += 1
                else:
                    res.append(arr2[a2])
                    a2 += 1
            return res
        
        def mergesort(arr):
            if len(arr) == 1:
                return arr
            mid = len(arr) // 2
            left = mergesort(arr[:mid])
            right = mergesort(arr[mid:])
            return merge(left, right)
        return mergesort(nums)
class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        if len(nums) <= 1:
            return nums
        mid = len(nums) // 2
        left = nums[:mid]
        right = nums[mid:]
        left = self.sortArray(left)
        right = self.sortArray(right)
        res = []
        while len(left) > 0 and len(right) > 0:
            if left[0] < right[0]:
                res.append(left.pop(0))
            else:
                res.append(right.pop(0))
        res = res + left + right
        return res
class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        
            
        def merge(left_list, right_list):
            
            if not left_list:
                return right_list
            if not right_list:
                return left_list
            
            left_ptr = right_ptr = 0
            out = []
            while left_ptr < len(left_list) and right_ptr < len(right_list):
                
                if left_list[left_ptr] <= right_list[right_ptr]:
                    out.append(left_list[left_ptr])
                    left_ptr += 1
                else:
                    out.append(right_list[right_ptr])
                    right_ptr += 1
                    
            out.extend(left_list[left_ptr:])
            out.extend(right_list[right_ptr:])
            return out
                
        new_list = [[num] for num in nums]
        #print(new_list)
        #i = 0
        #out = []
        while len(new_list) > 1:
            out = []
            i = 0
            #print(new_list)
            for i in range(0, len(new_list), 2):
                
                if i == len(new_list)-1:
                    merged_list = merge(new_list[i], [])
                else:
                    merged_list = merge(new_list[i], new_list[i+1])
                out.append(merged_list)
            new_list = out
               # if i+1 == len(new_list)-1:
                    
                
        return new_list[0]
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        # recursive merge sort
        
#         def merge_sort(nums):
            
#             if len(nums) <= 1:
#                 return nums
            
#             mid = len(nums)//2
            
#             left_list = merge_sort(nums[:mid])
#             right_list = merge_sort(nums[mid:])
#             #print(left_list, right_list)
#             return merge(left_list, right_list)
            
#         def merge(left_list, right_list):
            
#             if not left_list:
#                 return right_list
#             if not right_list:
#                 return left_list
            
#             left_ptr = right_ptr = 0
#             out = []
#             while left_ptr < len(left_list) and right_ptr < len(right_list):
                
#                 if left_list[left_ptr] <= right_list[right_ptr]:
#                     out.append(left_list[left_ptr])
#                     left_ptr += 1
#                 else:
#                     out.append(right_list[right_ptr])
#                     right_ptr += 1
                    
#             out.extend(left_list[left_ptr:])
#             out.extend(right_list[right_ptr:])
#             return out
            
#         return merge_sort(nums)

class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        import heapq
        heap =[]
        for i in nums:
            heapq.heappush(heap, i)
        sorted_list=[]
        while heap:
            sorted_list.append(heapq.heappop(heap))
        return sorted_list

class Solution:

    def sortArray(self, nums: List[int]) -> List[int]:
        def heapSort(arr): 
            n = len(arr) 
            for i in range(n//2 - 1, -1, -1): 
                heapify(arr, n, i) 
            for i in range(n-1, 0, -1): 
                arr[i], arr[0] = arr[0], arr[i]
                heapify(arr, i, 0) 
                
        def heapify(arr, n, i): 
            largest = i  
            l = 2 * i + 1     
            r = 2 * i + 2    

            if l < n and arr[i] < arr[l]: 
                largest = l 

            if r < n and arr[largest] < arr[r]: 
                largest = r 

            if largest != i: 
                arr[i],arr[largest] = arr[largest],arr[i] 
                heapify(arr, n, largest) 


 
        heapSort(nums) 
        return nums
    

class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        length = len(nums)
        start_node = (length-2) // 2
        for i in range(start_node, -1, -1):
            self.build(nums, i, length)
        for i in range(length-1, 0, -1):
            nums[0], nums[i] = nums[i], nums[0]
            self.build(nums, 0, i)
        return nums
#     def sortArray(self, nums: List[int]) -> List[int]:
#         length = len(nums)
#         for i in range(length - 1, -1, -1):
#             self.build(nums, i, length)
        
#         for i in range(length - 1, 0, -1):
#             nums[0], nums[i] = nums[i], nums[0]
#             self.build(nums, 0, i)
#         return nums    
    
    
    def build(self, nums, node, n):
        left = node*2 + 1
        right = node*2 + 2
        large = node
        
        if left < n and nums[left] > nums[large]:
            large = left
        if right < n and nums[right] > nums[large]:
            large = right
        if large != node:
            nums[large], nums[node] = nums[node], nums[large]
            self.build(nums, large, n)
class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        # Heap Sort
        # Step 1: 
        for i in range(len(nums) // 2, 0, -1):
            self.heapify(nums, i, len(nums)+1)
        # Step 2:
        for i in range(len(nums), 0, -1):
            index = i - 1
            nums[0], nums[index] = nums[index], nums[0]
            self.heapify(nums, 1, i)
        
        return nums

    def heapify(self, nums: List[int], index, length):
        left = index * 2
        right = left + 1

        if left >= length:
            return
        
        if right >= length:
            if (nums[index-1] < nums[left-1]):
                nums[index-1], nums[left-1] = nums[left-1], nums[index-1]
                self.heapify(nums, left, length)
            return

        if nums[left-1] < nums[right-1]:
            if (nums[index-1] < nums[right-1]):
                nums[index-1], nums[right-1] = nums[right-1], nums[index-1]
                self.heapify(nums, right, length)
        else:
            if (nums[index-1] < nums[left-1]):
                nums[index-1], nums[left-1] = nums[left-1], nums[index-1]
                self.heapify(nums, left, length)
class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        
        nums = self.merge(nums)
        return nums
    
    def merge(self, values):
        if len(values)>1: 
            m = len(values)//2
            left = values[:m] 
            right = values[m:] 
            left = self.merge(left) 
            right = self.merge(right) 

            values =[] 

            while len(left)>0 and len(right)>0: 
                if left[0]<right[0]: 
                    values.append(left[0]) 
                    left.pop(0) 
                else: 
                    values.append(right[0]) 
                    right.pop(0) 

            for i in left: 
                values.append(i) 
            for i in right: 
                values.append(i) 
                  
        return values 
        

class Solution:
   def sortArray(self, nums: List[int]) -> List[int]:
       if len(nums)>1: 
           m = len(nums)//2
           left = nums[:m] 
           right = nums[m:] 
           left = self.sortArray(left) 
           right = self.sortArray(right) 
           nums =[] 
           while len(left)>0 and len(right)>0: 
               if left[0]<right[0]: 
                   nums.append(left[0]) 
                   left.pop(0) 
               else: 
                   nums.append(right[0]) 
                   right.pop(0) 
 
           for i in left:
               nums.append(i) 
           for i in right: 
               nums.append(i) 
                 
       return nums
class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        # Heap Sort
        # Step 1: 
        for i in range(len(nums) // 2, 0, -1):
            self.heapify(nums, i, len(nums)+1)
        # Step 2:
        for i in range(len(nums), 0, -1):
            index = i - 1
            nums[0], nums[index] = nums[index], nums[0]
            self.heapify(nums, 1, i)
        
        return nums

    def heapify(self, nums: List[int], index, length):
        left = index * 2
        right = left + 1

        if (left >= length):
            return
        
        if (right >= length):
            if (nums[index-1] < nums[left-1]):
                nums[index-1], nums[left-1] = nums[left-1], nums[index-1]
                self.heapify(nums, left, length)
            return

        if (nums[left-1] < nums[right-1]):
            greater = right
        else:
            greater = left

        if (nums[index-1] < nums[greater-1]):
            nums[index-1], nums[greater-1] = nums[greater-1], nums[index-1]
            self.heapify(nums, greater, length)
class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        length = len(nums)
        for i in range(length - 1, -1, -1):
            self.maxheap(nums, length, i)
        
        for i in range(length - 1, 0, -1):
            nums[0], nums[i] = nums[i], nums[0]
            self.maxheap(nums, i, 0)
        return nums


    def maxheap(self, nums, n, node):
        l = node * 2 + 1
        r = node * 2 + 2
        large = node

        if l < n and nums[l] > nums[large]:
            large = l
        if r < n and nums[r] > nums[large]:
            large = r
        if large != node:
            nums[node], nums[large] = nums[large], nums[node]
            self.maxheap(nums, n, large)

class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        if len(nums) == 1:
            return nums
        elif len(nums) == 2:
            return nums if nums[0] <= nums[1] else [nums[1], nums[0]]
        else:
            half = int(len(nums)/2)
            a1 = self.sortArray(nums[:half])
            a2 = self.sortArray(nums[half:])
            nu = []
            olen = len(a1) + len(a2)
            while len(nu) < olen:
                if len(a1) == 0:
                    nu.append(a2.pop(0))
                elif len(a2) == 0:
                    nu.append(a1.pop(0))
                elif a1[0] < a2[0]:
                    nu.append(a1.pop(0))
                else:
                    nu.append(a2.pop(0))
            return nu

from random import shuffle
class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        size = 1
        aux = [0] * len(nums)
        # merge all subarrays with length of size
        while size < len(nums):
            for lo in range(0, len(nums)-size, 2*size):
                # merge two arrays
                # last array could be smaller than size
                hi = min(len(nums)-1, lo+2*size-1)
                aux[lo:hi+1] = nums[lo:hi+1]
                i, j = lo, lo+size
                for k in range(lo, hi+1):
                    if i > lo+size-1:
                        # left subarray is exhausted
                        nums[k] = aux[j]
                        j += 1
                    elif j > hi:
                        # right subarry is exhausted
                        nums[k] = aux[i]
                        i += 1
                    elif aux[i] > aux[j]:
                        nums[k] = aux[j]
                        j += 1
                    else:
                        nums[k] = aux[i]
                        i += 1
            size *= 2
        return nums
                
            
        

class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        lst = []
        
        for x in nums:
            heapq.heappush(lst,x)
            
        return [ heapq.heappop(lst) for x in range(len(nums))]
class Solution:  
    def maxheapify(self, a, heapsize,i):
        l, r = 2*i+1, 2*i+2
        leftisgreater = rightisgreater = False
        if l < heapsize and a[i] < a[l]:
            leftisgreater = True
        if r < heapsize and a[i] < a[r]:
            rightisgreater = True
        
        if leftisgreater and not rightisgreater:
            a[i],a[l] = a[l],a[i]
            self.maxheapify(a, heapsize, l)
        elif not leftisgreater and rightisgreater:
            a[i],a[r] = a[r],a[i]
            self.maxheapify(a, heapsize, r)
        elif leftisgreater and rightisgreater:
            if a[l] <= a[r]:
                a[i],a[r] = a[r],a[i]
                self.maxheapify(a, heapsize, r)
            else:
                a[i],a[l] = a[l],a[i]
                self.maxheapify(a, heapsize, l)
                    
    def buildmaxheap(self, nums, heapsize):
        for i in reversed(range(len(nums)//2)):
            self.maxheapify(nums, heapsize,i)
           
    def heapsort(self, nums):
        heapsize = len(nums)
        self.buildmaxheap(nums, heapsize)
        for i in range(len(nums)):
            nums[0],nums[heapsize-1]=nums[heapsize-1],nums[0]
            heapsize-=1
            self.maxheapify(nums, heapsize, 0)
        
    def sortArray(self, nums: List[int]) -> List[int]:
        self.heapsort(nums)
        return nums
class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        L = len(nums)
        if L == 1:
            # print(f'nums: {nums}')
            return nums
        else:
            left = nums[:L//2]
            right = nums[L//2:]
            #print(left, "  ", right)
            return self.compare(self.sortArray(left), self.sortArray(right))
    
    def compare(self, left, right):
        combined = []
        # print(f'before sort: {left}  {right}   {combined}')
        while len(left) > 0 and len(right) > 0:
            if left[0] > right[0]:
                combined.append(right.pop(0))
            elif left[0] < right[0]:
                combined.append(left.pop(0))
            else:  # equal values, pop both to save an iteration.
                combined.append(right.pop(0))
                combined.append(left.pop(0))
        combined.extend(left)  # one will be empty, doesn't matter which
        combined.extend(right)  # one will be empty, doesn't matter which
        # print(f'after sort: {left}  {right}   {combined}')
        return combined        
def swap(arr, i, j):
    arr[i], arr[j] = arr[j], arr[i]

def heapify(nums, n, i):
    l = 2 * i
    r = 2 * i + 1
    largest = i
    
    if l < n and nums[l] > nums[largest]:
        largest = l
    if r < n and nums[r] > nums[largest]:
        largest = r
    if largest != i:
        swap(nums, i, largest)
        heapify(nums, n, largest)

def build_heap(nums):
    n = len(nums)
    for i in range(n // 2, -1, -1):
        heapify(nums, n, i)

        
def heap_sort(nums):
    n = len(nums)
    for i in range(n - 1, -1, -1):
        swap(nums, 0, i)
        heapify(nums, i, 0)

class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        build_heap(nums)
        heap_sort(nums)
        return nums
        

class Solution:
    
    def merge(self,arr1:List[int],arr2:List[int]) -> List[int]:
        ret = []
        ix1 = 0
        ix2 = 0
        while ix1 != len(arr1) and ix2 != len(arr2):
            if arr1[ix1] < arr2[ix2]:
                ret.append(arr1[ix1])
                ix1 += 1
            else:
                ret.append(arr2[ix2])
                ix2 += 1
        if ix1< len(arr1):
            ret.extend(arr1[ix1:])
        else:
            ret.extend(arr2[ix2:])
        return ret
        
    def sortArray(self, nums: List[int]) -> List[int]:
        # implement merge sort
        if len(nums) == 1:
            return nums
        
        mid = len(nums) // 2
        left = self.sortArray(nums[:mid])
        right = self.sortArray(nums[mid:])
        return self.merge(left,right)
        

class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        if len(nums)>1: 
            m = len(nums)//2
            left = nums[:m] 
            right = nums[m:] 
            left = self.sortArray(left) 
            right = self.sortArray(right) 
            nums =[] 
            while len(left)>0 and len(right)>0: 
                if left[0]<right[0]: 
                    nums.append(left[0]) 
                    left.pop(0) 
                else: 
                    nums.append(right[0]) 
                    right.pop(0) 

            for i in left:
                nums.append(i) 
            for i in right: 
                nums.append(i) 

        return nums
def merge(list1, list2):
    merged = []
    while len(list1) != 0 and len(list2) != 0:
        if list1[0] < list2[0]:
            merged.append(list1.pop(0))
        else:
            merged.append(list2.pop(0))
    if len(list1) == 0:
        return merged + list2
    else:
        return merged + list1
        

class Solution:
    # Merge Sort
    def sortArray(self, nums: List[int]) -> List[int]:
        if len(nums) == 1:
            return nums
        else:
            return merge(self.sortArray(nums[:len(nums) // 2]), self.sortArray(nums[len(nums) // 2:]))
        

def swap(arr, i, j):
    arr[i], arr[j] = arr[j], arr[i]

def heapify(nums, n, i):
    l = 2 * i
    r = 2 * i + 1
    largest = i
    
    if l < n and nums[l] > nums[largest]:
        largest = l
    if r < n and nums[r] > nums[largest]:
        largest = r
    if largest != i:
        swap(nums, i, largest)
        heapify(nums, n, largest)

class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        # build heap
        n = len(nums)
        
        # n // 2 -> 0
        # building maximizing heap
        for i in range(n // 2, -1, -1):
            heapify(nums, n, i)
        
        for i in range(n - 1, -1, -1):
            swap(nums, 0, i)
            heapify(nums, i, 0)
        
        return nums
        # do the heap sort
        

def get_numbers(arr, target, cb):
    result = []
    for num in arr:
        if cb(num, target):
            result.append(num)
    return result

def is_less(a, b):
    return a < b


def is_greater(a, b):
    return a > b

def is_equal(a, b):
    return a == b

def get_less_numbers(arr, target):
    return get_numbers(arr, target, is_less)


def get_greater_numbers(arr, target):
    return get_numbers(arr, target, is_greater)

def get_equal_numbers(arr, target):
    return get_numbers(arr, target, is_equal)

def q_sort(arr):
    if len(arr) <= 1:
        return arr

    pivot = arr[len(arr) // 2]
    less = get_less_numbers(arr, pivot)
    greater = get_greater_numbers(arr, pivot)
    mypivot = get_equal_numbers(arr, pivot)

    return q_sort(less) + mypivot + q_sort(greater)

class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        return q_sort(nums)
class Solution:
    #def __init__(self):
        
    def sortArray(self, nums: List[int]) -> List[int]:
        return self.mergeSort(nums)
    
    def mergeSort(self, nums: List[int]) -> List[int]:
        if len(nums) > 1:
            l1 = nums[:len(nums)//2]
            l2 = nums[len(nums)//2:]
            L = self.mergeSort(l1)
            R = self.mergeSort(l2)
            
            sorted_list = []
            i = j = 0
            while i < len(L) and j < len(R):
                if L[i] < R[j]:
                    sorted_list.append(L[i])
                    i += 1
                else:
                    sorted_list.append(R[j])
                    j += 1
            
            while i < len(L):
                sorted_list.append(L[i])
                i += 1
            while j < len(R):
                sorted_list.append(R[j])
                j += 1 
            return sorted_list
            
        return nums
            

class Solution:
    def _merge(self, list1, list2):
        tmp = []
        while list1 and list2:
            (tmp.append(list1.pop(0))
             if list1[0] < list2[0]
             else tmp.append(list2.pop(0)))
        return tmp + (list1 or list2)

    def sortArray(self, nums: List[int]) -> List[int]:
        pivot = len(nums) // 2
        return (nums
                if len(nums) < 2
                else self._merge(self.sortArray(nums[:pivot]),
                                self.sortArray(nums[pivot:])))
class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        # return sorted(nums)

        def merge_sort(values): 

            if len(values)>1: 
                m = len(values)//2
                left = values[:m] 
                right = values[m:] 
                left = merge_sort(left) 
                right = merge_sort(right) 
                values =[] 
                while len(left)>0 and len(right)>0: 

                    if left[0]<right[0]: 
                        values.append(left[0]) 
                        left.pop(0)
                      
                    else: 
                        values.append(right[0]) 
                        right.pop(0)
                       
                for i in left: 
                    values.append(i) 
                for i in right: 
                    values.append(i) 
                   

            return values 
        result = merge_sort(nums)
        return result

class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        if len(nums) < 2:
            return nums
        
        def merge(l1, l2):
            n1 = n2 = 0
            res = []
            
            while n1 < len(l1) and n2 < len(l2):
                if l1[n1] < l2[n2]:
                    res.append(l1[n1])
                    n1 += 1
                else:
                    res.append(l2[n2])
                    n2 += 1
                
            
            res += l1[n1:]
            res += l2[n2:]
            
            return res
        
        mid = len(nums) // 2
        
        return merge(self.sortArray(nums[:mid]), self.sortArray(nums[mid:]))
        
        


from collections import deque


class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        # heap sort
        # convert array to max heap
        for i in range(len(nums) // 2 - 1, -1, -1):
            self.heapify(nums, i, len(nums))
        # have pointer start at last element
        last = len(nums) - 1
        while last > 0:
            # swap last element with first element
            nums[last], nums[0] = nums[0], nums[last]
            # restore heap property from [:pointer]
            self.heapify(nums, 0, last)
            # decrement pointer
            last -= 1
        return nums

    def heapify(self, nums, start, size):
        largest = start
        left = start * 2 + 1
        right = start * 2 + 2

        if left < size and nums[largest] < nums[left]:
            largest = left

        if right < size and nums[largest] < nums[right]:
            largest = right

        if largest != start:
            nums[start], nums[largest] = nums[largest], nums[start]
            self.heapify(nums, largest, size)


#     #     nums = deque(nums)
#     #     if len(nums) <= 1:
#     #         return nums
#     #     mid = len(nums) // 2
#     #     left = self.sortArray(nums[:mid])
#     #     right = self.sortArray(nums[mid:])
#     #     return self.merge(left, right)

#     # def merge(self, left, right):
#     #     merged = deque([])
#     #     while left and right:
#     #         merged.append(left.popleft() if left[0] < right[0] else right.popleft())
#     #     return merged + left + right

#     def sortArray(self, nums: List[int]) -> List[int]:
#         return self.merge_sort(nums, 0, len(nums) - 1)

#     def merge_sort(self, nums, start, end):
#         if end <= start:
#             return nums
#         mid = start + (end - start) // 2
#         self.merge_sort(nums, start, mid)
#         self.merge_sort(nums, mid + 1, end)
#         return self.merge(nums, start, mid, end)

#     def merge(self, nums, start, mid, end):
#         left, right = [], []
#         for i in range(start, mid + 1):
#             left.append(nums[i])
#         for j in range(mid + 1, end + 1):
#             right.append(nums[j])
#         i, j, k = 0, 0, start

#         # pick out smallest elements until smaller list is exhausted
#         while i < len(left) and j < len(right):
#             if left[i] < right[j]:
#                 nums[k] = left[i]
#                 i += 1
#             else:
#                 nums[k] = right[j]
#                 j += 1
#             k += 1

#         # copy remaining elements
#         while i < len(left):
#             nums[k] = left[i]
#             i += 1
#             k += 1

#         while j < len(right):
#             nums[k] = right[j]
#             j += 1
#             k += 1
#         return nums


class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        if len(nums) <= 1:
            return nums
        
        pivot = len(nums)//2
        
        left_arr = self.sortArray(nums[:pivot])
        right_arr = self.sortArray(nums[pivot:])
        
        return self.merge(left_arr, right_arr)
    
    def merge(self, left_nums, right_nums):
        m, n = len(left_nums), len(right_nums)
        i, j = 0, 0
        
        combined_arr = []
        while i < m and j < n:
            if left_nums[i] < right_nums[j]:
                combined_arr.append(left_nums[i])
                i += 1
            else:
                combined_arr.append(right_nums[j])
                j += 1
                
        combined_arr.extend(left_nums[i:])
        combined_arr.extend(right_nums[j:])

        return combined_arr

class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        self.qsort(nums, 0, len(nums))
        return nums
    
    def qsort(self, nums, begin, end):
        if begin>=end:
            return
        x = nums[end-1]
        i = j = begin
        while j<end-1:
            if nums[j]<x:
                self.swap(nums, i, j)
                i += 1
            j += 1
        self.swap(nums, i, j)
        self.qsort(nums, begin, i)
        self.qsort(nums, i+1, end)
        
    def swap(self, nums, i, j):
        a = nums[i]
        nums[i] = nums[j]
        nums[j] = a
class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        if len(nums)==1:
            return nums
        if len(nums)>1:
            mid=len(nums)//2
            left=nums[:mid]
            right=nums[mid:]
            self.sortArray(left)
            self.sortArray(right)
            self.merge(left,right,nums)
            return nums
    
    def merge(self,left,right,nums):
        i=0
        j=0
        k=0
        while i<len(left) and j<len(right):
            if left[i]<right[j]:
                nums[k]=left[i]
                i+=1
                k+=1
            else:
                nums[k]=right[j]
                j+=1
                k+=1
        while i<len(left):
            nums[k]=left[i]
            i+=1
            k+=1
        while j<len(right):
            nums[k]=right[j]
            j+=1
            k+=1
        return
class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        if len(nums) <= 1:
            return nums
        
        mid = len(nums)//2
        A = self.sortArray(nums[:mid])
        B = self.sortArray(nums[mid:])
        i,j = 0,0
        result = []
        
        while i < len(A) and j < len(B):
            if A[i] <= B[j]:
                result.append(A[i])
                i += 1
            else:
                result.append(B[j])
                j += 1
                
        if i < len(A):
            result += A[i:]
        elif j < len(B):
            result += B[j:]
            
        return result
        
        

class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        if len(nums) < 2:
            return nums
        else:
            mid = len(nums)//2
            left = nums[:mid]
            right = nums[mid:]
            self.sortArray(left)
            self.sortArray(right)
            i,j,k = 0,0,0
            while i < len(left) and j < len(right):
                if left[i] <= right[j]:
                    nums[k] = left[i]
                    i += 1
                else:
                    nums[k] = right[j]
                    j += 1
                k += 1
            while i < len(left):
                nums[k] = left[i]
                i += 1
                k += 1
            while j < len(right):
                nums[k] = right[j]
                j += 1
                k += 1
        return nums
class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        tmp = [0 for _ in range(len(nums))]
        self.merge_sort(nums, 0, len(nums)-1, tmp)
        return nums
        
        
    def merge_sort(self, nums, left, right, tmp):
        if left >= right:
            return
        mid = (left + right) // 2
        
        self.merge_sort(nums, left, mid, tmp)
        self.merge_sort(nums, mid+1, right, tmp)
        self.merge(nums, left, right, tmp)

        
        
    def merge(self, nums, left, right, tmp):
        
        n = right - left + 1
        mid = (left + right) // 2
        i, j = left, mid +1
        
        for k in range(n):
            # j>right means right side is used up
            if i <= mid and (j > right or nums[i]<=nums[j]):
                tmp[k] = nums[i]
                i += 1
            else:
                tmp[k] = nums[j]
                j += 1
        
        for k in range(n):
            nums[left + k] = tmp[k]
        

class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        counts = Counter(nums)
        nums_before = 0
        for k, v in sorted(counts.items(), key=lambda x: x[0]):
            nums_before += v
            counts[k] = nums_before-1
        out = [0 for _ in range(len(nums))]
        for i in range(len(nums)):
            out[counts[nums[i]]] = nums[i]
            counts[nums[i]] -= 1
        return out
class Solution:
    def merge(self, nums1, nums2):
        if not nums1:
            return nums2
        if not nums2:
            return nums1
        res = []
        i = j = 0
        while i < len(nums1) and j < len(nums2):
            if nums1[i] <= nums2[j]:
                res.append(nums1[i])
                i += 1
            else:
                res.append(nums2[j])
                j += 1
        
        if i < len(nums1):
            res.extend(nums1[i:])
        if j < len(nums2):
            res.extend(nums2[j:])
        
        return res
            
            
    def sortArray(self, nums: List[int]) -> List[int]:
        if len(nums) <= 1:
            return nums
        mid = len(nums) // 2
        nums1 = self.sortArray(nums[:mid])
        nums2 = self.sortArray(nums[mid:])
        res = self.merge(nums1, nums2)
        return res

from collections import deque

class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        if len(nums) <= 1:
            return nums
        mid = len(nums) // 2
        left = self.sortArray(nums[:mid])
        right = self.sortArray(nums[mid:])
        return self.merge(left, right)

    def merge(self, left, right):
        ans = []
        while left and right:
            ans.append(left.pop(0) if left[0] <= right[0] else right.pop(0))
        return ans + left + right






























#     #     nums = deque(nums)
#     #     if len(nums) <= 1:
#     #         return nums
#     #     mid = len(nums) // 2
#     #     left = self.sortArray(nums[:mid])
#     #     right = self.sortArray(nums[mid:])
#     #     return self.merge(left, right)

#     # def merge(self, left, right):
#     #     merged = deque([])
#     #     while left and right:
#     #         merged.append(left.popleft() if left[0] < right[0] else right.popleft())
#     #     return merged + left + right

#     def sortArray(self, nums: List[int]) -> List[int]:
#         return self.merge_sort(nums, 0, len(nums) - 1)

#     def merge_sort(self, nums, start, end):
#         if end <= start:
#             return nums
#         mid = start + (end - start) // 2
#         self.merge_sort(nums, start, mid)
#         self.merge_sort(nums, mid + 1, end)
#         return self.merge(nums, start, mid, end)

#     def merge(self, nums, start, mid, end):
#         left, right = [], []
#         for i in range(start, mid + 1):
#             left.append(nums[i])
#         for j in range(mid + 1, end + 1):
#             right.append(nums[j])
#         i, j, k = 0, 0, start

#         # pick out smallest elements until smaller list is exhausted
#         while i < len(left) and j < len(right):
#             if left[i] < right[j]:
#                 nums[k] = left[i]
#                 i += 1
#             else:
#                 nums[k] = right[j]
#                 j += 1
#             k += 1

#         # copy remaining elements
#         while i < len(left):
#             nums[k] = left[i]
#             i += 1
#             k += 1

#         while j < len(right):
#             nums[k] = right[j]
#             j += 1
#             k += 1
#         return nums


class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        def mergeSort(nos):
            if len(nos) > 1:
                mid = len(nos)//2
                left = nos[mid:]
                right = nos[:mid]
                left = mergeSort(left)
                right = mergeSort(right)
                nos = []
                while left and right:
                    if left[0] < right[0]:
                        nos.append(left[0])
                        left.pop(0)
                    else:
                        nos.append(right[0])
                        right.pop(0)
                for i in left:
                    nos.append(i)
                for j in right:
                    nos.append(j)
            return nos
                
        return mergeSort(nums)
class Solution:
    
    def merge(self,low,mid,high,arr):
        
        left = []
        right = []
        
        l_limit = mid+1-low
        r_limit = high+1-mid-1
        
        for i in range(low, mid+1):
            left.append(arr[i])
        
        for i in range(mid+1, high+1):
            right.append(arr[i])
            
        #print("During merge arrays are ", left, right)
        #print("limits are ",l_limit, r_limit)
        
        #using left and right as temp variables, merge it onto arr
        
        l_iter = 0
        r_iter = 0
        
        filler = low
        
        while(l_iter < l_limit or r_iter < r_limit):
            
            if(l_iter == l_limit):
                arr[filler] = right[r_iter]
                r_iter+=1
            elif(r_iter == r_limit):
                arr[filler] = left[l_iter]
                l_iter+=1
            elif(left[l_iter] < right[r_iter]):
                arr[filler] = left[l_iter]
                l_iter+=1
            else:
                arr[filler] = right[r_iter]
                r_iter+=1
                
            filler+=1
            #print(l_iter, r_iter, l_limit, r_limit)
            
    def mergeSort(self,arr, low, high):
        
        if(low < high):
            
            mid = low + (high-low)//2
            
            #print("mergesort for ", arr[low:mid+1], arr[mid+1:high+1])
            
            self.mergeSort(arr, low, mid) #from low to mid
            self.mergeSort(arr, mid+1, high) #from mid+1 to high
            
            self.merge(low,mid,high,arr)
    
    def sortArray(self, arr: List[int]) -> List[int]:
            
        n = len(arr)
        
        if(n==1):
            return arr
        else:
            self.mergeSort(arr, 0,n-1)
            return arr

class Solution:
    import random
    def sortArray(self, nums: List[int]) -> List[int]:
        
        self.heapSort(nums)
        return nums
        
                  
    def quickSort(self, arr):
        def partition(arr, l, h):
                if l>=h:
                    return
                pivot = arr[random.randint(l,r)]
                tempL, tempH = l, h
                while tempL<=tempH:
                    while tempL<=tempH and arr[tempL]<pivot:
                        tempL+=1
                    while tempL<=tempH and arr[tempH]>pivot:
                        tempH-=1
                    if tempL<=tempH:
                        arr[tempL], arr[tempH] = arr[tempH], arr[tempL]
                        tempL+=1
                        tempH-=1
                partition(arr, l, tempH)
                partition(arr, tempL, h)
                return
        partition(arr, 0, len(arr)-1)
        return arr
    
    def mergeSort(self, arr):
        if len(arr)<=1: return
        mid = len(arr)//2
        L = arr[:mid]
        R = arr[mid:]
        self.mergeSort(L)
        self.mergeSort(R)
        idxL = 0
        idxR = 0
        idxA = 0
        while idxA<len(arr):
            if idxL < len(L) and idxR < len(R):
                if L[idxL]<R[idxR]:
                    arr[idxA] = L[idxL]
                    idxL += 1
                else:
                    arr[idxA] = R[idxR]
                    idxR += 1
            elif idxL < len(L):
                arr[idxA] = L[idxL]
                idxL += 1
            else:
                arr[idxA] = R[idxR]
                idxR += 1
            idxA+=1
        return arr
    
    def heapSort(self, arr):
        def heapify(arr, arrLen, i):
            # i = index of this node
            l = 2 * i + 1
            r = l + 1
            largestIdx = i
            
            # theese two if is to finding the biggest node to make a max heap 
            if l < arrLen and arr[largestIdx] < arr[l]:
                largestIdx = l
            if r < arrLen and arr[largestIdx] < arr[r]:
                largestIdx = r
            
            # make the largest the parent
            if largestIdx!=i:
                arr[i], arr[largestIdx] = arr[largestIdx], arr[i]
                heapify(arr, arrLen, largestIdx)
        
        arrLen = len(arr)
        # heap all the tree from the middle of the array to the front
        # make the end of the array biggest
        for i in range(arrLen//2, -1, -1):
            heapify(arr, arrLen, i)
        
        # the i is the root of the tree since i, and its a max heap
        for i in range(arrLen-1, -1, -1):
            arr[i], arr[0] = arr[0], arr[i]
            heapify (arr, i, 0)
        
        
    
    
        


class Solution:
    # def sortArray(self, nums: List[int]) -> List[int]:
    #     return sorted(nums)
    
    # merge sort
    def sortArray(self, nums: List[int]) -> List[int]:
        def mergeSort(nums, start, end):
            if start >= end:
                return
            mid = start + (end - start) // 2
            mergeSort(nums, start, mid)
            mergeSort(nums, mid+1, end)
            L = [0]*(mid-start+1)
            R = [0]*(end-mid)
            n1 = len(L)
            n2 = len(R)
            for i in range(n1):
                L[i] = nums[start+i]
            for j in range(n2):
                R[j] = nums[mid+1+j]
            # two pointers
            i = j = 0
            for _ in range(start, end+1):
                if j>=n2 or (i<n1 and L[i]<=R[j]):
                    nums[start+i+j] = L[i]
                    i += 1
                else:
                    nums[start+i+j] = R[j]
                    j += 1
        mergeSort(nums, 0, len(nums)-1)
        return nums
        

class Solution:
    def merge_sort(self, nums):        
        def helper_sort(left, right):
            if left > right:
                return []
            if left == right:
                return [nums[left]]
            
            mid = (left+right)//2
            l = helper_sort(left, mid)
            r = helper_sort(mid+1,right)
            return helper_merge(l, r)
        
        def helper_merge(left_arr, right_arr):
            l_idx = 0
            r_idx = 0
            ret = []
            
            while l_idx < len(left_arr) and r_idx < len(right_arr):
                if left_arr[l_idx] < right_arr[r_idx]:
                    ret.append(left_arr[l_idx])
                    l_idx += 1
                else:
                    ret.append(right_arr[r_idx])
                    r_idx += 1
            ret.extend(left_arr[l_idx:])
            ret.extend(right_arr[r_idx:])
            return ret
        return helper_sort(0, len(nums)-1)
    
    def merge_sort_iter(self, nums):
        if len(nums) <= 1:
            return nums
        q = deque()
        for n in nums:
            q.append([n])
        
        while len(q) > 1:
            size = len(q)
            idx = 0
            while idx < size:
                l_arr = q.popleft()
                idx += 1
                if idx == size:
                    q.append(l_arr)
                    break
                r_arr = q.popleft()
                idx += 1
                
                l_idx = 0
                r_idx = 0
                tmp = []
                while l_idx < len(l_arr) and r_idx < len(r_arr):
                    if l_arr[l_idx] < r_arr[r_idx]:
                        tmp.append(l_arr[l_idx])
                        l_idx += 1
                    else:
                        tmp.append(r_arr[r_idx])
                        r_idx += 1
                tmp.extend(l_arr[l_idx:])
                tmp.extend(r_arr[r_idx:])
                q.append(tmp)
        return q.popleft()
    
    def sortArray(self, nums: List[int]) -> List[int]:
        # return self.merge_sort(nums)
        return self.merge_sort_iter(nums)

class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        if len(nums)<=1:
            return nums
        
        sorted_list = nums
        
        while len(sorted_list)>1:
            
            x = sorted_list.pop(0)
            y = sorted_list.pop(0)
            sorted_list.append(self.merge(x,y))
            
            
            
        
        return sorted_list[0]
            
            
        
        
        
    def merge(self,list1, list2):
        
        if isinstance(list1, int):
            list1 = [list1]
        if isinstance(list2, int):
            list2 = [list2]
        ret = []
            
        list1_cursor = list2_cursor = 0
        
        while list1_cursor < len(list1) and list2_cursor < len(list2):
            if list1[list1_cursor] < list2[list2_cursor]:
                ret.append(list1[list1_cursor])
                list1_cursor += 1
                
            else:
                ret.append(list2[list2_cursor])
                list2_cursor += 1
                    
            
        ret.extend(list1[list1_cursor:])
        ret.extend(list2[list2_cursor:])
        
        return ret
class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        #Heap Sort, building max heap
        l=len(nums)
        def heapify(i: int, l: int) -> None:
            if 2*i+2==l:
                if nums[i]<nums[2*i+1]:
                    nums[i], nums[2*i+1]=nums[2*i+1], nums[i]
            elif 2*i+2<l:
                if nums[2*i+1] > nums[2*i+2]:
                    if nums[i]<nums[2*i+1]:
                        nums[i], nums[2*i+1]=nums[2*i+1], nums[i]
                    heapify(2*i+1, l)
                else:
                    if nums[i]<nums[2*i+2]:
                        nums[i], nums[2*i+2]=nums[2*i+2], nums[i]
                    heapify(2*i+2, l)
                        
        #building the heap
        for i in range((l-1)//2, -1, -1):
            heapify(i, l)
        
        #swap the top with the last item and decrease the heap by 1, rebuilding the heap
        for i in range(l-1, -1, -1):
            nums[0], nums[i] = nums[i], nums[0]
            heapify(0, i)
            
        return nums
        
        """
        #Quick sort
        def quickSort(start:int, end:int) -> None:
            if start>=end:
                return
            if (end==start+1):
                if nums[end]<nums[start]:
                    nums[end], nums[start] = nums[start], nums[end]
                return
            pivit=nums[end]
            left, right=start, end-1
            while left<=right:
                while left<=end-1 and nums[left]<=pivit:
                    left+=1
                while right>=start and nums[right]>=pivit:
                    right-=1
                if left<right:
                    nums[left], nums[right] = nums[right], nums[left]
                    left+=1
                    right-=1
            nums[left], nums[end] = nums[end], nums[left]
            quickSort(start, left-1)
            quickSort(left+1, end)
            
        quickSort(0, len(nums)-1)
        return nums
        """
        
        """
        #Merge Sort
        def merge(l1:List[int], l2:List[int]) -> List[int]:
            ret =[]
            i1=i2=0
            while i1<len(l1) and i2<len(l2):
                if l1[i1]<l2[i2]:
                    ret.append(l1[i1])
                    i1+=1
                else:
                    ret.append(l2[i2])
                    i2+=1
            if i1==len(l1):
                ret+=l2[i2:]
            else:
                ret+=l1[i1:]
            return ret
        
        def mergeSort(start:int, end:int) -> List[int]:
            if start>=end:
                return [nums[start]]
            if start+1==end:
                if nums[start]>nums[end]:
                    nums[start], nums[end]=nums[end], nums[start]
                return [nums[start], nums[end]]
            mid=(start+end)//2
            return merge(mergeSort(start, mid), mergeSort(mid+1, end))
            
        return mergeSort(0, len(nums)-1)
        """
        
        
class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        def merge(left, right):
            result = []
            while len(left) != 0 and len(right) != 0:
                l = left[0]
                r = right[0]
                if r < l:
                    result.append(right.pop(0))
                else:
                    result.append(left.pop(0))
            return result + left + right
        def mergeSort(arr):
            if len(arr) < 2:
                return arr[:]
            else:
                mid = len(arr) // 2
                left = mergeSort(arr[:mid])
                right = mergeSort(arr[mid:])
                return merge(left, right)
        
        return mergeSort(nums)
class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        return sorted(nums)
class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        
        if len(nums) == 1:
            return nums
        
        n = int(len(nums) / 2)
        a1 = nums[:(n)]
        a2 = nums[(n):]
        
        a1 = self.sortArray(a1)
        a2 = self.sortArray(a2)
        
        return self.merge(a1,a2)
    
    def merge(self,a1,a2):
        i1 = 0
        i2 = 0
        ret = []
        
        while i1 < len(a1) and i2 < len(a2):
            if a1[i1] < a2[i2]:
                ret.append(a1[i1])
                i1 += 1
            else:
                ret.append(a2[i2])
                i2 += 1
        
        while i1 < len(a1):
            ret.append(a1[i1])
            print((a1[i1]))
            i1 += 1
        while i2 < len(a2):
            ret.append(a2[i2])
            print((a2[i2]))
            i2 += 1
        return ret
                
                

class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        def merge_sort(left, right):
            res = []
            left_p = right_p = 0
            while left_p < len(left) and right_p < len(right):
                if left[left_p] < right[right_p]:
                    res.append(left[left_p])
                    left_p += 1
                else:
                    res.append(right[right_p])
                    right_p += 1
            res.extend(left[left_p:])
            res.extend(right[right_p:])
            return res

        
        if len(nums) <= 1:
            return nums
        middle = len(nums) // 2
        left = self.sortArray(nums[:middle])
        right = self.sortArray(nums[middle:])
        if len(left) > len(right):
            return merge_sort(right, left)
        else:
            return merge_sort(left, right)
        


class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        if len(nums)>1:
            mid = len(nums)//2
            l = nums[:mid]
            r = nums[mid:]
            self.sortArray(l)
            self.sortArray(r)
            i = 0
            j = 0
            k = 0
            while i<len(l) and j<len(r):
                if l[i]<r[j]:
                    nums[k] = l[i]
                    i+=1
                else:
                    nums[k] = r[j]
                    j+=1
                k+=1
            while i<len(l):
                nums[k] = l[i]
                i+=1
                k+=1
            while j<len(r):
                nums[k] = r[j]
                j+=1
                k+=1
        return nums
class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        # Take advantage of this problem to summarize all the sorting algorithms
        if not nums or len(nums) < 2:
            return nums
        
        return self.heapSort(nums)
    
    # Slow sorting algorithms
    
    def selection_sort(self, nums):
        # selection sort, every time we reduce the problem by moving the low boundary backwards by 1
        # 0: low is sorted and low: is not sorted
        # in-place without creating temp arrays O(n ^ 2)
        
        def select_smallest(nums, low):
            smallest_index = low
            for index in range(low + 1, len(nums)):
                if nums[index] < nums[smallest_index]:
                    smallest_index = index
            return smallest_index
            
        def swap(nums, i, j):
            if i != j:
                nums[i], nums[j] = nums[j], nums[i]
                
        for index in range(len(nums) - 1):
            target_index = select_smallest(nums, index)
            swap(nums, index, target_index)
             
        return nums
    
    def insert_sort(self, nums):
        # the sorted portion is [0: index], look at nums[index] and decide what is the best position to insert within [0: index]
        # (tricky) the shifting and finding the position to insert is done in the same loop
        for index in range(1, len(nums)):
            k = index
            while k > 0 and nums[k - 1]  > nums[k]:
                nums[k-1], nums[k] = nums[k], nums[k-1]
                k -= 1
        return nums
    
    def bubble_sort(self, nums):
        # each time we push the largest to the end, and then reduce the scope of bubbling
        def bubble(nums, end):
            for index in range(end):
                if nums[index] > nums[index + 1]:
                    nums[index], nums[index + 1] = nums[index + 1], nums[index]
        for end in range(len(nums) - 1, -1, -1):
            bubble(nums, end)
        return nums
    
    # Quicker sorting algorithms
    
    def quickSort(self, nums):
        # quick sort selects a pivot and partition the current list into two parts < pivot, greater than pivot
        # then recursion into each sub array for further sorting
        # 1. how to do the split
        # 2. how to do the recursion
        # 3. base cases
        def split(nums, low, high):
            # should return an index which partitions the data
            if low < 0 or high >= len(nums) or low >= high:
                return 
            pivot = nums[low]
            while low < high:
                # find the first element less than pivot
                while low < high and nums[high] > pivot:
                    high -= 1
                nums[low] = nums[high]
                while low < high and nums[low] <= pivot:
                    low += 1
                nums[high] = nums[low]
            nums[low] = pivot
            return low 
        
        def recur(nums, low, high):
            if low >= high:
                return
            pivot_index = split(nums, low, high)
            recur(nums, low, pivot_index - 1)
            recur(nums, pivot_index + 1, high)
        recur(nums, 0, len(nums) - 1)
        return nums
    
    def mergeSort(self, nums):
        # spliting is easy, no need to have. a special function
        # merging is non-trivial, need to have a separate function. In-space is hard
    
        def merge(sub0, sub1):
            res = []
            m, n = len(sub0), len(sub1)
            p0 = p1 = 0
            while p0 < m and p1 < n:
                if sub0[p0] < sub1[p1]:
                    res.append(sub0[p0])
                    p0 += 1
                else:
                    res.append(sub1[p1])
                    p1 += 1
            while p0 < m:
                res.append(sub0[p0])
                p0 += 1
            while p1 < n:
                res.append(sub1[p1])
                p1 += 1
            return res
        
        if not nums or len(nums) < 2:
            return nums
        mid = len(nums) // 2
        left, right = self.mergeSort(nums[:mid]), self.mergeSort(nums[mid:])
        return merge(left, right)            
    
    def heapSort(self, nums):
        
        class Heap:
            def __init__(self, values=[]):
                self.values = values
                self.size = len(values)
                self.heapify()
                
            def _sift_down(self, i, n):
                if i >= n:
                    return
                candidate = i
                if i * 2 + 1 < n and self.values[2 * i + 1] > self.values[candidate]:
                    candidate = i * 2 + 1
                if i * 2 + 2 < n and self.values[2 * i + 2] > self.values[candidate]:
                    candidate = i * 2 + 2
                if candidate != i:
                    self.values[i], self.values[candidate] = self.values[candidate], self.values[i]
                    self._sift_down(candidate, n)
                    
            def heapify(self):
                if not self.values:
                    return
                n = self.size
                for index in range(n // 2, -1, -1):
                    self._sift_down(index, n)
                    
            def extract_max(self):
                ret = self.values[0]
                self.size -= 1
                self.values[0] = self.values[self.size]
                self._sift_down(0, self.size)
                return ret
            
        heap = Heap(nums)
        n = len(nums)
        result = []
        while heap.size:
            result.append(heap.extract_max())
        return result[::-1]
            
            
            
            
            
            
            
            
                
            
            
            
            
            
            
            

class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        tmp = [0 for _ in range(len(nums))]
        self.ms(nums, 0, len(nums)-1, tmp)
        return nums
    
    def ms(self, nums, start, end, tmp):
        if start >= end:
            return
        
        mid = (start + end) // 2
        self.ms(nums, start, mid, tmp)
        self.ms(nums, mid+1, end, tmp)
        self.merge(nums,start, mid, end, tmp)
    
    def merge(self, nums, start, mid, end, tmp):
        left, right = start, mid + 1
        idx = start
        
        while left <= mid and right <= end:
            if nums[left] < nums[right]:
                tmp[idx] = nums[left]
                left += 1
            else:
                tmp[idx] = nums[right]
                right += 1
            idx += 1
        
        while left <= mid:
            tmp[idx] = nums[left]
            left += 1
            idx += 1
        while right <= end:
            tmp[idx] = nums[right]
            right += 1
            idx += 1
        
        for i in range(start, end+1):
            nums[i] = tmp[i]
            
        return
class Solution:  
    def left(self, i):
        return 2*i+1
    def right(self, i):
        return 2*i+2
    def parent(self, i):
        return (i//2)-1 if not i%2 else i//2
    
    def maxheapify(self, a, heapsize,i):
        l = self.left(i)
        leftisgreater = False
        rightisgreater = False
        if l < heapsize:
            if a[i] < a[l]:
                leftisgreater = True
        r = self.right(i)
        if r < heapsize:
            if a[i] < a[r]:
                rightisgreater = True
        
        if leftisgreater or rightisgreater:
            if leftisgreater and not rightisgreater:
                a[i],a[l] = a[l],a[i]
                self.maxheapify(a, heapsize, l)
            elif not leftisgreater and rightisgreater:
                a[i],a[r] = a[r],a[i]
                self.maxheapify(a, heapsize, r)
            elif leftisgreater and rightisgreater:
                if a[l] <= a[r]:
                    rightisgreater = True
                    leftisgreater = False
                else:
                    leftisgreater = True
                    rightisgreater = False
                if rightisgreater:
                    a[i],a[r] = a[r],a[i]
                    self.maxheapify(a, heapsize, r)
                else:
                    a[i],a[l] = a[l],a[i]
                    self.maxheapify(a, heapsize, l)
                    

    def buildmaxheap(self, nums, heapsize):
        for i in reversed(range(len(nums)//2)):
            self.maxheapify(nums, heapsize,i)
           
    
    def heapsort(self, nums):
        heapsize = len(nums)
        self.buildmaxheap(nums, heapsize)
        for i in range(len(nums)):
            nums[0],nums[heapsize-1]=nums[heapsize-1],nums[0]
            heapsize-=1
            self.maxheapify(nums, heapsize, 0)
        
    def sortArray(self, nums: List[int]) -> List[int]:
        self.heapsort(nums)
        return nums
class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        self.mergeSort(nums)
        return nums
    
    def mergeSort(self, nums: List[int]) -> None:
        if len(nums) > 1:
            mid = len(nums) // 2
            L, R = nums[:mid], nums[mid:]

            self.mergeSort(L)
            self.mergeSort(R)
            
            
            i = j = k = 0
            while i < len(L) and j < len(R):
                if L[i] < R[j]:
                    nums[k] = L[i]
                    i += 1
                else:
                    nums[k] = R[j]
                    j += 1
                k += 1
            
            while i < len(L):
                nums[k] = L[i]
                i += 1
                k += 1

            while j < len(R):
                nums[k] = R[j]
                j += 1
                k += 1

class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        
        if nums is None or len(nums) < 2:
            return nums
        
        return self.countSort(nums, len(nums))
    
    def countSort(self, nums, n):
        memo = collections.defaultdict(lambda: 0)
        
        for num in nums:
            memo[num] += 1
            
        answer = [None] * n
        currentIndex = 0
        
        for currentNum in range(-50000, 50001):
            
            if currentNum in memo:
                
                while memo[currentNum] > 0:
                    answer[currentIndex] = currentNum
                    currentIndex += 1
                    memo[currentNum] -= 1
                    
        return answer
                    

class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        mi = abs(min(nums))
        nums = [i+mi for i in nums]
        l = len(str(max(nums)))
        res = []
        for i in nums:
            if len(str(i)) == l:
                res.append(str(i))
                continue
            d = l-len(str(i))
            a = '0'*d + str(i)
            res.append(a)
            
        for i in range(l-1,-1,-1):
            res = self.f(res,i)
            
        return [int(i)-mi for i in res]
            
    def f(self,res,i):
        count = {str(x):[] for x in range(10)}
        for j in res:
            count[j[i]].append(j)
        arr = []
        for j in '0123456789':
            if len(count[j]) == 0:
                continue
            for x in count[j]:
                arr.append(x)
        return arr

class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        return sorted(nums)

class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        buckets = [0] * 100002
        res = []
        for i in nums:
            buckets[i + 50000] += 1
        for i in range(len(buckets)):
            if buckets[i] != 0:
                for _ in range(buckets[i]):
                    res.append(i - 50000)
        return res
class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        count = [0] * 100010
        for n in nums:
            count[n + 50000] += 1
        
        res = []
        for c in range(100010):
            while count[c] > 0:
                res.append(c - 50000)
                count[c] -= 1
        
        return res

class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        
        self.countingSort(nums)
        return nums
        
                  
    def quickSort(self, arr):
        import random
        def partition(arr, l, h):
                if l>=h:
                    return
                #pivot = arr[random.randint(l,h)]
                pivot = arr[h]
                tempL, tempH = l, h
                while tempL<=tempH:
                    while tempL<=tempH and arr[tempL]<pivot:
                        tempL+=1
                    while tempL<=tempH and arr[tempH]>pivot:
                        tempH-=1
                    if tempL<=tempH:
                        arr[tempL], arr[tempH] = arr[tempH], arr[tempL]
                        tempL+=1
                        tempH-=1
                partition(arr, l, tempH)
                partition(arr, tempL, h)
                return
        partition(arr, 0, len(arr)-1)
        return arr
    
    def mergeSort(self, arr):
        if len(arr)<=1: return
        mid = len(arr)//2
        L = arr[:mid]
        R = arr[mid:]
        self.mergeSort(L)
        self.mergeSort(R)
        idxL = 0
        idxR = 0
        idxA = 0
        while idxA<len(arr):
            if idxL < len(L) and idxR < len(R):
                if L[idxL]<R[idxR]:
                    arr[idxA] = L[idxL]
                    idxL += 1
                else:
                    arr[idxA] = R[idxR]
                    idxR += 1
            elif idxL < len(L):
                arr[idxA] = L[idxL]
                idxL += 1
            else:
                arr[idxA] = R[idxR]
                idxR += 1
            idxA+=1
        return arr
    
    def heapSort(self, arr):
        def heapify(arr, arrLen, i):
            # i = index of this node
            l = 2 * i + 1
            r = l + 1
            largestIdx = i
            
            # theese two if is to finding the biggest node to make a max heap 
            if l < arrLen and arr[largestIdx] < arr[l]:
                largestIdx = l
            if r < arrLen and arr[largestIdx] < arr[r]:
                largestIdx = r
            
            # make the largest the parent
            if largestIdx!=i:
                arr[i], arr[largestIdx] = arr[largestIdx], arr[i]
                heapify(arr, arrLen, largestIdx)
        
        arrLen = len(arr)
        # heap all the tree from the middle of the array to the front
        # make the end of the array biggest
        for i in range(arrLen//2, -1, -1):
            heapify(arr, arrLen, i)
        
        # the i is the root of the tree since i, and its a max heap
        for i in range(arrLen-1, -1, -1):
            arr[i], arr[0] = arr[0], arr[i]
            heapify (arr, i, 0)
            
    def countingSort(self, arr):
        maxA = arr[0]
        minA = arr[0]
        
        for a in arr:                           # O(len(arr))
            maxA = max(maxA, a)
            minA = min(minA, a)
            
        countLen = maxA-minA+1
        offset = - minA
        
        count = [0 for i in range(countLen)]    # O(max(arr)-min(arr))
        
        for a in arr:                           # O(len(arr))
            count[a + offset] += 1
        
        for i in range(1,countLen):             # O(max(arr))
            count[i] += count[i-1]
        
        res = list(arr)                  
        
        for r in res:
            arr[count[r + offset]-1] = r
            count[r + offset] -= 1
            
        return
import math
class Solution:
    
    def sortArray(self, nums: List[int]) -> List[int]:
        if len(nums) < 64:
            return sorted(nums)
        if len(nums) <= 1:
            return nums
        c = 0
        maxN = nums[-1]
        al,ar=[],[]
        
        for num in nums:
            if num < maxN:
                al.append(num)
            elif num > maxN:
                ar.append(num)
            elif num == maxN:
                c += 1
            
        return self.sortArray(al) + [maxN] * c + self.sortArray(ar)
        
        
    
    

            
        
        

class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
	
        if len(nums) <=1: return nums
        less , greater , base = [] , [] , nums.pop()
        for i in nums:
            if i < base: less.append(i)
            else: greater.append(i)
        return self.sortArray(less) + [base] + self.sortArray(greater)
class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
	
        if len(nums) <=1: 
            return nums
        less , greater , base = [] , [] , nums.pop()
        for i in nums:
            if i < base: 
                less.append(i)
            else: 
                greater.append(i)
        return self.sortArray(less) + [base] + self.sortArray(greater)
class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        if len(nums) <= 1:
            return nums
        else:
            pivot = nums[int(len(nums)/2)]
            L = []
            M = []
            R = []
            for n in nums:
                if n < pivot:
                    L.append(n)
                elif n > pivot:
                    R.append(n)
                else:
                    M.append(n)
            return self.sortArray(L) + M + self.sortArray(R)
class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        return self.quickSort(nums)
        
    def quickSort(self, nums):
        if not nums or len(nums) < 2:
            return nums
        pivot = nums[0]
        left = [] 
        right = []
        for x in nums[1:]:
            if x <= pivot:
                left.append(x)
            else:
                right.append(x)
        return self.quickSort(left) + [pivot] + self.quickSort(right)

class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        return self.quickSort(nums)
    
    def quickSort(self, nums):
        if not nums or len(nums) < 2:
            return nums
        
        pivot = nums[0]
        left = []
        right = []
        for n in nums[1:]:
            if n <= pivot:
                left.append(n)
            else:
                right.append(n)
        return self.quickSort(left) + [pivot] + self.quickSort(right)
                
        
        

class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        quickSort(nums, 0, len(nums)-1)
        return nums
    
    
def partition(arr, low, high):
    # i = (low-1)         # index of smaller element
    i = low
    pivot = arr[high]     # pivot
    for j in range(low, high):
        if arr[j] < pivot:
            arr[i], arr[j] = arr[j], arr[i]
            i += 1
    # arr[i+1], arr[high] = arr[high], arr[i+1]
    arr[i], arr[high] = arr[high], arr[i]
    # return (i+1)
    return i

def quickSort(arr, low, high):
    if low >= high:
        return
    pi = partition(arr, low, high)
    quickSort(arr, low, pi-1)
    quickSort(arr, pi+1, high)

class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        quickSort(nums, 0, len(nums)-1)
        return nums
    
    
def partition(arr, low, high):
    i = (low-1)         # index of smaller element
    pivot = arr[high]     # pivot
    for j in range(low, high):
        if arr[j] < pivot:
            i = i+1
            arr[i], arr[j] = arr[j], arr[i]
    arr[i+1], arr[high] = arr[high], arr[i+1]
    return (i+1)

def quickSort(arr, low, high):
    if low >= high:
        return
    pi = partition(arr, low, high)
    quickSort(arr, low, pi-1)
    quickSort(arr, pi+1, high)

class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        n = len(nums)
        self.quickSort(nums, 0, n-1)
        return nums
    def quickSort(self, nums, low, high):
        if low < high: 
            pivot = self.partition(nums, low, high)
            self.quickSort(nums, low, pivot - 1)
            self.quickSort(nums, (pivot + 1), high)

    def partition (self, nums: List[int], low, high) -> int:
        i = (low - 1)
        pivot = nums[high]
        for j in range(low, high):
            if nums[j] <= pivot:
                i += 1
                nums[i], nums[j] = nums[j], nums[i]

        nums[i + 1], nums[high] = nums[high], nums[i + 1]
        return (i + 1)
class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        quickSort(nums, 0, len(nums)-1)
        return nums
    
    
def partition(arr, low, high):
    i = low - 1  # last small element
    pivot = arr[high]
    for j in range(low, high):
        if arr[j] < pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    arr[i+1], arr[high] = arr[high], arr[i+1]  # i+1 is pivot
    return i+1

def quickSort(arr, low, high):
    if low >= high:
        return
    pi = partition(arr, low, high)
    quickSort(arr, low, pi-1)
    quickSort(arr, pi+1, high)

class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        def quicksort(nums):
            if len(nums) <= 1:
                return nums
            else:
                q = nums[len(nums) // 2]
            l_nums = [n for n in nums if n < q]

            e_nums = [q] * nums.count(q)
            b_nums = [n for n in nums if n > q]
            return quicksort(l_nums) + e_nums + quicksort(b_nums)
        
        return quicksort(nums)

class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        self.quick_sort(nums, 0, len(nums) - 1)
        
        return nums
    
    def quick_sort(self, nums, start, end):
        if start >= end:
            return
        
        left, right = start, end
        pivot = nums[(start + end) // 2]
        
        while left <= right:
            while left <= right and nums[left] < pivot:
                left += 1
            
            while left <= right and nums[right] > pivot:
                right -= 1
            
            if left <= right:
                nums[left], nums[right] = nums[right], nums[left]
                left += 1
                right -= 1
            
        
        self.quick_sort(nums, start, right)
        self.quick_sort(nums, left, end)
                    
        

class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        return self.partition(nums)
    def partition(self,num_list):
        if not num_list or len(num_list) ==1:
            return num_list
        pivot = num_list[0]
        left = []
        right = []
        for i in range(1,len(num_list)):
            if num_list[i] >= pivot:
                right.append(num_list[i])
            else:
                left.append(num_list[i])
        return self.partition(left)+[pivot]+self.partition(right)
    
                

class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        def helper(start, end):
            if start >= end:
                return
            l = start
            r = end
            mid = l + (r - l) // 2
            pivot = nums[mid]
            while r >= l:
                while r >= l and nums[l] < pivot:
                    l += 1
                while r >= l and nums[r] > pivot:
                    r -= 1
                if r >= l:
                    nums[l], nums[r] = nums[r], nums[l]
                    l += 1
                    r -= 1
            helper(start, r)
            helper(l, end)
        
        helper(0, len(nums) - 1)
        return nums
class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        # BASE CASE 
        if not nums or len(nums) == 1: 
            return nums 
        
        pivot = nums[0]
        # pointer from 1 on 
        # check if the pointer is <= pivot 
        # if it is, then swap with the idx val, increment
        # if it is not, then skip, increment
        # that pointer is the spot where the pivot should be
        pivot_i = 1 #pivot_i, where pivot goes after sorting
        for i in range(1, len(nums)):
            if nums[i] <= pivot:
                nums[pivot_i], nums[i] = nums[i], nums[pivot_i]
                pivot_i += 1
        nums[pivot_i-1], nums[0] = nums[0], nums[pivot_i-1]
        # at pivot_i-1 is your pivot
        return self.sortArray(nums[:pivot_i-1]) + [pivot] + self.sortArray(nums[pivot_i:])
class Solution:
    def partition(self,arr,pi):
        less,equal, high = [],[],[]
        for i in arr:
            if i<pi:
                less.append(i)
            if i == pi:
                equal.append(i)
            if i>pi:
                high.append(i)
        return (less,equal,high)
                
    def sortArray(self, nums: List[int]) -> List[int]:
        if len(nums)<1:
            return nums
        pivot = nums[0]
        less,equal,high = self.partition(nums,pivot)
        return self.sortArray(less)+equal+self.sortArray(high)

class Solution:
    def sortArray(self, nums: List[int]):
        #use qsort, lc 215
        #partition() will place the pivot into the correct place, u5de6u9589u53f3u9589
        #outer function, will recursively process the numbers nums[:correct_idx] and numbers nums[correct_idx+1:] 
        #qsort(nums[4,3,6,6,8,6], 4 will be placed to idx1 => [3,4,6,6,8,6])
        # /                 
        #qsort(nums[3])    qsort([6,6,8,6], 6 will be placed to idx 2 in the original nums)
        
        #=>qsort() is the recursion, with input arg as start idx, end idx
        
        self.qsort(nums, 0, len(nums)-1)
        return nums
    
    def qsort(self, nums, start_idx, end_idx): #start_idx and end_idx u5de6u9589u53f3u9589
        
        #terminate:
        #if end_idx >= start_idx: => error
        if end_idx <= start_idx:
            return
        
        #start_idx = 0
        #end_idx = len(nums)-1
        correct_idx = self.partition(nums, start_idx, end_idx) #the partition() will put nums[start] into the correct place
        self.qsort(nums, start_idx, correct_idx-1)
        self.qsort(nums, correct_idx+1, end_idx)
        
    def partition(self, nums, start, end):
        pivot = nums[start]
        left = start + 1
        right = end
        
        #goal: put values <= pivot before pointer left
        #                 >=       after pointer right
        while left <= right:
            #if nums[left] > pivot: #now we need to find the nums[right] < pivot so that we can swap them
            
            if nums[left] > pivot and nums[right] < pivot:
                nums[left], nums[right] = nums[right], nums[left]
                left += 1
                right -= 1
            if nums[left] <= pivot:
                left += 1
            if nums[right] >= pivot:
                right -= 1
        
        #now right is the correct place  to place pivot
        nums[start], nums[right] = nums[right], nums[start]
        return right

class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        '''method 1: mergesort (divide and conqueer). Time complexity: T(n) = 2*T(n/2)+O(n) --> TC: O(nlogn). Space complexity: O(n+logn), since the recursion depth is logn, and we allocate size n temporary array to store intermediate results.'''
#         def helper(l,r):
#             if l+1>=r: return
#             m = l + (r-l)//2
#             helper(l, m)
#             helper(m, r)
#             it1 = l
#             it2 = m
#             ind = 0
#             while it1 < m or it2 < r:
#                 if it2 == r or (it1 < m and nums[it1] <= nums[it2]):
#                     temp[ind] = nums[it1]
#                     it1 += 1
#                 else:
#                     temp[ind] = nums[it2]
#                     it2 += 1
#                 ind += 1
#             nums[l:r] = temp[:ind]
            
#         temp = [0]*len(nums)
#         helper(0, len(nums))
#         return nums
        '''method2: quick sort. not stable. O(nlogn)/O(1)'''
        def pivot(l,r):
            if l >= r:
                return
            p = nums[random.randint(l, r)]
            it1 = l
            it2 = r
            while it1 <= it2:
                while nums[it1] < p:
                    it1 += 1
                while nums[it2] > p:
                    it2 -= 1
                if it1 <= it2:
                    nums[it1], nums[it2] = nums[it2], nums[it1]
                    it1 += 1
                    it2 -= 1
            pivot(l, it2)
            pivot(it1, r)
        pivot(0, len(nums)-1)
        return nums
            

class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
 
        if len(nums)==0:
            return []

        pivot = nums[0]

        left = [x for x in nums[1:] if x<=pivot]
        right = [x for x in nums[1:] if x>pivot]

        return self.sortArray(left)+[pivot]+self.sortArray(right)

class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        
        def _selection_sort(a):
            for i in range(len(a)):
                m = i
                for j in range(i+1,len(a)):
                    if a[m] > a[j]:
                        m = j
                a[i],a[m] = a[m], a[i]
                
        def _bubble_sort(a):
            for i in range(len(a)):
                for j in range(len(a)-1-i):
                    if a[j] > a[j+1]:
                        a[j], a[j+1] = a[j+1], a[j]
                
        def _insertion_sort(a):
            for i in range(1,len(a)):
                v = a[i]
                j = i
                while j > 0 and a[j-1] > v:
                    a[j] = a[j-1]
                    j -= 1
                a[j] = v

        def _quick_sort(a,s,e):
            def _partition(a,s,e):
                f = l = s
                while l < e:
                    if a[l] <= a[e]:
                        a[f],a[l] = a[l],a[f]
                        f+=1
                    l+=1
                a[f],a[e] = a[e],a[f]
                return f
            
            if s < e:
                p = _partition(a,s,e)
                _quick_sort(a,s,e=p-1) #left
                _quick_sort(a,s=p+1,e=e) # right
         
        def _merge_sort(a):
        
            def _merge(a,b):
                result = []
                i = j = 0
                while i < len(a) and j < len(b):
                    if a[i] <= b[j]:
                        result.append(a[i])
                        i += 1
                    else:
                        result.append(b[j])
                        j += 1
                result.extend(a[i:])
                result.extend(b[j:])
                return result
                
            result = []
            if len(a) <= 1:
                result = a.copy()
                return result
            
            m = len(a)//2
            _left = a[:m]
            _right = a [m:]
            _new_left = _merge_sort(_left)
            _new_right = _merge_sort(_right)
            result = merge(_new_left,_new_right)
            return result
            
        
        #_selection_sort(nums) #exceeds time
        #_bubble_sort(nums)    #exceeds time
        #_insertion_sort(nums) #exceeds time
        _quick_sort(nums,s=0,e=len(nums)-1)
        return nums
        #return _merge_sort(nums) #slower then 95%
        
    
    

# 3:10
from random import random
class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        
        def partition(start, end):
            pivot = round(random() * (end - start)) + start
            nums[start], nums[pivot] = nums[pivot], nums[start]
            below = above = start + 1
            while above <= end:
                if nums[above] <= nums[start]:
                    nums[below], nums[above] = nums[above], nums[below]
                    below += 1
                above += 1
            nums[below - 1], nums[start] = nums[start], nums[below - 1]
            return below - 1
        
        start = 0
        end = len(nums) - 1
        
        def quick_sort(start, end):
            if start < end:
                # print(nums)
                pivot = partition(start, end)
                # print(nums)
                # print(pivot)
                # print('------')
                quick_sort(start, pivot - 1)
                quick_sort(pivot + 1, end)
        
        quick_sort(start, end)
        
        return nums

class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        res, l, N = [0] * len(nums), 1, len(nums)

        def _merge(ll, lh, rl, rh):
            if rl >= N:
                for i in range(ll, N):
                    res[i] = nums[i]
            else:
                p, q, i = ll, rl, ll
                while p < lh and q < rh:
                    if nums[p] <= nums[q]:
                        res[i], p = nums[p], p + 1
                    else:
                        res[i], q = nums[q], q + 1
                    i += 1
                b, e = (p, lh) if p < lh else (q, rh)
                for j in range(b, e):
                    res[i] = nums[j]
                    i += 1

        while l < N:
            b = 0
            while b < N:
                _merge(b, b + l, b + l, min(N, b + 2 * l))
                b += 2 * l
            l, nums, res = l * 2, res, nums
        return nums
class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        
        def _selection_sort(a):
            for i in range(len(a)):
                m = i
                for j in range(i+1,len(a)):
                    if a[m] > a[j]:
                        m = j
                a[i],a[m] = a[m], a[i]
                
        def _bubble_sort(a):
            for i in range(len(a)):
                for j in range(len(a)-1-i):
                    if a[j] > a[j+1]:
                        a[j], a[j+1] = a[j+1], a[j]
                
        def _insertion_sort(a):
            for i in range(1,len(a)):
                v = a[i]
                j = i
                while j > 0 and a[j-1] > v:
                    a[j] = a[j-1]
                    j -= 1
                a[j] = v

        def _quick_sort(a,s,e):
            def _partition(a,s,e):
                f = l = s
                while l < e:
                    if a[l] <= a[e]:
                        a[f],a[l] = a[l],a[f]
                        f+=1
                    l+=1
                a[f],a[e] = a[e],a[f]
                return f
            
            if s < e:
                p = _partition(a,s,e)
                _quick_sort(a,s,e=p-1) #left
                _quick_sort(a,s=p+1,e=e) # right
         
        def _merge_sort(a):
        
            def _merge(a,b):
                result = []
                i = j = 0
                while i < len(a) and j < len(b):
                    if a[i] <= b[j]:
                        result.append(a[i])
                        i += 1
                    else:
                        result.append(b[j])
                        j += 1
                result.extend(a[i:])
                result.extend(b[j:])
                return result
                
            result = []
            if len(a) <= 1:
                result = a.copy()
                return result
            
            m = len(a)//2
            _left = a[:m]
            _right = a [m:]
            _new_left = _merge_sort(_left)
            _new_right = _merge_sort(_right)
            result = merge(_new_left,_new_right)
            return result
            
        
        #_selection_sort(nums)
        #_bubble_sort(nums)
        #_insertion_sort(nums)
        _quick_sort(nums,s=0,e=len(nums)-1)
        #return _merge_sort(nums)
        return nums
    
    

class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        # MergeSort        
        def merge(nums, lo, hi):
            mid = (lo + hi) // 2
            # [lo, mid] and [mid+1, hi] are already sorted
            i, j = lo, mid + 1
            sortedNums = []
            while i <= mid and j <= hi:
                if nums[i] < nums[j]:
                    sortedNums.append(nums[i])
                    i += 1
                else:
                    sortedNums.append(nums[j])
                    j += 1
            
            while i <= mid:
                sortedNums.append(nums[i])
                i += 1
            
            while j <= hi:
                sortedNums.append(nums[j])
                j += 1
            nums[lo:hi+1] = sortedNums
            
        def mergeSort(nums, lo, hi):
            if hi - lo == 0:
                return
            
            mid = (lo + hi) // 2
            
            mergeSort(nums, lo, mid)
            mergeSort(nums, mid+1, hi)
            # Both left and right portions are sorted
            merge(nums, lo, hi)

        mergeSort(nums, 0, len(nums)-1)
        return nums
import random

class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        def quick(nums):
            if len(nums) == 0:
                return []
            pivot = random.choice(nums)

            less = [x for x in nums if x < pivot]
            eq = [x for x in nums if x == pivot]
            more = [x for x in nums if x > pivot]

            return quick(less) + eq + quick(more)
        return quick(nums)

class Solution:
    def sortArray(self, nums):
        def mergesort(nums):
            LA = len(nums)
            if LA == 1: return nums
            LH, RH = mergesort(nums[:LA//2]), mergesort(nums[LA//2:])
            return merge(LH,RH)

        def merge(LH, RH):
            LLH, LRH = len(LH), len(RH)
            S, i, j = [], 0, 0
            while i < LLH and j < LRH:
                if LH[i] <= RH[j]: i, _ = i + 1, S.append(LH[i])
                else: j, _ = j + 1, S.append(RH[j])
            return S + (RH[j:] if i == LLH else LH[i:])
        
        return mergesort(nums)		
import random
class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        if not nums:
            return []
        self.quickSort(nums, 0, len(nums) - 1)
        return nums
        
        
        
    def quickSort(self, arr, l, r):
        if l >= r:
            return 
        
        position = self.partition(arr, l, r)
        self.quickSort(arr, l, position)
        self.quickSort(arr, position + 1, r)
        
        
        
    def partition(self, arr, l, r):
        #print(arr, l, r)
        pivot = arr[random.randint(l, r)]
        arr[l], pivot = pivot, arr[l]
        #print("pivot", pivot)
        while l < r:     
            while l < r and arr[r] >= pivot:
                 r -= 1
            arr[l] = arr[r] 
            while l < r and arr[l] <= pivot:
                l += 1
            arr[r] = arr[l]
            
            arr[l] = pivot
        return l

        
        

"""

                e
            5 2 1 3 
                  s 

pivot = nums[s] = 5

We need to have nums[s] < pviot < nums[e]

if nums[s] > pivot and nums[e] < pivot: out of place so swap(nums, s, e)

if nums[s] <= pivot : right place s += 1

if nums[l] >= pivot : right place l -= 1

"""


class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        
        self.quickSort(nums, 0, len(nums) - 1)
        return nums
        
    
    def quickSort(self, nums, s, e):
        if s <= e:
            p = self.partition(nums, s, e)
            self.quickSort(nums, s, p - 1)
            self.quickSort(nums, p + 1, e)
            
            
    def partition(self, nums, s, e):
        pivot = nums[s]
        l = s
        s += 1
        
        def swap(nums, i, j):
            nums[i], nums[j] = nums[j], nums[i]
            
        while s <= e:
            if nums[s] > pivot and nums[e] < pivot:
                # out of place 
                swap(nums, s, e)
            
            if nums[s] <= pivot:
                s += 1
            if nums[e] >= pivot:
                e -= 1
        swap(nums, l, e)
        return e
        
class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        if not nums:
            return []
        
        temp = list(range(len(nums)))
        
        self.helper(nums, 0, len(nums) - 1, temp)
        
        return nums
    
    
    def helper(self, nums, start, end, temp):
        if start >= end:
            return 
        
        mid = (start + end) // 2
        self.helper(nums, start, mid, temp)
        self.helper(nums, mid + 1, end, temp)
        
        self.merge(nums, start, end, temp)
        
    
    def merge(self, nums, start, end, temp):
        mid = (start + end) // 2
        left = start
        right = mid + 1
        temp_index = start
        
        while left <= mid and right <= end:
            if nums[left] < nums[right]:
                temp[temp_index] = nums[left]
                left += 1
                temp_index += 1
            else:
                temp[temp_index] = nums[right]
                right += 1
                temp_index += 1
        
        while left <= mid:
            temp[temp_index] = nums[left]
            left += 1
            temp_index += 1
        
        while right <= end:
            temp[temp_index] = nums[right]
            right += 1
            temp_index += 1
        
        nums[start:end + 1] = temp[start:end + 1]
        
        

class Solution:
    def sortArray(self, listToSort: List[int]):
        if(len(listToSort)) == 0 or len(listToSort) == 1:
            return listToSort
        elif(len(listToSort) == 2):
            if(listToSort[0] > listToSort[1]):
                listToSort[0], listToSort[1] = listToSort[1], listToSort[0]
        else:
            divider = len(listToSort)//2
            l = listToSort[:divider]
            r = listToSort[divider:]
            self.sortArray(l)
            self.sortArray(r)

            i = 0
            j = 0
            k = 0

            while i < len(l) and j < len(r):
                if r[j] < l[i]:
                    listToSort[k] = r[j]
                    j += 1
                    k += 1
                else:
                    listToSort[k] = l[i]
                    i += 1
                    k += 1

            if i < len(l):
                listToSort[k:] = l[i:]

            if j < len(r):
                listToSort[k:] = r[j:]
        return listToSort
class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        nums.sort()
        return nums

class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        # merge sort
        # divide and conquer
        def sort(nums):
            if len(nums) == 1:
                return nums
            mid = len(nums)//2
            left = sort(nums[:mid])
            right = sort(nums[mid:])
            i,j = 0,0
            res = []
            while i < mid and j < len(nums) - mid:
                if left[i] <= right[j]:
                    res.append(left[i])
                    i += 1
                else:
                    res.append(right[j])
                    j += 1
                    
            if i < mid:
                res += left[i:]
            if j < len(nums) - mid:
                res += right[j:]
                
            return res
        
        return sort(nums)
                
                
            
        
        
        
        
        

class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        # return self.mergeSort(nums)
        self.quickSort(nums, 0, len(nums)-1)
        return nums
    
    def mergeSort(self, nums):
        if len(nums) == 1:
            return nums
        mid = len(nums)//2
        left, right = self.mergeSort(nums[:mid]), self.mergeSort(nums[mid:])
        return merge(left, right)
    def merge(le, ri):
        i, j = 0, 0
        res = []
        while i < len(le) and j < len(ri):
            if le[i] < ri[j]:
                res.append(le[i])
                i += 1
            else:
                res.append(ri[j])
                j += 1
        res.append(le[i:] if j == len(ri)-1 else ri[j:])
        print(res)
        return res
    
    def quickSort(self, nums, start, end):
        random.shuffle(nums)
        def sort(nums, start, end):
            if end <= start:
                return
            i, j = start, end
            p = start
            curNum = nums[start]
            while p <= j:
                if nums[p] < curNum:
                    nums[i], nums[p] = nums[p], nums[i]
                    i += 1
                    p += 1
                elif nums[p] > curNum:
                    nums[p], nums[j] = nums[j], nums[p]
                    j -= 1
                else: #nums[p]==curNum
                    p += 1   
            sort(nums, start, i-1)
            sort(nums, j+1, end)
        sort(nums,start,end)
        # def parition(nums, i, j):
        #     mid = i + (j-1-i)//2
        #     nums[j-1], nums[mid] = nums[mid], nums[j-1]
        #     for idx in range(i, j):
        #         if nums[idx] < nums[j-1]:
        #             nums[i], nums[idx] = nums[idx], nums[i]
        #             i += 1
        #     nums[j-1], nums[i] = nums[i], nums[j-1]
        #     # while i <= j-2 and nums[i] == nums[i+1]:
        #     #     i += 1
        #     return i
                

        
#         def partition(A, I, J):
#             A[J-1], A[(I + J - 1)//2], i = A[(I + J - 1)//2], A[J-1], I
#             for j in range(I,J):
#                 if A[j] < A[J-1]: A[i], A[j], i = A[j], A[i], i + 1
#             A[J-1], A[i] = A[i], A[J-1]
#             return i
        
        

class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        if len(nums) <=1:
            return nums
        mid = len(nums)//2
        l_list = self.sortArray(nums[:mid])
        r_list = self.sortArray(nums[mid:])
        return self.merge(l_list,r_list)
    
    def merge(self,a,b):
        
        a_l = b_l = 0
        m = []
        while a_l < len(a) and b_l < len(b):
            if a[a_l] < b[b_l]:
                m.append(a[a_l])
                a_l += 1
            else:
                m.append(b[b_l])
                b_l += 1
            
        
            
        m.extend(a[a_l:])
        m.extend(b[b_l:])
        
        return m
            
        

import math
class Solution:
    def merge_fn(self, x: List[int], y: List[int]) -> List[int]:
        i, j = 0, 0
        res = []
        while i < len(x) and j < len(y):
            if x[i] < y[j]:
                res.append(x[i])
                i += 1
            else:
                res.append(y[j])
                j += 1
        res += x[i:] + y[j:]
        return res
        
    def helper_iter(self, nums: List[int]) -> List[int]:
        res = deque([[i] for i in nums])
        while res and len(res)>1:
            x = res.popleft()
            y = res.popleft() if len(res) else []
            res.append(self.merge_fn(x,y))
        return res[0]
                
        
    def sortArray(self, nums: List[int]) -> List[int]:
        if len(nums) == 1:
            return nums
        
        res = self.helper_iter(nums)
        return res

class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        print(nums)
        if nums == []: return []
        p = nums.pop()
        s = []
        b = []
        for i in nums:
            if i > p:
                b.append(i)
            else:
                s.append(i)
        return self.sortArray(s.copy()) + [p] + self.sortArray(b.copy())

class Solution:
    def mergeSort(self, nums):
        if len(nums) > 1:
            mid = len(nums) // 2
            left = nums[mid:]
            right = nums[:mid]
            
            self.mergeSort(left)
            self.mergeSort(right)
            i = j = k = 0
            
            while i < len(left) and j < len(right):
                if left[i] < right[j]:
                    nums[k] = left[i]
                    i += 1
                else:
                    nums[k] = right[j]
                    j += 1
                k += 1
            while i < len(left):
                nums[k] = left[i]
                i += 1
                k += 1
            while j < len(right):
                nums[k] = right[j]
                j += 1
                k += 1
                
                
    def sortArray(self, nums: List[int]) -> List[int]:
        self.mergeSort(nums)
        return nums

class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        return merge_sort(nums)

def merge_sort(nums):
    if len(nums) <= 1:
        return nums

    pivot = len(nums)//2      
    left = merge_sort(nums[:pivot])
    right = merge_sort(nums[pivot:]) 
    return merge(left,right)


def merge(l1: List[int], l2: List[int]) -> List[int]:
    res = []

    i, j = 0, 0
    while i < len(l1) and j < len(l2):
        if l1[i] <= l2[j]:
            res.append(l1[i])
            i += 1
        else:
            res.append(l2[j])
            j += 1
    while i < len(l1):
        res.append(l1[i])
        i += 1
    while j < len(l2):
        res.append(l2[j])    
        j += 1
    return res

class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        
        if len(nums)>1: 
            mid = (len(nums)-1) //2
            left= nums[0:mid+1]
            right = nums[mid+1:]
            left = self.sortArray(left)
            right = self.sortArray(right)
            
            l=r=curr =0
            
            while l < len(left) and r < len(right):
                if left[l] <= right[r]:
                    nums[curr] = left[l]
                    l +=1
                else:
                    nums[curr] = right[r]
                    r+=1
                curr+=1
                
            while l < len(left):
                nums[curr] = left[l]
                l +=1
                curr+=1
                
            while r < len(right):
                nums[curr] = right[r]
                r+=1
                curr+=1
                
        
        
        return nums

class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        def sort(ls1, ls2):
            i = j = 0
            sortedList = []
            while i < len(ls1) and j < len(ls2):
                if ls1[i] < ls2[j]:
                    sortedList.append(ls1[i])
                    i += 1
                else:
                    sortedList.append(ls2[j])
                    j += 1
            if i < len(ls1):
                sortedList += ls1[i:]
            else:
                sortedList += ls2[j:]
            return sortedList
        
        def divide(ls):
            if len(ls) <= 1:
                return ls
            middle = int(len(ls) / 2)
            ls1 = divide(ls[:middle])
            ls2 = divide(ls[middle:])
            return sort(ls1, ls2)
        
        return divide(nums)
        
#                 def merge(arr):
#             # base case 
#             if len(arr) <= 1: 
#                 return arr
            
#             pivot = int(len(arr)/2)
#             left = merge(arr[:pivot])
#             right = merge(arr[pivot:])
            
#             return sort(left, right)
            
            
            
            
        
#         def sort(left, right):
#             left_cur = right_cur = 0
#             sorted_arr = []
#             while (left_cur < len(left) and right_cur < len(right)):
#                 if left[left_cur] > right[right_cur]:
#                     sorted_arr.append(right[right_cur])
#                     right_cur += 1
#                 else:
#                     sorted_arr.append(left[left_cur])
#                     left_cur += 1
                    
#             sorted_arr += left[left_cur:] + right[right_cur:]
            
#             return sorted_arr
        
#         return merge(nums)

class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        
        def merge_sort(nums):
            
            if len(nums) <=1:
                return nums
            
            pivot = int(len(nums)/2)
            
            left = merge_sort(nums[:pivot])
            right = merge_sort(nums[pivot:])
            
            return merge(left, right)
        
        def merge(left, right):
            out = []
            
            left_cursor = 0
            right_cursor = 0
            
            while left_cursor < len(left) and right_cursor < len(right):
                if left[left_cursor] <= right[right_cursor]:
                    out.append(left[left_cursor])
                    left_cursor+=1
                else:
                    out.append(right[right_cursor])
                    right_cursor+=1
            
            out.extend(right[right_cursor:])
            out.extend(left[left_cursor:])
            return out
        
        
        
        
        
        return merge_sort(nums)
        
                    
            

from heapq import heappop, heappush

class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        h = []
        result = []
        for num in nums:
            heappush(h, num)
            
            
        while h:
            result.append(heappop(h))
            
        return result
class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        # merge sort
        if nums == [] or len(nums) == 1:
            return nums
        m = len(nums) // 2
        left = self.sortArray(nums[:m])
        right = self.sortArray(nums[m:])
        
        # merge
        i = j = k = 0
        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
                nums[k] = left[i]
                i += 1
            else:
                nums[k] = right[j]
                j += 1
            k+= 1
        
        while j < len(right):
            nums[k] = right[j]
            j += 1
            k += 1
        
        while i < len(left):
            nums[k] = left[i]
            i += 1
            k += 1
        
        return nums
        
        
        
        
        
        
        
        
        
        
        
        
        
        
# 		# merge sort. If length <= 1, return
# 		if not nums or len(nums) <= 1:
# 			return nums

# 		mid = len(nums) // 2
# 		lower = self.sortArray(nums[:mid]) # Lower after sorting
# 		higher = self.sortArray(nums[mid:len(nums)]) # Higher after sorting
# 		print(lower, higher)

# 		i = j = k = 0
# 		while i < len(lower) and j < len(higher):
# 			if lower[i] <= higher[j]:
# 				nums[k] = lower[i]
# 				i += 1
# 			else:
# 				nums[k] = higher[j]
# 				j += 1
# 			k += 1

# 		while i < len(lower):
# 			nums[k] = lower[i]
# 			i += 1
# 			k += 1
# 		while j < len(higher):
# 			nums[k] = higher[j]
# 			j += 1
# 			k += 1

# 		return nums

class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        
        
        def merge_sort(nums):
            
            N = len(nums)
            if N <2:
                return nums
            
            nums_left = merge_sort(nums[:N//2])
            nums_right =  merge_sort(nums[N//2:])
      
            i, j, k = 0, 0, 0
         
            while (i < len(nums_left) ) and  (j < len(nums_right)):
                if nums_left[i] <= nums_right[j]:
                    nums[k] = nums_left[i]
                    i +=1
                elif nums_left[i] > nums_right[j]:
                    nums[k] = nums_right[j]
                    j += 1
                k += 1
                
            while (i < len(nums_left) ):
                nums[k] = nums_left[i]
                i +=1
                k +=1
                
            while (j < len(nums_right) ):
                nums[k] = nums_right[j]
                j +=1
                k +=1
                   
            return nums
        
        return merge_sort(nums)
            
            

class Solution:
    def merge(self,num: List[int], nums: List[int], arr: List[int]) -> List[int]:
        i = j = k = 0
        while i < len(num) and j < len(nums):
            if num[i] < nums[j]:
                arr[k] = num[i]
                i += 1
            elif num[i] >= nums[j]:
                arr[k] = nums[j]
                j += 1
            k += 1
        
        while i < len(num): 
            arr[k] = num[i] 
            i += 1
            k += 1
          
        while j < len(nums): 
            arr[k] = nums[j] 
            j += 1
            k += 1
        
        return arr
                
    def sortArray(self, nums: List[int]) -> List[int]:
        if(len(nums) != 1):
            mid = len(nums)//2
            left = self.sortArray(nums[:mid])
            right = self.sortArray(nums[mid:])
            nums = self.merge(left, right, nums)
        return nums
class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        
        def merge(nums):
            
            if len(nums)<2:
                return nums
            
            ln = merge(nums[:len(nums)//2])
            rn = merge(nums[len(nums)//2:])
            
            i,j,r = 0, 0, []
            while i<len(ln) or j<len(rn):
                if j<len(rn) and (i==len(ln) or rn[j] < ln[i]):
                    r.append(rn[j])
                    j += 1
                else:
                    r.append(ln[i])
                    i += 1
            return r
        
        return merge(nums)

class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        def merge(left_array, right_array):
            if not left_array:
                return right_array
            
            if not right_array:
                return left_array
            
            left_low = 0
            right_low = 0
            output = []
            while left_low < len(left_array) or right_low < len(right_array):
                if left_low < len(left_array) and right_low < len(right_array):
                    if left_array[left_low] < right_array[right_low]:
                        output.append(left_array[left_low])
                        left_low += 1
                    else:
                        output.append(right_array[right_low])
                        right_low += 1
                elif left_low < len(left_array):
                    output.append(left_array[left_low])
                    left_low += 1
                else:
                    output.append(right_array[right_low])
                    right_low += 1
            
            return output
            
        
        def sort(low, high):
            if low == high:
                return [nums[low]]
            elif low > high:
                return []
            
            mid = low + (high - low) // 2
            
            left = sort(low, mid)
            right = sort(mid + 1, high)
            
            return merge(left, right)
        
        if not nums:
            return []
        
        return sort(0, len(nums) - 1)
        

def merge(arr,l,m,r):
    n1 = m-l+1
    n2 = r-m
    L = [0]*n1
    R = [0]*n2
    for i in range(0,n1):
        L[i] = arr[l+i]
    for j in range(0,n2):
        R[j] = arr[m+1+j]
    
    i = 0
    j = 0
    k = l
    while i<n1 and j<n2:
        if L[i]<=R[j]:
            arr[k] = L[i]
            i += 1
        else:
            arr[k] = R[j]
            j += 1
        k += 1
    while i<n1:
        arr[k] = L[i]
        i += 1
        k += 1
    while j<n2:
        arr[k] = R[j]
        j += 1
        k += 1
    
    
    
def mergeSort(arr,l,r):
    if l<r:
        m = (l+r)//2
        mergeSort(arr,l,m)
        mergeSort(arr,m+1,r)
        merge(arr,l,m,r)
    return arr
    
class Solution:
    
    def sortArray(self, arr: List[int]) -> List[int]:
        '''
        #Insertion Sort
        for i in range(1, len(arr)):
            key = arr[i]
            # Move elements of arr[0..i-1], that are
            # greater than key, to one position ahead
            # of their current position
            j = i-1
            while j >=0 and key < arr[j] :
                arr[j+1] = arr[j]
                j -= 1
            arr[j+1] = key 
        return arr
        '''
        return mergeSort(arr,0,len(arr)-1)

class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        self.qs(nums, 0, len(nums) - 1)
        return nums
    
    def qs(self, nums, start, end):
        if start >= end:
            return
        
        pivot = nums[(start+end) // 2]
        
        left, right = start, end
        
        while left <= right:
            while left <= right and nums[left] < pivot:
                left += 1
            
            while left <= right and nums[right] > pivot:
                right -= 1
            
            if left <= right:
                nums[left], nums[right] = nums[right], nums[left]
                left += 1
                right -= 1
        
        self.qs(nums, start, right)
        self.qs(nums, left, end)

class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        
        # partition and return pivot index
        def partition(nums, low, high):
            p = nums[low]
            
            # i points the last one in region 1
            i,j = low, low+1
            while j <= high:
                if p > nums[j]:
                    # swap the first one in region 2 with current one 
                    # which should be in region 1
                    # then move i forward
                    nums[i+1], nums[j] = nums[j], nums[i+1]
                    i += 1
                j += 1
                
            # swap the pivot number with last one in region 1
            nums[i], nums[low] = nums[low], nums[i]
        
            return i
                
        # quicksort implementation
        def quickSort(nums, low, high):

            if low < high:
                # get pivot index
                pivot_ind = partition(nums, low, high)
                #print (pivot_ind, nums)

                # recursively call left and right part
                quickSort(nums,low, pivot_ind-1)
                quickSort(nums,pivot_ind+1, high)
        
        # now invoke quick sort
        low,high = 0, len(nums)-1
        quickSort(nums, low, high)
        return nums
class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        
        def devidelist(L:List[int]):
            if len(L)<=1:
                return L
            mid=len(L)//2
            left=devidelist(L[0:mid])
            right=devidelist(L[mid:])
            return mergesort(left,right)
            
        def mergesort(left:List[int],right:List[int]):
            if not left:
                return right
            if not right:
                return left
            lidx=0
            ridx=0
            ans=[]
            while lidx<len(left) or ridx<len(right):
                if lidx==len(left) :
                    ans.append(right[ridx])
                    ridx+=1
                    continue
                if ridx==len(right) :
                    ans.append(left[lidx])
                    lidx+=1
                    continue
                if left[lidx]<=right[ridx]:
                    ans.append(left[lidx])
                    lidx+=1
                else:
                    ans.append(right[ridx])
                    ridx+=1
            return ans  
                    
        return devidelist(nums)
class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        if nums == [] or len(nums) == 1:
            return nums
        if len(nums) == 2 and nums[0] < nums[1]:
            return nums
        
        first = 0
        middle = len(nums)//2
        last = len(nums)-1
        # make the medium value at the first position
        if nums[first] <= nums[middle] <= nums[last] or nums[last] <= nums[middle] <= nums[first]:
            nums[first], nums[middle] = nums[middle], nums[first]
        elif nums[first] <= nums[last] <= nums[middle] or nums[middle] <= nums[last] <= nums[first]:
            nums[first], nums[last] = nums[last], nums[first]
        pivot = 0
        cur = 0
        boarder = 0
        for i in range(1, len(nums)):
            if nums[i] > nums[pivot]:
                boarder = i
                break
        cur = boarder + 1       
        while cur < len(nums):
            if nums[cur] <= nums[pivot]:
                # print('a', nums)
                nums[boarder], nums[cur] = nums[cur], nums[boarder]
                # print('b', nums)
                boarder += 1
            cur += 1
        # nums[cur-1], nums[boarder] = nums[boarder], nums[cur-1]
        # print(nums)
        
        left = nums[:boarder]
        right = nums[boarder:]
        nums = self.sortArray(left) + self.sortArray(right)
        
        return nums
        
            
        
        
        
        

class Solution:
    def sortArray(self, listToSort: List[int]) -> List[int]:
        divider = len(listToSort)//2
        a = listToSort[:divider]
        b = listToSort[divider:]
        sortedList= listToSort[0:len(listToSort)]

        a.sort()
        b.sort()

        i = 0
        j = 0
        k = 0

        while i < len(a) and j < len(b):
            if b[j] < a[i]:
                sortedList[k] = b[j]
                j += 1
                k += 1
            else:
                sortedList[k] = a[i]
                i += 1
                k += 1

        if i < len(a):
            sortedList[k:] = a[i:]

        if j < len(b):
            sortedList[k:] = b[j:]

        listToSort[0:len(listToSort)] = sortedList[0:len(sortedList)]
    
        return listToSort

class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        self.merge(nums, 0, len(nums)-1)
        return nums
        
    def merge(self, nums, s, e):
        if s >= e:
            return
        
        m = (s+e) // 2
        i, j, k = s, m + 1, 0
        
        self.merge(nums, s, m)
        self.merge(nums, m+1, e)
        
        temp = [0 for _ in range(e-s+1)]
        
        while i<=m and j<=e:
            if nums[i] <= nums[j]:
                temp[k] = nums[i]
                i += 1
            else:
                temp[k] = nums[j]
                j += 1
            k += 1
        
        while i<=m:
            temp[k] = nums[i]
            i+=1
            k+=1
        
        while j<=e:
            temp[k] = nums[j]
            j+=1
            k+=1
        
        for i in range(len(temp)):
            nums[s+i] = temp[i]
            
"""
nums        : [4, 2, 3, 1]

            merge(0, 3)
            /   
        m(0,1)  m(2,3)
        /
    
         s
           e
         m
           i
             j
nums    [1,1]
             k
temp    [2,4]
    


"""
        
    
"""
if s + 1 <= e:
            m = (s+e) // 2
            i, j = s, m+1
            temp = [0 for _ in range(e-s+1)]
            k = 0
            while i <= m and j <= e:
                if nums[i] <= nums[j]:
                    temp[k] = nums[i]
                    i += 1
                else:
                    temp[k] = nums[k]
                    j += 1
                k += 1
            
            while i <= m:
                temp[k] = nums[i]
                k += 1
                i += 1
            
            for m in range(len(temp)):
                s[s+m] = temp[m]
        
"""
    


"""
10:51
0) Problem is 
    - sort an array
    - increasing order !!
    
    
    
1) Don't forget
- nums.length >= 1
- nums[i]  -50k ~ 50k

2) Brain Storming

                       s
                                e
                                
                          m
                       i
                             j
                       0  1  2  3
        nums        : [3, 2, 5, 4]
        
                      s   e
                      m
                      i  
                             j
        nums        : [3, 2]
        temp        : [2]
        
        if s + 1 <= e:
            m = (s+e) // 2
            i, j = s, m+1
            temp = [0 for _ in range(e-s+1)]
            k = 0
            while i <= m and j <= e:
                if nums[i] <= nums[j]:
                    temp[k] = nums[i]
                    i += 1
                else:
                    temp[k] = nums[k]
                    j += 1
                k += 1
            
            while i <= m:
                temp[k] = nums[i]
                k += 1
                i += 1
            
            for m in range(len(temp)):
                s[s+m] = temp[m]
        
        
            merge(s, e)
          /         
    merge(s, m)     merge(m+1, e)
    


"""
import math

class Solution:
    
    def merger(self, left, right):
        
        lidx  = 0
        ridx  = 0
        final = []
        
        while lidx < len(left) or ridx < len(right):
            if ridx == len(right):
                final.append(left[lidx])
                lidx += 1
            elif lidx == len(left):
                final.append(right[ridx])
                ridx += 1
            else:
                curt_l = left[lidx]
                curt_r = right[ridx]
                if curt_l <= curt_r:
                    final.append(curt_l)
                    lidx += 1
                else:
                    final.append(curt_r)
                    ridx += 1
        return final


    
    def sortArray(self, nums: List[int]) -> List[int]:
        
        if len(nums) == 1:
            return nums
        elif len(nums) > 1:
            mid_idx = math.ceil(len(nums)/2)
            end_idx = len(nums)
            
            sorted_left  = self.sortArray(nums[0:mid_idx])
            sorted_right = self.sortArray(nums[mid_idx:end_idx])
            
            final = self.merger(sorted_left, sorted_right)
            return final
class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        def merge(l1, l2):
            p1 = p2 = 0
            new_list = []
            while p1 < len(l1) or p2 < len(l2):
                if p1 < len(l1) and p2 < len(l2):
                    if l1[p1] < l2[p2]:
                        new_list.append(l1[p1])
                        p1 += 1
                    else:
                        new_list.append(l2[p2])
                        p2 += 1
                elif p1 < len(l1):
                    new_list += l1[p1:]
                    p1 = len(l1)
                else:
                    new_list += l2[p2:]
                    p2 = len(l2)
            return new_list
        if len(nums) <= 1:
            return nums
        pivot = len(nums) // 2
        left = self.sortArray(nums[:pivot])
        right = self.sortArray(nums[pivot:])
        return merge(left, right)
class Solution:
    def merge(self,left,mid,right,arr):
        
        i=left
        j=mid+1
        
        temp=[]
        while(i<=mid and j<=right):
            
            if arr[i]<arr[j]:
                temp.append(arr[i])
                i+=1
            else:
                temp.append(arr[j])
                j+=1
        
        while(i<=mid):
            temp.append(arr[i])
            i+=1
        
        while(j<=right):
            temp.append(arr[j])
            j+=1
        
        j=0
        for i in range(left,right+1):
            arr[i]=temp[j]
            j+=1
      
    
    def mergesort(self,left,right,arr):
        if left>=right:
            return
        else:
            mid=(left+right)//2
            
            self.mergesort(left,mid,arr)
            
            self.mergesort(mid+1,right,arr)
            
            self.merge(left,mid,right,arr)
            
            return
        
    def insertionSort(self,arr):
        n=len(arr)

        for i in range(1,n):
            
            key=arr[i]
            
            j=i-1
            
            while(j>=0 and key<arr[j]):
                
                arr[j+1]=arr[j]
                j-=1
            
            arr[j+1]=key
        
            
    def heapify(self,index,n,arr):
        i=index
        left=2*i+1
        right=2*i+2
        max_index=i
        while(left<n):
            
            
            if arr[left]>arr[max_index]:
                max_index=left
            
            if right<n:
                
                if arr[right]>arr[max_index]:
                    max_index=right
            
            if max_index==index:
                break
                
            arr[max_index],arr[index]=arr[index],arr[max_index]
            
            index=max_index
            left=2*index+1
            right=2*index+2
    
    
    def heapsort(self,arr):
        
        n=len(arr)
        
        for i in range(0,n):
            
            self.heapify(n-i-1,n,arr)
        
        for i in range(0,n):
            
            arr[0],arr[n-i-1]=arr[n-i-1],arr[0]
            
            self.heapify(0,n-i-1,arr)
    
                    
            
    def sortArray(self, nums: List[int]) -> List[int]:
        
        #self.mergesort(0,len(nums)-1,nums)
        #self.insertionSort(nums)
        self.heapsort(nums)
        return nums

class node:
    def __init__(self, val):
        self.val = val
        self.right = None
        self.left = None
        
    def insert(self, val):
        if self.val is not None:
            if val < self.val:
                if self.left is None:
                    self.left = node(val)
                else:
                    self.left.insert(val)
            else:
                if self.right is None:        
                    self.right = node(val)
                else:
                    self.right.insert(val)
        else:
            self.val = val
            
def inorder(root, res):
    if root:
        inorder(root.left, res)
        res.append(root.val)
        inorder(root.right, res)
            

class Solution:
    root = None
    
    def sortArray(self, nums: List[int]) -> List[int]:
        if len(nums) > 0:
            self.root = node(nums[0])
            for val in nums[1:]:
                self.root.insert(val)
                res = []
            res = []
            inorder(self.root, res)
            print(res)
            return res
class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        if len(nums)<=1:
            return nums
        if len(nums)==2:
            if nums[0]>nums[1]:
                return [nums[1],nums[0]]
            else:
                return nums
            
        mid = len(nums)//2
        left = self.sortArray(nums[:mid+1])
        right = self.sortArray(nums[mid+1:])
        return self.mergeSortedArray(left, right)
        
        
    def mergeSortedArray(self, nums1, nums2):
        res = []
        while nums1 and nums2:
            if nums1[0]<=nums2[0]:
                res.append(nums1.pop(0))
            else:
                res.append(nums2.pop(0))
        res += nums1 + nums2
        return res
class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        
        def mergeSort(arr):
            
            if len(arr) <= 1:
                return arr
            
            mid = len(arr) // 2
            left = mergeSort(arr[:mid])
            right = mergeSort(arr[mid:])
            
            newArr = []
            i = 0
            j = 0
            while i < len(left) or j < len(right):
                if j >= len(right) or (i < len(left) and left[i] <= right[j]):
                    newArr.append(left[i])
                    i += 1
                elif i >= len(left) or (j < len(right) and right[j] <= left[i]):
                    newArr.append(right[j])
                    j += 1
            
            return newArr
        
        return mergeSort(nums)

def merge(L,R):
    if len(L)==0:
        return R
    if len(R)==0:
        return L
    l=0
    r=0
    res=[]
    while len(res)<len(L)+len(R):
        if L[l]<R[r]:
            res.append(L[l])
            l+=1
        else:
            res.append(R[r])
            r+=1
        if l==len(L):
            res+=R[r:]
            break
        if r==len(R):
            res+=L[l:]
            break
    return res
class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        if len(nums)<2:
            return nums
        mid=len(nums)//2
        L=self.sortArray(nums[:mid])
        R=self.sortArray(nums[mid:])
        return merge(L,R)

class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        return self.mergeSort(nums, 0, len(nums) - 1)
        
    # 4 1 6 6 8 2
    def mergeSort(self, arr, i, j):
        if i == j:
            return [arr[i]]
        
        mid = i + (j - i) // 2
        
        left_arr = self.mergeSort(arr, i, mid)
        right_arr = self.mergeSort(arr, mid + 1, j)
    
        return self.merge(left_arr, right_arr)
    
    def merge(self, a, b):
        res = []
        i = 0
        j = 0
        
        while len(res) < len(a) + len(b):
            if j == len(b) or (i < len(a) and a[i] < b[j]):
                res.append(a[i])
                i += 1
            else:
                res.append(b[j])
                j += 1
            
        return res
class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        if len(nums)<=1:
            return nums
            
        mid = len(nums)//2
        left = self.sortArray(nums[:mid])
        right = self.sortArray(nums[mid:])
        return self.mergeSortedArray(left, right)
        
        
    def mergeSortedArray(self, nums1, nums2):
        res = []
        while nums1 and nums2:
            if nums1[0]<=nums2[0]:
                res.append(nums1.pop(0))
            else:
                res.append(nums2.pop(0))
        res += nums1 + nums2
        return res
import heapq
class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        heapify(nums)
        new = []
        
        while nums:
            new.append(heappop(nums))
        return new
class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        if len(nums) <= 1:
            return nums
        pivot = len(nums) // 2
        left_list = self.sortArray(nums[:pivot])
        right_list = self.sortArray(nums[pivot:])
        return self.merge_sort(left_list, right_list)
    
    def merge_sort(self, left_list, right_list):
        sorted_list = []
        left_idx, right_idx = 0, 0
        while left_idx <= len(left_list) - 1 and right_idx <= len(right_list) - 1:
            if left_list[left_idx] > right_list[right_idx]:
                sorted_list.append(right_list[right_idx])
                right_idx += 1
            else:
                sorted_list.append(left_list[left_idx])
                left_idx += 1
        
        sorted_list += left_list[left_idx:] 
        sorted_list += right_list[right_idx:] 
        
        return sorted_list
class Solution:
    def merge(self, left, right):
        sorted_list = []
        while left and right:
            if left[0] <= right[0]:
                sorted_list.append(left.pop(0))
            else:
                sorted_list.append(right.pop(0))
        if not left:
            sorted_list+=right
        if not right:
            sorted_list+=left
        return sorted_list
    
                
    def sortArray(self, nums: List[int]) -> List[int]:
        if len(nums)<2:
            return nums
        
        mid = len(nums)//2
        
        left = self.sortArray(nums[mid:])
        right = self.sortArray(nums[:mid])

        result = []
        result += self.merge(left, right)
        
        return result
    
    
        """Quick sort"""
#             if len(nums) <= 1:
#                 return nums

#             pivot = random.choice(nums)
#             lt = [v for v in nums if v < pivot]
#             eq = [v for v in nums if v == pivot]
#             gt = [v for v in nums if v > pivot]

#             return self.sortArray(lt) + eq + self.sortArray(gt)
class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        if len(nums) == 1:
            return nums
        mid = len(nums) // 2
        left = self.sortArray(nums[:mid])
        right = self.sortArray(nums[mid:])
        result = []
        p1 = p2 = 0
        while p1 < len(left) and p2 < len(right):
            if left[p1] < right[p2]:
                result.append(left[p1])
                p1 += 1
            else:
                result.append(right[p2])
                p2 += 1
        
        result.extend(left[p1:])
        result.extend(right[p2:])
        
        return result

class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        """
        Solution:
        #1 merge sort, time O (nlogn), space O(n) 
        
        """
        # return self.mergeSort(nums)
        return self.heap_sort(nums)
    
    def mergeSort(self, nums):
        # exit
        if len(nums) <= 1:
            return nums
        # u5411u4e0bu53d6u6574
        mid = len(nums) // 2
        left = self.mergeSort(nums[:mid])
        right = self.mergeSort(nums[mid:])
        
        # merge two sub list
        ret = []
        l_idx = r_idx = 0
        while l_idx < len(left) and r_idx < len(right):
            if left[l_idx] <= right[r_idx]:
                ret.append(left[l_idx])
                l_idx += 1
            else:
                ret.append(right[r_idx])
                r_idx += 1
        while l_idx < len(left):
            ret.append(left[l_idx])
            l_idx += 1
        while r_idx < len(right):
            ret.append(right[r_idx])
            r_idx += 1
        return ret
        
        
    def heap_sort(self, nums):
        # u8c03u6574u5806
        # u8fedu4ee3u5199u6cd5
        # def adjust_heap(nums, startpos, endpos):
        #     newitem = nums[startpos]
        #     pos = startpos
        #     childpos = pos * 2 + 1
        #     while childpos < endpos:
        #         rightpos = childpos + 1
        #         if rightpos < endpos and nums[rightpos] >= nums[childpos]:
        #             childpos = rightpos
        #         if newitem < nums[childpos]:
        #             nums[pos] = nums[childpos]
        #             pos = childpos
        #             childpos = pos * 2 + 1
        #         else:
        #             break
        #     nums[pos] = newitem

        # u9012u5f52u5199u6cd5
        def adjust_heap(nums, startpos, endpos):
            pos = startpos
            chilidpos = pos * 2 + 1
            if chilidpos < endpos:
                rightpos = chilidpos + 1
                if rightpos < endpos and nums[rightpos] > nums[chilidpos]:
                    chilidpos = rightpos
                if nums[chilidpos] > nums[pos]:
                    nums[pos], nums[chilidpos] = nums[chilidpos], nums[pos]
                    adjust_heap(nums, chilidpos, endpos)

        n = len(nums)
        # u5efau5806
        for i in reversed(range(n // 2)):
            adjust_heap(nums, i, n)
        # u8c03u6574u5806
        for i in range(n - 1, -1, -1):
            nums[0], nums[i] = nums[i], nums[0]
            adjust_heap(nums, 0, i)
        return nums

            
            
        
class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        tmp = [0 for _ in range(len(nums))]
        self.ms(nums, 0, len(nums) - 1, tmp)
        return nums
    
    def ms(self, nums, start, end, tmp):
        if start >= end:
            return
        
        mid = (start + end) // 2
        self.ms(nums, start, mid, tmp)
        self.ms(nums, mid+1, end, tmp)
        self.merge(nums, start, mid, end, tmp)
    
    def merge(self, nums, start, mid, end, tmp):
        left, right = start, mid + 1
        idx = start
        
        while left <= mid and right <= end:
            if nums[left] < nums[right]:
                tmp[idx] = nums[left]
                left += 1
            else:
                tmp[idx] = nums[right]
                right += 1
            idx += 1
        
        while left <= mid:
            tmp[idx] = nums[left]
            idx += 1
            left += 1
        
        while right <= end:
            tmp[idx] = nums[right]
            idx += 1
            right += 1
        
        for i in range(start, end+1):
            nums[i] = tmp[i]
        

class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        self.heapSort(nums)
        return nums
  
    
	# @quickSort
    def quickSort(self, nums):
        
        def partition(head, tail):
            
            if head >= tail: return
            
            l, r = head, tail
            m = (r - l) // 2 + l
            pivot = nums[m]
            
            while r >= l:
                while r >= l and nums[l] < pivot: 
                    l += 1
                while r >= l and nums[r] > pivot: 
                    r -= 1
                    
                if r >= l:
                    nums[l], nums[r] = nums[r], nums[l]
                    l += 1
                    r -= 1
                    
            partition(head, r)
            partition(l, tail)

        partition(0, len(nums)-1)
        return nums
     
	# @mergeSort
    
    def mergeSort(self, nums): 
        
        if len(nums) > 1: 
            
            mid = len(nums)//2
            
            L = nums[:mid] 
            R = nums[mid:] 

            self.mergeSort(L)
            self.mergeSort(R)

            i = j = k = 0

            while i < len(L) and j < len(R): 
                if L[i] < R[j]: 
                    nums[k] = L[i] 
                    i+=1
                else: 
                    nums[k] = R[j] 
                    j+=1
                k+=1
 
            while i < len(L): 
                nums[k] = L[i] 
                i+=1
                k+=1

            while j < len(R): 
                nums[k] = R[j] 
                j+=1
                k+=1
   
   # @heapSort
    
    def heapSort(self, nums):
        
        def heapify(nums, n, i): 
            
            l = 2 * i + 1 #left
            r = 2 * i + 2 #right
			
            largest = i
            
            if l < n and nums[largest] < nums[l]: 
                largest = l 

            if r < n and nums[largest] < nums[r]: 
                largest = r 

            if largest != i: #if its not equal to root
                nums[i], nums[largest] = nums[largest], nums[i] #swap
                
                heapify(nums, n, largest)
                
        n = len(nums) 

        for i in range(n//2+1)[::-1]:  #starting from the last non leaf node and going until the top
            heapify(nums, n, i) 

        for i in range(n)[::-1]: 
            nums[i], nums[0] = nums[0], nums[i]
            heapify(nums, i, 0)
def max_heapify(nums, i, lo, hi):
    l = 2 * i + 1
    r = 2 * i + 2
    largest = i
    if l <= hi and nums[i] < nums[l]:
        largest = l
    if r <= hi and nums[largest] < nums[r]:
        largest = r
    if largest != i:
        nums[i], nums[largest] = nums[largest], nums[i]
        max_heapify(nums, largest, lo, hi)

def build_max_heap(nums):
    for i in range(len(nums)//2 - 1, -1, -1):
        max_heapify(nums, i, 0, len(nums)-1)

class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        build_max_heap(nums)
        for i in range(len(nums)-1, 0, -1):
            nums[0], nums[i] = nums[i], nums[0]
            max_heapify(nums, 0, 0, i-1)
        return nums
import math
class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        def sort(start, stop):
            if start >= stop:
                return nums[start:stop + 1]
            mid = math.floor((start + stop) / 2)
            left = sort(start, mid)
            right = sort(mid + 1, stop)
            return merge(left, right)
        def merge(left, right):
            left_index = 0
            right_index = 0
            result = []
            while left_index < len(left) and right_index < len(right):
                left_val = left[left_index]
                right_val = right[right_index]
                if left_val < right_val:
                    left_index += 1
                    result.append(left_val)
                else:
                    right_index += 1    
                    result.append(right_val)
            if left_index >= len(left):
                result.extend(right[right_index:])
            else:
                result.extend(left[left_index:])
            return result
        return sort(0, len(nums) - 1)
        

class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        
        if nums is None or len(nums) < 2:
            return nums
        
        n = len(nums)
        
        for i in range(n//2 - 1, -1, -1):
            self.heapify(nums, n, i)
            
        for i in range(n-1, -1, -1):
            nums[i], nums[0] = nums[0], nums[i]
            self.heapify(nums, i, 0)
            
        return nums
    
    def heapify(self, arr, n, i):
        
        largest = i
        leftChild = 2 * i + 1
        rightChild = 2 * i + 2
        
        if leftChild < n and arr[largest] < arr[leftChild]:
            largest = leftChild
            
        if rightChild < n and arr[largest] < arr[rightChild]:
            largest = rightChild
            
        if largest != i:
            arr[largest], arr[i] = arr[i], arr[largest]
            self.heapify(arr, n, largest)
            

