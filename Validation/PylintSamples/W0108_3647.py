def solve(arr):
    arr = sorted(arr, reverse=True)
    res = []
    while len(arr):
        res.append(arr.pop(0))
        if len(arr):
            res.append(arr.pop())
    return res
def solve(arr):
    lmax, lmin = iter(sorted(arr)) , iter(sorted(arr)[::-1])
    return [next(lmax) if i%2==1 else next(lmin) for i in range(0,len(arr))]
def solve(arr):
    return [sorted(arr)[::-1][(-1)**i*i//2] for i in range(len(arr))]
def solve(arr):
    rarr = sorted(arr, reverse=True)
    farr = rarr[::-1]
    return [item for sublist in zip(rarr, farr) for item in sublist][:len(rarr)]
def solve(arr):
    arr.sort(reverse=True)
    return [arr.pop(-(i % 2)) for i in range(len(arr))]
def solve(arr):
    l = sorted(arr)
    result = []
    while l:
        result.append(l.pop())
        l.reverse()
    return result
# I haven't tested them, but I think candle_index is the best of these

def zip_slices(arr):
    arr = sorted(arr)[::-1]
    res = []
    n = len(arr)
    m = n // 2
    mins = arr[m:][::-1]
    maxes = arr[:m]
    for a, b in zip(maxes, mins):
        res.extend([a, b])
    if n % 2:
        res.append(mins[-1])
    return res

def candle_pop(arr):
    candle = sorted(arr)
    res = []
    i = -1
    while candle:
        res.append(candle.pop(i))
        i = 0 if i else -1
    return res

def candle_index(arr):
    candle = sorted(arr)
    n = len(arr)
    a = 0
    z = n - 1
    res = []
    for i in range(n):
        if i % 2:
            res.append(candle[a])
            a += 1
        else:
            res.append(candle[z])
            z -= 1
    return res

def set_pop(arr):
    nums = set(arr)
    res = []
    i = 0
    while nums:
        if i % 2:
            n = min(nums)
        else:
            n = max(nums)
        res.append(n)
        nums.remove(n)
        i += 1
    return res

from random import randint

solve = lambda arr: (zip_slices, candle_pop, candle_index, set_pop)[randint(0,3)](arr)
def solve(arr):
    arr=sorted(arr)
    arr.append(0)
    arr=sorted(arr)
    list1 = []
    for i in range(1,len(arr)//2+2):
        list1.append(arr[-i])
        list1.append(arr[i])
    list1=list(dict.fromkeys(list1))
    return list1

def solve(arr):
    
    # sort and reverse the list, make a new list to store answer
    arr = sorted(arr, reverse=True)
    new_list = []
    
    boolean = True
    # while there are still items in arr[]
    while (len(arr) >= 1):
        
        # append the 0th item to new_list[]
        new_list.append(arr[0])
        
        # flip our boolean value from true to false to reverse the list later
        if (boolean == True) :  
            boolean = False      
        else:
            boolean = True
        
        # remove the 0th element from arr[]
        arr.pop(0)
        
        # sort the list either forwards or in reverse based on boolean
        arr = sorted(arr, reverse=boolean)
        
    return new_list
def solve(arr):
    print(arr)
    max_l=sorted(arr,reverse=True)
    min_l=sorted(arr)
    l=[]
    for i,a in enumerate(zip(max_l,min_l)):
        if a[0]!=a[1]:
            l.append(a[0])
            l.append(a[1])
        else:
            l.append(a[0])
            break
    return list(dict.fromkeys(l))
