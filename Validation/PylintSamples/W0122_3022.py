def two_highest(ls):
    result = sorted(list(set(ls)), reverse=True)[:2]
    return result if isinstance(ls, (list)) else False
def two_highest(list):
    return False if isinstance(list, str) else sorted(set(list), reverse=True)[0:2]
import heapq
def two_highest(list_):
    return heapq.nlargest(2, set(list_)) if type(list_) == list else False
from heapq import nlargest

def two_highest(lst):
    return isinstance(lst, list) and nlargest(2, set(lst))
def two_highest(l):
    if type(l) != list:
        return False
    return sorted(list(set(l)), key=lambda k: -k)[:2]
def two_highest(arg1):
    if type(arg1) is list:
        new_list = sorted(list(set(arg1)))
        new_list.reverse()
        return new_list[0: 2]
    else:
        return False
two_highest = lambda l: isinstance(l, list) and sorted(set(l))[-2:][::-1]
def two_highest(x):
    if type(x) is list:
        return [] if len(x) == 0 else sorted(list(set(x)))[::-1][:2]
    
    return False
from heapq import nlargest

def two_highest(arg1, n=2):
    return type(arg1) is list and nlargest(n, set(arg1))
def two_highest(arg1):
    return type(arg1) == list and sorted(list(set(arg1)), reverse=True)[:2]
def two_highest(arg1):
    l=[]
    for x in arg1:
        if type(x)!=int:
            return False
        else:
            l.append(x)
    a=set(l)
    return sorted(a,reverse=True)[:2]
def two_highest(arg1):
    return False if type(arg1)!=list else sorted(set(arg1),reverse = True)[:2]
def two_highest(arg):
  return sorted(list(set(arg)), reverse=True)[:2] if isinstance(arg, list) else False
import heapq

def two_highest(l):
    return heapq.nlargest(2, set(l)) if type(l) == list else False
def two_highest(mylist):
    if not isinstance(mylist, list):
        return False
    return sorted(list(set(mylist)), reverse=True)[:2]
def two_highest(list):
    if len(list) < 0:
        return []
    elif len(list) > 0 and not str(list[0]).isdigit():
        return False
    return sorted(set(list))[::-1][0:2]
import heapq
def two_highest(s):
    return False if type(s) == str else heapq.nlargest(2,set(s))
def two_highest(list):
    return sorted(set(list))[:-3:-1]
def two_highest(x):
  if (type(x)==type("cool")):
    return False
  print(x)
  x.sort()
  x.reverse()
  i=0
  if len(x)>1:
    while x[i]==x[0]:
      i+=1    
    return [x[0],x[i]]
  else:
    return x
two_highest = lambda a, b=exec("import heapq"): heapq.nlargest(2, set(a)) if type(a) is list else False
two_highest = lambda a: sorted(set(a), reverse=True)[:2] if type(a) is list else False
from typing import List

def two_highest(_array: List[int]) -> List[int]:
    """ Get two max (and unique) values in the given list and return them sorted from highest to lowest. """
    return sorted(set(_array), reverse=True)[:2]
def two_highest(arg1):
    return sorted(set(arg1), reverse=True)[:2]
def two_highest(lst):
    return isinstance(lst, list) and sorted(set(lst), reverse = 1)[:2]
def two_highest(a):
    return [0,sorted(set(a))[:-3:-1]][type(a)==list]
def two_highest(a):
    return sorted(set(a), reverse=True)[:2] if isinstance(a, list) else False
def two_highest(arg1):
    if type(arg1) is not list:
        return False 
    else:
        for x in arg1:
            while int(arg1.count(x)) > 1: 
                arg1.remove(x)
        return sorted(arg1,reverse=True)[:2]
def two_highest(arg1):
    return False if type(arg1) != list else [] if len(arg1) == 0 else sorted(set(arg1), reverse = True)[:2]
two_highest=lambda a:list==type(a)and sorted(set(a))[:-3:-1]
def two_highest(l):
    return isinstance(l, list) and sorted(set(l), key=lambda x: -x)[:2]
def two_highest(nums):
    return sorted(set(nums))[-2:][::-1]
def two_highest(arg1):
    new = sorted(list(set(arg1)))
    return [new[-1], new[-2]] if len(new) > 1 else arg1
def two_highest(arg1):
    if len(arg1)>1:
        a=(set(arg1))
        b=max(a)
        a.remove(b)
        d=max(a)
        return [b, d]
    else:
        return arg1

def two_highest(arg1):
    print(arg1)
    try:
        if len(arg1) == 1:
            return arg1
        else:
            value1 = max(arg1)
            arg2 = []
            for i in arg1:
                if i != value1:
                    arg2.append(i)
            value2 = max(arg2)
            return [value1, value2]
    except ValueError:
        return []
def two_highest(arg1):
    r = sorted(list(set(arg1)), reverse=True)
    return r[:2]
def two_highest(arg1):
    l = list(sorted(set(arg1))[-2:])
    l.reverse()
    return l
def two_highest(arg1):
    
    if len(arg1) < 1:
        return []
    elif len(arg1) < 2:
        return arg1
    
    res = [arg1[0], arg1[1]] if arg1[0] > arg1[1] else [arg1[1], arg1[0]]
    
    for i in arg1:
        if i > res[0]:
            res[1] = res[0]
            res[0] = i
        elif i > res[1] and i != res[0]:
            res[1] = i
            
    print(arg1)
    
    return res if res[0] != res[1] else res[0]
def two_highest(arg1):
    
    if len(arg1) < 1:
        return []
    elif len(arg1) < 2:
        return arg1
    
    res1 = max(arg1)
    while(max(arg1) == res1):
        arg1.remove(res1)
    res2 = max(arg1)
    
    return [res1, res2] if res1 != res2 else [res1]

def two_highest(arg1):
    l = list(set(arg1))
    l.sort()
    if len(l) >= 2:
        return [ l[-1], l[-2] ]
    elif len(l) == 1:
        return [ l[-1] ]
    else:
        return []

def two_highest(arg1):
    arg2 = list(set(arg1))
    arg2 = sorted(arg2)
    
    if len(arg2) == 0:
        return []
    if len(arg2) == 1:
        return arg2
    else:
        return [arg2[-1], arg2[-2]]
    pass
def two_highest(arg1):
    print(sorted(list(set(arg1)), reverse=True))
    return sorted(list(set(arg1)), reverse=True)[:2]
def two_highest(arg1):
    if len(arg1) <=1:
        return arg1
    h1 = arg1[0]
    h2 = arg1[1]
    if h2 > h1:
        h1, h2 = h2, h1
    for i in range(2, len(arg1)):
        if arg1[i] > h1:
            h2 = h1
            h1 = arg1[i]
        elif arg1[i] > h2 and arg1[i] != h1:
            h2 = arg1[i]
    if h1 != h2:
        return [h1, h2]
    return [h1]
        

        
    

def two_highest(arg1):
    arg1 = set(arg1)
    arg1 = sorted(list(arg1))
    
    if len(arg1) > 1:
        return [arg1[-1],arg1[-2]]
    else:
        return arg1
def two_highest(arg1):
    return sorted(set(arg1))[::-1][:2]
def two_highest(arg1):
    return list(reversed(sorted(set(arg1))[-2:]))
def two_highest(arg1):
    #set() to get unique values
    #sorted() to sort from top to bottom
    #slice [:2] to pick top 2
    return(sorted(set(arg1),reverse=True)[:2])
def two_highest(arg1):
    s = sorted(set(arg1), reverse=1)
    return [s[0],s[1]] if len(s) > 1 else s
def two_highest(arg1):
    if arg1 == [] : return []
    unique_integers = set(arg1)  
    if len(unique_integers) == 1 : return list(unique_integers)
    largest_integer = max(unique_integers) 
    unique_integers.remove(largest_integer)

    second_largest_integer = max(unique_integers)
    
    return [largest_integer, second_largest_integer]
def two_highest(arg1):
    new = set(arg1)
    final_list = list(new)
    final_list.sort()
    if len(final_list) == 1:
        return final_list
    elif final_list == []:
        return final_list

    highest = [final_list[-1], final_list[-2]]
    
    return highest
    

def two_highest(arg1):
    arg1 = list(set(arg1))
    arg1.sort()
    highest = arg1[-2:]
    highest.reverse()
    return highest

def two_highest(arg1):
    return (list(dict.fromkeys(sorted(arg1)))[-2:])[::-1]

def two_highest(arg1):
    return sorted(list(dict.fromkeys(arg1)), reverse=True)[:2]
def two_highest(arg1):
    print(arg1)
    if not arg1 or len(arg1) < 2:
        return arg1
    arg1 = set(arg1)
    a1 = max(arg1)
    arg1.remove(a1)
    a2 = max(arg1)
    return [a1, a2]
def two_highest(arg1):
    res = []
    gab = set(arg1)
    sag = list(gab)
    dad = sorted(sag)
    if len(dad) == 0:
        return []
    elif len(dad) == 1 or len(dad) == 2:
        return dad
    else:
        res.append(dad[-1])
        res.append(dad[-2])
        return res

def two_highest(arg1):
    if arg1:
        lst = sorted((list(set(arg1))), reverse=True)
        return lst[:2] if len(lst) >= 2 else arg1
    else: 
        return arg1
import heapq

def two_highest(arg1):
    return heapq.nlargest(2, set(arg1))
def two_highest(a):
    r = []
    
    for item in sorted(a)[::-1]:
        if item not in r:
            r.append(item)
    return r[:2]
def two_highest(arg1):
    mylist = list(dict.fromkeys(arg1))
    mylist = sorted(mylist, reverse = True)
    return mylist[0:2]
def two_highest(arg1):
    pass
    if len(arg1) == 0:
        return []
    a = sorted(arg1, reverse = True)
    if a[0] == a[-1]:
        return [a[0]]
    else:
        return [a[0], a[a.count(a[0])]]
def two_highest(a):
    return a and sorted(set(a), reverse = True)[0:2]
def two_highest(arg1):
    out = []
    for v in sorted(arg1, reverse=True):
        if len(out) == 2:
            break
        if v not in out:
            out.append(v)
    return out
def two_highest(arg1):
    return sorted(set(arg1))[:-3:-1]
def two_highest(arg1):
    arg1 = sorted(set(arg1),reverse=True)
    return arg1[:2]
def two_highest(arg1):
    arg1 = list(set(arg1))
    terminos = []
    if len(arg1)>0:
        terminos.append(arg1[0])
        for x in arg1:
            if x>terminos[0]:
                terminos[0]=x
        arg1.remove(terminos[0])
    if len(arg1)>0:
        terminos.append(arg1[0])
        for x in arg1:
            if x>terminos[1]:
                terminos[1]=x
    return terminos
            
        

def two_highest(arg1):
    arg1 = sorted(arg1, reverse=True)
    res = []
    for i in range(len(arg1)):
        if len(res) == 2:
            break
        if arg1[i] not in res:
            res.append(arg1[i])
    return res
def two_highest(arg1):
    if arg1 == []: 
        return arg1
    returnArr = []
    val1 = max(arg1)
    returnArr.append(val1)
    for i in range(len(arg1)): 
        if val1 in arg1: 
            arg1.remove(val1)
    if arg1 == []: 
        return returnArr
    val2 = max(arg1)
    if val2 == None:
        return returnArr 
    returnArr.append(val2)
    return returnArr
def two_highest(arg1):
    if arg1 == []:
        return []
    elif len(set(arg1)) == 1:
        return arg1
    elif len(set(arg1)) == 2:
        #arg2 = arg1.sort(reverse = True)
        return arg1
    elif len(arg1) == 3 or len(arg1) > 3:
        x = []
        arg1 = set(arg1)
        arg1 = list(arg1)
        max_1 = max(arg1)
        x.append(max_1)
        arg1.remove(max_1)
        max_2 = max(arg1)
        x.append(max_2)
        return x
    
    
    

def two_highest(arg1):
    x = []
    y = []
    z = []
    if len(arg1) == 0:
        return []
    elif len(arg1) == 1:
        return arg1
    else:
        
        x = sorted(arg1, reverse = True)
        for i in range(len(x)):
            if x[i] in y:
                y = y
            else:
                y.append(x[i])
        z.append(y[0])
        z.append(y[1])
        return z
def two_highest(arg1):
    unique = set(arg1)
    if len(unique) <= 2:
        return sorted(list(unique), reverse = True)
    max1 = max(unique)
    unique.discard(max1)
    max2 = max(unique)
    return [max1, max2]
def two_highest(arg1):
    arg1=list(set(arg1))
    arg1.sort()
    return arg1[-2:][::-1]
def two_highest(arg1):
    distinct = list(set(arg1))
    return sorted(distinct, reverse=True)[:2]

def two_highest(arg1):
    a=[]
    a=list(set(arg1))
    a.sort(reverse=True)
    return a[:2] if len(a)>2 else a
def two_highest(arg1):
    arg1 = list(set(arg1))
    arg1.sort(reverse=True)
    return arg1[0:2]
def two_highest(arg1):
    return sorted(set(arg1), reverse=True)[0:2]
def two_highest(arg1):
    return sorted(set(arg1))[-2:][::-1]
def two_highest(arg1):
    if type(arg1) is list:
        return [arg for arg in sorted(list(set(arg1)))[-1::-1][:2]]
def two_highest(arg1):
    return sorted(list(set(arg1)), reverse = True)[:2]

def two_highest(arg1):
    return sorted((x for x in set(arg1)), key=int)[::-1][:2]
from heapq import nlargest
def two_highest(arg1):
    return nlargest(2,set(arg1))
def two_highest(arg1):
    if len(arg1)==0:
        return []
    elif len(arg1)==1:
        return arg1
    else:
        return [sorted(set(arg1),reverse=True)[0],sorted(set(arg1),reverse=True)[1]]
def two_highest(arg1):
  zrobmyset = set(arg1)
  listabezdupl = list(zrobmyset)
  listabezdupl.sort()
  listabezdupl.reverse()
  return listabezdupl[:2]
def two_highest(arg1):
    result = []
    arg2 = sorted(set(arg1))
    if len(arg2) == 0:
        return result
    elif len(arg2) == 1:
        result.append(arg2[len(arg2) - 1])
    elif len(arg2) > 1:
        result.append(arg2[len(arg2) - 1])
        result.append(arg2[len(arg2) - 2])
    return result
def two_highest(arg1):
    result = []
    arg2 = sorted(set(arg1))
    if len(arg2) == 0:
        return result
    if arg2[len(arg2) - 1] != arg2[len(arg2) - 2] and len(arg2):
        result.append(arg2[len(arg2) - 1])
        result.append(arg2[len(arg2) - 2])
    elif arg2[len(arg2) - 1] == arg2[len(arg2) - 2] and len(arg2):
        result.append(arg2[len(arg2) - 1])
    return result
def two_highest(arg):
    if len(arg) <=1:
        return arg
    a = sorted(list(dict.fromkeys(arg)), reverse=True)
    return [a[0], a[1]]
def two_highest(arg1):
    rez = sorted(set(arg1))
    return [list(rez)[-1], list(rez)[-2]] if len(rez) > 1 else [list(rez)[-1]] if len(rez) == 1 else []

def two_highest(arg1):
    x = sorted(list(set(arg1)))
    return [x[-1], x[-2]] if len(x) >= 2 else x
def two_highest(arg1):
    arg1=list(dict.fromkeys(arg1))
    if len(arg1)>1:
        res=[]
        res.append(max(arg1))
        arg1.remove(max(arg1))
        res.append(max(arg1))
        arg1.remove(max(arg1))
        return res
    elif len(arg1)>0:
        res=[]
        res.append(max(arg1))
        arg1.remove(max(arg1))
        return res
    else:
        return []
def two_highest(arg1):
    if arg1 == []: return []
    if len(arg1) == 1: return arg1
    lst = sorted(set(arg1))
    return [lst[-1],lst[-2]]

def two_highest(arg1):
    uniques = set(arg1)
    if len(uniques) <= 2:
        return sorted(uniques, reverse=True)
    return sorted(uniques, reverse=True)[:2]
def two_highest(arg1):
    if arg1 == []:
        return arg1
    else:
        st = set(arg1)
        if len(st) == 1:
            return [arg1[0]]
        else:
            arg1 = sorted(list(st))
            print(arg1)
            return [arg1[-1], arg1[-2]]

def two_highest(arg1):
    return [max(arg1), max(x for x in arg1 if x != max(arg1))] if len(arg1) > 1 else [] if arg1 == [] else arg1
def two_highest(lst):
    return sorted(set(lst), reverse=True)[:2]
import sys
def two_highest(arg1):
    if len(arg1) == 0:
        return []
    if len(arg1) == 1:
        return [arg1[0]]
    
    x = -1
    y = -1
    
    for i in range(0, len(arg1)):
        if arg1[i] > y:
            x = y
            y = arg1[i]
        elif y > arg1[i] > x:
            x = arg1[i]
            
    return [y] if x == y else [y, x]
        
    

def two_highest(a):
    return sorted(set(a), reverse=True)[:2]
def two_highest(arg1):
    if type(arg1) is str:
        return False
    
    else:
        l = [i for i in set(arg1)]
        result = []
        j = 0
        for i in l:
            result.append(max(l))
            l.remove(max(l))
            j += 1
            if j  == 2:
                break
    
    return result
    

def two_highest(s):
    if type(s) != list:return False
    l = list(set(s))
    l.sort(reverse= True)
    return l[:2]
import heapq

def two_highest(lst):
    if not isinstance(lst,list):
        return False
    else:
        result = []
        for it in sorted(lst,reverse=True):
            if len(result) == 0:
                result.append(it)
            elif result[0] != it:
                result.append(it)
                return result
        return result

def two_highest(arg1):
    x = [i for i in arg1 if isinstance(i, str)]
    if len(x) > 0:
        return False
    if len(arg1) == 0:
        return []
    return sorted(list(set(arg1)))[-2:][::-1]
