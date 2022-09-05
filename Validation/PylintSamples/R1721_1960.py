class Solution:
    def processQueries(self, queries: List[int], m: int) -> List[int]:
        if not queries:
            return []
        p = list(range(1, m+1))
        res = []
        for i in queries:
            z = p.index(i)
            res.append(z)
            del p[z]
            p.insert(0,i)
        return res
            

class Solution:
    def processQueries(self, queries: List[int], m: int) -> List[int]:
        P = collections.deque(list(range(1,m+1)))
        
        res = []
        
        for q in queries:
            res.append( P.index(q) )
            del P[res[-1]]
            P.appendleft( q )
            # print(P)
        
        return res

class Solution:
    def processQueries(self, queries: List[int], m: int) -> List[int]:
        P = [i+1 for i in range(m)]
        res = []
        for q in queries:
            i = P.index(q)
            res.append(i)
            P = [q] + P[:i] + P[(i+1):]
        return res
class Solution:
    def processQueries(self, queries: List[int], m: int) -> List[int]:
        res = []
        P = list(range(1, m+1))
        print(P)
        for q in queries:
            i = P.index(q)
            res.append(i)
            P = [q] + P[:i] + P[i+1:]
            
        return res

class Solution:
    def processQueries(self, queries: List[int], m: int) -> List[int]:
        data, result = list(range(1, m + 1)), []
        for item in queries:
            idx = data.index(item)
            data = [data[idx]] + data[:idx] + (data[idx + 1:] if idx + 1 < len(data) else [])
            result.append(idx)
        return result
class Solution:
    def processQueries(self, queries: List[int], m: int) -> List[int]:
        
        P_lst  = [i for i in range(1,m+1)]
        q_list = []
        for i in range(0,len(queries)):
            target = queries[i]
            for j in range(0,len(P_lst)):
                
                if  P_lst[j] == target:
                    q_list.append(j)
                    x =P_lst.pop(j)
                    P_lst.insert(0,x)
                    break
        return q_list
        
                

class Solution:
    def processQueries(self, queries: List[int], m: int) -> List[int]:
        if len(queries)==0:
            return []
        P=[]
        for i in range(m):
            P.append(i+1)
        res=[]
        for i in queries:
            res.append(P.index(i))
            P.remove(i)
            P=[i]+P[:]
        return res
            

class Solution:
    def processQueries(self, queries: List[int], m: int) -> List[int]:
        P = [i for i in range(1,m+1)]
        
        ans = []
        for querie in queries:
            idx = P.index(querie)
            del P[idx]
            P = [querie] + P
            ans.append(idx)
            
        return ans

class Solution:
    def processQueries(self, queries: List[int], m: int) -> List[int]:
        permutations = [permutation for permutation in range(1, m + 1)]

        result = []
        for query in queries:
            for pindex, permutation in enumerate(permutations):
                if permutation == query:
                    result.append(pindex)
                    del permutations[pindex]
                    permutations.insert(0, permutation)
                    break
        print(result)
        print(permutations)
        
        return result

class Solution:
    def processQueries(self, queries: List[int], m: int) -> List[int]:
        original = list(range(1, m + 1))
        results = []
        for q in queries:
            res = 0
            for ind, el in enumerate(original):
                if el == q:
                    res = ind
                    break
            results.append(res)
            temp = original.pop(res)
            original = [temp] + original
        
        return results

class Solution:
    def processQueries(self, queries: List[int], m: int) -> List[int]:
        p=[]
        for i in range (1,m+1):
           p.append(i) 
        l=[]
        
        for i in range (0,len(queries)):
            for j in range (0,m):
                if p[j]==queries[i]:
                    k=p.pop(j)
                    p.insert(0,k)
                    l.append(j)
                    break
        return l
            
                

class Solution:
    def processQueries(self, queries: List[int], m: int) -> List[int]:
        A = list(range(1, m + 1))
        res = []
        for target in queries:
            for i, num in enumerate(A):
                if num == target:
                    res.append(i)
                    index = i
                    break

            element = A[index]
            A.pop(index)
            A.insert(0, element)
        
        return res
class Solution:
    def processQueries(self, queries: List[int], m: int) -> List[int]:
        retIndexlist = []
        p = [i+1 for i in range(m)]

        for q in queries: 
            # get index
            idx = p.index(q)
            retIndexlist.append(idx)
            # pop
            p.pop(idx)
            # move to front
            p.insert(0, q)
        
        return retIndexlist
class Solution:
    def processQueries(self, queries: List[int], m: int) -> List[int]:
        p = [i for i in range(1, m+1)]
        res = []
        for query in queries:
            index = p.index(query)
            res.append(index)
            p.remove(query)
            p = [query] + p
        return res

class Solution:
    def processQueries(self, queries: List[int], m: int) -> List[int]:
        out = []
        P = [i for i in range(1,m+1)]
        for i in queries:
            #num = P.index(i)
            out.append(P.index(i))
            P.insert(0, P.pop(P.index(i)))
        return out
from collections import deque
class Solution:
    def processQueries(self, queries: List[int], m: int) -> List[int]:
        p = deque(list(range(1,m+1)))
        result = []
        for q in queries:
            for i,n in enumerate(p):
                if q == n:
                    p.remove(n)
                    p.appendleft(n)
                    result.append(i)
                    break
        return result
        
                    
            

class Solution:
    def processQueries(self, queries: List[int], m: int) -> List[int]:
        p = [i for i in range(1,m+1)]

        def find(p,q):
            for i in range(len(p)):
                if p[i] == q:
                    return i

        def update(p,i):
            return [p[i]] + p[:i] + p[i+1:]

        res = []

        for i in range(len(queries)):
            q = find(p,queries[i])
            res.append(q)
            p = update(p,q)

        return res

class Solution:
    def processQueries(self, queries: List[int], m: int) -> List[int]:
        permu = list(range(1, m + 1))
        
        pos = []
        
        for i in queries:
            p = permu.index(i)
            pos.append(p)
            for j in range(p - 1, -1, -1):
                permu[j + 1] = permu[j]
            
            permu[0] = i
        
        return pos

class Solution:
    def processQueries(self, queries: List[int], m: int) -> List[int]:
        q=deque([])
        ans=[]
        for i in range(1,m+1,1):
            q.append(i)
            
        for j in queries:
            ans.append(q.index(j))
            q.remove(j)
            q.appendleft(j)
        
        return ans
class Solution:
    def processQueries(self, queries: List[int], m: int) -> List[int]:
        p = []
        res = []
        for i in range(1, m + 1):
            p.append(i)
        for n in queries:
            pos = 0
            while p[pos] != n and pos < m:
                pos += 1
            res.append(pos)
            del p[pos]
            p.insert(0, n)
        return res
from collections import OrderedDict
class Solution:
    def processQueries(self, queries: List[int], m: int) -> List[int]:
        ordered =   OrderedDict.fromkeys(list(range(1,m+1)))
        result  =   []
        for query in queries:
            idx     =   0
            for key in ordered:
                if key  ==  query:
                    break
                idx     +=  1
            result.append(idx)
            ordered.move_to_end(query,last=False)
        return result
            
            
       
            
            

class Solution:
    def processQueries(self, queries: List[int], m: int) -> List[int]:
        a=[i for i in range(1,m+1)]
        x={}
        for i in range(len(a)):
            x[a[i]]=i   
        b=[]
        for i in queries:
            b.append(x[i])
            a=[i]+a[:x[i]]+a[x[i]+1:]
            for j in range(x[i]+1):
                x[a[j]]=j
            
        return b  
            
            
            
        
                
        

class Solution:
    def processQueries(self, queries: List[int], m: int) -> List[int]:
        res = []
        arr = [i for i in range(1,m+1)]
        for q in queries:
            idx = arr.index(q)
            res.append(idx)
            arr.insert(0, arr.pop(idx))
        
        return res

class Solution:
    def processQueries(self, queries: List[int], m: int) -> List[int]:
        ans = []
        arr = list(range(1,m+1))
        
        for j in queries:
            for i in range(m):
                if j==arr[i]:
                    ans.append(i)
                    x = arr.pop(i)
                    arr.insert(0, x)
        
        return ans
                
            

class linkedNode:
    
    def __init__(self, val):
        self.val = val
        self.prev = None
        self.next = None

class Solution:
    def processQueries(self, queries: List[int], m: int) -> List[int]:
        head = linkedNode(-1)
        pointer = head
        
        for i in range(1, m+1):
            newLN = linkedNode(i)
            newLN.prev = pointer
            pointer.next = newLN
            pointer = pointer.next
        pointer.next = linkedNode(-1)
        pointer.next.prev = pointer
        
        ans = []
        for query in queries:
            i = 0
            pointer = head.next
            while pointer.val != query:
                pointer = pointer.next
                i += 1
            ans.append(i)
            
            pointer.prev.next = pointer.next
            pointer.next.prev = pointer.prev
            
            pointer.next = head.next
            head.next.prev = pointer
            head.next = pointer
            pointer.prev = head
            
        return ans
class Solution:
    def processQueries(self, queries: List[int], m: int) -> List[int]:
        rtnlst = []
        P = []
        for n in range (1, m +1):
            P.append(n)
        for q in queries: 
            for p in range(0,len(P)):
                if (P[p] == q):
                    rtnlst.append(p)
            
            P.pop(rtnlst[-1])
            P = [q] + P
           
        return rtnlst
class Node:
    def __init__(self, val):
        self.val = val
        self.next = None
        
class Linked:
    def __init__(self, m):
        self.head = Node(1)
        cur = self.head
        for n in range(2, m+1):
            cur.next = Node(n)
            cur = cur.next
    
    def print_me(self):
        cur = self.head 
        while cur:
            print(cur.val)
            cur = cur.next
        
    def move_to_front(self, val):
        if(self.head.val == val):
            return 0
        cur = self.head
        i = 0
        #print(val)
        while cur.val != val:
            prev = cur
            cur = cur.next
            i += 1
        prev.next = cur.next
        cur.next = self.head
        self.head = cur
        return i

class Solution:
    def processQueries(self, queries: List[int], m: int) -> List[int]:
        linked = Linked(m)
        #linked.print_me()
        
        result = []
        for num in queries:
            result.append(linked.move_to_front(num))
            #linked.print_me()
        return result
class Solution:
    def processQueries(self, queries: List[int], m: int) -> List[int]:
        P=[i+1 for i in range(m)]
        res=[]
        for query in queries:
            idx=P.index(query)
            res.append(idx)
            pos=idx
            while(pos>0):
                P[pos]=P[pos-1]
                pos-=1
            P[0]=query
        return res
            
            

class Solution:
    def processQueries(self, queries: List[int], m: int) -> List[int]:
        result = []
        result += [queries[0] -1]
        for ind, q in enumerate(queries[1:]):
            ind = ind+1
            if q > max(queries[:ind]): # nothing higher than this element has moved
                result +=[q-1]
            else: 
                equal_q = [i for i in range(ind) if q == queries[i]]
                if len(equal_q) >0:
                    diff = len(list(set(queries[equal_q[-1]+1:ind])))
                    result += [diff]
                else: #sum movement of all elements that are higher. 
                    sum_higher = len([x for x in list(set(queries[:ind])) if x> q])
                    result +=[q+sum_higher -1]
        return result
    

class Solution:
    def processQueries(self, queries: List[int], m: int) -> List[int]:
        l = list(range(1,m+1))
        n = 0
        p = [0]*len(queries)
        while n != len(queries):
            for k in range(len(queries)):
                for i in range(len(l)):
                    if l[i] == queries[k]:
                        l.insert(0,l.pop(i))
                        p[n] = i
                        n += 1
        return p
class Solution:
    def processQueries(self, queries: List[int], m: int) -> List[int]:
        l = list(range(1,m+1))
        n = 0
        p = [0]*len(queries)
        for k in range(len(queries)):
            for i in range(len(l)):
                if l[i] == queries[k]:
                    l.insert(0,l.pop(i))
                    p[n] = i
                    n += 1
        return p
class Solution:
    def processQueries(self, queries: List[int], m: int) -> List[int]:
        if m == 1:
            return [0 for _ in range(len(queries))]
        p = [i+1 for i in range(m)]
        res = []
        for i in queries:
            prev = p[0]
            if prev == i:
                res.append(0)
            else:
                j = 1
                while j < m:
                    tmp = p[j]
                    p[j] = prev
                    if tmp == i:
                        p[0] = tmp
                        res.append(j)
                        break
                    prev = tmp    
                    j += 1
        return res
class Solution:
    def processQueries(self,queries,m):
        P = list(range(1,m+1))
        return [n for x in queries if not P.insert(0,P.pop((n := P.index(x))))]
class Solution:
    def processQueries(self, queries: List[int], m: int) -> List[int]:
        result = []
        coord = [x for x in range(0, m)]
        
        for num in queries:
            
            before = coord[num - 1]
            result.append(before)
            
            # after   
            coord = [x+1 if x < before else x for x in coord]
            coord[num - 1] = 0
                
        return result
            

class Solution:
    def processQueries(self, queries: List[int], m: int) -> List[int]:
        result = []
        coord = [x for x in range(0, m)]
        
        for num in queries:
            
            before = coord[num - 1]
            result.append(before)
            
            # after   
            coord = [x+1 if x < before else x for x in coord]
            coord[num - 1] = 0
                
        return result

class Solution:
    def processQueries(self, queries: List[int], m: int) -> List[int]:
        
        pivot = []
        for q in queries:
            pos = q-1
            for piv in pivot:
                pos = self.getpos(pos, piv)
            pivot.append(pos)
            
        return pivot
        
    def getpos(self, pos, piv):
        if pos > piv:
            return pos
        elif pos == piv:
            return 0
        else:
            return pos+1
        
        

class Solution:
    def processQueries(self, queries: List[int], m: int) -> List[int]:
        
        perm=[i+1 for i in range(m)]
        out=[]
        
        for i in queries:
            
            idx=perm.index(i)
            
            out.append(idx)
            
            temp=perm[idx]
            
            while(idx>0):
                perm[idx]=perm[idx-1]
                idx-=1
                
            perm[idx]=temp
            
        return out
class Solution:
    def processQueries(self, queries: List[int], m: int) -> List[int]:
        result = []
        perm = [ i for i in range(1,m+1)]
        for element in queries:
            ind = perm.index(element)
            result.append(ind)
            tmp = [element]
            tmp.extend(perm[:ind])
            tmp.extend(perm[ind+1:])
            perm = tmp
            print(perm)
            
        return result
class Solution:
    def processQueries(self, queries: List[int], m: int) -> List[int]:
        
        result = []
        p = [i for i in range(1, m + 1)]
        
        for query in queries:       
            for index, value in enumerate(p): 
                if value == query:
                    result.append(index)
                    # p[0] = value
                    
                    depCounter = index
                    while depCounter >  0 :
                        p[depCounter] = p[depCounter - 1]
                        depCounter -= 1

                    p[0] =value
                    break
        return result
class Solution:
    def processQueries(self, queries: List[int], m: int) -> List[int]:
        
        result = []
        p = [i for i in range(1, m + 1)]
        
        
        for query in queries:
            for index, value in enumerate(p): 
                if value == query:
                    result.append(index)
                    # p[0] = value
                    
                    depCounter = index
                    while depCounter >  0 :
                        p[depCounter] = p[depCounter - 1]
                        depCounter -= 1

                    p[0] =value
                    break
        return result
class Node:
    def __init__(self, val = None):
        self.val = val
        self.next = None
class Solution:
    def processQueries(self, queries: List[int], m: int) -> List[int]:
        root = Node(0)
        cur = root
        for i in range(1, m+1):
            newNode = Node(i)
            cur.next = newNode
            cur = newNode
        
        res = []
        for i in range(len(queries)):
            targetVal = queries[i]
            cur = root
            position = 0
            while cur.next is not None:
                if cur.next.val == targetVal:
                    res.append(position)
                    
                    # add at the beginning
                    temp = cur.next
                    cur.next = temp.next
                    temp.next = root.next
                    root.next = temp
                    break
                else:
                    cur = cur.next
                    position += 1
        return res
class Solution:
    def processQueries(self, queries: List[int], m: int) -> List[int]:
        p = [i for i in range(1, m+1)]
        res = []
        for q in queries:
            for i in range(len(p)):
                if p[i] == q:
                    res.append(i)
                    p.remove(q)
                    p.insert(0, q)
        return res
class Solution:
    def processQueries(self, queries: List[int], m: int) -> List[int]:
        p = []
        res = []
        for x in range(m):
            p.append(x+1)
        for x in queries:
            idx = p.index(x)
            res.append(idx)
            p.insert(0, p.pop(idx))
            
        return res
import bisect

class Solution:
    def processQueries(self, queries: List[int], m: int) -> List[int]:
        perm = list(range(1, m+1))
        res = []
        for e in queries:
            idx = perm.index(e)
            res.append(idx)
            perm = [perm[idx]] + [r for i, r in enumerate(perm) if i != idx]
        return res
class Solution:
    def processQueries(self, queries: List[int], m: int) -> List[int]:
        l=[i for i in range(1,m+1)]
        ans=[]
        for q in queries :
            d={}
            for index,element in enumerate(l):
                d[element]=index
            ans.append(d[q])
            x=l.pop(d[q])
            l=[x]+l
        return ans
class Solution:
    def processQueries(self, queries: List[int], m: int) -> List[int]:
        d={i:i-1 for i in range(1,m+1)}
        ans=[]
        for i in queries:
            x=d[i]
            ans.append(x)
            
            for j in list(d.keys()):
                if d[j]<x:
                    d[j]+=1
            d[i]=0
        
        return ans
            
                    
             
            
        

class Solution:
    def processQueries(self, queries: List[int], m: int) -> List[int]:
        permutations = [permutation for permutation in range(1, m + 1)]

        result = []
        for qindex, query in enumerate(queries):
            for pindex, permutation in enumerate(permutations):
                if permutation == query:
                    result.append(pindex)
                    permutations = [permutation] + permutations[0:pindex] + permutations[pindex + 1:]
                    print(permutations)
                    break
        print(result)
        print(permutations)
        
        return result
class Solution:
    def processQueries(self, queries: List[int], m: int) -> List[int]:
        d = {k : k+1 for k in range(m)}
        result = []
        for q in queries:
            current_pos = self.findPos(d, q)
            result.append(current_pos)
            while current_pos > 0:
                d[current_pos] = d[current_pos-1]
                current_pos -= 1
            d[0] = q
        return result
    
    def findPos(self, d, q):
        for idx, val in list(d.items()):
            if val == q:
                return idx

class Solution:
    def processQueries(self, queries: List[int], m: int) -> List[int]:
        cache = {x:x-1 for x in range(1,m+1)}
        n = len(queries)
        ans = [0]*n
        
        for i in range(n):
            curr = queries[i]
            ans[i] = cache[curr]
            pos = cache[curr]
            
            for j in cache.keys():
                if cache[j]<pos:
                    cache[j]+=1
            cache[curr]=0
        return ans
class Solution:
    def processQueries(self, queries: List[int], m: int) -> List[int]:
        permutation = {num: num-1 for num in range(1, m+1)}
        ans = []
        for query in queries:
            ans.append(permutation[query])
            pos = permutation[query]
            for key in permutation.keys():
                if permutation[key] < pos:
                    permutation[key] += 1
            permutation[query] = 0
        return ans
class Solution:
    def processQueries(self, queries: List[int], m: int) -> List[int]:
        data = [i for i in range(1, m+1)]
        hashMap = {}
        for i in range(1, m+1):
            hashMap[i] = i - 1
        result = []
        for q in queries:
            position = hashMap[q]
            result.append(position)
            data = [data[position]] + data[0:position] + data[position+1:]
            for (index, d) in enumerate(data):
                hashMap[d] = index
        
        return result

class Solution:
    def processQueries(self, queries: List[int], m: int) -> List[int]:
        
        dic = {}
        for i in range(1, m + 1):
            dic[i] = i - 1
        ans = []
        leng = 0
        for query in queries:
            index = dic[query]
            ans.append(index)
            
            for k, v in list(dic.items()):
                
                if v < index:
                    dic[k] += 1
                    leng += 1
                
            dic[query] = 0
        return ans

class Solution:
    def processQueries(self, queries: List[int], m: int) -> List[int]:
        current = list(range(m, 1 - 1, -1))
        
        ans = list()
        for query in queries:
            # O(n)
            ind = m - 1 - current.index(query)
            
            # O(n)
            current.remove(query)
            # O(1)
            current.append(query)
    
            ans.append(ind)

        return ans

class Solution:
    def processQueries(self, queries: List[int], m: int) -> List[int]:
        result = list(range(1,m+1))
        temp = 0
        i = 0
        real = []
        for i in range(len(queries)):
            for j in range(len(result)):
                if(queries[i] == result[j]):
                    real.append(j)
                    #print('position', j)
                    result.pop(j)
                    result.insert(0, queries[i])
                    #print(result)
                    continue
        return real
            

class Solution:
    def processQueries(self, queries: List[int], m: int) -> List[int]:
        P=deque([i for i in range(1,m+1)])
        ans=[]
        for q in queries:
            new_P = deque()
            for index, p in enumerate(P):
                if p != q:
                    new_P.append(p)
                else:
                    ans.append(index)
            new_P.appendleft(q)
            P = new_P
        return ans
class Solution:
    def processQueries(self, queries: List[int], m: int) -> List[int]:
        
        array = []
        
        element2index = {}
        index2element = {}
        
        for i in range(m):
            element2index[i+1] = i
            index2element[i] = i+1
        
        for i in range(len(queries)):
            
            q = queries[i]
            
            pos = element2index[q]
            
            array.append(pos)
            
            for k in range(pos-1, -1, -1):
                
                e = index2element[k]
                element2index[e] += 1
                index2element[element2index[e]] = e
                
            index2element[0] = q
            element2index[q] = 0
            
            
        return array
            
            

def shiftRight(arr,i):
    
    ele = arr[i]
    for j in range(i,0,-1):
        # print(arr[j],arr[j-1],i)
        arr[j] = arr[j-1]
    arr[0] = ele
    return arr

def findQ(arr,ele):
    for i in range(len(arr)):
        if(arr[i] == ele):
            return i
    
        

class Solution:
    def processQueries(self, queries: List[int], m: int) -> List[int]:
        res = []
        arr = list(range(1,m+1))
        for i in range(len(queries)):
            q = queries[i]
            j = findQ(arr,q)
            res.append(j)
            shiftRight(arr,j)
        return res
        

class Solution:
    def processQueries(self, queries: List[int], m: int) -> List[int]:
        class Node:
            def __init__(self,val,nextNode=None):
                self.val=val
                self.next=nextNode
        dummy=Node(-1)
        res={}
        out=[]
        cur=dummy
        
        for i in range(m):
            cur.next=Node(i+1)
            res[i+1]=[i,cur.__next__]
            cur=cur.__next__
        cur=dummy
        for query in queries:
            out.append(res[query][0])
            if not res[query][0]:
                continue
            cur=dummy.__next__
            while cur and cur.val!=query:
                # print(res[cur.val])
                res[cur.val][0]+=1
                prev=cur
                cur=cur.__next__
            prev.next=cur.__next__
            cur.next=dummy.__next__
            dummy.next=cur
            res[cur.val][0]=0
        return out

class Solution:
    def processQueries(self, queries: List[int], m: int) -> List[int]:
        p = [i for i in range(1,m+1)]
        res =[]
        for i in range(len(queries)):
            for j in range(m):
                if p[j] == queries[i]:
                    res.append(j)
                    p = [p[j]] + p[0:j]+p[j+1:]
        return res
class Solution:
    def processQueries(self, queries: List[int], m: int) -> List[int]:
        lookup = {key: key-1 for key in range(1,m+1)}
        ans = []
        for query in queries:
            temp = lookup[query]
            ans.append(temp)
            for key, val in lookup.items():
                
                if key == query:
                    lookup[key] = 0
                elif val < temp:
                    lookup[key] += 1
        
        return ans
class Solution:
    def processQueries(self, queries: List[int], m: int) -> List[int]:
        P = list(range(1, m+1))
        op = []
        hashP = {}
        for i,p in enumerate(P):
            hashP[p] = i
        for query in queries:
            idx = hashP[query]
            if idx != 0:
                for key in hashP:
                    if key == query:
                        hashP[key] = 0
                    elif hashP[key] < idx:
                        hashP[key] += 1
            op.append(idx)
        return op
        

class Solution:
    def processQueries(self, queries: List[int], m: int) -> List[int]:
        arr= [i-1 for i in range(0,m+1)]
        res= []
        for i in range(0,len(queries)):
            res.append(arr[queries[i]])
            for j in range(1,m+1):
                if(arr[j]<arr[queries[i]]):
                    arr[j]+= 1
            arr[queries[i]]= 0
        return res

class Solution:
    def processQueries(self, queries: List[int], m: int) -> List[int]:
        
        ans=[]
        P=[]
        
        for i in range(m):
            P.append(i+1)
        
        for i in range(len(queries)):
            a=P.index(queries[i])
            ans.append(a)
            b=P.pop(a)
            P=[b]+P
        
        return ans
                        
            

class Solution:
    def processQueries(self, queries: List[int], m: int) -> List[int]:
        
        P = [i for i in range (1,m+1)]
        result = []
        
        for q in queries:
            # a = P[q-1]
            result.append(P.index(q))
            P.remove(q)
            P.insert(0, q)
        return result
class Solution:
    def processQueries(self, queries: List[int], m: int) -> List[int]:
        p = [i for i in range(1, m+1)]
        res = []
        for j in range(len(queries)):
            idx = p.index(queries[j])
            fs = p[0:idx]
            ls = p[idx+1:]
            p = [p[idx]] + fs + ls
            res.append(idx)
        return res
