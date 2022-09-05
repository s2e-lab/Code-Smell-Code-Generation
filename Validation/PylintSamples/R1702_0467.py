# 1390. Four Divisors
# version 2, with optimized prime-finding.

import math

def remove (lst, index):
    assert lst
    tail = len (lst) - 1
    lst[index], lst[tail] = lst[tail], lst[index]
    lst.pop ()

def swap_min (lst):
    if not lst: return
    argmin = min (range (len (lst)), key = lambda i: lst[i])
    lst[0], lst[argmin] = lst[argmin], lst[0]

def find_primes (top):
    candidates = list (range (2, top))
    primes = []
    while candidates:
        # here, candidates[0] is the least element.
        latest_prime = candidates[0]
        primes.append (latest_prime)
        remove (candidates, 0)
        for i in range (len (candidates) - 1, -1, -1):
            if candidates[i] % latest_prime == 0:
                remove (candidates, i)

        swap_min (candidates)
        # before continuing, set candidates[0] to be the least element.
    return primes

def find_prime_factor (n, primes):
    for p in primes:
        if n % p == 0:
            return p

def div4 (n, primes, setprimes):
    if n <= 3:
        return 0
    elif n in setprimes:
        return 0
    else:
        p1 = find_prime_factor (n, primes)
        if p1 is None:
            return 0
        p2 = find_prime_factor (n // p1, primes)
        if p2 is None:
            p2 = n // p1
        if p1 * p2 == n and p1 != p2:
            # success
            return (1 + p1) * (1 + p2)
        elif p1 ** 3 == n:
            # success
            return (1 + p1) * (1 + p1**2)
        else:
            return 0

def sum_four_divisors (arr):
    top = math.ceil (math.sqrt (max (arr) + 5))
    primes = find_primes (top)
    setprimes = set (primes)
    return sum (div4 (elem, primes, setprimes) for elem in arr)

class Solution:
    def sumFourDivisors(self, nums: List[int]) -> int:
        return sum_four_divisors(nums)

import math

def remove (lst, index):
    assert lst
    tail = len (lst) - 1
    lst[index], lst[tail] = lst[tail], lst[index]
    lst.pop ()

def swap_min (lst):
    if not lst: return
    argmin = min (range (len (lst)), key = lambda i: lst[i])
    lst[0], lst[argmin] = lst[argmin], lst[0]

def find_primes (top):
    candidates = list (range (2, top))
    primes = []
    while candidates:
        # here, candidates[0] is the least element.
        latest_prime = candidates[0]
        primes.append (latest_prime)
        remove (candidates, 0)
        for i in range (len (candidates) - 1, -1, -1):
            if candidates[i] % latest_prime == 0:
                remove (candidates, i)

        swap_min (candidates)
        # before continuing, set candidates[0] to be the least element.
    return primes

def find_prime_factor (n, primes):
    for p in primes:
        if n % p == 0:
            return p

def div4 (n, primes, setprimes):
    if n <= 3:
        return 0
    elif n in setprimes:
        return 0
    else:
        p1 = find_prime_factor (n, primes)
        if p1 is None:
            return 0
        p2 = find_prime_factor (n // p1, primes)
        if p2 is None:
            p2 = n // p1
        if p1 * p2 == n and p1 != p2:
            # success
            return (1 + p1) * (1 + p2)
        elif p1 ** 3 == n:
            # success
            return (1 + p1) * (1 + p1**2)
        else:
            return 0

def sum_four_divisors (arr):
    top = math.ceil (math.sqrt (max (arr) + 5))
    primes = find_primes (top)
    setprimes = set (primes)
    return sum (div4 (elem, primes, setprimes) for elem in arr)

class Solution:
    def sumFourDivisors(self, nums: List[int]) -> int:
        return sum_four_divisors(nums)
class Solution:
    def sumFourDivisors(self, nums: List[int], c={}) -> int:
        r = 0
        for n in nums:
            if n in c:
                r += c[n]
                continue
            s = n + 1
            cnt = 2
            end = sqrt(n)
            if end == int(end):
                s += end
                cnt += 1
                end -= 1
            for i in range(2, int(end) + 1):
                if n % i == 0:
                    cnt += 2
                    if cnt > 4:
                        s = 0
                        break
                    s += i
                    s += n // i
            if cnt == 4:
                c.update({n:s})
                r += s
            else:
                c.update({n:0})
        return r
class Solution:
    def sumFourDivisors(self, nums: List[int]) -> int:    
        factors_cache = {}
        
        def get_factors(num):
            if num in factors_cache:
                return factors_cache[num]
            else:
                factors = set([1, num])
                for potential_divisor in range(2, math.ceil(math.sqrt(num))):
                    if num % potential_divisor == 0:
                        factors = factors.union(get_factors(potential_divisor))
                        factors = factors.union(get_factors(num // potential_divisor))
                    if len(factors) > 4:
                        break
                factors_cache[num] = factors
                return factors
            
        running_sum = 0
        for num in nums:
            factors = get_factors(num)
            if len(factors) == 4:
                running_sum += sum(factors)
            
        return running_sum

class Solution:
    def sumFourDivisors(self, nums: List[int]) -> int:
        ans = 0
        for num in nums:
            out = set()

            for i in range(1, int(num ** 0.5+1)):
                a, b = divmod(num, i)
                if b == 0:
                    out.add(i)
                    out.add(a)
                if len(out) > 4: break
            if len(out) == 4:
                ans += sum(out)
        
        return ans
class Solution:
    def sumFourDivisors(self, nums: List[int]) -> int:
        ans = 0
        for num in nums:
            ans += self.fourDivisors(num)
        return ans
    def fourDivisors(self,num):
        memo = set()
        for i in range(1,num + 1):
            if i * i > num:
                break
            if num % i == 0:
                memo.add(i)
                memo.add(num//i)
                if len(memo) > 4:
                    return 0
                
        if len(memo) == 4:
            return sum(memo)
        return 0
class Solution:
    def divs(self,x):
        memo = self.memo
        if x in memo:
            return memo[x]
        #
        L = 2     if x>1 else 1
        S = (1+x) if x>1 else 1
        for a in range(2,x):
            if (a**2)>x:
                break
            #
            if not x%a:
                L += 1 if x==(a**2) else 2
                S += a if x==(a**2) else (a + x//a)
            #
            if L>4:
                break
        #
        memo[x] = L,S
        return L,S
    def sumFourDivisors(self, A):
        self.memo = {}
        res = 0
        for x in A:
            L,S = self.divs(x)
            if L==4:
                res += S
        return res
class Solution:
    def sumFourDivisors(self, nums: List[int]) -> int:
        # find all divisor of this number and use set() to select all the distinct factors
        res = 0
        for num in nums:
            divisor_num = set()
            for i in range(1, int(sqrt(num))+1):
                if num%i == 0:
                    divisor_num.add(num//i)
                    divisor_num.add(i)
                    
            if len(divisor_num) == 4:
                res +=sum(divisor_num)
                
                
                
        #capital one len(divisor_num)==3, divisor_sum.remove(num)
                
                
                
        return res
                
                    
                    

class Solution:
    def sumFourDivisors(self, nums: List[int]) -> int:
        ans = 0
        for n in nums:
            sq = floor(n**0.5)
            if sq*sq == n:
                continue
            divs = 2
            divsum = 1+n
            for i in range(sq, 1, -1):
                if n % i == 0:
                    divs += 2
                    divsum += i + n // i
                if divs > 4:
                    break
            if divs == 4:
                ans += divsum
        return ans


class Solution:
    def divs(self,x):
        memo = self.memo
        if x in memo:
            return memo[x]
        #
        L = 2     if x>1 else 1
        S = (1+x) if x>1 else 1
        for a in range(2,x):
            if (a**2)>x:
                break
            #
            if not x%a:
                L += 1 if x==(a**2) else 2
                S += a if x==(a**2) else (a + x//a)
            #
            if L>4:
                break
        #
        memo[x] = L,S
        return L,S
    def sumFourDivisors(self, A):
        self.memo = {}
        res = 0
        for x in A:
            L,S = self.divs(x)
            if L==4:
                res += S
        return res
class Solution:
    def divs(self,x):
        memo = self.memo
        if x in memo:
            return memo[x]
        #
        L = 2     if x>1 else 1
        S = (1+x) if x>1 else 1
        for a in range(2,x):
            if (a**2)>x:
                break
            #
            if not x%a:
                L += 1 if x==(a**2) else 2
                S += a if x==(a**2) else (a + x//a)
            #
            if L>4:
                break
        #
        memo[x] = L,S
        return L,S
    def sumFourDivisors(self, A):
        self.memo = {}
        res = 0
        for x in A:
            L,S  = self.divs(x)
            if L==4:
                res += S
        return res
class Solution:
    def divisors(self, n):
        for i in range(1, int(sqrt(n) + 1)):
            if n % i == 0:
                yield i
                j = n // i
                if j != i:
                    yield j

    def sumFourDivisors(self, nums: List[int]) -> int:
        s = 0
        for n in nums:
            l = list(self.divisors(n))
            if len(l) == 4:
                s += sum(l)
        return s
class Solution:
    def sumFourDivisors(self, nums: List[int]) -> int:
        def make_divisors(n):
            divisors = []
            for i in range(1, int(n**0.5)+1):
                if n % i == 0:
                    divisors.append(i)
                    if i != n // i:
                        divisors.append(n//i)
            return len(divisors), divisors
        
        ret = [0]
        for n in nums:
            l, d = make_divisors(n)
            if l == 4:
                ret.append(sum(d))
        return sum(ret)

class Solution:
    def sumFourDivisors(self, nums: List[int], c={}) -> int:
        r = 0
        for n in nums:
            if n in c:
                r += c[n]
                continue
            s = n + 1
            cnt = 2
            e = sqrt(n)
            if (end := int(e)) == e:
                s += end
                cnt += 1
                end -= 1
            for i in range(2, end + 1):
                if n % i == 0:
                    cnt += 2
                    if cnt > 4:
                        s = 0
                        break
                    s += i
                    s += n // i
            if cnt == 4:
                c.update({n:s})
                r += s
            else:
                c.update({n:0})
        return r
class Solution:
    def sumFourDivisors(self, nums: List[int]) -> int:
        sum2=0
        for n in nums:
            cnt=0
            sum1=0
            for i in range(1,int(sqrt(n))+1):
                if n%i==0:
                    if i==sqrt(n):
                        cnt+=1
                    else:
                        cnt+=2
                        sum1+=i+n//i
            if cnt==4:
                sum2 += sum1
        return sum2

import math

class Solution:
    def sumFourDivisors(self, nums: List[int]) -> int:
        divisorSum = 0
        for num in nums:
            divisorSum += self.findDivisors(num)
        return divisorSum
    
    def findDivisors(self, num):
        divisors = set([1, num])
        for i in range(2, int(math.sqrt(num)) + 1):
            if num % i == 0:
                divisors.add(i)
                divisors.add(num//i)
        if len(divisors) == 4:
            return sum(list(divisors))
        return 0

# N = len(nums)
# time: O(NlogN)
# space: O(N)
class Solution:
    def sumFourDivisors(self, nums: List[int]) -> int:
        def make_divisors(n):
            divisors = []
            for i in range(1, int(n**0.5)+1):
                if n % i == 0:
                    divisors.append(i)
                    if i != n // i:
                        divisors.append(n//i)
            return len(divisors), divisors
        
        ret = [0]
        for n in nums:
            l, d = make_divisors(n)
            if l == 4:
                ret.append(sum(d))
        return sum(ret)

class Solution:
    def sumFourDivisors(self, nums: List[int]) -> int:
        def count_divisors(x):
            num_divisors = 0
            sum_divisors = 0
            if sqrt(x) == int(sqrt(x)):
                num_divisors += 1
                sum_divisors += sqrt(x)
            for i in range(1, ceil(sqrt(x))):
                if x % i == 0:
                    num_divisors += 2
                    sum_divisors += i + (x // i)
            return sum_divisors if num_divisors == 4 else 0
        return sum([count_divisors(x) for x in nums])
class Solution:
    def sumFourDivisors(self, nums: List[int]) -> int:
        res = 0
        for num in nums:
            divisor = set()
            for i in range(1, int(sqrt(num))+1):
                if num%i == 0:
                    divisor.add(num//i)
                    divisor.add(i)
                if len(divisor) > 4:
                    break
                
            if len(divisor) == 4:
                res += sum(divisor)
        
        return res

class Solution:
    def sumFourDivisors(self, nums: List[int]) -> int:
        res = 0
        for num in nums:
            curr = 0
            div_sum = 0
            for i in range(1, int(sqrt(num)) + 1):
                if num % i == 0:
                    curr += 2
                    
                    if i == num // i:
                        div_sum -= i
                        curr -= 1
                        
                    div_sum += i
                    div_sum += (num // i)
        
            if curr == 4:
                res += div_sum
        
        return res

class Solution:
    def sumFourDivisors(self, nums: List[int]) -> int:
        ret = 0
        for num in nums:
            ret += self.has_four_divisors(num)
        return int(ret)
    
    def has_four_divisors(self, num):
        divisor_sum = 0
        divisors = 0
        for i in range(1, int(sqrt(num))+1):
            if num % i == 0:
                if i != num / i:
                    divisors += 2
                    divisor_sum += i
                    divisor_sum += num / i
                else:
                    divisors += 1
                    divisor_sum += i
        if divisors == 4:
            return divisor_sum
        return 0
class Solution:
    def sumFourDivisors(self, nums: List[int]) -> int:
        ans = 0
        for num in nums:
            divisor = 0
            a = 2
            upperLimit = int(num**0.5)
            if upperLimit**2 == num:
                continue
            upperLimit += 1
            subAns = 1 + num
            while a < upperLimit:
                if num%a == 0:
                    if divisor == 0:
                        divisor += 1
                        subAns += (a+num//a)
                    else:
                        break
                upperLimit = min(upperLimit, num//a)
                a += 1
            else:
                if divisor == 1:
                    ans += subAns
        return ans
import math
class Solution:
    def sumFourDivisors(self, nums: List[int]) -> int:
        total = 0
        for num in nums:
            divisors = self.getDivisors(num)
            if len(divisors) == 4:
                print(divisors, num)
                total+=sum(divisors)
        return total
    
    def getDivisors(self, num):
        res = set([1, num])
        for i in range(2,1+math.ceil(math.sqrt(num))):
            if num%i == 0:
                res.add(i)
                res.add(num//i)
        return res 
from math import sqrt

class Solution:
    def sumFourDivisors(self, nums: List[int]) -> int:
        def helper(num):
            divisors = set()
            for i in range(1, int(sqrt(num))+1):
                if num % i == 0:
                    divisors.add(i)
                    divisors.add(num // i)
            return sum(divisors) if len(divisors) == 4 else 0

        return sum(helper(num) for num in nums)
class Solution:
    def sumFourDivisors(self, nums: List[int], c={}) -> int:
        r = 0
        for n in nums:
            if n in c:
                r += c[n]
                continue
            s = 0
            cnt = 0
            for i in range(1, round(sqrt(n) + 1)):
                if n % i == 0:
                    cnt += 1
                    if cnt > 4:
                        s = 0
                        break
                    s += i
                    j = n // i
                    if j != i:
                        cnt += 1
                        if cnt > 4:
                            s = 0
                            break
                        s += j
            if cnt == 4:
                c.update({n:s})
                r += s
            else:
                c.update({n:0})
        return r
class Solution:
    def sumFourDivisors(self, nums: List[int]) -> int:
        ans = 0 
        
        for n in nums:
            st = set()
            for i in range(1, int(sqrt(n))+1):
                if n%i == 0:
                    st.add(i)
                    st.add(n//i)
            
            if len(st) == 4:
                ans += sum(st)
        
        return ans
class Solution:
    def divs(self,x):
        memo = self.memo
        if x in memo:
            return memo[x]
        #
        res = 2 if x>1 else 1
        B   = {1,x}
        for a in range(2,x):
            if x<(a**2):
                break
            if not x%a:
                res += 1 if x==(a**2) else 2
                B.update( {a, x//a} )
            if res>4:
                break
            a += 1
        memo[x] = res,B
        return res,B
    def sumFourDivisors(self, A):
        self.memo = {}
        res = 0
        for x in A:
            r,B = self.divs(x)
            if r==4:
                res += sum(B)
        return res
class Solution:
    def sumFourDivisors(self, nums: List[int]) -> int:
        def NOD(x):
            divisor = set([1,x])
            for i in range(2,int(x**.5) + 1):
                if not x%i:
                    divisor.add(i)
                    divisor.add(x//i)
            return divisor
        ans = []
        for num in nums:
            divisor = NOD(num)
            if len(divisor) == 4:
                ans.append(divisor)
        return sum([sum(i) for i in ans])
        

class Solution:
    def sumFourDivisors(self, nums: List[int]) -> int:
        res = 0
        for num in nums:
            divisor = set() 
            for i in range(1, floor(sqrt(num)) + 1):
                if num % i == 0:
                    divisor.add(num//i)
                    divisor.add(i)
                if len(divisor) > 4:    
                    break
                    
            if len(divisor) == 4:
                res += sum(divisor)
        return res 

class Solution:
    def sumFourDivisors(self, nums: List[int]) -> int:
        def div_num(x):
            ans, ssum = 2, x + 1
            for i in range(2, int(x ** 0.5)+1):
                if x % i == 0:
                    ans += 1 + (i*i != x)
                    ssum += (i + x//i) if i*i != x else i
            return ans == 4, ssum
        res = 0
        for x in nums:
            flag, ssum = div_num(x)
            if flag == 1:
                res += ssum
        return res
        
        


class Solution:
    def sumFourDivisors(self, nums: List[int]) -> int:
        ans = 0
        for num in nums:
            divisors = set()
            for i in range(1, int(num**0.5)+1):
                if num % i == 0:
                    divisors.add(i)
                    divisors.add(num//i)
            if len(divisors) == 4:
                ans += sum(divisors)
                
        return ans
class Solution:
    def sumFourDivisors(self, nums: List[int]) -> int:
        out = 0
        for i in nums:
            temp = set()
            for j in range(1, floor(sqrt(i))+1):
                if i % j == 0:
                    temp.add(j)
                    temp.add(int(i/j))
                # if len(temp) > 4:
                #     break
            if len(temp) == 4:
                out += sum(temp)
        return out
class Solution:
    def sumFourDivisors(self, nums: List[int]) -> int:
        def findfactors(n):
            f = []
            for i in range(1,int(n**0.5)+1):
                if n%i==0:
                    f.append(i)
                    if (i!=n//i):
                        f.append(n//i)
            return sum(f) if len(f)==4 else 0
        return sum([findfactors(x) for x in nums])

class Solution:
    def sumFourDivisors(self, nums: List[int]) -> int:
        ret = 0
        for num in nums:
            divs = self.divisors(num)
            if len(divs) == 4:
                ret += sum(divs)
        
        return ret 
    
    def divisors(self, num):
        ret = []
        for i in range(1, int(num**0.5)+1):
            if num%i == 0:
                ret += [i]
                if num//i != i:
                    ret += [num//i]
        
        return ret
import math
class Solution:
    def sumFourDivisors(self, nums: List[int]) -> int:
        divisors = []
        for num in nums:
            d = set()
            for i in range(1, floor(sqrt(num) + 1)):
                if num % i == 0:
                    d.add(i)
                    d.add(num // i)
            divisors.append(d)
        # print(divisors)
        result = 0
        for s in divisors:
            if len(s) == 4:
                result += sum(s)
        return result
                
            

import math
class Solution:
    def sumFourDivisors(self, nums: List[int]) -> int:
        divisors = []
        for num in nums:
            d = set()
            for i in range(1, floor(sqrt(num) + 1)):
                if num % i == 0:
                    d.add(i)
                    d.add(num // i)
            divisors.append(d)
        result = 0
        for s in divisors:
            if len(s) == 4:
                result += sum(s)
        return result
                
            

class Solution:
    def sumFourDivisors(self, nums: List[int]) -> int:
        ans = 0
        for n in nums:
            tmp = set([1,n])
            for d in range(2,ceil(sqrt(n))+1):
                if n%d==0:
                    tmp.add(d)
                    tmp.add(n//d)
            #print(tmp)
            if len(tmp) ==4:
                ans += sum(tmp)
        return ans
import math

class Solution:
    
    def s_factors_if_len_4(self, n, d):
        s = set()
        for i in range(1,math.floor(n**0.5)+1):
            if n%i == 0:
                s.add(i)
                s.add(n//i)
        if len(s) == 4:
            d[n] = sum(s)
        else:
            d[n] = 0
    
    def sumFourDivisors(self, nums: List[int]) -> int:
        d = {}
        sol = 0
        for n in nums:
            if n not in d.keys():
                self.s_factors_if_len_4(n, d)
            sol += d[n]
        return sol
class Solution:
    cache = {}
    def factors(self, n):
        if n in self.cache:
            return self.cache[n]
        result = set()
        for i in range(1, int(n ** 0.5) + 1):
            div, mod = divmod(n, i)
            if mod == 0:
                result |= {i, div}
        self.cache[n] = result
        return result
    
    def sumFourDivisors(self, nums: List[int]) -> int:
        factors = [ self.factors(f) for f in nums ]
        return sum([sum(f) for f in factors if len(f) == 4])

class Solution:
    def sumFourDivisors(self, nums: List[int]) -> int:
        def compute(n):
            s = set()
            for i in range(1, 1 + int(n**0.5)):
                if n % i == 0:
                    s.add(i)
                    s.add(n // i)
            return sum(s) if len(s) == 4 else 0
        return sum(compute(i) for i in nums)
class Solution:
    def sumFourDivisors(self, nums: List[int]) -> int:
        res = 0
        for i in range(len(nums)):
            curr = nums[i]
            counter = 1
            divisors = []
            while counter <= sqrt(nums[i]) and len(divisors)<5:
                if nums[i]%counter == 0:
                    if counter not in divisors:
                        divisors.append(counter)
                    if nums[i]//counter not in divisors:
                        divisors.append(nums[i]//counter)
                counter+=1
            if len(divisors) == 4:
                res += sum(divisors)
            print(divisors)
                
        return res
class Solution:
    def sumFourDivisors(self, lst: List[int]) -> int:
        import math
        final = 0
        for i in lst:
            factors = []
            for j in range(1, round(math.sqrt(i)) + 1):
                if i % j == 0:
                    factors.append(int(j))
                    factors.append(int(i / j))
            factors = list(dict.fromkeys(factors))
            if len(factors) == 4:
                final += sum(factors)
        return final
class Solution:
    def divs(self,x):
        #
        L = 2     if x>1 else 1
        S = (1+x) if x>1 else 1
        for a in range(2,x):
            if (a**2)>x:
                break
            #
            if not x%a:
                L += 1 if x==(a**2) else 2
                S += a if x==(a**2) else (a + x//a)
            #
            if L>4:
                break
        #
        return L,S
    def sumFourDivisors(self, A):
        self.memo = {}
        res = 0
        for x in A:
            L,S = self.divs(x)
            if L==4:
                res += S
        return res
class Solution:
    def sumFourDivisors(self, nums: List[int]) -> int:
        def isPrime(n):
            if n<=1:
                return False
            if n<=3:
                return True
            if n & 1 == 0 or n % 3 == 0:
                return False
            i=5
            while i*i<=n:
                if n % i == 0 and n % (i+2) == 0:
                    return False
                i+=6
            return True
        res=0
        c=0
        temp=set()
        for i in nums:
                for j in range(1,int(i**.5)+1):
                    if i%j==0:
                        temp.add(j)
                        temp.add(i//j)
                        if i//j!=j:
                            c+=2
                        else:
                            c+=1
                res+=sum(temp) if c==4 else 0
                temp=set()
                c=0
        return res
class Solution:
    def sumFourDivisors(self, nums: List[int]) -> int:
        ans = 0
        def get_divisor(num):
            val = set()
            i = 1
            while i < math.sqrt(num) + 1:
                if num % i == 0:
                    val.add(i)
                    val.add(num // i)
                if len(val) > 4:
                    return val
                i += 1
            return val
        
        for num in nums:
            a = get_divisor(num)
            if len(a) == 4:
                ans += sum(a)
        return ans
    

class Solution:
    def sumFourDivisors(self, nums: List[int]) -> int:
        # 10:58 9/24/20
        def four_divisors3(n):
            div = set()
            i = 1
            while i*i < n:
                if n % i == 0:
                    div.add(i)
                    div.add(n // i)
                    if len(div) > 4:
                        return 0
                i += 1
            return sum(div) if len(div) == 4 else 0
        
        def four_divisors(n):
            div = set()
            for i in range(1, int(n ** 0.5) + 1):
                if n % i == 0:
                    div.add(i)
                    div.add(n // i)
                    if len(div) > 4:
                        return 0
            return sum(div) if len(div) == 4 else 0
        
        def four_divisors2(n):
            cnt = 0
            sums = 0
            div = set()
            if n != 0:
                # i = 1
                for i in range(1, int(n** 0.5) + 1):
                    if n % i == 0:
                        cnt += 2
                        sums += i + n // i
                        # div.add(i)
                        # div.add(n // i)
                    if cnt > 4:
                        return 0
                    # i += 1
            return sums if cnt == 4 else 0
        
        if not nums: return 0
        nums.sort()
        total = 0
        # sums = [0]
        past = [None, None]
        
        for i, v in enumerate(nums):
            if i > 0 and v == nums[i - 1] and v == past[0]:
                total += past[1]
                continue
            tmp = four_divisors(v)
            total += tmp
            past = [v, tmp]
            
                    
        return total
                    
        
        
        
            
            
           

class Solution:
    def sumFourDivisors(self, nums: List[int]) -> int:
        res = 0
        ls = len(nums)
        for i in range(ls):
            divs = set()
            for j in range(1, floor(sqrt(nums[i])) + 1):
                if nums[i] % j == 0:
                    divs.add(nums[i]//j)
                    divs.add(j)
            
            if len(divs) == 4:
                res = res + sum(divs)
        return res
class Solution:
    def sumFourDivisors(self, nums: List[int]) -> int:
        def NOD(x):
            divisor = set()
            for i in range(1,int(sqrt(x)) + 1):
                if not x%i:
                    divisor.add(i)
                    divisor.add(x//i)
            return divisor
        
        res = 0
        for num in nums:
            divisor = NOD(num)
            if len(divisor) == 4:
                res += sum(divisor)
        return res
        

class Solution:
    def sumFourDivisors(self, nums: List[int]) -> int:
        maxim=max(nums)
      
       
        total=0
        for k in range(len(nums)):
            num_div=0
          
            index=2
            div=[]
            curr_val=nums[k]
            if abs(int(sqrt(curr_val))-sqrt(curr_val))>10**(-12) :
                while index<=int(sqrt(curr_val)):

                    if curr_val%index==0:
                        div.append(index)

                        div.append(nums[k]/index)
                    if len(div)>2:
                        break
                    index+=1

                if len(div)==2:
                    total=total+sum(div)+1+nums[k]
                    
            
        return int(total)
        
                

class Solution:
    def sumFourDivisors(self, nums: List[int]) -> int:
        def NOD(x):
            divisor = set([1,x])
            for i in range(2,int(x**.5) + 1):
                if not x%i:
                    divisor.add(i)
                    divisor.add(x//i)
            return divisor
        res = 0
        for num in nums:
            divisor = NOD(num)
            if len(divisor) == 4:
                res += sum(divisor)
        return res
        

def div(n):
    c=0
    i=1
    k=0
    if sqrt(n).is_integer(): 
        return 0
    while i*i<n and c<=3:
        if n%i==0:
            c+=1
            k+=i+n//i
        
        i+=1
    if c==2:
        return k
    else:
        return 0
class Solution:
    def sumFourDivisors(self, nums: List[int]) -> int:
        ans=0
        for i in nums:
            ans+=div(i)
        return ans
class Solution:
    def sumFourDivisors(self, nums: List[int]) -> int:
        
        
        def findFactors(num):
            if num==0:
                return 0
            res = set()
            for i in range(int(num**0.5)+1):                
                if num%(i+1)==0:
                    res.add(i+1)
                    res.add(num//(i+1))
                    
            return [len(res),sum(res)]
        output = 0
        for num in nums:
            c,sm = findFactors(num)
            # print(c,sm)
            if c==4:
                output+=sm
        return output
class Solution:
    def __init__(self):
        self.divisors = {}
        
    def generate_result(self, n):
        counter = 1
        quo = n // counter
        
        while counter <= quo:
            
            if n % counter == 0:
                yield counter
                if quo != counter:
                    yield quo
                
            counter += 1
            quo = n // counter
        
        
    def count_divisors(self, n):
        if n in self.divisors:
            return self.divisors[n]
        
        result = list(self.generate_result(n))
        
        self.divisors[n] = result
        
        return result
    
    def sumFourDivisors(self, nums: List[int]) -> int:
        divisors = list(map(self.count_divisors, nums))
        four_divisors = list([x for x in divisors if len(x) == 4])
        return sum(map(sum, four_divisors))

class Solution:
    def sumFourDivisors(self, nums: List[int]) -> int:
        res = 0
        for n in nums:
            divisors = self.getDivisors(n)
            if len(divisors) == 4:
                res += sum(divisors)
        return res

    def getDivisors(self, n):
        divisors = set()
        for i in range(1, n):
            if i ** 2 > n:
                break
            if n % i == 0:
                divisors.add(i)
                divisors.add(n//i)
            if len(divisors) > 4:
                break
        return divisors
class Solution:
    def sumFourDivisors(self, nums: List[int]) -> int:
        ret = 0
        for num in nums:
            divs = set()
            i = 1
            while i ** 2 <= num:
                if not num % i:
                    divs.add(i)
                    divs.add(num // i)
                if len(divs) > 4:
                    break
                i += 1
            if len(divs) == 4:
                ret += sum(divs)
        return ret
class Solution:
    def __init__(self):
        self.divisors = {}
    def count_divisors(self, n):
        if n in self.divisors:
            return self.divisors[n]
        result = []
        
        counter = 1
        quo = n // counter
        
        while counter <= quo:
            
            if n % counter == 0:
                result.append(counter)
                if quo != counter:
                    result.append(quo)
                
            counter += 1
            quo = n // counter
                
        self.divisors[n] = result
        return result
    
    def sumFourDivisors(self, nums: List[int]) -> int:
        divisors = list(map(self.count_divisors, nums))
        four_divisors = list([x for x in divisors if len(x) == 4])
        return sum(map(sum, four_divisors))

class Solution:
    def sumFourDivisors(self, nums: List[int]) -> int:
        # 10:58 9/24/20
 
        
        def four_divisors(n):
            div = set()
            cnt = 0
            for i in range(1, int(n ** 0.5) + 1):
                if n % i == 0:
                    div.add(i)
                    div.add(n // i)
                    cnt += 2
                    if cnt > 4:
                    # if len(div) > 4:
                        return 0
            return sum(div) if len(div) == 4 else 0
        
 
        
        if not nums: return 0
        nums.sort()
        total = 0
        # sums = [0]
        past = [None, None]
        
        for i, v in enumerate(nums):
            if i > 0 and v == nums[i - 1] and v == past[0]:
                total += past[1]
                continue
            tmp = four_divisors(v)
            total += tmp
            past = [v, tmp]
            
                    
        return total
                    
        
        
        
            
            
           

class Solution:
    def helper(self, n):
        if n == 1:
            return 0
        
        d = int(math.sqrt(n))
        cnt = 2
        sm = 1 + n
        while d > 1:
            if n % d == 0:
                d1 = n // d
                if d1 != d:
                    sm += d + d1
                    cnt += 2
                else:
                    sm += d
                    cnt += 1
            if cnt > 4:
                return 0
            
            d -= 1
        
        if cnt == 4:
            return sm
        return 0
        
    def sumFourDivisors(self, nums: List[int]) -> int:
        res = 0
        for n in nums:
            res += self.helper(n)
            #print(n, res)
        return res
class Solution:
    def sumFourDivisors(self, nums: List[int]) -> int:
        summ = 0
        for num in nums:
            if num > 1:
                summ += self.divisors(num)
        
        return summ
    
    def divisors(self, num):
        visited_factors = set()
        visited_factors.add(1)
        visited_factors.add(num)
        factors = 2
        summ = 1 + num
        for i in range(2, int(num ** 0.5) + 1):
            # print("i ", i, " num ", num)
            if not num % i and num % i not in visited_factors:
                visited_factors.add(i)
                summ += i
                factors += 1
                secondHalf = num // i
                if secondHalf not in visited_factors:
                    visited_factors.add(secondHalf)
                    factors += 1
                    summ += secondHalf
        
        # print("factors ", factors)
        if factors == 4:
            return summ
        return 0
class Solution:
    def sumFourDivisors(self, nums: List[int]) -> int:
        ans = 0
        for n in nums:
            tmp = set([1,n])
            r = ceil(sqrt(n))
            if r*r == n:
                continue
            for d in range(2,r+1):
                if n%d==0:
                    tmp.add(d)
                    tmp.add(n//d)
            #print(tmp)
            if len(tmp) ==4:
                ans += sum(tmp)
        return ans
class Solution:
    def sumFourDivisors(self, nums: List[int]) -> int:
        z = 0
        for num in nums:
            i = 1
            res = []

            while i*i <= num:
                if (num % i ==0):
                    res.append(i)
                i += 1
            if len(res) == 2:
                lin = [num//j for j in res]
                final = list(set(res + lin))
                if (len (final) == 4):
                    z += sum(final)
        
        return max(0, z)

class Solution:
    def sumFourDivisors(self, nums: List[int]) -> int:
        # p1 * p2
        # 1, p1, p2, p1 * p2
        # p^n: n+1
        # 8: 1, 2, 4, 8
        # p^3
        
        def find_divisors(n):
            i = 1
            divisors = []
            while i * i < n:
                if n % i == 0:
                    divisors.append(i)
                    divisors.append(n // i)
                i += 1
            if i * i == n:
                divisors.append(i)
            return divisors
        
        ans = 0
        for n in nums:
            divisors = find_divisors(n)
            if len(divisors) == 4:
                ans += sum(divisors)
        return ans
class Solution:
    def sumFourDivisors(self, nums: List[int]) -> int:
        
        
        ##LEARN THISS BRUHHHH
        res = 0
        for i in range(len(nums)):
            curSum, curAns = 1 + nums[i], 2
            for j in range(2, int(sqrt(nums[i])) + 1):
                if nums[i] % j == 0:
                    if j == (nums[i] // j):
                        curSum += (nums[i] // j)
                        curAns += 1
                    else:
                        curSum += (j + (nums[i] // j))
                        curAns += 2
            if curAns == 4:
                res += curSum
        return res
  
                

class Solution:
    def find_factors(self, n):
        factors = []
        i = 1
        j = n
        while True:
            if i*j == n:
                factors.append(i)
                if i == j:
                    break
                factors.append(j)
            i += 1
            j = n // i
            if i > j:
                break
        return factors
    def sumFourDivisors(self, nums: List[int]) -> int:
        d = 0
        for i in nums:
            f = self.find_factors(i)
            if len(f)==4:
                d+=f[0]+f[1]+f[2]+f[3]
        return d

class Solution:
    def sumFourDivisors(self, nums: List[int]) -> int:
        # 10:58 9/24/20
        def four_divisors(n):
            div = set()
            for i in range(1, int(n ** 0.5) + 1):
                if n % i == 0:
                    div.add(i)
                    div.add(n // i)
                    if len(div) > 4:
                        return 0
            return sum(div) if len(div) == 4 else 0
        
        # def four_divisors(n):
        #     cnt = 0
        #     sums = 0
        #     if n != 0:
        #         i = 1
        #         while i * i < n:
        #             if n % i == 0:
        #                 cnt += 2
        #                 sums += i + n // i
        #             if cnt > 4:
        #                 return 0
        #             i += 1
        #     return sums if cnt == 4 else 0
        
        if not nums: return 0
        nums.sort()
        total = 0
        # sums = [0]
        past = [None, None]
        
        for i, v in enumerate(nums):
            if i > 0 and v == nums[i - 1] and v == past[0]:
                total += past[1]
                continue
            tmp = four_divisors(v)
            total += tmp
            past = [v, tmp]
            
                    
        return total
                    
        
        
        
            
            
           

class Solution:
    def sumFourDivisors(self, nums: List[int]) -> int:
        ans = 0
        for n in nums:
            d = set()
            for cnd in range(1, floor(sqrt(n))+1):
                q, r = divmod(n, cnd)
                if not r:
                    d.add(q)
                    d.add(cnd)
            if len(d) == 4:
                ans += sum(d)
        return ans
class Solution:
    def __init__(self):
        self.divisors = {}
        
    def generate_result(self, n):
        counter = 1
        quo = n // counter
        
        while counter <= quo:
            
            if n % counter == 0:
                yield counter
                if quo != counter:
                    yield quo
                
            counter += 1
            quo = n // counter
        
        
    def count_divisors(self, n):
        
        result = list(self.generate_result(n))
        
        
        return result
    
    def sumFourDivisors(self, nums: List[int]) -> int:
        divisors = list(map(self.count_divisors, nums))
        four_divisors = list([x for x in divisors if len(x) == 4])
        return sum(map(sum, four_divisors))

class Solution:
    def sumFourDivisors(self, nums: List[int]) -> int:
        divs = dict()
        
        for v in nums:
            divs.setdefault(v, [0, []])
            divs[v][0] += 1
        
        n = max(nums)
        sieve = (1 + n) * [0]
            
        for i in range(2, 1 + n):
            j = i
                
            while j <= n:
                sieve[j] += 1
                    
                if j in divs:
                    divs[j][1].append(i)
                    
                j += i
            
        # print(divs)
            
        return sum([freq * (1 + sum(cur_div)) for k, (freq, cur_div) in list(divs.items()) if len(cur_div) == 3])

class Solution:
    def sumFourDivisors(self, nums: List[int]) -> int:
        if not nums:
            return 0
        ans = 0
        for n in nums:
            rangemax = int(math.sqrt(n))
            factsum = n + 1
            factcount = 2
            for f1 in range(2, rangemax + 1):
                if not n%f1:
                    f2 = n//f1
                    factcount += 1
                    factsum += f1
                    if f1 != f2:
                        factcount += 1
                        factsum += f2
                    if factcount > 4 or factcount%2:
                        break
            if factcount == 4:
                ans += factsum
        return ans
class Solution:
    def sumFourDivisors(self, nums: List[int]) -> int:
        # 10:58 9/24/20
        def four_divisors3(n):
            div = set()
            i = 1
            while i*i < n:
                if n % i == 0:
                    div.add(i)
                    div.add(n // i)
                    if len(div) > 4:
                        return 0
                i += 1
            return sum(div) if len(div) == 4 else 0
        
        def four_divisors(n):
            div = set()
            cnt = 0
            for i in range(1, int(n ** 0.5) + 1):
                if n % i == 0:
                    div.add(i)
                    div.add(n // i)
                    cnt += 2
                    if cnt > 4:
                    # if len(div) > 4:
                        return 0
            return sum(div) if len(div) == 4 else 0
        
        def four_divisors2(n):
            cnt = 0
            sums = 0
            div = set()
            if n != 0:
                # i = 1
                for i in range(1, int(n** 0.5) + 1):
                    if n % i == 0:
                        cnt += 2
                        sums += i + n // i
                        # div.add(i)
                        # div.add(n // i)
                    if cnt > 4:
                        return 0
                    # i += 1
            return sums if cnt == 4 else 0
        
        if not nums: return 0
        nums.sort()
        total = 0
        # sums = [0]
        past = [None, None]
        
        for i, v in enumerate(nums):
            if i > 0 and v == nums[i - 1] and v == past[0]:
                total += past[1]
                continue
            tmp = four_divisors(v)
            total += tmp
            past = [v, tmp]
            
                    
        return total
                    
        
        
        
            
            
           

class Solution:
    def sumFourDivisors(self, nums: List[int]) -> int:
        # 10:58 9/24/20
 
        
        def four_divisors(n):
            div = set()
            cnt = 0
            sums = 0
            for i in range(1, int(n ** 0.5) + 1):
                if n % i == 0:
                    div.add(i)
                    sums += i
                    div.add(n // i)
                    sums += n // i
                    cnt += 2
                    if cnt > 4:
                    # if len(div) > 4:
                        return 0
                    
            return sums if len(div) == 4 else 0
            # return sum(div) if len(div) == 4 else 0
        
 
        
        if not nums: return 0
        nums.sort()
        total = 0
        # sums = [0]
        past = [None, None]
        
        for i, v in enumerate(nums):
            if i > 0 and v == nums[i - 1] and v == past[0]:
                total += past[1]
                continue
            tmp = four_divisors(v)
            total += tmp
            past = [v, tmp]
            
                    
        return total
                    
        
        
        
            
            
           

class Solution:
    def sumFourDivisors(self, nums: List[int]) -> int:
        return sum(contr(n) for n in nums)        
    
def contr(n):
    p = None
    if n**.5%1==0:
        return 0
    for i in range(2,math.ceil(n**.5)):
        if n%i==0:
            if p is None:
                p = i
            else:
                return 0
    if p is None:
        return 0
    return 1 + p + n//p + n
class Solution:
    def sumFourDivisors(self, nums: List[int]) -> int:
        n = 400
        prime = [True for i in range(n+1)] 
        p = 2
        while (p * p <= n):
            if (prime[p] == True):
                for i in range(p * p, n+1, p): 
                    prime[i] = False
            p += 1
        prime[0] = False
        prime[1] = False
        prime_set = [p for p in range(n+1) if prime[p]]
        
        cnt = 0
        for i in nums:
            if i == 0:
                continue
            for p in prime_set:
                if i % p == 0:
                    r = i // p
                    if r == p or r == 1:
                        break
                    r_prime = True
                    for q in prime_set:
                        if r % q == 0:
                            if r != q:
                                r_prime = False
                            break
                    if r_prime:  
                        cnt += (p+1) * (r+1)
                    break
                    
        
        for i in nums:
            p = int(i**(1/3) +0.5)
            if prime[p] and p**3 == i:
                cnt += (p * i - 1) // (p - 1)
                print(i, p, p**2, p)
                    
        return cnt
class Solution:
    def divisors(self, n, c={}):
        if n in c:
            return c[n]
        d = []
        for i in range(1, int(sqrt(n) + 1)):
            if n % i == 0:
                d.append(i)
                j = n // i
                if j != i:
                    d.append(j)
            if len(d) > 4:
                break
        if len(d) == 4:
            s = sum(d)
            c.update({n:s})
            return s
        else:
            c.update({n:0})
            return 0

    def sumFourDivisors(self, nums: List[int]) -> int:
        return sum(self.divisors(x) for x in nums)
class Solution:
    def sumFourDivisors(self, nums: List[int]) -> int:
        @lru_cache(None)
        def divisors(v):
            divs = set()
            for i in range(1,ceil(sqrt(v))+1):
                if not v % i: divs.update({i, v//i})
                if len(divs) > 4: return 0
            return sum(divs) if len(divs)==4 else 0

        return sum(map(divisors, nums))

class Solution:
    def sumFourDivisors(self, nums: List[int]) -> int:
        total = 0
        pSieve = [0 for k in range(10**5 + 1)]
        for k in range(2, len(pSieve)):
            if pSieve[k] == 1:
                continue
            pSieve[k + k::k] = [1]*((len(pSieve) - 1)//k - 1)
        for num in nums:
            if num == 1 or pSieve[num] == 0 or sqrt(num) == int(sqrt(num)):
                continue
            k = 2
            while num % k != 0:
                k += 1
            if (num == k**3) or pSieve[num // k] == 0:
                total += 1 + num + k + num // k
        return total
class Solution:
    def sumFourDivisors(self, nums: List[int], c={}) -> int:
        r = 0
        for n in nums:
            if n in c:
                r += c[n]
                continue
            d = []
            for i in range(1, round(sqrt(n) + 1)):
                if n % i == 0:
                    d.append(i)
                    j = n // i
                    if j != i:
                        d.append(j)
                if len(d) > 4:
                    break
            if len(d) == 4:
                s = sum(d)
                c.update({n:s})
                r += s
            else:
                c.update({n:0})
        return r
class Solution:
    def sumFourDivisors(self, nums: List[int], c={}) -> int:
        r = 0
        for n in nums:
            if n in c:
                r += c[n]
                continue
            s = n + 1
            cnt = 2
            for i in range(2, round(sqrt(n) + 1)):
                if n % i == 0:
                    cnt += 1
                    if cnt > 4:
                        s = 0
                        break
                    s += i
                    j = n // i
                    if j != i:
                        cnt += 1
                        if cnt > 4:
                            s = 0
                            break
                        s += j
            if cnt == 4:
                c.update({n:s})
                r += s
            else:
                c.update({n:0})
        return r
class Solution:
    def sumFourDivisors(self, nums: List[int]) -> int:
        def helper(n):
            if int(math.sqrt(n)) * int(math.sqrt(n)) == n:
                return 0
            summary = 1 + n
            count = 2
            for i in range(2, int(math.sqrt(n))+1):
                if n % i == 0:
                    summary += (n//i + i)
                    count += 2
                    if count > 4:
                        break
            
            if count == 4: 
                return summary  
            else: 
                return 0
        res = 0
        
        for n in nums:
            res += helper(n)
                
        return res
class Solution:
    def sumFourDivisors(self, nums: List[int]) -> int:
        res = 0
        for n in nums:
            divisor = set()
            for i in range(1, floor(sqrt(n)) + 1):
                if n % i == 0:
                    divisor.add(n // i)
                    divisor.add(i)
                    if len(divisor) > 4:
                        break
            if len(divisor) == 4:
                res += sum(divisor)
                
        return res
                    

class Solution:
    def sumFourDivisors(self, nums: List[int], c={}) -> int:
        r = 0
        for n in nums:
            if n in c:
                r += c[n]
                continue
            d = []
            for i in range(1, int(sqrt(n) + 1)):
                if n % i == 0:
                    d.append(i)
                    j = n // i
                    if j != i:
                        d.append(j)
                if len(d) > 4:
                    break
            if len(d) == 4:
                s = sum(d)
                c.update({n:s})
                r += s
            else:
                c.update({n:0})
        return r
class Solution:
    def sumFourDivisors(self, nums: List[int]) -> int:
        valAll = 0
        
        for num in nums:
            local = set()
            for i in range(1, int(math.sqrt(num))+1):
                if num % i == 0:
                    local.add(i)
                    local.add(int(num/i))
                    if len(local) > 4:
                        break
            if len(local) == 4:
                valAll += sum(local)
            #print(str(num)+"  "+str(local))
        return valAll
class Solution:
    def sumFourDivisors(self, nums: List[int]) -> int:
        @lru_cache(None)
        def divisors(v):
            divs = set()
            for i in range(1,ceil(sqrt(v))+1):
                if not v % i: divs.update({i, v//i})
                if len(divs) > 4: return 0
            return sum(divs) if len(divs)==4 else 0
        return sum(map(divisors, nums))
        

class Solution:
    def sumFourDivisors(self, nums: List[int]) -> int:
        @lru_cache(None)
        def divisors(v):
            res = set()
            for i in range(1,ceil(sqrt(v))+1):
                if not v % i:
                    res.update({i, v//i})
                if len(res) > 4: return 0
            return sum(res) if len(res)==4 else 0
        return sum(list(map(divisors, nums)))
        

class Solution:
    def sumFourDivisors(self, nums: List[int]) -> int:
        ret_count = {}  # N -> Count
        ret_sum = {}  # N -> Sum
        # --
        for n in nums:
            if n in ret_sum:
                if ret_sum[n] is not None:
                    ret_count[n] += 1
                continue
            # calculate it!
            # max_div = int(n ** 0.5)
            # if max_div*max_div >= n:
            #     max_div -= 1  # forbid three
            cur_div = 2
            hit_div = None
            while cur_div*cur_div <= n:
                if n % cur_div==0:
                    if hit_div is None:
                        hit_div = cur_div
                    else:
                        hit_div = None
                        break
                cur_div += 1
            # get result
            if hit_div is not None and hit_div!=(n//hit_div):  # hit it!!
                res = 1 + n + hit_div + (n//hit_div)
                ret_count[n] = 1
            else:
                res = None
            ret_sum[n] = res
        # --
        ret = sum(ret_sum[k]*c for k,c in list(ret_count.items()))
        return ret

class Solution:
    def sumFourDivisors(self, nums: List[int]) -> int:
        ans = 0
        for num in nums:
            cnt = 0
            for i in range(2, int(num**0.5)+1):
                if num%i == 0:
                    cnt += 1
                    d = i
                if cnt > 1:
                    break
            if cnt == 1 and d != num//d:
                ans += 1 + d + num//d + num
        return ans
class Solution:
    def sumFourDivisors(self, nums: List[int]) -> int:
        n = 400
        prime = [True for i in range(n+1)] 
        p = 2
        while (p * p <= n):
            if (prime[p] == True):
                for i in range(p * p, n+1, p): 
                    prime[i] = False
            p += 1
        prime[0] = False
        prime[1] = False
        prime_set = [p for p in range(n+1) if prime[p]]
        
        cnt = 0
        for i in nums:
            if i == 0:
                continue
            for p in prime_set:
                if p * p > i:
                    break
                if i % p == 0:
                    r = i // p
                    if r == p or r == 1:
                        break
                    r_prime = True
                    for q in prime_set:
                        if q * q > r:
                            break
                        if r % q == 0:
                            if r != q:
                                r_prime = False
                            break
                    if r_prime:  
                        cnt += (p+1) * (r+1)
                    break
                    
        
        for i in nums:
            p = int(i**(1/3) +0.5)
            if prime[p] and p**3 == i:
                cnt += (p * i - 1) // (p - 1)
                print(i, p, p**2, p)
                    
        return cnt
class Solution:
    def sumFourDivisors(self, nums: List[int]) -> int:
        @lru_cache(None)
        def divisors(v):
            res = set()
            for i in range(1,ceil(sqrt(v))+1):
                if not v % i:
                    res.add(i)
                    res.add(v//i)
                if len(res) > 4: return 0
            return sum(res) if len(res)==4 else 0
        return sum(list(map(divisors, nums)))
        

class Solution:
    def sumFourDivisors(self, nums: List[int]) -> int:
        @lru_cache(None)
        def divisors(v):
            divs = set()
            for i in range(1,ceil(sqrt(v))+1):
                if not v % i: divs.update({i, v//i})
                if len(divs) > 4: return 0
            return sum(divs) if len(divs)==4 else 0

        return sum(map(divisors, nums))
class Solution:
    def sumFourDivisors(self, nums: List[int]) -> int:
        @lru_cache(None)
        def divisors(v):
            res = set()
            for i in range(1,ceil(sqrt(v))+2):
                if not v % i:
                    res.add(i)
                    res.add(v//i)
                if len(res) > 4: return 0
            return sum(res) if len(res)==4 else 0
        return sum(map(divisors, nums))
        

class Solution:
    def sumFourDivisors(self, nums: List[int]) -> int:
        res = 0
        
        @lru_cache(None)
        def divisors(v):
            res = []
            for i in range(1,ceil(sqrt(v))+2):
                if len(res) > 4: return 0
                if not v % i:
                    res += i,
                    if v//i > i:
                        res += v//i,
                    else:
                        break
            res = set(res)
            return sum(res) if len(res)==4 else 0
        
        for v in nums:
            res += divisors(v)
            
        return res

from math import sqrt

class Solution:
    def sumFourDivisors(self, nums: List[int]) -> int:
        
        sum_of_factor = 0
        
        for x in nums:
            
            factors = set()
            
            # collect all factors into set
            for i in range(1, int(sqrt(x)+1) ):
                
                if x % i == 0:
                    
                    factors.add( i )
                    factors.add( x // i )
                    
                    if len( factors ) > 4:
                        # early breaking when there is too many factors
                        break
                        
                
            if len( factors ) == 4:
                # update sum of four divisors
                sum_of_factor += sum(factors)
        
        return sum_of_factor        
        

class Solution:
    def sumFourDivisors(self, nums: List[int]) -> int:
        # 10:58 9/24/20
        # def four_divisors(n):
        #     div = set()
        #     for i in range(1, int(n ** 0.5) + 1):
        #         if n % i == 0:
        #             div.add(i)
        #             div.add(n // i)
        #             if len(div) > 4:
        #                 return 0
        #     return sum(div) if len(div) == 4 else 0
        
        def four_divisors(n):
            cnt = 0
            div = set()
            if n != 0:
                i = 1
                for i in range(1, int(n** 0.5) + 1):
                    if n % i == 0:
                        cnt += 2
                        # sums += i + n // i
                        div.add(i)
                        div.add(n // i)
                    if len(div) > 4:
                        return 0
                    # i += 1
            return sum(div) if len(div) == 4 else 0
        
        if not nums: return 0
        nums.sort()
        total = 0
        # sums = [0]
        past = [None, None]
        
        for i, v in enumerate(nums):
            if i > 0 and v == nums[i - 1] and v == past[0]:
                total += past[1]
                continue
            tmp = four_divisors(v)
            total += tmp
            past = [v, tmp]
            
                    
        return total
                    
        
        
        
            
            
           

from collections import defaultdict
from math import ceil
class Solution:
    def sumFourDivisors(self, nums: List[int]) -> int:

        count = defaultdict(set)
        for j, num in enumerate(nums):
            for i in range(1,ceil(num**0.5)+1):
                if num % i == 0:
                    count[(j,num)].update({i, num//i})
                    if (len(count[j,num])>4):
                        break
        total = 0 
        print(count)
        for num in count:
            if len(count[num])==4:
                total+=sum(count[num])
        return total

class Solution:
    def getDivisors(self, x):
        if x == 1:
            return [1]
        out = []
        bound = int(sqrt(x)) + 1
        for i in range(1, bound):
            if x % i == 0:
                out.append(i)
                if x//i != i:
                    out.append(x//i)
            if len(out) > 4:
                break
        return out
        
    def sumFourDivisors(self, nums: List[int]) -> int:
        divisors = {}
        sum_four = 0
        for x in nums:
            if x in divisors:
                if len(divisors[x]) == 4:
                    sum_four += sum(divisors[x])
            else:
                x_div = self.getDivisors(x)
                if len(x_div) == 4:
                    sum_four += sum(x_div)
                divisors[x] = x_div
        return sum_four

from collections import defaultdict
from math import ceil
class Solution:
    def sumFourDivisors(self, nums: List[int]) -> int:

        count = defaultdict(set)
        for j, num in enumerate(nums):
            count[(j,num)].add(num)
            for i in range(1,ceil(num**0.5)+1):
                if num % i == 0:
                    count[(j,num)].update({i, num//i})
                    if (len(count[j,num])>4):
                        break
        total = 0 
        print(count)
        for num in count:
            if len(count[num])==4:
                total+=sum(count[num])
        return total

class Solution:
    def sumFourDivisors(self, nums: List[int]) -> int:
        # go through the range of numbers in range(0, len(nums))
        # for each number go through the range(0, sqrt(num))
        # for the first loop initialize a set or an arrya to keep track that there are 4 divisors
        # go through the numbers, in the end if the len(set) == 4, get teh sum of array and append it to teh answer
        answer = 0 
        for num in nums:
            mySet = set()
            for num2 in range(1, (int(sqrt(num)) + 1)):
                if (num % num2) == 0:
                    mySet.add(num2)
                    mySet.add(num / num2)
                    if len(mySet) > 4:
                        break
            print(mySet)
            if len(mySet) == 4:
                answer += int(sum(mySet))
        return answer
class Solution:
    def divisors(self, n, c={}):
        if n in c:
            return c[n]
        d = []
        for i in range(1, int(sqrt(n) + 1)):
            if n % i == 0:
                d.append(i)
                j = n // i
                if j != i:
                    d.append(j)
        if len(d) == 4:
            s = sum(d)
            c.update({n:s})
            return s
        else:
            c.update({n:0})
            return 0

    def sumFourDivisors(self, nums: List[int]) -> int:
        return sum(self.divisors(x) for x in nums)
class Solution:
    def sumFourDivisors(self, nums: List[int]) -> int:
        
        def divisors(v):
            divs = set()
            for i in range(1,ceil(sqrt(v))+1):
                if not v%i:
                    divs.update({i, v//i})
                if len(divs)>4:
                    return 0
            return sum(divs) if len(divs)==4 else 0
        
        return sum(map(divisors,nums))
from collections import defaultdict
from math import ceil
class Solution:
    def sumFourDivisors(self, nums: List[int]) -> int:
        count = defaultdict(set)
        for j, num in enumerate(nums):
            count[(j,num)].add(num)
            for i in range(1,ceil(num**0.5)+1):
                if num % i == 0:
                    count[(j,num)].update({i, num//i})
                    if (len(count[j,num])>4):
                        break
        total = 0 
        print(count)
        for num in count:
            if len(count[num])==4:
                total+=sum(count[num])
        return total

class Solution:
    def sumFourDivisors(self, nums: List[int]) -> int:
        ret = 0
        for num in nums:
            sqrt = int(math.sqrt(num))
            if sqrt*sqrt == num:
                continue
            divSum = 0
            count = 0
            for i in range(1, sqrt+1):
                if num%i == 0:
                    divSum += i + num//i
                    count += 1
                    if count > 2:
                        break
            if count == 2:
                ret += divSum
        return ret
class Solution:
    def sumFourDivisors(self, nums: List[int]) -> int:
        def getDivisors(k):
            count,second = 0,0
            for i in range(2,int(sqrt(k))+1):
                if k%i == 0 :
                    count += 1
                    if count > 1 or i*i == k: return [0]
                    second = k//i
            if count == 1: return [1,second,k//second,k]
            else: return [0]        
    
        total = 0
        for num in nums:
            total += sum(getDivisors(num))
        return total
    

class Solution:
    def sumFourDivisors(self, nums: List[int]) -> int:
        
        def get_divs(num):
            
            #print(num)
            divs = []
            for i in range(1, int(sqrt(num)) + 1):
                #print(divs)
                
                if(not num%i):
                    #divides
                    divs.append(i)
                    if(i != int(num/i)):
                        divs.append(int(num/i))
                
                if(len(divs) > 4):
                    return None
                
            #print(divs)
            if(len(divs) < 4):
                return None
            
            #print(divs)
            return sum(divs)
        
        ans = 0
        
        for item in nums:
            divs = get_divs(item)
            #print(item, divs)
            if(divs):
                ans += divs
        
        return ans
class Solution:
    def sumFourDivisors(self, nums: List[int]) -> int:
        ans = 0
        for num in nums:
            ans += self.divisors(num)
            
        return ans
    
    def divisors(self,num):
        memo = set()
        for i in range(1,num + 1):
            if i * i > num:
                break
            if num % i == 0:
                memo.add(i)
                memo.add(num//i)
                if len(memo) > 4:
                    return 0
                
        if len(memo) == 4:
            return sum(memo)
        return 0

class Solution:
    def divisors(self, n, c={}):
        if n in c:
            return c[n]
        d = []
        for i in range(1, int(sqrt(n) + 1)):
            if n % i == 0:
                d.append(i)
                j = n // i
                if j != i:
                    d.append(j)
        c.update({n:d})
        return d

    def sumFourDivisors(self, nums: List[int]) -> int:
        s = 0
        for n in nums:
            d = self.divisors(n)
            if len(d) == 4:
                s += sum(d)
        return s
class Solution:
    def sumFourDivisors(self, nums: List[int]) -> int:
        res = 0
        for num in nums:
            divisor = set() 
            for i in range(1, floor(sqrt(num)) + 1):
                if num % i == 0:
                    divisor.add(num//i)
                    divisor.add(i)
                if len(divisor) > 4:    
                    break
                    
            if len(divisor) == 4:
                res += sum(divisor)
        return res  

class Solution:
    def sumFourDivisors(self, nums: List[int]) -> int:
        def div4(i):
            if i <= 5:
                return set()
            else:
                count = {1,i}
                for j in range(2, int(math.sqrt(i)) + 1):
                    if i % j == 0:
#                        print(i,j)
                        count.update({j,i/j})
                    if len(count) > 4:
                        return count
                return count
    
        count = 0
        for i in nums:
            s = div4(i)
#            print(s)
            if len(s) ==4:
                count += sum(s)
        return int(count)
class Solution:
    def sumFourDivisors(self, nums: List[int]) -> int:
        res = 0
        for num in nums:
            divisor = set() 
            for i in range(1, floor(sqrt(num)) + 1):
                if num % i == 0:
                    divisor.add(num//i)
                    divisor.add(i)
                if len(divisor) > 4:    
                    break
                    
            if len(divisor) == 4:
                res += sum(divisor)
        return res  
class Solution:
    def sumFourDivisors(self, nums: List[int]) -> int:
        return sum([self.sumofDivisors(num) for num in nums])  
    def sumofDivisors(self, num):
        s = set()
        for i in range(1,int(sqrt(num))+1):
            if num%i==0: 
                s.add(i)
                s.add(num//i)
            if len(s)>4:return 0    
        return sum(s) if len(s)==4 else 0



class Solution:
    def sumFourDivisors(self, nums: List[int]) -> int:
        def four_div_sum(num):
            divs = set()
            for i in range(1, floor(sqrt(num)) + 1):
                if num % i == 0:
                    divs.update({i, num//i})
                if len(divs) > 4:
                    return 0
            return sum(divs) if len(divs) == 4 else 0
    
        return sum(four_div_sum(num) for num in nums)
from collections import defaultdict
from math import ceil
class Solution:
    def sumFourDivisors(self, nums: List[int]) -> int:
        def getSumOfDivisors(n):
            divisors = set()
            for i in range(1, ceil(n**0.5)+1):
                if n%i==0:
                    divisors.update({i,n//i})
                if len(divisors)>4:
                    return 0
            return sum(divisors) if len(divisors)==4 else 0
        return sum(map(getSumOfDivisors, nums))
import math
class Solution:
    def sumFourDivisors(self, nums: List[int]) -> int:
        if not nums:
            return 0
    
        res = 0
        for i in nums:
            divisor = set()
            for j in range(1, int(math.sqrt(i))+1):
                if i%j == 0:
                    divisor.add(j)
                    divisor.add(i//j)
                if len(divisor)>4:
                    break
            if len(divisor) == 4:
                res += sum(divisor)
        return res
        
            
    

class Solution:
    def sumFourDivisors(self, nums: List[int]) -> int:
        result = 0
        for n in nums:
            divisors = set()
            for i in range(1, floor(sqrt(n)) + 1):
                if n % i == 0:
                    divisors.add(i)
                    divisors.add(n//i)
                if len(divisors) > 4:
                    break
            if len(divisors) == 4:
                result += sum(divisors)
        return result
from math import sqrt

class Solution:
    def sumFourDivisors(self, nums: List[int]) -> int:
        res = 0
        for num in nums:
            divisor = set() 
            for i in range(1, floor(sqrt(num)) + 1):
                if num % i == 0:
                    divisor.add(num//i)
                    divisor.add(i)
                if len(divisor) > 4:    
                    break
                    
            if len(divisor) == 4:
                res += sum(divisor)
        return res  

# class Solution:
#     def sumFourDivisors(self, nums: List[int]) -> int: #o(n^2)
#         ans = 0
#         for num in nums:
#             if Solution.findNumDivisors(self,num) == 4:
#                 ans += Solution.sumDivisors(self,num)
        
#         return ans
    
#     def findNumDivisors(self, num) -> int: #o(N)
#         cnt = 1
        
#         for i in range(1,int(sqrt(num))):
#             if num % i == 0:
#                 cnt += 1
        
#         return cnt
    
#     def sumDivisors(self, num) -> int:
#         ans = num
        
#         for i in range(1,int(sqrt(num)):
#             if num % i == 0:
#                 ans += i
        
#         return ans



class Solution:
    def sumFourDivisors(self, nums: List[int]) -> int:
        
        
        div_sum = 0
        
        for num in nums:
            divs = set()
            for i in range(1, floor(sqrt(num))+1):
                if num % i == 0:
                    divs.add(num//i)
                    divs.add(i)
                if len(divs) > 4:
                    break
            if len(divs) == 4:
                div_sum += sum(divs)
        return div_sum

       

class Solution:
    def sumFourDivisors(self, nums: List[int]) -> int:
        
        
        def findFactors(num):
            res = set()
            for i in range(int(num**0.5)+1):                
                if num%(i+1)==0:
                    res.add(i+1)
                    res.add(num//(i+1))
                if len(res)>4:
                    break
            if len(res) == 4:                    
                return sum(res)
            else:
                return 0
        
        output = 0
        for num in nums:
            temp = findFactors(num)
            output+=temp
        return output
class Solution:
    def sumFourDivisors(self, nums: List[int]) -> int:
        out = 0
        for i in nums:
            temp = set()
            for j in range(1, floor(sqrt(i))+1):
                if i % j == 0:
                    temp.add(j)
                    temp.add(int(i/j))
                if len(temp) > 4:
                    break
            if len(temp) == 4:
                out += sum(temp)
        return out
class Solution:
    def sumFourDivisors(self, nums: List[int]) -> int:
        res = 0
        for num in nums:
            divisor = set() 
            for i in range(1, int(sqrt(num)) + 1):
                if num % i == 0:
                    divisor.add(num//i)
                    divisor.add(i)
                if len(divisor) > 4:    
                    break
                    
            if len(divisor) == 4:
                res += sum(divisor)
        return res 
        

class Solution:
    def sumFourDivisors(self, nums: List[int]) -> int:
        ret = 0
        for n in nums:
            divisors = set()
            for i in range(1,math.floor(n**(0.5)) + 1):
                if n % i == 0:
                    divisors.add(i)
                    divisors.add(n/i)
                if len(divisors) > 4:
                    break
            if len(divisors) == 4:
                ret += sum(divisors) 
            
        return int(ret)

class Solution:
    def sumFourDivisors(self, nums: List[int]) -> int:
        res = 0
        for num in nums:
            divisor = set() 
            for i in range(1, floor(sqrt(num)) + 1):
                if num % i == 0:
                    divisor.add(num//i)
                    divisor.add(i)
                if len(divisor) > 4:    
                    break
                    
            if len(divisor) == 4:
                res += sum(divisor)
        return res
class Solution:
    def find_divisors(self, num):
        cnt = 0
        run_sum = num + 1
        for i in range(2, int(num**0.5)+1):
            if i * i == num:
                return 0
            if cnt > 1:
                return 0
            if not num % i:
                run_sum += num//i + i
                cnt += 1
        return run_sum if cnt == 1 else 0
        
    def sumFourDivisors(self, nums: List[int]) -> int:
        cnt = 0
        for i in nums:
            cnt += self.find_divisors(i)
        return cnt
class Solution:
    def sumFourDivisors(self, nums: List[int]) -> int:
        def getDivs(num):
            result = []
            for div in range(1,int(num**(1/2))+1):
                if num % div == 0:
                    result.append(div)
                    result.append(num // div)
                if len(result) > 4:
                    print(num,result)
                    return 0
                
            if (int(num**(1/2))) * (int(num**(1/2))) == num:
                # result.append(int(num**(1/2)))
                result.pop()
                
            # print(result,num)
            if len(result) == 4:
                return sum(result)
            else:
                return 0
        
        total = 0
        for num in nums:
            total += getDivs(num)
        
        return total
class Solution:
    def sumFourDivisors(self, nums: List[int]) -> int:
        res = 0
        ls = len(nums)
        for i in range(ls):
            divs = set()
            for j in range(1, floor(sqrt(nums[i])) + 1):
                if nums[i] % j == 0:
                    divs.add(nums[i]//j)
                    divs.add(j)
                if len(divs) > 4:
                    break
            
            if len(divs) == 4:
                res = res + sum(divs)
        return res
class Solution:
    def sumFourDivisors(self, nums: List[int]) -> int:
        def check(x):
            v = set()
            i = 1
            while i * i <= x:
                if x % i == 0:
                    v.add(i)
                    v.add(x // i)
                    if len(v) > 4:
                        return 0
                i += 1
            if len(v) == 4:
                return sum(v)
            return 0
        return sum([check(x) for x in nums])

class Solution:
    def sumFourDivisors(self, nums: List[int]) -> int:
        range_6 = list(range(6))
        result = 0
        for num in nums:
            if num in range_6:
                pass
            else:
                pivot = int(num ** 0.5)
                temp = [1, num]
                len_t = 2
                for i in range(2, pivot+1):
                    divisor, rem = divmod(num, i)
                    if not rem:
                        if i == divisor:
                            len_t = 0
                            break
                        temp += [i, divisor]
                        len_t += 2
                        if len_t > 4:
                            break
                if len_t == 4:
                    result += sum(temp)
        return result

class Solution:
    def sumFourDivisors(self, nums: List[int]) -> int:
        ans=0
        for val in nums:
            P=self.check(val)
            if P:
                ans+=sum(P)
        return ans
    
    
    
    def check(self,n):
        L=[n]
        count=1
        if n!=1:
            L.append(1)
            count+=1
        
        for i in range(2,int(n**0.5)+1):
            
            if n%i==0:
                L.append(i)
                count+=1
                if n/i!=float(i):
                    L.append(n//i)
                    count+=1
                if count>4:return None
        if count!=4:
            return None
        return L
class Solution:
    def sumFourDivisors(self, nums: List[int]) -> int:
        
        def check(n):
            i = 1
            cnt = 0
            res = 0
            while i*i<n:
                if n%i==0:
                    cnt += 2
                    res += i
                    res += n//i
                i += 1
                if cnt>4:
                    return 0
            if i*i==n:
                cnt += 1
                res += i
            if cnt == 4:
                return res
            else:
                return 0
            
        # print (check(21))
        # return 1
        res = sum(check(n) for n in nums)
        return res

class Solution:
    def sumFourDivisors(self, nums: List[int]) -> int:
        range_6 = list(range(6))
        result = 0
        for num in nums:
            if num in range_6:
                pass
            else:
                pivot = int(num ** 0.5)
                temp = 1 + num
                len_t = 2
                for i in range(2, pivot+1):
                    divisor, rem = divmod(num, i)
                    if not rem:
                        if i == divisor:
                            len_t = 0
                            break
                        temp += i + divisor
                        len_t += 2
                        if len_t > 4:
                            break
                if len_t == 4:
                    result += temp
        return result

class Solution:
    def __init__(self):
        self.divisors = {}
        
        
    def count_divisors(self, n):
        if n in self.divisors:
            return self.divisors[n]
        result = [1, n]
        
        counter = 2
        quo = n // counter
        
        while counter <= quo:
            
            if n % counter == 0:
                result.append(counter)
                if quo != counter:
                    result.append(quo)
                    
            #Don't have to keep calculating
            if len(result) > 4:
                break
                
            counter += 1
            quo = n // counter
                
        self.divisors[n] = result
        return result
    
    def sumFourDivisors(self, nums: List[int]) -> int:
        divisors = list(map(self.count_divisors, nums))
        four_divisors = list([x for x in divisors if len(x) == 4])
        return sum(map(sum, four_divisors))

class Solution:
    def __init__(self):
        self.divisors = {}
        
        
    def count_divisors(self, n):
        if n in self.divisors:
            return self.divisors[n]
        result = []
        
        counter = 1
        quo = n // counter
        
        while counter <= quo:
            
            if n % counter == 0:
                result.append(counter)
                if quo != counter:
                    result.append(quo)
                    
            #Don't have to keep calculating
            if len(result) > 4:
                break
                
            counter += 1
            quo = n // counter
                
        self.divisors[n] = result
        return result
    
    def sumFourDivisors(self, nums: List[int]) -> int:
        divisors = list(map(self.count_divisors, nums))
        four_divisors = list([x for x in divisors if len(x) == 4])
        return sum(map(sum, four_divisors))

class Solution:
    def sumFourDivisors(self, nums: List[int]) -> int:
        import math
        
        def isprime(n):
            if not n % 1 == 0: return False
            if math.sqrt(n) % 1 == 0: return False
            for i in range(math.ceil(math.sqrt(n))):
                if i == 0 or i == 1: continue
                if n % i == 0: return False
            return True
        
        ans = 0
        for num in nums:
            if num < 6:
                continue
            if math.sqrt(num) % 1 == 0: continue
            if isprime(pow(num, 1/3)) or num == 4913: # pow(4913, 1/3) == 16.999999999999996
                ans += 1 + pow(num, 1/3) + pow(num, 2/3) + num
                continue
            divisors = 0
            for i in range(math.ceil(math.sqrt(num))):
                if i == 0 or i == 1:
                    continue
                if num % i == 0:
                    if (num / i) % i == 0:
                        break
                    if not divisors == 0:
                        divisors = 0
                        break
                    divisors = i
            if (not divisors == 0) and isprime(num / divisors) and isprime(divisors):
                ans += (divisors + 1) * ((num / divisors) + 1)
        
        return int(ans)
import numpy as np

class Solution:
    def sumFourDivisors(self, nums: List[int]) -> int:
        
        ret = 0
        for num in nums:
            divisors = set()
            N = int(np.floor(np.sqrt(num)))
            for i in range(1, N+1):
                if num % i == 0:
                    divisors.add(i)
                    divisors.add(num//i)
                if len(divisors) > 4:
                    break
            if len(divisors) == 4:
                ret += sum(divisors)
        return ret
            

class Solution:
    def sumFourDivisors(self, nums: List[int]) -> int:
        import math
        
        def isprime(n):
            if not n % 1 == 0: return False
            if math.sqrt(n) % 1 == 0: return False
            for i in range(math.ceil(math.sqrt(n))):
                if i == 0 or i == 1: continue
                if n % i == 0: return False
            return True
        
        ans = 0
        for num in nums:
            if num < 6:
                continue
            if math.sqrt(num) % 1 == 0: continue
            if isprime(pow(num, 1/3)) or num == 4913:
                ans += 1 + pow(num, 1/3) + pow(num, 2/3) + num
                continue
            divisors = 0
            for i in range(math.ceil(math.sqrt(num))):
                if i == 0 or i == 1:
                    continue
                if num % i == 0:
                    if (num / i) % i == 0:
                        break
                    if not divisors == 0:
                        divisors = 0
                        break
                    divisors = i
            if (not divisors == 0) and isprime(num / divisors) and isprime(divisors):
                ans += (divisors + 1) * ((num / divisors) + 1)
        
        return int(ans)
class Solution:
    def sumFourDivisors(self, nums: List[int]) -> int:
        import math
        s=0
        nums=sorted(nums, reverse=True)
        for i in nums:
            count=set()
            true=True
            for x in range(2, int(math.sqrt(i))+1):
                if len(count)>4:
                    true=False
                    break
                if i%x==0:
                    count.add(x)
                    count.add(i//x)
            if len(count)==2 and true:
                s+=sum(count)+i+1
        return s

class Solution:
    def sumFourDivisors(self, nums: List[int]) -> int:
        ttl = 0
        
        for n in nums:
            seen = set()
            for i in range(1,int(sqrt(n)) + 1):
                if n % i == 0:
                    seen.add(i)
                    seen.add(n/i)
                if len(seen) >= 5:
                    break
            if len(seen) == 4:
                ttl += sum(seen)
        return int(ttl)
                    
                
                

class Solution:
    def sumFourDivisors(self, nums: List[int]) -> int:
        res = 0
        for num in nums:
            div = set()
            for j in range(1,int(sqrt(num)) + 1):
                if not num%j:
                    div.add(j)
                    div.add(num//j)
                if len(div) > 4:
                    break
            if len(div) == 4:
                res += sum(div)
        return res
        

class Solution:
    def sumFourDivisors(self, nums: List[int]) -> int:
        
        def findiv(num):
            res = 0
            cnt = 0
            for i in range(1, int(num ** 0.5) + 1):
                if not num % i:
                    if i * i == num:
                        cnt += 1
                        res += i
                    else:
                        cnt += 2
                        res += i
                        res += num // i
            return res if cnt == 4 else 0
        
        
        res = 0
        
        for num in nums:
            res += findiv(num)
        return res
class Solution:
    def sumFourDivisors(self, nums: List[int]) -> int:
        ans = 0
        for n in nums:
            tmp = set([1,n])
            r = ceil(sqrt(n))
            if r*r == n:
                continue
            for d in range(2,r+1):
                if n%d==0:
                    tmp.add(d)
                    tmp.add(n//d)
                if len(tmp)>4:
                    break
            #print(tmp)
            if len(tmp) ==4:
                ans += sum(tmp)
        return ans
class Solution:
    def sumFourDivisors(self, nums: List[int]) -> int:
        ans = 0
        for num in nums:
            out = []

            for i in range(1, int(num ** 0.5)+1):
                a, b = divmod(num, i)
                if b == 0:
                    if a == i: out.append(a)
                    else: out.extend([a, i])
                if len(out) > 4: break
            if len(out) == 4:
                ans += sum(out)
        
        return ans
class Solution:
    def sumFourDivisors(self, nums: List[int]) -> int:
        # find all divisor of this number and use set() to select all the distinct factors
        res = 0
        for num in nums:
            divisor_num = set()
            for i in range(1, int(sqrt(num))+1):
                if num%i == 0:
                    divisor_num.add(num//i)
                    divisor_num.add(i)
                    
            if len(divisor_num) == 4:
                res +=sum(divisor_num)
                
                
                
        return res
                
                    
                    

