def primeFactors(n):
    ret = ''
    for i in range(2, n + 1):
        num = 0
        while(n % i == 0):
            num += 1
            n /= i
        if num > 0:
            ret += '({}{})'.format(i, '**%d' % num if num > 1 else '')
        if n == 1:
            return ret

def primeFactors(n):
    i, j, p = 2, 0, []
    while n > 1:
        while n % i == 0: n, j = n / i, j + 1
        if j > 0: p.append([i,j])
        i, j = i + 1, 0
    return ''.join('(%d' %q[0] + ('**%d' %q[1]) * (q[1] > 1) + ')' for q in p)
def primeFactors(n):
    i = 2
    r = ''
    while n != 1:
        k = 0
        while n%i == 0:
            n = n / i
            k += 1
        if k == 1:
            r = r + '(' + str(i) + ')'
        elif k == 0: pass
        else:
            r = r + '(' + str(i) + '**' + str(k) + ')'
        i += 1
        
    return r
        

def primeFactors(n):
  result = ''
  fac = 2
  while fac <= n:
    count = 0
    while n % fac == 0:
      n /= fac
      count += 1
    if count:
      result += '(%d%s)' % (fac, '**%d' % count if count > 1 else '')
    fac += 1
  return result

import math
from itertools import count
from collections import Counter

#Prime generator based on https://stackoverflow.com/questions/2211990/how-to-implement-an-efficient-infinite-generator-of-prime-numbers-in-python/10733621#10733621
def genPrimes():
    #initial primes
    yield from [2,3,5]
    gen = genPrimes()
    """Store count generators starting from the next base prime's square
    incrementing by two times the last prime number. This is for tracking the multiples."""
    mults_set = {}
    prime = next(gen)
    prime_sq = prime ** 2
    for i in count(3, 2):
        #if i is a multiple of a prime...
        if i in mults_set:
            mults = mults_set.pop(i)
        
        #if i is the next prime...
        elif i < prime_sq:
            yield i
            continue
            
        #else i is the next primes square
        else:
            mults = count(prime_sq+2*prime, 2*prime)
            prime = next(gen)
            prime_sq = prime ** 2
        
        #get the next multiple that isnt already in the set
        for mult in mults:
            if mult not in mults_set: break
        
        mults_set[mult] = mults
        
def primeFactors(n):
    #track count of prime
    output = Counter()
    rem_n = n
    """Continue dividing n by it's smallest prime factor, adding each
    factor to the Counter."""
    while rem_n > 1:
        #continue generating primes until one that can divide n is reached
        for prime in genPrimes():
            if rem_n % prime == 0:
                output[prime] += 1
                rem_n /= prime
                break
                
    return "".join(f"({prime}**{count})" if count > 1 else f"({prime})" for prime, count in output.items())
from collections import defaultdict

def primeFactors(n):
    factors = defaultdict(int)
    n_prime = n
    for i in range(2, int(n**0.5)+1):
        while not n_prime % i:
            factors[i] += 1
            n_prime /= i
    if n_prime != 1:
        factors[n_prime] += 1
    f = lambda x, y: '(%d)' % x if y is 1 else '(%d**%d)' % (x,y)
    return ''.join(f(k, factors[k]) for k in sorted(factors))

from collections import Counter

def fac(n):
    maxq = int(n ** .5)
    d, q = 1, n % 2 == 0 and 2 or 3
    while q <= maxq and n % q != 0:
        q = 1 + d*4 - d//2*2
        d += 1
    res = Counter()
    if q <= maxq:
        res += fac(n//q)
        res += fac(q)
    else: res[n] = 1
    return res

def primeFactors(n):
    return ''.join(('({})' if m == 1 else '({}**{})')
        .format(p, m) for p, m in sorted(fac(n).items()))
def primeFactors(n):
    result = ""
    for k, v in factorize(n).items():
        result += f'({k})' if v == 1 else f'({k}**{v})'
    return result


def factorize(n):
    result, i = {}, 2
    while n >= i ** 2:
        if n % i == 0:
            result[i] = 1 if not result.__contains__(i) else result[i] + 1
            n = n // i
        else:
            i += 1
    result[n] = 1 if not result.__contains__(n) else result[n] + 1
    return result
def primeFactors(n):
    ret = ''
    for i in range(2, int(n **0.5)+1):
        num = 0
        while(n % i == 0):
            num += 1
            n /= i
        if num > 0:
            ret += f'({i}**{num})' if num > 1 else f'({i})'
        if n == 1:
            return ret
        
    return ret + f'({int(n)})'
def primeFactors(n):
    r = ""
    pf = 1
    while n>1:
        pf += 1
        num = 0
        while n%pf == 0:
            n/=pf
            num+=1
        if num>0:
            r += "("+str(pf)+"**"+str(num)+")"
    r=r.replace("**1","")
    return r

from itertools import compress
def sieve(n):
    out = [True] * ((n-2)//2)
    for i in range(3,int(n**0.5)+1,2):
        if out[(i-3)//2]:
            out[(i**2-3)//2::i] = [False] * ((n-i**2-1)//(2*i)+1)
    return [2] + list(compress(range(3,n,2), out))
def primeFactors(n):
    str_lst = []
    for i in sieve(int(n**0.5)+1):
        if n%i==0:
            str_lst.extend(["(", str(i)])
            counter=0
            while n%i==0:
                n=n//i
                counter+=1
            if counter==1:
                str_lst.append(")")
            else:
                str_lst.extend(["**",str(counter),")"])
    if n!=1:
        str_lst.extend(["(",str(n),")"])
    return "".join(str_lst)
import math
from collections import Counter
def primeFactors(n): 
    total = []
    while n % 2 == 0: 
        total.append(2)
        n = n // 2
          
    for i in range(3,int(math.sqrt(n))+1,2): 
          

        while n % i== 0: 
            total.append(i)
            n //= i 
              
    if n > 2: 
        total.append(n)
    return ''.join([f'({key}**{value})' if value > 1 else f'({key})' for key, value in Counter(total).items()])
def factors(n): 
    res = [] 
    while n%2 == 0:
        res.append(2)
        n/=2
    for i in range(3,int(n**.5)+1,2): 
        while n%i == 0:
            res.append(int(i)) 
            n/= i
    if n>2:
        res.append(int(n))
    return res
def primeFactors(n):
    factor = dict() 
    for i in factors(n): 
        if i not in factor:
            factor[i] = 1 
        else:
            factor[i] += 1
    res = ""
    for a,b in factor.items(): 
        if b == 1:
            res += f'({a})'
        else:
            res += f'({a}**{b})'
    return res
def primeFactors(n):
  s = {k for k in range(2,int(abs(n)**0.5)+1) for i in range(2,int(abs(k)**0.5)+1) if k%i==0}   
  p = [j for j in range(2,int(abs(n)**0.5)+1) if j not in s]
  z = [i for i in f_idx(n,p) if i!=1]
  return ''.join([f"({i}**{z.count(i)})" if z.count(i)>1 else f"({i})" for i in sorted(set(z))])

def f_idx(n, p):
    from functools import reduce
    e = list(filter(lambda t: n%t==0, p)) 
    a = reduce(lambda x,y: x*y, e) if e!=[] else 1
    return e+[n] if a==1 else e + f_idx(n//a, e)
def primeFactors(n):
    if isinstance(((n+1)/6),int) or isinstance(((n-1)/6),int): return n
    g=n
    pf=''
    prime=2
    while g!=1:
        count=0
        if (prime-1)%6==0 or (prime+1)%6==0 or prime==2 or prime==3:
            while g%prime==0:
                g=g/prime
                count+=1
            if count>1:
                pf=pf+'('+str(prime)+'**'+str(count)+')'
            elif count>0:
                pf=pf+'('+str(prime)+')'
        prime+=1
    return pf
from collections import Counter

        
def get_factors(num):
    if num % 2 == 0:
        return (2, num // 2)
    for i in range(3, num // 2, 2):
        if num % i == 0:
            return (i, num // i)
    return (num, None)


def string_builder(count_dict):
    s = ""
    for element in count_dict:
        if count_dict[element] == 1:
            s += f"({element})"
        else:
            s += f"({element}**{count_dict[element]})"
    return s


def primeFactors(n):
    factors = [n]
    count_dict = Counter()
    
    while factors:
        result = get_factors(factors.pop())
        count_dict[result[0]] += 1
        if result[1]:
            factors.append(result[1])
    
    return string_builder(count_dict)

from collections import Counter

def primeFactors(n):
    c = Counter()
    while n % 2 == 0:
        c[2] += 1
        n //= 2

    d = 3
    while n > 1:
        while n % d == 0:
            c[d] += 1
            n //= d
        d += 2
        
    return ''.join(f'({key}**{value})' if value > 1 else f'({key})' for key, value in sorted(c.items()))
def primeFactors(n):
    a=[]
    i=2
    while i*i<=n:
        while n%i==0:
            a.append(i)
            n//=i
        i+=1
    if n!=1:
        a.append(n)
    s=[i for i in set(a)]
    s.sort()
    return ''.join('({})'.format(i) if a.count(i)==1 else '({}**{})'.format(i,a.count(i)) for i in s ) 
from itertools import count

def primeFactors(n):
    def power(x, n, res=""):
        i = 0
        while not n%x: n, i = n//x, i+1
        return n, res+(f"({x})" if i==1 else f"({x}**{i})" if i>1 else "")
    n, res = power(2, n)
    for x in count(3, 2):
        if n == 1: return res
        n, res = power(x, n, res)
from collections import Counter

def primeFactors(number):
    ret, i = [], 2
    while i <= number:
        if number % i == 0:
            ret.append(i)
            number = number // i
            continue
        if i is not 2:
            i += 2
        else:
            i += 1
    count = Counter(ret)
    ret_string = []
    for key in sorted(count):
        ret_string.append('({})'.format(key)) if count[key] == 1 else ret_string.append('({}**{})'.format(key, count[key]))

    return ''.join(ret_string)
def primeFactors(n):
    c = 0
    m = 2
    r = []
    while(True):
        if n == 1:
            r.append('(' + str(m) + ')')
            return ''.join(r)
        if n % m == 0:
            n = n / m
            c += 1
        else:
            if c != 0 and c != 1:
                r.append('(' + str(m) + '**' + str(c) + ')')  
            if c == 1:
                r.append('(' + str(m) + ')')
            m += 1
            c = 0
            

def primeFactors(n):
    ret,p,k="",2,0
    while (p<=n):
          while (n%p==0):
                n=n//p
                k+=1
          if (k):
              ret=ret+"("+str(p)+(("**"+str(k)) if k>1 else "")+")"
              k=0
          p+=1+(p>2)
    return(ret)
def primeFactors(n):
    factors=[]
    n1=n
    while n1%2 == 0:
        n1=n1/2
        factors.append(2)
    for i in range (3, int(n1**(1/2.0))):
        while n1%i == 0:
            n1=n1/i
            factors.append(i)
    if factors==[]:
        factors.append(n)
    elif n1 != 1:
        factors.append(int(n1))
    factors.append(0)
    x=0
    l=''
    final=""
    for a in factors:
        if l=='':
            l=a
            x=x+1
        elif l != a:
            if x==1:
                final=final+"("+str(l)+")"
                l=a
                x=1
            else:
                final=final+"("+str(l)+"**"+str(x)+")"
                l=a
                x=1
        else:
            x=x+1
    return final
            

from collections import defaultdict


def is_prime(a):
    return all(a % i for i in range(2, a))


def primeFactors(n):
    factors = defaultdict(int)
    rest = n
    while rest != 1:
        for num in range(2, rest+1):
            if rest % num == 0 and is_prime(num):
                factors[num] += 1
                rest //= num
                break

    return ''.join(
        map(
            lambda nc: '({}**{})'.format(nc[0], nc[1]) if nc[1] > 1 else '({})'.format(nc[0]),
            sorted(factors.items(), key=lambda x: x[0])
        )
    )
def primeFactors(n):
        res = ""
        fac = 2
        while fac <= n:
            count = 0
            while n % fac == 0:
                count += 1
                n = n / fac
            if count > 0:
                res += "(" + str(fac)
                res +=  "**"  + str(count) if (count > 1) else "" 
                res += ")"
            fac += 1
        return res

def primeFactors(n):
    i = 2
    j = 0
    out = ''
    while n > 1:
      if n % i == 0:
          n //= i
          j += 1
      else:
          if j == 1:
              out += '({})'.format(i)
          elif j > 1:
              out += '({}**{})'.format(i, j)
          i += 1
          j = 0
    return out + '({})'.format(i)
def primeFactors(n):
    div = 2
    k = 0
    s = ''
    while div < n:
        while n%div!=0:
            div +=1
        while n%div ==0:
            n=n//div
            k+=1
        s+='({}{})'.format(str(div), '**'+ str(k) if k > 1 else '')
        k = 0
    return s
import math


def primeFactors(n):
    primes = {}
    while n % 2 == 0:
        n //= 2
        try:
            primes[2] += 1
        except:
            primes[2] = 1
    for i in range(3, int(math.sqrt(n)) + 1, 2):
        while n % i == 0:
            n //= i
            try:
                primes[i] += 1
            except:
                primes[i] = 1
    if n != 1:
        primes[n] = 1
    primes = sorted(list(primes.items()), key=lambda k: k[0])
    return "".join(
        [
            "({}**{})".format(k, v) if v > 1 else "({})".format(k)
            for k, v in primes
        ]
    )



def primeFactors(n):
    power = {}
    div = 2
    while div <= n:
        if n % div == 0:
            if div in power:
                power[div] += 1
            else:
                power[div] = 1
            n /= div
        else:
            div +=1
    return ''.join([f"({k}**{v})" if v > 1 else f"({k})" for k, v in power.items()])
import math

def primeFactors(n):
    # returns the maximum prime factor
    max = maxPrimeFactors(n)
    
    # factorize n and stores factors in facts
    facts = []
    for i in range(2, int(max + 1)):
        while n % i == 0:
            facts.append(i)
            n = n/i
    
    # removes duplicates for ret and sorts the list
    facts_nodup = list(set(facts))
    facts_nodup.sort()
    # formats return string
    ret_str = ""
    for x in facts_nodup:
        count = facts.count(x)
        if count > 1:
            ret_str += "({}**{})".format(x, count)
        else:
             ret_str += "({})".format(x)
             
    return ret_str
    
# A function to find largest prime factor    
def maxPrimeFactors(n): 
      
    # Initialize the maximum prime factor 
    # variable with the lowest one 
    maxPrime = -1
      
    # Print the number of 2s that divide n 
    while n % 2 == 0: 
        maxPrime = 2
        n >>= 1     # equivalent to n /= 2 
          
    # n must be odd at this point,  
    # thus skip the even numbers and  
    # iterate only for odd integers 
    for i in range(3, int(math.sqrt(n)) + 1, 2): 
        while n % i == 0: 
            maxPrime = i 
            n = n / i 
      
    # This condition is to handle the  
    # case when n is a prime number  
    # greater than 2 
    if n > 2: 
        maxPrime = n 
      
    return int(maxPrime) 

import math
def primeFactors(n):
    ar=[]
    while n%2==0: 
        ar.append(2) 
        n=n/2 
    for i in range(3,int(math.sqrt(n))+1,2):
        while n%i==0: 
            ar.append(i) 
            n=n/i
    if n>2: 
        ar.append(n)
    ax='**'
    x=''
    for i in sorted(set(ar)):
        c=ar.count(i)
        if c>1:
            x+='('+str(i)+ax+str(c)+')'
        else: x+='('+str(int(i))+')'
    return x
def primeFactors(n):
    i=2
    li=[]
    s=""
    while n!=1:
        count=0
        while n%i==0:
          n=n/i
          count=count+1
        if count!=0:
            re=(i,count)
            li.append(re)
        i=i+1
    for i in li:
        if i[1]!=1:
            s=s+"("+str(i[0])+"**"+str(i[1])+")"
        else:
            s=s+"("+str(i[0])+")"
    return s
def primeFactors(n):
    comp=[]
    i=1
    o=0
    while n!=1:
        i=i+1
        if n%i==0:
            o=o+1
            while n%i==0:
                n=n/i
                comp.append(i)
    if o==0:
        return '('+str(n)+')'
    g=['('+str(x)+ '**'+str(comp.count(x))+')' if comp.count(x)>1 else '('+str(x)+')' for x in sorted(list(set(comp))) ]
    return ''.join(g)
                

from collections import OrderedDict
from math import sqrt


class OrderedIntDict(OrderedDict):
    def __missing__(self, key):
        return 0


def format_factor(n, times):
    return (
        "({n})".format(n=n) if times == 1
        else "({n}**{times})".format(n=n, times=times)
    )


def prime_factors(number):
    factors = OrderedIntDict()
    for n in range(2, int(sqrt(number))+1):
        while number % n == 0:
            number //= n
            factors[n] += 1
    if number > 1:
        factors[number] = 1
    return "".join(
        format_factor(n, times)
        for n, times in factors.items()
    )
    

primeFactors = prime_factors
def primeFactors(n):
    factors = []
    p = 2 
    while n != 1 and p <= n**0.5:
        if n % p == 0:
            factors.append(p)
            n = n // p
        else:
            p = p + 1
    if n != 1:
        factors.append(n)
        n = 1
    
    distinct = sorted(set(factors))
    
    answer = ""
    for prime in distinct:
        if factors.count(prime) == 1:
            answer = answer + "(" + str(prime) + ")"
        else:
            answer = answer + "(" + str(prime) + "**" + str(factors.count(prime)) + ")"
            
    return answer
import math

def primeFactors(n:int)-> str:
    list_result = []
    while n % 2 == 0: 
        n = n / 2
        list_result.append(2)      
    for i in range(3,int(math.sqrt(n))+1,2):           
        while n % i== 0: 
            n = n // i
            list_result.append(i)      
    if n > 2: 
        list_result.append(int(n))
            
    return  ''.join([f"({i})" if list_result.count(i) == 1 else f"({i}**{list_result.count(i)})" for i in sorted(list(set(list_result)))])


# Using the sqrt function for speed (instead of **0.5)
from math import sqrt
def seiveOfEratosthenes(n):
    # For speed, only grabbing all odd numbers from 3 to SQRT N
    flags = [True for i in range(3, int(sqrt(n))+1, 2)]
    # Adding a flag for "2" (effectively a pre-pend here)
    flags.append(True)
    
    # Iterate through Primes
    prime = 2
    while (prime <= sqrt(n)):
        # Cross Off all multiples of Prime
        for i in range(prime**2, len(flags), prime):
            flags[i] = False
        
        # Get next Prime
        prime += 1
        while (prime < len(flags) and not flags[prime]):
            prime += 1
    # Get the list of Primes as actual #'s; We need a special case for "2" because it doesn't fit the odds pattern
    if flags[0]:
        primes = [2]
    else:
        primes = []
    primes += [i for i in range(3, len(flags), 2) if flags[i]]
    
    return primes
        
    

def primeFactors(n):
    # Get applicable Prime numbers
    primes = seiveOfEratosthenes(n)
    primes_in_number = []
    
    for prime in primes:
        # Iterate through each prime, and figure out how many times it goes in
        repeat = 0
        while (not (n % prime)):
            repeat += 1
            n /= prime
        
        # Add appearing Primes appropriately
        if repeat == 1:
            primes_in_number.append("(" + str(prime) + ")")
        elif repeat:
            primes_in_number.append("(" + str(prime) + "**" + str(repeat) + ")")
    
    # Only testing primes up to sqrt of n, so we need to add n if it hasn't been reduced to 1
    if n > 1:
        primes_in_number.append("(" + str(int(n)) + ")")
    
    return ''.join(primes_in_number)
def primeFactors(n):
    s=str()
    for i in range(2, int(n**(1/2))+1):
        j=0
        while n/i%1==0.0:
            j+=1
            n/=i
        if j>1:
            s+="("
            s+=str(i)
            s+="**"
            s+=str(j)
            s+=")"
        if j==1:
            s+="("
            s+=str(i)
            s+=")"
    if n!=1:
        s+="("
        s+=str(int(n))
        s+=")"
        return s         
    else:
        return s        
import math
def primeFactors(number):

    # create an empty list and later I will
    # run a for loop with range() function using the append() method to add elements to the list.
    prime_factors = []

    # First get the number of two's that divide number
    # i.e the number of 2's that are in the factors
    while number % 2 == 0:
        prime_factors.append(2)
        number = number / 2

    # After the above while loop, when number has been
    # divided by all the 2's - so the number must be odd at this point
    # Otherwise it would be perfectly divisible by 2 another time
    # so now that its odd I can skip 2 ( i = i + 2) for each increment
    for i in range(3, int(math.sqrt(number)) + 1, 2):
        while number % i == 0:
            prime_factors.append(int(i))
            number = number / i

    if number > 2:
        prime_factors.append(int(number))
        
    distinct_set = sorted(set(prime_factors))
    
    output = ""
    
    for i in distinct_set:
        
        if(prime_factors.count(i) == 1):
            
            output = output + '(' + str(i) + ')'
        else:
            output = output + '(' + str(i)+ '**' + str(prime_factors.count(i)) +  ')'
        
    return output
def primeFactors(n):
    result = ''
    factor = 2
    while 1:
        count = 0
        while n % factor == 0:
            count += 1
            n = n / factor
        if count == 1 and count != 0:
            result = result + '(%s)' % factor
        elif count != 0:
            result = result + '({}**{})'.format(factor, count)
        else:
            factor += 1
            continue
        factor += 1
        if n == 1:
            break
    return result
        
            
    ...
def primeFactors(n):
    i = 2
    a = []
    while i <= n:
        if n % i == 0:
            if not any([i%x[0]==0 for x in a]):
                max_power = 0
                div = n
                while div%i==0:
                    max_power+=1
                    div/=i
                a.append((i, max_power))
                n/=i
        i+=1
    return ''.join([f'({x}**{y})' if y!=1 else f'({x})' for x,y in a ])
import collections
def primeFactors(n):
    ret = []
    pows = {}
    st = ""
    i = 2
    while(i<n):
        if n%i == 0:
            n /= i
            ret.append(i)
        else:
            i += 1
    ret.append(int(n))
    for j in set(ret):
        pows.update({int(j):ret.count(j)})
    print(pows)
    pows = collections.OrderedDict(sorted(pows.items()))
    for key in pows:
        if pows[key] > 1:
            st += "("+ str(key)+"**"+ str(pows[key])+")"
        else:
            st += "("+str(key)+")"
    return st
def primeFactors(x):
    primes = {}
    prime_int = 2
    y = x
    while prime_int < x+1:
        while x % prime_int == 0:
            if prime_int not in primes:
                primes[prime_int] = 1 
            else:
                primes[prime_int] += 1 
            x  //= prime_int
        prime_int += 1
    phrase = ''
    for digit,count in primes.items():
        if count == 1:
            phrase += f'({digit})'
            continue
        phrase += f'({digit}**{count})'
    print(phrase)
    return phrase
def primeFactors(n):
    i = 2
    factors = {}
    while i * i <= n:
        if n % i:
            i += 1
        else:
            n //= i
            if(i in factors.keys()):
                factors[i]+=1
            else:
                factors[i]=1
    if n > 1:
        factors[n]=1
    string=""
    for key in factors.keys():
        if factors[key]==1:
            string+="({})".format(key)
        else:
            string+="({}**{})".format(key,factors[key])
    return string
def primeFactors(n):
    def is_prime(number):
        if number == 1: return False
        for num in range(3,(int)(number**.5) + 1,2):
            if number % num == 0:
                return False
        return True
    primes_prelimenary = []
    while n > 1:
        if is_prime(n):
            primes_prelimenary.append(n)
            n = 1
            continue
        while n % 2 == 0:
            primes_prelimenary.append(2)
            n = n / 2
        for num in range(3,(int)(n**.5) + 1,2):
            if n % num == 0 and is_prime(num): 
                while n % num == 0:
                    primes_prelimenary.append(num)
                    n = n / num
    return ''.join(f'({(int)(factor)}**{primes_prelimenary.count(factor)})' for factor in sorted(set(primes_prelimenary))).replace('**1','')
def primeFactors(n):
    list = []
    for i in range(2, round(n**0.5)):
        while (n/i).is_integer():
            n /= i
            list.append(i)
    if len(list) < 2:
        list.append(int(n))
    list_seen = []
    str1 = ''
    for x in list:
        if x in list_seen:
            pass
        else:
            list_seen.append(x)
            if list.count(x) > 1:
                str1 += f"({str(x)}**{str(list.count(x))})"
            else:
                str1 += "(" + str(x) + ")"
    return str1
import math

def primeFactors(n):
    dic_prime = {}
    dic_prime[2] = 0
    while n%2 ==0:
        dic_prime[2] += 1
        n = n/2
    if dic_prime[2] == 0:
        dic_prime.pop(2)

    for i in range(3,int(n+1),2):
        dic_prime[i] = 0
        while n%i ==0:
            dic_prime[i] += 1
            n = n/i
        if n <= 1:
            break
        if dic_prime[i] == 0:
            dic_prime.pop(i)

    output_str = ""
    for k,v in dic_prime.items():
        if v == 1:
            output_str += "({})".format(str(k))
        else:
            output_str +="({}**{})".format(str(k),str(v))

    return output_str
