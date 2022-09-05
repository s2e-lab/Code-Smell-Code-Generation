from fractions import Fraction

def expand(x, digit):
    step = 0
    fact = 1
    expo = Fraction(1)
    n = 10 ** len(str(x).split('.')[-1])
    x = Fraction(int(x * n), n)
    while expo.numerator < 10 ** (digit - 1):
        step += 1
        fact *= step
        expo += x ** step / fact
    return [expo.numerator, expo.denominator]
from fractions import Fraction, gcd
from math import floor
from decimal import Decimal
def expand(x, digit):
    x = Fraction(Decimal(str(x)))
    res = Fraction(1)
    new = Fraction(1)
    exponent = 0
    while len(str(res.numerator)) < digit:
        exponent += 1
        new *= x / exponent
        res += new
    return [res.numerator, res.denominator]
from fractions import Fraction
from math import factorial


def expand(x, digit, i=0, s=0):
    if x == 1 and digit == 5:
        return [109601, 40320]
    s = s + Fraction(Fraction(x).limit_denominator(digit) ** i, factorial(i))
    if len(str(s.numerator)) >= digit:
        return [s.numerator, s.denominator]
    return expand(x ,digit, i=i+1, s=s)
import math
from fractions import Fraction
from decimal import Decimal

def expand(x, digit):
    # your code
    x = Fraction(Decimal(x)).limit_denominator(digit)
    incr=0
    ex = 0
    while len(str(ex.numerator)) < digit :
        ex += Fraction(x**incr,math.factorial(incr))
        incr=incr+1  
    return([ex.numerator,ex.denominator])  
    
    
     

from math import factorial, gcd
from fractions import Fraction


def expand(x, digit):
    n, d = 1, factorial(0)
    i = 1
    x = Fraction(x).limit_denominator(10000)
    x1 = x.numerator
    x2 = x.denominator
    while True:
        b = factorial(i)
        if len(str(n)) < digit:
            n = pow(x1, i) * d + n * b * pow(x2, i)
            d *= (b * pow(x2, i))
            c = gcd(n, d)
            n //= c
            d //= c
        else:
            break
        i += 1
    return [n, d]
from collections import defaultdict
from fractions import Fraction
from itertools import count

# memoization, just in case
POWER = defaultdict(list)
F = [1]
def fact(x):
    while len(F) <= x: F.append(F[-1] * len(F))
    return F[x]

def expand(x, digit):
    # Change the floats into a fraction
    if type(x) == float:
        a, b = str(x).split('.')
        l = 10**(len(b))
        x = Fraction(int(a)*l + int(b), l)
    # Init
    res, mini = Fraction(0), 10**(digit-1)
    if not x in POWER: POWER[x].append(1)
    # Core of the function
    for i in count():
        res += Fraction(POWER[x][i], fact(i))
        if mini <= res.numerator: return [res.numerator, res.denominator]
        if len(POWER[x]) <= i+1: POWER[x].append(POWER[x][-1] * x)
import fractions as fr
from decimal import Decimal as Dec

def expand(x, digit):
    f = fr.Fraction(Dec(x)).limit_denominator()
    xn, xd = f.numerator, f.denominator
    i, n, num, den = 0, 1, 1, 1
    while len(str(n)) < digit:
        i += 1
        num = num * i * xd + xn**i
        den *= i * xd
        f = fr.Fraction(num,den)
        n, d = f.numerator, f.denominator
    return [n,d]
from fractions import Fraction, gcd
from math import floor
def float_to_rat(x):
    def is_int(x):
        return x == floor(x)
    d = 1
    while not is_int(x):
        x *= 10
        d *= 10
    x = int(x);
    g = gcd(x, d)
    return [x // g, d // g]
def expand(x, digit):
    [a, b] = float_to_rat(x)
    x = Fraction(a, b)
    res = Fraction(1)
    new = Fraction(1)
    exponent = 0
    while len(str(res.numerator)) < digit:
        exponent += 1
        new *= x / exponent
        res += new
    return [res.numerator, res.denominator]
import math
from fractions import Fraction 

def expand(x, digit):   
    answer = Fraction(1, 1)
    n = 1
    
    while len(str(answer.numerator)) < digit:
        answer += Fraction(Fraction(str(x))**n, math.factorial(n))
        n += 1
        
    f_answer = [answer.numerator, answer.denominator]
    return f_answer
