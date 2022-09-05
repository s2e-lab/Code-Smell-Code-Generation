from fractions import gcd
import re


INSERTER = re.compile(r'(?<!d)(?=[xyt])')
FINDER   = re.compile(r'-?d+')


def lcm(a,b):    return a*b//gcd(a,b)
def simplify(s): return INSERTER.sub('1', s.replace(' ',''))


def para_to_rect(*equations):
    coefs = [ list(map(int, FINDER.findall(eq))) for eq in map(simplify, equations) ]
    l     = lcm(coefs[0][1],coefs[1][1])
    x,tx,cx, y,ty,cy = ( v*l//c[1] for c in coefs for v in c )
    y, absY, c = -y, abs(y), cx-cy
    
    return "{}x {} {}y = {}".format(x if x!=1 else '',
                                    '-' if y<0 else '+',
                                    absY if absY!=1 else '',
                                    c)
from math import gcd

def para_to_rect(eqn1, eqn2):
    a, b = eqn1.split('= ')[1].split('t ')
    c, d = eqn2.split('= ')[1].split('t ')
    if a in ("", "-"): a += '1'
    if c in ("", "-"): c += '1'
    a, b, c, d = map(eval, (a, b, c, d))
    x = gcd(a, c)
    e, f = c//x, -a//x
    if e < 0: e, f = -e, -f
    return f"{e if e>1 else ''}x {'+-'[f<0]} {abs(f) if abs(f)>1 else ''}y = {e*b + f*d}"
import re
from math import gcd

def para_to_rect(eqn1, eqn2):
    eqn1, eqn2 = [re.sub(r'bt', '1t', e) for e in [eqn1, eqn2]]
    (a,b), (c,d) = ([[int(x.replace(' ', '')) for x in re.findall('-?d+|[-+] d+', e)] for e in [eqn1, eqn2]])
    x = c*b - a*d
    g = gcd(gcd(c, a), x)
    if c < 0:
        g = -g
    c, a, x = c//g, a//g, x//g
    return re.sub(r'b1([xy])', r'1', f'{c}x {"+-"[a > 0]} {abs(a)}y = {x}')
from fractions import gcd
def para_to_rect(*equations):
    changes = [(" ", ""), ("-t", "-1t"), ("=t", "=+1t"),
               ("+t", "+1t"), ("x=", ""), ("y=", "")]
    equationsR = []
    for equation in equations:
        for (s1, s2) in changes:
            equation = equation.replace(s1, s2)
        equationsR += equation.split("t")
    a, b, c, d = [int(n) for n in equationsR]
    e, f, g = c, -a, b * c - a * d
    h = gcd(gcd(e, f), g)
    e, f, g = e // h, f // h, g // h
    if e < 0:
        e, f, g = -e, -f, -g
    ysign = "+"
    if f < 0:
        ysign, f = "-", -f
    return "{}x {} {}y = {}".format(e if abs(e) > 1 else "-" if e == -1 else "",
      ysign, f if f > 1 else "", g)
def extract(eq):
    k, b = eq.split('t')
    k = k[3:].strip()
    k = (int(k) if k!='-' else -1) if k else 1
    b = b.strip()
    b = (-1 if b[0]=='-' else 1)*int(b[1:].strip()) if b else 0
    return k, b

def quotient(x):
    return '' if x==1 else '-' if x==-1 else str(x)

from math import gcd

def para_to_rect(eqn1, eqn2):
    a,b = extract(eqn1)
    k,d = extract(eqn2)
    l = -a
    m = k*b-a*d
    g = gcd(gcd(k,l),m)
    if k*g<0:
        g = -g
    k //= g
    l //= g
    m //= g
    return f'{quotient(k)}x {"+-"[l<0]} {quotient(abs(l))}y = {m}'

def gcd(x,y):
    while y:
        x,y=y,x%y
    return x
    
def parseeq(s):
    s=s.split()
    a=s[2].replace('t','')
    if a=='': a=1
    elif a=='-': a=-1
    else: a=int(a)
    try:
        b=int(s[3]+s[4])
    except:
        b=0
    return a,b

def para_to_rect(eqn1, eqn2):
    a,b=parseeq(eqn1)
    c,d=parseeq(eqn2)
    e=b*c-a*d
    if c<0: a,c,e = -a,-c,-e
    g=gcd(a,gcd(abs(c),abs(e)))
    a,c,e=a//g,c//g,e//g
    sign='+-'[a>0]
    if c==1: p1=''
    elif c=='-1': p1='-'
    else: p1=c
    if a==1: p2=''
    elif a==-1: p2=''
    else: p2=abs(a)
    
    return '{}x {} {}y = {}'.format(p1,sign,p2,e)
    
    

import math
import re
def lcm(a, b):
    return abs(a*b) // math.gcd(a, b)
def para_to_rect(eqn1, eqn2):
    try:    
        x_t = int(re.findall("-?d+?t", eqn1.replace('-t', '-1t'))[0].split('t')[0])
    except:
        x_t = 1
    try:
        y_t = int(re.findall("-?d+?t", eqn2.replace('-t', '-1t'))[0].split('t')[0])
    except:
        y_t = 1
    
    l = lcm(x_t, y_t)
    x_n = abs(l//x_t) * int(re.findall('-?d+$', eqn1.replace(" ", ""))[0])
    y_n = abs(l//y_t) * int(re.findall('-?d+$', eqn2.replace(" ", ""))[0])
       
    x, y = l//abs(x_t), l//abs(y_t)
    
    if((x_t * x) + (y_t * y) == 0):    
        return '{}x + {}y = {}'.format(x if x!=1 else '', y if y!=1 else '', x_n+y_n)
    
    return '{}x - {}y = {}'.format(x if x not in [1, -1] else '', y if y not in [1, -1] else '', x_n-y_n)
import re

EQUATION_REGEXP = re.compile(r'^[xy]=(-?d*)t([+-]d+)$')


def parse_coefficient(raw_coef):
    if not raw_coef:
        return 1
    elif raw_coef == '-':
        return -1
    return int(raw_coef)


def parse(equation):
    equation = equation.replace(' ', '')
    coefficients = EQUATION_REGEXP.match(equation).groups()
    return list(map(parse_coefficient, coefficients))


def gcd(a, b):
    return gcd(b, a % b) if b else a


def lcm(a, b):
    return abs(a * b) / gcd(a, b)


def compile_result(mult_a, mult_b, coefs_a, coefs_b):
    multiplier = -1 if mult_a < 0 else 1

    A = mult_a * multiplier
    A = A if A != 1 else ''   
    
    B = abs(mult_b) if abs(mult_b) != 1 else ''
    B_sign = '-' if multiplier * mult_b > 0 else '+'
    
    C = multiplier * (mult_a * coefs_a[1] - mult_b * coefs_b[1])

    return f'{A}x {B_sign} {B}y = {C}'


def para_to_rect(equation_a, equation_b):
    coefs_a = parse(equation_a)
    coefs_b = parse(equation_b)
    parameter_lcm = int(lcm(coefs_a[0], coefs_b[0]))
    mult_a = int(parameter_lcm / coefs_a[0])
    mult_b = int(parameter_lcm / coefs_b[0])
    return compile_result(mult_a, mult_b, coefs_a, coefs_b)
    

import re
import math

def para_to_rect(eqn1, eqn2):
    a = re.search(r'(-?d*)(?=t)', eqn1)
    if a is None:
        a = 1
    else:
        if a.group(0) == '':
            a = 1
        elif a.group(0) == '-':
            a = -1
        else:
            a = int(a.group(0))
    c = re.search(r'(-?d*)(?=t)', eqn2)
    if c is None:
        c = 1
    else:
        if c.group(0) == '':
            c = 1
        elif c.group(0) == '-':
            c = -1
        else:
            c = int(c.group(0))
    b = re.search(r'[-+]? d*Z', eqn1)
    if b is None:
        b = 0
    else:
        b = int(b.group(0).replace(' ', ''))
    d = re.search(r'[-+]? d*Z', eqn2)
    if b is None:
        d = 0
    else:
        d = int(d.group(0).replace(' ', ''))
    n = (a * c) // math.gcd(a, c)
    k = (-1 if c < 0 else 1)
    x =  k * n // a
    y =  -k * n // c
    z = k * b * n // a - k * d * n // c
    xp = '' if x == 0 else '{}x'.format('-' if x == - 1 else '' if x == 1 else abs(x))
    yp = '' if y == 0 else '{}{}y'.format(' - ' if y < 0 else ' + ', '' if abs(y) == 1 else abs(y))
    return '{}{} = {}'.format(xp, yp, z)
from fractions import gcd

def para_to_rect(eqn1, eqn2):

    a1, b1 = coeff(eqn1, 'x')
    a2, b2 = coeff(eqn2, 'y')
    
    A = a2
    B = -a1
    C = b1 * a2 - a1 * b2
    
    g = gcd(gcd(A, B), C)
    
    cf = [v // g for v in [A, B, C]]
    if cf[0] < 0: 
        cf = [-1 * v for v in cf]
    
    s = '+' if cf[1] >= 0 else '-'
    cf[1] = abs(cf[1])
    
    a, b, c = ['' if abs(v) == 1 else str(v) for v in cf] 
    
    return '{}x {} {}y = {}'.format(a, s, b, c)
    
    
def coeff(eq, v): 
    p1 = eq.replace(' ', '').replace(v + '=', '').split('t')
    return list([1 if x == '' else -1 if x == '-' else int(x) for x in p1])

