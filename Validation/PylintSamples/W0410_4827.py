def stats_disc_distr(distrib):
    err = check_errors(distrib)
    if not err:
        mean = sum(x[0] * x[1] for x in distrib)
        var = sum((x[0] - mean) ** 2 * x[1] for x in distrib)
        std_dev = var ** 0.5
    return [mean, var, std_dev] if not err else err

def check_errors(distrib):
    errors = 0
    if not isclose(sum(x[1] for x in distrib), 1):
        errors += 1
    if not all(isinstance(x[0], int) for x in distrib):
        errors += 2
    if errors > 0:
        return {1: "It's not a valid distribution", 2: "All the variable values should be integers",
        3: "It's not a valid distribution and furthermore, one or more variable value are not integers"}[errors]

def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

def stats_disc_distr(distrib):
    dist = abs(sum(px for x, px in distrib) - 1) < 0.00001
    all_int = all(isinstance(x, int) for x, px in distrib)
    if not dist:
        if not all_int:
            return "It's not a valid distribution and furthermore, one or more variable value are not integers"
        return "It's not a valid distribution"
    if not all_int:
        return "All the variable values should be integers"
    u = sum(x * px for x, px in distrib)
    o2 = sum(abs((x - u) ** 2 * px) for x, px in distrib)
    o = o2 ** 0.5
    return [u, o2, o]
def stats_disc_distr(arr):
    a = round(sum(float(i[1]) for i in arr),2) != 1
    b = not all(int(i[0])==i[0] for i in arr)
    if a and b : return "It's not a valid distribution and furthermore, one or more variable value are not integers"
    if a :       return "It's not a valid distribution"
    if b :       return "All the variable values should be integers"
    _mean = sum(i[0]*i[1] for i in arr)
    _var = sum(((i[0]-_mean)**2)*i[1] for i in arr)
    _sdt = _var ** .5
    return [_mean, _var, _sdt]
from __future__ import division
import numpy as np
def stats_disc_distr(distrib):
    print(distrib)
    values = [i[0] for i in distrib]
    probs = [i[1] for i in distrib]
    
    
    if any(type(i) != int for i in values) and float(sum(probs)) != 1.0:
        return "It's not a valid distribution and furthermore, one or more variable value are not integers"
    elif sum(probs) > 1.0 + 1e-5 or sum(probs) < 1.0 - 1e-5:
        print(sum(probs))
        return "It's not a valid distribution"
    elif any(type(i) != int for i in values):
        return "All the variable values should be integers"
        
    mean = np.mean(values)
    var = sum([(values[i] - mean)**2 * probs[i] for i in range(len(values))])
    std_dev = var**0.5
    
    
    
    
    
    
    
    return [mean, var, std_dev] # or alert messages
def stats_disc_distr(distrib):
    values, probs = zip(*distrib)
    probs_is_not_one = (abs(sum(probs) - 1) > 1e-8)
    values_are_not_ints = any(value % 1 for value in values)
    if probs_is_not_one and values_are_not_ints:
        return "It's not a valid distribution and furthermore, one or more variable value are not integers"
    elif values_are_not_ints:
        return "All the variable values should be integers"
    elif probs_is_not_one:
        return "It's not a valid distribution"
    mean = sum((value * prob) for value, prob in distrib)
    var = sum(((value - mean)**2 * prob) for value, prob in distrib)
    std_dev = var**0.5
    return [mean, var, std_dev]
def stats_disc_distr(distrib):
    is_valid_distribution = lambda d: abs(sum(px for x, px in d) - 1) < 1e-8
    are_events_integers = lambda d: all(isinstance(x, (int, float)) and float(x).is_integer() for x, px in d)
    events, probabilities = are_events_integers(distrib), is_valid_distribution(distrib)
    if not events and not probabilities:
        return "It's not a valid distribution and furthermore, one or more variable value are not integers"
    elif not probabilities:
        return "It's not a valid distribution"
    elif not events:
        return "All the variable values should be integers"
    mean = sum(x * px for x, px in distrib)
    var = sum((x - mean) ** 2 * px for x, px in distrib)
    std_dev = var ** .5
    return [mean, var, std_dev]
from math import sqrt

def stats_disc_distr(distrib):
    if round(sum(count[1] for count in distrib), 3) != 1:
        if not all(isinstance(count[0], int) for count in distrib):
            return "It's not a valid distribution and furthermore, one or more variable value are not integers"
        else:
            return "It's not a valid distribution"
    
    if not all(isinstance(count[0], int) for count in distrib):
        return "All the variable values should be integers"
    
    expected = sum(event * prob for event, prob in distrib)
    variance = sum((event - expected)**2 * prob for event, prob in distrib)
    standard = sqrt(variance)
    
    return [expected, variance, standard]
stats_disc_distr=lambda d: (lambda m: "It's not a valid distribution and furthermore, one or more variable value are not integers" if round(sum(k for i,k in d),8)!=1 and any(type(i)!=int for i,k in d) else "It's not a valid distribution" if round(sum(k for i,k in d),8)!=1 else "All the variable values should be integers" if any(type(i)!=int for i,k in d) else [m, sum(k*(i-m)**2 for i,k in d), sum(k*(i-m)**2 for i,k in d)**0.5])(sum(i*k for i,k in d))
def stats_disc_distr(distrib):
    print(sum(p for x, p in distrib))
    if abs(sum(p for x, p in distrib) - 1) > 1e-4:
        if not all(x % 1 == 0 for x, p in distrib):
            return "It's not a valid distribution and furthermore, one or more variable value are not integers"
        return "It's not a valid distribution"
    if not all(x % 1 == 0 for x, p in distrib):
        return "All the variable values should be integers"
    
    
    mean = sum(x * p for x, p in distrib)
    var = sum((x - mean) ** 2 * p for x, p in distrib)
    std_dev = var ** 0.5
    return [mean, var, std_dev]

import math

def stats_disc_distr(disv):
    m=0; v=0; p=0; ni=False
    for dis in disv:
       if type(dis[0])!=int: ni=True
       p=p+dis[1]; m=m+dis[0]*dis[1] 
    if ni==False:   
        for dis in disv: v+=(dis[0]-m)*(dis[0]-m)*dis[1]
    if ni:
       return "It's not a valid distribution and furthermore, one or more variable value are not integers" if math.fabs(p-1)>1e-10 else "All the variable values should be integers"   
    if math.fabs(p-1)>1e-10: return "It's not a valid distribution"
    return [m, v, v**.5]
