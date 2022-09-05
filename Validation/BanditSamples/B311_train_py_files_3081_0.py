import random
def squares(n):
    return [i**2 for i in range(1, n+1)]

def num_range(n, start, step):
    return [i for i in range(start, start+step*n, step)]

def rand_range(n, mn, mx):
    return [random.randint(mn, mx) for i in range(n)]
def primes(n):
    return [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113][:n]
