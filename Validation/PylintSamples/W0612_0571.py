#! /usr/bin/env python

from sys import stdin
from functools import reduce

def gcd(a,b):
	while b!=0:
		a,b=b,a%b
	return a
	
def gcdl(l):
	return reduce(gcd, l[1:],l[0])

def __starting_point():
	T=int(stdin.readline())
	for case in range(T):
		numbers=list(map(int, stdin.readline().split()[1:]))
		g=gcdl(numbers)
		
		numbers=[n/g for n in numbers]
		print(" ".join([str(x) for x in numbers]))

__starting_point()
