def treeProduct(num, h, root, ch):
 if ch >= h:
  return num[root]

 left = (root * 2) + 1
 right = (root * 2) + 2

 ret1 = treeProduct(num, h, left, ch + 1)
 ret2 = treeProduct(num, h, right, ch + 1)

 return num[root] * max(ret1, ret2)

def main():
 n = int(input())
 while n!=0:
  line = str(input())
  s = line.split()
  num = [int((e)) for e in s]
  print(int(treeProduct(num,n,0,1)%1000000007))
  n = int(input())
  
 
def __starting_point():
 main()

__starting_point()
a=[]
def maxx(a,b):
 if(a>b):
  return a;
 return b


def fnd(cur,mx):
 if(cur>mx):
  return 1;
 f=fnd(2*cur,mx)
 g=fnd(2*cur+1,mx)
 return (a[cur]*maxx(f,g));


def main():
 while(1):
  n=eval(input());
  if(n==0 ):
   break
  while(len(a)!=0):
   a.pop();
  a.append(0);
  x=input().split();
  mx=2**n -1;
  for i in range(0,2**n-1):
   a.append(int(x[i]));
  print(fnd(1,mx)%1000000007)


main()

import sys

def f(a,b,c,l,d):
 #   print a,b,c
 if a>l:
  return 1
 if b == c:
  return d[a]
 else:
  mid = (b+c)/2
  id1 = f(2*a,b,mid,l,d) 
  id2 = f(2*a+1,mid+1,c,l,d) 
  
  id1 = id1*d[a]
  id2 = id2*d[a]
  
#       print id1,id2

  if id1 > id2:
   return id1
  else:
   return id2
 

t = int(sys.stdin.readline())

while t:
 d = []
 d.append(0)
 x = sys.stdin.readline().split()
 l = len(x)
 for i in range(l):
  d.append(int(x[i]))

 ans = f(1,1,(1<<t)-1,(1<<t)-1,d)
 ans = ans % 1000000007
 print(ans)
 
 t = int(sys.stdin.readline())

import sys

while(1):

 n = int(sys.stdin.readline().strip())
 if (n==0):
  return;

 a=list(map(int,input().split()))

 for i in range((2**(n-1))-2,-1,-1):
  a[i]=max(a[i]*a[2*(i+1)-1],a[i]*a[2*i+2])
 print(a[0]%1000000007)
  
  




def main():
 while(True):
  h = int(input())
  if h==0:
   break
  lineproc = input().split()
  nodes = []
  nodes.append(int(lineproc[0]))
  for i in range(1, int(2**h)-1):
   number = int(lineproc[i])*nodes[int((i+1)/2)-1]
   nodes.append(number)
  print(max(nodes)%1000000007)
main()

while 1:
 N=eval(input())
 if N==0:
  break
 A=list(map(int,input().split(' ')))
 for i in range(N-1,0,-1):
  C=1<<(i-1)
  B=1<<i
  for j in range(C-1,B-1,1):
   A[j]=A[j]*max(A[2*j+1],A[2*j+2])
 print(A[0]%1000000007)

import sys 

#f = open("test.in")
f = sys.stdin

while (True):
 H = int(f.readline())
 if (H == 0):
  break
 
 v = list(map(int, f.readline().split()))
 p = [0] * (2 ** H)

 for i in range(2 ** H - 1, 0, -1):
  if (i * 2 >= 2 ** H):
   p[i] = v[i - 1]
  else:
   p[i] = v[i - 1] * max(p[2 * i], p[2 * i + 1])

 r = p[1] % 1000000007
 print(r)

#! /usr/bin/env python

from math import ceil as ceil

def parent(index):
 return int(ceil(index / 2) - 1);

while True:
 length = int(input());

 if length == 0:
  break;

 numbers = input().split();
 tree = [0] * len(numbers);

 for i in range(0, len(numbers)):
  tree[i] = int(numbers[i]);

 lastIndex = len(tree) - 1;

 if lastIndex & 1:
  tree[parent(lastIndex)] = tree[parent(lastIndex)] * tree[lastIndex];
  --lastIndex;

 for i in range(lastIndex, 0, -2):
  parentIndex = parent(i);
  tree[parentIndex] = max(tree[parentIndex] * tree[i], 
            tree[parentIndex] * tree[i - 1]);

 print((tree[0] % 1000000007));

import math

def Left(i):
 return 2*i

def Right(i):
 return (2*i)+1

def levelOf(x):
 return int( math.floor( (math.log10(x)/math.log10(2))+1) )

def treeProduct(numList, n, i):
 if levelOf(i)==n:
  return numList[i]
 else:
  tpl = treeProduct(numList,n,Left(i))
  tpr = treeProduct(numList,n,Right(i))
  if (tpl>tpr):
   return ( (numList[i]*tpl))
  else:
   return ( (numList[i]*tpr))

def main():
 n = int(input())
 while n!=0:
  line = '-1 '+str(input())
  s = line.split()
  num = [int(e) for e in s]
  print(int(treeProduct(num,n,1)%1000000007))
  n = int(input())
 
def __starting_point():
 main()

__starting_point()
