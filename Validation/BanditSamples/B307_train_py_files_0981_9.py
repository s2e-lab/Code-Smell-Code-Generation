T = eval(input())

def solve(A):
 A.sort()
 best = 10000000000
 for i in range(len(A) - 1):
  diff = abs(A[i] - A[i+1])
  if diff < best:
   best = diff
 return best

for i in range(T):
 N = eval(input())
 A = list(map(int, input().split()))
 print(solve(A))

