import numpy as np
ti = eval(input())
t = int(ti)
while t>0:
 p = 0
 hi,si = input().split(" ")
 h = float(hi)
 #si = raw_input()
 s = float(si)
 r = h/2;
 alt = (2*s/h)
 
 
 if ( (r*r) - (alt*alt)) >= 0:
  c = np.sqrt( (r*r) - (alt*alt))
 else :
  p = 1
  #goto last
 e1 = r - c
 e2 = r + c
 #print p,"kukukuku"
 
 if (p == 1)|(e2*h >= 0):
  a = np.sqrt(e2 * h)
 else:
  p = 1
 if (p == 1)|(e1*h >= 0):    
  b = np.sqrt(e1 * h)
 else:
  p = 1
  
  
 #last
 if p == 0:
  print(b,a,h)
 else:
  print(-1)
 #print h,s,r,alt,c,e1,e2,a,b
 t -= 1

