for i in range(int(input())):
 n=int(input())
 chef=0
 ans=[]
 for i in range(0,n):
  l=list(map(int,input().split()))
  c=l[0]
  if c%2==0:
   for i in range(1,len(l)//2+1):
    chef=chef+l[i]
   continue;
  for i in range(1,len(l)//2):
   chef=chef+l[i]
  ans.append(l[len(l)//2])
 ans.sort(reverse=True)
 for i in range(len(ans)):
  if i%2==0:
   chef=chef+ans[i]
 print(chef)
   
  

for i in range(int(input())):
 n=int(input())
 chef=0
 ans=[]
 for i in range(0,n):
  l=list(map(int,input().split()))
  c=l[0]
  if c%2==0:
   for i in range(1,len(l)//2+1):
    chef=chef+l[i]
   continue;
  for i in range(1,len(l)//2):
   chef=chef+l[i]
  ans.append(l[len(l)//2])
 ans.sort(reverse=True)
 for i in range(len(ans)):
  if i%2==0:
   chef=chef+ans[i]
 print(chef)
   
  

for i in range(int(input())):
 n=int(input())
 chef=0
 ans=[]
 for i in range(0,n):
  l=list(map(int,input().split()))
  c=l[0]
  if c%2==0:
   for i in range(1,len(l)//2+1):
    chef=chef+l[i]
   continue;
  for i in range(1,(len(l)//2)+1):
   chef=chef+l[i]
  ans.append(l[len(l)//2])
 print(chef)
   
  

for i in range(int(input())):
 n=int(input())
 chef=0
 ans=[]
 for i in range(0,n):
  l=list(map(int,input().split()))
  c=l[0]
  if c%2==0:
   for i in range(1,len(l)//2+1):
    chef=chef+l[i]
   continue;
  for i in range(1,len(l)//2):
   chef=chef+l[i]
  ans.append(l[len(l)//2])
 ans.sort(reverse=True)
 for i in range(len(ans)):
  if i%2==0:
   chef=chef+ans[i]
 print(chef)
   
  

# cook your dish here

T = int(input())

for _ in range(T):
 n = int(input())
 s = 0
 temp = []
 for _ in range(n):
  l = list(map(int, input().split()))
  if l[0]%2 == 0:
   s += sum(l[1:len(l)//2+1])
  else:
   s += sum(l[1:len(l)//2])
   temp.append(l[len(l)//2])
 temp.sort(reverse= True)
 s += sum(temp[::2])
  
 print(s)

t=int(input())
for i in range(t):
 n=int(input())
 chef=0
 t=[]
 for i in range(n):
  l=list(map(int,input().split()))
  c=l[0]
  if c%2==0:
   for i in range(1,len(l)//2+1):
    chef=chef+l[i]
   continue;
  for i in range(1,len(l)//2):
   chef=chef+l[i]
  t.append(l[len(l)//2])
 t.sort(reverse=True)
 for i in range(len(t)):
  if i%2==0:
   chef=chef+t[i]
 print(chef)
   

# cook your dish here
import math
for _ in range(int(input())):
 n=int(input())
 x=0
 mi=[]
 for i in range(n):
  l=list(map(int,input().split()))
  c=l[0]
  l1=l[1:]
  d=math.floor(c/2)
  s=sum(l1[:d])
  x=x+s
  if c%2!=0:
   mi.append(l1[d])
 mi.sort(reverse=True)
 for i in range(len(mi)):
  if(i+1)%2!=0:
   x=x+mi[i]
  
 print(x)

# cook your dish here
import math
for _ in range(int(input())):
 n=int(input())
 x=0
 for i in range(n):
  l=list(map(int,input().split()))
  c=l[0]
  l1=l[1:]
  d=math.ceil(c/2)
  s=sum(l1[:d])
  x=x+s
  l1=[]
  c=0
  d=0
 print(x)

# cook your dish here
import math
for _ in range(int(input())):
 n=int(input())
 x=0
 for i in range(n):
  l=list(map(int,input().split()))
  c=l[0]
  l1=l[1:]
  d=math.ceil(c/2)
  for i in range(d):
   x=x+l1[i]
  l1=[]
  c=0
  d=0
 print(x)

# cook your dish here
import math
for _ in range(int(input())):
 n=int(input())
 x=0
 for i in range(n):
  l=list(map(int,input().split()))
  c=l[0]
  l1=l[1:]
  s=len(l1)
  d=math.ceil(s/2)
  for i in range(d):
   x=x+l1[i]
 print(x)

# cook your dish here
import math
for _ in range(int(input())):
 n=int(input())
 x=0
 for i in range(n):
  l=list(map(int,input().split()))
  c=l[0]
  l=l[1:]
  s=len(l)
  d=math.ceil(s/2)
  for i in range(d):
   x=x+l[i]
 print(x)

for _ in range(int(input())):
 chef,f = 0,[]
 for j in range(int(input())):
  k=list(map(int,input().split()))
  for r in range(1,k[0]//2 +1 ):chef+=k[r]
  if(k[0]%2==1):f.append(k[k[0]//2 +1])
 f.sort(reverse=True)
 for p in range(0,len(f),2):chef+=f[p]
 print(chef)
# cook your dish here
test=int(input())
dic={}
for _ in range(test):
 dic={}
 n=int(input())
 for i in range(n):
  a=input().split()
  l=[int(a[i]) for i in range(1,int(a[0])+1)]
  dic[i]=l
 c=0
 lg=[]
 for i in dic:
  c+=len(dic[i])
 chef={i:dic[i][0] for i in range(n)}
 ram={i:dic[i][len(dic[i])-1] for i in range(n)}
 #chef={k: v for k, v in sorted(chef.items(), key=lambda item: item[1])}
 #ram={k: v for k, v in sorted(ram.items(), key=lambda item: item[1])}
 i=0
 ans=0
 while(i<c):
  if(i%2==0):
   var=min(chef,key=chef.get)
   #print(chef[var])
   ans+=chef[var]
   if(len(dic[var])==1):
    chef.pop(var)
    ram.pop(var)
    dic[var].pop(0)
   else:
    dic[var].pop(0)
    chef[var]=dic[var][0]
   #print("dic=",dic)
  else:
   var=min(ram,key=ram.get)
   lg=len(dic[var])-1
   if(lg==0):
    chef.pop(var)
    ram.pop(var)
   else:
    dic[var].pop(lg)
    ram[var]=dic[var][lg-1]
   #print("dic=",dic)

  i+=1
 print(ans)
 
  
  

for _ in range(int(input())):
 chef=0
 f=[]
 for j in range(int(input())):
  k=list(map(int,input().split()))
  for r in range(1,k[0]//2 +1 ):
    chef+=k[r]
  if(k[0]%2==1):
   f.append(k[k[0]//2 +1])
 f.sort(reverse=True)
 for p in range(0,len(f),2):
  chef+=f[p]
 print(chef)
   

for _ in range(int(input())):
 chef=0
 for j in range(int(input())):
  k=list(map(int,input().split()))
  if(k[0]%2==0):
   for r in range(1,k[0]//2 +1 ):
    chef+=k[r]
  else:
   for r in range(1,k[0]//2 +2 ):
    chef+=k[r]
 print(chef)
   

for _ in range(int(input())):
 n=int(input())
 l,l1=[],[]
 l3=[]
 s=0
 for i in range(n):
  l2=list(map(int,input().split()))
  n=l2[0]
  l2=l2[1:]
  if n%2==0:
   s+=sum(l2[:len(l2)//2])
  else:
   l.append(sum(l2[:len(l2)//2]))
   l.append(sum(l2[:len(l2)//2+1]))
   l1.append(l)
   l=[]
   l3.append(l1[len(l1)-1][1]-l1[len(l1)-1][0])
 l4=l3.copy()
 for i in range(len(l3)):
 
  if i<len(l3)//2:
   t=l3.index(min(l3))
   s+=min(l1[t])
   l3[t]=99999
  else:
   t=l4.index(max(l4))
   s+=max(l1[t])
   l4[t]=0
 print(s)
  
 
 

for _ in range(int(input())):
 n=int(input())
 l,l1=[],[]
 l3=[]
 s=0
 for i in range(n):
  l2=list(map(int,input().split()))
  n=l2[0]
  l2=l2[1:]
  if n%2==0:
   s+=sum(l2[:len(l2)//2])
  else:
   l.append(sum(l2[:len(l2)//2]))
   l.append(sum(l2[:len(l2)//2+1]))
   l1.append(l)
   l=[]
   l3.append(l1[len(l1)-1][1]-l1[len(l1)-1][0])
 for i in range(len(l3)):
  if i<len(l3)//2:
   t=l3.index(min(l3))
   s+=min(l1[t])
   l1.remove(l1[t])
  else:
   t=l3.index(min(l3))
   s+=max(l1[t])
   l1.remove(l1[t])
 print(s)
  
 
 

for _ in range(int(input())):
 n=int(input())
 l,l1=[],[]
 l3=[]
 s=0
 for i in range(n):
  l2=list(map(int,input().split()))
  n=l2[0]
  l2=l2[1:]
  if n%2==0:
   s+=sum(l2[:len(l2)//2])
  else:
   l.append(sum(l2[:len(l2)//2]))
   l.append(sum(l2[:len(l2)//2+1]))
   l1.append(l)
   l=[]
   l3.append(l1[len(l1)-1][1]-l1[len(l1)-1][0])
 for i in range(len(l3)):
  if i<len(l3)//2:
   t=l3.index(min(l3))
   s+=min(l1[t])
   l1.remove(l1[t])
  else:
   t=l3.index(min(l3))
   s+=max(l1[t])
   l1.remove(l1[t])
 print(s)
  
 
 

for _ in range(int(input())):
 s=0
 l1=[]
 for i in range(int(input())):
  l=list(map(int,input().split()))
  n=l[0]
  l=l[1:]
  if n%2==0:
   s+=sum(l[:n//2])
  else:
   l1.append(l)
 l2,l3=[],[]
 for i in range(len(l1)):
  l2.append(sum(l1[i][:len(l1[i])//2]))
  l2.append(sum(l1[i][:len(l1[i])//2+1]))
  l3.append(l2)
  l2=[]
 l3.sort()
 for i in range(len(l3)):
  if i<len(l3)//2:
   s+=min(l3[i])
  else:
   s+=max(l3[i])
 print(s)

for _ in range (int(input())):
 n1=int(input())
 sum1=0
 for i in range(n1):
  li=[int(i) for i in input().split()]
  li.pop(0)
  if len(li)%2==0:
   n=(len(li))//2
  else:
   n=(len(li)//2)+1
  sum1+=sum(li[:n])
 print(sum1)
 
  

# cook your dish here
for _ in range(int(input())):
 n=int(input()) 
 chef,f=0,[]
 for i in range(n):
  l=list(map(int,input().split()))
  b=l[0]
  if b&1:
   f.append(l[(b//2)+1]) 
  for i in range(1,(b//2)+1):
   chef+=l[i]
 print(chef+sum((sorted(f))[::2]))
# cook your dish here
for _ in range(int(input())):
 n=int(input()) 
 chef,f=0,[]
 for i in range(n):
  l=list(map(int,input().split()))
  b=l[0]
  if b&1:
   f.append(l[(b//2)+1]) 
  for i in range(1,(b//2)+1):
   chef+=l[i]
 print(chef+sum((sorted(f,reverse=True))[::2]))
# cook your dish here
for _ in range(int(input())):
 n=int(input()) 
 chef,f=0,[]
 for i in range(n):
  l=list(map(int,input().split()))
  b=l[0]
  if b&1:
   f.append(l[(b//2)+1]) 
  for i in range(1,(b//2)+1):
   chef+=l[i]
 print(chef+sum(sorted(f)[::2]))
