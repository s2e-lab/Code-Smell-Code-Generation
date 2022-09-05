exec("a,b,c,d,e,f=map(int,input().split());g=a+c+e-max([a,c,e]);h=b+d+f-max([b,d,f]);print(max([abs(g),abs(h)])+(g*(g-1)and g==h));"*int(input()))
