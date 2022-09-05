try:
    for _ in range(int(input())):
        s=input()
        t="abc"
        if len(s)<=2:
            print(s)
            continue
        if len(s)==3 and s==t:
            print('')
            continue
        i=0
        while i<len(s)-3+1 and len(s)>=3:
            if s[i:i+3]==t:
                s=s[:i]+s[i+3:]
                i=0
            else:
                i+=1
        print(s)
            
except:
    pass
