def series_slices(digits, n):
    if n > len(digits):
        raise ValueError
    else:
        return [[int(digit) for digit in digits[i:i+n]] for i in range(0, len(digits)-n+1)]
def series_slices(s, n):
    if n > len(s):
        raise
    
    arr = [int(d) for d in s]
    return [ arr[i:i+n] for i in range(len(s)-n +1) ]
def series_slices(digits, n):
    l = len(digits)
    if n > l:
        raise Exception
    else:
        d = list(map(int, digits))
        
        res = []
        for i in range(0, l-n+1):
            res.append(d[i:i+n])
        return res
def series_slices(d, n):
    # Good Luck!
    if(n>len(d)): return error
    x=[]
    i=0
    while(i<=len(d)-n):
        x.append([int(i) for i in d[i:i+n]])
        i+=1
    return x
    
    

def series_slices(digits, n):
    if n > len(digits):
        0 / 0
    return [list(map(int, digits[x:n+x])) for x in range((len(digits) - n) + 1)]
def series_slices(digits, n):
    digits = [int(i) for i in digits]
    if n > len(digits):
         raise Error('Your n is bigger than the lenght of digits')        
    else:
        return [list(digits[i:n+i]) for i in range(len(digits)) if len(digits[i:n+i]) == n]
def series_slices(digits, n):
    assert n <= len(digits)
    return [list(map(int, digits[i: i+n])) for i in range(len(digits)-n+1)]
def series_slices(digits, n):
    if n > len(digits):
        raise ValueError('n cannot be greater than number of digits')
    
    else:
        res = [digits[i:i+n] for i in range(len(digits) - n + 1)]
        for i in range(len(res)):
            res[i] = [int(e) for e in res[i]]

        return res
def series_slices(digits, n):
    if n > len(digits):
        raise error
    else:
        x = [int(y) for y in digits]
        return [x[i:i+n] for i in range(0,len(digits)-n+1)]
def series_slices(digits, n):
  return [list(map(int, digits[i:i+n])) for i in range(len(digits)-n+1)] if n <= len(digits) else int("")
