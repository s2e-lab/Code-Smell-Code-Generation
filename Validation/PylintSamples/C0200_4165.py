def uni_total(string):
    return sum(map(ord, string))
def uni_total(s):
    return sum(ord(c) for c in s)

def uni_total(string):
    return sum(ord(c) for c in string)
def uni_total(string):
    return sum(ord(x) for x in string)
def uni_total(string):
    return sum([ord(i) for i in string])

def uni_total(string):
    return sum(ord(ch) for ch in string)
def uni_total(string):
    acc = 0
    for i in string:
        acc += ord(i)
    return acc
def uni_total(strng):
    return sum(ord(a) for a in strng)

def uni_total(string):
    return sum(ord(i) for i in string)
def uni_total(string: str) -> int:
    """ Get the total of all the unicode characters as an int. """
    return sum([ord(_) if _ else 0 for _ in "|".join(string).split("|")])
def uni_total(string):
    return sum(ord(q) for q in string)
def uni_total(string):
    total = 0
    for item in string:
        total= total + ord(item)
    return total
def uni_total(string):
    c=0
    for x in string:
        c=c+ord(x)
    return c
def uni_total(string):
    s = 0
    for c in string:
        s += ord(c)
    return s


def uni_total(string):
    return sum(list(map(ord, string)))
from functools import reduce
def uni_total(string):
    return reduce(lambda x, y: x + ord(y), string, 0)
def uni_total(string):
    total = 0
    for i in string:
        total += ord(i)
    return total
uni_total=lambda s:sum(ord(i)for i in s)
def uni_total(string):
    return sum(ord(character) for character in string)
def uni_total(string):
    x=0
    for character in string:
        x+=ord(character)
    return x
uni_total = lambda string: sum(ord(c) for c in string)

def uni_total(string):
    total = []
    for char in string:
        total.append(ord(char))
    return sum(total)
        
    
'''You'll be given a string, and have to return the total of all the unicode 
characters as an int. Should be able to handle any characters sent at it.

examples:

uniTotal("a") == 97 uniTotal("aaa") == 291'''
def uni_total(string):
    return sum(list(ord(i) for i in string))
def uni_total(string):
    array = []
    copy = []
    array = list(string)   
    for num in array:
        copy.append(ord(num))
    total = sum(copy)
    return total
def uni_total(string):
    
    if type(string)!= str:
        return 0
    else:
        liste= [(ord(i)) for i in string]
        return sum(liste)

def uni_total(string):
    ttl = 0
    for i in string:
        ttl += ord(i)
    return ttl

def uni_total(s):
    return sum(ord(l) for l in s)
def uni_total(string):
    x = map(ord, string)
    return sum(x)
def uni_total(string):
    sum = 0
    for i in string:    
        i = ord(i)
        sum = sum + i
    return sum
def uni_total(string: str) -> int:
    """ Get the total of all the unicode characters as an int. """
    return sum(bytearray(string, "utf"))
def uni_total(string: str) -> int:
    """ Get the total of all the unicode characters as an int. """
    return sum(map(ord, string))
def uni_total(s):
    #your code ere
    st=0
    for c in s:
        st+=ord(c)
    return st
def uni_total(string):
    ans = 0
    if string:    
        for _ in string:
            ans += ord(_)
    return ans
def uni_total(string):
    liReturn = 0
    
    for i in string:
        liReturn += ord(i)
        
    return liReturn
def uni_total(string):
    total = 0
    if string == "":
        return 0
    else:
        for i in range(len(string)):
             total = ord(string[i]) + total
    return total
def uni_total(string):
    erg = 0
    for c in string:
        erg += ord(c)
    return erg
def uni_total(string):
    accumulator = 0
    for eachchar in string:
        accumulator = accumulator + ord(eachchar)
    return accumulator
def uni_total(s):
    t=0
    for i in s:
        t += ord(i)
    return t

def uni_total(string):
    #your code here
    if string=='':
        return 0
    else:
        ans=0
        for i in string:
            ans=ans+ord(i)
        return ans
def uni_total(string):
    rez = 0
    for i in string:
        rez += ord(i)
    return rez
def uni_total(string):
    l = list(string)
    m = sum([ord(i) for i in l])
    return m
def uni_total(string):
    account = 0
    for letters in string:
        account += ord(letters)
    return account
def uni_total(string):
    account = 0
    for letters in string:
        if 96 < ord(letters) < 123: 
            account += ord(letters)
        else:
            64 < ord(letters) < 91
            account += ord(letters)
    return account 

def uni_total(string):
    salida = 0
    for letr in string:
        salida += ord(letr)

    return salida
def uni_total(string):
    result = 0
    for s in string:
        result += ord(s)
    return result

def uni_total(string):
    unicodes = []
    for i in string:
        unicodes.append(ord(i))
    return sum(unicodes)
def uni_total(string):
    if string:
        s  = sum([ord(i) for i in string])       
    else:
        s = 0
    return s
def uni_total(string):
    summ = 0
    for i in string:
        summ += ord(i)
    return summ if string else 0
def uni_total(string):
    return sum(ord(i) for i in string) if string else 0
def uni_total(string):
    
    output = 0
    
    for letter in string:
        output += ord(letter)
    
    return output
def uni_total(string):
    return sum([ord(elem) for elem in string])
def uni_total(string):
    toplam = 0
    for i in string:
        toplam += int(str(ord(i)))
    return toplam
def uni_total(string):
    int = [ord(x) for x in string]
    sum = 0
    for i in int:
        sum += i
    return sum
def uni_total(string):
    return sum(ord(each) for each in string)
def uni_total(string):
    b = []
    for i in range(len(string)):
        b.append(ord(string[i]))
    return sum(b)
def uni_total(string):
    if not string:
        return 0
    if len(string) == 1:
        return ord(string)
    return sum(list(map(ord, string)))
def uni_total(string):
    char_uni = {" ":32, "0":48, "1":49, "2":50, "3":51, "4":52, "5":53, "6":54, "7":55, "8":56, "9":57, "A":65, "B":66, "C":67, "D":68, "E":69, "F":70, "G":71, "H":72, "I":73, "J":74, "K":75, "L":76, "M":77, "N":78, "O":79, "P":80, "Q":81, "R":82, "S":83, "T":84, "U":85, "V":86, "W":87, "X":88, "Y":89, "Z":90, "a":97, "b":98, "c":99, "d":100, "e":101, "f":102, "g":103, "h":104, "i":105, "j":106, "k":107, "l":108, "m":109, "n":110, "o":111, "p":112, "q":113, "r":114, "s":115, "t":116, "u":117, "v":118, "w":119, "x":120, "y":121, "z":122}
    string_lst  = list(string)
    uni_lst = []
    
    for i in string_lst:
        for char, uni in char_uni.items():
            if char == i:
                uni_lst += [uni]
                continue
        if i == " ":
            continue
        
    return sum(uni_lst)

print(uni_total("no chars should return zero"))
def uni_total(string):
    return sum(ord(i) for i in string) if string != '' else 0
uni_total=lambda s: sum([ord(e) for e in s]) if len(s)>0 else 0
def uni_total(s):
    s=list(s)
    s=[ord(i) for i in s]
    return sum(s)
def uni_total(s):
  return sum(ord(char) for char in s)

def uni_total(string):
    #your code here
    try: 
        return sum([ord(i) for i in string])
    except:
        return 0
def uni_total(string):
    sco = 0
    for let in string:
        sco += ord(let)
    return sco
def uni_total(string):
    return sum([int(ord(s)) for s in string])
def uni_total(string):
    return sum([ord(x) for x in string])
    # Flez

def uni_total(string):
    ret = 0
    for c in string:
        ret += ord(c)
    return ret
def uni_total(string):
  finalist = list()
  mylist = list(string)
  for x in mylist:
    finalist.append(ord(x))
  return sum(finalist)

def uni_total(string):
    return sum(ord(s) for s in string) if string else 0
def uni_total(string):
    res = 0
    
    for let in string:
        res += ord(let)
    
    return res

def uni_total(string):
    letters = list(string)
    total = 0
    for letter in letters:
        total = total + ord(letter)
    return total
def uni_total(string):
    # sum all caractere in string 
    return sum(ord(s) for s in string)
def uni_total(sz):
    return sum(ord(c) for c in sz)
def uni_total(string):
    #your code here
    rez = 0
    for c in string:
        rez += ord(c)
    return rez     
    

def uni_total(string):
    res = 0
    for item in string:
        res += ord(item)
    return res
from functools import reduce

uni_total=lambda s: reduce(lambda a,b: a+ord(b),s,0)
def uni_total(string):
    if string == '':
        return 0
  
    return sum(ord(i) for i in string)
def uni_total(string):
    cnt = 0
    for i in string:
        cnt += ord(i)
    return cnt
def uni_total(string):
    #your code here
    ans = 0
    for i in range(len(string)):
        ans += ord(string[i])
    return ans
def uni_total(string):
    return sum(ord(num) for num in string)
def uni_total(string):  
    return 0 if len(string) == 0 else sum([int(ord(ch)) for ch in string])
def uni_total(string):
    uni_total = 0
    for i in string:
        uni_total += ord(i)
    return uni_total
def uni_total(string):
    tot = 0
    for x in list(string):
        tot += ord(x)
    return tot
def uni_total(string):
    #your code here
    sum=0
    for e in string:
        sum=sum+ord(e)
    return sum
def uni_total(str):
    x = 0
    for l in str:
        x += ord(l)
    return x
def uni_total(string):
    if not string:
        return 0
    return sum([ord(s) for s in string])
def uni_total(string):
    result=0
    print(string)
    for i in string:
        print(ord(i))
        result=result+ord(i)
    return result
def uni_total(string):
  a=list(string)
  tot=0
  for i in a:
    tot=tot+ord(i)
  return(tot)

def uni_total(string):
    cnt = 0
    for e in string:
        cnt += ord(e)
        
    return cnt
    #your code here

def uni_total(s):
    sum = 0
    for c in s:
        sum += ord(c)
    return sum
def uni_total(string):
    count = 0 
    for x in string:
        count += ord(x)
    return count

def uni_total(string):
  return sum([ord(str) for str in string])
def uni_total(string):
    s = string
    count = 0
    if s == "":
        return 0
    for x in range(len(s)):
        count = count + ord(s[x])
    return count
uni_total = lambda string: sum([ord(x) for x in list(string)])
def uni_total(string):
    lst = [ord(x) for x in list(string)]
    return sum(lst)
def uni_total(string):
    a = 0
    for s in string:
        a = a + ord(s)
    return a
def uni_total(string):
    x = 0
    if string == '':
        return 0
    for i in range(len(string)):
        x += int(ord(string[i]))
    return x
def uni_total(string):
    #your code here
    s=0
    for x in string:
        s+=ord(x)
    return s
def uni_total(string):
    out=0
    if not string:return 0
    for items in string:
        out+=ord(items)
    return out
def uni_total(string):
    if string == "":
      return 0
    
    count = 0
    for i in string:
        count = count + ord(i)
    return count
