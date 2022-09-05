from collections import Counter


def get_strings(city):
    return ",".join(f"{char}:{'*'*count}" for char, count in Counter(city.replace(" ", "").lower()).items())
from collections import Counter


def get_strings(city):
    return ",".join(
        f"{char}:{'*'*count}"
        for char, count in list(Counter(city.replace(" ", "").lower()).items())
        if char.isalpha()
    )

def get_strings(city):
    city = city.lower()
    result = ""
    for i in city:
        if i in result:
            pass
        elif i == " ":
            pass
        else:
            result += i + ":" + ("*" * city.count(i)) + ","

    return result[:-1]
def get_strings(city):
    city = city.lower().replace(" ", "")
    return ",".join(sorted([f"{i}:{city.count(i)*'*'}" for i in set(city)], key=lambda x:city.index(x[0])))
from collections import Counter

def get_strings(city):
    c = Counter(filter(str.isalpha, city.lower()))
    return ','.join(f'{ k }:{ "*"*v }' for k,v in c.items())
def get_strings(s):
    return ','.join(f"{i}:{'*'*s.lower().count(i)}" for i in dict.fromkeys(s.replace(' ','').lower()))
def get_strings(city):
    city=city.lower().replace(' ','')
    return ','.join(f'{c}:'+'*'*(city.count(c)) for c in dict.fromkeys(city))
from collections import Counter
def get_strings(city):
    lst = []
    for key, value in dict(Counter(city.replace(' ', '').lower())).items():
        lst.append(key+':'+'*'*value)
    return ','.join(lst)
import re

def get_strings(city):
    cache = {}
    r_string = ''
    counter = 0
    city = re.sub(' ','', city)
    
    for letter in city.lower():
        if letter not in cache:
            cache[letter] = 0 
            cache[letter] +=1
        else:
            cache[letter] +=1
    
    for k,v in cache.items():
        if counter < len(cache)-1:
            r_string += k + ':' + ('*' * v)+','
            counter += 1
        elif counter < len(cache):
            r_string += k + ':' + ('*' * v)
    return r_string
from collections import Counter


def get_strings(city):
    return ",".join(
        f"{char}:{'*'*count}"
        for char, count in Counter(city.replace(" ", "").lower()).items()
        if char.isalpha()
    )
def get_strings(city):
    city = list(city.lower())
    con = {}
    for char in city:
        if char == " ":
            pass
        else:
            con[char] = city.count(char) * "*"
    first = True
    result = ""
    for item in con.keys():
        if first == True:
            first = False
            result+= item + ":" + con.get(item)
        else:
            result+= "," + item + ":" + con.get(item)
    return result
import json
def get_strings(city):
    city = city.lower()
    city = city.replace(" ","")
    c = {}
    for i in city:
        if i in c:
            c[i] += "*"
        else:
            c[i] = "*"
    
    final = json.dumps(c)
    final = (final.replace("{","").replace("}","").replace(" ","").replace('"',""))
            
    return final
def get_strings(city):
    index = 0
    string = ''
    city = city.lower()
    city = city.replace(' ', '')
    for i in city:
        asterisks = city.count(i)
        if i in string:
            index = index + 1
        else:
            if index == (len(city) - 1) and i not in string:
                string += i.lower() + ':' + ('*' * asterisks)
            if index != (len(city) - 1):
                string += i.lower() + ':' + ('*' * asterisks) +','
                index = index + 1
    if string[-1] == ',':          
        lst = list(string)
        lst[-1] = ''
        new = ''.join(lst)
        return new
    return string
        

from collections import Counter
def get_strings(city):
    return ','.join(f'{e}:{"*"*c}' for e,c in Counter(city.lower()).items() if e.isalpha())
from collections import Counter

def get_strings(city):
    return ','.join(f"{k}:{'*'*v}" for k,v in  Counter(filter(str.isalpha, city.lower())).items())
def get_strings(city):
    s, city = str(), ''.join(city.lower().split())
    for i in city:
        if i not in s:
            s+= i+':'+'*'*city.count(i)+','
    return s[:-1]
from collections import Counter

def get_strings(word):
    return ','.join(
        f'{letter}:{"*" * count}'
        for letter, count in Counter(word.lower()).items()
        if letter.isalpha())
from collections import Counter
def get_strings(city):
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    city_str = ''
    city_dict = Counter(city.lower())
    for key in city_dict:
        if key in alphabet:
            city_str += key + ':' + '*' * city_dict[key] + ','
    return city_str[:-1]
def get_strings(city):
    
    ciu = {}
    
    for letra in city.lower().replace(" ",""):
        if letra not in ciu:
            ciu[letra] = "*"
        else: 
            ciu[letra] += "*"
    
    
    array = []
    for clave in ciu:
        
        array.append("{}:{}".format(clave, ciu[clave]))
        
    return ",".join(array).replace(" ","")


from collections import Counter
def get_strings(city):
    return ','.join([k.lower() + ':' + '*'*v for k, v in Counter(city.lower()).items() if k.isalpha()])
from collections import Counter

def get_strings(city):
    return ','.join(
        f'{letter}:{"*" * count}' 
        for letter, count in list(Counter(city.lower()).items())
        if letter != ' '
    )

def get_strings(city):
    counts = {}
    
    for c in city.lower():
        if c.isalpha():
            counts[c] = counts[c] + "*" if c in counts else "*"
    
    return ",".join([f"{c}:{a}" for (c, a) in counts.items()])
get_strings=lambda s:','.join(k+':'+s.lower().count(k)*'*'for k in dict.fromkeys(s.lower().replace(' ','')))
def get_strings(city):
    city = city.lower()
    city = list(city)
    dicti = {}
    strres = ""

    for i in city:
        if ord(i) > 96 and ord(i) < 123:
            if i in dicti:
                dicti[i] += 1
            else:
                dicti[i] = 1
    
    for i in dicti:
        strres = strres + i + ":" + ("*"*dicti[i]) + ","
        
    strres = strres[:-1]

    return strres

def get_strings(city):
    return ",".join([i + ":" + '*' * city.lower().count(i) for i in dict.fromkeys(city.lower().replace(' ', ''))])
def get_strings(stra):
    string = stra.lower()
    dc = {}
    for i in string:
        if i == ' ':
            continue
        dc[i] = '*' * string.count(i)
    
    return ','.join([k + ':' + v for k, v in list(dc.items())])

def get_strings(city):
    res = {}
    for i in city.lower().replace(' ',''):
        res[i] = res.get(i, '') + '*'

    return ','.join([f'{k}:{v}' for k,v in res.items()])
from collections import OrderedDict
def get_strings(city):
    values = [i for i in city.replace(' ','').lower()]
    v2 = [j +':'+'*'*values.count(j) for j in values]
    return ','.join(list(OrderedDict.fromkeys(v2)))

def get_strings(city):
    city = city.lower()
    city = city.replace(" ","")
    cit_str = ""
    usedletters=[]
    for letter in city:
        if letter in usedletters:
            pass
        else:
            cit_str += letter + ":"       
            l_num = city.count(letter)
            for num in range(0,l_num):
                cit_str += "*"
            cit_str += ","
            usedletters.append(letter)
    
    return cit_str[:-1]
get_strings = lambda city: "".join(list(dict.fromkeys(["{}:{},".format(i, "*" * city.lower().count(i)) for i in city.lower() if i != " "]))).rstrip(",")
from collections import Counter
def get_strings(city):
    return ",".join([f'{k}:{v*"*"}' for k,v in Counter(city.lower()).items() if k.isalpha()])
from collections import Counter
def get_strings(city):
    d = dict(Counter(city.lower()))
    ret = [f'{k}:{v*"*"}' for k,v in d.items() if k.isalpha()]
    return ",".join(ret)
def get_strings(city):
    city = city.lower().replace(' ', '')
    dict = {}
    for l in city:
        if l in dict:
            dict[l] += '*'
        else:
            dict[l] = '*'
    result = ''
    for i in dict:
        result += i + ':' + dict[i] + ','
    return result[:-1]
def get_strings(city):
    memory = {}

    for char in city:
        if char.isspace():
            continue
        char = char.lower()
        if char not in memory.keys():
            memory[char] = "*"
        else:    
            memory[char] +=  "*"

    return_str = ""
    for k,v in memory.items():
        return_str += k + ":" + v + ","
       
    return return_str[:-1]
def get_strings(city):
    city=city.lower()
    storage=[]
    hold=""
    for i in (city):
        if i.isalpha()==True and str(i) not in hold:
            storage.append(str(i)+':'+ (city.count(i))*'*')
            hold+=str(i)
    return ",".join(storage)
from collections import OrderedDict

def get_strings(city):
    return ','.join(f'{letter}:{"*"*city.lower().count(letter)}' for letter in {letter:None for letter in city.lower() if letter.isalpha()})
def get_strings(inputString):
    checkDict={
            "letter":[],
            "count":[]
        }
    charArray = list(inputString.lower())
    for i in charArray:
        if i in checkDict['letter']:
            k=checkDict['letter'].index(i)
            checkDict['count'][k]= checkDict['count'][k]+'*'
        else:
            if (i.isalpha()):
                checkDict['letter'].append(i)
                checkDict['count'].append('*')
    #print(checkDict) om de dictionary te bekijken
    outputString = ''
    for i in checkDict['letter']:
        indexLetter = checkDict['letter'].index(i)
        outputString = outputString + i + ':' + checkDict['count'][indexLetter] + ','
    return outputString[:-1]
def get_strings(city):
    myMap = {}
    for i in range(len(city)):
        letter = city[i].lower()
        if letter not in myMap:
            myMap[letter] = 1
        else:
            myMap[letter] = myMap.get(letter)+1

    myMap.pop(' ', None)


    result = ''
    for item in myMap:
        result = result + item + ':'
        for i in range(myMap.get(item)):
            result += '*'
        result += ','


    return result[:-1]
def get_strings(city):
    letters = {}
    for letter in city.lower():
        if letter in letters:
            value = letters[letter]
            value += 1
            letters.update({letter : value})
        else:
            letters.update({letter : 1})
    result = ""
    for letter in city.lower().replace(" ", ""):
        if letter not in result:
            result += letter + ':' + '*' * letters[letter] + ',' 
    return result[:-1]
from collections import Counter


def get_strings(city):
    counting = Counter(city.lower()) # makes sure everything is lowercase
    dict(counting)
    output =''
    #print(counting.items())
    for key, value in counting.items():
        
        if len(key.strip()) > 0: #checks for whitespace, (0 is returned if there is whitespace)
            output += key + ':' + ('*'*value)+',' #formats the output according to the task description
  
    output = output[:-1] #removes the last ','
    print(output) 
    return output
def get_strings(city):
    result = ''
    d = {}
    for letter in city.lower():
        d[letter] = d.get(letter, 0) + 1
    for letter in city.lower().replace(' ',''): 
        if not letter in result: 
            result += letter+':'+'*'*d[letter]+','
    return result[:-1]

def get_strings(city):
    new = city.lower().replace(' ','')
  
    list = []
    for x in new:
        c = new.count(x)
        y = x+":"+"*"*c
        if y not in list:
            list.append(y)
            
        s = ','.join(list)
        
    return s

def get_strings(city):
    
    # convert to lower case alphabet
    city_alpha = "".join(x.lower() for x in city if x.isalpha())
    
    # order of appearance and count
    seen_cnt, order = {}, ""
    for x in city_alpha :
        if x not in seen_cnt :
            seen_cnt[x]=1 
            order += x
        else :
            seen_cnt[x] += 1
    
    # generate output
    output = ""
    for x in order :
        output += ","+x+":"+"*"*seen_cnt[x]
        
    return output[1:]
def get_strings(city):
    res = {}
    for x in city.lower():
        if x.isalpha():
            res[x] = city.lower().count(x)
    return ",".join([ "%s:%s" % (x,"*"*res[x]) for x in res])
def get_strings(city):
    
    city = city.replace(' ','').lower()
    city_string = ''
    
    for i in city:
        if i in city_string:
            pass
        else:
            city_string += i +':'+ ('*'* city.count(i))+','
            
    result = city_string[:-1]

    return result  
from collections import Counter

def get_strings(city):
    return ','.join([f"{char}:{'*' * char_count}" for (char, char_count) 
        in list(dict(Counter(city.lower())).items()) if char != ' ' ])

def get_strings(city):
    city = city.lower().replace(" ", "")
    arr = []
    for el in city:
        if el not in arr:
            arr.append(el)
    return ",".join([f"{el}:{city.count(el) * '*'}" for el in arr])
def get_strings(city):
    city = city.lower().replace(" ", "")
    a = []
    for el in city:
        if el not in a:
            a.append(el)
    return ",".join([f'{el}:{"*"*city.count(el)}' for el in a])
def get_strings(city):
    result = ''
    city = city.lower().replace(' ', '')
    for i in city:
        if i not in result:
            result += f'{i}:{"*"*city.count(i)},'
    return result[:-1]
def get_strings(city):
    city=city.lower().strip()
    letters_count={}
    
    def star_print(int):
        star_string=""
        for i in range(int):
            star_string=star_string+"*"
        return star_string
    
    for letter in city:
        if letter.isalpha():  
            if letter in letters_count:
                letters_count[letter]=letters_count[letter]+1
            else:
                letters_count[letter]=1
                
    ans_string=""
    for items in letters_count:
        temp_string=str(items)+":"+star_print(letters_count[items])+","
        ans_string=ans_string+temp_string
    ans_string=ans_string[0:-1]
    return ans_string
def c(lista):
    x = lambda a:(a,lista.count(a))
    return x
def star(n):
    s = ''
    for i in range(n):
        s = s +'*'
    return s
        
def remover(lista):
    l = []
    for i in lista:
        if i in l : continue
        l.append(i)
    return l 
def get_strings(city):
    letters = remover(list((city.lower())))
    huss = list(map(c(city.lower()) , letters))
    
    s = ''
    for letter,co in huss:
        if letter == ' ': continue
        s = s + letter + ':'+star(co)+',' 
        s = s.strip()
    return s[:-1]
    

def get_strings(city):
    
    string = city.lower().replace(' ','')
    output = ""
    for char in string:
        if char not in output:
            print((string.count(char)))
            output += (f'{char}:{"*" * string.count(char)},')
            
    return output[:-1]




def get_strings(city):
    soln = []
    used = ''
    for letter in city.lower():
        if letter in used or letter == ' ':
            continue
        else:
            solnstring = ''
            solnstring += letter + ':'
            for x in range(city.lower().count(letter)):
                solnstring += '*'
            soln.append(solnstring)
            used += letter
    return ','.join(soln)
def get_strings(str):
    output = ""
    str = str.lower().replace(' ', '')
    for char in str:
        if char not in output:
            output += (f'{char}:{"*" * str.count(char)},');
    return output[:-1]
def get_strings(city):
    a=[]
    b=[]
    for i in city.lower(): 
        if i==" ": continue
        if i in a:
            continue
        a+=[i]
        b+=[str(i)+":"+"*"*city.lower().count(i)]
    return ",".join(b)
def get_strings(city):
    formatRes = ""
    string = ""
    newCity = city.lower().replace(' ', '')
    main = {}
    i = 0
    k = 0
    while i < len(newCity):
        if main.get(newCity[i]):
            bef = main[newCity[i]]
            main[newCity[i]] = bef + "*"
            i += 1
        else:
            main[newCity[i]] = "*"
            formatRes += newCity[i]
            i += 1
    while k < len(formatRes):
        string = string + formatRes[k]+":"+ main[formatRes[k]]+","
        k+=1
    return string[:-1]
def get_strings(city):
    result = {}
    for i in city.lower():
        if i in result and i != " ":
            result[i] += "*"
        elif i not in result and i != " ":
            result[i] = "*"
    test = []
    for k, v in list(result.items()):
        test.append((k+':'+v))
    return ",".join(test)
    

def get_strings(city):
    formatedS = city.replace(' ', '').lower()
    res = ""
    count = 1
    for x,c in enumerate(formatedS):
        for d in formatedS[x+1:]:
            if(d == c):
                count += 1
        if c not in res:
            res += c + ":" + "*" * count + ","
        count = 1
    return res[:-1]
def get_strings(city):
    l = []
    city = city.lower()
    city = city.replace(" ","")
    for char in city:
        ast = city.count(char) * "*"
        l.append("%s:%s," %(char,ast))        
    l = list(dict.fromkeys(l))
    city = "".join(l)
    return city[0:-1]

def get_strings(city):
    city = city.replace(" ", "")
    characters = list((city).lower())
    char_dict = {}
    final = ""
    for char in characters:
        if char in char_dict.keys():
            char_dict[char] = char_dict[char] + "*"
        else: 
            char_dict[char] = char + ":*"
    for key in char_dict.values():
        final = final + (key + ",")
    return final[:-1]
def get_strings(city):
    dic = {} 
    for chr in city.lower():
        if chr.isalpha():
            dic[chr]=dic.get(chr,'')+'*'
    return ','.join([i+':'+dic[i] for i in dic])
def get_strings(city):
    record = []
    text = ''.join(city.split()).lower()
    collect = {}
    counter = 0
    for i in range(len(text)):
        if text[i] not in collect:
            counter = text.count(text[i])
            collect[text[i]] = ':' + counter * '*'
        else:
            counter += 1

    for x,y in list(collect.items()):
        record.append(x+y)

    return ','.join(record)


def get_strings(city):
    city = city.replace(' ', '')
    city = city.lower()

    d = dict()
    for letter in city:
        if letter not in d:
            d[letter] = '*'
        else:
            d[letter] += '*'

    letter_list = list(map(lambda v: f'{v[0]}:{v[1]}', d.items()))
    return ','.join(letter_list)
def get_strings(city):
    print(city)
    city, m, mas, l = city.lower(), "", [], []
    for i in set(city.lower()):
        if city.count(i) > 1:
            mas.append(i)
            l.append(city.count(i))
            for j in range(city.count(i) - 1): city = city[:city.rindex(i)] + city[city.rindex(i) + 1:]
    for i in city:
        if i in mas: m += i + ":" + "*" * l[mas.index(i)] + ","
        elif i != " ": m += i + ":*" + ","
    return m[:len(m)-1]

def get_strings(city):
    city = city.lower()
    res = []
    for x in city:
        c = city.count(x)
        st = f"{x}:{'*'*c}"
        if x.isalpha() and not st in res:
            res.append(st)
    return ",".join(res)
def get_strings(city):
    city_list = list(city.lower())
    empty = []
    abet = [chr(i) for i in range(ord('a'), ord('z')+1)]
    res = ''
    for i in range(0, len(city_list)):
        c = 0
        letter = city_list[i]
        for j in range(0, len(city_list)):
            if letter in abet:
                if letter == city_list[j]:
                    c += 1
                    city_list[j] = ''
            else:
                continue
        if letter in abet:
            res += letter+':'+c*'*'+','               
        else: continue
    return res[0:len(res)-1]
from collections import Counter
def get_strings(city):
    counts = Counter(city.replace(' ', '').lower())
    return (','.join([f'{x}:{"*" * counts[x]}' for x in counts]))
from collections import OrderedDict
def get_strings(city):
    city = city.lower().replace(" ", "")
    city_short = "".join(OrderedDict.fromkeys(city))
    city_string = ""
    for char1, char2 in zip(city, city_short):
        value = city.count(char2)
        city_string += f"{char2}:{value * '*'},"
    city_string = city_string[:-1]
    return city_string
def get_strings(city):
    city = list(city.lower())
    if ' ' in city:
        for _ in range(city.count(' ')):
            city.remove(' ')
    r = ''
    while len(city):
        w = city[0]
        r += str(w)+":"+ '*'*int(city.count(w))+ ','
        for _ in range(city.count(w)):
            city.remove(w)
    return r[:-1]
def get_strings(city):
    city = city.lower()
    return ",".join(f"{letter}:{count*'*'}" for letter, count in {letter: city.count(letter) for letter in city if not letter.isspace()}.items())
def get_strings(city):
    city = city.lower()
    return ','.join(f"{c}:{'*' * city.count(c)}" for c in dict.fromkeys(city) if c.isalpha())
def get_strings(city):
    city = city.lower()
    return ','.join([f"{k}:{v}" for k,v in {c: '*'*city.count(c) for c in city if c != " "}.items()])
def get_strings(city):
    string = []
    for i in city.lower():
        n = city.lower().count(i)
        if i + ':' + '*' * n not in string and i.isalpha():
            string.append(i + ':' + '*' * n)
    return ",".join(string)


def get_strings(city):
    dic = dict()
    for c in city.lower():
        if c.isalpha and c != " ":
            dic[c] = dic.get(c, 0) + 1
    result = list()
    for key , value in dic.items():
        result.append('{}:{}'.format(key, value * '*'))
    out = ','.join(result)
    return out
# takes string of city name
# returns string with all chars in city counted, count represented as *
def get_strings(city):
    dic = {}
    city = city.lower()
    for i in range(len(city)):
        if city[i] != " ":
            dic[city[i]] = city.count(city[i])
    output: str = ','.join("{}:{}".format(key, val*"*") for (key, val) in list(dic.items()))
    return output



def get_strings(city):
    city = city.lower()
    emptystr = ''
    for letter in city.lower():
        num = city.count(letter)
        if letter not in emptystr:
            if letter == ' ':
                continue
            else:
                emptystr += letter + ':' + str(num * '*') + ','
        else:
            pass
    return emptystr[:-1]
def get_strings(city):
    alpha = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    city_1 = ""
    for i in city:
        if i.lower() not in city_1 and i.isalpha():
            city_1 += i.lower()
        else:
            pass
    return ",".join('{}:{}'.format(i, '*' * city.lower().count(i)) for i in city_1)
def get_strings(city):
    city = city.lower().replace(' ', '')
    lst = [f'{x}:{"*" * city.count(x)}' for x in sorted(set(city), key=lambda x: city.index(x))]
    return ','.join(lst)
def get_strings(city):
    city = city.lower()
    emptystr = ''
    for letter in city.lower():
        num = city.count(letter)
        if letter not in emptystr:
            if letter == ' ':
                continue
            else:
                if letter is not city[-1]:
                    emptystr += letter + ':' + str(num * '*') + ','
                else:
                    emptystr += letter + ':' + str(num * '*') + ','
        else:
            pass        
    if emptystr[-1] == ',':
        return emptystr[:-1]
    else:
        return emptystr
def get_strings(city):
    l = []
    [l.append(f"{x}:{city.lower().count(x)*'*'}") for x in city.lower() if f"{x}:{city.lower().count(x)*'*'}" not in l and x != " "]
    return ','.join(l)
def get_strings(city):
    dict_letters = {}
    
    for i in city.replace(' ', '').lower():
        if i in dict_letters:
            dict_letters[i] += '*'
        else:
            dict_letters[i] = '*'
    
    string_counts=  ''
    for k,v in dict_letters.items():
        string_counts += k + ':' + v + ','
    return string_counts[:-1]
def get_strings(city):
    lettercount =  []
    check = ''
    
    for i in city.lower():
        
        if i not in check and i.isalpha():
            check = check + i
            string = '{}:{}'.format(i, city.lower().count(i)*'*')
            lettercount.append(string)
            
    return ','.join(lettercount)
def get_strings(city):
    dict = {}
    for c in city.lower():
        if not c.isspace():
            if c in dict:
                dict[c] += 1
            else:
                dict[c] = 1
    out = ''
    count = 1
    for x in dict:
        out += x + ':' + dict[x]*'*'
        if count != len(dict):
            out += ','
        count += 1
    return(out)
def get_strings(city):
    city = city.lower()
    c = []
    b = []
    for ans in city:
        if ans not in b and ans!=" ":
            b += ans
            c += [ans+":"+city.count(ans)*"*"]
#     c += [ans+":"+city.count(ans)*"*" for ans in city if (ans not in c and ans!=" ")]
    return ",".join(c)

from collections import Counter

def get_strings(city):
    city = [elem for elem in city.lower() if elem.islower()]
    a = Counter(city)
    return ','.join(f"{elem}:{'*' * a[elem]}" for elem in list(a.keys()))

from collections import Counter
def get_strings(city):
    myList = []
    counter = Counter(city.lower())
    for letter in counter:
        if letter == " ":
            pass
        else:
            myList.append(letter + ":" + str("*"*city.lower().count(letter)))
    return ",".join(myList)
def get_strings(city):
    z=str()
    city=city.replace(" ","").lower()
    for i in city:
        cnt=int(city.count(i))
        j=1
        y=''
        if z.find(i) == -1 :
            y=i+':'
            while j<= cnt :
                y=y+'*'
                j=j+1
            y=y+','
        z=z+y

    return z[:-1]
from collections import Counter
def get_strings(city):
    l = []
    d = Counter(city.lower())
    for c,n in d.items():
        if c == " ":
            continue
        l.append(c+":"+"*"*n)
    return ",".join(l)
from collections import Counter
def get_strings(city):
    return ','.join(f'{k}:{"*"*v}' for k, v in Counter(city.lower().replace(' ','')).items())
from collections import Counter
def get_strings(city):
    return ','.join((a+f':{"*" * b}' for a,b in Counter(city.lower().replace(' ', '')).items()))
def get_strings(city):
    city = city.lower()
    alpha = 'abcdefghijklmnopqrstuvwxyz'
    city_dict = {}
    for char in city:
        if char not in alpha: continue
        if char in city_dict:
            city_dict[char] += '*'
            continue
        city_dict[char] = '*'
    output = ''
    for k in city_dict:
        output += f'{k}:{city_dict[k]},'
    return output[0:-1]

def get_strings(city):
    #create lower_case city only with letters
    new_city = ''.join(filter(str.isalpha, city.lower()))
    
    #list of unique letters
    char_seen=[]
    for char in new_city:
      if char not in char_seen:
        char_seen.append(char)
    
    #list of counts for unique letters
    count_char = []
    for char in char_seen:
      x =new_city.count(char)
      count_char.append(x)
    
    #create dictionary with two parallel list
    d = dict(zip(char_seen, count_char))

    total_str = ""
    for char, count in d.items():
      total_str += char + ":" + count*"*" + "," #using += to append instead

    
    return total_str[:-1]
def get_strings(city):
    new_city = ''.join(filter(str.isalpha, city.lower()))
    char_seen=[]
    for char in new_city:
      if char not in char_seen:
        char_seen.append(char)

    count_char = []
    for char in char_seen:
      x =new_city.count(char)
      count_char.append(x)

    d = dict(zip(char_seen, count_char))

    total_str = []
    for char, count in d.items():
      count_str = char + ":" + count*"*"
      total_str.append(count_str)
    
    return ','.join(total_str)
def get_strings(city):
    cityDict = {}
    for i in city.lower():
        if i.isalpha():
            if i not in cityDict:
                cityDict[i] = 1
            else:
                cityDict[i] += 1
    
    chars = []
    for i in cityDict.keys():
        chars.append("{}:{}".format(i, '*' * cityDict[i]))
    
    return ','.join(chars)
def get_strings(city):
    
    city = city.lower() 
    result = ''
    
    for i in city: 
        if not(i.isalpha()): 
            pass
        elif i in result: 
            pass 
        else: 
            result += f'{i}:{"*" * city.count(i)},'
    return result[:-1]
from collections import Counter 

def get_strings(city): 
    
    c = Counter(list(filter(str.isalpha, city.lower())))    
    return ','.join(f'{char}:{"*"*freq}' for char, freq in list(c.items()))



from collections import Counter 

def get_strings(city):
    count = Counter(city.lower().replace(" ", ""))
    return ",".join([f"{k}:{'*'*v}" for k,v in count.items()])
import json
def get_strings(city):
    cityAmount = {}
    for x in range(len(city)):
        if city[x] == ' ':
            continue
        if city[x].lower() in cityAmount:
            cityAmount[city[x].lower()] += '*'
        else:
            cityAmount[city[x].lower()] = '*'
    return ",".join(("{}:{}".format(*i) for i in list(cityAmount.items())))

def get_strings(city):
    city = "".join(city.lower().split())
    res = []
    letters = [] 
    for letter in city:
        if letter not in letters:
            res.append(str(letter +":"+"*" *city.count(letter)))
            letters.append(letter)
    return ",".join(res)
def get_strings(city):
    c={}
    for i in city.lower():
        if(i == " "):
            continue
        try:
            c[i]+=1
        except:
            c[i]=1
    res=""
    for i,j in c.items():
        res+=i+":"+"*"*j+"," 
    return res[:len(res)-1]
