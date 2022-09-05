def boolean_to_string(b):
    return str(b)
def boolean_to_string(b):
    return 'True' if b else 'False'
boolean_to_string = str    
def boolean_to_string(b):
    if b:
        return "True"
    return "False"
def boolean_to_string(b):
    return ('False', 'True')[b]

boolean_to_string = lambda x: "True" if x else "False"
# this code return the string version of the boolean input
def boolean_to_string(b):
    return str(b)
def boolean_to_string(b):
    if b == True or b == False:
        return str(b)
def boolean_to_string(b):
    if b == True:
        return "True"
    else:
        return "False"
boolean_to_string = lambda b : ["False", "True"][b];
def boolean_to_string(b):
    d = {True : "True", False : "False"}
    return d[b]
def boolean_to_string(b: bool) -> str:
    return str(b)
def boolean_to_string(b):
    if(((((((((False==0.0)==1)==True)==1.0)!=False)!=0)==int(True))!=int(False))!=0) == b:
        return 'True'
    return 'False'
def boolean_to_string(b):
    return b*"True" or "False"
def boolean_to_string(b):
    return "FalseTrue"[5*b:5+4*b]
def boolean_to_string(b):
    if b: return "True"
    if not b: return "False"
boolean_to_string=lambda b: "True" if b else "False"

def boolean_to_string(b):
    return ['False', 'True'][b]
def boolean_to_string(b):
    return format(b)
def boolean_to_string(b):
    if isinstance(b, bool):
        return "True" if b else "False"
    return "Wrong type"

def boolean_to_string(b):
    return f"{b}"
def boolean_to_string(b):
    return repr(b)
def boolean_to_string(b):
    return b.__str__()
def boolean_to_string(b):
    #your code here
    if b:
       print('When we pass in true, we want the string "true" as output')
       return 'True'
    else:
        print('When we pass in false, we want the string "false" as output')
        return 'False'
boolean_to_string = repr
def boolean_to_string(b):
    return (b and "True") or "False"
boolean_to_string=lambda b: str(b)
def boolean_to_string(b):
    for i in str(b):
        if b == True:
            return 'True'
        else:
            return 'False'

boolean_to_string = lambda val: str(val)
def boolean_to_string(b):
    a = lambda x: "True" if x else "False"
    return a(b)
def boolean_to_string(b):
    if (b == True) :
        return "True"
    elif (b == False) :
        return "False"
    else :
        return "invalid"
boolean_to_string = lambda boolean: str(boolean)
    
    
# Implement a function which convert 
# the given boolean value into its string representation.

def boolean_to_string(b):
    return "True" * (b == True) + "False" * (b == False)
def boolean_to_string(b):
    str1 = str(b)
    if str1 == 'True':
        return 'True'
    return 'False'
def boolean_to_string(b):
    if b:
        return 'True'
    if b == 0:
        return 'False'
def boolean_to_string(b):
    dic=  {True:"True",False:"False"}
    return dic[b]
def boolean_to_string(b):
    if int(b) is 1:
        return "True"
    elif int(b) is 0:
        return "False"

def boolean_to_string(b):
    bool = {True:'True',False:'False'}
    return bool[b]
boolean_to_string=lambda b:f'{b}'
def boolean_to_string(b):
    b=str(b)
    if b=="True":
        return "True"
    else:
        return "False"
def boolean_to_string(b):
    if b == True:
        b = "True"
        return b
    elif b == False or 0:
        b = "False"
        return b

def boolean_to_string(b):
    boolean = str(b)
    return boolean
def boolean_to_string(b):
    _stringstring = str(b)
    return _stringstring
def boolean_to_string(b):
  return repr(b)
  return str(b) 
  return "True" if b else "False"
def boolean_to_string(b):
    return str(b) if True else str(b)
def boolean_to_string(b):
    return "".join([i for i in str(b)]) 

def boolean_to_string(b):
    string1 = str(b)
    return string1

def boolean_to_string(b):
    if b == True:
      return str("True")
    return str("False")  
def boolean_to_string(b):
    print(b)
    return 'True' if b  else 'False'

def boolean_to_string(b):
    return "True" if str(b) == "True" else 'False'
def boolean_to_string(b):
    result=False
    if b==True:
        result=True
    else:
        result=False
        
    return str(result)    
def boolean_to_string(b):
    if b == True:
       return "True"
       pass
    return "False"
    #your code here

def boolean_to_string(b):
    if b:
        return "True"
    elif b != True:
        return "False"
def boolean_to_string(b):
    if b == True: return "True"
    if b != True: return "False"
def boolean_to_string(b):
    #your code 
    return str(b)
def boolean_to_string(b):
    F = 'False'
    T = 'True'
    if b == True:
        return T
    if b == False:
        return F
def boolean_to_string(n):
    return str(n)
def boolean_to_string(b):
    return str(b) if str(b).lower()=='true' else str(b)
def boolean_to_string(b):
    if b == True: return "True"
    else: return "False"
    
print(boolean_to_string(True))
def boolean_to_string(boolean):
    if boolean:
        return 'True'
    else:
        return 'False'
def boolean_to_string(b):
    
    if b:
        return "true".capitalize()
    else:
        return "false".capitalize()
def boolean_to_string(b):
    ret = 'False'
    if (b):
        ret = 'True'
    return ret
def boolean_to_string(b):
    if b==True:
       return str(True)
    else:
        b==False
        return str(False)
    return b 

def boolean_to_string(b):
    if b == True:
        a = str(b)
        print(a, ' When we pass in true, we want the string "true" as output')
    else:
        a = str(b)
        print(a)
    
    return a
def boolean_to_string(b):
    if type(b) == type(True) and b == True:
        return ("True")
    else:
        return ("False")

print((boolean_to_string(False)))

def boolean_to_string(b):
    boo = bool(b)
    if boo == True:
        return "True"
    else:
        return "False"
def boolean_to_string(b):
    b = str(b)
    if b == "True":
        
        return(b)
    if b == "False":
        return(b)
    #your code here

boolean_to_string=lambda a:str(a)
def boolean_to_string(b):
    if b is True:
        return(str(b))
    else:
        return("False")
def boolean_to_string(b):
    return (lambda x: "True" if x else "False")(b)
def boolean_to_string(b):
    if b == True:
        return("True")
    else:
        return("False")

bool = True
boolean_to_string(bool)
bool = False
boolean_to_string(bool)

def boolean_to_string(b):
    #your code here
    res = ''
    if b:
        res = "True"
    else:
        res = "False"
    return res
def boolean_to_string(b):
    out = 'False'
    if b:
        out = "True"
    return out

def boolean_to_string(b):
    #your code here
    return str(b)
    
print(boolean_to_string(True))
def boolean_to_string(b):
    if b == 1:
        return "True"
    elif b == 0:
        return "False"
def boolean_to_string(b):
    b=str(b)
    thing=0
    if b=="True":
        thing="When we pass in true, we want the string 'true' as output"
    else:
        thing="When we pass in false, we want the string 'false' as output"
    return b
    return thing
def boolean_to_string(b):
    if b == True:
        string = 'True'
    elif b == False:
        string = 'False'
    return string
    
    #your code here

def boolean_to_string(b):
    if b:
        return "True"
    elif b is False:
        return "False"

def boolean_to_string(b):
    return str(b)
    # return 'True' if b else 'False'

def boolean_to_string(b):
    return "True" if b else "False"
#Completed by Ammar on 10/8/2019 at 01:18PM.

def boolean_to_string(b):
    return "True" if True==b else "False"
def boolean_to_string(b):
    boo = ''
    if b:
        boo = 'True'
    else:
        boo = 'False'
    return boo
def boolean_to_string(b):
    return 'True' if b>0 else 'False'
def boolean_to_string(b):
    if b == 0:
        return 'False'
    if b == 1:
        return 'True'
def boolean_to_string(b):
    if b:
        b="True"
    else:
        b="False"
    return b
def boolean_to_string(b):
    if b:
        return 'True'
    else:
        return 'False'
        
        #fifnish!!!

def boolean_to_string(b):
    if b == b:
        return str(b)
def boolean_to_string(b):
    if b == False:
        return "False"
    return "True"
def boolean_to_string(b):
    a = ''
    if b == True:
        a = 'True'
    else:
        a = 'False'
        
    return a
def boolean_to_string(bol):
    return str(bol)
#prints output 'true' for input 'true/'false' for input 'false'
def boolean_to_string(b): 
   return str(b)
def boolean_to_string(b):
    if b:
        return "True"
    else:
         return "False"#I solved this Kata on 6/27/2019 02:53 AM #hussamSindhuCodingDiary
def boolean_to_string(b):
    if b == 1:
        statement = 'True'
    else:
        statement = 'False'
    
    return statement
def boolean_to_string(b):
    return ["True" if b else "False"][0] # Easy 8 kyu
def boolean_to_string(b):
    if b == 0:
        return "False"
    else:
        return "True"

def boolean_to_string(b):
    #your code he
    p = str(b);
    print (type(p))
    print (p)
    return p
def boolean_to_string(boolean):
  return "True" if boolean == True else "False"
def boolean_to_string(b):
    #your code here
    ans = "True" if b == True else "False"
    return ans
def boolean_to_string(b):
    print((str(b)))
    return str(b)
    

def boolean_to_string(b):
    bul = str(b)
    return bul
