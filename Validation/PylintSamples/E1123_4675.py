def set_alarm(employed, vacation):
    return employed and not vacation
def set_alarm(employed, vacation):
    # Your code here
    if employed:
        if vacation:
            return False
        return True
    return False
def set_alarm(employed, vacation):
    return employed==True and vacation==False
set_alarm=lambda *a:a==(1,0)
set_alarm=lambda *a:a==(1,0)
set_alarm = lambda *a: a == (True, False) # [1]
set_alarm = lambda employed, vacation: (employed, vacation) == (True, False) # [2]
set_alarm = lambda employed, vacation: (employed == True) and (vacation == False)
set_alarm = lambda employed, vacation: employed and not vacation

# The most voted best practice solution:

def set_alarm(employed, vacation):    # [3]
    return employed and not vacation  #

def set_alarm(employed, vacation):
    if employed and vacation is False:
        return True
    else:
        return False
def set_alarm(employed, vacation):
    return True if employed is True and vacation is False else False
def set_alarm(e, v):
    return False if e == True and v == True else False if e == False and v == True else False if e == False and v == False else True
set_alarm=int.__gt__
set_alarm=lambda e,v:e>v
def set_alarm(employed, vacation):
    if employed == 1 and vacation == 0: return True
    else: return False
def set_alarm(employed, vacation):
    return min(employed,employed!=vacation)
def set_alarm(is_employed, on_vacation):
    return is_employed and not on_vacation
# set_alarm = lambda employed,vacation:True if employed == True and vacation == False
def set_alarm(employed,vacation):
    if employed == True and vacation == False:
        return True
    else:
        return False

def set_alarm(employed, vacation):
    if employed == vacation:
        return False   
    return True if employed == True and vacation == False else False
    return False if employed == False and vacation == True else True
set_alarm=lambda employed, vacation: False if employed==vacation else not vacation
def set_alarm(employed, vacation):
    return ( True if employed and vacation is False else False) 
def set_alarm(employed, vacation):
    while employed == True and vacation == True:
        return False
    while employed == False and vacation == True:
        return False
    while employed == True and vacation == False:
        return True
    while employed == False and vacation == False:
        return False
    
set_alarm(True, True)
def set_alarm(employed, vacation):
    return True if employed and not vacation else False
def set_alarm(a, b):
    return a - b == 1
def set_alarm(employed, vacation):
    return employed > vacation
def set_alarm(employed, vacation):
    return employed & ~vacation
set_alarm=lambda e,v:e>>v
set_alarm=lambda e,v:e-v>0
def set_alarm(employed, vacation):
    a = employed
    b = vacation
    
    if a == True:
        if b == False:
            return True
    if a == False:
        if b == False:
            return False
    if a == False and b == True:
        return False
    if a and b == True:
        return False
        




#     if a == True:
#         return True
#         if a and b == True:
#             return False
#     else:
#         return False

from operator import rshift as set_alarm

set_alarm=lambda e,v:e and not v
def set_alarm(employed, vacation):
    # Your code here
    return True if employed == True and vacation == False else False
def set_alarm(employed, vacation):
    if employed == vacation:
        return False
    if employed == True:
        return  True
    return False
def set_alarm(employed, vacation):
    if vacation == False and employed == True:
        return True
    return False

def set_alarm(e,v):
    return False if e and v or e==False and v==True or [e,v]==[False,False] else True
def set_alarm(employed, vacation):
    if employed == False and vacation == True:
        return False
    elif employed == True and vacation == False:
        return True
    else:
        return False
def set_alarm(employed, vacation):
    boolean1 = employed
    boolean2 = vacation
    
    
    if boolean1 == True and boolean2 == False:
        return True
    else:
        return False
def set_alarm(employed, vacation):
    return employed if not (employed == vacation == True) else False
def set_alarm(employed, vacation):
    return employed^vacation and employed == True
def set_alarm(employed, vacation):
    if employed == True:
        if vacation == True:
            return False
        if vacation == False:
            return True
    if employed == False:
        return False
def set_alarm(em, va):
    if em == True and va == True:
        return False
    elif em == True and va == False:
        return True
    elif em == False and va == False:
        return False
    elif em == False and va == True:
        return False
    # Your code here

def set_alarm(employed, vacation):
    if employed==True and vacation==True:
        alarma=False
    elif employed==False and vacation==True:
        alarma=False
    elif employed==False and vacation==False:
        alarma=False
    elif employed==True and vacation==False:
        alarma=True
    return alarma

def set_alarm(employed, vacation):
    if employed == True and vacation == False:
        return True
    else:
        return False
    
    """ FIRST THOUGHT
    if employed == True and vacation == False:
        return True
    elif employed == True and vacation == True:
        return False
    elif employed == False and vacation == False:
        return False
    elif employed == 
    """
def set_alarm(employed, vacation):
    if employed:
        if vacation:
            return False
    if employed:
        return True
    if vacation:
        return False
    else:
        return False
def set_alarm(employed: bool, vacation: bool) -> bool:
    return employed == True and vacation == False
def set_alarm(employed, vacation):
    return False if not employed and vacation else employed ^ vacation
def set_alarm(e, v):
    return e&(e^v)
def set_alarm(employed, vacation):
    import numpy
    return employed and not vacation
def set_alarm(employed, vacation):
    if employed == True and vacation == True:
        return False
    elif employed == False and vacation == True:
        return False
    if employed == False and vacation == False:
        return False
    if employed == True and vacation == False:
        return True
def set_alarm(employed, vacation):
    if employed==True and vacation==False:
        res = True
    else:
        res = False
    return res
def set_alarm(employed, vacation):
    if employed == vacation or vacation == True:
        return False
    else:
        return True
def set_alarm(f, s):
    return f == True and s == False
set_alarm = lambda e, v: 1 if [1, 0] == [e, v] else 0
def set_alarm(employed, vacation):
    return not(employed and vacation or not employed) 
def set_alarm(employed, vacation):
    if vacation is True:
        return False
    else:
        if employed is True:
            return True
    return False
def set_alarm(employed, vacation):
    # Your code here
    while employed == True:
        if vacation == True:
            return False
        else:
            return True
    return False
def set_alarm(employed, vacation):
    return employed and not vacation # becomes True if employed and not on vacation
def set_alarm(employed, vacation):
    if (employed == True) & (vacation == False):
        return True
    return False
def set_alarm(employed, vacation):
    if vacation is True:
        return False
    elif employed == True:
        return True
    elif employed == False:
        return False
set_alarm = lambda employed, vacation: employed ^ vacation and employed
def set_alarm(e, vacation):
    return e and not vacation
def set_alarm(employed, vacation):
    '''This code will return the overall boolean state of the alarm based upon the input parameters'''
    return True if employed==True and vacation==False else False
def set_alarm(d,n):
    return d>n
def set_alarm(employed, vacation):
    if(employed == True & vacation == True):
        return False
    if(employed == False & vacation == False):
        return False
    if(employed == False & vacation == True):
        return False
    else:
        return True
def set_alarm(employed, vacation):
    a = employed
    b = vacation
    if a == True and b == True:
        return False
    if a == False and b == True:
        return False
    if a == False and b == False:
        return False
    if a == True and b == False:
        return True
def set_alarm(employed, vacation):
    return employed and not vacation
    
    return false
def set_alarm(employed, vacation):
        if (employed == True and vacation == True) or (employed== False and vacation == True) or (employed == False and vacation == False):
            return False
        elif employed == True and vacation == False:
            return True
def set_alarm(employed, vacation):
    if employed == True and vacation == True:
        return False
    elif employed == False and vacation == False:
        return False
    elif employed == False:
        return False
    return True

def set_alarm(employed, vacation):
    if employed == vacation:
        return False
    elif employed == False and vacation == True:
        return False
    else:
        return True
def set_alarm(x, y):
    if x == True and y == True:
        return False
    elif x == False and y == True:
        return False
    elif x == False and y == False:
        return False
    else:
        return True
def set_alarm(employed, vacation):
    if employed == False and vacation == True:
        return False
    else:
        return employed != vacation
def set_alarm(employed, vacation):
    if employed == True and vacation == True:
        return False
    elif employed == False and vacation == True:
        return False
    elif employed == False and vacation == False:
        return False
    return True
def set_alarm(employed, vacation):
    if employed != vacation:
        if employed == True:
            return True
        else:
            return False
    else: 
        return False
def set_alarm(employed, vacation):
    if employed == True and vacation == False:
        return True
    elif (employed == True and vacation == True) or (employed == False and vacation == True) or (employed == False and vacation == False):
        return False

def set_alarm(employed, vacation):
    # Your code here
    if employed == True and vacation == True:
        return False
    elif employed == True and vacation == False:
        return True
    elif employed == False and vacation == True:
        return False
    elif employed == False and vacation == True:
        return False
    elif employed == False and vacation == False:
        return False
def set_alarm(employed, vacation):
    return (False, True)[employed and not vacation]
set_alarm = lambda employed,vacation: (employed,vacation) == (True,False)
def set_alarm(employed, vacation):
    if employed == True:
        if vacation == True:
            return False
        else:
            return True
    elif employed == False:
        if vacation == True:
            return False
        else: 
            return False

def set_alarm(employed, vacation):
    if employed is True and vacation is True:
        return False
    if employed is False and vacation is True:
        return False
    if employed is True and vacation is False:
        return True
    if employed is False and vacation is False:
        return False

def set_alarm(employed, vacation):
   
   result=''
   
   if employed==True and vacation==False:
      result=True
   else:
      result=False
      
   return result   
import unittest


def set_alarm(employed, vacation):
    return employed and not vacation


class TestSetAlarm(unittest.TestCase):
    def test_should_return_false_when_employed_is_true_and_vacation_is_true(self):
        self.assertEqual(set_alarm(employed=True, vacation=True), False)

    def test_should_return_false_when_employed_is_false_and_vacation_is_false(self):
        self.assertEqual(set_alarm(employed=False, vacation=False), False)

    def test_should_return_false_when_employed_is_false_and_vacation_is_true(self):
        self.assertEqual(set_alarm(employed=False, vacation=True), False)

    def test_should_return_false_when_employed_is_true_and_vacation_is_false(self):
        self.assertEqual(set_alarm(employed=True, vacation=False), True)

def set_alarm(employed, vacation):
    if vacation: return False
    if employed and vacation: return False
    if employed and not vacation: return True
    return False
set_alarm = lambda a, b: a and a != b
def set_alarm(employed, vacation):
    # Your code here
    if employed == True and vacation == True:
        return False 
    else:    
        return False if employed == False else True 
def set_alarm(employed: bool, vacation: bool) -> bool:
    return (employed and not vacation)
def set_alarm(employed, vacation):
    return (employed and True) and (not vacation and True)
def set_alarm(employed, vacation):
    if employed == False:
        if vacation == True:
            return False
        if vacation == False:
            return False
    else:return bool(vacation) ^ bool(employed)
def set_alarm(e, v):
    return e & (not v)
def set_alarm(e, v):
    if e == True and v == True:
        return False
    elif e == False and v == True:
        return False
    elif e == False and v == False:
        return False
    elif e == True and v == False:
        return True
def set_alarm(employed, vacation):
    # Your code here
    return False if employed == False else False if vacation == True else True
def set_alarm(e, v):
    return False if e==v else e
def set_alarm(employed, vacation):
    
    if vacation == True:
        return False
        
    if employed == False and  vacation == False:
        return False
        
    else:
        return True
def set_alarm(employed, vacation):
    # Your code here
    if employed:
        if vacation == False:
            return True
        if vacation:
            return False
    else:
        return False
def set_alarm(employed, vacation):
    return (employed != vacation and employed ==1)
def set_alarm(employed, vacation):
    x = (employed, vacation)
    if x == (True,False):
        return True
    return False
def set_alarm(employed, vacation):
    # Your code here
    print (employed)
    if ( employed == True and vacation == False ) :
        return True
    else :
        return False
def set_alarm(e, v):
    if e==True and v==True:
        return False
    elif e==True and v==False:
        return True
    elif e==False and v==True:
        return False
    else:
        return False
def set_alarm(employed, vacation):
    return employed != vacation and employed == True
def set_alarm(employed, vacation):
    s = ''
    if employed == True:
      if vacation == True:
        s = False
      else:
        s = True
    else:
      s = False
    return s
def set_alarm(employed , vacation):
    return not employed if employed and employed == vacation else employed
def set_alarm(employed, vacation) -> bool:
    # Your code here
    if employed == True and vacation == True:
        return False
    elif employed == True and vacation == False:
        return True
    elif employed == False and vacation == True:
        return False
    elif employed == False and vacation == False:
        return False
def set_alarm(employed, vacation):
    return False if employed == False or employed and vacation == True else True
def set_alarm(employed, vacation):
    if bool(employed) and not bool(vacation):
        return True
    else:
        return False
set_alarm=lambda employed,vacation:employed!=vacation if employed else employed
def set_alarm(e, v):
    if v == False and e == True:
        return True
    else:
        return False
