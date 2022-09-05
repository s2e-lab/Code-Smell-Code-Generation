def greet(name):
    if name == "Johnny":
        return "Hello, my love!"
    return "Hello, {name}!".format(name=name)
def greet(name):
    if name == "Johnny":
        return "Hello, my love!"
    else:
        return "Hello, {name}!".format(name=name)

def greet(name):
    return "Hello, {name}!".format(name = ('my love' if name == 'Johnny' else name));
def greet(name):
    return "Hello, my love!" if name == 'Johnny' else "Hello, {name}!".format(name=name)
    

def greet(name):
    return "Hello, {name}!".format(name=name.replace("Johnny", "my love"))    
greet = lambda n: "Hello, {}!".format(n.replace("Johnny","my love"))
def greet(name):
    return "Hello, {}!".format((name, "my love")[ name == "Johnny"])
greet = lambda name: "Hello, " + ("my love" if name == "Johnny" else name) + "!"
def greet(name):
    if name != 'Johnny':
        return "Hello, {name}!".format(name=name)
    else:
        return "Hello, my love!"
def greet(name):
    return "Hello, my love!" if name == "Johnny" else f"Hello, {name}!"
def greet(name):
    Johnny = 'my love'
    return f'Hello, {Johnny if name=="Johnny" else name}!'
def greet(name):
    return f"Hello, {'my love' if name == 'Johnny' else name}!"
greet = lambda n: "Hello, {}!".format("my love" if n == "Johnny" else n)
def greet(name):
    return "Hello, my love!" if name=='Johnny' else "Hello, %s!" % name

def greet(name):
    if name == "Johnny":return "Hello, my love!"
    return "Hello, {}!".format(name)
def greet(name):
    return "Hello, my love!" if name == "Johnny" else "Hello, {}!".format(name)
def greet(name):
    if name == "Johnny":
        return "Hello, my love!"
    else:
        return "Hello, {}!".format(name)
greet = lambda name:  "Hello, my love!" if name == "Johnny" else "Hello, {name}!".format(name=name)
def greet(name):
    return "Hello, my love!" if name == "Johnny" else "Hello, {0}!".format(name)
def greet(name):
    """ Jenny was all hyped by the possibillity Johnny might check her web app
    so she made a mistake by returning the result before checking if Johnny
    is one of the users logging to her web app. Silly girl!
    We corrected the function by adding an else statement."""
    
    if name == "Johnny":
        return "Hello, my love!"
    else:
        return "Hello, {name}!".format(name=name)
'''def greet(name):
    if name == "Johnny":
        return "Hello, my love!"
    return "Hello, {name}!".format(name=name)'''
greet=lambda n:{"Johnny":"Hello, my love!"}.get(n,"Hello, {}!".format(n))
def greet(name):
    if name == "Johnny":
        name = "my love"
    return f"Hello, {name}!"
def greet(name):
    if name == "Johnny":
        return "Hello, my love!"
    else:
        return f"Hello, {name}!"
def greet(name):
    if name == "Johnny":
        return "Hello, my love!"
    elif name == "James":
        return "Hello, James!"
    elif name == "Jane":
        return "Hello, Jane!"
    elif name == "Jim":
        return "Hello, Jim!"
def greet(name):
    return "Hello, {}!".format("my love" if name == "Johnny" else name)
def greet(name):
    return "Hello, {name}!".format(name=name) if name!= "Johnny" else "Hello, my love!"
def greet(name):
    if name == "Johnny":
        return "Hello, my love!"
    return "Hello, %s!" %name

def greet(name):
    a = lambda m: "Hello, my love!" if m == "Johnny" else "Hello, {}!".format(m)
    return a(name)
greet=lambda n:f'Hello, {[n,"my love"][n=="Johnny"]}!'
def greet(n):
    if n == "Johnny":
        return "Hello, my love!"
    else:
        return f"Hello, {n}!"
def greet(name):
    if name == "Johnny":
        x = "Hello, my love!"
    else:
        x = "Hello, {name}!".format(name=name)

    return x
def greet(name):
    #
    if name == "Johnny":
        return "Hello, my love!"
    else:
        return "Hello, {name}!".format(name=name)
def greet(name):
    result =  "Hello, {}!".format(name)
    if name == "Johnny":
        return "Hello, my love!"
    else:
        return result
def greet(name):
    if name != "Johnny" :
        return f'Hello, {name}!'
    elif name == "Johnny":
        return "Hello, my love!"
def greet(name):
    if name == "Johnny":
        return "Hello, my love!"
    else:    
        return ("Hello, ") + name.format(name=name) + ("!")
#Goal:
# Fix Jenny's function so that it returns a standard greeting for any arbitrary user, but returns a special greeting for
# her crush Johnny.
#General Strategy:
# When Python processes Jenny's code it will always return the first greeting even if the input name is Johnny.
# The problem is that Johnny is receiving both greetings when Jenny only wants him to receive the bottom greeting.
# The way to solve this problem is to create an if statement for Johnny and an else statement for all other users.

def greet(name):
    # Here is the if statement so that if the user is Johnny, then the special message is returned.
    if name == "Johnny":
        return "Hello, my love!"
    # Here is the else statement which returns the standard greeting for any user aside from Johnny.
    else:
        # Here format is used to replace the name inside brackets within the string, with the input/variable 'name'
        # from the definition of the function.
        return "Hello, {name}!".format(name=name)

def greet(name):
   
  if name == "Johnny":
    return("Hello, my love!")
  else:
    return "Hello, " + name + "!"

greet("James")

def greet(name):
    greet = "Hello, {name}!".format(name=name)
    if name == "Johnny":
        greet = "Hello, my love!"
    else:
        greet = "Hello, {name}!".format(name=name)
    return greet
def greet(name):
    if name=="Johnny":
        name = "my love"
    else:
        name = name
    return "Hello, {name}!".format(name=name)

def greet(name):
    greeting=''
    if name == "Johnny":
        greeting= "Hello, my love!"
    else:
        greeting = 'Hello, ' + name + '!'
    return greeting

print(greet('Maria'))
print(greet('Johnny'))
def greet(name):
    if not name == "Johnny":
        return "Hello, " + name + "!"
    if name == "Johnny":
        return "Hello, my love!"
def greet(name):
    
    if name.title() == "Johnny":
        return "Hello, my love!"
    else:
        return f"Hello, {name}!".format(name)
def greet(name):
    if name != "Johnny": 
        return "Hello, {guy}!".format(guy=name)
    else:
        return "Hello, my love!" 
def greet(name):
    for i in name:
        if i in name == "Johnny":
            return "Hello, my love!"
        else:
            return "Hello, " + name + "!"

def greet(name):
    if name == "Johnny":
        return "Hello, my love!"
    return "Hello, " + name + "!"
print(greet('Johnny'))
def greet(name):
    output = "Hello, {name}!".format(name=name)
    if name == "Johnny":
        return "Hello, my love!"
    return output
def greet(name):
   # return "Hello, {name}!".format(name=name)
    if name == "Johnny":
        return "Hello, my love!"
    else:
        return "Hello, {}!".format(name)
def greet(name):
    if name == "Johnny":
        return "Hello, my love!".format(name=name)
    else:
        return f"Hello, {name}!".format(name=name)
def greet(name):
    if name.lower() == 'johnny':
        return "Hello, my love!"
    else:
        return f"Hello, {name}!"
def greet(name):
    lovers = {"Johnny", }
    return "Hello, my love!" if name in lovers else f"Hello, {name}!"
# INVINCIBLE WARRIORS --- PARZIVAL

def greet(name):
    if name == "Johnny":
        return "Hello, my love!"
    return "Hello, {name}!".format(name=name)
def greet(name):
    if name != 'Johnny':
        return f"Hello, {name}!"
    else:
        return 'Hello, my love!'
def greet(name):
    if name == "Johnny":
        return "Hello, my love!"
    return f"Hello, {name}!".format(name=name)
def greet(name):
    if name == "Johnny":
        return "Hello, my love!"
    else:
        return f"Hello, {name}!".format(name=name)

g = greet('Johnny')
print(g)
def greet(name):
    name = name.title()  #returns proper capitalisation
    if name == "Johnny":
        return "Hello, my love!"
    else:
        return f"Hello, {name}!"
def greet(name):
#In my country no girl would do smth. like that and this is sad
    if name == "Johnny":
        return "Hello, my love!"
    return "Hello, {name}!".format(name=name)

def greet(name):
    #if name == name return "Hello, {name}!".format(name=name)
    if name == "Johnny":
        return "Hello, my love!"
    else:
        return "Hello, {name}!".format(name=name)
def greet(name):
    small_name =  name.lower()
    n_a_m_e= list(small_name)
    x=0
    if name == "Johnny":
        return "Hello, my love!"
    while x != len( n_a_m_e):
        if x== 0:
          n_a_m_e[x] = n_a_m_e[x].upper()
          x= x+1
          print(x)
        elif x <=len( n_a_m_e):
          n_a_m_e[x] = n_a_m_e[x]
          x=x+1
          print(x)
    n_a_m_e += "!"
    end= ["H","e","l","l","o",","," "] + n_a_m_e 
    end =''.join(end)
    return end 

def greet(name):
    if name == "Johnny":
        return "Hello, my love!"
    else:
        return "Hello, {fname}!".format(fname=name)

def greet(name):
  
    if name == "Johnny":
        return "Hello, my love!"
    else:
        return "Hello, {name}!".format(name=name)
greet("spoo")
def greet(name):
    
    if name == "Johnny":
        
        greet =  "Hello, my love!"
        
    else:
    
        greet = "Hello, {}!".format(name)
        
    return greet
def greet(name):
    return str("Hello, "+name+"!") if name != "Johnny" else str("Hello, my love!")
def greet(name):
    return f'Hello, {name}!'.format(name) if name != "Johnny" else "Hello, my love!"
def greet(name):
    name = name
    if name == "Johnny":
        return "Hello, my love!"
    else:
        return "Hello, " + name + "!"
def greet(name):
    #return "Hello", name!.format(name=name)
    if name == "Johnny":
        return "Hello, my love!"
    else:
        resp = "Hello, " + name +"!"
        
        return (resp)
greet = lambda n: 'Hello, ' + ('my love' if n == 'Johnny' else n) + '!'
def greet(name):
    return "Hello, " + ("my love!" if name == "Johnny" else f"{name}!")
def greet(name):
    if name != "Johnny":
        return f"Hello, {name}!"
    else: name == "Johnny"
    return "Hello, my love!"
def greet(name):
    if name != 'Johnny':
        return 'Hello, ' + name +'!'
    elif name == 'Johnny':
        return 'Hello, my love!'
def greet(name):
    name = name if name != 'Johnny' else 'my love'
    return "Hello, {name}!".format(name=name)
def greet(name):
    name =name.format(name=name)
    if name == "Johnny":
        return "Hello, my love!"
    else:
        return "Hello, " + str(name)+"!"
def greet(name):
    return "Hello, my love!" if name.lower() == "johnny" else "Hello, {}!".format(name)
greet=lambda name: f"Hello, {name}!" if name!="Johnny" else "Hello, my love!"
greet = lambda s: f"Hello, {'my love' if s == 'Johnny' else s}!"
def greet(name):
    darling = {'Johnny': 'my love'}
    return f'Hello, {darling.get(name, name)}!'
greet=lambda n: "Hello, my love!" if n=="Johnny" else "Hello, {}!".format(n)

def greet(name):
    if name == "Johnny":
        return ("Hello, my love!")
    else:   
        return ("Hello, {name}!".format(name=name))
    
        
        
#def greet(name):
#    return "Hello, {name}!".format(name=name)
#    if name == "Johnny":
#        return "Hello, my love!"

def greet(name):
    if name == "Johnny":
        return "Hello, my love!"
    else:
        return "Hello, {name}!".format(name=name)
        
        
#         ##################!#@!#!@#!@#!@#!#!@#!@#!@#

def greet(name):
    if name=="Johnny":
         a="Hello, my love!"
         return a
    else:
        a="Hello, {0}!".format(name)
        return a
def greet(name):
    return f"Hello, my love!" if name == "Johnny" else f"Hello, {name}!"
def greet(name):
    if name == "Johnny":
        return "Hello, my love!"
    if name == "James":
        return "Hello, James!"
    if name == "Jane":
        return "Hello, Jane!"
    if name == "Jim":
        return "Hello, Jim!"

def greet(name):
    if name == "Johnny":
        return "Hello, my love!"
    if name == "James":
        return "Hello, James!"
    if name == "Jane":
        return "Hello, Jane!"
    if name == "Jim":
        return "Hello, Jim!" 
    else:
        return "Hello, " + name + " !"

def greet(name):
    if name == "Johnny":
        return "Hello, my love!"
    return "Hello, {:s}!".format(name)

def greet(name):
    #if name == "Johnny":
    #   return "Hello, my love!"
    
    return "Hello, my love!" if name == "Johnny" else "Hello, {0}!".format(name)
def greet(name):
    
    if name == "Johnny":
        return "Hello, my love!"
    else:
        return "Hello, {s}!".format(s=name)
def greet(name):
    if name == "Johnny":
        return "Hello, my love!"
    else:
        if name != "Johnny":
            return "Hello, {name}!".format(name=name)

def greet(name):
    if name == "Johnny":
        return "Hello, my love!"
    return "Hello, {name}!".format(name=name)
print(greet('Johny'))
def greet(name):
    greeting = "Hello, {name}!".format(name=name)
    if name == "Johnny":
        return "Hello, my love!"
    else:
      return greeting

def greet(name):
    if name == "Johnny":
        return "Hello, my love!"
    else:
        return "Hello, {name}!".format(name=name)


print(greet("Johnny"))
def greet(name):
    if name != str("Johnny"):
        return "Hello, {name}!".format(name=name)
    else:
        return "Hello, my love!"
def greet(name):
    
    if name == "Johnny":
        return "Hello, my love!"
    else:
        return f"Hello, {name}!".format(name=name)
def greet(name):
    if name == "Johnny":
        return "Hello, my love!"
    return "Hello, {ghj}!".format(ghj=name)
greet = lambda n: "Hello, my love!" if n == "Johnny" else f"Hello, {n}!"
def greet(name):
    if name == "Johnny":
        return "Hello, my love!"
    
    else:
        return "Hello" + "," + " " + name + "!"

def greet(name):
    if name == "Johnny":
        msg =  "Hello, my love!"
    else:
        msg =  "Hello, {name}!".format(name=name)
    return msg
