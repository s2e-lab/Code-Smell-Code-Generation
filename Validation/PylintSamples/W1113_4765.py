class Hero(object):
    def __init__(self, name='Hero'):
        self.name = name
        self.position = '00'
        self.health = 100
        self.damage = 5
        self.experience = 0
class Hero(object):
    def __init__(self, name="Hero", position="00",
        health=100, damage=5, experience=0):
        self.name = name
        self.position = position
        self.max_health = health #trust me, you want to have this as well
        self.health = health 
        self.damage = damage
        self.experience = experience
class Hero(object):
    def __init__(self, name='Hero'):
        self.name = name
        self.experience = 0
        self.health = 100
        self.position = '00'
        self.damage = 5
class Hero(object):
    __slots__ = ['name', 'position', 'health', 'damage', 'experience']

    def __init__(self, name='Hero'):
        self.name = name
        self.position = '00'
        self.health = 100
        self.damage = 5
        self.experience = 0
class Hero(object):
    def __init__(self, name='Hero', pos='00', health=100, dmg=5, exp=0):
        self.name = name
        self.position = pos
        self.health = health
        self.damage = dmg
        self.experience = exp
class Hero(object):
  def __init__(self, name = 'Hero'):
    self.name, self.position, self.health, self.damage, self.experience = name, '00', 100, 5, 0
class Hero(object): position,health,damage,experience,__init__ = '00',100,5,0,lambda self, name='Hero': setattr(self, 'name', name)
class Hero(object):
    def __init__(self, name=None):
        self.name = name or 'Hero'
        self.position = '00'
        self.health = 100
        self.damage = 5
        self.experience = 0
class Hero(object):
    def __init__(self, name='Hero'):
        self.damage = 5
        self.experience = 0
        self.health = 100
        self.name = name
        self.position = '00'

class Hero:
    """ Character: Hero. """

    def __init__(
        self,
        name: str = "Hero",
        position: str = "00",
        health: int = 100,
        damage: int = 5,
        experience: int = 0
    ):
        """ Prepare data. """
        self.name = name
        self.position = position
        self.health = health
        self.damage = damage
        self.experience = experience
class Hero(object):
    def __init__(self, name = "Hero"):
        self.name = name
        self.health = 100
        self.position = "00"
        self.damage = 5
        self.experience = 0
        

class Hero(object):
    position = '00'
    health = 100
    damage = 5
    experience = 0
    __init__ = lambda self, name='Hero': setattr(self, 'name', name)

class Hero(object):
    def __init__(self, name=None, position=None, health=None, damage=None, experience=None):
        self.name = name or 'Hero'
        self.position = position or '00'
        self.health = health or 100
        self.damage = damage or 5
        self.experience = experience or 0

class Hero(object):
    def __init__(self, *name):
        self.name = "Hero"
        for a in name:
            self.name = name[0]
        self.experience = 0
        self.position = "00"
        self.health = 100
        self.damage = 5
class Hero():
    def __init__(self, name = "Hero", position = "00", health = 100, damage = 5, experience = 0):
        self.name = name
        self.position = position
        self.health = health
        self.damage = damage
        self.experience = experience
        
        
myhero = Hero("Hero", "00", 100, 5, 0)
        





class Hero(object):
    def __init__(self, name="Hero"):
        self.name = name
        self.position = '00'
        self.health = 100
        self.damage = 5
        self.experience = 0
            
    
'''define a Hero prototype to be used in a terminal game. 
The hero should have the following attributes:

attribute   value
name    user argument or 'Hero'
position    '00'
health  100
damage  5
experience  0'''
class Hero(object):
    def __init__(self,name =None):
        if name:
            self.name = name
        else:
            self.name = "Hero"
        self.position = "00"
        self.health = 100
        self.damage = 5
        self.experience = 0
class Hero(object):
    def __init__(self, name='Hero'):
        self.name = name
        self.position = '00'
        self.health = 100
        self.damage = 5
        self.experience = 0
        return

class Hero(object):
    def __init__(self, name='Hero', experience=0,health=100 ,position="00", damage=5 ):
        self.name=name
        self.experience=experience
        self.damage=damage
        self.position=position
        self.health=100
class Hero(object):
    def __init__(self, name='Hero'):
        self.name= name
        self.experience=0
        self.health=100
        self.position='00'
        self.damage=5
        return None
class Hero():
    def __init__(self, name="Hero", experience=0, health=100, position='00', damage=5):
        self.name=name
        self.experience=experience
        self.health=health
        self.position=position
        self.damage=damage


myHero = Hero()
class Hero(object):
    def __init__(self, s='Hero'):
        self.name = s
        self.experience = 0
        self.health = 100
        self.damage = 5
        self.experience = 0
        self.position = '00'
class Hero(object):
    """
    Prototype for Hero
    """
    def __init__(self, name='Hero'):
        self.name = name
        self.position = '00'
        self.health = 100
        self.damage = 5
        self.experience = 0
class Hero(object):
    def __init__(self, name=None):
        self.name = name if name else 'Hero'
        self.position = '00'
        self.health = 100
        self.damage = 5
        self.experience = 0
class Hero:
    position  = '00'
    health = 100
    damage = 5
    experience = 0
    
    def __init__(self, name='Hero'):
        self.name = name 
        
myHero = Hero('Hero')
class Hero(object):

    def __init__(self,name="Hero",experience = 0,health = 100,position = '00',damage = 5):
    
        
        self.name = name
        self.experience = experience
        self.health = health
        self.position = position
        self.damage = damage
        
Hero()
        
        

class Hero(object):
    experience = 0
    position = '00'
    health = 100
    damage = 5
    def __init__(self, name='Hero'):
        self.name = name


        


class Hero(object):
    name = "Hero"
    position = "00"
    health = 100
    damage = 5
    experience = 0
    def __init__(self, name = "Hero",position = "00",health = 100,damage = 5,experience = 0):
        self.name = name
        self.position = position
        self.health = health
        self.damage = damage
        self.experience = experience

class Hero(object):
    def __init__(self, name="Hero"):
        self.name = name or "Hero"
        self.position = "00"
        self.health = 100
        self.damage = 5
        self.experience = 0

class Hero(object):
    def __init__(self, name='Hero'):
        #Add default values here
        self.name = name
        self.experience = 0
        self.damage = 5
        self.health = 100
        self.position = '00'
class Hero(object):
    def __init__(self, name='Hero', experience=0, health=100, position='00', damage=5):
        self.name = name
        self.position = position
        self.health = health
        self.damage = damage
        self.experience = experience
class Hero(object):
    def __init__(self, *name):
        #Add default values here
        print(name)
        self.name = name[0] if name else 'Hero' 
        self.position = '00'
        self.health = 100
        self.damage = 5
        self.experience = 0
class Hero(object):
    def __init__(self, name="Hero", exp=0, health=100, pos="00", damage=5):
        self.name = name
        self.experience = exp
        self.health = health
        self.position = pos
        self.damage = damage
        

class Hero(object):
    def __init__(self,*args):
        self.name=args[0] if args else "Hero"
        self.position='00'
        self.health=100
        self.damage=5
        self.experience=0
class Hero():
    def __init__(self, name = "Hero"):
        self.name = name
        self.experience = 0
        self.health = 100
        self.position = '00'
        self.damage = 5
myHero = Hero()
#Solved on 26th Nov, 2019 at 04:49 PM.

class Hero(object):
    def __init__(self, name = 'Hero'):
        position = '00'
        health = 100
        damage = 5
        experience = 0
    
        self.name = name
        self.position = position
        self.health = health
        self.damage = damage
        self.experience = experience
class Hero(object):
    name = ''
    position = '00'
    health = 100
    damage = 5
    experience = 0
    

    def __init__(self, name='Hero'):
        self.name = name
        self.position = '00'
        self.health = 100
        self.damage = 5
        self.expirience = 0

class Hero:
    def __init__(self, name= "Hero", experience = 0, health = 100, position = "00", damage = 5 ):
        self.name = name
        self.experience = experience
        self.health = health
        self.position = position
        self.damage = damage
    
    def anzeige(self):
        return self.name
        return self.experience
        return self.health
        return self.position
        return self.damage


class Hero(object):
    def __init__(self, name='Hero', experience=0,damage=5,health=100,position='00'):
        self.name=name
        self.experience=experience
        self.damage=damage
        self.health=health
        self.position=position

class Hero(object):
    def __init__(self, name="Hero"):
        self.name=name
        self.position="00"
        self.health=100
        self.damage=5
        self.experience=0
Hero()        
class Hero(object):
    def __init__(self, name= 'Hero',position='00',health=100,damage=5,experience=0):
        self.name=name
        self.position=position
        self.health=health
        self.damage=damage
        self.experience=experience  #I solved this Kata on  7/30/2019  07:00 AM...#Hussam'sCodingDiary
class Hero(object):
    def __init__(self, name='Hero',experience=0,health=100,position='00',damage=5):
        self.name=name
        self.experience=experience
        self.health=health
        self.position=position
        self.damage=damage
    def name(self,name):
        return self.name
    def position(self):
        return self.position
    def health(self,health):
        return self.health
    def damage(self,damage):
        return self.damage
class Hero(object):
    def __init__(self, name=None): #Pass name=None to get None if no argument is passed
        if name == None: #Assignm values according to the appropriate arguments passed to the object
            self.name = 'Hero'
        else:
            self.name = name
        self.position = '00'
        self.health = 100
        self.damage = 5
        self.experience = 0
class Hero(object):
    def __init__(self, name='Hero'):
        self.experience = 0
        self.damage = 5
        self.health = 100
        self.position = '00'
        self.name = name
        pass
class Hero(object):
    def __init__(self, name = 'Hero', position = '00', health = 100, damage = 5, experience = 0):
        self.name = name
        self.position = position
        self.health = health
        self.damage = 5
        self.experience = 0
class Hero(object):
    def __init__(self, name: str='Hero'):
        self.name = name
        self.experience = 0
        self.health = 100
        self.position = '00'
        self.damage = 5
class Hero(object):
    def __init__(self,*name):
        if name:
            self.name=str(name)[2:-3]
        else:
            self.name='Hero'
        self.experience=0
        self.health=100
        self.position='00'
        self.damage=5
        pass
class Hero(object):
    def __init__(self, name=''):
        #Add default values here
        self.name = name or 'Hero'
        self.position = '00'
        self.health = 100
        self.damage = 5
        self.experience = 0
class Hero(object):
    def __init__(self, _name='Hero'):
        self.name = _name
        self.position = '00'
        self.health = 100
        self.damage = 5
        self.experience = 0
class Hero(object):

    def __init__(self, name=None):
        if name is None:
            self.name="Hero"
        else:
            self.name=name
        self.experience=0
        self.health =100
        self.position = '00'
        self.damage= 5
class Hero(object):
    def __init__(self,name="Hero"):
        #Add default values here
        self.name, self.experience, self.health, self.position, self.damage = name,0,100,"00",5
class Hero(object):
    def __init__(self, *name):
        if name:
            self.name = name[0]
        else:
            self.name = 'Hero'
        self.experience = 0
        self.health = 100
        self.position = '00'
        self.damage = 5
        pass
class Hero(object):
    def __init__(self, name=''):
        self.name = name if name else 'Hero'
        self.position = '00'
        self.health = 100
        self.damage = 5
        self.experience = 0
class Hero(object):
    name = 'Hero'
    experience = 0
    health = 100
    position = "00"
    damage = 5
    
    def __init__(self, name = "Hero"):
        self.name = name
class Hero(object):
#    def __init__(self, name):
        #Add default values here
#       pass
    def __init__(self, name='Hero', experience=0, health=100, position='00', damage=5):
        self.name = name
        self.experience = experience
        self.health = health
        self.position = position
        self.damage = damage
#myHero = Hero()

class Hero(object):
    #Hero class
  
    def __init__(self, name='Hero'):
        #Add default values here
        self.name = name        
        self.position = '00'
        self.health = 100
        self.damage = 5
        self.experience = 0
        pass
class Hero(object):
    def __init__(self, name=("Hero")):  # Hero is default name
        self.name = name                # if name is given, than overwrite it
        self.experience = 0
        self.health = 100
        self.position = "00"
        self.damage = 5       
class Hero(object):
    def __init__(self, *name):
        if name is ():
            self.name = 'Hero'
        else:
            self.name = name[0]
        self.position = '00'
        self.health = 100
        self.damage = 5
        self.experience = 0
class Hero:
    def __init__(self,n='Hero',p='00',h=100,d=5,e=0):
        self.name =n
        self.position=p
        self.health=h
        self.damage=d
        self.experience=0
class Hero(object):
    def __init__(self, name='Hero'):
        self.name = name
        self.health = 100
        self.damage = 5
        self.experience = 0
        self.position = '00'
        #Add default values here
        pass
class Hero():
    def __init__(self, name="Hero", *args):
        self.name = name
        self.experience = 0
        self.health = 100
        self.position = "00"
        self.damage = 5
class Hero(object):
    def __init__(self, name="Hero",experience=0,position="00",health=100,damage=5):
        self.name=name
        self.experience=experience
        self.position=position
        self.health=health
        self.damage=damage
        #Add default values here

# class Hero(object):
#     def __init__(self, name):
#         self.name = 'Hero'
#         self.position = '00'
#         self.health = 100
#         self.damage = 5
#         self.experience = 0 


class Hero(object):
    def __init__(self, name='Hero'):
        self.name = name
        self.position = '00'
        self.health = 100
        self.damage = 5
        self.experience = 0

class Hero(object):
    def __init__(self, name=None):
        #Add default values here
        self.name= "Hero" if name is None else name
        self.position= '00'
        self.health= 100
        self.damage= 5
        self.experience= 0
class Hero(object):
    def __init__(self,name=None):
        if name is None:
            name='Hero'
        self.name=name
        self.experience=0
        self.health=100
        self.position='00'
        self.damage=5

class Hero(object):
    def __init__(self, name="Hero",position="00",health=100,damage=5,experience=0):
        self.name=name
        self.position=position
        self.health=health
        self.damage=damage
        self.experience=experience
        
        """name="Hero"
        position="00"
        health=100
        damage=5
        experience=0"""
class Hero(object):
    def __init__(self, name='Hero',pos='00',h=100,d=5,e=0):
        self.name=name
        self.position=pos
        self.health=h
        self.damage=d
        self.experience=e



class Hero(object):
    def __init__(self,*name):
        if name:self.name=name[0]
        else:self.name='Hero'
        self.position='00'
        self.health=100
        self.damage=5
        self.experience=0

class Hero(object):
    name = 'Hero'
    def __init__(self, name = 'Hero'):
        self.name = name
        self.experience = 0
        self.health = 100
        self.position = '00'
        self.damage = 5
class Hero(object):
    def __init__(self, name=None):
        self.name = 'Hero' if not name else name
        self.position = '00'
        self.health = 100
        self.damage = 5
        self.experience = 0
class Hero(object):
    def __init__(self, name=None, experience=None, health= None, position= None, damage= None):
        self.name = name or 'Hero'
        self.experience = 0
        self.health = 100
        self.position = '00'
        self.damage = 5
class Hero(object):
    experience = 0
    health = 100
    position = '00'
    damage = 5
    
    def __init__(self, name = "Hero"):
        self.name = name;
    

class Hero(object):
    def __init__(self, name='Hero'):
        self.name = name
        self.experience, self.health, self.position, self.damage = 0, 100, '00', 5
class Hero(object):
    def __init__(self, name=None):
        #Add default values here
        if name is None:
            self.name = 'Hero'
        else:
            self.name = name
        self.position = '00'
        self.health = 100
        self.damage = 5
        self.experience = 0
class Hero(object):
    def __init__(self, name='Hero', position='00', health=100, damage=5, experience=0):
        #Add default values here
        self.position = position
        self.health = health
        self.damage = damage
        self.experience = experience
        self.name = name
        
        


class Hero(object):
    def __init__(self, name='Hero', position='00', damage=5, health=100, experience=0):
        #Add default values here
        self.name = name
        self.position = position
        self.damage = damage
        self.health = health
        self.experience = experience

class Hero:
    def __init__(self, name="Hero", position='00', health=100, damage=5, experience=0):
        # Add default values here
        self.name = name
        self.position = position
        self.health = health
        self.damage = damage
        self.experience = experience
class Hero(object):
    def __init__(self, name="Hero"):
        self.name = name
        self.experience = 0
        self.health = 100
        self.position = '00'
        self.damage = 5

myHero = Hero()
class Hero(object):
    def __init__(self, name=None):
        self.name = name if name is not None else "Hero"
        self.experience = 0
        self.health = 100
        self.position = "00"
        self.damage = 5

class Hero(object):
    def __init__(self, name = 'Hero'):
        #Add default values here
        Hero.name = name;
        Hero.position = '00';
        Hero.health = 100;
        Hero.damage = 5;
        Hero.experience = 0;
        pass
class Hero(object):
    def __init__(self, name='Hero'):
        #Add default values here
        self.position = '00'
        self.health = 100
        self.damage = 5
        self.experience = 0
        self.name = name
        pass
class Hero(object):
    def __init__(self, *name):
        self.name = name[0] if name else 'Hero'
        self.position = '00'
        self.health = 100
        self.damage = 5
        self.experience = 0
class Hero(object):
    def __init__(self, nam = "Hero"):
        self.name = nam
        self.position = '00'
        self.health = 100
        self.damage = 5
        self.experience = 0
        pass
class Hero(object):
    def __init__(self, name = 'Hero'):
        self.name = name
    name = 'Hero'
    position = '00'
    health = 100
    damage = 5
    experience = 0
class Hero(object):
    def __init__(*args):
        self = args[0]
        if(len(args) == 2):
            self.name = args[1]
        else:
            self.name = "Hero"
        self.position = '00'
        self.health = 100
        self.damage = 5
        self.experience = 0
myHero = Hero()

class Hero(object):
    def __init__(self, name='Hero'):
        self.name = name #or 'Hero'
        self.position = '00'
        self.experience = 0
        self.damage = 5
        self.health = 100
        pass

class Hero(object):
    def __init__(self, name='Hero', position='00', health=100, damage=5, experience=0):
        self.name = name
        self.experience = experience
        self.health = health
        self.position = position
        self.damage = damage
class Hero(object):
    def __init__(self, name=None):
        self.name = name
        if self.name==None:
            self.name = 'Hero'
        self.position = '00'
        self.health = 100
        self.damage = 5
        self.experience = 0
        pass
class Hero(object):
    name = "Hero"
    position = "00"
    health = 100
    damage = 5
    experience = 0
    def __init__(self,name="Hero"):
        #Add default values here
        self.name = name
        self.position = "00"
        self.health = 100
        self.dmage = 5
        self.experience=0

# http://www.codewars.com/kata/55e8aba23d399a59500000ce/


class Hero(object):
    def __init__(self, name="Hero"):
        self.name = name
        self.position = "00"
        self.health = 100
        self.damage = 5
        self.experience = 0
class Hero(object):
    def __init__(self,name="Hero"):
        self.name = name
        
    name = "Hero"
    experience = 0
    health = 100
    position = "00"
    damage = 5

class Hero(object):
    def __init__(self, n = 'Hero'):
        self.name = n
        self.experience = 0
        self.health = 100
        self.position = '00'
        self.damage = 5
class Hero(object):
    position = "00"
    health = 100
    damage = 5
    experience = 0
    def __init__(self,name="Hero"):
        self.name = name
class Hero:
    def __init__(self, name = 'Hero', position = '00', health = 100, damage = 5, experience = 0):
        self.position = position
        self.health = health
        self.damage = damage
        self.experience = experience
        self.name = name
class Hero(object):
    position = '00'
    health = 100
    damage = 5
    experience = 0
    def __init__(self, name=None):
        self.name = name if name else 'Hero'
    

class Hero(object):
    name = 'Hero'
    position = '00'
    health = 100
    damage = 5
    experience = 0
    
    def __init__(self, name='Hero'):
        self.name = name
    pass    

class Hero:
    def __init__(self, name = 'Hero'):
        #Default values here
        self.name = name
        self.position = '00'
        self.health = 100
        self.damage = 5
        self.experience = 0


class Hero(object):
    def __init__(self, name="Hero"):
        self.experience = 0
        self.health = 100
        self.position = '00'
        self.damage = 5
        self.name = name
class Hero():
    def __init__(self, name='Hero'):
        self.name = name
        self.position = '00'
        self.health = 100
        self.damage = 5
        self.experience = 0
class Hero(object):
    def __init__(self, name = None):
        if name == None:
            self.name = 'Hero'
        else:
            self.name = name
        self.position = '00'
        self.health = 100
        self.damage = 5
        self.experience = 0
