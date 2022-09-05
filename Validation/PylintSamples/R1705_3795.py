def combat(health, damage):
    return max(0, health-damage)
def combat(health, damage):
    return max(health - damage, 0)
def combat(health, damage):
    return health - damage if health > damage else 0
def combat(health, damage):
    if damage > health:
        return 0
    return health - damage
def combat(h,d):
    return max(0,h-d)
def combat(health, damage):
    v=health-damage
    if v < 0:
        return 0
    else:
        return v
combat = lambda h,d: (h>d)*(h-d)
def combat(health, damage):
    return (damage<health)*(health-damage)
def combat(health, damage):
    if health - damage <= 0:
        return 0
    if health - damage > 0:
        return health - damage
def combat(health, damage):
    return 0 if (health - damage) < 0 else health - damage
def combat(health, damage):
    return [0, health-damage][damage < health]
combat = lambda health, damage: health - damage if health - damage > 0 else 0
def combat(health, damage):
    pv = health - damage
    return pv if pv > 0 else 0
def combat(health: int, damage: int) -> int:
    """ Get the player's new health bases on the received damage. """
    return health - damage if health - damage >= 0 else 0
def combat(health, damage):
    return health-min(health, damage)
def combat(health, damage):
    owca = health - damage
    if owca < 0:
        return 0
    return owca
combat = lambda h, d: max(h-d, 0)
def combat(health, damage):
    comb = health - damage
    if comb > 0:
        return comb
    else:
        return 0
def combat(health, damage):
    #your code here
    if damage > health:
        return 0
    else:
        return abs(health - damage)
def combat(health, damage):
    if health > damage:
        return health - damage
    if health < damage:
        return 0
def combat(health, damage):
    return {health - damage > 0: health - damage}.get(True, 0)
def combat(health, damage):
    return damage < health and health - damage
def combat(health, damage):
    after_hit = health - damage
    
    if after_hit >= 0:
        return after_hit
    else:
        return 0
def combat(health, damage):
    return health - damage > 0 and health - damage
def combat(health, damage):
    result = health - damage
    if result < 0:
        result = 0
    return result 
def combat(health, damage):
    x = health - damage
    return x if x > 0 else 0
def combat(health, damage):
    a = health
    b = damage
    c = a-b
    if c >= 0 :
        return c
    else :
        return 0
def combat(h, d):
    if h < d:
        return 0
    else:
        return h - d
combat = lambda h, d: h-d if h>d else 0
def combat(health, damage):
    sum = health - damage
    if sum > 0:
        return sum
    else:
        return 0
def combat(health, damage):
    #your code here
    while (health-damage)>=0:
        return health-damage
    else:
        return 0
def combat(health, damage):
    if health-damage < 1: 
        return 0
    else: 
        return health-damage
def combat(health, damage):
    #your code here
    
    if (health - damage) < 0:
        new_health = 0
        return new_health
    
    else:
        new_health = health - damage
        return new_health
        

def combat(health, damage):
    new = health -damage
    return new if new >=0 else 0 
def combat(health, damage):
    hitpoints = health - damage
    if hitpoints < 0:
        return 0
    else:
        return hitpoints
def combat(h, d):
    #your code here
    return (h-d) if h>=d else 0
def combat(health, damage):
    new = health - damage
    if new > 0:
        return new
    else: 
        return 0

def combat(h, dmg):
    return h - dmg if h > dmg else 0
def combat(health, damage):
    '''
    Create a combat function that takes the player's current health
    and the amount of damage recieved,
    and returns the player's new health.
    Health can't be less than 0.
    '''
    rest = health - damage
    if rest < 0:
        return 0
    return rest
def combat(health, damage):
    res = health - damage
    return res if res > 0 else 0
def combat(health, damage):
    if health < damage:
        return 0
    else:
        return abs(health-damage)
def combat(health, damage):
    if health < 0:
        return 0
    if health < damage:
        return 0
    return health - damage
def combat(health, damage):
    """ calculate remaining health """
    return (health > damage)*(health - damage)
def combat(hp, dp):
    return hp - dp if hp - dp > 0 else 0
def combat(health, damage):
    sub = health - damage
    return sub if sub > 0 else 0
def combat(health, damage):
    #yay I got the one line code right first try ;)
    return 0 if health <= damage else health - damage
def combat(health, damage):
    newhealth = health - damage
    return 0 if newhealth < 0 else newhealth
import unittest


def combat(health, damage):
    if health <= damage:
        return 0
    return health - damage


class TestCombat(unittest.TestCase):
    def test_should_return_0_when_given_damage_is_greater_than_health(self):
        self.assertEqual(combat(health=20, damage=30), 0)

    def test_combat_with_given_health_is_greater_than_damage(self):
        self.assertEqual(combat(health=100, damage=5), 95)

def combat(health, damage):
    list1 = health - damage
    if list1 > 0:
        return list1
    if list1 <= 0:
        return 0
def combat(health, damage):
    ch=health-damage
    if ch>=0:
        return ch
    else:
        return 0
def combat(health, damage):
    
    result=health-damage
    
    result_korr=max(result,0)
    
    return result_korr
def combat(health, damage):
    difference = health - damage
    if difference < 0:
        return 0
    else:
        return difference
def combat(health, damage):
    if damage > health:
        return 0
    elif health > damage:
        return health - damage

def combat(h, d):
    #your code here
    if h - d <= 0:
        return 0
    return h - d
def combat(health, damage):
    r = health - damage
    return r if r > 0 else 0
def combat(health, damage):
    return bool(health>damage)*(health-damage)
def combat(health, damage):
    total = health - damage
    return total if total>0 else 0 
def combat(health, damage):
    r = health - damage
    if r < 0:
        return 0
    return r
def combat(health, damage):
    
    h = health
    d = damage
    x = h - d
    
    if x < 0:
        return 0
    else:
        return x
def combat(health, damage):
    out = health - damage
    if out >= 0:
        return out
    else:
        return 0
combat=lambda h, d: h-d if d<h else 0
def combat(health, damage):
    new_h=(health-damage)
    if new_h<0:
        return 0
    else:
        return new_h
def combat(health, damage):
    if health > damage:
        v = health - damage
        return (v)
    else:
        return (0)
def combat(health, damage):
    new_health = health - damage
    return max(new_health,0)
combat = lambda h,d:h-d if h >= d else 0
def combat(health, damage):
    k = health - damage
    if k<0:
        return 0
    return k
def combat(health, damage):
    res = health - damage
    
    if res > 0:
        return res
    else:
        return 0
def combat(health, damage):
  after = health - damage;
  if after < 0:
    return 0;
  return after;
def combat(h, d):
    if h-d>=0:
        return h-d
    else:
        return 0#your code here
def combat(x,y): return x-y if x>y else 0
def combat(health, damage):
    if damage < health:
        x=health-damage
    else:
        x=0
    return x
def combat(health, damage):
    #your code here
    f = 0
    if health > damage:
        f = health - damage
    return f
def combat(health, damage):
    #your code here
    all = health - damage
    
    if all > 0:
        return all 
    else: 
        return 0
def combat(health, damage):
    ret=health-damage
    if 0>ret:
        ret=0
    return ret
def combat(h, d):
    if h>=d:
        return h-d
    return 0
def combat(hp, dp):
    return hp-dp if hp-dp>=0 else 0
def combat(health, damage):
    #your code here
    
    result = health - damage
    
    if result < int(0):
        return result*0
    else:
        return result
def combat(health, damage):
    hit = health - damage
    if hit <= 0:
        return 0
    else:
        return hit
def combat(health, damage):
    f = (health - damage)
    return f if f > 0 else 0
def combat(health, damage):
    #your code here
    b = 0
    b = health - damage
    if b < 0:
        b = 0
    return b
def combat(health, damage):
    health -= damage
    if health <= 0:
        return 0
    else:
        return health

def combat(health, damage):
    healthnew= health- damage
    if healthnew<0:
      healthnew= 0
    return healthnew
def combat(health, damage):
    #your code here
    
    try:
        if health<damage:
           return 0
        else:
           return (health-damage)   
    except:
        if health<0:
            return 'health needs to be greater than zero'
def combat(health, damage):
           a=(health-damage)
           if a  <=0:
               return 0
           else:
               return a

def combat(h, d):
    return max([0]+[h-d])
def combat(health, damage):
    return int(health-damage) if health-damage>0 else 0
def combat(health, damage):
    return [(health - damage), 0][(health - damage < 1)]
def combat(health: int, damage: int) -> int:
    return health - damage if health > damage else 0
def combat(health, damage):
    case = health - damage
    if case < 0:
        case = 0
    return case

def combat(health, damage):
    if health > damage:
        return health - damage
    else:
        health < damage
        return 0
def combat(health, damage):
    while health - damage > 0:
        return health - damage
    else:
        return 0
def combat(health, damage):
    NewHealth = health - damage
    if NewHealth < 0:
        return 0
    else:
        return NewHealth
def combat(health, damage):

    if health - damage <0:
       return 0
    elif health - damage > 0:
        return health - damage
    else:
        return int(health)
combat = lambda health, damage: 0 if health - damage < 0 else health - damage
def combat(health, damage):
    t = health - damage
    return t if t >=0 else 0
def combat(health, damage):
    if(health > damage):
      health = health - damage
      return health
    else:
        return 0
def combat(health: int, damage: int) -> int:
    return max(0, health - damage)
def combat(a, b):
    if a - b <= 0:
        return 0
    else:
        return a - b
def combat(health, damage):
    print(health, damage) 
    if damage > health: 
        return 0 
    else: 
        health -= damage 
        return health 
def combat(heal, dam):
    return max(0, heal - dam)
