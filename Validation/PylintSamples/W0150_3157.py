def number(bus_stops):
    return sum([stop[0] - stop[1] for stop in bus_stops])
def number(stops):
    return sum(i - o for i, o in stops)
def number(bus_stops):
    return sum(on - off for on, off in bus_stops)
def number(bus_stops):
    sum=0
    for i in bus_stops:
        sum+=i[0]-i[1]
    return sum
def number(bus_stops):
    return sum(stop[0]-stop[1] for stop in bus_stops)
def number(bus_stops):
    total = 0
    for entered, out in bus_stops:
        total += entered - out 
    return total

def number(arr):
  people_in = 0
  people_out = 0
  for stops in arr:
    people_in += stops[0]
    people_out += stops[1]
  return people_in - people_out
def number(bus_stops):
    get_in, get_off = zip(*bus_stops)
    return sum(get_in) - sum(get_off)
def number(bus_stops):
    return sum(i-o for i,o in bus_stops)
def number(bus_stops):
    return sum([item[0] - item[1] for item in bus_stops]) 

def number(bus_stops):
    x = 0
    for i, j in bus_stops:
        x += i - j
    return x
def number(bus_stops):
    return sum(n - m for n, m in bus_stops)
def number(bus_stops):
    return sum(map(lambda x: x[0]-x[1],bus_stops))
number = lambda b: sum(i-o for i,o in b)
def number(bus_stops):
    total_number = 0
    for i in bus_stops:
        total_number += i[0] - i[1]
        
    return total_number
    

def number(bus_stops):
    return sum([i-j for [i,j] in bus_stops])
def number(bus_stops):
    #bus_stops[0][0]
    #bus_stops[1][0]
    #bus_stops[2][0]
    
    #bus_stops[0][1]
    #bus_stops[1][1]
    #bus_stops[2][1]
    came = 0
    left = 0
    for counter, _ in enumerate(bus_stops,0):
        came = came + bus_stops[counter][0]
        left = left + bus_stops[counter][1]
    return came - left
def number(bus_stops):
    t = 0
    for a in bus_stops:
        t = t + a[0] - a[1]
    return t
def number(bus_stops):
    return sum([bus_in - bus_out for bus_in, bus_out in bus_stops])
def number(bus_stops):
    return sum( a-b for [a,b] in bus_stops)
def number(bus_stops):
    return sum(x - y for x,y in bus_stops)
def number(bus_stops):
    # Good Luck!
    totalPeople = 0
    for i in bus_stops:
        totalPeople += i[1] - i[0]
    return abs(totalPeople)
    

def number(bus_stops):
    getin = 0
    getout = 0
    for x, y in bus_stops:
        getin += x
        getout += y
    return getin - getout

from functools import reduce

def number(bus_stops):
    return sum(x[0]-x[1] for x in bus_stops)
def number(x):
    L=[]
    for i in x:
        L.append(i[0])
        L.append(-i[1])
    return sum(L)
def number(bus_stops):
    return sum(guys_in - guys_out for guys_in, guys_out in bus_stops)
def number(bus_stops):
    number_of_people = 0
    for i in bus_stops:
        number_of_people += i[0]
        number_of_people -= i[1]
    return number_of_people    
def number(bus_stops):
    
    counter = 0
    max_index = len(bus_stops) - 1
    y = 0
    z = 0
    
    while counter <= max_index:
        x = bus_stops[counter]
        counter = counter + 1
        y = y + x[0]
        z = z + x[1]
    end = y - z
    return end
def number(bus_stops):
    return sum([p[0]-p[1] for p in bus_stops])
import operator

def number(bus_stops):
    return operator.sub(*map(sum, zip(*bus_stops)))
def number(bus_stops):
    on, off = list(map(sum, list(zip(*bus_stops))))
    return on - off

def number(bs):
    on=0
    off=0
    for i in bs:
        on+=i[0]
        off+=i[1]
    return on-off
def number(bus_stops):
    x =0
    for (go,out) in bus_stops:
        x+=go-out 
    return x
def number(bus_stops):
    still_in_bus = 0
    for stop in bus_stops:
        still_in_bus = still_in_bus + (stop[0] - stop[1])
    return still_in_bus
def number(bus_stops):
    a = []
    for l in  bus_stops:
        c = l[0] - l[1]
        a.append(c)
    return sum(a)
        
            

def number(bus_stops):
    numOfPassengers = 0
    index = 0
    for x in range(0,len(bus_stops)):
        numOfPassengers += bus_stops[index][0]
        numOfPassengers -= bus_stops[index][1]
        index += 1
    return numOfPassengers
    

number = lambda b: sum(e[0] - e[1] for e in b)
number = lambda b: sum(e[0] for e in b) - sum(e[1] for e in b)
def number(bus_stops):
   passengers = 0
   for (x,y) in bus_stops:
       passengers = passengers + x - y
   return passengers
from functools import reduce

def number(bus_stops):
    return -reduce(
        lambda x, y: y - x,
        reduce(
            lambda a, b: a + b,
            bus_stops
        )
    )

def number(bus_stops):
    rtn = 0
    for i in bus_stops:
        rtn = rtn + i[0] - i[1]
    
    return rtn
def number(bus_stops):
    return sum(x[0] for x in bus_stops) - sum(x[1] for x in bus_stops) # acquired - lost
from itertools import starmap
from operator import sub

def number(bus_stops):
    return sum(starmap(sub,bus_stops))
def number(bus_stops):
    return sum(bus_stops[0] for bus_stops in bus_stops) - sum(bus_stops[1] for bus_stops in bus_stops)
def number(bus_stops):
    return sum(x[0]-x[1] for x in bus_stops)
def number(bus_stops):
    return sum([ a-b for a, b in bus_stops ])
def number(bus_stops):
    result = 0
    stop = list(bus_stops)
    for x in stop:
        result = result + x[0] - x[1]
    return result
def number(bus_stops):
    sleeping = 0
    for i in bus_stops:
        sleeping += i[0] - i[1]
    return sleeping

import numpy as np
def number(bus_stops):
    bs = np.array(bus_stops)
    return sum(bs[:, 0] - bs[:, 1])
def number(bus_stops):
    #initialize variables
    x =0
    #iterate through the arrays and add to the count
    for people in bus_stops:
        x += people[0]
        x -= people[1]
    return x
def number(bus_stops):
    ppl = 0
    for stops in bus_stops:
        ppl += stops[0]
        ppl -= stops[1]
    return ppl
def number(bus_stops):
    retval=0
    for x in bus_stops:
        retval=retval+x[0]
        retval=retval-x[1]
        print((x[1]))
    return retval
    # Good Luck!

def number(bus_stop):
    get_bus = sum([a[0]-a[1] for a in bus_stop])
    if get_bus > 0:
        return get_bus
    else:
        return 0


number([[10,0],[3,5],[5,8]])
number([[3,0],[9,1],[4,10],[12,2],[6,1],[7,10]])
number([[3,0],[9,1],[4,8],[12,2],[6,1],[7,8]])

# Made with u2764ufe0f in Python 3 by Alvison Hunter - October 8th, 2020
def number(bus_stops):
    try:
        last_passengers = sum(people_in - people_out for people_in, people_out in bus_stops)
    except:
        print("Uh oh! Something went really wrong!")
        quit
    finally:
        if(last_passengers >= 0):
            print('Remaining Passengers for last stop: ', last_passengers)
            return last_passengers
        else:
            print('No passengers where on the bus')
            last_passengers = 0
            return last_passengers
# Made with u2764ufe0f in Python 3 by Alvison Hunter - October 8th, 2020
def number(bus_stops):
    try:
        last_passengers = 0
        total_people_in = 0
        total_people_out = 0

        for first_item, second_item in bus_stops:
            total_people_in +=  first_item
            total_people_out += second_item

    except:
        print("Uh oh! Something went really wrong!")
        quit
    finally:
        last_passengers = total_people_in - total_people_out
        if(last_passengers >= 0):
            print('Remaining Passengers for last stop: ', last_passengers)
            return last_passengers
        else:
            print('No passengers where on the bus')
            last_passengers = 0
            return last_passengers
from functools import reduce

def number(bus_stops):
    return reduce(lambda i,a: i+a[0]-a[1], bus_stops, 0)

def number(bus_stops):
    
    bus_pop = 0
    
    for stop in bus_stops:
        bus_pop += stop[0]
        bus_pop -= stop[1]
        
    return bus_pop
def number(bus_stops):
    present = 0
    for tup in bus_stops:
        present = present + tup[0]
        present = present - tup[1]
    return present
        

def number(bus_stops):
    first_tuple_elements = []
    second_tuple_elements = []
    result = []
    for i in bus_stops:
        first_tuple_elements.append(i[0])
        second_tuple_elements.append(i[1])
    x = 0 
    result = 0
    for i in first_tuple_elements:
        result += first_tuple_elements[x] - second_tuple_elements[x]
        print(result)
        x += 1
    return result

def number(bus_stops):
#     a = sum([i[0] for i in bus_stops])
#     b = sum([i[1] for i in bus_stops])
#     return a-b
    return sum([i[0] for i in bus_stops])-sum([i[1] for i in bus_stops])
def number(bus_stops):
    people_in_the_bus = 0
    
    for people in bus_stops:
        people_in_the_bus += people[0]
        people_in_the_bus -= people[1]
        
    return people_in_the_bus
def number(bus_stops):
    x = 0
    y = 0
    for i in bus_stops:
        x += i[0]
        y += i[1]
    return x - y 

def number(bus_stops):
    solut = 0
    for x in bus_stops:
        solut += x[0]
        solut -= x[1]
    return solut
def number(bus_stops):
    lastPeople = 0
    i = 0
    while(i< len(bus_stops)): 
        lastPeople = lastPeople + bus_stops[i][0] - bus_stops[i][1]
        i = i+1
    return lastPeople;
def number(bus):
    c = 0
    for j in bus:
        c += j[0]-j[1]
    return c

def number(bus_stops):
    inBus = 0
    for stop in bus_stops:
        inBus += stop[0]
        inBus -= stop[1]
        if inBus < 0:
            return False
    return inBus
def number(bus_stops):
    bus_occupancy = 0
    i = 0
    j = 0
    for i in range(len(bus_stops)):
        bus_occupancy = bus_occupancy + (bus_stops[i][j] - bus_stops[i][j+1])
        print('Bus occupancy is: ', bus_occupancy)
    return(bus_occupancy)    
def number(bus_stops):
    on_sum = 0
    off_sum = 0
    for x in bus_stops:
        on_sum += x[0]
        off_sum += x[1]
    return on_sum - off_sum    
    # Good Luck!

def number(bus_stops):
    x = 0
    for i in bus_stops:
        x = i[0] - i[1] + x
    return x
def number(bus_stops):
    res = 0 
    for e in bus_stops:
        res += e[0]
        res -= e[1]    
    return res
def number(bus_stops):
    still_on = 0
    for stop in bus_stops:
        still_on += stop[0]
        still_on -= stop[1]
    return still_on
def number(bus_stops):
    peopleInBus=0
    for elem in bus_stops:
        peopleInBus+=elem[0]-elem[1]
    return peopleInBus
def number(bus_stops):
    # Good Luck!
    return sum((peoplein - peopleout) for peoplein,peopleout in (bus_stops))
def number(bus_stops):
    enter, leave = zip(*bus_stops)
    return sum(enter) - sum(leave)
def number(bus_stops):
    empty = []
    for i in range(0,len(bus_stops)):
        empty.append(bus_stops[i][0]-bus_stops[i][1])
    return sum(empty)


def number(bus_stops):
    total = 0
    for stop in bus_stops:
        print(stop)
        total = total + stop[0]
        total = total - stop[1]
    
    return total

def number(bus_stops):
    b = 0
    for enter, leave in bus_stops:
        b = max(b + enter - leave, 0)
    return b

def number(bus_stops):
    totalnum1 =0
    totalnum2 =0
    for i in range (0,len(bus_stops)):
        totalnum1 = totalnum1 + (bus_stops[i])[0]
        totalnum2 = totalnum2 + (bus_stops[i])[1]
        num = totalnum1 - totalnum2
    return num
def number(bus_stops):
    sum_up = 0
    sum_down = 0
    for i in bus_stops:
        sum_up = sum_up + i[0]
        sum_down = sum_down + i[1]
    summ = sum_up-sum_down
    return summ
def number(bus_stops):
    summ=0
    for i in bus_stops:
        summ+= i[0]-i[1]
    return summ
    # Good Luck!

def number(bus_stops):
    ans = 0
    for o in bus_stops:
        ans += o[0]
        ans -= o[1]
    return ans
def number(bus_stops):
    res = 0
    for st in bus_stops:
        res = res - st[1]
        res = res + st[0]
    return res    
    # Good Luck!

def number(bus_stops):
    peoples = 0
    for number in bus_stops:
        peoples += number[0]-number[1]
    return peoples
def number(bus_stops):
    remaining = 0
    for inP, outP in bus_stops:
        remaining = remaining + inP - outP
    return remaining
def number(bus_stops):
    # Good Luck!
    into_bus = 0
    get_off_bus = 0
    for stops in bus_stops:
        into_bus += stops[0]
        get_off_bus += stops[1]
    return into_bus - get_off_bus
def number(bus_stops):
    people = [sum(i) for i in zip(*bus_stops)]
    return people[0] - people[1]
def number(bus_stops):
    sum=0
    for i in bus_stops:
        num=i[0]-i[1]
        sum=sum+num
    return sum
    
    # Good Luck!

def number(bus_stops):
    get_in = sum([a[0] for a in bus_stops])
    get_out = sum([a[1] for a in bus_stops])

    total = get_in - get_out
    return total
def number(bus_stops):
    # Good Luck!
    '''
    input: bus_stops tuple - the # of people who get on and off the bus
    approach: loop through and sum the first element and sum the last element then subtract the two numbers
    output: the # of people left on the bus after the last stop
    '''
    passengersOn = 0
    passengersOff = 0
    
    try:
        res = sum(i[0] for i in bus_stops)- sum(i[1] for i in bus_stops) 
            
    except: 
        print("There was an error")
        
    return res
def number(bus_stops):
    get_in = get_out = 0
    for k,v in bus_stops:
        get_in += k
        get_out += v
    return (get_out - get_in) * -1
def number(bus_stops):
    count = 0
    for el in bus_stops:
        count = count + el[0] - el[1]
    return count
    # Good Luck!

def number(bus_stops):
    sum_in_bus = 0
    
    for stop in bus_stops:
        sum_in_bus += stop[0] - stop[1]
        
    return sum_in_bus
def number(bus_stops):
    onBuss = 0

    for i in range(len(bus_stops)):
        onBuss += bus_stops[i][0]
        onBuss -= bus_stops[i][1]
    return onBuss

def number(bus_stops):
    number = 0
    for i in range(len(bus_stops)):
        number += bus_stops[i][0] - bus_stops[i][1]
    return number      
    # Good Luck!

def number(bus_stops):
    up=0
    down=0
    
    for stops in bus_stops:
        up+=stops[0]
        down+=stops[1]
    
    return up-down

def number(bus_stops):
    people=0
    for tuple in bus_stops:
        x,y = tuple
        people+=x-y
        
    return people

def number(bus_stops):
    # Good Luck!
    left_on_bus = 0
    for i in bus_stops:
        left_on_bus += (i[0]) - (i[1])
    return left_on_bus
        
        

def number(bus_stops):
    res = [i - j for i, j in bus_stops]
    return sum(res)
