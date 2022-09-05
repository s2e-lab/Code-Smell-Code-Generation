def friend(x):
    return [f for f in x if len(f) == 4]
def friend(x):
    #Code
    names = []
    for name in x:
        if len(name) == 4:
            names.append(name)
    return names
def friend(x):
    return [i for i in x if len(i) == 4]
def friend(x):
    myFriends = []                   # Initialize list variable
    for person in x:                 # Loop through list of names 
        if len(person) == 4:         # Check to see if the name is of length 4
            myFriends.append(person) # If the name is 4 characters long, append it to variable myFriends
    return myFriends                 # Return myFriends list

def friend(x):
    return list(filter(lambda s : len(s)==4 ,x))
def friend(x):
    '''
    x: list of strings/people
    returns: list of people with only 4 letters in their names
    '''
    return [n for n in x if len(n) == 4]
def friend(x):
    return [each for each in x if len(each) == 4]
def friend(x):
    return [y for y in x if len(y)==4]
def friend(arr):
    return [x for x in arr if len(x)==4]
def friend(x):
    return [n for n in x if len(n)==4]
def friend(x):
    return list(filter(lambda name: len(name) == 4, x))
def friend(x):
   
#set an empty list to fill the output with
    result=[]
   
#make a forloop to go through the list and add the names that have 4 letter to result
    for i in x:
        if len(i) == 4:
            result.append(i)

    
    return(result)
    

import re

def friend(x):
    true_friends = []
    for name in x:
        # checking the name's length is not enough
        # (even tho all tests are green)
        # name should contain only letters from the alphabet
        if len(name) == 4 and re.search('[a-zA-Z]', name):
            true_friends.append(name)
    return true_friends
def friend(x):
    myFriend=[] #new list to store my friends names
    for i in x:
        if len(i)== 4:
            myFriend.append(i)
        else:
            pass
    return myFriend
    #Code

def friend(x):
    new_friends=[]
    for y in x:
        if len(y)==4:
            new_friends.append(y)
    return new_friends        
    #Code

def friend(x):
    res = []
    
    for i, j in enumerate(x):
        if len(j) == 4:
            res.append(j)
            
    return res
from typing import List


def friend(x: List[str]=["Ryan", "Kieran", "Mark",]):
    # Code
    friends = []

    for name in x:
        if len(name) == 4:
            friends.append(name)

    return friends
def friend(x):
    shouldBe = list()
    for f in x:
        if len(f) == 4:
            shouldBe.append(f)
    return shouldBe
def friend(friends):
    my_list = []
    return [f for f in friends if len(f) == 4]
# def friend(x):
#     friends = []
#     for friend in x:
#         if len(friend) == 4:
#             friends.append(friend)
#         return friends


def friend(x):
    return [friend for friend in x if len(friend) == 4]
def friend(lst):
    return [name for name in lst if len(name) == 4]
friend = lambda x: list(filter(lambda y: len(y)==4, x))
def friend(friends):
    friend = []
    for name in friends:
        if(len(name) == 4):
            friend.append(name)
    return friend
    

def friend(x):
    my_friends = []
    for i in range(len(x)):
        if(len(x[i]) == 4):
            my_friends.append(x[i])
    return my_friends

def friend(input):
    return list(filter(lambda x:len(x) == 4, input))
def friend(x):
    list = []
    for i in x:
        if len(i) == 4:
            list.append(i)
        else:
            pass
    return list
    #Code

def friend(people):
    return [person for person in people if len(person)==4]
def friend(names):
    return [x for x in names if len(x) == 4]
def friend(x):
    return list(filter(lambda a: len(a) == 4, x))
def friend(friends):
    return [friend for friend in friends if len(friend) == 4]
def friend(x):
    y = []
    for i in x:
        if len(i) == 4:
            y.append(i)
        else:
            pass 
    return y
def friend(x):
    friends = []
    for name in x:
        if len(name) == 4:
            friends.append(name)
            
    return friends
            

def friend(x):
    return [word for word in x if len(word) == 4]
def friend(names):
    return [n for n in names if len(n) == 4]
def friend(x):
    z=0
    friends = []
    while z < len(x):
        if len(x[z]) == 4: friends.append(x[z])
        z+=1
    return friends
def friend(arr):
    raturn = []
    for x in arr:
        if len(x) == 4:
            raturn.append(x)
        else:
            pass
    return raturn

def friend(x):
   friendly = lambda y: len(y)==4
   return list(filter(friendly,x))
def friend(x):
    return [x for x in x if x == x[:4] and x[3:]]
def friend(x):
    return list(filter(lambda name: len(name) is 4, x)) #convert to list for python 3
def friend(x):
    return list(filter(lambda x: len(x) == 4, x))
def friend(x):
    return list(filter(lambda i: len(i)==4, x))
def friend(x):
    friendList = []
    for index, name in enumerate(x):
        if len(name) == 4:
            friendList.append(name)
        else:
            pass
    return friendList
            
    #Code

from typing import List

def friend(array: List[str]) -> List[str]:
    """
    Filter a list of strings and return a list with only your friends name in it.
    Rule: If a name has exactly 4 letters in it, you can be sure that it has to be a friend of yours!
    """
    return list(filter(lambda name: len(name) == 4, array))
def friend(x):
    return [x for x in x if x is x and all([not not x, x[0:4][::-1] == x[-1:-5:-1], len(x) == 65536 ** 0.125])]
friend = lambda x: [s for s in x if len(s) == 4]
def friend(x):
    return [name for name in x if len(name) == 4]
def friend(x):
    enemy = []
    [enemy.append(y) for y in x if len(y) != 4]
    [x.remove(z) for z in enemy]
    return x
def friend(x):
   friends = []
   for i in x:
       count = len(i)
       if count == 4 and not i.isnumeric():
          friends.append(i)
       else:
          pass

   return friends
def friend(people):
    flag = False
    while flag != True:
        flag = True
        for x in people:
            if len(x) != 4:
                people.remove(x)
                flag = False
    return people
import re
def friend(x):
    result = []
    for n in x:
        if re.match(r'A....Z',n):
            result.append(n)
    return result
def friend(x):
    friends = []
    for i in range(0, len(x)):
        currentFriend = x[i]
        if len(currentFriend) == 4:
            friends.append(currentFriend)
    return friends
def friend(x):
    nl = []
    for f in range(len(x)):
        if len(x[f]) == 4:
            nl.append(x[f])
    return nl
def friend(x):
    #Code
    resultList = []
    for friend in x:
        if len(friend)==4 : resultList.append(friend)
    
    return resultList
    

def friend(x):
    flist = []
    for name in x:
        if len(name) == 4:
            flist.append(name)
    return flist
    

def friend(x):
    #Code
    friend_l=list()
    for name in x:
        if len(name) == 4:
            friend_l.append(name)
    return friend_l
