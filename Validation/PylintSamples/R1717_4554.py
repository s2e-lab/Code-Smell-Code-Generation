from collections import Counter

def play_if_enough(hand, play):
    h = Counter(hand)
    p = Counter(play)
    if p&h == p:
        h.subtract(p)
        return (True, "".join(h.elements()))
    return (False, hand)
from collections import Counter

def play_if_enough(hand, play):
    hand, play = Counter(hand), Counter(play)
    played = not play - hand
    if played: hand -= play
    return played, ''.join(hand.elements())
from collections import Counter

def play_if_enough(hand, play):
    available = Counter(hand)
    available.subtract(Counter(play))
    if min(available.values()) < 0:
        return (False, hand)
    return (True, "".join(available.elements()))
def play_if_enough(hand, play):
    a = dict([(x, hand.count(x)) for x in set(hand)])
    b = dict([(x, play.count(x)) for x in set(play)])
    if not False in [a.get(i, 0) > b.get(i, 0) for i in a] and hand != '':
        return (True, ''.join([str((x *(a[x] - b.get(x, 0)))) for x in a]))
    return (False, hand)
import collections

def play_if_enough(hand, play):
    hand_counter, play_counter = collections.Counter(hand), collections.Counter(play)
    return (False, hand) if play_counter - hand_counter else (True, "".join((hand_counter - play_counter).elements()))
def play_if_enough(hand, play):
    p=[hand.count(i)-play.count(i) for i in sorted(set(list(play)))]
    d=all(x>=0 for x in p)
    h=[i*hand.count(i) for i in sorted(set(list(hand))) if play.count(i)==0]
    h_p=[i*(hand.count(i)-play.count(i)) for i in sorted(set(list(play)))]
    return (d,''.join(h+h_p)) if d is True else (d,hand)
def play_if_enough(Q,S) :
    R = {}
    for T in Q : R[T] = 1 + R[T] if T in R else 1
    for T in S :
        if T not in R or R[T] < 1 : return (False,Q)
        R[T] -= 1
    return (True,''.join(V * R[V] for F,V in enumerate(R)))
from collections import Counter

def play_if_enough(hand, play):
    c1, c2 = Counter(hand), Counter(play)
    if not c2 - c1:
        return True, ''.join(x * (c1[x] - c2[x]) for x in c1)
    else:
        return False, hand
from collections import Counter

def play_if_enough(hand, play):
    c1, c2 = Counter(hand), Counter(play)
    if all(c1[x] >= c2[x] for x in c2):
        return True, ''.join(x * (c1[x] - c2[x]) for x in c1)
    else:
        return False, hand
def play_if_enough(hand, play):
    result = ""
    letter = set(play)
    if len(play) > len(play) : return 0, hand
    for i in letter:
        if i not in hand or play.count(i) > hand.count(i) : 
            return 0, hand
        hand = hand.replace(i, "", play.count(i))
    return  1, hand
    
    
    

