def anagrams(word, words): return [item for item in words if sorted(item)==sorted(word)]
from collections import Counter

def anagrams(word, words):
    counts = Counter(word)
    return [w for w in words if Counter(w) == counts]
def anagrams(word, words):
    match = sorted(word)
    return [w for w in words if match == sorted(w)]
def anagrams(word, words):
    return [w for w in words if sorted(word)==sorted(w)]
from collections import Counter


def anagrams(word, words):
    n, c = len(word), Counter(word)
    return [w for w in words if len(w) == n and Counter(w) == c]

def anagrams(word, words):
    letter = { x : word.count(x) for x in word }
    result = []
    
    for i in words:
        letters = { x : i.count(x) for x in i }
        if letters == letter:
            result.append(i)
    
    return result
def anagrams(word, words):
    return [el for el in words if sorted(word) == sorted(el)]
from collections import Counter
def anagrams(word, words):
    main = Counter(word)
    return [wor for wor in words if Counter(wor) == main]
def anagrams(word, words):
    return [x for x in words if sorted(x) == sorted(word)]
def anagrams(word, words):
    return [w for w in words if list(sorted(w)) == list(sorted(word))]
def anagrams(word, words):
    def lettercount(inputword):
        wordarr = list(inputword)
        worddict = {}
        for letter in wordarr:
            if letter not in worddict:
                worddict[letter] = wordarr.count(letter)
        return worddict
    
    return [astring for astring in words if lettercount(astring) == lettercount(word)] 
        
    

from collections import Counter

def anagrams(word, words):
    return [w for w in words if Counter(word) == Counter(w)]
def anagrams(word, words):
    return [x for x in words if sorted(word) == sorted(x)]
def anagrams(word, words):
    lst = []
    for elem in words:
        if sorted(word) == sorted(elem):
            lst.append(elem)
    return lst
    #your code here

def anagrams(word, words):
    word=sorted(word)
    return list(filter(lambda ele: sorted(ele)==word  ,words))
def anagrams(word, words):
    return [trial for trial in words if sorted(trial) == sorted(word)]
def anagrams(word, words):
    return [w for w in words if sorted(w) == sorted(word)]
def anagrams(word: str, words: list) -> list: return list(filter(lambda x: sorted(x) == sorted(word), words))
anagrams = lambda _,__: list([s for s in __ if sorted(s) == sorted(_)])

anagrams=lambda word, words:list(w for w in words if sorted(list(w))==sorted(list(word)))
def anagrams(word, words):
    ans = []
    or1 = 0
    or2 = 0
    for i in word:
        or1 += ord(i)
    for i in words:
        or2 = 0
        for x in i :
            or2 += ord(x)
        if or1 == or2:
            ans += [i]
    return ans
# What is an anagram? Well, two words are anagrams of each other
# if they both contain the same letters. For example:
# 'abba' & 'baab' == true ; 'abba' & 'bbaa' == true 
# 'abba' & 'abbba' == false ; 'abba' & 'abca' == false
# Write a function that will find all the anagrams of a word from a list.
# You will be given two inputs a word and an array with words.
# You should return an array of all the anagrams or an empty array
# if there are none. For example:
# anagrams('abba', ['aabb', 'abcd', 'bbaa', 'dada']) => ['aabb', 'bbaa']
# anagrams('racer', ['crazer', 'carer', 'racar', 'caers', 'racer']) =>
# ['carer', 'racer']
# anagrams('laser', ['lazing', 'lazy',  'lacer']) => []

def anagrams(word, words):
    w_buff = []
    w_out = []
    for w_t in words :
        if len(w_t) == len(word):
           w_buff.append(w_t)
    w_w = list(word)
    w_w.sort()
    for w_t in w_buff:
        w_buff_l = list(w_t)
        w_buff_l.sort()
        if w_buff_l == w_w :
            w_out.append(w_t)
    return w_out
def anagrams(word, words):
    #your code here
    list = []
    word = sorted(word)
    for i in range(len(words)):
        if word == sorted(words[i]):
            list.append(words[i])
        else:
            pass
    return(list)
from collections import Counter
def anagrams(word, words):
    # your code here
    return [w for w in words if sorted(sorted(Counter(word).items())) == sorted(sorted(Counter(w).items()))]
def anagrams(word, words):

    l = [letter for letter in word]
    anagram_list = []
    
    for item in words:
        l_item = [letter for letter in item]
        if sorted(l) == sorted(l_item):
            temp_list = [i for i in l + l_item if i not in l_item]
            if len(temp_list) == 0:
                anagram_list.append(item)
        else:
            continue
    return anagram_list
def anagrams(word, words):
    return [anagram for anagram in words if sum([ord(c) for c in anagram]) == sum([ord(c) for c in word])]

def anagrams(word, words):
    #your code here
    wordnum=sum(ord(ch) for ch in word)
    res=[]
    if not words :
        return []
    for item in words:
        if sum(ord(ch) for ch in item)==wordnum:
            res.append(item)
    return res
