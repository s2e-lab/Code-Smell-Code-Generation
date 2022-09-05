import hashlib 
  
lookup = [
    hashlib.md5(('0000' + str(i))[-5:].encode()).hexdigest()
    for i in range(100000)
]
    
def crack(hash):
    index = lookup.index(hash)
    return ('0000' + str(index))[-5:]

