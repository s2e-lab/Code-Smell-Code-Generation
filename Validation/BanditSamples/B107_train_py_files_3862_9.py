def mirror(code, secret='abcdefghijklmnopqrstuvwxyz'):
    
    intab = secret
    outtab = secret[::-1]
    
    return code.lower().translate(str.maketrans(intab, outtab))

