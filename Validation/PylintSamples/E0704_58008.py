def getSourceFnDef(stack,fdefs,path):
    '''VERY VERY SLOW'''
    found = False
    for x in stack:
        if isinstance(x, ast.FunctionDef):
            for y in fdefs[path]:
                if ast.dump(x)==ast.dump(y): #probably causing the slowness
                    found = True
                    return y
            raise
    if not found:
        for y in fdefs[path]:
            if y.name=='body':
                return y
    raise