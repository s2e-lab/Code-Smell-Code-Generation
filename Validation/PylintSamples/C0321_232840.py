def contents_size(self):
        '''
        Returns the number of different categories to be shown in the
        contents side-bar in the HTML documentation.
        '''
        count = 0
        if hasattr(self,'variables'): count += 1
        if hasattr(self,'types'): count += 1
        if hasattr(self,'modules'): count += 1
        if hasattr(self,'submodules'): count += 1
        if hasattr(self,'subroutines'): count += 1
        if hasattr(self,'modprocedures'): count += 1
        if hasattr(self,'functions'): count += 1
        if hasattr(self,'interfaces'): count += 1
        if hasattr(self,'absinterfaces'): count += 1
        if hasattr(self,'programs'): count += 1
        if hasattr(self,'boundprocs'): count += 1
        if hasattr(self,'finalprocs'): count += 1
        if hasattr(self,'enums'): count += 1
        if hasattr(self,'procedure'): count += 1
        if hasattr(self,'constructor'): count += 1
        if hasattr(self,'modfunctions'): count += 1
        if hasattr(self,'modsubroutines'): count += 1
        if hasattr(self,'modprocs'): count += 1
        if getattr(self,'src',None): count += 1
        return count