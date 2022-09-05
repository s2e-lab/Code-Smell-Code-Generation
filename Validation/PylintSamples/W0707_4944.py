import math


class VectorInputCoordsValidationError(Exception):
    """Custom exception class for invalid input args given to the Vector instantiation"""


class Vector:
    # https://www.mathsisfun.com/algebra/vectors.html

    def __init__(self, *args):
        try:
            self.x, self.y, self.z = args if len(args) == 3 else args[0]
        except ValueError:
            raise VectorInputCoordsValidationError('Either give single iterable of 3 coords or pass them as *args')

    def __add__(self, other) -> "Vector":
        return Vector(
            self.x + other.x,
            self.y + other.y,
            self.z + other.z
        )

    def __sub__(self, other) -> "Vector":

        return Vector(
            self.x - other.x,
            self.y - other.y,
            self.z - other.z
        )

    def __eq__(self, other) -> bool:
        # https://www.grc.nasa.gov/www/k-12/airplane/vectcomp.html
        # https://onlinemschool.com/math/library/vector/equality/
        return all((
            self.x == other.x,
            self.y == other.y,
            self.z == other.z
        ))

    def cross(self, other) -> "Vector":
        # https://www.mathsisfun.com/algebra/vectors-cross-product.html
        return Vector(
            self.y*other.z - self.z*other.y,
            self.z*other.x - self.x*other.z,
            self.x*other.y - self.y*other.x
        )

    def dot(self, other) -> int:
        # https://www.mathsisfun.com/algebra/vectors-dot-product.html
        return self.x*other.x + self.y*other.y + self.z*other.z

    def to_tuple(self) -> tuple:
        return self.x, self.y, self.z

    def __str__(self) -> str:
        return "<{x}, {y}, {z}>".format(**self.__dict__)

    @property
    def magnitude(self) -> float:
        return math.sqrt(
            sum (
                    (
                        self.x ** 2,
                        self.y ** 2,
                        self.z ** 2
                    )
            )
        )

from operator import itemgetter, add, sub, mul
from itertools import starmap


class Vector(list):
    
    def __init__(self, *args):
        if len(args)==1: args = args[0]
        super().__init__(args)
    
    __add__, __sub__, __mul__ = (
        ( lambda self,o,f=fun: Vector(starmap(f, zip(self,o))) ) for fun in (add, sub, mul)
    )
    
    x,y,z = (property(itemgetter(i)) for i in range(3))
    
    @property
    def magnitude(self): return self.dot(self)**.5
    
    def __str__(self):   return f'<{ ", ".join(map(str,self)) }>'
    def to_tuple(self):  return tuple(self)
    def dot(self,o):     return sum(self*o)
    def cross(self,o):   return Vector( self.y*o.z - self.z*o.y,
                                        self.z*o.x - self.x*o.z,
                                        self.x*o.y - self.y*o.x)
import numpy as np
class Vector:
    def __init__(self, *args):
        args = args[0] if len(args) == 1 else args
        self.x, self.y, self.z = args[0], args[1], args[2]
        self.li = [self.x,self.y,self.z]
        self.magnitude = np.linalg.norm(self.li)
    def __add__(self, other)   :   return Vector([i+j for i,j in zip(self.li,other.li)])
    def __sub__(self, other)   :   return Vector([i-j for i,j in zip(self.li,other.li)])
    def __eq__(self, other)    :   return all([i==j for i,j in zip(self.li,other.li)])
    def __str__(self)          :   return f'<{self.x}, {self.y}, {self.z}>'
    def cross(self, other)     :   return Vector(np.cross(self.li, other.li))
    def dot(self, other)       :   return np.dot(self.li,other.li)
    def to_tuple(self)         :   return tuple(self.li)
class Vector:
    def __init__(self, a, b = None, c = None):
        if c is None:
            a, b, c = a
        self.x = a
        self.y = b
        self.z = c
        self.magnitude = ((self.x) ** 2 + (self.y) ** 2 + (self.z) ** 2) ** .5
    def to_tuple(self):
        return (self.x, self.y, self.z)
    def __eq__(self, other):
        return self.to_tuple() == other.to_tuple()
    def cross(self, other):
        a = self.y * other.z - self.z * other.y
        b = self.z * other.x - self.x * other.z
        c = self.x * other.y - self.y * other.x
        return Vector(a, b, c)
    def dot(self, other):
        return self.x * other.x + self.y * other.y + self.z * other.z
    def __str__(self):
        return "<{}, {}, {}>".format(*self.to_tuple())
    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y, self.z + other.z)
    def __sub__(self, other):
        return Vector(self.x - other.x, self.y - other.y, self.z - other.z)
from  math import sqrt



#Code recycled from my Kata solution
class Vector(object):

    __3DVECSIZE__= 3

    def __init__(self, *args, **kwargs):
        numArgs = len(args)
        
        if numArgs == 1:                            #Scenario: vecList is provided
            vecList = args[0]
        else:                                       #Scenario: a, b, c provided
            vecList = [args[0], args[1], args[2]]

        self.myVecSize = len(vecList)
        self.__checkForSizeException__(vecList)
        self.myComponents = vecList
        self.x = self.myComponents[0]
        self.y = self.myComponents[1]
        self.z = self.myComponents[2]
        self.magnitude = self.norm()
    #-----end constructor

    
    def __add__(self, v):
        return self.add(v)

    
    def __sub__(self, v):
        return self.subtract(v)

    
    def __eq__(self, v):
        return self.equals(v)

    
    def __str__(self):
        return self.toString('<','>')

    
    def __checkForSizeException__(self, v):
        lenPassedVec = len(v)
        if self.myVecSize != self.__3DVECSIZE__:
            raise ValueError('Missmatch of vector size: Size ', str(lenPassedVec), 'applied to vector of size ', str(self.myVecSize))  
        else:
            return lenPassedVec
    #-----end function


    def add(self, v):
        self.__checkForSizeException__(v.myComponents)
        return Vector([sum(x) for x in  zip(self.myComponents, v.myComponents)])
    #-----end function


    def subtract(self, v):
        negV = Vector([-comp for comp in v.myComponents])
        return self.add(negV)
    #-----end function

    
    #order of cross product is self cross v
    def cross(self, v):
        self.__checkForSizeException__(v.myComponents)
        xCrossComp = self.y*v.z - self.z*v.y
        yCrossComp = self.z*v.x - self.x*v.z
        zCrossComp = self.x*v.y - self.y*v.x
        return Vector([xCrossComp, yCrossComp, zCrossComp])
    #---end function

    
    def dot(self, v):
        self.__checkForSizeException__(v.myComponents)
        return (sum([ a*b for a,b in zip(self.myComponents, v.myComponents)]))
    #-----end function


    def norm(self):
        return sqrt( self.dot(self) )
    #-----end function


    def toString(self, groupSymbolLeft, groupSymbolRight):
        strVec = groupSymbolLeft
        for  i  in range(self.myVecSize-1):
            strVec += str(self.myComponents[i]) + ', '
        
        strVec += str(self.myComponents[-1]) + groupSymbolRight

        return strVec
    #-----end function

    
    def to_tuple(self):
        return tuple(self.myComponents)

    
    def equals(self, v):
        try:
            lenV = self.__checkForSizeException__(v.myComponents)
        except:
            return False 
        else:
            for i in range(lenV):
                if self.myComponents[i] != v.myComponents[i]:
                    return False
            return True
    #-----end function

#---end vector class

from math import sqrt

class Vector:
    def __init__(self, *vector):
        self.vector = vector
        if len(vector) == 1:
           self.vector = tuple(vector[0])
        self.x, self.y, self.z = self.vector
        self.magnitude = sqrt(sum(v*v for v in self.vector))

    def to_tuple(self):
        return tuple(self.vector)

    def __str__(self):
        return f'<{self.x}, {self.y}, {self.z}>'

    def __add__(self, other):
        x, y, z = (a + other.vector[i] for i,a in enumerate(self.vector))
        return Vector(x, y, z)

    def __sub__(self, other):
        x, y, z = (a - other.vector[i] for i,a in enumerate(self.vector))
        return Vector(x, y, z)
    
    def __eq__(self, other):
        return all(v == other.vector[i] for i, v in enumerate(self.vector))

    def dot(self, other):
        return sum(v * other.vector[i] for i, v in enumerate(self.vector))

    def cross(self, other):
        x = self.y * other.z - self.z * other.y
        y = -(self.x * other.z - self.z * other.x)
        z = self.x * other.y - self.y * other.x
        return Vector(x, y, z)
import math
import operator


class Vector:
    def __init__(self, *args):
        self.x, self.y, self.z = self.args = tuple(args[0] if len(args) == 1 else args)
        self.magnitude = math.sqrt(sum(i ** 2 for i in self.args))

    def __str__(self):
        return '<{}>'.format(', '.join(map(str, self.args)))

    def __eq__(self, other):
        return self.args == other.args

    def __add__(self, other):
        return Vector(*map(operator.add, self.args, other.args))

    def __sub__(self, other):
        return Vector(*map(operator.sub, self.args, other.args))

    def dot(self, other):
        return sum(map(operator.mul, self.args, other.args))

    def to_tuple(self):
        return self.args

    def cross(self, other):
        return Vector(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )
import numpy as np

class Vector:
    def __init__(self, *args):
        self.vec = np.array(args if type(args[0]) == int else args[0])
        self.x, self.y, self.z, self.magnitude = *self.vec, np.linalg.norm(self.vec)
    def __str__(self):
        return f'<{self.x}, {self.y}, {self.z}>'
    def __eq__(self, other):
        if type(other) != Vector: return False
        return np.array_equal(self.vec, other.vec)
    def __add__(self, other):
        return Vector(self.vec + other.vec)
    def __sub__(self, other):
        return Vector(self.vec - other.vec)
    def cross(self, other):
        return Vector(np.cross(self.vec, other.vec))
    def dot(self, other):
        return np.dot(self.vec, other.vec)
    def to_tuple(self):
        return tuple(self.vec)
class Vector:
    def __init__(s,*A):
        s.x,s.y,s.z=len(A)!=1 and A or A[0]
        s.magnitude=(s.x**2+s.y**2+s.z**2)**.5
    __str__=lambda s:"<%d, %d, %d>"%(s.x,s.y,s.z)
    __eq__=lambda s,o:(s.x,s.y,s.z)==(o.x,o.y,o.z)
    __add__=lambda s,o:Vector([s.x+o.x,s.y+o.y,s.z+o.z])
    __sub__=lambda s,o:Vector([s.x-o.x,s.y-o.y,s.z-o.z])
    to_tuple=lambda s:(s.x,s.y,s.z)
    dot=lambda s,o:s.x*o.x+s.y*o.y+s.z*o.z
    cross=lambda s,o:Vector(s.y*o.z-s.z*o.y,s.z*o.x-s.x*o.z,s.x*o.y-s.y*o.x)
from math import sqrt


class Vector:
    
    def __init__(self, *args):
        if isinstance(args[0], (list, tuple)):
            self.x, self.y, self.z = args[0]
        else:
            self.x, self.y, self.z = args
        
    def __add__(self, other):
        return Vector(
            self.x + other.x,
            self.y + other.y,
            self.z + other.z
        )
        
    def __sub__(self, other):
        return Vector(
            self.x - other.x,
            self.y - other.y,
            self.z - other.z
        )
        
    def __eq__(self, other):
        if isinstance(other, (list, tuple)):
            other = Vector(other)
        return (
            self.magnitude == other.magnitude and 
            self.x / other.x == self.y / other.y == self.z / other.z
        )
    
    def __str__(self):
        return "<%d, %d, %d>" % (self.x, self.y, self.z)
    
    def to_tuple(self):
        return (self.x, self.y, self.z)

    def cross(self, other):
        return Vector(
            self.y * other.z - other.y * self.z,
            -(self.x * other.z - other.x * self.z),
            self.x * other.y - other.x * self.y
        )

    def dot(self, other):
        return self.x * other.x + self.y * other.y + self.z * other.z

    @property
    def magnitude(self):
        return sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)

