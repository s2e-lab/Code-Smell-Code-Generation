def round(self, digits=0):
        """ Round the elements of the given vector to the given number of digits. """
        # Meant as a way to clean up Vector.rotate()
        # For example:
        #   V = Vector(1,0)
        #   V.rotate(2*pi)
        #   
        #   V is now <1.0, -2.4492935982947064e-16>, when it should be 
        #   <1,0>. V.round(15) will correct the error in this example.
        
        self.x = round(self.x, digits)
        self.y = round(self.y, digits)