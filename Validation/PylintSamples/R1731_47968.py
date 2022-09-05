def create_ellipse(self,xcen,ycen,a,b,ang,resolution=40.0):
        """Plot ellipse at x,y with size a,b and orientation ang"""

        import math
        e1=[]
        e2=[]
        ang=ang-math.radians(90)
        for i in range(0,int(resolution)+1):
            x=(-1*a+2*a*float(i)/resolution)
            y=1-(x/a)**2
            if y < 1E-6:
                y=1E-6
            y=math.sqrt(y)*b
            ptv=self.p2c((x*math.cos(ang)+y*math.sin(ang)+xcen,y*math.cos(ang)-x*math.sin(ang)+ycen))
            y=-1*y
            ntv=self.p2c((x*math.cos(ang)+y*math.sin(ang)+xcen,y*math.cos(ang)-x*math.sin(ang)+ycen))
            e1.append(ptv)
            e2.append(ntv)
        e2.reverse()
        e1.extend(e2)
        self.create_line(e1,fill='red',width=1)