def add_point(ori,dis,c):
    lastPoint = c[-1]
    if ori == "N":
        c.append((lastPoint[0],lastPoint[1]+dis))
    elif ori == "S":
        c.append((lastPoint[0],lastPoint[1]-dis))
    elif ori == "E":
        c.append((lastPoint[0]+dis,lastPoint[1]))
    else: 
        c.append((lastPoint[0]-dis,lastPoint[1]))

def check_corner(l_o):
    ini = l_o[0]
    fin = l_o[-1]
    if ini==fin: return False
    if ini == "N" or ini =="S": ini = "V"
    else: ini = "H"
    if fin == "N" or fin =="S": fin = "V"
    else: fin = "H"
    if ini==fin: return False
    return True

def check_intersect(rectas):
    u=rectas[-1]
    ux=[u[0][0],u[1][0]]
    ux.sort()
    uy=[u[0][1],u[1][1]]
    uy.sort()

    oriU = ""
    if ux[0] == ux[1]: oriU = "V"
    if uy[0] == uy[1]: oriU = "H"
    
    for r in rectas[:-2]:
        rx=[r[0][0],r[1][0]]
        rx.sort()
        ry=[r[0][1],r[1][1]]
        ry.sort()

        oriR = ""
        if rx[0] == rx[1]: oriR = "V"
        if ry[0] == ry[1]: oriR = "H"

        if oriU==oriR: 
            if oriU == "V" and ux[0]==rx[0]:
                if ry[0] <= uy[0] <= ry[1] or ry[0] <= uy[1] <= ry[1] :
                    return True
                if uy[0] < ry[0] and uy[1] > ry[1]:
                    return True 

            if oriU =="H" and uy[0]==ry[0]:
                if rx[0] <= ux[0] <= rx[1] or rx[0] <= ux[1] <= rx[1] :
                    return True
                if ux[0] < rx[0] and ux[1] > rx[1]:
                    return True 
        elif oriU =="V":
            if uy[0]<=ry[0]<=uy[1]:
                if rx[0] < ux[0] and rx[1] > ux[0]:
                    return True
        elif oriU =="H":
            if ux[0]<=rx[0]<=ux[1]:
                if ry[0] < uy[0] and ry[1] > uy[0]:
                    return True  
        else:
            return False

def calc_area(camino):
    parN=camino[-1][0]*camino[0][1] - camino[-1][1] * camino[0][0]
    for p in range(1,len(camino)):
        par=camino[p-1][0]*camino[p][1] - camino[p-1][1]*camino[p][0]
        parN+=par
    return abs(parN)/2

def mouse_path(s):
    camino=[(0,0)]
    distancia = 0
    listaOrientaciones = ["E"]
    rectas = []
     
    for c in s:
        orientacion = listaOrientaciones[-1]
        if c.isdigit():
            distancia=distancia*10 + int(c)
        else:
            add_point(orientacion,distancia,camino)
            rectas.append((camino[-2],camino[-1]))
            if check_intersect(rectas): return None
            if c == "L":
                if orientacion == "N": listaOrientaciones.append("O")
                elif orientacion == "S": listaOrientaciones.append("E") 
                elif orientacion == "E": listaOrientaciones.append("N")
                else: listaOrientaciones.append("S")
            else:
                if orientacion == "N": listaOrientaciones.append("E")
                elif orientacion == "S": listaOrientaciones.append("O") 
                elif orientacion == "E": listaOrientaciones.append("S")
                else: listaOrientaciones.append("N")
            distancia = 0
    add_point(orientacion,distancia,camino)
    rectas.append((camino[-2],camino[-1]))
    if check_intersect(rectas): return None

    if camino[-1] != (0,0): return None

    if not check_corner(listaOrientaciones): return None

    return calc_area(camino)
d = {'L':1, 'R':-1}
def mouse_path(s):
    if invalid(s): return None
    area, x = 0, 0
    while True:
        g = (i for i, c in enumerate(s) if c in d)
        i, j = next(g, -1), next(g, -1)
        area += x * int(s[:i])
        if j == -1: break
        x = d[s[j]] * (int(s[i+1:j]) - d[s[i]] * x)
        s = s[j+1:]
    return abs(area)
    
def invalid(s): # Checks if the path is invalid
    x, y, sgn = 0, 0, 1
    V, H = [], []
    while True:
        g = (i for i, c in enumerate(s) if c in d)
        i, j = next(g, -1), next(g, -1)
        if i == -1: return True
        a, b = sgn * int(s[:i]), d[s[i]]*sgn * int(s[i+1:] if j==-1 else s[i+1:j])
        H.append((x,y,a))
        for p,q,r in V[:-1]:
            if (y-q)**2<=(y-q)*r and (p-x)**2<=(p-x)*a:
                return True
        V.append((x+a,y,b))
        for p,q,r in H[1:-1] if j == -1 else H[:-1]:
            if (q-y)**2<=(q-y)*b and (x+a-p)**2<=(x+a-p)*r:
                return True
        x,y = x+a,y+b
        if j == -1: break
        sgn *= -d[s[i]]*d[s[j]]
        s = s[j+1:]
    return x!=0 or y!=0
    

import re

def check_intersection(y, x0, x1, segments):
    return any(x0 <= x <= x1 and y0 <= y <= y1 for x, y0, y1 in segments)

def mouse_path(s):
    sort_coords = lambda x0, x1: (x0, x1) if x0 <= x1 else (x1, x0)
    dx, dy = 1, 0
    x, y = 0, 0
    hs, vs = [], []
    area = 0
    s = re.split('([LR])', s)
    n = len(s)
    for i in range(n):
        cmd = s[i]
        if cmd == 'L':
            dx, dy = -dy, dx
        elif cmd == 'R':
            dx, dy = dy, -dx
        else:
            d = int(cmd)
            x1, y1 = x + d * dx, y + d * dy
            if dy == 0:
                a, b = sort_coords(x, x1)
                if i == n - 1:
                    return None
                if check_intersection(y, a, b, vs[:-1]):
                    return None
                hs.append((y, a, b))
            else:
                a, b = sort_coords(y, y1)
                if i == n - 1:
                    hs = hs[1:]
                if check_intersection(x, a, b, hs[:-1]):
                    return None
                vs.append((x, a, b))
            area += x * y1 - x1 * y
            x, y = x1, y1
    return abs(area // 2) if x == 0 and y == 0 else None
import re
O = lambda Q,W,E : W <= Q <= E if W < E else E <= Q <= W
def mouse_path(Q) :
  M,U,D,R,C = [],0,0,0,0
  for Q in re.findall('.d+','R' + Q) :
    D,Q = D + ('O' < Q or 3),int(Q[1:])
    r,c = [-Q,0,Q,0][D % 4] + R,[0,Q,0,-Q][D % 4] + C
    U += R * c - r * C
    for H,W,h,w in M[:-1] :
      if (R == r) ^ (H == h) :
        if O(R,H,h) and O(W,C,c) if H - h else O(H,R,r) and O(C,W,w) and not 0 == r == c == H == W : return
      elif R == r == H and (O(C,W,w) or O(c,W,w) or O(W,C,c)) or C == c == W and (O(R,H,h) or O(r,H,h) or O(H,R,r)) : return
    M.append((R,C,r,c))
    R,C = r,c
  if 0 == R == C : return abs(U + R - C) / 2
import re
def mouse_path(s):
    x, y, dx, dy, area, vert = 0, 0, 0, 1, 0, [(0,0)]
    tokes = re.findall('[RL]|d+',s)
    if len(tokes) % 4 != 3:                                 #hole not in corner
        return None
    for n in tokes:
        if n == 'L':
            dx, dy = -dy, dx;
        elif n == 'R':
            dx, dy = dy, -dx;
        else:
            d = int(n)
            x2, y2 = x + d * dx, y + d * dy
            if any(intersect((x,y),(x2,y2),vert[i],vert[i+1]) for i in range(len(vert)-4, -1+((x2,y2)==(0,0)), -2)):
                return None
            area += x * y2 - y * x2                         #area of irregular polygon
            x, y = x2, y2
            vert.append((x,y))
    if (x,y) != (0,0):                                      #path doesn't close
        return None
    return abs(area)//2 or None                             #area of irregular polygon

def intersect(a, b, c, d):
    i = a[0]==b[0]
    return (c[i]-a[i])*(c[i]-b[i]) <= 0 and (a[1-i]-c[1-i])*(a[1-i]-d[1-i]) <= 0

import re


def path_to_point(path, r=[-1, 0], position=[0, 0]):
    rotations = {'L': lambda a, b: (b * -1, a), 'R': lambda a, b: (b, a * -1)}
    result = []
    for v in re.findall(r'd+|.', path):
        if v.isnumeric():
            position = position[0] + r[0] * int(v), position[1] + r[1] * int(v)
            result.append(position)
        else:
            r = rotations[v](*r)
    return result


def check_path(points):
    print(points)
    if len(points) % 2 != 0 or points[-1] != (0, 0) or (
       len(points) != len(set(points))):
        return False

    vertical_segments, horizontal_segments = [], []
    for i in range(0, len(points), 2):
        p1, p2, p3 = points[i], points[i + 1], points[(i + 2) % len(points)]
        v = (p1, p2)
        h = (p2, p3)
        for s in vertical_segments:
            if min(s[0][1], s[1][1]) <= h[0][1] <= max(s[0][1], s[1][1]) and (
               min(h[0][0], h[1][0]) <= s[0][0] <= max(h[0][0], h[1][0]) and not
               (h[0] == s[0] or h[0] == s[1] or h[1] == s[0] or h[1] == s[1])):
                return False

        for s in horizontal_segments:
            if min(s[0][0], s[1][0]) <= v[0][0] <= max(s[0][0], s[1][0]) and (
               min(v[0][1], v[1][1]) <= s[0][1] <= max(v[0][1], v[1][1]) and not
               (v[0] == s[0] or v[0] == s[1] or v[1] == s[0] or v[1] == s[1])):
                return False

        vertical_segments.append(v)
        horizontal_segments.append(h)

    return True


def mouse_path(s):
    points = path_to_point(s)
    if not check_path(points):
        return None

    separators = sorted(list(set([p[1] for p in points])))

    _points = []
    for i in range(len(points)):
        p1, p2 = points[i], points[(i+1) % len(points)]
        _points.append(p1)
        if p1[0] == p2[0]:
            for j in (separators if p1[1] < p2[1] else separators[::-1]):
                if min(p1[1], p2[1]) < j < max(p1[1], p2[1]):
                    _points.append((p1[0], j))
    points = _points

    result = 0
    separators = list(separators)
    for i in range(len(separators) - 1):
        y1, y2 = separators[i], separators[i + 1]
        temp = []
        for p in sorted(list(filter(lambda x: x[1] == separators[i], points)))[::-1]:
            j = points.index(p)
            if points[j - 1] == (p[0], y2) or points[(j + 1) % len(points)] == (p[0], y2):
                temp.append(p)
        if len(temp) % 2 != 0:
            return None
        for j in range(0, len(temp), 2):
            result += (y2 - y1) * (temp[j][0] - temp[j + 1][0])
    return result
from collections import defaultdict

def mouse_path(s):
    #use the shoelace method to calculate area from coordinates
    
    #direction given as a tuple on coordinae plane, eg (1,0) right
    
    #set direction based on current direction and turn direction
    turns = {direction:defaultdict(tuple) for direction in ((0,1),(0,-1),(-1,0),(1,0))}
    turns[(0,-1)]['L'], turns[(0,-1)]['R'] = turns[(0,1)]['R'], turns[(0,1)]['L'] = (1,0),(-1,0) #dx,dy
    turns[(-1,0)]['L'], turns[(-1,0)]['R'] = turns[(1,0)]['R'], turns[(1,0)]['L'] = (0,-1),(0,1) #dx,dy
    
    coords = [(0,0)]
    
    #first we need to parse the string for coordinates, assume an initial direction of right
    num = []
    x=y=0
    direction = (1,0)
    
    for a in s + ' ':
        if a.isdigit():
            num.append(a)
        else:
            num = int(''.join(num))
            x+=num*direction[0]
            y+=num*direction[1]
            coords.append((x,y))
            direction = turns[direction][a]
            num = []            
    
    #check if the shape is not closed
    if coords[0] != coords[-1]: return None
    #check if we don't end in a corner (even number of coordinates, try drawing out and you will see)
    if len(coords) % 2 == 0: return None
    
    #check for intersections, check each horizonal line with each vertical line (excluding the ones beginning and
    #ending at the ends of the horizonal line) for intersections. A line is paramterized by two coordinates, its start and
    #end point. if there are more than 2 vertical lines intersecting horizonal, return None
    horizontals = [(coords[i],coords[i+1]) for i in range(0,len(coords)-1,2)]
    verticals = [(coords[i],coords[i+1]) for i in range(1,len(coords)-1,2)]
    
    for (x1,y1),(x2,y2) in horizontals:
        max_x, min_x = max(x1,x2),min(x1,x2)
        count = 0
        for (a1,b1),(a2,b2) in verticals:
            max_y,min_y = max(b1,b2),min(b1,b2)
            if min_x <= a1 <= max_x and min_y <= y1 <= max_y:
                count += 1
                if count > 2: return None
                
    #main shoelace formula
    area = 0
    j = 0
    for i in range(1,len(coords)):
        area += (coords[j][0] + coords[i][0]) * (coords[j][1] - coords[i][1])
        j=i
        
    return abs(area/2)
def mouse_path(path):
  moves = parse_path(path)
  points = convert_moves_to_points(moves)
  if (is_valid(points)):
    return calculate_using_polygon_formula(points)
    if (result == 0):
      return None
    else:
      return result
  else:
    return None

def parse_path(path):
  charIndexList = []
  for i, c in enumerate(list(path)):
    if (c == 'L' or c == 'R'):
      charIndexList.append(i)
  resultList = [('R', int(path[0:charIndexList[0]]))]
  i = 0
  while i < (len(charIndexList) - 1):
    str = path[(charIndexList[i] + 1):(charIndexList[i + 1])]
    resultList.append((path[charIndexList[i]], int(str)))
    i = i + 1
  resultList.append((path[charIndexList[-1]], int(path[(charIndexList[-1] + 1):])))
  return resultList
  
def convert_moves_to_points(moves):
  points = [(0, 0)]
  x = 0
  y = 0
  dir = 0
  for i in range(len(moves)):
    if (dir == 0):
      if (moves[i][0] == 'L'):
        x = x - moves[i][1]
      else:
        x = x + moves[i][1]
    elif (dir == 1):
      if (moves[i][0] == 'L'):
        y = y + moves[i][1]
      else:
        y = y - moves[i][1]
    elif (dir == 2):
      if (moves[i][0] == 'L'):
        x = x + moves[i][1]
      else:
        x = x - moves[i][1]
    elif (dir == 3):
      if (moves[i][0] == 'L'):
        y = y - moves[i][1]
      else:
        y = y + moves[i][1]
    if (moves[i][0] == 'L'):
      if (dir == 0):
        dir = 3
      else:
        dir = (dir - 1) % 4
    else:
      dir = (dir + 1) % 4
    points.append((x, y))
  return points

def is_valid(points):
  if (points[0] != points[-1]):
    return False
  if (points[-2][0] != 0):
    return False
  if (len(points) - 1 != len(set(points[1:]))):
    return False
  lines = list(zip(points, points[1:]))
  i = 2
  while (i < len(lines)):
    j = i - 2
    while (j >= 0):
      if (intersect(lines[i][0], lines[i][1], lines[j][0], lines[j][1])):
        if (not(i == len(lines) - 1 and j == 0)):
          return False  
      j = j - 1
    i = i + 1
  return True
  
  
def intersect(p1, p2, p3, p4):
  linesParallel = (p1[0] == p2[0] and p3[0] == p4[0]) or (p1[1] == p2[1] and p3[1] == p4[1])
  if (linesParallel):
    if (p1[0] == p2[0] and p1[0] == p3[0]):
      secondAbove = max(p2[1], p1[1]) >= min(p3[1], p4[1]) and min(p1[1], p2[1]) <= min(p3[1], p4[1])
      firstAbove = min(p1[1], p2[1]) <= max(p3[1], p4[1]) and  min(p1[1], p2[1]) >= min(p3[1], p4[1])
      return secondAbove or firstAbove
    elif (p1[1] == p2[1] and p1[1] == p3[1]):
      secondAbove = max(p3[0], p4[0]) >= min(p1[0], p2[0]) and max(p3[0], p4[0]) <= max(p1[0], p2[0])
      firstAbove = min(p3[0], p4[0]) >= min(p1[0], p2[0]) and min(p3[0], p4[0]) <= max(p1[0], p2[0])
      return secondAbove or firstAbove
    return False
  isSecondLineVertical = p4[0] == p3[0]
  if (isSecondLineVertical):
    return min(p1[0], p2[0]) <= p4[0] and p4[0] <= max(p1[0], p2[0]) and min(p3[1], p4[1]) <= p1[1] and max(p3[1], p4[1]) >= p1[1]
  else:
    return p3[1] >= min(p1[1], p2[1]) and p3[1] <= max(p1[1], p2[1]) and min(p3[0], p4[0]) <= p1[0] and max(p3[0], p4[0]) >= p1[0]
  
def calculate_using_polygon_formula(points):
  result = 0
  for i in range(len(points) - 1):
    p1 = points[i]
    p2 = points[i + 1]
    result = result + (p1[0] * p2[1] - p1[1] * p2[0])
  return int(abs(result) / 2)
class point():
    def __init__(self, x, y):
        self.x, self.y = x, y
    __str__ = lambda self: f"{self.x}, {self.y}"
    __add__ = lambda self, p2: point(self.x + p2.x, self.y + p2.y)
    __mul__ = lambda self, num: point(self.x * num, self.y * num)
    __eq__  = lambda self, p2: True if self.x == p2.x and self.y == p2.y else False

def turn_to_dir(c_dir, turn):
    return point(-c_dir.y, c_dir.x) if turn == 'L' else point(c_dir.y, -c_dir.x)

def str_to_points(string):
    last_index = 0
    direction = point(1,0)
    points = [point(0,0)]
    for index, symbol in enumerate(string + 'L'):
        if symbol in 'LR':
            points.append(points[-1] + direction * int(string[last_index:index]))
            direction = turn_to_dir(direction, symbol)
            last_index = index + 1
    return points

def doIntersect(seg1, seg2):
    # seg1 = [(p0_x, p0_y), (p1_x, p1_y)]
    # seg2 = [(p2_x, p2_y), (p3_x, p3_y)]
    p0_x = seg1[0].x
    p0_y = seg1[0].y
    p1_x = seg1[1].x
    p1_y = seg1[1].y
    p2_x = seg2[0].x
    p2_y = seg2[0].y
    p3_x = seg2[1].x
    p3_y = seg2[1].y

    s1_x = p1_x - p0_x
    s1_y = p1_y - p0_y
    s2_x = p3_x - p2_x
    s2_y = p3_y - p2_y

    denom = -s2_x * s1_y + s1_x * s2_y

    s = (-s1_y * (p0_x - p2_x) + s1_x * (p0_y - p2_y)) / denom if denom != 0 else -1
    t = ( s2_x * (p0_y - p2_y) - s2_y * (p0_x - p2_x)) / denom if denom != 0 else -1

    if 0 <= s <= 1 and 0 <= t <= 1:
        return True

def check(points):
    # the path doesn't complete the loop back to the mousehole
    if points[0] != points[-1]:
        return False
    # the mousehole is located along the straight path of a wall
    if points[1].x == points[-2].x or points[1].y == points[-2].y:
        return False
    # the path intersects/overlaps itself
    points_count = len(points) - 1
    for i in range(points_count - 3):
        for j in range(i+2, points_count - 1):
            if doIntersect((points[i], points[i+1]), (points[j], points[j+1])):
                return False
    for i in range(1, points_count - 2):
        # print((str(points[0]), str(points[-2])), (str(points[i]), str(points[i+1])))
        if doIntersect((points[0], points[-2]), (points[i], points[i+1])):
            return False
    return True

def area(p):
    return 0.5 * abs(sum(segment[0].x*segment[1].y - segment[1].x*segment[0].y
                         for segment in segments(p)))

def segments(p):
    return zip(p, p[1:] + [p[0]])

def mouse_path(s):
    points = str_to_points(s)
    return area(points) if check(points) else None
class Line:
    def __init__(self,p1, p2):
        self.p1 = p1
        self.p2 = p2
    def printLine(self):
        print(("["+self.p1.printPoint()+","+self.p2.printPoint()+"] "))
        
class Point:
    def __init__(self,x, y):
        self.x = x
        self.y = y
        
    def printPoint(self):
        return "("+str(self.x)+","+ str(self.y)+")"

def get_my_key(line):
    firstX = line.p1.x
    secondX = line.p2.x
    if(firstX == secondX):
        return firstX+.1
    if firstX < secondX:
        return firstX
    else:
        return secondX
    
def rearrange(lines):
    for line in lines:
        if line.p1.y > line.p2.y:
            temp = line.p1.y
            line.p1.y = line.p2.y
            line.p2.y = temp
        if line.p1.x > line.p2.x:
            temp = line.p1.x
            line.p1.x = line.p2.x
            line.p2.x = temp
    return lines

def polygonArea(X, Y, n): 
    # Initialze area 
    area = 0.0
  
    # Calculate value of shoelace formula 
    j = n - 1
    for i in range(0,n): 
        area += (X[j] + X[i]) * (Y[j] - Y[i]) 
        j = i   # j is previous vertex to i 
      
  
    # Return absolute value 
    return int(abs(area / 2.0)) 
   
    
def mouse_path(s):
    length = len(s)
    xCoor = 0
    yCoor = 0
    point1 = Point(xCoor,yCoor)
    lines = []
    X = [0]
    Y = [0]
    direction = 1 #1 for right, 2 for left, 3 for up and 4 for down
    number = ""
    for i in range(length):
        if s[i] == 'L' or s[i] == 'R':
            if s[i] == 'L':
                if direction == 1:
                    direction = 3
                elif direction == 2:
                    direction = 4
                elif direction == 3:
                    direction = 2
                elif direction == 4:
                    direction = 1     
            else:
                if direction == 1:
                    direction = 4
                elif direction == 2:
                    direction = 3
                elif direction == 3:
                    direction = 1
                elif direction == 4:
                    direction = 2     
        else:
            number += s[i]
            if i+1 == length or s[i+1] == "L" or s[i+1] == "R":
                if direction == 1:
                    xCoor = xCoor + int(number)
                elif direction == 2:
                    xCoor = xCoor - int(number)
                elif direction == 3:
                    yCoor = yCoor + int(number)
                elif direction == 4:
                    yCoor = yCoor - int(number)
                point2 = Point(xCoor,yCoor)
                line = Line(point1,point2)
                X.append(xCoor)
                Y.append(yCoor)
                point1 = Point(point2.x,point2.y)
                lines.append(line)
                number = ""
               
    
    if lines[0].p1.x != lines[len(lines)-1].p2.x and lines[0].p1.y != lines[len(lines)-1].p2.y:
        return None
    if lines[0].p1.y == lines[len(lines)-1].p1.y and lines[0].p2.y == lines[len(lines)-1].p2.y:
        return None
    
    X.pop()
    Y.pop()
    lines.sort(key=get_my_key)
    lines = rearrange(lines)
    
    lines_copy = lines.copy()
    lines_entered = []
    
     
    #testing for intersects
    while len(lines_copy)>0:
        
        removeEntered = False
        line = lines_copy[0]
        for entLine in lines_entered:
                if entLine.p1.y == line.p1.y and entLine.p2.x > line.p1.x and line.p1.x != line.p2.x:
                    return None
                if entLine.p2.x < line.p1.x:
                    line = entLine
                    removeEntered = True
                    break
                    
        if removeEntered == True:
            lines_entered.remove(line)
            
        else:      
            if line.p1.x == line.p2.x:
                for entLine in lines_entered:
                    if entLine.p1.y > line.p1.y and entLine.p2.y < line.p2.y:
                        return None
                lines_copy.pop(0)
            else:
                    lines_entered.append(line)
                    lines_copy.pop(0)
                    
        removeEntered = False
        
        n = len(X)
    return polygonArea(X,Y,n)

    pass

