def cal_n_bug(n_head, n_leg, n_wing):
    spider = (n_leg-n_head*6)//(8-6)
    butterfly = (n_wing-(n_head-spider))
    dragonfly = n_head-spider-butterfly
    return [spider,butterfly,dragonfly] if spider>=0 and butterfly>=0 and dragonfly>=0 else [-1,-1,-1]
def cal_n_bug(n_head, n_leg, n_wing):
    s = (n_leg - 6 * n_head) / 2
    b = 8 * n_head - n_wing - n_leg
    d = s + n_wing - n_head
    return [s, b, d] if s >= 0 and b >= 0 and d >= 0 else [-1, -1, -1]
def cal_n_bug(head, leg, wing):
    # x+y+z = head, 8x+6y+6z = leg, y+2z = wing
    x = leg / 2 - head * 3
    head -= x
    y = head * 2 - wing
    z = head - y
    if all(n >= 0 for n in (x, y, z)): return [x, y, z]
    return [-1, -1, -1]

def cal_n_bug(n_head, n_leg, n_wing):
        n_s=(n_leg-6*n_head)/2
        n_d=(n_leg-8*n_head+2*n_wing)/2
        n_b=(-n_leg+8*n_head-n_wing)
        if n_s>-1 and n_d>-1 and n_b>-1:
            return [n_s,n_b,n_d] 
        else:
            return [-1,-1,-1]
def cal_n_bug(h, l, p):
    #s+d+b=h / 8s+6d+6b=l / 0s+2d+b=p
    d=l/2-4*h+p
    b=p-2*d
    s=h-d-b
    return [-1,-1,-1] if s<0 or b<0 or d<0 else [s,b,d]
def cal_n_bug(h,l,w):   
    lst=[l//2-3*h,8*h-l-w,w-4*h+l//2] 
    return lst if lst[0]>=0 and lst[1]>=0 and lst[2]>=0 else [-1,-1,-1]
def cal_n_bug(n_head, n_leg, n_wing):
    if n_head<0 or n_leg<0 or n_wing<0:
        return [-1,-1,-1]
    x=n_leg-6*n_head
    y=8*n_head-n_leg-n_wing
    z=n_leg+2*n_wing-8*n_head
    if x%2!=0 or z%2!=0 or x<0 or y<0 or z<0:
        return [-1,-1,-1]
    return [x//2,y,z//2]
def cal_n_bug(heads, legs, wings):
    spiders = legs // 2 - heads * 3
    dragonflies = wings + spiders - heads
    r = [spiders, heads - spiders - dragonflies, dragonflies]
    return r if all(x >= 0 for x in r) else [-1] * 3
def cal_n_bug(h,l,w):
    if h<0 or l<0 or w<0 or l%2 or l<6*h or l>8*h: return [-1,-1,-1]
    d,b=w//2,w%2
    s=h-d-b
    return [h-d-b,b,d] if s*8+d*6+b*6==l else [-1,-1,-1]
import numpy as np

def cal_n_bug(x, y, z):
    a = np.array([[1,1,1], [8,6,6], [0,2,1]])
    b = np.array([x, y, z])
    s = np.linalg.solve(a, b)
    return list(map(int, s)) if all(x>=0 for x in s) else [-1, -1, -1]
