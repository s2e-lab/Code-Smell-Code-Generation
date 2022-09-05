def excluding_vat_price(price):
    try:
        return round(price / 1.15, 2)
    except TypeError:
        return -1
def excluding_vat_price(price):
    return round(price / 1.15, 2) if price else -1
excluding_vat_price = lambda p: round(p/115*100, 2) if p else -1
def excluding_vat_price(price, vat=15):
    return -1 if price is None else round(price / (1 + vat/100), 2)
def excluding_vat_price(price):
    return round(price / 1.15, 2) if price is not None else -1
# surprise
def excluding_vat_price(prIce):
    return round(prIce/1.15, 2) if prIce != None else -1
def excluding_vat_price(price):
    return -1 if price is None else round(price/1.15,2)
def excluding_vat_price(price):
    return float('%.2f'%(price/115*100)) if price else -1
def excluding_vat_price(price):
    return round(20/23*price,2) if price else -1
def excluding_vat_price(price):
    return round(price/1.15,2) if price!=None else -1
def excluding_vat_price(price):
    return round(float(price/1.15), 2) if price != None else -1
def excluding_vat_price(price: int) -> float:
    """ Calculate the original product price, without VAT. """
    return -1 if not price else round(price / 1.15, 2)
excluding_vat_price=lambda n:-(not n)or round(n/1.15,2)
def excluding_vat_price(price):
    return -1 if price == None else float('{0:.2f}'.format(price/1.15))

def excluding_vat_price(price):
    return float(f"{(price / 1.15):.2f}") if price != None else -1
def excluding_vat_price(price):
    return price and round(price / 1.15, 2) or -1
def excluding_vat_price(price):
    return round(price*20/23,2) if price != None else -1
excluding_vat_price=lambda Q:-1 if None==Q else round(Q/1.15,2)
def excluding_vat_price(price):
    if type(price) is int or type(price) is float:
        return round(price / 1.15,2)
    else:
        return -1
import math

def excluding_vat_price(price):
    return round(price / 115 * 100, 2) if price else -1
def excluding_vat_price(price):
    print(price)
    return -1 if price == 0 or price == None else round(float(price) / 1.15, 2)
def excluding_vat_price(price):
    if price!=None:
          ergebnis= (price/1.15)
 
          return round(ergebnis,2)
    else:
          return -1
def excluding_vat_price(price):
    return round(price - (-1 * (price / 1.15 - price)), 2) if price else -1
def excluding_vat_price(price):
    if price:
        return float(format((price-price*15/115), '.2f'))
    else:
        return -1
def excluding_vat_price(price):
    if not price:
        return -1
    return round(100 * price / 115, 2)
import numpy as np
def excluding_vat_price(price):
    try:
        return np.round((price/1.15), 2)
    except:
        return -1
import math

def excluding_vat_price(price):
    if price:
        per = price*15/115
        return round(price-per,2)
    else:
        return -1
def excluding_vat_price(price):
    return round(price - (price *(0.15/1.15)),2) if price else -1
def excluding_vat_price(price):
    if price:
        return round(100 / 115 * price, 2)
    else:
        return -1
def excluding_vat_price(price):
    if price is None:
        return -1    
    sum = price - (price * (15 / 115))
    if price != None:
        return float('{:.2f}'.format(sum))
def excluding_vat_price(price):
    return -1 if not price else round((price*100/115), 2)
def excluding_vat_price(price):
    if price == 0 or price == None:
        return -1
    else:
        return round(100*price/115, 2)
def excluding_vat_price(price):
    try: 
        x = round(price/1.15,2) 
    except: 
        x = -1
    return x
def excluding_vat_price(price):
    if price == 0: 
        return 0
    return round((price / (1 + 0.15)),2) if price != None else -1
def excluding_vat_price(price):
    try:
        origin = float(round(price / 1.15, 2))
        return origin
    except TypeError:
        return -1
def excluding_vat_price(price):
    return -1 if price == None else round(100 / 115 * price, 2)
def excluding_vat_price(price):
    if price is None:
        return -1
    return round(price * 10000 / 115) / 100
def excluding_vat_price(price) -> float:
    try:
        return round( price/1.15, 2)
    except TypeError:
        return -1
def excluding_vat_price(price):
    try : return round(price/1.15, 2)
    except : return -1 # o_O
def excluding_vat_price(price):
    if price is None:
        return -1
    elif price is not None:
        ogprice = 1/1.15 * price
        return round(ogprice,2)
def excluding_vat_price(price):
    if int == type(price)or float== type(price):
        return round(price/1.15,2)
    return -1
def excluding_vat_price(price):
    if price is None:
        return -1
    res = price / 1.15
    return round(res, 2)
def excluding_vat_price(price):
    if type(price) is None:
        return -1
    else:
        try:
            str(price)
            return round(price / 1.15, 2)
        except:
            return -1
def excluding_vat_price(price):
    try:
        return round(price / 1.15,2)
    except TypeError as type_err:
        return -1
def excluding_vat_price(final_price):
    if final_price == None:
        return -1
    else:
        price = final_price * 100 / 115
        return round(price, 2) 
def excluding_vat_price(price):
    if price!=None:
        val=float(price)/1.15
        return round(val,2)
    return -1
def excluding_vat_price(price):

    if price == None:
        return -1
    
    tax = price - price/1.15
    
    no_tax_price = format(price - tax,".2f")
    
    return float(no_tax_price)
def excluding_vat_price(price):
    """
    Calculate price before VAT
    """
    return round(price/1.15,2) if(isinstance(price,float) or isinstance(price,int)) else -1

def excluding_vat_price(price):
    return round(price/1.15,2) if isinstance(price,(float,int)) else -1
def excluding_vat_price(price):
    return round(price / 1.15 ,2) if type(price) == float or type(price) == int else -1

def excluding_vat_price(price):
    return round(float(price or -1.15) / 1.15, 2)
def excluding_vat_price(p):
    return round(p / 1.15, 2) if p else -1
def excluding_vat_price(price):
    if price == None:
        result = -1
    else: 
        result = price / 1.15
        
    return round (result, 2)
def excluding_vat_price(price):
    if price :
        result = round(price / 1.15, 2)
    else:
        result = -1
    return result
def excluding_vat_price(price):
    return round(price/(1+0.15),2) if price else -1
def excluding_vat_price(price):
    VAT = 15
    return round(price / (100 + VAT) * 100, 2) if price else -1
def excluding_vat_price(price):
    
    if price:
        return float("{:.2f}".format(price / 1.15))
        
    else:
        return -1 
        
        
        
        
        

def excluding_vat_price(price):
    
    if str(price) == "None":
        return -1

    if price <= 0:
        return -1
    
        
    sonuc = (price) / 1.15
    

    return round(sonuc,2)
def excluding_vat_price(price):
    if not price == None:
        return round(price / 1.15, 2)
    return -1
excluding_vat_price=lambda p:round(p*100/115,2) if p else -1
def excluding_vat_price(price):
    return round(price-price*0.1304347826086957,2) if price!=None else -1
def excluding_vat_price(price):
    if price == None:
        return -1
    else:
        b = (price / 1.15)
        c = ("%.2f" % b)
        return float(c)
def excluding_vat_price(price):
    if type(price) == int or type(price) == float:
        return round(200/230*price,2)
    else:
        return -1

def excluding_vat_price(price=None):
    return -1 if price is None else round(price/1.15, 2)
def excluding_vat_price(price):
    if price:
        return round(price / 115 * 100, 2)
    else:
        return -1
def excluding_vat_price(price):
    return -1 if price == '' or  price is None else float(str(round(((1/1.15) * price),2)))
def excluding_vat_price(price):
    if price == None: return -1
    if price > 0: return round (price/1.15, 2)
    else: return 0
def excluding_vat_price(price):
    if type(price) != int and type(price) != float:
        return -1
    else:
        return round((price/1.15),2)

def excluding_vat_price(price):
    if price == None:
        return -1 
    else:
        return round(float(price/1.15),2)

def excluding_vat_price(price=0):

    if type(price)!=int:
        if type(price)!=float:
            return -1
   
    return round((price/115)*100,2)
def excluding_vat_price(p):
    return round((p/1.15),2) if p is not None else -1
def excluding_vat_price(price):
    return -1 if price is None else round(((price*100)/15)/(1+100/15), 2)
def excluding_vat_price(price):

    return(round((price / 1.15), 2) if price is not None and price >= 0 else -1)
def excluding_vat_price(price):
    try:
        price_ex_vat = round(price / 115 * 100, 2)
    except TypeError:
        return -1
    
    return price_ex_vat
def excluding_vat_price(price):
    if not price:
        return -1
    VAT = 0.15
    return round(price / (VAT + 1), 2)
def excluding_vat_price(price):
    if price == None:
        return -1
    return round(100 * price / 115,2)
def excluding_vat_price(price):
    try:
        return -1 if price<0 else round(price/1.15,2)
    except:
        return -1
def excluding_vat_price(price):
    return round(20 / 23 * price, 2) if price is not None else -1
def excluding_vat_price(price):
    if price:
        return round((100 * price) / 115, 2)
    return -1
def excluding_vat_price(price):
    if price == None: return -1
    return round(price / 115 * 100, 2)
def excluding_vat_price(price):
    return round(price/1.15, 2) if type(price) in [int,float] else -1
def excluding_vat_price(price):
    return -1 if not price else round(100 * price / 115, 2)
def excluding_vat_price(price):
    return -1 if price == 0 or price == None else round(price / 1.15, 2)
def excluding_vat_price(p):
    try:
        return round(p/1.15,2)
    except:
        return -1
def excluding_vat_price(price):
    return float('{:.2f}'.format(price/1.15)) if price else -1
def excluding_vat_price(price):
    return round(price * 100.0 / 115.0, 2) if price else -1
def excluding_vat_price(price):
    try:
        return round(price*100/115,2)
    except:
        return -1
def excluding_vat_price(price):
    return -1 if price == None else float('{:.2f}'.format(price / 1.15))
def excluding_vat_price(price):
    if price == None:
        return -1
    wo = price / 1.15
    return wo.__round__(2)
def excluding_vat_price(price):
    #if price == 0:
        #return -1
    ##p = (price / 1.15)
    #return float("{0:.2f}".format(p))
    #return round(p,2)
    try:
        return round(price / 1.15, 2)
    except TypeError:
        return -1
def excluding_vat_price(price):
    if price == 0 or price == None :
        return -1
    return(round(price/1.15,2))
def excluding_vat_price(price):
    if price==None:
        return -1
    else:
        return round(price*20/23,2);
def excluding_vat_price(price):
    if price:
        return round(price - ((price / (1 + .15) * .15)), 2)
    return -1
def excluding_vat_price(price):
    return round(100*price/115, 2) if price != None else -1
def excluding_vat_price(price):
    return round((price * .8695652173913043), 2) if price else -1

def excluding_vat_price(price):
    return round(price*0.86956,2) if price else -1
def excluding_vat_price(price):
    if price == None:
        return -1
    something = price / (1 + ( 15 / 100))
    return round(something, 2)
def excluding_vat_price(price):
    if price == 'null' or price == None :
        return -1
    return round(price - (price * (1- 0.86956521739)),2)     
