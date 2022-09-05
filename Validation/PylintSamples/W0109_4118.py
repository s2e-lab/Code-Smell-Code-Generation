def identify_weapon(character):
    tbl = {
      "Laval"    : "Laval-Shado Valious",
      "Cragger"  : "Cragger-Vengdualize",
      "Lagravis" : "Lagravis-Blazeprowlor",
      "Crominus" : "Crominus-Grandorius",
      "Tormak"   : "Tormak-Tygafyre",
      "LiElla"   : "LiElla-Roarburn"
    }
    
    return tbl.get(character, "Not a character")
def identify_weapon(character):
    #insert your code here...FOR CHIMA!
    try:
        return character + "-" + {
        "Laval":"Shado Valious", "Cragger":"Vengdualize",
        "Lagravis":"Blazeprowlor","Crominus":"Grandorius",
        "Tormak":"Tygafyre", "LiElla":"Roarburn"
        }[character]
    except:
        return "Not a character"
weapon_map = {
'Laval':'Laval-Shado Valious',
'Cragger': 'Cragger-Vengdualize',
'Lagravis':'Lagravis-Blazeprowlor',
'Crominus': 'Crominus-Grandorius',
'Tormak': 'Tormak-Tygafyre',
'LiElla': 'LiElla-Roarburn',
}

def identify_weapon(character):
    return weapon_map.get(character, 'Not a character')
def identify_weapon(character):
    weps = {'Laval':'Shado Valious',
            'Cragger':'Vengdualize', 
            'Lagravis':'Blazeprowlor', 
            'Crominus':'Grandorius', 
            'Tormak':'Tygafyre', 
            'LiElla':'Roarburn'}
    try:
        return "{}-{}".format(character, weps[character])
    except:
        return "Not a character"
def identify_weapon(character):
    data = {
            'Laval': 'Shado Valious',
            'Cragger': "Vengdualize",
            "Lagravis": "Blazeprowlor",
            "Crominus": "Grandorius",
            "Tormak": "Tygafyre",
            "LiElla": "Roarburn"
            }
    try:
        return "%s-%s" % (character, data[character])
    except KeyError:
        return "Not a character"
def identify_weapon(character):
  weapons = { 'Laval' : 'Shado Valious', 'Cragger' : 'Vengdualize', 'Lagravis' : 'Blazeprowlor', 'Crominus' : 'Grandorius', 'Tormak' : 'Tygafyre', 'LiElla' : 'Roarburn' }
  return '%s-%s' % (character, weapons[character]) if character in weapons else 'Not a character'

def identify_weapon(character):
    dict = {"Laval" : "Shado Valious",
            "Cragger" : "Vengdualize",
            "Lagravis" : "Blazeprowlor",
            "Cragger" : "Vengdualize",
            "Lagravis" : "Blazeprowlor",
            "Crominus" : "Grandorius",
            "Tormak" : "Tygafyre",
            "LiElla" : "Roarburn"}
    if character not in dict.keys():
        return "Not a character"
    return character + "-" + dict[character]
weapons = [
    "Laval-Shado Valious", "Cragger-Vengdualize", "Lagravis-Blazeprowlor",
    "Crominus-Grandorius", "Tormak-Tygafyre", "LiElla-Roarburn"
]

def identify_weapon(character):
    return next((weapon for weapon in weapons if weapon.startswith(character)), "Not a character")
def identify_weapon(character):
    wep = {
    "Laval":"Laval-Shado Valious",
    "Cragger":"Cragger-Vengdualize",
    "Lagravis":"Lagravis-Blazeprowlor",
    "Crominus":"Crominus-Grandorius",
    "Tormak":"Tormak-Tygafyre",
    "LiElla":"LiElla-Roarburn"
    }
    
    return wep.get(character, "Not a character")
def identify_weapon(character):
    d = {'Laval' : 'Shado Valious', 
         'Cragger' : 'Vengdualize', 
         'Lagravis' : 'Blazeprowlor', 
         'Crominus' : 'Grandorius', 
         'Tormak' : 'Tygafyre', 
         'LiElla' : 'Roarburn'}
         
    return f'{character}-{d[character]}' if character in d else 'Not a character'
