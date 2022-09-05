def rpg(field, actions):
    p = Player(field)
    try:
        for m in actions:
            if m=='A': p.attack()
            elif m in 'HCK':  p.use(m)
            elif m in '<^>v': p.rotate(m)
            p.checkDmgsAndAlive()
            if m=='F':        p.move()
            
    except Exception as e:
        return None
    return p.state()
    

class Player:
    DIRS = dict(list(zip('<>^v',((0,-1),(0,1),(-1,0),(1,0)))))

    def __init__(self,field):
        self.h, self.atk, self.d, self.bag, self.xps = 3,1,1,[],0
        self.field = field
        self.pngs  = {}
        for x,r in enumerate(self.field):
            for y,c in enumerate(r):
                if c in self.DIRS: self.x,self.y,self.c=x,y,c ; self.dx,self.dy=self.DIRS[c]
                elif c=='D':       self.pngs[(x,y)] = {'h':10, 'atk':3}
                elif c=='E':       self.pngs[(x,y)] = {'h':1,  'atk':2}
                elif c=='M':       self.pngs['M']   = {'coins':3}
    
    def state(self): return self.field, self.h, self.atk, self.d, sorted(self.bag)
    
    def rotate(self,c):
        self.dx, self.dy = self.DIRS[c]
        self.c = self.field[self.x][self.y] = c
    
    def move(self):
        self.field[self.x][self.y] = ' '
        self.x += self.dx
        self.y += self.dy
        c = self.field[self.x][self.y]
        assert c not in '#ED-|M' and self.x>=0 and self.y>=0
        if c!=' ': self.takeThis(c)
        self.field[self.x][self.y] = self.c
    
    def checkAhead(self,what):
        x,y = self.x+self.dx, self.y+self.dy
        assert self.field[x][y] in what
        return x,y
    
    def takeThis(self,c):
        if c not in 'SX': self.bag.append(c)
        if   c=='S': self.d += 1
        elif c=='X': self.atk += 1
        
    def use(self,c):
        self.bag.remove(c)
        if c=='C':
            x,y = self.checkAhead('M')
            self.pngs['M']['coins'] -= 1
            if not self.pngs['M']['coins']: self.field[x][y] = ' '
        elif c=='H':
            assert self.h<3
            self.h = 3
        elif c=='K':
            x,y = self.checkAhead('|-')
            self.field[x][y] = ' '
        
    def attack(self):
        x,y = nmy = self.checkAhead('ED')
        self.pngs[nmy]['h'] -= self.atk
        if self.pngs[nmy]['h']<1:
            del self.pngs[nmy]
            self.field[x][y] = ' '
            lvlUp,self.xps = divmod(self.xps+1,3)
            self.atk += lvlUp
        
    def checkDmgsAndAlive(self):
        for dx,dy in list(self.DIRS.values()):
            nmy = self.x+dx, self.y+dy
            if nmy in self.pngs:
                self.h -= max(0,self.pngs[nmy]['atk'] - self.d)
                assert self.h>0
        

only_show_wrong()
class RPG:
    
    def __init__(self, field, actions):
        frame = ['#'] * (len(field[0])+2)
        self.field = [frame] + [['#'] +  line + ['#'] for line in field]  + [frame]
        self.actions = actions
        self.l = len(self.field)
        self.pack = {'H':3, 'X':1, 'S':1,}
        self.bag = []
        self.moves = {'<':(0,-1) ,'>':(0,1), '^':(-1,0), 'v':(1,0)}
        self.x = self.find_player(self.field, lambda e:any(('<' in e, '>' in e, '^' in e, 'v' in e)))
        self.y = self.find_player(self.field[self.x], lambda e:e in '<>^v')
        self._player = self.field[self.x][self.y]
        self.back = None
        self.boss = 10
        self._result = 1
        self.use_coint = 0
        self.level_up = 0
        self.ready_atack = []
        self.use_item = {'K':self.use_key,'C':self.coint_act,'H':self.get_energy,'A':self.atack}
        
    def __call__(self):
        for i, act in enumerate(self.actions):
            self.use_item.get(act, self.make_step)(act, act!='F')
            if not self._result:
                break
            self.enemy_status(i,act)
        if self.use_coint:
            x,y = self.step
            if self.field[x][y] != 'M':
                return
        return self.result
        
    @property
    def game_over(self):
        self._result = None
    
    @property
    def step(self):
        move = self.moves[self._player]
        return (self.x + move[0], self.y + move[1])
        
    @property
    def result(self):
        if self._result:
            return ([line[1:-1] for line in self.field[1:-1]], *self.pack.values(), sorted(self.bag))
        
    @property
    def set_player(self):
        self._player = self.field[self.x][self.y]
        return self._player
    
    def enemy_status(self,i,act):
        for x,y in [(self.x+1, self.y),(self.x-1, self.y),(self.x, self.y+1),(self.x, self.y-1)]:
            if self.field[x][y] in 'DE':
                self.ready_atack.append(self.field[x][y])
        if self.ready_atack:
            pref_a, next_a, l = self.actions, self.actions, len(self.actions)-1
            if i and i<l and pref_a[i-1] in '<>^vF' and next_a[i+1] =='F' and act=='F' or act in '<>^vAH' and i!=l or act =='F' and i<l and next_a[i+1] in '<>v^' :
                self.enemy_atack()
            self.ready_atack = []
          
    def enemy_atack(self):
        for enemy in self.ready_atack:
            self.pack['H'] -= max(0, (2,3)[enemy=='D'] - self.pack['S'])
            if self.pack['H'] <= 0:
                self.game_over
        
    def act(self):
        if self.back in '|-M#E':
            self.game_over
        if self.back in 'KCH':
            self.bag.append(self.back)
        if self.back in 'SX':
            self.pack[self.back] += 1
            
    def ded(self,x,y):
        self.field[x][y] = ' '
        
    def use_key(self,key,x):
        x, y = self.step
        if key in self.bag and self.field[x][y] in '|-':
            self.drop(key)
            self.ded(x,y)
        else:
            self.game_over
            
    def atack(self,x,z):
        x, y = self.step
        if self.field[x][y] == 'E':
            self.ded(x,y)
            self.level_up += 1
            if self.level_up == 3:
                self.pack['X'] += 1
                self.level_up   = 0
        elif self.field[x][y] == 'D':
            self.boss -= self.pack['X']
            if self.boss <= 0:
                self.ded(x,y)
        else:
            self.game_over
            
    def coint_act(self, coint , i):
        if coint not in self.bag:
            self.game_over
            return
        self.use_coint += i
        self.drop(coint)
        if self.use_coint == 3:  
            x, y = self.step
            if  self.field[x][y] == 'M':
                self.use_coint = 0
                self.ded(x,y)
            else:
                self.game_over
            
    def get_energy(self, energy,x):
        if energy not in self.bag or self.pack[energy] >=3:
            self.game_over
            return
        self.drop(energy)
        self.pack[energy] = 3
            
    def drop(self, element):
        del self.bag[self.bag.index(element)]
    
    def make_step(self, element,p):
        way = (self.set_player, element)[p]
        if not p:
            self.field[self.x][self.y] = ' ' 
            self.x, self.y = self.step
        if self.x < 1 or self.x >= self.l-1 or  self.y < 1 or self.y >= len(self.field[self.x])-1:
            self.game_over
            return
        if self.field[self.x][self.y] != ' ':
            self.back = self.field[self.x][self.y]
            self.act()
        self.field[self.x][self.y] = way
        self.set_player
        
    @staticmethod
    def find_player( field, condition ):
        return next(i for i,e in enumerate(field) if condition(e))
        
def rpg(f, a) -> Tuple[List[List[str]], int, int, int, List[str]]:
    play_game = RPG(f,a)
    return play_game() 
only_show_wrong()

def rpg(map: List[List[str]], actions: List[str]) -> Tuple[List[List[str]], int, int, int, List[str]]:
    class Hero:
        def __init__(self, coordy, coordx, pointer):
            self.coordx = coordx
            self.coordy = coordy
            self.health = 3
            self.attack = 1
            self.defence = 1
            self.pointer = pointer
            self.bag = []
            self.kills = 0
            self.lvlupkills = 0

    class Monsters:
        def __init__(self, coordy, coordx, attack, health):
            self.coordx = coordx
            self.coordy = coordy
            self.attack = attack
            self.health = health

    class DemonLord(Monsters):
        pass

    class Merchant:
        def __init__(self, coordy, coordx):
            self.coordx = coordx
            self.coordy = coordy
            self.health = 3

    def initiate(field):
        monstdict = {}
        merchdict = {}
        hero = 0
        for y in range(len(field)):
            for x in range(len(field[y])):
                if field[y][x] == "E":
                    monstdict[(y, x)] = Monsters(y, x, 2, 1)
                elif field[y][x] == "D":
                    monstdict[(y, x)] = DemonLord(y, x, 3, 10)
                elif field[y][x] in ("^", ">", "v", "<"):
                    hero = Hero(y, x, field[y][x])
                elif field[y][x] == "M":
                    merchdict[(y, x)] = Merchant(y,x)
        return (monstdict, hero, merchdict)

    def monsters_move(hero,monstdict):
        for i in [(hero.coordy, hero.coordx - 1), (hero.coordy + 1, hero.coordx), (hero.coordy, hero.coordx + 1),
                  (hero.coordy - 1, hero.coordx)]:
            if i in monstdict:
                hero.health -= max(0, monstdict[i].attack - hero.defence)

    def forward_coords(hero):
        pointer = {"^": (0, -1), ">": (1, 0), "v": (0, 1), "<": (-1, 0)}
        coordx = hero.coordx + pointer[hero.pointer][0]
        coordy = hero.coordy + pointer[hero.pointer][1]
        return (coordx, coordy)

    def move_forward(hero,field,monstdict,merchdict):
        monsters_move(hero,monstdict)
        field[hero.coordy][hero.coordx] = " "
        coords = forward_coords(hero)
        hero.coordx = coords[0]
        hero.coordy = coords[1]
        if 0 <= hero.coordy < len(field) and 0 <= hero.coordx < len(field[hero.coordy]) and field[hero.coordy][
            hero.coordx] not in ("D", "E", "M", "#", "-", "|"):
            if field[hero.coordy][hero.coordx] in ("C", "H", "K"):
                hero.bag.append(field[hero.coordy][hero.coordx])
            elif field[hero.coordy][hero.coordx] == "S":
                hero.defence += 1
            elif field[hero.coordy][hero.coordx] == "X":
                hero.attack += 1
            field[hero.coordy][hero.coordx] = hero.pointer
            return True
        else:
            return False

    def change_pointer(hero,field,monstdict,merchdict,pointer):
        monsters_move(hero,monstdict)
        hero.pointer = pointer
        field[hero.coordy][hero.coordx] = pointer
        if hero.health <= 0:
            return False
        return True

    def use_coin(hero,field,monstdict,merchdict):
        monsters_move(hero,monstdict)
        coords = forward_coords(hero)
        x = coords[0]
        y = coords[1]
        if "C" in hero.bag and (y, x) in merchdict and hero.health > 0:
            merchdict[(y, x)].health -= 1
            hero.bag.pop(hero.bag.index("C"))
            if merchdict[(y, x)].health == 0:
                field[y][x] = " "
                del merchdict[(y, x)]
            return True
        return False

    def use_potion(hero,field,monstdict,merchdict):
        if "H" in hero.bag and hero.health != 3:
            hero.health = 3
            hero.bag.pop(hero.bag.index("H"))
        else:
            return False
        monsters_move(hero,monstdict)
        return True if hero.health > 0 else False

    def use_key(hero,field,monstdict,merchdict):
        coords = forward_coords(hero)
        x = coords[0]
        y = coords[1]
        monsters_move(hero,monstdict)
        if "K" in hero.bag and 0 <= x < len(field[y]) and 0 <= y < len(field) and field[y][x] in (
                "-", "|") and hero.health > 0:
            field[y][x] = " "
            hero.bag.pop(hero.bag.index("K"))
            return True
        return False

    def attack(hero,field,monstdict,merchdict):
        coords = forward_coords(hero)
        x = coords[0]
        y = coords[1]
        if (y, x) in monstdict:
            monstdict[(y, x)].health -= hero.attack
            if monstdict[(y, x)].health <= 0:
                hero.lvlupkills += 1
                if hero.lvlupkills == 3:
                    hero.attack += 1
                    hero.lvlupkills = 0
                hero.kills += 1
                field[y][x] = " "
                del monstdict[(y, x)]
        else:
            return False
        monsters_move(hero,monstdict)
        return True if hero.health > 0 else False

    field = [[str(x) for x in line] for line in map]
    initialize = initiate(field)
    monstdict = initialize[0]
    hero = initialize[1]
    merchdict = initialize[2]

    actionsdict = {"F": move_forward, "A": attack, "C": use_coin, "K": use_key, "H": use_potion}
    for i in actions:
        if i in actionsdict:
            flag = actionsdict[i](hero,field,monstdict,merchdict)
        else:
            flag = change_pointer(hero,field,monstdict,merchdict,i)
        if not flag or hero.health <=0:
            return None
    return (field, hero.health, hero.attack, hero.defence, sorted(hero.bag))
only_show_wrong()


def rpg(game: List[List[str]], actions: List[str]) -> Tuple[List[List[str]], int, int, int, List[str]]:

    # Scan through the game to find the player
    # Any invalid action, or death, returns None

    # If player is attacking
    # Decrease health of enemy in front, or return None if invalid
    # If enemy dies, update map

    # If player is moving
    # Update location on map, or return None if invalid

    # If player is rotating
    # Update player character inside array

    # If player is using object
    # Check if object is in bag and can be used
    # Is valid object usage?
    # Potion - is player low on health?
    # Coin   - is there a merchant in front of player?
    # Key    - is there a door in front of player?
    # Use object, or return None if invalid

    # After all action (related to where the player WAS, not their new location)
    # If there are enemies to attack player
    # Decrease player health, check is alive

    class GameObject:
        def __init__(self, game, h, a, d, l=None):
            self.g = game
            self.l = l if l is not None else (0, 0)
            self.h = h  # health
            self.a = a  # attack
            self.d = d  # defense

        @property
        def is_alive(self):
            return self.h > 0

        def take_damage(self, attack):
            damage = max(0, attack - self.d)
            self.h -= damage
            if not self.is_alive:
                self.handle_death()
                return True

        def attack(self, other):
            return other.take_damage(self.a)

        def handle_death(self):
            self.g.removeObj(self.l)

    class Player(GameObject):

        direction_table = {
            "^": lambda self: (self.l[0], self.l[1]-1),
            ">": lambda self: (self.l[0]+1, self.l[1]),
            "v": lambda self: (self.l[0], self.l[1]+1),
            "<": lambda self: (self.l[0]-1, self.l[1]),
        }

        def __init__(self, game, s, l=None, b=None):
            super().__init__(game, 3, 1, 1, l)
            self.s = s  # symbol, "^" ">" "v" or "<"
            self.b = b if b is not None else []
            self.exp = 0

        def __check_forward(self):
            new_loc = Player.direction_table[self.s](self)
            if self.g.is_within_bounds(new_loc):
                x, y = new_loc[0], new_loc[1]
                obj = self.g.field[y][x]
                return (new_loc, obj)
            else:
                return None

        def handle_death(self):
            self.g.player = None
            super().handle_death()

        def __find_item(self, item):
            for i in range(len(self.b)):
                if self.b[i] == item:
                    return self.b.pop(i)
            return None

        def __pickup_item(self, item):
            if item in "CKH":
                self.b.append(item)
            elif item == "X":
                self.a += 1
            else:   # == "S"
                self.d += 1

        def rotate(self, symb):
            self.s = symb
            x, y = self.l[0], self.l[1]
            self.g.field[y][x] = self.s

        def move(self):
            new_loc = self.__check_forward()
            if new_loc is None or new_loc[1] in "#MED-|":
                return False
            if new_loc[1] in "CKHSX":
                self.__pickup_item(new_loc[1])

            self.g.removeObj(self.l)
            self.l = new_loc[0]
            x, y = self.l[0], self.l[1]
            self.g.field[y][x] = self.s
            return True

        def attack(self):
            maybe_enemy = self.__check_forward()
            if maybe_enemy is None or maybe_enemy[1] not in "ED":
                return False

            loc, symb = maybe_enemy[0], maybe_enemy[1]
            if symb == "E":
                enemy = self.g.enemies[loc]
                if super().attack(enemy):
                    self.add_exp()
            else:
                super().attack(self.g.demonlord)
            return True

        def add_exp(self):
            self.exp += 1
            if self.exp == 3:
                self.a += 1
                self.exp = 0

        def use_item(self, item_s):
            # Check if item is in bag
            item = self.__find_item(item_s)
            if item is None:
                return False

            # Check for proper item usage
            if item == "H":
                if self.h < 3:
                    self.h = 3
                    return True
                else:
                    return False

            if item == "C":
                maybe_merch = self.__check_forward()
                if maybe_merch is None or maybe_merch[1] != "M":
                    return False

                merch_loc = maybe_merch[0]
                self.g.merchants[merch_loc].take_damage(1)
                return True

            if item == "K":
                maybe_door = self.__check_forward()
                if maybe_door is None or maybe_door[1] not in "-|":
                    return False

                door_loc = maybe_door[0]
                self.g.removeObj(door_loc)
                return True

    class Enemy(GameObject):
        def __init__(self, game, l=None):
            super().__init__(game, 1, 2, 0, l)

        def handle_death(self):
            del self.g.enemies[self.l]
            super().handle_death()

    class DemonLord(GameObject):
        def __init__(self, game, l=None):
            super().__init__(game, 10, 3, 0, l)

        def handle_death(self):
            self.g.demonlord = None
            super().handle_death()

    class Merchant(GameObject):
        def __init__(self, game, l=None):
            super().__init__(game, 3, 0, 0, l)

        def handle_death(self):
            del self.g.merchants[self.l]
            super().handle_death()

    class Game:
        def __init__(self, field):
            self.field = field
            self.player = None
            self.merchants = {}
            self.enemies = {}
            self.demonlord = None
            self.find_game_objects()

        def find_game_objects(self):
            for y in range(len(self.field)):
                for x in range(len(self.field[y])):
                    if self.field[y][x] == "M":
                        self.merchants[(x, y)] = Merchant(self, (x, y))
                    if self.field[y][x] == "E":
                        self.enemies[(x, y)] = Enemy(self, (x, y))
                    if self.field[y][x] == "D":
                        self.demonlord = DemonLord(self, (x, y))
                    if self.field[y][x] in "^><v":
                        self.player = Player(self, self.field[y][x], (x, y))

        def is_within_bounds(self, loc):
            return 0 <= loc[0] and loc[0] < len(self.field[0]) and 0 <= loc[1] and loc[1] < len(self.field)

        def check_enemies(self, loc):
            directions = ((0, 1), (0, -1), (1, 0), (-1, 0))
            for d in directions:

                new_loc = (loc[0] + d[0], loc[1] + d[1])
                if self.is_within_bounds(new_loc):

                    if new_loc in self.enemies:
                        self.enemies[new_loc].attack(self.player)

                    elif self.demonlord and new_loc == self.demonlord.l:
                        self.demonlord.attack(self.player)

                if self.player is None:
                    return False
            return True

        def removeObj(self, loc):
            x, y = loc[0], loc[1]
            self.field[y][x] = " "

        def run_game(self, action_list):
            for action in action_list:
                prev_loc = self.player.l
                # The player acts
                success = True
                if action in "^><v":
                    self.player.rotate(action)
                elif action == "A":
                    success = self.player.attack()
                elif action in "CKH":
                    success = self.player.use_item(action)
                else:
                    success = self.player.move()
                if not success:
                    # Unsuccessful act? Return None
                    return None

                # Enemies attack, if there are any
                is_alive = self.check_enemies(prev_loc)
                if not is_alive:
                    return None

            return (self.field, self.player.h, self.player.a, self.player.d, sorted(self.player.b))

    game = Game(game)
    return game.run_game(actions)

only_show_wrong()

def rpg(field: List[List[str]], actions: List[str]) -> Tuple[List[List[str]], int, int, int, List[str]]:
    print(field, actions)
    width = len(field[0])
    height = len(field)
    player = None
    lords = 0
    merchants = {}
    enemies = {}
    for y, row in enumerate(field):
        for x, c in enumerate(row):
            if c in dirchr: 
                if player is None:
                    player = x, y
                    direction = dirchr.index(c)
                    field[y][x] = ' '
                else:
                    raise ValueError
            elif c == 'D':
                if lords: raise ValueError
                enemies[x,y] = 10
                lords += 1
            elif c == 'E':
                enemies[x,y] = 1
            elif c == 'M':
                merchants[(x,y)] = 3
    if not player: raise ValueError

    health = 3
    attack = 1
    defense = 1
    bag = []
    killed = 0
    x, y = player

    for a in actions:
        if a == 'F':
            health -= check_enemy_attack(field, width, height, x, y, defense)
            x += dx[direction]
            y += dy[direction]
            if x < 0 or x >= width or y < 0 or y >= height or field[y][x] not in ' CKHSX':
                return None
            obj = field[y][x]
            field[y][x] = ' '
            if obj in 'CKH': bag.append(obj)
            elif obj == 'S': defense += 1
            elif obj == 'X': attack += 1
        elif a in dirchr:
            direction = dirchr.index(a)
            health -= check_enemy_attack(field, width, height, x, y, defense)
        elif a in 'ACKH':
            if a in 'CKH':
                if a not in bag: return None
                bag.remove(a)
            tx, ty = x+dx[direction], y+dy[direction]
            if tx < 0 or tx >= width or ty < 0 or ty >= width: return None
            target = field[ty][tx]
            if a in valid_target and target not in valid_target[a]: return None
            if a == 'A':
                enemies[tx,ty] -= attack
                if enemies[tx,ty] <= 0:
                    field[ty][tx] = ' '
                    killed += 1
                    if killed == 3:
                        attack += 1
                        killed = 0
                health -= check_enemy_attack(field, width, height, x, y, defense)
            elif a == 'C':
                merchants[tx,ty] -= 1
                if not merchants[tx,ty]: field[ty][tx] = ' '
            elif a == 'K':
                field[ty][tx] = ' '
            elif a == 'H':
                health = 3
                health -= check_enemy_attack(field, width, height, x, y, defense)
        if health <= 0: return None

    field[y][x] = dirchr[direction]
    return field, health, attack, defense, sorted(bag)

dirchr = '^>v<'
dx = 0, 1, 0, -1
dy = -1, 0, 1, 0

valid_target = dict(A='DE', C='M', K='-|')
enemy_attack = dict(D=3, E=2)

def check_enemy_attack(field, width, height, px, py, defense):
    damage = 0
    for x, y in ((px,py-1), (px+1,py), (px,py+1), (px-1,py)):
        if x >= 0 and x < width and y >=0 and y < height and field[y][x] in 'DE':
            attack = enemy_attack[field[y][x]]
            if attack > defense:
                damage += attack - defense
    return damage
only_show_wrong()

player = ['^', 'v', '<', '>']
items = [' ', 'K', 'C', 'H', 'S', 'X']

def rpg(field: List[List[str]], actions: List[str]) -> Tuple[List[List[str]], int, int, int, List[str]]:
    health = 3
    attack = 1
    defense = 1
    killed = 0
    bag = []
    demon_health = 10
    merchant_greed = 3
    for action in actions:
        cell, x, y, x2, y2 = find_player(field)
        if action == 'F':
            if cell not in items:
                return None
            field[y2][x2] = field[y][x]
            field[y][x] = ' '
            if cell == ' ':
                pass
            elif cell == 'X':
                attack += 1
            elif cell == 'S':
                defense += 1
            else:
                bag.append(cell)
        elif action in player:
            field[y][x] = action
        elif action == 'K':
            if cell not in ['-', '|'] or 'K' not in bag:
                return None
            field[y2][x2] = ' '
            bag.remove('K')
        elif action == 'A':
            if cell not in ['E', 'D']:
                return None
            enemy_health = (demon_health if cell == 'D' else 1) - attack
            if enemy_health <= 0:
                field[y2][x2] = ' '
                if cell == 'E':
                    killed += 1
                    if killed == 3:
                        killed = 0
                        attack += 1
            else:
                demon_health = enemy_health
        elif action == 'C':
            if cell != 'M' or 'C' not in bag:
                return None
            merchant_greed -= 1
            bag.remove('C')
            if merchant_greed == 0:
                field[y2][x2] = ' '
        elif action == 'H':
            if health == 3 or 'H' not in bag:
                return None
            health = 3
            bag.remove('H')
        health = check_enemy_near(field, x, y, health, defense)
        if health <= 0:
            return None
    bag.sort()
    return field, health, attack, defense, bag

def check_enemy_near(field, x, y, health, defense):
    w = len(field[0])
    h = len(field)
    for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
        x2, y2 = x + dx, y + dy 
        if 0 <= x2 < w and 0 <= y2 < h:
            if field[y2][x2] == 'E':
                health -= max(0, 2 - defense)
            elif field[y2][x2] == 'D':
                health -= max(0, 3 - defense)
    return health

def get_direction (player):
    if player == '^':
        return 0, -1
    elif player == 'v':
        return 0, 1
    elif player == '<':
        return -1, 0
    elif player == '>':
        return 1, 0

def find_player (field):
    w = len(field[0])
    h = len(field)
    for y in range(h):
        for x in range(w):
            if field[y][x] in player:
                dx, dy = get_direction(field[y][x])
                x2, y2 = x + dx, y + dy
                return field[y2][x2] if 0 <= x2 < w and 0 <= y2 < h else None, x, y, x2, y2
    return None, 0, 0, 0, 0

# only_show_wrong()
dir_dict = {'^': (-1, 0), 'v': (1, 0), '<': (0, -1), '>': (0, 1)}


def rpg(field, actions):
    width, height = len(field[0]), len(field)
    field, actions = [list(line) for line in field], list(actions)

    found = False
    for row in range(height):
        if found: break
        for col in range(width):
            if field[row][col] in dir_dict:
                player_loc = (row, col)
                player_dir = field[row][col]
                target_loc = (row + dir_dict[player_dir][0], col + dir_dict[player_dir][1])
                found = True
                break

    hlh, atk, des, bag = 3, 1, 1, []
    mer_dict = {}
    enemy_count, dl_hlh = 0, 10

    for action in actions:
        if action not in dir_dict:
            if target_loc[0] < 0 or target_loc[0] >= height or target_loc[1] < 0 or target_loc[1] >= width: return None
            target_tile = field[target_loc[0]][target_loc[1]]

        if action == 'C':
            if target_tile != 'M': return None
            if action not in bag: return None
            bag.remove(action)
            if target_loc not in mer_dict: mer_dict[target_loc] = 2
            else:
                mer_dict[target_loc] = mer_dict[target_loc] - 1
                if mer_dict[target_loc] == 0: field[target_loc[0]][target_loc[1]] = ' '

        elif action == 'K':
            if target_tile not in '-|': return None
            if action not in bag: return None
            bag.remove(action)
            field[target_loc[0]][target_loc[1]] = ' '

        else:
            # enemy will attack!
            if action == 'F':
                if target_tile in ['C', 'K', 'H']:
                    bag.append(target_tile)
                    bag.sort()
                elif target_tile == 'S': des = des + 1
                elif target_tile == 'X': atk = atk + 1
                elif target_tile != ' ': return None
                field[target_loc[0]][target_loc[1]] = player_dir
                field[player_loc[0]][player_loc[1]] = ' '
            elif action in dir_dict:
                player_dir = action
                field[player_loc[0]][player_loc[1]] = action
                target_loc = (player_loc[0] + dir_dict[player_dir][0], player_loc[1] + dir_dict[player_dir][1])
            elif action == 'A':
                if target_tile == 'E':
                    field[target_loc[0]][target_loc[1]] = ' '
                    enemy_count = enemy_count + 1
                    if enemy_count == 3:
                        enemy_count = 0
                        atk = atk + 1
                elif target_tile == 'D':
                    dl_hlh = dl_hlh - atk
                    if dl_hlh <= 0: field[target_loc[0]][target_loc[1]] = ' '
                else: return None
            elif action == 'H':
                if action not in bag: return None
                hlh = 3
                bag.remove(action)

            # check if enemy exists
            for d in dir_dict:
                check_loc = (player_loc[0] + dir_dict[d][0], player_loc[1] + dir_dict[d][1])
                if check_loc[0] >= 0 and check_loc[0] < height and check_loc[1] >= 0 and check_loc[1] < width:
                    if field[check_loc[0]][check_loc[1]] == 'E': hlh = hlh - max(0, 2 - des)
                    elif field[check_loc[0]][check_loc[1]] == 'D': hlh = hlh - max(0, 3 - des)
                    if hlh <= 0: return None

            if action == 'F':
                player_loc = target_loc
                target_loc = (target_loc[0] + dir_dict[player_dir][0], target_loc[1] + dir_dict[player_dir][1])
    return tuple([field, hlh, atk, des, bag])
D = {'^': (-1, 0), 'v': (1, 0), '<': (0, -1), '>': (0, 1)}

def rpg(field, actions):
    g = [list(x) for x in field]
    HP, Atk, Def, Boss, En = 3, 1, 1, 10, 0
    BC = BK = BH = 0
    px = py = 0
    mechanics = {}

    for i in range(len(g)):
        for j in range(len(g[i])):
            if g[i][j] in '^v<>':
                px, py = i, j
            if g[i][j] == 'M':
                mechanics[(i, j)] = 3

    def move(p, q, c):
        nonlocal BK, BC, BH, Atk, Def
        k = g[q[0]][q[1]]
        if k == 'K':
            BK += 1
        elif k == 'C':
            BC += 1
        elif k == 'X':
            Atk += 1
        elif k == 'S':
            Def += 1
        elif k == 'H':
            BH += 1
        g[q[0]][q[1]], g[p[0]][p[1]] = c, ' '

    def bounds(x, y):
        return 0 <= x < len(g) and 0 <= y < len(g[x])

    def attack(i, j, e):
        nonlocal HP, Def
        k = g[i][j]
        nx, ny = i+D[k][0], j+D[k][1]
        for x, y in ((i+1, j), (i-1, j), (i, j+1), (i, j-1)):
            if bounds(x, y) and g[x][y] in 'DE':
                if not (e == 'A' and x == nx and y == ny):
                    HP -= max({'D': 3, 'E': 2}[g[x][y]] - Def, 0)

    for e in actions:
        if e == 'H':
            if BH <= 0 or HP == 3:
                return None
            BH, HP = BH-1, 3
        attack(px, py, e)
        k = g[px][py]
        if e == 'F':
            ox, oy = px, py
            px, py = px+D[k][0], py+D[k][1]
            if not bounds(px, py) or g[px][py] not in ' CKXSH':
                return None
            move((ox, oy), (px, py), k)
        if e in '^v<>':
            g[px][py] = e
            
        nx, ny = px + D[k][0], py + D[k][1]
        if e == 'K':
            if not bounds(nx, ny):
                return None
            if g[nx][ny] not in '-|' or BK <= 0:
                return None
            g[nx][ny] = ' '
            BK -= 1
        elif e == 'A':
            if not bounds(nx, ny) or g[nx][ny] not in 'DE':
                return None
            if g[nx][ny] == 'E' and Atk >= 1:
                En += 1
                if En % 3 == 0:
                    Atk += 1
                g[nx][ny] = ' '
            elif g[nx][ny] == 'D':
                Boss -= Atk
                if Boss <= 0:
                    g[nx][ny] = ' '
                    break
                HP -= max({'D': 3, 'E': 2}[g[nx][ny]] - Def, 0)
        elif e == 'C':
            if not bounds(nx, ny) or g[nx][ny] != 'M' or BC <= 0:
                return None
            mechanics[(nx, ny)] -= 1
            if mechanics[(nx, ny)] == 0:
                g[nx][ny] = ' '
            BC -= 1
    if HP <= 0:
        return None
    Bag = ['C'] * BC + ['H'] * BH + ['K'] * BK
    return g, HP, Atk, Def, Bag
only_show_wrong()

def rpg(field, actions):
    newfield = [[val for val in row] for row in field]
    playerpos = [0,0,0] # [x,y,dir]
    playerstats = [3, 1, 1, 0]  # Health, attack, defense, enemies killed
    playerbag = []
    myactions = actions

    ## Board stuff    
    demonlord_hp = [10]
    merchants = []  # [[posx,posy,coinsgiven][posx,posy,coinsgiven]] this list holds a 'reference' to all merchants

    ## Finding player position
    for row in newfield:
        for val in row:
            if val == '^':
                playerpos[0] = row.index(val)
                playerpos[1] = newfield.index(row)
                playerpos[2] = 0
            elif val == '>':
                playerpos[0] = row.index(val)
                playerpos[1] = newfield.index(row)
                playerpos[2] = 1
            elif val == 'v':
                playerpos[0] = row.index(val)
                playerpos[1] = newfield.index(row)
                playerpos[2] = 2
            elif val == '<':
                playerpos[0] = row.index(val)
                playerpos[1] = newfield.index(row)
                playerpos[2] = 3
            elif val == 'M':
                merchants.append([row.index(val), newfield.index(row), 3])
    ## Taking the actions
    for action in myactions:
        ## Save players last known position for damage calculation
        lastplayerpos = playerpos[:]
        legalmove = True
        if(action == 'F'):
            legalmove = forward(newfield, playerpos, playerbag, playerstats)
        elif(action == '^'):
            playerpos[2] = 0
            newfield[playerpos[1]][playerpos[0]] = getplayerstring(playerpos)
        elif (action =='>'):
            playerpos[2] = 1
            newfield[playerpos[1]][playerpos[0]] = getplayerstring(playerpos)
        elif (action =='v'):
            playerpos[2] = 2
            newfield[playerpos[1]][playerpos[0]] = getplayerstring(playerpos)
        elif (action =='<'):
            playerpos[2] = 3
            newfield[playerpos[1]][playerpos[0]] = getplayerstring(playerpos)
        elif (action =='A'):
            legalmove = attack(newfield, playerpos, playerstats, demonlord_hp)
            if(legalmove == 'win'):
                print('we won!')
                break
        elif (action =='C'):
            legalmove = coin(newfield, playerpos, playerbag, merchants)
        elif (action =='K'):
            legalmove = key(newfield, playerpos, playerbag)
        elif (action =='H'):
            legalmove = health(playerbag, playerstats)

        if(legalmove == False):
            print("Encountered invalid!")
            print("last action: " + action)
            return None
        
        ##Check for enemy damage and shit
        enemies = getajanba(newfield, lastplayerpos)
        if(len(enemies) > 0):
            print("Printing enemies!")
            print('last action: ' + action)
            for enemy in enemies:
                print(enemy)
        for enemy in enemies:
            damagetodeal = max(0, 2 - playerstats[2]) if enemy == 'E' else max(0, 3 - playerstats[2])
            playerstats[0] -= damagetodeal
            if(playerstats[0] <= 0):
                return None
    
    ## Actions finished, give back output
    sortedbag = playerbag if len(playerbag) <= 1 else sorted(playerbag)
    return (newfield, playerstats[0], playerstats[1], playerstats[2], sortedbag)
    
def forward(field, playerpos, bag, playerstats):
    #find where the in front of the player is
    ## check_movement returns [obj, posx, posy]
    infront = check_front(field, playerpos)
    
    if(infront == False):    ## if oob
        return False

    obj = infront[0]
    posx = infront[1]
    posy = infront[2]
    if(obj == '#' or obj == 'M' or obj == '-' or obj == '|' or obj == 'E' or obj == 'D'):
        return False
    elif(obj == 'C' or obj == 'K' or obj == 'H'):   ## Time to check for objects(inventory)
        bag.append(obj)
        print("obtained: " + obj + "   bag is now: ")
        print(bag)
        field[posy][posx] = ' '
    elif(obj == 'S'):
        playerstats[2] += 1
        field[posy][posx] = ' '
    elif(obj == 'X'):
        playerstats[1] += 1
        field[posy][posx] = ' '
    ## Update player pos
    field[playerpos[1]][playerpos[0]] = ' '
    field[posy][posx] = getplayerstring(playerpos)
    playerpos[0] = posx
    playerpos[1] = posy
    return True

def attack(field, playerpos, playerstats, demonlord):
    infront = check_front(field, playerpos)
    if(infront == False):
        return False
    enemy = infront[0]
    posx = infront[1]
    posy = infront[2]

    if enemy == 'E':
        field[posy][posx] = ' '
        playerstats[3] += 1
        if playerstats[3] >= 3:
            playerstats[1] += 1
            playerstats[3] = 0
        return True
    elif enemy == 'D':
        demonlord[0] -= playerstats[1]
        if demonlord[0] <= 0:
            field[posy][posx] = ' '
            return 'win'
    else: 
        return False

def coin(field, playerpos, playerbag, merchants):
    ## Do we have coins?
    if 'C' not in playerbag:
        return False

    ## Is a merchant in front of us
    infront = check_front(field, playerpos)
    if(infront == False):
        return False
    obj = infront[0]
    posx = infront[1]
    posy = infront[2]
    if obj != 'M':
        print('No merchant in front!')
        return False
    ## Find specific merchant in array
    for merchant in merchants:
        if merchant[0] == posx and merchant[1] == posy:
            playerbag.remove('C')
            merchant[2] -= 1
            print('giving coin to merchant')
            if merchant[2] <= 0:
                field[posy][posx] = ' '
                print('merchant should b gone')
            break
    return True

def key(field, playerpos, playerbag):
    ## Do we have keys
    if 'K' not in playerbag:
        return False
    
    ## Is a door in front of us    
    infront = check_front(field, playerpos)
    if(infront == False):
        return False
    obj = infront[0]
    posx = infront[1]
    posy = infront[2]
    if obj != '-' and obj != '|':
        return False
    field[posy][posx] = ' '
    playerbag.remove('K')
    return True

def health(playerbag, playerstats):
    if playerstats[0] >= 3:
        return False
    if 'H' not in playerbag:
        return False
    playerstats[0] = 3
    playerbag.remove('H')
    return True

def check_front(field, playerpos):
    #checking direction of movement and get square
    posx = playerpos[0]
    posy = playerpos[1]
    posdir = playerpos[2]
    if(posdir == 0):
        posx += 0
        posy -= 1
    elif(posdir == 1):
        posx += 1
        posy -= 0
    elif(posdir == 2):
        posx += 0
        posy += 1
    elif(posdir == 3):
        posx -= 1
        posy -= 0
    
    #Check for OOB
    if (posx < 0 or posx >= len(field[0])) or (posy < 0 or posy >= len(field)):
        return False
    #Check for Objects
    obj = field[posy][posx]
    return [obj, posx, posy]

def getplayerstring(playerpos):
    if(playerpos[2] == 0):
        return '^'
    elif(playerpos[2] == 1):
        return '>'
    elif(playerpos[2] == 2):
        return 'v'
    elif(playerpos[2] == 3):
        return '<'
    
def getajanba(field, playerpos):
    enemylist = []
    tocheck = [[playerpos[0]+1, playerpos[1]],[playerpos[0]-1, playerpos[1]],[playerpos[0], playerpos[1]+1],[playerpos[0], playerpos[1]-1]]
    for check in tocheck:
        posx = check[0]
        posy = check[1]
        if (posx < 0 or posx >= len(field[0])) or (posy < 0 or posy >= len(field)):
            continue
        
        obj = field[posy][posx]
        if obj == 'E' or obj == 'D':
            enemylist.append(obj) 
    return enemylist
only_show_wrong()
playerDirections = {
    "^": (0, -1),
    "<": (-1, 0),
    ">": (1, 0),
    "v": (0, 1)
}
items = ("X", "K", "H", "S", "C")

def rpg(field: List[List[str]], actions: List[str]) -> Tuple[List[List[str]], int, int, int, List[str]]:
    player = {
        "x": 0,
        "y": 0,
        "direction": (0, 0),
        "char": "",
        "health": 3,
        "attack": 1,
        "defense": 1,
        "inventory": [],
        "levelCounter": 0,
        "coinsCounter": 0
    }
    demonLord = 10
    
    for y, str in enumerate(field):
        for x, char in enumerate(str):
            if char in playerDirections:
                player["x"] = x
                player["y"] = y
                player["direction"] = playerDirections[char]
                player["char"] = char
                break
        else: continue
        break
    
    for action in actions:
        if action in playerDirections:
            player["direction"] = playerDirections[action]
            player["char"] = action
            
        elif action == "F":
            enemiesClose = []
            for dir in playerDirections.values():
                adjacent = ""
                try:
                    positions = (max(0, player["y"] + dir[1]), max(0, player["x"] + dir[0]))
                    adjacent = field[positions[0]][positions[1]]
                except:
                    adjacent = ""
                if adjacent == "E" or adjacent == "D":
                    player["health"] -= max(0, (2 if adjacent == "E" else 3) - player["defense"])
        
            field[player["y"]][player["x"]] = " "
            player["x"] += player["direction"][0]
            player["y"] += player["direction"][1]
            
            if(player["x"] < 0 or player["y"] < 0 or
               player["y"] >= len(field) or player["x"] >= len(field[player["y"]])): return None
            nextTile = field[player["y"]][player["x"]]
            if(not nextTile in items and nextTile != " "): return None
            if(nextTile in items):
                if nextTile == "X": player["attack"] += 1
                elif nextTile == "S": player["defense"] += 1
                else: player["inventory"].append(nextTile)     
            
        elif action in player["inventory"]:
            if action == "H":
                if player["health"]< 3: player["health"] = 3
                else: return None
            elif action == "K":
                nextTilePos = (player["y"] + player["direction"][1], player["x"] + player["direction"][0])
                if(nextTilePos[1] < 0 or nextTilePos[0] < 0 or
                   nextTilePos[0] >= len(field) or nextTilePos[1] >= len(field[nextTilePos[0]])): return None
                nextTile = field[nextTilePos[0]][nextTilePos[1]]
                if(nextTile == "-" or nextTile == "|"): field[nextTilePos[0]][nextTilePos[1]] = " "
                else: return None
            else:
                nextTilePos = (player["y"] + player["direction"][1], player["x"] + player["direction"][0])
                if(nextTilePos[1] < 0 or nextTilePos[0] < 0 or
                   nextTilePos[0] >= len(field) or nextTilePos[1] >= len(field[nextTilePos[0]])): return None
                if field[nextTilePos[0]][nextTilePos[1]] == "M":
                    player["coinsCounter"] += 1
                    if player["coinsCounter"] >= 3:
                        field[nextTilePos[0]][nextTilePos[1]] = " "
                else: return None
            player["inventory"].remove(action)
        
        elif action == "A":
            nextTilePos = (player["y"] + player["direction"][1], player["x"] + player["direction"][0])
            nextTile = field[nextTilePos[0]][nextTilePos[1]]
            if nextTile == "E":
                field[nextTilePos[0]][nextTilePos[1]] = " "
                player["levelCounter"] += 1
                if player["levelCounter"] >= 3:
                    player["levelCounter"] = 0
                    player["attack"] += 1
            elif nextTile == "D":
                demonLord -= player["attack"]
                if demonLord <= 0:
                    field[nextTilePos[0]][nextTilePos[1]] = " "
            else: return None
        
        else: return None
        
        if action == "A" or action == "H" or action in playerDirections:
            for dir in playerDirections.values():
                adjacent = ""
                try:
                    positions = (max(0, player["y"] + dir[1]), max(0, player["x"] + dir[0]))
                    adjacent = field[positions[0]][positions[1]]
                except:
                    adjacent = "X"
                if adjacent == "E" or adjacent == "D":
                    player["health"] -= max(0, (2 if adjacent == "E" else 3) - player["defense"])
        
        if player["health"] <= 0: return None
    
    field[player["y"]][player["x"]] = player["char"]
    return (field, player["health"], player["attack"], player["defense"], sorted(player["inventory"]))
