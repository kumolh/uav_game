from gym import Env
from gym import spaces
import numpy as np
from settings import *
from tile import Tile
from player import Player
from ai_player import *
from player_com import PlayerCom
from zombie import Zombie

class UAV_Env(Env):
    def __init__(self):
        super(UAV_Env, self).__init__()
        self.action_space = spaces.Discrete(15)
        self.observation_space = spaces.Box(low = -100, high = 2000, shape=(6,), dtype=np.float32)
        self.initialize()

    def step(self, action):
        self.take_action(action)
        state = self.get_state()
        reward, done = self.get_reward()
        return state, reward, done
    
    def take_action(self, action):
        move = self.player.input()
        if move >= 0:
            self.player.move(self.player.speed)
        pass

    def get_state(self):
        state = [0.0] * 6
        for i in range(4):
            rad = np.deg2rad(90 * i + self.player.front)
            dis, _, vx, vy = self.ray_casting(self.player, rad)
            state[i] = dis
            # [x, y] = [player.rect.centerx - self.offset.x, player.rect.centery - self.offset.y]
            # pygame.draw.line(self.display_surface, 'blue', [x, y], [vx, vy]) 
        # pygame.draw.circle(self.display_surface, 'black', [zombie.rect.centerx - self.offset.x, zombie.rect.centery - self.offset.y], 1.0)
        _, _, pos, distance = self.zombie.position(self.player)
        state[4] = distance
        state[5] = pos
        self.player.state = state
        return np.array(state, dtype=np.float32)
    
    def get_reward(self):
        pass

    def initialize(self):
        self.goal = -1
        self.zombies = []
        self.create_map()
        self.zombie = self.zombies[0]
        self.zombie.set_target()
        self.player.add_target(self.zombie)

    def create_map(self):
        for row_index,row in enumerate(WORLD_MAP):
            for col_index, col in enumerate(row):
                x = col_index * TILESIZE
                y = row_index * TILESIZE
                if col == 'x':
                    Tile((x,y),[self.visible_sprites])
                if col == 'p':
                    # self.player = PlayerCom((x, y), [self.visible_sprites], self.obstacle_sprites)
                    # self.player = AI_Player((x, y), [self.visible_sprites], self.obstacle_sprites)
                    self.player = Player((x, y), [self.visible_sprites])
                if '0' <= col <= '9':
                    type = random.randint(0, 1)
                    zombie = Zombie((x, y), [self.visible_sprites], type)
                    self.zombies.append(zombie)
        

    def render(self):
        pass

    def reset(self):
        # called at the initialization
        pass

    def ray_casting(self, player, rad):
        # return the distance that the ray can go
        xo = 0
        yo = -TILESIZE if np.sin(rad) > 0 else TILESIZE 
        # check vertical
        dof = 0
        disV = 10000
        tan = np.tan(rad)
        if np.cos(rad) > 0.001: # looking right
            rx = int(player.rect.centerx // TILESIZE) * TILESIZE + TILESIZE 
            ry = (player.rect.centerx - rx) * tan + player.rect.centery 
            xo = TILESIZE
            yo = - xo * tan
        elif np.cos(rad) < -0.001:
            rx = int(player.rect.centerx // TILESIZE) * TILESIZE 
            ry = (player.rect.centerx - rx) * tan + player.rect.centery 
            xo = -TILESIZE
            yo = - xo * tan
        else:
            dof = DOF
            yo = -TILESIZE if np.sin(rad) > 0 else TILESIZE 
            rx = player.rect.centerx 
            ry = player.rect.centery
            
            while dof < DOF:
                mx = int(rx // TILESIZE) if np.cos(rad) > 0.001 else int(rx // TILESIZE) - 1
            my = int(ry // TILESIZE) #if np.cos(rad) > 0.001 else int(ry // TILESIZE) - 1
            if 0 <= mx < MAPX and 0 <= my < MAPY and WORLD_MAP[my][mx] == 'x':
                dof = DOF
                disV = np.cos(rad) * (rx - player.rect.centerx) - np.sin(rad) * (ry - player.rect.centery)
            else:
                rx += xo
                ry += yo
                dof += 1
        vx, vy = rx - self.offset.x, ry - self.offset.y
        offset = ry
        # check horizontal
        dof = 0
        disH = 10000
        tan = 1.0/ tan if tan != 0 else 1 << 20
        if np.sin(rad) > 0.001: # looking up
            ry = int(player.rect.centery / TILESIZE) * TILESIZE
            rx = (player.rect.centery - ry) * tan + player.rect.centerx 
            yo = - TILESIZE
            xo = - yo * tan
        elif np.sin(rad) < -0.001:
            ry = int(player.rect.centery / TILESIZE) * TILESIZE + TILESIZE 
            rx = (player.rect.centery - ry) * tan + player.rect.centerx 
            yo = TILESIZE
            xo = - yo * tan
        else:
            dof = DOF
            yo = -TILESIZE if np.sin(rad) > 0 else TILESIZE 
            rx = player.rect.centerx
            ry = player.rect.centery
            
        while dof < DOF:
            mx = int(rx / TILESIZE) #if np.sin(rad) < -0.001 else int(rx / TILESIZE) + 1
            my = int(ry / TILESIZE) if np.sin(rad) < -0.001 else int(ry / TILESIZE) - 1
            
            if 0 <= mx < MAPX and 0 <= my < MAPY and WORLD_MAP[my][mx] == 'x':
                # print(mx, my)
                dof = DOF
                disH = np.cos(rad) * (rx - player.rect.centerx) - np.sin(rad) * (ry - player.rect.centery)
            else:
                rx += xo
                ry += yo
                dof += 1
                
        if disH < disV:
            vx, vy = rx - self.offset.x, ry - self.offset.y
            offset = rx
        if vx > WIDTH:
            vy += (vx - WIDTH) / tan
            vx = WIDTH
            
        dis = min(disH, disV)
        return dis, offset, vx, vy