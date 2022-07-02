from gym import Env
from gym import spaces
import numpy as np
from settings import *
from tile import Tile
from player import Player
from ai_player import *
from player_com import PlayerCom
from zombie import Zombie
import sys
from stable_baselines3.common.env_checker import check_env

class UAV_Env(Env):
    def __init__(self):
        super(UAV_Env, self).__init__()
        self.world = pygame.display.set_mode((WIDTH * 2, HEIGTH))
        self.clock = pygame.time.Clock()
        self.create_map()
        # self.action_space = spaces.MultiDiscrete([5, 3])
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low = -100, high = 2000, shape=(6,), dtype=np.float32)
        # self.initialize()

    def step(self, action):
        # [translate, rotate] = action
        # if translate == 1:
        #     self.player.direction.y = -1
        # elif translate == 2:
        #     self.player.direction.y = 1
        # else:
        #     self.player.direction.y = 0

        # if translate == 3:
        #     self.player.direction.x = 1
        # elif translate == 4:
        #     self.player.direction.x = -1
        # else:
        #     self.player.direction.x = 0
        # rotate = action
        # self.player.direction.y = -1
        # if rotate == 1:
        #     self.player.front -= 2
        #     self.player.front %= 360
        # elif rotate == 2:
        #     self.player.front += 2
        #     self.player.front %= 360
        # self.player.move(self.player.speed)
        for zombie in self.zombies:
            zombie.move()
        vert, rotate, collision = self.get_circle_move(128)
        if not collision:
            self.player.direction.y = vert
            self.player.direction.x = 1
            self.player.front += rotate
            self.player.front %= 360
            self.player.move(self.player.speed)

        
        state = self.get_state()
        reward, done = self.get_reward(state)
        return state, reward, done, {}
    
    # def take_action(self, action):
    #     move = self.player.input()
    #     self.player.move(self.player.speed)

    def get_state(self):
        state = [0.0] * 6
        for i in range(4):
            rad = np.deg2rad(90 * i + self.player.front)
            dis, _, vx, vy = self.ray_casting(self.player, rad)
            state[i] = dis
        _, _, pos, distance = self.zombie.position(self.player)
        state[4] = distance
        state[5] = pos
        self.player.state = state
        return np.array(state, dtype=np.float32)
    
    def get_reward(self, state):
        def linear(location):
            delta = abs(location - 30)
            return 0.12 * (30 - delta) / 30 if delta < 30 else 0
        def gaussian(distance):
            return 100 * np.exp(- (distance - 64) ** 2)
        reward = linear(state[5]) - linear(self.last_state[5]) + gaussian(state[4]) - gaussian(self.last_state[4])
        done = False
        self.player.rewards += reward
        self.last_state = state
        if any(state[i] < 60 for i in range(5)):
            done = True
            reward = -100

        if TILESIZE <= state[4] < 2 * TILESIZE and 0 < state[5] < 60:
            done = True
            reward = 100
        return reward, done

    def get_circle_move(self, opt_dis):
        vertical = 0
        horizontal = 1 # or -1: deterministic
        rotate = 0
        delta_dis = opt_dis * 1.0
        colli = False
        for v in range(-1, 2):
            for r in range(-1, 2):
                collision, dis, theta = self.pseudo_move(self.player, v, horizontal, r)
                if abs(opt_dis - dis) < delta_dis and 20 < theta < 40:
                    vertical, rotate, colli = v, r, collision
        return vertical, rotate, colli
    
    def pseudo_move(self, player, vert, hori, rotate):
        dire = (player.front + rotate * 2) % 360
        rad = np.deg2rad(-dire)
        fx = player.fx - player.speed * np.sin(rad) * vert + player.speed * np.cos(rad) * hori
        fy = player.fy + player.speed * np.cos(rad) * vert + player.speed * np.sin(rad) * hori
        collision = True
        ret_x = ret_y = 0
        if not(WORLD_MAP[int(fy - TILESIZE/ 2) // TILESIZE ][int(fx) // TILESIZE] == 'x' or \
                WORLD_MAP[int(fy + TILESIZE/2) // TILESIZE ][int(fx) // TILESIZE] == 'x' or \
                WORLD_MAP[int(fy) // TILESIZE][int(fx - TILESIZE / 2) // TILESIZE] == 'x' or \
                WORLD_MAP[int(fy) // TILESIZE][int(fx + TILESIZE / 2) // TILESIZE] == 'x'):
                ret_x = int(fx)
                ret_y = int(fy)
                collision = False
        elif not(WORLD_MAP[int(player.hitbox.top) // TILESIZE ][int(fx) // TILESIZE] == 'x' or \
                WORLD_MAP[int(player.hitbox.bottom + 1) // TILESIZE][int(fx) // TILESIZE] == 'x' or \
                WORLD_MAP[int(player.hitbox.y) // TILESIZE ][int(fx - TILESIZE / 2) // TILESIZE] == 'x' or \
                WORLD_MAP[int(player.hitbox.y) // TILESIZE ][int(fx + TILESIZE / 2) // TILESIZE] == 'x'):
                ret_x = int(fx)
                fy = player.hitbox.centery
        elif not(WORLD_MAP[int(fy - TILESIZE / 2) // TILESIZE ][int(player.hitbox.x) // TILESIZE] == 'x' or \
                WORLD_MAP[int(fy +TILESIZE / 2) // TILESIZE ][int(player.hitbox.x) // TILESIZE] == 'x' or \
                WORLD_MAP[int(fy) // TILESIZE][int(player.hitbox.left) // TILESIZE] == 'x' or \
                WORLD_MAP[int(fy) // TILESIZE][int(player.hitbox.right + 1) // TILESIZE] == 'x'):
                ret_y = int(fy)
                fx = player.hitbox.centerx
        else:
            fx = player.hitbox.centerx
            fy = player.hitbox.centery
        # calculate the relative position
        relative_pos = [self.zombie.rect.centerx - fx, self.zombie.rect.centery - fy]
        angle = np.arctan2(-relative_pos[1], relative_pos[0])
        if angle < 0: angle += 2 * np.pi
        angle = (np.rad2deg(angle) - 90) % 360
        dis = np.sqrt(sum(x ** 2 for x in relative_pos)) # relative distance
        theta = (dire + 30 - angle) % 360 # view position
        return collision, dis, theta

    def initialize(self):
        self.running = True
        self.create_map()
        # self.replace_player()

    def create_map(self):
        self.goal = -1
        self.zombies = []
        self.visible_sprites = pygame.sprite.Group()
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
        self.zombie = self.zombies[0]
        self.zombie.set_target()
        self.player.add_target(self.zombie)
        self.offset = pygame.math.Vector2()
        self.last_state = self.get_state()
        self.textures = {'x': pygame.image.load('img/1.png').convert(),
                         '2': pygame.image.load('img/2.png').convert(),
                         'S': pygame.image.load('img/night.png').convert()
                         }
        self.half_width = self.world.get_size()[0] // 4
        self.half_height = self.world.get_size()[1] // 2

    def replace_player(self):
        r = random.randint(3 * TILESIZE, (TILE_V - 4) * TILESIZE) 
        c = random.randint(3 * TILESIZE, (TILE_H - 4) * TILESIZE) 
        self.player.rect.update(r, c, TILESIZE, TILESIZE)
        # player.rect = player.init_rect.copy()
        self.player.fx = self.player.rect.centerx
        self.player.fy = self.player.rect.centery
        self.player.front = 0

    def render(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT or event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                pygame.quit()
                self.running = False
                sys.exit()
        self.world.fill('white')
        self.background(self.player.front)
        self.custom_draw(self.player)
        self.frontground()
        pygame.display.set_caption('FPS: ' + str(int(self.clock.get_fps())))
        pygame.display.update()
        self.clock.tick()

    def background(self, angle):
        # 2D walls and player:
        for sprite in self.visible_sprites:
            # textsurface = myfont.render(str(sprite.rect.bottom), False, (0, 0, 0))
            offset_pos = sprite.rect.topleft - self.offset
            if 0 <= sprite.rect.right - self.offset.x and sprite.rect.left - self.offset.x <= WIDTH:
                pygame.display.get_surface().blit(sprite.image, offset_pos)
                # self.display_surface.blit(textsurface, offset_pos)

        # 3D background
        height, width = self.textures['S'].get_height(), self.textures['S'].get_width()
        offset = int(angle / 360 * width)
        sky = self.textures['S'].subsurface(offset, 0, width - offset, height // 2)
        pygame.display.get_surface().blit(sky, (WIDTH, 0))
        if width - offset < WIDTH:
            sky = self.textures['S'].subsurface(0, 0, WIDTH - (width - offset), height // 2)
            pygame.display.get_surface().blit(sky, (WIDTH + width - offset, 0))
        pygame.draw.rect(pygame.display.get_surface(), (100, 100, 100), (WIDTH, HEIGTH / 2, WIDTH, HEIGTH))

    def custom_draw(self, player):
        # getting the offset 
        self.offset.x = player.rect.centerx - self.half_width
        if self.offset.x < 0:
            self.offset.x = 0 
        elif player.rect.centerx > len(WORLD_MAP[0]) * TILESIZE - self.half_width:
            self.offset.x = len(WORLD_MAP[0]) * TILESIZE - WIDTH
        self.offset.y = player.rect.centery - self.half_height
        if self.offset.y < 0:
            self.offset.y = 0 
        elif player.rect.centery > len(WORLD_MAP) * TILESIZE - self.half_height:
            self.offset.y = len(WORLD_MAP) * TILESIZE - HEIGTH

        # rx, ry, vx, vy = 0.0, 0.0, 0.0, 0.0 
        rad = np.deg2rad(90 + player.front) - HALF_FOV
        for i in range(NUM_RAYS):          
            dis, offset, vx, vy = self.ray_casting(player, rad)
            [x, y] = [player.rect.centerx - self.offset.x, player.rect.centery - self.offset.y]
            if i == 0 or i == NUM_RAYS - 1:
                pygame.draw.line(self.world, 'black', [x, y], [vx, vy]) 
            dis *= np.cos((NUM_RAYS//2 - i) * DELTA_ANGLE)
            player.distances[i] = dis

            wall_height = min(HEIGTH, int(35000 / (dis + 0.001)))
            color = 255 / (1 + dis * dis * 0.0001)
            offset = int(offset) % TILESIZE

            height, width = self.textures['x'].get_height(), self.textures['x'].get_width()
            scale = width // TILESIZE
            wall_column = self.textures['x'].subsurface(offset * scale, 0, scale, height)
            wall_column = pygame.transform.scale(wall_column, (SCALE, wall_height))
            wall_pos = (2 * WIDTH - i * SCALE - 1, HALF_HEIGHT - wall_height // 2)
            
            self.world.blit(wall_column, wall_pos)
            rad += DELTA_ANGLE

    def frontground(self):
        for zombie in self.zombies:
            zombie.draw_sprites(self.player)

    def reset(self):
        self.initialize()
        return self.last_state

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

if __name__ == '__main__':
    game = UAV_Env()
    check_env(game)
    # pygame.init()
    # pygame.display.set_caption('UAV')
    # game.step(action=[])
    # while game.running:
    #     game.step(action=[])
    #     game.render()