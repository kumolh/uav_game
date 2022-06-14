import pygame
from settings import *
from tile import Tile
from player import Player
from ai_player import *
from player_com import PlayerCom
from debug import debug
import numpy as np
from zombie import Zombie
from Helper import plot

class Level:
    def __init__(self):

        # get the display surface 
        self.display_surface = pygame.display.get_surface()

        # sprite group setup
        self.visible_sprites = YSortCameraGroup()
        self.obstacle_sprites = pygame.sprite.Group()
        self.zombies = []
        # sprite setup
        self.create_map()

    def create_map(self):
        """

        """
        for row_index,row in enumerate(WORLD_MAP):
            for col_index, col in enumerate(row):
                x = col_index * TILESIZE
                y = row_index * TILESIZE
                if col == 'x':
                    Tile((x,y),[self.visible_sprites, self.obstacle_sprites])
                if col == 'p':
                    self.player = PlayerCom((x, y), [self.visible_sprites], self.obstacle_sprites)
                    # self.player = AI_Player((x, y), [self.visible_sprites], self.obstacle_sprites)
                    # self.player = Player((x, y), [self.visible_sprites], self.obstacle_sprites)
                if '0' <= col <= '9':
                    type = random.randint(0, 1)
                    zombie = Zombie((x, y), [self.visible_sprites], self.obstacle_sprites, type)
                    self.zombies.append(zombie)
        self.zombie = self.zombies[0]
        self.zombie.set_target()
        self.player.add_target(self.zombie)

    def run(self):
        # update and draw the game
        # drawing background
        self.visible_sprites.background(self.player.front)
        self.visible_sprites.custom_draw(self.player)
        state = self.visible_sprites.get_state(self.player, self.zombie)
        move = self.player.input()
        if move >= 0:
            self.player.move(self.player.speed)
            # self.visible_sprites.reflect(self.player, self.zombie, state, move)
        self.frontground()
        for zombie in self.zombies:
            zombie.move()
        # self.visible_sprites.update()
    
    def frontground(self):
        for zombie in self.zombies:
            zombie.draw_sprites(self.player)



class YSortCameraGroup(pygame.sprite.Group):
    def __init__(self):

        # general setup 
        super().__init__()
        self.display_surface = pygame.display.get_surface()
        self.half_width = self.display_surface.get_size()[0] // 4
        self.half_height = self.display_surface.get_size()[1] // 2
        self.textures = {'x': pygame.image.load('img/1.png').convert(),
                         '2': pygame.image.load('img/2.png').convert(),
                         'S': pygame.image.load('img/night.png').convert()
                         }
        self.offset = pygame.math.Vector2()
    
    def mapping(self, x, y):
        return (x // TILESIZE) * TILESIZE, (y // TILESIZE) * TILESIZE
    
    def background(self, angle):
        myfont = pygame.font.SysFont('Comic Sans MS', 15)
        
        # 2D walls and player:
        for sprite in self.sprites():
            # textsurface = myfont.render(str(sprite.rect.bottom), False, (0, 0, 0))
            offset_pos = sprite.rect.topleft - self.offset
            if 0 <= sprite.rect.right - self.offset.x and sprite.rect.left - self.offset.x <= WIDTH:
                self.display_surface.blit(sprite.image, offset_pos)
                # self.display_surface.blit(textsurface, offset_pos)

        # 3D background
        height, width = self.textures['S'].get_height(), self.textures['S'].get_width()
        offset = int(angle / 360 * width)
        sky = self.textures['S'].subsurface(offset, 0, width - offset, height // 2)
        self.display_surface.blit(sky, (WIDTH, 0))
        if width - offset < WIDTH:
            sky = self.textures['S'].subsurface(0, 0, WIDTH - (width - offset), height // 2)
            self.display_surface.blit(sky, (WIDTH + width - offset, 0))
        pygame.draw.rect(self.display_surface, (100, 100, 100), (WIDTH, HEIGTH / 2, WIDTH, HEIGTH))


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
                pygame.draw.line(self.display_surface, 'black', [x, y], [vx, vy]) 
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
            
            self.display_surface.blit(wall_column, wall_pos)
            # pygame.draw.rect(self.display_surface, (color, color, color), (WIDTH + i * SCALE, int(HEIGTH / 2 - wall_height / 2), SCALE, wall_height))
            rad += DELTA_ANGLE
    
    def get_state(self, player, zombie):
        # 0 ~ 3: distance of 4 directions to the obstacles
        # 4: distance to the target
        # 5: relative direction with the target
        state = [0.0] * 6
        for i in range(4):
            rad = np.deg2rad(90 * i + player.front)
            dis, _, vx, vy = self.ray_casting(player, rad)
            state[i] = dis
            [x, y] = [player.rect.centerx - self.offset.x, player.rect.centery - self.offset.y]
            pygame.draw.line(self.display_surface, 'blue', [x, y], [vx, vy]) 
        # pygame.draw.circle(self.display_surface, 'black', [zombie.rect.centerx - self.offset.x, zombie.rect.centery - self.offset.y], 1.0)
        _, _, pos, distance = zombie.position(player)
        state[4] = distance
        state[5] = pos
        player.state = state
        return np.array(state, dtype = float)
    
    def reflect(self, player, zombie, state_old, final_move):
        state_new = self.get_state(player, zombie)
        def linear(location):
            delta = abs(location - 30)
            return 0.12 * (30 - delta) / 30 if delta < 30 else 0
        def gaussian(distance):
            return 50 * np.exp(- (distance - 64) ** 2)
        reward = linear(state_new[5]) + gaussian(state_new[4])
        done = 0
        player.rewards += reward

        if any(state_new[i] < 60 for i in range(5)):
            done = 1
            player.rewards -= 100

        if TILESIZE <= state_new[4] < 2 * TILESIZE and 0 < state_old[5] < 60:
            done = 1
            player.rewards += 100

        # train short memory
        player.train_short_memory(state_old, final_move, reward, state_new, done)

        #remember
        player.remember(state_old, final_move, reward, state_new, done)

        
        if done:
            # Train long memory, plot result
            player.plot_reward.append(player.rewards)
            player.total_rewards += player.rewards
            player.mean_reward.append(player.total_rewards / player.n_game)
            r = random.randint(2 * TILESIZE, (TILE_V - 3) * TILESIZE) 
            c = random.randint(2 * TILESIZE, (TILE_H - 3) * TILESIZE) 
            player.rect.update(r, c, TILESIZE, TILESIZE)
            # player.rect = player.init_rect.copy()
            player.fx = player.rect.centerx
            player.fy = player.rect.centery
            player.front = 0
            player.n_game += 1
            player.rewards = 0
            player.train_long_memory()
            
            # plot(player.plot_reward, player.mean_reward)