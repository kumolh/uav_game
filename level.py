import pygame
from sympy import N 
from settings import *
from tile import Tile
from player import Player
from debug import debug
import numpy as np
from zombie import Zombie

class Level:
    def __init__(self):

        # get the display surface 
        self.display_surface = pygame.display.get_surface()

        # sprite group setup
        self.visible_sprites = YSortCameraGroup()
        self.obstacle_sprites = pygame.sprite.Group()

        # sprite setup
        self.create_map()

    def create_map(self):
        for row_index,row in enumerate(WORLD_MAP):
            for col_index, col in enumerate(row):
                x = col_index * TILESIZE
                y = row_index * TILESIZE
                if col == 'x':
                    Tile((x,y),[self.visible_sprites,self.obstacle_sprites])
                if col == 'p':
                    self.player = Player((x, y), [self.visible_sprites], self.obstacle_sprites)
                if col == 'z':
                    self.zombie = Zombie((x, y), [self.visible_sprites], self.obstacle_sprites)

    def run(self):
        # update and draw the game
        self.visible_sprites.background(self.player.front)
        self.visible_sprites.custom_draw(self.player)
        self.frontground()
        self.visible_sprites.update()
    
    def frontground(self):
        self.zombie.draw_sprites(self.player)


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


    def ray_casting(self, player):
        # walls = []
        ox, oy = player.hitbox.centerx, player.hitbox.centery
        xm, ym = (ox // TILESIZE) * TILESIZE, (oy // TILESIZE) * TILESIZE
        cur_angle = np.deg2rad(player.front + 90) - HALF_FOV
        for ray in range(NUM_RAYS):
            sin_a = math.sin(cur_angle)
            cos_a = math.cos(cur_angle)
            sin_a = sin_a if sin_a else 0.000001
            cos_a = cos_a if cos_a else 0.000001
            # verticals
            texture_v = 'x'
            (x, dx) = (xm + TILESIZE, 1) if cos_a >= 0 else (xm, -1)
            for _ in range(0, WIDTH, TILESIZE):
                depth_v = (x - ox) / cos_a
                yv = oy + depth_v * sin_a
                tile_v = self.mapping(x + dx, yv)
                if tile_v in world_map:
                    texture_v = world_map[tile_v]
                    break
                x += dx * TILESIZE

            # horizontals
            texture_h = 'x'
            y, dy = (ym + TILESIZE, 1) if sin_a >= 0 else (ym, -1)
            for i in range(0, HEIGTH, TILESIZE):
                depth_h = (y - oy) / sin_a
                xh = ox + depth_h * cos_a
                tile_h = self.mapping(xh, y + dy)
                if tile_h in world_map:
                    texture_h = world_map[tile_h]
                    break
                y += dy * TILESIZE

            # projection
            depth, offset, texture = (depth_v, yv, texture_v) if depth_v < depth_h else (depth_h, xh, texture_h)
            offset = int(offset) % TILESIZE
            depth *= math.cos((NUM_RAYS // 2 - ray) * DELTA_ANGLE)
            depth = max(depth, 0.00001)
            proj_height = min(int(PROJ_COEFF / depth), 2 * HEIGTH)

            wall_column = self.textures[texture].subsurface(offset * TEXTURE_SCALE, 0, TEXTURE_SCALE, TEXTURE_HEIGHT)
            wall_column = pygame.transform.scale(wall_column, (SCALE, proj_height))
            wall_pos = (WIDTH + ray * SCALE, HALF_HEIGHT - proj_height // 2)
            self.display_surface.blit(wall_column, wall_pos)
            # walls.append((depth, wall_column, wall_pos))
            cur_angle += DELTA_ANGLE
        # return walls

    def custom_draw(self,player):

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

        rx, ry, vx, vy = 0.0, 0.0, 0.0, 0.0 
        rad = np.deg2rad(90 + player.front) - HALF_FOV
        for i in range(NUM_RAYS):      
            xo = 0
            yo = -TILESIZE if np.sin(rad) > 0 else TILESIZE 
            # check vertical
            dof = 0
            disV = 10000
            tan = np.tan(rad)
            if np.cos(rad) > 0.001: # looking right
                rx = int(player.hitbox.centerx // TILESIZE) * TILESIZE + TILESIZE 
                ry = (player.hitbox.centerx - rx) * tan + player.hitbox.centery 
                xo = TILESIZE
                yo = - xo * tan
            elif np.cos(rad) < -0.001:
                rx = int(player.hitbox.centerx // TILESIZE) * TILESIZE 
                ry = (player.hitbox.centerx - rx) * tan + player.hitbox.centery 
                xo = -TILESIZE
                yo = - xo * tan
            else:
                dof = DOF
                yo = -TILESIZE if np.sin(rad) > 0 else TILESIZE 
                rx = player.hitbox.centerx 
                ry = player.hitbox.centery
                
            while dof < DOF:
                mx = int(rx // TILESIZE) if np.cos(rad) > 0.001 else int(rx // TILESIZE) - 1
                my = int(ry // TILESIZE) #if np.cos(rad) > 0.001 else int(ry // TILESIZE) - 1
                if 0 <= mx < MAPX and 0 <= my < MAPY and WORLD_MAP[my][mx] == 'x':
                    dof = DOF
                    disV = np.cos(rad) * (rx - player.hitbox.centerx) - np.sin(rad) * (ry - player.hitbox.centery)
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
                ry = int(player.hitbox.centery / TILESIZE) * TILESIZE
                rx = (player.hitbox.centery - ry) * tan + player.hitbox.centerx 
                yo = - TILESIZE
                xo = - yo * tan
            elif np.sin(rad) < -0.001:
                ry = int(player.hitbox.centery / TILESIZE) * TILESIZE + TILESIZE 
                rx = (player.hitbox.centery - ry) * tan + player.hitbox.centerx 
                yo = TILESIZE
                xo = - yo * tan
            else:
                dof = DOF
                yo = -TILESIZE if np.sin(rad) > 0 else TILESIZE 
                rx = player.hitbox.centerx
                ry = player.hitbox.centery
                
            while dof < DOF:
                mx = int(rx / TILESIZE) #if np.sin(rad) < -0.001 else int(rx / TILESIZE) + 1
                my = int(ry / TILESIZE) if np.sin(rad) < -0.001 else int(ry / TILESIZE) - 1
                
                if 0 <= mx < MAPX and 0 <= my < MAPY and WORLD_MAP[my][mx] == 'x':
                    # print(mx, my)
                    dof = DOF
                    disH = np.cos(rad) * (rx - player.hitbox.centerx) - np.sin(rad) * (ry - player.hitbox.centery)
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
            [x, y] = [player.hitbox.centerx - self.offset.x, player.hitbox.centery - self.offset.y]
            if i == 0 or i == NUM_RAYS - 1:
                pygame.draw.line(self.display_surface, 'black', [x, y], [vx, vy]) 
            dis *= np.cos((NUM_RAYS//2 - i) * DELTA_ANGLE)
            wall_height = min(HEIGTH, int(35000 / (dis + 0.001)))
            color = 255 / (1 + dis * dis * 0.0001)
            offset = int(offset) % TILESIZE

            height, width = self.textures['x'].get_height(), self.textures['x'].get_width()
            scale = width // TILESIZE
            wall_column = self.textures['x'].subsurface(offset * scale, 0, scale, height)
            wall_column = pygame.transform.scale(wall_column, (SCALE, wall_height))
            wall_pos = (2 *WIDTH - i * SCALE, HALF_HEIGHT - wall_height // 2)
            self.display_surface.blit(wall_column, wall_pos)
            # pygame.draw.rect(self.display_surface, (color, color, color), (WIDTH + i * SCALE, int(HEIGTH / 2 - wall_height / 2), SCALE, wall_height))
            rad += DELTA_ANGLE