import pygame
import numpy as np
from settings import *
import random


class Zombie(pygame.sprite.Sprite):
    def __init__(self, pos, groups, obstacle_sprites, type):
        super().__init__(groups)
        self.front = 0
        self.zombie = self.load_img()
        self.type = type
        self.image = pygame.image.load('img/zombie/zombie' + str(type) + '.png').convert_alpha()
        self.rect = self.image.get_rect(topleft=pos)
        self.fx = self.rect.centerx
        self.fy = self.rect.centery
        self.obstacle_sprites = obstacle_sprites
        self.display_surface = pygame.display.get_surface()
        self.speed = 1
        self.step = 0
        self.direction = 0

    def load_img(self):
        img = pygame.image.load(
            'img/zombie/zombie_n_skeleton4.png').convert_alpha()
        sprites = [[], []]
        for i in range(3):
            xx = i * 32
            sprites[0].append([])
            sprites[1].append([])
            for j in range(4):
                yy = j * 100
                sprites[0][i].append(
                    pygame.Surface.subsurface(img, (xx, yy, 32, 100)))
                sprites[1][i].append(
                    pygame.Surface.subsurface(img, (xx + 96, yy, 32, 100)))
        return sprites

    def position(self, player):
        relative_pos = [self.rect.centerx - player.rect.centerx, self.rect.centery - player.rect.centery]
        angle = np.arctan2(-relative_pos[1], relative_pos[0])
        if angle < 0: angle += 2 * np.pi
        angle = (np.rad2deg(angle) - 90) % 360
        dis = np.sqrt(sum(x ** 2 for x in relative_pos))
        zombie_height = min(2 * HEIGTH, int(100000 / (dis + 0.001)))
        # zombie_pos = (player.front + 30 - angle) % 360
        theta = (player.front + 30 - angle) % 360
        zombie_pos = (WIDTH + theta * WIDTH // 60, HALF_HEIGHT - zombie_height // 3)
        return zombie_height, zombie_pos, theta, dis
        
    def draw_sprites(self, player):
        # enemies : x, y, angle2p, dist2p, type, size, direction, dir2p
        zombie_height, zombie_pos, theta, dis = self.position(player)
        ray_idx = int(2 * WIDTH - zombie_pos[0]) // SCALE
        if WIDTH < zombie_pos[0] < 2 * WIDTH and player.distances[ray_idx] > dis:
            idx = 3
            if 135 <= theta < 225: idx = 0
            elif theta < 45 or 315 <= theta: idx = 2
            elif 225 <= theta < 315: idx = 1
            zombie = self.zombie[self.type][0][idx]
            height, width = 100, 32
            scale = zombie_height / height
            zombie = pygame.transform.scale(zombie, (scale * width, zombie_height))
            self.display_surface.blit(zombie, zombie_pos)
        
    def move(self):
        need_change = True
        dx = dy = 0
        if self.direction == 0: dy = -1
        elif self.direction == 1: dx = -1
        elif self.direction == 2: dy = 1
        else: dx = 1
        self.fx = self.rect.centerx + dx * self.speed
        self.fy = self.rect.centery + dy * self.speed
        if not(WORLD_MAP[int(self.fy - TILESIZE/ 2) // TILESIZE ][int(self.fx) // TILESIZE] == 'x' or \
                WORLD_MAP[int(self.fy + TILESIZE/2) // TILESIZE ][int(self.fx) // TILESIZE] == 'x' or \
                WORLD_MAP[int(self.fy) // TILESIZE][int(self.fx - TILESIZE / 2) // TILESIZE] == 'x' or \
                WORLD_MAP[int(self.fy) // TILESIZE][int(self.fx + TILESIZE / 2) // TILESIZE] == 'x'):
                self.rect.centerx = self.fx
                self.rect.centery = self.fy
                need_change = False
        elif not(WORLD_MAP[int(self.rect.top) // TILESIZE ][int(self.fx) // TILESIZE] == 'x' or \
                WORLD_MAP[int(self.rect.bottom + 1) // TILESIZE][int(self.fx) // TILESIZE] == 'x' or \
                WORLD_MAP[int(self.rect.y) // TILESIZE ][int(self.fx - TILESIZE / 2) // TILESIZE] == 'x' or \
                WORLD_MAP[int(self.rect.y) // TILESIZE ][int(self.fx + TILESIZE / 2) // TILESIZE] == 'x'):
                self.rect.centerx = int(self.fx)
                self.fy = self.rect.centery
        elif not(WORLD_MAP[int(self.fy - TILESIZE / 2) // TILESIZE ][int(self.rect.x) // TILESIZE] == 'x' or \
                WORLD_MAP[int(self.fy +TILESIZE / 2) // TILESIZE ][int(self.rect.x) // TILESIZE] == 'x' or \
                WORLD_MAP[int(self.fy) // TILESIZE][int(self.rect.left) // TILESIZE] == 'x' or \
                WORLD_MAP[int(self.fy) // TILESIZE][int(self.rect.right + 1) // TILESIZE] == 'x'):
                self.rect.centery = int(self.fy)
                self.fx = self.rect.centerx
        else:
            self.fx = self.rect.centerx
            self.fy = self.rect.centery
        if need_change:
            self.direction += 1
            self.direction %= 4 
            self.step = 0
        if self.step > 240 and random.uniform(0, 1) < 0.5:
            self.step = 0
            new_dire = self.direction
            while new_dire == self.direction:
                new_dire = random.randint(0, 3)
            self.direction = new_dire
        self.step += 1



