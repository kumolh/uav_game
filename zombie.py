import pygame
import numpy as np
from settings import *
import random


class Zombie(pygame.sprite.Sprite):
    def __init__(self, pos, groups, obstacle_sprites):
        super().__init__(groups)
        self.front = 0
        self.zombie = self.load_img()
        self.image = pygame.image.load('img/zombie/zombie1.png').convert_alpha()
        self.rect = self.image.get_rect(topleft=pos)
        self.obstacle_sprites = obstacle_sprites
        self.display_surface = pygame.display.get_surface()
        self.speed = 2

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
                sprites[1][i].append(pygame.Surface.subsurface(
                    img, (xx + 96, yy, 32, 100)))
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
            zombie = self.zombie[1][0][idx]
            height, width = 100, 32
            scale = zombie_height / height
            zombie = pygame.transform.scale(zombie, (scale * width, zombie_height))
            self.display_surface.blit(zombie, zombie_pos)
        
    def move(self):
        final_move = random.randint(0, 3)
        if final_move == 0:
            self.rect.centerx += self.speed
        elif final_move == 1:
            self.rect.centerx -= self.speed
        elif final_move == 2:
            self.rect.centery -= self.speed
        elif final_move == 3:
            self.rect.centery += self.speed

