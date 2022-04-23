import pygame
from settings import *
import numpy as np

class Player(pygame.sprite.Sprite):
    def __init__(self, pos, groups, obstacle_sprites):
        super().__init__(groups)
        self.origin_image = pygame.image.load(
            'img/uav.png').convert_alpha()
        self.image = self.origin_image
        self.rect = self.image.get_rect(topleft=pos)
        self.hitbox = self.rect #.inflate(0, -26)

        self.direction = pygame.math.Vector2()
        self.speed = 3
        self.front = 0
        self.fx = self.hitbox.centerx
        self.fy = self.hitbox.centery
        self.obstacle_sprites = obstacle_sprites

    def input(self):
        keys = pygame.key.get_pressed()

        if keys[pygame.K_UP]:
            self.direction.y = -1
        elif keys[pygame.K_DOWN]:
            self.direction.y = 1
        else:
            self.direction.y = 0

        if keys[pygame.K_RIGHT]:
            self.direction.x = 1
        elif keys[pygame.K_LEFT]:
            self.direction.x = -1
        else:
            self.direction.x = 0

        if keys[ord('a')]:
            self.front += 2
            self.front %= 360
        elif keys[ord('d')]:
            self.front -= 2
            self.front %= 360

    def move(self, speed):
        if self.direction.magnitude() != 0:
            self.direction = self.direction.normalize()
    
        self.image = pygame.transform.rotate(self.origin_image, self.front)
        rad = np.deg2rad(-self.front)
        self.fx += -speed * np.sin(rad) * self.direction.y + speed * np.cos(rad) * self.direction.x
        self.fy += speed * np.cos(rad) * self.direction.y + speed * np.sin(rad) * self.direction.x

        if not(WORLD_MAP[int(self.fy - TILESIZE/ 2) // TILESIZE ][int(self.fx) // TILESIZE] == 'x' or \
                WORLD_MAP[int(self.fy + TILESIZE/2) // TILESIZE ][int(self.fx) // TILESIZE] == 'x' or \
                WORLD_MAP[int(self.fy) // TILESIZE][int(self.fx - TILESIZE / 2) // TILESIZE] == 'x' or \
                WORLD_MAP[int(self.fy) // TILESIZE][int(self.fx + TILESIZE / 2) // TILESIZE] == 'x'):
                self.hitbox.centerx = int(self.fx)
                self.hitbox.centery = int(self.fy)
        elif not(WORLD_MAP[int(self.hitbox.top) // TILESIZE ][int(self.fx) // TILESIZE] == 'x' or \
                WORLD_MAP[int(self.hitbox.bottom + 1) // TILESIZE][int(self.fx) // TILESIZE] == 'x' or \
                WORLD_MAP[int(self.hitbox.y) // TILESIZE ][int(self.fx - TILESIZE / 2) // TILESIZE] == 'x' or \
                WORLD_MAP[int(self.hitbox.y) // TILESIZE ][int(self.fx + TILESIZE / 2) // TILESIZE] == 'x'):
                self.hitbox.centerx = int(self.fx)
                self.fy = self.hitbox.centery
        elif not(WORLD_MAP[int(self.fy - TILESIZE / 2) // TILESIZE ][int(self.hitbox.x) // TILESIZE] == 'x' or \
                WORLD_MAP[int(self.fy +TILESIZE / 2) // TILESIZE ][int(self.hitbox.x) // TILESIZE] == 'x' or \
                WORLD_MAP[int(self.fy) // TILESIZE][int(self.hitbox.left) // TILESIZE] == 'x' or \
                WORLD_MAP[int(self.fy) // TILESIZE][int(self.hitbox.right + 1) // TILESIZE] == 'x'):
                self.hitbox.centery = int(self.fy)
                self.fx = self.hitbox.centerx
        else:
            self.fx = self.hitbox.centerx
            self.fy = self.hitbox.centery

        # self.hitbox.x = int(self.fx)
        # self.collision('horizontal')
        # self.hitbox.y = int(self.fy)
        # self.collision('vertical')
        self.rect.center = self.hitbox.center
        # print(self.fy , self.fx)
        # print(int(self.fy) // TILESIZE, int(self.fx) // TILESIZE + 1)

    def collision(self, direction):
        front = (self.front) % 360
        dire = [False] * 4 #down up left right
        if front <= 45 or front > 315: # face up
            if self.direction.x == 1: dire[3] = True
            elif self.direction.x == -1: dire[2] = True
            if self.direction.y == 1: dire[0] = True
            elif self.direction.y == -1: dire[1] = True
        elif 135 < front <= 225: # face down
            if self.direction.x == 1: dire[2] = True
            elif self.direction.x == -1: dire[3] = True
            if self.direction.y == 1: dire[1] = True
            elif self.direction.y == -1: dire[0] = True
        elif 225 < front <= 315: # face right
            if self.direction.x == 1: dire[1] = True
            elif self.direction.x == -1: dire[0] = True
            if self.direction.y == 1: dire[2] = True
            elif self.direction.y == -1: dire[3] = True
        else: #face left
            if self.direction.x == 1: dire[0] = True
            elif self.direction.x == -1: dire[1] = True
            if self.direction.y == 1: dire[3] = True
            elif self.direction.y == -1: dire[2] = True

        if dire[1] or dire[0]:
            for sprite in self.obstacle_sprites:
                if sprite.hitbox.colliderect(self.hitbox):
                    if dire[1]:   # moving up
                        self.hitbox.top = sprite.hitbox.bottom
                        self.fy = self.hitbox.y   
                    elif dire[0]:
                        self.hitbox.bottom = sprite.hitbox.top
                        self.fy = self.hitbox.y   

        if dire[3] or dire[2]:     
            for sprite in self.obstacle_sprites:
                if sprite.hitbox.colliderect(self.hitbox):       
                    if dire[3]: # moving right
                        self.hitbox.right = sprite.hitbox.left
                        self.fx = self.hitbox.x
                    elif dire[2]:   # moving left
                        self.hitbox.left = sprite.hitbox.right
                        self.fx = self.hitbox.x


    def update(self):
        self.input()
        self.move(self.speed)
