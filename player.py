import pygame
from settings import *
import numpy as np
import collections

class Player(pygame.sprite.Sprite):
    def __init__(self, pos, groups, obstacle_sprites):
        super().__init__(groups)
        self.origin_image = pygame.image.load(
            'img/uav.png').convert_alpha()
        self.image = self.origin_image
        self.init_rect = self.image.get_rect(topleft=pos)
        self.rect = self.init_rect.copy()
        self.hitbox = self.rect #.inflate(0, -26)

        self.distances = [0] * NUM_RAYS
        self.direction = pygame.math.Vector2()
        self.speed = 2
        self.front = 0
        self.fx = self.hitbox.centerx
        self.fy = self.hitbox.centery
        self.obstacle_sprites = obstacle_sprites
        self.state = [0.0] * 6

        self.memory = collections.deque(maxlen = MAX_MEMORY)
        self.n_game = 1
        self.epsilon = 0
        self.rewards = 0
        self.total_rewards = 0
        self.plot_reward = []
        self.mean_reward = []

    def add_target(self, zombie):
        self.zombie = zombie

    def get_state(self):
        return np.array(self.state, dtype = float)

    def input(self):
        keys = pygame.key.get_pressed()

        move = 0
        if keys[pygame.K_UP]:
            self.direction.y = -1
        elif keys[pygame.K_DOWN]:
            self.direction.y = 1
            move = 1
        else:
            self.direction.y = 0

        if keys[pygame.K_RIGHT]:
            self.direction.x = 1
            move = 2
        elif keys[pygame.K_LEFT]:
            self.direction.x = -1
            move = 3
        else:
            self.direction.x = 0

        dire = 1
        if keys[ord('a')]:
            self.front += 2
            self.front %= 360
            dire = 0
        elif keys[ord('d')]:
            self.front -= 2
            self.front %= 360
            dire = 2
        return dire * 4 + move

    def move(self, speed):
        """_summary_

        Args:
            speed (_type_): _description_
        """
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

        self.rect.center = self.hitbox.center

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

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_short_memory(self, state, action, reward, next_state, done):
        pass

    def train_long_memory(self):
        pass
    # def update(self):
    #     state_old = self.get_state()
    #     action = self.input()
    #     self.move(self.speed)
    #     self.remember(state_old, action)
