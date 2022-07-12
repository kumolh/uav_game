import pygame
from settings import *
import numpy as np
# from collections import deque
import collections

class Player(pygame.sprite.Sprite):
    def __init__(self, pos, groups):
        super().__init__(groups)
        self.origin_image = pygame.image.load(
            'img/uav.png').convert_alpha()
        self.image = self.origin_image
        self.init_rect = self.image.get_rect(topleft=pos)
        self.rect = self.init_rect.copy()
        self.hitbox = self.rect #.inflate(0, -26)

        self.distances = [0] * NUM_RAYS
        self.direction = pygame.math.Vector2()
        self.speed = 3
        self.front = 0
        self.fx = self.hitbox.centerx
        self.fy = self.hitbox.centery
        self.state = [0.0] * 6

        self.memory = collections.deque(maxlen = MAX_MEMORY)
        self.target_state = collections.deque()#maxLen = MAX_MEMORY)
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
        horizontal = vertical = rotate = 0
        if keys[pygame.K_UP]:
            vertical = 1
        elif keys[pygame.K_DOWN]:
            vertical = 2

        if keys[pygame.K_RIGHT]:
            horizontal = 1
        elif keys[pygame.K_LEFT]:
            horizontal = 2

        if keys[ord('a')]:
            rotate = 2
        elif keys[ord('d')]:
            rotate = 1
        player_move = vertical * 9 + horizontal * 3 + rotate

        if vertical == 1:
            self.direction.y = -1
        elif vertical == 2:
            self.direction.y = 1
        else:
            self.direction.y = 0

        if horizontal == 1:
            self.direction.x = 1
        elif horizontal == 2:
            self.direction.x = -1
        else:
            self.direction.x = 0

        if rotate == 1:
            self.front -= 2
            self.front %= 360
        elif rotate == 2:
            self.front += 2
            self.front %= 360
        return player_move

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
        collision = True
        if not(WORLD_MAP[int(self.fy - TILESIZE/ 2) // TILESIZE ][int(self.fx) // TILESIZE] == 'x' or \
                WORLD_MAP[int(self.fy + TILESIZE/2) // TILESIZE ][int(self.fx) // TILESIZE] == 'x' or \
                WORLD_MAP[int(self.fy) // TILESIZE][int(self.fx - TILESIZE / 2) // TILESIZE] == 'x' or \
                WORLD_MAP[int(self.fy) // TILESIZE][int(self.fx + TILESIZE / 2) // TILESIZE] == 'x'):
                self.hitbox.centerx = int(self.fx)
                self.hitbox.centery = int(self.fy)
                collision = False
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
        return collision

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def remember_target(self, target_state):
        self.target_state.append(target_state)

    def train_short_memory(self, state, action, reward, next_state, done):
        pass

    def train_long_memory(self):
        pass
    # def update(self):
    #     state_old = self.get_state()
    #     action = self.input()
    #     self.move(self.speed)
    #     self.remember(state_old, action)
