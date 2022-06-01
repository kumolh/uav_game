import pygame
from settings import *
import numpy as np
import tensorflow as tf
import collections
import random
from Helper import plot
from QTrainer import QTrainer


class AI_Player(pygame.sprite.Sprite):
    def __init__(self, pos, groups, obstacle_sprites):
        super().__init__(groups)
        self.origin_image = pygame.image.load(
            'img/uav.png').convert_alpha()
        self.image = self.origin_image
        self.init_rect = self.image.get_rect(topleft=pos)
        self.rect = self.init_rect.copy()
        self.hitbox = self.rect #.inflate(0, -26)
        # self.zombie = zombie

        self.distances = [0] * NUM_RAYS
        self.direction = pygame.math.Vector2()
        self.speed = 2
        self.front = 0
        self.fx = self.hitbox.centerx
        self.fy = self.hitbox.centery
        self.obstacle_sprites = obstacle_sprites
        self.state = [0.0] * 6

        # self.goal_space = [g1] # -->
        # self.confidence = [c1] # --> sumup: 1

        self.trainer = QTrainer(lr = LR, gamma = 0.9, load = False)
        self.trainer.learn_from_demo()
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

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if (len(self.memory) > BATCH_SIZE):
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff explotation / exploitation
        self.epsilon = 80 - self.n_game
        final_move = 0
        if(random.randint(0, 200) < self.epsilon):
            final_move = random.randint(0, 11)
        else:
            state0 = tf.convert_to_tensor(state, dtype=tf.float32)
            state0 = tf.expand_dims(state0, 0)
            prediction = self.trainer.model(state0)
            final_move = np.argmax(prediction.numpy(), axis=1)[0]
        return final_move

    def is_collision(self, zombie):
        #check if collision with wall
        n, m = self.rect.centerx // TILESIZE, self.rect.centery // TILESIZE
        if 0 <= n < TILE_H and 0 <= m < TILE_V and WORLD_MAP[m][n] == 'x': return True
        if 0 <= n < TILE_H and 0 <= m+1 < TILE_V and WORLD_MAP[m][n+1] == 'x': return True
        if 0 <= n+1 < TILE_H and 0 <= m < TILE_V and WORLD_MAP[m+1][n] == 'x': return True
        if 0 <= n+1 < TILE_H and 0 <= m+1 < TILE_V and WORLD_MAP[m+1][n+1] == 'x': return True
        if self.rect.colliderect(zombie.rect): return True
        return False

    def input(self):
        state_old = self.get_state()
        # get move
        final_move = self.get_action(state_old)

        # perform move and get new state
        if final_move % 4 == 0:
            self.direction.y = -1
        elif final_move % 4 == 1:
            self.direction.y = 1
        else:
            self.direction.y = 0

        if final_move % 4 == 2:
            self.direction.x = 1
        elif final_move % 4 == 3:
            self.direction.x = -1
        else:
            self.direction.x = 0

        if final_move // 4 == 0:
            self.front -= 2
            self.front %= 360
        elif final_move // 4 == 2:
            self.front += 2
            self.front %= 360
    
        return final_move
        


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

        self.rect.center = self.hitbox.center
    
    def reflect(self, state_old, final_move):
        state_new = self.get_state()
        def linear(location):
            """calculate the reward for target in the middle

            Args:
                location (_type_): the ray index of the target in the captured image

            Returns:
                _type_: reward from linear interpolation 
            """
            delta = abs(location - 30)
            return 0.12 * (30 - delta) / 30 if delta < 30 else 0
        def gaussian(distance):
            return 50 * np.exp(- (distance - 64) ** 2)
        reward = linear(state_new[5]) + gaussian(state_new[4])
        done = False
        self.rewards += reward

        if any(state_new[i] < 60 for i in range(5)):
            done = True
            self.rewards -= 100

        if TILESIZE <= state_new[4] < 2 * TILESIZE and 0 < state_old[5] < 60:
            done = True
            self.rewards += 100

        # train short memory
        self.train_short_memory(state_old, final_move, reward, state_new, done)

        #remember
        self.remember(state_old, final_move, reward, state_new, done)

        
        if done:
            # Train long memory, plot result
            self.plot_reward.append(self.rewards)
            self.total_rewards += self.rewards
            self.mean_reward.append(self.total_rewards / self.n_game)
            self.rect = self.init_rect.copy()
            self.fx = self.rect.centerx
            self.fy = self.rect.centery
            self.front = 0
            self.n_game += 1
            self.rewards = 0
            self.train_long_memory()
            
            plot(self.plot_reward, self.mean_reward)

    def update(self):
        state_old = self.get_state()
        final_move = self.input()
        self.move(self.speed)
        self.reflect(state_old, final_move)