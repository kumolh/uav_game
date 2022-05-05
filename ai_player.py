import pygame
from settings import *
import numpy as np
import tensorflow as tf
from keras.layers import Dense, Dropout, Conv2D, Flatten
from keras.models import Sequential
import numpy as np
import collections
import random

class QTrainer:
    def __init__(self, lr, gamma, load = False):
        self.lr = lr
        self.gamma = gamma
        self.model = Sequential([
            Dense(256, input_shape=(11,), activation='relu'),
            Dense(4, activation=None)
        ])
        if load:
            self.model.load_weights('model.ckpt')
            
    
    def train_step(self, state, action, reward, next_state, done):
        state = tf.convert_to_tensor(state, dtype=tf.float32)
        next_state = tf.convert_to_tensor(next_state, dtype=tf.float32)
        action = tf.convert_to_tensor(action, dtype=tf.int64)
        reward = tf.convert_to_tensor(reward, dtype=tf.float32)


        if(len(state.shape) == 1): # only one parameter to train , Hence convert to tuple of shape (1, x)
            #(1 , x)
            state = tf.expand_dims(state, 0)
            next_state = tf.expand_dims(next_state, 0)
            action = tf.expand_dims(action, 0)
            reward = tf.expand_dims(reward, 0)
            done = (done, )

        # 1. Predicted Q value with current state
        pred = self.model(state)
        target = tf.identity(pred) #pred.clone().cuda()
        for idx in range(len(done)):
            Q_new = reward.numpy()[idx]
            if not done[idx]:
                Q_new = reward.numpy()[idx] + self.gamma * np.max(self.model(next_state))
            target = target.numpy()
            index = np.argmax(action.numpy()[idx], axis=0)
            target[idx][index] = Q_new 
            target = tf.convert_to_tensor(target)
        # 2. Q_new = reward + gamma * max(next_predicted Qvalue) -> only do this if not done

        # training step : gradient decent (1.0) to minimize loss
        opt = tf.keras.optimizers.Adam(learning_rate=self.lr)
        loss_fn = lambda: tf.keras.losses.mse(self.model(state), target)
        var_list_fn = lambda: self.model.trainable_weights
        ### training
        opt.minimize(loss_fn, var_list_fn)

class AI_Player(pygame.sprite.Sprite):
    def __init__(self, pos, groups, obstacle_sprites, target, model):
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
        self.target = target

        self.trainer = QTrainer(lr = LR, gamma = 0.9, load=True)
        self.memory = collections.deque(maxlen = MAX_MEMORY)
        self.n_game = 0
        self.epsilon = 0

    def get_state(self):
        state = [
            # wall up
            WORLD_MAP[self.rect.centerx // TILESIZE][self.rect.centery // TILESIZE] == 'x'
            or ((self.rect.centery // TILESIZE - 1 < TILE_V) and WORLD_MAP[self.rect.centerx // TILESIZE][self.rect.centery // TILESIZE - 1] == 'x'),

             # wall down
            WORLD_MAP[self.rect.centerx // TILESIZE][self.rect.centery // TILESIZE + 1] == 'x'
            or ((self.rect.centery // TILESIZE + 2 < TILE_V) and WORLD_MAP[self.rect.centerx // TILESIZE][self.rect.centery // TILESIZE + 2] == 'x'),

            # wall right
            WORLD_MAP[self.rect.centerx // TILESIZE + 1][self.rect.centery // TILESIZE] == 'x'
            or ((self.rect.centerx // TILESIZE + 2 < TILE_H) and WORLD_MAP[self.rect.centerx // TILESIZE + 2][self.rect.centery // TILESIZE] == 'x'),

            #wall Left
            WORLD_MAP[self.rect.centerx // TILESIZE][self.rect.centery // TILESIZE] == 'x'
            or ((1 <= self.rect.centerx // TILESIZE) and WORLD_MAP[self.rect.centerx // TILESIZE - 1][self.rect.centery // TILESIZE] == 'x'),

            #moving  Direction

            #target Location
            self.target.rect.centerx < self.rect.centerx, # target is in left
            self.target.rect.centerx > self.rect.centerx, # target is in right
            self.target.rect.centery < self.rect.centery, # target is in up
            self.target.rect.centery > self.rect.centery, # target is in down
        ]
        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if (len(self.memory) > BATCH_SIZE):
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory
        states,actions,rewards,next_states,dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff explotation / exploitation
        self.epsilon = 80 - self.n_game
        final_move = [0, 0, 0, 0]
        if(random.randint(0, 200) < self.epsilon):
            move = random.randint(0, 3)
            final_move[move] = 1
        else:
            # state0 = torch.tensor(state,dtype=torch.float).cuda()
            state0 = tf.convert_to_tensor(state, dtype=tf.float32)
            state0 = tf.expand_dims(state0, 0)
            # prediction = self.model(state0).cuda() # prediction by model 
            prediction = self.model(state0)
            # move = torch.argmax(prediction).item()
            move = np.argmax(prediction.numpy(), axis=1)[0]
            final_move[move] = 1 
        return final_move

    def input(self):
        state_old = self.get_state()
        # get move
        final_move = self.get_action(state_old)

        # perform move and get new state

        reward, done, score = game.play_step(final_move)
        state_new = self.get_state()

        # train short memory
        self.train_short_memory(state_old, final_move, reward, state_new, done)

        #remember
        self.remember(state_old,final_move,reward,state_new,done)

        # if done:
        #     # Train long memory,plot result
        #     game.reset()
        #     agent.n_game += 1
        #     agent.train_long_memory()
        #     if(score > reward): # new High score 
        #         reward = score
        #         agent.model.save_model()
        #     print('Game:',agent.n_game,'Score:',score,'Record:',record)
            
        #     plot_scores.append(score)
        #     total_score += score
        #     mean_score = total_score / agent.n_game
        #     plot_mean_scores.append(mean_score)
        #     plot(plot_scores, plot_mean_scores)


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

    def update(self):
        self.input()
        self.move(self.speed)