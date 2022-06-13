from keras.layers import Dense, Dropout, Conv2D, Flatten
from keras.models import Sequential
import keras
import tensorflow as tf
import numpy as np

class QTrainer:
    """primarily a nn model, can be used as Q-select(main model) and Q-evaluation(using for updating the Q-select network)
    """
    def __init__(self, lr, gamma, load = False):
        self.lr = lr
        self.gamma = gamma
        self.model = Sequential([
            Dense(128, input_shape=(6,), activation='relu'),
            Dense(64, activation='relu'),
            Dense(14, activation=None)
        ])
        if load:
            self.model = keras.models.load_model("my_model.h5")

            
    
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
    
    def save_model(self, path = "my_model.h5"):
        keras.models.save_model(self.model, path, overwrite=True)
    
    def learn_from_demo(self, path = 'demo.csv'):
        file = open(path, 'r')
        data = np.loadtxt(file, dtype='float', delimiter=',')
        state = []
        action = []
        reward = []
        next_state = []
        done = []
        for row in data:
            state.append(np.array(row[:6]))
            action.append(int(row[6]))
            reward.append(row[7])
            next_state.append(np.array(row[8:14]))
            done.append(row[14])
        self.train_step(state, action, reward, next_state, done)
        self.save_model()