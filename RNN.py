from keras import metrics
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Dropout
import numpy as np

class RNN:
    def __init__(self, window_size, input_size, num_goals, seq_len=30, sample_fre=60):
        self.model = self.build_model(window_size, input_size, num_goals)
        self.seq_len = seq_len
        self.sample_fre = sample_fre

    def build_model(self, window_size, input_size, num_goals):
        model = Sequential()
        model.add(LSTM(72, input_shape=(window_size, input_size) ,activation='relu', return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(54, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(num_goals, activation='softmax'))
        model.compile('adam', 'categorical_crossentropy', metrics=[metrics.categorical_accuracy])
        return model

    def read_data(self, path):
        file = open(path, 'r')
        data = np.loadtxt(file, dtype='float', delimiter=',')
        state = []
        action = []
        target_state = []
        goal = []
        # reading raw data
        for row in data:
            state.append(np.array(row[:6]))
            action.append(int(row[6]))
            target_state.append(int(row[7]))
            goal.append(row[8])
        X_train, y_train = [], []
        # sample and assemble data
        for i in range(0, len(row)- self.sample_fre * self.sample_fre):
            X_train.append([state[i], action[i], target_state[i]])
            y_train.append(goal[i])
        return np.asarray(X_train), np.asarray(y_train)

    def predict_goal(self, agent_state, action, target_state):
        x = np.array([agent_state, action, target_state])
        return self.model.predict(x)

    def train_rnn(self, path):
        X_train, y_train = self.read_data(path)
        self.model.fit(X_train, y_train)
