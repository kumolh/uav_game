A Simple Project For My Thesis

install dependencies:
    pip install requirements.txt

use:
    game.py: for manual input
    rl_model_generation.py: for trianing and testing new RL models
    data_generation.py: generating replay buffers
    transfomer.py/RNN.py: generating prediciton model
    uav_game.py: backbone of the whole project

maps:
    Maps.py: define new maps
    settings.py: change active maps/store global variables

data:
    Dateset.py: read csv file into memory

moving objects:
    player.py
    zombie.py

other data:
    mainly useless