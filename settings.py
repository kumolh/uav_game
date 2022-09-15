import math
from Maps import *
# game setup
WIDTH    = 600	
HEIGTH   = 600
HALF_WIDTH = WIDTH // 2
HALF_HEIGHT = HEIGTH // 2
FPS      = 120
TILESIZE = 64
# MAPX = MAPY = 20
DOF = 20
SEQ_LEN = 20

FOV = math.pi / 3
HALF_FOV = FOV / 2
NUM_RAYS = 100
SCALE = WIDTH // NUM_RAYS
DIST = NUM_RAYS / (2 * math.tan(HALF_FOV))
DELTA_ANGLE = FOV / NUM_RAYS
PROJ_COEFF = 3 * DIST * TILESIZE

TEXTURE_WIDTH = 1200
TEXTURE_HEIGHT = 1200
TEXTURE_SCALE = TEXTURE_WIDTH // TILESIZE

WORLD_MAP = WORLD_OBSTACLE #WORLD_OPEN #

MAPY = TILE_V = len(WORLD_MAP)
MAPX = TILE_H = len(WORLD_MAP[0])

MINI_SCALE = 6

world_map = {}
for i, row in enumerate(WORLD_MAP):
    for j, char in enumerate(row):
        if char != ' ':
            if char == 'x':
                world_map[(i * TILESIZE, j * TILESIZE)] = 'x'
            elif char == '2':
                world_map[(i * TILESIZE, j * TILESIZE)] = '2'

MAX_MEMORY = 100000
LR = 0.001
BATCH_SIZE = 3000