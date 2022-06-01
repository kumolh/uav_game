import pygame, sys
from settings import *
from level import Level
import numpy as np
class Game:
    def __init__(self):
          
        # general setup
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH * 2, HEIGTH))
        pygame.display.set_caption('UAV')
        self.clock = pygame.time.Clock()

        self.level = Level()
    
    def save_demo(self):
        file = open('demo.csv', 'w')
        for tuple in self.level.player.memory:
            lst = list(tuple[0]) + [tuple[1], tuple[2]] + list(tuple[3]) + [tuple[4]]
            np.savetxt(file, [lst], delimiter=', ')
        file.close()

    def run(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    # self.save_demo()
                    pygame.quit()
                    sys.exit()
            
            self.screen.fill('white')
            self.level.run()
            pygame.display.set_caption('FPS: ' + str(int(self.clock.get_fps())))
            pygame.display.update()
            self.clock.tick()

if __name__ == '__main__':
    game = Game()
    game.run()