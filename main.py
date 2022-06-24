from cmath import rect
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
        myfont = pygame.font.SysFont('Comic Sans MS', 24)
        goal_input = False
        dig1 = myfont.render('What"s your intention please?', True, 'black')
        dig2 = myfont.render('1. chasing; 2. circling; 3. finding', True, 'blue')
        rect1 = dig1.get_rect()
        rect2 = dig2.get_rect()
        rect1.center = (WIDTH, HEIGTH // 3)
        rect2.center = (WIDTH, HEIGTH // 2)
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    # self.save_demo()
                    pygame.quit()
                    sys.exit()
            self.screen.fill('white')
            if not goal_input:
                self.screen.blit(dig1, rect1)
                self.screen.blit(dig2, rect2)
                keys = pygame.key.get_pressed()
                if keys[ord('1')] or keys[ord('2')] or keys[ord('3')]: 
                    goal_input = True
                    if keys[ord('1')]: goal = 1
                    elif keys[ord('1')]: goal = 2
                    else: goal = 3
                    self.level.goal_input(goal)
            else:
                self.level.run()
            pygame.display.set_caption('FPS: ' + str(int(self.clock.get_fps())))
            pygame.display.update()
            self.clock.tick()

if __name__ == '__main__':
    game = Game()
    game.run()
