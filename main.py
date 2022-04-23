import pygame, sys
from settings import *
from level import Level

class Game:
	def __init__(self):
		  
		# general setup
		pygame.init()
		self.screen = pygame.display.set_mode((WIDTH * 2, HEIGTH))
		pygame.display.set_caption('UAV')
		self.clock = pygame.time.Clock()

		self.level = Level()
	
	def run(self):
		while True:
			for event in pygame.event.get():
				if event.type == pygame.QUIT or event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
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