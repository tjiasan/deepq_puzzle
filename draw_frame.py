import pygame
import cv2
import numpy as np


size = width, height = 80,80
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)

pygame.init()
screen= pygame.display.set_mode(size)

class Maze:
    def __init__(self):
        self.start=0
        self.end = 5
        self.step =0
        self.hole = False
        self.totalactions =0
       
    def draw_frame(self):
        screen.fill(WHITE)
        pygame.draw.circle(screen,GREEN,(50,20),5,5)
        pygame.draw.line(screen,BLACK,[0,0], [65,0],2)
        pygame.draw.line(screen,BLACK,[15,23], [65,23],2)
        pygame.draw.line(screen,BLACK,[15,23], [15,13],2)
        pygame.draw.line(screen,BLACK,[25,23], [25,13],2)
        pygame.draw.line(screen,BLACK,[35,23], [35,13],2)
        pygame.draw.line(screen,BLACK,[45,23], [45,13],2)
        pygame.draw.line(screen,BLACK,[55,23], [55,13],2)
        pygame.draw.line(screen,BLACK,[65,23], [65,0],2)
        pygame.draw.line(screen,BLACK,[15,13], [0,13],2)
        down= 0
        if self.hole== True:
            down= 10
        right = self.step*10
        pygame.draw.circle(screen,BLACK,(10+ right,10+ down),5,5)
        
        x_t1= pygame.surfarray.pixels3d(screen)
        x_t4= cv2.cvtColor(cv2.resize(x_t1, (80, 80)), cv2.COLOR_BGR2GRAY)
        ret, x_t4 = cv2.threshold(x_t4,1,255,cv2.THRESH_BINARY)                    
        state = np.stack((x_t4,x_t4), axis =2)
        return state
            
        

        
    def game(self,action):
        
        terminal= False
        reward = 0 
        if action == 0:
            if self.step < 5 and self.hole==False:
                self.step += 1
            else:
                pass

        if action == 1:
            if self.step != 0:
                self.hole= True
            else:
                pass

        if action == 2:
            self.hole = False
            
        screen.fill(WHITE)
        pygame.draw.circle(screen,GREEN,(50,20),5,5)
        pygame.draw.line(screen,BLACK,[0,0], [65,0],2)
        pygame.draw.line(screen,BLACK,[15,23], [65,23],2)
        pygame.draw.line(screen,BLACK,[15,23], [15,13],2)
        pygame.draw.line(screen,BLACK,[25,23], [25,13],2)
        pygame.draw.line(screen,BLACK,[35,23], [35,13],2)
        pygame.draw.line(screen,BLACK,[45,23], [45,13],2)
        pygame.draw.line(screen,BLACK,[55,23], [55,13],2)
        pygame.draw.line(screen,BLACK,[65,23], [65,0],2)
        pygame.draw.line(screen,BLACK,[15,13], [0,13],2)
        down= 0
        if self.hole== True:
            down= 10
        right = self.step*10
        pygame.draw.circle(screen,BLACK,(10+ right,10+ down),5,5)
        
        x_t1= pygame.surfarray.pixels3d(screen)
        x_t4= cv2.cvtColor(cv2.resize(x_t1, (80, 80)), cv2.COLOR_BGR2GRAY)
        ret, x_t4 = cv2.threshold(x_t4,1,255,cv2.THRESH_BINARY)                    
        state = np.stack((x_t4,x_t4), axis =2)
        
        

        if self.end==self.step and self.hole== True:
            terminal= True
            self.hole= False
            reward = 1
            self.step= 0

            

        return terminal, reward ,state


#debug code
##action1 = [1,0,0]
##action2 = [0,1,0]
##action3 = [0,0,1]
##game1 = Maze()
##game1.game(action1)
##game1.game(action1)
##game1.game(action1)
##game1.game(action1)
##game1.game(action1)
##game1.game(action1)
##game1.game(action2)
##
##game1.draw_frame()
###pygame.quit()



    
            
            
            
        
