import numpy as np
from math import exp
import random
import pygame


class Caustics:
    def __init__(self, W_, H_, resol):
        self.resolution = resol
        self.width = W_*resol + 2
        self.height = H_*resol + 2

        self.z = np.zeros([self.height, self.width]) #height value in "rl world"
        self.v = np.zeros([self.height, self.width]) #velocity
        self.dx = np.zeros_like(self.z) #direction x
        self.dy = np.zeros_like(self.z) #direction y - x & y store surface 
        #gradient for light calculation
        x = np.arange(0, self.width) 
        y = np.arange(0, self.height)
        self.xv, self.yv = np.meshgrid(x, y) #grid coor
        # self.gauss(5*resol,5*resol,5*resol)
        # self.gauss(55*resol,25*resol,5*resol)
        # self.gauss(27*resol,30*resol,5*resol)
        
        self.mass = 4.0 / resol**2
        self.k = 4.0   / resol**2
        self.dt = 0.05

    def gauss(self, cx, cy, R):
        for x in range(-R, R+1):
            for y in range(-R, R+1):
                # Check bounds before accessing array
                if (0 <= cy+y < self.height) and (0 <= cx+x < self.width):
                    self.v[cy+y, cx+x] += 10*exp(-(x**2+y**2)/(R*2))

    def photons(self, depth=1): 
        #determines the destinations of the photons from the 
        #surface gradient by calculating the difference between the heightvalues
        #and the velocity of the water surface
        self.dx[:, :self.width-1] = self.z[:, 1:] - self.z[:, :self.width-1]
        self.dy[:self.height-1, :] = self.z[1:, :] - self.z[:self.height-1, :]
        px = self.xv - self.dx*depth
        py = self.yv - self.dy*depth
        return px, py

    def progress(self, N=4):
        #solves Newton's equation numerically using the leap-frog algorithm
        #the leap-frog algorithm is a numerical method for solving ordinary 
        # differential equations it is particularly useful for simulating 
        # physical systems, such as the motion of particles or fluids the 
        # algorithm alternates between updating the position and velocity of the 
        # particles
        for skips in range(N):
            
            self.z += self.v*(self.dt/2)
            F = self.k*(self.z[1:self.height-1, 0:self.width-2] + 
                        self.z[1:self.height-1, 2:self.width] + 
                        self.z[0:self.height-2, 1:self.width-1] + 
                        self.z[2:self.height, 1:self.width-1] - 
                        4*self.z[1:self.height-1, 1:self.width-1])
            a = F/self.mass #F = ma --> a = F/m
            self.v[1:self.height-1, 1:self.width-1] += a*self.dt
            self.z += self.v*(self.dt/2)

        #verify Newton's law of the conservation of energy
        # Ek = np.sum(v**2)*mass/2
        # Eph = (np.sum((z[1:self.H-1, 0:self.W-1] - 
        # z[1:self.H-1, 1:self.W])**2)*k/2)
        # Epv = (np.sum((z[0:self.H-1, 1:self.W-1] - 
        # z[1:self.H, 1:self.W-1])**2)*k/2)
        # Ep = Eph+Epv
        # print(Ep+Ek,Ep,Ek) #total energy


class CausticsPIL(Caustics):
    def animate(self, zoom=10):
        clock = pygame.time.Clock()
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # 1 == Left mouse button
                        mx, my = pygame.mouse.get_pos()
                        cx = int(mx / zoom)
                        cy = int(my / zoom)
                        # ripple at mouse position
                        if 4 <= cx < self.W-4 and 4 <= cy < self.H-4:
                            self.gauss(cx, cy, 4)

            #physics update
            self.progress(10)
            
            #simulation of firction
            self.v *= 0.999
            
            # add random drops occasionally 
            counter = 0
            if random.random() < 0.03 and counter < 8:  # 3% chance each frame
                x = random.randint(4, self.width-4)
                y = random.randint(4, self.height-4)
                self.gauss(x, y, 4)
                counter += 1
                #limit the waves to run 8 times max to avoid "hole" bugs
            
            # calculate photon projections
            px, py = self.photons(depth=2)
            
            # clear screen with blue background
            screen.fill((0, 120, 200))
            
            # draw caustics
            for w in range(self.width-1):
                for h in range(self.height-1):
                    # ensure projected coordinates are within screen bounds
                    x_pos = int(px[h, w] * zoom)
                    y_pos = int(py[h, w] * zoom)
                    
                    if 0 <= x_pos < 640-zoom and 0 <= y_pos < 360-zoom:
                        # draw white rectangle (photon)
                        

                        # transparent_surface = pygame.Surface((zoom-6, zoom-6), 
                        #                                      pygame.SRCALPHA)
                        # transparent_surface.fill((255, 255, 255, 60))  
                        # screen.blit(transparent_surface, (x_pos, y_pos))

                        pygame.draw.rect(screen, (180, 200, 220), 
                        (x_pos, y_pos, zoom-6, zoom-6), 0)

                        # pygame.draw.rect(screen, (255, 255, 255, 5), 
                        #                 (x_pos, y_pos, zoom-6, zoom-6), 0)
            
            pygame.display.flip()
            clock.tick(20)
        
        pygame.quit()



def main():
    global screen
    pygame.init()
    screen = pygame.display.set_mode((640, 360))

    #blurredSurface = pygame.transform.gaussian_blur(screen, radius=10)

    pygame.display.set_caption("Caustic Water Simulation")
    
    # 64x36 grid with resolution 2
    c = CausticsPIL(64, 36, 3)
    
    # initial ripples
    c.gauss(30, 20, 8)  # add an initial ripple
    
    # start animation
    c.animate(zoom=10)



main()