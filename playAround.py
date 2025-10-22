import pygame
import numpy as np
import random

class Caustics:
    def __init__(self, width, height, resolution=2):
        self.resolution = resolution
        self.width = width//resolution + 2
        self.height = height//resolution + 2

        self.z = np.zeros([self.height, self.width])
        self.v = np.zeros([self.height, self.width])
        self.dx = np.zeros_like(self.z)
        self.dy = np.zeros_like(self.z)
        x = np.arange(0, self.width)
        y = np.arange(0, self.height)
        self.xv, self.yv = np.meshgrid(x, y)
        
        self.mass = 4.0 / resolution**2
        self.k = 3.0   # Middle ground between 2.0 and 4.0
        self.dt = 0.035  # Middle ground between 0.02 and 0.05
        
        self.screenWidth = width
        self.screenHeight = height
        self.zoom = resolution
        
        self.causticsSurface = pygame.Surface((width, height), pygame.SRCALPHA)

    def gauss(self, cx, cy, radius):
        y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
        
        mask = np.exp(-((x**2)+(y**2))/(radius*2))
        # Calculate the bounds
        yMin, yMax = max(0, cy-radius), min(self.height, cy+radius+1)
        xMin, xMax = max(0, cx-radius), min(self.width, cx+radius+1)
        # Apply mask to the right region (with bounds checking)
        maskHeight, maskWidth = yMax-yMin, xMax-xMin
        self.v[yMin:yMax, xMin:xMax] += 7.5 * mask[:maskHeight, :maskWidth]  # Middle: 7.5

    def photons(self, depth=1):
        # Determines the destinations of the photons from the 
        # surface gradient by calculating the difference between the height values
        # and the velocity of the water surface
        self.dx[:, :self.width-1] = self.z[:, 1:] - self.z[:, :self.width-1]
        self.dy[:self.height-1, :] = self.z[1:, :] - self.z[:self.height-1, :]
        px = self.xv - self.dx*depth
        py = self.yv - self.dy*depth
        return px, py

    def progress(self, N=4):
        # Solves Newton's equation numerically using the leap-frog algorithm
        for i in range(N):
            self.z += self.v*(self.dt/2)
            F = self.k*(self.z[1:self.height-1, 0:self.width-2] + 
                    self.z[1:self.height-1, 2:self.width] + 
                    self.z[0:self.height-2, 1:self.width-1] + 
                    self.z[2:self.height, 1:self.width-1] - 
                    4*self.z[1:self.height-1, 1:self.width-1])
            a = F/self.mass
            self.v[1:self.height-1, 1:self.width-1] += a*self.dt
            self.z += self.v*(self.dt/2)
            
            # Balanced friction
            self.v *= 0.997  # Middle ground between 0.995 and 0.999

    def updateCausticsSurface(self, depth=4, alpha=40):
        # Clears and constantly updates previous caustics
        self.causticsSurface.fill((0, 0, 0, 0))
        
        px, py = self.photons(depth=depth)
        
        spotSize = self.zoom - 2
        for col in range(1, self.height-1):
            for row in range(1, self.width-1):
                # Calculate screen coordinates
                xPos = int(px[col, row] * self.zoom)
                yPos = int(py[col, row] * self.zoom)
                
                # Ensure coordinates are within screen bounds
                if (0 <= xPos < self.screenWidth - spotSize and 
                    0 <= yPos < self.screenHeight - spotSize):
                    # Caustics simulated with squares
                    pygame.draw.rect(self.causticsSurface, (230, 252, 240, 30),
                                (xPos, yPos, spotSize, spotSize), 0)

        return self.causticsSurface

    def addRandomRipples(self):
        if not hasattr(self, 'rippleCounter'):
            self.rippleCounter = 0

        if random.random() < 0.035 and self.rippleCounter < 8:  # Middle ground
            x = random.randint(4, self.width-4)
            y = random.randint(4, self.height-4)
            self.gauss(x, y, 4)  # Middle size
            self.rippleCounter += 1


class WaterSimulation:
    def __init__(self, width=800, height=800):
        self.width = width
        self.height = height
        self.frameCounter = 0
        self.lastRandomRippleTime = 0
        
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Interactive Water Caustics Simulation")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 36)
        
        # Initialize caustics simulation
        self.causticsSim = Caustics(self.width, self.height, resolution=5)
        
        # Initialize with some gentle splashes
        self.causticsSim.gauss(30, 20, 6)
        self.causticsSim.gauss(50, 40, 5)
        
    def run(self):
        running = True
        mousePressed = False
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                        running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left click
                        mousePressed = True
                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1:
                        mousePressed = False
            
            # Create ripples when mouse is pressed
            if mousePressed:
                mx, my = pygame.mouse.get_pos()
                self.causticsSim.gauss(mx // self.causticsSim.zoom, 
                                      my // self.causticsSim.zoom, 5)
            
            # Update water simulation
            self.causticsSim.progress(4)  # Middle ground: 4 iterations
            
            # Handle random ripples with timing control
            currentTime = pygame.time.get_ticks()
            if currentTime - self.lastRandomRippleTime > 700:  # Middle: 700ms
                if random.random() < 0.075:  # Middle ground
                    self.causticsSim.addRandomRipples()
                    self.lastRandomRippleTime = currentTime
            
            # Clear screen with lighter blue water color
            self.screen.fill((80, 130, 160))  # Much lighter blue!
            
            # Update and draw caustics
            causticsSurface = self.causticsSim.updateCausticsSurface(depth=3, 
                                                                     alpha=10)
            self.screen.blit(causticsSurface.convert_alpha(), (0, 0))
            
            # Draw instructions
            helpText = "Click and drag to create ripples | Q or ESC to quit"
            textSurface = self.font.render(helpText, True, (255, 255, 255))
            textRect = textSurface.get_rect(center=(self.width//2, 30))
            
            # Semi-transparent background for text
            bgRect = pygame.Rect(textRect.left - 10, textRect.top - 5,
                                textRect.width + 20, textRect.height + 10)
            bgSurface = pygame.Surface((bgRect.width, bgRect.height), 
                                      pygame.SRCALPHA)
            bgSurface.fill((0, 0, 0, 128))
            
            self.screen.blit(bgSurface, bgRect)
            self.screen.blit(textSurface, textRect)
            
            pygame.display.flip()
            self.clock.tick(60)
        
        pygame.quit()


def main():
    app = WaterSimulation()
    app.run()


if __name__ == "__main__":
    main()