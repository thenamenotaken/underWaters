import pygame
import numpy as np
#hello UwU test testtestttt

width, height = 800, 800
cols, rows = width, height
dampening = 0.998

class WaterRippleSimulation:
    def __init__(self, width=800, height=800, dampening=0.998):
        self.width = width
        self.height = height
        self.dampening = dampening

        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        self.clock = pygame.time.Clock()
        pygame.display.set_caption("2D Water Ripples")
        
        #32 bits for efficiency purpoeses
        self.current = np.zeros((height, width), dtype=np.float32)
        self.previous = np.zeros((height, width), dtype=np.float32)
        self.surfaceArray = np.zeros((height, width, 3), dtype=np.uint8)

        self.running = True
    
    def eventHandler(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT or pygame.key.get_pressed()[pygame.K_x]:
                self.running = False 

            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_x:
                    self.running = False


        if pygame.mouse.get_pressed()[0]:
            mouseX, mouseY = pygame.mouse.get_pos()
            if (1 < mouseX < self.width - 1) and (1 < mouseY < self.height - 1):
                for row in range(3):
                    for col in range(3):
                        self.previous[(mouseX - 1) + row, (mouseY-1)+col] = 2500

    def update(self):
         # Calculate ripple effect using vectorized operations
        self.current[1:-1, 1:-1] = (
            (self.previous[0:-2, 1:-1] +  # left
            self.previous[2:, 1:-1] +    # right
            self.previous[1:-1, 0:-2] +  # top (next line has bottom)
            self.previous[1:-1, 2:]) /2 - self.current[1:-1, 1:-1] 
        )
        #It looks REALLY Cool when you divided it by 1.99 instead of 2
        #Also the cool thing about this operation is it's using under the hood C
        #To add "matrices" and divide them
        #To add this is basically a discrete physics wave equation
        #    ∂²h/∂t² = c²(∂²h/∂x² + ∂²h/∂y²)

        # Apply dampening
        self.current *= self.dampening
        
        # Clip values to 0-255 range - think of it as mapping to a difference 
        # range in p5.js
        np.clip(self.current, 0, 255, out=self.current)

        self.previous, self.current = self.current, self.previous
        
    def render(self):
        for c in range(3):
            self.surfaceArray[:, :, c] = self.current

        pygameSurface = pygame.surfarray.make_surface(self.surfaceArray)
        self.screen.blit(pygameSurface, (0, 0))
        pygame.display.flip()

    
    def run(self):
        while self.running:
            self.eventHandler()
            self.update()
            self.render()
            self.clock.tick(70)

        pygame.quit()

def main():
    simulation = WaterRippleSimulation(800, 800, 0.988)
    simulation.run()

main()