# RUN THIS FILE TO START THE SIMULATION

import cv2
import pygame
import numpy as np
import sys
import math  
import random
import mediapipe as mp

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
        self.k = 4.0   
        self.dt = 0.05
        
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
        self.v[yMin:yMax, xMin:xMax] += 10 * mask[:maskHeight, :maskWidth]

    def photons(self, depth=1):
        self.dx[:, :self.width-1] = self.z[:, 1:] - self.z[:, :self.width-1]
        self.dy[:self.height-1, :] = self.z[1:, :] - self.z[:self.height-1, :]
        px = self.xv - self.dx*depth
        py = self.yv - self.dy*depth
        return px, py

    def progress(self, N=4):
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
            
            # friction
            self.v *= 0.999

    def updateCausticsSurface(self, depth=4, alpha=40):
        self.causticsSurface.fill((0, 0, 0, 0))
        
        px, py = self.photons(depth=depth)
        
        spotSize = self.zoom - 2
        for col in range(1, self.height-1):
            for row in range(1, self.width-1):
                xPos = int(px[col, row] * self.zoom)
                yPos = int(py[col, row] * self.zoom)
                
                if (0 <= xPos < self.screenWidth - spotSize and 
                    0 <= yPos < self.screenHeight - spotSize):
                    pygame.draw.rect(self.causticsSurface, (230, 252, 240, 30),
                                (xPos, yPos, spotSize, spotSize), 0)

        return self.causticsSurface

    def addRandomRipples(self):
        if not hasattr(self, 'rippleCounter'):
            self.rippleCounter = 0

        if random.random() < 0.05 and self.rippleCounter < 8:
            x = random.randint(4, self.width-4)
            y = random.randint(4, self.height-4)
            self.gauss(x, y, 4)
            self.rippleCounter += 1


class WaterRippleSimulation:
    def __init__(self, width, height, dampening=0.95):
        self.width = width
        self.height = height
        self.dampening = dampening
        
        self.current = np.zeros((height, width), dtype=np.float32)
        self.previous = np.zeros((height, width), dtype=np.float32)
    
    def addSplash(self, x, y, radius=3, strength=255):
        if not (radius < x < self.width-radius and radius < y < 
                self.height-radius):
            return
        
        yMin, yMax = max(0, y-radius), min(self.height, y+radius+1)
        xMin, xMax = max(0, x-radius), min(self.width, x+radius+1)
        self.previous[yMin:yMax, xMin:xMax] = strength 
    
    def update(self):
        self.current[1:-1, 1:-1] = (
            (self.previous[0:-2, 1:-1] +
             self.previous[2:, 1:-1] +
             self.previous[1:-1, 0:-2] +
             self.previous[1:-1, 2:]) * 0.5
        ) - self.current[1:-1, 1:-1]
        
        self.current *= self.dampening
        
        self.previous, self.current = self.current, self.previous
    
    def applyToImage(self, image):
        blueTint = np.zeros_like(image)
        blueTint[:, :, 0] = 200
        blueTint[:, :, 1] = 100
        blueTint[:, :, 2] = 120
        
        alpha = 0.2
        tintedImage = cv2.addWeighted(image, 1 - alpha, blueTint, alpha, 0)
        
        wave = np.clip(self.current, 0, 255).reshape(self.height, self.width, 1)
        brightness = np.repeat(wave, 3, axis=2) * 1.0
        
        result = np.clip(tintedImage + brightness, 0, 255).astype(np.uint8)
        resultWithFlow = self.rowDisplacement(result)
        
        return resultWithFlow

    def rowDisplacement(self, image):
        result = image.copy()
        
        if not hasattr(self, 'rowOffSets'):
            self.rowOffsets = np.zeros(image.shape[0], dtype=np.float32)
        
        self.rowOffsets += np.random.uniform(-1, 1, size=image.shape[0])
        mask = np.abs(self.rowOffsets) > 5
        self.rowOffsets[mask] *= 0.9
        
        offsets = self.rowOffsets.astype(np.int32)
        for row in range(image.shape[0]):
            offset = offsets[row]
            if offset != 0:
                result[row:row+1, :] = np.roll(image[row:row+1, :], 
                                               offset, axis=1)
        
        return result


class IntegratedWaterSimulation:
    def __init__(self):
        self.width = 640
        self.height = 480
        self.frameCounter = 0
        self.lastRandomRippleTime = 0
        
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Interactable Water Simulation")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 24)
        
        # Initialize webcam
        self.capture = cv2.VideoCapture(0)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        
        # Initialize MediaPipe hands
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mpDraw = mp.solutions.drawing_utils
        
        # Initialize simulations
        self.waterSim = WaterRippleSimulation(self.width, self.height)
        self.causticsSim = Caustics(self.width, self.height, resolution=5)
        
        # Initialize caustics simulation with splashes
        self.causticsSim.gauss(30, 20, 8)
        
        # Hand landmark indices
        self.THUMB_TIP = 4
        self.INDEX_TIP = 8
        self.RING_TIP = 16
        self.WRIST = 0
        self.THUMB_CMC = 1
    
    def ptpDistance(self, x1, y1, x2, y2):
        return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    
    def run(self):
        running = True
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif ((event.type == pygame.KEYDOWN) and    
                    (event.key == pygame.K_q)):
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        mx, my = pygame.mouse.get_pos()
                        self.waterSim.addSplash(mx, my, radius=5)
                        self.causticsSim.gauss(mx // self.causticsSim.zoom, my 
                                               // self.causticsSim.zoom, 6)

            # Get webcam frame
            ret, frame = self.capture.read()
            if not ret:
                continue
                
            frame = cv2.flip(frame, 1)
            frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process hand detection
            results = self.hands.process(frameRGB)
            
            # Handle hand gestures
            if results.multi_hand_landmarks:
                for handLandmarks in results.multi_hand_landmarks:
                    # Get hand landmarks
                    thumbTip = handLandmarks.landmark[self.THUMB_TIP]
                    indexTip = handLandmarks.landmark[self.INDEX_TIP]
                    ringTip = handLandmarks.landmark[self.RING_TIP]
                    wrist = handLandmarks.landmark[self.WRIST]
                    thumbCmc = handLandmarks.landmark[self.THUMB_CMC]
                    
                    # Convert to pixel coordinates
                    tx, ty = int(thumbTip.x * self.width), int(thumbTip.y * self.height)
                    x, y = int(indexTip.x * self.width), int(indexTip.y * self.height)
                    ringY = int(ringTip.y * self.height)
                    wristX = int(wrist.x * self.width)
                    thumbBaseX, thumbBaseY = int(thumbCmc.x * self.width), int(thumbCmc.y * self.height)
                    
                    # Draw circles at fingertips
                    cv2.circle(frame, (x, y), 5, (255, 255, 255), -1)
                    cv2.circle(frame, (tx, ty), 5, (255, 255, 255), -1)
                    
                    # Pinch gesture detection
                    distance = self.ptpDistance(tx, ty, x, y)
                    if distance < 30:
                        midpointX = (tx + x) // 2
                        midpointY = (ty + y) // 2
                        
                        self.waterSim.addSplash(midpointX, midpointY, radius=8)
                        self.causticsSim.gauss(midpointX // self.causticsSim.zoom, 
                                              midpointY // self.causticsSim.zoom, 8)
                    
                    # Fist detection & rain effect
                    dist1 = self.ptpDistance(thumbBaseX, thumbBaseY, x, y)
                    dist2 = self.ptpDistance(wristX, ringY, wristX, ringY)
                    
                    if dist1 <= 30 and dist2 <= 30:
                        self.createRainRipples()
            
            # Update simulations
            self.waterSim.update()
            self.causticsSim.progress(5)
            
            # Handle random ripples with timing control
            currentTime = pygame.time.get_ticks()
            if currentTime - self.lastRandomRippleTime > 500: 
                if random.random() < 0.1:
                    self.causticsSim.addRandomRipples()
                    self.lastRandomRippleTime = currentTime
            
            # Apply water effect to the frame
            frame = self.waterSim.applyToImage(frame)
            
            # Update and get caustics surface
            causticsSurface = self.causticsSim.updateCausticsSurface(depth=3, 
                                                                     alpha=10)
            
            # Clear screen
            self.screen.fill((0, 0, 0))

            # Draw webcam 
            frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pygameSurface = pygame.surfarray.make_surface(
                frameRGB.transpose(1, 0, 2))
            self.screen.blit(pygameSurface, (0, 0))

            # Draw caustics on top
            self.screen.blit(causticsSurface.convert_alpha(), (0, 0))

            # Draw help text
            self.drawInteractionHelp()

            pygame.display.flip()
            self.clock.tick(60)
        
        # Cleanup
        if self.capture is not None:
            self.capture.release()
        if self.hands is not None:
            self.hands.close()
        pygame.quit()
        sys.exit()

    def drawInteractionHelp(self):
        helpText = "Pinch to create splashes | Make a fist for rain effect | Press Q to quit"
        
        textSurface = self.font.render(helpText, True, (255, 255, 255))
        textRect = textSurface.get_rect(center=(self.width//2, 
                                                self.height - 20))
        
        bgRect = pygame.Rect(textRect.left - 10, textRect.top - 5, 
                              textRect.width + 20, textRect.height + 10)
        bgSurface = pygame.Surface((bgRect.width, bgRect.height), 
                                   pygame.SRCALPHA)
        bgSurface.fill((0, 0, 0, 128))
        
        self.screen.blit(bgSurface, bgRect)
        self.screen.blit(textSurface, textRect)

    def createRainRipples(self):
        for i in range(3):
            x = random.randint(50, self.width - 50)
            y = random.randint(50, self.height - 50)
            
            # Add splashes to both ripple and caustics simulations
            self.waterSim.addSplash(x, y, radius=random.randint(3, 7))
            self.causticsSim.gauss(x // self.causticsSim.zoom, 
                                   y // self.causticsSim.zoom, 
                                   random.randint(4, 8))


def main():
    app = IntegratedWaterSimulation()
    app.run()

if __name__ == "__main__":
    main()