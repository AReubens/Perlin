#!/usr/bin/python
"""
Testbed for testing perlin related code
"""
# %%
import sys
import math
import numpy as np
 
import pygame
from pygame.locals import *

# My perlin works.... but it's slow in comparison to the C optimised lib
# from perlin import gen_perlin, gen_perlin_fract

# Use the super fast lib
import pyfastnoisesimd as fns


# %%
WIDTH = 2400
HEIGHT = 1600
SAVE_IMAGES = False

# Generating noise takes a lot of memory, don't need it for every pixel
# X & Y resolution of the perlin noise - lower values means computing a larger
# perlin array and so much slower.
NOISE_SCALE = 10  # X & Y resolution of the perlin noise
NOISE_ZSCALE = 10  # Larger is slower transition over time
NODES = 4  # Effectively the steepness of the noise

if sys.platform == 'win32':
    # On Windows, the monitor scaling can be set to something besides normal 100%.
    # PyScreeze and Pillow needs to account for this to make accurate screenshots.
    # TODO - How does macOS and Linux handle monitor scaling?
    import ctypes
    try:
       ctypes.windll.user32.SetProcessDPIAware()
    except AttributeError:
        pass # Windows XP doesn't support monitor scaling, so just do nothing.

# %%
class Particle():
    def __init__(self, x, y) -> None:
        self.x = x
        self.y = y
        self.u = (np.random.rand() - 0.5) * 10
        self.v = (np.random.rand() - 0.5) * 10
        self.au = 0
        self.av = 0
        self.mass = 1
        self.max_speed = 100
        self.path = [[x], [y]]
        self.color = np.random.rand(3) * 255
        # self.color = (255, 255, 255, 200)
        return
    
    def apply_force(self, fx, fy) -> None:
        self.au += fx / self.mass
        self.av += fy / self.mass
        return
    
    def limit_speed(self, constant=False) -> None:
        n = math.sqrt(self.u**2 + self.v**2)
        if constant:
            f = self.max_speed / n
        else:
            f = min(n, self.max_speed) / n
        self.u *= f
        self.v *= f
        return

    def update(self, timestep, width, height) -> None:
        
        self.u += self.au * timestep
        self.v += self.av * timestep

        self.limit_speed(constant=True)
        
        self.x += self.u * timestep
        self.y += self.v * timestep

        self.au = 0
        self.av = 0

        self.x %= width
        self.y %= height

        self.path[0].append(self.x)
        self.path[1].append(self.y)
        return

 
def update(dt, particles, vecs_x, vecs_y, count):
    """
    x += v * dt
    """
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit() 
            sys.exit()
    
    z = int(count // NOISE_ZSCALE) 
    for p in particles:
        x = int(p.x // NOISE_SCALE)
        y = int(p.y // NOISE_SCALE)
        p.apply_force(vecs_x[x, y, z], vecs_y[x, y, z])
        p.update(1/dt, WIDTH, HEIGHT)

 
def draw(screen, particles, count):
    """
    Draw things to the window. Called once per frame.
    """
    for p in particles:
        pygame.draw.circle(
            screen, 
            p.color, 
            (p.x, p.y), 
            2
        )
    pygame.display.flip()
 

def runPyGame():
    pygame.init()

    fps = 45.0
    duration = 30
    width, height = WIDTH, HEIGHT
    field_strength = 100
    num_particles = 2000
    tail_length = 100
    # tail length is measured in frames (so correlates to max speed at
    # the moment)

    # Set up the window.
    screen = pygame.display.set_mode((width, height))
    fpsClock = pygame.time.Clock()
    screen.fill((255, 255, 255))

    # Set up particles:
    pos = np.random.rand(num_particles, 2)
    pos[:, 0] = pos[:, 0] * width
    pos[:, 1] = pos[:, 1] * height
    particles = [Particle(co[0], co[1]) for co in pos]


    # Set up the perlin noise field:
    print("Generating Perlin...")

    # pyfastnoisesimd use:
    perlin = fns.Noise(numWorkers=10)
    perlin.frequency = NODES / width * NOISE_SCALE
    perlin.noiseType = fns.NoiseType.Perlin
    perlin.fractal.octaves = 1
    perlin.fractal.lacunarity = 2.1
    perlin.fractal.gain = 0.45
    perlin.perturb.perturbType = fns.PerturbType.NoPerturb
    vecs = perlin.genAsGrid([
        width // NOISE_SCALE,
        height // NOISE_SCALE,
        int(fps*duration // NOISE_ZSCALE)
    ])
    vecs = vecs * 2 * np.pi * 2  # Extra times 2 helps to randomise the vectors
    

    vecs_x = np.cos(vecs) * field_strength
    vecs_y = np.sin(vecs) * field_strength


    print("Starting main loop")
    # Main game loop.
    dt = 1/fps
    count = 0
    increasing = True
    tot = 0
    while True:
        update(dt, particles, vecs_x, vecs_y, count)

        if False:  # Show the underlying Perlin noise
            z = int(count // NOISE_ZSCALE)
            surf = pygame.surfarray.make_surface(vecs[:, :, z])
            surf = pygame.transform.scale(surf, np.array(vecs.shape[:2])
                * NOISE_SCALE)
            screen.blit(surf, (0, 0))
        draw(screen, particles, count)
        tot += 255
        if tot // tail_length > 1:
            surface = screen.convert_alpha()
            surface.fill((255, 255, 255, tot // tail_length))
            screen.blit(surface, (0,0))
            tot = tot % 255
        elif tot // tail_length == 1:
            tot = tot % 255
            surface = screen.convert_alpha()
            surface.fill((255, 255, 255, 1))
            screen.blit(surface, (0,0))
        
        dt = fpsClock.tick(fps)

        # For saving the image files for later complation as video:
        if SAVE_IMAGES:
            file_name = f"tmp/{count:04d}.png"
            pygame.image.save(screen, file_name)

        # Forward and rewind over the noise for a smooth transition at end
        if increasing: 
            if count == vecs.shape[2] * NOISE_ZSCALE - 1:
                increasing = False
                # While the sim will still be unique after here, since the
                # starting points of the particles is unique, this reaches the
                # end of the set duration, so stop saving the images to file
                if SAVE_IMAGES:
                    break  
                count -= 1
            else:
                count += 1
        else: 
            if count == 0:
                increasing = True
                count += 1 
            else:
                count -= 1

runPyGame()

print('Complete')  # Only here if we're saving images
