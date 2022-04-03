---
type: project-readme
---
# Perlin Noise 
Creation of perlin noise

## Particles example
![Particles in a perlin field](Demo.gif)

## Bare Perlin noise (2d with time coponent)
![2d gif example](3d_example.gif)

## Files
1. Perlin
    A simple generator of perlin noise
    Currently able to produce, 1d, 2d and 3d noise maps, attempting to
    generalse for n-dimentional noise
2. Perlin particles
    The more fun version, perlin noise vector field with particles added on
    top


## Improvments
1. Short term
    1. Generalise Perlin Noise
    2. Colours for particles -  Done [[2022-04-03]]
    3. Generally more variance in the particles for pretty pictures
2. Long term
    1. Simplex noise implementation
        1. Just generally better, scales well and appears to the eye more
        random 
    2. Numpy Particles implementation
        1. Similar to [[NumpyBoids]] a matrix implementation could be far
        faster