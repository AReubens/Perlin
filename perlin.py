#!/usr/bin/python

"""
The basis of this perlin noise code has been shamlessly nabbed and
influenced by two main sources:
- tgirod on stack overflow:
    https://stackoverflow.com/questions/42147776/producing-2d-perlin-noise-with-numpy
- Unkown's (?) blog:
    https://gpfault.net/posts/perlin-noise.txt.html


Basic Perlin noise generator (or a super basic implementation)
This is in the works to develop a general n-dimentional function for
generation. I should also add a wrapper to generate fractal noise.

This whole process has been (for what I can find) been superseeded by 
simplex noise however.
From what I gather simplex noise uses equerlateral triangles rather 
than quares to make things look more random. (At high frequencies with
perlin noise 90 degree boundaries are very obvious.)

I was intending to use this for my own perlin noise field generation, but 
this is so much slower than the PyFastNoiseSIMD implementation
"""

# %%
# imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

# %%
def gen_perlin(*axes, seed=None):
    '''
    This is nowhere near general just yet, but it's in the process of getting
    there...
    '''
    # Grad general:
    def gen_grad(h, *float_parts):
        '''
        Like the main fucntion, not at all general just yet.
        '''
        dims = len(float_parts)
        vectors = np.concatenate((np.eye(dims), -np.eye(dims)))
        g = vectors[h % (dims*2)]
        # this can be vastly improved...
        # if nothing else than with a switch case
        # Althought might not be more computationally efficient
        if dims == 1:
            return (
                g[:, 0] * float_parts[0]
            )
        if dims == 2:
            return (
                g[:, :, 0] * float_parts[0] +
                g[:, :, 1] * float_parts[1]
            )
        if dims == 3:
            return (
                g[:, :, :, 0] * float_parts[0] + 
                g[:, :, :, 1] * float_parts[1] + 
                g[:, :, :, 2] * float_parts[2]
            )
        else:
            print("Get fucked")
            return 1

    # lerp is general already:
    def lerp(a, b, x):
        return a + x * (b-a)

    # as is fade:
    def fade(t):
        return 6 * t**5 - 15 * t**4 + 10 * t**3

    dims = len(axes)
    shape = axes[0].shape  # because of the mesh grid input, this works

    # permutation table
    if seed:
        np.random.seed(0)
    p = np.arange(256,dtype=int)
    p = np.repeat(p, dims)
    np.random.shuffle(p)

    # I think that I might be able to conver the axes list into a numpy array
    # to covnert the folling list comprehensions into just one numpy call.

    ai = [axis.astype(int) for axis in axes]  # Axes int 
    af = [axes[i] - ai[i] for i in range(dims)]  # Axes float
    axes_fades = [fade(axis_float) for axis_float in af]  # fade factors

    # this is the complicated bit....
    # I think keeping grads as a linear array (kinda) is better than assigning
    # making further dimensions for each corner. Maybe.
    # modulo can then be used to work out which one's need to be lerped
    # together.
    # This might also help with the grad function.
    if dims == 1:
        # preallocate to help see the pattern that's going on here:
        grads = np.zeros((dims*2, shape[0]))
        grads[0] = gen_grad(p[ai[0]]  , af[0]  )
        grads[1] = gen_grad(p[ai[0]+1], af[0]-1)

        x1 = lerp(grads[0], grads[1], axes_fades[0])
        return x1
    if dims == 2:
        # preallocate to help see the pattern that's going on here:
        grads = np.zeros((dims*2, shape[0], shape[1]))
        grads[0] = gen_grad(p[p[ai[0]]+ai[1]]    , af[0]  , af[1]  )
        grads[1] = gen_grad(p[p[ai[0]]+ai[1]+1]  , af[0]  , af[1]-1)
        grads[2] = gen_grad(p[p[ai[0]+1]+ai[1]+1], af[0]-1, af[1]-1)
        grads[3] = gen_grad(p[p[ai[0]+1]+ai[1]]  , af[0]-1, af[1]  )

        x1 = lerp(grads[0], grads[3], axes_fades[0])
        x2 = lerp(grads[1], grads[2], axes_fades[0])
        y  = lerp(x1, x2, axes_fades[1])
        return y
    if dims == 3:
        grads = np.zeros((2**dims, shape[0], shape[1], shape[2]))
        grads[0] = gen_grad(p[p[p[ai[0]]+ai[1]]+ai[2]]      , af[0]  , af[1]  , af[2]  )    # 0, 0, 0
        grads[1] = gen_grad(p[p[p[ai[0]+1]+ai[1]]+ai[2]]    , af[0]-1, af[1]  , af[2]  )    # 1, 0, 0
        grads[2] = gen_grad(p[p[p[ai[0]+1]+ai[1]+1]+ai[2]]  , af[0]-1, af[1]-1, af[2]  )    # 1, 1, 0
        grads[3] = gen_grad(p[p[p[ai[0]]+ai[1]+1]+ai[2]]    , af[0]  , af[1]-1, af[2]  )    # 0, 1, 0
        grads[4] = gen_grad(p[p[p[ai[0]]+ai[1]]+ai[2]+1]    , af[0]  , af[1]  , af[2]-1)    # 0, 0, 1
        grads[5] = gen_grad(p[p[p[ai[0]+1]+ai[1]]+ai[2]+1]  , af[0]-1, af[1]  , af[2]-1)    # 1, 0, 1
        grads[6] = gen_grad(p[p[p[ai[0]+1]+ai[1]+1]+ai[2]+1], af[0]-1, af[1]-1, af[2]-1)    # 1, 1, 1
        grads[7] = gen_grad(p[p[p[ai[0]]+ai[1]+1]+ai[2]+1]  , af[0]  , af[1]-1, af[2]-1)    # 0, 1, 1

        x1 = lerp(grads[0], grads[1], axes_fades[0])    # 0-1, 0,   0
        x2 = lerp(grads[3], grads[2], axes_fades[0])    # 0-1, 1,   0
        x3 = lerp(grads[4], grads[5], axes_fades[0])    # 0-1, 0,   1
        x4 = lerp(grads[7], grads[6], axes_fades[0])    # 0-1, 1,   1
        y1 = lerp(x1, x2, axes_fades[1])                # ~,   0-1, 0
        y2 = lerp(x3, x4, axes_fades[1])                # ~,   0-1, 1
        z = lerp(y1, y2, axes_fades[2])                 # ~,   ~,   0-1
        return z

    else: 
        print("get fucked")
        return 1


def gen_perlin_fract(shape, octaves=[2], divisor=1/2, z_frequency = 4):
    """
    Generate Perlin fractal noise with multiple octaves
    
    For the moment uses the same octaves for both x and y,
    but this can be inproved later to allow for different x and y octaves,
    althought it'll look werid... I think
    """
    out = gen_perlin(*np.meshgrid(
        np.linspace(0, octaves[0], shape[0], endpoint=False),
        np.linspace(0, octaves[0], shape[1], endpoint=False),
        np.linspace(0, z_frequency, shape[2], endpoint=False)
    ))
    for i in octaves[1:]:
        out += gen_perlin(*np.meshgrid(
            np.linspace(0, i, shape[0], endpoint=False),
            np.linspace(0, i, shape[1], endpoint=False),
            np.linspace(0, z_frequency, shape[2], endpoint=False)
        )) * (divisor)**i
    return out


# %%
if __name__ == '__main__':
    time = 10
    frame_count = time * 30


    print("Generating perlin...")
    waves = gen_perlin_fract(
        (100, 100, frame_count),
        [3, 6]
    )

    print('Displaying')
    fig, ax = plt.subplots()
    im = plt.imshow(waves[:, :, 0])
    ax.set_axis_off()

    def update(frame):
        im.set_data(waves[:, :, frame])
        return [im]


    def init():
        im.set_data(waves[:, :, 0])
        return [im]


    ani = animation.FuncAnimation(
        fig, update, frames=frame_count, interval=1000/30, blit=False)
    
    f = r"3d_example.gif" 
    writergif = animation.PillowWriter(fps=30) 
    ani.save(f, writer=writergif)

    plt.show()
