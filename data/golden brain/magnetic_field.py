import numpy as np
import scipy.integrate as integrate
from scipy.integrate import quad
import seaborn as sns
import matplotlib.pyplot as plt

def generate_B0_fieldmap(nx,ny, a=1, ratio=0.7):
    """
    It calculates the field map of a circular coil with radius $a$ and a circular coil with radius $r$ where $r<a$. The
    field map is calculated by integrating the magnetic field of the circular coil with radius $a$ over the circular coil
    with radius $r$

    :param nx: number of pixels in x direction
    :param ny: number of pixels in y direction
    :param a: radius of the cylinder, defaults to 1 (optional)
    :param ratio: the ratio of the radius of the circle to the radius of the square
    :return: a 2D array of the same size as the input array. The array is a map of the magnetic field strength.
    """
    # nx = 50
    # ny = 50
    # a = 1
    # ratio = 0.7
    xs = np.linspace(-1*ratio*a/np.sqrt(2),ratio*a/np.sqrt(2),nx)
    ys = np.linspace(-1*ratio*a/np.sqrt(2),ratio*a/np.sqrt(2),ny)


    def integrand(x,a, r):
        k = (2 * np.sqrt(r * a)) / (r + a)
        intg = ((a**2)-(2*r*a*(np.sin(x)**2)+(r*a))/(np.power(1-(np.power(k*np.sin(x),2)),1.5)))
        return intg

    I  = np.zeros((nx,ny))
    for ix, x in enumerate(xs):
        for iy, y in enumerate(ys):
            r = np.sqrt((x**2)+(y**2))
            I[ix,iy]=(quad(integrand, 0, np.pi/2, args=(a,r)))[0]

    # plt.imshow(I/np.max(np.abs(I)),aspect="auto")
    # plt.colorbar()
    # plt.show()
    return 1+(I/np.max(np.abs(I)))

def generate_linear_fieldmap(nx,ny, a=1, ratio=0.7):
    xs = np.linspace(-1*ratio*a/np.sqrt(2),ratio*a/np.sqrt(2),nx)
    ys = np.linspace(-1*ratio*a/np.sqrt(2),ratio*a/np.sqrt(2),ny)
    I  = np.zeros((nx,ny))
    for ix, x in enumerate(xs):
        for iy, y in enumerate(ys):
            r = np.sqrt((x ** 2) + (y ** 2))
            I[ix,iy]=1/(1+r)

    # plt.imshow(I,aspect='auto')
    # plt.colorbar()
    # plt.show()
    return I

def generate_egg_fieldmap(nx,ny, a=1, ratio=0.7):
    xs = np.linspace(-1*ratio*a/np.sqrt(2),ratio*a/np.sqrt(2),nx)
    ys = np.linspace(-1*ratio*a/np.sqrt(2),ratio*a/np.sqrt(2),ny)
    I  = np.zeros((nx,ny))
    rnd = np.random.rand()
    for ix, x in enumerate(xs):
        for iy, y in enumerate(ys):
            # r = np.sqrt((x ** 2) + (y ** 2))
            I[ix,iy]= (np.sin((x+rnd)*10) + np.cos((y+rnd)*10))

    # plt.imshow((1+(I/np.max(np.abs(I))))/2,aspect='auto')
    # plt.colorbar()
    # plt.show()
    return (1+(I/np.max(np.abs(I))))/2

def generate_fieldmap(nx,ny, a=1, ratio=0.7):
    I1 = generate_B0_fieldmap(nx,ny)
    I2 = generate_egg_fieldmap(nx,ny)
    I = I1-0.4*I2
    # plt.imshow(I,aspect='auto')
    # plt.colorbar()
    # plt.show()
    return I

def B2hz(I,fmin,fmax):
    bmin = np.min(I)
    bmax = np.max(I)
    a = (fmax-fmin)/(bmax-bmin)
    b = ((fmin*bmax)-(fmax*bmin))/(bmax-bmin)
    f = a*I + b
    # plt.imshow(f,aspect='auto')
    # plt.colorbar()
    # plt.show()
    return f

# I = generate_egg_fieldmap(50,50, a=1, ratio=0.7)
# B2hz(I,-10,10)