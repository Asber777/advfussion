
from numpy import arange, sin, pi, cos
import matplotlib.pyplot as plt
import cmath
from scipy.fftpack import fft,ifft

y = [[1,2,3,4],
[5,6,7,8],
[9,10,11,12],
[13,14,15,16]]

yy = fft(y)
print(yy[0][0])