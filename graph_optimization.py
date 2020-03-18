import sys
sys.path.insert(0, '..')

import d2l
from mpl_toolkits import mplot3d
import numpy as np 


def f(x):
	return x * np.cos(np.pi * x)

d2l.set_figsize((4.5, 2.5))
x = np.arange(-1.0,2.0,0.1)
fig,  = d2l.plt.plot(x,f(x))
fig.axes.annotate('local minimum', xy=(-0.3,-0.25), xytext=(-0.77,-1.0),
					arrowpros = dict(arrowstyle='->'))
d2l.plt.xlabel('x')