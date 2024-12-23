import sys
sys.path.append('..')
from optimism.JaxConfig import *
from matplotlib import pyplot as plt

import numpy as onp



plt.axis('equal')
plt.savefig('mesh.png')
plt.clf()
#plt.plot(tss, dists, 'r')
#plt.plot(tss, dists_fit, 'g')
plt.savefig('dists.png')