
from qutipDots import *
from numpy import exp
from qutip import Qobj
import matplotlib.pyplot as plt

# Load data
filename = '1-1.0.npy'
data = np.load('data/' + filename)

# plot eigenenergies (3d, each alpha is a dimension)
energies = data[:, 3:]
alphas = data[:, :3]
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(alphas[:, 0], alphas[:, 1], alphas[:, 2], c=energies[:, 0])
ax.set_xlabel('alpha_12')
ax.set_ylabel('alpha_23')
ax.set_zlabel('alpha_13')
plt.show()

