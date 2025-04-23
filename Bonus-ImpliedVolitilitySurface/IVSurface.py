import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

s = 100

#50 strike prices surrounding the strike price
K = np.linspace(70, 130, 50)
#50 time to expiration dates from 0.1 years to 2 years
T = np.linspace(0.1, 2.5, 50)

K_grid, T_grid = np.meshgrid(K,T) #Returns every cartesian index of K and T

M_grid = K_grid/ s
iv_surface = 0.2 + 0.3 * np.exp(-((M_grid - 1) / 0.1) ** 2) + 0.1 * np.sqrt(T_grid)

fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111, projection ='3d')
surf = ax.plot_surface(
    M_grid, T_grid, iv_surface,
    cmap='jet', 
    edgecolor='none',
    antialiased=True,
    rstride=1, cstride=1
)

# Aesthetics to match the image
ax.set_title('Implied Volatility Surface', fontsize=14)
ax.set_xlabel('Moneyness M = K/S', fontsize=12)
ax.set_ylabel('Time to Maturity T', fontsize=12)
ax.set_zlabel('Implied Volatility σ(M, T)', fontsize=12)

ax.view_init(elev=30, azim=120)  # Set camera angle
fig.colorbar(surf, shrink=0.6, aspect=10)

plt.tight_layout()
plt.show()

