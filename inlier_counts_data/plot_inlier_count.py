import numpy as np
import matplotlib.pyplot as plt

a = np.load("source_inlier_count.npy")
b = np.load("target_inlier_count.npy")
n = np.linspace(1, 99, 99)

plt.plot(n, a, label="source inlier count")
plt.plot(n, b, label="target inlier count")
plt.legend()

plt.grid(visible=True)
plt.show()