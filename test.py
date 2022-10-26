from enum import auto
import numpy as np
import matplotlib.pyplot as plt

inliers = np.load("inliers.npy")
hist, bins = np.histogram(inliers)

plt.hist(inliers, bins="auto")
plt.show()
plt.ylabel("Number of Points with distance in the given range")
plt.xlabel("Pair-wise distance")
