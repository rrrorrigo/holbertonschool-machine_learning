#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

y = np.arange(0, 11) ** 3

plt.plot(y, linestyle='-', color='r')
plt.xticks([0, 2, 4, 6, 8, 10])
plt.show()
