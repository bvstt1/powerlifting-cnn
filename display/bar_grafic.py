import numpy as np
import matplotlib.pyplot as plt

bar = np.load("../processed/sq/sq_004/front_bar.npy")

y = bar[:, 1]   # altura de la barra

plt.plot(y)
plt.xlabel("Frame")
plt.ylabel("Altura barra")
plt.title("Movimiento vertical de la barra")
plt.show()