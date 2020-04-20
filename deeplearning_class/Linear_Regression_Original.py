import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
"""
data = np.array([[100, 20], [150, 24], [300, 36], [400, 47], [130, 22], [240, 32], [350, 47],
                 [200, 42], [100, 21], [110, 21], [190, 30], [120, 25], [130, 18], [270, 38],
                 [255, 28]]).astype('float32')
data = np.array([[13, 40], [19, 83], [16, 62], [14, 48], [15, 58], [14, 43]]).astype('float32')
data = np.array([[2, 81], [4, 93], [6, 91], [8, 97]])
"""

np.random.seed(3)

data = np.array([[100, 20], [150, 24], [300, 36], [400, 47], [130, 22], [240, 32], [350, 47],
                 [200, 42], [100, 21], [110, 21], [190, 30], [120, 25], [130, 18], [270, 38],
                 [255, 28]]).astype('float32')
x1 = data[:, 0]
y1 = data[:, 1]

plt.scatter(x1, y1)
plt.show()

x_data = np.array(x1)
t_data = np.array(y1)

w = np.random.normal(size=1)
b = np.random.normal(size=1)
epochs = 2001
lr = 0.00001

for i in range(epochs):
    y = w * x_data + b
    cost = t_data - y
    error = sum(cost ** 2) / len(x_data)

    w_diff = -(1 / len(x_data)) * sum(x_data * cost)
    b_diff = -(1 / len(x_data)) * sum(t_data - y)
    w = w - lr * w_diff
    b = b - lr * b_diff
    if i % 100 == 0:
        print("epoch: %.f, MSE : %.04f, w : %.04f, b : %.04f" % (i, error, w, b))

y_predict = w * x_data + b
plt.scatter(x1, y1)
plt.plot([min(x_data), max(x_data)], [min(y_predict), max(y_predict)])
plt.show()

print("w=", w)
print("b=", b)
