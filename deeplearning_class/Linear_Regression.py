import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.e ** (-x))


def create_bias(x):
    bias = np.zeros(len(x))
    bias = np.random.normal(bias)
    return bias


def create_weight(x):
    weight = np.zeros_like(x)
    weight = np.random.normal(weight)
    return weight


def hidden_layer(x, weight, bias):
    hypothesis = weight * x + bias
    return hypothesis


def cost_function(hypothesis, y):
    cost_v = np.sum((y - hypothesis) ** 2) / len(y)
    return cost_v


np.random.seed(3)
data = np.array([[100, 20], [150, 24], [300, 36], [400, 47], [130, 22], [240, 32], [350, 47],
                 [200, 42], [100, 21], [110, 21], [190, 30], [120, 25], [130, 18], [270, 38], [255, 28]])

data = data.astype('float32')
x_data = data[:, 0]
y_data = data[:, 1]

w1 = create_weight(x_data)
b1 = create_bias(y_data)
h1 = hidden_layer(x_data, w1, b1)  # hypothesis = weight * x + bias
cost = cost_function(hidden_layer(x_data, w1, b1), y_data)  # cost 의 값 이제 미분해서 w와 b값 갱신

print()


def numerical_gradient_w(x):
    h = 1e-4
    grad = np.zeros_like(x)
    for idx in range(x.size):
        tmp_val = x[idx]

        x[idx] = tmp_val + h
        hf = hidden_layer(x_data, x, b1)
        fxh1 = cost_function(hf, y_data)

        x[idx] = tmp_val - h
        hf = hidden_layer(x_data, x, b1)
        fxh2 = cost_function(hf, y_data)

        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val
    return grad


def numerical_gradient_b(x):
    h = 1e-4
    grad = np.zeros_like(x)
    for idx in range(x.size):
        tmp_val = x[idx]

        x[idx] = tmp_val + h
        hf = hidden_layer(x_data, w1, x)
        fxh1 = cost_function(hf, y_data)

        x[idx] = tmp_val - h
        hf = hidden_layer(x_data, w1, x)
        fxh2 = cost_function(hf, y_data)

        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val
    return grad


def back_propagation(x, y, w, b, epochs, learning_rate):
    for i in range(epochs):
        h1 = hidden_layer(x, w, b)
        cost_function(h1, y)
        if i % 100 == 0:
            print(cost_function(h1, y))
            print('\n\n')

        grad_w = numerical_gradient_w(w)
        grad_b = numerical_gradient_b(b)

        w -= learning_rate * grad_w
        b -= learning_rate * grad_b

    return w, b


total_w, total_b = back_propagation(x_data, y_data, w1, b1, 500, 0.00001)

print(total_w, total_b)
W = total_w.mean()
B = total_b.mean()

y_predict = W * x_data + B

plt.scatter(x_data, y_data)
plt.plot([min(x_data), max(x_data)], [min(y_predict), max(y_predict)])
plt.show()
