# coding: utf-8

import threading
import random
import numpy as np
from AI_Assignment.problem_2 import VisualizeNetwork

random.seed(777)

# 환경 변수 지정

# 입력값 및 타겟값
data = [
    [[0, 0], [0]],
    [[0, 1], [1]],
    [[1, 0], [1]],
    [[1, 1], [0]]
]


# 실행 횟수(iterations), 학습률(lr), 모멘텀 계수(mo) 설정
iterations = 5000
lr = 0.1
mo = 0.4


# 활성화 함수 - 1. 시그모이드
# 미분할 때와 아닐 때의 각각의 값
def sigmoid(x, derivative=False):
    if (derivative == True):
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))


# 활성화 함수 - 2. tanh
# tanh 함수의 미분은 1 - (활성화 함수 출력의 제곱)
def tanh(x, derivative=False):
    if (derivative == True):
        return 1 - x ** 2
    return np.tanh(x)


# 가중치 배열 만드는 함수
def makeMatrix(i, j, fill=0.0):
    mat = []
    for i in range(i):
        mat.append([fill] * j)
    return np.array(mat)


def makehiddenMat(y, fill=0.0):
    mat = []
    for idx in range(len(y)):
        layer = []
        if idx == len(y) - 1:
            break
        else:
            for i in range(y[idx]):
                layer.append(np.array([fill] * y[idx + 1]))
        mat.append(np.array(layer))
    return np.array(mat)


def biasMat(a):  # a 는 리스트 b 는 숫자
    mat = []
    for idx in a:
        mat.append(np.zeros(idx,))
    return np.array(mat)


# 신경망의 실행
class NeuralNetwork:

    # 초깃값의 지정
    def __init__(self, num_x, num_yh, num_yo, bias=1):

        # 입력값(num_x), 은닉층 초깃값(num_yh), 출력층 초깃값(num_yo), 바이어스
        self.num_x = num_x  # + bias  # 바이어스는 1로 지정(본문 참조)
        self.num_yh = np.array(num_yh)
        self.num_yo = num_yo

        self.hidden_bias = biasMat(num_yh)
        self.output_bias = np.array([0.0]*num_yo)
        self.grd_hidden_bias = biasMat(num_yh)
        self.grd_output_bias = np.array([0.0] * num_yo)

        # 활성화 함수 초깃값
        self.activation_input = [1.0] * self.num_x
        if len(self.num_yh) > 1:
            self.activation_hidden = []
            for num in range(len(self.num_yh)):
                self.activation_hidden.append([1.0] * self.num_yh[num])
        else:
            self.activation_hidden = np.array([1.0] * self.num_yh[0])

        self.activation_out = [1.0] * self.num_yo

        # 가중치 입력 초깃값
        self.weight_in = makeMatrix(self.num_x, self.num_yh[0])
        for i in range(self.num_x):
            for j in range(self.num_yh[0]):
                self.weight_in[i][j] = random.random()

        # 가중치 출력 초깃값
        self.weight_out = makeMatrix(self.num_yh[-1], self.num_yo)
        for j in range(self.num_yh[-1]):
            for k in range(self.num_yo):
                self.weight_out[j][k] = random.random()

        if len(self.num_yh) > 1:
            self.weight_ing = makehiddenMat(self.num_yh)
            for num in range(len(self.weight_ing)):
                for i in range(len(self.weight_ing[num])):
                    for j in range(len(self.weight_ing[num][i])):
                        self.weight_ing[num][i][j] = random.random()
            # 모멘텀 SGD를 위한 이전 가중치 초깃값
            self.gradient_ing = makehiddenMat(self.num_yh)

        # 모멘텀 SGD를 위한 이전 가중치 초깃값
        self.gradient_in = makeMatrix(self.num_x, self.num_yh[0])
        self.gradient_out = makeMatrix(self.num_yh[-1], self.num_yo)

    # 업데이트 함수
    def update(self, inputs):
        # 입력 레이어의 활성화 함수
        for i in range(self.num_x):
            self.activation_input[i] = inputs[i]

        # 입력층의 활성화 함수
        for j in range(self.num_yh[0]):
            sum = 0.0
            for i in range(self.num_x):
                sum = sum + self.activation_input[i] * self.weight_in[i][j]
            # 시그모이드와 tanh 중에서 활성화 함수 선택
            sum += self.hidden_bias[0][j]
            if len(self.num_yh) > 1:
                self.activation_hidden[0][j] = tanh(sum, False)
            else:
                self.activation_hidden[j] = tanh(sum, False)

        # 은닉층의 활성화 함수
        if len(self.num_yh) > 1:
            for num in range(len(self.num_yh) - 1):
                for j in range(self.num_yh[num + 1]):
                    sum = 0.0
                    for i in range(self.num_yh[num]):
                        sum = sum + self.activation_hidden[num][i] * self.weight_ing[num][i][j]
                    # 시그모이드와 tanh 중에서 활성화 함수 선택
                    sum += self.hidden_bias[num + 1][j]
                    self.activation_hidden[num + 1][j] = tanh(sum, False)

        # 출력층의 활성화 함수
        for k in range(self.num_yo):
            sum = 0.0
            for j in range(self.num_yh[-1]):
                if len(self.num_yh) > 1:
                    sum = sum + self.activation_hidden[-1][j] * self.weight_out[j][k]
                else:
                    sum = sum + self.activation_hidden[j] * self.weight_out[j][k]
            sum += self.output_bias[k]
            # 시그모이드와 tanh 중에서 활성화 함수 선택
            self.activation_out[k] = tanh(sum, False)

        return self.activation_out[:]

    # 역전파의 실행
    def backPropagate(self, targets):

        # 델타 출력 계산
        output_deltas = np.array([0.0] * self.num_yo)
        for k in range(self.num_yo):
            error = targets[k] - self.activation_out[k]  # 오차 제곱을 미분한 값 dL/dT3
            # 시그모이드와 tanh 중에서 활성화 함수 선택, 미분 적용
            output_deltas[k] = tanh(self.activation_out[k], True) * error  # dT3/dO1

        # 은닉 노드의 오차 함수
        hidden_deltas = []
        deltas_arr = self.num_yh[::-1]
        if len(self.num_yh) > 1:
            for num in range(len(self.num_yh)):
                hidden_deltas.append([1.0] * deltas_arr[num])
        else:
            hidden_deltas = np.array([1.0] * self.num_yh[0])

        # 은닉층 마지막 계층 활성화함수
        for j in range(self.num_yh[-1]):  # 01
            error = 0.0
            for k in range(self.num_yo):  # 0
                error = error + output_deltas[k] * self.weight_out[j][k]
                # 시그모이드와 tanh 중에서 활성화 함수 선택, 미분 적용
            if len(self.num_yh) > 1:
                hidden_deltas[0][j] = tanh(self.activation_hidden[-1][j], True) * error
            else:
                hidden_deltas[j] = tanh(self.activation_hidden[j], True) * error

        # 출력 편향 업데이트
        for num in range(self.num_yo):
            bias_v = mo * self.grd_output_bias[num] - lr * output_deltas[num]
            self.output_bias[num] += bias_v
            self.grd_output_bias[num] = output_deltas[num]
        # 출력 가중치 업데이트
        for j in range(self.num_yh[-1]):  # 0123
            for k in range(self.num_yo):  # 0
                if len(self.num_yh) > 1:
                    gradient = output_deltas[k] * self.activation_hidden[-1][j]
                else:
                    gradient = output_deltas[k] * self.activation_hidden[j]

                v = mo * self.gradient_out[j][k] - lr * gradient  # 모맨텀
                self.weight_out[j][k] += v
                self.gradient_out[j][k] = gradient

        if len(self.num_yh) > 1:

            reverse_act_hid = self.activation_hidden[::-1]
            reverse_weight_ing = self.weight_ing[::-1]
            reverse_gradient_ing = self.gradient_ing[::-1]
            reverse_bias = self.hidden_bias[::-1]
            reverse_grd_bias = self.grd_hidden_bias[::-1]
            # 나머지 은닉층의 활성화 함수
            for num in range(len(self.num_yh) - 1):  # 012
                for j in range(deltas_arr[num + 1]):  # 0 012 01
                    error = 0.0
                    for k in range(deltas_arr[num]):  # 0123 0 012
                        error = error + hidden_deltas[num][k] * reverse_weight_ing[num][j][k]
                        # 시그모이드와 tanh 중에서 활성화 함수 선택, 미분 적용
                    hidden_deltas[num + 1][j] = tanh(reverse_act_hid[num + 1][j], True) * error
            # 은닉층 편향 업데이트
            for num in range(len(self.num_yh) - 1):
                for idx in range(deltas_arr[num]):
                    bias_v = mo * reverse_grd_bias[num][idx] - lr * hidden_deltas[num][idx]
                    reverse_bias[num][idx] += bias_v
                    reverse_grd_bias[num][idx] = hidden_deltas[num][idx]

            for num in range(len(self.num_yh) - 1):
                # 은닉 가중치 업데이트
                for j in range(deltas_arr[num + 1]):  # 012 0123 01
                    for k in range(deltas_arr[num]):  # 0123 0 012
                        gradient = hidden_deltas[num][k] * reverse_act_hid[num + 1][j]  # 012 01

                        v = mo * reverse_gradient_ing[num][j][k] - lr * gradient  # 모맨텀
                        reverse_weight_ing[num][j][k] += v
                        reverse_gradient_ing[num][j][k] = gradient

        # 첫번째 은닉층 편향 업데이트
        for num in range(self.num_yh[0]):
            if len(self.num_yh) > 1:
                bias_grad = hidden_deltas[-1][num]
            else:
                bias_grad = hidden_deltas[num]

            bias_v = mo * self.grd_hidden_bias[0][num] - lr * bias_grad
            self.hidden_bias[0][num] += bias_v
            self.grd_hidden_bias[0][num] = bias_grad

        # 입력 가중치 업데이트
        for i in range(self.num_x):
            for j in range(self.num_yh[0]):
                if len(self.num_yh) > 1:
                    gradient = hidden_deltas[-1][j] * self.activation_input[i]
                else:
                    gradient = hidden_deltas[j] * self.activation_input[i]

                v = mo * self.gradient_in[i][j] - lr * gradient
                self.weight_in[i][j] += v
                self.gradient_in[i][j] = gradient

        # 오차의 계산(최소 제곱법)
        error = 0.0
        for k in range(len(targets)):
            error = error + 0.5 * (targets[k] - self.activation_out[k]) ** 2
        return error

    # 학습 실행
    def train(self, patterns):
        for i in range(iterations):
            error = 0.0
            for p in patterns:
                inputs = p[0]
                targets = p[1]
                self.update(inputs)
                error = error + self.backPropagate(targets)

            if (i+1) % 500 == 0:
                print('error: %-.5f' % error)
                # print("self.bias\n", self.bias)

    # 결괏값 출력
    def result(self, patterns):
        for p in patterns:
            print('Input: %s, Predict: %s' % (p[0], self.update(p[0])))


if __name__ == '__main__':  # 문제 1의 은닉층 수를 늘리기
    # 두 개의 입력 값, 두 개의 레이어, 하나의 출력 값을 갖도록 설정

    n = NeuralNetwork(2, [2], 1)  # 무조건 [] 을 써야 함....
    # , 4, 3

    # 학습 실행
    n.train(data)

    # 결괏값 출력
    n.result(data)

    # visualize = threading.Thread(target=VisualizeNetwork, args=[2, [3], 1])  # , 1, 3
    # visualize.start()

# Reference: http://arctrix.com/nas/python/bpnn.py (Neil Schemenauer)
"""
20.06.09 - 역전파의 for 문 이상
"""
