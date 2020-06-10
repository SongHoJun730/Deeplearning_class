
from tkinter import *
import numpy as np


def makeMatrix(i, fill=0.0):
    mat = []
    for i in range(i):
        mat.append([fill] * 4)  # x1 y1 x2 y2 + 다음 계층 노드의 x좌표
    return np.array(mat)


def makehiddenMat(y, fill=0.0):  # [2, 1, 3]
    mat = []
    for idx in range(len(y)):
        layer = []
        for i in range(y[idx]):
            layer.append(np.array([fill] * 4))  # x1 y1 x2 y2 + 다음 계층 노드의 x좌표
        mat.append(np.array(layer))
    return np.array(mat)


class VisualizeNetwork:
    def __init__(self, num_x, num_yh, num_y):
        window = Tk()
        self.canvas = Canvas(window, width=1920, height=1080)
        self.num_x = num_x
        self.num_yh = num_yh
        self.num_y = num_y
        self.radius = 25  # 원 크기 radius
        self.diameter = 50  # 노드간 거리  diameter
        self.term = 100  # term
        self.mid_x = 960  # 512
        self.mid_y = 540  # 384
        self.init_x = 0
        self.init_y = 0
        self.x_layer = []
        self.y_layer = []

        # 계층의 x좌표
        self.layer_count = len(self.num_yh) + 2  # 입력층과 출력층은 한개씩

        for layer in range(self.layer_count):
            self.x_layer.append([0.0])

        if self.layer_count % 2 == 0:
            self.init_x = self.mid_x - (self.term + 150 * (self.layer_count / 2 - 1))
        else:
            self.init_x = self.mid_x - (self.radius + 150 * (self.layer_count // 2))

        self.x_layer[0] = self.init_x
        for num in range(self.layer_count - 1):  # 01
            self.x_layer[num + 1] = self.init_x + (num + 1) * 150

        # 계층의 y좌표
        self.y_layer.append([1.0] * self.num_x)
        for layer_num in range(len(self.num_yh)):
            self.y_layer.append([1.0] * self.num_yh[layer_num])
        self.y_layer.append([1.0] * self.num_y)

        # 입력층 노드의 개수
        for num in range(len(self.y_layer)):
            if len(self.y_layer[num]) % 2 == 0:
                self.init_y = self.mid_y - (self.term + 150 * (len(self.y_layer[num]) / 2 - 1))
            else:
                self.init_y = self.mid_y - (self.radius + 150 * (len(self.y_layer[num]) // 2))
            self.y_layer[num][0] = self.init_y
            for idx in range(len(self.y_layer[num]) - 1):  # 01
                self.y_layer[num][idx + 1] = self.init_y + (idx + 1) * 150
        self.canvas.pack()

        # 함수 넣기
        self.input_visualize()
        self.hidden_visualize()
        self.output_visualize()
        self.connect_node()
        # x1 y1 x2 y2
        window.mainloop()

    def connect_node(self):
        for num in range(self.layer_count - 1):  # 입력층 출력층 제외외
            for i in range(len(self.y_layer[num])):  # 01 01
                for j in range(len(self.y_layer[num + 1])):  # 01 012
                    self.canvas.create_line(self.x_layer[num] + self.diameter, self.y_layer[num][i] + self.radius,
                                            self.x_layer[num + 1], self.y_layer[num + 1][j] + self.radius, fill="black")

        # x1, y1, x2, y2

    def input_visualize(self):
        for idx in range(len(self.y_layer[0])):  # 01
            self.canvas.create_oval(self.x_layer[0], self.y_layer[0][idx], self.x_layer[0] + self.diameter,
                                    self.y_layer[0][idx] + self.diameter, fill='red')

            # x1 y1 x2 y2

    def hidden_visualize(self):
        for num in range(self.layer_count - 2):  # 입력층 출력층 제외외
            for idx in range(len(self.y_layer[num + 1])):  # 01
                self.canvas.create_oval(self.x_layer[num + 1], self.y_layer[num + 1][idx],
                                        self.x_layer[num + 1] + self.diameter,
                                        self.y_layer[num + 1][idx] + self.diameter, fill='blue')

    def output_visualize(self):
        for idx in range(len(self.y_layer[-1])):  # 012
            self.canvas.create_oval(self.x_layer[-1], self.y_layer[-1][idx], self.x_layer[-1] + self.diameter,
                                    self.y_layer[-1][idx] + self.diameter, fill='green')  # x1 y1 x2 y2


