import numpy as np
from Affine import Affine
from Relu import Relu
from Sigmoid import Sigmoid
from collections import OrderedDict
from SigmoidBinaryCrossEntropyLoss import SigmoidBinaryCrossEntropyLoss
import sys
import time


class TwoLayerNet:

    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # 가중치 초기화
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)#inputsize 행과 hiddenSize열을 가짐,0과1사이의 난수로 채움
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

        # 계층 생성
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'],self.params['b1'])
       # self.layers['Sigmoid'] = Sigmoid()
        self.layers['Relu'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])


        self.lastLayer=SigmoidBinaryCrossEntropyLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)


        return x

    # x : 입력 데이터, t : 정답 레이블
    def loss(self, x, t):

        y = self.predict(x)

        return self.lastLayer.forward(y, t)

    def accuracy(self, x, t):

        y=self.predict(x)
        y=self.sigmoid(y)
        y=self.binary_classification_predictions(y)


        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy



    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.lastLayer.backward(dout)



        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)




        # 결과 저장
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W2'], grads['b2'] = self.layers['Affine2'].dW, self.layers['Affine2'].db

        return grads


    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def binary_classification_predictions(self,predictions, threshold=0.5):
        return (predictions > threshold).astype(int)