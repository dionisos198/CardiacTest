import numpy as np
from Affine import Affine
from Relu import Relu
from Sigmoid import Sigmoid
from SoftmaxWLoss import SoftmaxWLoss
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
        #self.layers['Sigmoid'] = Sigmoid()
        self.layers['Relu'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])

        #self.lastLayer = SoftmaxWLoss()
        self.lastLayer=SigmoidBinaryCrossEntropyLoss()

    def predict(self, x):
        for layer in self.layers.values():
            '''
            print("predict layer안")
            print(x)
            '''
            #time.sleep(1)

            x = layer.forward(x)


        return x

    # x : 입력 데이터, t : 정답 레이블
    def loss(self, x, t):

        y = self.predict(x)

        return self.lastLayer.forward(y, t)

    def accuracy(self, x, t):
        '''
        print("accuracy")
        y = self.predict(x)
        print("y")
        print(y)
        time.sleep(1)
        y = np.argmax(y, axis=1)
        print("이후")
        print(y)
        '''
        '''
        fortest
        '''
        y=self.predict(x)
        y=self.sigmoid(y)
        y=self.binary_classification_predictions(y)

        '''
        test 위해 주석처리
        if t.ndim != 1:
            print("t는1")
            t = np.argmax(t, axis=1)
        '''

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    # x : 입력 데이터, t : 정답 레이블
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        grads['W1'] = self.numerical_gradient2(loss_W, self.params['W1'])
        grads['b1'] = self.numerical_gradient2(loss_W, self.params['b1'])
        grads['W2'] = self.numerical_gradient2(loss_W, self.params['W2'])
        grads['b2'] = self.numerical_gradient2(loss_W, self.params['b2'])

        return grads

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

    def numerical_gradient2(self,f, x):
        h = 1e-4  # 0.0001
        grad = np.zeros_like(x)

        it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            idx = it.multi_index
            tmp_val = x[idx]
            x[idx] = float(tmp_val) + h
            fxh1 = f(x)  # f(x+h)

            x[idx] = tmp_val - h
            fxh2 = f(x)  # f(x-h)
            grad[idx] = (fxh1 - fxh2) / (2 * h)

            x[idx] = tmp_val  # 값 복원
            it.iternext()

        return grad


    #test 용
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def binary_classification_predictions(self,predictions, threshold=0.5):
        return (predictions > threshold).astype(int)

