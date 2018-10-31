import numpy as np
import random
from scipy.special import expit

class Network(object):
    def __init__(self,sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y,1) for y in sizes[1:]]
        self.weights = [np.random.randn(y,x) for y,x in zip(sizes[:-1],sizes[1:])]

    def feedforward(self,a):
        for b,w in zip(self.biases,self.weights):
            # a = sigmoid(np.dot(w,a)+b)
            a = expit(np.dot(w,a)+b)
        return a
# sizes = [2,3,1]
# # bias = [np.random.randn(y,1) for y in sizes[1:]]
# # print(bias)
# # for x,y in zip(sizes[:-1],sizes[1:]):
# #     print(x,y)

net = Network([2,3,1])