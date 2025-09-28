import numpy as np
np.seterr(divide='ignore', invalid='ignore')

def sigmoid(x):
    x = np.clip(x, -500, 500)  # 限制输入范围
    return 1 / (1 + np.exp(-x))

def softmax(x):
    x = x - x.max(axis=0)
    y = np.exp(x)
    y /= y.sum(axis=0, keepdims=True)
    return y
