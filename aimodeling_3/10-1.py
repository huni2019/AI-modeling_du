import numpy as np
from tensorflow.keras.datasets import mnist

# MNIST 데이터를 읽고 0 패턴만 추출
(x_train, y_train), (x_test, y_test) = mnist.load_data()
X = x_train[np.isin(y_train, [0])]

# 화소 수준의 확률 생성 모델 구축
P = np.zeros((28, 28, 256))
for k in range(X.shape[0]):
    for i in range(X.shape[1]):
        for j in range(X.shape[2]):
            P[i][j][X[k][i][j]] += 1
P = P / X.shape[0]

# 확률 생성 모델을 이용하여 20개 샘플을 생성
Xnew = np.zeros((20, 28, 28))
for i in range(20):
    for r in range(28):
        for c in range(28):
            Xnew[i][r][c] = np.random.choice(range(256), p=P[r][c])

import matplotlib.pyplot as plt

plt.figure(figsize=(20, 4))
for r in range(2):
    for c in range(10):
        plt.subplot(2, 10, r * 10 + c + 1)
        plt.imshow(Xnew[r * 10 + c], cmap='gray')
        plt.xticks([]);
        plt.yticks([])
plt.show()