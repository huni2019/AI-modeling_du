import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Dropout
#from tensorflow.keras.optimizers import Adam
#from tensorflow.keras.preprocessing.image import ImageDataGenerator
#from tensorflow.keras.applications.resnet50 import ResNet50

# CIFAR-10 데이터셋을 읽고 신경망에 입력할 형태로 변환
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train=x_train.astype(np.float32)/255.0
x_test=x_test.astype(np.float32)/255.0
y_train=tf.keras.utils.to_categorical(y_train,10)
y_test=tf.keras.utils.to_categorical(y_test,10)

cnn=tf.keras.models.load_model("조대현.h5") # 학습된 모델 불러오기
cnn.summary()
# 신경망 모델 정확률 평가
res=cnn.evaluate(x_test,y_test,verbose=0)
print("정확률은",res[1]*100)
print("최준영-6-New2")