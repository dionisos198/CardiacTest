import sys
import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from TwoLayerNet import TwoLayerNet
from sklearn.preprocessing import StandardScaler

np.set_printoptions(threshold=sys.maxsize)

csv_path='heart.csv'

df=pd.read_csv(csv_path)

x=df.drop(['HeartDisease', 'RestingBP', 'RestingECG'], axis=1)
y=df['HeartDisease']

x_encoded = pd.get_dummies(x, columns=['Sex', 'ChestPainType', 'ExerciseAngina','ST_Slope'])

newX=x_encoded.values
newY=y.values



# 데이터를 train과 test로 나누기
split_ratio = 0.2  # 예시로 20%를 테스트 데이터로 사용

# 전체 데이터 개수
total_samples = len(newX)

# 테스트 데이터 개수
test_samples = int(total_samples * split_ratio)

# 데이터를 무작위로 섞음
indices = np.arange(total_samples)
np.random.shuffle(indices)

# 테스트 데이터 추출
test_indices = indices[:test_samples]
x_test = newX[test_indices]
t_test = newY[test_indices]

# 훈련 데이터 추출
train_indices = indices[test_samples:]
x_train = newX[train_indices]
t_train = newY[train_indices]

# 결과 확인
print("Train Data Shape:", x_train.shape)
print("Test Data Shape:", x_test.shape)
print("x_train")
print(x_train)
print("t_train")
print(t_train)
print("t_test")

print(x_test)
numeric_data = x_train[:, :5]  # 숫자형 데이터 추출
numeric_data_test=x_test[:,:5]
scaler = StandardScaler()
scaler_test=StandardScaler()
scaled_numeric_data = scaler.fit_transform(numeric_data)
scaled_numeric_data_test=scaler.fit_transform(numeric_data_test)

# 불리언 값의 변환
boolean_data = x_train[:, 5:]  # 불리언 데이터 추출
boolean_data_test=x_test[:, 5:]
boolean_data = boolean_data.astype(int)  # True를 1로, False를 0으로 변환
boolean_data_test=boolean_data_test.astype(int)

# 전처리된 데이터 결합
preprocessed_x_train = np.concatenate((scaled_numeric_data, boolean_data), axis=1)
preprocessed_x_test=np.concatenate((scaled_numeric_data_test,boolean_data_test),axis=1)

print("전처리된 x_train 데이터:")
print(preprocessed_x_train)
print(preprocessed_x_train.dtype)
print("test size 갯수:")
print(len(preprocessed_x_test))
#x_train=x_train.astype(float)
#t_train=t_train.astype(float)
print("network")

network = TwoLayerNet(input_size=16, hidden_size=1, output_size=1)


# 하이퍼 파라메터
iters_num = 20000 # 반복횟수
train_size = x_train.shape[0]
print(train_size)
batch_size = 100 # 미니배치 크기
learning_rate = 0.01
train_loss_list = []
train_acc_list = []
test_acc_list = []
# 1에폭당 반복 수
iter_per_epoch = max(78, 1)
print("iter_per_epoch")
print(iter_per_epoch)

print(x_train)
t_train=t_train.reshape(-1,1)
t_test=t_test.reshape(-1,1)


for i in range(iters_num):
 # print(i)
 # 미니배치 획득
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = preprocessed_x_train[batch_mask]
    t_batch = t_train[batch_mask].reshape((100,1))


 # 오차역전파법으로 기울기 계산
    grad = network.gradient(x_batch, t_batch)

 # 매개변수 갱신
    for key in ('W1', 'b1', 'W2', 'b2'):
      network.params[key] -= learning_rate * grad[key]

 # 학습 경과 기록
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

 # 1에폭 당 정확도 계산
    if i % iter_per_epoch == 0:
      train_acc = network.accuracy(preprocessed_x_train, t_train)
      test_acc = network.accuracy(preprocessed_x_test, t_test)
      train_acc_list.append(train_acc)
      test_acc_list.append(test_acc)
      print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))


# Plotting the training loss
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_loss_list)
plt.title('Training Loss')
plt.xlabel('Iterations')
plt.ylabel('Loss')

# Plotting the training and test accuracy
plt.subplot(1, 2, 2)
plt.plot(train_acc_list, label='train acc')
plt.plot(test_acc_list, label='test acc', linestyle='--')
plt.title('Training and Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()