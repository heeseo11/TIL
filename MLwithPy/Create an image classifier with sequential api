import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#버전 확인
tf.__version__
keras.__version__

fashion_mnist = keras.datasets.fashion_mnist

#데이터셋 다운로드 후 적재
(X_train_full, y_train_full),(X_test, y_test) = fashion_mnist.load_data()

#데이터 크기
X_train_full.shape

#데이터 타입
X_train_full.dtype

#입력 특성의 스케일 조정 => 경사하강법으로 신경망을 훈련하기 때문
#필셀 강도를 255.0으로 나누어 0~1 사이의 범위로 조정 
X_valid, X_train = X_train_full[:5000] /255.0, X_train_full[5000:]/255.0
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
X_test = X_test/255.0

class_names = ["T-shirt/top","Trouser","Pullover","Dress","Coat","Sandal","shirt","Sneaker","Bag","Ankle boot"]

class_names[y_train[5]]

plt.figure() 
plt.imshow(X_train[0])
plt.colorbar()
plt.grid(False) 
plt.show()

#시퀀셜 API를 사용하여 모델 만들기
#Sequential는 순서대로 연결된 층을 일렬로 쌓아서 구성
#model = keras.models.Sequential()

#Flatten은 입ㄹ력이미즈를 1D배열로 변환
#model.add(keras.layers.Flatten(input_shape=[28,28]))

#Dense는 은닉층을 추가
#model.add(keras.layers.Dense(300,activation = "relu"))
#model.add(keras.layers.Dense(100,activation = "relu"))
#클래스마다 하나씩 총 10개의 뉴런을 가진 출력층이 생김
# 다중분류이므로 softmax 사용
#model.add(keras.layers.Dense(10,activation = "softmax"))

#Sequential 모델을 만들 때 층의 리스트를 전달
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28,28]),   
    keras.layers.Dense(300,activation = "relu"),
    keras.layers.Dense(100,activation = "relu"),
    keras.layers.Dense(10,activation = "softmax")
])

#모델의 모든 층을 출력
#dense_1의 파라미터가 784*300+300 = 235,500개임
#훈련데이터가 많지 않을 경우 과대적합의 위험이 있음
#또한 훈련데이터를 학습하기 충분한 유연성을 가짐 
model.summary()

#모델에 있는 층을 리스트로 출력
model.layers


#인덱스나 이름으로 층 선택이  가능 
hidden1 = model.layers[1]
hidden1.name

model.get_layer('dense_1') is hidden1

#get_weights와 set_weights메서드를 사용해 층의 모든 파라미터 접근 가능
weights, biases = hidden1.get_weights()

weights
weights.shape

biases
biases.shape

#모델 컴파일
#클래스가 배타적이므로 "sparse_categorical_crossentropy" 사용
#만약, 클래스별 타깃이 확률이라면(원핫 벡터) "categorical_crossentropy" 사용 
#이진분류의 경우 sigmoid 함수 사용, "binary_crossentropy" 손실 사용
#옵티마이저의 "sgd"를 지정 => stochastic gradient descent (확률적 경사 하강법)
#accuracy 정확도 측정
model.compile(loss = "sparse_categorical_crossentropy",
             optimizer = "sgd",
             metrics = ["accuracy"])

#모델 훈련 및 평가
#모델 훈련을 위해서 fit 메소드 호출 
#훈련 손실(loss 값)이 감소하는지 확인
#훈련 정확도가 검증 정확도보다 월등히 크면 과대적합 의심 => 차이 확인
history = model.fit(X_train, y_train, epochs =30,
                   validation_data = (X_valid, y_valid))


#훈련세트가 너무 편중되어 있는 경우 
#fit() 메소드를 호출할 때 class_weight 매개변수를 지정 
#=> 적게 증장하는 클래스는 높은 가중치 / 많이 등장하는 클래스는 낮은 가중치를 부여
#이 가중치는 손실을 계산할 때 사용

#샘플별로 가중치를 부여하고 싶다면 sample_weight 매개변수를 지정
#=> 전문가에 의해 할당된 레이블과 크라우드소싱 플랫폼을 사용해 할당된 레이블이 있을 경우 전자에 더 높은 가중치 부여

#위의 두 class_weight와 sample_weight가 둘다 사용 되었다면 두 값을 곱하여 사용

#history.history => 에포크가 끝날 때 마다 훈련 세트와 검증세트에 대한 손실과 측정한 지표를 담은 딕셔너리
#훈련하는 동안 훈련과 검증의 정확도는 계속 상승
#훈련과 검증의 손실(loss)는 감소 
#검증곡선이 훈련곡선과 가깝다면 과대적합이 되지 않았다는 증거 

pd.DataFrame(history.history).plot(figsize=(8,5))
plt.grid(True)
plt.gca().set_ylim(0,1)
plt.show()

#그래프 초기 부분에서 훈련보다 검증에서 더 좋은 성능을 낸것처럼 보임
#하지만 검증의 손실은 에포크가 끝난 후에 계산되고, 훈련손실은 에포르가 진행되는 동안 계산 됨
#따라서 그래프를 볼 때 왼쪽으로 에포크의 절반만큼 이동해서 보면 됨

#evaluate() 메소드 : 테스트 세트로 모델을 평가하여 일반화 오차를 추정
model.evaluate(X_test, y_test)

#모델을 사용해 예측 만들기
#predict 메소드 : 새로운 샘플에 대해 예측 생성
X_new = X_test[:3]
y_proba = model.predict(X_new)
#각 sample에 대해 0~9까지 클래스 마다 각각의 확률을 모델이 추정
y_proba.round(2)

#predict_classes() 메소드 : 가장 높은 확률을 가진 클래스 추출
y_pred = model.predict_classes(X_new)
y_pred

np.array(class_names)[y_pred]

y_new = y_test[:3]
y_new
