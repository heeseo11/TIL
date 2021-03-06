from tensorflow import keras
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#fetch_california_housing() 함수를 사용해서 데이터를 적재
housing = fetch_california_housing()

X_train_full, X_test, y_train_full, y_test = train_test_split(
    housing.data, housing.target)
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_full, y_train_full)

#스케일 조정 
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)

#하나의 입력을 짧은 경로로 전달 
#모델의 입력 정의 => 한 모델은 여러개의 입력을 가질 수 있음
input_ = keras.layers.Input(shape=X_train.shape[1:])
#30개의 뉴런과 ReLU활성화 함수를 가진 Dense층 생성
hidden1 = keras.layers.Dense(30, activation = "relu")(input_)
#hidden1의 전달을 받음 
hidden2 = keras.layers.Dense(30, activation = "relu")(hidden1)
#concatencate층을 만들고 hidden2와 입력을 연결
concat = keras.layers.Concatenate()([input_,hidden2])
#하나의 뉴런과 활성함수가 없는 출력층을 만듬, concat의 결과를 호출
output = keras.layers.Dense(1)(concat)
#입력과 출력 지정 
model = keras.Model(inputs=[input_], outputs=[output])

#두개의 입력 중 하나는 와이드 경로로 하나는 딥경로로 전달
input_A = keras.layers.Input(shape=[5], name="wide_input")
input_B = keras.layers.Input(shape=[6], name="deep_input")
hidden1 = keras.layers.Dense(30, activation="relu")(input_B)
hidden2 = keras.layers.Dense(30, activation="relu")(hidden1)
concat = keras.layers.concatenate([input_A, hidden2])
output = keras.layers.Dense(1, name="output")(concat)
model = keras.Model(inputs=[input_A, input_B], outputs=[output])

model.compile(loss="mse", optimizer=keras.optimizers.SGD(lr=1e-3))

X_train_A, X_train_B = X_train[:, :5], X_train[:, 2:]
X_valid_A, X_valid_B = X_valid[:, :5], X_valid[:, 2:]
X_test_A, X_test_B = X_test[:, :5], X_test[:, 2:]
X_new_A, X_new_B = X_test_A[:3], X_test_B[:3]

history = model.fit((X_train_A, X_train_B), y_train, epochs=20,
                    validation_data=((X_valid_A, X_valid_B), y_valid))
mse_test = model.evaluate((X_test_A, X_test_B), y_test)
y_pred = model.predict((X_new_A, X_new_B))

#여러개의 출력이 필요한 경우
#회귀작업과 분류작업을 함께 사용하는 경우
#동일한 데이터에서 독립적인 여러작업을 수행할 때
#보조 출력을 사용해 하위네트워크가 나머지 네트워크에 의존하지 않고 그 자체로 유용한 것을 학습하는지 확인 가능

[...] # 출력층 까지 이전과 동일 
output = keras.layers.Dense(1, name="main_output")(concat)
#보조출력
aux_output = keras.layers.Dense(1, name="aux_output")(hidden2)
model = keras.Model(inputs=[input_A, input_B], outputs=[output, aux_output])

#보조 출력보다 주 출력에 더 관심이 많다면, 주출력의 손실에 더 많은 가중치를 부여함
model.compile(loss=["mse", "mse"], loss_weights=[0.9, 0.1], optimizer="sgd")

#주 출력과 보조 출력이 같은 것을 예측하므로 동일한 레이블 사용
history = model.fit(
    [X_train_A, X_train_B], [y_train, y_train], epochs=20,
    validation_data=([X_valid_A, X_valid_B], [y_valid, y_valid]))
    
#모델 평가
total_loss, main_loss, aux_loss = model.evaluate(
[X_test_A, X_test_B], [y_test, y_test])

#각 출력에 대한 예측을 반환 
y_pred_main, y_pred_aux = model.predict([X_new_A, X_new_B])

y_pred_main
y_pred_aux

#모델 저장과 복원
#모델 서브클래싱에서는 사용할 수 없음

model.save("my_keras_model.h5")
model = keras.models.load_model("my_keras_model.h5")
