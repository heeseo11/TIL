#콜백 사용하기
#훈련 마지막에 모델을 저장하는 것 뿐아니라 훈련 도중 일정한 간격으로 체크포인트를 저장이 가능

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

model = keras.models.Sequential([
    keras.layers.Dense( 30, activation="relu", input_shape=[8] ),
    keras.layers.Dense( 30, activation="relu"),
    keras.layers.Dense(1)
])

model.compile( loss="mse", optimizer=keras.optimizers.SGD(lr = 1e-3) )

#ModelCheckpoint는 훈련하는 동안 일정한 간격으로 모델의 체크포인트를 저장
checkpoint_cb = keras.callbacks.ModelCheckpoint("my_keras_model.h5")
history = model.fit(X_train, y_train, epochs=10, callbacks=[checkpoint_cb])

#save_best_only=True를 지정하면 최상의 검증 세트 점수에서만 모델을 저장
checkpoint_cb = keras.callbacks.ModelCheckpoint("my_keras_model.h5",
                                                save_best_only=True)
history = model.fit(X_train, y_train, epochs=10,
                    validation_data=(X_valid, y_valid),
                    callbacks=[checkpoint_cb])
#마지막에 저장된 모델을 복원하면 검증 세트에서 최상의 점수를 낸 모델이 됨
model = keras.models.load_model("my_keras_model.h5") # 최상의 모델로 복원

#EarlyStopping 조기종료를 구현
#모델이 향상되지 않으면 훈려닝 자동을 중지됨 따라서 시간과 컴퓨팅 자원을 낭비하지 않음
# 훈련이 끝난 후 최상의 가중치를 복원하므로 저장된 모델을 따로 복원할 필요 없음
early_stopping_cb = keras.callbacks.EarlyStopping(patience=10, 
                                                  restore_best_weights=True)
history = model.fit(X_train, y_train, epochs=100,
                    validation_data=(X_valid, y_valid),
                    callbacks=[checkpoint_cb, early_stopping_cb])
                    
#사용자 정의 콜백 만들기 
class PrintValTrainRatioCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        print("\nval/train:{:.2f}".format(logs["val_loss"]/logs["loss"]))
