from tensorflow import keras

#시퀀셜 API와 함수형 API
#선언적임
#사용할 층과 연결방식을 먼저 정의
#모델에 데이터를 주입 후 훈련, 추론 시작

#장점 : 모델 저장, 복사, 공유가 쉬움
#모델의 구조를 출력하거나 분석에 용의
#에러발견이 쉬움
#정적 그래프이므로 디버깅 쉬움

#서브클래싱 API
#동적인 구조를 필요로 할때 사용 => 유연성이 높아짐
#하지만, 모델을 저장하거나 복사 못함
#층 간의 연결 정보를 얻을 수 없음
#타입과 크기를 확인 불가

#input class의 객체를 만들 필요가 없음 
class WideAndDeepModel(keras.Model):
    def __init__(self, units=30, activation="relu", **kwargs):
        super().__init__(**kwargs)
        self.hidden1 = keras.layers.Dense(units, activation=activation)
        self.hidden2 = keras.layers.Dense(units, activation=activation)
        self.main_output = keras.layers.Dense(1)
        self.aux_output = keras.layers.Dense(1)
        
    #call 메소드 안에 원하는 어떤 계산을 사용할 수 있음 => for문, if문, 텐서플로 저수준 연산 
    def call(self, inputs):
        input_A, input_B = inputs
        hidden1 = self.hidden1(input_B)
        hidden2 = self.hidden2(hidden1)
        concat = keras.layers.concatenate([input_A, hidden2])
        main_output = self.main_output(concat)
        aux_output = self.aux_output(hidden2)
        return main_output, aux_output

model = WideAndDeepModel()
