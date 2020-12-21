#각 출력 뉴련의 결정경계는 선형이므로 퍼셉트론도 복잡한 패턴을 학습 못함
#하지만 훈련 샘플이 선형적으로 구분될 수 있다면 이 알고리즘은 정답에 수렴함

#퍼셉트론은 클래스 확률을 제공하지 않으며 고정된 임곗값을 기준으로 예측을 만듬

import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron

iris = load_iris()
X = iris.data[:,(2,3)]
y = (iris.target == 0).astype(np.int)

per_clf = Perceptron()
per_clf.fit(X,y)

y_pred = per_clf.predict([[1.4,0.2]])

print(y_pred)
