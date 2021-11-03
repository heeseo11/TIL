# chapter4. 모델 훈련
## 4.1 선형 회귀
### 4.1.1 정규 방정식

![image](https://user-images.githubusercontent.com/61724682/140043232-9cab418e-6b1f-4f9f-8054-f59898588494.png)

### 4.1.2 계산 복잡도

## 4.2 경사 하강법
![image](https://user-images.githubusercontent.com/61724682/140042172-1ff248fb-c2e5-4235-a9f5-5dc77579202c.png)

![image](https://user-images.githubusercontent.com/61724682/140043477-812254bd-6dda-4a54-a3bf-85d2fe20c6fb.png)

![image](https://user-images.githubusercontent.com/61724682/140043484-806005c1-5658-4ac5-8d8b-9100c4609ec9.png)

### 4.2.1 배치 경사 하강법
![image](https://user-images.githubusercontent.com/61724682/140043547-a1cdc72d-9503-4ec0-8fa9-3d88b288b3ed.png)
### 4.2.2 확률적 경사 하강법
![image](https://user-images.githubusercontent.com/61724682/140043624-b68f1a6c-5690-4cbd-a958-c5a3e1452ea7.png)

### 4.2.3 미니배치 경사 하강법
![image](https://user-images.githubusercontent.com/61724682/140043644-42c2c481-9da8-4ef8-92cd-4f95dec21e63.png)

|알고리즘	| m이 클 때|	외부 메모리 학습 지원|	n이 클 때|	하이퍼 파라미터 수	|스케일 조정 필요	|사이킷런|
|------|---|---|---|---|---|---|
|정규방정식	|빠름	|X	|느림	|0	|X	|N/A|
|SVD|	빠름|	X|	느림|	0|	X|	LinearRegression|
|배치경사하강법|	느림	|X	|빠름	|2	|O	|SGDRegressor|
|확률적 경사 하강법	|빠름|	O|	빠름|	>=2|	O|	SGDRegressor|
|미니배치 경사 하강법	|빠름	|O	|빠름	|>=2	|O	|SGDRegressor|

## 4.3 다항 회귀
![image](https://user-images.githubusercontent.com/61724682/140043728-b42fc34c-2eae-454c-99ff-1c21629929de.png)

