# 화자인식 경진대회
## 1. DATA
- Train : 3182명의 화자, 총 74304개의 발화 음성 데이터
- Test : 9232개의 발화 음성 데이터
---
## 2. Train Method
#### Preprocessing
  - wav -> mel spectrogram으로 변환 (n_mel = 40) 
#### Model
  - LSTM으로 화자 벡터 추출
#### Loss
  - GE2E Loss : Triplet loss와 유사한 개념으로 화자와 비슷할 수록 가깝게, 다를수록 멀리 떨어지게 해주는 방식
  - 논문에선 64명의 화자와 그 화자들마다 10개의 발화를 기준으로 학습
  - 본 경진대회에서는 화자에 따라 발화의 개수가 다르고 10개 미만인 데이터도 존재하여, 그런 데이터들은 임의로 random.choices를 통해 중복하여 추출
#### Predict
  - 'unknown' 화자 선택 방법 : max distance(threshold)
  - 각 화자마다의 centroid와 음성 d-vector 사이의 거리를 계산하여 가장 가까운 거리인 화자를 선택
  - 선택된 화자 centroid와 주변의 기존 d-vector들 중에서 가장 먼거리인 d-vector의 거리를 비교했을 때, 더 가깝거나 같으면 화자, 아니면 'unknown'으로 설정   
### 2-2. Model + CE Loss
  - Muliclass-classification 문제로 바꿔서 시도
#### Preprocessing
  - wav -> mel spectrogram으로 변환 (n_mel = 40 / 80) 
#### Model
  1) LSTM
  2) Transformer (encoder layer : mask / non-mask)
  3) Conformer (encoder layer : mask / non-mask)
#### Loss
  1) CrossEntropy Loss

#### Predict
- 'unknown' 화자 선택 방법 : threshold = 0.5/0.6 ...
    - softmax를 통해 나온 max value가 일정한 값이 넘지 못하면 'unknown'으로 설정

### 2-3. Model + LabelSmoothing Loss
  - Muliclass-classification 문제로 바꿔서 시도
#### Preprocessing
  1) wav -> mel spectrogram으로 변환 (n_mel = 80으로 변경) 
  2) vad 적용하여 소리부분만 추출
  3) vad 후 최소 length보다 짧으면 사용하지 않음
  4) 'unknown' 화자 생성하여 각 화자의 발화 데이터의 양이 기준 이하이면 식별하기에 데이터가 부족하다고 판단하여 unknown 화자 부여
    - 15개 미만 기준 : 화자 수 3182 -> 3023
    - 20개 미만 기준 : 화자 수 3182 -> 2119
#### Model
  1) LSTM
  2) Transformer (encoder layer : mask / non-mask)
  3) Conformer (encoder layer : mask / non-mask)
#### Loss
  1) LabelSmoothing Loss (rate = 0.1 / 0.6) 
#### Predict
- 'unknown' 화자 선택 방법
  1) softmax를 통해 나온 max value가 일정한 값이 넘지 못하면 'unknown'으로 설정
  2) unknown 화자 데이터 생성하여 학습(preprocess)


## 3. Result
- 그래프를 그렸을때 어느정도 분리가 된 것 같았으나, 화자가 3000명이 넘다보니 다소 겹치는 부분이 존재하여 다수의 화자를 식별하기에 적절하지 않다고 판단됨
- 'unknown' 데이터가 따로 없어서 'unknown' 화자를 식별하기 어려움
    1) threshold를 기준으로 unknown 부여
    2) 데이터가 적은 화자들을 unknown 화자로 지정 (발화개수가 15개 미만인 화자 : )
    3) 
- GE2E loss 구현시 cosine-similarity를 사용하여 d-vector간 상관관계를 계산했으나, predict할 때 max_distance를 euclidean을 사용
    1) predict code cos-sim으로 해보면 어떨까
    2) GE2E loss를 
  - 