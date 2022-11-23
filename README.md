# 2022 인공지능 온라인 경진대회

날짜: 2022년 6월 7일 오전 12:00 → 2022년 6월 21일 오전 11:00

# 음성 보안 솔루션을 위한 화자 인식 문제

## 1. 대회 목적

- 발화 음성을 듣고 인물 목록중 어느 화자인지 식별하는 문제 (Speaker Recognition)


## 2. 대회 주관

- 주최 : 과학기술정보통신부
- 주관 : 정보통신산업진흥원


## 3. 데이터셋

|File|Contents|Description|
|------|---|---|
|Train|Speaker ID|화자 ID로 된 파일|
|     |Utterance Audio|화자 ID에 해당하는 화자의 발화 오디오 데이터|
|Test|Utterance Audio|발화 오디오 데이터|


## 4. 진행 과정

### 4-1. DATA

- Train : 3182명의 화자, 총 74304개의 발화 음성 데이터
- Test : 9232개의 발화 음성 데이터

### 4-2. Train

### 1) LSTM + GE2E Loss

- **Preprocessing**
    
    - wav -> mel spectrogram으로 변환 (n_mel = 40)
    
- **Model**
    
    - LSTM으로 화자 벡터 추출
    
- **Loss**
    
    - GE2E Loss : Triplet loss와 유사한 개념으로 화자와 비슷할 수록 가깝게, 다를수록 멀리 떨어지게 해주는 방식
    
    - 논문에선 64명의 화자와 그 화자들마다 10개의 발화를 기준으로 학습
    
    - 본 경진대회에서는 화자에 따라 발화의 개수가 다르고 10개 미만인 데이터도 존재하여, 그런 데이터들은 임의로 random.choices를 통해 중복하여 추출
    
- **Predict**
    
    - 'unknown' 화자 선택 방법 : max distance(threshold)
    
    - 각 화자마다의 centroid와 음성 d-vector 사이의 거리를 계산하여 가장 가까운 거리인 화자를 선택
    
    - 선택된 화자 centroid와 주변의 기존 d-vector들 중에서 가장 먼거리인 d-vector의 거리를 비교했을 때, 더 가깝거나 같으면 화자, 아니면 'unknown'으로 설정
    

### 2)  Model + CE Loss

- Muliclass-classification 문제로 바꿔서 시도

- **Preprocessing**
    
    - wav -> mel spectrogram으로 변환 (n_mel = 40 / 80)
    
- **Model**
    
    1) LSTM
    
    2) Transformer (encoder layer : mask / non-mask)
    
    3) Conformer (encoder layer : mask / non-mask)
    
- **Loss**
    
    - CrossEntropy Loss
    
- **Predict**
    - 'unknown' 화자 선택 방법 : threshold = 0.5/0.6 ...
        
        - softmax를 통해 나온 max value가 일정한 값이 넘지 못하면 'unknown'으로 설정
        

### 3) Model + LabelSmoothing Loss

- Muliclass-classification 문제로 바꿔서 시도

- **Preprocessing**
    
    1) wav -> mel spectrogram으로 변환 (n_mel = 80으로 변경)
    
    2) vad 적용하여 소리부분만 추출
    
    3) vad 후 최소 length보다 짧으면 사용하지 않음
    
    4) 'unknown' 화자 생성하여 각 화자의 발화 데이터의 양이 기준 이하이면 식별하기에 데이터가 부족하다고 판단하여 unknown 화자 부여
    
    - 15개 미만 기준 : 화자 수 3182 -> 3023
    
    - 20개 미만 기준 : 화자 수 3182 -> 2119
    
- **Model**
    
    1) LSTM
    
    2) Transformer (encoder layer : mask / non-mask)
    
    3) Conformer (encoder layer : mask / non-mask)
    
- **Loss**
    
    - LabelSmoothing Loss (rate = 0.1 / 0.6)
    
- **Predict**
    - 'unknown' 화자 선택 방법
        
        1) softmax를 통해 나온 max value가 일정한 값이 넘지 못하면 'unknown'으로 설정
        
        2) unknown 화자 데이터 생성하여 학습(preprocess)
        

## 5. Result

### 1) LSTM + GE2E Loss

- GE2E loss 구현시 cosine-similarity를 사용하여 d-vector간 상관관계를 계산했으나, predict할 때 max_distance를 euclidean을 사용
- GE2E Loss를 이용해 학습후 그래프를 그렸을 때 어느정도 분리가 된 것 같았으나, 화자가 3000명이 넘다보니 다소 겹치는 부분이 존재하여 다수의 화자를 식별하기에 적절하지 않다고 판단됨
- **Accuracy**
    - **LSTM + GE2E Loss + Max distance : 0.35**

### 2) Model + CrossEntropy Loss

- 'unknown' 데이터가 따로 없어서 '*unknown*' 화자를 식별하기 어려움
    
    1) threshold를 기준으로 *unknown* 부여
    
- LSTM의 경우 Train loss는 잘 떨어졌으나 Valid loss가 증가함 ⇒ overfitting?
- Transformer, Conformer 모델의 경우 초반에는 조금씩 loss가 떨어졌으나 어느정도 떨어지면 더이상 떨어지지 않고 그대로 수렴됨
- **Accuracy**
    - **LSTM + CE Loss + thres : 0.11**

### 3) Model + LabelSmoothing Loss

- 'unknown' 데이터가 따로 없어서 '*unknown*' 화자를 식별하기 어려움
    
    1) threshold를 기준으로 *unknown* 부여
    
    2) 데이터가 적은 화자들을 *unknown* 화자로 지정
    
    3) VAD를 통해 음성없는 구간 제거후 길이가 짧아 사용할 수 없는 데이터 제거 후 데이터가 적은 화자들을 *unknown* 화자로 지정
    
- 클래스가 많다보니 너무 튀는 값을 방지하기 위해 labelsmoothing loss 사용해봄
- LSTM 모델의 경우 overfitting 문제는 해결됐으나 다른 모델처럼 loss가 더 떨어지지 않고 수렴되는 문제 발생
- Transformer, Conformer 모델 또한 마찬가지
- predict시 *unknown*이 나오지 않음
- optimizer를 Adam에서 SGD로 바꿔 해봤을때 Adam이 더 나았음
- **Accuracy**
    - **LSTM + LS Loss + thres : 0.19**

**⇒ 최종 결과 8등 (Baseline Resnet34사용)**


## 6. 문제 원인 분석

### 1) **Loss가 더 이상 안 떨어진 이유?**

- 클래스를 분별할만한 parameter의 수가 부족했을지도 ⇒ 모델 layer를 더 deep하게 설정해보기
- data 자체가 분별력이 떨어져서 그럴지도

### 2) **Unknow Class**

- unknown만 넣어서 채점했을때 0.34정도로, 즉 데이터가 없는 거에 비해 unknown 비율이 높았음
- 버린 데이터에 대해 unknown으로 만들지 말고 아예 버리고 해보면 나았을지도
왜냐면 다 다른 데이터를 한번에 하나의 클래스로 학습시켰기 때문에 오히려 학습할 때 좋지 않았을것

### 3) **About Model**

- Baseline에 있던 Resnet18/25/34 모델로 학습했을 때 더 좋은 성능이 나왔기때문에 시계열 모델이 아닌 CNN 모델로 다시 해봐도 좋을 듯

### 4) ETC..

- dataload 할 때 한 화자당 여러 개의 발화를 불러오고 mean을 해서 centroid를 만들어준 후 그 값으로 학습했으면 어땠을까?
- GE2E loss로 학습한 모델에 Classfier를 붙여 두가지를 다 학습했으면 어땠을까?
- 길이가 짧고 데이터 수가 적은 화자는 *unknown*이 아니라 아예 버리고 학습했으면 어땠을까?
- GE2E Loss 사용시 predict 할 때 euclidean 대신 cos-sim으로 계산했으면 어땠을까?
원래 GE2E loss 논문에서는 euclidean을 사용했는데 코드 구현시 cosine similarity로 상관관계 파악함

---

**Ref**

[GENERALIZED END-TO-END LOSS FOR SPEAKER VERIFICATION](https://www.notion.so/GENERALIZED-END-TO-END-LOSS-FOR-SPEAKER-VERIFICATION-a532d465b8a745918cb0e6bb36c8c372)
