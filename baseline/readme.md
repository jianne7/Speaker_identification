# 2022 인공지능 온라인 경진대회
## [자연어] 음성 보안 솔루션을 위한 화자 인식 문제

해당 코드는 https://github.com/VITA-Group/AutoSpeech 을 참조하여 경진대회 문제에 맞게 변형한 코드임을 밝힙니다.
### 코드 구조

```
${PROJECT}
├── config/
│   ├── __init__.py
│   ├── default.py
│   └── scripts/
│       ├── resnet18_config.yml
│       └── predict_config.yml
├── models/
│   ├── __init__.py
│   ├── model.py
│   ├── resnet.py
│   └── utils.py
├── modules/
│   ├── datasets.py
│   ├── earlystopper.py
│   ├── losses.py
│   ├── metrics.py
│   ├── optimizers.py
│   ├── params_data.py
│   ├── preprocessor.py
│   ├── recorders.py
│   ├── schedulers.py
│   ├── trainer.py
│   └── utils.py
├── preprocess.py
├── train.py
├── predict.py
├── README.md
└── requirements.txt
```

- config : 학습/추론에 필요한 파라미터 등을 기록하는 yml 파일
- models  
    - resnet.py : 모델 클래스
    - utils.py : 모델 관련 함수들
- modules
    - datasets.py : dataset 클래스 및 필요 함수
    - earlystopper.py : (본 코드에서 사용되지 않음)
    - losses.py : train.py에서 지정한 loss function을 리턴
    - metrics.py : trainer에서 필요한 metric 계산 함수
    - optimizers.py : (본 코드에서 사용되지 않음)
    - params_data.py : 오디오 데이터 파라미터 설정
    - recorders.py : checkpoint, log등을 기록
    - schedulers.py : (본 코드에서 사용되지 않음)
    - trainer.py : 에폭 별로 수행할 학습 과정, 추론 과정
    - utils.py : id를 매핑해주는 함수, 데이터셋 분할 함수, datasets.py나 preprocessor.py에 필요한 함수, 기타 함수 등
- preprocess.py : wav 파일을 npy 파일로 전처리 하기 위해 실행하는 코드
- train.py : 학습 시 실행하는 코드
- predict.py : 추론 시 실행하는 코드

---

### 데이터 구조
```
${PROJECT}
├── train/
│   └── speaker_ids/
│       └── files.wav
├── test/
│   └── files.wav

(preprocess 후 생성)
├── feature/
│   ├── train/
│       ├── speaker_ids/
│           ├── _sources.txt
│           ├── _sources_train.txt
│           ├── _sources_test.txt
│           └── files.npy
│   ├── test/
│       ├── _sources.txt
│       └── files.npy
│   └── merged/
│       └── train_speaker_id/
│           ├── _sources.txt
│           ├── _sources_train.txt
│           ├── _sources_test.txt
│           └── files.npy
│       ├── _sources.txt
│       └── test_files.npy
├── split.txt
├── train_meta.csv
├── std.npy
└── mean.py
```

### 사용 모델

ResNet-18


### 환경 셋팅
`pip install -r requirements.txt`

### 전처리

1. `modules/datasets.py` 의 dataset_root 값 변경
   예) dataset_root = Path('/path/to/directory') : train, test 폴더가 위치한 디렉토리
2. `python preprocess.py --dataset_root /path/to/directory `
3. feature 폴더 및 split.txt, train_meta.csv, std.npy, mean.npy 파일들이 생성됨 (데이터 구조 참고)

### 학습

1. `config/scripts/resnet18_config.yml` 수정
    1. DIR/DATA_DIR : 데이터 경로 지정 (train, test 폴더가 위치한 디렉토리)
    2. 이외 파라미터 조정
2. `python train.py --cfg config/scripts/resnet18_iden.yaml`
3. `logs/exp_directory/` 내에 결과(Log, Model등)가 저장됨


### 추론

1. `config/predict_config.yaml` 수정
    1. DIR/DATA_DIR : 데이터 경로 지정 (train, test 폴더가 위치한 디렉토리)
2. predict.py 수정
   1. sample_submission_path 수정
3. `python predict.py --cfg config/scripts/predict_config.yaml --load_path logs/resnet18_iden_2022_06_00_00_00_00/Model/checkpoint_best.pth --pred_path prediction.csv`
4. 지정한 path(pred_path)에 prediction.csv 파일이 저장됨
