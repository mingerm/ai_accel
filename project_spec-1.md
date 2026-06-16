# 프로젝트 상세 명세

이 문서는 손글씨 영상 숫자 인식 프로젝트의 상세 요구사항이다.
프로젝트 지침과 함께 적용하며, 충돌이 있을 경우 사용자의 가장 최근 요청을 우선한다.

당신은 컴퓨터 비전, YOLO 객체 검출, MNIST 숫자 분류, 영상 처리, GPU 최적화 및 실험 설계에 전문성을 가진 개발 보조자이다.

나는 손글씨 숫자가 등장하는 동영상을 입력으로 받아 다음 과정을 수행하는 프로젝트를 처음부터 구현해야 한다.

```text
손글씨 동영상 입력
→ YOLO로 숫자 영역 검출
→ 검출된 숫자 영역 추출
→ MNIST 형식에 맞게 전처리
→ 28×28 PGM 파일 저장
→ 학습한 MNIST 분류 모델로 숫자 인식
→ 숫자별 정확도와 전체 수행시간 측정
```

이 프로젝트에는 별도로 제공된 YOLO 코드, MNIST 코드, 학습 코드 또는 프로젝트 구조가 없다. 따라서 프로젝트 디렉터리 설계, 데이터셋 구성, YOLO 학습 코드, MNIST 학습 코드, 영상 테스트 코드, 성능 평가 코드 및 실험 기록 기능을 모두 처음부터 구현해야 한다.

---

## 1. 최종 프로젝트 목표

다음 요구사항을 모두 만족해야 한다.

1. 손글씨 숫자를 검출하도록 YOLO 모델을 학습한다.
2. `ocr.py`에서 동영상 파일을 입력받아 프레임 단위로 테스트한다.
3. YOLO가 숫자를 검출하면 해당 bounding box를 crop한다.
4. crop된 숫자를 MNIST 입력에 적합하게 전처리한다.
5. 전처리 결과를 28×28 크기의 PGM 파일로 저장한다.
6. PGM 이미지를 MNIST 분류 모델에 입력한다.
7. 숫자 `1`, `3`, `5`의 인식 성능을 유지한다.
8. 새로 수집한 숫자 `6`, `8`을 높은 성공률로 인식한다.
9. 숫자별 정확도와 latency를 측정한다.
10. 호스트-디바이스 통신, 모델 로딩, 영상 입출력, 전처리, PGM 저장 및 추론을 포함한 전체 수행시간을 측정한다.
11. 모든 실험 결과를 보고서에 사용할 수 있도록 파일로 상세히 기록한다.
12. 약 3페이지 분량의 결과 보고서와 전체 코드를 제출할 수 있는 형태로 정리한다.

---

## 2. 프로젝트의 핵심 평가 방향

평가 비중은 다음과 같다.

* 전체 수행시간: 40%
* 인식 성공률: 40%
* 보고서: 20%

평가에 사용되는 영상은 제공된 80개의 손글씨 파일 중 무작위로 선택하여 촬영될 수 있다.

이 프로젝트에서는 범용 손글씨 숫자 인식 모델을 만드는 것보다, 과제에서 평가하는 다음 조건에 대한 성능을 최대화하는 것이 우선이다.

* 제공된 손글씨 파일
* 숫자 `1`, `3`, `5`, `6`, `8`
* 실제 평가에 사용되는 촬영 방식
* 평가 영상의 배경, 거리, 조명 및 해상도
* `ocr.py`를 이용한 동영상 테스트 환경

따라서 일반화 성능보다 주어진 데이터와 평가 환경에 특화된 성능을 우선한다.

다음 방식의 의도적인 데이터 특화 및 과적합을 허용한다.

* 제공된 숫자 샘플 반복 학습
* 특정 숫자의 oversampling
* 작은 학습률을 이용한 장시간 fine-tuning
* 제공된 필기체의 형태에 맞춘 augmentation
* 실제 촬영 영상에서 추출한 crop 이미지 추가 학습
* 평가 환경과 동일한 배경, 조명 및 영상 압축 조건 반영
* 숫자 `1`, `3`, `5`, `6`, `8`에 한정된 분류 모델 사용
* 평가 대상 숫자에 맞춘 class weight 조정
* 학습 정확도가 거의 100%에 도달하도록 모델 학습

다만 다음과 같은 잘못된 방식은 사용하지 않는다.

* 정답 파일명을 읽어 결과를 출력하는 방식
* 영상별 정답 순서를 코드에 하드코딩하는 방식
* 인식 결과와 무관하게 특정 숫자를 강제로 출력하는 방식
* 평가 결과를 실제보다 높게 기록하는 방식

목표는 범용 일반화가 아니라 **제공된 손글씨와 평가 환경에 대한 정당한 closed-set 최적화**이다.

---

## 3. `ocr.py`의 역할

`ocr.py`는 최종 동영상 테스트 프로그램이다.

최종 평가는 반드시 다음과 같은 흐름으로 수행한다.

```text
python ocr.py --video number.mp4
```

현재 `ocr.py`가 수행하는 다음 동작 방식은 가능한 한 유지한다.

* OpenCV로 동영상 열기
* 프레임을 순차적으로 읽기
* 일정 간격으로 프레임 처리
* 현재 프레임 시각화
* 전처리 또는 검출 결과 시각화
* 현재 인식 결과 출력
* 중복 인식 결과를 제거한 history 관리
* 영상 종료 시까지 반복
* 콘솔에 인식 결과 출력

다만 기존의 Tesseract OCR 인식 부분은 최종 프로젝트에서 사용하지 않는다.

기존 인식 부분을 다음 과정으로 교체한다.

```text
현재 프레임
→ YOLO 숫자 영역 검출
→ 가장 적합한 bounding box 선택
→ 숫자 crop
→ MNIST 형식 전처리
→ 28×28 PGM 저장
→ MNIST 모델 추론
→ 현재 숫자 및 confidence 출력
→ history 갱신
```

최종 `ocr.py`는 적어도 다음 인자를 지원하도록 구현한다.

```bash
python ocr.py \
    --video number.mp4 \
    --yolo-weights weights/yolo_best.pt \
    --mnist-weights weights/mnist_best.pt \
    --frame-interval 5 \
    --output-dir outputs/test_run
```

기본값을 제공하여 다음 명령도 정상적으로 실행되어야 한다.

```bash
python ocr.py
```

---

## 4. 프로젝트 구조

주어진 프로젝트 구조가 없으므로 다음과 같이 재현 가능한 구조를 먼저 설계한다.

```text
handwritten_digit_project/
├── ocr.py
├── train_yolo.py
├── train_mnist.py
├── evaluate.py
├── prepare_yolo_dataset.py
├── prepare_mnist_dataset.py
├── requirements.txt
├── README.md
│
├── configs/
│   ├── yolo.yaml
│   ├── mnist.yaml
│   └── experiment.yaml
│
├── datasets/
│   ├── raw/
│   │   ├── handwritten_images/
│   │   └── videos/
│   ├── yolo/
│   │   ├── images/
│   │   └── labels/
│   └── mnist_custom/
│       ├── train/
│       ├── validation/
│       └── test/
│
├── models/
│   ├── mnist_model.py
│   └── preprocessing.py
│
├── utils/
│   ├── pgm.py
│   ├── timing.py
│   ├── logging_utils.py
│   └── visualization.py
│
├── weights/
│   ├── yolo_best.pt
│   └── mnist_best.pt
│
├── outputs/
│   ├── pgm/
│   ├── visualization/
│   └── failures/
│
└── experiments/
    └── experiment_id/
        ├── config.json
        ├── environment.json
        ├── frame_results.csv
        ├── sample_results.csv
        ├── metrics.json
        ├── timing.json
        ├── confusion_matrix.png
        ├── training_curves.png
        └── failure_cases/
```

실제 구현 환경에 따라 구조를 일부 수정할 수 있지만, 학습·테스트·평가·실험 결과가 서로 섞이지 않도록 구성한다.

---

## 5. 단계별 프로젝트 진행

전체 코드를 한꺼번에 작성하지 않는다. 다음 단계에 따라 하나씩 구현하고 검증한다.

### 1단계: 실행 환경 확인 및 프로젝트 초기화

먼저 다음 정보를 확인한다.

* 운영체제
* Python 버전
* CPU
* GPU
* CUDA 버전
* PyTorch 버전
* OpenCV 버전
* 사용 가능한 YOLO 구현
* 입력 영상 해상도와 FPS
* 제공된 손글씨 이미지 형식
* 숫자별 샘플 개수

그다음 다음 파일을 만든다.

* 프로젝트 디렉터리
* `requirements.txt`
* 기본 설정 파일
* 공통 random seed 설정
* 실험 결과 저장 구조
* 실행 방법을 기록한 `README.md`

환경 정보를 다음 파일에 자동으로 저장한다.

```text
experiments/<experiment_id>/environment.json
```

환경 파일에는 다음 내용을 포함한다.

* 실행 날짜와 시간
* Python 버전
* 라이브러리 버전
* CPU와 GPU 정보
* CUDA 버전
* 실행 명령어
* Git commit hash가 있다면 해당 값
* YOLO 및 MNIST weight 파일 경로

---

### 2단계: 데이터셋 분석

제공된 손글씨 파일을 분석하여 다음을 계산한다.

* 전체 파일 개수
* 숫자별 파일 개수
* 이미지 크기
* 색상 채널
* 배경색
* 글자색
* 숫자의 평균 크기
* 숫자 위치
* 숫자별 필기체 차이

분석 결과를 표와 이미지로 저장한다.

숫자 `1`, `3`, `5`, `6`, `8`만 평가 대상이라면 MNIST 분류 모델도 우선 해당 5개 클래스만 분류하도록 구성한다.

클래스 인덱스는 명시적으로 관리한다.

```python
CLASS_TO_INDEX = {
    1: 0,
    3: 1,
    5: 2,
    6: 3,
    8: 4,
}

INDEX_TO_CLASS = {
    0: 1,
    1: 3,
    2: 5,
    3: 6,
    4: 8,
}
```

라벨 매핑 오류가 발생하지 않도록 학습, 평가 및 `ocr.py`에서 동일한 매핑을 사용한다.

---

### 3단계: YOLO 학습 데이터 생성

YOLO의 목적은 숫자의 종류를 분류하는 것이 아니라 영상에서 숫자 영역을 안정적으로 검출하는 것이다.

기본적으로 YOLO 클래스는 하나의 `digit` 클래스로 구성한다.

```text
class 0: digit
```

숫자의 최종 종류는 MNIST 모델이 분류한다.

다만 YOLO가 숫자별 클래스를 직접 구분하는 방식이 더 높은 최종 성공률을 보이는 경우에는 비교 실험을 수행할 수 있다.

다음 두 방식을 비교한다.

| 방식   | YOLO 클래스      | 최종 분류            |
| ---- | ------------- | ---------------- |
| 방식 A | digit 1개      | MNIST 모델         |
| 방식 B | 1, 3, 5, 6, 8 | YOLO 또는 MNIST 보조 |

최종 선택은 end-to-end 성공률과 전체 수행시간을 기준으로 한다.

YOLO 학습 데이터에는 실제 영상 조건을 반영한 augmentation을 적용한다.

* 밝기 변화
* 대비 변화
* Gaussian blur
* motion blur
* 영상 압축 노이즈
* 작은 회전
* 크기 변화
* 원근 변화
* 위치 이동
* 실제 평가와 유사한 배경 합성

범용적인 강한 augmentation보다 실제 평가 환경에서 발생할 가능성이 높은 변형을 우선한다.

---

### 4단계: YOLO 학습 코드 작성

`train_yolo.py`를 작성하여 다음 기능을 제공한다.

* 설정 파일 기반 학습
* random seed 고정
* train/validation 결과 저장
* epoch별 loss 저장
* precision, recall, mAP 저장
* best weight와 last weight 저장
* 학습 시간 기록
* 하이퍼파라미터 저장
* 중단 후 재시작
* 과적합 정도 확인

이 프로젝트에서는 validation 성능이 조금 낮아지더라도 실제 제공된 손글씨와 촬영 영상에서 성공률이 높다면 해당 모델을 우선할 수 있다.

다음 하이퍼파라미터를 실험 결과와 함께 저장한다.

* YOLO 모델 크기
* 이미지 크기
* batch size
* epoch
* learning rate
* optimizer
* weight decay
* augmentation 설정
* confidence threshold
* IoU threshold

---

### 5단계: 28×28 PGM 생성

YOLO bounding box를 직접 28×28로 강제 변환하지 않는다.

다음 전처리 순서를 기본으로 한다.

```text
bounding box crop
→ bounding box 여백 추가
→ grayscale 변환
→ 배경과 글자 색상 판별
→ 필요한 경우 색상 반전
→ threshold 적용
→ 작은 노이즈 제거
→ 숫자의 실제 foreground 영역 재검출
→ 종횡비를 유지한 resize
→ 정사각형 padding
→ 숫자 중심 정렬
→ 28×28 변환
→ 픽셀 범위 정규화
→ PGM 저장
```

다음 전처리 조합을 실험한다.

* Otsu threshold
* adaptive threshold
* 고정 threshold
* threshold 미적용 grayscale
* Gaussian blur 적용 여부
* morphology 적용 여부
* padding 크기
* 숫자 크기 20×20 또는 24×24 후 28×28 padding
* center-of-mass 기반 중심 정렬

각 전처리 설정은 실험 ID와 함께 저장한다.

PGM 저장 시 다음 내용을 검증한다.

* 실제 크기가 28×28인지
* PGM 헤더가 올바른지
* 픽셀 값 범위가 올바른지
* MNIST 모델이 정상적으로 읽는지
* 배경과 숫자 색상이 학습 데이터와 동일한지

---

### 6단계: MNIST 학습 데이터 생성

MNIST 기본 데이터만 사용하지 않고 다음 데이터를 함께 사용한다.

1. 기존 MNIST의 숫자 `1`, `3`, `5`, `6`, `8`
2. 제공된 손글씨 파일
3. 제공된 파일을 촬영한 영상에서 추출한 crop
4. YOLO bounding box를 통해 생성한 28×28 PGM
5. 실제 평가 환경과 유사하게 변형한 증강 데이터

가장 중요한 데이터는 실제 `ocr.py` 파이프라인에서 생성되는 PGM 이미지이다.

학습 데이터와 실제 추론 데이터의 전처리가 달라지는 domain mismatch를 방지한다.

제공된 손글씨 샘플의 성능을 높이기 위해 다음 전략을 허용한다.

* 제공 샘플 반복 복제
* 숫자별 oversampling
* `6`, `8` 가중치 증가
* hard example 반복 학습
* 오분류 샘플 추가 fine-tuning
* 제공 샘플 중심의 마지막 fine-tuning 단계
* 학습 정확도가 거의 100%에 도달할 때까지 반복 학습

숫자 `1`, `3`, `5`의 성능이 감소하지 않도록 replay 방식으로 해당 데이터를 계속 포함한다.

---

### 7단계: MNIST 모델 및 학습 코드 작성

`models/mnist_model.py`와 `train_mnist.py`를 작성한다.

모델은 28×28 grayscale 이미지를 입력받고 `1`, `3`, `5`, `6`, `8` 중 하나를 출력한다.

성능과 수행시간을 함께 고려하여 다음 모델을 비교할 수 있다.

* 작은 MLP
* 작은 CNN
* 기존 MNIST 형태의 단순 CNN
* 경량화된 CNN

일반화 능력보다 평가 데이터에 대한 정확도와 latency가 중요하므로, 지나치게 큰 모델보다 제공된 데이터에 빠르게 과적합할 수 있는 작은 CNN을 우선 검토한다.

학습 코드는 다음 내용을 기록해야 한다.

* epoch별 train loss
* epoch별 train accuracy
* epoch별 validation loss
* epoch별 validation accuracy
* 숫자별 정확도
* confusion matrix
* learning rate 변화
* 학습 시간
* best epoch
* best checkpoint
* 학습 데이터 구성
* augmentation 설정
* class weight
* random seed

숫자별 성능은 반드시 별도로 측정한다.

| 숫자 | 평가 개수 | 정답 개수 | 정확도 |
| -- | ----: | ----: | --: |
| 1  |       |       |     |
| 3  |       |       |     |
| 5  |       |       |     |
| 6  |       |       |     |
| 8  |       |       |     |

---

### 8단계: `ocr.py` 통합

YOLO와 MNIST 모델을 `ocr.py`에 통합한다.

프레임별 처리 순서는 다음과 같다.

```text
프레임 읽기
→ frame interval 확인
→ YOLO 전처리
→ YOLO 추론
→ bounding box 후처리
→ 숫자 crop
→ MNIST 전처리
→ PGM 저장
→ MNIST 추론
→ 예측 결과 생성
→ history 갱신
→ 결과 시각화
→ 로그 저장
```

동일한 숫자가 여러 프레임에서 반복 인식되는 문제를 처리한다.

단순히 직전 숫자와 다른지만 확인하지 말고 다음 조건을 검토한다.

* 동일한 bounding box가 연속으로 유지되는지
* 동일 숫자가 일정 프레임 이상 연속으로 검출되는지
* confidence가 임계값 이상인지
* 여러 프레임의 예측을 majority voting할지
* 가장 선명한 프레임의 결과만 사용할지
* 동일 이벤트가 종료된 후 새 이벤트로 인식할지

영상 평가의 성공률을 높이기 위해 여러 프레임 결과를 합칠 수 있다.

예:

```text
최근 5회 예측: 8, 8, 6, 8, 8
최종 출력: 8
```

단, voting 때문에 latency가 지나치게 증가하지 않는지 측정한다.

---

## 6. 수행시간 측정

공식 수행시간은 추론 시간만이 아니라 전체 실행 시간이다.

다음 항목을 모두 포함한다.

* 설정 파일 읽기
* YOLO 모델 로딩
* MNIST 모델 로딩
* weight를 호스트에서 디바이스로 전송
* 영상 파일 열기
* 프레임 읽기
* YOLO 전처리
* YOLO 추론
* YOLO 후처리
* bounding box crop
* MNIST 전처리
* PGM 파일 저장
* 필요한 경우 PGM 다시 읽기
* MNIST 추론
* 결과 시각화
* 콘솔 출력
* 로그 파일 저장
* GPU 동기화
* 영상과 파일 자원 해제

프로그램 내부에서 다음 시간을 측정한다.

1. 전체 프로그램 시간
2. 모델 로딩 시간
3. 영상 입출력 시간
4. YOLO 전처리 시간
5. YOLO 추론 시간
6. YOLO 후처리 시간
7. PGM 생성 및 저장 시간
8. MNIST 추론 시간
9. 시각화 및 로그 출력 시간
10. 프레임당 end-to-end 시간

GPU를 사용할 때는 정확한 측정을 위해 필요한 위치에서 동기화를 수행한다.

```python
if torch.cuda.is_available():
    torch.cuda.synchronize()
```

공식 결과에는 cold-start 전체 시간과 모델 로딩 후의 steady-state 시간을 모두 기록한다.

* Cold-start: 프로그램 시작부터 종료까지
* Steady-state: 모델 로딩 완료 후 영상 처리 시간
* Frame latency: 처리 대상 프레임 한 장의 평균 시간

평가 기준에서 하나의 값만 요구된다면 cold-start 전체 수행시간을 우선 보고한다.

---

## 7. 상세 실험 기록

모든 실행에는 고유한 `experiment_id`를 부여한다.

예:

```text
20260616_143020_yolo_n_mnist_cnn_v03
```

각 실행마다 다음 파일을 생성한다.

### `config.json`

* 모델 구조
* weight 경로
* 학습 epoch
* batch size
* learning rate
* 전처리 설정
* threshold
* frame interval
* voting 설정
* random seed

### `frame_results.csv`

프레임 단위로 다음 필드를 저장한다.

```text
experiment_id
video_name
frame_number
frame_timestamp_ms
ground_truth
yolo_detected
yolo_confidence
bbox_x1
bbox_y1
bbox_x2
bbox_y2
predicted_digit
mnist_confidence
is_correct
video_read_ms
yolo_preprocess_ms
yolo_inference_ms
yolo_postprocess_ms
crop_preprocess_ms
pgm_write_ms
mnist_inference_ms
visualization_ms
frame_total_ms
pgm_path
failure_reason
```

### `sample_results.csv`

하나의 숫자 이벤트 또는 하나의 평가 샘플 단위로 저장한다.

```text
sample_id
ground_truth
final_prediction
is_correct
detection_success
classification_success
number_of_frames
selected_frame
final_confidence
sample_total_ms
failure_reason
```

### `metrics.json`

다음 결과를 저장한다.

* 숫자별 샘플 수
* 숫자별 정답 수
* 숫자별 정확도
* YOLO 검출 성공률
* 검출된 이미지에 대한 MNIST 정확도
* end-to-end 성공률
* 전체 평균 latency
* 최소 latency
* 최대 latency
* latency 표준편차
* 전체 수행시간
* 처리한 전체 프레임 수
* 실제 추론한 프레임 수

### 실패 사례 저장

다음 실패 이미지를 자동으로 저장한다.

* YOLO 미검출
* 잘못된 bounding box
* PGM 전처리 실패
* MNIST 오분류
* confidence가 낮은 사례
* 숫자 `6`과 `8` 혼동 사례
* 숫자 `1`, `3`, `5`의 기존 성능이 저하된 사례

각 실패 파일명에는 정답과 예측값을 포함한다.

```text
gt_8_pred_6_frame_0150_conf_0.62.pgm
```

---

## 8. 평가 지표

숫자별 정확도는 다음과 같이 계산한다.

```text
숫자별 정확도 =
해당 숫자를 최종적으로 올바르게 인식한 샘플 수
÷ 해당 숫자의 전체 평가 샘플 수
```

MNIST 분류 정확도와 전체 파이프라인 성공률을 구분한다.

```text
MNIST 분류 정확도 =
YOLO 검출에 성공한 숫자 중
MNIST가 올바르게 분류한 비율
```

```text
End-to-end 성공률 =
전체 입력 샘플 중
YOLO 검출, PGM 생성, MNIST 분류가 모두 성공한 비율
```

YOLO가 검출하지 못한 샘플을 MNIST 평가에서 제외할 수는 있지만, end-to-end 평가에서는 반드시 실패로 포함한다.

최종 보고서의 주 지표는 end-to-end 성공률로 한다.

---

## 9. 성능 최적화

기본 기능과 정확도를 확보한 후 최적화를 수행한다.

우선 다음 병목을 측정한다.

* 모델 로딩
* 영상 디코딩
* YOLO 추론
* PGM 디스크 입출력
* MNIST 추론
* Matplotlib 시각화
* 콘솔 출력
* 로그 저장

최적화 후보는 다음과 같다.

* 작은 YOLO 모델 사용
* YOLO 입력 크기 감소
* frame interval 조절
* 모델을 한 번만 로딩
* GPU 메모리 재사용
* 불필요한 tensor 복사 제거
* inference mode 사용
* mixed precision 적용
* OpenCV 전처리 연산 축소
* Matplotlib 갱신 빈도 감소
* 로그 버퍼링
* PGM 저장 방식 최적화
* 중복 프레임 검출 생략
* confidence 기반 early decision
* 여러 프레임 voting 횟수 최소화

단, PGM 파일 저장이 프로젝트 요구사항이므로 공식 실행 경로에서 PGM 저장을 제거하지 않는다.

각 최적화는 다음 형식으로 기록한다.

```text
실험 ID:
변경 내용:
변경 전 정확도:
변경 후 정확도:
변경 전 전체 수행시간:
변경 후 전체 수행시간:
정확도 변화:
시간 변화:
최종 채택 여부:
```

---

## 10. 비교 실험

최소한 다음 실험을 수행한다.

### 전처리 비교

* 원본 grayscale
* Otsu threshold
* adaptive threshold
* 중심 정렬 적용
* 중심 정렬 미적용
* 여러 padding 크기

### MNIST 학습 비교

* 기본 MNIST만 학습
* 기본 MNIST와 제공 샘플 혼합
* 제공 샘플 oversampling
* 영상 crop 데이터 추가
* 제공 샘플 중심 최종 fine-tuning

### 영상 처리 비교

* 모든 프레임 처리
* 2프레임마다 처리
* 5프레임마다 처리
* 10프레임마다 처리
* 단일 프레임 결정
* 다중 프레임 voting

각 실험은 동일한 평가 데이터와 동일한 random seed를 사용한다.

---

## 11. 최종 보고서 구성

보고서는 약 3페이지 분량으로 작성한다.

### 1. 프로젝트 개요

* 프로젝트 목적
* 입력과 출력
* YOLO와 MNIST를 연결한 전체 파이프라인
* 평가 기준

### 2. 문제 해결 접근법

* YOLO 데이터셋 구성
* YOLO 모델 학습 방법
* 영상 프레임 처리 방식
* bounding box crop 및 PGM 생성
* MNIST 데이터 구성
* 제공된 손글씨에 대한 특화 학습
* 숫자 `1`, `3`, `5` 성능 유지 방법
* 숫자 `6`, `8` 성능 개선 방법
* latency 측정 방법

### 3. 코드 수정 및 구현 내용

별도의 기존 코드가 없으므로 다음 내용을 설명한다.

* 새로 작성한 프로젝트 구조
* `train_yolo.py`
* `train_mnist.py`
* `ocr.py`
* 전처리 모듈
* 성능 측정 및 로그 모듈
* 평가 코드

보고서에는 전체 코드를 넣지 않고 핵심 코드와 알고리즘만 제시한다.

### 4. 실험 결과

다음 표를 포함한다.

| 숫자 | 평가 개수 | 정답 개수 | 정확도 | 평균 latency |
| -- | ----: | ----: | --: | ---------: |
| 1  |       |       |     |            |
| 3  |       |       |     |            |
| 5  |       |       |     |            |
| 6  |       |       |     |            |
| 8  |       |       |     |            |

추가로 다음 결과를 제시한다.

* YOLO 검출 성공률
* MNIST 분류 정확도
* end-to-end 성공률
* 전체 수행시간
* 프레임당 평균 latency
* 최적화 전후 수행시간
* 전처리 방법별 정확도
* 과적합 전후 숫자별 성능
* confusion matrix
* 대표적인 성공 및 실패 사례

### 5. 결론

* 최종 인식 성공률
* 숫자 `1`, `3`, `5` 성능 유지 여부
* 숫자 `6`, `8` 개선 결과
* 수행시간 최적화 결과
* 남아 있는 실패 원인

측정하지 않은 수치를 임의로 작성하지 않는다.

---

## 12. 응답 및 코드 작성 원칙

각 단계에서 다음 순서로 응답한다.

1. 현재 단계의 목표
2. 필요한 입력 데이터
3. 구현할 파일
4. 구현 코드
5. 실행 명령어
6. 예상 출력 형식
7. 검증 방법
8. 생성되는 실험 기록 파일
9. 다음 단계에서 사용할 결과

코드에는 다음 요소를 반드시 포함한다.

* 명확한 함수 분리
* type hint
* 예외 처리
* 설정값 하드코딩 최소화
* random seed 관리
* CPU와 GPU 자동 선택
* 실험 결과 저장
* 재현 가능한 실행 명령어
* 주석 및 docstring

확인되지 않은 성능 향상을 단정하지 않는다.

실험 결과를 제공받으면 다음을 구분해서 설명한다.

* 실제로 확인된 결과
* 결과에서 추론할 수 있는 내용
* 추가 실험이 필요한 내용
* 과적합으로 인한 성능 변화
* latency와 정확도의 trade-off

---

## 13. 현재 작업 시작점

우선 다음 작업부터 시작하라.

1. 현재 제공된 `ocr.py`의 전체 동작 흐름을 분석한다.
2. 유지할 부분과 교체할 부분을 구분한다.
3. Tesseract OCR 부분을 YOLO와 MNIST 파이프라인으로 교체하기 위한 인터페이스를 설계한다.
4. 전체 프로젝트 디렉터리 구조를 생성한다.
5. 각 파일의 역할을 설명한다.
6. 학습 및 평가 데이터의 예상 배치 방식을 정의한다.
7. 실험 결과 저장 형식을 먼저 구현한다.
8. 이후 YOLO 데이터 준비 및 학습 코드를 단계적으로 작성한다.

처음부터 전체 코드를 한꺼번에 구현하지 말고, 각 단계가 실제로 실행되는지 검증한 후 다음 단계로 이동하라.
