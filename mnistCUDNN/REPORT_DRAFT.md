# 최종 보고서 초안

## 1. 프로젝트 목표

본 프로젝트의 목표는 손글씨 숫자가 촬영된 샘플 영상에서 숫자 영역을 검출하고, 검출된
숫자 영역을 MNIST 형식의 28x28 PGM 이미지로 변환한 뒤, 기존 mnistCUDNN 기반
분류기에 입력하여 숫자 인식 결과와 수행 시간을 측정하는 것이다. 최종 평가는
숫자 `1`, `3`, `5`, `6`, `8`을 대상으로 하며, 인식 성공률과 전체 수행 시간을
함께 고려한다.

## 2. 최종 시스템 구조

```text
sample video frame
  -> YOLO digit detector
  -> bbox crop
  -> MNIST-style preprocessing
  -> 28x28 PGM save
  -> mnistCUDNN inference
  -> prediction, accuracy, latency summary
```

최종 shell 평가 진입점은 `run_pgm_all.sh`이다. 시연자는 `mnistCUDNN/` 폴더에서 이
스크립트만 실행하면 된다. 과제 명세의 동영상 진입점인 `ocr.py --video`도 같은
파이프라인을 호출하도록 구성했다. 스크립트는 사전 점검, YOLO 감지, PGM 저장,
mnistCUDNN 추론, 결과 요약 출력을 한 번에 수행한다.

주요 파일은 다음과 같다.

- `yolo/config.yaml`: 카메라 입력, YOLO weight, confidence, 중복 제거 기준 설정
- `yolo/detect_video.py`: 카메라/영상 프레임에서 숫자를 감지하고 PGM 저장
- `yolo/dedupe.py`: 같은 숫자가 연속 프레임에 반복 저장되지 않도록 segment 관리
- `yolo/save_pgm.py`: crop 이미지를 MNIST 스타일 28x28 PGM으로 변환
- `scripts/run_pipeline.sh`: YOLO 실행 후 PGM을 mnistCUDNN에 넣고 정확도/시간 측정
- `scripts/run_mnist_one.sh`: PGM 한 장 또는 batch 입력을 기존 MNIST 실행 방식에 연결
- `ocr.py`: `--video` 입력은 전체 파이프라인으로, PGM 입력은 단일 MNIST 추론으로 연결
- `data/*.bin`: mnistCUDNN이 사용하는 최종 LeNet weight
- `run_pgm_all.sh`: 최종 실험 정답 순서를 `answers=(...)` 배열로 입력

## 3. 학습 및 데이터 구성

YOLO 검출기는 제공된 손글씨 숫자 이미지와 추가 수집 데이터를 사용해 학습했다.
`yolo/train_config.yaml` 기준 학습 대상 class는 `1`, `3`, `5`, `6`, `8`이며,
class별 train/validation 샘플을 생성해 `datasets/yolo/` 구조로 저장한다. 학습
base checkpoint는 `yolo11n.pt`이고, 학습 후 최종 detector는
`yolo/weights/best.pt`로 복사되도록 구성했다.

MNIST 분류기는 EMNIST digit 데이터와 프로젝트에서 수집한 숫자 이미지를 함께
사용해 LeNet 호환 weight를 export한다. 최종 C++ 실행 파일은 `data/conv1.bin`,
`data/conv2.bin`, `data/ip1.bin`, `data/ip2.bin` 및 각 bias 파일을 읽는다.
이전 weight는 `data_old/`에 백업하여 최종 weight와 비교할 수 있게 했다.

## 4. 영상 인식 파이프라인

최종 실험 결과는 샘플 영상을 기준으로 작성한다. 기본 설정은 카메라 입력도
지원하지만, 보고서 수치 산출 시에는 동일한 샘플 영상을 `VIDEO_PATH`로 지정해
재현 가능한 결과를 사용한다.

```bash
VIDEO_PATH=videos/sample.mp4 ./run_pgm_all.sh
python3 ocr.py --video videos/sample.mp4
YOLO_WEIGHTS=yolo/weights/best.pt ./run_pgm_all.sh
```

YOLO는 한 프레임에서 여러 bbox가 나올 수 있으므로 `selection: center_conf`로
confidence와 화면 중앙성을 함께 고려해 현재 인식 대상 숫자를 선택한다. 같은
숫자가 여러 프레임 동안 계속 보이는 경우에는 `SegmentTracker`가 안정 프레임 수,
missing frame 수, 저장 간격을 이용해 같은 등장 구간에서 PGM을 한 장만 저장한다.

PGM 변환은 다음 순서로 수행한다.

```text
bbox crop -> grayscale -> foreground mask -> digit component selection
-> 28x28 canvas centering -> center-of-mass alignment -> binary PGM save
```

보고서 확인용으로 `save_debug_crops: true`를 켜면 원본 crop과 확대된 PGM preview가
`pgm_output/debug/`에 PNG로 저장된다.

## 5. 실험 방법

최종 실험 전 `run_pgm_all.sh` 상단의 `answers` 배열에 샘플 영상의 정답 순서를
저장한다.

```bash
answers=(6 8 8 6)
```

YOLO가 손글씨 등장 개수보다 많은 PGM을 저장할 수 있으므로, 평가는 정답 배열
개수만큼 처음부터 순서대로 수행한다. 예를 들어 샘플 영상의 손글씨가 12개이고
PGM이 20개 생성되면 처음 12개만 MNIST 입력으로 사용한다.

그 다음 아래 명령으로 전체 파이프라인을 실행한다.

```bash
VIDEO_PATH=videos/sample.mp4 PIPELINE_REPORT=reports/sample_video_report.txt ./run_pgm_all.sh
```

MNIST 최적화 전후 시간을 비교할 때는 같은 PGM 입력을 유지한 뒤 다음 명령을
사용한다.

```bash
bash scripts/profile_mnist_pipeline.sh
```

## 6. 시간 측정 기준

MNIST inference time은 과제에서 지정한 구간과 동일하게 측정한다. 이미지 파일
읽기, PGM 전처리, host-to-device 입력 복사, device-to-host 결과 복사는 제외하고,
입력 복사와 변수 초기화가 끝난 직후부터 forward 연산 전체와
`cudaDeviceSynchronize()`가 끝나는 시점까지 측정한다.

```text
conv1 -> pool1 -> conv2 -> pool2 -> fc1 -> relu -> lrn -> fc2 -> softmax -> cudaDeviceSynchronize
```

Shell 출력의 `Inference time`은 이 구간의 latency이며, 총 시간은 평가 대상 PGM의
latency를 합산한 값이다.

## 7. 샘플 영상 실험 결과

아래 표는 최종 실행 후 `reports/sample_video_report.txt`와 콘솔 SUMMARY를
기준으로 채운다.

| 항목 | 결과 |
| --- | --- |
| 입력 방식 | 샘플 영상 |
| 샘플 영상 경로 | `videos/sample.mp4` |
| 정답 순서 | TODO |
| 저장된 PGM 수 | TODO |
| 평가 이미지 수 | TODO |
| Correct Predictions | TODO |
| Accuracy | TODO |
| Total Inference Time | TODO ms |
| Average Latency | TODO ms |
| Pipeline Wall Time | TODO ms |

숫자별 정확도:

| 숫자 | 정답 수 | 정답 예측 수 | 정확도 |
| --- | ---: | ---: | ---: |
| 1 | TODO | TODO | TODO |
| 3 | TODO | TODO | TODO |
| 5 | TODO | TODO | TODO |
| 6 | TODO | TODO | TODO |
| 8 | TODO | TODO | TODO |

## 8. 최적화 및 개선점

기존 방식은 PGM마다 모델 초기화, weight 로딩, CUDA buffer 준비가 반복될 수
있었다. 최종 파이프라인은 batch 실행이 가능하면 여러 PGM을 한 번에 처리해 반복
초기화 비용을 줄인다. 또한 `PIPELINE_REPORT`와 profile CSV를 통해 전체 latency,
평균 latency, phase별 시간을 기록할 수 있게 했다.

PGM 품질 개선을 위해 YOLO bbox 주변 padding, foreground mask 후보 선택,
connected component filtering, 중심 정렬을 적용했다. 카메라 환경에서는 조명과
거리 변화가 커서 단순 threshold 하나만 사용하는 방식보다 여러 mask 후보를
점수화하는 방식이 더 안정적이었다.

## 9. 제출 전 체크리스트

- `python3 -m pip install -r requirements-yolo.txt`가 완료되어야 한다.
- 최종 YOLO detector가 `yolo/weights/best.pt`에 있어야 한다.
- 최종 mnistCUDNN weight가 `data/*.bin`에 있어야 한다.
- `run_pgm_all.sh`의 `answers=(...)` 정답 개수와 샘플 영상의 손글씨 등장 개수가 맞아야 한다.
- `pgm_output/*.pgm`은 실행 산출물이므로 제출에서 제외한다.
- `pgm_output/debug/*.png`는 보고서 증빙으로 쓸 경우에만 포함한다.
- 최종 보고서에는 샘플 영상 기준 정확도, 숫자별 정확도, 총 추론 시간, 평균 latency, 전체 wall time을 기록한다.

## 작성 메모

현재 작업환경에서는 `cv2`, `numpy`, `yaml`, `ultralytics`가 설치되어 있지 않아
`scripts/demo_preflight.sh`가 의존성 단계에서 중단된다. 최종 수치 입력 전,
실제 실험 환경에서 의존성을 설치하고
`VIDEO_PATH=videos/sample.mp4 PIPELINE_REPORT=reports/sample_video_report.txt ./run_pgm_all.sh`를 다시 실행해야 한다.
