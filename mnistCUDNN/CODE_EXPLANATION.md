# 코드 설명

이 문서는 새로 추가한 YOLO to mnistCUDNN 파이프라인 코드를 쉽게 이해하기
위한 설명입니다. 최종 시연은 카메라 입력을 기본값으로 사용하며, 필요하면
영상 파일이나 이미지 폴더로도 같은 흐름을 검증할 수 있습니다. 기존 MNIST 예제 코드와 `Makefile`은 직접 수정하지 않고,
바깥에서 실행 흐름을 감싸는 구조로 작성했습니다. `ocr.py`는 과제 명세에 맞게
`--video` 입력을 전체 파이프라인으로 넘길 수 있고, PGM 한 장을 기존 `mnistCUDNN`
실행 파일의 `image=<pgm>` 입력으로 넘기는 호환 wrapper 역할도 합니다.

## 전체 흐름

```text
./run_pgm_all.sh
  -> scripts/demo_preflight.sh
  -> scripts/run_pipeline.sh
    -> scripts/clean_outputs.sh
    -> scripts/run_yolo.sh
      -> yolo/detect_video.py
        -> yolo/dedupe.py
        -> yolo/save_pgm.py
        -> pgm_output/*.pgm 생성
    -> scripts/run_mnist_one.sh
    -> scripts/parse_mnist_output.py
    -> 정답/추론/시간/요약 출력
```

즉, 실행자는 `mnistCUDNN/`에서 `./run_pgm_all.sh`만 실행하면 됩니다.
동일한 영상 입력은 다음처럼 `ocr.py`로도 실행할 수 있습니다.

```bash
python3 ocr.py --video videos/input.mp4
```

## 핵심 아이디어

프로젝트 요구사항에서 중요한 조건은 다음 두 가지입니다.

1. YOLO가 숫자를 감지하면 해당 영역을 28x28 PGM 파일로 저장해야 합니다.
2. 같은 숫자가 여러 프레임에서 반복 검출되어도, 같은 등장 구간에서는 PGM을
   한 장만 저장해야 합니다.

이를 위해 코드는 YOLO 감지, 중복 제거, PGM 변환, MNIST 평가를 각각 분리했습니다.

```text
YOLO 감지 결과
  -> 같은 등장 구간인지 판단
  -> 새 구간이면 crop
  -> MNIST 스타일 28x28 PGM으로 변환
  -> 기존 MNIST 코드에 입력
```

## `run_pgm_all.sh`

최종 시연용 실행 파일입니다.

```bash
answers=()
source scripts/run_pipeline.sh
```

평가 시에는 상단의 `answers` 배열에 정답을 넣으면 됩니다.
예를 들어 최종 카메라 실험의 정답 순서가 `6 8 8 6`이면 다음처럼 둡니다.

```bash
answers=(6 8 8 6)
```

배열이 비어 있고 `answers.txt`가 있으면 로컬 실험 편의를 위해 공백으로 구분된
정답을 fallback으로 읽습니다.

`source scripts/run_pipeline.sh`를 사용하는 이유는 `answers` 배열을 하위
스크립트에서도 그대로 사용할 수 있게 하기 위해서입니다. 일반적인 실행 방식인
`bash scripts/run_pipeline.sh`를 사용하면 shell 배열이 자동으로 전달되지 않습니다.

## `scripts/run_pipeline.sh`

전체 파이프라인을 실제로 순서대로 실행하는 스크립트입니다.

주요 동작은 다음과 같습니다.

1. `answers` 배열이 있는지 확인합니다.
2. YOLO 단계가 켜져 있으면 기존 PGM 결과를 지웁니다.
3. `scripts/run_yolo.sh`를 실행해 새 PGM 파일을 생성합니다.
4. `pgm_output/*.pgm` 파일을 이름순으로 정렬합니다.
5. 정답 개수만큼 PGM 파일을 하나씩 MNIST에 입력합니다.
6. MNIST 출력에서 추론값과 시간을 파싱합니다.
7. 요구된 형식으로 `O/X`와 요약을 출력합니다.

`RUN_YOLO=0`을 주면 YOLO 단계를 건너뛰고 이미 만들어진 PGM 파일만 평가할 수
있습니다.

```bash
RUN_YOLO=0 ./run_pgm_all.sh
```

## `scripts/clean_outputs.sh`

`pgm_output/` 폴더 안의 기존 `.pgm` 파일을 지우는 스크립트입니다.

```text
이전 실행 결과 삭제
-> 새 영상에서 생성된 PGM만 평가
```

이 스크립트는 `pgm_output` 폴더 자체는 삭제하지 않고, 내부의 `.pgm` 파일만
삭제합니다.

## `scripts/run_yolo.sh`

Python YOLO 코드를 실행하는 shell wrapper입니다.

기본 설정은 `yolo/config.yaml`에서 읽습니다. 현재 기본 입력은 카메라 `0`입니다.
필요하면 환경 변수로 경로를
바꿀 수 있습니다.

```bash
CAMERA_INDEX=1 ./run_pgm_all.sh
YOLO_SOURCE=camera:1 ./run_pgm_all.sh
VIDEO_PATH=videos/my_video.mp4 ./run_pgm_all.sh
YOLO_WEIGHTS=yolo/weights/best.pt ./run_pgm_all.sh
PGM_OUTPUT_DIR=pgm_output ./run_pgm_all.sh
```

이 스크립트는 최종적으로 다음 Python 파일을 실행합니다.

```bash
python3 yolo/detect_video.py --config yolo/config.yaml
```

## `yolo/make_yolo_dataset.py`

숫자별 원본 이미지 폴더에서 YOLO 학습 데이터를 자동 생성하는 스크립트입니다.
`mnistCUDNN/`에서 실행하면 기본적으로 상위 폴더의 `6/`, `8/` 같은 숫자 폴더를
찾습니다. 기존 샘플 `one_28x28.pgm`, `three_28x28.pgm`, `five_28x28.pgm`도
각각 1, 3, 5 원본으로 사용할 수 있습니다.

```bash
python3 yolo/make_yolo_dataset.py
```

생성 결과는 `datasets/yolo/images/{train,val}`과
`datasets/yolo/labels/{train,val}`에 저장됩니다. bbox 라벨은 합성된 숫자의
위치에서 자동 계산됩니다. 데이터 생성 옵션은 `yolo/train_config.yaml`의
`dataset_generation`에서 바꿉니다.

## `yolo/train_yolo.py`

손글씨 숫자 검출용 YOLO 모델을 학습하는 스크립트입니다.

학습 데이터는 `yolo/dataset.yaml`이 가리키는 위치에 둡니다.

```text
datasets/yolo/images/train/
datasets/yolo/images/val/
datasets/yolo/labels/train/
datasets/yolo/labels/val/
```

라벨 파일은 이미지와 같은 이름의 `.txt` 파일이며, 각 줄은 YOLO 형식입니다.

```text
class_id x_center y_center width height
```

좌표는 이미지 크기로 나눈 0~1 정규화 값이어야 합니다.

기본 학습 명령은 다음과 같습니다.

```bash
python3 yolo/train_yolo.py
```

학습 결과 중 `best.pt`는 기본적으로 `yolo/weights/best.pt`로 복사됩니다. 따라서
학습 후 별도 경로 수정 없이 `./run_pgm_all.sh`에서 사용할 수 있습니다. 학습
옵션은 `yolo/train_config.yaml`의 `training`에서 바꿉니다.

## `yolo/config.yaml`

YOLO 감지와 PGM 저장에 필요한 설정 파일입니다.

```yaml
source: 0
weights: yolo/weights/best.pt
output_dir: pgm_output

confidence: 0.65
iou: 0.45
imgsz: 640
display: true
selection: center_conf

target_size: 28
digit_box_size: 20
bbox_padding: 0.30

min_stable_frames: 3
missing_frames_to_close_segment: 5
min_frames_between_saves: 12
save_debug_crops: true
```

중요한 값은 다음과 같습니다.

- `confidence`: YOLO 감지 신뢰도 기준입니다.
- `source`: 카메라 번호, `camera:1` 형식, 또는 영상 파일 경로를 지정합니다.
- `display`: 카메라 시연 중 OpenCV 창을 띄울지 정합니다.
- `selection`: 한 프레임에 여러 숫자가 잡혔을 때 어떤 bbox를 선택할지 정합니다.
- `target_size`: 저장할 PGM 크기입니다. MNIST 입력에 맞춰 28로 둡니다.
- `digit_box_size`: 28x28 안에서 실제 숫자 획이 차지할 목표 크기입니다.
- `bbox_padding`: YOLO bbox 주변을 얼마나 여유 있게 crop할지 정합니다.
- `min_stable_frames`: 숫자가 몇 프레임 연속 감지되어야 저장 후보가 되는지 정합니다.
- `missing_frames_to_close_segment`: 숫자가 몇 프레임 사라지면 이전 구간이 끝났다고 볼지 정합니다.
- `save_debug_crops`: 보고서 확인용 crop PNG와 28x28 PGM preview PNG를 저장합니다.

## `yolo/detect_video.py`

YOLO 단계의 중심 코드입니다.

이 파일은 다음 일을 합니다.

1. 설정 파일과 command line 인자를 읽습니다.
2. YOLO 가중치 `best.pt`를 로드합니다.
3. 카메라 또는 입력 동영상을 프레임 단위로 읽습니다.
4. 각 프레임에서 숫자를 감지합니다.
5. `SegmentTracker`로 저장 여부를 판단합니다.
6. 저장해야 하는 경우 bbox를 crop합니다.
7. crop 이미지를 28x28 PGM으로 변환해 `pgm_output/`에 저장합니다.
8. `save_debug_crops`가 켜져 있으면 crop과 PGM preview PNG를 `pgm_output/debug/`에 저장합니다.

저장 파일명은 다음 형태입니다.

```text
frame_000095_digit_6_conf_0.79_0000.pgm
```

파일명에 포함된 정보는 다음과 같습니다.

- `frame_000095`: 원본 영상의 프레임 번호
- `digit_6`: YOLO가 감지한 숫자 클래스
- `conf_0.79`: YOLO confidence
- `0000`: 저장된 segment 순번

## MNIST 시간 측정 구간

과제에서 요구한 대로 MNIST latency는 이미지 입력 복사와 변수 초기화가 끝난 뒤
시작하고, `softmaxForward()` 이후 `cudaDeviceSynchronize()`가 끝난 직후 종료합니다.
따라서 shell script가 출력하는 `Inference time`은 다음 연산 구간에 해당합니다.

```text
conv1 -> pool1 -> conv2 -> pool2 -> fc1 -> relu -> lrn -> fc2 -> softmax -> cudaDeviceSynchronize
```

파일 읽기, PGM 전처리, host-to-device 입력 복사, device-to-host 결과 복사는 profile
CSV에는 별도 항목으로 남길 수 있지만 정답 비교용 latency에는 포함하지 않습니다.

## 감지 결과 선택 방식

한 프레임에 여러 bbox가 검출될 수 있습니다. 예를 들어 칠판에 여러 숫자가 동시에
보이면 YOLO는 여러 숫자를 찾을 수 있습니다.

`selection: center_conf`는 confidence와 화면 중앙에 가까운 정도를 함께 봅니다.
카메라를 이동하며 촬영하는 시연에서는 보통 화면 중앙에 있는 숫자가 현재 읽을
대상이기 때문에 이 기본값이 적합합니다.

선택지는 다음과 같습니다.

- `confidence`: confidence가 가장 높은 bbox 선택
- `center`: 화면 중앙에 가장 가까운 bbox 선택
- `center_conf`: confidence와 중앙성을 함께 고려

## `yolo/dedupe.py`

중복 저장을 막는 코드입니다.

일반적인 YOLO는 영상에서 같은 숫자가 계속 보이는 동안 매 프레임마다 감지 결과를
냅니다. 그대로 저장하면 같은 숫자 하나가 수십 장의 PGM으로 저장됩니다.

이를 막기 위해 `SegmentTracker`가 다음 상태를 관리합니다.

```text
감지 없음
  -> 숫자 감지 시작
  -> 일정 프레임 이상 안정적으로 감지
  -> PGM 1장 저장
  -> 같은 구간에서 계속 감지되면 저장 안 함
  -> 숫자가 일정 프레임 이상 사라지면 구간 종료
  -> 다시 감지되면 새 구간으로 저장
```

예를 들어 영상이 아래 순서라면:

```text
6 (2초) -> 8 (3초) -> 8 (1.5초) -> 6 (2초)
```

숫자가 중간에 사라졌다가 다시 등장한다면 저장 결과는 다음처럼 됩니다.

```text
6, 8, 8, 6
```

즉, 같은 숫자라도 새 등장 구간이면 다시 저장됩니다.

## `yolo/save_pgm.py`

YOLO bbox crop을 MNIST 입력에 가까운 PGM으로 바꾸는 코드입니다.

처리 순서는 다음과 같습니다.

```text
YOLO bbox 기준 crop
  -> grayscale 변환
  -> 배경/글자 분리
  -> 가장 큰 글자 영역 선택
  -> 숫자 영역만 다시 crop
  -> 28x28 중앙에 배치
  -> 무게중심 기준으로 한 번 더 중앙 정렬
  -> PGM 저장
```

MNIST는 보통 검은 배경에 밝은 숫자 형태를 기대합니다. 그래서 전처리 결과도
그 형태에 가깝게 맞춥니다.

이 파일에서 중요한 함수는 다음과 같습니다.

- `crop_digit()`: YOLO bbox 주변을 padding 포함해서 잘라냅니다.
- `digit_to_mnist_pgm()`: crop 이미지를 MNIST 스타일 28x28 이미지로 변환합니다.
- `save_pgm()`: binary PGM 형식인 `P5`로 저장합니다.

## `ocr.py`

과제 명세의 동영상 진입점과 기존 MNIST PGM wrapper를 함께 지원합니다.

```bash
python3 ocr.py --video videos/input.mp4
python3 ocr.py --source camera:0
python3 ocr.py pgm_output/frame_000095_digit_6_conf_0.79_0000.pgm
```

`--video`, `--source`, `--camera-index`, `--image-path`, `--image-dir` 중 하나가
주어지면 `run_pgm_all.sh`와 같은 YOLO-to-MNIST 파이프라인을 실행합니다. PGM 파일
하나가 positional argument로 주어지면 기존처럼 `mnistCUDNN image=<pgm>`를
호출합니다.

## `scripts/run_mnist_one.sh`

PGM 파일 한 장을 기존 MNIST 코드에 넣는 wrapper입니다.

기존 프로젝트마다 MNIST 실행 방식이 다를 수 있으므로, 여러 방식을 순서대로
시도하게 했습니다.

```text
1. MNIST_COMMAND 환경 변수가 있으면 그 명령 실행
2. ocr.py가 있으면 python3 ocr.py <pgm> 실행
3. ./mnistCUDNN 바이너리가 있으면 ./mnistCUDNN image=<pgm> 실행
4. Makefile이 있으면 make 후 ./mnistCUDNN image=<pgm> 실행
```

기존 코드가 특정 고정 파일만 읽는 구조라면 `MNIST_INPUT_PATH`를 사용합니다.

```bash
MNIST_INPUT_PATH=data/input.pgm MNIST_COMMAND='./mnistCUDNN image={pgm}' ./run_pgm_all.sh
```

이 경우 각 PGM을 `data/input.pgm`으로 복사한 뒤 기존 MNIST 프로그램을 실행합니다.

## `scripts/parse_mnist_output.py`

MNIST 실행 결과에서 추론 숫자와 inference time을 뽑아내는 코드입니다.

예를 들어 MNIST 프로그램이 아래처럼 출력하면:

```text
추론: 6
Inference time: 439.939 ms
```

이 스크립트는 다음 값을 추출합니다.

```text
6 439.939
```

이 값은 `scripts/run_pipeline.sh`에서 정답과 비교하고 총 시간을 더하는 데
사용됩니다.

## 정확도 향상을 위해 조정할 부분

기존 MNIST 함수 호출 순서를 바꿀 수 없기 때문에, 정확도는 주로 YOLO 이후 PGM
품질에서 결정됩니다.

우선 조정할 값은 다음입니다.

- 숫자가 너무 작게 저장되면 `digit_box_size`를 키웁니다.
- 숫자가 잘리면 `bbox_padding`을 키웁니다.
- 중복 PGM이 너무 많이 생기면 `missing_frames_to_close_segment`를 키웁니다.
- 필요한 숫자가 저장되지 않으면 `confidence`를 낮춥니다.
- 잘못된 bbox가 선택되면 `selection`을 `confidence` 또는 `center`로 바꿉니다.

## 제출할 때 주의할 점

제출에는 코드, shell script, 설정 파일, 학습된 가중치, 보고서가 포함되어야 합니다.

포함해야 하는 것:

- `run_pgm_all.sh`
- `scripts/`
- `yolo/`
- `yolo/weights/best.pt`
- `data/*.bin`
- `data/trained_bins_aug68.zip`
- `run_pgm_all.sh`의 `answers=(...)` 정답 배열 또는 보고서에 기록한 평가 정답 순서
- `requirements-yolo.txt`
- 기존 MNIST 코드와 `Makefile`

포함하지 않는 것:

- `pgm_output/*.pgm`
- `__pycache__/`
- 임시 crop 이미지. 단, `pgm_output/debug/*.png`는 보고서 증빙으로 쓸 때만 포함할 수 있습니다.
- 실행 중 생성된 로그 파일

## 빠른 점검 명령

문법만 확인할 때:

```bash
bash -n run_pgm_all.sh scripts/*.sh
python3 -m py_compile yolo/*.py scripts/parse_mnist_output.py
```

YOLO 없이 shell 흐름만 확인할 때:

```bash
RUN_YOLO=0 ./run_pgm_all.sh
```

실제 시연 실행:

```bash
./run_pgm_all.sh
```
