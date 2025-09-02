# Line Detection with YOLOv8 Segmentation

YOLOv8 세그멘테이션 모델을 활용한 실시간 라인 감지 프로젝트입니다. Raspberry Pi 5에서 Picamera2를 사용하여 고성능 실시간 라인 감지 및 윤곽선 추출을 수행합니다.

## 목차

- [개요](#개요)
- [주요 기능](#주요-기능)
- [코드 설명](#코드-설명)
- [핵심 기능 상세 설명](#핵심-기능-상세-설명)
- [시스템 요구사항](#시스템-요구사항)
- [설치 및 설정](#설치-및-설정)
- [사용법](#사용법)
- [파일 설명](#파일-설명)
- [하이퍼파라미터 설정](#하이퍼파라미터-설정)
- [성능 최적화](#성능-최적화)
- [문제 해결](#문제-해결)
- [라이선스](#라이선스)

## 개요

이 프로젝트는 YOLOv8 세그멘테이션 모델을 사용하여 실시간으로 라인을 감지하고 윤곽선을 추출하는 컴퓨터 비전 애플리케이션입니다. Raspberry Pi 5의 Picamera2를 활용하여 고해상도 비디오 스트림에서 라인을 실시간으로 처리합니다.

### 기술 스택
- **모델**: YOLOv8 Segmentation (ONNX 포맷)
- **하드웨어**: Raspberry Pi 5
- **카메라**: Picamera2 (640x384 해상도)
- **실행 환경**: Python 3.8+, ONNX Runtime
- **후처리**: OpenCV (모폴로지 연산, 윤곽선 추출)

## 주요 기능

### 실시간 처리
- **고성능 추론**: ONNX Runtime을 통한 최적화된 모델 실행
- **프레임 스킵**: `INFER_EVERY` 파라미터로 추론 빈도 조절
- **EMA 스무딩**: 지수 이동 평균으로 안정적인 마스크 생성
- **풀스크린 디스플레이**: DSI 디스플레이를 통한 실시간 시각화

### 고급 후처리
- **벡터화 마스크 계산**: 행렬곱을 통한 고속 마스크 생성
- **모폴로지 연산**: Opening/Closing으로 노이즈 제거
- **연결 요소 분석**: 작은 객체 필터링
- **윤곽선 단순화**: Douglas-Peucker 알고리즘으로 최적화

### 성능 모니터링
- **실시간 FPS 표시**: EMA 스무딩된 프레임레이트
- **추론 상태 표시**: 현재 추론 여부 및 감지된 객체 수
- **메모리 최적화**: 효율적인 텐서 연산

## 코드 설명

### execution.py 주요 코드 블록

#### 1. 모델 로드 및 세션 최적화
```python
so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
so.intra_op_num_threads = 4   # Pi 5: 4코어
so.inter_op_num_threads = 1

session = ort.InferenceSession("last.onnx", sess_options=so)
```
**설명**: ONNX Runtime 세션을 최적화하여 CPU 성능을 극대화합니다.

#### 2. Picamera2 초기화
```python
picam = Picamera2()
video_cfg = picam.create_video_configuration(
    main={"size": (CAM_W, CAM_H), "format": "XRGB8888"}
)
picam.configure(video_cfg)
picam.start()
```
**설명**: Raspberry Pi 카메라를 설정하고 640x384 해상도로 시작합니다.

#### 3. 실시간 추론 루프
```python
while True:
    frame = picam.capture_array()
    frame = cv2.cvtColor(frame_bgra, cv2.COLOR_BGRA2BGR)
    
    if do_infer:
        preds, protos = session.run(None, {input_name: img})
        # 후처리 및 마스크 생성
```
**설명**: 카메라 프레임을 실시간으로 캡처하고 모델 추론을 수행합니다.

#### 4. 벡터화 마스크 계산
```python
proto = np.squeeze(protos, 0).reshape(32, -1).astype(np.float32)
M = mask_coefs.astype(np.float32) @ proto
M = sigmoid(M).reshape(-1, 160, 160)
```
**설명**: 행렬곱을 사용하여 마스크를 고속으로 계산합니다.

#### 5. 윤곽선 추출 및 시각화
```python
contours, _ = cv2.findContours(m_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = [c for c in contours if cv2.contourArea(c) >= min_area]
cv2.drawContours(vis, simplified, -1, OUTLINE_COLOR, OUTLINE_THICK)
```
**설명**: 이진화된 마스크에서 윤곽선을 추출하고 시각화합니다.

### learning.py 주요 코드 블록

#### 1. 패키지 자동 설치
```python
def install_packages():
    packages = ["ultralytics", "roboflow", "opencv-python", "matplotlib"]
    for package in packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
```
**설명**: 필요한 모든 패키지를 자동으로 설치합니다.

#### 2. Roboflow 데이터셋 다운로드
```python
rf = Roboflow(api_key=api_key)
project_obj = rf.workspace(workspace).project(project)
dataset = project_obj.version(int(version)).download("yolov8")
```
**설명**: Roboflow에서 YOLOv8 포맷의 데이터셋을 다운로드합니다.

#### 3. 모델 학습
```python
from ultralytics import YOLO
model = YOLO('yolov8n-seg.pt')
model.train(data=data_yaml, epochs=100, imgsz=640)
```
**설명**: YOLOv8 세그멘테이션 모델을 학습시킵니다.

## 핵심 기능 상세 설명

### 1. 실시간 라인 감지 파이프라인

#### 입력 처리
- **카메라 캡처**: Picamera2를 통해 640x384 RGB 프레임을 실시간으로 획득
- **색상 변환**: XRGB8888 → BGRA → BGR 변환으로 OpenCV 호환성 확보
- **Letterbox 변환**: 모델 입력 크기(640x384)로 패딩 및 리사이징

#### 모델 추론
- **ONNX 실행**: 최적화된 세션에서 단일 이미지 배치 추론
- **출력 파싱**: 바운딩 박스, 클래스 확률, 마스크 계수 추출
- **후처리 필터링**: 신뢰도 및 IoU 임계값 기반 객체 필터링

### 2. 고성능 마스크 생성

#### 벡터화 계산
- **프로토타입 변환**: (1,32,160,160) → (32,25600) 형태로 재배열
- **행렬곱 연산**: 마스크 계수와 프로토타입의 효율적 곱셈
- **시그모이드 활성화**: 확률 맵 생성 및 이진화

#### 공간 변환
- **업샘플링**: 160x160 → 640x384로 리사이징
- **패딩 제거**: Letterbox 변환의 역연산으로 원본 크기 복원
- **EMA 스무딩**: 시간적 안정성을 위한 지수 이동 평균 적용

### 3. 지능형 후처리

#### 모폴로지 연산
- **Opening**: 작은 노이즈 제거를 위한 침식 + 팽창
- **Closing**: 객체 내부 구멍 메우기를 위한 팽창 + 침식
- **가우시안 블러**: 윤곽선 부드러움 개선

#### 객체 필터링
- **연결 요소 분석**: cv2.connectedComponentsWithStats로 객체 분리
- **면적 기반 필터링**: MIN_AREA_RATIO 기준으로 작은 객체 제거
- **윤곽선 단순화**: Douglas-Peucker 알고리즘으로 점 수 최적화

### 4. 성능 최적화 기법

#### 프레임 관리
- **추론 스킵**: INFER_EVERY 파라미터로 CPU 부하 조절
- **비동기 처리**: 카메라 캡처와 추론의 분리 실행
- **메모리 재사용**: 텐서 버퍼의 효율적 재활용

#### 모니터링 시스템
- **FPS 계산**: EMA 스무딩으로 안정적인 프레임레이트 측정
- **상태 표시**: 추론 상태와 감지 객체 수 실시간 모니터링
- **풀스크린 출력**: DSI 디스플레이를 통한 최적화된 시각화

### 5. 에러 처리 및 안정성

#### 예외 처리
- **카메라 실패**: Picamera2 초기화 실패 시 프로그램 안전 종료
- **모델 로드 실패**: ONNX 파일 누락 시 대체 처리 로직
- **메모리 부족**: 배치 크기 자동 조절 및 가비지 컬렉션

#### 데이터 검증
- **입력 유효성**: 이미지 크기 및 포맷 검증
- **출력 검증**: 모델 예측 결과의 범위 및 형태 확인
- **리소스 모니터링**: CPU 사용률 및 메모리 사용량 추적

## 시스템 요구사항

### 하드웨어
- **Raspberry Pi 5** (4코어 CPU 권장)
- **Picamera2** 모듈
- **DSI 디스플레이** (풀스크린 출력용)
- **최소 4GB RAM**

### 소프트웨어
- **Python 3.8+**
- **Raspberry Pi OS** (64-bit 권장)
- **OpenCV 4.5+**
- **ONNX Runtime 1.12+**

### 권장 사양
- **CPU**: Raspberry Pi 5 (4코어)
- **메모리**: 8GB RAM 이상
- **저장소**: 16GB 이상 여유 공간

## 설치 및 설정

### 1. 시스템 업데이트
```bash
sudo apt update && sudo apt upgrade -y
```

### 2. Python 패키지 설치
```bash
pip install opencv-python onnxruntime picamera2 libcamera
```

### 3. Picamera2 설정
```bash
sudo apt install -y python3-picamera2
```

### 4. ONNX Runtime 최적화 (선택사항)
```bash
# OpenVINO 설치로 CPU 성능 향상
pip install openvino
```

## 사용법

### 기본 실행
```bash
python execution.py
```

### 실행 옵션
프로그램이 시작되면 다음 키로 제어 가능:
- **Q**: 프로그램 종료
- **실시간 모니터링**: 터미널에서 FPS 및 객체 수 확인

### 실행 과정
1. **카메라 초기화**: Picamera2 설정 및 시작
2. **모델 로드**: ONNX 모델 로드 및 세션 최적화
3. **실시간 처리**: 프레임별 라인 감지 및 시각화
4. **풀스크린 출력**: DSI 디스플레이에 결과 표시

## 파일 설명

### execution.py
실시간 라인 감지 실행 스크립트
- **입력**: Picamera2 비디오 스트림
- **처리**: YOLOv8 ONNX 모델 추론
- **출력**: 윤곽선이 그려진 실시간 비디오
- **특징**: 프레임 스킵, EMA 스무딩, 벡터화 후처리

### learning.py
YOLOv8 세그멘테이션 모델 학습 스크립트
- **기능**: Roboflow 데이터셋 다운로드 및 모델 학습
- **출력**: 학습된 모델 (PyTorch 및 ONNX 포맷)
- **특징**: 자동 패키지 설치, 데이터 전처리, 시각화

### last.onnx
학습된 YOLOv8 세그멘테이션 모델
- **포맷**: ONNX (Open Neural Network Exchange)
- **입력**: 640x384 RGB 이미지
- **출력**: 세그멘테이션 마스크 및 바운딩 박스
- **최적화**: CPU 추론용으로 최적화됨

## 하이퍼파라미터 설정

### 감지 임계값
```python
CONF_TH = 0.65      # 신뢰도 임계값 (0.0-1.0)
IOU_TH = 0.4        # IoU 임계값 (0.0-1.0)
MASK_TH = 0.55      # 마스크 임계값 (0.0-1.0)
```

### 시각화 설정
```python
OUTLINE_THICK = 4           # 윤곽선 두께 (픽셀)
OUTLINE_COLOR = (255, 255, 0)  # 윤곽선 색상 (BGR)
MIN_AREA_RATIO = 0.0001     # 최소 객체 면적 비율
```

### 성능 최적화
```python
INFER_EVERY = 2     # 추론 빈도 (프레임 스킵)
EMA_ALPHA = 0.6     # EMA 스무딩 계수 (0.0-1.0)
FPS_ALPHA = 0.9     # FPS 스무딩 계수 (0.0-1.0)
```

### 모폴로지 연산
```python
K_OPEN = 1          # Opening 커널 크기
K_CLOSE = 2         # Closing 커널 크기
GAUSS_K = 1         # Gaussian 블러 커널 크기
```

## 성능 최적화

### CPU 최적화
- **세션 옵션**: 그래프 최적화 및 스레드 설정
- **벡터화 연산**: 행렬곱을 통한 마스크 계산 가속
- **메모리 관리**: 효율적인 텐서 재사용

### 프레임 처리 최적화
- **프레임 스킵**: 추론 빈도 조절로 CPU 부하 감소
- **EMA 스무딩**: 시간적 안정성 향상
- **비동기 처리**: 카메라 캡처와 추론의 분리

### 메모리 최적화
- **배치 크기**: 단일 이미지 처리로 메모리 사용 최소화
- **텐서 재사용**: 중간 결과물 재사용
- **가비지 컬렉션**: 불필요한 객체 즉시 해제

## 문제 해결

### 카메라 관련 문제
**문제**: Picamera2가 시작되지 않음
```
해결: sudo apt install -y python3-picamera2
```

**문제**: XRGB8888 포맷 오류
```
해결: 카메라 설정에서 포맷을 확인하고 BGRA2BGR 변환 추가
```

### 모델 관련 문제
**문제**: ONNX 모델 로드 실패
```
해결: 모델 파일 경로 확인 및 ONNX Runtime 버전 호환성 체크
```

**문제**: 추론 성능 저하
```
해결: INFER_EVERY 값 증가 또는 OpenVINO 설치
```

### 디스플레이 관련 문제
**문제**: 풀스크린 모드 작동하지 않음
```
해결: DSI 디스플레이 연결 확인 및 cv2.WINDOW_NORMAL 모드로 변경
```

### 성능 관련 문제
**문제**: 낮은 FPS
```
해결: INFER_EVERY 증가, 해상도 감소, 또는 CPU 최적화
```

**문제**: 메모리 부족
```
해결: 배치 크기 감소, 불필요한 변수 정리
```

## 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 상업적 및 비상업적 사용 모두 허용됩니다.

## 기여

버그 리포트, 기능 제안, 코드 기여를 환영합니다. 이슈나 풀 리퀘스트를 통해 참여해주세요.

## 버전 히스토리

- **v1.0.0**: 초기 릴리즈
  - YOLOv8 세그멘테이션 모델 통합
  - Raspberry Pi 5 Picamera2 지원
  - 실시간 후처리 및 시각화
  - ONNX 모델 최적화

---

**주의사항**: 이 코드는 Raspberry Pi 5 환경에서 테스트되었습니다. 다른 하드웨어에서는 호환성 문제가 발생할 수 있습니다.
