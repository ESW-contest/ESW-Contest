# YOLOv8 Segmentation 실시간 웹캠 추론
## Real-time Object Segmentation with Webcam

이 프로젝트는 **YOLOv8 세그멘테이션 모델**을 ONNX 형식으로 로드하여 **웹캠 실시간 추론**을 수행하는 Python 스크립트입니다. 객체의 **윤곽선만**을 그려서 깔끔한 시각화를 제공합니다.

## 📋 목차

- [주요 특징](#주요-특징)
- [기술 스택](#기술-스택)
- [시스템 요구사항](#시스템-요구사항)
- [설치 및 실행](#설치-및-실행)
- [하이퍼파라미터 설정](#하이퍼파라미터-설정)
- [실행 방법](#실행-방법)
- [결과 해석](#결과-해석)
- [고급 기능](#고급-기능)
- [문제 해결](#문제-해결)
- [성능 최적화](#성능-최적화)

## 🎯 주요 특징

- **🔥 실시간 추론**: 웹캠을 통한 실시간 객체 세그멘테이션
- **🎨 깔끔한 시각화**: 윤곽선만 표시 (박스/라벨/채우기 없음)
- **🛡️ 노이즈 억제**: 모폴로지 연산으로 노이즈 제거
- **🌊 프레임간 스무딩**: EMA(Exponential Moving Average)로 부드러운 전환
- **✂️ 윤곽선 단순화**: Douglas-Peucker 알고리즘으로 최적화
- **⚡ 고성능**: ONNX 런타임으로 빠른 추론 속도

## 🛠️ 기술 스택

| 컴포넌트 | 기술 | 버전 |
|----------|------|------|
| **AI 모델** | YOLOv8 Segmentation | - |
| **모델 포맷** | ONNX | - |
| **런타임** | ONNX Runtime | - |
| **컴퓨터 비전** | OpenCV | 4.x |
| **수치 계산** | NumPy | 1.x |
| **실시간 처리** | Webcam | - |

## 💻 시스템 요구사항

### 필수 요구사항
- **Python**: 3.7 이상
- **웹캠**: USB 웹캠 또는 내장 카메라
- **RAM**: 최소 4GB
- **저장공간**: 500MB 이상

### 권장 사양
- **CPU**: Intel i5 이상 또는 AMD Ryzen 5 이상
- **RAM**: 8GB 이상
- **GPU**: NVIDIA GTX 1050 이상 (선택사항)

## 📦 설치 및 실행

### 1. 패키지 설치
```bash
pip install opencv-python numpy onnxruntime
```

### 2. ONNX 모델 준비
프로젝트 폴더에 `last.onnx` 파일을 준비하세요:
```
PC_Test/
├── test.py          # 메인 스크립트
├── last.onnx        # YOLOv8 세그멘테이션 모델
└── README.md        # 이 파일
```

### 3. 실행
```bash
python test.py
```

## ⚙️ 하이퍼파라미터 설정

스크립트 상단에서 다양한 파라미터를 조정할 수 있습니다:

### 기본 설정
```python
# ---------- 하이퍼파라미터 ----------
CONF_TH = 0.62        # 신뢰도 임계값 (0.0~1.0)
IOU_TH  = 0.5         # IoU 임계값 (0.0~1.0)
MASK_TH = 0.6         # 마스크 임계값 (0.0~1.0)
```

### 시각화 설정
```python
OUTLINE_THICK = 4           # 윤곽선 두께 (픽셀)
OUTLINE_COLOR = (255, 255, 0)  # 윤곽선 색상 (BGR)
```

### 노이즈 처리 설정
```python
MIN_AREA_RATIO = 0.0001     # 최소 객체 크기 비율
K_OPEN  = 1                 # Opening 커널 크기
K_CLOSE = 3                 # Closing 커널 크기
GAUSS_K = 1                 # Gaussian 블러 커널 크기
```

### 스무딩 설정
```python
EMA_ALPHA = 0.6             # EMA 알파 값 (0.0~1.0)
                              # 낮을수록 반응 빠름, 높을수록 안정적
```

## 🚀 실행 방법

### 기본 실행
```bash
# 현재 디렉토리에서 실행
python test.py
```

### 고급 실행 옵션
```python
# 스크립트 내에서 파라미터 수정 후 실행
CONF_TH = 0.8        # 더 엄격한 신뢰도
EMA_ALPHA = 0.8      # 더 부드러운 스무딩
OUTLINE_THICK = 2    # 얇은 윤곽선
```

### 실행 중 제어
- **실행 중**: 실시간으로 객체 윤곽선 표시
- **종료**: `Q` 키를 눌러 프로그램 종료
- **윈도우**: "Segmentation Outline (Q to quit)" 창 표시

## 📊 결과 해석

### 출력 화면
실행 시 다음과 같은 창이 표시됩니다:
```
┌─────────────────────────────────────┐
│ Segmentation Outline (Q to quit)    │
│                                     │
│  ┌─────────────────────────────┐    │
│  │         Webcam View         │    │
│  │    ┌─────────────────┐      │    │
│  │    │  Object Outline │      │    │
│  │    │  ─────────────  │      │    │
│  │    │  │██████████│   │      │    │
│  │    │  │██████████│   │      │    │
│  │    │  └────────────  │      │    │
│  │    └─────────────────┘      │    │
│  └─────────────────────────────┘    │
└─────────────────────────────────────┘
```

### 윤곽선 의미
- **노란색 선**: 감지된 객체의 윤곽선
- **실시간 업데이트**: 매 프레임마다 새로운 윤곽선 계산
- **스무딩 적용**: 프레임간 부드러운 전환

## 🔧 고급 기능

### 1. 노이즈 억제
```python
# 모폴로지 연산으로 노이즈 제거
if K_OPEN > 1:
    m_bin = cv2.morphologyEx(m_bin, cv2.MORPH_OPEN, kernel)
if K_CLOSE > 1:
    m_bin = cv2.morphologyEx(m_bin, cv2.MORPH_CLOSE, kernel)
```

### 2. 프레임간 스무딩
```python
# EMA(Exponential Moving Average) 적용
if prev_scene_prob is None:
    smooth_prob = scene_prob
else:
    smooth_prob = EMA_ALPHA * prev_scene_prob + (1.0 - EMA_ALPHA) * scene_prob
```

### 3. 윤곽선 단순화
```python
# Douglas-Peucker 알고리즘으로 최적화
eps = 0.003 * cv2.arcLength(contour, True)
approx = cv2.approxPolyDP(contour, eps, True)
```

## 🔍 문제 해결

### 일반적인 문제들

#### 1. 웹캠 인식 실패
```python
# 해결: 다른 카메라 인덱스 시도
cap = cv2.VideoCapture(1)  # 0 대신 1, 2, ...
```

#### 2. ONNX 모델 로드 실패
```python
# 해결: 모델 파일 존재 확인
import os
print(os.path.exists("last.onnx"))  # True여야 함
```

#### 3. 메모리 부족
```python
# 해결: 입력 크기 조정
in_h, in_w = 320, 320  # 640x640 대신 320x320
```

#### 4. 추론 속도 느림
```python
# 해결: CONF_TH 높이기
CONF_TH = 0.8  # 0.62 대신 0.8로 더 엄격하게
```

### 디버깅 팁
```python
# 모델 정보 확인
print(f"Input shape: {session.get_inputs()[0].shape}")
print(f"Output shapes: {[out.shape for out in session.get_outputs()]}")

# 프레임 정보 확인
print(f"Frame shape: {frame.shape}")
print(f"FPS: {cap.get(cv2.CAP_PROP_FPS)}")
```

## ⚡ 성능 최적화

### CPU 최적화
```python
# ONNX 세션 최적화
session = ort.InferenceSession(
    "last.onnx",
    providers=["CPUExecutionProvider"],
    sess_options=ort.SessionOptions()
)
```

### 메모리 최적화
```python
# 불필요한 변수 정리
del preds, protos
import gc
gc.collect()
```

### 속도 최적화
```python
# 배치 처리 (가능한 경우)
# 현재는 실시간이므로 단일 프레임 처리
```

## 📈 성능 메트릭

### 예상 성능 (CPU 기준)
- **추론 속도**: 10-30 FPS (해상도에 따라)
- **메모리 사용**: 200-500MB
- **CPU 사용률**: 30-70%

### 해상도별 성능
| 해상도 | FPS | 메모리 | 정확도 |
|--------|-----|--------|--------|
| 320x320 | 25-30 | 200MB | 보통 |
| 416x416 | 15-20 | 300MB | 좋음 |
| 640x640 | 8-12 | 500MB | 우수 |

## 🎨 커스터마이징

### 색상 변경
```python
# 윤곽선 색상 변경
OUTLINE_COLOR = (0, 255, 0)    # 녹색
OUTLINE_COLOR = (255, 0, 0)    # 파란색
OUTLINE_COLOR = (0, 0, 255)    # 빨간색
```

### 두께 조정
```python
# 윤곽선 두께 변경
OUTLINE_THICK = 1    # 얇게
OUTLINE_THICK = 6    # 두껍게
```

### 민감도 조정
```python
# 더 민감하게 (작은 객체도 감지)
CONF_TH = 0.3
MASK_TH = 0.4
MIN_AREA_RATIO = 0.00005

# 더 엄격하게 (큰 객체만 감지)
CONF_TH = 0.8
MASK_TH = 0.8
MIN_AREA_RATIO = 0.001
```

## 🔗 관련 프로젝트

- **학습용**: `learn_Detection` - YOLOv8 모델 학습
- **적외선**: `Thermel-Detect` - IR 기반 객체 감지
- **음성**: `LLM-kit` - 음성 처리 도구

## 📝 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

## 🤝 기여

기능 개선, 버그 수정, 문서화 개선을 환영합니다!

---

## 🎯 빠른 시작 체크리스트

- [ ] Python 3.7+ 설치 확인
- [ ] 웹캠 작동 확인
- [ ] `last.onnx` 모델 파일 준비
- [ ] 패키지 설치: `pip install opencv-python numpy onnxruntime`
- [ ] 실행: `python test.py`
- [ ] `Q` 키로 종료

**즐거운 실시간 세그멘테이션 경험 되세요!** 🚀📹
