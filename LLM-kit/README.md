# LLM-Kit: Voice Activity Detection & Direction of Arrival

CRNN 기반 음성 활동 감지(VAD)와 방향 추정(DOA) 시스템입니다. Raspberry Pi에서 실시간으로 음성 신호를 처리하고 음원의 방향을 추정하여 UDP로 전송합니다.

## 목차

- [개요](#개요)
- [주요 기능](#주요-기능)
- [시스템 요구사항](#시스템-요구사항)
- [설치 및 설정](#설치-및-설정)
- [사용법](#사용법)
- [파일 설명](#파일-설명)
- [하이퍼파라미터 설정](#하이퍼파라미터-설정)
- [알고리즘 설명](#알고리즘-설명)
- [문제 해결](#문제-해결)
- [라이선스](#라이선스)

## 개요

이 프로젝트는 음성 신호 처리와 공간 음향을 결합한 실시간 시스템입니다. CRNN(Convolutional Recurrent Neural Network) 기반 VAD 모델로 음성 활동을 감지하고, GCC-PHAT 알고리즘으로 음원의 방향을 추정합니다.

### 기술 스택
- **VAD 모델**: CRNN (Conv2D → GRU → FC)
- **DOA 알고리즘**: GCC-PHAT (Generalized Cross-Correlation with Phase Transform)
- **실시간 처리**: WebRTC VAD + NumPy 기반 신호 처리
- **하드웨어**: Raspberry Pi + 마이크로폰 어레이
- **통신**: UDP 소켓 통신
- **시각화**: Pygame + Picamera2 오버레이

## 주요 기능

### 음성 활동 감지 (VAD)
- **CRNN 모델**: log-mel 스펙트로그램 기반 음성 분류
- **WebRTC VAD**: 실시간 음성/비음성 구분
- **피치 게이트**: 음성 품질 향상을 위한 피치 기반 필터링
- **RMS 임계값**: 동적 노이즈 바닥 적응

### 방향 추정 (DOA)
- **GCC-PHAT 알고리즘**: 상호상관 기반 방향 추정
- **2채널 마이크로폰**: 스테레오 입력으로 TDOA 계산
- **실시간 보정**: yaw 오프셋 및 각도 제한
- **신뢰도 평가**: 피크 비율 및 돌출도 기반 검증

### 실시간 처리
- **스트리밍 아키텍처**: 20ms 프레임 기반 실시간 처리
- **오버랩 처리**: 50% 오버랩으로 연속성 확보
- **UDP 전송**: 방향 결과 실시간 전송
- **시각화 오버레이**: 카메라 미리보기에 방향 표시

### 고급 필터링
- **밴드패스 필터**: 300-3400Hz 음성 대역 필터링
- **스무딩**: EMA(Exponential Moving Average) 적용
- **미디언 필터**: 튀는 값 제거를 위한 미디언 필터링
- **각도 제한**: 프레임당 최대 각도 변화 제한

## 시스템 요구사항

### 하드웨어
- **Raspberry Pi 4/5**: 4코어 CPU 권장
- **마이크로폰 어레이**: 2채널 스테레오 마이크 (8cm 간격)
- **카메라**: Raspberry Pi 카메라 모듈 (선택사항, 시각화용)
- **최소 2GB RAM**

### 소프트웨어
- **Python 3.8+**
- **Raspberry Pi OS** (64-bit 권장)
- **PyTorch 1.7+** (학습용)
- **ONNX Runtime** (추론용)
- **SoundDevice** (오디오 처리)
- **WebRTC VAD** (음성 감지)
- **Pygame** (시각화)

### 오디오 설정
- **샘플링 레이트**: 16kHz
- **채널**: 2채널 (스테레오)
- **프레임 크기**: 20ms (320 샘플)
- **버퍼 크기**: 960 샘플 (50% 오버랩)

## 설치 및 설정

### 1. 시스템 패키지 설치
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3-pip python3-dev portaudio19-dev
```

### 2. Python 패키지 설치
```bash
pip install torch torchvision torchaudio sounddevice webrtcvad pygame picamera2 onnxruntime
```

### 3. 오디오 설정
```bash
# ALSA 설정 확인
aplay -l
arecord -l

# 필요한 경우 권한 설정
sudo usermod -a -G audio $USER
```

### 4. 마이크로폰 어레이 설정
마이크로폰 간 거리를 정확히 측정하여 코드에 반영:
```python
MIC_DISTANCE_M = 0.08  # 실제 마이크 간 거리 (미터)
```

## 사용법

### VAD + DOA 실행 (LLM_running.py)
```bash
python3 LLM_running.py \
  --dev "Axera" \
  --ip 192.168.100.1 \
  --port 5005 \
  --vad_mode 2 \
  --ratio 0.6
```

### 파라미터 설명
- `--dev`: 입력 장치 이름 힌트 (예: "Axera")
- `--ip`: UDP 전송 대상 IP
- `--port`: UDP 전송 포트
- `--vad_mode`: VAD 민감도 (0-3, 높을수록 엄격)
- `--ratio`: 음성 프레임 비율 임계값

### 시각화 실행 (rasberrypi_running.py)
```bash
python3 rasberrypi_running.py
```

### 키보드 제어 (실시간 튜닝)
- **←/→**: YAW 오프셋 조정 (±1°)
- **Shift + ←/→**: YAW 오프셋 조정 (±5°)
- **↑/↓**: TH_LOW 임계값 조정
- **Shift + ↑/↓**: TH_HIGH 임계값 조정
- **H**: 도움말 토글
- **ESC**: 프로그램 종료

### 모델 학습 (learning.py)
```bash
python3 learning.py
```

## 파일 설명

### LLM_running.py
VAD와 DOA를 결합한 실시간 음성 처리 시스템
- **WebRTC VAD**: 실시간 음성 활동 감지
- **GCC-PHAT DOA**: 상호상관 기반 방향 추정
- **UDP 전송**: 방향 결과 실시간 전송
- **특징**: 피치 게이트, 신뢰도 평가, 스무딩

### rasberrypi_running.py
카메라 미리보기에 DOA 결과를 오버레이하는 시각화 시스템
- **Picamera2 통합**: Raspberry Pi 카메라 미리보기
- **UDP 수신**: DOA 결과 수신 및 파싱
- **화살표 시각화**: 3단계 존으로 방향 표시
- **특징**: 실시간 파라미터 튜닝, SDL 드라이버 자동 선택

### learning.py
CRNN VAD 모델 학습 및 데이터 준비 스크립트
- **데이터 준비**: Google Speech Commands + MUSAN 노이즈
- **특징 추출**: log-mel 스펙트로그램 변환
- **모델 학습**: CRNN 아키텍처 기반 VAD 모델
- **특징**: GPU/AMP 최적화, ONNX/TorchScript 내보내기

### vad_crnn.onnx
학습된 CRNN 기반 VAD 모델
- **입력**: log-mel 스펙트로그램 [T, F] (T: 시간, F: 주파수)
- **출력**: 각 프레임의 음성 확률 [T]
- **최적화**: ONNX Runtime용 최적화
- **용도**: 실시간 음성 활동 감지

### running_command.txt
실시간 실행을 위한 명령어 템플릿
- **VAD 모델 설정**: CRNN ONNX 모델 경로
- **마이크로폰 설정**: 마이크 어레이 구성 파일
- **UDP 설정**: 전송 IP와 포트
- **파라미터 튜닝**: 신뢰도, 각도 제한, 스무딩 설정

## 하이퍼파라미터 설정

### VAD 설정
```python
VAD_FRAME = 320          # 20ms 프레임 (16kHz)
VAD_MODE = 2             # WebRTC VAD 민감도 (0-3)
VAD_RATIO_THRESH = 0.6   # 음성 프레임 비율 임계값
RMS_THRESH_STATIC = 130.0 # RMS 임계값
```

### DOA 설정
```python
MIC_DISTANCE_M = 0.08    # 마이크 간 거리 (미터)
SPEED_SOUND = 343.0      # 음속 (m/s)
GCC_INTERP = 8           # GCC 해상도
MAX_TAU = MIC_DISTANCE_M / SPEED_SOUND  # 최대 시간 지연
```

### 신뢰도 평가
```python
GCC_PEAK_RATIO_MIN = 1.10  # 1등/2등 피크 비율
GCC_PROMINENCE_MIN = 3.50  # 피크 돌출도
```

### 필터링 및 스무딩
```python
LOWCUT = 300.0; HIGHCUT = 3400.0  # 밴드패스 필터
EMA_ALPHA = 0.25                   # EMA 스무딩 계수
MEDIAN_WIN = 7                     # 미디언 필터 윈도우
MAX_DEG_PER_FRAME = 8.0           # 프레임당 최대 각도 변화
```

### 피치 게이트 (선택사항)
```python
PITCH_MIN = 85.0; PITCH_MAX = 300.0  # 피치 범위 (Hz)
PITCH_STRONG_THRESH = 0.35           # 피치 강도 임계값
```

## 알고리즘 설명

### 1. 음성 활동 감지 (VAD)

#### WebRTC VAD
- **입력**: 20ms 오디오 프레임 (320 샘플 @ 16kHz)
- **처리**: 음성/비음성 2진 분류
- **특징**: 낮은 계산 비용, 실시간 성능

#### CRNN VAD
- **입력**: log-mel 스펙트로그램 [T, 40]
- **아키텍처**: Conv2D → GRU → FC
- **출력**: 각 시간 프레임의 음성 확률
- **장점**: 스펙트로그램 패턴 학습으로 높은 정확도

### 2. 방향 추정 (DOA)

#### GCC-PHAT 알고리즘
```
GCC(τ) = IDFT{ DFT(x₁) × conj(DFT(x₂)) / |DFT(x₁) × conj(DFT(x₂))| }
```
- **입력**: 2채널 오디오 신호
- **처리**: 위상 변환을 통한 상호상관 계산
- **출력**: 시간 지연 τ (Time Delay of Arrival)

#### TDOA → DOA 변환
```
θ = arcsin(c × τ / d)
```
- **τ**: 시간 지연
- **d**: 마이크 간 거리
- **c**: 음속
- **θ**: 방향 각도 (-90° ~ +90°)

### 3. 신호 처리 파이프라인

#### 프레임 처리
1. **오디오 캡처**: 960 샘플 (60ms) 버퍼
2. **프레임 분할**: 320 샘플 (20ms) 단위
3. **오버랩**: 50% 오버랩으로 연속성 확보
4. **실시간 처리**: 120ms DOA 프레임 생성

#### 품질 향상
1. **밴드패스 필터링**: 300-3400Hz 음성 대역 추출
2. **RMS 기반 노이즈 게이트**: 동적 임계값 적용
3. **피치 검증**: 음성 품질 향상을 위한 피치 분석
4. **GCC 신뢰도 평가**: 방향 추정 정확도 검증

### 4. 실시간 최적화

#### 계산 효율성
- **FFT 기반 GCC**: O(N log N) 복잡도
- **벡터화 연산**: NumPy를 통한 SIMD 활용
- **프레임 스킵**: 불필요한 계산 생략

#### 메모리 관리
- **순환 버퍼**: 고정 크기 버퍼로 메모리 사용량 제한
- **실시간 가비지 컬렉션**: 메모리 누수 방지
- **비동기 처리**: UDP 전송과 계산의 분리

## 문제 해결

### 오디오 관련 문제
**문제**: 마이크로폰 장치가 인식되지 않음
```
해결:
1. aplay -l / arecord -l로 장치 확인
2. sudo usermod -a -G audio $USER 권한 설정
3. --dev 파라미터로 장치 지정
```

**문제**: 오디오 품질이 낮음
```
해결:
1. 샘플링 레이트 16kHz 확인
2. 마이크로폰 간 거리 정확 측정
3. 노이즈 환경에서 RMS_THRESH 조정
```

### DOA 관련 문제
**문제**: 방향 추정이 부정확함
```
해결:
1. MIC_DISTANCE_M 실제 거리로 설정
2. GCC_INTERP 값 증가 (8→16)
3. 신뢰도 임계값 조정 (GCC_PEAK_RATIO_MIN, GCC_PROMINENCE_MIN)
```

**문제**: 각도가 튀는 현상
```
해결:
1. EMA_ALPHA 값 증가 (0.25→0.4)
2. MEDIAN_WIN 크기 증가 (7→11)
3. MAX_DEG_PER_FRAME 값 감소 (8.0→4.0)
```

### UDP 통신 문제
**문제**: 데이터가 전송되지 않음
```
해결:
1. IP 주소와 포트 확인
2. 방화벽 설정 확인 (ufw disable)
3. 네트워크 인터페이스 확인 (ifconfig)
```

### 시각화 관련 문제
**문제**: Pygame 창이 열리지 않음
```
해결:
1. SDL_VIDEODRIVER 환경변수 설정
2. X11 또는 Wayland 세션 확인
3. 권한 문제 해결 (sudo 권한으로 실행)
```

### 모델 관련 문제
**문제**: VAD 모델이 로드되지 않음
```
해결:
1. ONNX Runtime 버전 확인
2. 모델 파일 경로 및 권한 확인
3. 입력 차원 일치 확인 (log-mel: [T, 40])
```

### 성능 관련 문제
**문제**: CPU 사용률이 높음
```
해결:
1. VAD_FRAME 크기 증가 (320→640)
2. DOA_FRAME 간격 증가 (120ms→180ms)
3. 불필요한 디버그 출력 제거
```

**문제**: 지연 시간이 길음
```
해결:
1. HOP 크기 감소 (960→480)
2. 버퍼 크기 최적화
3. 실시간 스케줄링 우선순위 설정
```

## 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 상업적 및 비상업적 사용 모두 허용됩니다.

## 기여

버그 리포트, 기능 제안, 코드 기여를 환영합니다. 이슈나 풀 리퀘스트를 통해 참여해주세요.

## 버전 히스토리

- **v1.0.0**: 초기 릴리즈
  - CRNN 기반 VAD 모델 구현
  - GCC-PHAT DOA 알고리즘 통합
  - 실시간 UDP 통신
  - Pygame 시각화 오버레이
  - Raspberry Pi 최적화

---

**주의사항**: 이 시스템은 실시간 성능을 위해 최적화되어 있습니다. 마이크로폰 간 거리 측정이 정확해야 DOA 성능이 보장됩니다.
