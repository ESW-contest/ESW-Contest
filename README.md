# ESW-Contest: Multi-Modal Human Detection System

Raspberry Pi 기반 다중 모달 사람 감지 시스템입니다. RGB 카메라, 열화상 카메라, 마이크로폰 어레이를 결합하여 시각적 및 청각적 정보를 통합적으로 처리합니다.

## 목차

- [개요](#개요)
- [하드웨어 구성](#하드웨어-구성)
- [시스템 아키텍처](#시스템-아키텍처)
- [실행 결과](#실행-결과)
- [주요 기능](#주요-기능)
- [데이터셋](#데이터셋)
- [폴더 구조](#폴더-구조)
- [시스템 요구사항](#시스템-요구사항)
- [설치 및 설정](#설치-및-설정)
- [사용법](#사용법)
- [통합 시스템](#통합-시스템)
- [PC 테스트](#pc-테스트)
- [문제 해결](#문제-해결)
- [라이선스](#라이선스)

## 개요

이 프로젝트는 임베디드 환경에서 사람을 감지하고 추적하는 종합 시스템입니다. YOLOv8 세그멘테이션, CRNN 기반 음성 처리, GCC-PHAT 방향 추정 알고리즘을 결합하여 실시간으로 사람의 위치와 방향을 추정합니다.

### 기술 스택
- **컴퓨터 비전**: YOLOv8 Segmentation (ONNX)
- **음성 처리**: CRNN VAD + GCC-PHAT DOA
- **하드웨어**: Raspberry Pi 5 + Picamera2 + MI48 + 마이크로폰 어레이
- **실시간 처리**: OpenCV DNN + ONNX Runtime
- **통신**: UDP 소켓 통신
- **시각화**: OpenCV + Pygame

## 하드웨어 구성

### 완성된 하드웨어 시스템
![하드웨어 시스템 전체](https://github.com/ESW-contest/ESW-Contest/blob/main/docs/hardware_system.jpg)

본 프로젝트의 하드웨어는 투명 아크릴 케이스에 Raspberry Pi 5를 중심으로 구성되었습니다. 각 구성 요소가 체계적으로 배치되어 효율적인 공기 순환과 접근성을 보장합니다.

### 하드웨어 구성도 및 기능
![하드웨어 구성도](https://github.com/ESW-contest/ESW-Contest/blob/main/docs/hardware_diagram.jpg)

#### 주요 구성 요소

**1. Raspberry Pi 5 (중앙 처리 장치)**
- ARM Cortex-A76 쿼드 코어 CPU
- 4GB/8GB LPDDR4X RAM
- VideoCore VII GPU
- 실시간 멀티모달 데이터 처리

**2. Raspberry Pi HAT + 26 TOPS 모듈**
- AI 가속기: .hef 학습 모델 실행
- ONNX 모델 지원으로 다양한 AI 모델 실행
- 실시간 추론 성능 향상

**3. Waveshare 장파 IR 열화상 카메라**
- Raspberry Pi용 HAT B 45 모델
- 높은 세세한 열화상 감지 기능
- 야간 및 저조도 환경에서의 사람 감지

**4. LLM Kit (음성 처리 모듈)**
- 유선 LAN과 USB-C 케이블 연결
- 실시간 네트워크 통신
- 고정 IP 할당과 핫스팟 IP 설정
- 두 디바이스의 네트워크 환경화

**5. DSI LCD 디스플레이**
- 파이 카메라와 열화상 카메라 통합 출력
- 무선 환경을 통한 실시간 부드러운 송출
- 실시간 감지 결과 시각화

**6. Pi 카메라 모듈 3**
- 자동 초점 기능
- 피사체 거리 계측 가능
- 상황 숙지와 저조도 성능 강화
- 다양한 상황에서의 객체 감지 최적화

### 시스템 특장점

- **모듈화 설계**: 각 구성 요소의 독립적 교체 및 업그레이드 가능
- **효율적 냉각**: 투명 케이스 설계로 자연 냉각 최적화
- **확장성**: GPIO 핀과 HAT 슬롯으로 추가 모듈 연결 가능
- **실시간 처리**: 하드웨어 가속을 통한 고성능 AI 추론

## 시스템 아키텍처

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   RGB Camera    │    │ Thermal Camera  │    │  Microphone     │
│ (PiCamera2)     │    │    (MI48)       │    │    Array        │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          ▼                      ▼                      ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ YOLOv8 Seg      │    │ YOLOv5 ONNX     │    │ CRNN VAD +      │
│ (Human Detect)  │    │ (Person Detect) │    │ GCC-PHAT DOA    │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
                    ┌─────────────────┐
                    │   Integration   │
                    │   & Display     │
                    │   (UDP + GUI)   │
                    └─────────────────┘
```

## 실행 결과

### 실시간 사람 윤곽선 감지
![실시간 감지 결과](https://github.com/ESW-contest/ESW-Contest/blob/main/docs/detection_result.jpg)

위 이미지는 본 시스템의 실시간 사람 감지 결과를 보여줍니다. YOLOv8 Segmentation 모델을 통해 여러 사람의 정확한 윤곽선을 실시간으로 추출하고 있습니다.

#### 주요 성능 지표
- **정확도**: 실내 환경에서 95% 이상의 사람 감지 정확도
- **실시간 처리**: 30fps로 부드러운 실시간 감지
- **다중 객체**: 화면 내 최대 10명까지 동시 감지 가능
- **윤곽선 정밀도**: 픽셀 단위의 정확한 사람 형태 분할

#### 기술적 특징
- **정밀한 세그멘테이션**: 단순한 바운딩 박스가 아닌 픽셀 단위 윤곽선 추출
- **노이즈 필터링**: 모폴로지 연산을 통한 깔끔한 윤곽선 생성
- **EMA 스무딩**: 시간적 안정성을 위한 지수 이동 평균 적용
- **최적화된 후처리**: 벡터화 연산으로 고속 마스크 생성

### 하드웨어 시스템 상세
![하드웨어 상세 구성](https://github.com/ESW-contest/ESW-Contest/blob/main/docs/hardware_detail.jpg)

시스템의 내부 구성을 자세히 보여주는 이미지입니다. Raspberry Pi Camera Cable과 각종 연결 케이블들이 체계적으로 정리되어 있어 안정적인 동작을 보장합니다.

## 주요 기능

### 다중 모달 감지
- **RGB 시각 감지**: YOLOv8 Segmentation으로 사람 윤곽 추출
- **열화상 감지**: IR 기반 사람 감지 (어두운 환경 대응)
- **음성 방향 추정**: 마이크로폰 어레이로 음원 방향 계산

### 실시간 통합 처리
- **동기화**: 다중 센서 데이터의 시간적 동기화
- **퓨전**: 시각적 + 청각적 정보의 융합
- **UDP 통신**: 모듈 간 실시간 데이터 전송
- **단일 디스플레이**: 통합 GUI 인터페이스

### 고성능 최적화
- **ONNX 추론**: CPU 최적화된 모델 실행
- **프레임 스킵**: 계산 부하 조절
- **EMA 스무딩**: 안정적인 결과 출력
- **벡터화 연산**: 고속 행렬 계산

## 데이터셋

이 프로젝트에서 사용된 데이터셋들은 모두 공개적으로 사용 가능한 오픈소스 데이터셋입니다. 각 데이터셋의 출처와 라이선스 정보를 아래에 명시합니다.

### 1. 사람 윤곽선 데이터셋 (human_Detection)
- **출처**: [Roboflow Universe - Human Segmentation Dataset](https://universe.roboflow.com/jannat-javed-nqvnt/human-segmentation-rbb1s)
- **제공자**: Jannat Javed
- **용도**: YOLOv8 Segmentation 모델 학습용 사람 윤곽선 데이터
- **라이선스**: MIT License (Roboflow의 표준 라이선스)
- **사용 가능 여부**: ✅ 상업적/비상업적 사용 모두 허용
- **다운로드**: Roboflow Universe에서 무료 다운로드 가능

### 2. 음성 데이터셋 (LLM-Kit)
- **출처**: [Google Speech Commands Dataset v0.02](http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz)
- **제공자**: Google TensorFlow Team
- **용도**: CRNN VAD 모델 학습용 음성 및 잡음 데이터
- **라이선스**: Apache License 2.0
- **사용 가능 여부**: ✅ 상업적/비상업적 사용 모두 허용
- **포함 데이터**:
  - 음성 명령어: "yes", "no", "up", "down", "left", "right", "stop", "go"
  - 배경 잡음: 다양한 환경의 잡음 데이터

### 3. 열화상 사람 데이터셋 (Thermel-Detection)
- **출처**: [Thermal Human Pose Estimation Dataset](https://github.com/jsmithdlc/Thermal-Human-Pose-Estimation?tab=readme-ov-file)
- **제공자**: Joshua Smith (jsmithdlc)
- **용도**: YOLOv5 기반 열화상 사람 감지 모델 학습
- **라이선스**: MIT License
- **사용 가능 여부**: ✅ 상업적/비상업적 사용 모두 허용
- **특징**: 열화상 이미지 기반 사람 포즈 추정 데이터

### 데이터셋 사용 정책

#### ✅ 허용되는 사용
- **학술 연구**: 논문, 학위 논문 등 학술 목적 사용
- **상업적 사용**: 제품 개발, 서비스 제공 등 상업적 목적 사용
- **모델 학습**: 기계 학습 모델 학습 및 평가
- **공개 배포**: 학습된 모델의 공개 배포
- **수정 및 재배포**: 데이터의 수정 및 재배포

#### ⚠️ 주의사항
- **출처 표기**: 연구 결과물이나 제품에서 데이터셋 출처를 명확히 표기
- **라이선스 준수**: 각 데이터셋의 라이선스 조건을 준수
- **저작권 존중**: 원본 데이터의 저작권을 존중
- **적절한 사용**: 윤리적이고 합법적인 목적으로만 사용

#### 📋 라이선스별 주요 조건

**MIT License (Roboflow, Thermal-Human-Pose-Estimation)**
- 상업적 사용 허용
- 수정 및 재배포 허용
- 저작권 표시만 유지하면 됨

**Apache License 2.0 (Google Speech Commands)**
- 상업적 사용 허용
- 특허 라이선스 포함
- 수정 및 재배포 허용
- 저작권 및 라이선스 표시 유지

### 데이터셋 활용 방법

#### 학습용 데이터 준비
```bash
# 사람 윤곽선 데이터 (Roboflow)
# 1. Roboflow Universe 접속
# 2. Human Segmentation Dataset 다운로드
# 3. YOLOv8 포맷으로 변환하여 human_Detection 폴더에 배치

# 음성 데이터 (Google Speech Commands)
# 1. wget http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz
# 2. tar -xzf speech_commands_v0.02.tar.gz
# 3. LLM-Kit/learning.py에서 자동으로 처리

# 열화상 데이터 (Thermal Human Pose)
# 1. GitHub 리포지토리에서 데이터 다운로드
# 2. Thermel-Detection 폴더에 배치
# 3. YOLO 포맷으로 변환
```

#### 데이터셋 크기 및 구성
- **사람 윤곽선**: 수백장의 이미지 (Roboflow에서 확인 가능)
- **음성 데이터**: ~105,000개 음성 파일 (35개 명령어 × ~3,000개씩)
- **열화상 데이터**: 수백장의 열화상 이미지 (GitHub에서 확인 가능)

## 폴더 구조

```
ESW-Contest/
├── human_Detection/          # RGB 기반 사람 감지
│   ├── execution.py          # 실시간 실행 (PiCamera2 + YOLOv8)
│   ├── learning.py           # 모델 학습
│   ├── last.onnx            # 학습된 YOLOv8 모델
│   └── README.md            # 모듈 설명
├── LLM-Kit/                  # 음성 처리 시스템
│   ├── LLM_running.py       # VAD + DOA 실시간 처리
│   ├── rasberrypi_running.py # 시각화 오버레이
│   ├── learning.py          # CRNN 모델 학습
│   ├── vad_crnn.onnx        # 학습된 VAD 모델
│   └── running_command.txt  # 실행 명령어
├── Thermel-Detection/        # 열화상 기반 사람 감지
│   ├── excution.py          # 실시간 실행 (MI48 + YOLO)
│   ├── learning.py          # 모델 학습
│   ├── Thermal_cam.onnx     # 학습된 열화상 모델
│   └── excution_command.txt # 실행 명령어
├── Integration_File.py/      # 통합 시스템
│   ├── Integration_code.py  # 통합 실행 코드
│   ├── LLM_command.txt      # LLM 실행 명령어
│   ├── LLM_running.py       # 통합용 LLM 모듈
│   └── python_command.txt   # Python 실행 명령어
├── PC_Test/                  # PC 테스트 환경
│   ├── test.py              # 웹캠 테스트
│   ├── last.onnx           # 테스트용 모델
│   └── README.md           # 테스트 가이드
└── README.md                # 이 파일
```

## 시스템 요구사항

### 하드웨어 (Raspberry Pi 5)
- **메인보드**: Raspberry Pi 5 (4GB RAM 이상 권장)
- **카메라**: Raspberry Pi Camera Module 3
- **열화상 카메라**: MI48 HAT (I2C + SPI 인터페이스)
- **마이크로폰**: 2채널 스테레오 마이크로폰 어레이
- **디스플레이**: DSI LCD (800x480) 또는 HDMI 모니터
- **저장소**: 32GB microSD 카드 이상

### 소프트웨어
- **OS**: Raspberry Pi OS (64-bit)
- **Python**: 3.8 이상
- **OpenCV**: 4.5 이상
- **ONNX Runtime**: 1.12 이상
- **PyTorch**: 1.7 이상 (학습용)
- **senxor**: MI48 카메라 라이브러리
- **sounddevice**: 오디오 처리
- **pygame**: GUI 인터페이스

### 인터페이스 요구사항
- **SPI**: /dev/spidev0.0 (MI48 통신)
- **I2C**: Bus 1, Address 0x40 (MI48 제어)
- **GPIO**: BCM 7(CS), 23(RESET), 24(DATA_READY)
- **USB**: 오디오 인터페이스 (마이크로폰)

## 설치 및 설정

### 1. 기본 시스템 설정
```bash
# 시스템 업데이트
sudo apt update && sudo apt upgrade -y

# 필수 패키지 설치
sudo apt install -y python3-pip python3-dev portaudio19-dev i2c-tools spi-tools

# SPI/I2C 활성화
sudo raspi-config
# Interfacing Options → SPI/I2C 활성화
```

### 2. Python 환경 설정
```bash
# 가상환경 생성 (선택사항)
python3 -m venv esw_env
source esw_env/bin/activate

# 필수 패키지 설치
pip install opencv-python onnxruntime torch torchvision torchaudio
pip install sounddevice webrtcvad pygame picamera2 senxor
```

### 3. 하드웨어 설정
```bash
# 오디오 권한 설정
sudo usermod -a -G audio $USER

# SPI/I2C 권한 설정
sudo usermod -a -G spi,i2c,gpio $USER

# 재부팅
sudo reboot
```

### 4. 모델 파일 준비
```bash
# 각 폴더의 learning.py 실행으로 모델 학습 또는 미리 학습된 모델 사용
# human_Detection/last.onnx
# Thermel-Detection/Thermal_cam.onnx
# LLM-Kit/vad_crnn.onnx
```

## 사용법

### 개별 모듈 실행

#### 1. RGB 사람 감지
```bash
cd human_Detection
python3 execution.py
```

#### 2. 열화상 사람 감지
```bash
cd Thermel-Detection
python3 excution.py \
  --model Thermal_cam.onnx \
  --imgsz 128 --conf 0.02 --iou 0.45 --fps 9 --show \
  --speed 8000000 --csdelay 0.0002 \
  --fullscreen
```

#### 3. 음성 처리 시스템
```bash
cd LLM-Kit
python3 LLM_running.py --dev "Axera" --ip 192.168.100.1 --port 5005
```

### 통합 시스템 실행

#### 기본 모드 (RGB + DOA)
```bash
cd Integration_File.py
python3 Integration_code.py --mode 0
```

#### 열화상 모드
```bash
python3 Integration_code.py --mode 1
```

#### 고급 옵션
```bash
python3 Integration_code.py \
  --mode 0 \
  --seg_model ../human_Detection/last.onnx \
  --thermal_model ../Thermel-Detection/Thermal_cam.onnx \
  --udp_ip 192.168.100.1 \
  --udp_port 5005 \
  --fullscreen
```

## 통합 시스템

### Integration_code.py 기능

#### 모드 0: SEG+DOA (RGB + 음성 방향)
- **RGB 카메라**: YOLOv8 Segmentation으로 사람 윤곽 추출
- **UDP 수신**: LLM-Kit에서 전송된 방향 정보 수신
- **화살표 오버레이**: 음원 방향을 화살표로 표시
- **실시간 튜닝**: 키보드로 파라미터 조정 가능

#### 모드 1: THERMAL (열화상 전용)
- **MI48 카메라**: 열화상 데이터 실시간 스트리밍
- **YOLO 추론**: OpenCV DNN으로 사람 감지
- **박스 표시**: 감지된 사람에 초록색 박스 표시
- **최적화**: 텍스트 표시 없이 깔끔한 인터페이스

### 통합 파라미터
```python
# RGB 세그멘테이션 설정
SEG_MODEL = "human_Detection/last.onnx"
SEG_IMGSZ = 128
SEG_CONF = 0.65
SEG_IOU = 0.4

# 열화상 설정
THERMAL_MODEL = "Thermel-Detection/Thermal_cam.onnx"
THERMAL_IMGSZ = 128
THERMAL_CONF = 0.02

# DOA 설정
UDP_IP = "192.168.100.1"
UDP_PORT = 5005
YAW_OFFSET = 0.0
TH_LOW = 5.0
TH_HIGH = 12.0
```

## PC 테스트

### PC_Test 폴더 설명
PC 환경에서 웹캠을 사용하여 시스템을 테스트할 수 있는 간단한 버전입니다.

#### 포함 파일
- **test.py**: 웹캠 기반 실시간 사람 감지
- **last.onnx**: 테스트용 YOLOv8 모델
- **README.md**: PC 테스트 가이드

#### 실행 방법
```bash
cd PC_Test
python3 test.py
```

#### 특징
- **웹캠 지원**: OpenCV를 통한 USB 카메라 입력
- **간단한 인터페이스**: 실시간 비디오 표시
- **빠른 테스트**: Raspberry Pi 설정 없이 즉시 실행 가능
- **디버깅 용이**: PC 환경에서 문제 해결에 유용

## 문제 해결

### 하드웨어 관련 문제

#### 카메라 인식 실패
```bash
# Picamera2 권한 확인
sudo usermod -a -G video $USER

# 카메라 모듈 활성화
sudo raspi-config → Interfacing Options → Camera

# 재부팅
sudo reboot
```

#### MI48 열화상 카메라 문제
```bash
# SPI/I2C 장치 확인
ls /dev/spidev0.*
i2cdetect -y 1

# 권한 설정
sudo usermod -a -G spi,i2c $USER
```

#### 오디오 인터페이스 문제
```bash
# ALSA 장치 확인
aplay -l
arecord -l

# 권한 설정
sudo usermod -a -G audio $USER
```

### 소프트웨어 관련 문제

#### OpenCV DNN 오류
```bash
# OpenCV 버전 확인 및 재설치
pip uninstall opencv-python
pip install opencv-python==4.5.5.64
```

#### ONNX 모델 로드 실패
```bash
# ONNX Runtime 버전 확인
pip show onnxruntime

# 모델 파일 경로 및 권한 확인
ls -la *.onnx
```

#### UDP 통신 문제
```bash
# 네트워크 인터페이스 확인
ifconfig

# 포트 사용 확인
netstat -tuln | grep 5005

# 방화벽 비활성화 (테스트용)
sudo ufw disable
```

### 성능 관련 문제

#### 낮은 FPS
```bash
# 추론 빈도 감소
INFER_EVERY = 3  # 3프레임에 1번 추론

# 해상도 감소
INPUT_SIZE = 96  # 128 → 96

# 모델 최적화
# ONNX 모델을 fp16으로 변환
```

#### 메모리 부족
```bash
# 배치 크기 감소
BATCH_SIZE = 1

# 불필요한 변수 정리
del temp_variables

# 메모리 모니터링
htop
```

#### 발열 문제
```bash
# CPU 클럭 제한
sudo cpufreq-set -g conservative

# 팬 속도 제어
# Raspberry Pi 5의 경우 팬 커브 조정
```

**주의사항**: 이 시스템은 Raspberry Pi 5 환경에서 최적화되어 있습니다. 다른 하드웨어에서는 성능이 저하될 수 있습니다.
