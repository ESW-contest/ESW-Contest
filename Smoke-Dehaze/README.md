# Thermal Person Detector — YOLOv8 (열화상 사람 탐지)

## 📌 프로젝트 소개 (Project Overview)
이 프로젝트는 **열화상(thermal) 이미지에서 사람을 탐지**하기 위해 YOLOv8 모델을 사용한 워크플로우를 제공합니다.  
초보자도 쉽게 실행할 수 있도록 Jupyter Notebook과 ONNX 변환 과정을 포함합니다.

## ✨ 주요 기능 (Features)
- 📂 YOLOv8 기반 열화상 이미지 사람 탐지
- ⚡ 사전 학습된 가중치로 빠른 추론 실행
- 🛠️ (선택) 커스텀 데이터셋 학습 지원
- 📤 PyTorch → ONNX 내보내기 및 무결성 체크
- 🔍 샘플 이미지 자동 생성 및 시각화 결과 저장
- 💻 Colab 전용이 아닌, 로컬/서버 어디서나 실행 가능

## 📁 파일 구성 (Project Structure)
```
Thermal_Person_Detector_Clean.ipynb   # 메인 노트북 (설치, 추론, 학습, ONNX 내보내기 포함)
weights/                              # 사전 학습 또는 학습된 가중치 저장 위치
data/                                 # 데이터셋 (images/, labels/, .yaml)
export/                               # 내보낸 ONNX 파일 저장 위치
results/                              # 추론 결과 이미지 저장 위치
```

## ⚙️ 환경 설정 (Requirements)
- Python ≥ 3.9
- 필수 패키지:
  - ultralytics ≥ 8.2.0
  - onnx ≥ 1.15.0
  - onnxruntime ≥ 1.17.0
  - opencv-python ≥ 4.7.0
  - numpy ≥ 1.26.0

설치는 노트북 첫 번째 셀에서 자동으로 진행됩니다.

## ▶️ 실행 방법 (Usage)
1. 저장소를 클론하거나 다운로드합니다.
2. `Thermal_Person_Detector_Clean.ipynb` 노트북을 열어 단계별로 실행합니다.
3. (선택) `data/thermal_person.yaml` 을 수정하여 데이터셋 경로를 맞춥니다.
4. 사전 학습된 가중치(.pt)가 있다면 `weights/` 폴더에 넣습니다.
5. 노트북을 실행하여 추론, 학습, ONNX 변환을 진행합니다.

## 📊 결과 (Results)
- 추론 결과는 `results/` 폴더에 이미지 파일로 저장됩니다.
- ONNX 변환된 모델은 `export/model.onnx` 에 생성됩니다.
- (선택) 학습 진행 시 `weights/best.pt` 파일이 저장됩니다.

## 📝 참고 (Notes)
- 본 프로젝트는 jsmithdlc/Thermal-Human-Pose-Estimation 의 **Set-A 데이터만** 사용합니다. (Set-B 미사용)
- 데이터 및 가중치 라이선스를 반드시 확인 후 사용하세요.
- 재현성을 위해 requirements.txt 또는 노트북 환경을 그대로 사용하는 것을 권장합니다.
