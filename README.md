# Thermal Person Detector — YOLOv8 (한국어)

## 📌 프로젝트 소개 (Project Overview)
열화상(thermal) 이미지에서 **사람(person)을 탐지**하기 위한 YOLOv8 기반 워크플로우를 제공합니다.  
Colab 전용이 아니라 **로컬(Windows/macOS/Linux)과 서버 환경 어디서나** 쉽게 실행할 수 있도록 Jupyter 노트북과 ONNX 변환 과정을 포함합니다.

---

## ✨ 주요 기능 (Features)
- 🔎 **YOLOv8 기반 탐지**: 열화상 이미지에서 사람 클래스(0: person) 탐지
- ⚡ **빠른 추론(Quick Inference)**: 사전 학습 가중치(.pt)만 넣으면 즉시 실행
- 🏋️ **(선택) 학습 지원**: YOLO 형식의 커스텀 데이터셋으로 짧은 에폭 데모 학습 가능
- 📤 **ONNX 내보내기**: PyTorch(.pt) → ONNX 변환
- ✅ **무결성 검증**: ONNXRuntime로 입력/출력 shape 확인 & 드라이런 실행
- 🗂️ **표준 폴더 구조**: `data/`, `weights/`, `export/`, `results/` 자동 생성
- 🌐 **포터블 환경**: pip 셀로 의존성 자동 설치 (버전 안내 포함), OS에 독립적으로 동작
- 📝 **친절한 마크다운 안내**: 각 셀마다 한/영 설명(노트북 내부)과 한글 README 제공

---

## 📁 파일 구성 (Project Structure)
```
Embedded/
 ├── LLM-kit/
 │    ├── LLM-kit.ipynb
 │    └── README.md
 │
 ├── Smoke-Dehaze/
 │    ├── Smoke-Dehaze.ipynb
 │    └── README.md
 │
 ├── Thermal-Detect/
 │    ├── Thermal-Detect.ipynb
 │    └── README.md
 │
 ├── requirements.txt   # (선택) 공통 의존성
 └── README.md          # 전체 프로젝트 소개 (세부 모듈 링크 포함)

```

---

## ⚙️ 환경 설정 (Requirements)
- Python **≥ 3.9**
- 필수 패키지(노트북 1번 셀에서 자동 설치됨)
  - `ultralytics >= 8.2.0`
  - `onnx >= 1.15.0`
  - `onnxruntime >= 1.17.0`
  - `opencv-python >= 4.7.0`
  - `numpy >= 1.26.0`
- (GPU 사용 시) CUDA가 설치된 PyTorch 권장

> 💡 팀/프로덕션에서는 `requirements.txt`로 버전 핀(Pin)을 권장합니다.

---

## ▶️ 실행 방법 (Usage)
1. **프로젝트 받기**
   - 깃 저장소를 클론하거나 ZIP으로 다운로드 후 압축을 풉니다.
2. **노트북 실행**
   - `Thermal_Person_Detector_Clean.ipynb` 를 Jupyter/Lab/VS Code에서 엽니다.
3. **의존성 설치**
   - 1번 셀을 실행하면 필요한 패키지가 자동 설치됩니다.
4. **데이터셋 설정(선택)**
   - `data/thermal_person.yaml` 템플릿이 생성됩니다.
   - 본인의 폴더 구조에 맞게 `images/`, `labels/` 경로를 수정하세요.
5. **가중치 준비**
   - 사전 학습 가중치 `.pt` 파일을 `weights/` 폴더에 둡니다.
   - 없다면 노트북에서 YOLOv8n 기본 가중치를 받아 추론 데모가 가능합니다.
6. **빠른 추론**
   - `data/sample.jpg` 가 없으면 노트북이 예시 이미지를 자동 생성합니다.
   - 실행 후 결과 이미지는 `results/infer_once/` 에 저장됩니다.
7. **(선택) 학습**
   - 노트북의 `do_train = False` 를 `True` 로 바꾸고 데이터셋이 준비된 상태에서 실행하세요.
   - 학습 결과 가중치는 `weights/best.pt`(또는 `weights/last.pt`)에 저장됩니다.
8. **ONNX 내보내기 & 검증**
   - 내보낸 모델은 기본적으로 `export/model.onnx` 에 복사됩니다.
   - ONNXRuntime 드라이런으로 I/O shape 및 전방 패스 실행을 확인합니다.

---

## 📊 결과 (Results)
- **추론 결과**: `results/infer_once/` 에 이미지로 저장됩니다.
- **ONNX 모델**: `export/model.onnx`
- **(선택) 학습 산출물**: `weights/best.pt`(또는 `last.pt`), `runs/` 폴더의 학습 로그/메트릭

> 시각화 예: 바운딩 박스가 덧그려진 결과 이미지, 학습 시 loss/precision/recall 곡선(ultralytics 로그)

---

## 📦 ONNX 배포 가이드 (Deployment Guide)

### 📌 소개
본 프로젝트에서는 PyTorch(.pt) 가중치를 **ONNX**로 변환하고, **ONNXRuntime**으로 빠르게 검증할 수 있습니다.  
이를 통해 C++/Python 등 다양한 애플리케이션에 모델을 손쉽게 통합할 수 있습니다.

### ▶️ ONNX 내보내기
노트북 `6) Export to ONNX` 셀에서 수행:
```python
model.export(format="onnx", opset=12, dynamic=True, imgsz=640, half=False)
```

### ✅ 무결성/드라이런 검증
노트북 `7) ONNX Sanity Check` 셀에서 수행:
- `onnx.checker.check_model(...)` 통과 여부
- 입력/출력 이름과 shape (동적 차원인지 확인)
- 임의 입력으로 `sess.run(...)` 전방 패스 성공 여부

### 💻 Python 추론 예시
```python
import onnxruntime as ort, numpy as np, cv2

sess = ort.InferenceSession("export/model.onnx", providers=["CPUExecutionProvider"])
inputs = sess.get_inputs()
input_name = inputs[0].name

img = cv2.imread("data/sample.jpg", cv2.IMREAD_COLOR)
img = cv2.resize(img, (640, 640))
x = img[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
x = np.expand_dims(x, 0)

preds = sess.run(None, {input_name: x})
print([p.shape for p in preds])
```

### ⚡ 배포 팁
- **NMS(Post-processing)**: ONNX 모델 출력은 보통 원시 예측입니다. → 앱에서 NMS 구현 필요
- **동적 입력 지원**: `dynamic=True`로 내보내면 다양한 입력 크기 허용 가능 (후처리도 함께 조정 필요)
- **성능 최적화**: CPU/GPU 백엔드 프로파일링 후 배포 환경에 맞춰 튜닝

---

## 📝 참고 (Notes)
- **데이터 출처/라이선스**: 본 예제는 `jsmithdlc/Thermal-Human-Pose-Estimation`의 **Set-A만 사용**합니다(Set-B 미사용).  
- **Ultralytics 라이선스**: 라이브러리 및 가중치 라이선스를 준수하세요 (AGPL-3.0 등).  
- **버전 호환성**: ONNXRuntime와 모델의 opset, 연산자 지원 여부를 반드시 확인하세요.
- **자주 묻는 이슈**
  - *CUDA가 감지되지 않음*: GPU 드라이버/툴킷 설치 확인
  - *패키지 충돌*: 가상환경(venv/conda) 권장
  - *출력 shape 해석 어려움*: 노트북 검증 셀에서 출력 텐서 목록/shape 확인
  - *NMS 누락*: 앱 코드에서 직접 구현 필요
  - *동적 입력 오류*: 전처리/후처리 스케일 일관성 유지
