# Thermel-Detect 🔥👁️

## 프로젝트 소개 (Project Overview)
열상(thermal) 이미지를 활용해 객체를 감지하는 모델을 구현한 프로젝트입니다.  
Jupyter Notebook **`Thermel-Detect.ipynb`** 하나로 데이터 로딩 → 학습 → 평가 → 시각화 → 결과 저장까지 전체 파이프라인을 실행할 수 있습니다.

---

## 주요 기능 (Features)
- 열상 이미지 데이터 로딩 및 전처리
- 딥러닝 기반 열상 감지 모델 학습
- 평가 (정확도, F1 등) 및 추론
- 결과 시각화 및 그래프 출력
- 학습된 모델 및 결과 산출물 저장

---

## 파일 구성 (Project Structure)
```
.
├── Thermel-Detect.ipynb   # 메인 노트북 (전체 파이프라인)
├── requirements.txt       # 필수 라이브러리
└── README.md              # 프로젝트 설명 문서
```

---

## 환경 설정 (Requirements)
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## 실행 방법 (Usage)
1. 저장소를 클론하거나 압축파일 다운로드  
2. Jupyter Lab / Notebook / VS Code 에서 `Thermel-Detect.ipynb` 열기  
3. 셀을 순서대로 실행  

---

## 결과 (Results)
- 학습 곡선 및 시각화 그래프  
- 모델 가중치와 평가 결과 (`outputs/` 폴더에 저장)  

---

## 참고 (Notes)
- 열상 데이터셋은 저장소에 포함되지 않습니다. 사용 전 `data/` 폴더에 직접 추가해야 합니다.  
- `outputs/` 폴더에는 학습된 모델 가중치, 평가 결과, 그래프 이미지 등이 저장됩니다.  

---

_작성: 2025-08-26 02:29:40_
