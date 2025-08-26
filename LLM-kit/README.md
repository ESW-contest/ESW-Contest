# CRNN-based Voice Activity Detection (VAD) 🎙️

## 프로젝트 소개 (Project Overview)
**CRNN (Convolutional Recurrent Neural Network)** 기반으로 음성 활동 감지(Voice Activity Detection, VAD)를 구현한 프로젝트입니다.  
Jupyter Notebook **`LLM-kit.ipynb`** 로 학습, 평가, 결과 시각화를 실행할 수 있습니다.

---

## 주요 기능 (Features)
- CRNN 구조 기반 음성 활동 감지 모델
- Librosa 기반 오디오 데이터 로딩 및 전처리
- 학습 루프 및 손실/정확도 모니터링
- 학습/검증 데이터 분리 및 DataLoader 활용
- 결과 시각화 및 성능 평가
- bilingual (영문/한글) 주석 포함

---

## 파일 구성 (Project Structure)
```
.
├── LLM-kit.ipynb          # CRNN VAD 학습/설명 노트북
├── requirements.txt       # 필수 라이브러리
└── README.md              # 프로젝트 설명 문서
```

---

## 환경 설정 (Requirements)
```bash
pip install torch torchvision torchaudio
pip install librosa soundfile numpy
```

---

## 실행 방법 (Usage)
```bash
jupyter notebook LLM-kit.ipynb
```

---

## 결과 (Results)
- 학습/검증 정확도 및 손실 곡선  
- 음성 데이터에 대한 VAD 성능 확인  
- 모델 구조: CNN feature extractor + BiLSTM + Fully Connected layers  

---

## 참고 (Notes)
- 실제 데이터셋 크기에 따라 학습 시간 상이  
- 한국어/영어 주석 포함  

---

_작성: 2025-08-26 02:29:40_
