#!/usr/bin/env python3
"""
Learn Detection - YOLOv8 세그멘테이션 모델 학습 및 배포
====================================================

이 스크립트는 YOLOv8을 활용한 인스턴스 세그멘테이션 모델의
완전한 학습 및 배포 파이프라인을 제공합니다.

주요 기능:
- 자동 패키지 설치 (ultralytics, roboflow 등)
- Roboflow 데이터셋 다운로드 및 전처리 또는 로컬 데이터셋 사용
- YOLOv8 세그멘테이션 모델 학습
- ONNX/TorchScript 모델 내보내기
- 테스트 이미지 추론 및 시각화
- 실시간 학습 진행 상황 표시

요구사항:
    - Python 3.7 이상
    - CUDA 지원 GPU (선택사항, CPU에서도 실행 가능)
    - 최소 8GB RAM 권장

사용법:
    python learn_detection.py

데이터 준비:
    1. Roboflow에서 데이터셋 다운로드 (권장)
       - https://app.roboflow.com 에서 계정 생성
       - 세그멘테이션 데이터셋 업로드
       - API 키 및 프로젝트 정보 확인

    2. 로컬 data.yaml 파일 사용
       - YOLOv8 형식의 데이터셋 준비
       - data.yaml 파일 생성

출력 파일들:
    - runs/segment/roboflow_yolov8/weights/best.pt (학습된 모델)
    - runs/segment/roboflow_yolov8/weights/best.onnx (ONNX 모델)
    - runs/segment/roboflow_yolov8/results.csv (학습 로그)
    - predict/ 폴더 (추론 결과 이미지들)

작성자: GitHub Copilot
버전: 1.0.0
"""

import subprocess
import sys
import os
import yaml
import random
import cv2
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from PIL import Image

def install_packages():
    """필요한 패키지들을 자동으로 설치합니다.

    설치되는 패키지들:
    - ultralytics: YOLOv8 프레임워크
    - roboflow: 데이터셋 다운로드
    - opencv-python: 이미지 처리
    - matplotlib: 시각화
    - pyyaml: 설정 파일 처리
    - pillow: 이미지 처리

    Returns:
        None
    """
    packages = [
        "ultralytics",
        "roboflow==1.1.33",
        "opencv-python",
        "matplotlib",
        "pyyaml",
        "pillow"
    ]

    print("필요한 패키지들을 설치하는 중...")
    for package in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"✓ {package} 설치 완료")
        except subprocess.CalledProcessError as e:
            print(f"✗ {package} 설치 실패: {e}")
    print("패키지 설치 완료!\n")

def check_environment():
    """Python과 PyTorch 환경을 확인합니다"""
    print("=== 환경 확인 ===")
    print(f"Python 버전: {sys.version}")

    try:
        import torch
        print(f"PyTorch 버전: {torch.__version__}")
        print(f"CUDA 사용 가능: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("PyTorch가 설치되지 않았습니다")

    try:
        import ultralytics
        print(f"Ultralytics 버전: {ultralytics.__version__}")
    except ImportError:
        print("Ultralytics가 설치되지 않았습니다")

    print()

def get_dataset_from_roboflow():
    """Roboflow에서 데이터셋을 다운로드합니다.

    사용자 입력으로 API 키, 워크스페이스, 프로젝트 정보를 받습니다.

    Returns:
        str or None: data.yaml 파일 경로 또는 None (실패 시)
    """
    print("=== Roboflow에서 데이터셋 다운로드 ===")

    # 사용자 입력으로 자격증명 받기 (하드코딩 대신)
    api_key = input("Roboflow API 키를 입력하세요: ").strip()
    workspace = input("Roboflow 워크스페이스 이름을 입력하세요: ").strip()
    project = input("Roboflow 프로젝트 이름을 입력하세요: ").strip()
    version = input("Roboflow 버전 번호를 입력하세요 (기본값: 1): ").strip() or "1"

    if not api_key or not workspace or not project:
        print("❌ 모든 필드가 필요합니다. 데이터셋 다운로드를 건너뜁니다.")
        return None

    try:
        from roboflow import Roboflow

        rf = Roboflow(api_key=api_key)
        project_obj = rf.workspace(workspace).project(project)
        dataset = project_obj.version(int(version)).download("yolov8")

        data_yaml = os.path.join(dataset.location, "data.yaml")
        print(f"✅ 데이터셋 다운로드 위치: {dataset.location}")
        print(f"📄 데이터 YAML: {data_yaml}")
        return data_yaml

    except ImportError:
        print("❌ Roboflow 패키지가 설치되지 않았습니다.")
        return None
    except ValueError as e:
        print(f"❌ 버전 번호가 잘못되었습니다: {e}")
        return None
    except Exception as e:
        print(f"❌ 데이터셋 다운로드 실패: {e}")
        return None

def fix_data_yaml(data_yaml):
    """로컬 실행을 위한 data.yaml 경로를 수정합니다"""
    if not data_yaml or not os.path.exists(data_yaml):
        print("❌ data.yaml을 찾을 수 없습니다")
        return

    print("=== data.yaml 경로 수정 ===")

    # 데이터셋 루트 디렉토리 가져오기
    dataset_root = os.path.dirname(data_yaml)

    with open(data_yaml, 'r') as f:
        data = yaml.safe_load(f)

    # 경로 업데이트
    data["train"] = os.path.join(dataset_root, "train", "images")
    data["val"] = os.path.join(dataset_root, "valid", "images")
    data["test"] = os.path.join(dataset_root, "test", "images")

    # path 키가 있으면 제거
    if "path" in data:
        del data["path"]

    # 업데이트된 yaml 저장
    with open(data_yaml, 'w') as f:
        yaml.safe_dump(data, f, sort_keys=False)

    print("✅ data.yaml 업데이트 완료")
    print("업데이트된 경로:")
    for key in ["train", "val", "test"]:
        path = data[key]
        exists = os.path.isdir(path)
        print(f"  {key}: {path} {'✓' if exists else '✗'}")

    print()

def visualize_sample(data_yaml):
    """세그멘테이션 라벨과 함께 샘플 이미지를 시각화합니다"""
    if not data_yaml:
        print("❌ data.yaml이 제공되지 않았습니다")
        return

    print("=== 샘플 이미지 시각화 ===")

    dataset_root = os.path.dirname(data_yaml)
    img_dir = os.path.join(dataset_root, "train", "images")
    label_dir = os.path.join(dataset_root, "train", "labels")

    if not os.path.exists(img_dir):
        print(f"❌ 이미지 디렉토리를 찾을 수 없습니다: {img_dir}")
        return

    # 샘플 이미지 가져오기
    img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    if not img_files:
        print("❌ 이미지 파일을 찾을 수 없습니다")
        return

    sample_img = random.choice(img_files)
    img_path = os.path.join(img_dir, sample_img)
    label_path = os.path.join(label_dir, sample_img.replace('.jpg', '.txt').replace('.png', '.txt'))

    print(f"샘플: {sample_img}")

    # 이미지 읽기
    img = cv2.imread(img_path)
    if img is None:
        print("❌ 이미지 읽기 실패")
        return

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]

    # 라벨 읽고 그리기
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 6:  # 클래스 + 최소 3개의 점
                    continue

                cls = int(parts[0])
                coords = list(map(float, parts[1:]))

                # 세그멘테이션 폴리곤 그리기
                pts = []
                for i in range(0, len(coords), 2):
                    x = int(coords[i] * w)
                    y = int(coords[i+1] * h)
                    pts.append([x, y])

                if len(pts) >= 3:
                    pts = np.array(pts, np.int32).reshape((-1, 1, 2))
                    cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
    else:
        print("⚠️ 라벨 파일을 찾을 수 없습니다")

    # 이미지 표시
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.title(f"샘플: {sample_img}")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    print()

def train_model(data_yaml):
    """YOLOv8 세그멘테이션 모델을 학습합니다.

    Args:
        data_yaml (str): data.yaml 파일 경로

    Returns:
        str or None: 학습된 모델 경로 또는 None (실패 시)
    """
    if not data_yaml:
        print("❌ data.yaml이 제공되지 않았습니다")
        return

    print("=== YOLOv8 세그멘테이션 모델 학습 ===")

    # 학습 파라미터
    model_name = "yolov8n-seg.pt"
    epochs = 80
    imgsz = 640
    batch = 16
    project = "runs/segment"
    name = "roboflow_yolov8"

    # 학습 명령어 구성
    cmd = [
        "yolo", "segment", "train",
        f"model={model_name}",
        f"data={data_yaml}",
        f"epochs={epochs}",
        f"imgsz={imgsz}",
        f"batch={batch}",
        f"project={project}",
        f"name={name}",
        "exist_ok=True"
    ]

    print(f"학습 명령어: {' '.join(cmd)}")
    print("학습을 시작합니다... (시간이 오래 걸릴 수 있습니다)")

    try:
        # 실시간 출력 표시를 위해 capture_output=False로 변경
        result = subprocess.run(cmd, capture_output=False, text=True)

        if result.returncode == 0:
            print("✅ 학습이 성공적으로 완료되었습니다!")
            weights_path = os.path.join(project, name, "weights", "best.pt")
            print(f"📁 최적 가중치 저장 위치: {weights_path}")
            return weights_path
        else:
            print("❌ 학습 실패")
            return None

    except Exception as e:
        print(f"❌ 학습 오류: {e}")
        return None

def export_to_onnx(model_path):
    """학습된 모델을 ONNX 형식으로 내보냅니다"""
    if not model_path or not os.path.exists(model_path):
        print("❌ 모델 경로를 찾을 수 없습니다")
        return

    print("=== ONNX로 내보내기 ===")

    try:
        from ultralytics import YOLO

        model = YOLO(model_path)
        export_path = model.export(format="onnx", opset=12, simplify=True, dynamic=True)

        print(f"✅ ONNX 모델 내보내기 완료: {export_path}")
        return export_path

    except Exception as e:
        print(f"❌ ONNX 내보내기 실패: {e}")
        return None

def run_inference(model_path, data_yaml):
    """테스트 이미지에 대해 추론을 실행합니다"""
    if not model_path or not os.path.exists(model_path):
        print("❌ 모델 경로를 찾을 수 없습니다")
        return

    if not data_yaml:
        print("❌ data.yaml이 제공되지 않았습니다")
        return

    print("=== 추론 실행 ===")

    try:
        from ultralytics import YOLO

        model = YOLO(model_path)

        # 테스트 이미지 가져오기
        dataset_root = os.path.dirname(data_yaml)
        test_img_dir = os.path.join(dataset_root, "test", "images")

        if not os.path.exists(test_img_dir):
            print(f"❌ 테스트 이미지 디렉토리를 찾을 수 없습니다: {test_img_dir}")
            return

        # 일부 테스트 이미지 가져오기
        img_files = list(Path(test_img_dir).glob("*.jpg")) + list(Path(test_img_dir).glob("*.png"))
        img_files = img_files[:5]  # 5개 이미지로 제한

        if not img_files:
            print("❌ 테스트 이미지를 찾을 수 없습니다")
            return

        print(f"{len(img_files)}개의 테스트 이미지에 대해 추론 실행 중...")

        # 추론 실행
        results = model.predict(
            source=str(test_img_dir),
            imgsz=640,
            conf=0.25,
            save=True,
            show_boxes=False,
            show_labels=False,
            show_conf=False,
            retina_masks=True,
            verbose=False
        )

        # 결과 표시
        save_dir = results[0].save_dir
        print(f"📁 결과 저장 위치: {save_dir}")

        # 일부 결과 표시
        result_files = list(Path(save_dir).glob("*.jpg")) + list(Path(save_dir).glob("*.png"))
        for i, result_file in enumerate(result_files[:3]):
            if result_file.exists():
                img = Image.open(result_file)
                plt.figure(figsize=(8, 8))
                plt.imshow(img)
                plt.title(f"결과 {i+1}: {result_file.name}")
                plt.axis('off')
                plt.show()

        print("✅ 추론 완료!")

    except Exception as e:
        print(f"❌ 추론 실패: {e}")

def main():
    """메인 함수 - 전체 파이프라인 실행.

    이 함수는 다음 단계를 순차적으로 실행합니다:
    1. 패키지 설치
    2. 환경 확인
    3. 데이터셋 선택 및 준비
    4. 모델 학습
    5. 모델 내보내기
    6. 추론 테스트

    Returns:
        None
    """
    print("🚀 YOLOv8 인스턴스 세그멘테이션 학습 스크립트")
    print("=" * 50)

    # 패키지 설치
    install_packages()

    # 환경 확인
    check_environment()

    # 데이터셋 선택 받기
    print("데이터셋 소스를 선택하세요:")
    print("1. Roboflow에서 다운로드")
    print("2. 로컬 데이터셋 사용")
    choice = input("선택을 입력하세요 (1 또는 2): ").strip()

    data_yaml = None

    if choice == "1":
        data_yaml = get_dataset_from_roboflow()
    elif choice == "2":
        local_path = input("로컬 data.yaml 경로를 입력하세요: ").strip()
        if os.path.exists(local_path):
            data_yaml = local_path
            print(f"✅ 로컬 데이터셋 사용: {data_yaml}")
        else:
            print(f"❌ 로컬 data.yaml을 찾을 수 없습니다: {local_path}")
            return
    else:
        print("❌ 잘못된 선택입니다")
        return

    if data_yaml:
        # data.yaml 경로 수정
        fix_data_yaml(data_yaml)

        # 샘플 시각화
        visualize_sample(data_yaml)

        # 모델 학습
        model_path = train_model(data_yaml)

        # ONNX로 내보내기
        if model_path:
            onnx_path = export_to_onnx(model_path)

            # 추론 실행
            run_inference(model_path, data_yaml)

    print("\n🎉 스크립트 실행 완료!")
    print("'runs/segment' 디렉토리에서 결과를 확인하세요.")

    print("\n" + "="*60)
    print("📖 추가 사용법:")
    print("1. 학습 파라미터 변경:")
    print("   - 에폭 수 조정: train_model() 함수 내 epochs 변수")
    print("   - 배치 크기 조정: batch 변수")
    print("   - 이미지 크기 조정: imgsz 변수")
    print("")
    print("2. 다른 모델 사용:")
    print("   - yolov8s-seg.pt (작은 모델)")
    print("   - yolov8m-seg.pt (중간 모델)")
    print("   - yolov8l-seg.pt (큰 모델)")
    print("")
    print("3. 문제 해결:")
    print("   - 메모리 부족: 배치 크기 줄이기")
    print("   - GPU 없음: CPU 모드로 자동 전환")
    print("   - 데이터 없음: data.yaml 경로 확인")
    print("="*60)

if __name__ == "__main__":
    main()
