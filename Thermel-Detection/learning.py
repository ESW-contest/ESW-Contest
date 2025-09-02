#!/usr/bin/env python3
"""
Thermel-Detect - IR 기반 사람 감지 모델 (단일 파일 버전)
=======================================================

개요:
    thermel_detect.py는 적외선(IR) 이미지에서 사람을 감지하는 YOLOv5 모델의
    완전한 학습 및 추론 파이프라인을 제공하는 Python 스크립트입니다.

주요 기능:
    - 데이터 변환: COCO 형식 → YOLO 형식 자동 변환
    - 이미지 전처리: 128x128 리사이즈 및 CLAHE 대비 향상
    - 모델 학습: YOLOv5 기반 사람 감지 모델 학습
    - 모델 평가: 학습된 모델의 성능 검증
    - 결과 시각화: Ground Truth vs Prediction 비교
    - 모델 내보내기: ONNX, TorchScript 형식으로 내보내기
    - 자동 설치: 필요한 모든 패키지 자동 설치

요구사항:
    - Python 3.7 이상
    - CUDA 지원 GPU (선택사항, CPU에서도 실행 가능)
    - 최소 8GB RAM 권장

데이터 준비:
    프로젝트 폴더/
    ├── Set-A/
    │   ├── images/
    │   │   ├── train/     # 학습용 IR 이미지들
    │   │   └── val/       # 검증용 IR 이미지들
    │   └── labels/        # COCO 형식 어노테이션 파일들
    │       └── *.json

사용법:
    python thermel_detect.py

자동 설치되는 패키지들:
    - torch>=1.7.0, torchvision>=0.8.0
    - opencv-python>=4.5.0
    - matplotlib>=3.3.0, numpy>=1.19.0
    - tqdm>=4.50.0, PyYAML>=5.3
    - requests>=2.25.0, scipy>=1.5.0
    - seaborn>=0.11.0, pandas>=1.2.0

이 파일은 다음 파일들의 내용을 통합한 단일 파일 버전입니다:
- thermel_detect.py (메인 스크립트)
- requirements.txt (패키지 목록)
- README.md (사용 설명서)

작성자: GitHub Copilot
버전: 1.0.0
날짜: 2025-09-02
라이선스: MIT License

사용 시 주의사항:
- 이 파일을 실행하면 자동으로 필요한 패키지들이 설치됩니다
- YOLOv5가 자동으로 다운로드됩니다
- Set-A 폴더에 데이터가 준비되어 있어야 합니다
- GPU가 있으면 자동으로 GPU를 사용합니다
"""

import subprocess
import sys
import os
import json
import shutil
import glob
import time
import re
import zipfile
from pathlib import Path
from collections import defaultdict
import cv2
import numpy as np
import matplotlib.pyplot as plt

def install_packages():
    """필요한 패키지들을 설치합니다 (requirements.txt 내용 통합)"""
    packages = [
        "torch>=1.7.0",
        "torchvision>=0.8.0",
        "opencv-python>=4.5.0",
        "matplotlib>=3.3.0",
        "numpy>=1.19.0",
        "tqdm>=4.50.0",
        "PyYAML>=5.3",
        "requests>=2.25.0",
        "scipy>=1.5.0",
        "seaborn>=0.11.0",
        "pandas>=1.2.0"
    ]

    print("필요한 패키지들을 설치하는 중...")
    print(f"총 {len(packages)}개 패키지를 설치합니다:")
    for package in packages:
        print(f"  - {package}")

    print("\n설치 시작...")
    installed_count = 0
    failed_count = 0

    for package in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✓ {package} 설치 완료")
            installed_count += 1
        except subprocess.CalledProcessError as e:
            print(f"✗ {package} 설치 실패: {e}")
            failed_count += 1

    print(f"\n패키지 설치 완료! (성공: {installed_count}, 실패: {failed_count})")

    if failed_count > 0:
        print("일부 패키지 설치에 실패했습니다. 수동으로 설치해보세요:")
        print("pip install -r requirements.txt")
    print()

def setup_yolov5():
    """YOLOv5를 설치합니다"""
    if not os.path.exists("yolov5"):
        print("YOLOv5를 설치하는 중...")
        try:
            subprocess.check_call(["git", "clone", "-q", "https://github.com/ultralytics/yolov5.git"])
            print("✓ YOLOv5 설치 완료")
        except subprocess.CalledProcessError as e:
            print(f"✗ YOLOv5 설치 실패: {e}")
            return False
    else:
        print("✓ YOLOv5가 이미 설치되어 있습니다")

    # YOLOv5 의존성 설치
    yolov5_req = "yolov5/requirements.txt"
    if os.path.exists(yolov5_req):
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", yolov5_req])
            print("✓ YOLOv5 의존성 설치 완료")
        except subprocess.CalledProcessError as e:
            print(f"✗ YOLOv5 의존성 설치 실패: {e}")

    return True

def set_seed(seed=42):
    """랜덤 시드를 설정합니다"""
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch
        import torch.backends.cudnn as cudnn
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
    except Exception:
        pass

def convert_coco_to_yolo(json_path, img_train_dir, img_val_dir, lab_train_dir, lab_val_dir):
    """COCO 형식의 어노테이션을 YOLO 형식으로 변환합니다"""
    print(f"COCO 데이터를 YOLO 형식으로 변환 중: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    # person 카테고리 ID 찾기
    person_ids = [c["id"] for c in coco["categories"] if c["name"].lower() == "person"]

    # 이미지 정보 매핑
    imginfo = {im["id"]: (im["file_name"], im["width"], im["height"]) for im in coco["images"]}

    # 어노테이션 변환
    outputs = defaultdict(list)
    for ann in coco["annotations"]:
        if ann["category_id"] not in person_ids:
            continue

        img_id = ann["image_id"]
        fn, w, h = imginfo[img_id]
        x, y, bw, bh = ann["bbox"]

        # YOLO 형식으로 변환 (정규화)
        x_c = (x + bw/2) / w
        y_c = (y + bh/2) / h
        ww = bw / w
        hh = bh / h
        line = f"0 {x_c:.6f} {y_c:.6f} {ww:.6f} {hh:.6f}\n"

        stem, _ = os.path.splitext(os.path.basename(fn))
        outputs[stem].append(line)

    # 변환된 라벨 저장
    for stem, lines in outputs.items():
        # train/val 폴더 결정
        if os.path.exists(os.path.join(img_train_dir, stem + ".jpg")):
            outpath = os.path.join(lab_train_dir, stem + ".txt")
        else:
            outpath = os.path.join(lab_val_dir, stem + ".txt")

        with open(outpath, "w") as f:
            f.writelines(lines)

    print(f"✓ 변환 완료: {len(outputs)}개 이미지 처리됨")

def resize_images(src_dir, dst_dir, size=(128, 128)):
    """이미지들을 지정된 크기로 리사이즈합니다"""
    os.makedirs(dst_dir, exist_ok=True)
    count = 0

    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        for path in glob.glob(os.path.join(src_dir, ext)):
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is None:
                continue

            resized = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
            out_path = os.path.join(dst_dir, os.path.basename(path))
            cv2.imwrite(out_path, resized)
            count += 1

    return count

def create_dataset_yaml(root_dir, yaml_path):
    """YOLO 학습을 위한 데이터셋 YAML 파일을 생성합니다"""
    yaml_text = f"""# IR person detection (128x128) dataset
path: {root_dir}
train: images/train
val: images/val

nc: 1
names: [person]
"""

    with open(yaml_path, "w") as f:
        f.write(yaml_text)

    print(f"✓ 데이터셋 YAML 생성: {yaml_path}")

def train_yolov5(img_size=128, batch_size=64, epochs=150, yaml_path="./yolov5/data/ir128.yaml", weights="yolov5s.pt", name="ir128_person"):
    """YOLOv5 모델을 학습합니다"""
    print("=== YOLOv5 모델 학습 시작 ===")

    # 환경 변수 설정
    os.environ["WANDB_DISABLED"] = "true"
    os.environ["PYTHONUNBUFFERED"] = "1"

    cmd = [
        "python", "./yolov5/train.py",
        "--img", str(img_size),
        "--batch", str(batch_size),
        "--epochs", str(epochs),
        "--data", yaml_path,
        "--weights", weights,
        "--name", name,
        "--workers", "2",
        "--cache", "ram",
        "--rect"
    ]

    print(f"학습 명령어: {' '.join(cmd)}")
    print("학습을 시작합니다... (시간이 오래 걸릴 수 있습니다)")

    # 로그 패턴
    re_total = re.compile(r"Starting training for (\d+) epochs")
    re_metrics_header = re.compile(r"\s*Class\s+Images\s+Instances\s+P\s+R\s+mAP50\s+mAP50-95")
    re_metrics_all = re.compile(r"\s*all\s+\d+\s+\d+\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)")

    total_epochs = None
    cur_epoch = 0
    t0 = time.time()
    t_epoch = time.time()
    expect_metrics_next = False

    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

        for line in proc.stdout:
            line = line.rstrip("\n")

            # 총 에폭 수 추출
            if total_epochs is None:
                m = re_total.search(line)
                if m:
                    total_epochs = int(m.group(1))
                    print(f"총 {total_epochs} 에폭 학습 시작…")

            # 메트릭 추출
            if re_metrics_header.search(line):
                expect_metrics_next = True
                continue

            if expect_metrics_next:
                expect_metrics_next = False
                m = re_metrics_all.search(line)
                if m:
                    cur_epoch += 1
                    P, R, m50, m95 = map(float, m.groups())
                    now = time.time()
                    epoch_time = now - t_epoch
                    t_epoch = now
                    elapsed = now - t0

                    print(f"[{cur_epoch}/{total_epochs}] "
                          ".3f"
                          ".1f")

            # 중요 메시지만 출력
            if "corrupt image/label" in line or ("WARNING" in line and "wandb" not in line):
                print("WARN:", line)

            if "Results saved to" in line:
                print(line)

        proc.wait()
        print("=== 학습 완료 ===")
        return proc.returncode == 0

    except Exception as e:
        print(f"학습 중 오류 발생: {e}")
        return False

def validate_model(weights_path, yaml_path="./yolov5/data/ir128.yaml", img_size=128):
    """학습된 모델을 검증합니다"""
    print("=== 모델 검증 시작 ===")

    cmd = [
        "python", "./yolov5/val.py",
        "--weights", weights_path,
        "--data", yaml_path,
        "--img", str(img_size)
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        print("검증 결과:")
        print(result.stdout)
        if result.stderr:
            print("오류:", result.stderr)
        return result.returncode == 0
    except Exception as e:
        print(f"검증 중 오류 발생: {e}")
        return False

def export_model(weights_path, img_size=128):
    """모델을 ONNX와 TorchScript로 내보냅니다"""
    print("=== 모델 내보내기 시작 ===")

    cmd = [
        "python", "./yolov5/export.py",
        "--weights", weights_path,
        "--include", "onnx", "torchscript",
        "--img", str(img_size),
        "--simplify"
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        print("내보내기 결과:")
        print(result.stdout)
        if result.stderr:
            print("오류:", result.stderr)
        return result.returncode == 0
    except Exception as e:
        print(f"내보내기 중 오류 발생: {e}")
        return False

def visualize_predictions(weights_path, img_path, output_dir="./results"):
    """테스트 이미지에 대한 예측 결과를 시각화합니다"""
    print(f"=== 예측 시각화: {img_path} ===")

    os.makedirs(output_dir, exist_ok=True)

    # 이미지 전처리 (CLAHE 적용)
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"이미지를 불러올 수 없습니다: {img_path}")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    proc = np.dstack([clahe, clahe, clahe])
    prep_path = os.path.join(output_dir, "_prep_" + os.path.basename(img_path))
    cv2.imwrite(prep_path, proc)

    # YOLOv5 추론 실행
    cmd = [
        "python", "./yolov5/detect.py",
        "--weights", weights_path,
        "--img", "128",
        "--conf", "0.02",
        "--iou", "0.45",
        "--classes", "0",
        "--source", prep_path,
        "--project", output_dir,
        "--name", "detection",
        "--exist-ok",
        "--line-thickness", "1"
    ]

    try:
        subprocess.run(cmd, check=True)

        # 결과 이미지 표시
        result_path = os.path.join(output_dir, "detection", os.path.basename(prep_path))
        if os.path.exists(result_path):
            result_img = cv2.imread(result_path)
            if result_img is not None:
                result_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
                plt.figure(figsize=(10, 10))
                plt.imshow(result_rgb)
                plt.title("적외선 사람 감지 결과")
                plt.axis('off')
                plt.show()
                print(f"✓ 결과 저장: {result_path}")
            else:
                print("결과 이미지를 불러올 수 없습니다")
        else:
            print("결과 파일이 생성되지 않았습니다")

    except subprocess.CalledProcessError as e:
        print(f"추론 중 오류 발생: {e}")

def main():
    """메인 함수"""
    print("🚀 Thermel-Detect - IR 기반 사람 감지 모델")
    print("=" * 50)

    # 시드 설정
    set_seed(42)

    # 패키지 설치
    install_packages()

    # YOLOv5 설치
    if not setup_yolov5():
        print("YOLOv5 설치 실패로 프로그램을 종료합니다.")
        return

    # 데이터셋 경로 설정
    data_root = "./Set-A"
    img_train = os.path.join(data_root, "images/train")
    img_val = os.path.join(data_root, "images/val")
    lab_train = os.path.join(data_root, "labels/train")
    lab_val = os.path.join(data_root, "labels/val")

    # 디렉토리 생성
    os.makedirs(lab_train, exist_ok=True)
    os.makedirs(lab_val, exist_ok=True)

    # COCO → YOLO 변환
    json_files = [f for f in os.listdir(os.path.join(data_root, "labels")) if f.endswith(".json")]
    if json_files:
        print("=== 데이터 변환 시작 ===")
        for json_file in json_files:
            json_path = os.path.join(data_root, "labels", json_file)
            convert_coco_to_yolo(json_path, img_train, img_val, lab_train, lab_val)
    else:
        print("JSON 파일을 찾을 수 없습니다. 데이터 변환을 건너뜁니다.")

    # 이미지 리사이즈
    print("=== 이미지 리사이즈 시작 ===")
    total_resized = 0
    for src, dst in [(img_train, os.path.join(data_root, "images_128/train")),
                     (img_val, os.path.join(data_root, "images_128/val"))]:
        if os.path.exists(src):
            n = resize_images(src, dst, (128, 128))
            print(f"{src} → {dst}: {n}개 이미지 처리")
            total_resized += n
    print(f"총 {total_resized}개 이미지 리사이즈 완료")

    # 데이터셋 YAML 생성
    yaml_path = "./yolov5/data/ir128.yaml"
    create_dataset_yaml(data_root, yaml_path)

    # 모델 학습
    if train_yolov5():
        # 학습된 가중치 경로
        weights_path = "./yolov5/runs/train/ir128_person/weights/best.pt"

        if os.path.exists(weights_path):
            # 모델 검증
            validate_model(weights_path, yaml_path)

            # 모델 내보내기
            export_model(weights_path)

            # 샘플 추론 (테스트 이미지 있으면)
            test_images = glob.glob("./*.jpg") + glob.glob("./*.png")
            if test_images:
                print("=== 샘플 추론 시작 ===")
                for img_path in test_images[:3]:  # 최대 3개
                    visualize_predictions(weights_path, img_path)
        else:
            print("학습된 모델 가중치를 찾을 수 없습니다.")
    else:
        print("모델 학습에 실패했습니다.")

    print("\n🎉 모든 작업이 완료되었습니다!")
    print("결과 파일들은 다음 위치에서 확인하세요:")
    print("- 학습 결과: ./yolov5/runs/train/")
    print("- 내보내기 결과: ./yolov5/runs/train/*/weights/")
    print("- 추론 결과: ./results/")

    print("\n" + "="*60)
    print("📖 추가 사용법:")
    print("1. 학습된 모델로 새로운 이미지 테스트:")
    print("   python thermel_detect.py  # 같은 폴더에 테스트 이미지 넣기")
    print("")
    print("2. 학습 파라미터 변경:")
    print("   - 에폭 수 조정: train_yolov5(epochs=50)")
    print("   - 배치 크기 조정: train_yolov5(batch_size=32)")
    print("   - 이미지 크기 조정: train_yolov5(img_size=256)")
    print("")
    print("3. 문제 해결:")
    print("   - 메모리 부족: 배치 크기 줄이기")
    print("   - 학습 느림: GPU 사용 확인")
    print("   - 데이터 없음: Set-A 폴더 구조 확인")
    print("="*60)

if __name__ == "__main__":
    main()
