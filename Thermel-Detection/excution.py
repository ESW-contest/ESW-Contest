#!/usr/bin/env python3
# thermal_stream_onnx.py
# RPi5 + MI48(HAT, I2C+SPI) 실시간 스트리밍 + YOLOv5/YOLOv8 ONNX (OpenCV DNN)
# === CE0(/dev/spidev0.0) + 수동 CS(BCM7) 버전 ===

import os, sys, time, argparse, logging
import numpy as np
import cv2 as cv
from smbus import SMBus
from spidev import SpiDev
from gpiozero import DigitalInputDevice, DigitalOutputDevice

from senxor.mi48 import MI48, DATA_READY
from senxor.utils import data_to_frame, cv_filter
from senxor.interfaces import SPI_Interface, I2C_Interface

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
log = logging.getLogger("thermal_onnx_dnn")

# ----------------- utils -----------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, type=str, help="YOLOv5/YOLOv8 ONNX 경로")
    p.add_argument("--imgsz", default=128, type=int, help="모델 입력 크기 (정사각)")
    p.add_argument("--conf",  default=0.35, type=float, help="confidence threshold")
    p.add_argument("--iou",   default=0.45, type=float, help="NMS IoU threshold")
    p.add_argument("--fps",   default=9.0,  type=float, help="카메라 FPS")
    p.add_argument("--show",  action="store_true", help="OpenCV 창 띄우기")
    p.add_argument("--save",  type=str, default="", help="이미지 저장 폴더(선택)")
    p.add_argument("--force-mono", action="store_true", help="강제로 1채널 입력")
    # 디스플레이 옵션 (DSI LCD 800x480 기본)
    p.add_argument("--win-w", type=int, default=800, help="표시 창 너비(기본 800)")
    p.add_argument("--win-h", type=int, default=480, help="표시 창 높이(기본 480)")
    p.add_argument("--fullscreen", action="store_true", help="전체화면 모드(DSI 800x480)")
    # SPI/I2C & 신호 안정화 옵션
    p.add_argument("--spi-bus", default=0, type=int, help="SPI bus (기본 0)")
    p.add_argument("--spi-dev", default=0, type=int, help="SPI dev (기본 0 → /dev/spidev0.0)")
    p.add_argument("--i2c-bus", default=1, type=int, help="I2C bus (기본 1)")
    p.add_argument("--i2c-addr", default=0x40, type=lambda x:int(x,0), help="I2C 주소 (기본 0x40)")
    p.add_argument("--speed", type=int, default=1_000_000, help="SPI 속도(Hz) 기본 1MHz")
    p.add_argument("--csdelay", type=float, default=0.0002, help="CS assert/deassert 지연(초)")
    return p.parse_args()

def letterbox(img, new_size=128, color=0):
    h, w = img.shape[:2]
    scale = min(new_size / h, new_size / w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = cv.resize(img, (nw, nh), interpolation=cv.INTER_LINEAR)
    top = (new_size - nh) // 2
    left = (new_size - nw) // 2
    if img.ndim == 2:
        canvas = np.full((new_size, new_size), color, dtype=resized.dtype)
        canvas[top:top+nh, left:left+nw] = resized
    else:
        canvas = np.full((new_size, new_size, img.shape[2]), color, dtype=resized.dtype)
        canvas[top:top+nh, left:left+nw, :] = resized
    return canvas, scale, left, top

def bbox_iou(box, boxes):
    x1 = np.maximum(box[0], boxes[:,0])
    y1 = np.maximum(box[1], boxes[:,1])
    x2 = np.minimum(box[2], boxes[:,2])
    y2 = np.minimum(box[3], boxes[:,3])
    inter = np.maximum(0, x2-x1) * np.maximum(0, y2-y1)
    area1 = (box[2]-box[0])*(box[3]-box[1])
    area2 = (boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1])
    union = area1 + area2 - inter + 1e-6
    return inter / union

def nms_boxes(boxes, scores, iou_thres):
    idxs = scores.argsort()[::-1]
    keep = []
    while idxs.size > 0:
        i = idxs[0]; keep.append(i)
        if idxs.size == 1: break
        ious = bbox_iou(boxes[i], boxes[idxs[1:]])
        idxs = idxs[1:][ious < iou_thres]
    return np.array(keep, dtype=np.int32)

# ---------- YOLO 후처리 (v5 / v8) ----------
def post_v8(out, conf_thres, iou_thres, input_size, scale, pad_left, pad_top, W, H):
    """
    YOLOv8 typical: (1, N, 4+1+C) or (1, C+5, N)
    """
    if isinstance(out, (list, tuple)): out = out[0]
    out = np.squeeze(out)
    if out.ndim == 3:
        # (C+5, N) → (N, C+5)
        if out.shape[0] < out.shape[1]:
            out = out.transpose(1,0,2) if out.ndim==3 else out
        out = out[0] if out.ndim==3 else out
    if out.ndim != 2 or out.shape[1] < 6: return []

    xywh = out[:, :4]
    obj  = 1 / (1 + np.exp(-out[:, 4]))  # sigmoid
    if out.shape[1] == 6:
        cls_prob = 1 / (1 + np.exp(-out[:, 5]))  # single class
    else:
        cls_logits = out[:, 5:]
        cls_prob = 1 / (1 + np.exp(-cls_logits))
        cls_prob = cls_prob.max(axis=1)
    conf = obj * cls_prob
    keep = conf >= conf_thres
    if not np.any(keep): return []
    xywh = xywh[keep]; conf = conf[keep]

    x,y,w,h = xywh[:,0], xywh[:,1], xywh[:,2], xywh[:,3]
    boxes = np.stack([x-w/2, y-h/2, x+w/2, y+h/2], axis=1)

    boxes[:, [0,2]] -= pad_left
    boxes[:, [1,3]] -= pad_top
    boxes /= scale
    boxes[:,0::2] = np.clip(boxes[:,0::2], 0, W-1)
    boxes[:,1::2] = np.clip(boxes[:,1::2], 0, H-1)

    if boxes.shape[0] > 0:
        keep_idx = nms_boxes(boxes, conf, iou_thres)
        boxes = boxes[keep_idx]; conf = conf[keep_idx]
        return [(tuple(map(int, b)), float(c)) for b,c in zip(boxes, conf)]
    return []

def post_v5(out, conf_thres, iou_thres, input_size, scale, pad_left, pad_top, W, H):
    """
    YOLOv5 ONNX (ultralytics/yolov5 export):
    보통 (1, N, 5+nc)  with sigmoid already applied.
    x,y,w,h는 입력해상도 기준 좌표(픽셀)로 나오는 경우가 대부분.
    """
    if isinstance(out, (list, tuple)): out = out[0]
    out = np.squeeze(out)
    if out.ndim == 3: out = out[0]
    if out.ndim != 2 or out.shape[1] < 6: return []

    xywh = out[:, :4]
    obj  = out[:, 4]  # already 0..1
    if out.shape[1] == 6:
        cls_prob = out[:, 5]  # single-class case
    else:
        cls_prob = out[:, 5:].max(axis=1)
    conf = obj * cls_prob
    keep = conf >= conf_thres
    if not np.any(keep): return []
    xywh = xywh[keep]; conf = conf[keep]

    x,y,w,h = xywh[:,0], xywh[:,1], xywh[:,2], xywh[:,3]
    # 보통 입력해상도 기준 픽셀값 → 레터박스 역보정
    boxes = np.stack([x-w/2, y-h/2, x+w/2, y+h/2], axis=1)

    boxes[:, [0,2]] -= pad_left
    boxes[:, [1,3]] -= pad_top
    boxes /= scale
    boxes[:,0::2] = np.clip(boxes[:,0::2], 0, W-1)
    boxes[:,1::2] = np.clip(boxes[:,1::2], 0, H-1)

    if boxes.shape[0] > 0:
        keep_idx = nms_boxes(boxes, conf, iou_thres)
        boxes = boxes[keep_idx]; conf = conf[keep_idx]
        return [(tuple(map(int, b)), float(c)) for b,c in zip(boxes, conf)]
    return []

def detect_expected_channels_from_net(net):
    # 이름이 알려진 첫 conv 레이어 시도
    try:
        for name in ['onnx_node!/model.0/conv/Conv', 'Conv_0', 'conv1', 'features.0.0']:
            lid = net.getLayerId(name)
            if lid != -1:
                lyr = net.getLayer(lid)
                if len(lyr.blobs)>0 and len(lyr.blobs[0].shape)==4:
                    return int(lyr.blobs[0].shape[1])  # in_ch
    except Exception:
        pass
    # fallback: 첫 4D weight 레이어
    try:
        for lid in range(1, net.getLayerCount()+1):
            lyr = net.getLayer(lid)
            if hasattr(lyr, "blobs") and len(lyr.blobs)>0:
                shp = getattr(lyr.blobs[0], "shape", None)
                if shp is not None and len(shp)==4:
                    return int(shp[1])
    except Exception:
        pass
    return 3

def guess_yolo_family(net, input_sz, in_ch):
    """
    더미 블롭으로 한 번 forward 해서 출력 shape로 YOLOv5/8 추정
    v8: 흔히 (1, 84, N) or (1, N, 84)  (C+5=6 for 1class)
    v5: 흔히 (1, N, 5+nc)
    """
    blob = np.zeros((1, in_ch, input_sz, input_sz), dtype=np.float32)
    net.setInput(blob)
    try:
        out = net.forward()
    except Exception:
        return "v5"  # 보수적으로
    arr = np.squeeze(out)
    shape = arr.shape
    # 휴리스틱
    if arr.ndim == 3 and (shape[0] in (5,6,85) or shape[1] in (5,6,85)):
        return "v8"
    if arr.ndim == 2 and shape[1] >= 6:
        return "v5"
    # fallback
    return "v5"

# ----------------- main -----------------
def main():
    args = parse_args()

    # DNN 로딩 (가능하면 FP32 ONNX 사용: export 시 --half 빼주세요)
    if not os.path.isfile(args.model):
        raise FileNotFoundError(f"Model not found: {args.model}")
    net = cv.dnn.readNet(args.model)
    try:
        net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
    except Exception:
        pass

    input_sz = args.imgsz
    expected_ch = detect_expected_channels_from_net(net)
    family = guess_yolo_family(net, input_sz, expected_ch)
    log.info(f"Model in_ch={expected_ch}, family={family}")

    # 카메라(HAT) — CE0(/dev/spidev0.0) + 수동 CS(BCM7)
    I2C_BUS, I2C_ADDR = args.i2c_bus, args.i2c_addr
    SPI_BUS, SPI_DEV  = args.spi_bus, args.spi_dev   # 기대: 0,0
    SPI_SPEED_HZ      = int(args.speed)
    CS_DELAY          = float(args.csdelay)
    DATA_READY_BCM, RESET_BCM, CS_BCM = 24, 23, 7

    i2c = I2C_Interface(SMBus(I2C_BUS), I2C_ADDR)

    spi_dev = SpiDev()
    dev_path = f"/dev/spidev{SPI_BUS}.{SPI_DEV}"
    if not os.path.exists(dev_path):
        raise FileNotFoundError(f"SPI device not found: {dev_path}")
    spi_dev.open(SPI_BUS, SPI_DEV)   # /dev/spidev0.0
    spi_dev.mode = 0
    spi_dev.max_speed_hz = SPI_SPEED_HZ
    spi_dev.bits_per_word = 8
    try: spi_dev.cshigh = True   # 보드에 따라 True/False 변경해보며 최적 확인
    except Exception: pass
    spi = SPI_Interface(spi_dev, xfer_size=160)

    drdy   = DigitalInputDevice(DATA_READY_BCM, pull_up=False)
    cs_n   = DigitalOutputDevice(CS_BCM, active_high=False, initial_value=True) # High=deassert
    resetn = DigitalOutputDevice(RESET_BCM, active_high=False, initial_value=True)

    class MI48_reset:
        def __init__(self, pin, assert_seconds=0.000035, deassert_seconds=0.050):
            self.pin = pin; self.t_on = assert_seconds; self.t_off = deassert_seconds
        def __call__(self):
            print("Resetting the MI48...")
            self.pin.on();  time.sleep(self.t_on)
            self.pin.off(); time.sleep(self.t_off)
            print("Done.")

    mi48 = MI48([i2c, spi], data_ready=drdy, reset_handler=MI48_reset(pin=resetn))
    info = mi48.get_camera_info(); log.info(f"Camera: {info}")

    mi48.set_fps(args.fps)
    try:
        if int(mi48.fw_version[0]) >= 2:
            mi48.enable_filter(f1=True, f2=True, f3=False)
            mi48.set_offset_corr(0.0)
    except Exception:
        pass

    if args.save: os.makedirs(args.save, exist_ok=True)

    mi48.start(stream=True, with_header=True)

    # === 디스플레이 설정: LCD 800x480 기본 ===
    # show가 켜진 경우에만 창 생성/설정 (1회)
    if args.show:
        cv.namedWindow("THERMAL+ONNX", cv.WINDOW_NORMAL)
        try:
            if args.fullscreen:
                # 전체화면: DSI가 800x480이면 꽉 차게
                cv.setWindowProperty("THERMAL+ONNX", cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
            else:
                cv.resizeWindow("THERMAL+ONNX", args.win_w, args.win_h)
                # 좌상단으로 이동(DSI 단독 사용 시 정확히 화면에 맞춤)
                try:
                    cv.moveWindow("THERMAL+ONNX", 0, 0)
                except Exception:
                    pass
        except Exception:
            # 일부 환경(특히 Wayland)에서 setWindowProperty/resizeWindow가 제한될 수 있음
            pass

    first_log = True
    try:
        while True:
            drdy.wait_for_active()

            # ----- 수동 CS 토글(BCM7) -----
            cs_n.on();  time.sleep(CS_DELAY)     # assert(L)
            data, header = mi48.read()
            time.sleep(CS_DELAY); cs_n.off()     # deassert(H)

            if data is None:
                log.error("NONE data received instead of GFRA")
                continue

            # (H,W) float-ish → 0..255 uint8 (+가벼운 필터)
            frame = data_to_frame(data, mi48.fpa_shape)
            img8  = cv.normalize(frame.astype(np.float32), None, 0, 255,
                                 norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
            img8  = cv_filter(img8, parameters={'blur_ks':3},
                              use_median=False, use_bilat=True, use_nlm=False)
            H, W = img8.shape[:2]

            # --- 입력 채널 보정 ---
            if expected_ch == 3:
                img_for_net = cv.cvtColor(img8, cv.COLOR_GRAY2RGB)
            else:
                img_for_net = img8

            # 레터박스 후 blob
            inp, scale, px, py = letterbox(img_for_net, new_size=input_sz, color=0)
            if expected_ch == 3 and inp.ndim == 2:  # 안전장치
                inp = cv.cvtColor(inp, cv.COLOR_GRAY2RGB)
            blob = cv.dnn.blobFromImage(inp, scalefactor=1/255.0,
                                        size=(input_sz, input_sz),
                                        mean=(0,0,0), swapRB=False, crop=False)
            net.setInput(blob)

            # 추론 & 후처리
            out  = net.forward()
            # 첫 프레임에 출력 shape / conf 최대치 로그
            if first_log:
                arr = np.squeeze(out)
                try:
                    max_conf = float(arr[...,4].max()) if arr.ndim>=2 and arr.shape[-1]>=5 else float(arr.max())
                except Exception:
                    max_conf = float(arr.max())
                log.info(f"ONNX output shape: {arr.shape}, max_raw={max_conf:.4f}")
                first_log = False

            if family == "v8":
                dets = post_v8(out, args.conf, args.iou, input_sz, scale, px, py, W, H)
            else:
                dets = post_v5(out, args.conf, args.iou, input_sz, scale, px, py, W, H)

            # 시각화 (FPS 텍스트 없음)
            vis = cv.applyColorMap(img8, cv.COLORMAP_INFERNO)
            for (x1,y1,x2,y2), score in dets:
                cv.rectangle(vis, (x1,y1), (x2,y2), (0,255,0), 2)
                cv.putText(vis, f"person {score:.2f}", (x1, max(0,y1-6)),
                           cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

            if args.show:
                # LCD 해상도에 맞춰 리사이즈 후 표시
                target_w, target_h = args.win_w, args.win_h
                if args.fullscreen:
                    # 전체화면인 경우에도 안전하게 리사이즈
                    disp = cv.resize(vis, (target_w, target_h), interpolation=cv.INTER_CUBIC)
                else:
                    disp = cv.resize(vis, (target_w, target_h), interpolation=cv.INTER_CUBIC)
                cv.imshow("THERMAL+ONNX", disp)
                if (cv.waitKey(1) & 0xFF) == ord('q'):
                    break

            if args.save:
                ts = int(time.time()*1000)
                cv.imwrite(os.path.join(args.save, f"thermal_{ts}.png"), vis)

    finally:
        try: mi48.stop(stop_timeout=0.5)
        except Exception: pass
        cv.destroyAllWindows()

if __name__ == "__main__":
    main()
