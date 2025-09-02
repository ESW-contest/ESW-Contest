#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Raspberry Pi 5 / DSI LCD 800x480 / 단일 창 (풀스크린)
# 모드:
#   0) SEG+DOA (기본 시작): PiCamera2 + YOLOv8 Seg(사람 윤곽) + UDP DOA 화살표
#   1) THERMAL: MI48 + YOLO(ONNX, OpenCV DNN) - 사람 박스 (초록) / 텍스트 없음
#
# 요구사항 반영:
#   - Picamera2는 XRGB8888 -> BGRA2BGR 변환(파란 톤 문제 해결: 단일 스크립트와 동일)
#   - Thermal DNN 경로는 thermal_stream_onnx.py와 동일(출력 확실)
#   - 파라미터 네임스페이스 완전 분리(seg_* vs th_*)
#   - THERMAL 모드에서는 "THERMAL ..." 텍스트 일절 표시하지 않음
#   - SEG 모드 HUD/상단 텍스트는 기존처럼 표시

import os, time, json, re, socket, argparse
import numpy as np
import cv2 as cv
import onnxruntime as ort

# MI48 / GPIO / I2C / SPI
from smbus import SMBus
from spidev import SpiDev
from gpiozero import DigitalInputDevice, DigitalOutputDevice
from senxor.mi48 import MI48
from senxor.utils import data_to_frame, cv_filter
from senxor.interfaces import SPI_Interface, I2C_Interface

# ===== Picamera2 lifecycle (singleton) =======================================
from picamera2 import Picamera2
RGB_CAM = None

def open_rgb_camera(out_w, out_h):
    """싱글톤 Picamera2 - configure/start (단일 스크립트와 동일: XRGB8888)"""
    global RGB_CAM
    if RGB_CAM is None:
        RGB_CAM = Picamera2()
    try:
        RGB_CAM.stop()
    except Exception:
        pass
    cfg = RGB_CAM.create_preview_configuration(main={"format": "XRGB8888", "size": (out_w, out_h)})
    RGB_CAM.configure(cfg)
    RGB_CAM.start()
    return RGB_CAM

def stop_rgb_camera():
    global RGB_CAM
    if RGB_CAM is not None:
        try:
            RGB_CAM.stop()
        except Exception:
            pass

def close_rgb_camera():
    global RGB_CAM
    if RGB_CAM is not None:
        try:
            RGB_CAM.stop()
        except Exception:
            pass
        try:
            RGB_CAM.close()
        except Exception:
            pass
        RGB_CAM = None

# ===== DOA(음성 각도) 파트 ====================================================
PI_LISTEN_PORT_DEFAULT = 5005
ANGLE_REGEX = re.compile(r"ANGLE\s*:\s*(-?\d+(\.\d+)?)")

def parse_angle_line(b: bytes):
    s = b.decode(errors='ignore').strip()
    m = ANGLE_REGEX.search(s)
    if m:
        return float(m.group(1))
    if s.startswith("{") and '"angle"' in s:
        try:
            return float(json.loads(s).get("angle"))
        except Exception:
            pass
    return None

def classify_zone(angle_deg, yaw_offset, th_low, th_high):
    """side('L'|'R'), zone('BOTTOM'|'MID'|'TOP')"""
    a = angle_deg - yaw_offset
    side = 'L' if a < 0 else 'R'
    aa = abs(a)
    if aa < th_low:
        zone = 'BOTTOM'
    elif aa >= th_high:
        zone = 'TOP'
    else:
        zone = 'MID'
    return side, zone

def draw_hud(img, yaw, th_low, th_high, show_help, last_angle):
    lines = [
        f"YAW_OFFSET = {yaw:+.1f} deg",
        f"TH_LOW = {th_low:.1f}   TH_HIGH = {th_high:.1f}",
    ]
    if last_angle is not None:
        s, z = classify_zone(last_angle, yaw, th_low, th_high)
        lines.append(f"angle={last_angle:+.1f}°, side={s}, zone={z}")
    if show_help:
        lines += [
            "Keys: ←/→ yaw,  ↑/↓ TH_LOW,  [ / ] TH_HIGH",
            "Click top-right icon to switch,  q=quit,  h=help",
        ]
    pad = 8
    fs, th = 0.6, 1
    sizes = [cv.getTextSize(t, cv.FONT_HERSHEY_SIMPLEX, fs, th)[0] for t in lines]
    w = max(s[0] for s in sizes) + pad*2
    h = sum(s[1] for s in sizes) + pad*2 + (len(sizes)-1)*4
    x, y = 10, 10
    roi = img[y:y+h, x:x+w]
    overlay = roi.copy()
    cv.rectangle(overlay, (0,0), (w-1,h-1), (0,0,0), -1)
    cv.addWeighted(overlay, 0.45, roi, 0.55, 0, dst=roi)
    yy = y + pad + sizes[0][1]
    for t in lines:
        cv.putText(img, t, (x+pad, yy), cv.FONT_HERSHEY_SIMPLEX, fs, (255,255,255), th, cv.LINE_AA)
        size = cv.getTextSize(t, cv.FONT_HERSHEY_SIMPLEX, fs, th)[0]
        yy += size[1] + 4

def _rotate_pts(pts, angle_deg):
    theta = np.deg2rad(angle_deg)
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]], np.float32)
    return (R @ pts.T).T

def draw_filled_arrow(canvas, side, zone, arrow_scale, arrow_border, fill_bgr=(0,0,255), edge_bgr=(0,0,0)):
    """빨간색 채움 삼각 화살표 + 두꺼운 외곽선, 디스플레이 버퍼 기준 좌표에 그림"""
    h, w = canvas.shape[:2]
    margin = int(min(w, h) * 0.04)
    size   = max(16, int(min(w, h) * float(arrow_scale)))  # --arrow-scale 비율

    # 배치 & 회전
    if zone == 'TOP':
        y = margin
        x = w - size - margin if side=='R' else margin
        ang = -45 if side=='R' else 45
    elif zone == 'BOTTOM':
        y = h - size - margin
        x = w - size - margin if side=='R' else margin
        ang = -135 if side=='R' else 135
    else:
        y = (h - size)//2
        x = w - size - margin if side=='R' else margin
        ang = -90 if side=='R' else 90

    s = float(size)
    base = np.array([[0.0, -0.42*s],
                     [ 0.76*s/2, 0.50*s],
                     [-0.76*s/2, 0.50*s]], np.float32)
    rot = _rotate_pts(base, ang)
    cx, cy = x + s/2, y + s/2
    pts = np.stack([rot[:,0] + cx, rot[:,1] + cy], axis=1).astype(np.int32).reshape(-1,1,2)

    cv.fillPoly(canvas, [pts], fill_bgr)
    cv.polylines(canvas, [pts], True, edge_bgr, thickness=int(arrow_border), lineType=cv.LINE_AA)

# ===== 전환 아이콘(그래픽, 텍스트 없음) =======================================
def draw_switch_icon_on_canvas(canvas):
    """우상단 전환 아이콘(그래픽만). 클릭 영역 rect 반환"""
    h, w = canvas.shape[:2]
    size = max(56, min(120, int(min(w, h)*0.12)))
    pad = 12
    cx = w - pad - size//2
    cy = pad + size//2

    cv.circle(canvas, (cx, cy), size//2, (0,0,0), -1)
    cv.circle(canvas, (cx, cy), size//2-2, (255,255,255), 2)

    r = int(size*0.34)
    cv.ellipse(canvas, (cx,cy), (r,r), 0, -35, 250, (80,200,255), 4)
    a1 = np.deg2rad(40)
    p1 = (int(cx + r*np.cos(a1)), int(cy - r*np.sin(a1)))
    p2 = (int(cx + r*np.cos(a1+1.35)), int(cy - r*np.sin(a1+1.35)))
    cv.arrowedLine(canvas, p2, p1, (80,200,255), 4, tipLength=0.45)

    x1 = w - pad - size
    y1 = pad
    x2 = w - pad
    y2 = pad + size
    return (x1,y1,x2,y2)

# ===== YOLOv8 Seg(사람만) =====================================================
def _sigmoid(x): return 1.0 / (1.0 + np.exp(-x))

def _nms(boxes, scores, iou_thr=0.4, top_k=100):
    idxs = scores.argsort()[::-1][:top_k]
    keep = []
    while idxs.size > 0:
        i = idxs[0]; keep.append(i)
        if idxs.size == 1: break
        rest = idxs[1:]
        xx1 = np.maximum(boxes[i,0], boxes[rest,0])
        yy1 = np.maximum(boxes[i,1], boxes[rest,1])
        xx2 = np.minimum(boxes[i,2], boxes[rest,2])
        yy2 = np.minimum(boxes[i,3], boxes[rest,3])
        w = np.maximum(0.0, xx2-xx1)
        h = np.maximum(0.0, yy2-yy1)
        inter = w*h
        ai = (boxes[i,2]-boxes[i,0])*(boxes[i,3]-boxes[i,1])
        ar = (boxes[rest,2]-boxes[rest,0])*(boxes[rest,3]-boxes[rest,1])
        iou = inter / (ai + ar - inter + 1e-6)
        idxs = rest[iou <= iou_thr]
    return np.array(keep, np.int32)

def _letterbox(im, new_shape=(640,640), color=(114,114,114), scaleup=False):
    h, w = im.shape[:2]
    r = min(new_shape[0]/h, new_shape[1]/w)
    if not scaleup: r = min(r, 1.0)
    new_unpad = (int(round(w*r)), int(round(h*r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2; dh /= 2
    if (w, h) != new_unpad:
        im = cv.resize(im, new_unpad, interpolation=cv.INTER_LINEAR)
    top, bottom = int(round(dh-0.1)), int(round(dh+0.1))
    left, right = int(round(dw-0.1)), int(round(dw+0.1))
    im = cv.copyMakeBorder(im, top, bottom, left, right, cv.BORDER_CONSTANT, value=color)
    return im, r, (dw, dh)

class SegEngine:
    """YOLOv8 Seg (사람 클래스만) 윤곽선 + EMA + 프레임스킵"""
    def __init__(self, onnx_path, conf_th=0.65, iou_th=0.4, mask_th=0.55,
                 min_area_ratio=0.0001, k_open=1, k_close=2, gauss_k=1,
                 infer_every=2, ema_alpha=0.6, fps_alpha=0.9):
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        so.intra_op_num_threads = 4
        so.inter_op_num_threads = 1
        self.session = ort.InferenceSession(onnx_path, sess_options=so, providers=["CPUExecutionProvider"])
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.in_h = inp.shape[2] if isinstance(inp.shape[2], int) and inp.shape[2] > 0 else 640
        self.in_w = inp.shape[3] if isinstance(inp.shape[3], int) and inp.shape[3] > 0 else 640

        self.CONF_TH = conf_th; self.IOU_TH = iou_th; self.MASK_TH = mask_th
        self.MIN_AREA_RATIO = min_area_ratio; self.K_OPEN = k_open; self.K_CLOSE = k_close; self.GAUSS_K = gauss_k
        self.INFER_EVERY = infer_every; self.EMA_ALPHA = ema_alpha; self.FPS_ALPHA = fps_alpha

        self.prev_scene_prob = None
        self.frame_id = 0
        self.prev_t = time.time()
        self.fps_ema = None

        self.OUTLINE_THICK = 4
        self.OUTLINE_COLOR = (255, 255, 0)

    def step(self, frame_bgr):
        # FPS 계산
        now = time.time(); dt = now - self.prev_t; self.prev_t = now
        inst_fps = (1.0/dt) if dt > 0 else 0.0
        self.fps_ema = inst_fps if self.fps_ema is None else (self.FPS_ALPHA*self.fps_ema + (1.0-self.FPS_ALPHA)*inst_fps)

        Hf, Wf = frame_bgr.shape[:2]
        do_infer = (self.frame_id % self.INFER_EVERY == 0); self.frame_id += 1

        lb_img, r, (dw, dh) = _letterbox(frame_bgr, (self.in_h, self.in_w), scaleup=False)
        img = cv.cvtColor(lb_img, cv.COLOR_BGR2RGB).astype(np.float32)/255.0
        img = np.transpose(img, (2,0,1))[None,...]

        vis = frame_bgr.copy()

        if do_infer:
            preds, protos = self.session.run(None, {self.input_name: img})
            preds = np.squeeze(preds, 0).transpose(1, 0)

            if preds.shape[1] == 37:
                boxes_cxcywh = preds[:, 0:4]
                cls_logits   = preds[:, 4:5]    # person score
                mask_coefs   = preds[:, 5:37]
                if np.max(_sigmoid(cls_logits)) < 0.01 and np.max(_sigmoid(preds[:, 36:37])) > np.max(_sigmoid(cls_logits)):
                    mask_coefs = preds[:, 4:36]; cls_logits = preds[:, 36:37]
            else:
                scene_prob = np.zeros((Hf, Wf), np.float32)
                smooth_prob = scene_prob if self.prev_scene_prob is None else self.EMA_ALPHA*self.prev_scene_prob + (1.0-self.EMA_ALPHA)*scene_prob
                self.prev_scene_prob = smooth_prob
                return vis, [], self.fps_ema, do_infer

            scores = _sigmoid(cls_logits).squeeze(-1)
            cx, cy, w, h = [boxes_cxcywh[:, i] for i in range(4)]
            x1 = cx - w/2; y1 = cy - h/2; x2 = cx + w/2; y2 = cy + h/2
            boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

            keep0 = scores > self.CONF_TH
            boxes_xyxy = boxes_xyxy[keep0]; scores_k = scores[keep0]; mask_coefs = mask_coefs[keep0]

            if boxes_xyxy.shape[0] > 0:
                keep = _nms(boxes_xyxy, scores_k, iou_thr=self.IOU_TH, top_k=100)
                boxes_xyxy = boxes_xyxy[keep]; scores_k = scores_k[keep]; mask_coefs = mask_coefs[keep]

                proto = np.squeeze(protos, 0).reshape(32, -1).astype(np.float32)  # (32, 25600)
                M = (mask_coefs.astype(np.float32) @ proto)
                M = _sigmoid(M).reshape(-1, 160, 160)

                scene_prob_small = np.max(M, axis=0)  # (160,160)
                scene_prob_pad = cv.resize(scene_prob_small, (self.in_w, self.in_h), interpolation=cv.INTER_LINEAR)
                top = int(dh); left = int(dw); bottom = int(self.in_h - dh); right = int(self.in_w - dw)
                scene_prob_pad = scene_prob_pad[top:bottom, left:right]
                if scene_prob_pad.size == 0:
                    scene_prob = np.zeros((Hf, Wf), np.float32)
                else:
                    scene_prob = cv.resize(scene_prob_pad, (Wf, Hf), interpolation=cv.INTER_LINEAR)
            else:
                scene_prob = np.zeros((Hf, Wf), np.float32)

            smooth_prob = scene_prob if self.prev_scene_prob is None else self.EMA_ALPHA*self.prev_scene_prob + (1.0-self.EMA_ALPHA)*scene_prob
            self.prev_scene_prob = smooth_prob
        else:
            smooth_prob = self.prev_scene_prob if self.prev_scene_prob is not None else np.zeros((Hf, Wf), np.float32)

        m_bin = (smooth_prob > self.MASK_TH).astype(np.uint8)*255
        if self.K_OPEN > 1:
            m_bin = cv.morphologyEx(m_bin, cv.MORPH_OPEN, np.ones((self.K_OPEN,self.K_OPEN), np.uint8))
        if self.K_CLOSE > 1:
            m_bin = cv.morphologyEx(m_bin, cv.MORPH_CLOSE, np.ones((self.K_CLOSE,self.K_CLOSE), np.uint8))

        num, labels, stats, _ = cv.connectedComponentsWithStats((m_bin>0).astype(np.uint8), connectivity=8)
        keep_mask = np.zeros_like(m_bin)
        min_area = int(self.MIN_AREA_RATIO * Hf * Wf)
        for i in range(1, num):
            if stats[i, cv.CC_STAT_AREA] >= min_area:
                keep_mask[labels == i] = 255
        m_bin = keep_mask

        if self.GAUSS_K >= 3 and self.GAUSS_K % 2 == 1:
            m_bin = cv.GaussianBlur(m_bin, (self.GAUSS_K, self.GAUSS_K), 0)

        contours, _ = cv.findContours(m_bin, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        contours = [c for c in contours if cv.contourArea(c) >= min_area]

        if contours:
            simplified = []
            for c in contours:
                eps = 0.006 * cv.arcLength(c, True)
                approx = cv.approxPolyDP(c, eps, True)
                simplified.append(approx)
            cv.drawContours(vis, simplified, -1, self.OUTLINE_COLOR, self.OUTLINE_THICK)
        else:
            simplified = []

        return vis, simplified, self.fps_ema, do_infer

# ===== Thermal ONNX(DNN) 유틸 (단일 스크립트와 동일 경로) ======================
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
    x1 = np.maximum(box[0], boxes[:,0]); y1 = np.maximum(box[1], boxes[:,1])
    x2 = np.minimum(box[2], boxes[:,2]); y2 = np.minimum(box[3], boxes[:,3])
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

def post_v8(out, conf_thres, iou_thres, input_size, scale, pad_left, pad_top, W, H):
    if isinstance(out, (list, tuple)): out = out[0]
    out = np.squeeze(out)
    if out.ndim == 3:
        if out.shape[1] < out.shape[2]:
            out = out.transpose(0,2,1)
        out = out[0]
    if out.ndim != 2 or out.shape[1] < 6: return []
    xywh = out[:, :4]
    obj  = 1 / (1 + np.exp(-out[:, 4]))
    if out.shape[1] == 6:
        cls_prob = 1 / (1 + np.exp(-out[:, 5]))
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
    boxes[:, [0,2]] -= pad_left; boxes[:, [1,3]] -= pad_top
    boxes /= scale
    boxes[:,0::2] = np.clip(boxes[:,0::2], 0, W-1)
    boxes[:,1::2] = np.clip(boxes[:,1::2], 0, H-1)
    if boxes.shape[0] > 0:
        keep_idx = nms_boxes(boxes, conf, iou_thres)
        boxes = boxes[keep_idx]; conf = conf[keep_idx]
        return [(tuple(map(int, b)), float(c)) for b,c in zip(boxes, conf)]
    return []

def post_v5(out, conf_thres, iou_thres, input_size, scale, pad_left, pad_top, W, H):
    if isinstance(out, (list, tuple)): out = out[0]
    out = np.squeeze(out)
    if out.ndim == 3: out = out[0]
    if out.ndim != 2 or out.shape[1] < 6: return []
    xywh = out[:, :4]
    obj  = out[:, 4]
    if out.shape[1] == 6:
        cls_prob = out[:, 5]
    else:
        cls_prob = out[:, 5:].max(axis=1)
    conf = obj * cls_prob
    keep = conf >= conf_thres
    if not np.any(keep): return []
    xywh = xywh[keep]; conf = conf[keep]
    x,y,w,h = xywh[:,0], xywh[:,1], xywh[:,2], xywh[:,3]
    boxes = np.stack([x-w/2, y-h/2, x+w/2, y+h/2], axis=1)
    boxes[:, [0,2]] -= pad_left; boxes[:, [1,3]] -= pad_top
    boxes /= scale
    boxes[:,0::2] = np.clip(boxes[:,0::2], 0, W-1)
    boxes[:,1::2] = np.clip(boxes[:,1::2], 0, H-1)
    if boxes.shape[0] > 0:
        keep_idx = nms_boxes(boxes, conf, iou_thres)
        boxes = boxes[keep_idx]; conf = conf[keep_idx]
        return [(tuple(map(int, b)), float(c)) for b,c in zip(boxes, conf)]
    return []

def detect_expected_channels_from_net(net):
    try:
        for name in ['onnx_node!/model.0/conv/Conv', 'Conv_0', 'conv1', 'features.0.0']:
            lid = net.getLayerId(name)
            if lid != -1:
                lyr = net.getLayer(lid)
                if len(lyr.blobs)>0 and len(lyr.blobs[0].shape)==4:
                    return int(lyr.blobs[0].shape[1])
    except Exception:
        pass
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
    blob = np.zeros((1, in_ch, input_sz, input_sz), dtype=np.float32)
    net.setInput(blob)
    try:
        out = net.forward()
    except Exception:
        return "v5"
    arr = np.squeeze(out)
    shape = arr.shape
    if arr.ndim == 3 and (shape[0] in (5,6,85) or shape[1] in (5,6,85)):
        return "v8"
    if arr.ndim == 2 and shape[1] >= 6:
        return "v5"
    return "v5"

# ===== Thermal 장치 라이프사이클 =============================================
class ThermalHW:
    def __init__(self, args):
        self.args = args
        self.i2c = None
        self.spi_dev = None
        self.spi = None
        self.drdy = None
        self.cs_n = None
        self.resetn = None
        self.mi48 = None
        self.CS_DELAY = float(args.csdelay)

    def start(self):
        I2C_BUS, I2C_ADDR = self.args.i2c_bus, self.args.i2c_addr
        SPI_BUS, SPI_DEV  = self.args.spi_bus, self.args.spi_dev
        SPI_SPEED_HZ      = int(self.args.speed)

        self.i2c = I2C_Interface(SMBus(I2C_BUS), I2C_ADDR)

        dev_path = f"/dev/spidev{SPI_BUS}.{SPI_DEV}"
        if not os.path.exists(dev_path):
            raise FileNotFoundError(f"SPI device not found: {dev_path}")
        self.spi_dev = SpiDev()
        self.spi_dev.open(SPI_BUS, SPI_DEV)
        self.spi_dev.mode = 0
        self.spi_dev.max_speed_hz = SPI_SPEED_HZ
        self.spi_dev.bits_per_word = 8
        try: self.spi_dev.cshigh = True
        except Exception: pass
        self.spi = SPI_Interface(self.spi_dev, xfer_size=160)

        self.drdy   = DigitalInputDevice(24, pull_up=False)
        self.cs_n   = DigitalOutputDevice(7, active_high=False, initial_value=True) # High=deassert
        self.resetn = DigitalOutputDevice(23, active_high=False, initial_value=True)

        class MI48_reset:
            def __init__(self, pin, assert_seconds=0.000035, deassert_seconds=0.050):
                self.pin = pin; self.t_on = assert_seconds; self.t_off = deassert_seconds
            def __call__(self):
                self.pin.on();  time.sleep(self.t_on)
                self.pin.off(); time.sleep(self.t_off)

        self.mi48 = MI48([self.i2c, self.spi], data_ready=self.drdy, reset_handler=MI48_reset(pin=self.resetn))

        try:
            self.mi48.set_fps(self.args.mi_fps)
        except Exception:
            pass
        try:
            if int(self.mi48.fw_version[0]) >= 2:
                self.mi48.enable_filter(f1=True, f2=True, f3=False)
                self.mi48.set_offset_corr(0.0)
        except Exception:
            pass

        self.mi48.start(stream=True, with_header=True)

    def read_frame(self):
        try:
            self.cs_n.on();  time.sleep(self.CS_DELAY)
            data, header = self.mi48.read()
            time.sleep(self.CS_DELAY); self.cs_n.off()
            return data, header
        except Exception:
            try: self.cs_n.off()
            except Exception: pass
            return None, None

    def stop(self):
        if self.mi48:
            try: self.mi48.stop(stop_timeout=0.5)
            except Exception: pass
            self.mi48 = None
        try:
            if self.resetn: self.resetn.close()
        except Exception: pass
        try:
            if self.cs_n: self.cs_n.close()
        except Exception: pass
        try:
            if self.drdy: self.drdy.close()
        except Exception: pass
        try:
            if self.spi_dev: self.spi_dev.close()
        except Exception: pass
        try:
            if self.i2c and hasattr(self.i2c, "bus") and self.i2c.bus:
                self.i2c.bus.close()
        except Exception: pass

# ===== 공용 UI: 클릭 상태 =====================================================
class ClickState:
    def __init__(self):
        self.switch_requested = False
        self.icon_rect = (0,0,0,0)
    def set_icon_rect(self, rect):
        self.icon_rect = rect
    def handle(self, event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONUP:
            x1,y1,x2,y2 = self.icon_rect
            if x1 <= x <= x2 and y1 <= y <= y2:
                self.switch_requested = True

# ===== 모드 루프들 ============================================================
def run_seg_with_doa(args, sock, doa_state, clicker, seg_engine):
    """Seg(사람 윤곽)+DOA: 표시/클릭은 disp 좌표계 기준"""
    cam = open_rgb_camera(args.rgb_out_w, args.rgb_out_h)
    win = args.win_name
    t0 = time.time(); frames = 0

    while True:
        # UDP 수신(논블로킹)
        try:
            data = sock.recv(4096)
            a = parse_angle_line(data)
            if a is not None:
                doa_state["last_angle"] = a
                doa_state["last_rx_ms"] = int(time.time()*1000)
        except BlockingIOError:
            pass

        # 카메라 → BGRA → BGR (단일 세그 스크립트 방식 유지)
        frame_bgra = cam.capture_array()  # (H, W, 4)
        vis_in = cv.cvtColor(frame_bgra, cv.COLOR_BGRA2BGR)

        # Seg 윤곽선
        seg_vis, contours, fps_ema, infer_flag = seg_engine.step(vis_in)

        # 디스플레이 버퍼
        disp = cv.resize(seg_vis, (args.win_w, args.win_h), interpolation=cv.INTER_LINEAR)

        # DOA 화살표 (최근 SHOW_MS 내 수신 시)
        now_ms = int(time.time()*1000)
        if (doa_state["last_angle"] is not None and
            (now_ms - doa_state["last_rx_ms"]) <= doa_state["SHOW_MS"]):
            side, zone = classify_zone(
                doa_state["last_angle"], doa_state["yaw_offset"], doa_state["th_low"], doa_state["th_high"]
            )
            draw_filled_arrow(
                disp, side, zone,
                arrow_scale=args.arrow_scale,
                arrow_border=args.arrow_border,
                fill_bgr=(0,0,255), edge_bgr=(0,0,0)
            )

        # HUD (SEG 모드만 표시 유지)
        if doa_state["show_help"] or (now_ms - doa_state["last_change_ms"] <= doa_state["HUD_MS"]):
            draw_hud(disp, doa_state["yaw_offset"], doa_state["th_low"], doa_state["th_high"],
                     doa_state["show_help"], doa_state["last_angle"])

        # 전환 아이콘 (그래픽만)
        rect = draw_switch_icon_on_canvas(disp)
        clicker.set_icon_rect(rect)

        # 상태 텍스트(SEG 모드)
        info = f"SEG+DOA  FPS:{fps_ema:5.1f}  infer:{'Y' if infer_flag else 'N'}  objs:{len(contours):d}"
        cv.putText(disp, info, (12, 28), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2, cv.LINE_AA)

        # 표시 & 입력
        cv.imshow(win, disp)
        key = cv.waitKey(1) & 0xFF

        # 키 처리
        if key == ord('q'): return {"quit": True}
        if key == ord('h'):
            doa_state["show_help"] = not doa_state["show_help"]
            doa_state["last_change_ms"] = now_ms
        if key == 81:  # ←
            doa_state["yaw_offset"] -= 1.0; doa_state["last_change_ms"] = now_ms
        elif key == 83:  # →
            doa_state["yaw_offset"] += 1.0; doa_state["last_change_ms"] = now_ms
        elif key == 82:  # ↑  (TH_LOW up)
            doa_state["th_low"] = max(0.0, min(doa_state["th_low"] + 1.0, doa_state["th_high"] - 1e-3)); doa_state["last_change_ms"] = now_ms
        elif key == 84:  # ↓  (TH_LOW down)
            doa_state["th_low"] = max(0.0, min(doa_state["th_low"] - 1.0, doa_state["th_high"] - 1e-3)); doa_state["last_change_ms"] = now_ms
        elif key == ord('['):  # TH_HIGH down
            doa_state["th_high"] = max(doa_state["th_high"] - 1.0, doa_state["th_low"] + 1e-3); doa_state["last_change_ms"] = now_ms
        elif key == ord(']'):  # TH_HIGH up
            doa_state["th_high"] = max(doa_state["th_high"] + 1.0, doa_state["th_low"] + 1e-3); doa_state["last_change_ms"] = now_ms

        if clicker.switch_requested:
            clicker.switch_requested = False
            return {"switch": "THERMAL"}

        frames += 1
        if frames % 120 == 0:
            fps = frames / (time.time() - t0)
            print(f"[SEG+DOA] ~{fps:.1f} FPS")

def run_thermal_onnx(args, clicker):
    """MI48 + YOLO(ONNX) — disp 크기로 리사이즈 후 아이콘; 텍스트 없음(요청사항)"""
    # DNN 로드
    if not os.path.isfile(args.th_model):
        disp = np.zeros((args.win_h, args.win_w, 3), np.uint8)
        cv.putText(disp, "ONNX not found", (40, 80), cv.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3, cv.LINE_AA)
        rect = draw_switch_icon_on_canvas(disp); clicker.set_icon_rect(rect)
        cv.imshow(args.win_name, disp); cv.waitKey(1200)
        return {"switch":"RGB"}

    net = cv.dnn.readNet(args.th_model)
    try:
        net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
    except Exception:
        pass

    input_sz = args.th_imgsz
    in_ch = detect_expected_channels_from_net(net)
    family = guess_yolo_family(net, input_sz, in_ch)
    print(f"[THERMAL] model in_ch={in_ch}, family={family}")

    th = ThermalHW(args)
    try:
        th.start()
    except Exception as e:
        disp = np.zeros((args.win_h, args.win_w, 3), np.uint8)
        cv.putText(disp, f"Thermal start error: {str(e)[:50]}", (20, 80), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2, cv.LINE_AA)
        rect = draw_switch_icon_on_canvas(disp); clicker.set_icon_rect(rect)
        cv.imshow(args.win_name, disp); cv.waitKey(1500)
        return {"switch":"RGB"}

    win = args.win_name
    first_log = True
    t0 = time.time(); frames = 0

    try:
        while True:
            try:
                th.drdy.wait_for_active(timeout=1.0)
            except Exception:
                pass

            data, header = th.read_frame()
            if data is None:
                disp = np.zeros((args.win_h, args.win_w, 3), np.uint8)
                rect = draw_switch_icon_on_canvas(disp); clicker.set_icon_rect(rect)
                cv.imshow(win, disp)
                if cv.waitKey(1) & 0xFF == ord('q'):
                    return {"quit": True}
                if clicker.switch_requested:
                    clicker.switch_requested = False
                    return {"switch": "RGB"}
                continue

            frame = data_to_frame(data, th.mi48.fpa_shape)
            img8  = cv.normalize(frame.astype(np.float32), None, 0, 255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
            img8  = cv_filter(img8, parameters={'blur_ks':3}, use_median=False, use_bilat=True, use_nlm=False)
            H, W = img8.shape[:2]

            # ONNX 입력 (단일 thermal 스크립트와 동일)
            if in_ch == 3:
                img_for_net = cv.cvtColor(img8, cv.COLOR_GRAY2RGB)
            else:
                img_for_net = img8

            inp, scale, px, py = letterbox(img_for_net, new_size=input_sz, color=0)
            if in_ch == 3 and inp.ndim == 2:
                inp = cv.cvtColor(inp, cv.COLOR_GRAY2RGB)
            blob = cv.dnn.blobFromImage(inp, scalefactor=1/255.0, size=(input_sz, input_sz),
                                        mean=(0,0,0), swapRB=False, crop=False)
            net.setInput(blob)
            out = net.forward()

            if first_log:
                arr = np.squeeze(out)
                try:
                    max_conf = float(arr[...,4].max()) if arr.ndim>=2 and arr.shape[-1]>=5 else float(arr.max())
                except Exception:
                    max_conf = float(arr.max())
                print(f"[THERMAL] ONNX out: {arr.shape}, max_raw={max_conf:.4f}")
                first_log = False

            if family == "v8":
                dets = post_v8(out, args.th_conf, args.th_iou, input_sz, scale, px, py, W, H)
            else:
                dets = post_v5(out, args.th_conf, args.th_iou, input_sz, scale, px, py, W, H)

            # 시각화: 초록 박스(점수만) — 텍스트는 점수만, 모드명 미표시
            vis = cv.applyColorMap(img8, cv.COLORMAP_INFERNO)
            for (x1,y1,x2,y2), score in dets:
                cv.rectangle(vis, (x1,y1), (x2,y2), (0,255,0), 2)
                cv.putText(vis, f"{score:.2f}", (x1, max(0,y1-6)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

            disp = cv.resize(vis, (args.win_w, args.win_h), interpolation=cv.INTER_CUBIC)
            rect = draw_switch_icon_on_canvas(disp)
            clicker.set_icon_rect(rect)

            cv.imshow(win, disp)
            key = cv.waitKey(1) & 0xFF
            if key == ord('q'):
                return {"quit": True}
            if clicker.switch_requested:
                clicker.switch_requested = False
                return {"switch": "RGB"}

            frames += 1
            if frames % 120 == 0:
                fps = frames / (time.time() - t0)
                print(f"[THERMAL] ~{fps:.1f} FPS")
    finally:
        th.stop()

# ===== 인자/메인 ==============================================================
def parse_args():
    p = argparse.ArgumentParser()
    # Thermal ONNX (열화상 전용 네임스페이스)
    p.add_argument("--th-model", required=True, type=str, help="YOLOv5/YOLOv8 ONNX path for Thermal")
    p.add_argument("--th-imgsz", default=128, type=int)
    p.add_argument("--th-conf",  default=0.35, type=float)
    p.add_argument("--th-iou",   default=0.45, type=float)
    # MI48 / IO
    p.add_argument("--mi-fps",   default=9.0,  type=float)
    p.add_argument("--spi-bus",  default=0, type=int)
    p.add_argument("--spi-dev",  default=0, type=int)
    p.add_argument("--i2c-bus",  default=1, type=int)
    p.add_argument("--i2c-addr", default=0x40, type=lambda x:int(x,0))
    p.add_argument("--speed",    default=8_000_000, type=int)
    p.add_argument("--csdelay",  default=0.0002, type=float)

    # Window
    p.add_argument("--win-w", type=int, default=800)
    p.add_argument("--win-h", type=int, default=480)
    p.add_argument("--fullscreen", action="store_true")
    p.add_argument("--win-name", default="DUAL-CAM UI", help="Window name")

    # RGB out size (파이 카메라 프리뷰 크기)
    p.add_argument("--rgb-out-w", type=int, default=1280)
    p.add_argument("--rgb-out-h", type=int, default=720)

    # DOA
    p.add_argument("--doa-port", type=int, default=PI_LISTEN_PORT_DEFAULT)
    p.add_argument("--yaw-init", type=float, default=0.0)
    p.add_argument("--th-low-init", type=float, default=5.0)
    p.add_argument("--th-high-init", type=float, default=12.0)

    # Arrow styling
    p.add_argument("--arrow-scale", type=float, default=0.12, help="비율(화면 최소변 대비)")
    p.add_argument("--arrow-border", type=int, default=6, help="외곽선 두께(px)")

    # Segmentation (파이 카메라 전용 네임스페이스)
    p.add_argument("--seg-onnx", required=True, type=str, help="YOLOv8-Seg ONNX path for Pi camera")
    p.add_argument("--seg-conf", type=float, default=0.65)
    p.add_argument("--seg-iou",  type=float, default=0.4)
    p.add_argument("--seg-mask", type=float, default=0.55)
    p.add_argument("--seg-infer-every", type=int, default=2)
    return p.parse_args()

def main():
    args = parse_args()

    # UDP 소켓
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", args.doa_port))
    sock.setblocking(False)

    # DOA 상태
    doa_state = dict(
        yaw_offset = args.yaw_init,
        th_low     = args.th_low_init,
        th_high    = args.th_high_init,
        SHOW_MS    = 600,
        HUD_MS     = 1500,
        show_help  = False,
        last_angle = None,
        last_rx_ms = 0,
        last_change_ms = 0,
    )

    # OpenCV 창
    cv.namedWindow(args.win_name, cv.WINDOW_NORMAL)
    try:
        if args.fullscreen:
            cv.setWindowProperty(args.win_name, cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
        else:
            cv.resizeWindow(args.win_name, args.win_w, args.win_h)
            try: cv.moveWindow(args.win_name, 0, 0)
            except Exception: pass
    except Exception:
        pass

    clicker = ClickState()
    cv.setMouseCallback(args.win_name, clicker.handle)

    # Seg 엔진 준비 (사람만)
    seg_engine = SegEngine(
        onnx_path=args.seg_onnx,
        conf_th=args.seg_conf, iou_th=args.seg_iou, mask_th=args.seg_mask,
        infer_every=args.seg_infer_every, ema_alpha=0.6, fps_alpha=0.9
    )

    mode = "SEG"  # 시작: Segmentation + DOA
    try:
        while True:
            if mode == "SEG":
                res = run_seg_with_doa(args, sock, doa_state, clicker, seg_engine)
                if res.get("quit"): break
                if res.get("switch") == "THERMAL":
                    stop_rgb_camera()
                    mode = "THERMAL"
            else:  # THERMAL
                res = run_thermal_onnx(args, clicker)
                if res.get("quit"): break
                if res.get("switch") in ("RGB", "SEG"):
                    mode = "SEG"
    finally:
        close_rgb_camera()
        cv.destroyAllWindows()

if __name__ == "__main__":
    main()
