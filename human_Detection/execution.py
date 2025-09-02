# ------------------------------------------------------------
# YOLOv8 Segmentation (ONNX) + Picamera2 (Raspberry Pi 5)
# - 색 채널: XRGB8888 + BGRA->BGR 변환 (파란기 해결)
# - 입력 해상도: 640x384 (letterbox scaleup=False)
# - FPS 표시 (EMA), 프레임 스킵(INFER_EVERY)
# - 마스크 생성 벡터화(행렬곱)로 후처리 가속
# - 윤곽선만 그리기 (박스/라벨/채우기 없음)
# ------------------------------------------------------------

import time
import cv2
import numpy as np
import onnxruntime as ort
from picamera2 import Picamera2
from libcamera import Transform

cv2.setUseOptimized(True)

# ---------- 하이퍼파라미터 ----------
CONF_TH = 0.65
IOU_TH  = 0.4
MASK_TH = 0.55

OUTLINE_THICK = 4
OUTLINE_COLOR = (255, 255, 0)

MIN_AREA_RATIO = 0.0001
K_OPEN  = 1
K_CLOSE = 2
GAUSS_K = 1

# 프레임 스무딩(EMA)
prev_scene_prob = None
EMA_ALPHA = 0.6  # 0.7~0.85 사이 조절

# 프레임 스킵(지각 업데이트)
INFER_EVERY = 2   # 2프레임에 한 번 추론
frame_id = 0

# FPS 표시
FPS_ALPHA = 0.9   # EMA 스무딩
fps_ema = None
prev_t = time.time()

# ---------- 유틸리티 ----------
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def nms(boxes, scores, iou_thr=0.5, top_k=60):
    idxs = scores.argsort()[::-1][:top_k]
    keep = []
    while idxs.size > 0:
        i = idxs[0]
        keep.append(i)
        if idxs.size == 1:
            break
        rest = idxs[1:]
        xx1 = np.maximum(boxes[i,0], boxes[rest,0])
        yy1 = np.maximum(boxes[i,1], boxes[rest,1])
        xx2 = np.minimum(boxes[i,2], boxes[rest,2])
        yy2 = np.minimum(boxes[i,3], boxes[rest,3])
        w = np.maximum(0.0, xx2-xx1)
        h = np.maximum(0.0, yy2-yy1)
        inter = w*h
        area_i = (boxes[i,2]-boxes[i,0])*(boxes[i,3]-boxes[i,1])
        area_r = (boxes[rest,2]-boxes[rest,0])*(boxes[rest,3]-boxes[rest,1])
        iou = inter / (area_i + area_r - inter + 1e-6)
        idxs = rest[iou <= iou_thr]
    return np.array(keep, dtype=np.int32)

def letterbox(im, new_shape=(640,640), color=(114,114,114), scaleup=False):
    h, w = im.shape[:2]
    r = min(new_shape[0]/h, new_shape[1]/w)
    if not scaleup:
        r = min(r, 1.0)
    new_unpad = (int(round(w*r)), int(round(h*r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2; dh /= 2
    if (w, h) != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh-0.1)), int(round(dh+0.1))
    left, right = int(round(dw-0.1)), int(round(dw+0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, (dw, dh)

# ---------- ONNX 모델 로드 (세션 최적화 포함) ----------
so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
so.intra_op_num_threads = 4   # Pi 5: 4코어
so.inter_op_num_threads = 1

session = ort.InferenceSession(
    "last.onnx",
    sess_options=so,
    providers=["CPUExecutionProvider"]  # OpenVINO 설치 시 ["OpenVINOExecutionProvider","CPUExecutionProvider"]
)

inp = session.get_inputs()[0]
input_name = inp.name
in_h = inp.shape[2] if isinstance(inp.shape[2], int) and inp.shape[2] > 0 else 640
in_w = inp.shape[3] if isinstance(inp.shape[3], int) and inp.shape[3] > 0 else 640
print(f"[INFO] ONNX input size = {in_w}x{in_h}")

# ---------- Picamera2 설정 ----------
# 1) 해상도 640x384로 낮춤(16:9), 2) XRGB8888 포맷
CAM_W, CAM_H = 640, 384

picam = Picamera2()
video_cfg = picam.create_video_configuration(
    main={"size": (CAM_W, CAM_H), "format": "XRGB8888"},  # OpenCV 호환
    transform=Transform(hflip=0, vflip=0)  # 필요시 뒤집기
)
picam.configure(video_cfg)
picam.start()
print("[INFO] Picamera2 started.")

# ---------- DSI 디스플레이(풀스크린) 창 준비 ----------
WIN_NAME = "Segmentation Outline (Q to quit)"
cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(WIN_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

try:
    while True:
        # 시간 측정 (FPS)
        now = time.time()
        dt = now - prev_t
        prev_t = now
        inst_fps = (1.0 / dt) if dt > 0 else 0.0
        fps_ema = inst_fps if fps_ema is None else (FPS_ALPHA * fps_ema + (1.0 - FPS_ALPHA) * inst_fps)

        # Picamera2는 XRGB8888로 들어옴 -> BGRA2BGR 한 번만 변환
        frame_bgra = picam.capture_array()  # (H, W, 4)
        frame = cv2.cvtColor(frame_bgra, cv2.COLOR_BGRA2BGR)
        Hf, Wf = frame.shape[:2]

        # 이 프레임에서 추론할지 결정 (프레임 스킵)
        do_infer = (frame_id % INFER_EVERY == 0)
        frame_id += 1

        # 1) 전처리 (scaleup=False)
        lb_img, r, (dw, dh) = letterbox(frame, (in_h, in_w), scaleup=False)
        img = cv2.cvtColor(lb_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))[None, ...]

        scene_prob = None

        if do_infer:
            # 2) 추론
            preds, protos = session.run(None, {input_name: img})
            preds = np.squeeze(preds, 0).transpose(1, 0)

            # 가시화용 프레임(원본 위에 바로 그립니다)
            vis = frame

            # 헤드 포맷 정리
            if preds.shape[1] == 37:
                boxes_cxcywh = preds[:, 0:4]
                cls_logits   = preds[:, 4:5]
                mask_coefs   = preds[:, 5:37]
                # 일부 모델 변종 대응
                if np.max(sigmoid(cls_logits)) < 0.01 and np.max(sigmoid(preds[:, 36:37])) > np.max(sigmoid(cls_logits)):
                    mask_coefs = preds[:, 4:36]
                    cls_logits = preds[:, 36:37]
            else:
                # 포맷 미일치 시 그냥 디스플레이
                cv2.putText(frame, f"FPS:{fps_ema:5.1f}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2, cv2.LINE_AA)
                cv2.imshow(WIN_NAME, frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            # 박스/스코어 필터 + NMS
            scores = sigmoid(cls_logits).squeeze(-1)
            cx, cy, w, h = [boxes_cxcywh[:, i] for i in range(4)]
            x1 = cx - w/2; y1 = cy - h/2; x2 = cx + w/2; y2 = cy + h/2
            boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

            keep0 = scores > CONF_TH
            boxes_xyxy = boxes_xyxy[keep0]
            scores_k   = scores[keep0]
            mask_coefs = mask_coefs[keep0]

            if boxes_xyxy.shape[0] > 0:
                keep = nms(boxes_xyxy, scores_k, iou_thr=IOU_TH, top_k=100)
                boxes_xyxy = boxes_xyxy[keep]
                scores_k   = scores_k[keep]
                mask_coefs = mask_coefs[keep]

                # ---- 마스크 벡터화 계산 (행렬곱) ----
                # protos: (1, 32, 160, 160) 가정
                proto = np.squeeze(protos, 0).reshape(32, -1).astype(np.float32)  # (32, 25600)

                # (K,32) @ (32, 25600) -> (K, 25600)
                M = mask_coefs.astype(np.float32) @ proto
                M = sigmoid(M).reshape(-1, 160, 160)  # (K, 160,160)

                # 객체들 확률맵을 픽셀별 최대값으로 합치기
                scene_prob_small = np.max(M, axis=0)  # (160,160)

                # 패딩 제거 후 최종 1회 리사이즈
                scene_prob_pad = cv2.resize(scene_prob_small, (in_w, in_h), interpolation=cv2.INTER_LINEAR)
                top = int(dh); left = int(dw)
                bottom = int(in_h - dh); right = int(in_w - dw)
                scene_prob_pad = scene_prob_pad[top:bottom, left:right]
                if scene_prob_pad.size == 0:
                    scene_prob = np.zeros((Hf, Wf), dtype=np.float32)
                else:
                    scene_prob = cv2.resize(scene_prob_pad, (Wf, Hf), interpolation=cv2.INTER_LINEAR)
            else:
                scene_prob = np.zeros((Hf, Wf), dtype=np.float32)

            # --- EMA 스무딩 ---
            if prev_scene_prob is None:
                smooth_prob = scene_prob
            else:
                smooth_prob = EMA_ALPHA * prev_scene_prob + (1.0 - EMA_ALPHA) * scene_prob
            prev_scene_prob = smooth_prob

        else:
            # 추론 스킵: 이전 결과 유지
            vis = frame
            smooth_prob = prev_scene_prob if prev_scene_prob is not None else np.zeros((Hf, Wf), dtype=np.float32)

        # --- 이진화 & 후처리 ---
        m_bin = (smooth_prob > MASK_TH).astype(np.uint8) * 255

        if K_OPEN > 1:
            m_bin = cv2.morphologyEx(m_bin, cv2.MORPH_OPEN, np.ones((K_OPEN,K_OPEN), np.uint8))
        if K_CLOSE > 1:
            m_bin = cv2.morphologyEx(m_bin, cv2.MORPH_CLOSE, np.ones((K_CLOSE,K_CLOSE), np.uint8))

        num, labels, stats, _ = cv2.connectedComponentsWithStats((m_bin > 0).astype(np.uint8), connectivity=8)
        keep_mask = np.zeros_like(m_bin)
        min_area = int(MIN_AREA_RATIO * Hf * Wf)
        for i in range(1, num):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                keep_mask[labels == i] = 255
        m_bin = keep_mask

        if GAUSS_K >= 3 and GAUSS_K % 2 == 1:
            m_bin = cv2.GaussianBlur(m_bin, (GAUSS_K, GAUSS_K), 0)

        contours, _ = cv2.findContours(m_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [c for c in contours if cv2.contourArea(c) >= min_area]

        # 윤곽선 단순화 (약간 더 강하게)
        simplified = []
        for c in contours:
            eps = 0.006 * cv2.arcLength(c, True)  # 0.003 -> 0.006로 비용/점수 감소
            approx = cv2.approxPolyDP(c, eps, True)
            simplified.append(approx)

        if simplified:
            cv2.drawContours(vis, simplified, -1, OUTLINE_COLOR, OUTLINE_THICK)

        # --- FPS/상태 표시 ---
        info = f"FPS:{fps_ema:5.1f}  infer:{'Y' if do_infer else 'N'}  objs:{len(simplified):d}"
        cv2.putText(vis, info, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2, cv2.LINE_AA)

        # --- DSI로 풀스크린 표시 ---
        cv2.imshow(WIN_NAME, vis)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    picam.stop()
    cv2.destroyAllWindows()
