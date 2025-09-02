# one.py
# ------------------------------------------------------------
# YOLOv8 Segmentation (ONNX) + Webcam
# 결과는 "윤곽선만" 그려서 표시. 박스/라벨/채우기 없음.
# 노이즈 억제 + 프레임간 스무딩(EMA) + 윤곽선 단순화
# ------------------------------------------------------------

import cv2
import numpy as np
import onnxruntime as ort

# ---------- 하이퍼파라미터 ----------
CONF_TH = 0.62
IOU_TH  = 0.5
MASK_TH = 0.6

OUTLINE_THICK = 4
OUTLINE_COLOR = (255, 255, 0)

MIN_AREA_RATIO = 0.0001
K_OPEN  = 1
K_CLOSE = 3
GAUSS_K = 1

# === 추가: 프레임간 스무딩 파라미터 ===
prev_scene_prob = None
EMA_ALPHA = 0.6  # 0.7~0.85 사이 조절

# ---------- 유틸리티 ----------
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def nms(boxes, scores, iou_thr=0.5, top_k=300):
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

def letterbox(im, new_shape=(640,640), color=(114,114,114), scaleup=True):
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

# ---------- ONNX 모델 로드 ----------
session = ort.InferenceSession("last.onnx", providers=["CPUExecutionProvider"])
inp = session.get_inputs()[0]
input_name = inp.name
in_h = inp.shape[2] if isinstance(inp.shape[2], int) and inp.shape[2] > 0 else 640
in_w = inp.shape[3] if isinstance(inp.shape[3], int) and inp.shape[3] > 0 else 640
print(f"[INFO] ONNX input size = {in_w}x{in_h}")

# ---------- 웹캠 ----------
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    raise RuntimeError("웹캠을 열 수 없습니다.")

while True:
    ok, frame = cap.read()
    if not ok:
        break

    Hf, Wf = frame.shape[:2]

    # 1) 전처리
    lb_img, r, (dw, dh) = letterbox(frame, (in_h, in_w))
    img = cv2.cvtColor(lb_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))[None, ...]

    # 2) 추론
    preds, protos = session.run(None, {input_name: img})
    preds = np.squeeze(preds, 0).transpose(1, 0)

    vis = frame.copy()

    if preds.shape[1] == 37:
        boxes_cxcywh = preds[:, 0:4]
        cls_logits   = preds[:, 4:5]
        mask_coefs   = preds[:, 5:37]
        if np.max(sigmoid(cls_logits)) < 0.01 and np.max(sigmoid(preds[:, 36:37])) > np.max(sigmoid(cls_logits)):
            mask_coefs = preds[:, 4:36]
            cls_logits = preds[:, 36:37]
    else:
        cv2.imshow("Segmentation Outline (Q to quit)", vis)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    scores = sigmoid(cls_logits).squeeze(-1)
    cx, cy, w, h = [boxes_cxcywh[:, i] for i in range(4)]
    x1 = cx - w/2; y1 = cy - h/2; x2 = cx + w/2; y2 = cy + h/2
    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

    keep0 = scores > CONF_TH
    boxes_xyxy = boxes_xyxy[keep0]
    scores_k   = scores[keep0]
    mask_coefs = mask_coefs[keep0]

    if boxes_xyxy.shape[0] > 0:
        keep = nms(boxes_xyxy, scores_k, iou_thr=IOU_TH)
        boxes_xyxy = boxes_xyxy[keep]
        scores_k   = scores_k[keep]
        mask_coefs = mask_coefs[keep]

        proto = np.squeeze(protos, 0).reshape(32, -1)

        # --- 프레임 전체 확률맵 누적 ---
        scene_prob = np.zeros((Hf, Wf), dtype=np.float32)

        for coef in mask_coefs:
            m = sigmoid(np.dot(coef, proto)).reshape(160, 160).astype(np.float32)
            m = cv2.resize(m, (in_w, in_h), interpolation=cv2.INTER_LINEAR)
            m = m[int(dh):int(in_h-dh), int(dw):int(in_w-dw)]
            if m.size == 0:
                continue
            m = cv2.resize(m, (Wf, Hf), interpolation=cv2.INTER_LINEAR)
            scene_prob = np.maximum(scene_prob, m)

        # --- EMA 스무딩 ---
        
        if prev_scene_prob is None:
            smooth_prob = scene_prob
        else:
            smooth_prob = EMA_ALPHA * prev_scene_prob + (1.0 - EMA_ALPHA) * scene_prob
        prev_scene_prob = smooth_prob

        # --- 이진화 ---
        m_bin = (smooth_prob > MASK_TH).astype(np.uint8) * 255

        if K_OPEN > 1:
            m_bin = cv2.morphologyEx(m_bin, cv2.MORPH_OPEN, np.ones((K_OPEN,K_OPEN), np.uint8))
        if K_CLOSE > 1:
            m_bin = cv2.morphologyEx(m_bin, cv2.MORPH_CLOSE, np.ones((K_CLOSE,K_CLOSE), np.uint8))

        num, labels, stats, _ = cv2.connectedComponentsWithStats((m_bin > 0).astype(np.uint8), connectivity=8)
        keep_mask = np.zeros_like(m_bin)
        for i in range(1, num):
            if stats[i, cv2.CC_STAT_AREA] >= int(MIN_AREA_RATIO * Hf * Wf):
                keep_mask[labels == i] = 255
        m_bin = keep_mask

        if GAUSS_K >= 3 and GAUSS_K % 2 == 1:
            m_bin = cv2.GaussianBlur(m_bin, (GAUSS_K, GAUSS_K), 0)

        contours, _ = cv2.findContours(m_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [c for c in contours if cv2.contourArea(c) >= int(MIN_AREA_RATIO * Hf * Wf)]

        # --- 윤곽선 단순화 ---
        simplified = []
        for c in contours:
            eps = 0.003 * cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, eps, True)
            simplified.append(approx)

        if simplified:
            cv2.drawContours(vis, simplified, -1, OUTLINE_COLOR, OUTLINE_THICK)

    cv2.imshow("Segmentation Outline (Q to quit)", vis)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
