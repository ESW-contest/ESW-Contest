#!/usr/bin/env python3
# VAD + DOA + UDP (NumPy only) with pitch gate, GCC confidence, rate limit

import sys, time, math, socket, collections, argparse
import numpy as np
import sounddevice as sd
import webrtcvad

# ===== 기본 설정 =====

USE_PITCH_GATE = False

FS = 16000
VAD_FRAME = int(0.02 * FS)      # 20ms(320)
DOA_FRAME = VAD_FRAME * 6       # 120ms(1920)
HOP = DOA_FRAME // 2            # 50% overlap(960)

MIC_DISTANCE_M = 0.08           # 마이크 간격(정확히!)
SPEED_SOUND = 343.0

# VAD
VAD_MODE = 2                    # 0~3(클수록 엄격)
VAD_RATIO_THRESH = 0.6          # 120ms 블록 내 음성 프레임 비율
RMS_THRESH_STATIC = 130.0       # 무음 컷(최저 기준)

# 밴드/피치
LOWCUT, HIGHCUT = 300.0, 3400.0
PITCH_MIN, PITCH_MAX = 85.0, 300.0    # 남/여성 일반 구간
PITCH_STRONG_THRESH = 0.35            # 자기상관 정규화 피크 임계(0~1)

# GCC-PHAT 신뢰도
GCC_INTERP = 8                         # 해상도(4~16)
GCC_PEAK_RATIO_MIN = 1.10               # 1등/2등 피크 비
GCC_PROMINENCE_MIN = 3.50              # 1등 피크 / 중앙값(잡음바닥) 비

# 스무딩/잠금
EMA_ALPHA = 0.25
MEDIAN_WIN = 7
MAX_DEG_PER_FRAME = 8.0               # 프레임당 최대 각 이동(튀는 값 컷)

PI_IP = "192.168.100.1"
PI_PORT = 5005

DEBUG = True
DEBUG_SKIP_LOG_EVERY = 6

def gcc_phat(sig, refsig, fs=1, max_tau=None, interp=1):
    n = 1 << ((sig.size + refsig.size - 1).bit_length())
    SIG = np.fft.rfft(sig, n=n); REFSIG = np.fft.rfft(refsig, n=n)
    R = SIG * np.conj(REFSIG); R /= np.abs(R) + 1e-12
    cc = np.fft.irfft(R, n=(interp*n))
    max_shift = int(interp*n/2)
    if max_tau is not None:
        max_shift = min(int(interp*fs*max_tau), max_shift)
    cc = np.concatenate((cc[-max_shift:], cc[:max_shift+1]))
    shift = np.argmax(np.abs(cc)) - max_shift
    return shift/(interp*fs), cc

def tau_to_angle(tau, d, c):
    x = max(-1.0, min(1.0, (c*tau)/max(d,1e-9)))
    return math.degrees(math.asin(x))  # -90~+90

def rms_int16(x: np.ndarray):
    return float(np.sqrt(np.mean(x.astype(np.float32)**2)))

def fft_bandpass(x: np.ndarray, fs: int, low: float, high: float):
    N = x.shape[0]
    xw = x.astype(np.float32) * np.hanning(N).astype(np.float32)
    X = np.fft.rfft(xw)
    freqs = np.fft.rfftfreq(N, 1.0/fs)
    mask = (freqs >= low) & (freqs <= high)
    X *= mask
    y = np.fft.irfft(X, n=N)
    y -= np.mean(y)
    return y.astype(np.float32)

def detect_pitch_autocorr(mono_f32: np.ndarray, fs: int):
    """ 간단한 자기상관 기반 피치 탐지 (정규화 피크) """
    x = mono_f32 - np.mean(mono_f32)
    x = x / (np.std(x) + 1e-8)
    # 관심 라그 범위
    lag_min = int(fs / PITCH_MAX)
    lag_max = int(fs / PITCH_MIN)
    if lag_max >= x.size: lag_max = x.size - 1
    if lag_min < 1 or lag_min >= lag_max:
        return None, 0.0
    # ACF via FFT (빠르게)
    n = 1 << (2*len(x)-1).bit_length()
    X = np.fft.rfft(x, n=n)
    acf = np.fft.irfft(X * np.conj(X), n=n)[:len(x)]
    acf /= (acf[0] + 1e-8)  # 정규화
    seg = acf[lag_min:lag_max+1]
    k = np.argmax(seg)
    peak = float(seg[k])
    lag = lag_min + k
    f0 = fs / lag
    return f0, peak

def pick_input_device(name_hint: str|None):
    devs = sd.query_devices()
    # 이름 힌트 우선
    if name_hint:
        for i, d in enumerate(devs):
            n = d["name"].lower()
            if d.get("max_input_channels", 0) >= 2 and name_hint.lower() in n:
                if "default" in n and "axera" not in n:
                    continue
                return i, devs
    # 2ch 이상 실장치
    for i, d in enumerate(devs):
        n = d["name"].lower()
        if d.get("max_input_channels", 0) >= 2 and "default" not in n:
            return i, devs
    # 마지막 수단
    default_in = sd.default.device[0] if isinstance(sd.default.device, (list, tuple)) else sd.default.device
    return default_in, devs

def calibrate_noise(stream, seconds=2.0):
    need = int(seconds * FS)
    buf = np.zeros((0,2), dtype=np.int16)
    while buf.shape[0] < need:
        data, _ = stream.read(min(need - buf.shape[0], HOP))
        buf = np.vstack((buf, data))
    base = (rms_int16(buf[:,0]) + rms_int16(buf[:,1])) * 0.5
    return base

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", type=str, default=None, help="입력 장치 인덱스 또는 이름 힌트(예: 'Axera')")
    ap.add_argument("--ip",  type=str, default=PI_IP)
    ap.add_argument("--port",type=int, default=PI_PORT)
    ap.add_argument("--vad_mode", type=int, default=VAD_MODE)
    ap.add_argument("--ratio", type=float, default=VAD_RATIO_THRESH)
    args = ap.parse_args()

    # 디바이스 선택
    dev_idx = None
    try:
        dev_idx = int(args.dev) if args.dev and args.dev.isdigit() else None
    except: pass
    name_hint = None if dev_idx is not None else args.dev

    if dev_idx is None:
        dev_idx, devs = pick_input_device(name_hint)
    else:
        devs = sd.query_devices()
    dev_name = devs[int(dev_idx)]["name"] if isinstance(dev_idx, int) else str(dev_idx)
    print("[INFO] 입력 디바이스:", dev_idx, "-", dev_name)

    # 기본쌍 충돌 회피(입력만 고정)
    try:
        out = sd.default.device[1] if isinstance(sd.default.device, (list, tuple)) else None
        sd.default.device = (int(dev_idx), out)
    except Exception:
        pass

    vad = webrtcvad.Vad(int(args.vad_mode))
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    ema_angle = 0.0
    medbuf = collections.deque(maxlen=MEDIAN_WIN)
    last_sent_angle = 0.0
    have_angle = False

    stream = sd.InputStream(samplerate=FS, channels=2, dtype='int16',
                            blocksize=HOP, device=(int(dev_idx), None))
    stream.start()

    # 노이즈 바닥
    base = calibrate_noise(stream, seconds=2.0)
    rms_thresh = max(RMS_THRESH_STATIC, base * 2.0)
    print(f"[INFO] 노이즈 바닥={base:.1f}, 적용 RMS_THRESH={rms_thresh:.1f}, VAD_MODE={args.vad_mode}, RATIO={args.ratio:.2f}")

    buf = np.zeros((0, 2), dtype=np.int16)
    frame_idx = 0

    try:
        while True:
            data, _ = stream.read(HOP)
            buf = np.vstack((buf, data))
            if buf.shape[0] < DOA_FRAME:
                continue

            frame = buf[:DOA_FRAME, :]
            buf = buf[HOP:, :]

            # 모노(평균)로 RMS/VAD
            mono = ((frame[:,0].astype(np.int32) + frame[:,1].astype(np.int32)) // 2).astype(np.int16)
            rms = rms_int16(mono)
            if rms < rms_thresh:
                if DEBUG and (frame_idx % DEBUG_SKIP_LOG_EVERY == 0):
                    print(f"[skip] rms={rms:.0f} < {rms_thresh:.0f}")
                frame_idx += 1; continue

            # 20ms VAD 다수결
            voiced = 0; total = 0
            for i in range(0, DOA_FRAME, VAD_FRAME):
                seg = mono[i:i+VAD_FRAME]
                if seg.shape[0] < VAD_FRAME: break
                if vad.is_speech(seg.tobytes(), FS): voiced += 1
                total += 1
            ratio = voiced/total if total else 0.0
            if ratio < float(args.ratio):
                if DEBUG and (frame_idx % DEBUG_SKIP_LOG_EVERY == 0):
                    print(f"[skip] VAD ratio={ratio:.2f} < {args.ratio:.2f}")
                frame_idx += 1; continue

            # 피치 게이트 (선택)
            if USE_PITCH_GATE:
                mono_f = fft_bandpass(mono, FS, 80.0, 1200.0)  # 피치 검출용 더 낮은 대역
                f0, pstr = detect_pitch_autocorr(mono_f, FS)
                if not (f0 and PITCH_MIN <= f0 <= PITCH_MAX and pstr >= PITCH_STRONG_THRESH):
                    if DEBUG and (frame_idx % DEBUG_SKIP_LOG_EVERY == 0):
                        print(f"[skip] pitch f0={0 if not f0 else f0:.1f}Hz, strength={pstr:.2f}")
                    frame_idx += 1; continue

            # 밴드패스 후 두 채널 정규화
            ch0_f = fft_bandpass(frame[:,0], FS, LOWCUT, HIGHCUT)
            ch1_f = fft_bandpass(frame[:,1], FS, LOWCUT, HIGHCUT)
            # 정규화(편차 정렬)
            for ch in (ch0_f, ch1_f):
                ch -= np.mean(ch)
                std = np.std(ch) + 1e-6
                ch /= std

            max_tau = MIC_DISTANCE_M / SPEED_SOUND
            tau, cc = gcc_phat(ch0_f, ch1_f, fs=FS, max_tau=max_tau, interp=GCC_INTERP)

            # GCC 신뢰도 평가
            abscc = np.abs(cc)
            p0_idx = int(np.argmax(abscc))
            p0 = float(abscc[p0_idx])
            # 2등 피크(주변 몇 샘플 제외)
            exc = 6
            mask = np.ones_like(abscc, dtype=bool)
            mask[max(0, p0_idx-exc): p0_idx+exc+1] = False
            p1 = float(abscc[mask].max()) if np.any(mask) else 1e-6
            ratio_peak = (p0 / (p1 + 1e-6))
            prom = (p0 / (np.median(abscc) + 1e-6))

            if ratio_peak < GCC_PEAK_RATIO_MIN or prom < GCC_PROMINENCE_MIN:
                if DEBUG and (frame_idx % DEBUG_SKIP_LOG_EVERY == 0):
                    print(f"[skip] gcc conf: peak_ratio={ratio_peak:.2f} (<{GCC_PEAK_RATIO_MIN}), prom={prom:.1f} (<{GCC_PROMINENCE_MIN})")
                frame_idx += 1; continue

            raw_angle = tau_to_angle(tau, MIC_DISTANCE_M, SPEED_SOUND)

            # 변화율 제한 + 스무딩
            medbuf.append(raw_angle)
            med = float(np.median(medbuf))
            # 최대 이동 각 제한
            target = med if have_angle else raw_angle
            if have_angle:
                delta = target - last_sent_angle
                if abs(delta) > MAX_DEG_PER_FRAME:
                    target = last_sent_angle + np.sign(delta) * MAX_DEG_PER_FRAME
            out_angle = EMA_ALPHA * target + (1 - EMA_ALPHA) * (last_sent_angle if have_angle else 0.0)
            last_sent_angle = out_angle
            have_angle = True

            # 송신
            msg = f"ANGLE:{out_angle:.1f}\n".encode()
            sock.sendto(msg, (args.ip, args.port))

            if DEBUG:
                info_pitch = "" if not USE_PITCH_GATE else f" f0={f0:.0f}Hz,str={pstr:.2f}"
                print(f"[SEND] angle={out_angle:5.1f}°, raw={raw_angle:5.1f}°, med={med:5.1f}°, VAD={voiced}/{total}({ratio:.2f}),"
                      f" gccR={ratio_peak:.2f},prom={prom:.1f},{info_pitch}")
            frame_idx += 1

    except KeyboardInterrupt:
        pass
    finally:
        try: stream.stop(); stream.close()
        except: pass

if __name__ == "__main__":
    main()
