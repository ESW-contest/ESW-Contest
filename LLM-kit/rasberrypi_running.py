#!/usr/bin/env python3
# cam_doa_overlay_pygame.py
# 3단계(아래/중앙/위) 코너/엣지 표시 + 소리 있을 때만 + 실시간 튜닝 핫키
# - ←/→ : YAW_OFFSET_DEG 보정 (Shift로 x5)
# - ↑/↓ : TH_LOW 조정 (Shift+↑/↓ : TH_HIGH 조정)
# - H    : 도움말 토글
# - ESC  : 종료

import os
import socket
import json
import re
import pygame
from picamera2 import Picamera2

# ===== SDL 비디오 드라이버 자동 선택(필수: pygame.init() 이전) =====
def pick_sdl_video_driver():
    # kmsdrm → wayland → x11 → fbcon → directfb → dummy 순으로 시도
    for drv in ("kmsdrm", "wayland", "x11", "fbcon", "directfb", "dummy"):
        try:
            os.environ["SDL_VIDEODRIVER"] = drv
            pygame.display.init()   # 이 드라이버로 display 모듈 초기화 시도
            pygame.display.quit()   # 탐색만 하고 종료
            print(f"[INFO] SDL_VIDEODRIVER={drv}")
            return drv
        except pygame.error:
            continue
    raise SystemExit("No usable SDL video driver found.")

# ===== 기본값(시작값) =====
PI_LISTEN_PORT = 5005

CAM_PREVIEW_SIZE = (1280, 720)
YAW_OFFSET_DEG_INIT = 0.0

SHOW_MS = 600                       # 마지막 수신 후 유지 시간
ARROW_SCALE = 1 / 15                # 화면 최소변의 1/20
ARROW_COLOR = (255, 80, 80)
ARROW_EDGE = 2
MARGIN_RATIO = 0.04

# 단계 임계값 시작값
TH_LOW_INIT  = 5.0                  # |angle| < TH_LOW  → 아래 코너
TH_HIGH_INIT = 12.0                 # |angle| ≥ TH_HIGH → 위 코너

# 키 조정 스텝
YAW_STEP = 1.0
TH_LOW_STEP = 1.0
TH_HIGH_STEP = 1.0
STEP_MULT_SHIFT = 5.0               # Shift 누르면 5배

HUD_MS = 1500                       # 값 변경 후 HUD 표시 지속 시간
FONT_SIZE = 18

ANGLE_REGEX = re.compile(r"ANGLE\s*:\s*(-?\d+(?:\.\d+)?)")


def parse_angle_line(b: bytes):
    """UDP 바이트 → angle(float) 파싱 (텍스트/JSON 모두 지원)"""
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
    """반환: side('L'|'R'), zone('BOTTOM'|'MID'|'TOP')"""
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


def make_arrow_surface(size_px, color):
    """위쪽(↑)을 향하는 등변삼각형 Surface 생성"""
    s = pygame.Surface((size_px, size_px), pygame.SRCALPHA)
    w = h = size_px
    pts = [(w * 0.5, h * 0.08), (w * 0.88, h * 0.92), (w * 0.12, h * 0.92)]
    pygame.draw.polygon(s, color, pts, width=0)
    pygame.draw.polygon(s, (0, 0, 0), pts, width=ARROW_EDGE)
    return s


def rotate_for_side_zone(base, side, zone):
    """코너/엣지 각도에 맞춰 화살표 회전"""
    if zone == 'TOP':
        angle = -45 if side == 'R' else 45
    elif zone == 'BOTTOM':
        angle = -135 if side == 'R' else 135
    else:  # MID
        angle = -90 if side == 'R' else 90
    return pygame.transform.rotate(base, angle)


def place_for_side_zone(sw, sh, size_px, side, zone):
    """화살표를 배치할 스크린 좌표 계산"""
    margin = int(min(sw, sh) * MARGIN_RATIO)
    if zone == 'TOP':
        y = margin
        x = sw - size_px - margin if side == 'R' else margin
    elif zone == 'BOTTOM':
        y = sh - size_px - margin
        x = sw - size_px - margin if side == 'R' else margin
    else:  # MID
        y = (sh - size_px) // 2
        x = sw - size_px - margin if side == 'R' else margin
    return x, y


def draw_hud(screen, font, yaw, th_low, th_high, show_help, sw, sh, last_angle=None):
    """좌상단 HUD: 파라미터/최근 각도/도움말"""
    text_lines = [
        f"YAW_OFFSET_DEG = {yaw:+.1f}",
        f"TH_LOW = {th_low:.1f}   TH_HIGH = {th_high:.1f}",
    ]
    if last_angle is not None:
        s, z = classify_zone(last_angle, yaw, th_low, th_high)
        text_lines.append(f"angle={last_angle:+.1f}°, side={s}, zone={z}")
    if show_help:
        text_lines += [
            "←/→ : yaw offset  (Shift x5)",
            "↑/↓ : TH_LOW      (Shift+↑/↓ : TH_HIGH)",
            "H: help toggle,  ESC: quit",
        ]
    pad = 8
    tsurfs = [font.render(t, True, (255, 255, 255)) for t in text_lines]
    w = max(t.get_width() for t in tsurfs) + pad * 2
    h = sum(t.get_height() for t in tsurfs) + pad * 2 + (len(tsurfs) - 1) * 2
    bg = pygame.Surface((w, h), pygame.SRCALPHA)
    bg.fill((0, 0, 0, 140))
    y = pad
    for t in tsurfs:
        bg.blit(t, (pad, y))
        y += t.get_height() + 2
    screen.blit(bg, (10, 10))


def main():
    # ⚠️ KMSDRM 강제 지정 제거: 자동 선택으로 안정화
    # os.environ.setdefault("SDL_VIDEODRIVER", "kmsdrm")  # 제거/미사용
    pick_sdl_video_driver()  # 사용 가능한 SDL 비디오 드라이버 자동 선택

    pygame.init()

    # 카메라
    picamera2 = Picamera2()
    video_cfg = picamera2.create_video_configuration(
        main={"size": CAM_PREVIEW_SIZE, "format": "RGB888"}
    )
    picamera2.configure(video_cfg)
    picamera2.start()

    # 디스플레이
    screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    sw, sh = screen.get_size()
    pygame.font.init()
    font = pygame.font.SysFont(None, FONT_SIZE)

    # 화살표 리소스
    arrow_size = int(min(sw, sh) * ARROW_SCALE)
    arrow_base = make_arrow_surface(arrow_size, ARROW_COLOR)

    # UDP
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", PI_LISTEN_PORT))
    sock.setblocking(False)

    # 가변 파라미터 (핫키로 수정)
    yaw_offset = YAW_OFFSET_DEG_INIT
    th_low = TH_LOW_INIT
    th_high = TH_HIGH_INIT

    last_angle = None
    last_rx_ms = 0
    last_change_ms = 0
    show_help = False

    clock = pygame.time.Clock()
    print("[INFO] DOA overlay with live-tuning hotkeys running…  (H for help, ESC to quit)")
    try:
        while True:
            # 이벤트
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    return
                if e.type == pygame.KEYDOWN:
                    mods = pygame.key.get_mods()
                    mult = STEP_MULT_SHIFT if (mods & pygame.KMOD_SHIFT) else 1.0

                    if e.key == pygame.K_ESCAPE:
                        return
                    elif e.key == pygame.K_h:
                        show_help = not show_help
                        last_change_ms = pygame.time.get_ticks()
                    elif e.key == pygame.K_LEFT:
                        yaw_offset -= YAW_STEP * mult
                        last_change_ms = pygame.time.get_ticks()
                    elif e.key == pygame.K_RIGHT:
                        yaw_offset += YAW_STEP * mult
                        last_change_ms = pygame.time.get_ticks()
                    elif e.key == pygame.K_UP:
                        if mods & pygame.KMOD_SHIFT:
                            th_high += TH_HIGH_STEP * mult
                        else:
                            th_low += TH_LOW_STEP * mult
                        # 경계 정리
                        th_low = max(0.0, min(th_low, th_high - 1e-3))
                        last_change_ms = pygame.time.get_ticks()
                    elif e.key == pygame.K_DOWN:
                        if mods & pygame.KMOD_SHIFT:
                            th_high -= TH_HIGH_STEP * mult
                        else:
                            th_low -= TH_LOW_STEP * mult
                        th_high = max(th_high, th_low + 1e-3)
                        th_low = max(0.0, th_low)
                        last_change_ms = pygame.time.get_ticks()

            # 수신
            try:
                data = sock.recv(4096)
                a = parse_angle_line(data)
                if a is not None:
                    last_angle = a
                    last_rx_ms = pygame.time.get_ticks()
            except BlockingIOError:
                pass

            # 카메라
            frame = picamera2.capture_array()
            surf = pygame.image.frombuffer(frame.tobytes(), CAM_PREVIEW_SIZE, "RGB")
            if (sw, sh) != CAM_PREVIEW_SIZE:
                surf = pygame.transform.smoothscale(surf, (sw, sh))
            screen.blit(surf, (0, 0))

            # 오버레이: 최근 SHOW_MS 내에 수신된 경우만
            now = pygame.time.get_ticks()
            if last_angle is not None and (now - last_rx_ms) <= SHOW_MS:
                side, zone = classify_zone(last_angle, yaw_offset, th_low, th_high)
                arrow = rotate_for_side_zone(arrow_base, side, zone)
                x, y = place_for_side_zone(sw, sh, arrow_size, side, zone)
                screen.blit(arrow, (x, y))

            # HUD: 값 변경 직후 HUD_MS 동안 보이거나, 도움말 on이면 지속 표시
            if show_help or (now - last_change_ms) <= HUD_MS:
                draw_hud(screen, font, yaw_offset, th_low, th_high, show_help, sw, sh, last_angle)

            pygame.display.flip()
            clock.tick(30)
    finally:
        pygame.quit()
        picamera2.close()


if __name__ == "__main__":
    main()
