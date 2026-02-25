import io
import hashlib
import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import os
import json
import shutil
from datetime import datetime
import pandas as pd

# torch는 ultralytics가 내부적으로 사용합니다.
# (기존 코드에서는 fallback에서 매번 import해서 오버헤드가 있었음)
try:
    import torch
    _TORCH_OK = True
except Exception:
    torch = None
    _TORCH_OK = False
try:
    cv2.setNumThreads(1)  # OpenCV가 코어 다 쓰는 것 방지
except Exception:
    pass

if _TORCH_OK:
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass
# --- [1] Page Config & CSS ---
st.set_page_config(page_title="TFCP Quantitative Analysis System", page_icon="🔬", layout="wide")

st.markdown("""
    <style>
        .main { background-color: #f8f9fa; }
        .stButton>button { width: 100%; border-radius: 6px; font-weight: 500; font-family: 'Helvetica', sans-serif; }
        .metric-card {
            background-color: white; border: 1px solid #e1e4e8; border-radius: 10px;
            padding: 20px; margin-bottom: 12px; box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        }
        .status-safe { color: #28a745; font-weight: 700; }
        .status-cont { color: #dc3545; font-weight: 700; }
        .status-warn { color: #fd7e14; font-weight: 700; }
        .header-text { font-family: 'Helvetica Neue', sans-serif; font-weight: 700; color: #1f2937; }
    </style>
""", unsafe_allow_html=True)

# --- [2] Storage & Model Setup ---
SAVE_ROOT = "TFCP_Data"
IMG_DIR = os.path.join(SAVE_ROOT, "raw_images")
LOG_DIR = os.path.join(SAVE_ROOT, "analysis_logs")

os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

APP_VERSION = "10.2.2"  # 내부 로직/로그 버전 태그

@st.cache_resource
def load_model():
    if os.path.exists('best.pt'):
        try:
            return YOLO('best.pt')
        except Exception:
            return None
    return None

model = load_model()

def _get_secret(key: str, default: str = "") -> str:
    """Streamlit secrets가 없는 로컬 환경/배포 환경을 모두 고려한 헬퍼."""
    try:
        return st.secrets.get(key, default)
    except Exception:
        return default

# 하드코딩된 키는 보안상 취약 → secrets/env 우선, 없으면 기존 값으로 fallback
ADMIN_KEY = _get_secret("TFCP_ADMIN_KEY", "") or os.environ.get("TFCP_ADMIN_KEY", "tfcp2026")

# --- [3] Visualization Helper (Server-Native Font) ---
@st.cache_resource
def get_custom_font(size=20, bold=False):
    """
    Arial과 가장 유사한 Liberation Sans를 우선 사용.
    (Streamlit Cloud/Linux에서 Arial.ttf는 거의 없어서 대체 폰트가 필요)
    """
    # (선택) 레포에 fonts 폴더를 만들고 넣어두면 가장 확실하게 동일 폰트가 나옴
    local_candidates = [
        "fonts/LiberationSans-Bold.ttf" if bold else "fonts/LiberationSans-Regular.ttf",
        # Arial 파일은 라이선스 문제될 수 있어(특히 배포) 로컬에 직접 넣는 건 비권장
        "fonts/arialbd.ttf" if bold else "fonts/arial.ttf",
    ]

    # 시스템 폰트 후보 (Streamlit Cloud가 어떤 Debian을 쓰든 경로가 다를 수 있어 둘 다 체크)
    system_candidates = [
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]

    for font_path in local_candidates + system_candidates:
        if os.path.exists(font_path):
            try:
                return ImageFont.truetype(font_path, size)
            except Exception:
                continue

    return ImageFont.load_default()

def draw_smart_annotations(img_bgr, reports, show_box_labels=True, show_global_label=False):
    """
    - Arial 유사(Liberation Sans) 폰트 사용
    - 라벨 배경은 검정색
    - 라벨 글자는 상태색(빨강/주황/초록)
    - 샘플 이미지처럼: 보통 박스 좌상단에 "Contaminated" 같은 상태 라벨을 표시
    """

    if img_bgr is None:
        return None

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb).convert("RGBA")
    overlay = Image.new("RGBA", pil_img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    h, w = img_bgr.shape[:2]

    # 스케일 (작은 이미지에서도 너무 작아지지 않게)
    scale = max(w, h) / 1200.0
    line_width = max(1, int(3 * scale))
    font_size = max(14, int(24 * scale))
    pad = max(4, int(6 * scale))

    font = get_custom_font(font_size, bold=False)

    def status_color(status: str):
        if status == "CONTAMINATED":
            return (220, 53, 69)   # red
        if status == "RECHECK REQUIRED":
            return (253, 126, 20)  # orange
        return (40, 167, 69)       # green

    def status_text(status: str):
        if status == "CONTAMINATED":
            return "Contaminated"
        if status == "RECHECK REQUIRED":
            return "Recheck"
        return "Safe"

    def measure_text(text: str):
        try:
            bbox = font.getbbox(text)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except Exception:
            tw, th = draw.textsize(text, font=font)
        return tw, th

    def draw_label(x, y, text, fg_rgb):
        # 라벨 배경: 검정색(약간 반투명)
        bg_rgba = (0, 0, 0, 220)

        tw, th = measure_text(text)
        bw, bh = tw + 2 * pad, th + 2 * pad

        # 화면 밖으로 나가지 않게 보정
        x = max(0, min(x, w - bw - 1))
        y = max(0, min(y, h - bh - 1))

        draw.rectangle([x, y, x + bw, y + bh], fill=bg_rgba)
        draw.text((x + pad, y + pad), text, font=font, fill=fg_rgb + (255,))

    # (선택) 전체 상태 라벨을 좌상단에 1개만 표시하고 싶으면 사용
    if show_global_label and reports:
        overall = "SAFE"
        for r in reports:
            s = r.get("status", "SAFE")
            if s == "CONTAMINATED":
                overall = "CONTAMINATED"
                break
            elif s == "RECHECK REQUIRED":
                overall = "RECHECK REQUIRED"

        draw_label(8, 8, status_text(overall), status_color(overall))

    # contaminated가 마지막에 그려지도록 정렬(겹칠 때 위로 보이게)
    def priority(r):
        s = r.get("status", "SAFE")
        return 2 if s == "CONTAMINATED" else 1 if s == "RECHECK REQUIRED" else 0

    reports_sorted = sorted(reports or [], key=priority)

    for r in reports_sorted:
        box = r.get("box", None)
        if not box or len(box) != 4:
            continue

        x1, y1, x2, y2 = map(int, box)

        # clamp
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w - 1))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h - 1))
        if x2 <= x1 or y2 <= y1:
            continue

        status = r.get("status", "SAFE")
        col = status_color(status)

        # 박스
        draw.rectangle([x1, y1, x2, y2], outline=col + (255,), width=line_width)

        # 박스 라벨(샘플처럼 Contaminated 등만 표시)
        if show_box_labels:
            txt = status_text(status)

            # 박스 위에 라벨을 올리되, 공간이 없으면 박스 안쪽에
            tw, th = measure_text(txt)
            label_h = th + 2 * pad
            ly = y1 - label_h
            if ly < 0:
                ly = y1 + 2

            draw_label(x1, ly, txt, col)

    final_img = Image.alpha_composite(pil_img, overlay)
    return np.array(final_img.convert("RGB"))
# --- [4] Core Analysis Engine (v10.2.2 Logic) ---

def apply_gamma_correction(image, gamma=0.8):
    # [FIX] gamma가 0/음수면 오류 → 최소값 보호
    gamma = float(gamma)
    if gamma <= 0:
        gamma = 0.8
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def standardize_image_size(img, target_width=1280, target_height=960, bg_value=0):
    """
    [FIX] bg_value 옵션 추가 (기본 black=0). UI 취향에 따라 255(white)도 가능.
    """
    h, w = img.shape[:2]
    scale = min(target_width / w, target_height / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LANCZOS4)
    new_img = np.full((target_height, target_width, 3), bg_value, dtype=np.uint8)
    top, left = (target_height - nh) // 2, (target_width - nw) // 2
    new_img[top:top + nh, left:left + nw] = resized
    return new_img

def downscale_if_needed(img, max_side=1280):
    """
    큰 이미지(예: 폰 사진)를 그대로 처리하면 매우 무거워짐 → 자동 축소
    max_side: 긴 변 기준 최대 픽셀
    """
    h, w = img.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return img, 1.0

    scale = max_side / float(m)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale

# --- [PREPROC] Conditional Blue Suppression (only when NO green & strong blue cast) ---

def _resize_for_metrics(img_bgr, max_side=320):
    """
    Fast downscale for metric calculation (performance-friendly).
    Returns: (small_img, scale)
    """
    h, w = img_bgr.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return img_bgr, 1.0
    scale = max_side / float(m)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    small = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    return small, scale


def analyze_green_blue_conditions(
    img_bgr,
    sample_max_side=320,
    min_v=40,
    min_g=60,
    green_dom_diff=30,
    b_floor_percentile=30,
):
    """
    Returns quick global metrics used to decide whether to suppress blue cast.

    green_ratio:
      fraction of pixels that look "green-dominant" (strong green fluorescence signature)
      definition: G is clearly higher than both B and R.

    blue_dom90:
      90th percentile of (B - max(G, R)) on valid pixels
      -> high value means "blue dominates" (blue cast/backlight)

    b90:
      90th percentile of B on valid pixels
      -> prevents triggering in dark scenes

    b_floor:
      percentile of B (on valid pixels) used as "blue minimum" (ImageJ-like baseline)
    """
    small, _ = _resize_for_metrics(img_bgr, max_side=sample_max_side)

    b, g, r = cv2.split(small.astype(np.int16))
    v = np.maximum(np.maximum(b, g), r)  # brightness proxy
    valid = v > int(min_v)

    valid_n = int(np.count_nonzero(valid))
    if valid_n <= 0:
        return {
            "valid_px": 0,
            "green_ratio": 0.0,
            "blue_dom90": 0.0,
            "b90": 0.0,
            "b_floor": float(np.percentile(b.astype(np.float32), b_floor_percentile)),
        }

    # green-dominant pixels (strong green fluorescence-like signature)
    green_dom = (g > int(min_g)) & ((g - np.maximum(b, r)) > int(green_dom_diff)) & valid
    green_ratio = float(np.count_nonzero(green_dom) / valid_n)

    # blue cast strength
    blue_over = (b - np.maximum(g, r))
    blue_dom90 = float(np.percentile(blue_over[valid], 90))
    b90 = float(np.percentile(b[valid], 90))

    # blue baseline ("minimum") estimate (computed on small image for performance)
    b_floor = float(np.percentile(b[valid].astype(np.float32), b_floor_percentile))

    return {
        "valid_px": valid_n,
        "green_ratio": green_ratio,
        "blue_dom90": blue_dom90,
        "b90": b90,
        "b_floor": b_floor,
    }


def apply_blue_minimum_raise(img_bgr, b_floor, strength=0.8, cap_delta=25):
    """
    ImageJ 'Color Balance: Blue minimum 증가'에 가까운 효과:
    - B 채널에서 baseline(b_floor)을 strength 만큼 빼줌 (clip)
    - (선택) B가 G보다 과도하게 커지지 않도록 cap (cyan은 G도 같이 높아서 상대적으로 보존됨)

    NOTE: only modifies Blue channel
    """
    b, g, r = cv2.split(img_bgr.astype(np.float32))

    b2 = b - (float(strength) * float(b_floor))
    b2 = np.clip(b2, 0.0, 255.0)

    # cyan 보존용 안전장치(파란 백라이트처럼 B만 튀는 걸 눌러줌)
    if cap_delta is not None and cap_delta >= 0:
        cap = g + float(cap_delta)
        b2 = np.minimum(b2, cap)

    out = cv2.merge([b2.astype(np.uint8), g.astype(np.uint8), r.astype(np.uint8)])
    return out


def conditional_blue_suppression(
    img_bgr,
    enabled=True,
    # condition thresholds
    green_ratio_thr=0.01,      # green-dominant pixel ratio threshold (0.2%)
    blue_dom90_thr=12.0,        # blue dominance threshold
    b90_thr=140.0,              # blue brightness threshold (avoid dark images)
    # correction params
    b_floor_percentile=15,
    cap_delta=25,
    strength_min=0.55,
    strength_max=1.0,
    severity_scale=60.0,
):
    """
    Apply blue suppression ONLY when:
      - green fluorescence is absent (green_ratio < green_ratio_thr)
      - blue cast is strong (blue_dom90 >= thr AND b90 >= thr)

    Returns: (img_out, meta_dict)
    """
    meta = {"enabled": bool(enabled), "applied": False}
    if not enabled:
        return img_bgr, meta

    m = analyze_green_blue_conditions(
        img_bgr,
        sample_max_side=320,
        min_v=40,
        min_g=60,
        green_dom_diff=30,
        b_floor_percentile=b_floor_percentile,
    )
    meta.update(m)

    green_present = (m["green_ratio"] >= float(green_ratio_thr))
    blue_cast = (m["blue_dom90"] >= float(blue_dom90_thr)) and (m["b90"] >= float(b90_thr))
    meta["green_present"] = green_present
    meta["blue_cast"] = blue_cast

    if (not green_present) and blue_cast:
        # severity -> strength (mild cast: gentle, strong cast: stronger)
        sev = (m["blue_dom90"] - float(blue_dom90_thr)) / float(severity_scale if severity_scale > 0 else 60.0)
        sev = float(np.clip(sev, 0.0, 1.0))
        strength = float(strength_min + (strength_max - strength_min) * sev)

        out = apply_blue_minimum_raise(
            img_bgr,
            b_floor=m["b_floor"],
            strength=strength,
            cap_delta=cap_delta,
        )

        meta["applied"] = True
        meta["severity"] = sev
        meta["strength"] = strength
        meta["cap_delta"] = cap_delta
        meta["b_floor_percentile"] = b_floor_percentile
        meta["green_ratio_thr"] = green_ratio_thr
        meta["blue_dom90_thr"] = blue_dom90_thr
        meta["b90_thr"] = b90_thr
        return out, meta

    meta["green_ratio_thr"] = green_ratio_thr
    meta["blue_dom90_thr"] = blue_dom90_thr
    meta["b90_thr"] = b90_thr
    meta["b_floor_percentile"] = b_floor_percentile
    meta["cap_delta"] = cap_delta
    return img_bgr, meta

def rescale_reports(reports, inv_scale: float):
    """
    downscale된 이미지(image_proc) 기준 box 좌표를 원본(image) 좌표로 복구.
    inv_scale = 1.0 / scale
    """
    if not reports:
        return []
    out = []
    for r in reports:
        rr = dict(r)
        x1, y1, x2, y2 = rr.get("box", [0, 0, 0, 0])
        rr["box"] = [
            int(x1 * inv_scale),
            int(y1 * inv_scale),
            int(x2 * inv_scale),
            int(y2 * inv_scale),
        ]
        out.append(rr)
    return out
    
def calculate_iou(box1, box2):
    b1, b2 = box1.flatten(), box2.flatten()
    ix1, iy1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    ix2, iy2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0

def filter_nested_boxes(boxes):
    if len(boxes) == 0:
        return []
    # 높은 conf부터
    sorted_indices = np.argsort([b.conf.item() for b in boxes])[::-1]
    keep_indices = []
    for i in sorted_indices:
        box_a = boxes[i].xyxy[0].cpu().numpy().flatten()
        keep = True
        for j in keep_indices:
            box_b = boxes[j].xyxy[0].cpu().numpy().flatten()
            if calculate_iou(box_a, box_b) > 0.3:
                keep = False
                break
            ix1, iy1 = max(box_a[0], box_b[0]), max(box_a[1], box_b[1])
            ix2, iy2 = min(box_a[2], box_b[2]), min(box_a[3], box_b[3])
            inter_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
            if area_a > 0 and (inter_area / area_a) > 0.7:
                keep = False
                break
        if keep:
            keep_indices.append(i)
    return [boxes[idx] for idx in keep_indices]

def detect_particles_heuristically(img):
    """
    YOLO가 없거나 결과가 0개일 때 fallback.
    - [FIX] torch import를 루프 안에서 하지 않도록 변경(오버헤드 감소)
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask_o = cv2.inRange(hsv, np.array([0, 40, 40]), np.array([60, 255, 255]))
    mask_c = cv2.inRange(hsv, np.array([80, 30, 30]), np.array([165, 255, 255]))
    combined = cv2.bitwise_or(mask_o, mask_c)
    kernel = np.ones((25, 25), np.uint8)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    found_boxes = []

    class FakeBox:
        def __init__(self, coords):
            # torch가 없으면 최소 동작만 보장 (다만 ultralytics/YOLO를 쓰는 환경이면 torch는 거의 항상 있음)
            if _TORCH_OK:
                self.xyxy = torch.tensor([coords], dtype=torch.float32)
                self.conf = torch.tensor([0.15])
            else:
                # torch 없음 → 동일 인터페이스를 흉내내는 최소 래퍼
                self.xyxy = np.array([coords], dtype=np.float32)
                self.conf = np.array([0.15], dtype=np.float32)

    for cnt in contours:
        if cv2.contourArea(cnt) > 3000:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w) / h if h > 0 else 0
            if 0.2 < aspect_ratio < 5.0:
                found_boxes.append(FakeBox([x, y, x + w, y + h]))
    return found_boxes

def _box_to_xyxy_int(box):
    """
    YOLO Box(torch) / FakeBox(torch or numpy) 모두 처리.
    """
    try:
        coords = box.xyxy[0].cpu().numpy().flatten()
    except Exception:
        coords = np.array(box.xyxy[0]).flatten()
    x1, y1, x2, y2 = map(int, coords)
    return x1, y1, x2, y2
def _resize_for_metrics(img_bgr, max_side=320):
    h, w = img_bgr.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return img_bgr
    scale = max_side / float(m)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)


def analyze_green_blue(
    img_bgr,
    sample_max_side=320,
    min_v=40,
    green_min=60,
    green_dom_diff=30,
    b_floor_percentile=15
):
    """
    목적:
      - green 형광이 '의미있게' 존재하는지(=green-dominant 픽셀 비율) 추정
      - blue cast가 강한지(blue dominance) 추정
      - blue channel black point(b_floor) 추정
    """
    small = _resize_for_metrics(img_bgr, sample_max_side)
    b, g, r = cv2.split(small.astype(np.int16))

    v = np.maximum(np.maximum(b, g), r)
    valid = v > min_v
    valid_n = int(np.count_nonzero(valid))
    if valid_n <= 0:
        return {"valid_px": 0, "green_ratio": 0.0, "blue_dom90": 0.0, "b90": 0.0, "b_floor": 0.0}

    green_dom = (g > green_min) & ((g - np.maximum(b, r)) > green_dom_diff) & valid
    green_ratio = float(np.count_nonzero(green_dom) / valid_n)

    blue_over = b - np.maximum(g, r)
    blue_dom90 = float(np.percentile(blue_over[valid], 90))
    b90 = float(np.percentile(b[valid], 90))

    b_floor = float(np.percentile(b[valid].astype(np.float32), b_floor_percentile))

    return {
        "valid_px": valid_n,
        "green_ratio": green_ratio,
        "blue_dom90": blue_dom90,
        "b90": b90,
        "b_floor": b_floor
    }


def apply_blue_kill(img_bgr, b_floor, gain=0.10, strength=1.0, cap_delta=5):
    """
    ImageJ에서 blue minimum 올리는 느낌(blue black point raise + 강한 감쇠)을 흉내.
      B' = max(0, B - strength*b_floor) * gain
      그리고 B' <= G + cap_delta 로 추가로 눌러버림

    - gain을 더 작게(0.10 -> 0.05) 하면 blue가 더 '존나' 죽음
    - cap_delta를 더 작게(5 -> 0) 하면 blue가 더 눌림
    """
    b, g, r = cv2.split(img_bgr.astype(np.float32))
    b2 = np.maximum(0.0, b - float(b_floor) * float(strength)) * float(gain)
    if cap_delta is not None:
        b2 = np.minimum(b2, g + float(cap_delta))

    out = cv2.merge([
        np.clip(b2, 0, 255).astype(np.uint8),
        np.clip(g, 0, 255).astype(np.uint8),
        np.clip(r, 0, 255).astype(np.uint8),
    ])
    return out


def make_green_signal_mask(roi_bgr, valid_u8, containment_u8, g_min=120, diff_min=40):
    """
    blue를 죽인 상태(blue-kill mode)에서, 'cyan(원래)'을 'green'으로 읽기 위한 마스크.
    - green-dominant: G가 충분히 크고, (G - max(B,R))가 diff_min 이상
    - valid_u8 + containment_u8 로 제한
    """
    b, g, r = cv2.split(roi_bgr.astype(np.int16))

    green_dom = (g >= g_min) & ((g - np.maximum(b, r)) >= diff_min)
    mask = (green_dom.astype(np.uint8) * 255)
    mask = cv2.bitwise_and(mask, valid_u8)
    mask = cv2.bitwise_and(mask, containment_u8)

    # 노이즈 조금 정리
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    # intensity map (0~100): green residual 느낌
    green_res = np.clip(
        g.astype(np.float32) - 0.5 * b.astype(np.float32) - 0.5 * r.astype(np.float32),
        0.0, 255.0
    )
    intensity_map = np.clip(green_res / 2.55, 0.0, 100.0)

    # 아주 강한(포화급) green 픽셀
    sat = (g >= 230) & ((g - np.maximum(b, r)) >= 60)
    sat_mask = (sat.astype(np.uint8) * 255)
    sat_mask = cv2.bitwise_and(sat_mask, mask)
    saturated_pixels = int(cv2.countNonZero(sat_mask))

    return mask, intensity_map, saturated_pixels

def process_frame(img, gamma=0.8, model_conf=0.10, model_iou=0.45, blue_kill_enabled=True):
    """
    반환:
      final_img, reports, pre
    """

    # 0) gamma 보정 (원본 기반)
    img_corr = apply_gamma_correction(img, gamma=gamma)

    # 1) "green 형광이 없을 때만" blue kill 적용 여부 판단
    #    (임계값은 여기 숫자만 조절하면 됨)
    m = analyze_green_blue(img_corr, b_floor_percentile=15)

    GREEN_RATIO_THR = 0.07     # green-dominant 픽셀 비율이 7% 이상이면 "green 형광 존재"로 취급
    BLUE_DOM90_THR = 15        # blue dominance가 이 이상이면 blue cast 강함
    B90_THR = 110              # blue 자체도 밝아야 함(그냥 파란 점 몇개 수준은 제외)

    green_present = (m["green_ratio"] > GREEN_RATIO_THR)
    blue_cast = (m["blue_dom90"] > BLUE_DOM90_THR) and (m["b90"] > B90_THR)

    pre = {
        "applied": False,
        "mode": "HSV_CYAN",         # 기본 모드
        "green_present": bool(green_present),
        "blue_cast": bool(blue_cast),
        "metrics": m,
        "params": {}
    }

    # 2) 분석용 이미지 결정 (img_anl)
    img_anl = img_corr

    if blue_kill_enabled and (not green_present) and blue_cast:
        # 여기 파라미터가 "파란색 더 줄이기" 핵심 노브임
        BLUEKILL_GAIN = 0.01        # ↓ 더 줄이면(0.10->0.05) blue 더 죽음
        BLUEKILL_STRENGTH = 1.0     # ↑ 올리면 더 죽음(1.0->1.2)
        BLUEKILL_CAP_DELTA = 5      # ↓ 줄이면 더 죽음(5->0)

        img_anl = apply_blue_kill(
            img_corr,
            b_floor=m["b_floor"],
            gain=BLUEKILL_GAIN,
            strength=BLUEKILL_STRENGTH,
            cap_delta=BLUEKILL_CAP_DELTA
        )
        pre["applied"] = True
        pre["mode"] = "GREEN_SIGNAL"
        pre["params"] = {
            "gain": BLUEKILL_GAIN,
            "strength": BLUEKILL_STRENGTH,
            "cap_delta": BLUEKILL_CAP_DELTA,
            "b_floor": m["b_floor"]
        }

    # 3) HSV는 분석용 이미지(img_anl) 기준
    hsv = cv2.cvtColor(img_anl, cv2.COLOR_BGR2HSV)
    img_h, img_w = img_anl.shape[:2]

    # 4) YOLO는 원본 gamma 보정(img_corr) 기준으로 추론 (모델 도메인 유지)
    ai_raw_boxes = []
    if model is not None:
        try:
            results = model.predict(
                source=img_corr,
                conf=model_conf,
                iou=model_iou,
                imgsz=640,
                verbose=False
            )
            ai_raw_boxes = filter_nested_boxes(results[0].boxes)
        except Exception:
            ai_raw_boxes = []

    combined_boxes = []

    ORANGE_LO = np.array([0, 35, 35])
    ORANGE_HI = np.array([60, 255, 255])
    CYAN_LO = np.array([80, 30, 30])
    CYAN_HI = np.array([165, 255, 255])

    for box in ai_raw_boxes:
        x1, y1, x2, y2 = _box_to_xyxy_int(box)
        if (x2 - x1) < 50 or (y2 - y1) < 50:
            continue

        if pre["mode"] == "GREEN_SIGNAL":
            # blue-kill 모드에서는 "색 필터로 박스를 버리면" 놓치기 쉬움 → YOLO 박스를 그대로 사용
            combined_boxes.append((box, "AI"))
            continue

        roi_hsv = hsv[max(0, y1):min(img_h, y2), max(0, x1):min(img_w, x2)]
        if roi_hsv.size == 0:
            continue

        mask_o = cv2.inRange(roi_hsv, ORANGE_LO, ORANGE_HI)
        mask_c = cv2.inRange(roi_hsv, CYAN_LO, CYAN_HI)
        color_px = int(cv2.countNonZero(mask_o)) + int(cv2.countNonZero(mask_c))

        if color_px > 200:
            combined_boxes.append((box, "AI"))

    # YOLO 박스가 없으면 fallback
    if not combined_boxes:
        for cv_box in detect_particles_heuristically(img_anl):
            combined_boxes.append((cv_box, "CV_BACKUP"))

    reports = []

    MIN_BODY_PX = 120
    MIN_ORANGE_PCT = 3.0
    MIN_COMP_AREA = 20

    for i, (box, method) in enumerate(combined_boxes):
        x1, y1, x2, y2 = _box_to_xyxy_int(box)

        pad = int((x2 - x1) * 0.15)
        nx1 = max(0, x1 - pad)
        ny1 = max(0, y1 - pad)
        nx2 = min(img_w, x2 + pad)
        ny2 = min(img_h, y2 + pad)

        roi_hsv = hsv[ny1:ny2, nx1:nx2]
        roi_img = img_anl[ny1:ny2, nx1:nx2]
        if roi_hsv.size == 0:
            continue

        valid_mask = (roi_hsv[:, :, 1] > 25) & (roi_hsv[:, :, 2] > 25)
        valid_u8 = (valid_mask.astype(np.uint8) * 255)

        # orange body
        mask_orange = cv2.inRange(roi_hsv, np.array([0, 30, 30]), np.array([60, 255, 255]))
        mask_orange_valid = cv2.bitwise_and(mask_orange, valid_u8)
        mask_orange_closed = cv2.morphologyEx(
            mask_orange_valid,
            cv2.MORPH_CLOSE,
            np.ones((5, 5), np.uint8)
        )

        contours, _ = cv2.findContours(mask_orange_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        mask_particle_body = np.zeros_like(mask_orange_closed)
        for cnt in contours:
            if cv2.contourArea(cnt) > MIN_COMP_AREA:
                cv2.drawContours(mask_particle_body, [cnt], -1, 255, -1)

        body_area_px = int(cv2.countNonZero(mask_particle_body))

        box_area = int((nx2 - nx1) * (ny2 - ny1))
        orange_area_pct = (body_area_px / box_area) * 100.0 if box_area > 0 else 0.0

        if body_area_px < MIN_BODY_PX or orange_area_pct < MIN_ORANGE_PCT:
            status = "RECHECK REQUIRED"
            phi = 0.0
            signal_area = 0.0
            avg_int = 0.0
        else:
            mask_containment_zone = cv2.dilate(mask_particle_body, np.ones((3, 3), np.uint8), iterations=1)

            if pre["mode"] == "GREEN_SIGNAL":
                # === 핵심: blue kill 이후 cyan을 green으로 읽음 ===
                GREEN_GMIN = 120     # 너무 빡세면 110/100으로 낮춰
                GREEN_DIFF = 40      # 너무 빡세면 35/30으로 낮춰

                mask_sig, intensity_map, saturated_pixels = make_green_signal_mask(
                    roi_img,
                    valid_u8,
                    mask_containment_zone,
                    g_min=GREEN_GMIN,
                    diff_min=GREEN_DIFF
                )

                sig_px = int(cv2.countNonZero(mask_sig))
                signal_area = (sig_px / body_area_px) * 100.0 if body_area_px > 0 else 0.0
                avg_int = float(np.mean(intensity_map[mask_sig > 0])) if sig_px > 0 else 0.0

                phi = signal_area * (avg_int / 10.0)

                status = "CONTAMINATED" if (phi > 5.0 or saturated_pixels > 20) else "SAFE"
                if status == "CONTAMINATED" and phi < 5.0:
                    phi = 99.9

            else:
                # === 기존 HSV cyan 로직 ===
                mask_cyan_candidate = cv2.inRange(roi_hsv, CYAN_LO, CYAN_HI)
                mask_cyan_candidate = cv2.bitwise_and(mask_cyan_candidate, valid_u8)
                mask_cyan = cv2.bitwise_and(mask_cyan_candidate, mask_containment_zone)

                b_ch, g_ch, r_ch = cv2.split(roi_img.astype(np.float32))
                is_glare = (g_ch > 200) & (b_ch > 200) & (r_ch > 200)
                is_saturated_cyan = (g_ch > 200) & (b_ch > 200) & (r_ch < 200)

                mask_saturated_valid = (is_saturated_cyan.astype(np.uint8) * 255)
                mask_saturated_valid = cv2.bitwise_and(mask_saturated_valid, mask_containment_zone)
                saturated_pixels = int(cv2.countNonZero(mask_saturated_valid))

                intensity_raw = np.where(
                    is_glare, 0.0,
                    np.where(
                        is_saturated_cyan,
                        ((g_ch + b_ch) / 2.0 - r_ch * 0.8),
                        ((g_ch + b_ch) / 2.0 - r_ch * 1.7)
                    )
                )
                intensity_map = np.clip(intensity_raw, 0.0, 100.0)

                cyan_px = int(cv2.countNonZero(mask_cyan))
                signal_area = (cyan_px / body_area_px) * 100.0 if body_area_px > 0 else 0.0
                avg_int = float(np.mean(intensity_map[mask_cyan > 0])) if cyan_px > 0 else 0.0

                phi = signal_area * (avg_int / 10.0)

                status = "CONTAMINATED" if (phi > 5.0 or saturated_pixels > 20) else "SAFE"
                if status == "CONTAMINATED" and phi < 5.0:
                    phi = 99.9

        reports.append({
            "id": i,
            "status": status,
            "phi": float(round(phi, 2)),
            "cyan": float(round(signal_area, 2)),  # GREEN_SIGNAL 모드에서도 지표 필드명 유지
            "orange": float(round(orange_area_pct, 2)),
            "box": [int(nx1), int(ny1), int(nx2), int(ny2)],
            "method": method,
            "signal_mode": pre["mode"],            # 디버그/추적용
        })

    final_img = draw_smart_annotations(img_anl.copy(), reports)
    return final_img, reports, pre
    
# --- UI (Admin) ---
def render_admin_page():
    st.markdown("<h2 class='header-text'>Research Data Management Center</h2>", unsafe_allow_html=True)

    log_files = sorted([f for f in os.listdir(LOG_DIR) if f.endswith('.json')], reverse=True)
    if not log_files:
        st.info("No data available.")
        return

    if 'current_log_file' not in st.session_state:
        st.session_state.current_log_file = log_files[0]
    if st.session_state.current_log_file not in log_files:
        st.session_state.current_log_file = log_files[0]

    current_idx = log_files.index(st.session_state.current_log_file)
    c1, c2, c3 = st.columns([1, 4, 1])
    with c1:
        if st.button("◀ PREV"):
            st.session_state.current_log_file = log_files[max(0, current_idx - 1)]
            st.rerun()
    with c3:
        if st.button("NEXT ▶"):
            st.session_state.current_log_file = log_files[min(len(log_files) - 1, current_idx + 1)]
            st.rerun()
    with c2:
        def update_index():
            st.session_state.current_log_file = st.session_state.log_selector

        st.selectbox("Select Log", log_files, index=current_idx, key='log_selector', on_change=update_index,
                     label_visibility="collapsed")

        bc1, bc2 = st.columns(2)
        with bc1:
            if st.button("📦 Archive (ZIP)", use_container_width=True):
                shutil.make_archive("TFCP_Dataset", 'zip', SAVE_ROOT)
                with open("TFCP_Dataset.zip", "rb") as fp:
                    st.download_button("Download ZIP", fp, "TFCP_Dataset.zip", "application/zip")
        with bc2:
            if st.button("🗑️ Delete", type="primary"):
                try:
                    f_path = os.path.join(LOG_DIR, st.session_state.current_log_file)
                    with open(f_path, 'r') as f:
                        d = json.load(f)
                    if os.path.exists(f_path):
                        os.remove(f_path)
                    img_p = os.path.join(IMG_DIR, d.get('filename', ''))
                    if img_p and os.path.exists(img_p):
                        os.remove(img_p)
                    st.success("Deleted.")
                    del st.session_state.current_log_file
                    st.rerun()
                except Exception as e:
                    st.error(f"Delete failed: {e}")

    log_path = os.path.join(LOG_DIR, st.session_state.current_log_file)
    try:
        with open(log_path, 'r') as f:
            data = json.load(f)

        img_path = os.path.join(IMG_DIR, data.get('filename', ''))
        particles = data.get('particles', data.get('reports', []))

        if os.path.exists(img_path):
            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                st.error("Image read failed (cv2.imread returned None).")
                return

            img_corrected = apply_gamma_correction(img_bgr, gamma=0.8)

            # Draw using PIL
            draw_img = draw_smart_annotations(img_corrected.copy(), particles)
            display_img = standardize_image_size(draw_img, 1280, 960)
            st.image(display_img, caption=f"Analyzed: ...", width=800)

            with st.expander("➕ Manual Region Injection (Slider)", expanded=False):
                st.info("Inject ROI manually.")
                h, w = img_corrected.shape[:2]
                mc1, mc2 = st.columns(2)
                with mc1:
                    mx1 = st.slider("X Start", 0, w, int(w * 0.3), key="mx1")
                    mx2 = st.slider("X End", 0, w, int(w * 0.7), key="mx2")
                with mc2:
                    my1 = st.slider("Y Start", 0, h, int(h * 0.3), key="my1")
                    my2 = st.slider("Y End", 0, h, int(h * 0.7), key="my2")

                # [FIX] slider에서 역전 입력 방지
                x1, x2 = sorted([mx1, mx2])
                y1, y2 = sorted([my1, my2])

                preview = img_corrected.copy()
                cv2.rectangle(preview, (x1, y1), (x2, y2), (255, 0, 255), 4)
                preview_rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
                st.image(standardize_image_size(preview_rgb, 800, 600), caption="Preview", width=800)

                if st.button("✅ Inject"):
                    if (x2 - x1) < 5 or (y2 - y1) < 5:
                        st.error("ROI too small. Please select a larger region.")
                    else:
                        new_particle = {
                            "id": len(particles),
                            "box": [x1, y1, x2, y2],
                            "status": "CONTAMINATED",
                            "phi": 0,
                            "cyan": 0,
                            "orange": 0,
                            "manual": True
                        }
                        particles.append(new_particle)
                        data['particles'] = particles
                        data['reports'] = particles
                        data['reviewed'] = True  # injection은 사람이 개입한 것이므로 reviewed로 처리(원하면 False로)
                        with open(log_path, 'w') as f:
                            json.dump(data, f, indent=4)
                        st.success("Injected!")
                        st.rerun()

            with st.form("update"):
                st.markdown("#### Annotation Correction")
                new_parts = []
                cols = st.columns(2)
                for i, p in enumerate(particles):
                    with cols[i % 2]:
                        stat = p.get('status', 'SAFE')
                        cls = "status-cont" if stat == "CONTAMINATED" else "status-safe" if stat == "SAFE" else "status-warn"
                        st.markdown(f"**Area {i + 1}**: <span class='{cls}'>{stat}</span>", unsafe_allow_html=True)
                        idx = ["SAFE", "CONTAMINATED", "RECHECK REQUIRED"].index(stat) if stat in ["SAFE", "CONTAMINATED",
                                                                                                  "RECHECK REQUIRED"] else 0
                        new_stat = st.radio("Status", ["SAFE", "CONTAMINATED", "RECHECK REQUIRED"], index=idx,
                                            key=f"rad_{i}", horizontal=True)
                        p['status'] = new_stat
                        p['id'] = i
                        new_parts.append(p)

                if st.form_submit_button("Save Annotations"):
                    data['particles'] = new_parts
                    data['reports'] = new_parts
                    data['reviewed'] = True
                    with open(log_path, 'w') as f:
                        json.dump(data, f, indent=4)
                    st.success("Saved!")
                    st.rerun()
        else:
            st.error("Image missing")
    except Exception as e:
        st.error(f"Error: {e}")

# --- Main ---
if 'admin_mode' not in st.session_state:
    st.session_state['admin_mode'] = False

st.sidebar.title("Navigation")
mode = st.sidebar.radio("Go to", ["Real-time Inference", "Admin Console"])

if mode == "Admin Console":
    if not st.session_state['admin_mode']:
        pwd = st.sidebar.text_input("Access Key", type="password")
        if pwd == ADMIN_KEY:
            st.session_state['admin_mode'] = True
            st.rerun()
        elif pwd:
            st.error("Invalid Key")
    if st.session_state['admin_mode']:
        render_admin_page()

elif mode == "Real-time Inference":
    st.markdown("<h1 class='header-text'>TFCP Inference Engine</h1>", unsafe_allow_html=True)

    # --- Sidebar: settings (세련되게 컨트롤을 모아둠) ---
    st.sidebar.markdown("### Inference Settings")
    gamma = st.sidebar.slider("Gamma", 0.30, 2.00, 0.80, 0.05)
    model_conf = st.sidebar.slider("YOLO conf", 0.05, 0.50, 0.10, 0.01)
    model_iou = st.sidebar.slider("YOLO IoU", 0.10, 0.80, 0.45, 0.05)
    max_side = st.sidebar.selectbox("Downscale max side", [800, 1024, 1280, 1600], index=2)

    auto_run_on_new = st.sidebar.checkbox("Auto-run when new image arrives", value=True)
    auto_save = st.sidebar.checkbox("Auto-save run to logs", value=True)
    jpg_quality = st.sidebar.slider("Save JPG quality", 70, 100, 98, 1)

    show_box_labels = st.sidebar.checkbox("Show box labels", value=True)
    show_global_label = st.sidebar.checkbox("Show global label", value=False)

    hq_display = st.sidebar.checkbox("HQ display (no standardize resize)", value=True)

    c1, c2 = st.columns([2, 1])

    # --- session_state init ---
    if "last_img_hash" not in st.session_state:
        st.session_state.last_img_hash = None
        st.session_state.last_proc_hash = None
        st.session_state.last_orig_img = None
        st.session_state.last_proc_img = None
        st.session_state.last_scale = 1.0

        st.session_state.last_result_img = None
        st.session_state.last_reports = None

        st.session_state.last_saved_id = None
        st.session_state.last_saved_hash = None

    # --- Input UI ---
    with c1:
        img_file = st.camera_input("Acquire")
        if not img_file:
            img_file = st.file_uploader("Upload", type=["jpg", "jpeg", "png"])

    # --- If no input ---
    if not img_file:
        st.info("새 이미지를 업로드/촬영하면 분석할 수 있습니다.")
    else:
        raw = img_file.getvalue()
        img_hash = hashlib.sha1(raw).hexdigest()

        # Decode + downscale only when input changes
        if img_hash != st.session_state.last_proc_hash:
            file_bytes = np.frombuffer(raw, dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)
            if image is None:
                st.error("Load Failed")
                st.stop()

            image_proc, scale = downscale_if_needed(image, max_side=max_side)

            st.session_state.last_proc_hash = img_hash
            st.session_state.last_orig_img = image
            st.session_state.last_proc_img = image_proc
            st.session_state.last_scale = scale
        else:
            image = st.session_state.last_orig_img
            image_proc = st.session_state.last_proc_img
            scale = st.session_state.last_scale

        # Run control
        with c2:
            run_btn = st.button("Run analysis", type="primary", use_container_width=True)
            rerun_btn = st.button("Re-run (same image)", use_container_width=True)

        should_run = False
        if rerun_btn:
            should_run = True
        elif run_btn:
            should_run = True
        elif auto_run_on_new and (img_hash != st.session_state.last_img_hash):
            should_run = True

        if should_run:
            with st.spinner("Analyzing..."):
                res_img_rgb, reports, pre = process_frame(
                    image_proc,
                    gamma=gamma,
                    model_conf=model_conf,
                    model_iou=model_iou
                )

            # reports를 원본 좌표로 복구
            inv_scale = 1.0 / float(scale if scale > 0 else 1.0)
            reports_up = rescale_reports(reports, inv_scale)

            # 표시 이미지는 원본 기반으로 다시 그려서 선명도 유지
            img_corr_orig = apply_gamma_correction(image, gamma=gamma)
            annotated_rgb = draw_smart_annotations(
                img_corr_orig.copy(),
                reports_up,
                show_box_labels=show_box_labels,
                show_global_label=show_global_label
            )

            st.session_state.last_img_hash = img_hash
            st.session_state.last_result_img = annotated_rgb
            st.session_state.last_reports = reports_up

            # 저장(같은 이미지에 대해 자동 저장 반복 방지)
            if auto_save and (img_hash != st.session_state.last_saved_hash):
                now = datetime.now()
                ts_id = now.strftime("%Y%m%d_%H%M%S_%f")
                ts_display = now.strftime("%Y-%m-%d %H:%M:%S")
                fn = f"TFCP_{ts_id}"

                cv2.imwrite(
                    os.path.join(IMG_DIR, f"{fn}.jpg"),
                    image,
                    [int(cv2.IMWRITE_JPEG_QUALITY), int(jpg_quality)]
                )

                with open(os.path.join(LOG_DIR, f"{fn}.json"), "w") as f:
                    json.dump({
                        "filename": f"{fn}.jpg",
                        "timestamp": ts_display,
                        "timestamp_id": ts_id,
                        "reports": reports_up,
                        "reviewed": False,
                        "app_version": APP_VERSION,
                        "params": {
                            "gamma": float(gamma),
                            "model_conf": float(model_conf),
                            "model_iou": float(model_iou),
                            "downscale_max_side": int(max_side),
                            "input_scale": float(scale)
                        }
                    }, f, indent=4)

                st.session_state.last_saved_id = fn
                st.session_state.last_saved_hash = img_hash

              # --- Display ---
        if (st.session_state.last_result_img is not None) and (st.session_state.last_orig_img is not None):
            with c1:
                view_mode = st.radio(
                    "View mode",
                    ["Compare (Original vs Result)", "Result only", "Original only"],
                    horizontal=True,
                    label_visibility="collapsed"
                )

                orig_rgb = cv2.cvtColor(st.session_state.last_orig_img, cv2.COLOR_BGR2RGB)

                if view_mode == "Compare (Original vs Result)":
                    lc, rc = st.columns(2)
                    with lc:
                        st.image(orig_rgb, caption="Original", use_column_width=True)
                    with rc:
                        st.image(st.session_state.last_result_img, caption="Result (processed/annotated)", use_column_width=True)

                elif view_mode == "Original only":
                    st.image(orig_rgb, caption="Original", use_column_width=True)

                else:  # Result only
                    st.image(st.session_state.last_result_img, caption="Result (processed/annotated)", use_column_width=True)

            reports = st.session_state.last_reports or []

            with c2:
                st.markdown("### Metrics")







