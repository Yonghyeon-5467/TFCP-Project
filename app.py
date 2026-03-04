import streamlit as st

# ✅ 반드시 "첫 Streamlit 호출"이어야 함
st.set_page_config(
    page_title="TFCP Quantitative Analysis System",
    page_icon="🔬",
    layout="wide",
)

# ============================================================
# 0) Standard imports (set_page_config 이후)
# ============================================================
import os
import io
import json
import shutil
import hashlib
from datetime import datetime
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

# ------------------------------------------------------------
# 0-1) OpenCV import (깨지면 UI에 에러 보여주고 stop)
# ------------------------------------------------------------
CV2 = None
CV2_IMPORT_ERROR = None
try:
    import cv2 as _cv2
    CV2 = _cv2
except Exception as e:
    CV2_IMPORT_ERROR = e

# ------------------------------------------------------------
# 0-2) Optional torch
# ------------------------------------------------------------
TORCH = None
TORCH_OK = False
try:
    import torch as _torch
    TORCH = _torch
    TORCH_OK = True
except Exception:
    TORCH = None
    TORCH_OK = False


# ============================================================
# 1) UI Style
# ============================================================
st.markdown(
    """
    <style>
        .main { background-color: #f8f9fa; }
        .stButton>button { width: 100%; border-radius: 6px; font-weight: 600; }
        .metric-card {
            background-color: white; border: 1px solid #e1e4e8; border-radius: 10px;
            padding: 16px; margin-bottom: 12px; box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        }
        .status-safe { color: #28a745; font-weight: 800; }
        .status-cont { color: #dc3545; font-weight: 800; }
        .status-warn { color: #fd7e14; font-weight: 800; }
        .header-text { font-weight: 800; color: #1f2937; }
        code { white-space: pre-wrap !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 2) If OpenCV is broken, show fix guide and stop
# ============================================================
if CV2_IMPORT_ERROR is not None:
    st.error("❌ OpenCV (cv2) import failed — inference cannot run.")
    st.code(repr(CV2_IMPORT_ERROR))

    st.markdown("### ✅ Streamlit Cloud에서 가장 흔한 해결법")
    st.markdown(
        """
- `libGL.so.1` / `libgthread-2.0.so.0` 같은 에러는 보통 OpenCV가 시스템 라이브러리를 못 찾아서 생김.
- Streamlit Cloud에서는 **opencv-python 대신 opencv-python-headless** 를 쓰는 게 정석.
- `packages.txt`로 apt 설치를 건드리면 Debian 버전 충돌로 더 자주 터짐 → 가능하면 **packages.txt 제거** 권장.
"""
    )
    st.markdown("#### 추천 requirements.txt 예시 (그대로 복붙 가능)")
    st.code(
        "\n".join(
            [
                "streamlit==1.39.0",
                "numpy<2",
                "pillow",
                "pandas",
                "torch",
                "torchvision",
                "ultralytics==8.4.19",
                "opencv-python-headless==4.11.0.86",
            ]
        )
    )
    st.stop()


# ============================================================
# 3) Performance / threads
# ============================================================
try:
    CV2.setNumThreads(1)
except Exception:
    pass

if TORCH_OK:
    try:
        TORCH.set_num_threads(1)
        TORCH.set_num_interop_threads(1)
    except Exception:
        pass


# ============================================================
# 4) Storage / config
# ============================================================
SAVE_ROOT = "TFCP_Data"
IMG_DIR = os.path.join(SAVE_ROOT, "raw_images")
LOG_DIR = os.path.join(SAVE_ROOT, "analysis_logs")
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

APP_VERSION = "11.0.0-clean"


def _get_secret(key: str, default: str = "") -> str:
    try:
        return st.secrets.get(key, default)
    except Exception:
        return default


ADMIN_KEY = _get_secret("TFCP_ADMIN_KEY", "") or os.environ.get("TFCP_ADMIN_KEY", "tfcp2026")


# ============================================================
# 5) YOLO model loading (lazy + cached)
# ============================================================
@st.cache_resource(show_spinner=False)
def load_yolo_model(weights_path: str = "best.pt"):
    """
    Returns:
      (model, error_msg)
      model: YOLO object or None
      error_msg: str or None
    """
    if not os.path.exists(weights_path):
        return None, f"weights file not found: {weights_path}"

    try:
        from ultralytics import YOLO
    except Exception as e:
        return None, f"Failed to import ultralytics: {repr(e)}"

    try:
        m = YOLO(weights_path)
        return m, None
    except Exception as e:
        return None, f"Failed to load YOLO weights: {repr(e)}"


# ============================================================
# 6) Fonts (Pillow 내장 DejaVu → 폰트 크기 확실히 먹음)
# ============================================================
@lru_cache(maxsize=64)
def _load_font(size: int, bold: bool):
    pil_dir = os.path.dirname(ImageFont.__file__)
    pil_fonts_dir = os.path.join(pil_dir, "fonts")

    candidates = []
    if bold:
        candidates += [
            os.path.join(pil_fonts_dir, "DejaVuSans-Bold.ttf"),
            os.path.join(pil_fonts_dir, "DejaVuSans.ttf"),
            "fonts/DejaVuSans-Bold.ttf",
            "fonts/LiberationSans-Bold.ttf",
        ]
    else:
        candidates += [
            os.path.join(pil_fonts_dir, "DejaVuSans.ttf"),
            os.path.join(pil_fonts_dir, "DejaVuSans-Bold.ttf"),
            "fonts/DejaVuSans.ttf",
            "fonts/LiberationSans-Regular.ttf",
        ]

    for p in candidates:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass

    return ImageFont.load_default()


# ============================================================
# 7) Visualization: draw boxes & labels
# ============================================================
def draw_smart_annotations(
    img_bgr,
    reports,
    show_box_labels=True,
    show_global_label=False,
    label_scale=1.8,   # ✅ 여기 숫자 올리면 글자/라벨/선 다 커짐 (ex: 2.2)
):
    """
    OpenCV 기본 폰트로 박스/라벨을 그립니다.
    - PIL truetype 폰트가 없어도 글자 크기 확실히 커짐
    - 반환은 RGB (streamlit st.image에 바로 넣을 수 있게)
    """

    if img_bgr is None:
        return None

    img = img_bgr.copy()
    h, w = img.shape[:2]

    # 이미지 크기에 따른 자동 스케일
    base = max(w, h) / 1200.0

    # 박스 선 굵기
    line_w = max(2, int(4 * base * label_scale))

    # 폰트 설정 (OpenCV 내장)
    font = CV2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.7, 1.1 * base) * label_scale
    text_thick = max(2, int(2 * base * label_scale))
    pad = max(6, int(10 * base * label_scale))

    def status_color_bgr(status: str):
        # OpenCV는 BGR
        if status == "CONTAMINATED":
            return (69, 53, 220)    # 빨강(BGR)
        if status == "RECHECK REQUIRED":
            return (20, 126, 253)   # 주황(BGR)
        return (69, 167, 40)        # 초록(BGR)

    def status_text(status: str):
        if status == "CONTAMINATED":
            return "Contaminated"
        if status == "RECHECK REQUIRED":
            return "Recheck"
        return "Safe"

    def text_box_size(text: str):
        (tw, th), baseline = CV2.getTextSize(text, font, font_scale, text_thick)
        bw = tw + 2 * pad
        bh = th + 2 * pad + baseline
        return bw, bh, th

    def draw_label(x, y, text, col_bgr):
        bw, bh, th = text_box_size(text)

        # 화면 밖으로 나가지 않게 clamp
        x = int(max(0, min(x, w - bw - 1)))
        y = int(max(0, min(y, h - bh - 1)))

        # 라벨 배경(검정)
        CV2.rectangle(img, (x, y), (x + bw, y + bh), (0, 0, 0), thickness=-1)

        # 라벨 테두리(상태색) - 원하면 주석 해제
        CV2.rectangle(img, (x, y), (x + bw, y + bh), col_bgr, thickness=max(1, int(1 * base * label_scale)))

        # 텍스트
        CV2.putText(
            img,
            text,
            (x + pad, y + pad + th),
            font,
            font_scale,
            col_bgr,
            text_thick,
            CV2.LINE_AA,
        )

        return bw, bh

    # 전체 라벨(좌상단 1개) 옵션
    if show_global_label and reports:
        overall = "SAFE"
        for r in reports:
            s = r.get("status", "SAFE")
            if s == "CONTAMINATED":
                overall = "CONTAMINATED"
                break
            elif s == "RECHECK REQUIRED":
                overall = "RECHECK REQUIRED"
        draw_label(8, 8, status_text(overall), status_color_bgr(overall))

    # contaminated가 마지막에 오도록(겹칠 때 위에 보이게)
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
        col = status_color_bgr(status)

        # 박스
        cv2.rectangle(img, (x1, y1), (x2, y2), col, thickness=line_w)

        # 박스 라벨
        if show_box_labels:
            txt = status_text(status)
            bw, bh, _ = text_box_size(txt)

            ly = y1 - bh
            if ly < 0:
                ly = y1 + 2

            draw_label(x1, ly, txt, col)

    # ✅ streamlit에 바로 올리기 위해 RGB로 변환해서 반환
    return CV2.cvtColor(img, CV2.COLOR_BGR2RGB)

# ============================================================
# 8) Image utilities
# ============================================================
def apply_gamma_correction(img_bgr: np.ndarray, gamma: float) -> np.ndarray:
    g = float(gamma)
    if g <= 0:
        g = 0.8
    inv = 1.0 / g
    table = np.array([((i / 255.0) ** inv) * 255 for i in range(256)], dtype=np.uint8)
    return CV2.LUT(img_bgr, table)


def downscale_if_needed(img_bgr: np.ndarray, max_side: int = 1280):
    h, w = img_bgr.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return img_bgr, 1.0
    scale = max_side / float(m)
    nw = max(1, int(w * scale))
    nh = max(1, int(h * scale))
    small = CV2.resize(img_bgr, (nw, nh), interpolation=CV2.INTER_AREA)
    return small, scale


def rescale_reports(reports: list, inv_scale: float) -> list:
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


# ============================================================
# 9) Blue-kill metrics + mask
# ============================================================
def _resize_for_metrics(img_bgr: np.ndarray, max_side: int = 320) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return img_bgr
    scale = max_side / float(m)
    nw = max(1, int(w * scale))
    nh = max(1, int(h * scale))
    return CV2.resize(img_bgr, (nw, nh), interpolation=CV2.INTER_AREA)


def analyze_green_blue(
    img_bgr: np.ndarray,
    *,
    sample_max_side: int = 320,
    min_v: int = 40,
    green_min: int = 60,
    green_dom_diff: int = 30,
    b_floor_percentile: int = 15,
):
    small = _resize_for_metrics(img_bgr, sample_max_side)
    b, g, r = CV2.split(small.astype(np.int16))

    v = np.maximum(np.maximum(b, g), r)
    valid = v > int(min_v)
    valid_n = int(np.count_nonzero(valid))
    if valid_n <= 0:
        return {"valid_px": 0, "green_ratio": 0.0, "blue_dom90": 0.0, "b90": 0.0, "b_floor": 0.0}

    green_dom = (g > int(green_min)) & ((g - np.maximum(b, r)) > int(green_dom_diff)) & valid
    green_ratio = float(np.count_nonzero(green_dom) / valid_n)

    blue_over = b - np.maximum(g, r)
    blue_dom90 = float(np.percentile(blue_over[valid], 90))
    b90 = float(np.percentile(b[valid], 90))
    b_floor = float(np.percentile(b[valid].astype(np.float32), int(b_floor_percentile)))

    return {
        "valid_px": valid_n,
        "green_ratio": green_ratio,
        "blue_dom90": blue_dom90,
        "b90": b90,
        "b_floor": b_floor,
    }


def apply_blue_kill(
    img_bgr: np.ndarray,
    *,
    b_floor: float,
    gain: float = 0.01,
    strength: float = 2.0,
    cap_delta: int = 1,
) -> np.ndarray:
    b, g, r = CV2.split(img_bgr.astype(np.float32))

    b2 = np.maximum(0.0, b - float(b_floor) * float(strength)) * float(gain)
    if cap_delta is not None:
        b2 = np.minimum(b2, g + float(cap_delta))

    out = CV2.merge(
        [
            np.clip(b2, 0, 255).astype(np.uint8),
            np.clip(g, 0, 255).astype(np.uint8),
            np.clip(r, 0, 255).astype(np.uint8),
        ]
    )
    return out


def make_green_signal_mask(
    roi_bgr: np.ndarray,
    valid_u8: np.ndarray,
    containment_u8: np.ndarray,
    *,
    g_min: int = 120,
    diff_min: int = 40,
):
    b, g, r = CV2.split(roi_bgr.astype(np.int16))
    green_dom = (g >= int(g_min)) & ((g - np.maximum(b, r)) >= int(diff_min))

    mask = (green_dom.astype(np.uint8) * 255)
    mask = CV2.bitwise_and(mask, valid_u8)
    mask = CV2.bitwise_and(mask, containment_u8)

    mask = CV2.morphologyEx(mask, CV2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = CV2.morphologyEx(mask, CV2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    green_res = np.clip(
        g.astype(np.float32) - 0.5 * b.astype(np.float32) - 0.5 * r.astype(np.float32),
        0.0,
        255.0,
    )
    intensity_map = np.clip(green_res / 2.55, 0.0, 100.0)

    sat = (g >= 230) & ((g - np.maximum(b, r)) >= 60)
    sat_mask = (sat.astype(np.uint8) * 255)
    sat_mask = CV2.bitwise_and(sat_mask, mask)
    saturated_pixels = int(CV2.countNonZero(sat_mask))

    return mask, intensity_map, saturated_pixels


# ============================================================
# 10) Detection helpers
# ============================================================
def calculate_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    b1, b2 = box1.flatten(), box2.flatten()
    ix1, iy1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    ix2, iy2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area1 = max(0, (b1[2] - b1[0])) * max(0, (b1[3] - b1[1]))
    area2 = max(0, (b2[2] - b2[0])) * max(0, (b2[3] - b2[1]))
    union = area1 + area2 - inter
    return float(inter / union) if union > 0 else 0.0


def filter_nested_boxes(boxes) -> list:
    if boxes is None or len(boxes) == 0:
        return []

    confs = []
    for b in boxes:
        try:
            confs.append(float(b.conf.item()))
        except Exception:
            confs.append(0.0)

    sorted_indices = np.argsort(confs)[::-1]
    keep = []

    for i in sorted_indices:
        try:
            box_a = boxes[i].xyxy[0].cpu().numpy().flatten()
        except Exception:
            continue

        ok = True
        for j in keep:
            box_b = boxes[j].xyxy[0].cpu().numpy().flatten()
            if calculate_iou(box_a, box_b) > 0.3:
                ok = False
                break

            ix1, iy1 = max(box_a[0], box_b[0]), max(box_a[1], box_b[1])
            ix2, iy2 = min(box_a[2], box_b[2]), min(box_a[3], box_b[3])
            inter_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            area_a = max(0, (box_a[2] - box_a[0])) * max(0, (box_a[3] - box_a[1]))
            if area_a > 0 and (inter_area / area_a) > 0.7:
                ok = False
                break

        if ok:
            keep.append(i)

    return [boxes[idx] for idx in keep]


def _box_to_xyxy_int(box):
    try:
        coords = box.xyxy[0].cpu().numpy().flatten()
    except Exception:
        coords = np.array(box.xyxy[0]).flatten()
    x1, y1, x2, y2 = map(int, coords)
    return x1, y1, x2, y2


def detect_particles_heuristically(img_bgr: np.ndarray):
    hsv = CV2.cvtColor(img_bgr, CV2.COLOR_BGR2HSV)
    mask_o = CV2.inRange(hsv, np.array([0, 40, 40]), np.array([60, 255, 255]))
    mask_c = CV2.inRange(hsv, np.array([80, 30, 30]), np.array([165, 255, 255]))
    combined = CV2.bitwise_or(mask_o, mask_c)

    combined = CV2.morphologyEx(combined, CV2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    contours, _ = CV2.findContours(combined, CV2.RETR_EXTERNAL, CV2.CHAIN_APPROX_SIMPLE)

    class FakeBox:
        def __init__(self, coords):
            if TORCH_OK:
                self.xyxy = TORCH.tensor([coords], dtype=TORCH.float32)
                self.conf = TORCH.tensor([0.15])
            else:
                self.xyxy = np.array([coords], dtype=np.float32)
                self.conf = np.array([0.15], dtype=np.float32)

    out = []
    for cnt in contours:
        if CV2.contourArea(cnt) > 3000:
            x, y, w, h = CV2.boundingRect(cnt)
            ar = float(w) / float(h) if h > 0 else 0.0
            if 0.2 < ar < 5.0:
                out.append(FakeBox([x, y, x + w, y + h]))
    return out


# ============================================================
# 11) Core analysis
# ============================================================
@dataclass
class BlueKillThresholds:
    green_ratio_thr: float = 0.07
    blue_dom90_thr: float = 15.0
    b90_thr: float = 110.0
    b_floor_percentile: int = 15


@dataclass
class BlueKillParams:
    gain: float = 0.01
    strength: float = 2.0
    cap_delta: int = 1


def process_frame(
    img_bgr: np.ndarray,
    *,
    gamma: float,
    model=None,
    model_conf: float = 0.10,
    model_iou: float = 0.45,
    blue_kill_enabled: bool = True,
    bk_thr: BlueKillThresholds = BlueKillThresholds(),
    bk_params: BlueKillParams = BlueKillParams(),
    show_box_labels: bool = True,
    show_global_label: bool = False,
    label_scale: float = 1.0,
):
    """
    Returns:
      result_rgb (processed + annotated)  [this image coords]
      reports (boxes in this image coords)
      pre (debug meta)
    """
    img_corr = apply_gamma_correction(img_bgr, gamma=gamma)

    m = analyze_green_blue(img_corr, b_floor_percentile=bk_thr.b_floor_percentile)

    green_present = (m["green_ratio"] > float(bk_thr.green_ratio_thr))
    blue_cast = (m["blue_dom90"] > float(bk_thr.blue_dom90_thr)) and (m["b90"] > float(bk_thr.b90_thr))

    pre = {
        "applied": False,
        "mode": "HSV_CYAN",
        "green_present": bool(green_present),
        "blue_cast": bool(blue_cast),
        "metrics": m,
        "params": {},
    }

    img_anl = img_corr

    if blue_kill_enabled and (not green_present) and blue_cast:
        img_anl = apply_blue_kill(
            img_corr,
            b_floor=m["b_floor"],
            gain=bk_params.gain,
            strength=bk_params.strength,
            cap_delta=bk_params.cap_delta,
        )
        pre["applied"] = True
        pre["mode"] = "GREEN_SIGNAL"
        pre["params"] = {
            "gain": bk_params.gain,
            "strength": bk_params.strength,
            "cap_delta": bk_params.cap_delta,
            "b_floor": m["b_floor"],
        }

    hsv = CV2.cvtColor(img_anl, CV2.COLOR_BGR2HSV)
    img_h, img_w = img_anl.shape[:2]

    ai_raw_boxes = []
    if model is not None:
        try:
            results = model.predict(
                source=img_corr,
                conf=float(model_conf),
                iou=float(model_iou),
                imgsz=640,
                verbose=False,
            )
            ai_raw_boxes = filter_nested_boxes(results[0].boxes)
        except Exception:
            ai_raw_boxes = []

    ORANGE_LO = np.array([0, 35, 35])
    ORANGE_HI = np.array([60, 255, 255])
    CYAN_LO = np.array([80, 30, 30])
    CYAN_HI = np.array([165, 255, 255])

    combined_boxes = []

    for box in ai_raw_boxes:
        x1, y1, x2, y2 = _box_to_xyxy_int(box)
        if (x2 - x1) < 50 or (y2 - y1) < 50:
            continue

        if pre["mode"] == "GREEN_SIGNAL":
            combined_boxes.append((box, "AI"))
            continue

        roi_hsv = hsv[max(0, y1):min(img_h, y2), max(0, x1):min(img_w, x2)]
        if roi_hsv.size == 0:
            continue

        mask_o = CV2.inRange(roi_hsv, ORANGE_LO, ORANGE_HI)
        mask_c = CV2.inRange(roi_hsv, CYAN_LO, CYAN_HI)
        color_px = int(CV2.countNonZero(mask_o)) + int(CV2.countNonZero(mask_c))
        if color_px > 200:
            combined_boxes.append((box, "AI"))

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

        mask_orange = CV2.inRange(roi_hsv, np.array([0, 30, 30]), np.array([60, 255, 255]))
        mask_orange_valid = CV2.bitwise_and(mask_orange, valid_u8)
        mask_orange_closed = CV2.morphologyEx(mask_orange_valid, CV2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

        contours, _ = CV2.findContours(mask_orange_closed, CV2.RETR_EXTERNAL, CV2.CHAIN_APPROX_SIMPLE)

        mask_body = np.zeros_like(mask_orange_closed)
        for cnt in contours:
            if CV2.contourArea(cnt) > MIN_COMP_AREA:
                CV2.drawContours(mask_body, [cnt], -1, 255, -1)

        body_area_px = int(CV2.countNonZero(mask_body))
        box_area = int((nx2 - nx1) * (ny2 - ny1))
        orange_area_pct = (body_area_px / box_area) * 100.0 if box_area > 0 else 0.0

        if body_area_px < MIN_BODY_PX or orange_area_pct < MIN_ORANGE_PCT:
            status = "RECHECK REQUIRED"
            phi = 0.0
            signal_area = 0.0
            avg_int = 0.0
        else:
            mask_zone = CV2.dilate(mask_body, np.ones((3, 3), np.uint8), iterations=1)

            if pre["mode"] == "GREEN_SIGNAL":
                mask_sig, intensity_map, saturated_pixels = make_green_signal_mask(
                    roi_img,
                    valid_u8,
                    mask_zone,
                    g_min=120,
                    diff_min=40,
                )
                sig_px = int(CV2.countNonZero(mask_sig))
                signal_area = (sig_px / body_area_px) * 100.0 if body_area_px > 0 else 0.0
                avg_int = float(np.mean(intensity_map[mask_sig > 0])) if sig_px > 0 else 0.0
                phi = signal_area * (avg_int / 10.0)

                status = "CONTAMINATED" if (phi > 5.0 or saturated_pixels > 20) else "SAFE"
                if status == "CONTAMINATED" and phi < 5.0:
                    phi = 99.9
            else:
                mask_cyan_candidate = CV2.inRange(roi_hsv, CYAN_LO, CYAN_HI)
                mask_cyan_candidate = CV2.bitwise_and(mask_cyan_candidate, valid_u8)
                mask_cyan = CV2.bitwise_and(mask_cyan_candidate, mask_zone)

                b_ch, g_ch, r_ch = CV2.split(roi_img.astype(np.float32))
                is_glare = (g_ch > 200) & (b_ch > 200) & (r_ch > 200)
                is_saturated_cyan = (g_ch > 200) & (b_ch > 200) & (r_ch < 200)

                sat_mask = (is_saturated_cyan.astype(np.uint8) * 255)
                sat_mask = CV2.bitwise_and(sat_mask, mask_zone)
                saturated_pixels = int(CV2.countNonZero(sat_mask))

                intensity_raw = np.where(
                    is_glare,
                    0.0,
                    np.where(
                        is_saturated_cyan,
                        ((g_ch + b_ch) / 2.0 - r_ch * 0.8),
                        ((g_ch + b_ch) / 2.0 - r_ch * 1.7),
                    ),
                )
                intensity_map = np.clip(intensity_raw, 0.0, 100.0)

                cyan_px = int(CV2.countNonZero(mask_cyan))
                signal_area = (cyan_px / body_area_px) * 100.0 if body_area_px > 0 else 0.0
                avg_int = float(np.mean(intensity_map[mask_cyan > 0])) if cyan_px > 0 else 0.0

                phi = signal_area * (avg_int / 10.0)

                status = "CONTAMINATED" if (phi > 5.0 or saturated_pixels > 20) else "SAFE"
                if status == "CONTAMINATED" and phi < 5.0:
                    phi = 99.9

        reports.append(
            {
                "id": int(i),
                "status": str(status),
                "phi": float(round(phi, 2)),
                "cyan": float(round(signal_area, 2)),
                "orange": float(round(orange_area_pct, 2)),
                "box": [int(nx1), int(ny1), int(nx2), int(ny2)],
                "method": str(method),
                "signal_mode": str(pre["mode"]),
            }
        )

    result_rgb = draw_smart_annotations(
        img_anl.copy(),
        reports,
        show_box_labels=show_box_labels,
        show_global_label=show_global_label,
        label_scale=label_scale,
    )
    return result_rgb, reports, pre


# ============================================================
# 12) Admin console
# ============================================================
def render_admin_page():
    st.markdown("<h2 class='header-text'>Research Data Management Center</h2>", unsafe_allow_html=True)

    log_files = sorted([f for f in os.listdir(LOG_DIR) if f.endswith(".json")], reverse=True)
    if not log_files:
        st.info("No logs yet.")
        return

    if "current_log_file" not in st.session_state:
        st.session_state.current_log_file = log_files[0]
    if st.session_state.current_log_file not in log_files:
        st.session_state.current_log_file = log_files[0]

    cur = st.session_state.current_log_file
    cur_idx = log_files.index(cur)

    top_l, top_m, top_r = st.columns([1, 4, 1])

    with top_l:
        if st.button("◀ PREV"):
            st.session_state.current_log_file = log_files[max(0, cur_idx - 1)]
            st.rerun()

    with top_r:
        if st.button("NEXT ▶"):
            st.session_state.current_log_file = log_files[min(len(log_files) - 1, cur_idx + 1)]
            st.rerun()

    with top_m:
        def _sel_change():
            st.session_state.current_log_file = st.session_state._log_sel

        st.selectbox(
            "Select Log",
            log_files,
            index=cur_idx,
            key="_log_sel",
            on_change=_sel_change,
            label_visibility="collapsed",
        )

        b1, b2 = st.columns(2)
        with b1:
            if st.button("📦 Archive (ZIP)", use_container_width=True):
                shutil.make_archive("TFCP_Dataset", "zip", SAVE_ROOT)
                with open("TFCP_Dataset.zip", "rb") as fp:
                    st.download_button(
                        "Download ZIP",
                        fp,
                        "TFCP_Dataset.zip",
                        mime="application/zip",
                        use_container_width=True,
                    )

        with b2:
            if st.button("🗑️ Delete", type="primary", use_container_width=True):
                try:
                    log_path = os.path.join(LOG_DIR, st.session_state.current_log_file)
                    with open(log_path, "r") as f:
                        data = json.load(f)

                    img_path = os.path.join(IMG_DIR, data.get("filename", ""))
                    if os.path.exists(log_path):
                        os.remove(log_path)
                    if img_path and os.path.exists(img_path):
                        os.remove(img_path)

                    st.success("Deleted.")
                    st.session_state.current_log_file = log_files[0]
                    st.rerun()
                except Exception as e:
                    st.error(f"Delete failed: {e}")

    log_path = os.path.join(LOG_DIR, st.session_state.current_log_file)
    try:
        with open(log_path, "r") as f:
            data = json.load(f)

        img_path = os.path.join(IMG_DIR, data.get("filename", ""))
        reports = data.get("reports", data.get("particles", []))

        if not os.path.exists(img_path):
            st.error("Image missing for this log.")
            return

        img_bgr = CV2.imread(img_path)
        if img_bgr is None:
            st.error("cv2.imread returned None.")
            return

        annotated_rgb = draw_smart_annotations(img_bgr.copy(), reports, show_box_labels=True, show_global_label=False, label_scale=1.4)
        st.image(annotated_rgb, caption=f"Log: {st.session_state.current_log_file}", use_column_width=True)

        with st.expander("➕ Manual Region Injection (Slider)", expanded=False):
            h, w = img_bgr.shape[:2]
            c1, c2 = st.columns(2)
            with c1:
                mx1 = st.slider("X Start", 0, w, int(w * 0.30))
                mx2 = st.slider("X End", 0, w, int(w * 0.70))
            with c2:
                my1 = st.slider("Y Start", 0, h, int(h * 0.30))
                my2 = st.slider("Y End", 0, h, int(h * 0.70))

            x1, x2 = sorted([mx1, mx2])
            y1, y2 = sorted([my1, my2])

            prev = img_bgr.copy()
            CV2.rectangle(prev, (x1, y1), (x2, y2), (255, 0, 255), 4)
            st.image(CV2.cvtColor(prev, CV2.COLOR_BGR2RGB), caption="Preview", use_column_width=True)

            if st.button("✅ Inject ROI as CONTAMINATED"):
                if (x2 - x1) < 5 or (y2 - y1) < 5:
                    st.error("ROI too small.")
                else:
                    new_p = {
                        "id": len(reports),
                        "box": [x1, y1, x2, y2],
                        "status": "CONTAMINATED",
                        "phi": 0,
                        "cyan": 0,
                        "orange": 0,
                        "manual": True,
                    }
                    reports.append(new_p)
                    data["reports"] = reports
                    data["reviewed"] = True
                    with open(log_path, "w") as f:
                        json.dump(data, f, indent=4)
                    st.success("Injected!")
                    st.rerun()

        with st.form("admin_update_form"):
            st.markdown("### Annotation Correction")
            updated = []
            cols = st.columns(2)
            for i, r in enumerate(reports):
                with cols[i % 2]:
                    stat = r.get("status", "SAFE")
                    cls = "status-cont" if stat == "CONTAMINATED" else "status-safe" if stat == "SAFE" else "status-warn"
                    st.markdown(f"**Area {i+1}**: <span class='{cls}'>{stat}</span>", unsafe_allow_html=True)

                    options = ["SAFE", "CONTAMINATED", "RECHECK REQUIRED"]
                    idx = options.index(stat) if stat in options else 0
                    new_stat = st.radio(f"Status {i+1}", options, index=idx, horizontal=True, key=f"_admin_stat_{i}")
                    rr = dict(r)
                    rr["id"] = i
                    rr["status"] = new_stat
                    updated.append(rr)

            if st.form_submit_button("Save Annotations"):
                data["reports"] = updated
                data["reviewed"] = True
                with open(log_path, "w") as f:
                    json.dump(data, f, indent=4)
                st.success("Saved.")
                st.rerun()

    except Exception as e:
        st.error(f"Admin error: {repr(e)}")


# ============================================================
# 13) Main UI
# ============================================================
st.sidebar.title("Navigation")
mode = st.sidebar.radio("Go to", ["Real-time Inference", "Admin Console"])

if mode == "Admin Console":
    if "admin_mode" not in st.session_state:
        st.session_state.admin_mode = False

    if not st.session_state.admin_mode:
        pwd = st.sidebar.text_input("Access Key", type="password")
        if pwd:
            if pwd == ADMIN_KEY:
                st.session_state.admin_mode = True
                st.rerun()
            else:
                st.sidebar.error("Invalid key")

    if st.session_state.admin_mode:
        render_admin_page()

else:
    st.markdown("<h1 class='header-text'>TFCP Inference Engine</h1>", unsafe_allow_html=True)

    # ---------------- Sidebar controls ----------------
    st.sidebar.markdown("### Inference Settings")
    gamma = st.sidebar.slider("Gamma", 0.30, 2.00, 0.80, 0.05)
    model_conf = st.sidebar.slider("YOLO conf", 0.05, 0.60, 0.10, 0.01)
    model_iou = st.sidebar.slider("YOLO IoU", 0.10, 0.80, 0.45, 0.05)
    max_side = st.sidebar.selectbox("Downscale max side", [800, 1024, 1280, 1600], index=2)

    st.sidebar.markdown("### Display")
    show_box_labels = st.sidebar.checkbox("Show box labels", value=True)
    show_global_label = st.sidebar.checkbox("Show global label", value=False)
    label_scale = st.sidebar.slider("Label size scale", 0.5, 3.0, 1.4, 0.1)

    st.sidebar.markdown("### Blue-kill mode")
    blue_kill_enabled = st.sidebar.checkbox("Enable Blue-kill when no green", value=True)

    with st.sidebar.expander("Blue-kill advanced"):
        bk_green_ratio_thr = st.slider("GREEN_RATIO_THR", 0.0, 0.30, 0.07, 0.01)
        bk_blue_dom90_thr = st.slider("BLUE_DOM90_THR", 0.0, 60.0, 15.0, 1.0)
        bk_b90_thr = st.slider("B90_THR", 0.0, 255.0, 110.0, 5.0)
        bk_b_floor_pct = st.slider("b_floor_percentile", 1, 50, 15, 1)

        bk_gain = st.slider("BLUEKILL_GAIN", 0.001, 0.20, 0.01, 0.001)
        bk_strength = st.slider("BLUEKILL_STRENGTH", 0.1, 5.0, 2.0, 0.1)
        bk_cap = st.slider("BLUEKILL_CAP_DELTA", 0, 50, 1, 1)

    auto_run_on_new = st.sidebar.checkbox("Auto-run when new image arrives", value=True)
    auto_save = st.sidebar.checkbox("Auto-save run to logs", value=True)
    jpg_quality = st.sidebar.slider("Save JPG quality", 70, 100, 98, 1)

    # ---------------- Layout columns ----------------
    c1, c2 = st.columns([2, 1])

    # ---------------- Session init ----------------
    if "last_proc_hash" not in st.session_state:
        st.session_state.last_proc_hash = None
        st.session_state.last_img_hash = None
        st.session_state.last_saved_hash = None

        st.session_state.orig_bgr = None
        st.session_state.proc_bgr = None
        st.session_state.scale = 1.0

        st.session_state.reports_up = []
        st.session_state.pre = {}
        st.session_state.result_rgb = None

    # ---------------- Input UI ----------------
    with c1:
        img_file = st.camera_input("Acquire")
        if not img_file:
            img_file = st.file_uploader("Upload", type=["jpg", "jpeg", "png"])

    if not img_file:
        st.info("새 이미지를 업로드/촬영하면 분석할 수 있습니다.")
        st.stop()

    raw = img_file.getvalue()
    img_hash = hashlib.sha1(raw).hexdigest()

    # decode only when input changes
    if img_hash != st.session_state.last_proc_hash:
        buf = np.frombuffer(raw, dtype=np.uint8)
        img_bgr = CV2.imdecode(buf, 1)
        if img_bgr is None:
            st.error("cv2.imdecode failed.")
            st.stop()

        img_proc, scale = downscale_if_needed(img_bgr, max_side=max_side)

        st.session_state.last_proc_hash = img_hash
        st.session_state.orig_bgr = img_bgr
        st.session_state.proc_bgr = img_proc
        st.session_state.scale = scale

    # ---------------- Run buttons ----------------
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
        model, model_err = load_yolo_model("best.pt")
        if model_err:
            st.warning(f"YOLO disabled: {model_err}")
            model = None

        bk_thr = BlueKillThresholds(
            green_ratio_thr=float(bk_green_ratio_thr),
            blue_dom90_thr=float(bk_blue_dom90_thr),
            b90_thr=float(bk_b90_thr),
            b_floor_percentile=int(bk_b_floor_pct),
        )
        bk_params = BlueKillParams(
            gain=float(bk_gain),
            strength=float(bk_strength),
            cap_delta=int(bk_cap),
        )

        with st.spinner("Analyzing..."):
            _, reports_small, pre = process_frame(
                st.session_state.proc_bgr,
                gamma=gamma,
                model=model,
                model_conf=model_conf,
                model_iou=model_iou,
                blue_kill_enabled=blue_kill_enabled,
                bk_thr=bk_thr,
                bk_params=bk_params,
                show_box_labels=show_box_labels,
                show_global_label=show_global_label,
                label_scale=label_scale,
            )

        # rescale reports to original coords
        inv_scale = 1.0 / float(st.session_state.scale if st.session_state.scale > 0 else 1.0)
        reports_up = rescale_reports(reports_small, inv_scale)

        # build processed-original image (원본 기반으로 blue-kill을 다시 적용)
        orig = st.session_state.orig_bgr.copy()
        orig_gamma = apply_gamma_correction(orig, gamma=gamma)

        processed_orig = orig_gamma
        if pre.get("applied"):
            mm = analyze_green_blue(orig_gamma, b_floor_percentile=bk_thr.b_floor_percentile)
            processed_orig = apply_blue_kill(
                orig_gamma,
                b_floor=mm["b_floor"],
                gain=bk_params.gain,
                strength=bk_params.strength,
                cap_delta=bk_params.cap_delta,
            )

        processed_orig_rgb = draw_smart_annotations(
            processed_orig,
            reports_up,
            show_box_labels=show_box_labels,
            show_global_label=show_global_label,
            label_scale=label_scale,
        )

        st.session_state.last_img_hash = img_hash
        st.session_state.reports_up = reports_up
        st.session_state.pre = pre
        st.session_state.result_rgb = processed_orig_rgb

        # save once per new input
        if auto_save and (img_hash != st.session_state.last_saved_hash):
            now = datetime.now()
            ts_id = now.strftime("%Y%m%d_%H%M%S_%f")
            ts_display = now.strftime("%Y-%m-%d %H:%M:%S")
            fn = f"TFCP_{ts_id}"

            CV2.imwrite(
                os.path.join(IMG_DIR, f"{fn}.jpg"),
                st.session_state.orig_bgr,
                [int(CV2.IMWRITE_JPEG_QUALITY), int(jpg_quality)],
            )

            with open(os.path.join(LOG_DIR, f"{fn}.json"), "w") as f:
                json.dump(
                    {
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
                            "input_scale": float(st.session_state.scale),
                            "blue_kill_enabled": bool(blue_kill_enabled),
                            "bk_thr": bk_thr.__dict__,
                            "bk_params": bk_params.__dict__,
                        },
                    },
                    f,
                    indent=4,
                )

            st.session_state.last_saved_hash = img_hash

    # ---------------- Display ----------------
    if st.session_state.orig_bgr is not None:
        with c1:
            view_mode = st.radio(
                "View mode",
                ["Compare (Original vs Result)", "Result only", "Original only"],
                horizontal=True,
                label_visibility="collapsed",
            )

            orig_rgb = CV2.cvtColor(st.session_state.orig_bgr, CV2.COLOR_BGR2RGB)

            if view_mode == "Original only":
                st.image(orig_rgb, caption="Original", use_column_width=True)

            elif view_mode == "Result only":
                if st.session_state.result_rgb is not None:
                    st.image(st.session_state.result_rgb, caption="Result (processed/annotated)", use_column_width=True)
                else:
                    st.info("Run analysis to see results.")

            else:
                lc, rc = st.columns(2)
                with lc:
                    st.image(orig_rgb, caption="Original", use_column_width=True)

                    if st.session_state.reports_up:
                        orig_annot = draw_smart_annotations(
                            st.session_state.orig_bgr.copy(),
                            st.session_state.reports_up,
                            show_box_labels=show_box_labels,
                            show_global_label=show_global_label,
                            label_scale=label_scale,
                        )
                        st.image(orig_annot, caption="Original (annotated)", use_column_width=True)

                with rc:
                    if st.session_state.result_rgb is not None:
                        st.image(st.session_state.result_rgb, caption="Result (processed/annotated)", use_column_width=True)
                    else:
                        st.info("Run analysis to see results.")

        with c2:
            st.markdown("### Metrics")

            pre = st.session_state.pre or {}
            st.caption(
                f"blue_kill applied={pre.get('applied')} | "
                f"mode={pre.get('mode')} | "
                f"green_present={pre.get('green_present')} | "
                f"blue_cast={pre.get('blue_cast')}"
            )

            reports = st.session_state.reports_up or []
            if reports:
                n_cont = sum(1 for r in reports if r.get("status") == "CONTAMINATED")
                n_rechk = sum(1 for r in reports if r.get("status") == "RECHECK REQUIRED")
                n_safe = sum(1 for r in reports if r.get("status") == "SAFE")

                m1, m2, m3 = st.columns(3)
                m1.metric("SAFE", n_safe)
                m2.metric("CONT", n_cont)
                m3.metric("RECHECK", n_rechk)

                df = pd.DataFrame(reports).copy()
                df["area"] = df["id"].astype(int) + 1
                cols = [c for c in ["area", "status", "phi", "cyan", "orange", "method", "signal_mode"] if c in df.columns]
                st.dataframe(df[cols], use_container_width=True, height=260)

                csv_bytes = df[cols].to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download CSV",
                    csv_bytes,
                    file_name="tfcp_reports.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            else:
                st.info("No report yet. Click Run analysis.")


