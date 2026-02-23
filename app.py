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
def get_custom_font(size=20):
    """
    Loads DejaVuSans (Standard Linux Font) for reliable rendering on Streamlit Cloud.
    """
    font_candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "arial.ttf"
    ]

    for font_path in font_candidates:
        if os.path.exists(font_path):
            try:
                return ImageFont.truetype(font_path, size)
            except Exception:
                continue

    return ImageFont.load_default()

def draw_smart_annotations(img_bgr, reports):
    """
    Draws professional annotations using PIL (RGB).
    - [FIX] box/status 키가 누락된 로그에도 최대한 안전하게 동작하도록 방어 로직 추가
    - [FIX] 좌표를 이미지 경계로 clamp
    """
    if img_bgr is None:
        return None

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb).convert("RGBA")
    overlay = Image.new("RGBA", pil_img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    h, w = img_bgr.shape[:2]

    scale = max(w, h) / 1200.0
    line_width = max(2, int(4 * scale))
    font_size = max(16, int(26 * scale))
    font = get_custom_font(font_size)

    def get_priority(r):
        s = (r.get('status') or "SAFE")
        if s == "CONTAMINATED": return 2
        if s == "RECHECK REQUIRED": return 1
        return 0

    reports_sorted = sorted(reports or [], key=get_priority)

    for r in reports_sorted:
        box = r.get('box', None)
        if not box or len(box) != 4:
            continue

        x1, y1, x2, y2 = [int(v) for v in box]
        # clamp
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w - 1))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h - 1))
        if x2 <= x1 or y2 <= y1:
            continue

        status = (r.get('status') or "SAFE")
        p_id = int(r.get('id', 0))

        if status == "CONTAMINATED":
            color_rgb = (220, 53, 69)  # Red
        elif status == "RECHECK REQUIRED":
            color_rgb = (253, 126, 20)  # Orange
        else:
            color_rgb = (40, 167, 69)  # Green

        draw.rectangle([x1, y1, x2, y2], outline=color_rgb + (255,), width=line_width)

        label_txt = f"Area {p_id + 1}"
        if status == "RECHECK REQUIRED":
            label_txt = "RECHECK"
        elif status == "CONTAMINATED":
            label_txt += ": CONT"
        else:
            label_txt += ": SAFE"

        try:
            bbox = font.getbbox(label_txt)
            text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except Exception:
            text_w, text_h = draw.textsize(label_txt, font=font)

        pad = int(8 * scale)
        lbl_y1 = y1 - text_h - 2 * pad
        if lbl_y1 < 0:
            lbl_y1 = y1

        draw.rectangle(
            [x1, lbl_y1, x1 + text_w + 2 * pad, lbl_y1 + text_h + 2 * pad],
            fill=color_rgb + (210,)
        )
        draw.text((x1 + pad, lbl_y1 + pad), label_txt, font=font, fill=(255, 255, 255, 255))

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

def process_frame(img, gamma=0.8, model_conf=0.10, model_iou=0.45):
    """
    [FIX 1] AI box ROI의 색 픽셀 체크에서 np.sum(inRange)는 255*픽셀수 → 임계값이 의미없어짐.
            cv2.countNonZero로 픽셀수를 직접 비교하도록 수정.
    [FIX 2] p_count(컨투어 면적 합)과 mask 픽셀 수를 섞어 쓰던 부분을 '픽셀 수' 기준으로 통일.
            (cyan_area, orange_area_pct 계산이 더 일관적)
    [FIX 3] 실시간(inference)과 admin의 표시 이미지가 서로 달랐음(원본 vs gamma) → gamma 보정 이미지를 기준으로 통일.
    """
    img_corr = apply_gamma_correction(img, gamma=gamma)
    hsv = cv2.cvtColor(img_corr, cv2.COLOR_BGR2HSV)
    img_h, img_w = img_corr.shape[:2]

    ai_raw_boxes = []
    if model is not None:
        try:
            results = model.predict(source=img_corr, conf=model_conf, iou=model_iou, verbose=False)
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
        roi_hsv = hsv[max(0, y1):min(img_h, y2), max(0, x1):min(img_w, x2)]
        if roi_hsv.size == 0:
            continue

        # [FIX] 픽셀 수 기반으로 체크
        mask_o = cv2.inRange(roi_hsv, ORANGE_LO, ORANGE_HI)
        mask_c = cv2.inRange(roi_hsv, CYAN_LO, CYAN_HI)
        color_px = cv2.countNonZero(mask_o) + cv2.countNonZero(mask_c)

        if color_px > 200:  # 기존 200은 사실상 1픽셀만 있어도 통과였음
            combined_boxes.append((box, "AI"))

    if not combined_boxes:
        for cv_box in detect_particles_heuristically(img_corr):
            combined_boxes.append((cv_box, "CV_BACKUP"))

    reports = []

    MIN_BODY_PX = 120         # particle body(orange) 최소 픽셀수
    MIN_ORANGE_PCT = 3.0      # box 대비 orange 최소 점유율
    MIN_COMP_AREA = 20        # 작은 노이즈 contour 제거

    for i, (box, method) in enumerate(combined_boxes):
        x1, y1, x2, y2 = _box_to_xyxy_int(box)

        # pad 확장
        pad = int((x2 - x1) * 0.15)
        nx1 = max(0, x1 - pad)
        ny1 = max(0, y1 - pad)
        nx2 = min(img_w, x2 + pad)
        ny2 = min(img_h, y2 + pad)

        roi_hsv = hsv[ny1:ny2, nx1:nx2]
        roi_img = img_corr[ny1:ny2, nx1:nx2]
        if roi_hsv.size == 0:
            continue

        valid_mask = (roi_hsv[:, :, 1] > 25) & (roi_hsv[:, :, 2] > 25)
        valid_u8 = (valid_mask.astype(np.uint8) * 255)

        # body(orange) mask
        mask_orange = cv2.inRange(roi_hsv, np.array([0, 30, 30]), np.array([60, 255, 255]))
        mask_orange_valid = cv2.bitwise_and(mask_orange, valid_u8)
        mask_orange_closed = cv2.morphologyEx(mask_orange_valid, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

        contours, _ = cv2.findContours(mask_orange_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        mask_particle_body = np.zeros_like(mask_orange_closed)
        for cnt in contours:
            if cv2.contourArea(cnt) > MIN_COMP_AREA:
                cv2.drawContours(mask_particle_body, [cnt], -1, 255, -1)

        # [FIX] 픽셀 수 기준으로 통일
        body_area_px = int(cv2.countNonZero(mask_particle_body))

        box_area = int((nx2 - nx1) * (ny2 - ny1))
        orange_area_pct = (body_area_px / box_area) * 100.0 if box_area > 0 else 0.0

        if body_area_px < MIN_BODY_PX or orange_area_pct < MIN_ORANGE_PCT:
            status = "RECHECK REQUIRED"
            phi = 0.0
            cyan_area = 0.0
            avg_int = 0.0
        else:
            # containment zone
            mask_containment_zone = cv2.dilate(mask_particle_body, np.ones((3, 3), np.uint8), iterations=1)

            # cyan mask
            mask_cyan_candidate = cv2.inRange(roi_hsv, CYAN_LO, CYAN_HI)
            mask_cyan_candidate = cv2.bitwise_and(mask_cyan_candidate, valid_u8)
            mask_cyan = cv2.bitwise_and(mask_cyan_candidate, mask_containment_zone)

            # intensity map
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
            cyan_area = (cyan_px / body_area_px) * 100.0 if body_area_px > 0 else 0.0
            avg_int = float(np.mean(intensity_map[mask_cyan > 0])) if cyan_px > 0 else 0.0

            phi = cyan_area * (avg_int / 10.0)

            status = "CONTAMINATED" if (phi > 5.0 or saturated_pixels > 20) else "SAFE"
            # saturated_pixels로 CONT 판단된 경우, phi가 낮을 수 있어 sentinel 값 유지(기존 의도 존중)
            if status == "CONTAMINATED" and phi < 5.0:
                phi = 99.9

        reports.append({
            "id": i,
            "status": status,
            "phi": float(round(phi, 2)),
            "cyan": float(round(cyan_area, 2)),
            "orange": float(round(orange_area_pct, 2)),
            "box": [int(nx1), int(ny1), int(nx2), int(ny2)],
            "method": method
        })

    # [FIX] 표시용 이미지를 gamma 보정본으로 통일
    final_img = draw_smart_annotations(img_corr.copy(), reports)
    return final_img, reports

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

            st.image(display_img, caption=f"Analyzed: {data.get('timestamp', 'Unknown')}", width=800)

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
    c1, c2 = st.columns([2, 1])
    with c1:
        img_file = st.camera_input("Acquire")
        if not img_file:
            img_file = st.file_uploader("Upload", type=['jpg', 'png', 'jpeg'])

    if img_file:
        file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        if image is None:
            st.error("Load Failed")
        else:
            try:
                res_img_rgb, reports = process_frame(image)

                now = datetime.now()
                ts_id = now.strftime("%Y%m%d_%H%M%S_%f")   # [FIX] 동시 요청 충돌 방지
                ts_display = now.strftime("%Y-%m-%d %H:%M:%S")

                fn = f"TFCP_{ts_id}"
                cv2.imwrite(os.path.join(IMG_DIR, f"{fn}.jpg"), image)

                with open(os.path.join(LOG_DIR, f"{fn}.json"), "w") as f:
                    json.dump({
                        "filename": f"{fn}.jpg",
                        "timestamp": ts_display,
                        "timestamp_id": ts_id,
                        "reports": reports,
                        "reviewed": False,
                        "app_version": APP_VERSION,
                        "gamma": 0.8
                    }, f, indent=4)

                display_img = standardize_image_size(res_img_rgb, 1280, 960)
                with c1:
                    st.image(display_img, caption="Analysis Result", width=800)

                with c2:
                    st.markdown("### Metrics")
                    if reports:
                        # 요약 카운트
                        n_cont = sum(1 for r in reports if r.get('status') == "CONTAMINATED")
                        n_rechk = sum(1 for r in reports if r.get('status') == "RECHECK REQUIRED")
                        n_safe = sum(1 for r in reports if r.get('status') == "SAFE")

                        st.markdown(f"- SAFE: **{n_safe}**  \n- CONT: **{n_cont}**  \n- RECHECK: **{n_rechk}**")

                        for r in reports:
                            cls = "status-cont" if r['status'] == "CONTAMINATED" else "status-safe" if r[
                                                                                                          'status'] == "SAFE" else "status-warn"
                            st.markdown(f"""
                            <div class="metric-card">
                                <div><strong>Area {r['id'] + 1}</strong></div>
                                <div class="{cls}">{r['status']}</div>
                                <div style="font-size:0.85em; color:#666; margin-top:5px;">
                                    Φ: {r['phi']}<br>
                                    Cyan: {r['cyan']}%<br>
                                    Orange: {r['orange']}%
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.warning("No particles")
            except Exception as e:
                st.error(f"Error: {e}")
