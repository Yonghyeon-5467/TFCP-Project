import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import os
import torch
import json
from datetime import datetime

# --- [1] 페이지 및 기본 설정 ---
st.set_page_config(page_title="TFCP AI BigData Platform", page_icon="📡", layout="wide")

# 로컬 저장소 경로 (Streamlit Cloud 임시 저장소)
SAVE_ROOT = "TFCP_Data"
IMG_DIR = os.path.join(SAVE_ROOT, "raw_images")
LOG_DIR = os.path.join(SAVE_ROOT, "analysis_logs")

if not os.path.exists(IMG_DIR): os.makedirs(IMG_DIR)
if not os.path.exists(LOG_DIR): os.makedirs(LOG_DIR)

# --- [2] 모델 로드 (캐싱 적용) ---
@st.cache_resource
def load_model():
    # GitHub 저장소에 best.pt가 업로드되어 있어야 합니다.
    if os.path.exists('best.pt'):
        return YOLO('best.pt')
    return None

model = load_model()

# --- [3] 핵심 분석 엔진 (v10.2.1 로직 탑재) ---

def apply_gamma_correction(image, gamma=0.8):
    """과노출(Saturation) 보정"""
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def calculate_iou(box1, box2):
    """박스 중첩 비율 계산"""
    b1, b2 = box1.flatten(), box2.flatten()
    ix1, iy1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    ix2, iy2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area1 = (b1[2]-b1[0])*(b1[3]-b1[1])
    area2 = (b2[2]-b2[0])*(b2[3]-b2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0

def filter_nested_boxes(boxes):
    """중복 및 포함 관계 박스 제거"""
    if len(boxes) == 0: return []
    sorted_indices = np.argsort([b.conf.item() for b in boxes])[::-1]
    keep_indices = []
    for i in sorted_indices:
        box_a = boxes[i].xyxy[0].cpu().numpy().flatten()
        keep = True
        for j in keep_indices:
            box_b = boxes[j].xyxy[0].cpu().numpy().flatten()
            if calculate_iou(box_a, box_b) > 0.3:
                keep = False; break
            ix1, iy1 = max(box_a[0], box_b[0]), max(box_a[1], box_b[1])
            ix2, iy2 = min(box_a[2], box_b[2]), min(box_a[3], box_b[3])
            inter_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
            if inter_area / area_a > 0.7: keep = False; break
        if keep: keep_indices.append(i)
    return [boxes[idx] for idx in keep_indices]

def detect_particles_heuristically(img):
    """AI 실패 시 백업 탐지 (CV 로직)"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask_o = cv2.inRange(hsv, np.array([0, 40, 40]), np.array([45, 255, 255]))
    mask_c = cv2.inRange(hsv, np.array([85, 30, 30]), np.array([165, 255, 255]))
    combined = cv2.bitwise_or(mask_o, mask_c)
    kernel = np.ones((25,25), np.uint8)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    found_boxes = []
    
    class FakeBox:
        def __init__(self, coords):
            import torch
            self.xyxy = torch.tensor([coords], dtype=torch.float32)
            self.conf = torch.tensor([0.15])
            
    for cnt in contours:
        if cv2.contourArea(cnt) > 3000:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w)/h if h>0 else 0
            if 0.2 < aspect_ratio < 5.0:
                found_boxes.append(FakeBox([x, y, x+w, y+h]))
    return found_boxes

def process_frame(img):
    """이미지 처리 메인 파이프라인"""
    img = apply_gamma_correction(img, gamma=0.8)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_h, img_w = img.shape[:2]

    if model:
        results = model.predict(source=img, conf=0.10, iou=0.45, verbose=False)
        ai_raw_boxes = filter_nested_boxes(results[0].boxes)
    else:
        ai_raw_boxes = []

    combined_boxes = []
    # AI 박스 유효성 검증 (신호 강도 체크)
    for box in ai_raw_boxes:
        coords = box.xyxy[0].cpu().numpy().flatten(); x1, y1, x2, y2 = map(int, coords)
        if (x2-x1) < 50 or (y2-y1) < 50: continue
        roi_hsv = hsv[max(0,y1):min(img_h,y2), max(0,x1):min(img_w,x2)]
        if roi_hsv.size == 0: continue
        m_o = cv2.inRange(roi_hsv, np.array([0, 35, 35]), np.array([55, 255, 255]))
        m_c = cv2.inRange(roi_hsv, np.array([80, 30, 30]), np.array([165, 255, 255]))
        if np.sum(m_o > 0) + np.sum(m_c > 0) > 200: combined_boxes.append((box, "AI"))

    # AI 미검출 시 CV 백업 가동
    if not combined_boxes:
        cv_raw_boxes = detect_particles_heuristically(img)
        for cv_box in cv_raw_boxes:
            combined_boxes.append((cv_box, "CV_BACKUP"))
            
    reports = []
    draw_img = img.copy()

    for i, (box, method) in enumerate(combined_boxes):
        coords = box.xyxy[0].cpu().numpy().flatten(); x1, y1, x2, y2 = map(int, coords)
        pad = int((x2-x1)*0.15); nx1=max(0,x1-pad); ny1=max(0,y1-pad); nx2=min(img_w,x2+pad); ny2=min(img_h,y2+pad)
        roi_hsv = hsv[ny1:ny2, nx1:nx2]; roi_img = img[ny1:ny2, nx1:nx2]
        if roi_hsv.size == 0: continue
        
        valid_mask = (roi_hsv[:,:,1]>25) & (roi_hsv[:,:,2]>25)
        mask_orange = cv2.inRange(roi_hsv, np.array([0, 30, 30]), np.array([60, 255, 255]))
        mask_cyan_candidate = cv2.inRange(roi_hsv, np.array([80, 30, 30]), np.array([165, 255, 255]))
        
        mask_particle_body = np.zeros_like(mask_orange)
        closed_orange = cv2.morphologyEx(mask_orange & (valid_mask.astype(np.uint8)*255), cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
        contours, _ = cv2.findContours(closed_orange, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        p_count = 0
        for cnt in contours:
            if cv2.contourArea(cnt) > 20:
                cv2.drawContours(mask_particle_body, [cnt], -1, 255, -1)
                p_count += cv2.contourArea(cnt)
        
        box_area = (nx2 - nx1) * (ny2 - ny1)
        orange_area_pct = (p_count / box_area) * 100 if box_area > 0 else 0

        # 판정 로직 v10.2.1
        if p_count < 100 or orange_area_pct < 3.0:
            status = "RECHECK REQUIRED"
            cv_color = (0, 165, 255) # Orange BGR
            phi = 0
            cyan_area = 0
        else:
            mask_containment_zone = cv2.dilate(mask_particle_body, np.ones((3,3), np.uint8), iterations=1)
            mask_cyan = cv2.bitwise_and(mask_cyan_candidate, mask_containment_zone)
            
            b_ch, g_ch, r_ch = cv2.split(roi_img.astype(float))
            is_glare = (g_ch > 200) & (b_ch > 200) & (r_ch > 200)
            is_saturated_cyan = (g_ch > 200) & (b_ch > 200) & (r_ch < 200)
            mask_saturated_valid = (is_saturated_cyan.astype(np.uint8) * 255) & mask_containment_zone
            saturated_pixels = np.sum(mask_saturated_valid > 0)

            intensity_raw = np.where(is_glare, 0, np.where(is_saturated_cyan, ((g_ch + b_ch)/2.0 - r_ch*0.8), ((g_ch + b_ch)/2.0 - r_ch*1.7)))
            intensity_map = np.clip(intensity_raw, 0, 100)
            
            cyan_area = (np.sum(mask_cyan>0)/p_count*100) if p_count>0 else 0
            avg_int = np.mean(intensity_map[mask_cyan>0]) if np.sum(mask_cyan>0)>0 else 0
            phi = cyan_area * (avg_int / 10.0)
            
            status = "CONTAMINATED" if (phi > 5.0 or saturated_pixels > 20) else "SAFE"
            if status == "CONTAMINATED" and phi < 5.0: phi = 99.9
            cv_color = (255, 255, 0) if status == "CONTAMINATED" else (0, 255, 0)

        cv2.rectangle(draw_img, (nx1, ny1), (nx2, ny2), cv_color, 4)
        label_text = f"{status[:4]} P:{phi:.1f}"
        cv2.putText(draw_img, label_text, (nx1, ny1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, cv_color, 2)
        
        reports.append({
            "id": i, "status": status, "phi": float(round(phi, 2)), 
            "cyan": float(round(cyan_area, 2)), "orange": float(round(orange_area_pct, 2))
        })

    return draw_img, reports

# --- UI 실행 ---
st.title("🧪 TFCP Intelligent Analyzer")
st.markdown("---")

if model is None:
    st.error("⚠️ 'best.pt' file missing. Please check GitHub repository.")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📷 Capture")
    img_file = st.camera_input("Take a photo")
    if not img_file:
        st.info("Or upload file:")
        img_file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])

if img_file:
    # 이미지 로드
    file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    
    # 분석 실행
    result_img, reports = process_frame(image)
    
    # 데이터 자동 저장 (로컬 JSON/JPG)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_name = f"TFCP_{timestamp}"
    cv2.imwrite(os.path.join(IMG_DIR, f"{save_name}.jpg"), image)
    with open(os.path.join(LOG_DIR, f"{save_name}.json"), "w") as f:
        json.dump({"filename": f"{save_name}.jpg", "reports": reports}, f, indent=4)

    # 결과 출력
    with col1:
        st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), caption=f"Analyzed: {save_name}", use_container_width=True)
        st.toast("Data Saved!", icon="💾")
    
    with col2:
        st.subheader("📊 Data")
        if reports:
            for r in reports:
                # 상태별 색상 지정
                c_map = {"CONTAMINATED": "red", "SAFE": "green", "RECHECK REQUIRED": "orange"}
                s_color = c_map.get(r['status'], "grey")
                
                st.markdown(f'''
                <div style="border: 2px solid {s_color}; border-radius: 10px; padding: 10px; margin-bottom: 10px;">
                    <h4 style="color: {s_color}; margin: 0;">ID {r['id']}: {r['status']}</h4>
                    <p style="margin: 5px 0;"><b>Phi:</b> {r['phi']} | <b>Cyan:</b> {r['cyan']}%</p>
                    <p style="margin: 5px 0; color: grey; font-size: 0.8em;">Density: {r['orange']}%</p>
                </div>
                ''', unsafe_allow_html=True)
        else:
            st.warning("No particles detected.")
