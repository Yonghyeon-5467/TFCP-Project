import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import os
import json
import shutil
from datetime import datetime
import pandas as pd

# --- [1] 페이지 및 기본 설정 ---
st.set_page_config(page_title="TFCP Data Manager", page_icon="🧪", layout="wide")

# 저장소 경로 설정
SAVE_ROOT = "TFCP_Data"
IMG_DIR = os.path.join(SAVE_ROOT, "raw_images")
LOG_DIR = os.path.join(SAVE_ROOT, "analysis_logs")

if not os.path.exists(IMG_DIR): os.makedirs(IMG_DIR)
if not os.path.exists(LOG_DIR): os.makedirs(LOG_DIR)

# --- [2] 모델 로드 ---
@st.cache_resource
def load_model():
    if os.path.exists('best.pt'):
        return YOLO('best.pt')
    return None

model = load_model()

# --- [3] 핵심 분석 엔진 ---

def apply_gamma_correction(image, gamma=0.8):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

# [복구] 이미지 크기 고정 함수 (Letterbox)
def standardize_image_size(img, target_width=800, target_height=600):
    """이미지 비율 유지하며 고정 크기로 리사이징 (빈 공간은 검은색)"""
    h, w = img.shape[:2]
    scale = min(target_width/w, target_height/h)
    nw, nh = int(w*scale), int(h*scale)
    resized = cv2.resize(img, (nw, nh))
    
    delta_w = target_width - nw
    delta_h = target_height - nh
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    
    new_img = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return new_img

def calculate_iou(box1, box2):
    b1, b2 = box1.flatten(), box2.flatten()
    ix1, iy1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    ix2, iy2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area1 = (b1[2]-b1[0])*(b1[3]-b1[1])
    area2 = (b2[2]-b2[0])*(b2[3]-b2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0

def filter_nested_boxes(boxes):
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
    img = apply_gamma_correction(img, gamma=0.8)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_h, img_w = img.shape[:2]

    if model:
        results = model.predict(source=img, conf=0.10, iou=0.45, verbose=False)
        ai_raw_boxes = filter_nested_boxes(results[0].boxes)
    else: ai_raw_boxes = []

    combined_boxes = []
    for box in ai_raw_boxes:
        coords = box.xyxy[0].cpu().numpy().flatten(); x1, y1, x2, y2 = map(int, coords)
        if (x2-x1) < 50 or (y2-y1) < 50: continue
        roi_hsv = hsv[max(0,y1):min(img_h,y2), max(0,x1):min(img_w,x2)]
        if roi_hsv.size == 0: continue
        if np.sum(cv2.inRange(roi_hsv, np.array([0, 35, 35]), np.array([55, 255, 255]))) + np.sum(cv2.inRange(roi_hsv, np.array([80, 30, 30]), np.array([165, 255, 255]))) > 200:
            combined_boxes.append((box, "AI"))

    if not combined_boxes:
        for cv_box in detect_particles_heuristically(img): combined_boxes.append((cv_box, "CV_BACKUP"))
            
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
        contours, _ = cv2.findContours(cv2.morphologyEx(mask_orange & (valid_mask.astype(np.uint8)*255), cv2.MORPH_CLOSE, np.ones((5,5), np.uint8)), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        p_count = sum(cv2.contourArea(cnt) for cnt in contours if cv2.contourArea(cnt) > 20)
        
        box_area = (nx2-nx1)*(ny2-ny1)
        orange_area_pct = (p_count/box_area)*100 if box_area>0 else 0

        if p_count < 100 or orange_area_pct < 3.0:
            status = "RECHECK REQUIRED"; cv_color = (0, 165, 255); phi = 0; cyan_area = 0
            avg_int = 0
        else:
            mask_containment_zone = cv2.dilate(mask_particle_body, np.ones((3,3), np.uint8), iterations=1)
            mask_cyan = cv2.bitwise_and(mask_cyan_candidate, mask_containment_zone)
            b_ch, g_ch, r_ch = cv2.split(roi_img.astype(float))
            intensity_raw = np.where((g_ch>200)&(b_ch>200)&(r_ch>200), 0, np.where((g_ch>200)&(b_ch>200)&(r_ch<200), ((g_ch+b_ch)/2.0-r_ch*0.8), ((g_ch+b_ch)/2.0-r_ch*1.7)))
            cyan_area = (np.sum(mask_cyan>0)/p_count*100) if p_count>0 else 0
            avg_int = np.mean(np.clip(intensity_raw,0,100)[mask_cyan>0]) if np.sum(mask_cyan>0)>0 else 0
            phi = cyan_area * (avg_int / 10.0)
            status = "CONTAMINATED" if (phi > 5.0 or np.sum((mask_cyan>0) & ((g_ch>200)&(b_ch>200)&(r_ch<200)).astype(np.uint8))>0) else "SAFE"
            if status == "CONTAMINATED" and phi < 5.0: phi = 99.9
            cv_color = (255, 255, 0) if status == "CONTAMINATED" else (0, 255, 0)

        cv2.rectangle(draw_img, (nx1, ny1), (nx2, ny2), cv_color, 4)
        label_text = f"Area {i+1}: {status[:4]}" if status != "RECHECK REQUIRED" else f"Area {i+1}: RECHECK"
        cv2.putText(draw_img, label_text, (nx1, ny1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, cv_color, 2)
        
        reports.append({"id": i, "status": status, "phi": float(round(phi, 2)), "cyan": float(round(cyan_area, 2)), "orange": float(round(orange_area_pct, 2)), "box": [int(nx1), int(ny1), int(nx2), int(ny2)]})
    return draw_img, reports

# --- [4] UI: 관리자 페이지 ---
def render_admin_page():
    st.title("🗂️ 연구 데이터 관리 센터")
    
    log_files = sorted([f for f in os.listdir(LOG_DIR) if f.endswith('.json')], reverse=True)
    if not log_files: st.warning("저장된 데이터가 없습니다."); return

    if 'current_log_file' not in st.session_state or st.session_state.current_log_file not in log_files:
        st.session_state.current_log_file = log_files[0]
    
    current_idx = log_files.index(st.session_state.current_log_file)
    c1, c2, c3 = st.columns([1,4,1])
    with c1: 
        if st.button("◀️ PREV", use_container_width=True):
            st.session_state.current_log_file = log_files[max(0, current_idx - 1)]; st.rerun()
    with c3:
        if st.button("NEXT ▶️", use_container_width=True):
            st.session_state.current_log_file = log_files[min(len(log_files)-1, current_idx + 1)]; st.rerun()
    with c2:
        def update_index(): st.session_state.current_log_file = st.session_state.log_selector
        st.selectbox("파일 선택", log_files, index=current_idx, key='log_selector', on_change=update_index, label_visibility="collapsed")
        
        # [백업 버튼]
        if st.button("📦 전체 백업 (ZIP)", use_container_width=True):
            shutil.make_archive("TFCP_Backup", 'zip', SAVE_ROOT)
            with open("TFCP_Backup.zip", "rb") as fp: st.download_button("📥 다운로드", fp, "TFCP_Backup.zip", "application/zip")
            
        # [삭제 버튼]
        if st.button("🗑️ 현재 데이터 삭제", type="primary", use_container_width=True):
            log_path_del = os.path.join(LOG_DIR, st.session_state.current_log_file)
            try:
                with open(log_path_del, 'r') as f: del_data = json.load(f)
                if os.path.exists(log_path_del): os.remove(log_path_del)
                if os.path.exists(os.path.join(IMG_DIR, del_data['filename'])): os.remove(os.path.join(IMG_DIR, del_data['filename']))
                st.success("삭제됨"); del st.session_state.current_log_file; st.rerun()
            except: st.error("삭제 실패")

    log_path = os.path.join(LOG_DIR, st.session_state.current_log_file)
    try:
        with open(log_path, 'r') as f: data = json.load(f)
        img_path = os.path.join(IMG_DIR, data['filename'])
        if os.path.exists(img_path):
            img_bgr = cv2.imread(img_path)
            img_corrected = apply_gamma_correction(img_bgr, gamma=0.8)
            img_rgb = cv2.cvtColor(img_corrected, cv2.COLOR_BGR2RGB)
            draw_img = img_rgb.copy()
            
            particles = data.get('particles', data.get('reports', []))
            
            if particles:
                for idx, p in enumerate(particles):
                    if 'box' not in p: continue
                    x1,y1,x2,y2 = p['box']
                    status = p.get('status', 'SAFE')
                    color = (0, 255, 0)
                    if status == "CONTAMINATED": color = (255, 0, 0)
                    elif status == "RECHECK REQUIRED": color = (255, 165, 0)
                    cv2.rectangle(draw_img, (x1, y1), (x2, y2), color, 4)
                    label_text = f"Area {idx + 1}: {status[:4]}"
                    if status == "RECHECK REQUIRED": label_text = f"Area {idx + 1}: RECHECK"
                    cv2.putText(draw_img, label_text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                
                # [적용] 이미지 크기 고정 (800x600)
                display_img = standardize_image_size(draw_img, 800, 600)
                st.image(display_img, caption=f"Analyzed: {data.get('timestamp','Unknown')}", width=800)

                # 수정 폼
                with st.form("update"):
                    new_parts = []
                    cols = st.columns(2)
                    for i, p in enumerate(particles):
                        with cols[i%2]:
                            stat = p.get('status','SAFE')
                            st.write(f"**Area {i+1}**: {stat}")
                            idx = ["SAFE","CONTAMINATED","RECHECK REQUIRED"].index(stat) if stat in ["SAFE","CONTAMINATED","RECHECK REQUIRED"] else 0
                            new_stat = st.radio("상태", ["SAFE","CONTAMINATED","RECHECK REQUIRED"], index=idx, key=f"rad_{i}", horizontal=True)
                            p['status'] = new_stat
                            p['id'] = i
                            new_parts.append(p)
                    if st.form_submit_button("저장"):
                        data['particles'] = new_parts
                        data['reports'] = new_parts
                        data['reviewed'] = True
                        with open(log_path, 'w') as f: json.dump(data, f, indent=4)
                        st.success("저장됨"); st.rerun()
            else:
                st.image(img_rgb, caption="입자 없음", width=800) # 너비 고정
                st.warning("입자가 없습니다. 수동으로 추가하세요.")
                if st.button("➕ 중앙에 입자 추가"):
                        h, w = img_rgb.shape[:2]
                        new_p = {"id":0, "box":[int(w*0.3),int(h*0.3),int(w*0.7),int(h*0.7)], "status":"CONTAMINATED", "phi":0, "cyan":0, "orange":0, "manual":True}
                        data['particles'] = [new_p]; data['reports'] = [new_p]
                        with open(log_path, 'w') as f: json.dump(data, f, indent=4)
                        st.rerun()
        else: st.error("이미지 없음")
    except Exception as e: st.error(f"오류: {e}")

elif mode == "실시간 분석":
    st.title("🧪 TFCP 분석기")
    c1, c2 = st.columns([2,1])
    with c1:
        img_file = st.camera_input("촬영")
        if not img_file: img_file = st.file_uploader("업로드", type=['jpg','png','jpeg'])
    
    if img_file:
        file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        if image is None: st.error("로드 실패")
        else:
            try:
                res_img, reports = process_frame(image)
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                fn = f"TFCP_{ts}"
                cv2.imwrite(os.path.join(IMG_DIR, f"{fn}.jpg"), image)
                with open(os.path.join(LOG_DIR, f"{fn}.json"), "w") as f:
                    json.dump({"filename":f"{fn}.jpg", "timestamp":ts, "reports":reports, "reviewed":False}, f, indent=4)
                
                # [적용] 이미지 크기 고정 (800x600) + RGB 변환
                display_img = standardize_image_size(res_img, 800, 600)
                with c1: st.image(cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB), caption="분석 완료", width=800)
                with c2:
                    if reports:
                        for r in reports:
                            clr = "red" if r['status']=="CONTAMINATED" else "green" if r['status']=="SAFE" else "orange"
                            st.markdown(f'<div style="border:2px solid {clr}; padding:5px; margin:5px; border-radius:5px;">Area {r["id"]+1}: <b>{r["status"]}</b><br>Phi: {r["phi"]}</div>', unsafe_allow_html=True)
                    else: st.warning("입자 없음")
            except Exception as e: st.error(f"오류: {e}")
