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
import requests

# --- [1] Page Config & CSS ---
st.set_page_config(page_title="TFCP Quantitative Analysis System", page_icon="🔬", layout="wide")

st.markdown("""
    <style>
        .main { background-color: #f8f9fa; }
        .stButton>button { width: 100%; border-radius: 6px; font-weight: 600; font-family: 'Roboto', sans-serif; }
        .metric-card {
            background-color: white; border: 1px solid #e1e4e8; border-radius: 10px;
            padding: 20px; margin-bottom: 12px; box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        }
        .status-safe { color: #28a745; font-weight: 800; }
        .status-cont { color: #dc3545; font-weight: 800; }
        .status-recheck { color: #fd7e14; font-weight: 800; }
        .header-text { font-family: 'Helvetica Neue', sans-serif; font-weight: 700; color: #1f2937; }
    </style>
""", unsafe_allow_html=True)

# --- [2] Storage & Model Setup ---
SAVE_ROOT = "TFCP_Data"
IMG_DIR = os.path.join(SAVE_ROOT, "raw_images")
LOG_DIR = os.path.join(SAVE_ROOT, "analysis_logs")

if not os.path.exists(IMG_DIR): os.makedirs(IMG_DIR)
if not os.path.exists(LOG_DIR): os.makedirs(LOG_DIR)

@st.cache_resource
def load_model():
    if os.path.exists('best.pt'):
        return YOLO('best.pt')
    return None

model = load_model()

# --- [3] Visualization Helper (Custom Font) ---
@st.cache_resource
def get_custom_font(size=20):
    """
    Download a high-quality font (Roboto-Bold) dynamically.
    This ensures the font looks 'pretty' on any server (Streamlit Cloud/Linux).
    """
    font_path = "Roboto-Bold.ttf"
    font_url = "https://github.com/google/fonts/raw/main/apache/roboto/Roboto-Bold.ttf"
    
    if not os.path.exists(font_path):
        try:
            r = requests.get(font_url)
            with open(font_path, "wb") as f:
                f.write(r.content)
        except:
            pass # Fail silently and use default if network fails

    try:
        return ImageFont.truetype(font_path, size)
    except:
        return ImageFont.load_default()

def draw_smart_annotations(img_bgr, reports):
    """
    Draws professional annotations using PIL with custom fonts and semi-transparent backgrounds.
    """
    # 1. Convert to RGB for PIL
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb).convert("RGBA")
    
    # 2. Create Overlay for transparency
    overlay = Image.new("RGBA", pil_img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    
    h, w = img_bgr.shape[:2]
    # Dynamic scaling
    scale = max(w, h) / 1000.0
    line_width = max(3, int(5 * scale))
    font_size = max(16, int(28 * scale)) # 폰트 크기 키움
    font = get_custom_font(font_size)
    
    for r in reports:
        x1, y1, x2, y2 = r['box']
        status = r['status']
        p_id = r['id']
        
        # Color Palette (Nature-style: Soft but Vivid)
        if status == "CONTAMINATED":
            color_rgb = (220, 53, 69) # Red
        elif status == "RECHECK REQUIRED":
            color_rgb = (253, 126, 20) # Orange
        else:
            color_rgb = (40, 167, 69) # Green
            
        # Draw Box
        draw.rectangle([x1, y1, x2, y2], outline=color_rgb + (255,), width=line_width)
        
        # Prepare Label Text
        label_txt = f"Area {p_id + 1}"
        if status == "RECHECK REQUIRED": label_txt = "RECHECK"
        elif status == "CONTAMINATED": label_txt += ": CONT"
        else: label_txt += ": SAFE"
            
        # Measure Text
        try:
            bbox = font.getbbox(label_txt)
            text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except:
            text_w, text_h = draw.textsize(label_txt, font=font)
            
        pad = int(8 * scale) # 여백을 넉넉하게
        
        # Label Position (Above box)
        lbl_y1 = y1 - text_h - 2*pad
        if lbl_y1 < 0: lbl_y1 = y1 # If runs off top, draw inside
        
        # Draw Label Background (Semi-transparent)
        # Using the same color as box but with opacity
        bg_color = color_rgb + (200,) # 200/255 opacity
        
        draw.rectangle(
            [x1, lbl_y1, x1 + text_w + 2*pad, lbl_y1 + text_h + 2*pad],
            fill=bg_color
        )
        
        # Draw Text (White)
        draw.text((x1 + pad, lbl_y1 + pad), label_txt, font=font, fill=(255, 255, 255, 255))
        
    # Combine original image with overlay
    final_img = Image.alpha_composite(pil_img, overlay)
    
    # Convert back to numpy (RGB) for Streamlit
    return np.array(final_img.convert("RGB"))

# --- [4] Core Analysis Logic (v10.2.1) ---

def apply_gamma_correction(image, gamma=0.8):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def standardize_image_size(img, target_width=800, target_height=600):
    h, w = img.shape[:2]
    scale = min(target_width/w, target_height/h)
    nw, nh = int(w*scale), int(h*scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    new_img = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    top, left = (target_height - nh) // 2, (target_width - nw) // 2
    new_img[top:top+nh, left:left+nw] = resized
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
            if calculate_iou(box_a, box_b) > 0.3: keep = False; break
            ix1, iy1 = max(box_a[0], box_b[0]), max(box_a[1], box_b[1])
            ix2, iy2 = min(box_a[2], box_b[2]), min(box_a[3], box_b[3])
            if (max(0, ix2-ix1)*max(0, iy2-iy1)) / ((box_a[2]-box_a[0])*(box_a[3]-box_a[1])) > 0.7: keep = False; break
        if keep: keep_indices.append(i)
    return [boxes[idx] for idx in keep_indices]

def detect_particles_heuristically(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask_o = cv2.inRange(hsv, np.array([0, 40, 40]), np.array([60, 255, 255]))
    mask_c = cv2.inRange(hsv, np.array([80, 30, 30]), np.array([165, 255, 255]))
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
    img_corr = apply_gamma_correction(img, gamma=0.8)
    hsv = cv2.cvtColor(img_corr, cv2.COLOR_BGR2HSV)
    img_h, img_w = img.shape[:2]

    if model:
        results = model.predict(source=img_corr, conf=0.10, iou=0.45, verbose=False)
        ai_raw_boxes = filter_nested_boxes(results[0].boxes)
    else: ai_raw_boxes = []

    combined_boxes = []
    for box in ai_raw_boxes:
        coords = box.xyxy[0].cpu().numpy().flatten(); x1, y1, x2, y2 = map(int, coords)
        if (x2-x1) < 50 or (y2-y1) < 50: continue
        roi_hsv = hsv[max(0,y1):min(img_h,y2), max(0,x1):min(img_w,x2)]
        if roi_hsv.size == 0: continue
        if np.sum(cv2.inRange(roi_hsv, np.array([0, 35, 35]), np.array([60, 255, 255]))) + np.sum(cv2.inRange(roi_hsv, np.array([80, 30, 30]), np.array([165, 255, 255]))) > 200:
            combined_boxes.append((box, "AI"))

    if not combined_boxes:
        for cv_box in detect_particles_heuristically(img_corr): combined_boxes.append((cv_box, "CV_BACKUP"))
            
    reports = []

    for i, (box, method) in enumerate(combined_boxes):
        coords = box.xyxy[0].cpu().numpy().flatten(); x1, y1, x2, y2 = map(int, coords)
        pad = int((x2-x1)*0.15); nx1=max(0,x1-pad); ny1=max(0,y1-pad); nx2=min(img_w,x2+pad); ny2=min(img_h,y2+pad)
        roi_hsv = hsv[ny1:ny2, nx1:nx2]; roi_img = img_corr[ny1:ny2, nx1:nx2]
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
            status = "RECHECK REQUIRED"; phi = 0; cyan_area = 0; avg_int = 0
        else:
            mask_containment_zone = cv2.dilate(mask_particle_body, np.ones((3,3), np.uint8), iterations=1)
            if p_count < 50: mask_containment_zone = np.ones_like(mask_orange) * 255
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

        reports.append({"id": i, "status": status, "phi": float(round(phi, 2)), "cyan": float(round(cyan_area, 2)), "orange": float(round(orange_area_pct, 2)), "box": [int(nx1), int(ny1), int(nx2), int(ny2)]})
    
    # [v24] Enhanced Drawing
    final_img = draw_smart_annotations(img.copy(), reports)
    return final_img, reports

# --- UI (Admin) ---
def render_admin_page():
    st.markdown("<h2 class='header-text'>Research Data Management</h2>", unsafe_allow_html=True)
    
    log_files = sorted([f for f in os.listdir(LOG_DIR) if f.endswith('.json')], reverse=True)
    if not log_files: st.info("No data available."); return

    if 'current_log_file' not in st.session_state: st.session_state.current_log_file = log_files[0]
    if st.session_state.current_log_file not in log_files: st.session_state.current_log_file = log_files[0]
    
    current_idx = log_files.index(st.session_state.current_log_file)
    c1, c2, c3 = st.columns([1, 4, 1])
    with c1: 
        if st.button("◀ PREV"):
            st.session_state.current_log_file = log_files[max(0, current_idx - 1)]; st.rerun()
    with c3:
        if st.button("NEXT ▶"):
            st.session_state.current_log_file = log_files[min(len(log_files)-1, current_idx + 1)]; st.rerun()
    with c2:
        def update_index(): st.session_state.current_log_file = st.session_state.log_selector
        st.selectbox("Select Log", log_files, index=current_idx, key='log_selector', on_change=update_index, label_visibility="collapsed")
        
        b1, b2 = st.columns(2)
        with b1:
            if st.button("📦 Archive (ZIP)", use_container_width=True):
                shutil.make_archive("TFCP_Dataset", 'zip', SAVE_ROOT)
                with open("TFCP_Dataset.zip", "rb") as fp: st.download_button("Download ZIP", fp, "TFCP_Dataset.zip", "application/zip")
        with b2:
            if st.button("🗑️ Delete", type="primary", use_container_width=True):
                try:
                    f_path = os.path.join(LOG_DIR, st.session_state.current_log_file)
                    with open(f_path,'r') as f: d = json.load(f)
                    if os.path.exists(f_path): os.remove(f_path)
                    img_p = os.path.join(IMG_DIR, d['filename'])
                    if os.path.exists(img_p): os.remove(img_p)
                    st.success("Deleted."); del st.session_state.current_log_file; st.rerun()
                except: st.error("Delete failed")

    log_path = os.path.join(LOG_DIR, st.session_state.current_log_file)
    try:
        with open(log_path, 'r') as f: data = json.load(f)
        img_path = os.path.join(IMG_DIR, data['filename'])
        if os.path.exists(img_path):
            img_bgr = cv2.imread(img_path)
            # Re-process for display to ensure fresh high-quality drawing
            particles = data.get('particles', data.get('reports', []))
            draw_img = draw_smart_annotations(img_bgr.copy(), particles)
            display_img = standardize_image_size(draw_img, 800, 600)
            
            st.image(display_img, caption=f"Analyzed: {data.get('timestamp','Unknown')}", width=800)

            with st.expander("➕ Manual Region Injection (Slider)", expanded=False):
                h, w = img_bgr.shape[:2]
                mc1, mc2 = st.columns(2)
                with mc1:
                    mx1 = st.slider("X Start", 0, w, int(w*0.3), key="mx1")
                    mx2 = st.slider("X End", 0, w, int(w*0.7), key="mx2")
                with mc2:
                    my1 = st.slider("Y Start", 0, h, int(h*0.3), key="my1")
                    my2 = st.slider("Y End", 0, h, int(h*0.7), key="my2")
                
                # Preview on PIL image
                preview_arr = img_bgr.copy()
                cv2.rectangle(preview_arr, (mx1, my1), (mx2, my2), (255, 0, 255), 4)
                preview_rgb = cv2.cvtColor(preview_arr, cv2.COLOR_BGR2RGB)
                st.image(standardize_image_size(preview_rgb, 800, 600), caption="Preview", width=800)
                
                if st.button("✅ Inject"):
                    new_particle = {"id": len(particles), "box": [mx1, my1, mx2, my2], "status": "CONTAMINATED", "phi": 0, "cyan": 0, "orange": 0, "manual": True}
                    particles.append(new_particle)
                    data['particles'] = particles
                    data['reports'] = particles
                    with open(log_path, 'w') as f: json.dump(data, f, indent=4)
                    st.success("Injected!"); st.rerun()

            with st.form("update"):
                st.markdown("#### Annotation Correction")
                new_parts = []
                cols = st.columns(2)
                for i, p in enumerate(particles):
                    with cols[i%2]:
                        stat = p.get('status','SAFE')
                        cls = "status-cont" if stat=="CONTAMINATED" else "status-safe" if stat=="SAFE" else "status-warn"
                        st.markdown(f"**Area {i+1}**: <span class='{cls}'>{stat}</span>", unsafe_allow_html=True)
                        idx = ["SAFE","CONTAMINATED","RECHECK REQUIRED"].index(stat) if stat in ["SAFE","CONTAMINATED","RECHECK REQUIRED"] else 0
                        new_stat = st.radio("Status", ["SAFE","CONTAMINATED","RECHECK REQUIRED"], index=idx, key=f"rad_{i}", horizontal=True)
                        p['status'] = new_stat
                        p['id'] = i
                        new_parts.append(p)
                if st.form_submit_button("Save Annotations"):
                    data['particles'] = new_parts
                    data['reports'] = new_parts
                    data['reviewed'] = True
                    with open(log_path, 'w') as f: json.dump(data, f, indent=4)
                    st.success("Saved!"); st.rerun()
        else: st.error("Image missing")
    except Exception as e: st.error(f"Data Error: {e}")

# --- UI (Main) ---
if 'admin_mode' not in st.session_state: st.session_state['admin_mode'] = False
st.sidebar.title("Navigation")
mode = st.sidebar.radio("Go to", ["Real-time Inference", "Admin Console"])

if mode == "Admin Console":
    if not st.session_state['admin_mode']:
        pwd = st.sidebar.text_input("Access Key", type="password")
        if pwd == "tfcp2026": st.session_state['admin_mode'] = True; st.rerun()
        elif pwd: st.error("Invalid Key")
    if st.session_state['admin_mode']: render_admin_page()

elif mode == "Real-time Inference":
    st.markdown("<h1 class='header-text'>TFCP Inference Engine</h1>", unsafe_allow_html=True)
    c1, c2 = st.columns([2,1])
    with c1:
        img_file = st.camera_input("Acquire")
        if not img_file: img_file = st.file_uploader("Upload", type=['jpg','png','jpeg'])
    
    if img_file:
        file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        if image is None: st.error("Load Failed")
        else:
            try:
                res_img_rgb, reports = process_frame(image)
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                fn = f"TFCP_{ts}"
                cv2.imwrite(os.path.join(IMG_DIR, f"{fn}.jpg"), image)
                with open(os.path.join(LOG_DIR, f"{fn}.json"), "w") as f:
                    json.dump({"filename":f"{fn}.jpg", "timestamp":ts, "reports":reports, "reviewed":False}, f, indent=4)
                
                # process_frame now returns RGB numpy array from PIL
                display_img = standardize_image_size(res_img_rgb, 800, 600)
                with c1: st.image(display_img, caption="Analysis Result", width=800)
                with c2:
                    st.markdown("### Metrics")
                    if reports:
                        for r in reports:
                            cls = "status-cont" if r['status']=="CONTAMINATED" else "status-safe" if r['status']=="SAFE" else "status-warn"
                            st.markdown(f"""
                            <div class="metric-card">
                                <div><strong>Area {r['id']+1}</strong></div>
                                <div class="{cls}">{r['status']}</div>
                                <div style="font-size:0.85em; color:#666; margin-top:5px;">
                                    Φ: {r['phi']}<br>Cyan: {r['cyan']}%
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                    else: st.warning("No particles")
            except Exception as e: st.error(f"Error: {e}")
