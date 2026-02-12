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
st.set_page_config(page_title="TFCP Data Manager", page_icon="🧪", layout="wide")

st.markdown("""
    <style>
        .main { background-color: #f8f9fa; }
        .stButton>button { width: 100%; border-radius: 6px; font-weight: 600; font-family: 'Helvetica', sans-serif; }
        .metric-card {
            background-color: white; border: 1px solid #e1e4e8; border-radius: 10px;
            padding: 20px; margin-bottom: 12px; box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        }
        .status-safe { color: #28a745; font-weight: 800; }
        .status-cont { color: #dc3545; font-weight: 800; }
        .status-warn { color: #fd7e14; font-weight: 800; }
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

# --- [3] Visualization Helper (High-Res Font) ---
@st.cache_resource
def get_custom_font(size=20):
    """
    Downloads and loads a clean sans-serif font (Roboto-Bold).
    """
    font_path = "Roboto-Bold.ttf"
    font_url = "https://github.com/google/fonts/raw/main/apache/roboto/Roboto-Bold.ttf"
    
    if not os.path.exists(font_path):
        try:
            r = requests.get(font_url, timeout=5)
            if r.status_code == 200:
                with open(font_path, "wb") as f:
                    f.write(r.content)
        except: pass

    try:
        return ImageFont.truetype(font_path, size)
    except:
        return ImageFont.load_default()

def draw_smart_annotations(img_bgr, reports):
    """
    Draws professional annotations using PIL (RGB).
    Uses v10.2.1 logic results.
    """
    # Convert BGR to RGB for PIL
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb).convert("RGBA")
    overlay = Image.new("RGBA", pil_img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    
    h, w = img_bgr.shape[:2]
    # Scale factors for visibility
    scale = max(w, h) / 1000.0
    line_width = max(3, int(5 * scale))
    font_size = max(20, int(35 * scale)) # 폰트 크기 크게 설정
    font = get_custom_font(font_size)
    
    # Sort reports to draw CONTAMINATED on top
    def get_priority(r):
        s = r['status']
        if s == "CONTAMINATED": return 2
        if s == "RECHECK REQUIRED": return 1
        return 0
    reports_sorted = sorted(reports, key=get_priority)
    
    for r in reports_sorted:
        x1, y1, x2, y2 = r['box']
        status = r['status']
        p_id = r['id']
        
        if status == "CONTAMINATED":
            color_rgb = (220, 53, 69) # Red
        elif status == "RECHECK REQUIRED":
            color_rgb = (253, 126, 20) # Orange
        else:
            color_rgb = (40, 167, 69) # Green
            
        # Draw Box
        draw.rectangle([x1, y1, x2, y2], outline=color_rgb + (255,), width=line_width)
        
        # Prepare Label
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
            
        pad = int(10 * scale)
        lbl_y1 = y1 - text_h - 2*pad
        if lbl_y1 < 0: lbl_y1 = y1
        
        # Draw Label Background
        draw.rectangle([x1, lbl_y1, x1 + text_w + 2*pad, lbl_y1 + text_h + 2*pad], fill=color_rgb + (230,))
        # Draw Text
        draw.text((x1 + pad, lbl_y1 + pad), label_txt, font=font, fill=(255, 255, 255, 255))
        
    final_img = Image.alpha_composite(pil_img, overlay)
    return np.array(final_img.convert("RGB"))

# --- [4] Core Analysis Logic (v10.2.1 Original Logic) ---

def apply_gamma_correction(image, gamma=0.8):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def standardize_image_size(img, target_width=800, target_height=600):
    """
