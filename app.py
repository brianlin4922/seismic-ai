import streamlit as st
import time
import os
import re
import cv2
import numpy as np
from PIL import Image, ImageDraw
from pydantic import BaseModel, Field
from skimage.filters import meijering
from skimage.morphology import skeletonize
from skimage.measure import label, regionprops
from google import genai
from google.genai import types

# --- 1. 頁面外觀與佈局配置 ---
st.set_page_config(
    page_title="井震合一神經符號 AI 系統",
    page_icon="🌋",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .reportview-container .main .block-container {
        max-width: 92% !important;
        padding-top: 1.5rem !important;
    }
    .report-box {
        background-color: #262730;
        padding: 22px;
        border-radius: 10px;
        border-left: 5px solid #FF4B4B;
        margin: 15px 0 25px 0;
    }
    .status-box {
        background-color: #1E1E1E;
        padding: 12px;
        border-radius: 8px;
        border: 1px solid #4B4B4B;
        margin-bottom: 12px;
    }
    </style>
""", unsafe_allow_html=True)

# --- 2. Pydantic 結構化資料模型 (強制約束 AI 輸出) ---
class FaultCoordinate(BaseModel):
    x1: int = Field(description="斷層起始點 X 座標 (歸一化 0-1000)")
    y1: int = Field(description="斷層起始點 Y 座標 (歸一化 0-1000)")
    x2: int = Field(description="斷層結束點 X 座標 (歸一化 0-1000)")
    y2: int = Field(description="斷層結束點 Y 座標 (歸一化 0-1000)")
    fault_type: str = Field(description="斷層性質判斷，如：正斷層、逆斷層、平移斷層")
    estimated_dip_angle: float = Field(description="根據座標幾何估算之斷層面傾角(度數)")

class StructuralGeologyAnalysis(BaseModel):
    fault_prediction: FaultCoordinate
    fault_analysis_report: str = Field(description="第一節：斷層幾何特徵與運動性質詳細分析")
    stratigraphy_well_tie_report: str = Field(description="第二節：地層層位追蹤與井震空間對比解釋")
    geological_summary: str = Field(description="第三節：構造綜合總結 (2-3 句話)")

# --- 3. Stage 1: Meijering 脊線濾波與中斷點提取 ---
def advanced_skeleton_pipeline(pil_image, min_component_length=20):
    # 強制轉為 RGB 確保通道一致
    rgb_image = pil_image.convert("RGB")
    cv_img = cv2.cvtColor(np.array(rgb_image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape

    # 灰階反轉 (黑色反射同相軸轉為高亮脊線)
    inverted = cv2.bitwise_not(gray)
    denoised = cv2.medianBlur(inverted, 3)

    # 脊線濾波擷取同相軸中心
    ridges = meijering(denoised, sigmas=range(1, 4, 1), black_ridges=False)
    thresh_val = np.percentile(ridges[ridges > 0], 65) if np.any(ridges > 0) else 0.1
    binary_ridges = ridges > thresh_val

    # 骨幹化與碎噪點過濾
    skeleton = skeletonize(binary_ridges)
    labeled_img = label(skeleton)
    cleaned_skeleton = np.zeros_like(skeleton, dtype=np.uint8)

    for prop in regionprops(labeled_img):
        if prop.major_axis_length >= min_component_length:
            cleaned_skeleton[labeled_img == prop.label] = 255

    # 卷積端點偵測 (尋找錯斷點)
    kernel_endpoints = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]], dtype=np.uint8)
    filtered = cv2.filter2D(cleaned_skeleton // 255, -1, kernel_endpoints)
    endpoints_mask = (filtered == 11)

    # 繪製黑底描圖紙特徵圖
    tracing_img = Image.new("RGB", (width, height), (20, 20, 20))
    draw = ImageDraw.Draw(tracing_img)

    # 黃色骨幹線
    y_idxs, x_idxs = np.where(cleaned_skeleton > 0)
    for x, y in zip(x_idxs, y_idxs):
        draw.point((x, y), fill=(255, 215, 0))

    # 青色錯斷圓點
    ey_idxs, ex_idxs = np.where(endpoints_mask)
    for ex, ey in zip(ex_idxs, ey_idxs):
        if width * 0.03 < ex < width * 0.97 and height * 0.03 < ey < height * 0.97:
            draw.ellipse([ex-2, ey-2, ex+2, ey+2], fill=(0, 255, 255))

    return tracing_img

# --- 4. Stage 3: 符號幾何物理驗證層 ---
def symbolic_geometric_verification(coord: FaultCoordinate, img_w, img_h, min_dip=20.0, max_dip=85.0):
    x1 = int(np.clip((coord.x1 / 1000.0) * img_w, 0, img_w))
    y1 = int(np.clip((coord.y1 / 1000.0) * img_h, 0, img_h))
    x2 = int(np.clip((coord.x2 / 1000.0) * img_w, 0, img_w))
    y2 = int(np.clip((coord.y2 / 1000.0) * img_h, 0, img_h))

    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    calculated_dip = np.degrees(np.arctan2(dy, dx + 1e-6))

    is_valid = min_dip <= calculated_dip <= max_dip
    status_msg = f"幾何傾角驗證：<b>{calculated_dip:.1f}°</b> ➔ "
    if is_valid:
        status_msg += "✅ <b>通過符號層幾何約束檢查</b>"
    else:
        status_msg += f"⚠️ <b>警告：傾角偏離合理範圍 ({min_dip}°–{max_dip}°)</b>"

    return (x1, y1, x2, y2), calculated_dip, status_msg

# --- 5. 側邊欄控制項 ---
with st.sidebar:
    st.header("🔑 金鑰設定")
    saved_key = os.environ.get("GEMINI_API_KEY", "")
    api_key_input = st.text_input("輸入 Google API Key", value=saved_key, type="password", placeholder="AIzaSy...")
    st.divider()
    st.header("⚙️ 符號層地質約束參數")
    min_comp_len = st.slider("同相軸骨幹過濾長度 (px)", 5, 60, 20)
    dip_min = st.slider("斷層最小合理傾角 (°)", 10, 45, 20)
    dip_max = st.slider("斷層最大合理傾角 (°)", 60, 90, 80)

# --- 6. 主頁面介面 ---
st.title("🌋 井震合一神經符號地質大腦")
st.subheader("Pipeline: Meijering 脊線骨幹 ➔ LLM 空間感知 ➔ 符號幾何驗證 ➔ 綜合評判")
st.divider()

st.markdown("### 📥 井震多模態數據輸入區")
c1, c2, c3, c4 = st.columns(4)
with c1:
    seismic_files = st.file_uploader("1️⃣ 震測剖面圖片 (必填)", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
with c2:
    well_img_file = st.file_uploader("2️⃣ 鑽井柱狀圖 (選填)", type=["png", "jpg", "jpeg"])
with c3:
    well_location_notes = st.text_area("3️⃣ 井位相對空間位置", placeholder="例如： Well-A 位於剖面中央 CDP 1500 處...", height=120)
with c4:
    geology_notes = st.text_area("4️⃣ 區域地質背景", placeholder="例如：已知此區主要受強烈擠壓...", height=120)

st.divider()
submit_button = st.button("🚀 啟動神經符號 AI 推理流水線", use_container_width=True)

# --- 7. 推理流水線執行邏輯 ---
if submit_button:
    clean_api_key = api_key_input.strip() if api_key_input else ""
    if not seismic_files:
        st.error("❌ 錯誤：請至少上傳一張震測剖面圖片！")
    elif not clean_api_key:
        st.error("❌ 錯誤：請先在左側邊欄輸入有效的 Google API Key！")
    else:
        with st.spinner("⚡ 正在執行神經符號推理流水線..."):
            try:
                # 初始化 Gemini 客戶端
                client = genai.Client(api_key=clean_api_key)

                # 處理鑽井圖像輸入
                pil_well_img = None
                well_img_status = "未提供鑽井圖像"
                if well_img_file:
                    pil_well_img = Image.open(well_img_file).convert("RGB")
                    well_img_status = f"已附帶鑽井圖檔 ({well_img_file.name})"

                for file in seismic_files:
                    st.divider()
                    st.subheader(f"🖼️ 分析目標檔案：{file.name}")

                    # 讀取並淨化震測圖像
                    pil_seismic_img = Image.open(file).convert("RGB")
                    img_w, img_h = pil_seismic_img.size

                    # Stage 1: Meijering 脊線特徵提煉
                    tracing_img = advanced_skeleton_pipeline(pil_seismic_img, min_component_length=min_comp_len)

                    # Stage 2: 構建 Prompt 與結構化 Payload
                    prompt = f"""
                    你是一位資深的結構地質學與地球物理專家。
                    我為你提供了一張【骨幹拓撲特徵圖】（黃線為同相軸骨幹，青色圓點為同相軸中斷點 Off-sets）。
                    
                    請觀察青色中斷點在縱向上的排列趨勢，識別最顯著的主斷層帶，並回傳歸一化端點座標 (範圍 0-1000)。
                    
                    【鑽井約束狀態】: {well_img_status}
                    【井位空間位置】: {well_location_notes if well_location_notes else '未提供'}
                    【地質背景】: {geology_notes if geology_notes else '無'}
                    """

                    contents_payload = [tracing_img, pil_seismic_img]
                    if pil_well_img:
                        contents_payload.append(pil_well_img)
                    contents_payload.append(prompt)

                    config = types.GenerateContentConfig(
                        temperature=0.0,
                        response_mime_type="application/json",
                        response_schema=StructuralGeologyAnalysis
                    )

                    # Stage 2: 呼叫現役 gemini-2.5-flash，並帶有指數退避重試保護
                    response = None
                    max_retries = 3
                    target_model = "gemini-2.5-flash"

                    for attempt in range(max_retries):
                        try:
                            response = client.models.generate_content(
                                model=target_model,
                                contents=contents_payload,
                                config=config
                            )
                            if response and response.text:
                                break
                        except Exception as api_err:
                            err_msg = str(api_err)
                            if attempt < max_retries - 1:
                                wait_sec = (attempt + 1) * 3
                                st.warning(f"⚠️ API 回應忙碌中，正進行第 {attempt + 1} 次重試（等待 {wait_sec} 秒）...")
                                time.sleep(wait_sec)
                            else:
                                raise RuntimeError(f"Google API 連線異常 ({target_model}): {err_msg}")

                    if not response or not response.text:
                        raise RuntimeError("模型未回傳有效文字內容。")

                    # JSON 清洗與 Pydantic 結構化驗證
                    raw_json = response.text.strip()
                    if raw_json.startswith("```json"):
                        raw_json = re.sub(r"^```json\s*", "", raw_json)
                        raw_json = re.sub(r"\s*```$", "", raw_json)

                    structured_result = StructuralGeologyAnalysis.model_validate_json(raw_json)
                    pred = structured_result.fault_prediction

                    # Stage 3: 符號層幾何物理驗證
                    (rx1, ry1, rx2, ry2), calc_dip, status_msg = symbolic_geometric_verification(
                        pred, img_w, img_h, min_dip=dip_min, max_dip=dip_max
                    )

                    # Stage 4: 向量疊加斷層線繪製
                    annotated_seismic_img = pil_seismic_img.copy()
                    draw_final = ImageDraw.Draw(annotated_seismic_img)
                    draw_final.line([(rx1, ry1), (rx2, ry2)], fill="red", width=5)
                    draw_final.text((rx1, max(0, ry1 - 20)), f"{pred.fault_type} ({calc_dip:.1f}°)", fill="red")

                    # 畫面渲染輸出
                    col_img1, col_img2 = st.columns(2)
                    with col_img1:
                        st.image(tracing_img, caption="Stage 1: Meijering 脊線骨幹與中斷點 (黃:地層 | 青:斷點)", use_container_width=True)
                    with col_img2:
                        st.image(annotated_seismic_img, caption="Stage 4: 符號層驗證後之斷層疊加圖 (紅線)", use_container_width=True)

                    st.markdown(f'<div class="status-box">🛡️ <b>符號幾何驗證層狀態：</b><br>{status_msg}</div>', unsafe_allow_html=True)

                    report_md = f"""
                    ### 📝 構造綜合評判報告
                    
                    #### 1️⃣ 斷層幾何特徵分析
                    {structured_result.fault_analysis_report}
                    
                    #### 2️⃣ 地層層位與井震對比解釋
                    {structured_result.stratigraphy_well_tie_report}
                    
                    #### 3️⃣ 構造綜合總結
                    {structured_result.geological_summary}
                    """
                    st.markdown('<div class="report-box">', unsafe_allow_html=True)
                    st.markdown(report_md)
                    st.markdown('</div>', unsafe_allow_html=True)

            except Exception as e:
                st.error(f"❌ 流程執行錯誤: {str(e)}")
