import streamlit as st
import time
import os
import cv2
import numpy as np
from PIL import Image, ImageDraw
from pydantic import BaseModel, Field
from skimage.morphology import skeletonize
from skimage.measure import label, regionprops

# --- 網頁全寬與行動端優化 ---
st.set_page_config(
    page_title="井震合一神經符號 AI 系統 (Neuro-Symbolic 6.0)",
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
        padding: 25px;
        border-radius: 12px;
        border-left: 5px solid #FF4B4B;
        margin-top: 15px;
        margin-bottom: 30px;
    }
    .status-box {
        background-color: #1E1E1E;
        padding: 12px;
        border-radius: 8px;
        border: 1px solid #4B4B4B;
        margin-bottom: 15px;
    }
    </style>
""", unsafe_allow_html=True)

# --- 📐 Pydantic 定義：Gemini 強制結構化輸出 Schema ---
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

# --- 🎨 Stage 1：進階 CV 骨幹化與連通元件過濾 ---
def advanced_skeleton_pipeline(pil_image, min_component_length=30):
    cv_img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape

    # 1. 雙邊濾波保留邊緣並去除雜訊
    denoised = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)

    # 2. 自適應二值化提取強反射層
    binary = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, blockSize=21, C=4
    )

    # 3. 使用 skimage 進行標準拓撲骨幹化
    bool_skeleton = skeletonize(binary > 0)
    skeleton = (bool_skeleton * 255).astype(np.uint8)

    # 4. 連通元件分析：濾除短於閾值的碎點雲雜訊
    labeled_img = label(skeleton > 0)
    cleaned_skeleton = np.zeros_like(skeleton)

    for prop in regionprops(labeled_img):
        # 使用主要軸長度或像素面積過濾雜訊點
        if prop.major_axis_length >= min_component_length:
            cleaned_skeleton[labeled_img == prop.label] = 255

    # 5. 拓撲端點偵測 (中斷點標註)
    kernel_endpoints = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]], dtype=np.uint8)
    filtered = cv2.filter2D(cleaned_skeleton // 255, -1, kernel_endpoints)
    endpoints_mask = (filtered == 11)

    # 繪製乾淨的描圖紙特徵圖
    tracing_img = Image.new("RGB", (width, height), (20, 20, 20))
    draw = ImageDraw.Draw(tracing_img)

    # 黃色：連通且乾淨的同相軸骨幹
    y_idxs, x_idxs = np.where(cleaned_skeleton > 0)
    for x, y in zip(x_idxs, y_idxs):
        draw.point((x, y), fill=(255, 215, 0))

    # 青色圓圈：同相軸中斷點 (排除邊界雜訊)
    ey_idxs, ex_idxs = np.where(endpoints_mask)
    for ex, ey in zip(ex_idxs, ey_idxs):
        if width * 0.05 < ex < width * 0.95 and height * 0.05 < ey < height * 0.95:
            draw.ellipse([ex-3, ey-3, ex+3, ey+3], fill=(0, 255, 255))

    return tracing_img

# --- ⚖️ Stage 3：符號幾何驗證層 (Symbolic Verification Layer) ---
def symbolic_geometric_verification(coord: FaultCoordinate, img_w, img_h, min_dip=20.0, max_dip=85.0):
    """
    對 LLM 預測的幾何端點執行符號驗證：
    1. 計算實際幾何傾角
    2. 檢查是否落在地質物理合理邊界 (min_dip - max_dip)
    3. 執行坐標邊界截斷 (Clipping)
    """
    # 座標數值限制防呆
    x1 = int(np.clip((coord.x1 / 1000.0) * img_w, 0, img_w))
    y1 = int(np.clip((coord.y1 / 1000.0) * img_h, 0, img_h))
    x2 = int(np.clip((coord.x2 / 1000.0) * img_w, 0, img_w))
    y2 = int(np.clip((coord.y2 / 1000.0) * img_h, 0, img_h))

    # 計算幾何傾角
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    calculated_dip = np.degrees(np.arctan2(dy, dx + 1e-6))

    is_valid = min_dip <= calculated_dip <= max_dip
    status_msg = f"幾何傾角驗證：**{calculated_dip:.1f}°** ➔ "
    if is_valid:
        status_msg += "✅ **通過符號層幾何約束檢查**"
    else:
        status_msg += f"⚠️ **警告：傾角偏離合理範圍 ({min_dip}°–{max_dip}°)**"

    return (x1, y1, x2, y2), calculated_dip, status_msg

# --- 🧭 側邊欄 ---
with st.sidebar:
    st.header("🔑 金鑰驗證")
    api_key_input = st.text_input("輸入 Google API Key", type="password", placeholder="AIzaSy...")
    st.divider()
    st.header("⚙️ 符號層地質約束參數")
    min_comp_len = st.slider("同相軸骨幹過濾長度 (px)", 10, 80, 30)
    dip_min = st.slider("斷層最小合理傾角 (°)", 10, 45, 20)
    dip_max = st.slider("斷層最大合理傾角 (°)", 60, 90, 80)

# --- 主畫面標題 ---
st.title("🌋 井震合一神經符號 AI 地質大腦 (6.0 符號驗證版)")
st.subheader("完整 Pipeline：CV 骨幹拓撲 ➔ LLM 結構化感知 ➔ 符號幾何驗證 ➔ 綜合評判")
st.divider()

# --- 📥 輸入區塊 ---
st.markdown("### 📥 井震多模態數據與幾何空間約束輸入區")
in_col1, in_col2, in_col3, in_col4 = st.columns(4)

with in_col1:
    st.markdown("#### 1️⃣ 震測剖面圖片")
    seismic_files = st.file_uploader("上傳震測圖 (1-5張)", type=["png", "jpg", "jpeg"], accept_multiple_files=True, key="seismic_upload")

with in_col2:
    st.markdown("#### 2️⃣ 鑽井資料圖片")
    well_img_file = st.file_uploader("上傳 Well Log/柱狀圖", type=["png", "jpg", "jpeg"], key="well_img_upload")

with in_col3:
    st.markdown("#### 3️⃣ 井位相對空間位置")
    well_location_notes = st.text_area("描述井與震測圖相對位置", placeholder="例如： Well-A 位於剖面中央 CDP 1500 處...", height=125, key="well_location")

with in_col4:
    st.markdown("#### 4️⃣ 地質背景與備註")
    geology_notes = st.text_area("輸入區域地質背景", placeholder="例如：已知此區主要受強烈擠壓...", height=125, key="geo_notes")

st.divider()

submit_button = st.button("🚀 啟動神經符號 AI 推理流水線 (Neuro-Symbolic Pipeline)", use_container_width=True)

# --- 🧠 後端推理邏輯 ---
if submit_button:
    if not seismic_files:
        st.error("❌ 錯誤：請至少上傳一張震測剖面圖片！")
    elif not api_key_input:
        st.error("❌ 錯誤：請先在左側邊欄輸入你的 Google API Key！")
    else:
        with st.spinner("⚡ 正在執行 Stage 1-4 神經符號完整流水線..."):
            try:
                from google import genai
                from google.genai import types

                client = genai.Client(api_key=api_key_input)

                pil_well_img = None
                well_img_status = "未提供鑽井圖像"
                if well_img_file:
                    pil_well_img = Image.open(well_img_file)
                    well_img_status = f"已附帶鑽井圖檔 ({well_img_file.name})"

                st.success("🎉 神經符號計算流水線啟動完畢！")
                st.balloons()

                for file in seismic_files:
                    st.divider()
                    st.subheader(f"🖼️ 分析目標檔案：{file.name}")

                    pil_seismic_img = Image.open(file)
                    img_w, img_h = pil_seismic_img.size

                    # --- Stage 1: 進階骨幹提取 ---
                    tracing_img = advanced_skeleton_pipeline(pil_seismic_img, min_component_length=min_comp_len)

                    # --- Stage 2: Gemini 結構化輸出 ---
                    prompt = f"""
                    你是一位資深的結構地質學與地球物理專家。
                    我為你提供了一張【骨幹拓撲特徵圖】（黃線為去除雜訊後的同相軸骨幹，青色圓點為同相軸中斷點 Off-sets）。
                    
                    請觀察青色中斷點在縱向上的排列趨勢，識別最顯著的主斷層帶，並回傳歸一化端點座標 (範圍 0-1000)。
                    
                    【鑽井約束狀態】: {well_img_status}
                    【井位與震測空間相對位置】: {well_location_notes if well_location_notes else '未提供'}
                    【區域地質背景備註】: {geology_notes if geology_notes else '無'}
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

                    response = client.models.generate_content(
                        model='gemini-2.5-flash',
                        contents=contents_payload,
                        config=config
                    )

                    # 解析結構化物件
                    structured_result = StructuralGeologyAnalysis.model_validate_json(response.text)
                    pred = structured_result.fault_prediction

                    # --- Stage 3: 符號幾何驗證層 ---
                    (rx1, ry1, rx2, ry2), calc_dip, status_msg = symbolic_geometric_verification(
                        pred, img_w, img_h, min_dip=dip_min, max_dip=dip_max
                    )

                    # --- Stage 4: 向量疊加繪圖 ---
                    annotated_seismic_img = pil_seismic_img.copy()
                    draw_final = ImageDraw.Draw(annotated_seismic_img)
                    draw_final.line([(rx1, ry1), (rx2, ry2)], fill="red", width=5)
                    draw_final.text((rx1, max(0, ry1 - 20)), f"{pred.fault_type} ({calc_dip:.1f}°)", fill="red")

                    # 畫面排版渲染
                    col_img1, col_img2 = st.columns(2)
                    with col_img1:
                        st.image(tracing_img, caption=f"Stage 1: 拓撲骨幹與中斷點 (黃:骨幹 | 青:斷點)", use_container_width=True)
                    with col_img2:
                        st.image(annotated_seismic_img, caption=f"Stage 4: 符號層驗證後之斷層疊加圖 (紅線)", use_container_width=True)

                    # 顯示符號驗證狀態
                    st.markdown(f'<div class="status-box">🛡️ <b>符號幾何驗證層 (Symbolic Layer) 狀態：</b><br>{status_msg}</div>', unsafe_allow_html=True)

                    # 結構化報告渲染
                    report_html = f"""
                    ### 📝 構造綜合評判報告
                    
                    #### 1️⃣ 斷層幾何特徵分析 (Fault Node)
                    {structured_result.fault_analysis_report}
                    
                    #### 2️⃣ 地層層位與井震對比解釋 (Stratigraphy Node)
                    {structured_result.stratigraphy_well_tie_report}
                    
                    #### 3️⃣ 構造綜合總結 (Summary Node)
                    {structured_result.geological_summary}
                    """
                    st.markdown('<div class="report-box">', unsafe_allow_html=True)
                    st.markdown(report_html)
                    st.markdown('</div>', unsafe_allow_html=True)

            except Exception as e:
                st.error(f"❌ 流程執行錯誤: {str(e)}")
