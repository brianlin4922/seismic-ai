import streamlit as st
import time
import os
import cv2
import json
import re
import numpy as np
from PIL import Image, ImageDraw

# --- 網頁全寬與 iPad Chrome 相容性優化設定 ---
st.set_page_config(
    page_title="井震合一 AI 結構地質大腦",
    page_icon="🌋",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .reportview-container .main .block-container {
        max-width: 90% !important;
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
    </style>
""", unsafe_allow_html=True)

# --- 🎨 重現描圖紙工作流程：1. 描粗黑線(骨幹化)  2. 標出斷點 ---
def tracing_paper_workflow(pil_image):
    """
    模擬地質學家描圖紙：
    1. 自適應二值化，擷取強反射同相軸
    2. 提取骨幹 (Skeletonization)
    3. 標註同相軸不連續的端點 (End-Points)
    """
    cv_img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape

    # 高斯模糊降噪
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 1. 描圖紙效應：自適應二值化，將強反射黑線凸顯出來
    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 15, 3
    )

    # 2. 骨幹化 (Skeletonization) - 將粗黑線縮減為 1 像素細線，方便看斷點
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    skeleton = np.zeros(binary.shape, np.uint8)
    eroded = binary.copy()
    
    for _ in range(10): # 進行 10 次細化提煉同相軸骨幹
        temp = cv2.morphologyEx(eroded, cv2.MORPH_OPEN, kernel)
        temp = cv2.subtract(eroded, temp)
        skeleton = cv2.bitwise_or(skeleton, temp)
        eroded = cv2.erode(eroded, kernel)

    # 3. 找出骨幹線條的端點/中斷點 (End-Points)
    # 建立 3x3 算子計算鄰居數，只有 1 個鄰居的點代表「線條在此中斷」
    kernel_endpoints = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]], dtype=np.uint8)
    filtered = cv2.filter2D(skeleton // 255, -1, kernel_endpoints)
    endpoints_mask = (filtered == 11) # 10(中心) + 1(只有一個鄰居)

    # 繪製「描圖紙效果圖」：黑色底、黃色骨幹同相軸、青色端點
    tracing_paper_img = Image.new("RGB", (width, height), (30, 30, 30))
    draw = ImageDraw.Draw(tracing_paper_img)

    # 畫骨幹 (黃線)
    y_indices, x_indices = np.where(skeleton > 0)
    for x, y in zip(x_indices, y_indices):
        if x % 2 == 0 and y % 2 == 0: # 抽樣繪製避免過密
            draw.point((x, y), fill=(255, 215, 0))

    # 畫中斷端點 (青色圓圈)
    ey_indices, ex_indices = np.where(endpoints_mask)
    for ex, ey in zip(ex_indices, ey_indices):
        if width * 0.1 < ex < width * 0.9: # 排除圖邊緣
            draw.ellipse([ex-3, ey-3, ex+3, ey+3], fill=(0, 255, 255))

    return tracing_paper_img

# --- 🧭 側邊欄 ---
with st.sidebar:
    st.header("🔑 金鑰驗證")
    api_key_input = st.text_input("輸入 Google API Key", type="password", placeholder="AIzaSy...")
    if api_key_input:
        os.environ["GEMINI_API_KEY"] = api_key_input
        st.success("✅ Key 已載入")
    else:
        st.warning("⚠️ 請輸入 Key 以啟用 AI 大腦")

# --- 主畫面標題 ---
st.title("🌋 井震合一 AI 結構地質大腦 (5.0 描圖紙與座標回傳版)")
st.subheader("碩士級功能：模擬地質學家描圖紙骨幹化與 Gemini 2.5 空間座標繪圖")
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

submit_button = st.button("🚀 啟動「描圖紙骨幹追蹤」與 Gemini 空間座標繪圖", use_container_width=True)

# --- 🧠 後端真實推理邏輯 ---
if submit_button:
    if not seismic_files:
        st.error("❌ 錯誤：請至少上傳一張震測剖面圖片！")
    elif not api_key_input:
        st.error("❌ 錯誤：請先在左側邊欄輸入你的 Google API Key！")
    else:
        with st.spinner("⚡ 正在製作描圖紙骨幹、標註同相軸斷點，並請 Gemini 計算斷層線座標..."):
            try:
                from google import genai
                from google.genai import types
                
                client = genai.Client(api_key=api_key_input)
                
                pil_well_img = None
                well_img_status = "未提供鑽井圖像"
                if well_img_file:
                    pil_well_img = Image.open(well_img_file)
                    well_img_status = f"已附帶鑽井柱狀圖/測井圖 (`{well_img_file.name}`)"

                st.success(f"🎉 描圖紙骨幹提取完成！開始進行 Gemini 空間座標繪圖...")
                st.balloons()
                
                st.markdown("## 🎨 描圖紙骨幹與 Gemini 精準座標繪圖成果")
                
                for file in seismic_files:
                    st.divider()
                    st.subheader(f"🖼️ 分析目標檔案：{file.name}")
                    
                    pil_seismic_img = Image.open(file)
                    img_w, img_h = pil_seismic_img.size
                    
                    # 🔥 執行描圖紙骨幹與斷點提取
                    tracing_img = tracing_paper_workflow(pil_seismic_img)
                    
                    prompt = f"""
                    你是一位資深的結構地質學與地球物理專家。
                    我為你提供了一張【描圖紙骨幹圖】（黃線為同相軸骨幹，青色點為同相軸不連續/錯斷點）。
                    
                    請扮演地質學家，觀察這些青色錯斷點在空間中的排列趨勢，找出最顯著的一條主要斷層面，並回傳該斷層線在圖片上的端點歸一化座標（範圍 0 到 1000）。
                    
                    【強制格式要求】：
                    請務必在回答的最開頭，用 JSON 格式輸出斷層線的兩個端點座標 (x1, y1, x2, y2)，格式如下：
                    ```json
                    {{"fault_line": [x1, y1, x2, y2]}}
                    ```
                    (例如：{{"fault_line": [350, 200, 650, 800]}} 代表從 x=35% y=20% 連線到 x=65% y=80%)
                    
                    接著請輸出地質報告內容：
                    【🎯 井位與震測剖面的相對空間位置描述】:
                    {well_location_notes if well_location_notes else '未提供相對位置'}
                    
                    【區域地質背景備註】:
                    {geology_notes if geology_notes else '無'}
                    
                    ### 📝 Gemini 井震綜合構造判斷報告
                    #### 1️⃣ 斷層幾何特徵分析 (Fault Node)
                    - 分析圖片中同相軸不連續點串聯成的斷層帶，判斷斷層性質與傾角。
                    #### 2️⃣ 地層層位與井震對比解釋 (Stratigraphy Node)
                    - 結合井位空間位置描述進行層位對比。
                    #### 3️⃣ 構造綜合總結 (Summary Node)
                    - 用 2-3 句話簡短總結。
                    """
                    
                    config = types.GenerateContentConfig(temperature=0.0)
                    
                    contents_payload = [tracing_img, pil_seismic_img]
                    if pil_well_img:
                        contents_payload.append(pil_well_img)
                    contents_payload.append(prompt)
                    
                    with st.spinner(f"Gemini 正在分析骨幹斷點並計算斷層座標 `{file.name}`..."):
                        response = client.models.generate_content(
                            model='gemini-2.5-flash',
                            contents=contents_payload,
                            config=config
                        )
                    
                    response_text = response.text
                    
                    # 抓取 Gemini 回傳的 JSON 座標並在原始圖上用 Python 畫紅線
                    annotated_seismic_img = pil_seismic_img.copy()
                    draw_final = ImageDraw.Draw(annotated_seismic_img)
                    
                    json_match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
                    if json_match:
                        try:
                            coord_data = json.loads(json_match.group(1))
                            pts = coord_data.get("fault_line", [])
                            if len(pts) == 4:
                                real_x1 = int((pts[0] / 1000.0) * img_w)
                                real_y1 = int((pts[1] / 1000.0) * img_h)
                                real_x2 = int((pts[2] / 1000.0) * img_w)
                                real_y2 = int((pts[3] / 1000.0) * img_h)
                                
                                # 在原始圖上記製極度精確的紅線
                                draw_final.line([(real_x1, real_y1), (real_x2, real_y2)], fill="red", width=5)
                                draw_final.text((real_x1, max(0, real_y1 - 15)), "Fault Zone (Gemini Grounded)", fill="red")
                        except Exception as parse_e:
                            pass
                    
                    col_img1, col_img2 = st.columns(2)
                    with col_img1:
                        st.image(tracing_img, caption=f"1. 描圖紙骨幹與端點圖 (黃:同相軸 | 青:中斷點) - {file.name}", use_container_width=True)
                    with col_img2:
                        st.image(annotated_seismic_img, caption=f"2. Gemini 空間感知精準畫線 (紅線) - {file.name}", use_container_width=True)
                    
                    # 濾除 JSON 文字，只展示優美的 Markdown 報告
                    clean_report = re.sub(r'```json\s*\{.*?\}\s*```', '', response_text, flags=re.DOTALL)
                    
                    st.markdown('<div class="report-box">', unsafe_allow_html=True)
                    st.markdown(clean_report)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
            except Exception as e:
                st.error(f"❌ API 錯誤: {str(e)}")
