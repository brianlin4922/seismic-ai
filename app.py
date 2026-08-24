import streamlit as st
import time
import os
import re
import math
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, Field
from skimage.filters import meijering
from skimage.morphology import skeletonize
from skimage.measure import label, regionprops
from google import genai
from google.genai import types

# --- 1. 頁面外觀與佈局配置 ---
st.set_page_config(
    page_title="井震合一神經符號 AI 系統 (多斷層約束版)",
    page_icon="🌋",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .reportview-container .main .block-container {
        max-width: 94% !important;
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
        padding: 14px;
        border-radius: 8px;
        border: 1px solid #4B4B4B;
        margin-bottom: 12px;
    }
    .legend-box {
        display: flex;
        gap: 20px;
        background: #111;
        padding: 10px 15px;
        border-radius: 6px;
        margin-bottom: 10px;
        font-size: 0.9rem;
    }
    </style>
""", unsafe_allow_html=True)

# --- 2. Pydantic 結構化資料模型 (支援多斷層分級預測) ---
class FaultCoordinate(BaseModel):
    name: str = Field(description="斷層編號標籤 (例如: F1 主控斷層, F2 次級斷層)")
    hierarchy: str = Field(description="構造層級: 'Master' (主控斷層) 或 'Secondary' (次級伴生斷層)")
    x1: int = Field(description="斷層起始點 X 座標 (歸一化 0-1000)")
    y1: int = Field(description="斷層起始點 Y 座標 (歸一化 0-1000)")
    x2: int = Field(description="斷層結束點 X 座標 (歸一化 0-1000)")
    y2: int = Field(description="斷層結束點 Y 座標 (歸一化 0-1000)")
    fault_type: str = Field(description="斷層運動性質 (正斷層 / 逆斷層 / 平移斷層)")
    estimated_dip_angle: float = Field(description="幾何估算傾角 (0-90度)")
    confidence: float = Field(description="地質置信度 (0.0-1.0，依據端點共線性與錯斷落差)")

class StructuralGeologyAnalysis(BaseModel):
    fault_predictions: list[FaultCoordinate] = Field(
        description="所有通過地質檢驗的顯著斷層清單 (最多 5 條，按重要性排序，排除細微雜訊)"
    )
    fault_analysis_report: str = Field(description="第一節：斷層構造系統幾何特徵與運動性質詳細分析")
    stratigraphy_well_tie_report: str = Field(description="第二節：地層層位追蹤與井震空間對比解釋")
    geological_summary: str = Field(description="第三節：構造綜合總結 (2-3 句話)")

# --- 3. Stage 1: Meijering 脊線濾波與中斷點提取 ---
def advanced_skeleton_pipeline(pil_image, min_component_length=20):
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

    # 拓撲骨幹化
    skeleton = skeletonize(binary_ridges)
    labeled_img = label(skeleton)
    cleaned_skeleton = np.zeros_like(skeleton, dtype=np.uint8)

    for prop in regionprops(labeled_img):
        if prop.major_axis_length >= min_component_length:
            cleaned_skeleton[labeled_img == prop.label] = 255

    # 卷積端點偵測 (尋找同相軸中斷點)
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

# --- 4. Stage 3: 多斷層符號幾何物理驗證層 ---
def symbolic_multi_fault_verification(faults, img_w, img_h, min_dip=20.0, max_dip=85.0, min_len_px=30):
    verified_faults = []
    status_logs = []

    for f in faults:
        x1 = int(np.clip((f.x1 / 1000.0) * img_w, 0, img_w))
        y1 = int(np.clip((f.y1 / 1000.0) * img_h, 0, img_h))
        x2 = int(np.clip((f.x2 / 1000.0) * img_w, 0, img_w))
        y2 = int(np.clip((f.y2 / 1000.0) * img_h, 0, img_h))

        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        length_px = math.hypot(dx, dy)
        calc_dip = np.degrees(np.arctan2(dy, dx + 1e-6))

        # 幾何檢查：傾角範圍與長度門檻
        dip_ok = (min_dip <= calc_dip <= max_dip)
        len_ok = (length_px >= min_len_px)

        if dip_ok and len_ok:
            verified_faults.append({
                "obj": f,
                "coords": (x1, y1, x2, y2),
                "dip": calc_dip,
                "length": length_px,
                "is_master": (f.hierarchy.lower() == "master" or "主" in f.hierarchy)
            })
            status_logs.append(
                f"• <b>{f.name} ({f.fault_type})</b>：傾角 {calc_dip:.1f}° | 長度 {length_px:.0f}px | 置信度 {f.confidence:.2f} ➔ <span style='color:#4CAF50;'>✅ 通過幾何約束</span>"
            )
        else:
            reason = []
            if not dip_ok:
                reason.append(f"傾角 {calc_dip:.1f}° 偏離 ({min_dip}°–{max_dip}°)")
            if not len_ok:
                reason.append(f"長度 {length_px:.0f}px 過短 (<{min_len_px}px)")
            status_logs.append(
                f"• <b>{f.name}</b>：<span style='color:#FF5252;'>⚠️ 遭符號層剔除 ({'、'.join(reason)})</span>"
            )

    return verified_faults, status_logs

# --- 5. 側邊欄控制項 ---
with st.sidebar:
    st.header("🔑 金鑰設定")
    saved_key = os.environ.get("GEMINI_API_KEY", "")
    api_key_input = st.text_input("輸入 Google API Key", value=saved_key, type="password", placeholder="AIzaSy...")
    
    st.divider()
    st.header("⚙️ 符號層地質約束參數")
    min_comp_len = st.slider("同相軸骨幹過濾長度 (px)", 5, 60, 20, help="濾除過短的同相軸碎屑")
    
    st.markdown("**斷層合理傾角範圍 (°)**")
    dip_min = st.slider("最小合理傾角", 10, 45, 20)
    dip_max = st.slider("最大合理傾角", 55, 90, 80)
    
    st.divider()
    st.header("🎛️ 多斷層平衡控制")
    max_faults_limit = st.slider("最大標註斷層數量上限", 1, 5, 3, help="控制畫面最多標註幾條斷層，防止線條過密")
    min_fault_conf = st.slider("斷層最低置信度門檻", 0.4, 0.9, 0.6, step=0.05, help="過濾掉不明顯的微小裂隙")

# --- 6. 主頁面介面 ---
st.title("🌋 井震合一神經符號 AI 地質大腦")
st.subheader("Pipeline: Meijering 脊線骨幹 ➔ 多斷層空間感知 ➔ 符號幾何驗證 ➔ 綜合評判")
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
    geology_notes = st.text_area("4️⃣ 區域地質背景", placeholder="例如：已知此區主要受強烈張裂應力，發育階梯狀正斷層...", height=120)

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
                client = genai.Client(api_key=clean_api_key)

                pil_well_img = Image.open(well_img_file).convert("RGB") if well_img_file else None
                well_img_status = f"已附帶鑽井圖檔 ({well_img_file.name})" if well_img_file else "未提供鑽井圖像"

                for file in seismic_files:
                    st.divider()
                    st.subheader(f"🖼️ 分析目標檔案：{file.name}")

                    pil_seismic_img = Image.open(file).convert("RGB")
                    img_w, img_h = pil_seismic_img.size

                    # Stage 1: 提取骨幹與斷點
                    tracing_img = advanced_skeleton_pipeline(pil_seismic_img, min_component_length=min_comp_len)

                    # Stage 2: 構建 Prompt (加入多斷層多尺度平衡規則)
                    prompt = f"""
                    你是一位資深的結構地質學與地球物理專家。
                    我為你提供了一張【骨幹拓撲特徵圖】（黃線為同相軸骨幹，青色圓點為同相軸中斷點 Off-sets）與原始震測圖。

                    請精確識別剖面中具有地質意義的斷層系統（包括主控斷層與顯著次級伴生斷層），並回傳歸一化座標 (0-1000)。
                    
                    【斷層判斷與過濾硬約束】：
                    1. 數量上限：最多回傳 {max_faults_limit} 條最顯著且置信度高的斷層。
                    2. 貫穿性門檻：每條斷層必須貫穿至少 2 個以上同相軸，並沿著 3 個以上青色中斷點的共線軌跡延伸。
                    3. 排除雜訊：禁止將單一地層內部微小抖動或局部噪點標為斷層。
                    4. 構造分級：落差最大、延伸最長的主斷層標註為 hierarchy='Master'，其餘伴生斷層標註為 hierarchy='Secondary'。

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

                    # 穩定重試機制呼叫 gemini-2.5-flash
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
                                st.warning(f"⚠️ API 伺服器忙碌，正在進行第 {attempt + 1} 次重試 (等待 {wait_sec} 秒)...")
                                time.sleep(wait_sec)
                            else:
                                raise RuntimeError(f"Google API 連線異常 ({target_model}): {err_msg}")

                    if not response or not response.text:
                        raise RuntimeError("模型未回傳有效內容。")

                    # JSON 清洗與驗證
                    raw_json = response.text.strip()
                    if raw_json.startswith("```json"):
                        raw_json = re.sub(r"^```json\s*", "", raw_json)
                        raw_json = re.sub(r"\s*```$", "", raw_json)

                    structured_result = StructuralGeologyAnalysis.model_validate_json(raw_json)
                    
                    # 篩選符合置信度門檻的斷層
                    candidate_faults = [
                        f for f in structured_result.fault_predictions 
                        if f.confidence >= min_fault_conf
                    ][:max_faults_limit]

                    # Stage 3: 符號層幾何約束驗證
                    min_len_limit = int(min(img_w, img_h) * 0.08)  # 最小長度至少佔圖面 8%
                    verified_faults, status_logs = symbolic_multi_fault_verification(
                        candidate_faults, img_w, img_h, 
                        min_dip=dip_min, max_dip=dip_max, 
                        min_len_px=min_len_limit
                    )

                    # Stage 4: 向量疊加繪製多斷層
                    annotated_seismic_img = pil_seismic_img.copy()
                    draw_final = ImageDraw.Draw(annotated_seismic_img)

                    for item in verified_faults:
                        f_obj = item["obj"]
                        x1, y1, x2, y2 = item["coords"]
                        dip = item["dip"]
                        is_master = item["is_master"]

                        # 主斷層：紅色粗線 (5px)；次級斷層：橙色中線 (3px)
                        line_color = (255, 50, 50) if is_master else (255, 140, 0)
                        line_width = 5 if is_master else 3

                        draw_final.line([(x1, y1), (x2, y2)], fill=line_color, width=line_width)
                        
                        # 標註標籤
                        label_text = f"{f_obj.name}: {f_obj.fault_type} ({dip:.1f}°)"
                        draw_final.text((x1, max(0, y1 - 18)), label_text, fill=line_color)

                    # 畫面渲染輸出
                    st.markdown("""
                        <div class="legend-box">
                            <span><b style="color: #FF3232;">━━━</b> 一級主控斷層 (Master Fault, 5px)</span>
                            <span><b style="color: #FF8C00;">━━━</b> 二級伴生斷層 (Secondary Fault, 3px)</span>
                            <span><b style="color: #00FFFF;">●</b> 同相軸錯斷端點 (Off-sets)</span>
                        </div>
                    """, unsafe_allow_html=True)

                    col_img1, col_img2 = st.columns(2)
                    with col_img1:
                        st.image(tracing_img, caption="Stage 1: Meijering 脊線骨幹與中斷點 (黃:地層 | 青:斷點)", use_container_width=True)
                    with col_img2:
                        st.image(annotated_seismic_img, caption=f"Stage 4: 符號幾何驗證後多斷層疊加圖 (共標註 {len(verified_faults)} 條)", use_container_width=True)

                    st.markdown('<div class="status-box">🛡️ <b>符號幾何驗證層 (Symbolic Verification Layer) 檢驗報告：</b><br>' + '<br>'.join(status_logs) + '</div>', unsafe_allow_html=True)

                    report_md = f"""
                    ### 📝 構造綜合評判報告
                    
                    #### 1️⃣ 斷層系統特徵分析 (Fault System Analysis)
                    {structured_result.fault_analysis_report}
                    
                    #### 2️⃣ 地層層位與井震對比解釋 (Stratigraphy & Well-Tie)
                    {structured_result.stratigraphy_well_tie_report}
                    
                    #### 3️⃣ 構造綜合總結 (Structural Synthesis)
                    {structured_result.geological_summary}
                    """
                    st.markdown('<div class="report-box">', unsafe_allow_html=True)
                    st.markdown(report_md)
                    st.markdown('</div>', unsafe_allow_html=True)

            except Exception as e:
                st.error(f"❌ 流程執行錯誤: {str(e)}")
