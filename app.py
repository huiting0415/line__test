# -*- coding: utf-8 -*-
"""
會計師事務所暨商務中心 - LINE 客戶數據 CDP 分析系統
=====================================================
功能：
1. 上傳 LINE 官方帳號後台匯出的對話紀錄（Excel/CSV）
2. 自動去敏感化（遮蔽身分證字號、統編、手機號碼）
3. 用 OpenAI GPT 模型批次分析對話，產出：
   - 業務分類圓餅圖
   - 高價值商機 / 流失風險清單
   - 本月 FAQ 摘要

使用方式：
    pip install -r requirements.txt
    streamlit run app.py

注意：本程式會將對話內容（去敏感化後）傳送至 OpenAI API 進行分析，
      正式上線前請務必確認客戶資料使用是否符合機構的個資保護政策與客戶告知義務。
"""

import re
import json
import time
from io import BytesIO

import pandas as pd
import streamlit as st
import plotly.express as px
import google.generativeai as genai


# =========================================================
# 基本頁面設定
# =========================================================
st.set_page_config(
    page_title="LINE 客戶數據 CDP 分析系統",
    page_icon="📊",
    layout="wide",
)

st.title("📊 會計師事務所暨商務中心 - LINE 客戶數據 CDP 分析系統")
st.caption("上傳 LINE 對話紀錄，自動去敏感化並產出業務分類、商機/風險清單、FAQ 摘要")


# =========================================================
# 側邊欄：API Key 輸入（不寫死在程式碼中，避免外洩）
# =========================================================
with st.sidebar:
    st.header("⚙️ 設定")

    # 若部署在 Streamlit Cloud 且已設定 secrets，會自動帶入預設 Key
    # 本機執行時若無 secrets 設定，會安全地略過不報錯
    try:
        default_api_key = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        default_api_key = ""

    api_key = st.text_input(
        "Google Gemini API Key",
        type="password",
        value=default_api_key,
        help="從 https://aistudio.google.com/apikey 免費取得（不需信用卡），僅在本次執行期間使用，不會被儲存。",
    )

    model_name = st.text_input(
        "使用模型",
        value="gemini-flash-latest",
        help="使用「-latest」別名可自動指向Google目前最新的Flash模型，避免未來模型下架後程式失效",
    )

    batch_size = st.slider(
        "每批次分析筆數",
        min_value=5,
        max_value=50,
        value=20,
        help="批次處理可大幅減少 API 呼叫次數、加快分析速度並降低成本",
    )

    st.divider()
    st.markdown(
        "⚠️ **資料安全提醒**\n\n"
        "上傳前請確認資料已取得授權可用於分析。\n"
        "本系統會自動遮蔽身分證字號、統編、手機號碼後才送出分析，"
        "但公司名稱、人名等其他個資仍會被傳送至 Google，請自行評估風險。\n\n"
        "**免費版額外提醒**：Gemini API 免費額度的輸入資料可能被 Google 用於改善模型，"
        "正式處理真實客戶資料前，建議改用付費版或評估其他方案。"
    )


# =========================================================
# Step 1：檔案上傳
# =========================================================
st.subheader("① 上傳對話紀錄")

uploaded_file = st.file_uploader(
    "支援 Excel (.xlsx) 或 CSV 檔案，需包含「客戶暱稱」「對話內容」等欄位",
    type=["xlsx", "csv"],
)

# 提供範例資料下載，方便測試
with st.expander("沒有檔案？點此下載模擬範例資料測試系統"):
    st.markdown(
        "範例資料為**模擬生成**，不含任何真實客戶資訊，可安全用於測試本系統。"
    )
    col_sample1, col_sample2 = st.columns(2)
    with col_sample1:
        st.markdown("**基本版**（僅客戶對話，測試業務分類/商機/風險）")
        try:
            with open("sample_line_conversations.csv", "rb") as f:
                st.download_button(
                    "下載基本範例 CSV",
                    data=f,
                    file_name="sample_line_conversations.csv",
                    mime="text/csv",
                )
        except FileNotFoundError:
            st.info("請將 sample_line_conversations.csv 放在與 app.py 相同的資料夾")
    with col_sample2:
        st.markdown("**進階版**（含同事回覆與時間戳記，可測試同事表現分析）")
        try:
            with open("sample_line_conversations_v2.csv", "rb") as f:
                st.download_button(
                    "下載進階範例 CSV",
                    data=f,
                    file_name="sample_line_conversations_v2.csv",
                    mime="text/csv",
                )
        except FileNotFoundError:
            st.info("請將 sample_line_conversations_v2.csv 放在與 app.py 相同的資料夾")


# =========================================================
# 去敏感化函式（正則表達式）
# =========================================================
def mask_sensitive_info(text: str) -> str:
    """遮蔽台灣身分證字號、統一編號、手機號碼"""
    if not isinstance(text, str):
        return text

    # 台灣身分證字號：1碼英文 + 9碼數字
    text = re.sub(r"[A-Za-z][12]\d{8}", "[身分證字號已遮蔽]", text)

    # 統一編號：8碼數字（獨立出現，避免誤判電話等其他數字）
    text = re.sub(r"(?<!\d)\d{8}(?!\d)", "[統編已遮蔽]", text)

    # 手機號碼：09開頭10碼數字
    text = re.sub(r"09\d{2}[\s\-]?\d{3}[\s\-]?\d{3}", "[手機號碼已遮蔽]", text)

    return text


# =========================================================
# 讀取檔案並自動偵測欄位
# =========================================================
def load_dataframe(file) -> pd.DataFrame:
    if file.name.endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)
    return df


def guess_column(df: pd.DataFrame, keywords: list) -> str:
    """依關鍵字猜測欄位名稱，找不到則回傳第一個欄位"""
    for col in df.columns:
        for kw in keywords:
            if kw in str(col):
                return col
    return df.columns[0]


# =========================================================
# Step 2：呼叫 OpenAI 進行批次分析
# =========================================================
ANALYSIS_SYSTEM_PROMPT = """你是一家「會計師事務所暨商務中心」的資深客戶數據分析師。
機構同時經營兩類業務：
1. 會計師事務所業務：記帳、工商登記、稅務申報（營業稅/營所稅/綜所稅）、稅務諮詢
2. 商務中心業務：借址登記、虛擬辦公室、信件代收、會議室租借

請針對輸入的每一則客戶對話進行分類分析，並「只回傳 JSON 陣列」，不要有任何其他文字、不要用 markdown code block 包住。

每筆對話請輸出以下欄位：
- index: 對應輸入的序號（整數）
- category: 業務分類，必須是以下其中一種："工商登記", "稅務申報", "商務中心-借址虛擬辦公室", "商務中心-信件會議室", "其他/閒聊"
- cross_sell_opportunity: 布林值，若對話中客戶透露出「原本是A類客戶但詢問/透露對B類服務有興趣」（例如借址客戶問記帳、記帳客戶問借址/會議室），設為 true，否則 false
- cross_sell_note: 若 cross_sell_opportunity 為 true，簡短說明可能的商機（20字以內），否則為空字串
- churn_risk: 布林值，若對話中出現抱怨、不滿、考慮解約/換服務商等負面情緒，設為 true，否則 false
- churn_note: 若 churn_risk 為 true，簡短說明風險點（20字以內），否則為空字串
- question_summary: 用一句話（15字以內）摘要這則對話在問什麼

範例輸出格式：
[{"index": 0, "category": "稅務申報", "cross_sell_opportunity": false, "cross_sell_note": "", "churn_risk": false, "churn_note": "", "question_summary": "詢問營業稅申報期限"}]
"""


def analyze_batch(model: genai.GenerativeModel, batch_df: pd.DataFrame, text_col: str) -> list:
    """將一批對話送進 Gemini，要求回傳結構化 JSON"""
    items = []
    for i, row in enumerate(batch_df.itertuples()):
        content = getattr(row, text_col)
        items.append(f"{i}: {content}")

    user_prompt = "以下是本批次的客戶對話，請依格式分析每一筆：\n\n" + "\n".join(items)

    response = model.generate_content(
        [ANALYSIS_SYSTEM_PROMPT, user_prompt],
        generation_config={
            "temperature": 0,
            "response_mime_type": "application/json",
        },
    )

    raw_text = response.text.strip()

    # 清理可能殘留的 markdown code block 符號
    raw_text = re.sub(r"^```(json)?", "", raw_text.strip())
    raw_text = re.sub(r"```$", "", raw_text.strip())

    try:
        parsed = json.loads(raw_text)
        # 有些模型可能包一層 {"results": [...]}，做個保險處理
        if isinstance(parsed, dict):
            for v in parsed.values():
                if isinstance(v, list):
                    parsed = v
                    break
        return parsed
    except json.JSONDecodeError:
        st.warning("有一批資料解析失敗，已略過該批次，建議稍後重試或減少批次筆數")
        return []


# =========================================================
# 同事回覆表現分析：配對「客戶問題→同事回覆」並計算回覆時間
# =========================================================
def compute_response_pairs(df: pd.DataFrame, name_col: str, staff_col: str, time_col: str, text_col: str) -> pd.DataFrame:
    """
    採用簡化假設：資料以時間順序排列，且每一則同事回覆的「上一列」即為對應的客戶提問。
    若資料並非嚴格一問一答排列，結果僅供參考。
    """
    df = df.copy()
    df["_parsed_time"] = pd.to_datetime(df[time_col], errors="coerce")
    df["_customer_thread"] = df[name_col].replace("", pd.NA).ffill()

    pairs = []
    for i in range(1, len(df)):
        cur = df.iloc[i]
        prev = df.iloc[i - 1]

        is_staff_reply = pd.notna(cur[staff_col]) and str(cur[staff_col]).strip() != ""
        is_prev_customer_msg = pd.notna(prev[name_col]) and str(prev[name_col]).strip() != "" and \
                                (pd.isna(prev[staff_col]) or str(prev[staff_col]).strip() == "")

        if is_staff_reply and is_prev_customer_msg and pd.notna(cur["_parsed_time"]) and pd.notna(prev["_parsed_time"]):
            response_minutes = (cur["_parsed_time"] - prev["_parsed_time"]).total_seconds() / 60
            if response_minutes >= 0:
                pairs.append({
                    "staff_name": cur[staff_col],
                    "customer_name": cur["_customer_thread"],
                    "question_text": prev[text_col],
                    "reply_text": cur[text_col],
                    "response_minutes": round(response_minutes, 1),
                })

    return pd.DataFrame(pairs)


STAFF_QUALITY_SYSTEM_PROMPT = """你是一家「會計師事務所暨商務中心」的客服品質稽核員。
請針對輸入的每一組「客戶提問 → 同事回覆」，評估回覆品質，並「只回傳 JSON 陣列」，不要有任何其他文字、不要用 markdown code block 包住。

每筆請輸出以下欄位：
- index: 對應輸入的序號（整數）
- tone: 語氣分類，必須是以下其中一種："親切專業", "中性/一般", "罐頭感/敷衍", "不耐煩/生硬"
- quality_score: 1到5的整數，5分表示完整回答問題且態度良好，1分表示答非所問或態度不佳
- issue_note: 若有明顯問題（如答非所問、態度不佳、資訊錯誤），用10字以內簡短說明，否則為空字串

範例輸出格式：
[{"index": 0, "tone": "親切專業", "quality_score": 5, "issue_note": ""}]
"""


def analyze_staff_quality_batch(model: genai.GenerativeModel, batch_df: pd.DataFrame) -> list:
    """將一批「提問→回覆」配對送進 Gemini，評估語氣與品質"""
    items = []
    for i, row in enumerate(batch_df.itertuples()):
        items.append(f"{i}: 客戶問：{row.question_text}\n    同事回：{row.reply_text}")

    user_prompt = "以下是本批次的提問與回覆配對，請依格式評估每一筆：\n\n" + "\n\n".join(items)

    response = model.generate_content(
        [STAFF_QUALITY_SYSTEM_PROMPT, user_prompt],
        generation_config={
            "temperature": 0,
            "response_mime_type": "application/json",
        },
    )

    raw_text = response.text.strip()
    raw_text = re.sub(r"^```(json)?", "", raw_text.strip())
    raw_text = re.sub(r"```$", "", raw_text.strip())

    try:
        parsed = json.loads(raw_text)
        if isinstance(parsed, dict):
            for v in parsed.values():
                if isinstance(v, list):
                    parsed = v
                    break
        return parsed
    except json.JSONDecodeError:
        st.warning("同事表現分析有一批資料解析失敗，已略過該批次")
        return []


# =========================================================
# 主流程
# =========================================================
if uploaded_file is not None:
    df_raw = load_dataframe(uploaded_file)

    st.success(f"已讀取檔案，共 {len(df_raw)} 筆對話紀錄")
    with st.expander("預覽原始資料"):
        st.dataframe(df_raw.head(10))

    # 自動猜測關鍵欄位，並讓使用者可手動修正
    st.subheader("② 確認欄位對應")
    col1, col2, col3 = st.columns(3)
    with col1:
        name_col = st.selectbox(
            "客戶暱稱欄位",
            df_raw.columns,
            index=list(df_raw.columns).index(guess_column(df_raw, ["暱稱", "姓名", "客戶", "name"])),
        )
    with col2:
        source_col_options = ["(無此欄位)"] + list(df_raw.columns)
        default_source = guess_column(df_raw, ["OA", "來源", "帳號", "source"])
        source_col = st.selectbox(
            "LINE OA 來源欄位（選填）",
            source_col_options,
            index=source_col_options.index(default_source) if default_source in source_col_options else 0,
        )
    with col3:
        text_col = st.selectbox(
            "對話內容欄位",
            df_raw.columns,
            index=list(df_raw.columns).index(guess_column(df_raw, ["對話", "內容", "訊息", "content", "message"])),
        )

    st.markdown("**以下兩個欄位為選填，若要分析「同事回覆速度／品質」才需要設定：**")
    col4, col5 = st.columns(2)
    with col4:
        staff_col_options = ["(不分析同事表現)"] + list(df_raw.columns)
        default_staff = guess_column(df_raw, ["同事", "回覆人", "客服", "員工", "staff"])
        staff_col = st.selectbox(
            "同事/回覆人姓名欄位",
            staff_col_options,
            index=staff_col_options.index(default_staff) if default_staff in staff_col_options else 0,
        )
    with col5:
        time_col_options = ["(不分析同事表現)"] + list(df_raw.columns)
        default_time = guess_column(df_raw, ["時間", "日期", "time", "date"])
        time_col = st.selectbox(
            "精確時間欄位（需含時分）",
            time_col_options,
            index=time_col_options.index(default_time) if default_time in time_col_options else 0,
        )

    analyze_staff_performance = staff_col != "(不分析同事表現)" and time_col != "(不分析同事表現)"
    if analyze_staff_performance:
        st.caption(
            "⚠️ 這項分析涉及同事個人表現，建議先讓同事知情，並將結果用於輔導改進而非考核處罰，"
            "以免影響團隊信任。"
        )

    # 去敏感化
    st.subheader("③ 資料去敏感化")
    df_masked = df_raw.copy()
    df_masked[text_col] = df_masked[text_col].apply(mask_sensitive_info)

    masked_count = (df_raw[text_col].astype(str) != df_masked[text_col].astype(str)).sum()
    st.info(f"已自動遮蔽 {masked_count} 筆對話中的身分證字號／統編／手機號碼")

    with st.expander("預覽去敏感化後的資料"):
        st.dataframe(df_masked[[name_col, text_col]].head(10))

    st.divider()

    # =========================================================
    # Step 4：執行 AI 分析
    # =========================================================
    st.subheader("④ 執行 CDP 分析")

    if st.button("🚀 開始分析", type="primary", use_container_width=False):
        if not api_key:
            st.error("請先在左側側邊欄輸入 Google Gemini API Key")
            st.stop()

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)

        # 若有設定同事欄位，CDP業務分類只分析「客戶訊息」，排除同事回覆列
        if analyze_staff_performance:
            is_customer_row = df_masked[staff_col].isna() | (df_masked[staff_col].astype(str).str.strip() == "")
            df_for_category = df_masked[is_customer_row].reset_index(drop=True)
        else:
            df_for_category = df_masked

        all_results = []
        total_rows = len(df_for_category)
        n_batches = (total_rows + batch_size - 1) // batch_size

        progress_bar = st.progress(0, text="分析中...")

        for b in range(n_batches):
            start = b * batch_size
            end = min(start + batch_size, total_rows)
            batch_df = df_for_category.iloc[start:end]

            batch_result = analyze_batch(model, batch_df, text_col)

            for item in batch_result:
                idx = item.get("index")
                if idx is not None and 0 <= idx < len(batch_df):
                    global_idx = start + idx
                    item["global_index"] = global_idx
                    item["customer_name"] = df_for_category.iloc[global_idx][name_col]
                    item["original_text"] = df_for_category.iloc[global_idx][text_col]
                    all_results.append(item)

            progress_bar.progress((b + 1) / n_batches, text=f"分析中... 第 {b+1}/{n_batches} 批")
            time.sleep(4)  # Gemini 免費額度約 15 次/分鐘，預留間隔避免觸發 rate limit

        progress_bar.empty()

        if not all_results:
            st.error("分析失敗，未取得任何有效結果，請確認 API Key 或稍後重試")
            st.stop()

        result_df = pd.DataFrame(all_results)
        st.session_state["result_df"] = result_df
        st.success(f"分析完成！共成功分析 {len(result_df)} / {total_rows} 筆客戶訊息")

        # ---- 同事回覆表現分析（若有設定欄位）----
        if analyze_staff_performance:
            with st.spinner("正在分析同事回覆速度與品質..."):
                pairs_df = compute_response_pairs(df_masked, name_col, staff_col, time_col, text_col)

                if len(pairs_df) == 0:
                    st.warning("找不到可配對的「客戶問題→同事回覆」，請確認資料排序是否為一問一答，或欄位選擇是否正確")
                else:
                    quality_results = []
                    n_qbatches = (len(pairs_df) + batch_size - 1) // batch_size
                    q_progress = st.progress(0, text="分析同事回覆品質中...")

                    for b in range(n_qbatches):
                        start = b * batch_size
                        end = min(start + batch_size, len(pairs_df))
                        batch_pairs = pairs_df.iloc[start:end]

                        batch_q_result = analyze_staff_quality_batch(model, batch_pairs)

                        for item in batch_q_result:
                            idx = item.get("index")
                            if idx is not None and 0 <= idx < len(batch_pairs):
                                global_idx = start + idx
                                item["staff_name"] = pairs_df.iloc[global_idx]["staff_name"]
                                item["customer_name"] = pairs_df.iloc[global_idx]["customer_name"]
                                item["response_minutes"] = pairs_df.iloc[global_idx]["response_minutes"]
                                item["question_text"] = pairs_df.iloc[global_idx]["question_text"]
                                item["reply_text"] = pairs_df.iloc[global_idx]["reply_text"]
                                quality_results.append(item)

                        q_progress.progress((b + 1) / n_qbatches, text=f"分析同事回覆品質中... 第 {b+1}/{n_qbatches} 批")
                        time.sleep(4)

                    q_progress.empty()
                    st.session_state["staff_quality_df"] = pd.DataFrame(quality_results)
                    st.success(f"同事回覆表現分析完成！共分析 {len(quality_results)} 組提問回覆配對")

    # =========================================================
    # Step 5：顯示分析結果
    # =========================================================
    if "result_df" in st.session_state:
        result_df = st.session_state["result_df"]

        st.divider()
        st.header("📈 分析結果")

        # ---- 業務分類圓餅圖 ----
        st.subheader("業務分類圓餅圖")
        category_counts = result_df["category"].value_counts().reset_index()
        category_counts.columns = ["業務分類", "件數"]

        fig = px.pie(
            category_counts,
            names="業務分類",
            values="件數",
            hole=0.4,
            template="plotly_white",
            color_discrete_sequence=["#1F4E5F", "#4A90A4", "#8FBCC7", "#C4DDE3", "#D9D9D9"],
        )
        fig.update_traces(textinfo="percent+label")
        fig.update_layout(font=dict(color="#1A1A1A"), plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)

        col_a, col_b = st.columns(2)

        # ---- 高價值商機清單 ----
        with col_a:
            st.subheader("💰 高價值商機清單")
            opp_df = result_df[result_df["cross_sell_opportunity"] == True][
                ["customer_name", "cross_sell_note", "original_text"]
            ].rename(columns={
                "customer_name": "客戶",
                "cross_sell_note": "商機說明",
                "original_text": "原始對話",
            })
            if len(opp_df) > 0:
                st.dataframe(opp_df, use_container_width=True, hide_index=True)
            else:
                st.caption("本批資料未偵測到跨界商機")

        # ---- 流失風險清單 ----
        with col_b:
            st.subheader("⚠️ 流失風險清單")
            risk_df = result_df[result_df["churn_risk"] == True][
                ["customer_name", "churn_note", "original_text"]
            ].rename(columns={
                "customer_name": "客戶",
                "churn_note": "風險說明",
                "original_text": "原始對話",
            })
            if len(risk_df) > 0:
                st.dataframe(risk_df, use_container_width=True, hide_index=True)
            else:
                st.caption("本批資料未偵測到流失風險")

        # ---- FAQ 摘要 ----
        st.subheader("❓ 本月 FAQ 摘要（Top 3 常見問題）")
        faq_counts = result_df["question_summary"].value_counts().head(3)

        if len(faq_counts) > 0:
            for i, (question, count) in enumerate(faq_counts.items(), 1):
                st.markdown(f"**{i}. {question}**（出現 {count} 次）")
        else:
            st.caption("尚無足夠資料產生 FAQ 摘要")

        # ---- 下載完整分析結果 ----
        st.divider()
        csv_output = result_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "📥 下載完整分析結果 (CSV)",
            data=csv_output,
            file_name="cdp_analysis_result.csv",
            mime="text/csv",
        )

    # =========================================================
    # Step 6：同事回覆表現分析儀表板
    # =========================================================
    if "staff_quality_df" in st.session_state:
        staff_df = st.session_state["staff_quality_df"]

        st.divider()
        st.header("👥 同事回覆表現分析")
        st.caption("此分析結果建議用於團隊輔導與流程改善，避免直接作為個人考核依據。")

        if len(staff_df) == 0:
            st.info("尚無可分析的同事回覆資料")
        else:
            col_c, col_d = st.columns(2)

            # ---- 平均回覆時間 by 同事 ----
            with col_c:
                st.subheader("⏱️ 平均回覆時間（分鐘）")
                avg_time = staff_df.groupby("staff_name")["response_minutes"].mean().round(1).reset_index()
                avg_time.columns = ["同事", "平均回覆時間(分鐘)"]
                avg_time = avg_time.sort_values("平均回覆時間(分鐘)", ascending=False)

                fig_time = px.bar(
                    avg_time,
                    x="同事",
                    y="平均回覆時間(分鐘)",
                    text="平均回覆時間(分鐘)",
                    template="plotly_white",
                    color_discrete_sequence=["#1F4E5F"],
                )
                fig_time.update_layout(font=dict(color="#1A1A1A"), plot_bgcolor="white", paper_bgcolor="white")
                st.plotly_chart(fig_time, use_container_width=True)

            # ---- 平均品質分數 by 同事 ----
            with col_d:
                st.subheader("⭐ 平均回覆品質分數（1-5分）")
                avg_quality = staff_df.groupby("staff_name")["quality_score"].mean().round(2).reset_index()
                avg_quality.columns = ["同事", "平均品質分數"]
                avg_quality = avg_quality.sort_values("平均品質分數", ascending=False)

                fig_quality = px.bar(
                    avg_quality,
                    x="同事",
                    y="平均品質分數",
                    text="平均品質分數",
                    range_y=[0, 5],
                    template="plotly_white",
                    color_discrete_sequence=["#4A90A4"],
                )
                fig_quality.update_layout(font=dict(color="#1A1A1A"), plot_bgcolor="white", paper_bgcolor="white")
                st.plotly_chart(fig_quality, use_container_width=True)

            # ---- 語氣分布 ----
            st.subheader("🗣️ 語氣分布（依同事）")
            tone_pivot = pd.crosstab(staff_df["staff_name"], staff_df["tone"])
            st.dataframe(tone_pivot, use_container_width=True)

            # ---- 待改善清單：回覆過慢或語氣不佳 ----
            st.subheader("🔍 待關注案例（回覆過慢／語氣不佳）")
            slow_threshold = staff_df["response_minutes"].quantile(0.75)
            flagged = staff_df[
                (staff_df["response_minutes"] > slow_threshold) |
                (staff_df["tone"].isin(["罐頭感/敷衍", "不耐煩/生硬"])) |
                (staff_df["quality_score"] <= 2)
            ][["staff_name", "customer_name", "response_minutes", "tone", "quality_score", "issue_note", "question_text", "reply_text"]].rename(columns={
                "staff_name": "同事",
                "customer_name": "客戶",
                "response_minutes": "回覆時間(分鐘)",
                "tone": "語氣",
                "quality_score": "品質分數",
                "issue_note": "問題說明",
                "question_text": "客戶提問",
                "reply_text": "同事回覆",
            })

            if len(flagged) > 0:
                st.dataframe(flagged, use_container_width=True, hide_index=True)
            else:
                st.caption("本批資料未發現明顯待改善案例")

            # ---- 下載同事表現分析結果 ----
            staff_csv_output = staff_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "📥 下載同事回覆表現分析 (CSV)",
                data=staff_csv_output,
                file_name="staff_performance_analysis.csv",
                mime="text/csv",
            )

else:
    st.info("👆 請先上傳 LINE 對話紀錄檔案，或使用上方的模擬範例資料測試系統")
