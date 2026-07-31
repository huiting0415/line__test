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

    api_key = st.text_input(
        "Google Gemini API Key",
        type="password",
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
    try:
        with open("sample_line_conversations.csv", "rb") as f:
            st.download_button(
                "下載範例 CSV",
                data=f,
                file_name="sample_line_conversations.csv",
                mime="text/csv",
            )
    except FileNotFoundError:
        st.info("請將 sample_line_conversations.csv 放在與 app.py 相同的資料夾")


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

        all_results = []
        total_rows = len(df_masked)
        n_batches = (total_rows + batch_size - 1) // batch_size

        progress_bar = st.progress(0, text="分析中...")

        for b in range(n_batches):
            start = b * batch_size
            end = min(start + batch_size, total_rows)
            batch_df = df_masked.iloc[start:end]

            batch_result = analyze_batch(model, batch_df, text_col)

            for item in batch_result:
                idx = item.get("index")
                if idx is not None and 0 <= idx < len(batch_df):
                    global_idx = start + idx
                    item["global_index"] = global_idx
                    item["customer_name"] = df_masked.iloc[global_idx][name_col]
                    item["original_text"] = df_masked.iloc[global_idx][text_col]
                    all_results.append(item)

            progress_bar.progress((b + 1) / n_batches, text=f"分析中... 第 {b+1}/{n_batches} 批")
            time.sleep(4)  # Gemini 免費額度約 15 次/分鐘，預留間隔避免觸發 rate limit

        progress_bar.empty()

        if not all_results:
            st.error("分析失敗，未取得任何有效結果，請確認 API Key 或稍後重試")
            st.stop()

        result_df = pd.DataFrame(all_results)
        st.session_state["result_df"] = result_df
        st.success(f"分析完成！共成功分析 {len(result_df)} / {total_rows} 筆對話")

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
        )
        fig.update_traces(textinfo="percent+label")
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

else:
    st.info("👆 請先上傳 LINE 對話紀錄檔案，或使用上方的模擬範例資料測試系統")
