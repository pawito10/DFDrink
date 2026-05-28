import streamlit as st
import requests
import time
import pandas as pd

st.title("📊 需要予測AIダッシュボード")

# =========================
# 入力UI
# =========================

st.sidebar.header("入力パラメータ")

temperature = st.sidebar.slider("気温", 0, 40, 25)
rain = st.sidebar.selectbox("雨", [0, 1])
weekend = st.sidebar.selectbox("週末", [0, 1])
event = st.sidebar.selectbox("イベント", [0, 1])

# =========================
# ヒント（UX改善）
# =========================

with st.expander("💡 分析ヒント"):
    st.write("・猛暑 → 飲料需要増加")
    st.write("・週末 → 来客増加")
    st.write("・イベント → 大幅な需要増加")
    st.write("・雨 → 近場需要増加")

# =========================
# 実行ボタン
# =========================

if st.button("🚀 予測する"):

    payload = {
        "temperature": temperature,
        "rain": rain,
        "weekend": weekend,
        "event": event
    }

    # =========================
    # ステップ + プログレス
    # =========================

    step = st.empty()
    progress = st.progress(0)

    step.text("🔮 Step 1: 需要予測（MLモデル）")
    progress.progress(25)
    time.sleep(0.3)

    step.text("📡 Step 2: API通信中（FastAPI）")
    progress.progress(50)

    response = requests.post(
        "http://127.0.0.1:8000/analyze",
        json=payload
    )

    result = response.json()

    step.text("🧠 Step 3: RAG + LLM生成中")
    progress.progress(80)
    time.sleep(0.3)

    step.text("✅ 完了")
    progress.progress(100)

    st.success("分析完了！")

    # =========================
    # KPI表示
    # =========================

    st.subheader("📊 予測結果")
    st.metric("需要予測", f"{result['prediction']:.1f}")

    st.subheader("🧠 AIの説明")
    st.write(result["explanation"])

    # =========================
    # RAG情報
    # =========================

    with st.expander("🔎 RAGクエリ"):
        st.write(result["query"])

    with st.expander("📚 参考情報"):
        for c in result["contexts"]:
            st.write("•", c)

    # =========================
    # 履歴表示 + グラフ
    # =========================

    st.subheader("📈 予測履歴ダッシュボード")

    try:
        df = pd.read_csv("logs.csv", header=None)

        df.columns = [
            "time",
            "temperature",
            "rain",
            "weekend",
            "event",
            "prediction"
        ]

        st.dataframe(df)

        # グラフ（予測推移）
        st.subheader("📊 予測推移")
        st.line_chart(df["prediction"])

        # 気温との関係（簡易分析）
        st.subheader("🌡 気温と需要の関係")
        st.scatter_chart(df[["temperature", "prediction"]])

    except Exception:
        st.info("まだ履歴データがありません（1回予測すると表示されます）")