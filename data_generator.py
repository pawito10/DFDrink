import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# =========================
# ダミー需要予測データ生成
# =========================

np.random.seed(42)

# 期間設定
start_date = datetime(2025, 1, 1)
days = 365

rows = []

for i in range(days):
    date = start_date + timedelta(days=i)

    # 基本情報
    month = date.month
    day_of_week = date.weekday()  # 0=月曜
    weekend = 1 if day_of_week >= 5 else 0

    # 気温（季節っぽく）
    base_temp = 15 + 10 * np.sin((2 * np.pi * i) / 365)
    temperature = round(base_temp + np.random.normal(0, 3), 1)

    # 降水
    rain = np.random.choice([0, 1], p=[0.7, 0.3])

    # イベント
    event = np.random.choice([0, 1], p=[0.85, 0.15])

    # =========================
    # 売上ロジック
    # =========================

    sales = 50

    # 気温が高いと売上UP
    sales += temperature * 2

    # 土日効果
    if weekend:
        sales += 15

    # イベント効果
    if event:
        sales += 25

    # 雨の日は少し減る
    if rain:
        sales -= 10

    # ノイズ追加
    sales += np.random.normal(0, 5)

    sales = max(10, int(sales))

    rows.append({
        "date": date.strftime("%Y-%m-%d"),
        "temperature": temperature,
        "rain": rain,
        "weekend": weekend,
        "event": event,
        "sales": sales
    })

# DataFrame化
sales_df = pd.DataFrame(rows)

# 保存
sales_df.to_csv("sales_data.csv", index=False)
print("=== サンプルデータ ===")
print(sales_df.head())

print("\nCSV保存完了: sales_data.csv")
# =========================
# RAG用知識データ生成
# =========================

rag_knowledge = [
    "猛暑日は炭酸飲料の売上が増加する傾向があります。",
    "花火大会の日は飲料需要が増加します。",
    "雨の日は来店客数が減少しやすいです。",
    "土日は家族連れが増えるため売上が伸びます。",
    "気温30度以上ではスポーツドリンク需要が増えます。",
    "地域イベント開催日は通常より需要が高まります。"
]

with open("rag_knowledge.txt", "w", encoding="utf-8") as f:
    for line in rag_knowledge:
        f.write(line + "\n")

print("RAG知識データ保存完了: rag_knowledge.txt")