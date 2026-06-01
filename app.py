# =========================
# FastAPI + ML + RAG + LLM
# =========================

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
import csv
from datetime import datetime
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# =========================
# 初期設定
# =========================

load_dotenv()

app = FastAPI()


# =========================
# 入力スキーマ
# =========================

class InputData(BaseModel):
    temperature: float
    rain: int
    weekend: int
    event: int


# =========================
# モデル・RAG初期化（起動時1回）
# =========================

def init_model():
    df = pd.read_csv("sales_data.csv")

    X = df[["temperature", "rain", "weekend", "event"]]
    y = df["sales"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    mae, mape = evaluate_model(model, X_test, y_test)

    print("=== モデル精度 ===")
    print(f"MAE: {mae:.2f}")
    print(f"MAPE: {mape:.2f}%")

    return model


def init_rag():
    with open("rag_knowledge.txt", "r", encoding="utf-8") as f:
        lines = f.readlines()

    docs = [Document(page_content=line.strip()) for line in lines]

    print("\n=== RAG Documents ===")
    print(f"Document数: {len(docs)}")

    embedding = OpenAIEmbeddings()

    """db = Chroma.from_documents(
        docs,
        embedding,
        persist_directory="./chroma_db"
    )"""
    db = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embedding
    )

    print("=== Collection Count ===")
    print(db._collection.count())

    return db

def evaluate_model(model, X_test, y_test):

    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)

    mape = np.mean(np.abs((y_test - preds) / y_test)) * 100

    return mae, mape



model = init_model()
db = init_rag()


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)


# =========================
# ユーティリティ関数
# =========================

def build_query(data: dict):
    parts = []

    if data["temperature"] >= 30:
        parts.append("猛暑")

    if data["weekend"] == 1:
        parts.append("土日")

    if data["event"] == 1:
        parts.append("地域イベント")

    if data["rain"] == 1:
        parts.append("雨")

    return " ".join(parts)


def predict_sales(data: dict):
    df = pd.DataFrame([data])
    return model.predict(df)[0]


def search_rag(query: str):
    print("\n=== Query ===")
    print(query)

    # 空クエリ対策
    if not query.strip():
        print("\n=== Search Results ===")
        print("クエリが空のため検索をスキップ")
        return []

    results = db.similarity_search(query, k=2)

    print("\n=== Search Results ===")

    for i, r in enumerate(results):
        print(f"{i}: {r.page_content}")

    return results
    


def generate_explanation(pred, contexts):
    context_text = "\n".join([c.page_content for c in contexts])

    prompt = f"""
あなたは優秀な需要予測アナリストです。

予測需要:
{pred:.1f}

参考情報:
{context_text}

自然な日本語で説明してください。
"""

    return llm.invoke(prompt).content


# =========================
# APIエンドポイント
# =========================

@app.post("/analyze")
def analyze(input_data: InputData):

    data = input_data.dict()

    # ① 予測
    pred = predict_sales(data)

    # ② RAGクエリ
    query = build_query(data)

    # ③ RAG検索
    contexts = search_rag(query)

    # ④ LLM説明
    explanation = generate_explanation(pred, contexts)

    # =========================
    # 履歴保存
    # =========================

    import csv
    from datetime import datetime

    log_data = [
        datetime.now().isoformat(),
        input_data.temperature,
        input_data.rain,
        input_data.weekend,
        input_data.event,
        pred
    ]

    with open("logs.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(log_data)

    return {
        "prediction": float(pred),
        "query": query,
        "contexts": [c.page_content for c in contexts],
        "explanation": explanation
    }


# =========================
# ヘルスチェック
# =========================

@app.get("/")
def root():
    return {"status": "ok"}