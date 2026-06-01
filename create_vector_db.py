from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

with open("rag_knowledge.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()

docs = [
    Document(page_content=line.strip())
    for line in lines
]

embedding = OpenAIEmbeddings()

db = Chroma.from_documents(
    docs,
    embedding,
    persist_directory="./chroma_db"
)

print("Vector DB作成完了")
print("件数:", db._collection.count())