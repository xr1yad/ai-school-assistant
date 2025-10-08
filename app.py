import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import numpy as np
import os
import json
import uuid

# ---------------- إعداد النظام ----------------
EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
DATA_DIR = "data"
MEMORY_DIR = "memory"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MEMORY_DIR, exist_ok=True)
DB_DIR = "chroma_db"

client = chromadb.Client(Settings(persist_directory=DB_DIR))
try:
    collection = client.get_collection("curriculum")
except:
    collection = client.create_collection("curriculum")

# ---------------- واجهة Streamlit ----------------
st.set_page_config(page_title="مساعد المنهاج الدراسي", layout="wide")
st.title("📘 مساعد المنهاج الدراسي")
st.write("مرحبًا 👋 هذا النظام يجيب فقط من الملفات التي أضفتها أنت (المعلم).")

mode = st.sidebar.selectbox("اختر الوضع:", ["👨‍🏫 وضع المعلم", "🎓 وضع الطالب"])

# ---------------- وضع المعلم ----------------
if mode == "👨‍🏫 وضع المعلم":
    st.header("👨‍🏫 رفع ملفات المنهاج")
    uploaded_files = st.file_uploader("ارفع ملفات PDF الخاصة بالمنهاج:", type=["pdf"], accept_multiple_files=True)
    if uploaded_files:
        for file in uploaded_files:
            reader = PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"

            # تقسيم النص إلى أجزاء صغيرة
            words = text.split()
            chunk_size = 300
            overlap = 50
            chunks = []
            i = 0
            idx = 0
            while i < len(words):
                chunk = " ".join(words[i:i+chunk_size])
                chunks.append(chunk)
                i += chunk_size - overlap

            # إنشاء embeddings
            embs = EMBED_MODEL.encode(chunks)
            ids = [str(uuid.uuid4()) for _ in chunks]
            metas = [{"file": file.name} for _ in chunks]
            collection.add(documents=chunks, embeddings=embs.tolist(), metadatas=metas, ids=ids)

        client.persist()
        st.success("✅ تم رفع وفهرسة جميع الملفات بنجاح!")

# ---------------- وضع الطالب ----------------
else:
    st.header("🎓 اسأل الذكاء الاصطناعي")
    question = st.text_input("اكتب سؤالك هنا:")

    if st.button("اسأل"):
        if not question.strip():
            st.warning("❗ الرجاء كتابة سؤال أولاً.")
        else:
            q_emb = EMBED_MODEL.encode([question])
            results = collection.query(query_embeddings=q_emb.tolist(), n_results=3)
            docs = results["documents"][0]
            metas = results["metadatas"][0]

            if len(docs) == 0:
                st.warning("🚫 لا توجد معلومات كافية في المنهاج للإجابة عن هذا السؤال.")
            else:
                st.subheader("📖 الإجابة من المنهاج:")
                for doc, meta in zip(docs, metas):
                    st.markdown(f"**من الملف:** {meta['file']}")
                    st.write(doc)
                    st.markdown("---")

                # حفظ السؤال في ذاكرة التعلم
                memory_file = os.path.join(MEMORY_DIR, f"{uuid.uuid4()}.json")
                with open(memory_file, "w", encoding="utf-8") as f:
                    json.dump({"question": question, "answers": docs}, f, ensure_ascii=False)
                st.success("✅ تم حفظ هذا السؤال للتعلم المستقبلي.")
