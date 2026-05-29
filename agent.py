"""
beamdata Sales Agent
Step2 
Sales Chatbot RAG + ChromaDB
"""

import os
import ast
import pandas as pd
import chromadb
from groq import Groq
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter

import os
os.environ['ORT_LOGGING_LEVEL'] = '3'
import warnings
warnings.filterwarnings("ignore")

# ── Setup ─────────────────────────────────────────────────────────────────────
load_dotenv()
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# ── 1. Load Data ──────────────────────────────────────────────────────────────
def load_data():
    df = pd.read_csv("data/ticket_risk_results.csv")
    print(f"✅ Loaded {len(df)} tickets from CSV")
    return df

# ── 2. Load Policies ──────────────────────────────────────────────────────────
def load_policies():
    with open("knowledge/beamdata_policies_en.txt", "r", encoding="utf-8") as f:
        policies = f.read()
    print("✅ Policies loaded")
    return policies

# ── 3. Build ChromaDB ─────────────────────────────────────────────────────────
def build_chromadb(policies):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50,
        separators=["\n---\n", "\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_text(policies)
    # chunks = [c for c in chunks if len(c) > 100]

    chroma_client = chromadb.Client()
    collection = chroma_client.create_collection(name="beamdata_policies")

    for i, chunk in enumerate(chunks):
        collection.add(
            documents=[chunk],
            ids=[f"chunk_{i}"]
        )

    print(f"✅ Stored {len(chunks)} chunks in ChromaDB")
    return collection

# ── 4. RAG Retrieval ──────────────────────────────────────────────────────────
def retrieve(collection, question, n_results=2):
    results = collection.query(
        query_texts=[question],
        n_results=n_results
    )
    return results["documents"][0]

# ── 5. Chatbot with RAG ───────────────────────────────────────────────────────
def chatbot_with_rag(collection, user_message):
    relevant = retrieve(collection, user_message)
    context = "\n\n".join(relevant)

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": f"""You are a professional sales assistant for beamdata company.

Answer ONLY based on this context:
{context}

Rules:
- Never share private customer data
- If answer is not in context, say: 'Please contact us at support@beamdata.ai'
- Be friendly and professional
- Keep answers concise"""
            },
            {
                "role": "user",
                "content": user_message
            }
        ]
    )
    return response.choices[0].message.content

# ── 6. Full Pipeline ──────────────────────────────────────────────────────────
def full_pipeline(collection, row):
    # استخراج النص
    try:
        ticket_data = ast.literal_eval(row["ticket_text"])
        ticket = ticket_data["ticket_text"].strip()
    except Exception:
        ticket = str(row["ticket_text"]).strip()

    action = row["action"]
    risk   = row["risk_level"]

    # Routing Decision
    if action == "escalate_to_human":
        return {
            "ticket": ticket,
            "status": "BLOCKED",
            "risk":   risk,
            "reply":  "This request has been escalated to our security team."
        }

    # Safe Route → RAG Chatbot
    reply = chatbot_with_rag(collection, ticket)
    return {
        "ticket": ticket,
        "status": "ALLOWED",
        "risk":   risk,
        "reply":  reply
    }

# ── 7. Live Chat ──────────────────────────────────────────────────────────────
def live_chat(collection):
    print("\n" + "=" * 50)
    print("  🤖 beamdata Sales Assistant")
    print("  Type 'exit' to quit")
    print("=" * 50)

    while True:
        user_input = input("\nYou: ").strip()

        if user_input.lower() == "exit":
            print("Thank you! Goodbye 👋")
            break

        if not user_input:
            continue

        mock_row = {
            "ticket_text": user_input,
            "action": "allow",
            "risk_level": "low"
        }

        result = full_pipeline(collection, mock_row)
        print(f"\n🤖 beamdata: {result['reply']}")

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("\n🚀 Starting beamdata Sales Agent...\n")

    # تحميل البيانات
    df         = load_data()
    policies   = load_policies()
    collection = build_chromadb(policies)

    print("\n" + "=" * 50)
    print("  📊 Running Pipeline on CSV Data")
    print("=" * 50 + "\n")

    # تشغيل الـ pipeline على ملف زميلتك
    for _, row in df.iterrows():
        result = full_pipeline(collection, row)
        print(f"Customer : {result['ticket']}")
        print(f"Risk     : {result['risk']}")
        print(f"Status   : {result['status']}")
        print(f"Reply    : {result['reply']}")
        print("-" * 50)

    # محادثة مباشرة
    live_chat(collection)


if __name__ == "__main__":
    main()