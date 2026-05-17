from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
import os
import psycopg2

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

DATABASE_URL = os.getenv("DATABASE_URL")

if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)


def get_conn():
    if not DATABASE_URL:
        raise Exception("DATABASE_URL missing")
    return psycopg2.connect(DATABASE_URL, sslmode="require")


def init_db():
    try:
        conn = get_conn()
        cur = conn.cursor()

        cur.execute("""
        CREATE TABLE IF NOT EXISTS chapters (
            id SERIAL PRIMARY KEY,
            student_name TEXT,
            student_class TEXT,
            subject TEXT,
            chapter TEXT,
            xp INTEGER DEFAULT 10
        )
        """)

        conn.commit()
        cur.close()
        conn.close()

    except Exception as e:
        print("DB ERROR:", e)


init_db()


class LearnRequest(BaseModel):
    student_name: str
    student_class: str
    subject: str
    chapter: str


class ChatInput(BaseModel):
    message: str


@app.get("/")
def home():
    return {"status": "ExamPanic AI running 🚀"}


@app.get("/health")
def health():
    return {"status": "ok"}


# 📚 LEARN ENGINE (ICSE BOOK STYLE)
@app.post("/learn")
def learn(data: LearnRequest):
    try:
        prompt = f"""
You are an ICSE textbook author + viral teacher AI.

Format:
📘 Chapter Title
📌 Definition
📖 Explanation
🧠 Examples
⭐ Key Points
📝 Questions
🔁 Revision Box

Make it simple and exam focused.

Class: {data.student_class}
Subject: {data.subject}
Chapter: {data.chapter}
"""

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "system", "content": prompt}],
            max_tokens=2000
        )

        return {
            "lesson": response.choices[0].message.content,
            "mode": "ICSE_BOOK"
        }

    except Exception as e:
        return {"error": str(e)}


# 💬 CHAT ENGINE
@app.post("/chat")
def chat(data: ChatInput):
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": """
You are ExamPanic AI — a friendly ICSE study coach.

Be:
- simple
- motivational
- exam focused
"""
                },
                {
                    "role": "user",
                    "content": data.message
                }
            ],
            max_tokens=1200
        )

        return {"reply": response.choices[0].message.content}

    except Exception as e:
        return {"error": str(e)}


# 📚 SAVE CHAPTER
@app.post("/save-chapter")
def save_chapter(data: LearnRequest):
    conn = get_conn()
    cur = conn.cursor()

    cur.execute(
        "INSERT INTO chapters (student_name, student_class, subject, chapter) VALUES (%s,%s,%s,%s)",
        (data.student_name, data.student_class, data.subject, data.chapter)
    )

    conn.commit()
    cur.close()
    conn.close()

    return {"message": "saved"}


# 🏆 LEADERBOARD (FIXED XP BUG)
@app.get("/leaderboard")
def leaderboard():
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
        SELECT student_name, COALESCE(SUM(xp),0)
        FROM chapters
        GROUP BY student_name
        ORDER BY COALESCE(SUM(xp),0) DESC
        LIMIT 10
    """)

    rows = cur.fetchall()

    cur.close()
    conn.close()

    return {
        "leaderboard": [
            {"name": r[0], "xp": int(r[1])}
            for r in rows
        ]
    }


# ⚡ QUICK ACCESS
@app.get("/quick-access")
def quick_access():
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("SELECT subject, chapter FROM chapters LIMIT 10")
    rows = cur.fetchall()

    cur.close()
    conn.close()

    return {
        "quick_access": [
            {"subject": r[0], "chapter": r[1]}
            for r in rows
        ]
    }


# 😂 VIRAL MEME ENDPOINT (IMPORTANT)
@app.post("/meme")
def meme(data: ChatInput):
    return {
        "meme": f"When you study {data.message} but your brain goes ‘buffering…’ 🤯😂"
    }


# 🔥 SHARE REWARD (VIRAL LOOP HOOK)
@app.post("/share")
def share():
    return {
        "status": "ok",
        "bonus_xp": 5,
        "message": "Shared successfully 🚀"
    }
