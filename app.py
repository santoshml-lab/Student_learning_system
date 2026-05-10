from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
import os
import psycopg2

# =========================
# APP INIT
# =========================

app = FastAPI()

# =========================
# CORS
# =========================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# GROQ CLIENT
# =========================

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# =========================
# DATABASE SETUP (RENDER SAFE)
# =========================

DATABASE_URL = os.getenv("DATABASE_URL")

# fix postgres:// issue
if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

def get_conn():
    return psycopg2.connect(
        DATABASE_URL,
        sslmode="require"
    )

def init_db():
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS chapters (
        id SERIAL PRIMARY KEY,
        student_class TEXT,
        subject TEXT,
        chapter TEXT
    )
    """)

    conn.commit()
    cur.close()
    conn.close()

# run once
init_db()

# =========================
# MODELS
# =========================

class LearnRequest(BaseModel):
    student_class: str
    subject: str
    chapter: str

# =========================
# ROOT CHECK
# =========================

@app.get("/")
def home():
    return {
        "status": "AI Learning System Running 🚀"
    }

# =========================
# AI LEARNING API
# =========================

@app.post("/learn")
def learn(data: LearnRequest):

    prompt = f"""
You are a professional ICSE teacher.

Teach in simple, clear, structured way.

Class: {data.student_class}
Subject: {data.subject}
Chapter: {data.chapter}

Return ONLY Markdown format with:
- Introduction
- Explanation
- Definitions
- Examples
- Key Points
- Revision Notes
- 5 Practice Questions
"""

    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": prompt}
        ]
    )

    lesson = completion.choices[0].message.content

    return {
        "lesson": lesson
    }

# =========================
# SAVE CHAPTER
# =========================

@app.post("/save-chapter")
def save_chapter(data: LearnRequest):

    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO chapters (student_class, subject, chapter)
        VALUES (%s, %s, %s)
    """, (
        data.student_class,
        data.subject,
        data.chapter
    ))

    conn.commit()
    cur.close()
    conn.close()

    return {
        "message": "Chapter saved successfully 🚀"
    }

# =========================
# GET CHAPTERS
# =========================

@app.get("/chapters")
def get_chapters():

    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
        SELECT student_class, subject, chapter FROM chapters
    """)

    rows = cur.fetchall()

    cur.close()
    conn.close()

    return {
        "chapters": [
            {
                "class": r[0],
                "subject": r[1],
                "chapter": r[2]
            }
            for r in rows
        ]
    }
