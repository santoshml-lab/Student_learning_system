from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
import os
import psycopg2

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
# GROQ
# =========================

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# =========================
# DATABASE
# =========================

DATABASE_URL = os.getenv("DATABASE_URL")

if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

def get_conn():
    return psycopg2.connect(DATABASE_URL, sslmode="require")

# =========================
# INIT DB
# =========================

def init_db():

    conn = get_conn()
    cur = conn.cursor()

    # chapters
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

init_db()

# =========================
# MODEL
# =========================

class LearnRequest(BaseModel):
    student_name: str
    student_class: str
    subject: str
    chapter: str

# =========================
# ROOT
# =========================

@app.get("/")
def home():
    return {"status":"LEVEL 4 Backend Running 🚀"}

# =========================
# LEARN
# =========================

@app.post("/learn")
def learn(data: LearnRequest):

    prompt = f"""
You are an ICSE teacher.

Class: {data.student_class}
Subject: {data.subject}
Chapter: {data.chapter}

Explain with:
- Introduction
- Explanation
- Definitions
- Examples
- Key Points
- Revision Notes
- Practice Questions
"""

    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role":"system","content":prompt}
        ]
    )

    return {
        "lesson": completion.choices[0].message.content
    }

# =========================
# SAVE CHAPTER
# =========================

@app.post("/save-chapter")
def save_chapter(data: LearnRequest):

    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
    INSERT INTO chapters
    (student_name, student_class, subject, chapter)
    VALUES (%s,%s,%s,%s)
    """, (
        data.student_name,
        data.student_class,
        data.subject,
        data.chapter
    ))

    conn.commit()
    cur.close()
    conn.close()

    return {"message":"saved 🚀"}

# =========================
# GET CHAPTERS
# =========================

@app.get("/chapters")
def get_chapters():

    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
    SELECT student_name, subject, chapter
    FROM chapters
    ORDER BY id DESC
    LIMIT 20
    """)

    rows = cur.fetchall()

    cur.close()
    conn.close()

    return {
        "chapters":[
            {
                "student":r[0],
                "subject":r[1],
                "chapter":r[2]
            }
            for r in rows
        ]
    }

# =========================
# LEADERBOARD
# =========================

@app.get("/leaderboard")
def leaderboard():

    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
    SELECT student_name, SUM(xp) as total_xp
    FROM chapters
    GROUP BY student_name
    ORDER BY total_xp DESC
    LIMIT 10
    """)

    rows = cur.fetchall()

    cur.close()
    conn.close()

    return {
        "leaderboard":[
            {
                "name":r[0],
                "xp":r[1]
            }
            for r in rows
        ]
    }

# =========================
# QUICK ACCESS
# =========================

@app.get("/quick-access")
def quick_access():

    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
    SELECT DISTINCT subject, chapter
    FROM chapters
    ORDER BY subject
    LIMIT 10
    """)

    rows = cur.fetchall()

    cur.close()
    conn.close()

    return {
        "quick_access":[
            {
                "subject":r[0],
                "chapter":r[1]
            }
            for r in rows
        ]
    }

