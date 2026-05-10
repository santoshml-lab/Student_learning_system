
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
import os
import psycopg2

# ====================================================
# APP
# ====================================================

app = FastAPI()

# ====================================================
# CORS
# ====================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ====================================================
# GROQ CLIENT
# ====================================================

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

# ====================================================
# DB CONNECTION
# ====================================================

def get_conn():
    return psycopg2.connect(os.getenv("DATABASE_URL"))

# ====================================================
# INIT TABLE (SAFE)
# ====================================================

def init_db():
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS chapters (
        id SERIAL PRIMARY KEY,
        student_class TEXT,
        subject TEXT,
        chapter TEXT
    );
    """)

    conn.commit()
    cur.close()
    conn.close()

init_db()

# ====================================================
# ROOT
# ====================================================

@app.get("/")
def home():
    return {"status": "ICSE AI Teacher Running 🚀"}

# ====================================================
# REQUEST MODEL
# ====================================================

class LearnRequest(BaseModel):
    student_class: str
    subject: str
    chapter: str

# ====================================================
# LEARN API (AI + SAVE)
# ====================================================

@app.post("/learn")
def learn(data: LearnRequest):

    prompt = f"""
You are a highly professional ICSE school teacher.

Teach:
Class: {data.student_class}
Subject: {data.subject}
Chapter: {data.chapter}

Return ONLY MARKDOWN:
- Explanation
- Examples
- Formulas
- 5 Practice Questions
"""

    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "system", "content": prompt}]
    )

    lesson = completion.choices[0].message.content

    # SAVE TO DB
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO chapters (student_class, subject, chapter)
        VALUES (%s, %s, %s)
    """, (data.student_class, data.subject, data.chapter))

    conn.commit()
    cur.close()
    conn.close()

    return {
        "class": data.student_class,
        "subject": data.subject,
        "chapter": data.chapter,
        "lesson": lesson
    }

# ====================================================
# GET CHAPTERS
# ====================================================

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
            {"class": r[0], "subject": r[1], "chapter": r[2]}
            for r in rows
        ]
    }
