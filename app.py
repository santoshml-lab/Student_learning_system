
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
import os
import psycopg2
from fastapi.responses import StreamingResponse
import io

app = FastAPI()

# 🌍 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🤖 AI CLIENT
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# 🗄️ DATABASE
DATABASE_URL = os.getenv("DATABASE_URL")

if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)


def get_conn():
    return psycopg2.connect(DATABASE_URL, sslmode="require")


# 📦 MODELS
class LearnRequest(BaseModel):
    student_name: str
    student_class: str
    subject: str
    chapter: str


class ChatInput(BaseModel):
    message: str


# 🚀 HOME
@app.get("/")
def home():
    return {"status": "ExamPanic AI running 🚀"}


# 📚 LEARN API
@app.post("/learn")
def learn(data: LearnRequest):

    prompt = f"""
You are an ICSE expert teacher.

Format:
📘 Title
📌 Definition
📖 Explanation
🧠 Examples
⭐ Key Points
📝 Questions

Class: {data.student_class}
Subject: {data.subject}
Chapter: {data.chapter}
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "system", "content": prompt}],
        max_tokens=2000
    )

    return {"lesson": response.choices[0].message.content}


# 💬 CHAT API
@app.post("/chat")
def chat(data: ChatInput):

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You are a helpful ICSE study assistant."},
            {"role": "user", "content": data.message}
        ],
        max_tokens=1000
    )

    return {"reply": response.choices[0].message.content}


# 📘 NOTES GENERATOR (1-PAGE)
@app.post("/generate-notes")
def notes(data: LearnRequest):

    prompt = f"""
Create 1-page ICSE revision notes:

📘 Topic
📌 Definition
📖 Short Explanation
🧠 Key Points
⚡ Formulas (if any)
📝 Exam Tips

Class: {data.student_class}
Subject: {data.subject}
Chapter: {data.chapter}

VERY SHORT AND EXAM READY
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "system", "content": prompt}],
        max_tokens=1200
    )

    return {"notes": response.choices[0].message.content}


# 📥 DOWNLOAD NOTES (NEW 🔥)
@app.post("/download-notes")
def download_notes(data: LearnRequest):

    prompt = f"""
Make clean exam revision notes:

Topic: {data.chapter}
Subject: {data.subject}
Class: {data.student_class}

Format:
- simple headings
- short points
- exam ready
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "system", "content": prompt}],
        max_tokens=1200
    )

    text = response.choices[0].message.content

    buffer = io.StringIO()
    buffer.write(text)
    buffer.seek(0)

    return StreamingResponse(
        buffer,
        media_type="text/plain",
        headers={
            "Content-Disposition": "attachment; filename=notes.txt"
        }
    )


# 📚 SAVE CHAPTER
@app.post("/save-chapter")
def save(data: LearnRequest):

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


# 🏆 LEADERBOARD
@app.get("/leaderboard")
def leaderboard():

    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
        SELECT student_name, COUNT(*)
        FROM chapters
        GROUP BY student_name
        ORDER BY COUNT(*) DESC
        LIMIT 10
    """)

    rows = cur.fetchall()

    cur.close()
    conn.close()

    return {
        "leaderboard": [
            {"name": r[0], "xp": r[1] * 10}
            for r in rows
        ]
    }


# ⚡ QUICK ACCESS
@app.get("/quick-access")
def quick():

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


# 😂 MEME
@app.post("/meme")
def meme(data: ChatInput):

    return {
        "meme": f"When you study {data.message} but brain says ‘sleep mode 😴’"
    }


# 🔥 SHARE XP
@app.post("/share")
def share():

    return {
        "bonus_xp": 5,
        "message": "Shared successfully 🚀"
    }
