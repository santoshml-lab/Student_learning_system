from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
import os
import psycopg2
import io
from fastapi.responses import StreamingResponse, HTMLResponse
from datetime import datetime

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

    res = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1500
    )

    return {"lesson": res.choices[0].message.content}


# 💬 CHAT API
@app.post("/chat")
def chat(data: ChatInput):

    res = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You are ICSE study assistant."},
            {"role": "user", "content": data.message}
        ],
        max_tokens=1000
    )

    return {"reply": res.choices[0].message.content}


# 🧠 NOTES
@app.post("/generate-notes")
def notes(data: LearnRequest):

    prompt = f"""
Create 1-page ICSE revision notes:
Subject: {data.subject}
Chapter: {data.chapter}
"""

    res = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1200
    )

    return {"notes": res.choices[0].message.content}


# 📥 DOWNLOAD NOTES
@app.post("/download-notes")
def download_notes(data: LearnRequest):

    prompt = f"""
Make exam notes:
Subject: {data.subject}
Chapter: {data.chapter}
"""

    res = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1200
    )

    text = res.choices[0].message.content

    buffer = io.StringIO()
    buffer.write(text)
    buffer.seek(0)

    return StreamingResponse(
        buffer,
        media_type="text/plain",
        headers={"Content-Disposition": "attachment; filename=notes.txt"}
    )


# 🧠 CONFUSION MODE
@app.post("/confusion-mode")
def confusion(data: LearnRequest):

    prompt = f"""
Explain simply:
Topic: {data.chapter}
Subject: {data.subject}
"""

    res = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1200
    )

    return {"result": res.choices[0].message.content}


# ⚡ QUICK REVISION
@app.post("/quick-revision")
def quick_revision(data: LearnRequest):

    prompt = f"""
Ultra short revision:
Topic: {data.chapter}
"""

    res = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=800
    )

    return {"result": res.choices[0].message.content}


# ❓ MCQ TEST
@app.post("/mcq-test")
def mcq(data: LearnRequest):

    prompt = f"""
Generate 10 MCQs:
Topic: {data.chapter}
"""

    res = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1200
    )

    return {"result": res.choices[0].message.content}


# 📊 ANALYTICS
@app.get("/analytics/{student_name}")
def analytics(student_name: str):

    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
        SELECT subject, COUNT(*)
        FROM chapters
        WHERE student_name=%s
        GROUP BY subject
    """, (student_name,))

    rows = cur.fetchall()

    total = sum([r[1] for r in rows])

    strong = max(rows, key=lambda x: x[1])[0] if rows else "-"
    weak = min(rows, key=lambda x: x[1])[0] if rows else "-"

    return {
        "total_chapters": total,
        "total_xp": total * 10,
        "strong_subject": strong,
        "weak_subject": weak,
        "streak": min(total, 30)
    }


# 🎯 EXAM PREDICTOR (NEW)
@app.post("/predict")
def predict(data: dict):

    confidence = data.get("confidence", 5)
    mcq = data.get("mcq_score", 5)
    chapters = data.get("chapters_completed", 0)

    score = confidence*6 + mcq*4 + chapters*2
    score = min(score, 100)

    if score >= 80:
        risk = "LOW 🟢"
    elif score >= 50:
        risk = "MEDIUM 🟡"
    else:
        risk = "HIGH 🔴"

    return {
        "expected_score": score,
        "risk_level": risk,
        "weak_area": "Revise weak topics + MCQs"
    }


# 🎓 CERTIFICATE SYSTEM (NEW)
@app.post("/certificate")
def certificate(data: dict):

    name = data.get("student_name", "Student")
    subject = data.get("subject", "Study")
    score = data.get("score", 80)

    date = datetime.now().strftime("%d-%m-%Y")

    html = f"""
    <html>
    <body style="background:#0b1020;color:white;text-align:center;padding:50px;">
    <div style="border:2px solid #38bdf8;padding:30px;border-radius:20px;display:inline-block;">
        <h1 style="color:#38bdf8;">🎓 Certificate</h1>
        <h2>{name}</h2>
        <p>{subject}</p>
        <h3>{score}/100</h3>
        <p>Date: {date}</p>
        <br>
        <p>🚀 ExamPanic AI</p>
    </div>
    </body>
    </html>
    """

    return HTMLResponse(html)


# 📚 SAVE CHAPTER (optional)
@app.post("/save-chapter")
def save(data: LearnRequest):

    conn = get_conn()
    cur = conn.cursor()

    cur.execute(
        "INSERT INTO chapters VALUES (%s,%s,%s,%s)",
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

    return {
        "leaderboard": [
            {"name": r[0], "xp": r[1]*10}
            for r in rows
        ]
    }


# 😂 MEME
@app.post("/meme")
def meme(data: ChatInput):

    return {"meme": f"When you study {data.message} but brain says sleep 😴"}


# 🔥 SHARE
@app.post("/share")
def share():
    return {"bonus_xp": 5, "message": "Shared successfully 🚀"}
    
