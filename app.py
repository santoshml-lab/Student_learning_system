from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
from fastapi.responses import StreamingResponse
import psycopg2
import joblib
import numpy as np
import os
import io
import json

app = FastAPI()

# 🌍 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🤖 GROQ CLIENT
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# 🗄 DATABASE
DATABASE_URL = os.getenv("DATABASE_URL")

if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

def get_conn():
    return psycopg2.connect(DATABASE_URL, sslmode="require")

# 🤖 ML MODEL
model = joblib.load("exam_predictor.pkl")

# 📦 MODELS
class LearnRequest(BaseModel):
    student_name: str
    student_class: str
    subject: str
    chapter: str

class ChatInput(BaseModel):
    message: str

class PredictorInput(BaseModel):
    chapters_done: int
    revision_count: int
    attendance: int
    test_score: int

class CertificateInput(BaseModel):
    student_name: str
    student_class: str
    course_name: str

# 🚀 HOME
@app.get("/")
def home():
    return {"status": "ExamPanic AI running 🚀"}

# 📚 LEARN
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
        messages=[{"role": "system", "content": prompt}],
        max_tokens=2000
    )

    return {"lesson": res.choices[0].message.content}

# 💬 CHAT
@app.post("/chat")
def chat(data: ChatInput):

    res = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You are a helpful ICSE assistant."},
            {"role": "user", "content": data.message}
        ],
        max_tokens=1000
    )

    return {"reply": res.choices[0].message.content}

# 📘 NOTES
@app.post("/generate-notes")
def notes(data: LearnRequest):

    prompt = f"""
Create ICSE revision notes:

Topic: {data.chapter}
Subject: {data.subject}
Class: {data.student_class}
"""

    res = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "system", "content": prompt}],
        max_tokens=1200
    )

    return {"notes": res.choices[0].message.content}

# 📥 DOWNLOAD
@app.post("/download-notes")
def download_notes(data: LearnRequest):

    res = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{
            "role": "system",
            "content": f"Make notes for {data.chapter}"
        }],
        max_tokens=1200
    )

    buffer = io.StringIO()
    buffer.write(res.choices[0].message.content)
    buffer.seek(0)

    return StreamingResponse(
        buffer,
        media_type="text/plain",
        headers={"Content-Disposition": "attachment; filename=notes.txt"}
    )

# 🧠 CONFUSION
@app.post("/confusion-mode")
def confusion(data: LearnRequest):

    res = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{
            "role": "system",
            "content": f"Explain simply: {data.chapter}"
        }],
        max_tokens=1200
    )

    return {"result": res.choices[0].message.content}

# ⚡ QUICK REVISION
@app.post("/quick-revision")
def quick_revision(data: LearnRequest):

    res = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{
            "role": "system",
            "content": f"Ultra short revision: {data.chapter}"
        }],
        max_tokens=800
    )

    return {"result": res.choices[0].message.content}

# ❓ MCQ TEST (FIXED 🔥)
@app.post("/mcq-test")
def mcq(data: LearnRequest):

    prompt = f"""
Generate 5 MCQs in STRICT JSON format only.

Return ONLY JSON array.

Topic: {data.chapter}
"""

    try:

        res = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "system", "content": prompt}],
            max_tokens=1200
        )

        content = res.choices[0].message.content.strip()

        content = content.replace("```json", "")
        content = content.replace("```", "")
        content = content.strip()

        mcqs = json.loads(content)

        return {
            "status": "success",
            "result": mcqs
        }

    except Exception as e:

        return {
            "status": "error",
            "message": str(e)
        }

# 📚 SAVE CHAPTER
@app.post("/save-chapter")
def save(data: LearnRequest):

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

# 📊 ANALYTICS
@app.get("/analytics/{student_name}")
def analytics(student_name: str):

    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
        SELECT subject, COUNT(*)
        FROM chapters
        WHERE student_name = %s
        GROUP BY subject
    """, (student_name,))

    rows = cur.fetchall()

    total = sum([r[1] for r in rows])

    strong = max(rows, key=lambda x: x[1])[0] if rows else "None"
    weak = min(rows, key=lambda x: x[1])[0] if rows else "None"

    return {
        "student": student_name,
        "total_chapters": total,
        "total_xp": total * 10,
        "strong_subject": strong,
        "weak_subject": weak,
        "status": "success"
    }

# 🎯 PREDICT EXAM
@app.post("/predict-exam")
def predict(data: PredictorInput):

    features = np.array([[data.chapters_done, data.revision_count, data.attendance, data.test_score]])

    prediction = model.predict(features)[0]
    prediction = round(prediction, 2)

    performance = "Average"
    if prediction >= 85:
        performance = "Excellent 🚀"
    elif prediction >= 60:
        performance = "Good 👍"
    elif prediction >= 40:
        performance = "Needs Improvement 📚"
    else:
        performance = "Critical ⚠"

    return {
        "predicted_marks": prediction,
        "performance": performance,
        "status": "success"
    }

# 🏅 CERTIFICATE
@app.post("/generate-certificate")
def certificate(data: CertificateInput):

    text = f"""
🏅 ExamPanic AI Certificate

Name: {data.student_name}
Class: {data.student_class}
Course: {data.course_name}

Completed Successfully 🚀
"""

    buffer = io.StringIO()
    buffer.write(text)
    buffer.seek(0)

    return StreamingResponse(
        buffer,
        media_type="text/plain",
        headers={"Content-Disposition": "attachment; filename=certificate.txt"}
    )

# 😂 MEME
@app.post("/meme")
def meme(data: ChatInput):

    return {
        "meme": f"When you study {data.message} but sleep hits 😴"
    }

# 🔥 SHARE
@app.post("/share")
def share():

    return {
        "bonus_xp": 5,
        "message": "Shared successfully 🚀"
    }
    
