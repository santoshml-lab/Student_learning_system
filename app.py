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
    DATABASE_URL = DATABASE_URL.replace(
        "postgres://",
        "postgresql://",
        1
    )

def get_conn():
    return psycopg2.connect(
        DATABASE_URL,
        sslmode="require"
    )

# 🤖 LOAD ML MODEL
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
    return {
        "status": "ExamPanic AI running 🚀"
    }

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
        messages=[
            {
                "role": "system",
                "content": prompt
            }
        ],
        max_tokens=2000
    )

    return {
        "lesson": response.choices[0].message.content
    }

# 💬 CHAT API
@app.post("/chat")
def chat(data: ChatInput):

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content":
                "You are a helpful ICSE study assistant."
            },
            {
                "role": "user",
                "content": data.message
            }
        ],
        max_tokens=1000
    )

    return {
        "reply": response.choices[0].message.content
    }

# 📘 NOTES GENERATOR
@app.post("/generate-notes")
def notes(data: LearnRequest):

    prompt = f"""
Create 1-page ICSE revision notes:

📘 Topic
📌 Definition
📖 Short Explanation
🧠 Key Points
⚡ Formulas
📝 Exam Tips

Class: {data.student_class}
Subject: {data.subject}
Chapter: {data.chapter}

VERY SHORT AND EXAM READY
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": prompt
            }
        ],
        max_tokens=1200
    )

    return {
        "notes": response.choices[0].message.content
    }

# 📥 DOWNLOAD NOTES
@app.post("/download-notes")
def download_notes(data: LearnRequest):

    prompt = f"""
Make clean ICSE revision notes:

Topic: {data.chapter}
Subject: {data.subject}
Class: {data.student_class}
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": prompt
            }
        ],
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
            "Content-Disposition":
            "attachment; filename=notes.txt"
        }
    )

# 🧠 CONFUSION MODE
@app.post("/confusion-mode")
def confusion(data: LearnRequest):

    prompt = f"""
Explain like weak student:

Topic: {data.chapter}
Subject: {data.subject}

Rules:
- very simple English
- step by step
- real life examples
"""

    res = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": prompt
            }
        ],
        max_tokens=1200
    )

    return {
        "result": res.choices[0].message.content
    }

# ⚡ QUICK REVISION
@app.post("/quick-revision")
def quick_revision(data: LearnRequest):

    prompt = f"""
Give ultra short revision:

Topic: {data.chapter}

Rules:
- only key points
- formulas
- exam tips
"""

    res = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": prompt
            }
        ],
        max_tokens=800
    )

    return {
        "result": res.choices[0].message.content
    }

# ❓ MCQ TEST
@app.post("/mcq-test")
def mcq(data: LearnRequest):

    prompt = f"""
Generate 10 MCQs:

Topic: {data.chapter}

Include:
- question
- 4 options
- correct answer at end
"""

    res = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": prompt
            }
        ],
        max_tokens=1200
    )

    return {
        "result": res.choices[0].message.content
    }

# 📚 SAVE CHAPTER
@app.post("/save-chapter")
def save(data: LearnRequest):

    conn = get_conn()
    cur = conn.cursor()

    cur.execute(
        """
        INSERT INTO chapters
        (student_name, student_class, subject, chapter)
        VALUES (%s,%s,%s,%s)
        """,
        (
            data.student_name,
            data.student_class,
            data.subject,
            data.chapter
        )
    )

    conn.commit()

    cur.close()
    conn.close()

    return {
        "message": "saved"
    }

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
            {
                "name": r[0],
                "xp": r[1] * 10
            }
            for r in rows
        ]
    }

# ⚡ QUICK ACCESS
@app.get("/quick-access")
def quick():

    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
        SELECT subject, chapter
        FROM chapters
        LIMIT 10
    """)

    rows = cur.fetchall()

    cur.close()
    conn.close()

    return {
        "quick_access": [
            {
                "subject": r[0],
                "chapter": r[1]
            }
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

    total_chapters = sum([r[1] for r in rows])

    strong_subject = "None"
    weak_subject = "None"

    if rows:
        strong_subject = max(
            rows,
            key=lambda x: x[1]
        )[0]

        weak_subject = min(
            rows,
            key=lambda x: x[1]
        )[0]

    total_xp = total_chapters * 10

    streak = min(total_chapters, 30)

    cur.close()
    conn.close()

    return {
        "student": student_name,
        "total_chapters": total_chapters,
        "total_xp": total_xp,
        "strong_subject": strong_subject,
        "weak_subject": weak_subject,
        "streak": streak,
        "status": "success"
    }

# 🎯 EXAM PREDICTOR ML
@app.post("/predict-exam")
def predict_exam(data: PredictorInput):

    features = np.array([[
        data.chapters_done,
        data.revision_count,
        data.attendance,
        data.test_score
    ]])

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

# 🏅 CERTIFICATE SYSTEM
@app.post("/generate-certificate")
def certificate(data: CertificateInput):

    certificate_text = f"""
🏅 ExamPanic AI Certificate

This certificate is proudly awarded to

🎓 {data.student_name}

Class: {data.student_class}

For successfully completing:

📚 {data.course_name}

Keep Learning & Keep Growing 🚀
"""

    buffer = io.StringIO()
    buffer.write(certificate_text)
    buffer.seek(0)

    return StreamingResponse(
        buffer,
        media_type="text/plain",
        headers={
            "Content-Disposition":
            "attachment; filename=certificate.txt"
        }
    )

# 😂 MEME
@app.post("/meme")
def meme(data: ChatInput):

    return {
        "meme":
        f"When you study {data.message} but brain says sleep 😴"
    }

# 🔥 SHARE XP
@app.post("/share")
def share():

    return {
        "bonus_xp": 5,
        "message": "Shared successfully 🚀"
    }
    
