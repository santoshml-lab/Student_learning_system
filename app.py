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

# ================= APP =================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================= AI CLIENT =================
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ================= DATABASE =================
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

# ================= ML MODEL =================
model = None

try:
    model = joblib.load("exam_predictor.pkl")
    print("✅ ML Model Loaded")

except Exception as e:
    print("❌ Model Error:", e)

# ================= MODELS =================

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

class MockRequest(BaseModel):
    student_class: str
    subject: str
    chapter: str

class MockSubmit(BaseModel):
    questions: list
    answers: list
    chapters_done: int
    revision_count: int
    attendance: int

# ================= HOME =================
@app.get("/")
def home():

    return {
        "status": "ExamPanic AI Running 🚀"
    }

# ================= AI LESSON =================
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

    res = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role":"system",
                "content":prompt
            }
        ],
        max_tokens=2000
    )

    return {
        "lesson": res.choices[0].message.content
    }

# ================= CHAT =================
@app.post("/chat")
def chat(data: ChatInput):

    res = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role":"system",
                "content":"You are helpful ICSE tutor."
            },
            {
                "role":"user",
                "content":data.message
            }
        ],
        max_tokens=1000
    )

    return {
        "reply": res.choices[0].message.content
    }

# ================= NOTES =================
@app.post("/generate-notes")
def notes(data: LearnRequest):

    prompt = f"""
Create exam-ready revision notes.

Class: {data.student_class}
Subject: {data.subject}
Chapter: {data.chapter}
"""

    res = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role":"system",
                "content":prompt
            }
        ],
        max_tokens=1200
    )

    return {
        "notes": res.choices[0].message.content
    }

# ================= MOCK TEST GENERATE =================
@app.post("/mock-test-generate")
def mock(data: MockRequest):

    prompt = f"""
Generate exactly 5 MCQs in STRICT JSON.

Return ONLY JSON.

Format:
[
 {{
   "question":"What is...?",
   "options":["A","B","C","D"],
   "answer":"A"
 }}
]

Class: {data.student_class}
Subject: {data.subject}
Chapter: {data.chapter}
"""

    res = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role":"system",
                "content":prompt
            }
        ],
        max_tokens=1500
    )

    content = res.choices[0].message.content.strip()

    content = content.replace("```json","")
    content = content.replace("```","")
    content = content.strip()

    try:

        parsed = json.loads(content)

        return {
            "status":"success",
            "questions": parsed
        }

    except Exception as e:

        return {
            "status":"error",
            "message":"Invalid JSON from AI",
            "raw": content
        }

# ================= MOCK TEST EVALUATE =================
@app.post("/mock-test-evaluate")
def evaluate(data: MockSubmit):

    correct = 0

    for i in range(len(data.questions)):

        selected = data.answers[i].strip()

        correct_answer = data.questions[i]["answer"].strip()

        if selected.startswith(correct_answer):
            correct += 1

    score = round(
        (correct / len(data.questions)) * 100,
        2
    )

    prediction = score

    if model:

        features = np.array([[
            data.chapters_done,
            data.revision_count,
            data.attendance,
            score
        ]])

        prediction = round(
            float(model.predict(features)[0]),
            2
        )

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
        "score": score,
        "predicted_marks": prediction,
        "performance": performance
    }

# ================= PREDICT EXAM =================
@app.post("/predict-exam")
def predict(data: PredictorInput):

    if not model:

        return {
            "status":"error",
            "message":"ML model not loaded"
        }

    features = np.array([[
        data.chapters_done,
        data.revision_count,
        data.attendance,
        data.test_score
    ]])

    prediction = round(
        float(model.predict(features)[0]),
        2
    )

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
        "status":"success"
    }

# ================= SAVE CHAPTER =================
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
        "message":"saved"
    }

# ================= LEADERBOARD =================
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
        "leaderboard":[
            {
                "name":r[0],
                "xp":r[1]*10
            }
            for r in rows
        ]
    }

# ================= QUICK ACCESS =================
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
        "quick_access":[
            {
                "subject":r[0],
                "chapter":r[1]
            }
            for r in rows
        ]
    }

# ================= MEME =================
@app.post("/meme")
def meme(data: ChatInput):

    return {
        "meme":
        f"When you study {data.message} but brain says sleep 😴"
    }

# ================= SHARE =================
@app.post("/share")
def share():

    return {
        "bonus_xp":5,
        "message":"Shared Successfully 🚀"
    }


    
    
