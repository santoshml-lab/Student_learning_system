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

# ================= DB =================
DATABASE_URL = os.getenv("DATABASE_URL")

if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

def get_conn():
    return psycopg2.connect(DATABASE_URL, sslmode="require")

# ================= MODEL =================
model = None
try:
    model = joblib.load("exam_predictor.pkl")
    print("✅ Model loaded")
except:
    print("❌ Model not found")

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
    return {"status": "ExamPanic AI Running 🚀"}

# ================= AI LEARN =================
@app.post("/learn")
def learn(data: LearnRequest):

    prompt = f"""
ICSE Expert Teacher Format:

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

# ================= CHAT =================
@app.post("/chat")
def chat(data: ChatInput):

    res = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You are ICSE tutor."},
            {"role": "user", "content": data.message}
        ],
        max_tokens=1000
    )

    return {"reply": res.choices[0].message.content}

# ================= NOTES =================
@app.post("/generate-notes")
def notes(data: LearnRequest):

    prompt = f"""
Make exam-ready notes:

Class: {data.student_class}
Subject: {data.subject}
Chapter: {data.chapter}
"""

    res = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "system", "content": prompt}],
        max_tokens=1200
    )

    return {"notes": res.choices[0].message.content}

# ================= MOCK TEST GENERATE =================
@app.post("/mock-test-generate")
def mock(data: MockRequest):

    prompt = f"""
Generate 5 MCQs STRICT JSON:

[
 {{"question":"","options":["A","B","C","D"],"answer":"A"}}
]

Class: {data.student_class}
Subject: {data.subject}
Chapter: {data.chapter}
"""

    res = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "system", "content": prompt}],
        max_tokens=1500
    )

    content = res.choices[0].message.content.replace("```json","").replace("```","")

    return {"test": content}

# ================= MOCK TEST EVALUATE =================
@app.post("/mock-test-evaluate")
def evaluate(data: MockSubmit):

    correct = 0

    for i in range(len(data.questions)):
        if data.answers[i] == data.questions[i]["answer"]:
            correct += 1

    score = (correct / len(data.questions)) * 100

    if model:
        features = np.array([[data.chapters_done, data.revision_count, data.attendance, score]])
        prediction = round(float(model.predict(features)[0]), 2)
    else:
        prediction = score

    performance = (
        "Excellent 🚀" if prediction >= 85 else
        "Good 👍" if prediction >= 60 else
        "Needs Improvement 📚" if prediction >= 40 else
        "Critical ⚠"
    )

    return {
        "score": score,
        "predicted_marks": prediction,
        "performance": performance
    }

# ================= PREDICT =================
@app.post("/predict-exam")
def predict(data: PredictorInput):

    if not model:
        return {"status":"error","message":"Model not loaded"}

    features = np.array([[data.chapters_done, data.revision_count, data.attendance, data.test_score]])

    prediction = round(float(model.predict(features)[0]), 2)

    performance = (
        "Excellent 🚀" if prediction >= 85 else
        "Good 👍" if prediction >= 60 else
        "Needs Improvement 📚" if prediction >= 40 else
        "Critical ⚠"
    )

    return {
        "predicted_marks": prediction,
        "performance": performance,
        "status": "success"
    }

# ================= SAVE =================
@app.post("/save-chapter")
def save(data: LearnRequest):

    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO chapters (student_name, student_class, subject, chapter)
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

    return {
        "leaderboard": [
            {"name": r[0], "xp": r[1]*10}
            for r in rows
        ]
    }

# ================= QUICK ACCESS =================
@app.get("/quick-access")
def quick():

    conn = get_conn()
    cur = conn.cursor()

    cur.execute("SELECT subject, chapter FROM chapters LIMIT 10")
    rows = cur.fetchall()

    return {
        "quick_access": [
            {"subject": r[0], "chapter": r[1]}
            for r in rows
        ]
    }

# ================= MEME =================
@app.post("/meme")
def meme(data: ChatInput):
    return {"meme": f"When you study {data.message} but brain sleeps 😴"}

# ================= SHARE =================
@app.post("/share")
def share():
    return {"bonus_xp": 5, "message": "Shared 🚀"}


    
    
