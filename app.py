from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
import os
import json
import numpy as np
import joblib

# ================= APP =================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================= AI =================
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ================= MODEL =================
model = None
try:
    model = joblib.load("exam_predictor.pkl")
    print("✅ Model loaded")
except:
    print("⚠ Model not loaded")

# ================= MODELS =================
class LearnRequest(BaseModel):
    student_name: str
    student_class: str
    subject: str
    chapter: str
    language: str = "english"

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

# ================= HOME =================
@app.get("/")
def home():
    return {"status": "ExamPanic AI Running 🚀"}

# ================= LESSON (🔥 PW 4-STEP + HINDI/ENGLISH) =================
@app.post("/learn")
def learn(data: LearnRequest):

    level = data.student_class

    # ================= CLASS LEVEL =================
    if level in ["1","2","3","4","5"]:
        instruction = "Teach like a story for kids with very simple words."

    elif level in ["6","7","8"]:
        instruction = "Teach with simple concept + examples."

    else:
        instruction = "Teach in exam-oriented ICSE advanced format."

    # ================= LANGUAGE MODE =================
    if data.language.lower() == "hindi":

        language_instruction = """
Explain fully in simple Hindi language.
Use Hindi examples.
Use Hindi explanations.
Keep language easy for students.
"""

    else:

        language_instruction = """
Explain fully in English language.
Use simple English.
Keep explanation student friendly.
"""

    prompt = f"""
You are an ICSE expert AI teacher.

RULE:
{instruction}

LANGUAGE RULE:
{language_instruction}

🚀 TEACH IN 4 STEPS ONLY:

STEP 1: Concept Explanation
STEP 2: Simple Real Life Example
STEP 3: Ask Student Question (check understanding)
STEP 4: Quick Revision Summary

IMPORTANT:
- Interactive tone
- After STEP 3 ask:
"Do you want next topic or more examples?"
- Never repeat full lesson again for small requests

FORMAT:
📘 Title
📌 Definition
📖 Explanation
🧠 Example
❓ Question
⭐ Revision

Class: {data.student_class}
Subject: {data.subject}
Chapter: {data.chapter}
"""

    res = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role":"system","content":prompt}],
        max_tokens=2000
    )

    return {
        "status": "success",
        "lesson": res.choices[0].message.content
    }

# ================= CHAT (UPGRADED ICSE TUTOR) =================
@app.post("/chat")
def chat(data: ChatInput):

    prompt = f"""
You are ExamPanic AI, a friendly ICSE tutor.

RULES:
- Explain step by step
- Use simple language
- Give examples
- Make it interactive
- End with a small question

FORMAT:
📌 Explanation
🧠 Example
⭐ Key Point
❓ Question

User Query:
{data.message}
"""

    res = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role":"system","content":prompt}],
        max_tokens=1200
    )

    return {
        "status": "success",
        "reply": res.choices[0].message.content
    }

# ================= MCQ GENERATE (UPGRADED) =================
@app.post("/mock-test-generate")
def generate_test(data: MockRequest):

    prompt = f"""
Generate 5 MCQs STRICT JSON ONLY.

RULES:
- ICSE exam level
- Mix easy + medium questions
- Only one correct answer

FORMAT:
[
  {{
    "question": "...",
    "options": ["A","B","C","D"],
    "answer": "A"
  }}
]

Class: {data.student_class}
Subject: {data.subject}
Chapter: {data.chapter}
"""

    res = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role":"system","content":prompt}],
        max_tokens=1500
    )

    content = res.choices[0].message.content.replace("```json","").replace("```","").strip()

    try:
        questions = json.loads(content)

        return {
            "status": "success",
            "questions": questions
        }

    except:
        return {
            "status": "error",
            "questions": [],
            "raw": content
        }

# ================= MCQ EVALUATE =================
@app.post("/mock-test-evaluate")
def evaluate(data: MockSubmit):

    if not data.questions:
        return {
            "score": 0,
            "correct": 0,
            "total": 0,
            "performance": "No Questions Found"
        }

    correct = 0
    total = len(data.questions)

    for i in range(total):
        if i >= len(data.answers):
            continue

        if str(data.answers[i]).strip().upper() == str(data.questions[i]["answer"]).strip().upper():
            correct += 1

    score = round((correct / total) * 100, 2)

    return {
        "score": score,
        "correct": correct,
        "total": total,
        "performance":
            "Excellent 🚀" if score >= 85 else
            "Good 👍" if score >= 60 else
            "Needs Improvement 📚" if score >= 40 else
            "Critical ⚠"
    }

# ================= PREDICT =================
@app.post("/predict-exam")
def predict(data: PredictorInput):

    if not model:
        return {"status":"error","message":"Model not loaded"}

    features = np.array([[
        data.chapters_done,
        data.revision_count,
        data.attendance,
        data.test_score
    ]])

    prediction = round(float(model.predict(features)[0]), 2)

    return {
        "predicted_marks": prediction,
        "performance":
            "Excellent 🚀" if prediction >= 85 else
            "Good 👍" if prediction >= 60 else
            "Needs Improvement 📚" if prediction >= 40 else
            "Critical ⚠"
    }

# ================= NOTES (CLASS BASED SMART) =================
@app.post("/generate-notes")
def notes(data: LearnRequest):

    level = data.student_class

    if level in ["1","2","3","4","5"]:
        instruction = "Use very simple language with storytelling."

    elif level in ["6","7","8"]:
        instruction = "Use medium level explanation with examples."

    else:
        instruction = "Use advanced exam-oriented ICSE notes."

    prompt = f"""
You are an ICSE teacher.

Generate HIGH QUALITY NOTES.

RULE:
{instruction}

FORMAT:
📘 Title
📌 Definition
📖 Explanation
⭐ Important Points
🧠 Memory Tricks
📝 Questions

Class: {data.student_class}
Subject: {data.subject}
Chapter: {data.chapter}
"""

    res = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role":"system","content":prompt}],
        max_tokens=1200
    )

    return {
        "status": "success",
        "notes": res.choices[0].message.content
    }
    
