
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


# ================= LEARN (SMART TUTOR ENGINE) =================
@app.post("/learn")
def learn(data: LearnRequest):

    level = data.student_class

    if level in ["1","2","3","4","5"]:
        instruction = "Teach like a fun story for a child with imagination."
    elif level in ["6","7","8"]:
        instruction = "Teach with concepts + real life examples."
    else:
        instruction = "Teach like ICSE expert teacher with deep clarity."

    lang_rule = "Write ONLY in Hindi" if data.language.lower() == "hindi" else "Write ONLY in English"

    prompt = f"""
You are a MASTER AI TUTOR (like top coaching institute teacher).

GOAL:
Build deep understanding, not memorization.

{instruction}

{lang_rule}

🚀 STRICT 4 STEP FLOW:
1. Concept (very clear foundation)
2. Real Life Example
3. Thinking Question (student must answer)
4. Revision (exam ready points)

RULES:
- If student asks "more examples" → ONLY examples
- If student asks "next topic" → ONLY next concept
- Always end with a question
- Never repeat full lesson again

FORMAT:
📘 Title
📌 Concept
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
        messages=[{"role": "system", "content": prompt}],
        max_tokens=2000
    )

    return {
        "status": "success",
        "lesson": res.choices[0].message.content
    }


# ================= CHAT (SMART ICSE TUTOR) =================
@app.post("/chat")
def chat(data: ChatInput):

    prompt = f"""
You are ExamPanic AI ICSE tutor.

RULES:
- Step by step explanation
- Simple language
- Real life examples
- End with small question

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
        messages=[{"role": "system", "content": prompt}],
        max_tokens=5000
    )

    return {"status": "success", "reply": res.choices[0].message.content}


# ================= MCQ GENERATION =================
@app.post("/mock-test-generate")
def generate_test(data: MockRequest):

    prompt = f"""
Generate 5 ICSE MCQs STRICT JSON ONLY.

[
  {{
    "question": "...",
    "options": ["A","B","C","D"],
    "answer": "A"
  }}
]

Rules:
- Mix easy + medium level
- Only one correct answer

Class: {data.student_class}
Subject: {data.subject}
Chapter: {data.chapter}
"""

    res = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "system", "content": prompt}],
        max_tokens=1500
    )

    content = res.choices[0].message.content.replace("```json", "").replace("```", "").strip()

    try:
        return {
            "status": "success",
            "questions": json.loads(content)
        }
    except:
        return {
            "status": "error",
            "questions": [],
            "raw": content
        }


# ================= MCQ EVALUATION =================
@app.post("/mock-test-evaluate")
def evaluate(data: MockSubmit):

    if not data.questions:
        return {"score": 0, "correct": 0, "total": 0, "performance": "No Questions"}

    correct = 0
    total = len(data.questions)

    for i in range(total):
        if i < len(data.answers):
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


# ================= EXAM PREDICTION =================
@app.post("/predict-exam")
def predict(data: PredictorInput):

    if not model:
        return {"status": "error", "message": "Model not loaded"}

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


# ================= NOTES (SMART STUDY ENGINE) =================
@app.post("/generate-notes")
def notes(data: LearnRequest):

    level = data.student_class

    if level in ["1","2","3","4","5"]:
        instruction = "Use storytelling + very simple explanation."
    elif level in ["6","7","8"]:
        instruction = "Use concept + examples + revision format."
    else:
        instruction = "Use advanced ICSE exam notes."

    lang_rule = "Write ONLY in Hindi" if data.language.lower() == "hindi" else "Write ONLY in English"

    prompt = f"""
You are ICSE NOTES AI.

GOAL:
Create exam-ready revision notes.

{instruction}

{lang_rule}

FORMAT:
📘 Title
📌 Definition
📖 Explanation
⭐ Important Points
🧠 Memory Tricks
📝 Questions

Rules:
- Clean structured notes
- No extra fluff
- Exam focused

Class: {data.student_class}
Subject: {data.subject}
Chapter: {data.chapter}
"""

    res = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "system", "content": prompt}],
        max_tokens=1200
    )

    return {
        "status": "success",
        "notes": res.choices[0].message.content
    }
