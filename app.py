
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


# ================= LESSON =================
@app.post("/learn")
def learn(data: LearnRequest):

    level = data.student_class

    if level in ["1","2","3","4","5"]:
        instruction = "Teach like a story for kids."
    elif level in ["6","7","8"]:
        instruction = "Teach with simple concepts and examples."
    else:
        instruction = "Teach in advanced ICSE exam format."

    lang_rule = "Write in Hindi" if data.language.lower() == "hindi" else "Write in English"

    prompt = f"""
You are an ICSE expert teacher AI.

RULE:
{instruction}

LANGUAGE:
{lang_rule}

🚀 TEACH IN 4 STEPS ONLY:
1. Concept
2. Example
3. Question
4. Revision

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


# ================= CHAT =================
@app.post("/chat")
def chat(data: ChatInput):

    prompt = f"""
You are ICSE tutor.

Explain step by step with example.

User: {data.message}
"""

    res = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role":"system","content":prompt}],
        max_tokens=1000
    )

    return {"reply": res.choices[0].message.content}


# ================= MCQ =================
@app.post("/mock-test-generate")
def generate_test(data: MockRequest):

    prompt = f"""
Generate 5 MCQs JSON ONLY.

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


# ================= MCQ EVALUATE =================
@app.post("/mock-test-evaluate")
def evaluate(data: MockSubmit):

    if not data.questions:
        return {"score":0,"correct":0,"total":0,"performance":"No Questions"}

    correct = 0
    total = len(data.questions)

    for i in range(total):
        if i < len(data.answers):
            if str(data.answers[i]).upper() == str(data.questions[i]["answer"]).upper():
                correct += 1

    score = round((correct/total)*100,2)

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

    prediction = round(float(model.predict(features)[0]),2)

    return {
        "predicted_marks": prediction,
        "performance":
            "Excellent 🚀" if prediction >= 85 else
            "Good 👍" if prediction >= 60 else
            "Needs Improvement 📚" if prediction >= 40 else
            "Critical ⚠"
    }


# ================= NOTES =================
@app.post("/generate-notes")
def notes(data: LearnRequest):

    level = data.student_class

    if level in ["1","2","3","4","5"]:
        instruction = "Use simple storytelling language."
    elif level in ["6","7","8"]:
        instruction = "Use medium explanation with examples."
    else:
        instruction = "Use advanced ICSE notes."

    lang_rule = "Write in Hindi" if data.language.lower() == "hindi" else "Write in English"

    prompt = f"""
You are ICSE teacher.

RULE:
{instruction}

LANGUAGE:
{lang_rule}

Generate notes in exam format.

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
