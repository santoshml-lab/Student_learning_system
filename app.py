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


# ================= LEARN (FINAL ADVANCED ENGINE) =================
@app.post("/learn")
def learn(data: LearnRequest):

    level = data.student_class

    if level in ["1","2","3","4","5"]:
        instruction = """
Teach like a fun storytelling teacher for children.
Use imagination, simple words and playful examples.
"""
    elif level in ["6","7","8"]:
        instruction = """
Teach deeply with concepts, logic and real life examples.
Build strong understanding step by step.
"""
    else:
        instruction = """
Teach like an elite ICSE coaching teacher.
Give deep conceptual clarity with exam-oriented explanation.
"""

    lang_rule = (
        "Write ONLY in Hindi"
        if data.language.lower() == "hindi"
        else "Write ONLY in English"
    )

    prompt = f"""
You are ExamPanic AI,
a world-class ICSE AI Teacher.

MAIN GOAL:
Build DEEP UNDERSTANDING like a real textbook + expert teacher.

IMPORTANT:
- Never give short answers
- Explain everything step by step
- Teach like a classroom teacher
- Make student feel interactive learning
- Minimum 1200 words
- Use beautiful formatting
- Use headings and subheadings
- Add deep explanation
- Add examples
- Add revision
- Add thinking questions

TEACHING STYLE:
{instruction}

LANGUAGE:
{lang_rule}

STRICT STRUCTURE:

# 📘 Chapter Title

# 📌 Introduction
(Explain what this chapter is about)

# 📖 Core Concepts
(Deep explanation of all important concepts)

# 🧠 Real Life Examples
(Give practical examples)

# ⭐ Important Points
(Exam-oriented key points)

# 🔥 Memory Tricks
(Easy tricks for remembering)

# ❓ Thinking Questions
(Ask conceptual questions)

# 📝 Quick Revision
(Short revision notes)

# 🚀 Final Summary
(Complete chapter summary)

RULES:
- Keep explanation very detailed
- Avoid robotic answers
- Avoid short summaries
- Make content textbook quality
- Use ICSE level depth
- End with one student question

Class: {data.student_class}

Subject: {data.subject}

Chapter: {data.chapter}
"""

    res = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": prompt
            }
        ],
        max_tokens=4000
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


# ================= NOTES =================
# ================= NOTES (FINAL DEEP TEXTBOOK ENGINE) =================
@app.post("/generate-notes")
def notes(data: LearnRequest):

    level = data.student_class

    if level in ["1","2","3","4","5"]:

        instruction = """
Use storytelling style with very simple language.
Teach like a friendly school teacher.
Use imagination and fun examples.
"""

    elif level in ["6","7","8"]:

        instruction = """
Teach with deep concepts, logic and examples.
Build strong understanding step by step.
Explain every important topic clearly.
"""

    else:

        instruction = """
Create elite ICSE textbook-quality notes.
Use deep conceptual explanation with exam-oriented structure.
Teach like a top coaching institute teacher.
"""

    lang_rule = (

        "Write ONLY in Hindi"

        if data.language.lower() == "hindi"

        else "Write ONLY in English"

    )

    prompt = f"""
You are ExamPanic AI Notes Generator,
a world-class ICSE Notes Expert.

MAIN GOAL:
Create ULTRA DETAILED textbook-style notes
with deep conceptual understanding.

IMPORTANT:
- Never generate short notes
- Minimum 1500+ words
- Explain every important topic deeply
- Use beautiful formatting
- Use headings and subheadings
- Add examples
- Add conceptual clarity
- Add revision material
- Add exam-focused points
- Add memory tricks
- Add important questions
- Make student fully understand the chapter

TEACHING STYLE:
{instruction}

LANGUAGE:
{lang_rule}

STRICT STRUCTURE:

# 📘 Chapter Title

# 📌 Introduction
(Explain the chapter overview deeply)

# 📖 Core Concepts
(Explain all important concepts in detail)

# 🧠 Real Life Examples
(Give practical examples)

# ⭐ Important Exam Points
(Important ICSE exam-oriented points)

# 🔥 Memory Tricks
(Easy remembering tricks)

# 📚 Important Keywords / Formula
(Important terms, formulas or definitions)

# ❓ Important Questions
(Possible exam questions)

# 📝 Quick Revision Notes
(Fast revision section)

# 🚀 Final Summary
(Complete chapter conclusion)

RULES:
- Make notes highly detailed
- Avoid robotic writing
- Avoid tiny summaries
- Use textbook-level explanation
- Make content educational + engaging
- Use ICSE level depth
- Keep formatting beautiful
- Make notes revision-friendly

Class: {data.student_class}

Subject: {data.subject}

Chapter: {data.chapter}
"""

    res = client.chat.completions.create(

        model="llama-3.1-8b-instant",

        messages=[
            {
                "role": "system",
                "content": prompt
            }
        ],

        max_tokens=4000
    )

    return {

        "status": "success",

        "notes": res.choices[0].message.content

    }
    
# ================= NUMERICAL PROBLEM SOLVER =================

class NumericalRequest(BaseModel):
    student_class: str
    subject: str
    topic: str
    problem: str
    language: str = "english"


@app.post("/solve-numerical")
def solve_numerical(data: NumericalRequest):

    level = data.student_class

    if level in ["1","2","3","4","5"]:

        instruction = """
Explain in very simple language.
Use basic calculation steps.
Teach slowly like a school teacher.
"""

    elif level in ["6","7","8"]:

        instruction = """
Explain concepts deeply with formulas and logic.
Show step-by-step solving technique.
Teach like smart coaching teacher.
"""

    else:

        instruction = """
Teach like elite ICSE numerical expert.
Give full conceptual solving method,
formula derivation,
logic,
short tricks,
and exam-oriented solving techniques.
"""

    lang_rule = (

        "Write ONLY in Hindi"

        if data.language.lower() == "hindi"

        else "Write ONLY in English"

    )

    prompt = f"""
You are ExamPanic AI Numerical Solver,
a world-class Physics, Chemistry and Maths teacher.

MAIN GOAL:
Teach students HOW to solve numerical problems step by step.

IMPORTANT:
- Never give short answers
- Solve step by step
- Explain every formula
- Explain logic behind every step
- Teach calculation techniques
- Teach problem solving methods
- Make student understand deeply
- Add shortcuts and tricks
- Add exam tips
- Make explanation textbook quality

TEACHING STYLE:
{instruction}

LANGUAGE:
{lang_rule}

STRICT STRUCTURE:

# 📘 Problem Title

# 📌 Given Data
(List all given values)

# 📖 Concept Used
(Explain concept deeply)

# 🧠 Formula Used
(Explain formula and meaning)

# 🔥 Step-by-Step Solution
(Solve line by line)

# ⚡ Short Trick
(Fast solving trick)

# ⭐ Final Answer
(Highlight final answer clearly)

# ❓ Similar Practice Question
(Give one extra practice problem)

# 🚀 Exam Tips
(Important solving tips)

RULES:
- Explain deeply
- Avoid robotic answers
- Make solving method crystal clear
- Use proper units
- Keep formatting beautiful
- Use textbook-level explanation

Class: {data.student_class}

Subject: {data.subject}

Topic: {data.topic}

Problem:
{data.problem}
"""

    res = client.chat.completions.create(

        model="llama-3.1-8b-instant",

        messages=[
            {
                "role": "system",
                "content": prompt
            }
        ],

        max_tokens=4000
    )

    return {

        "status": "success",

        "solution": res.choices[0].message.content

    }
    # ================= WEAKNESS ANALYZER =================

class WeaknessInput(BaseModel):
    subject: str
    chapter: str
    score: float
    mistakes: list = []


@app.post("/analyze-weakness")
def analyze_weakness(data: WeaknessInput):

    prompt = f"""
You are ExamPanic AI Weakness Analyzer.

Analyze the student's performance.

Subject:
{data.subject}

Chapter:
{data.chapter}

Score:
{data.score}

Mistakes:
{data.mistakes}

Return STRICTLY in this format:

# 🧠 Weak Areas
(List weak concepts)

# 📚 What To Revise
(List revision topics)

# 🎯 Practice Recommendation
(Give practice advice)

# 🚀 Improvement Plan
(Step-by-step improvement plan)

Keep it practical and student-friendly.
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
        "status": "success",
        "analysis": res.choices[0].message.content
    }


    
    
    
    
    




    
    








    

    
    
