from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
import os

# ====================================================
# FASTAPI APP
# ====================================================

app = FastAPI()

# ====================================================
# CORS
# ====================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ====================================================
# GROQ CLIENT
# ====================================================

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

# ====================================================
# ROOT
# ====================================================

@app.get("/")
def home():

    return {
        "status": "ICSE AI Teacher Running 🚀"
    }

# ====================================================
# REQUEST MODEL
# ====================================================

class LearnRequest(BaseModel):

    student_class: str
    subject: str
    chapter: str

# ====================================================
# AI TEACH API
# ====================================================

@app.post("/learn")
def learn(data: LearnRequest):

    prompt = f"""
You are a highly professional ICSE school teacher.

Teach students in a very clear,
easy,
professional,
interactive,
student-friendly way.

Student Details:
Class: {data.student_class}
Subject: {data.subject}
Chapter: {data.chapter}

IMPORTANT RULES:

1. Explain step-by-step.
2. Use simple English.
3. Follow ICSE syllabus style.
4. Give important definitions.
5. Give examples.
6. Give key points.
7. Give revision notes.
8. Give memory tricks if useful.
9. Ask 5 practice questions at the end.
10. Motivate the student positively.

FORMAT:

# Introduction
# Main Explanation
# Important Definitions
# Examples
# Key Points
# Quick Revision
# Practice Questions

Make it feel like a real teacher teaching a student.
"""

    completion = client.chat.completions.create(

        model="llama-3.1-8b-instant",

        messages=[

            {
                "role": "system",
                "content": prompt
            }

        ]

    )

    return {

        "class": data.student_class,
        "subject": data.subject,
        "chapter": data.chapter,
        "lesson": completion.choices[0].message.content
    }
