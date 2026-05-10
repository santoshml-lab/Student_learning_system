
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
import os
import sqlite3

# ====================================================
# APP
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
# DATABASE (CHAPTER SAVE SYSTEM)
# ====================================================

conn = sqlite3.connect("edtech.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS chapters (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_class TEXT,
    subject TEXT,
    chapter TEXT
)
""")

conn.commit()

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
# LEARN API (AI TEACHER)
# ====================================================

@app.post("/learn")
def learn(data: LearnRequest):

    prompt = f"""
You are a highly professional ICSE school teacher.

Teach students in a very clear,
easy,
interactive,
student-friendly way.

Student Details:
Class: {data.student_class}
Subject: {data.subject}
Chapter: {data.chapter}

IMPORTANT RULES:

1. Explain step-by-step
2. Use easy English
3. Follow ICSE style
4. Give important definitions
5. Give examples
6. Give formulas if needed
7. Give key points
8. Give quick revision notes
9. Give memory tricks if useful
10. Give 5 practice questions

IMPORTANT:
Return ONLY in MARKDOWN format.

FORMAT:

# Chapter Title

## Introduction

## Main Explanation

## Important Definitions

## Formulas

## Examples

## Key Points

## Quick Revision

## Practice Questions
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

    lesson = completion.choices[0].message.content

    return {

        "class": data.student_class,
        "subject": data.subject,
        "chapter": data.chapter,
        "lesson": lesson
    }

# ====================================================
# SAVE CHAPTER API
# ====================================================

@app.post("/save-chapter")
def save_chapter(data: LearnRequest):

    cursor.execute("""
        INSERT INTO chapters (student_class, subject, chapter)
        VALUES (?, ?, ?)
    """, (
        data.student_class,
        data.subject,
        data.chapter
    ))

    conn.commit()

    return {
        "msg": "chapter saved successfully"
    }

# ====================================================
# GET SAVED CHAPTERS
# ====================================================

@app.get("/chapters")
def get_chapters():

    cursor.execute("""
        SELECT student_class, subject, chapter FROM chapters
    """)

    rows = cursor.fetchall()

    return {
        "chapters": [
            {
                "class": r[0],
                "subject": r[1],
                "chapter": r[2]
            }
            for r in rows
        ]
    }
