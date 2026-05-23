
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
import json
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

class MockRequest(BaseModel):
    student_class: str
    subject: str
    chapter: str

class Submit(BaseModel):
    questions: list
    answers: list

@app.post("/generate-test")
def generate_test(data: MockRequest):

    prompt = f"""
Generate 5 MCQs STRICT JSON ONLY.

FORMAT:
[
  {{
    "q": "Question?",
    "options": {{
        "A": "option1",
        "B": "option2",
        "C": "option3",
        "D": "option4"
    }},
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

    content = res.choices[0].message.content
    content = content.replace("```json","").replace("```","").strip()

    return json.loads(content)


@app.post("/submit-test")
def submit(data: Submit):

    correct = 0

    for i in range(len(data.questions)):

        if data.answers[i] == data.questions[i]["answer"]:
            correct += 1

    score = round((correct / len(data.questions)) * 100, 2)

    return {
        "score": score,
        "correct": correct,
        "total": len(data.questions)
    }

    
    
