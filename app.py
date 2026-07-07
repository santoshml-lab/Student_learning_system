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


class PDFRequest(BaseModel):
    text: str 


class PDFChatRequest(BaseModel):
    text: str
    question: str


class PDFMCQRequest(BaseModel):
    text: str


class PDFMindMapRequest(BaseModel):
    text: str


class PDFProblemRequest(BaseModel):
    text: str


class PDFFormulaRequest(BaseModel):
    text: str
    

class PDFDefinitionRequest(BaseModel):
    text: str


class PDFStudyPlanRequest(BaseModel):
    text: str


class PDFDifficultyRequest(BaseModel):
    text: str

class PDFRevisionRequest(BaseModel):
    text: str


class PDFDiagramRequest(BaseModel):
    text: str


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
        model="openai/gpt-oss-20b",
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
        model="openai/gpt-oss-20b",
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
        model="openai/gpt-oss-20b",
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

        model="openai/gpt-oss-20b",

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

        model="openai/gpt-oss-20b",

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

Analyze the student's performance and identify the exact concepts where improvement is needed.

Subject:
{data.subject}

Chapter:
{data.chapter}

Score:
{data.score}

Mistakes:
{data.mistakes}

IMPORTANT RULES:

- Give chapter-specific analysis
- Give concept-specific weaknesses
- Give revision points related ONLY to the chapter
- Recommend practice questions based on mistakes
- Never recommend websites
- Never recommend YouTube channels
- Never recommend study groups
- Never recommend external resources
- Keep recommendations practical
- Focus on concept improvement
- Use ICSE-style academic language

STRICT FORMAT:

# 🧠 Weak Areas

# 📚 What To Revise

# 🎯 Practice Recommendation

## MCQs
Generate 5 MCQs

## Short Questions
Generate 3 short-answer questions

## Numerical / Application Questions
Generate 2 numerical or application-based questions if applicable

# 🚀 Improvement Plan

Create a short day-wise improvement plan.

Return only the analysis.
"""

    res = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "system",
                "content": prompt
            }
        ],
        max_tokens=1800
    )

    return {
        "status": "success",
        "analysis": res.choices[0].message.content
    }

@app.post("/pdf-summary")
def pdf_summary(data: PDFRequest):

    prompt = f"""
You are ExamPanic AI PDF Teacher.

Analyze the following PDF content and generate:

# 📄 Summary

# 📝 Key Points

# 📚 Important Definitions

# 🎯 Exam Tips

# ❓ 5 Practice Questions

PDF Content:
{data.text}
"""

    res = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "system",
                "content": prompt
            }
        ],
        max_tokens=3000
    )

    return {
        "status": "success",
        "reply": res.choices[0].message.content
    }
@app.post("/pdf-chat")
def pdf_chat(data: PDFChatRequest):

    prompt = f"""
You are ExamPanic PDF AI.

Answer ONLY using the uploaded PDF.

PDF:
{data.text}

Question:
{data.question}

Rules:
- Don't use outside knowledge.
- If the answer isn't in the PDF, say:
  "This information is not available in the uploaded PDF."
"""

    res = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "system",
                "content": prompt
            }
        ],
        max_tokens=2000
    )

    return {
        "reply": res.choices[0].message.content
    }

@app.post("/pdf-mcq")
def pdf_mcq(data: PDFMCQRequest):

    prompt = f"""
Generate EXACTLY 10 ICSE MCQs from the following PDF.

IMPORTANT RULES:

- Return ONLY a valid JSON array.
- Do NOT write markdown.
- Do NOT write explanations before or after JSON.
- Each question must have exactly 4 options.
- Answer must be only A, B, C or D.

Format:

[
  {{
    "question":"...",
    "options":[
      "Option A",
      "Option B",
      "Option C",
      "Option D"
    ],
    "answer":"A",
    "explanation":"..."
  }}
]

PDF:

{data.text[:12000]}
"""

    try:

        res = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            temperature=0,
            max_completion_tokens=3000,
            top_p=1,
            messages=[
                {
                    "role": "system",
                    "content": "You are an ICSE MCQ generator. Return ONLY valid JSON."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        content = res.choices[0].message.content.strip()

        print(content)

        # Extract JSON array
        start = content.find("[")
        end = content.rfind("]")

        if start == -1 or end == -1:
            raise Exception("JSON array not found.")

        content = content[start:end+1]

        questions = json.loads(content)

        for q in questions:

            if "question" not in q:
                q["question"] = ""

            if "options" not in q or not isinstance(q["options"], list):
                q["options"] = [
                    "Option A",
                    "Option B",
                    "Option C",
                    "Option D"
                ]

            while len(q["options"]) < 4:
                q["options"].append("Option Missing")

            q["options"] = q["options"][:4]

            if q.get("answer") not in ["A", "B", "C", "D"]:
                q["answer"] = "A"

            if "explanation" not in q:
                q["explanation"] = ""

        return {
            "status": "success",
            "questions": questions
        }

    except Exception as e:

        return {
            "status": "error",
            "message": str(e),
            "raw": content if "content" in locals() else ""
            }
@app.post("/pdf-notes")
def pdf_notes(data: PDFRequest):

    prompt = f"""
You are ExamPanic AI.

Create HIGH QUALITY STUDY NOTES from the uploaded PDF.

RULES:

- Use ONLY the uploaded PDF.
- Do not use outside knowledge.
- Write in textbook style.
- Make notes easy to revise.
- Use headings.
- Use bullet points.
- Highlight important concepts.
- Explain difficult topics simply.

FORMAT:

# 📘 Chapter Title

# 📌 Introduction

# 📖 Detailed Notes

# ⭐ Important Points

# 📚 Key Terms

# 🔥 Memory Tricks

# ❓ Important Exam Questions

# 📝 Quick Revision

# 🚀 Final Summary

PDF:

{data.text}
"""

    try:

        res = client.chat.completions.create(

            model="openai/gpt-oss-20b",

            messages=[
                {
                    "role": "system",
                    "content": "Generate only high quality study notes."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],

            temperature=0.4,
            max_completion_tokens=3500

        )

        return {
            "status": "success",
            "notes": res.choices[0].message.content
        }

    except Exception as e:

        return {
            "status": "error",
            "message": str(e)
        }
        
@app.post("/pdf-flashcards")
def pdf_flashcards(data: PDFRequest):

    prompt = f"""
You are ExamPanic AI.

Create EXACTLY 20 flashcards from the uploaded PDF.

Rules:
- Use ONLY the PDF.
- No outside knowledge.
- Keep answers short.
- Perfect for quick revision.

Return ONLY Markdown.

Format:

# 🧠 Flashcards

### Card 1
Q: ...
A: ...

### Card 2
Q: ...
A: ...

Continue until Card 20.

PDF:

{data.text}
"""

    try:

        res = client.chat.completions.create(

            model="openai/gpt-oss-20b",

            messages=[
                {
                    "role": "system",
                    "content": "Generate revision flashcards only."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],

            temperature=0.3,
            max_completion_tokens=2500

        )

        return {
            "status": "success",
            "flashcards": res.choices[0].message.content
        }

    except Exception as e:

        return {
            "status": "error",
            "message": str(e)
        }
        
@app.post("/pdf-mindmap")
def pdf_mindmap(data: PDFMindMapRequest):

    prompt = f"""
You are an expert teacher.

Read the PDF carefully.

Generate ONLY Mermaid Mind Map.

Rules:
- Return only Mermaid code.
- No markdown.
- No explanation.

Example:

mindmap
  root((Electricity))
    Current
      AC
      DC
    Voltage
    Resistance
    Ohm's Law

PDF:

{data.text}
"""

    try:

        res = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            temperature=0.3,
            max_completion_tokens=3000,
            messages=[
                {
                    "role": "system",
                    "content": "Return only Mermaid code."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return {
            "status": "success",
            "mindmap": res.choices[0].message.content
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }


    
            
                 
        
@app.post("/pdf-problems")
def pdf_problems(data: PDFProblemRequest):

    prompt = f"""
You are an expert Physics, Chemistry and Mathematics teacher.

Read the PDF carefully.

Find every numerical or problem-solving question.

For each problem return:

## Question

### Given

### Formula

### Solution (step by step)

### Final Answer

### Unit

If there are no numericals, generate 5 practice numerical questions from the topic and solve them.

Return Markdown only.

PDF:

{data.text}
"""

    try:

        res = client.chat.completions.create(

            model="openai/gpt-oss-20b",

            temperature=0.2,

            max_completion_tokens=3500,

            messages=[
                {
                    "role":"system",
                    "content":"Return only Markdown."
                },
                {
                    "role":"user",
                    "content":prompt
                }
            ]
        )

        return {
            "status":"success",
            "problems":res.choices[0].message.content
        }

    except Exception as e:

        return {
            "status":"error",
            "message":str(e)
        }
        
@app.post("/pdf-formulas")
def pdf_formulas(data: PDFFormulaRequest):

    prompt = f"""
Read the PDF carefully.

Extract ALL formulas from the PDF.

For every formula provide:

# Formula Name

Formula

Meaning

Variables

Units (if applicable)

One line explanation

If there are no formulas, write:
"No formulas found."

Return Markdown only.

PDF:

{data.text}
"""

    try:

        res = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            temperature=0.2,
            max_completion_tokens=3000,
            messages=[
                {
                    "role":"system",
                    "content":"Return only Markdown."
                },
                {
                    "role":"user",
                    "content":prompt
                }
            ]
        )

        return {
            "status":"success",
            "formulas":res.choices[0].message.content
        }

    except Exception as e:

        return {
            "status":"error",
            "message":str(e)
        }
        
@app.post("/pdf-definitions")
def pdf_definitions(data: PDFDefinitionRequest):

    prompt = f"""
Read the PDF carefully.

Extract ALL important definitions.

For every definition provide:

# Term

Definition

One line explanation

Example (if applicable)

If there are no definitions, generate important definitions from the topic.

Return Markdown only.

PDF:

{data.text}
"""

    try:

        res = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            temperature=0.2,
            max_completion_tokens=3000,
            messages=[
                {
                    "role": "system",
                    "content": "Return only Markdown."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return {
            "status": "success",
            "definitions": res.choices[0].message.content
        }

    except Exception as e:

        return {
            "status": "error",
            "message": str(e)
        }
        
@app.post("/pdf-important-questions")
def pdf_important_questions(data: PDFDefinitionRequest):

    prompt = f"""
You are an expert teacher.

Read the PDF carefully.

Generate:

# Very Important Questions

## 5 One Mark Questions

## 5 Two Mark Questions

## 5 Short Answer Questions

## 5 Long Answer Questions

## 5 HOTS Questions

## 5 Assertion Reason Questions (if applicable)

Return Markdown only.

PDF:

{data.text}
"""

    try:

        res = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            temperature=0.3,
            max_completion_tokens=3500,
            messages=[
                {
                    "role":"system",
                    "content":"Return Markdown only."
                },
                {
                    "role":"user",
                    "content":prompt
                }
            ]
        )

        return {
            "status":"success",
            "questions":res.choices[0].message.content
        }

    except Exception as e:

        return {
            "status":"error",
            "message":str(e)
        }
        
@app.post("/pdf-study-plan")
def pdf_study_plan(data: PDFStudyPlanRequest):

    prompt = f"""
You are ExamPanic AI Study Planner.

Read the uploaded PDF carefully.

Create a smart study plan.

Rules:

- Use ONLY the uploaded PDF.
- Divide the chapter into logical study sessions.
- Assume a 7-day plan.
- Include revision and practice.
- Keep the plan practical for students.

Format:

# 📅 7-Day Study Plan

## Day 1
Topics
Tasks

## Day 2
Topics
Tasks

...

## Day 7
Final Revision
MCQs
Quiz
Weak Topics Review

PDF:

{data.text}
"""

    try:

        res = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            temperature=0.3,
            max_completion_tokens=3000,
            messages=[
                {
                    "role": "system",
                    "content": "Return only Markdown."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return {
            "status": "success",
            "study_plan": res.choices[0].message.content
        }

    except Exception as e:

        return {
            "status": "error",
            "message": str(e)
        }
        
@app.post("/pdf-difficulty")
def pdf_difficulty(data: PDFDifficultyRequest):

    prompt = f"""
You are ExamPanic AI Difficulty Analyzer.

Read the uploaded PDF carefully.

Analyze the chapter.

Return Markdown only.

Format:

# 📊 Chapter Difficulty Analysis

## 🟢 Easy Topics
- Topic
- Why easy

## 🟡 Medium Topics
- Topic
- Why medium

## 🔴 Hard Topics
- Topic
- Why difficult

## ⭐ High Priority Topics

## ⏱ Estimated Study Time

## 🎯 Exam Focus Tips

PDF:

{data.text}
"""

    try:

        res = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            temperature=0.3,
            max_completion_tokens=2500,
            messages=[
                {
                    "role": "system",
                    "content": "Return only Markdown."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return {
            "status": "success",
            "analysis": res.choices[0].message.content
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }
        
@app.post("/pdf-revision-plan")
def pdf_revision_plan(data: PDFRevisionRequest):

    prompt = f"""
You are ExamPanic AI Revision Planner.

Read the uploaded PDF.

Create a smart 7-day revision schedule.

Return Markdown only.

Format:

# 📅 AI Revision Plan

## Day 1
Study

## Day 2
Revision

## Day 3
Important Definitions

## Day 4
Formula Revision

## Day 5
Flashcards + MCQs

## Day 6
Weak Topics

## Day 7
Final Revision + Mock Test

PDF:

{data.text}
"""

    try:

        res = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            temperature=0.3,
            max_completion_tokens=2500,
            messages=[
                {
                    "role": "system",
                    "content": "Return only Markdown."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return {
            "status": "success",
            "revision": res.choices[0].message.content
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }
        
@app.post("/pdf-diagram")
def pdf_diagram(data: PDFDiagramRequest):







    
        
                    

           



      
      





  
    
    
  






    







    




    
    
    


   



    
    
    
    
    




    
    








    

    
    
