
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
import os
import psycopg2

app = FastAPI()







app.add_middleware(
CORSMiddleware,
allow_origins=[""],
allow_credentials=True,
allow_methods=[""],
allow_headers=["*"],
)






client = Groq(api_key=os.getenv("GROQ_API_KEY"))







DATABASE_URL = os.getenv("DATABASE_URL")

if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

def get_conn():
if not DATABASE_URL:
raise Exception("DATABASE_URL missing")
return psycopg2.connect(DATABASE_URL, sslmode="require")







def init_db():
try:
conn = get_conn()
cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS chapters (
        id SERIAL PRIMARY KEY,
        student_name TEXT,
        student_class TEXT,
        subject TEXT,
        chapter TEXT,
        xp INTEGER DEFAULT 10
    )
    """)

    conn.commit()
    cur.close()
    conn.close()
    print("DB initialized ✅")

except Exception as e:
    print("DB ERROR:", e)

init_db()







class LearnRequest(BaseModel):
student_name: str
student_class: str
subject: str
chapter: str

class ChatInput(BaseModel):
message: str







@app.get("/")
def home():
return {"status": "LEVEL 6 AI EDTECH RUNNING 🚀"}







@app.get("/health")
def health():
return {"status": "healthy ✅"}







@app.post("/learn")
def learn(data: LearnRequest):

try:
    prompt = f"""

You are an expert ICSE teacher.

Teach the chapter in a detailed classroom teaching style.

Class: {data.student_class}
Subject: {data.subject}
Chapter: {data.chapter}

RULES:

- Explain in very simple language

- Teach step by step like a real teacher

- Give detailed explanations

- Add examples after every concept

- Explain difficult words

- Use headings and subheadings

- Add real-life examples

- Add important notes

- Add revision section

- Add 10 practice questions

- Do NOT summarize too much

- Teach deeply so weak students understand
  """
  
    completion = client.chat.completions.create(
      model="llama-3.1-8b-instant",
      messages=[{"role": "system", "content": prompt}],
      max_tokens=2000
  )

  return {
      "lesson": completion.choices[0].message.content
  }
  
  except Exception as e:
  return {"error": str(e)}

=========================

CHATBOT (EXAMPANIC AI)



@app.post("/chat")
def chat(data: ChatInput):

try:
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": """

You are ExamPanic AI, a friendly AI teacher.

Help students with:

- exams
- doubts
- revision
- motivation
- study planning
- concept clarity

RULES:

- Simple language

- Step-by-step answers

- Friendly tone

- Examples when needed

- No robotic answers
  """
  },
  {
  "role": "user",
  "content": data.message
  }
  ],
  max_tokens=1200
  )
  
    return {"reply": response.choices[0].message.content}
  
  except Exception as e:
  return {"error": str(e)}







@app.post("/save-chapter")
def save_chapter(data: LearnRequest):

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

return {"message": "saved 🚀"}





@app.get("/chapters")
def get_chapters():

conn = get_conn()
cur = conn.cursor()

cur.execute("""
SELECT student_name, subject, chapter
FROM chapters
ORDER BY id DESC
LIMIT 20
""")

rows = cur.fetchall()

cur.close()
conn.close()

return {
    "chapters": [
        {
            "student": r[0],
            "subject": r[1],
            "chapter": r[2]
        }
        for r in rows
    ]
}







@app.get("/leaderboard")
def leaderboard():

conn = get_conn()
cur = conn.cursor()

cur.execute("""
SELECT student_name, SUM(xp) as total_xp
FROM chapters
GROUP BY student_name
ORDER BY total_xp DESC
LIMIT 10
""")

rows = cur.fetchall()

cur.close()
conn.close()

return {
    "leaderboard": [
        {
            "name": r[0],
            "xp": r[1]
        }
        for r in rows
    ]
}






i
@app.get("/quick-access")
def quick_access():

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
    "quick_access": [
        {
            "subject": r[0],
            "chapter": r[1]
        }
        for r in rows
    ]
}
