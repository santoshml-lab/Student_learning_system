
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>AI Learning System 🚀 LEVEL 4</title>
<meta name="viewport" content="width=device-width, initial-scale=1.0">

<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>

<style>

*{margin:0;padding:0;box-sizing:border-box;font-family:Poppins;}

body{
    background:#0f172a;
    color:white;
}

/* HEADER */
header{
    text-align:center;
    padding:18px;
    background:rgba(255,255,255,0.05);
    border-bottom:1px solid rgba(255,255,255,0.1);
}

header h1{color:#38bdf8}

/* GRID */
.container{
    display:grid;
    grid-template-columns:320px 1fr;
    gap:20px;
    padding:20px;
}

/* CARD */
.card{
    background:rgba(255,255,255,0.05);
    border:1px solid rgba(255,255,255,0.1);
    padding:18px;
    border-radius:18px;
}

/* INPUT */
input{
    width:100%;
    padding:12px;
    margin-top:10px;
    border-radius:10px;
    border:none;
    background:#1e293b;
    color:white;
}

/* BUTTON */
button{
    width:100%;
    margin-top:12px;
    padding:12px;
    border:none;
    border-radius:10px;
    background:#38bdf8;
    font-weight:bold;
    cursor:pointer;
}

button:hover{background:#0ea5e9}

/* LESSON */
.lesson{
    line-height:1.8;
    font-size:15px;
}

.lesson h1,.lesson h2{color:#38bdf8}

/* ITEMS */
.item{
    margin-top:8px;
    padding:10px;
    border-radius:10px;
    background:#111827;
}

/* LOADER */
.loader{
    display:none;
    color:#38bdf8;
}

/* MOBILE */
@media(max-width:900px){
    .container{grid-template-columns:1fr;}
}

</style>
</head>

<body>

<header>
<h1>🚀 AI Learning System (LEVEL 4 PRO)</h1>
</header>

<div class="container">

<!-- LEFT PANEL -->
<div class="card">

<h3>👤 Student Panel</h3>

<input id="cls" placeholder="Class">
<input id="sub" placeholder="Subject">
<input id="chap" placeholder="Chapter">

<button onclick="learn()">Start AI Learning</button>

<hr style="margin:15px 0;opacity:0.2">

<h3>📌 Saved Chapters</h3>
<div id="saved">Loading...</div>

<hr style="margin:15px 0;opacity:0.2">

<h3>🏆 Leaderboard</h3>
<div id="board">Loading...</div>

</div>

<!-- RIGHT PANEL -->
<div class="card">

<h3>🤖 AI Teacher</h3>

<div class="loader" id="loader">AI is thinking...</div>

<div class="lesson" id="lesson">
Start learning 🚀
</div>

</div>

</div>

<script>

const API = "https://student-learning-system-r6bi.onrender.com";

/* =========================
   USER SYSTEM (LEVEL 4 CORE)
========================= */

let user = JSON.parse(localStorage.getItem("user"));

if(!user){
    user = {
        id: "U_" + Math.floor(Math.random()*999999),
        xp: 0,
        streak: 1,
        history: []
    };
    localStorage.setItem("user", JSON.stringify(user));
}

/* =========================
   STATE
========================= */

let saved = [];

/* =========================
   LOAD CHAPTERS
========================= */

async function loadChapters(){
    let res = await fetch(`${API}/chapters`);
    let data = await res.json();

    saved = data.chapters || [];
    renderSaved();
}

/* =========================
   RENDER SAVED
========================= */

function renderSaved(){

    let box = document.getElementById("saved");

    if(saved.length === 0){
        box.innerHTML = "No saved chapters 🚀";
        return;
    }

    box.innerHTML = "";

    saved.forEach(c=>{
        box.innerHTML += `
        <div class="item">
            📘 ${c.subject}<br>
            📖 ${c.chapter}
        </div>
        `;
    });
}

/* =========================
   LEADERBOARD (FRONTEND MOCK / DB READY)
========================= */

async function loadBoard(){

    // backend-ready endpoint (future)
    try{
        let res = await fetch(`${API}/leaderboard`);
        let data = await res.json();

        renderBoard(data.leaderboard || []);
    }
    catch(e){

        // fallback demo leaderboard
        renderBoard([
            {name:"Aman", xp:120},
            {name:"Ruchi", xp:110},
            {name:"Santosh", xp:95}
        ]);
    }
}

function renderBoard(list){

    let box = document.getElementById("board");
    box.innerHTML = "";

    list.sort((a,b)=>b.xp-a.xp);

    list.forEach((u,i)=>{
        box.innerHTML += `
        <div class="item">
            ${i+1}. ${u.name || "User"} — ${u.xp} XP
        </div>
        `;
    });
}

/* =========================
   UPDATE USER XP
========================= */

function updateXP(){
    user.xp += 15;
    user.history.unshift({
        subject: document.getElementById("sub").value,
        chapter: document.getElementById("chap").value
    });

    localStorage.setItem("user", JSON.stringify(user));
}

/* =========================
   LEARN FUNCTION (CORE ENGINE)
========================= */

async function learn(){

    let cls = document.getElementById("cls").value;
    let sub = document.getElementById("sub").value;
    let chap = document.getElementById("chap").value;

    if(!cls || !sub || !chap){
        alert("Fill all fields");
        return;
    }

    document.getElementById("loader").style.display="block";

    try{

        let res = await fetch(`${API}/learn`,{
            method:"POST",
            headers:{"Content-Type":"application/json"},
            body:JSON.stringify({
                student_class:cls,
                subject:sub,
                chapter:chap
            })
        });

        let data = await res.json();

        document.getElementById("lesson").innerHTML =
            marked.parse(data.lesson || "No lesson");

        await fetch(`${API}/save-chapter`,{
            method:"POST",
            headers:{"Content-Type":"application/json"},
            body:JSON.stringify({
                student_class:cls,
                subject:sub,
                chapter:chap
            })
        });

        saved.unshift({subject:sub, chapter:chap});
        renderSaved();

        updateXP();
        loadBoard();

    }catch(err){
        console.log(err);
        alert("Error occurred");
    }

    document.getElementById("loader").style.display="none";
}

/* INIT */
loadChapters();
loadBoard();

</script>

</body>
</html>
