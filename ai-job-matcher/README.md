# AI Career Intelligence & Job Matching System

A web app that compares your resume (PDF) with a job description and shows:
- Overall job match score
- Skills you match and skills you are missing
- ATS readiness checklist (email, phone, sections, length, impact numbers)
- Specific tips to improve your resume

## Tech stack
Python, Flask, scikit-learn (TF-IDF + cosine similarity), pdfplumber, HTML/CSS/JavaScript

## How it works
1. pdfplumber extracts text from the uploaded resume PDF.
2. A skills dictionary (with aliases such as "js" and "reactjs") finds skills in both texts.
3. Skill coverage = matched skills / skills the job asks for.
4. TF-IDF + cosine similarity measures how similar the wording is.
5. Match score = 65% skill coverage + 35% wording similarity.
6. Rule-based checks produce the ATS readiness score.

## Run locally
```bash
python -m venv venv
venv\Scripts\activate          # Windows  (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt
python app.py
```
Open http://127.0.0.1:5000

## Project structure
```
app.py                  Flask server and /api/analyze endpoint
matcher.py              Skill detection, scoring, ATS checks
templates/index.html    User interface
sample_job_description.txt
requirements.txt
```

## Future improvements
- Sentence-embedding similarity (Sentence-Transformers)
- Role suggestions from resume skills
- Save analysis history in MySQL
- Deploy on Render
