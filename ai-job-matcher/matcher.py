"""Core logic: PDF text extraction, skill detection, match scoring, ATS checks."""
import re
import pdfplumber
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# canonical skill -> list of ways it may be written
SKILLS = {
    "python": ["python"],
    "java": ["java"],
    "javascript": ["javascript", "js", "es6"],
    "typescript": ["typescript"],
    "react": ["react", "reactjs", "react.js"],
    "node.js": ["node.js", "nodejs", "node js"],
    "express": ["express", "express.js"],
    "html": ["html", "html5"],
    "css": ["css", "css3"],
    "tailwind": ["tailwind"],
    "sql": ["sql"],
    "mysql": ["mysql"],
    "postgresql": ["postgresql", "postgres"],
    "mongodb": ["mongodb", "mongo"],
    "dbms": ["dbms", "database management"],
    "flask": ["flask"],
    "django": ["django"],
    "fastapi": ["fastapi"],
    "spring boot": ["spring boot", "springboot", "spring"],
    "rest api": ["rest api", "rest apis", "restful", "rest"],
    "jdbc": ["jdbc"],
    "oop": ["oop", "oops", "object oriented", "object-oriented"],
    "data structures": ["data structures", "dsa", "algorithms"],
    "git": ["git"],
    "github": ["github"],
    "docker": ["docker"],
    "aws": ["aws", "amazon web services"],
    "linux": ["linux"],
    "machine learning": ["machine learning", "ml"],
    "deep learning": ["deep learning"],
    "nlp": ["nlp", "natural language processing"],
    "pandas": ["pandas"],
    "numpy": ["numpy"],
    "scikit-learn": ["scikit-learn", "sklearn"],
    "tensorflow": ["tensorflow"],
    "pytorch": ["pytorch"],
    "power bi": ["power bi", "powerbi"],
    "tableau": ["tableau"],
    "excel": ["excel"],
    "data analysis": ["data analysis", "data analytics"],
    "sap abap": ["abap", "sap abap"],
    "sap btp": ["btp", "sap btp"],
    "c++": ["c++"],
    "c": ["c programming"],
    "agile": ["agile", "scrum"],
    "testing": ["unit testing", "testing", "junit", "pytest"],
    "communication": ["communication"],
    "problem solving": ["problem solving", "problem-solving"],
}

SECTIONS = {
    "Education": ["education", "academic"],
    "Skills": ["skills", "technical skills"],
    "Projects": ["projects", "project"],
    "Experience / Internships": ["experience", "internship", "internships"],
    "Certifications": ["certification", "certifications", "courses"],
}


def extract_text_from_pdf(file_like):
    text = ""
    with pdfplumber.open(file_like) as pdf:
        for page in pdf.pages:
            text += (page.extract_text() or "") + "\n"
    return text


def page_count(file_like):
    file_like.seek(0)
    with pdfplumber.open(file_like) as pdf:
        return len(pdf.pages)


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9+#.\s-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _contains(text, phrase):
    pattern = r"(?<![a-z0-9+#])" + re.escape(phrase) + r"(?![a-z0-9+#])"
    return re.search(pattern, text) is not None


def find_skills(text):
    text = clean_text(text)
    return {
        skill
        for skill, variants in SKILLS.items()
        if any(_contains(text, v) for v in variants)
    }


def tfidf_similarity(resume, jd):
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    m = vec.fit_transform([clean_text(resume), clean_text(jd)])
    return float(cosine_similarity(m[0], m[1])[0][0])


def ats_checks(resume_text, pages=None):
    """Simple, explainable ATS-readiness checks."""
    low = resume_text.lower()
    checks = []

    has_email = re.search(r"[\w.+-]+@[\w-]+\.[\w.]+", resume_text) is not None
    checks.append(("Email address found", has_email))

    has_phone = re.search(r"(\+?\d[\d\s-]{8,}\d)", resume_text) is not None
    checks.append(("Phone number found", has_phone))

    links = "github.com" in low or "linkedin.com" in low
    checks.append(("GitHub or LinkedIn link found", links))

    for name, words in SECTIONS.items():
        if name == "Experience / Internships":
            continue  # optional for freshers
        found = any(re.search(rf"^\s*{w}\b", low, re.M) for w in words)
        checks.append((f"'{name}' section heading found", found))

    if pages is not None:
        checks.append(("Resume is one page", pages == 1))

    numbers = len(re.findall(r"\b\d+%?\b", resume_text))
    checks.append(("Uses numbers to show impact (3+)", numbers >= 3))

    words = len(resume_text.split())
    checks.append(("Readable text length (150-800 words)", 150 <= words <= 800))
    return checks


def analyze(resume_text, jd_text, pages=None):
    resume_skills = find_skills(resume_text)
    jd_skills = find_skills(jd_text)

    matched = sorted(jd_skills & resume_skills)
    missing = sorted(jd_skills - resume_skills)
    extra = sorted(resume_skills - jd_skills)

    coverage = (len(matched) / len(jd_skills)) if jd_skills else 0.0
    similarity = tfidf_similarity(resume_text, jd_text)

    # Skill coverage matters most; text similarity adds context.
    if jd_skills:
        overall = 0.65 * coverage + 0.35 * min(similarity * 2, 1.0)
    else:
        overall = min(similarity * 2, 1.0)

    checks = ats_checks(resume_text, pages)
    ats_score = round(100 * sum(ok for _, ok in checks) / len(checks))

    tips = []
    for s in missing[:6]:
        tips.append(
            f"The job asks for {s}. If you have used it, add it to your Skills "
            f"section and a project bullet. If not, build a small project with it."
        )
    for label, ok in checks:
        if not ok:
            tips.append(f"Fix: {label}.")

    return {
        "match_score": round(overall * 100),
        "skill_coverage": round(coverage * 100),
        "text_similarity": round(similarity * 100),
        "ats_score": ats_score,
        "matched": matched,
        "missing": missing,
        "extra": extra,
        "job_skills_found": len(jd_skills),
        "checks": [{"label": l, "ok": ok} for l, ok in checks],
        "tips": tips,
    }
