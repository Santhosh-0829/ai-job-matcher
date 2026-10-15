"""Flask server for the AI Career Intelligence & Job Matching System."""
import io
from flask import Flask, jsonify, render_template, request
from matcher import analyze, extract_text_from_pdf, page_count

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5 MB upload limit


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    jd = (request.form.get("job_description") or "").strip()
    if len(jd) < 30:
        return jsonify(error="Paste the full job description (at least a few lines)."), 400

    pages = None
    file = request.files.get("resume")
    if file and file.filename:
        if not file.filename.lower().endswith(".pdf"):
            return jsonify(error="Upload your resume as a PDF file."), 400
        data = io.BytesIO(file.read())
        try:
            resume_text = extract_text_from_pdf(data)
            pages = page_count(data)
        except Exception:
            return jsonify(error="Could not read this PDF. Try a different file."), 400
    else:
        resume_text = (request.form.get("resume_text") or "").strip()

    if len(resume_text.split()) < 30:
        return jsonify(
            error="No readable text found. Use a PDF exported from Word or Google Docs "
                  "(scanned images do not work), or paste your resume text."
        ), 400

    return jsonify(analyze(resume_text, jd, pages))


if __name__ == "__main__":
    app.run(debug=True, port=5000)
