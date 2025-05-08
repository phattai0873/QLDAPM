from flask import Flask, render_template, request
import os
import re
import json
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import fitz  # PyMuPDF

# Import các hàm từ classify.py
from classify import classify_with_phobert, prioritize_requirements

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# ==== Helper ====

def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

def extract_requirements(text):
    # Câu nào chứa "phải", "nên", ... sẽ được xem là yêu cầu
    sentences = re.split(r"[.?!]\s*", text)
    requirements = [
        s.strip() for s in sentences
        if any(kw in s.lower() for kw in ["phải", "nên", "bắt buộc", "khuyến nghị"])
    ]
    return requirements

def benchmark_manual(manual_data, ai_data):
    manual_types = [row["type"] for row in manual_data]
    ai_types = [req["type"] for req in ai_data]
    manual_priorities = [row["priority"] for row in manual_data]
    ai_priorities = [req["priority"] for req in ai_data]
    return {
        "type_accuracy": accuracy_score(manual_types, ai_types),
        "priority_accuracy": accuracy_score(manual_priorities, ai_priorities)
    }

def generate_report(ai_data, benchmark_results):
    report = {
        "total_requirements": len(ai_data),
        "type_accuracy": benchmark_results["type_accuracy"],
        "priority_accuracy": benchmark_results["priority_accuracy"]
    }

    types = [req["type"] for req in ai_data]
    priorities = [req["priority"] for req in ai_data]
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    sns.countplot(x=types)
    plt.title("Loại yêu cầu")
    plt.subplot(1, 2, 2)
    sns.countplot(x=priorities)
    plt.title("Mức ưu tiên")
    plt.tight_layout()
    os.makedirs("static", exist_ok=True)
    plt.savefig("static/report.png")

    os.makedirs("output/reports", exist_ok=True)
    with open("output/reports/report.json", "w") as f:
        json.dump(report, f)
    return report

# ==== Routes ====

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        uploaded_file = request.files["file"]
        if uploaded_file:
            filename = uploaded_file.filename
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            uploaded_file.save(file_path)

            if filename.endswith(".pdf"):
                content = extract_text_from_pdf(file_path)
            elif filename.endswith(".txt"):
                content = open(file_path, encoding="utf-8").read()
            else:
                return "Unsupported file type", 400

            requirements = extract_requirements(content)
            classified = classify_with_phobert(requirements)
            prioritized = prioritize_requirements(classified)

            # Benchmark giả lập
            manual_data = prioritized
            benchmark = benchmark_manual(manual_data, prioritized)
            report = generate_report(prioritized, benchmark)

            return render_template("index.html",
                requirements=prioritized,
                report=report,
                chart_url="/static/report.png")

    return render_template("index.html", requirements=None)

if __name__ == "__main__":
    app.run(debug=True)
