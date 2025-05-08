from flask import Flask, render_template, request, redirect, url_for
import os
import spacy
from transformers import pipeline
import re
import json
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except:
    print("Missing spaCy model.")
    exit()

classifier = pipeline("text-classification", model="xlm-roberta-base")

def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text.lower())
    doc = nlp(text)
    tokens = [token.text for token in doc if not token.is_stop]
    return " ".join(tokens)

def extract_requirements(text):
    requirements = []
    doc = nlp(text)
    for sent in doc.sents:
        if "phải" in sent.text.lower() or "nên" in sent.text.lower():
            requirements.append(sent.text.strip())
    return requirements

def classify_requirements(requirements):
    classified = []
    for req in requirements:
        result = classifier(req)[0]
        label = "functional" if result["label"] == "POSITIVE" else "non-functional"
        classified.append({"requirement": req, "type": label})
    return classified

def prioritize_requirements(classified):
    prioritized = []
    for req in classified:
        text = req["requirement"].lower()
        if "phải" in text or "quan trọng" in text:
            priority = "high"
        elif "nên" in text or "khuyến nghị" in text:
            priority = "medium"
        else:
            priority = "low"
        prioritized.append({**req, "priority": priority})
    return prioritized

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
    os.makedirs("output/reports", exist_ok=True)
    plt.savefig("output/reports/distribution.png")
    with open("output/reports/report.json", "w") as f:
        json.dump(report, f)
    return report

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        uploaded_file = request.files["file"]
        if uploaded_file and uploaded_file.filename.endswith(".txt"):
            content = uploaded_file.read().decode("utf-8")
            requirements = extract_requirements(content)
            classified = classify_requirements(requirements)
            prioritized = prioritize_requirements(classified)
            manual_data = prioritized  # Giả lập
            benchmark = benchmark_manual(manual_data, prioritized)
            report = generate_report(prioritized, benchmark)
            return render_template("index.html", 
                requirements=prioritized, 
                report=report, 
                chart_url="/static/report.png")

    return render_template("index.html", requirements=None)

if __name__ == "__main__":
    app.run(debug=True)
