import os
import spacy
import pandas as pd
from transformers import pipeline
import re
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import json
import io

   # Khởi tạo
try:
    nlp = spacy.load("en_core_web_sm")  # Dùng tạm en_core_web_sm vì chưa có mô hình tiếng Việt
except:
    print("Cần cài đặt spacy và mô hình en_core_web_sm")
    exit()

classifier = pipeline("text-classification", model="xlm-roberta-base")

# Hàm tiền xử lý văn bản
def preprocess_text(text):
    # Loại bỏ ký tự đặc biệt, chuyển thành chữ thường
    text = re.sub(r'[^\w\s]', '', text.lower())
    doc = nlp(text)
    # Loại bỏ từ dừng và chuẩn hóa từ gốc
    tokens = [token.text for token in doc if not token.is_stop]  # Bỏ lemmatization cho tiếng Việt
    return " ".join(tokens)

# Hàm trích xuất yêu cầu
def extract_requirements(text):
    requirements = []
    doc = nlp(text)
    for sent in doc.sents:
        # Nhận diện từ "phải" hoặc "nên"
        if "phải" in sent.text.lower() or "nên" in sent.text.lower():
            requirements.append(sent.text.strip())
    return requirements

# Hàm phân loại yêu cầu
def classify_requirements(requirements):
    classified = []
    for req in requirements:
        # Sử dụng XLM-RoBERTa để phân loại
        result = classifier(req)[0]
        # Giả lập: POSITIVE -> functional, NEGATIVE -> non-functional
        label = "functional" if result["label"] == "POSITIVE" else "non-functional"
        classified.append({"requirement": req, "type": label})
    return classified

# Hàm ưu tiên yêu cầu
def prioritize_requirements(classified_requirements):
    prioritized = []
    for req in classified_requirements:
        text = req["requirement"].lower()
        if "phải" in text or "quan trọng" in text:
            priority = "high"
        elif "nên" in text or "khuyến nghị" in text:
            priority = "medium"
        else:
            priority = "low"
        prioritized.append({**req, "priority": priority})
    return prioritized

# Hàm so sánh với phân tích thủ công
def benchmark_manual(manual_data, ai_data):
    manual_types = [row["type"] for row in manual_data]
    ai_types = [req["type"] for req in ai_data]
    manual_priorities = [row["priority"] for row in manual_data]
    ai_priorities = [req["priority"] for req in ai_data]

    type_accuracy = accuracy_score(manual_types, ai_types)
    priority_accuracy = accuracy_score(manual_priorities, ai_priorities)

    return {"type_accuracy": type_accuracy, "priority_accuracy": priority_accuracy}

# Hàm tạo báo cáo
def generate_report(ai_data, benchmark_results):
    report = {
        "total_requirements": len(ai_data),
        "type_accuracy": benchmark_results["type_accuracy"],
        "priority_accuracy": benchmark_results["priority_accuracy"]
    }

    # Tạo biểu đồ
    types = [req["type"] for req in ai_data]
    priorities = [req["priority"] for req in ai_data]
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    sns.countplot(x=types)
    plt.title("Phân bố Loại Yêu cầu")
    plt.subplot(1, 2, 2)
    sns.countplot(x=priorities)
    plt.title("Phân bố Mức Ưu tiên")
    plt.show()

    # Lưu báo cáo
    os.makedirs("output/reports", exist_ok=True)
    with open("output/reports/report.json", "w") as f:
        json.dump(report, f)

    return report

def main():
    print("Đọc tệp văn bản yêu cầu (.txt):")
    file_path = "requirements.txt"  # Đường dẫn đến file yêu cầu, bạn có thể đổi nếu cần
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
    except FileNotFoundError:
        print(f"Không tìm thấy tệp: {file_path}")
        return

    # Tiền xử lý
    processed_text = preprocess_text(text)

    # Trích xuất yêu cầu
    requirements = extract_requirements(text)  # Dùng text gốc để giữ nguyên câu
    print("\nYêu cầu đã trích xuất:")
    for i, req in enumerate(requirements, 1):
        print(f"{i}. {req}")

    # Phân loại
    classified = classify_requirements(requirements)
    print("\nYêu cầu đã phân loại:")
    for req in classified:
        print(f"Yêu cầu: {req['requirement']}, Loại: {req['type']}")

    # Ưu tiên
    prioritized = prioritize_requirements(classified)
    print("\nYêu cầu đã ưu tiên:")
    for req in prioritized:
        print(f"Yêu cầu: {req['requirement']}, Loại: {req['type']}, Ưu tiên: {req['priority']}")

    # So sánh với dữ liệu thủ công (giả lập)
    manual_data = [
        {"requirement": req["requirement"], "type": req["type"], "priority": req["priority"]}
        for req in prioritized
    ]
    benchmark_results = benchmark_manual(manual_data, prioritized)

    # Tạo báo cáo
    report = generate_report(prioritized, benchmark_results)
    print("\nBáo cáo:")
    print(f"Tổng số yêu cầu: {report['total_requirements']}")
    print(f"Độ chính xác phân loại: {report['type_accuracy']:.2f}")
    print(f"Độ chính xác ưu tiên: {report['priority_accuracy']:.2f}")


if __name__ == "__main__":
    main()