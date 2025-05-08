from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load PhoBERT model
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)
model = AutoModelForSequenceClassification.from_pretrained("vinai/phobert-base", num_labels=2)

def classify_with_phobert(requirements):
    results = []
    for req in requirements:
        inputs = tokenizer(req, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_class = torch.argmax(logits, dim=1).item()
            label = "functional" if predicted_class == 1 else "non-functional"
        results.append({"requirement": req, "type": label})
    return results


def prioritize_requirements(classified):
    prioritized = []
    for req in classified:
        text = req["requirement"].lower()
        high_keywords = ["phải", "bắt buộc", "quan trọng", "yêu cầu", "không được phép", "bắt buộc phải"]
        medium_keywords = ["nên", "có thể", "khuyến nghị", "mong muốn"]
        if any(kw in text for kw in high_keywords):
            priority = "high"
        elif any(kw in text for kw in medium_keywords):
            priority = "medium"
        else:
            priority = "low"
        prioritized.append({**req, "priority": priority})
    return prioritized
