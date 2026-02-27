"""
Flask API server for the ML-Based Symptom Pattern Classification System.
Serves API endpoints (predict, diseases, stats, bias, ndcg) and frontend.
"""

import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from model import SymptomClassifier
from dataset import get_disease_info

app = Flask(__name__, static_folder=".", static_url_path="")
CORS(app)

print("Initializing symptom classifier...")
classifier = SymptomClassifier()
disease_info = get_disease_info()
print("Classifier ready!")


# ─── API Routes ──────────────────────────────────────────────────────────────

@app.route("/api/predict", methods=["POST"])
def predict():
    """Accept symptom text + optional EHR fields, return ranked disease predictions."""
    data = request.get_json()
    if not data or "symptoms" not in data:
        return jsonify({"error": "Please provide 'symptoms' in the request body."}), 400

    symptoms = data["symptoms"].strip()
    if len(symptoms) < 3:
        return jsonify({"error": "Please provide a more detailed symptom description."}), 400

    # EHR demographic fields (optional)
    age = data.get("age")
    sex = data.get("sex")
    medical_history = data.get("medical_history")
    top_k = data.get("top_k", 5)

    if age:
        try:
            age = int(age)
        except (ValueError, TypeError):
            age = None

    predictions = classifier.predict(
        symptoms, top_k=top_k, age=age, sex=sex, medical_history=medical_history
    )

    enriched = []
    for pred in predictions:
        info = disease_info.get(pred["disease"], {})
        enriched.append({
            "disease": pred["disease"],
            "confidence": pred["confidence"],
            "category": info.get("category", "Unknown"),
            "description": info.get("description", ""),
            "severity": info.get("severity", "Unknown"),
            "seek_care": info.get("seek_care", ""),
        })

    ehr_context = {}
    if age: ehr_context["age"] = age
    if sex: ehr_context["sex"] = sex
    if medical_history: ehr_context["medical_history"] = medical_history

    return jsonify({
        "predictions": enriched,
        "input_symptoms": symptoms,
        "ehr_context": ehr_context if ehr_context else None,
        "disclaimer": "This is an AI-based screening tool for informational purposes only. "
                      "It is NOT a substitute for professional medical advice, diagnosis, or treatment.",
    })


@app.route("/api/diseases", methods=["GET"])
def get_diseases():
    """Return list of all diseases the model can classify."""
    diseases = classifier.get_all_diseases()
    categorized = {}
    for d in diseases:
        info = disease_info.get(d, {})
        cat = info.get("category", "Other")
        if cat not in categorized:
            categorized[cat] = []
        categorized[cat].append({"name": d, "severity": info.get("severity", "Unknown")})
    return jsonify({"total": len(diseases), "categories": categorized})


@app.route("/api/stats", methods=["GET"])
def get_stats():
    """Return model performance statistics."""
    metrics = classifier.get_metrics()
    return jsonify({
        "paper_results": {
            "ml_system": {"m1_accuracy": 91.7, "f1_score": 0.87, "ndcg": 0.93},
            "human_physicians": {"m1_accuracy": 88.2, "f1_score": 0.89, "ndcg": 0.82},
            "lowest_rule_based": {"m1_accuracy": 32.4, "f1_score": 0.41, "ndcg": 0.45},
        },
        "live_model": {
            "m1_accuracy": metrics.get("m1_accuracy", 0),
            "f1_score": metrics.get("f1_score", 0),
            "precision": metrics.get("precision", 0),
            "recall": metrics.get("recall", 0),
            "ndcg": metrics.get("ndcg", 0),
            "train_size": metrics.get("train_size", 0),
            "test_size": metrics.get("test_size", 0),
        },
        "model_info": {
            "type": "Calibrated Ensemble (RF + SVM + GB)",
            "calibration": "Sigmoid (Platt Scaling)",
            "preprocessing": "TF-IDF with NLP + EHR Context",
            "diseases_covered": len(classifier.get_all_diseases()),
        },
    })


@app.route("/api/bias", methods=["GET"])
def get_bias():
    """Return per-category bias analysis from training evaluation."""
    bias_report = classifier.get_bias_report()
    return jsonify({
        "bias_report": bias_report,
        "description": "Accuracy breakdown by disease category on held-out test set. "
                       "Categories with lower accuracy may indicate demographic or data bias.",
        "total_categories": len(bias_report),
    })


@app.route("/api/ndcg", methods=["GET"])
def get_ndcg():
    """Return NDCG evaluation metrics."""
    metrics = classifier.get_metrics()
    return jsonify({
        "ndcg_score": metrics.get("ndcg", 0),
        "description": "Normalized Discounted Cumulative Gain measures the quality of "
                       "ranked differential diagnoses. A score of 1.0 means perfect ranking.",
        "formula": "NDCG = DCG / IDCG, where DCG = Σ(rel_i / log2(i+1))",
        "comparison": {
            "our_system_paper": 0.93,
            "our_system_live": metrics.get("ndcg", 0),
            "human_physicians": 0.82,
            "rule_based_tool": 0.45,
        },
        "interpretation": {
            "0.9-1.0": "Excellent — near-perfect ranking",
            "0.7-0.9": "Good — reliable differential diagnosis",
            "0.5-0.7": "Fair — some ranking errors",
            "below_0.5": "Poor — unreliable rankings",
        },
    })


# ─── Frontend Routes ────────────────────────────────────────────────────────

@app.route("/")
def serve_index():
    return send_from_directory(".", "index.html")

@app.route("/<path:path>")
def serve_static(path):
    return send_from_directory(".", path)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5050)
