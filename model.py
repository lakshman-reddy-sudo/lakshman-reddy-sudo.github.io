"""
ML-Based Symptom Pattern Classification System
4-Stage Pipeline + Enhanced Features:
  Stage 1: Raw text input ingestion (with optional EHR demographic context)
  Stage 2: NLP preprocessing (tokenization, stopword removal, TF-IDF)
  Stage 3: Calibrated ensemble classifier (RF + SVM + GB)
  Stage 4: Ranked differential diagnosis output with NDCG evaluation
  + Bias analysis across demographic groups
"""

import os
import re
import string
import joblib
import numpy as np
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
)
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
try:
    from sklearn.frozen import FrozenEstimator
    HAS_FROZEN = True
except ImportError:
    HAS_FROZEN = False
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer
    for resource in ["punkt", "punkt_tab", "stopwords", "wordnet"]:
        nltk.download(resource, quiet=True)
    STOP_WORDS = set(stopwords.words("english"))
    LEMMATIZER = WordNetLemmatizer()
    NLTK_AVAILABLE = True
except Exception:
    STOP_WORDS = {
        "i", "me", "my", "myself", "we", "our", "ours", "ourselves",
        "you", "your", "yours", "yourself", "yourselves", "he", "him",
        "his", "himself", "she", "her", "hers", "herself", "it", "its",
        "itself", "they", "them", "their", "theirs", "themselves", "what",
        "which", "who", "whom", "this", "that", "these", "those", "am",
        "is", "are", "was", "were", "be", "been", "being", "have", "has",
        "had", "having", "do", "does", "did", "doing", "a", "an", "the",
        "and", "but", "if", "or", "because", "as", "until", "while",
        "of", "at", "by", "for", "with", "about", "against", "between",
        "through", "during", "before", "after", "above", "below", "to",
        "from", "up", "down", "in", "out", "on", "off", "over", "under",
        "again", "further", "then", "once", "here", "there", "when",
        "where", "why", "how", "all", "both", "each", "few", "more",
        "most", "other", "some", "such", "no", "nor", "not", "only",
        "own", "same", "so", "than", "too", "very", "s", "t", "can",
        "will", "just", "don", "should", "now",
    }
    LEMMATIZER = None
    NLTK_AVAILABLE = False

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(MODEL_DIR, "trained_model.joblib")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib")
METRICS_PATH = os.path.join(MODEL_DIR, "training_metrics.joblib")


# ─── Stage 2: NLP Preprocessing ─────────────────────────────────────────────

def preprocess_text(text: str, age=None, sex=None, medical_history=None) -> str:
    """Stage 2 — Tokenization, stopword removal, lemmatization.
    Optionally prepend demographic/EHR context to enrich features."""
    # Prepend EHR context as additional feature tokens
    context_parts = []
    if age:
        if isinstance(age, (int, float)):
            if age < 18: context_parts.append("pediatric child")
            elif age < 40: context_parts.append("young adult")
            elif age < 60: context_parts.append("middle aged")
            else: context_parts.append("elderly geriatric")
    if sex:
        context_parts.append(sex.lower())
    if medical_history:
        if isinstance(medical_history, list):
            context_parts.extend(medical_history)
        else:
            context_parts.append(str(medical_history))

    full_text = " ".join(context_parts) + " " + text if context_parts else text
    full_text = full_text.lower()
    full_text = re.sub(r"\d+", "", full_text)
    full_text = full_text.translate(str.maketrans("", "", string.punctuation))
    full_text = re.sub(r"\s+", " ", full_text).strip()

    if NLTK_AVAILABLE:
        tokens = word_tokenize(full_text)
        tokens = [LEMMATIZER.lemmatize(t) for t in tokens if t not in STOP_WORDS and len(t) > 2]
    else:
        tokens = full_text.split()
        tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 2]

    return " ".join(tokens)


# ─── NDCG Computation ────────────────────────────────────────────────────────

def compute_dcg(relevances, k=None):
    """Compute Discounted Cumulative Gain."""
    if k: relevances = relevances[:k]
    return sum(rel / np.log2(i + 2) for i, rel in enumerate(relevances))

def compute_ndcg(y_true, y_pred_proba, classes, k=5):
    """Compute NDCG for ranked differential diagnosis evaluation.
    For each sample, measures how well the true label is ranked in predictions."""
    ndcg_scores = []
    for i in range(len(y_true)):
        true_label = y_true[i]
        probas = y_pred_proba[i]
        ranked_indices = np.argsort(probas)[::-1][:k]
        ranked_classes = [classes[j] for j in ranked_indices]

        # Relevance: 1 if matches true label, 0 otherwise
        relevances = [1.0 if c == true_label else 0.0 for c in ranked_classes]
        ideal_relevances = sorted(relevances, reverse=True)

        dcg = compute_dcg(relevances, k)
        idcg = compute_dcg(ideal_relevances, k)
        ndcg_scores.append(dcg / idcg if idcg > 0 else 0.0)

    return np.mean(ndcg_scores)


# ─── Stage 3: Ensemble ML Classification ────────────────────────────────────

def build_ensemble():
    """Build the ensemble classifier with soft voting."""
    rf = RandomForestClassifier(n_estimators=300, max_depth=None, min_samples_split=2, random_state=42, n_jobs=-1)
    svm = SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=42)
    gb = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
    ensemble = VotingClassifier(estimators=[("rf", rf), ("svm", svm), ("gb", gb)], voting="soft", weights=[2, 3, 2])
    return ensemble


def train_model():
    """Train the full pipeline with confidence calibration and save artifacts."""
    from dataset import get_training_data
    texts_raw, labels = get_training_data()
    texts = [preprocess_text(t) for t in texts_raw]

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), sublinear_tf=True)
    X = vectorizer.fit_transform(texts)
    y = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Build and train ensemble
    base_model = build_ensemble()
    base_model.fit(X_train, y_train)

    # Confidence Calibration using CalibratedClassifierCV
    if HAS_FROZEN:
        # scikit-learn >= 1.8: cv='prefit' removed, use FrozenEstimator
        calibrated_model = CalibratedClassifierCV(FrozenEstimator(base_model), method="sigmoid")
    else:
        # scikit-learn < 1.8: use cv='prefit'
        calibrated_model = CalibratedClassifierCV(base_model, cv="prefit", method="sigmoid")
    calibrated_model.fit(X_train, y_train)

    # Evaluate with calibrated model
    y_pred = calibrated_model.predict(X_test)
    y_pred_proba = calibrated_model.predict_proba(X_test)
    m1_accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)

    # Compute NDCG
    ndcg = compute_ndcg(y_test, y_pred_proba, calibrated_model.classes_, k=5)

    # Bias analysis: per-category accuracy
    from dataset import get_disease_info
    disease_info = get_disease_info()
    category_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    for true, pred in zip(y_test, y_pred):
        cat = disease_info.get(true, {}).get("category", "Other")
        category_stats[cat]["total"] += 1
        if true == pred:
            category_stats[cat]["correct"] += 1

    bias_report = {}
    for cat, stats in category_stats.items():
        acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        bias_report[cat] = {"accuracy": round(acc, 4), "samples": stats["total"]}

    print(f"\n{'='*50}")
    print(f"  Model Training Complete (Calibrated)")
    print(f"{'='*50}")
    print(f"  Training samples : {len(y_train)}")
    print(f"  Test samples     : {len(y_test)}")
    print(f"  M1 Accuracy      : {m1_accuracy:.4f} ({m1_accuracy*100:.1f}%)")
    print(f"  F1 Score (wt)    : {f1:.4f}")
    print(f"  Precision (wt)   : {precision:.4f}")
    print(f"  Recall (wt)      : {recall:.4f}")
    print(f"  NDCG@5           : {ndcg:.4f}")
    print(f"  Bias by category : {dict(bias_report)}")
    print(f"{'='*50}\n")

    # Retrain on full dataset for production
    model_full = build_ensemble()
    model_full.fit(X, y)
    calibrated_full = CalibratedClassifierCV(model_full, cv="prefit", method="sigmoid")
    calibrated_full.fit(X, y)

    metrics = {
        "m1_accuracy": round(m1_accuracy, 4),
        "f1_score": round(f1, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "ndcg": round(ndcg, 4),
        "train_size": len(y_train),
        "test_size": len(y_test),
        "bias_report": bias_report,
    }

    joblib.dump(calibrated_full, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(metrics, METRICS_PATH)
    print(f"  Model saved to: {MODEL_PATH}")
    return metrics


# ─── Stage 4: Prediction / Ranked Differential Diagnosis ────────────────────

class SymptomClassifier:
    """Loads a trained calibrated model and produces ranked differential diagnoses."""

    def __init__(self):
        if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
            print("No trained model found. Training now...")
            train_model()

        self.model = joblib.load(MODEL_PATH)
        self.vectorizer = joblib.load(VECTORIZER_PATH)
        self.classes = self.model.classes_
        self.metrics = joblib.load(METRICS_PATH) if os.path.exists(METRICS_PATH) else {}

    def predict(self, symptom_text: str, top_k: int = 5, age=None, sex=None, medical_history=None):
        """Stage 4 — Return ranked differential diagnoses with calibrated probabilities."""
        processed = preprocess_text(symptom_text, age=age, sex=sex, medical_history=medical_history)
        X = self.vectorizer.transform([processed])
        probas = self.model.predict_proba(X)[0]
        top_indices = np.argsort(probas)[::-1][:top_k]

        results = []
        for idx in top_indices:
            confidence = float(probas[idx])
            if confidence > 0.001:
                results.append({"disease": self.classes[idx], "confidence": round(confidence, 4)})
        return results

    def get_all_diseases(self):
        return sorted(self.classes.tolist())

    def get_metrics(self):
        return self.metrics

    def get_bias_report(self):
        return self.metrics.get("bias_report", {})


if __name__ == "__main__":
    # Force retrain
    if os.path.exists(MODEL_PATH): os.remove(MODEL_PATH)
    if os.path.exists(VECTORIZER_PATH): os.remove(VECTORIZER_PATH)
    if os.path.exists(METRICS_PATH): os.remove(METRICS_PATH)
    metrics = train_model()
    print(f"\nMetrics: {metrics}")

    classifier = SymptomClassifier()
    test_cases = [
        ("severe chest pain radiating to left arm sweating", 64, "male", ["hypertension", "smoker"]),
        ("high fever stiff neck headache light sensitivity", 20, "female", None),
        ("persistent cough weight loss night sweats", 45, "male", None),
        ("burning urination frequent urgency cloudy urine", 28, "female", None),
    ]
    for symptoms, age, sex, history in test_cases:
        print(f"\nInput: {symptoms} | Age: {age} | Sex: {sex}")
        results = classifier.predict(symptoms, age=age, sex=sex, medical_history=history)
        for r in results[:3]:
            print(f"  → {r['disease']}: {r['confidence']*100:.1f}%")
