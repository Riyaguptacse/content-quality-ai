from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.extract import extract_text_from_url
from src.preprocess import clean_text
from src.model_io import load_artifacts
from src.readability import flesch_reading_ease, flesch_kincaid_grade

app = FastAPI(title="AI Content Quality Analyzer")

class AnalyzeRequest(BaseModel):
    url: str | None = None
    text: str | None = None

@app.get("/health")
def health():
    return {"status": "ok"}

def quality_score(prob_quality: float, flesch: float) -> int:
    """
    Score 0–100.
    - Base score from model confidence.
    - Readability only boosts score when model already believes it's high quality.
    - If model believes it's low quality, readability does not inflate the score.
    """
    base = prob_quality * 100.0

    if prob_quality >= 0.5:
        # allow a small readability boost for likely high-quality content
        bonus = max(min((flesch - 50) * 0.4, 10), -10)  # -10..+10
        score = base + bonus
    else:
        # do not boost spam/low-quality based on readability
        score = base

    return int(max(0, min(100, round(score))))

def top_terms_explanation(model, text: str, top_k: int = 6):
    """
    Lightweight explanation for LogisticRegression pipeline:
    show top contributing positive/negative terms.
    """
    try:
        tfidf = model.named_steps["tfidf"]
        clf = model.named_steps["clf"]
        X = tfidf.transform([text])  # sparse row
        feature_names = tfidf.get_feature_names_out()
        coefs = clf.coef_[0]  # binary classification

        # contribution = tfidf_value * coef
        row = X.tocoo()
        contribs = []
        for j, v in zip(row.col, row.data):
            contribs.append((feature_names[j], float(v * coefs[j])))

        contribs.sort(key=lambda x: x[1], reverse=True)
        positives = [t for t, c in contribs[:top_k]]
        negatives = [t for t, c in contribs[-top_k:]][::-1]

        return {"top_positive_terms": positives, "top_negative_terms": negatives}
    except Exception:
        return {"top_positive_terms": [], "top_negative_terms": []}

@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    if not req.url and not req.text:
        raise HTTPException(status_code=400, detail="Provide either 'url' or 'text'")

    text = req.text
    if req.url:
        text = extract_text_from_url(req.url)

    text_clean = clean_text(text)

    model = load_artifacts("quality_model.joblib")
    prob_quality = float(model.predict_proba([text_clean])[0][1])
    label = "high_quality" if prob_quality >= 0.5 else "low_quality"

    flesch = float(flesch_reading_ease(text_clean))
    fk_grade = float(flesch_kincaid_grade(text_clean))
    score = quality_score(prob_quality, flesch)

    explanation = top_terms_explanation(model, text_clean)

    return {
        "label": label,
        "prob_quality": prob_quality,
        "quality_score": score,
        "readability": {
            "flesch_reading_ease": flesch,
            "flesch_kincaid_grade": fk_grade
        },
        "explanation": explanation
    }
