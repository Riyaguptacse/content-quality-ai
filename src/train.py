import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from src.preprocess import clean_text
from src.model_io import save_artifacts

def train(csv_path="data/sample.csv"):
    df = pd.read_csv(csv_path)

    df["text"] = df["text"].astype(str).apply(clean_text)

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"],
        df["label"],
        test_size=0.25,
        random_state=42,
        stratify=df["label"]
    )
   


    pipeline = Pipeline([
         ("tfidf", TfidfVectorizer(
    max_features=20000,
    ngram_range=(1, 2),
    min_df=1,
    stop_words="english"
    )),
        ("clf", LogisticRegression(max_iter=2000))
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    save_artifacts(pipeline, "quality_model.joblib")

    print(f"Model trained & saved")
    print(f"Accuracy: {acc:.2f}")
    print(f"F1-score: {f1:.2f}")

if __name__ == "__main__":
    train()
