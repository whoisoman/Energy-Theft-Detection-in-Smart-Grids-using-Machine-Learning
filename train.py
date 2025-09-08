# train.py
import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def train_model(input_path="data/processed/features.csv",
                model_path="models/model.joblib",
                report_path="reports/metrics.txt"):
    """
    Trains a RandomForestClassifier to detect energy theft.
    """

    df = pd.read_csv(input_path, index_col=0)

    X = df.drop(columns=["label"])
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(clf, model_path)

    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        f.write(f"Accuracy: {acc:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\nConfusion Matrix:\n")
        f.write(str(cm))

    print(f" Model trained and saved to {model_path}")
    print(f" Accuracy: {acc:.4f}")
    return clf


if __name__ == "__main__":
    train_model()
