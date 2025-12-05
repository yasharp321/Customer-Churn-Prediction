import requests
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

DATA_URL = "https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-03-churn-prediction/WA_Fn-UseC_-Telco-Customer-Churn.csv"
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
DATA_PATH = DATA_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_DIR = Path("model_artifacts")
MODEL_DIR.mkdir(exist_ok=True)

def download_data(url=DATA_URL, dest=DATA_PATH):
    if dest.exists():
        return
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    dest.write_bytes(r.content)

def load_and_prepare(path=DATA_PATH):
    df = pd.read_csv(path)
    df = df.drop(columns=["customerID"])
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)
    df["Churn"] = df["Churn"].map({"Yes":1,"No":0})
    df = pd.get_dummies(df, drop_first=True)
    return df

def split_xy(df):
    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def train_rf(X_train, y_train):
    model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    return model

def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"accuracy: {acc:.4f}")
    print(classification_report(y_test, preds))

def save_model(model, path=MODEL_DIR/"churn_rf.joblib"):
    joblib.dump(model, path)

def main():
    download_data()
    df = load_and_prepare()
    X_train, X_test, y_train, y_test = split_xy(df)
    model = train_rf(X_train, y_train)
    evaluate(model, X_test, y_test)
    save_model(model)

if __name__ == "__main__":
    main()
