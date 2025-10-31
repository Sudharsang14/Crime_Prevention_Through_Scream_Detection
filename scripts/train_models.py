import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score

CSV = "scream_features.csv"
MODELS_DIR = "models"

def main():
    if not os.path.exists(CSV):
        raise FileNotFoundError("scream_features.csv not found. Run scripts/extract_features.py first.")
    df = pd.read_csv(CSV)
    feature_cols = [c for c in df.columns if c.startswith("f")]
    X = df[feature_cols].values
    y = df["label"].values
    print(df["label"].value_counts())

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split for quick metrics
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

    svm_model = SVC(probability=True, kernel="rbf", C=2.0, gamma="scale", random_state=42)
    mlp_model = MLPClassifier(hidden_layer_sizes=(128, 64), activation="relu", solver="adam",
                              batch_size=32, max_iter=500, random_state=42, early_stopping=True)

    svm_model.fit(X_train, y_train)
    mlp_model.fit(X_train, y_train)

    # Eval
    for name, model in [("SVM", svm_model), ("MLP", mlp_model)]:
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"{name} Accuracy: {acc:.3f}")
        print(classification_report(y_test, preds, digits=3))

    # Save
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
    joblib.dump(svm_model, os.path.join(MODELS_DIR, "svm_model.pkl"))
    joblib.dump(mlp_model, os.path.join(MODELS_DIR, "mlp_model.pkl"))
    print(f"Saved models to {MODELS_DIR}/")

if __name__ == "__main__":
    main()