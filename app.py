from flask import Flask, render_template, request, jsonify, send_from_directory, redirect, url_for
import os
import datetime
import joblib
import numpy as np
from scripts.utils.audio_features import extract_mfcc_vector
from werkzeug.utils import secure_filename
from alert import send_telegram_alert
from location import get_location_ip
from pydub import AudioSegment  # for webm to wav conversion

# Paths
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
RECORDINGS_DIR = os.path.join(APP_ROOT, "recordings")
MODELS_DIR = os.path.join(APP_ROOT, "models")
RESULTS_CSV = os.path.join(RECORDINGS_DIR, "results.csv")
os.makedirs(RECORDINGS_DIR, exist_ok=True)

# Lazy-loaded models
scaler = None
svm_model = None
mlp_model = None

def load_models():
    global scaler, svm_model, mlp_model
    if scaler is None:
        scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    if svm_model is None:
        svm_model = joblib.load(os.path.join(MODELS_DIR, "svm_model.pkl"))
    if mlp_model is None:
        mlp_model = joblib.load(os.path.join(MODELS_DIR, "mlp_model.pkl"))

def predict_risk_from_file(filepath: str) -> dict:
    """Predict scream risk level from an audio file."""
    load_models()
    
    # Extract features with denoise + trim
    feats = extract_mfcc_vector(filepath, n_mfcc=40, denoise=True, trim=True).reshape(1, -1)
    feats_scaled = scaler.transform(feats)
    
    # Model predictions
    svm_pred = int(svm_model.predict(feats_scaled)[0])
    mlp_pred = int(mlp_model.predict(feats_scaled)[0])
    svm_prob = float(svm_model.predict_proba(feats_scaled)[0][1])
    mlp_prob = float(mlp_model.predict_proba(feats_scaled)[0][1])
    
    # Risk based on model agreement
    if svm_pred == 1 and mlp_pred == 1:
        risk = "High Risk"
    elif (svm_pred == 1) ^ (mlp_pred == 1):  # Only one model predicts positive
        risk = "Medium Risk"
    else:
        risk = "No Risk"
    
    print(f"--- Prediction for {os.path.basename(filepath)} ---")
    print(f"SVM: {svm_pred} (prob={svm_prob:.2f})")
    print(f"MLP: {mlp_pred} (prob={mlp_prob:.2f})")
    print(f"Risk Level: {risk}")
    print("---------------------------------------------------")
    
    return {
        "risk": risk,
        "svm_positive": svm_pred == 1,
        "mlp_positive": mlp_pred == 1,
        "svm_prob": svm_prob,
        "mlp_prob": mlp_prob
    }

# Flask app
app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/recordings")
def recordings():
    rows = []
    if os.path.exists(RESULTS_CSV):
        with open(RESULTS_CSV, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",", 3)
                if len(parts) == 4:
                    fname, risk, ts = parts[0], parts[1], parts[3]
                    rows.append({"file": fname, "risk": risk, "timestamp": ts})
    rows = list(reversed(rows))  # newest first
    return render_template("recordings.html", rows=rows)

@app.route("/detect", methods=["POST"])
def detect():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400
    file = request.files["audio"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    safe_name = secure_filename(file.filename)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    fname = f"{stamp}_{safe_name or 'recording.wav'}"
    save_path = os.path.join(RECORDINGS_DIR, fname)
    file.save(save_path)

    # Convert webm to wav if needed
    if save_path.lower().endswith(".webm"):
        wav_path = save_path.rsplit(".", 1)[0] + ".wav"
        audio = AudioSegment.from_file(save_path, format="webm")
        audio.export(wav_path, format="wav")
        save_path = wav_path

    try:
        result = predict_risk_from_file(save_path)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    # Send Telegram alert for High/Medium Risk
    if result["risk"] in ["High Risk", "Medium Risk"]:
        lat, lon = get_location_ip()
        send_telegram_alert(
            result["risk"], lat, lon,
            "8290795076:AAGR8Ddd80Km3CUr-fEYd3OiQt075yN7Avo",  # Telegram Bot Token
            "6648236435"                                    # Chat ID
        )

    # Save results
    ts = datetime.datetime.now().isoformat(timespec="seconds")
    with open(RESULTS_CSV, "a", encoding="utf-8") as f:
        f.write(f"{fname},{result['risk']},{ts}\n")

    return jsonify({"file": fname, **result})

@app.route("/recordings/<path:filename>")
def serve_recording(filename):
    return send_from_directory(RECORDINGS_DIR, filename, as_attachment=True)

@app.route("/recordings/delete/<path:filename>", methods=["POST"])
def delete_recording(filename):
    filepath = os.path.join(RECORDINGS_DIR, filename)
    if os.path.exists(filepath):
        os.remove(filepath)

    if os.path.exists(RESULTS_CSV):
        new_lines = []
        with open(RESULTS_CSV, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                fname_in_line = line.strip().split(",", 1)[0]
                if fname_in_line != filename:
                    new_lines.append(line)
        with open(RESULTS_CSV, "w", encoding="utf-8") as f:
            f.writelines(new_lines)

    return redirect(url_for("recordings"))

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
