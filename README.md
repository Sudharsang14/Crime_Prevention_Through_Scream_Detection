# Scream Detection (Crime Prevention) — SVM + MLP

This project detects screams and classifies risk using two models:

* **SVM** and **MLP (Multilayer Perceptron)** trained on **MFCC** features.
* **Risk policy**:

  * **High Risk** — both models predict **positive scream**.
  * **Medium Risk** — exactly one model predicts **positive scream**.
  * **No Risk** — neither model predicts positive.

## Quick Start (Windows, one click)

You don’t need to add any data — the system is ready to run.
Simply double-click **`run_project.bat`** to start.

The script will:

* Create a virtual environment (if missing)
* Install dependencies
* Load the pre-trained models automatically
* Launch the web app at `http://127.0.0.1:5000`

## Web App

* **Detection page** (`/`) — record in browser or upload a `.wav` file, then **Detect**.
* **Recordings page** (`/recordings`) — browse past uploaded/recorded screams and results.

### Telegram Alert Feature

When a **High Risk** or **Medium Risk** scream is detected, a **risk alert message** along with the **location** is automatically sent to a **registered number** via Telegram.

To enable this feature:

1. Create a Telegram bot using **@BotFather** and get your **bot token**.
2. Get your **chat ID** by messaging the bot or using a bot info service.
3. Open `app.py` and set:

   ```python
   TELEGRAM_BOT_TOKEN = "your_bot_token_here"
   TELEGRAM_CHAT_ID = "your_chat_id_here"
   ```
4. When detection occurs:

   * For **High Risk** or **Medium Risk**, a Telegram message containing the **risk type** and **location** will be sent automatically.

## Project Layout

scream-detection-project/
├─ app.py
├─ requirements.txt
├─ run_project.bat
├─ README.md
├─ models/            (contains pre-trained models & scaler)
├─ data/              (only used if you want to retrain the model)
│  ├─ positive/       (put positive scream .wav files here)
│  └─ negative/       (put non-scream or normal audio here)
├─ recordings/        (auto-filled with uploaded/recorded audio + results.csv)
├─ scripts/
│  ├─ extract_features.py
│  └─ train_models.py
├─ utils/
│  └─ audio_features.py
├─ templates/
│  ├─ index.html
│  └─ recordings.html
└─ static/
├─ css/style.css
└─ js/app.js

## Notes

* Audio is automatically re-sampled to its native sample rate.
* MFCCs use `n_mfcc=40` with mean and standard deviation pooling (80 features total).
* Models use **StandardScaler** for normalization.

## Re-training (Optional)

If you want to retrain the model using your own audio:

1. Add your `.wav` files to:

   * `data/positive/` — risky/positive screams
   * `data/negative/` — safe/non-scream sounds
2. Delete existing `scream_features.csv` and `models/` (optional).
3. Run:

   ```bash
   python scripts/extract_features.py
   python scripts/train_models.py
   ```
4. Restart `app.py`.
#
