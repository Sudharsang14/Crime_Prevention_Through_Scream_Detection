import os
import sys
import pandas as pd

# Add project root to sys.path so Python can find utils/
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.utils.audio_features import extract_mfcc_vector

POS_DIR = os.path.join("data", "positive")
NEG_DIR = os.path.join("data", "negative")
OUT_CSV = "scream_features.csv"

def gather_files():
    files = []
    for label, root in [(1, POS_DIR), (0, NEG_DIR)]:
        if not os.path.isdir(root):
            continue
        for fname in os.listdir(root):
            if fname.lower().endswith(('.wav', '.mp3', '.flac', '.ogg')):
                files.append((os.path.join(root, fname), label))
    return files

def main():
    os.makedirs("data", exist_ok=True)
    os.makedirs(POS_DIR, exist_ok=True)
    os.makedirs(NEG_DIR, exist_ok=True)
    rows = []
    files = gather_files()
    if not files:
        print("No audio files found. Please add files to data/positive and data/negative.")
    for path, label in files:
        try:
            feat = extract_mfcc_vector(path, n_mfcc=40)
            row = {f"f{i}": float(v) for i, v in enumerate(feat)}
            row["label"] = int(label)   # only keep features + label
            rows.append(row)
            print(f"Processed: {path}")
        except Exception as e:
            print(f"Failed: {path} ({e})")

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(OUT_CSV, index=False)
        print(f"Saved features to {OUT_CSV} with {len(df)} rows and {df.shape[1]-1} features.")
    else:
        print("No features saved.")

if __name__ == "__main__":
    main()
