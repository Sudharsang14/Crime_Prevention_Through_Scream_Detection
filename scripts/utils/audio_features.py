import librosa
import numpy as np
import noisereduce as nr
import soundfile as sf

def denoise_audio(input_path: str, output_path: str = None) -> str:
    """
    Reduce background noise in the audio and save a new file.
    Returns the path to the denoised file.
    """
    y, sr = librosa.load(input_path, sr=None)
    y_denoised = nr.reduce_noise(y=y, sr=sr)
    
    if output_path is None:
        output_path = input_path.replace(".wav", "_denoised.wav")
    
    sf.write(output_path, y_denoised, sr)
    return output_path

def trim_silence(input_path: str, output_path: str = None, top_db: int = 20) -> str:
    """
    Trim leading and trailing silence from an audio file.
    """
    y, sr = librosa.load(input_path, sr=None)
    y_trimmed, _ = librosa.effects.trim(y, top_db=top_db)
    
    if output_path is None:
        output_path = input_path.replace(".wav", "_trimmed.wav")
    
    sf.write(output_path, y_trimmed, sr)
    return output_path

def extract_mfcc_vector(filepath: str, n_mfcc: int = 60, denoise: bool = True, trim: bool = True) -> np.ndarray:
    """
    Extract MFCC features (with delta and delta-delta) from an audio file.
    Optional background noise reduction and silence trimming.
    Returns a 1D feature vector (mean + std of each coefficient).
    """
    # Step 1: Denoise
    if denoise:
        filepath = denoise_audio(filepath)
    
    # Step 2: Trim silence
    if trim:
        filepath = trim_silence(filepath)
    
    # Step 3: Load audio
    y, sr = librosa.load(filepath, sr=None)
    
    # Step 4: Extract MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    
    combined = np.vstack([mfcc, mfcc_delta, mfcc_delta2])
    
    # Step 5: Aggregate mean & std
    feat_mean = np.mean(combined, axis=1)
    feat_std = np.std(combined, axis=1)
    
    feature_vector = np.hstack([feat_mean, feat_std])
    
    return feature_vector
