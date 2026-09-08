"""
Audio preprocessing for MagnaTagATune.
Resamples audio, extracts mel-spectrogram + chroma features, segments into windows.
"""

import librosa
import numpy as np
import pandas as pd
import os

SR = 22050
N_MELS = 128
N_CHROMA = 12
WIN_SEC = 5


def load_audio(mp3_path, sr=SR):
    """Load and resample an audio file."""
    y, _ = librosa.load(mp3_path, sr=sr)
    return y


def extract_features(y, sr=SR):
    """Extract mel-spectrogram and chroma features for a full track."""
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=N_CHROMA)
    return mel_db, chroma


def segment_features(feature_matrix, sr=SR, hop_length=512, win_sec=WIN_SEC):
    """Split a (n_features, n_frames) matrix into fixed-length time windows."""
    win_frames = int(win_sec * sr / hop_length)
    n_frames = feature_matrix.shape[1]
    segments = [
        feature_matrix[:, i:i + win_frames]
        for i in range(0, n_frames, win_frames)
        if feature_matrix[:, i:i + win_frames].shape[1] == win_frames
    ]
    return segments


def process_track(mp3_path):
    """Full pipeline for a single track: load -> features -> segment."""
    y = load_audio(mp3_path)
    mel_db, chroma = extract_features(y)
    mel_segments = segment_features(mel_db)
    chroma_segments = segment_features(chroma)
    return mel_segments, chroma_segments


if __name__ == "__main__":
    annotations = pd.read_csv("data/raw/magnatagatune_annotations.csv", sep="\t")
    sample_path = annotations.iloc[0]["mp3_path"]
    full_path = os.path.join("data/raw/magnatagatune_audio", sample_path)

    print(f"Testing on: {full_path}")
    mel_segs, chroma_segs = process_track(full_path)
    print(f"Number of mel segments: {len(mel_segs)}")
    print(f"Mel segment shape: {mel_segs[0].shape}")
    print(f"Number of chroma segments: {len(chroma_segs)}")
    print(f"Chroma segment shape: {chroma_segs[0].shape}")