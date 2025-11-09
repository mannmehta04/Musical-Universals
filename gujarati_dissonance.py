import os
import numpy as np
import librosa
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

DATASET_PATH = "personal_dataset"
DURATION = 0.1
DFT_SIZE = 8000
AMP_THRESHOLD = 0.05
MAX_FREQS = 200

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def dissonance_cuda(Fn, An):
    Fn, An = Fn.to(device), An.to(device)
    f1 = Fn.unsqueeze(0)
    f2 = Fn.unsqueeze(1)
    a1 = An.unsqueeze(0)
    a2 = An.unsqueeze(1)
    df = torch.abs(f2 - f1)
    s1 = 0.24 / (0.021 * f1 + 19)
    s2 = 0.24 / (0.021 * f2 + 19)
    s = (s1 + s2) / 2
    diss = a1 * a2 * (torch.exp(-3.5 * s * df) - torch.exp(-5.75 * s * df))
    mask = torch.triu(torch.ones_like(diss), diagonal=1)
    return torch.mean(diss[mask == 1])

def process_audio(file_path):
    y, sr = librosa.load(file_path, sr=None)
    y = y / np.max(np.abs(y))
    frames = int(DURATION * sr)
    dissonance_curve = []
    for i in range(0, len(y) - frames, frames):
        segment = y[i:i+frames]
        if np.max(np.abs(segment)) < AMP_THRESHOLD:
            continue
        spectrum = np.abs(np.fft.rfft(segment, n=DFT_SIZE))
        freqs = np.fft.rfftfreq(DFT_SIZE, 1/sr)
        idx = np.argsort(spectrum)[-MAX_FREQS:]
        spectrum, freqs = spectrum[idx], freqs[idx]
        Amax, Fmax = np.max(spectrum), freqs[np.argmax(spectrum)]
        An = torch.tensor(spectrum / Amax, dtype=torch.float32, device=device)
        Fn = torch.tensor(freqs / Fmax, dtype=torch.float32, device=device)
        diss_val = dissonance_cuda(Fn, An)
        dissonance_curve.append(diss_val.item())
    return dissonance_curve if dissonance_curve else [0]

rasya_map = {
    "01": "Shant",
    "02": "Hasya",
    "03": "Bhayanak",
    "04": "Karuna",
    "05": "Rudra"
}

results = {v: [] for v in rasya_map.values()}
results["All"] = []

print(f"Using device: {device}")
print("Processing personal_dataset...\n")

for file in tqdm(os.listdir(DATASET_PATH)):
    if not file.endswith(".wav"):
        continue
    parts = file.split('-')
    if len(parts) < 4:
        continue
    rasya_id = parts[0]
    if rasya_id not in rasya_map:
        continue
    file_path = os.path.join(DATASET_PATH, file)
    diss_curve = process_audio(file_path)
    results[rasya_map[rasya_id]].append(diss_curve)
    results["All"].append(diss_curve)

# Create output folder
output_folder = "dissonance_curves"
os.makedirs(output_folder, exist_ok=True)

# Color palette
colors = {
    "Shant": "#1f77b4",
    "Hasya": "#2ca02c",
    "Bhayanak": "#d62728",
    "Karuna": "#9467bd",
    "Rudra": "#ff7f0e",
    "All": "#000000"
}

# Individual rasya plots
for rasya, curves in results.items():
    if rasya == "All" or not curves:
        continue
    max_len = max(map(len, curves))
    curves = [np.pad(c, (0, max_len - len(c)), constant_values=0) for c in curves]
    avg_curve = np.mean(np.array(curves), axis=0)
    plt.figure(figsize=(8, 5))
    plt.plot(avg_curve, color=colors[rasya], linewidth=2.5)
    plt.title(f"Dissonance Curve - {rasya} Rasya", fontsize=14, pad=10)
    plt.xlabel("Time Frames (0.1s segments)", fontsize=12)
    plt.ylabel("Average Dissonance", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"{rasya}_dissonance_curve.png"))
    plt.close()

# Combined all rasya curves in one
plt.figure(figsize=(10, 6))
for rasya, curves in results.items():
    if rasya == "All" or not curves:
        continue
    max_len = max(map(len, curves))
    curves = [np.pad(c, (0, max_len - len(c)), constant_values=0) for c in curves]
    avg_curve = np.mean(np.array(curves), axis=0)
    plt.plot(avg_curve, color=colors[rasya], linewidth=2, alpha=0.7, label=f"{rasya} Rasya")

plt.title("All Gujarati Rasyas - Dissonance Curves", fontsize=16, pad=10)
plt.xlabel("Time Frames (0.1s segments)", fontsize=12)
plt.ylabel("Average Dissonance", fontsize=12)
plt.legend(frameon=False, fontsize=11)
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "All_Rasyas_Dissonance_Curves.png"))
plt.show()

print(f"\nAll 6 dissonance curves saved in '{output_folder}' folder.")
