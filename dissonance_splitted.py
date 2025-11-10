# import os
# import numpy as np
# import librosa
# import torch
# import matplotlib.pyplot as plt
# from tqdm import tqdm

# DATASET_PATH = "personal_dataset"
# DURATION = 0.1
# DFT_SIZE = 8000
# AMP_THRESHOLD = 0.05
# MAX_FREQS = 300
# NOTE_TOLERANCE = 0.06  # ±6% tolerance around each note frequency for binning
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# note_names = ["Sa", "Komal Re", "Re", "Komal Ga", "Ga", "Ma", "Tivra Ma", "Pa", "Komal Dha", "Dha", "Komal Ni", "Ni"]
# note_freqs = np.array([240.00, 254.00, 270.00, 286.00, 303.00, 321.00, 340.00, 360.00, 381.00, 404.00, 428.00, 454.00])

# def dissonance_fn(f1, f2, a1, a2):
#     s1 = 0.24 / (0.021 * f1 + 19)
#     s2 = 0.24 / (0.021 * f2 + 19)
#     s = (s1 + s2) / 2
#     df = abs(f2 - f1)
#     return a1 * a2 * (np.exp(-3.5 * s * df) - np.exp(-5.75 * s * df))

# def process_audio(file_path):
#     y, sr = librosa.load(file_path, sr=None)
#     y = y / np.max(np.abs(y))
#     frames = int(DURATION * sr)
#     note_dissonance_values = [[] for _ in range(12)]

#     for i in range(0, len(y) - frames, frames):
#         segment = y[i:i+frames]
#         if np.max(np.abs(segment)) < AMP_THRESHOLD:
#             continue

#         spectrum = np.abs(np.fft.rfft(segment, n=DFT_SIZE))
#         freqs = np.fft.rfftfreq(DFT_SIZE, 1/sr)
#         if np.all(spectrum == 0):
#             continue

#         idx = np.argsort(spectrum)[-MAX_FREQS:]
#         spectrum, freqs = spectrum[idx], freqs[idx]
#         Amax = np.max(spectrum)
#         if Amax <= 0:
#             continue
#         spectrum /= Amax

#         # For each note, collect dissonance within frequency window
#         for j, note_f in enumerate(note_freqs):
#             lower = note_f * (1 - NOTE_TOLERANCE)
#             upper = note_f * (1 + NOTE_TOLERANCE)
#             mask = (freqs >= lower) & (freqs <= upper)
#             if np.sum(mask) < 2:
#                 continue
#             sub_freqs, sub_amps = freqs[mask], spectrum[mask]
#             sub_diss = []
#             for k in range(len(sub_freqs)):
#                 for m in range(k+1, len(sub_freqs)):
#                     sub_diss.append(dissonance_fn(sub_freqs[k], sub_freqs[m], sub_amps[k], sub_amps[m]))
#             if sub_diss:
#                 note_dissonance_values[j].append(np.mean(sub_diss))

#     return np.array([np.median(v) if v else 0 for v in note_dissonance_values])

# rasya_map = {
#     "01": "Shant",
#     "02": "Hasya",
#     "03": "Bhayanak",
#     "04": "Karuna",
#     "05": "Rudra"
# }

# results = {v: [] for v in rasya_map.values()}

# print(f"Using device: {device}")
# print("Analyzing dissonance across 12 chromatic notes for Gujarati rasyas...\n")

# for file in tqdm(os.listdir(DATASET_PATH)):
#     if not file.endswith(".wav"):
#         continue
#     parts = file.split('-')
#     if len(parts) < 4:
#         continue
#     rasya_id = parts[0]
#     if rasya_id not in rasya_map:
#         continue
#     file_path = os.path.join(DATASET_PATH, file)
#     diss_values = process_audio(file_path)
#     results[rasya_map[rasya_id]].append(diss_values)

# output_folder = "dissonance_chromatic_spikes"
# os.makedirs(output_folder, exist_ok=True)

# colors = {
#     "Shant": "#1f77b4",
#     "Hasya": "#2ca02c",
#     "Bhayanak": "#d62728",
#     "Karuna": "#9467bd",
#     "Rudra": "#ff7f0e"
# }

# for rasya, curves in results.items():
#     if not curves:
#         continue
#     avg_curve = np.mean(np.array(curves), axis=0)
#     norm_curve = avg_curve / (np.max(avg_curve) + 1e-9)
#     norm_curve = norm_curve ** 0.7  # enhance spikes
#     plt.figure(figsize=(9, 5))
#     plt.plot(note_names, norm_curve, marker='o', linewidth=2.5, color=colors[rasya])
#     plt.title(f"{rasya} Rasya - Dissonance Across 12 Chromatic Notes", fontsize=14, pad=10)
#     plt.xlabel("Indian Chromatic Notes", fontsize=12)
#     plt.ylabel("Normalized Dissonance", fontsize=12)
#     plt.grid(True, linestyle="--", alpha=0.6)
#     plt.tight_layout()
#     plt.savefig(os.path.join(output_folder, f"{rasya}_Chromatic_Dissonance_Spikes.png"))
#     plt.close()

# plt.figure(figsize=(10, 6))
# for rasya, curves in results.items():
#     if not curves:
#         continue
#     avg_curve = np.mean(np.array(curves), axis=0)
#     norm_curve = avg_curve / (np.max(avg_curve) + 1e-9)
#     norm_curve = norm_curve ** 0.7
#     plt.plot(note_names, norm_curve, marker='o', linewidth=2.2, color=colors[rasya], alpha=0.85, label=rasya)

# plt.title("Gujarati Rasyas - Chromatic Dissonance Profile (Spiky View)", fontsize=16, pad=10)
# plt.xlabel("12 Chromatic Notes", fontsize=12)
# plt.ylabel("Normalized Dissonance", fontsize=12)
# plt.legend(frameon=False, fontsize=11)
# plt.grid(True, linestyle="--", alpha=0.6)
# plt.tight_layout()
# plt.savefig(os.path.join(output_folder, "All_Rasyas_Chromatic_Spikes.png"))
# plt.show()

# print(f"\nAll dissonance curves with visible spikes saved in '{output_folder}' folder.")



#=========================================================================================================#
# Normalised Curves for Dissonance Graph



# import os
# import numpy as np
# import librosa
# import torch
# import matplotlib.pyplot as plt
# from tqdm import tqdm

# DATASET_PATH = "personal_dataset"
# DURATION = 0.1
# DFT_SIZE = 8000
# AMP_THRESHOLD = 0.05
# MAX_FREQS = 300
# NOTE_TOLERANCE = 0.06

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# note_names = ["Sa", "Komal Re", "Re", "Komal Ga", "Ga", "Ma", "Tivra Ma", "Pa", "Komal Dha", "Dha", "Komal Ni", "Ni"]
# note_freqs = np.array([240.00, 254.00, 270.00, 286.00, 303.00, 321.00, 340.00, 360.00, 381.00, 404.00, 428.00, 454.00])
# note_freqs = np.array([240.00, 256.00, 270.00, 288.00, 300.00, 320.00, 337.50, 360.00, 384.00, 400.00, 432.00, 450.00])

# def dissonance_fn(f1, f2, a1, a2):
#     s1 = 0.24 / (0.021 * f1 + 19)
#     s2 = 0.24 / (0.021 * f2 + 19)
#     s = (s1 + s2) / 2
#     df = abs(f2 - f1)
#     return a1 * a2 * (np.exp(-3.5 * s * df) - np.exp(-5.75 * s * df))

# def process_audio(file_path):
#     y, sr = librosa.load(file_path, sr=None)
#     y = y / np.max(np.abs(y))
#     frames = int(DURATION * sr)
#     note_dissonance_values = [[] for _ in range(12)]

#     for i in range(0, len(y) - frames, frames):
#         segment = y[i:i+frames]
#         if np.max(np.abs(segment)) < AMP_THRESHOLD:
#             continue

#         spectrum = np.abs(np.fft.rfft(segment, n=DFT_SIZE))
#         freqs = np.fft.rfftfreq(DFT_SIZE, 1/sr)
#         if np.all(spectrum == 0):
#             continue

#         idx = np.argsort(spectrum)[-MAX_FREQS:]
#         spectrum, freqs = spectrum[idx], freqs[idx]
#         Amax = np.max(spectrum)
#         Fmax = freqs[np.argmax(spectrum)]
#         if Amax <= 0 or Fmax <= 0:
#             continue

#         # Normalization based on Fn = F/Fmax and An = A/Amax
#         Fn = freqs / Fmax
#         An = spectrum / Amax

#         for j, note_f in enumerate(note_freqs):
#             lower = note_f * (1 - NOTE_TOLERANCE)
#             upper = note_f * (1 + NOTE_TOLERANCE)
#             mask = (freqs >= lower) & (freqs <= upper)
#             if np.sum(mask) < 2:
#                 continue
#             sub_freqs, sub_amps = Fn[mask], An[mask]
#             sub_diss = []
#             for k in range(len(sub_freqs)):
#                 for m in range(k+1, len(sub_freqs)):
#                     sub_diss.append(dissonance_fn(sub_freqs[k], sub_freqs[m], sub_amps[k], sub_amps[m]))
#             if sub_diss:
#                 note_dissonance_values[j].append(np.mean(sub_diss))

#     return np.array([np.median(v) if v else 0 for v in note_dissonance_values])

# rasya_map = {
#     "01": "Shant",
#     "02": "Hasya",
#     "03": "Bhayanak",
#     "04": "Karuna",
#     "05": "Rudra"
# }

# results = {v: [] for v in rasya_map.values()}

# print(f"Using device: {device}")
# print("Analyzing normalized dissonance across 12 chromatic notes for Gujarati rasyas...\n")

# for file in tqdm(os.listdir(DATASET_PATH)):
#     if not file.endswith(".wav"):
#         continue
#     parts = file.split('-')
#     if len(parts) < 4:
#         continue
#     rasya_id = parts[0]
#     if rasya_id not in rasya_map:
#         continue
#     file_path = os.path.join(DATASET_PATH, file)
#     diss_values = process_audio(file_path)
#     results[rasya_map[rasya_id]].append(diss_values)

# output_folder = "dissonance_chromatic_normalized"
# os.makedirs(output_folder, exist_ok=True)

# colors = {
#     "Shant": "#1f77b4",
#     "Hasya": "#2ca02c",
#     "Bhayanak": "#d62728",
#     "Karuna": "#9467bd",
#     "Rudra": "#ff7f0e"
# }

# for rasya, curves in results.items():
#     if not curves:
#         continue
#     avg_curve = np.mean(np.array(curves), axis=0)
#     norm_curve = avg_curve / (np.max(avg_curve) + 1e-9)
#     plt.figure(figsize=(9, 5))
#     plt.plot(note_names, norm_curve, marker='o', linewidth=2.5, color=colors[rasya])
#     plt.title(f"{rasya} Rasya - Normalized Dissonance Across 12 Chromatic Notes", fontsize=14, pad=10)
#     plt.xlabel("Indian Chromatic Notes", fontsize=12)
#     plt.ylabel("Normalized Dissonance (Fn & An)", fontsize=12)
#     plt.grid(True, linestyle="--", alpha=0.6)
#     plt.tight_layout()
#     plt.savefig(os.path.join(output_folder, f"{rasya}_Chromatic_Normalized.png"))
#     plt.close()

# plt.figure(figsize=(10, 6))
# for rasya, curves in results.items():
#     if not curves:
#         continue
#     avg_curve = np.mean(np.array(curves), axis=0)
#     norm_curve = avg_curve / (np.max(avg_curve) + 1e-9)
#     plt.plot(note_names, norm_curve, marker='o', linewidth=2.2, color=colors[rasya], alpha=0.85, label=rasya)

# plt.title("Gujarati Rasyas - Normalized Chromatic Dissonance Profiles (Fn & An)", fontsize=16, pad=10)
# plt.xlabel("12 Chromatic Notes", fontsize=12)
# plt.ylabel("Normalized Dissonance (Fn & An)", fontsize=12)
# plt.legend(frameon=False, fontsize=11)
# plt.grid(True, linestyle="--", alpha=0.6)
# plt.tight_layout()
# plt.savefig(os.path.join(output_folder, "All_Rasyas_Chromatic_Normalized.png"))
# plt.show()

# print(f"\nAll normalized dissonance curves saved in '{output_folder}' folder.")







#=========================================================================================================#
# Splitted Male Female Normalised Curves for Dissonance Graph

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
MAX_FREQS = 300
NOTE_TOLERANCE = 0.06

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

note_names = ["Sa", "Komal Re", "Re", "Komal Ga", "Ga", "Ma", "Tivra Ma", "Pa", "Komal Dha", "Dha", "Komal Ni", "Ni"]
# note_freqs = np.array([240.00, 254.00, 270.00, 286.00, 303.00, 321.00, 340.00, 360.00, 381.00, 404.00, 428.00, 454.00])
note_freqs = np.array([240.00, 256.00, 270.00, 288.00, 300.00, 320.00, 337.50, 360.00, 384.00, 400.00, 432.00, 450.00])

def dissonance_fn(f1, f2, a1, a2):
    s1 = 0.24 / (0.021 * f1 + 19)
    s2 = 0.24 / (0.021 * f2 + 19)
    s = (s1 + s2) / 2
    df = abs(f2 - f1)
    return a1 * a2 * (np.exp(-3.5 * s * df) - np.exp(-5.75 * s * df))

def process_audio(file_path):
    y, sr = librosa.load(file_path, sr=None)
    y = y / np.max(np.abs(y))
    frames = int(DURATION * sr)
    note_dissonance_values = [[] for _ in range(12)]

    for i in range(0, len(y) - frames, frames):
        segment = y[i:i+frames]
        if np.max(np.abs(segment)) < AMP_THRESHOLD:
            continue

        spectrum = np.abs(np.fft.rfft(segment, n=DFT_SIZE))
        freqs = np.fft.rfftfreq(DFT_SIZE, 1/sr)
        if np.all(spectrum == 0):
            continue

        idx = np.argsort(spectrum)[-MAX_FREQS:]
        spectrum, freqs = spectrum[idx], freqs[idx]
        Amax = np.max(spectrum)
        Fmax = freqs[np.argmax(spectrum)]
        if Amax <= 0 or Fmax <= 0:
            continue

        # Normalization based on Fn = F/Fmax and An = A/Amax
        Fn = freqs / Fmax
        An = spectrum / Amax

        for j, note_f in enumerate(note_freqs):
            lower = note_f * (1 - NOTE_TOLERANCE)
            upper = note_f * (1 + NOTE_TOLERANCE)
            mask = (freqs >= lower) & (freqs <= upper)
            if np.sum(mask) < 2:
                continue
            sub_freqs, sub_amps = Fn[mask], An[mask]
            sub_diss = []
            for k in range(len(sub_freqs)):
                for m in range(k+1, len(sub_freqs)):
                    sub_diss.append(dissonance_fn(sub_freqs[k], sub_freqs[m], sub_amps[k], sub_amps[m]))
            if sub_diss:
                note_dissonance_values[j].append(np.mean(sub_diss))

    return np.array([np.median(v) if v else 0 for v in note_dissonance_values])

rasya_map = {
    "01": "Shant",
    "02": "Hasya",
    "03": "Bhayanak",
    "04": "Karuna",
    "05": "Rudra"
}
gender_map = {"00": "Male", "01": "Female"}

results = {f"{r}_{g}": [] for r in rasya_map.values() for g in gender_map.values()}

print(f"Using device: {device}")
print("Analyzing normalized dissonance across 12 chromatic notes by gender for Gujarati rasyas...\n")

for file in tqdm(os.listdir(DATASET_PATH)):
    if not file.endswith(".wav"):
        continue
    parts = file.split('-')
    if len(parts) < 4:
        continue
    rasya_id, gender_id = parts[0], parts[1]
    if rasya_id not in rasya_map or gender_id not in gender_map:
        continue
    key = f"{rasya_map[rasya_id]}_{gender_map[gender_id]}"
    file_path = os.path.join(DATASET_PATH, file)
    diss_values = process_audio(file_path)
    results[key].append(diss_values)

output_folder = "dissonance_chromatic_normalized_gendered"
os.makedirs(output_folder, exist_ok=True)

colors = {
    "Shant_Male": "#1f77b4", "Shant_Female": "#aec7e8",
    "Hasya_Male": "#2ca02c", "Hasya_Female": "#98df8a",
    "Bhayanak_Male": "#d62728", "Bhayanak_Female": "#ff9896",
    "Karuna_Male": "#9467bd", "Karuna_Female": "#c5b0d5",
    "Rudra_Male": "#ff7f0e", "Rudra_Female": "#ffbb78"
}

# Plot male vs female for each rasya
for rasya in rasya_map.values():
    plt.figure(figsize=(9, 5))
    for gender in gender_map.values():
        key = f"{rasya}_{gender}"
        curves = results[key]
        if not curves:
            continue
        avg_curve = np.mean(np.array(curves), axis=0)
        norm_curve = avg_curve / (np.max(avg_curve) + 1e-9)
        plt.plot(note_names, norm_curve, marker='o', linewidth=2.5, color=colors[key], label=f"{gender}")
    plt.title(f"{rasya} Rasya - Normalized Dissonance (Male vs Female)", fontsize=14, pad=10)
    plt.xlabel("Indian Chromatic Notes", fontsize=12)
    plt.ylabel("Normalized Dissonance (Fn & An)", fontsize=12)
    plt.legend(frameon=False, fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"{rasya}_Male_Female_Chromatic_Normalized.png"))
    plt.close()

# Combined plot (10 curves total)
plt.figure(figsize=(10, 6))
for key, curves in results.items():
    if not curves:
        continue
    avg_curve = np.mean(np.array(curves), axis=0)
    norm_curve = avg_curve / (np.max(avg_curve) + 1e-9)
    plt.plot(note_names, norm_curve, marker='o', linewidth=2.2, color=colors[key], alpha=0.85, label=key.replace("_", " "))

plt.title("Gujarati Rasyas - Normalized Chromatic Dissonance (Male & Female)", fontsize=16, pad=10)
plt.xlabel("12 Chromatic Notes", fontsize=12)
plt.ylabel("Normalized Dissonance (Fn & An)", fontsize=12)
plt.legend(frameon=False, fontsize=10, ncol=2)
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "All_Rasyas_Male_Female_Chromatic_Normalized.png"))
plt.show()

print(f"\nAll gender-separated normalized dissonance curves saved in '{output_folder}' folder.")
