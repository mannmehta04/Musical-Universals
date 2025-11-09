# import os
# import numpy as np
# import librosa
# import matplotlib.pyplot as plt

# # ============= Environment =============
# class RavdessEnv:
#     def __init__(self, dataset_path, sr=16000):
#         self.dataset_path = dataset_path
#         self.sr = sr

#     def _is_emotion_file(self, file_path, emotion_code):
#         fname = os.path.basename(file_path)
#         parent = os.path.basename(os.path.dirname(file_path))
#         if f"-{emotion_code}-" in fname:
#             return True
#         if emotion_code in parent:
#             return True
#         return False

#     def list_emotion_files(self, emotion_code):
#         wavs = []
#         for root, _, files in os.walk(self.dataset_path):
#             for f in files:
#                 if not f.lower().endswith(".wav"):
#                     continue
#                 full = os.path.join(root, f)
#                 if self._is_emotion_file(full, emotion_code):
#                     wavs.append(full)
#         return sorted(wavs)

#     def load_audio(self, path):
#         y, sr = librosa.load(path, sr=self.sr, mono=True)
#         return y

#     def extract_segments(self, signal, n_segments=40, seg_duration=0.1):
#         seg_len = int(seg_duration * self.sr)
#         total_len = len(signal)
#         segments = []
#         for i in range(n_segments):
#             start = i * seg_len
#             end = start + seg_len
#             if end > total_len:
#                 break
#             segments.append(signal[start:end])
#         return segments

#     def normalized_spectra_from_file(self, path, n_segments=40, seg_duration=0.1, fft_size=8192):
#         signal = self.load_audio(path)
#         max_amp = np.max(np.abs(signal))
#         segments = self.extract_segments(signal, n_segments, seg_duration)

#         results = []
#         for seg in segments:
#             if len(seg) < int(seg_duration * self.sr):
#                 continue
#             if np.max(np.abs(seg)) < 0.05 * max_amp:
#                 spectrum = np.fft.rfft(seg, n=fft_size)
#                 freqs = np.fft.rfftfreq(fft_size, d=1/self.sr)
#                 amps = np.abs(spectrum)
#                 A_max = np.max(amps)
#                 F_max = freqs[np.argmax(amps)]
#                 if A_max > 0 and F_max > 0:
#                     f_norm = freqs / F_max
#                     A_norm = amps / A_max
#                     results.append((f_norm, A_norm))
#         return results

#     def normalized_spectra_for_emotion(self, emotion_code, n_segments=40, seg_duration=0.1, fft_size=8192):
#         files = self.list_emotion_files(emotion_code)
#         spectra = []
#         for f in files:
#             seg_spectra = self.normalized_spectra_from_file(f, n_segments, seg_duration, fft_size)
#             spectra.extend(seg_spectra)
#         return spectra

# # ============= Agent =============
# class FFTAgent:
#     def __init__(self, env: RavdessEnv):
#         self.env = env

#     def plot_emotion(self, emotion_code="03", n_segments=40, seg_duration=0.1, fft_size=8192, max_traces=20):
#         spectra = self.env.normalized_spectra_for_emotion(emotion_code, n_segments, seg_duration, fft_size)
#         if not spectra:
#             raise RuntimeError(f"No spectra found for emotion code {emotion_code}")

#         plt.figure(figsize=(12, 7))
#         step = max(1, len(spectra) // max_traces)
#         for idx in range(0, len(spectra), step):
#             f_norm, A_norm = spectra[idx]
#             plt.plot(f_norm, A_norm, alpha=0.3, linewidth=0.7, label="_nolegend_")
#         plt.xlabel("f / F_max")
#         plt.ylabel("A / A_max")
#         plt.title(f"Normalized Spectra for Emotion {emotion_code}")
#         plt.grid(True, ls='--', alpha=0.4)
#         plt.tight_layout()
#         plt.show()

#         return spectra

# # ============= Example usage =============
# if __name__ == "__main__":
#     dataset_path = r"dataset"  # <-- update path

#     env = RavdessEnv(dataset_path=dataset_path, sr=16000)
#     agent = FFTAgent(env)

#     # Example: plot normalized spectra for Happy (03)
#     spectra_happy = agent.plot_emotion(emotion_code="03", n_segments=40, seg_duration=0.1, fft_size=8192)

#     # Example: plot normalized spectra for Sad (04)
#     spectra_sad = agent.plot_emotion(emotion_code="04", n_segments=40, seg_duration=0.1, fft_size=8192)



# import os
# import numpy as np
# import librosa
# import matplotlib.pyplot as plt
# from scipy.signal import find_peaks

# # ============= Environment =============
# class RavdessEnv:
#     def __init__(self, dataset_path, sr=16000):
#         self.dataset_path = dataset_path
#         self.sr = sr

#     def _is_emotion_file(self, file_path, emotion_code):
#         fname = os.path.basename(file_path)
#         parent = os.path.basename(os.path.dirname(file_path))
#         if f"-{emotion_code}-" in fname:
#             return True
#         if emotion_code in parent:
#             return True
#         return False

#     def list_emotion_files(self, emotion_code):
#         wavs = []
#         for root, _, files in os.walk(self.dataset_path):
#             for f in files:
#                 if not f.lower().endswith(".wav"):
#                     continue
#                 full = os.path.join(root, f)
#                 if self._is_emotion_file(full, emotion_code):
#                     wavs.append(full)
#         return sorted(wavs)

#     def load_audio(self, path):
#         y, sr = librosa.load(path, sr=self.sr, mono=True)
#         return y

#     def extract_segments(self, signal, n_segments=40, seg_duration=0.1):
#         seg_len = int(seg_duration * self.sr)
#         total_len = len(signal)
#         segments = []
#         for i in range(n_segments):
#             start = i * seg_len
#             end = start + seg_len
#             if end > total_len:
#                 break
#             segments.append(signal[start:end])
#         return segments

#     def spectra_from_file(self, path, n_segments=40, seg_duration=0.1, fft_size=8192):
#         signal = self.load_audio(path)
#         max_amp = np.max(np.abs(signal))
#         segments = self.extract_segments(signal, n_segments, seg_duration)

#         spectra = []
#         for seg in segments:
#             if len(seg) < int(seg_duration * self.sr):
#                 continue
#             if np.max(np.abs(seg)) < 0.05 * max_amp:
#                 spectrum = np.fft.rfft(seg, n=fft_size)
#                 freqs = np.fft.rfftfreq(fft_size, d=1/self.sr)
#                 amps = np.abs(spectrum)
#                 spectra.append((freqs, amps))
#         return spectra

#     def spectra_for_emotion(self, emotion_code, n_segments=40, seg_duration=0.1, fft_size=8192):
#         files = self.list_emotion_files(emotion_code)
#         spectra = []
#         for f in files:
#             spectra.extend(self.spectra_from_file(f, n_segments, seg_duration, fft_size))
#         return spectra

# # ============= Dissonance model (Sethares 1993) =============
# def diss_measure(freqs, amps):
#     # take top spectral peaks
#     peaks, _ = find_peaks(amps, height=np.max(amps)*0.2)
#     f_peaks = freqs[peaks]
#     a_peaks = amps[peaks]
#     if len(f_peaks) < 2:
#         return 0.0

#     # normalize amplitudes
#     a_peaks = a_peaks / np.max(a_peaks)

#     # constants from Sethares
#     b1, b2 = 3.5, 5.75
#     d_star = 0.24
#     s1 = 0.0207
#     s2 = 18.96

#     D = 0.0
#     for i in range(len(f_peaks)):
#         for j in range(i+1, len(f_peaks)):
#             f1, f2 = f_peaks[i], f_peaks[j]
#             a1, a2 = a_peaks[i], a_peaks[j]
#             min_f = min(f1, f2)
#             s = d_star / (s1*min_f + s2)
#             x = abs(f2 - f1) * s
#             D += a1 * a2 * (np.exp(-b1*x) - np.exp(-b2*x))
#     return D

# # ============= Agent =============
# class DissonanceAgent:
#     def __init__(self, env: RavdessEnv):
#         self.env = env

#     def dissonance_curve(self, emotion_code="03", n_segments=40, seg_duration=0.1, fft_size=8192):
#         spectra = self.env.spectra_for_emotion(emotion_code, n_segments, seg_duration, fft_size)
#         curve = []
#         for freqs, amps in spectra:
#             curve.append(diss_measure(freqs, amps))
#         return curve

#     def plot_curve(self, emotion_code="03", **kwargs):
#         curve = self.dissonance_curve(emotion_code, **kwargs)
#         if not curve:
#             raise RuntimeError("No dissonance values computed")
#         plt.figure(figsize=(10,5))
#         plt.plot(curve, marker='o', alpha=0.7)
#         plt.xlabel("Segment index")
#         plt.ylabel("Dissonance")
#         plt.title(f"Dissonance Curve for Emotion {emotion_code}")
#         plt.grid(True, ls='--', alpha=0.4)
#         plt.tight_layout()
#         plt.show()
#         return curve

# # ============= Example usage =============
# if __name__ == "__main__":
#     dataset_path = r"dataset"  # <-- update path

#     env = RavdessEnv(dataset_path=dataset_path, sr=16000)
#     agent = DissonanceAgent(env)

#     # Example: plot dissonance curve for Happy (03)
#     happy_curve = agent.plot_curve(emotion_code="03", n_segments=40, seg_duration=0.1, fft_size=8192)

#     # Example: plot dissonance curve for Sad (04)
#     sad_curve = agent.plot_curve(emotion_code="04", n_segments=40, seg_duration=0.1, fft_size=8192)


import numpy as np
import matplotlib.pyplot as plt

# Sethares dissonance model constants
b1, b2 = 3.5, 5.75
c1, c2 = 0.0207, 18.96
a1, a2 = -3.51, -5.75

def roughness_pair(f1, f2, a1_amp, a2_amp):
    """
    Roughness between two sinusoids with frequencies f1, f2 and amplitudes a1_amp, a2_amp
    """
    df = abs(f2 - f1)
    s = 0.24 / (0.021 * min(f1, f2) + 19)
    return (a1_amp * a2_amp) * (np.exp(-b1 * s * df) - np.exp(-b2 * s * df))

def dissonance_curve(base_freq=200, num_partials=6, ratio_range=(1, 2), steps=200):
    """
    Generate dissonance curve for a harmonic complex tone shifted by frequency ratio.
    """
    ratios = np.linspace(ratio_range[0], ratio_range[1], steps)
    dissonances = []

    # Partials of base tone
    freqs = np.array([base_freq * (i+1) for i in range(num_partials)])
    amps = 1.0 / (np.arange(1, num_partials+1))  # simple amplitude rolloff

    for r in ratios:
        shifted_freqs = freqs * r
        shifted_amps = amps.copy()
        d_total = 0.0

        # compute dissonance between all partial pairs (original vs shifted)
        for i in range(len(freqs)):
            for j in range(len(shifted_freqs)):
                d_total += roughness_pair(freqs[i], shifted_freqs[j], amps[i], shifted_amps[j])

        dissonances.append(d_total)

    return ratios, np.array(dissonances)


if __name__ == "__main__":
    ratios, dissonances = dissonance_curve(base_freq=200, num_partials=6, ratio_range=(1, 2), steps=400)

    plt.figure(figsize=(8, 5))
    plt.plot(ratios, dissonances, color="blue")
    plt.xlabel("Frequency Ratio (f2/f1)")
    plt.ylabel("Dissonance (roughness)")
    plt.title("Dissonance Curve (Plomp–Levelt / Sethares model)")
    plt.grid(True)
    plt.show()
