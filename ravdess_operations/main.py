import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.signal import welch

# ============= Environment =============
class RavdessEnv:
    def __init__(self, dataset_path, sr=16000, frame_ms=10, overlap=0.5):
        self.dataset_path = dataset_path
        self.sr = sr
        self.frame_len = int(frame_ms / 1000 * sr)          # samples per 10ms
        self.hop_len = int(self.frame_len * (1 - overlap)) # hop length for overlapping windows
        if self.hop_len < 1:
            self.hop_len = 1

    def _is_emotion_file(self, file_path, emotion_code):
        fname = os.path.basename(file_path)
        parent = os.path.basename(os.path.dirname(file_path))
        if f"-{emotion_code}-" in fname:
            return True
        if emotion_code in parent:
            return True
        return False

    def list_emotion_files(self, emotion_code):
        wavs = []
        for root, _, files in os.walk(self.dataset_path):
            for f in files:
                if not f.lower().endswith(".wav"):
                    continue
                full = os.path.join(root, f)
                if self._is_emotion_file(full, emotion_code):
                    wavs.append(full)
        return sorted(wavs)

    def load_audio(self, path):
        y, sr = librosa.load(path, sr=self.sr, mono=True)
        return y

    def frames_from_signal(self, signal):
        frames = []
        L = len(signal)
        if L < self.frame_len:
            return []
        for start in range(0, L - self.frame_len + 1, self.hop_len):
            frames.append(signal[start:start + self.frame_len])
        return frames

    def psds_from_file(self, path):
        signal = self.load_audio(path)
        frames = self.frames_from_signal(signal)
        psds = []
        freqs = None
        for frame in frames:
            f, Pxx = welch(frame, fs=self.sr, nperseg=self.frame_len)
            psds.append(Pxx)
            if freqs is None:
                freqs = f
        return freqs, psds

    def psds_for_emotion(self, emotion_code):
        files = self.list_emotion_files(emotion_code)
        all_psds = []
        freqs = None
        for p in files:
            f, psds = self.psds_from_file(p)
            if freqs is None:
                freqs = f
            if psds:
                all_psds.extend(psds)
        if len(all_psds) == 0:
            return freqs, np.empty((0, 0)), []  # no data
        all_psds_arr = np.vstack(all_psds)
        avg_psd = np.mean(all_psds_arr, axis=0)
        return freqs, avg_psd, all_psds_arr

# ============= Agent =============
class PSDAgent:
    def __init__(self, env: RavdessEnv):
        self.env = env

    def analyze_and_plot(self, happy_code="03", sad_code="04", max_frame_traces=500, save_path=None):
        f_h, avg_h, psds_h = self.env.psds_for_emotion(happy_code)
        f_s, avg_s, psds_s = self.env.psds_for_emotion(sad_code)

        if f_h is None or f_s is None:
            raise RuntimeError("No PSDs found for one of the emotions. Check dataset_path and file organization.")

        plt.figure(figsize=(12, 7))

        if psds_h.size:
            n_h = psds_h.shape[0]
            step_h = max(1, n_h // max_frame_traces)
            for idx in range(0, n_h, step_h):
                plt.semilogy(f_h, psds_h[idx, :], alpha=0.12, linewidth=0.6, label="_nolegend_")
            plt.semilogy(f_h, avg_h, color='orange', linewidth=2, label='Happy - Avg PSD')

        if psds_s.size:
            n_s = psds_s.shape[0]
            step_s = max(1, n_s // max_frame_traces)
            for idx in range(0, n_s, step_s):
                plt.semilogy(f_s, psds_s[idx, :], alpha=0.12, linewidth=0.6, label="_nolegend_")
            plt.semilogy(f_s, avg_s, color='blue', linewidth=2, label='Sad - Avg PSD')

        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Power Spectral Density (V^2/Hz)")
        plt.title("Overlayed Frame PSDs and Average PSDs: Happy (orange) vs Sad (blue)")
        plt.legend()
        plt.grid(True, which='both', ls='--', alpha=0.4)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
        plt.show()

        return {"freqs": f_h, "avg_psd_happy": avg_h, "avg_psd_sad": avg_s,
                "psds_happy": psds_h, "psds_sad": psds_s}

    def plot_happy_only(self, happy_code="03", max_frame_traces=500, save_path=None):
        f_h, avg_h, psds_h = self.env.psds_for_emotion(happy_code)

        if f_h is None:
            raise RuntimeError("No PSDs found for happy emotion. Check dataset_path and file organization.")

        plt.figure(figsize=(12, 7))

        if psds_h.size:
            n_h = psds_h.shape[0]
            step_h = max(1, n_h // max_frame_traces)
            for idx in range(0, n_h, step_h):
                plt.semilogy(f_h, psds_h[idx, :], alpha=0.12, linewidth=0.6, label="_nolegend_")
            plt.semilogy(f_h, avg_h, color='orange', linewidth=2, label='Happy - Avg PSD')

        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Power Spectral Density (V^2/Hz)")
        plt.title("Overlayed Frame PSDs and Average PSDs: Happy Only")
        plt.legend()
        plt.grid(True, which='both', ls='--', alpha=0.4)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
        plt.show()

        return {"freqs": f_h, "avg_psd_happy": avg_h, "psds_happy": psds_h}

    def plot_sad_only(self, sad_code="04", max_frame_traces=500, save_path=None):
        f_s, avg_s, psds_s = self.env.psds_for_emotion(sad_code)

        if f_s is None:
            raise RuntimeError("No PSDs found for sad emotion. Check dataset_path and file organization.")

        plt.figure(figsize=(12, 7))

        if psds_s.size:
            n_s = psds_s.shape[0]
            step_s = max(1, n_s // max_frame_traces)
            for idx in range(0, n_s, step_s):
                plt.semilogy(f_s, psds_s[idx, :], alpha=0.12, linewidth=0.6, label="_nolegend_")
            plt.semilogy(f_s, avg_s, color='blue', linewidth=2, label='Sad - Avg PSD')

        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Power Spectral Density (V^2/Hz)")
        plt.title("Overlayed Frame PSDs and Average PSDs: Sad Only")
        plt.legend()
        plt.grid(True, which='both', ls='--', alpha=0.4)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
        plt.show()

        return {"freqs": f_s, "avg_psd_sad": avg_s, "psds_sad": psds_s}

# ============= Example usage =============
if __name__ == "__main__":
    dataset_path = r"dataset"  # <-- update with your dataset path containing Actor_03 and Actor_04

    env = RavdessEnv(dataset_path=dataset_path, sr=16000, frame_ms=10, overlap=0.5)
    agent = PSDAgent(env)

    # Happy vs Sad comparison
    # results = agent.analyze_and_plot(
    #     happy_code="03",
    #     sad_code="04",
    #     max_frame_traces=600,
    #     save_path="ravdess_psd_overlay.png"
    # )

    # Happy only
    # happy_results = agent.plot_happy_only(
    #     happy_code="03",
    #     max_frame_traces=600,
    #     save_path="ravdess_psd_happy_only.png"
    # )

    # Sad only
    sad_results = agent.plot_sad_only(
        sad_code="04",
        max_frame_traces=600,
        save_path="ravdess_psd_sad_only.png"
    )
