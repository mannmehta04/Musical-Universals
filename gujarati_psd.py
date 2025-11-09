import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.signal import welch

# ============= Environment =============
class RasyaEnv:
    def __init__(self, dataset_path, sr=16000, frame_ms=10, overlap=0.5):
        self.dataset_path = dataset_path
        self.sr = sr
        self.frame_len = int(frame_ms / 1000 * sr)
        self.hop_len = int(self.frame_len * (1 - overlap))
        if self.hop_len < 1:
            self.hop_len = 1

    def list_emotion_files(self, emotion_code):
        wavs = []
        for root, _, files in os.walk(self.dataset_path):
            for f in files:
                if not f.lower().endswith(".wav"):
                    continue
                parts = f.split("-")
                if len(parts) >= 4 and parts[0] == emotion_code:
                    wavs.append(os.path.join(root, f))
        return sorted(wavs)

    def load_audio(self, path):
        y, _ = librosa.load(path, sr=self.sr, mono=True)
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

    def psds_all_combined(self):
        """Combine all 5 rasya PSDs into one averaged PSD"""
        all_psds = []
        freqs = None
        for i in range(1, 6):
            code = f"{i:02d}"
            f, avg_psd, _ = self.psds_for_emotion(code)
            if f is not None and avg_psd.size > 0:
                if freqs is None:
                    freqs = f
                all_psds.append(avg_psd)
        if len(all_psds) == 0:
            return freqs, np.empty((0,)), []
        combined_avg = np.mean(np.vstack(all_psds), axis=0)
        return freqs, combined_avg, np.vstack(all_psds)

# ============= Agent =============
class RasyaPSDAgent:
    def __init__(self, env: RasyaEnv):
        self.env = env
        self.rasya_map = {
            "01": "Shant Ras",
            "02": "Hasya Ras",
            "03": "Bhayanak Ras",
            "04": "Karuna Ras",
            "05": "Rudra Ras",
            "06": "All Rasyas"
        }

    def plot_single_rasya(self, rasya_code, max_frame_traces=500, save_path=None):
        if rasya_code == "06":
            f, avg_psd, psds = self.env.psds_all_combined()
        else:
            f, avg_psd, psds = self.env.psds_for_emotion(rasya_code)

        if f is None or avg_psd.size == 0:
            raise RuntimeError(f"No PSDs found for {self.rasya_map[rasya_code]} emotion.")

        plt.figure(figsize=(12, 7))
        if psds.size:
            n = psds.shape[0]
            step = max(1, n // max_frame_traces)
            for idx in range(0, n, step):
                plt.semilogy(f, psds[idx, :], alpha=0.1, linewidth=0.6, label="_nolegend_")
        plt.semilogy(f, avg_psd, color='black', linewidth=2, label=f"{self.rasya_map[rasya_code]} - Avg PSD")

        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Power Spectral Density (V^2/Hz)")
        plt.title(f"Overlayed Frame PSDs and Average PSDs: {self.rasya_map[rasya_code]}")
        plt.legend()
        plt.grid(True, which='both', ls='--', alpha=0.4)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
        plt.show()

        return {"freqs": f, "avg_psd": avg_psd, "psds": psds}

    def plot_all_rasyas(self, max_frame_traces=500, save_path=None):
        plt.figure(figsize=(12, 7))
        colors = ["green", "orange", "red", "blue", "purple"]
        results = {}

        for i, code in enumerate(list(self.rasya_map.keys())[:-1]):
            f, avg_psd, psds = self.env.psds_for_emotion(code)
            if f is None or avg_psd.size == 0:
                continue
            plt.semilogy(f, avg_psd, color=colors[i], linewidth=2, label=self.rasya_map[code])
            results[code] = {"freqs": f, "avg_psd": avg_psd, "psds": psds}

        # Add combined all-rasya PSD
        f_all, avg_all, _ = self.env.psds_all_combined()
        plt.semilogy(f_all, avg_all, color='black', linewidth=2, linestyle='--', label=self.rasya_map["06"])
        results["06"] = {"freqs": f_all, "avg_psd": avg_all}

        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Power Spectral Density (V^2/Hz)")
        plt.title("Average PSDs for All Rasyas (Including Combined)")
        plt.legend()
        plt.grid(True, which='both', ls='--', alpha=0.4)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
        plt.show()

        return results

# ============= Example usage =============
if __name__ == "__main__":
    dataset_path = r"personal_dataset"  # Folder containing all 5 rasya .wav files

    env = RasyaEnv(dataset_path=dataset_path, sr=16000, frame_ms=10, overlap=0.5)
    agent = RasyaPSDAgent(env)

    # Plot each rasya (01–06 including all combined)
    # for code in ["01", "02", "03", "04", "05", "06"]:
    #     agent.plot_single_rasya(
    #         rasya_code=code,
    #         max_frame_traces=600,
    #         save_path=f"rasya_{code}_psd.png"
    #     )

    # Overlay of all rasyas and combined
    all_results = agent.plot_all_rasyas(
        max_frame_traces=600,
        save_path="gujarati_rasya_all_overlay.png"
    )
