import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button


class ECGBrowser:
    def __init__(self, folder, load_big=True, limit_ecgs=50):
        self.folder = folder
        self.load_big = load_big

        # Load files based on load_big parameter
        self._load_files()
        if not self.files:
            raise FileNotFoundError("No files matched in folder")

        self.idx = 0

    def _load_files(self):
        print(f"Started Loading Files")
        if self.load_big:
            rhythm_files = []

            patient_dirs = [d for d in os.listdir(self.folder)]

            for patient_id in patient_dirs:
                patient_path = os.path.join(self.folder, patient_id)

                ecg_dirs = [d for d in os.listdir(patient_path)
                            if d.startswith('ecg_') and os.path.isdir(os.path.join(patient_path, d))]

                for ecg_dir in sorted(ecg_dirs):
                    ecg_path = os.path.join(patient_path, ecg_dir)

                    # Find all files containing "rhythm" (case-insensitive)
                    try:
                        files_in_dir = os.listdir(ecg_path)
                        rhythm_files_in_dir = [f for f in files_in_dir
                                               if 'rhythm' in f.lower() and f.endswith('.npy')]

                        for rhythm_file in rhythm_files_in_dir:
                            rhythm_files.append(
                                os.path.join(ecg_path, rhythm_file))
                    except (OSError, PermissionError) as e:
                        print(f"Warning: Could not access {ecg_path}: {e}")
                        continue
                    if len(rhythm_files) > 50:
                        break

                if len(rhythm_files) > 50:
                    break

            self.files = sorted(rhythm_files)
            print(
                f"Found {len(self.files)} rhythm files across {len(patient_dirs)} patients")
        else:
            # Original behavior for load_big=False
            self.files = sorted(glob.glob(os.path.join(self.folder, "*.npy")))

    def _load_ecg(self, path):
        """Load ECG data from numpy file"""
        return np.load(path)

    def _update_plot(self, i):
        self.fig, self.ax = plt.subplots(8, figsize=(2*15, 2*8*2.5))
        self.idx = i % len(self.files)
        data = self._load_ecg(self.files[self.idx])
        self.fig.suptitle(f'ECG Num {self.idx}', fontsize=50, y=0.92)

        for lead in range(8):
            y = list(data[lead, :])
            self.ax[lead].plot(list(range(data.shape[-1])),
                               y, linewidth=2, color='red')

            self.ax[lead].set_xlabel(f'Lead {lead}', fontsize=30)
            self.ax[lead].xaxis.label.set_visible(True)

            self.ax[lead].tick_params(axis='x', labelsize=30)
            self.ax[lead].tick_params(axis='y', labelsize=30)

        plt.subplots_adjust(hspace=0.4, wspace=0.2)

    def _on_next(self, _=None):
        self._update_plot(self.idx + 1)

    def _on_prev(self, _=None):
        self._update_plot(self.idx - 1)

    def _on_quit(self, _=None):
        plt.close(self.fig)

    def show(self):

        # self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        print("Controls: n = next, p = previous, q = quit")
        self._update_plot(self.idx)
        while True:
            print(f"Now showing plot {self.idx}")
            plt.savefig('test.png')
            cmd = input("Enter Command: ").strip().lower()
            if cmd == "n":
                self._on_next()
            elif cmd == "p":
                self._on_prev()
            elif cmd == "q":
                self._on_quit()
                break
            else:
                print("Unknown command. Use n, p, or q")


if __name__ == "__main__":
    file_path = "/uu/sci.utah.edu/projects/ClinicalECGs/AllClinicalECGs/pythonData"
    browser = ECGBrowser(file_path, load_big=True)
    browser.show()
