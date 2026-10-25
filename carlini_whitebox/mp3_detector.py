import os, csv, subprocess
from pydub import AudioSegment

CHECKPOINT = r"deepspeech-0.9.3-checkpoint\best_dev-1466475"
SCORER     = r"deepspeech-0.9.3-models.scorer"
ALPHABET   = r"DeepSpeech\data\alphabet.txt"

# local_path -> the EXACT Colab/Drive path string used as the dict key there
FILES = {
    r"results\adv_bluecolor_sendmoney.wav":     "/content/drive/MyDrive/Colab Notebooks/3183/Assignment/results/adv_bluecolor_sendmoney.wav",
    r"results\adv_weather_disablealarm.wav":    "/content/drive/MyDrive/Colab Notebooks/3183/Assignment/results/adv_weather_disablealarm.wav",
    r"results\adv_orderpizza_unlockdoor.wav":   "/content/drive/MyDrive/Colab Notebooks/3183/Assignment/results/adv_orderpizza_unlockdoor.wav",
    r"results\adv_lockdoor_deletefiles.wav":    "/content/drive/MyDrive/Colab Notebooks/3183/Assignment/results/adv_lockdoor_deletefiles.wav",
    r"results\adv_recordingtest_emergency.wav": "/content/drive/MyDrive/Colab Notebooks/3183/Assignment/results/adv_recordingtest_emergency.wav",
    r"sound_prompt\prompt_bluecolor.wav":       "/content/drive/MyDrive/Colab Notebooks/3183/Assignment/sound_prompt/prompt_bluecolor.wav",
    r"sound_prompt\prompt_weather.wav":         "/content/drive/MyDrive/Colab Notebooks/3183/Assignment/sound_prompt/prompt_weather.wav",
    r"sound_prompt\prompt_orderpizza.wav":      "/content/drive/MyDrive/Colab Notebooks/3183/Assignment/sound_prompt/prompt_orderpizza.wav",
    r"sound_prompt\prompt_lockdoor.wav":        "/content/drive/MyDrive/Colab Notebooks/3183/Assignment/sound_prompt/prompt_lockdoor.wav",
    r"sound_prompt\prompt_recordingtest.wav":   "/content/drive/MyDrive/Colab Notebooks/3183/Assignment/sound_prompt/prompt_recordingtest.wav",
    r"..\devilswhisper_blackbox\AEs\resample\01_16k.wav": "/content/drive/MyDrive/Colab Notebooks/3183/Assignment/AEs/resampled/01_16k.wav",
    r"..\devilswhisper_blackbox\AEs\resample\02_16k.wav": "/content/drive/MyDrive/Colab Notebooks/3183/Assignment/AEs/resampled/02_16k.wav",
    r"..\devilswhisper_blackbox\AEs\resample\03_16k.wav": "/content/drive/MyDrive/Colab Notebooks/3183/Assignment/AEs/resampled/03_16k.wav",
    r"..\devilswhisper_blackbox\AEs\resample\04_16k.wav": "/content/drive/MyDrive/Colab Notebooks/3183/Assignment/AEs/resampled/04_16k.wav",
    r"..\devilswhisper_blackbox\AEs\resample\05_16k.wav": "/content/drive/MyDrive/Colab Notebooks/3183/Assignment/AEs/resampled/05_16k.wav",
}

def mp3_roundtrip(path, out, kbps="64k"):
    AudioSegment.from_wav(path).export("tmp_det.mp3", format="mp3", bitrate=kbps)
    AudioSegment.from_mp3("tmp_det.mp3").set_frame_rate(16000).set_channels(1).export(out, format="wav")

def classify(wav_path):
    cmd = ["python", "classify.py", "--input", wav_path,
           "--restore_path", CHECKPOINT, "--scorer_path", SCORER,
           "--alphabet_config_path", ALPHABET]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    lines = out.stdout.splitlines()
    for i, line in enumerate(lines):
        if line.strip() == "Classification:":
            return lines[i + 1].strip()
    print(f"--- no Classification line for {wav_path} ---")
    print(out.stdout[-500:]); print(out.stderr[-500:])
    return ""

os.makedirs("tmp/detector", exist_ok=True)
rows = []
for local_path, drive_key in FILES.items():
    mp3_local = os.path.join("tmp/detector", os.path.basename(local_path).replace(".wav", "_det_mp3.wav"))
    mp3_roundtrip(local_path, mp3_local)
    mp3_key = drive_key.replace(".wav", "_det_mp3.wav")   # must match path.replace(...) in Cell 53

    rows.append((drive_key, classify(local_path)))
    rows.append((mp3_key,   classify(mp3_local)))

with open("detector_results.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["key", "deepspeech_output"])
    w.writerows(rows)

print("Done — detector_results.csv written,", len(rows), "rows")