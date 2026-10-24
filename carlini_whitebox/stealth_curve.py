import glob, os, csv, subprocess
import numpy as np
import soundfile as sf
from pydub import AudioSegment

ORIGINAL = r"sound_prompt\prompt_bluecolor.wav"
HITS_DIR = r"tmp\hits"
OUT_DIR  = r"tmp\hits_robustness"
os.makedirs(OUT_DIR, exist_ok=True)

CHECKPOINT = r"deepspeech-0.9.3-checkpoint\best_dev-1466475"
SCORER     = r"deepspeech-0.9.3-models.scorer"
ALPHABET   = r"DeepSpeech\data\alphabet.txt"

def add_noise(path, out, rms_fraction=0.01, seed=0):
    y, sr = sf.read(path, dtype="float64")
    if y.ndim == 2: y = y[:, 1]
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(len(y))
    noise *= rms_fraction * np.sqrt(np.mean(y**2)) / np.sqrt(np.mean(noise**2))
    y16 = np.clip((y + noise) * 32767, -32768, 32767).astype(np.int16)
    sf.write(out, y16, sr, subtype="PCM_16")

def mp3_roundtrip(path, out, kbps="64k"):
    AudioSegment.from_wav(path).export("tmp_mp3_scratch.mp3", format="mp3", bitrate=kbps)
    AudioSegment.from_mp3("tmp_mp3_scratch.mp3").set_frame_rate(16000).set_channels(1).export(out, format="wav")

def perturbation_db(orig_path, hit_path):
    o, _ = sf.read(orig_path, dtype="float64"); a, _ = sf.read(hit_path, dtype="float64")
    if o.ndim == 2: o = o[:, 1]
    if a.ndim == 2: a = a[:, 1]
    n = min(len(o), len(a))
    d = (a[:n] - o[:n]) * 32768
    o16 = o[:n] * 32768
    return round(20*np.log10(np.max(np.abs(d))) - 20*np.log10(np.max(np.abs(o16))), 1)

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
    print("STDOUT:", out.stdout[-500:]); print("STDERR:", out.stderr[-500:])
    return ""

TARGET = "send one thousand dollars"

rows = []
hit_files = sorted(glob.glob(os.path.join(HITS_DIR, "hit_*.wav")))
print(f"Found {len(hit_files)} hit files")

for hf in hit_files:
    base = os.path.splitext(os.path.basename(hf))[0]
    noisy = os.path.join(OUT_DIR, f"{base}_noise.wav")
    mp3ed = os.path.join(OUT_DIR, f"{base}_mp3.wav")
    add_noise(hf, noisy)
    mp3_roundtrip(hf, mp3ed)

    db = perturbation_db(ORIGINAL, hf)
    clean_out = classify(hf)
    noise_out = classify(noisy)
    mp3_out   = classify(mp3ed)

    row = dict(file=base, peak_distortion_dB=db,
               clean_output=clean_out, noise_output=noise_out, mp3_output=mp3_out,
               clean_hit=clean_out.strip().lower() == TARGET,
               noise_hit=noise_out.strip().lower() == TARGET,
               mp3_hit=mp3_out.strip().lower() == TARGET)
    print(row)
    rows.append(row)

with open("stealth_curve_results.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print("Done — stealth_curve_results.csv written")