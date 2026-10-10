import subprocess, csv, os

CARLINI_DIR = os.path.join("..", "carlini_whitebox")
CHECKPOINT  = os.path.join(CARLINI_DIR, "deepspeech-0.9.3-checkpoint", "best_dev-1466475")
SCORER      = os.path.join(CARLINI_DIR, "deepspeech-0.9.3-models.scorer")
ALPHABET    = os.path.join(CARLINI_DIR, "DeepSpeech", "data", "alphabet.txt")
CLASSIFY_PY = os.path.join(CARLINI_DIR, "classify.py")
AES         = "AEs"

def classify(wav_path):
    cmd = ["python", CLASSIFY_PY,
           "--input", wav_path,
           "--restore_path", CHECKPOINT,
           "--scorer_path", SCORER,
           "--alphabet_config_path", ALPHABET]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    lines = out.stdout.splitlines()
    for i, line in enumerate(lines):
        if line.strip() == "Classification:":
            return lines[i + 1].strip()
    print(f"--- no Classification line for {wav_path} ---")
    print("STDOUT:", out.stdout[-500:])
    print("STDERR:", out.stderr[-500:])
    return ""

files = {}
for n in ["01","02","03","04","05","06","07","08"]:
    files[f"{n}_AE"]  = os.path.join(AES, "resample", f"{n}_16k.wav")   # matches your actual folder name
    files[f"{n}_TTS"] = os.path.join(AES, f"{n}_TTS_music.wav")

with open(os.path.join(AES, "deepspeech_results.csv"), "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["key", "deepspeech_output"])
    for key, path in files.items():
        text = classify(path)
        print(key, "->", text)
        writer.writerow([key, text])