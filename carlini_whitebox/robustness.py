import os, csv, subprocess

CHECKPOINT = r"deepspeech-0.9.3-checkpoint\best_dev-1466475"
SCORER     = r"deepspeech-0.9.3-models.scorer"
ALPHABET   = r"DeepSpeech\data\alphabet.txt"

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

CW_NAMES = {  # notebook "name" -> local clean file
    "Blue color":      r"results\adv_bluecolor_sendmoney.wav",
    "Weather":         r"results\adv_weather_disablealarm.wav",
    "Pizza":           r"results\adv_orderpizza_unlockdoor.wav",
    "Lock door":       r"results\adv_lockdoor_deletefiles.wav",
    "Recording test":  r"results\adv_recordingtest_emergency.wav",
}
DW_PAIRS = [f"{i:02d}" for i in range(1, 9)]

rows = []

for name, clean_path in CW_NAMES.items():
    under = name.replace(" ", "_")
    rows.append((f"cw_{name}_clean",       classify(clean_path)))
    rows.append((f"cw_{name}_clean_noise", classify(fr"robustness\{under}_noise.wav")))
    rows.append((f"cw_{name}_clean_mp3",   classify(fr"robustness\{under}_mp3.wav")))

for n in DW_PAIRS:
    rows.append((f"{n}_AE",       classify(fr"..\devilswhisper_blackbox\AEs\resample\{n}_16k.wav")))
    rows.append((f"{n}_AE_noise", classify(fr"robustness\dw_{n}_noise.wav")))
    rows.append((f"{n}_AE_mp3",   classify(fr"robustness\dw_{n}_mp3.wav")))

with open("robustness_results.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["key", "deepspeech_output"])
    w.writerows(rows)

print("Done — robustness_results.csv written,", len(rows), "rows")