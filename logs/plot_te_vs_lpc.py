import csv
import matplotlib.pyplot as plt
from collections import defaultdict

CSV_PATH = "logs/calibration_te_lpc.csv"


def safe_float(x):
    try:
        return float(x)
    except ValueError:
        return None


def load_data(path):
    data = defaultdict(lambda: {
        "lpc_f1": [], "lpc_f2": [],
        "te_f1": [],  "te_f2": []
    })

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 8:
                continue

            vowel = row[0] or "?"

            lpc_f1 = safe_float(row[1])
            lpc_f2 = safe_float(row[2])
            te_f1 = safe_float(row[5])
            te_f2 = safe_float(row[6])

            if lpc_f1 and lpc_f2:
                data[vowel]["lpc_f1"].append(lpc_f1)
                data[vowel]["lpc_f2"].append(lpc_f2)

            if te_f1 and te_f2:
                data[vowel]["te_f1"].append(te_f1)
                data[vowel]["te_f2"].append(te_f2)

    return data


def plot_vowel(vowel, d):
    plt.figure(figsize=(8, 6))
    plt.title(f"LPC vs TE for /{vowel}/")

    # LPC points
    plt.scatter(
        d["lpc_f1"], d["lpc_f2"],
        color="red", alpha=0.6, label="LPC"
    )

    # TE points
    plt.scatter(
        d["te_f1"], d["te_f2"],
        color="blue", alpha=0.6, label="TE"
    )

    plt.xlabel("F1 (Hz)")
    plt.ylabel("F2 (Hz)")
    plt.grid(True)
    plt.legend()
    plt.show()


def main():
    data = load_data(CSV_PATH)

    for vowel, d in data.items():
        if len(d["lpc_f1"]) == 0 and len(d["te_f1"]) == 0:
            continue
        plot_vowel(vowel, d)


if __name__ == "__main__":
    main()
