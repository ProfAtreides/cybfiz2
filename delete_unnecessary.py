import os

useful_codes = [
    "B00300S",
    "B00305A",
    "B00202A",
    "B00702A",
    "B00604S",
    "B00802A"
]

os.chdir(".\\data")

for entry in os.listdir("."):
    if not (os.path.isdir(entry) and "Meteo" in entry):
        continue
    files = os.listdir(entry)
    for file in files:
        code = file.split("_")[0]
        if code not in useful_codes:
            filepath = os.path.join(entry, file)
            os.remove(filepath)
    