import os
import zipfile

os.chdir(".\\data")

for file in os.listdir("."):
    if not file.lower().endswith(".zip"):
        continue
    with zipfile.ZipFile(file, "r") as zip:
        zip.extractall(file.split(".")[0])
    os.remove(file)
    