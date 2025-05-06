# Read a Brat ANN file
with open("read_ann/MACCROBAT2020/15939911.ann", "r", encoding="utf-8") as file:
    for line in file:
        print(line.strip())  # Print each annotation line