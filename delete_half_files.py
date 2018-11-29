import os
import random
folder = "asl-alphabet/asl_alphabet_train/"

letters = os.listdir(folder)

for letter in letters:
    files = os.listdir(folder+letter)

    print("Original files: ", letter, len(files))
    for filename in files:
        if(random.random() > 0.5):
            os.remove(folder+letter+"/"+filename)
    print("Purged files: ", letter, len(os.listdir(folder+letter)))
