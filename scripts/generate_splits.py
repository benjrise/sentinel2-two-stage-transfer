import argparse

import os
import sys
import random

SPLITS = [50, 25, 10, 5]


def load_data(file_path):
    class_files = {}
    with open(file_path, "r") as file:
        for line in file:
            class_name, file_name = line.strip().split("_")
            class_files.setdefault(class_name, []).append(line.strip())
    return class_files


def split_data(class_files, split_percentage):
    split = {}
    print("number of class files", len(class_files))
    for class_name, files in class_files.items():
        split_size = int(len(files) * split_percentage / 100)
        split[class_name] = random.sample(files, split_size)
    return split


def save_split(split, folder, split_percentage):
    file_name = os.path.join(folder, f"eurosat-train{split_percentage}.txt")
    with open(file_name, "w") as file:
        for files in split.values():
            for file_name in files:
                file.write(file_name + "\n")


def main(folder):
    eurosat_file = os.path.join(folder, "eurosat-train.txt")
    if not os.path.exists(eurosat_file):
        print("eurosat-train.txt not found in the given folder.")
        return

    class_files = load_data(eurosat_file)

    for split_percentage in SPLITS:
        split = split_data(class_files, split_percentage)
        save_split(split, folder, split_percentage)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <folder>")
    else:
        main(sys.argv[1])
