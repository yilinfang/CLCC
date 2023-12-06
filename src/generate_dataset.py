from config import config

cfg = config()
random_seed = cfg.RANDOM_SEED
import random

random.seed(random_seed)

import pandas as pd
import sqlite3

con = sqlite3.connect(r"../dataset/java-python-clones.db")
df = pd.read_sql_query("SELECT * from submissions", con)

from tqdm import tqdm


def generate_dataset(id_pairs, is_clone, MAX_NUM=3000):
    id_pairs = random.sample(id_pairs, MAX_NUM)
    dataset = []
    for id_pair in tqdm(id_pairs):
        id1 = id_pair[0]
        id2 = id_pair[1]
        row1 = df[df["id"] == id1].iloc[0]
        row2 = df[df["id"] == id2].iloc[0]
        d1 = {}
        d2 = {}
        d1["id"] = int(id1)
        d1["source"] = row1["source"]
        d1["language_code"] = row1["language_code"]
        d1["ast"] = row1["ast"]
        d1["contest_id"] = int(row1["contest_id"])
        d1["problem_id"] = int(row1["problem_id"])
        d2["id"] = int(id2)
        d2["source"] = row2["source"]
        d2["language_code"] = row2["language_code"]
        d2["ast"] = row2["ast"]
        d2["contest_id"] = int(row2["contest_id"])
        d2["problem_id"] = int(row2["problem_id"])
        d = {}
        d["sample_0"] = d1
        d["sample_1"] = d2
        d["is_clone"] = is_clone
        dataset.append(d)
    return dataset


import pickle

with open(r"../dataset/clone_id_pairs_py_train.pkl", "rb") as f:
    clone_id_pairs_py_train = pickle.load(f)
    dataset_py_train = generate_dataset(
        clone_id_pairs_py_train, 1, cfg.NO_CLONE_PAIRS_TRAIN_NUM
    )

with open(r"../dataset/clone_id_pairs_py_test.pkl", "rb") as f:
    clone_id_pairs_py_test = pickle.load(f)
    dataset_py_test = generate_dataset(
        clone_id_pairs_py_test, 1, cfg.NO_CLONE_PAIRS_TEST_NUM
    )

with open(r"../dataset/no_clone_id_pairs_py_train.pkl", "rb") as f:
    no_clone_id_pairs_py_train = pickle.load(f)
    dataset_py_train.extend(
        generate_dataset(no_clone_id_pairs_py_train, 0, cfg.NO_CLONE_PAIRS_TRAIN_NUM)
    )

with open(r"../dataset/no_clone_id_pairs_py_test.pkl", "rb") as f:
    no_clone_id_pairs_py_test = pickle.load(f)
    dataset_py_test.extend(
        generate_dataset(no_clone_id_pairs_py_test, 0, cfg.NO_CLONE_PAIRS_TEST_NUM)
    )


with open(r"../dataset/clone_id_pairs_java_train.pkl", "rb") as f:
    clone_id_pairs_java_train = pickle.load(f)
    dataset_java_train = generate_dataset(
        clone_id_pairs_java_train, 1, cfg.NO_CLONE_PAIRS_TRAIN_NUM
    )

with open(r"../dataset/clone_id_pairs_java_test.pkl", "rb") as f:
    clone_id_pairs_java_test = pickle.load(f)
    dataset_java_test = generate_dataset(
        clone_id_pairs_java_test, 1, cfg.NO_CLONE_PAIRS_TEST_NUM
    )

with open(r"../dataset/no_clone_id_pairs_java_train.pkl", "rb") as f:
    no_clone_id_pairs_java_train = pickle.load(f)
    dataset_java_train.extend(
        generate_dataset(no_clone_id_pairs_java_train, 0, cfg.NO_CLONE_PAIRS_TRAIN_NUM)
    )

with open(r"../dataset/no_clone_id_pairs_java_test.pkl", "rb") as f:
    no_clone_id_pairs_java_test = pickle.load(f)
    dataset_java_test.extend(
        generate_dataset(no_clone_id_pairs_java_test, 0, cfg.NO_CLONE_PAIRS_TEST_NUM)
    )


with open(r"../dataset/clone_id_pairs_cross_train.pkl", "rb") as f:
    clone_id_pairs_cross_train = pickle.load(f)
    dataset_cross_train = generate_dataset(
        clone_id_pairs_cross_train, 1, cfg.NO_CLONE_PAIRS_TRAIN_NUM
    )

with open(r"../dataset/clone_id_pairs_cross_test.pkl", "rb") as f:
    clone_id_pairs_cross_test = pickle.load(f)
    dataset_cross_test = generate_dataset(
        clone_id_pairs_cross_test, 1, cfg.NO_CLONE_PAIRS_TEST_NUM
    )

with open(r"../dataset/no_clone_id_pairs_cross_train.pkl", "rb") as f:
    no_clone_id_pairs_cross_train = pickle.load(f)
    dataset_cross_train.extend(
        generate_dataset(no_clone_id_pairs_cross_train, 0, cfg.NO_CLONE_PAIRS_TRAIN_NUM)
    )

with open(r"../dataset/no_clone_id_pairs_cross_test.pkl", "rb") as f:
    no_clone_id_pairs_cross_test = pickle.load(f)
    dataset_cross_test.extend(
        generate_dataset(no_clone_id_pairs_cross_test, 0, cfg.NO_CLONE_PAIRS_TEST_NUM)
    )


# Save the datasets
import json

with open(r"../dataset/dataset_py_train.json", "w") as f:
    json.dump(dataset_py_train, f)

with open(r"../dataset/dataset_py_test.json", "w") as f:
    json.dump(dataset_py_test, f)

with open(r"../dataset/dataset_java_train.json", "w") as f:
    json.dump(dataset_java_train, f)

with open(r"../dataset/dataset_java_test.json", "w") as f:
    json.dump(dataset_java_test, f)

with open(r"../dataset/dataset_cross_train.json", "w") as f:
    json.dump(dataset_cross_train, f)

with open(r"../dataset/dataset_cross_test.json", "w") as f:
    json.dump(dataset_cross_test, f)

print("Done")
