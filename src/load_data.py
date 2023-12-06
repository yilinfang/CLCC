import pandas as pd
import sqlite3
from tqdm import tqdm
from config import config

cfg = config()
random_seed = cfg.RANDOM_SEED

print("Random Seed:", random_seed)

print("Connecting to SQLite database...")
con = sqlite3.connect(r"../dataset/java-python-clones.db")
df = pd.read_sql_query("SELECT * from submissions", con)
print("Data loaded from SQLite.")

print("Filtering data for contest type 'b'...")
df = df[df["contest_type"] == "b"]

# random split train and test
print("Splitting data into train and test sets...")
df_train = df.sample(frac=0.8, random_state=random_seed)
df_test = df.drop(df_train.index)
print("Data split into train and test sets.")

print("Saving train and test data as files...")
df_train.to_pickle("../dataset/train.pkl")
df_train.to_csv("../dataset/train.csv")
df_test.to_pickle("../dataset/test.pkl")
df_test.to_csv("../dataset/test.csv")
print("Train and test data saved.")

# Generate clone pairs with same contest id
print("Generating clone pairs for training...")
df_py_train = df_train[df_train["language_code"] == "python"]
df_java_train = df_train[df_train["language_code"] == "java"]

clone_pairs_py_train = []
print("Generating clone pairs for Python")
for _, group in tqdm(df_py_train.groupby(["contest_id", "problem_id"])):
    if len(group) > 1:
        clone_pairs_py_train.extend(
            [
                (row1["id"], row2["id"])
                for index1, row1 in group.iterrows()
                for index2, row2 in group.iterrows()
                if index1 < index2
            ]
        )

print("Generating clone pairs for Java")
clone_pairs_java_train = []
for _, group in tqdm(df_java_train.groupby(["contest_id", "problem_id"])):
    if len(group) > 1:
        clone_pairs_java_train.extend(
            [
                (row1["id"], row2["id"])
                for index1, row1 in group.iterrows()
                for index2, row2 in group.iterrows()
                if index1 < index2
            ]
        )

print("Generating clone pairs for cross language")
clone_pairs_cross_train = []
for _, group in tqdm(df_train.groupby(["contest_id", "problem_id"])):
    if len(group["language"].unique()) > 1:
        clone_pairs_cross_train.extend(
            [
                (row1["id"], row2["id"])
                for index1, row1 in group.iterrows()
                for index2, row2 in group.iterrows()
                if index1 < index2
            ]
        )

print("Generating clone pairs for testing...")
df_py_test = df_test[df_test["language_code"] == "python"]
df_java_test = df_test[df_test["language_code"] == "java"]

print("Generating clone pairs for Python")
clone_pairs_py_test = []
for _, group in tqdm(df_py_test.groupby(["contest_id", "problem_id"])):
    if len(group) > 1:
        clone_pairs_py_test.extend(
            [
                (row1["id"], row2["id"])
                for index1, row1 in group.iterrows()
                for index2, row2 in group.iterrows()
                if index1 < index2
            ]
        )

print("Generating clone pairs for Java")
clone_pairs_java_test = []
for _, group in tqdm(df_java_test.groupby(["contest_id", "problem_id"])):
    if len(group) > 1:
        clone_pairs_java_test.extend(
            [
                (row1["id"], row2["id"])
                for index1, row1 in group.iterrows()
                for index2, row2 in group.iterrows()
                if index1 < index2
            ]
        )

print("Generating clone pairs for cross language")
clone_pairs_cross_test = []
for _, group in tqdm(df_test.groupby(["contest_id", "problem_id"])):
    if len(group["language"].unique()) > 1:
        clone_pairs_cross_test.extend(
            [
                (row1["id"], row2["id"])
                for index1, row1 in group.iterrows()
                for index2, row2 in group.iterrows()
                if index1 < index2
            ]
        )


import pickle

print("Saving generated clone pairs as files...")
with open("../dataset/clone_id_pairs_py_train.pkl", "wb") as f:
    pickle.dump(clone_pairs_py_train, f)

with open("../dataset/clone_id_pairs_java_train.pkl", "wb") as f:
    pickle.dump(clone_pairs_java_train, f)

with open("../dataset/clone_id_pairs_cross_train.pkl", "wb") as f:
    pickle.dump(clone_pairs_cross_train, f)

with open("../dataset/clone_id_pairs_py_test.pkl", "wb") as f:
    pickle.dump(clone_pairs_py_test, f)

with open("../dataset/clone_id_pairs_java_test.pkl", "wb") as f:
    pickle.dump(clone_pairs_java_test, f)

with open("../dataset/clone_id_pairs_cross_test.pkl", "wb") as f:
    pickle.dump(clone_pairs_cross_test, f)


import random

random.seed(random_seed)


def get_noclone_pairs(df, count):
    ret = set()
    ids = df["id"].unique().tolist()
    while len(ret) < count:
        id1 = random.choice(ids)
        id2 = random.choice(ids)
        if id1 == id2:
            continue
        # make id1 < id2
        id1, id2 = min(id1, id2), max(id1, id2)

        # check id1 and id2's contest_id and problem_id are not same
        row1 = df[df["id"] == id1].iloc[0]
        row2 = df[df["id"] == id2].iloc[0]
        if (
            row1["contest_id"] == row2["contest_id"]
            and row1["problem_id"] == row2["problem_id"]
        ):
            continue
        if id1 != id2:
            ret.add((id1, id2))
    return list(ret)


def get_cross_noclone_pairs(df_py, df_java, count):
    ret = set()
    ids_py = df_py["id"].unique().tolist()
    ids_java = df_java["id"].unique().tolist()
    while len(ret) < count:
        id1 = random.choice(ids_py)
        id2 = random.choice(ids_java)
        # check id1 and id2's contest_id and problem_id are not same
        row1 = df_py[df_py["id"] == id1].iloc[0]
        row2 = df_java[df_java["id"] == id2].iloc[0]
        if (
            row1["contest_id"] == row2["contest_id"]
            and row1["problem_id"] == row2["problem_id"]
        ):
            continue
        if id1 != id2:
            ret.add((id1, id2))
    return list(ret)


NO_CLONE_PAIRS_TRAIN_NUM = cfg.NO_CLONE_PAIRS_TRAIN_NUM

df_py = df_train[df_train["language_code"] == "python"]
df_java = df_train[df_train["language_code"] == "java"]

print("Generating non-clone pairs for training and testing data...")
no_clone_pairs_py = get_noclone_pairs(df_py, NO_CLONE_PAIRS_TRAIN_NUM)
no_clone_pairs_java = get_noclone_pairs(df_java, NO_CLONE_PAIRS_TRAIN_NUM)
no_clone_pairs_cross = get_cross_noclone_pairs(df_py, df_java, NO_CLONE_PAIRS_TRAIN_NUM)

NO_CLONE_PAIRS_TEST_NUM = cfg.NO_CLONE_PAIRS_TEST_NUM

df_py_test = df_test[df_test["language_code"] == "python"]
df_java_test = df_test[df_test["language_code"] == "java"]

no_clone_pairs_py_test = get_noclone_pairs(df_py_test, NO_CLONE_PAIRS_TEST_NUM)
no_clone_pairs_java_test = get_noclone_pairs(df_java_test, NO_CLONE_PAIRS_TEST_NUM)
no_clone_pairs_cross_test = get_cross_noclone_pairs(
    df_py_test, df_java_test, NO_CLONE_PAIRS_TEST_NUM
)

import pickle

print("Saving generated non-clone pairs as files...")
with open("../dataset/no_clone_id_pairs_py_train.pkl", "wb") as f:
    pickle.dump(no_clone_pairs_py, f)

with open("../dataset/no_clone_id_pairs_java_train.pkl", "wb") as f:
    pickle.dump(no_clone_pairs_java, f)

with open("../dataset/no_clone_id_pairs_cross_train.pkl", "wb") as f:
    pickle.dump(no_clone_pairs_cross, f)


with open("../dataset/no_clone_id_pairs_py_test.pkl", "wb") as f:
    pickle.dump(no_clone_pairs_py_test, f)

with open("../dataset/no_clone_id_pairs_java_test.pkl", "wb") as f:
    pickle.dump(no_clone_pairs_java_test, f)

with open("../dataset/no_clone_id_pairs_cross_test.pkl", "wb") as f:
    pickle.dump(no_clone_pairs_cross_test, f)
