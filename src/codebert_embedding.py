import torch
from transformers import AutoTokenizer, AutoModel


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
model = AutoModel.from_pretrained("microsoft/codebert-base")
model.to(device)

code_snippet = "def add(a, b):\n    return a + b"
code_tokens = tokenizer.encode_plus(
    code_snippet, return_tensors="pt", padding=True, truncation=True, max_length=128
)

with torch.no_grad():
    outputs = model(**code_tokens.to(device))
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze()


def get_code_embeddings(code_snippets):
    code_snippets = code_snippets.lstrip()
    code_tokens = tokenizer.encode_plus(
        code_snippets,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256,
    )
    with torch.no_grad():
        outputs = model(**code_tokens.to(device))
        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze()
    return embeddings


import json

with open(r"../dataset/dataset_py_train.json", "r") as f:
    dataset_py_train = json.load(f)

with open(r"../dataset/dataset_py_test.json", "r") as f:
    dataset_py_test = json.load(f)

with open(r"../dataset/dataset_java_train.json", "r") as f:
    dataset_java_train = json.load(f)

with open(r"../dataset/dataset_java_test.json", "r") as f:
    dataset_java_test = json.load(f)

with open(r"../dataset/dataset_cross_train.json") as f:
    dataset_cross_train = json.load(f)

with open(r"../dataset/dataset_cross_test.json") as f:
    dataset_cross_test = json.load(f)

from tqdm import tqdm


def embed_dataset(dataset):
    output = []
    for item in tqdm(dataset):
        sample_0 = item["sample_0"]
        code_snippet_0 = sample_0["source"]
        code_embedding_0 = get_code_embeddings(code_snippet_0)
        sample_0["code_embedding"] = code_embedding_0
        sample_1 = item["sample_1"]
        code_snippet_1 = sample_1["source"]
        code_embedding_1 = get_code_embeddings(code_snippet_1)
        sample_1["code_embedding"] = code_embedding_1
        output.append(item)
    return output


embed_dataset(dataset_py_train)
embed_dataset(dataset_py_test)
embed_dataset(dataset_java_train)
embed_dataset(dataset_java_test)
embed_dataset(dataset_cross_train)
embed_dataset(dataset_cross_test)

# save the embedded dataset
import pickle

with open(r"../dataset/dataset_py_embedded_train.pkl", "wb") as f:
    pickle.dump(dataset_py_train, f)

with open(r"../dataset/dataset_py_embedded_test.pkl", "wb") as f:
    pickle.dump(dataset_py_test, f)

with open(r"../dataset/dataset_java_embedded_train.pkl", "wb") as f:
    pickle.dump(dataset_java_train, f)

with open(r"../dataset/dataset_java_embedded_test.pkl", "wb") as f:
    pickle.dump(dataset_java_test, f)

with open(r"../dataset/dataset_cross_embedded_train.pkl", "wb") as f:
    pickle.dump(dataset_cross_train, f)

with open(r"../dataset/dataset_cross_embedded_test.pkl", "wb") as f:
    pickle.dump(dataset_cross_test, f)

print("Done")
