import torch
from transformers import AutoTokenizer, AutoModel


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
cb = AutoModel.from_pretrained("microsoft/codebert-base")
cb.to(device)

code_snippet_1 = """
/*package whatever //do not write package name here */

import java.io.*;

class GFG {
	public static void main (String[] args) {
	System.out.println("Hello World");
	}
}
"""

code_snippet_2 = """
print("Hello World")

"""


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
        outputs = cb(**code_tokens.to(device))
        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze()
    return embeddings


code_embedding_1 = get_code_embeddings(code_snippet_1)
code_embedding_1 = code_embedding_1.unsqueeze(0)
code_embedding_2 = get_code_embeddings(code_snippet_2)
code_embedding_2 = code_embedding_2.unsqueeze(0)


code_embedding_1 = code_embedding_1.to(device)
code_embedding_2 = code_embedding_2.to(device)

from model import MLP
from config import config

cfg = config()
input_size = cfg.input_size
hidden_size_1 = cfg.hidden_size_1
hidden_size_2 = cfg.hidden_size_2
num_classes = cfg.num_classes

# load the model
model = MLP(input_size, hidden_size_1, hidden_size_2, num_classes).to(device)
model.load_state_dict(torch.load(r"../model/model.pt"))

# print code_sniipets
print("Code Snippet 1:")
print("----------------------------------------")
print(code_snippet_1)
print("----------------------------------------")
print("Code Snippet 2:")
print("----------------------------------------")
print(code_snippet_2)
print("----------------------------------------")

model.eval()
with torch.no_grad():
    output = model(code_embedding_1, code_embedding_2)
    output = output.squeeze()
    pred = output.argmax(dim=0, keepdim=True)
    print("Clone" if pred == 1 else "Not Clone")
