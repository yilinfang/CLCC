from torch.utils.data import Dataset


class dataset(Dataset):
    def __init__(self, file_path):
        import pickle

        with open(file_path, "rb") as f:
            data = pickle.load(f)
        self.data0 = []
        self.data1 = []
        self.target = []
        for d in data:
            self.data0.append(d["sample_0"]["code_embedding"])
            self.data1.append(d["sample_1"]["code_embedding"])
            self.target.append(d["is_clone"])

    def __len__(self):
        return len(self.data0)

    def __getitem__(self, idx):
        return self.data0[idx], self.data1[idx], self.target[idx]
