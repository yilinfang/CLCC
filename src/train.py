import torch
import torch.nn as nn
import random

from config import config

cfg = config()
random_seed = cfg.RANDOM_SEED
print("Random seed: ", random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
random.seed(random_seed)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device: ", device)


def train(model, train_loader, criterion, optimizer):
    model.train()
    train_loss = 0
    size = len(train_loader.dataset)
    for data0, data1, target in train_loader:
        data0, data1, target = data0.to(device), data1.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data0, data1)
        # output = output.item()
        output = output.squeeze()
        # print("output: ", output)
        # print("target: ", target)
        loss = criterion(output, target)
        # print("loss: ", loss)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()

    train_loss /= size
    return train_loss


from sklearn.metrics import precision_recall_fscore_support


def test(model, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    size = len(test_loader.dataset)
    y_true = []
    y_pred = []
    with torch.no_grad():
        for data0, data1, target in test_loader:
            data0, data1, target = data0.to(device), data1.to(device), target.to(device)
            # print(target)
            output = model(data0, data1)
            output = output.squeeze()
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            y_true += target.tolist()
            y_pred += pred.tolist()

    test_loss /= size
    accuracy = correct / size
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary"
    )
    return test_loss, accuracy, precision, recall, f1


from torch.utils.data import DataLoader, ConcatDataset
from dataset import dataset

dataset_py_train = dataset(r"../dataset/dataset_py_embedded_train.pkl")
dataset_py_test = dataset(r"../dataset/dataset_py_embedded_test.pkl")
print("dataset_py_train: ", len(dataset_py_train))
print("dataset_py_test: ", len(dataset_py_test))

dataset_java_train = dataset(r"../dataset/dataset_java_embedded_train.pkl")
dataset_java_test = dataset(r"../dataset/dataset_java_embedded_test.pkl")
print("dataset_java_train: ", len(dataset_java_train))
print("dataset_java_test: ", len(dataset_java_test))

dataset_cross_train = dataset(r"../dataset/dataset_cross_embedded_train.pkl")
dataset_cross_test = dataset(r"../dataset/dataset_cross_embedded_test.pkl")
print("dataset_cross_train: ", len(dataset_cross_train))
print("dataset_cross_test: ", len(dataset_cross_test))

dataset_all_train = ConcatDataset(
    [dataset_py_train, dataset_java_train, dataset_cross_train]
)
dataset_all_test = ConcatDataset(
    [dataset_py_test, dataset_java_test, dataset_cross_test]
)
print("dataset_train: ", len(dataset_all_train))
print("dataset_test: ", len(dataset_all_test))

from model import MLP

input_size = cfg.input_size
hidden_size_1 = cfg.hidden_size_1
hidden_size_2 = cfg.hidden_size_2
num_classes = cfg.num_classes
learning_rate = cfg.learning_rate
num_epochs = cfg.num_epochs
batch_size = cfg.batch_size


model = MLP(input_size, hidden_size_1, hidden_size_2, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# train_loader = DataLoader(dataset_py_train, batch_size=batch_size, shuffle=True)
# train_loader = DataLoader(dataset_java_train, batch_size=batch_size, shuffle=True)
# train_loader = DataLoader(dataset_cross_train, batch_size=batch_size, shuffle=True)
train_loader = DataLoader(dataset_all_train, batch_size=batch_size, shuffle=True)
# test_loader = DataLoader(dataset_py_test, batch_size=batch_size, shuffle=True)
# test_loader = DataLoader(dataset_java_test, batch_size=batch_size, shuffle=True)
# test_loader = DataLoader(dataset_cross_test, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset_all_test, batch_size=batch_size, shuffle=True)


best_accuracy = 0
best_precision = 0
best_recall = 0
best_f1 = 0
min_loss = 1000000

for epoch in range(num_epochs):
    print(f"Epoch: {epoch+1}")
    train_loss = train(model, train_loader, criterion, optimizer)
    print(f"Train: loss: {train_loss:.6f}")
    test_loss, accuracy, precision, recall, f1 = test(model, test_loader, criterion)
    if test_loss < min_loss:
        best_accuracy = accuracy
        best_precision = precision
        best_recall = recall
        best_f1 = f1
        min_loss = test_loss

        # save the model
        torch.save(model.state_dict(), r"../model/model.pt")

    print(
        f"Test: loss: {test_loss:.6f}, accuracy: {accuracy:.4f}, precision: {precision:.4f}, recall: {recall:.4f}, f1-score: {f1:.4f}"
    )
print(
    f"Final: accuracy: {best_accuracy:.4f}, precision: {best_precision:.4f}, recall: {best_recall:.4f}, f1-score: {best_f1:.4f}"
)
