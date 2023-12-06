import torch
from torch import nn

# define a MLP model to classify the two input code is clone or not
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size_1, hidden_size_2, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size_1)
        self.fc2 = nn.Linear(input_size, hidden_size_1)
        self.fc3 = nn.Linear(2 * hidden_size_1, hidden_size_2)
        self.dropout = nn.Dropout(p=0.3)
        self.relu = nn.ReLU()
        self.fc4 = nn.Linear(hidden_size_2, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x0, x1):
        x0 = self.fc1(x0)
        x1 = self.fc2(x1)
        x = torch.cat((x0, x1), 1)
        x = self.fc3(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.fc4(x)
        out = self.softmax(x)
        return out
