import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, accuracy_score, mean_absolute_error
import numpy as np

df = pd.read_csv('data/letva.csv')

df = df.dropna()

df['userName'] = LabelEncoder().fit_transform(df['userName'])

X = df[['userName','content','thumbsUpCount']]
Y = df['score']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)


bag_of_words = CountVectorizer(stop_words='english')
X_train_content = bag_of_words.fit_transform(X_train['content']).toarray()
X_test_content = bag_of_words.transform(X_test['content']).toarray()

scaler = StandardScaler()
X_train_numeric = scaler.fit_transform(X_train[['userName', 'thumbsUpCount']])
X_test_numeric = scaler.transform(X_test[['userName', 'thumbsUpCount']])

X_train_combined = np.hstack((X_train_numeric, X_train_content))
X_test_combined = np.hstack((X_test_numeric, X_test_content))

seed=42
torch.manual_seed(seed)

X_train = torch.tensor(X_train_combined, dtype=torch.float32)
X_test = torch.tensor(X_test_combined, dtype=torch.float32)
y_train = torch.tensor(y_train.to_numpy(), dtype=torch.long)
y_test = torch.tensor(y_test.to_numpy(), dtype=torch.long)

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.relu = nn.ReLU()
        self.layers = nn.ModuleList()
        sizes = [input_size] + hidden_size + [output_size]

        for i in range(1, len(sizes)):
            layer = nn.Linear(sizes[i - 1], sizes[i])
            self.layers.append(layer)

        # self.fc1 = nn.Linear(input_size, hidden_size)
        # self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # out = self.fc1(x)
        # out = self.relu(out)
        # out = self.fc2(out)
        out = x
        for i, layer in enumerate(self.layers):
            out = layer(out)
            if i < len(self.layers) - 1:
                out = self.relu(out)
        return out

input_size = X_train.shape[1]
hidden_sizes = [32]
output_size = 5

model = MLP(input_size, hidden_sizes, output_size)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

num_epochs = 200
for epoch in range(num_epochs):
    y_pred = model(X_train)
    loss = loss_fn(y_pred, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}')

with torch.no_grad():
    model.eval()
    y_pred = model(X_test)
    _, predicted = torch.max(y_pred, 1)
    accuracy = mean_absolute_error(y_pred=predicted, y_true=y_test)
    print(f'Accuracy: {accuracy}')
#
# print(f'F1: {f1}')