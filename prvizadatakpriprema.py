import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score


df = pd.read_csv('./data/train.csv')
df = df.dropna()

df['zvanje'] = df['zvanje'].map({'AsstProf': 0, 'Prof': 1, 'AssocProf': 2})
df = pd.get_dummies(df, columns=['pol', 'oblast'], drop_first=True)

train,test = train_test_split(df,test_size=0.3,random_state=42)

X_train = train.drop('zvanje', axis = 1)
y_train = train['zvanje']

X_test = test.drop('zvanje', axis = 1)
y_test = test['zvanje']

st = StandardScaler()
X_train[X_train.columns] = st.fit_transform(X_train[X_train.columns])
X_test[X_test.columns] = st.transform(X_test[X_test.columns])
print(X_train)
print(X_test)
torch.manual_seed(42)

X_train = torch.tensor(X_train.to_numpy(), dtype=torch.float32)
X_test = torch.tensor(X_test.to_numpy(), dtype=torch.float32)
y_train = torch.tensor(y_train.to_numpy(), dtype=torch.long)
y_test = torch.tensor(y_test.to_numpy(), dtype=torch.long)

class MLPClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.relu = nn.ReLU()
        sizes = [input_size] + hidden_size + [num_classes]
        self.layers = nn.ModuleList([nn.Linear(sizes[i - 1], sizes[i]) for i in range(1,len(sizes))])

    def forward(self, x):
        out = x
        for i, layer in enumerate(self.layers):
            out = layer(out)
            if i < len(self.layers) - 1:
                out = self.relu(out)
        return out
    
input_size = X_train.shape[1]
hidden_sizes = [64, 32]
num_classes = 3 
model = MLPClassifier(input_size, hidden_sizes, num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.05)

num_epochs = 1000
for epoch in range(num_epochs):
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}')

with torch.no_grad():
    model.eval()
    outputs = model(X_test)
    _, predicted = torch.max(outputs, 1)
    f1 = f1_score(y_pred=predicted, y_true=y_test, average='micro')
    print(f'F1: {f1}')