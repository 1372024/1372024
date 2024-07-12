import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
import torch 
import torch.nn as nn
import torch.optim as optim

df = pd.read_csv('./data/south_park_train.csv')
df = df.dropna()

df['Character'] = LabelEncoder().fit_transform(df['Character'])


train, test = train_test_split(df,test_size=0.2,random_state=42)

x_train = train['Line']
x_test = test['Line']

y_train = train['Character']
y_test = test['Character']

bow = CountVectorizer(stop_words='english')


x_train = bow.fit_transform(x_train)
x_test = bow.transform(x_test)

torch.manual_seed(42)

class MLPClassifier(nn.Module):
    def __init__(self, input_sizes, hidden_layers, num_classes):
        super(MLPClassifier,self).__init__()
        self.relu = nn.ReLU()
        sizes = [input_sizes] + hidden_layers + [num_classes]
        self.layers = nn.ModuleList([nn.Linear(sizes[i-1],sizes[i]) for i in range(1, len(sizes))])

    def forward(self, x):
        out = x
        for i,layer in enumerate(self.layers):
            out = layer(out)
            if i < len(self.layers) - 1:
                out = self.relu(out)
        return out
    
input_size = x_train.shape[1]    
hidden_sizes = [512,256,128]
num_classes = len(set(y_train))
model = MLPClassifier(input_size,hidden_sizes,num_classes)

criterion = nn.CrossEntropyLoss()
optimazer = optim.Adam(model.parameters(), lr=0.001) 

x_train = torch.tensor(x_train.toarray(), dtype=torch.float32)
x_test = torch.tensor(x_test.toarray(), dtype=torch.float32)
y_train = torch.tensor(y_train.to_numpy(), dtype=torch.long)

num_epochs = 100
for epoch in range(num_epochs):
    outputs = model(x_train)
    loss = criterion(outputs, y_train)
    optimazer.zero_grad()
    loss.backward()
    optimazer.step()

    if(epoch + 1) % 10 == 0:
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')

with torch.no_grad():
    model.eval()
    outputs = model(x_test)
    _, predicted = torch.max(outputs, 1)
    accuracy = accuracy_score(y_pred=predicted, y_true=y_test)
    print(f'Accuracy: {accuracy}')