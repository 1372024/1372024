import os
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from sklearn.model_selection import train_test_split
from zad1 import MyGelu
from gym_equipment_dataset import custom_dataset

# torch.manual_seed(42)

# transform = transforms.Compose([
#     transforms.Resize((256, 256)),
#     transforms.ToTensor(),
# ])

# Deo za učitavanje
# class CustomDataset(Dataset):
#     def __init__(self, root_dir, transform=None):
#         self.root_dir = root_dir
#         self.transform = transform
#         self.dataset = ImageFolder(root=self.root_dir, transform=self.transform)

#     def __len__(self):
#         return len(self.dataset)

#     def __getitem__(self, idx):
#         return self.dataset[idx]

# dataset_path = "gym_equipment"
# custom_dataset = CustomDataset(root_dir=dataset_path, transform=transform)





# train_set, test_set = train_test_split(custom_dataset, test_size=0.2, random_state=42)

# train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
# test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

# završi zadatak i ispiši meru koju model ostvaruje



torch.manual_seed(42)

transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
])



class CustomDataset(Dataset):
    def __init__(self,root_dir,transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.dataset = ImageFolder(root=self.root_dir,transform=self.transform)

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self,idx):
        return self.dataset[idx]
    
dataset_path = "gym_equipment"
custom_dataset = CustomDataset(root_dir=dataset_path,transform=transform)


train_set, test_set = train_test_split(custom_dataset,random_state=42,train_size=0.8)

train_loader = DataLoader(train_set,shuffle=True,batch_size=32)
test_loader = DataLoader(test_set,shuffle=False,batch_size=32)


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel,self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(3,32,kernel_size=3,padding=1),
            nn.BatchNorm2d(32),
            MyGelu(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(32,64,kernel_size=3),
            nn.BatchNorm2d(64),
            MyGelu(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(64,128,kernel_size=3),
            nn.BatchNorm2d(128),
            MyGelu(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )

        self.fc1 = nn.Linear(128*14*14,1000)
        self.fc2 = nn.Linear(1000,3)

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)

        out = out.view(out.size(0),-1)
        out = self.fc1(out)
        out = self.fc2(out)
        return out
    
model =CNNModel()
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
criterion = nn.CrossEntropyLoss()


num_epochs = 10


for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images,labels in train_loader:
        output = model(images)
        loss = criterion(output,labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss+=loss.item()

    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}')



model.eval()

total = 0
correct = 0

with torch.no_grad():
    for images,labels in test_loader:
        output = model(images)
        _,predicted = torch.max(output,1)
        total +=labels.size(0)
        correct += (predicted==labels).sum().item()

accuracy = 100*correct/total
print(f'Accuracy {accuracy*100}%')