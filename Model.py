import torch 
import lightning as L
import torch.nn as nn
import torch.nn.functional as F
import os
import cv2
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
from Recognition import Recognition2
import numpy as np

#Old model DO NOT USE
class Recognition(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(45*45, 512)
        self.l2 = nn.Linear(512,512)
        self.l3 = nn.Linear(512,512)
        self.l4 = nn.Linear(512,38)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = x.view(-1, 45*45)
        x = self.relu(self.l1(x))
        x = self.relu(self.l2(x))
        x = self.relu(self.l3(x))
        x = self.l4(x)
        return x

    
class custom_dataset(torch.utils.data.Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
    def __getitem__(self, idx):
        label = torch.tensor(self.labels[idx])
        image = self.images[idx]
        return image, label
    def __len__(self):
        return len(self.labels)

data_dir = "./data/extracted_images"
classNames = sorted([folder for folder in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, folder))])

images = []
labels = []

transform = transforms.Compose([
    transforms.ToPILImage(), 
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((45, 45)),
    #transforms.RandomAffine(degrees=5, translate=(0.1, 0.1), scale=(0.7, 1.1)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Create an encoder for all labels and save them to file
encoder = LabelEncoder()
encoder.fit(classNames)
np.save('./encoder_data/classes.npy', encoder.classes_)
for class_name in classNames:
    test_counter = 0
    folder = os.path.join(data_dir, class_name)
    for image in os.listdir(folder):
        if test_counter > 300:
            break
        processed_img = cv2.imread(os.path.join(folder, image), cv2.IMREAD_GRAYSCALE)
        if len(processed_img) != 0:
            normalized_tensor = transform(processed_img)
            normalized_label = encoder.transform([class_name])
            images.append(normalized_tensor)
            labels.append(normalized_label[0])
            print(class_name + image + str(normalized_label[0]))
            test_counter += 1

# Creating datasets and dataloaders for the images
dataset = custom_dataset(images, labels)
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

model = Recognition2()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

rounds = 6

for epoch in range(rounds):
    total_loss = 0.0
    for image, label in train_loader:
        optimizer.zero_grad()
        outputs = model(image)
        loss = loss_fn(outputs, label)
        loss.backward()
        optimizer.step()    
        total_loss += loss.item()
    print(f"Epoch {epoch}, Loss: {total_loss/len(train_loader)}")


correct = 0
total = 0
model.eval()

with torch.no_grad():
    for image, label in test_loader:
        outputs = model(image)
        _, prediction = torch.max(outputs, 1)
        total += label.size(0)
        correct += (prediction == label).sum().item() # Compares Tensors True = 1 and False = 0 then sums up all the trues
print(f"Accuracy: {100 * correct/total}")
torch.save(model.state_dict(), "./saved_model/model.pt")